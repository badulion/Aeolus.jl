# Batch B land-surface add-ons for the Tier 3 urban example.
#
# All functions in this file are stand-alone helpers layered *on top of*
# the existing `urban_physics.jl` code. Nothing here mutates the core
# Aeolus package; the routines are shaped to be dropped into the same
# step loop as `apply_surface_flux!`, `step_surfaces!`, etc.
#
# Contents:
#   * L2 — vegetation drag + tree shade
#       `vegetation_canopy_drag`, `apply_vegetation_drag!` (implicit drag,
#       delegating to `Aeolus.PlantCanopy`), `apply_tree_shade!`
#   * L3 — soil below the surface slab: the production
#       `Aeolus.SoilColumn` (thin adapter)
#       `UrbanSoil`, `step_soil!`, `soil_layer_temperature`,
#       `urban_soil_energy`
#   * L5 — snow / ice / dew surface state
#       `SnowState`, `update_snow_state!`, `snow_modify_params!`,
#       `apply_dew_condensation!`
#
# The file re-uses the surface-class enum constants from
# `urban_geometry.jl` (which is transitively included by
# `urban_physics.jl`) — SURFACE_STREET, ROOF, PARK, WATER, WALL.

include(joinpath(@__DIR__, "urban_physics.jl"))

export vegetation_canopy_drag, apply_vegetation_drag!, apply_tree_shade!,
       UrbanSoil, step_soil!, soil_layer_temperature, urban_soil_energy,
       SnowState, update_snow_state!, snow_modify_params!,
       apply_dew_condensation!,
       dew_point

# ---------------------------------------------------------------------------
# L2. Vegetation drag + tree shade
# ---------------------------------------------------------------------------

"""
    vegetation_canopy_drag(mesh, veg_mask, LAI_arr; cd=0.2, k_height=nothing)
        -> Aeolus.PlantCanopy.ResolvedCanopyDrag

Build the resolved canopy drag for the vegetated columns of the city.
Each column with `veg_mask[i, j]` and `LAI_arr[i, j] > 0` gets a uniform
leaf-area-density crown `UniformLAD(LAI, h)` with the canopy height
`h = k_height · dz` (`k_height === nothing` → `Nz ÷ 4` layers, the old
default). The physics — LAD profiles, the implicit drag factor and the
staggered-grid application — lives in `Aeolus.PlantCanopy`; this is
only the city-specific wiring.

Build it once, outside the time loop, and apply it with
[`apply_vegetation_drag!`](@ref) from the `post_predictor!` hook.
"""
function vegetation_canopy_drag(mesh, veg_mask::AbstractMatrix{Bool},
                                LAI_arr::AbstractMatrix{<:Real};
                                cd::Real = 0.2,
                                k_height::Union{Nothing, Integer} = nothing)
    Nz = mesh.Nz
    k_top = clamp(k_height === nothing ? max(1, Nz ÷ 4) : Int(k_height), 1, Nz)
    h = k_top * mesh.dz
    profiles = Union{Nothing, UniformLAD{Float64}}[
        (veg_mask[i, j] && LAI_arr[i, j] > 0) ? UniformLAD(LAI_arr[i, j], h) : nothing
        for i in axes(veg_mask, 1), j in axes(veg_mask, 2)]
    return ResolvedCanopyDrag(mesh, canopy_lad_field(mesh, profiles); C_d = cd)
end

"""
    apply_vegetation_drag!(sim, drag, dt)

Apply the canopy drag `-C_d a |U| U` *implicitly* — delegates to
`Aeolus.PlantCanopy.apply_canopy_drag!`, which multiplies the predicted
velocity by `1 / (1 + dt C_d a |U|)`. Call it from the `post_predictor!`
hook of `step!`; `drag` comes from [`vegetation_canopy_drag`](@ref).

History: this used to be an explicit tendency `-cd · LAI · |u| · u` added
in `body_force!`. It used LAI (m² m⁻²) where the leaf-area *density*
(m² m⁻³) belongs and read `u`, `v`, `w` co-located at the same index on
the staggered grid. The example shipped with it disabled after an
instability at dt = 2 s was reported; re-running it in #22 (cd = 0.01 and
0.03, 1 h at dt = 2 s) did not reproduce that, so its cause is undiagnosed.
The implicit form is stable at any dt (issue #22).
"""
apply_vegetation_drag!(sim, drag::ResolvedCanopyDrag, dt::Real) =
    apply_canopy_drag!(sim, drag, dt)

"""
    apply_tree_shade!(sw_arr, veg_mask, LAI_arr; k_leaf = 0.5)

Attenuate the shortwave flux reaching the ground beneath tree canopies
using Beer's law with an extinction coefficient `k_leaf` (typical
`0.4..0.6` for a randomly-distributed broadleaf canopy):

    SW_ground ← SW_ground · exp(-k_leaf · LAI)

Modifies `sw_arr` in place at every cell where `veg_mask[i, j]` is
`true`. Non-vegetated cells are untouched. Returns `sw_arr`.
"""
function apply_tree_shade!(sw_arr::AbstractMatrix{<:Real},
                           veg_mask::AbstractMatrix{Bool},
                           LAI_arr::AbstractMatrix{<:Real};
                           k_leaf::Real = 0.5)
    Nx, Ny = size(sw_arr)
    k = Float64(k_leaf)
    @inbounds for j in 1:Ny, i in 1:Nx
        if veg_mask[i, j]
            lai = Float64(LAI_arr[i, j])
            sw_arr[i, j] *= exp(-k * lai)
        end
    end
    return sw_arr
end

# ---------------------------------------------------------------------------
# L3. Soil below the surface slab — the production SoilColumn
# ---------------------------------------------------------------------------
#
# Every non-water cell carries one `Aeolus.SoilThermalColumn` (Phase 46:
# enthalpy heat equation with Johansen/Farouki conductivity, freeze–thaw,
# CLM layers). This file only couples it to the example's surface slab;
# all soil physics lives in `src/SoilColumn.jl`.
#
# It replaces a hand-written three-layer explicit scheme (layers 0.05 /
# 0.20 / 0.60 m, fixed k and C, relaxed to 288 K at 0.85 m). That scheme's
# `SoilGrid(mesh)` constructor also extended the exported
# `Aeolus.SoilGrid` type once Phase 46 landed — type piracy on the soil
# grid of the package.

const _SOIL_T_DEEP = 288.0

"""
    UrbanSoil(sfc; T0 = 288.0, T_deep = 288.0, grid = clm_exponential_soil_grid(10),
              hydraulics = carsel_parrish_texture(:loam),
              thermal = clm_soil_thermal_properties(40, 20), θ = 0.20,
              h_contact = 6.0)

The ground below the urban surface slab: one production
`SoilThermalColumn` (from `Aeolus.SoilColumn`) for every cell of the
surface-class map `sfc` that is not `SURFACE_WATER` (water keeps its own
deep heat capacity in `SurfaceParams`).

Defaults: the 10 exponentially thickening CLM2–CLM4.5 layers down to
3.43 m (top node 7 mm), a moist loam — Carsel & Parrish (1988) hydraulics
at ``θ = 0.20`` and the CLM thermal constants for 40 % sand / 20 % clay,
giving ``k = 1.64`` W m⁻¹ K⁻¹ and ``C = 2.10`` MJ m⁻³ K⁻¹ — uniformly at
`T0`, with the base held at the climatological `T_deep`
(`BottomTemperature`). The soil water is not stepped (`hydrology = false`):
the example's latent flux comes from its own moisture-availability table.

`h_contact` (W m⁻² K⁻¹) is the conductance between the slab temperature
`Ts` and the soil surface — the conduction path from the middle of the
paving/roof slab to its base. 6 W m⁻² K⁻¹ is the skin coupling of the
three-layer scheme this replaces (for asphalt, ``k ≈ 0.75`` W m⁻¹ K⁻¹ over
half of a 0.15 m slab gives 10).

Fields: `columns` (one per soil cell), `cells` (their `CartesianIndex`),
`G` (last time-mean ground heat flux per soil cell, W m⁻², positive into
the soil), `h_contact`.
"""
struct UrbanSoil{C<:SoilThermalColumn}
    columns::Vector{C}
    cells::Vector{CartesianIndex{2}}
    G::Vector{Float64}
    h_contact::Float64
end

function UrbanSoil(sfc::AbstractMatrix{UInt8};
                   T0::Real = _SOIL_T_DEEP, T_deep::Real = _SOIL_T_DEEP,
                   grid = clm_exponential_soil_grid(10),
                   hydraulics = carsel_parrish_texture(:loam),
                   thermal = clm_soil_thermal_properties(40, 20),
                   θ = 0.20, h_contact::Real = 6.0)
    h_contact > 0 || throw(ArgumentError("UrbanSoil: h_contact > 0"))
    cells = [I for I in CartesianIndices(sfc) if sfc[I] != SURFACE_WATER]
    make() = SoilThermalColumn(SoilWaterColumn(grid, hydraulics; θ = θ), thermal;
                               T_soil = T0,
                               bottom = Aeolus.SoilColumn.BottomTemperature(T_deep),
                               hydrology = false)
    columns = [make() for _ in cells]
    return UrbanSoil(columns, cells, zeros(length(cells)), Float64(h_contact))
end

"""
    step_soil!(soil::UrbanSoil, Ts, sfc, dt, C_surface) -> soil

Exchange heat between the surface slab `Ts` (K, per cell, areal heat
capacity `C_surface[class]` J m⁻² K⁻¹ — pass `params.C_s` or the
`SurfaceParams`) and the soil columns over `dt` seconds, and advance the
soil. The slab has already taken its atmospheric fluxes (`step_surfaces!`)
to `Ts*`; over `dt` it now loses the ground heat flux

    G = h (Ts' - T_g),    C_s (Ts' - Ts*) = -G dt

to the soil surface ``T_g`` through the contact conductance `h`.
Eliminating the new slab temperature ``Ts'`` gives a flux linear in
``T_g``, ``G = (Ts* - T_g) / (1/h + dt/C_s)``, which is passed to
`step_soil_heat!` as a `LinearizedSkinBalance(0, -1/(1/h + dt/C_s), Ts*)`
and solved implicitly with the soil heat equation. The slab then pays
exactly what the soil received, ``Ts' = Ts* - G dt / C_s``, so slab + soil
conserve energy to round-off (the soil base exchanges with `T_deep`).

Being implicit, the exchange is stable for any `dt`; the driver may call
it every few dynamics steps with the accumulated interval (operator
splitting: the slab then misses its ground flux for one soil interval,
``ΔT ≈ G dt / C_s``).

`sfc` is accepted for the old call signature and must be the map the soil
was built from.
"""
function step_soil!(soil::UrbanSoil, Ts::AbstractMatrix{<:Real},
                    sfc::AbstractMatrix{UInt8}, dt::Real, C_surface)
    Cs = C_surface isa SurfaceParams ? C_surface.C_s : C_surface
    Δt = Float64(dt)
    hinv = 1 / soil.h_contact
    @inbounds for (n, I) in enumerate(soil.cells)
        C = Float64(Cs[sfc[I]])
        Tstar = Float64(Ts[I])
        top = Aeolus.SoilColumn.LinearizedSkinBalance(0.0, -1 / (hinv + Δt / C), Tstar)
        st = step_soil_heat!(soil.columns[n], Δt, top)
        soil.G[n] = st.G
        Ts[I] = Tstar - st.G * Δt / C
    end
    return soil
end

"""
    soil_layer_temperature(soil::UrbanSoil, dims, layer = 1) -> Matrix{Float64}

Temperature (K) of soil layer `layer` on the `dims = (Nx, Ny)` grid;
`NaN` where there is no soil (water).
"""
function soil_layer_temperature(soil::UrbanSoil, dims, layer::Integer = 1)
    out = fill(NaN, dims)
    for (n, I) in enumerate(soil.cells)
        out[I] = soil.columns[n].water.T_soil[layer]
    end
    return out
end

"""
    urban_soil_energy(soil::UrbanSoil, Ts, C_surface, sfc) -> J m⁻² (sum over soil cells)

Slab plus soil enthalpy summed over the soil cells, ``Σ (C_s T_s +
∫H dz)`` — the quantity `step_soil!` conserves apart from the exchange
with the deep boundary.
"""
function urban_soil_energy(soil::UrbanSoil, Ts::AbstractMatrix{<:Real}, C_surface, sfc)
    Cs = C_surface isa SurfaceParams ? C_surface.C_s : C_surface
    return sum(Cs[sfc[I]] * Ts[I] + soil_enthalpy(soil.columns[n]) for (n, I) in enumerate(soil.cells))
end

# ---------------------------------------------------------------------------
# L5. Snow / ice / dew surface state
# ---------------------------------------------------------------------------

"""
    SnowState(Nx, Ny) -> Matrix{Bool}
    SnowState(mesh)  -> Matrix{Bool}

Allocate the boolean snow / ice cover flag matrix, size `(Nx, Ny)`,
initialized to `false` (no snow). When the flag is `true` for a cell,
callers should treat that cell as having snow-like radiative properties
(high albedo, high emissivity) and zero latent flux from soil moisture.
"""
SnowState(Nx::Integer, Ny::Integer) = falses(Int(Nx), Int(Ny))
SnowState(mesh)                     = falses(mesh.Nx, mesh.Ny)

"""
    update_snow_state!(snow, Ts, precip = 0)

Flip the per-cell snow / ice flag in place using a simple two-threshold
rule (avoids the ping-pong you get with a single threshold):

  * cell becomes snowy when `Ts < 273 K` **and** `precip > 0`
  * cell becomes bare  when `Ts > 275 K` (unconditionally)

`precip` may be a scalar (uniform precipitation) or a matrix the same
shape as `Ts`. Returns `snow`.
"""
function update_snow_state!(snow::AbstractMatrix{Bool},
                            Ts::AbstractMatrix{<:Real},
                            precip = 0)
    Nx, Ny = size(snow)
    is_field = precip isa AbstractMatrix
    p_scalar = is_field ? 0.0 : Float64(precip)
    @inbounds for j in 1:Ny, i in 1:Nx
        Tsij = Float64(Ts[i, j])
        p    = is_field ? Float64(precip[i, j]) : p_scalar
        if snow[i, j]
            # currently snowy: melt only above the upper threshold
            Tsij > 275.0 && (snow[i, j] = false)
        else
            # currently bare: freeze only if cold AND wet
            (Tsij < 273.0 && p > 0.0) && (snow[i, j] = true)
        end
    end
    return snow
end

# Snow-modified radiative constants for the surface energy balance.
const _SNOW_ALBEDO     = 0.75      # fresh-to-slightly-aged snow
const _SNOW_EMISSIVITY = 0.99      # snow radiates almost as a black body

"""
    snow_modify_params!(alb_view, emis_view, alb_orig, emis_orig, snow;
                        snow_albedo = 0.75, snow_emissivity = 0.99)

Populate `alb_view` and `emis_view` (both `AbstractMatrix{Float64}` of
size `(Nx, Ny)`) with the *effective* albedo and emissivity for the
current time step: cells with `snow[i, j] == true` are overridden with
the snow values; the rest are copied straight from `alb_orig` /
`emis_orig`.

This is the recommended way to plug snow into `step_surfaces!` without
mutating the class-level `SurfaceParams` struct. Typical call site:

    for j in 1:Ny, i in 1:Nx
        alb_orig[i, j]  = params.albedo[sfc[i, j]]
        emis_orig[i, j] = params.emissivity[sfc[i, j]]
    end
    snow_modify_params!(alb_eff, emis_eff, alb_orig, emis_orig, snow)

Returns `(alb_view, emis_view)`.
"""
function snow_modify_params!(alb_view::AbstractMatrix{Float64},
                             emis_view::AbstractMatrix{Float64},
                             alb_orig::AbstractMatrix{<:Real},
                             emis_orig::AbstractMatrix{<:Real},
                             snow::AbstractMatrix{Bool};
                             snow_albedo::Real = _SNOW_ALBEDO,
                             snow_emissivity::Real = _SNOW_EMISSIVITY)
    Nx, Ny = size(snow)
    a_snow = Float64(snow_albedo)
    e_snow = Float64(snow_emissivity)
    @inbounds for j in 1:Ny, i in 1:Nx
        if snow[i, j]
            alb_view[i, j]  = a_snow
            emis_view[i, j] = e_snow
        else
            alb_view[i, j]  = Float64(alb_orig[i, j])
            emis_view[i, j] = Float64(emis_orig[i, j])
        end
    end
    return (alb_view, emis_view)
end

"""
    dew_point(qv, p = 1.01325e5) -> K

Dew-point temperature (K) for specific humidity `qv` [kg/kg] at pressure
`p` [Pa]. Delegates to `Aeolus.MoistThermodynamics.dewpoint_temperature`,
the same saturation curve `q_saturation` uses.
"""
dew_point(qv::Real, p::Real = 1.01325e5) =
    Aeolus.MoistThermodynamics.dewpoint_temperature(p, max(Float64(qv), 1e-12))

"""
    apply_dew_condensation!(Ts, qv_air, LE; ε_warm = 5e-3, p = 1.01325e5)

Legacy L5 proxy, superseded by `apply_fog_dew!` (moisture_addons.jl,
backed by `Aeolus.RadiationFog`), which removes the vapour that forms the
dew and derives the surface warming from `L_v · m` instead of a relaxation
constant. Kept for reference; the urban run no longer calls it.

When a surface cell is at or below the dew point of the air above it,
water condenses onto it rather than evaporating. Enforce this by

  * clamping the corresponding latent flux `LE[i, j]` to `≤ 0`
    (evaporation-positive convention: no water leaves the ground when
    it's below dew point);
  * warming the surface slightly by `ε_warm · (T_dew − Ts)` to represent
    the released latent heat of condensation.

`qv_air` may be a scalar (uniform) or a matrix the shape of `Ts`.
`LE` is modified in place; `Ts` receives a small in-place warming.
Returns `LE`.
"""
function apply_dew_condensation!(Ts::AbstractMatrix{<:Real},
                                 qv_air,
                                 LE::AbstractMatrix{<:Real};
                                 ε_warm::Real = 5e-3,
                                 p::Real = 1.01325e5)
    Nx, Ny = size(Ts)
    qv_is_field = qv_air isa AbstractMatrix
    qv_scalar   = qv_is_field ? 0.0 : Float64(qv_air)
    p_f = Float64(p)
    ε   = Float64(ε_warm)
    @inbounds for j in 1:Ny, i in 1:Nx
        q   = qv_is_field ? Float64(qv_air[i, j]) : qv_scalar
        Tdp = dew_point(q, p_f)
        Tij = Float64(Ts[i, j])
        if Tij < Tdp
            # forbid evaporation from a below-dew-point surface
            if LE[i, j] > 0
                LE[i, j] = zero(eltype(LE))
            end
            # small latent-heat-of-condensation warming
            Ts[i, j] = Tij + ε * (Tdp - Tij)
        end
    end
    return LE
end
