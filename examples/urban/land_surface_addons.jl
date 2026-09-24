# Batch B land-surface add-ons for the Tier 3 urban example.
#
# All functions in this file are stand-alone helpers layered *on top of*
# the existing `urban_physics.jl` code. Nothing here mutates the core
# Aeolus package; the routines are shaped to be dropped into the same
# step loop as `apply_surface_flux!`, `step_surfaces!`, etc.
#
# Contents:
#   * L2 — vegetation drag + tree shade
#       `apply_vegetation_drag!`, `apply_tree_shade!`
#   * L3 — multi-layer soil thermal diffusion
#       `SoilGrid`, `step_soil!`
#   * L5 — snow / ice / dew surface state
#       `SnowState`, `update_snow_state!`, `snow_modify_params!`,
#       `apply_dew_condensation!`
#
# The file re-uses the surface-class enum constants from
# `urban_geometry.jl` (which is transitively included by
# `urban_physics.jl`) — SURFACE_STREET, ROOF, PARK, WATER, WALL.

include(joinpath(@__DIR__, "urban_physics.jl"))

export apply_vegetation_drag!, apply_tree_shade!,
       SoilGrid, step_soil!,
       SnowState, update_snow_state!, snow_modify_params!,
       apply_dew_condensation!,
       dew_point

# ---------------------------------------------------------------------------
# L2. Vegetation drag + tree shade
# ---------------------------------------------------------------------------

"""
    apply_vegetation_drag!(sim, veg_mask, LAI_arr; cd=0.2, k_height=nothing)

Add a canopy drag body force `−cd · LAI · |u| · u` to the momentum
tendencies inside vegetation cells, for every interior z-layer whose
index `k` satisfies `k ≤ k_height`. When `k_height === nothing`, the
default is `sim.mesh.Nz ÷ 4` — i.e. the lower quarter of the domain,
which matches the typical canopy scale for a city LES.

Arguments:
  * `sim`          — the `BoussinesqSimulation`. The routine writes into
                     `sim.tu.data`, `sim.tv.data`, `sim.tw.data`, so it
                     must be called from inside a `body_force!` callback
                     passed to `step!`.
  * `veg_mask`     — `AbstractMatrix{Bool}` of size `(Nx, Ny)`. `true`
                     marks a canopy column.
  * `LAI_arr`      — `AbstractMatrix{<:Real}` of leaf-area index per
                     ground cell (unitless, ~0..6). Only the values
                     underneath `veg_mask` cells are consulted.

Keyword arguments:
  * `cd`           — canopy drag coefficient (typical 0.15..0.3).
  * `k_height`     — top interior-z-index (inclusive) at which drag is
                     still applied. `nothing` → `Nz ÷ 4`.

Notes:
  * The drag is evaluated on the *co-located* horizontal velocity at
    each interior cell — u and v are staggered on FaceX / FaceY but,
    consistent with `add_geostrophic_wind!` above, we read them at the
    same `(Hx+i, Hy+j, Hz+k)` index for simplicity. The vertical
    component uses `w` at the same index.
  * The routine is fully in-place and allocation-free.
"""
function apply_vegetation_drag!(sim, veg_mask::AbstractMatrix{Bool},
                                LAI_arr::AbstractMatrix{<:Real};
                                cd::Real = 0.2,
                                k_height::Union{Nothing, Integer} = nothing)
    Nx, Ny, Nz = sim.mesh.Nx, sim.mesh.Ny, sim.mesh.Nz
    Hx, Hy, Hz = sim.mesh.Hx, sim.mesh.Hy, sim.mesh.Hz
    k_top = k_height === nothing ? max(1, Nz ÷ 4) : Int(k_height)
    k_top = clamp(k_top, 1, Nz)
    cdF   = Float64(cd)
    ud    = sim.u.data
    vd    = sim.v.data
    wd    = sim.w.data
    tud   = sim.tu.data
    tvd   = sim.tv.data
    twd   = sim.tw.data
    @inbounds for j in 1:Ny, i in 1:Nx
        veg_mask[i, j] || continue
        lai = Float64(LAI_arr[i, j])
        lai > 0.0 || continue
        α = cdF * lai
        for k in 1:k_top
            iu = Hx + i; jv = Hy + j; kw = Hz + k
            uu = ud[iu, jv, kw]
            vv = vd[iu, jv, kw]
            ww = wd[iu, jv, kw]
            speed = sqrt(uu*uu + vv*vv + ww*ww)
            f = α * speed
            tud[iu, jv, kw] -= f * uu
            tvd[iu, jv, kw] -= f * vv
            twd[iu, jv, kw] -= f * ww
        end
    end
    return sim
end

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
# L3. Multi-layer soil column
# ---------------------------------------------------------------------------

"""
Layer heat capacities [J/m³/K] for the three soil layers. Values are
representative of moist mineral soil; the top layer is deliberately
lighter (thinner) so it responds fast to surface forcing while the deep
layer damps the diurnal cycle.
"""
const _SOIL_C = (1.4e6, 2.1e6, 2.5e6)

"""
Layer thicknesses [m] — geometrically expanding into the ground.
"""
const _SOIL_DZ = (0.05, 0.20, 0.60)

"""
Inter-layer conduction [W/m/K]. Third element is the surface-to-top
conduction (skin coupling); it is intentionally larger so `Ts` and
`T1` remain tightly coupled on time scales of order minutes.
"""
const _SOIL_K = (1.5, 1.2, 6.0)     # (k_12, k_23, k_skin)

"""
Climatological deep-soil temperature [K] used as the lower boundary
condition for the bottom layer.
"""
const _SOIL_T_DEEP = 288.0

"""
    SoilGrid(mesh; T0 = 288.0)

Allocate a 3-D `Array{Float64, 3}` of size `(Nx, Ny, 3)` storing the
three-layer soil temperature `(T1, T2, T3)` for every horizontal cell,
initialized uniformly to `T0` (K).
"""
function SoilGrid(mesh; T0::Real = _SOIL_T_DEEP)
    return fill(Float64(T0), mesh.Nx, mesh.Ny, 3)
end

"""
    step_soil!(soil, Ts, sfc, dt, C_surface; T_deep = 288.0)

Advance the 3-layer soil temperature field by `dt` (s), **and apply the
equal and opposite ground-heat flux to the surface**. Applied
per-column:

    C_s        · dTs/dt = ... - k_skin · (Ts   - T1)          ← the surface
    C1 · Δz1 · dT1/dt = k_skin · (Ts   - T1) + k_12 · (T2  - T1) / d12
    C2 · Δz2 · dT2/dt = k_12   · (T1  - T2) / d12 + k_23 · (T3  - T2) / d23
    C3 · Δz3 · dT3/dt = k_23   · (T2  - T3) / d23 + k_deep · (T_d - T3) / d3d

where distances between layer centres are `d = (Δz_k + Δz_{k+1}) / 2`
and `k_deep = k_23`, `d3d = Δz3` (a virtual layer at the same thickness).

`C_surface` is the surface heat capacity per class (J m⁻² K⁻¹) — pass
`params.C_s`, or a `SurfaceParams`.

!!! warning "This used to be a one-way sink, and it was an energy bug"
    Earlier versions took `Ts` as read-only: `F_skin` warmed soil layer 1
    but was never subtracted from the surface, so the column created heat
    from nothing AND the surface lost its ground-conduction sink
    entirely. Since `k_skin` = 6.0 W m⁻² K⁻¹ is comparable to the
    radiative `4εσT³` ≈ 6.4 and about five times the sensible coupling,
    that omission removed roughly half of the total nocturnal cooling.
    Surfaces in the Alexanderplatz run stayed near 315 K at 03:00 —
    about 20 K too warm — and the deficiency was initially misattributed
    to the single-slab surface scheme being structurally incapable of
    nocturnal cooling. It was not; it was this line.

Soil under water surfaces is intentionally skipped — water columns have
their own effective heat capacity in `SurfaceParams`.

Returns `soil`.
"""
function step_soil!(soil::AbstractArray{<:Real, 3},
                    Ts::AbstractMatrix{<:Real},
                    sfc::AbstractMatrix{UInt8},
                    dt::Real,
                    C_surface;
                    T_deep::Real = _SOIL_T_DEEP)
    Cs = C_surface isa SurfaceParams ? C_surface.C_s : C_surface
    Nx, Ny, NL = size(soil)
    @assert NL == 3 "step_soil! expects a 3-layer soil array"
    dz1, dz2, dz3 = _SOIL_DZ
    C1,  C2,  C3  = _SOIL_C
    k12, k23, k_s = _SOIL_K
    # centre-to-centre distances
    d12 = 0.5 * (dz1 + dz2)
    d23 = 0.5 * (dz2 + dz3)
    d3d = dz3
    k_deep = k23
    Td  = Float64(T_deep)
    Δt  = Float64(dt)
    @inbounds for j in 1:Ny, i in 1:Nx
        # skip open-water cells — no soil column beneath.
        sfc[i, j] == SURFACE_WATER && continue
        T1 = soil[i, j, 1]
        T2 = soil[i, j, 2]
        T3 = soil[i, j, 3]
        Tsij = Float64(Ts[i, j])

        # heat fluxes [W/m²]
        F_skin = k_s   * (Tsij - T1)          # surface -> layer 1
        F_12   = k12   * (T2   - T1) / d12
        F_21   = k12   * (T1   - T2) / d12
        F_23   = k23   * (T3   - T2) / d23
        F_32   = k23   * (T2   - T3) / d23
        F_deep = k_deep * (Td   - T3) / d3d

        # capacity per column area [J/m²/K]
        cap1 = C1 * dz1
        cap2 = C2 * dz2
        cap3 = C3 * dz3

        soil[i, j, 1] = T1 + Δt * (F_skin + F_12) / cap1
        soil[i, j, 2] = T2 + Δt * (F_21   + F_23) / cap2
        soil[i, j, 3] = T3 + Δt * (F_32   + F_deep) / cap3
        # The surface pays for what the soil gained. Without this the
        # column is not energy conserving and the surface has no ground
        # sink at all — see the warning in the docstring.
        Ts[i, j] = Tsij - Δt * F_skin / Cs[sfc[i, j]]
    end
    return soil
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
