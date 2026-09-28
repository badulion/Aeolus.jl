# The ground below the slab: one production Aeolus.SoilColumn
# `SoilThermalColumn` under every non-water column, exchanging heat with
# the slab temperature through a contact conductance (implicit and
# energy-conserving), sub-cycled at the `:soil` cadence.
#
# It replaced a hand-written three-layer explicit scheme (#18). All soil
# physics lives in src/SoilColumn.jl; this file is only the coupling.
# KNOWN: roofs sit on 3.4 m of soil too (the surface-exchange issue moves
# them to UrbanFacet skins and replaces this file with a SoilThermalColumn
# ground surface).

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


"""
    UrbanSoilCoupling

Slot `:ground`, provider `:urban_soil`. Builds [`UrbanSoil`](@ref) at
`init!` (`cfg.T_soil0`, base at the same temperature) and runs
[`step_soil!`](@ref) at the `:soil` cadence, after the slab update, over
the time elapsed since the previous exchange (the cadence period `P`
when `dt` divides it; exact with a variable `dt`, #133). Outputs
`T_soil1` and `G_soil`.
"""
mutable struct UrbanSoilCoupling <: AbstractCoupling
    soil::Union{Nothing,UrbanSoil}
    last_step::Int       # step on which the soil last exchanged heat (for the ledger)
    last_t::Float64      # simulation time at the end of that step (s)
    interval::Float64    # the interval that exchange covered (s)
end

coupling_name(::Type{UrbanSoilCoupling}) = :soil_urban
slot(::Type{UrbanSoilCoupling}) = :ground
provider(::Type{UrbanSoilCoupling}) = :urban_soil
modules_used(::Type{UrbanSoilCoupling}) = (:SoilColumn,)
stages(::Type{UrbanSoilCoupling}) = ((:surface_update, 20, Cadence(:soil)),)
build(::Type{UrbanSoilCoupling}, case, o) = UrbanSoilCoupling(nothing, 0, 0.0, 0.0)

function init!(c::UrbanSoilCoupling, case)
    T0 = case.cfg.T_soil0
    c.soil = UrbanSoil(case.geom.sfc; T0 = T0, T_deep = T0)
    c.last_t = case.sim.t
    return nothing
end

function run_stage!(c::UrbanSoilCoupling, case, ::Val{:surface_update}, dt)
    # :surface_update runs after step!, so sim.t is already the end of
    # this step. With a fixed dt dividing P the interval is exactly P
    # (the times are exact multiples of dt).
    t = case.sim.t
    c.interval = t - c.last_t
    step_soil!(c.soil, case.bus.Ts, case.geom.sfc, c.interval, case.bus.params)
    c.last_step = case.clock.n
    c.last_t = t
    return nothing
end

# Ledger: on the step the soil exchanges, the slab pays G over the whole
# soil interval at once (Ts' = Ts* − G P / C_s), so G·P is booked on
# that step. The soil's own enthalpy change is below the surface layer
# and not part of the surface balance.
function budget!(c::UrbanSoilCoupling, case, l)
    c.last_step == case.clock.n || return nothing
    P = c.interval
    G = zeros(size(case.bus.Ts))
    for (n, I) in enumerate(c.soil.cells)
        G[I] = c.soil.G[n]
    end
    add_surface!(l, :G, G, case.geom.sfc, P)
    return nothing
end

function smoke_check(c::UrbanSoilCoupling, case)
    # Heat diffusion with a Dirichlet base at T_soil0 and a top flux
    # towards Ts obeys a maximum principle: no soil temperature can leave
    # the range spanned by T_soil0 and the surface temperatures (a margin
    # of 1 K covers Ts changing within the run).
    T0, Ts = case.cfg.T_soil0, case.bus.Ts
    lo, hi = min(T0, minimum(Ts)) - 1, max(T0, maximum(Ts)) + 1
    Tall = reduce(vcat, (col.water.T_soil for col in c.soil.columns))
    return ["soil exchanged heat during the run" => c.last_step > 0,
            "soil temperatures within [min(T_soil0, Ts) − 1, max(T_soil0, Ts) + 1] K" =>
                all(T -> lo <= T <= hi, Tall),
            "no soil under water" => all(I -> case.geom.sfc[I] != SURFACE_WATER, c.soil.cells)]
end

"Ground heat flux of the last soil exchange on the (Nx, Ny) grid, NaN under water."
function G_map(soil::UrbanSoil, dims)
    out = fill(NaN, dims)
    for (n, I) in enumerate(soil.cells)
        out[I] = soil.G[n]
    end
    return out
end

function output_fields(c::UrbanSoilCoupling, case)
    dims = (case.mesh.Nx, case.mesh.Ny)
    return OutputField[
        OutputField("T_soil1", "K", "soil temperature, top layer (node 7 mm; NaN under water)",
                    :xy, cs -> soil_layer_temperature(c.soil, dims, 1)),
        OutputField("G_soil", "W m-2",
                    "ground heat flux from the surface slab into the soil (NaN under water)",
                    :xy, cs -> G_map(c.soil, dims))]
end

register!(UrbanSoilCoupling)
