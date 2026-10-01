# The ground: one production Aeolus.SoilColumn `SoilThermalColumn` per
# street, building-adjacent and park column, used directly as that
# column's surface (an `AbstractSurface`: skin energy balance, heat
# conduction and freeze–thaw, Richards soil water, and a moisture
# availability β from the water of the top layer).
#
# It replaces the #126 pair "slab temperature + UrbanSoil below it through
# a contact conductance" and the example's per-class β table: the skin is
# now the soil's own massless skin, solved implicitly with the soil heat
# equation, and β follows the soil water (Lee & Pielke 1992, in
# SoilColumn). All soil physics lives in src/SoilColumn.jl; this file only
# registers the factory the surface exchange builds the columns with.
#
# KNOWN LIMITATIONS (stated, not hidden):
#   * the streets are bare loam: SoilColumn has no sealed (impervious)
#     surface, so a street evaporates from its top soil layer until that
#     layer dries (a sealed-surface option belongs in src);
#   * the park is bare soil too until #140 puts LandSurface's vegetated
#     column there;
#   * no rain reaches the soil: the microphysics books surface
#     precipitation as a domain total, not per column.

"""
    GroundSoil

Slot `:ground`, provider `:soil_column` (default). Registers a surface
factory for the classes in `classes` (default street, park and the
building-adjacent `SURFACE_WALL` cells): a `SoilThermalColumn` over a
`SoilWaterColumn` with

* the CLM 10-layer exponential grid (top node 7 mm, base 3.43 m),
  Carsel & Parrish (1988) hydraulics of `texture` and the CLM thermal
  constants for `sand`/`clay` percent;
* initial volumetric water `θ0[class]` (default: 0.10 under streets and
  the building-adjacent cells, a dry bedding with β ≈ 0.44 at the start;
  0.25 in the park, above field capacity, β = 1), and temperature
  `cfg.T_soil0` with the base held there (`BottomTemperature`);
* the skin starting at the case's surface temperature `bus.Ts`;
* albedo and emissivity of the class from `bus.params` (the radiosity
  materials), roughness `z0m` from the surface-exchange options;
* `hydrology = true`: evaporation is drawn from, and dew added to, the
  soil water, so β changes as the top layer dries or wets.

Outputs `T_soil1` (top layer, 7 mm), `G_soil` (ground heat flux of the
last surface step) and `theta_soil1` (top-layer liquid water); NaN where
the column is not soil.
"""
struct GroundSoil{H,P} <: AbstractCoupling
    classes::Vector{UInt8}
    θ0::Dict{UInt8,Float64}
    hydraulics::H
    thermal::P
    nlayers::Int
    cols::Vector{CartesianIndex{2}}
end

coupling_name(::Type{<:GroundSoil}) = :ground_soil
slot(::Type{<:GroundSoil}) = :ground
provider(::Type{<:GroundSoil}) = :soil_column
modules_used(::Type{<:GroundSoil}) = (:SoilColumn,)
default_options(::Type{<:GroundSoil}) =
    (classes = (SURFACE_STREET, SURFACE_WALL, SURFACE_PARK),
     θ0 = Dict(SURFACE_STREET => 0.10, SURFACE_WALL => 0.10, SURFACE_PARK => 0.25),
     texture = :loam, sand = 40, clay = 20, nlayers = 10)

function build(::Type{<:GroundSoil}, case, o)
    c = GroundSoil(UInt8[o.classes...], Dict{UInt8,Float64}(o.θ0),
                   carsel_parrish_texture(o.texture), clm_soil_thermal_properties(o.sand, o.clay),
                   o.nlayers, CartesianIndex{2}[])
    for cls in c.classes
        register_surface_factory!(case, cls, :ground_soil, (cs, I) -> soil_surface(c, cs, I))
    end
    return c
end

"""
    soil_surface(c::GroundSoil, case, I) -> SoilThermalColumn

The soil column of column `I` (see [`GroundSoil`](@ref)).
"""
function soil_surface(c::GroundSoil, case, I)
    cls = case.geom.sfc[I]
    p, T0 = case.bus.params, case.cfg.T_soil0
    se = find_coupling(case, :surface_exchange)
    z0m = se === nothing ? 0.05 : get(se.z0m, cls, 0.05)
    ratio = se === nothing ? 0.1 : se.z0h_ratio
    water = SoilWaterColumn(clm_exponential_soil_grid(c.nlayers), c.hydraulics;
                            θ = get(c.θ0, cls, 0.2))
    col = SoilThermalColumn(water, c.thermal; T_soil = T0, T_skin = case.bus.Ts[I],
                            bottom = Aeolus.SoilColumn.BottomTemperature(T0),
                            albedo = p.albedo[cls], emissivity = p.emissivity[cls],
                            z0m = z0m, z0h = ratio * z0m, hydrology = true)
    push!(c.cols, I)
    return col
end

# The surface exchange's per-column closure check: the soil's enthalpy
# less what left through the base and what the moved water carried, so
# its change over a step is the ground heat flux G the step returned.
surface_store_energy(s::SoilThermalColumn) =
    soil_enthalpy(s) + s.budget.bottom - s.budget.water

"(Nx, Ny) map of `f(col, I)` over the soil columns, NaN elsewhere."
function soil_map(f, c::GroundSoil, case)
    out = fill(NaN, case.mesh.Nx, case.mesh.Ny)
    for I in c.cols
        s = case.bus.surface[I]
        s isa SoilThermalColumn && (out[I] = f(s, I))
    end
    return out
end

function output_fields(c::GroundSoil, case)
    return OutputField[
        OutputField("T_soil1", "K", "soil temperature, top layer (node 7 mm; NaN where not soil)",
                    :xy, cs -> soil_map((s, I) -> s.water.T_soil[1], c, cs)),
        OutputField("G_soil", "W m-2", "ground heat flux into the soil, last surface step (NaN where not soil)",
                    :xy, cs -> soil_map((s, I) -> cs.bus.G[I], c, cs)),
        OutputField("theta_soil1", "m3 m-3", "liquid soil water, top layer (NaN where not soil)",
                    :xy, cs -> soil_map((s, I) -> s.water.θ_liq[1], c, cs))]
end

function report(c::GroundSoil, case)
    isempty(c.cols) && return nothing
    β = [moisture_availability(case.bus.surface[I]) for I in c.cols]
    θ1 = [case.bus.surface[I].water.θ_liq[1] for I in c.cols]
    return @sprintf("           soil: β %.3f–%.3f (mean %.3f), top-layer water %.4f–%.4f",
                    minimum(β), maximum(β), mean(β), minimum(θ1), maximum(θ1))
end

function smoke_check(c::GroundSoil, case)
    cols = [case.bus.surface[I] for I in c.cols]
    isempty(cols) && return ["soil columns in the window" => false]
    T0, Ts = case.cfg.T_soil0, case.bus.Ts
    # heat diffusion with a Dirichlet base at T_soil0 and the skin on top
    # obeys a maximum principle (1 K margin for the skin's move in a step)
    sk = [s.T_skin for s in cols]
    lo, hi = min(T0, minimum(sk)) - 1, max(T0, maximum(sk)) + 1
    Tall = reduce(vcat, (s.water.T_soil for s in cols))
    h = c.hydraulics
    θall = reduce(vcat, (s.water.θ_liq for s in cols))
    β = moisture_availability.(cols)
    return ["soil columns in the window, one per street/wall/park column with air" =>
                length(c.cols) == count(I -> case.geom.sfc[I] in c.classes &&
                                             case.bus.surface[I] !== nothing,
                                        CartesianIndices(case.geom.sfc)),
            "soil temperatures within [min(T_soil0, T_skin) − 1, max(T_soil0, T_skin) + 1] K" =>
                all(T -> lo <= T <= hi, Tall),
            "soil liquid water within [θ_r, θ_s]" =>
                all(θ -> h.θ_r <= θ <= h.θ_s, θall),
            "β ∈ [0, 1], and < 1 where the top layer is below field capacity" =>
                all(i -> 0 <= β[i] <= 1 &&
                         (cols[i].water.θ_liq[1] >= soil_field_capacity(h) || β[i] < 1),
                    eachindex(cols)),
            "the skin is the bus surface temperature" =>
                all(I -> case.bus.surface[I].T_skin == Ts[I], c.cols)]
end

register!(GroundSoil)
