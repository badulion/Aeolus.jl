# Vehicle heat: a constant urban-area mean released over the road cells.
# The physics is `BuildingEnergy.add_traffic_heat!` (ONE-e, #454); this
# file picks the road cells and the heat capacity of the first layer.

"""
    TrafficHeat

Additive coupling: [`add_traffic_heat!`](@ref) every step in
`:anthropogenic` (after the buildings), option `Q_urban` (W m⁻²).
"""
struct TrafficHeat <: AbstractCoupling
    Q_urban::Float64
    road::Vector{CartesianIndex{2}}
end

coupling_name(::Type{TrafficHeat}) = :traffic_heat
default_options(::Type{TrafficHeat}) = (Q_urban = 8.0,)
stages(::Type{TrafficHeat}) = ((:anthropogenic, 20, EVERY_STEP),)
build(::Type{TrafficHeat}, case, o) = TrafficHeat(o.Q_urban, CartesianIndex{2}[])

function init!(c::TrafficHeat, case)
    sfc = case.geom.sfc
    append!(c.road, [I for I in CartesianIndices(sfc)
                     if sfc[I] == SURFACE_STREET || sfc[I] == SURFACE_WALL])
    return nothing
end

run_stage!(c::TrafficHeat, case, ::Val{:anthropogenic}, dt) =
    add_traffic_heat!(case.sim, c.road, length(case.geom.sfc), dt; Q_urban = c.Q_urban,
                      ρcp = 1 / ref_θ_per_heat(case, 1))

register!(TrafficHeat)
