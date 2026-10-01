# Vehicle heat: a constant urban-area mean released over the road cells.
# Inventory data of the example, not a building quantity.

"""
    add_traffic_heat!(sim, road, n_cells, dt; Q_urban = 8.0, ρcp = 1005.0)

Vehicle heat: a constant `Q_urban` W m⁻² of urban area (`n_cells`
columns), released into the first air layer over the `road` cells
(streets and the building-adjacent cells). The default 8.0 W m⁻² is the
traffic heat Bueno et al. (2012, Geosci. Model Dev. 5, 433, Table 3)
prescribe for the Toulouse centre (their Table 4 uses the same value for
Athens); the tables give no source for it, and it is not a Berlin
inventory.
"""
function add_traffic_heat!(sim, road::AbstractVector{CartesianIndex{2}}, n_cells::Integer, dt::Real;
                           Q_urban::Real = 8.0, ρcp::Real = 1005.0)
    isempty(road) && return sim
    q = Q_urban * n_cells / length(road)
    Hx, Hy, Hz = sim.mesh.Hx, sim.mesh.Hy, sim.mesh.Hz
    for I in road
        i, j = Tuple(I)
        sim.θ.data[Hx+i, Hy+j, Hz+1] += q / ρcp * dt / sim.mesh.dz
    end
    return sim
end

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
                      ρcp = ref_ρcp(case, 1))

register!(TrafficHeat)
