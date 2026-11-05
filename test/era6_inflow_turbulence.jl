# N-inflow-c (#264, epic #246): synthetic turbulence applied on the inflow
# faces of an open-boundary child, with zero added mass flux: U = (3, 1, 0),
# the added flux of every face is round-off relative to its mean-flow flux,
# every update; nothing on the outflow faces. (Stepped runs: validation.)
#
# References: Lund, Wu & Squires (1998), JCP 140, 233; Xie & Castro (2008),
# FTaC 81, 449; Kadasch et al. (2021), GMD 14, 5435 (PALM's STG: mean
# removal, lateral faces only); Panofsky & Dutton (1984), §7.4.

using Test
using Aeolus
using Aeolus.Fields: interior_view
using Aeolus.Simulations: BoussinesqSimulation

# A statistics source with a uniform tensor and fixed lengths (the test's
# own target; the shipped sources are N-inflow-b's).
struct _ITConst <: InflowStatistics
    R::NTuple{6,Float64}
    L_s::Float64
    L_z::Float64
    L_n::Float64
end
Aeolus.Transition.inflow_stresses(s::_ITConst, z::AbstractVector, t::Real) = [s.R for _ in z]
Aeolus.Transition.inflow_lengths(s::_ITConst, z::AbstractVector, t::Real) =
    (L_s = fill(s.L_s, 3, length(z)), L_z = fill(s.L_z, 3, length(z)),
     T = fill(s.L_n, 3, length(z)))
Aeolus.Transition.inflow_normal_length(s::_ITConst, z::AbstractVector, t::Real) =
    fill(s.L_n, 3, length(z))

# Panofsky & Dutton (1984) neutral surface layer, u* = 0.3 m/s, frame (n, s, z)
const _IT_US = 0.3
const _IT_R = ((2.39_IT_US)^2, (1.92_IT_US)^2, (1.25_IT_US)^2, 0.0, -_IT_US^2, 0.0)

function _it_sim(m, U, V; C_s = 0.18)
    dir = Ref((Float64(U), Float64(V)))
    data = FunctionBoundaryData(u = (x, y, z, t) -> dir[][1], v = (x, y, z, t) -> dir[][2],
                                w = 0.0, θ = 300.0)
    sim = BoussinesqSimulation(m; θ₀ = 300.0, C_s = C_s, bcs = open_boundary_conditions(m, data))
    sim.u.data .= U; sim.v.data .= V; sim.θ.data .= 300.0
    return sim, dir
end

@testset "N-inflow-c: no mass added on the inflow faces (#264)" begin
    m = RegularGrid(size = (24, 16, 12), extent = (240.0, 160.0, 120.0), halo = (3, 3, 3))
    sim, _ = _it_sim(m, 3.0, 1.0)
    it = InflowTurbulence(sim; statistics = _ITConst(_IT_R, 20.0, 20.0, 20.0))
    worst = Dict(s => 0.0 for s in (:west, :east, :south, :north))
    worst_level = 0.0; δmax = 0.0; ke = 0.0
    for n in 1:20
        # the boundary update every core calls once per step, before the projection
        Aeolus.BoundaryConditions.update_open_boundaries!(sim, 0.5n, 0.5)
        a = added_mass_flux(it)
        for (s, x) in pairs(a)
            worst[s] = max(worst[s], x.relative)
        end
        worst_level = max(worst_level, a.west.Q_level / (3.0 * m.Ly * m.dz),
                          a.south.Q_level / (1.0 * m.Lx * m.dz))
        # the data balance (Q_data = 0 for uniform U), untouched by the perturbations
        δmax = max(δmax, abs(open_boundary_ledger(sim).δ))
        ke += injected_ke_flux(it).total
    end
    @test all(<(1e-12), values(worst)) && worst_level < 1e-12
    @test worst[:east] == 0 && worst[:north] == 0           # outflow faces: nothing added
    @test δmax < 1e-13                                       # the correction sees only the data
    @test ke > 0                                             # perturbations were injected
end
