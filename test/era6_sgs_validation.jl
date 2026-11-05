# Era VI, Phase 56 (P56-e, #192): the API the filtered-DNS validation of
# the TKE (P56-b) and Reynolds-stress (P56-d) SGS closures relies on —
# the step_rk3! hooks (body_force!, post_predictor!) and the closures
# seeded from a subfilter stress. The validation against filtered DNS
# itself is in validation/era6_sgs_validation.jl.

using Test
using Aeolus
using Aeolus.Fields: interior_view

@testset "P56-e: API — step_rk3! hooks, closures seeded from a stress" begin
    g = RegularGrid(; size = (8, 8, 8), extent = (1.0, 1.0, 1.0), halo = (3, 3, 3))
    # body_force! runs at each of the three substages: a constant θ
    # tendency q advances θ by exactly q·dt over one SSP-RK3 step
    sim = BoussinesqSimulation(g)
    q = 0.25
    step_rk3!(sim, 0.1; body_force! = (s, Δt) -> (interior_view(s.tθ) .+= q; nothing))
    @test all(x -> isapprox(x, q * 0.1; rtol = 1e-14), interior_view(sim.θ))
    @test sim.t ≈ 0.1
    # post_predictor! runs after each predictor, before the projection: a
    # hook that zeroes u makes every stage value 0, so the Shu–Osher
    # combination leaves uⁿ⁺¹ = ⅓uⁿ + ⅔·0 = ⅓ from uⁿ = 1
    sim2 = BoussinesqSimulation(g)
    interior_view(sim2.u) .= 1.0
    step_rk3!(sim2, 0.1; post_predictor! = (s, Δt) -> (interior_view(s.u) .= 0; nothing))
    @test all(x -> isapprox(x, 1 / 3; atol = 1e-14), interior_view(sim2.u))
    # seeding from a subfilter stress: e = ½ tr τ; a non-PSD cell is
    # projected (its negative eigenvalue clipped), a PSD cell is copied
    τ = SymmetricTensorField(g)
    τ.xx .= 0.3; τ.yy .= 0.2; τ.zz .= 0.1; τ.xy .= 0.05
    τ.xy[1, 1, 1] = 1.0                       # τ₁₁τ₂₂ < τ₁₂²: not realizable
    c = initialize_tke!(TKEClosure(g), τ)
    @test interior_view(c.e)[2, 2, 2] ≈ 0.3
    r, n = initialize_reynolds_stress!(ReynoldsStressClosure(g), τ)
    @test n == 1
    @test interior_view(r.τ[4])[2, 2, 2] == 0.05
    t1 = ntuple(m -> interior_view(r.τ[m])[1, 1, 1], 6)
    @test is_realizable(t1)
    @test interior_view(r.τ[1])[1, 1, 1] + interior_view(r.τ[2])[1, 1, 1] > 0.5   # the trace only grows
    bad = SymmetricTensorField{Float64}(4, 4, 4)
    @test_throws DimensionMismatch initialize_tke!(TKEClosure(g), bad)
    @test_throws DimensionMismatch initialize_reynolds_stress!(ReynoldsStressClosure(g), bad)
end
