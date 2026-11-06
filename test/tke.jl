# Era VI, Phase 56 (P56-b, #182): the Deardorff (1980) 1.5-order prognostic
# SGS TKE closure (src/PrognosticSGS.jl) with the Moeng (1984) coefficients,
#
# The 500-step positivity run and the reduced CBL are validation runs, in
# validation/tke.jl.

using Test
using Aeolus
using Aeolus.Fields: interior_view, set!
using Aeolus.Operators: advance_sgs!, sgs_state, restore_sgs_state!
using Random: Xoshiro, randn

@testset "Era VI — prognostic SGS TKE (P56-b, #182)" begin

@testset "1. free decay at l = Δ (c_ε = 0.70)" begin
    m = RegularGrid(; size = (4, 4, 4), extent = (40.0, 40.0, 40.0))
    sim = BoussinesqSimulation(m)
    Δ, e0, T = 10.0, 1.0, 600.0
    @test tke_dissipation_coefficient(Δ, Δ) ≈ 0.70
    exact(t) = (e0^-0.5 + 0.70 * t / (2Δ))^-2           # de/dt = −c_ε e^{3/2}/l
    c = TKEClosure(m; e0 = e0, dissipation = :exact)
    for _ in 1:60
        advance_sgs!(c, sim, T / 60)
    end
    @test maximum(abs.(interior_view(c.e) .- exact(T))) / exact(T) < 1e-6   # measured ≤ 1.7e-14
    @test tke_statistics(c).clips == 0
end

@testset "2. shear equilibrium: C_s = c_k^{3/4} c_ε^{-1/4}; critical Ri" begin
    S, Δ = 0.05, 10.0
    eq = 0.1 / 0.7 * Δ^2 * S^2                            # e = (c_k/c_ε) l² S²
    @test tke_equilibrium(S^2, 0.0, Δ) ≈ eq rtol = 1e-12
    # ν_t = c_k Δ √e at that e is Smagorinsky's with C_s = c_k^{3/4} c_ε^{-1/4} = 0.19441
    @test sqrt(0.1 * Δ * sqrt(eq) / (Δ^2 * S)) ≈ 0.1^0.75 * 0.70^-0.25 rtol = 1e-12
    Ric = tke_critical_richardson()
    @test Ric ≈ 1 / (1 + 0.19 / (0.1 * 0.76^2)) rtol = 1e-14
    @test tke_equilibrium(S^2, 0.25 * S^2, Δ) == 0          # above Ri_c: no equilibrium
    # below l = Δ, P_s + P_b = ε is linear in x = l/Δ: the closed-form root
    Ri = 0.2; a = 0.1 * 0.76^2
    x = (1 / Ri - 1 - 0.19 / a) / (2 + 0.51 / a)
    @test tke_equilibrium(S^2, Ri * S^2, Δ) ≈ (x * Δ * sqrt(Ri) * S / 0.76)^2 rtol = 1e-10
end

@testset "contract: pure coefficients, exact rollback" begin
    m = RegularGrid(; size = (8, 8, 8), extent = (2π, 2π, 2π), halo = (3, 3, 3))
    sim = BoussinesqSimulation(m)
    rng = Xoshiro(5)
    for f in (sim.u, sim.v, sim.w)
        interior_view(f) .= 0.3 .* randn(rng, 8, 8, 8)
    end
    c = TKEClosure(m)
    initialize_tke!(c, sim)
    e0 = copy(c.e.data)
    compute_sgs!(c, sim); compute_sgs!(c, sim)
    @test c.e.data == e0                                   # coefficients change no memory
    νt = interior_view(sim.νt)
    @test minimum(νt) >= 0 && all(interior_view(sim.κt) .>= νt)   # κ_t = (1 + 2l/Δ) ν_t
    snap = sgs_state(c)
    advance_sgs!(c, sim, 0.01)
    @test c.e.data != e0
    restore_sgs_state!(c, snap)
    @test c.e.data == e0 && tke_statistics(c).steps == 0   # copied back, not computed
end

end
