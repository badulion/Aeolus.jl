# Phase 59 (P59-b, #184): validation of the σ-model SGS closure of
# Nicoud, Baya Toda, Cabrit, Bose & Lee (2011, Phys. Fluids 23, 085106) in
# decaying isotropic turbulence (32³, t ∈ [1, 3], two seeds): its kinetic
# energy decay exponent against Smagorinsky C_s = 0.17 (±15 %), and the
# closure, not ν, doing the decay. The closure stays tested in
# test/sigma_sgs.jl (analytic invariants, near-wall scaling, laminar
# Couette, the contract); the CBC row is in validation/cbc_decay.jl.

using Test, Printf, Random
using Statistics: mean
using Aeolus
using Aeolus.Fields: interior_view
isdefined(@__MODULE__, :kolmogorov_field!) || include(joinpath(@__DIR__, "..", "test", "support", "kolmogorov.jl"))

@testset "σ-model SGS closure in decaying isotropic turbulence (P59-b, #184)" begin

    # -----------------------------------------------------------------------
    @testset "4. decaying isotropic turbulence: KE decay vs Smagorinsky 0.17" begin
        # 32³ in a 2π box, ν = 1e-4: a k^{-5/3} solenoidal field up to N/4
        # with u_rms = 1 (the era2 dynamic-Smagorinsky set-up), SSP-RK3 at
        # dt = 0.01 to t = 3. The decay exponent n of K ∝ t^{-n} over
        # t ∈ [1, 3] is n = −ln(K(3)/K(1))/ln 3. C_σ = 1.35 is the value
        # Nicoud et al. (2011) obtained in isotropic turbulence, so the
        # σ-model should dissipate like Smagorinsky there; target ±15 %.
        mesh = RegularGrid(; size = (32, 32, 32), extent = (2π, 2π, 2π), halo = (3, 3, 3))
        ke(sim) = 0.5 * sum(f -> mean(abs2, interior_view(f)), (sim.u, sim.v, sim.w))
        function hit(sgs, seed)
            sim = BoussinesqSimulation(mesh; ν = 1e-4)
            kolmogorov_field!(sim, Xoshiro(seed))   # k_c = 8 = N/4, u_rms = 1
            K = Float64[]
            for n in 1:300
                step_rk3!(sim, 0.01; sgs = sgs)
                n % 100 == 0 && push!(K, ke(sim))
            end
            return (; K1 = K[1], K3 = K[3], n = -log(K[3] / K[1]) / log(3),
                      νt = mean(interior_view(sim.νt)))
        end
        for seed in (1, 2)
            s = hit(StaticSmagorinsky(0.17), seed)
            σ = hit(SigmaModel(1.35), seed)
            z = hit(nothing, seed)
            @printf("    seed %d: n = %.4f (Smagorinsky 0.17), %.4f (σ 1.35; ratio %.4f), %.4f (no model); K(3)/K(1) = %.4f / %.4f / %.4f; ⟨ν_t⟩(t=3) %.2e / %.2e\n",
                    seed, s.n, σ.n, σ.n / s.n, z.n, s.K3 / s.K1, σ.K3 / σ.K1, z.K3 / z.K1, s.νt, σ.νt)
            @test abs(σ.n / s.n - 1) <= 0.15
            @test σ.n > 5 * z.n                 # the closure, not ν, does the decay
        end
    end
end
