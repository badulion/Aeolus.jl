# Era IV, Phase 45 — turbulence-microphysics interaction, issue #13.
#
# The sub-grid supersaturation distribution of issue #12 fed into
# aerosol activation (Phase 42) and condensational droplet growth
# (Phase 41). Headline checks: σ_S → 0 recovers the Phase 41/42 results;
# the sign of the activation change is the sign of the curvature of the
# CCN spectrum (Jensen); the spread of r² matches the analytic
# Ornstein-Uhlenbeck result; relative dispersion rises with σ_S while
# total water is conserved to round-off.

using Test, Random, Statistics
using Aeolus
using Aeolus.AerosolCloud: ccn_fraction, ccn_spectrum

@testset "Era IV — SGS activation and growth (Phase 45, #13)" begin

    modes = [arg_mode(300e6, 0.04e-6, 1.8, 0.6),
             arg_mode(50e6, 0.15e-6, 1.6, 0.6)]
    T_K, p = 283.0, 90000.0
    Nccn(S) = ccn_spectrum(modes, S; T_K = T_K)

    @testset "CCN spectrum: one copy, shared with arg_activation" begin
        r = arg_activation(modes, 0.5, T_K, p)
        @test values_agree([ccn_fraction(m, r.S_max; T_K = T_K) for m in modes], r.fraction)
        @test values_agree(Nccn(r.S_max), r.N_act_total)
        # N_CCN(S) is a CDF in S: zero at S ≤ 0, monotone, → N_tot
        @test all(diff([Nccn(S) for S in 10 .^ range(-5, -1; length = 50)]) .>= 0)
        @test Nccn(0.0) == 0
        @test Nccn(10.0) ≈ sum(m.N for m in modes) rtol = 1e-12
        # at S = S_m exactly half of the mode activates (median particle)
        @test ccn_fraction(modes[1], mode_critical_supersaturation(modes[1], T_K);
                           T_K = T_K) ≈ 0.5 atol = 1e-14
    end

    @testset "SGS activation: σ_S → 0 limit, quadrature, Jensen" begin
        r42 = arg_activation(modes, 0.5, T_K, p)
        r45 = sgs_activation(modes, 0.5, T_K, p, 0.0)
        @test r45.N_act_total == r42.N_act_total && r45.mean.S_max == r42.S_max
        @test sgs_activation(modes, 0.5, T_K, p, 1e-9).N_act_total ≈ r42.N_act_total rtol = 1e-9
        @test sgs_activation(modes, 0.0, T_K, p, 1e-3).N_act_total == 0
        # against an independent Simpson integral of N_CCN over the Gaussian
        S̄, σ = 1e-3, 5e-4
        n = 4000; a = 0.0; b = S̄ + 12σ; h = (b - a) / n
        f(S) = Nccn(S) * exp(-(S - S̄)^2 / (2σ^2)) / (sqrt(2π) * σ)
        ref = (f(a) + f(b) + sum((isodd(i) ? 4 : 2) * f(a + i * h) for i in 1:n-1)) * h / 3
        @test sgs_ccn_spectrum(modes, S̄, σ; T_K = T_K) ≈ ref rtol = 1e-8
        # Jensen: ⟨N(S)⟩ − N(S̄) = ½ N''(S̄) σ² + O(σ⁴). Convex at 0.02 %
        # (coarse mode, S_m = 0.033 %), concave at 0.2 %.
        for S̄ in (2e-4, 2e-3)
            hh = 1e-3 * S̄
            d2 = (Nccn(S̄ + hh) - 2Nccn(S̄) + Nccn(S̄ - hh)) / hh^2
            σ = 0.05 * S̄
            @test (sgs_ccn_spectrum(modes, S̄, σ; T_K = T_K) - Nccn(S̄)) / (0.5 * d2 * σ^2) ≈ 1 atol = 0.05
        end
        # σ_S from a SubgridCloud is the same as passing the number
        sc = subgrid_cloud(288.0, 9e4, 9.5e-3; var_qt = (4e-4)^2)
        @test sc.σ_S > 0
        @test sgs_activation(modes, 0.5, T_K, p, sc).N_act_total ==
              sgs_activation(modes, 0.5, T_K, p, sc.σ_S).N_act_total
    end

    @testset "Ornstein-Uhlenbeck supersaturation" begin
        # stationary unit variance, lag-one correlation e^{-dt/τ}; 20 000
        # samples: s.e. of var ≈ 0.01, of cor ≈ 0.006
        fl = supersaturation_fluctuations(20_000; σ_S = 1e-3, τ = 4.0, rng = MersenneTwister(3))
        ξ0 = copy(fl.ξ)
        advance_fluctuations!(fl, 2.0)
        @test var(fl.ξ) ≈ 1 atol = 0.04
        @test cor(ξ0, fl.ξ) ≈ exp(-0.5) atol = 0.025
        # review fix: stepping cell by cell ages each droplet ONCE per dt
        nd = 20_000
        sdm = init_monodisperse_superdroplets(nd, 1e8, 10e-6, 2.0)
        sdm.cell[1:nd÷2] .= 1; sdm.cell[nd÷2+1:end] .= 2
        cells = [EulerianCell(temperature = 283.0, q_vap = 8.6e-3, density = 1.1, volume = 1.0) for _ in 1:2]
        flm = supersaturation_fluctuations(sdm; σ_S = 1e-3, τ = 4.0, rng = MersenneTwister(5))
        ξm0 = copy(flm.ξ)
        for k in 1:2
            sgs_condensation_step!(sdm, cells[k], 2.0, flm; index = k)
        end
        @test cor(ξm0, flm.ξ) ≈ exp(-0.5) atol = 0.025      # was e^{-1} before the fix
        # review fix: τ = 1e-30 used to overflow ceil(Int, 1e32)
        @test Aeolus.TurbulentMicrophysics._ou_substeps(
                  supersaturation_fluctuations(2; σ_S = 1e-3, τ = 1e-30), 10.0) == 100_000
        # the analytic integral variance and its limits
        @test supersaturation_integral_variance(1e-3, Inf, 30.0) ≈ (1e-3 * 30)^2
        @test supersaturation_integral_variance(1e-3, 0.01, 1000.0) ≈ 2 * (1e-3)^2 * 0.01 * 1000 rtol = 1e-4
    end

    @testset "SGS growth: Var(r²) against the analytic OU result" begin
        # Var(r²) = (2G)² · 2σ²τ[t − τ(1−e^{−t/τ})] (Sardina et al. 2015).
        # 4000 droplets: sampling s.e. of a variance ≈ 2.2 %; 7 % ≈ 3 s.e.
        T, pr = 283.0, 9e4
        G = 1 / (fk_thermal(T) + fd_vapor(T, pr))
        S̄, σ, t, τ = 5e-3, 2e-3, 30.0, 5.0
        sd = init_monodisperse_superdroplets(4000, 1e8, 10e-6, 1.0)
        fl = supersaturation_fluctuations(sd; σ_S = σ, τ = τ, rng = MersenneTwister(1))
        for _ in 1:round(Int, t / (τ / 10))
            sgs_condensation_growth!(sd, fl, S̄, T, pr, τ / 10)
        end
        @test var(sd.radius .^ 2) / ((2G)^2 * supersaturation_integral_variance(σ, τ, t)) ≈ 1 atol = 0.07
    end

    @testset "SGS growth: σ_S = 0 is Phase 41; dispersion rises with σ_S, water conserved" begin
        function setup()
            sd = init_monodisperse_superdroplets(300, 1e8, 5e-6, 1.0; solute_mass = 1e-17)
            for k in eachindex(sd.radius)
                sd.radius[k] *= 1 + 0.4 * (k / 300 - 0.5)
            end
            return sd, EulerianCell(temperature = 283.0, q_vap = 9.6e-3, density = 1.1, volume = 1.0)
        end
        function growth_case(σ_S)
            sd, cell = setup()
            fl = supersaturation_fluctuations(sd; σ_S = σ_S, τ = 10.0, rng = MersenneTwister(7))
            W0 = cell_total_water(sd, cell)
            for _ in 1:40
                sgs_condensation_step!(sd, cell, 0.5, fl)
            end
            return (; sd, cell, ε = relative_dispersion(sd)[2],
                      dW = (cell_total_water(sd, cell) - W0) / W0)
        end
        runs = [growth_case(σ) for σ in (0.0, 1e-3, 4e-3)]
        sd, cell = setup()
        ε0 = relative_dispersion(sd)[2]
        for _ in 1:40
            condensation_step!(sd, cell, 0.5)
        end
        # σ_S = 0 runs the Phase 41 routine (identical copy of the same arithmetic)
        @test runs[1].sd.radius == sd.radius && runs[1].cell.q_vap == cell.q_vap
        @test runs[1].ε < ε0                                   # condensation narrows
        @test issorted([r.ε for r in runs]) && allunique([r.ε for r in runs])
        @test all(abs(r.dW) < 1e-14 for r in runs)             # total water, round-off
    end
end
