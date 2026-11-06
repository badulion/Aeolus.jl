# Phase 45 — turbulence-microphysics interaction, issue #14: the
# validation case. Spectral broadening by sub-grid supersaturation
# fluctuations, the bulk moment closure against the Phase 41 superdroplet
# reference in an ascending parcel (300 s, n_sd = 8000, three σ_S), and
# against the DNS scaling σ_{r²} ∝ t^{1/2} under stationary fluctuations
# (Lanotte et al. 2009; Sardina et al. 2015). The closure stays tested in
# test/broadening.jl (gamma moment ratio, moments, the exact OU limit,
# conservation, fit_power_law).

using Test, Printf, Random, Statistics
using Aeolus

@testset "spectral broadening validation (Phase 45, #14)" begin

    # ---------------------------------------------------------------------
    # The validation case (issue #14): superdroplet reference vs bulk
    # ---------------------------------------------------------------------
    relerr(a, b) = a / b - 1
    Wdrift(H) = maximum(abs.(H.total_water ./ H.total_water[1] .- 1))

    @testset "ascending parcel: bulk vs superdroplet" begin
        # 100 cm⁻³ of 5 μm droplets (log-normal, ε₀ = 0.1) in a parcel
        # rising at 0.5 m/s for 300 s, τ = 5 s. Tolerances:
        #   σ_{r²}: closure-free, so only sampling — the standard error of a
        #           sample standard deviation is 1/√(2 n_sd) = 0.79 % for
        #           n_sd = 8000; 3 standard errors.
        #   ε:      sampling as above, plus the gamma-closure shape error,
        #           which is measured on the *initial* spectrum (the same
        #           droplets, before any growth) and carried as is.
        #   r̄, S̄:  1e-3 and 2 % (measured 2e-5 and ≤ 0.6 %).
        n_sd = 8000
        tol_samp = 3 / sqrt(2n_sd)
        res = Dict{Float64,Any}()
        for σ in (0.0, 1e-3, 2e-3)
            case = BroadeningCase(σ_S = σ, n_sd = n_sd, seed = 1)
            Hs = run_broadening_case(case)
            Hb = run_broadening_case(case; model = :bulk)
            res[σ] = (Hs, Hb)
            err0 = abs(relerr(Hb.ε[1], Hs.ε[1]))
            @printf("  parcel σ_S = %.1f %%: ε = %.4f (SD) / %.4f (bulk), σ_r² = %.2f / %.2f μm², r̄ = %.3f / %.3f μm, S̄ = %.4f / %.4f %% [closure at t=0: %.2f %%]\n",
                    100σ, Hs.ε[end], Hb.ε[end], 1e12Hs.σ_r2[end], 1e12Hb.σ_r2[end],
                    1e6Hs.mean_radius[end], 1e6Hb.mean_radius[end],
                    100Hs.S[end], 100Hb.S[end], 100err0)
            @test Wdrift(Hs) < 1e-13 && Wdrift(Hb) < 1e-13
            @test abs(relerr(Hb.σ_r2[end], Hs.σ_r2[end])) < tol_samp
            if σ <= 1e-3
                @test abs(relerr(Hb.ε[end], Hs.ε[end])) < tol_samp + err0
                @test abs(relerr(Hb.mean_radius[end], Hs.mean_radius[end])) < 1e-3
                @test abs(relerr(Hb.S[end], Hs.S[end])) < 0.02
            else
                # Known limitation, measured, not tolerated away: when the
                # fluctuation-made variance dominates, r² is close to
                # Gaussian (symmetric) while the closure assumes a gamma
                # (skewness 2/√k); for the same Var(r²) the gamma puts less
                # weight in the small-r tail, so the bulk ε is LOW.
                # Measured over seeds 1-3: −2.9 %, −5.1 %, −4.3 % (counter-based
                # streams, #305; −4.1 %, −4.1 %, −3.5 % with the old draws).
                @test 0.02 < -relerr(Hb.ε[end], Hs.ε[end]) < 0.06
                @test abs(relerr(Hb.mean_radius[end], Hs.mean_radius[end])) < 2e-3
            end
        end
        # without fluctuations the spectrum narrows (Phase 41), in both —
        # after the first second: the parcel starts at S = 0, below the
        # Kelvin equilibrium of a 5 μm droplet (a/r ≈ 0.024 %), so in the
        # first step the smallest droplets lag and ε rises by 0.06 %
        # (0.09958 → 0.09964 SD, 0.10033 → 0.10039 bulk) before S̄ passes
        # a/r. The original "narrows from t = 0" expectation was wrong.
        for H in res[0.0]
            @test H.ε[2] > H.ε[1]
            @test all(diff(H.ε)[2:end] .< 0)
        end
        # the fluctuations double the final width, in both representations
        rs = res[1e-3][1].ε[end] / res[0.0][1].ε[end]
        rb = res[1e-3][2].ε[end] / res[0.0][2].ε[end]
        @test rs > 2 && rb > 2
        @test rb ≈ rs rtol = tol_samp + 0.012
    end

    @testset "stationary fluctuations: σ_{r²} ∝ t^{1/2} (DNS scaling)" begin
        # The DNS configuration (Sardina et al. 2015; Lanotte et al. 2009):
        # no mean updraught, statistically stationary S', monodisperse
        # 10 μm droplets. S̄ relaxes to the Kelvin equilibrium (~0.01 %)
        # and every droplet random-walks in r². Fit window t ≥ 10τ.
        τ, σ = 5.0, 1e-3
        base = (ε0 = 0.0, r0 = 10e-6, w = 0.0, σ_S = σ, τ = τ, t_end = 600.0)
        win(H) = H.t .>= 10τ
        pfit(H) = fit_power_law(H.t[win(H)], H.σ_r2[win(H)])[1]
        t_w = collect(10τ:1.0:600.0)
        p_OU = fit_power_law(t_w, sqrt.(supersaturation_integral_variance.(σ, τ, t_w)))[1]

        # (i) Kelvin off: the moment scheme is the exact OU result
        mc0 = Aeolus.Superdroplet.MicrophysicalConstants(σ_water0 = 0.0, dσ_dT = 0.0)
        case = BroadeningCase(; base..., n_sd = 4000, seed = 1)
        Hb0 = run_broadening_case(case; model = :bulk, mc = mc0)
        Hs0 = run_broadening_case(case; mc = mc0)
        G = 1 / (Aeolus.Superdroplet.fk_thermal(Hb0.T[end], mc0) +
                 Aeolus.Superdroplet.fd_vapor(Hb0.T[end], 9e4, mc0))
        pref0 = Hb0.σ_r2[end] / (2G * sqrt(supersaturation_integral_variance(σ, τ, 600.0)))
        # (ii) full physics
        Hb = run_broadening_case(case; model = :bulk)
        Hs = run_broadening_case(case)
        pref = Hb.σ_r2[end] / (2G * sqrt(supersaturation_integral_variance(σ, τ, 600.0)))
        @printf("  σ_r² ∝ t^p over t ∈ [%g, 600] s: p_OU = %.4f; Kelvin off p_bulk = %.4f, p_SD = %.4f; full p_bulk = %.4f, p_SD = %.4f; prefactor/OU = %.4f (off), %.4f (full)\n",
                10τ, p_OU, pfit(Hb0), pfit(Hs0), pfit(Hb), pfit(Hs), pref0, pref)
        # the analytic OU curve in this window is t^{0.514}: → 1/2 as t/τ → ∞
        @test abs(p_OU - 0.5) < 0.02
        @test pfit(Hb0) ≈ p_OU atol = 2e-3
        @test pref0 ≈ 1 atol = 5e-3
        # the reference reproduces the DNS law; SD exponent scatter between
        # seeds is ±0.008 (full physics, seeds 1-3: 0.540, 0.529, 0.526 with the
        # counter-based streams of #305; 0.529, 0.514, 0.518 before), so 0.03 ≈ 4σ
        @test abs(pfit(Hs0) - p_OU) < 0.03
        @test abs(pfit(Hs) - pfit(Hb)) < 0.03
        @test abs(pfit(Hs) - 0.5) < 0.05
        # Kelvin curvature adds its own broadening (small droplets grow
        # more slowly: d ln Var/dt = 2Ga μ^{-3/2} ≈ 2e-4 s⁻¹ at 10 μm), so
        # the full width and exponent sit slightly above OU
        @test 1.0 < pref < 1.06
        @test pfit(Hb) > pfit(Hb0)
        @test Wdrift(Hs) < 1e-13 && Wdrift(Hb) < 1e-13

        # frozen sub-volumes (τ = ∞, Paoli & Shariff 2009 short-time limit):
        # σ_r² = 2Gσ_S t, exponent 1
        casef = BroadeningCase(; base..., τ = Inf, t_end = 60.0, n_sd = 4000)
        Hsf = run_broadening_case(casef)
        Hbf = run_broadening_case(casef; model = :bulk)
        pf_s = fit_power_law(Hsf.t[2:end], Hsf.σ_r2[2:end])[1]
        pf_b = fit_power_law(Hbf.t[2:end], Hbf.σ_r2[2:end])[1]
        @printf("  frozen (τ = ∞): p_SD = %.4f, p_bulk = %.4f\n", pf_s, pf_b)
        @test pf_s ≈ 1 atol = 0.01
        @test pf_b ≈ 1 atol = 0.01
    end
end
