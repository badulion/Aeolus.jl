# Era IV, Phase 45 — turbulence-microphysics interaction, issue #14.
#
# Spectral broadening by sub-grid supersaturation fluctuations: the bulk
# moment closure against the Phase 41 superdroplet reference and against
# the DNS scaling σ_{r²} ∝ t^{1/2} (Lanotte et al. 2009; Sardina et al.
# 2015).

using Test, Printf, Random, Statistics
using Aeolus

@testset "Era IV — spectral broadening validation (Phase 45, #14)" begin

    @testset "gamma moment ratio Γ(k+p)/(Γ(k)kᵖ)" begin
        # closed forms: Γ(3/2) = √π/2, Γ(1/2) = √π, Γ(k+1) = kΓ(k)
        @test gamma_moment_ratio(1.0, 0.5) ≈ sqrt(π) / 2 rtol = 1e-14
        @test gamma_moment_ratio(0.5, 0.5) ≈ 1 / sqrt(π * 0.5) rtol = 1e-14
        @test gamma_moment_ratio(1.5, -0.5) ≈ 2 / sqrt(π) * sqrt(1.5) rtol = 1e-14
        for k in (0.3, 1.0, 7.5, 15.9, 16.0, 40.0, 1e4)
            @test gamma_moment_ratio(k, 1.0) ≈ 1.0 rtol = 1e-13
            # Γ(k+2) = (k+1) k Γ(k)
            @test gamma_moment_ratio(k, 2.0) ≈ (k + 1) / k rtol = 1e-13
            # recurrence in p: Γ(k+p+1) = (k+p) Γ(k+p)
            @test gamma_moment_ratio(k, 1.5) ≈
                  gamma_moment_ratio(k, 0.5) * (k + 0.5) / k rtol = 1e-13
        end
        # narrow-spectrum asymptote 1 − 1/(8k) + 1/(128k²) + 5/(1024k³)
        for k in (1e3, 1e6)
            @test gamma_moment_ratio(k, 0.5) ≈
                  1 - 1 / (8k) + 1 / (128k^2) + 5 / (1024k^3) rtol = 1e-13
        end
        @test gamma_moment_ratio(Inf, 0.5) == 1.0
        @test_throws DomainError gamma_moment_ratio(0.4, -0.5)
    end

    @testset "gamma closure against a sampled gamma population" begin
        # w = r² ~ Gamma(k = 40): the sum of 40 exponentials (no
        # Distributions.jl), 200 000 draws.  The closure has no freedom
        # here, so it must match the sample to Monte-Carlo accuracy.
        rng = MersenneTwister(14)
        k, μ = 40, (8e-6)^2
        n = 200_000
        w = [sum(-log(rand(rng)) for _ in 1:k) * μ / k for _ in 1:n]
        m = SpectralMoments(n, mean(w), var(w; corrected = false), 0.0)
        r = sqrt.(w)
        @test closure_radius_moment(m, 2) == m.μ
        for p in (1, 3, -1)
            est = mean(r .^ p)
            se = std(r .^ p) / sqrt(n)
            @test abs(closure_radius_moment(m, p) - est) < 4se
        end
        σ, ε = relative_dispersion(m)
        @test ε ≈ std(r; corrected = false) / mean(r) rtol = 0.01
        # ε ≈ 1/(2√k) for a narrow gamma in r²
        @test ε ≈ 1 / (2sqrt(k)) rtol = 0.02
        @test r2_standard_deviation(m) ≈ std(w; corrected = false) rtol = 1e-12
        # monodisperse limit
        m0 = SpectralMoments(1e6, 25e-12, 0.0, 0.0)
        @test relative_dispersion(m0) == (0.0, 0.0)
        @test mean_radius(m0) ≈ 5e-6 rtol = 1e-15
        @test effective_radius(m0) ≈ 5e-6 rtol = 1e-15
    end

    @testset "spectral_moments of a superdroplet set" begin
        V = 1.0
        sd = init_monodisperse_superdroplets(500, 1e8, 6e-6, V; solute_mass = 1e-18)
        rng = MersenneTwister(3)
        for i in eachindex(sd.radius)
            sd.radius[i] *= exp(0.1 * randn(rng))
            sd.ξ[i] = 1 + (i % 7)                 # unequal multiplicities
        end
        m = spectral_moments(sd)
        wts = Float64.(sd.ξ); w = Float64.(sd.radius) .^ 2
        @test m.N == sum(wts)
        @test m.μ ≈ sum(wts .* w) / sum(wts) rtol = 1e-14
        @test m.V ≈ sum(wts .* (w .- m.μ) .^ 2) / sum(wts) rtol = 1e-12
        @test m.C == 0
        # the bulk and the ensemble agree on r̄ and ε to the closure error:
        # log-normal ensemble vs gamma-in-r² closure differ in the third
        # cumulant of r², which enters r̄ at O(ε⁴) = 1e-4 here (measured
        # 2.0e-4, i.e. a coefficient ≈ 2)
        @test mean_radius(m) ≈ mean_radius(sd) rtol = 5e-4
        @test relative_dispersion(m)[2] ≈ relative_dispersion(sd)[2] rtol = 0.02
        @test r2_standard_deviation(sd) ≈ sqrt(m.V) rtol = 1e-14
        # covariance with a fluctuation field that *is* r² is Var(r²)/σ
        fl = supersaturation_fluctuations(sd; σ_S = 1e-3)
        fl.ξ .= (w .- m.μ) ./ sqrt(m.V)
        @test spectral_moments(sd; fl = fl).C ≈ sqrt(m.V) rtol = 1e-10
    end
    @testset "parcel_ascent!: adiabatic source of supersaturation" begin
        # Without droplets, lifting a saturated parcel must raise S at the
        # rate of Phase 42's α(T, p) — an independent (linearised)
        # derivation. α uses the dry-air molar mass in its pressure term,
        # an O(0.16 q_v) ≈ 0.1-0.25 % difference: measured ratios 1.0015
        # (283 K, 900 hPa), 1.0024 (293 K, 1000 hPa), 1.0012 (275 K, 700 hPa).
        for (T, p) in ((283.0, 9e4), (293.0, 1e5), (275.0, 7e4))
            qs = saturation_specific_humidity(T, p)
            cell = cell_from_state(T, p, qs, 1.0)
            S0 = cell_supersaturation(cell)
            m0 = cell_air_mass(cell)
            for _ in 1:10
                parcel_ascent!(cell, 1.0, 0.1)
            end
            dSdz = cell_supersaturation(cell) - S0
            α = Aeolus.AerosolCloud.supersaturation_source(T, p; q = PhasePartition(qs))
            @test dSdz / α ≈ 1 atol = 3e-3
            @test cell_air_mass(cell) ≈ m0 rtol = 1e-14      # fixed mass
            @test cell.temperature ≈ T - 1.0 * dry_adiabatic_lapse_rate(PhasePartition(qs)) rtol = 1e-12
        end
    end

    @testset "bulk moments: exact Ornstein-Uhlenbeck limit" begin
        # With the Kelvin term switched off (zero surface tension) the
        # moment equations are exact and closed: an initially monodisperse
        # population must acquire Var(r²) = (2G)² Var∫S'dt exactly — the
        # analytic result of issue #13 — for any τ, including frozen.
        mc0 = Aeolus.Superdroplet.MicrophysicalConstants(σ_water0 = 0.0, dσ_dT = 0.0)
        T, p = 283.0, 9e4
        for τ in (0.5, 5.0, 50.0, Inf), σ in (5e-4, 2e-3)
            # one droplet in 1e6 m³: the cell state does not move
            cell = cell_from_state(T, p, 1.002saturation_specific_humidity(T, p), 1e6)
            G = 1 / (Aeolus.Superdroplet.fk_thermal(T, mc0) +
                     Aeolus.Superdroplet.fd_vapor(T, cell_pressure(cell), mc0))
            m = SpectralMoments(1.0, 25e-12, 0.0, 0.0)
            bulk_condensation_step!(m, cell, 60.0; σ_S = σ, τ = τ, mc = mc0)
            @test m.V ≈ (2G)^2 * supersaturation_integral_variance(σ, τ, 60.0) rtol = 1e-9
            # Cov(r², ξ) → 2Gσ_S τ (1 − e^{−t/τ}), σ_S t when frozen
            @test m.C ≈ (isfinite(τ) ? 2G * σ * τ * -expm1(-60 / τ) : 2G * σ * 60) rtol = 1e-9
        end
        # σ_S = 0: no covariance, no variance from fluctuations
        cell = cell_from_state(T, p, 1.002saturation_specific_humidity(T, p), 1e6)
        m = SpectralMoments(1.0, 25e-12, 1e-24, 0.0)
        bulk_condensation_step!(m, cell, 60.0; mc = mc0)
        @test m.C == 0 && m.V == 1e-24
    end

    @testset "bulk step: conservation, narrowing without fluctuations" begin
        T, p = 283.0, 9e4
        mc = Aeolus.Superdroplet.DEFAULT_MICRO
        ε_end = Dict{Float64,Float64}(); V_end = Dict{Float64,Float64}()
        for σ in (0.0, 1e-3)
            cell = cell_from_state(T, p, 1.003saturation_specific_humidity(T, p), 1.0)
            m = SpectralMoments(1e8, 25e-12, (0.2 * 25e-12)^2, 0.0)
            fac = mc.ρ_water * 4π / 3
            W(m, cell) = cell.q_vap + fac * m.N * closure_radius_moment(m, 3) / cell_air_mass(cell)
            W0 = W(m, cell)
            ε0 = relative_dispersion(m)[2]
            εs = Float64[]
            for _ in 1:60
                parcel_ascent!(cell, 0.5, 1.0)
                bulk_condensation_step!(m, cell, 1.0; σ_S = σ, τ = 5.0)
                push!(εs, relative_dispersion(m)[2])
            end
            @test abs(W(m, cell) / W0 - 1) < 1e-13
            @test mean_radius(m) > 5e-6
            if σ == 0
                # dr²/dt the same for all: the r-spectrum narrows monotonically
                @test all(diff([ε0; εs]) .< 0)
                @test m.C == 0
            end
            ε_end[σ] = εs[end]; V_end[σ] = m.V
        end
        # same parcel, same mean growth: the fluctuations widen the spectrum,
        # by the OU variance evaluated with G at the initial state. Measured
        # ratio 1.025: G changes as the parcel cools and warms, and the
        # Kelvin feedback on C (λ = 1/τ − Ga μ^{-3/2}) lengthens its memory.
        G = 1 / (Aeolus.Superdroplet.fk_thermal(T, mc) +
                 Aeolus.Superdroplet.fd_vapor(T, p, mc))
        ΔV_ou = (2G)^2 * supersaturation_integral_variance(1e-3, 5.0, 60.0)
        @printf("  bulk parcel 60 s: ε = %.4f (σ_S = 0) vs %.4f (σ_S = 0.1 %%); ΔVar(r²)/OU = %.4f\n",
                ε_end[0.0], ε_end[1e-3], (V_end[1e-3] - V_end[0.0]) / ΔV_ou)
        @test ε_end[1e-3] > ε_end[0.0]
        @test (V_end[1e-3] - V_end[0.0]) / ΔV_ou ≈ 1 atol = 0.05
    end
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
                # Measured over seeds 1-3: −4.1 %, −4.1 %, −3.5 %.
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
        # seeds is ±0.008 (seeds 1-3: 0.529, 0.514, 0.518), so 0.03 ≈ 4σ
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

    @testset "no broadening without fluctuations" begin
        # identical droplets in a uniform S stay identical: σ_{r²} is
        # round-off, not physics, in both representations
        case = BroadeningCase(ε0 = 0.0, r0 = 10e-6, w = 0.5, σ_S = 0.0,
                              t_end = 300.0, n_sd = 500)
        Hs = run_broadening_case(case)
        Hb = run_broadening_case(case; model = :bulk)
        μ_end = Hs.mean_radius[end]^2
        @test maximum(Hs.σ_r2) < 1e-10μ_end
        @test maximum(Hb.σ_r2) < 1e-10μ_end
        @test maximum(Hs.ε) < 1e-6 && maximum(Hb.ε) < 1e-6
        @test Hs.mean_radius[end] > 1.1 * Hs.mean_radius[1]      # they did grow
        # and σ_S = 0 in the reference is Phase 41's condensation_step!
        sd, cell = broadening_initial_population(BroadeningCase(n_sd = 200))
        sd2, cell2 = deepcopy(sd), deepcopy(cell)
        fl = supersaturation_fluctuations(sd; σ_S = 0.0, τ = 5.0)
        sgs_condensation_step!(sd, cell, 1.0, fl)
        condensation_step!(sd2, cell2, 1.0)
        @test sd.radius == sd2.radius && cell.q_vap == cell2.q_vap
    end
end
