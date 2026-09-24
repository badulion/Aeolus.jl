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
end
