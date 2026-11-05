# Era IV, Phase 45 — turbulence-microphysics interaction, issue #14.
#
# Spectral broadening by sub-grid supersaturation fluctuations: the bulk
# moment closure and its building blocks. Its validation against the
# Phase 41 superdroplet reference and the DNS scaling σ_{r²} ∝ t^{1/2}
# (Lanotte et al. 2009; Sardina et al. 2015) is
# validation/era4_broadening.jl.

using Test, Random, Statistics
using Aeolus

@testset "Era IV — spectral broadening (Phase 45, #14)" begin

    @testset "gamma moment ratio Γ(k+p)/(Γ(k)kᵖ)" begin
        # closed forms: Γ(3/2) = √π/2, Γ(k+2) = (k+1)kΓ(k), recurrence in p
        @test gamma_moment_ratio(1.0, 0.5) ≈ sqrt(π) / 2 rtol = 1e-14
        ks = (0.3, 7.5, 16.0, 1e4)
        @test maximum(abs(gamma_moment_ratio(k, 2.0) / ((k + 1) / k) - 1) for k in ks) < 1e-13
        @test maximum(abs(gamma_moment_ratio(k, 1.5) /
                          (gamma_moment_ratio(k, 0.5) * (k + 0.5) / k) - 1) for k in ks) < 1e-13
        # narrow-spectrum asymptote
        k = 1e3
        @test gamma_moment_ratio(k, 0.5) ≈ 1 - 1 / (8k) + 1 / (128k^2) + 5 / (1024k^3) rtol = 1e-13
        @test_throws DomainError gamma_moment_ratio(0.4, -0.5)
    end

    @testset "gamma closure against a sampled gamma population" begin
        # w = r² ~ Gamma(k = 40), 40 000 draws: the closure has no freedom,
        # so it must match the sample to Monte-Carlo accuracy (4 s.e.).
        rng = MersenneTwister(14)
        k, μ, n = 40, (8e-6)^2, 40_000
        w = [sum(-log(rand(rng)) for _ in 1:k) * μ / k for _ in 1:n]
        m = SpectralMoments(n, mean(w), var(w; corrected = false), 0.0)
        r = sqrt.(w)
        @test all(abs(closure_radius_moment(m, p) - mean(r .^ p)) < 4std(r .^ p) / sqrt(n) for p in (1, 3, -1))
        # ε ≈ 1/(2√k) for a narrow gamma in r²
        @test relative_dispersion(m)[2] ≈ 1 / (2sqrt(k)) rtol = 0.02
        @test relative_dispersion(SpectralMoments(1e6, 25e-12, 0.0, 0.0)) == (0.0, 0.0)
    end

    @testset "spectral_moments of a superdroplet set" begin
        sd = init_monodisperse_superdroplets(500, 1e8, 6e-6, 1.0; solute_mass = 1e-18)
        rng = MersenneTwister(3)
        for i in eachindex(sd.radius)
            sd.radius[i] *= exp(0.1 * randn(rng))
            sd.ξ[i] = 1 + (i % 7)                 # unequal multiplicities
        end
        m = spectral_moments(sd)
        wts = Float64.(sd.ξ); w = Float64.(sd.radius) .^ 2
        @test m.μ ≈ sum(wts .* w) / sum(wts) rtol = 1e-14
        @test m.V ≈ sum(wts .* (w .- m.μ) .^ 2) / sum(wts) rtol = 1e-12
        # bulk vs ensemble r̄: closure error O(ε⁴) ≈ 1e-4 (measured 2.0e-4)
        @test mean_radius(m) ≈ mean_radius(sd) rtol = 5e-4
    end

    @testset "parcel_ascent!: adiabatic source of supersaturation" begin
        # S rises at Phase 42's α(T, p) (independent, linearised; differs by
        # O(0.16 q_v) ≈ 0.15 % here through the dry-air molar mass)
        T, p = 283.0, 9e4
        qs = saturation_specific_humidity(T, p)
        cell = cell_from_state(T, p, qs, 1.0)
        S0 = cell_supersaturation(cell); m0 = cell_air_mass(cell)
        for _ in 1:10
            parcel_ascent!(cell, 1.0, 0.1)
        end
        α = Aeolus.AerosolCloud.supersaturation_source(T, p; q = PhasePartition(qs))
        @test (cell_supersaturation(cell) - S0) / α ≈ 1 atol = 3e-3
        @test cell_air_mass(cell) ≈ m0 rtol = 1e-14
        @test cell.temperature ≈ T - dry_adiabatic_lapse_rate(PhasePartition(qs)) rtol = 1e-12
    end

    @testset "bulk moments: exact Ornstein-Uhlenbeck limit" begin
        # Kelvin term off: moment equations exact and closed; Var(r²) =
        # (2G)² Var∫S'dt for any τ, including frozen.
        mc0 = Aeolus.Superdroplet.MicrophysicalConstants(σ_water0 = 0.0, dσ_dT = 0.0)
        T, p = 283.0, 9e4
        for τ in (5.0, Inf)
            σ = 2e-3
            cell = cell_from_state(T, p, 1.002saturation_specific_humidity(T, p), 1e6)
            G = 1 / (Aeolus.Superdroplet.fk_thermal(T, mc0) +
                     Aeolus.Superdroplet.fd_vapor(T, cell_pressure(cell), mc0))
            m = SpectralMoments(1.0, 25e-12, 0.0, 0.0)
            bulk_condensation_step!(m, cell, 60.0; σ_S = σ, τ = τ, mc = mc0)
            @test m.V ≈ (2G)^2 * supersaturation_integral_variance(σ, τ, 60.0) rtol = 1e-9
            @test m.C ≈ (isfinite(τ) ? 2G * σ * τ * -expm1(-60 / τ) : 2G * σ * 60) rtol = 1e-9
        end
    end

    @testset "bulk step: water conserved, narrowing without fluctuations" begin
        T, p = 283.0, 9e4
        mc = Aeolus.Superdroplet.DEFAULT_MICRO
        fac = mc.ρ_water * 4π / 3
        W(m, cell) = cell.q_vap + fac * m.N * closure_radius_moment(m, 3) / cell_air_mass(cell)
        ε_end = Dict{Float64,Float64}()
        for σ in (0.0, 1e-3)
            cell = cell_from_state(T, p, 1.003saturation_specific_humidity(T, p), 1.0)
            m = SpectralMoments(1e8, 25e-12, (0.2 * 25e-12)^2, 0.0)
            W0 = W(m, cell); εs = [relative_dispersion(m)[2]]
            for _ in 1:30
                parcel_ascent!(cell, 0.5, 1.0)
                bulk_condensation_step!(m, cell, 1.0; σ_S = σ, τ = 5.0)
                push!(εs, relative_dispersion(m)[2])
            end
            @test abs(W(m, cell) / W0 - 1) < 1e-13
            σ == 0 && @test all(diff(εs) .< 0)
            ε_end[σ] = εs[end]
        end
        @test ε_end[1e-3] > ε_end[0.0]
    end

    @testset "fit_power_law recovers an exact power law" begin
        t = collect(range(1.0, 600.0; length = 200))
        pf, af = fit_power_law(t, 7.5 .* t .^ -1.3)
        @test pf ≈ -1.3 rtol = 1e-12
        @test af ≈ 7.5 rtol = 1e-11
        @test_throws DomainError fit_power_law([0.0, 1.0], [1.0, 2.0])
    end

    @testset "no broadening without fluctuations" begin
        # identical droplets in a uniform S stay identical in both models
        case = BroadeningCase(ε0 = 0.0, r0 = 10e-6, w = 0.5, σ_S = 0.0, t_end = 60.0, n_sd = 100)
        Hs = run_broadening_case(case)
        Hb = run_broadening_case(case; model = :bulk)
        μ_end = Hs.mean_radius[end]^2
        @test maximum(Hs.σ_r2) < 1e-10μ_end && maximum(Hb.σ_r2) < 1e-10μ_end
        @test Hs.mean_radius[end] > Hs.mean_radius[1]      # they did grow
    end
end
