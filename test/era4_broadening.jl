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
end
