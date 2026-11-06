# Era IV, Phase 41 — superdroplet Lagrangian microphysics: the headline
# validation. The stochastic all-or-nothing coalescence algorithm against
# GOLOVIN'S ANALYTIC SOLUTION for the additive kernel (Shima et al. 2009
# benchmark, 3600 one-second steps of 2^15 superdroplets), and its
# convergence with the superdroplet count. A closed form the solver knows
# nothing about: the check that can catch a subtly wrong multiplicity rule.
# The module (Köhler, kernels, exact mass conservation of coalescence, the
# Golovin closed form itself, condensation) stays tested in
# test/superdroplet.jl.

using Test, Printf, Random, Statistics
using Aeolus

@testset "Era IV — Superdroplet microphysics (Phase 41)" begin

    @testset "Golovin analytic solution — the headline validation" begin
        # Shima et al. (2009) benchmark: n0 = 2^23 m^-3, mean mass
        # x̄ = 3.3e-12 kg (r̄ = 30.531 μm), b = 1.5 m³ kg⁻¹ s⁻¹, V = 1e6 m³.
        n0 = 2.0^23; r̄ = 30.531e-6; V = 1e6; b = 1.5
        rng = MersenneTwister(2009)
        sd = init_exponential_superdroplets(rng, 2^15, n0, r̄, V)
        K = GolovinKernel(b_mass = b)
        m0 = total_water_mass(sd)

        edges = exp.(range(log(1e-6), log(5e-3); length = 41))
        centres = sqrt.(edges[1:end-1] .* edges[2:end])

        function mass_density(sd)
            g = zeros(length(centres))
            for k in eachindex(sd.ξ)
                j = searchsortedlast(edges, sd.radius[k])
                (1 <= j <= length(centres)) || continue
                g[j] += sd.ξ[k] * (4π/3) * 1000.0 * sd.radius[k]^3 /
                        (V * log(edges[j+1] / edges[j]))
            end
            return g
        end

        errs = Float64[]
        t = 0.0
        for t_target in (1200.0, 2400.0, 3600.0)
            while t < t_target
                coalescence_step!(rng, sd, K, 1.0, V); t += 1.0
            end
            ana = [golovin_mass_density(r, t, n0, r̄, b) for r in centres]
            num = mass_density(sd)
            sel = findall(a -> a > 1e-8 * maximum(ana), eachindex(ana))
            err = sum(abs(num[i] - ana[i]) for i in sel) / sum(ana[i] for i in sel)
            push!(errs, err)
            @test total_water_mass(sd) ≈ m0 rtol = 1e-12
            @info @sprintf("Golovin t=%4.0fs: L1 error = %.4f", t, err)
        end
        # Published superdroplet results reproduce Golovin to a few percent
        # at early times, degrading as the distribution spreads into the tail
        # where finite N_sd leaves statistical scatter.
        @test errs[1] < 0.10
        @test errs[2] < 0.15
        @test errs[3] < 0.25
        @info "Golovin L1 errors" t1200=errs[1] t2400=errs[2] t3600=errs[3]
    end

    @testset "Golovin convergence with superdroplet count" begin
        # The error should fall as N_sd rises — the defining property of a
        # Monte-Carlo method, and the thing a wrong multiplicity rule breaks
        # (it would converge to the wrong answer instead).
        n0 = 2.0^23; r̄ = 30.531e-6; V = 1e6; b = 1.5
        edges = exp.(range(log(1e-6), log(5e-3); length = 41))
        centres = sqrt.(edges[1:end-1] .* edges[2:end])
        ana = [golovin_mass_density(r, 2400.0, n0, r̄, b) for r in centres]
        sel = findall(a -> a > 1e-8 * maximum(ana), eachindex(ana))
        errs = Float64[]
        for N_sd in (2^11, 2^13, 2^15)
            rng = MersenneTwister(11)
            sd = init_exponential_superdroplets(rng, N_sd, n0, r̄, V)
            K = GolovinKernel(b_mass = b)
            for _ in 1:2400; coalescence_step!(rng, sd, K, 1.0, V); end
            g = zeros(length(centres))
            for k in eachindex(sd.ξ)
                j = searchsortedlast(edges, sd.radius[k])
                (1 <= j <= length(centres)) || continue
                g[j] += sd.ξ[k]*(4π/3)*1000.0*sd.radius[k]^3 /
                        (V*log(edges[j+1]/edges[j]))
            end
            push!(errs, sum(abs(g[i]-ana[i]) for i in sel)/sum(ana[i] for i in sel))
        end
        @test errs[end] < errs[1]
        @info "Golovin convergence vs N_sd (2^11, 2^13, 2^15)" errors=errs
    end
end
