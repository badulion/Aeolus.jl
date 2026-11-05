# Phase 43, issue #5 — prognostic ice-nucleating-particle species.
#
# Asserted: analytic log-normal moments against an independent
# quadrature, conservation of number and mass through every
# transport-free source and sink, the available/consumed split that stops
# an INP nucleating twice, and the published Stokes settling speeds.

using Test, Printf, Random
using Aeolus
using Aeolus.Superdroplet: stokes_settling_velocity, MicrophysicalConstants,
                           DEFAULT_MICRO

# Independent reference: midpoint rule in ln r (not the module's
# shifted standard-normal quadrature).
function _ref_moment_above(m::ARGMode, k::Integer, r_cut::Real; n = 200_000)
    lnσ = log(m.σ_g)
    lo = max(log(r_cut), log(m.r_g) - 14lnσ)
    hi = log(m.r_g) + 14lnσ
    lo >= hi && return 0.0
    h = (hi - lo) / n
    s = 0.0
    for j in 1:n
        x = lo + (j - 0.5) * h
        s += exp(k * x) * exp(-0.5((x - log(m.r_g)) / lnσ)^2)
    end
    return m.N / (sqrt(2π) * lnσ) * s * h
end

@testset "Prognostic INP species (issue #5)" begin
    dust = lognormal_mode(:mineral_dust, 1.0, 0.4, 2.0)

    @testset "Shared ingredients and Stokes settling" begin
        # biological INP are a Phase 37 species; Stokes settling is regime 1
        # of the droplet terminal velocity (one drag law)
        @test aerosol_species(:biological) === aerosol_species(:bioaerosol)
        @test stokes_settling_velocity(2e-6, DEFAULT_MICRO.ρ_water) == terminal_velocity(2e-6)
        # Seinfeld & Pandis (2016) Table 9.4, unit density: 3.5e-3 cm/s at
        # 1 μm, 0.31 cm/s at 10 μm (3 %: their slip correction and 298 K)
        @test stokes_settling_velocity(0.5e-6, 1000.0) ≈ 3.5e-5 rtol = 0.03
        @test stokes_settling_velocity(5e-6, 1000.0) ≈ 3.1e-3 rtol = 0.03
    end

    @testset "Log-normal surface area and N(>r): analytic vs quadrature" begin
        worst = 0.0
        for σ in (1.4, 2.8), r_g in (5e-8, 2e-6)
            m = arg_mode(1e6, r_g, σ, 0.03; ρ_p = 2600.0)
            A = mode_surface_area(m)
            worst = max(worst, abs(A - 4π * _ref_moment_above(m, 2, 0.0)) / A)
            for rc in (0.3r_g, 3r_g)
                aref = 4π * _ref_moment_above(m, 2, rc); nref = _ref_moment_above(m, 0, rc)
                aref > 1e-12A && (worst = max(worst, abs(surface_area_above_radius(m, rc) - aref) / aref))
                nref > 1e-12m.N && (worst = max(worst, abs(number_above_radius(m, rc) - nref) / nref))
            end
        end
        @test worst < 1e-8
        m = arg_mode(1e6, 3e-7, 2.0, 0.03; ρ_p = 2600.0)
        @test surface_area_above_radius(m, m.r_g * exp(2log(2.0)^2)) ≈ mode_surface_area(m) / 2 rtol = 1e-12
        # population diagnostics: dry mass agrees with Phase 37, modal closure
        inp = inp_population([dust])
        @test inp_mass(inp; mode = 1) ≈ mode_mass(dust) * 1e-9 rtol = 1e-12
        @test inp_median_radius(inp, 1) ≈ 0.4e-6 rtol = 1e-12
        # density comes from the species (arg_mode's default is sulphate's)
        @test_throws ErrorException inp_population([arg_mode(1e5, 5e-7, 1.8, 0.03)], :mineral_dust)
    end

    @testset "Transport-free conservation of number and mass" begin
        # a random sequence of every source and sink: budgets close,
        # nothing negative, particles never move without their mass
        rng = MersenneTwister(43)
        worst = 0.0; bad = false
        inp = inp_population([lognormal_mode(:mineral_dust, 0.5, 0.3, 2.0),
                              lognormal_mode(:biological, 0.02, 1.0, 1.6)])
        env_N = [4e5, 1e4]; env_M = [inp_mass(inp; mode = i) * 0.8 for i in 1:2]
        for step in 1:48
            i = rand(rng, 1:2); op = mod1(step, 8)
            if op == 1
                inp_emit!(inp, i, rand(rng) * 1e4, rand(rng) * 1e-12)
            elseif op == 2
                inp_entrain!(inp, env_N, env_M, 0.1rand(rng))
            elseif op == 3
                inp_consume!(inp, i, rand(rng) * 0.2 * inp_number(inp; mode = i);
                             selection = rand(rng, (:largest, :random)))
            elseif op == 4
                inp_consume!(inp, i, r -> 1 - exp(-1e10 * rand(rng) * 4π * r^2))
            elseif op == 5
                inp_nucleate_singular!(inp, i, r -> 1 - exp(-10.0^(9 + 3rand(rng)) * 4π * r^2))
            elseif op == 6
                inp_release!(inp, 0.3rand(rng); mode = rand(rng, 0:2))
            elseif op == 7
                inp_scavenge!(inp, r -> 1e8 * r^2, 100rand(rng); population = :both)
            else
                inp_sediment!(inp, 50.0, 600rand(rng))
            end
            worst = max(worst, maximum(abs, values(inp_budget_residual(inp))))
            bad |= any(<(0), inp.N_avail) || any(<(0), inp.M_avail) ||
                   any(<(0), inp.N_cons) || any(<(0), inp.M_cons) ||
                   any((inp.N_cons .> 0) .!= (inp.M_cons .> 0)) ||
                   any((inp.N_avail .> 0) .!= (inp.M_avail .> 0))
        end
        @test worst < 1e-13
        @test !bad
    end

    @testset "An INP cannot nucleate twice" begin
        inp = inp_population([dust])
        N0 = inp_number(inp)
        p(ns) = r -> 1 - exp(-ns * 4π * r^2)
        inp_nucleate_singular!(inp, 1, p(1e10))
        @test inp_nucleate_singular!(inp, 1, p(1e10)) == 0.0       # same T: nothing
        Δ2 = inp_nucleate_singular!(inp, 1, p(1e11))               # colder: increment
        @test inp_number(inp; population = :consumed) ≈
              first(inp_frozen_fraction(inp, 1, p(1e11))) * N0 rtol = 1e-12
        @test Δ2 > 0
        # released INP re-freeze, exactly the released number
        rel = inp_release!(inp, 0.5)
        @test inp_nucleate_singular!(inp, 1, p(1e11)) ≈ rel rtol = 1e-10
        # exhaustion: never more than was available
        inp = inp_population([dust])
        @test sum(inp_consume!(inp, 1, 0.6N0) for _ in 1:3) ≈ N0 rtol = 1e-14
        # INAS small-n_s limit: frozen number → n_s × surface area
        # (Hoose & Möhler 2012)
        inp = inp_population([dust])
        fN, _ = inp_frozen_fraction(inp, 1, p(1e2))
        @test fN * N0 ≈ 1e2 * inp_surface_area(inp) rtol = 1e-6
    end

    @testset "Scavenging and sedimentation" begin
        a = inp_population([dust])
        inp_scavenge!(a, 2e-4, 3600.0)
        @test inp_number(a) ≈ 1e6 * exp(-2e-4 * 3600) rtol = 1e-12      # exact exponential
        # moment-weighted Stokes speeds: v_k ∝ r_g² exp(2(k... )ln²σ); mass/number = e^{6 ln²σ}
        ns = MicrophysicalConstants(λ_air = 0.0)
        inp = inp_population([dust])
        @test inp_settling_velocity(inp, 1; moment = 3, micro = ns) /
              inp_settling_velocity(inp, 1; micro = ns) ≈ exp(6log(2.0)^2) rtol = 1e-10
        inp_sediment!(inp, 1e-6, 1e12)            # absurd step: empties, stays ≥ 0
        @test inp_number(inp) >= 0 && inp_mass(inp) >= 0
        @test maximum(abs, values(inp_budget_residual(inp))) < 1e-14
    end
end
