# Phase 41 — superdroplet Lagrangian microphysics.
#
# The headline validation, the stochastic all-or-nothing coalescence
# algorithm against GOLOVIN'S ANALYTIC SOLUTION for the additive kernel,
# and its convergence with the superdroplet count, is
# validation/superdroplet.jl. Here: the closed form itself and the
# unit properties (Köhler, kernels, exact mass conservation, condensation).

using Test, Random, Statistics
using Aeolus

@testset "Superdroplet microphysics (Phase 41)" begin

    @testset "Köhler delegates to the Phase 39 contract" begin
        # parameterised by SOLUTE MASS here, by dry radius and κ in the
        # contract: the two must agree to round-off.
        a, rd = AMMONIUM_SULFATE, 1e-7
        ms = dry_mass(rd, a)
        @test dry_radius(ms, a) ≈ rd rtol = 1e-12
        @test droplet_critical_radius(ms, 293.15, a) ≈
              first(kohler_critical_point(rd, a.κ, 293.15)) rtol = 1e-12
        @test droplet_critical_supersaturation(ms, 293.15, a) ≈
              last(kohler_critical_point(rd, a.κ, 293.15)) rtol = 1e-12
        @test raoult_b(ms, a) ≈ solute_from_kappa(rd, a.κ) rtol = 1e-12
    end

    @testset "Collision kernels" begin
        r1, r2 = 10e-6, 20e-6
        v1 = 4π/3 * r1^3; v2 = 4π/3 * r2^3
        # the additive kernel is exactly b(v1+v2)
        @test collision_kernel(GolovinKernel(b_mass = 1.5), r1, r2) ≈ 1.5 * 1000.0 * (v1 + v2) rtol = 1e-12
        # a kernel is a property of the PAIR
        Kh = HydrodynamicKernel(LongEfficiency())
        @test collision_kernel(Kh, r1, r2) ≈ collision_kernel(Kh, r2, r1) rtol = 1e-12
        # equal drops never collide gravitationally (no differential fall speed)
        @test collision_kernel(GravitationalKernel(), 15e-6, 15e-6) ≈ 0 atol = 1e-30
        @test all(diff([terminal_velocity(r) for r in (5e-6, 20e-6, 100e-6, 500e-6)]) .> 0)
    end

    @testset "Coalescence conserves mass exactly" begin
        rng = MersenneTwister(7)
        V = 1e6
        sd = init_exponential_superdroplets(rng, 2048, 2.0^23, 30.531e-6, V)
        @test total_number(sd) / V ≈ 2.0^23 rtol = 0.05     # requested concentration
        m0 = total_water_mass(sd); n0 = total_number(sd)
        K = GolovinKernel(b_mass = 1.5)
        for _ in 1:100
            coalescence_step!(rng, sd, K, 1.0, V)
        end
        @test total_water_mass(sd) ≈ m0 rtol = 1e-12
        @test total_number(sd) < n0
    end

    @testset "Golovin closed form: initial exponential, mass conserved" begin
        # (The Monte-Carlo coalescence against this solution is the
        # validation run validation/superdroplet.jl.)
        n0 = 2.0^23; r̄ = 30.531e-6; b = 1.5
        x̄ = (4π/3) * 1000.0 * r̄^3
        xr(r) = (4π/3) * 1000.0 * r^3
        @test maximum(abs(golovin_mass_density(r, 0.0, n0, r̄, b) /
                          (3xr(r)^2 * n0 / x̄ * exp(-xr(r) / x̄)) - 1)
                      for r in (5e-6, 30.531e-6, 80e-6)) < 1e-12
        # ∫ g d(ln r) = n₀ x̄ at every time (trapezoid, 20001 points, g → 0
        # at both ends; measured |error| ≤ 2e-12)
        lr = range(log(1e-7), log(1e-1); length = 20001); h = step(lr)
        g = [golovin_mass_density(exp(l), 3600.0, n0, r̄, b) for l in lr]
        @test h * (sum(g) - 0.5 * (g[1] + g[end])) ≈ n0 * x̄ rtol = 1e-9
    end

    @testset "Condensational growth narrows the spectrum" begin
        # r dr/dt = const ⇒ small droplets gain radius faster: the relative
        # width shrinks (why condensation alone cannot make rain).
        V = 1.0
        sd = init_monodisperse_superdroplets(200, 1e8, 5e-6, V)
        for k in eachindex(sd.radius)
            sd.radius[k] *= 1 + 0.4 * (k / length(sd.radius) - 0.5)
        end
        σ0 = std(sd.radius) / mean(sd.radius)
        cell = EulerianCell(temperature = 283.0, q_vap = 9.6e-3, density = 1.1, volume = V)
        @test cell_supersaturation(cell) > 0
        for _ in 1:50
            condensation_step!(sd, cell, 0.5)
        end
        @test std(sd.radius) / mean(sd.radius) < σ0
    end

    @testset "Pure-water droplet evaporates, not vanishes (regression)" begin
        # The evaporating branch used to return the 1e-12 m floor at once.
        # Backward Euler on r² with the Kelvin term at the new radius.
        T, p = 283.0, 9e4
        G = 1 / (fk_thermal(T) + fd_vapor(T, p))
        r0, S, dt = 10e-6, -1e-3, 0.5
        r = grow_droplet(r0, 0.0, S, T, p, dt, NACL)
        @test r^2 ≈ r0^2 + 2G * (S - kelvin_a(T) / r) * dt rtol = 1e-8
        @test grow_droplet(0.5e-6, 0.0, -5e-2, T, p, 10.0, NACL) <= 1e-11
    end

    @testset "Diagnostics (monodisperse)" begin
        sd = init_monodisperse_superdroplets(100, 1e8, 10e-6, 1.0)
        @test liquid_water_content(sd, 1.0) ≈ 1e8 * (4π/3) * 1000.0 * (10e-6)^3 rtol = 1e-6
        @test effective_radius(sd) ≈ 10e-6 rtol = 1e-9
        @test relative_dispersion(sd)[2] ≈ 0 atol = 1e-12
    end
end
