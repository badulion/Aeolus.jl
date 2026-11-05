# Era IV, Phase 41 — superdroplet Lagrangian microphysics.
#
# The headline validation, the stochastic all-or-nothing coalescence
# algorithm against GOLOVIN'S ANALYTIC SOLUTION for the additive kernel,
# and its convergence with the superdroplet count, is
# validation/era4_superdroplet.jl. Here: the closed form itself and the
# unit properties (Köhler, kernels, exact mass conservation, condensation).

using Test, Printf, Random, Statistics
using Aeolus

@testset "Era IV — Superdroplet microphysics (Phase 41)" begin

    @testset "Köhler delegates to the Phase 39 contract" begin
        # This module is parameterised by SOLUTE MASS (what a superdroplet
        # carries); the contract by dry radius and κ. Same physics, and the
        # test is that they agree, not that they merely look similar.
        for (a, rd) in ((NACL, 5e-8), (AMMONIUM_SULFATE, 1e-7),
                        (SEASALT_KAPPA, 2e-7))
            ms = dry_mass(rd, a)
            @test dry_radius(ms, a) ≈ rd rtol = 1e-12
            @test droplet_critical_radius(ms, 293.15, a) ≈
                  first(kohler_critical_point(rd, a.κ, 293.15)) rtol = 1e-12
            @test droplet_critical_supersaturation(ms, 293.15, a) ≈
                  last(kohler_critical_point(rd, a.κ, 293.15)) rtol = 1e-12
            @test raoult_b(ms, a) ≈ solute_from_kappa(rd, a.κ) rtol = 1e-12
        end
        # solute-mass scaling: s_crit ∝ m_s^(-1/2)
        ms = dry_mass(1e-7, NACL)
        s1 = droplet_critical_supersaturation(ms, 293.15, NACL)
        s2 = droplet_critical_supersaturation(10ms, 293.15, NACL)
        @test s1 / s2 ≈ sqrt(10) rtol = 0.05
        @info "solute scaling" s_crit_1x=s1 s_crit_10x=s2 ratio=s1/s2
    end

    @testset "Initialisation and bookkeeping" begin
        rng = MersenneTwister(1)
        V = 1e6
        sd = init_exponential_superdroplets(rng, 4096, 2.0^23, 30.531e-6, V)
        @test nsuperdroplets(sd) == 4096
        @test all(sd.ξ .> 0)
        @test all(sd.radius .> 0)
        # the ensemble must reproduce the requested number concentration
        @test total_number(sd) / V ≈ 2.0^23 rtol = 0.05
        @test total_water_mass(sd) > 0
        # monodisperse variant
        sd2 = init_monodisperse_superdroplets(100, 1e8, 10e-6, V)
        @test all(sd2.radius .≈ 10e-6)
    end

    @testset "Collision kernels" begin
        K = GolovinKernel(b_mass = 1.5)
        # the additive kernel is exactly b(v1+v2)
        r1, r2 = 10e-6, 20e-6
        v1 = 4π/3 * r1^3; v2 = 4π/3 * r2^3
        @test collision_kernel(K, r1, r2) ≈ 1.5 * 1000.0 * (v1 + v2) rtol = 1e-12
        # symmetry: a kernel is a property of the PAIR
        for Kn in (GolovinKernel(), GravitationalKernel(),
                   HydrodynamicKernel(LongEfficiency()))
            @test collision_kernel(Kn, r1, r2) ≈ collision_kernel(Kn, r2, r1) rtol=1e-12
            @test collision_kernel(Kn, r1, r2) >= 0
        end
        # equal-sized drops cannot collide gravitationally — no differential
        # fall speed — which is why pure sedimentation never starts rain
        @test collision_kernel(GravitationalKernel(), 15e-6, 15e-6) ≈ 0 atol = 1e-30
        # terminal velocity rises with size
        vts = [terminal_velocity(r) for r in (5e-6, 20e-6, 100e-6, 500e-6)]
        @test all(diff(vts) .> 0)
        @info "terminal velocity (m/s)" r5um=vts[1] r20um=vts[2] r100um=vts[3] r500um=vts[4]
    end

    @testset "Coalescence conserves mass exactly" begin
        rng = MersenneTwister(7)
        V = 1e6
        sd = init_exponential_superdroplets(rng, 8192, 2.0^23, 30.531e-6, V)
        m0 = total_water_mass(sd)
        n0 = total_number(sd)
        K = GolovinKernel(b_mass = 1.5)
        for _ in 1:600
            coalescence_step!(rng, sd, K, 1.0, V)
        end
        # water is neither created nor destroyed by coalescence
        @test total_water_mass(sd) ≈ m0 rtol = 1e-12
        # but the number of REAL droplets strictly falls
        @test total_number(sd) < n0
        @info "600 s of coalescence" mass_drift=abs(total_water_mass(sd)-m0)/m0 number_ratio=total_number(sd)/n0
    end

    @testset "Golovin closed form: initial exponential, mass conserved" begin
        # (The Monte-Carlo coalescence against this solution is the
        # validation run validation/era4_superdroplet.jl.)
        n0 = 2.0^23; r̄ = 30.531e-6; b = 1.5
        x̄ = (4π/3) * 1000.0 * r̄^3
        # t = 0: the exponential n(x) = n₀/x̄ e^(−x/x̄), as g(ln r) = 3x² n(x)
        for r in (5e-6, 20e-6, 30.531e-6, 80e-6)
            x = (4π/3) * 1000.0 * r^3
            @test golovin_mass_density(r, 0.0, n0, r̄, b) ≈ 3x^2 * n0 / x̄ * exp(-x / x̄) rtol = 1e-12
        end
        # ∫ g d(ln r) = n₀ x̄ at every time: coalescence conserves mass.
        # Trapezoid rule on 20001 points over 0.1 μm - 10 cm, where g decays
        # to zero at both ends (measured |error| ≤ 2e-12).
        lr = range(log(1e-7), log(1e-1); length = 20001); h = step(lr)
        for t in (0.0, 1200.0, 3600.0)
            g = [golovin_mass_density(exp(l), t, n0, r̄, b) for l in lr]
            @test h * (sum(g) - 0.5 * (g[1] + g[end])) ≈ n0 * x̄ rtol = 1e-9
        end
    end

    @testset "Reproducibility" begin
        function run(seed)
            rng = MersenneTwister(seed)
            sd = init_exponential_superdroplets(rng, 1024, 2.0^23, 30.531e-6, 1e6)
            K = GolovinKernel()
            for _ in 1:200; coalescence_step!(rng, sd, K, 1.0, 1e6); end
            return copy(sd.radius), copy(sd.ξ)
        end
        r1, x1 = run(123); r2, x2 = run(123); r3, x3 = run(124)
        @test values_agree(r1, r2) && values_agree(x1, x2)        # the same
        @test !(r1 == r3)                  # a different seed really differs
    end

    @testset "Condensational growth narrows the spectrum" begin
        # Diffusional growth obeys r dr/dt = const, so dr/dt ∝ 1/r: small
        # droplets gain radius faster than large ones and the distribution
        # NARROWS. That is precisely why condensation alone cannot make
        # rain, and why collision-coalescence is needed.
        rng = MersenneTwister(5)
        V = 1.0
        sd = init_monodisperse_superdroplets(200, 1e8, 5e-6, V)
        # spread the radii a little so there is a width to narrow
        for k in eachindex(sd.radius)
            sd.radius[k] *= 1 + 0.4 * (k / length(sd.radius) - 0.5)
        end
        σ0 = std(sd.radius) / mean(sd.radius)
        # Must be SUPERsaturated or the droplets correctly evaporate:
        # q_sat at 283 K and ~893 hPa is 8.6e-3, so 8e-3 is subsaturated.
        cell = EulerianCell(temperature = 283.0, q_vap = 9.6e-3,
                            density = 1.1, volume = V)
        @test cell_supersaturation(cell) > 0
        for _ in 1:50
            condensation_step!(sd, cell, 0.5)
        end
        σ1 = std(sd.radius) / mean(sd.radius)
        @test mean(sd.radius) > 5e-6 * 0.9         # droplets exist
        @test σ1 < σ0                               # relative width shrank
        @info "condensational narrowing" rel_dispersion_before=σ0 after=σ1
    end

    @testset "Pure-water droplet evaporates, not vanishes" begin
        # Regression (found by the Phase 45 stochastic-growth tests): with
        # no solute, the Kelvin term makes the backward-Euler residual
        # positive at the radius floor for ANY subsaturation, and the
        # evaporating branch used to return the floor (1e-12 m) at once.
        # Away from the floor, r² must fall by ≈ 2 G |S| dt.
        T, p = 283.0, 9e4
        G = 1 / (fk_thermal(T) + fd_vapor(T, p))
        for (r0, S, dt) in ((10e-6, -1e-3, 0.5), (5e-6, -1e-2, 1.0),
                            (20e-6, -2e-3, 30.0))
            r = grow_droplet(r0, 0.0, S, T, p, dt, NACL)
            A = kelvin_a(T)
            # backward Euler on r² with the Kelvin term at the new radius
            @test r^2 ≈ r0^2 + 2G * (S - A / r) * dt rtol = 1e-8
            @test r > 0.9r0
        end
        # a droplet whose whole r² is gone within dt really does vanish
        @test grow_droplet(0.5e-6, 0.0, -5e-2, T, p, 10.0, NACL) <= 1e-11
    end

    @testset "Diagnostics" begin
        rng = MersenneTwister(3)
        V = 1.0
        sd = init_monodisperse_superdroplets(100, 1e8, 10e-6, V)
        @test number_concentration(sd, V) ≈ 1e8 rtol = 1e-6
        lwc = liquid_water_content(sd, V)
        @test lwc ≈ 1e8 * (4π/3) * 1000.0 * (10e-6)^3 rtol = 1e-6
        @test mean_radius(sd) ≈ 10e-6 rtol = 1e-9
        # for a monodisperse population every moment radius coincides
        @test effective_radius(sd) ≈ 10e-6 rtol = 1e-9
        @test relative_dispersion(sd)[2] ≈ 0 atol = 1e-12   # (σ, σ/r̄)
        @info "monodisperse diagnostics" N=number_concentration(sd,V) LWC=lwc r_eff=effective_radius(sd)
    end
end
