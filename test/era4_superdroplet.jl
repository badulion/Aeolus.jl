# Era IV, Phase 41 — superdroplet Lagrangian microphysics.
#
# Headline validation: the stochastic all-or-nothing coalescence algorithm
# against GOLOVIN'S ANALYTIC SOLUTION for the additive kernel. That is a
# hard external check — a closed form the solver knows nothing about — and
# it is the only test here that can catch a subtly wrong multiplicity rule.

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

    @testset "Reproducibility" begin
        function run(seed)
            rng = MersenneTwister(seed)
            sd = init_exponential_superdroplets(rng, 1024, 2.0^23, 30.531e-6, 1e6)
            K = GolovinKernel()
            for _ in 1:200; coalescence_step!(rng, sd, K, 1.0, 1e6); end
            return copy(sd.radius), copy(sd.ξ)
        end
        r1, x1 = run(123); r2, x2 = run(123); r3, x3 = run(124)
        @test r1 == r2 && x1 == x2        # bitwise identical
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
