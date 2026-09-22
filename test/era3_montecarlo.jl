using Test
using Random
using Statistics
using Aeolus
using Aeolus.RadiationCore
using Aeolus.TwoStream
using Aeolus.MonteCarloRT

@testset "Era III — 3-D Monte-Carlo RT (Phase 34)" begin

    # A single-band shortwave grid with a clean solar source, so every
    # reference number below is analytic rather than spectrally weighted.
    unit_sw_grid(T = Float64; S = 1000.0) =
        SpectralGrid(:shortwave, T[4000.0], T[25000.0], [1], T[1.0], T[S])

    # ---------------------------------------------------------------------
    # Sampling routines. Every one takes an rng, and every one is checked
    # against a moment of the distribution it is supposed to reproduce.
    # ---------------------------------------------------------------------
    @testset "Free-path sampling" begin
        rng = MersenneTwister(2718)
        n = 200_000
        τ = [sample_free_path(rng, Float64) for _ in 1:n]
        @test all(>=(0), τ)
        @test all(isfinite, τ)
        # τ ~ Exp(1): mean 1, variance 1, so SE of the mean is 1/√n
        se = 1 / sqrt(n)
        @info "free path" mean_tau=mean(τ) var_tau=var(τ) se
        @test abs(mean(τ) - 1) < 4se
        @test abs(var(τ) - 1) < 0.02
        # survival fraction must be e^{-1}
        @test abs(count(>(1.0), τ) / n - exp(-1.0)) < 4 * sqrt(exp(-1) * (1 - exp(-1)) / n)
    end

    @testset "Henyey-Greenstein sampling" begin
        n = 400_000
        for g in (0.0, 0.5, 0.85, -0.3)
            rng = MersenneTwister(4242)
            μ = [sample_henyey_greenstein(rng, g) for _ in 1:n]
            @test all(x -> -1 <= x <= 1, μ)
            m = mean(μ)
            se = std(μ) / sqrt(n)
            @info "HG ⟨cosΘ⟩" g measured=m target=g se
            # the first moment of the HG phase function is exactly g
            @test abs(m - g) < 4se
        end
        # the g → 0 branch really is isotropic
        rng = MersenneTwister(1)
        μ0 = [sample_henyey_greenstein(rng, 0.0) for _ in 1:50_000]
        @test abs(mean(μ0 .^ 2) - 1 / 3) < 0.01      # ⟨μ²⟩ = 1/3 isotropic
        @test !any(isnan, μ0)
    end

    @testset "Rayleigh sampling" begin
        rng = MersenneTwister(97)
        n = 400_000
        μ = [sample_rayleigh(rng, Float64) for _ in 1:n]
        @test all(x -> -1 <= x <= 1, μ)
        m1 = mean(μ); m2 = mean(μ .^ 2)
        se1 = std(μ) / sqrt(n)
        se2 = std(μ .^ 2) / sqrt(n)
        @info "Rayleigh moments" mean_mu=m1 se1 mean_mu2=m2 se2
        @test abs(m1 - 0.0) < 4se1            # symmetric: ⟨cosΘ⟩ = 0
        @test abs(m2 - 0.4) < 4se2            # ⟨cos²Θ⟩ = 2/5 exactly
    end

    @testset "Phase-function normalisation" begin
        # ∫ P dΩ / 4π = ½∫₋₁¹ P dμ = 1 for both phase functions.
        nq = 20_001
        μs = range(-1, 1; length = nq)
        trap(f) = begin
            s = 0.0
            for i in 1:nq-1
                s += 0.5 * (f(μs[i]) + f(μs[i+1])) * (μs[i+1] - μs[i])
            end
            s / 2
        end
        @test abs(trap(rayleigh_phase) - 1) < 1e-6
        for g in (0.0, 0.3, 0.7)
            @test abs(trap(μ -> henyey_greenstein_phase(g, μ)) - 1) < 2e-4
        end
        # forward peaking for g > 0
        @test henyey_greenstein_phase(0.85, 1.0) > henyey_greenstein_phase(0.85, -1.0)
        @test henyey_greenstein_phase(0.0, 0.3) ≈ 1.0 atol = 1e-12
        @test rayleigh_phase(1.0) ≈ 1.5 atol = 1e-12
        @test rayleigh_phase(0.0) ≈ 0.75 atol = 1e-12
    end

    @testset "Scattering-frame rotation" begin
        rng = MersenneTwister(31415)
        # generic direction: the rotation must preserve the norm and realise
        # exactly the requested scattering angle
        worst_norm = 0.0; worst_angle = 0.0
        for _ in 1:20_000
            u = (randn(rng), randn(rng), randn(rng))
            n = sqrt(sum(abs2, u)); u = u ./ n
            μ = 2rand(rng) - 1
            φ = 2π * rand(rng)
            v = rotate_scatter(u, μ, φ)
            worst_norm = max(worst_norm, abs(sqrt(sum(abs2, v)) - 1))
            worst_angle = max(worst_angle, abs(sum(u .* v) - μ))
        end
        @info "rotate_scatter over 20000 random frames" worst_norm worst_angle
        @test worst_norm < 1e-12
        @test worst_angle < 1e-10
        # the |μ| → 1 special case: a vertically travelling photon. The
        # general formula divides by √(1-u_z²) = 0 here.
        for uz in (1.0, -1.0)
            for μ in (-0.9, 0.0, 0.37, 1.0)
                v = rotate_scatter(0.0, 0.0, uz, μ, 1.234)
                @test !any(isnan, v)
                @test isapprox(sqrt(sum(abs2, v)), 1.0; atol = 1e-12)
                @test isapprox(v[3], uz * μ; atol = 1e-12)
            end
        end
        # and just inside the branch, the answer must be continuous
        ε = 1e-7
        v1 = rotate_scatter(ε, 0.0, sqrt(1 - ε^2), 0.5, 0.9)
        v2 = rotate_scatter(0.0, 0.0, 1.0, 0.5, 0.9)
        @test maximum(abs, v1 .- v2) < 1e-5
    end

    @testset "Lambertian reflection sampling" begin
        rng = MersenneTwister(5)
        n = 200_000
        v = [sample_lambertian(rng, 0.0, 0.0, 1.0) for _ in 1:n]
        μ = [x[3] for x in v]
        @test all(>(0), μ)                       # never into the surface
        # cosine-weighted hemisphere: ⟨μ⟩ = 2/3, ⟨μ²⟩ = 1/2
        @test abs(mean(μ) - 2 / 3) < 4 * std(μ) / sqrt(n)
        @test abs(mean(μ .^ 2) - 0.5) < 0.005
        # about a wall normal the distribution is the same, just rotated
        w = [sample_lambertian(rng, 1.0, 0.0, 0.0) for _ in 1:20_000]
        @test all(x -> x[1] > 0, w)
        @test abs(mean(x -> x[1], w) - 2 / 3) < 0.01
    end

    # ---------------------------------------------------------------------
    # Geometry and medium plumbing.
    # ---------------------------------------------------------------------
    @testset "Domain and medium construction" begin
        d = mc_domain(Float64; size = (8, 4, 10), spacing = (10.0, 20.0, 5.0))
        @test size(d) == (8, 4, 10)
        @test domain_extent(d) == (80.0, 80.0, 50.0)
        @test domain_top(d) == 50.0

        med = homogeneous_medium(d; τ = 3.0, ssa = 0.9, g = 0.6)
        @test isapprox(optical_depth_column(med, 1, 1), 3.0; atol = 1e-12)
        @test all(med.ssa .== 0.9)
        @test med.phase === :henyey_greenstein
        @test_throws ErrorException mc_medium(d; ssa = 1.5)
        @test_throws ErrorException mc_medium(d; phase = :mie)

        # the RegularGrid / SolidMask bridge to the flow solver
        mesh = RegularGrid(Float64; size = (8, 4, 10), extent = (80.0, 80.0, 50.0))
        d2 = mc_domain(mesh)
        @test size(d2) == (8, 4, 10)
        @test d2.dx == mesh.dx && d2.dz == mesh.dz
        sm = SolidMask(mesh)
        mark_solid!(sm, (x, y, z) -> x < 20 && z < 15)
        med2 = mc_medium(d2)
        solid_from_mask!(med2, sm)
        @test med2.solid[1, 1, 1] && med2.solid[2, 3, 3]
        @test !med2.solid[3, 1, 1] && !med2.solid[1, 1, 4]
        @test count(med2.solid) == 2 * 4 * 3

        # the OpticalProperties bridge to the rest of the radiation stack
        sw = unit_sw_grid()
        opt = allocate_optics(Float64, (10,), sw)
        opt.τ .= 0.25; opt.ssa .= 0.8; opt.g .= 0.4
        med3 = medium_from_optics(d, opt, 1)
        @test isapprox(optical_depth_column(med3, 3, 2), 2.5; atol = 1e-12)
        @test all(med3.g .== 0.4)

        # sun direction conventions (east-north-up, pointing at the sun)
        @test all(isapprox.(mc_sun_direction(1.0), (0.0, 0.0, 1.0); atol = 1e-12))
        s = mc_sun_direction(0.5; azimuth = 90.0)
        @test isapprox(s[1], sqrt(0.75); atol = 1e-12)   # due east
        @test isapprox(s[2], 0.0; atol = 1e-12)
        @test isapprox(s[3], 0.5; atol = 1e-12)
    end

    # ---------------------------------------------------------------------
    # Beer-Lambert. The local estimator makes the unscattered beam exact,
    # so this is a machine-precision test, not a statistical one.
    # ---------------------------------------------------------------------
    @testset "Beer-Lambert direct beam" begin
        dom = mc_domain(Float64; size = (1, 1, 20), spacing = (100.0, 100.0, 100.0))
        solver = monte_carlo_solver(; nphoton = 4000, nbatch = 8)
        worst = 0.0
        for τ in (0.5, 2.0, 5.0), μ₀ in (1.0, 0.6, 0.35)
            med = homogeneous_medium(dom; τ = τ, ssa = 0.0, g = 0.0, sfc_albedo = 0.0)
            r = mc_solve(med, solver; rng = MersenneTwister(11),
                         sun_dir = μ₀, solar_flux = 1000.0)
            expect = exp(-τ / μ₀)
            worst = max(worst, abs(r.direct_transmittance - expect))
            @test isapprox(r.direct_transmittance, expect;
                           atol = max(4 * r.direct_transmittance_err, 1e-12))
            # a purely absorbing medium over a black surface reflects nothing
            @test r.albedo < 1e-12
            @test r.albedo_analogue == 0.0
        end
        @info "Beer-Lambert: worst |T_dir - exp(-τ/μ₀)|" worst
        @test worst < 1e-12

        # and the level-by-level direct profile follows the same law
        med = homogeneous_medium(dom; τ = 2.0, ssa = 0.0, g = 0.0)
        r = mc_solve(med, monte_carlo_solver(; nphoton = 2000, nbatch = 8);
                     rng = MersenneTwister(3), sun_dir = 0.8, solar_flux = 1000.0)
        wl = 0.0
        for kl in 1:21
            τ_above = 2.0 * (21 - kl) / 20
            wl = max(wl, abs(r.flux_dn_dir[1, 1, kl] - 800.0 * exp(-τ_above / 0.8)))
        end
        @info "Beer-Lambert profile: worst level error (W m⁻²)" wl
        @test wl < 1e-9
    end

    # ---------------------------------------------------------------------
    # Conservative scattering. Exact physics: nothing is absorbed, so
    # everything is either reflected to space or transmitted to the black
    # ground. This is the strongest single check on the weighting.
    # ---------------------------------------------------------------------
    @testset "Conservative scattering closes exactly" begin
        dom = mc_domain(Float64; size = (1, 1, 20), spacing = (100.0, 100.0, 100.0))
        solver = monte_carlo_solver(; nphoton = 24_000, nbatch = 16)
        worst = 0.0
        for (τ, g) in ((1.0, 0.0), (1.0, 0.85), (10.0, 0.0), (10.0, 0.85)),
            μ₀ in (1.0, 0.5)
            med = homogeneous_medium(dom; τ = τ, ssa = 1.0, g = g, sfc_albedo = 0.0)
            r = mc_solve(med, solver; rng = MersenneTwister(77),
                         sun_dir = μ₀, solar_flux = 1.0)
            s = r.albedo + r.transmittance
            se = sqrt(r.albedo_err^2 + r.transmittance_err^2)
            worst = max(worst, abs(s - 1))
            @test r.absorptance_atm == 0.0           # ω₀ = 1 ⇒ nothing absorbed
            @test isapprox(s, 1.0; atol = 4se)
            # the analogue estimator closes to machine precision
            @test isapprox(r.albedo_analogue + r.absorptance_sfc, 1.0; atol = 1e-12)
        end
        @info "conservative scattering: worst |α + T - 1|" worst
        @test worst < 0.01
    end

    # ---------------------------------------------------------------------
    # Energy closure — every photon's weight is accounted for exactly.
    # ---------------------------------------------------------------------
    @testset "Energy closure" begin
        dom = mc_domain(Float64; size = (8, 8, 10), spacing = (25.0, 25.0, 25.0))
        solver = monte_carlo_solver(; nphoton = 16_000, nbatch = 16)
        worst = 0.0
        # clear, cloudy, reflecting surface, and with buildings
        cases = Any[]
        push!(cases, ("clear black", mc_medium(dom; kext = 0.0, sfc_albedo = 0.0)))
        push!(cases, ("absorbing", mc_medium(dom; kext = 2e-3, ssa = 0.5, g = 0.3,
                                             sfc_albedo = 0.0)))
        push!(cases, ("bright surface", mc_medium(dom; kext = 4e-3, ssa = 0.95,
                                                  g = 0.7, sfc_albedo = 0.8)))
        let m = mc_medium(dom; kext = 2e-3, ssa = 0.9, g = 0.5, sfc_albedo = 0.2,
                          solid_albedo = 0.25)
            add_box_building!(m, 3:5, 3:5, 6)
            push!(cases, ("buildings", m))
        end
        for (name, med) in cases
            for μ₀ in (0.9, 0.4)
                r = mc_solve(med, solver; rng = MersenneTwister(9),
                             sun_dir = mc_sun_direction(μ₀; azimuth = 135.0),
                             solar_flux = 1000.0)
                b = mc_energy_budget(r)
                res = mc_closure_residual(r)
                worst = max(worst, abs(res))
                @test abs(res) < 1e-10
                @test b.total ≈ 1.0 atol = 1e-10
                @test all(x -> x >= 0, (b.albedo, b.atmosphere, b.surface, b.buildings))
                if name == "buildings"
                    @test b.buildings > 0.0
                    # absorbed field is zero inside solid voxels
                    @test all(r.absorbed[med.solid] .== 0)
                    @test sum(r.solid_incident) > sum(r.solid_absorbed) > 0
                else
                    @test b.buildings == 0.0
                end
            end
        end
        @info "energy closure: worst |1 - Σ sinks|" worst

        # the per-voxel absorption field integrates to the bulk absorptance
        med = mc_medium(dom; kext = 4e-3, ssa = 0.7, g = 0.4, sfc_albedo = 0.1)
        r = mc_solve(med, solver; rng = MersenneTwister(13), sun_dir = 0.7,
                     solar_flux = 1000.0)
        vol = dom.dx * dom.dy * dom.dz
        A = (dom.nx * dom.dx) * (dom.ny * dom.dy)
        @test isapprox(sum(r.absorbed) * vol / (r.incident * A), r.absorptance_atm;
                       atol = 1e-10)
        # heating rate is just that divided by ρcp
        hr = mc_heating_rate(r, 1.1)
        @test isapprox(maximum(hr), maximum(r.absorbed) / (1.1 * 1004.6); atol = 1e-12)
        @test all(>=(0), hr)
        @info "column absorptance and peak heating" r.absorptance_atm maximum(hr)*86400
    end

    # ---------------------------------------------------------------------
    # THE headline validation: the 1-D homogeneous limit must reproduce the
    # delta-Eddington two-stream result. The two-stream answer is itself an
    # approximation, so the agreement expected is "a few percent", tighter
    # for conservative scattering; the measured discrepancies are printed.
    # ---------------------------------------------------------------------
    @testset "1-D limit versus two-stream" begin
        sw = unit_sw_grid(; S = 1000.0)
        ts = two_stream_solver(sw)
        nz = 20
        dom = mc_domain(Float64; size = (1, 1, nz), spacing = (100.0, 100.0, 100.0))
        solver = monte_carlo_solver(; nphoton = 24_000, nbatch = 16)

        function two_stream_ref(τ, ssa, g, μ₀)
            opt = allocate_optics(Float64, (nz,), sw)
            opt.τ .= τ / nz; opt.ssa .= ssa; opt.g .= g
            fl = allocate_fluxes(Float64, (), nz + 1)
            solve_sw!(fl, ts, opt; μ₀ = μ₀, sfc_albedo_direct = 0.0)
            inc = 1000.0 * μ₀
            return (fl.flux_up[end] / inc, fl.flux_dn[1] / inc)
        end

        rows = String[]
        worst_a = 0.0; worst_t = 0.0; worst_rel = 0.0
        for (τ, ssa, g) in ((1.0, 1.0, 0.0), (1.0, 1.0, 0.85),
                            (10.0, 1.0, 0.0), (10.0, 1.0, 0.85),
                            (1.0, 0.9, 0.0), (1.0, 0.9, 0.85),
                            (10.0, 0.9, 0.0), (10.0, 0.9, 0.85)),
            μ₀ in (1.0, 0.5)

            med = homogeneous_medium(dom; τ = τ, ssa = ssa, g = g, sfc_albedo = 0.0)
            r = mc_solve(med, solver; rng = MersenneTwister(2024),
                         sun_dir = μ₀, solar_flux = 1000.0)
            a2, t2 = two_stream_ref(τ, ssa, g, μ₀)
            da = r.albedo - a2
            dt = r.transmittance - t2
            worst_a = max(worst_a, abs(da)); worst_t = max(worst_t, abs(dt))
            a2 > 0.05 && (worst_rel = max(worst_rel, abs(da) / a2))

            push!(rows, string(
                "τ=", rpad(τ, 5), " ω=", rpad(ssa, 4), " g=", rpad(g, 5),
                " μ₀=", rpad(μ₀, 4),
                " | MC α=", rpad(round(r.albedo; digits = 4), 7),
                "±", rpad(round(r.albedo_err; digits = 4), 6),
                " 2S α=", rpad(round(a2; digits = 4), 7),
                " Δ=", rpad(round(da; digits = 4), 8),
                " | MC T=", rpad(round(r.transmittance; digits = 4), 7),
                "±", rpad(round(r.transmittance_err; digits = 4), 6),
                " 2S T=", rpad(round(t2; digits = 4), 7),
                " Δ=", round(dt; digits = 4)))

            # conservative scattering: delta-Eddington is at its best
            tol = ssa == 1.0 ? max(4 * r.albedo_err, 0.025) :
                               max(5 * r.albedo_err, 0.03)
            @test isapprox(r.albedo, a2; atol = tol)
            toltr = ssa == 1.0 ? max(4 * r.transmittance_err, 0.025) :
                                 max(5 * r.transmittance_err, 0.03)
            @test isapprox(r.transmittance, t2; atol = toltr)
        end
        @info "Monte-Carlo vs delta-Eddington two-stream\n" * join(rows, "\n")
        @info "worst discrepancies" worst_albedo=worst_a worst_transmittance=worst_t worst_relative_albedo=worst_rel
        @test worst_a < 0.025
        @test worst_t < 0.03
    end

    # ---------------------------------------------------------------------
    # Full `solve_rt!` contract: the Monte-Carlo solver must be drop-in
    # compatible with the two-stream solver on the same optics.
    # ---------------------------------------------------------------------
    @testset "solve_rt! contract" begin
        sw = unit_sw_grid(; S = 1000.0)
        nz = 20
        opt = allocate_optics(Float64, (nz,), sw)
        opt.τ .= 3.0 / nz; opt.ssa .= 0.95; opt.g .= 0.6
        col = AtmosphericColumn(Float64, nz; dz = fill(100.0, nz), ρ = fill(1.0, nz))

        mcs = monte_carlo_solver(sw; nphoton = 16_000, nbatch = 16)
        fl = allocate_fluxes(Float64, (), nz + 1)
        solve_rt!(fl, mcs, opt, col; rng = MersenneTwister(9), μ₀ = 0.7,
                  sfc_albedo = 0.2)

        ts = two_stream_solver(sw)
        fl2 = allocate_fluxes(Float64, (), nz + 1)
        solve_sw!(fl2, ts, opt; μ₀ = 0.7, sfc_albedo_direct = 0.2)

        @test length(fl.flux_up) == nz + 1
        @test all(isfinite, fl.flux_up) && all(isfinite, fl.flux_dn)
        @test maximum(abs, fl.flux_net .- (fl.flux_dn .- fl.flux_up)) == 0.0
        @test fl.flux_dn[end] ≈ 700.0 atol = 1e-9        # F₀μ₀ at the top
        @test fl.flux_dn_dir[end] ≈ 700.0 atol = 1e-9
        # the *true* unscattered beam at the surface, which delta-Eddington
        # deliberately does not give (it folds the forward peak into it)
        @test fl.flux_dn_dir[1] ≈ 700.0 * exp(-3.0 / 0.7) atol = 1e-9
        @test fl2.flux_dn_dir[1] > 3 * fl.flux_dn_dir[1]

        rel_up = abs(fl.flux_up[end] - fl2.flux_up[end]) / fl2.flux_up[end]
        rel_dn = abs(fl.flux_dn[1] - fl2.flux_dn[1]) / fl2.flux_dn[1]
        @info "solve_rt! vs two-stream" TOA_up_MC=fl.flux_up[end] TOA_up_2S=fl2.flux_up[end] rel_up sfc_dn_MC=fl.flux_dn[1] sfc_dn_2S=fl2.flux_dn[1] rel_dn
        @test rel_up < 0.05
        @test rel_dn < 0.05
        @test all(diff(fl.flux_dn_dir) .>= -1e-12)       # beam grows upward

        # heating rate from the flux divergence, and its column integral
        hr = zeros(nz)
        heating_rate!(hr, fl, 100.0, 1.0)
        @test all(isfinite, hr)
        col_int = sum(hr) * 1.0 * 1004.6 * 100.0
        expect = (fl.flux_dn[1] - fl.flux_up[1]) - (fl.flux_dn[end] - fl.flux_up[end])
        @test isapprox(col_int, -expect; rtol = 1e-9)

        # a solver built without a spectral grid cannot serve the contract
        bare = monte_carlo_solver(; nphoton = 100, nbatch = 2)
        @test_throws ErrorException solve_rt!(fl, bare, opt, col;
                                              rng = MersenneTwister(1))
        # non-uniform layers are rejected rather than silently mangled
        badcol = AtmosphericColumn(Float64, nz;
                                   dz = [100.0 + 5i for i in 1:nz], ρ = ones(nz))
        @test_throws ErrorException solve_rt!(fl, mcs, opt, badcol;
                                              rng = MersenneTwister(1))
    end

    # ---------------------------------------------------------------------
    # Convergence: the standard error must fall as N^{-1/2}. The batch
    # variance estimate is itself noisy (≈1/√(2(nb-1)) relative), so the
    # comparison is made on the mean standard error over several seeds.
    # ---------------------------------------------------------------------
    @testset "Convergence as 1/√N" begin
        dom = mc_domain(Float64; size = (1, 1, 20), spacing = (100.0, 100.0, 100.0))
        med = homogeneous_medium(dom; τ = 2.0, ssa = 0.9, g = 0.5, sfc_albedo = 0.1)
        function mean_se(N, seeds)
            a = Float64[]; t = Float64[]; v = Float64[]
            for s in seeds
                solver = monte_carlo_solver(; nphoton = N, nbatch = 32)
                r = mc_solve(med, solver; rng = MersenneTwister(s),
                             sun_dir = 0.7, solar_flux = 1.0)
                push!(a, r.albedo_err); push!(t, r.transmittance_err)
                push!(v, r.absorptance_atm_err)
            end
            return (mean(a), mean(t), mean(v))
        end
        a1, t1, v1 = mean_se(6000, 1:6)
        a4, t4, v4 = mean_se(24_000, 101:106)
        @info "SE scaling (N → 4N, expect ratio 2)" albedo=a1/a4 transmittance=t1/t4 absorptance=v1/v4 se_N=a1 se_4N=a4
        for ratio in (a1 / a4, t1 / t4, v1 / v4)
            @test 1.6 < ratio < 2.5
        end
        # and the error really does shrink
        @test a4 < a1 && t4 < t1 && v4 < v1
    end

    # ---------------------------------------------------------------------
    # Variance reduction: the local estimator and the analogue counter are
    # two unbiased estimators of the same quantity. They must agree, and
    # the local estimate of the *direct* beam must be essentially exact.
    # ---------------------------------------------------------------------
    @testset "Local estimation is unbiased and sharper" begin
        dom = mc_domain(Float64; size = (1, 1, 20), spacing = (100.0, 100.0, 100.0))
        for (τ, ssa, g) in ((0.3, 0.8, 0.0), (1.0, 1.0, 0.0), (5.0, 0.9, 0.85))
            med = homogeneous_medium(dom; τ = τ, ssa = ssa, g = g, sfc_albedo = 0.0)
            solver = monte_carlo_solver(; nphoton = 32_000, nbatch = 32)
            r = mc_solve(med, solver; rng = MersenneTwister(21), sun_dir = 0.7,
                         solar_flux = 1.0)
            dα = abs(r.albedo - r.albedo_analogue)
            seα = sqrt(r.albedo_err^2 + r.albedo_analogue_err^2)
            dT = abs(r.transmittance - r.transmittance_analogue)
            seT = sqrt(r.transmittance_err^2 + r.transmittance_analogue_err^2)
            @info "LE vs analogue" τ ssa g albedo_LE=r.albedo albedo_ana=r.albedo_analogue dα seα trans_LE=r.transmittance trans_ana=r.transmittance_analogue
            @test dα < 4seα
            @test dT < 4seT
        end
        # the direct beam: local estimation removes the variance entirely
        med = homogeneous_medium(dom; τ = 1.0, ssa = 0.0, g = 0.0, sfc_albedo = 0.0)
        solver = monte_carlo_solver(; nphoton = 16_000, nbatch = 32)
        r = mc_solve(med, solver; rng = MersenneTwister(2), sun_dir = 0.6,
                     solar_flux = 1.0)
        binom = sqrt(exp(-1 / 0.6) * (1 - exp(-1 / 0.6)) / 16_000)
        @info "direct beam variance reduction" le_se=r.direct_transmittance_err analogue_se=r.transmittance_analogue_err binomial_se=binom factor=binom/max(r.direct_transmittance_err, eps())
        @test r.direct_transmittance_err < 1e-6
        @test r.direct_transmittance_err < binom / 1000
        @test isapprox(r.transmittance_analogue, r.direct_transmittance;
                       atol = 4 * r.transmittance_analogue_err)
    end

    # ---------------------------------------------------------------------
    # Top-of-domain radiance by next-event estimation.
    # ---------------------------------------------------------------------
    @testset "Radiance local estimator" begin
        dom = mc_domain(Float64; size = (1, 1, 20), spacing = (100.0, 100.0, 100.0))
        vals = Float64[]
        for τ in (1.0, 5.0)
            solver = monte_carlo_solver(; nphoton = 16_000, nbatch = 16,
                                        view_dirs = [(0.0, 0.0, 1.0)])
            med = homogeneous_medium(dom; τ = τ, ssa = 1.0, g = 0.0, sfc_albedo = 0.0)
            r = mc_solve(med, solver; rng = MersenneTwister(4), sun_dir = 0.8,
                         solar_flux = 1000.0)
            @test length(r.radiance_up) == 1
            @test r.radiance_up[1] > 0
            @test r.radiance_up_err[1] / r.radiance_up[1] < 0.05
            # a diffuse reflector cannot return more than its hemispheric mean
            # would allow: πL ≤ F_up for a forward-peaked-free conservative slab
            @test π * r.radiance_up[1] <= r.albedo * r.incident * 1.15
            push!(vals, r.radiance_up[1])
            @info "nadir radiance" τ L=r.radiance_up[1] err=r.radiance_up_err[1] piL_over_F=π*r.radiance_up[1]/(r.albedo*r.incident)
        end
        @test vals[2] > vals[1]           # thicker cloud is brighter
        # no viewing directions requested ⇒ no radiance, and no cost
        solver = monte_carlo_solver(; nphoton = 1000, nbatch = 10)
        med = homogeneous_medium(dom; τ = 1.0, ssa = 1.0, g = 0.0)
        r = mc_solve(med, solver; rng = MersenneTwister(1), sun_dir = 1.0)
        @test isempty(r.radiance_up)
    end

    # ---------------------------------------------------------------------
    # 3-D effects: an isolated cubic cloud. Independent-column theory is
    # applied to the *same* field with the *same* solver (one-column
    # domains), so the difference measured is purely the 3-D transport.
    # ---------------------------------------------------------------------
    @testset "3-D cloud versus the independent-column approximation" begin
        nx = ny = nz = 16
        dxs = 50.0
        dom = mc_domain(Float64; size = (nx, ny, nz), spacing = (dxs, dxs, dxs))
        kc = 20.0 / (6 * dxs)                      # τ = 20 through the cube
        kext = zeros(nx, ny, nz); ssa = zeros(nx, ny, nz); gg = zeros(nx, ny, nz)
        for k in 6:11, j in 6:11, i in 6:11
            kext[i, j, k] = kc; ssa[i, j, k] = 0.99; gg[i, j, k] = 0.85
        end
        frac = 36 / (nx * ny)
        med3 = mc_medium(dom; kext = kext, ssa = ssa, g = gg, sfc_albedo = 0.0)

        dcol = mc_domain(Float64; size = (1, 1, nz), spacing = (dxs, dxs, dxs))
        kc1 = zeros(1, 1, nz); wc1 = zeros(1, 1, nz); gc1 = zeros(1, 1, nz)
        for k in 6:11
            kc1[1, 1, k] = kc; wc1[1, 1, k] = 0.99; gc1[1, 1, k] = 0.85
        end
        medc = mc_medium(dcol; kext = kc1, ssa = wc1, g = gc1, sfc_albedo = 0.0)
        medk = mc_medium(dcol; kext = 0.0, sfc_albedo = 0.0)

        s3 = monte_carlo_solver(; nphoton = 32_000, nbatch = 16)
        s1 = monte_carlo_solver(; nphoton = 16_000, nbatch = 16)
        for μ₀ in (1.0, 0.5)
            sd = mc_sun_direction(μ₀; azimuth = 90.0)     # sun in the east
            r3 = mc_solve(med3, s3; rng = MersenneTwister(11), sun_dir = sd,
                          solar_flux = 1000.0)
            rc = mc_solve(medc, s1; rng = MersenneTwister(12), sun_dir = sd,
                          solar_flux = 1000.0)
            rk = mc_solve(medk, s1; rng = MersenneTwister(13), sun_dir = sd,
                          solar_flux = 1000.0)
            ica = frac * rc.albedo + (1 - frac) * rk.albedo
            ica_err = frac * rc.albedo_err
            se = sqrt(r3.albedo_err^2 + ica_err^2)
            @info "3-D vs ICA albedo" μ₀ albedo_3D=r3.albedo err_3D=r3.albedo_err albedo_ICA=ica err_ICA=ica_err difference=r3.albedo-ica relative_percent=100*(r3.albedo-ica)/ica n_sigma=(r3.albedo-ica)/se
            @test abs(r3.albedo - ica) > 5se          # a real, resolved 3-D effect
            @test abs(mc_closure_residual(r3)) < 1e-10
            if μ₀ == 1.0
                # overhead sun: photons leak out of the cloud sides, so the
                # finite cloud is *darker* than the plane-parallel columns
                @test r3.albedo < ica
            else
                # low sun: the illuminated side presents a larger cross
                # section than the vertical column, so the cloud brightens
                @test r3.albedo > ica
                # and the sun-facing (east) half of the cloud absorbs more
                east = sum(r3.absorbed[10:11, 6:11, 6:11])
                west = sum(r3.absorbed[6:7, 6:11, 6:11])
                @info "cloud-side illumination (low sun from the east)" east west ratio=east/west
                @test east > 2 * west
            end
        end

        # shadow displacement: with a compact cloud the darkest ground cell
        # must sit where geometry says the shadow falls, not under the cloud
        kext2 = zeros(nx, ny, nz); ssa2 = zeros(nx, ny, nz); gg2 = zeros(nx, ny, nz)
        for k in 9:12, j in 6:11, i in 6:11
            kext2[i, j, k] = 20.0 / (4 * dxs); ssa2[i, j, k] = 0.99; gg2[i, j, k] = 0.85
        end
        med4 = mc_medium(dom; kext = kext2, ssa = ssa2, g = gg2, sfc_albedo = 0.0)
        sd = mc_sun_direction(0.866; azimuth = 90.0)
        r4 = mc_solve(med4, s3; rng = MersenneTwister(3), sun_dir = sd,
                      solar_flux = 1000.0)
        prof = vec(sum(mc_surface_irradiance(r4); dims = 2)) ./ ny
        zc = (9 + 12 - 1) / 2 * dxs                     # cloud centroid height
        i_expect = mod(round(Int, 8.5 - (sd[1] / sd[3]) * zc / dxs) - 1, nx) + 1
        @info "shadow displacement" expected_i=i_expect measured_i=argmin(prof) under_cloud=mean(prof[8:9]) in_shadow=prof[i_expect]
        @test argmin(prof) == i_expect
        @test prof[i_expect] < 0.8 * mean(prof[8:9])    # the shadow is dark
        @test mean(prof[8:9]) > 0.9 * maximum(prof)     # under the cloud is bright
        @test prof[i_expect] > 0                        # but never black: diffuse light
    end

    # ---------------------------------------------------------------------
    # Urban geometry: a tall building, an opaque solid that shadows and
    # reflects rather than transmits.
    # ---------------------------------------------------------------------
    @testset "Buildings cast shadows in the right direction" begin
        nx = ny = 24; nz = 12
        dom = mc_domain(Float64; size = (nx, ny, nz), spacing = (5.0, 5.0, 5.0))
        med = mc_medium(dom; kext = 2e-3, ssa = 0.9, g = 0.0, sfc_albedo = 0.15,
                        solid_albedo = 0.3)
        add_box_building!(med, 11:13, 11:13, 8)          # a 40 m tower
        @test count(med.solid) == 3 * 3 * 8

        sd = mc_sun_direction(1 / sqrt(2); azimuth = 90.0)   # 45° from the east
        solver = monte_carlo_solver(; nphoton = 24_000, nbatch = 16)
        r = mc_solve(med, solver; rng = MersenneTwister(5), sun_dir = sd,
                     solar_flux = 1000.0)

        @test abs(mc_closure_residual(r)) < 1e-10
        @test r.absorptance_solid > 0.03

        row = r.sfc_direct[:, 12]
        dif = r.sfc_diffuse[:, 12]
        # sun in the east ⇒ photons travel west ⇒ the shadow lies west of
        # the tower: 40 m of height at 45° is exactly 8 cells of 5 m
        shadow = 3:10
        lit = vcat(1:2, 14:nx)
        @info "building shadow" direct_in_shadow=maximum(row[shadow]) direct_in_sun=minimum(row[lit]) diffuse_in_shadow=minimum(dif[shadow]) mean_diffuse_lit=mean(dif[lit])
        @test all(row[shadow] .== 0.0)                   # fully shadowed
        @test all(row[lit] .> 0.9 * 1000 / sqrt(2) * 0.9)
        @test all(dif[shadow] .> 0.0)                    # but lit diffusely
        @test minimum(dif[shadow]) > 0.01 * mean(row[lit])
        # the footprint itself receives nothing — there is no ground there
        @test all(r.sfc_direct[11:13, 11:13] .== 0.0)
        @test all(r.sfc_diffuse[11:13, 11:13] .== 0.0)

        # the east (sun-facing) wall is brighter than the west wall
        east_wall = sum(r.solid_incident[13, 11:13, 1:8])
        west_wall = sum(r.solid_incident[11, 11:13, 1:8])
        @info "wall irradiance" east_wall west_wall
        @test east_wall > west_wall

        # opaque means opaque: no photon may be scored inside the block
        @test all(r.absorbed[11:13, 11:13, 1:8] .== 0.0)

        # rotating the sun rotates the shadow
        sd2 = mc_sun_direction(1 / sqrt(2); azimuth = 180.0)  # from the south
        r2 = mc_solve(med, solver; rng = MersenneTwister(5), sun_dir = sd2,
                      solar_flux = 1000.0)
        # azimuth 180° is the sun due south, so the photons travel north and
        # the shadow falls on the far (northern) side of the block
        @test all(r2.sfc_direct[12, 15:20] .== 0.0)
        @test all(r2.sfc_direct[12, 3:10] .> 0.0)
        @test r2.sfc_direct[3, 12] > 0.0                     # the old shadow is lit
    end

    # ---------------------------------------------------------------------
    # Determinism.
    # ---------------------------------------------------------------------
    @testset "Reproducibility" begin
        dom = mc_domain(Float64; size = (6, 6, 8), spacing = (20.0, 20.0, 20.0))
        med = mc_medium(dom; kext = 3e-3, ssa = 0.9, g = 0.6, sfc_albedo = 0.25)
        add_box_building!(med, 2:3, 4:5, 4)
        solver = monte_carlo_solver(; nphoton = 8000, nbatch = 8,
                                    view_dirs = [(0.0, 0.0, 1.0)])
        a = mc_solve(med, solver; rng = MersenneTwister(31337), sun_dir = 0.65,
                     solar_flux = 800.0)
        b = mc_solve(med, solver; rng = MersenneTwister(31337), sun_dir = 0.65,
                     solar_flux = 800.0)
        @test a.albedo === b.albedo
        @test a.albedo_err === b.albedo_err
        @test a.absorbed == b.absorbed
        @test a.sfc_direct == b.sfc_direct
        @test a.sfc_diffuse == b.sfc_diffuse
        @test a.flux_up == b.flux_up
        @test a.radiance_up == b.radiance_up
        # a different seed must actually change the answer
        c = mc_solve(med, solver; rng = MersenneTwister(4242), sun_dir = 0.65,
                     solar_flux = 800.0)
        @test c.albedo != a.albedo
        @test isapprox(c.albedo, a.albedo;
                       atol = 5 * sqrt(a.albedo_err^2 + c.albedo_err^2))

        # unstratified launching is a different estimator, same expectation
        solver2 = monte_carlo_solver(; nphoton = 8000, nbatch = 8, stratify = false)
        d = mc_solve(med, solver2; rng = MersenneTwister(31337), sun_dir = 0.65,
                     solar_flux = 800.0)
        @test isapprox(d.albedo, a.albedo;
                       atol = 5 * sqrt(a.albedo_err^2 + d.albedo_err^2))
        @test abs(mc_closure_residual(d)) < 1e-10

        # the sun has to be up
        @test_throws ErrorException mc_solve(med, solver;
                                             rng = MersenneTwister(1),
                                             sun_dir = (1.0, 0.0, -0.5))
    end
end
