using Test
using Random
using Statistics
using Aeolus
using Aeolus.RadiationCore
using Aeolus.TwoStream
using Aeolus.MonteCarloRT

@testset "3-D Monte-Carlo RT (Phase 34)" begin

    unit_sw_grid(T = Float64; S = 1000.0) =
        SpectralGrid(:shortwave, T[4000.0], T[25000.0], [1], T[1.0], T[S])

    # Each sampler reproduces a moment of its distribution (4σ sampling bands).
    @testset "Sampling routines" begin
        rng = MersenneTwister(2718)
        n = 50_000
        τ = [sample_free_path(rng, Float64) for _ in 1:n]
        @test all(>=(0), τ) && abs(mean(τ) - 1) < 4 / sqrt(n)               # Exp(1)
        μ = [sample_henyey_greenstein(rng, 0.85) for _ in 1:n]
        @test all(x -> -1 <= x <= 1, μ) && abs(mean(μ) - 0.85) < 4std(μ) / sqrt(n)  # ⟨cosΘ⟩ = g
        μr = [sample_rayleigh(rng, Float64) for _ in 1:n]
        @test abs(mean(μr .^ 2) - 0.4) < 4std(μr .^ 2) / sqrt(n)           # ⟨cos²Θ⟩ = 2/5
        v = [sample_lambertian(rng, 0.0, 0.0, 1.0)[3] for _ in 1:n]
        @test all(>(0), v) && abs(mean(v) - 2 / 3) < 4std(v) / sqrt(n)     # cosine-weighted
        # phase functions are normalised: ½∫P dμ = 1
        μs = range(-1, 1; length = 20_001)
        trap(f) = sum(0.5 * (f(μs[i]) + f(μs[i+1])) * step(μs) for i in 1:20_000) / 2
        @test abs(trap(rayleigh_phase) - 1) < 1e-6
        @test abs(trap(x -> henyey_greenstein_phase(0.7, x)) - 1) < 2e-4
        # rotation preserves the norm and realises the scattering angle,
        # including the |u_z| = 1 branch
        worst = 0.0
        for _ in 1:2000
            r = (randn(rng), randn(rng), randn(rng)); u = r ./ sqrt(sum(abs2, r))
            m = 2rand(rng) - 1
            w = rotate_scatter(u, m, 2π * rand(rng))
            worst = max(worst, abs(sqrt(sum(abs2, w)) - 1), abs(sum(u .* w) - m))
        end
        @test worst < 1e-10
        w = rotate_scatter(0.0, 0.0, -1.0, 0.37, 1.234)
        @test !any(isnan, w) && isapprox(w[3], -0.37; atol = 1e-12)
    end

    @testset "Domain and medium construction" begin
        d = mc_domain(Float64; size = (8, 4, 10), spacing = (10.0, 20.0, 5.0))
        @test domain_extent(d) == (80.0, 80.0, 50.0)
        med = homogeneous_medium(d; τ = 3.0, ssa = 0.9, g = 0.6)
        @test isapprox(optical_depth_column(med, 1, 1), 3.0; atol = 1e-12)
        @test_throws ErrorException mc_medium(d; ssa = 1.5)
        # the SolidMask bridge to the flow solver
        mesh = RegularGrid(Float64; size = (8, 4, 10), extent = (80.0, 80.0, 50.0))
        sm = SolidMask(mesh)
        mark_solid!(sm, (x, y, z) -> x < 20 && z < 15)
        med2 = mc_medium(mc_domain(mesh))
        solid_from_mask!(med2, sm)
        @test count(med2.solid) == 2 * 4 * 3
        # the OpticalProperties bridge
        opt = allocate_optics(Float64, (10,), unit_sw_grid())
        opt.τ .= 0.25; opt.ssa .= 0.8; opt.g .= 0.4
        @test isapprox(optical_depth_column(medium_from_optics(d, opt, 1), 3, 2), 2.5; atol = 1e-12)
        # east-north-up, pointing at the sun
        @test all(isapprox.(mc_sun_direction(0.5; azimuth = 90.0), (sqrt(0.75), 0.0, 0.5); atol = 1e-12))
    end

    col_dom = mc_domain(Float64; size = (1, 1, 20), spacing = (100.0, 100.0, 100.0))

    # The local estimator makes the unscattered beam exact (machine precision,
    # not statistical), and a purely absorbing slab over black ground reflects nothing.
    @testset "Beer-Lambert direct beam" begin
        solver = monte_carlo_solver(; nphoton = 1000, nbatch = 8)
        worst = 0.0
        for (τ, μ₀) in ((0.5, 1.0), (5.0, 0.35))
            med = homogeneous_medium(col_dom; τ = τ, ssa = 0.0, g = 0.0, sfc_albedo = 0.0)
            r = mc_solve(med, solver; rng = MersenneTwister(11), sun_dir = μ₀, solar_flux = 1000.0)
            worst = max(worst, abs(r.direct_transmittance - exp(-τ / μ₀)), r.albedo)
        end
        @test worst < 1e-12
        med = homogeneous_medium(col_dom; τ = 2.0, ssa = 0.0, g = 0.0)
        r = mc_solve(med, solver; rng = MersenneTwister(3), sun_dir = 0.8, solar_flux = 1000.0)
        @test maximum(abs(r.flux_dn_dir[1, 1, kl] - 800exp(-2.0 * (21 - kl) / 20 / 0.8)) for kl in 1:21) < 1e-9
    end

    # Energy: every photon weight ends in exactly one sink (clear, absorbing,
    # with buildings), and a conservative slab absorbs nothing.
    @testset "Energy closure" begin
        dom = mc_domain(Float64; size = (6, 6, 8), spacing = (25.0, 25.0, 25.0))
        solver = monte_carlo_solver(; nphoton = 4000, nbatch = 8)
        m = mc_medium(dom; kext = 2e-3, ssa = 0.9, g = 0.5, sfc_albedo = 0.2, solid_albedo = 0.25)
        add_box_building!(m, 2:3, 2:3, 5)
        r = mc_solve(m, solver; rng = MersenneTwister(9),
                     sun_dir = mc_sun_direction(0.4; azimuth = 135.0), solar_flux = 1000.0)
        b = mc_energy_budget(r)
        @test abs(mc_closure_residual(r)) < 1e-10 && abs(b.total - 1) < 1e-10
        @test b.buildings > 0 && all(r.absorbed[m.solid] .== 0)
        vol = dom.dx * dom.dy * dom.dz; A = dom.nx * dom.dx * dom.ny * dom.dy
        @test isapprox(sum(r.absorbed) * vol / (r.incident * A), r.absorptance_atm; atol = 1e-10)

        med = homogeneous_medium(col_dom; τ = 10.0, ssa = 1.0, g = 0.85, sfc_albedo = 0.0)
        r = mc_solve(med, monte_carlo_solver(; nphoton = 8000, nbatch = 8);
                     rng = MersenneTwister(77), sun_dir = 0.5, solar_flux = 1.0)
        @test r.absorptance_atm == 0.0
        @test isapprox(r.albedo + r.transmittance, 1.0; atol = 4sqrt(r.albedo_err^2 + r.transmittance_err^2))
        @test isapprox(r.albedo_analogue + r.absorptance_sfc, 1.0; atol = 1e-12)
    end

    # Headline: the 1-D homogeneous limit reproduces delta-Eddington within
    # its own few-percent error (two cases, 16k photons: σ ≈ 0.003 < 0.03).
    # The worst case (τ=1, ω=0.9, g=0.85, μ₀=0.5; ΔT = −0.027) is in
    # validation-scale statistics only and is not repeated here.
    @testset "1-D limit versus two-stream" begin
        sw = unit_sw_grid()
        solver = monte_carlo_solver(; nphoton = 16_000, nbatch = 16)
        for (τ, ssa, g, μ₀) in ((10.0, 1.0, 0.85, 1.0), (1.0, 0.9, 0.0, 0.5))
            opt = allocate_optics(Float64, (20,), sw)
            opt.τ .= τ / 20; opt.ssa .= ssa; opt.g .= g
            fl = allocate_fluxes(Float64, (), 21)
            solve_sw!(fl, two_stream_solver(sw), opt; μ₀ = μ₀, sfc_albedo_direct = 0.0)
            med = homogeneous_medium(col_dom; τ = τ, ssa = ssa, g = g, sfc_albedo = 0.0)
            r = mc_solve(med, solver; rng = MersenneTwister(2024), sun_dir = μ₀, solar_flux = 1000.0)
            @test isapprox(r.albedo, fl.flux_up[end] / 1000μ₀; atol = max(5r.albedo_err, 0.03))
            @test isapprox(r.transmittance, fl.flux_dn[1] / 1000μ₀; atol = max(5r.transmittance_err, 0.03))
        end
    end

    # solve_rt! is drop-in for the two-stream solver; the MC direct beam is
    # the true unscattered beam (delta-Eddington folds the forward peak in).
    @testset "solve_rt! contract" begin
        sw = unit_sw_grid()
        nz = 20
        opt = allocate_optics(Float64, (nz,), sw)
        opt.τ .= 3.0 / nz; opt.ssa .= 0.95; opt.g .= 0.6
        col = AtmosphericColumn(Float64, nz; dz = fill(100.0, nz), ρ = fill(1.0, nz))
        fl = allocate_fluxes(Float64, (), nz + 1)
        solve_rt!(fl, monte_carlo_solver(sw; nphoton = 16_000, nbatch = 16), opt, col;
                  rng = MersenneTwister(9), μ₀ = 0.7, sfc_albedo = 0.2)
        fl2 = allocate_fluxes(Float64, (), nz + 1)
        solve_sw!(fl2, two_stream_solver(sw), opt; μ₀ = 0.7, sfc_albedo_direct = 0.2)
        @test fl.flux_dn[end] ≈ 700.0 atol = 1e-9
        @test fl.flux_dn_dir[1] ≈ 700.0 * exp(-3.0 / 0.7) atol = 1e-9
        @test abs(fl.flux_up[end] / fl2.flux_up[end] - 1) < 0.05
        @test abs(fl.flux_dn[1] / fl2.flux_dn[1] - 1) < 0.05
        @test_throws ErrorException solve_rt!(fl, monte_carlo_solver(; nphoton = 100, nbatch = 2),
                                              opt, col; rng = MersenneTwister(1))
    end

    # Local and analogue estimators are two unbiased estimators of the same
    # quantity; nadir radiance from next-event estimation is bounded by πL ≤ F_up.
    @testset "Local estimation and radiance" begin
        med = homogeneous_medium(col_dom; τ = 1.0, ssa = 1.0, g = 0.0, sfc_albedo = 0.0)
        r = mc_solve(med, monte_carlo_solver(; nphoton = 8000, nbatch = 16, view_dirs = [(0.0, 0.0, 1.0)]);
                     rng = MersenneTwister(21), sun_dir = 0.8, solar_flux = 1000.0)
        @test abs(r.albedo - r.albedo_analogue) < 4sqrt(r.albedo_err^2 + r.albedo_analogue_err^2)
        @test 0 < π * r.radiance_up[1] <= r.albedo * r.incident * 1.15
    end

    # An opaque tower casts an exact geometric shadow: 40 m at 45° from the
    # east is 8 cells of 5 m to the west, lit only diffusely.
    @testset "Buildings cast shadows in the right direction" begin
        nx = ny = 24; nz = 12
        dom = mc_domain(Float64; size = (nx, ny, nz), spacing = (5.0, 5.0, 5.0))
        med = mc_medium(dom; kext = 2e-3, ssa = 0.9, g = 0.0, sfc_albedo = 0.15, solid_albedo = 0.3)
        add_box_building!(med, 11:13, 11:13, 8)
        r = mc_solve(med, monte_carlo_solver(; nphoton = 4000, nbatch = 8); rng = MersenneTwister(5),
                     sun_dir = mc_sun_direction(1 / sqrt(2); azimuth = 90.0), solar_flux = 1000.0)
        @test abs(mc_closure_residual(r)) < 1e-10
        @test all(r.sfc_direct[3:10, 12] .== 0.0)
        @test all(r.sfc_direct[vcat(1:2, 14:nx), 12] .> 0.9 * 1000 / sqrt(2) * 0.9)
        # lit diffusely (summed over the shadow: with 4000 photons single
        # shadow cells can receive no scattered photon at all)
        @test sum(r.sfc_diffuse[3:10, 12]) > 0.0
        @test sum(r.solid_incident[13, 11:13, 1:8]) > sum(r.solid_incident[11, 11:13, 1:8])
        @test_throws ErrorException mc_solve(med, monte_carlo_solver(; nphoton = 100, nbatch = 2);
                                             rng = MersenneTwister(1), sun_dir = (1.0, 0.0, -0.5))
    end
end
