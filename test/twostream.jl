using Test
using Aeolus
using Aeolus.RadiationCore
using Aeolus.TwoStream
using Aeolus.GasOptics: us_standard_column, correlated_k_optics

@testset "Two-stream RT (Phase 33)" begin

    # single-band grids with a clean source, so references are analytic
    unit_sw_grid(T = Float64; S = 1000.0) =
        SpectralGrid(:shortwave, T[4000.0], T[25000.0], [1], T[1.0], T[S])
    unit_lw_grid(T = Float64) =
        SpectralGrid(:longwave, T[10.0], T[3250.0], [1], T[1.0], T[0.0])

    @testset "Delta-scaling (JWW identities)" begin
        τ, ω, g = 10.0, 0.9, 0.85
        τs, ωs, gs = delta_scale(τ, ω, g)
        f = g^2
        @test isapprox(τs, (1 - ω * f) * τ; atol = 1e-14)
        @test isapprox(ωs, (1 - f) * ω / (1 - ω * f); atol = 1e-14)
        @test isapprox(gs, g / (1 + g); atol = 1e-14)
        @test all(isapprox.(delta_scale(5.0, 0.5, 0.0), (5.0, 0.5, 0.0); atol = 1e-14))
    end

    # Without scattering the direct beam is exact Beer-Lambert.
    @testset "Beer-Lambert direct beam is exact" begin
        sw = unit_sw_grid()
        nz = 20; μ₀ = 0.6
        opt = allocate_optics(Float64, (nz,), sw)
        opt.τ .= 0.05; opt.ssa .= 0.0; opt.g .= 0.0
        fl = allocate_fluxes(Float64, (), nz + 1)
        solve_sw!(fl, two_stream_solver(sw), opt; μ₀ = μ₀, sfc_albedo_direct = 0.0)
        expect = [1000μ₀ * exp(-0.05 * (nz + 1 - i) / μ₀) for i in 1:nz+1]
        @test maximum(abs.(fl.flux_dn_dir .- expect)) < 1e-10
        @test maximum(fl.flux_up) < 1e-10
    end

    # Energy: surface reflection = albedo, passive column absorbs ≥ 0; a
    # conservative column over a white surface returns everything.
    @testset "Energy conservation (shortwave)" begin
        sw = unit_sw_grid()
        solver = two_stream_solver(sw)
        nz = 30
        opt = allocate_optics(Float64, (nz,), sw)
        for k in 1:nz
            opt.τ[k, 1] = 0.02 + 0.3k / nz; opt.ssa[k, 1] = 0.3 + 0.6k / nz
            opt.g[k, 1] = 0.5 + 0.3sin(k)
        end
        fl = allocate_fluxes(Float64, (), nz + 1)
        solve_sw!(fl, solver, opt; μ₀ = 0.7, sfc_albedo_direct = 0.25)
        absorbed = (fl.flux_dn[end] - fl.flux_up[end]) - (fl.flux_dn[1] - fl.flux_up[1])
        @test absorbed >= -1e-10
        @test isapprox(fl.flux_up[1], 0.25 * fl.flux_dn[1]; rtol = 1e-10)

        o = allocate_optics(Float64, (10,), sw)
        o.τ .= 0.5; o.ssa .= 1.0; o.g .= 0.0
        f = allocate_fluxes(Float64, (), 11)
        solve_sw!(f, solver, o; μ₀ = 0.8, sfc_albedo_direct = 1.0, sfc_albedo_diffuse = 1.0)
        @test isapprox(f.flux_up[end], f.flux_dn[end]; rtol = 2e-3)
    end

    # τ→0, ω→1 and the kμ₀ = 1 resonance must stay finite in every variant.
    @testset "Singular limits produce no NaN" begin
        for variant in (DeltaEddington(), PIFM(), Quadrature())
            R, Tt, _, _, Tb = layer_reflectance_transmittance_sw(variant, 0.0, 0.5, 0.5, 0.6)
            @test R == 0 && Tt == 1 && Tb == 1
            rs = [layer_reflectance_transmittance_sw(variant, 5.0, ω, 0.85, 0.6)
                  for ω in (1.0, 1 - 1e-15, 1 - 1e-9)]
            append!(rs, [layer_reflectance_transmittance_sw(variant, 1.0, 0.9, 0.7, μ)
                         for μ in 0.01:0.01:1.0])
            @test all(r -> all(isfinite, r) && 0 <= r[1] <= 1 && 0 <= r[2] <= 1, rs)
            # diffusion regime at huge τ: T ~ 1/τ, not exponential, but small
            r = layer_reflectance_transmittance_sw(variant, 1e4, 0.999, 0.85, 0.5)
            @test all(isfinite, r) && r[2] < 1e-2
        end
    end

    # Opaque isothermal column radiates as a blackbody (band fraction of σT⁴);
    # a transparent one passes the surface emission.
    @testset "Longwave blackbody and thin limits" begin
        lw = unit_lw_grid()
        solver = two_stream_solver(lw)
        nz = 40; T = 280.0
        col = AtmosphericColumn(Float64, nz; p_lay = fill(5e4, nz), T_lay = fill(T, nz),
            p_lev = fill(5e4, nz + 1), T_lev = fill(T, nz + 1),
            dz = fill(100.0, nz), ρ = fill(1.0, nz), T_sfc = T)
        B = π * planck_band(T, 10.0, 3250.0)
        opt = allocate_optics(Float64, (nz,), lw)
        opt.τ .= 1.0; opt.ssa .= 0.0; opt.g .= 0.0
        fl = allocate_fluxes(Float64, (), nz + 1)
        solve_lw!(fl, solver, opt, col; sfc_emissivity = 1.0)
        @test isapprox(fl.flux_up[end], B; rtol = 1e-3)
        @test isapprox(fl.flux_dn[1], B; rtol = 1e-3)
        opt.τ .= 1e-6
        solve_lw!(fl, solver, opt, col; sfc_emissivity = 1.0)
        @test isapprox(fl.flux_up[end], B; rtol = 1e-3)
        @test fl.flux_dn[1] < 0.01B
    end

    # Linear-in-τ source is closer to a 20x sub-layered reference than a constant one.
    @testset "Linear-in-τ Planck source" begin
        lw = unit_lw_grid()
        function toa(nz, sub, linear)
            Ts = range(300.0, 220.0; length = nz + 1)
            col = AtmosphericColumn(Float64, nz; p_lay = fill(5e4, nz),
                T_lay = [(Ts[k] + Ts[k+1]) / 2 for k in 1:nz], p_lev = fill(5e4, nz + 1),
                T_lev = collect(Ts), dz = fill(1000.0 / sub, nz), ρ = fill(1.0, nz), T_sfc = 300.0)
            o = allocate_optics(Float64, (nz,), lw); o.τ .= 0.5 / sub; o.ssa .= 0.0
            f = allocate_fluxes(Float64, (), nz + 1)
            solve_lw!(f, two_stream_solver(lw; linear_lw_source = linear), o, col)
            f.flux_up[end]
        end
        ref = toa(120, 20, true)
        e_lin, e_con = abs(toa(6, 1, true) - ref), abs(toa(6, 1, false) - ref)
        @test e_lin <= e_con && e_lin < 2.0
    end

    # Cloud optics: τ = 1.5 LWP/r_eff, conservative and forward scattering in the visible.
    @testset "Cloud optics and overlap" begin
        sw = shortwave_bands(; ngpt_per_band = 1)
        cloud = mie_cloud_optics(sw)
        nz = 10
        col = AtmosphericColumn(Float64, nz; p_lay = fill(8e4, nz), T_lay = fill(280.0, nz),
            p_lev = fill(8e4, nz + 1), T_lev = fill(280.0, nz + 1), dz = fill(100.0, nz), ρ = fill(1.0, nz))
        col.ql[5] = 100.0 / (1000 * 100.0)
        col.r_eff_liq .= 10.0
        @test isapprox(liquid_water_path(col), 100.0; rtol = 1e-10)
        opt = allocate_optics(Float64, (nz,), sw)
        compute_optics!(opt, cloud, col)
        gv = first(gpoint_range(sw, 10))
        τv = sum(@view opt.τ[:, gv])
        @test 10 < τv < 20                         # 1.5·100/10 = 15
        @test opt.ssa[5, gv] > 0.999 && 0.80 < opt.g[5, gv] < 0.90
        @test opt.ssa[5, first(gpoint_range(sw, 3))] < opt.ssa[5, gv]
        col.r_eff_liq .= 20.0
        compute_optics!(opt, cloud, col)
        @test isapprox(sum(@view opt.τ[:, gv]), τv / 2; rtol = 1e-10)

        @test isapprox(maximum_random_overlap([0.3, 0.7]), 0.7; atol = 1e-14)      # maximum
        @test isapprox(maximum_random_overlap([0.5, 0.0, 0.5]), 0.75; atol = 1e-14) # random
    end

    # Real gas optics: LW cools the troposphere at plausible rates, a
    # greenhouse effect, and SW absorption warms.
    @testset "Heating rates with correlated-k gas optics" begin
        lw = longwave_bands(; ngpt_per_band = 2)
        col = us_standard_column(; nz = 20)
        opt = allocate_optics(Float64, (20,), lw)
        compute_optics!(opt, correlated_k_optics(lw), col)
        fl = allocate_fluxes(Float64, (), 21)
        solve_lw!(fl, two_stream_solver(lw), opt, col; sfc_emissivity = 1.0)
        @test 150 < fl.flux_up[end] < 350 && fl.flux_up[1] > fl.flux_up[end]
        dθ = zeros(20)
        radiative_heating!(dθ, fl, col)
        @test sum(dθ) < 0 && maximum(abs, dθ) * 86400 < 30

        sw = unit_sw_grid()
        sopt = allocate_optics(Float64, (20,), sw)
        sopt.τ .= 0.05; sopt.ssa .= 0.5; sopt.g .= 0.0
        sfl = allocate_fluxes(Float64, (), 21)
        solve_sw!(sfl, two_stream_solver(sw), sopt; μ₀ = 0.8, sfc_albedo_direct = 0.2)
        radiative_heating!(dθ, sfl, col)
        @test sum(dθ) > 0
    end
end
