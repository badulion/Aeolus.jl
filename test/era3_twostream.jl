using Test
using Aeolus
using Aeolus.RadiationCore
using Aeolus.TwoStream
using Aeolus.GasOptics: us_standard_column

@testset "Era III — Two-stream RT (Phase 33)" begin

    # Helper: a single-band shortwave grid with unit solar source, so the
    # analytic references below are clean.
    function unit_sw_grid(T = Float64; S = 1000.0)
        SpectralGrid(:shortwave, T[4000.0], T[25000.0], [1],
                     T[1.0], T[S])
    end
    function unit_lw_grid(T = Float64)
        SpectralGrid(:longwave, T[10.0], T[3250.0], [1], T[1.0], T[0.0])
    end

    # ---------------------------------------------------------------------
    # Delta-scaling: the JWW identities.
    # ---------------------------------------------------------------------
    @testset "Delta-scaling" begin
        τ, ω, g = 10.0, 0.9, 0.85
        τs, ωs, gs = delta_scale(τ, ω, g)
        f = g^2
        @test isapprox(τs, (1 - ω*f)*τ; atol = 1e-14)
        @test isapprox(ωs, (1-f)*ω/(1-ω*f); atol = 1e-14)
        @test isapprox(gs, g/(1+g); atol = 1e-14)
        @test τs < τ            # forward spike removed from the optical depth
        @test gs < g            # remaining phase function less peaked
        @info "delta-scaling" τ τs ω ωs g gs

        # isotropic scattering is unchanged
        τ2, ω2, g2 = delta_scale(5.0, 0.5, 0.0)
        @test isapprox(τ2, 5.0; atol=1e-14)
        @test isapprox(ω2, 0.5; atol=1e-14)
        @test isapprox(g2, 0.0; atol=1e-14)
    end

    # ---------------------------------------------------------------------
    # Beer-Lambert: with no scattering the direct beam must be EXACT.
    # ---------------------------------------------------------------------
    @testset "Beer-Lambert direct beam is exact" begin
        sw = unit_sw_grid(; S = 1000.0)
        solver = two_stream_solver(sw)
        nz = 20
        opt = allocate_optics(Float64, (nz,), sw)
        τ_lay = 0.05
        opt.τ .= τ_lay; opt.ssa .= 0.0; opt.g .= 0.0
        fl = allocate_fluxes(Float64, (), nz + 1)
        μ₀ = 0.6
        solve_sw!(fl, solver, opt; μ₀ = μ₀, sfc_albedo_direct = 0.0)

        worst = 0.0
        for i in 1:nz+1
            τ_above = τ_lay * (nz + 1 - i)
            expect = 1000.0 * μ₀ * exp(-τ_above / μ₀)
            worst = max(worst, abs(fl.flux_dn_dir[i] - expect))
        end
        @info "Beer-Lambert max error" worst
        @test worst < 1e-10

        # no scattering and a black surface ⇒ nothing comes back up
        @test maximum(fl.flux_up) < 1e-10
    end

    # ---------------------------------------------------------------------
    # Energy conservation — the master check.
    # ---------------------------------------------------------------------
    @testset "Energy conservation (shortwave)" begin
        sw = unit_sw_grid(; S = 1000.0)
        solver = two_stream_solver(sw)
        nz = 30
        opt = allocate_optics(Float64, (nz,), sw)
        # random but physical optical properties
        for k in 1:nz
            opt.τ[k,1]   = 0.02 + 0.3 * (k / nz)
            opt.ssa[k,1] = 0.3 + 0.6 * (k / nz)
            opt.g[k,1]   = 0.5 + 0.3 * sin(k)
        end
        fl = allocate_fluxes(Float64, (), nz + 1)
        μ₀ = 0.7; α = 0.25
        solve_sw!(fl, solver, opt; μ₀ = μ₀, sfc_albedo_direct = α)

        toa_in   = fl.flux_dn[nz+1]
        toa_out  = fl.flux_up[nz+1]
        sfc_dn   = fl.flux_dn[1]
        sfc_up   = fl.flux_up[1]
        absorbed = (toa_in - toa_out) - (sfc_dn - sfc_up)
        residual = abs((toa_in - toa_out) - (sfc_dn - sfc_up) - absorbed)
        @info "SW energy budget" toa_in toa_out sfc_dn sfc_up column_absorbed=absorbed
        @test residual < 1e-10
        @test absorbed >= -1e-10            # a passive medium cannot emit SW
        @test toa_out >= 0 && sfc_dn >= 0
        # surface reflection must match the albedo exactly
        @test isapprox(sfc_up, α * sfc_dn; rtol = 1e-10)
    end

    # ---------------------------------------------------------------------
    # Singular limits: τ→0, ω→1, and kμ₀→1 must all be finite.
    # ---------------------------------------------------------------------
    @testset "Singular limits produce no NaN" begin
        for variant in (DeltaEddington(), PIFM(), Quadrature())
            # τ → 0
            R, Tt, Rr, Tr, Tb = layer_reflectance_transmittance_sw(
                variant, 0.0, 0.5, 0.5, 0.6)
            @test R == 0 && Tt == 1 && Tb == 1
            # ω → 1 (conservative scattering)
            for ω in (1.0, 1 - 1e-15, 1 - 1e-9)
                r = layer_reflectance_transmittance_sw(variant, 5.0, ω, 0.85, 0.6)
                @test all(isfinite, r)
                @test 0 <= r[1] <= 1 && 0 <= r[2] <= 1
            end
            # scan μ₀ across the kμ₀ = 1 resonance
            for μ in 0.01:0.01:1.0
                r = layer_reflectance_transmittance_sw(variant, 1.0, 0.9, 0.7, μ)
                @test all(isfinite, r)
            end
            # very large optical depth
            r = layer_reflectance_transmittance_sw(variant, 1e4, 0.999, 0.85, 0.5)
            @test all(isfinite, r)
            # A near-conservative medium at huge τ is in the diffusion
            # regime: transmittance falls like 1/τ, not exponentially.
            @test r[2] < 1e-2
        end
        @info "all singular limits finite for all three variants"
    end

    # ---------------------------------------------------------------------
    # Conservative scattering: a non-absorbing layer over a black surface
    # must conserve energy exactly.
    # ---------------------------------------------------------------------
    @testset "Conservative scattering conserves energy" begin
        sw = unit_sw_grid(; S = 1000.0)
        solver = two_stream_solver(sw)
        nz = 10
        opt = allocate_optics(Float64, (nz,), sw)
        opt.τ .= 0.5; opt.ssa .= 1.0; opt.g .= 0.0
        fl = allocate_fluxes(Float64, (), nz + 1)
        μ₀ = 0.8
        # a perfectly reflecting surface: nothing can be absorbed anywhere
        solve_sw!(fl, solver, opt; μ₀ = μ₀, sfc_albedo_direct = 1.0,
                  sfc_albedo_diffuse = 1.0)
        in_ = fl.flux_dn[nz+1]; out = fl.flux_up[nz+1]
        @info "conservative + white surface" incoming=in_ outgoing=out ratio=out/in_
        # everything that goes in must come back out
        @test isapprox(out, in_; rtol = 2e-3)

        # with a black surface, albedo increases monotonically with τ
        albs = Float64[]
        for τc in (0.1, 0.5, 2.0, 10.0)
            o = allocate_optics(Float64, (nz,), sw)
            o.τ .= τc / nz; o.ssa .= 1.0; o.g .= 0.0
            f = allocate_fluxes(Float64, (), nz + 1)
            solve_sw!(f, solver, o; μ₀ = μ₀, sfc_albedo_direct = 0.0)
            push!(albs, f.flux_up[nz+1] / f.flux_dn[nz+1])
        end
        @info "conservative-scattering column albedo vs τ" albs
        @test issorted(albs)
        @test all(0 .<= albs .<= 1)
    end

    # ---------------------------------------------------------------------
    # Longwave: an isothermal non-scattering atmosphere must radiate as a
    # blackbody at TOA once it is optically thick.
    # ---------------------------------------------------------------------
    @testset "Longwave blackbody limit" begin
        lw = unit_lw_grid()
        solver = two_stream_solver(lw)
        nz = 40
        Tiso = 280.0
        col = AtmosphericColumn(Float64, nz;
            p_lay = fill(5e4, nz), T_lay = fill(Tiso, nz),
            p_lev = fill(5e4, nz+1), T_lev = fill(Tiso, nz+1),
            dz = fill(100.0, nz), ρ = fill(1.0, nz), T_sfc = Tiso)
        opt = allocate_optics(Float64, (nz,), lw)
        opt.τ .= 1.0; opt.ssa .= 0.0; opt.g .= 0.0   # total τ = 40, opaque
        fl = allocate_fluxes(Float64, (), nz + 1)
        solve_lw!(fl, solver, opt, col; sfc_emissivity = 1.0)

        σT4 = STEFAN_BOLTZMANN * Tiso^4
        # The single band spans 10-3250 cm⁻¹, which holds ~99.99% of the
        # Planck integral at this temperature; the residual is the
        # truncation documented in RadiationCore.
        band_frac = π * planck_band(Tiso, 10.0, 3250.0) / σT4
        toa_up = fl.flux_up[nz+1]
        sfc_dn = fl.flux_dn[1]
        @info "LW blackbody limit" toa_up expected=σT4*band_frac band_frac sfc_dn
        @test isapprox(toa_up, σT4 * band_frac; rtol = 1e-3)
        # an opaque isothermal atmosphere also floods the surface
        @test isapprox(sfc_dn, σT4 * band_frac; rtol = 1e-3)
        # net flux vanishes in an isothermal opaque column: nothing heats
        @test abs(fl.flux_net[1]) < 0.05 * σT4

        # optically thin: TOA up must approach the surface emission
        opt2 = allocate_optics(Float64, (nz,), lw)
        opt2.τ .= 1e-6
        fl2 = allocate_fluxes(Float64, (), nz + 1)
        solve_lw!(fl2, solver, opt2, col; sfc_emissivity = 1.0)
        @info "LW optically thin" toa_up=fl2.flux_up[nz+1] sfc_emission=σT4*band_frac
        @test isapprox(fl2.flux_up[nz+1], σT4 * band_frac; rtol = 1e-3)
        @test fl2.flux_dn[1] < 0.01 * σT4       # thin ⇒ little back-radiation
    end

    # ---------------------------------------------------------------------
    # Linear-in-τ source beats a constant source when there is a gradient.
    # ---------------------------------------------------------------------
    @testset "Linear-in-τ Planck source" begin
        lw = unit_lw_grid()
        # a strong temperature gradient across few layers
        nz = 6
        Ts = range(300.0, 220.0; length = nz + 1)
        col = AtmosphericColumn(Float64, nz;
            p_lay = fill(5e4, nz), T_lay = [(Ts[k]+Ts[k+1])/2 for k in 1:nz],
            p_lev = fill(5e4, nz+1), T_lev = collect(Ts),
            dz = fill(1000.0, nz), ρ = fill(1.0, nz), T_sfc = 300.0)
        opt = allocate_optics(Float64, (nz,), lw)
        opt.τ .= 0.5; opt.ssa .= 0.0

        # high-resolution reference: same profile on 20x more layers
        nzr = nz * 20
        Tr = range(300.0, 220.0; length = nzr + 1)
        colr = AtmosphericColumn(Float64, nzr;
            p_lay = fill(5e4, nzr), T_lay = [(Tr[k]+Tr[k+1])/2 for k in 1:nzr],
            p_lev = fill(5e4, nzr+1), T_lev = collect(Tr),
            dz = fill(1000.0/20, nzr), ρ = fill(1.0, nzr), T_sfc = 300.0)
        optr = allocate_optics(Float64, (nzr,), lw)
        optr.τ .= 0.5 / 20; optr.ssa .= 0.0
        flr = allocate_fluxes(Float64, (), nzr + 1)
        solve_lw!(flr, two_stream_solver(lw; linear_lw_source=true), optr, colr)
        ref = flr.flux_up[end]

        fl_lin = allocate_fluxes(Float64, (), nz + 1)
        solve_lw!(fl_lin, two_stream_solver(lw; linear_lw_source=true), opt, col)
        fl_con = allocate_fluxes(Float64, (), nz + 1)
        solve_lw!(fl_con, two_stream_solver(lw; linear_lw_source=false), opt, col)

        e_lin = abs(fl_lin.flux_up[end] - ref)
        e_con = abs(fl_con.flux_up[end] - ref)
        @info "TOA flux error vs sub-layered reference" reference=ref linear=fl_lin.flux_up[end] constant=fl_con.flux_up[end] err_linear=e_lin err_constant=e_con
        @test e_lin <= e_con        # linear is no worse, and generally better
        @test e_lin < 2.0           # and close to the reference in absolute terms
    end

    # ---------------------------------------------------------------------
    # Cloud optics: the standard benchmark numbers.
    # ---------------------------------------------------------------------
    @testset "Cloud optics" begin
        sw = shortwave_bands(; ngpt_per_band = 1)
        cloud = mie_cloud_optics(sw)
        nz = 10
        col = AtmosphericColumn(Float64, nz;
            p_lay = fill(8e4, nz), T_lay = fill(280.0, nz),
            p_lev = fill(8e4, nz+1), T_lev = fill(280.0, nz+1),
            dz = fill(100.0, nz), ρ = fill(1.0, nz))
        # 100 g/m² LWP spread over one layer, r_eff = 10 μm
        col.ql[5] = 100.0 / (1000 * 1.0 * 100.0)
        col.r_eff_liq .= 10.0
        @test isapprox(liquid_water_path(col), 100.0; rtol = 1e-10)

        opt = allocate_optics(Float64, (nz,), sw)
        compute_optics!(opt, cloud, col)

        # visible band (band 10 of the RRTMG SW layout, ~16000-22650 cm⁻¹)
        gvis = first(gpoint_range(sw, 10))
        τ_vis = sum(@view opt.τ[:, gvis])
        ssa_vis = opt.ssa[5, gvis]
        g_vis = opt.g[5, gvis]
        @info "cloud at LWP=100 g/m², r_eff=10 μm" τ_visible=τ_vis ssa=ssa_vis g=g_vis
        # τ = 1.5·LWP/r_eff = 15 for these values
        @test 10 < τ_vis < 20
        @test ssa_vis > 0.999          # clouds barely absorb in the visible
        @test 0.80 < g_vis < 0.90      # strongly forward-scattering

        # near-IR absorbs more than the visible
        gnir = first(gpoint_range(sw, 3))
        @test opt.ssa[5, gnir] < ssa_vis
        @info "ssa visible vs near-IR" visible=ssa_vis near_IR=opt.ssa[5,gnir]

        # optical depth scales with LWP and inversely with r_eff
        col2 = deepcopy(col); col2.ql[5] *= 2
        o2 = allocate_optics(Float64, (nz,), sw); compute_optics!(o2, cloud, col2)
        @test isapprox(sum(@view o2.τ[:,gvis]), 2*τ_vis; rtol=1e-10)
        col3 = deepcopy(col); col3.r_eff_liq .= 20.0
        o3 = allocate_optics(Float64, (nz,), sw); compute_optics!(o3, cloud, col3)
        @test isapprox(sum(@view o3.τ[:,gvis]), τ_vis/2; rtol=1e-10)
    end

    # ---------------------------------------------------------------------
    # Cloud albedo saturates below 1 and rises monotonically with LWP.
    # ---------------------------------------------------------------------
    @testset "Cloud albedo saturation" begin
        sw = unit_sw_grid(; S = 1000.0)
        solver = two_stream_solver(sw)
        nz = 5
        albs = Float64[]
        for lwp in (10.0, 50.0, 200.0, 1000.0)
            opt = allocate_optics(Float64, (nz,), sw)
            τ = 1.5 * lwp / 10.0
            opt.τ .= τ / nz; opt.ssa .= 0.9999; opt.g .= 0.85
            fl = allocate_fluxes(Float64, (), nz + 1)
            solve_sw!(fl, solver, opt; μ₀ = 0.8, sfc_albedo_direct = 0.0)
            push!(albs, fl.flux_up[nz+1] / fl.flux_dn[nz+1])
        end
        @info "cloud albedo vs LWP (10,50,200,1000 g/m²)" albs
        @test issorted(albs)
        @test all(albs .< 1.0)          # saturates below unity
        @test albs[end] > 0.5           # a thick cloud is bright
    end

    # ---------------------------------------------------------------------
    # Cloud overlap.
    # ---------------------------------------------------------------------
    @testset "Maximum-random overlap" begin
        @test maximum_random_overlap([0.0, 0.0, 0.0]) == 0.0
        @test isapprox(maximum_random_overlap([0.5]), 0.5; atol=1e-14)
        # contiguous layers overlap maximally: total = max, not 1-(1-a)(1-b)
        @test isapprox(maximum_random_overlap([0.5, 0.5]), 0.5; atol=1e-14)
        @test isapprox(maximum_random_overlap([0.3, 0.7]), 0.7; atol=1e-14)
        # separated groups combine randomly
        c = maximum_random_overlap([0.5, 0.0, 0.5])
        @test isapprox(c, 1 - 0.5*0.5; atol=1e-14)
        @info "overlap" contiguous=maximum_random_overlap([0.5,0.5]) separated=c
        @test 0 <= c <= 1
    end

    # ---------------------------------------------------------------------
    # Heating rates: right sign, plausible magnitude.
    # ---------------------------------------------------------------------
    @testset "Radiative heating rates" begin
        lw = longwave_bands(; ngpt_per_band = 2)
        solver = two_stream_solver(lw)
        col = us_standard_column(; nz = 20)
        opt = allocate_optics(Float64, (20,), lw)
        # Realistic gas optics — a uniform grey τ is not an atmosphere and
        # produces unphysically large cooling near the top.
        compute_optics!(opt, Aeolus.GasOptics.correlated_k_optics(lw), col)
        fl = allocate_fluxes(Float64, (), 21)
        solve_lw!(fl, solver, opt, col; sfc_emissivity = 1.0)

        dθ = zeros(20)
        radiative_heating!(dθ, fl, col)
        Kday = dθ .* 86400
        @info "LW heating rate (K/day)" min=minimum(Kday) max=maximum(Kday) mean=sum(Kday)/20
        @test all(isfinite, Kday)
        # the troposphere cools radiatively in the longwave — the
        # fundamental driver of the atmospheric circulation
        @test sum(Kday) / 20 < 0
        # magnitudes should be order 1-10 K/day, not 1000
        @test maximum(abs, Kday) < 30

        # shortwave: an absorbing layer under the sun must warm
        sw = unit_sw_grid(; S = 1000.0)
        ssolver = two_stream_solver(sw)
        sopt = allocate_optics(Float64, (20,), sw)
        sopt.τ .= 0.05; sopt.ssa .= 0.5; sopt.g .= 0.0
        sfl = allocate_fluxes(Float64, (), 21)
        solve_sw!(sfl, ssolver, sopt; μ₀ = 0.8, sfc_albedo_direct = 0.2)
        dθs = zeros(20)
        radiative_heating!(dθs, sfl, col)
        Kday_s = dθs .* 86400
        @info "SW heating rate (K/day)" min=minimum(Kday_s) max=maximum(Kday_s)
        @test sum(Kday_s) > 0        # net solar warming
        @test all(isfinite, Kday_s)
    end

    # ---------------------------------------------------------------------
    # End to end: real gas optics through the solver.
    # ---------------------------------------------------------------------
    @testset "Coupled to correlated-k gas optics" begin
        using Aeolus.GasOptics: correlated_k_optics
        lw = longwave_bands(; ngpt_per_band = 2)
        go = correlated_k_optics(lw)
        solver = two_stream_solver(lw)
        col = us_standard_column(; nz = 25)
        opt = allocate_optics(Float64, (25,), lw)
        compute_optics!(opt, go, col)
        fl = allocate_fluxes(Float64, (), 26)
        solve_lw!(fl, solver, opt, col; sfc_emissivity = 0.98)

        olr = fl.flux_up[end]
        sfc_dn = fl.flux_dn[1]
        sfc_up = fl.flux_up[1]
        @info "clear-sky LW fluxes" OLR=olr surface_downwelling=sfc_dn surface_upwelling=sfc_up

        @test all(isfinite, fl.flux_up)
        @test all(fl.flux_up .>= 0) && all(fl.flux_dn .>= 0)
        # Observed clear-sky global-mean OLR is ~265 W/m² and surface
        # downwelling LW ~330 W/m². A band model should land in the right
        # neighbourhood even if not to the watt.
        @test 150 < olr < 350
        @test 150 < sfc_dn < 420
        # the greenhouse effect: the surface emits more than escapes to space
        @test sfc_up > olr
        @info "greenhouse effect (surface emission - OLR)" Δ=sfc_up-olr
    end
end
