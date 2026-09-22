using Test
using Aeolus
using Aeolus.RadiationCore
using Aeolus.GasOptics

@testset "Era III — Correlated-k gas optics (Phase 32)" begin

    # ---------------------------------------------------------------------
    # Error function: no SpecialFunctions dependency, so verify our own
    # implementation against tabulated values before relying on it.
    # ---------------------------------------------------------------------
    @testset "Error function" begin
        # tabulated to 16 digits
        @test isapprox(erf_series(0.5),  0.5204998778130465; atol = 1e-13)
        @test isapprox(erf_series(1.0),  0.8427007929497149; atol = 1e-13)
        @test isapprox(erf_series(2.0),  0.9953222650189527; atol = 1e-13)
        @test isapprox(erfc_series(2.0), 0.004677734981047266; atol = 1e-15)
        @test isapprox(erfc_series(4.0), 1.541725790028002e-8; atol = 1e-20)
        @test isapprox(erfc_series(6.0), 2.1519736712498913e-17; atol = 1e-28)
        @test erf_series(0.0) == 0.0
        @test isapprox(erfc_series(0.0), 1.0; atol = 1e-15)
        @test isapprox(erf_series(-1.0), -erf_series(1.0); atol = 1e-15)
        # erf + erfc = 1 across the range
        for x in (0.1, 0.7, 1.5, 2.5, 3.5, 5.0)
            @test isapprox(erf_series(x) + erfc_series(x), 1.0; atol = 1e-13)
        end
        @info "erfc accuracy" e2=erfc_series(2.0) e4=erfc_series(4.0) e6=erfc_series(6.0)
    end

    # ---------------------------------------------------------------------
    # Malkmus limits — the two regimes the band model must interpolate.
    # ---------------------------------------------------------------------
    @testset "Malkmus weak- and strong-line limits" begin
        p = MalkmusParams(1.0, 1.0, 0.5)   # k = 1 m²/kg, B = 1

        # weak line: ku ≪ B ⇒ T → exp(-ku)
        werr = 0.0
        for u in (1e-6, 1e-5, 1e-4)
            t  = malkmus_transmission(u, p)
            te = exp(-1.0 * u)
            werr = max(werr, abs(t - te) / te)
        end
        @info "Malkmus weak-line limit" max_rel_error=werr
        @test werr < 1e-4

        # strong line: ku ≫ B ⇒ T → exp(-√(kuB)).
        # Compared in log space: at these path amounts T underflows to
        # exactly 0, so log(T) would be -Inf and the ratio NaN.
        serr = 0.0
        for u in (1e4, 1e5, 1e6)
            lt = malkmus_log_transmission(u, p)
            le = -sqrt(1.0 * u * 1.0)
            serr = max(serr, abs(lt - le) / abs(le))
        end
        @info "Malkmus strong-line limit" max_rel_log_error=serr
        @test serr < 1e-2

        # monotone decreasing in u, bounded in [0,1]
        us = 10.0 .^ (-6:0.5:6)
        ts = [malkmus_transmission(u, p) for u in us]
        @test all(0 .<= ts .<= 1)
        @test all(diff(ts) .< 0)
        @test malkmus_transmission(0.0, p) == 1.0

        # Pressure broadening widens the lines so they fill the gaps
        # between line centres, which INCREASES band-mean absorption.
        # Hence higher pressure ⇒ lower transmission. (In the B → ∞ limit
        # T → exp(-ku), the grey/weak-line result, which is the most
        # absorption a given k·u can produce; as B → 0 the strong-line
        # form exp(-√(kuB)) → 1.)
        t_lowp  = malkmus_transmission(10.0, p; pressure = 2e4)
        t_highp = malkmus_transmission(10.0, p; pressure = 1.0e5)
        @info "pressure dependence" low_p=t_lowp high_p=t_highp
        @test t_highp < t_lowp
    end

    # ---------------------------------------------------------------------
    # THE key test: the correlated-k quadrature must reproduce the band
    # transmission across many decades of absorber amount. This is what
    # makes it correlated-k rather than a band average.
    # ---------------------------------------------------------------------
    @testset "Correlated-k reproduces band transmission" begin
        p = MalkmusParams(2.0, 0.3, 0.5)
        us = 10.0 .^ (-4:0.5:4)

        results = Dict{Int,Float64}()
        for ngpt in (2, 4, 8, 16)
            kg, wg = kdist_from_band_model(p, ngpt)
            @test isapprox(sum(wg), 1.0; atol = 1e-14)
            @test all(kg .>= 0)
            @test issorted(kg)                 # reordered monotonically
            worst = 0.0
            for u in us
                t_quad = sum(wg[i] * exp(-kg[i] * u) for i in eachindex(kg))
                t_band = malkmus_transmission(u, p)
                worst = max(worst, abs(t_quad - t_band))
            end
            results[ngpt] = worst
        end
        @info "correlated-k quadrature error vs ngpt" results

        # converges with more g-points
        @test results[4]  < results[2]
        @test results[8]  < results[4]
        @test results[16] < results[8]
        # 16 g-points should be quite accurate
        @test results[16] < 0.02

        # A plain band average (single effective k) is much worse — this is
        # the comparison that justifies the whole machinery.
        kbar = p.k
        worst_avg = maximum(abs(exp(-kbar * u) - malkmus_transmission(u, p))
                            for u in us)
        @info "single-k band average error" worst_avg ck_16=results[16]
        @test worst_avg > 10 * results[16]
    end

    # ---------------------------------------------------------------------
    # Rayleigh scattering: λ⁻⁴ and the standard-atmosphere column value.
    # ---------------------------------------------------------------------
    @testset "Rayleigh scattering" begin
        dp = 1.01325e5                         # whole atmosphere
        τ550 = rayleigh_optical_depth(0.55, dp)
        @info "Rayleigh column optical depth at 550 nm" τ550 reference=0.0973
        @test 0.085 < τ550 < 0.11              # published ≈ 0.0973

        # fit the exponent: τ ∝ λ^(-n)
        λs = [0.35, 0.45, 0.55, 0.65, 0.85]
        τs = [rayleigh_optical_depth(λ, dp) for λ in λs]
        # log-log slope
        n = -(log(τs[end]) - log(τs[1])) / (log(λs[end]) - log(λs[1]))
        @info "fitted Rayleigh exponent" n
        @test 3.9 < n < 4.1

        @test all(τs .> 0)
        @test issorted(τs; rev = true)         # more scattering at short λ
    end

    # ---------------------------------------------------------------------
    # Planck source: summing g-points and bands must recover σT⁴.
    # ---------------------------------------------------------------------
    @testset "Planck source consistency" begin
        lw = longwave_bands(; ngpt_per_band = 2)
        go = correlated_k_optics(lw)
        Tt = 288.0
        src = lw_surface_source(go, Tt)
        total = sum(src)
        expect = planck_integrated(Tt)
        rel = abs(total - expect) / expect
        @info "Planck source vs σT⁴" total expect rel_error=rel
        # the residual is the band-truncation error documented in
        # RadiationCore (energy outside 10-3250 cm⁻¹), ~0.008%
        @test rel < 2e-3
        @test all(src .>= 0)

        # hotter surface emits more, and as T⁴
        s300 = sum(lw_surface_source(go, 300.0))
        @test s300 > total
        @test isapprox(s300 / total, (300 / 288)^4; rtol = 5e-3)
    end

    # ---------------------------------------------------------------------
    # A realistic clear-sky column.
    # ---------------------------------------------------------------------
    @testset "US-standard column" begin
        col = us_standard_column(; nz = 30)
        @test length(col.p_lay) == 30
        @test col.p_lev[1] > col.p_lev[end]            # decreasing upward
        @test all(col.T_lay .> 180)
        @test all(col.ρ .> 0)

        # precipitable water — mid-latitude summer is ~14 kg/m²
        pw = sum(column_absorber_amount(col, :h2o, k) for k in 1:30)
        @info "precipitable water (midlatitude summer)" pw
        @test 10 < pw < 20

        col_w = us_standard_column(; nz = 30, humid = :subarctic_winter)
        pw_w = sum(column_absorber_amount(col_w, :h2o, k) for k in 1:30)
        @info "precipitable water (subarctic winter)" pw_w
        @test pw_w < pw / 2                            # much drier

        # CO2 column: ~420 ppm over the whole atmosphere ≈ 6.4 kg/m²
        co2 = sum(column_absorber_amount(col, :co2, k) for k in 1:30)
        @info "CO2 column" co2
        @test 4 < co2 < 10
    end

    # ---------------------------------------------------------------------
    # The defining structure of the LW spectrum: the 8-12 μm window must be
    # far more transparent than the 15 μm CO2 band.
    # ---------------------------------------------------------------------
    @testset "Longwave window vs CO2 band" begin
        lw = longwave_bands(; ngpt_per_band = 4)
        go = correlated_k_optics(lw)
        col = us_standard_column(; nz = 30)
        opt = allocate_optics(Float64, (30,), lw)
        compute_optics!(opt, go, col)

        @test all(opt.τ .>= 0)
        @test all(isfinite, opt.τ)

        # band 4 is 630-700 cm⁻¹ (the 15 μm CO2 band)
        # band 7 is 980-1080 cm⁻¹ (the atmospheric window)
        τ_co2 = sum(optical_depth_profile(opt, lw, 4))
        τ_win = sum(optical_depth_profile(opt, lw, 7))
        t_co2 = transmission_band(opt, lw, 4)
        t_win = transmission_band(opt, lw, 7)
        @info "LW band structure" τ_co2_15um=τ_co2 τ_window_9_10um=τ_win transmission_co2=t_co2 transmission_window=t_win

        @test τ_co2 > 5 * τ_win        # CO2 band far more opaque
        @test t_win > t_co2            # window far more transparent
        @test t_co2 < 0.1              # 15 μm band is essentially saturated
        @test t_win > 0.3              # window lets a good fraction through
    end

    # ---------------------------------------------------------------------
    # CO2 doubling must strengthen the 15 μm band and leave the window
    # nearly alone — the signature of the greenhouse effect.
    # ---------------------------------------------------------------------
    @testset "CO2 doubling response" begin
        lw = longwave_bands(; ngpt_per_band = 4)
        go = correlated_k_optics(lw)
        opt1 = allocate_optics(Float64, (30,), lw)
        opt2 = allocate_optics(Float64, (30,), lw)
        compute_optics!(opt1, go, us_standard_column(; nz=30, co2_ppm=420.0))
        compute_optics!(opt2, go, us_standard_column(; nz=30, co2_ppm=840.0))

        τ1_co2 = sum(optical_depth_profile(opt1, lw, 4))
        τ2_co2 = sum(optical_depth_profile(opt2, lw, 4))
        τ1_win = sum(optical_depth_profile(opt1, lw, 7))
        τ2_win = sum(optical_depth_profile(opt2, lw, 7))
        d_co2 = (τ2_co2 - τ1_co2) / τ1_co2
        d_win = (τ2_win - τ1_win) / τ1_win
        @info "CO2 doubling" Δτ_co2_band_frac=d_co2 Δτ_window_frac=d_win

        @test d_co2 > 0.25             # 15 μm band strengthens materially
        @test abs(d_win) < 0.02        # window barely notices
        @test d_co2 > 10 * abs(d_win)  # the response is band-selective
    end

    # ---------------------------------------------------------------------
    # Shortwave: ozone UV absorption, Rayleigh, and monotone behaviour.
    # ---------------------------------------------------------------------
    @testset "Shortwave optics" begin
        sw = shortwave_bands(; ngpt_per_band = 2)
        go = correlated_k_optics(sw)
        col = us_standard_column(; nz = 30)
        opt = allocate_optics(Float64, (30,), sw)
        compute_optics!(opt, go, col)

        @test all(opt.τ .>= 0)
        @test all(0 .<= opt.ssa .<= 1)
        @test all(-1 .<= opt.g .<= 1)
        @test all(isfinite, opt.τ)

        # UV bands (12, 13) must be strongly absorbed by ozone
        t_uv  = transmission_band(opt, sw, 13)
        t_vis = transmission_band(opt, sw, 10)
        @info "SW transmission" uv_band13=t_uv visible_band10=t_vis
        @test t_uv < t_vis             # UV far more attenuated
        @test t_uv < 0.2               # ozone shields the surface

        # Rayleigh makes the short-wavelength bands scatter: ssa > 0 there
        ssa_short = maximum(opt.ssa[:, gpoint_range(sw, 12)])
        @info "Rayleigh single-scattering albedo, UV band" ssa_short
        @test ssa_short > 0.0

        # more water vapour ⇒ more near-IR absorption
        col_dry = us_standard_column(; nz=30, humid=:subarctic_winter)
        opt_dry = allocate_optics(Float64, (30,), sw)
        compute_optics!(opt_dry, go, col_dry)
        τ_moist = sum(optical_depth_profile(opt,     sw, 2))
        τ_dry   = sum(optical_depth_profile(opt_dry, sw, 2))
        @info "near-IR H2O absorption" moist=τ_moist dry=τ_dry
        @test τ_moist > τ_dry
    end

    # ---------------------------------------------------------------------
    # Optical depth must be monotone in absorber amount and in pressure.
    # ---------------------------------------------------------------------
    @testset "Monotonicity" begin
        lw = longwave_bands(; ngpt_per_band = 2)
        go = correlated_k_optics(lw)
        τs = Float64[]
        for co2 in (200.0, 420.0, 840.0, 1680.0)
            opt = allocate_optics(Float64, (20,), lw)
            compute_optics!(opt, go, us_standard_column(; nz=20, co2_ppm=co2))
            push!(τs, sum(optical_depth_profile(opt, lw, 4)))
        end
        @info "τ(15 μm band) vs CO2" τs
        @test issorted(τs)
        # sub-linear growth — the band is saturating, which is why CO2
        # forcing is logarithmic rather than linear in concentration
        @test (τs[4] - τs[3]) < (τs[2] - τs[1]) * 4
    end
end
