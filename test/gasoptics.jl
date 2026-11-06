using Test
using Aeolus
using Aeolus.RadiationCore
using Aeolus.GasOptics

@testset "Era III — Correlated-k gas optics (Phase 32)" begin

    # Own erf/erfc (no SpecialFunctions): tabulated 16-digit values.
    @testset "Error function" begin
        ref = ((erf_series, 0.5, 0.5204998778130465), (erf_series, 2.0, 0.9953222650189527),
               (erfc_series, 2.0, 0.004677734981047266), (erfc_series, 6.0, 2.1519736712498913e-17))
        @test all(isapprox(f(x), v; rtol = 1e-11) for (f, x, v) in ref)
        @test maximum(abs(erf_series(x) + erfc_series(x) - 1) for x in (0.1, 1.5, 3.5, 5.0)) < 1e-13
    end

    # Malkmus: weak-line exp(-ku), strong-line exp(-√(kuB)), and pressure broadening.
    @testset "Malkmus weak- and strong-line limits" begin
        p = MalkmusParams(1.0, 1.0, 0.5)
        @test maximum(abs(malkmus_transmission(u, p) / exp(-u) - 1) for u in (1e-6, 1e-4)) < 1e-4
        # log space: T underflows to 0 at these path amounts
        @test maximum(abs(malkmus_log_transmission(u, p) / -sqrt(u) - 1) for u in (1e4, 1e6)) < 1e-2
        ts = [malkmus_transmission(u, p) for u in 10.0 .^ (-6:1:6)]
        @test all(0 .<= ts .<= 1) && all(diff(ts) .< 0)
        @test malkmus_transmission(10.0, p; pressure = 1e5) < malkmus_transmission(10.0, p; pressure = 2e4)
    end

    # The correlated-k quadrature reproduces band transmission over 8 decades,
    # converges with g-points and beats a single band-mean k by >10x.
    @testset "Correlated-k reproduces band transmission" begin
        p = MalkmusParams(2.0, 0.3, 0.5)
        us = 10.0 .^ (-4:0.5:4)
        err(ngpt) = begin
            kg, wg = kdist_from_band_model(p, ngpt)
            @test isapprox(sum(wg), 1.0; atol = 1e-14) && issorted(kg) && all(kg .>= 0)
            maximum(abs(sum(wg .* exp.(-kg .* u)) - malkmus_transmission(u, p)) for u in us)
        end
        e4, e16 = err(4), err(16)
        @test e16 < e4 && e16 < 0.02
        @test maximum(abs(exp(-p.k * u) - malkmus_transmission(u, p)) for u in us) > 10 * e16
    end

    @testset "Rayleigh and Planck" begin
        dp = 1.01325e5
        @test 0.085 < rayleigh_optical_depth(0.55, dp) < 0.11      # published ≈ 0.0973
        n = -log(rayleigh_optical_depth(0.85, dp) / rayleigh_optical_depth(0.35, dp)) / log(0.85 / 0.35)
        @test 3.9 < n < 4.1                                        # λ⁻⁴
        # Σ bands of the surface source recovers σT⁴ up to the 10-3250 cm⁻¹ truncation
        go = correlated_k_optics(longwave_bands(; ngpt_per_band = 2))
        s = sum(lw_surface_source(go, 288.0))
        @test abs(s / planck_integrated(288.0) - 1) < 2e-3
        @test isapprox(sum(lw_surface_source(go, 300.0)) / s, (300 / 288)^4; rtol = 5e-3)
    end

    # LW spectral structure: 8-12 μm window vs 15 μm CO2 band, CO2 doubling,
    # and ozone UV shielding in the shortwave.
    @testset "Band structure and CO2 doubling" begin
        col = us_standard_column(; nz = 30)
        pw = sum(column_absorber_amount(col, :h2o, k) for k in 1:30)
        @test 10 < pw < 20                                  # midlatitude summer ≈ 14 kg/m²
        lw = longwave_bands(; ngpt_per_band = 4)
        go = correlated_k_optics(lw)
        τband(ppm, b) = (o = allocate_optics(Float64, (30,), lw);
                         compute_optics!(o, go, us_standard_column(; nz = 30, co2_ppm = ppm));
                         (sum(optical_depth_profile(o, lw, b)), transmission_band(o, lw, b)))
        (τc, tc), (τw, tw) = τband(420.0, 4), τband(420.0, 7)
        @test τc > 5τw && tc < 0.1 && tw > 0.3
        dco2 = τband(840.0, 4)[1] / τc - 1
        dwin = τband(840.0, 7)[1] / τw - 1
        @test dco2 > 0.25 && abs(dwin) < 0.02

        sw = shortwave_bands(; ngpt_per_band = 2)
        o = allocate_optics(Float64, (30,), sw)
        compute_optics!(o, correlated_k_optics(sw), col)
        @test all(o.τ .>= 0) && all(0 .<= o.ssa .<= 1)
        @test transmission_band(o, sw, 13) < min(0.2, transmission_band(o, sw, 10))
    end

    # #130: an LES column stitched onto the standard column must give the
    # standard column's fluxes, to the vertical-resolution difference.
    @testset "column_from_profiles and stitch_columns" begin
        std = us_standard_column(; nz = 60, T_sfc = 293.0)
        dz = 6.25; nz = 32
        Tz = 293.0 .- 6.5e-3 .* [(k - 0.5) * dz for k in 1:nz]
        q  = std.vmr[:h2o][1] * 18.01528 / 28.9647
        θ = copy(Tz)
        for _ in 1:5
            θ .*= Tz ./ column_from_profiles(θ, dz; qv = fill(q, nz)).T_lay
        end
        low = column_from_profiles(θ, dz; qv = fill(q, nz), T_sfc = 293.0)
        T̄ = 293.0 - 6.5e-3 * 100           # hydrostatic p(200 m)
        @test isapprox(low.p_lev[end], 1.01325e5 * exp(-9.80665 * 200 / (287.05 * T̄)); rtol = 1e-4)
        st = stitch_columns(low, std)
        @test sum(st.dz) ≈ sum(std.dz) rtol = 1e-12
        @test st.p_lev[nz + 1] == low.p_lev[end] && all(diff(st.p_lev) .< 0)

        function fluxes(col, spec)
            o = allocate_optics(Float64, (length(col.dz),), spec)
            compute_optics!(o, correlated_k_optics(spec), col)
            f = allocate_fluxes(Float64, (), length(col.dz) + 1)
            spec.kind === :longwave ?
                solve_lw!(f, two_stream_solver(spec), o, col; sfc_emissivity = 1.0) :
                solve_sw!(f, two_stream_solver(spec), o; μ₀ = 0.87, sfc_albedo_direct = 0.15)
        end
        l1, l2 = fluxes(std, longwave_bands()), fluxes(st, longwave_bands())
        s1, s2 = fluxes(std, shortwave_bands()), fluxes(st, shortwave_bands())
        @test abs(l2.flux_dn[1] - l1.flux_dn[1]) < 0.5        # measured 0.02 W/m²
        @test abs(l2.flux_up[end] - l1.flux_up[end]) < 0.5    # measured 0.001 W/m²
        @test abs(s2.flux_dn[1] - s1.flux_dn[1]) < 1.0        # measured 0.12 W/m²
        @test 300 < l2.flux_dn[1] < 380
    end
end
