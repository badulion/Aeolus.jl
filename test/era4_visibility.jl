# Era IV, Phase 44 — visibility diagnostic.
#
# Koschmieder visibility V = −ln ε / β_ext with β_ext built from droplet
# Mie extinction, κ-Köhler haze (AerosolOptics) and molecular Rayleigh
# scattering (GasOptics). Validated against analytic limits and the
# Kunkel (1984) and Gultepe et al. (2006) fog parameterisations.

using Test, Printf, Random
using Aeolus

@testset "Era IV — Visibility diagnostic (Phase 44)" begin

    @testset "Koschmieder's law" begin
        @test KOSCHMIEDER_THRESHOLD == 0.02
        @test MOR_THRESHOLD == 0.05
        # the two textbook constants
        @test koschmieder_visibility(1.0) ≈ 3.912 atol = 5e-4
        @test koschmieder_visibility(1.0; ε = MOR_THRESHOLD) ≈ 2.996 atol = 5e-4
        # MOR is shorter by exactly ln20/ln50
        β = 0.0137
        @test koschmieder_visibility(β; ε = MOR_THRESHOLD) /
              koschmieder_visibility(β) ≈ log(20) / log(50) rtol = 1e-14
        # V ∝ 1/β and the inverse round-trips
        @test koschmieder_visibility(2β) ≈ koschmieder_visibility(β) / 2
        for ε in (KOSCHMIEDER_THRESHOLD, MOR_THRESHOLD)
            @test extinction_from_visibility(koschmieder_visibility(β; ε = ε);
                                             ε = ε) ≈ β rtol = 1e-14
        end
        @test koschmieder_visibility(0.0) == Inf
        # contrast at the visibility distance is exactly ε
        V = koschmieder_visibility(β)
        @test exp(-β * V) ≈ KOSCHMIEDER_THRESHOLD rtol = 1e-12
        @test_throws ErrorException koschmieder_visibility(β; ε = 1.5)
    end

    @testset "Q_ext → 2 for large water drops" begin
        λ = VISIBILITY_WAVELENGTH
        Q = [water_extinction_efficiency(r, λ) for r in (1e-5, 1e-4, 1e-3, 5e-3)]
        # always above 2, and approaching it
        @test all(>(2), Q)
        @test issorted(Q; rev = true)
        @test abs(water_extinction_efficiency(1e-2, λ) - 2) < 1e-3
        # The approach is SLOW: the edge (Fock) term 1.99 x^{-2/3} still
        # leaves 1.3e-3 at r = 5 mm (x = 5.7e4). The first draft of this
        # test expected Q − 2 < 1e-3 there; that expectation was wrong.
        x5 = 2π * 5e-3 / (λ * 1e-6)
        @test Q[end] - 2 ≈ 1.9923861 * x5^(-2 / 3) rtol = 1e-12
        # On the Mie side the excess oscillates (interference ripple) about
        # the Nussenzveig-Wiscombe asymptote; a pointwise power-law slope
        # between r = 0.1 and 0.8 mm came out −0.42, not −2/3, for that
        # reason (second wrong first-draft expectation). The ±10 %-in-x
        # average must match the asymptote:
        m = water_refractive_index(λ)
        for x0 in (1000.0, 3000.0, 9000.0)
            xs = range(0.9x0, 1.1x0; length = 2001)
            excess = sum(mie_sphere(x, m).Q_ext - 2 for x in xs) / length(xs)
            @test excess ≈ 1.9923861 * x0^(-2 / 3) rtol = 3e-3
        end
        # continuity across the Mie → asymptote switch at x = 1e4
        r_sw = 1e4 * λ * 1e-6 / (2π)
        Qm = water_extinction_efficiency(r_sw * (1 - 1e-6))
        Qa = water_extinction_efficiency(r_sw * (1 + 1e-6))
        @test abs(Qa - Qm) / Qm < 1e-4
        # small drops are in the Rayleigh regime: Q ≪ 2
        @test water_extinction_efficiency(1e-8) < 1e-3
        @printf("  Q_ext(550 nm): r=10μm %.4f, 100μm %.4f, 1mm %.5f, 5mm %.5f\n",
                Q...)
        @printf("  Mie-vs-asymptote jump at x=1e4: %.2e\n", abs(Qa - Qm) / Qm)
    end

    @testset "monodisperse: analytic β = N π r² Q_ext" begin
        N, r = 1.0e8, 8e-6
        s = MonodisperseDroplets(N, r)
        Q = mie_sphere(2π * r / 0.55e-6, water_refractive_index(0.55)).Q_ext
        @test droplet_extinction(s) ≈ N * π * r^2 * Q rtol = 1e-12
        @test droplet_extinction(s; method = :geometric) ≈ 2π * N * r^2 rtol = 1e-14
        # the monodisperse spectrum from (LWC, N) sits at the volume radius
        lwc = 2e-4
        m = gamma_droplets(lwc, N; v_eff = 0)
        @test m isa MonodisperseDroplets
        @test m.r ≈ droplet_volume_radius(lwc, N) rtol = 1e-14
        @test spectrum_water_content(m) ≈ lwc rtol = 1e-12
        # geometric-limit closed form 3 LWC / (2 ρ_w r)
        @test droplet_extinction(m; method = :geometric) ≈
              3lwc / (2 * RHO_WATER * m.r) rtol = 1e-12
    end

    @testset "β ∝ N r² at fixed Q" begin
        base = droplet_extinction(MonodisperseDroplets(1e8, 5e-6); method = :geometric)
        for fN in (0.5, 3.0), fr in (0.5, 2.0, 7.0)
            β = droplet_extinction(MonodisperseDroplets(1e8fN, 5e-6fr);
                                   method = :geometric)
            @test β / base ≈ fN * fr^2 rtol = 1e-13
        end
        # with Mie, β is exactly linear in N; the r-dependence departs from
        # r² only through Q_ext(x), which the normalised ratio removes
        for r in (2e-6, 7e-6, 2e-5)
            b1 = droplet_extinction(MonodisperseDroplets(1e8, r))
            b3 = droplet_extinction(MonodisperseDroplets(3e8, r))
            @test b3 / b1 ≈ 3 rtol = 1e-13
            @test b1 / (1e8 * π * r^2) ≈ water_extinction_efficiency(r) rtol = 1e-13
        end
        # gamma spectra: β/(πN⟨r²⟩) is the effective Q; → 2 with size
        Qeff = [droplet_extinction(GammaDroplets(1e8, re, 0.1)) /
                (π * 1e8 * spectrum_moment(GammaDroplets(1e8, re, 0.1), 2))
                for re in (3e-6, 1e-5, 3e-5, 1e-4)]
        @test all(>(2), Qeff)
        @test issorted(Qeff; rev = true)
        @test Qeff[end] < 2.02
        @printf("  effective Q_ext, gamma v_eff=0.1: r_eff = 3/10/30/100 μm → %.4f %.4f %.4f %.4f\n",
                Qeff...)
    end

    @testset "gamma spectrum: moments and quadrature" begin
        lwc, N = 2e-4, 1e8
        for v in (0.05, 0.1, 0.2, 0.3)
            s = gamma_droplets(lwc, N; v_eff = v)
            # r_eff = ⟨r³⟩/⟨r²⟩ and the LWC are reproduced exactly
            @test spectrum_moment(s, 3) / spectrum_moment(s, 2) ≈ s.r_eff rtol = 1e-12
            @test spectrum_water_content(s) ≈ lwc rtol = 1e-12
            @test spectrum_moment(s, 0) == 1
            # effective variance: ⟨(r − r_eff)² r²⟩/(r_eff²⟨r²⟩) = v_eff
            m2, m3, m4 = spectrum_moment(s, 2), spectrum_moment(s, 3), spectrum_moment(s, 4)
            @test (m4 - 2s.r_eff * m3 + s.r_eff^2 * m2) / (s.r_eff^2 * m2) ≈ v rtol = 1e-10
            # shape factor k = (r_v/r_eff)^3 = (1-v)(1-2v)
            @test (droplet_volume_radius(lwc, N) / s.r_eff)^3 ≈ (1 - v) * (1 - 2v) rtol = 1e-12
        end
        # independent reference: the same gamma pdf on 40 000 linear bins
        s = GammaDroplets(1e8, 8e-6, 0.1)
        α = (1 - 3 * 0.1) / 0.1; θ = 8e-6 * 0.1
        r = range(1e-8, 60e-6; length = 40_000)
        w = [exp(α * log(ri / θ) - ri / θ) for ri in r]
        ref = BinnedDroplets(collect(r), 1e8 .* w ./ sum(w))
        βq = droplet_extinction(s); βr = droplet_extinction(ref)
        # Narrow Mie resonances of (nearly non-absorbing) water limit any
        # practical quadrature to ~1e-4..1e-3; see the docstring.
        @test βq ≈ βr rtol = 1e-3
        @test spectrum_moment(ref, 2) ≈ spectrum_moment(s, 2) rtol = 1e-4
        for (re, v) in ((2e-6, 0.05), (3e-6, 0.1), (1.5e-5, 0.2))
            g = GammaDroplets(1e8, re, v)
            @test droplet_extinction(g) ≈ droplet_extinction(g; nquad = 60_000) rtol = 2.5e-3
        end
        @printf("  gamma quadrature vs 40k-bin reference: %.2e\n", abs(βq / βr - 1))
        # Broad spectra (v_eff → 0.5, α → −1): most of the NUMBER sits below
        # the quadrature's lower bound. A review found that normalising by
        # the numerically integrated number inflated β there (effective Q
        # 2.55 at v_eff = 0.45, 6.57 at 0.49, for r_eff = 10 μm, where
        # physics requires ≈ 2.1). Reference: an independent 60 000-bin
        # linear-r histogram of the area distribution r² n(r).
        for v in (0.45, 0.49)
            g = GammaDroplets(1e8, 1e-5, v)
            αg = (1 - 3v) / v; θg = 1e-5 * v
            rr = range(1e-9, 40θg * (αg + 3); length = 60_000)
            aw = [exp((αg + 2) * log(ri / θg) - ri / θg) for ri in rr]   # r² n(r) per dr
            Qref = sum(aw[i] * water_extinction_efficiency(rr[i]) for i in eachindex(rr)) / sum(aw)
            Qg = droplet_extinction(g) / (π * 1e8 * spectrum_moment(g, 2))
            @test Qg ≈ Qref rtol = 2e-3
            @test 2 < Qg < 2.2
        end
        # narrow gamma → monodisperse at r_eff
        narrow = GammaDroplets(1e8, 8e-6, 0.002)
        mono = MonodisperseDroplets(1e8, 8e-6)
        @test droplet_extinction(narrow; method = :geometric) ≈
              droplet_extinction(mono; method = :geometric) rtol = 0.01
    end

    @testset "one source of truth: geometric β matches cloud optical depth" begin
        # τ = 3 LWP/(2 ρ_w r_eff) (AerosolCloud / TwoStream) over a 1 m slab
        # must equal the geometric-limit β of the same spectrum.
        for (lwc, N) in ((1e-4, 5e7), (3e-4, 3e8))
            s = gamma_droplets(lwc, N)
            τ1m = cloud_optical_depth_lwp(cloud_water_path(lwc, 1.0), s.r_eff)
            @test droplet_extinction(s; method = :geometric) ≈ τ1m rtol = 1e-12
        end
        # rayleigh_extinction uses the GasOptics cross-section, and the
        # GasOptics column optical depth is the same n σ integrated:
        # β_R(ρ) × (1 m) = τ_R over dp = ρ g × 1 m.
        p, T = 90000.0, 280.0
        m_air = 28.9647e-3 / 6.02214076e23
        ρ = p / (1.380649e-23 * T) * m_air
        @test rayleigh_extinction(; p = p, T = T) ≈
              rayleigh_optical_depth(0.55, ρ * 9.80665) rtol = 1e-12
        @test rayleigh_extinction() ≈ 101325 / (1.380649e-23 * 288.15) *
                                      rayleigh_cross_section(0.55) rtol = 1e-14
    end

    @testset "superdroplet ensemble → binned spectrum" begin
        vol = 1e-6
        sd = init_monodisperse_superdroplets(100, 1e8, 1e-5, vol)
        spec = droplet_spectrum(sd; volume = vol)
        @test spectrum_number(spec) ≈ 1e8 rtol = 1e-10
        @test spectrum_water_content(spec) ≈ liquid_water_content(sd, vol) rtol = 1e-12
        @test droplet_extinction(spec) ≈
              droplet_extinction(MonodisperseDroplets(1e8, 1e-5)) rtol = 1e-10
    end

    @testset "fog: Koschmieder vs Kunkel (1984) and Gultepe et al. (2006)" begin
        N = 100e6                         # 100 cm⁻³
        println("  LWC [g/m³]  r_eff [μm]  β_Mie/β_geo  V_Kosch [m]  V_MOR [m]  Kunkel [m]  Gultepe [m]")
        for lwc_g in (0.1, 0.2, 0.3)
            lwc = lwc_g * 1e-3
            s = gamma_droplets(lwc, N)
            β = droplet_extinction(s)
            βg = droplet_extinction(s; method = :geometric)
            V = koschmieder_visibility(β)
            Vk = kunkel_visibility(lwc)
            Vg = gultepe_visibility(lwc, N)
            @printf("  %9.2f  %10.2f  %11.4f  %11.1f  %9.1f  %10.1f  %11.1f\n",
                    lwc_g, s.r_eff * 1e6, β / βg, V,
                    koschmieder_visibility(β; ε = MOR_THRESHOLD), Vk, Vg)
            # dense fog: ~100 m, well below the 1 km fog definition
            @test 50 < V < 500
            # Mie adds a few percent over the geometric limit for fog drops
            @test 1.02 < β / βg < 1.10
            # the bulk fits agree with explicit Mie within a factor 1.5
            @test 1 / 1.5 < V / Vk < 1.5
            @test 1 / 1.5 < V / Vg < 1.5
        end
        # the fits themselves at their quoted reference points
        @test kunkel_visibility(1e-3) ≈ 27.0             # LWC = 1 g/m³ → 27 m
        @test gultepe_visibility(1e-3, 1e6) ≈ 1002.0     # LWC·N = 1 → 1.002 km
        @test kunkel_visibility(0.0) == Inf
        # Kunkel's 0.027 km is 3.912/144.7: the fit assumes ε = 0.02
        @test 3.912 / 144.7 ≈ 0.027 rtol = 2e-3
    end

    @testset "monotonicity: V falls with LWC and with N_d" begin
        lwcs = (0.005:0.005:0.6) .* 1e-3
        V = [koschmieder_visibility(fog_extinction(l, 1e8)) for l in lwcs]
        @test issorted(V; rev = true)
        @test all(diff(V) .< 0)
        # Twomey for visibility: more droplets, same water → worse visibility
        Vn = [koschmieder_visibility(fog_extinction(2e-4, n)) for n in (2e7, 5e7, 1e8, 3e8, 1e9)]
        @test all(diff(Vn) .< 0)
        # geometric limit scaling β ∝ N^{1/3} LWC^{2/3}
        b1 = fog_extinction(1e-4, 1e8; method = :geometric)
        @test fog_extinction(8e-4, 1e8; method = :geometric) / b1 ≈ 4 rtol = 1e-12
        @test fog_extinction(1e-4, 8e8; method = :geometric) / b1 ≈ 2 rtol = 1e-12
        # Kunkel is LWC-only and also monotone; Gultepe falls with N_d
        @test all(diff([kunkel_visibility(l) for l in lwcs]) .< 0)
        @test gultepe_visibility(2e-4, 3e8) < gultepe_visibility(2e-4, 1e8)
        @test fog_extinction(0.0, 1e8) == 0.0
    end

    @testset "clear air and haze" begin
        βR = rayleigh_extinction()
        VR = koschmieder_visibility(βR)
        # Rayleigh-limited visibility at sea level: O(300 km)
        @test 250e3 < VR < 400e3
        # λ⁻⁴ and ∝ density
        @test rayleigh_extinction(; λ = 0.45) / βR ≈ (0.55 / 0.45)^4 rtol = 1e-12
        @test rayleigh_extinction(; p = 50000.0) / βR ≈ 50000 / 101325 rtol = 1e-12
        @test visibility() ≈ VR
        # clean continental haze: tens to hundreds of km, decreasing with RH
        clean = aerosol_mixture(:continental_clean)
        Vc = [visibility(; aerosol = clean, rh = rh) for rh in (0.0, 0.5, 0.8, 0.95)]
        @test all(v -> 20e3 < v < 400e3, Vc)
        @test all(diff(Vc) .< 0)
        @test all(Vc .< VR)                         # aerosol only removes range
        # urban haze at 80 % RH: order 10 km
        Vu = visibility(; aerosol = aerosol_mixture(:urban), rh = 0.8)
        @test 3e3 < Vu < 30e3
        # contributions add
        s = gamma_droplets(1e-5, 1e8)
        βt = total_extinction(; droplets = s, aerosol = clean, rh = 0.8)
        @test βt ≈ droplet_extinction(s) + aerosol_extinction(clean; rh = 0.8, T_K = 288.15) +
                   βR rtol = 1e-12
        # one air temperature for Rayleigh density AND haze Kelvin term
        # (review fix: the haze term used to ignore T and run at 298.15 K)
        βc = total_extinction(; aerosol = clean, rh = 0.9, T = 263.15, rayleigh = false)
        @test βc ≈ aerosol_extinction(clean; rh = 0.9, T_K = 263.15) rtol = 1e-14
        @test βc != aerosol_extinction(clean; rh = 0.9, T_K = 298.15)
        # a thin fog (LWC 0.01 g/m³) dominates any haze
        @test droplet_extinction(s) > 10 * aerosol_extinction(clean; rh = 0.8)
        @printf("  Rayleigh β=%.3e m⁻¹, V=%.0f km; continental clean RH 0/50/80/95%%: %.0f/%.0f/%.0f/%.0f km; urban RH80: %.1f km\n",
                βR, VR / 1e3, (Vc ./ 1e3)..., Vu / 1e3)
    end
end
