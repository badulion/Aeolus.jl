# Era IV, Phase 44 — visibility diagnostic.
#
# Koschmieder visibility V = −ln ε / β_ext with β_ext built from droplet
# Mie extinction, κ-Köhler haze (AerosolOptics) and molecular Rayleigh
# scattering (GasOptics). Validated against analytic limits and the
# Kunkel (1984) and Gultepe et al. (2006) fog parameterisations.

using Test
using Aeolus

@testset "Era IV — Visibility diagnostic (Phase 44)" begin

    @testset "Koschmieder's law" begin
        # the two textbook constants
        @test koschmieder_visibility(1.0) ≈ 3.912 atol = 5e-4
        @test koschmieder_visibility(1.0; ε = MOR_THRESHOLD) ≈ 2.996 atol = 5e-4
        β = 0.0137
        @test extinction_from_visibility(koschmieder_visibility(β)) ≈ β rtol = 1e-14
        @test exp(-β * koschmieder_visibility(β)) ≈ KOSCHMIEDER_THRESHOLD rtol = 1e-12
        @test koschmieder_visibility(0.0) == Inf
        # review fix: NaN extinction used to come back as Inf (perfectly clear)
        @test isnan(koschmieder_visibility(NaN))
    end

    @testset "Q_ext → 2 for large water drops" begin
        λ = VISIBILITY_WAVELENGTH
        Q = [water_extinction_efficiency(r, λ) for r in (1e-5, 1e-3, 5e-3)]
        @test all(>(2), Q) && issorted(Q; rev = true)
        # the approach is slow: edge (Fock) term 1.99 x^{-2/3} (1.3e-3 at 5 mm;
        # a first-draft expectation of < 1e-3 there was wrong)
        x5 = 2π * 5e-3 / (λ * 1e-6)
        @test Q[end] - 2 ≈ 1.9923861 * x5^(-2 / 3) rtol = 1e-12
        # continuity across the Mie → asymptote switch at x = 1e4
        r_sw = 1e4 * λ * 1e-6 / (2π)
        @test abs(water_extinction_efficiency(r_sw * (1 + 1e-6)) /
                  water_extinction_efficiency(r_sw * (1 - 1e-6)) - 1) < 1e-4
        # Rayleigh regime: Q ≪ 2
        @test water_extinction_efficiency(1e-8) < 1e-3
    end

    @testset "droplet spectra: analytic β and moments" begin
        N, r = 1.0e8, 8e-6
        Q = mie_sphere(2π * r / 0.55e-6, water_refractive_index(0.55)).Q_ext
        @test droplet_extinction(MonodisperseDroplets(N, r)) ≈ N * π * r^2 * Q rtol = 1e-12
        # geometric limit 3 LWC / (2 ρ_w r) at the volume radius
        lwc = 2e-4
        m = gamma_droplets(lwc, N; v_eff = 0)
        @test m isa MonodisperseDroplets
        @test droplet_extinction(m; method = :geometric) ≈ 3lwc / (2 * RHO_WATER * m.r) rtol = 1e-12
        # gamma: r_eff = ⟨r³⟩/⟨r²⟩, LWC, effective variance, shape factor
        v = 0.2
        s = gamma_droplets(lwc, N; v_eff = v)
        m2, m3, m4 = spectrum_moment(s, 2), spectrum_moment(s, 3), spectrum_moment(s, 4)
        @test m3 / m2 ≈ s.r_eff rtol = 1e-12
        @test spectrum_water_content(s) ≈ lwc rtol = 1e-12
        @test (m4 - 2s.r_eff * m3 + s.r_eff^2 * m2) / (s.r_eff^2 * m2) ≈ v rtol = 1e-10
        @test (droplet_volume_radius(lwc, N) / s.r_eff)^3 ≈ (1 - v) * (1 - 2v) rtol = 1e-12
        # Broad spectra (review regression): normalising by the numerically
        # integrated number inflated the effective Q to 6.57 at v_eff = 0.49,
        # r_eff = 10 μm, where physics requires ≈ 2.1.
        g = GammaDroplets(1e8, 1e-5, 0.49)
        @test 2 < droplet_extinction(g) / (π * 1e8 * spectrum_moment(g, 2)) < 2.2
        # superdroplet ensemble → binned spectrum
        sd = init_monodisperse_superdroplets(100, 1e8, 1e-5, 1e-6)
        spec = droplet_spectrum(sd; volume = 1e-6)
        @test droplet_extinction(spec) ≈ droplet_extinction(MonodisperseDroplets(1e8, 1e-5)) rtol = 1e-10
    end

    @testset "one source of truth: cloud optical depth and Rayleigh" begin
        # τ = 3 LWP/(2 ρ_w r_eff) over a 1 m slab = geometric β
        s = gamma_droplets(3e-4, 3e8)
        @test droplet_extinction(s; method = :geometric) ≈
              cloud_optical_depth_lwp(cloud_water_path(3e-4, 1.0), s.r_eff) rtol = 1e-12
        # β_R(ρ) × 1 m = GasOptics τ_R over dp = ρ g × 1 m
        p, T = 90000.0, 280.0
        ρ = p / (1.380649e-23 * T) * 28.9647e-3 / 6.02214076e23
        @test rayleigh_extinction(; p = p, T = T) ≈ rayleigh_optical_depth(0.55, ρ * 9.80665) rtol = 1e-12
    end

    @testset "fog: Koschmieder vs Kunkel (1984) and Gultepe et al. (2006)" begin
        lwc, N = 0.2e-3, 100e6
        s = gamma_droplets(lwc, N)
        β = droplet_extinction(s)
        V = koschmieder_visibility(β)
        @test 50 < V < 500                                       # dense fog, ~100 m
        @test 1.02 < β / droplet_extinction(s; method = :geometric) < 1.10
        @test 1 / 1.5 < V / kunkel_visibility(lwc) < 1.5
        @test 1 / 1.5 < V / gultepe_visibility(lwc, N) < 1.5
        @test kunkel_visibility(1e-3) ≈ 27.0                     # their reference points
        @test gultepe_visibility(1e-3, 1e6) ≈ 1002.0
        # V falls with LWC; Twomey for visibility: more droplets, same water
        @test all(diff([koschmieder_visibility(fog_extinction(l, 1e8)) for l in (0.01, 0.1, 0.5) .* 1e-3]) .< 0)
        @test all(diff([koschmieder_visibility(fog_extinction(2e-4, n)) for n in (2e7, 1e8, 1e9)]) .< 0)
        # geometric limit β ∝ N^{1/3} LWC^{2/3}
        b1 = fog_extinction(1e-4, 1e8; method = :geometric)
        @test fog_extinction(8e-4, 1e8; method = :geometric) / b1 ≈ 4 rtol = 1e-12
        @test fog_extinction(1e-4, 8e8; method = :geometric) / b1 ≈ 2 rtol = 1e-12
    end

    @testset "clear air and haze" begin
        βR = rayleigh_extinction()
        VR = koschmieder_visibility(βR)
        @test 250e3 < VR < 400e3                                 # Rayleigh-limited: O(300 km)
        @test rayleigh_extinction(; λ = 0.45) / βR ≈ (0.55 / 0.45)^4 rtol = 1e-12
        clean = aerosol_mixture(:continental_clean)
        Vc = [visibility(; aerosol = clean, rh = rh) for rh in (0.0, 0.8, 0.95)]
        @test all(v -> 20e3 < v < VR, Vc) && all(diff(Vc) .< 0)
        @test 3e3 < visibility(; aerosol = aerosol_mixture(:urban), rh = 0.8) < 30e3
        # contributions add
        s = gamma_droplets(1e-5, 1e8)
        @test total_extinction(; droplets = s, aerosol = clean, rh = 0.8) ≈
              droplet_extinction(s) + aerosol_extinction(clean; rh = 0.8, T_K = 288.15) + βR rtol = 1e-12
        # review fix: one air temperature for Rayleigh AND the haze Kelvin term
        @test total_extinction(; aerosol = clean, rh = 0.9, T = 263.15, rayleigh = false) ≈
              aerosol_extinction(clean; rh = 0.9, T_K = 263.15) rtol = 1e-14
    end
end
