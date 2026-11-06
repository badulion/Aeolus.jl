using Test, Aeolus

using Aeolus.RadiationCore
using Aeolus.GasOptics
using Aeolus.AerosolOptics
using Aeolus.TwoStream

const AO = Aeolus.AerosolOptics

# Phase 37 — aerosol optics. Mie core against Wiscombe (1980) and its two
# analytic limits; size integration against log-normal moment identities;
# κ-Köhler growth against its analytic law; column AOD normalisation and
# the forcing budget identity; critical albedo against Chylek-Coakley.

@testset "Phase 37 — aerosol optics" begin

@testset "Mie: Wiscombe benchmarks and limits" begin
    # Wiscombe (1980), Appl. Opt. 19, 1505 — MIEV0 test cases
    r = mie_sphere(10.0, complex(1.5, 0.0))
    @test isapprox(r.Q_ext, 2.881999; rtol = 1e-5)
    @test isapprox(r.g, 0.742913; rtol = 1e-5)
    @test abs(r.Q_abs) < 1e-12 && isapprox(r.Q_sca, r.Q_ext; rtol = 1e-12)
    @test isapprox(mie_sphere(0.101, complex(0.75, 0.0)).Q_ext, 8.033542e-6; rtol = 1e-5)
    r3 = mie_sphere(100.0, complex(1.5, 0.0))
    @test isapprox(r3.Q_ext, 2.094387; rtol = 1e-5) && isapprox(r3.g, 0.818243; rtol = 1e-4)
    @test isapprox(mie_sphere(1000.0, complex(1.5, 0.0)).Q_ext, 2.013079; rtol = 1e-5)

    # Rayleigh limit: Q_sca/x⁴ = 8/3 |(m²-1)/(m²+2)|², g → 0
    m = complex(1.5, 0.0)
    theory = 8 / 3 * abs2((m^2 - 1) / (m^2 + 2))
    qs = [mie_sphere(x, m) for x in (2e-3, 1e-2, 5e-2)]
    @test maximum(abs(q.Q_sca / x^4 / theory - 1) for (q, x) in zip(qs, (2e-3, 1e-2, 5e-2))) < 1e-2
    @test all(q -> 0 <= q.g < 1e-3, qs)
    # the analytic branch below x = 1e-3 joins the series smoothly
    mm = complex(1.75, 0.44)
    @test isapprox(mie_sphere(0.9999e-3, mm).Q_ext, mie_sphere(1.0001e-3, mm).Q_ext; rtol = 1e-3)
    # extinction paradox: Q_ext → 2 for large spheres
    @test 1.9 < mie_sphere(5000.0, complex(1.33, 0.0)).Q_ext < 2.15
    # absorbing sphere: Q_abs > 0, Q_sca < Q_ext
    @test all(x -> (q = mie_sphere(x, mm); q.Q_abs > 0 && 0 <= q.Q_sca < q.Q_ext && -1 <= q.g <= 1),
              (0.05, 2.0, 50.0))
    @test_throws ErrorException mie_sphere(1.0, complex(1.5, -0.01))
end

@testset "Refractive indices" begin
    @test all(real(refractive_index(aerosol_species(n), λ)) > 1 && imag(refractive_index(aerosol_species(n), λ)) >= 0
              for n in aerosol_species_names(), λ in (0.25, 0.55, 3.0, 30.0))
    @test isapprox(real(water_refractive_index(0.55)), 1.333; atol = 0.002)
    @test imag(water_refractive_index(3.0)) > 0.1                       # 3 μm OH stretch
    @test imag(refractive_index(aerosol_species(:black_carbon), 0.55)) > 0.3
    @test imag(refractive_index(aerosol_species(:sulphate), 0.55)) < 1e-6
    @test_throws ErrorException aerosol_species(:unobtainium)
end

@testset "Size distribution identities" begin
    m = lognormal_mode(:sulphate, 1000.0, 0.05, 1.8)
    lnσ = log(1.8)
    a = log(0.05) - 8lnσ; h = 16lnσ / 4000
    s = sum(((i == 0 || i == 4000) ? 0.5 : 1.0) * number_pdf(m, exp(a + i * h)) * h for i in 0:4000)
    @test isapprox(s, 1000.0; rtol = 1e-8)
    @test isapprox(effective_radius(m), 0.05 * exp(2.5lnσ^2); rtol = 1e-12)
    @test isapprox(mode_mass(m), 1000.0 * (4π / 3) * 0.05^3 * exp(4.5lnσ^2) * 1.7; rtol = 1e-12)
    # extinction is linear in N; the default quadrature is converged
    e1 = bulk_optics([m], 0.55)[1]
    @test isapprox(bulk_optics([lognormal_mode(:sulphate, 3000.0, 0.05, 1.8)], 0.55)[1], 3 * e1; rtol = 1e-12)
    urb = aerosol_mixture(:urban)
    @test abs(bulk_optics(urb, 0.55)[1] / bulk_optics(urb, 0.55; nquad = 7680)[1] - 1) < 3e-3
    # β_ext = β_sca + β_abs with 0 ≤ ω₀ ≤ 1 for every species and wavelength
    @test all(aerosol_species_names()) do n
        all((0.3, 1.6, 11.0)) do λ
            βe, βs, βa, ω, g = bulk_optics([lognormal_mode(n, 1000.0, 0.05, 2.0)], λ)
            βa >= 0 && 0 <= ω <= 1 && -1 <= g <= 1 && isapprox(βe, βs + βa; rtol = 1e-10)
        end
    end
end

@testset "Species optics" begin
    bc   = [lognormal_mode(:black_carbon, 1e4, 0.0118, 2.00)]
    sul  = [lognormal_mode(:sulphate, 1e3, 0.0212, 2.24)]
    dust = [lognormal_mode(:mineral_dust, 30.5, 0.39, 2.00)]
    @test 0.10 < bulk_optics(bc, 0.55)[4] < 0.4
    @test bulk_optics(sul, 0.55)[4] > 0.98
    @test 0.7 < bulk_optics(dust, 0.55)[4] < 0.99
    # BC mass absorption cross-section: Bond & Bergstrom (2006) 7.5 ± 1.2 m²/g
    @test 5.0 < AO.mass_absorption_coefficient(bc, 0.55) < 11.0
    # Ångström: fine smoke > 1.5, coarse dust < 0.5
    @test angstrom_exponent([lognormal_mode(:organic_carbon, 5e3, 0.08, 1.6)], 0.44, 0.87) > 1.5
    @test angstrom_exponent([lognormal_mode(:mineral_dust, 1.0, 1.9, 2.15)], 0.44, 0.87) < 0.5
    @test length(AEROSOL_MIXTURE_NAMES) == 7
end

@testset "Hygroscopic growth (κ-Köhler)" begin
    κ = aerosol_species(:sea_salt).κ
    @test aerosol_species(:black_carbon).κ == 0 && growth_factor(0.1, 0.99, 0.0) == 1.0
    # Large-particle law GF = (1 + κ RH/(1-RH))^(1/3) drops the Kelvin term, so
    # it is an upper bound (deficit 0.26 % at RH 0.95, r = 1 μm) that the
    # solver approaches as the particle grows.
    for rh in (0.8, 0.95)
        law = (1 + κ * rh / (1 - rh))^(1 / 3)
        gf = growth_factor(1.0, rh, κ)
        @test gf < law && isapprox(gf, law; rtol = 5e-3)
        @test isapprox(growth_factor(100.0, rh, κ), law; rtol = 1e-4)
    end
    @test maximum(abs(kohler_saturation(wet_radius(0.1, rh, k), 0.1, k) / rh - 1)
                  for k in (0.1, 1.1), rh in (0.5, 0.99)) < 1e-6
    # volume mixing of m towards water
    md = refractive_index(aerosol_species(:sea_salt), 0.55)
    @test isapprox(real(wet_refractive_index(md, 0.1, 0.25, 0.55)),
                   0.4^3 * real(md) + (1 - 0.4^3) * real(water_refractive_index(0.55)); rtol = 1e-12)
    ss = [lognormal_mode(:sea_salt, 20.0, 0.209, 2.03)]
    @test bulk_optics(ss, 0.55; rh = 0.99)[1] > 8 * bulk_optics(ss, 0.55)[1]
end

# Column tests use small fine-mode populations: the band Mie integration of a
# coarse mode (sea salt, dust) costs seconds per profile and tests nothing
# extra about the profile, AOD or forcing algebra.
sw = shortwave_bands(Float64; ngpt_per_band = 2)
nz = 30
z = [(k - 0.5) * 1000.0 for k in 1:nz]
col = us_standard_column(Float64; nz = nz, z_top = 30000.0)
sul = [lognormal_mode(:sulphate, 1e3, 0.0212, 2.24)]
bc  = [lognormal_mode(:black_carbon, 1.3e5, 0.0118, 2.0)]
mix = [lognormal_mode(:sulphate, 1e3, 0.0212, 2.24), lognormal_mode(:black_carbon, 3e3, 0.0118, 2.0)]

@testset "Column profile, AOD normalisation, compute_optics! contract" begin
    p = aerosol_profile(sul, nz, z; aod_550 = 0.3, scale_height = 1500.0, spec = sw)
    @test isapprox(aerosol_optical_depth_550(p), 0.3; rtol = 1e-10)
    β = aerosol_extinction_profile(p)
    @test isapprox(β[2] / β[1], exp(-1000.0 / 1500.0); rtol = 1e-8)
    @test aerosol_optical_depth(p, 10) > 0.3 > aerosol_optical_depth(p, 8)
    pe = aerosol_profile(sul, nz, z; aod_550 = 0.6, scale_height = 1000.0, spec = sw,
                         elevated_layer = (3000.0, 6000.0), elevated_fraction = 0.6)
    @test isapprox(aerosol_optical_depth_550(pe), 0.6; rtol = 1e-10)
    @test aerosol_extinction_profile(pe)[4] > aerosol_extinction_profile(pe)[3]
    lw = longwave_bands(Float64; ngpt_per_band = 2)
    @test isapprox(aerosol_optical_depth_550(rebuild_bands(p, lw)), 0.3; rtol = 1e-12)

    # overwrite reproduces the table; additive mode == combine_optics!
    gas = correlated_k_optics(sw)
    oa = allocate_optics(Float64, (nz,), sw)
    compute_optics!(oa, p, col, sw; add = false)
    @test isapprox(sum(@view oa.τ[:, first(gpoint_range(sw, 10))]), aerosol_optical_depth(p, 10); rtol = 1e-12)
    o1 = allocate_optics(Float64, (nz,), sw); o2 = allocate_optics(Float64, (nz,), sw)
    compute_optics!(o1, gas, col); compute_optics!(o2, gas, col)
    τgas = copy(o1.τ)
    compute_optics!(o1, p, col, sw)
    combine_optics!(o2, oa)
    @test maximum(abs, o1.τ .- o2.τ) < 1e-14 && maximum(abs, o1.ssa .- o2.ssa) < 1e-14
    @test all(o1.τ .>= τgas .- 1e-15)
    @test_throws ErrorException compute_optics!(allocate_optics(Float64, (nz,), lw), p, col, lw)
end

@testset "Radiative forcing and critical albedo" begin
    μ₀ = 0.5
    p_sca = aerosol_profile(sul, nz, z; aod_550 = 0.2, scale_height = 1000.0, spec = sw)
    f = aerosol_radiative_forcing(p_sca, col; μ₀ = μ₀, sfc_albedo = 0.05)
    @test f.toa < 0 && f.surface < 0 && abs(f.atmosphere) < 0.1 * abs(f.toa)   # measured −25.3, 0.28 W/m²
    @test isapprox(f.atmosphere, f.toa - f.surface; rtol = 1e-12)              # budget identity
    p_bc = aerosol_profile(bc, nz, z; aod_550 = 0.2, scale_height = 1000.0, spec = sw)
    fb = aerosol_radiative_forcing(p_bc, col; μ₀ = μ₀, sfc_albedo = 0.05)
    @test fb.atmosphere > 50 && fb.surface < -50                               # measured 134, −127 W/m²
    hr = aerosol_heating_rate(p_bc, col; μ₀ = μ₀, sfc_albedo = 0.05)
    @test hr[1] == maximum(hr) > 1.0 && all(hr[20:end] .< 0.05)

    # critical surface albedo vs the Chylek & Coakley (1974) estimate from the
    # solar-weighted ω₀, g of the same profile (measured 0.153 vs 0.163, ω̄ = 0.71)
    p = aerosol_profile(mix, nz, z; aod_550 = 0.3, scale_height = 1500.0, spec = sw)
    αc = critical_surface_albedo(p, col; μ₀ = μ₀)
    wsum = ω̄ = ḡ = 0.0
    for b in 1:nbands(sw)
        s = sum(sw.solar_src[g] for g in gpoint_range(sw, b)) * aerosol_optical_depth(p, b)
        wsum += s; ω̄ += s * p.ssa[1, b]; ḡ += s * p.g[1, b]
    end
    ω̄ /= wsum; ḡ /= wsum
    bq = 2 + 2 * (1 - ω̄) / ((1 - ḡ) / 2 * ω̄)
    @test 0.05 < αc < 0.45 && isapprox(αc, (bq - sqrt(bq^2 - 4)) / 2; rtol = 0.5)
    @test aerosol_radiative_forcing(p, col; μ₀ = μ₀, sfc_albedo = αc - 0.05).toa < 0 <
          aerosol_radiative_forcing(p, col; μ₀ = μ₀, sfc_albedo = αc + 0.05).toa
end

end # Phase 37
