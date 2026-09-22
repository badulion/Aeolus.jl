using Test, Aeolus

using Aeolus.RadiationCore
using Aeolus.GasOptics
using Aeolus.AerosolOptics
using Aeolus.TwoStream

const AO = Aeolus.AerosolOptics

# =========================================================================
# Phase 37 — Aerosol and gas radiative feedback
#
# The validation strategy is layered, because each layer can be checked
# against something external:
#
#   Mie core          → Wiscombe (1980) MIEV0 published benchmarks, plus
#                       the two analytic limits (Rayleigh, geometric).
#   Refractive index  → sign and magnitude of absorption per species.
#   Size integration  → moment identities of the log-normal, and
#                       self-convergence of the quadrature.
#   Hygroscopic       → the analytic κ-Köhler growth law.
#   Bulk optics       → OPAC/observational ω₀, MAC and Ångström values.
#   Column + RT       → exact AOD normalisation, the ΔF_atm identity, and
#                       the Chylek-Coakley critical-albedo formula.
#
# Every quantitative number printed with @info below is a measurement
# from this implementation, not a target.
# =========================================================================

@testset "Phase 37 — aerosol optics" begin

# -------------------------------------------------------------------------
@testset "Mie: Wiscombe benchmarks" begin
    # Wiscombe (1980), Appl. Opt. 19, 1505 — MIEV0 test cases. These are
    # hard external numbers; the solver is wrong if they do not appear.
    r = mie_sphere(10.0, complex(1.5, 0.0))
    @info "Mie benchmark x=10, m=1.5" Qext=r.Q_ext Qsca=r.Q_sca g=r.g nterms=r.nterms
    @test isapprox(r.Q_ext, 2.8820; atol = 1e-3)      # the named benchmark
    @test isapprox(r.Q_ext, 2.881999; rtol = 1e-5)
    @test isapprox(r.g, 0.742913; rtol = 1e-5)
    @test r.Q_abs ≈ 0.0 atol = 1e-12                  # m real ⇒ no absorption
    @test isapprox(r.Q_sca, r.Q_ext; rtol = 1e-12)

    r2 = mie_sphere(0.101, complex(0.75, 0.0))
    @info "Mie benchmark x=0.101, m=0.75" Qext=r2.Q_ext g=r2.g
    @test isapprox(r2.Q_ext, 8.033542e-6; rtol = 1e-5)
    @test r2.g > 0 && r2.g < 0.01

    r3 = mie_sphere(100.0, complex(1.5, 0.0))
    @info "Mie benchmark x=100, m=1.5" Qext=r3.Q_ext g=r3.g
    @test isapprox(r3.Q_ext, 2.094387; rtol = 1e-5)
    @test isapprox(r3.g, 0.818243; rtol = 1e-4)

    r4 = mie_sphere(1000.0, complex(1.5, 0.0))
    @info "Mie benchmark x=1000, m=1.5" Qext=r4.Q_ext g=r4.g
    @test isapprox(r4.Q_ext, 2.013079; rtol = 1e-5)
end

# -------------------------------------------------------------------------
@testset "Mie: Rayleigh limit" begin
    m = complex(1.5, 0.0)
    μ = (m^2 - 1) / (m^2 + 2)
    theory = 8 / 3 * abs2(μ)                 # Q_sca / x⁴
    xs = [1.0e-3, 2.0e-3, 5.0e-3, 1.0e-2, 2.0e-2, 5.0e-2]
    ratios = Float64[]
    for x in xs
        q = mie_sphere(x, m)
        push!(ratios, q.Q_sca / x^4)
        @test q.g < 1e-3                     # g → 0 in the dipole limit
        @test q.g >= 0
    end
    @info "Rayleigh limit: Q_sca/x⁴ (theory $(round(theory; sigdigits=6)))" x=xs ratio=round.(ratios; sigdigits=6)
    for (x, ρ) in zip(xs, ratios)
        @test isapprox(ρ, theory; rtol = 1e-2)
    end
    # x⁴ scaling: doubling x must multiply Q_sca by 16
    q1 = mie_sphere(2.0e-3, m); q2 = mie_sphere(4.0e-3, m)
    @test isapprox(q2.Q_sca / q1.Q_sca, 16.0; rtol = 1e-3)

    # The analytic branch below x = 1e-3 must join the series smoothly.
    for mm in (complex(1.5, 0.0), complex(1.75, 0.44), complex(1.53, 0.0037))
        below = mie_sphere(0.9999e-3, mm)     # analytic branch
        above = mie_sphere(1.0001e-3, mm)     # Mie series branch
        @test isapprox(below.Q_ext, above.Q_ext; rtol = 1e-3)
        @test isapprox(below.Q_sca, above.Q_sca; rtol = 1e-3)
    end
    ray = AO.rayleigh_efficiencies(5.0e-3, complex(1.75, 0.44))
    ser = mie_sphere(5.0e-3, complex(1.75, 0.44))
    @info "Rayleigh vs Mie series at x=5e-3, m=1.75+0.44i" Qext_analytic=ray.Q_ext Qext_series=ser.Q_ext
    @test isapprox(ray.Q_ext, ser.Q_ext; rtol = 1e-3)
end

# -------------------------------------------------------------------------
@testset "Mie: geometric-optics limit and absorption" begin
    # The extinction paradox: Q_ext → 2, not 1, as x → ∞.
    qs = [(x, mie_sphere(x, complex(1.33, 0.0)).Q_ext)
          for x in (10.0, 100.0, 1000.0, 5000.0)]
    @info "Extinction paradox, m=1.33" x=first.(qs) Qext=round.(last.(qs); digits=4)
    @test last(qs[end]) > 1.9
    @test last(qs[end]) < 2.15
    @test abs(last(qs[end]) - 2) < abs(last(qs[1]) - 2)   # converging to 2

    # An absorbing sphere: Q_abs > 0 and Q_sca < Q_ext strictly.
    for x in (0.05, 0.5, 2.0, 10.0, 50.0)
        q = mie_sphere(x, complex(1.75, 0.44))
        @test q.Q_abs > 0
        @test q.Q_sca < q.Q_ext
        @test q.Q_sca >= 0
        @test -1 <= q.g <= 1
        @test q.Q_back >= 0
    end
    # Wrong sign convention must be rejected, not silently mis-computed.
    @test_throws ErrorException mie_sphere(1.0, complex(1.5, -0.01))
    @test_throws ErrorException mie_sphere(1.0, complex(-1.5, 0.01))
    @test mie_sphere(0.0, complex(1.5, 0.0)).Q_ext == 0.0
    @test size_parameter(0.55 / (2π), 0.55) ≈ 1.0
end

# -------------------------------------------------------------------------
@testset "Refractive indices" begin
    for name in aerosol_species_names()
        sp = aerosol_species(name)
        @test sp isa AerosolSpecies
        @test sp.ρ > 0
        @test sp.κ >= 0
        for λ in (0.25, 0.55, 1.0, 3.0, 10.0, 30.0)
            m = refractive_index(sp, λ)
            @test real(m) > 1
            @test imag(m) >= 0
        end
        # clamping, not extrapolation, outside the table
        @test refractive_index(sp, 1e-3) == refractive_index(sp, sp.λ[1] / 2)
        @test refractive_index(sp, 1e4) == refractive_index(sp, sp.λ[end] * 2)
    end
    @test_throws ErrorException aerosol_species(:unobtainium)
    @test aerosol_species(:soot) === aerosol_species(:black_carbon)
    @test aerosol_species(:dust) === aerosol_species(:mineral_dust)

    w = water_refractive_index(0.55)
    @info "Liquid water m(0.55 μm)" m=w
    @test isapprox(real(w), 1.333; atol = 0.002)
    @test imag(w) < 1e-8                       # water is transparent here
    @test imag(water_refractive_index(3.0)) > 0.1   # the 3 μm OH stretch

    # Black carbon must be strongly absorbing at every solar wavelength.
    bc = aerosol_species(:black_carbon)
    for λ in (0.3, 0.55, 1.0, 2.0)
        @test imag(refractive_index(bc, λ)) > 0.3
    end
    # Sulphate and sea salt must not be.
    for name in (:sulphate, :sea_salt)
        sp = aerosol_species(name)
        @test imag(refractive_index(sp, 0.55)) < 1e-6
        @test imag(refractive_index(sp, 10.0)) > 1e-3   # but absorbing in the LW
    end
    # Dust: UV/blue absorption an order of magnitude above the red.
    du = aerosol_species(:mineral_dust)
    @test imag(refractive_index(du, 0.30)) > 5 * imag(refractive_index(du, 0.70))
    @test imag(refractive_index(du, 9.5)) > 0.3        # the Si-O band
end

# -------------------------------------------------------------------------
@testset "Size distribution identities" begin
    m = lognormal_mode(:sulphate, 1000.0, 0.05, 1.8)
    @test mode_number(m) ≈ 1000.0
    # ∫ dN/dln r dln r = N, by fine trapezoid over ±8σ
    lnσ = log(1.8)
    a = log(0.05) - 8lnσ; b = log(0.05) + 8lnσ
    n = 4000; h = (b - a) / n
    s = 0.0
    for i in 0:n
        w = (i == 0 || i == n) ? 0.5 : 1.0
        s += w * number_pdf(m, exp(a + i * h)) * h
    end
    @info "Log-normal normalisation ∫dN/dlnr dlnr" N_integrated=s N_nominal=1000.0
    @test isapprox(s, 1000.0; rtol = 1e-8)

    # effective radius = r_g exp(2.5 ln²σ)
    @test isapprox(effective_radius(m), 0.05 * exp(2.5 * lnσ^2); rtol = 1e-12)
    @test effective_radius(m) > m.r_g
    @test effective_radius([m]) ≈ effective_radius(m)

    # mass: N ρ (4π/3) r_g³ exp(4.5 ln²σ) in μg m⁻³
    expect = 1000.0 * (4π / 3) * 0.05^3 * exp(4.5 * lnσ^2) * 1.7
    @test isapprox(mode_mass(m), expect; rtol = 1e-12)
    @test mode_mass([m, m]) ≈ 2 * mode_mass(m)

    # extinction is exactly linear in number concentration
    e1 = bulk_optics([m], 0.55)[1]
    m2 = lognormal_mode(:sulphate, 3000.0, 0.05, 1.8)
    e2, _, _, ssa2, g2 = bulk_optics([m2], 0.55)
    @test isapprox(e2, 3 * e1; rtol = 1e-12)
    @test isapprox(ssa2, bulk_optics([m], 0.55)[4]; rtol = 1e-12)

    @test_throws ErrorException lognormal_mode(:sulphate, 1.0, -1.0, 2.0)
    @test_throws ErrorException lognormal_mode(:sulphate, 1.0, 0.1, 0.9)
end

# -------------------------------------------------------------------------
@testset "Size-integration convergence" begin
    # Self-convergence against a 7680-interval reference.
    rels = Float64[]
    for name in AEROSOL_MIXTURE_NAMES
        mix = aerosol_mixture(name)
        ref = bulk_optics(mix, 0.55; nquad = 7680)[1]
        cur = bulk_optics(mix, 0.55)[1]
        push!(rels, abs(cur - ref) / ref)
        @test abs(cur - ref) / ref < 3e-3
    end
    @info "Quadrature convergence of β_ext(550 nm) vs nquad=7680" mixture=collect(AEROSOL_MIXTURE_NAMES) rel_error=round.(rels; sigdigits=3)
    @test maximum(rels) < 3e-3
end

# -------------------------------------------------------------------------
@testset "Energy consistency, every species and band" begin
    sw = shortwave_bands(Float64; ngpt_per_band = 2)
    lw = longwave_bands(Float64; ngpt_per_band = 2)
    for name in aerosol_species_names()
        mode = lognormal_mode(name, 1000.0, 0.05, 2.0)
        for spec in (sw, lw)
            o = aerosol_band_optics([mode], spec)
            for b in 1:nbands(spec)
                @test o.β_ext[b] > 0
                @test 0 <= o.ssa[b] <= 1
                @test -1 <= o.g[b] <= 1
            end
        end
        for λ in (0.3, 0.55, 1.6, 4.0, 11.0)
            βe, βs, βa, ssa, g = bulk_optics([mode], λ)
            @test βs <= βe * (1 + 1e-12)
            @test βa >= 0
            @test 0 <= ssa <= 1
            @test -1 <= g <= 1
            @test isapprox(βe, βs + βa; rtol = 1e-10)
        end
    end
end

# -------------------------------------------------------------------------
@testset "Species optics: who absorbs" begin
    # OPAC size distributions for each component.
    bc   = [lognormal_mode(:black_carbon, 1e4, 0.0118, 2.00)]
    sul  = [lognormal_mode(:sulphate, 1e3, 0.0212, 2.24)]
    ss   = [lognormal_mode(:sea_salt, 20.0, 0.209, 2.03)]
    oc   = [lognormal_mode(:organic_carbon, 1e3, 0.0212, 2.24)]
    dust = [lognormal_mode(:mineral_dust, 30.5, 0.39, 2.00)]

    ssa = Dict(n => bulk_optics(m, 0.55)[4] for (n, m) in
               (("black_carbon", bc), ("sulphate", sul), ("sea_salt", ss),
                ("organic_carbon", oc), ("mineral_dust", dust)))
    @info "Visible (550 nm) single-scattering albedo by species" ssa...

    @test ssa["black_carbon"] < 0.4            # required: strongly absorbing
    @test ssa["black_carbon"] > 0.10           # but not a perfect absorber
    @test ssa["sulphate"] > 0.98               # required: pure scatterer
    @test ssa["sea_salt"] > 0.98
    @test ssa["organic_carbon"] > 0.9
    @test ssa["organic_carbon"] < ssa["sulphate"]
    @test 0.7 < ssa["mineral_dust"] < 0.99     # intermediate

    # Mass absorption cross-section of BC: Bond & Bergstrom (2006) give
    # 7.5 ± 1.2 m² g⁻¹ at 550 nm for fresh soot.
    mac = AO.mass_absorption_coefficient(bc, 0.55)
    mec = mass_extinction_coefficient(bc, 0.55)
    @info "Black carbon at 550 nm" MAC_m2_per_g=round(mac; digits=3) MEC_m2_per_g=round(mec; digits=3) ssa=round(ssa["black_carbon"]; digits=4)
    @test 5.0 < mac < 11.0
    @test mec > mac

    # Dust: absorbing in the UV/blue, nearly transparent in the red.
    d_uv  = bulk_optics(dust, 0.35)[4]
    d_vis = bulk_optics(dust, 0.55)[4]
    d_nir = bulk_optics(dust, 0.9)[4]
    d_lw  = bulk_optics(dust, 10.0)[4]
    @info "Mineral dust ω₀ vs wavelength" ssa_350nm=round(d_uv; digits=4) ssa_550nm=round(d_vis; digits=4) ssa_900nm=round(d_nir; digits=4) ssa_10um=round(d_lw; digits=4)
    @test d_uv < d_vis < d_nir                 # absorption falls with λ
    @test d_lw < 0.6                           # strongly absorbing in the LW
    @test d_uv < 0.9

    # Forward scattering: every accumulation-mode aerosol has g in a
    # narrow band around 0.6-0.75 in the visible.
    for (n, m) in (("sulphate", sul), ("sea_salt", ss), ("dust", dust))
        g = bulk_optics(m, 0.55)[5]
        @test 0.4 < g < 0.85
    end
end

# -------------------------------------------------------------------------
@testset "Hygroscopic growth (κ-Köhler)" begin
    κs = Dict(n => aerosol_species(n).κ for n in aerosol_species_names())
    @test κs[:sea_salt] > 1.0
    @test 0.5 < κs[:sulphate] < 0.8
    @test κs[:black_carbon] == 0.0

    rhs = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]
    gfs = Dict{Symbol, Vector{Float64}}()
    for n in aerosol_species_names()
        gf = [growth_factor(0.1, rh, κs[n]) for rh in rhs]
        gfs[n] = gf
        # strictly monotone in RH (except the κ = 0 hydrophobic case)
        for i in 2:length(rhs)
            if κs[n] > 0
                @test gf[i] > gf[i - 1]
            else
                @test gf[i] == gf[i - 1] == 1.0
            end
        end
        @test gf[1] ≈ 1.0
    end
    @info "Growth factor at r_dry = 0.1 μm, RH = $(rhs)" pairs(gfs)...

    @test 1.8 < gfs[:sea_salt][6] < 2.4        # RH = 0.90
    @test 1.5 < gfs[:sulphate][6] < 2.0
    @test gfs[:black_carbon][6] == 1.0
    @test gfs[:sea_salt][end] > 4.0            # RH = 0.99
    @test gfs[:mineral_dust][6] < 1.15         # κ = 0.03, barely grows

    # Against the analytic large-particle law GF = (1 + κ RH/(1-RH))^(1/3).
    #
    # NOTE ON TOLERANCE. That law drops the Kelvin term, so it is an
    # *upper bound* on the true κ-Köhler growth, not an equality: the
    # implementation solves S(r_w) = RH including exp(A/r_w), which
    # requires a slightly larger solute term and hence a slightly smaller
    # droplet. The deficit is tiny in S (≈ 4×10⁻⁴ at r_w = 2.5 μm) but is
    # amplified by dGF/dS ∝ (1-S)⁻¹, reaching 0.26 % at RH = 0.95. An
    # earlier version of this test asserted rtol = 2e-3 at RH = 0.95 and
    # failed; the *test* was wrong (it compared against a formula missing
    # a term the implementation correctly has), not the solver. The
    # correct statements are the three below: bounded agreement, the
    # right sign of the discrepancy, and convergence to the law as the
    # Kelvin term is switched off by making the particle large.
    for n in (:sea_salt, :sulphate, :ammonium_nitrate)
        κ = κs[n]
        for rh in (0.8, 0.9, 0.95)
            analytic = (1 + κ * rh / (1 - rh))^(1 / 3)
            gf = growth_factor(1.0, rh, κ)
            @test isapprox(gf, analytic; rtol = 5e-3)
            @test gf < analytic                 # Kelvin always suppresses
            # ...and the suppression vanishes for a large particle
            @test isapprox(growth_factor(100.0, rh, κ), analytic; rtol = 1e-4)
        end
    end
    kelvin_deficit = [1 - growth_factor(r, 0.95, 1.1) /
                          (1 + 1.1 * 0.95 / 0.05)^(1 / 3)
                      for r in (0.01, 0.1, 1.0, 10.0)]
    @info "Kelvin suppression of growth at RH=0.95, κ=1.1 (r_dry = 0.01…10 μm)" deficit=round.(kelvin_deficit; sigdigits=3)
    @test issorted(kelvin_deficit; rev = true)   # smaller particle, more Kelvin
    @test all(>(0), kelvin_deficit)
    @test growth_factor(0.01, 0.95, 1.1) < growth_factor(1.0, 0.95, 1.1)

    # Köhler curve consistency: S(wet_radius(RH)) == RH
    for κ in (0.1, 0.6, 1.1), rh in (0.5, 0.9, 0.99)
        rw = wet_radius(0.1, rh, κ)
        @test isapprox(kohler_saturation(rw, 0.1, κ), rh; rtol = 1e-6)
    end

    # Volume mixing of the refractive index moves m towards water.
    m_dry = refractive_index(aerosol_species(:sea_salt), 0.55)
    m_wet = wet_refractive_index(m_dry, 0.1, 0.25, 0.55)
    m_w   = water_refractive_index(0.55)
    @info "Volume-mixed refractive index, sea salt at GF=2.5" m_dry m_wet m_water=m_w
    @test real(m_w) < real(m_wet) < real(m_dry)
    @test isapprox(real(m_wet),
                   (0.1 / 0.25)^3 * real(m_dry) + (1 - (0.1 / 0.25)^3) * real(m_w);
                   rtol = 1e-12)
    @test wet_refractive_index(m_dry, 0.1, 0.1, 0.55) == m_dry

    # Extinction at high RH is several times the dry value.
    ss = [lognormal_mode(:sea_salt, 20.0, 0.209, 2.03)]
    β_dry = bulk_optics(ss, 0.55)[1]
    ratios = [bulk_optics(ss, 0.55; rh = rh)[1] / β_dry
              for rh in (0.5, 0.8, 0.9, 0.95, 0.98, 0.99)]
    @info "Sea-salt β_ext(550 nm) enhancement vs dry" RH=[0.5, 0.8, 0.9, 0.95, 0.98, 0.99] ratio=round.(ratios; digits=3)
    @test issorted(ratios)
    @test ratios[3] > 3.0                      # RH 0.90
    @test ratios[end] > 8.0                    # RH 0.99 — "several times"
    # Black carbon does not respond at all.
    bc = [lognormal_mode(:black_carbon, 1e4, 0.0118, 2.0)]
    @test bulk_optics(bc, 0.55; rh = 0.99)[1] == bulk_optics(bc, 0.55)[1]
    # Growth makes a scatterer *more* conservative (water is transparent).
    @test bulk_optics([lognormal_mode(:organic_carbon, 1e3, 0.0212, 2.24)],
                      0.55; rh = 0.95)[4] >
          bulk_optics([lognormal_mode(:organic_carbon, 1e3, 0.0212, 2.24)],
                      0.55)[4]
end

# -------------------------------------------------------------------------
@testset "Standard mixtures" begin
    @test length(AEROSOL_MIXTURE_NAMES) == 7
    summary = Dict{Symbol, NTuple{5, Float64}}()
    for name in AEROSOL_MIXTURE_NAMES
        mix = aerosol_mixture(name)
        @test !isempty(mix)
        @test all(m -> m isa LogNormalMode, mix)
        βe, _, _, ssa, g = bulk_optics(mix, 0.55)
        α = angstrom_exponent(mix, 0.44, 0.87)
        summary[name] = (mode_number(mix), mode_mass(mix), βe, ssa, α)
        @test βe > 0
        @test 0 < ssa <= 1
        @test 0 < g < 1
        # scaling the number concentration scales β but not ω₀ or g
        m2 = aerosol_mixture(name; scale = 2.0)
        @test isapprox(bulk_optics(m2, 0.55)[1], 2βe; rtol = 1e-12)
        @test isapprox(bulk_optics(m2, 0.55)[4], ssa; rtol = 1e-12)
    end
    @info "Standard mixtures: (N cm⁻³, M μg m⁻³, β₅₅₀ m⁻¹, ω₀, α₄₄₀₋₈₇₀)" summary...

    @test_throws ErrorException aerosol_mixture(:mars)
    @test aerosol_mixture(:urban) == urban()
    @test length(continental_clean()) >= 2

    # Physical ordering of the mixtures.
    @test summary[:urban][4] < summary[:continental_polluted][4] <
          summary[:continental_clean][4]              # ω₀ falls with soot
    @test summary[:maritime_clean][4] > 0.98          # nearly conservative
    @test summary[:antarctic][4] > 0.98
    @test summary[:urban][4] < 0.75                   # strongly absorbing
    @test summary[:antarctic][1] < summary[:urban][1] / 1000
    @test summary[:desert][2] > summary[:maritime_clean][2]   # dust is heavy
end

# -------------------------------------------------------------------------
@testset "Ångström exponent" begin
    sw = shortwave_bands(Float64; ngpt_per_band = 2)
    αs = Dict{Symbol, Float64}()
    for name in AEROSOL_MIXTURE_NAMES
        o = aerosol_band_optics(aerosol_mixture(name), sw)
        αs[name] = angstrom_exponent(o, 10, 8)   # 0.52 μm vs 0.96 μm
    end
    @info "Ångström exponent from SW bands 10 (≈0.52 μm) and 8 (≈0.96 μm)" αs...

    @test αs[:urban] > 1.0                     # fine mode — required
    @test αs[:continental_polluted] > 1.0
    @test αs[:continental_clean] > 1.0
    @test αs[:desert] < 0.8                    # coarse mode — required
    @test αs[:desert] > -0.3
    @test αs[:maritime_clean] < 0.8
    @test αs[:urban] > αs[:maritime_clean]

    # Monochromatic form must agree with the band form to within the
    # band-averaging difference.
    o = aerosol_band_optics(urban(), sw)
    α_band = angstrom_exponent(o, 10, 8)
    α_mono = angstrom_exponent(urban(), o.λ[10], o.λ[8])
    @info "Urban α: band-integrated vs monochromatic" α_band α_mono
    @test isapprox(α_band, α_mono; atol = 0.15)

    # Pure fine-mode smoke-like population → strongly fine; a pure coarse
    # dust population → near-neutral.
    fine   = [lognormal_mode(:organic_carbon, 5e3, 0.08, 1.6)]
    coarse = [lognormal_mode(:mineral_dust, 1.0, 1.9, 2.15)]
    α_fine   = angstrom_exponent(fine, 0.44, 0.87)
    α_coarse = angstrom_exponent(coarse, 0.44, 0.87)
    @info "Ångström: pure fine smoke vs pure coarse dust" α_fine α_coarse
    @test α_fine > 1.5
    @test α_coarse < 0.5
    @test α_fine > α_coarse

    # Hygroscopic growth pushes the population coarse ⇒ α falls.
    mc = maritime_clean()
    @test angstrom_exponent(mc, 0.44, 0.87; rh = 0.95) <
          angstrom_exponent(mc, 0.44, 0.87)
end

# -------------------------------------------------------------------------
@testset "Humidity table interpolation" begin
    sw = shortwave_bands(Float64; ngpt_per_band = 2)
    mix = maritime_clean()
    tab = hygroscopic_table(mix, sw)
    @test length(tab.rh) > 5
    @test issorted(tab.rh)
    # at a node, interpolation must be exact
    for j in (1, 5, length(tab.rh))
        βe, ssa, g, β550 = interpolate_rh(tab, tab.rh[j])
        @test isapprox(βe[10], tab.β_ext[j, 10]; rtol = 1e-10)
        @test isapprox(β550, tab.β_550[j]; rtol = 1e-10)
    end
    # between nodes, against a direct Mie integration
    for rh in (0.65, 0.87, 0.965, 0.9875)
        βe, ssa, g, β550 = interpolate_rh(tab, rh)
        direct = bulk_optics(mix, 0.55; rh = rh)[1]
        @test isapprox(β550, direct; rtol = 0.02)
        @test 0 <= ssa[10] <= 1
    end
    # above the top node: clamped, never extrapolated
    @test interpolate_rh(tab, 0.9999)[4] == interpolate_rh(tab, tab.rh[end])[4]
    # a hydrophobic mixture collapses to a single node
    tab_bc = hygroscopic_table([lognormal_mode(:black_carbon, 1e4, 0.0118, 2.0)], sw)
    @test length(tab_bc.rh) == 1
    @test interpolate_rh(tab_bc, 0.99)[4] == interpolate_rh(tab_bc, 0.0)[4]
end

# -------------------------------------------------------------------------
@testset "Column profile and AOD normalisation" begin
    sw = shortwave_bands(Float64; ngpt_per_band = 2)
    lw = longwave_bands(Float64; ngpt_per_band = 2)
    nz = 30
    z = [(k - 0.5) * 1000.0 for k in 1:nz]

    for target in (0.05, 0.3, 1.2)
        p = aerosol_profile(:urban, nz, z; aod_550 = target,
                            scale_height = 1500.0, spec = sw)
        @test isapprox(aerosol_optical_depth_550(p), target; rtol = 1e-10)
    end
    p = aerosol_profile(:urban, nz, z; aod_550 = 0.3, scale_height = 1500.0,
                        spec = sw)
    @info "Urban AOD profile" aod550=aerosol_optical_depth_550(p) aod_per_band=round.(aerosol_optical_depth(p); digits=4)
    @test isapprox(aerosol_optical_depth_550(p), 0.3; rtol = 1e-10)
    @test AO.nlayers(p) == nz
    @test length(aerosol_optical_depth(p)) == nbands(sw)
    @test all(>(0), aerosol_optical_depth(p))
    # the visible band AOD must bracket the 550 nm value
    @test aerosol_optical_depth(p, 10) > 0.3 > aerosol_optical_depth(p, 8)

    # exponential decay with the requested scale height
    β = aerosol_extinction_profile(p)
    @test issorted(β; rev = true)
    @test isapprox(β[2] / β[1], exp(-1000.0 / 1500.0); rtol = 1e-8)
    @test isapprox(β[11] / β[1], exp(-10000.0 / 1500.0); rtol = 1e-6)
    @test sum(β[k] * p.dz[k] for k in 1:nz) ≈ 0.3

    # scalar-z form builds the same uniform grid
    p2 = aerosol_profile(:urban, nz, 30000.0; aod_550 = 0.3,
                         scale_height = 1500.0, spec = sw)
    @test isapprox(aerosol_optical_depth_550(p2), 0.3; rtol = 1e-10)
    @test p2.z ≈ z

    # elevated layer carries exactly its requested share
    pe = aerosol_profile(:desert, nz, z; aod_550 = 0.6, scale_height = 1000.0,
                         spec = sw, elevated_layer = (3000.0, 6000.0),
                         elevated_fraction = 0.6)
    @test isapprox(aerosol_optical_depth_550(pe), 0.6; rtol = 1e-10)
    βe = aerosol_extinction_profile(pe)
    τ_elev = sum(βe[k] * pe.dz[k] for k in 4:6)
    @info "Elevated dust layer 3-6 km" total_aod=aerosol_optical_depth_550(pe) aod_in_layer=τ_elev
    @test isapprox(τ_elev, 0.6 * 0.6 + 0.4 * 0.6 *
                   sum(exp(-z[k] / 1000.0) for k in 4:6) /
                   sum(exp(-z[k] / 1000.0) for k in 1:nz); rtol = 0.02)
    @test βe[4] > βe[3]                   # the layer sticks out above the BL

    # humidity profile drives the growth
    col = us_standard_column(Float64; nz = nz, z_top = 30000.0)
    rh = relative_humidity_profile(col)
    @test length(rh) == nz
    @test all(0 .<= rh .< 1)
    pr = aerosol_profile(:maritime_clean, nz, z; aod_550 = 0.15,
                         scale_height = 800.0, spec = sw, rh = rh)
    @test isapprox(aerosol_optical_depth_550(pr), 0.15; rtol = 1e-10)
    pd = aerosol_profile(:maritime_clean, nz, z; aod_550 = 0.15,
                         scale_height = 800.0, spec = sw, rh = 0.0)
    # At fixed AOD, humid particles are bigger ⇒ more forward-scattering.
    @info "Maritime g(band 10): dry vs ambient RH" g_dry=pd.g[1, 10] g_humid=pr.g[1, 10] rh_sfc=rh[1]
    @test pr.g[1, 10] > pd.g[1, 10]
    # ...and they need fewer of them, so the scale factor is smaller
    @test pr.scale[1] < pd.scale[1]

    # longwave rebuild keeps the geometry and the 550 nm AOD
    pl = rebuild_bands(pd, lw)
    @test isapprox(aerosol_optical_depth_550(pl),
                   aerosol_optical_depth_550(pd); rtol = 1e-12)
    @test pl.scale == pd.scale
    @test nbands(pl.spec) == nbands(lw)
    pdust_lw = rebuild_bands(
        aerosol_profile(:desert, nz, z; aod_550 = 0.5, scale_height = 2500.0,
                        spec = sw), lw)
    τ_lw = aerosol_optical_depth(pdust_lw)
    @info "Desert dust: LW band AOD (SW 550 nm AOD = 0.5)" band_9_7um=τ_lw[7] band_8_9um=τ_lw[8] max=maximum(τ_lw)
    @test all(>(0), τ_lw)
    @test maximum(τ_lw) > 0.05          # dust has a real longwave effect
    @test pdust_lw.ssa[1, 7] < 0.6      # and it is absorbing at 9.7 μm

    @test_throws ErrorException aerosol_profile(:urban, 5, z; spec = sw)
    @test_throws ErrorException aerosol_profile(:urban, nz, z; aod_550 = -1,
                                                spec = sw)
end

# -------------------------------------------------------------------------
@testset "compute_optics! contract" begin
    sw = shortwave_bands(Float64; ngpt_per_band = 2)
    nz = 30
    z = [(k - 0.5) * 1000.0 for k in 1:nz]
    col = us_standard_column(Float64; nz = nz, z_top = 30000.0)
    p = aerosol_profile(:urban, nz, z; aod_550 = 0.3, scale_height = 1500.0,
                        spec = sw)
    @test p isa Aeolus.RadiationCore.AbstractAerosolOptics

    # --- overwrite mode reproduces the stored table
    oa = allocate_optics(Float64, (nz,), sw)
    compute_optics!(oa, p, col, sw; add = false)
    for b in 1:nbands(sw), g in gpoint_range(sw, b)
        @test oa.τ[1, g] ≈ p.τ[1, b]
        @test oa.ssa[1, g] ≈ p.ssa[1, b]
        @test oa.g[1, g] ≈ p.g[1, b]
    end
    @test isapprox(sum(@view oa.τ[:, first(gpoint_range(sw, 10))]),
                   aerosol_optical_depth(p, 10); rtol = 1e-12)

    # --- additive mode must be identical to combine_optics!
    gas = correlated_k_optics(sw)
    o1 = allocate_optics(Float64, (nz,), sw)
    o2 = allocate_optics(Float64, (nz,), sw)
    compute_optics!(o1, gas, col)
    compute_optics!(o2, gas, col)
    τ_gas = copy(o1.τ)
    compute_optics!(o1, p, col, sw)                    # merge in place
    buf = allocate_optics(Float64, (nz,), sw)
    compute_optics!(buf, p, col, sw; add = false)
    combine_optics!(o2, buf)                           # the reference path
    @info "compute_optics! additive vs combine_optics!" max_dτ=maximum(abs, o1.τ .- o2.τ) max_dssa=maximum(abs, o1.ssa .- o2.ssa) max_dg=maximum(abs, o1.g .- o2.g)
    @test maximum(abs, o1.τ .- o2.τ) < 1e-14
    @test maximum(abs, o1.ssa .- o2.ssa) < 1e-14
    @test maximum(abs, o1.g .- o2.g) < 1e-14
    @test all(o1.τ .>= τ_gas .- 1e-15)                 # aerosol only adds
    @test all(0 .<= o1.ssa .<= 1)
    @test all(-1 .<= o1.g .<= 1)

    # --- merging is commutative in the optical sense
    o3 = allocate_optics(Float64, (nz,), sw)
    compute_optics!(o3, p, col, sw; add = false)
    gasbuf = allocate_optics(Float64, (nz,), sw)
    compute_optics!(gasbuf, gas, col)
    combine_optics!(o3, gasbuf)
    @test maximum(abs, o3.τ .- o1.τ) < 1e-12
    @test maximum(abs, o3.ssa .- o1.ssa) < 1e-12

    # --- column-free and 3-argument forms agree
    o4 = allocate_optics(Float64, (nz,), sw)
    compute_optics!(o4, p; add = false)
    o5 = allocate_optics(Float64, (nz,), sw)
    compute_optics!(o5, p, col; add = false)
    @test o4.τ == o5.τ == buf.τ

    # --- a grid the profile was not built on is rejected, not mis-used
    lw = longwave_bands(Float64; ngpt_per_band = 2)
    olw = allocate_optics(Float64, (nz,), lw)
    @test_throws ErrorException compute_optics!(olw, p, col, lw)
    # ...unless it is the same band structure with a different g-point count
    sw4 = shortwave_bands(Float64; ngpt_per_band = 4)
    o6 = allocate_optics(Float64, (nz,), sw4)
    compute_optics!(o6, p, col, sw4; add = false)
    @test isapprox(sum(@view o6.τ[:, first(gpoint_range(sw4, 10))]),
                   aerosol_optical_depth(p, 10); rtol = 1e-12)
end

# -------------------------------------------------------------------------
@testset "Radiative forcing" begin
    sw = shortwave_bands(Float64; ngpt_per_band = 2)
    nz = 30
    z = [(k - 0.5) * 1000.0 for k in 1:nz]
    col = us_standard_column(Float64; nz = nz, z_top = 30000.0)
    μ₀ = 0.5

    # --- a scattering aerosol over a dark surface must cool the TOA
    p_sca = aerosol_profile(:maritime_clean, nz, z; aod_550 = 0.2,
                            scale_height = 1000.0, spec = sw)
    f_sca = aerosol_radiative_forcing(p_sca, col; μ₀ = μ₀, sfc_albedo = 0.05)
    @info "Maritime-clean forcing (AOD 0.2, α_s = 0.05, μ₀ = 0.5)" toa=round(f_sca.toa; digits=3) sfc=round(f_sca.surface; digits=3) atm=round(f_sca.atmosphere; digits=3) eff=round(f_sca.forcing_efficiency; digits=2)
    @test f_sca.toa < 0                        # required: cooling
    @test f_sca.surface < 0                    # and dimming
    @test abs(f_sca.atmosphere) < 0.1 * abs(f_sca.toa)   # ~no absorption
    @test f_sca.forcing_efficiency < -50       # W m⁻² per unit AOD
    @test isapprox(f_sca.atmosphere, f_sca.toa - f_sca.surface; rtol = 1e-12)
    @test f_sca.aod_550 ≈ 0.2

    # forcing scales roughly linearly with AOD in the optically thin limit
    p_sca2 = aerosol_profile(:maritime_clean, nz, z; aod_550 = 0.1,
                             scale_height = 1000.0, spec = sw)
    f_sca2 = aerosol_radiative_forcing(p_sca2, col; μ₀ = μ₀, sfc_albedo = 0.05)
    @test f_sca2.toa > f_sca.toa               # less aerosol, less cooling
    @test isapprox(f_sca.toa / f_sca2.toa, 2.0; atol = 0.25)

    # --- black carbon: heats the column, dims the surface
    p_bc = aerosol_profile([lognormal_mode(:black_carbon, 1.3e5, 0.0118, 2.0)],
                           nz, z; aod_550 = 0.2, scale_height = 1000.0,
                           spec = sw)
    f_bc = aerosol_radiative_forcing(p_bc, col; μ₀ = μ₀, sfc_albedo = 0.05)
    @info "Black-carbon forcing (AOD 0.2, α_s = 0.05, μ₀ = 0.5)" toa=round(f_bc.toa; digits=3) sfc=round(f_bc.surface; digits=3) atm=round(f_bc.atmosphere; digits=3)
    @test f_bc.atmosphere > 0                  # required: warms the column
    @test f_bc.atmosphere > 50                 # and strongly so
    @test f_bc.surface < -50                   # required: strong dimming
    @test f_bc.surface < f_sca.surface         # more dimming than scatterers
    @test f_bc.atmosphere > 10 * abs(f_sca.atmosphere)
    @test isapprox(f_bc.atmosphere, f_bc.toa - f_bc.surface; rtol = 1e-12)

    # TOA forcing rises monotonically with surface albedo for BC
    toas = [aerosol_radiative_forcing(p_bc, col; μ₀ = μ₀, sfc_albedo = a).toa
            for a in (0.0, 0.1, 0.2, 0.4, 0.7)]
    @info "BC TOA forcing vs surface albedo (0, 0.1, 0.2, 0.4, 0.7)" toa=round.(toas; digits=2)
    @test issorted(toas)
    @test toas[1] < 0                          # over a black surface: cooling
    @test toas[end] > 0                        # over a bright one: warming

    # ...and falls with albedo for a scatterer only in magnitude
    toas_s = [aerosol_radiative_forcing(p_sca, col; μ₀ = μ₀, sfc_albedo = a).toa
              for a in (0.0, 0.2, 0.5)]
    @test all(<(0), toas_s)                    # always cooling
    @test issorted(toas_s)

    # --- per-layer heating
    hr = aerosol_heating_rate(p_bc, col; μ₀ = μ₀, sfc_albedo = 0.05)
    @info "BC shortwave heating-rate anomaly" K_per_day_layer1=round(hr[1]; digits=3) max=round(maximum(hr); digits=3)
    @test length(hr) == nz
    @test maximum(hr) > 1.0                    # K day⁻¹, a real plume signal
    @test hr[1] == maximum(hr)                 # peaks where the aerosol is
    @test all(hr[20:end] .< 0.05)              # nothing aloft
    hr_s = aerosol_heating_rate(p_sca, col; μ₀ = μ₀, sfc_albedo = 0.05)
    @test maximum(hr_s) < 0.2 * maximum(hr)    # scatterers barely heat
end

# -------------------------------------------------------------------------
@testset "Critical surface albedo" begin
    sw = shortwave_bands(Float64; ngpt_per_band = 2)
    nz = 30
    z = [(k - 0.5) * 1000.0 for k in 1:nz]
    col = us_standard_column(Float64; nz = nz, z_top = 30000.0)
    μ₀ = 0.5

    p_urb = aerosol_profile(:urban, nz, z; aod_550 = 0.3,
                            scale_height = 1500.0, spec = sw)
    αc_urb = critical_surface_albedo(p_urb, col; μ₀ = μ₀)

    # Chylek & Coakley (1974) single-scattering estimate, from the
    # solar-weighted ω₀ and g of this very profile:
    #   β ω (1-α)² = 2 α (1-ω),  β ≈ (1-g)/2
    wsum = 0.0; ω̄ = 0.0; ḡ = 0.0
    for b in 1:nbands(sw)
        s = sum(sw.solar_src[g] for g in gpoint_range(sw, b))
        τb = aerosol_optical_depth(p_urb, b)
        wsum += s * τb
        ω̄ += s * τb * p_urb.ssa[1, b]
        ḡ += s * τb * p_urb.g[1, b]
    end
    ω̄ /= wsum; ḡ /= wsum
    βup = (1 - ḡ) / 2
    # solve βω(1-α)² = 2α(1-ω)  ⇒  α² - (2 + 2(1-ω)/(βω))α + 1 = 0
    bq = 2 + 2 * (1 - ω̄) / (βup * ω̄)
    αc_theory = (bq - sqrt(bq^2 - 4)) / 2
    @info "Urban critical surface albedo" numerical=round(αc_urb; digits=4) chylek_coakley=round(αc_theory; digits=4) ω̄=round(ω̄; digits=4) ḡ=round(ḡ; digits=4)

    @test 0.05 < αc_urb < 0.45                 # the plausible range
    @test isapprox(αc_urb, αc_theory; rtol = 0.5)   # same physics, ±50 %
    # the sign really does flip there
    @test aerosol_radiative_forcing(p_urb, col; μ₀ = μ₀,
                                    sfc_albedo = αc_urb - 0.05).toa < 0
    @test aerosol_radiative_forcing(p_urb, col; μ₀ = μ₀,
                                    sfc_albedo = αc_urb + 0.05).toa > 0

    # Pure external black carbon (ω₀ ≈ 0.21) has a *much* lower critical
    # albedo than a BC-containing mixture: almost any real surface is
    # bright enough for it to warm the planet.
    p_bc = aerosol_profile([lognormal_mode(:black_carbon, 1.3e5, 0.0118, 2.0)],
                           nz, z; aod_550 = 0.2, scale_height = 1000.0,
                           spec = sw)
    αc_bc = critical_surface_albedo(p_bc, col; μ₀ = μ₀)
    @info "Pure black-carbon critical surface albedo" αc_bc=round(αc_bc; digits=4)
    @test 0.0 < αc_bc < 0.10
    @test αc_bc < αc_urb

    # Dust, being far less absorbing, flips only over a bright surface.
    p_dust = aerosol_profile(:desert, nz, z; aod_550 = 0.5,
                             scale_height = 2500.0, spec = sw)
    αc_dust = critical_surface_albedo(p_dust, col; μ₀ = μ₀)
    @info "Desert-dust critical surface albedo" αc_dust=round(αc_dust; digits=4)
    @test αc_dust > αc_urb
    @test 0.2 < αc_dust < 0.75

    # A purely scattering aerosol never flips sign: NaN is the correct
    # answer, not a spurious root.
    p_ss = aerosol_profile([lognormal_mode(:sea_salt, 20.0, 0.209, 2.03)],
                           nz, z; aod_550 = 0.2, scale_height = 1000.0,
                           spec = sw)
    αc_ss = critical_surface_albedo(p_ss, col; μ₀ = μ₀)
    @info "Sea-salt critical surface albedo (expected NaN — never warms)" αc_ss
    @test isnan(αc_ss) || αc_ss > 0.8
end

end # Phase 37
