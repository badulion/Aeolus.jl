# Phase 43, issue #6 — immersion and deposition freezing.
#
# Organised by the physics each piece must respect: published INAS and
# INP values at reference temperatures, the analytic limits of the
# Poisson frozen fraction, zero nucleation where the physics forbids it
# (T ≥ 0 °C, S_i ≤ 1), monotonic increase with supercooling, and — for
# the source terms — conservation of INP number, water mass and energy,
# and consistency with the Phase 39 mixed-phase ramp.

using Test, Printf, Random
using Aeolus
using Aeolus.IceNucleation
using Aeolus.MoistThermodynamics: DEFAULT_CONSTANTS, ThermodynamicConstants,
                                  liquid_fraction, latent_heat_fusion,
                                  latent_heat_sublim, latent_heat_vapor,
                                  saturation_vapor_pressure,
                                  saturation_vapor_pressure_ice

const _C = DEFAULT_CONSTANTS

# Independent midpoint quadrature in ln r of the INAS frozen number
# fraction of a log-normal (not the module's shifted-Gaussian rule).
function _ref_frozen_fraction(n_s, m::ARGMode; n = 100_000)
    lnσ = log(m.σ_g)
    lo = log(m.r_g) - 12lnσ
    hi = log(m.r_g) + 12lnσ
    h = (hi - lo) / n
    s = 0.0
    for j in 1:n
        x = lo + (j - 0.5) * h
        z = (x - log(m.r_g)) / lnσ
        s += -expm1(-n_s * 4π * exp(2x)) * exp(-0.5z^2)
    end
    return s * h / (sqrt(2π) * lnσ)
end

@testset "Immersion and deposition freezing (issue #6)" begin

    # ---------------------------------------------------------------------
    @testset "Published INAS densities at reference temperatures" begin
        # Niemand et al. (2012) Eq. 5, T in °C: n_s = exp(-0.517 T + 8.934)
        ns20 = ice_active_site_density(Niemand2012(), 253.15)
        @test ns20 ≈ exp(0.517 * 20 + 8.934) rtol = 1e-12   # 253.15-273.15 rounds
        @test ns20 ≈ 2.347e8 rtol = 1e-3
        @test ice_active_site_density(Niemand2012(), 243.15) ≈ 4.14e10 rtol = 1e-2
        @info "Niemand et al. (2012) n_s" n_s_m2_at_m20C = ns20 n_s_m2_at_m30C =
              ice_active_site_density(Niemand2012(), 243.15)

        # Ullrich et al. (2017) immersion: same slope, constant offset
        # exp(150.577 - 0.517·273.15 - 8.934) = e^0.424 = 1.528
        for T in 243.15:1.0:259.15
            @test ice_active_site_density(Ullrich2017Immersion(), T) /
                  ice_active_site_density(Niemand2012(), T) ≈
                  exp(150.577 - 0.517 * 273.15 - 8.934) rtol = 1e-12
        end
        @test exp(150.577 - 0.517 * 273.15 - 8.934) ≈ 1.528 rtol = 1e-3

        # Ullrich et al. (2017) deposition, dust coefficients, against the
        # published expression written out independently (with the
        # module's `- 1`, which is below 1 m⁻² in absolute terms).
        pub(T, S) = exp(285.692 * (S - 1)^0.25 * cos(0.017 * (T - 256.692))^2 *
                        (π / 2 - atan(0.080 * (T - 200.745))) / π)
        for T in (206.0, 215.0, 220.0, 230.0, 240.0), S in (1.02, 1.1, 1.2, 1.3)
            ns = ice_active_site_density(Ullrich2017Deposition(), T, S)
            @test pub(T, S) - ns ≈ 1 atol = 1e-6 * pub(T, S)
        end
        ns_dep = ice_active_site_density(Ullrich2017Deposition(), 220.0, 1.1)
        @test ns_dep ≈ 2.662e8 rtol = 1e-3
        @info "Ullrich et al. (2017) deposition n_s" at_220K_Si_1p1 = ns_dep at_230K_Si_1p2 =
              ice_active_site_density(Ullrich2017Deposition(), 230.0, 1.2)
    end

    # ---------------------------------------------------------------------
    @testset "Published INP concentrations (aerosol-count schemes)" begin
        # At standard T and p the conversions are the identity.
        stdT = STANDARD_TEMPERATURE
        # DeMott et al. (2015) Eq. 2, cf = 3, N = 1 std cm⁻³ at -20 °C —
        # evaluate at the standard state by passing p so that p/T is
        # standard (the formula's T enters only through 273.16 - T).
        T = 253.15
        p_std_at_T = STANDARD_PRESSURE * T / stdT
        d = singular_inp_concentration(DeMott2015(), T, 1e6; p = p_std_at_T)
        @test d ≈ 3 * exp(0.46 * 20.01 - 11.6) * 1e3 rtol = 1e-12
        @test singular_inp_concentration(DeMott2015(cf = 1.0), T, 1e6;
                                         p = p_std_at_T) ≈ d / 3 rtol = 1e-14
        # Tobo et al. (2013): N^(-0.074ΔT + 3.8) exp(0.414ΔT - 9.671)
        t = singular_inp_concentration(Tobo2013(), T, 1e6; p = p_std_at_T)
        @test t ≈ exp(0.414 * 20.01 - 9.671) * 1e3 rtol = 1e-12
        @test t ≈ 249.9 rtol = 1e-3                       # 0.25 std L⁻¹
        t10 = singular_inp_concentration(Tobo2013(), T, 1e7; p = p_std_at_T)
        @test t10 / t ≈ 10^(-0.074 * 20.01 + 3.8) rtol = 1e-12
        @info "INP at -20 °C, N>0.5μm = 1 std cm⁻³ (std L⁻¹)" DeMott2015_cf3 = d / 1e3 Tobo2013 = t / 1e3

        # Standard-concentration conversion: at half the standard density
        # the ambient N_05 is half a standard N_05 of the same air, and the
        # prediction scales back. DeMott: n ∝ N^1.25, so
        # n_amb(N/2 at ρ/2) = ½ · n_std(N) exactly.
        full = singular_inp_concentration(DeMott2015(), T, 2e6; p = p_std_at_T)
        half = singular_inp_concentration(DeMott2015(), T, 1e6; p = p_std_at_T / 2)
        @test half ≈ full / 2 rtol = 1e-12

        # Cross-check DeMott (2015, cf = 3) against Niemand (2012) for a
        # Saharan-layer-like dust mode (N_{>0.5} ≈ 11 cm⁻³, A ≈ 8e-5 m² m⁻³,
        # cf. DeMott et al. 2015 §3.3: > 8e-11 m² cm⁻³, up to 30 cm⁻³).
        # DeMott et al. report "excellent agreement" for their measured
        # size distribution; for a σ_g = 2 log-normal the two independent
        # fits agree within a factor of five over -20…-35 °C.
        m = arg_mode(1.5e7, 0.4e-6, 2.0, 0.03; ρ_p = 2600.0)
        N05 = number_above_radius(m, DEMOTT_CUTOFF_RADIUS)
        @test N05 ≈ 1.127e7 rtol = 1e-3
        @test mode_surface_area(m) ≈ 7.88e-5 rtol = 1e-3
        ratios = Float64[]
        for T in (238.15, 243.15, 248.15, 253.15)
            nN = inas_frozen_fraction(Niemand2012(), m, T) * m.N
            nD = singular_inp_concentration(DeMott2015(), T, N05;
                                            p = STANDARD_PRESSURE * T / stdT)
            push!(ratios, nD / nN)
        end
        @test all(r -> 0.2 < r < 5, ratios)
        @info "DeMott2015(cf=3) / Niemand2012, Saharan-like mode, -35…-20 °C" ratios
    end

    # ---------------------------------------------------------------------
    @testset "No ice at T ≥ 0 °C; no deposition at S_i ≤ 1" begin
        inas = (Niemand2012(), Ullrich2017Immersion(), Ullrich2017Deposition())
        count = (DeMott2015(), Tobo2013())
        for T in (_C.T_freeze, _C.T_freeze + 1e-9, 275.0, 300.0)
            for s in inas
                @test ice_active_site_density(s, T, 1.5) == 0.0
            end
            for s in count
                @test singular_inp_concentration(s, T, 1e7) == 0.0
            end
        end
        # Even with a custom scheme range reaching 0 °C, T_freeze wins:
        # the rule is in `_fit_temperature`, shared by all schemes.
        for T in 206.0:0.5:240.0, S in (0.0, 0.5, 0.999999, 1.0)
            @test ice_active_site_density(Ullrich2017Deposition(), T, S) == 0.0
        end
        # continuous at S_i → 1⁺ (the published exp{…} would give 1 m⁻²)
        @test ice_active_site_density(Ullrich2017Deposition(), 220.0, 1 + 1e-12) < 1e-1
        # deposition is clamped at water saturation
        for T in (210.0, 225.0, 239.0)
            Siw = saturation_vapor_pressure(T) / saturation_vapor_pressure_ice(T)
            @test ice_active_site_density(Ullrich2017Deposition(), T, 5.0) ==
                  ice_active_site_density(Ullrich2017Deposition(), T, Siw)
            @test ice_active_site_density(Ullrich2017Deposition(), T, Siw) >
                  ice_active_site_density(Ullrich2017Deposition(), T, 1 + 0.5(Siw - 1))
        end
        # warmer than the fitted range: no ice (no extrapolation to 0 °C)
        @test ice_active_site_density(Niemand2012(), 262.0) == 0.0
        @test ice_active_site_density(Ullrich2017Deposition(), 241.0, 1.2) == 0.0
        @test singular_inp_concentration(Tobo2013(), 265.0, 1e7) == 0.0
    end

    # ---------------------------------------------------------------------
    @testset "Monotonic increase with supercooling" begin
        Ts = collect(_C.T_freeze:-0.25:200.0)
        for s in (Niemand2012(), Ullrich2017Immersion())
            ns = [ice_active_site_density(s, T) for T in Ts]
            @test all(diff(ns) .>= 0)
            lo, hi = scheme_temperature_range(s)
            inside = [ice_active_site_density(s, T) for T in hi:-0.25:lo]
            @test all(diff(inside) .> 0)                 # strictly, where fitted
        end
        for S in (1.05, 1.2, 1.35)
            ns = [ice_active_site_density(Ullrich2017Deposition(), T, S) for T in Ts]
            @test all(diff(ns) .>= 0)
            inside = [ice_active_site_density(Ullrich2017Deposition(), T, S)
                      for T in 240.0:-0.25:206.0]
            @test all(diff(inside) .> 0)
        end
        # deposition also increases with ice supersaturation
        for T in (210.0, 225.0, 238.0)
            ns = [ice_active_site_density(Ullrich2017Deposition(), T, S)
                  for S in 1.0:0.01:1.35]
            @test all(diff(ns) .>= 0)
        end
        # Aerosol-count schemes: monotonic at fixed air density (p ∝ T),
        # which isolates the scheme's own temperature dependence; the
        # tolerance is rounding in p/T.
        for s in (DeMott2015(), Tobo2013()), N in (1e5, 1e6, 1e7)
            n = [singular_inp_concentration(s, T, N;
                     p = STANDARD_PRESSURE * T / STANDARD_TEMPERATURE) for T in Ts]
            @test all(diff(n) .>= -1e-14 .* n[2:end])
        end
        # At fixed ambient N and p, colder air is denser, so the same
        # ambient N_05 is fewer particles per std volume. Below the fitted
        # range (formula held) DeMott's n ∝ N_std^1.25 · k with
        # k = (p/p₀)(T₀/T) gives n ∝ T^0.25 — a real ideal-gas effect,
        # not a violation of monotonic freezing (this was the first,
        # mis-specified, form of the test above).
        @test singular_inp_concentration(DeMott2015(), 210.0, 1e6) /
              singular_inp_concentration(DeMott2015(), 230.0, 1e6) ≈
              (210 / 230)^0.25 rtol = 1e-12
    end

    # ---------------------------------------------------------------------
    @testset "Poisson frozen fraction: bounds and small-n_s A limit" begin
        for x in (0.0, 1e-300, 1e-12, 1e-3, 0.5, 1.0, 10.0, 1e3, 1e300)
            f = inas_frozen_fraction(x, 1.0)
            @test 0 <= f <= 1
        end
        @test inas_frozen_fraction(0.0, 1e-12) == 0.0
        @test inas_frozen_fraction(1e20, 1e-12) == 1.0
        # f = x - x²/2 + x³/6 - …: the linear limit is exact to O(x)
        for x in (1e-12, 1e-8, 1e-4)
            f = inas_frozen_fraction(x, 1.0)
            @test f / x ≈ 1 - x / 2 + x^2 / 6 rtol = 1e-12
        end

        # Log-normal mode: against an independent quadrature, and → n_s A/N
        m = arg_mode(1e6, 0.4e-6, 2.0, 0.03; ρ_p = 2600.0)
        a = mode_surface_area(m) / m.N
        worst = 0.0
        for T in (258.0, 250.0, 243.0, 238.0)
            n_s = ice_active_site_density(Niemand2012(), T)
            F = inas_frozen_fraction(Niemand2012(), m, T)
            Fref = _ref_frozen_fraction(n_s, m)
            worst = max(worst, abs(F - Fref) / Fref)
            @test 0 < F < 1
            @test F <= n_s * a * (1 + 1e-12)            # 1 - e^{-x} ≤ x
        end
        @test worst < 1e-8
        n_s = ice_active_site_density(Niemand2012(), 258.0)   # n_s a ≈ 1e-4
        F = inas_frozen_fraction(Niemand2012(), m, 258.0)
        @test F / (n_s * a) ≈ 1 rtol = 1e-3
        @info "INAS mode frozen fraction" worst_rel_vs_quadrature = worst small_limit_ratio = F / (n_s * a)
    end

    # ---------------------------------------------------------------------
    @testset "Default schemes dispatch on the Phase 37 species" begin
        dust = lognormal_mode(:mineral_dust, 1.0, 0.4, 2.0)
        bio = lognormal_mode(:biological, 0.01, 1.0, 1.6)
        ss = lognormal_mode(:sea_salt, 1.0, 0.4, 2.0)
        inp = inp_population([dust, bio, ss])
        sch = default_freezing_schemes(inp)
        @test sch[1].immersion isa Niemand2012
        @test sch[1].deposition isa Ullrich2017Deposition
        @test sch[2].immersion isa Tobo2013
        @test sch[2].deposition === nothing
        @test sch[3] == (immersion = nothing, deposition = nothing)
        @test freezing_mechanism(Ullrich2017Deposition()) === :deposition
        @test freezing_mechanism(Niemand2012()) === :immersion
    end
end
