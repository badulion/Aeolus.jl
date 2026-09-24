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

    # ---------------------------------------------------------------------
    # Source terms. A reference parcel: 1 cm⁻³ of dust (r_g 0.4 μm,
    # σ_g 2) and 0.01 cm⁻³ of biological particles, in a cloud of
    # 100 cm⁻³ droplets holding 0.2 g m⁻³ of liquid.
    _parcel() = inp_population([lognormal_mode(:mineral_dust, 1.0, 0.4, 2.0),
                                lognormal_mode(:biological, 0.01, 1.0, 1.6)])
    N_liq, M_liq = 1e8, 2e-4
    p_air = 7.0e4

    @testset "INP consumed = ice crystals formed; budgets close" begin
        inp = _parcel()
        T = 248.15
        b0 = inp_budget(inp)
        t = ice_nucleation!(inp, T, 1.2, 10.0; p = p_air, N_liq, M_liq)
        b1 = inp_budget(inp)
        @test t.N_inp > 0
        @test ice_number_formed(t) ≈ t.N_inp rtol = 1e-12
        @test t.N_ice_imm ≈ t.N_inp rtol = 1e-12       # all immersed in cloud
        @test t.N_ice_dep == 0.0 && t.N_ice_hom == 0.0
        @test t.N_inp ≈ b1.N_cons - b0.N_cons rtol = 1e-12
        @test t.M_inp ≈ b1.M_cons - b0.M_cons rtol = 1e-12
        r = inp_budget_residual(inp)
        @test maximum(abs, values(r)) < 1e-14

        # First call on a fresh population: the dust increment is the INAS
        # frozen fraction of the mode, the biological one Tobo (2013) on
        # its N_{>0.5} — the source term is exactly the parameterisation.
        inp2 = _parcel()
        dust0 = inp_modes(inp2)[1]
        N05bio = inp_number_above_radius(inp2, DEMOTT_CUTOFF_RADIUS; mode = 2)
        ice_nucleation!(inp2, T, 1.2, 10.0; p = p_air, N_liq, M_liq)
        @test inp_number(inp2; mode = 1, population = :consumed) ≈
              inas_frozen_fraction(Niemand2012(), dust0, T) * dust0.N rtol = 1e-10
        @test inp_number(inp2; mode = 2, population = :consumed) ≈
              singular_inp_concentration(Tobo2013(), T, N05bio; p = p_air) rtol = 1e-10
        @info "Ice nucleated at -25 °C (L⁻¹)" dust = 1e-3 * inp_number(inp2; mode = 1, population = :consumed) biological =
              1e-3 * inp_number(inp2; mode = 2, population = :consumed)
    end

    @testset "Water and energy: liquid/vapour sinks and latent heat" begin
        # immersion: each frozen droplet carries the mean droplet mass
        inp = _parcel()
        T = 243.15
        t = ice_nucleation!(inp, T, 1.3, 5.0; p = p_air, N_liq, M_liq)
        @test t.M_ice == t.M_liq + t.M_vap
        @test t.N_liq == t.N_ice_imm
        @test t.M_liq ≈ t.N_ice_imm * M_liq / N_liq rtol = 1e-14
        @test t.M_vap == 0.0
        @test t.heat ≈ latent_heat_fusion(T) * t.M_liq rtol = 1e-14
        @info "Immersion freezing at -30 °C" N_ice_L = 1e-3 * t.N_ice_imm frozen_liquid_g_m3 = 1e3 * t.M_liq heat_J_m3 = t.heat

        # deposition: out of cloud, ice supersaturated, below water
        # saturation. Vapour → ice releases L_s = L_v + L_f.
        inp = _parcel()
        T, S = 225.0, 1.25
        t = ice_nucleation!(inp, T, S, 5.0; p = 3.0e4)
        @test t.N_ice_dep > 0
        @test t.N_ice_imm == 0 && t.M_liq == 0
        @test t.N_ice_dep ≈ t.N_inp rtol = 1e-12
        @test t.M_vap ≈ t.N_ice_dep * NUCLEATED_ICE_MASS rtol = 1e-14
        @test t.M_ice == t.M_vap
        @test t.heat / t.M_vap ≈ latent_heat_sublim(T) rtol = 1e-14
        @test latent_heat_sublim(T) ≈ latent_heat_vapor(T) + latent_heat_fusion(T) rtol = 1e-14
        @info "Deposition nucleation at 225 K, S_i = 1.25" N_ice_L = 1e-3 * t.N_ice_dep

        # vapour cap: a crystal mass so large that the nucleated crystals
        # would take more than the excess over ice saturation
        inp = _parcel()
        t = ice_nucleation!(inp, T, S, 5.0; m_ice0 = 1e-3)
        Rv = 8.314462618 / 0.018015268
        excess = (S - 1) * saturation_vapor_pressure_ice(T) / (Rv * T)
        @test t.M_vap ≈ excess rtol = 1e-12
        @test t.N_ice_dep ≈ t.N_inp rtol = 1e-12       # crystals still form

        # droplet cap: more immersion-frozen INP than droplets
        inp = _parcel()
        t = ice_nucleation!(inp, 238.15, 1.4, 5.0; N_liq = 10.0, M_liq = 1e-10)
        @test t.N_inp > 10
        @test t.N_ice_imm == 10.0
        @test t.M_liq ≈ 1e-10 rtol = 1e-14

        # rates are increments over Δt
        inp = _parcel()
        t = ice_nucleation!(inp, 245.0, 1.3, 4.0; N_liq, M_liq)
        r = nucleation_rates(t)
        @test r.dNi_dt * 4 ≈ ice_number_formed(t) rtol = 1e-14
        @test r.heating * 4 ≈ t.heat rtol = 1e-14
        @test r.dMl_dt * 4 ≈ t.M_liq rtol = 1e-14
    end

    @testset "Mixed immersion/deposition: split conserves number" begin
        inp = _parcel()
        T = 236.0
        Siw = saturation_vapor_pressure(T) / saturation_vapor_pressure_ice(T)
        t = ice_nucleation!(inp, T, 0.5 * (1 + Siw), 5.0; N_liq, M_liq,
                            immersed_fraction = 0.5)
        @test t.N_ice_imm > 0 && t.N_ice_dep > 0
        @test ice_number_formed(t) ≈ t.N_inp rtol = 1e-12
        @test t.M_ice == t.M_liq + t.M_vap
        @test maximum(abs, values(inp_budget_residual(inp))) < 1e-14
    end

    @testset "Singular: no double counting" begin
        inp = _parcel()
        t1 = ice_nucleation!(inp, 248.15, 1.2, 10.0; N_liq, M_liq)
        t2 = ice_nucleation!(inp, 248.15, 1.2, 10.0; N_liq, M_liq)
        @test t1.N_inp > 0
        @test t2.N_inp == 0.0 && ice_number_formed(t2) == 0.0 && t2.heat == 0.0
        t3 = ice_nucleation!(inp, 245.15, 1.2, 10.0; N_liq, M_liq)   # cooler
        @test t3.N_inp > 0
        t4 = ice_nucleation!(inp, 250.15, 1.2, 10.0; N_liq, M_liq)   # warmer
        @test t4.N_inp == 0.0
        # the cumulative ice equals a single step straight to -28 °C
        fresh = _parcel()
        once = ice_nucleation!(fresh, 245.15, 1.2, 10.0; N_liq, M_liq)
        @test t1.N_inp + t3.N_inp ≈ once.N_inp rtol = 1e-10
    end

    @testset "Zero nucleation: T ≥ 0 °C, and S_i ≤ 1 without liquid" begin
        for T in (_C.T_freeze, 274.0, 290.0)
            inp = _parcel()
            t = ice_nucleation!(inp, T, 1.5, 1.0; N_liq, M_liq)
            @test ice_number_formed(t) == 0 && t.M_ice == 0 && t.heat == 0
            @test inp_number(inp; population = :consumed) == 0
        end
        for T in (210.0, 225.0, 238.0), S in (0.8, 0.99, 1.0)
            inp = _parcel()
            t = ice_nucleation!(inp, T, S, 1.0)
            @test ice_number_formed(t) == 0 && t.M_vap == 0 && t.N_inp == 0
            @test inp_number(inp; population = :consumed) == 0
        end
    end

    @testset "Ice formed increases with supercooling" begin
        # At fixed air density (p ∝ T), so that the ambient INP numbers
        # describe the same air at every T: at fixed p the std-volume
        # conversion of the Tobo (2013) biological mode lowers its INP by
        # ≈ 0.06 % per K below its fitted range (see the n ∝ T^0.25 test
        # above), which is the ideal-gas law, not a freezing effect.
        Ts = collect(272.15:-0.5:233.65)
        Ni = [ice_number_formed(ice_nucleation!(_parcel(), T, 1.2, 1.0;
                                                p = p_air * T / 250.0, N_liq, M_liq))
              for T in Ts]
        @test all(diff(Ni) .>= -1e-14 .* Ni[2:end])
        @test Ni[1] == 0                         # -1 °C: warmer than every fit
        @test Ni[end] > 1e3 * Ni[findfirst(>(0), Ni)]
        Nd = [ice_number_formed(ice_nucleation!(_parcel(), T, 1.2, 1.0))
              for T in 240.0:-0.5:210.0]
        @test all(diff(Nd) .>= 0) && Nd[end] > Nd[1]
    end

    # ---------------------------------------------------------------------
    @testset "Consistency with the Phase 39 liquid_fraction ramp" begin
        # The regime boundaries are the ramp's endpoints, from one set of
        # constants: :liquid ⇔ λ = 1, :homogeneous ⇔ λ = 0.
        for T in 220.0:0.05:280.0
            λ = liquid_fraction(T)
            reg = freezing_regime(T)
            @test (reg === :liquid) == (λ == 1)
            @test (reg === :homogeneous) == (λ == 0)
            @test (reg === :heterogeneous) == (0 < λ < 1)
        end

        # Warm end, λ = 1: nothing freezes, however much INP and however
        # supersaturated.
        inp = inp_population([lognormal_mode(:mineral_dust, 100.0, 0.4, 2.0)])
        t = ice_nucleation!(inp, _C.T_freeze, 1.5, 1.0; N_liq, M_liq)
        @test liquid_fraction(_C.T_freeze) == 1
        @test t.M_liq == 0 && ice_number_formed(t) == 0 && t.N_inp == 0

        # Cold end, λ = 0: all liquid is ice, exactly — the frozen liquid
        # fraction equals 1 - λ = 1. Immersed INP are inside the frozen
        # drops (consumed), and form no extra crystals.
        for T in (_C.T_icenuc, _C.T_icenuc - 5)
            inp = _parcel()
            t = ice_nucleation!(inp, T, 1.0, 1.0; N_liq, M_liq)
            @test liquid_fraction(T) == 0
            @test t.M_liq / M_liq == 1 - liquid_fraction(T)
            @test t.N_ice_hom == N_liq && t.N_ice_imm == 0
            @test t.N_inp ≈ inp_number(_parcel()) rtol = 1e-12
            @test inp_number(inp) ≈ 0 atol = 1e-9
            @test t.heat ≈ latent_heat_fusion(T) * M_liq rtol = 1e-14
        end
        # … while deposition on interstitial INP continues below T_icenuc
        inp = _parcel()
        t = ice_nucleation!(inp, 225.0, 1.2, 1.0)
        @test t.N_ice_dep > 0 && t.N_ice_hom == 0

        # Just inside the ramp at the cold end: heterogeneous only, a
        # fraction of the liquid freezes.
        T = _C.T_icenuc + 0.01
        t = ice_nucleation!(_parcel(), T, 1.0, 1.0; N_liq, M_liq)
        @test 0 < t.M_liq / M_liq < 1 && t.N_ice_hom == 0

        # Inside the ramp, heterogeneous nucleation never freezes more of
        # the liquid than the ramp's ice fraction 1 - λ(T): checked for a
        # heavy dust load (10 cm⁻³, all immersed) — nucleation sets the
        # crystal number, and depositional growth (the host's) the rest.
        worst = 0.0
        for T in (_C.T_icenuc + 0.01):0.25:(_C.T_freeze - 0.01)
            inp = inp_population([lognormal_mode(:mineral_dust, 10.0, 0.4, 2.0)])
            t = ice_nucleation!(inp, T, 1.0, 1.0; N_liq, M_liq)
            f = t.M_liq / M_liq
            @test f <= 1 - liquid_fraction(T)
            worst = max(worst, f / (1 - liquid_fraction(T)))
        end
        @info "Heterogeneous frozen liquid fraction / (1 - λ), 10 cm⁻³ dust" worst

        # One set of constants: moving T_icenuc moves both together.
        c = ThermodynamicConstants{Float64}(T_icenuc = 238.15)
        @test liquid_fraction(236.0, c) == 0
        @test freezing_regime(236.0, c) === :homogeneous
        @test freezing_regime(236.0) === :heterogeneous
        t = ice_nucleation!(_parcel(), 236.0, 1.0, 1.0; N_liq, M_liq, c)
        @test t.M_liq == M_liq
        t = ice_nucleation!(_parcel(), 236.0, 1.0, 1.0; N_liq, M_liq)
        @test t.M_liq < M_liq
    end
end
