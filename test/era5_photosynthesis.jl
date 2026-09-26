# Phase 47 (#19) — Farquhar C3 / Collatz C4 leaf photosynthesis.
#
# Independent references used below (none of them is the code under test):
#   * Bernacchi et al. (2001), Plant Cell Environ. 24, 253-259, Table 1:
#     in-vivo temperature fits in the form  k(T) = exp(c - ΔHa / (R T)),
#     R = 8.314e-3 kJ mol⁻¹ K⁻¹, T in K, k in µmol mol⁻¹ (Kc, Γ*),
#     mmol mol⁻¹ (Ko) or relative to 25 °C (Vcmax, Rd). (c, ΔHa [kJ/mol]):
#         Kc (38.05, 79.43)   Ko (20.30, 36.38)   Γ* (19.02, 37.83)
#         Vcmax (26.35, 65.33)   Rd (18.72, 46.39)
#   * CLM4.5 Technical Note (Oleson et al. 2013, NCAR/TN-503+STR),
#     Table 8.1 (per-PFT CN_L, FLNR, SLA0, Vcmax25) and eq. 8.17.
#   * Closed-form limits of the Farquhar equations (derived in the tests).

using Test
using Aeolus
using Aeolus.Photosynthesis

const _Rkj = 8.314e-3                     # Bernacchi's R, kJ mol⁻¹ K⁻¹
_bernacchi(c, ΔHa, T) = exp(c - ΔHa / (_Rkj * T))

@testset "Photosynthesis (Phase 47, #19)" begin

@testset "temperature responses" begin
    # Normalisation: every response is 1 at 25 °C except the C4 Q10 form,
    # whose inhibition terms CLM does not renormalise.
    for r in (ArrheniusResponse(79430.0), PeakedArrheniusResponse(65330.0, 149250.0, 485.0))
        @test temperature_factor(r, 298.15) ≈ 1.0 atol = 1e-12
    end
    @test temperature_factor(Q10Response(), 308.15) ≈ 2.0 atol = 1e-12

    # Rubisco kinetics against Bernacchi et al. (2001)'s own functional
    # form. 404.9 / 278.4 / 42.75 are Bernacchi's tabulated 25 °C values
    # (Table 1), which CLM adopts; evaluating the exp(c − ΔHa/RT) fit with
    # the published two-decimal c gives 406.07 / 276.90 / 42.89 instead, a
    # difference of up to 0.54 % that is the rounding of c, not an error.
    # With identical ΔHa the *temperature shape* is a transcription check
    # of the activation energies and must agree to the precision of R.
    tc = bonan2011_c3_temperature()
    worst = 0.0
    for (resp, c, ΔH, k25) in ((tc.Kc, 38.05, 79.43, 404.9),
                               (tc.Ko, 20.30, 36.38, 278.4),
                               (tc.Γstar, 19.02, 37.83, 42.75))
        ref25 = _bernacchi(c, ΔH, 298.15)
        @test abs(ref25 / k25 - 1) < 1e-2                  # 25 °C value
        for Tc in 5.0:5.0:40.0
            T = Tc + 273.15
            ratio_model = temperature_factor(resp, T)
            ratio_ref = _bernacchi(c, ΔH, T) / ref25
            worst = max(worst, abs(ratio_model / ratio_ref - 1))
            # absolute value, CLM constants vs Bernacchi's fit
            @test abs(k25 * ratio_model / _bernacchi(c, ΔH, T) - 1) < 1e-2
        end
    end
    @test worst < 1e-3
    @info "Kc/Ko/Γ* temperature shape vs Bernacchi (2001), 5–40 °C" max_rel_err = worst
    Γ15 = 42.75 * temperature_factor(tc.Γstar, 288.15)
    Γ35 = 42.75 * temperature_factor(tc.Γstar, 308.15)
    @info "Γ* (µmol/mol) model vs Bernacchi" Γ15 ref15 = _bernacchi(19.02, 37.83, 288.15) Γ35 ref35 = _bernacchi(19.02, 37.83, 308.15)

    # Vcmax and Rd: Bernacchi's fits are pure Arrhenius (no deactivation).
    # The activation part of the shipped Bonan-2011 responses must
    # reproduce them (read ΔHa from the parameter set, not a literal).
    for (Ha, c, ΔH) in ((tc.Vcmax.ΔHa, 26.35, 65.33), (tc.Rd.ΔHa, 18.72, 46.39))
        for Tc in 10.0:5.0:40.0
            T = Tc + 273.15
            @test abs(temperature_factor(ArrheniusResponse(Ha), T) / _bernacchi(c, ΔH, T) - 1) < 1e-2
        end
    end
    # The full peaked (Bonan 2011) form does NOT reproduce Bernacchi's
    # Arrhenius below the optimum, and it should not be expected to: its
    # deactivation term is already active at 25 °C (it removes 13 % of
    # Vcmax there) and is renormalised away, which raises the curve below
    # 25 °C and lowers it above. Assert that sign structure and record the
    # size of the difference.
    dev = Dict{Float64,Float64}()
    for Tc in (10.0, 15.0, 20.0, 30.0, 35.0, 40.0)
        T = Tc + 273.15
        dev[Tc] = temperature_factor(tc.Vcmax, T) / _bernacchi(26.35, 65.33, T) - 1
    end
    @test dev[10.0] > 0 && dev[15.0] > 0 && dev[20.0] > 0
    @test dev[35.0] < 0 && dev[40.0] < dev[35.0]
    @test temperature_factor(tc.Vcmax, 313.15) < 0.8 * _bernacchi(26.35, 65.33, 313.15)
    @info "Bonan-2011 peaked Vcmax vs Bernacchi Arrhenius (relative)" dev
    # Optimum temperature: closed form (Medlyn et al. 2002 eq. 19) against a
    # brute-force maximisation of the same function.
    for r in (tc.Vcmax, tc.Jmax, kattge_knorr_c3_temperature(298.15).Vcmax)
        Topt = optimum_temperature(r)
        Ts = 280.0:0.001:330.0
        Tnum = Ts[argmax([temperature_factor(r, T) for T in Ts])]
        @test abs(Topt - Tnum) < 2e-3
    end
    @info "Vcmax optimum temperature (°C)" bonan2011 = optimum_temperature(tc.Vcmax) - 273.15 kattge_knorr_Tg25 = optimum_temperature(kattge_knorr_c3_temperature(298.15).Vcmax) - 273.15

    # Kattge & Knorr acclimation: the Vcmax and Jmax optima rise with growth
    # temperature, Jmax25/Vcmax25 falls, and both clamp outside 11-35 °C.
    Tg = (11.0:4.0:35.0) .+ 273.15
    topt_v = [optimum_temperature(kattge_knorr_c3_temperature(t).Vcmax) for t in Tg]
    topt_j = [optimum_temperature(kattge_knorr_c3_temperature(t).Jmax) for t in Tg]
    @test issorted(topt_v) && issorted(topt_j)
    @test topt_v[end] - topt_v[1] > 5.0
    @test kattge_knorr_jmax_ratio(298.15) ≈ 2.59 - 0.035 * 25
    @test kattge_knorr_jmax_ratio(273.15) == kattge_knorr_jmax_ratio(284.15)
    @test kattge_knorr_jmax_ratio(320.0) == kattge_knorr_jmax_ratio(308.15)
    @info "Kattge–Knorr Vcmax T_opt (°C) at growth 11 → 35 °C" topt_v[1] - 273.15 topt_v[end] - 273.15
end

@testset "co-limitation and electron transport" begin
    for (a, b) in ((3.0, 7.0), (5.0, 5.0), (0.0, 4.0), (12.0, 0.1))
        @test colimited_rate(a, b, 1.0) ≈ min(a, b) atol = 1e-12
        a + b > 0 && @test colimited_rate(a, b, 0.0) ≈ a * b / (a + b) atol = 1e-12
        for Θ in (0.0, 0.5, 0.7, 0.95, 0.98)
            x = colimited_rate(a, b, Θ)
            @test x <= min(a, b) + 1e-12
            @test Θ * x^2 - (a + b) * x + a * b ≈ 0 atol = 1e-9 * max(1.0, a * b)
            @test colimited_rate(3a, 3b, Θ) ≈ 3x                   # homogeneity
        end
    end
    @test colimited_rate(0.0, 0.0, 0.9) == 0.0
    # J: zero in the dark, initial slope Φ/2, saturates at Jmax.
    Jmax = 120.0
    @test electron_transport_rate(Jmax, 0.0) == 0.0
    @test electron_transport_rate(Jmax, 1e-3) / 1e-3 ≈ 0.5 * 0.85 rtol = 1e-4
    @test electron_transport_rate(Jmax, 1e9) ≈ Jmax rtol = 1e-6
    @test par_to_photon_flux(100.0) ≈ 460.0
end

@testset "C3 Farquhar: analytic limits" begin
    p = C3Parameters(60.0)
    T = 298.15
    c = leaf_capacities(p, T)
    # 25 °C capacities are the 25 °C parameters, and Ap = 3Tp ≈ Vcmax/2 (CLM).
    @test c.Vcmax ≈ 60.0 && c.Jmax ≈ 1.97 * 60 && c.Γstar ≈ 42.75 && c.Kc ≈ 404.9
    @test 3 * c.Tp ≈ 0.501 * c.Vcmax
    Km = c.Kc * (1 + 209 / c.Ko)

    # Darkness: no electron transport, so no gross assimilation: An = −Rd.
    for ci in (60.0, 280.0, 1000.0)
        r = leaf_assimilation(p, ci, 0.0, T)
        @test r.A == 0.0 && r.An == -c.Rd
        @test net_assimilation(C4Parameters(40.0), ci, 0.0, T) == -leaf_capacities(C4Parameters(40.0), T).Rd
    end
    # At ci = Γ* carboxylation exactly balances photorespiration.
    for Q in (50.0, 500.0, 2000.0)
        @test leaf_assimilation(p, c.Γstar, Q, T).A ≈ 0 atol = 1e-12
    end
    # Below Γ* gross assimilation is negative and continuous; at ci = 0 the
    # Rubisco-limited rate is −Vcmax Γ*/Km (Farquhar et al. 1980).
    @test leaf_assimilation(p, 0.5 * c.Γstar, 1500.0, T).A < 0
    @test leaf_assimilation(p, 0.0, 1e7, T; colimitation = false).A ≈ -c.Vcmax * c.Γstar / Km rtol = 1e-4
    @test abs(net_assimilation(p, c.Γstar - 1e-7, 1500.0, T) - net_assimilation(p, c.Γstar + 1e-7, 1500.0, T)) < 1e-6

    # Γ* ∝ O₂ (Γ* = O / (2 S_c/o)).
    @test photorespiratory_compensation_point(p, T; oi = 20.0) ≈ 42.75 * 20 / 209

    # CO₂ compensation point, Rubisco-limited: Γ = (Γ* Vcmax + Km Rd)/(Vcmax − Rd).
    Γ_analytic = (c.Γstar * c.Vcmax + Km * c.Rd) / (c.Vcmax - c.Rd)
    Γ_num = co2_compensation_point(p, 1e7, T; colimitation = false)
    @test Γ_num ≈ Γ_analytic rtol = 1e-8
    Γ_colim = co2_compensation_point(p, 1500.0, T)
    @test c.Γstar < Γ_colim < 60.0
    @test isnan(co2_compensation_point(p, 0.0, T))
    @info "C3 CO₂ compensation point at 25 °C (µmol/mol)" Γstar = c.Γstar Γ_rubisco_analytic = Γ_analytic Γ_colimited_Q1500 = Γ_colim

    # Initial quantum yield: dA/dQ → ½Φ_PSII (ci − Γ*)/(4ci + 8Γ*) at Q → 0.
    for ci in (150.0, 280.0, 800.0)
        dQ = 1e-3
        slope = (net_assimilation(p, ci, dQ, T) - net_assimilation(p, ci, 0.0, T)) / dQ
        expected = 0.5 * 0.85 * (ci - c.Γstar) / (4ci + 8c.Γstar)
        @test slope ≈ expected rtol = 2e-3
    end
    φ280 = 0.5 * 0.85 * (280 - c.Γstar) / (4 * 280 + 8c.Γstar)
    @info "C3 initial quantum yield at ci = 280, 25 °C (mol CO₂/mol absorbed photons)" φ280

    # Light saturation: An rises monotonically with Q to the J = Jmax limit.
    Qs = 10.0 .^ range(0, 7; length = 200)
    As = [net_assimilation(p, 280.0, Q, T) for Q in Qs]
    @test all(diff(As) .>= -1e-12)
    wc = c.Vcmax / (280 + Km); wj = c.Jmax / (4 * 280 + 8c.Γstar)
    Asat = colimited_rate((280 - c.Γstar) * colimited_rate(wc, wj, 0.98), 3c.Tp, 0.95) - c.Rd
    @test As[end] ≈ Asat rtol = 1e-4
    @test net_assimilation(p, 280.0, 2000.0, T) > 0.9 * Asat

    # A–ci transition between Rubisco and RuBP-regeneration limitation
    # (Farquhar minimum): Vcmax(4ci + 8Γ*) = J(ci + Km).
    Q = 400.0
    J = electron_transport_rate(c.Jmax, Q)
    ci_t = (J * Km - 8c.Γstar * c.Vcmax) / (4c.Vcmax - J)
    @test leaf_assimilation(p, ci_t - 1, Q, T; colimitation = false).limitation == :rubisco
    @test leaf_assimilation(p, ci_t + 1, Q, T; colimitation = false).limitation == :light
    r = leaf_assimilation(p, ci_t, Q, T; colimitation = false)
    @test r.Ac ≈ r.Aj rtol = 1e-10
    @info "C3 Rubisco→RuBP transition ci at Q = 400, 25 °C" ci_t
    # TPU. Aj can never exceed Jmax/4 (ci → ∞), so with CLM's Tp25 =
    # 0.167 Vcmax25 export limitation needs Jmax25/Vcmax25 > 4·3·0.167 =
    # 2.004 at 25 °C. The default 1.97 is just below: no TPU limitation
    # anywhere on the A–ci curve. With the Kattge–Knorr ratio of a
    # cool-grown leaf (2.2 at 11 °C) the A–ci curve does plateau at 3Tp.
    @test c.Jmax / 4 < 3c.Tp
    @test all(leaf_assimilation(p, ci, 1e6, T; colimitation = false).limitation != :export
              for ci in 50.0:50.0:5000.0)
    pk = C3Parameters(60.0; T_growth = 284.15)
    @test pk.Jmax25 / pk.Vcmax25 ≈ 2.205
    ck = leaf_capacities(pk, T)
    r = leaf_assimilation(pk, 3000.0, 1e6, T; colimitation = false)
    @test r.limitation == :export && r.A ≈ 3ck.Tp

    # Co-limitation never exceeds the Farquhar minimum.
    for ci in 50.0:50.0:1500.0, Q in (100.0, 800.0, 2000.0)
        @test net_assimilation(p, ci, Q, T) <= net_assimilation(p, ci, Q, T; colimitation = false) + 1e-10
    end
    # A–ci curve: monotone in ci.
    Aci = [net_assimilation(p, ci, 1500.0, T) for ci in 0.0:10.0:1500.0]
    @test all(diff(Aci) .>= -1e-12)

    # Warburg effect: 2 % O₂ suppresses photorespiration and raises
    # Rubisco-limited assimilation.
    @test net_assimilation(p, 200.0, 1500.0, T; oi = 20.0) > 1.3 * net_assimilation(p, 200.0, 1500.0, T)
end

@testset "C4 Collatz: analytic limits" begin
    p = C4Parameters(40.0)
    T = 298.15
    c = leaf_capacities(p, T)
    # CLM does not renormalise the C4 inhibition terms at 25 °C.
    @test c.Vcmax / 40.0 ≈ 1 / ((1 + exp(0.3 * (298.15 - 313.15))) * (1 + exp(0.2 * (288.15 - 298.15)))) rtol = 1e-12
    @test c.kp ≈ 20000 * 40 && c.Rd / 40 ≈ 0.025 / (1 + exp(1.3 * (298.15 - 328.15)))
    # Initial quantum yield is α.
    slope = (net_assimilation(p, 300.0, 1e-3, T) - net_assimilation(p, 300.0, 0.0, T)) / 1e-3
    @test slope ≈ p.α rtol = 1e-3
    # CO₂ compensation point: kp Γ = Rd (Farquhar minimum), a few µmol/mol.
    Γ4 = co2_compensation_point(p, 1500.0, T; colimitation = false)
    @test Γ4 ≈ c.Rd / c.kp * 1e6 rtol = 1e-8
    Γ4c = co2_compensation_point(p, 1500.0, T)
    @test Γ4c < 5.0
    @info "C4 CO₂ compensation point (µmol/mol)" Γ_min = Γ4 Γ_colimited = Γ4c
    # CO₂ saturation: the carbon-concentrating mechanism saturates a C4 leaf
    # near ambient ci, and O₂ does not matter.
    A150 = net_assimilation(p, 150.0, 1500.0, T); A1000 = net_assimilation(p, 1000.0, 1500.0, T)
    @test A150 > 0.9 * A1000
    @test net_assimilation(p, 150.0, 1500.0, T; oi = 20.0) == A150
    # Light-saturated, CO₂-saturated → Vcmax.
    @test leaf_assimilation(p, 1e5, 1e7, T; colimitation = false).A ≈ c.Vcmax
    # Temperature: C4 Vcmax peaks between the two inhibition thresholds.
    Ts = 273.15:0.1:323.15
    Topt4 = Ts[argmax([leaf_capacities(p, t).Vcmax for t in Ts])]
    @test 303.15 < Topt4 < 313.15
    @info "C4 Vcmax optimum (°C)" Topt4 - 273.15

    # Collatz et al. (1992) leaf: light- and CO₂-saturated An = Vmax − Rd.
    pc = collatz1992_c4()
    @test pc.Vcmax25 == 39.0 && pc.kp25 == 0.7e6 && pc.Rd25 == 0.8
    @test leaf_assimilation(pc, 1e5, 1e8, 298.15; colimitation = false).A ≈ leaf_capacities(pc, 298.15).Vcmax

    # C3 vs C4 at the same Vcmax25 at 35 °C, ci = 0.4 ca vs 0.7 ca for
    # ca = 400: photorespiration makes the C3 leaf lose.
    p3 = C3Parameters(40.0)
    @test net_assimilation(p, 160.0, 1500.0, 308.15) > net_assimilation(p3, 280.0, 1500.0, 308.15)
end

@testset "CLM4.5 plant functional types" begin
    # Transcription check: every Vcmax25 of CLM4.5 Table 8.1 follows from
    # the same row's CN_L, SLA0 and FLNR through eq. 8.17 (FNR = 7.16,
    # aR25 = 60). A mistyped digit in any of the four columns fails here.
    @test length(CLM45_PFT_PHOTOSYNTHESIS) == 19
    worst = 0.0
    for (k, t) in CLM45_PFT_PHOTOSYNTHESIS
        v = vcmax25_from_leaf_nitrogen(t.CN_L, t.SLA0, t.FLNR)
        worst = max(worst, abs(v / t.Vcmax25 - 1))
        @test abs(v / t.Vcmax25 - 1) < 1e-3
        @test t.pathway in (:C3, :C4)
        @test (t.pathway === :C4) == !isnan(t.α)
    end
    @info "CLM4.5 Table 8.1: Vcmax25 from leaf N vs tabulated (max rel diff)" worst
    @test pft_photosynthesis_parameters(:c4_grass) isa C4Parameters
    @test pft_photosynthesis_parameters(:corn) isa C4Parameters
    @test pft_photosynthesis_parameters(:c3_grass) isa C3Parameters
    @test pft_photosynthesis_parameters(:net_boreal).Vcmax25 == 62.6
    @test pft_photosynthesis_parameters(:c3_grass; T_growth = 298.15).Jmax25 ≈ (2.59 - 0.035 * 25) * 78.2
    @test_throws ArgumentError pft_photosynthesis_parameters(:palm_tree)

    # Light-saturated net assimilation at 25 °C, ci/ca = 0.7 (C3) and 0.3
    # (C4) of ca = 400 µmol/mol, absorbed Q = 1500 µmol m⁻² s⁻¹. Ordered by
    # Vcmax25 within a pathway, and C4 grass above C3 grass despite a
    # lower Vcmax25.
    An = Dict(k => net_assimilation(pft_photosynthesis_parameters(k),
                                    CLM45_PFT_PHOTOSYNTHESIS[k].pathway === :C4 ? 120.0 : 280.0,
                                    1500.0, 298.15) for k in keys(CLM45_PFT_PHOTOSYNTHESIS))
    c3 = sort([k for k in keys(An) if CLM45_PFT_PHOTOSYNTHESIS[k].pathway === :C3],
              by = k -> CLM45_PFT_PHOTOSYNTHESIS[k].Vcmax25)
    @test issorted([An[k] for k in c3])
    @test An[:c4_grass] > An[:c3_grass]
    @info "An at 25 °C, Q = 1500 (µmol m⁻² s⁻¹)" c3_grass = An[:c3_grass] c4_grass = An[:c4_grass] net_temperate = An[:net_temperate] bdt_tropical = An[:bdt_tropical] crop = An[:crop] corn = An[:corn]
end

end # Photosynthesis testset
