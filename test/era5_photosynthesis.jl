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
    # form. The 25 °C values of that form (Kc 406.8, Ko 277.1, Γ* 42.93)
    # differ from the rounded CLM values (404.9, 278.4, 42.75) by < 0.5 %;
    # the *temperature shape* must agree to the precision of R.
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
    # The activation part of the CLM responses must reproduce them.
    for (Ha, c, ΔH) in ((65330.0, 26.35, 65.33), (46390.0, 18.72, 46.39))
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

end # Photosynthesis testset
