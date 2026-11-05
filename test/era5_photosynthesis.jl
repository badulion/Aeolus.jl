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
    @test temperature_factor(PeakedArrheniusResponse(65330.0, 149250.0, 485.0), 298.15) ≈ 1.0 atol = 1e-12
    @test temperature_factor(Q10Response(), 308.15) ≈ 2.0 atol = 1e-12
    # Rubisco kinetics: temperature shape vs Bernacchi et al. (2001)'s fit
    # (the 25 °C values differ ≤ 0.54 % by the rounding of the published c)
    tc = bonan2011_c3_temperature()
    worst = 0.0; worst_abs = 0.0
    for (resp, c, ΔH, k25) in ((tc.Kc, 38.05, 79.43, 404.9), (tc.Ko, 20.30, 36.38, 278.4),
                               (tc.Γstar, 19.02, 37.83, 42.75)), Tc in (5.0, 20.0, 40.0)
        T = Tc + 273.15
        worst = max(worst, abs(temperature_factor(resp, T) /
                               (_bernacchi(c, ΔH, T) / _bernacchi(c, ΔH, 298.15)) - 1))
        worst_abs = max(worst_abs, abs(k25 * temperature_factor(resp, T) / _bernacchi(c, ΔH, T) - 1))
    end
    @test worst < 1e-3
    @test worst_abs < 1e-2
    # optimum temperature: closed form (Medlyn et al. 2002 eq. 19) vs brute force
    r = tc.Vcmax
    Ts = 280.0:0.001:330.0
    @test abs(optimum_temperature(r) - Ts[argmax([temperature_factor(r, T) for T in Ts])]) < 2e-3
    @test kattge_knorr_jmax_ratio(298.15) ≈ 2.59 - 0.035 * 25
end

@testset "co-limitation and electron transport" begin
    # x is the smaller root of Θx² − (a+b)x + ab = 0
    worst = 0.0
    for (a, b) in ((3.0, 7.0), (12.0, 0.1)), Θ in (0.0, 0.7, 0.98)
        x = colimited_rate(a, b, Θ)
        worst = max(worst, abs(Θ * x^2 - (a + b) * x + a * b) / (a * b), max(x - min(a, b), 0.0))
    end
    @test worst < 1e-9
    @test colimited_rate(3.0, 7.0, 1.0) ≈ 3.0 atol = 1e-12
    # J: initial slope Φ/2, saturates at Jmax
    @test electron_transport_rate(120.0, 1e-3) / 1e-3 ≈ 0.5 * 0.85 rtol = 1e-4
    @test electron_transport_rate(120.0, 1e9) ≈ 120.0 rtol = 1e-6
end

@testset "C3 Farquhar: analytic limits" begin
    p = C3Parameters(60.0)
    T = 298.15
    c = leaf_capacities(p, T)
    Km = c.Kc * (1 + 209 / c.Ko)
    # darkness: An = −Rd; at ci = Γ* carboxylation balances photorespiration
    @test leaf_assimilation(p, 280.0, 0.0, T).An == -c.Rd
    @test leaf_assimilation(p, c.Γstar, 500.0, T).A ≈ 0 atol = 1e-12
    # at ci = 0 the Rubisco-limited rate is −Vcmax Γ*/Km (Farquhar et al. 1980)
    @test leaf_assimilation(p, 0.0, 1e7, T; colimitation = false).A ≈ -c.Vcmax * c.Γstar / Km rtol = 1e-4
    # CO₂ compensation point, Rubisco-limited: (Γ* Vcmax + Km Rd)/(Vcmax − Rd)
    @test co2_compensation_point(p, 1e7, T; colimitation = false) ≈
          (c.Γstar * c.Vcmax + Km * c.Rd) / (c.Vcmax - c.Rd) rtol = 1e-8
    # initial quantum yield ½Φ (ci − Γ*)/(4ci + 8Γ*)
    ci = 280.0
    slope = (net_assimilation(p, ci, 1e-3, T) - net_assimilation(p, ci, 0.0, T)) / 1e-3
    @test slope ≈ 0.5 * 0.85 * (ci - c.Γstar) / (4ci + 8c.Γstar) rtol = 2e-3
    # Rubisco → RuBP-regeneration transition: Vcmax(4ci + 8Γ*) = J(ci + Km)
    J = electron_transport_rate(c.Jmax, 400.0)
    ci_t = (J * Km - 8c.Γstar * c.Vcmax) / (4c.Vcmax - J)
    r = leaf_assimilation(p, ci_t, 400.0, T; colimitation = false)
    @test r.Ac ≈ r.Aj rtol = 1e-10
    # TPU limitation with a cool-grown leaf (Kattge–Knorr Jmax/Vcmax = 2.2)
    pk = C3Parameters(60.0; T_growth = 284.15)
    r = leaf_assimilation(pk, 3000.0, 1e6, T; colimitation = false)
    @test r.limitation == :export && r.A ≈ 3leaf_capacities(pk, T).Tp
    # co-limitation never exceeds the Farquhar minimum; A–ci monotone
    @test all(net_assimilation(p, ci, Q, T) <= net_assimilation(p, ci, Q, T; colimitation = false) + 1e-10
              for ci in 50.0:150.0:1500.0, Q in (100.0, 2000.0))
    @test all(diff([net_assimilation(p, ci, 1500.0, T) for ci in 0.0:50.0:1500.0]) .>= -1e-12)
end

@testset "C4 Collatz: analytic limits" begin
    p = C4Parameters(40.0)
    T = 298.15
    c = leaf_capacities(p, T)
    # initial quantum yield is α; compensation point kp Γ = Rd
    slope = (net_assimilation(p, 300.0, 1e-3, T) - net_assimilation(p, 300.0, 0.0, T)) / 1e-3
    @test slope ≈ p.α rtol = 1e-3
    @test co2_compensation_point(p, 1500.0, T; colimitation = false) ≈ c.Rd / c.kp * 1e6 rtol = 1e-8
    # light- and CO₂-saturated → Vcmax; O₂ does not matter
    @test leaf_assimilation(p, 1e5, 1e7, T; colimitation = false).A ≈ c.Vcmax
    @test net_assimilation(p, 150.0, 1500.0, T; oi = 20.0) == net_assimilation(p, 150.0, 1500.0, T)
end

@testset "CLM4.5 plant functional types" begin
    # every Vcmax25 of Table 8.1 follows from the same row's CN_L, SLA0 and
    # FLNR through eq. 8.17: a mistyped digit in any column fails here
    @test length(CLM45_PFT_PHOTOSYNTHESIS) == 19
    @test maximum(abs(vcmax25_from_leaf_nitrogen(t.CN_L, t.SLA0, t.FLNR) / t.Vcmax25 - 1)
                  for t in values(CLM45_PFT_PHOTOSYNTHESIS)) < 1e-3
    @test pft_photosynthesis_parameters(:c4_grass) isa C4Parameters
    @test net_assimilation(pft_photosynthesis_parameters(:c4_grass), 120.0, 1500.0, 298.15) >
          net_assimilation(pft_photosynthesis_parameters(:c3_grass), 280.0, 1500.0, 298.15)
end

end # Photosynthesis testset
