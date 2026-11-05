# Phase 43, issue #6 — immersion and deposition freezing.
#
# Asserted: published INAS and INP values at reference temperatures, the
# analytic limits of the Poisson frozen fraction, zero nucleation where
# the physics forbids it (T ≥ 0 °C, S_i ≤ 1), and — for the source terms —
# conservation of INP number, water mass and energy, and consistency with
# the Phase 39 mixed-phase ramp.

using Test, Printf, Random
using Aeolus
using Aeolus.MoistThermodynamics: DEFAULT_CONSTANTS, liquid_fraction, latent_heat_fusion,
                                  latent_heat_sublim, saturation_vapor_pressure,
                                  saturation_vapor_pressure_ice

const _C = DEFAULT_CONSTANTS

# Independent midpoint quadrature in ln r of the INAS frozen number
# fraction of a log-normal (not the module's shifted-Gaussian rule).
function _ref_frozen_fraction(n_s, m::ARGMode; n = 100_000)
    lnσ = log(m.σ_g)
    lo = log(m.r_g) - 12lnσ
    h = 24lnσ / n
    s = 0.0
    for j in 1:n
        x = lo + (j - 0.5) * h
        s += -expm1(-n_s * 4π * exp(2x)) * exp(-0.5((x - log(m.r_g)) / lnσ)^2)
    end
    return s * h / (sqrt(2π) * lnσ)
end

@testset "Immersion and deposition freezing (issue #6)" begin

    @testset "Published INAS densities and INP concentrations" begin
        # Niemand et al. (2012) Eq. 5: n_s = exp(-0.517 T[°C] + 8.934)
        @test ice_active_site_density(Niemand2012(), 253.15) ≈ 2.347e8 rtol = 1e-3
        @test ice_active_site_density(Niemand2012(), 243.15) ≈ 4.14e10 rtol = 1e-2
        # Ullrich et al. (2017) immersion: Niemand × e^0.424 = 1.528
        @test ice_active_site_density(Ullrich2017Immersion(), 250.0) /
              ice_active_site_density(Niemand2012(), 250.0) ≈ 1.528 rtol = 1e-3
        # Ullrich et al. (2017) deposition, dust
        @test ice_active_site_density(Ullrich2017Deposition(), 220.0, 1.1) ≈ 2.662e8 rtol = 1e-3
        # DeMott et al. (2015) Eq. 2 (cf = 3) and Tobo et al. (2013) at -20 °C,
        # N_{>0.5μm} = 1 std cm⁻³, evaluated at standard density
        T = 253.15
        pstd = STANDARD_PRESSURE * T / STANDARD_TEMPERATURE
        @test singular_inp_concentration(DeMott2015(), T, 1e6; p = pstd) ≈
              3 * exp(0.46 * 20.01 - 11.6) * 1e3 rtol = 1e-12
        @test singular_inp_concentration(Tobo2013(), T, 1e6; p = pstd) ≈ 249.9 rtol = 1e-3
    end

    @testset "No ice at T ≥ 0 °C or S_i ≤ 1; monotone in supercooling" begin
        @test all(ice_active_site_density(s, T, 1.5) == 0
                  for s in (Niemand2012(), Ullrich2017Immersion(), Ullrich2017Deposition()),
                      T in (_C.T_freeze, 300.0))
        @test singular_inp_concentration(DeMott2015(), _C.T_freeze, 1e7) == 0
        @test all(ice_active_site_density(Ullrich2017Deposition(), T, S) == 0
                  for T in 206.0:2.0:240.0, S in (0.5, 1.0))
        ns = [ice_active_site_density(Niemand2012(), T) for T in _C.T_freeze:-0.5:200.0]
        @test all(diff(ns) .>= 0)
        # deposition clamped at water saturation
        Siw = saturation_vapor_pressure(225.0) / saturation_vapor_pressure_ice(225.0)
        @test ice_active_site_density(Ullrich2017Deposition(), 225.0, 5.0) ==
              ice_active_site_density(Ullrich2017Deposition(), 225.0, Siw)
    end

    @testset "Poisson frozen fraction: quadrature and small-n_s A limit" begin
        m = arg_mode(1e6, 0.4e-6, 2.0, 0.03; ρ_p = 2600.0)
        worst = maximum(abs(inas_frozen_fraction(Niemand2012(), m, T) /
                            _ref_frozen_fraction(ice_active_site_density(Niemand2012(), T), m) - 1)
                        for T in (250.0, 238.0))
        @test worst < 1e-8
        a = mode_surface_area(m) / m.N
        @test inas_frozen_fraction(Niemand2012(), m, 258.0) /
              (ice_active_site_density(Niemand2012(), 258.0) * a) ≈ 1 rtol = 1e-3
    end

    _parcel() = inp_population([lognormal_mode(:mineral_dust, 1.0, 0.4, 2.0),
                                lognormal_mode(:biological, 0.01, 1.0, 1.6)])
    N_liq, M_liq = 1e8, 2e-4
    p_air = 7.0e4

    @testset "INP consumed = ice formed; water, energy; no double counting" begin
        inp = _parcel()
        T = 248.15
        b0 = inp_budget(inp)
        t = ice_nucleation!(inp, T, 1.2, 10.0; p = p_air, N_liq, M_liq)
        @test t.N_inp > 0
        @test ice_number_formed(t) ≈ t.N_inp rtol = 1e-12
        @test t.N_inp ≈ inp_budget(inp).N_cons - b0.N_cons rtol = 1e-12
        @test maximum(abs, values(inp_budget_residual(inp))) < 1e-14
        # immersion: each frozen droplet carries the mean droplet mass, L_f
        @test t.M_liq ≈ t.N_ice_imm * M_liq / N_liq rtol = 1e-14
        @test t.heat ≈ latent_heat_fusion(T) * t.M_liq rtol = 1e-14
        # singular: the same temperature again freezes nothing
        @test ice_number_formed(ice_nucleation!(inp, T, 1.2, 10.0; p = p_air, N_liq, M_liq)) == 0
        # deposition out of cloud: vapour → ice at L_s
        t = ice_nucleation!(_parcel(), 225.0, 1.25, 5.0; p = 3.0e4)
        @test t.N_ice_dep > 0 && t.M_liq == 0
        @test t.heat / t.M_vap ≈ latent_heat_sublim(225.0) rtol = 1e-14
    end

    @testset "Consistency with the Phase 39 liquid_fraction ramp" begin
        @test all((freezing_regime(T) === :heterogeneous) == (0 < liquid_fraction(T) < 1)
                  for T in 220.0:0.25:280.0)
        # cold end, λ = 0: all liquid freezes homogeneously, exactly
        t = ice_nucleation!(_parcel(), _C.T_icenuc - 5, 1.0, 1.0; N_liq, M_liq)
        @test t.M_liq == M_liq && t.N_ice_hom == N_liq
        # in cloud below T_icenuc with every INP immersed none nucleates by
        # deposition (review fix: 2.8e5 m⁻³ extra crystals at 230 K before)
        T = 230.0
        Siw = saturation_vapor_pressure(T) / saturation_vapor_pressure_ice(T)
        t = ice_nucleation!(inp_population([lognormal_mode(:mineral_dust, 1.0, 0.4, 2.0)]),
                            T, Siw, 1.0; N_liq, M_liq)
        @test t.N_ice_dep == 0.0 && ice_number_formed(t) == N_liq
        # inside the ramp heterogeneous freezing never exceeds 1 - λ(T)
        @test all(T -> ice_nucleation!(inp_population([lognormal_mode(:mineral_dust, 10.0, 0.4, 2.0)]),
                                       T, 1.0, 1.0; N_liq, M_liq).M_liq / M_liq <= 1 - liquid_fraction(T),
                  (_C.T_icenuc + 0.01):1.0:(_C.T_freeze - 0.01))
    end
end
