# Era IV, Phase 39 — exact moist thermodynamics.
#
# The load-bearing test here is CLAUSIUS-CLAPEYRON CONSISTENCY: that the
# saturation vapour pressure and the latent heat used everywhere else in
# the module satisfy d(ln e_s)/dT = L(T)/(R_v T²) to round-off. That is
# the module's central claim, and it is what makes moist static energy
# conserve instead of drifting.

using Test
using Aeolus.MoistThermodynamics
const MT = Aeolus.MoistThermodynamics

@testset "Era IV — Moist thermodynamics (Phase 39)" begin
    c = DEFAULT_CONSTANTS
    Rv = 8.314462618 / 0.018015268

    @testset "Constants, latent heats, saturation vapour pressure" begin
        @test MT.R_d(c)  ≈ 287.047 rtol=1e-5
        @test MT.R_v(c)  ≈ 461.523 rtol=1e-5
        @test latent_heat_vapor(273.16) ≈ 2.5008e6 rtol=1e-6
        @test latent_heat_fusion(273.15) ≈ 3.337e5 rtol=1e-3
        # sublimation = vaporisation + fusion, identically
        @test latent_heat_sublim(253.15) ≈
              latent_heat_vapor(253.15) + latent_heat_fusion(253.15) atol=1e-9
        # anchored at the triple point; liquid and ice curves meet there
        @test saturation_vapor_pressure(273.16) ≈ 611.657 rtol=1e-10
        @test saturation_vapor_pressure_ice(273.16) ≈ 611.657 rtol=1e-10
        # independent reference values (Smithsonian tables)
        @test saturation_vapor_pressure(293.15) ≈ 2339.0 rtol=2e-3
        @test saturation_vapor_pressure_ice(253.15) ≈ 103.3 rtol=5e-3
        # boiling point: a 100 K extrapolation, nothing fitted; the 1.6 %
        # shortfall is the constant-heat-capacity assumption's error.
        @test saturation_vapor_pressure(373.15) ≈ 101325.0 rtol=2e-2
        @test_throws DomainError saturation_vapor_pressure(-1.0)
        # Wegener-Bergeron-Findeisen gap: absolute maximum ~27 Pa near 261 K
        Ts = 230.0:0.5:273.15
        gmax, idx = findmax([saturation_vapor_pressure(T) - saturation_vapor_pressure_ice(T)
                             for T in Ts])
        @test 25.0 < gmax < 29.0 && 258.0 < Ts[idx] < 265.0
    end

    @testset "Clausius-Clapeyron consistency (the central claim)" begin
        # 4th-order central difference (a 2nd-order stencil's truncation
        # error, ~1.2e-8, would dominate the module's round-off).
        d4(f, T, h) = (f(T - 2h) - 8f(T - h) + 8f(T + h) - f(T + 2h)) / (12h)
        rel(fes, L, T) = (d = d4(x -> log(fes(x)), T, 1e-3T); r = L(T) / (Rv * T^2);
                          abs(d - r) / r)
        @test maximum(rel(saturation_vapor_pressure, latent_heat_vapor, T)
                      for T in 210.0:5.0:340.0) < 1e-10
        @test maximum(rel(saturation_vapor_pressure_ice, latent_heat_sublim, T)
                      for T in 200.0:5.0:273.0) < 1e-10
    end

    @testset "Phase partition, gas properties, potential temperatures" begin
        @test liquid_fraction(273.15) == 1.0 && liquid_fraction(233.15) == 0.0
        @test all(diff([liquid_fraction(T) for T in 230.0:1.0:276.0]) .>= -1e-15)
        q = phase_partition(250.0, 1e-2, 2e-3)
        @test q.liq + q.ice ≈ 2e-3 && q.liq > 0 && q.ice > 0
        @test gas_constant_moist(PhasePartition(0.0)) ≈ MT.R_d(c)
        @test exner_function(5.0e4) ≈ 0.5^(MT.R_d(c)/c.cp_dry) rtol=1e-12
        T, p = 300.0, 8.0e4
        q1 = PhasePartition(1e-2, 3e-3, 0.0)
        @test temperature_from_theta(potential_temperature(T, p, PhasePartition(0.0)),
                                     p, PhasePartition(0.0)) ≈ T rtol=1e-12
        @test temperature_from_theta_li(
                  liquid_ice_potential_temperature(T, p, q1), p, q1) ≈ T rtol=1e-9
        # density temperature includes condensate loading (~0.9 K at 3 g/kg)
        loading = density_temperature(T, PhasePartition(1e-2)) - density_temperature(T, q1)
        @test 0.5 < loading < 1.5
    end

    @testset "Saturation adjustment (diagnosis from θ_li)" begin
        p = 9.0e4
        # subsaturated: nothing condenses
        θli = liquid_ice_potential_temperature(295.0, p, PhasePartition(5e-3))
        T, q = saturation_adjustment(θli, p, 5e-3)
        @test q.liq == 0 && q.ice == 0 && T ≈ 295.0
        # supersaturated: cloud forms, heats, conserves water and θ_li
        qt = 2.0e-2
        θli = liquid_ice_potential_temperature(290.0, p, PhasePartition(qt))
        T, q = saturation_adjustment(θli, p, qt)
        @test q.liq > 0 && T > 290.0 && q.tot ≈ qt
        @test MT.vapor(q) ≈ saturation_specific_humidity(T, p) rtol=1e-6
        @test liquid_ice_potential_temperature(T, p, q) ≈ θli rtol=1e-9
        # cold: ice only
        θli_c = liquid_ice_potential_temperature(225.0, p, PhasePartition(3e-3))
        Tc, qc = saturation_adjustment(θli_c, p, 3e-3)
        @test qc.ice > 0 && qc.liq == 0
        @test liquid_ice_potential_temperature(Tc, p, qc) ≈ θli_c rtol=1e-9
        # negative total water is clamped, not propagated as NaN
        Tn, qn = saturation_adjustment(θli, p, -1e-4)
        @test isfinite(Tn) && qn.tot == 0
        # the diagnosis inverts θ_li to the bisection tolerance (1e-10 K
        # in T, ~3e-12 in θ_li) over 200-330 K, 300-1050 hPa
        worst = 0.0
        for T0 in 200.0:10.0:330.0, p in (3.0e4, 7.0e4, 1.05e5), S in (0.5, 1.3)
            qs0 = saturation_specific_humidity(T0, p)
            qs0 > 0.2 && continue
            θ = liquid_ice_potential_temperature(T0, p, PhasePartition(S * qs0))
            Ta, qa = saturation_adjustment(θ, p, S * qs0)
            worst = max(worst, abs(liquid_ice_potential_temperature(Ta, p, qa) - θ) / θ)
        end
        @test worst < 1e-11
    end

    # Issue #82: (b) isobaric_saturation_adjustment is a PROCESS and must
    # conserve moist enthalpy exactly, with non-adjusting rain and ice in c_pm.
    @testset "Moist enthalpy inverse and isobaric process adjustment" begin
        worst = 0.0; worst_slope = 0.0
        for T in 200.0:20.0:320.0, (qt, ql, qi) in ((0.0, 0.0, 0.0), (2e-2, 3e-3, 0.0),
                                                    (1e-2, 2e-3, 1e-3))
            q = PhasePartition(qt, ql, qi)
            h = moist_enthalpy(T, q)
            for Tr in (c.T_triple, 150.0)
                worst = max(worst, abs(temperature_from_moist_enthalpy(h, q; T_ref = Tr) - T) / T)
            end
            # the slope of h in T at fixed partition IS c_pm (Kirchhoff)
            worst_slope = max(worst_slope, abs((moist_enthalpy(T + 1, q) - h) /
                                               heat_capacity_moist(q) - 1))
        end
        @test worst < 1e-14
        @test worst_slope < 1e-9

        worst_h = 0.0; worst_sat = 0.0; n_cond = 0; n_evap = 0; bad = 0
        for T0 in 200.0:10.0:330.0, p in (3.0e4, 7.0e4, 1.05e5),
            S in (0.3, 1.02, 1.3), qc in (0.0, 3e-3), (qr, qi) in ((0.0, 0.0), (2e-3, 1e-3))
            qs0 = saturation_specific_humidity(T0, p)
            qs0 > 0.2 && continue
            qv = S * qs0
            r = isobaric_saturation_adjustment(T0, p, qv, qc; q_r = qr, q_i = qi)
            qt = qv + qc + qr + qi
            h0 = moist_enthalpy(T0, PhasePartition(qt, qc + qr, qi))
            h1 = moist_enthalpy(r.T, PhasePartition(qt, qc + r.Δq + qr, qi))
            worst_h = max(worst_h, abs(h1 - h0) / abs(h0))
            qv1 = qv - r.Δq; qc1 = qc + r.Δq; qs1 = saturation_specific_humidity(r.T, p)
            # end state: saturated with cloud, or cloud-free and subsaturated;
            # Δq bounded by what exists; heating has the sign of condensation
            if qc1 > 0
                worst_sat = max(worst_sat, abs(qv1 - qs1) / qs1)
            elseif qv1 > qs1 * (1 + 1e-12)
                bad += 1
            end
            (-qc <= r.Δq <= qv && sign(r.T - T0) == sign(r.Δq)) || (bad += 1)
            r.Δq > 0 && (n_cond += 1); r.Δq < 0 && (n_evap += 1)
        end
        @test n_cond > 50 && n_evap > 50
        @test worst_h < 1e-12
        @test worst_sat < 1e-8
        @test bad == 0
        # #85 exact-root regression: Newton hitting g = 0 exactly must keep
        # that root, not the bracket midpoint (which returned 1.143e-6).
        Tx, px, qx = 204.2477594857438, 72681.13710255225, 7.421642749398895e-6
        r = isobaric_saturation_adjustment(Tx, px, qx, 0.0; q_r = 1e-6)
        @test r.Δq ≈ 2.28605e-6 rtol = 1e-4
        @test abs(qx - r.Δq - saturation_specific_humidity(r.T, px)) < 1e-14
        # #82: leaving 5 g/kg rain (ice) out of c_pm overstates ΔT/Δq by
        # exactly the heat-capacity ratio (~1.5 %, ~0.5 %)
        T0, p = 285.0, 9.0e4
        qv = 1.1 * saturation_specific_humidity(T0, p)
        per(qr, qi) = (r = isobaric_saturation_adjustment(T0, p, qv, 0.0; q_r = qr, q_i = qi);
                       (r.T - T0) / r.Δq)
        cp(qr, qi) = heat_capacity_moist(PhasePartition(qv + qr + qi, qr, qi))
        @test per(0.0, 0.0) / per(5e-3, 0.0) ≈ cp(5e-3, 0.0) / cp(0.0, 0.0) rtol = 1e-4
        @test per(0.0, 0.0) / per(0.0, 5e-3) ≈ cp(0.0, 5e-3) / cp(0.0, 0.0) rtol = 1e-4
    end

    @testset "Saturated ascent conserves θ_li" begin
        p0, T0, qt = 1.0e5, 298.0, 1.6e-2
        θli = liquid_ice_potential_temperature(T0, p0, PhasePartition(qt))
        p = p0; T = T0; q = PhasePartition(qt); z = 0.0
        h0 = moist_static_energy(T, z, q); drift_θ = 0.0; drift_h = 0.0
        for _ in 1:20                      # 6 km in 300 m hydrostatic steps
            p -= air_density(T, p, q) * c.grav * 300.0; z += 300.0
            T, q = saturation_adjustment(θli, p, qt)
            drift_θ = max(drift_θ, abs(liquid_ice_potential_temperature(T, p, q) - θli))
            drift_h = max(drift_h, abs(moist_static_energy(T, z, q) - h0))
        end
        @test drift_θ < 1e-8
        # MSE is approximate (drops condensate enthalpy, ~1 % of h over 6 km)
        @test drift_h / h0 < 4e-2
        @test q.liq > 0
    end

    @testset "Derived diagnostics" begin
        @test dry_adiabatic_lapse_rate() ≈ 9.80665 / 1004.6 rtol=1e-12
        @test 3.0e-3 < moist_adiabatic_lapse_rate(300.0, 1.0e5) < 5.0e-3
        p, qv = 9.5e4, 1.0e-2
        Td = dewpoint_temperature(p, qv)
        @test relative_humidity(Td, p, qv) ≈ 1.0 rtol=1e-6
        @test frost_point_temperature(8e4, 1e-3) > dewpoint_temperature(8e4, 1e-3)
        ε_exact = 0.018015268 / 0.02896546
        @test vapor_pressure(qv, p) / p ≈ qv / (ε_exact + (1-ε_exact)*qv) rtol=1e-12
        T_lcl, p_lcl = lifting_condensation_level(300.0, 1.0e5, 1.2e-2)
        @test saturation_specific_humidity(T_lcl, p_lcl) ≈ 1.2e-2 rtol=1e-5
        @test 330.0 < equivalent_potential_temperature(300.0, 1e5, PhasePartition(1.5e-2)) < 360.0
        @test buoyancy(302.0, 1e5, PhasePartition(1.6e-2, 4e-3, 0.0), 300.0, 1e5,
                       PhasePartition(8e-3)) <
              buoyancy(302.0, 1e5, PhasePartition(1.6e-2), 300.0, 1e5, PhasePartition(8e-3))
        sounding = [(1.0e5 - 5000.0k, 300.0 - 6.5e-3 * (k * 5000.0 / 1.0e5 * 8000.0),
                     max(1e-5, 1.2e-2 * exp(-k/4))) for k in 0:16]
        cape, cin = cape_cin(301.0, 1.0e5, 1.6e-2, sounding)
        @test cape >= 0 && cin <= 0
        # inverse of RH, over a T × RH grid
        pa = 101325.0
        @test maximum(abs(relative_humidity(T, pa, specific_humidity_from_relative_humidity(T, pa, RH)) - RH)
                      for T in (263.15, 300.0, 313.15), RH in (0.0, 0.6, 1.0)) < 1e-12
    end

    @testset "Thermodynamic wet-bulb vs ASHRAE eq. 35" begin
        # ASHRAE Fundamentals (2017) ch. 1 eq. 35 gives W at (t, t*); our
        # wet bulb fed that W must return t* (measured 4.4e-3 K).
        p = 101325.0
        errs = map(((26.67, 19.44), (35.0, 23.89), (20.0, 12.0), (40.0, 30.0))) do (t, tw)
            Ws = saturation_mixing_ratio(tw + 273.15, p)
            W = ((2501 - 2.326tw) * Ws - 1.006 * (t - tw)) / (2501 + 1.86t - 4.186tw)
            abs(wet_bulb_temperature(t + 273.15, p, W / (1 + W)) - 273.15 - tw)
        end
        @test maximum(errs) < 0.02
        @test wet_bulb_temperature(300.0, p, saturation_specific_humidity(300.0, p)) == 300.0
    end

    @testset "Float32" begin
        c32 = ThermodynamicConstants{Float32}()
        @test saturation_vapor_pressure(273.16f0, c32) isa Float32
        @test heat_capacity_moist(PhasePartition(1f-2, 1f-3, 0f0), c32) isa Float32
    end
end
