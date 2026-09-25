# Era IV, Phase 39 — exact moist thermodynamics.
#
# The load-bearing test here is CLAUSIUS-CLAPEYRON CONSISTENCY: that the
# saturation vapour pressure and the latent heat used everywhere else in
# the module satisfy d(ln e_s)/dT = L(T)/(R_v T²) to round-off. That is
# the module's central claim, and it is what makes moist static energy
# conserve instead of drifting.

using Test
using Printf
using Aeolus.MoistThermodynamics
const MT = Aeolus.MoistThermodynamics

@testset "Era IV — Moist thermodynamics (Phase 39)" begin
    c = DEFAULT_CONSTANTS
    Rv = 8.314462618 / 0.018015268

    @testset "Derived constants" begin
        @test MT.R_d(c)  ≈ 287.047 rtol=1e-5
        @test MT.R_v(c)  ≈ 461.523 rtol=1e-5
        @test MT.molmass_ratio(c) ≈ 0.62196 rtol=1e-4
        @test MT.LH_f0(c) ≈ 3.336e5 rtol=1e-3
    end

    @testset "Latent heats (Kirchhoff)" begin
        @test latent_heat_vapor(273.16) ≈ 2.5008e6 rtol=1e-6
        @test latent_heat_vapor(273.15) ≈ 2.501e6  rtol=1e-3
        @test latent_heat_vapor(373.15) ≈ 2.257e6  rtol=6e-3
        @test latent_heat_fusion(273.15) ≈ 3.337e5 rtol=1e-3
        # L_v FALLS with temperature (c_pv < c_l)
        @test latent_heat_vapor(223.15) > latent_heat_vapor(273.15) >
              latent_heat_vapor(323.15)
        # L_f RISES with temperature
        @test latent_heat_fusion(233.15) < latent_heat_fusion(273.15)
        # sublimation = vaporisation + fusion, identically
        for T in (233.15, 253.15, 273.15)
            @test latent_heat_sublim(T) ≈
                  latent_heat_vapor(T) + latent_heat_fusion(T) atol=1e-9
        end
        @info "latent heats" Lv_273=latent_heat_vapor(273.15) Lv_373=latent_heat_vapor(373.15) Lf_273=latent_heat_fusion(273.15)
    end

    @testset "Saturation vapour pressure" begin
        # Anchored at the triple point by construction.
        @test saturation_vapor_pressure(273.16) ≈ 611.657 rtol=1e-10
        @test saturation_vapor_pressure_ice(273.16) ≈ 611.657 rtol=1e-10
        # Liquid and ice curves must MEET at the triple point.
        @test saturation_vapor_pressure(273.16) ≈
              saturation_vapor_pressure_ice(273.16) rtol=1e-12

        # Independent reference values (Smithsonian tables).
        @test saturation_vapor_pressure(273.15) ≈ 611.2  rtol=2e-3
        @test saturation_vapor_pressure(293.15) ≈ 2339.0 rtol=2e-3
        @test saturation_vapor_pressure(303.15) ≈ 4246.0 rtol=5e-3
        @test saturation_vapor_pressure_ice(253.15) ≈ 103.3 rtol=5e-3
        @test saturation_vapor_pressure_ice(233.15) ≈ 12.84 rtol=1e-2

        # The boiling point is a 100 K EXTRAPOLATION from the anchor and
        # nothing was fitted to it, so it is a genuine external check.
        # Constant heat capacities are the limiting assumption; the 1.6 %
        # shortfall is that assumption's error, not a coding error, and it
        # is recorded rather than tuned away.
        e_boil = saturation_vapor_pressure(373.15)
        @test e_boil ≈ 101325.0 rtol=2e-2
        @info "boiling point (100 K extrapolation, nothing fitted)" e_s_373=e_boil ref=101325.0 rel_error=(e_boil-101325)/101325

        @test saturation_vapor_pressure(200.0) > 0
        @test_throws DomainError saturation_vapor_pressure(-1.0)
    end

    @testset "Clausius-Clapeyron consistency (the central claim)" begin
        # d(ln e_s)/dT must equal L(T)/(R_v T²) to round-off, because e_s
        # is the analytic integral of exactly that ODE. A fitted e_s paired
        # with an independent L would fail this at the percent level.
        # 4th-order central difference. A 2nd-order stencil leaves a
        # truncation error of ~1.2e-8 relative here, which is the
        # DIFFERENCING error, not the module's — using it would mean the
        # test measured my finite difference instead of the physics.
        d4(f, T, h) = (f(T - 2h) - 8f(T - h) + 8f(T + h) - f(T + 2h)) / (12h)
        worst_liq = 0.0; worst_ice = 0.0
        for T in 210.0:5.0:340.0
            h = 1e-3 * T
            dlnes = d4(x -> log(saturation_vapor_pressure(x)), T, h)
            rhs = latent_heat_vapor(T) / (Rv * T^2)
            worst_liq = max(worst_liq, abs(dlnes - rhs) / rhs)
        end
        for T in 200.0:5.0:273.0
            h = 1e-3 * T
            dlnes = d4(x -> log(saturation_vapor_pressure_ice(x)), T, h)
            rhs = latent_heat_sublim(T) / (Rv * T^2)
            worst_ice = max(worst_ice, abs(dlnes - rhs) / rhs)
        end
        @test worst_liq < 1e-10
        @test worst_ice < 1e-10
        @info "Clausius-Clapeyron residual (relative)" liquid=worst_liq ice=worst_ice
    end

    @testset "Wegener-Bergeron-Findeisen gap" begin
        # Below freezing, e_s over ice is LOWER than over liquid. That gap
        # is what lets ice grow while droplets evaporate in a mixed-phase
        # cloud. Its ABSOLUTE maximum is ~27 Pa near 261 K (the RATIO,
        # by contrast, falls monotonically — a metric worth not confusing).
        for T in 235.0:5.0:270.0
            @test saturation_vapor_pressure_ice(T) < saturation_vapor_pressure(T)
        end
        Ts = 230.0:0.5:273.15
        gaps = [saturation_vapor_pressure(T) - saturation_vapor_pressure_ice(T)
                for T in Ts]
        gmax, idx = findmax(gaps)
        @test 25.0 < gmax < 29.0
        @test 258.0 < Ts[idx] < 265.0
        # the ratio is monotone in T, unlike the difference
        ratios = [saturation_vapor_pressure_ice(T) / saturation_vapor_pressure(T)
                  for T in 235.0:1.0:273.0]
        @test all(diff(ratios) .> 0)
        @info "WBF gap" max_Pa=gmax at_K=Ts[idx]
    end

    @testset "Phase partition and liquid fraction" begin
        @test liquid_fraction(300.0) == 1.0
        @test liquid_fraction(273.15) == 1.0
        @test liquid_fraction(233.15) == 0.0
        @test liquid_fraction(200.0)  == 0.0
        @test 0 < liquid_fraction(253.0) < 1
        # monotone through the mixed-phase ramp
        λs = [liquid_fraction(T) for T in 230.0:1.0:276.0]
        @test all(diff(λs) .>= -1e-15)
        # smoothstep ⇒ zero slope at both ends, so no kink in the heating
        h = 1e-4
        @test abs(liquid_fraction(233.15 + h) - liquid_fraction(233.15)) / h < 1e-2
        @test abs(liquid_fraction(273.15) - liquid_fraction(273.15 - h)) / h < 1e-2

        q = phase_partition(250.0, 1e-2, 2e-3)
        @test q.liq + q.ice ≈ 2e-3
        @test MT.vapor(q) ≈ 8e-3
        @test q.liq > 0 && q.ice > 0        # genuinely mixed phase
    end

    @testset "Moist gas properties" begin
        q = PhasePartition(1.5e-2, 2e-3, 0.0)
        # moist air: larger R, larger c_p than dry
        @test gas_constant_moist(q) > MT.R_d(c)
        @test heat_capacity_moist(q) > c.cp_dry
        @test gas_constant_moist(PhasePartition(0.0)) ≈ MT.R_d(c)
        @test heat_capacity_moist(PhasePartition(0.0)) ≈ c.cp_dry
        # condensate dilutes R (mass without vapour)
        @test gas_constant_moist(PhasePartition(1.5e-2, 1.4e-2, 0.0)) <
              gas_constant_moist(PhasePartition(1.5e-2, 0.0, 0.0))
        @test exner_function(1.0e5) ≈ 1.0
        @test exner_function(1.0e5, PhasePartition(0.0)) ≈ 1.0
        @test exner_function(5.0e4) ≈ 0.5^(MT.R_d(c)/c.cp_dry) rtol=1e-12
    end

    @testset "Potential temperatures" begin
        T, p = 300.0, 8.0e4
        q_dry = PhasePartition(0.0)
        θ = potential_temperature(T, p, q_dry)
        @test θ > T                                   # p < p_ref ⇒ θ > T
        @test temperature_from_theta(θ, p, q_dry) ≈ T rtol=1e-12
        @test potential_temperature(T, 1.0e5, q_dry) ≈ T rtol=1e-12

        # θ_li reduces to θ with no condensate
        q0 = PhasePartition(1e-2, 0.0, 0.0)
        @test liquid_ice_potential_temperature(T, p, q0) ≈
              potential_temperature(T, p, q0) rtol=1e-12
        # condensate depresses θ_li
        q1 = PhasePartition(1e-2, 3e-3, 0.0)
        @test liquid_ice_potential_temperature(T, p, q1) <
              potential_temperature(T, p, q1)
        # ice depresses it more than liquid (L_s > L_v)
        q2 = PhasePartition(1e-2, 0.0, 3e-3)
        @test liquid_ice_potential_temperature(250.0, p, q2) <
              liquid_ice_potential_temperature(250.0, p,
                  PhasePartition(1e-2, 3e-3, 0.0))
        # round trip
        @test temperature_from_theta_li(
                  liquid_ice_potential_temperature(T, p, q1), p, q1) ≈ T rtol=1e-9

        # density temperature must include condensate loading
        qv_only = PhasePartition(1e-2, 0.0, 0.0)
        qv_cond = PhasePartition(1e-2, 3e-3, 0.0)
        @test virtual_temperature(T, qv_only) > T
        @test density_temperature(T, qv_cond) < density_temperature(T, qv_only)
        loading = density_temperature(T, qv_only) - density_temperature(T, qv_cond)
        @test 0.5 < loading < 1.5          # ~0.9 K at 3 g/kg
        @info "condensate loading at 3 g/kg" delta_T_rho=loading
    end

    @testset "Saturation adjustment" begin
        p = 9.0e4
        # --- subsaturated: nothing should condense --------------------
        qt = 5e-3
        θli = liquid_ice_potential_temperature(295.0, p, PhasePartition(qt))
        T, q = saturation_adjustment(θli, p, qt)
        @test q.liq == 0 && q.ice == 0
        @test T ≈ 295.0 rtol=1e-9

        # --- supersaturated: cloud forms and heats the parcel ---------
        qt = 2.0e-2
        T0 = 290.0
        θli = liquid_ice_potential_temperature(T0, p, PhasePartition(qt))
        T, q = saturation_adjustment(θli, p, qt)
        @test q.liq > 0
        @test T > T0                       # latent heating
        @test q.tot ≈ qt                   # total water conserved exactly
        @test MT.vapor(q) ≈ saturation_specific_humidity(T, p) rtol=1e-6
        @info "saturation adjustment (warm)" T_before=T0 T_after=T q_liq=q.liq heating=T-T0

        # --- the invariant: θ_li is unchanged by the adjustment -------
        @test liquid_ice_potential_temperature(T, p, q) ≈ θli rtol=1e-9

        # --- cold: ice forms instead ----------------------------------
        qt = 3e-3
        θli_c = liquid_ice_potential_temperature(225.0, p, PhasePartition(qt))
        Tc, qc = saturation_adjustment(θli_c, p, qt)
        @test qc.ice > 0
        @test qc.liq == 0                  # below the homogeneous threshold
        @test liquid_ice_potential_temperature(Tc, p, qc) ≈ θli_c rtol=1e-9

        # --- mixed phase ----------------------------------------------
        θli_m = liquid_ice_potential_temperature(258.0, p, PhasePartition(5e-3))
        Tm, qm = saturation_adjustment(θli_m, p, 5e-3)
        if qm.liq + qm.ice > 0
            @test qm.liq > 0 && qm.ice > 0
        end

        # --- ice-only variant ------------------------------------------
        Ti, qi = saturation_adjustment_ice(θli_c, p, qt)
        @test qi.liq == 0
        @test liquid_ice_potential_temperature(Ti, p, qi) ≈ θli_c rtol=1e-9

        # --- edge cases -------------------------------------------------
        T_zero, q_zero = saturation_adjustment(
            liquid_ice_potential_temperature(280.0, p, PhasePartition(0.0)), p, 0.0)
        @test q_zero.liq == 0 && q_zero.ice == 0
        @test T_zero ≈ 280.0 rtol=1e-9
        # negative total water must be clamped, not propagate a NaN
        Tn, qn = saturation_adjustment(θli, p, -1e-4)
        @test isfinite(Tn) && qn.tot == 0
    end

    # Issue #82: two different operations, two different invariants.
    #   (a) saturation_adjustment            — DIAGNOSIS from prognostic θ_li,
    #       must invert the θ_li definition exactly;
    #   (b) isobaric_saturation_adjustment   — PROCESS, must conserve moist
    #       enthalpy exactly, with non-adjusting rain and ice in c_pm.
    @testset "Enthalpy inverse: temperature_from_moist_enthalpy" begin
        worst = 0.0
        for T in 200.0:10.0:330.0, (qt, ql, qi) in ((0.0, 0.0, 0.0),
                (1e-2, 0.0, 0.0), (2e-2, 3e-3, 0.0), (1e-2, 2e-3, 1e-3),
                (5e-3, 0.0, 4e-3))
            q = PhasePartition(qt, ql, qi)
            h = moist_enthalpy(T, q)
            for Tr in (c.T_triple, 150.0, T + 7.0)
                worst = max(worst, abs(temperature_from_moist_enthalpy(h, q; T_ref = Tr) - T) / T)
            end
            # the slope of h in T at fixed partition IS c_pm (Kirchhoff)
            @test moist_enthalpy(T + 1, q) - moist_enthalpy(T, q) ≈
                  heat_capacity_moist(q) rtol = 1e-9
        end
        @printf("    T(h) round trip: max |ΔT|/T = %.2e\n", worst)
        @test worst < 1e-14
    end

    @testset "Process adjustment conserves moist enthalpy (200-330 K, 300-1050 hPa)" begin
        worst_h = 0.0; worst_first = 0.0; worst_sat = 0.0; worst_w = 0.0
        n = 0; n_cond = 0; n_evap = 0; n_fail = 0
        for T0 in 200.0:2.5:330.0, p in (3.0e4, 5.0e4, 7.0e4, 8.5e4, 1.0e5, 1.05e5),
            S in (0.3, 0.9, 1.02, 1.3), qc in (0.0, 1e-5, 3e-3),
            (qr, qi) in ((0.0, 0.0), (5e-3, 0.0), (0.0, 5e-3), (2e-3, 1e-3))
            qs0 = saturation_specific_humidity(T0, p)
            qs0 > 0.2 && continue          # near boiling at low p: not air
            qv = S * qs0
            r = isobaric_saturation_adjustment(T0, p, qv, qc; q_r = qr, q_i = qi)
            qt = qv + qc + qr + qi
            q0 = PhasePartition(qt, qc + qr, qi)
            q1 = PhasePartition(qt, qc + r.Δq + qr, qi)
            h0 = moist_enthalpy(T0, q0); h1 = moist_enthalpy(r.T, q1)
            worst_h = max(worst_h, abs(h1 - h0) / abs(h0))
            # first-law form: c_pm(final) ΔT = L_v(T0) Δq, residual in J/kg
            # relative to h
            worst_first = max(worst_first, abs(heat_capacity_moist(q1) * (r.T - T0) -
                                               latent_heat_vapor(T0) * r.Δq) / abs(h0))
            # water: only vapour ↔ cloud liquid, and never more than exists
            @test -qc <= r.Δq <= qv
            qv1 = qv - r.Δq; qc1 = qc + r.Δq
            worst_w = max(worst_w, abs((qv1 + qc1) - (qv + qc)) / (qv + qc))
            # end state: saturated over liquid, or all cloud gone and sub-
            # saturated (never supersaturated with cloud left unadjusted)
            qs1 = saturation_specific_humidity(r.T, p)
            if qc1 > 0
                worst_sat = max(worst_sat, abs(qv1 - qs1) / qs1)
            elseif qv1 > qs1 * (1 + 1e-12)
                n_fail += 1
            end
            r.Δq > 0 && (n_cond += 1); r.Δq < 0 && (n_evap += 1)
            # heating has the sign of condensation
            @test sign(r.T - T0) == sign(r.Δq)
            n += 1
        end
        @printf("    isobaric adjustment, %d states (%d condense, %d evaporate): max |Δh|/h = %.2e, max first-law residual/h = %.2e, max |q_v/q_s-1| = %.2e\n",
                n, n_cond, n_evap, worst_h, worst_first, worst_sat)
        @test n > 4000 && n_cond > 1000 && n_evap > 1000
        @test worst_h < 1e-12
        @test worst_first < 1e-12
        @test worst_w < 1e-15
        @test worst_sat < 1e-8
        @test n_fail == 0
        # subsaturated, cloud-free air is returned bit-for-bit untouched
        r = isobaric_saturation_adjustment(290.0, 9e4, 5e-3, 0.0; q_r = 1e-3)
        @test r.Δq == 0 && r.T == 290.0
    end

    @testset "Process adjustment: rain and ice enter c_pm (issue #82)" begin
        # Leaving 5 g/kg of rain (ice) out of c_pm overstates ΔT per unit
        # condensate by (c_l - c_pd)·5e-3/c_pm ≈ 1.5 % ((c_i - c_pd)·5e-3/c_pm
        # ≈ 0.5 %) — the numbers issue #82 measured on P3.
        T0, p = 285.0, 9.0e4
        qv = 1.1 * saturation_specific_humidity(T0, p)
        per(qr, qi) = (r = isobaric_saturation_adjustment(T0, p, qv, 0.0; q_r = qr, q_i = qi);
                       (r.T - T0) / r.Δq)
        k0 = per(0.0, 0.0)
        over_r = k0 / per(5e-3, 0.0) - 1
        over_i = k0 / per(0.0, 5e-3) - 1
        @printf("    ΔT/Δq overstated without 5 g/kg rain: %.3f %%, ice: %.3f %%\n",
                100over_r, 100over_i)
        @test 0.014 < over_r < 0.017
        @test 0.004 < over_i < 0.006
        # ... and exactly by the heat-capacity ratio (to the small change in
        # the vapour share, since the final q_v differs slightly)
        cp(qr, qi) = heat_capacity_moist(PhasePartition(qv + qr + qi, qr, qi))
        @test 1 + over_r ≈ cp(5e-3, 0.0) / cp(0.0, 0.0) rtol = 1e-4
        @test 1 + over_i ≈ cp(0.0, 5e-3) / cp(0.0, 0.0) rtol = 1e-4
    end

    @testset "Diagnosis inverts θ_li exactly (200-330 K, 300-1050 hPa)" begin
        worst = 0.0; worst_rt = 0.0; n = 0
        for T0 in 200.0:5.0:330.0, p in (3.0e4, 5.0e4, 7.0e4, 8.5e4, 1.0e5, 1.05e5),
            S in (0.5, 1.02, 1.3)
            qs0 = saturation_specific_humidity(T0, p)
            qs0 > 0.2 && continue
            qt = S * qs0
            θli = liquid_ice_potential_temperature(T0, p, PhasePartition(qt))
            T, q = saturation_adjustment(θli, p, qt)
            worst = max(worst, abs(liquid_ice_potential_temperature(T, p, q) - θli) / θli)
            # a diagnosis is idempotent: re-diagnosing the diagnosed state
            # returns it (the round trip a model makes every step)
            T2, _ = saturation_adjustment(liquid_ice_potential_temperature(T, p, q), p, qt)
            worst_rt = max(worst_rt, abs(T2 - T))
            n += 1
        end
        @printf("    θ_li diagnosis, %d states: max |Δθ_li|/θ_li = %.2e, re-diagnosis |ΔT| = %.2e K\n",
                n, worst, worst_rt)
        # "Exactly" = to the bisection tolerance, tol = 1e-10 K in T, which
        # at saturation moves θ_li by up to ~3e-12 relative (measured 3.07e-12)
        @test worst < 1e-11
        @test worst_rt < 1e-9
    end

    @testset "Diagnosis ≠ process: θ_li is not isobaric enthalpy" begin
        # Same initial parcel, both adjustments. (b) is exact; (a) conserves
        # θ_li, whose Exner exponent R_m/c_pm moves with the partition, so
        # its c_pm ΔT/(L_v Δq) departs from 1 roughly ∝ ln(p_ref/p).
        # This is why (a) is kept for diagnosis only.
        for (p, lo, hi) in ((1.0e5, 0.995, 1.001), (9.0e4, 1.005, 1.02),
                            (6.0e4, 1.05, 1.08))
            T0 = 285.0
            qv = 1.1 * saturation_specific_humidity(T0, p)
            θli = liquid_ice_potential_temperature(T0, p, PhasePartition(qv))
            Ta, qa = saturation_adjustment(θli, p, qv)
            ra = heat_capacity_moist(qa) * (Ta - T0) / (latent_heat_vapor(T0) * qa.liq)
            rb = isobaric_saturation_adjustment(T0, p, qv, 0.0)
            qb = PhasePartition(qv, rb.Δq, 0.0)
            rbr = heat_capacity_moist(qb) * (rb.T - T0) / (latent_heat_vapor(T0) * rb.Δq)
            @printf("    p = %4.0f hPa: c_pm ΔT/(L_v(T0) Δq) θ_li-diagnosis %.5f, isobaric %.14f\n",
                    p / 100, ra, rbr)
            @test lo < ra < hi
            @test rbr ≈ 1 atol = 1e-12
        end
    end

    @testset "Naive fixed point diverges where bisection does not" begin
        # This records WHY saturation_adjustment is written as bisection.
        # The obvious iteration T <- T_dry + (L/c_p) q_liq(T) oscillates:
        # condensing enough to warm the parcel raises q_sat past q_tot,
        # which un-condenses everything, which cools it back. A capped
        # loop silently returns zero liquid and the cloud never forms.
        p, qt = 9.17e4, 9.0e-3
        thl = 289.0
        exner = (p / 1.0e5)^(287.05 / 1004.6)
        Tdry = thl * exner
        Tn = Tdry
        hist = Float64[]
        for _ in 1:40
            qs = saturation_specific_humidity(Tn, p)
            ql = max(0.0, qt - qs)
            Tn = Tdry + (2.5e6 / 1004.6) * ql
            push!(hist, Tn)
        end
        # it never settles: the last ten iterates still swing by > 1 K
        @test maximum(hist[end-9:end]) - minimum(hist[end-9:end]) > 1.0

        # the bisection solver handles the same state without trouble
        θli = liquid_ice_potential_temperature(Tdry, p, PhasePartition(qt))
        Tb, qb = saturation_adjustment(θli, p, qt)
        @test isfinite(Tb)
        @test liquid_ice_potential_temperature(Tb, p, qb) ≈ θli rtol=1e-9
        @info "naive fixed point vs bisection" swing_K=(maximum(hist[end-9:end])-minimum(hist[end-9:end])) bisection_T=Tb bisection_qliq=qb.liq
    end

    @testset "Conserved variables under adiabatic ascent" begin
        # Lift a saturated parcel and check the invariants hold. θ_li and
        # q_tot are conserved by construction; moist static energy is the
        # independent check, since it is built from L_v and c_pm rather
        # than from θ_li.
        p0, T0, qt = 1.0e5, 298.0, 1.6e-2
        θli = liquid_ice_potential_temperature(T0, p0, PhasePartition(qt))
        z = 0.0; p = p0; T = T0
        q = PhasePartition(qt)
        h0 = moist_static_energy(T, z, q)
        drift_h = 0.0; drift_θ = 0.0
        for _ in 1:60
            # hydrostatic step of 100 m
            ρ = air_density(T, p, q)
            dz = 100.0
            p -= ρ * c.grav * dz
            z += dz
            T, q = saturation_adjustment(θli, p, qt)
            drift_θ = max(drift_θ,
                abs(liquid_ice_potential_temperature(T, p, q) - θli))
            h = moist_static_energy(T, z, q)
            drift_h = max(drift_h, abs(h - h0))
        end
        @test drift_θ < 1e-8                     # exact by construction
        # MSE is only an APPROXIMATE invariant and it is worth being
        # precise about why. h = c_pm T + gz + L_v q_v drops the enthalpy
        # of the condensate and treats c_pm as if it did not change as
        # vapour becomes liquid. Over a 6 km saturated ascent those
        # neglected terms are of order q_tot (c_l - c_pd) ΔT plus
        # q_v (dL_v/dT) ΔT ≈ 3.5 kJ/kg, i.e. ~1 % of h. The measured
        # drift below is that, not a coding error — θ_li, which IS exact,
        # is conserved to 1e-8 in the same integration.
        @test drift_h / h0 < 4e-2
        @test q.liq > 0                          # still cloudy at the top
        @info "6 km saturated ascent" theta_li_drift=drift_θ MSE_rel_drift=drift_h/h0 p_top=p T_top=T q_liq_top=q.liq
    end

    @testset "Lapse rates" begin
        @test dry_adiabatic_lapse_rate() ≈ 9.80665 / 1004.6 rtol=1e-12
        Γw = moist_adiabatic_lapse_rate(300.0, 1.0e5)
        Γc = moist_adiabatic_lapse_rate(220.0, 3.0e4)
        @test 3.0e-3 < Γw < 5.0e-3               # warm tropics ≈ 4 K/km
        @test Γc > 9.0e-3                        # cold: approaches dry
        @test Γc < dry_adiabatic_lapse_rate()
        # monotone: colder ⇒ closer to dry
        Γs = [moist_adiabatic_lapse_rate(T, 8e4) for T in 230.0:5.0:310.0]
        @test all(diff(Γs) .< 0)
        @info "moist adiabatic lapse rate (K/km)" warm=1000Γw cold=1000Γc dry=1000*dry_adiabatic_lapse_rate()
    end

    @testset "Dewpoint, frost point, humidity" begin
        p, qv = 9.5e4, 1.0e-2
        Td = dewpoint_temperature(p, qv)
        # at the dewpoint the parcel is exactly saturated
        @test saturation_specific_humidity(Td, p) ≈ qv rtol=1e-6
        @test relative_humidity(Td, p, qv) ≈ 1.0 rtol=1e-6
        # warmer than the dewpoint ⇒ subsaturated
        @test relative_humidity(Td + 5, p, qv) < 1
        @test relative_humidity(Td - 5, p, qv) > 1
        # frost point is ABOVE the dewpoint below freezing
        p2, qv2 = 8e4, 1e-3
        @test frost_point_temperature(p2, qv2) > dewpoint_temperature(p2, qv2)
        # supersaturation sign convention
        @test supersaturation(Td, p, qv) ≈ 0 atol=1e-6
        @test supersaturation(Td - 3, p, qv) > 0
        @info "dewpoint" T_d=Td T_frost=frost_point_temperature(p2, qv2)

        # vapour pressure round trip
        ε_exact = 0.018015268 / 0.02896546     # not the 5 s.f. rounding
        @test vapor_pressure(qv, p) / p ≈ qv / (ε_exact + (1-ε_exact)*qv) rtol=1e-12
    end

    @testset "Equivalent and wet-bulb potential temperature" begin
        T, p, qv = 300.0, 1.0e5, 1.5e-2
        q = PhasePartition(qv)
        θe = equivalent_potential_temperature(T, p, q)
        @test θe > potential_temperature(T, p, q)    # latent heat adds
        @test 330.0 < θe < 360.0                     # tropical value
        # θ_e is conserved along a moist adiabat: lift and re-check
        θli = liquid_ice_potential_temperature(T, p, q)
        T2, q2 = saturation_adjustment(θli, 7.0e4, qv)
        θe2 = equivalent_potential_temperature(T2, 7.0e4, q2)
        @test θe2 ≈ θe rtol=2e-2
        θw = wet_bulb_potential_temperature(T, p, q)
        @test θw < θe
        @info "theta_e / theta_w" theta_e=θe theta_e_lifted=θe2 theta_w=θw
    end

    @testset "Lifting condensation level" begin
        T, p, qv = 300.0, 1.0e5, 1.2e-2
        T_lcl, p_lcl = lifting_condensation_level(T, p, qv)
        @test p_lcl < p
        @test T_lcl < T
        # at the LCL the parcel is exactly saturated
        @test saturation_specific_humidity(T_lcl, p_lcl) ≈ qv rtol=1e-5
        # drier air ⇒ higher LCL (lower pressure)
        _, p_dry = lifting_condensation_level(T, p, 6e-3)
        @test p_dry < p_lcl
        # already-saturated parcel has its LCL at its own level
        qs = saturation_specific_humidity(T, p)
        Ts_, ps_ = lifting_condensation_level(T, p, qs * 1.001)
        @test ps_ ≈ p rtol=1e-6
        @info "LCL" T_lcl=T_lcl p_lcl=p_lcl p_lcl_dry=p_dry
    end

    @testset "Buoyancy and CAPE" begin
        # A warm moist parcel in a cooler environment must be buoyant.
        qp = PhasePartition(1.6e-2)
        qe = PhasePartition(8e-3)
        b = buoyancy(302.0, 1e5, qp, 300.0, 1e5, qe)
        @test b > 0
        # ...and condensate loading must reduce it
        b_loaded = buoyancy(302.0, 1e5, PhasePartition(1.6e-2, 4e-3, 0.0),
                            300.0, 1e5, qe)
        @test b_loaded < b
        @info "buoyancy" dry_parcel=b with_4g_condensate=b_loaded

        # Conditionally unstable sounding ⇒ positive CAPE, negative CIN.
        sounding = [(1.0e5 - 5000.0*k,
                     300.0 - 6.5e-3 * (k * 5000.0 / (1.0e5) * 8000.0),
                     max(1e-5, 1.2e-2 * exp(-k/4)))
                    for k in 0:16]
        cape, cin = cape_cin(301.0, 1.0e5, 1.6e-2, sounding)
        @test isfinite(cape) && isfinite(cin)
        @test cape >= 0
        @test cin <= 0
        @info "CAPE / CIN" CAPE=cape CIN=cin
    end

    @testset "Type stability and Float32" begin
        c32 = ThermodynamicConstants{Float32}()
        @test saturation_vapor_pressure(273.16f0, c32) isa Float32
        @test latent_heat_vapor(273.16f0, c32) isa Float32
        q = PhasePartition(1f-2, 1f-3, 0f0)
        @test q isa PhasePartition{Float32}
        @test heat_capacity_moist(q, c32) isa Float32
    end
end
