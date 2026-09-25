# Era IV, Phase 44 (#11) — fog / dew condensation coupled to vapour.
#
# The scheme this replaces (examples/urban `apply_fog_dew!`) shipped
# disabled because its warming "amplified". The tests below pin the three
# properties whose absence caused that: energy conservation through the
# phase change, no condensation in subsaturated air, and no overshoot of
# saturation for any time step.

using Test
using Printf
using Aeolus.MoistThermodynamics
using Aeolus.RadiationFog

@testset "Era IV — Radiation fog / dew coupling (Phase 44)" begin
    c = DEFAULT_CONSTANTS
    qsat(T, p) = saturation_specific_humidity(T, p)

    @testset "Contract: moist_enthalpy is the exact isobaric invariant" begin
        T0, qt = 285.0, 0.012
        for (ql0, Δ) in ((0.0, 1e-3), (5e-4, 2e-3), (1e-3, -8e-4))
            q0 = PhasePartition(qt, ql0, 0.0)
            q1 = PhasePartition(qt, ql0 + Δ, 0.0)
            # dh/dT at fixed partition is c_pm (Kirchhoff consistency)
            dh = moist_enthalpy(T0 + 1, q1) - moist_enthalpy(T0, q1)
            @test dh ≈ heat_capacity_moist(q1) rtol = 1e-9
            # c_pm(q1) ΔT = L_v(T0) Δq exactly at fixed h
            ΔT = latent_heat_vapor(T0) * Δ / heat_capacity_moist(q1)
            @test moist_enthalpy(T0 + ΔT, q1) ≈ moist_enthalpy(T0, q0) rtol = 1e-14
        end
        # Ice: dh/dT = c_pm with ice present as well
        qi = PhasePartition(0.01, 0.0, 2e-3)
        @test moist_enthalpy(251.0, qi) - moist_enthalpy(250.0, qi) ≈
              heat_capacity_moist(qi) rtol = 1e-9
        # moist_static_energy is NOT invariant under the same change — it
        # differs by (c_pv - c_l) T q_v; measure the size of its error.
        q0 = PhasePartition(qt, 0.0, 0.0); q1 = PhasePartition(qt, 1e-3, 0.0)
        ΔT = latent_heat_vapor(T0) * 1e-3 / heat_capacity_moist(q1)
        err = moist_static_energy(T0 + ΔT, 0.0, q1) - moist_static_energy(T0, 0.0, q0)
        @test abs(err) / (latent_heat_vapor(T0) * 1e-3) > 0.2
        @info "MSE non-conservation through 1 g/kg condensation" err_J_per_kg=err frac_of_LΔq=err / (latent_heat_vapor(T0) * 1e-3)
    end

    @testset "relaxation_fraction" begin
        @test relaxation_fraction(2.0, 0.0) == 1.0
        @test relaxation_fraction(1e-6, 10.0) ≈ 1e-7 rtol = 1e-6
        for dt in (0.1, 1.0, 1e2, 1e6)
            f = relaxation_fraction(dt, 10.0)
            @test 0 < f <= 1
        end
        # composition: two half steps == one full step (exact exponential)
        f1 = relaxation_fraction(5.0, 7.0); f2 = relaxation_fraction(10.0, 7.0)
        @test 1 - (1 - f1)^2 ≈ f2 rtol = 1e-14
    end

    @testset "Fog: saturation adjustment conserves water and enthalpy" begin
        for (T, p, S) in ((285.0, 1.0e5, 1.05), (275.0, 9.5e4, 1.02),
                          (265.0, 1.0e5, 1.10), (300.0, 1.0e5, 1.01))
            qv = S * qsat(T, p)
            r = fog_condensation(T, p, qv, 0.0, 2.0)
            @test r.Δq > 0
            @test r.qv + r.ql ≈ qv rtol = 1e-15                   # water
            @test abs(r.qv - qsat(r.T, p)) < 1e-12                # saturated
            h0 = moist_enthalpy(T, PhasePartition(qv, 0.0, 0.0))
            h1 = moist_enthalpy(r.T, PhasePartition(qv, r.ql, 0.0))
            @test h1 ≈ h0 rtol = 1e-14                            # energy
            # heat released == L·Δq, and it all goes into c_pm ΔT
            cpm1 = heat_capacity_moist(PhasePartition(qv, r.ql, 0.0))
            @test r.latent_heat ≈ latent_heat_vapor(T) * r.Δq rtol = 1e-14
            @test cpm1 * (r.T - T) ≈ r.latent_heat rtol = 1e-10
            # Analytic first-order limit: Δq ≈ (q_v - q_s)/(1 + L/c_p dq_s/dT)
            Lv = latent_heat_vapor(T); qs = qsat(T, p)
            γ = Lv / heat_capacity_moist(PhasePartition(qv)) *
                qs * Lv / (461.52 * T^2)
            @test r.Δq ≈ (qv - qs) / (1 + γ) rtol = 0.05
        end
        T, p = 285.0, 1.0e5
        r = fog_condensation(T, p, 1.05qsat(T, p), 0.0, 2.0)
        @info "fog adjustment, 285 K, S=1.05" Δq=r.Δq ΔT=r.T - T
    end

    @testset "Fog: no condensation in subsaturated air" begin
        for T in (270.0, 285.0, 300.0), RH in (0.3, 0.9, 0.999999)
            p = 1.0e5
            qv = RH * qsat(T, p)
            r = fog_condensation(T, p, qv, 0.0, 1e4)
            @test r.Δq == 0 && r.T == T && r.qv == qv && r.ql == 0
        end
        # Existing fog in subsaturated air evaporates, cooling to
        # saturation (partial) or evaporating completely.
        T, p = 285.0, 1.0e5
        r = fog_condensation(T, p, 0.98qsat(T, p), 2e-3, 2.0)
        @test r.Δq < 0 && r.ql > 0 && r.T < T
        @test abs(r.qv - qsat(r.T, p)) < 1e-12
        r = fog_condensation(T, p, 0.5qsat(T, p), 1e-4, 2.0)
        @test r.ql == 0 && r.Δq ≈ -1e-4 && r.qv < qsat(r.T, p)
    end

    @testset "Fog: never overshoots saturation, any dt" begin
        T0, p = 285.0, 1.0e5
        for S in (1.05, 0.97), τ in (0.0, 10.0),
            dt in (0.1, 2.0, 30.0, 1e3, 1e8)
            T, qv, ql = T0, S * qsat(T0, p), S > 1 ? 0.0 : 1e-3
            s0 = sign(qv - qsat(T, p))
            prev = abs(qv - qsat(T, p))
            for _ in 1:50
                r = fog_condensation(T, p, qv, ql, dt; τ = τ)
                T, qv, ql = r.T, r.qv, r.ql
                ex = qv - qsat(T, p)
                @test sign(ex) == s0 || abs(ex) < 1e-12   # never crosses
                @test abs(ex) <= prev + 1e-13              # monotone (to round-off)
                prev = abs(ex)
                @test ql >= 0
            end
        end
        # Composition with a finite τ: two half steps == one step
        qv = 1.05qsat(T0, p)
        a = fog_condensation(T0, p, qv, 0.0, 10.0; τ = 7.0)
        b = fog_condensation(a.T, p, a.qv, a.ql, 10.0; τ = 7.0)
        c2 = fog_condensation(T0, p, qv, 0.0, 20.0; τ = 7.0)
        @test b.ql ≈ c2.ql rtol = 1e-9
        @test b.T ≈ c2.T rtol = 1e-12
    end

    @testset "Fog: solver lands on saturation across 200-330 K (regression)" begin
        # Found in review by random sampling: when Newton hit g = 0 exactly,
        # the safeguard replaced the converged root by the bracket
        # midpoint, returning half the equilibrium condensate (this input:
        # 1.143e-6 instead of 2.286e-6 kg/kg) or, when fog evaporated,
        # overshooting into supersaturation (82 of 200,000 samples, mostly
        # below 230 K where q_s ~ 1e-6).
        Δ, Te = fog_equilibrium(204.2477594857438, 72681.13710255225,
                                7.421642749398895e-6, 0.0)
        @test Δ ≈ 2.28605e-6 rtol = 1e-4
        @test abs(7.421642749398895e-6 - Δ - qsat(Te, 72681.13710255225)) < 1e-14
        # Deterministic sweep: full adjustment always ends saturated (or
        # with all fog evaporated), conserving enthalpy.
        worst = 0.0; worst_h = 0.0; n_fail = 0
        for T in 200.0:2.5:330.0, p in (3.0e4, 6.0e4, 1.0e5),
            S in (0.3, 0.9, 1.02, 1.3), ql in (0.0, 1e-5, 3e-3)
            qv = S * qsat(T, p)
            r = fog_condensation(T, p, qv, ql, 1.0)
            ex = r.qv - qsat(r.T, p)
            if r.ql > 0
                worst = max(worst, abs(ex) / qsat(r.T, p))
            elseif ex > 1e-14 * qsat(r.T, p) + 1e-18
                n_fail += 1                        # no fog left but supersaturated
            end
            h0 = moist_enthalpy(T, PhasePartition(qv + ql, ql, 0.0))
            h1 = moist_enthalpy(r.T, PhasePartition(qv + ql, r.ql, 0.0))
            worst_h = max(worst_h, abs(h1 - h0) / h0)
        end
        @test worst < 1e-8          # relative distance from saturation
        @test n_fail == 0
        @test worst_h < 1e-14
    end

    @testset "Fog: delegates to the contract's process adjustment (#82)" begin
        # One source of truth: fog_equilibrium IS the contract's
        # isobaric_saturation_adjustment with no rain or ice — the same
        # numbers bit for bit, not merely close ones.
        n_same = 0; n = 0
        for T in 230.0:10.0:320.0, p in (5.0e4, 8.0e4, 1.0e5),
            S in (0.5, 0.99, 1.01, 1.2), ql in (0.0, 1e-6, 2e-3)
            qv = S * qsat(T, p)
            Δ, Te = fog_equilibrium(T, p, qv, ql)
            r = isobaric_saturation_adjustment(T, p, qv, ql)
            n_same += (Δ === r.Δq && Te === r.T); n += 1
        end
        @test n_same == n

        # Formerly two @test_broken (#85): they compared fog with the
        # contract's saturation_adjustment, which is the θ_li DIAGNOSIS
        # (it must reproduce θ_li, not moist enthalpy), and so encoded the
        # wrong operation. The property they were waiting for — the
        # contract providing an enthalpy-conserving adjustment — now holds:
        # at 800 hPa, 285 K, S = 1.05 the contract's PROCESS adjustment
        # agrees with fog exactly and conserves moist enthalpy.
        T0, p = 285.0, 8.0e4
        qv0 = 1.05qsat(T0, p)
        r = fog_condensation(T0, p, qv0, 0.0, 2.0)
        rc = isobaric_saturation_adjustment(T0, p, qv0, 0.0)
        @test rc.Δq == r.ql
        @test abs(moist_enthalpy(rc.T, PhasePartition(qv0, rc.Δq, 0.0)) -
                  moist_enthalpy(T0, PhasePartition(qv0))) <
              1e-12 * latent_heat_vapor(T0) * rc.Δq
        @test moist_enthalpy(r.T, PhasePartition(qv0, r.ql, 0.0)) ≈
              moist_enthalpy(T0, PhasePartition(qv0)) rtol = 1e-14

        # The θ_li diagnosis of the same parcel is a different operation and
        # legitimately differs. Recorded, not a defect: at 800 hPa it
        # over-heats by +2.76 % of L_v Δq, so 1.75 % less condenses; near
        # p_ref the two agree to the exponential-θ_li approximation
        # (≤ 3.2e-3).
        θli = liquid_ice_potential_temperature(T0, p, PhasePartition(qv0))
        Tc, qc = saturation_adjustment(θli, p, qv0)
        dq = qc.liq / r.ql - 1
        dh = (moist_enthalpy(Tc, qc) - moist_enthalpy(T0, PhasePartition(qv0))) /
             (latent_heat_vapor(T0) * qc.liq)
        @printf("    800 hPa: θ_li diagnosis vs isobaric process: Δq_c %+.3f %%, Δh/(L_v Δq) %+.3f %%\n",
                100dq, 100dh)
        @test -0.025 < dq < -0.01
        @test 0.02 < dh < 0.035
        for T0 in (275.0, 285.0, 300.0, 310.0), S in (1.01, 1.05, 1.10)
            p = c.p_ref
            qv0 = S * qsat(T0, p)
            r = fog_condensation(T0, p, qv0, 0.0, 2.0)
            θli = liquid_ice_potential_temperature(T0, p, PhasePartition(qv0))
            Tc, qc = saturation_adjustment(θli, p, qv0)
            @test qc.ice == 0
            @test abs(qc.liq - r.ql) / r.ql < 5e-3
            @test abs(Tc - r.T) < 2e-3
        end
    end

    @testset "Fog: gridded fog_adjust! on potential temperature" begin
        Nx, Ny, Nz = 4, 3, 5
        pcol = [1.0e5 * exp(-(k - 0.5) * 6.25 / 8000) for k in 1:Nz]
        θ  = fill(285.0, Nx, Ny, Nz)
        qv = zeros(Nx, Ny, Nz); ql = zeros(Nx, Ny, Nz)
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            Π = exner_function(pcol[k])
            qv[i, j, k] = (0.9 + 0.05 * i) * qsat(θ[i, j, k] * Π, pcol[k])
        end
        qt0 = qv .+ ql
        h0 = [moist_enthalpy(θ[i,j,k] * exner_function(pcol[k]),
                             PhasePartition(qt0[i,j,k], 0.0, 0.0))
              for i in 1:Nx, j in 1:Ny, k in 1:Nz]
        total = fog_adjust!(θ, qv, ql, pcol, 2.0)
        @test total ≈ sum(ql) rtol = 1e-12
        @test qv .+ ql ≈ qt0 rtol = 1e-15
        @test all(ql[1:2, :, :] .== 0)            # RH < 1 columns untouched
        @test all(ql[3:4, :, :] .> 0)             # RH > 1 columns fogged
        h1 = [moist_enthalpy(θ[i,j,k] * exner_function(pcol[k]),
                             PhasePartition(qt0[i,j,k], ql[i,j,k], 0.0))
              for i in 1:Nx, j in 1:Ny, k in 1:Nz]
        @test h1 ≈ h0 rtol = 1e-14
        # levels keyword restricts the adjustment
        θ2 = fill(285.0, 2, 2, 3); q2 = fill(1.05qsat(285.0, 1e5), 2, 2, 3)
        l2 = zeros(2, 2, 3)
        fog_adjust!(θ2, q2, l2, 1.0e5, 2.0; levels = 1:1)
        @test all(l2[:, :, 1] .> 0) && all(l2[:, :, 2:3] .== 0)
    end

    @testset "Dew: surface deposition, coupled and bounded" begin
        p, dz, ρ = 1.0e5, 6.25, 1.0
        # No dew when the surface is above the dew point
        r = dew_deposition(0.008, 290.0, p, 2.0; dz = dz, ρ = ρ, dew = 0.0)
        @test r.mass == 0 && r.Δqv == 0 && r.surface_heat == 0
        # Dew forms on a surface below the dew point; water and heat balance
        qv = 0.010; Ts = 280.0
        @test Ts < dewpoint_temperature(p, qv)
        r = dew_deposition(qv, Ts, p, 2.0; dz = dz, ρ = ρ)
        @test r.mass > 0
        @test r.mass ≈ -ρ * dz * r.Δqv rtol = 1e-15
        @test r.surface_heat ≈ latent_heat_vapor(Ts) * r.mass rtol = 1e-15
        # Exact exponential in time: one step of 2dt == two steps of dt
        a = dew_deposition(qv, Ts, p, 50.0; dz = dz, ρ = ρ)
        b = dew_deposition(qv + a.Δqv, Ts, p, 50.0; dz = dz, ρ = ρ)
        c2 = dew_deposition(qv, Ts, p, 100.0; dz = dz, ρ = ρ)
        @test a.Δqv + b.Δqv ≈ c2.Δqv rtol = 1e-12
        # Stable for any dt: can at most dry the air to q_s(T_s)
        for dt in (2.0, 1e3, 1e6, 1e12)
            r = dew_deposition(qv, Ts, p, dt; dz = dz, ρ = ρ)
            @test qv + r.Δqv >= qsat(Ts, p) - 1e-15
        end
        r = dew_deposition(qv, Ts, p, 1e12; dz = dz, ρ = ρ)
        @test qv + r.Δqv ≈ qsat(Ts, p) rtol = 1e-12
        # Dew evaporation is limited by the dew actually present
        r = dew_deposition(0.005, 290.0, p, 1e6; dz = dz, ρ = ρ, dew = 0.01)
        @test r.mass ≈ -0.01 rtol = 1e-12
        @test r.surface_heat < 0
        # Dew rate is physical: Monteith (1957) puts the ceiling on dewfall
        # at ~0.07-0.08 mm/h. With the urban example's bulk coefficients
        # (C_e = 1e-3, U = 3 m/s) and a surface 7 K below T_d:
        rate = dew_deposition(qv, Ts, p, 1.0; dz = dz, ρ = 1.2).mass * 3600
        @test 0.005 < rate < 0.08                     # kg m⁻² h⁻¹ = mm/h
        @info "dew rate, T_s = 280 K, q_v = 10 g/kg" mm_per_h=rate Td=dewpoint_temperature(p, qv)
    end

    @testset "Dew: deposit-then-evaporate cycle closes water and energy" begin
        # Night: surface at 280 K collects dew from a closed layer. Day:
        # surface at 295 K re-evaporates it. Column water ρ dz q_v + dew is
        # conserved at every step, all dew returns to the air, and the
        # surface's net latent heating is exactly (L_v(280) - L_v(295))·m
        # — deposition at the cold, evaporation at the warm temperature.
        p, dz, ρ, dt = 1.0e5, 6.25, 1.0, 2.0
        q3 = fill(0.010, 1, 1, 1); q2 = fill(0.010, 1, 1)
        d3 = zeros(1, 1); d2 = zeros(1, 1)
        W0 = ρ * dz * 0.010
        E = 0.0; m_dep = 0.0
        paths_agree = true; max_werr = 0.0; min_dew = Inf
        for (Tsv, nsteps) in ((280.0, 3600), (295.0, 20000))
            Ts = fill(Tsv, 1, 1)
            for _ in 1:nsteps
                H3 = dew_exchange!(q3, d3, Ts, p, dt; dz = dz, ρ = ρ)
                H2 = dew_exchange!(q2, d2, Ts, p, dt; dz = dz, ρ = ρ)
                paths_agree &= (H2 == H3)
                E += H3[1] * dt
                max_werr = max(max_werr, abs(ρ * dz * q3[1] + d3[1] - W0) / W0)
                min_dew = min(min_dew, d3[1])
            end
            Tsv == 280.0 && (m_dep = d3[1])
        end
        @test paths_agree                           # 2-D and 3-D paths agree
        @test max_werr < 1e-12                      # column water, every step
        @test min_dew >= 0
        @test m_dep > 0.01                          # dew did form (kg m⁻²)
        @test d3[1] == 0                            # and all of it evaporated
        @test q3[1] ≈ 0.010 rtol = 1e-12
        @test q2 == q3[:, :, 1] && d2 == d3
        @test E ≈ (latent_heat_vapor(280.0) - latent_heat_vapor(295.0)) * m_dep rtol = 1e-9
    end

    @testset "Previously amplifying configuration now stays bounded" begin
        # The configuration that made the urban scheme unusable: a cold
        # surface (280 K) under subsaturated air (288 K, q_v = 10 g/kg,
        # RH ≈ 0.94), vapour resupplied by evaporation / advection so that
        # the lowest layer is held at q_v = 10 g/kg, dt = 2 s for 24 h.
        # The old scheme heated the air by 75 K (288 -> 363 K) in that time.
        p, dz, ρ, dt = 1.0e5, 6.25, 1.0, 2.0
        θ  = fill(288.0, 1, 1, 1); qv = fill(0.010, 1, 1, 1)
        ql = zeros(1, 1, 1); qd = zeros(1, 1); Ts = fill(280.0, 1, 1)
        heat = 0.0; supplied = 0.0
        for n in 1:43200
            H = dew_exchange!(qv, qd, Ts, p, dt; dz = dz, ρ = ρ)
            fog_adjust!(θ, qv, ql, p, dt)
            heat += H[1] * dt
            supplied += ρ * dz * (0.010 - qv[1])
            qv[1] = 0.010
        end
        @test θ[1] == 288.0                   # the air is never heated
        @test ql[1] == 0                      # and never fogged: RH < 1
        @test qd[1] ≈ supplied rtol = 1e-10   # every gram accounted for
        @test heat ≈ latent_heat_vapor(280.0) * qd[1] rtol = 1e-10
        @info "24 h cold-surface box (old: θ 288 -> 363 K)" θ_final=θ[1] dew_mm=qd[1] surface_heat_Wm2=heat / 86400

        # Without resupply, the air is dried towards q_s(T_s) and no further.
        qv = fill(0.010, 1, 1, 1); qd = zeros(1, 1)
        for n in 1:43200
            dew_exchange!(qv, qd, Ts, p, dt; dz = dz, ρ = ρ)
        end
        @test qsat(280.0, p) <= qv[1] < 0.010
        @test ρ * dz * (0.010 - qv[1]) ≈ qd[1] rtol = 1e-10

        # Supersaturated, large explicit step: the old scheme overshot to
        # q_v - q_s = -2.9e-3 at rate·dt = 3. Now bounded and monotone.
        T0 = 285.0; qv0 = 1.05qsat(T0, p)
        r = fog_condensation(T0, p, qv0, 0.0, 30.0)
        @test abs(r.qv - qsat(r.T, p)) < 1e-12
        @info "S=1.05, dt=30 s (old overshoot -2.9e-3 kg/kg)" excess_after=r.qv - qsat(r.T, p)
    end
end
