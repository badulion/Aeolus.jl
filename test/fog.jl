# Era IV, Phase 44 (#11) — fog / dew condensation coupled to vapour.
#
# The scheme this replaces (examples/urban `apply_fog_dew!`) shipped
# disabled because its warming "amplified". The tests below pin the three
# properties whose absence caused that: energy conservation through the
# phase change, no condensation in subsaturated air, and no overshoot of
# saturation for any time step.

using Test
using Aeolus.MoistThermodynamics
using Aeolus.RadiationFog

@testset "Era IV — Radiation fog / dew coupling (Phase 44)" begin
    c = DEFAULT_CONSTANTS
    qsat(T, p) = saturation_specific_humidity(T, p)

    @testset "relaxation_fraction is the exact exponential" begin
        @test relaxation_fraction(2.0, 0.0) == 1.0
        @test relaxation_fraction(1e-6, 10.0) ≈ 1e-7 rtol = 1e-6
        f1 = relaxation_fraction(5.0, 7.0); f2 = relaxation_fraction(10.0, 7.0)
        @test 1 - (1 - f1)^2 ≈ f2 rtol = 1e-14       # two half steps == one step
    end

    @testset "Fog: saturation adjustment conserves water and enthalpy" begin
        worst_w = 0.0; worst_h = 0.0; worst_s = 0.0; worst_L = 0.0; worst_lin = 0.0
        for (T, p, S) in ((285.0, 1.0e5, 1.05), (265.0, 1.0e5, 1.10), (300.0, 1.0e5, 1.01))
            qv = S * qsat(T, p)
            r = fog_condensation(T, p, qv, 0.0, 2.0)
            worst_w = max(worst_w, abs(r.qv + r.ql - qv) / qv)
            worst_s = max(worst_s, abs(r.qv - qsat(r.T, p)))
            worst_h = max(worst_h, abs(moist_enthalpy(r.T, PhasePartition(qv, r.ql, 0.0)) /
                                       moist_enthalpy(T, PhasePartition(qv)) - 1))
            # all of L·Δq goes into c_pm ΔT
            worst_L = max(worst_L, abs(heat_capacity_moist(PhasePartition(qv, r.ql, 0.0)) *
                                       (r.T - T) / (latent_heat_vapor(T) * r.Δq) - 1))
            # first-order analytic limit Δq ≈ (q_v - q_s)/(1 + L/c_p dq_s/dT)
            Lv = latent_heat_vapor(T); qs = qsat(T, p)
            γ = Lv / heat_capacity_moist(PhasePartition(qv)) * qs * Lv / (461.52 * T^2)
            worst_lin = max(worst_lin, abs(r.Δq / ((qv - qs) / (1 + γ)) - 1))
        end
        @test worst_w < 1e-15 && worst_s < 1e-12 && worst_h < 1e-14
        @test worst_L < 1e-10
        @test worst_lin < 0.05
        # no condensation in subsaturated air, any dt
        @test all(let r = fog_condensation(T, 1e5, RH * qsat(T, 1e5), 0.0, 1e4)
                      r.Δq == 0 && r.T == T && r.ql == 0
                  end for T in (270.0, 300.0), RH in (0.3, 0.999999))
        # existing fog in subsaturated air evaporates to saturation or completely
        r = fog_condensation(285.0, 1e5, 0.98qsat(285.0, 1e5), 2e-3, 2.0)
        @test r.Δq < 0 && r.ql > 0 && abs(r.qv - qsat(r.T, 1e5)) < 1e-12
        r = fog_condensation(285.0, 1e5, 0.5qsat(285.0, 1e5), 1e-4, 2.0)
        @test r.ql == 0 && r.Δq ≈ -1e-4
    end

    @testset "Fog: never overshoots saturation, any dt" begin
        T0, p = 285.0, 1.0e5
        bad = 0
        for S in (1.05, 0.97), τ in (0.0, 10.0), dt in (0.1, 30.0, 1e8)
            T, qv, ql = T0, S * qsat(T0, p), S > 1 ? 0.0 : 1e-3
            s0 = sign(qv - qsat(T, p)); prev = abs(qv - qsat(T, p))
            for _ in 1:10
                r = fog_condensation(T, p, qv, ql, dt; τ = τ)
                T, qv, ql = r.T, r.qv, r.ql
                ex = qv - qsat(T, p)
                # never crosses, monotone to round-off, no negative fog
                ((sign(ex) == s0 || abs(ex) < 1e-12) && abs(ex) <= prev + 1e-13 && ql >= 0) || (bad += 1)
                prev = abs(ex)
            end
        end
        @test bad == 0
        # finite τ composes: two steps of 10 s == one of 20 s
        qv = 1.05qsat(T0, p)
        a = fog_condensation(T0, p, qv, 0.0, 10.0; τ = 7.0)
        b = fog_condensation(a.T, p, a.qv, a.ql, 10.0; τ = 7.0)
        @test b.ql ≈ fog_condensation(T0, p, qv, 0.0, 20.0; τ = 7.0).ql rtol = 1e-9
    end

    @testset "Fog: delegates to the contract's process adjustment (#82, #85)" begin
        # #85 regression: Newton hitting g = 0 exactly kept the bracket
        # midpoint (1.143e-6 instead of 2.286e-6 kg/kg)
        Δ, Te = fog_equilibrium(204.2477594857438, 72681.13710255225, 7.421642749398895e-6, 0.0)
        @test Δ ≈ 2.28605e-6 rtol = 1e-4
        @test abs(7.421642749398895e-6 - Δ - qsat(Te, 72681.13710255225)) < 1e-14
        # One source of truth: fog_equilibrium IS isobaric_saturation_adjustment
        # (the same numbers, copied through, not merely close ones)
        @test all(begin
                      qv = S * qsat(T, p)
                      Δ, Te = fog_equilibrium(T, p, qv, ql)
                      r = isobaric_saturation_adjustment(T, p, qv, ql)
                      Δ === r.Δq && Te === r.T
                  end for T in (230.0, 280.0, 320.0), p in (5.0e4, 1.0e5),
                          S in (0.5, 1.01, 1.2), ql in (0.0, 2e-3))
        # the θ_li DIAGNOSIS is a different operation and legitimately
        # differs: at 800 hPa 1.75 % less condenses (recorded, not a defect)
        T0, p = 285.0, 8.0e4
        qv0 = 1.05qsat(T0, p)
        r = fog_condensation(T0, p, qv0, 0.0, 2.0)
        _, qc = saturation_adjustment(liquid_ice_potential_temperature(T0, p, PhasePartition(qv0)), p, qv0)
        @test -0.025 < qc.liq / r.ql - 1 < -0.01
    end

    @testset "Fog: gridded fog_adjust! on potential temperature" begin
        Nx, Ny, Nz = 4, 2, 3
        pcol = [1.0e5 * exp(-(k - 0.5) * 6.25 / 8000) for k in 1:Nz]
        θ  = fill(285.0, Nx, Ny, Nz)
        qv = [(0.9 + 0.05 * i) * qsat(285.0 * exner_function(pcol[k]), pcol[k])
              for i in 1:Nx, j in 1:Ny, k in 1:Nz]
        ql = zeros(Nx, Ny, Nz)
        qt0 = copy(qv)
        h(θ, ql) = [moist_enthalpy(θ[i,j,k] * exner_function(pcol[k]),
                                   PhasePartition(qt0[i,j,k], ql[i,j,k], 0.0))
                    for i in 1:Nx, j in 1:Ny, k in 1:Nz]
        h0 = h(θ, ql)
        total = fog_adjust!(θ, qv, ql, pcol, 2.0)
        @test total ≈ sum(ql) rtol = 1e-12
        @test qv .+ ql ≈ qt0 rtol = 1e-15
        @test all(ql[1:2, :, :] .== 0) && all(ql[3:4, :, :] .> 0)   # RH < 1 / > 1 columns
        @test h(θ, ql) ≈ h0 rtol = 1e-14
    end

    @testset "Dew: surface deposition, coupled and bounded" begin
        p, dz, ρ = 1.0e5, 6.25, 1.0
        r = dew_deposition(0.008, 290.0, p, 2.0; dz = dz, ρ = ρ, dew = 0.0)
        @test r.mass == 0 && r.surface_heat == 0       # surface above the dew point
        qv = 0.010; Ts = 280.0
        r = dew_deposition(qv, Ts, p, 2.0; dz = dz, ρ = ρ)
        @test r.mass > 0
        @test r.mass ≈ -ρ * dz * r.Δqv rtol = 1e-15
        @test r.surface_heat ≈ latent_heat_vapor(Ts) * r.mass rtol = 1e-15
        # stable for any dt: dries the air to q_s(T_s) and no further
        @test qv + dew_deposition(qv, Ts, p, 1e12; dz = dz, ρ = ρ).Δqv ≈ qsat(Ts, p) rtol = 1e-12
        # evaporation limited by the dew present
        @test dew_deposition(0.005, 290.0, p, 1e6; dz = dz, ρ = ρ, dew = 0.01).mass ≈ -0.01 rtol = 1e-12
        # physical rate: Monteith (1957) ceiling ~0.07-0.08 mm/h
        @test 0.005 < dew_deposition(qv, Ts, p, 1.0; dz = dz, ρ = 1.2).mass * 3600 < 0.08
    end

    @testset "Dew: deposit-then-evaporate cycle closes water and energy" begin
        # Night at 280 K collects dew from a closed layer, day at 295 K
        # re-evaporates it; net surface heating is exactly
        # (L_v(280) - L_v(295))·m. dt = 200 s: the relaxation is an exact
        # exponential, so the step size does not enter the budgets.
        p, dz, ρ, dt = 1.0e5, 6.25, 1.0, 200.0
        q3 = fill(0.010, 1, 1, 1); q2 = fill(0.010, 1, 1)
        d3 = zeros(1, 1); d2 = zeros(1, 1)
        W0 = ρ * dz * 0.010
        E = 0.0; m_dep = 0.0; agree = true; max_werr = 0.0
        for (Tsv, nsteps) in ((280.0, 36), (295.0, 200))
            Ts = fill(Tsv, 1, 1)
            for _ in 1:nsteps
                H3 = dew_exchange!(q3, d3, Ts, p, dt; dz = dz, ρ = ρ)
                agree &= dew_exchange!(q2, d2, Ts, p, dt; dz = dz, ρ = ρ) == H3
                E += H3[1] * dt
                max_werr = max(max_werr, abs(ρ * dz * q3[1] + d3[1] - W0) / W0)
            end
            Tsv == 280.0 && (m_dep = d3[1])
        end
        @test agree                                 # 2-D and 3-D paths agree
        @test max_werr < 1e-12
        @test m_dep > 0.01 && d3[1] == 0            # dew formed, all evaporated
        @test E ≈ (latent_heat_vapor(280.0) - latent_heat_vapor(295.0)) * m_dep rtol = 1e-9
    end

    @testset "Previously amplifying configuration now stays bounded" begin
        # Cold surface (280 K) under subsaturated air (288 K, q_v = 10 g/kg)
        # held at q_v by resupply, dt = 2 s. The old scheme heated the air
        # by 75 K in 24 h (~3 K per hour); 1 h is enough to see it.
        p, dz, ρ, dt = 1.0e5, 6.25, 1.0, 2.0
        θ  = fill(288.0, 1, 1, 1); qv = fill(0.010, 1, 1, 1)
        ql = zeros(1, 1, 1); qd = zeros(1, 1); Ts = fill(280.0, 1, 1)
        heat = 0.0; supplied = 0.0
        for _ in 1:1800
            H = dew_exchange!(qv, qd, Ts, p, dt; dz = dz, ρ = ρ)
            fog_adjust!(θ, qv, ql, p, dt)
            heat += H[1] * dt
            supplied += ρ * dz * (0.010 - qv[1])
            qv[1] = 0.010
        end
        @test θ[1] == 288.0                   # the air is never heated (untouched)
        @test ql[1] == 0                      # and never fogged: RH < 1
        @test qd[1] ≈ supplied rtol = 1e-10
        @test heat ≈ latent_heat_vapor(280.0) * qd[1] rtol = 1e-10
    end
end
