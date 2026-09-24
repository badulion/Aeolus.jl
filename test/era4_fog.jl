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

end
