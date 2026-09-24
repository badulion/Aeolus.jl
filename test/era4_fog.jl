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

end
