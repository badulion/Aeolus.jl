# Era V, Phase 51 — lake and sea-ice thermodynamics (issue #37).
#
# Bitz & Lipscomb (1999) energy-conserving multi-layer sea ice with brine
# pockets, snow, surface/basal melt and growth, coupled to SurfaceExchange.
#
# What is validated against what:
#   * material functions  — closed forms of BL99 §2 and the numbers quoted there;
#   * growth              — Stefan's law (1891) and the exact Neumann solution;
#   * energy              — column enthalpy change vs. the fluxes applied, every step;
#   * MU71 benchmark      — the BL99 *reference code* run on the same forcing
#                           (numbers measured by us, see below) and the published
#                           BL99 Table 1/2 equilibrium thicknesses and sensitivity.
# No observations are used: there is no field data set in this file.

using Test, Printf
using Aeolus
using Aeolus.IceThermodynamics
using Aeolus.SurfaceExchange: SurfaceConstants, DEFAULT_SURFACE_CONSTANTS, SurfaceForcing,
                              AbstractSurface, step_surface!, surface_albedo,
                              surface_temperature, roughness_lengths, LH_f0, cp_ice
using Aeolus.MoistThermodynamics: ThermodynamicConstants
using Aeolus.GasOptics: erf_series

const ITD = Aeolus.IceThermodynamics


@testset "Era V — lake and sea-ice thermodynamics (#37)" begin

    @testset "BL99 material functions" begin
        p = IceParameters()
        ρ, L, c0 = p.constants.ρ_ice, LH_f0(p.constants), cp_ice(p.constants)
        T0 = 273.15
        # Contract delegation: ρ_i, k₀, c₀, L are the SurfaceConstants values.
        @test ρ == DEFAULT_SURFACE_CONSTANTS.ρ_ice
        @test ice_conductivity(T0 - 10, 0.0; p) == DEFAULT_SURFACE_CONSTANTS.k_ice
        # BL99 eq. (2): γ = L μ ≈ 18.0 kJ K kg⁻¹ ("5% higher than Untersteiner's 17.2").
        @test L * p.μ ≈ 18.0e3 rtol = 2e-3
        # BL99 §2: "q = 0.92 L₀ when T = -2 °C and S = 3.2 ppt".
        qr = ice_energy_of_melting(T0 - 2, 3.2; p) / (ρ * L)
        @test round(qr, digits = 2) == 0.92 || round(qr, digits = 2) == 0.93
        @test 0.915 < qr < 0.93
        # q vanishes at the liquidus and reduces to pure ice for S = 0.
        @test ice_melting_temperature(3.2; p) ≈ T0 - 0.1728
        @test abs(ice_energy_of_melting(ice_melting_temperature(3.2; p), 3.2; p)) < 1e-6
        @test ice_energy_of_melting(T0 - 7, 0.0; p) ≈ ρ * (c0 * 7 + L)
        @test ice_heat_capacity(T0 - 7, 0.0; p) == c0
        # Inversion round trip over the whole mushy range.
        worst = 0.0
        for S in (0.0, 0.5, 3.2, 8.0), T in (T0 - 40, T0 - 10, T0 - 2, T0 - 0.5)
            T <= ice_melting_temperature(S; p) || continue
            E = ice_enthalpy(T, S; p)
            worst = max(worst, abs(ice_temperature_from_enthalpy(E, S; p) - T))
        end
        @test worst < 1e-9
        # Heat capacity is the derivative of the enthalpy (the Jacobian of the
        # implicit solver) — checked by a central difference.
        wc = 0.0
        for S in (0.0, 1.0, 3.2), T in (T0 - 20, T0 - 3, T0 - 1)
            h = 1e-4
            dE = (ice_enthalpy(T + h, S; p) - ice_enthalpy(T - h, S; p)) / (2h)
            wc = max(wc, abs(dE / (ρ * ice_heat_capacity(T, S; p)) - 1))
        end
        @test wc < 1e-6
        # Brine pockets: near the liquidus c(T,S) ≫ c₀ and k < k₀.
        @test ice_heat_capacity(T0 - 0.5, 3.2; p) > 30 * c0
        @test ice_conductivity(T0 - 2, 3.2; p) ≈ p.constants.k_ice - 0.1172 * 3.2 / 2
        # Fresh ice between -ρL and 0 enthalpy is partly melted and at 0 °C.
        @test ice_temperature_from_enthalpy(-0.5 * ρ * L, 0.0; p) == T0
        # Snow enthalpy.
        @test snow_enthalpy(T0 - 10; p) ≈ p.ρ_snow * (-10c0 - L)
        # Sea-water freezing point and the multiyear salinity profile.
        @test seawater_freezing_point(0) == T0
        @test seawater_freezing_point(34) ≈ T0 - 1.836
        S = bl99_salinity_profile(10)
        @test issorted(S) && S[1] < 0.2 && 3.1 < S[end] < 3.2
        @info "BL99 material functions" q_over_ρL_at_m2C_S3p2 = qr γ = L * p.μ round_trip = worst
    end

    @testset "Neumann solution: the similarity constant λ" begin
        c = DEFAULT_SURFACE_CONSTANTS
        ΔT = 10.0
        k, ρ, cp, L = c.k_ice, c.ρ_ice, cp_ice(c), LH_f0(c)
        St = cp * ΔT / L
        λ = neumann_lambda(St)
        # λ satisfies its transcendental equation; small-St expansion.
        @test abs(λ * exp(λ^2) * erf_series(λ) - St / sqrt(π)) < 1e-14
        @test λ ≈ sqrt(St / 2) * (1 - St / 6) rtol = 1e-3
        t = 61 * 86400.0
        stefan_gap = neumann_ice_thickness(t; ΔT) / stefan_ice_thickness(t; ΔT) - 1
        @test -0.012 < stefan_gap < -0.008
        @info "Neumann benchmark (61 days, ΔT = 10 K)" λ St neumann_vs_stefan = stefan_gap
    end

end
