# Era V, Phase 51 — lake and sea-ice thermodynamics (issue #37).
#
# Bitz & Lipscomb (1999) energy-conserving multi-layer sea ice with brine
# pockets, snow, surface/basal melt and growth, coupled to SurfaceExchange.
#
# What is validated against what:
#   * material functions  — closed forms of BL99 §2 and the numbers quoted there;
#   * growth              — Stefan's law (1891) and the exact Neumann solution;
#   * energy              — column enthalpy change vs. the fluxes applied, every step;
#   * MU71 benchmark      — a validation run, validation/era5_ice_thermodynamics.jl:
#                           the BL99 *reference code* run on the same forcing and
#                           the published BL99 Table 1/2 equilibrium thicknesses.
# No observations are used: there is no field data set in this file.

using Test
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
        # contract delegation: ρ_i, k₀ are the SurfaceConstants values
        @test ρ == DEFAULT_SURFACE_CONSTANTS.ρ_ice
        @test ice_conductivity(T0 - 10, 0.0; p) == DEFAULT_SURFACE_CONSTANTS.k_ice
        # BL99 eq. (2): γ = L μ ≈ 18.0 kJ K kg⁻¹
        @test L * p.μ ≈ 18.0e3 rtol = 2e-3
        # BL99 §2: "q = 0.92 L₀ when T = -2 °C and S = 3.2 ppt"
        @test 0.915 < ice_energy_of_melting(T0 - 2, 3.2; p) / (ρ * L) < 0.93
        # q vanishes at the liquidus and reduces to pure ice for S = 0
        @test abs(ice_energy_of_melting(ice_melting_temperature(3.2; p), 3.2; p)) < 1e-6
        @test ice_energy_of_melting(T0 - 7, 0.0; p) ≈ ρ * (c0 * 7 + L)
        # inversion round trip and c = dE/dT (the implicit solver's Jacobian)
        worst = 0.0; wc = 0.0
        for S in (0.0, 3.2, 8.0), T in (T0 - 40, T0 - 10, T0 - 2)
            worst = max(worst, abs(ice_temperature_from_enthalpy(ice_enthalpy(T, S; p), S; p) - T))
            h = 1e-4
            dE = (ice_enthalpy(T + h, S; p) - ice_enthalpy(T - h, S; p)) / (2h)
            wc = max(wc, abs(dE / (ρ * ice_heat_capacity(T, S; p)) - 1))
        end
        @test worst < 1e-9
        @test wc < 1e-6
        @test seawater_freezing_point(34) ≈ T0 - 1.836
    end

    @testset "Neumann solution: growth including the heat capacity of ice" begin
        c = DEFAULT_SURFACE_CONSTANTS
        ΔT = 10.0
        k, ρ, cp, L = c.k_ice, c.ρ_ice, cp_ice(c), LH_f0(c)
        St = cp * ΔT / L
        λ = neumann_lambda(St)
        @test abs(λ * exp(λ^2) * erf_series(λ) - St / sqrt(π)) < 1e-14
        α = k / (ρ * cp)
        t0 = 86400.0
        h0 = 2λ * sqrt(α * t0)
        N = 10
        z = ((1:N) .- 0.5) ./ N .* h0
        Tl = (273.15 - ΔT) .+ ΔT .* erf_series.(z ./ (2sqrt(α * t0))) ./ erf_series(λ)
        col = lake_ice_column(h_ice = h0, N = N, T_surface = 273.15 - ΔT, T_layers = Tl)
        bc = PrescribedSurfaceTemperature(273.15 - ΔT)
        mr = 0.0
        for _ in 1:60*24
            b = step_ice!(col, bc, 3600.0)
            mr = max(mr, abs(b.energy_residual))
        end
        t = t0 + 60 * 86400
        @test mr < 1e-8
        @test abs(col.h_ice / neumann_ice_thickness(t; ΔT) - 1) < 2e-3
        # the ~1 % by which the sensible heat of the ice slows growth vs Stefan
        @test -0.012 < neumann_ice_thickness(t; ΔT) / stefan_ice_thickness(t; ΔT) - 1 < -0.008
        # conservative remap: layer enthalpy integral preserved
        dz = [0.1, 0.3, 0.05, 0.2]; E = [-3.1e8, -2.9e8, -2.5e8, -2.8e8]
        Enew, H = ITD._remap(dz, E, 7)
        @test sum(Enew) * H / 7 ≈ sum(E .* dz) rtol = 1e-14
    end

    @testset "Basal accretion and ocean heat flux (BL99 eq. 11)" begin
        col = sea_ice_column(h_ice = 1.0, T_surface = 263.15, F_ocean = 0.0)
        col2 = sea_ice_column(h_ice = 1.0, T_surface = 263.15, F_ocean = 30.0)
        bc = PrescribedSurfaceTemperature(263.15)
        for _ in 1:20; step_ice!(col, bc, 3600.0); step_ice!(col2, bc, 3600.0); end
        b = step_ice!(col, bc, 3600.0)
        qb = ice_energy_of_melting(col.T_bottom, col.S_new; p = col.params)
        @test b.dh_bottom > 0
        @test b.dh_bottom ≈ b.F_conductive_bottom * 3600 / qb rtol = 1e-12
        b2 = step_ice!(col2, bc, 3600.0)
        @test b2.F_conductive_bottom < 30 && b2.dh_bottom < 0
    end

    @testset "Surface melt: snow first, then ice; melt-through to open water" begin
        col = lake_ice_column(h_ice = 0.3, h_snow = 0.05, T_surface = 272.15)
        bc = PrescribedSurfaceFluxes(sw_down = 600.0, lw_down = 330.0, H = -20.0)
        hs0, hi0 = col.h_snow, col.h_ice
        b = step_ice!(col, bc, 3600.0)
        @test b.melting && col.h_snow < hs0
        @test col.h_ice ≥ hi0 - 1e-12         # ice untouched while snow remains
        mr = 0.0; steps = 0
        while col.h_ice > 0 && steps < 5000
            b = step_ice!(col, bc, 3600.0); steps += 1
            mr = max(mr, abs(b.energy_residual))
        end
        @test col.h_ice == 0 && col.h_snow == 0
        @test mr < 1e-8
    end

    @testset "Snow-ice: flooded snow becomes ice, mass and energy kept" begin
        col = lake_ice_column(h_ice = 0.10, h_snow = 0.0, T_surface = 263.15)
        p = col.params
        bc = PrescribedSurfaceTemperature(263.15; snowfall = 0.40 * p.ρ_snow / 86400,
                                          T_snowfall = 263.15)
        m0 = p.constants.ρ_ice * col.h_ice + p.ρ_snow * col.h_snow
        m_in = 0.0; si = 0.0; mr = 0.0; m_growth = 0.0
        for _ in 1:24
            b = step_ice!(col, bc, 3600.0)
            m_in += 0.40 * p.ρ_snow / 24
            m_growth += p.constants.ρ_ice * b.dh_bottom
            si += b.dh_snow_ice; mr = max(mr, abs(b.energy_residual))
        end
        m1 = p.constants.ρ_ice * col.h_ice + p.ρ_snow * col.h_snow
        @test si > 0.01 && ice_freeboard(col) > -1e-12
        @test m1 ≈ m0 + m_in + m_growth rtol = 1e-12
        @test mr < 1e-8
    end

    @testset "AbstractSurface: Monin–Obukhov coupling, sublimation, freeze-up" begin
        col = lake_ice_column(h_ice = 0.0)
        @test col isa AbstractSurface
        f = SurfaceForcing(sw_down = 0.0, lw_down = 200.0, T_air = 253.15, q_air = 5e-4,
                           U = 6.0, p = 101325.0, z_ref = 10.0)
        fl = step_surface!(col, f, 3600.0)
        @test col.h_ice > 0 && fl.H > 0 && fl.LE > 0    # open water freezes over
        seb = 0.0; mr = 0.0
        E_sum = 0.0; m0 = col.params.constants.ρ_ice * col.h_ice; growth = 0.0
        for _ in 1:48
            fl = step_surface!(col, f, 3600.0)
            b = col.budget
            seb = max(seb, abs(fl.Rn - fl.H - fl.LE - fl.G))
            mr = max(mr, abs(b.energy_residual))
            E_sum += fl.E * 3600
            growth += col.params.constants.ρ_ice * b.dh_bottom
        end
        @test seb < 1e-6 && mr < 1e-8
        # sublimation mass: ice lost at the top = ∫E dt (no melt, no snow)
        @test col.params.constants.ρ_ice * col.h_ice ≈ m0 + growth - E_sum rtol = 1e-9
        @test E_sum > 0
    end
end
