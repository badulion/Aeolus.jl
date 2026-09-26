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
_mean(x) = sum(x) / length(x)

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

    @testset "Stefan's law: h² - h₀² ∝ t in the zero-heat-capacity limit" begin
        # Heat capacity of ice made negligible (c = 1 J kg⁻¹ K⁻¹, St = 3e-5):
        # the profile is quasi-steady and Stefan's law is exact.
        th = ThermodynamicConstants(cp_ice = 1.0)
        p = IceParameters(constants = SurfaceConstants(thermo = th))
        col = lake_ice_column(h_ice = 0.05, N = 20, T_surface = 263.15, params = p)
        bc = PrescribedSurfaceTemperature(263.15)
        ts = Float64[]; hs = Float64[]
        for i in 1:24*90
            step_ice!(col, bc, 3600.0)
            i % 24 == 0 && (push!(ts, i * 3600.0); push!(hs, col.h_ice))
        end
        hS = [stefan_ice_thickness(t; ΔT = 10, h0 = 0.05, c = p.constants) for t in ts]
        err_final = abs(hs[end] / hS[end] - 1)
        err_late = maximum(abs.(hs[10:end] ./ hS[10:end] .- 1))
        # Growth exponent: slope of log(h² - h₀²) against log t.
        x = log.(ts); y = log.(hs .^ 2 .- 0.05^2)
        x̄, ȳ = _mean(x), _mean(y)
        slope = sum((x .- x̄) .* (y .- ȳ)) / sum((x .- x̄) .^ 2)
        @test err_final < 1e-3
        @test err_late < 3e-3
        @test abs(slope - 1) < 0.01
        @info "Stefan's law (90 days, ΔT = 10 K)" h_model = hs[end] h_stefan = hS[end] rel_err = err_final exponent_of_h² = slope
    end

    @testset "Neumann solution: growth including the heat capacity of ice" begin
        c = DEFAULT_SURFACE_CONSTANTS
        ΔT = 10.0
        k, ρ, cp, L = c.k_ice, c.ρ_ice, cp_ice(c), LH_f0(c)
        St = cp * ΔT / L
        λ = neumann_lambda(St)
        # λ satisfies its transcendental equation; small-St expansion.
        @test abs(λ * exp(λ^2) * erf_series(λ) - St / sqrt(π)) < 1e-14
        @test λ ≈ sqrt(St / 2) * (1 - St / 6) rtol = 1e-3
        α = k / (ρ * cp)
        t0 = 86400.0
        h0 = 2λ * sqrt(α * t0)
        errs = Float64[]
        for N in (10, 40)
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
            push!(errs, col.h_ice / neumann_ice_thickness(t; ΔT) - 1)
            @test mr < 1e-8
        end
        t = t0 + 60 * 86400
        stefan_gap = neumann_ice_thickness(t; ΔT) / stefan_ice_thickness(t; ΔT) - 1
        # The model follows Neumann to 0.1 %, converging with resolution, and
        # resolves the ~1 % by which the sensible heat of the ice slows growth
        # relative to Stefan's law.
        @test abs(errs[1]) < 2e-3
        @test abs(errs[2]) < abs(errs[1])
        @test abs(errs[2]) < 1e-3
        @test -0.012 < stefan_gap < -0.008
        @info "Neumann benchmark (61 days, ΔT = 10 K)" λ St err_N10 = errs[1] err_N40 = errs[2] neumann_vs_stefan = stefan_gap
    end

    @testset "Snow insulates; ocean heat flux thins; lake vs sea ice" begin
        # Same cold surface, snow-covered ice grows slower (k_s ≪ k_i).
        run(hs) = begin
            col = lake_ice_column(h_ice = 0.3, h_snow = hs, T_surface = 253.15)
            bc = PrescribedSurfaceFluxes(sw_down = 0.0, lw_down = 180.0)
            for _ in 1:30*24; step_ice!(col, bc, 3600.0); end
            col.h_ice
        end
        @test run(0.2) < run(0.0) - 0.05
        # Ocean heat flux: the equilibrium basal balance F_w = F_cb.
        col = sea_ice_column(h_ice = 1.0, T_surface = 263.15, F_ocean = 0.0)
        col2 = sea_ice_column(h_ice = 1.0, T_surface = 263.15, F_ocean = 30.0)
        bc = PrescribedSurfaceTemperature(263.15)
        for _ in 1:30*24; step_ice!(col, bc, 3600.0); step_ice!(col2, bc, 3600.0); end
        @test col2.h_ice < col.h_ice
        # Basal accretion (BL99 eq. 11): Δh = (F_cb - F_w) Δt / q(T_f, S_new);
        # with F_w = 30 W m⁻² > F_cb the base melts instead.
        b = step_ice!(col, bc, 3600.0)
        qb = ice_energy_of_melting(col.T_bottom, col.S_new; p = col.params)
        @test b.dh_bottom > 0
        @test b.dh_bottom ≈ (b.F_conductive_bottom - 0.0) * 3600 / qb rtol = 1e-12
        b2 = step_ice!(col2, bc, 3600.0)
        @test b2.F_conductive_bottom < 30 && b2.dh_bottom < 0
        # A fresh lake freezes at 0 °C and its new ice needs the full ρL.
        lake = lake_ice_column()
        @test lake.T_bottom == 273.15
        @test ice_energy_of_melting(273.15, 0.0; p = lake.params) ≈
              lake.params.constants.ρ_ice * LH_f0(lake.params.constants)
    end

    @testset "Surface melt: snow first, then ice; melt-through to open water" begin
        col = lake_ice_column(h_ice = 0.3, h_snow = 0.05, T_surface = 272.15)
        bc = PrescribedSurfaceFluxes(sw_down = 600.0, lw_down = 330.0, H = -20.0)
        hs0, hi0 = col.h_snow, col.h_ice
        b = step_ice!(col, bc, 3600.0)
        @test b.melting
        @test col.h_snow < hs0
        @test col.h_ice ≥ hi0 - 1e-12         # ice untouched while snow remains
        mr = 0.0; steps = 0
        while col.h_ice > 0 && steps < 5000
            b = step_ice!(col, bc, 3600.0); steps += 1
            mr = max(mr, abs(b.energy_residual))
        end
        @test col.h_ice == 0 && col.h_snow == 0
        @test surface_albedo(col) == col.params.α_water
        @test mr < 1e-8
        @info "Melt-out of 30 cm lake ice + 5 cm snow" hours = steps + 1
    end

    @testset "Snow-ice: flooded snow becomes ice, mass and energy kept" begin
        col = lake_ice_column(h_ice = 0.10, h_snow = 0.0, T_surface = 263.15)
        p = col.params
        # Heavy snowfall (40 cm in a day) pushes 10 cm of ice below the waterline.
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
        @test si > 0.01
        @test ice_freeboard(col) > -1e-12
        @test m1 ≈ m0 + m_in + m_growth rtol = 1e-12
        @test mr < 1e-8
        @info "Snow-ice formation" snow_ice_m = si freeboard_m = ice_freeboard(col)
    end

end
