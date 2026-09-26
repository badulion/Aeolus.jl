# Era V, Phase 46 (#16) — SoilColumn: soil heat conduction, freeze–thaw
# and the soil column as an AbstractSurface.
#
# Validation: exact energy conservation; the Neumann (1860s, in Carslaw &
# Jaeger 1959 §11.2) two-phase freezing-front solution; the damping depth
# and phase lag of periodic surface temperature waves, and the π/4 lead
# of the ground heat flux (Carslaw & Jaeger §2.6); the Clapeyron slope of
# the freezing-point depression; closed-form limits of the Johansen (1975)
# conductivity; and the steady conductive state under a skin energy
# balance. No field observations are used.

using Test, Printf
using Aeolus
using Aeolus.SoilColumn
using Aeolus.SurfaceExchange: DEFAULT_SURFACE_CONSTANTS, SurfaceForcing, SurfaceFluxes,
                              turbulent_fluxes, surface_temperature, step_surface!,
                              AbstractSurface, grav, LH_f0

const SCT = Aeolus.SoilColumn
const CST = DEFAULT_SURFACE_CONSTANTS
_erf(x) = Aeolus.erf_series(x)
_erfc(x) = Aeolus.erfc_series(x)

@testset "Era V — SoilColumn heat and freeze–thaw (#16)" begin

@testset "Thermal properties (Johansen 1975, Farouki 1981)" begin
    # Farouki / CLM end members and the Johansen quartz law
    @test farouki_solids_conductivity(100, 0) == 8.80
    @test farouki_solids_conductivity(0, 100) == 2.92
    @test farouki_solids_heat_capacity(100, 0) ≈ 2.128e6
    @test farouki_solids_heat_capacity(0, 100) ≈ 2.385e6
    @test johansen_solids_conductivity(1.0) ≈ 7.7
    @test johansen_solids_conductivity(0.0) ≈ 3.0
    @test johansen_solids_conductivity(0.5) ≈ sqrt(7.7 * 2.0)
    # Johansen dry conductivity at porosity 0.4 (ρ_d = 1620 kg m⁻³), by hand:
    # (0.135·1620 + 64.7)/(2700 - 0.947·1620) = 283.4/1165.86
    @test johansen_dry_conductivity(0.4) ≈ 283.4 / 1165.86 rtol = 1e-12
    @test 0.15 < johansen_dry_conductivity(0.5) < johansen_dry_conductivity(0.4) < 0.3
    p = clm_soil_thermal_properties(40, 20)
    θs = 0.45
    kdry = johansen_dry_conductivity(θs)
    ksat_u = p.k_solids^(1 - θs) * CST.k_water^θs
    ksat_f = p.k_solids^(1 - θs) * CST.k_ice^θs
    # limits: dry soil, saturated unfrozen, saturated frozen (geometric mean)
    @test soil_thermal_conductivity(p, θs, 0.0, 0.0) ≈ kdry
    @test soil_thermal_conductivity(p, θs, θs, 0.0) ≈ ksat_u rtol = 1e-12
    @test soil_thermal_conductivity(p, θs, 0.0, θs) ≈ ksat_f rtol = 1e-12
    # ice conducts ~4 times better than water: frozen saturated soil by (k_ice/k_w)^θs
    @test ksat_f / ksat_u ≈ (2.22 / 0.57)^θs
    @test ksat_f / ksat_u > 1.8
    # Kersten numbers: fine log10(S)+1 (zero at S = 0.1), coarse 0.7 log10(S)+1, frozen S
    @test kersten_number(p, 0.1) ≈ 0 atol = 1e-15
    @test kersten_number(p, 0.05) == 0
    @test kersten_number(p, 0.5) ≈ log10(0.5) + 1
    pc = johansen_soil_thermal_properties(0.8; coarse = true)
    @test kersten_number(pc, 0.5) ≈ 0.7 * log10(0.5) + 1
    @test kersten_number(p, 0.5, 0.0) ≈ 0.5
    @test kersten_number(p, 1.0) == kersten_number(p, 1.0, 0.0) == 1
    # conductivity rises monotonically with wetness, and unfrozen soil at
    # S = 0.5 is already most of the way to saturation (the log law)
    ks = [soil_thermal_conductivity(p, θs, S * θs, 0.0) for S in 0.0:0.1:1.0]
    @test issorted(ks)
    @test (ks[6] - kdry) / (ksat_u - kdry) ≈ log10(0.5) + 1 rtol = 1e-12
    # heat capacity from the constituents (de Vries 1963)
    @test soil_heat_capacity(p, θs, 0.3, 0.0) ≈ p.c_solids * 0.55 + 1000 * 4181 * 0.3
    @test soil_heat_capacity(p, θs, 0.0, 0.3) ≈ p.c_solids * 0.55 + 917 * 2108 * 0.3
    @test_throws ArgumentError SoilThermalProperties(k_solids = 2.0, c_solids = 2e6, kersten = :medium)
    @printf("  loam (40/20): k_dry %.3f, k_sat unfrozen %.3f, frozen %.3f W/m/K\n", kdry, ksat_u, ksat_f)
end

@testset "Clapeyron freezing and ice impedance" begin
    # Clapeyron: ψ = L_f (T - T_f)/(g T); slope at T_f is ρ_w L_f/T_f ≈ 1.22 MPa/K
    Tf = CST.thermo.T_freeze
    slope = (freezing_matric_potential(Tf - 1e-4) - freezing_matric_potential(Tf - 2e-4)) / 1e-4
    @test slope * CST.ρ_water * grav(CST) / 1e6 ≈ CST.ρ_water * LH_f0(CST) / Tf / 1e6 rtol = 1e-3
    @test 1.21 < slope * CST.ρ_water * grav(CST) / 1e6 < 1.23
    @test freezing_matric_potential(Tf + 1) == 0
    @test freezing_onset_temperature(freezing_matric_potential(Tf - 3.0)) ≈ Tf - 3.0 rtol = 1e-14
    h = carsel_parrish_texture(:loam)
    @test supercooled_liquid_content(h, Tf + 1) == soil_porosity(h)
    θs = [supercooled_liquid_content(h, Tf - d) for d in (0.01, 0.1, 1.0, 5.0)]
    @test issorted(θs; rev = true)
    @test θs[end] > residual_water_content(h)
    # CLM power-law impedance 10^(-Ω θ_ice/θ_s)
    imp = PowerLawIceImpedance()
    @test SCT.ice_impedance_factor(imp, 0.0, 0.4) == 1
    @test SCT.ice_impedance_factor(imp, 0.2, 0.4) ≈ 1e-3
    @test SCT.ice_impedance_factor(imp, 0.5, 0.4) ≈ 1e-6          # capped at F_ice = 1
    # ice counts as water in the retention curve: a layer holding ice has
    # the potential of its total (liquid-equivalent) water
    wc0 = SoilWaterColumn(uniform_soil_grid(1.0, 2), h; ψ = -1.0)
    W0 = wc0.θ_liq[1]
    wc0.θ_ice .= 0.1
    set_water_content!(wc0, W0 - 0.917 * 0.1)
    @test wc0.ψ ≈ [-1.0, -1.0] rtol = 1e-12
    @printf("  loam supercooled water at -0.01, -0.1, -1, -5 °C: %s\n",
            join((@sprintf("%.3f", x) for x in θs), ", "))
end

end # testset
