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

@testset "Freezing equilibrium in a layer" begin
    Tf = CST.thermo.T_freeze
    h = carsel_parrish_texture(:loam)
    # a layer at matric potential ψ starts to freeze at T*(ψ): drier soil later
    p = clm_soil_thermal_properties(40, 20)
    wc = SoilWaterColumn(uniform_soil_grid(1.0, 2), h; ψ = [-1.0, -10.0])
    col = SoilThermalColumn(wc, p; T_soil = 280.0, hydrology = false)
    W = copy(wc.θ_liq)
    Ts = [freezing_onset_temperature(-1.0), freezing_onset_temperature(-10.0)]
    @test Ts[2] < Ts[1] < Tf
    set_soil_temperature!(col, Ts .+ 1e-6)
    @test all(wc.θ_ice .== 0)
    set_soil_temperature!(col, Ts .- 0.05)
    @test all(wc.θ_ice .> 0)
    # equilibrium liquid content is the retention curve at the Clapeyron ψ,
    # water is conserved, and ψ (from total water) is untouched by freezing
    for i in 1:2
        @test wc.θ_liq[i] ≈ supercooled_liquid_content(h, Ts[i] - 0.05) rtol = 1e-12
        @test wc.θ_liq[i] + 0.917 * wc.θ_ice[i] ≈ W[i] rtol = 1e-14
    end
    @test wc.ψ ≈ [-1.0, -10.0]
    # enthalpy ↔ state is a bijection: re-deriving the state from H recovers it
    for T in (280.0, Tf - 0.01, Tf - 0.5, Tf - 20)
        set_soil_temperature!(col, T)
        Tb, θl, θi = copy(wc.T_soil), copy(wc.θ_liq), copy(wc.θ_ice)
        SCT._apply_enthalpy!(col)
        @test wc.T_soil ≈ Tb rtol = 1e-12
        @test wc.θ_ice ≈ θi rtol = 1e-9 atol = 1e-14
    end
    # sharp phase change: all but the unfrozen floor freezes just below T_f
    wc2 = SoilWaterColumn(uniform_soil_grid(1.0, 1), h; θ = 0.3)
    col2 = SoilThermalColumn(wc2, p; T_soil = Tf - 1e-6, freezing_point_depression = false)
    floor = residual_water_content(h) + 1e-3 * (soil_porosity(h) - residual_water_content(h))
    @test wc2.θ_liq[1] ≈ floor
    @test wc2.θ_ice[1] ≈ (0.3 - floor) / 0.917
    # the impedance and viscosity factors follow the layer's ice and temperature
    imp = PowerLawIceImpedance()
    wc3 = SoilWaterColumn(uniform_soil_grid(1.0, 4), h; ψ = -1.0, ice_impedance = imp)
    col3 = SoilThermalColumn(wc3, p; T_soil = 280.0)
    K0 = soil_layer_conductivity(wc3, 2)
    set_soil_temperature!(col3, [280.0, Tf - 2, 280.0, 280.0])
    @test soil_layer_conductivity(wc3, 2) / K0 ≈
          viscosity_conductivity_factor(Tf - 2) / viscosity_conductivity_factor(280.0) *
          10^(-6 * wc3.θ_ice[2] / soil_porosity(h)) rtol = 1e-10
end

@testset "Neumann two-phase freezing front" begin
    # Semi-infinite soil at T_i > T_f whose surface drops to T_s < T_f at
    # t = 0. Neumann's similarity solution (Carslaw & Jaeger 1959, §11.2):
    # the front is at X = 2λ√(D_f t), with λ the root of
    #   k_f ΔT_f e^{-λ²}/(erf λ √(π D_f)) - k_u ΔT_u e^{-λ² D_f/D_u}/(erfc(λ√(D_f/D_u)) √(π D_u))
    #     = L λ √D_f
    # Constant properties per phase come from the model's own constitutive
    # functions for the fully frozen and fully thawed soil.
    h = GardnerExponential(θ_r = 0.0, θ_s = 0.4, α = 1.0, K_s = 1e-6)
    p = clm_soil_thermal_properties(40, 20)
    W, Ti, Tsurf, Tf = 0.3, 275.15, 263.15, CST.thermo.T_freeze
    fl = 1e-3 * 0.4
    θimax = (W - fl) / 0.917
    kf, ku = soil_thermal_conductivity(p, 0.4, fl, θimax), soil_thermal_conductivity(p, 0.4, W, 0.0)
    Cf, Cu = soil_heat_capacity(p, 0.4, fl, θimax), soil_heat_capacity(p, 0.4, W, 0.0)
    L = 917 * LH_f0(CST) * θimax
    Df, Du = kf / Cf, ku / Cu
    F(λ) = kf * (Tf - Tsurf) * exp(-λ^2) / (_erf(λ) * sqrt(π * Df)) -
           ku * (Ti - Tf) * exp(-λ^2 * Df / Du) / (_erfc(λ * sqrt(Df / Du)) * sqrt(π * Du)) -
           L * λ * sqrt(Df)
    lo, hi = 1e-4, 3.0
    for _ in 1:200
        m = (lo + hi) / 2
        F(m) > 0 ? (lo = m) : (hi = m)
    end
    λ = (lo + hi) / 2
    t_end = 30 * 86400.0
    Xa = 2λ * sqrt(Df * t_end)
    Tan(z) = z < Xa ? Tsurf + (Tf - Tsurf) * _erf(z / (2sqrt(Df * t_end))) / _erf(λ) :
                      Ti - (Ti - Tf) * _erfc(z / (2sqrt(Du * t_end))) / _erfc(λ * sqrt(Df / Du))
    errs, terrs = Float64[], Float64[]
    # uniform Δz over the top 2.5 m (the front reaches 1.27 m), then
    # stretching by 10 % per layer to 15 m so the far field is unbounded
    stretched(Δz) = begin
        zh = collect(0.0:Δz:2.5)
        while zh[end] < 15
            push!(zh, zh[end] + 1.1 * (zh[end] - zh[end-1]))
        end
        SoilGrid(zh)
    end
    for (Δz, dt) in ((0.04, 3600.0), (0.02, 900.0), (0.01, 225.0))
        g = stretched(Δz)
        wc = SoilWaterColumn(g, h; θ = W)
        col = SoilThermalColumn(wc, p; T_soil = Ti, freezing_point_depression = false,
                                hydrology = false)
        E0 = soil_enthalpy(col)
        for _ in 1:round(Int, t_end / dt)
            step_soil_heat!(col, dt, SkinTemperatureTop(Tsurf))
        end
        # front position: the frozen depth equivalent of the ice in the column
        X = sum(g.dz .* wc.θ_ice) / θimax
        push!(errs, X / Xa - 1)
        # temperatures away from the one partly frozen cell, whose node is
        # pinned at T_f (the enthalpy method's staircase)
        mushy = findall(0 .< wc.θ_ice .< 0.999θimax)
        away = [i for i in eachindex(g.z) if g.z[i] < 5 && all(abs(i - m) > 1 for m in mushy)]
        Terr = maximum(abs.(wc.T_soil[away] .- Tan.(g.z[away])))
        push!(terrs, Terr)
        cons = soil_enthalpy(col) - E0 - (col.budget.surface - col.budget.bottom)
        @test abs(cons) < 1e-12 * abs(E0)
        @test length(mushy) <= 1
        @printf("  Neumann Δz = %.0f mm, Δt = %.0f s: X = %.4f m vs %.4f m (%.2e), max |ΔT| off the front cell %.4f K, energy residual %.1e J/m²\n",
                1000Δz, dt, X, Xa, X / Xa - 1, Terr, cons)
    end
    @test all(abs.(errs) .< 0.006)
    @test abs(errs[3]) < 0.0015
    @test all(terrs .< 0.1) && terrs[3] < terrs[1]
    # first-order convergence of the front on the enthalpy grid
    @test abs(errs[1]) / abs(errs[2]) > 1.6 && abs(errs[2]) / abs(errs[3]) > 1.6
    # Stefan limit: with λ from the one-phase quasi-steady law X ≈ √(2 k_f ΔT t / L)
    @test Xa < sqrt(2kf * (Tf - Tsurf) * t_end / L)          # warm soil below slows the front
end

@testset "Periodic surface temperature: damping depth and phase" begin
    # T(z,t) = T̄ + A e^{-z/d} sin(ωt - z/d),  d = √(2k/(Cω)),  and the
    # ground heat flux leads the surface temperature by π/4 with amplitude
    # A √(k C ω) (Carslaw & Jaeger 1959, §2.6).
    h = carsel_parrish_texture(:loam)
    p = clm_soil_thermal_properties(40, 20)
    for (name, P, depth, N, dt, ncyc) in (("diurnal", 86400.0, 1.0, 200, 150.0, 6),
                                         ("annual", 365 * 86400.0, 20.0, 200, 43200.0, 5))
        g = uniform_soil_grid(depth, N)
        wc = SoilWaterColumn(g, h; θ = 0.3)
        col = SoilThermalColumn(wc, p; T_soil = 288.15, hydrology = false)
        k = soil_thermal_layer_conductivity(col, 1)
        C = soil_heat_capacity(p, soil_porosity(h), 0.3, 0.0)
        ω = 2π / P
        d = sqrt(2k / (C * ω))
        A = 8.0
        nper = round(Int, P / dt)
        S, Cc = zeros(N), zeros(N)
        Gs = Gc = 0.0
        t = 0.0
        for cyc in 1:ncyc, _ in 1:nper
            t += dt
            st = step_soil_heat!(col, dt, SkinTemperatureTop(288.15 + A * sin(ω * (t - dt / 2))))
            if cyc == ncyc
                S .+= (wc.T_soil .- 288.15) .* sin(ω * t)
                Cc .+= (wc.T_soil .- 288.15) .* cos(ω * t)
                Gs += st.G * sin(ω * (t - dt / 2)); Gc += st.G * cos(ω * (t - dt / 2))
            end
        end
        amp = 2 .* sqrt.(S .^ 2 .+ Cc .^ 2) ./ nper
        lag = atan.(Cc, S) .* -1
        for zq in (d, 2d)
            i = argmin(abs.(g.z .- zq))
            z = g.z[i]
            @test amp[i] / A ≈ exp(-z / d) rtol = 0.012
            @test lag[i] ≈ z / d atol = 0.01
            @printf("  %s: z = %.3f m (%.1f d): amplitude ratio %.4f vs %.4f, lag %.4f vs %.4f rad\n",
                    name, z, z / d, amp[i] / A, exp(-z / d), lag[i], z / d)
        end
        Gamp, Glead = 2sqrt(Gs^2 + Gc^2) / nper, atan(Gc, Gs)
        @test Gamp ≈ A * sqrt(k * C * ω) rtol = 1e-3
        @test Glead ≈ π / 4 atol = 0.01
        @printf("  %s: d = %.4f m; G amplitude %.2f vs %.2f W/m², lead %.4f vs π/4 = %.4f\n",
                name, d, Gamp, A * sqrt(k * C * ω), Glead, π / 4)
    end
end

@testset "Energy conservation with freeze–thaw" begin
    # Freezing and thawing under a prescribed flux, both phase models,
    # layered soil: Δ enthalpy = ∫(G - F_bottom) dt to round-off, and the
    # latent heat shows up as a zero curtain at T_f.
    g = clm5_soil_grid(12)
    hyd = [i <= 4 ? carsel_parrish_texture(:silt_loam) : carsel_parrish_texture(:clay) for i in 1:12]
    th = [i <= 4 ? clm_soil_thermal_properties(20, 15) : clm_soil_thermal_properties(20, 50) for i in 1:12]
    for fpd in (true, false)
        wc = SoilWaterColumn(g, hyd; ψ = -0.5)
        col = SoilThermalColumn(wc, th; T_soil = 274.15, freezing_point_depression = fpd,
                                bottom = GeothermalHeatFlux(0.065), hydrology = false)
        E0, S0 = soil_enthalpy(col), soil_water_storage(wc)
        worst = 0.0
        for n in 1:24 * 40
            G = n <= 24 * 20 ? -80.0 : 80.0              # 20 days of cooling, 20 of warming
            E1 = soil_enthalpy(col)
            st = step_soil_heat!(col, 3600.0, GroundHeatFluxTop(G))
            worst = max(worst, abs(st.enthalpy_change - (st.G - st.bottom_flux) * 3600))
            @test st.enthalpy_change ≈ soil_enthalpy(col) - E1
            n == 24 * 20 && @test soil_ice_mass(col) > 50
        end
        cum = soil_enthalpy(col) - E0 - (col.budget.surface - col.budget.bottom)
        @test worst < 1e-6
        @test abs(cum) < 1e-6
        @test col.budget.surface ≈ 0 atol = 1e-6                     # ±80 W/m² for equal times
        @test col.budget.bottom ≈ -0.065 * 40 * 86400 rtol = 1e-12
        @test soil_water_storage(wc) ≈ S0 rtol = 1e-14               # phase change conserves water
        @printf("  fpd = %s: worst per-step energy residual %.1e J/m², cumulative %.1e J/m²\n",
                fpd, worst, cum)
    end
    # zero curtain: a saturated sharp-freezing layer sits at T_f while latent
    # heat is removed, for exactly ρ_ice L_f θ_ice,max Δz / |G| seconds
    h = GardnerExponential(θ_r = 0.0, θ_s = 0.4, α = 1.0, K_s = 1e-6)
    wc = SoilWaterColumn(uniform_soil_grid(0.1, 1), h; θ = 0.3)
    col = SoilThermalColumn(wc, clm_soil_thermal_properties(40, 20); T_soil = 273.15 + 1e-9,
                            freezing_point_depression = false, hydrology = false)
    θimax = (0.3 - 4e-4) / 0.917
    t_latent = 917 * LH_f0(CST) * θimax * 0.1 / 50.0
    nsteps = floor(Int, t_latent / 600) - 1
    for _ in 1:nsteps
        step_soil_heat!(col, 600.0, GroundHeatFluxTop(-50.0))
    end
    @test wc.T_soil[1] == CST.thermo.T_freeze
    @test soil_ice_mass(col) ≈ 917 * θimax * 0.1 * nsteps * 600 / t_latent rtol = 1e-6
    for _ in 1:3
        step_soil_heat!(col, 600.0, GroundHeatFluxTop(-50.0))
    end
    @test wc.T_soil[1] < CST.thermo.T_freeze
end

end # testset
