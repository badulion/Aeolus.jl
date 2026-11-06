# Phase 46 (#16) — SoilColumn: soil heat conduction, freeze–thaw
# and the soil column as an AbstractSurface.
#
# Validation: exact energy conservation; the Neumann (Carslaw & Jaeger
# 1959 §11.2) two-phase freezing-front solution; the damping depth and
# phase lag of periodic surface temperature waves and the π/4 lead of the
# ground heat flux (Carslaw & Jaeger §2.6); the Clapeyron slope of the
# freezing-point depression; closed-form limits of the Johansen (1975)
# conductivity; the skin energy balance.

using Test
using Aeolus
using Aeolus.SoilColumn
using Aeolus.SurfaceExchange: DEFAULT_SURFACE_CONSTANTS, SurfaceForcing, SurfaceFluxes,
                              turbulent_fluxes, surface_temperature, step_surface!,
                              AbstractSurface, grav, LH_f0

const SCT = Aeolus.SoilColumn
const CST = DEFAULT_SURFACE_CONSTANTS
_erf(x) = Aeolus.erf_series(x)
_erfc(x) = Aeolus.erfc_series(x)

@testset "SoilColumn heat and freeze–thaw (#16)" begin

@testset "Thermal properties (Johansen 1975)" begin
    # Johansen dry conductivity at porosity 0.4 (ρ_d = 1620 kg m⁻³), by hand
    @test johansen_dry_conductivity(0.4) ≈ 283.4 / 1165.86 rtol = 1e-12
    p = clm_soil_thermal_properties(40, 20)
    θs = 0.45
    kdry = johansen_dry_conductivity(θs)
    ksat_u = p.k_solids^(1 - θs) * CST.k_water^θs
    ksat_f = p.k_solids^(1 - θs) * CST.k_ice^θs
    # limits: dry, saturated unfrozen, saturated frozen (geometric mean)
    @test soil_thermal_conductivity(p, θs, 0.0, 0.0) ≈ kdry
    @test soil_thermal_conductivity(p, θs, θs, 0.0) ≈ ksat_u rtol = 1e-12
    @test soil_thermal_conductivity(p, θs, 0.0, θs) ≈ ksat_f rtol = 1e-12
    # fine-soil Kersten number log10(S) + 1
    @test (soil_thermal_conductivity(p, θs, 0.5θs, 0.0) - kdry) / (ksat_u - kdry) ≈
          log10(0.5) + 1 rtol = 1e-12
    @test soil_heat_capacity(p, θs, 0.0, 0.3) ≈ p.c_solids * 0.55 + 917 * 2108 * 0.3
end

@testset "Clapeyron freezing and freezing equilibrium" begin
    # slope of ψ(T) at T_f is ρ_w L_f / T_f ≈ 1.22 MPa/K
    Tf = CST.thermo.T_freeze
    slope = (freezing_matric_potential(Tf - 1e-4) - freezing_matric_potential(Tf - 2e-4)) / 1e-4
    @test slope * CST.ρ_water * grav(CST) / 1e6 ≈ CST.ρ_water * LH_f0(CST) / Tf / 1e6 rtol = 1e-3
    h = carsel_parrish_texture(:loam)
    p = clm_soil_thermal_properties(40, 20)
    wc = SoilWaterColumn(uniform_soil_grid(1.0, 2), h; ψ = [-1.0, -10.0])
    col = SoilThermalColumn(wc, p; T_soil = 280.0, hydrology = false)
    W = copy(wc.θ_liq)
    Ts = [freezing_onset_temperature(-1.0), freezing_onset_temperature(-10.0)]
    @test Ts[2] < Ts[1] < Tf
    set_soil_temperature!(col, Ts .- 0.05)
    # equilibrium liquid = retention curve at the Clapeyron ψ; water conserved
    @test wc.θ_liq ≈ [supercooled_liquid_content(h, T - 0.05) for T in Ts] rtol = 1e-12
    @test wc.θ_liq .+ 0.917 .* wc.θ_ice ≈ W rtol = 1e-14
    @test wc.ψ ≈ [-1.0, -10.0]
    # CLM power-law impedance 10^(-Ω θ_ice/θ_s)
    @test SCT.ice_impedance_factor(PowerLawIceImpedance(), 0.2, 0.4) ≈ 1e-3
end

@testset "Neumann two-phase freezing front" begin
    # Semi-infinite soil at T_i > T_f whose surface drops to T_s < T_f;
    # front at X = 2λ√(D_f t) with λ from Neumann's transcendental equation.
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
    # coarsest of the original three resolutions (Δz = 4 cm, Δt = 1 h)
    zh = collect(0.0:0.04:2.5)
    while zh[end] < 15
        push!(zh, zh[end] + 1.1 * (zh[end] - zh[end-1]))
    end
    g = SoilGrid(zh)
    wc = SoilWaterColumn(g, h; θ = W)
    col = SoilThermalColumn(wc, p; T_soil = Ti, freezing_point_depression = false, hydrology = false)
    E0 = soil_enthalpy(col)
    for _ in 1:round(Int, t_end / 3600.0)
        step_soil_heat!(col, 3600.0, SkinTemperatureTop(Tsurf))
    end
    X = sum(g.dz .* wc.θ_ice) / θimax
    mushy = findall(0 .< wc.θ_ice .< 0.999θimax)
    away = [i for i in eachindex(g.z) if g.z[i] < 5 && all(abs(i - m) > 1 for m in mushy)]
    @test abs(X / Xa - 1) < 0.006
    @test maximum(abs.(wc.T_soil[away] .- Tan.(g.z[away]))) < 0.1
    @test abs(soil_enthalpy(col) - E0 - (col.budget.surface - col.budget.bottom)) < 1e-12 * abs(E0)
end

@testset "Periodic surface temperature: damping depth and phase" begin
    # T = T̄ + A e^{-z/d} sin(ωt − z/d), d = √(2k/(Cω)); G leads T_s by π/4
    # with amplitude A √(k C ω) (Carslaw & Jaeger 1959, §2.6). Diurnal wave.
    h = carsel_parrish_texture(:loam)
    p = clm_soil_thermal_properties(40, 20)
    P, N, dt, ncyc = 86400.0, 200, 150.0, 6
    g = uniform_soil_grid(1.0, N)
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
    lag = -atan.(Cc, S)
    i = argmin(abs.(g.z .- d)); z = g.z[i]
    @test amp[i] / A ≈ exp(-z / d) rtol = 0.012
    @test lag[i] ≈ z / d atol = 0.005
    @test 2sqrt(Gs^2 + Gc^2) / nper ≈ A * sqrt(k * C * ω) rtol = 1e-3
    @test atan(Gc, Gs) ≈ π / 4 atol = 0.005
end

@testset "Initial soil from data: a profile on the nodes (#521)" begin
    zd = [0.035, 0.175, 0.64, 1.945]                       # ERA5's layer centres
    g = clm_exponential_soil_grid(10)
    # a linear profile is reproduced at every node between the levels (round-off)
    lin(z) = 290.0 - 2.5z
    T = soil_profile_on_nodes(zd, lin.(zd), g.z)
    inside = zd[1] .<= g.z .<= zd[end]
    @test maximum(abs.(T[inside] .- lin.(g.z[inside]))) < 1e-12
    # above the first level the shallowest value, below the last the deepest (held)
    @test all(T[g.z .< zd[1]] .== lin(zd[1])) && all(T[g.z .> zd[end]] .== lin(zd[end]))
    @test count(g.z .> zd[end]) == 1                       # the CLM column's 2.87 m node (the 9th is at 1.73 m)
    # a fill value (NaN) is skipped: interpolation across it
    T2 = soil_profile_on_nodes(zd, [lin(zd[1]), NaN, lin(zd[3]), lin(zd[4])], g.z)
    @test maximum(abs.(T2[inside] .- lin.(g.z[inside]))) < 1e-12
    @test soil_profile_on_nodes(zd, fill(NaN, 4), g.z) === nothing
    @test_throws ArgumentError soil_profile_on_nodes(zd[[2, 1, 3, 4]], lin.(zd), g.z)
end

@testset "Initial soil without data: the annual wave removes the slow mode (#521)" begin
    # A 15 m column (the base 6.6 damping depths down) under the annual
    # surface wave, 30 days of hourly steps. Started on the wave it stays on
    # the analytic solution (Carslaw & Jaeger §2.6) within the discretisation
    # error; started uniform at the mean (what [urban] t_soil0 does) it is
    # off by the slow mode, which decays with τ of months.
    h = carsel_parrish_texture(:loam)
    p = clm_soil_thermal_properties(40, 20)
    g = uniform_soil_grid(15.0, 150)
    θ = 0.3
    k = soil_thermal_conductivity(p, soil_porosity(h), θ, 0.0)
    C = soil_heat_capacity(p, soil_porosity(h), θ, 0.0)
    d = soil_damping_depth(k, C)
    @test 1.5 < d < 3.5                                     # "2–3 m in moist loam"
    @test d ≈ sqrt(2 * (k / C) / (2π / SOIL_YEAR)) rtol = 1e-14
    wave(z, t) = annual_soil_temperature(z, t; T_mean = 283.0, amplitude = 9.0, damping_depth = d,
                                         t_max = 200 * 86400.0)
    t0, dt, n = 120 * 86400.0, 3600.0, 30 * 24              # 30 days from day 120 (spring)
    function run(Tinit)
        col = SoilThermalColumn(SoilWaterColumn(g, h; θ = θ), p; T_soil = Tinit, hydrology = false,
                                bottom = Aeolus.SoilColumn.BottomTemperature(283.0))
        t = t0
        for _ in 1:n
            step_soil_heat!(col, dt, SkinTemperatureTop(wave(0.0, t + dt / 2)))
            t += dt
        end
        return col.water.T_soil .- wave.(g.z, t)
    end
    top3 = g.z .<= 3.0
    e_wave = maximum(abs.(run(wave.(g.z, t0))[top3]))
    e_unif = maximum(abs.(run(283.0)[top3]))
    @info "annual wave: max |T − T_analytic| over the top 3 m after 30 d" e_wave e_unif
    @test e_wave < 0.05                                     # dz = 0.1 m, dt = 1 h
    @test e_unif > 1.0                                      # the slow mode is still there
end

@testset "Energy conservation with freeze–thaw, zero curtain" begin
    g = clm5_soil_grid(12)
    hyd = [i <= 4 ? carsel_parrish_texture(:silt_loam) : carsel_parrish_texture(:clay) for i in 1:12]
    th = [i <= 4 ? clm_soil_thermal_properties(20, 15) : clm_soil_thermal_properties(20, 50) for i in 1:12]
    wc = SoilWaterColumn(g, hyd; ψ = -0.5)
    col = SoilThermalColumn(wc, th; T_soil = 274.15, freezing_point_depression = true,
                            bottom = GeothermalHeatFlux(0.065), hydrology = false)
    E0, S0 = soil_enthalpy(col), soil_water_storage(wc)
    worst = 0.0; icemax = 0.0
    for n in 1:24 * 40
        G = n <= 24 * 20 ? -80.0 : 80.0              # 20 days cooling, 20 warming
        st = step_soil_heat!(col, 3600.0, GroundHeatFluxTop(G))
        worst = max(worst, abs(st.enthalpy_change - (st.G - st.bottom_flux) * 3600))
        icemax = max(icemax, soil_ice_mass(col))
    end
    @test icemax > 50
    @test worst < 1e-6
    @test abs(soil_enthalpy(col) - E0 - (col.budget.surface - col.budget.bottom)) < 1e-6
    @test soil_water_storage(wc) ≈ S0 rtol = 1e-14               # phase change conserves water
    # zero curtain: a sharp-freezing layer sits at T_f for ρ_i L_f θ_i,max Δz/|G|
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
end

@testset "Skin energy balance: an AbstractSurface, Q_extra (#129)" begin
    h = carsel_parrish_texture(:loam)
    p = clm_soil_thermal_properties(40, 20)
    wc = SoilWaterColumn(clm5_soil_grid(), h; ψ = -3.0)
    col = SoilThermalColumn(wc, p; T_soil = 288.15, hydrology = false)
    @test col isa AbstractSurface
    E0 = soil_enthalpy(col)
    worst = 0.0; lin = Float64[]
    f = SurfaceForcing(sw_down = 0.0, lw_down = 300.0, T_air = 283.15, q_air = 0.005, U = 3.0)
    for n in 1:36
        fl = step_surface!(col, f, 600.0)
        worst = max(worst, abs(fl.Rn - fl.H - fl.LE - fl.G))
        fn = turbulent_fluxes(col, f)
        push!(lin, abs((fn.Rn - fn.H - fn.LE) - fl.G))
    end
    @test worst < 1e-10
    @test soil_enthalpy(col) - E0 ≈ col.budget.surface rtol = 1e-12
    # one linearisation per step: error second order in ΔT_skin
    @test maximum(lin[4:end]) < 0.5
    # extra energy onto the skin: G = Rn − H − LE + Q, part goes to the air
    mk() = SoilThermalColumn(SoilWaterColumn(clm_exponential_soil_grid(10), h; θ = 0.2), p;
                             T_soil = 288.0, bottom = BottomTemperature(288.0), hydrology = false)
    f2 = SurfaceForcing(sw_down = 400.0, lw_down = 330.0, T_air = 290.0, q_air = 0.007, U = 3.0)
    base, warm = mk(), mk()
    E0 = soil_enthalpy(warm)
    local fb, fw
    for _ in 1:60
        fb = step_surface!(base, f2, 30.0)
        fw = step_surface!(warm, f2, 30.0; Q_extra = 100.0)
    end
    @test fw.G ≈ fw.Rn - fw.H - fw.LE + 100.0 atol = 1e-9
    @test soil_enthalpy(warm) - E0 ≈ warm.budget.surface - warm.budget.bottom rtol = 1e-10
    @test 0 < fw.G - fb.G < 100 && fw.H > fb.H
end

@testset "Soil-limited evaporation returns the delivered water (#129)" begin
    h = carsel_parrish_texture(:loam)
    p = clm_soil_thermal_properties(40, 20)
    col = SoilThermalColumn(SoilWaterColumn(clm_exponential_soil_grid(10), h; θ = 0.10), p;
                            T_soil = 290.0, bottom = BottomTemperature(290.0))
    f = SurfaceForcing(sw_down = 800.0, lw_down = 340.0, T_air = 298.0, q_air = 0.005, U = 2.0,
                       z_ref = 3.0)
    ρw = col.water.constants.ρ_water
    E0 = soil_enthalpy(col) + col.budget.bottom - col.budget.water
    W0 = soil_water_storage(col.water)
    drain0 = col.water.budget.drainage
    Eair = 0.0; Gsum = 0.0; limited = 0
    for n in 1:120
        m0 = col.budget.latent_mismatch
        fl = step_surface!(col, f, 30.0)
        limited += col.budget.latent_mismatch > m0
        Eair += fl.E * 30.0; Gsum += fl.G * 30.0
    end
    @test limited > 0
    ΔW = soil_water_storage(col.water) - W0
    drain = col.water.budget.drainage - drain0
    @test Eair / ρw ≈ -(ΔW + drain) rtol = 1e-8
    @test soil_enthalpy(col) + col.budget.bottom - col.budget.water - E0 ≈ Gsum rtol = 1e-9
end

@testset "Coupled heat and water: frozen soil sheds rain" begin
    h = carsel_parrish_texture(:loam)
    p = clm_soil_thermal_properties(40, 20)
    runoff = Dict{Bool,Float64}()
    for frozen in (false, true)
        wc = SoilWaterColumn(clm5_soil_grid(), h; ψ = -0.3, ice_impedance = PowerLawIceImpedance())
        col = SoilThermalColumn(wc, p; T_soil = frozen ? 268.15 : 278.15)
        E0, S0 = soil_enthalpy(col), soil_water_storage(wc)
        for _ in 1:12                               # 6 h of 5 mm/h rain
            f = SurfaceForcing(sw_down = 0.0, lw_down = 280.0, T_air = frozen ? 268.15 : 278.15,
                               q_air = 0.003, U = 2.0, rain = 5.0 / 3600)
            step_surface!(col, f, 1800.0)
        end
        b, wb = col.budget, wc.budget
        frozen && @test soil_enthalpy(col) - E0 ≈ b.surface - b.bottom + b.water rtol = 1e-12 atol = 1e-3
        frozen && @test soil_water_storage(wc) - S0 ≈ wb.infiltration - wb.drainage rtol = 1e-10 atol = 1e-15
        runoff[frozen] = wb.runoff
    end
    @test runoff[false] < 0.1 * 0.03 && runoff[true] > 0.5 * 0.03
end

end # testset
