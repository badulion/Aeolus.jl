# Phase 47 (#21) — the vegetated land column: soil-water stress, big-leaf
# canopy radiation, canopy aerodynamics and the coupled column.
#
# Independent references used below (none of them is the code under test):
#   * CLM4.5 Technical Note (Oleson et al. 2013, NCAR/TN-503+STR):
#     Table 8.1 (ψo, ψc in mm), Table 8.3 (Zeng 2001 ra, rb), Table 3.1
#     (leaf optics), Table 5.1 (Rz0m, Rd), eqs. 8.26-8.30 (βt, wilting
#     factor, root fractions), 5.126-5.128 (roughness), 5.115-5.119
#     (under-canopy transfer), 7.104-7.107 (root uptake).
#   * Closed-form integrals of the Beer's-law profiles, re-derived and
#     evaluated here by independent quadrature in depth and in direction.
#   * Kirchhoff / radiative equilibrium: an isothermal enclosure has no
#     net longwave exchange.

using Test
using Aeolus
using Aeolus.LandSurface
using Aeolus.SoilColumn: SoilThermalColumn

# composite Simpson on [a, b] with n (even) intervals
function _simpson(f, a, b; n = 2000)
    h = (b - a) / n
    s = f(a) + f(b)
    for i in 1:n-1
        s += (isodd(i) ? 4 : 2) * f(a + i * h)
    end
    return s * h / 3
end

@testset "LandSurface (Phase 47, #21)" begin

@testset "CLM4.5 PFT tables" begin
    # Table 8.1: ψo/ψc in mm, stored in m.
    @test CLM45_WATER_STRESS_POTENTIALS[:net_boreal] == (-66.0, -255.0)
    @test CLM45_WATER_STRESS_POTENTIALS[:bdt_temperate] == (-35.0, -224.0)
    @test CLM45_WATER_STRESS_POTENTIALS[:bes_temperate] == (-83.0, -428.0)
    @test CLM45_WATER_STRESS_POTENTIALS[:c4_grass] == (-74.0, -275.0)
    # Table 8.3
    @test CLM45_ROOT_PARAMETERS[:bet_tropical] == (7.0, 1.0)
    @test CLM45_ROOT_PARAMETERS[:c3_grass] == (11.0, 2.0)
    @test CLM45_ROOT_PARAMETERS[:corn] == (6.0, 3.0)
    # Table 3.1 / 5.1
    @test CLM45_LEAF_OPTICS[:net_temperate].ρ_nir == 0.35
    @test CLM45_LEAF_OPTICS[:bdt_temperate].τ_nir == 0.25
    @test CLM45_LEAF_OPTICS[:c3_grass].τ_nir == 0.34
    @test CLM45_AERODYNAMIC_PARAMETERS[:bet_tropical].R_z0m == 0.075
    @test CLM45_AERODYNAMIC_PARAMETERS[:c3_grass].R_d == 0.68
    # Every PFT of the photosynthesis table has the land-column columns.
    for k in keys(Aeolus.Photosynthesis.CLM45_PFT_PHOTOSYNTHESIS)
        @test haskey(CLM45_WATER_STRESS_POTENTIALS, k)
        @test haskey(CLM45_ROOT_PARAMETERS, k)
        @test haskey(CLM45_LEAF_OPTICS, k)
        @test haskey(CLM45_AERODYNAMIC_PARAMETERS, k)
    end
    v = pft_vegetation(:net_boreal)
    @test v.stomata isa MedlynStomata && v.stomata.g1 == 2.35
    @test v.photosynthesis.Vcmax25 == 62.6
    @test pft_vegetation(:c4_grass; model = :ball_berry).stomata.g1 == 4.0
    @test pft_vegetation(:c3_grass; K_n = 0.3).K_n == 0.3
    @test_throws ArgumentError pft_vegetation(:oak)
end

@testset "Root fractions (Zeng 2001)" begin
    for g in (clm5_soil_grid(), clm5_soil_grid(8), uniform_soil_grid(2.0, 7)),
        (ra, rb) in values(CLM45_ROOT_PARAMETERS)
        r = zeng_root_fractions(g, ra, rb)
        @test sum(r) ≈ 1 atol = 1e-14
        @test all(>=(0), r)
        # cumulative fraction above every interior interface = Y(z)
        Y(z) = 1 - (exp(-ra * z) + exp(-rb * z)) / 2
        for i in 1:length(g)-1
            @test sum(r[1:i]) ≈ Y(g.zh[i+1]) atol = 1e-14
        end
    end
    # Deeper-rooted types (smaller r_b) put more roots below 1 m.
    g = clm5_soil_grid()
    below1m(p) = sum(zeng_root_fractions(g, CLM45_ROOT_PARAMETERS[p]...)[g.z .> 1.0])
    @test below1m(:bet_tropical) > below1m(:net_boreal) > below1m(:c3_grass)
end

@testset "Wilting factor and root-zone stress βt" begin
    st = CLMWaterStress(-66.0, -255.0)
    @test wilting_factor(st, 0.0) == 1
    @test wilting_factor(st, -66.0) == 1
    @test wilting_factor(st, -255.0) == 0
    @test wilting_factor(st, -1000.0) == 0
    @test wilting_factor(st, (-66.0 - 255.0) / 2) ≈ 0.5
    @test_throws ArgumentError CLMWaterStress(-300.0, -100.0)

    g = clm5_soil_grid(10)
    roots = zeng_root_fractions(g, 6.0, 2.0)
    w = SoilWaterColumn(g, clapp_hornberger_texture(:loam); ψ = -1.0)
    β, wl = soil_water_stress(st, w, roots)
    @test β ≈ 1 atol = 1e-14                     # wet soil: no stress
    # Drying the profile uniformly: βt follows the wilting factor exactly.
    for ψ in (-50.0, -100.0, -160.0, -200.0, -254.0, -300.0)
        set_matric_potential!(w, ψ)
        β, wl = soil_water_stress(st, w, roots)
        @test β ≈ wilting_factor(st, ψ) atol = 1e-14
    end
    # A dry top over a wet subsoil: βt is the root fraction of the wet layers.
    ψprof = [i <= 4 ? -400.0 : -10.0 for i in 1:length(g)]
    set_matric_potential!(w, ψprof)
    β, _ = soil_water_stress(st, w, roots)
    @test β ≈ sum(roots[5:end]) atol = 1e-14
    # A frozen layer (T < T_f − 2) contributes nothing; pore ice scales w.
    set_matric_potential!(w, -10.0)
    w.T_soil[1] = 270.0
    β, wl = soil_water_stress(st, w, roots)
    @test wl[1] == 0
    @test β ≈ 1 - roots[1] atol = 1e-14
    w.T_soil[1] = 293.15
    w.θ_ice[2] = 0.25 * soil_porosity(w.hydraulics[2])
    _, wl = soil_water_stress(st, w, roots)
    @test wl[2] ≈ 0.75 atol = 1e-14

    # Root uptake: effective root fractions r_i w_i / βt, total E_t/ρw.
    w.θ_ice .= 0
    set_matric_potential!(w, [-(20.0 + 30i) for i in 1:length(g)])
    β, wl = soil_water_stress(st, w, roots)
    sink = root_water_uptake(3e-5, roots, wl, β)
    @test sum(sink) ≈ 3e-5 / 1000 rtol = 1e-13
    @test sink ≈ 3e-8 .* roots .* wl ./ β rtol = 1e-13
    @test all(iszero, root_water_uptake(3e-5, roots, wl, 0.0))
    @test all(iszero, root_water_uptake(-1e-6, roots, wl, β))
end

@testset "Big-leaf shortwave: conservation and closed forms" begin
    worst = 0.0
    for Ib in (0.0, 300.0), Id in (0.0, 120.0), μ in (0.1, 0.5, 1.0),
        L in (0.0, 0.3, 2.0, 6.0), σ in (0.0, 0.15, 0.8), αg in (0.0, 0.2, 1.0)
        r = bigleaf_shortwave(Ib, Id, μ, L; σ = σ, α_ground = αg)
        worst = max(worst, abs(r.sunlit + r.shaded + r.ground + r.reflected - Ib - Id))
        @test r.sunlit >= -1e-12 && r.shaded >= -1e-12 && r.ground >= 0
        @test r.L_sun + r.L_sha ≈ L atol = 1e-12
    end
    @test worst < 1e-12
    @info "big-leaf shortwave: worst energy residual (W m⁻²)" worst

    # Black leaves on a black ground: canopy absorbs the unintercepted
    # complement, sunlit leaves the whole beam, L_sun = (1 − e^{−kL})/k.
    μ, L = 0.6, 3.0
    kb = 0.5 / μ
    r = bigleaf_shortwave(500.0, 0.0, μ, L; σ = 0.0, α_ground = 0.0)
    @test r.sunlit ≈ 500 * (1 - exp(-kb * L)) rtol = 1e-13
    @test r.shaded ≈ 0 atol = 1e-10
    @test r.L_sun ≈ (1 - exp(-kb * L)) / kb rtol = 1e-13
    @test r.reflected == 0

    # Sunlit/shaded split against direct quadrature of the absorption
    # profiles in depth (per unit leaf area at depth l, times the sunlit
    # fraction e^{-k_b l}).
    for σ in (0.15, 0.8), L in (1.0, 5.0), μ in (0.3, 0.9)
        Ib, Id = 400.0, 100.0
        r = bigleaf_shortwave(Ib, Id, μ, L; σ = σ, α_ground = 0.0)
        kb = 0.5 / μ; kbp = kb * sqrt(1 - σ)
        # effective diffuse coefficient from the isotropic-sky transmittance,
        # computed independently by Simpson quadrature over the sky
        τd = _simpson(m -> 2m * exp(-0.5 / max(m, 1e-12) * sqrt(1 - σ) * L), 0.0, 1.0; n = 20000)
        kdp = -log(τd) / L
        ρh = (1 - sqrt(1 - σ)) / (1 + sqrt(1 - σ))
        ρcb = 1 - exp(-2ρh * kb / (1 + kb))
        ρcd = _simpson(m -> 2m * (1 - exp(-2ρh * (0.5 / max(m, 1e-12)) / (1 + 0.5 / max(m, 1e-12)))), 0.0, 1.0; n = 20000)
        tot(l) = (1 - ρcb) * kbp * Ib * exp(-kbp * l) + (1 - ρcd) * kdp * Id * exp(-kdp * l)
        dirdir(l) = (1 - σ) * kb * Ib                   # per sunlit leaf area
        sunabs(l) = exp(-kb * l) * (dirdir(l) + (1 - ρcd) * kdp * Id * exp(-kdp * l) +
                                    (1 - ρcb) * kbp * Ib * exp(-kbp * l) - (1 - σ) * kb * Ib * exp(-kb * l))
        @test r.sunlit ≈ _simpson(sunabs, 0.0, L) rtol = 1e-6
        @test r.sunlit + r.shaded ≈ _simpson(tot, 0.0, L) rtol = 1e-6
    end

    # Deep canopy: nothing reaches the ground and the reflection is ρ_cb.
    r = bigleaf_shortwave(1.0, 0.0, 0.5, 60.0; σ = 0.15, α_ground = 0.3)
    ρh = (1 - sqrt(0.85)) / (1 + sqrt(0.85))
    @test r.reflected ≈ 1 - exp(-2ρh * 1.0 / 2.0) rtol = 1e-10
    @test r.ground < 1e-12
    # No canopy: the ground takes everything it does not reflect.
    r = bigleaf_shortwave(300.0, 100.0, 0.5, 0.0; σ = 0.15, α_ground = 0.25)
    @test r.ground ≈ 300 && r.reflected ≈ 100 && r.sunlit == r.shaded == 0
    # Night: no sunlit leaves.
    r = bigleaf_shortwave(0.0, 50.0, -0.2, 3.0; σ = 0.15, α_ground = 0.2)
    @test r.L_sun == 0 && r.sunlit == 0
end

@testset "Big-leaf longwave" begin
    σSB = Aeolus.SurfaceExchange.DEFAULT_SURFACE_CONSTANTS.σ_SB
    worst = 0.0
    for Ldn in (250.0, 400.0), Tv in (270.0, 300.0), Tg in (265.0, 310.0),
        LS in (0.0, 0.5, 3.0, 8.0), εg in (0.9, 0.96, 1.0)
        r = bigleaf_longwave(Ldn, Tv, Tg, LS; ε_ground = εg, σ_SB = σSB)
        worst = max(worst, abs(r.canopy + r.ground + r.upward - Ldn))
    end
    @test worst < 1e-10
    # Isothermal enclosure: sky, leaves and ground at one temperature
    # exchange no net radiation, whatever the emissivities (Kirchhoff).
    for T in (260.0, 300.0), LS in (0.4, 4.0), εg in (0.9, 1.0)
        r = bigleaf_longwave(σSB * T^4, T, T, LS; ε_ground = εg, σ_SB = σSB)
        @test abs(r.canopy) < 1e-10 && abs(r.ground) < 1e-10
    end
    # No canopy: the ground sees the sky directly.
    r = bigleaf_longwave(320.0, 280.0, 290.0, 0.0; ε_ground = 0.95, σ_SB = σSB)
    @test r.ground ≈ 0.95 * (320 - σSB * 290.0^4) rtol = 1e-13
    @test r.canopy == 0
end

@testset "Canopy aerodynamics" begin
    # Full canopy (L+S ≥ 2): z0 = Rz0m z_top, d = Rd z_top (CLM4.5 5.126-5.128).
    a = canopy_roughness(20.0, 5.0, 0.055, 0.67, 0.01)
    @test a.z0m ≈ 1.1 rtol = 1e-13
    @test a.d ≈ 13.4 rtol = 1e-13
    @test canopy_roughness(20.0, 2.0, 0.055, 0.67, 0.01).z0m ≈ 1.1 rtol = 1e-13
    # No canopy: the ground's roughness and no displacement.
    b = canopy_roughness(20.0, 0.0, 0.055, 0.67, 0.01)
    @test b.z0m ≈ 0.01 && b.d == 0
    # Monotone in between.
    zs = [canopy_roughness(20.0, x, 0.055, 0.67, 0.01).z0m for x in 0:0.25:2]
    @test issorted(zs)
    # Under-canopy transfer: dense limit 0.004 (neutral/unstable), bare-soil
    # limit (κ/a)(z0 u*/ν)^−0.45, stable reduction by 1 + γ min(S, 10).
    u = 0.4
    @test undercanopy_transfer_coefficient(u, 50.0, 0.01; T_ac = 290.0, T_g = 295.0, z_top = 20.0) ≈ 0.004 rtol = 1e-12
    @test undercanopy_transfer_coefficient(u, 0.0, 0.01; T_ac = 290.0, T_g = 295.0, z_top = 20.0) ≈
          0.4 / 0.13 * (0.01 * u / 1.5e-5)^(-0.45) rtol = 1e-13
    S = 9.80665 * 20 * 2 / (292 * u^2)
    @test undercanopy_transfer_coefficient(u, 50.0, 0.01; T_ac = 292.0, T_g = 290.0, z_top = 20.0) ≈
          0.004 / (1 + 0.5 * min(S, 10)) rtol = 1e-12
end

# ---------------------------------------------------------------------------
# The coupled column under an idealised clear-sky diurnal cycle
# ---------------------------------------------------------------------------

function _ls_column(; ψ = -3.37, pft = :bdt_temperate, LAI = 4.0, SAI = 0.5,
                    z_top = 20.0, T = 292.0, hydrology = true)
    g = clm5_soil_grid(13)
    w = SoilWaterColumn(g, clapp_hornberger_texture(:loam); ψ = ψ,
                        ice_impedance = PowerLawIceImpedance())
    soil = SoilThermalColumn(w, clm_soil_thermal_properties(40.0, 20.0); T_soil = T,
                             hydrology = hydrology)
    return VegetatedLandColumn(soil, pft_vegetation(pft); LAI = LAI, SAI = SAI, z_top = z_top)
end

# Clear-sky day: sun up 6-18 h, peak 850 W m⁻², air 287-299 K, dry air.
function _ls_forcing(t_hours; rain = 0.0, q = 0.008, U = 3.0)
    h = mod(t_hours, 24)
    μ = max(0.0, sin(π * (h - 6) / 12)) * 0.9
    Ta = 293.0 + 6 * sin(π * (h - 9) / 12)
    return μ, SurfaceForcing(sw_down = 850μ, lw_down = 330.0, T_air = Ta, q_air = q,
                             U = U, z_ref = 40.0, rain = rain)
end

function _ls_run!(s, days; dt = 1800.0, rain_at = nothing, record = nothing)
    n = round(Int, days * 86400 / dt)
    for k in 0:n-1
        t = k * dt / 3600
        rain = rain_at !== nothing && rain_at(t) ? 2e-3 : 0.0     # 7.2 mm h⁻¹
        μ, f = _ls_forcing(t; rain = rain)
        fl = step_surface!(s, f, dt; cosz = μ, diffuse_fraction = 0.25)
        record === nothing || record(t, f, fl, s.last)
    end
    return s
end

@testset "Vegetated column: energy and water closure" begin
    s = _ls_column()
    soil = s.soil
    ρw = soil.water.constants.ρ_water
    S0 = soil_water_storage(soil.water)
    G0 = soil.budget.surface
    b0 = deepcopy(soil.water.budget)
    worst_col = worst_leaf = worst_sw = worst_G = 0.0
    nconv = nsteps = 0
    ΣG = 0.0
    _ls_run!(s, 4; rain_at = t -> 50 <= t < 52, record = (t, f, fl, d) -> begin
        worst_col = max(worst_col, abs(fl.Rn - fl.H - fl.LE - fl.G))
        worst_leaf = max(worst_leaf, abs(d.Rn_v - d.H_v - d.LE_v))
        worst_sw = max(worst_sw, abs(d.SW_v + d.SW_g + d.SW_up - f.sw_down))
        @test fl.H ≈ d.H_v + d.H_g
        @test fl.LE ≈ d.LE_v + d.LE_g
        @test fl.Rn ≈ d.Rn_v + d.Rn_g
        ΣG += fl.G * 1800.0
        nconv += d.converged; nsteps += 1
    end)
    # G handed back is exactly the heat the soil received
    worst_G = abs(ΣG - (soil.budget.surface - G0))
    @test worst_col < 1e-8
    @test worst_leaf < 1e-8
    @test worst_sw < 1e-10
    @test worst_G < 1e-6 * abs(ΣG) + 1e-3
    @test nconv == nsteps
    @test abs(s.budget.leaf_residual) < 1e-4
    # Water: storage change = rain + dew − transpiration − ground evaporation
    # delivered − runoff − drainage (kg m⁻² → m of water).
    wb = soil.water.budget
    ΔS = soil_water_storage(soil.water) - S0
    @test wb.extraction - b0.extraction ≈ s.budget.transpiration / ρw rtol = 1e-12
    inflow = (s.budget.rain + s.budget.dew) / ρw
    outflow = s.budget.transpiration / ρw + (wb.evaporation - b0.evaporation) +
              (wb.runoff - b0.runoff) + (wb.drainage - b0.drainage)
    water_res = ΔS - (inflow - outflow)
    @test abs(water_res) < 1e-12
    @info "vegetated column, 4 days: worst |Rn−H−LE−G|, leaf residual (W m⁻²), SW residual, water residual (m)" worst_col worst_leaf worst_sw water_res
    @info "vegetated column, 4 days: transpiration, ground evaporation (kg m⁻²), converged steps" s.budget.transpiration s.budget.ground_evaporation nconv / nsteps
end

@testset "Vegetated column: limits and interface" begin
    # Bare column (LAI = SAI = 0): no canopy fluxes, no photosynthesis.
    s = _ls_column(LAI = 0.0, SAI = 0.0)
    _ls_run!(s, 1; record = (t, f, fl, d) -> begin
        @test d.H_v == 0 && d.LE_v == 0 && d.GPP == 0
        @test abs(d.Rn_v) < 1e-12 && abs(d.SW_v) < 1e-12
        @test abs(fl.Rn - fl.H - fl.LE - fl.G) < 1e-9
    end)
    # Isothermal column under a black-body sky of the same temperature:
    # Rn = ε_eff (L↓ − σT⁴) = 0 (Kirchhoff), and in general
    # Rn = ε_eff (L↓ − σT⁴) with ε_eff = 1 − (1 − ε_g)(1 − ε_v)².
    s = _ls_column()
    σSB = s.soil.water.constants.σ_SB
    T = s.soil.T_skin
    s.T_v = T
    εeff = surface_emissivity(s)
    @test εeff ≈ 1 - (1 - s.soil.emissivity) * exp(-2 * (s.LAI + s.SAI)) rtol = 1e-14
    for Ldn in (σSB * T^4, 300.0, 400.0)
        f = SurfaceForcing(sw_down = 0.0, lw_down = Ldn, T_air = T)
        @test net_radiation(s, f) ≈ εeff * (Ldn - σSB * T^4) atol = 1e-10
    end
    @test radiative_temperature(s) ≈ T rtol = 1e-12
    @test moisture_availability(s) ≈ 1 atol = 1e-12
    z = roughness_lengths(s)
    @test z.z0m ≈ 0.055 * 20 && z.z0h == z.z0m == z.z0q
    _ls_run!(s, 0.5)
    @test 0 < surface_albedo(s) < 0.3
    @test surface_temperature(s) == s.T_ac
    @test surface_specific_humidity(s, SurfaceForcing()) == s.q_ac
    @test_throws ArgumentError step_surface!(s, SurfaceForcing(z_ref = 10.0), 1800.0)
end

@testset "Soil-water stress: Bowen ratio response to drought" begin
    # One clear day at fixed soil water, uniform ψ from field capacity to
    # beyond ψ_c of the broadleaf deciduous tree (−224 m): midday fluxes.
    ψs = (-3.37, -50.0, -100.0, -150.0, -200.0, -260.0)
    rows = []
    for ψ in ψs
        s = _ls_column(ψ = ψ, hydrology = false)
        mid = Ref{Any}(nothing)
        _ls_run!(s, 2; record = (t, f, fl, d) -> (t == 36.0 && (mid[] = (fl, d, f))))
        fl, d, f = mid[]
        push!(rows, (ψ = ψ, βt = d.β_t, LE = fl.LE, H = fl.H, B = fl.H / fl.LE,
                     Et = d.E_t, GPP = d.GPP, dT = d.T_v - f.T_air))
    end
    βt = [r.βt for r in rows]; LE = [r.LE for r in rows]; H = [r.H for r in rows]
    B = [r.B for r in rows]; GPP = [r.GPP for r in rows]; dT = [r.dT for r in rows]
    @test βt[1] ≈ 1 && βt[end] == 0
    @test issorted(βt; rev = true)
    @test issorted(LE; rev = true) && issorted(GPP; rev = true)
    @test issorted(H) && issorted(B) && issorted(dT)
    @test rows[end].Et == 0                 # beyond ψ_c the roots supply nothing
    @test B[end] > 5 * B[1]
    for r in rows
        @info "drought response at noon" r.ψ r.βt r.LE r.H r.B r.GPP r.dT
    end
end

@testset "Soil-water stress: dry-down" begin
    # 30 rain-free clear days from field capacity: the root zone dries,
    # βt falls, and the midday Bowen ratio rises day by day.
    s = _ls_column(pft = :c3_grass, LAI = 3.0, SAI = 0.3, z_top = 0.5)
    S0 = soil_water_storage(s.soil.water)
    days = Dict{Int,Any}()
    _ls_run!(s, 30; record = (t, f, fl, d) -> (mod(t, 24) == 12.0 &&
                                              (days[Int(t ÷ 24)] = (βt = d.β_t, B = fl.H / fl.LE, LE = fl.LE))))
    β = [days[k].βt for k in 0:29]
    Bw = [days[k].B for k in 0:29]
    @test all(diff(β) .<= 1e-12)
    @test β[end] < β[1]
    @test Bw[end] > Bw[1]
    ET = s.budget.transpiration + s.budget.ground_evaporation
    @test soil_water_storage(s.soil.water) - S0 ≈
          (s.budget.dew - ET) / 1000 - s.soil.water.budget.drainage rtol = 1e-9
    @info "dry-down, C3 grass: midday βt day 1/10/20/30, Bowen ratio day 1/30, ET (mm)" β[1] β[10] β[20] β[30] Bw[1] Bw[30] ET
end

end # LandSurface
