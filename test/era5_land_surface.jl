# Phase 47 (#21) — the vegetated land column: soil-water stress, big-leaf
# canopy radiation, canopy aerodynamics and the coupled column.
#
# Independent references used below (none of them is the code under test):
#   * CLM4.5 Technical Note (Oleson et al. 2013, NCAR/TN-503+STR):
#     Table 8.1 (ψo, ψc), Table 8.3 (Zeng 2001 ra, rb), eqs. 8.26-8.30 (βt,
#     wilting factor, root fractions), 5.126-5.128 (roughness), 5.115-5.119
#     (under-canopy transfer), 7.104-7.107 (root uptake).
#   * Closed-form integrals of the Beer's-law profiles, evaluated here by
#     independent quadrature in depth and in direction.
#   * Kirchhoff: an isothermal enclosure has no net longwave exchange.

using Test
using Aeolus
using Aeolus.LandSurface
using Aeolus.SoilColumn: SoilThermalColumn

# composite Simpson on [a, b] with n (even) intervals
function _ls_simpson(f, a, b; n = 2000)
    h = (b - a) / n
    s = f(a) + f(b)
    for i in 1:n-1
        s += (isodd(i) ? 4 : 2) * f(a + i * h)
    end
    return s * h / 3
end

@testset "LandSurface (Phase 47, #21)" begin

@testset "CLM4.5 PFT tables" begin
    @test CLM45_WATER_STRESS_POTENTIALS[:bdt_temperate] == (-35.0, -224.0)
    @test CLM45_ROOT_PARAMETERS[:c3_grass] == (11.0, 2.0)
    @test all(k -> haskey(CLM45_WATER_STRESS_POTENTIALS, k) && haskey(CLM45_ROOT_PARAMETERS, k) &&
                   haskey(CLM45_LEAF_OPTICS, k) && haskey(CLM45_AERODYNAMIC_PARAMETERS, k),
              keys(Aeolus.Photosynthesis.CLM45_PFT_PHOTOSYNTHESIS))
    v = pft_vegetation(:net_boreal)
    @test v.stomata isa MedlynStomata && v.stomata.g1 == 2.35
end

@testset "Root fractions (Zeng 2001)" begin
    worst = 0.0
    for g in (clm5_soil_grid(), uniform_soil_grid(2.0, 7)), (ra, rb) in values(CLM45_ROOT_PARAMETERS)
        r = zeng_root_fractions(g, ra, rb)
        Y(z) = 1 - (exp(-ra * z) + exp(-rb * z)) / 2
        worst = max(worst, abs(sum(r) - 1),
                    maximum(abs(sum(r[1:i]) - Y(g.zh[i+1])) for i in 1:length(g)-1))
    end
    @test worst < 1e-14
end

@testset "Wilting factor, root-zone stress βt, root uptake" begin
    st = CLMWaterStress(-66.0, -255.0)
    @test wilting_factor(st, -66.0) == 1 && wilting_factor(st, -255.0) == 0
    @test wilting_factor(st, (-66.0 - 255.0) / 2) ≈ 0.5
    g = clm5_soil_grid(10)
    roots = zeng_root_fractions(g, 6.0, 2.0)
    w = SoilWaterColumn(g, clapp_hornberger_texture(:loam); ψ = -1.0)
    set_matric_potential!(w, -160.0)
    @test soil_water_stress(st, w, roots)[1] ≈ wilting_factor(st, -160.0) atol = 1e-14
    # a dry top over a wet subsoil: βt is the root fraction of the wet layers
    set_matric_potential!(w, [i <= 4 ? -400.0 : -10.0 for i in 1:length(g)])
    @test soil_water_stress(st, w, roots)[1] ≈ sum(roots[5:end]) atol = 1e-14
    # a frozen layer contributes nothing
    set_matric_potential!(w, -10.0)
    w.T_soil[1] = 270.0
    β, wl = soil_water_stress(st, w, roots)
    @test wl[1] == 0
    @test β ≈ 1 - roots[1] atol = 1e-14
    w.T_soil[1] = 293.15
    set_matric_potential!(w, [-(20.0 + 30i) for i in 1:length(g)])
    β, wl = soil_water_stress(st, w, roots)
    sink = root_water_uptake(3e-5, roots, wl, β)
    @test sink ≈ 3e-8 .* roots .* wl ./ β rtol = 1e-13
    @test sum(sink) ≈ 3e-5 / 1000 rtol = 1e-13
end

@testset "Big-leaf shortwave: conservation and closed forms" begin
    worst = 0.0
    for Ib in (0.0, 300.0), μ in (0.1, 1.0), L in (0.0, 2.0, 6.0), σ in (0.15, 0.8), αg in (0.0, 1.0)
        r = bigleaf_shortwave(Ib, 120.0, μ, L; σ = σ, α_ground = αg)
        worst = max(worst, abs(r.sunlit + r.shaded + r.ground + r.reflected - Ib - 120.0))
    end
    @test worst < 1e-12
    # black leaves, black ground: sunlit leaves take the whole beam
    μ, L = 0.6, 3.0
    kb = 0.5 / μ
    r = bigleaf_shortwave(500.0, 0.0, μ, L; σ = 0.0, α_ground = 0.0)
    @test r.sunlit ≈ 500 * (1 - exp(-kb * L)) rtol = 1e-13
    @test r.L_sun ≈ (1 - exp(-kb * L)) / kb rtol = 1e-13
    # sunlit/shaded split against quadrature in depth and over the sky
    σ, L, μ, Ib, Id = 0.15, 5.0, 0.3, 400.0, 100.0
    r = bigleaf_shortwave(Ib, Id, μ, L; σ = σ, α_ground = 0.0)
    kb = 0.5 / μ; kbp = kb * sqrt(1 - σ)
    τd = _ls_simpson(m -> 2m * exp(-0.5 / max(m, 1e-12) * sqrt(1 - σ) * L), 0.0, 1.0; n = 20000)
    kdp = -log(τd) / L
    ρh = (1 - sqrt(1 - σ)) / (1 + sqrt(1 - σ))
    ρcb = 1 - exp(-2ρh * kb / (1 + kb))
    ρcd = _ls_simpson(m -> 2m * (1 - exp(-2ρh * (0.5 / max(m, 1e-12)) / (1 + 0.5 / max(m, 1e-12)))), 0.0, 1.0; n = 20000)
    tot(l) = (1 - ρcb) * kbp * Ib * exp(-kbp * l) + (1 - ρcd) * kdp * Id * exp(-kdp * l)
    sunabs(l) = exp(-kb * l) * ((1 - σ) * kb * Ib + (1 - ρcd) * kdp * Id * exp(-kdp * l) +
                                (1 - ρcb) * kbp * Ib * exp(-kbp * l) - (1 - σ) * kb * Ib * exp(-kb * l))
    @test r.sunlit ≈ _ls_simpson(sunabs, 0.0, L) rtol = 1e-6
    @test r.sunlit + r.shaded ≈ _ls_simpson(tot, 0.0, L) rtol = 1e-6
end

@testset "Big-leaf longwave: conservation, Kirchhoff" begin
    σSB = Aeolus.SurfaceExchange.DEFAULT_SURFACE_CONSTANTS.σ_SB
    r = bigleaf_longwave(250.0, 300.0, 265.0, 3.0; ε_ground = 0.96, σ_SB = σSB)
    @test r.canopy + r.ground + r.upward ≈ 250.0 atol = 1e-10
    r = bigleaf_longwave(σSB * 280.0^4, 280.0, 280.0, 4.0; ε_ground = 0.9, σ_SB = σSB)
    @test abs(r.canopy) < 1e-10 && abs(r.ground) < 1e-10
end

@testset "Canopy aerodynamics" begin
    a = canopy_roughness(20.0, 5.0, 0.055, 0.67, 0.01)
    @test a.z0m ≈ 1.1 rtol = 1e-13
    @test a.d ≈ 13.4 rtol = 1e-13
    u = 0.4
    @test undercanopy_transfer_coefficient(u, 50.0, 0.01; T_ac = 290.0, T_g = 295.0, z_top = 20.0) ≈ 0.004 rtol = 1e-12
    S = 9.80665 * 20 * 2 / (292 * u^2)
    @test undercanopy_transfer_coefficient(u, 50.0, 0.01; T_ac = 292.0, T_g = 290.0, z_top = 20.0) ≈
          0.004 / (1 + 0.5 * min(S, 10)) rtol = 1e-12
end

# ---------------------------------------------------------------------------
# The coupled column under an idealised clear-sky diurnal cycle
# ---------------------------------------------------------------------------

function _ls_column(; ψ = -3.37, pft = :bdt_temperate, LAI = 4.0, SAI = 0.5,
                    z_top = 20.0, T = 292.0, hydrology = true, model = :medlyn)
    g = clm5_soil_grid(13)
    w = SoilWaterColumn(g, clapp_hornberger_texture(:loam); ψ = ψ,
                        ice_impedance = PowerLawIceImpedance())
    soil = SoilThermalColumn(w, clm_soil_thermal_properties(40.0, 20.0); T_soil = T,
                             hydrology = hydrology)
    return VegetatedLandColumn(soil, pft_vegetation(pft; model = model); LAI = LAI, SAI = SAI, z_top = z_top)
end

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
        rain = rain_at !== nothing && rain_at(t) ? 2e-3 : 0.0
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
    worst_col = worst_leaf = worst_sw = worst_part = 0.0
    nconv = nsteps = 0
    ΣG = 0.0
    _ls_run!(s, 2; rain_at = t -> 26 <= t < 28, record = (t, f, fl, d) -> begin
        worst_col = max(worst_col, abs(fl.Rn - fl.H - fl.LE - fl.G))
        worst_leaf = max(worst_leaf, abs(d.Rn_v - d.H_v - d.LE_v))
        worst_sw = max(worst_sw, abs(d.SW_v + d.SW_g + d.SW_up - f.sw_down))
        worst_part = max(worst_part, abs(fl.H - d.H_v - d.H_g), abs(fl.LE - d.LE_v - d.LE_g))
        ΣG += fl.G * 1800.0
        nconv += d.converged; nsteps += 1
    end)
    @test worst_col < 1e-8
    @test worst_leaf < 1e-8
    @test worst_sw < 1e-10
    @test worst_part < 1e-9
    @test abs(ΣG - (soil.budget.surface - G0)) < 1e-6 * abs(ΣG) + 1e-3
    @test nconv == nsteps
    # water: ΔS = rain + dew − transpiration − evaporation − runoff − drainage
    wb = soil.water.budget
    ΔS = soil_water_storage(soil.water) - S0
    @test wb.extraction - b0.extraction ≈ s.budget.transpiration / ρw rtol = 1e-12
    inflow = (s.budget.rain + s.budget.dew) / ρw
    outflow = s.budget.transpiration / ρw + (wb.evaporation - b0.evaporation) +
              (wb.runoff - b0.runoff) + (wb.drainage - b0.drainage)
    @test abs(ΔS - (inflow - outflow)) < 1e-12
    @test s.budget.transpiration > 0
end

@testset "Vegetated column: limits" begin
    # bare column: no canopy fluxes
    s = _ls_column(LAI = 0.0, SAI = 0.0)
    ok = true
    _ls_run!(s, 0.5; record = (t, f, fl, d) -> (ok &= d.H_v == 0 && d.LE_v == 0 && d.GPP == 0 &&
                                                       abs(fl.Rn - fl.H - fl.LE - fl.G) < 1e-9))
    @test ok
    # isothermal column: Rn = ε_eff (L↓ − σT⁴), ε_eff = 1 − (1 − ε_g) e^{−2(L+S)}
    s = _ls_column()
    σSB = s.soil.water.constants.σ_SB
    T = s.soil.T_skin
    s.T_v = T
    εeff = surface_emissivity(s)
    @test εeff ≈ 1 - (1 - s.soil.emissivity) * exp(-2 * (s.LAI + s.SAI)) rtol = 1e-14
    @test net_radiation(s, SurfaceForcing(sw_down = 0.0, lw_down = 300.0, T_air = T)) ≈
          εeff * (300.0 - σSB * T^4) atol = 1e-10
end

@testset "Soil-water stress: Bowen ratio response to drought" begin
    # Midday of one clear day at fixed soil water: field capacity vs beyond
    # ψ_c of the broadleaf deciduous tree (−224 m).
    rows = map((-3.37, -260.0)) do ψ
        s = _ls_column(ψ = ψ, hydrology = false)
        mid = Ref{Any}(nothing)
        _ls_run!(s, 0.6; record = (t, f, fl, d) -> (t == 12.0 && (mid[] = (fl, d))))
        fl, d = mid[]
        (βt = d.β_t, LE = fl.LE, B = fl.H / fl.LE, Et = d.E_t, GPP = d.GPP)
    end
    @test rows[1].βt ≈ 1 && rows[2].βt == 0 && rows[2].Et == 0
    @test rows[2].LE < rows[1].LE && rows[2].GPP < rows[1].GPP
    @test rows[2].B > 5 * rows[1].B
end

end # LandSurface
