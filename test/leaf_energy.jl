# Era V, Phase 48 — PlantCanopy (issue #24): leaf energy balance and leaf
# temperature. Leaf boundary layer (Campbell & Norman 1998 / CLM5),
# transpiration through Photosynthesis (#20), sensible heat, the
# Penman–Monteith limit, prognostic leaf temperature, and the coupled
# sunlit/shaded canopy solution with the #23 radiation.

using Test
using Aeolus
using Aeolus.PlantCanopy
using Aeolus.Photosynthesis
using Aeolus.MoistThermodynamics: saturation_vapor_pressure, latent_heat_vapor, DEFAULT_CONSTANTS
using Aeolus.RadiationCore: STEFAN_BOLTZMANN

const _σ = STEFAN_BOLTZMANN
const _cpm = DEFAULT_CONSTANTS.cp_dry * DEFAULT_CONSTANTS.molmass_dry
_λm(T) = latent_heat_vapor(T) * DEFAULT_CONSTANTS.molmass_water

# Plain bisection, independent of the module's Illinois solver.
function _bisect(f, a, b; n = 200)
    fa = f(a)
    for _ in 1:n
        m = (a + b) / 2; fm = f(m)
        (fm == 0 || b - a < 1e-13) && return m
        sign(fm) == sign(fa) ? (a = m; fa = fm) : (b = m)
    end
    return (a + b) / 2
end

@testset "Era V — PlantCanopy: leaf energy balance (#24)" begin

@testset "leaf boundary layer" begin
    # Campbell & Norman (1998) ch. 7: forced convection, both sides for heat
    bl = CampbellNormanBoundaryLayer(leaf_width = 0.05, free_convection = false)
    d = 0.72 * 0.05
    gH, gv = leaf_boundary_layer_conductances(bl, 2.0, 300.0, 298.0, 101325.0)
    @test gH ≈ 2 * 1.4 * 0.135 * sqrt(2.0 / d) rtol = 1e-14
    @test gv ≈ 1.4 * 0.147 * sqrt(2.0 / d) rtol = 1e-14
    # free convection in still air, (ΔT/d)^¼
    blf = CampbellNormanBoundaryLayer(leaf_width = 0.05)
    @test leaf_boundary_layer_conductances(blf, 0.0, 303.0, 298.0, 1e5)[1] ≈ 2 * 0.05 * (5.0 / d)^0.25 rtol = 1e-14
    # CLM delegates to Photosynthesis (one source of truth)
    gH, gv = leaf_boundary_layer_conductances(CLMLeafBoundaryLayer(), 1.0, 300.0, 293.15, 101325.0)
    @test gH == gv == leaf_boundary_layer_conductance(1.0, 293.15, 101325.0)
end

@testset "single leaf: energy closure and analytic limits" begin
    Ta = 298.15
    worst = 0.0; worst_HL = 0.0; allconv = true
    p = pft_photosynthesis_parameters(:bdt_temperate)
    for s in (0.0, 0.4, CoupledLeafPhysiology(p, pft_stomatal_model(:bdt_temperate))),
        bl in (CampbellNormanBoundaryLayer(), CLMLeafBoundaryLayer()),
        (U, Rsw, RH, T) in ((0.3, 600.0, 0.3, 303.15), (2.0, 0.0, 0.9, 283.15))
        ea = RH * saturation_vapor_pressure(T)
        R = Rsw + 0.98 * (330.0 + _σ * T^4)
        r = leaf_energy_balance(s, bl; R_abs = R, T_air = T, e_air = ea, U = U, Q = 2.0 * Rsw)
        allconv &= r.converged
        worst = max(worst, abs(r.R_abs - r.emitted - r.H - r.LE - r.storage) / R)
        worst_HL = max(worst_HL, abs(r.H - _cpm * r.g_H * (r.T_leaf - T)), abs(r.LE - _λm(r.T_leaf) * r.E))
    end
    @test allconv
    @test worst < 1e-12
    @test worst_HL < 1e-9
    # isothermal equilibrium: zero isothermal net radiation, saturated air
    r = leaf_energy_balance(0.3, CampbellNormanBoundaryLayer(); R_abs = 1.96 * _σ * Ta^4, T_air = Ta,
                            e_air = saturation_vapor_pressure(Ta), U = 1.0)
    @test abs(r.T_leaf - Ta) < 1e-9
    # psychrometer: no radiation, open stomata, g_H = g_v:
    # c_p (T_a − T) = λ(T)(e_s(T) − e_a)/p, root by bisection
    ea = 0.4 * saturation_vapor_pressure(Ta)
    rw = leaf_energy_balance(1e9, CLMLeafBoundaryLayer(); R_abs = 0.0, emission = 0.0,
                             T_air = Ta, e_air = ea, U = 1.0)
    Twb = _bisect(T -> _cpm * (Ta - T) - _λm(T) * (saturation_vapor_pressure(T) - ea) / 101325.0,
                  250.0, Ta)
    @test rw.T_leaf ≈ Twb atol = 1e-6
    # still air, sunlit thin leaf: 6–10 K above air (Ansari & Loomis 1959,
    # via Leuzinger & Körner 2007); forcing assumed, measured 8.20 K
    r = leaf_energy_balance(0.3, CampbellNormanBoundaryLayer(leaf_width = 0.05);
                            R_abs = 500 + 0.98 * (350 + _σ * Ta^4), T_air = Ta,
                            e_air = saturation_vapor_pressure(Ta) - 2000.0, U = 0.0)
    @test 6 <= r.T_leaf - Ta <= 10
end

@testset "Penman–Monteith limit" begin
    # Fixed g_s, CLM layer: the exact balance approaches the linearised one
    # quadratically as the forcing is scaled down (two levels).
    Ta = 298.15; εe = 1.96; bl = CLMLeafBoundaryLayer()
    gH, gv = leaf_boundary_layer_conductances(bl, 1.0, Ta, Ta, 101325.0)
    R0, D0, gs = 300.0, 1000.0, 0.25
    errs = map((1.0, 0.5)) do ε
        R = εe * _σ * Ta^4 + ε * R0; ea = saturation_vapor_pressure(Ta) - ε * D0
        r = leaf_energy_balance(gs, bl; R_abs = R, T_air = Ta, e_air = ea, U = 1.0)
        pm = leaf_penman_monteith(; R_abs = R, T_air = Ta, e_air = ea, g_H = gH, g_s = gs, g_v = gv)
        abs(r.T_leaf - pm.T_leaf)
    end
    @test 3.8 < errs[1] / errs[2] < 4.2                      # O(ΔT²)
    @test errs[1] < 0.05
end

@testset "prognostic leaf temperature" begin
    Ta = 298.15; bl = CLMLeafBoundaryLayer(); gs = 0.25; ea = 1800.0; R = 1100.0; C = 1300.0
    kw = (; R_abs = R, T_air = Ta, e_air = ea, U = 1.0)
    rs = leaf_energy_balance(gs, bl; kw...)
    @test leaf_energy_balance(gs, bl; kw..., T_prev = 280.0, heat_capacity = C, dt = Inf).T_leaf == rs.T_leaf
    # backward-Euler relaxation: δ decays by 1/(1 + Δt/τ), τ = C/(−∂F/∂T)
    Tl = rs.T_leaf
    gH, gv = leaf_boundary_layer_conductances(bl, 1.0, Tl, Ta, 101325.0)
    gw = gs * gv / (gs + gv)
    s = _λm(Tl) / DEFAULT_CONSTANTS.molmass_water * saturation_vapor_pressure(Tl) /
        (DEFAULT_CONSTANTS.gas_constant / DEFAULT_CONSTANTS.molmass_water * Tl^2)
    dLE = gw / 101325.0 * (_λm(Tl) * s +
          (_λm(Tl + 1e-4) - _λm(Tl - 1e-4)) / 2e-4 * (saturation_vapor_pressure(Tl) - ea))
    τ = C / (_cpm * gH + 4 * 1.96 * _σ * Tl^3 + dLE)
    r = leaf_energy_balance(gs, bl; kw..., T_prev = Tl + 1e-3, heat_capacity = C, dt = τ)
    @test (r.T_leaf - Tl) / 1e-3 ≈ 1 / 2 rtol = 1e-3
    # stored energy = time-integrated imbalance, from far away
    T = 273.15; stored = 0.0; imbalance = 0.0
    for n in 1:20
        r = leaf_energy_balance(gs, bl; kw..., T_prev = T, heat_capacity = C, dt = 5τ)
        stored += C * (r.T_leaf - T)
        imbalance += (r.R_abs - r.emitted - r.H - r.LE) * 5τ
        T = r.T_leaf
    end
    @test stored ≈ imbalance rtol = 1e-10
end

@testset "coupled photosynthesis–stomata" begin
    Ta = 298.15; ea = saturation_vapor_pressure(Ta) - 1500.0
    p = pft_photosynthesis_parameters(:bdt_temperate)
    m = pft_stomatal_model(:bdt_temperate)
    bl = CampbellNormanBoundaryLayer()
    r = leaf_energy_balance(CoupledLeafPhysiology(p, m; ca = 400.0), bl;
                            R_abs = 450 + 0.98 * (350 + _σ * Ta^4), T_air = Ta,
                            e_air = ea, U = 1.0, Q = 4.6 * 250)
    # delegation: gs, E, A_n are Photosynthesis' at the leaf temperature
    ge = leaf_gas_exchange(p, m; ca = 400.0, Q = 4.6 * 250, T_leaf = r.T_leaf, e_air = ea, gb = r.g_v)
    @test r.gs ≈ ge.gs rtol = 1e-12
    @test r.An ≈ ge.An rtol = 1e-12
    # drought (β = 0) closes the stomata and heats the leaf
    rd = leaf_energy_balance(CoupledLeafPhysiology(p, m; β = 0.0), bl; R_abs = r.R_abs,
                             T_air = Ta, e_air = ea, U = 1.0, Q = 4.6 * 250)
    @test rd.T_leaf > r.T_leaf + 1 && rd.LE < 0.05 * r.LE
end

# ---- resolved canopy: sunlit/shaded leaves of every layer -------------------

function _forest(; LAI = 5.0, h = 30.0, N = 30, μ = 0.888, Sdir = 780.0, Sdif = 120.0,
                 leaves = SphericalLeaves())
    prof = LalicMihailovicLAD(LAI, h; z_m = 0.7h)
    zf = collect(range(0, h, length = N + 1))
    lad = layer_leaf_area_density(prof, zf)
    par = canopy_shortwave(zf, lad; sun = μ, sw_direct = 0.45Sdir, sw_diffuse = 0.45Sdif,
                           optics = LeafOptics(0.10, 0.05), albedo_direct = 0.10, leaves = leaves)
    nir = canopy_shortwave(zf, lad; sun = μ, sw_direct = 0.55Sdir, sw_diffuse = 0.55Sdif,
                           optics = LeafOptics(0.45, 0.25), albedo_direct = 0.20, leaves = leaves)
    zc = (zf[1:end-1] .+ zf[2:end]) ./ 2
    return zf, lad, (par, nir), zc
end
_wind(zc, h, Uh) = [canopy_exponential_wind(z; h = h, u_h = Uh, C_d = 0.2, a = 5 / 15) for z in zc]

@testset "canopy: energy closure and consistency with the single leaf" begin
    Ta = 298.15; ea = saturation_vapor_pressure(Ta) - 1500.0
    p = pft_photosynthesis_parameters(:bdt_temperate)
    stom = CoupledLeafPhysiology(p, pft_stomatal_model(:bdt_temperate))
    bl = CampbellNormanBoundaryLayer(leaf_width = 0.045)
    leaves = RossGoudriaanLeaves(0.3)
    zf, lad, sw, zc = _forest(; N = 12, leaves = leaves)
    U = _wind(zc, 30.0, 1.5)
    T_air = Ta .- 0.02 .* (30 .- zc)             # slightly stable in-canopy air
    s = canopy_leaf_energy_balance(zf, lad; shortwave = sw, par = 1, T_air = T_air,
                                   e_air = ea, U = U, lw_down = 350.0, T_ground = 296.0,
                                   stomata = stom, boundary_layer = bl, leaves = leaves)
    @test s.converged
    @test s.residual < 1e-6
    sw_in = sum(b.incident - b.reflected - b.ground_absorbed for b in sw)
    lw_in = 350.0 - s.longwave.outgoing - s.longwave.ground_net
    @test abs(sw_in + lw_in - (s.H_total + s.LE_total + sum(s.storage))) /
          (sum(b.incident for b in sw) + 350.0) < 1e-10
    Δz = diff(zf)
    @test sum(s.heat_source .* Δz) ≈ s.H_total rtol = 1e-12
    # every class is the single-leaf balance with the final longwave
    lw = s.longwave
    worst_leaf = 0.0
    for k in eachindex(lad)
        dΛ = lad[k] * Δz[k]
        dΛ > 0 || continue
        εl = lw.emissivity_layer[k]
        Teff4 = (s.lai_sunlit[k] * s.T_sunlit[k]^4 + s.lai_shaded[k] * s.T_shaded[k]^4) / dΛ
        L_abs = (lw.net_absorbed[k] + 2εl * _σ * Teff4) / dΛ
        for (la, T, ab) in ((s.lai_sunlit[k], s.T_sunlit[k], :absorbed_sunlit),
                            (s.lai_shaded[k], s.T_shaded[k], :absorbed_shaded))
            la > 0 || continue
            q = sum(getfield(b, ab)[k] for b in sw) / la
            Qp = 4.6 * getfield(sw[1], ab)[k] / la
            r = leaf_energy_balance(stom, bl; R_abs = q + L_abs, emission = 2εl / dΛ,
                                    T_air = T_air[k], e_air = ea, U = U[k], Q = Qp)
            worst_leaf = max(worst_leaf, abs(r.T_leaf - T))
        end
    end
    @test worst_leaf < 1e-7
    @test all(s.T_sunlit[k] > s.T_shaded[k] for k in eachindex(lad) if s.lai_sunlit[k] > 1e-6)
end

@testset "canopy: isothermal limit, prognostic march" begin
    Ta = 290.0
    zf, lad, sw, zc = _forest(; N = 12)
    s = canopy_leaf_energy_balance(zf, lad; shortwave = nothing, T_air = Ta,
                                   e_air = saturation_vapor_pressure(Ta), U = 1.0,
                                   lw_down = _σ * Ta^4, T_ground = Ta, ε_ground = 1.0,
                                   stomata = 0.05)
    @test maximum(abs.(s.T_shaded .- Ta)) < 1e-9
    kw = (; shortwave = sw, T_air = 298.15, e_air = 1700.0, U = _wind(zc, 30.0, 1.0),
          lw_down = 350.0, T_ground = 298.15, stomata = 0.2)
    ss = canopy_leaf_energy_balance(zf, lad; kw...)
    prev = (sunlit = fill(290.0, length(lad)), shaded = fill(290.0, length(lad)))
    stored = 0.0; imbalance = 0.0; dt = 600.0
    for n in 1:20
        s = canopy_leaf_energy_balance(zf, lad; kw..., T_prev = prev, heat_capacity = 1300.0, dt = dt)
        stored += sum(s.storage) * dt
        imbalance += (sum(s.sw_absorbed) + sum(s.lw_net) - s.H_total - s.LE_total) * dt
        prev = s
    end
    @test stored ≈ imbalance rtol = 1e-8
    @test maximum(abs.(prev.T_sunlit .- ss.T_sunlit)) < 1e-6
end

@testset "canopy ΔT vs Leuzinger & Körner (2007), Swiss Canopy Crane" begin
    # Thermal-camera canopy temperatures of a 35 m mixed deciduous forest,
    # 18 July 2004 late morning, T_air = 25 °C, VPD = 19.7 hPa (their
    # Table 1): broad-leaved mean leaf-to-air ΔT 1.7–4.8 K, T_max − T_air
    # 6.4–12.0 K. Radiation, wind and canopy structure ASSUMED (clear sky,
    # LAI 5, h = 30 m, U_h = 1 m/s); measured 2.57 K / 8.15 K at N = 30.
    Ta = 298.15; ea = saturation_vapor_pressure(Ta) - 1970.0
    zf, lad, sw, zc = _forest()
    p = pft_photosynthesis_parameters(:bdt_temperate); m = pft_stomatal_model(:bdt_temperate)
    s = canopy_leaf_energy_balance(zf, lad; shortwave = sw, par = 1, T_air = Ta, e_air = ea,
                                   U = _wind(zc, 30.0, 1.0), lw_down = 350.0, T_ground = Ta,
                                   stomata = CoupledLeafPhysiology(p, m),
                                   boundary_layer = CampbellNormanBoundaryLayer(leaf_width = 0.045))
    Λ = s.Λ_faces; Tv = 0.0; W = 0.0
    for k in eachindex(lad)
        dΛ = Λ[k] - Λ[k+1]
        dΛ > 0 || continue
        w = exp(-0.5Λ[k+1]) * (1 - exp(-0.5dΛ)) / 0.5
        fs = s.lai_sunlit[k] / dΛ
        Tv += w * (fs * s.T_sunlit[k] + (1 - fs) * s.T_shaded[k]); W += w
    end
    @test 1.7 <= Tv / W - Ta <= 4.8
    @test 6.4 <= maximum(s.T_sunlit) - Ta <= 12.0
end

end # testset
