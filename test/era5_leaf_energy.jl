# Era V, Phase 48 — PlantCanopy (issue #24): leaf energy balance and leaf
# temperature. Leaf boundary layer (Campbell & Norman 1998 / CLM5),
# transpiration through Photosynthesis (#20), sensible heat, the
# Penman–Monteith limit, prognostic leaf temperature, and the coupled
# sunlit/shaded canopy solution with the #23 radiation.

using Test, Printf
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
    bl = CampbellNormanBoundaryLayer(leaf_width = 0.05, free_convection = false)
    d = 0.72 * 0.05
    for U in (0.1, 0.5, 1.0, 2.0, 5.0)
        gH, gv = leaf_boundary_layer_conductances(bl, U, 300.0, 298.0, 101325.0)
        @test gH ≈ 2 * 1.4 * 0.135 * sqrt(U / d) rtol = 1e-14      # C&N eq. 7.30, both sides
        @test gv ≈ 1.4 * 0.147 * sqrt(U / d) rtol = 1e-14          # eq. 7.33, one stomatal side
    end
    # √U scaling of forced convection
    g1 = leaf_boundary_layer_conductances(bl, 1.0, 300.0, 298.0, 1e5)[1]
    g4 = leaf_boundary_layer_conductances(bl, 4.0, 300.0, 298.0, 1e5)[1]
    @test g4 / g1 ≈ 2 rtol = 1e-14
    # amphistomatous leaves double the vapour path, not the heat path
    bl2 = CampbellNormanBoundaryLayer(leaf_width = 0.05, free_convection = false, stomatal_sides = 2)
    @test leaf_boundary_layer_conductances(bl2, 1.0, 300.0, 298.0, 1e5)[2] ≈
          2 * leaf_boundary_layer_conductances(bl, 1.0, 300.0, 298.0, 1e5)[2]
    @test leaf_boundary_layer_conductances(bl2, 1.0, 300.0, 298.0, 1e5)[1] ==
          leaf_boundary_layer_conductances(bl, 1.0, 300.0, 298.0, 1e5)[1]
    # free convection: the floor in still air, (ΔT/d)^¼ (C&N eq. 7.36)
    blf = CampbellNormanBoundaryLayer(leaf_width = 0.05)
    for ΔT in (1.0, 5.0, 16.0)
        gH, gv = leaf_boundary_layer_conductances(blf, 0.0, 298.0 + ΔT, 298.0, 1e5)
        @test gH ≈ 2 * 0.05 * (ΔT / d)^0.25 rtol = 1e-14
        @test gv ≈ 0.055 * (ΔT / d)^0.25 rtol = 1e-14
    end
    @test leaf_boundary_layer_conductances(blf, 0.0, 298.0, 298.0, 1e5) == (0.0, 0.0)
    # forced convection wins at ordinary wind speeds
    @test leaf_boundary_layer_conductances(blf, 1.0, 303.0, 298.0, 1e5)[1] ≈ g1
    # CLM delegates to Photosynthesis (one source of truth): 20 s m⁻¹ at 1 m s⁻¹, 20 °C
    clm = CLMLeafBoundaryLayer()
    gH, gv = leaf_boundary_layer_conductances(clm, 1.0, 300.0, 293.15, 101325.0)
    @test gH == gv == leaf_boundary_layer_conductance(1.0, 293.15, 101325.0)
    @test gH ≈ 2.0786 rtol = 1e-4
    # Two independent parameterisations of the same leaf (d = 4 cm, U = 1 m/s):
    # C&N outdoor forced convection (both sides) vs CLM's Cv√(U/d).
    cn = CampbellNormanBoundaryLayer(leaf_width = 0.04 / 0.72, free_convection = false)
    gcn = leaf_boundary_layer_conductances(cn, 1.0, 293.15, 293.15, 101325.0)[1]
    @printf("  C&N vs CLM leaf heat conductance, d = 4 cm, U = 1 m/s: %.4f vs %.4f mol m⁻² s⁻¹ (ratio %.3f)\n",
            gcn, gH, gcn / gH)
    @test 0.85 < gcn / gH < 1.15
    @test_throws ArgumentError CampbellNormanBoundaryLayer(leaf_width = 0)
    @test_throws ArgumentError CampbellNormanBoundaryLayer(stomatal_sides = 3)
end

@testset "single leaf: energy closure and analytic limits" begin
    Ta = 298.15
    worst = 0.0
    p = pft_photosynthesis_parameters(:bdt_temperate)
    stom = (0.0, 0.1, 0.4, CoupledLeafPhysiology(p, pft_stomatal_model(:bdt_temperate)),
            CoupledLeafPhysiology(p, pft_stomatal_model(:bdt_temperate; model = :ball_berry)))
    for s in stom, bl in (CampbellNormanBoundaryLayer(), CLMLeafBoundaryLayer()),
        U in (0.0, 0.3, 2.0), Rsw in (0.0, 200.0, 600.0), RH in (0.3, 0.9), T in (283.15, 303.15)
        (bl isa CLMLeafBoundaryLayer && U == 0) && continue
        ea = RH * saturation_vapor_pressure(T)
        R = Rsw + 0.98 * (330.0 + _σ * T^4)
        r = leaf_energy_balance(s, bl; R_abs = R, T_air = T, e_air = ea, U = U, Q = 2.0 * Rsw)
        @test r.converged
        worst = max(worst, abs(r.R_abs - r.emitted - r.H - r.LE - r.storage) / R)
        @test r.H ≈ _cpm * r.g_H * (r.T_leaf - T) rtol = 1e-12 atol = 1e-12
        @test r.LE ≈ _λm(r.T_leaf) * r.E rtol = 1e-12 atol = 1e-12
    end
    @printf("  single-leaf energy closure, worst |R_abs − εσT⁴ − H − LE|/R_abs = %.2e\n", worst)
    @test worst < 1e-12

    # Isothermal equilibrium: isothermal net radiation zero, saturated air
    # → the leaf sits exactly at air temperature, H = LE = 0.
    for bl in (CampbellNormanBoundaryLayer(), CLMLeafBoundaryLayer()), gs in (0.0, 0.3)
        r = leaf_energy_balance(gs, bl; R_abs = 1.96 * _σ * Ta^4, T_air = Ta,
                                e_air = saturation_vapor_pressure(Ta), U = 1.0)
        @test abs(r.T_leaf - Ta) < 1e-9
        @test abs(r.H) < 1e-7 && abs(r.LE) < 1e-7
    end
    # Perfect coupling: U → ∞ drives the leaf to air temperature.
    r = leaf_energy_balance(0.3, CampbellNormanBoundaryLayer(); R_abs = 900.0, T_air = Ta,
                            e_air = 1500.0, U = 1e8)
    @test abs(r.T_leaf - Ta) < 1e-3
    # Closed stomata: no transpiration, all the load goes into H and emission,
    # and the leaf is warmer than the transpiring one.
    r0 = leaf_energy_balance(0.0; R_abs = 1000.0, T_air = Ta, e_air = 1500.0, U = 1.0)
    r1 = leaf_energy_balance(0.3; R_abs = 1000.0, T_air = Ta, e_air = 1500.0, U = 1.0)
    @test r0.E == 0 && r0.LE == 0
    @test r0.T_leaf > r1.T_leaf + 1
    # No radiation exchange, wide-open stomata, CLM layer (g_H = g_v):
    # the leaf is a psychrometer and sits at the root of
    # c_p (T_a − T) = λ(T) (e_s(T) − e_a)/p, found here by bisection.
    ea = 0.4 * saturation_vapor_pressure(Ta)
    rw = leaf_energy_balance(1e9, CLMLeafBoundaryLayer(); R_abs = 0.0, emission = 0.0,
                             T_air = Ta, e_air = ea, U = 1.0)
    Twb = _bisect(T -> _cpm * (Ta - T) - _λm(T) * (saturation_vapor_pressure(T) - ea) / 101325.0,
                  250.0, Ta)
    @printf("  psychrometric limit: T_leaf − T_a = %.4f K, bisection %.4f K\n", rw.T_leaf - Ta, Twb - Ta)
    @test rw.T_leaf ≈ Twb atol = 1e-6
end

@testset "Penman–Monteith limit" begin
    # With a fixed stomatal conductance and a CLM boundary layer (no free
    # convection, so g_H is constant) the only nonlinearities are σT⁴,
    # e_s(T) and λ(T): the exact balance must approach the linearised one
    # quadratically as the forcing (R_ni, D) is scaled down.
    Ta = 298.15; εe = 1.96; bl = CLMLeafBoundaryLayer()
    gH, gv = leaf_boundary_layer_conductances(bl, 1.0, Ta, Ta, 101325.0)
    for (R0, D0, gs) in ((300.0, 1000.0, 0.25), (-80.0, 500.0, 0.05), (500.0, 2500.0, 0.6))
        errs = Float64[]
        for ε in (1.0, 0.5, 0.25, 0.125)
            R = εe * _σ * Ta^4 + ε * R0; ea = saturation_vapor_pressure(Ta) - ε * D0
            r = leaf_energy_balance(gs, bl; R_abs = R, T_air = Ta, e_air = ea, U = 1.0)
            pm = leaf_penman_monteith(; R_abs = R, T_air = Ta, e_air = ea, g_H = gH, g_s = gs, g_v = gv)
            # the linearised balance closes exactly with linearised emission
            @test pm.R_ni - _cpm * pm.g_r * pm.ΔT ≈ pm.H + pm.LE rtol = 1e-12
            push!(errs, abs(r.T_leaf - pm.T_leaf))
            ε == 1.0 && @printf("  PM (R_ni=%.0f W m⁻², D=%.0f Pa, g_s=%.2f): ΔT exact %.4f K, PM %.4f K, LE %.1f vs %.1f W m⁻²\n",
                                R0, D0, gs, r.T_leaf - Ta, pm.ΔT, r.LE, pm.LE)
        end
        ratios = errs[1:end-1] ./ errs[2:end]
        @printf("    |ΔT_exact − ΔT_PM| = %s K; halving ratios %s\n",
                join((@sprintf("%.2e", e) for e in errs), ", "), join((@sprintf("%.3f", q) for q in ratios), ", "))
        @test all(q -> 3.8 < q < 4.2, ratios)                      # O(ΔT²)
        @test errs[1] < 0.05                                      # < 0.05 K at full forcing
    end
    # Closed stomata limit of PM: ΔT = R_ni / (c_p (g_H + g_r))
    pm = leaf_penman_monteith(; R_abs = 1000.0, T_air = Ta, e_air = 1000.0, g_H = gH, g_s = 0.0, g_v = gv)
    @test pm.ΔT ≈ pm.R_ni / (_cpm * (gH + pm.g_r)) rtol = 1e-14
    @test pm.LE == 0
    # s is the derivative of MoistThermodynamics' e_s
    h = 1e-3
    @test pm.s ≈ (saturation_vapor_pressure(Ta + h) - saturation_vapor_pressure(Ta - h)) / (2h) rtol = 1e-6
end

@testset "prognostic leaf temperature" begin
    Ta = 298.15; bl = CLMLeafBoundaryLayer(); gs = 0.25; ea = 1800.0; R = 1100.0; C = 1300.0
    kw = (; R_abs = R, T_air = Ta, e_air = ea, U = 1.0)
    rs = leaf_energy_balance(gs, bl; kw...)
    # Δt → ∞ (or C = 0) is the diagnostic balance
    @test leaf_energy_balance(gs, bl; kw..., T_prev = 280.0, heat_capacity = C, dt = Inf).T_leaf == rs.T_leaf
    @test leaf_energy_balance(gs, bl; kw..., T_prev = 280.0, heat_capacity = C, dt = 1e12).T_leaf ≈ rs.T_leaf atol = 1e-8
    @test_throws ArgumentError leaf_energy_balance(gs, bl; kw..., heat_capacity = C, dt = 10.0)
    # Linear relaxation: a small departure δ from the steady state decays by
    # 1/(1 + Δt/τ) per backward-Euler step, τ = C / (−∂F/∂T) with the
    # Penman–Monteith derivative c_p (g_H + g_r) + λ g_w s / p at T_leaf.
    Tl = rs.T_leaf
    gH, gv = leaf_boundary_layer_conductances(bl, 1.0, Tl, Ta, 101325.0)
    gw = gs * gv / (gs + gv)
    s = _λm(Tl) / DEFAULT_CONSTANTS.molmass_water * saturation_vapor_pressure(Tl) /
        (DEFAULT_CONSTANTS.gas_constant / DEFAULT_CONSTANTS.molmass_water * Tl^2)
    dLE = gw / 101325.0 * (_λm(Tl) * s +
          (_λm(Tl + 1e-4) - _λm(Tl - 1e-4)) / 2e-4 * (saturation_vapor_pressure(Tl) - ea))
    k = _cpm * gH + 4 * 1.96 * _σ * Tl^3 + dLE
    τ = C / k
    @printf("  leaf thermal time constant: τ = %.2f s (C = %.0f J m⁻² K⁻¹)\n", τ, C)
    for dt in (0.1τ, τ, 10τ)
        δ = 1e-3
        r = leaf_energy_balance(gs, bl; kw..., T_prev = Tl + δ, heat_capacity = C, dt = dt)
        @test (r.T_leaf - Tl) / δ ≈ 1 / (1 + dt / τ) rtol = 1e-3
        @test r.storage ≈ C * (r.T_leaf - (Tl + δ)) / dt rtol = 1e-12
    end
    # Unconditional stability and monotone approach from far away, any Δt;
    # the stored energy equals the time-integrated imbalance.
    for dt in (0.5τ, 5τ, 500τ)
        T = 273.15; stored = 0.0; imbalance = 0.0; mono = true
        for n in 1:60
            r = leaf_energy_balance(gs, bl; kw..., T_prev = T, heat_capacity = C, dt = dt)
            mono &= (T - 1e-9 <= r.T_leaf <= Tl + 1e-9)
            stored += C * (r.T_leaf - T)
            imbalance += (r.R_abs - r.emitted - r.H - r.LE) * dt
            T = r.T_leaf
        end
        @test mono
        @test stored ≈ imbalance rtol = 1e-10
        dt >= 5τ && @test abs(T - Tl) < 1e-6
    end
end

@testset "coupled photosynthesis–stomata" begin
    Ta = 298.15; ea = saturation_vapor_pressure(Ta) - 1500.0
    p = pft_photosynthesis_parameters(:bdt_temperate)
    for model in (:medlyn, :ball_berry)
        m = pft_stomatal_model(:bdt_temperate; model = model)
        st = CoupledLeafPhysiology(p, m; ca = 400.0)
        bl = CampbellNormanBoundaryLayer()
        r = leaf_energy_balance(st, bl; R_abs = 450 + 0.98 * (350 + _σ * Ta^4), T_air = Ta,
                                e_air = ea, U = 1.0, Q = 4.6 * 250)
        @test r.converged
        # delegation: gs, E and A_n are Photosynthesis' at the leaf temperature
        ge = leaf_gas_exchange(p, m; ca = 400.0, Q = 4.6 * 250, T_leaf = r.T_leaf, e_air = ea,
                               gb = r.g_v)
        @test r.gs ≈ ge.gs rtol = 1e-12
        @test r.E ≈ ge.E rtol = 1e-12
        @test r.LE ≈ ge.LE rtol = 1e-10
        @test r.An ≈ ge.An rtol = 1e-12
        # drought (β = 0) closes the stomata and heats the leaf
        rd = leaf_energy_balance(CoupledLeafPhysiology(p, m; β = 0.0), bl; R_abs = r.R_abs,
                                 T_air = Ta, e_air = ea, U = 1.0, Q = 4.6 * 250)
        @printf("  %s: sunlit leaf ΔT = %.2f K (g_s = %.3f), β = 0: ΔT = %.2f K (g_s = %.1e)\n",
                model, r.T_leaf - Ta, r.gs, rd.T_leaf - Ta, rd.gs)
        @test rd.T_leaf > r.T_leaf + 1
        @test rd.LE < 0.05 * r.LE
    end
end

@testset "observed leaf-to-air ΔT (Ansari & Loomis 1959, via Leuzinger & Körner 2007)" begin
    # Leuzinger & Körner (2007, Agric. For. Meteorol. 146, §1) cite the rule
    # of thumb of Ansari & Loomis (1959, thermocouple survey): sunlit thin
    # leaves are 6–10 K above air temperature in still air. Forcing assumed
    # here (not measured by them): a horizontal leaf under a 1000 W m⁻²
    # overhead sun, shortwave absorptance 0.5, sky 350 W m⁻², ground at air
    # temperature, 25 °C, VPD 2 kPa, 5 cm leaf, g_s 0.2–0.4 mol m⁻² s⁻¹.
    Ta = 298.15; ea = saturation_vapor_pressure(Ta) - 2000.0
    R = 0.5 * 1000 + 0.98 * (350 + _σ * Ta^4)
    bl = CampbellNormanBoundaryLayer(leaf_width = 0.05)
    for gs in (0.2, 0.3, 0.4)
        r = leaf_energy_balance(gs, bl; R_abs = R, T_air = Ta, e_air = ea, U = 0.0)
        @printf("  still air, g_s = %.1f: ΔT = %.2f K (rule of thumb 6–10 K)\n", gs, r.T_leaf - Ta)
        @test 6 <= r.T_leaf - Ta <= 10
    end
    # Their second figure, 3–5 K at 5 m/s, is NOT reproduced (see the PR):
    r5 = leaf_energy_balance(0.3, bl; R_abs = R, T_air = Ta, e_air = ea, U = 5.0)
    @printf("  U = 5 m/s, g_s = 0.3: ΔT = %.2f K (rule of thumb 3–5 K; not asserted)\n", r5.T_leaf - Ta)
    @test r5.T_leaf - Ta > 0
end

end # testset
