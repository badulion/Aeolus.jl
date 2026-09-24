# Era IV, Phase 44 (#9) — radiation fog with the droplet-radiative feedback.
#
# Reference numbers quoted below come from two SIRTA (Paris) studies:
#   * Wærsted et al. (2017, ACP 17, 10811): radiative budgets of six winter
#     fogs from ground-based remote sensing + radiative transfer.
#   * Mazoyer et al. (2017, ACP 17, 13017): the ParisFog radiation fog of
#     14–15 Nov 2011, observations and LES.

using Test
using Printf
using Aeolus
using Aeolus.MoistThermodynamics
using Aeolus.RadiationFog
using Aeolus.AerosolCloud: droplet_effective_radius

# A static column with a uniform fog layer of depth `hfog` under a
# subsaturated, slightly stable layer — the radiation-only setup.
function _static_fog_column(; nz = 50, H = 1000.0, hfog = 150.0, ql_fog = 3e-4,
                            T0 = 278.0)
    zf = collect(range(0, H, length = nz + 1)); dz = diff(zf)
    zc = (zf[1:end-1] .+ zf[2:end]) ./ 2
    Tl = [z < hfog ? T0 - 6.5e-3z : T0 - 6.5e-3hfog + 3e-3(z - hfog) for z in zc]
    pf = zeros(nz + 1); pf[1] = 1.013e5
    for k in 1:nz
        pf[k + 1] = pf[k] * exp(-9.81 * dz[k] / (287.0 * Tl[k]))
    end
    pl = (pf[1:end-1] .+ pf[2:end]) ./ 2
    ρ  = pl ./ (287.0 .* Tl)
    qv = [saturation_specific_humidity(Tl[k], pl[k]) * (zc[k] < hfog ? 1.0 : 0.6) for k in 1:nz]
    ql = [zc[k] < hfog ? ql_fog : 0.0 for k in 1:nz]
    return (; nz, H, zf, dz, zc, Tl, pf, pl, ρ, qv, ql, T0,
              kt = findlast(>(0), ql))
end

@testset "Era IV — Radiation fog, droplet-radiative feedback (Phase 44 #9)" begin

    @testset "Droplet optics inputs delegate to AerosolCloud" begin
        for (ql, ρ, N) in ((2e-4, 1.25, 50e6), (5e-5, 1.2, 200e6), (4e-4, 1.3, 20e6))
            @test fog_effective_radius(ql, ρ, N) ≈
                  1e6 * droplet_effective_radius(ql * ρ, N) rtol = 1e-14
        end
        # Twomey scaling r_eff ∝ N^{-1/3} at fixed LWC, and ∝ LWC^{1/3}
        @test fog_effective_radius(2e-4, 1.2, 8 * 50e6) ≈
              fog_effective_radius(2e-4, 1.2, 50e6) / 2 rtol = 1e-12
        @test fog_effective_radius(0.0, 1.2, 50e6) == 1.0     # r_min guard
        # A typical fog: 0.2 g/m³ at 50 cm⁻³ gives ≈ 11 μm (Mazoyer et al.
        # 2017 observed 95 % of drops below 20 μm diameter)
        r = fog_effective_radius(0.2e-3 / 1.25, 1.25, 50e6)
        @test 8 < r < 14
        @info "fog effective radius, 0.2 g/m³, 50 cm⁻³" r_eff_μm = r
    end

    @testset "Longwave: an opaque fog moves the cooling from the ground to its top" begin
        c = _static_fog_column()
        rad = fog_radiation(c.nz, c.H, c.pf[end], c.Tl[end], c.qv[end])
        res = Dict{Bool, Any}()
        for fb in (true, false)
            col = fog_atmospheric_column(rad, c.Tl, c.pl, c.pf, c.dz, c.ρ, c.qv,
                                         c.ql, c.T0; N_d = 100e6, droplet_feedback = fb)
            fog_radiative_transfer!(rad, col)
            Q = zeros(c.nz); fog_radiative_heating!(Q, rad, col)
            f = rad.flux_lw
            # Σ Q Δz telescopes to the net-flux difference: the radiative
            # heating the column model applies is exactly the boundary flux
            @test sum(Q .* c.dz) ≈ f.flux_net[c.nz + 1] - f.flux_net[1] rtol = 1e-12
            jump_top, _ = fog_top_flux_jump(rad, c.kt - 4, c.kt)   # top 100 m
            jump_all, _ = fog_top_flux_jump(rad, 1, c.kt)
            res[fb] = (; sfc = f.flux_net[1], dn = f.flux_dn[1], top = f.flux_net[c.kt + 1],
                         Q, jump_top, jump_all,
                         Kday = Q ./ (c.ρ .* 1004.6) .* 86400)
        end
        on, off = res[true], res[false]
        lwp = sum(c.ρ .* c.ql .* c.dz) * 1000
        @info "static fog LW" LWP_g_m2 = lwp sfc_net_clear = off.sfc sfc_net_fog = on.sfc LWdn_clear = off.dn LWdn_fog = on.dn fog_layer_loss = on.jump_all top100m_loss = on.jump_top top_layer_K_day = on.Kday[c.kt] net_at_fog_top = on.top
        # Clear sky: the ground loses ≈ 60 W/m² before fog forms at SIRTA
        # (Wærsted et al. 2017, 27 Oct 2014 case).
        @test -80 < off.sfc < -45
        # Opaque fog (LWP > 30 g/m²): net LW at the ground → ≈ 0; Wærsted
        # et al. report it "frequently reaches −10 W m⁻²" in opaque fog.
        @test lwp > 30
        @test -15 < on.sfc < 0
        @test on.dn - off.dn > 40
        # The net LW at the fog top is what drives the fog: −73 to −100
        # W m⁻² at SIRTA; this column's moister, warmer upper air gives less.
        @test -100 < on.top < -45
        # The loss is concentrated in the uppermost 100 m of the fog
        # ("restricted to the uppermost 50–100 m", Wærsted et al. 2017)
        @test on.jump_top / on.jump_all > 0.8
        # Fog-top cooling: tens of K/day in a 20 m layer (the Mazoyer et
        # al. LES at 1 m resolution peaks above 5.5 K/h = 132 K/day)
        @test -250 < on.Kday[c.kt] < -24
        # Without the droplets the same humid layer barely cools
        @test abs(off.jump_all) < 0.1 * abs(on.jump_all)
        @test abs(off.Kday[c.kt]) < 5
    end

    @testset "Shortwave: fog reflects, and absorbs a little" begin
        c = _static_fog_column()
        rad = fog_radiation(c.nz, c.H, c.pf[end], c.Tl[end], c.qv[end])
        out = Dict{Bool, Any}()
        for fb in (true, false)
            col = fog_atmospheric_column(rad, c.Tl, c.pl, c.pf, c.dz, c.ρ, c.qv,
                                         c.ql, c.T0; N_d = 100e6, droplet_feedback = fb)
            fog_radiative_transfer!(rad, col; μ₀ = 0.35, albedo = 0.22)
            _, jsw = fog_top_flux_jump(rad, 1, c.kt)
            out[fb] = (; sfc = rad.flux_sw.flux_net[1], toa_up = rad.flux_sw.flux_up[end], jsw)
        end
        @info "static fog SW, μ₀ = 0.35" sfc_abs_clear = out[false].sfc sfc_abs_fog = out[true].sfc fog_absorbed = out[true].jsw
        @test out[true].sfc < 0.6 * out[false].sfc       # fog shades the ground
        @test out[true].toa_up > out[false].toa_up + 50  # and brightens the scene
        @test out[true].jsw > out[false].jsw             # droplets absorb SW
        # Night: no SW at all
        col = fog_atmospheric_column(rad, c.Tl, c.pl, c.pf, c.dz, c.ρ, c.qv, c.ql, c.T0; N_d = 100e6)
        fog_radiative_transfer!(rad, col; μ₀ = 0.0)
        @test all(==(0), rad.flux_sw.flux_net)
    end

    @testset "Radiation-driven condensation (Wærsted et al. 2017, Eq. 5)" begin
        # Their Sect. 2.4: condensed water per radiative loss rises from
        # 0.55 to 0.90 g m⁻² h⁻¹ per W m⁻² between −2 and 15 °C.
        per_W(Tc) = begin
            T = 273.15 + Tc; p = 1.0e5
            ρ = air_density(T, p, PhasePartition(saturation_specific_humidity(T, p)))
            radiative_condensation_rate(-1.0, T, ρ) * 1000 * 3600
        end
        @test per_W(-2.0) ≈ 0.55 rtol = 0.06
        @test per_W(15.0) ≈ 0.90 rtol = 0.03
        @test per_W(-2.0) < per_W(5.0) < per_W(15.0)
        @info "g m⁻² h⁻¹ per W m⁻²" at_m2C = per_W(-2.0) at_15C = per_W(15.0)
        # Heating evaporates, cooling condenses, zero does nothing
        @test radiative_condensation_rate(1.0, 280.0, 1.25) < 0
        @test radiative_condensation_rate(0.0, 280.0, 1.25) == 0
        # Cross-check against this module's exact enthalpy saturation
        # adjustment: remove Q·dt from a saturated parcel and adjust. The
        # linearised formula (dry c_p, ρ_s-based) agrees within 3 %.
        for T in (271.15, 280.0, 288.15)
            p = 1.0e5; qv = saturation_specific_humidity(T, p)
            ρ = air_density(T, p, PhasePartition(qv)); dt = 600.0
            T1 = T - dt / ρ / heat_capacity_moist(PhasePartition(qv))
            r = fog_condensation(T1, p, qv, 0.0, dt)
            @test ρ * r.Δq / dt ≈ radiative_condensation_rate(-1.0, T, ρ) rtol = 0.03
        end
    end

    # ---------------------------------------------------------------------
    # The 1-D column model
    # ---------------------------------------------------------------------
    @testset "Column numerics: conservative implicit mixing" begin
        n = 12
        φ = [sin(0.7k) + 2 for k in 1:n]; m = [1.0 + 0.1k for k in 1:n]
        D = [0.0; fill(3.0, n - 1); 0.0]
        tot0 = sum(m .* φ)
        Aeolus.RadiationFog._implicit_diffuse!(φ, D, m, 50.0)
        @test sum(m .* φ) ≈ tot0 rtol = 1e-14
        # huge dt: fully mixed to the mass-weighted mean
        Aeolus.RadiationFog._implicit_diffuse!(φ, D, m, 1e9)
        @test maximum(φ) - minimum(φ) < 1e-6
        @test all(isapprox.(φ, tot0 / sum(m); rtol = 1e-6))
        zf = stretched_levels(1000.0)
        @test zf[1] == 0 && zf[end] == 1000 && zf[2] ≈ 1.0
        @test all(diff(zf) .<= 25.0 + 1e-9) && issorted(zf)
    end

    @testset "Moist stability: fog turns a dry-stable layer unstable" begin
        # 8 K/km lapse rate: stable for dry air (Γ_d = 9.8 K/km), unstable
        # for saturated air at 275 K (Γ_m ≈ 6 K/km).
        zf = collect(0.0:20.0:1000.0); nz = length(zf) - 1
        zc = (zf[1:end-1] .+ zf[2:end]) ./ 2
        T = 276.0 .- 8e-3 .* zc
        p = 1.0e5 .* exp.(-zc ./ 8000)
        dry = FogColumn(zf, T, 0.7 .* saturation_specific_humidity.(T, p))
        k = 8
        N2dry = moist_stability(dry, k)
        θρ = [density_potential_temperature(dry.T[i], dry.p[i], PhasePartition(dry.qv[i])) for i in 1:nz]
        @test N2dry > 0
        # unsaturated: the parcel estimate reduces to g/θ_ρ ∂θ_ρ/∂z.
        # (A first version compared against plain θ and was 13 % off: the
        # vapour decreasing with height lowers the θ_ρ gradient. The
        # expectation was wrong, not the code.)
        @test N2dry ≈ 9.80665 / θρ[k] * (θρ[k + 1] - θρ[k]) / 20 rtol = 0.02
        wet = FogColumn(zf, T, 1.001 .* saturation_specific_humidity.(T, p))
        @test all(wet.ql .> 0)
        N2wet = moist_stability(wet, k)
        @test N2wet < 0
        @info "8 K/km layer" N2_dry = N2dry N2_saturated = N2wet
        # neutral surface layer: stability function 1
        ex = surface_exchange(dry)
        @test ex.C_H > 0 && ex.C_D > 0
    end

    @testset "Bulk droplet settling" begin
        ρ = 1.25
        v = settling_velocity(0.2e-3 / ρ, ρ, 50e6)
        @test 0.5e-2 < v < 2e-2                    # ≈ 1 cm/s for fog
        # Stokes regime: v ∝ r_eff² ∝ LWC^{2/3}
        v8 = settling_velocity(8 * 0.2e-3 / ρ, ρ, 50e6)
        @test v8 / v ≈ 4 rtol = 0.03
        @test settling_velocity(0.0, ρ, 50e6) == 0
        @info "fog settling velocity, 0.2 g/m³, 50 cm⁻³" v_cm_s = 100v
    end

    @testset "Column energy and water budgets close to round-off" begin
        # 6 h from 23:20 UTC: radiative cooling, dew, fog formation,
        # settling and turbulent mixing all active.
        col = parisfog_2011_column()
        E0 = column_energy(col); W0 = column_water(col)
        run!(col, 6 * 3600)
        b = col.budget
        @test fog_liquid_water_path(col) > 1e-3        # fog did form
        @test b.fallout_mass < 0 && b.vapour_mass != 0
        src = b.radiation + b.sensible + b.vapour_enthalpy + b.fallout_enthalpy
        scale = abs(b.radiation) + abs(b.sensible) + abs(b.vapour_enthalpy) +
                abs(b.fallout_enthalpy)
        ΔE = column_energy(col) - E0
        @info "6 h column energy budget (J/m²)" ΔE sources = src radiation = b.radiation sensible = b.sensible vapour = b.vapour_enthalpy fallout = b.fallout_enthalpy residual = ΔE - src
        @test abs(ΔE - src) < 1e-9 * scale
        ΔW = column_water(col) - W0
        @test ΔW ≈ b.vapour_mass + b.fallout_mass rtol = 1e-9
        @info "6 h column water budget (kg/m²)" ΔW vapour = b.vapour_mass fallout = b.fallout_mass
    end
end
