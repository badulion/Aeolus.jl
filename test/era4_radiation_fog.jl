# Era IV, Phase 44 (#9) — radiation fog with the droplet-radiative feedback.
#
# The validation runs (time-step convergence over 3 h, budgets over 6 h,
# the 14 h ParisFog 2011 case) are validation/era4_radiation_fog.jl.
#
# Reference numbers quoted below come from two SIRTA (Paris) studies:
#   * Wærsted et al. (2017, ACP 17, 10811): radiative budgets of six winter
#     fogs from ground-based remote sensing + radiative transfer.
#   * Mazoyer et al. (2017, ACP 17, 13017): the ParisFog radiation fog of
#     14–15 Nov 2011, observations and LES.

using Test
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
        @test all(isapprox(fog_effective_radius(ql, ρ, N), 1e6 * droplet_effective_radius(ql * ρ, N);
                           rtol = 1e-14) for (ql, ρ, N) in ((2e-4, 1.25, 50e6), (4e-4, 1.3, 20e6)))
        # Twomey scaling r_eff ∝ N^{-1/3} at fixed LWC, and ∝ LWC^{1/3}
        @test fog_effective_radius(2e-4, 1.2, 8 * 50e6) ≈
              fog_effective_radius(2e-4, 1.2, 50e6) / 2 rtol = 1e-12
        @test fog_effective_radius(0.0, 1.2, 50e6) == 1.0     # r_min guard
        # A typical fog: 0.2 g/m³ at 50 cm⁻³ gives ≈ 11 μm (Mazoyer et al.
        # 2017 observed 95 % of drops below 20 μm diameter)
        r = fog_effective_radius(0.2e-3 / 1.25, 1.25, 50e6)
        @test 8 < r < 14
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
        # Heating evaporates, cooling condenses, zero does nothing
        @test radiative_condensation_rate(1.0, 280.0, 1.25) < 0
        @test radiative_condensation_rate(0.0, 280.0, 1.25) == 0
        # Cross-check against this module's exact enthalpy saturation
        # adjustment: remove Q·dt from a saturated parcel and adjust. The
        # linearised formula (dry c_p, ρ_s-based) agrees within 3 %.
        for T in (280.0,)
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
        # 8 K/km: stable for dry air (Γ_d = 9.8 K/km), unstable when
        # saturated at 275 K (Γ_m ≈ 6 K/km). Unsaturated, the parcel estimate
        # is g/θ_ρ ∂θ_ρ/∂z (a first version compared with plain θ and was
        # 13 % off: the expectation was wrong, not the code).
        zf = collect(0.0:20.0:1000.0)
        zc = (zf[1:end-1] .+ zf[2:end]) ./ 2
        T = 276.0 .- 8e-3 .* zc
        p = 1.0e5 .* exp.(-zc ./ 8000)
        dry = FogColumn(zf, T, 0.7 .* saturation_specific_humidity.(T, p))
        k = 8
        θρ(i) = density_potential_temperature(dry.T[i], dry.p[i], PhasePartition(dry.qv[i]))
        @test moist_stability(dry, k) ≈ 9.80665 / θρ(k) * (θρ(k + 1) - θρ(k)) / 20 rtol = 0.02
        @test moist_stability(FogColumn(zf, T, 1.001 .* saturation_specific_humidity.(T, p)), k) < 0
    end

    @testset "Bulk droplet settling" begin
        ρ = 1.25
        v = settling_velocity(0.2e-3 / ρ, ρ, 50e6)
        @test 0.5e-2 < v < 2e-2                    # ≈ 1 cm/s for fog
        # Stokes regime: v ∝ r_eff² ∝ LWC^{2/3}
        v8 = settling_velocity(8 * 0.2e-3 / ρ, ρ, 50e6)
        @test v8 / v ≈ 4 rtol = 0.03
        @test settling_velocity(0.0, ρ, 50e6) == 0
    end

    @testset "Column energy and water budgets close to round-off (2 min)" begin
        # The ParisFog column with a fog layer in its lowest 100 m (vapour
        # at 100.2 % of saturation, condensed by the constructor), so that
        # radiative cooling, dew, condensation, settling and turbulent
        # mixing are all active from the first step. Same tolerances as the
        # 6 h run of validation/era4_radiation_fog.jl; measured residuals
        # 9e-12 (energy, of the source scale) and 8e-13 (water) over 10 min;
        # 2 min (24 steps of 5 s) keeps every process active and the budget
        # identity is per step, so the tolerances stay.
        c0 = parisfog_2011_column()
        qv = [c0.z[k] < 100 ? 1.002 * saturation_specific_humidity(c0.T[k], c0.p[k]) : c0.qv[k]
              for k in eachindex(c0.T)]
        col = FogColumn(c0.zf, c0.T, qv; u = c0.u, v = c0.v, Ts = c0.Ts, p_sfc = 1.02e5)
        E0 = column_energy(col); W0 = column_water(col)
        run!(col, 120)
        b = col.budget
        @test fog_liquid_water_path(col) > 0           # the fog is there (2 min: 0.6 g/m², 10 min: > 1)
        @test b.fallout_mass < 0 && b.vapour_mass != 0
        src = b.radiation + b.sensible + b.vapour_enthalpy + b.fallout_enthalpy
        scale = abs(b.radiation) + abs(b.sensible) + abs(b.vapour_enthalpy) +
                abs(b.fallout_enthalpy)
        ΔE = column_energy(col) - E0
        @test abs(ΔE - src) < 1e-9 * scale
        ΔW = column_water(col) - W0
        @test ΔW ≈ b.vapour_mass + b.fallout_mass rtol = 1e-9
    end

    @testset "fog_life_cycle: onset and dissipation of a synthetic history" begin
        # 10-min records from 23:00 UTC; fog at z_obs in records 3-4 (a
        # 600 s blip, shorter than min_duration), 6-12 (3600 s) and 16-20
        # (2400 s); LWP peaks at record 9.
        fog = falses(22); fog[[3, 4]] .= true; fog[6:12] .= true; fog[16:20] .= true
        mk(fog) = [(t = 600.0 * (i - 1), utc_hour = 23 + (i - 1) / 6, fog_obs = fog[i],
                    fog_top = fog[i] ? 10.0 * i : 0.0, lwp = fog[i] ? 100.0 - (i - 9)^2 : 0.0)
                   for i in eachindex(fog)]
        lc = fog_life_cycle(mk(fog))
        @test lc.onset ≈ 23 + 5 / 6            # record 6: the blip does not count
        @test lc.dissipation ≈ 23 + 20 / 6     # record 21: end of the last episode
        @test lc.max_top == 200.0 && lc.max_lwp == 100.0
        @test lc.t_max_lwp ≈ 23 + 8 / 6
        # min_duration below the blip: the blip is the onset
        @test fog_life_cycle(mk(fog); min_duration = 600).onset ≈ 23 + 2 / 6
        # fog still there at the end of the record: no dissipation
        fog[21:22] .= true
        @test isnan(fog_life_cycle(mk(fog)).dissipation)
        # no fog at all
        lc0 = fog_life_cycle(mk(falses(22)))
        @test isnan(lc0.onset) && isnan(lc0.dissipation)
    end
end
