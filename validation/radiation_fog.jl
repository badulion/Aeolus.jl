# Era IV, Phase 44 (#9) — radiation fog with the droplet-radiative
# feedback: the validation runs of the ParisFog 2011 column. The default
# time step converged against a halved step over 3 h; the column's energy
# and water budgets closed over 6 h of fog formation; and the 14 h ParisFog
# case (14–15 Nov 2011): night-time formation, the droplet-radiative
# feedback, the opaque-fog radiative budget and solar dissipation. The
# module stays tested in test/radiation_fog.jl (optics, longwave and
# shortwave of a static fog, radiation-driven condensation, the column
# numerics, a 10 min column run's budgets, fog_life_cycle).
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

@testset "Era IV — Radiation fog, droplet-radiative feedback (Phase 44 #9)" begin

    @testset "Column time-step convergence (no K-closure fibrillation)" begin
        # Review finding: the first default, dt = 10 s, was not converged —
        # the stability-dependent K oscillated with period 2Δt and the
        # surface ran 0.32 K colder after 3 h than at dt = 2.5 s, moving
        # ground-fog onset by 45 min. The default must agree with a halved
        # step before fog forms (23:20 → 02:20 UTC).
        res = Dict{Float64, Any}()
        for dt in (FogColumnSettings().dt, FogColumnSettings().dt / 2)
            col = parisfog_2011_column(; settings = FogColumnSettings(dt = dt, dt_out = 3600.0))
            run!(col, 3 * 3600)
            res[dt] = col
        end
        a, b = res[FogColumnSettings().dt], res[FogColumnSettings().dt / 2]
        @info "dt vs dt/2 after 3 h" ΔTs = a.Ts - b.Ts max_ΔT = maximum(abs.(a.T .- b.T))
        @test abs(a.Ts - b.Ts) < 0.05
        @test maximum(abs.(a.T .- b.T)) < 0.05
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

    # ---------------------------------------------------------------------
    # ParisFog, 14–15 Nov 2011 (Mazoyer et al. 2017): the feedback at work
    # ---------------------------------------------------------------------
    @testset "ParisFog 2011 case: droplet-radiative feedback and dissipation" begin
        dur = 14 * 3600                                  # 23:20 → 13:20 UTC
        runs = Dict{Symbol, Any}()
        for (name, set) in ((:ref, FogColumnSettings()),
                            (:nofeedback, FogColumnSettings(droplet_feedback = false)),
                            (:noSW, FogColumnSettings(shortwave = false)))
            col = parisfog_2011_column(; settings = set)
            E0 = column_energy(col)
            run!(col, dur)
            runs[name] = (; col, lc = fog_life_cycle(col.history), E0)
        end
        ref, nof, nosw = runs[:ref], runs[:nofeedback], runs[:noSW]
        H = ref.col.history
        sunrise = 24 + 6 + 59 / 60                       # 06:59 UTC, 15 Nov
        at(h, hr) = h[argmin([abs(x.utc_hour - hr) for x in h])]
        @info "ParisFog life cycle (UTC hours since 00 UTC 14 Nov)" reference = ref.lc no_feedback = nof.lc no_shortwave = nosw.lc

        # --- Formation: at night, aloft first, then down to the ground
        # (observed: liquid at 150 m at 02:00, fog at the ground 02:30)
        @test 24 < ref.lc.onset < 30
        first_fog = H[findfirst(h -> h.fog_top > 0, H)]
        @test !first_fog.fog_obs && first_fog.fog_base > 10

        # --- The feedback: transparent droplets give later, thinner fog
        @test isnan(nof.lc.onset) || nof.lc.onset > ref.lc.onset
        @test nof.lc.max_lwp < 0.5 * ref.lc.max_lwp
        d_ref = at(H, 30.0); d_nof = at(nof.col.history, 30.0)   # 06:00 UTC
        @test d_ref.lwp > 3 * d_nof.lwp
        @test d_ref.fog_top > d_nof.fog_top
        @info "06:00 UTC, feedback on vs off" top_on = d_ref.fog_top top_off = d_nof.fog_top lwp_on = d_ref.lwp lwp_off = d_nof.lwp

        # --- Opaque fog at night: radiative budget vs Wærsted et al. (2017)
        night_opaque = filter(h -> h.lwp > 30 && h.utc_hour < sunrise, H)
        @test length(night_opaque) >= 6
        sfc  = [h.lw_net_sfc for h in night_opaque]
        top  = [h.lw_net_top for h in night_opaque]
        cool = [h.lw_top_cooling for h in night_opaque]
        CLW  = [h.C_LW for h in night_opaque]
        perW = [h.C_LW / -h.lw_fog for h in night_opaque]
        @info "opaque fog, night" sfc_net_LW = extrema(sfc) net_LW_fog_top = extrema(top) max_cooling_K_day = extrema(cool) C_LW_g_m2_h = extrema(CLW) g_per_W = extrema(perW)
        @test all(-20 .< sfc .< 0)               # observed: → 0, often −10
        @test all(-100 .< top .< -45)            # observed −73 … −100
        @test all(-250 .< cool .< -24)           # tens of K/day at the top
        # Wærsted et al. find C_LW = 40–70 g m⁻² h⁻¹ for opaque fog;
        # Nakanishi's (2000) LES ≈ 30–40. This column's fog has a smaller
        # net LW deficit at its top (≈ −64 … −68 W m⁻² vs their −73 … −100)
        # and sits at the low end.
        @test all(25 .< CLW .< 70)
        @test all(0.5 .< perW .< 0.7)            # their 0.55 at −2 °C … 0.90 at 15 °C

        # --- Dissipation: the sun removes the fog, the night does not
        @test !isnan(ref.lc.dissipation)
        @test sunrise < ref.lc.dissipation < sunrise + 6
        @test H[end].lwp < 1.0                                   # gone by 13:20
        @test nosw.col.history[end].lwp > 30                     # persists
        day = filter(h -> h.utc_hour > sunrise && h.lwp > 1, H)
        @test maximum(h.E_SW for h in day) > 1
        @test maximum(h.E_SW for h in day) < 15          # Wærsted: 0–15
        @test maximum(h.sw_abs_sfc for h in day) > 40    # Wærsted: 40–120 at SIRTA
        @info "dissipation" hours_after_sunrise = ref.lc.dissipation - sunrise noSW_ground_clear = nosw.lc.dissipation - sunrise lwp_end_ref = H[end].lwp lwp_end_noSW = nosw.col.history[end].lwp max_E_SW = maximum(h.E_SW for h in day)

        # --- Energy closes over the whole 14 h, day included
        b = ref.col.budget
        src = b.radiation + b.sensible + b.vapour_enthalpy + b.fallout_enthalpy
        scale = abs(b.radiation) + abs(b.sensible) + abs(b.vapour_enthalpy) + abs(b.fallout_enthalpy)
        @test abs(column_energy(ref.col) - ref.E0 - src) < 1e-9 * scale
    end
end
