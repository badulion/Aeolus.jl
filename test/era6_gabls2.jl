# N-parent-c (#248), ONE-g (#456) — LargeScaleForcing: GABLS2 (reduced) on
# the one model, the validation run of large-scale subsidence (check 8 of
# the issue): the configuration examples/sbl/configs/gabls2.toml with its
# dynamic driver (the surface θ series and the subsidence from 40 h), the
# heat budget with the subsidence measured as the large_scale_forcing
# term, the residual-layer inversion lowered by about the passive descent,
# and night 2 against the gabls2_reference.csv envelopes. Checks 1-7 of
# ProfileNudging and Subsidence run in the fast tier
# (era6_nudging_subsidence.jl); this mini-simulation stays full-only.

using Test, Printf
using Aeolus
using Aeolus.Config: load_run_config, build
using Aeolus.SimulationDriver: SimulationDriverType
using Aeolus.Components: Components, Domain, LargeScale, SurfaceLayer, step_adaptive!, with_budgets
using Aeolus.BudgetSystem: stage_budget
using Aeolus.Fields: interior_view
using Statistics: mean

isdefined(Main, :gabls3_data) || include(joinpath(@__DIR__, "support", "sbl_data.jl"))

_g2comp(d, T) = only(c for c in d.components if c isa T)

"Advance the domain `d` to the model time `t_end` (adaptive steps)."
function g2_run!(d, t_end)
    dt_prev = nothing
    while d.state.t < t_end - 1e-9
        _, dt, _ = step_adaptive!(d; dt_prev = dt_prev, t_end = t_end)
        dt_prev = dt
    end
    return d
end

"The energy budget of the air since its start: storage, the subsidence term and the closure (J m⁻²)."
function g2_budget(d)
    b = stage_budget(Components.budgets(d).measured[:energy])
    return (storage = b.storage, subsidence = get(b.terms, (:body_force, :large_scale_forcing), 0.0),
            closure = b.closure, scale = maximum(abs, values(b.terms)))
end

@testset "N-parent-c — GABLS2 (reduced) with large-scale subsidence (#248)" begin

    @testset "8. GABLS2 (reduced): subsidence from 40 h, the heat budget, inversion, night 2" begin
        # 16 × 16 × 40 on 2.4 × 2.4 × 1.6 km (Δx = 150 m, Δz = 40 m), 16 LT
        # 22 Oct to 06 LT 24 Oct (16-54 h). One run to 40 h, then the case
        # with subsidence and a copy without it (the D = 0 sensitivity run).
        cfg = load_run_config(joinpath(SBL_CONFIGS, "gabls2.toml");
                              overrides = ["grid.size=[16,16,40]", "output.variables=[\"budget_energy\"]",
                                           "checkpoint.every_seconds=inf"])
        d = only(build(SimulationDriverType(), cfg).model.domains)
        ls = _g2comp(d, LargeScale)
        @test count(f -> f isa Subsidence, ls.extra) == 1
        # the driver's surface θ is the case formula, sampled every minute (θ with p_s = 972 hPa)
        sl = _g2comp(d, SurfaceLayer)
        for h in (16.0, 30.0, 40.0, 52.5)
            @test surface_temperature(sl.Ts, 3600h) ≈ (gabls2_surface_temperature_C(h) + 273.15) * (1e5 / 97200)^0.286 rtol = 1e-12
        end
        g2_run!(d, 40 * 3600.0)
        b40 = g2_budget(d)
        @test b40.subsidence == 0                       # off before 40 h local
        # the copy without subsidence, its budget started at 40 h
        dc = deepcopy(d)
        lc = _g2comp(dc, LargeScale)
        nosubs = LargeScale(lc.geostrophic, Tuple(f for f in lc.extra if !(f isa Subsidence)), lc.U_g, lc.V_g, lc.t0)
        d0 = with_budgets(Domain(dc; components = Tuple(c isa LargeScale ? nosubs : c for c in dc.components)), [:energy])
        prof(d) = Components.sl_profiles(_g2comp(d, SurfaceLayer), d)
        zinv(d) = (p = prof(d); inversion_height(p.z, p.θ))
        screen(d) = (s = Components.sl_screen(_g2comp(d, SurfaceLayer), d);
                     (T2 = s.θ_screen * (97200 / 1e5)^0.286 - 273.15, U10 = s.U_screen))
        z40 = zinv(d)
        for x in (d, d0); g2_run!(x, 42 * 3600.0); end
        z42 = zinv(d); z42_0 = zinv(d0)
        for x in (d, d0); g2_run!(x, 48 * 3600.0); end
        s48 = screen(d); s48_0 = screen(d0)
        for x in (d, d0); g2_run!(x, 54 * 3600.0); end
        z54 = zinv(d); z54_0 = zinv(d0)
        # the passive descent dz/dt = w_s(z) of a level that starts at the
        # 18 LT (42 h) inversion, after the evening transition
        W = only(f for f in ls.extra if f isa Subsidence).w_s
        zp = z42
        for t in 42 * 3600.0:10.0:(54 * 3600.0 - 10)
            k1 = forcing_value(W, zp, t); k2 = forcing_value(W, zp + 5k1, t + 5)
            k3 = forcing_value(W, zp + 5k2, t + 5); k4 = forcing_value(W, zp + 10k3, t + 10)
            zp += 10 * (k1 + 2k2 + 2k3 + k4) / 6
        end
        dz = d.state.mesh.dz
        B = g2_budget(d); B0 = g2_budget(d0)
        ref = gabls2_reference()
        rows = [r for r in ref.rows if (startswith(r[1], "obs_") || startswith(r[1], "les_")) && 42 <= r[2] <= 54]
        val(s, q) = endswith(q, "T2m") ? s.T2 : s.U10
        inside = [(r[1], r[3] <= val(s48, r[1]) <= r[4], r[3] <= val(s48_0, r[1]) <= r[4]) for r in rows]
        @info @sprintf("GABLS2 (16×16×40): residual-layer inversion %.1f m at 40 h; at 06 LT 24 Oct (54 h) %.1f m with subsidence, %.1f m without (change %.1f m); passive descent from the 42 h inversion (%.1f m) → %.1f m (model − passive %.1f m = %.2f Δz)",
                       z40, z54, z54_0, z54 - z54_0, z42, zp, z54 - zp, (z54 - zp) / dz)
        @info @sprintf("GABLS2 (16×16×40) at 48 h: T2 %.2f °C with / %.2f °C without subsidence, U10 %.2f / %.2f m/s; envelopes (inside with, without): %s",
                       s48.T2, s48_0.T2, s48.U10, s48_0.U10, join(("$(q) $(a ? "in" : "out")/$(b ? "in" : "out")" for (q, a, b) in inside), ", "))
        @info @sprintf("GABLS2 (16×16×40) heat budget of the air: closure %.1e J/m² (with, 16-54 h), %.1e J/m² (without, 40-54 h), of the largest term %.2e; subsidence heating %.4e J/m²",
                       B.closure, B0.closure, B.scale, B.subsidence)
        # the heat budget closes, every change attributed (the sbl run type's
        # ledger closed to 8e-13 of the column content)
        @test abs(B.closure) <= 1e-9 * B.scale && abs(B0.closure) <= 1e-9 * B0.scale
        @test B.subsidence > 0 && B0.subsidence == 0      # descent into a stable column warms it
        @test z42_0 == z40 || abs(z42_0 - z40) < dz       # without: the residual inversion stays
        @test abs(z54_0 - z40) < dz
        @test z54 < z54_0 - 3dz                           # with: it descends …
        @test abs(z54 - zp) < dz                          # … as a passive surface, within one Δz
        # night 2 against the digitised envelopes: subsidence moves the
        # screen values by less than the digitisation half-widths and does
        # not change which envelopes the run is inside
        @test abs(s48.T2 - s48_0.T2) < 0.7 && abs(s48.U10 - s48_0.U10) < 0.4
        @test all(a == b for (_, a, b) in inside)
    end
end
