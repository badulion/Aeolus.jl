# N-parent-c (#248) — LargeScaleForcing: GABLS2 (reduced), the validation
# run of large-scale subsidence (check 8 of the issue): subsidence from
# 40 h, the heat ledger with it booked, the residual-layer inversion
# lowered by about the passive descent, and night 2 against the
# gabls2_reference.csv envelopes. Split from era6_nudging_subsidence.jl
# (#320) so that checks 1-7 of ProfileNudging and Subsidence run in the
# fast tier (their only fast check before was the Berlin parent smoke
# test, test/berlin_parent_smoke.jl, deleted with the Berlin suite) while
# this mini-simulation stays full-only.

using Test, Printf
using Aeolus
using Aeolus.Fields: interior_view, set!
using Statistics: mean

using Aeolus.StableBoundaryLayer
isdefined(Main, :gabls3_data) || include(joinpath(@__DIR__, "support", "sbl_data.jl"))

@testset "N-parent-c — GABLS2 (reduced) with large-scale subsidence (#248)" begin

    @testset "8. GABLS2 (reduced): subsidence from 40 h, ledger, inversion, night 2" begin
        # 16 × 16 × 40 on 2.4 × 2.4 × 1.6 km (Δx = 150 m, Δz = 40 m), 16 LT
        # 22 Oct to 06 LT 24 Oct (16-54 h). One run to 40 h, then the case
        # with subsidence and a copy without it (the D = 0 sensitivity run).
        # The full-resolution numbers are from examples/sbl/gabls2.jl (PR body).
        # The case of examples/sbl/configs/gabls2.toml (the `sbl` run driver,
        # CFG-d #359), its surface temperature sampled to 56 h.
        c = sbl_build("gabls2"; overrides = ["grid.size=[16,16,40]",
            "surface.temperature={kind=\"gabls2\", p_sfc=97200.0, exponent=0.286, sample_every=60.0, until=201600.0}"]).case
        @test count(f -> f isa Subsidence, c.forcing) == 1
        H0 = sbl_column_heat(c)
        ledger(c) = sbl_column_heat(c) - H0 - c.heat_applied - c.adv_heat_applied - c.ls_heat_applied
        res = Dict(:subs => 0.0, :nosubs => 0.0)
        cb(tag) = c -> (res[tag] = max(res[tag], abs(ledger(c))))
        sbl_run!(c, 40 * 3600.0; callback = cb(:subs))
        @test c.ls_heat_applied == 0                    # off before 40 h local
        c0 = deepcopy(c)
        c0.forcing = Tuple(f for f in c0.forcing if !(f isa Subsidence))
        res[:nosubs] = res[:subs]
        prof(c) = sbl_profiles(c)
        zinv(c) = (p = prof(c); sbl_inversion_height(p.z, p.θ))
        screen(c) = (s = sbl_screen(c); (T2 = s.θ_screen * (97200 / 1e5)^0.286 - 273.15, U10 = s.U_screen))
        z40 = zinv(c)
        for cc in (c, c0); sbl_run!(cc, 42 * 3600.0; callback = cb(cc === c ? :subs : :nosubs)); end
        z42 = zinv(c); z42_0 = zinv(c0)
        for cc in (c, c0); sbl_run!(cc, 48 * 3600.0; callback = cb(cc === c ? :subs : :nosubs)); end
        s48 = screen(c); s48_0 = screen(c0)
        for cc in (c, c0); sbl_run!(cc, 54 * 3600.0; callback = cb(cc === c ? :subs : :nosubs)); end
        z54 = zinv(c); z54_0 = zinv(c0)
        # the passive descent dz/dt = w_s(z) of a level that starts at the
        # 18 LT (42 h) inversion, after the evening transition
        W = only(f for f in c.forcing if f isa Subsidence).w_s
        zp = z42
        for t in 42 * 3600.0:10.0:(54 * 3600.0 - 10)
            k1 = forcing_value(W, zp, t); k2 = forcing_value(W, zp + 5k1, t + 5)
            k3 = forcing_value(W, zp + 5k2, t + 5); k4 = forcing_value(W, zp + 10k3, t + 10)
            zp += 10 * (k1 + 2k2 + 2k3 + k4) / 6
        end
        dz = c.sim.mesh.dz
        content = sbl_column_heat(c)
        ref = gabls2_reference()
        rows = [r for r in ref.rows if (startswith(r[1], "obs_") || startswith(r[1], "les_")) && 42 <= r[2] <= 54]
        val(s, q) = endswith(q, "T2m") ? s.T2 : s.U10
        inside = [(r[1], r[3] <= val(s48, r[1]) <= r[4], r[3] <= val(s48_0, r[1]) <= r[4]) for r in rows]
        @info @sprintf("GABLS2 (16×16×40): residual-layer inversion %.1f m at 40 h; at 06 LT 24 Oct (54 h) %.1f m with subsidence, %.1f m without (change %.1f m); passive descent from the 42 h inversion (%.1f m) → %.1f m (model − passive %.1f m = %.2f Δz)",
                       z40, z54, z54_0, z54 - z54_0, z42, zp, z54 - zp, (z54 - zp) / dz)
        @info @sprintf("GABLS2 (16×16×40) at 48 h: T2 %.2f °C with / %.2f °C without subsidence, U10 %.2f / %.2f m/s; envelopes (inside with, without): %s",
                       s48.T2, s48_0.T2, s48.U10, s48_0.U10, join(("$(q) $(a ? "in" : "out")/$(b ? "in" : "out")" for (q, a, b) in inside), ", "))
        @info @sprintf("GABLS2 (16×16×40) θ ledger: max |Δ∫⟨θ⟩dz − surface − advective − subsidence| %.1e K m (with), %.1e K m (without) = %.1e of the column content %.0f K m; booked subsidence heating %.2f K m (%.3f K column mean)",
                       res[:subs], res[:nosubs], res[:subs] / content, content, c.ls_heat_applied, c.ls_heat_applied / c.sim.mesh.Lz)
        @test res[:subs] < 1e-11 * content && res[:nosubs] < 1e-11 * content
        @test c.ls_heat_applied > 0 && c0.ls_heat_applied == 0   # descent into a stable column warms it
        @test z42_0 == z40 || abs(z42_0 - z40) < dz              # without: the residual inversion stays
        @test abs(z54_0 - z40) < dz
        @test z54 < z54_0 - 3dz                                  # with: it descends …
        @test abs(z54 - zp) < dz                                 # … as a passive surface, within one Δz
        # night 2 against the digitised envelopes: subsidence moves the
        # screen values by less than the digitisation half-widths and does
        # not change which envelopes the run is inside
        @test abs(s48.T2 - s48_0.T2) < 0.7 && abs(s48.U10 - s48_0.U10) < 0.4
        @test all(a == b for (_, a, b) in inside)
    end
end
