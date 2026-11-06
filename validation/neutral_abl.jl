# Phase 54 (P54-d, #174): validation runs of the wall-bounded
# benchmark harness. The pressure-driven half channel (24³, Smagorinsky
# 0.1) reaches the momentum budget of Porté-Agel et al. (2000): wall stress
# = u*², total stress = u*²(1 - z/H); with Smagorinsky 0.17 and no damping
# it reproduces the near-wall Φ_m overshoot of Mason & Thomson (1992). The
# two-wall channel at Re_τ = 1000 reaches ⟨τ_w⟩ = u_τ² (Moin & Kim 1982).
# The harness and its diagnostics stay tested in test/neutral_abl.jl.

using Test
using Printf
using Random
using Aeolus
# the wall-bounded benchmark harness (half channel, channel) is a test fixture (ONE-l, #475)
isdefined(@__MODULE__, :ChannelCases) || include(joinpath(@__DIR__, "..", "test", "fixtures", "channel_cases.jl"))
using .ChannelCases
using Aeolus.SurfaceExchange: von_karman
using Aeolus.WallModels

@testset "neutral ABL / channel benchmarks (#174)" begin

# ---------------------------------------------------------------------------
# The Phase 54 benchmarks (Porté-Agel et al. 2000; Moin & Kim 1982)
# ---------------------------------------------------------------------------

@testset "Half channel 24³, Smagorinsky 0.1: wall stress = u*², τ(z) = u*²(1 - z/H)" begin
    # The issue planned "about 10 H/u*, average over the last 5". That
    # WINDOW was too short, not the code: the 5-10 H/u* mean wall stress
    # over 10 seeded runs ranged 0.946-1.027 u*² (0.922-0.985 over 12 more
    # with noise = 5; the flow still accelerates toward its Smagorinsky
    # equilibrium, and 5 H/u* averages of a 24³ box scatter by ±3 %), so a
    # ±5 % test on it fails for some seeds (seed 4: 0.946). Averaged over 8-20 H/u* the same quantity was 0.961-0.991
    # (12 seeds) and max|τ_tot - u*²(1 - z/H)| ≤ 0.040 u*². The tolerances
    # (±5 %, ±10 %) are the issue's.
    t0 = time()
    hc = half_channel_case(; N = 24, closure = StaticSmagorinsky(0.1))
    run_channel!(hc, 8.0)
    st = run_channel!(hc, 20.0; average_from = 8.0)
    ref = 1 .- st.stress.z
    dev = maximum(abs.(st.stress.total .- ref))
    @info @sprintf("  half channel 24³, C_s = 0.1 (%d steps, %.1f s): ⟨τ_w⟩ = %.4f u*² over 8-20 H/u*; max|τ_tot - u*²(1 - z/H)| = %.3f u*²",
                   st.nsteps, st.wall_time, st.τ_wall, dev)
    @info "  τ_tot / resolved / sgs at z/H = 0, 0.25, 0.5, 0.75: " *
          join((@sprintf("%.3f/%.3f/%.3f", st.stress.total[k], st.stress.resolved[k], st.stress.sgs[k]) for k in (1, 7, 13, 19)), ", ")
    @test abs(st.τ_wall - 1) < 0.05
    # ±10 % of the wall value u*² at every level (a relative tolerance is
    # meaningless near the free-slip top, where τ → 0)
    @test dev < 0.10
    @test st.stress.total[end] == 0                  # nothing leaves through the free-slip top
    @test st.stress.resolved[13] > st.stress.sgs[13] # resolved LES: mid-depth flux is resolved

    # Known failure reproduced: StaticSmagorinsky(0.17), no damping function.
    # From the developed C_s = 0.1 state (at 24³ a C_s = 0.17 start from
    # noise stays laminar-like for > 40 H/u*, with τ_w = 0.68-0.93 u*²),
    # 10 H/u* more, averaged over the last 5. Over 12 seeds max Φ_m of the
    # first three faces was 1.45-1.65, and ⟨τ_w⟩ 0.91-1.06 u*² (the flow
    # re-adjusts to the more dissipative closure).
    hc.closure = StaticSmagorinsky(0.17)
    s2 = run_channel!(hc, 30.0; average_from = 25.0)
    Φ3 = maximum(s2.phi_m[1:3])
    @info @sprintf("  C_s = 0.17, no damping: Φ_m at z/H = 1/24, 2/24, 3/24 = %.3f, %.3f, %.3f (max %.3f); max below z = H/4: %.3f; ⟨τ_w⟩ = %.4f u*²",
                   s2.phi_m[1], s2.phi_m[2], s2.phi_m[3], Φ3, maximum(s2.phi_m[1:6]), s2.τ_wall)
    @test Φ3 > 1.3        # Mason & Thomson 1992; Porté-Agel et al. 2000 Fig. 7: ≈ 1.5-2
    @test abs(s2.τ_wall - 1) < 0.15
    @info @sprintf("  half-channel benchmark wall time: %.1f s", time() - t0)
end

@testset "Two-wall channel 16×8×16 at Re_τ = 1000: ⟨τ_w⟩ = u_τ²" begin
    ch = channel_case(; Re_τ = 1000, N = (16, 8, 16))
    run_channel!(ch, 2.0)                               # 2 δ/u_τ from the law profile + noise
    st = run_channel!(ch, 10.0; average_from = 2.0)
    @info @sprintf("  channel 16×8×16, Re_τ = 1000, C_s = 0.065: ⟨τ_w⟩ over both walls, 2-10 δ/u_τ = %.4f u_τ² (bottom %.4f, top %.4f); %d steps",
                   st.τ_wall, st.τ_wall_bottom, st.τ_wall_top, st.nsteps)
    @test abs(st.τ_wall - 1) < 0.02
    # the total stress is antisymmetric, u_τ²(1 - z/δ), within 0.1 u_τ²
    @test maximum(abs.(st.stress.total .- (1 .- st.stress.z))) < 0.10
end

end # top testset
