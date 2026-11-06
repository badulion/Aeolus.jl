# Phase 57 (P57-b, #183): validation of the log-layer mismatch fix
# — the matching height off the first cell (Kawai & Larsson 2012), the
# exponential time filter (Yang, Park & Moin 2017) and the 3 × 3
# wall-parallel filter (Bou-Zeid, Meneveau & Parlange 2005) — in simulated
# flows: the wall-modelled channel at Re_τ = 1000 (bulk velocity against
# Lee & Moser 2015, Φ_m against the log law) and the neutral rough half
# channel. The input filters and the matching height stay tested in
# test/wall_models_channel.jl.

using Test
using Printf
using Random
using Aeolus
# the wall-bounded benchmark harness (half channel, channel) is a test fixture (ONE-l, #475)
isdefined(@__MODULE__, :ChannelCases) || include(joinpath(@__DIR__, "..", "test", "fixtures", "channel_cases.jl"))
using .ChannelCases
using Aeolus.WallModels

isdefined(@__MODULE__, :lm2015_ub_plus) || include(joinpath(@__DIR__, "..", "test", "data", "lee_moser_2015.jl"))

@testset "log-layer mismatch: channel and half-channel runs (#183)" begin

# A channel run: U_b⁺ (mean over both halves), Φ_m of the averaged profile
# on the faces in 0.1δ < z < 0.3δ, the first-face Φ_m, ⟨τ_w⟩.
function llm_channel(; kw...)
    c = channel_case(; Re_τ = 1000, N = (24, 12, 16), closure = StaticSmagorinsky(0.1),
                     wall_law = SpaldingLaw(), seed = 1, kw...)
    st = run_channel!(c, 8.0; average_from = 5.0)
    sel = findall(z -> 0.1 < z < 0.3, st.zf)
    return (Ub = sum(st.U) / length(st.U), Φ = st.phi_m, Φmax = maximum(abs.(st.phi_m[sel] .- 1)),
            τ = st.τ_wall, t = st.wall_time)
end

@testset "Channel Re_τ = 1000 (24×12×16, Smagorinsky 0.1, Spalding): matching height + time filter" begin
    Ub_LM = lm2015_ub_plus(1000)
    dz = 2 / 16
    a = llm_channel()                                          # first-cell matching
    b = llm_channel(; h_wm = 3dz, filter = :temporal)          # Kawai-Larsson + Yang et al.
    ea, eb = a.Ub / Ub_LM - 1, b.Ub / Ub_LM - 1
    @info @sprintf("  channel Re_τ=1000: first cell U_b⁺ %.3f (%+.2f %%), Φ_m(Δz) %.3f, max|Φ_m−1| %.3f, ⟨τ_w⟩ %.3f (%.1f s)",
                   a.Ub, 100ea, a.Φ[1], a.Φmax, a.τ, a.t)
    @info @sprintf("  channel Re_τ=1000: h_wm = 3Δz + T_f  U_b⁺ %.3f (%+.2f %%), Φ_m(Δz) %.3f, max|Φ_m−1| %.3f, ⟨τ_w⟩ %.3f (%.1f s)",
                   b.Ub, 100eb, b.Φ[1], b.Φmax, b.τ, b.t)
    # the momentum balance holds either way: ⟨τ_w⟩ = u_τ²
    @test abs(a.τ - 1) < 0.05 && abs(b.τ - 1) < 0.05
    # the log-layer mismatch of first-cell matching: Φ_m overshoots at the
    # first face (Kawai & Larsson 2012); the fix brings the profile onto the
    # log law over 0.1δ < z < 0.3δ. Seeds 1-6 (5-8 δ/u_τ): first-cell
    # Φ_m(Δz) 1.26-1.44 (the issue's "> 1.3" in 5 of 6), fixed 0.97-1.18;
    # max|Φ_m−1| fixed 0.04-0.22. The test asserts what holds for all six.
    @test a.Φ[1] > 1.2
    @test a.Φ[1] - b.Φ[1] > 0.1
    @test b.Φmax < 0.25
    @test b.Φmax < a.Φmax
    # U_b⁺ against Lee & Moser (2015)
    @test abs(eb) < 0.05
    # EXPECTATION THAT PROVED WRONG (kept as a broken test): "the U_b⁺
    # error is reduced at least 2×". On this grid first-cell matching is
    # already within 1 % of Lee & Moser (−0.7 % here; −0.75 to +0.02 % over
    # seeds 1-6, fixed −1.18 to −1.57 %): its mismatch is a kink in the two
    # cells next to the wall (U⁺ − U⁺_LM = −0.51, +0.21 at z = 0.06δ,
    # 0.19δ), which nearly cancels in the bulk integral, while U_b⁺ is set
    # by the outer-layer deficit of the coarse LES (−0.8 at the centre)
    # that no wall-model input changes. At Re_τ = 182, where the first
    # cell centre is at y⁺ = 11, the factor is met: +5.6 % → −1.2 %
    # (the former examples/validation/log_layer_mismatch.jl, 10-20 δ/u_τ).
    @test_broken abs(eb) <= abs(ea) / 2
end

@testset "Neutral rough half channel (24³, RoughLogLaw): h_wm + filtering vs first cell" begin
    function abl(; kw...)
        c = half_channel_case(; N = 24, seed = 1, kw...)
        st = run_channel!(c, 12.0; average_from = 6.0)
        sel = findall(z -> z < 0.3, st.zf)
        return (Φ1 = st.phi_m[1], peak = maximum(st.phi_m[sel]), τ = st.τ_wall, t = st.wall_time)
    end
    a = abl()
    b = abl(; h_wm = 3 / 24, filter = :both)
    @info @sprintf("  half channel: first cell Φ_m(Δz) %.3f, peak Φ_m (z < 0.3H) %.3f, ⟨τ_w⟩ %.3f (%.1f s)", a.Φ1, a.peak, a.τ, a.t)
    @info @sprintf("  half channel: h_wm = 3Δz, time + 3×3 filter  Φ_m(Δz) %.3f, peak Φ_m %.3f, ⟨τ_w⟩ %.3f (%.1f s)", b.Φ1, b.peak, b.τ, b.t)
    # (6 H/u* of averaging: seeds 1-4 give 0.947-1.031; the ±5 % of
    # neutral_abl.jl needs its 12 H/u* window)
    @test abs(a.τ - 1) < 0.07 && abs(b.τ - 1) < 0.07
    # the overshoot at the first face, the part the wall-model input sets,
    # is lowered (1.19 → 1.11 here; by 0.07-0.10 over seeds 1-4)
    @test b.Φ1 < a.Φ1 - 0.05
    # EXPECTATION THAT PROVED WRONG (broken test): "h_wm plus filtering
    # lowers the Φ_m overshoot". The peak, 1.68-1.69 at z ≈ 3Δz, is the
    # Smagorinsky overshoot of the closure itself (Mason & Thomson 1992),
    # not of the wall model: with the fix it is 1.694 / 1.659 / 1.616 /
    # 1.738 (seeds 1-4) against 1.685 / 1.682 / 1.718 / 1.653, a change
    # within the seed scatter.
    # Only an under-resolved 16³ run, where τ_w is 0.80-0.92 u*² with
    # first-cell matching, sees it drop (2.62 → 1.73, 2.03 → 1.87).
    @test_broken b.peak < a.peak
end

end
