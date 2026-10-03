# Era VI, Phase 57 (P57-b, #183): log-layer mismatch. The wall-model
# input — matching height off the first cell (Kawai & Larsson 2012), the
# exponential time filter (Yang, Park & Moin 2017) and the 3 × 3
# wall-parallel filter (Bou-Zeid, Meneveau & Parlange 2005) — and its
# effect on the wall-modelled channel at Re_τ = 1000 and on the neutral
# rough half channel.
#
# Physics asserted: the time filter's exact exponential step response;
# the wall-parallel filter's plane-mean conservation and exact transfer
# function; exact interpolation to the matching height; the channel's
# bulk velocity against Lee & Moser (2015) and its Φ_m against the log
# law; the reference data against Dean's (1978) correlation.

using Test
using Printf
using Random
using Aeolus
using Aeolus.WallModels

include(joinpath(@__DIR__, "data", "lee_moser_2015.jl"))

@testset "Era VI — log-layer mismatch: matching height, input filters (#183)" begin

# periodic in x, y; free-slip z-walls
mi_sim(; N = (12, 8, 10), extent = (6.0, 4.0, 2.0)) =
    BoussinesqSimulation(RegularGrid(; size = N, extent = extent, halo = (2, 2, 2));
                         bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))

@testset "Time filter: exact exponential response to a step (1e-12)" begin
    @test exponential_filter_weight(0.3, 0.0) == 1.0
    @test exponential_filter_weight(1e-9, 2.0) ≈ 5e-10 rtol = 1e-9     # expm1: no cancellation
    sim = mi_sim()
    T_f = 0.7
    mi = MatchingInput(; T_f = T_f)
    Aeolus.Fields.interior_view(sim.u) .= 0.0
    update_input_filter!(mi, sim, 0.1)               # primes with the current sample: 0
    @test mi.primed && all(==(0.0), mi.uf)
    Aeolus.Fields.interior_view(sim.u) .= 1.0        # the step
    rng = Random.Xoshiro(4)
    t = 0.0; err = 0.0
    for n in 1:60
        dt = 0.02 + 0.2 * rand(rng)                  # any step sequence
        update_input_filter!(mi, sim, dt)
        t += dt
        err = max(err, maximum(abs.(mi.uf .- (1 - exp(-t / T_f)))))
    end
    @info @sprintf("  time filter: max |ũ − (1 − e^{-t/T_f})| over 60 random steps = %.1e (t = %.2f T_f)", err, t / T_f)
    @test err < 1e-12
    # the law sees the filtered speed, unchanged by the substages of a step
    w, h = matching_input!(mi, sim)
    @test all(w .≈ 1 - exp(-t / T_f)) && all(==(0.1), h)
    Aeolus.Fields.interior_view(sim.u) .= 5.0
    @test matching_input!(mi, sim)[1] == w
    @test_throws ArgumentError MatchingInput(; T_f = -1)
end

@testset "Wall-parallel filter: plane mean preserved, exact box transfer function" begin
    sim = mi_sim()
    g = sim.mesh
    rng = Random.Xoshiro(11)
    Aeolus.Fields.interior_view(sim.u) .= randn(rng, g.Nx, g.Ny, g.Nz)
    Aeolus.Fields.interior_view(sim.v) .= randn(rng, g.Nx, g.Ny, g.Nz)
    uF, vF = Field(FaceX(), g), Field(FaceY(), g)
    wall_parallel_filter!(uF, vF, sim)
    for (f, F) in ((sim.u, uF), (sim.v, vF))
        a, b = Aeolus.Fields.interior_view(f), Aeolus.Fields.interior_view(F)
        @test maximum(abs.(sum(a; dims = (1, 2)) .- sum(b; dims = (1, 2)))) < 1e-12
        @test sum(abs2, b) < sum(abs2, a)            # it smooths
    end
    # a Fourier mode: the 3-point box multiplies it by (1 + 2cos kΔ)/3 per direction
    kx, ky = 2π / g.Lx * 2, 2π / g.Ly * 1
    xs = [(i - 1) * g.dx for i in 1:g.Nx]; ys = [(j - 0.5) * g.dy for j in 1:g.Ny]
    Aeolus.Fields.interior_view(sim.u) .= [sin(kx * x) * cos(ky * y) for x in xs, y in ys, _ in 1:g.Nz]
    Aeolus.Fields.interior_view(sim.v) .= 0.0
    wall_parallel_filter!(uF, vF, sim)
    G = (1 + 2cos(kx * g.dx)) / 3 * (1 + 2cos(ky * g.dy)) / 3
    @test maximum(abs.(Aeolus.Fields.interior_view(uF) .- G .* Aeolus.Fields.interior_view(sim.u))) < 1e-13
    # through MatchingInput: the input is the filtered field's
    mi = MatchingInput(; spatial_filter = true)
    w, _ = matching_input!(mi, sim)
    ref = [abs((Aeolus.Fields.interior_view(uF)[i, j, 1] +
                Aeolus.Fields.interior_view(uF)[mod1(i + 1, g.Nx), j, 1]) / 2) for i in 1:g.Nx, j in 1:g.Ny]
    @test maximum(abs.(w .- ref)) < 1e-13
    @test_throws ArgumentError matching_input!(mi, sim; mask = trues(g.Nx, g.Ny))
end

@testset "Matching height: exact interpolation, both walls, the stress at the wall face" begin
    sim = mi_sim()
    g = sim.mesh; dz = g.dz
    zc = [(k - 0.5) * dz for k in 1:g.Nz]
    Aeolus.Fields.interior_view(sim.u) .= reshape(1 .+ 3 .* zc, 1, 1, :)    # u = 1 + 3z
    Aeolus.Fields.interior_view(sim.v) .= reshape(-2 .* zc, 1, 1, :)        # v = −2z
    for (kw, h) in (((; h_wm_cells = 1), dz / 2), ((; h_wm_cells = 3), 2.5dz),
                    ((; h_wm = 0.53), 0.53), ((; h_wm = dz / 2), dz / 2))
        mi = MatchingInput(; kw...)
        w, hh = matching_input!(mi, sim)
        @test all(≈(h; rtol = 1e-14), hh)
        @test maximum(abs.(w .- hypot(1 + 3h, 2h))) < 1e-13
        # the top wall: distance from z = L_z
        wt, ht = matching_input!(mi, sim; side = :top)
        @test maximum(abs.(wt .- hypot(1 + 3(g.Lz - h), 2(g.Lz - h)))) < 1e-13
    end
    @test matching_distance(MatchingInput(; h_wm_cells = 4), dz) ≈ 3.5dz
    # per-column cells with a level mask (an immersed floor at level 2)
    cells = ones(Int, g.Nx, g.Ny); cells[1:4, :] .= 3
    lv = fill(2, g.Nx, g.Ny); lv[end, end] = 99
    w, hh = matching_input!(MatchingInput(; h_wm_cells = cells), sim; mask = lv)
    @test hh[1, 1] ≈ 2.5dz && hh[6, 1] ≈ 0.5dz && hh[end, end] == 0
    @test w[1, 1] ≈ hypot(1 + 3zc[4], 2zc[4]) && w[6, 1] ≈ hypot(1 + 3zc[2], 2zc[2])
    @test_throws ArgumentError matching_input!(MatchingInput(; h_wm = 0.4dz), sim)
    @test_throws ArgumentError matching_input!(MatchingInput(; h_wm_cells = 11), sim)
    @test_throws ArgumentError MatchingInput(; h_wm = 0.5, h_wm_cells = 2)
    # applied: the law at the matching height, the stress at the first level only
    law = RoughLogLaw(1e-3)
    mi = MatchingInput(; h_wm_cells = 3)
    u0 = copy(Aeolus.Fields.interior_view(sim.u))
    w, hh = matching_input!(mi, sim)
    F = apply_wall_stress!(sim, law, 1e-9; wind = w, height = hh)
    @test -F.Fx / (g.Lx * g.Ly) ≈ wall_stress(law, w[1], 2.5dz) * (1 + 3zc[1]) / hypot(1 + 3zc[1], 2zc[1]) rtol = 1e-6
    @test Aeolus.Fields.interior_view(sim.u)[:, :, 2:end] == u0[:, :, 2:end]
end


@testset "Reference data: Lee & Moser (2015) U_b⁺ against Dean (1978)" begin
    # from the published file headers, U_b⁺ = 1/(ν Re_τ) (test/data/lee_moser_2015.jl)
    @test [round(c.Ub_plus; digits = 2) for c in LM2015_CHANNEL] == [15.69, 18.40, 19.99, 24.10]
    # Dean's correlation is an independent fit to experiments: within 3.5 %
    for c in LM2015_CHANNEL
        @test abs(dean_ub_plus(c.Re_τ) / c.Ub_plus - 1) < 0.035
    end
    @test dean_ub_plus(1000) ≈ (sqrt(2 / 0.073) * 2000^(1 / 8))^(8 / 7)
end

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
    # log law over 0.1δ < z < 0.3δ
    @test a.Φ[1] > 1.3
    @test a.Φmax > 0.25
    @test b.Φmax < 0.25
    # U_b⁺ against Lee & Moser (2015)
    @test abs(eb) < 0.05
    # EXPECTATION THAT PROVED WRONG (kept as a broken test): "the U_b⁺
    # error is reduced at least 2×". On this grid first-cell matching is
    # already within 1 % of Lee & Moser (−0.7 % here; −0.45 to −1.1 % over
    # three seeds averaged 10-20 δ/u_τ): its mismatch is a kink in the two
    # cells next to the wall (U⁺ − U⁺_LM = −0.51, +0.21 at z = 0.06δ,
    # 0.19δ), which nearly cancels in the bulk integral, while U_b⁺ is set
    # by the outer-layer deficit of the coarse LES (−0.8 at the centre)
    # that no wall-model input changes. The fixed run is −1.4 to −1.5 %.
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
    @test abs(a.τ - 1) < 0.05 && abs(b.τ - 1) < 0.05
    # the overshoot at the first face, the part the wall-model input sets,
    # is lowered (1.19 → 1.11 here, both seeds 1 and 2)
    @test b.Φ1 < a.Φ1 - 0.05
    # EXPECTATION THAT PROVED WRONG (broken test): "h_wm plus filtering
    # lowers the Φ_m overshoot". The peak, 1.68-1.69 at z ≈ 3Δz, is the
    # Smagorinsky overshoot of the closure itself (Mason & Thomson 1992),
    # not of the wall model: with the fix it is 1.694 (seed 1) and 1.659
    # (seed 2) against 1.685 / 1.682, a change within the seed scatter.
    # Only an under-resolved 16³ run, where τ_w is 0.80-0.92 u*² with
    # first-cell matching, sees it drop (2.62 → 1.73, 2.03 → 1.87).
    @test_broken b.peak < a.peak
end

end
