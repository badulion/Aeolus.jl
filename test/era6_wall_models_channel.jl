# Era VI, Phase 57 (P57-b, #183): log-layer mismatch. The wall-model
# input — matching height off the first cell (Kawai & Larsson 2012), the
# exponential time filter (Yang, Park & Moin 2017) and the 3 × 3
# wall-parallel filter (Bou-Zeid, Meneveau & Parlange 2005) — and its
# effect on the wall-modelled channel at Re_τ = 1000 and on the neutral
# rough half channel.
#
# Physics asserted: the time filter's exact exponential step response;
# the wall-parallel filter's plane-mean conservation and exact transfer
# function; exact interpolation to the matching height; the reference
# data against Dean's (1978) correlation. The channel at Re_τ = 1000 and
# the rough half channel are validation runs, in
# validation/era6_wall_models_channel.jl.

using Test
using Printf
using Random
using Aeolus
# the wall-bounded benchmark harness (half channel, channel) is a test fixture (ONE-l, #475)
isdefined(@__MODULE__, :ChannelCases) || include(joinpath(@__DIR__, "fixtures", "channel_cases.jl"))
using .ChannelCases
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

end
