# The update of the SGS TKE as one cache-blocked pass (#554, part 2):
# `advance_sgs!(::TKEClosure)` on the blocked driver of #533
# (src/prognostic_sgs/tke_blocked.jl) against the separate loops it
# replaces (`_advance_separate!`, the reference, what loop_fusion =
# "never" runs).
#
# Every value of the pass is computed by the point body of the separate
# loop, so e is expected to agree to round-off; the budget sums are the
# same row sums combined in the same order. Tolerances: e within 4 eps
# of max |e| (a different vectorisation may reorder a row's sum; measured
# 0 in every case), the budget sums within a relative 1e-12 (the same
# row sums; measured ≤ 2.3e-16), the cancelling transport sum on the
# scale of the shear production.

using Test
using Aeolus
using Aeolus.Fields: interior_view
using Aeolus.Operators: advance_sgs!
using Aeolus.Backends: CPUBackend, with_backend
using Aeolus.IBM: SolidMask, mark_solid!
using Aeolus.WallModels: immersed_walls, RoughLogLaw
using Random: Xoshiro, randn

const TKB = Aeolus.PrognosticSGS

# A closure on a random flow: θ stably stratified with noise (N² of both
# signs), e with zeros and a large range (the limiter active), a first
# level, wall distance; optionally a solid mask with wall-modelled
# immersed walls and a surface value.
function tkb_case(g; transport = true, dissipation = :semi_implicit, mask = false, walls = false,
                  surface = false, bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
    sim = BoussinesqSimulation(g; θ₀ = 300.0, bcs = bcs, θ_flux_bottom = 0.05)
    rng = Xoshiro(554)
    n = (g.Nx, g.Ny, g.Nz)
    for f in (sim.u, sim.v, sim.w)
        interior_view(f) .= 2.0 .* randn(rng, size(interior_view(f)))
    end
    interior_view(sim.θ) .= 300 .+ 0.02 .* randn(rng, n) .+ reshape(0.004 .* (1:g.Nz), 1, 1, :)
    sm = nothing
    if mask
        sm = SolidMask(g)
        mark_solid!(sm, (x, y, z) -> (0.3 < x / (g.Nx * g.dx) < 0.5) && (0.2 < y / (g.Ny * g.dy) < 0.6) &&
                                     z < 0.4 * g.Nz * g.dz)
    end
    fl = falses(n); fl[:, :, 1] .= true
    sv = surface ? (u_star = 0.3 .+ 0.1 .* rand(rng, g.Nx, g.Ny), k = ones(Int, g.Nx, g.Ny)) : nothing
    c = TKEClosure(g; mask = sm === nothing ? nothing : sm.mask, transport = transport,
                   dissipation = dissipation, first_level = fl, surface = sv,
                   wall_distance = 1.0 .+ 30 .* rand(rng, n...),
                   walls = walls ? immersed_walls(sim, sm, RoughLogLaw(0.1)) : nothing)
    initialize_tke!(c, sim; e_min = 1e-3)
    interior_view(c.e) .*= 4 .* rand(rng, n...) .* (rand(rng, n...) .> 0.15)
    return sim, c
end

# e and the statistics after one update by `f!`, from the same start. The
# scratch the blocked pass reads at the boundary (ν, κt, the limiter's
# ratio) is first set to NaN: a reference run on the same closure leaves
# correct values in their interior, which would hide a boundary cell the
# pass failed to compute itself (a NaN read reaches e).
function tkb_run!(f!, c, sim, e0, Δt)
    copyto!(c.e.data, e0); fill!(c.stats, 0)
    fill!(c.ν.data, NaN); fill!(c.κt.data, NaN)
    c.ws === nothing || fill!(c.ws.ratio.data, NaN)
    f!(c, sim, Δt)
    return copy(c.e.data), tke_statistics(c)
end

tkb_stats_agree(a, b) = a.clips == b.clips && a.steps == b.steps &&
    all(isapprox(getfield(a, f), getfield(b, f); rtol = 1e-12, atol = 1e-12 * abs(a.shear))
        for f in (:clipped_energy, :surface_energy, :shear, :buoyancy, :dissipation,
                  :buoyant_destruction, :transport, :wall))

@testset "TKE update as one cache-blocked pass (#554)" begin
    n = (14, 10, 40)        # 40 levels: 5 slabs of the WENO-Z5 pass (4 edge planes per end)
    rg = RegularGrid(; size = n, extent = (140.0, 100.0, 200.0), halo = (3, 3, 3))
    rg2 = RegularGrid(; size = n, extent = (140.0, 100.0, 200.0), halo = (2, 2, 2))   # centred
    sg = StretchedGrid(; size = n, extent = (140.0, 100.0, 400.0), halo = (2, 2, 2), stretching = (:tanh, 1.5))
    Δt = 0.4
    nlim = Ref(0)            # cases in which the limiter acted (ratio < 1 somewhere)
    for (gname, g) in (("regular WENO-Z5", rg), ("regular centred", rg2), ("stretched", sg)),
        transport in (true, false), diss in (:semi_implicit, :exact), solid in (false, true)
        solid && g isa StretchedGrid && continue      # the mask is RegularGrid-only
        sim, c = tkb_case(g; transport = transport, dissipation = diss, mask = solid,
                          walls = solid && g === rg, surface = true)
        e0 = copy(c.e.data)
        ref_e, ref_s = tkb_run!(TKB._advance_separate!, c, sim, e0, Δt)
        @test minimum(ref_e) >= 0
        transport && (nlim[] += any(<(1), interior_view(c.ws.ratio)))
        scale = maximum(abs, ref_e)
        lz = TKB.tke_blocked_lags(c)[2]
        for (bname, b) in (("serial", CPUBackend(serial = true)), ("threaded", CPUBackend(threshold = 0))),
            ns in 1:5, rev in (false, true)
            (ns > 1 && 2lz * ns > g.Nz) && continue
            (rev && ns == 1) && continue
            e, s = with_backend(b) do
                tkb_run!((c, sim, Δt) -> TKB._advance_blocked!(c, sim, Δt; ns = ns, reverse_slabs = rev),
                         c, sim, e0, Δt)
            end
            ok_e = maximum(abs.(e .- ref_e)) <= 4eps() * scale
            ok_s = tkb_stats_agree(s, ref_s)
            ok_e && ok_s || @info "TKE blocked pass disagrees" gname transport diss solid bname ns rev maximum(abs.(e .- ref_e)) s ref_s
            @test ok_e
            @test ok_s
        end
    end
    @test nlim[] > 0          # the limiter was exercised
end

@testset "TKE blocked pass: no-flux lateral faces, a short column (#554)" begin
    for (bcs, n) in ((FieldBoundaryConditions(:noflux, :noflux, :wall), (9, 7, 12)),
                     (FieldBoundaryConditions(:periodic, :periodic, :wall), (6, 1, 3)))
        g = RegularGrid(; size = n, extent = (90.0, 70.0, 60.0), halo = (3, 3, 3))
        sim, c = tkb_case(g; bcs = bcs)
        e0 = copy(c.e.data)
        ref_e, ref_s = tkb_run!(TKB._advance_separate!, c, sim, e0, 0.4)
        for ns in (1, 2)
            (ns > 1 && 8ns > g.Nz) && continue
            e, s = tkb_run!((c, sim, Δt) -> TKB._advance_blocked!(c, sim, Δt; ns = ns), c, sim, e0, 0.4)
            @test maximum(abs.(e .- ref_e)) <= 4eps() * maximum(abs, ref_e)
            @test tkb_stats_agree(s, ref_s)
        end
    end
end

@testset "TKE blocked pass: open lateral faces and top (#554)" begin
    # an open box (boundary data, a wall below): ν and the ratio have no
    # boundary data, so their open-face halos are the mirror of the first
    # interior layer, which the pass's boundary shell reproduces
    for halo in (2, 3)
        g = RegularGrid(; size = (12, 10, 16), extent = (120.0, 100.0, 160.0), halo = (halo, halo, halo))
        bcs = open_boundary_conditions(g, ProfileBoundaryData(u = 3.0, v = -1.0, w = 0.0, θ = 300.0); halo = halo)
        sim, c = tkb_case(g; bcs = bcs)
        Aeolus.BoundaryConditions.update_open_boundaries!(sim, 0.0, 0.4)
        @test TKB._tke_blocked_blocker(c, sim) === nothing
        e0 = copy(c.e.data)
        ref_e, ref_s = tkb_run!(TKB._advance_separate!, c, sim, e0, 0.4)
        for ns in (1, 2), rev in (false, true)
            (ns > 1 && 2 * TKB.tke_blocked_lags(c)[2] * ns > g.Nz) && continue
            e, s = tkb_run!((c, sim, Δt) -> TKB._advance_blocked!(c, sim, Δt; ns = ns, reverse_slabs = rev),
                            c, sim, e0, 0.4)
            @test maximum(abs.(e .- ref_e)) <= 4eps() * maximum(abs, ref_e)
            @test tkb_stats_agree(s, ref_s)
        end
    end
end

@testset "TKE blocked pass: advance_sgs!, loop_fusion, report, allocation (#554)" begin
    g = RegularGrid(; size = (16, 12, 16), extent = (160.0, 120.0, 160.0), halo = (3, 3, 3))
    sim, c = tkb_case(g)
    e0 = copy(c.e.data)
    # the default (if_possible) runs the blocked pass, "never" the separate loops
    e_b, s_b = tkb_run!(advance_sgs!, c, sim, e0, 0.4)
    Aeolus.Simulations.set_loop_fusion!(sim, :never)
    e_s, s_s = tkb_run!(advance_sgs!, c, sim, e0, 0.4)
    @test maximum(abs.(e_b .- e_s)) <= 4eps() * maximum(abs, e_s)
    @test tkb_stats_agree(s_b, s_s)
    r = Aeolus.Simulations.sweep_report(sim)
    @test r["tke_update_blocked"] == "some" && !haskey(r, "tke_not_blocked")
    # allocation: the pass reuses its scratch (measured: as many bytes as the
    # separate loops' launches; bounded at 64 KiB per call on a serial backend)
    Aeolus.Simulations.set_loop_fusion!(sim, :if_possible)
    with_backend(CPUBackend(serial = true)) do
        advance_sgs!(c, sim, 0.4)
        a = @allocated advance_sgs!(c, sim, 0.4)
        @test a <= 64 * 1024
    end
    sim2, c2 = tkb_case(g)
    advance_sgs!(c2, sim2, 0.4)
    @test Aeolus.Simulations.sweep_report(sim2)["tke_update_blocked"] == "all"
end
