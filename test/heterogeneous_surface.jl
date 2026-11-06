# Phase 53 — heterogeneous-surface coupling (issue #43): the code paths of
# the patchy-surface LES harness (test/support/patchy_surface.jl). Over a
# uniform surface the resolved, tiled and parameter-aggregated modes are
# the same surface and give the same flow; the column heat content changes
# by exactly the applied surface flux (10 steps of 5 s, 16 × 8 × 8).
#
# The heterogeneous runs (resolved LES against tiles and parameter
# aggregation, the λ/z_i dependence, a background wind) are validation:
# validation/heterogeneous_surface.jl.

using Test
using Aeolus

isdefined(@__MODULE__, :patchy_les) || include(joinpath(@__DIR__, "support", "patchy_surface.jl"))

@testset "heterogeneous-surface coupling (#43)" begin

    @testset "Coupling: conservation and code-path identity" begin
        # Over a uniform surface (both strip types at 305 K) the three modes
        # — column surfaces, a two-tile mosaic per column, one aggregated
        # surface per column — describe the same surface and must give the
        # same flow.
        # half of HS_GRID in each direction: the identity and the budget are
        # exact properties of the coupling, independent of resolution
        grid = (Nx = 16, Ny = 8, Nz = 8, Lx = 3200.0, Ly = 1600.0, Lz = 1600.0)
        runs = Dict(mode => patchy_les(; grid..., λ = 800.0, T_hot = 305.0, T_cool = 305.0,
                                       mode = mode) for mode in (:resolved, :tiled, :parameter))
        h0 = Dict(k => patchy_heat_content(v) for (k, v) in runs)
        for _ in 1:10, pl in values(runs)
            patchy_step!(pl, 5.0)
        end
        θr = runs[:resolved].sim.θ.data
        worst = maximum(maximum(abs.(runs[m].sim.θ.data .- θr)) for m in (:tiled, :parameter))
        @test worst < 1e-10
        # Column heat content changes by exactly the applied surface flux
        # (walls, no-flux θ, conservative advection and SGS diffusion).
        @test maximum(abs((patchy_heat_content(pl) - h0[k] - pl.applied_heat) / pl.applied_heat)
                      for (k, pl) in runs) < 1e-8
        # The applied flux is warming: θ_s = 305 K above a 300 K mixed layer.
        @test runs[:resolved].applied_heat > 0
    end
end
