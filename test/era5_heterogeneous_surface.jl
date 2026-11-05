# Phase 53 — heterogeneous-surface coupling (issue #43): the code paths of
# the patchy-surface LES harness (test/support/patchy_surface.jl). Over a
# uniform surface the resolved, tiled and parameter-aggregated modes are
# the same surface and give the same flow; the column heat content changes
# by exactly the applied surface flux (40 steps of 5 s, 32 × 16 × 16).
#
# The heterogeneous runs (resolved LES against tiles and parameter
# aggregation, the λ/z_i dependence, a background wind) are validation:
# validation/era5_heterogeneous_surface.jl.

using Test, Printf
using Aeolus

isdefined(@__MODULE__, :patchy_les) || include(joinpath(@__DIR__, "support", "patchy_surface.jl"))

@testset "Era V — heterogeneous-surface coupling (#43)" begin

    @testset "Coupling: conservation and code-path identity" begin
        # Over a uniform surface (both strip types at 305 K) the three modes
        # — column surfaces, a two-tile mosaic per column, one aggregated
        # surface per column — describe the same surface and must give the
        # same flow.
        runs = Dict(mode => patchy_les(; HS_GRID..., λ = 800.0, T_hot = 305.0, T_cool = 305.0,
                                       mode = mode) for mode in (:resolved, :tiled, :parameter))
        h0 = Dict(k => patchy_heat_content(v) for (k, v) in runs)
        for _ in 1:40, pl in values(runs)
            patchy_step!(pl, 5.0)
        end
        θr = runs[:resolved].sim.θ.data
        worst = maximum(maximum(abs.(runs[m].sim.θ.data .- θr)) for m in (:tiled, :parameter))
        @info "Uniform surface: max |Δθ| between resolved, tiled and parameter runs after 200 s" worst
        @test worst < 1e-10
        # Column heat content changes by exactly the applied surface flux
        # (walls, no-flux θ, conservative advection and SGS diffusion).
        for (k, pl) in runs
            err = (patchy_heat_content(pl) - h0[k] - pl.applied_heat) / pl.applied_heat
            @test abs(err) < 1e-8
        end
        # The applied flux is warming: θ_s = 305 K above a 300 K mixed layer.
        @test runs[:resolved].applied_heat > 0
    end
end
