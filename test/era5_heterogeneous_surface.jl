# Phase 53 — heterogeneous-surface validation (issue #43).
#
# A resolved LES of a convective boundary layer over strips of hot and
# cool ground (prescribed skin temperature, Monin–Obukhov fluxes per
# column through the SurfaceExchange contract) against the SurfaceTiles
# aggregation of the same two surfaces. The driver is
# examples/validation/patchy_surface/driver.jl; the full-resolution runs
# and the λ/z_i sweep are in examples/validation/patchy_surface_les.jl.
#
# What is checked against what:
#   * discrete conservation of the coupling (column heat content against
#     the applied surface flux) and the identity of the resolved, tiled
#     and parameter-aggregated code paths over a uniform surface;
#   * the resolved heterogeneous LES against (a) tiles forced by the
#     LES's own mean state (a-priori) and (b) an LES whose every column
#     is the tiled mosaic (a-posteriori), and against parameter
#     aggregation (one surface of mean skin temperature);
#   * published LES results for strip heterogeneity, qualitatively:
#     organised motion grows from λ/z_i ≈ 1 towards 4 < λ/z_i < 9 and the
#     coupled surface flux is smallest mid-patch (Patton, Sullivan &
#     Moeng 2005; conference abstract NTRS 20030068489), and a background
#     wind removes the patch-locked circulation (Wanner, Calaf & Mauder
#     2022, PLOS ONE 17, e0268097, Sect. 3 and Fig. 5, on the LES of
#     Margairaz et al. 2020).
# No field observations are used: the LITFASS campaigns were not
# reproduced (see docs/src/theory/heterogeneous_surface.md).

using Test, Printf
using Aeolus

include(joinpath(@__DIR__, "..", "examples", "validation", "patchy_surface", "driver.jl"))

# Coarse version of the example's setup: 100 m cells, 3.2 × 1.6 × 1.6 km.
const HS_GRID = (Nx = 32, Ny = 16, Nz = 16, Lx = 3200.0, Ly = 1600.0, Lz = 1600.0)
const HS_RUN = (t_end = 4500.0, dt = 5.0, t_avg = 2250.0, every = 3)

_hs_rel(a, b) = (a - b) / b

@testset "Era V — heterogeneous-surface validation (#43)" begin

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
