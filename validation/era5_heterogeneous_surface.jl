# Phase 53 — heterogeneous-surface validation (issue #43).
#
# A resolved LES of a convective boundary layer over strips of hot and
# cool ground (prescribed skin temperature, Monin–Obukhov fluxes per
# column through the SurfaceExchange contract) against the SurfaceTiles
# aggregation of the same two surfaces. The harness is
# test/support/patchy_surface.jl (moved from examples/validation/ by
# ONE-g, #456: the patchy-surface example, its full-resolution runs and the
# λ/z_i sweep, waits for a heterogeneous-surface component of the model).
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
#     Moeng 2005: the λ/z_i range is quoted from the conference abstract
#     NTRS 20030068489, the mid-patch statement from the J. Atmos. Sci.
#     abstract), and a background
#     wind removes the patch-locked circulation (Wanner, Calaf & Mauder
#     2022, PLOS ONE 17, e0268097, Sect. 3 and Fig. 5, on the LES of
#     Margairaz et al. 2020).
# No field observations are used: the LITFASS campaigns were not
# reproduced (see docs/src/theory/heterogeneous_surface.md).

using Test, Printf
using Aeolus

include(joinpath(@__DIR__, "..", "test", "support", "patchy_surface.jl"))

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

    # The five runs of the comparison (≈ 25 s each).
    cases = [(:hom,    (λ = 800.0,  T_hot = 305.0, T_cool = 305.0)),
             (:het1,   (λ = 800.0,  T_hot = 310.0, T_cool = 300.0)),
             (:het4,   (λ = 3200.0, T_hot = 310.0, T_cool = 300.0)),
             (:tiled,  (λ = 800.0,  T_hot = 310.0, T_cool = 300.0, mode = :tiled)),
             (:het4w,  (λ = 3200.0, T_hot = 310.0, T_cool = 300.0, Ug = 5.0, f_cor = 1e-4))]
    R = Dict{Symbol,Any}()
    for (name, cfg) in cases
        R[name] = patchy_run!(patchy_les(; HS_GRID..., cfg...); HS_RUN...)
        r = R[name]
        @printf("    %-6s λ/zi %5.2f  w'θ'₀ LES %.4f | tiles(patch air) %.4f  tiles(mean air) %.4f  param %.4f K m/s | w* %.2f  A_u/w* %.3f  heat err %.1e\n",
                name, (name === :hom ? NaN : cfg.λ / r.zi), r.les, r.tile_patch, r.tile_speed, r.param,
                r.wstar, patchy_amplitude(r.u_p) / r.wstar, r.heat_error)
    end

    @testset "Conservation in every run" begin
        @test all(abs(r.heat_error) < 1e-8 for r in values(R))
    end

    @testset "Tiled aggregation against the resolved heterogeneous LES" begin
        het = (:het1, :het4, :het4w)
        # A-priori: tiles forced by the mean air over their own patches
        # reproduce the resolved domain-mean flux; forced by the domain-
        # mean air (what one grid cell of this size sees) they are low by
        # the patch-scale air-temperature and wind contrasts, most at the
        # largest λ/z_i.
        for k in het
            r = R[k]
            @test abs(_hs_rel(r.tile_patch, r.les)) < 0.03
            @test abs(_hs_rel(r.tile_speed, r.les)) < 0.10
        end
        # Parameter aggregation (one surface at the mean skin temperature)
        # misses a third of the flux: the cool strips (below the air
        # temperature) are stable and nearly flux-less, the hot ones carry
        # it all — the flux is convex in θ_s.
        for k in het
            @test R[k].param < 0.8 * R[k].les
        end
        # A-posteriori: an LES whose every column is the tiled mosaic has the
        # same domain-mean flux as the resolved strips at λ/z_i ≈ 1, while the
        # LES with the mean skin temperature everywhere (:hom — which *is* the
        # parameter-aggregated surface) is far below.
        @info "A-posteriori domain-mean surface flux" resolved = R[:het1].les tiled = R[:tiled].les parameter = R[:hom].les
        @test abs(_hs_rel(R[:tiled].les, R[:het1].les)) < 0.08
        @test R[:hom].les < 0.8 * R[:het1].les
        # With a 5 m/s background wind the domain-mean air is representative
        # and the tiles need no patch information.
        @test abs(_hs_rel(R[:het4w].tile_vec, R[:het4w].les)) < 0.03
        # So does a BlendedMosaic (#42) that finds the blended state at the
        # first level from the mean air two levels up, when there is wind.
        # In free convection it is reported, not asserted (measured +2 % at
        # λ/z_i ≈ 0.9, −10 % at 3.2: one Monin–Obukhov layer does not
        # describe the mean profile under a patch-locked circulation).
        @info "BlendedMosaic estimate (rel. to LES)" het1 = _hs_rel(R[:het1].blended, R[:het1].les) het4 = _hs_rel(R[:het4].blended, R[:het4].les) het4w = _hs_rel(R[:het4w].blended, R[:het4w].les)
        @test abs(_hs_rel(R[:het4w].blended, R[:het4w].les)) < 0.03
    end

    @testset "Organised motion: Patton et al. (2005); Wanner et al. (2022)" begin
        A(k) = patchy_amplitude(R[k].u_p) / R[k].wstar
        # A patch-locked near-surface circulation well above the phase-average
        # noise of the uniform surface, and stronger at λ/z_i ≈ 3 than ≈ 1
        # (Patton et al.: strongest for 4 < λ/z_i < 9, little below).
        @test A(:het1) > 3 * A(:hom)
        @test A(:het4) > 2 * A(:het1)
        # It converges on the hot strip (phases 1:n/2): surface wind towards
        # its centre from both sides, updraft over it.
        r = R[:het4]
        n = length(r.u_p) ÷ 2
        @test sum(r.u_p[1:n÷2]) > 0 && sum(r.u_p[n÷2+1:n]) < 0
        @test argmax(r.w_p) in 1:n
        # Mean ascent over the hot strip, descent over the cool one (the
        # plane mean of w is zero, so this is the sign of the circulation).
        @info "Phase-averaged w at z_mid, λ = 3.2 km" hot = sum(r.w_p[1:n]) / n cool = sum(r.w_p[n+1:end]) / n
        @test sum(r.w_p[1:n]) > 0 > sum(r.w_p[n+1:end])
        # Coupled flux smallest mid-patch, largest at the edges (Patton et
        # al.: "surface fluxes in the middle of the patches are small
        # compared to the patch edges").
        h = r.wθ_p[1:n]
        mid = sum(h[n÷2:n÷2+1]) / 2
        edge = (h[1] + h[2] + h[n-1] + h[n]) / 4
        @info "Hot-strip flux, λ = 3.2 km" mid edge ratio = mid / edge
        @test mid < 0.95 * edge
        # A 5 m/s geostrophic wind across the strips removes the
        # patch-locked circulation (Wanner et al. 2022: the cellular
        # circulations "disappear with increasing wind speed").
        @test A(:het4w) < 0.1 * A(:het4)
    end
end
