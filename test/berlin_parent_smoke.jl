# The Berlin parent smoke test (N-parent-e, #259): every parent forcing
# on — the tiled surface, column radiation, the geostrophic pressure
# gradient, θ/qv nudging, subsidence, θ damping, the fixed-gradient top and
# the TKE closure — for 12 steps on a coarse parent (Δx = 250 m, Δz = 50 m,
# 24 × 24 × 33), and the heat and water ledgers of the air close.

using Test
using Aeolus
using Aeolus.Fields: interior_view
using Printf

isdefined(Main, :BerlinCase) ||
    include(joinpath(@__DIR__, "..", "examples", "urban", "BerlinCase.jl"))
const BPS = Main.BerlinCase

@testset "Berlin parent smoke (#259)" begin
    child = BPS.CaseConfig(; verbose = false, output_path = nothing)
    # 10:00 LT so the sun is up and the boundary layer convective
    child.start_h = 10.0
    pc = BPS.ParentConfig(child; refinement = (16, 16, 8), verbose = false)
    p = BPS.build_parent(pc)
    @test p.mesh.Nx == 24 && p.mesh.Ny == 24
    @test all(f -> f !== nothing, values(p.forcing))
    @test p.surface !== nothing && p.rad !== nothing && p.closure isa TKEClosure && p.top !== nothing
    for _ in 1:12
        BPS.step_parent!(p)
    end
    θ = interior_view(p.sim.θ)
    @test all(isfinite, θ) && all(isfinite, interior_view(p.sim.qv))
    @test minimum(interior_view(p.closure.e)) >= 0

    # the heat of the air: every change attributed, closure at round-off
    hb = BPS.heat_budget(p.heat)
    scale = maximum(abs, values(hb.terms))
    @test abs(hb.closure) <= 1e-10 * scale
    for key in ((:surface_flux, :tiles), (:column_tendency, :column_radiation),
                (:dynamics, :nudging), (:dynamics, :subsidence), (:dynamics, :damping),
                (:dynamics, :top_flux), (:dynamics, :transport))
        @test haskey(hb.terms, key)
    end
    # the surface stage changed the heat of the air by exactly what the tile
    # sources delivered (two routes: the measured θ change and Σ w_k ṡ_k τ;
    # the surfaces start at the child's Ts0 = 285 K, below the 10:00 air, so
    # the first steps' sensible heat flux is downward)
    @test hb.terms[(:surface_flux, :tiles)] ≈ p.sfc_mean[:delivered_H] rtol = 1e-10
    # the column transport conserves
    @test abs(hb.terms[(:dynamics, :transport)]) <= 1e-10 * scale
    # the water of the air
    wb = BPS.heat_budget(p.water)
    wscale = maximum(abs, values(wb.terms))
    @test abs(wb.closure) <= 1e-10 * wscale
    @test haskey(wb.terms, (:surface_flux, :tiles)) && haskey(wb.terms, (:dynamics, :nudging))
    @test wb.terms[(:surface_flux, :tiles)] ≈ p.sfc_mean[:delivered_E] rtol = 1e-10

    # the radiation: daylight shortwave and a sky longwave at the ground
    @test p.sw_down > 100 && 250 < p.lw_down < 450
    # the tiles: the surface sensible heat flux the ledger saw is the tiles'
    el = sum(p.heat.elapsed)
    @info @sprintf("parent smoke: %d steps, %.1f s, dt %.2f–%.2f s; surface %.1f W/m², radiation %.1f, subsidence %.1f, nudging %.2f, damping %.2f, top %.3f W/m²; closure %.1e (heat) %.1e (water)",
                   p.nsteps, el, minimum(p.dt_hist), maximum(p.dt_hist),
                   hb.terms[(:surface_flux, :tiles)] / el, hb.terms[(:column_tendency, :column_radiation)] / el,
                   hb.terms[(:dynamics, :subsidence)] / el, hb.terms[(:dynamics, :nudging)] / el,
                   hb.terms[(:dynamics, :damping)] / el, hb.terms[(:dynamics, :top_flux)] / el,
                   hb.closure / scale, wb.closure / wscale)
end
