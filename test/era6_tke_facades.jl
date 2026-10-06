# #232: the shear production of the prognostic SGS TKE closure at
# wall-modelled immersed walls (the Berlin facades).
#
# The facade wall model (P57-d, #179) takes the SGS stencil flux through
# every facade out of the momentum tendency (`remove_stencil_wall_flux!`)
# and applies a log-law stress instead (`apply_immersed_wall_stress!`).
# Before #232 the TKE closure still booked the stencil's production there
# (ν_t,e (q_F/Δ)², a stress the step no longer applies) and none of the
# wall stress's work. With the walls attached (`attach_immersed_walls!`)
# the production at those faces is the wall stress times the resolved
# shear across the wall, a q_F².
#
# Physics asserted: (1) the discrete budget with facades on — the resolved
# kinetic energy the SGS operator (as modified by the wall model) and the
# wall stress take out of the flow is the TKE shear production, the
# interior dual part plus the wall-stress work, to round-off, and the
# wall-stress work is the energy `apply_immersed_wall_stress!` actually
# removes; without the walls attached the closure books the removed
# stencil share instead; (2) uniform flow along a wall: the facade
# production per unit wall area is τ_w U = u*³ ln(h/z0)/κ, the integrated
# shear production of the log layer between z0 and h; (3) a stepped run
# with the `step!` hooks of the Berlin facades: e ≥ 0 with no clips, the
# booked `wall` statistic is the wall-stress work, and the log-law stress
# does more work than the ν_t stencil share it replaces.
#
# Numbers measured on this branch are quoted next to each assertion.

using Test
using Printf
using Aeolus
using Aeolus.WallModels
using Aeolus.IBM: SolidMask, mark_solid!, implicit_ibm_correction!
using Aeolus.Fields: interior_view
using Aeolus.Operators: compute_sgs!, add_sgs_momentum!
using Aeolus.Simulations: kinetic_energy
using Random: Xoshiro

@testset "TKE production at wall-modelled facades (#232)" begin

# A periodic box (walls in z) with a 4 × 3 block of solid columns up to
# 5 of 8 levels: walls in ±x and ±y for u, v and w, corners included.
function facade_box(; seed = 1, law = RoughLogLaw(0.05))
    g = RegularGrid(; size = (16, 12, 8), extent = (32.0, 24.0, 16.0), halo = (3, 3, 3))
    sim = BoussinesqSimulation(g; θ₀ = 300.0,
                               bcs = FieldBoundaryConditions(:periodic, :periodic, :wall),
                               θ_bcs = FieldBoundaryConditions(:periodic, :periodic, :noflux))
    sm = SolidMask(g)
    mark_solid!(sm, (x, y, z) -> 10 < x < 18 && 8 < y < 14 && z < 10)
    rng = Xoshiro(seed)
    for f in (sim.u, sim.v, sim.w)
        interior_view(f) .= 0.5 .* randn(rng, g.Nx, g.Ny, g.Nz)
    end
    interior_view(sim.u) .+= 2.0
    sim.w.data[:, :, g.Hz+1] .= 0
    implicit_ibm_correction!(sim, sm)                # zero every solid face
    for f in (sim.u, sim.v, sim.w); fill_halos!(f, sim.bcs); end
    c = TKEClosure(g; mask = sm.mask)
    interior_view(c.e) .= 0.02 .+ 0.01 .* sin.((1:g.Nx) .* 0.4) .* cos.((1:g.Ny)' .* 0.3)
    c.e.data[sm.mask] .= 0
    iw = immersed_walls(sim, sm, law)
    return sim, c, iw, sm
end

dotV(sim, t) = sum(sum(interior_view(q) .* interior_view(tq))
                   for (q, tq) in zip((sim.u, sim.v, sim.w), t)) * sim.mesh.dx * sim.mesh.dy * sim.mesh.dz

@testset "discrete budget with facades on" begin
    sim, c, iw, sm = facade_box()
    g = sim.mesh
    @test all(>(0), immersed_wall_count(iw))
    # the SGS momentum tendency of the closure, and the facade flux the
    # wall model takes out of it (SGS part only: ν = 0)
    f = compute_sgs!(c, sim)
    tu, tv, tw = Field(FaceX(), g), Field(FaceY(), g), Field(FaceZ(), g)
    add_sgs_momentum!(tu, tv, tw, c, f, sim)
    L_full = -dotV(sim, (tu, tv, tw))                      # stencil sink, unmodified
    sim.tu.data .= tu.data; sim.tv.data .= tv.data; sim.tw.data .= tw.data
    remove_stencil_wall_flux!(sim, iw; ν = 0, νt = f.νh, sgs = true)
    L_sgs = -dotV(sim, (sim.tu, sim.tv, sim.tw))           # sink of the operator the step applies
    # the KE the wall stress removes, measured on the operator itself
    q0 = (copy(sim.u.data), copy(sim.v.data), copy(sim.w.data))
    E0 = kinetic_energy(sim)
    δt = 1e-6
    apply_immersed_wall_stress!(sim, iw, δt)
    W_op = (E0 - kinetic_energy(sim)) / δt
    sim.u.data .= q0[1]; sim.v.data .= q0[2]; sim.w.data .= q0[3]

    # without the walls: the stencil's production (the bug of #232)
    tb0 = tke_budget(c, sim)
    @test abs(tb0.shear - L_full) / L_full < 1e-12                       # 3.1e-16
    @test tb0.wall_work == 0 && tb0.wall_stencil == 0
    # with the walls attached
    @test attach_immersed_walls!(c, iw)
    tb = tke_budget(c, sim)
    @info @sprintf("  facade budget: stencil sink %.4e, after removal %.4e, removed share %.4e, wall work %.4e (operator %.4e), Σ P_s V %.4e",
                   L_full, L_sgs, tb.wall_stencil, tb.wall_work, W_op, tb.shear)
    @test L_sgs > 0 && tb.wall_stencil > 0 && tb.wall_work > 0
    # KE lost by the resolved flow = TKE shear production (interior dual part + wall-stress work)
    @test abs(tb.shear - (L_sgs + tb.wall_work)) / tb.shear < 1e-12        # 0.0
    @test abs((tb.shear - tb.wall_work) - L_sgs) / L_sgs < 1e-12           # 0.0: the interior part is the dual
    @test abs(tb.wall_stencil - (L_full - L_sgs)) / tb.wall_stencil < 1e-12 # 4.6e-15: exactly what was removed
    # the wall work is the KE apply_immersed_wall_stress! removes (O(δt) implicit-step error)
    @test abs(tb.wall_work - W_op) / W_op < 1e-5                           # 4.7e-8
    # D stays ≥ 0 in the air after the stencil share is taken out; Pw ≥ 0
    air = .!sm.mask[g.Hx+1:g.Hx+g.Nx, g.Hy+1:g.Hy+g.Ny, g.Hz+1:g.Hz+g.Nz]
    @test minimum(interior_view(c.D)[air]) >= -1e-12 * maximum(interior_view(c.D))  # min 0.12, max 3.7
    @test minimum(interior_view(c.Pw)) >= 0
    @test all(interior_view(c.Pw)[.!air] .== 0)          # nothing booked in a building
    # attach / detach
    @test attach_immersed_walls!(c, nothing)
    @test tke_budget(c, sim).shear ≈ tb0.shear rtol = 1e-14
    @test !attach_immersed_walls!(nothing, iw)          # a closure with no facade production
    @test_throws ArgumentError attach_immersed_walls!(c, 1.0)
end

@testset "uniform flow along a wall: P = τ_w U = u*³ ln(h/z0)/κ per unit area" begin
    # u = U past solid rows j = 9, 10 of a periodic-in-y channel: every
    # interior stencil flux vanishes, only the walls see shear
    g = RegularGrid(; size = (8, 10, 4), extent = (8.0, 10.0, 4.0), halo = (3, 3, 3))
    sim = BoussinesqSimulation(g; bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
    sm = SolidMask(g); mark_solid!(sm, (x, y, z) -> y > 8)
    U, z0 = 2.0, 0.01
    interior_view(sim.u) .= U
    implicit_ibm_correction!(sim, sm)
    c = TKEClosure(g; mask = sm.mask, walls = immersed_walls(sim, sm, RoughLogLaw(z0)))
    interior_view(c.e) .= 0.05; c.e.data[sm.mask] .= 0
    tb = tke_budget(c, sim)
    iw = c.walls[]
    area = count(f -> f.c == 1, iw.faces) * g.dx * g.dz            # two walls of 8 × 4 faces
    κ = 0.4; h = g.dy / 2
    us = κ * U / log(h / z0)
    @test tb.wall_work / area ≈ us^3 * log(h / z0) / κ rtol = 1e-12  # = τ_w U (exact to round-off)
    # the uniform flow has no interior shear: all production is the wall's
    @test abs(tb.shear - tb.wall_work) <= 1e-12 * tb.wall_work
    # the stencil share it replaces, ν_t,e U²/Δ² per face, is gone
    @test tb.wall_stencil > 0
end

@testset "stepped: facade production in a running closure" begin
    function run(attach)
        sim, c, iw, sm = facade_box(; seed = 3)
        attach && attach_immersed_walls!(c, iw)
        bf! = (s, dt) -> remove_stencil_wall_flux!(s, iw; sgs = true)
        pp! = (s, dt) -> (implicit_ibm_correction!(s, sm; dt = dt); apply_immersed_wall_stress!(s, iw, dt))
        for _ in 1:20
            step!(sim, 0.05; sgs = c, body_force! = bf!, post_predictor! = pp!)
        end
        return sim, c, iw, sm
    end
    sim1, c1, iw, sm = run(true)
    sim0, c0, _, _ = run(false)
    st1, st0 = tke_statistics(c1), tke_statistics(c0)
    @test st1.wall > 0 && st0.wall == 0
    @test minimum(interior_view(c1.e)) >= 0 && st1.clips == 0
    # the booked wall term is the wall-stress work of the committed state
    tb = tke_budget(c1, sim1)
    @test st1.wall ≈ tb.wall_work rtol = 1e-12
    @test tb.wall_work > tb.wall_stencil      # the log-law stress does more work than ν_t's stencil
    @info @sprintf("  stepped: wall %.3e (booked), wall work %.3e, stencil share %.3e at the end",
                   st1.wall, tb.wall_work, tb.wall_stencil)
end

end
