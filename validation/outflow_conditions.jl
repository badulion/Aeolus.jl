# N-open-c (#260, epic #246): validation of the radiation outflow
# conditions on open faces (`outflow = :dirichlet | :convective |
# :orlanski`) with an outgoing Lamb–Oseen vortex:
#
#   6. outgoing Lamb–Oseen vortex (validation/support/open_boundary_vortex_harness.jl):
#      R for Dirichlet (perfect and uniform-U parent), convective, Orlanski
#   7. stability: 2000 steps with a time-interpolated parent, ∫|u|² vs parent
#
# The conditions stay tested in test/outflow_conditions.jl (the API,
# uniform flow through radiating faces on every core, the ∫θ budget of
# boundary_scalar_flux, a scalar pulse leaving through the outflow, plane
# Poiseuille flow with an open x axis).
#
# References: Orlanski (1976), JCP 21, 251 (radiation condition, diagnosed
# phase speed); Miller & Thorpe (1981), QJRMS 107, 615 (clipping
# 0 ≤ c ≤ Δx/Δt, upstream radiation schemes); Pauley, Moin & Reynolds
# (1990), JFM 220, 397 (convective outflow condition in incompressible
# LES); Colonius (2004), Annu. Rev. Fluid Mech. 36, 315 (outgoing-vortex
# tests of non-reflecting conditions).

using Test
using Aeolus
using Aeolus.Grids: interior, CellCenter, FaceX, FaceY, FaceZ
using Aeolus.Fields: Field, interior_view, set!
using Aeolus.BoundaryConditions: OpenBoundary, RadiationPlane
using Aeolus.Simulations: BoussinesqSimulation, step_rk3!, max_divergence
using Aeolus.TimeIntegration: step_imex!, ARK2, Rosenbrock2, step_adaptive!, CFLConstraints
using Aeolus.Anelastic: AnelasticSimulation, PseudoIncompressibleSimulation
using Aeolus.Operators: advect_centered!
using Printf

include(joinpath(@__DIR__, "support", "open_boundary_vortex_harness.jl"))

# ---------------------------------------------------------------------------
# The outgoing Lamb–Oseen vortex. R at the end of the run, when the vortex
# centre is 3 r_c past the outflow face.
@testset "outgoing Lamb–Oseen vortex (r_c = 8Δx): reflection R" begin
    h = VortexHarness()
    perfect = run_vortex_harness(h; outflow = :dirichlet, parent = :truth)
    dir_u   = run_vortex_harness(h; outflow = :dirichlet, parent = :uniform)
    conv    = run_vortex_harness(h; outflow = :convective, parent = :uniform)
    orl     = run_vortex_harness(h; outflow = :orlanski, parent = :uniform)
    @printf("  vortex R (end; max over the run): Dirichlet perfect parent %.2e (%.2e), Dirichlet uniform-U %.3f (%.3f), convective %.2e (%.2e), Orlanski %.2e (%.2e)\n",
            perfect.R, perfect.Rmax, dir_u.R, dir_u.Rmax, conv.R, conv.Rmax, orl.R, orl.Rmax)
    @test perfect.R < 1e-3
    @test conv.R < 1e-2
    @test orl.R < 1e-2
    @test dir_u.R >= 10 * conv.R
    @test dir_u.R >= 10 * orl.R
    @test max_divergence(conv.child) < 1e-10 && max_divergence(orl.child) < 1e-10
end

# ---------------------------------------------------------------------------
@testset "stability: 2000 steps, time-interpolated parent, ∫|u|² within 1 %" begin
    h = VortexHarness()
    for mode in (:dirichlet, :convective, :orlanski)
        r = run_vortex_harness(h; outflow = mode, parent = :truth, parent_every = 5,
                               nsteps = 2000, record_every = 20)
        d = maximum(abs, r.ke_child ./ r.ke_truth .- 1)
        @printf("  %-10s 2000 steps, parent every 5 steps: max |∫|u|²_child / ∫|u|²_parent − 1| = %.2e\n",
                mode, d)
        @test d < 0.01
        @test all(isfinite, r.child.u.data)
        @test max_divergence(r.child) < 1e-10
    end
end
