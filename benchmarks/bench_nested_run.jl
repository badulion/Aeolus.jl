# NestedRun on a Berlin-sized nest (N-nest-c, #257): substep statistics
# and the parent / child / boundary / feedback cost split.
#
# Parent: 6 × 6 × 3 km, 192 × 192 × 120 (Δx = 31.25 m, uniform Δz = 25 m),
# periodic sides, ground and lid walls. Child sides and top are nest faces
# (OpenBoundary, N-nest-d #261), its ground a wall. Child: 1 × 1 km in the centre,
# refinement (2, 2, 4) — Δx = 15.625 m, Δz = 6.25 m — 32 levels (200 m, the
# issue's case) and 64 levels (400 m, the maintainer's deepened child).
# The flow is synthetic: a 5 m/s westerly with random velocity and θ
# perturbations over a stable θ profile, Smagorinsky SGS on both grids.
#
# Run: julia --project=. benchmarks/bench_nested_run.jl [nsteps=12]

using Aeolus
using Aeolus.BoundaryConditions: FieldBoundaryConditions
using Aeolus.Fields: interior_view
using Printf, Random

const NSTEPS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 12

function build(child_levels; feedback = :one_way, kw...)
    pm = RegularGrid(Float64; size = (192, 192, 120), extent = (6000.0, 6000.0, 3000.0),
                     halo = (3, 3, 3))
    pbcs = FieldBoundaryConditions(:periodic, :periodic, :wall)
    psim = BoussinesqSimulation(pm; θ₀ = 300.0, C_s = 0.17, bcs = pbcs)
    Random.seed!(2024)
    zc(k) = pm.z0 + (k - pm.Hz - 0.5) * pm.dz
    for k in axes(psim.θ.data, 3)
        psim.θ.data[:, :, k] .= 300.0 + 0.003 * zc(k)
    end
    interior_view(psim.θ) .+= 0.1 .* randn(size(interior_view(psim.θ)))
    fill!(psim.u.data, 5.0)
    interior_view(psim.u) .+= 0.5 .* randn(size(interior_view(psim.u)))
    interior_view(psim.v) .+= 0.5 .* randn(size(interior_view(psim.v)))
    step!(psim, 1e-3); psim.t = 0.0                  # project the noise

    depth = child_levels * 6.25
    nest = nested_grid(pm, ((2500.0, 3500.0), (2500.0, 3500.0), (0.0, depth)), (2, 2, 4))
    cm = nest_child_mesh(nest)
    # nest faces laterally and on top (OpenBoundary), a wall at the ground;
    # the default solver is the all-Neumann CosineFFTPoissonSolver
    cbcs = nest_bcs(cm, :nest, :nest; bottom = :wall, top = :nest)
    csim = BoussinesqSimulation(cm; θ₀ = 300.0, C_s = 0.17, bcs = cbcs)
    st = nest_interp_state(nest)
    interpolate_parent_state!(st, psim, nest; region = :all)
    for (cf, sf) in ((csim.u, st.u), (csim.v, st.v), (csim.w, st.w), (csim.θ, st.θ))
        cf.data .= sf.data
    end
    return NestedRun(psim, csim, nest; feedback, cfl_parent = CFLConstraints(), kw...)
end

# Two-way runs with the interface flux correction: since N-nest-d the
# child's top is an open nest face carrying the parent's w, so the child's
# interface fluxes equal the parent's (N-nest-c had to switch the correction
# off: it forced the parent's flux through the child's wall lid to w = 0).
for (levels, fb, kw) in ((32, :one_way, NamedTuple()), (64, :one_way, NamedTuple()),
                         (64, :two_way, NamedTuple()))
    run = build(levels; feedback = fb, kw...)
    step_nested!(run)                                # compile
    run.stats = NestRunStats()
    t0 = time()
    for _ in 1:NSTEPS
        step_nested!(run)
    end
    wall = time() - t0
    s = run.stats
    cm = run.nest.child_mesh
    @printf("child %d×%d×%d (%s), %d parent steps to t = %.1f s, total %.2f s\n",
            cm.Nx, cm.Ny, cm.Nz, fb, NSTEPS, run.t, wall)
    @printf("  substeps n = %d..%d (%d total), retries %d/%d; Δt_p ∈ [%.3f, %.3f] s, Δt_c ∈ [%.3f, %.3f] s\n",
            s.n_min, s.n_max, s.child_substeps, s.parent_retries, s.child_retries,
            s.dt_parent_min, s.dt_parent_max, s.dt_child_min, s.dt_child_max)
    @printf("  max child CFL %.3f; max clock error %.1e\n", s.max_child_cfl, s.max_clock_error)
    @printf("  wall per parent step: parent %.3f s, child %.3f s, boundary %.3f s, feedback %.3f s\n",
            s.wall_parent / NSTEPS, s.wall_child / NSTEPS, s.wall_boundary / NSTEPS,
            s.wall_feedback / NSTEPS)
    @printf("  child / parent = %.2f; max |u| child %.2f m/s, parent %.2f m/s\n",
            s.wall_child / s.wall_parent, maximum(abs, interior_view(run.child.u)),
            maximum(abs, interior_view(run.parent.u)))
end
