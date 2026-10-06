# The pressure solver of the projection step, in the `:pressure_solver`
# slot. The solver is part of the dynamical core, so the providers act
# through `sim_options` (a `solver` keyword of BoussinesqSimulation) and
# have no stage of their own.
#
#   :cosine_fft  (default) PressureSolvers.CosineFFTPoissonSolver — direct,
#                exact to round-off for periodic x, y and walls in z.
#   :multigrid   Multigrid.MultigridPoissonSolver — geometric V-cycles to
#                a relative residual `tol`.
#   :mg_cg       Multigrid.MGPreconditionedCG — CG with one V-cycle as the
#                preconditioner.
#
#   :density_weighted_cg  Anelastic.DensityWeightedPoissonSolver — the
#                Jacobi-preconditioned CG of the sound-proof cores (#133),
#                ∇·(W∇φ) = ∇·(W u*) with W = ρ̄ (anelastic) or ρ̄θ̄ (pseudo-
#                incompressible). The core builds it from its own base state
#                and weighting; this provider passes its `tol` and `maxit`.
#
# The first three solve the constant-coefficient equation of the
# Boussinesq core and are incompatible with the sound-proof cores; the
# last solves only the weighted one and is incompatible with Boussinesq.
#
# Both multigrid options solve the same 7-point Laplacian with the same
# boundary conditions as the FFT, so on this uniform grid they only cost
# more; they are here because the stretched-grid (#151) and cut-cell (#152)
# solver work needs them, and so that the Berlin case keeps them exercised.
#
# Every provider takes its per-axis Poisson closure from the case's
# velocity boundary conditions (`case_poisson_axes`, N-open-a #251):
# periodic sides give a periodic axis, walls and prescribed-velocity
# (open / nest) sides a Neumann one. The default case is periodic in x, y
# with walls in z — (:periodic, :periodic, :neumann), the solver of before
# #251 bit for bit; `cfg.lateral = :open` (#262) gives the all-Neumann
# solve. The sound-
# proof cores derive the same axes from the same `bcs` inside
# `Anelastic._build_mass_weighted`. Each smoke check asserts the solver's
# axes are the case's.

"Maximum |∇·u| of the case velocity (fills the velocity halos)."
case_divergence(case) = max_divergence(case.sim)

"""
    case_poisson_axes(cfg, mesh) -> (bc_x, bc_y, bc_z)

The per-axis pressure-Poisson kinds (`:periodic` / `:neumann`) implied by
the velocity boundary conditions of the case's core
([`case_boundary_conditions`](@ref)`(cfg, mesh).bcs`: `cfg.lateral`), via
`PressureSolvers.poisson_axis_bcs`.
"""
case_poisson_axes(cfg, mesh) = poisson_axis_bcs(case_boundary_conditions(cfg, mesh).bcs)

"Smoke-check entry: the solver's per-axis closure is the one the case's `bcs` imply."
solver_axes_check(case) =
    "the solver's axis BCs are the case's ($(join(values(poisson_axis_bcs(case.sim.bcs)), ", ")))" =>
        poisson_axis_bcs(case.sim.solver) == poisson_axis_bcs(case.sim.bcs) ==
        case_poisson_axes(case.cfg, case.mesh)

"""
    PressureCosineFFT

Slot `:pressure_solver`, provider `:cosine_fft` (default):
`CosineFFTPoissonSolver(mesh; case_poisson_axes(cfg, mesh)...)`, the solver
`BoussinesqSimulation` picks for the case's boundary conditions
(periodic x, y and walls in z by default; all-Neumann for `lateral = :open`
or a nested child).
"""
struct PressureCosineFFT <: AbstractCoupling end
coupling_name(::Type{PressureCosineFFT}) = :pressure_fft
slot(::Type{PressureCosineFFT}) = :pressure_solver
provider(::Type{PressureCosineFFT}) = :cosine_fft
modules_used(::Type{PressureCosineFFT}) = (:PressureSolvers,)
sim_options(::Type{PressureCosineFFT}, cfg, mesh, o) =
    (solver = CosineFFTPoissonSolver(mesh; case_poisson_axes(cfg, mesh)...),)
smoke_check(c::PressureCosineFFT, case) =
    ["the solver is the cosine FFT" => case.sim.solver isa CosineFFTPoissonSolver,
     solver_axes_check(case),
     "projected velocity divergence-free (max|∇·u| < 1e-10 s⁻¹)" => case_divergence(case) < 1e-10]

"""
    PressureMultigrid

Slot `:pressure_solver`, provider `:multigrid`: `MultigridPoissonSolver`
with the case's axis BCs (`case_poisson_axes`), options `tol = 1e-10`
(relative residual), `maxcycles = 50`, `cycle = :V`.
"""
struct PressureMultigrid <: AbstractCoupling end
coupling_name(::Type{PressureMultigrid}) = :pressure_multigrid
slot(::Type{PressureMultigrid}) = :pressure_solver
provider(::Type{PressureMultigrid}) = :multigrid
is_default(::Type{PressureMultigrid}) = false
modules_used(::Type{PressureMultigrid}) = (:Multigrid,)
default_options(::Type{PressureMultigrid}) = (tol = 1e-10, maxcycles = 50, cycle = :V)
sim_options(::Type{PressureMultigrid}, cfg, mesh, o) =
    (solver = MultigridPoissonSolver(mesh; case_poisson_axes(cfg, mesh)...,
                                     tol = o.tol, maxcycles = o.maxcycles, cycle = o.cycle),)
function report(c::PressureMultigrid, case)
    s = case.sim.solver
    return @sprintf("multigrid: %d levels, last solve %d cycles, rel. residual %.1e",
                    mg_level_count(s), s.last_cycles[], mg_residual_norm(s))
end
function smoke_check(c::PressureMultigrid, case)
    s = case.sim.solver
    return ["the solver is multigrid" => s isa MultigridPoissonSolver,
            solver_axes_check(case),
            "last solve converged (rel. residual < tol)" => mg_residual_norm(s) < s.tol,
            "projected velocity divergence-free (max|∇·u| < 1e-8 s⁻¹)" => case_divergence(case) < 1e-8]
end

"""
    PressureMGCG

Slot `:pressure_solver`, provider `:mg_cg`: `MGPreconditionedCG` (CG with
one multigrid V-cycle as the preconditioner) with the case's axis BCs
(`case_poisson_axes`), options `tol = 1e-10`,
`maxit = 200`.
"""
struct PressureMGCG <: AbstractCoupling end
coupling_name(::Type{PressureMGCG}) = :pressure_mg_cg
slot(::Type{PressureMGCG}) = :pressure_solver
provider(::Type{PressureMGCG}) = :mg_cg
is_default(::Type{PressureMGCG}) = false
modules_used(::Type{PressureMGCG}) = (:Multigrid,)
default_options(::Type{PressureMGCG}) = (tol = 1e-10, maxit = 200)
sim_options(::Type{PressureMGCG}, cfg, mesh, o) =
    (solver = MGPreconditionedCG(mesh; case_poisson_axes(cfg, mesh)...,
                                 tol = o.tol, maxit = o.maxit),)
report(c::PressureMGCG, case) =
    @sprintf("MG-CG: last solve %d iterations, rel. residual %.1e",
             case.sim.solver.last_iters[], mg_residual_norm(case.sim.solver))
function smoke_check(c::PressureMGCG, case)
    s = case.sim.solver
    return ["the solver is MG-preconditioned CG" => s isa MGPreconditionedCG,
            solver_axes_check(case),
            "last solve converged (rel. residual < tol)" => mg_residual_norm(s) < s.tol,
            "projected velocity divergence-free (max|∇·u| < 1e-8 s⁻¹)" => case_divergence(case) < 1e-8]
end

"""
    PressureDensityWeighted

Slot `:pressure_solver`, provider `:density_weighted_cg`: the projection
of the sound-proof cores (`core = :anelastic`, `:pseudo_incompressible`),
`Anelastic.DensityWeightedPoissonSolver`, which the core builds from its
base state (`case.ref.state`) and its weighting. Options `tol = 1e-10`
(relative residual of the CG), `maxit = 500`, passed to the core
constructor as `sim_options`. Incompatible with the Boussinesq core.
"""
struct PressureDensityWeighted <: AbstractCoupling end
coupling_name(::Type{PressureDensityWeighted}) = :pressure_density_weighted
slot(::Type{PressureDensityWeighted}) = :pressure_solver
provider(::Type{PressureDensityWeighted}) = :density_weighted_cg
is_default(::Type{PressureDensityWeighted}) = false
modules_used(::Type{PressureDensityWeighted}) = (:Anelastic,)
default_options(::Type{PressureDensityWeighted}) = (tol = 1e-10, maxit = 500)
incompatible(::Type{PressureDensityWeighted}) = (:core => :boussinesq,)
incompatible_reason(::Type{PressureDensityWeighted}) =
    "the density-weighted CG solves the projection ∇·(W∇φ) = ∇·(W u*) of the sound-proof " *
    "cores (core = :anelastic or :pseudo_incompressible); the Boussinesq core takes one of " *
    "cosine_fft, multigrid, mg_cg"
sim_options(::Type{PressureDensityWeighted}, cfg, mesh, o) = (tol = o.tol, maxit = o.maxit)
report(c::PressureDensityWeighted, case) =
    @sprintf("density-weighted CG: last solve %d iterations, ‖r‖ %.1e",
             solver_iterations(case.sim.solver), solver_residual(case.sim.solver))
function smoke_check(c::PressureDensityWeighted, case)
    s, sim = case.sim.solver, case.sim
    ref = Aeolus.Anelastic.reference_state(sim)
    W = sim isa Aeolus.Anelastic.AnelasticSimulation ? ref.ρ̄ : ref.ρ̄ .* ref.θ̄
    o = coupling_options(PressureDensityWeighted, case.cfg)
    return ["the solver is the density-weighted CG" => s isa DensityWeightedPoissonSolver,
            "with this provider's tol and maxit" => s.tol == o.tol && s.maxit == o.maxit,
            solver_axes_check(case),
            "its weight is the core's (ρ̄ or ρ̄θ̄ of case.ref, rel. 1e-12)" =>
                isapprox(s.w_c, W; rtol = 1e-12),
            "last solve converged within maxit" => 0 < solver_iterations(s) < s.maxit,
            "∇·(W u) at the solver tolerance (< 1e-6 s⁻¹)" =>
                Aeolus.Anelastic.constraint_residual(sim) < 1e-6]
end

register!(PressureCosineFFT)
register!(PressureMultigrid)
register!(PressureMGCG)
register!(PressureDensityWeighted)
