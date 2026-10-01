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
# Both multigrid options solve the same 7-point Laplacian with the same
# boundary conditions as the FFT (periodic, periodic, Neumann), so on this
# uniform grid they only cost more; they are here because the stretched-
# grid (#151) and cut-cell (#152) solver work needs them, and so that the
# Berlin case keeps them exercised.

"Maximum |∇·u| of the case velocity (fills the velocity halos)."
case_divergence(case) = max_divergence(case.sim)

"""
    PressureCosineFFT

Slot `:pressure_solver`, provider `:cosine_fft` (default):
`CosineFFTPoissonSolver(mesh)`, the solver `BoussinesqSimulation` picks
for walls in z.
"""
struct PressureCosineFFT <: AbstractCoupling end
coupling_name(::Type{PressureCosineFFT}) = :pressure_fft
slot(::Type{PressureCosineFFT}) = :pressure_solver
provider(::Type{PressureCosineFFT}) = :cosine_fft
modules_used(::Type{PressureCosineFFT}) = (:PressureSolvers,)
sim_options(::Type{PressureCosineFFT}, cfg, mesh, o) = (solver = CosineFFTPoissonSolver(mesh),)
smoke_check(c::PressureCosineFFT, case) =
    ["the solver is the cosine FFT" => case.sim.solver isa CosineFFTPoissonSolver,
     "projected velocity divergence-free (max|∇·u| < 1e-10 s⁻¹)" => case_divergence(case) < 1e-10]

"""
    PressureMultigrid

Slot `:pressure_solver`, provider `:multigrid`: `MultigridPoissonSolver`
with `bc = (:periodic, :periodic, :neumann)`, options `tol = 1e-10`
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
    (solver = MultigridPoissonSolver(mesh; bc_x = :periodic, bc_y = :periodic, bc_z = :neumann,
                                     tol = o.tol, maxcycles = o.maxcycles, cycle = o.cycle),)
function report(c::PressureMultigrid, case)
    s = case.sim.solver
    return @sprintf("multigrid: %d levels, last solve %d cycles, rel. residual %.1e",
                    mg_level_count(s), s.last_cycles[], mg_residual_norm(s))
end
function smoke_check(c::PressureMultigrid, case)
    s = case.sim.solver
    return ["the solver is multigrid" => s isa MultigridPoissonSolver,
            "last solve converged (rel. residual < tol)" => mg_residual_norm(s) < s.tol,
            "projected velocity divergence-free (max|∇·u| < 1e-8 s⁻¹)" => case_divergence(case) < 1e-8]
end

"""
    PressureMGCG

Slot `:pressure_solver`, provider `:mg_cg`: `MGPreconditionedCG` (CG with
one multigrid V-cycle as the preconditioner), options `tol = 1e-10`,
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
    (solver = MGPreconditionedCG(mesh; bc_x = :periodic, bc_y = :periodic, bc_z = :neumann,
                                 tol = o.tol, maxit = o.maxit),)
report(c::PressureMGCG, case) =
    @sprintf("MG-CG: last solve %d iterations, rel. residual %.1e",
             case.sim.solver.last_iters[], mg_residual_norm(case.sim.solver))
function smoke_check(c::PressureMGCG, case)
    s = case.sim.solver
    return ["the solver is MG-preconditioned CG" => s isa MGPreconditionedCG,
            "last solve converged (rel. residual < tol)" => mg_residual_norm(s) < s.tol,
            "projected velocity divergence-free (max|∇·u| < 1e-8 s⁻¹)" => case_divergence(case) < 1e-8]
end

register!(PressureCosineFFT)
register!(PressureMultigrid)
register!(PressureMGCG)
