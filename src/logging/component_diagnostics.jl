# The library's implementations of the diagnostics registry (LOG-c, #335):
# `RunLog.diagnostics(component, sim)` for the simulations, the SGS
# closures, the wall models and the nested run. Included at the top level
# of `module Aeolus`, after every submodule, because it needs their types;
# each source delegates to the component's own diagnostic function (one
# source of truth): `kinetic_energy`, `max_divergence`, `sgs_fields`,
# `apply_wall_stress!`'s friction velocity, the TBLE columns' u_τ,
# `NestRunStats`.

import .RunLog: diagnostics
using .RunLog: Diagnostic, prefixed
using .Fields: interior_view

# ---------------------------------------------------------------------------
# simulations
# ---------------------------------------------------------------------------

_domain_volume(m) = m.Lx * m.Ly * m.Lz

"""
    diagnostics(sim::BoussinesqSimulation, sim) -> Vector{Diagnostic}

The dynamical core's costly diagnostics: `ke_mean` (the volume-mean
kinetic energy per unit mass, [`kinetic_energy`](@ref Aeolus.Simulations.kinetic_energy)
over the domain volume), `max_abs_u`, `max_abs_v`, `max_abs_w`,
`theta_min`, `theta_max`, `max_div` ([`max_divergence`](@ref
Aeolus.Simulations.max_divergence)), `nu_t_max` (the eddy viscosity of
the last evaluation) and, with moisture, `qv_max`.
"""
function diagnostics(sim::Simulations.BoussinesqSimulation, ::Simulations.BoussinesqSimulation)
    ds = Diagnostic[
        Diagnostic("ke_mean", "m2 s-2", () -> Simulations.kinetic_energy(sim) / _domain_volume(sim.mesh);
                   description = "volume-mean resolved kinetic energy per unit mass"),
        Diagnostic("max_abs_u", "m s-1", () -> interior_view(sim.u); reduction = :maxabs),
        Diagnostic("max_abs_v", "m s-1", () -> interior_view(sim.v); reduction = :maxabs),
        Diagnostic("max_abs_w", "m s-1", () -> interior_view(sim.w); reduction = :maxabs),
        Diagnostic("theta_min", "K", () -> interior_view(sim.θ); reduction = :min),
        Diagnostic("theta_max", "K", () -> interior_view(sim.θ); reduction = :max),
        Diagnostic("max_div", "s-1", () -> Simulations.max_divergence(sim);
                   description = "max |∇·u| over the interior cells"),
        Diagnostic("nu_t_max", "m2 s-1", () -> interior_view(sim.νt); reduction = :max,
                   description = "max SGS eddy viscosity of the last evaluation"),
    ]
    sim.has_moisture &&
        push!(ds, Diagnostic("qv_max", "kg kg-1", () -> interior_view(sim.qv); reduction = :max))
    return ds
end

"""
    diagnostics(sim::MassWeightedSimulation, sim) -> Vector{Diagnostic}

The sound-proof cores' costly diagnostics: `ke` (the density-weighted
kinetic energy `½∫ρ̄|u|²dV`, [`kinetic_energy`](@ref
Aeolus.Anelastic.kinetic_energy)), `max_abs_u`, `max_abs_v`, `max_abs_w`,
`theta_min`, `theta_max` and `max_div`, the residual of the core's own
continuity constraint in s⁻¹: [`max_anelastic_divergence`](@ref
Aeolus.Anelastic.max_anelastic_divergence) for the anelastic weighting,
[`max_pseudo_incompressible_divergence`](@ref
Aeolus.Anelastic.max_pseudo_incompressible_divergence) for the
pseudo-incompressible one (the plain `∇·u` is non-zero by design).
"""
_constraint_residual(sim::Anelastic.MassWeightedSimulation{Anelastic.AnelasticWeighting}) =
    Anelastic.max_anelastic_divergence(sim)
_constraint_residual(sim::Anelastic.MassWeightedSimulation{Anelastic.PseudoIncompressibleWeighting}) =
    Anelastic.max_pseudo_incompressible_divergence(sim)

function diagnostics(sim::Anelastic.MassWeightedSimulation, ::Anelastic.MassWeightedSimulation)
    return Diagnostic[
        Diagnostic("ke", "J", () -> Simulations.kinetic_energy(sim);
                   description = "density-weighted resolved kinetic energy"),
        Diagnostic("max_abs_u", "m s-1", () -> interior_view(sim.u); reduction = :maxabs),
        Diagnostic("max_abs_v", "m s-1", () -> interior_view(sim.v); reduction = :maxabs),
        Diagnostic("max_abs_w", "m s-1", () -> interior_view(sim.w); reduction = :maxabs),
        Diagnostic("theta_min", "K", () -> interior_view(sim.θ); reduction = :min),
        Diagnostic("theta_max", "K", () -> interior_view(sim.θ); reduction = :max),
        Diagnostic("max_div", "s-1", () -> _constraint_residual(sim);
                   description = "max residual of the core's continuity constraint"),
    ]
end

# ---------------------------------------------------------------------------
# SGS closures
# ---------------------------------------------------------------------------

function _sgs_diagnostics(c, sim)
    f() = Operators.sgs_fields(c, sim)
    return Diagnostic[
        Diagnostic("nu_h_max", "m2 s-1", () -> interior_view(f().νh); reduction = :max,
                   description = "max horizontal eddy viscosity of the last evaluation"),
        Diagnostic("nu_v_max", "m2 s-1", () -> interior_view(f().νv); reduction = :max,
                   description = "max vertical eddy viscosity of the last evaluation"),
        Diagnostic("kappa_h_max", "m2 s-1", () -> interior_view(f().κh); reduction = :max,
                   description = "max horizontal eddy diffusivity of the last evaluation"),
    ]
end

"""
    diagnostics(closure::AbstractSGSClosure, sim) -> Vector{Diagnostic}

Any SGS closure: `nu_h_max`, `nu_v_max`, `kappa_h_max`, the maxima of
the eddy coefficients of its last evaluation, read through the closure
contract's [`sgs_fields`](@ref Aeolus.Operators.sgs_fields) (costly
cadence). A `TKEClosure` adds `e_mean` (mean over the interior cells)
and `e_max` of its SGS energy.
"""
diagnostics(c::Operators.AbstractSGSClosure, sim) = _sgs_diagnostics(c, sim)

function diagnostics(c::PrognosticSGS.TKEClosure, sim)
    ds = _sgs_diagnostics(c, sim)
    push!(ds, Diagnostic("e_mean", "m2 s-2", () -> interior_view(c.e); reduction = :mean,
                         description = "SGS TKE, mean over the interior cells"))
    push!(ds, Diagnostic("e_max", "m2 s-2", () -> interior_view(c.e); reduction = :max))
    return ds
end

# ---------------------------------------------------------------------------
# wall models
# ---------------------------------------------------------------------------

# The friction velocity of every column of the bottom wall, from the wall
# model's own operator: a step of length zero leaves the velocity exactly
# as it is ((u − 0)/(1 + 0) = u) and fills `ustar` with the law's u*.
function _wall_ustar(wm::WallModels.WallModel, sim)
    m = sim.mesh
    us = zeros(Float64, m.Nx, m.Ny)
    WallModels.apply_wall_stress!(sim, wm, 0.0; side = :bottom, ustar = us)
    return us
end

"""
    diagnostics(wm::WallModels.WallModel, sim) -> Vector{Diagnostic}

An equilibrium wall model: `ustar_mean` and `ustar_max`, the friction
velocity of its law over the columns of the bottom wall (whole plane,
`:local` formulation, first-cell speed), computed by
[`apply_wall_stress!`](@ref Aeolus.WallModels.apply_wall_stress!) itself
with a zero step. A run whose wall model is applied with a mask, an
input filter or at the top (the Berlin ground, a `WallBoundedCase`)
reports through its own component instead.
"""
function diagnostics(wm::WallModels.WallModel, sim)
    return Diagnostic[
        Diagnostic("ustar_mean", "m s-1", () -> _wall_ustar(wm, sim); reduction = :mean,
                   description = "plane-mean friction velocity of the bottom wall"),
        Diagnostic("ustar_max", "m s-1", () -> _wall_ustar(wm, sim); reduction = :max),
    ]
end

function _tble_utau(wm, side::Symbol)
    w = get(wm.walls, side, nothing)
    w === nothing && return Float64[]
    return Float64[w.cols[I].u_τ for I in eachindex(w.cols) if w.active[I]]
end

"""
    diagnostics(wm::WallModels.TBLEWallModel, sim) -> Vector{Diagnostic}

The non-equilibrium wall model, per wall it has been initialised on
(`bottom`, `top`): `<wall>.utau_mean` (the friction velocity of its
active TBLE columns after their last solve, the stress the model
applied) and `<wall>.kappa_wm` (the relaxed dynamic κ of the wall, or
the fixed one). The walls are created at the model's first application,
so declare after that (or the list is empty).
"""
function diagnostics(wm::WallModels.TBLEWallModel, sim)
    ds = Diagnostic[]
    for side in sort!(collect(keys(wm.walls)))
        s = string(side)
        push!(ds, Diagnostic(s * ".utau_mean", "m s-1", () -> _tble_utau(wm, side);
                             reduction = :mean,
                             description = "mean friction velocity of the active TBLE columns"))
        push!(ds, Diagnostic(s * ".kappa_wm", "1", () -> wm.walls[side].κ_dyn;
                             description = "the wall's (dynamic) von Kármán constant"))
    end
    return ds
end

"""
    diagnostics(case::WallModels.WallBoundedCase, sim) -> Vector{Diagnostic}

The wall-bounded harness: `tau_wall_bottom`, `tau_wall_top` (the
plane-mean kinematic wall stresses the last step applied, m² s⁻²), and,
with a scalar, `q_wall_bottom`, `q_wall_top`; plus the diagnostics of its
wall model under `wall.` (the TBLE model's; an equilibrium model's own
stress is already `tau_wall_*`). Pass `case.closure` separately for the
closure's.
"""
function diagnostics(case::WallModels.WallBoundedCase, sim)
    ds = Diagnostic[
        Diagnostic("tau_wall_bottom", "m2 s-2", () -> case.τ_wall_bottom; cadence = :step),
        Diagnostic("tau_wall_top", "m2 s-2", () -> case.τ_wall_top; cadence = :step),
    ]
    if case.θ_walls !== nothing
        push!(ds, Diagnostic("q_wall_bottom", "K m s-1", () -> case.q_wall_bottom; cadence = :step))
        push!(ds, Diagnostic("q_wall_top", "K m s-1", () -> case.q_wall_top; cadence = :step))
    end
    case.wall isa WallModels.TBLEWallModel && append!(ds, prefixed("wall", diagnostics(case.wall, case.sim)))
    return ds
end

# ---------------------------------------------------------------------------
# nesting
# ---------------------------------------------------------------------------

"""
    diagnostics(run::NestedRun, run) -> Vector{Diagnostic}

A nested run: from its [`NestRunStats`](@ref Aeolus.Nesting.NestRunStats),
every step, `parent_steps`, `child_substeps`, `parent_retries`,
`child_retries`, `substeps_last` and `max_child_cfl`, `max_clock_error`;
on the costly cadence the wall times `wall_parent`, `wall_child`,
`wall_boundary`, `wall_feedback`; and the diagnostics of the two domains
under `parent.` and `child.`. Use it as the run's `sim`:
`collect_diagnostics(run)`.
"""
function diagnostics(run::Nesting.NestedRun, _)
    st() = run.stats
    everystep(name, units, f; kw...) = Diagnostic(name, units, f; cadence = :step, kw...)
    ds = Diagnostic[
        everystep("parent_steps", "1", () -> st().parent_steps),
        everystep("child_substeps", "1", () -> st().child_substeps),
        everystep("parent_retries", "1", () -> st().parent_retries),
        everystep("child_retries", "1", () -> st().child_retries),
        everystep("substeps_last", "1", () -> st().n_last; description = "child substeps of the last parent step"),
        everystep("max_child_cfl", "1", () -> st().max_child_cfl),
        everystep("max_clock_error", "1", () -> st().max_clock_error),
        Diagnostic("wall_parent", "s", () -> st().wall_parent),
        Diagnostic("wall_child", "s", () -> st().wall_child),
        Diagnostic("wall_boundary", "s", () -> st().wall_boundary),
        Diagnostic("wall_feedback", "s", () -> st().wall_feedback),
    ]
    append!(ds, prefixed("parent", diagnostics(run.parent, run.parent)))
    append!(ds, prefixed("child", diagnostics(run.child, run.child)))
    return ds
end
