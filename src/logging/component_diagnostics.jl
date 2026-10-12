# The library's implementations of the diagnostics registry (LOG-c, #335):
# `RunLog.diagnostics(component, sim)` for the simulations, the SGS
# closures, the wall models and the nested run. Included at the top level
# of `module Aeolus`, after every submodule, because it needs their types;
# each source delegates to the component's own diagnostic function (one
# source of truth): `kinetic_energy`, `max_divergence`, `sgs_fields`,
# `apply_wall_stress!`'s friction velocity, the TBLE columns' u_τ,
# `NestRunStats`.

import .RunLog: diagnostics, profile_diagnostics, reduce_diagnostic
using .RunLog: Diagnostic, ProfileDiagnostic, prefixed
using .Backends: plane_mean!, range_max, plane_reduce!, ordered_sum
using .Simulations: step_record

# ---------------------------------------------------------------------------
# the deterministic reductions of a Field (#336)
# ---------------------------------------------------------------------------
# A source that returns a `Field` is reduced over the field's interior with
# the thread-invariant reductions of the KernelAbstractions port: `range_max`
# for the extrema (exact, NaN-propagating: the bits of any serial maximum)
# and the canonical plane sums of `plane_reduce!` + `ordered_sum` for the
# sums (the same bits at every thread count).

@inline _val_at(i, j, k, a) = @inbounds a[i, j, k]
@inline _neg_at(i, j, k, a) = @inbounds -a[i, j, k]
@inline _absv_at(i, j, k, a) = @inbounds abs(a[i, j, k])
@inline _sq_at(i, j, k, a) = @inbounds abs2(a[i, j, k])

function _field_sum(body, f::Fields.Field)
    I3 = Grids.interior(f.mesh)
    p = plane_reduce!(zeros(Float64, length(I3[3])), Backends.backend(f.data), body, I3, f.data)
    return Float64(ordered_sum(p))
end

"""
    reduce_diagnostic(reduction, f::Field) -> Float64

A [`Diagnostic`](@ref Aeolus.RunLog.Diagnostic) source that returns a
`Field` is reduced over the field's **interior** with the
KernelAbstractions port's thread-invariant reductions (#336): `:max`,
`:min`, `:maxabs` with `range_max` (exact and NaN-propagating, the bits
of a serial maximum), `:sum`, `:mean`, `:rms` with the canonical plane
sums (`plane_reduce!`, `ordered_sum`), the same bits at any thread count.
"""
function reduce_diagnostic(r::Symbol, f::Fields.Field)
    I3 = Grids.interior(f.mesh)
    n = prod(length, I3)
    n == 0 && return NaN
    b = Backends.backend(f.data)
    r === :max && return Float64(range_max(b, _val_at, I3, f.data))
    r === :min && return -Float64(range_max(b, _neg_at, I3, f.data))
    r === :maxabs && return Float64(range_max(b, _absv_at, I3, f.data))
    r === :sum && return _field_sum(_val_at, f)
    r === :mean && return _field_sum(_val_at, f) / n
    r === :rms && return sqrt(_field_sum(_sq_at, f) / n)
    throw(ArgumentError("reduce_diagnostic: unknown reduction :$r"))
end

# ---------------------------------------------------------------------------
# the step cores' record (#336)
# ---------------------------------------------------------------------------
# The cheap quantities the step cores record (`Simulations.StepRecord`):
# every row, over the steps since the previous row. The wall times per
# stage: cumulative, on the costly cadence.

function _step_diagnostics(sim)
    r = step_record(sim)
    ds = Diagnostic[
        Diagnostic("cfl_adv", "1", () -> r.cfl_adv; cadence = :step,
                   description = "largest advective Courant number of a step since the last row, from its dt pick"),
        Diagnostic("cfl_diff", "1", () -> r.cfl_diff; cadence = :step,
                   description = "largest diffusive number dt max(nu+nu_t, kappa_t) sum(dx^-2) of a step since the last row"),
        Diagnostic("solver_iters", "1", () -> r.solver_iters; cadence = :step,
                   description = "pressure-solver iterations since the last row (0 for a direct solver)"),
        Diagnostic("solver_residual", "1", () -> r.solver_residual; cadence = :step,
                   description = "largest final residual of a pressure solve since the last row, the solver's own measure"),
        Diagnostic("wall_step", "s", () -> r.wall_step;
                   description = "wall time inside the step cores, cumulative"),
        Diagnostic("wall_pressure", "s", () -> r.wall_pressure;
                   description = "wall time in the pressure solves, cumulative"),
        Diagnostic("wall_sgs", "s", () -> r.wall_sgs;
                   description = "wall time in the SGS closure, cumulative"),
        Diagnostic("wall_cfl", "s", () -> r.wall_cfl;
                   description = "wall time in the CFL sweeps of the dt picks, cumulative"),
        Diagnostic("wall_guard", "s", () -> r.wall_guard;
                   description = "wall time in step_adaptive!'s snapshot, guard and rollbacks, cumulative"),
    ]
    return ds
end

# The CFL numbers of the current state at the last step's dt, from one
# fused sweep (the advective rate and the SGS maxima of the last
# evaluation), shared by the two diagnostics of a row.
mutable struct _StateCFL
    step::Int
    adv::Float64
    diff::Float64
end

function _state_cfl!(c::_StateCFL, sim::Simulations.BoussinesqSimulation)
    r = step_record(sim)
    c.step == r.steps && return c
    for f in (sim.u, sim.v, sim.w)
        BoundaryConditions.fill_halos!(f, sim.bcs)
    end
    rate, νh, νv, κh, κv = TimeIntegration._cfl_maxima(sim, Operators.SGSFields(sim.νt, sim.κt), true)
    m = sim.mesh
    inv2 = 1 / m.dx^2 + 1 / m.dy^2 + 1 / Simulations._dz_min(m)^2
    c.adv = r.dt * rate
    c.diff = r.dt * max(sim.ν + max(νh, νv), max(κh, κv)) * inv2
    c.step = r.steps
    return c
end

function _state_cfl!(c::_StateCFL, sim::Anelastic.MassWeightedSimulation)
    r = step_record(sim)
    c.step == r.steps && return c
    for f in (sim.u, sim.v, sim.w)
        BoundaryConditions.fill_halos!(f, sim.bcs)
    end
    c.adv = r.dt * TimeIntegration._max_advective_rate(sim.u.data, sim.v.data, sim.w.data, sim.mesh)
    c.diff = NaN
    c.step = r.steps
    return c
end

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
the last evaluation) and, with moisture, `qv_max`; the extrema with the
port's thread-invariant reductions of a Field
([`reduce_diagnostic`](@ref Aeolus.RunLog.reduce_diagnostic)).

From the step cores' [`StepRecord`](@ref Aeolus.Simulations.StepRecord)
(#336), **every row**, over the steps since the previous row: `cfl_adv`
and `cfl_diff` (the largest Courant and diffusive numbers of a step, from
its dt pick; `NaN` for steps taken without one), `solver_iters`,
`solver_residual`, `implicit_iters` (`step_imex!`'s diffusion CG). On
the costly cadence: the cumulative wall times `wall_step`,
`wall_pressure`, `wall_sgs`, `wall_cfl`, `wall_guard`, and
`cfl_adv_state`, `cfl_diff_state`, the Courant and diffusive numbers of
the current state at the last step's dt (one fused sweep: the CFL of a
fixed-step run, which makes no dt pick).
"""
function diagnostics(sim::Simulations.BoussinesqSimulation, ::Simulations.BoussinesqSimulation)
    ds = Diagnostic[
        Diagnostic("ke_mean", "m2 s-2", () -> Simulations.kinetic_energy(sim) / _domain_volume(sim.mesh);
                   description = "volume-mean resolved kinetic energy per unit mass"),
        Diagnostic("max_abs_u", "m s-1", () -> sim.u; reduction = :maxabs),
        Diagnostic("max_abs_v", "m s-1", () -> sim.v; reduction = :maxabs),
        Diagnostic("max_abs_w", "m s-1", () -> sim.w; reduction = :maxabs),
        Diagnostic("theta_min", "K", () -> sim.θ; reduction = :min),
        Diagnostic("theta_max", "K", () -> sim.θ; reduction = :max),
        Diagnostic("max_div", "s-1", () -> Simulations.max_divergence(sim);
                   description = "max |∇·u| over the interior cells"),
        Diagnostic("nu_t_max", "m2 s-1", () -> sim.νt; reduction = :max,
                   description = "max SGS eddy viscosity of the last evaluation"),
    ]
    sim.has_moisture &&
        push!(ds, Diagnostic("qv_max", "kg kg-1", () -> sim.qv; reduction = :max))
    append!(ds, _step_diagnostics(sim))
    r = step_record(sim)
    push!(ds, Diagnostic("implicit_iters", "1", () -> r.implicit_iters; cadence = :step,
                         description = "implicit-diffusion CG iterations of step_imex! since the last row"))
    sc = _StateCFL(-1, NaN, NaN)
    push!(ds, Diagnostic("cfl_adv_state", "1", () -> _state_cfl!(sc, sim).adv;
                         description = "advective Courant number of the current state at the last step's dt"))
    push!(ds, Diagnostic("cfl_diff_state", "1", () -> _state_cfl!(sc, sim).diff;
                         description = "diffusive number of the current state (SGS coefficients of the last evaluation) at the last step's dt"))
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
pseudo-incompressible one (the plain `∇·u` is non-zero by design). And
the step-record diagnostics of the Boussinesq core (#336) except
`implicit_iters` and `cfl_diff_state` (the sound-proof cores have no
IMEX step; their CFL of the state is the advective one, `cfl_adv_state`).
"""
_constraint_residual(sim::Anelastic.MassWeightedSimulation{Anelastic.AnelasticWeighting}) =
    Anelastic.max_anelastic_divergence(sim)
_constraint_residual(sim::Anelastic.MassWeightedSimulation{Anelastic.PseudoIncompressibleWeighting}) =
    Anelastic.max_pseudo_incompressible_divergence(sim)

function diagnostics(sim::Anelastic.MassWeightedSimulation, ::Anelastic.MassWeightedSimulation)
    ds = Diagnostic[
        Diagnostic("ke", "J", () -> Simulations.kinetic_energy(sim);
                   description = "density-weighted resolved kinetic energy"),
        Diagnostic("max_abs_u", "m s-1", () -> sim.u; reduction = :maxabs),
        Diagnostic("max_abs_v", "m s-1", () -> sim.v; reduction = :maxabs),
        Diagnostic("max_abs_w", "m s-1", () -> sim.w; reduction = :maxabs),
        Diagnostic("theta_min", "K", () -> sim.θ; reduction = :min),
        Diagnostic("theta_max", "K", () -> sim.θ; reduction = :max),
        Diagnostic("max_div", "s-1", () -> _constraint_residual(sim);
                   description = "max residual of the core's continuity constraint"),
    ]
    append!(ds, _step_diagnostics(sim))
    sc = _StateCFL(-1, NaN, NaN)
    push!(ds, Diagnostic("cfl_adv_state", "1", () -> _state_cfl!(sc, sim).adv;
                         description = "advective Courant number of the current state at the last step's dt"))
    return ds
end

# ---------------------------------------------------------------------------
# SGS closures
# ---------------------------------------------------------------------------

function _sgs_diagnostics(c, sim)
    f() = Operators.sgs_fields(c, sim)
    return Diagnostic[
        Diagnostic("nu_h_max", "m2 s-1", () -> f().νh; reduction = :max,
                   description = "max horizontal eddy viscosity of the last evaluation"),
        Diagnostic("nu_v_max", "m2 s-1", () -> f().νv; reduction = :max,
                   description = "max vertical eddy viscosity of the last evaluation"),
        Diagnostic("kappa_h_max", "m2 s-1", () -> f().κh; reduction = :max,
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
    push!(ds, Diagnostic("e_mean", "m2 s-2", () -> c.e; reduction = :mean,
                         description = "SGS TKE, mean over the interior cells"))
    push!(ds, Diagnostic("e_max", "m2 s-2", () -> c.e; reduction = :max))
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
input filter or at the top (the Berlin ground, a wall-bounded channel)
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

# ---------------------------------------------------------------------------
# nesting
# ---------------------------------------------------------------------------

"""
    diagnostics(run::NestedRun, run) -> Vector{Diagnostic}

A nested run: from its [`NestRunStats`](@ref Aeolus.Nesting.NestRunStats),
every step, `parent_steps`, `child_substeps`, `parent_retries`,
`child_retries`, `substeps_last` and `max_child_cfl`, `max_clock_error`;
on the costly cadence the wall times `wall_parent`, `wall_child`,
`wall_boundary`, `wall_feedback`; from the run's step record (#336)
`cfl_adv` (every row: the largest child-substep CFL since the last row)
and `wall_step` (the windows' wall time); for a two-way run that records
its ledger, `ledger_<quantity>` (the relative conservation error of the
last window: each refluxed scalar, `momentum_x`, `mass`); and the
diagnostics of the two domains under `parent.` and `child.`. Use it as
the run's `sim`: `collect_diagnostics(run)`.
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
    r = run.step_record
    push!(ds, everystep("cfl_adv", "1", () -> r.cfl_adv;
                        description = "largest advective CFL of a child substep since the last row"))
    push!(ds, Diagnostic("wall_step", "s", () -> r.wall_step;
                         description = "wall time of the parent-step windows, cumulative"))
    if run.work !== nothing && run.record           # the two-way conservation ledger
        nest = run.nest
        for key in vcat(run.work.keys, [:momentum_x, :mass])
            push!(ds, Diagnostic("ledger_" * string(key), "1", () -> get(nest.ledger, key, NaN);
                                 description = "relative two-way conservation error of $key in the last window"))
        end
    end
    append!(ds, prefixed("parent", diagnostics(run.parent, run.parent)))
    append!(ds, prefixed("child", diagnostics(run.child, run.child)))
    return ds
end

# ---------------------------------------------------------------------------
# profiles (LOG-e, #337): diagnostics.nc
# ---------------------------------------------------------------------------
# Each source delegates to the library's own reduction: `plane_mean!` (the
# deterministic plane sum of the KernelAbstractions port),
# `horizontal_variance`, `resolved_flux_wθ`.

_zc(m) = collect(Float64, Grids.znodes(m, Grids.CellCenter()))
_zf(m) = collect(Float64, Grids.znodes(m, Grids.FaceZ()))
_pmean(f) = () -> plane_mean!(zeros(Float64, f.mesh.Nz), f)

function _core_profiles(sim)
    m = sim.mesh
    zc, zf = _zc(m), _zf(m)
    return ProfileDiagnostic[
        ProfileDiagnostic("u", "m s-1", "z", zc, _pmean(sim.u); description = "horizontal mean of u"),
        ProfileDiagnostic("v", "m s-1", "z", zc, _pmean(sim.v); description = "horizontal mean of v"),
        ProfileDiagnostic("theta", "K", "z", zc, _pmean(sim.θ);
                          description = "horizontal mean potential temperature"),
        ProfileDiagnostic("w_var", "m2 s-2", "zf", zf, () -> Simulations.horizontal_variance(sim.w);
                          description = "horizontal variance of w at the w faces"),
    ]
end

"""
    profile_diagnostics(sim::BoussinesqSimulation, sim) -> Vector{ProfileDiagnostic}

The dynamical core's profiles for `diagnostics.nc`: horizontal means `u`,
`v`, `theta` (and `qv` with moisture) on the cell-centre heights `z`
(the staggered `u`, `v` sit at the centre heights), `w_var`, the
horizontal variance of `w` on the face heights `zf`
([`horizontal_variance`](@ref Aeolus.Simulations.horizontal_variance)),
and `w_theta`, the resolved kinematic heat flux `⟨w'θ'⟩` on `z`
([`resolved_flux_wθ`](@ref Aeolus.Simulations.resolved_flux_wθ)).
"""
function profile_diagnostics(sim::Simulations.BoussinesqSimulation, ::Simulations.BoussinesqSimulation)
    ps = _core_profiles(sim)
    zc = _zc(sim.mesh)
    sim.has_moisture &&
        push!(ps, ProfileDiagnostic("qv", "kg kg-1", "z", zc, _pmean(sim.qv);
                                    description = "horizontal mean water-vapour mixing ratio"))
    push!(ps, ProfileDiagnostic("w_theta", "K m s-1", "z", zc, () -> Simulations.resolved_flux_wθ(sim);
                                description = "resolved kinematic heat flux <w'theta'>"))
    return ps
end

"""
    profile_diagnostics(sim::MassWeightedSimulation, sim) -> Vector{ProfileDiagnostic}

The sound-proof cores' profiles: `u`, `v`, `theta` on `z` and `w_var` on
`zf`, as for the Boussinesq core.
"""
profile_diagnostics(sim::Anelastic.MassWeightedSimulation, ::Anelastic.MassWeightedSimulation) =
    _core_profiles(sim)

"""
    profile_diagnostics(closure::AbstractSGSClosure, sim) -> Vector{ProfileDiagnostic}

Any SGS closure: `nu_v`, the horizontal mean of its vertical eddy
viscosity of the last evaluation, through
[`sgs_fields`](@ref Aeolus.Operators.sgs_fields). A `TKEClosure` adds
`e`, the horizontal mean of its SGS energy.
"""
profile_diagnostics(c::Operators.AbstractSGSClosure, sim) = ProfileDiagnostic[
    ProfileDiagnostic("nu_v", "m2 s-1", "z", _zc(sim.mesh),
                      () -> plane_mean!(zeros(Float64, sim.mesh.Nz), Operators.sgs_fields(c, sim).νv);
                      description = "horizontal mean vertical eddy viscosity")]

function profile_diagnostics(c::PrognosticSGS.TKEClosure, sim)
    ps = invoke(profile_diagnostics, Tuple{Operators.AbstractSGSClosure,Any}, c, sim)
    push!(ps, ProfileDiagnostic("e", "m2 s-2", "z", _zc(sim.mesh), _pmean(c.e);
                                description = "horizontal mean SGS TKE"))
    return ps
end

"""
    profile_diagnostics(run::NestedRun, run) -> Vector{ProfileDiagnostic}

The profiles of both domains, under `parent.` and `child.` (names and
axes).
"""
profile_diagnostics(run::Nesting.NestedRun, _) =
    vcat(prefixed("parent", profile_diagnostics(run.parent, run.parent)),
         prefixed("child", profile_diagnostics(run.child, run.child)))

# A nested run's monitor row covers the steps of the run and of its two
# domains since the previous row (#336).
IO._reset_records!(run::Nesting.NestedRun) =
    (Simulations.reset_interval!(run.step_record);
     Simulations.reset_interval!(Simulations.step_record(run.parent));
     Simulations.reset_interval!(Simulations.step_record(run.child)); nothing)
