module Simulations

using ..Grids
using ..Grids: AbstractMesh, RegularGrid, CellCenter, FaceX, FaceY, FaceZ, interior
using ..Fields
using ..Fields: Field, interior_view, interior_map!
using ..Backends: Backends, plane_partials!, plane_reduce!, plane_mean!, ordered_sum, range_max,
                  range_map!, interior_max, launch_weighted!
using ..BoundaryConditions
using ..BoundaryConditions: FieldBoundaryConditions, fill_halos!, Periodic, ImpermeableWall,
                            update_open_boundaries!, prepare_open_boundaries!,
                            assert_open_face_correction, open_boundary_snapshot,
                            combine_open_boundaries!, mark_open_boundary_stage!
using ..Operators
using ..Operators: divergence, momentum_advect!, laplacian, ∂x, ∂y, ∂z,
                   advect_centered!,
                   add_buoyancy!, add_coriolis!,
                   compute_smagorinsky_νt!,
                   add_eddy_viscosity_u!, add_eddy_viscosity_v!, add_eddy_viscosity_w!,
                   add_eddy_diffusion_scalar!,
                   add_surface_flux!, add_sponge!,
                   AbstractSGSClosure, compute_sgs!, add_sgs_momentum!, add_sgs_scalar!,
                   advance_sgs!, _active_sgs, _check_sgs_stress, sgs_tracer_diffusivity!,
                   _sq_dev_at                  # (f - m_k)², shared with abl_diagnostics
using ..PressureSolvers
using ..PressureSolvers: AbstractPressureSolver, FFTPoissonSolver, CosineFFTPoissonSolver, solve!,
                        poisson_axis_bcs, solver_stats, source_buffer, solve_source_buffer!
# the fused step-core kernels (SC-d, #404)
using ..Operators.FiniteVolume: _fused_tendencies!, _correct_velocity!, _face_gradient_slabs!,
                                _divergence_into!
# the blocked tendency + predictor sweep (SC-e, #405)
using ..Operators.FiniteVolume: blocked_predictor!, sweep_buffers, sweep_slabs
import ..TimeSteppers: step!    # extend the generic `step!` from TimeSteppers

export BoussinesqSimulation, StepWorkspaces, StepRecord, step_record, set_step_hook!, step_hook,
       step_begun, guard_covered,
       reset_interval!, run!, step_rk3!, kinetic_energy, max_divergence,
       max_νt, mean_θ_profile, sponge_target,
       horizontal_mean, horizontal_variance, resolved_flux_wθ

include("simulations/step_record.jl")   # the per-step record and diagnostics hook (#336)

"""
    StepWorkspaces()

The per-simulation scratch of the step drivers: the step-start state of
[`step_rk3!`](@ref) ([`RK3Workspace`](@ref)), the retry snapshot of
`step_adaptive!` (`TimeIntegration.StateSnapshot`) and the IMEX stage
state of `step_imex!` (`TimeIntegration.IMEXWorkspace`), and the
[`StepRecord`](@ref) of the step cores (`rec`, #336). Each workspace slot is
allocated on first use, on the simulation's backend
([`Backends.allocate`](@ref Aeolus.Backends.allocate)), and lives as long
as the simulation (`rk3_workspace`, `state_snapshot`, `imex_workspace`).

Before KA-j (#306) these lived in process-global `WeakKeyDict`s keyed by
the simulation. A lookup there is not thread-safe, so stepping two domains
concurrently could race on the cache; a field of the simulation cannot.
"""
mutable struct StepWorkspaces
    rk3::Any
    snapshot::Any
    imex::Any
    sweep::Any            # FiniteVolume.SweepBuffers of the blocked core (SC-e, #405)
    rec::StepRecord
end
StepWorkspaces() = StepWorkspaces(nothing, nothing, nothing, nothing, StepRecord())

"""
    BoussinesqSimulation(mesh; kwargs...)

Dry incompressible Boussinesq LES on a staggered Arakawa-C mesh with a
fractional-step / pressure-projection driver.

`mesh` is a uniform [`RegularGrid`](@ref) or a vertically stretched
[`StretchedGrid`](@ref Aeolus.StretchedGrids.StretchedGrid) (#151): on the
latter every operator of the core (momentum and scalar advection, molecular
and SGS diffusion, the Smagorinsky length scale, buoyancy, the projection)
uses the local cell thickness `cell_thickness(mesh)[k]` and centre spacing
`center_spacing(mesh)[k]`, and the default pressure solver is the direct
stretched-z solver (see the [Stretched vertical grid](@ref) theory page).
A `:uniform` `StretchedGrid` reproduces the `RegularGrid` run.

Prognostic state:

  * `u @ FaceX`, `v @ FaceY`, `w @ FaceZ` — velocity components.
  * `p @ CellCenter`                       — pressure.
  * `θ @ CellCenter`                       — potential temperature.

Constructor keyword arguments (each optional physics is disabled when
its parameter is `0`):

| Keyword       | Default                      | Effect                              |
| :------------ | :--------------------------- | :---------------------------------- |
| `ν`           | `0`                          | Molecular viscosity.                |
| `θ₀`          | `0`                          | Buoyancy reference (`0` → off).    |
| `g_accel`     | `9.81`                       | Gravitational acceleration.         |
| `f_coriolis`  | `0`                          | Coriolis parameter.                 |
| `C_s`         | `0`                          | Smagorinsky coefficient (`0` → off).|
| `Pr_t`        | `1/3`                        | Turbulent Prandtl number.           |

The SGS coefficients of the last tendency evaluation are kept in
`sim.νt` (eddy viscosity) and `sim.κt` (eddy diffusivity of θ and, with `moisture = true`, of qv), both
cell-centred: with `C_s ≠ 0` and no `sgs` keyword the built-in closure is
[`StaticSmagorinsky`](@ref Aeolus.Operators.StaticSmagorinsky)`(C_s)`,
`κt = νt/Prₜ`; any [`AbstractSGSClosure`](@ref
Aeolus.Operators.AbstractSGSClosure) passed as `sgs` replaces it.
| `sponge_α_max`, `sponge_z_start` | `0`, top | Rayleigh sponge on `u, v, w` above `z_start`. |
| `sponge_target` | `nothing` (zero)           | What the sponge relaxes toward (N-open-d, #262; see [`sponge_target`](@ref)). Required with an open top. |
| `bcs`         | `FieldBoundaryConditions(:periodic)` | Velocity BCs.               |
| `θ_bcs`       | `bcs`                        | Scalar BCs (may differ from velocity). |
| `solver`      | `FFTPoissonSolver(mesh)`     | Any `AbstractPressureSolver`.       |

Advance with `step!(sim, dt)` — see [`step!`](@ref). All buffers are
allocated at construction; `step!` does no heap allocation.

Diagnostics: [`kinetic_energy`](@ref), [`max_divergence`](@ref),
[`max_νt`](@ref), [`mean_θ_profile`](@ref).

See the [Fractional-step projection](@ref) theory page for the
algorithm and the [Quick start](@ref) for a worked example.
"""
mutable struct BoussinesqSimulation{T,
                                    M<:AbstractMesh{T},
                                    BCS<:FieldBoundaryConditions,
                                    TBCS<:FieldBoundaryConditions,
                                    PS<:AbstractPressureSolver}
    mesh::M
    bcs::BCS               # velocity BCs
    θ_bcs::TBCS            # scalar BCs (may differ from velocity)
    solver::PS
    # physics parameters (0 or NaN → disabled)
    ν::T
    θ₀::T
    g_accel::T
    f_coriolis::T
    C_s::T
    Pr_t::T
    # Phase 7: surface flux for θ and Rayleigh sponge on velocity
    θ_flux_bottom::T
    θ_flux_top::T
    sponge_α_max::T
    sponge_z_start::T
    sponge_target::NamedTuple{(:u, :v, :w)}   # relaxation targets of the sponge (N-open-d, #262)
    # enable flags (frozen at construction from parameter values)
    has_viscosity::Bool
    has_buoyancy::Bool
    has_coriolis::Bool
    has_sgs::Bool
    has_surface_flux::Bool
    has_sponge::Bool
    has_moisture::Bool
    # clock
    t::T

    # prognostic
    u::Field{FaceX,      T, Array{T,3}, M}
    v::Field{FaceY,      T, Array{T,3}, M}
    w::Field{FaceZ,      T, Array{T,3}, M}
    p::Field{CellCenter, T, Array{T,3}, M}
    θ::Field{CellCenter, T, Array{T,3}, M}
    qv::Field{CellCenter, T, Array{T,3}, M}   # water vapor mixing ratio (Phase 11)

    # tendencies (persistent buffers; unused by the hook-free core, SC-e)
    tu::Field{FaceX,      T, Array{T,3}, M}
    tv::Field{FaceY,      T, Array{T,3}, M}
    tw::Field{FaceZ,      T, Array{T,3}, M}
    tθ::Field{CellCenter, T, Array{T,3}, M}
    tqv::Field{CellCenter, T, Array{T,3}, M}

    # scratch buffers
    div_buf::Field{CellCenter, T, Array{T,3}, M}
    φ::Field{CellCenter,       T, Array{T,3}, M}
    dφdx::Field{FaceX,         T, Array{T,3}, M}
    dφdy::Field{FaceY,         T, Array{T,3}, M}
    dφdz::Field{FaceZ,         T, Array{T,3}, M}
    lap_u::Field{FaceX,        T, Array{T,3}, M}
    lap_v::Field{FaceY,        T, Array{T,3}, M}
    lap_w::Field{FaceZ,        T, Array{T,3}, M}
    lap_θ::Field{CellCenter,   T, Array{T,3}, M}
    νt::Field{CellCenter,      T, Array{T,3}, M}
    κt::Field{CellCenter,      T, Array{T,3}, M}   # SGS scalar diffusivity (#167)
    # the drivers' scratch (RK3 stage state, retry snapshot, IMEX), per simulation (KA-j, #306)
    workspaces::StepWorkspaces
end

function BoussinesqSimulation(mesh::AbstractMesh{T};
                              ν::Real                 = 0,
                              θ₀::Real                = 0,     # 0 → buoyancy off
                              g_accel::Real           = 9.81,
                              f_coriolis::Real        = 0,
                              C_s::Real               = 0,     # 0 → SGS off
                              Pr_t::Real              = 1/3,
                              θ_flux_bottom::Real     = 0,     # Phase 7
                              θ_flux_top::Real        = 0,
                              sponge_α_max::Real      = 0,     # 0 → sponge off
                              sponge_z_start::Real    = mesh.z0 + mesh.Lz,   # never triggers
                              sponge_target           = nothing,         # N-open-d (#262)
                              moisture::Bool          = false,           # Phase 11
                              bcs::FieldBoundaryConditions = FieldBoundaryConditions(:periodic),
                              θ_bcs::FieldBoundaryConditions = bcs,
                              solver::AbstractPressureSolver = _default_solver(mesh, bcs)) where {T}
    _check_boussinesq_mesh(mesh)
    Tp = T
    u = Field(FaceX(), mesh); v = Field(FaceY(), mesh); w = Field(FaceZ(), mesh)
    p = Field(CellCenter(), mesh); θ = Field(CellCenter(), mesh)

    tu = Field(FaceX(), mesh); tv = Field(FaceY(), mesh); tw = Field(FaceZ(), mesh)
    tθ = Field(CellCenter(), mesh)
    qv = Field(CellCenter(), mesh); tqv = Field(CellCenter(), mesh)

    div_buf = Field(CellCenter(), mesh)
    φ    = Field(CellCenter(), mesh)
    dφdx = Field(FaceX(), mesh)
    dφdy = Field(FaceY(), mesh)
    dφdz = Field(FaceZ(), mesh)
    lu   = Field(FaceX(), mesh); lv = Field(FaceY(), mesh); lw = Field(FaceZ(), mesh)
    lθ   = Field(CellCenter(), mesh)
    νt   = Field(CellCenter(), mesh)
    κt   = Field(CellCenter(), mesh)

    has_visc = ν != 0
    has_buoy = θ₀ != 0
    has_cor  = f_coriolis != 0
    has_sgs  = C_s != 0
    has_sflux  = θ_flux_bottom != 0 || θ_flux_top != 0
    has_sponge = sponge_α_max != 0 && sponge_z_start < (mesh.z0 + mesh.Lz)
    has_moist  = moisture
    target = _sponge_target(sponge_target, mesh, bcs, has_sponge)

    return BoussinesqSimulation(mesh, bcs, θ_bcs, solver,
                                Tp(ν), Tp(θ₀), Tp(g_accel), Tp(f_coriolis),
                                Tp(C_s), Tp(Pr_t),
                                Tp(θ_flux_bottom), Tp(θ_flux_top),
                                Tp(sponge_α_max), Tp(sponge_z_start), target,
                                has_visc, has_buoy, has_cor, has_sgs,
                                has_sflux, has_sponge, has_moist,
                                zero(Tp),
                                u, v, w, p, θ, qv,
                                tu, tv, tw, tθ, tqv,
                                div_buf, φ, dφdx, dφdy, dφdz,
                                lu, lv, lw, lθ, νt, κt, StepWorkspaces())
end

# -------------------------------------------------------------------------
# The sponge target (N-open-d, #262)
# -------------------------------------------------------------------------

"""
    sponge_target(data, mesh) -> NamedTuple{(:u, :v, :w)}
    sponge_target(; u = 0, v = 0, w = 0) -> NamedTuple{(:u, :v, :w)}

The relaxation targets of the core's Rayleigh sponge, one per velocity
component, in a form the one damping kernel (`_add_damping!`, through
[`add_sponge!`](@ref Aeolus.Operators.add_sponge!)) takes: a number, a
level vector (length `Nz`, the interior levels of that component's own
location) or a `Field`.

The first form evaluates boundary data on the levels of each component:
a [`ProfileBoundaryData`](@ref Aeolus.BoundaryConditions.ProfileBoundaryData)
profile that is a number stays a number, a function of height `z -> value`
becomes the level vector at the heights of `u`, `v` (cell centres) and
`w` (faces), and a storage-indexed vector (length `Nz + 2Hz`) gives its
interior levels. `w` defaults to zero. This is the target of an **open
top** (N-open-d, #262): the sponge layer then relaxes the flow toward the
state the boundary data imposes on the top face (the parent's, in a
nest), not toward rest. A sponge that relaxes toward zero below an open
top is a momentum sink against the inflow: with a free-atmosphere wind
`U` it removes `α(z)·U` everywhere in the layer, and nothing imposes the
wind there.

Pass the result as `sponge_target` to [`BoussinesqSimulation`](@ref) or a
sound-proof core.
"""
sponge_target(; u = 0, v = 0, w = 0) = (u = u, v = v, w = w)
function sponge_target(data::ProfileBoundaryData, mesh::AbstractMesh)
    f = data.fields
    lev(name, loc) = _sponge_levels(haskey(f, name) ? getfield(f, name) : 0, mesh, loc)
    return (u = lev(:u, FaceX()), v = lev(:v, FaceY()), w = lev(:w, FaceZ()))
end

_sponge_levels(x::Real, mesh, loc) = x
_sponge_levels(x::AbstractVector, mesh, loc) = [x[mesh.Hz + k] for k in 1:mesh.Nz]
_sponge_levels(f, mesh, loc) = [f(z) for z in Operators.damping_coordinates(mesh, loc)]

# The sponge target a core stores: `nothing` is the Phase-7 zero target,
# which an open top refuses (the error of N-open-d); a NamedTuple may omit
# `w` (zero); boundary data is evaluated on the mesh.
function _sponge_target(target, mesh, bcs::FieldBoundaryConditions, has_sponge::Bool)
    if target === nothing
        has_sponge && bcs.top isa BoundaryConditions.OpenBoundary &&
            throw(ArgumentError(
                "the Rayleigh sponge below an open top would relax u, v, w toward zero, a " *
                "momentum sink against the boundary data; pass sponge_target (e.g. " *
                "sponge_target(data, mesh) for the open boundary's data, or " *
                "sponge_target(u = U, v = V)) — see Simulations.sponge_target (N-open-d, #262)"))
        return (u = 0, v = 0, w = 0)
    end
    target isa BoundaryConditions.AbstractBoundaryData && return sponge_target(target, mesh)
    target isa NamedTuple ||
        throw(ArgumentError("sponge_target must be a NamedTuple (u, v[, w]) or boundary data, " *
                            "got $(typeof(target))"))
    for k in keys(target)
        k in (:u, :v, :w) || throw(ArgumentError("sponge_target: unknown component :$k (use u, v, w)"))
    end
    return (u = get(target, :u, 0), v = get(target, :v, 0), w = get(target, :w, 0))
end

"""
    add_velocity_sponge!(sim) -> sim

Add the core's Rayleigh sponge `−α(z)(u − u_target)` to the momentum
tendencies `sim.tu, sim.tv, sim.tw`, with the profile `sim.sponge_α_max`,
`sim.sponge_z_start` and the targets `sim.sponge_target`
([`sponge_target`](@ref)). The one sponge call of every core (`step!`,
`step_rk3!`, `step_imex!`, `step_adaptive!` and the sound-proof cores):
the zero target of a closed top is `add_sponge!`'s own default, so a
closed run is unchanged bit for bit.
"""
add_velocity_sponge!(sim) =
    add_velocity_sponge!(sim, sim.sponge_α_max, sim.sponge_z_start, sim.sponge_target)

"""
    add_velocity_sponge!(sim, α_max, z_start, target) -> sim

The same sponge with the profile and targets given: the one implementation
behind the core's built-in sponge and the `Sponge` component of the stage
pipeline ([`Aeolus.Components.Sponge`](@ref), ONE-a #450).
"""
function add_velocity_sponge!(sim, α_max, z_start, tg::NamedTuple)
    z_top = sim.mesh.z0 + sim.mesh.Lz
    add_sponge!(sim.tu, sim.u, α_max, z_start, z_top; f_ref = tg.u)
    add_sponge!(sim.tv, sim.v, α_max, z_start, z_top; f_ref = tg.v)
    add_sponge!(sim.tw, sim.w, α_max, z_start, z_top; f_ref = tg.w)
    return sim
end

# -------------------------------------------------------------------------
# One Euler-projection step.
# -------------------------------------------------------------------------

"""
    step!(sim::BoussinesqSimulation, dt; body_force! = nothing, post_predictor! = nothing,
          momentum_advection = momentum_advect!, sgs = nothing,
          sgs_stress = :laplacian)

Advance `sim` by `dt`: the Boussinesq core's step.

A run does not call this: a run is a configuration of the one model, and
[`Components.Model`](@ref Aeolus.Components.Model) steps each domain
through the stage pipeline, which hands its stages 7 and 8
(`:body_force`, `:post_predictor`) and the `[sgs]` closure to the core
through the keywords below. Physics reaches a run as a component, never
by passing a hook here; the hooks are the pipeline's mechanism. Their
other callers are the module tests, the urban site's step
(`Aeolus.Urban`, which passes the pipeline's hooks of its domain)
and the wall-bounded channel harness of the tests.

Two optional callbacks are supported:

  * `body_force!(sim, Δt)` — invoked *after* the built-in tendencies
    are assembled and *before* the predictor step. Explicit source terms
    (Coriolis-like body forces, weak IBM penalty, geostrophic nudging)
    belong here.

  * `post_predictor!(sim, Δt)` — invoked *after* the predictor `u* = u + Δt F`
    and *before* the divergence + Poisson solve. This is the correct
    place for stiff / implicit-style corrections: implicit IBM
    velocity-damping, mask enforcement, anything that would blow up if
    added as an explicit tendency. The pressure projection then makes
    the resulting field divergence-free.

Two keywords swap numerical operators without changing the algorithm:

  * `momentum_advection = momentum_advect!` — the operator
    `f(tu, tv, tw, u, v, w)` that adds `-∂(uⱼuᵢ)/∂xⱼ` to the momentum
    tendencies (halos of `u, v, w` are filled). The default is the
    2nd-order centred flux form; e.g.
    `(tu, tv, tw, u, v, w) -> momentum_advect_weno_z!(tu, tv, tw, u, v, w)`
    (`AdvectionSchemes`) selects the WENO-Z form.

  * `sgs = nothing` — the subgrid-scale closure, any
    [`AbstractSGSClosure`](@ref Aeolus.Operators.AbstractSGSClosure)
    (`StaticSmagorinsky`, `DynamicSmagorinsky`, `ML.MLSGS`, …) or a plain
    function `f(νt, u, v, w)`. When given it replaces the built-in
    Smagorinsky closure and switches the SGS terms on even if `C_s = 0`.
    The closure's coefficients are computed from the current state (halos
    filled) into `sim.νt`/`sim.κt`, its momentum and scalar tendencies are
    added where the built-in closure adds them (the scalar flux to θ and,
    with `moisture = true`, to qv, which also gets `ν∇²qv`), and
    [`advance_sgs!`](@ref Aeolus.Operators.advance_sgs!) commits its memory
    once, after the projection. `StaticSmagorinsky(sim.C_s)` reproduces the
    built-in run bit for bit.

  * `eddy_viscosity! = nothing` — the pre-contract name of `sgs` for a
    function `f(νt, u, v, w)`; still accepted (it is wrapped in a
    `FunctionSGS`, `κt = νt/Prₜ`). Pass one of `sgs`, `eddy_viscosity!`.

**The hook-free core** (SC-e, #405). When nothing adds to the tendencies
between advection/diffusion and the predictor (the default momentum
advection, no SGS closure, buoyancy, Coriolis, surface flux, sponge,
hooks or open boundaries: a DNS), the tendencies and the predictor of u,
v, w, θ (and qv) run as one cache-blocked sweep
(`FiniteVolume.blocked_predictor!`) with the same bits. The tendency
fields `sim.tu, tv, tw, tθ, tqv` are then neither written nor read.

  * `sgs_stress = :laplacian` — the form of the SGS momentum divergence,
    passed to [`add_sgs_momentum!`](@ref Aeolus.Operators.add_sgs_momentum!):
    `:laplacian` is `∂ⱼ(νₜ∂ⱼuᵢ)` (the default; the runs before the SGS closure
    contract bit for bit), `:symmetric` the stress form `∂ⱼ(2νᵢⱼSᵢⱼ)` with the closure's
    horizontal and vertical viscosities
    ([`add_sgs_stress_divergence!`](@ref Aeolus.Operators.add_sgs_stress_divergence!),
    P55-b). `step_rk3!`, `step_imex!`, `step_adaptive!` and the
    sound-proof cores take the same keyword.

Every step core records the cheap quantities of its step (dt and its
limiter, retries, CFL numbers of the dt pick, pressure-solver iterations
and residual, wall time per stage) in the simulation's
[`StepRecord`](@ref) and then calls the simulation's diagnostics hook, if
one is installed ([`set_step_hook!`](@ref), #336). A core called inside
another (`step!` inside `step_adaptive!`) is part of the outer step.
"""
function step!(sim::BoussinesqSimulation{T}, dt::Real; kwargs...) where {T}
    rec = sim.workspaces.rec
    outer = _step_begin!(rec, sim)
    try
        _step_euler!(sim, T(dt); kwargs...)
    catch
        _step_abort!(rec); rethrow()
    end
    return _step_end!(rec, sim, T(dt), outer)
end

function _step_euler!(sim::BoussinesqSimulation{T}, dt::Real;
                      body_force!::Union{Nothing, Function} = nothing,
                      post_predictor!::Union{Nothing, Function} = nothing,
                      momentum_advection::F = momentum_advect!,
                      sgs = nothing,
                      eddy_viscosity! = nothing,
                      sgs_stress::Symbol = :laplacian) where {T, F}
    Δt = T(dt)
    closure = _active_sgs(sim, sgs, eddy_viscosity!)
    prepare_open_boundaries!(sim)          # no-op unless an open face is unfilled
    _euler_projection!(sim, Δt, closure, body_force!, post_predictor!, momentum_advection,
                       _check_sgs_stress(sgs_stress))
    # the one commit of the closure's memory, after the projection
    closure === nothing || advance_sgs!(closure, sim, Δt)
    return sim
end

# The SGS flux of a transported scalar other than θ (qv): the closure's
# `add_sgs_scalar!` when it diffuses every scalar with θ's κ_t (the default
# `sgs_tracer_diffusivity!` returns `f.κh`, so a Prandtl-number closure is
# bit for bit unchanged), else eddy diffusion with the scalar's own
# diffusivity, computed into `scratch` (AMD, #176). The same rule as
# `step_imex!`'s frozen qv coefficient. Halos of `ϕ` must be filled.
function _add_sgs_tracer!(tϕ, ϕ, closure, f, sim, scratch)
    κ = sgs_tracer_diffusivity!(scratch, closure, f, sim, ϕ)
    if κ === f.κh
        add_sgs_scalar!(tϕ, ϕ, closure, f, sim)
    else
        add_eddy_diffusion_scalar!(tϕ, ϕ, κ)
    end
    return tϕ
end

# One forward-Euler predictor + projection with the SGS closure `closure`
# (or none). No commit: the drivers (step!, step_rk3!) commit once per step.
# `t_bc` is the time the projected velocity belongs to, at which open
# boundaries are evaluated (the end of this Euler step; step_rk3! passes the
# Shu–Osher substage times).
function _euler_projection!(sim::BoussinesqSimulation{T}, Δt::T,
                            closure::C, body_force!, post_predictor!,
                            momentum_advection::F,
                            sgs_stress::Symbol = :laplacian,
                            t_bc = sim.t + Δt, stage = nothing) where {T, C, F}
    # 1. Halo fill so all stencils see valid neighbours.
    fill_state_halos!(sim)
    mark_open_boundary_stage!(sim)   # radiating outflow: stage-start layers (N-open-c)

    # Nothing between the base tendencies and the predictor: steps 2-7 and
    # 11 in one blocked sweep (SC-e, #405), the same bits.
    _blocked_core(sim, closure, body_force!, post_predictor!, momentum_advection) &&
        return _blocked_projection!(sim, Δt, t_bc, stage)

    # 2.-3. The tendencies from zero: momentum + scalar advection, then the
    #       molecular viscosity / diffusion (qv: the same molecular
    #       diffusivity as θ, #177). With the built-in centred momentum
    #       advection all of it is one fused pass (SC-d, #404), the bits of
    #       the separate passes below.
    if momentum_advection === momentum_advect!
        _base_tendencies!(sim)
    else
        zero_storage!(sim.tu); zero_storage!(sim.tv); zero_storage!(sim.tw)
        zero_storage!(sim.tθ)
        sim.has_moisture && zero_storage!(sim.tqv)
        momentum_advection(sim.tu, sim.tv, sim.tw, sim.u, sim.v, sim.w)
        advect_centered!(sim.tθ, sim.u, sim.v, sim.w, sim.θ)
        sim.has_moisture && advect_centered!(sim.tqv, sim.u, sim.v, sim.w, sim.qv)
        if sim.has_viscosity
            laplacian(sim.lap_u, sim.u); laplacian(sim.lap_v, sim.v); laplacian(sim.lap_w, sim.w)
            laplacian(sim.lap_θ, sim.θ)
            axpy!(sim.tu, sim.ν, sim.lap_u)
            axpy!(sim.tv, sim.ν, sim.lap_v)
            axpy!(sim.tw, sim.ν, sim.lap_w)
            axpy!(sim.tθ, sim.ν, sim.lap_θ)
            if sim.has_moisture
                laplacian(sim.lap_θ, sim.qv)
                axpy!(sim.tqv, sim.ν, sim.lap_θ)
            end
        end
    end

    # 4. SGS closure (the built-in Smagorinsky, or the `sgs` keyword): the
    #    coefficients from the current state, then the tendencies. Every
    #    transported scalar gets the closure's scalar flux: θ and, when
    #    moist, qv (P56-a, #177 — before it qv was advected but never
    #    diffused, a code bug).
    if closure !== nothing
        t_sgs = time_ns()
        f = compute_sgs!(closure, sim)
        add_sgs_momentum!(sim.tu, sim.tv, sim.tw, closure, f, sim; stress = sgs_stress)
        add_sgs_scalar!(sim.tθ, sim.θ, closure, f, sim)
        sim.has_moisture && _add_sgs_tracer!(sim.tqv, sim.qv, closure, f, sim, sim.lap_θ)
        sim.workspaces.rec.wall_sgs += _wall(t_sgs)
    end

    # 5. Buoyancy on w-momentum. Moist buoyancy uses virtual θ when qv is on.
    if sim.has_buoyancy
        if sim.has_moisture
            add_buoyancy!(sim.tw, sim.θ, sim.qv, sim.θ₀, sim.g_accel)
        else
            add_buoyancy!(sim.tw, sim.θ, sim.θ₀, sim.g_accel)
        end
    end

    # 6. Coriolis on horizontal momenta.
    if sim.has_coriolis
        add_coriolis!(sim.tu, sim.tv, sim.u, sim.v, sim.f_coriolis)
    end

    # 6a. Prescribed surface fluxes on θ (Phase 7).
    if sim.has_surface_flux
        add_surface_flux!(sim.tθ, sim.θ_flux_bottom, sim.θ_flux_top)
    end

    # 6b. Rayleigh sponge on velocity (Phase 7), toward `sim.sponge_target`
    #     (zero for a closed top; the boundary data's state for an open one).
    sim.has_sponge && add_velocity_sponge!(sim)

    # 6c. User body force / IBM penalty — last tendency contribution.
    body_force! === nothing || body_force!(sim, Δt)

    # 7. Predictor: u* = u + Δt F  (in-place).
    axpy!(sim.u, Δt, sim.tu)
    axpy!(sim.v, Δt, sim.tv)
    axpy!(sim.w, Δt, sim.tw)

    # 7b. Post-predictor hook — stiff corrections (implicit IBM, mask
    # enforcement, ...). Runs after u* is formed and before the Poisson
    # solve so the projection folds any velocity correction into a
    # divergence-free result.
    post_predictor! === nothing || post_predictor!(sim, Δt)

    # 7c. Open boundaries (N-open-b): boundary data at the end of the step,
    # mass-corrected so the projection's Neumann problem is compatible. The
    # halo fill below writes the corrected normal velocity into u*.
    update_open_boundaries!(sim, t_bc, Δt)

    # 8.–10. Chorin projection of the predicted velocity; the correction
    # also writes the physical pressure p = φ/Δt (step 12) and, inside
    # step_rk3!, the stage combination with the step-start state (SC-d).
    _project_velocity!(sim, _DivBy(Δt), stage)

    # 11. Advance θ (no projection needed for scalar), combined with the
    # step-start θ inside step_rk3!.
    _advance_scalar!(sim.θ, Δt, sim.tθ, stage, stage === nothing ? nothing : stage.ws.θ0)
    sim.has_moisture &&
        _advance_scalar!(sim.qv, Δt, sim.tqv, stage, stage === nothing ? nothing : stage.ws.q0)

    # 12. Advance the clock.
    sim.t += Δt

    return sim
end

# Whether a substage of `_euler_projection!` has nothing that adds to the
# tendencies or reads the state between the base tendencies (steps 2-3)
# and the scalar update (step 11): the built-in momentum advection, no SGS
# closure, buoyancy, Coriolis, surface flux, sponge, hooks or open
# boundaries, on the CPU. The DNS core (SC-e, #405).
_blocked_core(sim::BoussinesqSimulation, closure, body_force!, post_predictor!, momentum_advection) =
    momentum_advection === momentum_advect! && closure === nothing && body_force! === nothing &&
    post_predictor! === nothing &&
    !(sim.has_buoyancy || sim.has_coriolis || sim.has_surface_flux || sim.has_sponge) &&
    !has_open_boundaries(sim.bcs) && Backends.backend(sim.u.data) isa Backends.CPUBackend

# Steps 2-12 of `_euler_projection!` for a `_blocked_core`: the tendencies
# and the predictors of u, v, w and of θ (qv) in one sweep
# (`FiniteVolume.blocked_predictor!`), then the projection. The scalars'
# update of step 11 moves before the projection, which does not read them;
# the tendency fields are not used.
function _blocked_projection!(sim::BoussinesqSimulation{T}, Δt::T, t_bc, stage) where {T}
    moist = sim.has_moisture
    ii, jj, kk = interior(sim.mesh)
    ns = sweep_slabs(Backends.backend(sim.u.data), length(ii) * length(jj) * length(kk), length(kk))
    bufs = sweep_buffers(sim.workspaces.sweep, sim.u.data, moist ? 5 : 4, ns, length(kk))
    sim.workspaces.sweep = bufs
    if sim.has_viscosity
        _blocked_sweep!(bufs, sim, Δt, stage, sim.ν)
    else
        _blocked_sweep!(bufs, sim, Δt, stage, nothing)
    end
    update_open_boundaries!(sim, t_bc, Δt)
    _project_velocity!(sim, _DivBy(Δt), stage)
    if stage !== nothing                    # the halos of the RK3 combination (step 11)
        _shell_map!(stage.c, (sim.θ.data,), (stage.ws.θ0,), sim.mesh)
        moist && _shell_map!(stage.c, (sim.qv.data,), (stage.ws.q0,), sim.mesh)
    end
    sim.t += Δt
    return sim
end
# the sweep with concrete arguments: the scalars' update is the axpy of
# step 11 (`step!`), or with the RK3 combination and the step-start θ, qv
_blocked_sweep!(bufs, sim, Δt, ::Nothing, ν) =
    _blocked_sweep!(bufs, sim, Δt, ν, _Axpy(Δt), nothing, nothing)
_blocked_sweep!(bufs, sim, Δt, stage, ν) =
    _blocked_sweep!(bufs, sim, Δt, ν, _AxpyStage(_Axpy(Δt), stage.c), stage.ws.θ0, stage.ws.q0)
function _blocked_sweep!(bufs, sim, Δt, ν, ps, θ0, q0)
    s1 = (sim.θ, ps, θ0)
    if sim.has_moisture
        blocked_predictor!(bufs, sim.u, sim.v, sim.w, ν, _Axpy(Δt), s1, (sim.qv, ps, q0), sim.mesh)
    else
        blocked_predictor!(bufs, sim.u, sim.v, sim.w, ν, _Axpy(Δt), s1, nothing, sim.mesh)
    end
    return nothing
end

# The tendencies of steps 2-3 in one launch (`_fused_tendencies!`), and the
# zero of their halos (`zero_storage!` zeroed the whole storage).
function _base_tendencies!(sim::BoussinesqSimulation)
    if sim.has_moisture
        _shell_map!(_Zero(), (sim.tu.data, sim.tv.data, sim.tw.data, sim.tθ.data, sim.tqv.data),
                    nothing, sim.mesh)
        _base_tendencies!(sim, (sim.tqv, sim.qv))
    else
        _shell_map!(_Zero(), (sim.tu.data, sim.tv.data, sim.tw.data, sim.tθ.data), nothing, sim.mesh)
        _base_tendencies!(sim, nothing)
    end
    return sim
end
function _base_tendencies!(sim::BoussinesqSimulation, q)
    ν = sim.ν
    if sim.has_viscosity
        _fused_tendencies!(sim.tu, sim.tv, sim.tw, sim.u, sim.v, sim.w, ν, (sim.tθ, sim.θ), q, sim.mesh)
    else
        _fused_tendencies!(sim.tu, sim.tv, sim.tw, sim.u, sim.v, sim.w, nothing, (sim.tθ, sim.θ), q,
                           sim.mesh)
    end
    return sim
end

# ϕ ← ϕ + Δt·tϕ over the interior; inside step_rk3! (`stage`) combined with
# the step-start ϕ0, ϕ ← c(ϕ + Δt·tϕ, ϕ0) over the interior and ϕ ← c(ϕ, ϕ0)
# on the halo: the axpy then `combine_storage!`, in one pass (SC-d, #404).
_advance_scalar!(ϕ::Field, Δt, tϕ::Field, ::Nothing, ::Nothing) = axpy!(ϕ, Δt, tϕ)
function _advance_scalar!(ϕ::Field, Δt, tϕ::Field, stage, ϕ0::AbstractArray)
    size(ϕ0) == size(ϕ.data) == size(tϕ.data) || throw(DimensionMismatch("_advance_scalar!: sizes differ"))
    range_map!(Backends.backend(ϕ.data), _AxpyStage(_Axpy(Δt), stage.c), ϕ.data, interior(ϕ.mesh),
               ϕ.data, tϕ.data, ϕ0)
    _shell_map!(stage.c, (ϕ.data,), (ϕ0,), ϕ.mesh)
    return ϕ
end
struct _AxpyStage{A, C}
    ax::A
    c::C
end
@inline (f::_AxpyStage)(y, x, snap) = f.c(f.ax(y, x), snap)

# Steps 8–10 of the fractional step: the divergence of the velocity, the
# Poisson solve ∇²φ = ∇·u, the correction u ← u − ∇φ. `sim.φ` holds the
# potential afterwards; `sim.p` is not touched. Shared by step!/step_rk3!,
# the per-stage projection of the IMEX steppers and the re-projection of a
# two-way nest parent after the feedback (N-nest-e, #263).
#
# Fused (SC-d, #404): a spectral solver's source buffer receives the
# divergence directly (no `div_buf`, no copy-in); the correction is one
# pass over u, v, w that reads φ once (no `dφdx`, `dφdy`, `dφdz`; only an
# open face's slab is computed, for its check). `pf` (the step cores'
# `_DivBy(Δt)`) also writes p = φ/Δt in that pass, and `stage` (inside
# step_rk3!: `(c = _Lin2r(a, b), ws = RK3Workspace)`) writes the stage
# combination c(u − ∇φ, u0) on the interior and c(u, u0) on the halo,
# which is what `combine_storage!` made of the corrected field. The same
# bits as the separate passes.
function _project_velocity!(sim::BoussinesqSimulation{T}, pf = nothing, stage = nothing) where {T}
    # 8. Divergence of predicted velocity, 9. solve ∇²φ = ∇·u*.
    fill_halos!((sim.u, sim.v, sim.w), sim.bcs)
    buf = source_buffer(sim.solver)
    if buf === nothing
        divergence(sim.div_buf, sim.u, sim.v, sim.w)
        t_solve = time_ns()
        solve!(sim.φ, sim.div_buf, sim.solver)
    else
        _divergence_into!(buf, sim.u, sim.v, sim.w, sim.mesh)
        t_solve = time_ns()
        solve_source_buffer!(sim.φ, sim.solver)
    end
    _note_solve!(sim.workspaces.rec, sim.solver, t_solve)   # iterations, residual, wall (#336)

    # 10. Correct velocity.
    fill_halos!(sim.φ, sim.bcs)
    b = sim.bcs
    if has_open_boundaries(b)                                # ∂φ/∂n = 0 on open faces
        _face_gradient_slabs!(sim.dφdx, sim.dφdy, sim.dφdz, sim.φ, sim.mesh, b.west isa OpenBoundary,
                              b.south isa OpenBoundary, b.bottom isa OpenBoundary)
        assert_open_face_correction(b, sim.dφdx, sim.dφdy, sim.dφdz)
    end
    p = pf === nothing ? nothing : sim.p.data
    if stage === nothing
        _correct_velocity!(sim.u, sim.v, sim.w, sim.φ, sim.mesh; p, pf)
    else
        ws = stage.ws
        _correct_velocity!(sim.u, sim.v, sim.w, sim.φ, sim.mesh; p, pf, stage = stage.c,
                           u0 = ws.u0, v0 = ws.v0, w0 = ws.w0)
        _shell_map!(stage.c, (sim.u.data, sim.v.data, sim.w.data), (ws.u0, ws.v0, ws.w0), sim.mesh)
    end
    return sim
end

"""
    run!(sim, dt, nsteps; callback = nothing)

Advance `sim` by `nsteps` steps of size `dt`. If `callback !== nothing`
it is invoked as `callback(sim, n)` at the end of each step (useful for
diagnostics, plotting, checkpointing).
"""
function run!(sim::BoussinesqSimulation, dt::Real, nsteps::Integer;
              callback = nothing)
    for n in 1:nsteps
        step!(sim, dt)
        callback === nothing || callback(sim, n)
    end
    return sim
end

"""
    RK3Workspace(sim)

The step-start state `uⁿ` (`u0, v0, w0, θ0, q0`: storage-sized copies of
`sim.u, v, w, θ, qv`) that [`step_rk3!`](@ref) combines with its later
substages. Allocated once per simulation by [`rk3_workspace`](@ref), so a
step allocates nothing for it (before #295 every step copied 9 fields).
"""
struct RK3Workspace{A<:AbstractArray}
    u0::A; v0::A; w0::A; θ0::A; q0::A
end

function RK3Workspace(sim::BoussinesqSimulation)
    al(f) = storage_like(f.data)
    return RK3Workspace(al(sim.u), al(sim.v), al(sim.w), al(sim.θ), al(sim.θ))
end

"""
    rk3_workspace(sim) -> RK3Workspace

The [`RK3Workspace`](@ref) of `sim`: allocated on first use and kept in
the simulation's [`StepWorkspaces`](@ref) (KA-j, #306; before, a global
weak cache).
"""
function rk3_workspace(sim::BoussinesqSimulation)
    w = sim.workspaces
    w.rk3 === nothing && (w.rk3 = RK3Workspace(sim))
    return w.rk3::RK3Workspace
end

# -------------------------------------------------------------------------
# Projected RK3 — Phase 10.
# -------------------------------------------------------------------------
# One `step_rk3!(sim, dt)` performs three Chorin-projected Shu-Osher SSP-
# RK3 substages. Third-order accurate in time for divergence-free flows.
# Cost: 3× the pressure solves per step.

"""
    step_rk3!(sim::BoussinesqSimulation, dt; sgs = nothing, eddy_viscosity! = nothing,
              body_force! = nothing, post_predictor! = nothing,
              sgs_stress = :laplacian, workspace = rk3_workspace(sim))

Advance one Shu-Osher SSP-RK3 step with a pressure projection at each of
the three substages. Third-order-accurate in time for divergence-free
motion. Costs three Poisson solves per step (vs. one for `step!`), so
use only when the extra accuracy is needed.

`sgs` is the SGS closure, as for [`step!`](@ref): its coefficients are
recomputed at each substage from the substage state, and its memory is
committed ([`advance_sgs!`](@ref Aeolus.Operators.advance_sgs!)) once, with
`dt`, after the third substage — not three times. With `moisture = true`
`qv` is combined over the substages like `θ` (before #177 it was not, so
`qv` advanced by three Euler steps per call).

`body_force!(sim, Δt)` and `post_predictor!(sim, Δt)` are `step!`'s hooks
(added for the wall-bounded turbulence cases, P54-d #174): each SSP-RK3
substage is one forward-Euler predictor + projection of length `dt`, and
the hooks run inside every substage exactly where `step!` runs them (a
tendency before the predictor; an implicit correction such as
`WallModels.apply_wall_stress!` after it, before the projection). The
step's net effect of a hook is the SSP-RK3 combination of its three
substage effects, with weights 1/6, 1/6, 2/3.

In P56-e (#192) the same hooks carry the advection of a uniform
background gradient, ``-w\\,∂_zΘ``, in a triply periodic box, and the
immersed-boundary direct forcing of the surface-mounted cube.

**Open boundaries** (N-open-b, #255): each substage is a forward-Euler
step projected with the boundary data of its own end time — `t+Δt`,
`t+2Δt` and `t+3Δt/2` for the three Shu–Osher substages — and the
boundary buffers are combined with the fields' weights, so every stage's
halos and boundary faces are exactly those of the combined state.

The step-start state lives in `workspace` ([`RK3Workspace`](@ref), cached
per simulation), so a step allocates nothing but the cache lookup (#295).
"""
function step_rk3!(sim::BoussinesqSimulation{T}, dt::Real; kwargs...) where {T}
    rec = sim.workspaces.rec
    outer = _step_begin!(rec, sim)
    try
        _step_rk3!(sim, T(dt); kwargs...)
    catch
        _step_abort!(rec); rethrow()
    end
    return _step_end!(rec, sim, T(dt), outer)
end

function _step_rk3!(sim::BoussinesqSimulation{T}, dt::Real;
                    sgs = nothing, eddy_viscosity! = nothing,
                    body_force!::Union{Nothing, Function} = nothing,
                    post_predictor!::Union{Nothing, Function} = nothing,
                    sgs_stress::Symbol = :laplacian,
                    workspace::RK3Workspace = rk3_workspace(sim)) where {T}
    Δt = T(dt)
    closure = _active_sgs(sim, sgs, eddy_viscosity!)
    _check_sgs_stress(sgs_stress)
    prepare_open_boundaries!(sim)
    t0 = sim.t
    bc0 = open_boundary_snapshot(sim)      # nothing for closed boundaries
    if bc0 !== nothing                     # uⁿ carries its boundary state into the combinations
        fill_state_halos!(sim, sim.θ_bcs)
    end

    # Save current state (into the workspace: no allocation per step, #295;
    # copy and combinations are kernels over the whole storage, KA-j #306)
    ws = workspace
    u0 = copy_storage!(ws.u0, sim.u.data); v0 = copy_storage!(ws.v0, sim.v.data)
    w0 = copy_storage!(ws.w0, sim.w.data); θ0 = copy_storage!(ws.θ0, sim.θ.data)
    moist = sim.has_moisture
    q0 = moist ? copy_storage!(ws.q0, sim.qv.data) : θ0      # qv is combined like θ (#177)

    _rk3_substep!(sim, Δt, closure, body_force!, post_predictor!, sgs_stress,
                  t0 + Δt)                                                        # u1 = uⁿ + Δt L(uⁿ), then project
    # u1 stays in sim (the next substage advances it in place)

    # u1 → u1 + Δt L(u1), projected and combined to u2 = 3/4 uⁿ + 1/4 (…):
    # the combination is written by the projection's correction and the
    # scalar update (SC-d, #404), the bits of `combine_storage!` after them
    c2 = _Lin2r(0.75, 0.25)
    _rk3_substep!(sim, Δt, closure, body_force!, post_predictor!, sgs_stress,
                  t0 + 2Δt, (c = c2, ws = _RK3Stage(u0, v0, w0, θ0, q0)))
    combine_open_boundaries!(sim, 0.75, bc0, 0.25)

    # u2 → u2 + Δt L(u2), projected and combined to uⁿ⁺¹ = 1/3 uⁿ + 2/3 (…)
    inv3 = one(T)/3; two3 = 2*inv3
    c3 = _Lin2r(inv3, two3)
    _rk3_substep!(sim, Δt, closure, body_force!, post_predictor!, sgs_stress,
                  t0 + 3Δt / 2, (c = c3, ws = _RK3Stage(u0, v0, w0, θ0, q0)))
    combine_open_boundaries!(sim, inv3, bc0, two3)

    # sim.t was advanced 3× by _rk3_substep!; correct to one nominal step
    sim.t = sim.t - 2Δt
    closure === nothing || advance_sgs!(closure, sim, Δt)   # once per step
    return sim
end

# One projected Euler-like substep — shared with step! (the pressure
# projection), without step!'s commit of the closure memory.
_rk3_substep!(sim::BoussinesqSimulation, Δt, closure, body_force! = nothing,
              post_predictor! = nothing, sgs_stress::Symbol = :laplacian,
              t_bc = sim.t + Δt, stage = nothing) =
    _euler_projection!(sim, Δt, closure, body_force!, post_predictor!, momentum_advect!,
                       sgs_stress, t_bc, stage)

# The step-start arrays a combined substage reads (`q0` is `θ0` when dry).
struct _RK3Stage{A<:AbstractArray}
    u0::A; v0::A; w0::A; θ0::A; q0::A
end

# -------------------------------------------------------------------------
# Diagnostics
# -------------------------------------------------------------------------

"""
    kinetic_energy(sim) -> Float64

Volume-integrated kinetic energy `½ ∫(u² + v² + w²) dV` over the
interior cells. `u`, `v`, `w` are used *as stored* on their respective
faces — no interpolation to a common location. This is a slight bias
but the standard diagnostic for staggered LES.
"""
function kinetic_energy(sim::BoussinesqSimulation)
    m = sim.mesh
    m isa RegularGrid || return _kinetic_energy_levels(sim, m)
    dV = m.dx * m.dy * m.dz
    I3 = interior(m)
    # canonical plane sums, then ascending k: thread-invariant (KA-b, #296)
    p = plane_partials!(zeros(length(I3[3])), Backends.backend(sim.u.data), _ke_point, I3,
                        sim.u.data, sim.v.data, sim.w.data)
    return 0.5 * ordered_sum(p) * dV
end
@inline _ke_point(u, v, w) = u^2 + v^2 + w^2

# The same sum with per-level volumes: u, v (at cell-centre heights) weigh
# with the cell thickness, w (on z-faces) with the centre spacing, i.e. the
# heights of the u- and w-control volumes of the staggered grid.
function _kinetic_energy_levels(sim::BoussinesqSimulation, m)
    I3 = interior(m)
    p = plane_reduce!(zeros(length(I3[3])), Backends.backend(sim.u.data), _ke_levels_at, I3,
                      sim.u.data, sim.v.data, sim.w.data, m)
    return 0.5 * ordered_sum(p) * m.dx * m.dy
end
@inline _ke_levels_at(i, j, k, ud, vd, wd, m) =
    @inbounds (ud[i, j, k]^2 + vd[i, j, k]^2) * _dzc(m, k) + wd[i, j, k]^2 * _dzf(m, k)

# |a| at a point: the body of the max-norm reductions
@inline _abs_at(i, j, k, a) = @inbounds abs(a[i, j, k])

"""
    fill_state_halos!(sim, qv_bcs = passive_scalar_bcs(sim.θ_bcs)) -> sim

Fill the halos of the prognostic state of a [`BoussinesqSimulation`](@ref)
— `u`, `v`, `w` with `sim.bcs`, `θ` with `sim.θ_bcs` and, with moisture,
`qv` with `qv_bcs` — as one set ([`fill_halos!`](@ref)`(fields::Tuple, …)`):
two threaded launches for the whole state (SC-b, #402), the same bits as
filling the fields one after the other.
"""
function fill_state_halos!(sim, qv_bcs = passive_scalar_bcs(sim.θ_bcs))
    b = sim.bcs
    if sim.has_moisture
        fill_halos!((sim.u, sim.v, sim.w, sim.θ, sim.qv), (b, b, b, sim.θ_bcs, qv_bcs))
    else
        fill_halos!((sim.u, sim.v, sim.w, sim.θ), (b, b, b, sim.θ_bcs))
    end
    return sim
end

"""
    max_divergence(sim) -> Float64

Maximum `|∇·u|` over the interior cells. Should sit at floating-point
epsilon after the pressure projection (Chorin's method is exact on the
discrete Laplacian).
"""
function max_divergence(sim::BoussinesqSimulation)
    fill_halos!((sim.u, sim.v, sim.w), sim.bcs)
    divergence(sim.div_buf, sim.u, sim.v, sim.w)
    # max is exact: the same bits as `maximum(abs, interior_view(div_buf))`
    return range_max(Backends.backend(sim.div_buf.data), _abs_at, interior(sim.mesh), sim.div_buf.data)
end

"""
    max_νt(sim) -> Float64

Maximum SGS eddy viscosity `νₜ` over the interior. Zero when
`C_s = 0`; otherwise responds to local strain and gives a sense of
where the SGS model is most active.
"""
max_νt(sim::BoussinesqSimulation) = interior_max(sim.νt)   # exact: the bits of maximum

"""
    horizontal_mean(field) -> Vector

Horizontal mean of a cell-centered field over the interior, one entry
per interior `z`-level.
"""
function horizontal_mean(field::Field{CellCenter})
    # canonical plane sums / (Nx Ny): thread-invariant (KA-b, #296)
    return plane_mean!(zeros(eltype(field.data), size(interior_view(field), 3)), field)
end

"""
    horizontal_variance(field) -> Vector

Horizontal variance of a field over the interior, `⟨(f - ⟨f⟩)²⟩`, one
entry per interior `z`-level (the face levels for a `FaceZ` field).
"""
function horizontal_variance(field::Field)
    # the plane mean of any location (horizontal_mean is cell-centred only),
    # then the canonical plane sums of (f - m_k)²: thread-invariant (KA-b, #296)
    m = plane_mean!(zeros(eltype(field.data), size(interior_view(field), 3)), field)
    I3 = interior(field.mesh)
    n = length(I3[1]) * length(I3[2])
    s = plane_reduce!(similar(m), Backends.backend(field.data), _sq_dev_at, I3,
                      field.data, m, first(I3[3]) - 1)
    return s ./ n
end

"""
    resolved_flux_wθ(sim) -> Vector

Horizontally-averaged resolved kinematic scalar flux ``⟨w'\\,θ'⟩`` at
cell centers (using `w` interpolated in z). One entry per interior
`z`-level.
"""
function resolved_flux_wθ(sim::BoussinesqSimulation)
    m_θ = horizontal_mean(sim.θ)
    I3 = interior(sim.mesh)
    s = plane_reduce!(similar(m_θ), Backends.backend(sim.θ.data), _wθ_flux_at, I3,
                      sim.w.data, sim.θ.data, m_θ, first(I3[3]) - 1)
    return s ./ (length(I3[1]) * length(I3[2]))
end
# w (centred in z) × θ′ at a point of level k
@inline _wθ_flux_at(i, j, k, wd, θd, m, ok) =
    @inbounds 0.5 * (wd[i, j, k] + wd[i, j, k + 1]) * (θd[i, j, k] - m[k - ok])

"""
    mean_θ_profile(sim) -> Vector

Horizontal-mean potential-temperature profile `⟨θ⟩(k)` over the
interior, one entry per interior `z`-level. Useful for diagnosing
stratification.
"""
mean_θ_profile(sim::BoussinesqSimulation) = horizontal_mean(sim.θ)   # canonical (KA-b, #296)

# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

"""
    _default_solver(mesh::RegularGrid, bcs) -> AbstractPressureSolver

Pick the direct pressure solver from all six velocity faces of `bcs`,
each mapped to `:periodic` or `:neumann` by
[`axis_poisson_bc`](@ref Aeolus.BoundaryConditions.axis_poisson_bc)
(see [`poisson_axis_bcs`](@ref Aeolus.PressureSolvers.poisson_axis_bcs)):

* triply periodic → [`FFTPoissonSolver`](@ref);
* anything else → [`CosineFFTPoissonSolver`](@ref) with those axis kinds
  — the channel `(:periodic, :periodic, :neumann)` (bit-for-bit the solver
  chosen before #251), and the all-Neumann box of a nested or open
  child.

A face type with no Poisson kind, or an axis whose two faces disagree
(periodic against Neumann), throws an `ArgumentError`.
"""
function _default_solver(mesh::RegularGrid, bcs::FieldBoundaryConditions)
    ax = poisson_axis_bcs(bcs)
    if ax.bc_x === :periodic && ax.bc_y === :periodic && ax.bc_z === :periodic
        return FFTPoissonSolver(mesh)
    else
        return CosineFFTPoissonSolver(mesh; ax...)
    end
end

# --- vertical metrics the core needs, generic over the mesh --------------
# `_dzc(m, k)`: thickness of cell k (storage index), the height of a
# u/v/θ control volume; `_dzf(m, k)`: distance between the centres k-1 and
# k, the height of the w control volume on face k; `_dz_min(m)`: the
# thinnest interior cell (the CFL limits). Uniform here; the StretchedGrid
# methods (in `StretchedGrids`) read `cell_thickness` / `center_spacing`.
_dzc(m::RegularGrid, k) = m.dz
_dzf(m::RegularGrid, k) = m.dz
_dz_min(m::RegularGrid) = m.dz

# A mesh the Boussinesq core has a complete operator set for. RegularGrid
# here; StretchedGrids adds its grid. Anything else (a terrain-following
# grid, a spectral-element mesh) is refused at construction instead of
# failing with a MethodError inside the first step.
_check_boussinesq_mesh(::RegularGrid) = nothing
_check_boussinesq_mesh(m::AbstractMesh) =
    throw(ArgumentError("BoussinesqSimulation: no operator set for a $(typeof(m)); " *
                        "use a RegularGrid or a StretchedGrid"))

# -------------------------------------------------------------------------
# Field algebra of the step cores, as kernels (KA-j, #306)
# -------------------------------------------------------------------------
# The step drivers (step!, step_rk3!, the IMEX and adaptive drivers of
# TimeIntegration, the sound-proof cores of Anelastic) update their fields
# only through these: pointwise launches (`interior_map!`, `range_map!`),
# hence the same bits as the broadcasts they replace, serially and threaded.

"""
    axpy!(y::Field, a, x::Field) -> y

`y = y + a·x` over the interior ([`interior_map!`](@ref
Aeolus.Fields.interior_map!)); halos untouched. The one axpy of the step
cores (KA-j, #306: it replaces three copies of `_axpy_interior!`).
"""
axpy!(y::Field, a::Number, x::Field) = interior_map!(_Axpy(a), y, y, x)
struct _Axpy{T}
    a::T
end
@inline (f::_Axpy)(y, x) = y + f.a * x

# out = a·A + b·B, the same expression as the broadcast `a .* A .+ b .* B`
struct _Lin2{T, S}
    a::T
    b::S
end
@inline (f::_Lin2)(A, B) = f.a * A + f.b * B
# the RK3 combination `out = a·snap + b·out`, called as f(out, snap) so that
# the in-place array is the first argument of `range_map!`
struct _Lin2r{T, S}
    a::T
    b::S
end
@inline (f::_Lin2r)(out, snap) = f.a * snap + f.b * out

"""
    pressure_from_potential!(p::Field, φ::Field, Δt) -> p

`p = φ/Δt` over the interior: the pressure of a projection with potential
`φ` (a pointwise kernel).
"""
pressure_from_potential!(p::Field, φ::Field, Δt) = interior_map!(_DivBy(Δt), p, φ)
struct _DivBy{T}
    d::T
end
@inline (f::_DivBy)(x) = x / f.d


"""
    copy_storage!(dst, src) -> dst

`dst .= src` over the whole storage (halos included); `dst` and `src`
must have the same size. It is `copyto!` (a `memcpy`, which a GPU array
also implements). A copy kernel measured no reliable gain (KA-j, #306;
`benchmarks/bench_ka_step.jl kernels`): 1.01-1.07× the `memcpy` serially,
and 1.09-1.30× at 64×64×32 and 0.82-1.58× at 128×128×64 on 4-24 threads.
"""
function copy_storage!(dst::AbstractArray{T,3}, src::AbstractArray{T,3}) where {T}
    size(dst) == size(src) || throw(DimensionMismatch("copy_storage!: $(size(dst)) ≠ $(size(src))"))
    copyto!(dst, src)
    return dst
end


"""
    zero_storage!(f::Field) -> f

Zero every value of `f`, halos included: the reset of the tendencies at
the start of a step. It is `fill!` (a `memset`, which a GPU array also
implements). A pointwise kernel measured slower at every thread count
that matters: 15.8 µs against 11.2 µs serially at 64×64×32, and 28 µs at
4 threads (KA-j, #306; `benchmarks/bench_ka_step.jl kernels`).
"""
function zero_storage!(f::Field)
    fill!(f.data, zero(eltype(f.data)))
    return f
end

# ---- the halo shell (SC-d, #404) ------------------------------------------
# The fused kernels write the interior; what the separate passes did to the
# rest of the storage (zero the tendencies' halos, combine the halos of the
# RK3 stages) is done by one launch over the halo shell: the (j, k) rows of
# the storage, each row its halo points only (all of it outside the
# interior's j, k), for a tuple of arrays.

struct _Zero end
@inline (::_Zero)(x, s) = zero(x)

@inline _shell_point!(f, ::Tuple{}, ins, i, j, k) = nothing
@inline function _shell_point!(f, outs::Tuple, ins, i, j, k)
    o = first(outs)
    @inbounds o[i, j, k] = f(o[i, j, k], _shell_in(ins, i, j, k))
    _shell_point!(f, Base.tail(outs), _shell_tail(ins), i, j, k)
    return nothing
end
@inline _shell_in(::Nothing, i, j, k) = nothing
@inline _shell_in(ins::Tuple, i, j, k) = @inbounds first(ins)[i, j, k]
@inline _shell_tail(::Nothing) = nothing
@inline _shell_tail(ins::Tuple) = Base.tail(ins)

@inline function _b_shell!(j, k, f, outs, ins, ir, jr, kr, ni)
    if j in jr && k in kr
        for i in 1:first(ir) - 1
            _shell_point!(f, outs, ins, i, j, k)
        end
        for i in last(ir) + 1:ni
            _shell_point!(f, outs, ins, i, j, k)
        end
    else
        for i in 1:ni
            _shell_point!(f, outs, ins, i, j, k)
        end
    end
    return nothing
end

# out = f(out, in) at every storage point outside `interior(mesh)`, for
# each array of `outs` (and the matching array of `ins`, or `nothing`):
# one launch, the same bits as the pointwise passes over the storage.
function _shell_map!(f::F, outs::Tuple, ins, mesh) where {F}
    o1 = first(outs)
    all(o -> size(o) == size(o1), outs) || throw(DimensionMismatch("_shell_map!: sizes differ"))
    ins === nothing || all(a -> size(a) == size(o1), ins) ||
        throw(DimensionMismatch("_shell_map!: sizes differ"))
    ni, nj, nk = size(o1)
    ir, jr, kr = interior(mesh)
    launch_weighted!(Backends.backend(o1), ni, _b_shell!, (1:nj, 1:nk), f, outs, ins, ir, jr, kr, ni)
    return nothing
end

# The whole storage of an array (halos included) as launch ranges.
_storage(a::AbstractArray{T,3}) where {T} = map(r -> UnitRange{Int}(r), axes(a))

# A zero-filled array shaped like `a`, on its backend.
storage_like(a::AbstractArray{T}) where {T} = Backends.allocate(Backends.backend(a), T, size(a))

end # module
