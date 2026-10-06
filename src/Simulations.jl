module Simulations

using ..Grids
using ..Grids: AbstractMesh, RegularGrid, CellCenter, FaceX, FaceY, FaceZ, interior
using ..Fields
using ..Fields: Field, interior_view
using ..BoundaryConditions
using ..BoundaryConditions: FieldBoundaryConditions, fill_halos!, Periodic, ImpermeableWall,
                            update_open_boundaries!, prepare_open_boundaries!,
                            assert_open_face_correction, open_boundary_snapshot,
                            combine_open_boundaries!
using ..Operators
using ..Operators: divergence, momentum_advect!, laplacian, ∂x, ∂y, ∂z,
                   advect_centered!,
                   add_buoyancy!, add_coriolis!,
                   compute_smagorinsky_νt!,
                   add_eddy_viscosity_u!, add_eddy_viscosity_v!, add_eddy_viscosity_w!,
                   add_eddy_diffusion_scalar!,
                   add_surface_flux!, add_sponge!,
                   AbstractSGSClosure, compute_sgs!, add_sgs_momentum!, add_sgs_scalar!,
                   advance_sgs!, _active_sgs, _check_sgs_stress, sgs_tracer_diffusivity!
using ..PressureSolvers
using ..PressureSolvers: AbstractPressureSolver, FFTPoissonSolver, CosineFFTPoissonSolver, solve!,
                        poisson_axis_bcs
import ..TimeSteppers: step!    # extend the generic `step!` from TimeSteppers

export BoussinesqSimulation, run!, step_rk3!, kinetic_energy, max_divergence,
       max_νt, mean_θ_profile,
       horizontal_mean, horizontal_variance, resolved_flux_wθ

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

    # tendencies (persistent buffers)
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
    κt::Field{CellCenter,      T, Array{T,3}, M}   # SGS scalar diffusivity (Era VI, #167)
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

    return BoussinesqSimulation(mesh, bcs, θ_bcs, solver,
                                Tp(ν), Tp(θ₀), Tp(g_accel), Tp(f_coriolis),
                                Tp(C_s), Tp(Pr_t),
                                Tp(θ_flux_bottom), Tp(θ_flux_top),
                                Tp(sponge_α_max), Tp(sponge_z_start),
                                has_visc, has_buoy, has_cor, has_sgs,
                                has_sflux, has_sponge, has_moist,
                                zero(Tp),
                                u, v, w, p, θ, qv,
                                tu, tv, tw, tθ, tqv,
                                div_buf, φ, dφdx, dφdy, dφdz,
                                lu, lv, lw, lθ, νt, κt)
end

# -------------------------------------------------------------------------
# One Euler-projection step.
# -------------------------------------------------------------------------

"""
    step!(sim::BoussinesqSimulation, dt; body_force! = nothing, post_predictor! = nothing,
          momentum_advection = momentum_advect!, sgs = nothing,
          sgs_stress = :laplacian)

Advance `sim` by `dt`. Two optional callbacks are supported:

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

  * `sgs_stress = :laplacian` — the form of the SGS momentum divergence,
    passed to [`add_sgs_momentum!`](@ref Aeolus.Operators.add_sgs_momentum!):
    `:laplacian` is `∂ⱼ(νₜ∂ⱼuᵢ)` (the default; the pre-Era-VI runs bit for
    bit), `:symmetric` the stress form `∂ⱼ(2νᵢⱼSᵢⱼ)` with the closure's
    horizontal and vertical viscosities
    ([`add_sgs_stress_divergence!`](@ref Aeolus.Operators.add_sgs_stress_divergence!),
    P55-b). `step_rk3!`, `step_imex!`, `step_adaptive!` and the
    sound-proof cores take the same keyword.
"""
function step!(sim::BoussinesqSimulation{T}, dt::Real;
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
                            t_bc = sim.t + Δt) where {T, C, F}
    # 1. Halo fill so all stencils see valid neighbours.
    fill_halos!(sim.u, sim.bcs)
    fill_halos!(sim.v, sim.bcs)
    fill_halos!(sim.w, sim.bcs)
    fill_halos!(sim.θ, sim.θ_bcs)
    sim.has_moisture && fill_halos!(sim.qv, homogeneous_bcs(sim.θ_bcs))

    # 2. Zero tendencies and add momentum + scalar advection.
    fill!(sim.tu.data, 0); fill!(sim.tv.data, 0); fill!(sim.tw.data, 0)
    fill!(sim.tθ.data, 0)
    sim.has_moisture && fill!(sim.tqv.data, 0)
    momentum_advection(sim.tu, sim.tv, sim.tw, sim.u, sim.v, sim.w)
    advect_centered!(sim.tθ, sim.u, sim.v, sim.w, sim.θ)
    sim.has_moisture && advect_centered!(sim.tqv, sim.u, sim.v, sim.w, sim.qv)

    # 3. Molecular viscosity / diffusion.
    if sim.has_viscosity
        laplacian(sim.lap_u, sim.u); laplacian(sim.lap_v, sim.v); laplacian(sim.lap_w, sim.w)
        laplacian(sim.lap_θ, sim.θ)
        _axpy_interior!(sim.tu, sim.ν, sim.lap_u, sim.mesh)
        _axpy_interior!(sim.tv, sim.ν, sim.lap_v, sim.mesh)
        _axpy_interior!(sim.tw, sim.ν, sim.lap_w, sim.mesh)
        _axpy_interior!(sim.tθ, sim.ν, sim.lap_θ, sim.mesh)
        if sim.has_moisture                 # qv: the same molecular diffusivity as θ (#177)
            laplacian(sim.lap_θ, sim.qv)
            _axpy_interior!(sim.tqv, sim.ν, sim.lap_θ, sim.mesh)
        end
    end

    # 4. SGS closure (the built-in Smagorinsky, or the `sgs` keyword): the
    #    coefficients from the current state, then the tendencies. Every
    #    transported scalar gets the closure's scalar flux: θ and, when
    #    moist, qv (P56-a, #177 — before it qv was advected but never
    #    diffused, a code bug).
    if closure !== nothing
        f = compute_sgs!(closure, sim)
        add_sgs_momentum!(sim.tu, sim.tv, sim.tw, closure, f, sim; stress = sgs_stress)
        add_sgs_scalar!(sim.tθ, sim.θ, closure, f, sim)
        sim.has_moisture && _add_sgs_tracer!(sim.tqv, sim.qv, closure, f, sim, sim.lap_θ)
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

    # 6b. Rayleigh sponge on velocity (Phase 7).
    if sim.has_sponge
        z_top = sim.mesh.z0 + sim.mesh.Lz
        add_sponge!(sim.tu, sim.u, sim.sponge_α_max, sim.sponge_z_start, z_top)
        add_sponge!(sim.tv, sim.v, sim.sponge_α_max, sim.sponge_z_start, z_top)
        add_sponge!(sim.tw, sim.w, sim.sponge_α_max, sim.sponge_z_start, z_top)
    end

    # 6c. User body force / IBM penalty — last tendency contribution.
    body_force! === nothing || body_force!(sim, Δt)

    # 7. Predictor: u* = u + Δt F  (in-place).
    _axpy_interior!(sim.u, Δt, sim.tu, sim.mesh)
    _axpy_interior!(sim.v, Δt, sim.tv, sim.mesh)
    _axpy_interior!(sim.w, Δt, sim.tw, sim.mesh)

    # 7b. Post-predictor hook — stiff corrections (implicit IBM, mask
    # enforcement, ...). Runs after u* is formed and before the Poisson
    # solve so the projection folds any velocity correction into a
    # divergence-free result.
    post_predictor! === nothing || post_predictor!(sim, Δt)

    # 7c. Open boundaries (N-open-b): boundary data at the end of the step,
    # mass-corrected so the projection's Neumann problem is compatible. The
    # halo fill below writes the corrected normal velocity into u*.
    update_open_boundaries!(sim, t_bc, Δt)

    # 8. Divergence of predicted velocity.
    fill_halos!(sim.u, sim.bcs)
    fill_halos!(sim.v, sim.bcs)
    fill_halos!(sim.w, sim.bcs)
    divergence(sim.div_buf, sim.u, sim.v, sim.w)

    # 9. Solve ∇²φ = ∇·u*.
    solve!(sim.φ, sim.div_buf, sim.solver)

    # 10. Correct velocity.
    fill_halos!(sim.φ, sim.bcs)
    ∂x(sim.dφdx, sim.φ); ∂y(sim.dφdy, sim.φ); ∂z(sim.dφdz, sim.φ)
    assert_open_face_correction(sim.bcs, sim.dφdx, sim.dφdy, sim.dφdz)   # ∂φ/∂n = 0 on open faces
    _axpy_interior!(sim.u, -one(T), sim.dφdx, sim.mesh)
    _axpy_interior!(sim.v, -one(T), sim.dφdy, sim.mesh)
    _axpy_interior!(sim.w, -one(T), sim.dφdz, sim.mesh)

    # 11. Advance θ (no projection needed for scalar).
    _axpy_interior!(sim.θ, Δt, sim.tθ, sim.mesh)
    sim.has_moisture && _axpy_interior!(sim.qv, Δt, sim.tqv, sim.mesh)

    # 12. Recover physical pressure and advance clock.
    ii, jj, kk = interior(sim.mesh)
    @inbounds @views sim.p.data[ii, jj, kk] .= sim.φ.data[ii, jj, kk] ./ Δt
    sim.t += Δt

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

# -------------------------------------------------------------------------
# Projected RK3 — Phase 10.
# -------------------------------------------------------------------------
# One `step_rk3!(sim, dt)` performs three Chorin-projected Shu-Osher SSP-
# RK3 substages. Third-order accurate in time for divergence-free flows.
# Cost: 3× the pressure solves per step.

"""
    step_rk3!(sim::BoussinesqSimulation, dt; sgs = nothing, eddy_viscosity! = nothing,
              body_force! = nothing, post_predictor! = nothing,
              sgs_stress = :laplacian)

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
(added for the Era VI wall-bounded cases, P54-d #174): each SSP-RK3
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
"""
function step_rk3!(sim::BoussinesqSimulation{T}, dt::Real;
                   sgs = nothing, eddy_viscosity! = nothing,
                   body_force!::Union{Nothing, Function} = nothing,
                   post_predictor!::Union{Nothing, Function} = nothing,
                   sgs_stress::Symbol = :laplacian) where {T}
    Δt = T(dt)
    closure = _active_sgs(sim, sgs, eddy_viscosity!)
    _check_sgs_stress(sgs_stress)
    prepare_open_boundaries!(sim)
    t0 = sim.t
    bc0 = open_boundary_snapshot(sim)      # nothing for closed boundaries
    if bc0 !== nothing                     # uⁿ carries its boundary state into the combinations
        fill_halos!(sim.u, sim.bcs); fill_halos!(sim.v, sim.bcs); fill_halos!(sim.w, sim.bcs)
        fill_halos!(sim.θ, sim.θ_bcs)
        sim.has_moisture && fill_halos!(sim.qv, sim.θ_bcs)
    end

    # Save current state
    m  = sim.mesh
    ii, jj, kk = interior(m)
    u0 = copy(sim.u.data); v0 = copy(sim.v.data); w0 = copy(sim.w.data)
    θ0 = copy(sim.θ.data)
    moist = sim.has_moisture
    q0 = moist ? copy(sim.qv.data) : θ0      # qv is combined like θ (#177)

    _rk3_substep!(sim, Δt, closure, body_force!, post_predictor!, sgs_stress,
                  t0 + Δt)                                                        # u1 = uⁿ + Δt L(uⁿ), then project
    # combine into u1 (already there); save
    u1 = copy(sim.u.data); v1 = copy(sim.v.data); w1 = copy(sim.w.data)
    θ1 = copy(sim.θ.data)

    _rk3_substep!(sim, Δt, closure, body_force!, post_predictor!, sgs_stress,
                  t0 + 2Δt)                                                       # u1 → u1 + Δt L(u1), then project
    # combine to u2 = 3/4 uⁿ + 1/4 (…)
    @views begin
        sim.u.data .= 0.75 .* u0 .+ 0.25 .* sim.u.data
        sim.v.data .= 0.75 .* v0 .+ 0.25 .* sim.v.data
        sim.w.data .= 0.75 .* w0 .+ 0.25 .* sim.w.data
        sim.θ.data .= 0.75 .* θ0 .+ 0.25 .* sim.θ.data
        moist && (sim.qv.data .= 0.75 .* q0 .+ 0.25 .* sim.qv.data)
    end
    combine_open_boundaries!(sim, 0.75, bc0, 0.25)

    _rk3_substep!(sim, Δt, closure, body_force!, post_predictor!, sgs_stress,
                  t0 + 3Δt / 2)                                                   # u2 → u2 + Δt L(u2), then project
    # combine to uⁿ⁺¹ = 1/3 uⁿ + 2/3 (…)
    inv3 = one(T)/3; two3 = 2*inv3
    @views begin
        sim.u.data .= inv3 .* u0 .+ two3 .* sim.u.data
        sim.v.data .= inv3 .* v0 .+ two3 .* sim.v.data
        sim.w.data .= inv3 .* w0 .+ two3 .* sim.w.data
        sim.θ.data .= inv3 .* θ0 .+ two3 .* sim.θ.data
        moist && (sim.qv.data .= inv3 .* q0 .+ two3 .* sim.qv.data)
    end
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
              t_bc = sim.t + Δt) =
    _euler_projection!(sim, Δt, closure, body_force!, post_predictor!, momentum_advect!,
                       sgs_stress, t_bc)

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
    ke = 0.0
    ui = interior_view(sim.u); vi = interior_view(sim.v); wi = interior_view(sim.w)
    @inbounds for I in eachindex(ui)
        ke += ui[I]^2 + vi[I]^2 + wi[I]^2
    end
    return 0.5 * ke * dV
end

# The same sum with per-level volumes: u, v (at cell-centre heights) weigh
# with the cell thickness, w (on z-faces) with the centre spacing, i.e. the
# heights of the u- and w-control volumes of the staggered grid.
function _kinetic_energy_levels(sim::BoussinesqSimulation, m)
    ii, jj, kk = interior(m)
    ud, vd, wd = sim.u.data, sim.v.data, sim.w.data
    ke = 0.0
    @inbounds for k in kk
        hc = _dzc(m, k); hf = _dzf(m, k)
        for j in jj, i in ii
            ke += (ud[i, j, k]^2 + vd[i, j, k]^2) * hc + wd[i, j, k]^2 * hf
        end
    end
    return 0.5 * ke * m.dx * m.dy
end

"""
    max_divergence(sim) -> Float64

Maximum `|∇·u|` over the interior cells. Should sit at floating-point
epsilon after the pressure projection (Chorin's method is exact on the
discrete Laplacian).
"""
function max_divergence(sim::BoussinesqSimulation)
    fill_halos!(sim.u, sim.bcs)
    fill_halos!(sim.v, sim.bcs)
    fill_halos!(sim.w, sim.bcs)
    divergence(sim.div_buf, sim.u, sim.v, sim.w)
    return maximum(abs, interior_view(sim.div_buf))
end

"""
    max_νt(sim) -> Float64

Maximum SGS eddy viscosity `νₜ` over the interior. Zero when
`C_s = 0`; otherwise responds to local strain and gives a sense of
where the SGS model is most active.
"""
max_νt(sim::BoussinesqSimulation) = maximum(interior_view(sim.νt))

"""
    horizontal_mean(field) -> Vector

Horizontal mean of a cell-centered field over the interior, one entry
per interior `z`-level.
"""
function horizontal_mean(field::Field{CellCenter})
    v = interior_view(field)
    return dropdims(sum(v; dims=(1,2)) ./ (size(v,1) * size(v,2)); dims=(1,2))
end

"""
    horizontal_variance(field) -> Vector

Horizontal variance of a cell-centered field over the interior,
`⟨(f - ⟨f⟩)²⟩`, one entry per interior `z`-level.
"""
function horizontal_variance(field::Field{CellCenter})
    v = interior_view(field)
    m = dropdims(sum(v; dims=(1,2)) ./ (size(v,1) * size(v,2)); dims=(1,2))
    var_profile = similar(m)
    @inbounds for k in eachindex(m)
        s = 0.0
        for j in 1:size(v,2), i in 1:size(v,1)
            d = v[i,j,k] - m[k]
            s += d*d
        end
        var_profile[k] = s / (size(v,1) * size(v,2))
    end
    return var_profile
end

"""
    resolved_flux_wθ(sim) -> Vector

Horizontally-averaged resolved kinematic scalar flux ``⟨w'\\,θ'⟩`` at
cell centers (using `w` interpolated in z). One entry per interior
`z`-level.
"""
function resolved_flux_wθ(sim::BoussinesqSimulation)
    θi = interior_view(sim.θ)
    Nz = size(θi, 3)
    profile = zeros(eltype(θi), Nz)
    m_θ = dropdims(sum(θi; dims=(1,2)) ./ (size(θi,1) * size(θi,2)); dims=(1,2))
    Hz = sim.mesh.Hz
    wd = sim.w.data
    ii, jj, _ = interior(sim.mesh)
    @inbounds for k in 1:Nz
        kw = Hz + k
        s = 0.0
        for j in jj, i in ii
            w_cc = 0.5 * (wd[i, j, kw] + wd[i, j, kw + 1])
            s += w_cc * (θi[i - first(ii) + 1, j - first(jj) + 1, k] - m_θ[k])
        end
        profile[k] = s / (length(ii) * length(jj))
    end
    return profile
end

"""
    mean_θ_profile(sim) -> Vector

Horizontal-mean potential-temperature profile `⟨θ⟩(k)` over the
interior, one entry per interior `z`-level. Useful for diagnosing
stratification.
"""
function mean_θ_profile(sim::BoussinesqSimulation)
    θi = interior_view(sim.θ)
    return dropdims(sum(θi; dims=(1,2)) ./ (size(θi,1) * size(θi,2)); dims=(1,2))
end

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

"In-place y[interior] += a · x[interior]."
function _axpy_interior!(y::Field, a::Number, x::Field, m::AbstractMesh)
    ii, jj, kk = interior(m)
    @inbounds @views y.data[ii, jj, kk] .+= a .* x.data[ii, jj, kk]
    return y
end

end # module
