module TimeIntegration

# =========================================================================
# Era II, Phases 23 & 25 — adaptive time stepping and semi-implicit IMEX.
# =========================================================================
#
# Phase 23 (adaptive dt):
#   * `CFLConstraints`      — per-process safety factors + dt bounds.
#   * `cfl_diagnostics`     — the individual limiting dt of every process.
#   * `adaptive_dt`         — the new dt, growth-limited and clamped.
#   * `SubsteppingSchedule` — per-process substep counts (dt/n).
#   * `step_adaptive!`      — production driver: pick dt, step, retry on
#                             failure from a preallocated snapshot, then
#                             substep the fast processes.
#
# Phase 25 (semi-implicit IMEX):
#   * `solve_implicit_diffusion!` — matrix-free Jacobi-PCG solve of
#                             `(I - dt ∇·(ν∇)) φ = rhs`.
#   * `ARK2`, `Rosenbrock2` — 2nd-order additive RK (ARS(2,2,2)) and a
#                             2-stage Rosenbrock-W method.
#   * `step_imex!`          — explicit advection/buoyancy/Coriolis/forcing,
#                             implicit diffusion, projection per stage.
#   * `imex_stability_limit`— explicit-only vs diffusive dt limit.
#
# Everything here operates on `BoussinesqSimulation` and reuses its
# preallocated buffers where possible; the extra state each driver needs
# lives in a workspace object cached per simulation (see `imex_workspace`
# and `state_snapshot`) so that a step does not allocate.

using ..Grids
using ..Grids: AbstractMesh, RegularGrid, CellCenter, FaceX, FaceY, FaceZ,
               Location, interior, totalsize
using ..Fields
using ..Fields: Field, interior_view, location, interior_map!
using ..Backends: Backends, plane_mean!, range_max, interior_max
using ..BoundaryConditions
using ..BoundaryConditions: FieldBoundaryConditions, fill_halos!,
                            Periodic, NoFlux, ImpermeableWall, ValueBC, FluxBC,
                            is_wall, homogeneous_bcs, pinned_face_indices,
                            has_open_boundaries, update_open_boundaries!,
                            prepare_open_boundaries!, assert_open_face_correction,
                            open_boundary_snapshot, restore_open_boundaries!,
                            open_intermediate_velocity!, rewind_open_boundaries!,
                            has_radiating_outflow
using ..Operators
using ..Operators: divergence, laplacian, ∂x, ∂y, ∂z,
                   momentum_advect!, advect_centered!,
                   add_buoyancy!, add_coriolis!, compute_smagorinsky_νt!,
                   add_surface_flux!, add_sponge!,
                   AbstractSGSClosure, SGSFields, compute_sgs!, sgs_fields, advance_sgs!,
                   sgs_state, restore_sgs_state!, sgs_closure, _active_sgs,
                   add_sgs_stress_divergence!, _check_sgs_stress,
                   sgs_tracer_diffusivity!,
                   sgs_scalar_tensor, add_tensor_diffusion_scalar!, bounded_directions,
                   sgs_explicit_stress
using ..PressureSolvers: solve!
using ..Simulations
using ..Simulations: BoussinesqSimulation, fill_state_halos!, max_divergence,
                     _dzc, _dzf, _dz_min,     # vertical metrics, stretched-aware (#151)
                     _project_velocity!,      # the projection (steps 8–10 of step!)
                     axpy!, pressure_from_potential!, copy_storage!, storage_like,
                     _Lin2,                   # the field algebra kernels (KA-j, #306)
                     _step_begin!, _step_end!, _step_abort!, _note_pick!, _wall   # #336
using ..Operators.FiniteVolume: grid_metrics, _div_at   # the one divergence stencil
using ..IO: allfinite                    # the one finiteness check (#346)
using ..TimeSteppers: step!
using ..RunLog: current_domain   # structured events (LOG-a, #333)
using ..TurbulenceStability: buoyancy_frequency_squared   # the one N² (P58-a)

export CFLConstraints, cfl_diagnostics, adaptive_dt, stress_form_cfl_viscosity,
       SubsteppingSchedule, substeps_for,
       StateSnapshot, state_snapshot, save_state!, restore_state!,
       step_adaptive!, NumericalFailure, is_numerical_failure,
       DiffusionWorkspace, solve_implicit_diffusion!,
       IMEXScheme, ARK2, Rosenbrock2,
       IMEXWorkspace, imex_workspace, step_imex!, imex_stability_limit

# =========================================================================
# Small shared helpers
# =========================================================================

_tendency(sim::BoussinesqSimulation, name::Symbol) =
    name === :u ? sim.tu : name === :v ? sim.tv : name === :w ? sim.tw :
    name === :θ ? sim.tθ : sim.tqv

"Prognostic fields advanced by the IMEX drivers (`qv` only when moist)."
_active(sim::BoussinesqSimulation) =
    sim.has_moisture ? (:u, :v, :w, :θ, :qv) : (:u, :v, :w, :θ)

_is_scalar_name(name::Symbol) = name === :θ || name === :qv


# =========================================================================
# Phase 23 — CFL constraints and diagnostics
# =========================================================================

"""
    CFLConstraints(; C_advective = 0.5, C_diffusive = 0.25, C_buoyancy = 0.5,
                     C_coriolis = 0.5, dt_min = 1e-8, dt_max = Inf,
                     growth_max = 1.1)

Safety factors and bounds for adaptive time stepping. Each `C_*` is the
fraction of the corresponding *linear stability limit* that a step is
allowed to use:

| Field         | Limit enforced                                                |
| :------------ | :------------------------------------------------------------ |
| `C_advective` | `dt (|u|/Δx + |v|/Δy + |w|/Δz) ≤ C_advective` (per cell)      |
| `C_diffusive` | `dt (ν+νₜ) (Δx⁻²+Δy⁻²+Δz⁻²) ≤ C_diffusive` (2νₜ, stress form) |
| `C_buoyancy`  | `N dt ≤ C_buoyancy` (Brunt–Väisälä / gravity-wave limit)      |
| `C_coriolis`  | `|f| dt ≤ C_coriolis`                                         |

`dt_min` / `dt_max` clamp the result; `growth_max` bounds the ratio
between consecutive steps so `dt` ramps up smoothly instead of jumping
(a jump is the classic way to trip a leapfrog-free explicit scheme even
though every instantaneous CFL number looks fine).

The explicit diffusive stability bound of the 7-point Laplacian with
forward Euler is `dt ν (Δx⁻²+Δy⁻²+Δz⁻²) ≤ 1/2`, so the default
`C_diffusive = 0.25` is a factor-2 safety margin. Likewise
`C_advective = 0.5` is half the `dt Σ|uᵢ|/Δxᵢ ≤ 1` bound of a 2nd-order
centered / RK3 combination.

See [`cfl_diagnostics`](@ref), [`adaptive_dt`](@ref), [`step_adaptive!`](@ref).
"""
struct CFLConstraints{T<:AbstractFloat}
    C_advective::T
    C_diffusive::T
    C_buoyancy::T
    C_coriolis::T
    dt_min::T
    dt_max::T
    growth_max::T
end

function CFLConstraints(::Type{T} = Float64;
                        C_advective::Real = 0.5,
                        C_diffusive::Real = 0.25,
                        C_buoyancy::Real  = 0.5,
                        C_coriolis::Real  = 0.5,
                        dt_min::Real      = 1e-8,
                        dt_max::Real      = Inf,
                        growth_max::Real  = 1.1) where {T<:AbstractFloat}
    C_advective > 0 || throw(ArgumentError("C_advective must be > 0"))
    C_diffusive > 0 || throw(ArgumentError("C_diffusive must be > 0"))
    C_buoyancy  > 0 || throw(ArgumentError("C_buoyancy must be > 0"))
    C_coriolis  > 0 || throw(ArgumentError("C_coriolis must be > 0"))
    0 < dt_min <= dt_max || throw(ArgumentError("need 0 < dt_min ≤ dt_max"))
    growth_max >= 1 || throw(ArgumentError("growth_max must be ≥ 1"))
    return CFLConstraints{T}(T(C_advective), T(C_diffusive), T(C_buoyancy),
                             T(C_coriolis), T(dt_min), T(dt_max), T(growth_max))
end

# --- per-cell advective rate ---------------------------------------------
# The correct multidimensional CFL number is the *sum* of the directional
# ones evaluated in the same cell, not the min/max of global maxima: a
# cell with u = v = w = U is three times as restrictive as a cell with
# only u = U. Each cell takes the larger of its two opposing face values
# in every direction (conservative, and exact for uniform flow).
#
# `max` is exact and commutative, so the thread-invariant `range_max` gives
# the bits of the old serial loop (NaN-propagating; KA-b, #296).
function _max_advective_rate(ud, vd, wd, m::AbstractMesh{T}) where {T}
    r = range_max(Backends.backend(ud), _adv_rate_at, interior(m), ud, vd, wd,
                  1 / m.dx, 1 / m.dy, _rate_iz(m))
    return max(zero(T), T(r))
end

# The vertical factor of the rate: 1/Δz on a RegularGrid; on a vertically
# stretched grid (#151) the local 1/Δz_c[k], so the thin near-surface cells
# bind, not the mean spacing.
_rate_iz(m::RegularGrid) = 1 / m.dz
_rate_iz(m::AbstractMesh) = m
@inline _iz_at(iz::Real, k) = iz
@inline _iz_at(m, k) = 1 / _dzc(m, k)

@inline function _adv_rate_at(i, j, k, ud, vd, wd, idx, idy, iz)
    @inbounds begin
        au = max(abs(ud[i, j, k]), abs(ud[i+1, j, k]))
        av = max(abs(vd[i, j, k]), abs(vd[i, j+1, k]))
        aw = max(abs(wd[i, j, k]), abs(wd[i, j, k+1]))
    end
    return au * idx + av * idy + aw * _iz_at(iz, k)
end

# The fused CFL sweep (KA-j, #306): the advective rate and the maxima of the
# four SGS coefficients in one pass (one `range_max` with a tuple body, each
# component NaN-propagating): before, one sweep per maximum.
@inline _cfl_at(i, j, k, ud, vd, wd, idx, idy, iz, νh, νv, κh, κv) =
    @inbounds (_adv_rate_at(i, j, k, ud, vd, wd, idx, idy, iz),
               νh[i, j, k], νv[i, j, k], κh[i, j, k], κv[i, j, k])

# (rate, νh_max, νv_max, κh_max, κv_max) of the state; the SGS maxima are 0
# when no eddy viscosity acts (`active = false`; then only u, v, w are read).
function _cfl_maxima(sim::BoussinesqSimulation{T}, f, active::Bool) where {T}
    m = sim.mesh
    ud, vd, wd = sim.u.data, sim.v.data, sim.w.data
    active || return (_max_advective_rate(ud, vd, wd, m), zero(T), zero(T), zero(T), zero(T))
    r = range_max(Backends.backend(ud), _cfl_at, interior(m), ud, vd, wd, 1 / m.dx, 1 / m.dy,
                  _rate_iz(m), f.νh.data, f.νv.data, f.κh.data, f.κv.data)
    return (max(zero(T), T(r[1])), T(r[2]), T(r[3]), T(r[4]), T(r[5]))
end

# Brunt-Väisälä frequency from the horizontal-mean (resolved) θ profile:
# the largest N² across the faces between the plane means, from the
# TurbulenceStability contract (Boussinesq form, g/θ₀). Allocation-free:
# carries the previous level's mean in a scalar.
function _brunt_vaisala(sim::BoussinesqSimulation{T}) where {T}
    sim.has_buoyancy || return zero(T)
    m = sim.mesh
    kk = interior(m)[3]
    # the plane means in the canonical, thread-invariant order (KA-b, #296)
    θm = plane_mean!(Vector{T}(undef, length(kk)), sim.θ)
    N²max = zero(T)
    prev  = zero(T)
    @inbounds for (kc, k) in enumerate(kk)
        mean_k = θm[kc]
        if kc > 1
            # the two plane means sit Δz_f[k] apart (= m.dz on a RegularGrid)
            N²max = max(N²max, buoyancy_frequency_squared(prev, mean_k, _dzf(m, k),
                                                          sim.g_accel, sim.θ₀))
        end
        prev = mean_k
    end
    return sqrt(max(N²max, zero(T)))
end

"""
    cfl_diagnostics(sim, constraints = CFLConstraints(); sgs = nothing,
                    update_νt = true, sgs_active = <closure present>,
                    sgs_stress = :laplacian, limits = ())

Report the time-step limit imposed by *each* process separately. Returns a
`NamedTuple`:

| Key                 | Meaning                                                        |
| :------------------ | :------------------------------------------------------------- |
| `dt_adv`            | advective limit, per-cell `Σ|uᵢ|/Δxᵢ` form                      |
| `dt_diff`           | total (molecular + SGS) momentum-diffusion limit               |
| `dt_buoy`           | `C_buoyancy / N` from the resolved mean θ profile               |
| `dt_cor`            | `C_coriolis / |f|`                                             |
| `dt_sgs`            | SGS-only limit `C_diffusive Δ²_harm / max(νt_cfl, κt_max)`      |
| `dt_limits`         | the smallest of `limits` (`Inf` for none)                      |
| `limits_limiter`    | its label (`:none` for none)                                   |
| `dt_limiting`       | `min` of the five and `limits`                                 |
| `limiter`           | `:advective`, `:diffusive`, `:buoyancy`, `:coriolis`, `:sgs` or the label of a limit |
| `advective_rate`    | `max_cell Σ|uᵢ|/Δxᵢ` (so the CFL number is `dt · rate`)         |
| `N`                 | Brunt–Väisälä frequency (0 when buoyancy is off)                |
| `νt_max`            | maximum SGS eddy viscosity                                     |
| `κt_max`            | maximum SGS eddy diffusivity (θ)                               |
| `νt_cfl`            | SGS viscosity of the momentum limits (see `sgs_stress`)        |
| `ν_total`           | `ν + νt_cfl`                                                   |

The SGS closure is the one the next step will run: the `sgs` keyword (an
[`AbstractSGSClosure`](@ref Aeolus.Operators.AbstractSGSClosure) or a
function `f(νt, u, v, w)`), else the built-in Smagorinsky when `C_s ≠ 0`.
With `update_νt = true` (the default) its coefficients are recomputed from
the current state into `sim.νt` / `sim.κt` with the **pure**
[`compute_sgs!`](@ref Aeolus.Operators.compute_sgs!) — no closure memory
advances, so the estimate can be made any number of times per step; pass
`false` to reuse what is in `sim.νt` / `sim.κt`. `sgs_active` (default:
a closure is present) says whether an eddy viscosity acts on the step at all.

`sgs_stress` is the form of the SGS momentum divergence the step runs
(the cores' keyword). For `:laplacian` (the default), `νt_cfl = νt_max`.
For `:symmetric`, the stress form's factor 2 on the normal stresses can
double the operator's spectral radius, so `νt_cfl =`
[`stress_form_cfl_viscosity`](@ref)`(νh_max, νv_max, mesh)`, which equals
`2 νt_max` when `νh = νv`.

The SGS limit uses the larger of the momentum viscosity `νt_cfl` and the eddy
diffusivity, each as stored: for a Prandtl-number closure `κt = νt/Prₜ`,
for a closure with its own κ (Era VI) whatever it computed.

`limits` are the explicit limits of a model's components (ONE-a, #450:
[`limit_dt`](@ref Aeolus.Components.limit_dt), e.g. the Coriolis
component's `C_coriolis/|f|`), a tuple of `(dt, label)` pairs or `nothing`
(no limit). They are compared right after the core's own Coriolis limit,
in their order, so a component that carries the Coriolis term gives the
pick and the limiter of the core's built-in Coriolis term, bit for bit.

Ties are broken in the table order above, so a pure-molecular run where
`dt_diff == dt_sgs` reports `:diffusive`.

The rates and the limiter are also noted as the dt pick of the current
state in the simulation's [`StepRecord`](@ref Aeolus.Simulations.StepRecord):
the next step taken from this state reports `dt · advective_rate` as its
advective CFL number, and the limiter (#336).
"""
function cfl_diagnostics(sim::BoussinesqSimulation{T},
                         c::CFLConstraints = CFLConstraints(T);
                         sgs = nothing,
                         update_νt::Bool = true,
                         sgs_active::Union{Nothing,Bool} = nothing,
                         sgs_stress::Symbol = :laplacian, limits = ()) where {T}
    m = sim.mesh
    INF = T(Inf)
    t_cfl = time_ns()

    fill_halos!((sim.u, sim.v, sim.w), sim.bcs)

    # --- the SGS coefficients the step will use ---------------------------
    closure = _active_sgs(sim, sgs, nothing)
    active  = sgs_active === nothing ? closure !== nothing : sgs_active
    if update_νt && closure !== nothing
        fill_halos!(sim.θ, sim.θ_bcs)                      # θ-aware closures
        sim.has_moisture && fill_halos!(sim.qv, passive_scalar_bcs(sim.θ_bcs))
        compute_sgs!(closure, sim)                         # pure: u, v, w unchanged
    end
    f = closure === nothing ? SGSFields(sim.νt, sim.κt) : sgs_fields(closure, sim)

    # --- one sweep: the advective rate and the SGS maxima (KA-j, #306) ----
    rate, νh_max, νv_max, κh_max, κv_max = _cfl_maxima(sim, f, active)

    # --- advective --------------------------------------------------------
    # A NaN velocity must poison the limit rather than silently reporting
    # "no constraint"; `adaptive_dt` then collapses to dt_min.
    dt_adv = isnan(rate) ? T(NaN) : rate > 0 ? c.C_advective / rate : INF

    # --- diffusive (molecular + SGS) --------------------------------------
    νt_max  = max(νh_max, νv_max)
    κt_max  = max(κh_max, κv_max)
    # the thinnest cell binds the explicit diffusive limit (#151)
    Δ²harm  = one(T) / (1 / m.dx^2 + 1 / m.dy^2 + 1 / _dz_min(m)^2)
    # the SGS viscosity of the momentum limit: νt_max for the Laplacian form,
    # the stress form's own spectral bound (≤ 2 νt_max) for `:symmetric`
    νt_cfl  = _check_sgs_stress(sgs_stress) === :symmetric ?
              stress_form_cfl_viscosity(νh_max, νv_max, m) : νt_max
    ν_total = sim.ν + νt_cfl
    dt_diff = ν_total > 0 ? c.C_diffusive * Δ²harm / ν_total : INF

    # SGS on its own: the larger of the eddy viscosity and the scalar eddy
    # diffusivity (κt = νt/Prₜ exceeds νt for Prₜ < 1).
    κ_sgs  = max(νt_cfl, κt_max)
    dt_sgs = κ_sgs > 0 ? c.C_diffusive * Δ²harm / κ_sgs : INF

    # --- buoyancy (gravity-wave / Brunt-Väisälä) --------------------------
    N       = _brunt_vaisala(sim)
    dt_buoy = N > 0 ? c.C_buoyancy / N : INF

    # --- Coriolis ---------------------------------------------------------
    f      = abs(sim.f_coriolis)
    dt_cor = f > 0 ? c.C_coriolis / f : INF

    # --- who binds? -------------------------------------------------------
    dt_limiting = dt_adv
    limiter     = :advective
    if dt_diff < dt_limiting; dt_limiting = dt_diff; limiter = :diffusive; end
    if dt_buoy < dt_limiting; dt_limiting = dt_buoy; limiter = :buoyancy;  end
    if dt_cor  < dt_limiting; dt_limiting = dt_cor;  limiter = :coriolis;  end
    dt_lim = INF; lim_label = :none                # the components' limits (ONE-a, #450)
    for l in limits
        l === nothing && continue
        d_l = T(first(l))
        d_l < dt_lim && (dt_lim = d_l; lim_label = last(l)::Symbol)
        if d_l < dt_limiting; dt_limiting = d_l; limiter = last(l)::Symbol; end
    end
    if dt_sgs  < dt_limiting; dt_limiting = dt_sgs;  limiter = :sgs;       end

    # the pick of this state, for the step record of the step it is used for (#336)
    rec = sim.workspaces.rec
    _note_pick!(rec, sim, rate, max(ν_total, κ_sgs) / Δ²harm, dt_limiting, limiter)
    rec.wall_cfl += _wall(t_cfl)

    return (dt_adv = dt_adv, dt_diff = dt_diff, dt_buoy = dt_buoy,
            dt_cor = dt_cor, dt_sgs = dt_sgs, dt_limits = dt_lim, limits_limiter = lim_label,
            dt_limiting = dt_limiting, limiter = limiter,
            advective_rate = rate, N = N, νt_max = νt_max, κt_max = κt_max,
            νt_cfl = νt_cfl, ν_total = ν_total)
end

"""
    stress_form_cfl_viscosity(νh_max, νv_max, mesh) -> ν_cfl

The eddy viscosity that, put into the Laplacian diffusive limit
`dt ν (Δx⁻²+Δy⁻²+Δz⁻²) ≤ 2` of forward Euler, bounds the explicit stress-form
SGS momentum operator `−∂ⱼτᵢⱼ`, `τᵢⱼ = −2νᵢⱼSᵢⱼ`
([`add_sgs_stress_divergence!`](@ref Aeolus.Operators.add_sgs_stress_divergence!)),
for *any* velocity field and any `0 ≤ νh ≤ νh_max`, `0 ≤ νv ≤ νv_max`:

    ν_cfl = 2 max(νh_max (Δx⁻²+Δy⁻²) + νv_max Δz⁻²,  νv_max (Δx⁻²+Δy⁻²+Δz⁻²))
              / (Δx⁻²+Δy⁻²+Δz⁻²)

so `ν_cfl = 2 νt` when `νh = νv`. Derivation: the operator is symmetric
negative semi-definite, `−⟨u, L u⟩ = Σ 2νᵢⱼSᵢⱼSᵢⱼ ΔV`, so its spectral radius
is the largest Rayleigh quotient. Each diagonal strain is one difference,
`(u_{i+1}−u_i)²/Δx² ≤ 2(u_{i+1}²+u_i²)/Δx²`, and each off-diagonal pair is
bounded by `(a+b)² ≤ 2a²+2b²`; every face value enters two differences per
direction, so the `u` and `v` rows contribute at most
`8(νh_max Δx⁻² + νh_max Δy⁻² + νv_max Δz⁻²)` and the `w` row (which sees
only `νv`) `8 νv_max (Δx⁻²+Δy⁻²+Δz⁻²)`. That is twice the 7-point-Laplacian
radius `4ν(Δx⁻²+Δy⁻²+Δz⁻²)`. The factor 2 comes from the normal stresses
`τᵢᵢ = −2ν∂uᵢ/∂xᵢ`. The bound is sharp: with constant ν it is reached by the
irrotational grid-scale mode `u = ∇φ`, φ a 3-D checkerboard, where the
operator is `ν(∇² + ∇∇·)` and both terms give `−4ν(Δx⁻²+Δy⁻²+Δz⁻²)`.

On *projected* fields the radius of the projected operator `P L P` is no
larger than the Laplacian bound, because `Σ 2S:S = Σ|∇u|²` for solenoidal
periodic `u`. [`cfl_diagnostics`](@ref) still uses this unprojected bound
for `sgs_stress = :symmetric`: it holds for every velocity field, not just
for one an exact orthogonal projection has made divergence-free (see the
theory page, "Explicit stability of the stress form").
"""
function stress_form_cfl_viscosity(νh_max::Real, νv_max::Real, m)
    ix2, iy2, iz2 = 1 / m.dx^2, 1 / m.dy^2, 1 / _dz_min(m)^2
    s = ix2 + iy2 + iz2
    return 2 * max(νh_max * (ix2 + iy2) + νv_max * iz2, νv_max * s) / s
end

"""
    adaptive_dt(sim, constraints; dt_prev = nothing, diagnostics = nothing,
                scheme = nothing, sgs = nothing, sgs_stress = :laplacian, limits = ())

The time step to take next: the smallest per-process limit from
[`cfl_diagnostics`](@ref) — or, when `scheme` is an [`IMEXScheme`](@ref),
the smallest limit of the processes that scheme treats *explicitly*
(advection, buoyancy, Coriolis, scaled by [`explicit_cfl_factor`](@ref);
the `dt_explicit` of [`imex_stability_limit`](@ref)), because the
diffusive limits do not bind an implicit diffusion solve — then

  1. limited to `growth_max · dt_prev` when `dt_prev` is given, and
  2. clamped to `[dt_min, dt_max]`.

A non-finite limit (which only happens if the state already contains
`NaN`/`Inf`) collapses to `dt_min`, so `step_adaptive!`'s retry logic sees
a usable number rather than propagating the `NaN`. Pass `diagnostics` to
reuse a `cfl_diagnostics` result instead of recomputing it; `sgs` is the
closure the step will run and `sgs_stress` the form of its momentum
divergence (see [`cfl_diagnostics`](@ref)); `limits` are a model's
component limits (`(dt, label)` pairs, see [`cfl_diagnostics`](@ref)),
which bind before the growth limit and the clamp. The step and its limiter
(`:growth`, `:dt_max`, `:dt_min` when a clamp binds) are noted in the
simulation's [`StepRecord`](@ref Aeolus.Simulations.StepRecord) (#336).
"""
function adaptive_dt(sim::BoussinesqSimulation{T}, c::CFLConstraints;
                     dt_prev = nothing, diagnostics = nothing,
                     scheme = nothing, sgs = nothing,
                     sgs_stress::Symbol = :laplacian, limits = ()) where {T}
    d  = diagnostics === nothing ? cfl_diagnostics(sim, c; sgs = sgs, sgs_stress = sgs_stress,
                                                   limits = limits) :
                                   diagnostics
    dt, limiter = scheme isa IMEXScheme ? _explicit_limit(d, scheme) : (d.dt_limiting, d.limiter)
    isnan(dt) && return c.dt_min
    if dt_prev !== nothing
        dtp = T(dt_prev)
        dtp > 0 && c.growth_max * dtp < dt && (dt = c.growth_max * dtp; limiter = :growth)
    end
    dt < c.dt_min && (limiter = :dt_min)
    dt > c.dt_max && (limiter = :dt_max)
    dt = clamp(dt, c.dt_min, c.dt_max)
    _note_pick!(sim.workspaces.rec, sim, dt, limiter)    # the step record's limiter (#336)
    return dt
end

# =========================================================================
# Phase 23 — process substepping
# =========================================================================

"""
    SubsteppingSchedule(; default = 1, kwargs...)
    SubsteppingSchedule(:microphysics => 4, :chemistry => 2)

How many substeps each *fast* process takes inside one dynamics step.
A process registered with `n` substeps is called `n` times with `dt/n`,
which is how stiff-but-cheap parameterizations (Kessler microphysics,
chemical kinetics, droplet sedimentation) are kept stable without
shrinking the dynamical `dt` for the whole model.

```julia
sched = SubsteppingSchedule(microphysics = 4, chemistry = 2)
substeps_for(sched, :microphysics)   # 4
substeps_for(sched, :radiation)      # 1  (the default)
```
"""
struct SubsteppingSchedule
    counts::Dict{Symbol,Int}
    default::Int
end

function SubsteppingSchedule(pairs::Pair{Symbol,<:Integer}...; default::Integer = 1)
    default >= 1 || throw(ArgumentError("default substep count must be ≥ 1"))
    counts = Dict{Symbol,Int}()
    for (k, v) in pairs
        v >= 1 || throw(ArgumentError("substep count for :$k must be ≥ 1, got $v"))
        counts[k] = Int(v)
    end
    return SubsteppingSchedule(counts, Int(default))
end

SubsteppingSchedule(; default::Integer = 1, kwargs...) =
    SubsteppingSchedule((Symbol(k) => Int(v) for (k, v) in kwargs)...; default = default)

"""
    substeps_for(schedule, name) -> Int

Number of substeps registered for process `name` (the schedule's
`default`, normally `1`, when `name` is not registered). `nothing`
behaves like an empty schedule.
"""
substeps_for(s::SubsteppingSchedule, name::Symbol) = get(s.counts, name, s.default)
substeps_for(::Nothing, ::Symbol) = 1

Base.show(io::IO, s::SubsteppingSchedule) =
    print(io, "SubsteppingSchedule(", join(("$k=>$v" for (k, v) in sort!(collect(s.counts), by = first)), ", "),
          "; default=", s.default, ")")

# =========================================================================
# Phase 23 — state snapshot (for step retries)
# =========================================================================

"""
    StateSnapshot(sim)

Preallocated copy of the state (`u, v, w, p, θ, qv`, the SGS coefficients
`νt, κt`, and the clock) used by [`step_adaptive!`](@ref) to roll a failed
step back. The buffers are allocated once, on the simulation's backend;
[`save_state!`](@ref) and [`restore_state!`](@ref) only copy, as pointwise
kernels over the whole storage. Halos are copied too, so a restore is
bit-identical to the pre-step state.

`νt` and `κt` are part of it since KA-j (#306): a rejected try leaves the
coefficients of its blown-up state in `sim.νt` / `sim.κt`, which the
diagnostics and a checkpoint read; the rollback now restores them with the
prognostic fields (the next try recomputes them before use either way).
"""
mutable struct StateSnapshot{T, A<:AbstractArray{T,3}}
    u::A
    v::A
    w::A
    p::A
    θ::A
    qv::A
    νt::A
    κt::A
    t::T
    valid::Bool
end

function StateSnapshot(sim::BoussinesqSimulation{T}) where {T}
    al(f) = storage_like(f.data)
    return StateSnapshot(al(sim.u), al(sim.v), al(sim.w), al(sim.p), al(sim.θ), al(sim.qv),
                         al(sim.νt), al(sim.κt), sim.t, false)
end

# The (snapshot buffer, simulation field) pairs a snapshot holds. A
# sound-proof core carries no qv (its buffer is empty, Anelastic).
_snapshot_pairs(snap::StateSnapshot, sim) =
    ((snap.u, sim.u.data), (snap.v, sim.v.data), (snap.w, sim.w.data), (snap.p, sim.p.data),
     (snap.θ, sim.θ.data), (snap.νt, sim.νt.data), (snap.κt, sim.κt.data))

"Copy `sim`'s state into `snap` (pointwise kernels, no allocation)."
function save_state!(snap::StateSnapshot, sim::BoussinesqSimulation)
    for (a, f) in _snapshot_pairs(snap, sim)
        copy_storage!(a, f)
    end
    copy_storage!(snap.qv, sim.qv.data)
    snap.t     = sim.t
    snap.valid = true
    return snap
end

"""
    restore_state!(sim, snap) -> sim

Restore `sim`'s state (fields with their halos, `νt`, `κt` and the clock)
from `snap`. The copies are kernels; the backend is synchronized before
returning, so the caller (a retry) reads the restored state.
"""
function restore_state!(sim::BoussinesqSimulation, snap::StateSnapshot)
    snap.valid || throw(ArgumentError("StateSnapshot has never been saved"))
    for (a, f) in _snapshot_pairs(snap, sim)
        copy_storage!(f, a)
    end
    copy_storage!(sim.qv.data, snap.qv)
    Backends.synchronize(Backends.backend(sim.u.data))
    sim.t = snap.t
    return sim
end

"""
    state_snapshot(sim) -> StateSnapshot

The snapshot buffer of `sim`: allocated on first use and kept in the
simulation's [`StepWorkspaces`](@ref Aeolus.Simulations.StepWorkspaces)
(KA-j, #306; before, a process-global weak cache). This is what makes
`step_adaptive!`'s retry path allocation-free.
"""
state_snapshot(sim::BoussinesqSimulation) = _workspace_snapshot(sim)

# the snapshot slot of a core's StepWorkspaces (the sound-proof cores of
# Anelastic use it too)
function _workspace_snapshot(sim)
    w = sim.workspaces
    w.snapshot === nothing && (w.snapshot = StateSnapshot(sim))
    return w.snapshot::StateSnapshot
end

"""
    NumericalFailure(msg) <: Exception

The exception a dynamical core, a solver or a physics step throws when it
fails **numerically** — an iterative solver that diverges or does not
converge, a state that has left its physical range — as opposed to a
programming error. Step-rejecting drivers
([`NestedRun`](@ref Aeolus.Nesting.NestedRun)) roll back and retry at a
smaller step on it ([`is_numerical_failure`](@ref)); any other exception is
rethrown.
"""
struct NumericalFailure <: Exception
    msg::String
end
Base.showerror(io::IO, e::NumericalFailure) = print(io, "NumericalFailure: ", e.msg)

"""
    is_numerical_failure(err, sim) -> Bool

Whether the exception `err`, thrown by a step of `sim`, is a numerical
failure that a smaller step may cure, rather than a bug to be reported:

* a [`NumericalFailure`](@ref) (a solver that diverged);
* a `DomainError` (`sqrt`/`log` of a negative value), an `InexactError`
  (a non-finite value converted to an integer, e.g. a step count
  `ceil(Int, R/Δt)` from an infinite rate) or an `OverflowError`;
* **any** exception thrown while the prognostic state of `sim` has become
  non-finite (the throw is a consequence of the blow-up).

A `MethodError`, `BoundsError`, `ArgumentError`, `DimensionMismatch`, …
thrown from a finite state is a programming or set-up error: retrying it at
half the step only hides it (#327, M6). An `InterruptException` is never a
numerical failure.
"""
function is_numerical_failure(err, sim)
    err isa InterruptException && return false
    err isa Union{NumericalFailure, DomainError, InexactError, OverflowError} && return true
    return !_finite_state(sim)
end

# `_all_finite`, but false (not a throw) for a state it cannot inspect.
_finite_state(sim) = try
    _all_finite(sim)
catch
    false
end

# Every prognostic interior value finite: the one finiteness check of the
# blowup guard, `IO.allfinite` (branch-free; #346).
function _all_finite(sim::BoussinesqSimulation)
    allfinite(interior_view(sim.u)) || return false
    allfinite(interior_view(sim.v)) || return false
    allfinite(interior_view(sim.w)) || return false
    allfinite(interior_view(sim.θ)) || return false
    allfinite(interior_view(sim.p)) || return false
    sim.has_moisture && !allfinite(interior_view(sim.qv)) && return false
    return true
end

# max |f| over the interior: exact and NaN-propagating, the bits of
# `maximum(abs, interior_view(f))` at any thread count (KA-b, #296); the
# acceptance test of a nested run (Nesting) uses it
_max_abs(f::Field) = range_max(Backends.backend(f.data), Simulations._abs_at, interior(f.mesh), f.data)

# ---- the fused step guard (KA-j, #306) ------------------------------------
#
# What `step_adaptive!` checks after a step, in one sweep over the interior
# (before: one finiteness sweep per field, the divergence kernel and a
# maximum over it, and one maximum per velocity component). Per point:
#   * `x * 0` summed over every prognostic value: 0 if all are finite, NaN
#     otherwise (`Inf * 0` is NaN);
#   * |∇·u| with the divergence stencil of `divergence` (`_div_at`);
#   * max(|u|, |v|, |w|).
# The NaN-propagating component-wise maximum of `range_max` combines them:
# `max` is exact, so |∇·u| and the speed are the bits of `max_divergence`
# and the per-component maxima they replace, and any non-finite value
# anywhere turns the first component into NaN. The velocity halos are
# filled first, as `max_divergence` did.
@inline function _guard_at(i, j, k, ud, vd, wd, θd, pd, qd, gm)
    @inbounds begin
        u = ud[i, j, k]; v = vd[i, j, k]; w = wd[i, j, k]
        fin = u * 0 + v * 0 + w * 0 + θd[i, j, k] * 0 + pd[i, j, k] * 0 + _nf(qd, i, j, k)
        return (fin, abs(_div_at(i, j, k, ud, vd, wd, gm)), max(abs(u), abs(v), abs(w)))
    end
end
@inline _nf(::Nothing, i, j, k) = false
@inline _nf(q::AbstractArray, i, j, k) = @inbounds q[i, j, k] * 0

"""
    step_guard(sim) -> (; finite, divergence, speed)

The post-step checks of [`step_adaptive!`](@ref) in one fused,
NaN-propagating sweep over the interior (KA-j, #306): `finite` (every
prognostic value `u, v, w, θ, p` — and `qv` when moist — is finite),
`divergence = max|∇·u|` and `speed = max(|u|, |v|, |w|)`. The velocity
halos are filled first. `divergence` and `speed` are the bits of
[`max_divergence`](@ref) and of the per-component maxima (the maximum is
exact); any NaN or Inf in the velocities also makes them NaN or Inf.
Thread-invariant.
"""
function step_guard(sim::BoussinesqSimulation{T}) where {T}
    fill_halos!((sim.u, sim.v, sim.w), sim.bcs)
    m = sim.mesh
    qd = sim.has_moisture ? sim.qv.data : nothing
    fin, d, s = range_max(Backends.backend(sim.u.data), _guard_at, interior(m), sim.u.data,
                          sim.v.data, sim.w.data, sim.θ.data, sim.p.data, qd, grid_metrics(m))
    return (finite = fin == 0, divergence = d, speed = s)
end

# The reason a step is rejected, in the order of the separate checks it
# replaces: non-finite state, then divergence, then the speed cap.
function _guard_reason(sim, div_tol, u_cap)
    g = step_guard(sim)
    g.finite || return :nonfinite
    (!isfinite(g.divergence) || g.divergence > div_tol) && return :divergence
    g.speed > u_cap && return :velocity
    return :ok
end

# =========================================================================
# Phase 23 — the adaptive driver
# =========================================================================

_fast_process_pairs(::Nothing) = ()
_fast_process_pairs(f::Function) = (:default => f,)
_fast_process_pairs(t::Tuple) = t
_fast_process_pairs(v::AbstractVector) = Tuple(v)
_fast_process_pairs(d::AbstractDict) = Tuple(k => v for (k, v) in d)
_fast_process_pairs(nt::NamedTuple) = Tuple(k => getfield(nt, k) for k in keys(nt))

"""
    step_adaptive!(sim, constraints; dt = nothing, dt_prev = nothing,
                   body_force! = nothing, post_predictor! = nothing,
                   fast_processes = nothing, schedule = SubsteppingSchedule(),
                   max_retries = 3, stepper = step!, scheme = nothing,
                   snapshot = state_snapshot(sim),
                   divergence_tol = 1e-6, max_velocity = Inf, sgs = nothing)
    -> (sim, dt_used, n_retries)

Production adaptive driver. One call:

 1. picks `dt` from [`adaptive_dt`](@ref) (or uses the `dt` you pass,
    which is how you deliberately overshoot the stability limit); with
    `scheme = ARK2()` the pick uses only the explicit processes' limit,
    since the diffusion is implicit;
 2. saves the prognostic state into a preallocated
    [`StateSnapshot`](@ref);
 3. takes the step with `stepper(sim, dt; body_force!, post_predictor!)` —
    pass `scheme = ARK2()` to step with [`step_imex!`](@ref) instead;
 4. validates the result. A step **fails** if it throws, if any
    prognostic value is non-finite, if `max_divergence(sim) >
    divergence_tol` (the projection should leave it at roundoff, so a
    finite but large divergence means the solve diverged), or if
    `max|u| > max_velocity` (an optional physical ceiling — the cheapest
    reliable runaway detector). On failure the state is restored, `dt` is
    halved (never below `constraints.dt_min`) and the step is retried, up
    to `max_retries` times. If every retry fails the pre-step state is
    restored and an error is thrown.
 5. runs each fast process `substeps_for(schedule, name)` times with
    `dt/n`.

**SGS closure.** `sgs` is the closure the step runs (an
[`AbstractSGSClosure`](@ref Aeolus.Operators.AbstractSGSClosure) or a
function `f(νt, u, v, w)`; default: the built-in Smagorinsky when
`C_s ≠ 0`). It sets the diffusive limit of the `dt` pick, it is passed to
the default stepper (`step!` / `step_imex!`), and its memory is saved with
[`sgs_state`](@ref Aeolus.Operators.sgs_state) next to the prognostic
snapshot and restored with
[`restore_sgs_state!`](@ref Aeolus.Operators.restore_sgs_state!) whenever a
step is rejected — the commit ([`advance_sgs!`](@ref
Aeolus.Operators.advance_sgs!)) a rejected step made is undone, so after a
retry the closure's memory is that of a run that took the short step
directly. A custom `stepper` is called without `sgs` and must pass the same
closure to the step it takes itself. `sgs_stress` (`:laplacian` |
`:symmetric`) is passed to the default stepper with the closure, and to
[`adaptive_dt`](@ref) for the stress form's diffusive limit.

`fast_processes` accepts a `NamedTuple`, `Dict`, vector/tuple of
`:name => callback` pairs, or a bare callback (registered as
`:default`). Each callback is invoked as `callback(sim, dt_sub)`.

Every rejected step is a structured event (`subsystem = :timestepping`,
keys `reason`, `from`, `to`, `retry`; a throw also carries `exception`)
at `Debug`: a retry the driver recovers from is not a problem, so a run log
records it while the console stays quiet (install a logger at `Debug` to see
it there). A step that fails after `max_retries` throws; a fast process that
leaves non-finite values is a warning. See the Logging page (#333, #347).

The whole call is **one step** of the simulation's
[`StepRecord`](@ref Aeolus.Simulations.StepRecord) (#336), with its
retries, the CFL numbers of its dt pick and the limiter (`:retry` when
rejected attempts shortened it); the steps it takes inside are part of
it, and the simulation's diagnostics hook is called once, at the end.

```julia
c = CFLConstraints(dt_max = 10.0)
sched = SubsteppingSchedule(microphysics = 4)
dt = nothing
for n in 1:100
    _, dt, nretry = step_adaptive!(sim, c; dt_prev = dt, schedule = sched,
        fast_processes = (microphysics = (s, Δt) -> apply_kessler_step!(s, Δt),))
end
```
"""
function step_adaptive!(sim::BoussinesqSimulation, constraints::CFLConstraints; kwargs...)
    rec = sim.workspaces.rec
    outer = _step_begin!(rec, sim)
    res = try
        _step_adaptive!(sim, constraints; kwargs...)
    catch
        _step_abort!(rec); rethrow()
    end
    _step_end!(rec, sim, res[2], outer; retries = res[3])
    return res
end

# The driver of every core (ONE-j, #460): the Boussinesq one here, the
# sound-proof ones through their methods of `adaptive_dt`, the snapshot and
# `step_guard` (Anelastic).
function _step_adaptive!(sim, constraints::CFLConstraints;
                         dt = nothing,
                         dt_prev = nothing,
                         body_force!::Union{Nothing,Function} = nothing,
                         post_predictor!::Union{Nothing,Function} = nothing,
                         fast_processes = nothing,
                         schedule::Union{Nothing,SubsteppingSchedule} = nothing,
                         max_retries::Integer = 3,
                         stepper = nothing,
                         scheme::Union{Nothing,Any} = nothing,
                         snapshot::StateSnapshot = state_snapshot(sim),
                         divergence_tol::Real = 1e-6,
                         max_velocity::Real = Inf,
                         sgs = nothing,
                         sgs_stress::Symbol = :laplacian)
    T = eltype(sim.u.data)
    closure = sgs_closure(sgs)
    Δt = dt === nothing ? adaptive_dt(sim, constraints; dt_prev = dt_prev, scheme = scheme,
                                      sgs = closure, sgs_stress = sgs_stress) : T(dt)
    take_step! = if stepper !== nothing
        stepper
    elseif scheme isa IMEXScheme
        (s, h; kw...) -> step_imex!(s, h, scheme; kw..., sgs = closure, sgs_stress = sgs_stress)
    else
        (s, h; kw...) -> step!(s, h; kw..., sgs = closure, sgs_stress = sgs_stress)
    end

    rec = sim.workspaces.rec
    t_guard = time_ns()
    save_state!(snapshot, sim)
    # open-boundary buffers are state too: a retry must start from them
    bc_snap = open_boundary_snapshot(sim)
    # the closure's memory: the step commits it once (advance_sgs!); a
    # rejected step's commit is rolled back with the prognostic state
    sgs_snap = closure === nothing ? nothing : sgs_state(closure)
    rec.wall_guard += _wall(t_guard)
    div_tol = T(divergence_tol)
    u_cap   = T(max_velocity)
    n_retries = 0

    while true
        failed = false
        reason = :ok
        try
            take_step!(sim, Δt; body_force! = body_force!,
                                post_predictor! = post_predictor!)
        catch err
            err isa InterruptException && rethrow()
            failed = true
            reason = :exception
            # a throw the driver recovers from (by a retry): Debug, run.log only
            @debug("step_adaptive!: stepper threw",
                    subsystem = :timestepping, domain = current_domain(), dt = Δt,
                    retry = n_retries, exception = err)
        end
        if !failed
            # finiteness, divergence and speed cap in one fused sweep (KA-j, #306)
            t_guard = time_ns()
            reason = _guard_reason(sim, div_tol, u_cap)
            rec.wall_guard += _wall(t_guard)
            failed = reason !== :ok
        end

        failed || break

        if n_retries >= max_retries
            restore_state!(sim, snapshot)
            restore_open_boundaries!(sim, bc_snap)
            closure === nothing || restore_sgs_state!(closure, sgs_snap)
            error("step_adaptive!: step failed ($reason) after $max_retries " *
                  "retries; last dt = $Δt. State has been restored to the " *
                  "beginning of the step.")
        end
        t_guard = time_ns()
        restore_state!(sim, snapshot)
        restore_open_boundaries!(sim, bc_snap)
        closure === nothing || restore_sgs_state!(closure, sgs_snap)
        rec.wall_guard += _wall(t_guard)
        Δt_new = max(Δt / 2, constraints.dt_min)
        if Δt_new == Δt
            error("step_adaptive!: step failed ($reason) at dt_min = " *
                  "$(constraints.dt_min); cannot reduce dt further. State " *
                  "has been restored to the beginning of the step.")
        end
        # every retry is in run.log; the console shows it at Debug only
        @debug("step_adaptive!: retry",
                subsystem = :timestepping, domain = current_domain(), reason,
                from = Δt, to = Δt_new, retry = n_retries + 1, max_retries)
        Δt = Δt_new
        n_retries += 1
    end

    # --- fast processes, substepped --------------------------------------
    procs = _fast_process_pairs(fast_processes)
    if !isempty(procs)
        for (name, cb) in procs
            n = substeps_for(schedule, name)
            Δt_sub = Δt / n
            for _ in 1:n
                cb(sim, Δt_sub)
            end
        end
        _all_finite(sim) ||
            @warn("step_adaptive!: state became non-finite inside a fast process",
                  subsystem = :timestepping, domain = current_domain(), dt = Δt,
                  processes = Tuple(first.(procs)))
    end

    return (sim, Δt, n_retries)
end

# =========================================================================
# Phase 25 — variable-coefficient diffusion operator
# =========================================================================

# ν may be a scalar or a 3-D array. Arrays are accepted either at full
# storage size (halos included) or at interior size; indices are clamped,
# which is a zero-gradient extension of ν outside the array.
struct _Coef{T,A<:AbstractArray{T,3}}
    a::A
    ox::Int
    oy::Int
    oz::Int
end

@inline function _at(c::_Coef, i::Int, j::Int, k::Int)
    a = c.a
    @inbounds return a[clamp(i - c.ox, 1, size(a, 1)),
                       clamp(j - c.oy, 1, size(a, 2)),
                       clamp(k - c.oz, 1, size(a, 3))]
end

_coef(ν::Real, m::RegularGrid{T}) where {T} = T(ν)

function _coef(ν::AbstractArray{T,3}, m::RegularGrid{T}) where {T}
    if size(ν) == totalsize(m)
        return _Coef{T,typeof(ν)}(ν, 0, 0, 0)
    elseif size(ν) == (m.Nx, m.Ny, m.Nz)
        return _Coef{T,typeof(ν)}(ν, m.Hx, m.Hy, m.Hz)
    else
        throw(ArgumentError("ν_eff array has size $(size(ν)); expected " *
                            "$(totalsize(m)) (with halos) or " *
                            "$((m.Nx, m.Ny, m.Nz)) (interior only)"))
    end
end

_coef(c::_Coef, ::RegularGrid) = c

@inline _νf(ν::Real, i1, j1, k1, i2, j2, k2) = ν
@inline _νf(c::_Coef, i1, j1, k1, i2, j2, k2) =
    (_at(c, i1, j1, k1) + _at(c, i2, j2, k2)) / 2

# --- per-direction coefficients (P55-b, #175) ----------------------------
# A coefficient answers, for the unknown at (i, j, k), the six face values
# `_νxp … _νzm` the flux-form operator uses (x+, x−, y+, y−, z+, z−). The
# face between unknowns n and n+1 gets the same value from both sides, so
# every coefficient below gives a symmetric operator (SPD in I − Δt L).

# Scalar or one array: the 2-point average of the two adjacent cells — the
# Phase-25 operator, unchanged.
@inline _νxp(ν, i, j, k) = _νf(ν, i,   j, k, i+1, j, k)
@inline _νxm(ν, i, j, k) = _νf(ν, i-1, j, k, i,   j, k)
@inline _νyp(ν, i, j, k) = _νf(ν, i, j,   k, i, j+1, k)
@inline _νym(ν, i, j, k) = _νf(ν, i, j-1, k, i, j,   k)
@inline _νzp(ν, i, j, k) = _νf(ν, i, j, k,   i, j, k+1)
@inline _νzm(ν, i, j, k) = _νf(ν, i, j, k-1, i, j, k  )

# (νh, νv) for a cell-centred unknown: νh on the x and y faces, νv on the
# z faces, each the 2-point average.
struct _DirCoef{H,V}
    h::H
    v::V
end
@inline _νxp(c::_DirCoef, i, j, k) = _νxp(c.h, i, j, k)
@inline _νxm(c::_DirCoef, i, j, k) = _νxm(c.h, i, j, k)
@inline _νyp(c::_DirCoef, i, j, k) = _νyp(c.h, i, j, k)
@inline _νym(c::_DirCoef, i, j, k) = _νym(c.h, i, j, k)
@inline _νzp(c::_DirCoef, i, j, k) = _νzp(c.v, i, j, k)
@inline _νzm(c::_DirCoef, i, j, k) = _νzm(c.v, i, j, k)

# (νh, νv) for a staggered velocity component at location L: the
# coefficients of the diagonal (Laplacian) part ∂ⱼ(νᵢⱼ ∂ⱼuᵢ) of the stress
# form, `add_sgs_stress_divergence!(…; part = :diagonal)`, with the same
# interpolation — the cell value for the normal flux, the 4-cell edge
# average for the tangential ones, νh for the pairs (11, 22, 12) and νv
# for the pairs involving z (33, 13, 23; the CM1 convention).
struct _StressCoef{L<:Location,H<:_Coef,V<:_Coef}
    h::H
    v::V
end
_StressCoef(::L, h::_Coef, v::_Coef) where {L<:Location} =
    _StressCoef{L,typeof(h),typeof(v)}(h, v)
@inline _e4(c::_Coef, i1, j1, k1, i2, j2, k2, i3, j3, k3, i4, j4, k4) =
    0.25 * (_at(c, i1, j1, k1) + _at(c, i2, j2, k2) + _at(c, i3, j3, k3) + _at(c, i4, j4, k4))
# the edge averages of FiniteVolume._νe12/_νe13/_νe23 (same order of terms)
@inline _e12(c, i, j, k) = _e4(c, i-1, j-1, k, i, j-1, k, i-1, j, k, i, j, k)
@inline _e13(c, i, j, k) = _e4(c, i-1, j, k-1, i, j, k-1, i-1, j, k, i, j, k)
@inline _e23(c, i, j, k) = _e4(c, i, j-1, k-1, i, j, k-1, i, j-1, k, i, j, k)

const _SCX = _StressCoef{FaceX}; const _SCY = _StressCoef{FaceY}; const _SCZ = _StressCoef{FaceZ}
# u @ FaceX: x faces are the cells i-1, i; y faces xy-edges; z faces xz-edges
@inline _νxp(c::_SCX, i, j, k) = _at(c.h, i, j, k)
@inline _νxm(c::_SCX, i, j, k) = _at(c.h, i-1, j, k)
@inline _νyp(c::_SCX, i, j, k) = _e12(c.h, i, j+1, k)
@inline _νym(c::_SCX, i, j, k) = _e12(c.h, i, j, k)
@inline _νzp(c::_SCX, i, j, k) = _e13(c.v, i, j, k+1)
@inline _νzm(c::_SCX, i, j, k) = _e13(c.v, i, j, k)
# v @ FaceY
@inline _νxp(c::_SCY, i, j, k) = _e12(c.h, i+1, j, k)
@inline _νxm(c::_SCY, i, j, k) = _e12(c.h, i, j, k)
@inline _νyp(c::_SCY, i, j, k) = _at(c.h, i, j, k)
@inline _νym(c::_SCY, i, j, k) = _at(c.h, i, j-1, k)
@inline _νzp(c::_SCY, i, j, k) = _e23(c.v, i, j, k+1)
@inline _νzm(c::_SCY, i, j, k) = _e23(c.v, i, j, k)
# w @ FaceZ
@inline _νxp(c::_SCZ, i, j, k) = _e13(c.v, i+1, j, k)
@inline _νxm(c::_SCZ, i, j, k) = _e13(c.v, i, j, k)
@inline _νyp(c::_SCZ, i, j, k) = _e23(c.v, i, j+1, k)
@inline _νym(c::_SCZ, i, j, k) = _e23(c.v, i, j, k)
@inline _νzp(c::_SCZ, i, j, k) = _at(c.v, i, j, k)
@inline _νzm(c::_SCZ, i, j, k) = _at(c.v, i, j, k-1)

# (Kxx, Kyy, Kzz) for a cell-centred unknown: the diagonal of a diffusivity
# tensor (P55-c, #181), each on the faces of its own direction with the
# 2-point average — exactly the `part = :diagonal` of
# `add_tensor_diffusion_scalar!`.
struct _DirCoef3{X,Y,Z}
    x::X
    y::Y
    z::Z
end
@inline _νxp(c::_DirCoef3, i, j, k) = _νxp(c.x, i, j, k)
@inline _νxm(c::_DirCoef3, i, j, k) = _νxm(c.x, i, j, k)
@inline _νyp(c::_DirCoef3, i, j, k) = _νyp(c.y, i, j, k)
@inline _νym(c::_DirCoef3, i, j, k) = _νym(c.y, i, j, k)
@inline _νzp(c::_DirCoef3, i, j, k) = _νzp(c.z, i, j, k)
@inline _νzm(c::_DirCoef3, i, j, k) = _νzm(c.z, i, j, k)

_coef(c::Union{_DirCoef,_DirCoef3,_StressCoef}, ::RegularGrid) = c
# a pair (νh, νv): νh on the x and y faces, νv on the z faces
_coef(ν::Tuple{Any,Any}, m::RegularGrid) = _DirCoef(_coef(ν[1], m), _coef(ν[2], m))

# ∇·(ν ∇f) at a single cell, face-averaged ν (symmetric ⇒ SPD operator).
@inline function _diff_at(d, ν, i, j, k, idx2, idy2, idz2)
    @inbounds begin
        c   = d[i, j, k]
        νxp = _νxp(ν, i, j, k); νxm = _νxm(ν, i, j, k)
        νyp = _νyp(ν, i, j, k); νym = _νym(ν, i, j, k)
        νzp = _νzp(ν, i, j, k); νzm = _νzm(ν, i, j, k)
        return (νxp * (d[i+1, j, k] - c) - νxm * (c - d[i-1, j, k])) * idx2 +
               (νyp * (d[i, j+1, k] - c) - νym * (c - d[i, j-1, k])) * idy2 +
               (νzp * (d[i, j, k+1] - c) - νzm * (c - d[i, j, k-1])) * idz2
    end
end

# Diagonal of (I - Δt ∇·(ν∇)) — the Jacobi preconditioner.
@inline function _diag_at(ν, i, j, k, idx2, idy2, idz2, Δt)
    s = (_νxp(ν, i, j, k) + _νxm(ν, i, j, k)) * idx2 +
        (_νyp(ν, i, j, k) + _νym(ν, i, j, k)) * idy2 +
        (_νzp(ν, i, j, k) + _νzm(ν, i, j, k)) * idz2
    return one(Δt) + Δt * s
end

"""
    _diffusion!(out, f, ν, mesh)

Write `∇·(ν ∇f)` into `out`'s interior. `ν` is a scalar or a per-cell
array (see [`solve_implicit_diffusion!`](@ref)); halos of `f` must already
be filled. This is *exactly* the operator the implicit solve inverts, so
the IMEX split is consistent by construction.
"""
_diffusion!(out::Field{L,T}, f::Field{L,T}, ν_eff, m::RegularGrid{T}) where {L,T} =
    _diffusion_kernel!(out, f, _coef(ν_eff, m), m)

@inline function _diffusion_body!(i, j, k, o, d, ν, idx2, idy2, idz2)
    @inbounds o[i, j, k] = _diff_at(d, ν, i, j, k, idx2, idy2, idz2)
    return nothing
end

# a pointwise launch over the interior (KA-g, #300)
function _diffusion_kernel!(out::Field{L,T}, f::Field{L,T}, ν, m::RegularGrid{T}) where {L,T}
    I3 = interior(m)
    checkbounds(out.data, I3...); checkbounds(f.data, I3...)
    idx2 = 1 / m.dx^2; idy2 = 1 / m.dy^2; idz2 = 1 / m.dz^2
    Backends.launch!(Backends.backend(out.data), _diffusion_body!, I3, out.data, f.data, ν,
                     idx2, idy2, idz2)
    return out
end

# Rows that are *not* unknowns: `w` on an impermeable wall face is pinned
# to zero by the BC, and the normal velocity on a west/south open face
# (N-open-b) to its boundary value, so they are removed from the CG system
# (keeping the reduced operator SPD) instead of being solved for. Free-slip
# and no-slip walls (`is_wall`) pin `w` alike; the top, east and north
# faces lie in the halo. `_constraint_planes` gives `(ic, jc, kc)`, 0 =
# none; `_constraint_index` is its z entry (the wall row), as before.
_constraint_planes(loc::Location, m::RegularGrid, bcs::FieldBoundaryConditions) =
    pinned_face_indices(loc, m, bcs)
_constraint_index(loc::Location, m::RegularGrid, bcs::FieldBoundaryConditions) =
    _constraint_planes(loc, m, bcs)[3]


function _apply_constraint!(a::AbstractArray{T,3}, c::NTuple{3,Int}) where {T}
    @inbounds begin
        c[1] == 0 || (a[c[1], :, :] .= zero(T))
        c[2] == 0 || (a[:, c[2], :] .= zero(T))
        c[3] == 0 || (a[:, :, c[3]] .= zero(T))
    end
    return a
end

function _assert_supported_bcs(bcs::FieldBoundaryConditions)
    for bc in (bcs.west, bcs.east, bcs.south, bcs.north, bcs.bottom, bcs.top)
        # a FluxBC top (#254) is handled like a moving wall: its affine part
        # goes to the right-hand side (see _cg_implicit!)
        bc isa FluxBC && bc === bcs.top && continue
        if bc isa ValueBC || bc isa FluxBC
            throw(ArgumentError(
                "solve_implicit_diffusion! supports the BC types handled by fill_halos! " *
                "(Periodic, NoFlux, ImpermeableWall, NoSlipWall, OpenBoundary, " *
                "FreeStreamBoundary, and a FluxBC top). " *
                "$(typeof(bc)) makes the operator affine; move the " *
                "inhomogeneous part into `rhs` and pass a homogeneous BC."))
        end
    end
    return nothing
end

# =========================================================================
# Phase 25 — the implicit-diffusion CG solver
# =========================================================================

"""
    DiffusionWorkspace(field)

Preallocated scratch for [`solve_implicit_diffusion!`](@ref): four arrays
the size of `field`'s storage (search direction, residual, operator
image, preconditioned residual) and the reduction partials of the CG
driver ([`Aeolus.Backends.PCGWorkspace`](@ref)). Reuse one per
`(location, mesh)` pair to keep the solve allocation-free. The diagonal
is recomputed on the fly rather than stored (a handful of flops per cell,
far cheaper than the matvec, and one array less).
"""
struct DiffusionWorkspace{T,F<:Field}
    p_field::F
    r::Array{T,3}
    Ap::Array{T,3}
    z::Array{T,3}
    pcg::Backends.PCGWorkspace{T,Array{T,3}}
end

function DiffusionWorkspace(f::Field{L,T}) where {L,T}
    pf = Field(L(), f.mesh)
    sz = size(f.data)
    r, Ap, z = zeros(T, sz), zeros(T, sz), zeros(T, sz)
    pcg = Backends.PCGWorkspace(r, z, pf.data, Ap; ranges = interior(f.mesh))
    return DiffusionWorkspace{T,typeof(pf)}(pf, r, Ap, z, pcg)
end

# (I − Δt ∇·(ν∇)) d at one point: the operator of the implicit solve, a
# `PointOperator` body (KA-g, #300). The driver zeroes the pinned rows.
@inline _implicit_at(i, j, k, d, ν, Δt, idx2, idy2, idz2) =
    @inbounds d[i, j, k] - Δt * _diff_at(d, ν, i, j, k, idx2, idy2, idz2)

# The Jacobi step z = r / diag(I − Δt ∇·(ν∇)).
@inline _implicit_jacobi_at(rv, i, j, k, ν, idx2, idy2, idz2, Δt) =
    rv / _diag_at(ν, i, j, k, idx2, idy2, idz2, Δt)

# The halo fill before each operator application: of the field itself
# (the initial residual of a nonzero guess) or of the search direction.
struct _ImplicitHalos{F,P,B,K}
    field::F
    pf::P
    bcs::B
    key::K
end
(h::_ImplicitHalos)(a) = fill_halos!(a === h.field.data ? h.field : h.pf, h.bcs; key = h.key)

@inline function _wall_rhs_body!(i, j, k, r, b, p, ν, Δt, idx2, idy2, idz2)
    @inbounds r[i, j, k] = b[i, j, k] + Δt * _diff_at(p, ν, i, j, k, idx2, idy2, idz2)
    return nothing
end

"""
    solve_implicit_diffusion!(field, rhs, ν_eff, dt, mesh, bcs;
                              tol = 1e-10, maxit = 200,
                              workspace = DiffusionWorkspace(field),
                              precondition = :jacobi, warn = true)
    -> (; iterations, residual, relative_residual, converged, tol)

Solve `(I − dt ∇·(ν_eff ∇)) φ = rhs` for `φ` (written into `field`),
matrix-free, with Jacobi-preconditioned conjugate gradients.

  * `ν_eff` is either a scalar or an `Array{T,3}` — per-cell eddy
    viscosity, given at full storage size (halos included) or at interior
    size. Arrays use the face average `ν_{i+1/2} = ½(νᵢ + ν_{i+1})`,
    which keeps the operator symmetric, hence CG-legal (`I − dt L` with
    `L` symmetric negative-semidefinite is SPD for `dt ≥ 0`). A pair
    `(ν_h, ν_v)` of such coefficients is a per-direction coefficient:
    `ν_h` on the x and y faces, `ν_v` on the z faces (P55-b, #175).
  * Boundary conditions are applied by calling `fill_halos!(·, bcs)` on
    the search direction inside every matrix-vector product, so exactly
    the BC types `fill_halos!` supports are supported here: `Periodic`,
    `NoFlux`, `ImpermeableWall` and `NoSlipWall`. `ValueBC`/`FluxBC` make
    the operator affine and are rejected with an explanatory error. For
    `w` on an impermeable wall the pinned wall-face row is eliminated from
    the system.
  * A *moving* `NoSlipWall` (`u_wall` or `v_wall ≠ 0`) also makes the
    operator affine, `∇·(ν∇φ) = L₀φ + g`; this one is handled: the CG
    iterates with the stationary wall ([`homogeneous_bcs`](@ref
    Aeolus.BoundaryConditions.homogeneous_bcs)) and the wall term
    `dt·g` is added to the right-hand side, so `φ` is the solution with
    the wall moving. Pass `homogeneous_bcs(bcs)` yourself when `φ` is an
    *increment* (a Rosenbrock stage), whose wall value is zero.
  * An [`OpenBoundary`](@ref Aeolus.BoundaryConditions.OpenBoundary)
    (N-open-b) is affine in the same way and handled the same way: the CG
    iterates with zero boundary values and the Dirichlet lift (the
    boundary data in the halos and in the pinned west/south face rows) is
    carried in the right-hand side; after the solve the field's halos and
    boundary faces are refilled with the data. `key` names the field whose
    boundary data apply (default `field.data`; a Rosenbrock increment
    passes the prognostic field's array).
  * `field` doubles as the initial guess, and `rhs === field` is
    explicitly allowed (build the right-hand side in place, then solve
    over it — that is what [`step_imex!`](@ref) does).
  * `ν_eff == 0` or `dt == 0` short-circuits to a copy.

Convergence is `‖r‖₂ ≤ tol·‖b‖₂` over the interior. The returned
`NamedTuple` reports the iteration count and the final (absolute and
relative) residual; `converged = false` plus a warning if `maxit` is hit.

Preconditioning is diagonal (Jacobi), which is enough to make the
iteration count nearly independent of `dt·ν/Δ²` in practice for the
grid-aligned anisotropies LES actually has. Geometric-multigrid
preconditioning — the right answer for very stiff `dt ν / Δz²` on tall
stretched grids — is the Phase-29 follow-up (`src/Multigrid.jl`); this
solver is deliberately written so that only `z ← M⁻¹r` needs replacing.
"""
function solve_implicit_diffusion!(field::Field{L,T}, rhs::Field{L,T}, ν_eff,
                                   dt::Real, m::RegularGrid{T},
                                   bcs::FieldBoundaryConditions;
                                   tol::Real = 1e-10, maxit::Integer = 200,
                                   workspace::DiffusionWorkspace = DiffusionWorkspace(field),
                                   precondition::Symbol = :jacobi,
                                   warn::Bool = true, key = field.data) where {L,T}
    _assert_supported_bcs(bcs)
    precondition in (:jacobi, :none) ||
        throw(ArgumentError("precondition must be :jacobi or :none, got :$precondition"))
    Δt = T(dt)
    Δt >= 0 || throw(ArgumentError("dt must be ≥ 0"))
    ν = _coef(ν_eff, m)
    ii, jj, kk = interior(m)
    τ = T(tol)

    if ν isa T && (Δt == 0 || ν == 0)
        field !== rhs && interior_map!(identity, field, rhs)
        return (iterations = 0, residual = zero(T), relative_residual = zero(T),
                converged = true, tol = τ)
    end

    return _cg_implicit!(field, rhs, ν, Δt, m, bcs, workspace, τ, Int(maxit),
                         precondition, warn, key)
end

# Function barrier: specializes on the concrete type of ν. The iteration
# is the one CG driver `Backends.pcg!` (KA-g, #300).
function _cg_implicit!(field::Field{L,T}, rhs::Field{L,T}, ν, Δt::T,
                       m::RegularGrid{T}, bcs_full::FieldBoundaryConditions,
                       ws::DiffusionWorkspace, τ::T, maxit::Int,
                       precondition::Symbol, warn::Bool, key = field.data) where {L,T}
    I3 = interior(m)
    idx2 = 1 / m.dx^2; idy2 = 1 / m.dy^2; idz2 = 1 / m.dz^2
    # A moving no-slip wall makes the operator affine: iterate with the
    # stationary wall and carry the wall term in the right-hand side.
    bcs = homogeneous_bcs(bcs_full)
    affine = bcs !== bcs_full
    kc = _constraint_planes(L(), m, bcs)
    x  = field.data
    pf = ws.p_field
    bk = Backends.backend(x)

    _apply_constraint!(x, kc)

    # Wall term g = ∇·(ν∇0) with the moving-wall halo: b_eff = b + Δt g,
    # into r (the driver reads it as b before it writes r there).
    b = rhs.data
    if affine
        fill!(pf.data, zero(T))
        fill_halos!(pf, bcs_full; key = key)
        Backends.launch!(bk, _wall_rhs_body!, I3, ws.r, b, pf.data, ν, Δt, idx2, idy2, idz2)
        b = ws.r
    end

    A = Backends.PointOperator(_implicit_at, ν, Δt, idx2, idy2, idz2;
                               prepare = _ImplicitHalos(field, pf, bcs, key))
    # one call per preconditioner: a `Union` of the two would box M (80 B per solve)
    st = if precondition === :jacobi
        Backends.pcg!(A, Backends.PointPreconditioner(_implicit_jacobi_at, ν, idx2, idy2, idz2, Δt),
                      x, b, ws.pcg; ranges = I3, tol = τ, maxit = maxit, constraint = kc,
                      zero_guess = false, backend = bk)
    else
        Backends.pcg!(A, Backends.IdentityPreconditioner(), x, b, ws.pcg; ranges = I3, tol = τ,
                      maxit = maxit, constraint = kc, zero_guess = false, backend = bk)
    end

    if st.status === :zero_rhs
        has_open_boundaries(bcs_full) && fill_halos!(field, bcs_full; key = key)
        return (iterations = 0, residual = zero(T), relative_residual = zero(T),
                converged = true, tol = τ)
    end
    iters = st.iterations
    rnorm, bnorm = st.residual, st.bnorm
    if st.status === :breakdown
        iters += 1                     # the iteration that found it, as before
        warn && @warn("solve_implicit_diffusion!: CG breakdown (pᵀAp = $(st.curvature)); " *
                      "operator is not positive definite for these BCs",
                      subsystem = :diffusion, domain = current_domain(),
                      reason = :breakdown, iterations = iters, curvature = st.curvature)
    elseif st.status === :maxit
        warn && @warn("solve_implicit_diffusion! hit maxit=$maxit with " *
                      "‖r‖/‖b‖ = $(rnorm / bnorm)",
                      subsystem = :diffusion, domain = current_domain(),
                      reason = :maxit, iterations = iters,
                      relative_residual = rnorm / bnorm, tol = τ)
    end
    # open faces: the pinned face rows and the halos carry the boundary data
    has_open_boundaries(bcs_full) && fill_halos!(field, bcs_full; key = key)
    return (iterations = iters, residual = rnorm,
            relative_residual = rnorm / bnorm, converged = st.status === :converged, tol = τ)
end

# =========================================================================
# Phase 25 — IMEX schemes
# =========================================================================

"""
    IMEXScheme

Additive / semi-implicit time-integration schemes. Concrete types:
[`ARK2`](@ref) (additive Runge–Kutta, ARS(2,2,2)) and
[`Rosenbrock2`](@ref) (2-stage Rosenbrock-W). Drive with
[`step_imex!`](@ref).
"""
abstract type IMEXScheme end

# --- ARS(2,2,2) -----------------------------------------------------------
# Ascher, Ruuth & Spiteri (1997), scheme (2,2,2). γ makes the implicit
# part L-stable and 2nd order (γ² - 2γ + ½ = 0 ⇒ γ = 1 - 1/√2);
# δ = 1 - 1/(2γ) makes the explicit part 2nd order.
const ARK2_γ = 1 - 1 / sqrt(2)
const ARK2_δ = 1 - 1 / (2 * ARK2_γ)
"Explicit weights `(c₂ stage, δ, 1-δ)` of ARS(2,2,2)."
const ARK2_EXPLICIT_WEIGHTS = (ARK2_γ, ARK2_δ, 1 - ARK2_δ)
"Implicit weights `(γ, 1-γ, γ)` of ARS(2,2,2)."
const ARK2_IMPLICIT_WEIGHTS = (ARK2_γ, 1 - ARK2_γ, ARK2_γ)

"""
    ARK2([T = Float64])

2nd-order, 2-stage additive Runge–Kutta (ARS(2,2,2), Ascher–Ruuth–Spiteri
1997) with

```
U₁    = uⁿ + γΔt E(uⁿ) + γΔt I(U₁)
uⁿ⁺¹  = uⁿ + Δt[δ E(uⁿ) + (1-δ) E(U₁)] + Δt[(1-γ) I(U₁) + γ I(uⁿ⁺¹)]
```

with `γ = 1 - 1/√2`, `δ = 1 - 1/(2γ)`. The implicit tableau is
L-stable (`R(∞) = 0`), so the stiff diffusion is damped rather than
oscillating, and both halves are 2nd-order accurate. Two implicit solves
and two pressure projections per step.
"""
struct ARK2{T<:AbstractFloat} <: IMEXScheme
    γ::T
    δ::T
    a_explicit::NTuple{3,T}
    a_implicit::NTuple{3,T}
end

ARK2(::Type{T} = Float64) where {T<:AbstractFloat} =
    ARK2{T}(T(ARK2_γ), T(ARK2_δ),
            map(T, ARK2_EXPLICIT_WEIGHTS), map(T, ARK2_IMPLICIT_WEIGHTS))

# --- ROS2 ----------------------------------------------------------------
# Verwer et al. (1999). γ = 1 + 1/√2 solves γ² - 2γ + ½ = 0, giving
# R(∞) = 0 (L-stable) and 2nd order.
const ROS2_γ = 1 + 1 / sqrt(2)
"Stage coupling `(a₂₁, c₂₁)` of ROS2."
const ROS2_STAGE_COEFFS = (1.0, -2.0)
"Update weights `(b₁, b₂)` of ROS2."
const ROS2_WEIGHTS = (1.5, 0.5)

"""
    Rosenbrock2([T = Float64])

2-stage, 2nd-order Rosenbrock-W method (ROS2, Verwer et al. 1999):

```
(I - γΔt J) k₁ = Δt F(uⁿ)
(I - γΔt J) k₂ = Δt F(uⁿ + k₁) - 2k₁
uⁿ⁺¹           = uⁿ + 3/2 k₁ + 1/2 k₂
```

with `γ = 1 + 1/√2`. It is a *W*-method: only the diffusion part of the
Jacobian is used (`J ≈ ∇·(ν_eff∇)`), which costs nothing extra to form
and does not degrade the order — advection and buoyancy simply behave
explicitly. Same cost as [`ARK2`](@ref): two solves, two projections.
"""
struct Rosenbrock2{T<:AbstractFloat} <: IMEXScheme
    γ::T
    a21::T
    c21::T
    b::NTuple{2,T}
end

Rosenbrock2(::Type{T} = Float64) where {T<:AbstractFloat} =
    Rosenbrock2{T}(T(ROS2_γ), T(ROS2_STAGE_COEFFS[1]), T(ROS2_STAGE_COEFFS[2]),
                   map(T, ROS2_WEIGHTS))

"Formal temporal order of accuracy."
order(::ARK2) = 2
order(::Rosenbrock2) = 2

"Number of implicit stages (= implicit solves and projections per step)."
stages(::ARK2) = 2
stages(::Rosenbrock2) = 2

"""
    explicit_cfl_factor(scheme)

Fraction of the nominal advective CFL limit the *explicit* half of
`scheme` can use. Both shipped schemes have a 2-stage explicit tableau
whose stability region along the imaginary axis is comparable to
forward-Euler-with-safety, so this is `1` and the limit is whatever
`CFLConstraints.C_advective` says.
"""
explicit_cfl_factor(::IMEXScheme) = 1.0

# =========================================================================
# Phase 25 — IMEX workspace
# =========================================================================

"""
    IMEXWorkspace(sim)

Scratch state for [`step_imex!`](@ref): the step-start state, one
tendency/stage set, a lazily allocated second stage set (Rosenbrock
only), one `DiffusionWorkspace` per staggered location (`θ` and `qv`
share the cell-centered one), plus the frozen effective-viscosity arrays
when SGS is on.

Memory is roughly 26 field-sized arrays for [`ARK2`](@ref) and 31 for
[`Rosenbrock2`](@ref) — comparable to what `BoussinesqSimulation` itself
holds. It also records the last step's solver statistics in
`total_iterations` / `max_residual`.
"""
mutable struct IMEXWorkspace{T,S<:NamedTuple,D<:NamedTuple}
    state0::S
    aux::S                        # E(uⁿ) for ARK2 / k₁ for Rosenbrock2
    k2::Union{Nothing,S}
    scratch::Field{CellCenter,T,Array{T,3},RegularGrid{T}}
    diff::D
    ν_mom::Union{Nothing,Array{T,3}}
    ν_scal::Union{Nothing,Array{T,3}}
    ν_mom_v::Union{Nothing,Array{T,3}}    # vertical ν_eff (sgs_stress = :symmetric)
    ν_scal_v::Union{Nothing,Array{T,3}}   # vertical κ_eff (a closure with κv ≢ κh)
    ν_scal_y::Union{Nothing,Array{T,3}}   # y κ_eff (a closure with a scalar tensor)
    ν_qv::Union{Nothing,Array{T,3}}       # qv's own frozen coefficient (AMD)
    total_iterations::Int
    max_residual::T
end

function IMEXWorkspace(sim::BoussinesqSimulation{T}) where {T}
    m = sim.mesh
    mkset() = (u  = Field(FaceX(), m), v = Field(FaceY(), m), w = Field(FaceZ(), m),
               θ  = Field(CellCenter(), m), qv = Field(CellCenter(), m))
    state0 = mkset()
    aux    = mkset()
    diff   = (u = DiffusionWorkspace(sim.u), v = DiffusionWorkspace(sim.v),
              w = DiffusionWorkspace(sim.w), c = DiffusionWorkspace(sim.θ))
    return IMEXWorkspace{T,typeof(state0),typeof(diff)}(
        state0, aux, nothing, Field(CellCenter(), m), diff, nothing, nothing,
        nothing, nothing, nothing, nothing, 0, zero(T))
end

"""
    imex_workspace(sim) -> IMEXWorkspace

The [`IMEXWorkspace`](@ref) of `sim`: allocated on first use and kept in
the simulation's [`StepWorkspaces`](@ref Aeolus.Simulations.StepWorkspaces)
(KA-j, #306; before, a process-global weak cache). Pass it explicitly to
`step_imex!` if you want to control the lifetime (or read the solver
statistics it records).
"""
function imex_workspace(sim::BoussinesqSimulation)
    # the implicit diffusion operator and its CG solve are uniform-grid
    # stencils; a stretched mesh (#151) runs the explicit drivers
    sim.mesh isa RegularGrid ||
        throw(ArgumentError("step_imex! (implicit diffusion) is implemented on a RegularGrid " *
                            "only, not on a $(nameof(typeof(sim.mesh))); use step!, " *
                            "step_rk3! or step_adaptive!"))
    w = sim.workspaces
    w.imex === nothing && (w.imex = IMEXWorkspace(sim))
    return w.imex::IMEXWorkspace
end

function _k2_set!(ws::IMEXWorkspace{T,S}) where {T,S}
    ws.k2 === nothing && (ws.k2 = map(f -> Field(location(f), f.mesh), ws.state0))
    return ws.k2::S
end

# =========================================================================
# Phase 25 — explicit tendencies, diffusion, projection
# =========================================================================

# Everything except diffusion: advection, buoyancy, Coriolis, prescribed
# surface flux, sponge and the user body force. Mirrors Simulations.step!
# with the viscous/SGS terms removed — those are the implicit half.
#
# `sgs_transpose = (νh, νv)` (arrays, halos filled) adds the explicit half of
# the stress-form SGS split (`sgs_stress = :symmetric`): the transposed part
# ∂ⱼ(νᵢⱼ ∂uⱼ/∂xᵢ), whose diagonal partner ∂ⱼ(νᵢⱼ ∂uᵢ/∂xⱼ) is implicit.
#
# `scalar_tensor` (a DiffusivityTensor, halos filled) adds the explicit half
# of a tensorial scalar diffusivity (P55-c): its off-diagonal (cross) part on
# θ and qv, whose diagonal partner is implicit.
function _explicit_tendencies!(sim::BoussinesqSimulation{T}, Δt;
                               body_force!::Union{Nothing,Function} = nothing,
                               momentum_advection = momentum_advect!,
                               sgs_transpose = nothing,
                               scalar_tensor = nothing) where {T}
    fill_state_halos!(sim)

    fill!(sim.tu.data, 0); fill!(sim.tv.data, 0); fill!(sim.tw.data, 0)
    fill!(sim.tθ.data, 0)
    sim.has_moisture && fill!(sim.tqv.data, 0)

    momentum_advection(sim.tu, sim.tv, sim.tw, sim.u, sim.v, sim.w)
    advect_centered!(sim.tθ, sim.u, sim.v, sim.w, sim.θ)
    sim.has_moisture && advect_centered!(sim.tqv, sim.u, sim.v, sim.w, sim.qv)
    sgs_transpose === nothing ||
        add_sgs_stress_divergence!(sim.tu, sim.tv, sim.tw, sim.u, sim.v, sim.w,
                                   sgs_transpose[1], sgs_transpose[2]; part = :transpose)
    if scalar_tensor !== nothing
        bnd = bounded_directions(sim.θ_bcs)
        add_tensor_diffusion_scalar!(sim.tθ, sim.θ, scalar_tensor; bounded = bnd, part = :offdiagonal)
        sim.has_moisture &&
            add_tensor_diffusion_scalar!(sim.tqv, sim.qv, scalar_tensor; bounded = bnd, part = :offdiagonal)
    end

    if sim.has_buoyancy
        if sim.has_moisture
            add_buoyancy!(sim.tw, sim.θ, sim.qv, sim.θ₀, sim.g_accel)
        else
            add_buoyancy!(sim.tw, sim.θ, sim.θ₀, sim.g_accel)
        end
    end
    sim.has_coriolis && add_coriolis!(sim.tu, sim.tv, sim.u, sim.v, sim.f_coriolis)
    sim.has_surface_flux && add_surface_flux!(sim.tθ, sim.θ_flux_bottom, sim.θ_flux_top)
    sim.has_sponge && Simulations.add_velocity_sponge!(sim)   # toward sim.sponge_target (#262)

    body_force! === nothing || body_force!(sim, Δt)
    return sim
end

# Freeze ν_eff = ν + νt (momentum) and ν + κt (scalars) at the start of
# the step — the standard lagged-coefficient linearization; the implicit
# operator stays linear and SPD. νt, κt are the SGS closure's coefficients
# (`compute_sgs!`: the built-in Smagorinsky, κt = νt/Prₜ, or the `sgs`
# keyword, the same rule as `step!`).
#
# Momentum, `stress = :laplacian`: one isotropic coefficient ν + νh with the
# Phase-25 co-located 2-point face average for all three components (an
# anisotropic closure contributes its horizontal νh). `stress = :symmetric`
# (P55-b): per component, the coefficients of the diagonal part of the
# stress form (`_StressCoef`: νh on the 11/22/12 faces, νv on 33/13/23,
# with the explicit kernel's edge interpolation), so the implicit operator is
# exactly the Laplacian part of the explicit stress divergence; the
# transposed part is returned as `sgs_transpose` for the explicit side.
# Scalars: ν + κh, or the per-direction pair (ν + κh, ν + κv) when the
# closure has a separate κv, or — for a closure with a scalar diffusivity
# tensor (`sgs_scalar_tensor`, P55-c) — the per-direction diagonal
# (ν + K11, ν + K22, ν + K33), with the off-diagonal part returned as
# `scalar_tensor` for the explicit side.
#
# Returns (νm, νs, sgs_transpose, scalar_tensor): νm a coefficient or a
# NamedTuple (u, v, w).
function _freeze_viscosity!(sim::BoussinesqSimulation{T}, ws::IMEXWorkspace{T},
                            closure, stress::Symbol = :laplacian) where {T}
    closure === nothing && return (T(sim.ν), T(sim.ν), nothing, nothing)
    sgs_explicit_stress(closure) &&
        throw(ArgumentError("step_imex!: the SGS closure $(nameof(typeof(closure))) has its own " *
                            "momentum stress (sgs_explicit_stress), which the implicit eddy-viscosity " *
                            "operator cannot represent; use step! or step_rk3!"))
    fill_state_halos!(sim)                             # θ-aware closures
    t_sgs = time_ns()
    f = compute_sgs!(closure, sim)
    sim.workspaces.rec.wall_sgs += _wall(t_sgs)
    ws.ν_mom  === nothing && (ws.ν_mom  = similar(sim.νt.data))
    ws.ν_scal === nothing && (ws.ν_scal = similar(sim.νt.data))
    νm = ws.ν_mom::Array{T,3}; νs = ws.ν_scal::Array{T,3}
    _plus_ν!(νm, sim.ν, f.νh.data)
    _plus_ν!(νs, sim.ν, f.κh.data)
    m = sim.mesh
    Ks = sgs_scalar_tensor(closure, f, sim)
    cs = if Ks !== nothing
        ws.ν_scal_v === nothing && (ws.ν_scal_v = similar(sim.νt.data))
        ws.ν_scal_y === nothing && (ws.ν_scal_y = similar(sim.νt.data))
        νsy = ws.ν_scal_y::Array{T,3}; νsz = ws.ν_scal_v::Array{T,3}
        _plus_ν!(νs, sim.ν, Ks.K11.data)
        _plus_ν!(νsy, sim.ν, Ks.K22.data)
        _plus_ν!(νsz, sim.ν, Ks.K33.data)
        _DirCoef3(_coef(νs, m), _coef(νsy, m), _coef(νsz, m))
    elseif f.κv === f.κh
        _coef(νs, m)
    else
        ws.ν_scal_v === nothing && (ws.ν_scal_v = similar(sim.νt.data))
        νsv = ws.ν_scal_v::Array{T,3}
        _plus_ν!(νsv, sim.ν, f.κv.data)
        _DirCoef(_coef(νs, m), _coef(νsv, m))
    end
    if sim.has_moisture
        # qv: the closure's diffusivity for a scalar other than θ — f.κh for a
        # Prandtl-number closure (shared coefficient, so qv shares `cs`,
        # including a separate κv or a scalar tensor, P55-c), its own
        # isotropic one for AMD.
        κq = sgs_tracer_diffusivity!(ws.scratch, closure, f, sim, sim.qv)
        if κq !== f.κh
            ws.ν_qv === nothing && (ws.ν_qv = similar(sim.νt.data))
            νq = ws.ν_qv::Array{T,3}
            _plus_ν!(νq, sim.ν, κq.data)
            cs = (θ = cs, qv = _coef(νq, m))
        end
    end
    stress === :symmetric || return (_coef(νm, m), cs, nothing, Ks)
    νmv = if f.νv === f.νh
        νm
    else
        ws.ν_mom_v === nothing && (ws.ν_mom_v = similar(sim.νt.data))
        a = ws.ν_mom_v::Array{T,3}
        _plus_ν!(a, sim.ν, f.νv.data)
        a
    end
    h = _coef(νm, m); v = _coef(νmv, m)
    cm = (u = _StressCoef(FaceX(), h, v), v = _StressCoef(FaceY(), h, v),
          w = _StressCoef(FaceZ(), h, v))
    return (cm, cs, (νm, νmv), Ks)
end

# out = ν + K over the whole storage (a pointwise kernel; the bits of the
# broadcast `out .= ν .+ K`)
_plus_ν!(out::AbstractArray{T,3}, ν, K::AbstractArray{T,3}) where {T} =
    Backends.range_map!(Backends.backend(out), _PlusNu(ν), out, Simulations._storage(out), K)
struct _PlusNu{T}
    ν::T
end
@inline (f::_PlusNu)(K) = f.ν + K

# The implicit coefficient of the field `name`: momentum `νm` (one coefficient,
# or `(u, v, w)` for the stress form); scalars `νs`, which is one coefficient
# for θ and qv, or `(θ = …, qv = …)` when the SGS closure gives qv its own
# diffusivity (AMD, `sgs_tracer_diffusivity!`).
@inline _coef_for(νm, νs, name::Symbol) =
    _is_scalar_name(name) ? _scalar_coef(νs, name) :
                            (νm isa NamedTuple ? getfield(νm, name) : νm)
@inline _scalar_coef(νs, ::Symbol) = νs
@inline _scalar_coef(νs::NamedTuple{(:θ, :qv)}, name::Symbol) = getfield(νs, name)

_diffusion_buffer(sim::BoussinesqSimulation, ws::IMEXWorkspace, name::Symbol) =
    name === :u ? sim.lap_u : name === :v ? sim.lap_v : name === :w ? sim.lap_w :
    name === :θ ? sim.lap_θ : ws.scratch

# tendency += ∇·(ν_eff ∇ field) — used to build the full F(u) a
# Rosenbrock method needs. Halos of the state must be filled.
function _add_diffusion_tendencies!(sim::BoussinesqSimulation{T}, ws::IMEXWorkspace{T},
                                    νm, νs, names) where {T}
    for name in names
        f   = getfield(sim, name)
        buf = _diffusion_buffer(sim, ws, name)
        ν   = _coef_for(νm, νs, name)
        _diffusion!(buf, f, ν, sim.mesh)
        axpy!(_tendency(sim, name), one(T), buf)
    end
    return sim
end

# Open faces (N-open-b): the implicit stage solves for the intermediate
# velocity u* = u + τ∇p, so its tangential boundary data get Kim & Moin's
# τ ∂p/∂s; the data themselves are restored (from the returned snapshot)
# before the projection. `nothing` and a no-op for closed boundaries.
function _open_intermediate!(sim::BoussinesqSimulation, τ)
    has_open_boundaries(sim) || return nothing
    snap = open_boundary_snapshot(sim)
    open_intermediate_velocity!(sim, τ)
    return snap
end

# Chorin projection — identical to the one inside Simulations.step!, but
# callable per IMEX stage so every stage value is discretely
# divergence-free.
function _project!(sim::BoussinesqSimulation{T}, Δt::T) where {T}
    _project_velocity!(sim)
    pressure_from_potential!(sim.p, sim.φ, Δt)
    return sim
end

# --- small field algebra over the interior -------------------------------

function _copy_fields!(dst::NamedTuple, sim::BoussinesqSimulation, names)
    for name in names
        copy_storage!(getfield(dst, name).data, getfield(sim, name).data)
    end
    return dst
end

function _copy_tendencies!(dst::NamedTuple, sim::BoussinesqSimulation, names)
    for name in names
        copy_storage!(getfield(dst, name).data, _tendency(sim, name).data)
    end
    return dst
end

# The stage combinations over the interior, as `interior_map!` kernels
# (KA-j, #306); each evaluates the expression of the broadcast it replaced,
# left to right: ((a·A + b·B) + c·C) + d·D.
struct _Lin3{T}
    a::T; b::T; c::T
end
@inline (f::_Lin3)(A, B, C) = f.a * A + f.b * B + f.c * C
struct _Lin4{T}
    a::T; b::T; c::T; d::T
end
@inline (f::_Lin4)(A, B, C, D) = f.a * A + f.b * B + f.c * C + f.d * D

# out = a·A + b·B (interior)
_combine2!(out::Field, a, A::Field, b, B::Field) = interior_map!(_Lin2(a, b), out, A, B)
# out = a·A + b·B + c·C (interior)
_combine3!(out::Field, a, A::Field, b, B::Field, c, C::Field) =
    interior_map!(_Lin3(promote(a, b, c)...), out, A, B, C)
# out = a·A + b·B + c·C + d·D (interior)
_combine4!(out::Field, a, A::Field, b, B::Field, c, C::Field, d, D::Field) =
    interior_map!(_Lin4(promote(a, b, c, d)...), out, A, B, C, D)

# Solve (I - dtγ ∇·(ν∇)) f = f in place, recording solver statistics.
# `increment = true` (Rosenbrock stages k₁, k₂) solves for a change of the
# state, whose wall value is zero: the moving-wall term is dropped.
function _solve_field!(sim::BoussinesqSimulation, ws::IMEXWorkspace, name::Symbol,
                       f::Field, νm, νs, dtγ, tol, maxit; increment::Bool = false)
    scalar = _is_scalar_name(name)
    ν   = _coef_for(νm, νs, name)
    bcs = scalar ? sim.θ_bcs : sim.bcs
    # qv shares θ's walls and open-boundary data but not θ's prescribed top
    # gradient (#254); an increment has zero boundary values
    bcs = increment ? homogeneous_bcs(bcs) : name === :qv ? passive_scalar_bcs(bcs) : bcs
    dw  = scalar ? ws.diff.c : getfield(ws.diff, name)
    stats = solve_implicit_diffusion!(f, f, ν, dtγ, sim.mesh, bcs;
                                      tol = tol, maxit = maxit, workspace = dw,
                                      key = getfield(sim, name).data)
    ws.total_iterations += stats.iterations
    ws.max_residual = max(ws.max_residual, stats.relative_residual)
    return stats
end

# =========================================================================
# Phase 25 — the IMEX drivers
# =========================================================================

"""
    step_imex!(sim, dt, scheme::IMEXScheme; body_force! = nothing,
               post_predictor! = nothing, workspace = imex_workspace(sim),
               tol = 1e-10, maxit = 200,
               momentum_advection = momentum_advect!, sgs = nothing,
               sgs_stress = :laplacian)

Advance `sim` by `dt` with a semi-implicit IMEX scheme
([`ARK2`](@ref) or [`Rosenbrock2`](@ref)):

  * **explicit**: momentum and scalar advection, buoyancy, Coriolis,
    prescribed surface flux, Rayleigh sponge, and the user `body_force!`;
  * **implicit**: diffusion of `u, v, w, θ` (and `qv` when moist) with
    the frozen effective viscosity `ν + νₜ` (`ν + κₜ` for scalars;
    `κₜ = νₜ/Prₜ` for the Smagorinsky closures; `qv` takes the closure's
    [`sgs_tracer_diffusivity!`](@ref Aeolus.Operators.sgs_tracer_diffusivity!),
    which for AMD is its own diffusivity from the `qv` gradient),
    solved by [`solve_implicit_diffusion!`](@ref);
  * **pressure projection at the end of every stage**, so each stage
    value — and therefore the returned state — is discretely
    divergence-free.

The payoff is that `dt` is no longer restricted by `Δz²/ν_eff`: see
[`imex_stability_limit`](@ref) for the two limits side by side. Both
callbacks follow `step!`'s convention — `body_force!(sim, Δt_stage)`
contributes to the explicit tendency, `post_predictor!(sim, Δt_stage)`
runs after the implicit solve and before the projection, so a velocity
correction it applies is folded into a divergence-free result.

The two operator keywords are `step!`'s as well, so a driver can pass
the same keywords to either: `momentum_advection(tu, tv, tw, u, v, w)`
is the (explicit) momentum-advection operator, and `sgs` (an
[`AbstractSGSClosure`](@ref Aeolus.Operators.AbstractSGSClosure), or a
function `f(νt, u, v, w)`; the legacy name `eddy_viscosity!` is still
accepted), when given, replaces the Smagorinsky closure in the frozen
`ν_eff = ν + νt` (momentum) and `ν + κt` (scalars) — it is applied even
with `C_s = 0`, as in `step!`. Its coefficients are computed once, at
`uⁿ`, and its memory is committed
([`advance_sgs!`](@ref Aeolus.Operators.advance_sgs!)) once, after the last
stage.

Notes and limitations:

  * The implicit operator is `∇·(ν_eff ∇·)` with `ν_eff` face-averaged
    from cell centers. With `C_s = 0` this is exactly the molecular term
    `ν∇²` that `step!` applies explicitly, so the two drivers differ only
    by truncation error. With SGS on and `sgs_stress = :laplacian` (the
    default), the implicit operator is a co-located approximation of the
    Laplacian eddy-viscosity divergence (the edge interpolation that
    `add_eddy_viscosity_u!` uses is not reproduced exactly); it is
    2nd-order consistent and SPD, which is what the implicit solve needs,
    and the coefficient is lagged by one step either way.
  * `sgs_stress = :symmetric` (P55-b) splits the stress-form divergence
    `∂ⱼ[νᵢⱼ(∂ⱼuᵢ + ∂ᵢuⱼ)]` of `step!` into its diagonal part
    `∂ⱼ(νᵢⱼ∂ⱼuᵢ)`, **implicit**, with per-direction coefficients (`ν + νh`
    on the x/y faces of the 11, 22, 12 fluxes, `ν + νv` on the 33, 13, 23
    ones) and exactly the interpolation of the explicit kernel — the
    operator is the explicit `part = :diagonal` of
    [`add_sgs_stress_divergence!`](@ref Aeolus.Operators.add_sgs_stress_divergence!)
    and stays symmetric, so `I − γΔt L` is SPD — and the transposed cross
    terms `∂ⱼ(νᵢⱼ∂ᵢuⱼ)`, **explicit**, evaluated at each stage value with
    the same frozen coefficients. For constant ν the cross terms are
    `ν∂ᵢ(∇·u)`, zero on the projected stage values, so the split is exact
    there; they matter only where ν varies, where they are not stiff (no
    second derivative of `uᵢ` along `xⱼ` of its own component).
  * `ν_eff` is frozen at `uⁿ` for the whole step.
  * Returns `sim`. The step's solver statistics are in
    `workspace.total_iterations` and `workspace.max_residual`.
"""
step_imex!(sim::BoussinesqSimulation{T}, dt::Real, scheme::ARK2; kwargs...) where {T} =
    _recorded_imex!(sim, T(dt), scheme; kwargs...)

function _step_imex!(sim::BoussinesqSimulation{T}, dt::Real, scheme::ARK2;
                     body_force!::Union{Nothing,Function} = nothing,
                     post_predictor!::Union{Nothing,Function} = nothing,
                     workspace::IMEXWorkspace = imex_workspace(sim),
                     tol::Real = 1e-10, maxit::Integer = 200,
                     momentum_advection::F = momentum_advect!,
                     sgs = nothing,
                     eddy_viscosity! = nothing,
                     sgs_stress::Symbol = :laplacian) where {T, F}
    Δt = T(dt)
    γ  = T(scheme.γ)
    δ  = T(scheme.δ)
    m  = sim.mesh
    ws = workspace
    names = _active(sim)
    t0 = sim.t
    ws.total_iterations = 0
    ws.max_residual = zero(T)

    closure = _active_sgs(sim, sgs, eddy_viscosity!)
    prepare_open_boundaries!(sim)
    bc0 = has_radiating_outflow(sim) ? open_boundary_snapshot(sim) : nothing  # N-open-c
    νm, νs, sgsT, Ks = _freeze_viscosity!(sim, ws, closure, _check_sgs_stress(sgs_stress))
    _copy_fields!(ws.state0, sim, names)

    # ---- stage 1: (I - γΔt D) U₁ = uⁿ + γΔt E(uⁿ) ------------------------
    _explicit_tendencies!(sim, γ * Δt; body_force! = body_force!,
                          momentum_advection = momentum_advection,
                          sgs_transpose = sgsT, scalar_tensor = Ks)
    _copy_tendencies!(ws.aux, sim, names)                 # E(uⁿ)
    update_open_boundaries!(sim, t0 + γ * Δt, γ * Δt)     # stage-1 boundary data (lift)
    bsnap = _open_intermediate!(sim, γ * Δt)
    for name in names
        f = getfield(sim, name)
        _combine2!(f, one(T), getfield(ws.state0, name),
                      γ * Δt, getfield(ws.aux, name))
        _solve_field!(sim, ws, name, f, νm, νs, γ * Δt, tol, maxit)
    end
    restore_open_boundaries!(sim, bsnap)
    post_predictor! === nothing || post_predictor!(sim, γ * Δt)
    _project!(sim, γ * Δt)
    sim.t = t0 + γ * Δt

    # ---- I(U₁) = ∇·(ν∇U₁) on the projected stage value -------------------
    fill_state_halos!(sim)
    for name in names
        _diffusion!(_diffusion_buffer(sim, ws, name), getfield(sim, name),
                    _coef_for(νm, νs, name), m)
    end

    # ---- stage 2 --------------------------------------------------------
    _explicit_tendencies!(sim, Δt; body_force! = body_force!,
                          momentum_advection = momentum_advection,
                          sgs_transpose = sgsT, scalar_tensor = Ks)   # E(U₁) → sim.t*
    rewind_open_boundaries!(sim, bc0)      # stage 2 restarts from uⁿ
    update_open_boundaries!(sim, t0 + Δt, Δt)
    bsnap = _open_intermediate!(sim, Δt)
    for name in names
        f = getfield(sim, name)
        _combine4!(f, one(T), getfield(ws.state0, name),
                      δ * Δt, getfield(ws.aux, name),           # E(uⁿ)
                      (one(T) - δ) * Δt, _tendency(sim, name),  # E(U₁)
                      (one(T) - γ) * Δt, _diffusion_buffer(sim, ws, name))
        _solve_field!(sim, ws, name, f, νm, νs, γ * Δt, tol, maxit)
    end
    restore_open_boundaries!(sim, bsnap)
    post_predictor! === nothing || post_predictor!(sim, Δt)
    _project!(sim, Δt)
    sim.t = t0 + Δt
    closure === nothing || advance_sgs!(closure, sim, Δt)   # once per step

    return sim
end

step_imex!(sim::BoussinesqSimulation{T}, dt::Real, scheme::Rosenbrock2; kwargs...) where {T} =
    _recorded_imex!(sim, T(dt), scheme; kwargs...)

function _step_imex!(sim::BoussinesqSimulation{T}, dt::Real, scheme::Rosenbrock2;
                     body_force!::Union{Nothing,Function} = nothing,
                     post_predictor!::Union{Nothing,Function} = nothing,
                     workspace::IMEXWorkspace = imex_workspace(sim),
                     tol::Real = 1e-10, maxit::Integer = 200,
                     momentum_advection::F = momentum_advect!,
                     sgs = nothing,
                     eddy_viscosity! = nothing,
                     sgs_stress::Symbol = :laplacian) where {T, F}
    Δt  = T(dt)
    γ   = T(scheme.γ)
    a21 = T(scheme.a21)
    c21 = T(scheme.c21)
    b1, b2 = T(scheme.b[1]), T(scheme.b[2])
    m   = sim.mesh
    ws  = workspace
    names = _active(sim)
    t0 = sim.t
    ws.total_iterations = 0
    ws.max_residual = zero(T)

    closure = _active_sgs(sim, sgs, eddy_viscosity!)
    prepare_open_boundaries!(sim)
    bc0 = has_radiating_outflow(sim) ? open_boundary_snapshot(sim) : nothing  # N-open-c
    νm, νs, sgsT, Ks = _freeze_viscosity!(sim, ws, closure, _check_sgs_stress(sgs_stress))
    _copy_fields!(ws.state0, sim, names)
    k2set = _k2_set!(ws)

    # ---- k₁: (I - γΔt J) k₁ = Δt F(uⁿ) ----------------------------------
    _explicit_tendencies!(sim, Δt; body_force! = body_force!,
                          momentum_advection = momentum_advection,
                          sgs_transpose = sgsT, scalar_tensor = Ks)
    _add_diffusion_tendencies!(sim, ws, νm, νs, names)
    for name in names
        k1 = getfield(ws.aux, name)
        _combine2!(k1, zero(T), k1, Δt, _tendency(sim, name))
        _solve_field!(sim, ws, name, k1, νm, νs, γ * Δt, tol, maxit; increment = true)
    end

    # ---- stage state uⁿ + a₂₁k₁, projected ------------------------------
    for name in names
        _combine2!(getfield(sim, name), one(T), getfield(ws.state0, name),
                   a21, getfield(ws.aux, name))
    end
    post_predictor! === nothing || post_predictor!(sim, a21 * Δt)
    update_open_boundaries!(sim, t0 + a21 * Δt, a21 * Δt)
    _project!(sim, a21 * Δt)
    sim.t = t0 + a21 * Δt

    # ---- k₂: (I - γΔt J) k₂ = Δt F(u+k₁) + c₂₁ k₁ ------------------------
    _explicit_tendencies!(sim, Δt; body_force! = body_force!,
                          momentum_advection = momentum_advection,
                          sgs_transpose = sgsT, scalar_tensor = Ks)
    _add_diffusion_tendencies!(sim, ws, νm, νs, names)
    for name in names
        k2 = getfield(k2set, name)
        _combine2!(k2, Δt, _tendency(sim, name), c21, getfield(ws.aux, name))
        _solve_field!(sim, ws, name, k2, νm, νs, γ * Δt, tol, maxit; increment = true)
    end

    # ---- update ---------------------------------------------------------
    for name in names
        _combine3!(getfield(sim, name), one(T), getfield(ws.state0, name),
                   b1, getfield(ws.aux, name), b2, getfield(k2set, name))
    end
    post_predictor! === nothing || post_predictor!(sim, Δt)
    rewind_open_boundaries!(sim, bc0)      # stage 2 restarts from uⁿ
    update_open_boundaries!(sim, t0 + Δt, Δt)
    _project!(sim, Δt)
    sim.t = t0 + Δt
    closure === nothing || advance_sgs!(closure, sim, Δt)   # once per step

    return sim
end

# One IMEX step inside the step record's bracket (#336), with the
# implicit-diffusion CG work of the step.
function _recorded_imex!(sim::BoussinesqSimulation{T}, Δt::T, scheme; kwargs...) where {T}
    rec = sim.workspaces.rec
    outer = _step_begin!(rec, sim)
    try
        _step_imex!(sim, Δt, scheme; kwargs...)
    catch
        _step_abort!(rec); rethrow()
    end
    ws = get(kwargs, :workspace, imex_workspace(sim))
    rec.implicit_iters += ws.total_iterations
    rec.implicit_residual = Simulations._nmax(rec.implicit_residual, ws.max_residual)
    return _step_end!(rec, sim, Δt, outer)
end

# The step limit of the processes an IMEX scheme treats explicitly
# (advection, buoyancy, Coriolis) from a `cfl_diagnostics` result, and
# which of them binds. Shared by `adaptive_dt(...; scheme)` and
# `imex_stability_limit`. A NaN advective limit stays NaN.
function _explicit_limit(d, scheme::IMEXScheme)
    dt = d.dt_adv
    limiter = :advective
    if d.dt_buoy < dt; dt = d.dt_buoy; limiter = :buoyancy; end
    if d.dt_cor  < dt; dt = d.dt_cor;  limiter = :coriolis; end
    if d.dt_limits < dt; dt = d.dt_limits; limiter = d.limits_limiter; end   # explicit components
    return (dt * oftype(dt, explicit_cfl_factor(scheme)), limiter)
end

"""
    imex_stability_limit(sim, scheme; constraints = CFLConstraints(), sgs = nothing,
                         sgs_stress = :laplacian)
    -> (; dt_explicit, dt_diffusive, dt_fully_explicit, speedup, limiter,
          explicit_limiter, diagnostics)

Quantify what treating diffusion implicitly buys.

  * `dt_explicit`   — the limit from the *explicit* half alone
    (advection, buoyancy, Coriolis), scaled by
    [`explicit_cfl_factor`](@ref). This is the step
    [`step_imex!`](@ref) can take.
  * `dt_diffusive`  — the `Δ²/ν_eff` limit that would apply if diffusion
    were explicit (`min(dt_diff, dt_sgs)`).
  * `dt_fully_explicit` — `min` of the two, i.e. what `step!` is bound
    by.
  * `speedup`       — `dt_explicit / dt_diffusive`, the achievable factor
    when diffusion is the binding constraint (values ≤ 1 mean diffusion
    was not limiting and IMEX buys nothing but robustness).

`limiter` names the process that binds the IMEX step, `explicit_limiter`
the one that binds a fully explicit step, and `diagnostics` is the full
[`cfl_diagnostics`](@ref) result (with the SGS closure `sgs` and the
momentum form `sgs_stress`, as there).
"""
function imex_stability_limit(sim::BoussinesqSimulation{T}, scheme::IMEXScheme;
                              constraints::CFLConstraints = CFLConstraints(T),
                              sgs = nothing, sgs_stress::Symbol = :laplacian) where {T}
    d = cfl_diagnostics(sim, constraints; sgs = sgs, sgs_stress = sgs_stress)
    dt_explicit, limiter = _explicit_limit(d, scheme)

    dt_diffusive = min(d.dt_diff, d.dt_sgs)
    dt_fully_explicit = min(dt_explicit, dt_diffusive)
    explicit_limiter = dt_diffusive < dt_explicit ?
                       (d.dt_sgs < d.dt_diff ? :sgs : :diffusive) : limiter

    return (dt_explicit = dt_explicit, dt_diffusive = dt_diffusive,
            dt_fully_explicit = dt_fully_explicit,
            speedup = dt_explicit / dt_diffusive,
            limiter = limiter, explicit_limiter = explicit_limiter,
            diagnostics = d)
end

end # module
