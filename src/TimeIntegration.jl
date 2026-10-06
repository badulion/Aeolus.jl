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
using ..Fields: Field, interior_view, location
using ..BoundaryConditions
using ..BoundaryConditions: FieldBoundaryConditions, fill_halos!,
                            Periodic, NoFlux, ImpermeableWall, ValueBC, FluxBC,
                            is_wall, homogeneous_bcs
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
using ..Simulations: BoussinesqSimulation, max_divergence,
                     _dzc, _dzf, _dz_min      # vertical metrics, stretched-aware (#151)
using ..TimeSteppers: step!
using ..TurbulenceStability: buoyancy_frequency_squared   # the one N² (P58-a)

export CFLConstraints, cfl_diagnostics, adaptive_dt, stress_form_cfl_viscosity,
       SubsteppingSchedule, substeps_for,
       StateSnapshot, state_snapshot, save_state!, restore_state!,
       step_adaptive!,
       DiffusionWorkspace, solve_implicit_diffusion!,
       IMEXScheme, ARK2, Rosenbrock2,
       IMEXWorkspace, imex_workspace, step_imex!, imex_stability_limit

# =========================================================================
# Small shared helpers
# =========================================================================

"In-place `y[interior] += a · x[interior]`."
function _axpy_interior!(y::Field, a::Number, x::Field, m::AbstractMesh)
    ii, jj, kk = interior(m)
    @inbounds @views y.data[ii, jj, kk] .+= a .* x.data[ii, jj, kk]
    return y
end

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
function _max_advective_rate(ud, vd, wd, m::RegularGrid{T}) where {T}
    ii, jj, kk = interior(m)
    idx = 1 / m.dx; idy = 1 / m.dy; idz = 1 / m.dz
    r = zero(T)
    @inbounds for k in kk, j in jj, i in ii
        au = max(abs(ud[i, j, k]), abs(ud[i+1, j, k]))
        av = max(abs(vd[i, j, k]), abs(vd[i, j+1, k]))
        aw = max(abs(wd[i, j, k]), abs(wd[i, j, k+1]))
        r  = max(r, au * idx + av * idy + aw * idz)   # NaN-propagating
    end
    return r
end

# The same rate with the local cell thickness: on a vertically stretched
# grid (#151) the vertical CFL number of a cell is |w| / Δz_c[k], so the
# thin near-surface cells bind, not the mean spacing.
function _max_advective_rate(ud, vd, wd, m::AbstractMesh{T}) where {T}
    ii, jj, kk = interior(m)
    idx = 1 / m.dx; idy = 1 / m.dy
    r = zero(T)
    @inbounds for k in kk
        idz = 1 / _dzc(m, k)
        for j in jj, i in ii
            au = max(abs(ud[i, j, k]), abs(ud[i+1, j, k]))
            av = max(abs(vd[i, j, k]), abs(vd[i, j+1, k]))
            aw = max(abs(wd[i, j, k]), abs(wd[i, j, k+1]))
            r  = max(r, au * idx + av * idy + aw * idz)   # NaN-propagating
        end
    end
    return r
end

# Brunt-Väisälä frequency from the horizontal-mean (resolved) θ profile:
# the largest N² across the faces between the plane means, from the
# TurbulenceStability contract (Boussinesq form, g/θ₀). Allocation-free:
# carries the previous level's mean in a scalar.
function _brunt_vaisala(sim::BoussinesqSimulation{T}) where {T}
    sim.has_buoyancy || return zero(T)
    m = sim.mesh
    ii, jj, kk = interior(m)
    n_h = length(ii) * length(jj)
    N²max = zero(T)
    prev  = zero(T)
    d = sim.θ.data
    @inbounds for (kc, k) in enumerate(kk)
        s = zero(T)
        for j in jj, i in ii
            s += d[i, j, k]
        end
        mean_k = s / n_h
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
                    sgs_stress = :laplacian)

Report the time-step limit imposed by *each* process separately. Returns a
`NamedTuple`:

| Key                 | Meaning                                                        |
| :------------------ | :------------------------------------------------------------- |
| `dt_adv`            | advective limit, per-cell `Σ|uᵢ|/Δxᵢ` form                      |
| `dt_diff`           | total (molecular + SGS) momentum-diffusion limit               |
| `dt_buoy`           | `C_buoyancy / N` from the resolved mean θ profile               |
| `dt_cor`            | `C_coriolis / |f|`                                             |
| `dt_sgs`            | SGS-only limit `C_diffusive Δ²_harm / max(νt_cfl, κt_max)`      |
| `dt_limiting`       | `min` of the five                                              |
| `limiter`           | `:advective`, `:diffusive`, `:buoyancy`, `:coriolis` or `:sgs`  |
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

Ties are broken in the table order above, so a pure-molecular run where
`dt_diff == dt_sgs` reports `:diffusive`.
"""
function cfl_diagnostics(sim::BoussinesqSimulation{T},
                         c::CFLConstraints = CFLConstraints(T);
                         sgs = nothing,
                         update_νt::Bool = true,
                         sgs_active::Union{Nothing,Bool} = nothing,
                         sgs_stress::Symbol = :laplacian) where {T}
    m = sim.mesh
    INF = T(Inf)

    fill_halos!(sim.u, sim.bcs)
    fill_halos!(sim.v, sim.bcs)
    fill_halos!(sim.w, sim.bcs)

    # --- advective --------------------------------------------------------
    rate = _max_advective_rate(sim.u.data, sim.v.data, sim.w.data, m)
    # A NaN velocity must poison the limit rather than silently reporting
    # "no constraint"; `adaptive_dt` then collapses to dt_min.
    dt_adv = isnan(rate) ? T(NaN) : rate > 0 ? c.C_advective / rate : INF

    # --- diffusive (molecular + SGS) --------------------------------------
    closure = _active_sgs(sim, sgs, nothing)
    active  = sgs_active === nothing ? closure !== nothing : sgs_active
    if update_νt && closure !== nothing
        fill_halos!(sim.θ, sim.θ_bcs)                      # θ-aware closures
        sim.has_moisture && fill_halos!(sim.qv, homogeneous_bcs(sim.θ_bcs))
        compute_sgs!(closure, sim)
    end
    f = closure === nothing ? SGSFields(sim.νt, sim.κt) : sgs_fields(closure, sim)
    νh_max  = active ? _max_interior(f.νh) : zero(T)
    νv_max  = active ? _max_interior(f.νv) : zero(T)
    νt_max  = max(νh_max, νv_max)
    κt_max  = active ? max(_max_interior(f.κh), _max_interior(f.κv)) : zero(T)
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
    if dt_sgs  < dt_limiting; dt_limiting = dt_sgs;  limiter = :sgs;       end

    return (dt_adv = dt_adv, dt_diff = dt_diff, dt_buoy = dt_buoy,
            dt_cor = dt_cor, dt_sgs = dt_sgs,
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

_max_interior(f::Field) = maximum(interior_view(f))

"""
    adaptive_dt(sim, constraints; dt_prev = nothing, diagnostics = nothing,
                scheme = nothing, sgs = nothing, sgs_stress = :laplacian)

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
divergence (see [`cfl_diagnostics`](@ref)).
"""
function adaptive_dt(sim::BoussinesqSimulation{T}, c::CFLConstraints;
                     dt_prev = nothing, diagnostics = nothing,
                     scheme = nothing, sgs = nothing,
                     sgs_stress::Symbol = :laplacian) where {T}
    d  = diagnostics === nothing ? cfl_diagnostics(sim, c; sgs = sgs, sgs_stress = sgs_stress) :
                                   diagnostics
    dt = scheme isa IMEXScheme ? first(_explicit_limit(d, scheme)) : d.dt_limiting
    isnan(dt) && return c.dt_min
    if dt_prev !== nothing
        dtp = T(dt_prev)
        dtp > 0 && (dt = min(dt, c.growth_max * dtp))
    end
    return clamp(dt, c.dt_min, c.dt_max)
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

Preallocated copy of the prognostic state (`u, v, w, p, θ, qv` plus the
clock) used by [`step_adaptive!`](@ref) to roll a failed step back. The
buffers are allocated once; [`save_state!`](@ref) and
[`restore_state!`](@ref) only copy. Halos are copied too, so a restore is
bit-identical to the pre-step state.
"""
mutable struct StateSnapshot{T}
    u::Array{T,3}
    v::Array{T,3}
    w::Array{T,3}
    p::Array{T,3}
    θ::Array{T,3}
    qv::Array{T,3}
    t::T
    valid::Bool
end

StateSnapshot(sim::BoussinesqSimulation{T}) where {T} =
    StateSnapshot{T}(similar(sim.u.data), similar(sim.v.data), similar(sim.w.data),
                     similar(sim.p.data), similar(sim.θ.data), similar(sim.qv.data),
                     sim.t, false)

"Copy `sim`'s prognostic state into `snap` (no allocation)."
function save_state!(snap::StateSnapshot, sim::BoussinesqSimulation)
    copyto!(snap.u,  sim.u.data)
    copyto!(snap.v,  sim.v.data)
    copyto!(snap.w,  sim.w.data)
    copyto!(snap.p,  sim.p.data)
    copyto!(snap.θ,  sim.θ.data)
    copyto!(snap.qv, sim.qv.data)
    snap.t     = sim.t
    snap.valid = true
    return snap
end

"Restore `sim`'s prognostic state (and clock) from `snap`."
function restore_state!(sim::BoussinesqSimulation, snap::StateSnapshot)
    snap.valid || throw(ArgumentError("StateSnapshot has never been saved"))
    copyto!(sim.u.data,  snap.u)
    copyto!(sim.v.data,  snap.v)
    copyto!(sim.w.data,  snap.w)
    copyto!(sim.p.data,  snap.p)
    copyto!(sim.θ.data,  snap.θ)
    copyto!(sim.qv.data, snap.qv)
    sim.t = snap.t
    return sim
end

# One snapshot per simulation, created on demand. Weak keys so a
# discarded simulation does not leak its snapshot.
const _SNAPSHOTS = WeakKeyDict{Any,Any}()

"""
    state_snapshot(sim) -> StateSnapshot

The snapshot buffer associated with `sim`, allocated on first use and
cached afterwards (weakly, so it is freed with `sim`). This is what makes
`step_adaptive!`'s retry path allocation-free.
"""
state_snapshot(sim::BoussinesqSimulation) =
    get!(() -> StateSnapshot(sim), _SNAPSHOTS, sim)::StateSnapshot

function _all_finite(sim::BoussinesqSimulation)
    all(isfinite, interior_view(sim.u)) || return false
    all(isfinite, interior_view(sim.v)) || return false
    all(isfinite, interior_view(sim.w)) || return false
    all(isfinite, interior_view(sim.θ)) || return false
    all(isfinite, interior_view(sim.p)) || return false
    sim.has_moisture && !all(isfinite, interior_view(sim.qv)) && return false
    return true
end

_max_speed(sim::BoussinesqSimulation) =
    max(maximum(abs, interior_view(sim.u)),
        maximum(abs, interior_view(sim.v)),
        maximum(abs, interior_view(sim.w)))

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
function step_adaptive!(sim::BoussinesqSimulation{T}, constraints::CFLConstraints;
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
                        sgs_stress::Symbol = :laplacian,
                        verbose::Bool = false) where {T}

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

    save_state!(snapshot, sim)
    # the closure's memory: the step commits it once (advance_sgs!); a
    # rejected step's commit is rolled back with the prognostic state
    sgs_snap = closure === nothing ? nothing : sgs_state(closure)
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
            verbose && @warn "step_adaptive!: stepper threw" dt=Δt exception=err
        end
        if !failed
            if !_all_finite(sim)
                failed = true; reason = :nonfinite
            else
                d = max_divergence(sim)
                if !isfinite(d) || d > div_tol
                    failed = true; reason = :divergence
                elseif _max_speed(sim) > u_cap
                    failed = true; reason = :velocity
                end
            end
        end

        failed || break

        if n_retries >= max_retries
            restore_state!(sim, snapshot)
            closure === nothing || restore_sgs_state!(closure, sgs_snap)
            error("step_adaptive!: step failed ($reason) after $max_retries " *
                  "retries; last dt = $Δt. State has been restored to the " *
                  "beginning of the step.")
        end
        restore_state!(sim, snapshot)
        closure === nothing || restore_sgs_state!(closure, sgs_snap)
        Δt_new = max(Δt / 2, constraints.dt_min)
        if Δt_new == Δt
            error("step_adaptive!: step failed ($reason) at dt_min = " *
                  "$(constraints.dt_min); cannot reduce dt further. State " *
                  "has been restored to the beginning of the step.")
        end
        verbose && @info "step_adaptive!: retry" reason from=Δt to=Δt_new
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
            @warn "step_adaptive!: state became non-finite inside a fast process" dt=Δt
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

function _diffusion_kernel!(out::Field{L,T}, f::Field{L,T}, ν, m::RegularGrid{T}) where {L,T}
    ii, jj, kk = interior(m)
    idx2 = 1 / m.dx^2; idy2 = 1 / m.dy^2; idz2 = 1 / m.dz^2
    d = f.data; o = out.data
    @inbounds for k in kk, j in jj, i in ii
        o[i, j, k] = _diff_at(d, ν, i, j, k, idx2, idy2, idz2)
    end
    return out
end

# Rows that are *not* unknowns: `w` on an impermeable wall face is pinned
# to zero by the BC, so it is removed from the CG system (keeping the
# reduced operator SPD) instead of being solved for. Free-slip and no-slip
# walls (`is_wall`) pin it alike; the top wall face lies in the halo.
_constraint_index(::Location, ::RegularGrid, ::FieldBoundaryConditions) = 0
_constraint_index(::FaceZ, m::RegularGrid, bcs::FieldBoundaryConditions) =
    is_wall(bcs.bottom) ? m.Hz + 1 : 0

function _apply_constraint!(a::AbstractArray{T,3}, kc::Int) where {T}
    kc == 0 && return a
    @inbounds a[:, :, kc] .= zero(T)
    return a
end

function _assert_supported_bcs(bcs::FieldBoundaryConditions)
    for bc in (bcs.west, bcs.east, bcs.south, bcs.north, bcs.bottom, bcs.top)
        # a FluxBC top (#254) is handled like a moving wall: its affine part
        # goes to the right-hand side (see _cg_implicit!)
        bc isa FluxBC && bc === bcs.top && continue
        if bc isa ValueBC || bc isa FluxBC
            throw(ArgumentError(
                "solve_implicit_diffusion! supports the BC types handled by " *
                "fill_halos! (Periodic, NoFlux, ImpermeableWall, NoSlipWall, and a FluxBC top). " *
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
image, preconditioned residual). Reuse one per `(location, mesh)` pair to
keep the solve allocation-free. The diagonal is recomputed on the fly
rather than stored (a handful of flops per cell, far cheaper than the
matvec, and one array less).
"""
struct DiffusionWorkspace{T,F<:Field}
    p_field::F
    r::Array{T,3}
    Ap::Array{T,3}
    z::Array{T,3}
end

function DiffusionWorkspace(f::Field{L,T}) where {L,T}
    pf = Field(L(), f.mesh)
    sz = size(f.data)
    return DiffusionWorkspace{T,typeof(pf)}(pf, zeros(T, sz), zeros(T, sz), zeros(T, sz))
end

function _apply_implicit!(out::AbstractArray{T,3}, xf::Field{L,T}, ν, Δt::T,
                          m::RegularGrid{T}, bcs::FieldBoundaryConditions,
                          kc::Int) where {L,T}
    fill_halos!(xf, bcs)
    ii, jj, kk = interior(m)
    idx2 = 1 / m.dx^2; idy2 = 1 / m.dy^2; idz2 = 1 / m.dz^2
    d = xf.data
    @inbounds for k in kk, j in jj, i in ii
        out[i, j, k] = d[i, j, k] - Δt * _diff_at(d, ν, i, j, k, idx2, idy2, idz2)
    end
    return _apply_constraint!(out, kc)
end

@inline function _idot(a::AbstractArray{T,3}, b::AbstractArray{T,3}, ii, jj, kk) where {T}
    s = zero(T)
    @inbounds for k in kk, j in jj, i in ii
        s += a[i, j, k] * b[i, j, k]
    end
    return s
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
                                   warn::Bool = true) where {L,T}
    _assert_supported_bcs(bcs)
    precondition in (:jacobi, :none) ||
        throw(ArgumentError("precondition must be :jacobi or :none, got :$precondition"))
    Δt = T(dt)
    Δt >= 0 || throw(ArgumentError("dt must be ≥ 0"))
    ν = _coef(ν_eff, m)
    ii, jj, kk = interior(m)
    τ = T(tol)

    if ν isa T && (Δt == 0 || ν == 0)
        if field !== rhs
            @inbounds @views field.data[ii, jj, kk] .= rhs.data[ii, jj, kk]
        end
        return (iterations = 0, residual = zero(T), relative_residual = zero(T),
                converged = true, tol = τ)
    end

    return _cg_implicit!(field, rhs, ν, Δt, m, bcs, workspace, τ, Int(maxit),
                         precondition, warn)
end

# Function barrier: specializes on the concrete type of ν.
function _cg_implicit!(field::Field{L,T}, rhs::Field{L,T}, ν, Δt::T,
                       m::RegularGrid{T}, bcs_full::FieldBoundaryConditions,
                       ws::DiffusionWorkspace, τ::T, maxit::Int,
                       precondition::Symbol, warn::Bool) where {L,T}
    ii, jj, kk = interior(m)
    idx2 = 1 / m.dx^2; idy2 = 1 / m.dy^2; idz2 = 1 / m.dz^2
    # A moving no-slip wall makes the operator affine: iterate with the
    # stationary wall and carry the wall term in the right-hand side.
    bcs = homogeneous_bcs(bcs_full)
    affine = bcs !== bcs_full
    kc = _constraint_index(L(), m, bcs)
    jacobi = precondition === :jacobi

    x  = field.data
    b  = rhs.data
    r  = ws.r; Ap = ws.Ap; z = ws.z
    pf = ws.p_field; p = pf.data

    _apply_constraint!(x, kc)

    # Wall term g = ∇·(ν∇0) with the moving-wall halo, into z (free until
    # the preconditioner first writes it): b_eff = b + Δt g.
    if affine
        fill!(p, zero(T))
        fill_halos!(pf, bcs_full)
        @inbounds for k in kk, j in jj, i in ii
            z[i, j, k] = Δt * _diff_at(p, ν, i, j, k, idx2, idy2, idz2)
        end
    end

    # r = b - A x   (safe when b === x: A x is formed into Ap first)
    _apply_implicit!(Ap, field, ν, Δt, m, bcs, kc)
    bnorm² = zero(T)
    @inbounds for k in kk, j in jj, i in ii
        if k == kc
            r[i, j, k] = zero(T)
            continue
        end
        bv = affine ? b[i, j, k] + z[i, j, k] : b[i, j, k]
        r[i, j, k] = bv - Ap[i, j, k]
        bnorm² += bv * bv
    end
    bnorm = sqrt(bnorm²)

    if bnorm == 0
        @inbounds for k in kk, j in jj, i in ii
            x[i, j, k] = zero(T)
        end
        return (iterations = 0, residual = zero(T), relative_residual = zero(T),
                converged = true, tol = τ)
    end

    # z = M⁻¹ r ; p = z
    rz = zero(T)
    @inbounds for k in kk, j in jj, i in ii
        zz = if k == kc
            zero(T)
        elseif jacobi
            r[i, j, k] / _diag_at(ν, i, j, k, idx2, idy2, idz2, Δt)
        else
            r[i, j, k]
        end
        z[i, j, k] = zz
        p[i, j, k] = zz
        rz += r[i, j, k] * zz
    end

    rnorm     = sqrt(_idot(r, r, ii, jj, kk))
    converged = rnorm <= τ * bnorm
    iters     = 0

    while !converged && iters < maxit
        iters += 1
        _apply_implicit!(Ap, pf, ν, Δt, m, bcs, kc)
        pAp = _idot(p, Ap, ii, jj, kk)
        if !(pAp > 0)
            warn && @warn "solve_implicit_diffusion!: CG breakdown (pᵀAp = $pAp); " *
                          "operator is not positive definite for these BCs"
            break
        end
        α = rz / pAp
        @inbounds for k in kk, j in jj, i in ii
            x[i, j, k] += α * p[i, j, k]
            r[i, j, k] -= α * Ap[i, j, k]
        end
        rnorm = sqrt(_idot(r, r, ii, jj, kk))
        if rnorm <= τ * bnorm
            converged = true
            break
        end
        rz_new = zero(T)
        @inbounds for k in kk, j in jj, i in ii
            zz = if k == kc
                zero(T)
            elseif jacobi
                r[i, j, k] / _diag_at(ν, i, j, k, idx2, idy2, idz2, Δt)
            else
                r[i, j, k]
            end
            z[i, j, k] = zz
            rz_new += r[i, j, k] * zz
        end
        β = rz_new / rz
        @inbounds for k in kk, j in jj, i in ii
            p[i, j, k] = z[i, j, k] + β * p[i, j, k]
        end
        _apply_constraint!(p, kc)
        rz = rz_new
    end

    converged || (warn && @warn "solve_implicit_diffusion! hit maxit=$maxit with " *
                                "‖r‖/‖b‖ = $(rnorm / bnorm)")
    return (iterations = iters, residual = rnorm,
            relative_residual = rnorm / bnorm, converged = converged, tol = τ)
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

const _IMEX_WORKSPACES = WeakKeyDict{Any,Any}()

"""
    imex_workspace(sim) -> IMEXWorkspace

The [`IMEXWorkspace`](@ref) associated with `sim`, allocated on first use
and cached weakly so it is freed with `sim`. Pass it explicitly to
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
    return get!(() -> IMEXWorkspace(sim), _IMEX_WORKSPACES, sim)::IMEXWorkspace
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
    fill_halos!(sim.u, sim.bcs)
    fill_halos!(sim.v, sim.bcs)
    fill_halos!(sim.w, sim.bcs)
    fill_halos!(sim.θ, sim.θ_bcs)
    sim.has_moisture && fill_halos!(sim.qv, homogeneous_bcs(sim.θ_bcs))

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
    if sim.has_sponge
        z_top = sim.mesh.z0 + sim.mesh.Lz
        add_sponge!(sim.tu, sim.u, sim.sponge_α_max, sim.sponge_z_start, z_top)
        add_sponge!(sim.tv, sim.v, sim.sponge_α_max, sim.sponge_z_start, z_top)
        add_sponge!(sim.tw, sim.w, sim.sponge_α_max, sim.sponge_z_start, z_top)
    end

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
    fill_halos!(sim.u, sim.bcs)
    fill_halos!(sim.v, sim.bcs)
    fill_halos!(sim.w, sim.bcs)
    fill_halos!(sim.θ, sim.θ_bcs)                      # θ-aware closures
    sim.has_moisture && fill_halos!(sim.qv, homogeneous_bcs(sim.θ_bcs))
    f = compute_sgs!(closure, sim)
    ws.ν_mom  === nothing && (ws.ν_mom  = similar(sim.νt.data))
    ws.ν_scal === nothing && (ws.ν_scal = similar(sim.νt.data))
    νm = ws.ν_mom::Array{T,3}; νs = ws.ν_scal::Array{T,3}
    @inbounds νm .= sim.ν .+ f.νh.data
    @inbounds νs .= sim.ν .+ f.κh.data
    m = sim.mesh
    Ks = sgs_scalar_tensor(closure, f, sim)
    cs = if Ks !== nothing
        ws.ν_scal_v === nothing && (ws.ν_scal_v = similar(sim.νt.data))
        ws.ν_scal_y === nothing && (ws.ν_scal_y = similar(sim.νt.data))
        νsy = ws.ν_scal_y::Array{T,3}; νsz = ws.ν_scal_v::Array{T,3}
        @inbounds νs  .= sim.ν .+ Ks.K11.data
        @inbounds νsy .= sim.ν .+ Ks.K22.data
        @inbounds νsz .= sim.ν .+ Ks.K33.data
        _DirCoef3(_coef(νs, m), _coef(νsy, m), _coef(νsz, m))
    elseif f.κv === f.κh
        _coef(νs, m)
    else
        ws.ν_scal_v === nothing && (ws.ν_scal_v = similar(sim.νt.data))
        νsv = ws.ν_scal_v::Array{T,3}
        @inbounds νsv .= sim.ν .+ f.κv.data
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
            @inbounds νq .= sim.ν .+ κq.data
            cs = (θ = cs, qv = _coef(νq, m))
        end
    end
    stress === :symmetric || return (_coef(νm, m), cs, nothing, Ks)
    νmv = if f.νv === f.νh
        νm
    else
        ws.ν_mom_v === nothing && (ws.ν_mom_v = similar(sim.νt.data))
        a = ws.ν_mom_v::Array{T,3}
        @inbounds a .= sim.ν .+ f.νv.data
        a
    end
    h = _coef(νm, m); v = _coef(νmv, m)
    cm = (u = _StressCoef(FaceX(), h, v), v = _StressCoef(FaceY(), h, v),
          w = _StressCoef(FaceZ(), h, v))
    return (cm, cs, (νm, νmv), Ks)
end

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
        _axpy_interior!(_tendency(sim, name), one(T), buf, sim.mesh)
    end
    return sim
end

# Chorin projection — identical to the one inside Simulations.step!, but
# callable per IMEX stage so every stage value is discretely
# divergence-free.
function _project!(sim::BoussinesqSimulation{T}, Δt::T) where {T}
    fill_halos!(sim.u, sim.bcs)
    fill_halos!(sim.v, sim.bcs)
    fill_halos!(sim.w, sim.bcs)
    divergence(sim.div_buf, sim.u, sim.v, sim.w)
    solve!(sim.φ, sim.div_buf, sim.solver)
    fill_halos!(sim.φ, sim.bcs)
    ∂x(sim.dφdx, sim.φ); ∂y(sim.dφdy, sim.φ); ∂z(sim.dφdz, sim.φ)
    _axpy_interior!(sim.u, -one(T), sim.dφdx, sim.mesh)
    _axpy_interior!(sim.v, -one(T), sim.dφdy, sim.mesh)
    _axpy_interior!(sim.w, -one(T), sim.dφdz, sim.mesh)
    ii, jj, kk = interior(sim.mesh)
    @inbounds @views sim.p.data[ii, jj, kk] .= sim.φ.data[ii, jj, kk] ./ Δt
    return sim
end

# --- small field algebra over the interior -------------------------------

function _copy_fields!(dst::NamedTuple, sim::BoussinesqSimulation, names)
    for name in names
        copyto!(getfield(dst, name).data, getfield(sim, name).data)
    end
    return dst
end

function _copy_tendencies!(dst::NamedTuple, sim::BoussinesqSimulation, names)
    for name in names
        copyto!(getfield(dst, name).data, _tendency(sim, name).data)
    end
    return dst
end

# out = a·A + b·B (interior)
function _combine2!(out::Field, a, A::Field, b, B::Field, m::AbstractMesh)
    ii, jj, kk = interior(m)
    @inbounds @views out.data[ii, jj, kk] .= a .* A.data[ii, jj, kk] .+
                                             b .* B.data[ii, jj, kk]
    return out
end

# out = a·A + b·B + c·C (interior)
function _combine3!(out::Field, a, A::Field, b, B::Field, c, C::Field, m::AbstractMesh)
    ii, jj, kk = interior(m)
    @inbounds @views out.data[ii, jj, kk] .= a .* A.data[ii, jj, kk] .+
                                             b .* B.data[ii, jj, kk] .+
                                             c .* C.data[ii, jj, kk]
    return out
end

# out = a·A + b·B + c·C + d·D (interior)
function _combine4!(out::Field, a, A::Field, b, B::Field, c, C::Field,
                    d, D::Field, m::AbstractMesh)
    ii, jj, kk = interior(m)
    @inbounds @views out.data[ii, jj, kk] .= a .* A.data[ii, jj, kk] .+
                                             b .* B.data[ii, jj, kk] .+
                                             c .* C.data[ii, jj, kk] .+
                                             d .* D.data[ii, jj, kk]
    return out
end

# Solve (I - dtγ ∇·(ν∇)) f = f in place, recording solver statistics.
# `increment = true` (Rosenbrock stages k₁, k₂) solves for a change of the
# state, whose wall value is zero: the moving-wall term is dropped.
function _solve_field!(sim::BoussinesqSimulation, ws::IMEXWorkspace, name::Symbol,
                       f::Field, νm, νs, dtγ, tol, maxit; increment::Bool = false)
    scalar = _is_scalar_name(name)
    ν   = _coef_for(νm, νs, name)
    bcs = scalar ? sim.θ_bcs : sim.bcs
    # qv shares θ's walls but not θ's prescribed top gradient (#254)
    (increment || name === :qv) && (bcs = homogeneous_bcs(bcs))
    dw  = scalar ? ws.diff.c : getfield(ws.diff, name)
    stats = solve_implicit_diffusion!(f, f, ν, dtγ, sim.mesh, bcs;
                                      tol = tol, maxit = maxit, workspace = dw)
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
function step_imex!(sim::BoussinesqSimulation{T}, dt::Real, scheme::ARK2;
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
    νm, νs, sgsT, Ks = _freeze_viscosity!(sim, ws, closure, _check_sgs_stress(sgs_stress))
    _copy_fields!(ws.state0, sim, names)

    # ---- stage 1: (I - γΔt D) U₁ = uⁿ + γΔt E(uⁿ) ------------------------
    _explicit_tendencies!(sim, γ * Δt; body_force! = body_force!,
                          momentum_advection = momentum_advection,
                          sgs_transpose = sgsT, scalar_tensor = Ks)
    _copy_tendencies!(ws.aux, sim, names)                 # E(uⁿ)
    for name in names
        f = getfield(sim, name)
        _combine2!(f, one(T), getfield(ws.state0, name),
                      γ * Δt, getfield(ws.aux, name), m)
        _solve_field!(sim, ws, name, f, νm, νs, γ * Δt, tol, maxit)
    end
    post_predictor! === nothing || post_predictor!(sim, γ * Δt)
    _project!(sim, γ * Δt)
    sim.t = t0 + γ * Δt

    # ---- I(U₁) = ∇·(ν∇U₁) on the projected stage value -------------------
    fill_halos!(sim.u, sim.bcs); fill_halos!(sim.v, sim.bcs); fill_halos!(sim.w, sim.bcs)
    fill_halos!(sim.θ, sim.θ_bcs)
    sim.has_moisture && fill_halos!(sim.qv, homogeneous_bcs(sim.θ_bcs))
    for name in names
        _diffusion!(_diffusion_buffer(sim, ws, name), getfield(sim, name),
                    _coef_for(νm, νs, name), m)
    end

    # ---- stage 2 --------------------------------------------------------
    _explicit_tendencies!(sim, Δt; body_force! = body_force!,
                          momentum_advection = momentum_advection,
                          sgs_transpose = sgsT, scalar_tensor = Ks)   # E(U₁) → sim.t*
    for name in names
        f = getfield(sim, name)
        _combine4!(f, one(T), getfield(ws.state0, name),
                      δ * Δt, getfield(ws.aux, name),           # E(uⁿ)
                      (one(T) - δ) * Δt, _tendency(sim, name),  # E(U₁)
                      (one(T) - γ) * Δt, _diffusion_buffer(sim, ws, name), m)
        _solve_field!(sim, ws, name, f, νm, νs, γ * Δt, tol, maxit)
    end
    post_predictor! === nothing || post_predictor!(sim, Δt)
    _project!(sim, Δt)
    sim.t = t0 + Δt
    closure === nothing || advance_sgs!(closure, sim, Δt)   # once per step

    return sim
end

function step_imex!(sim::BoussinesqSimulation{T}, dt::Real, scheme::Rosenbrock2;
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
        _combine2!(k1, zero(T), k1, Δt, _tendency(sim, name), m)
        _solve_field!(sim, ws, name, k1, νm, νs, γ * Δt, tol, maxit; increment = true)
    end

    # ---- stage state uⁿ + a₂₁k₁, projected ------------------------------
    for name in names
        _combine2!(getfield(sim, name), one(T), getfield(ws.state0, name),
                   a21, getfield(ws.aux, name), m)
    end
    post_predictor! === nothing || post_predictor!(sim, a21 * Δt)
    _project!(sim, a21 * Δt)
    sim.t = t0 + a21 * Δt

    # ---- k₂: (I - γΔt J) k₂ = Δt F(u+k₁) + c₂₁ k₁ ------------------------
    _explicit_tendencies!(sim, Δt; body_force! = body_force!,
                          momentum_advection = momentum_advection,
                          sgs_transpose = sgsT, scalar_tensor = Ks)
    _add_diffusion_tendencies!(sim, ws, νm, νs, names)
    for name in names
        k2 = getfield(k2set, name)
        _combine2!(k2, Δt, _tendency(sim, name), c21, getfield(ws.aux, name), m)
        _solve_field!(sim, ws, name, k2, νm, νs, γ * Δt, tol, maxit; increment = true)
    end

    # ---- update ---------------------------------------------------------
    for name in names
        _combine3!(getfield(sim, name), one(T), getfield(ws.state0, name),
                   b1, getfield(ws.aux, name), b2, getfield(k2set, name), m)
    end
    post_predictor! === nothing || post_predictor!(sim, Δt)
    _project!(sim, Δt)
    sim.t = t0 + Δt
    closure === nothing || advance_sgs!(closure, sim, Δt)   # once per step

    return sim
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
