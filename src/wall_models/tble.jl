# Non-equilibrium wall model (Phase 57, P57-c, #189): the thin-boundary-
# layer equations (TBLE) integrated on an embedded wall-normal grid under
# every wall column (Balaras, Benocci & Piomelli 1996; Wang & Moin 2002;
# Kawai & Larsson 2012; Park & Moin 2014). The input (the LES velocity at
# the matching height) is P57-b's `MatchingInput`; the stress is applied
# by the same implicit face loop as the equilibrium `WallModel`
# (`_apply_coefficients!`, operator.jl). Nothing of either is restated.

export AbstractTBLEEddyViscosity, DampedMixingLength, NoEddyViscosity, tble_eddy_viscosity,
       TBLEColumn, tble_grid, tble_solve!, tble_column_momentum,
       TBLEWallModel, update_wall_model!, tble_kappa, tble_columns

# =========================================================================
# The eddy viscosity of the wall-model layer (a generic)
# =========================================================================

"""
    AbstractTBLEEddyViscosity

The eddy viscosity ``ν_{t,wm}(y)`` of the thin-boundary-layer equations
of [`TBLEWallModel`](@ref). A concrete model implements

    tble_eddy_viscosity(model, y, u_τ, ν, κ, z0) -> ν_t    (m² s⁻¹)

at the distance `y` (m) from the wall, for the friction velocity `u_τ`
of the wall model's own wall stress, the kinematic viscosity `ν`, the
(possibly dynamic) constant `κ` and the column's roughness length `z0`
(m; 0 = smooth). It is a generic so that a later phase (P58, buoyancy)
adds a stability-corrected model by a new subtype, without touching the
solver.
"""
abstract type AbstractTBLEEddyViscosity end

"""
    DampedMixingLength(; A_plus = 17)

The eddy viscosity of Cabot & Moin (2000) and Wang & Moin (2002): the
mixing-length viscosity with van Driest damping,

```math
ν_{t,wm} = κ\\,y\\,u_τ\\,\\left(1 - e^{-y^+/A^+}\\right)^2, \\qquad y^+ = y u_τ/ν,
```

with ``A^+ = 17``. Over a rough wall (`z0 > 0`) it is the undamped
``κ u_τ (y + z_0)``, whose equilibrium solution is the rough log law
``U = (u_τ/κ)\\ln[(y + z_0)/z_0]`` (u = 0 at y = 0).
"""
Base.@kwdef struct DampedMixingLength{T<:AbstractFloat} <: AbstractTBLEEddyViscosity
    A_plus::T = 17.0
end

"""
    NoEddyViscosity()

``ν_{t,wm} ≡ 0``: the laminar boundary-layer equations (the Stokes and
Poiseuille limits of `test/era6_wall_models_tble.jl`).
"""
struct NoEddyViscosity <: AbstractTBLEEddyViscosity end

"""
    tble_eddy_viscosity(model::AbstractTBLEEddyViscosity, y, u_τ, ν, κ, z0) -> ν_t

The eddy viscosity (m² s⁻¹) of the wall-model layer at distance `y`
from the wall; see [`AbstractTBLEEddyViscosity`](@ref).
"""
function tble_eddy_viscosity end

@inline function tble_eddy_viscosity(m::DampedMixingLength, y, u_τ, ν, κ, z0)
    z0 > 0 && return κ * u_τ * (y + z0)
    x = y * u_τ
    x > 0 || return zero(x)
    d = ν > 0 ? -expm1(-x / (ν * m.A_plus)) : one(x)
    return κ * x * d * d
end
@inline tble_eddy_viscosity(::NoEddyViscosity, y, u_τ, ν, κ, z0) = zero(float(y))

# whether ν_t depends on u_τ (if not, one linear solve is the solution)
_needs_utau(::AbstractTBLEEddyViscosity) = true
_needs_utau(::NoEddyViscosity) = false

# =========================================================================
# One column
# =========================================================================

"""
    tble_grid(h, N = 30, r = 1.2) -> y

The `N + 1` nodes ``y_j = h\\,(r^j - 1)/(r^N - 1)``, ``j = 0…N``, of the
wall-normal grid of a TBLE column of height `h`: geometric stretching of
ratio `r` from the wall (``y_0 = 0``) to the matching height
(``y_N = h``); `r = 1` is uniform. The defaults put the first node at
``8.5·10^{-4} h`` (``y^+ = 2.5`` at ``h^+ = 3000``).
"""
function tble_grid(h::Real, N::Integer = 30, r::Real = 1.2)
    N >= 2 || throw(ArgumentError("tble_grid: need N ≥ 2, got $N"))
    h > 0 || throw(ArgumentError("tble_grid: need h > 0, got $h"))
    r > 0 || throw(ArgumentError("tble_grid: need r > 0, got $r"))
    T = float(promote_type(typeof(h), typeof(r)))
    r == 1 && return T[T(h) * j / N for j in 0:N]
    y = T[T(h) * (T(r)^j - 1) / (T(r)^N - 1) for j in 0:N]
    y[end] = T(h)
    return y
end

"""
    TBLEColumn(h; N = 30, r = 1.2, z0 = 0)

One column of the thin-boundary-layer equations: the profiles of the
two wall-parallel velocity components on the grid [`tble_grid`](@ref)`(h,
N, r)` (node 1 is the wall, node `N + 1` the matching height), advanced
by [`tble_solve!`](@ref). Fields: `y`, `u`, `v` (the profiles), `τx`,
`τy` (the kinematic wall stress of the last solve, m² s⁻²), `u_τ`
(``|τ_w|^{1/2}``), `G` (the conductances of the `N` segments,
``G_{j+1/2} = [∫_{y_j}^{y_{j+1}} dy/(ν + ν_t)]^{-1}``, of the last
solve), `cx`, `cy` (the convective tendency per node, m s⁻², used with
`terms = :full`) and `z0` (the roughness length of the column, m).
"""
mutable struct TBLEColumn{T<:AbstractFloat}
    y::Vector{T}
    u::Vector{T}
    v::Vector{T}
    un::Vector{T}            # the profile at the start of the step
    vn::Vector{T}
    G::Vector{T}
    cx::Vector{T}
    cy::Vector{T}
    zv::Vector{T}            # zeros: the convective tendency of the other terms
    τx::T
    τy::T
    u_τ::T
    z0::T
    # tridiagonal work
    cp::Vector{T}
    dx::Vector{T}
    dy::Vector{T}
end

function TBLEColumn(h::Real; N::Integer = 30, r::Real = 1.2, z0::Real = 0.0)
    y = tble_grid(h, N, r)
    T = eltype(y)
    z() = zeros(T, N + 1)
    return TBLEColumn{T}(y, z(), z(), z(), z(), zeros(T, N), z(), z(), z(),
                         zero(T), zero(T), zero(T), T(z0), z(), z(), z())
end

# 4-point Gauss–Legendre on [-1, 1]
const _GL4_X = (-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526)
const _GL4_W = (0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538)

# Conductances G_{j+1/2} = 1/∫ dy/(ν + ν_t) of every segment for the
# friction velocity s. The resistance of a segment is its exact integral
# (to the quadrature), so the steady, unforced solution is the integral of
# the equilibrium ODE at the nodes whatever the grid.
function _conductances!(col::TBLEColumn{T}, eddy, s, ν, κ) where {T}
    y, G = col.y, col.G
    @inbounds for j in 1:length(G)
        a, b = y[j], y[j+1]
        m, hw = (a + b) / 2, (b - a) / 2
        R = zero(T)
        for q in 1:4
            yq = m + hw * T(_GL4_X[q])
            R += T(_GL4_W[q]) / (ν + tble_eddy_viscosity(eddy, yq, s, ν, κ, col.z0))
        end
        G[j] = 1 / (hw * R)
    end
    return G
end

# One linear solve at friction velocity s: backward Euler (or steady) for
# both components, with the profile at the start of the step in un, vn.
# Returns (τx, τy).
function _tble_pass!(col::TBLEColumn{T}, s, Ut, Vt, Fx, Fy, cx, cy, dpx, dpy, Δt, steady,
                     ν, κ, eddy) where {T}
    _conductances!(col, eddy, s, ν, κ)
    y, G, u, v, un, vn = col.y, col.G, col.u, col.v, col.un, col.vn
    cp, dX, dY = col.cp, col.dx, col.dy
    N = length(G)                       # nodes 1…N+1; unknowns 2…N
    u[1] = 0; v[1] = 0; u[N+1] = Ut; v[N+1] = Vt
    # Thomas algorithm, forward sweep over j = 2…N
    @inbounds for j in 2:N
        V = (y[j+1] - y[j-1]) / 2
        c0 = steady ? zero(T) : V / Δt
        lo = -G[j-1]; up = -G[j]
        di = c0 + G[j-1] + G[j]
        rx = c0 * un[j] + V * (Fx - cx[j])
        ry = c0 * vn[j] + V * (Fy - cy[j])
        j == N && (rx -= up * Ut; ry -= up * Vt; up = zero(T))
        if j == 2
            cp[j] = up / di; dX[j] = rx / di; dY[j] = ry / di
        else
            den = di - lo * cp[j-1]
            cp[j] = up / den
            dX[j] = (rx - lo * dX[j-1]) / den
            dY[j] = (ry - lo * dY[j-1]) / den
        end
    end
    u[N] = dX[N]; v[N] = dY[N]
    @inbounds for j in N-1:-1:2
        u[j] = dX[j] - cp[j] * u[j+1]
        v[j] = dY[j] - cp[j] * v[j+1]
    end
    # the wall stress: the flux of the first segment, carried to the wall
    # with the wall value of ∂τ/∂y (= ∂p/∂x: u = 0 and conv = 0 there)
    h1 = y[2] / 2
    return (G[1] * u[2] - h1 * dpx, G[1] * v[2] - h1 * dpy)
end

const _TBLE_TERMS = (:equilibrium_ode, :pressure_gradient, :full)

"""
    tble_solve!(col::TBLEColumn, Ut, Vt, dpdx, dpdy, Δt; ν, κ = 0.4,
                eddy = DampedMixingLength(), terms = :pressure_gradient,
                tol = 1e-10, maxiter = 50) -> (τx, τy, u_τ, iterations)

Advance the column `col` over `Δt` (s) with the thin-boundary-layer
equations

```math
\\frac{∂u_i}{∂t} + \\mathrm{conv}_i = -\\frac{∂p}{∂x_i}
  + \\frac{∂}{∂y}\\left[(ν + ν_{t,wm})\\frac{∂u_i}{∂y}\\right],
\\qquad u_i(0) = 0,\\quad u_i(h) = (U_t, V_t)_i,
```

`dpdx`, `dpdy` the kinematic pressure gradient (m s⁻²) and `ν`, `κ`,
`eddy` those of ``ν_{t,wm}`` ([`tble_eddy_viscosity`](@ref)).
`terms`:

  * `:equilibrium_ode` — the steady, unforced equation
    ``∂_y[(ν+ν_t)∂_y u_i] = 0`` (``τ`` constant across the layer), the
    ODE of the equilibrium models, solved directly (`Δt` and the pressure
    gradient are ignored);
  * `:pressure_gradient` — the unsteady equation with the pressure
    gradient and no convection;
  * `:full` — plus the convective tendency `col.cx`, `col.cy`, set by
    the caller ([`update_wall_model!`](@ref) computes it from the
    neighbouring columns, Park & Moin 2014).

`Δt = Inf` gives the steady state of `:pressure_gradient` / `:full`.

**Discretisation.** Finite volumes on the stretched nodes, backward
Euler in time (L-stable: any `Δt`), a tridiagonal (Thomas) solve per
component. The flux of segment ``[y_j, y_{j+1}]`` is
``G_{j+1/2}(u_{j+1} - u_j)`` with the conductance
``G = [∫ dy/(ν + ν_t)]^{-1}`` integrated exactly (4-point Gauss–Legendre),
so the unforced steady solution is the quadrature of the equilibrium ODE
at every node whatever the grid, and with ``ν_t ≡ 0`` the scheme is
exact for quadratic profiles (Poiseuille). The wall stress is the flux of
the first segment carried to the wall with the wall value of
``∂τ/∂y = ∂p/∂x``: ``τ_{w} = G_{1/2}u_1 - (y_1/2)\\,∂p/∂x``, so in
steady state ``τ(y) = τ_w + y\\,∂p/∂x`` at every segment midpoint.

**Momentum.** Over the control volumes ``V_j = (y_{j+1} - y_{j-1})/2``
of the interior nodes the step conserves momentum exactly:
``Σ V_j (u_j^{n+1} - u_j^n)/Δt = τ_{top} - G_{1/2}u_1 + Σ V_j F_j``
(``τ_{top}`` the flux of the last segment; [`tble_column_momentum`](@ref)).

**Nonlinearity.** ``ν_{t,wm}`` depends on the column's own ``u_τ``; the
step is iterated to ``|Δu_τ| ≤ `` `tol`·``u_τ`` by a secant iteration on
``u_τ = |τ_w(u_τ)|^{1/2}`` (falling back to the fixed-point step), each
iterate a full implicit solve from the start-of-step profile.

# References

  * Balaras, E., Benocci, C. & Piomelli, U. (1996). Two-layer
    approximate boundary conditions for large-eddy simulations.
    *AIAA J.*, 34, 1111-1119.
  * Wang, M. & Moin, P. (2002). Dynamic wall modeling for large-eddy
    simulation of complex turbulent flows. *Phys. Fluids*, 14, 2043-2051.
  * Kawai, S. & Larsson, J. (2012). Wall-modeling in large eddy
    simulation: length scales, grid resolution, and accuracy.
    *Phys. Fluids*, 24, 015105.
  * Park, G. I. & Moin, P. (2014). An improved dynamic non-equilibrium
    wall-model for large eddy simulation. *Phys. Fluids*, 26, 015108.
"""
function tble_solve!(col::TBLEColumn{T}, Ut::Real, Vt::Real, dpdx::Real, dpdy::Real, Δt::Real;
                     ν::Real, κ::Real = 0.4, eddy::AbstractTBLEEddyViscosity = DampedMixingLength(),
                     terms::Symbol = :pressure_gradient, tol::Real = 1e-10,
                     maxiter::Integer = 50) where {T}
    terms in _TBLE_TERMS ||
        throw(ArgumentError("tble_solve!: terms must be :equilibrium_ode, :pressure_gradient or :full, got :$terms"))
    eq = terms === :equilibrium_ode
    steady = eq || isinf(Δt)
    steady || Δt > 0 || throw(ArgumentError("tble_solve!: Δt must be > 0, got $Δt"))
    dpx = eq ? zero(T) : T(dpdx); dpy = eq ? zero(T) : T(dpdy)
    Fx, Fy = -dpx, -dpy
    cx = terms === :full ? col.cx : col.zv
    cy = terms === :full ? col.cy : col.zv
    copyto!(col.un, col.u); copyto!(col.vn, col.v)
    Ut, Vt = T(Ut), T(Vt); νT, κT = T(ν), T(κ)
    pass(s) = _tble_pass!(col, s, Ut, Vt, Fx, Fy, cx, cy, dpx, dpy, T(Δt), steady, νT, κT, eddy)
    if !_needs_utau(eddy)
        τx, τy = pass(zero(T))
        return _store!(col, τx, τy, 1)
    end
    # initial guess: the previous u_τ, else the larger of the laminar and a
    # log-law estimate
    U = hypot(Ut, Vt); h = col.y[end]
    s0 = col.u_τ > 0 ? col.u_τ : max(sqrt(νT * U / h), κT * U / log(max(h * U / max(νT, eps(T)), T(10))),
                                     sqrt(sqrt(dpx^2 + dpy^2) * h), eps(T))
    τx, τy = pass(s0); r0 = sqrt(hypot(τx, τy)) - s0
    it = 1
    abs(r0) <= tol * s0 && return _store!(col, τx, τy, it)
    s1 = s0 + r0
    while it < maxiter
        τx, τy = pass(s1); it += 1
        r1 = sqrt(hypot(τx, τy)) - s1
        (abs(r1) <= tol * max(s1, eps(T)) || s1 == 0) && break
        sn = r1 != r0 ? s1 - r1 * (s1 - s0) / (r1 - r0) : s1 + r1
        (isfinite(sn) && sn > 0) || (sn = s1 + r1)          # the fixed-point step
        s0, r0, s1 = s1, r1, sn
    end
    return _store!(col, τx, τy, it)
end

function _store!(col::TBLEColumn, τx, τy, it)
    col.τx = τx; col.τy = τy; col.u_τ = sqrt(hypot(τx, τy))
    return (τx = τx, τy = τy, u_τ = col.u_τ, iterations = it)
end

"""
    tble_column_momentum(col::TBLEColumn) -> (Mx, My, τtop_x, τtop_y, τ1_x, τ1_y)

The column momentum ``M_i = Σ_j V_j u_{i,j}`` over the control volumes
``V_j = (y_{j+1} - y_{j-1})/2`` of the interior nodes, the flux of the
top segment ``τ_{top} = G_{N-1/2}(u_{N} - u_{N-1})`` and that of the
first segment ``τ_1 = G_{1/2}u_1``, all of the last solve: the terms of
the exact step budget ``ΔM_i/Δt = τ_{top} - τ_1 + F_i Σ V_j`` of
[`tble_solve!`](@ref).
"""
function tble_column_momentum(col::TBLEColumn{T}) where {T}
    y, u, v, G = col.y, col.u, col.v, col.G
    N = length(G)
    Mx = My = zero(T)
    @inbounds for j in 2:N
        V = (y[j+1] - y[j-1]) / 2
        Mx += V * u[j]; My += V * v[j]
    end
    return (Mx = Mx, My = My, τtop_x = G[N] * (u[N+1] - u[N]), τtop_y = G[N] * (v[N+1] - v[N]),
            τ1_x = G[1] * u[2], τ1_y = G[1] * v[2])
end

# =========================================================================
# The wall model: one column per wall column, plugged into the operator
# =========================================================================

mutable struct _TBLEWall{T}
    cols::Matrix{TBLEColumn{T}}
    active::Matrix{Bool}
    input::MatchingInput            # the default input (first cell, unfiltered) of this wall
    κ::T                            # κ_wm of this wall, used by the columns
    κ_dyn::T                        # the relaxed dynamic κ_wm (= κ with feedback)
    κ_target::T                     # the last instantaneous dynamic target (NaN: none yet)
    dpdx::Matrix{T}                 # the pressure gradient of the last update (m s⁻²)
    dpdy::Matrix{T}
    nupdates::Int
end

"""
    TBLEWallModel(; ν = 1.5e-5, κ = SurfaceExchange.von_karman(),
                  eddy = DampedMixingLength(), terms = :pressure_gradient,
                  N = 30, r = 1.2, z0 = 0.0, pressure = true,
                  dynamic_kappa = :off, T_κ = 0.0, κ_range = (0.05, 0.6),
                  tol = 1e-10, maxiter = 50)

The non-equilibrium wall model: a [`TBLEColumn`](@ref) under every wall
column, integrated with [`tble_solve!`](@ref) (Balaras, Benocci &
Piomelli 1996; Wang & Moin 2002; Kawai & Larsson 2012; Park & Moin 2014)
between the wall (``u = 0``) and the matching height, where the profile
equals the LES velocity sampled by P57-b's [`MatchingInput`](@ref)
(matching height, time and wall-parallel filters). An
[`AbstractWallModel`](@ref): [`apply_wall_stress!`](@ref) applies the
columns' wall stress with the operator's implicit face loop.

**Use.** Per step, `apply_wall_stress!(sim, wm, Δt; side, mask, input)`
as the `post_predictor!` hook (every substage applies the stress of the
columns' current state, implicitly) and, once per accepted step after
the projection, `update_wall_model!(sim … )` to advance the columns to the
end of the step with the end-of-step LES velocity and pressure. The
first call of either primes the columns with the equilibrium solution
(`:equilibrium_ode`) for the current LES velocity. State is kept per
wall (`:bottom`, `:top`) and per column.

**Stress direction.** The columns' stress ``\\boldsymbol τ_w`` need not
point along the resolved first-cell velocity ``\\mathbf u`` (a pressure
gradient turns it). It is applied as ``τ_{w,i} = a u_i + b_i`` with the
implicit part ``a = \\max(\\boldsymbol τ_w·\\hat{\\mathbf u}, 0)/U`` and the
explicit rest ``b_i = τ_{w,i} - a u_i`` (the linear form of the Moeng
formulation), so a stress along the flow keeps the unconditional
stability of the equilibrium operator.

**Keywords.** `ν` (m² s⁻¹), `κ` (the initial or fixed ``κ_{wm}``,
default `SurfaceExchange.von_karman()`), `eddy`
([`AbstractTBLEEddyViscosity`](@ref)), `terms` (see
[`tble_solve!`](@ref)), `N` and `r` (the grid, [`tble_grid`](@ref)), `z0`
(the roughness length, m, a number or an `(Nx, Ny)` matrix; 0 smooth),
`pressure` (add the resolved pressure gradient ``∂p/∂x_i`` of `sim.p` at
the matching cell; a gradient across a solid neighbour is one-sided).

**Dynamic κ** (Wang & Moin 2002). `dynamic_kappa = :total` sets
``κ_{wm}`` of each wall so that the TBLE eddy viscosity at the matching
height equals the LES's effective eddy viscosity there: the plane-mean
resolved plus SGS stress (the momentum flux the TBLE eddy viscosity
stands for) over the resolved mean shear, both projected on the mean
wall-stress direction ``\\hat e``:

```math
κ_{wm} = \\frac{⟨-u'w' + τ_{sgs}⟩(h)·\\hat e}
              {(∂_z⟨\\mathbf u⟩(h)·\\hat e)\\;⟨h\\,u_τ D(h^+)^2⟩}.
```

`:sgs` matches the SGS viscosity alone, ``ν_{t,wm}(h) = ⟨ν_{sgs}⟩(h)``
(Wang & Moin's original form: the resolved eddies are carried by the
convective terms of the full TBLE). `:off` keeps `κ`. The target is
relaxed with the exponential filter of time scale `T_κ` (s; 0
instantaneous) and clamped to `κ_range`; `κ_feedback = false` computes
and relaxes it ([`tble_kappa`](@ref)`(…; dynamic = true)`) without
handing it to the columns. Plane walls only (`mask = nothing`).

*Matching the stress itself* (``⟨ν_{t,wm}∂_y u⟩(h)`` = LES stress) does
not determine κ: in a statistically steady flow both sides equal
``τ_w + h\\,∂p/∂x`` by the momentum balance whatever κ is (the first
version did this and drifted to the lower clamp). Measured in the
Re_τ = 1000 channel (`test/era6_wall_models_tble.jl`): the effective
LES viscosity at ``h = 0.375δ`` gives κ_wm ≈ 0.23, close to
``0.41(1 - h/δ) = 0.26`` (the outer-layer decrease of the stress that
the mixing-length ``ν_{t,wm}`` lacks) and to Wang & Moin's ≈ 0.2. Fed
back into `:pressure_gradient` (which has no convective terms to carry
the difference) it lowers ``τ_w`` near the wall too; see the theory
page.

**Not modelled.** No Monin–Obukhov stability correction is carried into
the TBLE eddy viscosity (neutral; P58 adds buoyancy through
[`AbstractTBLEEddyViscosity`](@ref)): over the Berlin ground the
non-equilibrium option is neutral where the equilibrium `MOSTWallLaw` is
stability-corrected.

# References

  * Balaras, E., Benocci, C. & Piomelli, U. (1996). *AIAA J.*, 34,
    1111-1119.
  * Cabot, W. & Moin, P. (2000). Approximate wall boundary conditions in
    the large-eddy simulation of high Reynolds number flow. *Flow Turbul.
    Combust.*, 63, 269-291.
  * Wang, M. & Moin, P. (2002). *Phys. Fluids*, 14, 2043-2051.
  * Kawai, S. & Larsson, J. (2012). *Phys. Fluids*, 24, 015105.
  * Park, G. I. & Moin, P. (2014). *Phys. Fluids*, 26, 015108.
"""
mutable struct TBLEWallModel{E<:AbstractTBLEEddyViscosity, Z} <: AbstractWallModel
    ν::Float64
    κ::Float64
    eddy::E
    terms::Symbol
    N::Int
    r::Float64
    z0::Z
    pressure::Bool
    dynamic_kappa::Symbol
    κ_feedback::Bool
    T_κ::Float64
    κ_range::Tuple{Float64, Float64}
    tol::Float64
    maxiter::Int
    walls::Dict{Symbol, _TBLEWall{Float64}}
end

function TBLEWallModel(; ν::Real = 1.5e-5, κ::Real = von_karman(),
                       eddy::AbstractTBLEEddyViscosity = DampedMixingLength(),
                       terms::Symbol = :pressure_gradient, N::Integer = 30, r::Real = 1.2,
                       z0 = 0.0, pressure::Bool = true, dynamic_kappa::Symbol = :off,
                       κ_feedback::Bool = true, T_κ::Real = 0.0, κ_range = (0.05, 0.6), tol::Real = 1e-10,
                       maxiter::Integer = 50)
    terms in _TBLE_TERMS ||
        throw(ArgumentError("TBLEWallModel: terms must be :equilibrium_ode, :pressure_gradient or :full, got :$terms"))
    dynamic_kappa in (:off, :total, :sgs) ||
        throw(ArgumentError("TBLEWallModel: dynamic_kappa must be :off, :total or :sgs, got :$dynamic_kappa"))
    ν > 0 || throw(ArgumentError("TBLEWallModel: ν must be > 0, got $ν"))
    T_κ >= 0 || throw(ArgumentError("TBLEWallModel: T_κ must be ≥ 0, got $T_κ"))
    z = z0 isa Real ? float(z0) : Matrix{Float64}(z0)
    all(>=(0), z) || throw(ArgumentError("TBLEWallModel: z0 must be ≥ 0"))
    return TBLEWallModel(float(ν), float(κ), eddy, terms, Int(N), float(r), z, pressure,
                         dynamic_kappa, κ_feedback, float(T_κ), (float(κ_range[1]), float(κ_range[2])),
                         float(tol), Int(maxiter), Dict{Symbol, _TBLEWall{Float64}}())
end

@inline _z0(z::Real, i, j) = z
@inline _z0(z::AbstractMatrix, i, j) = @inbounds z[i, j]

"""
    tble_kappa(wm::TBLEWallModel, side = :bottom; dynamic = false) -> κ_wm

The ``κ_{wm}`` the columns of the wall `side` use (the dynamic value
with `dynamic_kappa ≠ :off` and `κ_feedback`; the constant otherwise,
also before the wall's first use). `dynamic = true` returns the relaxed
dynamic value whether or not it is fed back.
"""
function tble_kappa(wm::TBLEWallModel, side::Symbol = :bottom; dynamic::Bool = false)
    haskey(wm.walls, side) || return wm.κ
    w = wm.walls[side]
    return dynamic ? w.κ_dyn : w.κ
end

"""
    tble_columns(wm::TBLEWallModel, side = :bottom) -> (cols, active)

The [`TBLEColumn`](@ref)s of the wall `side` (an `(Nx, Ny)` matrix) and
the matrix of the columns that have a wall.
"""
function tble_columns(wm::TBLEWallModel, side::Symbol = :bottom)
    haskey(wm.walls, side) || throw(ArgumentError("tble_columns: the wall :$side has not been used yet"))
    w = wm.walls[side]
    return (cols = w.cols, active = w.active)
end

# the LES velocity at the matching point: (u, v, h) per column (filtered
# when the input has a primed time filter)
function _tble_input(w::_TBLEWall, sim, side, mask, input)
    mi = input === nothing ? w.input : input
    uc, vc, hh = _sample_matching(mi, sim, side, mask)
    if mi.T_f > 0
        mi.primed || _prime!(mi, uc, vc)
        return mi.uf, mi.vf, hh
    end
    return uc, vc, hh
end

function _wall_state!(wm::TBLEWallModel, sim, side, mask, input)
    w = get(wm.walls, side, nothing)
    if w === nothing
        g = sim.mesh
        Nx, Ny = g.Nx, g.Ny
        active = [_wall_level(side, mask, i, j, g.Nz) != 0 for i in 1:Nx, j in 1:Ny]
        cols = [TBLEColumn(g.dz / 2; N = wm.N, r = wm.r, z0 = _z0(wm.z0, i, j))
                for i in 1:Nx, j in 1:Ny]
        w = _TBLEWall{Float64}(cols, active, MatchingInput(), wm.κ, wm.κ, NaN,
                               zeros(Nx, Ny), zeros(Nx, Ny), 0)
        wm.walls[side] = w
        _prime_columns!(wm, w, _tble_input(w, sim, side, mask, input)...; everything = true)
    end
    return w
end

# the equilibrium solution for the current input in every column (all),
# or only in the columns whose matching height changed
function _prime_columns!(wm, w, uin, vin, hh; everything::Bool = false)
    @inbounds for j in axes(w.cols, 2), i in axes(w.cols, 1)
        w.active[i, j] || continue
        col = w.cols[i, j]
        h = hh[i, j]
        regrid = abs(col.y[end] - h) > 1e-12 * h
        (everything || regrid) || continue
        regrid && (w.cols[i, j] = col = TBLEColumn(h; N = wm.N, r = wm.r, z0 = col.z0))
        tble_solve!(col, uin[i, j], vin[i, j], 0.0, 0.0, Inf; ν = wm.ν, κ = w.κ,
                    eddy = wm.eddy, terms = :equilibrium_ode, tol = wm.tol, maxiter = wm.maxiter)
    end
    return w
end

"""
    apply_wall_stress!(sim, wm::TBLEWallModel, Δt; side = :bottom, mask = nothing,
                       input = nothing, formulation = :local,
                       ustar = nothing, wind_out = nothing) -> (Fx, Fy)

Apply the wall stress of the TBLE columns of `wm` (their state after the
last [`update_wall_model!`](@ref); primed with the equilibrium solution
on first use) with the implicit face loop of the equilibrium operator:
per column ``τ_{w,i} = a u_i + b_i``, ``a = \\max(\\boldsymbol τ_w·\\hat{\\mathbf
u}, 0)/U`` along the resolved first-cell velocity, ``b_i`` the rest
(see [`TBLEWallModel`](@ref)). `side`, `mask` and the return value are
those of the equilibrium method; `input` is the [`MatchingInput`](@ref)
whose velocity primes the columns; `ustar` / `wind_out` receive each
column's ``u_τ`` and the speed at its matching point.
"""
function apply_wall_stress!(sim, wm::TBLEWallModel, Δt::Real; side::Symbol = :bottom,
                            mask = nothing, input = nothing, formulation::Symbol = :local,
                            ustar = nothing, wind_out = nothing)
    side in (:bottom, :top, :both) ||
        throw(ArgumentError("apply_wall_stress!: side must be :bottom, :top or :both, got :$side"))
    formulation === :local ||
        throw(ArgumentError("apply_wall_stress!: a TBLEWallModel is local per column (formulation = :local), got :$formulation"))
    side === :both && (ustar !== nothing || wind_out !== nothing || input !== nothing) &&
        throw(ArgumentError("apply_wall_stress!: ustar / wind_out / input take one side, not :both"))
    fill_halos!(sim.u, sim.bcs)
    fill_halos!(sim.v, sim.bcs)
    mesh = sim.mesh
    Fx = Fy = zero(float(mesh.dz))
    for (s, kw) in ((:bottom, 1), (:top, mesh.Nz))
        (side === s || side === :both) || continue
        w = _wall_state!(wm, sim, s, mask, input)
        fx, fy = _tble_side!(sim, w, Δt, kw, mask, ustar, wind_out)
        Fx += fx; Fy += fy
    end
    return (Fx = Fx, Fy = Fy)
end

function _tble_side!(sim, w::_TBLEWall, Δt, kw, mask, ustar, wind_out)
    mesh = sim.mesh
    Nx, Ny, Nz, Hx, Hy, Hz = mesh.Nx, mesh.Ny, mesh.Nz, mesh.Hx, mesh.Hy, mesh.Hz
    T = typeof(float(mesh.dz))
    u, v = sim.u.data, sim.v.data
    a = zeros(T, Nx, Ny); bx = zeros(T, Nx, Ny); by = zeros(T, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        k = _level(mask, i, j, kw, Nz)
        (k == 0 || !w.active[i, j]) && continue
        I, J, K = Hx + i, Hy + j, Hz + k
        uc = (u[I, J, K] + u[I+1, J, K]) / 2
        vc = (v[I, J, K] + v[I, J+1, K]) / 2
        U = sqrt(uc^2 + vc^2)
        col = w.cols[i, j]
        τx, τy = col.τx, col.τy
        if U > 0
            a[i, j] = max(τx * uc + τy * vc, zero(T)) / U^2      # (τ·û)/U
        end
        bx[i, j] = τx - a[i, j] * uc
        by[i, j] = τy - a[i, j] * vc
        ustar === nothing || (ustar[i, j] = col.u_τ)
        wind_out === nothing || (wind_out[i, j] = hypot(col.u[end], col.v[end]))
    end
    return _apply_coefficients!(sim, a, bx, by, Δt, kw, mask)
end

"""
    update_wall_model!(wm::TBLEWallModel, sim, Δt; side = :bottom, mask = nothing,
                       input = nothing, dpdx = 0, dpdy = 0) -> wm

Advance the TBLE columns of `wm` at the wall `side` (`:bottom`, `:top`
or `:both`) over the accepted step `Δt` (call once per step, after the
projection, and after `update_input_filter!` of a time-filtered
`input`): per column [`tble_solve!`](@ref) with the LES velocity at the
matching point of `input` (a [`MatchingInput`](@ref); `nothing` is the
first cell, unfiltered) as the upper boundary condition and the pressure
gradient `dpdx`, `dpdy` (m s⁻², a uniform imposed part such as the mean
pressure gradient ``-G`` of a forced channel) plus, with `pressure =
true`, the resolved gradient of `sim.p` at the matching cell. With
`terms = :full` the convective tendency ``u ∂_x u_i + v ∂_y u_i + w ∂_n
u_i`` is formed from the neighbouring columns' profiles (first-order
upwind differences between active neighbours of equal matching height —
explicit centred advection would be unconditionally unstable; ``w`` from
continuity, ``w = -∫_0^y (∂_x u + ∂_y v)\\,dy``), as in Park & Moin
(2014). Then the dynamic ``κ_{wm}`` is updated (see
[`TBLEWallModel`](@ref)).
"""
function update_wall_model!(wm::TBLEWallModel, sim, Δt::Real; side::Symbol = :bottom,
                            mask = nothing, input = nothing, dpdx::Real = 0.0, dpdy::Real = 0.0)
    side in (:bottom, :top, :both) ||
        throw(ArgumentError("update_wall_model!: side must be :bottom, :top or :both, got :$side"))
    side === :both && input !== nothing &&
        throw(ArgumentError("update_wall_model!: input takes one side, not :both"))
    for s in (:bottom, :top)
        (side === s || side === :both) || continue
        w = _wall_state!(wm, sim, s, mask, input)
        uin, vin, hh = _tble_input(w, sim, s, mask, input)
        _prime_columns!(wm, w, uin, vin, hh)                 # a changed matching height only
        _pressure_gradients!(w, sim, s, mask, hh, wm.pressure, dpdx, dpdy)
        wm.terms === :full && _convection!(w, sim.mesh, sim.bcs)
        @inbounds for j in axes(w.cols, 2), i in axes(w.cols, 1)
            w.active[i, j] || continue
            tble_solve!(w.cols[i, j], uin[i, j], vin[i, j], w.dpdx[i, j], w.dpdy[i, j], Δt;
                        ν = wm.ν, κ = w.κ, eddy = wm.eddy, terms = wm.terms,
                        tol = wm.tol, maxiter = wm.maxiter)
        end
        wm.dynamic_kappa === :off || _dynamic_kappa!(wm, w, sim, s, mask, Δt)
        w.nupdates += 1
    end
    return wm
end

# The kinematic pressure gradient at the matching cell of every column:
# centred where both neighbours at that level are air, one-sided where one
# is, zero where neither is; plus the imposed part.
function _pressure_gradients!(w, sim, side, mask, hh, resolved, dpdx, dpdy)
    g = sim.mesh
    Nx, Ny, Nz, Hx, Hy, Hz, dz = g.Nx, g.Ny, g.Nz, g.Hx, g.Hy, g.Hz, g.dz
    p = sim.p.data
    s = side === :bottom ? 1 : -1
    fill!(w.dpdx, dpdx); fill!(w.dpdy, dpdy)
    resolved || return w
    air(i, j, k) = (1 <= i <= Nx && 1 <= j <= Ny) &&
                   (mask isa AbstractMatrix{<:Integer} && !(mask isa AbstractMatrix{Bool}) ?
                    (1 <= mask[i, j] <= k) : true)
    # the one BC-aware neighbour index (#262): wrapped across a periodic
    # side, 0 (no neighbour: not air) across any other
    wrapx(i) = neighbor_index(g, sim.bcs, i, 1; outside = :none)
    wrapy(j) = neighbor_index(g, sim.bcs, j, 2; outside = :none)
    @inbounds for j in 1:Ny, i in 1:Nx
        w.active[i, j] || continue
        kw = _wall_level(side, mask, i, j, Nz)
        n = clamp(floor(Int, hh[i, j] / dz + 0.5 + 1e-9), 1, Nz)
        k = clamp(kw + s * (n - 1), 1, Nz)
        K = Hz + k
        pc = p[Hx+i, Hy+j, K]
        for (d, out, sp) in ((1, w.dpdx, g.dx), (2, w.dpdy, g.dy))
            ip, jp = d == 1 ? (wrapx(i + 1), j) : (i, wrapy(j + 1))
            im, jm = d == 1 ? (wrapx(i - 1), j) : (i, wrapy(j - 1))
            ap, am = air(ip, jp, k), air(im, jm, k)
            gr = if ap && am
                (p[Hx+ip, Hy+jp, K] - p[Hx+im, Hy+jm, K]) / (2sp)
            elseif ap
                (p[Hx+ip, Hy+jp, K] - pc) / sp
            elseif am
                (pc - p[Hx+im, Hy+jm, K]) / sp
            else
                0.0
            end
            out[i, j] += gr
        end
    end
    return w
end

# The convective tendency of :full from the neighbouring columns (Park &
# Moin 2014): first-order upwind differences in x and y between active
# neighbours with the same grid (else that derivative is 0; explicit and
# centred would be unconditionally unstable), w from continuity (centred
# divergence), upwind in the wall-normal direction.
function _convection!(w, g, bcs)
    Nx, Ny = size(w.cols)
    nb(i, j) = (1 <= i <= Nx && 1 <= j <= Ny) ? (w.active[i, j] ? w.cols[i, j] : nothing) : nothing
    # the one BC-aware neighbour index (#262); 0 = none (nb gives nothing)
    wrapx(i) = neighbor_index(g, bcs, i, 1; outside = :none)
    wrapy(j) = neighbor_index(g, bcs, j, 2; outside = :none)
    same(c, d) = d !== nothing && length(d.y) == length(c.y) && d.y[end] == c.y[end]
    # upwind derivative of the profile `f` (:u or :v) at node m along a
    # direction with neighbours P (+) and M (−) and spacing d, advected at `a`
    @inline upw(a, fc, fP, fM, d) = a > 0 ? (fc - fM) / d : (fP - fc) / d
    @inbounds for j in 1:Ny, i in 1:Nx
        w.active[i, j] || continue
        c = w.cols[i, j]
        E, W = nb(wrapx(i + 1), j), nb(wrapx(i - 1), j)
        Nn, S = nb(i, wrapy(j + 1)), nb(i, wrapy(j - 1))
        hx = same(c, E) && same(c, W); hy = same(c, Nn) && same(c, S)
        y = c.y; M = length(y)
        div_prev = 0.0; wv = 0.0
        c.cx[1] = 0; c.cy[1] = 0; c.cx[M] = 0; c.cy[M] = 0
        for m in 1:M
            dv = (hx ? (E.u[m] - W.u[m]) / (2g.dx) : 0.0) + (hy ? (Nn.v[m] - S.v[m]) / (2g.dy) : 0.0)
            m > 1 && (wv -= (div_prev + dv) / 2 * (y[m] - y[m-1]))
            div_prev = dv
            (m == 1 || m == M) && continue
            uc, vc = c.u[m], c.v[m]
            ax = hx ? uc * upw(uc, uc, E.u[m], W.u[m], g.dx) : 0.0
            ay = hy ? vc * upw(vc, uc, Nn.u[m], S.u[m], g.dy) : 0.0
            bx = hx ? uc * upw(uc, vc, E.v[m], W.v[m], g.dx) : 0.0
            by = hy ? vc * upw(vc, vc, Nn.v[m], S.v[m], g.dy) : 0.0
            dn = wv > 0 ? y[m] - y[m-1] : y[m+1] - y[m]
            an = wv * (wv > 0 ? (uc - c.u[m-1]) : (c.u[m+1] - uc)) / dn
            bn = wv * (wv > 0 ? (vc - c.v[m-1]) : (c.v[m+1] - vc)) / dn
            c.cx[m] = ax + ay + an
            c.cy[m] = bx + by + bn
        end
    end
    return w
end

# Dynamic κ_wm (Wang & Moin 2002): the TBLE eddy viscosity at h equals the
# LES's effective one there, (resolved [+] SGS stress)/(resolved mean
# shear), along the mean wall stress, in the plane mean.
function _dynamic_kappa!(wm, w, sim, side, mask, Δt)
    mask === nothing ||
        throw(ArgumentError("TBLEWallModel: the dynamic κ is for a plane wall (mask = nothing)"))
    g = sim.mesh
    fill_halos!(sim.u, sim.bcs); fill_halos!(sim.v, sim.bcs); fill_halos!(sim.w, sim.bcs)
    fill_halos!(sim.νt, sim.bcs)
    px = total_stress_profile(sim.u, sim.v, sim.w, sim.νt; ν = 0, component = :x)
    py = total_stress_profile(sim.u, sim.v, sim.w, sim.νt; ν = 0, component = :y)
    les(p) = wm.dynamic_kappa === :total ? p.resolved .+ p.sgs : p.sgs
    sx, sy = les(px), les(py)
    um, vm = _uv_means(sim)
    # the matching height (uniform on a plane wall), as a face coordinate
    c1 = w.cols[findfirst(w.active)]
    h = c1.y[end]
    z = side === :bottom ? h : g.Lz - h
    x = z / g.dz + 1                        # face index (face k at (k − 1)Δz)
    k = clamp(floor(Int, x), 2, g.Nz)
    f = x - k
    interp(a) = (1 - f) * a[k] + f * a[k+1]
    # the mean shear on the faces k and k + 1 (the wall faces take the next one)
    shear(m) = (kk = clamp(k, 2, g.Nz); kp = clamp(k + 1, 2, g.Nz);
                (1 - f) * (m[kk] - m[kk-1]) / g.dz + f * (m[kp] - m[kp-1]) / g.dz)
    sgn = side === :bottom ? 1.0 : -1.0     # wall coordinates: y away from the wall
    τLx, τLy = sgn * interp(sx), sgn * interp(sy)
    Sx, Sy = sgn * shear(um), sgn * shear(vm)
    # mean wall-stress direction
    tx = ty = 0.0; n = 0; fm = 0.0
    for c in w.cols[w.active]
        tx += c.τx; ty += c.τy; n += 1
        fm += tble_eddy_viscosity(wm.eddy, h, c.u_τ, wm.ν, 1.0, c.z0)   # h u_τ D(h⁺)²
    end
    tm = hypot(tx, ty)
    tm > 0 || return w
    ex, ey = tx / tm, ty / tm
    S = Sx * ex + Sy * ey
    target = (τLx * ex + τLy * ey) / (S * fm / n)
    (S > 0 && fm > 0 && isfinite(target)) || return w
    w.κ_target = target
    target = clamp(target, wm.κ_range[1], wm.κ_range[2])
    w.κ_dyn += exponential_filter_weight(Δt, wm.T_κ) * (target - w.κ_dyn)
    wm.κ_feedback && (w.κ = w.κ_dyn)
    return w
end
