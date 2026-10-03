"""
    PrognosticSGS

Era VI, Phase 56 (P56-b, #182): the Deardorff (1980) 1.5-order closure with a
prognostic subgrid-scale turbulent kinetic energy ``e``, with the
coefficients of Moeng (1984),

```math
\\frac{∂e}{∂t} + \\mathbf u·∇e = P_s + P_b − ε + ∇·(2ν_t ∇e),
```

```math
ν_t = c_k\\, l \\sqrt e, \\qquad κ_t = \\Big(1 + \\frac{2l}{Δ}\\Big) ν_t, \\qquad
ε = c_ε \\frac{e^{3/2}}{l}, \\qquad c_ε = 0.19 + 0.51\\,\\frac{l}{Δ},
```

```math
l = \\min\\!\\Big(Δ,\\; 0.76\\frac{\\sqrt e}{N},\\; κ\\, d_{wall}\\Big)
\\quad (N^2 > 0; \\text{ otherwise } \\min(Δ, κ d_{wall})),
```

with ``c_k = 0.1``, ``Δ = (Δx Δy Δz)^{1/3}`` and ``c_ε`` multiplied by 3.9
at the first level above a surface (Moeng 1984).

* [`TKEClosure`](@ref) is an `Operators.AbstractSGSClosure` (the P54-a
  contract, #167): its memory is ``e``; `sgs_coefficients!` reads it,
  `advance_sgs!` integrates its equation over each accepted step, and
  `sgs_state` / `restore_sgs_state!` make a rejected adaptive step exact.
* The shear production is the **discrete dual** of the SGS momentum
  operator ([`dual_strain!`](@ref)): ``Σ P_s V = −Σ \\mathbf u·\\mathbf T_{sgs} V``
  exactly, so the energy the resolved flow loses to the closure is the
  energy ``e`` gains.
* The buoyancy production ``P_b = −κ_t N^2`` uses the ``N^2`` of
  `Operators.buoyancy_frequency_squared!` (P58-a, #170), which on the
  uniform grid is the face average the κ_t scalar operator differences:
  ``Σ P_b V`` is exactly the potential-energy change of the SGS heat flux.
* The TKE closure has its own stability length scale (the ``0.76\\sqrt e/N``
  limit), so no `TurbulenceStability` stability function (nor the P55-e
  `StabilityCorrected` wrapper, #187) is applied on top of it.

References:

  * Deardorff, J. W. (1980), *Stratocumulus-capped mixed layers derived
    from a three-dimensional model*, Boundary-Layer Meteorol. 18, 495–527.
  * Moeng, C.-H. (1984), *A large-eddy-simulation model for the study of
    planetary boundary-layer turbulence*, J. Atmos. Sci. 41, 2052–2062.
  * Moeng, C.-H. & Wyngaard, J. C. (1988), *Spectral analysis of
    large-eddy simulations of the convective boundary layer*, J. Atmos.
    Sci. 45, 3573–3587 (the c_k = 0.1 calibration).
  * Patankar, S. V. (1980), *Numerical Heat Transfer and Fluid Flow*, §7.2
    (the linearised implicit sink that keeps ``e ≥ 0``).
"""
module PrognosticSGS

using ..Grids: RegularGrid, CellCenter, interior
using ..Fields: Field, interior_view, mesh
using ..BoundaryConditions: FieldBoundaryConditions, Periodic, fill_halos!
using ..Operators: AbstractSGSClosure, SGSFields, sgs_fields, add_eddy_diffusion_scalar!,
                   buoyancy_frequency_squared!
import ..Operators: sgs_coefficients!, advance_sgs!, sgs_state, restore_sgs_state!
using ..AdvectionSchemes: advect_positive!, FluxLimiterWorkspace

export TKEClosure, tke_length_scale, tke_dissipation_coefficient, tke_surface_value,
       tke_eddy_coefficients,
       tke_critical_richardson, tke_equilibrium, dual_strain!, shear_production!,
       buoyancy_production!, tke_budget, initialize_tke!, tke_statistics

# =========================================================================
# Pointwise physics (the one definition; the 3-D kernels call these)
# =========================================================================

"""
    tke_length_scale(e, N2, Δ; l_wall = Inf, c_l = 0.76) -> (l, s)

Deardorff's (1980) mixing length ``l = \\min(Δ, l_{wall}, c_l\\sqrt e/N)``
(the stability limit only for ``N^2 > 0``), and ``s = \\sqrt e / l`` (s⁻¹),
the rate every sink is written with. `s` stays finite where `l → 0`
(stable air, ``e → 0``): there ``s = N/c_l``. With ``N^2 ≤ 0`` and
``e = 0``, ``l = \\min(Δ, l_{wall})`` and ``s = 0``.
"""
@inline function tke_length_scale(e, N2, Δ; l_wall = oftype(float(Δ), Inf), c_l = 0.76)
    sq = sqrt(max(e, zero(e)))
    lmax = min(Δ, l_wall)
    if N2 > 0
        N = sqrt(N2)
        l = min(lmax, c_l * sq / N)
        s = max(sq / lmax, N / c_l)
    else
        l = lmax
        s = sq / lmax
    end
    return l, s
end

"""
    tke_dissipation_coefficient(l, Δ; c_ε0 = 0.19, c_ε1 = 0.51, factor = 1) -> c_ε

Moeng's (1984) ``c_ε = (c_{ε0} + c_{ε1} l/Δ) × factor`` (0.70 at ``l = Δ``);
`factor` is 3.9 at the first level above a surface.
"""
@inline tke_dissipation_coefficient(l, Δ; c_ε0 = 0.19, c_ε1 = 0.51, factor = 1) =
    (c_ε0 + c_ε1 * l / Δ) * factor

"""
    tke_eddy_coefficients(e, l, Δ; c_k = 0.1) -> (ν_t, κ_t)

Deardorff's (1980) eddy viscosity ``ν_t = c_k l \\sqrt e`` and diffusivity
``κ_t = (1 + 2l/Δ) ν_t`` (Moeng 1984) for the mixing length `l` of
[`tke_length_scale`](@ref). The one definition: [`TKEClosure`](@ref) and
[`ReynoldsStressClosure`](@ref) (its ``ν_t`` for the ``2ν_t`` transport and
the calibration of its Daly–Harlow coefficient) both call it.
"""
@inline function tke_eddy_coefficients(e, l, Δ; c_k = 0.1)
    ν = c_k * l * sqrt(max(e, zero(e)))
    return ν, (1 + 2l / Δ) * ν
end

"""
    tke_surface_value(u_star, c_ε; c_k = 0.1) -> e_s

The SGS TKE in local equilibrium with a constant-stress surface layer,
``e_s = u_*^2/\\sqrt{c_k c_ε}``: from ``ν_t S = u_*^2`` and ``ν_t S^2 = ε``
with ``ν_t = c_k l\\sqrt e``, ``ε = c_ε e^{3/2}/l``.
"""
tke_surface_value(u_star, c_ε; c_k = 0.1) = u_star^2 / sqrt(c_k * c_ε)

"""
    tke_critical_richardson(; c_k = 0.1, c_ε0 = 0.19, c_l = 0.76) -> Ri_c

The gradient Richardson number above which the Deardorff closure has no
equilibrium ``e > 0`` (the ``l → 0`` limit of ``P_s + P_b = ε``):
``Ri_c = 1/(1 + c_{ε0}/(c_k c_l^2))`` = 0.233 with the Moeng coefficients.
"""
tke_critical_richardson(; c_k = 0.1, c_ε0 = 0.19, c_l = 0.76) = 1 / (1 + c_ε0 / (c_k * c_l^2))

"""
    tke_equilibrium(S2, N2, Δ; c_k = 0.1, c_ε0 = 0.19, c_ε1 = 0.51, c_l = 0.76,
                    l_wall = Inf, factor = 1) -> e

The equilibrium SGS TKE of a homogeneous shear ``S^2`` and stratification
``N^2``: the root of ``ν_t S^2 − κ_t N^2 − ε = 0`` by bisection on
``\\sqrt e`` (0 if there is none, e.g. ``N^2/S^2 ≥ `` [`tke_critical_richardson`](@ref);
`Inf` without dissipation, ``c_{ε0} = c_{ε1} = 0``).
Built from [`tke_length_scale`](@ref) and [`tke_dissipation_coefficient`](@ref),
the same pointwise functions as the 3-D closure. Neutral, ``l = Δ``:
``e = (c_k/c_ε) Δ^2 S^2``, i.e. ``ν_t = (C_s Δ)^2 S`` with
``C_s = c_k^{3/4} c_ε^{-1/4}`` = 0.1944.
"""
function tke_equilibrium(S2, N2, Δ; c_k = 0.1, c_ε0 = 0.19, c_ε1 = 0.51, c_l = 0.76,
                         l_wall = Inf, factor = 1)
    # residual divided by √e > 0: c_k l (S² − (1+2l/Δ) N²) − c_ε e/l·... written per √e
    function g(q)
        e = q^2
        l, s = tke_length_scale(e, N2, Δ; l_wall = l_wall, c_l = c_l)
        cε = tke_dissipation_coefficient(l, Δ; c_ε0 = c_ε0, c_ε1 = c_ε1, factor = factor)
        # (ν S² − κ N² − ε)/e = (c_k/s)(S² − (1+2l/Δ)N²) − c_ε s
        return (c_k / s) * (S2 - (1 + 2l / Δ) * N2) - cε * s
    end
    # bracket: dissipation wins at large e (no root without dissipation: Inf)
    hi = sqrt(Δ^2 * S2 + Δ^2 * abs(N2)) + one(float(Δ))
    for _ in 1:2000
        g(hi) <= 0 && break
        hi *= 2
    end
    g(hi) <= 0 || return oftype(float(S2), Inf)
    lo = hi * 1e-12
    g(lo) <= 0 && return zero(float(S2))
    for _ in 1:200
        mid = (lo + hi) / 2
        g(mid) > 0 ? (lo = mid) : (hi = mid)
        hi - lo <= 4eps(hi) * hi && break
    end
    return ((lo + hi) / 2)^2
end

# =========================================================================
# The closure
# =========================================================================

"""
    TKEClosure(mesh::RegularGrid; e = nothing, e0 = 0, c_k = 0.1, c_ε0 = 0.19,
               c_ε1 = 0.51, c_l = 0.76, κ = 0.4, surface_factor = 3.9,
               wall_distance = nothing, first_level = nothing, mask = nothing,
               surface = nothing, θ₀ = nothing, g_accel = nothing, qv = :auto,
               transport = true, dissipation = :semi_implicit, advection = :auto)

The Deardorff (1980) / Moeng (1984) prognostic-TKE closure, an
`AbstractSGSClosure` for every core's `sgs` keyword (see the module
docstring for the equations).

Memory: the cell-centred SGS TKE `c.e` (m² s⁻²); `e` passes in a field to
share (the Berlin case passes its registered tracer), else it is allocated
with the value `e0`. Seed it with [`initialize_tke!`](@ref): ``e = 0`` is a
fixed point of the equation (``ν_t = 0`` produces nothing).

Keywords:

* `c_k`, `c_ε0`, `c_ε1`, `c_l`, `κ` — the coefficients (Moeng 1984; von
  Kármán κ for the wall cap); `surface_factor` multiplies ``c_ε`` in the
  `first_level` cells (Moeng's 3.9).
* `wall_distance` — interior-sized distances (m) from
  `IBM.wall_distance`; ``l`` is capped at ``κ d``. `nothing`: no cap.
* `first_level` — interior-sized `Bool`s (`IBM.first_level_cells`): the
  cells where ``c_ε`` gets `surface_factor`. `nothing`: none.
* `mask` — storage-sized `Bool` solid mask: ``e = ν_t = κ_t = 0`` in solids
  and no TKE flux through a solid face.
* `surface` — `nothing`, or `(u_star = M, k = K)`, per-column friction
  velocity (m s⁻¹) and the interior level of the first air cell; after each
  step that cell is set to [`tke_surface_value`](@ref)`(u*, c_ε)` (the
  value is read at every step, so `M` may be updated in place).
* `θ₀`, `g_accel` — buoyancy reference; `nothing` takes `sim.θ₀`, `sim.g_accel`
  (no buoyancy terms if the simulation has none). `qv` — `:auto` (the
  simulation's `qv` when it is moist), `nothing` (dry), or a field: ``N^2``
  then uses ``θ_v``.
* `transport` — advect (flux-limited, `AdvectionSchemes.advect_positive!`
  with `advection`: `:auto` = WENO-Z5 if the halo is ≥ 3, else centred) and
  diffuse (``∇·(2ν_t∇e)``, the three-argument
  `Operators.add_eddy_diffusion_scalar!`) ``e`` inside `advance_sgs!`.
  `false` when a driver transports ``e`` itself (the Berlin `:transport`
  stage) or for a sources-only box.
* `dissipation` — `:semi_implicit` (default): the linearised implicit sink
  ``e^{n+1} = e^*/(1 + Δt\\,(c_ε s^n + r_b^n))`` (Patankar), ``e^*`` the
  value after transport and production, ``s^n = \\sqrt{e^n}/l^n``, ``r_b``
  the buoyant destruction rate ``κ_t N^2/e``; positive with no clipping,
  first order, and its fixed point is the exact equilibrium. `:exact`: the
  exact solution of ``de/dt = −c_ε e^{3/2}/l`` over the step at frozen
  ``l``, ``e^{n+1} = (e^{*−1/2} + c_εΔt/(2l))^{−2}`` (then the buoyant sink).
"""
struct TKEClosure{T, F<:Field, LW, FL, MK, SF, Q, WS} <: AbstractSGSClosure
    e::F
    c_k::T; c_ε0::T; c_ε1::T; c_l::T; κ::T; surface_factor::T
    Δ::T
    θ₀::T                 # NaN: from the simulation
    g::T                  # NaN: from the simulation
    qv::Q                 # :auto, nothing or a Field
    l_wall::LW            # interior-sized κ·d_wall, or nothing
    first_level::FL       # interior-sized Bool, or nothing
    mask::MK              # storage-sized Bool, or nothing
    surface::SF           # nothing or (u_star, k)
    transport::Bool
    dissipation::Symbol
    advection::Symbol
    # scratch (not memory)
    ν::F; κt::F; N2::F; D::F; K2::F; tend::F
    ws::WS
    snap::Array{T,3}
    stats::Vector{T}      # see tke_statistics
    snapstats::Vector{T}
end

const _NSTATS = 9

function TKEClosure(g::RegularGrid{T}; e = nothing, e0::Real = 0, c_k::Real = 0.1,
                    c_ε0::Real = 0.19, c_ε1::Real = 0.51, c_l::Real = 0.76, κ::Real = 0.4,
                    surface_factor::Real = 3.9, wall_distance = nothing, first_level = nothing,
                    mask = nothing, surface = nothing, θ₀ = nothing, g_accel = nothing,
                    qv = :auto, transport::Bool = true, dissipation::Symbol = :semi_implicit,
                    advection::Symbol = :auto) where {T}
    dissipation in (:semi_implicit, :exact) ||
        throw(ArgumentError("dissipation must be :semi_implicit or :exact, got $(repr(dissipation))"))
    ef = e === nothing ? Field(CellCenter(), g) : e
    e === nothing && (ef.data .= T(e0))
    N = (g.Nx, g.Ny, g.Nz)
    lw = wall_distance === nothing ? nothing : (size(wall_distance) == N ? T(κ) .* T.(wall_distance) :
         throw(DimensionMismatch("wall_distance must be interior-sized $N, got $(size(wall_distance))")))
    fl = first_level === nothing ? nothing : (size(first_level) == N ? Array{Bool}(first_level) :
         throw(DimensionMismatch("first_level must be interior-sized $N")))
    mk = mask === nothing ? nothing : (size(mask) == size(ef.data) ? Array{Bool}(mask) :
         throw(DimensionMismatch("mask must be storage-sized $(size(ef.data))")))
    adv = advection === :auto ? (min(g.Hx, g.Hy, g.Hz) >= 3 ? :weno_z5 : :centered) : advection
    ws = transport ? FluxLimiterWorkspace(g) : nothing
    Δ = T((g.dx * g.dy * g.dz)^(1 / 3))
    mkf() = Field(CellCenter(), g)
    return TKEClosure(ef, T(c_k), T(c_ε0), T(c_ε1), T(c_l), T(κ), T(surface_factor), Δ,
                      θ₀ === nothing ? T(NaN) : T(θ₀), g_accel === nothing ? T(NaN) : T(g_accel),
                      qv, lw, fl, mk, surface, transport, dissipation, adv,
                      mkf(), mkf(), mkf(), mkf(), mkf(), mkf(), ws,
                      similar(ef.data), zeros(T, _NSTATS), zeros(T, _NSTATS))
end

Base.show(io::IO, c::TKEClosure) =
    print(io, "TKEClosure(c_k = $(c_k(c)), Δ = $(round(c.Δ; sigdigits = 4)) m, ",
          "dissipation = :$(c.dissipation), transport = $(c.transport))")
c_k(c::TKEClosure) = c.c_k

# --- buoyancy reference (shared with ReynoldsStressClosure: c.θ₀, c.g, c.qv, c.N2)
_θref(c, sim) = isnan(c.θ₀) ? (hasproperty(sim, :θ₀) ? float(sim.θ₀) : zero(c.θ₀)) : c.θ₀
_gref(c, sim) = isnan(c.g) ? (hasproperty(sim, :g_accel) ? float(sim.g_accel) : zero(c.g)) : c.g
_qv(c, sim) = c.qv === :auto ?
    (hasproperty(sim, :has_moisture) && sim.has_moisture ? sim.qv : nothing) : c.qv

# N² at cell centres (zero without buoyancy). Fills the θ (and qv) halos.
function _stratification!(c, sim)
    θ₀ = _θref(c, sim); g = _gref(c, sim)
    if θ₀ > 0 && g > 0
        fill_halos!(sim.θ, sim.θ_bcs)
        q = _qv(c, sim)
        if q === nothing
            buoyancy_frequency_squared!(c.N2, sim.θ; θ₀ = θ₀, g = g)
        else
            fill_halos!(q, sim.θ_bcs)
            buoyancy_frequency_squared!(c.N2, sim.θ, q; θ₀ = θ₀, g = g)
        end
    else
        fill!(c.N2.data, 0)
    end
    return c.N2
end

# The buoyancy production of a prescribed surface heat flux (the
# Boussinesq `θ_flux_bottom` / `θ_flux_top`, added by `add_surface_flux!`
# into the first / last level): the flux sits on the boundary face, so the
# face's half goes to the adjacent cell, exactly as the κ_t face fluxes are
# apportioned (P_b = (g/θ₀)·F/2 there). Zero without buoyancy or flux.
function _surface_buoyancy(c::TKEClosure{T}, sim) where {T}
    θ₀ = _θref(c, sim); g = _gref(c, sim)
    (θ₀ > 0 && g > 0 && hasproperty(sim, :has_surface_flux) && sim.has_surface_flux) ||
        return zero(T), zero(T)
    return T(g / θ₀ * sim.θ_flux_bottom / 2), T(g / θ₀ * sim.θ_flux_top / 2)
end

@inline _solid(::Nothing, i, j, k) = false
@inline _solid(m, i, j, k) = @inbounds m[i, j, k]
@inline _lwall(::Nothing, i, j, k, T) = T(Inf)
@inline _lwall(a, i, j, k, T) = @inbounds a[i, j, k]
@inline _first(::Nothing, i, j, k) = false
@inline _first(a, i, j, k) = @inbounds a[i, j, k]

# ν_t, κ_t of the closure's current e into ν, κ (interior); N² must be current.
function _coefficients!(ν::Field, κt::Field, c::TKEClosure{T}, N2::Field) where {T}
    g = mesh(c.e)
    Hx, Hy, Hz = g.Hx, g.Hy, g.Hz
    ed, nd, kd, n2 = c.e.data, ν.data, κt.data, N2.data
    Δ, ck, cl = c.Δ, c.c_k, c.c_l
    @inbounds for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        I, J, K = Hx + i, Hy + j, Hz + k
        if _solid(c.mask, I, J, K)
            nd[I, J, K] = zero(T); kd[I, J, K] = zero(T)
            continue
        end
        e = ed[I, J, K]
        l, _ = tke_length_scale(e, n2[I, J, K], Δ; l_wall = _lwall(c.l_wall, i, j, k, T), c_l = cl)
        νv, κv = tke_eddy_coefficients(e, l, Δ; c_k = ck)
        nd[I, J, K] = νv
        kd[I, J, K] = κv
    end
    return ν, κt
end

"""
    sgs_coefficients!(f::SGSFields, c::TKEClosure, sim)

``ν_t = c_k l\\sqrt e`` and ``κ_t = (1 + 2l/Δ)ν_t`` from the committed ``e``
and the current ``N^2`` (`buoyancy_frequency_squared!`, θ_v when moist).
Pure in the closure's memory (it writes only its ``N^2`` scratch and the
halos of θ / qv).
"""
function sgs_coefficients!(f::SGSFields, c::TKEClosure, sim)
    _stratification!(c, sim)
    _coefficients!(f.νh, f.κh, c, c.N2)
    return f
end

# -------------------------------------------------------------------------
# The discrete dual of the SGS momentum operator
# -------------------------------------------------------------------------

"""
    dual_strain!(D, u, v, w, bcs) -> D

The velocity factor of the shear production that is the exact discrete
dual (summation by parts) of the Laplacian SGS momentum operator
`Operators.add_eddy_viscosity_u!` / `_v!` / `_w!`:

```math
\\sum_c ν_c D_c = −\\sum_{\\text{faces}} \\mathbf u·\\mathbf T_{sgs}(\\mathbf u, ν)
\\quad\\text{for every cell-centred } ν ≥ 0,
```

so ``P_s = ν_t D`` (see [`shear_production!`](@ref)). Every flux of the
momentum operator, ``f_L = ν_L δu_L/Δ`` with ``ν_L`` the cell value (normal
fluxes) or the 0.25-weighted average of the four cells around an edge
(cross fluxes), contributes ``−u_p\\,(±f_L)/Δ`` to ``\\mathbf u·\\mathbf T``
at each velocity point ``p`` it enters; that is apportioned back to the
cells of ``ν_L`` with the same weights. Halo shares are folded onto the
interior cells they are copies of (periodic image or mirror, from `bcs`,
the boundary conditions ``ν_t`` was filled with), which is exact because a
halo ``ν`` equals its image. In the interior each flux contributes
``(δu/Δ)^2 ≥ 0``, so ``D ≥ 0``; for ``u = Sz``, ``D = S^2 = |S|^2``.

`D` lives at `CellCenter`; the halos of `u, v, w` must be filled. Built
from the same flux locations as the momentum kernels (the face-to-face and
edge differences), not from `Operators._velocity_gradient_at`, whose
off-diagonal two-cell differences are not the dual of that operator.
"""
dual_strain!(D::Field, u::Field, v::Field, w::Field, bcs::FieldBoundaryConditions) =
    _dual_strain!(D, u, v, w, bcs, mesh(u))

function _dual_strain!(D::Field, u::Field, v::Field, w::Field, bcs, g::RegularGrid{T}) where {T}
    idx2 = 1 / g.dx^2; idy2 = 1 / g.dy^2; idz2 = 1 / g.dz^2
    d = D.data
    fill!(d, zero(T))
    ud, vd, wd = u.data, v.data, w.data
    ii, jj, kk = interior(g)
    q4(x) = T(0.25) * x
    @inbounds for k in kk, j in jj, i in ii
        # ---- u at FaceX (i,j,k) ----
        up = ud[i, j, k]
        a = -up * (ud[i+1, j, k] - ud[i, j, k]) * idx2         # fxE, +; cell (i,j,k)
        d[i, j, k] += a
        a = up * (ud[i, j, k] - ud[i-1, j, k]) * idx2          # fxW, −; cell (i-1,j,k)
        d[i-1, j, k] += a
        a = q4(-up * (ud[i, j+1, k] - ud[i, j, k]) * idy2)     # fyN, +
        d[i-1, j, k] += a; d[i, j, k] += a; d[i-1, j+1, k] += a; d[i, j+1, k] += a
        a = q4(up * (ud[i, j, k] - ud[i, j-1, k]) * idy2)      # fyS, −
        d[i-1, j-1, k] += a; d[i, j-1, k] += a; d[i-1, j, k] += a; d[i, j, k] += a
        a = q4(-up * (ud[i, j, k+1] - ud[i, j, k]) * idz2)     # fzT, +
        d[i-1, j, k] += a; d[i, j, k] += a; d[i-1, j, k+1] += a; d[i, j, k+1] += a
        a = q4(up * (ud[i, j, k] - ud[i, j, k-1]) * idz2)      # fzB, −
        d[i-1, j, k-1] += a; d[i, j, k-1] += a; d[i-1, j, k] += a; d[i, j, k] += a

        # ---- v at FaceY (i,j,k) ----
        vp = vd[i, j, k]
        a = -vp * (vd[i, j+1, k] - vd[i, j, k]) * idy2         # fyN, +; cell (i,j,k)
        d[i, j, k] += a
        a = vp * (vd[i, j, k] - vd[i, j-1, k]) * idy2          # fyS, −; cell (i,j-1,k)
        d[i, j-1, k] += a
        a = q4(-vp * (vd[i+1, j, k] - vd[i, j, k]) * idx2)     # fxE, +
        d[i, j-1, k] += a; d[i+1, j-1, k] += a; d[i, j, k] += a; d[i+1, j, k] += a
        a = q4(vp * (vd[i, j, k] - vd[i-1, j, k]) * idx2)      # fxW, −
        d[i-1, j-1, k] += a; d[i, j-1, k] += a; d[i-1, j, k] += a; d[i, j, k] += a
        a = q4(-vp * (vd[i, j, k+1] - vd[i, j, k]) * idz2)     # fzT, +
        d[i, j-1, k] += a; d[i, j, k] += a; d[i, j-1, k+1] += a; d[i, j, k+1] += a
        a = q4(vp * (vd[i, j, k] - vd[i, j, k-1]) * idz2)      # fzB, −
        d[i, j-1, k-1] += a; d[i, j, k-1] += a; d[i, j-1, k] += a; d[i, j, k] += a

        # ---- w at FaceZ (i,j,k) ----
        wp = wd[i, j, k]
        a = -wp * (wd[i, j, k+1] - wd[i, j, k]) * idz2         # fzT, +; cell (i,j,k)
        d[i, j, k] += a
        a = wp * (wd[i, j, k] - wd[i, j, k-1]) * idz2          # fzB, −; cell (i,j,k-1)
        d[i, j, k-1] += a
        a = q4(-wp * (wd[i+1, j, k] - wd[i, j, k]) * idx2)     # fxE, +
        d[i, j, k-1] += a; d[i+1, j, k-1] += a; d[i, j, k] += a; d[i+1, j, k] += a
        a = q4(wp * (wd[i, j, k] - wd[i-1, j, k]) * idx2)      # fxW, −
        d[i-1, j, k-1] += a; d[i, j, k-1] += a; d[i-1, j, k] += a; d[i, j, k] += a
        a = q4(-wp * (wd[i, j+1, k] - wd[i, j, k]) * idy2)     # fyN, +
        d[i, j, k-1] += a; d[i, j+1, k-1] += a; d[i, j, k] += a; d[i, j+1, k] += a
        a = q4(wp * (wd[i, j, k] - wd[i, j-1, k]) * idy2)      # fyS, −
        d[i, j-1, k-1] += a; d[i, j, k-1] += a; d[i, j-1, k] += a; d[i, j, k] += a
    end
    _fold_halos!(d, bcs, g)
    return D
end

# Fold the first halo layer onto the interior cells it is a copy of (the
# periodic image, or the mirror for no-flux / wall boundaries), x then y
# then z, so that edge and corner halo shares are carried twice correctly.
function _fold_halos!(d, bcs, g)
    Hx, Hy, Hz, Nx, Ny, Nz = g.Hx, g.Hy, g.Hz, g.Nx, g.Ny, g.Nz
    sx, sy, sz = size(d)
    # x: halo index Hx (low) and Hx+Nx+1 (high)
    lo, hi = _images(bcs.west, Hx, Nx)
    @inbounds for k in 1:sz, j in 1:sy
        d[lo, j, k] += d[Hx, j, k];        d[Hx, j, k] = 0
        d[hi, j, k] += d[Hx+Nx+1, j, k];   d[Hx+Nx+1, j, k] = 0
    end
    lo, hi = _images(bcs.south, Hy, Ny)
    @inbounds for k in 1:sz, i in 1:sx
        d[i, lo, k] += d[i, Hy, k];        d[i, Hy, k] = 0
        d[i, hi, k] += d[i, Hy+Ny+1, k];   d[i, Hy+Ny+1, k] = 0
    end
    lo, hi = _images(bcs.bottom, Hz, Nz)
    @inbounds for j in 1:sy, i in 1:sx
        d[i, j, lo] += d[i, j, Hz];        d[i, j, Hz] = 0
        d[i, j, hi] += d[i, j, Hz+Nz+1];   d[i, j, Hz+Nz+1] = 0
    end
    return d
end
# (image of the low halo, image of the high halo)
_images(::Periodic, H, N) = (H + N, H + 1)
_images(_, H, N) = (H + 1, H + N)          # NoFlux, ImpermeableWall: mirror

"""
    shear_production!(P, u, v, w, νt, bcs) -> P

``P_s = ν_t D`` with ``D`` from [`dual_strain!`](@ref) (`P` holds ``D`` on
the way): the shear production whose volume sum is exactly the resolved
kinetic energy the SGS momentum operator removes,
``Σ P_s V = −Σ \\mathbf u·\\mathbf T_{sgs}(ν_t) V``. `νt` must have filled
halos (with `bcs`), as the momentum operator needs.
"""
function shear_production!(P::Field, u::Field, v::Field, w::Field, νt::Field,
                           bcs::FieldBoundaryConditions)
    dual_strain!(P, u, v, w, bcs)
    g = mesh(u)
    ii, jj, kk = interior(g)
    @inbounds @views P.data[ii, jj, kk] .*= νt.data[ii, jj, kk]
    return P
end

"""
    buoyancy_production!(P, κt, N2) -> P

``P_b = −κ_t N^2`` at cell centres. With ``N^2`` from
`Operators.buoyancy_frequency_squared!` (on a uniform grid the average of
the two face gradients) this is the discrete dual of the κ_t scalar
operator: ``Σ P_b V = (g/θ_0) Σ z\\, T_θ V``, the potential-energy change
of the SGS heat flux. (A prescribed surface heat flux adds ``(g/θ_0)F/2``
to the boundary cells inside `advance_sgs!`; not here.)
"""
function buoyancy_production!(P::Field, κt::Field, N2::Field)
    g = mesh(P)
    ii, jj, kk = interior(g)
    @inbounds @views P.data[ii, jj, kk] .= .-κt.data[ii, jj, kk] .* N2.data[ii, jj, kk]
    return P
end

# -------------------------------------------------------------------------
# The time integration of e (the one commit per accepted step)
# -------------------------------------------------------------------------

"""
    advance_sgs!(c::TKEClosure, sim, dt)

Integrate the TKE equation over the accepted step `dt` (called once per
step, or once per accepted piece of an adaptive step, after the projection;
the sum over the pieces is the step):

1. the coefficients ``ν_t, κ_t, l, s`` from ``e^n`` and the new state's
   ``N^2``; ``P_s = ν_t D`` ([`dual_strain!`](@ref)), ``P_b = −κ_t N^2``;
2. with `transport`: ``T = −∇·(\\mathbf u e)`` (flux-limited) ``+ ∇·(2ν_t∇e)``;
3. ``e^* = e^n + Δt (T + P_s + \\max(P_b, 0))``, clipped at 0 if negative
   (booked: [`tke_statistics`](@ref));
4. the sinks (dissipation, buoyant destruction), linearised implicit or
   exact (`dissipation`), so ``e^{n+1} ≥ 0``;
5. ``e = 0`` in solids; the first air cell of each column set to
   [`tke_surface_value`](@ref) when `surface` is given (booked).
"""
function advance_sgs!(c::TKEClosure{T}, sim, dt) where {T}
    g = mesh(c.e)
    Δt = T(dt)
    bcs = sim.bcs
    fill_halos!(sim.u, bcs); fill_halos!(sim.v, bcs); fill_halos!(sim.w, bcs)
    _stratification!(c, sim)
    _coefficients!(c.ν, c.κt, c, c.N2)
    fill_halos!(c.ν, bcs)
    dual_strain!(c.D, sim.u, sim.v, sim.w, bcs)
    fill!(c.tend.data, 0)
    if c.transport
        ebcs = sim.θ_bcs
        fill_halos!(c.e, ebcs)
        advect_positive!(c.tend, sim.u, sim.v, sim.w, c.e, Δt;
                         scheme = c.advection, workspace = c.ws, bcs = ebcs)
        c.K2.data .= 2 .* c.ν.data
        add_eddy_diffusion_scalar!(c.tend, c.e, c.K2; mask = c.mask)
    end
    _update!(c, Δt, _surface_buoyancy(c, sim))
    return c
end

function _update!(c::TKEClosure{T}, Δt::T, pbs) where {T}
    g = mesh(c.e)
    Hx, Hy, Hz = g.Hx, g.Hy, g.Hz
    V = g.dx * g.dy * g.dz
    ed, nd, n2, dd, td = c.e.data, c.ν.data, c.N2.data, c.D.data, c.tend.data
    Δ, ck, cl = c.Δ, c.c_k, c.c_l
    sPs = zero(T); sPb = zero(T); sε = zero(T); sB = zero(T); sT = zero(T)
    nclip = zero(T); eclip = zero(T)
    exact = c.dissipation === :exact
    @inbounds for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        I, J, K = Hx + i, Hy + j, Hz + k
        if _solid(c.mask, I, J, K)
            ed[I, J, K] = zero(T)
            continue
        end
        e = ed[I, J, K]
        N2 = n2[I, J, K]
        l, s = tke_length_scale(e, N2, Δ; l_wall = _lwall(c.l_wall, i, j, k, T), c_l = cl)
        fac = _first(c.first_level, i, j, k) ? c.surface_factor : one(T)
        cε = tke_dissipation_coefficient(l, Δ; c_ε0 = c.c_ε0, c_ε1 = c.c_ε1, factor = fac)
        Ps = nd[I, J, K] * dd[I, J, K]
        κv = (1 + 2l / Δ) * nd[I, J, K]
        Psf = (k == 1 ? pbs[1] : zero(T)) + (k == g.Nz ? pbs[2] : zero(T))
        Pb = -κv * N2 + Psf
        # sources: P_s and the positive buoyancy production; sinks, linear in
        # e: dissipation c_ε s e, the buoyant destruction κ_t N² = r_N e
        # (r_N = (1+2l/Δ) c_k N²/s, finite as e → 0) and a stable surface flux
        src = Ps + (N2 < 0 ? -κv * N2 : zero(T)) + max(Psf, zero(T))
        rb = (N2 > 0 && s > 0 ? (1 + 2l / Δ) * ck * N2 / s : zero(T)) +
             (Psf < 0 ? -Psf / max(e, floatmin(T)) : zero(T))
        tr = td[I, J, K]
        estar = e + Δt * (tr + src)
        if estar < 0
            nclip += 1; eclip -= estar * V
            estar = zero(T)
        end
        rε = cε * s
        if exact && l > 0 && estar > 0
            e1 = inv(inv(sqrt(estar)) + cε * Δt / (2l))^2
            enew = e1 / (1 + Δt * rb)
            dε = (estar - e1) / Δt
            dB = (e1 - enew) / Δt
        else
            R = rε + rb
            enew = estar / (1 + Δt * R)
            rem = (estar - enew) / Δt
            dε = R > 0 ? rem * (rε / R) : zero(T)
            dB = R > 0 ? rem * (rb / R) : zero(T)
        end
        ed[I, J, K] = enew
        sPs += Ps; sPb += Pb; sT += tr
        sε += dε; sB += dB
    end
    st = c.stats
    st[1] += nclip; st[2] += eclip
    st[4] = sPs * V; st[5] = sPb * V; st[6] = sε * V; st[7] = sB * V; st[8] = sT * V
    st[9] += 1
    c.surface === nothing || _surface!(c, Δt)
    return c
end

function _surface!(c::TKEClosure{T}, Δt) where {T}
    g = mesh(c.e)
    us, kk = c.surface.u_star, c.surface.k
    Hx, Hy, Hz = g.Hx, g.Hy, g.Hz
    V = g.dx * g.dy * g.dz
    ed = c.e.data
    de = zero(T)
    @inbounds for j in 1:g.Ny, i in 1:g.Nx
        k = kk[i, j]
        1 <= k <= g.Nz || continue
        I, J, K = Hx + i, Hy + j, Hz + k
        _solid(c.mask, I, J, K) && continue
        u = us[i, j]
        isfinite(u) || continue
        # c_ε at the surface value itself (l = min(Δ, κ d): neutral surface layer)
        l, _ = tke_length_scale(zero(T), zero(T), c.Δ; l_wall = _lwall(c.l_wall, i, j, k, T))
        fac = _first(c.first_level, i, j, k) ? c.surface_factor : one(T)
        cε = tke_dissipation_coefficient(l, c.Δ; c_ε0 = c.c_ε0, c_ε1 = c.c_ε1, factor = fac)
        es = tke_surface_value(T(u), cε; c_k = c.c_k)
        de += (es - ed[I, J, K]) * V
        ed[I, J, K] = es
    end
    c.stats[3] += de
    return c
end

"""
    sgs_state(c::TKEClosure) / restore_sgs_state!(c, s)

Snapshot of the memory (``e`` and the booked statistics) into buffers the
closure owns, and its restoration: a step `step_adaptive!` rejects leaves
``e`` bit for bit as it was.
"""
function sgs_state(c::TKEClosure)
    copyto!(c.snap, c.e.data)
    copyto!(c.snapstats, c.stats)
    return c.snap
end
function restore_sgs_state!(c::TKEClosure, s)
    copyto!(c.e.data, s)
    copyto!(c.stats, c.snapstats)
    return c
end

"""
    tke_statistics(c::TKEClosure) -> NamedTuple

What [`advance_sgs!`](@ref) booked: `clips` (number of cell updates where
``e^*`` was negative and set to 0, cumulative), `clipped_energy` (the TKE
those clips added, m⁵ s⁻², cumulative), `surface_energy` (TKE added by
setting the surface value, m⁵ s⁻², cumulative), and the volume integrals of
the last step (m⁵ s⁻³): `shear`, `buoyancy` (``Σ P_b V``, signed),
`dissipation` (``Σ ε V``), `buoyant_destruction` (the implicit sink of the
negative ``P_b``), `transport`; `steps` (commits).
"""
tke_statistics(c::TKEClosure) =
    (clips = Int(c.stats[1]), clipped_energy = c.stats[2], surface_energy = c.stats[3],
     shear = c.stats[4], buoyancy = c.stats[5], dissipation = c.stats[6],
     buoyant_destruction = c.stats[7], transport = c.stats[8], steps = Int(c.stats[9]))

"""
    tke_budget(c::TKEClosure, sim) -> NamedTuple

Instantaneous volume-integrated terms of the TKE equation at the current
state and ``e`` (m⁵ s⁻³; pure: no memory changes): `shear` ``Σ P_s V``,
`buoyancy` ``Σ P_b V`` (with the half-face share of a prescribed surface
heat flux, as in `advance_sgs!`), `dissipation` ``Σ ε V`` (``ε = c_ε e^{3/2}/l``),
`transport` (``Σ V`` of the flux-limited advection and the ``2ν_t``
diffusion: zero up to boundary fluxes), and `tke` ``Σ e V`` (m⁵ s⁻²).
"""
function tke_budget(c::TKEClosure{T}, sim) where {T}
    g = mesh(c.e)
    bcs = sim.bcs
    fill_halos!(sim.u, bcs); fill_halos!(sim.v, bcs); fill_halos!(sim.w, bcs)
    _stratification!(c, sim)
    _coefficients!(c.ν, c.κt, c, c.N2)
    fill_halos!(c.ν, bcs)
    dual_strain!(c.D, sim.u, sim.v, sim.w, bcs)
    V = g.dx * g.dy * g.dz
    tr = zero(T)
    if c.transport
        fill!(c.tend.data, 0)
        fill_halos!(c.e, sim.θ_bcs)
        advect_positive!(c.tend, sim.u, sim.v, sim.w, c.e, one(T);
                         scheme = c.advection, workspace = c.ws, bcs = sim.θ_bcs)
        c.K2.data .= 2 .* c.ν.data
        add_eddy_diffusion_scalar!(c.tend, c.e, c.K2; mask = c.mask)
        tr = sum(interior_view(c.tend)) * V
    end
    Hx, Hy, Hz = g.Hx, g.Hy, g.Hz
    pbs = _surface_buoyancy(c, sim)
    sP = zero(T); sB = zero(T); sε = zero(T); se = zero(T)
    @inbounds for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        I, J, K = Hx + i, Hy + j, Hz + k
        _solid(c.mask, I, J, K) && continue
        e = c.e.data[I, J, K]; N2 = c.N2.data[I, J, K]
        l, s = tke_length_scale(e, N2, c.Δ; l_wall = _lwall(c.l_wall, i, j, k, T), c_l = c.c_l)
        fac = _first(c.first_level, i, j, k) ? c.surface_factor : one(T)
        cε = tke_dissipation_coefficient(l, c.Δ; c_ε0 = c.c_ε0, c_ε1 = c.c_ε1, factor = fac)
        sP += c.ν.data[I, J, K] * c.D.data[I, J, K]
        sB += -c.κt.data[I, J, K] * N2 + (k == 1 ? pbs[1] : zero(T)) + (k == g.Nz ? pbs[2] : zero(T))
        sε += cε * s * e
        se += e
    end
    return (shear = sP * V, buoyancy = sB * V, dissipation = sε * V, transport = tr,
            tke = se * V)
end

"""
    initialize_tke!(c::TKEClosure, sim; e_min = 0) -> c

Seed ``e`` with the local equilibrium of the current resolved flow,
[`tke_equilibrium`](@ref)`(D, N², Δ)` per cell with the shear factor ``D``
of [`dual_strain!`](@ref) (``= |S|^2`` for a linear shear) and the cell's
``N^2``, wall cap and ``c_ε`` factor, and at least `e_min`. ``e = 0`` is a
fixed point of the TKE equation, so a closure started from rest needs
this (or a surface value, or a small `e_min`) to spin up.
"""
function initialize_tke!(c::TKEClosure{T}, sim; e_min::Real = 0) where {T}
    g = mesh(c.e)
    bcs = sim.bcs
    fill_halos!(sim.u, bcs); fill_halos!(sim.v, bcs); fill_halos!(sim.w, bcs)
    _stratification!(c, sim)
    dual_strain!(c.D, sim.u, sim.v, sim.w, bcs)
    Hx, Hy, Hz = g.Hx, g.Hy, g.Hz
    @inbounds for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        I, J, K = Hx + i, Hy + j, Hz + k
        if _solid(c.mask, I, J, K)
            c.e.data[I, J, K] = zero(T)
            continue
        end
        fac = _first(c.first_level, i, j, k) ? c.surface_factor : one(T)
        eq = tke_equilibrium(max(c.D.data[I, J, K], zero(T)), c.N2.data[I, J, K], c.Δ;
                             c_k = c.c_k, c_ε0 = c.c_ε0, c_ε1 = c.c_ε1, c_l = c.c_l,
                             l_wall = _lwall(c.l_wall, i, j, k, T), factor = fac)
        c.e.data[I, J, K] = max(T(eq), T(e_min))
    end
    return c
end

end # module
