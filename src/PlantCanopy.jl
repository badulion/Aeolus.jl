"""
    PlantCanopy

Resolved plant canopies for the LES (Era V, Phase 48).

This module owns the vertical structure of vegetation and the momentum
sink it exerts on the resolved flow:

  1. **Leaf-area density profiles** — [`AbstractLeafAreaDensity`](@ref)
     with a uniform crown ([`UniformLAD`](@ref)), the empirical forest
     profile of Lalic & Mihailovic (2004) ([`LalicMihailovicLAD`](@ref))
     and the beta-distribution profile of Markkanen et al. (2003)
     ([`BetaLAD`](@ref)). Every profile integrates to its leaf-area index
     by construction; [`layer_leaf_area_density`](@ref) maps a profile
     onto model layers so that the discrete column still sums to the LAI.
  2. **Implicit canopy drag** — the form drag
     ``F_i = -C_d\\,a(z)\\,|U|\\,U_i`` applied as the pointwise factor
     ``U^{n+1} = U^* / (1 + Δt\\,C_d\\,a\\,|U^*|)``
     ([`canopy_drag_factor`](@ref), [`apply_canopy_drag!`](@ref)). For the
     pure drag equation ``dU/dt = -C_d a |U| U`` this factor is the
     *exact* solution over the step, so it is unconditionally stable and
     never reverses the flow; explicit Euler of the same term is unstable
     once ``Δt\\,C_d\\,a\\,|U| > 2``.
  3. **In-canopy mean wind** — a 1-D column solver
     ([`solve_canopy_wind_column`](@ref)) with a mixing-length closure and
     the same implicit drag, and the analytic exponential profile of
     Inoue (1963) in the form of Harman & Finnigan (2007)
     ([`canopy_exponential_wind`](@ref)) that it reproduces.

Radiation attenuation (#23) and the leaf energy balance (#24) extend this
module; they use the cumulative leaf area [`cumulative_leaf_area`](@ref)
defined here.

# References

  * Lalic, B. & Mihailovic, D. T. (2004). An empirical relation describing
    leaf-area density inside the forest for environmental modeling.
    *J. Appl. Meteorol.*, 43, 641-645.
  * Markkanen, T., Rannik, Ü., Marcolla, B., Cescatti, A. & Vesala, T.
    (2003). Footprints and fetches for fluxes over forest canopies with
    varying structure and density. *Boundary-Layer Meteorol.*, 106, 437-459.
  * Shaw, R. H. & Schumann, U. (1992). Large-eddy simulation of turbulent
    flow above and within a forest. *Boundary-Layer Meteorol.*, 61, 47-64.
  * Inoue, E. (1963). On the turbulent structure of airflow within crop
    canopies. *J. Meteorol. Soc. Japan*, 41, 317-326.
  * Harman, I. N. & Finnigan, J. J. (2007). A simple unified theory for
    flow in the canopy and roughness sublayer. *Boundary-Layer Meteorol.*,
    123, 339-363.
"""
module PlantCanopy

using ..Grids: RegularGrid
using ..BoundaryConditions: fill_halos!
using ..RadiationCore: _gauss_legendre
using ..P3Microphysics: lgamma_lanczos

export AbstractLeafAreaDensity, UniformLAD, LalicMihailovicLAD, BetaLAD,
       leaf_area_density, leaf_area_index, canopy_height, cumulative_leaf_area,
       layer_leaf_area_density,
       canopy_drag_factor, canopy_drag_force, implicit_canopy_drag,
       ResolvedCanopyDrag, canopy_lad_field, apply_canopy_drag!,
       CanopyWindSolution, solve_canopy_wind_column,
       harman_finnigan_mixing_length, canopy_attenuation_coefficient,
       canopy_exponential_wind

# =========================================================================
# 1. Leaf-area density profiles
# =========================================================================

"""
    AbstractLeafAreaDensity

Supertype of vertical leaf-area-density (LAD) profiles ``a(z)`` [m² m⁻³],
one-sided leaf area per unit volume, with ``z`` the height above ground.
An implementer provides

  * `leaf_area_density(p, z)` — ``a(z)``, zero outside the canopy;
  * `canopy_height(p)`        — the canopy top ``h`` [m];
  * `leaf_area_index(p)`      — ``\\int_0^h a\\,dz`` [m² m⁻²];

and may add height breakpoints (kinks of ``a``) through
`_lad_breakpoints(p)` so that layer integrals stay at round-off accuracy.
"""
abstract type AbstractLeafAreaDensity end

"""
    UniformLAD(LAI, h; z_base = 0)

Leaf area spread uniformly over the crown ``z_{base} ≤ z ≤ h``:
``a = LAI / (h - z_{base})``. With `z_base = 0` this is the uniform
canopy of the Inoue / Harman–Finnigan theory.
"""
struct UniformLAD{T<:AbstractFloat} <: AbstractLeafAreaDensity
    LAI::T
    h::T
    z_base::T
    function UniformLAD{T}(LAI, h, z_base) where {T}
        LAI >= 0 || throw(ArgumentError("UniformLAD: LAI must be ≥ 0"))
        0 <= z_base < h || throw(ArgumentError("UniformLAD: need 0 ≤ z_base < h"))
        new{T}(LAI, h, z_base)
    end
end
UniformLAD(LAI::Real, h::Real; z_base::Real = 0) =
    UniformLAD{Float64}(Float64(LAI), Float64(h), Float64(z_base))

leaf_area_density(p::UniformLAD, z::Real) =
    (p.z_base <= z <= p.h) ? p.LAI / (p.h - p.z_base) : zero(p.LAI)
_lad_breakpoints(p::UniformLAD) = (p.z_base,)

# Lalic & Mihailovic (2004) shape, normalised to 1 at z = z_m.
function _lm_shape(z, h, z_m, n_below, n_above)
    (z < 0 || z >= h) && return zero(float(z))
    r = (h - z_m) / (h - z)
    n = z < z_m ? n_below : n_above
    return r^n * exp(n * (1 - r))
end

"""
    LalicMihailovicLAD(LAI, h; z_m, n_below = 6, n_above = 1/2)

Empirical forest LAD profile of Lalic & Mihailovic (2004, eq. 1):

```math
a(z) = L_m \\left(\\frac{h - z_m}{h - z}\\right)^n
       \\exp\\!\\left[n\\left(1 - \\frac{h - z_m}{h - z}\\right)\\right],
\\qquad n = \\begin{cases} 6 & 0 ≤ z < z_m \\\\ 1/2 & z_m ≤ z ≤ h \\end{cases}
```

``z_m`` is the height of maximum LAD and ``L_m = a(z_m)``. Here ``L_m`` is
not a free parameter: it is fixed at construction so that the profile
integrates to `LAI` (the paper's own empirical fit for ``L_m`` is not
used). The profile vanishes smoothly at the canopy top.
"""
struct LalicMihailovicLAD{T<:AbstractFloat} <: AbstractLeafAreaDensity
    LAI::T
    h::T
    z_m::T
    n_below::T
    n_above::T
    L_m::T
end
function LalicMihailovicLAD(LAI::Real, h::Real; z_m::Real,
                            n_below::Real = 6, n_above::Real = 1/2)
    LAI >= 0 || throw(ArgumentError("LalicMihailovicLAD: LAI must be ≥ 0"))
    0 < z_m < h || throw(ArgumentError("LalicMihailovicLAD: need 0 < z_m < h"))
    hF, zF, nb, na = Float64(h), Float64(z_m), Float64(n_below), Float64(n_above)
    shape_int = _integrate(z -> _lm_shape(z, hF, zF, nb, na), 0.0, hF, (zF,))
    return LalicMihailovicLAD{Float64}(Float64(LAI), hF, zF, nb, na,
                                       Float64(LAI) / shape_int)
end
leaf_area_density(p::LalicMihailovicLAD, z::Real) =
    p.L_m * _lm_shape(z, p.h, p.z_m, p.n_below, p.n_above)
_lad_breakpoints(p::LalicMihailovicLAD) = (p.z_m,)

"""
    BetaLAD(LAI, h; p = 3, q = 2)

Beta-distribution LAD profile (Markkanen et al. 2003):

```math
a(z) = \\frac{LAI}{h}\\,
       \\frac{(z/h)^{p-1}\\,(1 - z/h)^{q-1}}{B(p, q)},\\qquad 0 ≤ z ≤ h,
```

which integrates to LAI exactly. The mean leaf height is
``h\\,p/(p+q)`` and the maximum is at ``h\\,(p-1)/(p+q-2)``. `p, q ≥ 1`
keeps the profile finite. ``B(p,q)`` is evaluated from
`P3Microphysics.lgamma_lanczos`.
"""
struct BetaLAD{T<:AbstractFloat} <: AbstractLeafAreaDensity
    LAI::T
    h::T
    p::T
    q::T
    logB::T
end
function BetaLAD(LAI::Real, h::Real; p::Real = 3, q::Real = 2)
    LAI >= 0 || throw(ArgumentError("BetaLAD: LAI must be ≥ 0"))
    h > 0 || throw(ArgumentError("BetaLAD: h must be > 0"))
    (p >= 1 && q >= 1) || throw(ArgumentError("BetaLAD: need p, q ≥ 1"))
    logB = lgamma_lanczos(p) + lgamma_lanczos(q) - lgamma_lanczos(p + q)
    return BetaLAD{Float64}(Float64(LAI), Float64(h), Float64(p), Float64(q), logB)
end
function leaf_area_density(b::BetaLAD, z::Real)
    (z < 0 || z > b.h) && return zero(b.LAI)
    x = z / b.h
    # (p-1) or (q-1) may be 0 with x at an end: 0^0 = 1 as intended.
    return b.LAI / b.h * x^(b.p - 1) * (1 - x)^(b.q - 1) * exp(-b.logB)
end
_lad_breakpoints(::BetaLAD) = ()

"""
    canopy_height(p::AbstractLeafAreaDensity) -> h [m]
"""
canopy_height(p::AbstractLeafAreaDensity) = p.h

"""
    leaf_area_index(p::AbstractLeafAreaDensity) -> LAI [m² m⁻²]

The leaf-area index the profile was constructed with, equal to
``\\int_0^h a(z)\\,dz``.
"""
leaf_area_index(p::AbstractLeafAreaDensity) = p.LAI

_lad_breakpoints(::AbstractLeafAreaDensity) = ()

# Composite Gauss–Legendre on [lo, hi], split at the given interior
# breakpoints (kinks of the integrand). Each smooth piece is cut into
# sub-intervals graded geometrically towards both of its ends (`nlev`
# halvings), so algebraic end-point singularities of the derivative —
# the (1 - z/h)^(q-1) of a beta profile with non-integer q, the n = 1/2
# branch of Lalic–Mihailovic — are integrated to round-off as well.
const _GRADED = let nlev = 20
    f = [0.5^k / 2 for k in nlev:-1:0]            # 2^-21 … 1/2
    [0.0; f; reverse(1 .- f[1:end-1]); 1.0]
end
function _integrate(f, lo::Real, hi::Real, breaks = (); npts::Int = 16)
    hi <= lo && return 0.0
    x, w = _gauss_legendre(npts)
    edges = sort!(Float64[lo; [b for b in breaks if lo < b < hi]; hi])
    s = 0.0
    for e in 1:length(edges)-1
        a0, a1 = edges[e], edges[e+1]
        L = a1 - a0
        for m in 1:length(_GRADED)-1
            c0 = a0 + _GRADED[m] * L; c1 = a0 + _GRADED[m+1] * L
            c = 0.5 * (c0 + c1); half = 0.5 * (c1 - c0)
            for i in eachindex(x)
                s += w[i] * half * f(c + half * x[i])
            end
        end
    end
    return s
end

_profile_breaks(p::AbstractLeafAreaDensity) = (_lad_breakpoints(p)..., canopy_height(p))

"""
    cumulative_leaf_area(p::AbstractLeafAreaDensity, z) -> Λ(z) [m² m⁻²]

Leaf area *above* height `z`, ``Λ(z) = \\int_z^{h} a(z')\\,dz'``: zero at
the canopy top, the LAI at the ground. This is the optical-depth
coordinate of canopy radiative transfer (Beer's law
``I(z) = I_0 e^{-G Λ(z)/\\mu}``).
"""
function cumulative_leaf_area(p::AbstractLeafAreaDensity, z::Real)
    h = canopy_height(p)
    z >= h && return 0.0
    return _integrate(zz -> leaf_area_density(p, zz), max(Float64(z), 0.0), h,
                      _profile_breaks(p))
end

"""
    layer_leaf_area_density(p, z_faces) -> Vector{Float64}

Layer-mean LAD [m² m⁻³] of profile `p` on the layers bounded by the
increasing heights `z_faces` (length `N+1` → `N` layers):
``a_k = \\frac{1}{Δz_k}\\int_{z_k}^{z_{k+1}} a\\,dz``.

Because the layer *integral* rather than a point sample is used, the
discrete column integral ``\\sum_k a_k Δz_k`` equals the leaf area between
`z_faces[1]` and `z_faces[end]` to quadrature accuracy (round-off), however
coarse the grid — a point sample at layer centres would not conserve LAI
when a layer straddles the canopy top.
"""
function layer_leaf_area_density(p::AbstractLeafAreaDensity, z_faces::AbstractVector{<:Real})
    N = length(z_faces) - 1
    N >= 1 || throw(ArgumentError("layer_leaf_area_density: need ≥ 2 faces"))
    br = _profile_breaks(p)
    a = zeros(Float64, N)
    for k in 1:N
        z0, z1 = Float64(z_faces[k]), Float64(z_faces[k+1])
        z1 > z0 || throw(ArgumentError("layer_leaf_area_density: faces must increase"))
        lo, hi = max(z0, 0.0), min(z1, canopy_height(p))
        a[k] = hi > lo ? _integrate(zz -> leaf_area_density(p, zz), lo, hi, br) / (z1 - z0) : 0.0
    end
    return a
end

# =========================================================================
# 2. Implicit canopy drag
# =========================================================================

"""
    canopy_drag_factor(C_d, a, speed, dt) -> f ∈ (0, 1]

Factor by which the canopy drag ``dU_i/dt = -C_d\\,a\\,|U|\\,U_i``
(Shaw & Schumann 1992) multiplies the velocity over a step `dt`:

```math
f = \\frac{1}{1 + Δt\\,C_d\\,a\\,|U|}.
```

With the speed taken at the start of the step, ``U\\,f`` is the *exact*
solution of the pure drag equation — the speed obeys
``d|U|/dt = -C_d a |U|^2`` with solution ``|U|_0/(1 + C_d a |U|_0 t)``
and the direction does not change — and equivalently a backward-Euler
step with the drag coefficient lagged. Because ``0 < f ≤ 1`` for every
`dt`, the update is unconditionally stable and can never reverse the
flow, which explicit Euler (``1 - Δt C_d a |U|``) does once
``Δt\\,C_d\\,a\\,|U| > 1`` and amplifies once it exceeds 2.
"""
canopy_drag_factor(C_d::Real, a::Real, speed::Real, dt::Real) =
    1 / (1 + dt * C_d * a * speed)

"""
    canopy_drag_force(C_d, a, u, v, w) -> (Fu, Fv, Fw)

Instantaneous kinematic canopy drag force per unit mass [m s⁻²],
``F_i = -C_d\\,a\\,|U|\\,U_i``. A diagnostic; the model applies the drag
through [`implicit_canopy_drag`](@ref).
"""
function canopy_drag_force(C_d::Real, a::Real, u::Real, v::Real, w::Real)
    k = -C_d * a * sqrt(u^2 + v^2 + w^2)
    return (k * u, k * v, k * w)
end

"""
    implicit_canopy_drag(C_d, a, u, v, w, dt) -> (u', v', w')

Velocity after one step `dt` of canopy drag alone, all three components
scaled by the same [`canopy_drag_factor`](@ref) of the local speed. The
momentum removed, ``(U - U')/Δt``, is the step-mean drag of the exact
solution; it tends to ``C_d a |U| U`` as ``Δt → 0``.
"""
function implicit_canopy_drag(C_d::Real, a::Real, u::Real, v::Real, w::Real, dt::Real)
    f = canopy_drag_factor(C_d, a, sqrt(u^2 + v^2 + w^2), dt)
    return (f * u, f * v, f * w)
end

"""
    ResolvedCanopyDrag(mesh, lad; C_d = 0.2)

Canopy drag for a [`BoussinesqSimulation`](@ref Aeolus.Simulations.BoussinesqSimulation)
(or any simulation with `mesh`, `bcs` and staggered `u`, `v`, `w` fields)
on a `RegularGrid`. `lad` is the cell-centred leaf-area density
[m² m⁻³], an `(Nx, Ny, Nz)` array of interior cells (see
[`canopy_lad_field`](@ref)). At construction the LAD is averaged onto the
`u`, `v` and `w` faces (periodic wrap in x and y, as the LES is
horizontally periodic); the struct also carries the scratch arrays of
drag factors, so [`apply_canopy_drag!`](@ref) does not allocate.

`C_d` is the leaf drag coefficient, ≈ 0.15-0.3 for forests (Shaw &
Schumann 1992 use 0.15).
"""
struct ResolvedCanopyDrag{T<:AbstractFloat}
    C_d::T
    lad::Array{T,3}
    a_u::Array{T,3}
    a_v::Array{T,3}
    a_w::Array{T,3}
    f_u::Array{T,3}
    f_v::Array{T,3}
    f_w::Array{T,3}
end
function ResolvedCanopyDrag(mesh::RegularGrid{T}, lad::AbstractArray{<:Real,3};
                            C_d::Real = 0.2) where {T}
    Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
    size(lad) == (Nx, Ny, Nz) ||
        throw(DimensionMismatch("ResolvedCanopyDrag: lad must be (Nx, Ny, Nz) = $((Nx, Ny, Nz))"))
    all(>=(0), lad) || throw(ArgumentError("ResolvedCanopyDrag: lad must be ≥ 0"))
    C_d >= 0 || throw(ArgumentError("ResolvedCanopyDrag: C_d must be ≥ 0"))
    a = T.(lad)
    a_u = similar(a); a_v = similar(a); a_w = similar(a)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        im = i == 1 ? Nx : i - 1          # u face i: between cells i-1 and i
        jm = j == 1 ? Ny : j - 1          # v face j: between cells j-1 and j
        km = max(k - 1, 1)                # w face k: between cells k-1 and k
        a_u[i, j, k] = (a[im, j, k] + a[i, j, k]) / 2
        a_v[i, j, k] = (a[i, jm, k] + a[i, j, k]) / 2
        a_w[i, j, k] = (a[i, j, km] + a[i, j, k]) / 2
    end
    return ResolvedCanopyDrag{T}(T(C_d), a, a_u, a_v, a_w,
                                 ones(T, Nx, Ny, Nz), ones(T, Nx, Ny, Nz), ones(T, Nx, Ny, Nz))
end

"""
    canopy_lad_field(mesh, profiles) -> Array{Float64,3}

Cell-centred leaf-area density `(Nx, Ny, Nz)` on a `RegularGrid` from a
per-column profile: `profiles[i, j]` is an [`AbstractLeafAreaDensity`](@ref)
or `nothing` (no vegetation). Heights are measured from `mesh.z0`. Each
column's layer values are layer integrals
([`layer_leaf_area_density`](@ref)), so every column sums to its LAI.

    canopy_lad_field(mesh, profile::AbstractLeafAreaDensity, mask::AbstractMatrix{Bool})

is the same with one profile wherever `mask` is `true`.
"""
function canopy_lad_field(mesh::RegularGrid, profiles::AbstractMatrix)
    Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
    size(profiles) == (Nx, Ny) ||
        throw(DimensionMismatch("canopy_lad_field: profiles must be (Nx, Ny)"))
    zf = [(k - 1) * mesh.dz for k in 1:Nz+1]
    lad = zeros(Float64, Nx, Ny, Nz)
    cache = IdDict{Any, Vector{Float64}}()
    for j in 1:Ny, i in 1:Nx
        p = profiles[i, j]
        p === nothing && continue
        col = get!(() -> layer_leaf_area_density(p, zf), cache, p)
        @views lad[i, j, :] .= col
    end
    return lad
end
function canopy_lad_field(mesh::RegularGrid, profile::AbstractLeafAreaDensity,
                          mask::AbstractMatrix{Bool})
    return canopy_lad_field(mesh, [m ? profile : nothing for m in mask])
end

"""
    apply_canopy_drag!(sim, drag::ResolvedCanopyDrag, dt) -> (Fx, Fy, Fz)

Apply the canopy drag implicitly to the velocity of `sim`: every face
velocity is multiplied by ``1/(1 + Δt\\,C_d\\,a\\,|U|)`` with ``a`` the LAD
at that face and ``|U|`` the full speed there (the other two components
averaged from their four neighbouring faces), all factors computed from
the velocity *before* any is applied.

Call it from the `post_predictor!` hook of
`step!`, i.e. on the predicted velocity
``u^*`` and before the pressure projection:

```julia
drag = ResolvedCanopyDrag(mesh, canopy_lad_field(mesh, LalicMihailovicLAD(4.0, 20.0; z_m = 12.0), mask))
step!(sim, dt; post_predictor! = (s, Δt) -> apply_canopy_drag!(s, drag, Δt))
```

This replaces the explicit tendency ``-C_d a |U| U``: that one is
unstable once ``Δt\\,C_d\\,a\\,|U| > 2`` (for ``C_d = 0.2``,
``a = 1\\,\\mathrm{m^{-1}}`` and ``|U| = 5\\,\\mathrm{m\\,s^{-1}}`` that is any
``Δt > 2\\,\\mathrm{s}``); this one is stable for every ``Δt``.

Returns the volume-integrated kinematic force the step exerted,
``\\sum (U' - U)/Δt\\,ΔV`` [m⁴ s⁻²] per component (negative for a flow
in the positive direction); divide by the domain area for the canopy
momentum sink per unit ground area (a stress, m² s⁻²).
"""
function apply_canopy_drag!(sim, drag::ResolvedCanopyDrag{T}, dt::Real) where {T}
    mesh = sim.mesh
    Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
    Hx, Hy, Hz = mesh.Hx, mesh.Hy, mesh.Hz
    Δt = T(dt); C = drag.C_d
    fill_halos!(sim.u, sim.bcs); fill_halos!(sim.v, sim.bcs); fill_halos!(sim.w, sim.bcs)
    u = sim.u.data; v = sim.v.data; w = sim.w.data
    q = T(1) / 4
    # 1. All factors from the pre-drag velocity.
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        I = Hx + i; J = Hy + j; K = Hz + k
        au = drag.a_u[i, j, k]
        if au > 0
            uu = u[I, J, K]
            vv = q * (v[I-1, J, K] + v[I, J, K] + v[I-1, J+1, K] + v[I, J+1, K])
            ww = q * (w[I-1, J, K] + w[I, J, K] + w[I-1, J, K+1] + w[I, J, K+1])
            drag.f_u[i, j, k] = canopy_drag_factor(C, au, sqrt(uu^2 + vv^2 + ww^2), Δt)
        end
        av = drag.a_v[i, j, k]
        if av > 0
            vv = v[I, J, K]
            uu = q * (u[I, J-1, K] + u[I+1, J-1, K] + u[I, J, K] + u[I+1, J, K])
            ww = q * (w[I, J-1, K] + w[I, J, K] + w[I, J-1, K+1] + w[I, J, K+1])
            drag.f_v[i, j, k] = canopy_drag_factor(C, av, sqrt(uu^2 + vv^2 + ww^2), Δt)
        end
        aw = drag.a_w[i, j, k]
        if aw > 0
            ww = w[I, J, K]
            uu = q * (u[I, J, K-1] + u[I+1, J, K-1] + u[I, J, K] + u[I+1, J, K])
            vv = q * (v[I, J, K-1] + v[I, J+1, K-1] + v[I, J, K] + v[I, J+1, K])
            drag.f_w[i, j, k] = canopy_drag_factor(C, aw, sqrt(uu^2 + vv^2 + ww^2), Δt)
        end
    end
    # 2. Apply, and integrate the momentum removed.
    Fx = zero(T); Fy = zero(T); Fz = zero(T)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        I = Hx + i; J = Hy + j; K = Hz + k
        if drag.a_u[i, j, k] > 0
            u0 = u[I, J, K]; u[I, J, K] = u0 * drag.f_u[i, j, k]; Fx += u[I, J, K] - u0
        end
        if drag.a_v[i, j, k] > 0
            v0 = v[I, J, K]; v[I, J, K] = v0 * drag.f_v[i, j, k]; Fy += v[I, J, K] - v0
        end
        if drag.a_w[i, j, k] > 0
            w0 = w[I, J, K]; w[I, J, K] = w0 * drag.f_w[i, j, k]; Fz += w[I, J, K] - w0
        end
    end
    dV = mesh.dx * mesh.dy * mesh.dz
    return (Fx = Fx * dV / Δt, Fy = Fy * dV / Δt, Fz = Fz * dV / Δt)
end

# =========================================================================
# 3. In-canopy mean wind: 1-D column and the exponential profile
# =========================================================================

"""
    harman_finnigan_mixing_length(β, C_d, a) -> l [m]

Canopy mixing length of Harman & Finnigan (2007),
``l = 2β^3 L_c`` with the canopy-drag length ``L_c = 1/(C_d a)`` and
``β = u_*/U_h`` the ratio of friction velocity to wind speed at the
canopy top (≈ 0.3 in neutral flow over dense canopies).
"""
harman_finnigan_mixing_length(β::Real, C_d::Real, a::Real) = 2 * β^3 / (C_d * a)

"""
    canopy_attenuation_coefficient(C_d, a, l) -> γ [m⁻¹]

Exponential decay rate of the mean wind into a uniform canopy for a
constant mixing length `l` (Inoue 1963):

```math
γ = \\left(\\frac{C_d\\,a}{2\\,l^2}\\right)^{1/3},
```

from the steady balance ``\\frac{d}{dz}\\left(l^2 \\left|\\frac{dU}{dz}\\right|\\frac{dU}{dz}\\right)
= C_d\\,a\\,U^2`` with ``U = U_h e^{γ(z-h)}``. Inoue's dimensionless
attenuation coefficient is ``γh``. With the Harman–Finnigan mixing
length this is ``γ = β/l``.
"""
canopy_attenuation_coefficient(C_d::Real, a::Real, l::Real) = cbrt(C_d * a / (2 * l^2))

"""
    canopy_exponential_wind(z; h, u_h, C_d, a, β = 0.3, l_mix = harman_finnigan_mixing_length(β, C_d, a))

The exponential in-canopy wind of Inoue (1963) / Harman & Finnigan
(2007): ``U(z) = U_h \\exp[γ (z - h)]`` for ``z ≤ h`` with
``γ = `` [`canopy_attenuation_coefficient`](@ref)`(C_d, a, l_mix)`. It is
the exact solution of the uniform-canopy mixing-length balance for a
canopy deep enough (``γh ≫ 1``) that the ground is not felt.
"""
function canopy_exponential_wind(z::Real; h::Real, u_h::Real, C_d::Real, a::Real,
                                 β::Real = 0.3,
                                 l_mix::Real = harman_finnigan_mixing_length(β, C_d, a))
    γ = canopy_attenuation_coefficient(C_d, a, l_mix)
    return u_h * exp(γ * (min(z, h) - h))
end

"""
    CanopyWindSolution

Result of [`solve_canopy_wind_column`](@ref): layer-centre heights `z`
and wind `u` [m s⁻¹], the friction velocity at the column top `u_star`
(from the resolved stress ``K\\,∂U/∂z`` there), the canopy drag
integrated over the column `drag_integral` = ``\\sum C_d a_k |U_k| U_k Δz_k``
[m² s⁻²], the kinematic stress on the ground `tau_ground` (zero for a
free-slip bottom), the number of iterations and the final update norm
`residual` (max ``|ΔU|/U_{top}``).
"""
struct CanopyWindSolution
    z::Vector{Float64}
    u::Vector{Float64}
    u_star::Float64
    drag_integral::Float64
    tau_ground::Float64
    iterations::Int
    residual::Float64
end

_lmix_at(l::Real, z) = Float64(l)
_lmix_at(l::Function, z) = Float64(l(z))

"""
    solve_canopy_wind_column(z_faces, lad; C_d, l_mix, u_top, ν = 1.5e-5,
                             bottom = :free_slip, dt = 1e4, maxiter = 5_000,
                             tol = 1e-12) -> CanopyWindSolution

Steady mean wind in a 1-D canopy column driven by a prescribed wind
`u_top` at the top face `z_faces[end]`:

```math
0 = \\frac{∂}{∂z}\\left(K \\frac{∂U}{∂z}\\right) - C_d\\,a(z)\\,|U|\\,U,\\qquad
K = l^2 \\left|\\frac{∂U}{∂z}\\right| + ν .
```

`lad` holds layer-mean LAD values (e.g. from
[`layer_leaf_area_density`](@ref)); `l_mix` is a number or a function of
height evaluated at layer faces. `bottom = :free_slip` (zero stress) or
`:no_slip` (``U = 0`` at `z_faces[1]`).

The steady state is reached by pseudo-time marching in which the
diffusion is backward Euler with ``K`` lagged one iteration and the
drag is the implicit factor of [`canopy_drag_factor`](@ref) folded into
the same tridiagonal solve — so every iteration is unconditionally
stable and, at convergence, the discrete balance holds exactly: the
stress at the top equals `drag_integral + tau_ground`.

Keep `ν` small against ``l^2|∂U/∂z|`` where the comparison with the
inviscid exponential profile matters: deep in a dense canopy the
mixing-length viscosity decays like ``U`` and a molecular floor becomes
comparable.
"""
function solve_canopy_wind_column(z_faces::AbstractVector{<:Real}, lad::AbstractVector{<:Real};
                                  C_d::Real, l_mix, u_top::Real, ν::Real = 1.5e-5,
                                  bottom::Symbol = :free_slip, dt::Real = 1.0e4,
                                  maxiter::Integer = 5_000, tol::Real = 1e-12)
    N = length(z_faces) - 1
    length(lad) == N || throw(DimensionMismatch("solve_canopy_wind_column: lad must have one value per layer"))
    bottom in (:free_slip, :no_slip) ||
        throw(ArgumentError("solve_canopy_wind_column: bottom must be :free_slip or :no_slip"))
    zf = Float64.(z_faces)
    Δz = diff(zf)
    zc = (zf[1:N] .+ zf[2:N+1]) ./ 2
    # Distances between the nodes either side of each face; face 1 is the
    # ground (node: z_faces[1]), face N+1 the top (node: z_faces[end]).
    δ = similar(zf)
    δ[1] = zc[1] - zf[1]
    for f in 2:N
        δ[f] = zc[f] - zc[f-1]
    end
    δ[N+1] = zf[N+1] - zc[N]
    l2 = [_lmix_at(l_mix, zf[f])^2 for f in 1:N+1]
    Cd = Float64(C_d); a = Float64.(lad); ut = Float64(u_top); νF = Float64(ν); Δt = Float64(dt)
    u = [ut * (zc[k] - zf[1]) / (zf[N+1] - zf[1]) for k in 1:N]    # linear first guess
    unew = similar(u); K = zeros(N + 1); Kn = zeros(N + 1)
    lo = zeros(N); di = zeros(N); up = zeros(N); rhs = zeros(N)
    cw = zeros(N); dw = zeros(N)
    it = 0; res = Inf
    while it < maxiter
        it += 1
        # Eddy viscosity at faces from the current iterate, averaged with
        # the previous iteration's: plain Picard (K fully lagged) on the
        # quadratic flux l²|∂U/∂z|∂U/∂z overshoots and oscillates at large
        # pseudo-time steps; the average damps that two-cycle.
        Kn[1] = bottom === :no_slip ? l2[1] * abs(u[1]) / δ[1] + νF : 0.0
        for f in 2:N
            Kn[f] = l2[f] * abs(u[f] - u[f-1]) / δ[f] + νF
        end
        Kn[N+1] = l2[N+1] * abs(ut - u[N]) / δ[N+1] + νF
        for f in 1:N+1
            K[f] = it == 1 ? Kn[f] : (K[f] + Kn[f]) / 2
        end
        # (u' - u)/Δt = [K_{k+1}(u'_{k+1}-u'_k)/δ_{k+1} - K_k(u'_k-u'_{k-1})/δ_k]/Δz_k - Cd a_k |u_k| u'_k
        for k in 1:N
            cb = K[k] / (δ[k] * Δz[k]); ct = K[k+1] / (δ[k+1] * Δz[k])
            lo[k] = -Δt * cb
            up[k] = -Δt * ct
            di[k] = 1 + Δt * (cb + ct) + Δt * Cd * a[k] * abs(u[k])
            rhs[k] = u[k]
        end
        # Boundary nodes: ground (u = 0, no slip) and prescribed top wind.
        lo[1] = 0.0                     # u_0 = 0 (no slip) or K[1] = 0 (free slip)
        up[N] = 0.0; rhs[N] += Δt * K[N+1] / (δ[N+1] * Δz[N]) * ut
        _thomas!(unew, lo, di, up, rhs, cw, dw)
        res = 0.0
        for k in 1:N
            res = max(res, abs(unew[k] - u[k]))
        end
        res /= abs(ut)
        u, unew = unew, u
        res < tol && break
    end
    # Diagnostics with the converged K (at convergence K = Kn).
    τ_top = K[N+1] * (ut - u[N]) / δ[N+1]
    drag = 0.0
    for k in 1:N
        drag += Cd * a[k] * abs(u[k]) * u[k] * Δz[k]
    end
    τ_g = K[1] * u[1] / δ[1]
    return CanopyWindSolution(zc, u, sqrt(abs(τ_top)), drag, τ_g, it, res)
end

# Tridiagonal solve (Thomas algorithm); lo[1] and up[end] are ignored.
function _thomas!(x, lo, di, up, rhs, c, d)
    n = length(di)
    c[1] = up[1] / di[1]; d[1] = rhs[1] / di[1]
    for i in 2:n
        m = di[i] - lo[i] * c[i-1]
        c[i] = i < n ? up[i] / m : 0.0
        d[i] = (rhs[i] - lo[i] * d[i-1]) / m
    end
    x[n] = d[n]
    for i in n-1:-1:1
        x[i] = d[i] - c[i] * x[i+1]
    end
    return x
end

end # module
