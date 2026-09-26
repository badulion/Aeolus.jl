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
       layer_leaf_area_density

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

end # module
