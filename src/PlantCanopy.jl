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

  4. **Radiation in the canopy** — leaf-angle distributions with Ross's
     ``G``-function in the Goudriaan (1977) fit ([`RossGoudriaanLeaves`](@ref)),
     Beer–Lambert beam and diffuse transmittance, and the canopy
     two-stream model of Sellers (1985) for the shortwave
     ([`canopy_shortwave`](@ref)) with the sunlit/shaded split of
     de Pury & Farquhar (1997), plus the longwave exchange between sky,
     leaf layers and ground ([`canopy_longwave`](@ref)). The optical-depth
     coordinate is the cumulative leaf area [`cumulative_leaf_area`](@ref),
     and the two-stream layer algebra and adding solver are those of the
     Era III `TwoStream` module, supplied with the canopy γ coefficients.

The leaf energy balance (#24) extends this module.

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
  * Ross, J. (1981). *The Radiation Regime and Architecture of Plant
    Stands.* Junk, The Hague.
  * Goudriaan, J. (1977). *Crop Micrometeorology: A Simulation Study.*
    Pudoc, Wageningen.
  * Sellers, P. J. (1985). Canopy reflectance, photosynthesis and
    transpiration. *Int. J. Remote Sens.*, 6, 1335-1372.
  * de Pury, D. G. G. & Farquhar, G. D. (1997). Simple scaling of
    photosynthesis from leaves to canopies without the errors of big-leaf
    models. *Plant Cell Environ.*, 20, 537-557.
  * CLM5.0 Technical Note (Lawrence et al., NCAR), §2.3.1 "Canopy
    Radiative Transfer" — the two-stream equations and the
    ``G``, ``\bar μ``, ``β``, ``β_0`` expressions used here.
"""
module PlantCanopy

using ..Grids: RegularGrid
using ..BoundaryConditions: fill_halos!
using ..RadiationCore: _gauss_legendre
using ..P3Microphysics: lgamma_lanczos
using ..RadiationCore: STEFAN_BOLTZMANN
using ..TwoStream: TwoStreamVariant, layer_reflectance_transmittance_sw, adding_solver!
import ..TwoStream: two_stream_coefficients
using ..SolarGeometry: SolarPosition

export AbstractLeafAreaDensity, UniformLAD, LalicMihailovicLAD, BetaLAD,
       leaf_area_density, leaf_area_index, canopy_height, cumulative_leaf_area,
       layer_leaf_area_density,
       canopy_drag_factor, canopy_drag_force, implicit_canopy_drag,
       ResolvedCanopyDrag, canopy_lad_field, apply_canopy_drag!,
       CanopyWindSolution, solve_canopy_wind_column,
       harman_finnigan_mixing_length, canopy_attenuation_coefficient,
       canopy_exponential_wind,
       AbstractLeafAngleDistribution, RossGoudriaanLeaves, SphericalLeaves,
       HorizontalLeaves, LeafOptics, leaf_projection_function,
       beam_extinction_coefficient, average_inverse_optical_depth,
       canopy_scattering_parameters, canopy_beam_transmittance,
       sunlit_leaf_fraction, canopy_diffuse_transmittance,
       CanopyShortwaveSolution, canopy_shortwave,
       CanopyLongwaveSolution, canopy_longwave, canopy_shortwave_field!

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

# =========================================================================
# 4. Radiation in the canopy (#23)
# =========================================================================
#
# The optical-depth coordinate is the cumulative leaf area Λ above a
# height (`cumulative_leaf_area`). What decides how much of a beam a unit
# of leaf area intercepts is the leaf orientation, through the
# Ross (1981) projection function G(μ); what happens to the intercepted
# light is the leaf single-scattering albedo ω = α + τ_l. The multiple
# scattering between leaves and soil is the two-stream problem of
# Sellers (1985), which is solved here with the Era III two-stream layer
# formulae and adding solver of `TwoStream` — not a second copy of them —
# by supplying the canopy's γ coefficients through a `TwoStreamVariant`.

"""
    AbstractLeafAngleDistribution

Leaf orientation of a canopy. An implementer provides

  * `leaf_projection_function(d, μ)` — Ross's ``G(μ)``: mean projection of
    unit leaf area onto the plane normal to a direction with cosine of
    zenith angle ``μ``;
  * `average_inverse_optical_depth(d)` — Sellers' ``\\bar μ =
    \\int_0^1 μ'/G(μ')\\,dμ'``, the diffuse-stream cosine;
  * `canopy_scattering_parameters(d, optics, μ)` — ``(ω, β, β_0)``, the
    leaf single-scattering albedo and the up-scatter fractions of diffuse
    and of direct-beam radiation.
"""
abstract type AbstractLeafAngleDistribution end

"""
    RossGoudriaanLeaves(χ_L = 0)

Leaf-angle distribution described by Ross's departure index ``χ_L``
(``-1`` vertical, ``0`` spherical/random, ``+1`` horizontal), with the
projection function fitted by Goudriaan (1977):

```math
G(μ) = φ_1 + φ_2 μ,\\qquad φ_1 = 0.5 - 0.633χ_L - 0.33χ_L^2,\\qquad
φ_2 = 0.877\\,(1 - 2φ_1).
```

The fit is only valid for ``-0.4 ≤ χ_L ≤ 0.6`` (outside it ``G`` can turn
negative), which the constructor enforces — the same range CLM uses
(CLM5.0 Technical Note, §2.3.1). `χ_L = 0` gives the spherical distribution,
``G = 1/2`` exactly. For perfectly horizontal leaves use
[`HorizontalLeaves`](@ref).
"""
struct RossGoudriaanLeaves <: AbstractLeafAngleDistribution
    χ::Float64
    φ1::Float64
    φ2::Float64
    function RossGoudriaanLeaves(χ::Real = 0.0)
        -0.4 <= χ <= 0.6 ||
            throw(ArgumentError("RossGoudriaanLeaves: χ_L must lie in [-0.4, 0.6] (validity of the Goudriaan fit)"))
        φ1 = 0.5 - 0.633χ - 0.33χ^2
        new(Float64(χ), φ1, 0.877 * (1 - 2φ1))
    end
end

"""
    SphericalLeaves() = RossGoudriaanLeaves(0.0)

Randomly (spherically) oriented leaves: ``G(μ) = 1/2`` for every direction.
"""
SphericalLeaves() = RossGoudriaanLeaves(0.0)

"""
    HorizontalLeaves()

All leaves horizontal: ``G(μ) = μ``, so every direction is extinguished at
exactly one unit per unit leaf area (``K = 1``, ``\\bar μ = 1``) and a
scattering event sends the leaf reflectance up and the leaf transmittance
down whatever the incidence angle (``β = β_0 = α/ω``). For this canopy the
two-stream equations are exact rather than an approximation, which makes
it a reference case (Goudriaan 1977).
"""
struct HorizontalLeaves <: AbstractLeafAngleDistribution end

"""
    LeafOptics(reflectance, transmittance)

Hemispherical reflectance ``α`` and transmittance ``τ_l`` of a single leaf
in one spectral band (bi-Lambertian leaf). Their sum is the leaf
single-scattering albedo ``ω``. Typical broadleaf values are
``α ≈ 0.10, τ_l ≈ 0.05`` in the PAR band and ``α ≈ 0.45, τ_l ≈ 0.25``
in the near infrared; in the thermal infrared ``α = 1 - ε_{leaf}``,
``τ_l ≈ 0``.
"""
struct LeafOptics
    reflectance::Float64
    transmittance::Float64
    function LeafOptics(α::Real, τ::Real)
        (α >= 0 && τ >= 0 && α + τ <= 1) ||
            throw(ArgumentError("LeafOptics: need α ≥ 0, τ ≥ 0, α + τ ≤ 1"))
        new(Float64(α), Float64(τ))
    end
end

"""
    leaf_projection_function(d::AbstractLeafAngleDistribution, μ) -> G(μ)

Ross's ``G``-function: the projected area of unit leaf area in the
direction whose zenith-angle cosine is `μ`, averaged over the leaf-angle
distribution. For any real distribution ``\\int_0^1 G\\,dμ = 1/2``.
"""
leaf_projection_function(d::RossGoudriaanLeaves, μ::Real) = d.φ1 + d.φ2 * μ
leaf_projection_function(::HorizontalLeaves, μ::Real) = Float64(μ)

"""
    beam_extinction_coefficient(d, μ) -> K = G(μ)/μ

Extinction coefficient of the direct beam per unit leaf area for a sun
with zenith-angle cosine `μ`: the beam transmitted to cumulative leaf area
``Λ`` is ``e^{-KΛ}`` (Beer–Lambert).
"""
beam_extinction_coefficient(d::AbstractLeafAngleDistribution, μ::Real) =
    leaf_projection_function(d, μ) / μ

"""
    average_inverse_optical_depth(d) -> μ̄

Sellers' (1985) ``\\bar μ = \\int_0^1 μ'/G(μ')\\,dμ'``, the average inverse
diffuse optical depth per unit leaf area. The two-stream diffuse fluxes
are extinguished at ``1/\\bar μ`` per unit leaf area. For the Goudriaan fit

```math
\\bar μ = \\frac{1}{φ_2}\\left[1 - \\frac{φ_1}{φ_2}
          \\ln\\frac{φ_1 + φ_2}{φ_1}\\right],
```

evaluated by its series when ``φ_2/φ_1`` is small (the closed form is
``0/0`` at ``χ_L = 0``, where ``\\bar μ = 1``).
"""
# ∫₀¹ μ'/(a + b μ') dμ' = [1 - ln(1 + x)/x]/b with x = b/a, for a > 0 and
# a + b > 0. The closed form is 0/0 as b → 0, so for small |x| its series
# (1/a)(1/2 - x/3 + x²/4 - x³/5 + x⁴/6 - …) is used (truncation < 2e-16).
function _int_mu_over_linear(a::Float64, b::Float64)
    x = b / a
    abs(x) < 1e-3 && return (0.5 - x / 3 + x^2 / 4 - x^3 / 5 + x^4 / 6) / a
    return (1 - log1p(x) / x) / b
end

average_inverse_optical_depth(d::RossGoudriaanLeaves) = _int_mu_over_linear(d.φ1, d.φ2)
average_inverse_optical_depth(::HorizontalLeaves) = 1.0

"""
    canopy_scattering_parameters(d, optics::LeafOptics, μ) -> (ω, β, β₀)

Leaf single-scattering albedo ``ω = α + τ_l`` and the fractions of
scattered radiation sent upward, ``β`` for diffuse and ``β_0`` for direct
radiation from a sun at zenith cosine `μ`, as in Sellers (1985) and
the CLM5.0 Technical Note (§2.3.1, eqs. 2.3.13–2.3.16):

```math
ωβ = \\tfrac12\\left[α + τ_l + (α - τ_l)\\cos^2\\bar θ\\right],
\\qquad \\cos\\bar θ = \\frac{1 + χ_L}{2},
\\qquad ωβ_0 = \\frac{1 + \\bar μ K}{\\bar μ K}\\,a_s(μ),
```

with the single-scattering albedo of the canopy for isotropically
scattering leaves

```math
a_s(μ) = \\frac{ω}{2}\\int_0^1 \\frac{μ' G(μ)}{μ G(μ') + μ' G(μ)}\\,dμ'
       = \\frac{ω}{2}\\frac{G}{μφ_2 + G}\\left[1 - \\frac{μφ_1}{μφ_2 + G}
         \\ln\\frac{μφ_1 + μφ_2 + G}{μφ_1}\\right].
```
"""
function canopy_scattering_parameters(d::RossGoudriaanLeaves, o::LeafOptics, μ::Real)
    α, τ = o.reflectance, o.transmittance
    ω = α + τ
    ω <= 0 && return (0.0, 0.0, 0.0)
    cosθ = (1 + d.χ) / 2
    β = (α + τ + (α - τ) * cosθ^2) / (2ω)
    μ = Float64(μ)
    G = leaf_projection_function(d, μ)
    K = G / μ
    μ̄ = average_inverse_optical_depth(d)
    φ1, φ2 = d.φ1, d.φ2
    # the integrand's denominator is μφ₁ + μ'(μφ₂ + G); μφ₂ + G passes
    # through zero near μ = 1 for χ_L → -0.4, where the closed form is 0/0
    a_s = (ω / 2) * G * _int_mu_over_linear(μ * φ1, μ * φ2 + G)
    β0 = (1 + μ̄ * K) / (μ̄ * K) * a_s / ω
    return (ω, β, β0)
end
function canopy_scattering_parameters(::HorizontalLeaves, o::LeafOptics, μ::Real)
    ω = o.reflectance + o.transmittance
    ω <= 0 && return (0.0, 0.0, 0.0)
    β = o.reflectance / ω
    return (ω, β, β)
end

"""
    canopy_beam_transmittance(d, μ, Λ) -> e^{-G(μ)Λ/μ}

Beer–Lambert transmittance of the direct beam through cumulative leaf area
`Λ`. It is also the probability that a leaf at depth `Λ` is sunlit.
"""
canopy_beam_transmittance(d::AbstractLeafAngleDistribution, μ::Real, Λ::Real) =
    μ > 0 ? exp(-beam_extinction_coefficient(d, μ) * Λ) : 0.0

"""
    sunlit_leaf_fraction(d, μ, Λ) -> f_sun

Fraction of the leaves at cumulative leaf area `Λ` that see the sun, which
for randomly placed leaves equals the beam transmittance ``e^{-KΛ}``
(Monsi & Saeki 1953; de Pury & Farquhar 1997).
"""
sunlit_leaf_fraction(d::AbstractLeafAngleDistribution, μ::Real, Λ::Real) =
    canopy_beam_transmittance(d, μ, Λ)

"""
    canopy_diffuse_transmittance(d, Λ; n = 64) -> T_d

Exact Beer–Lambert transmittance of isotropic (uniform-radiance) diffuse
light through cumulative leaf area `Λ` of **black** leaves,

```math
T_d(Λ) = 2\\int_0^1 e^{-G(μ)Λ/μ}\\,μ\\,dμ ,
```

which for spherical leaves is ``2E_3(Λ/2)``. The two-stream model replaces
this by ``e^{-Λ/\\bar μ}``; the difference is the angular-closure error of
the two-stream approximation (see the theory page).
"""
function canopy_diffuse_transmittance(d::AbstractLeafAngleDistribution, Λ::Real; n::Int = 64)
    Λ <= 0 && return 1.0
    # graded Gauss-Legendre on (0,1): the integrand has an essential
    # singularity e^{-GΛ/μ} at μ → 0, so resolve small μ geometrically
    x, w = _gauss_legendre(n)
    s = 0.0
    lo = 0.0
    for m in 12:-1:0
        hi = 2.0^(-m)
        c = (hi + lo) / 2; r = (hi - lo) / 2
        for q in eachindex(x)
            μ = c + r * x[q]
            s += w[q] * r * exp(-leaf_projection_function(d, μ) * Λ / μ) * μ
        end
        lo = hi
    end
    return 2s
end

# ---- two-stream variant carrying the canopy γ coefficients --------------

# Sellers (1985) eqs. in the Meador–Weaver form used by `TwoStream`, with
# optical depth τ = Λ, beam "μ₀" = 1/K and:
#   γ₁ = [1 - (1-β)ω]/μ̄,  γ₂ = ωβ/μ̄,  γ₃ = β₀,  γ₄ = 1 - β₀.
struct SellersCanopyVariant <: TwoStreamVariant
    μ̄::Float64
    β::Float64
    β0::Float64
end
@inline function two_stream_coefficients(v::SellersCanopyVariant, ω::T, g::T, μ₀::T) where {T}
    γ1 = (1 - (1 - v.β) * ω) / v.μ̄
    γ2 = ω * v.β / v.μ̄
    return (T(γ1), T(γ2), T(v.β0), T(1 - v.β0))
end

"""
    CanopyShortwaveSolution

Result of [`canopy_shortwave`](@ref) for one spectral band. Face arrays
(length `N+1`) and layer arrays (length `N`) are ordered bottom-up like
`z_faces`; fluxes are per unit ground area [W m⁻²].

  * `z_faces`, `Λ_faces` — heights and cumulative leaf area above each face
  * `flux_dn_dir`, `flux_dn_dif`, `flux_up` — direct and diffuse downward,
    and upward, fluxes at the faces
  * `absorbed` — absorbed by the leaves of each layer
  * `absorbed_sunlit`, `absorbed_shaded` — the same split between sunlit
    and shaded leaves (they sum to `absorbed`)
  * `lai_sunlit`, `lai_shaded` — sunlit and shaded leaf area of each layer
    (sum to the layer LAI)
  * `incident`, `reflected`, `canopy_absorbed`, `ground_absorbed` — column
    totals; `incident = reflected + canopy_absorbed + ground_absorbed`
  * `K`, `μ̄`, `ω`, `β`, `β0` — the optical parameters used
"""
struct CanopyShortwaveSolution
    z_faces::Vector{Float64}
    Λ_faces::Vector{Float64}
    flux_dn_dir::Vector{Float64}
    flux_dn_dif::Vector{Float64}
    flux_up::Vector{Float64}
    absorbed::Vector{Float64}
    absorbed_sunlit::Vector{Float64}
    absorbed_shaded::Vector{Float64}
    lai_sunlit::Vector{Float64}
    lai_shaded::Vector{Float64}
    incident::Float64
    reflected::Float64
    canopy_absorbed::Float64
    ground_absorbed::Float64
    K::Float64
    μ̄::Float64
    ω::Float64
    β::Float64
    β0::Float64
end

_cos_zenith(μ::Real) = Float64(μ)
_cos_zenith(p::SolarPosition) = Float64(p.μ₀)

"""
    canopy_shortwave(z_faces, lad; sun, sw_direct, sw_diffuse, leaves = SphericalLeaves(),
                     optics::LeafOptics, albedo_direct, albedo_diffuse = albedo_direct)
        -> CanopyShortwaveSolution
    canopy_shortwave(profile::AbstractLeafAreaDensity, z_faces; kwargs...)

Shortwave radiation in one spectral band through a canopy of layer-mean
leaf-area densities `lad` [m² m⁻³] on the layers bounded by `z_faces`
(bottom-up), by the canopy two-stream model of Sellers (1985).

  * `sun` — cosine of the solar zenith angle, or a
    `SolarGeometry.SolarPosition` (its apparent ``μ_0`` is used); a sun at
    or below the horizon has no direct beam and no sunlit leaves
  * `sw_direct` — direct-beam irradiance **on a horizontal plane** at the
    top face [W m⁻²] (i.e. DNI·μ, or the `flux_dn_dir` of an Era III
    atmospheric solve at the surface level)
  * `sw_diffuse` — diffuse downward irradiance at the top face [W m⁻²]
  * `albedo_direct`, `albedo_diffuse` — ground (soil) albedo for the two

The layer reflectances and transmittances and the adding solution are
those of `TwoStream` (`layer_reflectance_transmittance_sw`,
`adding_solver!`) with the canopy γ coefficients
(``τ = Λ``, ``μ_0 → 1/K``, see [`canopy_scattering_parameters`](@ref)).
Because the equations depend on height only through ``Λ``, the solution
is independent of how the leaf area is layered.

Absorption is split between sunlit and shaded leaves as in
de Pury & Farquhar (1997) and CLM: sunlit leaves absorb the fraction
``1-ω`` of the unscattered beam they intercept, and the absorbed diffuse
(sky plus scattered) radiation of a layer is shared in proportion to leaf
area. The sunlit leaf area of a layer between cumulative leaf areas
``Λ_t < Λ_b`` is ``(e^{-KΛ_t} - e^{-KΛ_b})/K``.
"""
function canopy_shortwave(z_faces::AbstractVector{<:Real}, lad::AbstractVector{<:Real};
                          sun, sw_direct::Real, sw_diffuse::Real,
                          leaves::AbstractLeafAngleDistribution = SphericalLeaves(),
                          optics::LeafOptics,
                          albedo_direct::Real, albedo_diffuse::Real = albedo_direct)
    N = length(lad)
    length(z_faces) == N + 1 ||
        throw(DimensionMismatch("canopy_shortwave: need length(z_faces) == length(lad) + 1"))
    (sw_direct >= 0 && sw_diffuse >= 0) ||
        throw(ArgumentError("canopy_shortwave: irradiances must be ≥ 0"))
    zf = Float64.(z_faces)
    dΛ = [Float64(lad[k]) * (zf[k+1] - zf[k]) for k in 1:N]
    all(>=(0), dΛ) || throw(ArgumentError("canopy_shortwave: lad ≥ 0 and increasing z_faces required"))
    Λ = zeros(N + 1)
    for k in N:-1:1
        Λ[k] = Λ[k+1] + dΛ[k]
    end

    μ = _cos_zenith(sun)
    daylight = μ > 1e-6
    μs = daylight ? μ : 1.0                      # dummy geometry at night
    S0 = daylight ? Float64(sw_direct) : 0.0
    K = beam_extinction_coefficient(leaves, μs)
    μ̄ = average_inverse_optical_depth(leaves)
    ω, β, β0 = canopy_scattering_parameters(leaves, optics, μs)
    var = SellersCanopyVariant(μ̄, β, β0)
    μ0 = 1 / K

    Rd = zeros(N); Td = zeros(N); Rr = zeros(N); Tr = zeros(N); Tb = zeros(N)
    for k in 1:N
        Rd[k], Td[k], Rr[k], Tr[k], Tb[k] =
            layer_reflectance_transmittance_sw(var, dΛ[k], ω, 0.0, μ0)
    end
    fu = zeros(N + 1); fd = zeros(N + 1); fdir = zeros(N + 1)
    adding_solver!(fu, fd, fdir, Rd, Td, Rr, Tr, Tb,
                   Float64(albedo_diffuse), Float64(albedo_direct),
                   S0 * K, μ0; F_dif_top = Float64(sw_diffuse))

    absorbed = zeros(N); a_sun = zeros(N); a_sha = zeros(N)
    l_sun = zeros(N); l_sha = zeros(N)
    net(i) = fd[i] + fdir[i] - fu[i]
    for k in 1:N
        absorbed[k] = net(k + 1) - net(k)
        dΛ[k] > 0 || continue
        ls = daylight ? (exp(-K * Λ[k+1]) - exp(-K * Λ[k])) / K : 0.0
        ls = clamp(ls, 0.0, dΛ[k])
        l_sun[k] = ls; l_sha[k] = dΛ[k] - ls
        beam_abs = (1 - ω) * (fdir[k+1] - fdir[k])
        dif_abs = absorbed[k] - beam_abs
        a_sun[k] = beam_abs + dif_abs * ls / dΛ[k]
        a_sha[k] = dif_abs * (1 - ls / dΛ[k])
    end
    inc = S0 + Float64(sw_diffuse)
    return CanopyShortwaveSolution(zf, Λ, fdir, fd, fu, absorbed, a_sun, a_sha,
                                   l_sun, l_sha, inc, fu[N+1], sum(absorbed), net(1),
                                   K, μ̄, ω, β, β0)
end
canopy_shortwave(p::AbstractLeafAreaDensity, z_faces::AbstractVector{<:Real}; kwargs...) =
    canopy_shortwave(z_faces, layer_leaf_area_density(p, z_faces); kwargs...)

"""
    CanopyLongwaveSolution

Result of [`canopy_longwave`](@ref); arrays ordered bottom-up, fluxes
[W m⁻²] per unit ground area.

  * `flux_up`, `flux_dn` — at the faces
  * `net_absorbed` — net longwave gain (absorbed minus emitted) of the
    leaves of each layer
  * `ground_net` — net longwave into the ground
  * `outgoing` — upward longwave leaving the canopy top
  * `emissivity_layer` — emissivity ``1 - R - T`` of each layer
"""
struct CanopyLongwaveSolution
    z_faces::Vector{Float64}
    Λ_faces::Vector{Float64}
    flux_up::Vector{Float64}
    flux_dn::Vector{Float64}
    net_absorbed::Vector{Float64}
    emissivity_layer::Vector{Float64}
    ground_net::Float64
    outgoing::Float64
end

"""
    canopy_longwave(z_faces, lad; lw_down, T_leaf, T_ground, ε_leaf = 0.98,
                    ε_ground = 0.95, leaves = SphericalLeaves()) -> CanopyLongwaveSolution
    canopy_longwave(profile::AbstractLeafAreaDensity, z_faces; kwargs...)

Thermal-infrared exchange between sky, leaves and ground. `T_leaf` [K] is
a scalar or one value per layer; `lw_down` is the sky longwave at the top
face [W m⁻²].

Each layer is a two-stream slab with leaf reflectance ``1 - ε_{leaf}`` and
zero leaf transmittance, whose diffuse reflectance ``R`` and transmittance
``T`` come from the same `TwoStream` layer formulae as the shortwave
(diffuse extinction ``1/\\bar μ`` per unit leaf area). By Kirchhoff's law
the layer emits ``(1 - R - T)\\,σT_{leaf}^4`` both up and down. The
coupled layer–ground system, including every multiple reflection, is then
solved exactly as a linear system in the face fluxes, so the result
conserves energy to round-off and gives zero net exchange when sky, leaves
and ground share one temperature and the sky is black.
"""
function canopy_longwave(z_faces::AbstractVector{<:Real}, lad::AbstractVector{<:Real};
                         lw_down::Real, T_leaf, T_ground::Real,
                         ε_leaf::Real = 0.98, ε_ground::Real = 0.95,
                         leaves::AbstractLeafAngleDistribution = SphericalLeaves())
    N = length(lad)
    length(z_faces) == N + 1 ||
        throw(DimensionMismatch("canopy_longwave: need length(z_faces) == length(lad) + 1"))
    (0 < ε_leaf <= 1 && 0 <= ε_ground <= 1) ||
        throw(ArgumentError("canopy_longwave: emissivities must lie in (0, 1]"))
    Tl = T_leaf isa Real ? fill(Float64(T_leaf), N) : Float64.(T_leaf)
    length(Tl) == N || throw(DimensionMismatch("canopy_longwave: T_leaf must be scalar or length N"))
    zf = Float64.(z_faces)
    dΛ = [Float64(lad[k]) * (zf[k+1] - zf[k]) for k in 1:N]
    Λ = zeros(N + 1)
    for k in N:-1:1
        Λ[k] = Λ[k+1] + dΛ[k]
    end
    μ̄ = average_inverse_optical_depth(leaves)
    ω, β, _ = canopy_scattering_parameters(leaves, LeafOptics(1 - ε_leaf, 0.0), 1.0)
    var = SellersCanopyVariant(μ̄, β, β)
    R = zeros(N); Tt = zeros(N); E = zeros(N); εl = zeros(N)
    for k in 1:N
        Rd, Td, _, _, _ = layer_reflectance_transmittance_sw(var, dΛ[k], ω, 0.0, 1.0)
        R[k] = Rd; Tt[k] = Td
        εl[k] = max(1 - Rd - Td, 0.0)
        E[k] = εl[k] * STEFAN_BOLTZMANN * Tl[k]^4
    end
    # Unknowns x = [up_1..up_{N+1}, dn_1..dn_{N+1}]
    n = N + 1
    A = zeros(2n, 2n); b = zeros(2n)
    up(i) = i; dn(i) = n + i
    r = 1
    A[r, up(1)] = 1; A[r, dn(1)] = -(1 - ε_ground)
    b[r] = ε_ground * STEFAN_BOLTZMANN * Float64(T_ground)^4; r += 1
    A[r, dn(n)] = 1; b[r] = Float64(lw_down); r += 1
    for k in 1:N
        # dn_k = T dn_{k+1} + R up_k + E
        A[r, dn(k)] = 1; A[r, dn(k+1)] = -Tt[k]; A[r, up(k)] = -R[k]; b[r] = E[k]; r += 1
        # up_{k+1} = T up_k + R dn_{k+1} + E
        A[r, up(k+1)] = 1; A[r, up(k)] = -Tt[k]; A[r, dn(k+1)] = -R[k]; b[r] = E[k]; r += 1
    end
    x = A \ b
    fu = x[1:n]; fd = x[n+1:2n]
    netk = [(fd[k+1] - fu[k+1]) - (fd[k] - fu[k]) for k in 1:N]
    return CanopyLongwaveSolution(zf, Λ, fu, fd, netk, εl, fd[1] - fu[1], fu[n])
end
canopy_longwave(p::AbstractLeafAreaDensity, z_faces::AbstractVector{<:Real}; kwargs...) =
    canopy_longwave(z_faces, layer_leaf_area_density(p, z_faces); kwargs...)

"""
    canopy_shortwave_field!(q, mesh::RegularGrid, lad; sun, sw_direct, sw_diffuse,
                            optics, albedo_direct, albedo_diffuse = albedo_direct,
                            leaves = SphericalLeaves()) -> ground_absorbed::Matrix

Shortwave absorbed by the leaves of every cell of a resolved canopy,
written to `q` `(Nx, Ny, Nz)` as a volumetric source [W m⁻³]; returns the
`(Nx, Ny)` shortwave absorbed by the ground under each column [W m⁻²].
`lad` is the cell-centred leaf-area density of
[`canopy_lad_field`](@ref). `sw_direct`, `sw_diffuse` and the albedos are
scalars or `(Nx, Ny)` matrices (e.g. the cloud-shadowed direct beam of
`SolarGeometry.cloud_shadow_map` times the clear-sky flux).

Each grid column is solved independently with [`canopy_shortwave`](@ref):
the independent-column approximation, which ignores the horizontal
displacement of a slanted beam through the crown. Columns without leaves
put all the incident radiation into the ground budget.
"""
function canopy_shortwave_field!(q::AbstractArray{<:Real,3}, mesh::RegularGrid,
                                 lad::AbstractArray{<:Real,3};
                                 sun, sw_direct, sw_diffuse, optics::LeafOptics,
                                 albedo_direct, albedo_diffuse = albedo_direct,
                                 leaves::AbstractLeafAngleDistribution = SphericalLeaves())
    Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
    size(lad) == (Nx, Ny, Nz) || throw(DimensionMismatch("canopy_shortwave_field!: lad must be (Nx, Ny, Nz)"))
    size(q) == (Nx, Ny, Nz) || throw(DimensionMismatch("canopy_shortwave_field!: q must be (Nx, Ny, Nz)"))
    pick(x, i, j) = x isa Real ? Float64(x) : Float64(x[i, j])
    zf = [(k - 1) * mesh.dz for k in 1:Nz+1]
    ground = zeros(Nx, Ny)
    fill!(q, 0)
    for j in 1:Ny, i in 1:Nx
        col = @view lad[i, j, :]
        Sd = pick(sw_direct, i, j); Df = pick(sw_diffuse, i, j)
        ad = pick(albedo_direct, i, j); af = pick(albedo_diffuse, i, j)
        if all(iszero, col)
            μ = _cos_zenith(sun)
            ground[i, j] = (μ > 1e-6 ? (1 - ad) * Sd : 0.0) + (1 - af) * Df
            continue
        end
        s = canopy_shortwave(zf, col; sun = sun, sw_direct = Sd, sw_diffuse = Df,
                             leaves = leaves, optics = optics,
                             albedo_direct = ad, albedo_diffuse = af)
        for k in 1:Nz
            q[i, j, k] = s.absorbed[k] / mesh.dz
        end
        ground[i, j] = s.ground_absorbed
    end
    return ground
end

end # module
