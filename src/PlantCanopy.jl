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

  5. **Leaf energy balance** — leaf temperature from the balance of
     absorbed radiation, thermal emission, sensible heat and
     transpiration ([`leaf_energy_balance`](@ref)), with the leaf
     boundary layer of Campbell & Norman (1998)
     ([`CampbellNormanBoundaryLayer`](@ref)) or CLM5
     ([`CLMLeafBoundaryLayer`](@ref)) and transpiration from a fixed
     stomatal conductance or the coupled photosynthesis–stomata solution
     of `Photosynthesis` ([`CoupledLeafPhysiology`](@ref)). Diagnostic
     (steady) or prognostic (backward Euler with a leaf heat capacity).
     Over a resolved canopy [`canopy_leaf_energy_balance`](@ref) solves
     the sunlit and shaded leaves of every layer together with the
     longwave exchange of item 4, and returns the heat and vapour sources
     for the air. [`leaf_penman_monteith`](@ref) is the linearised
     (Penman–Monteith) limit.

  6. **Within-canopy profiles** — measured leaf-area profiles
     ([`TabulatedLAD`](@ref)), the wall-limited canopy mixing length
     ([`canopy_mixing_length`](@ref)), the displacement height as the
     centroid of momentum absorption ([`canopy_displacement_height`](@ref)),
     and the steady K-theory scalar column
     ([`solve_canopy_scalar_column`](@ref)) with the Harman & Finnigan (2008)
     diffusivity ([`harman_finnigan_scalar_diffusivity`](@ref)), which turns
     the leaf heat sources of item 5 into an air-temperature profile. These
     are what the #25 validation against wind-tunnel, orchard and forest
     profiles uses (`validation/era5_canopy_profiles.jl`).

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
  * Campbell, G. S. & Norman, J. M. (1998). *An Introduction to
    Environmental Biophysics*, 2nd ed. Springer — ch. 7 (leaf
    boundary-layer conductances) and ch. 14 (leaf energy balance and its
    linearised solution).
  * Monteith, J. L. & Unsworth, M. H. (2013). *Principles of Environmental
    Physics*, 4th ed. Academic Press — ch. 13 (Penman–Monteith).
  * Leuzinger, S. & Körner, C. (2007). Tree species diversity affects
    canopy leaf temperatures in a mature temperate forest. *Agric. For.
    Meteorol.*, 146, 29-37 (observed leaf-to-air temperature differences).
  * Incropera, F. P., DeWitt, D. P., Bergman, T. L. & Lavine, A. S. (2007).
    *Fundamentals of Heat and Mass Transfer*, 6th ed. Wiley — the laminar
    flat-plate correlations behind the Campbell & Norman coefficients.
  * Harman, I. N. & Finnigan, J. J. (2008). Scalar concentration profiles
    in the canopy and roughness sublayer. *Boundary-Layer Meteorol.*, 129,
    323-351.
  * Thom, A. S. (1971). Momentum absorption by vegetation. *Q. J. R.
    Meteorol. Soc.*, 97, 414-428; Jackson, P. S. (1981). On the
    displacement height in the logarithmic velocity profile. *J. Fluid
    Mech.*, 111, 15-25.
"""
module PlantCanopy

using ..Grids: RegularGrid
using ..BoundaryConditions: fill_halos!
using ..Backends: launch!, backend     # KA-k (#307): the canopy drag is a launch
using ..RadiationCore: _gauss_legendre
using ..P3Microphysics: lgamma_lanczos
using ..RadiationCore: STEFAN_BOLTZMANN
using ..TwoStream: TwoStreamVariant, layer_reflectance_transmittance_sw, adding_solver!
import ..TwoStream: two_stream_coefficients
using ..SolarGeometry: SolarPosition
using ..MoistThermodynamics: ThermodynamicConstants, DEFAULT_CONSTANTS,
                             saturation_vapor_pressure, latent_heat_vapor, R_v
using ..Photosynthesis: AbstractPhotosynthesisParameters, AbstractStomatalModel,
                        leaf_gas_exchange, leaf_boundary_layer_conductance,
                        leaf_transpiration, par_to_photon_flux
using ..SurfaceExchange: _illinois, von_karman

export AbstractLeafAreaDensity, UniformLAD, LalicMihailovicLAD, BetaLAD, TabulatedLAD,
       leaf_area_density, leaf_area_index, canopy_height, cumulative_leaf_area,
       layer_leaf_area_density,
       canopy_drag_factor, canopy_drag_force, implicit_canopy_drag,
       ResolvedCanopyDrag, canopy_lad_field, apply_canopy_drag!, vegetation_canopy_drag, apply_tree_shade!,
       CanopyWindSolution, solve_canopy_wind_column,
       harman_finnigan_mixing_length, canopy_attenuation_coefficient,
       canopy_exponential_wind,
       AbstractLeafAngleDistribution, RossGoudriaanLeaves, SphericalLeaves,
       HorizontalLeaves, LeafOptics, leaf_projection_function,
       beam_extinction_coefficient, average_inverse_optical_depth,
       canopy_scattering_parameters, canopy_beam_transmittance,
       sunlit_leaf_fraction, canopy_diffuse_transmittance,
       CanopyShortwaveSolution, canopy_shortwave,
       CanopyLongwaveSolution, canopy_longwave, canopy_shortwave_field!,
       AbstractLeafBoundaryLayer, CampbellNormanBoundaryLayer, CLMLeafBoundaryLayer,
       leaf_boundary_layer_conductances,
       AbstractLeafStomata, FixedStomatalConductance, CoupledLeafPhysiology,
       LeafEnergyBalance, leaf_energy_balance, leaf_penman_monteith,
       CanopyLeafEnergySolution, canopy_leaf_energy_balance,
       CanopyScalarSolution, solve_canopy_scalar_column, canopy_displacement_height,
       canopy_mixing_length, harman_finnigan_scalar_diffusivity

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
    TabulatedLAD(z, a; LAI = nothing)

A measured leaf-area-density profile given at heights `z` [m] (strictly
increasing, `z[1] ≥ 0`) with values `a` [m² m⁻³], linearly interpolated
between the nodes and zero outside ``[z_1, z_n]``. The canopy height is
`z[end]`.

The leaf-area index is the exact integral of the interpolant (the
trapezoidal sum of the table). If `LAI` is given, the values are rescaled
by one common factor so that the profile integrates to it — for a
published profile shape whose absolute scale is quoted separately (e.g. a
normalised ``a h`` against ``z/h`` plus a stand LAI).
"""
struct TabulatedLAD{T<:AbstractFloat} <: AbstractLeafAreaDensity
    z::Vector{T}
    a::Vector{T}
    LAI::T
    h::T
end
function TabulatedLAD(z::AbstractVector{<:Real}, a::AbstractVector{<:Real};
                      LAI::Union{Nothing,Real} = nothing)
    n = length(z)
    n >= 2 && length(a) == n ||
        throw(ArgumentError("TabulatedLAD: need ≥ 2 nodes and length(a) == length(z)"))
    zF = Float64.(z); aF = Float64.(a)
    zF[1] >= 0 && all(>(0), diff(zF)) ||
        throw(ArgumentError("TabulatedLAD: z must start at ≥ 0 and increase strictly"))
    all(>=(0), aF) || throw(ArgumentError("TabulatedLAD: a must be ≥ 0"))
    L = sum((aF[i] + aF[i+1]) / 2 * (zF[i+1] - zF[i]) for i in 1:n-1)
    if LAI !== nothing
        LAI >= 0 || throw(ArgumentError("TabulatedLAD: LAI must be ≥ 0"))
        L > 0 || LAI == 0 ||
            throw(ArgumentError("TabulatedLAD: cannot rescale an all-zero profile to LAI > 0"))
        L > 0 && (aF .*= LAI / L)
        L = Float64(LAI)
    end
    return TabulatedLAD{Float64}(zF, aF, L, zF[end])
end
function leaf_area_density(p::TabulatedLAD, z::Real)
    (z < p.z[1] || z > p.z[end]) && return zero(p.LAI)
    i = clamp(searchsortedlast(p.z, z), 1, length(p.z) - 1)
    t = (z - p.z[i]) / (p.z[i+1] - p.z[i])
    return (1 - t) * p.a[i] + t * p.a[i+1]
end
_lad_breakpoints(p::TabulatedLAD) = Tuple(p.z)

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
horizontally periodic). The struct also lists the faces with leaves
(`act`: `(c, i, j, k)`, component `c` = 1, 2, 3 for `u`, `v`, `w`) and
carries one drag factor and one velocity change per listed face, so
[`apply_canopy_drag!`](@ref) works on the canopy only and does not
allocate.

`C_d` is the leaf drag coefficient, ≈ 0.15-0.3 for forests (Shaw &
Schumann 1992 use 0.15).
"""
struct ResolvedCanopyDrag{T<:AbstractFloat}
    C_d::T
    lad::Array{T,3}
    a_u::Array{T,3}
    a_v::Array{T,3}
    a_w::Array{T,3}
    act::Vector{NTuple{4,Int32}}
    fac::Vector{T}
    dq::Vector{T}
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
    # the faces with leaves (KA-k, #307): u, then v, then w, each in the
    # loop order k, j, i; the drag is applied over this list only
    act = NTuple{4,Int32}[]
    for (c, ac) in enumerate((a_u, a_v, a_w)), k in 1:Nz, j in 1:Ny, i in 1:Nx
        ac[i, j, k] > 0 && push!(act, (Int32(c), Int32(i), Int32(j), Int32(k)))
    end
    return ResolvedCanopyDrag{T}(T(C_d), a, a_u, a_v, a_w, act, ones(T, length(act)),
                                 zeros(T, length(act)))
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
    vegetation_canopy_drag(mesh, veg_mask, LAI; cd = 0.2, k_height = nothing)
        -> ResolvedCanopyDrag

The resolved canopy drag of a vegetation map: each column with
`veg_mask[i, j]` and `LAI[i, j] > 0` gets a uniform crown
`UniformLAD(LAI[i, j], h)` of depth `h = k_height · Δz` above the ground
(`k_height === nothing`: `Nz ÷ 4` layers, at least one; clamped to
`1 … Nz`), with leaf drag coefficient `cd`. Build it once and apply it
with [`apply_canopy_drag!`](@ref) after the predictor (moved from the
urban case's `veg_drag` coupling, ONE-e #454).
"""
function vegetation_canopy_drag(mesh, veg_mask::AbstractMatrix{Bool},
                                LAI_arr::AbstractMatrix{<:Real};
                                cd::Real = 0.2,
                                k_height::Union{Nothing, Integer} = nothing)
    Nz = mesh.Nz
    k_top = clamp(k_height === nothing ? max(1, Nz ÷ 4) : Int(k_height), 1, Nz)
    h = k_top * mesh.dz
    profiles = Union{Nothing, UniformLAD{Float64}}[
        (veg_mask[i, j] && LAI_arr[i, j] > 0) ? UniformLAD(LAI_arr[i, j], h) : nothing
        for i in axes(veg_mask, 1), j in axes(veg_mask, 2)]
    return ResolvedCanopyDrag(mesh, canopy_lad_field(mesh, profiles); C_d = cd)
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

The factors and the update are two launches over the faces with leaves
(KA-k, #307), each face written by one work item: the same bits at every
thread count and as the former full-grid loops; the force is folded per
component in the former loop order.

Returns the volume-integrated kinematic force the step exerted,
``\\sum (U' - U)/Δt\\,ΔV`` [m⁴ s⁻²] per component (negative for a flow
in the positive direction); divide by the domain area for the canopy
momentum sink per unit ground area (a stress, m² s⁻²).
"""
function apply_canopy_drag!(sim, drag::ResolvedCanopyDrag{T}, dt::Real) where {T}
    mesh = sim.mesh
    Hx, Hy, Hz = mesh.Hx, mesh.Hy, mesh.Hz
    Δt = T(dt); C = drag.C_d
    fill_halos!(sim.u, sim.bcs); fill_halos!(sim.v, sim.bcs); fill_halos!(sim.w, sim.bcs)
    u = sim.u.data; v = sim.v.data; w = sim.w.data
    act = drag.act
    r = (1:length(act),)
    b = backend(u)
    # 1. All factors from the pre-drag velocity.
    launch!(b, _canopy_factor_body!, r, drag.fac, act, drag.a_u, drag.a_v, drag.a_w,
            u, v, w, C, Δt, Hx, Hy, Hz)
    # 2. Apply, and record the change of every face.
    launch!(b, _canopy_apply_body!, r, drag.dq, drag.fac, act, u, v, w, Hx, Hy, Hz)
    # 3. The momentum removed, folded per component in the list order
    #    (k, j, i: the order of the former loop).
    Fx = zero(T); Fy = zero(T); Fz = zero(T)
    @inbounds for n in eachindex(act)
        c = act[n][1]; d = drag.dq[n]
        c == 1 ? (Fx += d) : c == 2 ? (Fy += d) : (Fz += d)
    end
    dV = mesh.dx * mesh.dy * mesh.dz
    return (Fx = Fx * dV / Δt, Fy = Fy * dV / Δt, Fz = Fz * dV / Δt)
end

# The drag factor of listed face n (pointwise; reads u, v, w only): the
# full speed there, the other two components averaged from their four
# neighbouring faces.
@inline function _canopy_factor_body!(n, fac, act, a_u, a_v, a_w, u, v, w,
                                      C, Δt::T, Hx, Hy, Hz) where {T}
    q = T(1) / 4
    @inbounds begin
        c, i32, j32, k32 = act[n]
        i, j, k = Int(i32), Int(j32), Int(k32)
        I = Hx + i; J = Hy + j; K = Hz + k
        if c == 1
            uu = u[I, J, K]
            vv = q * (v[I-1, J, K] + v[I, J, K] + v[I-1, J+1, K] + v[I, J+1, K])
            ww = q * (w[I-1, J, K] + w[I, J, K] + w[I-1, J, K+1] + w[I, J, K+1])
            fac[n] = canopy_drag_factor(C, a_u[i, j, k], sqrt(uu^2 + vv^2 + ww^2), Δt)
        elseif c == 2
            vv = v[I, J, K]
            uu = q * (u[I, J-1, K] + u[I+1, J-1, K] + u[I, J, K] + u[I+1, J, K])
            ww = q * (w[I, J-1, K] + w[I, J, K] + w[I, J-1, K+1] + w[I, J, K+1])
            fac[n] = canopy_drag_factor(C, a_v[i, j, k], sqrt(uu^2 + vv^2 + ww^2), Δt)
        else
            ww = w[I, J, K]
            uu = q * (u[I, J, K-1] + u[I+1, J, K-1] + u[I, J, K] + u[I+1, J, K])
            vv = q * (v[I, J, K-1] + v[I, J+1, K-1] + v[I, J, K] + v[I, J+1, K])
            fac[n] = canopy_drag_factor(C, a_w[i, j, k], sqrt(uu^2 + vv^2 + ww^2), Δt)
        end
    end
    return nothing
end

# Listed face n: q ← f q, dq[n] = the change (writes only its own face).
@inline function _canopy_apply_body!(n, dq, fac, act, u, v, w, Hx, Hy, Hz)
    @inbounds begin
        c, i, j, k = act[n]
        I = Hx + Int(i); J = Hy + Int(j); K = Hz + Int(k)
        q = c == 1 ? u : c == 2 ? v : w
        q0 = q[I, J, K]
        q[I, J, K] = q0 * fac[n]
        dq[n] = q[I, J, K] - q0
    end
    return nothing
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
`residual` (max ``|ΔU|/U_{top}``). `z_faces` are the layer faces and `K`
the eddy viscosity ``l^2|∂U/∂z| + ν`` [m² s⁻¹] at those faces at
convergence (`K[1]` is the ground exchange coefficient, zero for a
free-slip bottom): the diffusivity that
[`solve_canopy_scalar_column`](@ref) needs, divided by a turbulent Schmidt
number.
"""
struct CanopyWindSolution
    z::Vector{Float64}
    u::Vector{Float64}
    u_star::Float64
    drag_integral::Float64
    tau_ground::Float64
    iterations::Int
    residual::Float64
    z_faces::Vector{Float64}
    K::Vector{Float64}
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
    return CanopyWindSolution(zc, u, sqrt(abs(τ_top)), drag, τ_g, it, res, zf, copy(K))
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
    apply_tree_shade!(sw_arr, veg_mask, LAI_arr; k_leaf = 0.5)

Attenuate the shortwave flux reaching the ground beneath tree canopies
using Beer's law with an extinction coefficient `k_leaf` (typical
`0.4..0.6` for a randomly-distributed broadleaf canopy):

    SW_ground ← SW_ground · exp(-k_leaf · LAI)

Modifies `sw_arr` in place at every cell where `veg_mask[i, j]` is
`true`. Non-vegetated cells are untouched. Returns `sw_arr`. A single
big-leaf extinction with a constant `k_leaf`, the spherical-leaf
``G/μ = 0.5`` at the zenith; [`canopy_beam_transmittance`](@ref) is the
angle-resolved form. Moved from the urban case's `sw_modifiers` coupling
(ONE-e, #454).
"""
function apply_tree_shade!(sw_arr::AbstractMatrix{<:Real},
                           veg_mask::AbstractMatrix{Bool},
                           LAI_arr::AbstractMatrix{<:Real};
                           k_leaf::Real = 0.5)
    Nx, Ny = size(sw_arr)
    k = Float64(k_leaf)
    @inbounds for j in 1:Ny, i in 1:Nx
        if veg_mask[i, j]
            lai = Float64(LAI_arr[i, j])
            sw_arr[i, j] *= exp(-k * lai)
        end
    end
    return sw_arr
end

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

# =========================================================================
# 5. Leaf energy balance (#24)
# =========================================================================
#
# Per unit (one-sided) leaf area, a leaf at temperature T_ℓ obeys
#
#   C dT_ℓ/dt = R_abs − ε_e σ T_ℓ⁴ − c_p g_H (T_ℓ − T_a) − λ E(T_ℓ),
#
# with R_abs the absorbed short- and longwave radiation, ε_e the emission
# coefficient of the leaf (2ε for an isolated flat leaf, which emits from
# both sides), g_H the boundary-layer conductance to heat of both sides,
# and E the transpiration through stomata and boundary layer in series.
# Molar conductances (mol m⁻² s⁻¹) are used throughout, as in
# `Photosynthesis`; c_p is then the molar heat capacity of air.
# Saturation vapour pressure and latent heat are `MoistThermodynamics`',
# transpiration and the coupled photosynthesis–stomata solution are
# `Photosynthesis`', the radiation is section 4 above.

"""
    AbstractLeafBoundaryLayer

Leaf boundary-layer model. Implementers extend
[`leaf_boundary_layer_conductances`](@ref), which returns the conductance
to heat (both leaf sides) and to water vapour (the stomata-bearing
sides), in mol m⁻² s⁻¹ per unit one-sided leaf area. Concrete models:
[`CampbellNormanBoundaryLayer`](@ref), [`CLMLeafBoundaryLayer`](@ref).
"""
abstract type AbstractLeafBoundaryLayer end

"""
    CampbellNormanBoundaryLayer(; leaf_width = 0.05, turbulence_factor = 1.4,
                                  free_convection = true, stomatal_sides = 1)

Leaf boundary layer of Campbell & Norman (1998, ch. 7) for a flat leaf of
width `leaf_width` [m], characteristic dimension ``d = 0.72 w``. Per leaf
side, in mol m⁻² s⁻¹:

  * forced convection (laminar flat plate), multiplied by `turbulence_factor` for outdoor turbulence (the factor
    1.4 they recommend):
    ``g_{Ha} = 0.135\\sqrt{u/d}``, ``g_{va} = 0.147\\sqrt{u/d}``;
  * free convection (laminar, heated plate):
    ``g_{Ha} = 0.05\\,(|T_\\ell - T_a|/d)^{1/4}``,
    ``g_{va} = 0.055\\,(|T_\\ell - T_a|/d)^{1/4}``;

the larger of the two is used (`free_convection = false` keeps only the
forced value). These are the textbook flat-plate correlations
``\\overline{Nu} = 0.664\\,Re^{1/2}Pr^{1/3}`` and ``\\overline{Nu} = 0.54\\,Ra^{1/4}``
(Incropera et al. 2007, chs. 7 and 9) evaluated for air at 20 °C;
`test/era5_leaf_energy.jl` re-derives them to 1–3 %. The heat conductance of the leaf is that of both sides,
``2g_{Ha}``; the vapour conductance is ``n\\,g_{va}`` with `n =
stomatal_sides` = 1 for hypostomatous leaves (most broadleaf trees) and
2 for amphistomatous ones. The coefficients are C&N's values at 20 °C and
101.3 kPa; their weak temperature and pressure dependence is neglected.
"""
struct CampbellNormanBoundaryLayer <: AbstractLeafBoundaryLayer
    leaf_width::Float64
    d::Float64
    turbulence_factor::Float64
    free_convection::Bool
    stomatal_sides::Int
    function CampbellNormanBoundaryLayer(; leaf_width::Real = 0.05, turbulence_factor::Real = 1.4,
                                         free_convection::Bool = true, stomatal_sides::Integer = 1)
        leaf_width > 0 || throw(ArgumentError("CampbellNormanBoundaryLayer: leaf_width must be > 0"))
        turbulence_factor > 0 || throw(ArgumentError("CampbellNormanBoundaryLayer: turbulence_factor must be > 0"))
        stomatal_sides in (1, 2) || throw(ArgumentError("CampbellNormanBoundaryLayer: stomatal_sides must be 1 or 2"))
        new(Float64(leaf_width), 0.72 * Float64(leaf_width), Float64(turbulence_factor),
            free_convection, Int(stomatal_sides))
    end
end

"""
    CLMLeafBoundaryLayer(; d_leaf = 0.04, Cv = 0.01)

The leaf boundary layer of CLM5 (eq. 2.5.122), evaluated by
`Photosynthesis.leaf_boundary_layer_conductance`: one conductance
``C_v\\sqrt{U/d_{leaf}}`` (m s⁻¹, converted to molar units) used for both
heat and water vapour per unit one-sided leaf area, as in CLM's canopy
fluxes. No free convection.
"""
struct CLMLeafBoundaryLayer <: AbstractLeafBoundaryLayer
    d_leaf::Float64
    Cv::Float64
    function CLMLeafBoundaryLayer(; d_leaf::Real = 0.04, Cv::Real = 0.01)
        (d_leaf > 0 && Cv > 0) || throw(ArgumentError("CLMLeafBoundaryLayer: d_leaf and Cv must be > 0"))
        new(Float64(d_leaf), Float64(Cv))
    end
end

"""
    leaf_boundary_layer_conductances(bl, U, T_leaf, T_air, p_air; c = DEFAULT_CONSTANTS)
        -> (g_H, g_v)

Leaf boundary-layer conductances [mol m⁻² s⁻¹ per unit one-sided leaf
area] for wind speed `U` [m s⁻¹] past the leaf: `g_H` to sensible heat
(all leaf sides) and `g_v` to water vapour (the sides with stomata, in
series with the stomatal conductance). `T_leaf` enters only through free
convection.
"""
function leaf_boundary_layer_conductances(bl::CampbellNormanBoundaryLayer, U::Real,
                                          T_leaf::Real, T_air::Real, p_air::Real;
                                          c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    r = sqrt(max(Float64(U), 0.0) / bl.d)
    gH = bl.turbulence_factor * 0.135 * r
    gv = bl.turbulence_factor * 0.147 * r
    if bl.free_convection
        x = sqrt(sqrt(abs(Float64(T_leaf) - Float64(T_air)) / bl.d))
        gH = max(gH, 0.05 * x)
        gv = max(gv, 0.055 * x)
    end
    return (2gH, bl.stomatal_sides * gv)
end
function leaf_boundary_layer_conductances(bl::CLMLeafBoundaryLayer, U::Real,
                                          T_leaf::Real, T_air::Real, p_air::Real;
                                          c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    g = leaf_boundary_layer_conductance(U, T_air, p_air; d_leaf = bl.d_leaf, Cv = bl.Cv, c = c)
    return (g, g)
end

"""
    AbstractLeafStomata

What controls transpiration in the leaf energy balance:
[`FixedStomatalConductance`](@ref) or [`CoupledLeafPhysiology`](@ref).
A plain number passed where a stomata model is expected is a fixed
conductance.
"""
abstract type AbstractLeafStomata end

"""
    FixedStomatalConductance(gs)

A prescribed stomatal conductance to water vapour `gs` [mol m⁻² s⁻¹];
`0` is a leaf with closed stomata (no transpiration). Transpiration is
`Photosynthesis.leaf_transpiration`.
"""
struct FixedStomatalConductance <: AbstractLeafStomata
    gs::Float64
    function FixedStomatalConductance(gs::Real)
        gs >= 0 || throw(ArgumentError("FixedStomatalConductance: gs must be ≥ 0"))
        new(Float64(gs))
    end
end

"""
    CoupledLeafPhysiology(photosynthesis, stomata; ca = 400, β = 1,
                          stress_targets = (:vcmax, :rd, :g0))

Stomatal conductance and transpiration from the coupled
photosynthesis–stomata solution `Photosynthesis.leaf_gas_exchange`, with
photosynthetic parameters `photosynthesis` (C3/C4), a stomatal closure
`stomata` (Ball-Berry / Medlyn), ambient CO₂ `ca` [µmol mol⁻¹] and the
soil-water-stress factor `β`. The leaf boundary-layer vapour conductance
of the energy balance is passed to it as `gb`, so the CO₂, water and heat
exchange of the leaf see the same boundary layer, and the leaf
temperature of the energy balance is the one the biochemistry runs at.
"""
struct CoupledLeafPhysiology{P<:AbstractPhotosynthesisParameters,M<:AbstractStomatalModel} <: AbstractLeafStomata
    photosynthesis::P
    stomata::M
    ca::Float64
    β::Float64
    stress_targets::Tuple
end
CoupledLeafPhysiology(p::AbstractPhotosynthesisParameters, m::AbstractStomatalModel;
                      ca::Real = 400.0, β::Real = 1.0, stress_targets = (:vcmax, :rd, :g0)) =
    CoupledLeafPhysiology(p, m, Float64(ca), Float64(β), Tuple(stress_targets))

_as_stomata(s::AbstractLeafStomata) = s
_as_stomata(g::Real) = FixedStomatalConductance(g)

# -> (E [mol m⁻² s⁻¹], gs, An [µmol m⁻² s⁻¹], converged)
function _leaf_vapour(s::FixedStomatalConductance, T, gv, Q, e_air, p_air, c)
    (s.gs == 0 || gv == 0) && return (0.0, s.gs, 0.0, true)
    return (leaf_transpiration(s.gs, gv, T, e_air, p_air; c = c), s.gs, 0.0, true)
end
function _leaf_vapour(s::CoupledLeafPhysiology, T, gv, Q, e_air, p_air, c)
    ge = leaf_gas_exchange(s.photosynthesis, s.stomata; ca = s.ca, Q = max(Q, 0.0),
                           T_leaf = T, e_air = e_air, gb = max(gv, 1e-6), p_air = p_air,
                           β = s.β, stress_targets = s.stress_targets, c = c)
    return (ge.E, ge.gs, ge.An, ge.converged)
end

"""
    LeafEnergyBalance

Result of [`leaf_energy_balance`](@ref), per unit one-sided leaf area:

| field | unit | meaning |
| :-- | :-- | :-- |
| `T_leaf` | K | leaf temperature |
| `R_abs` | W m⁻² | absorbed short- and longwave radiation (input) |
| `emitted` | W m⁻² | thermal emission ``ε_e σ T_\\ell^4`` |
| `H`, `LE` | W m⁻² | sensible and latent heat to the air (positive away from the leaf) |
| `storage` | W m⁻² | ``C (T_\\ell - T_{prev})/Δt`` (zero when diagnostic) |
| `E` | mol m⁻² s⁻¹ | transpiration |
| `gs`, `g_H`, `g_v` | mol m⁻² s⁻¹ | stomatal, boundary-layer heat and vapour conductances |
| `An` | µmol m⁻² s⁻¹ | net assimilation (0 for a fixed conductance) |
| `iterations`, `converged` | – | root-finder diagnostics |

``R_{abs} = emitted + H + LE + storage`` holds at the solution.
"""
struct LeafEnergyBalance
    T_leaf::Float64
    R_abs::Float64
    emitted::Float64
    H::Float64
    LE::Float64
    storage::Float64
    E::Float64
    gs::Float64
    g_H::Float64
    g_v::Float64
    An::Float64
    iterations::Int
    converged::Bool
end

function _solve_leaf(st::AbstractLeafStomata, bl::AbstractLeafBoundaryLayer, R_abs, ε_e,
                     T_air, e_air, U, p_air, Q, T_prev, C, dt, c, maxiter)
    cpm = c.cp_dry * c.molmass_dry                   # J mol⁻¹ K⁻¹
    σ = STEFAN_BOLTZMANN
    prognostic = C > 0 && isfinite(dt)
    prognostic && T_prev === nothing &&
        throw(ArgumentError("leaf_energy_balance: a prognostic step (heat_capacity > 0, finite dt) needs T_prev"))
    function parts(T)
        gH, gv = leaf_boundary_layer_conductances(bl, U, T, T_air, p_air; c = c)
        E, gs, An, ok = _leaf_vapour(st, T, gv, Q, e_air, p_air, c)
        H = cpm * gH * (T - T_air)
        LE = latent_heat_vapor(T, c) * c.molmass_water * E
        em = ε_e * σ * T^4
        S = prognostic ? C * (T - T_prev) / dt : 0.0
        return (R_abs - em - H - LE - S, em, H, LE, S, E, gs, gH, gv, An, ok)
    end
    R(T) = -parts(T)[1]                              # increasing in T
    lo = T_air - 40.0; hi = T_air + 40.0
    if prognostic
        lo = min(lo, T_prev - 40.0); hi = max(hi, T_prev + 40.0)
    end
    lo = max(lo, 100.0)
    flo = R(lo); fhi = R(hi)
    n = 0
    while flo > 0 && lo > 100.0 && n < 20
        lo = max(lo - 40.0, 100.0); flo = R(lo); n += 1
    end
    n = 0
    while fhi < 0 && n < 20
        hi += 40.0; fhi = R(hi); n += 1
    end
    (flo <= 0 <= fhi) || throw(ErrorException("leaf_energy_balance: could not bracket the leaf temperature"))
    T, it, conv = flo == 0 ? (lo, 0, true) : fhi == 0 ? (hi, 0, true) :
                  _illinois(R, lo, hi, flo, fhi, maxiter)
    _, em, H, LE, S, E, gs, gH, gv, An, ok = parts(T)
    return LeafEnergyBalance(T, R_abs, em, H, LE, S, E, gs, gH, gv, An, it, conv && ok)
end

"""
    leaf_energy_balance(stomata, boundary_layer = CampbellNormanBoundaryLayer();
                        R_abs, T_air, e_air, U, p_air = 101325, Q = 0,
                        emissivity = 0.98, emission = 2emissivity,
                        T_prev = nothing, heat_capacity = 0, dt = Inf,
                        c = DEFAULT_CONSTANTS, maxiter = 200) -> LeafEnergyBalance

Temperature of a leaf from its energy balance, per unit one-sided leaf
area (Campbell & Norman 1998, ch. 14; Monteith & Unsworth 2013):

```math
C\\,\\frac{T_\\ell - T_{prev}}{Δt} = R_{abs} - ε_e σ T_\\ell^4
  - c_p g_H (T_\\ell - T_a) - λ(T_\\ell)\\,E(T_\\ell).
```

  * `R_abs` — absorbed short- and longwave radiation [W m⁻²]
  * `T_air` [K], `e_air` [Pa], `U` [m s⁻¹], `p_air` [Pa] — the air around the leaf
  * `Q` — absorbed PAR photon flux [µmol m⁻² s⁻¹], used by a
    [`CoupledLeafPhysiology`](@ref)
  * `stomata` — a [`FixedStomatalConductance`](@ref), a number (the same),
    or a [`CoupledLeafPhysiology`](@ref)
  * `boundary_layer` — an [`AbstractLeafBoundaryLayer`](@ref) giving
    ``g_H`` and the vapour conductance ``g_v``
  * `emission` — ``ε_e``, thermal emission per unit leaf area in units of
    ``σT^4``: ``2ε`` for an isolated flat leaf (both sides); inside a
    canopy [`canopy_leaf_energy_balance`](@ref) supplies the two-stream
    value
  * `heat_capacity` ``C`` [J m⁻² K⁻¹], `dt` [s], `T_prev` [K] — with
    ``C > 0`` and finite ``Δt`` this is one backward-Euler step of the
    prognostic leaf temperature (unconditionally stable); the default is
    the diagnostic (steady) balance, which is also the ``Δt → ∞`` limit.
    A leaf holding ≈ 0.3 mm of water has ``C ≈ 1.3\\,\\mathrm{kJ\\,m^{-2}\\,K^{-1}}``.

``c_p`` is the molar heat capacity of dry air, ``λ`` the latent heat of
vaporisation at leaf temperature (`MoistThermodynamics`), ``E`` the
transpiration of the stomata model with ``g_v`` in series. The scalar
equation is solved on a bracket by the Illinois method of
`SurfaceExchange` (the residual is monotone in ``T_\\ell`` except through
the photosynthetic response of ``g_s``, which a bracketing method
tolerates). Negative ``E`` (dew onto a leaf colder than the dew point)
is returned as computed, through the stomatal path only; wet-leaf
interception and dew are the land column's (#21).
"""
function leaf_energy_balance(stomata, boundary_layer::AbstractLeafBoundaryLayer = CampbellNormanBoundaryLayer();
                             R_abs::Real, T_air::Real, e_air::Real, U::Real,
                             p_air::Real = 101325.0, Q::Real = 0.0,
                             emissivity::Real = 0.98, emission::Real = 2 * emissivity,
                             T_prev = nothing, heat_capacity::Real = 0.0, dt::Real = Inf,
                             c::ThermodynamicConstants = DEFAULT_CONSTANTS, maxiter::Integer = 200)
    emission >= 0 || throw(ArgumentError("leaf_energy_balance: emission must be ≥ 0"))
    heat_capacity >= 0 || throw(ArgumentError("leaf_energy_balance: heat_capacity must be ≥ 0"))
    dt > 0 || throw(ArgumentError("leaf_energy_balance: dt must be > 0"))
    Tp = T_prev === nothing ? nothing : Float64(T_prev)
    return _solve_leaf(_as_stomata(stomata), boundary_layer, Float64(R_abs), Float64(emission),
                       Float64(T_air), Float64(e_air), Float64(U), Float64(p_air), Float64(Q),
                       Tp, Float64(heat_capacity), Float64(dt), c, maxiter)
end

"""
    leaf_penman_monteith(; R_abs, T_air, e_air, g_H, g_s, g_v, p_air = 101325,
                           emissivity = 0.98, emission = 2emissivity,
                           c = DEFAULT_CONSTANTS)
        -> (; ΔT, T_leaf, H, LE, E, R_ni, g_r, s, g_w)

The linearised (Penman–Monteith) solution of the diagnostic leaf energy
balance for a fixed stomatal conductance (Campbell & Norman 1998,
§14.3; Monteith & Unsworth 2013, ch. 13). Linearising the emission,
``ε_eσT_\\ell^4 ≈ ε_eσT_a^4 + c_p g_r ΔT`` with the radiative conductance
``g_r = 4ε_eσT_a^3/c_p``, and the saturation vapour pressure,
``e_s(T_\\ell) ≈ e_s(T_a) + sΔT``, gives

```math
ΔT = T_\\ell - T_a = \\frac{R_{ni} - λ g_w D/p}{c_p (g_H + g_r) + λ g_w s/p},
\\qquad R_{ni} = R_{abs} - ε_eσT_a^4,
```

with ``D = e_s(T_a) - e_a``, ``g_w = g_s g_v/(g_s + g_v)`` the total leaf
conductance to vapour, ``λ`` in J mol⁻¹, all evaluated at air
temperature. Then ``H = c_p g_H ΔT`` and ``λE = λ g_w (D + sΔT)/p``.
``s = de_s/dT = L_v e_s/(R_v T^2)`` is the exact derivative of the
`MoistThermodynamics` saturation curve. The nonlinear
[`leaf_energy_balance`](@ref) tends to this as the forcing
(``R_{ni}``, ``D``) goes to zero, with an error quadratic in ``ΔT``.
"""
function leaf_penman_monteith(; R_abs::Real, T_air::Real, e_air::Real, g_H::Real, g_s::Real,
                              g_v::Real, p_air::Real = 101325.0, emissivity::Real = 0.98,
                              emission::Real = 2 * emissivity,
                              c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    Ta = Float64(T_air); p = Float64(p_air)
    cpm = c.cp_dry * c.molmass_dry
    λ = latent_heat_vapor(Ta, c) * c.molmass_water
    es = saturation_vapor_pressure(Ta, c)
    s = latent_heat_vapor(Ta, c) * es / (R_v(c) * Ta^2)
    D = es - Float64(e_air)
    gw = (g_s == 0 || g_v == 0) ? 0.0 : isinf(g_v) ? Float64(g_s) : g_s * g_v / (g_s + g_v)
    g_r = 4 * emission * STEFAN_BOLTZMANN * Ta^3 / cpm
    R_ni = Float64(R_abs) - emission * STEFAN_BOLTZMANN * Ta^4
    ΔT = (R_ni - λ * gw * D / p) / (cpm * (g_H + g_r) + λ * gw * s / p)
    E = gw * (D + s * ΔT) / p
    return (ΔT = ΔT, T_leaf = Ta + ΔT, H = cpm * g_H * ΔT, LE = λ * E, E = E,
            R_ni = R_ni, g_r = g_r, s = s, g_w = gw)
end

"""
    CanopyLeafEnergySolution

Result of [`canopy_leaf_energy_balance`](@ref). Layer arrays are ordered
bottom-up like `z_faces`. Per-class quantities are per unit leaf area of
that class; layer quantities are per unit **ground** area of the layer.

  * `T_sunlit`, `T_shaded` [K], `gs_sunlit`, `gs_shaded` [mol m⁻² s⁻¹] —
    for a layer without sunlit (or any) leaves the sunlit value repeats
    the shaded one (and a layer without leaves reports the air
    temperature), so the fields can seed the next prognostic step
  * `lai_sunlit`, `lai_shaded` — leaf area of each class in each layer
  * `sw_absorbed`, `lw_net` — absorbed shortwave (all bands) and net
    longwave of the leaves [W m⁻²]
  * `H`, `LE`, `storage` [W m⁻²], `E` [mol H₂O m⁻² s⁻¹], `An`
    [µmol CO₂ m⁻² s⁻¹] — per layer; `sw_absorbed + lw_net = H + LE + storage`
  * `heat_source` [W m⁻³] and `vapour_source` [kg m⁻³ s⁻¹] — `H` and the
    transpired mass divided by the layer depth: the sources of the air's
    heat and moisture equations
  * `H_total`, `LE_total`, `E_total` — column sums
  * `longwave` — the [`CanopyLongwaveSolution`](@ref) at the final leaf
    temperatures
  * `residual` — max over layers of ``|sw\\_absorbed + lw\\_net - H - LE - storage|``
    evaluated with that final longwave [W m⁻²]; `iterations`, `converged`
"""
struct CanopyLeafEnergySolution
    z_faces::Vector{Float64}
    Λ_faces::Vector{Float64}
    T_sunlit::Vector{Float64}
    T_shaded::Vector{Float64}
    gs_sunlit::Vector{Float64}
    gs_shaded::Vector{Float64}
    lai_sunlit::Vector{Float64}
    lai_shaded::Vector{Float64}
    sw_absorbed::Vector{Float64}
    lw_net::Vector{Float64}
    H::Vector{Float64}
    LE::Vector{Float64}
    storage::Vector{Float64}
    E::Vector{Float64}
    An::Vector{Float64}
    heat_source::Vector{Float64}
    vapour_source::Vector{Float64}
    H_total::Float64
    LE_total::Float64
    E_total::Float64
    longwave::CanopyLongwaveSolution
    residual::Float64
    iterations::Int
    converged::Bool
end

_layer_value(x::Real, k) = Float64(x)
_layer_value(x::AbstractVector, k) = Float64(x[k])
_sw_bands(s::CanopyShortwaveSolution) = (s,)
_sw_bands(::Nothing) = ()
_sw_bands(s) = Tuple(s)

"""
    canopy_leaf_energy_balance(z_faces, lad; shortwave, par = 1, T_air, e_air, U,
                               p_air = 101325, lw_down, T_ground, ε_leaf = 0.98,
                               ε_ground = 0.95, leaves = SphericalLeaves(), stomata,
                               boundary_layer = CampbellNormanBoundaryLayer(),
                               T_prev = nothing, heat_capacity = 0, dt = Inf,
                               c = DEFAULT_CONSTANTS, maxiter = 100, tol = 1e-9)
        -> CanopyLeafEnergySolution
    canopy_leaf_energy_balance(profile::AbstractLeafAreaDensity, z_faces; kwargs...)

Temperatures of the sunlit and shaded leaves of every layer of a resolved
canopy, and the heat and water vapour they give to the air.

  * `shortwave` — a [`CanopyShortwaveSolution`](@ref) from
    [`canopy_shortwave`](@ref) on the same layers, or a collection of them
    (one per band, e.g. `(par, nir)`), or `nothing` at night. The absorbed
    shortwave of each class is the sum over bands of `absorbed_sunlit` /
    `absorbed_shaded` divided by the class leaf area, and the sunlit and
    shaded leaf areas are those of the first band.
  * `par` — index into `shortwave` of the PAR band, whose absorbed flux per
    leaf area is converted to photons by `Photosynthesis.par_to_photon_flux`
    for a [`CoupledLeafPhysiology`](@ref); `nothing` for no PAR
  * `T_air` [K], `e_air` [Pa], `U` [m s⁻¹] — scalars or one value per layer
  * `lw_down`, `T_ground`, `ε_leaf`, `ε_ground`, `leaves` — as in
    [`canopy_longwave`](@ref)
  * `stomata`, `boundary_layer` — as in [`leaf_energy_balance`](@ref)
  * `T_prev` — a previous `CanopyLeafEnergySolution` (or a named tuple
    `(sunlit = …, shaded = …)` of per-layer temperatures) with
    `heat_capacity` ``C`` [J m⁻² K⁻¹] and `dt` for a prognostic step

The leaves exchange longwave radiation with each other, the sky and the
ground, so the layers are coupled. They are solved by an outer iteration:
the longwave field of [`canopy_longwave`](@ref) is computed with each
layer at the leaf-area-weighted ``T^4`` of its two classes; the longwave
*absorbed* per unit leaf area of a layer is then held fixed while every
class solves its own [`leaf_energy_balance`](@ref), with its emission
``ε_e σT^4``, ``ε_e = 2(1 - R_k - T_k)/ΔΛ_k`` the layer's two-stream
emissivity per unit leaf area (``2ε_{leaf}`` for spherical leaves in the
thin-layer limit). That is repeated until no leaf temperature changes by
more than `tol` [K]. The longwave is recomputed at the final temperatures
for the reported fluxes, and `residual` measures how well each layer's
energy balance closes with it.

For a prognostic step each class uses its own previous temperature. That
conserves energy exactly while the class leaf areas are unchanged; when
the sun moves, leaves that change class carry their old class's
temperature, which is not tracked.
"""
function canopy_leaf_energy_balance(z_faces::AbstractVector{<:Real}, lad::AbstractVector{<:Real};
                                    shortwave, par = 1, T_air, e_air, U,
                                    p_air::Real = 101325.0, lw_down::Real, T_ground::Real,
                                    ε_leaf::Real = 0.98, ε_ground::Real = 0.95,
                                    leaves::AbstractLeafAngleDistribution = SphericalLeaves(),
                                    stomata, boundary_layer::AbstractLeafBoundaryLayer = CampbellNormanBoundaryLayer(),
                                    T_prev = nothing, heat_capacity::Real = 0.0, dt::Real = Inf,
                                    c::ThermodynamicConstants = DEFAULT_CONSTANTS,
                                    maxiter::Integer = 100, tol::Real = 1e-9)
    N = length(lad)
    length(z_faces) == N + 1 ||
        throw(DimensionMismatch("canopy_leaf_energy_balance: need length(z_faces) == length(lad) + 1"))
    zf = Float64.(z_faces)
    Δz = diff(zf)
    all(>(0), Δz) || throw(ArgumentError("canopy_leaf_energy_balance: z_faces must increase"))
    dΛ = [Float64(lad[k]) * Δz[k] for k in 1:N]
    all(>=(0), dΛ) || throw(ArgumentError("canopy_leaf_energy_balance: lad must be ≥ 0"))
    for x in (T_air, e_air, U)
        x isa Real || length(x) == N ||
            throw(DimensionMismatch("canopy_leaf_energy_balance: T_air, e_air, U must be scalars or length N"))
    end
    st = _as_stomata(stomata)
    bands = _sw_bands(shortwave)
    for b in bands
        length(b.absorbed) == N && isapprox(b.z_faces, zf; rtol = 1e-12, atol = 1e-12) ||
            throw(DimensionMismatch("canopy_leaf_energy_balance: shortwave solution is on different layers"))
    end
    par === nothing || isempty(bands) || 1 <= par <= length(bands) ||
        throw(ArgumentError("canopy_leaf_energy_balance: par must index a band of `shortwave` or be nothing"))

    # Class leaf areas and absorbed shortwave per unit class leaf area.
    lsun = isempty(bands) ? zeros(N) : copy(bands[1].lai_sunlit)
    lsha = dΛ .- lsun
    for k in 1:N
        lsha[k] = max(lsha[k], 0.0)
        lsun[k] = dΛ[k] - lsha[k]
        abs(lsun[k] + lsha[k] - dΛ[k]) <= 1e-9 * max(dΛ[k], 1.0) ||
            throw(DimensionMismatch("canopy_leaf_energy_balance: shortwave leaf area differs from lad"))
    end
    sw_layer = zeros(N); q_sun = zeros(N); q_sha = zeros(N); Q_sun = zeros(N); Q_sha = zeros(N)
    for (ib, b) in enumerate(bands), k in 1:N
        sw_layer[k] += b.absorbed[k]
        lsun[k] > 0 && (q_sun[k] += b.absorbed_sunlit[k] / lsun[k])
        lsha[k] > 0 && (q_sha[k] += b.absorbed_shaded[k] / lsha[k])
        if par !== nothing && ib == par
            lsun[k] > 0 && (Q_sun[k] = par_to_photon_flux(b.absorbed_sunlit[k] / lsun[k]))
            lsha[k] > 0 && (Q_sha[k] = par_to_photon_flux(b.absorbed_shaded[k] / lsha[k]))
        end
    end

    Ta = [_layer_value(T_air, k) for k in 1:N]
    ea = [_layer_value(e_air, k) for k in 1:N]
    Ua = [_layer_value(U, k) for k in 1:N]
    C = Float64(heat_capacity)
    prognostic = C > 0 && isfinite(dt)
    Tp_sun = Tp_sha = nothing
    if T_prev !== nothing
        Tp_sun = Float64.(T_prev isa CanopyLeafEnergySolution ? T_prev.T_sunlit : T_prev.sunlit)
        Tp_sha = Float64.(T_prev isa CanopyLeafEnergySolution ? T_prev.T_shaded : T_prev.shaded)
        (length(Tp_sun) == N && length(Tp_sha) == N) ||
            throw(DimensionMismatch("canopy_leaf_energy_balance: T_prev must have one value per layer"))
    end
    prognostic && T_prev === nothing &&
        throw(ArgumentError("canopy_leaf_energy_balance: a prognostic step needs T_prev"))

    Tsun = Tp_sun === nothing ? copy(Ta) : copy(Tp_sun)
    Tsha = Tp_sha === nothing ? copy(Ta) : copy(Tp_sha)
    σ = STEFAN_BOLTZMANN
    teff(k) = dΛ[k] > 0 ? ((lsun[k] * Tsun[k]^4 + lsha[k] * Tsha[k]^4) / dΛ[k])^(1 / 4) : Ta[k]
    lwsolve() = canopy_longwave(zf, lad; lw_down = lw_down, T_leaf = [teff(k) for k in 1:N],
                                T_ground = T_ground, ε_leaf = ε_leaf, ε_ground = ε_ground,
                                leaves = leaves)
    res_sun = Vector{Union{Nothing,LeafEnergyBalance}}(nothing, N)
    res_sha = Vector{Union{Nothing,LeafEnergyBalance}}(nothing, N)
    it = 0; conv = false; all_ok = true
    while it < maxiter
        it += 1
        lw = lwsolve()
        δ = 0.0; all_ok = true
        for k in 1:N
            dΛ[k] > 0 || continue
            εe = 2 * lw.emissivity_layer[k] / dΛ[k]
            Te4 = teff(k)^4
            L_abs = (lw.net_absorbed[k] + 2 * lw.emissivity_layer[k] * σ * Te4) / dΛ[k]
            for (cls, larea, qsw, Qp, Tc, Tpc, store) in
                    ((:sun, lsun[k], q_sun[k], Q_sun[k], Tsun, Tp_sun, res_sun),
                     (:sha, lsha[k], q_sha[k], Q_sha[k], Tsha, Tp_sha, res_sha))
                larea > 0 || continue
                r = _solve_leaf(st, boundary_layer, qsw + L_abs, εe, Ta[k], ea[k], Ua[k],
                                Float64(p_air), Qp, Tpc === nothing ? nothing : Tpc[k],
                                C, Float64(dt), c, 200)
                all_ok &= r.converged
                δ = max(δ, abs(r.T_leaf - Tc[k]))
                Tc[k] = r.T_leaf
                store[k] = r
            end
        end
        if δ <= tol
            conv = true
            break
        end
    end

    lw = lwsolve()
    H = zeros(N); LE = zeros(N); S = zeros(N); E = zeros(N); An = zeros(N)
    gsun = zeros(N); gsha = zeros(N)
    resid = 0.0
    for k in 1:N
        if dΛ[k] == 0
            Tsun[k] = Ta[k]; Tsha[k] = Ta[k]
            continue
        end
        for (larea, r) in ((lsun[k], res_sun[k]), (lsha[k], res_sha[k]))
            (larea > 0 && r !== nothing) || continue
            H[k] += larea * r.H; LE[k] += larea * r.LE; S[k] += larea * r.storage
            E[k] += larea * r.E; An[k] += larea * r.An
        end
        gsha[k] = res_sha[k] === nothing ? 0.0 : res_sha[k].gs
        gsun[k] = res_sun[k] === nothing ? gsha[k] : res_sun[k].gs
        if lsun[k] == 0
            Tsun[k] = Tsha[k]
        elseif lsha[k] == 0
            Tsha[k] = Tsun[k]; gsha[k] = gsun[k]
        end
        resid = max(resid, abs(sw_layer[k] + lw.net_absorbed[k] - H[k] - LE[k] - S[k]))
    end
    Mw = c.molmass_water
    return CanopyLeafEnergySolution(zf, lw.Λ_faces, Tsun, Tsha, gsun, gsha, lsun, lsha,
                                    sw_layer, copy(lw.net_absorbed), H, LE, S, E, An,
                                    H ./ Δz, E .* Mw ./ Δz, sum(H), sum(LE), sum(E),
                                    lw, resid, it, conv && all_ok)
end
canopy_leaf_energy_balance(p::AbstractLeafAreaDensity, z_faces::AbstractVector{<:Real}; kwargs...) =
    canopy_leaf_energy_balance(z_faces, layer_leaf_area_density(p, z_faces); kwargs...)

# =========================================================================
# 6. Within-canopy profiles: scalar column and displacement height (#25)
# =========================================================================

"""
    canopy_displacement_height(sol::CanopyWindSolution, lad; C_d) -> d [m]

Zero-plane displacement of a canopy as the mean height of momentum
absorption (Thom 1971; Jackson 1981):

```math
d = \\frac{\\sum_k z_k\\,C_d a_k |U_k| U_k\\,Δz_k + 0\\cdot τ_g}
         {\\sum_k C_d a_k |U_k| U_k\\,Δz_k + τ_g},
```

with the drag and ground stress ``τ_g`` of a converged
[`solve_canopy_wind_column`](@ref) solution on the same layers (`lad` the
layer LAD it was solved with). Heights are measured from `z_faces[1]`.
This is what tower studies report as ``d`` when they infer it from the
within-canopy momentum-flux profile.

  * Thom, A. S. (1971). Momentum absorption by vegetation. *Q. J. R.
    Meteorol. Soc.*, 97, 414-428.
  * Jackson, P. S. (1981). On the displacement height in the logarithmic
    velocity profile. *J. Fluid Mech.*, 111, 15-25.
"""
function canopy_displacement_height(sol::CanopyWindSolution, lad::AbstractVector{<:Real}; C_d::Real)
    N = length(sol.u)
    length(lad) == N || throw(DimensionMismatch("canopy_displacement_height: lad must have one value per layer"))
    Δz = diff(sol.z_faces)
    num = 0.0; den = sol.tau_ground
    for k in 1:N
        D = Float64(C_d) * Float64(lad[k]) * abs(sol.u[k]) * sol.u[k] * Δz[k]
        num += (sol.z[k] - sol.z_faces[1]) * D
        den += D
    end
    den > 0 || throw(ArgumentError("canopy_displacement_height: no momentum is absorbed"))
    return num / den
end

"""
    canopy_mixing_length(z; h, l_c, κ = SurfaceExchange.von_karman()) -> l [m]

Mixing-length profile for [`solve_canopy_wind_column`](@ref) over a canopy
of height `h` with canopy mixing length `l_c` (e.g.
[`harman_finnigan_mixing_length`](@ref)):

```math
l(z) = \\min(l_c,\\ κ z)\\quad (z ≤ h),\\qquad l(z) = l_c + κ (z - h)\\quad (z > h).
```

Inside the canopy the length is the constant ``l_c`` of the
Inoue / Harman–Finnigan theory, limited by the wall value ``κz`` near the
ground so that eddies cannot be larger than their distance from it (with a
constant ``l_c`` down to the ground a no-slip floor takes a spurious share
of the stress). Above the canopy it grows at the surface-layer rate
``κ`` from ``l_c``, continuous at ``h``; the roughness-sublayer enhancement
of Harman & Finnigan (2007) is not included.
"""
function canopy_mixing_length(z::Real; h::Real, l_c::Real, κ::Real = von_karman())
    (h > 0 && l_c > 0) || throw(ArgumentError("canopy_mixing_length: h and l_c must be > 0"))
    z <= h && return min(Float64(l_c), κ * max(Float64(z), 0.0))
    return l_c + κ * (z - h)
end

"""
    harman_finnigan_scalar_diffusivity(z; h, u_star, l_mix, β = 0.3, Sc = 0.5,
                                       κ = SurfaceExchange.von_karman()) -> K_c [m² s⁻¹]

Turbulent diffusivity for scalars in and above a canopy in the form of
Harman & Finnigan (2008), as used by the multilayer canopy of Bonan et al.
(2018, eqs. 22-23):

```math
K_c(z) = \\frac{l\\,u_*}{Sc}\\,\\exp\\!\\left[\\frac{(z-h)\\,β}{l}\\right]\\quad (z ≤ h),
\\qquad K_c(z) = \\frac{(l + κ(z-h))\\,u_*}{Sc}\\quad (z > h),
```

with the canopy mixing length `l_mix` (constant in the canopy, e.g.
[`harman_finnigan_mixing_length`](@ref)), ``β = u_*/U_h`` and the canopy-top
turbulent Schmidt number `Sc` (0.5 in neutral flow, Harman & Finnigan
2008). Inside the canopy the diffusivity decays with the wind,
``e^{(z-h)β/l}`` being the exponential wind profile of
[`canopy_exponential_wind`](@ref). Above the canopy the constant-stress
eddy viscosity ``l\\,u_*`` of the mixing length ``l + κ(z-h)`` is used,
which is continuous at ``z = h``; the roughness-sublayer correction of
Harman & Finnigan is not included.

Unlike the local ``l^2|∂U/∂z|`` of [`solve_canopy_wind_column`](@ref), this
form does not collapse in a sheltered trunk space, where turbulence is
imported from the crown rather than made by local shear.
"""
function harman_finnigan_scalar_diffusivity(z::Real; h::Real, u_star::Real, l_mix::Real,
                                            β::Real = 0.3, Sc::Real = 0.5,
                                            κ::Real = von_karman())
    (h > 0 && l_mix > 0 && β > 0 && Sc > 0) ||
        throw(ArgumentError("harman_finnigan_scalar_diffusivity: h, l_mix, β, Sc must be > 0"))
    z <= h && return l_mix * u_star / Sc * exp((z - h) * β / l_mix)
    return (l_mix + κ * (z - h)) * u_star / Sc
end

"""
    CanopyScalarSolution

Result of [`solve_canopy_scalar_column`](@ref): layer-centre heights `z`,
the scalar `c` there, the faces `z_faces` and the upward turbulent flux
`flux` at every face (`flux[1]` into the air from the ground, `flux[end]`
leaving through the top), in the units of `K·c/z`.
"""
struct CanopyScalarSolution
    z::Vector{Float64}
    c::Vector{Float64}
    z_faces::Vector{Float64}
    flux::Vector{Float64}
end

"""
    solve_canopy_scalar_column(z_faces, K, source; top_value,
                               bottom_flux = nothing, bottom_value = nothing)
        -> CanopyScalarSolution

Steady profile of a scalar ``c`` (potential temperature, humidity, a
tracer) in a canopy column with first-order (K-theory) turbulent
transport:

```math
\\frac{d}{dz}\\left(K \\frac{dc}{dz}\\right) + S(z) = 0,\\qquad
F = -K\\frac{dc}{dz} \\ \\text{(upward flux)},
```

with the per-layer source `source` [c-units s⁻¹] (e.g. the leaf
`heat_source` of [`canopy_leaf_energy_balance`](@ref) divided by
``ρc_p``), the diffusivity `K` [m² s⁻¹] at the `N+1` faces (e.g. the
`K` of a [`CanopyWindSolution`](@ref) divided by a turbulent Schmidt
number), `c = top_value` at the top face and at the bottom either an
upward flux `bottom_flux` (face 1) or a surface value `bottom_value`
exchanged through `K[1]` over the half layer. Exactly one bottom condition
must be given.

In one dimension the steady flux is fixed by the sources alone,
``F_{f+1} = F_f + S_k Δz_k``, so the discrete budget holds exactly:
the flux out of the top is the bottom flux plus the column-integrated
source. The scalar then follows by integrating ``dc/dz = -F/K`` down from
the top.

This is the closure of Harman & Finnigan (2008) and of multilayer land
models (e.g. Bonan et al. 2018). K-theory cannot represent
counter-gradient transport by canopy-scale eddies (Denmead & Bradley
1985), so in a canopy with a sparse trunk space it describes the crown
better than the air beneath it.

  * Harman, I. N. & Finnigan, J. J. (2008). Scalar concentration profiles
    in the canopy and roughness sublayer. *Boundary-Layer Meteorol.*,
    129, 323-351.
  * Bonan, G. B. et al. (2018). Modeling canopy-induced turbulence in the
    Earth system. *Geosci. Model Dev.*, 11, 1467-1496.
  * Denmead, O. T. & Bradley, E. F. (1985). Flux-gradient relationships
    in a forest canopy. In *The Forest-Atmosphere Interaction*, Reidel,
    421-442.
"""
function solve_canopy_scalar_column(z_faces::AbstractVector{<:Real}, K::AbstractVector{<:Real},
                                    source::AbstractVector{<:Real}; top_value::Real,
                                    bottom_flux::Union{Nothing,Real} = nothing,
                                    bottom_value::Union{Nothing,Real} = nothing)
    N = length(z_faces) - 1
    N >= 1 || throw(ArgumentError("solve_canopy_scalar_column: need ≥ 2 faces"))
    length(K) == N + 1 || throw(DimensionMismatch("solve_canopy_scalar_column: K must be given at the N+1 faces"))
    length(source) == N || throw(DimensionMismatch("solve_canopy_scalar_column: one source per layer"))
    (bottom_flux === nothing) != (bottom_value === nothing) ||
        throw(ArgumentError("solve_canopy_scalar_column: give exactly one of bottom_flux, bottom_value"))
    zf = Float64.(z_faces)
    Δz = diff(zf)
    all(>(0), Δz) || throw(ArgumentError("solve_canopy_scalar_column: z_faces must increase"))
    Kf = Float64.(K)
    all(>(0), @view Kf[2:N+1]) ||
        throw(ArgumentError("solve_canopy_scalar_column: K must be > 0 at every face above the ground"))
    zc = (zf[1:N] .+ zf[2:N+1]) ./ 2
    δ = similar(zf)
    δ[1] = zc[1] - zf[1]
    for f in 2:N
        δ[f] = zc[f] - zc[f-1]
    end
    δ[N+1] = zf[N+1] - zc[N]
    # Flux profile from a zero bottom flux: P_f = Σ_{j<f} S_j Δz_j.
    P = zeros(N + 1)
    for k in 1:N
        P[k+1] = P[k] + Float64(source[k]) * Δz[k]
    end
    # c_k = top − Σ_{faces above k} F_f δ_f / K_f is affine in the bottom flux F₁:
    # c_k = a_k + b_k F₁.
    a = zeros(N); b = zeros(N)
    acc_a = Float64(top_value); acc_b = 0.0
    for k in N:-1:1
        acc_a += P[k+1] * δ[k+1] / Kf[k+1]
        acc_b += δ[k+1] / Kf[k+1]
        a[k] = acc_a; b[k] = acc_b
    end
    F1 = if bottom_flux !== nothing
        Float64(bottom_flux)
    else
        Kf[1] > 0 || throw(ArgumentError("solve_canopy_scalar_column: bottom_value needs K[1] > 0"))
        # c_g − c₁ = F₁ δ₁ / K₁  ⇒  F₁ = (c_g − a₁) / (b₁ + δ₁/K₁)
        (Float64(bottom_value) - a[1]) / (b[1] + δ[1] / Kf[1])
    end
    c = a .+ b .* F1
    return CanopyScalarSolution(zc, c, zf, P .+ F1)
end

end # module
