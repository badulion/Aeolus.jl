"""
    TurbulenceStability

The Era VI stability contract: one family of gradient-Richardson
stability functions, the one definition of the squared buoyancy frequency
from a θ difference, and the one definition of the gradient Richardson
number. Every Ri-dependent closure, mixing length, transition diagnostic
and stable-boundary-layer analysis reads ``N^2``, ``Ri_g`` and
``f_m, f_h`` from here.

The module is pure (no dependencies, no state) and is included *before*
`Operators`, so that the LES closure and the pointwise operators of
`Operators` (`buoyancy_frequency_squared!`, `gradient_richardson!`) can
call it.

# The contract

A local first-order closure writes the eddy viscosity and diffusivity as

```math
K_m = l^2 |S| \\, f_m(Ri), \\qquad
K_h = \\frac{l^2 |S|}{Pr_n} \\, f_h(Ri), \\qquad
Ri = \\frac{N^2}{S^2},
```

with ``l`` a mixing length, ``|S|`` the shear (or strain-rate) magnitude,
and ``Pr_n`` the neutral turbulent Prandtl number. An implementer
`sf <: AbstractRiStability` provides

    stability_factor_m(sf, Ri)   # f_m, with f_m(0) = 1
    stability_factor_h(sf, Ri)   # f_h, with f_h(0) = 1
    neutral_prandtl(sf)          # Pr_n (default 1)

Both factors are normalised to 1 at neutral stratification; the neutral
Prandtl number is kept separate in [`neutral_prandtl`](@ref), so that the
factors of different families can be compared directly.

# Implementers

| type | stable tail | source |
| :-- | :-- | :-- |
| [`LillyCutoff`](@ref) | ``\\sqrt{1 - Ri/Ri_c}``, 0 above ``Ri_c`` | Lilly (1962); Mason (1989) |
| [`BrownEtAl1994`](@ref) | ``(1 - Ri/Ri_c)^4``, 0 above ``Ri_c`` | Brown, Derbyshire & Mason (1994), the LEM / MONC functions |
| [`ShortTail`](@ref) | ``(1 + 5Ri)^{-2}`` | the radiation-fog column of Phase 44 |
| [`LongTail`](@ref) | ``[1 + 10Ri/\\sqrt{1 + 5Ri}]^{-1}`` | Louis (1979) |
| `SurfaceExchange.MOSTConsistent` | from ``φ_m, φ_h`` | Monin–Obukhov similarity |

# References

  * Lilly, D. K. (1962). On the numerical simulation of buoyant
    convection. *Tellus*, 14, 148-172.
  * Mason, P. J. (1989). Large-eddy simulation of the convective
    atmospheric boundary layer. *J. Atmos. Sci.*, 46, 1492-1516.
  * Brown, A. R., Derbyshire, S. H. & Mason, P. J. (1994). Large-eddy
    simulation of stable atmospheric boundary layers with a revised
    stochastic subgrid model. *Q. J. R. Meteorol. Soc.*, 120, 1485-1512.
  * Louis, J.-F. (1979). A parametric model of vertical eddy fluxes in
    the atmosphere. *Boundary-Layer Meteorol.*, 17, 187-202.
"""
module TurbulenceStability

export AbstractRiStability, stability_factor_m, stability_factor_h, neutral_prandtl,
       LillyCutoff, BrownEtAl1994, ShortTail, LongTail,
       gradient_richardson, buoyancy_frequency_squared, RI_SHEAR2_FLOOR,
       boundary_layer_height

# =========================================================================
# The generics
# =========================================================================

"""
    AbstractRiStability

A family of gradient-Richardson stability functions. A subtype implements
[`stability_factor_m`](@ref) and [`stability_factor_h`](@ref), both equal
to 1 at `Ri = 0`, non-negative, and non-increasing in `Ri > 0`, and may
override [`neutral_prandtl`](@ref).
"""
abstract type AbstractRiStability end

"""
    stability_factor_m(sf::AbstractRiStability, Ri) -> f_m

Stability factor of the eddy viscosity, ``K_m = l^2 |S| f_m(Ri)``, with
``f_m(0) = 1``.
"""
function stability_factor_m end

"""
    stability_factor_h(sf::AbstractRiStability, Ri) -> f_h

Stability factor of the eddy diffusivity of heat and scalars,
``K_h = l^2 |S| f_h(Ri) / Pr_n``, with ``f_h(0) = 1``
(``Pr_n`` = [`neutral_prandtl`](@ref)).
"""
function stability_factor_h end

"""
    neutral_prandtl(sf::AbstractRiStability) -> Pr_n

Turbulent Prandtl number at neutral stratification, ``K_m/K_h`` at
``Ri = 0``. Default 1.
"""
neutral_prandtl(::AbstractRiStability) = 1.0

# =========================================================================
# N² and Ri_g — the one definition
# =========================================================================

"""
    buoyancy_frequency_squared(θ_below, θ_above, Δz, g, θ_ref) -> N²
    buoyancy_frequency_squared(θ_below, θ_above, Δz, g)        -> N²

Squared buoyancy (Brunt–Väisälä) frequency across a layer of thickness
`Δz` from the potential temperature (or virtual potential temperature)
below and above it.

* With a reference temperature `θ_ref`, the Boussinesq form
  ``N^2 = (g/θ_{\\rm ref})\\,(θ_{\\rm above} - θ_{\\rm below})/Δz``,
  evaluated as `(g / θ_ref) * (θ_above - θ_below) * (1 / Δz)`.
* Without, the logarithmic form ``N^2 = g\\,(\\ln θ_{\\rm above} -
  \\ln θ_{\\rm below})/Δz``, which is exact for the constant-``N^2``
  profile ``θ = θ_s e^{N^2 z/g}`` of a deep (anelastic) reference state.

The pointwise field (`Operators.buoyancy_frequency_squared!`), the
CFL buoyancy limit of `TimeIntegration` and the reference profile of
`Anelastic` all evaluate ``N^2`` through this function.
"""
@inline buoyancy_frequency_squared(θ_below, θ_above, Δz, g, θ_ref) =
    (g / θ_ref) * (θ_above - θ_below) * (1 / Δz)

@inline buoyancy_frequency_squared(θ_below, θ_above, Δz, g) =
    g * (log(θ_above) - log(θ_below)) / Δz

"""
    RI_SHEAR2_FLOOR = 1e-10  (s⁻²)

Default floor on ``S^2`` in [`gradient_richardson`](@ref). It
corresponds to a shear of ``10^{-5}`` s⁻¹ (1 cm s⁻¹ over 1 km), far below
any resolved LES shear, and keeps ``Ri`` finite (``|Ri| ≤ |N^2|/10^{-10}``)
in shear-free cells (solid cells, a fluid at rest). Every implementer
returns a finite factor for any finite ``Ri``.
"""
const RI_SHEAR2_FLOOR = 1e-10

"""
    gradient_richardson(N², S²; S2_min = RI_SHEAR2_FLOOR) -> Ri

Gradient Richardson number ``Ri = N^2 / \\max(S^2, S^2_{\\min})``. `S²`
is the squared shear, ``(∂u/∂z)^2 + (∂v/∂z)^2`` in a column model or
``|S|^2 = 2 S_{ij} S_{ij}`` in an LES (identical for a pure vertical
shear). The floor `S2_min` (see [`RI_SHEAR2_FLOOR`](@ref)) keeps `Ri`
finite where the flow is shear-free.
"""
@inline gradient_richardson(N2, S2; S2_min = RI_SHEAR2_FLOOR) = N2 / max(S2, S2_min)

# =========================================================================
# Implementers
# =========================================================================

"""
    LillyCutoff(; Ri_c = 0.25)

The Smagorinsky–Lilly stability correction (Lilly 1962; Mason 1989):

```math
f_m = f_h = \\sqrt{\\max(0,\\, 1 - Ri/Ri_c)}.
```

Lilly (1962) obtained it from the equilibrium TKE budget
``K_m S^2 - K_h N^2 = ε`` with ``ε ∝ K^3/l^4``, which gives
``K = l^2 (S^2 - N^2/Ri_c)^{1/2}``; turbulence is switched off at the
critical Richardson number ``Ri_c`` (= the turbulent Prandtl number in
Lilly's derivation; 0.25 here). Mason (1989) uses the same single
expression on the unstable side, where it grows as ``\\sqrt{1 + |Ri|/Ri_c}``.
`f` is exactly 0 for ``Ri ≥ Ri_c`` and continuous there (it goes to 0 as
``\\sqrt{Ri_c - Ri}``).
"""
Base.@kwdef struct LillyCutoff{T<:AbstractFloat} <: AbstractRiStability
    Ri_c::T = 0.25
end

@inline _lilly(sf::LillyCutoff, Ri) = sqrt(max(zero(Ri), 1 - Ri / sf.Ri_c))
stability_factor_m(sf::LillyCutoff, Ri::Real) = _lilly(sf, Ri)
stability_factor_h(sf::LillyCutoff, Ri::Real) = _lilly(sf, Ri)

"""
    BrownEtAl1994(; Ri_c = 0.25, c = 16, b = 40, h = 1.2, Pr_n = 0.7)

The stability functions of the UK Met Office LEM and MONC
(Brown, Derbyshire & Mason 1994), with the constants of the LEM/MONC
code:

*Stable* (``0 ≤ Ri < Ri_c``; both 0 for ``Ri ≥ Ri_c``):

```math
f_m = (1 - Ri/Ri_c)^4, \\qquad f_h = (1 - Ri/Ri_c)^4\\,(1 - h\\,Ri).
```

*Unstable* (``Ri < 0``):

```math
f_m = (1 - c\\,Ri)^{1/2}, \\qquad f_h = (1 - b\\,Ri)^{1/2},
```

with ``Ri_c = 0.25``, ``c = 16``, ``b = 40``, ``h = 1.2`` and the
neutral Prandtl number ``Pr_n = 0.7`` (the LEM's ``a = 1/Pr_n ≈ 1.43``
multiplying ``f_h``; here returned by [`neutral_prandtl`](@ref), so that
``f_h(0) = 1``).
"""
Base.@kwdef struct BrownEtAl1994{T<:AbstractFloat} <: AbstractRiStability
    Ri_c::T = 0.25
    c::T    = 16.0
    b::T    = 40.0
    h::T    = 1.2
    Pr_n::T = 0.7
end

function stability_factor_m(sf::BrownEtAl1994, Ri::Real)
    Ri < 0 && return sqrt(1 - sf.c * Ri)
    Ri >= sf.Ri_c && return zero(float(Ri))
    return (1 - Ri / sf.Ri_c)^4
end

function stability_factor_h(sf::BrownEtAl1994, Ri::Real)
    Ri < 0 && return sqrt(1 - sf.b * Ri)
    Ri >= sf.Ri_c && return zero(float(Ri))
    return (1 - Ri / sf.Ri_c)^4 * (1 - sf.h * Ri)
end

neutral_prandtl(sf::BrownEtAl1994) = sf.Pr_n

"""
    ShortTail()

The short-tailed function of the radiation-fog column (Phase 44), moved
here unchanged from `RadiationFog` (`_fstab`):

```math
f_m = f_h = \\begin{cases} (1 + 5Ri)^{-2} & Ri ≥ 0 \\\\
                           \\sqrt{1 - 16 Ri} & Ri < 0 \\end{cases}
```

The stable branch decays as ``Ri^{-2}``, which lets turbulence collapse in
a strong inversion (as radiation fog needs); the unstable branch is the
Businger–Dyer ``φ_m^{-2}`` with ``γ = 16`` at ``Ri = ζ``. Same factor for
momentum and heat, ``Pr_n = 1``.
"""
struct ShortTail <: AbstractRiStability end

@inline _short_tail(Ri) = Ri >= 0 ? 1 / (1 + 5Ri)^2 : sqrt(1 - 16Ri)
stability_factor_m(::ShortTail, Ri::Real) = _short_tail(Ri)
stability_factor_h(::ShortTail, Ri::Real) = _short_tail(Ri)

"""
    LongTail(; b = 5, d = 5, Pr_n = 0.74)

The long-tailed functions of Louis (1979):

*Stable* (``Ri ≥ 0``):

```math
f_m = \\left[1 + \\frac{2 b\\,Ri}{\\sqrt{1 + d\\,Ri}}\\right]^{-1}, \\qquad
f_h = \\left[1 + 3 b\\,Ri \\sqrt{1 + d\\,Ri}\\right]^{-1},
```

with Louis's ``b = d = 5``. ``f_m`` decays only as ``Ri^{-1/2}``: there
is no critical Richardson number, and mixing persists in strong
stability (the behaviour of operational NWP schemes).

*Unstable* (``Ri < 0``): Louis's unstable form,
``f = 1 - 2b\\,Ri/(1 + 3bc\\,C\\sqrt{|Ri|})`` for momentum
(``3b`` for heat), has a coefficient ``C`` built from the surface
roughness (``C = κ^2 (z/z_0)^{1/2}/\\ln^2(z/z_0)``). A gradient function
away from the surface has no roughness length, so the free-atmosphere
limit ``C → 0`` is used: ``f_m = 1 - 2b\\,Ri``, ``f_h = 1 - 3b\\,Ri``.

``Pr_n = 0.74``: Louis's ratio ``R`` of the neutral momentum and heat
transfer coefficients (from Businger et al. 1971).
"""
Base.@kwdef struct LongTail{T<:AbstractFloat} <: AbstractRiStability
    b::T    = 5.0
    d::T    = 5.0
    Pr_n::T = 0.74
end

function stability_factor_m(sf::LongTail, Ri::Real)
    Ri < 0 && return 1 - 2 * sf.b * Ri
    return 1 / (1 + 2 * sf.b * Ri / sqrt(1 + sf.d * Ri))
end

function stability_factor_h(sf::LongTail, Ri::Real)
    Ri < 0 && return 1 - 3 * sf.b * Ri
    return 1 / (1 + 3 * sf.b * Ri * sqrt(1 + sf.d * Ri))
end

neutral_prandtl(sf::LongTail) = sf.Pr_n

# =========================================================================
# Boundary-layer height — the one z_i diagnostic (N-parent-a, #247)
# =========================================================================

"""
    boundary_layer_height(z, profile; method = :parcel, kwargs...) -> z_i (m)

Height of the top of the boundary layer from a profile sampled at the
strictly increasing heights `z` (m). The one ``z_i`` diagnostic of the
package: it is applied identically to radiosonde profiles
(`LargeScaleForcing.sounding_profiles`) and to LES output (the CBL
harness `RunDrivers.CBLDriver.tke_cbl`, the nesting epic #246).

`method`:

* `:parcel` — `profile` is the (virtual) potential temperature. The
  height at which a parcel rising dry-adiabatically from the lowest
  level with the temperature excess `excess` (K, default 0) becomes
  neutrally buoyant: the first crossing of ``θ_v(z) = θ_v(z_1) +
  \\text{excess}``, linearly interpolated (Holzworth 1964; Seibert et al.
  2000, §3.1). On a stable profile it returns `z[1]`.
* `:bulk_richardson` — `profile` is ``θ_v``; keywords `u`, `v` (wind
  components at `z`, default calm), `g = 9.81`, `θ_ref = profile[1]`,
  `Ri_c = 0.25`, `u_star = 0` and `b = 100`. The lowest height at which
  the bulk Richardson number from the lowest level,

  ```math
  Ri_b(z) = \\frac{(g/θ_{ref})\\,(θ_v(z) - θ_v(z_1))\\,(z - z_1)}
                 {(u(z) - u(z_1))^2 + (v(z) - v(z_1))^2 + b\\,u_*^2},
  ```

  reaches `Ri_c`, linearly interpolated between levels (Vogelezang &
  Holtslag 1996, eq. 3, with their b = 100). It is evaluated as
  [`gradient_richardson`](@ref)`(N², S²)` of the layer
  ``[z_1, z]`` with [`buoyancy_frequency_squared`](@ref), so the shear
  floor `RI_SHEAR2_FLOOR` handles a calm profile (``Ri_b → +∞`` above
  a jump, the height then is the level below it).
* `:max_gradient` — `profile` is θ; the midpoint of the layer with the
  largest ``∂θ/∂z`` (the inversion).
* `:min_flux` — `profile` is the total (resolved + SGS) kinematic heat
  flux ``\\overline{w'θ'}`` (any normalisation); the height of its
  minimum, `z[argmin(profile)]` (Sullivan et al. 1998, §3; the
  definition of `tke_cbl`'s ``z_i``).

References:

* Holzworth, G. C. (1964). Estimates of mean maximum mixing depths in
  the contiguous United States. *Mon. Wea. Rev.*, 92, 235-242.
* Seibert, P., Beyrich, F., Gryning, S.-E., Joffre, S., Rasmussen, A. &
  Tercier, P. (2000). Review and intercomparison of operational methods
  for the determination of the mixing height. *Atmos. Environ.*, 34,
  1001-1027.
* Vogelezang, D. H. P. & Holtslag, A. A. M. (1996). Evaluation and
  model impacts of alternative boundary-layer height formulations.
  *Boundary-Layer Meteorol.*, 81, 245-269.
* Sullivan, P. P., Moeng, C.-H., Stevens, B., Lenschow, D. H. & Mayor,
  S. D. (1998). Structure of the entrainment zone capping the convective
  atmospheric boundary layer. *J. Atmos. Sci.*, 55, 3042-3064.
"""
function boundary_layer_height(z::AbstractVector, profile::AbstractVector;
                               method::Symbol = :parcel, excess::Real = 0.0,
                               u = nothing, v = nothing, g::Real = 9.81,
                               θ_ref::Real = profile[1], Ri_c::Real = 0.25,
                               u_star::Real = 0.0, b::Real = 100.0)
    n = length(z)
    n == length(profile) ||
        throw(ArgumentError("boundary_layer_height: z and profile differ in length ($n, $(length(profile)))"))
    n >= 2 || throw(ArgumentError("boundary_layer_height: need at least two levels"))
    all(>(0), diff(z)) || throw(ArgumentError("boundary_layer_height: z must be strictly increasing"))
    if method === :min_flux
        return z[argmin(profile)]
    elseif method === :max_gradient
        k = argmax([(profile[i+1] - profile[i]) / (z[i+1] - z[i]) for i in 1:n-1])
        return (z[k] + z[k+1]) / 2
    elseif method === :parcel
        return _first_crossing(z, profile, profile[1] + excess)
    elseif method === :bulk_richardson
        uu = u === nothing ? zeros(n) : u
        vv = v === nothing ? zeros(n) : v
        (length(uu) == n && length(vv) == n) ||
            throw(ArgumentError("boundary_layer_height: u and v must have the length of z"))
        Rib = zeros(float(eltype(profile)), n)
        for k in 2:n
            Δz = z[k] - z[1]
            N2 = buoyancy_frequency_squared(profile[1], profile[k], Δz, g, θ_ref)
            S2 = ((uu[k] - uu[1])^2 + (vv[k] - vv[1])^2 + b * u_star^2) / Δz^2
            Rib[k] = gradient_richardson(N2, S2)
        end
        return _first_crossing(z, Rib, Ri_c)
    end
    throw(ArgumentError("boundary_layer_height: unknown method :$method (:parcel, :bulk_richardson, :max_gradient, :min_flux)"))
end

# First height above z[1] where p exceeds `target`, interpolated linearly
# between the bracketing levels (clamped to them); z[end] if never.
function _first_crossing(z, p, target)
    for k in 2:length(z)
        if p[k] > target
            p[k-1] >= target && return float(z[k-1])
            w = (target - p[k-1]) / (p[k] - p[k-1])
            return z[k-1] + clamp(w, 0, 1) * (z[k] - z[k-1])
        end
    end
    return float(z[end])
end

end # module
