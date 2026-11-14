"""
    SurfaceExchange

The surface shared core: the physics every surface model needs, in
one place, so the soil, canopy, urban-facet, snow, sea-ice, ocean and
tile models delegate rather than re-derive. It is to the surface models what
`RadiationCore` is to radiation and `MoistThermodynamics` to microphysics.

Three parts:

  1. **Material constants** — [`SurfaceConstants`](@ref): densities and
     conductivities of water, ice and air, the Stefan–Boltzmann and von
     Kármán constants. Latent heats, heat capacities of water and ice and
     `g` are *delegated* to `MoistThermodynamics.ThermodynamicConstants`
     through accessors; they are never restated here.
  2. **Monin–Obukhov similarity** — the stability functions
     ``φ_m, φ_h, ψ_m, ψ_h`` as a dispatch hierarchy
     ([`AbstractStabilityFunctions`](@ref)) and a surface-layer solver
     ([`solve_surface_layer`](@ref)) for ``u_*``, ``θ_*``, ``q_*``, the
     Obukhov length and the bulk transfer coefficients.
  3. **The surface interface** — [`AbstractSurface`](@ref) and the
     generics a surface model implements (`surface_temperature`,
     `roughness_lengths`, `surface_albedo`, `surface_emissivity`,
     `step_surface!`), plus a default [`turbulent_fluxes`](@ref) that
     turns any implementer into sensible, latent and momentum fluxes.

# Sign convention

All turbulent and radiative fluxes are **positive upward** (away from
the surface) except net radiation `Rn`, which is positive *into* the
surface, and `G`, positive *into* the substrate — so the surface energy
balance reads ``R_n = H + LE + G``. Scales follow
``\\overline{w'θ'} = -u_* θ_*`` and ``\\overline{w'q'} = -u_* q_*``: a
surface warmer than the air gives ``θ_* < 0``, ``H > 0`` and ``L < 0``
(unstable). ``ζ = z/L`` is positive for stable stratification.

# References

  * Businger, J. A., Wyngaard, J. C., Izumi, Y. & Bradley, E. F. (1971).
    Flux-profile relationships in the atmospheric surface layer.
    *J. Atmos. Sci.*, 28, 181-189.
  * Dyer, A. J. (1974). A review of flux-profile relationships.
    *Boundary-Layer Meteorol.*, 7, 363-372.
  * Paulson, C. A. (1970). The mathematical representation of wind speed
    and temperature profiles in the unstable atmospheric surface layer.
    *J. Appl. Meteorol.*, 9, 857-861.
  * Beljaars, A. C. M. (1995). The parametrization of surface fluxes in
    large-scale models under free convection. *Q. J. R. Meteorol. Soc.*,
    121, 255-270.
  * Beljaars, A. C. M. & Holtslag, A. A. M. (1991). Flux parameterization
    over land surfaces for atmospheric models. *J. Appl. Meteorol.*, 30,
    327-341.
  * Fukusako, S. (1990). Thermophysical properties of ice, snow, and sea
    ice. *Int. J. Thermophys.*, 11, 353-372.
"""
module SurfaceExchange

using ..MoistThermodynamics
using ..MoistThermodynamics: ThermodynamicConstants, DEFAULT_CONSTANTS,
                             PhasePartition, R_d, R_v, θ_per_heat
using ..Backends: column_sum, backend
import ..MoistThermodynamics: LH_f0
using ..RadiationCore: STEFAN_BOLTZMANN
# The broadband albedo / emissivity of an AbstractSurface are further
# methods of the RadiationCore generics — one function, not two.
import ..RadiationCore: surface_albedo, surface_emissivity
# MOST-consistent gradient-Richardson factors are an implementer of the
# stability contract (P58-a, #170).
import ..TurbulenceStability: AbstractRiStability, stability_factor_m, stability_factor_h

export SurfaceConstants, DEFAULT_SURFACE_CONSTANTS,
       LH_v0, LH_s0, LH_f0, grav, cp_water, cp_ice, von_karman,
       AbstractStabilityFunctions, BusingerDyerBeljaarsHoltslag, BusingerDyerLogLinear,
       phi_m, phi_h, psi_m, psi_h,
       SurfaceLayerSolution, solve_surface_layer, aerodynamic_resistance, bulk_richardson, obukhov_length,
       kinematic_heat_flux, similarity_theta, similarity_wind,
       AbstractSurface, SurfaceForcing, SurfaceFluxes,
       has_packed_state, packed_state_length, pack_state!, unpack_state!,
       surface_temperature, roughness_lengths, surface_albedo, surface_emissivity,
       moisture_availability, surface_specific_humidity, surface_latent_heat,
       net_radiation, turbulent_fluxes, surface_layer_solution, step_surface!,
       MOSTConsistent, most_gradient_richardson, most_stability_parameter,
       log_linear_slope, MSHFSolution, mshf_solution, max_sustainable_heat_flux,
       flux_friction_velocity, surface_mshf_ratio,
       deliver_surface_fluxes!

# =========================================================================
# 1. Material constants
# =========================================================================

"""
    SurfaceConstants(; kwargs...)

Material constants of the surface models. Fields (SI units):

| field     | default          | meaning                                          |
|:--------- |:---------------- |:------------------------------------------------ |
| `ρ_water` | 1000             | liquid water density, kg m⁻³                     |
| `ρ_ice`   | 917              | ice density at 0 °C, kg m⁻³                      |
| `k_water` | 0.57             | liquid water conductivity near 0–10 °C, W m⁻¹ K⁻¹ |
| `k_ice`   | 2.22             | ice conductivity at 0 °C, W m⁻¹ K⁻¹ (Fukusako 1990) |
| `k_air`   | 0.024            | still-air conductivity near 0 °C, W m⁻¹ K⁻¹      |
| `σ_SB`    | `STEFAN_BOLTZMANN` | Stefan–Boltzmann, W m⁻² K⁻⁴ (CODATA 2018, from `RadiationCore`) |
| `κ`       | 0.40             | von Kármán constant                              |
| `thermo`  | `DEFAULT_CONSTANTS` | the `MoistThermodynamics` constants            |

Latent heats, the heat capacities of water and ice, and gravity are
**not** fields: they belong to `thermo` and are read through the
accessors [`LH_v0`](@ref), [`LH_s0`](@ref), `LH_f0`, [`grav`](@ref),
[`cp_water`](@ref), [`cp_ice`](@ref), so a surface model can never
disagree with the atmosphere about how much energy evaporation costs.
"""
Base.@kwdef struct SurfaceConstants{T<:AbstractFloat, C<:ThermodynamicConstants}
    ρ_water::T = 1000.0
    ρ_ice::T   = 917.0
    k_water::T = 0.57
    k_ice::T   = 2.22
    k_air::T   = 0.024
    σ_SB::T    = STEFAN_BOLTZMANN
    κ::T       = 0.40
    thermo::C  = DEFAULT_CONSTANTS
end

"The default [`SurfaceConstants`](@ref) (Float64)."
const DEFAULT_SURFACE_CONSTANTS = SurfaceConstants()

"Latent heat of vaporisation at the triple point (delegated to `MoistThermodynamics`)."
LH_v0(c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS) = c.thermo.LH_v0
"Latent heat of sublimation at the triple point (delegated to `MoistThermodynamics`)."
LH_s0(c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS) = c.thermo.LH_s0
LH_f0(c::SurfaceConstants) = LH_f0(c.thermo)
"Gravitational acceleration (delegated to `MoistThermodynamics`)."
grav(c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS) = c.thermo.grav
"Specific heat of liquid water, `thermo.cp_liquid`."
cp_water(c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS) = c.thermo.cp_liquid
"Specific heat of ice, `thermo.cp_ice`."
cp_ice(c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS) = c.thermo.cp_ice
"The von Kármán constant."
von_karman(c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS) = c.κ

# =========================================================================
# 2. Monin–Obukhov stability functions
# =========================================================================

"""
    AbstractStabilityFunctions

A family of Monin–Obukhov similarity functions. A concrete subtype
implements the four generics

    phi_m(sf, ζ), phi_h(sf, ζ)   # dimensionless gradients
    psi_m(sf, ζ), psi_h(sf, ζ)   # integrated forms

with ``ζ = z/L`` (positive stable), normalised so ``φ(0) = 1`` and
``ψ(0) = 0``, and related by ``ψ(ζ) = ∫₀^ζ [1 - φ(x)]/x dx``, i.e.
``dψ/dζ = (1 - φ)/ζ``. Profiles then read

    u(z) = (u_*/κ) [ln(z/z₀ₘ) - ψ_m(z/L) + ψ_m(z₀ₘ/L)]

and likewise for θ and q with ``ψ_h``. The surface-layer solver only
calls `psi_m` and `psi_h`, so a new family (e.g. the COARE convective
form of Grachev & Fairall) needs nothing else to be usable.
"""
abstract type AbstractStabilityFunctions end

"""
    BusingerDyerBeljaarsHoltslag(; γ_m = 16, γ_h = 16, a = 1, b = 2/3, c = 5, d = 0.35)

The default stability functions.

*Unstable* (``ζ < 0``): the Businger–Dyer gradients in Dyer's (1974)
form with κ = 0.40,
``φ_m = (1 - γ_m ζ)^{-1/4}``, ``φ_h = (1 - γ_h ζ)^{-1/2}``, and their
exact integrals (Paulson 1970), with ``x = (1 - γ_m ζ)^{1/4}``,
``y = (1 - γ_h ζ)^{1/2}``:

    ψ_m = 2 ln[(1+x)/2] + ln[(1+x²)/2] - 2 atan(x) + π/2
    ψ_h = 2 ln[(1+y)/2]

*Stable* (``ζ ≥ 0``): Beljaars & Holtslag (1991), which stay finite and
well-behaved in very stable conditions where the log-linear
``φ = 1 + 5ζ`` shuts turbulence off:

    ψ_m = -[a ζ + b (ζ - c/d) e^{-dζ} + b c/d]
    ψ_h = -[(1 + 2aζ/3)^{3/2} + b (ζ - c/d) e^{-dζ} + b c/d - 1]

with φ obtained analytically as ``φ = 1 - ζ dψ/dζ``.
"""
Base.@kwdef struct BusingerDyerBeljaarsHoltslag{T<:AbstractFloat} <: AbstractStabilityFunctions
    γ_m::T = 16.0
    γ_h::T = 16.0
    a::T   = 1.0
    b::T   = 2 / 3
    c::T   = 5.0
    d::T   = 0.35
end

"Dimensionless wind shear ``φ_m(ζ) = (κz/u_*) ∂u/∂z``."
function phi_m end
"Dimensionless potential-temperature (and scalar) gradient ``φ_h(ζ)``."
function phi_h end
"Integrated stability correction for momentum, ``ψ_m(ζ)``."
function psi_m end
"Integrated stability correction for heat and scalars, ``ψ_h(ζ)``."
function psi_h end

function phi_m(sf::BusingerDyerBeljaarsHoltslag, ζ::Real)
    ζ < 0 && return (1 - sf.γ_m * ζ)^(-1 / 4)
    return 1 + ζ * (sf.a + sf.b * exp(-sf.d * ζ) * (1 + sf.c - sf.d * ζ))
end

function phi_h(sf::BusingerDyerBeljaarsHoltslag, ζ::Real)
    ζ < 0 && return (1 - sf.γ_h * ζ)^(-1 / 2)
    return 1 + ζ * (sf.a * sqrt(1 + 2 * sf.a * ζ / 3) +
                    sf.b * exp(-sf.d * ζ) * (1 + sf.c - sf.d * ζ))
end

function psi_m(sf::BusingerDyerBeljaarsHoltslag, ζ::Real)
    if ζ < 0
        x = (1 - sf.γ_m * ζ)^(1 / 4)
        return 2 * log((1 + x) / 2) + log((1 + x^2) / 2) - 2 * atan(x) + π / 2
    end
    a, b, c, d = sf.a, sf.b, sf.c, sf.d
    return -(a * ζ + b * (ζ - c / d) * exp(-d * ζ) + b * c / d)
end

function psi_h(sf::BusingerDyerBeljaarsHoltslag, ζ::Real)
    if ζ < 0
        y = sqrt(1 - sf.γ_h * ζ)
        return 2 * log((1 + y) / 2)
    end
    a, b, c, d = sf.a, sf.b, sf.c, sf.d
    return -((1 + 2 * a * ζ / 3)^(3 / 2) + b * (ζ - c / d) * exp(-d * ζ) + b * c / d - 1)
end

"""
    BusingerDyerLogLinear(; γ_m = 16, γ_h = 16, β = 5, β_m = β, β_h = β)

Monin–Obukhov functions with the classical *log-linear* stable branch,
``φ_m = 1 + β_m ζ``, ``φ_h = 1 + β_h ζ`` and ``ψ_m = -β_m ζ``,
``ψ_h = -β_h ζ`` for ``ζ ≥ 0`` (Businger et al. 1971; Dyer 1974). The
unstable branch is delegated to [`BusingerDyerBeljaarsHoltslag`](@ref)
with the same γ, so the two families agree exactly for ``ζ < 0``.

`β` sets both coefficients; `β_m`, `β_h` set them separately. Uses:

  * ``β_m = β_h = 5`` — the water-side ``φ_t`` of Zeng & Beljaars (2005)
    (IFS Cy43r1 eq. 8.158), used by `OceanMixedLayer.ZengBeljaarsWarmLayer`;
  * ``β_m = 4.8``, ``β_h = 7.8`` — the surface-layer functions prescribed
    by the GABLS1 LES intercomparison (Beare et al. 2006, BLM 118, 247;
    quoted by Basu & Porté-Agel 2006, JAS 63, 2074, §4a), the
    `[surface] stability` (`model = "most"`) of examples/sbl/gabls1.toml.
"""
struct BusingerDyerLogLinear{T<:AbstractFloat} <: AbstractStabilityFunctions
    γ_m::T
    γ_h::T
    β_m::T
    β_h::T
end
function BusingerDyerLogLinear(; γ_m::Real = 16.0, γ_h::Real = 16.0, β::Real = 5.0,
                               β_m::Real = β, β_h::Real = β)
    T = float(promote_type(typeof(γ_m), typeof(γ_h), typeof(β_m), typeof(β_h)))
    return BusingerDyerLogLinear{T}(γ_m, γ_h, β_m, β_h)
end
_unstable(sf::BusingerDyerLogLinear) = BusingerDyerBeljaarsHoltslag(γ_m = sf.γ_m, γ_h = sf.γ_h)
phi_m(sf::BusingerDyerLogLinear, ζ::Real) = ζ < 0 ? phi_m(_unstable(sf), ζ) : 1 + sf.β_m * ζ
phi_h(sf::BusingerDyerLogLinear, ζ::Real) = ζ < 0 ? phi_h(_unstable(sf), ζ) : 1 + sf.β_h * ζ
psi_m(sf::BusingerDyerLogLinear, ζ::Real) = ζ < 0 ? psi_m(_unstable(sf), ζ) : -sf.β_m * ζ
psi_h(sf::BusingerDyerLogLinear, ζ::Real) = ζ < 0 ? psi_h(_unstable(sf), ζ) : -sf.β_h * ζ
# the stable momentum branch is log-linear: the closed-form MSHF applies
# with α = β_m (only φ_m enters the flux curve; P58-e, #193)
log_linear_slope(sf::BusingerDyerLogLinear) = sf.β_m

# -------------------------------------------------------------------------
# MOST-consistent gradient-Richardson stability factors (P58-a)
# -------------------------------------------------------------------------

"""
    most_gradient_richardson(sf::AbstractStabilityFunctions, ζ) -> Ri

The gradient Richardson number of the Monin–Obukhov profiles at
``ζ = z/L``:

```math
Ri = \\frac{(g/θ_0)\\,∂θ/∂z}{(∂u/∂z)^2} = ζ\\,\\frac{φ_h(ζ)}{φ_m(ζ)^2},
```

from ``∂u/∂z = u_* φ_m/(κz)``, ``∂θ/∂z = θ_* φ_h/(κz)`` and
``L = u_*^2 θ_0/(κ g θ_*)``. For the log-linear ``φ = 1 + αζ`` it tends to
the critical value ``1/α`` as ``ζ → ∞``; for Beljaars–Holtslag it grows
without bound (no critical Richardson number).
"""
most_gradient_richardson(sf::AbstractStabilityFunctions, ζ::Real) =
    ζ * phi_h(sf, ζ) / phi_m(sf, ζ)^2

"""
    MOSTConsistent(sf = BusingerDyerBeljaarsHoltslag(); ζ_min = -1e3, ζ_max = 1e4)

The gradient-Richardson stability factors that make a local closure
``K = (κz)^2 |∂u/∂z| f(Ri)`` reproduce Monin–Obukhov similarity with the
functions `sf` exactly:

```math
f_m = φ_m(ζ)^{-2}, \\qquad f_h = [φ_m(ζ)\\, φ_h(ζ)]^{-1}, \\qquad
Ri = ζ\\,φ_h(ζ)/φ_m(ζ)^2 .
```

With ``∂u/∂z = u_*φ_m/(κz)`` this gives ``K_m ∂u/∂z = u_*^2`` and
``K_h ∂θ/∂z = u_*θ_*`` at every height (the derivation is on the theory
page `theory/turbulence_stability.md`). The factor needs ``ζ(Ri)``,
the inverse of [`most_gradient_richardson`](@ref), found by the same
Illinois solver as [`solve_surface_layer`](@ref) on ``[ζ_{\\min}, 0]``
or ``[0, ζ_{\\max}]`` (``Ri(ζ)`` is monotone for BDBH and the log-linear
form). A stable ``Ri`` at or above ``Ri(ζ_{\\max})`` — beyond the
critical Richardson number of a log-linear family — gives ``f = 0``; an
unstable ``Ri`` below ``Ri(ζ_{\\min})`` is evaluated at ``ζ_{\\min}``.
``Pr_n = φ_h(0)/φ_m(0) = 1``.
"""
Base.@kwdef struct MOSTConsistent{S<:AbstractStabilityFunctions, T<:AbstractFloat} <: AbstractRiStability
    sf::S = BusingerDyerBeljaarsHoltslag()
    ζ_min::T = -1e3
    ζ_max::T = 1e4
end
MOSTConsistent(sf::AbstractStabilityFunctions; kw...) = MOSTConsistent(; sf = sf, kw...)

"""
    most_stability_parameter(m::MOSTConsistent, Ri) -> ζ

``ζ = z/L`` with ``Ri = ζφ_h/φ_m^2`` (see [`MOSTConsistent`](@ref));
`Inf` for a stable `Ri` at or beyond ``Ri(ζ_{\\max})``.
"""
function most_stability_parameter(m::MOSTConsistent, Ri::Real)
    T = float(typeof(Ri))
    Ri == 0 && return zero(T)
    F(ζ) = most_gradient_richardson(m.sf, ζ) - Ri
    if Ri > 0
        fb = F(T(m.ζ_max))
        fb <= 0 && return T(Inf)
        ζ, _, _ = _illinois(F, zero(T), T(m.ζ_max), T(-Ri), T(fb), 200)
        return ζ
    else
        fa = F(T(m.ζ_min))
        fa >= 0 && return T(m.ζ_min)
        ζ, _, _ = _illinois(F, T(m.ζ_min), zero(T), T(fa), T(-Ri), 200)
        return ζ
    end
end

function stability_factor_m(m::MOSTConsistent, Ri::Real)
    ζ = most_stability_parameter(m, Ri)
    isinf(ζ) && return zero(ζ)
    return 1 / phi_m(m.sf, ζ)^2
end

function stability_factor_h(m::MOSTConsistent, Ri::Real)
    ζ = most_stability_parameter(m, Ri)
    isinf(ζ) && return zero(ζ)
    return 1 / (phi_m(m.sf, ζ) * phi_h(m.sf, ζ))
end

# =========================================================================
# 3. Surface-layer solver
# =========================================================================

"""
    SurfaceLayerSolution{T}

Result of [`solve_surface_layer`](@ref). Fields:

  * `u_star`, `θ_star`, `q_star` — friction velocity (m s⁻¹) and the
    temperature (K) and humidity (kg kg⁻¹) scales, with
    ``\\overline{w'θ'} = -u_* θ_*``, ``\\overline{w'q'} = -u_* q_*``;
  * `θv_star` — the virtual-temperature scale that sets the buoyancy flux;
  * `L`, `ζ` — Obukhov length (m) and ``ζ = z_m/L``;
  * `C_D`, `C_H`, `C_E` — bulk transfer coefficients referred to the
    effective wind `U`: ``τ = ρ C_D U²``, ``H = ρ c_p C_H U (θ_s - θ_a)``,
    ``E = ρ C_E U (q_s - q_a)``;
  * `U` — the effective wind speed actually used (after the gustiness /
    minimum-wind safeguard);
  * `iterations`, `converged` — `converged = false` means the root lay
    beyond `[ζ_min, ζ_max]` and ζ was clamped there.
"""
struct SurfaceLayerSolution{T<:AbstractFloat}
    u_star::T
    θ_star::T
    q_star::T
    θv_star::T
    L::T
    ζ::T
    C_D::T
    C_H::T
    C_E::T
    U::T
    iterations::Int
    converged::Bool
end

"""
    aerodynamic_resistance(sol::SurfaceLayerSolution) -> s m⁻¹

The scalar aerodynamic resistance of the surface layer,
``r_a = 1/(C_H U)``: the heat flux is ``H = ρ c_p (θ_s − θ_a)/r_a``.
Dry deposition (`Chemistry.dry_deposition_velocity`) puts it in series
with the quasi-laminar and surface resistances.
"""
aerodynamic_resistance(sol::SurfaceLayerSolution) = 1 / (sol.C_H * sol.U)

# ε' = R_v/R_d - 1 ≈ 0.608, the virtual-temperature coefficient.
_virtual_coefficient(c::SurfaceConstants) = R_v(c.thermo) / R_d(c.thermo) - 1

"""
    bulk_richardson(z, U, θv_a, θv_s; c = DEFAULT_SURFACE_CONSTANTS) -> Ri_B

Bulk Richardson number between the surface and height `z`,

    Ri_B = g z (θv_a - θv_s) / (θv_a U²)

from virtual potential temperatures `θv_a` (at `z`) and `θv_s` (surface).
Positive is stable. With ``z_m = z_h`` and no moisture it is related to
the solver's ζ by ``Ri_B = ζ Φ_h/Φ_m²``, with Φ the stability-corrected
log terms.
"""
bulk_richardson(z::Real, U::Real, θv_a::Real, θv_s::Real;
                c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS) =
    grav(c) * z * (θv_a - θv_s) / (θv_a * U^2)

"""
    obukhov_length(u_star, θv_star, θv_ref; c = DEFAULT_SURFACE_CONSTANTS) -> L

Obukhov length ``L = u_*² θ_v / (κ g θ_{v*})``, equivalently
``-u_*³ θ_v / (κ g \\overline{w'θ_v'})``. Negative for an upward buoyancy
flux (unstable), positive for stable, `±Inf` when neutral.
"""
obukhov_length(u_star::Real, θv_star::Real, θv_ref::Real;
               c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS) =
    u_star^2 * θv_ref / (c.κ * grav(c) * θv_star)

# Stability-corrected log terms Φ = ln(z/z0) - ψ(z/L) + ψ(z0/L), written
# in terms of ζ = z_m/L so that ζ = 0 is exactly neutral.
_Φm(sf, ζ, z_m, z0m) = log(z_m / z0m) - psi_m(sf, ζ) + psi_m(sf, ζ * z0m / z_m)
_Φh(sf, ζ, z_m, z, z0) = log(z / z0) - psi_h(sf, ζ * z / z_m) + psi_h(sf, ζ * z0 / z_m)

"""
    kinematic_heat_flux(sol::SurfaceLayerSolution) -> w'θ'₀

The kinematic surface heat flux ``w'θ'_0 = -u_* θ_*`` (K m s⁻¹, upward
positive) of a Monin–Obukhov solution.
"""
kinematic_heat_flux(sol::SurfaceLayerSolution) = -sol.u_star * sol.θ_star

"""
    similarity_theta(sf, z; θ_s, θ_star, L, z0h, c = DEFAULT_SURFACE_CONSTANTS) -> θ(z)
    similarity_wind(sf, z; u_star, L, z0m, c = DEFAULT_SURFACE_CONSTANTS) -> U(z)

The Monin–Obukhov profiles below the first model level (screen-level
2 m temperature, 10 m wind) from the scales of a surface-layer solution,
with the stability-corrected log terms of [`solve_surface_layer`](@ref):

```math
θ(z) = θ_s + \\frac{θ_*}{κ}\\Big[\\ln\\frac{z}{z_{0h}} - ψ_h\\Big(\\frac{z}{L}\\Big) + ψ_h\\Big(\\frac{z_{0h}}{L}\\Big)\\Big],
\\qquad
U(z) = \\frac{u_*}{κ}\\Big[\\ln\\frac{z}{z_{0m}} - ψ_m\\Big(\\frac{z}{L}\\Big) + ψ_m\\Big(\\frac{z_{0m}}{L}\\Big)\\Big].
```

Reference: Stull (1988), *An Introduction to Boundary Layer Meteorology*, §9.7.
"""
similarity_theta(sf::AbstractStabilityFunctions, z::Real; θ_s::Real, θ_star::Real, L::Real, z0h::Real,
                 c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS) =
    θ_s + θ_star / c.κ * _Φh(sf, z / L, z, z, z0h)
"See [`similarity_theta`](@ref)."
similarity_wind(sf::AbstractStabilityFunctions, z::Real; u_star::Real, L::Real, z0m::Real,
                c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS) =
    u_star / c.κ * _Φm(sf, z / L, z, z0m)

"""
    solve_surface_layer(sf = BusingerDyerBeljaarsHoltslag();
                        U, z_m, z_h = z_m, θ_a, θ_s, q_a = 0, q_s = 0,
                        z0m, z0h, z0q = z0h,
                        c = DEFAULT_SURFACE_CONSTANTS,
                        U_min = 0.1, gustiness = 0,
                        ζ_min = -1000, ζ_max = 1000, maxiter = 100)
        -> SurfaceLayerSolution

Monin–Obukhov similarity between the surface and the lowest model level:
wind `U` at height `z_m`, potential temperature `θ_a` and specific
humidity `q_a` at `z_h`; surface (skin) values `θ_s`, `q_s`; roughness
lengths `z0m`, `z0h`, `z0q`. Solves

    u_* = κ U / Φ_m(ζ),  θ_* = κ (θ_a - θ_s) / Φ_h(ζ),  q_* = κ (q_a - q_s) / Φ_q(ζ)
    ζ   = z_m κ g θ_{v*} / (u_*² θ_{va})

with ``Φ_m = \\ln(z_m/z_{0m}) - ψ_m(z_m/L) + ψ_m(z_{0m}/L)`` (and Φ_h, Φ_q
likewise with ψ_h and `z0h`, `z0q`), buoyancy from the virtual potential
temperature ``θ_v = θ(1 + ε' q)``, ``θ_{v*} = θ_*(1 + ε' q_a) + ε' θ_a q_*``.

The single equation for ζ is solved by a bracketed secant (Illinois)
iteration, which cannot diverge the way the textbook fixed-point on
``L`` does in strongly stable conditions. The root lies in ``[ζ_min, 0]``
for unstable and ``[0, ζ_max]`` for stable stratification; if it lies
beyond, ζ is clamped and `converged = false`.

Safeguards: the effective wind is ``U_e = \\max(\\sqrt{U² + w_g²}, U_{min})``
with gustiness `w_g` (e.g. ``1.2 w_*``, Beljaars 1995), so free convection
at ``U → 0`` keeps a finite ``u_*`` instead of ``L → 0``.

`θ_a` and `θ_s` must be potential temperatures on a common reference
(e.g. ``θ_a = T_a + g z/c_p``, ``θ_s = T_s``).
"""
function solve_surface_layer(sf::AbstractStabilityFunctions = BusingerDyerBeljaarsHoltslag();
                             U::Real, z_m::Real, z_h::Real = z_m,
                             θ_a::Real, θ_s::Real, q_a::Real = 0.0, q_s::Real = 0.0,
                             z0m::Real, z0h::Real, z0q::Real = z0h,
                             c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS,
                             U_min::Real = 0.1, gustiness::Real = 0.0,
                             ζ_min::Real = -1000.0, ζ_max::Real = 1000.0,
                             maxiter::Integer = 100)
    (z_m > z0m > 0 && z_h > z0h > 0 && z_h > z0q > 0) ||
        throw(ArgumentError("solve_surface_layer: need z_m > z0m > 0 and z_h > z0h, z0q > 0"))
    T = float(promote_type(typeof(U), typeof(z_m), typeof(θ_a), typeof(θ_s),
                           typeof(q_a), typeof(q_s), typeof(z0m), typeof(z0h)))
    κ, g, ε = T(c.κ), T(grav(c)), T(_virtual_coefficient(c))
    Ue = max(sqrt(T(U)^2 + T(gustiness)^2), T(U_min))
    Δθ, Δq = T(θ_a) - T(θ_s), T(q_a) - T(q_s)
    θva = T(θ_a) * (1 + ε * T(q_a))

    # The closures' locals are declared `local`: the function assigns the
    # same names below, and without it they would be shared with it in
    # Core.Boxes — every solve then allocated (~9 KB) and ran its iteration
    # by dynamic dispatch (#537).
    scales(ζ) = begin
        local Φm = _Φm(sf, ζ, z_m, z0m)
        local Φh = _Φh(sf, ζ, z_m, z_h, z0h)
        local Φq = _Φh(sf, ζ, z_m, z_h, z0q)
        local u = κ * Ue / Φm
        local θ = κ * Δθ / Φh
        local q = κ * Δq / Φq
        local θv = θ * (1 + ε * T(q_a)) + ε * T(θ_a) * q
        (u, θ, q, θv, Φm, Φh, Φq)
    end
    # Residual F(ζ) = ζ - ζ_implied(ζ); increasing through the root.
    F(ζ) = begin
        local u, _, _, θv = scales(ζ)
        ζ - z_m * κ * g * θv / (u^2 * θva)
    end

    F0 = F(zero(T))
    ζ, iters, conv = zero(T), 0, true
    if F0 != 0
        a, b = F0 > 0 ? (T(ζ_min), zero(T)) : (zero(T), T(ζ_max))
        fa, fb = F0 > 0 ? (F(a), F0) : (F0, F(b))
        if fa >= 0 || fb <= 0            # root outside the admissible range
            ζ, conv = (fa >= 0 ? a : b), false
        else
            ζ, iters, conv = _illinois(F, a, b, fa, fb, maxiter)
        end
    end

    u, θ, q, θv, Φm, Φh, Φq = scales(ζ)
    L = ζ == 0 ? T(Inf) : T(z_m) / ζ
    return SurfaceLayerSolution{T}(u, θ, q, θv, L, ζ,
                                   κ^2 / Φm^2, κ^2 / (Φm * Φh), κ^2 / (Φm * Φq),
                                   Ue, iters, conv)
end

# Illinois (modified regula falsi) on a bracket with f(a) < 0 < f(b).
# The shared internal root solver of the code base: solve_surface_layer,
# PlantCanopy, Photosynthesis, AirSeaFlux and WallModels (the Obukhov
# length of a wall law; the dynamic roughness α of P57-e) all call this
# one, none carries a copy. Returns (x, iterations, converged).
function _illinois(f, a::T, b::T, fa::T, fb::T, maxiter::Integer) where {T}
    side = 0
    x = (a + b) / 2
    for it in 1:maxiter
        x = (a * fb - b * fa) / (fb - fa)
        (a < x < b) || (x = (a + b) / 2)
        fx = f(x)
        if fx == 0 || (b - a) <= 4 * eps(T) * max(abs(a), abs(b), one(T))
            return x, it, true
        end
        if fx > 0
            b, fb = x, fx
            side == 1 && (fa /= 2)
            side = 1
        else
            a, fa = x, fx
            side == -1 && (fb /= 2)
            side = -1
        end
        abs(fx) <= 4 * eps(T) * max(abs(x), one(T)) && return x, it, true
    end
    return x, maxiter, false
end

# =========================================================================
# 3b. Maximum sustainable heat flux (P58-e)
# =========================================================================
#
# With the wind U at height z held fixed, Monin–Obukhov similarity ties
# the downward buoyancy flux to u_* through the momentum relation alone:
# u_* = κU/Φ_m(ζ) and ζ = z/L = z κ g Q/(u_*³ θ_ref) for the downward
# kinematic flux Q = -w'θ_v' > 0. Eliminating u_*,
#
#     Q(ζ) = κ² U³ θ_ref ζ / (g z Φ_m(ζ)³),
#
# a curve that rises from 0 at neutral, peaks, and falls again: there is a
# largest downward flux the surface layer can carry at that wind (Van de
# Wiel et al. 2012). The derivation for the log-linear closed form is on
# the theory page `theory/turbulence_stability.md`.

"""
    log_linear_slope(sf::AbstractStabilityFunctions) -> α or nothing

The slope α of a *log-linear* stable branch, ``φ_m = 1 + αζ`` and
``ψ_m = -αζ`` for ``ζ ≥ 0``, or `nothing` if `sf` is not log-linear
there. [`max_sustainable_heat_flux`](@ref) uses the closed form when it
is a number. The default is `nothing`; a family with a log-linear stable
branch (e.g. [`BusingerDyerLogLinear`](@ref), with ``α = β_m``) extends it.
"""
log_linear_slope(::AbstractStabilityFunctions) = nothing

"""
    MSHFSolution{T}

The maximum of the Monin–Obukhov flux curve, from [`mshf_solution`](@ref):

  * `Q_max` — the maximum sustainable *kinematic* downward buoyancy flux
    ``-\\overline{w'θ_v'}`` (K m s⁻¹, positive); ``ρc_p Q_{max}`` is
    ``H_{max}`` in W m⁻²;
  * `u_star` — the friction velocity at which it is attained (m s⁻¹);
  * `ζ`, `L` — the stability ``ζ = z/L`` and the Obukhov length there;
  * `method` — `:closed_form` (log-linear) or `:numerical`;
  * `interior` — `false` when the maximum lay at the search bound
    `ζ_max` (a family whose flux curve does not turn over there: no
    finite MSHF up to that stability).
"""
struct MSHFSolution{T<:AbstractFloat}
    Q_max::T
    u_star::T
    ζ::T
    L::T
    method::Symbol
    interior::Bool
end

# ζ/Φ_m(ζ)³ — the shape of the flux curve Q(ζ) (Q = κ²U³θ/(g z)·this).
_mshf_shape(sf, ζ, z, z0m) = ζ / _Φm(sf, ζ, z, z0m)^3

"""
    mshf_solution(sf, U, z, z0m, θ_ref; c = DEFAULT_SURFACE_CONSTANTS,
                  method = :auto, ζ_max = 1000) -> MSHFSolution

The maximum sustainable heat flux and where it is attained; see
[`max_sustainable_heat_flux`](@ref) for the physics and the arguments.
`method = :auto` uses the closed form when [`log_linear_slope`](@ref)
returns a number and the numerical maximisation otherwise;
`:numerical` forces the maximisation (used to validate it against the
closed form); `:closed_form` requires a log-linear `sf`.

The numerical maximum of ``f(ζ) = ζ/Φ_m(ζ)^3`` is found in ``s = \\ln ζ``:
a 48-point scan of ``ζ ∈ [10^{-6}, ζ_{max}]`` brackets the largest
sample, then golden-section search narrows the bracket to
``|Δs| ≤ 10^{-12}``. One maximisation costs about 120 evaluations of
``ψ_m``.
"""
function mshf_solution(sf::AbstractStabilityFunctions, U::Real, z::Real, z0m::Real,
                       θ_ref::Real; c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS,
                       method::Symbol = :auto, ζ_max::Real = 1000.0)
    (z > z0m > 0) || throw(ArgumentError("mshf_solution: need z > z0m > 0, got z = $z, z0m = $z0m"))
    U >= 0 || throw(ArgumentError("mshf_solution: need U ≥ 0, got $U"))
    θ_ref > 0 || throw(ArgumentError("mshf_solution: need θ_ref > 0, got $θ_ref"))
    method in (:auto, :numerical, :closed_form) ||
        throw(ArgumentError("mshf_solution: method must be :auto, :numerical or :closed_form"))
    T = float(promote_type(typeof(U), typeof(z), typeof(z0m), typeof(θ_ref)))
    κ, g = T(c.κ), T(grav(c))
    Ut, zt, z0t, θt = T(U), T(z), T(z0m), T(θ_ref)
    pre = κ^2 * Ut^3 * θt / (g * zt)            # Q = pre · ζ/Φ_m³
    α = log_linear_slope(sf)
    if method === :closed_form && α === nothing
        throw(ArgumentError("mshf_solution: method = :closed_form needs a log-linear stable branch " *
                            "(log_linear_slope(sf) returned nothing)"))
    end
    if α !== nothing && method !== :numerical
        # Φ_m(ζ) = Λ + β ζ with β = α (z - z0m)/z (ψ_m(ζ) - ψ_m(ζ z0m/z) = -β ζ):
        # d/dζ [ζ/(Λ + βζ)³] = 0 at ζ* = Λ/(2β), where Φ_m = 3Λ/2, so
        # u* = 2κU/(3Λ) and Q_max = (4/27) κ² U³ θ_ref/(α g (z - z0m) Λ²).
        Λ = log(zt / z0t)
        β = T(α) * (zt - z0t) / zt
        ζs = Λ / (2β)
        Qm = 4 * κ^2 * Ut^3 * θt / (27 * T(α) * g * (zt - z0t) * Λ^2)
        return MSHFSolution{T}(Qm, 2κ * Ut / (3Λ), ζs, zt / ζs, :closed_form, true)
    end
    # numerical: maximise f(s) = e^s/Φ_m(e^s)³ in s = ln ζ
    f(s) = _mshf_shape(sf, exp(s), zt, z0t)
    smin, smax = log(T(1e-6)), log(T(ζ_max))
    n = 48
    ds = (smax - smin) / (n - 1)
    ib, fb = 1, f(smin)
    for i in 2:n
        fi = f(smin + (i - 1) * ds)
        fi > fb && ((ib, fb) = (i, fi))
    end
    interior = 1 < ib < n
    a = smin + (max(ib - 1, 1) - 1) * ds; b = smin + (min(ib + 1, n) - 1) * ds
    # golden-section search on [a, b]
    r = (sqrt(T(5)) - 1) / 2
    x1 = b - r * (b - a); x2 = a + r * (b - a)
    f1, f2 = f(x1), f(x2)
    while b - a > T(1e-12) * max(one(T), abs(a))
        if f1 >= f2
            b, x2, f2 = x2, x1, f1
            x1 = b - r * (b - a); f1 = f(x1)
        else
            a, x1, f1 = x1, x2, f2
            x2 = a + r * (b - a); f2 = f(x2)
        end
    end
    s = f1 >= f2 ? x1 : x2
    ib == n && (s = smax)                        # no turn-over below ζ_max
    ζs = exp(s)
    return MSHFSolution{T}(pre * f(s), κ * Ut / _Φm(sf, ζs, zt, z0t), ζs, zt / ζs,
                           :numerical, interior)
end

"""
    max_sustainable_heat_flux(sf, U, z, z0m, z0h, θ_ref;
                              c = DEFAULT_SURFACE_CONSTANTS, method = :auto,
                              ζ_max = 1000) -> Q_max

The **maximum sustainable heat flux** (MSHF) of Van de Wiel et al. (2012):
the largest *downward* kinematic buoyancy flux
``Q = -\\overline{w'θ_v'}`` (K m s⁻¹, positive; multiply by ``ρc_p`` for
W m⁻²) that Monin–Obukhov similarity with the stability functions `sf`
can carry when the wind speed at height `z` (m) over roughness `z0m` (m)
is `U` (m s⁻¹); `θ_ref` (K) is the reference (virtual) potential
temperature of the buoyancy parameter ``g/θ_{ref}``.

With `U` fixed, ``u_* = κU/Φ_m(ζ)`` and ``ζ = zκgQ/(u_*^3θ_{ref})``
give the flux as a function of the stability alone,

```math
Q(ζ) = \\frac{κ^2 U^3 θ_{ref}}{g z}\\,\\frac{ζ}{Φ_m(ζ)^3},
\\qquad Φ_m = \\ln\\frac{z}{z_{0m}} - ψ_m(ζ) + ψ_m\\!\\left(ζ\\frac{z_{0m}}{z}\\right),
```

which is zero at neutral, rises to a maximum and falls as the very
stable regime is approached. A surface that has to give away more than
``Q_{max}`` (radiative cooling of a skin with little heat supply from
below) cannot be in a turbulent equilibrium at that wind: the
turbulence collapses (Van de Wiel et al. 2012; Donda et al. 2015).

*Log-linear* ``φ_m = 1 + αζ`` ([`log_linear_slope`](@ref) returns α):
``Φ_m = Λ + α(z - z_{0m})ζ/z`` with ``Λ = \\ln(z/z_{0m})``, and in terms
of ``u_*``, ``Q = (κU u_*^2 - Λu_*^3)θ_{ref}/(α g (z - z_{0m}) κ)``, a
cubic whose maximum is

```math
Q_{max} = \\frac{4}{27}\\,\\frac{κ^2 U^3 θ_{ref}}{α\\, g\\,(z - z_{0m})\\,Λ^2}
\\quad\\text{at}\\quad u_* = \\frac{2κU}{3Λ},\\; ζ = \\frac{Λ\\,z}{2α(z - z_{0m})}
```

(for ``z_{0m} ≪ z`` the form of Van de Wiel et al. 2012 with ``z`` in
place of ``z - z_{0m}``). Other families are maximised numerically
([`mshf_solution`](@ref)).

`z0h` does not enter: with the wind (not the temperature difference)
held, the flux–stability relation is the momentum half of
Monin–Obukhov similarity, so ``Q_{max}`` is independent of the heat
roughness. It is accepted so that the call has the same surface
parameters as [`solve_surface_layer`](@ref). Every solution of that
solver with wind `U` (its effective wind, `sol.U`), `z = z_m` and
``θ_{ref} = θ_{va}`` lies on the curve, so its downward buoyancy flux
``u_*θ_{v*}`` is at most `Q_max`.

# References

  * Van de Wiel, B. J. H., Moene, A. F., Jonker, H. J. J., Baas, P.,
    Basu, S., Donda, J. M. M., Sun, J. & Holtslag, A. A. M. (2012). The
    minimum wind speed for sustainable turbulence in the nocturnal
    boundary layer. *J. Atmos. Sci.*, 69, 3116-3127.
  * Donda, J. M. M., van Hooijdonk, I. G. S., Moene, A. F., Jonker,
    H. J. J., van Heijst, G. J. F., Clercx, H. J. H. & van de Wiel,
    B. J. H. (2015). Collapse of turbulence in stably stratified channel
    flow: a transient phenomenon. *Q. J. R. Meteorol. Soc.*, 141,
    2137-2147.
"""
max_sustainable_heat_flux(sf::AbstractStabilityFunctions, U::Real, z::Real, z0m::Real,
                          z0h::Real, θ_ref::Real; kwargs...) =
    mshf_solution(sf, U, z, z0m, θ_ref; kwargs...).Q_max

"""
    flux_friction_velocity(sf, U, z, z0m, Q, θ_ref; c = DEFAULT_SURFACE_CONSTANTS,
                           ζ_min = -1000, ζ_max = 1000) -> u_*

The friction velocity of Monin–Obukhov similarity for a **prescribed**
upward kinematic buoyancy flux `Q` (K m s⁻¹; negative = surface cooling)
at wind `U` and height `z` over roughness `z0m`: the root of
``u_* = κU/Φ_m(ζ)`` with ``ζ = -zκgQ/(u_*^3θ_{ref})``.

  * ``Q ≥ 0`` (neutral or unstable): the unique root, ``u_* ≥ κU/Λ``
    (ζ clamped at `ζ_min`).
  * ``0 < -Q ≤ Q_{max}`` ([`mshf_solution`](@ref)): two roots; the
    *weakly stable* one, on the rising branch of the flux curve between
    neutral and the MSHF point, which is the one continuously connected
    to the neutral state.
  * ``-Q > Q_{max}``: no turbulent solution — the surface layer cannot
    carry the flux; returns `0` (collapse; Van de Wiel et al. 2012).

This is the flux-prescribed wall law of a stably stratified LES
(`WallModels.MOSTFluxWallLaw`) and the Berlin MSHF diagnostic's
definition of collapse.
"""
function flux_friction_velocity(sf::AbstractStabilityFunctions, U::Real, z::Real,
                                z0m::Real, Q::Real, θ_ref::Real;
                                c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS,
                                ζ_min::Real = -1000.0, ζ_max::Real = 1000.0)
    T = float(promote_type(typeof(U), typeof(z), typeof(z0m), typeof(Q), typeof(θ_ref)))
    κ, g = T(c.κ), T(grav(c))
    Ut, zt, z0t = T(U), T(z), T(z0m)
    Ut > 0 || return zero(T)
    Q == 0 && return κ * Ut / log(zt / z0t)
    pre = κ^2 * Ut^3 * T(θ_ref) / (g * zt)
    target = -T(Q) / pre                        # ζ/Φ_m³ at the root (sign of ζ)
    if Q > 0
        # -ζ/Φ_m³ increases without bound as ζ → -∞ (Φ_m falls): unique root
        F(ζ) = _mshf_shape(sf, ζ, zt, z0t) - target        # increasing in ζ ≤ 0
        fa, fb = F(T(ζ_min)), F(zero(T))
        fa >= 0 && return κ * Ut / _Φm(sf, T(ζ_min), zt, z0t)
        ζ, _, _ = _illinois(F, T(ζ_min), zero(T), fa, fb, 200)
        return κ * Ut / _Φm(sf, ζ, zt, z0t)
    end
    m = mshf_solution(sf, Ut, zt, z0t, T(θ_ref); c = c, ζ_max = ζ_max)
    -T(Q) > m.Q_max && return zero(T)
    G(ζ) = _mshf_shape(sf, ζ, zt, z0t) - target            # increasing on [0, ζ*]
    ga, gb = G(zero(T)), G(m.ζ)
    gb <= 0 && return m.u_star
    ζ, _, _ = _illinois(G, zero(T), m.ζ, ga, gb, 200)
    return κ * Ut / _Φm(sf, ζ, zt, z0t)
end

# =========================================================================
# 4. The surface interface
# =========================================================================

"""
    AbstractSurface

Supertype of every surface model (slab, soil column, canopy, urban
facet, snowpack, sea ice, ocean mixed layer, tile mosaic). An implementer
`S <: AbstractSurface` provides

| method                                   | returns                                  |
|:---------------------------------------- |:---------------------------------------- |
| `surface_temperature(s)`                 | skin temperature, K                      |
| `roughness_lengths(s)`                   | `(z0m = …, z0h = …, z0q = …)`, m         |
| `surface_albedo(s)`                      | broadband shortwave albedo               |
| `surface_emissivity(s)`                  | broadband longwave emissivity            |
| `step_surface!(s, forcing, dt)`          | advances `s`, returns the `SurfaceFluxes` used |

and may override the optional hooks, whose defaults describe a dry,
liquid-water surface:

| hook                                     | default                                  |
|:---------------------------------------- |:---------------------------------------- |
| `moisture_availability(s)`               | `0` (no evaporation)                     |
| `surface_specific_humidity(s, forcing)`  | ``q_a + β (q_{sat}(T_s, p) - q_a)``      |
| `surface_latent_heat(s)`                 | ``L_v(T_s)`` (snow/ice: override with ``L_s``) |

[`turbulent_fluxes`](@ref) then works for any implementer. The spectral
`surface_albedo(s, band, μ₀)` / `surface_emissivity(s, band)` of
`RadiationCore` are the *same* functions; the one-argument broadband
methods are what surface energy balances use.
"""
abstract type AbstractSurface end

"""
    has_packed_state(s) -> Bool
    packed_state_length(s) -> Int
    pack_state!(buf, i, s) -> Int
    unpack_state!(s, buf, i) -> Int

The evolving state of a surface as consecutive `Float64`s (#526): what a
checkpoint must hold to continue `s` exactly, without its constants (grid,
material properties, roughness) or its scratch (solver workspaces).
`pack_state!` writes `buf[i:i+n-1]` (`n = packed_state_length(s)`) and
`unpack_state!` reads them back into a surface built from the same
configuration; both return `i + n`. A surface without these methods has no
packed form (`has_packed_state(s) == false`) and is checkpointed another
way (`Aeolus.IO.structural_state`). Implemented by `SlabLand`,
`SoilThermalColumn` and `SurfaceMosaic`.
"""
has_packed_state(::Any) = false
function packed_state_length end
function pack_state! end
function unpack_state! end

"""
    SurfaceForcing(; sw_down = 0, lw_down = 350, T_air = 288.15, q_air = 0.008,
                     U = 5, p = 101325, z_ref = 10, rain = 0, snow = 0)

Atmospheric forcing seen by a surface over one step: downwelling
shortwave and longwave irradiance (W m⁻²), air temperature (K) and
specific humidity (kg kg⁻¹) at the reference height `z_ref` (m), wind
speed `U` (m s⁻¹) at `z_ref`, surface pressure `p` (Pa), and rain and
snow mass fluxes (kg m⁻² s⁻¹, positive downward).
"""
Base.@kwdef struct SurfaceForcing{T<:AbstractFloat}
    sw_down::T = 0.0
    lw_down::T = 350.0
    T_air::T   = 288.15
    q_air::T   = 0.008
    U::T       = 5.0
    p::T       = 101325.0
    z_ref::T   = 10.0
    rain::T    = 0.0
    snow::T    = 0.0
end
SurfaceForcing(xs::Vararg{Real,9}) = SurfaceForcing(promote(map(float, xs)...)...)

"""
    SurfaceFluxes{T}

Surface energy and momentum fluxes (W m⁻², N m⁻², kg m⁻² s⁻¹):

  * `Rn` — net radiation, positive **into** the surface;
  * `H`, `LE` — sensible and latent heat, positive **upward**;
  * `G` — heat into the substrate, positive **downward**; the default
    [`turbulent_fluxes`](@ref) sets it to the residual ``R_n - H - LE``;
  * `τ` — momentum flux magnitude ``ρ u_*²``;
  * `E` — evaporation mass flux ``-ρ u_* q_*``, positive upward
    (negative = dew / deposition).
"""
struct SurfaceFluxes{T<:AbstractFloat}
    Rn::T
    H::T
    LE::T
    G::T
    τ::T
    E::T
end

"Skin temperature (K) of a surface."
function surface_temperature end
"Roughness lengths of a surface as a named tuple `(z0m, z0h, z0q)` (m)."
function roughness_lengths end

"""
    column_heat_content(col; kwargs...) -> J m⁻²

Heat stored per unit surface area in the 1-D column below (or behind) a
surface — a wall or roof conduction column, a lake or ocean water column.
One generic owned by the contract, extended by each implementer
(`UrbanSurface.ConductionColumn`, `OceanMixedLayer.WaterColumn`), so the
name has a single binding when several surface modules are loaded together.
"""
function column_heat_content end

"""
    step_surface!(s, forcing::SurfaceForcing, dt) -> SurfaceFluxes

Advance surface `s` by `dt` seconds under `forcing` and return the
fluxes that were applied — which is what the atmosphere needs back.
"""
function step_surface! end

"""
    moisture_availability(s) -> β ∈ [0, 1]

Fraction of potential evaporation a surface delivers (the "β" of bucket
models). `0` — the default — is a dry surface, `1` open water.
"""
moisture_availability(::AbstractSurface) = 0.0

"""
    surface_specific_humidity(s, forcing; c = DEFAULT_SURFACE_CONSTANTS) -> q_s

Effective specific humidity at the surface. The default is the β form
``q_s = q_a + β\\,(q_{sat}(T_s, p) - q_a)`` with
β = [`moisture_availability`](@ref), which gives exactly
``E = β ρ C_E U (q_{sat} - q_a)``. A dry surface (β = 0) therefore has
``q_s = q_a`` and *zero* moisture flux — note that ``q_s = 0`` would
instead be a strong vapour sink. Saturation is from
`MoistThermodynamics.saturation_specific_humidity` (over liquid).
Implementers with a resistance formulation, dew, or an ice surface
override this.
"""
function surface_specific_humidity(s::AbstractSurface, f::SurfaceForcing;
                                   c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS)
    β = moisture_availability(s)
    β == 0 && return f.q_air
    q_sat = saturation_specific_humidity(surface_temperature(s), f.p, c.thermo)
    return f.q_air + β * (q_sat - f.q_air)
end

"""
    surface_latent_heat(s; c = DEFAULT_SURFACE_CONSTANTS) -> J kg⁻¹

Latent heat that converts the evaporation mass flux into `LE`. Default
``L_v(T_s)`` from `MoistThermodynamics.latent_heat_vapor`; snow and ice
surfaces override with `latent_heat_sublim`.
"""
surface_latent_heat(s::AbstractSurface; c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS) =
    latent_heat_vapor(surface_temperature(s), c.thermo)

"""
    net_radiation(s, forcing; c = DEFAULT_SURFACE_CONSTANTS) -> W m⁻²

``R_n = (1-α) SW↓ + ε LW↓ - ε σ T_s⁴``, positive into the surface
(Kirchhoff: the longwave absorptivity equals the emissivity).
"""
function net_radiation(s::AbstractSurface, f::SurfaceForcing;
                       c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS)
    α, ε, Ts = surface_albedo(s), surface_emissivity(s), surface_temperature(s)
    return (1 - α) * f.sw_down + ε * f.lw_down - ε * c.σ_SB * Ts^4
end

"""
    default_stability_functions() -> BusingerDyerBeljaarsHoltslag()

The stability functions of a surface's fluxes when none are passed
([`surface_layer_solution`](@ref), [`turbulent_fluxes`](@ref)): what a
diagnostic of the same surface layer (its 2 m temperature) must use too.
"""
default_stability_functions() = BusingerDyerBeljaarsHoltslag()

"""
    surface_layer_solution(s, forcing; sf = default_stability_functions(),
                           c = DEFAULT_SURFACE_CONSTANTS, kwargs...) -> SurfaceLayerSolution

The Monin–Obukhov solution ([`solve_surface_layer`](@ref)) between the
skin of surface `s` (`surface_temperature`, `surface_specific_humidity`,
`roughness_lengths`) and the forcing level of `forcing`, the air
temperature referred to the surface by the dry adiabat, ``θ_a = T_a +
Γ_d z_{ref}``, ``θ_s = T_s``: the one solve of [`turbulent_fluxes`](@ref)
(its ``u_*``, ``θ_*``, ``L``, on that surface-referenced θ scale), and of
the screen-level diagnostics of a tile (the 2 m temperature of a tiled
surface, #536). `kwargs` (`U_min`, `gustiness`, …) go to
`solve_surface_layer`.
"""
function surface_layer_solution(s::AbstractSurface, f::SurfaceForcing;
                                sf::AbstractStabilityFunctions = default_stability_functions(),
                                c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS, kwargs...)
    Ts = surface_temperature(s)
    z0 = roughness_lengths(s)
    qa = PhasePartition(f.q_air)
    θ_a = f.T_air + dry_adiabatic_lapse_rate(qa, c.thermo) * f.z_ref
    q_s = surface_specific_humidity(s, f; c = c)
    return solve_surface_layer(sf; U = f.U, z_m = f.z_ref, θ_a = θ_a, θ_s = Ts,
                               q_a = f.q_air, q_s = q_s,
                               z0m = z0.z0m, z0h = z0.z0h, z0q = z0.z0q, c = c, kwargs...)
end

"""
    turbulent_fluxes(s, forcing; sf = default_stability_functions(),
                     c = DEFAULT_SURFACE_CONSTANTS, kwargs...) -> SurfaceFluxes

Default flux computation for any [`AbstractSurface`](@ref): solves the
surface layer between the skin (`surface_temperature`,
`surface_specific_humidity`, `roughness_lengths`) and the forcing level
([`surface_layer_solution`](@ref), which takes `sf` and the `kwargs`),
then

    τ = ρ u_*²,   H = -ρ c_pm u_* θ_*,   E = -ρ u_* q_*,   LE = L E

with ρ and ``c_{pm}`` of the moist air from `MoistThermodynamics`, and
the air temperature referred to the surface by the dry adiabat,
``θ_a = T_a + Γ_d z_{ref}``, ``θ_s = T_s``. `Rn` is
[`net_radiation`](@ref) and `G` the residual ``R_n - H - LE``. `kwargs`
(`U_min`, `gustiness`, …) are passed to [`solve_surface_layer`](@ref).
"""
function turbulent_fluxes(s::AbstractSurface, f::SurfaceForcing;
                          c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS, kwargs...)
    qa = PhasePartition(f.q_air)
    sol = surface_layer_solution(s, f; c = c, kwargs...)
    ρ = air_density(f.T_air, f.p, qa, c.thermo)
    cp = heat_capacity_moist(qa, c.thermo)
    H = -ρ * cp * sol.u_star * sol.θ_star
    E = -ρ * sol.u_star * sol.q_star
    LE = surface_latent_heat(s; c = c) * E
    Rn = net_radiation(s, f; c = c)
    return SurfaceFluxes(promote(Rn, H, LE, Rn - H - LE, ρ * sol.u_star^2, E)...)
end

"""
    surface_mshf_ratio(sf, fl::SurfaceFluxes, forcing::SurfaceForcing, z0m;
                       U_min = 0.1, gustiness = 0, Q_max = nothing,
                       c = DEFAULT_SURFACE_CONSTANTS)
        -> (ratio, Q_down, Q_max, H_max, ζ_mshf)

How close the surface fluxes `fl` of a column are to its maximum
sustainable heat flux (P58-e, #193). With the air state of `forcing`
converted exactly as in [`turbulent_fluxes`](@ref) — ``ρ``, ``c_{pm}``,
``θ_a = T_a + Γ_d z_{ref}``, ``θ_{va} = θ_a(1 + ε' q_a)`` — the downward
kinematic buoyancy flux is

```math
Q_{down} = -\\overline{w'θ_v'} = -\\frac{H}{ρ c_{pm}}(1 + ε' q_a) - ε' θ_a \\frac{E}{ρ}
\\;(= u_* θ_{v*}),
```

and ``Q_{max}`` = [`max_sustainable_heat_flux`](@ref)`(sf, U_e, z_ref, z0m,
z0m, θ_va)` at the effective wind ``U_e = \\max(\\sqrt{U^2 + w_g^2},
U_{min})`` of [`solve_surface_layer`](@ref). Returns `ratio = Q_down/Q_max`
(negative for an upward, unstable flux), both fluxes, and
``H_{max} = ρ c_{pm} Q_{max}`` (W m⁻²). For fluxes that
`turbulent_fluxes` computed with the same `sf`, `U_min` and `gustiness`
the ratio is at most 1 (the solver's solutions lie on the MSHF curve);
`ζ_mshf` is the stability ``z_{ref}/L`` at which the curve peaks
(`NaN` when `Q_max` is passed in): a column whose ``z_{ref}/L`` exceeds it
is on the *very stable* branch, beyond the MSHF point, where a stronger
temperature contrast carries less heat — the collapsed regime;
the moisture term makes it the buoyancy flux, which is what MOST
constrains, rather than the sensible heat flux alone. `Q_max` (K m s⁻¹)
skips the maximisation and uses a value already computed for the same
column and forcing (the Berlin coupling compares two sets of fluxes of
one column against one MSHF).
"""
function surface_mshf_ratio(sf::AbstractStabilityFunctions, fl::SurfaceFluxes,
                            f::SurfaceForcing, z0m::Real; U_min::Real = 0.1,
                            gustiness::Real = 0.0, Q_max = nothing,
                            c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS)
    qa = PhasePartition(f.q_air)
    ε = _virtual_coefficient(c)
    θ_a = f.T_air + dry_adiabatic_lapse_rate(qa, c.thermo) * f.z_ref
    θva = θ_a * (1 + ε * f.q_air)
    ρ = air_density(f.T_air, f.p, qa, c.thermo)
    cp = heat_capacity_moist(qa, c.thermo)
    Qdown = -(fl.H / (ρ * cp)) * (1 + ε * f.q_air) - ε * θ_a * fl.E / ρ
    Ue = max(sqrt(f.U^2 + gustiness^2), U_min)
    if Q_max === nothing
        m = mshf_solution(sf, Ue, f.z_ref, z0m, θva; c = c)
        Qmax, ζs = m.Q_max, m.ζ
    else
        Qmax, ζs = oftype(Qdown, Q_max), oftype(Qdown, NaN)
    end
    return (ratio = Qdown / Qmax, Q_down = Qdown, Q_max = Qmax, H_max = ρ * cp * Qmax,
            ζ_mshf = ζs)
end

# =========================================================================
# The surface fluxes into the first air cell (ONE-e, #454: from the urban
# case's surface exchange)
# =========================================================================

"""
    deliver_surface_fluxes!(θ, qv, H, E, k_air, surfaces, cp, ρ̄, Π̄, dz, τ;
                            solid, offsets) -> (evaporated, solid_writes)

Hand the sensible heat flux `H` (W m⁻²) and the evaporation `E`
(kg m⁻² s⁻¹) of every surface column to its first air cell for `τ`
seconds: at `(i, j, k_air[i, j])` of the interior arrays `θ` and `qv`,

```math
Δθ = \\frac{H\\,τ}{Δz}\\,\\frac{1}{ρ̄_k\\, c_p\\, Π̄_k}, \\qquad
Δq_v = \\frac{E\\,τ}{ρ̄_k\\,Δz},
```

([`θ_per_heat`](@ref Aeolus.MoistThermodynamics.θ_per_heat) with the
column's moist heat capacity `cp[i, j]`), so that the heat and the water
the air receives are exactly `H τ` and `E τ` per unit area. Columns whose
`surfaces[i, j] === nothing` have no surface and are skipped. `ρ̄`, `Π̄`
are the reference density and Exner function per level, `dz` the cell
height. `solid` is the solid mask of the storage array (offset by the
halos `offsets = (Hx, Hy, Hz)`); a delivery into a solid cell is counted
as a `solid_write` (a geometry error). Returns the evaporated water
`Σ E τ` (kg m⁻², summed over the columns in column-major order — the
canonical order of a 2-D sum, thread-invariant) and the number of solid
writes.
"""
function deliver_surface_fluxes!(θ, qv, H, E, k_air, surfaces, cp, ρ̄, Π̄, dz, τ; solid, offsets)
    Nx, Ny = size(k_air)
    Hx, Hy, Hz = offsets
    return column_sum(backend(E), _deliver_column!, (1:Nx, 1:Ny),
                      θ, qv, H, E, k_air, surfaces, cp, ρ̄, Π̄, solid, Hx, Hy, Hz, dz, τ)
end

@inline function _deliver_column!(i, j, θ, qv, H, E, k_air, surfaces, cp, ρ̄, Π̄, mask, Hx, Hy, Hz, dz, τ)
    @inbounds begin
        surfaces[i, j] === nothing && return (0.0, 0.0)
        k = k_air[i, j]
        solid = mask[Hx+i, Hy+j, Hz+k]
        ρk = ρ̄[k]
        θ[i, j, k] += H[i, j] * τ / dz * θ_per_heat(ρ̄[k], cp[i, j], Π̄[k])
        qv[i, j, k] += E[i, j] * τ / (ρk * dz)
        return (E[i, j] * τ, solid ? 1.0 : 0.0)
    end
end

end # module
