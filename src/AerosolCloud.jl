"""
    AerosolCloud

Aerosol-cloud interaction — Phase 42.

The two aerosol indirect effects, computed rather than prescribed.

WHY THIS MODULE EXISTS

A bulk microphysics scheme that carries a *fixed* cloud droplet number
concentration cannot represent an aerosol indirect effect at all: the
entire signal — Twomey brightening, Albrecht lifetime lengthening — is
carried by the dependence of `N_d` on the aerosol population and on the
dynamics that lift it. So `N_d` here is a *diagnosed* quantity, obtained
from the aerosol size distribution, its hygroscopicity, and the
updraught velocity, through activation physics.

THE CHAIN

    log-normal dry modes (N, r_g, σ_g, κ)
              │
              ├─ κ-Köhler  ──→  s_crit(r_d, κ),  r_crit(r_d, κ)
              │                        │
    updraught w ──→ α w  ─────────────┤
              │                        │
    T, p ──→ α, γ, G  ────────────────┤
                                       ↓
              ┌────────────────────────┴───────────────────────┐
              │                                                │
     Abdul-Razzak & Ghan (2000)                    explicit adiabatic parcel
     algebraic S_max, erfc cutoff                  dS/dt = αw − γ dχ/dt
              │                                     bin-resolved r_j(t)
              └────────────────────────┬───────────────────────┘
                                       ↓
                                  N_d,  r_eff
                                       │
                        ┌──────────────┴──────────────┐
                   Twomey (τ, albedo)          Albrecht (autoconversion)

The parcel model is the *reference*. The parameterisation is the
workhorse. Both are here, and the test suite measures the gap between
them instead of asserting it.

WHAT IS AND IS NOT RE-DERIVED HERE

Nothing thermodynamic is re-derived: saturation vapour pressure, latent
heat, air density and heat capacity all come from `MoistThermodynamics`
(Phase 39). Nothing hygroscopic is re-derived either: the κ-Köhler
equilibrium saturation ratio is `AerosolOptics.kohler_saturation`
(Phase 37), and the Kelvin coefficient is scaled from the exported
`AerosolOptics.KELVIN_A_298` so there is exactly one value of
`2σ_w M_w / (R T ρ_w)` in the package. The one analytic expression that
*is* written out again is `∂S/∂r` of the Köhler curve — a derivative has
to be written somewhere — and the test suite checks it against a
finite difference of `kohler_saturation` to guard the duplication.

THE THREE COEFFICIENTS THAT MATTER

    α = g M_w L / (c_p R T²) − g M_a / (R T)          [m⁻¹]
    γ = R T / (e_s M_w) + M_w L² / (c_p p M_a T)      [m³ kg⁻¹]
    G = [ ρ_w R T/(e_s D_v M_w) + L ρ_w/(K_a T)·(L M_w/(R T) − 1) ]⁻¹  [m² s⁻¹]

`α` is adiabatic cooling expressed as a supersaturation source, `γ`
converts condensed mass into a supersaturation sink, and `G` is the
condensational growth coefficient in `r dr/dt = G(S − S_eq)`. Note that
`γ` as written multiplies the tendency of the liquid water *content*
(kg m⁻³), not the mixing ratio — the mixing-ratio form is `γ ρ_a`. That
factor of `ρ_a` is a classic and silent source of ~20 % errors in
`S_max`, so it is stated explicitly and checked in the tests.

REFERENCES

  * Köhler, H. (1936). The nucleus in and the growth of hygroscopic
    droplets. *Trans. Faraday Soc.*, 32, 1152-1161.
  * Petters, M. D. & Kreidenweis, S. M. (2007). A single parameter
    representation of hygroscopic growth and CCN activity.
    *Atmos. Chem. Phys.*, 7, 1961-1971.
  * Abdul-Razzak, H., Ghan, S. J. & Rivera-Carpio, C. (1998). A
    parameterization of aerosol activation 1. Single aerosol type.
    *J. Geophys. Res.*, 103, 6123-6131.
  * Abdul-Razzak, H. & Ghan, S. J. (2000). A parameterization of aerosol
    activation 2. Multiple aerosol types. *J. Geophys. Res.*, 105,
    6837-6844.
  * Twomey, S. (1974). Pollution and the planetary albedo.
    *Atmos. Environ.*, 8, 1251-1256.
  * Twomey, S. (1991). Aerosols, clouds and radiation.
    *Atmos. Environ.*, 25A, 2435-2442.
  * Albrecht, B. A. (1989). Aerosols, cloud microphysics, and fractional
    cloudiness. *Science*, 245, 1227-1230.
  * Khairoutdinov, M. & Kogan, Y. (2000). A new cloud physics
    parameterization in a large-eddy simulation model of marine
    stratocumulus. *Mon. Wea. Rev.*, 128, 229-243.
  * Martin, G. M., Johnson, D. W. & Spice, A. (1994). The measurement and
    parameterization of effective radius of droplets in warm
    stratocumulus clouds. *J. Atmos. Sci.*, 51, 1823-1842.
  * Pruppacher, H. R. & Klett, J. D. (1997). *Microphysics of Clouds and
    Precipitation*, 2nd ed. Ch. 6 and 13.
  * Nenes, A. & Seinfeld, J. H. (2003). Parameterization of cloud droplet
    formation in global climate models. *J. Geophys. Res.*, 108, 4415.
"""
module AerosolCloud

export # --- physical constants of the module ---------------------------
       RHO_WATER, KAPPA_MIN,
       # --- Köhler theory -------------------------------------------------
       kelvin_A, kohler_equilibrium_supersaturation, kohler_slope,
       kohler_critical, critical_supersaturation, critical_radius,
       equilibrium_radius, kohler_stable,
       # --- activation coefficients ----------------------------------------
       ActivationCoefficients, activation_coefficients,
       water_vapor_diffusivity, thermal_conductivity_air,
       growth_coefficient, supersaturation_source, supersaturation_sink,
       # --- aerosol modes ----------------------------------------------------
       ARGMode, arg_mode, arg_modes, mode_critical_supersaturation,
       number_above_radius, mass_above_radius, mode_dry_mass,
       # --- Abdul-Razzak & Ghan activation -----------------------------------
       ActivationResult, arg_activation, arg_maximum_supersaturation,
       activated_fraction, ccn_fraction, ccn_spectrum,
       # --- explicit adiabatic parcel ----------------------------------------
       ParcelSettings, ParcelResult, parcel_activation,
       # --- prognostic aerosol -----------------------------------------------
       AerosolReservoir, aerosol_reservoir, interstitial_modes,
       activate!, regenerate!, process_collisions!,
       interstitial_number, activated_number, aerosol_total_number,
       aerosol_total_mass, reservoir_dry_radius,
       # --- cloud droplet and indirect-effect diagnostics ---------------------
       cloud_droplet_number, droplet_volume_radius, droplet_effective_radius,
       droplet_number_from_effective_radius, cloud_water_path,
       cloud_optical_depth_lwp, cloud_albedo,
       activation_susceptibility, twomey_radius_exponent,
       twomey_albedo_change, albedo_susceptibility,
       autoconversion_rate, autoconversion_suppression,
       albrecht_lifetime_factor

# The MoistThermodynamics shared core (Phase 39) for thermodynamics and Köhler theory; Phase 37
# for the aerosol size-distribution types, so this module does not carry a
# second copy of either.
using ..MoistThermodynamics
using ..AerosolOptics
using ..GasOptics: erf_series, erfc_series
import ..AerosolOptics: kohler_saturation, LogNormalMode, AerosolMode

# =========================================================================
# 0. Constants
# =========================================================================

"Density of liquid water, kg m⁻³. Matches the value inside `AerosolOptics`."
const RHO_WATER = 1000.0

"""
Hygroscopicity below which a particle is treated as non-CCN-active.

`AerosolOptics.wet_radius` returns the dry radius unchanged for
`κ ≤ 1e-8`; the same threshold is used here, so a black-carbon mode
(`κ = 0`) neither takes up water nor activates in either module.
"""
const KAPPA_MIN = 1.0e-8

# Reference temperature at which `AerosolOptics.KELVIN_A_298` is defined.
const _T_KELVIN_REF = 298.15

# =========================================================================
# 1. Köhler theory
# =========================================================================

"""
    kelvin_A(T_K) -> m

Kelvin (curvature) coefficient

```math
A = \\frac{2 σ_w M_w}{R T ρ_w}
```

in **metres**, ≈ 1.1 nm at cloud-base temperatures. It enters the Köhler
curve as `exp(A/r)` and, being ∝ 1/T, is the only explicitly temperature
dependent part of the equilibrium curve.

The value is scaled from the exported `AerosolOptics.KELVIN_A_298` (which
is in μm at 298.15 K) so that the surface tension, molar mass and water
density behind it exist in exactly one place in the package. `A ∝ 1/T`
exactly, so the scaling is not an approximation.
"""
# Delegates to the Phase 39 contract so there is ONE Kelvin coefficient in
# the package. It previously scaled `AerosolOptics.KELVIN_A_298`, which is
# built on a slightly different surface tension than the contract's
# Pruppacher & Klett fit; the two differed by 0.8 %, which becomes 1.2 % in
# s_crit because s_crit ~ A^{3/2}. Small, but it is exactly the kind of
# quiet divergence that makes two modules disagree about whether a given
# particle activates.
kelvin_A(T_K::Real) = kelvin_coefficient(float(T_K))


"""
    kohler_equilibrium_supersaturation(r_wet, r_dry, κ; T_K = 298.15) -> s

Equilibrium **supersaturation** `s = S − 1` over a solution droplet of
wet radius `r_wet` grown on a dry particle of radius `r_dry` (both in
**metres**), in the κ-Köhler representation.

This delegates to [`AerosolOptics.kohler_saturation`](@ref) — there is
one κ-Köhler implementation in the package and this is not it. The only
differences are the unit (metres here, μm there) and the offset by one,
both of which exist because activation physics is written in terms of a
small supersaturation rather than a saturation ratio near unity.

The curve rises from `s = −1` at `r_wet → r_dry⁺`, peaks at
[`critical_radius`](@ref), and decays back to zero from above. Below the
peak an equilibrium is *stable* (a perturbed droplet is restored);
above it the droplet grows without bound and is said to have activated.
"""
kohler_equilibrium_supersaturation(r_wet::Real, r_dry::Real, κ::Real;
                                   T_K::Real = _T_KELVIN_REF) =
    kohler_saturation(float(r_wet) * 1e6, float(r_dry) * 1e6, κ; T_K = T_K) - 1

"""
    kohler_slope(r_wet, r_dry, κ; T_K = 298.15) -> ∂S/∂r   [m⁻¹]

Analytic radial derivative of the κ-Köhler saturation ratio.

With `u = r³`, `a = r_d³`, `b = a(1−κ)` and `f = (u−a)/(u−b)`:

```math
\\frac{∂S}{∂r} = e^{A/r}\\left[ 3r^2 \\frac{κ a}{(u-b)^2}
                 - f\\,\\frac{A}{r^2} \\right]
```

This is the *one* place where the κ-Köhler expression is differentiated
rather than called, because the exponential integrator used by the
parcel model needs the local relaxation rate `λ = 2G ∂s_eq/∂(r²)` to
step the haze population stably. The test suite verifies it against a
centred finite difference of `AerosolOptics.kohler_saturation`.

The sign is the branch indicator: `> 0` on the stable (haze) branch,
`< 0` beyond the critical radius.
"""
function kohler_slope(r_wet::Real, r_dry::Real, κ::Real;
                      T_K::Real = _T_KELVIN_REF)
    r = float(r_wet); rd = float(r_dry); kap = float(κ)
    r > rd || return 0.0
    A = kelvin_A(T_K)
    u = r^3; a = rd^3; b = a * (1 - kap)
    f = (u - a) / (u - b)
    return exp(A / r) * (3 * r^2 * kap * a / (u - b)^2 - f * A / (r * r))
end

"""
    critical_supersaturation(r_dry, κ; T_K = 298.15) -> s_crit

Critical (peak) supersaturation of the Köhler curve, in the dilute
limit where `r_crit ≫ r_dry`:

```math
s_\\mathrm{crit} = \\sqrt{\\frac{4A^3}{27\\,κ\\,r_d^3}}
```

obtained by maximising `s ≈ A/r − κ r_d³/r³`. It falls as `r_d^{-3/2}`
and as `κ^{-1/2}`: big particles and soluble particles activate first,
which is the whole reason a polluted cloud has more but smaller
droplets than a clean one.

For `κ ≤ KAPPA_MIN` the solute term vanishes and the curve is pure
Kelvin, monotonically decreasing from its value at `r = r_d`; the peak
then sits at the dry radius and `s_crit = exp(A/r_d) − 1`. That limit is
returned instead, which is continuous with the exact numerical result
from [`kohler_critical`](@ref).

Accuracy of the dilute form: the neglected term is `O((r_d/r_c)^3)`, so
the error is well under 0.1 % for accumulation-mode particles and rises
to a few per cent for `r_d ≲ 20 nm`. Use [`kohler_critical`](@ref) when
that matters.
"""
function critical_supersaturation(r_dry::Real, κ::Real;
                                  T_K::Real = _T_KELVIN_REF)
    rd = float(r_dry)
    rd > 0 || throw(DomainError(r_dry, "critical_supersaturation: r_dry must be > 0"))
    A = kelvin_A(T_K)
    kap = float(κ)
    kap <= KAPPA_MIN && return expm1(A / rd)
    return sqrt(4 * A^3 / (27 * kap * rd^3))
end

"""
    critical_radius(r_dry, κ; T_K = 298.15) -> m

Wet radius at the Köhler peak, `r_crit = sqrt(3 κ r_d³ / A)`, in the
dilute limit. A 100 nm sulphate particle has `r_crit ≈ 1.3 μm`: the
droplet must grow by more than an order of magnitude in radius — three
in volume — before it is free of the barrier.

For `κ ≤ KAPPA_MIN` the curve has no interior maximum and `r_dry` is
returned.
"""
function critical_radius(r_dry::Real, κ::Real; T_K::Real = _T_KELVIN_REF)
    rd = float(r_dry)
    rd > 0 || throw(DomainError(r_dry, "critical_radius: r_dry must be > 0"))
    kap = float(κ)
    kap <= KAPPA_MIN && return rd
    return sqrt(3 * kap * rd^3 / kelvin_A(T_K))
end

"""
    kohler_critical(r_dry, κ; T_K = 298.15) -> (r_crit = …, s_crit = …)

Critical radius and critical supersaturation found by **numerically
maximising** `AerosolOptics.kohler_saturation`, with no dilute
approximation.

A golden-section search on `ln r` is used: the Köhler curve is unimodal
on `(r_d, ∞)` for every `κ > 0`, so the search is globally convergent
without a derivative.

This is the reference against which the closed forms
[`critical_supersaturation`](@ref) and [`critical_radius`](@ref) are
checked, and it is what defines "activated" in the parcel model.
"""
function kohler_critical(r_dry::Real, κ::Real; T_K::Real = _T_KELVIN_REF,
                         tol::Real = 1e-12)
    rd = float(r_dry)
    rd > 0 || throw(DomainError(r_dry, "kohler_critical: r_dry must be > 0"))
    kap = float(κ)
    if kap <= KAPPA_MIN
        return (r_crit = rd, s_crit = expm1(kelvin_A(T_K) / rd))
    end
    f(lr) = kohler_saturation(exp(lr) * 1e6, rd * 1e6, kap; T_K = T_K)
    # Bracket: the dilute estimate is always within a factor of a few.
    r0 = critical_radius(rd, kap; T_K = T_K)
    lo = log(max(rd * (1 + 1e-10), r0 * 1e-3))
    hi = log(max(r0 * 1e3, rd * 10))
    invφ = (sqrt(5.0) - 1) / 2
    c = hi - invφ * (hi - lo)
    d = lo + invφ * (hi - lo)
    fc = f(c); fd = f(d)
    for _ in 1:400
        if fc > fd
            hi = d; d = c; fd = fc
            c = hi - invφ * (hi - lo); fc = f(c)
        else
            lo = c; c = d; fc = fd
            d = lo + invφ * (hi - lo); fd = f(d)
        end
        (hi - lo) <= tol * max(1.0, abs(hi)) && break
    end
    lr = fc > fd ? c : d
    return (r_crit = exp(lr), s_crit = f(lr) - 1)
end

"""
    kohler_stable(r_wet, r_dry, κ; T_K = 298.15) -> Bool

`true` when the droplet sits on the stable (rising) branch of the Köhler
curve, i.e. `∂S/∂r > 0`. Equivalent to `r_wet < r_crit`, but evaluated
from the exact slope rather than the dilute critical radius.
"""
kohler_stable(r_wet::Real, r_dry::Real, κ::Real; T_K::Real = _T_KELVIN_REF) =
    kohler_slope(r_wet, r_dry, κ; T_K = T_K) > 0

"""
    equilibrium_radius(r_dry, s, κ; T_K = 298.15) -> m

Wet radius of the *stable* equilibrium at ambient supersaturation `s`,
i.e. the smaller root of `s_eq(r) = s`. Returns `NaN` if `s` exceeds the
critical supersaturation, because no equilibrium exists then — the
particle is activated and grows without bound.

Bisection between `r_dry` and the (numerically exact) critical radius,
on which branch `s_eq` is strictly increasing.

For `κ ≤ KAPPA_MIN` the dry radius is returned, matching
`AerosolOptics.wet_radius`: a hydrophobic particle takes up no water.
"""
function equilibrium_radius(r_dry::Real, s::Real, κ::Real;
                            T_K::Real = _T_KELVIN_REF)
    rd = float(r_dry)
    kap = float(κ)
    kap <= KAPPA_MIN && return rd
    crit = kohler_critical(rd, kap; T_K = T_K)
    ss = float(s)
    ss >= crit.s_crit && return NaN
    target = 1 + ss
    lo = rd * (1 + 1e-12)
    hi = crit.r_crit
    for _ in 1:300
        mid = sqrt(lo * hi)                      # geometric bisection
        (mid <= lo || mid >= hi) && break
        if kohler_saturation(mid * 1e6, rd * 1e6, kap; T_K = T_K) < target
            lo = mid
        else
            hi = mid
        end
        (hi - lo) <= 1e-14 * hi && break
    end
    return 0.5 * (lo + hi)
end

# =========================================================================
# 2. Thermodynamic and kinetic coefficients
# =========================================================================

"""
    water_vapor_diffusivity(T_K, p) -> m² s⁻¹

Diffusivity of water vapour in air (Hall & Pruppacher 1976):

```math
D_v = 2.11\\times10^{-5}\\,(T/T_0)^{1.94}\\,(p_0/p)
```

with `T_0 = 273.15 K`, `p_0 = 101325 Pa`. ≈ 2.2 × 10⁻⁵ m² s⁻¹ near
cloud base. The `1/p` dependence is why droplet growth is faster — and
supersaturations therefore lower — at altitude for the same updraught.
"""
water_vapor_diffusivity(T_K::Real, p::Real) =
    2.11e-5 * (float(T_K) / 273.15)^1.94 * (101325.0 / float(p))

"""
    thermal_conductivity_air(T_K) -> W m⁻¹ K⁻¹

Thermal conductivity of moist air, `K_a = 10⁻³(4.39 + 0.071 T)`
(Beard & Pruppacher 1971). ≈ 0.0245 W m⁻¹ K⁻¹ at 283 K.
"""
thermal_conductivity_air(T_K::Real) = 1.0e-3 * (4.39 + 0.071 * float(T_K))

"""
    growth_coefficient(T_K, p; constants, r = Inf, α_c = 1.0, α_T = 1.0) -> m² s⁻¹

Condensational growth coefficient `G` in

```math
r\\,\\frac{dr}{dt} = G\\,(S - S_\\mathrm{eq})
```

combining vapour diffusion to the droplet and conduction of latent heat
away from it (Pruppacher & Klett eq. 13-28):

```math
G^{-1} = \\frac{ρ_w R T}{e_s D_v M_w}
       + \\frac{L ρ_w}{K_a T}\\left(\\frac{L M_w}{R T} - 1\\right)
```

`G ≈ 1 × 10⁻¹⁰ m² s⁻¹ = 100 μm² s⁻¹` near cloud base, with the two terms
of comparable size — neglecting either is a factor-of-two error.

**Kinetic (non-continuum) corrections.** When a finite droplet radius
`r` is supplied, `D_v` and `K_a` are reduced by the Fuchs-Sutugin type
factors

```math
D_v' = \\frac{D_v}{1 + \\frac{D_v}{α_c r}\\sqrt{2π M_w/(RT)}},\\qquad
K_a' = \\frac{K_a}{1 + \\frac{K_a}{α_T r ρ_a c_p}\\sqrt{2π M_a/(RT)}}
```

which matter below ~1 μm and therefore *during* activation. They are off
by default (`r = Inf`) because the Abdul-Razzak & Ghan parameterisation
is derived in the continuum limit, and mixing the two would confuse
parameterisation error with kinetic-limitation physics in the
validation. Supply `r` (and, realistically, `α_c ≈ 0.06`-`1.0`) to turn
them on.
"""
function growth_coefficient(T_K::Real, p::Real;
                            constants::ThermodynamicConstants = DEFAULT_CONSTANTS,
                            r::Real = Inf, α_c::Real = 1.0, α_T::Real = 1.0,
                            q::PhasePartition = PhasePartition(0.0))
    Tk = float(T_K); pp = float(p)
    c = constants
    R = c.gas_constant
    Mw = c.molmass_water
    Ma = c.molmass_dry
    L = latent_heat_vapor(Tk, c)
    e_s = saturation_vapor_pressure(Tk, c)
    Dv = water_vapor_diffusivity(Tk, pp)
    Ka = thermal_conductivity_air(Tk)
    if isfinite(r) && r > 0
        ρ_a = air_density(Tk, pp, q, c)
        cp = heat_capacity_moist(q, c)
        Dv = Dv / (1 + Dv / (α_c * r) * sqrt(2π * Mw / (R * Tk)))
        Ka = Ka / (1 + Ka / (α_T * r * ρ_a * cp) * sqrt(2π * Ma / (R * Tk)))
    end
    term_diff = RHO_WATER * R * Tk / (e_s * Dv * Mw)
    term_heat = L * RHO_WATER / (Ka * Tk) * (L * Mw / (R * Tk) - 1)
    return 1 / (term_diff + term_heat)
end

"""
    supersaturation_source(T_K, p; q, constants) -> α   [m⁻¹]

Adiabatic supersaturation production per unit ascent,

```math
α = \\frac{g M_w L}{c_p R T^2} - \\frac{g M_a}{R T}
```

so that `dS/dt|_dyn = α w`. The first term is the cooling-driven fall in
`e_s`; the second is the fall in the ambient vapour pressure with
pressure, which *opposes* it. The difference is ≈ 5 × 10⁻⁴ m⁻¹ at 283 K
and 900 hPa — an updraught of 1 m s⁻¹ would drive the supersaturation up
at 0.05 % per second if nothing condensed.
"""
function supersaturation_source(T_K::Real, p::Real;
                                q::PhasePartition = PhasePartition(0.0),
                                constants::ThermodynamicConstants = DEFAULT_CONSTANTS)
    c = constants
    Tk = float(T_K)
    L = latent_heat_vapor(Tk, c)
    cp = heat_capacity_moist(q, c)
    R = c.gas_constant
    return c.grav * c.molmass_water * L / (cp * R * Tk^2) -
           c.grav * c.molmass_dry / (R * Tk)
end

"""
    supersaturation_sink(T_K, p; q, constants) -> γ   [m³ kg⁻¹]

Supersaturation depletion per unit condensed water,

```math
γ = \\frac{R T}{e_s M_w} + \\frac{M_w L^2}{c_p\\,p\\,M_a\\,T}
```

so that `dS/dt|_cond = −γ dχ/dt` with `χ` the liquid water **content**
in kg m⁻³. The first term is the direct removal of vapour; the second is
the reduction of the supersaturation by latent heating.

**Units warning.** The form above is Abdul-Razzak & Ghan's, and it
multiplies a mass *concentration* tendency. The equivalent coefficient
for the liquid water *mixing ratio* (kg kg⁻¹) is `γ ρ_a`, i.e.
`1/w_s + L²/(R_v c_p T²)`. The two differ by ρ_a ≈ 1.1, which is small
enough to hide and large enough to matter.
"""
function supersaturation_sink(T_K::Real, p::Real;
                              q::PhasePartition = PhasePartition(0.0),
                              constants::ThermodynamicConstants = DEFAULT_CONSTANTS)
    c = constants
    Tk = float(T_K); pp = float(p)
    L = latent_heat_vapor(Tk, c)
    cp = heat_capacity_moist(q, c)
    e_s = saturation_vapor_pressure(Tk, c)
    R = c.gas_constant
    return R * Tk / (e_s * c.molmass_water) +
           c.molmass_water * L^2 / (cp * pp * c.molmass_dry * Tk)
end

"""
    ActivationCoefficients{T}

Everything the activation physics needs about the ambient state,
evaluated once so that the parcel integrator and the parameterisation
provably use identical numbers.

  * `T_K`, `p`  — temperature (K) and pressure (Pa)
  * `e_s`       — saturation vapour pressure over liquid, Pa
  * `L`, `c_p`  — latent heat (J kg⁻¹) and moist heat capacity (J kg⁻¹ K⁻¹)
  * `ρ_a`       — air density, kg m⁻³
  * `A`         — Kelvin coefficient, m
  * `α`         — supersaturation source, m⁻¹
  * `γ`         — supersaturation sink, m³ kg⁻¹
  * `G`         — growth coefficient, m² s⁻¹
"""
struct ActivationCoefficients{T<:AbstractFloat}
    T_K::T
    p::T
    e_s::T
    L::T
    c_p::T
    ρ_a::T
    A::T
    α::T
    γ::T
    G::T
end

"""
    activation_coefficients(T_K, p; q, constants, kinetic_radius = Inf,
                            α_c = 1.0, α_T = 1.0) -> ActivationCoefficients

Bundle the state-dependent activation coefficients at `(T_K, p)`. All
thermodynamics comes from `MoistThermodynamics`; nothing is refitted.
"""
function activation_coefficients(T_K::Real, p::Real;
                                 q::PhasePartition = PhasePartition(0.0),
                                 constants::ThermodynamicConstants = DEFAULT_CONSTANTS,
                                 kinetic_radius::Real = Inf,
                                 α_c::Real = 1.0, α_T::Real = 1.0)
    c = constants
    Tk = float(T_K); pp = float(p)
    Tk > 0 || throw(DomainError(T_K, "activation_coefficients: T must be > 0"))
    pp > 0 || throw(DomainError(p, "activation_coefficients: p must be > 0"))
    return ActivationCoefficients{Float64}(
        Tk, pp,
        saturation_vapor_pressure(Tk, c),
        latent_heat_vapor(Tk, c),
        heat_capacity_moist(q, c),
        air_density(Tk, pp, q, c),
        kelvin_A(Tk),
        supersaturation_source(Tk, pp; q = q, constants = c),
        supersaturation_sink(Tk, pp; q = q, constants = c),
        growth_coefficient(Tk, pp; constants = c, r = kinetic_radius,
                           α_c = α_c, α_T = α_T, q = q))
end

Base.show(io::IO, co::ActivationCoefficients) = print(io,
    "ActivationCoefficients(T=", round(co.T_K; digits = 2), " K, p=",
    round(co.p / 100; digits = 1), " hPa, α=", co.α, " m⁻¹, γ=", co.γ,
    " m³/kg, G=", co.G, " m²/s)")

# =========================================================================
# 3. Aerosol modes in activation (SI) units
# =========================================================================

"""
    ARGMode{T}

One log-normal aerosol mode in the units activation physics is written
in — SI throughout, unlike `AerosolOptics.LogNormalMode`, which uses the
aerosol-measurement conventions of cm⁻³ and μm:

  * `N`   — number concentration, **m⁻³**
  * `r_g` — geometric median **dry** radius, **m**
  * `σ_g` — geometric standard deviation (> 1)
  * `κ`   — hygroscopicity parameter
  * `ρ_p` — dry particle density, **kg m⁻³**

The unit change is deliberate and explicit rather than hidden inside the
routines: mixing cm⁻³ with m⁻³ inside an `η` coefficient produces a
10⁶-fold error in the competition term and a silently wrong `S_max`.
"""
struct ARGMode{T<:AbstractFloat}
    N::T
    r_g::T
    σ_g::T
    κ::T
    ρ_p::T
end

"""
    arg_mode(N, r_g, σ_g, κ; ρ_p = 1770.0) -> ARGMode

Construct a mode from SI quantities: `N` in m⁻³, `r_g` in m.
The default density is that of ammonium sulphate.
"""
function arg_mode(N::Real, r_g::Real, σ_g::Real, κ::Real;
                  ρ_p::Real = 1770.0)
    N >= 0 || error("arg_mode: N must be non-negative")
    r_g > 0 || error("arg_mode: r_g must be positive")
    σ_g > 1 || error("arg_mode: σ_g must exceed 1")
    κ >= 0 || error("arg_mode: κ must be non-negative")
    ρ_p > 0 || error("arg_mode: ρ_p must be positive")
    return ARGMode{Float64}(float(N), float(r_g), float(σ_g), float(κ),
                            float(ρ_p))
end

"""
    arg_mode(m::AerosolOptics.LogNormalMode) -> ARGMode

Convert a Phase 37 optical mode into activation units, taking `κ` and
the particle density from the species record so that the *same* aerosol
that scatters light is the aerosol that nucleates droplets. The
conversions are `N[m⁻³] = 10⁶ N[cm⁻³]`, `r[m] = 10⁻⁶ r[μm]` and
`ρ[kg m⁻³] = 10³ ρ[g cm⁻³]`.
"""
arg_mode(m::LogNormalMode) =
    ARGMode{Float64}(Float64(m.N) * 1e6, Float64(m.r_g) * 1e-6,
                     Float64(m.σ_g), m.species.κ, m.species.ρ * 1e3)

"""
    arg_modes(modes) -> Vector{ARGMode}

Convert a whole `AerosolOptics` mixture (e.g. `aerosol_mixture(:maritime_clean)`)
into activation units.
"""
arg_modes(modes::AbstractVector{<:AerosolMode}) = [arg_mode(m) for m in modes]
arg_modes(modes::AbstractVector{<:ARGMode}) = collect(modes)

Base.show(io::IO, m::ARGMode) = print(io,
    "ARGMode(N=", m.N * 1e-6, " cm⁻³, r_g=", m.r_g * 1e6, " μm, σ_g=",
    m.σ_g, ", κ=", m.κ, ")")

"""
    mode_critical_supersaturation(m, T_K) -> s_m

Critical supersaturation of a particle at the mode's *median* dry
radius. This is Abdul-Razzak & Ghan's `S_mi`, and their algebra is
entirely written in terms of it and `σ_g`.
"""
mode_critical_supersaturation(m::ARGMode, T_K::Real = _T_KELVIN_REF) =
    critical_supersaturation(m.r_g, m.κ; T_K = T_K)

"Dry mass concentration of a mode, kg m⁻³: `M = N ρ (4π/3) r_g³ exp(4.5 ln²σ_g)`."
mode_dry_mass(m::ARGMode) =
    m.N * m.ρ_p * (4π / 3) * m.r_g^3 * exp(4.5 * log(m.σ_g)^2)

"""
    number_above_radius(m, r_cut) -> m⁻³

Number concentration of a log-normal mode above a cut-off dry radius,

```math
N(>r) = \\frac{N}{2}\\,\\mathrm{erfc}\\!\\left(
        \\frac{\\ln(r/r_g)}{\\sqrt2 \\ln σ_g}\\right)
```

exact for the log-normal, using `GasOptics.erfc_series`.
"""
function number_above_radius(m::ARGMode, r_cut::Real)
    r_cut <= 0 && return m.N
    u = log(float(r_cut) / m.r_g) / (sqrt(2.0) * log(m.σ_g))
    return m.N * 0.5 * erfc_series(u)
end

"""
    mass_above_radius(m, r_cut) -> kg m⁻³

Dry mass above a cut-off radius. The mass distribution of a log-normal
is itself log-normal with the same `σ_g` and a median shifted to
`r_g exp(3 ln²σ_g)`, so the cut-off simply moves by `3 lnσ_g/√2` in the
erfc argument. For `σ_g = 2` that shift is 1.47 — i.e. the mass sits
about 1.5 standard deviations coarser than the number, which is why a
small activated *number* fraction can still remove most of the aerosol
*mass*.
"""
function mass_above_radius(m::ARGMode, r_cut::Real)
    M = mode_dry_mass(m)
    r_cut <= 0 && return M
    lnσ = log(m.σ_g)
    u = log(float(r_cut) / m.r_g) / (sqrt(2.0) * lnσ) - 3 * lnσ / sqrt(2.0)
    return M * 0.5 * erfc_series(u)
end

# =========================================================================
# 3b. The CCN spectrum N(S)
# =========================================================================
#
# Number of particles of a log-normal mode whose critical supersaturation
# lies below S. With the dilute κ-Köhler law s_crit ∝ r_d^{-3/2} the
# cut-off dry radius is r_g (S_m/S)^{2/3} and the log-normal integral
# above it is ½ erfc(2 ln(S_m/S) / (3√2 ln σ_g)). This is the last step
# of Abdul-Razzak & Ghan's scheme and it is also what any caller needs
# who knows the supersaturation already (a sub-grid PDF of S, a measured
# CCN counter setting). One copy, used by `arg_activation` below.

@inline _ccn_fraction(S_m, S, lnσ) =
    0.5 * erfc_series(2 * log(S_m / S) / (3 * sqrt(2.0) * lnσ))

"""
    ccn_fraction(m::ARGMode, S; T_K = 298.15) -> fraction in [0, 1]

Fraction of the particles of mode `m` that are activated at the
(fractional) supersaturation `S`, i.e. whose dilute κ-Köhler critical
supersaturation is below `S`:

```math
\\frac{N_\\mathrm{CCN}(S)}{N} = \\tfrac12\\,\\mathrm{erfc}\\!\\left(
   \\frac{2\\ln(S_m/S)}{3\\sqrt2\\ln σ_g}\\right),
```

with `S_m` = [`mode_critical_supersaturation`](@ref)`(m, T_K)`. This is
the CCN spectrum of a log-normal mode (Abdul-Razzak et al. 1998, eq. 13)
and **the** expression [`arg_activation`](@ref) evaluates at `S_max`, so
`ccn_fraction(m, r.S_max; T_K)` reproduces `r.fraction` bit for bit.

Returns 0 for `S ≤ 0` and for non-CCN-active modes (`κ ≤ KAPPA_MIN`).
"""
function ccn_fraction(m::ARGMode, S::Real; T_K::Real = _T_KELVIN_REF)
    (S > 0 && m.κ > KAPPA_MIN) || return 0.0
    S_m = mode_critical_supersaturation(m, T_K)
    isfinite(S_m) || return 0.0
    isfinite(S) || return 1.0
    return _ccn_fraction(S_m, float(S), log(m.σ_g))
end

"""
    ccn_spectrum(modes, S; T_K = 298.15) -> m⁻³

CCN number concentration `N_CCN(S) = Σ_i N_i ccn_fraction(m_i, S)` of a
multi-mode population at supersaturation `S`. Accepts `ARGMode`s or
`AerosolOptics` modes.
"""
ccn_spectrum(modes::AbstractVector{<:ARGMode}, S::Real; T_K::Real = _T_KELVIN_REF) =
    sum((m.N * ccn_fraction(m, S; T_K = T_K) for m in modes); init = 0.0)
ccn_spectrum(modes::AbstractVector{<:AerosolMode}, S::Real; kw...) =
    ccn_spectrum(arg_modes(modes), S; kw...)

# =========================================================================
# 4. Abdul-Razzak & Ghan (2000) activation
# =========================================================================

"""
    ActivationResult{T}

Outcome of an activation calculation, from the parameterisation or (in
its own struct) from the parcel model.

  * `S_max`        — maximum supersaturation reached (fractional, not %)
  * `S_m`          — per-mode critical supersaturation at the median radius
  * `fraction`     — activated number fraction per mode
  * `N_act`        — activated number concentration per mode, m⁻³
  * `N_act_total`  — summed activated number, m⁻³
  * `r_cut`        — per-mode dry radius at which `s_crit = S_max`, m
  * `coeffs`       — the coefficients used
"""
struct ActivationResult{T<:AbstractFloat}
    S_max::T
    S_m::Vector{T}
    fraction::Vector{T}
    N_act::Vector{T}
    N_act_total::T
    r_cut::Vector{T}
    coeffs::ActivationCoefficients{T}
end

Base.show(io::IO, r::ActivationResult) = print(io,
    "ActivationResult(S_max=", round(r.S_max * 100; digits = 4), " %, N_act=",
    round(r.N_act_total * 1e-6; digits = 2), " cm⁻³, fractions=",
    round.(r.fraction; digits = 4), ")")

"""
    arg_activation(modes, w, T_K, p; q, constants) -> ActivationResult
    arg_activation(modes, w, coeffs)              -> ActivationResult

Abdul-Razzak & Ghan (2000) multi-mode activation parameterisation.

Given the updraught `w` (m s⁻¹) and the ambient state, it returns the
maximum supersaturation and the activated fraction of each log-normal
mode, without integrating anything.

**The algebra.** Define the dimensionless updraught and competition
parameters

```math
ζ = \\frac{2A}{3}\\sqrt{\\frac{α w}{G}}, \\qquad
η_i = \\frac{(α w / G)^{3/2}}{2π ρ_w γ N_i}
```

`ζ` measures curvature against the kinetic growth forced by the
updraught; `η_i` measures how fast mode *i* can remove vapour. Then
(ARG 2000, eq. 6)

```math
\\frac{1}{S_\\max^2} = \\sum_i \\frac{1}{S_{mi}^2}\\left[
   f_i\\left(\\frac{ζ}{η_i}\\right)^{3/2}
 + g_i\\left(\\frac{S_{mi}^2}{η_i + 3ζ}\\right)^{3/4}\\right]
```

with `f_i = 0.5 exp(2.5 ln²σ_i)` and `g_i = 1 + 0.25 ln σ_i` — empirical
shape factors fitted by ARG to their own parcel solutions. Finally each
mode's activated fraction follows from the log-normal integral above the
dry radius whose critical supersaturation equals `S_max`:

```math
\\frac{N_{a,i}}{N_i} = \\frac12\\,\\mathrm{erfc}\\!\\left(
   \\frac{2\\ln(S_{mi}/S_\\max)}{3\\sqrt2 \\ln σ_i}\\right)
```

**Limits and known behaviour.** Modes with `κ ≤ KAPPA_MIN` have
`S_mi = ∞`; they contribute nothing to the sum and activate nothing,
which is the correct treatment of insoluble aerosol in this framework
(it ignores the weak Kelvin-only activation of very large hydrophobic
particles). With `w ≤ 0` there is no source of supersaturation and the
result is identically zero. With no CCN-active aerosol at all the
denominator vanishes and `S_max = Inf` is returned — physically the
parcel would supersaturate until homogeneous nucleation, which is
outside this parameterisation's scope.

ARG report ~10-20 % accuracy in `N_act` against parcel models over the
range of updraughts and aerosol loadings encountered in warm clouds; the
test suite measures it rather than assuming it.
"""
function arg_activation(modes::AbstractVector{<:ARGMode}, w::Real,
                        co::ActivationCoefficients{T}) where {T}
    n = length(modes)
    S_m   = fill(T(Inf), n)
    frac  = zeros(T, n)
    N_act = zeros(T, n)
    r_cut = fill(T(Inf), n)
    ww = T(w)
    if !(ww > 0) || n == 0
        for i in 1:n
            S_m[i] = mode_critical_supersaturation(modes[i], co.T_K)
        end
        return ActivationResult{T}(zero(T), S_m, frac, N_act, zero(T), r_cut, co)
    end

    AwG = co.α * ww / co.G                      # m⁻²
    ζ = (2 / 3) * co.A * sqrt(AwG)
    η = fill(T(Inf), n)
    acc = zero(T)
    for i in 1:n
        m = modes[i]
        S_m[i] = mode_critical_supersaturation(m, co.T_K)
        (m.N > 0 && m.κ > KAPPA_MIN && isfinite(S_m[i])) || continue
        η[i] = AwG^1.5 / (2π * RHO_WATER * co.γ * m.N)
        lnσ = log(m.σ_g)
        f = 0.5 * exp(2.5 * lnσ^2)
        g = 1 + 0.25 * lnσ
        acc += (f * (ζ / η[i])^1.5 +
                g * (S_m[i]^2 / (η[i] + 3ζ))^0.75) / S_m[i]^2
    end
    S_max = acc > 0 ? 1 / sqrt(acc) : T(Inf)

    total = zero(T)
    if isfinite(S_max) && S_max > 0
        for i in 1:n
            m = modes[i]
            (m.N > 0 && m.κ > KAPPA_MIN && isfinite(S_m[i])) || continue
            frac[i] = _ccn_fraction(S_m[i], S_max, log(m.σ_g))
            N_act[i] = m.N * frac[i]
            total += N_act[i]
            # dry radius whose critical supersaturation equals S_max
            r_cut[i] = m.r_g * (S_m[i] / S_max)^(2 / 3)
        end
    end
    return ActivationResult{T}(S_max, S_m, frac, N_act, total, r_cut, co)
end

function arg_activation(modes::AbstractVector{<:ARGMode}, w::Real,
                        T_K::Real, p::Real;
                        q::PhasePartition = PhasePartition(0.0),
                        constants::ThermodynamicConstants = DEFAULT_CONSTANTS,
                        kinetic_radius::Real = Inf)
    co = activation_coefficients(T_K, p; q = q, constants = constants,
                                 kinetic_radius = kinetic_radius)
    return arg_activation(modes, w, co)
end

arg_activation(modes::AbstractVector{<:AerosolMode}, args...; kwargs...) =
    arg_activation(arg_modes(modes), args...; kwargs...)

"Maximum supersaturation from [`arg_activation`](@ref) (fractional)."
arg_maximum_supersaturation(args...; kwargs...) =
    arg_activation(args...; kwargs...).S_max

"""
    activated_fraction(result) -> Vector
    activated_fraction(result, i) -> Real

Activated number fraction, per mode or bulk. The bulk fraction of an
`ActivationResult` is `N_act_total / Σ N_i`.
"""
activated_fraction(r::ActivationResult) = r.fraction
activated_fraction(r::ActivationResult, i::Integer) = r.fraction[i]

# =========================================================================
# 5. Explicit adiabatic parcel model — the reference
# =========================================================================

"""
    ParcelSettings

Numerics of the bin-resolved adiabatic parcel.

  * `nbins`      — size bins per mode (log-spaced in dry radius)
  * `n_sigma`    — half-width of the binned range in units of `ln σ_g`
  * `dt_max`     — ceiling on the time step, s
  * `ds_step`    — target supersaturation change per step from the `αw`
                   term; the main accuracy control
  * `t_max`, `z_max` — integration limits
  * `s_init`     — initial supersaturation (0 = exactly saturated, the
                   conventional cloud-base start)
  * `stop_fraction` — stop once `S` has fallen to this fraction of its peak
  * `max_steps`  — hard iteration cap
  * `store_trace`— keep the `t`, `z`, `S`, `LWC` time series

Defaults resolve `S_max` to better than 0.5 % and the activated fraction
to better than ~1 % (self-convergence under bin and step refinement is
checked in the test suite).
"""
Base.@kwdef struct ParcelSettings
    nbins::Int = 100
    n_sigma::Float64 = 4.0
    dt_max::Float64 = 0.05
    dt_min::Float64 = 1.0e-7
    ds_step::Float64 = 1.0e-5
    t_max::Float64 = 900.0
    z_max::Float64 = 600.0
    s_init::Float64 = 0.0
    stop_fraction::Float64 = 0.6
    max_steps::Int = 400_000
    store_trace::Bool = true
end

"""
    ParcelResult{T}

Result of an explicit parcel integration.

  * `S_max`, `t_smax`, `z_smax` — peak supersaturation and where it occurs
  * `N_act`, `fraction`, `N_act_total` — activated number using the
    **dynamic** criterion `r > r_crit` evaluated at the time of `S_max`
  * `N_act_end`, `N_act_total_end` — the same criterion at the end of the run
  * `N_crit`, `N_crit_total` — the **equilibrium** criterion
    `s_crit < S_max`, i.e. what ARG's erfc cutoff would give from this
    parcel's own `S_max`. The gap between `N_crit` and `N_act` is the
    kinetic limitation: particles whose barrier `S_max` cleared but which
    had no time to grow past it.
  * `lwc`, `nsteps`, and optional `t`/`z`/`S`/`lwc_trace` time series
"""
struct ParcelResult{T<:AbstractFloat}
    S_max::T
    t_smax::T
    z_smax::T
    N_act::Vector{T}
    fraction::Vector{T}
    N_act_total::T
    N_act_end::Vector{T}
    N_act_total_end::T
    N_crit::Vector{T}
    N_crit_total::T
    lwc::T
    nsteps::Int
    t::Vector{T}
    z::Vector{T}
    S::Vector{T}
    lwc_trace::Vector{T}
end

Base.show(io::IO, r::ParcelResult) = print(io,
    "ParcelResult(S_max=", round(r.S_max * 100; digits = 4), " %, N_act=",
    round(r.N_act_total * 1e-6; digits = 2), " cm⁻³, steps=", r.nsteps, ")")

# φ₁(z) = (eᶻ − 1)/z, the exponential-integrator weight. Stable for
# large negative z (stiff relaxation), series-expanded near zero.
@inline function _phi1(z::Float64)
    az = abs(z)
    az < 1e-8 && return 1 + 0.5z
    az < 1e-3 && return 1 + z * (0.5 + z * (1 / 6 + z / 24))
    return expm1(z) / z
end

"""
    parcel_activation(modes, w, T_K, p; settings, q, constants) -> ParcelResult

Explicit adiabatic-parcel activation: the reference calculation.

**What is integrated.** Each log-normal mode is discretised into
`nbins` dry-size bins whose number concentrations come from the *exact*
log-normal integral (`erf` differences of the bin edges, so the
discretisation conserves number by construction). The state is the
parcel temperature, pressure, height, supersaturation, and one wet
radius per bin. The equations are

```math
\\frac{dr_j}{dt} = \\frac{G}{r_j}\\bigl(s - s_\\mathrm{eq}(r_j, r_{d,j}, κ_j)\\bigr)
```
```math
\\frac{ds}{dt} = α w - γ\\frac{dχ}{dt}, \\qquad
χ = \\frac{4π ρ_w}{3}\\sum_j n_j r_j^3
```
```math
\\frac{dT}{dt} = -\\frac{g w}{c_p} + \\frac{L}{c_p ρ_a}\\frac{dχ}{dt},
\\qquad \\frac{dp}{dt} = -ρ_a g w
```

with `s_eq` the **exact** κ-Köhler supersaturation (no dilute
approximation), so the haze population equilibrates correctly and the
activation barrier is the real one.

**Why it needs a special integrator.** The growth equation is violently
stiff: a 10 nm haze particle relaxes to its equilibrium radius with a
time constant of ~50 μs, while the supersaturation itself evolves over
seconds. Explicit stepping would need 10⁷ steps. Instead the equation is
written for `y = r²`, where it becomes `dy/dt = 2G(s − s_eq(y))`, `s_eq`
is linearised about the current radius, and each bin is advanced with
the *exponential* integrator

```math
Δy_j = 2GΔt\\,φ_1(-λ_jΔt)\\,(\\bar s - s_{eq,j}),\\qquad
λ_j = 2G\\,\\frac{∂s_\\mathrm{eq}}{∂y}
```

which is exact for the linearised problem and unconditionally stable for
`λ > 0` — it drives strongly-relaxing bins straight onto their
quasi-equilibrium instead of oscillating. Because `Δy_j` is *affine* in
the step-mean supersaturation `s̄`, the supersaturation equation closes
into a single scalar linear equation each step, solved directly (no
iteration, no operator-splitting error):

```math
\\bar s = θ s^{n+1} + (1-θ)s^n,\\quad Δχ = P\\bar s - Q,\\quad
s^{n+1} = \\frac{s^n(1-γP(1-θ)) + αwΔt + γQ}{1 + γPθ}
```

with `θ = 1/2` (trapezoidal). The step is shortened whenever `γP > 1/2`
(the phase-relaxation limit), whenever an unstable bin would be advanced
by more than `0.3/|λ|`, or whenever any bin's `r²` would more than
double.

**Definition of "activated".** The primary count is the *dynamic* one:
bins whose wet radius exceeds their (numerically exact) critical radius
at the moment `S` peaks. The cut-off dry radius is interpolated
logarithmically between the two bins that straddle `r = r_crit` and the
activated number is then evaluated from the *analytic* log-normal
integral above that cut, so the answer is not quantised by the bin
width.

**What is deliberately absent.** No kinetic (non-continuum) growth
correction, no solute-dependent surface tension, no entrainment, no
collision-coalescence, and no ice. Each of those would move `S_max` and
none of them is part of the ARG derivation, so including them here would
corrupt the comparison this model exists to make.
"""
function parcel_activation(modes::AbstractVector{<:ARGMode}, w::Real,
                           T_K::Real, p::Real;
                           settings::ParcelSettings = ParcelSettings(),
                           q::PhasePartition = PhasePartition(0.0),
                           constants::ThermodynamicConstants = DEFAULT_CONSTANTS)
    nm = length(modes)
    ww = Float64(w)
    ww > 0 || error("parcel_activation: updraught must be positive")

    # ---- bin the modes -------------------------------------------------
    bmode = Int[]; r_d = Float64[]; nconc = Float64[]; bκ = Float64[]
    for (i, m) in enumerate(modes)
        (m.N > 0 && m.κ > KAPPA_MIN) || continue
        lnσ = log(m.σ_g)
        lo = log(m.r_g) - settings.n_sigma * lnσ
        hi = log(m.r_g) + settings.n_sigma * lnσ
        nb = settings.nbins
        s2 = sqrt(2.0) * lnσ
        for j in 1:nb
            e1 = lo + (hi - lo) * (j - 1) / nb
            e2 = lo + (hi - lo) * j / nb
            u1 = (e1 - log(m.r_g)) / s2
            u2 = (e2 - log(m.r_g)) / s2
            nj = m.N * 0.5 * (erf_series(u2) - erf_series(u1))
            nj > 0 || continue
            push!(bmode, i)
            push!(r_d, exp(0.5 * (e1 + e2)))
            push!(nconc, nj)
            push!(bκ, m.κ)
        end
    end
    nb = length(r_d)

    Tk = Float64(T_K); pp = Float64(p)
    s = settings.s_init

    y   = zeros(nb)                      # r² in m²
    ymin = zeros(nb)
    for k in 1:nb
        r0 = equilibrium_radius(r_d[k], s, bκ[k]; T_K = Tk)
        isfinite(r0) || (r0 = critical_radius(r_d[k], bκ[k]; T_K = Tk))
        y[k] = r0^2
        ymin[k] = (r_d[k] * (1 + 1e-9))^2
    end

    ŝ = zeros(nb); cs = zeros(nb); E = zeros(nb); Δy = zeros(nb); rr = zeros(nb)

    t = 0.0; z = 0.0
    s_max = s; t_smax = 0.0; z_smax = 0.0
    y_at = copy(y); Tk_at = Tk
    tr_t = Float64[]; tr_z = Float64[]; tr_S = Float64[]; tr_l = Float64[]
    χ = 4π / 3 * RHO_WATER * sum(nconc[k] * y[k]^1.5 for k in 1:nb; init = 0.0)
    nsteps = 0
    dt = settings.dt_max

    while t < settings.t_max && z < settings.z_max && nsteps < settings.max_steps
        co = activation_coefficients(Tk, pp; q = q, constants = constants)
        G = co.G; A = co.A
        for k in 1:nb
            r = sqrt(y[k]); rr[k] = r
            ŝ[k] = kohler_saturation(r * 1e6, r_d[k] * 1e6, bκ[k]; T_K = Tk) - 1
            dsdr = kohler_slope(r, r_d[k], bκ[k]; T_K = Tk)
            cs[k] = dsdr / (2r)
        end
        # --- step size ---------------------------------------------------
        dt = settings.dt_max
        αw = co.α * ww
        αw > 0 && (dt = min(dt, settings.ds_step / αw))
        λneg = 0.0
        for k in 1:nb
            λ = 2G * cs[k]
            λ < λneg && (λneg = λ)
        end
        λneg < 0 && (dt = min(dt, 0.3 / (-λneg)))
        dt = max(dt, settings.dt_min)
        dt = min(dt, settings.t_max - t + 1e-12)

        accepted = false
        s_new = s; Δχ = 0.0
        for _ in 1:14
            P = 0.0; Q = 0.0
            @inbounds for k in 1:nb
                λ = 2G * cs[k]
                E[k] = 2G * dt * _phi1(-λ * dt)
                w_k = nconc[k] * rr[k] * E[k]
                P += w_k
                Q += w_k * ŝ[k]
            end
            P *= 2π * RHO_WATER
            Q *= 2π * RHO_WATER
            if co.γ * P > 0.5 && dt > settings.dt_min
                dt *= 0.5; continue
            end
            θ = 0.5
            s_new = (s * (1 - co.γ * P * (1 - θ)) + αw * dt + co.γ * Q) /
                    (1 + co.γ * P * θ)
            s_bar = θ * s_new + (1 - θ) * s
            too_big = false
            @inbounds for k in 1:nb
                Δy[k] = E[k] * (s_bar - ŝ[k])
                if Δy[k] > y[k]
                    too_big = true; break
                end
            end
            if too_big && dt > settings.dt_min
                dt *= 0.5; continue
            end
            Δχ = P * s_bar - Q
            accepted = true
            break
        end
        accepted || break

        @inbounds for k in 1:nb
            y[k] = max(y[k] + Δy[k], ymin[k])
        end
        Tk += (-constants.grav / co.c_p) * ww * dt + co.L / (co.c_p * co.ρ_a) * Δχ
        pp += -co.ρ_a * constants.grav * ww * dt
        z += ww * dt
        t += dt
        s = s_new
        χ = 4π / 3 * RHO_WATER * sum(nconc[k] * y[k]^1.5 for k in 1:nb; init = 0.0)
        nsteps += 1

        if s > s_max
            s_max = s; t_smax = t; z_smax = z
            copyto!(y_at, y); Tk_at = Tk
        end
        if settings.store_trace
            push!(tr_t, t); push!(tr_z, z); push!(tr_S, s); push!(tr_l, χ)
        end
        if s_max > 0 && t > t_smax && s < settings.stop_fraction * s_max
            break
        end
    end

    # ---- activation diagnostics ---------------------------------------
    r_crit = zeros(nb); s_crit = zeros(nb)
    for k in 1:nb
        kc = kohler_critical(r_d[k], bκ[k]; T_K = Tk_at)
        r_crit[k] = kc.r_crit
        s_crit[k] = kc.s_crit
    end
    N_act = zeros(nm); N_end = zeros(nm); N_crit = zeros(nm); frac = zeros(nm)
    for i in 1:nm
        idx = findall(==(i), bmode)
        isempty(idx) && continue
        m = modes[i]
        # activated at the peak: s_crit below the peak supersaturation
        N_act[i] = _activated_number(m, idx, r_d, s_crit, s_max)
        # still larger than critical at the end of the integration — a
        # strictly smaller number once the parcel has begun to relax
        N_end[i] = sum(nconc[k] for k in idx
                       if sqrt(y[k]) > r_crit[k]; init = 0.0)
        frac[i] = m.N > 0 ? N_act[i] / m.N : 0.0
        if s_max > 0 && m.κ > KAPPA_MIN
            r_cut = m.r_g * (mode_critical_supersaturation(m, Tk_at) / s_max)^(2 / 3)
            N_crit[i] = number_above_radius(m, r_cut)
        end
    end

    return ParcelResult{Float64}(s_max, t_smax, z_smax, N_act, frac, sum(N_act),
                                 N_end, sum(N_end), N_crit, sum(N_crit),
                                 χ, nsteps, tr_t, tr_z, tr_S, tr_l)
end

parcel_activation(modes::AbstractVector{<:AerosolMode}, args...; kwargs...) =
    parcel_activation(arg_modes(modes), args...; kwargs...)

# Activated number of one mode: locate the dry radius at which the bins
# cross their critical radius, interpolating logarithmically, then use
# the analytic log-normal integral above that cut so the answer is not
# quantised by the bin width.
function _activated_number(m::ARGMode, idx::Vector{Int}, r_d::Vector{Float64},
                           s_crit::Vector{Float64}, S_max::Real)
    # A particle is activated iff the parcel's PEAK supersaturation exceeded
    # that particle's critical supersaturation. Crossing the Köhler barrier
    # is irreversible for an ascending parcel: past `r_crit` the equilibrium
    # curve falls away, so the droplet grows freely thereafter and never
    # returns.
    #
    # The obvious-looking alternative — "has the droplet grown past r_crit
    # yet?" evaluated at the instant of S_max — is WRONG, and wrong in a way
    # that silently returns zero. At the moment S peaks, a just-activated
    # droplet is only just AT its critical radius; growing visibly past it
    # takes further time. Testing radius at the peak therefore reports no
    # activation at all, even for a 5 m/s updraught over 0.1 μm sea-salt
    # particles whose s_crit (0.043 %) is an order of magnitude below the
    # S_max actually reached (0.67 %).
    K = length(idx)
    Smax = Float64(S_max)
    (isfinite(Smax) && Smax > 0) || return 0.0
    margin(j) = Smax - s_crit[idx[j]]        # > 0 ⇒ that bin activates
    margin(K) > 0 || return 0.0              # not even the largest activates
    jcut = 0
    for j in K:-1:1
        if margin(j) <= 0
            jcut = j; break
        end
    end
    lnσ = log(m.σ_g)
    if jcut == 0
        lnrcut = log(r_d[idx[1]])            # the whole mode activates
    else
        m1 = margin(jcut); m2 = margin(jcut + 1)
        x1 = log(r_d[idx[jcut]]); x2 = log(r_d[idx[jcut + 1]])
        lnrcut = m2 > m1 ? x1 + (x2 - x1) * (-m1) / (m2 - m1) : x2
    end
    return m.N * 0.5 * erfc_series((lnrcut - log(m.r_g)) / (sqrt(2.0) * lnσ))
end

# =========================================================================
# 6. Prognostic aerosol: activation, regeneration, in-cloud processing
# =========================================================================

"""
    AerosolReservoir{T}

Prognostic aerosol state for a grid cell: a two-moment (number + dry
mass) representation of each mode, split into an **interstitial**
population — unactivated, still free — and an **activated** population,
the solute currently dissolved inside cloud droplets.

Fields are per mode: `σ_g`, `κ`, `ρ_p` (fixed shape parameters) and
`N_int`, `M_int`, `N_act`, `M_act` (prognostic, m⁻³ and kg m⁻³).

**Why two moments and not a spectrum.** Number and mass are what an LES
can afford to advect. The median radius of each population is *derived*
from them,

```math
r_g = \\left[\\frac{M}{N ρ_p (4π/3)\\exp(4.5\\ln^2σ_g)}\\right]^{1/3}
```

with `σ_g` held fixed — the standard modal closure (Whitby & McMurry
1997; the MAM and M7 schemes). It is an approximation, and it is the
*right* one here, because it is exactly what makes aerosol processing
representable: coalescence removes number at fixed mass, `r_g` grows,
and the enlarged particles are returned to the interstitial population
on evaporation. A one-moment scheme cannot express that at all.
"""
mutable struct AerosolReservoir{T<:AbstractFloat}
    σ_g::Vector{T}
    κ::Vector{T}
    ρ_p::Vector{T}
    N_int::Vector{T}
    M_int::Vector{T}
    N_act::Vector{T}
    M_act::Vector{T}
end

"""
    aerosol_reservoir(modes) -> AerosolReservoir

Initialise a reservoir from a list of `ARGMode`s (or `AerosolOptics`
modes). All aerosol starts interstitial.
"""
function aerosol_reservoir(modes::AbstractVector{<:ARGMode})
    n = length(modes)
    r = AerosolReservoir{Float64}(zeros(n), zeros(n), zeros(n),
                                  zeros(n), zeros(n), zeros(n), zeros(n))
    for (i, m) in enumerate(modes)
        r.σ_g[i] = m.σ_g; r.κ[i] = m.κ; r.ρ_p[i] = m.ρ_p
        r.N_int[i] = m.N; r.M_int[i] = mode_dry_mass(m)
    end
    return r
end
aerosol_reservoir(modes::AbstractVector{<:AerosolMode}) =
    aerosol_reservoir(arg_modes(modes))

Base.length(r::AerosolReservoir) = length(r.N_int)

Base.show(io::IO, r::AerosolReservoir) = print(io,
    "AerosolReservoir(", length(r), " modes, N_int=",
    round.(r.N_int .* 1e-6; digits = 2), " cm⁻³, N_act=",
    round.(r.N_act .* 1e-6; digits = 2), " cm⁻³)")

_rg_from_moments(N, M, ρ, σ) =
    (N > 0 && M > 0) ?
    (M / (N * ρ * (4π / 3) * exp(4.5 * log(σ)^2)))^(1 / 3) : 0.0

"""
    reservoir_dry_radius(res, i; phase = :interstitial) -> m

Median dry radius of one population of mode `i`, diagnosed from its
number and mass. `phase` is `:interstitial`, `:activated` or `:total`.

This is the quantity that carries the memory of in-cloud processing: a
mode that has been through a cloud cycle with coalescence comes back
with a larger `r_g` than it went in with.
"""
function reservoir_dry_radius(res::AerosolReservoir, i::Integer;
                              phase::Symbol = :interstitial)
    N, M = if phase === :interstitial
        res.N_int[i], res.M_int[i]
    elseif phase === :activated
        res.N_act[i], res.M_act[i]
    elseif phase === :total
        res.N_int[i] + res.N_act[i], res.M_int[i] + res.M_act[i]
    else
        error("reservoir_dry_radius: phase must be :interstitial, :activated or :total")
    end
    return _rg_from_moments(N, M, res.ρ_p[i], res.σ_g[i])
end

"""
    interstitial_modes(res) -> Vector{ARGMode}

The interstitial aerosol as activation-ready modes, with each mode's
median radius re-diagnosed from its current number and mass. Feed this
back into [`arg_activation`](@ref) to close the loop: activation depletes
the reservoir, the reservoir determines the next activation.
"""
interstitial_modes(res::AerosolReservoir) =
    [ARGMode{Float64}(res.N_int[i],
                      max(_rg_from_moments(res.N_int[i], res.M_int[i],
                                           res.ρ_p[i], res.σ_g[i]), 1e-12),
                      res.σ_g[i], res.κ[i], res.ρ_p[i])
     for i in 1:length(res)]

"Interstitial number concentration, per mode or total (m⁻³)."
interstitial_number(res::AerosolReservoir) = sum(res.N_int)
interstitial_number(res::AerosolReservoir, i::Integer) = res.N_int[i]

"Activated (in-droplet) number concentration, per mode or total (m⁻³)."
activated_number(res::AerosolReservoir) = sum(res.N_act)
activated_number(res::AerosolReservoir, i::Integer) = res.N_act[i]

"Total aerosol number, interstitial + activated (m⁻³)."
aerosol_total_number(res::AerosolReservoir) = sum(res.N_int) + sum(res.N_act)

"Total dry aerosol mass, interstitial + activated (kg m⁻³)."
aerosol_total_mass(res::AerosolReservoir) = sum(res.M_int) + sum(res.M_act)

"""
    activate!(res, S_max; T_K = 283.15) -> N_newly_activated

Move the interstitial particles whose critical supersaturation is below
`S_max` into the activated population, mode by mode.

The cut-off dry radius follows from inverting
`s_crit = sqrt(4A³/27κr³)`:

```math
r_\\mathrm{cut} = \\left(\\frac{4A^3}{27 κ S_\\max^2}\\right)^{1/3}
```

and the transferred number and mass are the log-normal integrals above
it, [`number_above_radius`](@ref) and [`mass_above_radius`](@ref). Mass
moves with a coarser cut-off than number (by `3lnσ/√2` in the erfc
argument), so activating 30 % of the particles typically removes 70-90 %
of the aerosol *mass* — which is why the interstitial mode left behind
has a markedly smaller median radius.

Number and dry mass are conserved exactly; nothing is created or
destroyed, only relabelled.

Modes with `κ ≤ KAPPA_MIN` never activate.
"""
function activate!(res::AerosolReservoir, S_max::Real; T_K::Real = 283.15)
    moved = 0.0
    (isfinite(S_max) && S_max > 0) || return moved
    A = kelvin_A(T_K)
    for i in 1:length(res)
        (res.N_int[i] > 0 && res.κ[i] > KAPPA_MIN) || continue
        r_g = _rg_from_moments(res.N_int[i], res.M_int[i], res.ρ_p[i], res.σ_g[i])
        r_g > 0 || continue
        m = ARGMode{Float64}(res.N_int[i], r_g, res.σ_g[i], res.κ[i], res.ρ_p[i])
        r_cut = (4 * A^3 / (27 * res.κ[i] * S_max^2))^(1 / 3)
        dN = min(number_above_radius(m, r_cut), res.N_int[i])
        dM = min(mass_above_radius(m, r_cut), res.M_int[i])
        res.N_int[i] -= dN; res.N_act[i] += dN
        res.M_int[i] -= dM; res.M_act[i] += dM
        moved += dN
    end
    return moved
end

"""
    activate!(res, result::ActivationResult) -> N_newly_activated

Use the per-mode cut-off radii an activation calculation already
computed, rather than re-inverting the Köhler relation. Preferred when
the `ActivationResult` came from the same aerosol state.
"""
function activate!(res::AerosolReservoir, r::ActivationResult)
    moved = 0.0
    for i in 1:length(res)
        (res.N_int[i] > 0 && res.κ[i] > KAPPA_MIN && isfinite(r.r_cut[i])) || continue
        r_g = _rg_from_moments(res.N_int[i], res.M_int[i], res.ρ_p[i], res.σ_g[i])
        r_g > 0 || continue
        m = ARGMode{Float64}(res.N_int[i], r_g, res.σ_g[i], res.κ[i], res.ρ_p[i])
        dN = min(number_above_radius(m, r.r_cut[i]), res.N_int[i])
        dM = min(mass_above_radius(m, r.r_cut[i]), res.M_int[i])
        res.N_int[i] -= dN; res.N_act[i] += dN
        res.M_int[i] -= dM; res.M_act[i] += dM
        moved += dN
    end
    return moved
end

"""
    regenerate!(res, f = 1.0) -> N_returned

Evaporate a fraction `f` of the cloud droplets and return their solute
to the **interstitial** population of the mode it came from.

This is not bookkeeping pedantry. A cloud that forms and evaporates
without precipitating returns its CCN, so the aerosol is available to
the next cloud; but it does not return them *unchanged*, because while
they were droplets they coalesced and took up secondary mass. Since the
interstitial population is reconstructed from its number and mass
moments, the returned particles automatically widen the interstitial
median radius — see [`process_collisions!`](@ref) and
[`reservoir_dry_radius`](@ref).

Number and mass are conserved exactly.
"""
function regenerate!(res::AerosolReservoir, f::Real = 1.0)
    0 <= f <= 1 || error("regenerate!: f must be in [0, 1]")
    returned = 0.0
    for i in 1:length(res)
        dN = res.N_act[i] * f
        dM = res.M_act[i] * f
        res.N_act[i] -= dN; res.N_int[i] += dN
        res.M_act[i] -= dM; res.M_int[i] += dM
        returned += dN
    end
    return returned
end

"""
    process_collisions!(res, f; mode = 0) -> N_merged

In-cloud aerosol processing by collision-coalescence.

When two cloud droplets collide they make one droplet carrying **both**
solute masses. If that droplet later evaporates it leaves **one** dry
particle of mass `m₁ + m₂` — larger, by `2^{1/3}` in radius, than either
of its parents. The aerosol population that emerges from a cloud is
therefore shifted coarser than the one that went in, even though no
material was added. Since CCN activity goes as `r_d^{3/2}`, those
processed particles activate at *lower* supersaturation next time: the
cloud has made its own CCN more effective. This is a first-order control
on the second indirect effect and is invisible to any scheme that treats
aerosol as a fixed background.

`f` is the fraction of activated droplets removed by coalescence.
Activated **number** falls by `f`; activated **mass** is unchanged.
Pass `mode = i` to process a single mode, or `0` (the default) for all.

Number is intentionally *not* conserved here — coalescence destroys
particle number, which is the whole point. Mass is conserved exactly.
"""
function process_collisions!(res::AerosolReservoir, f::Real; mode::Integer = 0)
    0 <= f < 1 || error("process_collisions!: f must be in [0, 1)")
    merged = 0.0
    rng = mode == 0 ? (1:length(res)) : (mode:mode)
    for i in rng
        dN = res.N_act[i] * f
        res.N_act[i] -= dN
        merged += dN
    end
    return merged
end

# =========================================================================
# 7. Cloud droplet number, effective radius, and the indirect effects
# =========================================================================

"""
    cloud_droplet_number(modes, w, T_K, p; kwargs...) -> m⁻³

Cloud droplet number concentration diagnosed from the aerosol population
and the updraught via [`arg_activation`](@ref). This is the quantity
that makes an indirect effect an *effect* rather than a prescribed
number.
"""
cloud_droplet_number(modes, w::Real, T_K::Real, p::Real; kwargs...) =
    arg_activation(modes, w, T_K, p; kwargs...).N_act_total

"Droplet number from a precomputed result."
cloud_droplet_number(r::ActivationResult) = r.N_act_total
cloud_droplet_number(r::ParcelResult) = r.N_act_total

"""
    droplet_volume_radius(lwc, N_d) -> m

Mean-volume droplet radius, `r_v = (3 χ / (4π ρ_w N_d))^{1/3}`, for a
liquid water content `χ` (kg m⁻³) shared among `N_d` droplets (m⁻³).

At fixed `χ` it is exactly `∝ N_d^{-1/3}` — the Twomey scaling, before
any spectral-shape correction.
"""
function droplet_volume_radius(lwc::Real, N_d::Real)
    (lwc > 0 && N_d > 0) || return 0.0
    return (3 * float(lwc) / (4π * RHO_WATER * float(N_d)))^(1 / 3)
end

"""
    droplet_effective_radius(lwc, N_d; k = 0.8) -> m

Radiatively effective droplet radius `r_eff = ⟨r³⟩/⟨r²⟩`, obtained from
the mean-volume radius through the spectral-shape factor
`k = (r_v/r_eff)³`:

```math
r_\\mathrm{eff} = \\left(\\frac{3χ}{4π ρ_w N_d k}\\right)^{1/3}
```

Martin et al. (1994) measured `k ≈ 0.80 ± 0.07` in continental
stratocumulus and `k ≈ 0.67 ± 0.07` in maritime — maritime spectra are
broader, so the effective radius sits further above the volume radius.
`k = 1` would be a monodisperse cloud.

This is the radius `TwoStream.mie_cloud_optics` (Phase 33) consumes, and
`τ = 3·LWP/(2 ρ_w r_eff)` is the link between the two — see
[`cloud_optical_depth_lwp`](@ref). Note Phase 33 works in μm and g m⁻²;
this function returns metres.
"""
function droplet_effective_radius(lwc::Real, N_d::Real; k::Real = 0.8)
    (lwc > 0 && N_d > 0 && k > 0) || return 0.0
    return (3 * float(lwc) / (4π * RHO_WATER * float(N_d) * float(k)))^(1 / 3)
end

"""
    droplet_number_from_effective_radius(lwc, r_eff; k = 0.8) -> m⁻³

Inverse of [`droplet_effective_radius`](@ref) — the droplet number a
retrieval of `r_eff` and `LWC` implies.
"""
function droplet_number_from_effective_radius(lwc::Real, r_eff::Real;
                                              k::Real = 0.8)
    (lwc > 0 && r_eff > 0) || return 0.0
    return 3 * float(lwc) / (4π * RHO_WATER * float(k) * float(r_eff)^3)
end

"""
    cloud_water_path(lwc, h) -> kg m⁻²

Liquid water path of a slab of depth `h` (m) with uniform water content
`lwc` (kg m⁻³). Named to avoid colliding with `TwoStream.liquid_water_path`,
which integrates a full `AtmosphericColumn` and returns g m⁻².
"""
cloud_water_path(lwc::Real, h::Real) = float(lwc) * float(h)

"""
    cloud_optical_depth_lwp(lwp, r_eff) -> τ

Geometric-optics cloud optical depth,

```math
τ = \\frac{3\\,\\mathrm{LWP}}{2 ρ_w r_\\mathrm{eff}}
```

with `LWP` in kg m⁻² and `r_eff` in m. This is the same relation
`TwoStream.compute_optics!` uses (there written as `τ = 1.5·LWP/r_eff`
with LWP in g m⁻² and `r_eff` in μm, where the unit factors cancel), so
the two phases agree by construction: 100 g m⁻² at 10 μm gives τ = 15.

Named `_lwp` to avoid colliding with the exported
`TwoStream.cloud_optical_depth`.
"""
cloud_optical_depth_lwp(lwp::Real, r_eff::Real) =
    r_eff > 0 ? 3 * float(lwp) / (2 * RHO_WATER * float(r_eff)) : 0.0

"""
    cloud_albedo(τ; g = 0.85) -> A

Two-stream albedo of a conservatively scattering cloud over a black
surface,

```math
A = \\frac{(1-g)τ}{2 + (1-g)τ}
```

i.e. Twomey's (1991) formula, with `(1−g)τ` the similarity-scaled
optical depth. It is 0 at τ = 0 and → 1 as τ → ∞, with the steepest
logarithmic response near `A = 0.5`. That is exactly why the Twomey
effect is largest for optically intermediate clouds and nearly absent
for very thin or very thick ones.

`g = 0.85` is the standard visible asymmetry parameter for liquid cloud
and matches the value `TwoStream.MieCloudOptics` uses in the solar bands.
"""
function cloud_albedo(τ::Real; g::Real = 0.85)
    τs = (1 - float(g)) * float(τ)
    τs <= 0 && return 0.0
    return τs / (2 + τs)
end

"""
    albedo_susceptibility(A) -> dA/dlnN_d

Twomey's (1991) susceptibility of cloud albedo to droplet number at
fixed liquid water path:

```math
\\frac{dA}{d\\ln N_d} = \\frac{A(1-A)}{3}
```

The factor 1/3 is `dlnτ/dlnN_d`, from `τ ∝ r_eff^{-1} ∝ N_d^{1/3}` at
fixed LWP; `A(1−A)` is `dA/dlnτ` for the two-stream albedo above. It
peaks at 1/12 = 0.083 for `A = 0.5`.
"""
albedo_susceptibility(A::Real) = float(A) * (1 - float(A)) / 3

"""
    twomey_albedo_change(A, ΔlnN_d) -> ΔA

First-indirect (Twomey) albedo change for a given fractional change in
droplet number at fixed liquid water path, to first order:
`ΔA = A(1−A)/3 · Δln N_d`.

Doubling `N_d` in a cloud of albedo 0.5 brightens it by
`0.083 × ln2 ≈ 0.058` — about six albedo points, which is a very large
radiative perturbation by climate standards.
"""
twomey_albedo_change(A::Real, ΔlnN_d::Real) =
    albedo_susceptibility(A) * float(ΔlnN_d)

"""
    twomey_radius_exponent(lwc, N1, N2; k = 0.8) -> dln r_eff / dln N_d

Measured exponent of the effective-radius response to droplet number at
fixed liquid water content. Analytically exactly `−1/3`; this function
computes it from the implementation, so it is a check rather than a
restatement.
"""
function twomey_radius_exponent(lwc::Real, N1::Real, N2::Real; k::Real = 0.8)
    r1 = droplet_effective_radius(lwc, N1; k = k)
    r2 = droplet_effective_radius(lwc, N2; k = k)
    return log(r2 / r1) / log(float(N2) / float(N1))
end

"""
    activation_susceptibility(modes, w, T_K, p; δ = 0.05, kwargs...)
        -> dln N_d / dln N_a

The **activation susceptibility**: the fractional response of cloud
droplet number to a fractional change in total aerosol number, with the
size distribution shape, updraught and thermodynamic state held fixed.
Evaluated by a centred logarithmic finite difference through
[`arg_activation`](@ref), scaling every mode by `1 ± δ`.

This is the quantity that controls how much of an aerosol perturbation
survives into a radiative forcing, and it is emphatically **not** 1.
Values are typically 0.6-0.9 in clean air and fall towards 0.2-0.4 in
polluted air, because adding particles increases the competition for
vapour, lowers `S_max`, and so lowers the *fraction* that activates. In
the extreme updraught-limited regime it approaches zero: no matter how
much aerosol is added, the vapour supply fixes the droplet number.

The corresponding albedo response is
`ΔA = albedo_susceptibility(A) × (dlnN_d/dlnN_a) × ΔlnN_a`.
"""
function activation_susceptibility(modes::AbstractVector{<:ARGMode}, w::Real,
                                   T_K::Real, p::Real; δ::Real = 0.05,
                                   kwargs...)
    scale(f) = [ARGMode{Float64}(m.N * f, m.r_g, m.σ_g, m.κ, m.ρ_p)
                for m in modes]
    Np = cloud_droplet_number(scale(1 + δ), w, T_K, p; kwargs...)
    Nm = cloud_droplet_number(scale(1 - δ), w, T_K, p; kwargs...)
    (Np > 0 && Nm > 0) || return 0.0
    return log(Np / Nm) / log((1 + δ) / (1 - δ))
end

activation_susceptibility(modes::AbstractVector{<:AerosolMode}, args...; kwargs...) =
    activation_susceptibility(arg_modes(modes), args...; kwargs...)

"""
    autoconversion_rate(q_c, N_d; A = 1350, a = 2.47, b = 1.79) -> kg kg⁻¹ s⁻¹

Warm-rain autoconversion following Khairoutdinov & Kogan (2000):

```math
\\left.\\frac{∂q_r}{∂t}\\right|_\\mathrm{auto}
   = A\\,q_c^{2.47}\\,N_d^{-1.79}
```

with the cloud water mixing ratio `q_c` in kg kg⁻¹ and the droplet
number `N_d` in **cm⁻³**. The exponents were fitted to explicit
bin-microphysics LES of marine stratocumulus and are the standard choice
in LES; Kessler-type schemes instead use a threshold and no `N_d`
dependence at all, and therefore have no second indirect effect.

The `N_d^{-1.79}` is the heart of the Albrecht mechanism: collision
efficiency depends steeply on droplet size, so spreading the same liquid
water over more, smaller droplets suppresses rain formation far faster
than linearly.
"""
function autoconversion_rate(q_c::Real, N_d::Real; A::Real = 1350.0,
                             a::Real = 2.47, b::Real = 1.79)
    (q_c > 0 && N_d > 0) || return 0.0
    return float(A) * float(q_c)^float(a) * float(N_d)^(-float(b))
end

"""
    autoconversion_suppression(N_d, N_d_ref; b = 1.79) -> factor ≤ 1

Ratio of autoconversion rates at two droplet numbers and the same liquid
water, `(N_d/N_d_ref)^{-b}`. This is the **second indirect (Albrecht)
effect** expressed as a single number: doubling the droplet number
suppresses autoconversion to `2^{-1.79} = 0.289` of its former rate, a
71 % reduction.

`N_d` may be in any consistent units, since only the ratio enters.
"""
autoconversion_suppression(N_d::Real, N_d_ref::Real; b::Real = 1.79) =
    (float(N_d) / float(N_d_ref))^(-float(b))

"""
    albrecht_lifetime_factor(N_d, N_d_ref; b = 1.79) -> factor ≥ 1

Cloud-lifetime lengthening implied by the autoconversion suppression,
`(N_d/N_d_ref)^{b}`, on the argument that the lifetime of a
non-precipitating cloud scales inversely with its rain-production rate
at fixed water supply.

This is Albrecht's (1989) mechanism and the factor is deliberately
crude: a real lifetime also responds to entrainment, drizzle-driven
decoupling and mesoscale organisation, several of which act in the
*opposite* direction (the cloud-lifetime term in AR6 is much smaller
than this scaling alone implies, and the sign of the liquid-water-path
adjustment is still contested). It is exported as a diagnostic of the
microphysical suppression, not as a forcing.
"""
albrecht_lifetime_factor(N_d::Real, N_d_ref::Real; b::Real = 1.79) =
    (float(N_d) / float(N_d_ref))^(float(b))

end # module
