# Equilibrium wall laws: the stress a wall exerts on the flow at height h
# above it, given the wall-parallel speed U there.

export AbstractWallLaw, SpaldingLaw, RoughLogLaw, MOSTWallLaw, LEE_MOSER_2015,
       friction_velocity, wall_stress, wall_drag,
       spalding_yplus, spalding_uplus, matching_obukhov_length, MOSTFluxWallLaw

"""
    AbstractWallLaw

An equilibrium wall law: the relation between the wall-parallel speed
`U` (m s⁻¹) at a distance `h` (m) from the wall and the friction velocity
``u_τ = \\sqrt{τ_w/ρ}``. A concrete law implements

    friction_velocity(law, U, h; ν) -> u_τ        (m s⁻¹)

and inherits

    wall_stress(law, U, h; ν) = u_τ²             (kinematic, m² s⁻²)
    wall_drag(law, U, h; ν)   = τ_w / U          (m s⁻¹, the implicit drag coefficient)

`ν` is the kinematic viscosity (m² s⁻¹); laws of a fully rough wall
ignore it. The stress acts against the wall-parallel velocity,
``τ_{w,i} = τ_w\\,u_i/U`` (Piomelli & Balaras 2002).
"""
abstract type AbstractWallLaw end

"""
    friction_velocity(law::AbstractWallLaw, U, h; ν = nothing) -> u_τ

Friction velocity (m s⁻¹) of `law` for the wall-parallel speed `U`
(m s⁻¹, its magnitude is used) at distance `h` (m) from the wall. `ν`
(m² s⁻¹) is required by smooth-wall laws ([`SpaldingLaw`](@ref)).

Not to be confused with `AirSeaFlux.andreas_friction_velocity`, the
empirical sea-spray u_* of a 10 m neutral wind.
"""
function friction_velocity end

"""
    wall_stress(law::AbstractWallLaw, U, h; ν = nothing) -> τ_w

Kinematic wall stress ``τ_w/ρ = u_τ^2`` (m² s⁻²), the momentum flux
through the wall per unit area, from [`friction_velocity`](@ref).

`CutCell.wall_shear_stress` is the *resolved* (linear, ``ν u_t/Δn``)
stress at a cut-cell probe; this is the *modelled* one. The two agree in
the viscous sublayer (``h^+ < 1``) for [`SpaldingLaw`](@ref).
"""
wall_stress(law::AbstractWallLaw, U::Real, h::Real; ν = nothing) =
    friction_velocity(law, U, h; ν = ν)^2

"""
    wall_drag(law::AbstractWallLaw, U, h; ν = nothing) -> D

The stress per unit wind, ``D = τ_w/U`` (m s⁻¹): the coefficient of the
linear, implicit drag ``τ_{w,i} = D\\,u_i`` of
[`apply_wall_stress!`](@ref). At ``U = 0`` it is the limit ``U → 0⁺``
(`ν/h` for a smooth wall, 0 for a rough one), so a calm cell is never
divided by zero. [`MOSTWallLaw`](@ref) divides by ``\\max(U, U_{min})``,
its minimum wind.
"""
function wall_drag(law::AbstractWallLaw, U::Real, h::Real; ν = nothing)
    a = abs(U)
    a > 0 || return zero(float(a))
    return wall_stress(law, a, h; ν = ν) / a
end

# =========================================================================
# Spalding (1961): smooth wall, viscous sublayer through log layer
# =========================================================================

"""
    SpaldingLaw(; κ = 0.41, B = 5.2)

Spalding's (1961) single formula for the law of the wall over a smooth
wall, valid from the viscous sublayer through the buffer layer to the
log layer:

```math
y^+ = u^+ + e^{-κB}\\left[e^{κu^+} - 1 - κu^+ - \\frac{(κu^+)^2}{2} - \\frac{(κu^+)^3}{6}\\right]
```

with ``u^+ = U/u_τ`` and ``y^+ = h u_τ/ν``. It tends to ``u^+ = y^+`` for
``y^+ → 0`` and to the log law ``u^+ = \\ln(y^+)/κ + B`` for
``y^+ → ∞``. [`friction_velocity`](@ref) solves
``u^+ y^+(u^+) = Uh/ν`` for ``u^+`` by a safeguarded Newton iteration
(the left side is convex and increasing, so Newton from above the root
converges monotonically; a step that leaves the bracket bisects).

The defaults are the classical constants. The instance fitted to the
channel DNS of Lee & Moser (2015, ``Re_τ ≈ 5200``), κ = 0.384 and
B = 4.27, is [`LEE_MOSER_2015`](@ref).
"""
Base.@kwdef struct SpaldingLaw{T<:AbstractFloat} <: AbstractWallLaw
    κ::T = 0.41
    B::T = 5.2
end

"""
    LEE_MOSER_2015

`SpaldingLaw(κ = 0.384, B = 4.27)`: the log-law constants of the channel
DNS of Lee & Moser (2015) at ``Re_τ ≈ 5200``.
"""
const LEE_MOSER_2015 = SpaldingLaw(κ = 0.384, B = 4.27)

# e^x − Σ_{n<m} xⁿ/n! (m = 3 or 4), without the cancellation of the
# direct form at small x: the Taylor tail for |x| < 1/2.
function _exp_tail(x::T, m::Int) where {T<:AbstractFloat}
    if abs(x) < T(0.5)
        term = x^m / T(factorial(m))
        s = term
        n = m
        while abs(term) > eps(T) * abs(s) / 4
            n += 1
            term *= x / n
            s += term
        end
        return s
    end
    s = expm1(x) - x - x^2 / 2
    return m == 3 ? s : s - x^3 / 6
end

"""
    spalding_yplus(law::SpaldingLaw, uplus) -> y⁺

The explicit direction of Spalding's formula: ``y^+`` of a given
``u^+ ≥ 0``.
"""
function spalding_yplus(law::SpaldingLaw, up::Real)
    x = law.κ * up
    return up + exp(-law.κ * law.B) * _exp_tail(float(x), 4)
end

_spalding_dyplus(law::SpaldingLaw, up) =
    1 + law.κ * exp(-law.κ * law.B) * _exp_tail(float(law.κ * up), 3)

# Safeguarded Newton for an increasing f on (lo, hi): `fg(x) -> (f, f′)`.
# A step outside the bracket, or a non-finite value (overflow of e^{κu⁺}
# far above the root), bisects.
function _increasing_root(fg, lo::T, hi::T, x::T; maxiter::Int = 200) where {T<:AbstractFloat}
    for _ in 1:maxiter
        f, g = fg(x)
        if !isfinite(f) || f > 0
            hi = x
        elseif f < 0
            lo = x
        else
            return x
        end
        xn = (isfinite(f) && g > 0) ? x - f / g : (lo + hi) / 2
        (lo < xn < hi) || (xn = (lo + hi) / 2)
        abs(xn - x) <= 2 * eps(T) * abs(xn) && return xn
        x = xn
    end
    return x
end

"""
    spalding_uplus(law::SpaldingLaw, yplus) -> u⁺

The inverse of [`spalding_yplus`](@ref): ``u^+`` at a given ``y^+ ≥ 0``,
by safeguarded Newton (``y^+(u^+)`` is convex and increasing, and
``u^+ ≤ y^+`` brackets the root).
"""
function spalding_uplus(law::SpaldingLaw, yp::Real)
    Y = float(yp)
    Y > 0 || return zero(Y)
    x0 = Y < 1 ? Y : min(Y, log(Y) / law.κ + law.B)
    return _increasing_root(u -> (spalding_yplus(law, u) - Y, _spalding_dyplus(law, u)),
                            zero(Y), Y, x0)
end

function friction_velocity(law::SpaldingLaw, U::Real, h::Real; ν = nothing)
    ν === nothing && throw(ArgumentError("SpaldingLaw: the kinematic viscosity ν is required"))
    a = float(abs(U))
    a > 0 || return zero(a)
    Re = a * h / ν                      # = u⁺ y⁺
    # u⁺ y⁺(u⁺) is convex and increasing; y⁺ ≥ u⁺ gives u⁺ ≤ √Re
    hi = sqrt(Re)
    x0 = hi
    if Re > 1                           # start near the log law: u⁺ = ln(Re/u⁺)/κ + B
        x0 = 10 * one(Re)
        for _ in 1:3
            x0 = max(log(Re / x0) / law.κ + law.B, one(Re))
        end
        x0 = min(x0, hi)
    end
    fg(up) = (y = spalding_yplus(law, up);
              (up * y - Re, y + up * _spalding_dyplus(law, up)))
    up = _increasing_root(fg, zero(Re), hi, x0)
    return a / up
end

function wall_drag(law::SpaldingLaw, U::Real, h::Real; ν = nothing)
    ν === nothing && throw(ArgumentError("SpaldingLaw: the kinematic viscosity ν is required"))
    a = float(abs(U))
    a > 0 || return float(ν / h)        # the viscous limit τ_w = νU/h
    return friction_velocity(law, a, h; ν = ν)^2 / a
end

# =========================================================================
# Rough walls: neutral log law and Monin–Obukhov
# =========================================================================

"""
    RoughLogLaw(z0; c = DEFAULT_SURFACE_CONSTANTS)

The neutral log law over a fully rough wall of roughness length `z0`
(m):

```math
u_τ = \\frac{κ U}{\\ln(h/z_0)},
```

with κ **delegated** to `SurfaceExchange.von_karman(c)`, so the wall
model and the surface-layer solver cannot disagree about it. Requires
``h > z_0``. Independent of ν.
"""
struct RoughLogLaw{T<:AbstractFloat, C<:SurfaceConstants} <: AbstractWallLaw
    z0::T
    c::C
end
RoughLogLaw(z0::Real; c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS) =
    RoughLogLaw(float(z0), c)

function friction_velocity(law::RoughLogLaw, U::Real, h::Real; ν = nothing)
    h > law.z0 || throw(ArgumentError("RoughLogLaw: need h > z0, got h = $h, z0 = $(law.z0)"))
    return von_karman(law.c) * abs(U) / log(h / law.z0)
end

"""
    MOSTWallLaw(sf, z0m, L; U_min = 0.1, c = DEFAULT_SURFACE_CONSTANTS)
    MOSTWallLaw(z0m, L; kwargs...)          # sf = BusingerDyerBeljaarsHoltslag()

The rough-wall law with the Monin–Obukhov stability correction for a
given Obukhov length `L` (m; `Inf` neutral, `> 0` stable):

```math
u_τ = \\frac{κ\\,\\max(U, U_{min})}{\\ln(h/z_{0m}) - ψ_m(h/L) + ψ_m(z_{0m}/L)}.
```

Everything here is **delegated** to `SurfaceExchange`: κ is its
`von_karman`, and the denominator is the stability-corrected log term
of `solve_surface_layer` evaluated with the stability functions `sf`
(their `psi_m`), so for the same ζ = h/L, wind and roughness the
result is `solve_surface_layer(...).u_star` bit for bit. `U_min` is the
minimum wind of that solver (default the same 0.1 m s⁻¹).

`L` is held: the law is the momentum half of Monin–Obukhov similarity
with the buoyancy of the last surface-layer solve.
[`matching_obukhov_length`](@ref) recovers that `L` from a solve's
`u_*`.
"""
struct MOSTWallLaw{S<:AbstractStabilityFunctions, T<:AbstractFloat, C<:SurfaceConstants} <: AbstractWallLaw
    sf::S
    z0m::T
    L::T
    U_min::T
    c::C
end
function MOSTWallLaw(sf::AbstractStabilityFunctions, z0m::Real, L::Real;
                     U_min::Real = 0.1, c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS)
    z0, LL, Um = promote(float(z0m), float(L), float(U_min))
    return MOSTWallLaw(sf, z0, LL, Um, c)
end
MOSTWallLaw(z0m::Real, L::Real; kwargs...) =
    MOSTWallLaw(BusingerDyerBeljaarsHoltslag(), z0m, L; kwargs...)

# The stability-corrected log term of SurfaceExchange (one source of truth).
_most_Φm(law::MOSTWallLaw, h) = SurfaceExchange._Φm(law.sf, h / law.L, h, law.z0m)

function friction_velocity(law::MOSTWallLaw, U::Real, h::Real; ν = nothing)
    h > law.z0m || throw(ArgumentError("MOSTWallLaw: need h > z0m, got h = $h, z0m = $(law.z0m)"))
    # the effective wind of solve_surface_layer (no gustiness), the same operations
    Ue = max(sqrt(float(U)^2), law.U_min)
    return von_karman(law.c) * Ue / _most_Φm(law, h)
end

wall_drag(law::MOSTWallLaw, U::Real, h::Real; ν = nothing) =
    friction_velocity(law, U, h)^2 / max(abs(float(U)), law.U_min)

"""
    matching_obukhov_length(sf, u_star, U, h, z0m; U_min = 0.1,
                            c = DEFAULT_SURFACE_CONSTANTS,
                            ζ_min = -1000, ζ_max = 1000) -> L

The Obukhov length `L` for which [`MOSTWallLaw`](@ref)`(sf, z0m, L)`
gives the friction velocity `u_star` at wind `U` and height `h`: the
root ζ = h/L of ``\\ln(h/z_{0m}) - ψ_m(ζ) + ψ_m(ζ z_{0m}/h) = κ\\max(U,
U_{min})/u_*`` (increasing in ζ; solved by the same Illinois iteration
as `solve_surface_layer`, over the same `[ζ_min, ζ_max]`, clamped there).
`Inf` when the stratification is neutral.

This is how a surface step that hands back only ``u_*`` (the stress of
`SurfaceExchange.SurfaceFluxes`) hands its stability to the wall model:
with the wind of that step, `MOSTWallLaw(sf, z0m, L)` reproduces
`u_star` to round-off.
"""
function matching_obukhov_length(sf::AbstractStabilityFunctions, u_star::Real, U::Real,
                                 h::Real, z0m::Real; U_min::Real = 0.1,
                                 c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS,
                                 ζ_min::Real = -1000.0, ζ_max::Real = 1000.0)
    u_star > 0 || throw(ArgumentError("matching_obukhov_length: u_star must be > 0"))
    T = float(promote_type(typeof(u_star), typeof(U), typeof(h), typeof(z0m)))
    Φt = T(von_karman(c)) * max(sqrt(T(U)^2), T(U_min)) / T(u_star)
    F(ζ) = SurfaceExchange._Φm(sf, ζ, T(h), T(z0m)) - Φt
    F0 = F(zero(T))
    F0 == 0 && return T(Inf)
    a, b = F0 > 0 ? (T(ζ_min), zero(T)) : (zero(T), T(ζ_max))
    fa, fb = F0 > 0 ? (F(a), F0) : (F0, F(b))
    fa >= 0 && return T(h) / a                      # beyond the admissible range: clamp
    fb <= 0 && return T(h) / b
    ζ, _, _ = SurfaceExchange._illinois(F, a, b, fa, fb, 200)
    return ζ == 0 ? T(Inf) : T(h) / ζ
end

"""
    MOSTFluxWallLaw(sf, z0m, Q, θ_ref; U_min = 0.1, c = DEFAULT_SURFACE_CONSTANTS)
    MOSTFluxWallLaw(z0m, Q, θ_ref; kwargs...)   # sf = BusingerDyerBeljaarsHoltslag()

The rough-wall Monin–Obukhov law for a **prescribed** surface buoyancy
flux: `Q` is the upward kinematic flux ``\\overline{w'θ'}`` (K m s⁻¹,
negative = surface cooling) and `θ_ref` (K) the reference potential
temperature of ``g/θ_{ref}``. Unlike [`MOSTWallLaw`](@ref), which holds
the Obukhov length, the stability follows the wind: ``L = -u_τ^3
θ_{ref}/(κ g Q)`` with the ``u_τ`` of the law itself, so

```math
u_τ = \\frac{κ\\,\\max(U, U_{min})}{\\ln(h/z_{0m}) - ψ_m(h/L) + ψ_m(z_{0m}/L)},
\\qquad L = -\\frac{u_τ^3 θ_{ref}}{κ g Q},
```

solved by `SurfaceExchange.flux_friction_velocity` (one source of
truth). Under cooling the relation has a turbulent solution only while
``-Q`` is at most the maximum sustainable heat flux of the wind at
height h (`SurfaceExchange.max_sustainable_heat_flux`, Van de Wiel et
al. 2012); then ``u_τ`` is on the weakly stable branch. Beyond it the
law gives ``u_τ = 0``: the wall stops exerting stress — the collapse of
the surface layer. This is the wall law of the flux-cooled half channel
(`half_channel_case(surface_heat_flux = Q)`, P58-e, #193).
"""
struct MOSTFluxWallLaw{S<:AbstractStabilityFunctions, T<:AbstractFloat, C<:SurfaceConstants} <: AbstractWallLaw
    sf::S
    z0m::T
    Q::T
    θ_ref::T
    U_min::T
    c::C
end
function MOSTFluxWallLaw(sf::AbstractStabilityFunctions, z0m::Real, Q::Real, θ_ref::Real;
                         U_min::Real = 0.1, c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS)
    θ_ref > 0 || throw(ArgumentError("MOSTFluxWallLaw: θ_ref must be > 0, got $θ_ref"))
    z0, q, θ, Um = promote(float(z0m), float(Q), float(θ_ref), float(U_min))
    return MOSTFluxWallLaw(sf, z0, q, θ, Um, c)
end
MOSTFluxWallLaw(z0m::Real, Q::Real, θ_ref::Real; kwargs...) =
    MOSTFluxWallLaw(BusingerDyerBeljaarsHoltslag(), z0m, Q, θ_ref; kwargs...)

function friction_velocity(law::MOSTFluxWallLaw, U::Real, h::Real; ν = nothing)
    h > law.z0m || throw(ArgumentError("MOSTFluxWallLaw: need h > z0m, got h = $h, z0m = $(law.z0m)"))
    Ue = max(abs(float(U)), law.U_min)
    return SurfaceExchange.flux_friction_velocity(law.sf, Ue, h, law.z0m, law.Q, law.θ_ref;
                                                  c = law.c)
end

wall_drag(law::MOSTFluxWallLaw, U::Real, h::Real; ν = nothing) =
    friction_velocity(law, U, h)^2 / max(abs(float(U)), law.U_min)
