module LargeScaleForcing

# Era VI, Phase 58 (#171) — prescribed large-scale (case) forcing.
#
# The forcing a single-column or LES case is driven by when the larger
# scales are not resolved: the geostrophic pressure gradient, nudging of
# the horizontal mean toward a target wind, prescribed large-scale
# advective tendencies and a prescribed surface-temperature series.
# #248 (N-parent-c) generalised the nudging to the plane mean of any of
# u, v, θ, qv (`ProfileNudging`, of which `GeostrophicNudging` is now a
# constructor) and added the state-dependent large-scale subsidence
# (`Subsidence`).
# GABLS1-4, CASES-99 and (Phase 67) ERA5-forced cases are built from
# these types; this module only defines them and applies them.
#
# Every forcing quantity is a constant, a height profile or a height-time
# table (`ForcingTable`), linearly interpolated in both and held constant
# beyond the table ends. A forcing is applied over a step `[t, t + Δt]`
# with its exact step mean (`forcing_mean`), so a piecewise-linear
# table is integrated exactly in time by any explicit step.

import ..SurfaceExchange: surface_temperature
using ..StretchedGrids: StretchedGrid
using ..MoistThermodynamics: ThermodynamicConstants, DEFAULT_CONSTANTS, PhasePartition,
                             exner_function, virtual_temperature,
                             specific_humidity_from_relative_humidity

export ForcingTable, forcing_value, forcing_mean,
       AbstractLargeScaleForcing, GeostrophicPressureGradient, GeostrophicNudging,
       AdvectiveTendencies, SurfaceTemperatureSeries, apply_forcing!,
       Sounding, SoundingLaunch, read_sounding, sounding_profiles, sounding_time
export ProfileNudging, NudgingWeight, nudging_weight, Subsidence, tendency_integrals

# =========================================================================
# Tables and their interpolation
# =========================================================================

"""
    ForcingTable(z, t, values)
    ForcingTable(z, values::AbstractVector)

A forcing quantity tabulated at heights `z` (m, strictly increasing) and
times `t` (s, strictly increasing): `values[k, n]` is the value at
`(z[k], t[n])`. The second form is a time-independent profile.

The table is interpolated linearly in `z` and in `t` and held at its end
values outside the tabulated range (no extrapolation), the convention of
the GABLS and DICE forcing files. See [`forcing_value`](@ref) and
[`forcing_mean`](@ref).
"""
struct ForcingTable{T<:AbstractFloat}
    z::Vector{T}
    t::Vector{T}
    values::Matrix{T}
    function ForcingTable(z::AbstractVector, t::AbstractVector, values::AbstractMatrix)
        T = float(promote_type(eltype(z), eltype(t), eltype(values)))
        size(values) == (length(z), length(t)) ||
            throw(ArgumentError("ForcingTable: values is $(size(values)), expected (length(z), length(t)) = $((length(z), length(t)))"))
        (isempty(z) || isempty(t)) && throw(ArgumentError("ForcingTable: empty z or t"))
        (issorted(z; lt = <=) && allunique(z)) || throw(ArgumentError("ForcingTable: z must be strictly increasing"))
        (issorted(t; lt = <=) && allunique(t)) || throw(ArgumentError("ForcingTable: t must be strictly increasing"))
        return new{T}(Vector{T}(z), Vector{T}(t), Matrix{T}(values))
    end
end

ForcingTable(z::AbstractVector, values::AbstractVector) =
    ForcingTable(z, [0.0], reshape(collect(values), :, 1))

# (i, w): the value at x is (1 - w)·p[i] + w·p[i+1] (i+1 clamped), held
# at the ends.
@inline function _bracket(x::AbstractVector, xq::Real)
    n = length(x)
    (n == 1 || xq <= x[1]) && return 1, zero(eltype(x))
    xq >= x[n] && return n - 1, one(eltype(x))
    i = searchsortedlast(x, xq)
    return i, (xq - x[i]) / (x[i+1] - x[i])
end

@inline function _lerp(p, i::Int, w)
    n = length(p)
    return n == 1 ? p[1] : (1 - w) * p[i] + w * p[min(i + 1, n)]
end

# Antiderivative G(x) = ∫_{t[1]}^{x} p dt of the piecewise-linear p(t) held
# constant beyond the ends.
function _antiderivative(t::AbstractVector, p::AbstractVector, x::Real)
    n = length(t)
    x <= t[1] && return (x - t[1]) * p[1]
    G = zero(promote_type(eltype(p), typeof(x)))
    for i in 1:n-1
        if x <= t[i+1]
            s = x - t[i]
            slope = (p[i+1] - p[i]) / (t[i+1] - t[i])
            return G + s * (p[i] + s * slope / 2)
        end
        G += (t[i+1] - t[i]) * (p[i] + p[i+1]) / 2
    end
    return G + (x - t[n]) * p[n]
end

# Mean of p(t) over [t0, t1] (the value at t0 when t1 == t0).
function _series_mean(t::AbstractVector, p::AbstractVector, t0::Real, t1::Real)
    if t1 == t0
        i, w = _bracket(t, t0)
        return _lerp(p, i, w)
    end
    return (_antiderivative(t, p, t1) - _antiderivative(t, p, t0)) / (t1 - t0)
end

"""
    forcing_value(x, z, t)

The forcing quantity `x` at height `z` (m) and time `t` (s): `x` itself
for a number, the bilinear interpolant for a [`ForcingTable`](@ref)
(held at the table ends), `x(z, t)` for a function.
"""
forcing_value(x::Real, z, t) = x
forcing_value(f::Function, z, t) = f(z, t)
function forcing_value(tab::ForcingTable, z, t)
    k, wz = _bracket(tab.z, z)
    n, wt = _bracket(tab.t, t)
    V = tab.values
    nz, nt = size(V)
    k2 = min(k + 1, nz); n2 = min(n + 1, nt)
    a = nz == 1 ? V[1, n] : (1 - wz) * V[k, n] + wz * V[k2, n]
    nt == 1 && return a
    b = nz == 1 ? V[1, n2] : (1 - wz) * V[k, n2] + wz * V[k2, n2]
    return (1 - wt) * a + wt * b
end

"""
    forcing_mean(x, z, t0, t1)

The time mean of the forcing quantity `x` at height `z` over the step
`[t0, t1]`. For a [`ForcingTable`](@ref) it is exact (the table is
piecewise linear in `t`), so that summing `Δt · forcing_mean` over steps
reproduces ``\\int F\\,dt`` to round-off whatever the step size; for a
function it is Simpson's rule; a number is its own mean. With
`t1 == t0` it is [`forcing_value`](@ref).
"""
forcing_mean(x::Real, z, t0, t1) = x
forcing_mean(f::Function, z, t0, t1) =
    t1 == t0 ? f(z, t0) : (f(z, t0) + 4f(z, (t0 + t1) / 2) + f(z, t1)) / 6
function forcing_mean(tab::ForcingTable, z, t0, t1)
    k, wz = _bracket(tab.z, z)
    nz = size(tab.values, 1)
    a = _series_mean(tab.t, view(tab.values, k, :), t0, t1)
    nz == 1 && return a
    b = _series_mean(tab.t, view(tab.values, min(k + 1, nz), :), t0, t1)
    return (1 - wz) * a + wz * b
end

# =========================================================================
# Forcings
# =========================================================================

"""
    AbstractLargeScaleForcing

Supertype of the prescribed large-scale forcings. Each one implements
[`apply_forcing!`](@ref)`(sim, forcing, t, Δt)`, which adds its
tendency to the simulation's tendency fields (`sim.tu`, `sim.tv`,
`sim.tθ`, …) and is meant to be called from the `body_force!` hook of
`step!`. A `Tuple` of forcings is applied in order.
"""
abstract type AbstractLargeScaleForcing end

"""
    GeostrophicPressureGradient(U_g, V_g = 0)

The large-scale pressure gradient in geostrophic balance with the wind
`(U_g, V_g)` (m/s):

```math
\\left.\\frac{\\partial u}{\\partial t}\\right|_{pg} = -f V_g, \\qquad
\\left.\\frac{\\partial v}{\\partial t}\\right|_{pg} = +f U_g,
```

with `f = sim.f_coriolis`, so that together with the Coriolis term
(`+f v`, `-f u`) the forced, frictionless flow has the steady state
`(u, v) = (U_g, V_g)` about which ageostrophic departures perform
inertial oscillations of period ``2π/f`` (Blackadar 1957), and a
viscous flow over a no-slip wall forms the Ekman spiral (Ekman 1905).

`U_g` and `V_g` are each a number, a height profile or a height-time
[`ForcingTable`](@ref) (or a function `(z, t) -> value`).

References: Holton & Hakim (2013), *An Introduction to Dynamic
Meteorology*, 5th ed., §3.2 and §8.4; Blackadar (1957), Bull. Amer.
Meteor. Soc. 38, 283–290; Ekman (1905), Ark. Mat. Astron. Fys. 2(11).
"""
struct GeostrophicPressureGradient{U,V} <: AbstractLargeScaleForcing
    U_g::U
    V_g::V
end
GeostrophicPressureGradient(U_g) = GeostrophicPressureGradient(U_g, 0.0)

"""
    NudgingWeight(z_n; factor = 1, depth = 0)

The height weight ``w(z) \\in [0, 1]`` of a [`ProfileNudging`](@ref):
zero up to ``z_0 = \\mathrm{factor}\\cdot z_n``, one above
``z_0 + \\mathrm{depth}``, and the smooth ramp
``\\sin^2\\!\\big(\\tfrac{π}{2}(z - z_0)/\\mathrm{depth}\\big)`` in between
(the profile of the core's Rayleigh sponge; `depth = 0` is a step).

`z_n` is a height (m) — "nudge above ``z_n``" — or any callable
`sim -> height`, evaluated once per call of [`apply_forcing!`](@ref) on
the current state. The second form is the "multiple of a diagnosed
boundary-layer depth" of LES case set-ups that nudge only the free
troposphere (Heinze et al. 2017, ACP 17, 7083, §2.3), e.g.
`NudgingWeight(sim -> zi(sim); factor = 1.5, depth = 200)` with `zi` a
boundary-layer-height diagnostic.

Any other callable `z -> w` can be passed as the `weight` of a
`ProfileNudging` directly.
"""
struct NudgingWeight{Z,T<:AbstractFloat}
    z_n::Z
    factor::T
    depth::T
    function NudgingWeight(z_n::Z, factor::Real, depth::Real) where {Z}
        factor > 0 || throw(ArgumentError("NudgingWeight: factor must be positive, got $factor"))
        depth >= 0 || throw(ArgumentError("NudgingWeight: depth must be non-negative, got $depth"))
        T = float(promote_type(typeof(factor), typeof(depth)))
        return new{Z,T}(z_n, T(factor), T(depth))
    end
end
NudgingWeight(z_n; factor::Real = 1.0, depth::Real = 0.0) = NudgingWeight(z_n, factor, depth)

@inline function _ramp(z, z0, depth)
    z <= z0 && return zero(float(z))
    (depth == 0 || z >= z0 + depth) && return one(float(z))
    s = sin(π / 2 * (z - z0) / depth)
    return s * s
end

"""
    nudging_weight(W::NudgingWeight, z, sim) -> w ∈ [0, 1]

The weight of `W` at height `z` (m) for the state `sim` (used when
`W.z_n` is a callable).
"""
nudging_weight(W::NudgingWeight, z, sim) =
    _ramp(z, W.factor * _resolve_height(W.z_n, sim), W.depth)

_resolve_height(z::Real, sim) = z
_resolve_height(f, sim) = f(sim)

# the per-call weight function: `nothing` (w ≡ 1, the multiplication is
# skipped so that unweighted nudging is bit for bit the #171 code), the
# ramp of a NudgingWeight at its height for this state, or the user's
# callable z -> w
_weight_fn(::Nothing, sim) = nothing
function _weight_fn(W::NudgingWeight, sim)
    z0 = W.factor * _resolve_height(W.z_n, sim)
    d = W.depth
    return z -> _ramp(z, z0, d)
end
_weight_fn(f, sim) = f

const _LSF_FIELDS = (:u, :v, :θ, :qv)

"""
    ProfileNudging(targets::NamedTuple, τ; weight = nothing)

Relaxation of the **plane (horizontal) mean** of any of `u`, `v`, `θ`,
`qv` toward a reference on the time scale `τ` (s):

```math
\\left.\\frac{\\partial φ}{\\partial t}\\right|_{n} =
-\\,w(z)\\,\\frac{\\langle φ\\rangle_{xy}(z) - φ_{ref}(z, t)}{τ},
```

added uniformly over each model level, so the fluctuations
``φ - \\langle φ\\rangle`` are not touched. `targets` names the fields and
their references, e.g. `(θ = θ_profile, qv = q_profile)`; each reference
is any forcing quantity (a number, a profile `ForcingTable(z, values)`, a
height-time [`ForcingTable`](@ref) or a function `(z, t) -> value`) and
enters as its exact step mean ([`forcing_mean`](@ref)) over
`[t, t + Δt]`. `weight` is `nothing` (``w ≡ 1``), a
[`NudgingWeight`](@ref) ("above ``z_n``" or above a multiple of a
diagnosed boundary-layer depth, with a smooth ramp) or any callable
`z -> w`.

This is the nudging of the LES large-scale forcing frameworks: DALES
(Heus et al. 2010, GMD 3, 415, Eq. 92), the HOPE PALM runs of Heinze et
al. (2017, ACP 17, 7083, Eq. 4; τ = 6 h for θ and q) and MicroHH (van
Heerwaarden et al. 2017, GMD 10, 3145). With a forward-Euler step the
departure of a level mean from a constant reference decays by exactly
``1 - wΔt/τ`` per step.

[`apply_forcing!`](@ref) returns the volume integrals
``\\int (\\partial φ/\\partial t)_n\\,dV`` it added, a `NamedTuple` keyed
like `targets` (units of φ times m³ s⁻¹); they are also kept for
[`tendency_integrals`](@ref). [`GeostrophicNudging`](@ref) is
`ProfileNudging((u = U_g, v = V_g), τ)`.
"""
struct ProfileNudging{N<:NamedTuple,T<:Real,W} <: AbstractLargeScaleForcing
    targets::N
    τ::T
    weight::W
    last::Base.RefValue{NamedTuple}
    function ProfileNudging(targets::N, τ::T; weight::W = nothing) where {N<:NamedTuple,T<:Real,W}
        τ > 0 || throw(ArgumentError("ProfileNudging: τ must be positive, got $τ"))
        isempty(targets) && throw(ArgumentError("ProfileNudging: no target fields"))
        for k in keys(targets)
            k in _LSF_FIELDS ||
                throw(ArgumentError("ProfileNudging: cannot nudge :$k (supported: $(_LSF_FIELDS))"))
        end
        return new{N,T,W}(targets, τ, weight, Ref{NamedTuple}(NamedTuple()))
    end
end

"""
    GeostrophicNudging(U_g, V_g, τ) -> ProfileNudging

Rayleigh relaxation of the **horizontal mean** wind toward `(U_g, V_g)`
on the time scale `τ` (s):

```math
\\left.\\frac{\\partial u}{\\partial t}\\right|_{n} = -\\frac{\\langle u\\rangle_{xy} - U_g}{τ},
\\qquad
\\left.\\frac{\\partial v}{\\partial t}\\right|_{n} = -\\frac{\\langle v\\rangle_{xy} - V_g}{τ},
```

applied uniformly over each level, so turbulent fluctuations are not
damped. This is a nudging, not a pressure gradient: it has no Coriolis
counterpart, suppresses inertial oscillations and the nocturnal low-level
jet, and is the behaviour of the Berlin case before #171. Since #248 it
is a constructor of [`ProfileNudging`](@ref), `ProfileNudging((u = U_g,
v = V_g), τ)`, not a second implementation (bit for bit the former
type). `U_g`, `V_g` take the same forms as in
[`GeostrophicPressureGradient`](@ref).
"""
function GeostrophicNudging(U_g, V_g, τ::Real)
    τ > 0 || throw(ArgumentError("GeostrophicNudging: τ must be positive, got $τ"))
    return ProfileNudging((u = U_g, v = V_g), τ)
end

"""
    Subsidence(w_s; fields = (:θ, :qv), form = :mean_gradient)

Large-scale subsidence (vertical advection by the prescribed large-scale
vertical velocity ``w_s(z, t)``, m s⁻¹, negative downward) of the fields
`fields` (any of `:u`, `:v`, `:θ`, `:qv`):

```math
\\left.\\frac{\\partial φ}{\\partial t}\\right|_{s} =
-\\,w_s(z, t)\\,\\frac{\\partial \\langle φ\\rangle_{xy}}{\\partial z}
\\quad (\\texttt{form = :mean\\_gradient}), \\qquad
-\\,w_s(z, t)\\,\\frac{\\partial φ}{\\partial z}
\\quad (\\texttt{form = :local}).
```

`:mean_gradient` (the default; DALES, Heus et al. 2010, GMD 3, 415,
Eq. 91; MicroHH, van Heerwaarden et al. 2017, GMD 10, 3145) uses the
centred gradient of the plane mean,
``(⟨φ⟩_{k+1} - ⟨φ⟩_{k-1})/(z_{k+1} - z_{k-1})``, and adds the tendency
uniformly per level, so it moves the mean profile and leaves the
fluctuations alone; it is exact for a linear profile on any grid.
`:local` (Heinze et al. 2017, ACP 17, 7083, Eq. 3) uses each column's own
gradient, first-order upwind with respect to ``w_s``. On the bottom and
top levels the gradient is the one-sided difference toward the interior.
A centred `:mean_gradient` needs a time scheme stable for centred
advection (SSP-RK3); with forward Euler use `:local` or a small
``|w_s|Δt/Δz``.

`w_s` is any forcing quantity (number, profile, height-time
[`ForcingTable`](@ref), function `(z, t) -> value`), applied as its step
mean. The default `fields = (:θ, :qv)` subsides `qv` only when the
simulation carries moisture; a field named explicitly that the
simulation does not carry is an error.

Unlike the prescribed [`AdvectiveTendencies`](@ref), the tendency
depends on the state and does not conserve ``\\int φ\\,dV``:
[`apply_forcing!`](@ref) returns the volume integrals
``\\int (\\partial φ/\\partial t)_s\\,dV`` it added, a `NamedTuple` keyed
by field (also kept for [`tendency_integrals`](@ref)), for the ledgers.
"""
struct Subsidence{W} <: AbstractLargeScaleForcing
    w_s::W
    fields::Tuple{Vararg{Symbol}}
    explicit::Bool
    form::Symbol
    buf::Vector{Float64}
    last::Base.RefValue{NamedTuple}
    function Subsidence(w_s::W, fields, explicit::Bool, form::Symbol) where {W}
        form in (:mean_gradient, :local) ||
            throw(ArgumentError("Subsidence: form must be :mean_gradient or :local, got :$form"))
        fs = Tuple(Symbol.(fields))
        isempty(fs) && throw(ArgumentError("Subsidence: no fields"))
        for k in fs
            k in _LSF_FIELDS ||
                throw(ArgumentError("Subsidence: cannot subside :$k (supported: $(_LSF_FIELDS))"))
        end
        return new{W}(w_s, fs, explicit, form, Float64[], Ref{NamedTuple}(NamedTuple()))
    end
end
Subsidence(w_s; fields = nothing, form::Symbol = :mean_gradient) =
    fields === nothing ? Subsidence(w_s, (:θ, :qv), false, form) :
                         Subsidence(w_s, fields, true, form)

"""
    AdvectiveTendencies(z, t, dθdt, dqdt, dudt, dvdt)
    AdvectiveTendencies(; θ = nothing, q = nothing, u = nothing, v = nothing)

Prescribed large-scale advective tendencies of potential temperature
(K/s), specific humidity (kg kg⁻¹ s⁻¹) and the horizontal wind (m s⁻²).
In the first form each of `dθdt`, `dqdt`, `dudt`, `dvdt` is a
`length(z) × length(t)` matrix (or `nothing` for none) on the common
height-time grid; in the second each is any forcing quantity (number,
[`ForcingTable`](@ref), function of `(z, t)`).

The tendency is added uniformly over each model level. With no surface
flux it changes ``\\int θ\\,dV`` by exactly ``\\iint (∂θ/∂t)_{adv}\\,dV\\,dt``
(the step mean is used, see [`forcing_mean`](@ref)). Large-scale
subsidence, which depends on the model state, is not part of it: it is
[`Subsidence`](@ref).

Reference: the GABLS2 and DICE intercomparison forcing (Svensson et al.
2011, Bound.-Layer Meteor. 140, 177–206; Best et al. 2013, DICE
protocol).
"""
struct AdvectiveTendencies{A,B,C,D} <: AbstractLargeScaleForcing
    θ::A
    q::B
    u::C
    v::D
end
AdvectiveTendencies(; θ = nothing, q = nothing, u = nothing, v = nothing) =
    AdvectiveTendencies(θ, q, u, v)

_table(z, t, ::Nothing) = nothing
_table(z, t, m::AbstractMatrix) = ForcingTable(z, t, m)
AdvectiveTendencies(z::AbstractVector, t::AbstractVector, dθdt, dqdt, dudt, dvdt) =
    AdvectiveTendencies(_table(z, t, dθdt), _table(z, t, dqdt),
                        _table(z, t, dudt), _table(z, t, dvdt))

"""
    SurfaceTemperatureSeries(t, T_s)

A prescribed surface (skin) temperature `T_s` (K) at times `t` (s),
linearly interpolated and held at the end values (GABLS1: 265 K cooling
at 0.25 K/h; GABLS2 / CASES-99: the observed series). Read with
`surface_temperature(series, t)`, a method of the
`SurfaceExchange.surface_temperature` generic. This is not a surface
model: it is the boundary value a wall law (P57-a, #169) or a surface
step is given.
"""
struct SurfaceTemperatureSeries{T<:AbstractFloat}
    t::Vector{T}
    T_s::Vector{T}
    function SurfaceTemperatureSeries(t::AbstractVector, T_s::AbstractVector)
        length(t) == length(T_s) && !isempty(t) ||
            throw(ArgumentError("SurfaceTemperatureSeries: t and T_s must be non-empty and of equal length"))
        (issorted(t; lt = <=) && allunique(t)) ||
            throw(ArgumentError("SurfaceTemperatureSeries: t must be strictly increasing"))
        T = float(promote_type(eltype(t), eltype(T_s)))
        return new{T}(Vector{T}(t), Vector{T}(T_s))
    end
end

"""
    surface_temperature(s::SurfaceTemperatureSeries, t) -> T_s (K)

The prescribed surface temperature at time `t` (s), linear between the
tabulated times and held at the end values.
"""
function surface_temperature(s::SurfaceTemperatureSeries, t::Real)
    i, w = _bracket(s.t, t)
    return _lerp(s.T_s, i, w)
end

# =========================================================================
# Application to a simulation
# =========================================================================

# Height (m) of the cell centre of level k — where u, v (FaceX/FaceY)
# and the scalars live on the C-grid — and the thickness of that cell,
# read from the mesh: the affine formula of a `RegularGrid` (unchanged
# from #171, so uniform runs are bit for bit), the stored metric of a
# `StretchedGrid`.
@inline _zc(m, k) = m.z0 + (k - 0.5) * m.dz
@inline _zc(m::StretchedGrid, k) = m.zc[m.Hz+k]
@inline _dzc(m, k) = m.dz
@inline _dzc(m::StretchedGrid, k) = m.dzc[m.Hz+k]

"""
    apply_forcing!(sim, forcing, t, Δt) -> sim

Add the tendency of `forcing` (an [`AbstractLargeScaleForcing`](@ref)
or a `Tuple` of them) over the step `[t, t + Δt]` to the tendency fields
of `sim`. Call it from the `body_force!` hook of `step!`, e.g.
`step!(sim, dt; body_force! = (s, Δt) -> apply_forcing!(s, F, s.t, Δt))`.
`sim` is duck-typed on `mesh`, `u`, `v`, `tu`, `tv`, `tθ` and
`f_coriolis`, so any dynamical core of the package works.

For [`AdvectiveTendencies`](@ref) the humidity tendency goes to the
keyword `tq` (a cell-centred field; default `sim.tqv` when the
simulation carries moisture). A humidity tendency with no field to
receive it is an error.
"""
function apply_forcing! end

function apply_forcing!(sim, F::Tuple, t, Δt; kw...)
    for f in F
        apply_forcing!(sim, f, t, Δt; kw...)
    end
    return sim
end

function apply_forcing!(sim, F::GeostrophicPressureGradient, t, Δt; kw...)
    m = sim.mesh
    tu, tv = sim.tu.data, sim.tv.data
    T = eltype(tu)
    f = sim.f_coriolis
    Nx, Ny, Nz = m.Nx, m.Ny, m.Nz
    Hx, Hy, Hz = m.Hx, m.Hy, m.Hz
    t1 = t + Δt
    @inbounds for k in 1:Nz
        z = _zc(m, k)
        fu = T(-(f * forcing_mean(F.V_g, z, t, t1)))
        fv = T(f * forcing_mean(F.U_g, z, t, t1))
        for j in 1:Ny, i in 1:Nx
            tu[Hx+i, Hy+j, Hz+k] += fu
            tv[Hx+i, Hy+j, Hz+k] += fv
        end
    end
    return sim
end

_has_moisture(sim) = hasproperty(sim, :qv) && hasproperty(sim, :tqv) &&
                     (!hasproperty(sim, :has_moisture) || sim.has_moisture)

# (state field, tendency field) of a forcing target
function _state_tendency(sim, name::Symbol, who)
    name === :u && return sim.u, sim.tu
    name === :v && return sim.v, sim.tv
    name === :θ && return sim.θ, sim.tθ
    if name === :qv
        _has_moisture(sim) ||
            throw(ArgumentError("$who: :qv requested but the simulation carries no moisture"))
        return sim.qv, sim.tqv
    end
    throw(ArgumentError("$who: unknown field :$name"))
end

"""
    tendency_integrals(F) -> NamedTuple

The volume integrals ``\\int (\\partial φ/\\partial t)\\,dV`` (units of φ
times m³ s⁻¹) that the last [`apply_forcing!`](@ref) of the
state-dependent forcing `F` ([`ProfileNudging`](@ref),
[`Subsidence`](@ref)) added, keyed by field; an empty `NamedTuple` for
forcings that keep no record and before the first call.
`tendency_integrals(F::Tuple, name)` sums the entry `name` over a tuple
of forcings (0 when none has it). Multiplied by the stage's weight times
Δt they are what a ledger books.
"""
tendency_integrals(F::Union{ProfileNudging,Subsidence}) = F.last[]
tendency_integrals(::AbstractLargeScaleForcing) = NamedTuple()
tendency_integrals(F::Tuple, name::Symbol) =
    sum((Float64(get(tendency_integrals(f), name, 0.0)) for f in F); init = 0.0)
tendency_integrals(F::AbstractLargeScaleForcing, name::Symbol) =
    Float64(get(tendency_integrals(F), name, 0.0))

function apply_forcing!(sim, F::ProfileNudging, t, Δt; kw...)
    m = sim.mesh
    invτ = 1 / F.τ
    wfn = _weight_fn(F.weight, sim)
    t1 = t + Δt
    names = keys(F.targets)
    vals = ntuple(length(names)) do n
        φ, tφ = _state_tendency(sim, names[n], "ProfileNudging")
        _nudge_levels!(φ.data, tφ.data, m, F.targets[n], invτ, wfn, t, t1)
    end
    res = NamedTuple{names}(vals)
    F.last[] = res
    return res
end

# −w(z)(⟨φ⟩ − φ_ref)/τ on every point of each level; returns ∫ tendency dV.
# The order of operations is that of the #171 GeostrophicNudging (and of
# the Berlin case's add_geostrophic_wind! before it), so unweighted u, v
# nudging is bit for bit the former code.
function _nudge_levels!(d, td, m, ref, invτ, wfn, t, t1)
    Nx, Ny, Nz = m.Nx, m.Ny, m.Nz
    Hx, Hy, Hz = m.Hx, m.Hy, m.Hz
    dA = m.dx * m.dy * (Nx * Ny)
    I = 0.0
    @inbounds for k in 1:Nz
        z = _zc(m, k)
        φref = forcing_mean(ref, z, t, t1)
        φ̄ = 0.0
        for j in 1:Ny, i in 1:Nx
            φ̄ += d[Hx+i, Hy+j, Hz+k]
        end
        φ̄ /= (Nx * Ny)
        c = invτ * (φ̄ - φref)
        wfn === nothing || (c = wfn(z) * c)
        for j in 1:Ny, i in 1:Nx
            td[Hx+i, Hy+j, Hz+k] -= c
        end
        I -= c * dA * _dzc(m, k)
    end
    return I
end

function apply_forcing!(sim, F::Subsidence, t, Δt; kw...)
    m = sim.mesh
    t1 = t + Δt
    names = F.explicit ? F.fields :
            Tuple(f for f in F.fields if f !== :qv || _has_moisture(sim))
    vals = map(names) do name
        φ, tφ = _state_tendency(sim, name, "Subsidence")
        F.form === :mean_gradient ?
            _subside_mean!(φ.data, tφ.data, m, F.w_s, t, t1, F.buf) :
            _subside_local!(φ.data, tφ.data, m, F.w_s, t, t1)
    end
    res = NamedTuple{names}(vals)
    F.last[] = res
    return res
end

# indices of the two levels of the gradient at level k: centred inside,
# one-sided toward the interior on the bottom and top levels
@inline _grad_levels(k, Nz) = (k == 1 ? 1 : k - 1, k == Nz ? Nz : k + 1)

function _subside_mean!(d, td, m, w_s, t, t1, buf)
    Nx, Ny, Nz = m.Nx, m.Ny, m.Nz
    Hx, Hy, Hz = m.Hx, m.Hy, m.Hz
    Nz >= 2 || return 0.0
    resize!(buf, Nz)
    @inbounds for k in 1:Nz
        s = 0.0
        for j in 1:Ny, i in 1:Nx
            s += d[Hx+i, Hy+j, Hz+k]
        end
        buf[k] = s / (Nx * Ny)
    end
    dA = m.dx * m.dy * (Nx * Ny)
    I = 0.0
    @inbounds for k in 1:Nz
        kl, ku = _grad_levels(k, Nz)
        G = (buf[ku] - buf[kl]) / (_zc(m, ku) - _zc(m, kl))
        z = _zc(m, k)
        s = -forcing_mean(w_s, z, t, t1) * G
        for j in 1:Ny, i in 1:Nx
            td[Hx+i, Hy+j, Hz+k] += s
        end
        I += s * dA * _dzc(m, k)
    end
    return I
end

function _subside_local!(d, td, m, w_s, t, t1)
    Nx, Ny, Nz = m.Nx, m.Ny, m.Nz
    Hx, Hy, Hz = m.Hx, m.Hy, m.Hz
    Nz >= 2 || return 0.0
    dAc = m.dx * m.dy
    I = 0.0
    @inbounds for k in 1:Nz
        z = _zc(m, k)
        w = forcing_mean(w_s, z, t, t1)
        # upwind: from above for descent (w < 0), from below for ascent;
        # one-sided toward the interior where the upwind level is missing
        kl, ku = w < 0 ? (k == Nz ? (Nz - 1, Nz) : (k, k + 1)) :
                         (k == 1 ? (1, 2) : (k - 1, k))
        idz = 1 / (_zc(m, ku) - _zc(m, kl))
        Ik = 0.0
        for j in 1:Ny, i in 1:Nx
            s = -w * (d[Hx+i, Hy+j, Hz+ku] - d[Hx+i, Hy+j, Hz+kl]) * idz
            td[Hx+i, Hy+j, Hz+k] += s
            Ik += s
        end
        I += Ik * dAc * _dzc(m, k)
    end
    return I
end

_default_tq(sim) = (hasproperty(sim, :tqv) && hasproperty(sim, :has_moisture) && sim.has_moisture) ?
                   sim.tqv : nothing

function apply_forcing!(sim, F::AdvectiveTendencies, t, Δt; tq = _default_tq(sim))
    m = sim.mesh
    F.θ === nothing || _add_level_tendency!(sim.tθ.data, m, F.θ, t, Δt)
    if F.q !== nothing
        tq === nothing && throw(ArgumentError(
            "AdvectiveTendencies: a humidity tendency needs a field to go to (sim has no moisture; pass `tq`)"))
        _add_level_tendency!(tq.data, m, F.q, t, Δt)
    end
    F.u === nothing || _add_level_tendency!(sim.tu.data, m, F.u, t, Δt)
    F.v === nothing || _add_level_tendency!(sim.tv.data, m, F.v, t, Δt)
    return sim
end

function _add_level_tendency!(d, m, x, t, Δt)
    T = eltype(d)
    t1 = t + Δt
    @inbounds for k in 1:m.Nz
        s = T(forcing_mean(x, _zc(m, k), t, t1))
        for j in 1:m.Ny, i in 1:m.Nx
            d[m.Hx+i, m.Hy+j, m.Hz+k] += s
        end
    end
    return d
end

# =========================================================================
# Radiosonde soundings (N-parent-a, #247)
# =========================================================================

include("large_scale_forcing/sounding.jl")

end # module
