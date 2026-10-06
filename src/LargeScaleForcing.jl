module LargeScaleForcing

# Era VI, Phase 58 (#171) — prescribed large-scale (case) forcing.
#
# The forcing a single-column or LES case is driven by when the larger
# scales are not resolved: the geostrophic pressure gradient, nudging of
# the horizontal mean toward a target wind, prescribed large-scale
# advective tendencies and a prescribed surface-temperature series.
# GABLS1-4, CASES-99 and (Phase 67) ERA5-forced cases are built from
# these types; this module only defines them and applies them.
#
# Every forcing quantity is a constant, a height profile or a height-time
# table (`ForcingTable`), linearly interpolated in both and held constant
# beyond the table ends. A forcing is applied over a step `[t, t + Δt]`
# with its exact step mean (`forcing_mean`), so a piecewise-linear
# table is integrated exactly in time by any explicit step.

import ..SurfaceExchange: surface_temperature
using ..MoistThermodynamics: ThermodynamicConstants, DEFAULT_CONSTANTS, PhasePartition,
                             exner_function, virtual_temperature,
                             specific_humidity_from_relative_humidity
using ..Fields: location
using ..Operators: DampingProfile
using ..Operators.FiniteVolume: _add_damping!, damping_coordinates

export ForcingTable, forcing_value, forcing_mean,
       AbstractLargeScaleForcing, GeostrophicPressureGradient, GeostrophicNudging,
       AdvectiveTendencies, SurfaceTemperatureSeries, apply_forcing!,
       Sounding, SoundingLaunch, read_sounding, sounding_profiles, sounding_time,
       ReferenceDamping, damping_tendency     # N-parent-b (#254)

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
    GeostrophicNudging(U_g, V_g, τ)

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
jet, and is the behaviour of the Berlin case before #171 (moved here
verbatim from `examples/urban/couplings/geostrophic.jl`). `U_g`, `V_g`
take the same forms as in [`GeostrophicPressureGradient`](@ref).
"""
struct GeostrophicNudging{U,V,T<:Real} <: AbstractLargeScaleForcing
    U_g::U
    V_g::V
    τ::T
    function GeostrophicNudging(U_g::U, V_g::V, τ::T) where {U,V,T<:Real}
        τ > 0 || throw(ArgumentError("GeostrophicNudging: τ must be positive, got $τ"))
        return new{U,V,T}(U_g, V_g, τ)
    end
end

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
subsidence, which depends on the model state, is not part of it.

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

# Height (m) of the cell centres of level k — where u, v (FaceX/FaceY)
# and the scalars live on the C-grid.
@inline _zc(m, k) = m.z0 + (k - 0.5) * m.dz

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

function apply_forcing!(sim, F::GeostrophicNudging, t, Δt; kw...)
    Nx, Ny = sim.mesh.Nx, sim.mesh.Ny
    Hx, Hy, Hz = sim.mesh.Hx, sim.mesh.Hy, sim.mesh.Hz
    Nz = sim.mesh.Nz
    invτ = 1 / F.τ
    t1 = t + Δt
    @inbounds for k in 1:Nz
        z = _zc(sim.mesh, k)
        U_g = forcing_mean(F.U_g, z, t, t1)
        V_g = forcing_mean(F.V_g, z, t, t1)
        ū = 0.0; v̄ = 0.0
        for j in 1:Ny, i in 1:Nx
            ū += sim.u.data[Hx+i, Hy+j, Hz+k]
            v̄ += sim.v.data[Hx+i, Hy+j, Hz+k]
        end
        ū /= (Nx * Ny); v̄ /= (Nx * Ny)
        for j in 1:Ny, i in 1:Nx
            sim.tu.data[Hx+i, Hy+j, Hz+k] -= invτ * (ū - U_g)
            sim.tv.data[Hx+i, Hy+j, Hz+k] -= invτ * (v̄ - V_g)
        end
    end
    return sim
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
include("large_scale_forcing/reference_damping.jl")   # N-parent-b (#254)

end # module
