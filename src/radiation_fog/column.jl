# Phase 44 (#9) — 1-D radiation-fog column.
#
# Included into `RadiationFog`. A single-column model of the nocturnal
# boundary layer just detailed enough for radiation fog to form, thicken
# by its own fog-top cooling, and burn off after sunrise:
#
#   * radiation      — `fog_radiative_transfer!` (Era III stack) with the
#                      droplet field in the optics (coupling.jl)
#   * phase change   — `fog_condensation` (saturation adjustment at exact
#                      moist enthalpy, this module)
#   * dew / surface  — `dew_deposition` (this module) and a force-restore
#                      surface energy balance
#   * turbulence     — first-order K closure whose stability is measured
#                      by displacing parcels with the same saturation
#                      adjustment, so it sees the moist (fog) stability
#   * settling       — one bulk fall speed, `Superdroplet.terminal_velocity`
#                      at the effective radius (the fall-speed spectrum is
#                      Phase 44 #8's job, not this file's)
#
# Energy variable. The column carries T, q_v, q_l; turbulence mixes the
# static energy s = h + g z, h the contract's `moist_enthalpy`, and the
# total water. Because h is exactly linear in T at fixed partition, T is
# recovered from s without iteration, and the column integral of ρ Δz s
# changes only through boundary fluxes: radiative net flux at the model
# top and surface, the surface sensible heat flux, the enthalpy of the
# vapour exchanged with the surface and that of the droplets that settle
# out. `FogColumn` accumulates each of those, and the tests close the
# budget to round-off.

# =========================================================================
# Settings and state
# =========================================================================

"""
    FogColumnSettings(; kwargs...)

Parameters of a [`FogColumn`](@ref). Defaults describe the ParisFog case
of 14–15 Nov 2011 at SIRTA (see [`parisfog_2011_column`](@ref)); every
field can be overridden by keyword.

Switches (what the feedback tests toggle):

  * `droplet_feedback = true` — fog droplets are seen by the radiation.
    `false` leaves the fog in the thermodynamics but transparent to
    radiation.
  * `shortwave = true` — the sun rises. `false` keeps the night going.
  * `settling = true` — droplets sediment at `terminal_velocity(r_eff)`.

Site and time: `latitude`, `longitude` (degrees, east positive), `start`
= `(year, month, day, hour, minute)` in UTC.

Surface: `z0m`, `z0h` roughness lengths (m); `albedo`, `emissivity`;
`admittance` (thermal admittance of the soil/vegetation top,
J m⁻² K⁻¹ s⁻½) and `T_deep` (K) of the force-restore equation; `β`
the soil moisture availability for bare evaporation.

Turbulence: `λ` asymptotic mixing length (m), `K_min` background
diffusivity (m² s⁻¹), `ug`, `vg` geostrophic wind (m s⁻¹).

Microphysics: `N_d` droplet number (m⁻³) and `k_shape` (see
[`fog_effective_radius`](@ref)).

Numerics: `dt` (s), radiation every `dt_rad` s with gas optics every
`dt_gas` s, diagnostics every `dt_out` s. The eddy diffusivity is
evaluated from the start-of-step state (explicit coefficients in an
implicit solve), which with the 1 m lowest layer of
[`parisfog_2011_column`](@ref) develops the 2Δt "fibrillation" of
stability-dependent K closures (Kalnay & Kanamitsu 1988, MWR 116, 1945)
for `dt ≳ 10` s: at `dt = 10` s the surface is 0.32 K colder after 3 h
than at `dt = 2.5` s and ground fog forms 45 min earlier. `dt = 5` s is
converged to 0.006 K over the same 3 h (tested), and the 14 h life cycle
agrees with `dt = 1` s to 10 min in onset and 5 min in clearance. Fog is `LWC ≥ lwc_fog`
(kg m⁻³); "fog at the ground" is fog in the layer containing `z_obs` m.
"""
Base.@kwdef struct FogColumnSettings
    latitude::Float64 = 48.713
    longitude::Float64 = 2.208
    start::NTuple{5, Int} = (2011, 11, 14, 23, 20)
    z0m::Float64 = 0.05
    z0h::Float64 = 0.005
    albedo::Float64 = 0.22
    emissivity::Float64 = 0.97
    admittance::Float64 = 800.0
    T_deep::Float64 = 280.0
    β::Float64 = 0.3
    λ::Float64 = 30.0
    K_min::Float64 = 0.01
    ug::Float64 = 8.0
    vg::Float64 = 0.0
    N_d::Float64 = 53e6
    k_shape::Float64 = 0.8
    droplet_feedback::Bool = true
    shortwave::Bool = true
    settling::Bool = true
    dt::Float64 = 5.0
    dt_rad::Float64 = 60.0
    dt_gas::Float64 = 900.0
    dt_out::Float64 = 300.0
    lwc_fog::Float64 = 1e-5
    z_obs::Float64 = 3.0
end

"""
    FogBudget

Running boundary-flux integrals of a [`FogColumn`](@ref), J m⁻² for
energy and kg m⁻² for water, all counted positive **into** the column:

  * `radiation` — `∫ (F_net(top) − F_net(sfc)) dt`, LW + SW
  * `sensible`  — `∫ H dt`, surface sensible heat flux
  * `vapour_enthalpy` — enthalpy the surface vapour exchange (dew,
    evaporation) added to the lowest layer
  * `fallout_enthalpy` — enthalpy of the droplets that settled out
    (negative)
  * `vapour_mass`, `fallout_mass` — the matching water fluxes
"""
Base.@kwdef mutable struct FogBudget
    radiation::Float64 = 0.0
    sensible::Float64 = 0.0
    vapour_enthalpy::Float64 = 0.0
    fallout_enthalpy::Float64 = 0.0
    vapour_mass::Float64 = 0.0
    fallout_mass::Float64 = 0.0
end

"""
    FogColumn

State of the 1-D radiation-fog model. Layer arrays are bottom-up:
heights `z` (centres) and `zf` (interfaces), thickness `dz`, fixed
hydrostatic pressure `p`, `pf` and density `ρ`; prognostic temperature
`T` (K), vapour `qv` and fog water `ql` (kg/kg), wind `u`, `v` (m s⁻¹);
surface temperature `Ts`, surface water `w_sfc` (dew + settled droplets,
kg m⁻²), elapsed time `t` (s). `Q` holds the latest radiative energy
input per layer (W m⁻³), `rad` the [`FogRadiation`](@ref) and `budget`
the boundary integrals. Diagnostics are appended to `history` every
`dt_out` (see [`fog_diagnostics`](@ref)).
"""
mutable struct FogColumn{R<:FogRadiation}
    set::FogColumnSettings
    z::Vector{Float64}
    zf::Vector{Float64}
    dz::Vector{Float64}
    p::Vector{Float64}
    pf::Vector{Float64}
    ρ::Vector{Float64}
    T::Vector{Float64}
    qv::Vector{Float64}
    ql::Vector{Float64}
    u::Vector{Float64}
    v::Vector{Float64}
    Ts::Float64
    w_sfc::Float64
    t::Float64
    Q::Vector{Float64}
    Q_lw::Vector{Float64}
    lw_net_sfc::Float64
    sw_abs_sfc::Float64
    μ₀::Float64
    rad::R
    budget::FogBudget
    history::Vector{NamedTuple}
    last_rad::Float64
    last_gas::Float64
    last_out::Float64
end

# von Kármán constant: the Era V SurfaceExchange value, not a local copy.
const _KARMAN = DEFAULT_SURFACE_CONSTANTS.κ

# =========================================================================
# Construction
# =========================================================================

"""
    stretched_levels(H; dz0 = 1.0, r = 1.08, dzmax = 25.0) -> zf

Interface heights from 0 to `H`: the first layer `dz0` thick, each next
one `r` times thicker up to `dzmax`; the last layer absorbs the remainder
to land on `H`, so it can be up to 1.5 times the current spacing thick.
Fog forms in the lowest metres and its top is where the cooling
concentrates, so both need metre-scale layers.
"""
function stretched_levels(H::Real; dz0::Real = 1.0, r::Real = 1.08,
                          dzmax::Real = 25.0)
    zf = [0.0]; d = float(dz0)
    while zf[end] + d < H - 0.5d
        push!(zf, zf[end] + d); d = min(d * r, float(dzmax))
    end
    push!(zf, float(H))
    return zf
end

"""
    FogColumn(zf, T, qv; u = ug, v = vg, Ts = T[1], p_sfc = 1.0e5,
              settings = FogColumnSettings()) -> FogColumn

Column on interfaces `zf` (from 0, bottom-up) with initial layer
temperature `T`, vapour `qv` (and optionally wind). Pressure and density
are integrated hydrostatically from `p_sfc` with the initial virtual
temperature and then held fixed (the column is shallow and
quasi-Boussinesq). Any initial supersaturation is removed with
[`fog_condensation`](@ref).
"""
function FogColumn(zf::AbstractVector, T0::AbstractVector, qv0::AbstractVector;
                   u = nothing, v = nothing, Ts::Union{Nothing,Real} = nothing,
                   p_sfc::Real = 1.0e5,
                   settings::FogColumnSettings = FogColumnSettings(),
                   c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    s = settings
    nz = length(zf) - 1
    # N_d → 0 would silently fall back to the 1 μm r_min guard of
    # fog_effective_radius (optically huge, non-settling droplets).
    s.N_d > 0 || throw(ArgumentError("FogColumn: N_d must be > 0, got $(s.N_d)"))
    s.dt > 0 && s.dt_rad >= s.dt ||
        throw(ArgumentError("FogColumn: need 0 < dt <= dt_rad, got dt = $(s.dt), dt_rad = $(s.dt_rad)"))
    length(T0) == nz == length(qv0) ||
        throw(DimensionMismatch("FogColumn: T and qv need $(nz) layers"))
    zf = collect(float.(zf)); dz = diff(zf); z = (zf[1:end-1] .+ zf[2:end]) ./ 2
    T = collect(float.(T0)); qv = collect(float.(qv0)); ql = zeros(nz)
    Rd = c.gas_constant / c.molmass_dry
    pf = zeros(nz + 1); pf[1] = p_sfc
    for k in 1:nz
        Tv = virtual_temperature(T[k], PhasePartition(qv[k]), c)
        pf[k + 1] = pf[k] * exp(-c.grav * dz[k] / (Rd * Tv))
    end
    p = [sqrt(pf[k] * pf[k + 1]) for k in 1:nz]
    ρ = [air_density(T[k], p[k], PhasePartition(qv[k]), c) for k in 1:nz]
    for k in 1:nz
        r = fog_condensation(T[k], p[k], qv[k], 0.0, s.dt; c = c)
        T[k] = r.T; qv[k] = r.qv; ql[k] = r.ql
    end
    uu = u === nothing ? fill(s.ug, nz) : collect(float.(u))
    vv = v === nothing ? fill(s.vg, nz) : collect(float.(v))
    rad = fog_radiation(nz, zf[end], pf[end], T[end], qv[end])
    col = FogColumn(s, z, zf, dz, p, pf, ρ, T, qv, ql, uu, vv,
                    Ts === nothing ? T[1] : float(Ts), 0.0, 0.0,
                    zeros(nz), zeros(nz), 0.0, 0.0, 0.0, rad, FogBudget(),
                    NamedTuple[], -Inf, -Inf, -Inf)
    return col
end

"""
    parisfog_2011_column(; H = 1000.0, settings = FogColumnSettings(), kwargs...)
        -> FogColumn

Semi-idealised initial state for the ParisFog radiation fog of
14–15 November 2011 at SIRTA (48.71 °N, 2.21 °E), documented by
Mazoyer et al. (2017): fog appeared aloft at 02:00 UTC and at the ground
at 02:30 UTC, became optically thick at ≈ 03:00 UTC when ≈ 80 m deep,
peaked at LWP ≈ 70 g m⁻² around 07:30 UTC with a top of up to 300 m
(sodar), and dissipated at the ground around 10:00 UTC, ≈ 3 h after the
06:59 UTC sunrise. Their LES starts at 23:20 UTC from the Trappes
sounding adjusted to the SIRTA mast, with a surface temperature of 276 K
and an 8 m s⁻¹ geostrophic wind; the column starts at the same time
with the same surface temperature and geostrophic wind.

**What is not reproduced.** The sounding itself is shown only as a
figure (their Fig. A.1), so the profile here is constructed, not
digitised: a 60 m surface inversion of 2 K topped by a slightly stable
layer to 400 m (the depth to which they adjusted the sounding), relative
humidity 97 % at the ground (their observed value from 00:00 to
01:30 UTC) falling to 88 % at 400 m and 50 % above, and a
4 K km⁻¹ stable free troposphere. The comparison is therefore of fog
*life-cycle timing and structure*, not of profiles.
"""
function parisfog_2011_column(; H::Real = 1000.0,
                              settings::FogColumnSettings = FogColumnSettings(),
                              T_sfc::Real = 276.0, p_sfc::Real = 1.0200e5,
                              c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    zf = stretched_levels(H)
    z = (zf[1:end-1] .+ zf[2:end]) ./ 2
    nz = length(z)
    # potential-temperature profile (K), referenced to the surface
    Π(p) = exner_function(p, PhasePartition(0.0), c)     # dry Exner, contract
    θs = T_sfc / Π(p_sfc)
    θ(z) = z < 60  ? θs + 2.0 * z / 60 :
           z < 400 ? θs + 2.0 + 1.5e-3 * (z - 60) :
                     θs + 2.0 + 1.5e-3 * 340 + 4.0e-3 * (z - 400)
    RH(z) = z < 400 ? 0.97 - 0.09 * z / 400 :
            z < 500 ? 0.88 - 0.38 * (z - 400) / 100 : 0.50
    T = zeros(nz); qv = zeros(nz)
    Rd = c.gas_constant / c.molmass_dry
    p = p_sfc
    for k in 1:nz
        dzk = zf[k + 1] - zf[k]
        # pressure at the layer centre, isothermal step with the θ estimate
        pk = p * exp(-c.grav * (z[k] - zf[k]) / (Rd * θ(z[k]) * Π(p)))
        T[k] = θ(z[k]) * Π(pk)
        qv[k] = RH(z[k]) * saturation_specific_humidity(T[k], pk, c)
        p = p * exp(-c.grav * dzk / (Rd * T[k]))
    end
    # Wind: a 1-D Ekman-like start, weak near the ground (the observed
    # 10 m wind was ≈ 1.8 m s⁻¹) turning into the geostrophic value.
    u = [settings.ug * min(1.0, log1p(z[k] / 0.05) / log1p(300 / 0.05)) for k in 1:nz]
    v = zeros(nz)
    return FogColumn(zf, T, qv; u = u, v = v, Ts = T_sfc, p_sfc = p_sfc,
                     settings = settings, c = c)
end

# =========================================================================
# Energy bookkeeping helpers
# =========================================================================

@inline _h(T, qv, ql, c) = moist_enthalpy(T, PhasePartition(qv + ql, ql, 0.0), c)

"""
    column_energy(col; c = DEFAULT_CONSTANTS) -> J m⁻²

`Σ ρ_k Δz_k (h_k + g z_k)` with `h` the contract's `moist_enthalpy`: the
quantity the column conserves up to its boundary fluxes.
"""
column_energy(col::FogColumn; c::ThermodynamicConstants = DEFAULT_CONSTANTS) =
    sum(col.ρ[k] * col.dz[k] * (_h(col.T[k], col.qv[k], col.ql[k], c) +
        c.grav * col.z[k]) for k in eachindex(col.T))

"Column water `Σ ρ Δz (q_v + q_l)`, kg m⁻²."
column_water(col::FogColumn) =
    sum(col.ρ[k] * col.dz[k] * (col.qv[k] + col.ql[k]) for k in eachindex(col.T))

"Liquid water path `Σ ρ Δz q_l`, kg m⁻²."
fog_liquid_water_path(col::FogColumn) =
    sum(col.ρ[k] * col.dz[k] * col.ql[k] for k in eachindex(col.T))

# Set layer k's (qv, ql) and change its enthalpy by Δh (J/kg), solving
# for T exactly (h is linear in T at fixed partition).
@inline function _set_layer!(col::FogColumn, k, qv, ql, h_new, c)
    T0 = col.T[k]
    q1 = PhasePartition(qv + ql, ql, 0.0)
    col.T[k] = temperature_from_moist_enthalpy(h_new, q1, c; T_ref = T0)
    col.qv[k] = qv; col.ql[k] = ql
    return col
end

# =========================================================================
# Radiation step
# =========================================================================

function _utc(col::FogColumn)
    y, mo, d, h, mi = col.set.start
    return solar_time(y, mo, d, h, mi, col.t; latitude = col.set.latitude,
                      longitude = col.set.longitude)
end

"""
    update_radiation!(col; force_gas = false) -> col

Recompute the radiative energy input `col.Q` (W m⁻³, LW + SW), its LW
part `col.Q_lw`, and the surface net LW and absorbed SW from the current
state. Solar geometry comes from `SolarGeometry.solar_position` at the
column's UTC time; the SW is scaled by the Earth–Sun distance factor.
"""
function update_radiation!(col::FogColumn; force_gas::Bool = false,
                           c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    s = col.set
    μ₀ = 0.0; scale = 1.0
    if s.shortwave
        pos = solar_position(_utc(col))
        μ₀ = pos.μ₀; scale = earth_sun_distance_factor(pos)
    end
    col.μ₀ = μ₀
    acol = fog_atmospheric_column(col.rad, col.T, col.p, col.pf, col.dz,
                                  col.ρ, col.qv, col.ql, col.Ts;
                                  N_d = s.N_d, k = s.k_shape,
                                  droplet_feedback = s.droplet_feedback)
    upd = force_gas || (col.t - col.last_gas >= s.dt_gas - 1e-9)
    fog_radiative_transfer!(col.rad, acol; μ₀ = μ₀, solar_scale = scale,
                            albedo = s.albedo, emissivity = s.emissivity,
                            update_gas = upd)
    upd && (col.last_gas = col.t)
    fog_radiative_heating!(col.Q, col.rad, acol)
    ntot = length(acol.dz)
    div = zeros(ntot)
    flux_divergence!(div, col.rad.flux_lw, acol.dz)
    col.Q_lw .= @view div[1:length(col.T)]
    fl = col.rad.flux_lw; fs = col.rad.flux_sw
    col.lw_net_sfc = fl.flux_net[1]
    col.sw_abs_sfc = fs.flux_net[1]
    col.last_rad = col.t
    return col
end

# =========================================================================
# Turbulence
# =========================================================================

# Parcel from layer `a` displaced to layer `b`, conserving s = h + g z and
# total water, then saturation-adjusted at p_b (condensing or evaporating
# with the contract's isobaric adjustment via `fog_equilibrium`). Returns θ_ρ of the parcel.
function _displaced_θρ(col::FogColumn, a::Int, b::Int, c)
    qt = col.qv[a] + col.ql[a]; ql = col.ql[a]
    h_b = _h(col.T[a], col.qv[a], ql, c) + c.grav * (col.z[a] - col.z[b])
    q1 = PhasePartition(qt, ql, 0.0)
    T = temperature_from_moist_enthalpy(h_b, q1, c; T_ref = col.T[a])
    Δ, Teq = fog_equilibrium(T, col.p[b], qt - ql, ql, c)
    return density_potential_temperature(Teq, col.p[b],
                                         PhasePartition(qt, ql + Δ, 0.0), c)
end

"""
    moist_stability(col, k) -> N² (s⁻²)

Squared buoyancy frequency across interface `k+½`, measured by
displacing a parcel from each side to the other with conserved static
energy and total water and the fog saturation adjustment, and comparing
density potential temperatures (`MoistThermodynamics.
density_potential_temperature`). In unsaturated air this is the usual
dry `N²` of `θ_ρ`; inside fog it is the saturated stability, which is
what lets fog-top cooling overturn the fog layer.
"""
function moist_stability(col::FogColumn, k::Int;
                         c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    θρ_env_up = density_potential_temperature(col.T[k + 1], col.p[k + 1],
                    PhasePartition(col.qv[k + 1] + col.ql[k + 1], col.ql[k + 1], 0.0), c)
    θρ_env_dn = density_potential_temperature(col.T[k], col.p[k],
                    PhasePartition(col.qv[k] + col.ql[k], col.ql[k], 0.0), c)
    θup = _displaced_θρ(col, k, k + 1, c)     # lower parcel lifted
    θdn = _displaced_θρ(col, k + 1, k, c)     # upper parcel lowered
    Δz = col.z[k + 1] - col.z[k]
    N2_up = -c.grav * (θup - θρ_env_up) / θρ_env_up / Δz
    N2_dn =  c.grav * (θdn - θρ_env_dn) / θρ_env_dn / Δz
    return (N2_up + N2_dn) / 2
end

# Stability function: `TurbulenceStability.ShortTail` — `(1 + 5 Ri)⁻²` in
# stable air, a short-tailed form that lets turbulence collapse in a strong
# inversion, as radiation fog needs, and `sqrt(1 − 16 Ri)` in unstable air
# (Businger-Dyer). The column has one K for heat, water and momentum, so
# it uses the momentum factor throughout (ShortTail's f_h is the same).
const _FOG_STABILITY = ShortTail()

"""
    eddy_diffusivity(col) -> K (m² s⁻¹) at interfaces 2..nz

First-order local closure `K = l² |∂U/∂z| f(Ri) + K_min`, Blackadar
mixing length `l = κ z / (1 + κ z / λ)`, gradient Richardson number from
[`moist_stability`](@ref), and `f` the short-tailed stability function
`TurbulenceStability.ShortTail` (through `stability_factor_m`). The same
`K` mixes heat, water and momentum.
"""
function eddy_diffusivity(col::FogColumn;
                          c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    s = col.set; nz = length(col.T)
    K = zeros(nz + 1)
    for k in 1:nz - 1
        zi = col.zf[k + 1]
        Δz = col.z[k + 1] - col.z[k]
        l  = _KARMAN * zi / (1 + _KARMAN * zi / s.λ)
        S2 = ((col.u[k + 1] - col.u[k])^2 + (col.v[k + 1] - col.v[k])^2) / Δz^2
        S2 = max(S2, 1e-8)
        N2 = moist_stability(col, k; c = c)
        Ri = gradient_richardson(N2, S2)
        K[k + 1] = l^2 * sqrt(S2) * stability_factor_m(_FOG_STABILITY, Ri) + s.K_min
    end
    return K
end

# Implicit (backward-Euler) flux-form diffusion of φ with interface
# conductances D (kg m⁻² s⁻¹; D[1] = D[end] = 0), an optional implicit
# sink `a1` (kg m⁻² s⁻¹) on layer 1, weights m_k = ρ_k Δz_k. Conserves
# Σ m φ exactly when a1 = 0. Thomas algorithm.
function _implicit_diffuse!(φ::AbstractVector, D::AbstractVector,
                            m::AbstractVector, dt::Real; a1::Real = 0.0)
    n = length(φ)
    a = zeros(n); b = zeros(n); cc = zeros(n); r = zeros(n)
    for k in 1:n
        lo = D[k] * dt; hi = D[k + 1] * dt
        a[k] = -lo; cc[k] = -hi
        b[k] = m[k] + lo + hi + (k == 1 ? a1 * dt : 0.0)
        r[k] = m[k] * φ[k]
    end
    for k in 2:n
        w = a[k] / b[k - 1]
        b[k] -= w * cc[k - 1]; r[k] -= w * r[k - 1]
    end
    φ[n] = r[n] / b[n]
    for k in n - 1:-1:1
        φ[k] = (r[k] - cc[k] * φ[k + 1]) / b[k]
    end
    return φ
end

# =========================================================================
# Surface layer
# =========================================================================

"""
    surface_exchange(col) -> (; C_D, C_H, U, Ri_b)

Bulk transfer coefficients between the surface and the lowest layer:
neutral values `κ²/ln(z/z0m)²` and `κ²/(ln(z/z0m) ln(z/z0h))`, times
the same stability function as the interior (`ShortTail`) with the bulk
Richardson number of the surface layer.
"""
function surface_exchange(col::FogColumn;
                          c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    s = col.set
    z1 = col.z[1]
    U = max(hypot(col.u[1], col.v[1]), 0.1)
    q1 = PhasePartition(col.qv[1] + col.ql[1], col.ql[1], 0.0)
    θv1 = density_potential_temperature(col.T[1], col.p[1], q1, c)
    θvs = virtual_potential_temperature(col.Ts, col.pf[1],
                                        PhasePartition(col.qv[1]), c)
    Ri_b = c.grav * z1 * (θv1 - θvs) / (θv1 * U^2)
    lm = log(z1 / s.z0m); lh = log(z1 / s.z0h)
    f = stability_factor_m(_FOG_STABILITY, Ri_b)
    return (C_D = _KARMAN^2 / lm^2 * f, C_H = _KARMAN^2 / (lm * lh) * f,
            U = U, Ri_b = Ri_b)
end

# Surface fluxes and the force-restore update of T_s. Changes layer 1
# (heat, vapour) and records the boundary integrals.
function _surface_step!(col::FogColumn, ex, dt, c)
    s = col.set; b = col.budget
    ρ1 = col.ρ[1]; dz1 = col.dz[1]; m1 = ρ1 * dz1
    # --- sensible heat: dry static energy difference surface→layer 1
    H = ρ1 * c.cp_dry * ex.C_H * ex.U *
        (col.Ts - (col.T[1] + c.grav * col.z[1] / c.cp_dry))
    # --- vapour exchange: dew / surface-water evaporation by the module's
    # own implicit bulk law, bare-soil evaporation (β) when dry
    qs_s = saturation_specific_humidity(col.Ts, col.pf[1], c)
    if col.qv[1] > qs_s || col.w_sfc > 0
        r = dew_deposition(col.qv[1], col.Ts, col.pf[1], dt; dz = dz1,
                           ρ = ρ1, C_e = ex.C_H, U = ex.U,
                           dew = col.w_sfc, c = c)
        Δqv = r.Δqv; L_sfc = r.surface_heat / dt       # W m⁻², + when dew forms
        col.w_sfc = max(col.w_sfc + r.mass, 0.0)
    else
        f = relaxation_fraction(dt, dz1 / (s.β * ex.C_H * ex.U))
        Δqv = s.β > 0 ? (qs_s - col.qv[1]) * f : 0.0
        L_sfc = -latent_heat_vapor(col.Ts, c) * m1 * Δqv / dt
    end
    # apply to layer 1: heat at fixed partition, then vapour at fixed T
    h0 = _h(col.T[1], col.qv[1], col.ql[1], c)
    _set_layer!(col, 1, col.qv[1], col.ql[1], h0 + H * dt / m1, c)
    hA = _h(col.T[1], col.qv[1], col.ql[1], c)
    hB = _h(col.T[1], col.qv[1] + Δqv, col.ql[1], c)
    col.qv[1] += Δqv
    b.sensible += H * dt
    b.vapour_enthalpy += m1 * (hB - hA)
    b.vapour_mass += m1 * Δqv
    # --- force-restore surface temperature (Bhumralkar 1975; Deardorff 1978)
    τd = 86400.0
    # Net LW at the *current* T_s: the downwelling of the last radiation
    # call, the upwelling re-evaluated every step (it is the stiff term).
    LWn = s.emissivity * (col.rad.flux_lw.flux_dn[1] - STEFAN_BOLTZMANN * col.Ts^4)
    G = col.sw_abs_sfc + LWn - H + L_sfc
    CT = 2 * sqrt(π / τd) / s.admittance
    col.Ts += dt * (CT * G - 2π / τd * (col.Ts - s.T_deep))
    return (H = H, L = L_sfc, G = G, LW = LWn)
end

# =========================================================================
# Droplet settling
# =========================================================================

"""
    settling_velocity(ql, ρ, N_d; k = 0.8) -> m s⁻¹

Bulk sedimentation velocity of the fog water: `Superdroplet.
terminal_velocity` (Stokes with slip correction at fog sizes) at the
effective radius of [`fog_effective_radius`](@ref). One velocity for the
whole spectrum — the per-moment fall-speed spectrum is Phase 44 #8.
For `N_d` = 50 cm⁻³ and 0.2 g m⁻³ this gives ≈ 1 cm s⁻¹.
"""
settling_velocity(ql::Real, ρ::Real, N_d::Real; k::Real = 0.8) =
    ql > 0 ? terminal_velocity(1e-6 * fog_effective_radius(ql, ρ, N_d; k = k)) : 0.0

# Explicit upwind settling; each layer passes F_k dt (kg m⁻²) of liquid,
# with its enthalpy at the layer's own temperature, to the layer below.
function _settle!(col::FogColumn, dt, c)
    s = col.set; nz = length(col.T); b = col.budget
    F = zeros(nz)
    for k in 1:nz
        col.ql[k] > 0 || continue
        m = col.ρ[k] * col.dz[k]
        F[k] = min(col.ρ[k] * col.ql[k] *
                   settling_velocity(col.ql[k], col.ρ[k], s.N_d; k = s.k_shape) * dt,
                   m * col.ql[k])
    end
    for k in 1:nz
        F[k] > 0 || continue
        m = col.ρ[k] * col.dz[k]
        # layer k loses the liquid and the enthalpy it carries at T_k
        h0 = _h(col.T[k], col.qv[k], col.ql[k], c)
        qln = col.ql[k] - F[k] / m
        h1 = _h(col.T[k], col.qv[k], qln, c)
        E = m * (h0 - h1)                       # J m⁻² leaving layer k
        col.ql[k] = qln                         # T_k unchanged
        if k == 1
            b.fallout_enthalpy -= E; b.fallout_mass -= F[k]
            col.w_sfc += F[k]
        else
            j = k - 1; mj = col.ρ[j] * col.dz[j]
            hj = _h(col.T[j], col.qv[j], col.ql[j], c)
            _set_layer!(col, j, col.qv[j], col.ql[j] + F[k] / mj, hj + E / mj, c)
        end
    end
    return col
end

# =========================================================================
# Time stepping
# =========================================================================

"""
    step!(col) -> col

Advance the column by `col.set.dt`: radiation (every `dt_rad`),
radiative heating, surface fluxes and `T_s`, implicit turbulent mixing
of `s = h + g z`, `q_v`, `q_l` and momentum, Coriolis turning toward the
geostrophic wind, droplet settling, and finally fog saturation
adjustment with [`fog_condensation`](@ref). Diagnostics are recorded
every `dt_out`.
"""
function step!(col::FogColumn; c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    s = col.set; dt = s.dt; nz = length(col.T); b = col.budget
    col.t - col.last_rad >= s.dt_rad - 1e-9 && update_radiation!(col; c = c)
    col.t - col.last_out >= s.dt_out - 1e-9 && _record!(col, c)

    # 1. radiative heating at fixed partition
    for k in 1:nz
        m = col.ρ[k] * col.dz[k]
        h = _h(col.T[k], col.qv[k], col.ql[k], c)
        _set_layer!(col, k, col.qv[k], col.ql[k], h + col.Q[k] * dt / col.ρ[k], c)
        b.radiation += col.Q[k] * col.dz[k] * dt
    end

    # 2. surface
    ex = surface_exchange(col; c = c)
    _surface_step!(col, ex, dt, c)

    # 3. turbulent mixing (implicit, flux form)
    K = eddy_diffusivity(col; c = c)
    D = zeros(nz + 1)
    for k in 1:nz - 1
        ρi = (col.ρ[k] * col.dz[k + 1] + col.ρ[k + 1] * col.dz[k]) / (col.dz[k] + col.dz[k + 1])
        D[k + 1] = ρi * K[k + 1] / (col.z[k + 1] - col.z[k])
    end
    m = col.ρ .* col.dz
    sv = [_h(col.T[k], col.qv[k], col.ql[k], c) + c.grav * col.z[k] for k in 1:nz]
    _implicit_diffuse!(sv, D, m, dt)
    _implicit_diffuse!(col.qv, D, m, dt)
    _implicit_diffuse!(col.ql, D, m, dt)
    for k in 1:nz
        col.ql[k] = max(col.ql[k], 0.0)
        _set_layer!(col, k, col.qv[k], col.ql[k], sv[k] - c.grav * col.z[k], c)
    end
    drag = col.ρ[1] * ex.C_D * ex.U
    _implicit_diffuse!(col.u, D, m, dt; a1 = drag)
    _implicit_diffuse!(col.v, D, m, dt; a1 = drag)

    # 4. Coriolis: exact rotation of the ageostrophic wind
    f = 2 * 7.2921e-5 * sind(s.latitude)
    cs, sn = cos(f * dt), sin(f * dt)
    for k in 1:nz
        a = col.u[k] - s.ug; bb = col.v[k] - s.vg
        col.u[k] = s.ug + a * cs + bb * sn
        col.v[k] = s.vg - a * sn + bb * cs
    end

    # 5. settling, 6. saturation adjustment
    s.settling && _settle!(col, dt, c)
    for k in 1:nz
        r = fog_condensation(col.T[k], col.p[k], col.qv[k], col.ql[k], dt; c = c)
        r.Δq == 0 && continue
        col.T[k] = r.T; col.qv[k] = r.qv; col.ql[k] = r.ql
    end
    col.t += dt
    return col
end

"""
    run!(col, duration) -> col

Step the column for `duration` seconds (rounded to whole steps) and
record a final diagnostic.
"""
function run!(col::FogColumn, duration::Real;
              c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    n = round(Int, duration / col.set.dt)
    for _ in 1:n
        step!(col; c = c)
    end
    update_radiation!(col; c = c)
    _record!(col, c)
    return col
end

# =========================================================================
# Diagnostics
# =========================================================================

"""
    fog_diagnostics(col) -> NamedTuple

Instantaneous fog diagnostics (using the fluxes of the last radiation
call):

  * `t` (s since start), `utc_hour` (hours since 00 UTC of the start day,
    may exceed 24), `μ₀`
  * `fog_top` (m; highest interface of a fog layer, 0 if none),
    `fog_base`, `lwp` (g m⁻²), `fog_obs` (fog at `z_obs`)
  * `T_obs`, `Ts` (K), `lw_net_sfc`, `sw_abs_sfc`, `lw_dn_sfc` (W m⁻²)
  * `lw_net_top` — net LW flux at the fog top (W m⁻², `NaN` without fog)
  * `lw_fog` — longwave net-flux convergence of the fog layer (W m⁻²,
    negative = cooling), `lw_top_cooling` — the strongest LW cooling rate
    in the fog (K day⁻¹, `Q/(ρ c_pd)`)
  * `C_LW`, `E_SW` — the Wærsted et al. (2017) radiation-driven
    condensation and evaporation of the fog layer, g m⁻² h⁻¹
"""
function fog_diagnostics(col::FogColumn;
                         c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    s = col.set; nz = length(col.T)
    fog = [col.ρ[k] * col.ql[k] >= s.lwc_fog for k in 1:nz]
    kt = findlast(fog); kb = findfirst(fog)
    kobs = clamp(searchsortedlast(col.zf, s.z_obs), 1, nz)
    lw_fog = 0.0; cool = 0.0; CLW = 0.0; ESW = 0.0; lw_top = NaN
    if kt !== nothing
        lw_fog, _ = fog_top_flux_jump(col.rad, kb, kt)
        lw_top = col.rad.flux_lw.flux_net[kt + 1]
        for k in kb:kt
            fog[k] || continue
            cool = min(cool, col.Q_lw[k] / (col.ρ[k] * c.cp_dry) * 86400)
            CLW += radiative_condensation_rate(col.Q_lw[k], col.T[k], col.ρ[k]; c = c) * col.dz[k]
            ESW -= radiative_condensation_rate(col.Q[k] - col.Q_lw[k], col.T[k], col.ρ[k]; c = c) * col.dz[k]
        end
    end
    y, mo, d, h, mi = s.start
    return (t = col.t, utc_hour = h + mi / 60 + col.t / 3600, μ₀ = col.μ₀,
            fog_top = kt === nothing ? 0.0 : col.zf[kt + 1],
            fog_base = kb === nothing ? NaN : col.zf[kb],
            lwp = 1000 * fog_liquid_water_path(col), fog_obs = fog[kobs],
            T_obs = col.T[kobs], Ts = col.Ts,
            lw_net_sfc = col.rad.flux_lw.flux_net[1],
            lw_dn_sfc = col.rad.flux_lw.flux_dn[1],
            sw_abs_sfc = col.sw_abs_sfc, lw_fog = lw_fog, lw_net_top = lw_top,
            lw_top_cooling = cool, C_LW = CLW * 1000 * 3600,
            E_SW = ESW * 1000 * 3600, w_sfc = col.w_sfc)
end

function _record!(col::FogColumn, c)
    push!(col.history, fog_diagnostics(col; c = c))
    col.last_out = col.t
    return col
end

"""
    fog_life_cycle(history; min_duration = 1800) -> NamedTuple

Onset and dissipation of fog at the observation height from a
diagnostic `history` (see [`fog_diagnostics`](@ref)), in UTC hours:
`onset` is the start of the first fog episode at `z_obs` lasting at
least `min_duration` s, `dissipation` the end of the last episode
(`NaN` if fog is still there at the end of the record), plus
`max_top` (m), `max_lwp` (g m⁻²) and the time of the latter.
"""
function fog_life_cycle(hist::AbstractVector; min_duration::Real = 1800)
    onset = NaN; dissip = NaN
    n = length(hist)
    i = 1
    while i <= n
        if hist[i].fog_obs
            j = i
            while j < n && hist[j + 1].fog_obs
                j += 1
            end
            if hist[j].t - hist[i].t >= min_duration
                isnan(onset) && (onset = hist[i].utc_hour)
                dissip = j == n ? NaN : hist[j + 1].utc_hour
            end
            i = j + 1
        else
            i += 1
        end
    end
    tops = [h.fog_top for h in hist]; lwps = [h.lwp for h in hist]
    imax = argmax(lwps)
    return (onset = onset, dissipation = dissip, max_top = maximum(tops),
            max_lwp = lwps[imax], t_max_lwp = hist[imax].utc_hour)
end
