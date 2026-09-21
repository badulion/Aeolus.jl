# Air-quality / aerosol add-ons for the Tier 3 urban flagship example.
#
# Batch E — five gaps:
#
#   * A1 Multi-species chemistry — a small hard-coded KPP-style mechanism
#     (NO, NO2, O3, VOC) with photolysis + Ox coupling.
#   * A2 Dry + wet deposition — per-species surface-class-dependent dry
#     deposition velocities plus rain-scavenging.
#   * A3 Biogenic + traffic emissions — hour-of-day, surface-class-keyed
#     emission maps (traffic on streets/walls, biogenic on parks).
#   * A4 Two-moment aerosol — a mass mixing ratio + number concentration,
#     first-order coagulation, mass-conserving sedimentation, deposition.
#
# All routines are surface-class-aware and reuse the enum defined by
# `urban_geometry.jl` (SURFACE_STREET=1, ROOF=2, PARK=3, WATER=4, WALL=5,
# GLASS=6 once `glass_material.jl` is also included).
#
# Nothing here mutates the Aeolus core. The chemistry / aerosol state is
# a NamedTuple of Field{CellCenter} that the caller advects (via
# `advect_centered!` / `advect_weno5!`) and passes back into these
# kernels for the reactive substeps.

include(joinpath(@__DIR__, "urban_geometry.jl"))
using Aeolus
using Aeolus.Fields: interior_view, set!

export ChemistryState, apply_photochem_step!,
       deposition_velocity, apply_dry_deposition!, apply_wet_deposition!,
       emission_rate, add_emissions!,
       AerosolState, apply_aerosol_step!

# ---------------------------------------------------------------------------
# A1 — Multi-species chemistry state
# ---------------------------------------------------------------------------
#
# Concentration units: mass-mixing-ratio-like scalars, non-dimensional
# "background of unity" — we store ppb-scaled values as Float64. The
# rate constants below are chosen so that a single Euler substep of
# dt ≈ 1 s produces a stable, sensible photostationary state on the
# 0..100 ppb range; treat this as a *toy* mechanism whose primary job
# is to exercise transport + surface coupling with more than one
# species, not to be quantitative.

# Background concentrations (in the ppb-scaled internal unit).
const _NO_BG    = 0.5
const _NO2_BG   = 5.0
const _O3_BG    = 40.0
const _VOC_BG   = 10.0

# Rate constants (chosen for numerical realism in the toy unit system).
const _J_MAX = 0.01        # 1/s — NO2 photolysis at solar noon
const _K1    = 1.8e-14 * 1.0   # NO + O3 → NO2 + O2 ;  ρ absorbed = 1
const _K2    = 1.0e-16     # VOC + O3 → 0.5*O3 + products (toy)

"""
    ChemistryState(mesh) -> NamedTuple

Return a NamedTuple `(NO, NO2, O3, VOC)` of `Field{CellCenter}` all
initialized to the background concentrations
`(0.5, 5.0, 40.0, 10.0)` in ppb-scaled internal units.

Caller advects each field with the existing tracer operators and
passes the whole tuple into [`apply_photochem_step!`](@ref) for the
chemistry substep.
"""
function ChemistryState(mesh)
    NO  = Field(CellCenter(), mesh); set!(NO,  _NO_BG)
    NO2 = Field(CellCenter(), mesh); set!(NO2, _NO2_BG)
    O3  = Field(CellCenter(), mesh); set!(O3,  _O3_BG)
    VOC = Field(CellCenter(), mesh); set!(VOC, _VOC_BG)
    return (NO = NO, NO2 = NO2, O3 = O3, VOC = VOC)
end

"Diurnal proxy for the NO2 photolysis rate: `J = J_max · max(0, cos((h-12)π/12))`."
@inline function _J_photolysis(hour::Real)
    return _J_MAX * max(0.0, cos((hour - 12) * pi / 12))
end

"""
    apply_photochem_step!(state, hour_of_day, dt)

Advance the chemistry state one explicit-Euler substep of size `dt`
seconds at local hour `hour_of_day`. Reactions:

  * NO2 + hν → NO + O3        (rate `J`)
  * NO + O3 → NO2 + O2        (rate `k1`)
  * VOC + O3 → 0.5·O3 + prod. (rate `k2`, toy)

Concentrations are clamped ≥ 0 at each cell after the update.
"""
function apply_photochem_step!(state::NamedTuple, hour_of_day::Real, dt::Real)
    J = _J_photolysis(hour_of_day)
    k1 = _K1
    k2 = _K2
    NO_v  = interior_view(state.NO)
    NO2_v = interior_view(state.NO2)
    O3_v  = interior_view(state.O3)
    VOC_v = interior_view(state.VOC)
    Nx, Ny, Nz = size(NO_v)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        no   = NO_v[i, j, k]
        no2  = NO2_v[i, j, k]
        o3   = O3_v[i, j, k]
        voc  = VOC_v[i, j, k]

        r_photo = J  * no2
        r_titr  = k1 * no  * o3
        r_vocox = k2 * voc * o3

        d_no   = ( r_photo - r_titr) * dt
        d_no2  = (-r_photo + r_titr) * dt
        d_o3   = ( r_photo - r_titr - 0.5 * r_vocox) * dt
        d_voc  = (-r_vocox) * dt

        NO_v[i, j, k]  = max(0.0, no  + d_no)
        NO2_v[i, j, k] = max(0.0, no2 + d_no2)
        O3_v[i, j, k]  = max(0.0, o3  + d_o3)
        VOC_v[i, j, k] = max(0.0, voc + d_voc)
    end
    return state
end

# ---------------------------------------------------------------------------
# A2 — Deposition velocities and dry / wet deposition
# ---------------------------------------------------------------------------

"""
    deposition_velocity(species::Symbol, cls::UInt8) -> m/s

Species-and-surface-class dependent dry deposition velocity.
`species ∈ (:NO, :NO2, :O3, :PM)`. Values (m/s):

| species | park  | water | street | roof   | wall   | glass  |
|---------|-------|-------|--------|--------|--------|--------|
| O3      | 0.007 | 0.001 | 0.0005 | 0.0005 | 0.0005 | 0.0002 |
| NO/NO2  | 0.002 | 0.001 | 0.002  | 0.002  | 0.002  | 0.001  |
| PM      | 0.005 | 0.001 | 0.001  | 0.001  | 0.001  | 0.0005 |

Unknown species/class combinations fall back to 0.001 m/s.
"""
@inline function deposition_velocity(species::Symbol, cls::UInt8)
    if species === :O3
        cls == SURFACE_PARK   && return 0.007
        cls == SURFACE_WATER  && return 0.001
        cls == SURFACE_STREET && return 0.0005
        cls == SURFACE_ROOF   && return 0.0005
        cls == SURFACE_WALL   && return 0.0005
        cls == 0x06           && return 0.0002   # SURFACE_GLASS
        return 0.001
    elseif species === :NO || species === :NO2
        cls == SURFACE_WATER && return 0.001
        cls == 0x06          && return 0.001
        return 0.002
    elseif species === :PM
        cls == SURFACE_PARK   && return 0.005
        cls == SURFACE_WATER  && return 0.001
        cls == 0x06           && return 0.0005
        return 0.001
    else
        return 0.001
    end
end

"""
    apply_dry_deposition!(field, sfc, species, dt; dz)

Remove dry-deposited mass from the first interior z-layer of `field`
(a `Field{CellCenter}`) using an explicit first-order sink

    c_new = c · (1 - V_d · dt / dz)

with the surface-class-dependent [`deposition_velocity`](@ref) for
`species`. `sfc::AbstractMatrix{UInt8}` is the ground surface class map.
The removal fraction is clamped to `[0, 1]` for stability.
"""
function apply_dry_deposition!(field, sfc::AbstractMatrix{UInt8},
                               species::Symbol, dt::Real; dz::Real)
    fv = interior_view(field)
    Nx, Ny, _ = size(fv)
    invdz = 1.0 / dz
    @inbounds for j in 1:Ny, i in 1:Nx
        Vd = deposition_velocity(species, sfc[i, j])
        frac = clamp(Vd * dt * invdz, 0.0, 1.0)
        fv[i, j, 1] *= (1.0 - frac)
    end
    return field
end

"""
    apply_wet_deposition!(field, qr, dt; scav = 5e-4)

Rain-scavenging sink applied everywhere:

    dc/dt = -scav · qr · c

Explicit Euler; the effective loss fraction is clamped to `[0, 1]`.
`qr` may be a `Field{CellCenter}` (rain mixing ratio) or an interior-sized
3-D array.
"""
function apply_wet_deposition!(field, qr, dt::Real; scav::Real = 5e-4)
    fv = interior_view(field)
    qv = qr isa AbstractArray{<:Real, 3} ? qr : interior_view(qr)
    size(qv) == size(fv) || error("apply_wet_deposition!: qr size mismatch $(size(qv)) vs $(size(fv))")
    Nx, Ny, Nz = size(fv)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        frac = clamp(scav * qv[i, j, k] * dt, 0.0, 1.0)
        fv[i, j, k] *= (1.0 - frac)
    end
    return field
end

# ---------------------------------------------------------------------------
# A3 — Emissions (traffic + biogenic)
# ---------------------------------------------------------------------------
#
# Rates below are chosen so that during rush hour a street cell adds
# ~1 ppb/s to NO in a 20 m surface layer — the right order of magnitude
# for a busy European avenue in the toy unit system.

const _TRAFFIC_NO_PEAK  = 2.0e-3   # (unit-conc)/s peak NO from traffic
const _TRAFFIC_NO2_PEAK = 5.0e-4   # (unit-conc)/s peak NO2 from traffic
const _BIOGENIC_PEAK    = 2.0e-3   # (unit-conc)/s peak biogenic VOC

"Traffic diurnal profile: bimodal Gaussian, morning 08:00 + evening 18:00."
@inline function _traffic_profile(hour::Real)
    return exp(-((hour - 8.0)  / 1.5)^2) +
           exp(-((hour - 18.0) / 1.5)^2)
end

"Biogenic isoprene profile: single Gaussian bump centred at 14:00."
@inline function _biogenic_profile(hour::Real)
    return exp(-((hour - 14.0) / 3.0)^2)
end

"""
    emission_rate(cls::UInt8, species::Symbol, hour::Real) -> conc/s

Instantaneous emission rate (in internal concentration units per second)
for `species` at a ground cell of surface class `cls` and local hour
`hour`. Traffic (NO, NO2) fires on `SURFACE_STREET` and `SURFACE_WALL`
with a bimodal rush-hour profile. Biogenic isoprene (counted as `:VOC`)
fires on `SURFACE_PARK` with an early-afternoon peak. Everything else
returns zero.
"""
@inline function emission_rate(cls::UInt8, species::Symbol, hour::Real)
    cls == SURFACE_WATER && return 0.0
    if species === :NO
        if cls == SURFACE_STREET || cls == SURFACE_WALL
            return _TRAFFIC_NO_PEAK * _traffic_profile(hour)
        end
        return 0.0
    elseif species === :NO2
        if cls == SURFACE_STREET || cls == SURFACE_WALL
            return _TRAFFIC_NO2_PEAK * _traffic_profile(hour)
        end
        return 0.0
    elseif species === :VOC
        if cls == SURFACE_PARK
            return _BIOGENIC_PEAK * _biogenic_profile(hour)
        end
        return 0.0
    else
        return 0.0
    end
end

"""
    add_emissions!(state, sfc, hour, dt; dz)

Add class- and hour-dependent surface emissions to the first interior
z-layer of every species in `state` (the [`ChemistryState`](@ref)
NamedTuple). The concentration increment per cell is
`E · dt / dz`-normalized so the rate has the same units as the state
regardless of the surface-layer thickness.
"""
function add_emissions!(state::NamedTuple, sfc::AbstractMatrix{UInt8},
                        hour::Real, dt::Real; dz::Real)
    NO_v  = interior_view(state.NO)
    NO2_v = interior_view(state.NO2)
    VOC_v = interior_view(state.VOC)
    Nx, Ny, _ = size(NO_v)
    inv_layer = dt / dz
    @inbounds for j in 1:Ny, i in 1:Nx
        c = sfc[i, j]
        NO_v[i, j, 1]  += emission_rate(c, :NO,  hour) * inv_layer
        NO2_v[i, j, 1] += emission_rate(c, :NO2, hour) * inv_layer
        VOC_v[i, j, 1] += emission_rate(c, :VOC, hour) * inv_layer
    end
    return state
end

# ---------------------------------------------------------------------------
# A4 — Two-moment aerosol prognostics
# ---------------------------------------------------------------------------

"""
    AerosolState(mesh) -> NamedTuple(q_a, N_a)

Two-moment aerosol state:

  * `q_a::Field{CellCenter}` — aerosol mass mixing ratio [kg/kg-like].
  * `N_a::Field{CellCenter}` — aerosol number concentration [#/kg-like].

Both are initialized to zero; the caller populates them (e.g. with an
initial urban background) and advects them alongside the chemistry
species.
"""
function AerosolState(mesh)
    q_a = Field(CellCenter(), mesh)
    N_a = Field(CellCenter(), mesh)
    return (q_a = q_a, N_a = N_a)
end

"""
    apply_aerosol_step!(state, dt; V_sed = 0.01, sfc = nothing, dz = 1.0,
                        k_coag = nothing)

Advance the aerosol state by `dt`:

  * First-order coagulation on number:  `dN/dt = -k · N²`, with
    `k = k_coag` if given, otherwise `1e-13 · V_sed`. Mass `q_a` is
    preserved (particles merge, mass conserved).
  * Mass-conserving sedimentation of both `q_a` and `N_a` using a
    1st-order upwind vertical flux at settling velocity `V_sed`
    (m/s, positive downward). Requires `dz`.
  * Dry deposition at the surface via [`deposition_velocity(:PM, cls)`](@ref)
    when a surface-class map `sfc` is supplied.

Numerical constraints: the sedimentation Courant number `V_sed·dt/dz`
is clamped to `≤ 1` per substep for stability.
"""
function apply_aerosol_step!(state::NamedTuple, dt::Real;
                             V_sed::Real = 0.01,
                             sfc::Union{Nothing, AbstractMatrix{UInt8}} = nothing,
                             dz::Real = 1.0,
                             k_coag::Union{Nothing, Real} = nothing)
    q_v = interior_view(state.q_a)
    N_v = interior_view(state.N_a)
    Nx, Ny, Nz = size(q_v)

    # ---- Coagulation (number only; mass preserved) ----
    k = k_coag === nothing ? 1.0e-13 * V_sed : Float64(k_coag)
    @inbounds for kk in 1:Nz, j in 1:Ny, i in 1:Nx
        Nc = N_v[i, j, kk]
        # Semi-implicit for stability at large N: dN/dt = -k N²
        # N_new = N / (1 + k · N · dt)
        N_v[i, j, kk] = Nc / (1.0 + k * Nc * dt)
    end

    # ---- Sedimentation (1st-order upwind, mass conserving) ----
    C = clamp(V_sed * dt / dz, 0.0, 1.0)
    if C > 0
        # Sweep top → bottom so we can update in place.
        @inbounds for j in 1:Ny, i in 1:Nx
            # top face flux = 0 (no aerosol source from above the domain)
            for kk in Nz:-1:2
                # flux from cell kk into kk-1
                dq = C * q_v[i, j, kk]
                dN = C * N_v[i, j, kk]
                q_v[i, j, kk]     -= dq
                N_v[i, j, kk]     -= dN
                q_v[i, j, kk - 1] += dq
                N_v[i, j, kk - 1] += dN
            end
        end
    end

    # ---- Dry deposition at the surface ----
    if sfc !== nothing
        invdz = 1.0 / dz
        @inbounds for j in 1:Ny, i in 1:Nx
            Vd = deposition_velocity(:PM, sfc[i, j])
            frac = clamp(Vd * dt * invdz, 0.0, 1.0)
            q_v[i, j, 1] *= (1.0 - frac)
            N_v[i, j, 1] *= (1.0 - frac)
        end
    end

    return state
end
