# The reference state of the case: a hydrostatic p̄(z), ρ̄(z), T̄(z), Π̄(z).
#
# The Boussinesq core carries neither pressure nor density, but the moist
# physics needs both: saturation is a function of (T, p), a flux in W m⁻²
# becomes a mixing-ratio or θ increment through ρ, and θ becomes T through
# the Exner function. Before #131 every coupling made its own choice
# (an isothermal 8 km scale height from 1013.25 hPa for fog and dew,
# ρ = 1 in the surface, building and ledger conversions, p = 1013.25 hPa
# for evaporation). This file is the one place they now come from.
#
# The profile is `Aeolus.Anelastic.reference_from_profile` — the
# discretely hydrostatic base state of the anelastic core — integrated
# from the initial θ(z) of the case and `case_surface_pressure(cfg)` at the ground,
# with the dry-air constants of `MoistThermodynamics` (R_d, c_pd, p_ref =
# 1000 hPa) so that T = θ Π̄ here is the same T the fog adjustment and
# the saturation functions see, and with the gravity of the dynamics
# (`cfg.g`). It is fixed for the run.

"""
    CaseReference

Hydrostatic reference state of a case (see `core/reference_state.jl`).

* `p̄`, `ρ̄`, `θ̄`, `T̄`, `π̄` — cell-centred profiles, length `Nz`
  (Pa, kg m⁻³, K, K, 1), copied from `state` so that couplings see
  concrete vectors;
* `state` — the `Aeolus.Anelastic.ReferenceState` itself (face profiles,
  `max_hydrostatic_residual`);
* `p_surface` — pressure at the ground (the bottom wall face), Pa;
* `cp` — the specific heat of dry air the conversions use, J kg⁻¹ K⁻¹.

Accessors: [`ref_pressure`](@ref), [`ref_density`](@ref),
[`ref_exner`](@ref), [`surface_pressure`](@ref), [`ref_ρcp`](@ref),
[`ref_θ_per_heat`](@ref).
"""
struct CaseReference
    p̄::Vector{Float64}
    ρ̄::Vector{Float64}
    θ̄::Vector{Float64}
    T̄::Vector{Float64}
    π̄::Vector{Float64}
    state::Any
    p_surface::Float64
    cp::Float64
end

# ---------------------------------------------------------------------------
# The initial profile: the Lindenberg sounding (#247), or an idealised one
# ---------------------------------------------------------------------------

const _SOUNDING_CACHE = Dict{String,Any}()

"""
Top (m) of the sounding's height-time tables: 1 km above the 3 km top of
the nesting parent (#259), so that its fixed-gradient θ top, its damping
layer and its nudging see the observed profile above the domain.
"""
const SOUNDING_TABLE_TOP = 4000.0

"The model start of `cfg` as the ISO-8601 UTC `init_time` of a dynamic driver."
case_init_time(cfg::CaseConfig) = init_time_string(cfg.date..., cfg.start_h - cfg.utc_offset)

"""
    initial_driver(cfg) -> path or nothing

The dynamic driver the initial state of `cfg` comes from: `cfg.dynamic`,
unless the nest is fed offline from it (`parent_driver = :offline`: then
`cfg.dynamic` is the parent-made driver of the nest faces, and the
initial profile is `cfg.sounding`'s).
"""
function initial_driver(cfg::CaseConfig)
    cfg.dynamic === nothing && return nothing
    nest_offline(cfg) && return nothing
    return cfg.dynamic
end

"""
    case_dynamic_inputs(cfg) -> DynamicInputs

The sounding preprocessor of the case (CFG-g, #362): the radiosonde CSV
`cfg.sounding` as the tables of a dynamic driver
([`sounding_dynamic_inputs`](@ref)) up to max([`SOUNDING_TABLE_TOP`](@ref),
the domain top), times since the case start
([`case_init_time`](@ref)), with the ground pressure the launch-site
pressure at the start reduced hypsometrically from the launch site
(`sounding.elevation`, 112 m) to `cfg.elevation` (34 m)
([`surface_pressure_at`](@ref)). What AeolusPrep.jl's `aeolusprep dynamic
--from-sounding` writes, and what a case reading the CSV uses in memory.
"""
function case_dynamic_inputs(cfg::CaseConfig)
    cfg.sounding === nothing && error("case_dynamic_inputs: the case has no sounding (cfg.sounding = nothing)")
    snd = read_sounding(cfg.sounding)
    ztop = max(SOUNDING_TABLE_TOP, cfg.extent[3])
    t0 = sounding_time(snd, cfg.date..., cfg.start_h - cfg.utc_offset)
    P = sounding_profiles(snd; z_top = ztop)
    ps = surface_pressure_at(P, t0, snd.elevation, cfg.elevation)
    return sounding_dynamic_inputs(snd; t0 = t0, init_time = case_init_time(cfg), z_top = ztop, p_surface = ps)
end

"""
    write_sounding_driver(cfg, path; preprocessor) -> path

The sounding driver of the case `cfg` at `path` (CFG-g, #362): the tables
of [`case_dynamic_inputs`](@ref), `cfg.sounding` hashed as the source and
`preprocessor` recorded. For an open, non-nested case (`cfg.lateral ===
:open`, no nesting coupling) also the steady planes of its profile
boundary data on its faces, one `time_bc` record at t = 0, taken from the
CSV in memory (`cfg.dynamic` is ignored: the file being written is not
read). The one writer of a sounding driver: AeolusPrep.jl's `aeolusprep
dynamic --from-sounding` and the tests call it.
"""
function write_sounding_driver(cfg::CaseConfig, path::AbstractString; preprocessor::AbstractString)
    mesh = RegularGrid(; size = cfg.size, extent = cfg.extent, halo = cfg.halo)
    ox, oy = anchor_xy(cfg, mesh)
    d = case_dynamic_inputs(cfg)
    kw = (; source = cfg.sounding, preprocessor = preprocessor, origin_lat = cfg.lat0, origin_lon = cfg.lon0,
          origin_x = ox, origin_y = oy)
    mkpath(dirname(abspath(path)))
    if !case_nested(cfg) && cfg.lateral === :open
        cfg0 = deepcopy(cfg); cfg0.dynamic = nothing
        st = only(open_boundary_states(case_boundary_conditions(cfg0, mesh).bcs))
        boundary_values!(st.planes, st.data, 0.0)
        record_boundary!(NestBoundaryRecorder(path, st.planes, mesh; inputs = d, kw...), 0.0, NaN,
                         plane_snapshot(st.planes, mesh))
    else
        write_dynamic_driver(path, d; dx = mesh.dx, dy = mesh.dy, kw...)
    end
    return path
end

"""
    case_sounding(cfg) -> (; profiles, t0, times, p_surface, inputs) or nothing

The observed initial state of the case: its sounding's height-time tables
`profiles` (`θ`, `qv`, `u`, `v`, `p` `ForcingTable`s and the common
height grid `z`, heights above the launch site) on a time axis in seconds
from 00 UTC of the case date, the case start `t0` on that axis, the
launch times `times`, the ground pressure `p_surface` and the
`DynamicInputs` they came from. From the sounding driver `cfg.dynamic`
(CFG-g, #362; [`initial_driver`](@ref)), or else from the CSV
`cfg.sounding` preprocessed in memory ([`case_dynamic_inputs`](@ref), the
same tables value for value); `nothing` for the idealised profile. The
driver holds the whole sounding series, so a case starting at another
time than the driver's `init_time` reads the same tables with the time
axis moved (exact for whole seconds) and its ground pressure
recomputed at its own start ([`surface_pressure_at`](@ref), as the
preprocessor does); at the driver's `init_time` it is the stored
`init_p_surface`. Read once per source and cached.
"""
function case_sounding(cfg::CaseConfig)
    path = initial_driver(cfg)
    path === nothing && cfg.sounding === nothing && return nothing
    it = case_init_time(cfg)
    src = path === nothing ? string(cfg.sounding, "|", max(SOUNDING_TABLE_TOP, cfg.extent[3]), "|", cfg.elevation) :
          string(path, "|", isfile(path) ? file_fnv1a64(path) : "", "|", cfg.elevation)
    return get!(_SOUNDING_CACHE, string(src, "|", it)) do
        d = path === nothing ? case_dynamic_inputs(cfg) : read_dynamic_driver(path; planes = false)
        _case_sounding(d, path === nothing ? cfg.sounding : path, it, cfg.elevation)
    end
end

_utc(s::AbstractString) = DateTime(rstrip(s, 'Z'))

function _case_sounding(d::DynamicInputs, src, it, elevation)
    get(d.attributes, "dynamic_source", "") == "sounding" ||
        error("$src is not a sounding driver (dynamic_source = $(repr(get(d.attributes, "dynamic_source", "")))): " *
              "[initial] dynamic of a case without the offline nest is the sounding's " *
              "(AeolusPrep.jl: aeolusprep dynamic --from-sounding)")
    isempty(d.init_time) && error("the sounding driver $src has no init_time")
    for k in (:profile_theta, :profile_qv, :profile_u, :profile_v, :profile_p)
        haskey(d.tables, k) || error("the sounding driver $src has no ls_$k")
    end
    d.p_surface === nothing && error("the sounding driver $src has no init_p_surface")
    # the tables' axis: seconds from 00 UTC of the case date
    t0 = init_day_seconds(it)
    off = Dates.value(_utc(d.init_time) - DateTime(Dates.Date(_utc(it)))) / 1000
    tab(k) = shift_table(d.tables[k], off)
    θ = tab(:profile_theta)
    P = (θ = θ, qv = tab(:profile_qv), u = tab(:profile_u), v = tab(:profile_v), p = tab(:profile_p), z = θ.z)
    ps = d.p_surface
    if d.init_time != it                     # another start than the driver's: the pressure at this one
        haskey(d.attributes, "sounding_elevation") || error("the sounding driver $src has no sounding_elevation")
        ps = surface_pressure_at(P, t0, Float64(d.attributes["sounding_elevation"]), elevation)
    end
    return (; profiles = P, t0 = t0, times = copy(θ.t), p_surface = ps, inputs = d)
end

"""
    case_start_time(cfg) -> t (s)

The case start (`cfg.date`, local hour `cfg.start_h`, local = UTC +
`cfg.utc_offset`) on the time axis of the case's sounding tables
([`case_sounding`](@ref): seconds from 00 UTC of the case date).
06:00 CEST on 29 June 2019 is 04:00 UTC, 14400 s, between the 22:45 and
04:45 UTC releases.
"""
function case_start_time(cfg::CaseConfig)
    S = case_sounding(cfg)
    S === nothing && error("case_start_time: the case has no sounding (cfg.sounding = nothing)")
    return S.t0
end

"""
    initial_θ(cfg) -> (x, y, z) -> θ

The initial potential temperature of the case, the one definition:
`build_case` sets `sim.θ` from it and [`build_reference`](@ref)
integrates the reference state from it (so the nesting parent, which
uses the same reference, follows).

* With a sounding — the observed profiles (the Berlin case: Lindenberg,
  WMO 10393, 29/30 June 2019; the sounding driver `cfg.dynamic` or the CSV `cfg.sounding`,
  [`case_sounding`](@ref)) at the case start [`case_start_time`](@ref):
  `forcing_value(θ_table, z, t₀)` of `sounding_profiles`, linear in
  height between the reported levels and linear in time between the two
  launches that bracket t₀. θ uses the dry Exner function of
  `MoistThermodynamics`; heights are hypsometric heights above the
  launch site, applied as heights above the case's ground (the profile
  is assumed horizontally homogeneous over the 60 km to Berlin and
  terrain-following over the 78 m elevation difference).
* no sounding (`cfg.sounding === nothing` and no sounding driver) — the **idealised** profile
  ``θ(z) = θ_s + \\max(0, z - z_i)Γ`` (`cfg.θ_surface`,
  `cfg.θ_inversion_z`, `cfg.θ_lapse`), kept for `:winter` (no winter
  sounding yet) and `:smoke` (its checks need a capping inversion inside
  a 100 m box).
"""
function initial_θ(cfg::CaseConfig)
    S = case_sounding(cfg)
    if S === nothing
        θs, zi, Γ = cfg.θ_surface, cfg.θ_inversion_z, cfg.θ_lapse
        return (x, y, z) -> θs + max(0, z - zi) * Γ
    end
    tab, t0 = S.profiles.θ, case_start_time(cfg)
    return (x, y, z) -> forcing_value(tab, z, t0)
end

"""
    initial_qv(cfg) -> (x, y, z) -> q_v

The initial water-vapour specific humidity (kg kg⁻¹), from the same
source as [`initial_θ`](@ref): the sounding's dew points at the case
start (``q_v`` with ``e = e_s(T_d)``, `MoistThermodynamics`), or for
the idealised profile ``q_v = q_{v0} e^{-z/H_q}`` (`cfg.qv0`,
`cfg.qv_scale`).
"""
function initial_qv(cfg::CaseConfig)
    S = case_sounding(cfg)
    if S === nothing
        qv0, Hq = cfg.qv0, cfg.qv_scale
        return (x, y, z) -> qv0 * exp(-z / Hq)
    end
    tab, t0 = S.profiles.qv, case_start_time(cfg)
    return (x, y, z) -> forcing_value(tab, z, t0)
end

"""
    initial_inversion_height(cfg) -> z_i (m)

Height of the capping inversion of the initial profile: `cfg.θ_inversion_z`
for the idealised profile; for a sounding, the parcel-method
`boundary_layer_height` (TurbulenceStability, the one z_i diagnostic) of
the initial virtual potential temperature on the sounding's height grid.
At 06 CEST on 29 June 2019 the surface-based inversion gives z_i = 0
(stable from the ground); at noon the mixed layer reaches about 1.2 km.
"""
function initial_inversion_height(cfg::CaseConfig)
    S = case_sounding(cfg)
    S === nothing && return cfg.θ_inversion_z
    MT = Aeolus.MoistThermodynamics
    t0, P = case_start_time(cfg), S.profiles
    θv = [MT.virtual_temperature(forcing_value(P.θ, z, t0), MT.PhasePartition(forcing_value(P.qv, z, t0)))
          for z in P.z]                                  # T_v/T does not depend on T
    return boundary_layer_height(P.z, θv; method = :parcel)
end

"""
    case_surface_pressure(cfg) -> p (Pa)

Pressure at the case's ground: `cfg.p_surface` if set; otherwise, with a
sounding, its `init_p_surface` ([`case_sounding`](@ref)): the launch-site
surface pressure at the case start reduced hypsometrically from the
launch site (112 m) to `cfg.elevation` (34 m) with the surface virtual
temperature ([`surface_pressure_at`](@ref)); for the idealised profile
101325 Pa.
"""
function case_surface_pressure(cfg::CaseConfig)
    cfg.p_surface === nothing || return cfg.p_surface
    S = case_sounding(cfg)
    S === nothing && return 101325.0
    return S.p_surface
end

"""
    build_reference(cfg, mesh) -> CaseReference

The hydrostatic reference state of the case: θ̄(z) is the initial
potential-temperature profile [`initial_θ`](@ref) of `cfg` sampled at
the cell centres, integrated from [`case_surface_pressure`](@ref) with
`Aeolus.Anelastic.reference_from_profile`, the dry-air constants of
`Aeolus.MoistThermodynamics` and `g = cfg.g`.
"""
function build_reference(cfg::CaseConfig, mesh)
    c = Aeolus.MoistThermodynamics.DEFAULT_CONSTANTS
    R_d = c.gas_constant / c.molmass_dry
    θ0 = initial_θ(cfg)
    ps = case_surface_pressure(cfg)
    state = Aeolus.Anelastic.reference_from_profile(mesh, z -> θ0(0.0, 0.0, z), ps;
                                                    R_d = R_d, cp = c.cp_dry, p₀ = c.p_ref,
                                                    g_accel = cfg.g)
    return CaseReference(copy(state.p̄), copy(state.ρ̄), copy(state.θ̄), copy(state.T̄),
                         copy(state.π̄), state, ps, Float64(c.cp_dry))
end

"Reference pressure (Pa) at the centre of level `k`, or the whole profile."
ref_pressure(case, k::Integer) = case.ref.p̄[k]
ref_pressure(case) = case.ref.p̄
"""
    ref_pressure_storage(case) -> Vector

The reference pressure indexed by the storage level `k` of the case mesh
(length `Nz + 2Hz`, the halos take the nearest interior level): the `p`
of the closures' cloudy-air ``N^2`` (`qc`/`p` keywords, #239), which
evaluate the saturated ``N^2`` at ``(θ, p̄_k)``, the pressure fog is
adjusted at.
"""
function ref_pressure_storage(case)
    m = case.mesh
    return [case.ref.p̄[clamp(k - m.Hz, 1, m.Nz)] for k in 1:m.Nz + 2m.Hz]
end

"Reference density (kg m⁻³) at the centre of level `k`, or the whole profile."
ref_density(case, k::Integer) = case.ref.ρ̄[k]
ref_density(case) = case.ref.ρ̄
"Reference Exner function ``Π̄ = (p̄/p_{ref})^{R_d/c_{pd}}`` at level `k` (T = θ Π̄)."
ref_exner(case, k::Integer) = case.ref.π̄[k]
"Pressure at the ground (Pa)."
surface_pressure(case) = case.ref.p_surface
"``ρ̄_k c_{pd}`` (J m⁻³ K⁻¹): the heat that raises the *temperature* of level `k` by 1 K."
ref_ρcp(case, k::Integer) = case.ref.ρ̄[k] * case.ref.cp

"""
    ref_θ_per_heat(case, k; cp = case.ref.cp) -> K m³ J⁻¹

The potential-temperature increment of level `k` per J m⁻³ of heat,
``1 / (ρ̄_k c_p Π̄_k)``. Heat raises the temperature by ``q / (ρ̄ c_p)``
and, since ``T = θ Π̄``, θ by that over ``Π̄``. A flux `F` (W m⁻²) into a
cell of thickness `Δz` over `τ` s raises θ by
`F * τ / Δz * ref_θ_per_heat(case, k)`. The one conversion every
coupling that heats the air uses, and the inverse of the weight with
which the air-heat ledger (`src/budgets/meter.jl`) counts θ, so a flux `F`
appears in the ledger as exactly `F τ`. (Before #253 the couplings
divided by ``ρ̄ c_p`` only and the air received ``Π̄_k F`` — 0.4 % too
much at the ground, where ``Π̄ ≈ 1.004``.) `cp` defaults to the case's
dry-air value; the surface exchange passes the moist heat capacity of
the first air cell.
"""
ref_θ_per_heat(case, k::Integer; cp::Real = case.ref.cp) =
    1 / (case.ref.ρ̄[k] * cp * case.ref.π̄[k])

"""
    hydrostatic_check(cfg, ref, mesh; nsub = 1000) -> max relative |p̄ − p_hyd| / p_hyd

The reference pressure against an independent integration of the
continuous hydrostatic equation ``dp/dz = -g p / (R_d θ(z) Π(p))`` by
RK4 with `nsub` sub-steps per level, from the same surface pressure and
θ(z) (the one used for `ref.state`, evaluated at the exact heights, not
interpolated). The discrete balance of the ReferenceState is exact on
the grid by construction; this measures how far that grid balance is
from the continuous atmosphere.
"""
function hydrostatic_check(cfg::CaseConfig, ref::CaseReference, mesh; nsub::Int = 1000)
    θ0 = initial_θ(cfg)
    zc = [(k - 0.5) * mesh.dz for k in 1:mesh.Nz]
    p = hydrostatic_pressure(z -> θ0(0.0, 0.0, z), zc, ref.p_surface; g = cfg.g, nsub = nsub)
    return maximum(abs.(ref.p̄ .- p) ./ p)
end

"""
    hydrostatic_pressure(θ, z, p_surf; g = 9.81, nsub = 1000) -> Vector (Pa)

The pressure at the heights `z` (m, increasing, above the ground) of a
hydrostatic atmosphere with potential temperature `θ(z)` (a function) and
ground pressure `p_surf`: the continuous ``dp/dz = -g p/(R_d θ(z) Π(p))``
integrated by RK4 with `nsub` sub-steps per interval, with the dry-air
constants of `MoistThermodynamics` (R_d, c_pd, p_ref = 1000 hPa). The one
integration of the case: [`hydrostatic_check`](@ref) measures the child's
discrete reference state against it, and the nesting parent (#259),
whose stretched levels the discrete `Anelastic` reference does not take,
uses it as its reference pressure ([`parent_reference`](@ref)).
"""
function hydrostatic_pressure(θ, z::AbstractVector, p_surf::Real; g::Real = 9.81, nsub::Int = 1000)
    c = Aeolus.MoistThermodynamics.DEFAULT_CONSTANTS
    R_d = c.gas_constant / c.molmass_dry
    κ = R_d / c.cp_dry
    f(zz, pp) = -g * pp / (R_d * θ(zz) * (pp / c.p_ref)^κ)
    out = similar(z, Float64)
    p, zz = Float64(p_surf), 0.0
    for k in eachindex(z)
        h = (z[k] - zz) / nsub
        for _ in 1:nsub
            k1 = f(zz, p); k2 = f(zz + h / 2, p + h / 2 * k1)
            k3 = f(zz + h / 2, p + h / 2 * k2); k4 = f(zz + h, p + h * k3)
            p += h / 6 * (k1 + 2k2 + 2k3 + k4); zz += h
        end
        out[k] = p
    end
    return out
end
