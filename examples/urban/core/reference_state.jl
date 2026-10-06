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
[`ref_exner`](@ref), [`surface_pressure`](@ref), [`ref_ρcp`](@ref).
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
    case_sounding(cfg) -> (; sounding, profiles) or nothing

The radiosonde sounding of the case (`cfg.sounding`, read with
`Aeolus.LargeScaleForcing.read_sounding`) and its height-time tables
(`sounding_profiles`, heights above the launch site up to
max(3 km, the domain top)); `nothing` for the idealised profile
(`cfg.sounding === nothing`). Read once per file and cached.
"""
function case_sounding(cfg::CaseConfig)
    cfg.sounding === nothing && return nothing
    ztop = max(3000.0, cfg.extent[3])
    key = string(cfg.sounding, "|", ztop)
    return get!(_SOUNDING_CACHE, key) do
        snd = read_sounding(cfg.sounding)
        (; sounding = snd, profiles = sounding_profiles(snd; z_top = ztop))
    end
end

"""
    case_start_time(cfg) -> t (s)

The case start (`cfg.date`, local hour `cfg.start_h`, local = UTC +
`cfg.utc_offset`) on the time axis of the case's sounding (seconds from
its epoch, `sounding_time`). 06:00 CEST on 29 June 2019 is 04:00 UTC,
between the 22:45 and 04:45 UTC releases.
"""
function case_start_time(cfg::CaseConfig)
    S = case_sounding(cfg)
    S === nothing && error("case_start_time: the case has no sounding (cfg.sounding = nothing)")
    return sounding_time(S.sounding, cfg.date..., cfg.start_h - cfg.utc_offset)
end

"""
    initial_θ(cfg) -> (x, y, z) -> θ

The initial potential temperature of the case, the one definition:
`build_case` sets `sim.θ` from it and [`build_reference`](@ref)
integrates the reference state from it (so the nesting parent, which
uses the same reference, follows).

* Default — the observed sounding `cfg.sounding` (Lindenberg, WMO 10393,
  29/30 June 2019) at the case start [`case_start_time`](@ref):
  `forcing_value(θ_table, z, t₀)` of `sounding_profiles`, linear in
  height between the reported levels and linear in time between the two
  launches that bracket t₀. θ uses the dry Exner function of
  `MoistThermodynamics`; heights are hypsometric heights above the
  launch site, applied as heights above the case's ground (the profile
  is assumed horizontally homogeneous over the 60 km to Berlin and
  terrain-following over the 78 m elevation difference).
* `cfg.sounding === nothing` — the **idealised** profile
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
sounding, the sounding's surface pressure at the case start reduced
hypsometrically from the launch site (`sounding.elevation`, 112 m) to
`cfg.elevation` (34 m) with the surface virtual temperature,
``p = p_L \\exp\\bigl(g_0 (z_L - z)/(R_d T_v)\\bigr)``; for the idealised
profile 101325 Pa.
"""
function case_surface_pressure(cfg::CaseConfig)
    cfg.p_surface === nothing || return cfg.p_surface
    S = case_sounding(cfg)
    S === nothing && return 101325.0
    MT = Aeolus.MoistThermodynamics
    c = MT.DEFAULT_CONSTANTS
    t0 = case_start_time(cfg)
    P = S.profiles
    pL = forcing_value(P.p, 0.0, t0)
    T = forcing_value(P.θ, 0.0, t0) * MT.exner_function(pL)
    Tv = MT.virtual_temperature(T, MT.PhasePartition(forcing_value(P.qv, 0.0, t0)))
    R_d = c.gas_constant / c.molmass_dry
    return pL * exp(c.grav * (S.sounding.elevation - cfg.elevation) / (R_d * Tv))
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
"Reference density (kg m⁻³) at the centre of level `k`, or the whole profile."
ref_density(case, k::Integer) = case.ref.ρ̄[k]
ref_density(case) = case.ref.ρ̄
"Reference Exner function ``Π̄ = (p̄/p_{ref})^{R_d/c_{pd}}`` at level `k` (T = θ Π̄)."
ref_exner(case, k::Integer) = case.ref.π̄[k]
"Pressure at the ground (Pa)."
surface_pressure(case) = case.ref.p_surface
"``ρ̄_k c_{pd}`` (J m⁻³ K⁻¹): converts a heat flux into level `k` to a θ increment."
ref_ρcp(case, k::Integer) = case.ref.ρ̄[k] * case.ref.cp

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
    c = Aeolus.MoistThermodynamics.DEFAULT_CONSTANTS
    R_d = c.gas_constant / c.molmass_dry
    κ = R_d / c.cp_dry
    θ0 = initial_θ(cfg)
    θ(z) = θ0(0.0, 0.0, z)
    f(z, p) = -cfg.g * p / (R_d * θ(z) * (p / c.p_ref)^κ)
    err = 0.0
    p, z = ref.p_surface, 0.0
    zc = [(k - 0.5) * mesh.dz for k in 1:mesh.Nz]
    for k in 1:mesh.Nz
        h = (zc[k] - z) / nsub
        for _ in 1:nsub
            k1 = f(z, p); k2 = f(z + h / 2, p + h / 2 * k1)
            k3 = f(z + h / 2, p + h / 2 * k2); k4 = f(z + h, p + h * k3)
            p += h / 6 * (k1 + 2k2 + 2k3 + k4); z += h
        end
        err = max(err, abs(ref.p̄[k] - p) / p)
    end
    return err
end
