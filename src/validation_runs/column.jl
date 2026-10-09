# The `column` run driver (CFG-e, #360): the single-column and box cases
# that have no 3-D dynamics, selected by [column] case:
#
#   dycoms_rf01    the DYCOMS-II RF01 column and its longwave solve (gas +
#                  cloud optics, two-stream), against the intercomparison's
#                  prescribed flux (rf01.jl);
#   parisfog_2011  the 1-D RadiationFog column of ParisFog, 14-15 Nov 2011
#                  (parisfog_2011_column, run! of Aeolus.RadiationFog);
#   kessler_box    Kessler microphysics in a small box with no dynamics
#                  (apply_kessler_step! with a sequence of steps);
#   droplet_parcel the ascending / stationary parcel of the spectral-
#                  broadening validation (run_broadening_case, Era IV P45).
#
# Each case has its own section; a key set in the section of a case that is
# not selected is an error. Moved from examples/validation/column_driver.jl
# (EX-c #440).
# Configs: examples/validation/configs/{dycoms_rf01, parisfog_2011,
# spectral_broadening}.toml, examples/kernels/configs/cloud_spinup.toml.

"""
    ColumnDriver()

The `column` run type: `[run] model = "column"`, the single-column and
box cases with no 3-D dynamics, selected by `[column] case` (see the
option reference). Each case writes its numbers to
`output/results.toml`. Configs: `examples/validation/configs/`
(`dycoms_rf01`, `parisfog_2011`, `spectral_broadening`, …).
"""
struct ColumnDriver <: AbstractRunDriver end

Config.driver_name(::ColumnDriver) = "column"
Config.config_dir(::ColumnDriver) = VALIDATION_CONFIG_DIR

const _CASE_SECTIONS = ("dycoms_rf01" => "dycoms_rf01", "parisfog_2011" => "fog",
                        "kessler_box" => "kessler_box", "droplet_parcel" => "parcel")
# doc and units of each option (units: CF-style strings; "1": dimensionless)
const _RF01_DOCS = Dict(:zi => ("inversion height", "m"), :psfc => ("surface pressure", "Pa"),
                        :sst => ("sea-surface temperature", "K"), :qt_bl => ("total water below the inversion", "kg kg^-1"),
                        :qt_ft => ("total water above", "kg kg^-1"), :thl_bl => ("θ_l below the inversion", "K"),
                        :reff => ("droplet effective radius", "μm"), :f0 => ("F0 of the prescribed flux", "W m^-2"),
                        :f1 => ("F1 of the prescribed flux", "W m^-2"), :kappa => ("κ of the prescribed flux", "m^2 kg^-1"),
                        :dz_fine => ("layer depth of the fine (LES) part", "m"), :z_les => ("top of the fine part", "m"),
                        :n_coarse => ("stretched layers above it", ""), :z_top => ("top of the column", "m"))

const _FOG_DOCS = Dict(
    :latitude => ("site latitude", "degrees_north"), :longitude => ("site longitude (east positive)", "degrees_east"),
    :start => ("start (year, month, day, hour, minute) UTC", ""), :z0m => ("momentum roughness length", "m"),
    :z0h => ("heat roughness length", "m"), :albedo => ("surface albedo", "1"), :emissivity => ("surface emissivity", "1"),
    :admittance => ("thermal admittance of the soil top, force-restore", "J m^-2 K^-1 s^-1/2"),
    :T_deep => ("deep soil temperature of the force-restore equation", "K"),
    :β => ("soil moisture availability for bare evaporation", "1"), :λ => ("asymptotic mixing length", "m"),
    :K_min => ("background diffusivity", "m^2 s^-1"), :ug => ("geostrophic wind u", "m s^-1"),
    :vg => ("geostrophic wind v", "m s^-1"), :N_d => ("droplet number", "m^-3"),
    :k_shape => ("shape factor of the effective radius (fog_effective_radius)", "1"),
    :droplet_feedback => ("fog droplets are seen by the radiation (false: transparent)", ""),
    :shortwave => ("the sun rises (false: the night goes on)", ""),
    :settling => ("droplets sediment at terminal_velocity(r_eff)", ""), :dt => ("the step; 5 s is converged", "s"),
    :dt_rad => ("radiation every dt_rad", "s"), :dt_gas => ("gas optics every dt_gas", "s"),
    :dt_out => ("diagnostics every dt_out", "s"), :lwc_fog => ("fog is LWC ≥ lwc_fog", "kg m^-3"),
    :z_obs => ("fog at the ground: fog in the layer containing z_obs", "m"))
const _PARCEL_DOCS = Dict(
    :T0 => ("initial temperature", "K"), :p0 => ("initial pressure", "Pa"),
    :S0 => ("initial supersaturation (the parcel starts at cloud base)", "1"),
    :N => ("droplet number concentration", "m^-3"), :r0 => ("median radius of the initial spectrum", "m"),
    :ε0 => ("log-normal width of the initial spectrum (0: monodisperse)", "1"),
    :solute_mass => ("NaCl per droplet", "kg"), :n_sd => ("superdroplets (reference only)", ""),
    :w => ("updraught", "m s^-1"), :σ_S => ("sub-grid supersaturation spread", "1"),
    :τ => ("Lagrangian correlation time of the fluctuations", "s"), :t_end => ("duration", "s"),
    :dt => ("output and coupling step", "s"), :seed => ("initial radii and fluctuation noise", ""))
_col_docs(d) = Dict(k => v[1] for (k, v) in d)
_col_units(d) = Dict(k => v[2] for (k, v) in d if !isempty(v[2]))

function Config.driver_sections(::ColumnDriver)
    col = section("column"; doc = "Which column case runs.") do s
        option!(s, "case", Symbol; default = :dycoms_rf01, allowed = Symbol.(first.(_CASE_SECTIONS)),
                doc = "dycoms_rf01 ([dycoms_rf01]), parisfog_2011 ([fog], [fog_sounding]), " *
                      "kessler_box ([kessler_box]), droplet_parcel ([parcel])")
    end
    rf01 = section("dycoms_rf01"; doc = "DYCOMS-II RF01 (Stevens et al. 2005): sounding, prescribed-flux constants, column grid, longwave.") do s
        for (k, v) in pairs(RF01_DEFAULTS)
            option!(s, String(k), typeof(v); default = v, doc = _RF01_DOCS[k][1], units = _RF01_DOCS[k][2])
        end
        option!(s, "ngpt_per_band", Int; default = 4, range = (1, 64), doc = "g-points per longwave band")
        option!(s, "sfc_emissivity", Float64; units = "1", default = 0.98, range = (0, 1), doc = "surface emissivity")
    end
    fog = kwdef_section("fog", FogColumnSettings; docs = _col_docs(_FOG_DOCS), units = _col_units(_FOG_DOCS),
                        doc = "ParisFog 2011: the RadiationFog column settings (FogColumnSettings; keys are its fields, β → beta, λ → lambda).")
    snd = section("fog_sounding"; doc = "ParisFog 2011: the initial sounding (parisfog_2011_column).") do s
        option!(s, "h", Float64; default = 1000.0, range = (0, Inf), open_lower = true, units = "m", doc = "column depth")
        option!(s, "t_sfc", Float64; default = 276.0, range = (0, Inf), open_lower = true, units = "K",
                doc = "surface air temperature")
        option!(s, "p_sfc", Float64; default = 1.02e5, range = (0, Inf), open_lower = true, units = "Pa",
                doc = "surface pressure")
        option!(s, "duration", Float64; default = 14 * 3600.0, range = (0, Inf), units = "s",
                doc = "simulated time (whole steps of fog.dt)")
    end
    kb = section("kessler_box"; doc = "Kessler microphysics in a box with no dynamics.") do s
        option!(s, "size", NTuple{3,Int}; default = (4, 4, 32), range = (1, 4096), doc = "cells in x, y, z")
        option!(s, "height", Float64; default = 2000.0, range = (0, Inf), open_lower = true, units = "m",
                doc = "box side (a cube)")
        option!(s, "qv", Float64; default = 0.01, range = (0, Inf), units = "kg kg^-1", doc = "uniform vapour")
        option!(s, "ql", Float64; default = 3e-3, range = (0, Inf), units = "kg kg^-1",
                doc = "cloud water of the initial cloud layer")
        option!(s, "cloud_layer", NTuple{2,Float64}; units = "1", default = (0.6, 0.9), range = (0, 1),
                doc = "the cloud layer, as fractions of the height (open interval)")
        option!(s, "dt", Vector{Float64}; default = [60.0, 60.0, 120.0, 120.0, 120.0], range = (0, Inf),
                open_lower = true, units = "s", doc = "the steps, one microphysics step each")
    end
    pc = kwdef_section("parcel", BroadeningCase; docs = _col_docs(_PARCEL_DOCS), units = _col_units(_PARCEL_DOCS),
                       doc = "The parcel of the spectral-broadening validation (BroadeningCase; σ_S → sigma_s, τ → tau, ε0 → epsilon0).")
    option!(pc, "model", Symbol; default = :superdroplet, allowed = [:superdroplet, :bulk],
            doc = "superdroplet (per-droplet stochastic supersaturation, the reference) or bulk (the moment closure)")
    return Section[col, rf01, fog, snd, kb, pc]
end

function _column_checks(cfg)
    e = no_checkpoints(cfg)
    append!(e, selected_case_sections(cfg, "column.case", (_CASE_SECTIONS..., "parisfog_2011" => "fog_sounding")))
    kb = cfg.kessler_box
    kb.cloud_layer[1] < kb.cloud_layer[2] ||
        push!(e, ConfigError("kessler_box.cloud_layer", "expected [bottom, top] with bottom < top"))
    return e
end
Config.driver_checks(::ColumnDriver) = Any[
    """`[checkpoint]` stays off: a column case runs to the end in one call
    a key of the section of a case that `column.case` does not select must keep its default (it would be ignored)
    `kessler_box.cloud_layer` is [bottom, top] with bottom < top""" => _column_checks]

"""
    ColumnRun

What the `column` driver builds: `case` (a Symbol), `state` (the case's
object: the RF01 result, a `FogColumn`, the Kessler fields, the parcel
histories), `snapshots` (kessler_box: the horizontal-mean profiles after
each step) and `cfg`.
"""
mutable struct ColumnRun
    case::Symbol
    state::Any
    snapshots::Vector{Any}
    cfg::RunConfig
end

_rf01_params(cfg) = NamedTuple{keys(RF01_DEFAULTS)}(Tuple(cfg.dycoms_rf01[String(k)] for k in keys(RF01_DEFAULTS)))

function Config.build(::ColumnDriver, cfg::RunConfig)
    case = cfg.column.case
    if case === :parisfog_2011
        s = cfg.fog_sounding
        col = parisfog_2011_column(; H = s.h, T_sfc = s.t_sfc, p_sfc = s.p_sfc,
                                   settings = kwdef_value(FogColumnSettings, cfg.fog))
        return ColumnRun(case, col, Any[], cfg)
    elseif case === :kessler_box
        k = cfg.kessler_box; H = k.height
        mesh = RegularGrid(; size = k.size, extent = (H, H, H), halo = (1, 1, 1))
        qv = Field(CellCenter(), mesh); set!(qv, (x, y, z) -> k.qv)
        ql = Field(CellCenter(), mesh)
        z0, z1 = k.cloud_layer .* H
        set!(ql, (x, y, z) -> z0 < z < z1 ? k.ql : 0.0)
        qr = Field(CellCenter(), mesh); set!(qr, (x, y, z) -> 0.0)
        return ColumnRun(case, (; qv, ql, qr, mesh), Any[], cfg)
    end
    return ColumnRun(case, nothing, Any[], cfg)        # dycoms_rf01, droplet_parcel: built in run!
end

"The horizontal-mean profile of a cell-centred field."
function hmean_profile(f)
    v = interior_view(f)
    return [sum(view(v, :, :, k)) / (size(v, 1) * size(v, 2)) for k in 1:size(v, 3)]
end

function Simulations.run!(::ColumnDriver, r::ColumnRun, rf::RunFolder, ::CheckpointSchedule; callback = nothing)
    cfg = r.cfg; nsteps = 1
    if r.case === :dycoms_rf01
        d = cfg.dycoms_rf01
        r.state = rf01_radiation(_rf01_params(cfg); ngpt_per_band = d.ngpt_per_band,
                                 sfc_emissivity = d.sfc_emissivity)
        s = r.state
        write_results(rf, Dict("z_lev" => s.z_lev, "z_lay" => s.z_lay, "F_up_all" => s.Fup_all,
                               "F_up_clear" => s.Fup_clear, "F_spec" => s.Fspec, "heating_K_per_day" => s.Kday,
                               "ql" => s.col.ql, "rho" => s.col.ρ, "dz" => s.col.dz, "n_fine" => s.nf))
    elseif r.case === :parisfog_2011
        col = r.state
        Aeolus.RadiationFog.run!(col, cfg.fog_sounding.duration)
        nsteps = round(Int, cfg.fog_sounding.duration / col.set.dt)
        lc = fog_life_cycle(col.history)
        @info("fog life cycle", subsystem = :run, onset = lc.onset, dissipation = lc.dissipation,
              max_lwp = lc.max_lwp, max_top = lc.max_top)
        H = col.history
        write_results(rf, Dict("onset_utc_hour" => lc.onset, "dissipation_utc_hour" => lc.dissipation,
                               "max_lwp" => lc.max_lwp, "max_top" => lc.max_top, "t_max_lwp_utc_hour" => lc.t_max_lwp,
                               "utc_hour" => [h.utc_hour for h in H], "lwp" => [h.lwp for h in H],
                               "fog_top" => [h.fog_top for h in H], "lw_net_sfc" => [h.lw_net_sfc for h in H],
                               "lw_net_top" => [h.lw_net_top for h in H], "lw_fog" => [h.lw_fog for h in H],
                               "lw_top_cooling" => [h.lw_top_cooling for h in H], "C_LW" => [h.C_LW for h in H]))
    elseif r.case === :kessler_box
        s = r.state
        push!(r.snapshots, (t = 0.0, ql = hmean_profile(s.ql), qr = hmean_profile(s.qr)))
        t = 0.0
        for dt in cfg.kessler_box.dt
            apply_kessler_step!(s.qv, s.ql, s.qr, dt; Vt = :spectrum)   # the Marshall-Palmer speed of each cell
            t += dt
            push!(r.snapshots, (t = t, ql = hmean_profile(s.ql), qr = hmean_profile(s.qr)))
        end
        nsteps = length(cfg.kessler_box.dt)
        write_results(rf, Dict("t" => [x.t for x in r.snapshots],
                               "ql_max" => [maximum(x.ql) for x in r.snapshots],
                               "qr_max" => [maximum(x.qr) for x in r.snapshots],
                               "ql_final" => r.snapshots[end].ql, "qr_final" => r.snapshots[end].qr,
                               "ql_profiles" => [x.ql for x in r.snapshots], "qr_profiles" => [x.qr for x in r.snapshots]))
    else                                                  # droplet_parcel
        case = kwdef_value(BroadeningCase, cfg.parcel; exclude = ())
        r.state = run_broadening_case(case; model = cfg.parcel.model)
        h = r.state
        nsteps = length(h.t) - 1
        write_results(rf, Dict("t" => h.t, "epsilon" => h.ε, "sigma_r2" => h.σ_r2, "S" => h.S, "T" => h.T,
                               "mean_radius" => h.mean_radius, "sigma_r2_ou" => parcel_ou_sigma_r2(h, case)))
    end
    callback === nothing || callback(r, nsteps)
    return r
end

"""
    parcel_ou_sigma_r2(h, case) -> Vector

The width ``σ_{r^2}(t) = 2G\\,\\sqrt{\\mathrm{Var}\\int_0^t S'\\,dt'}`` that
Ornstein–Uhlenbeck supersaturation fluctuations (`supersaturation_integral_variance`,
σ_S and τ of the [`BroadeningCase`](@ref) `case`) give a monodisperse
population, at the times of the parcel history `h`, with the growth
factor ``G = 1/(F_k + F_d)`` at the final temperature and `case.p0`: the
analytic curve the stationary spectral-broadening case is compared with
(Sardina et al. 2015, PRL 115, 184501).
"""
function parcel_ou_sigma_r2(h, case)
    G = 1 / (Aeolus.Superdroplet.fk_thermal(h.T[end]) + Aeolus.Superdroplet.fd_vapor(h.T[end], case.p0))
    return [2G * sqrt(supersaturation_integral_variance(case.σ_S, case.τ, t)) for t in h.t]
end

function Config.state_fields(r::ColumnRun)
    s = r.state
    r.case === :dycoms_rf01 && return (F_up_all = s.Fup_all, F_up_clear = s.Fup_clear, K_day = s.Kday, ql = s.col.ql)
    r.case === :parisfog_2011 && return (T = s.T, qv = s.qv, ql = s.ql, u = s.u, v = s.v)
    r.case === :kessler_box && return (qv = s.qv.data, ql = s.ql.data, qr = s.qr.data)
    return (epsilon = s.ε, sigma_r2 = s.σ_r2, S = s.S, T = s.T)
end

register_driver!(ColumnDriver())
