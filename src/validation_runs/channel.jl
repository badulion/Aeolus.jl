# The `channel` run driver (CFG-e, #360): the wall-modelled channel LES of
# Aeolus.WallModels — the neutral half channel (`half_channel_case`, P54-d)
# and the smooth-wall plane channel (`channel_case`, P57-b) — built from a
# config and run with `run_channel!` (its own adaptive step, the wall
# stress, the statistics and the checkpoints of #245).
#
# A driver of its own, not more keys of `simulation`: a WallBoundedCase
# has its own stepper (`advance_channel!`: wall stress in every RK3
# substage, mean pressure gradient, matching-height inputs and temporal
# filters) and its own statistics; repeating that in the simulation driver
# would be a second copy of the physics.
#
# Moved from examples/validation/channel_driver.jl (EX-c #440). Configs:
# examples/validation/configs/neutral_abl.toml, log_layer_mismatch.toml.

"""
    ChannelDriver()

The `channel` run type: `[run] driver = "channel"`, the wall-modelled
channel LES of `WallModels` (`[channel] kind`: the neutral half channel
or the smooth-wall plane channel), run with `run_channel!`; its averaged
profiles go to `output/results.toml`. Configs:
`examples/validation/configs/neutral_abl.toml`, `log_layer_mismatch.toml`.
"""
struct ChannelDriver <: AbstractRunDriver end

Config.driver_name(::ChannelDriver) = "channel"
Config.config_dir(::ChannelDriver) = VALIDATION_CONFIG_DIR

function Config.driver_sections(::ChannelDriver)
    ch = section("channel"; doc = "The wall-bounded case (Aeolus.WallModels).") do s
        option!(s, "kind", Symbol; default = :half, allowed = [:half, :full],
                doc = "half: half_channel_case (2πH × 2πH × H, free-slip top, rough log-law wall, inviscid, -∂p/∂x = u*²/H); " *
                      "full: channel_case (2πδ × πδ × 2δ, smooth walls, ν = u_τ δ/Re_τ)")
        option!(s, "size", NTuple{3,Int}; default = (24, 24, 24), range = (1, 4096),
                doc = "cells in x, y, z")
        option!(s, "re_tau", Float64; units = "1", default = 1000.0, range = (0, Inf), open_lower = true,
                doc = "friction Reynolds number u_τ δ/ν (full)")
        option!(s, "z0_over_h", Float64; units = "1", default = 1e-4, range = (0, Inf), open_lower = true,
                doc = "roughness length over the depth H (half)")
        option!(s, "walls", Symbol; default = :both, allowed = [:both, :bottom],
                doc = "modelled walls (full): both, or bottom with a free-slip top")
        option!(s, "noise", Float64; units = "u*", default = 3.0, range = (0, Inf),
                doc = "initial noise amplitude (in u*), seeded with [run] seed")
        option!(s, "stepper", Symbol; default = :rk3, allowed = [:rk3, :euler],
                doc = "rk3 (step_rk3!) or euler (step!)")
    end
    wm = section("wall_model"; doc = "The wall stress and the input it is fed.") do s
        option!(s, "law", Symbol; default = :auto, allowed = [:auto, :spalding, :rough_log],
                doc = "auto: the kind's law (half: RoughLogLaw(z0), full: SpaldingLaw()); spalding; rough_log (RoughLogLaw(z0_over_h H))")
        option!(s, "h_wm", Optional{Float64}; units = "H or δ", range = (0, Inf), open_lower = true,
                doc = "matching height (in H or δ); \"none\": h_wm_cells cells")
        option!(s, "h_wm_cells", Int; default = 1, range = (1, 64),
                doc = "matching height in cells, when h_wm is \"none\"")
        option!(s, "filter", Symbol; default = :none, allowed = [:none, :temporal, :spatial, :both],
                doc = "input filter: temporal (Yang et al. 2017), spatial (the 2Δ filter of P57-b), both")
        option!(s, "t_f", Optional{Float64}; units = "δ/u_τ", range = (0, Inf), open_lower = true,
                doc = "time-filter scale (in δ/u_τ); \"none\": 0.5")
    end
    sgs = section("sgs"; doc = "The subgrid closure (no wall damping).") do s
        option!(s, "closure", Symbol; default = :smagorinsky,
                allowed = [:smagorinsky, :dynamic_smagorinsky, :lasd, :lasi],
                doc = "smagorinsky (StaticSmagorinsky(c_s)), dynamic_smagorinsky (plane-averaged Germano-Lilly), " *
                      "lasd / lasi (LagrangianDynamicSmagorinsky, scale dependent / invariant)")
        option!(s, "c_s", Float64; units = "1", default = 0.1, range = (0, Inf), open_lower = true,
                doc = "the Smagorinsky coefficient (smagorinsky)")
    end
    st = section("stepper"; doc = "The run (run_channel!).") do s
        option!(s, "end_time", Float64; units = "H/u* or δ/u_τ", default = 20.0, range = (0, Inf), open_lower = true,
                doc = "simulated time, in H/u* (half) or δ/u_τ (full)")
        option!(s, "average_from_fraction", Float64; units = "1", default = 0.5, range = (0, 1),
                doc = "statistics from this fraction of end_time on")
        option!(s, "cfl", Float64; units = "1", default = 0.5, range = (0, Inf), open_lower = true,
                doc = "advective Courant number of the adaptive step (channel_dt)")
        option!(s, "dt", Optional{Float64}; units = "H/u* or δ/u_τ", range = (0, Inf), open_lower = true,
                doc = "a fixed step instead of the adaptive one")
    end
    return Section[ch, wm, sgs, st]
end

function _channel_checks(cfg)
    e = ConfigError[]
    c = cfg.channel; w = cfg.wall_model
    if c.kind === :half
        c.walls === :both || push!(e, ConfigError("channel.walls", "the half channel has one (bottom) wall: walls applies to kind = \"full\""))
        w.law === :spalding && push!(e, ConfigError("wall_model.law", "the half channel is inviscid: no smooth-wall (spalding) law"))
    end
    (w.t_f !== nothing && !(w.filter in (:temporal, :both))) &&
        push!(e, ConfigError("wall_model.t_f", "t_f is the scale of the temporal filter: set wall_model.filter = \"temporal\" or \"both\""))
    return e
end
Config.driver_checks(::ChannelDriver) = Any[
    """the half channel (`channel.kind = "half"`) has one wall (`channel.walls = "both"`) and no smooth-wall `spalding` law
    `wall_model.t_f` needs the temporal filter (`wall_model.filter = "temporal"` or `"both"`)""" => _channel_checks]

"""
    ChannelRun

What the `channel` driver builds: `case` (a `WallBoundedCase`), `stats`
(the averages `run_channel!` returns, after the run) and `cfg`.
"""
mutable struct ChannelRun
    case::Any
    stats::Any
    cfg::RunConfig
end

function _channel_closure(cfg)
    s = cfg.sgs; c = cfg.channel
    s.closure === :smagorinsky && return StaticSmagorinsky(s.c_s)
    ext = c.kind === :half ? (2π, 2π, 1.0) : (2π, π, 2.0)
    g = RegularGrid(; size = c.size, extent = ext)
    s.closure === :dynamic_smagorinsky && return DynamicSmagorinsky(g)
    return LagrangianDynamicSmagorinsky(g; scale_dependent = s.closure === :lasd)
end

function _channel_law(cfg)
    w = cfg.wall_model; c = cfg.channel
    w.law === :spalding && return SpaldingLaw()
    w.law === :rough_log && return RoughLogLaw(c.z0_over_h)
    return c.kind === :half ? nothing : SpaldingLaw()
end

function Config.build(::ChannelDriver, cfg::RunConfig)
    c = cfg.channel; w = cfg.wall_model
    kw = (; N = c.size, closure = _channel_closure(cfg), h_wm = w.h_wm, h_wm_cells = w.h_wm_cells,
          filter = w.filter, T_f = w.t_f, noise = c.noise, seed = cfg.run.seed, stepper = c.stepper)
    case = c.kind === :half ?
        half_channel_case(; kw..., wall_law = _channel_law(cfg), z0_over_H = c.z0_over_h) :
        channel_case(; kw..., Re_τ = c.re_tau, wall_law = _channel_law(cfg), walls = c.walls)
    return ChannelRun(case, nothing, cfg)
end

function Simulations.run!(::ChannelDriver, r::ChannelRun, rf::RunFolder, ck::CheckpointSchedule;
                          callback = nothing)
    st = r.cfg.stepper
    cb = callback === nothing ? nothing : (case -> callback(r, case.nsteps))
    r.stats = run_channel!(r.case, st.end_time; dt = st.dt, cfl = st.cfl,
                           average_from = st.average_from_fraction * st.end_time,
                           callback = cb, checkpoint = ck)
    s = r.stats
    if s.nsamples > 0
        @info("channel statistics", subsystem = :run, nsteps = s.nsteps, nsamples = s.nsamples,
              τ_wall = s.τ_wall)
        write_results(rf, Dict("nsteps" => s.nsteps, "nsamples" => s.nsamples, "tau_wall" => s.τ_wall,
                               "tau_wall_bottom" => s.τ_wall_bottom, "tau_wall_top" => s.τ_wall_top,
                               "zc" => s.zc, "U" => s.U, "zf" => s.zf, "phi_m" => s.phi_m,
                               "wall_time" => s.wall_time))
    end
    return r
end

function Config.state_fields(r::ChannelRun)
    s = r.case.sim
    return (u = s.u.data, v = s.v.data, w = s.w.data, p = s.p.data)
end

register_driver!(ChannelDriver())
