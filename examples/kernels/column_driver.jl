# The `column` run driver (CFG-e, #360): the Kessler box of the kernel
# demo (examples/kernels/configs/cloud_spinup.toml), Kessler microphysics
# in a small box with no dynamics (apply_kessler_step! with a sequence of
# steps), selected by [column] case = "kessler_box".
#
# The single-column validation cases this driver also ran (DYCOMS-II RF01,
# ParisFog 2011, the spectral-broadening parcel) went with the maintainer's
# decision of 2026-10-09 (ONE-g, #456: no single-column cases); the box
# stays a legacy case until the kernel demos become model configurations
# (ONE-i, #458). Loaded on first use (Aeolus.Config.DRIVER_FILES).

using Aeolus
using Aeolus.Config: Config, AbstractRunDriver, RunConfig, ConfigError, Section,
                     section, option!, register_driver!, write_results, no_checkpoints
using Aeolus.Fields: set!, interior_view
using Aeolus.IO: RunFolder, CheckpointSchedule
import Aeolus.Simulations

struct ColumnDriver <: AbstractRunDriver end

Config.driver_name(::ColumnDriver) = "column"

function Config.driver_sections(::ColumnDriver)
    col = section("column"; doc = "Which column case runs.") do s
        option!(s, "case", Symbol; default = :kessler_box, allowed = [:kessler_box],
                doc = "kessler_box ([kessler_box])")
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
    return Section[col, kb]
end

function _checks(cfg)
    e = no_checkpoints(cfg)
    kb = cfg.kessler_box
    kb.cloud_layer[1] < kb.cloud_layer[2] ||
        push!(e, ConfigError("kessler_box.cloud_layer", "expected [bottom, top] with bottom < top"))
    return e
end
Config.driver_checks(::ColumnDriver) = Any[
    """`[checkpoint]` stays off: a column case runs to the end in one call
    `kessler_box.cloud_layer` is [bottom, top] with bottom < top""" => _checks]

"""
    ColumnRun

What the `column` driver builds: `case` (`:kessler_box`), `state` (the
Kessler fields and the mesh), `snapshots` (the horizontal-mean profiles
after each step) and `cfg`.
"""
mutable struct ColumnRun
    case::Symbol
    state::Any
    snapshots::Vector{Any}
    cfg::RunConfig
end

function Config.build(::ColumnDriver, cfg::RunConfig)
    k = cfg.kessler_box; H = k.height
    mesh = RegularGrid(; size = k.size, extent = (H, H, H), halo = (1, 1, 1))
    qv = Field(CellCenter(), mesh); set!(qv, (x, y, z) -> k.qv)
    ql = Field(CellCenter(), mesh)
    z0, z1 = k.cloud_layer .* H
    set!(ql, (x, y, z) -> z0 < z < z1 ? k.ql : 0.0)
    qr = Field(CellCenter(), mesh); set!(qr, (x, y, z) -> 0.0)
    return ColumnRun(cfg.column.case, (; qv, ql, qr, mesh), Any[], cfg)
end

"The horizontal-mean profile of a cell-centred field."
function hmean_profile(f)
    v = interior_view(f)
    return [sum(view(v, :, :, k)) / (size(v, 1) * size(v, 2)) for k in 1:size(v, 3)]
end

function Simulations.run!(::ColumnDriver, r::ColumnRun, rf::RunFolder, ::CheckpointSchedule; callback = nothing)
    cfg = r.cfg; s = r.state
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
    callback === nothing || callback(r, nsteps)
    return r
end

Config.state_fields(r::ColumnRun) = (qv = r.state.qv.data, ql = r.state.ql.data, qr = r.state.qr.data)

register_driver!(ColumnDriver())
