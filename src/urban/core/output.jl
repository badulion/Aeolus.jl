# NetCDF output driven by the registry, written by the one writer of the
# model (`Aeolus.Output`, ONE-c #452).
#
# Full 3-D model state in NetCDF-4 through `src/NCIO.jl` (NCDatasets, #310)
# — the format every tool in the field reads (ncview, xarray, Paraview,
# CDO, Panoply) without Aeolus installed. Float32, the usual precision for
# model output (64x64x32 is 512 kB per 3-D field and record before
# compression), deflated and chunked by level and record (`nc_defvar!`).
#
# The variables are: the core ones (coordinates, time, hour, solar
# elevation, static geometry, u v w θ p), every tracer registered with
# `output = true`, the bus fields Ts / sw_in / lw_in, and whatever the
# enabled couplings return from `output_fields` (their catalogues of
# `OutputVariable`s). The renderers
# (`alexanderplatz_animation.jl`, `alexanderplatz_surface_diagnostics.jl`)
# read the file by variable name.

"Open NetCDF output of a case: the one writer of `Aeolus.Output` (ONE-c, #452)."
mutable struct CaseOutput
    writer::OutputWriter
end

function core_output_fields(case)
    iv(f) = interior_view(f)
    fs = OutputVariable[
        OutputVariable("hour", "h", "local hour of day", :scalar, c -> Float64(c.clock.hour)),
        OutputVariable("solar_elevation", "degree", "solar elevation angle", :scalar,
                       c -> Float64(c.clock.sun.elevation)),
        OutputVariable("u",     "m s-1",  "eastward wind",           :xyz, c -> iv(c.sim.u)),
        OutputVariable("v",     "m s-1",  "northward wind",          :xyz, c -> iv(c.sim.v)),
        OutputVariable("w",     "m s-1",  "upward wind",             :xyz, c -> iv(c.sim.w)),
        OutputVariable("theta", "K",      "potential temperature",   :xyz, c -> iv(c.sim.θ)),
        OutputVariable("p",     "m2 s-2", "kinematic pressure",      :xyz, c -> iv(c.sim.p))]
    for t in case.tracers
        t.output || continue
        f = t.field
        push!(fs, OutputVariable(string(t.name), t.units, t.long_name, :xyz, c -> iv(f)))
    end
    append!(fs, OutputVariable[
        OutputVariable("Ts",    "K",     "surface temperature",                 :xy, c -> c.bus.Ts),
        OutputVariable("sw_in", "W m-2", "shortwave irradiance at the surface", :xy, c -> c.bus.sw),
        OutputVariable("lw_in", "W m-2", "downwelling longwave at the surface", :xy, c -> c.bus.lw)])
    return fs
end

"The static geometry the output file carries (written once)."
static_output_fields(case) = OutputVariable[
    OutputVariable("sky_view_factor", "1", "sky view factor, area-weighted per column", :xy, c -> c.geom.svf),
    OutputVariable("building_height", "m", "building height", :xy, c -> building_height(c.geom, c.mesh)),
    OutputVariable("surface_class", "1", "surface class (1 street 2 roof 3 park 4 water 5 wall 6 glass)", :xy,
                   c -> c.geom.sfc)]

"""
    case_start_utc(cfg) -> DateTime

UTC time of the case start: `cfg.date` at local hour `cfg.start_h`, less
`cfg.utc_offset`. The reference time of the output's CF time axis.
"""
case_start_utc(cfg) = DateTime(cfg.date...) +
                      Millisecond(round(Int, (cfg.start_h - cfg.utc_offset) * 3.6e6))

"""
    open_output!(case; resume = false)

Create `cfg.output_path` with the one writer of `Aeolus.Output`: the
core, tracer, bus and coupling variables per record, the coordinates and
the static geometry. With `resume = true` and the file present (a run
resumed from a checkpoint, #269), reopen it instead: the records continue
at the checkpoint's `clock.nrecord + 1`, over any its interrupted session
wrote after it.
"""
function open_output!(case; resume::Bool = false)
    cfg, mesh = case.cfg, case.mesh
    x, y, z = cell_centres(mesh)
    attrs = Dict{String,Any}(
        "title"       => "Aeolus.jl urban LES — $(cfg.name)",
        "institution" => "Aeolus.jl",
        "source"      => "Boussinesq LES, cut-cell IBM, Era III radiation",
        "comment"     => @sprintf("scenario %s from %05.2f local on %d-%02d-%02d",
                                  cfg.scenario, cfg.start_h, cfg.date...),
        "couplings"   => join(string.(coupling_name.(case.couplings)), " "),
        "references"  => "src/UrbanCase.jl")
    w = OutputWriter(cfg.output_path; x = x, y = y, z = z, records = _output_fields(case),
                     statics = static_output_fields(case), ctx = case,
                     time_units = cf_time_units(case_start_utc(cfg)), attributes = attrs,
                     resume_at = resume ? case.clock.nrecord + 1 : nothing)
    case.output = CaseOutput(w)
    return case
end

# the record variables: the core's, then every coupling's
function _output_fields(case)
    fields = core_output_fields(case)
    for c in case.couplings
        append!(fields, output_fields(c, case))
    end
    return check_catalogue(fields)
end

"Close the NetCDF file if one is open."
function close_output!(case)
    case.output === nothing && return case
    Output.close_output!(case.output.writer)
    case.output = nothing
    return case
end

"""
    write_output!(case)

Append one record (when a file is open) and log the frame
([`log_frame`](@ref): the frame's values and each coupling's
diagnostics, as structured `Info` events).
"""
function write_output!(case)
    clk, sim = case.clock, case.sim
    clk.nrecord += 1
    case.output === nothing || write_record!(case.output.writer, case, sim.t; at = clk.nrecord)
    log_frame(case)
    return case
end
