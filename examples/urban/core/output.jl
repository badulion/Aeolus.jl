# NetCDF output driven by the registry.
#
# Full 3-D model state in NetCDF-3, written with the dependency-free
# writer in `src/NetCDFIO.jl` — the format every tool in the field reads
# (ncview, xarray, Paraview, CDO, Panoply) without Aeolus installed.
# Float32, the usual precision for model output: 64x64x32 is 512 kB per
# 3-D field and record.
#
# The variables are: the core ones (coordinates, time, hour, solar
# elevation, static geometry, u v w θ p), every tracer registered with
# `output = true`, the bus fields Ts / sw_in / lw_in, and whatever the
# enabled couplings return from `output_fields`. The renderers
# (`alexanderplatz_animation.jl`, `alexanderplatz_surface_diagnostics.jl`)
# read the file by variable name.

"""
    OutputField(name, units, long_name, dims, getter)

One output variable a coupling contributes: `dims` is `:xyz` (3-D, per
record), `:xy` (2-D, per record) or `:z` (a vertical profile, per record);
`getter(case)` returns the interior-sized array to write.
"""
struct OutputField
    name::String
    units::String
    long_name::String
    dims::Symbol
    getter::Function
end

"Open NetCDF state of a case."
mutable struct CaseOutput
    ncf::Any
    fields::Vector{OutputField}
end

function core_output_fields(case)
    iv(f) = interior_view(f)
    fs = OutputField[
        OutputField("u",     "m s-1",  "eastward wind",           :xyz, c -> iv(c.sim.u)),
        OutputField("v",     "m s-1",  "northward wind",          :xyz, c -> iv(c.sim.v)),
        OutputField("w",     "m s-1",  "upward wind",             :xyz, c -> iv(c.sim.w)),
        OutputField("theta", "K",      "potential temperature",   :xyz, c -> iv(c.sim.θ)),
        OutputField("p",     "m2 s-2", "kinematic pressure",      :xyz, c -> iv(c.sim.p))]
    for t in case.tracers
        t.output || continue
        f = t.field
        push!(fs, OutputField(string(t.name), t.units, t.long_name, :xyz, c -> iv(f)))
    end
    append!(fs, OutputField[
        OutputField("Ts",    "K",     "surface temperature",                 :xy, c -> c.bus.Ts),
        OutputField("sw_in", "W m-2", "shortwave irradiance at the surface", :xy, c -> c.bus.sw),
        OutputField("lw_in", "W m-2", "downwelling longwave at the surface", :xy, c -> c.bus.lw)])
    return fs
end

"""
    open_output!(case)

Create `cfg.output_path`, define the core, tracer, bus and coupling
variables, and write the coordinates and the static geometry.
"""
function open_output!(case)
    cfg, mesh, geom = case.cfg, case.mesh, case.geom
    Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
    mkpath(dirname(cfg.output_path))
    ncf = nc_create(cfg.output_path; attributes = Dict(
        "title"       => "Aeolus.jl LES — Alexanderplatz, Berlin",
        "institution" => "Aeolus.jl",
        "source"      => "Boussinesq LES, cut-cell IBM, Era III radiation",
        "comment"     => @sprintf("Real OSM geometry; scenario %s from %05.2f local on %d-%02d-%02d",
                                  cfg.scenario, cfg.start_h, cfg.date...),
        "couplings"   => join(string.(coupling_name.(case.couplings)), " "),
        "references"  => "examples/urban/BerlinCase.jl"))

    nc_defdim!(ncf, "x", Nx); nc_defdim!(ncf, "y", Ny); nc_defdim!(ncf, "z", Nz)
    nc_defdim!(ncf, "time", nothing)          # unlimited record dimension
    for (d, L) in (("x", mesh.Lx), ("y", mesh.Ly), ("z", mesh.Lz))
        nc_defvar!(ncf, d, Float32, (d,);
                   attributes = Dict("units" => "m", "axis" => uppercase(d),
                                     "long_name" => "$d cell centre"))
    end
    nc_defvar!(ncf, "time", Float64, ("time",);
               attributes = Dict("units" => "seconds since simulation start",
                                 "axis" => "T", "long_name" => "simulation time"))
    nc_defvar!(ncf, "hour", Float64, ("time",);
               attributes = Dict("units" => "h", "long_name" => "local hour of day"))
    nc_defvar!(ncf, "solar_elevation", Float64, ("time",);
               attributes = Dict("units" => "degree", "long_name" => "solar elevation angle"))
    nc_defvar!(ncf, "sky_view_factor", Float32, ("x", "y");
               attributes = Dict("units" => "1",
                   "long_name" => "sky view factor, area-weighted per column"))
    nc_defvar!(ncf, "building_height", Float32, ("x", "y");
               attributes = Dict("units" => "m", "long_name" => "building height"))
    nc_defvar!(ncf, "surface_class", Float32, ("x", "y");
               attributes = Dict("units" => "1",
                   "long_name" => "surface class (1 street 2 roof 3 park 4 water 5 wall 6 glass)"))

    fields = core_output_fields(case)
    for c in case.couplings
        append!(fields, output_fields(c, case))
    end
    names = [f.name for f in fields]
    allunique(names) || error("duplicate output variable names: $(names)")
    for f in fields
        dims = f.dims === :xyz ? ("x", "y", "z", "time") :
               f.dims === :xy  ? ("x", "y", "time") :
               f.dims === :z   ? ("z", "time") :
               error("output field $(f.name): dims must be :xyz, :xy or :z")
        nc_defvar!(ncf, f.name, Float32, dims;
                   attributes = Dict("units" => f.units, "long_name" => f.long_name,
                                     "coordinates" => f.dims === :xyz ? "x y z" :
                                                      f.dims === :z ? "z" : "x y"))
    end
    nc_endef!(ncf)
    nc_write!(ncf, "x", Float32[(i - 0.5) * mesh.dx for i in 1:Nx])
    nc_write!(ncf, "y", Float32[(j - 0.5) * mesh.dy for j in 1:Ny])
    nc_write!(ncf, "z", Float32[(k - 0.5) * mesh.dz for k in 1:Nz])
    nc_write!(ncf, "sky_view_factor", Float32.(geom.svf))
    nc_write!(ncf, "building_height", Float32.(building_height(geom, mesh)))
    nc_write!(ncf, "surface_class", Float32.(geom.sfc))
    case.output = CaseOutput(ncf, fields)
    return case
end

"Close the NetCDF file if one is open."
function close_output!(case)
    case.output === nothing && return case
    nc_close!(case.output.ncf)
    case.output = nothing
    return case
end

"The frame line of the current output record and every coupling's `report`."
function frame_message(case)
    clk, sim, bus = case.clock, case.sim, case.bus
    io = IOBuffer()
    θ1 = view(interior_view(sim.θ), :, :, 1)
    qv1 = view(interior_view(tracer(case, :qv)), :, :, 1)
    ql1 = view(interior_view(tracer(case, :ql)), :, :, 1)
    @printf(io, "frame %2d @ t=%5.1fh  wall=%5.0fs  Ts=%5.1f/%5.1f  SW=%4.0f/%4.0f W/m²  elev=%5.1f°  θ1=%5.1f/%5.1f  qv1min=%.4f  dew=%.3f mm  fog ql1max=%.2e",
            clk.nrecord, clk.hour, time() - clk.t_wall0,
            minimum(bus.Ts), maximum(bus.Ts),
            minimum(bus.sw), maximum(bus.sw), clk.sun.elevation,
            minimum(θ1), maximum(θ1), minimum(qv1), maximum(bus.dew), maximum(ql1))
    for c in case.couplings
        r = report(c, case)
        r === nothing || print(io, '\n', r)
    end
    return String(take!(io))
end

"""
    write_output!(case)

Append one record (when a file is open) and log the frame: one `Info`
record (`subsystem = :berlin`, `event = :frame`) whose message is the core
frame line followed by every coupling's `report` ([`frame_message`](@ref)).
"""
function write_output!(case)
    clk, sim = case.clock, case.sim
    clk.nrecord += 1
    if case.output !== nothing
        rec = Dict{String,Any}(
            "time"            => [Float64(sim.t)],
            "hour"            => [Float64(clk.hour)],
            "solar_elevation" => [Float64(clk.sun.elevation)])
        for f in case.output.fields
            rec[f.name] = Float32.(f.getter(case))
        end
        nc_append_record!(case.output.ncf, rec)
    end
    # the frame line and the couplings' reports: one Info record, built only
    # when a logger wants it (#347; the reports move onto the diagnostics
    # registry with #338)
    @info(frame_message(case), subsystem = :berlin, domain = current_domain(),
          event = :frame, frame = clk.nrecord, hour = clk.hour)
    return case
end
