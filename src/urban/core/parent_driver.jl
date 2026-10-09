# The `urban_parent` run driver (EX-a, #438): the nest parent of an urban
# case run on its own, configured by the same schema as the `urban` driver
# (the parent is `[physics.options.nesting.parent]` of the case's config):
#
#   [run]
#   driver = "urban_parent"
#
# It runs the parent for `hours` and writes its record tables and its
# results (the air-heat ledger by window, z_i, the dt history) into the
# run folder's `output/`, where analysis scripts read them. It replaces the
# Berlin example's parent scripts, which built and ran the parent
# themselves (berlin_parent_diurnal.jl, the parent part of berlin_checks.jl).

"""
    UrbanParentDriver()

The `urban_parent` run driver ([`AbstractRunDriver`](@ref Aeolus.Config.AbstractRunDriver)):
the nest parent of an urban case alone ([`build_parent`](@ref) of
[`nest_parent_config`](@ref)), run for its `hours` with
[`run_parent!`](@ref). Same schema as the `urban` driver (a config may
`extends` an urban case and change only `[run] driver`), with
`[checkpoint]` off (the parent runs to the end in one call). Writes, into
the run folder's `output/` ([`write_parent_tables`](@ref),
[`parent_results`](@ref)):

* `parent.csv`: one row per output (`output_period`): time, local hour,
  z_i (minimum flux and parcel), the cumulative surface fluxes and the
  cumulative terms of the column's air-heat ledger (J m⁻²);
* `parent_theta.csv`: the plane-mean θ profile of every output;
* `parent_surface.nc`: the tiled surface ([`write_parent_surface`](@ref));
* `results.toml`: the grid, steps, dt history, retries, wall time, the
  air-heat budget of the whole run and of the night (20–05 LT) and day
  (10–16 LT) windows (W m⁻²), its closure, and the failure if the run
  stopped (the tables are written up to that point, the run is then
  marked failed).
"""
struct UrbanParentDriver <: AbstractRunDriver end

Config.driver_name(::UrbanParentDriver) = "urban_parent"
Config.driver_sections(::UrbanParentDriver) = Config.driver_sections(UrbanDriver())
Config.driver_defaults(::UrbanParentDriver) = ("output.cadence" => 1800.0, "run.name" => "urban_parent")
Config.driver_checks(::UrbanParentDriver) =
    Any[_URBAN_CHECKS_DOC => _urban_checks,
        "[checkpoint] is off: the parent alone runs to the end in one call" => Config.no_checkpoints]

"""
    UrbanParentRun

What the `urban_parent` driver builds: the `cfg` (`RunConfig`) and the
built [`ParentCase`](@ref) (`parent`).
"""
mutable struct UrbanParentRun
    cfg::RunConfig
    parent::Any
end

function Config.build(::UrbanParentDriver, cfg::RunConfig)
    cc = CaseConfig(cfg)
    f = cc.parent_static
    if f !== nothing && !isfile(f)
        src = first(config_sources(cfg)["files"])
        throw(ConfigError("geometry.parent_static", "the static driver $f does not exist; make it with AeolusPrep.jl:\n" *
                                                    "  julia --project -m AeolusPrep static $src --source <file.osm>"))
    end
    return UrbanParentRun(cfg, build_parent(nest_parent_config(cc)))
end

"The air-heat ledger terms of the parent's tables, in their column order."
const PARENT_TABLE_TERMS = [(:surface_flux, :tiles), (:column_tendency, :column_radiation),
                            (:dynamics, :subsidence), (:dynamics, :nudging), (:dynamics, :damping),
                            (:dynamics, :top_flux), (:dynamics, :transport), (:dynamics, :redistribution)]

"""
    write_parent_tables(dir, p; failure = nothing) -> (records, profiles)

Write `parent.csv` (one row per record of `p.diag`: t, local hour, z_i
by minimum flux and by parcel, the cumulative H and LE (J m⁻²), the
downwelling short- and longwave (W m⁻²), the air-heat storage, the
[`PARENT_TABLE_TERMS`](@ref) and the closure, cumulative J m⁻² of the
column) and `parent_theta.csv` (the plane-mean θ at the level centres)
into `dir`; `failure` (a message) is noted in the header.
"""
function write_parent_tables(dir::AbstractString, p; failure = nothing)
    a = joinpath(dir, "parent.csv")
    open(a, "w") do io
        println(io, "# urban nest parent alone (urban_parent driver): one row per output; ledger terms cumulative J/m2 (column)",
                failure === nothing ? "" : "; FAILED at $(round(p.sim.t / 3600; digits = 2)) h: $(first(split(failure, '\n')))")
        println(io, "t_s,hour_LT,zi_min_flux_m,zi_parcel_m,H_cum_J_m2,LE_cum_J_m2,sw_down,lw_down,storage_J_m2,",
                join(("$(x)_$(y)" for (x, y) in PARENT_TABLE_TERMS), ","), ",closure_J_m2")
        for r in p.diag
            @printf(io, "%.0f,%.3f,%.1f,%.1f,%.6e,%.6e,%.2f,%.2f,%.6e", r.t, r.hour, r.zi, r.zi_parcel,
                    r.sfc[:H], r.sfc[:LE], r.sw_down, r.lw_down, r.heat.storage)
            for k in PARENT_TABLE_TERMS
                @printf(io, ",%.6e", get(r.heat.terms, k, 0.0))
            end
            @printf(io, ",%.3e\n", r.heat.closure)
        end
    end
    b = joinpath(dir, "parent_theta.csv")
    open(b, "w") do io
        println(io, "hour_LT,", join((@sprintf("z%.1f", z) for z in level_centres(p.mesh)), ","))
        for r in p.diag
            println(io, @sprintf("%.3f", r.hour), ",", join((@sprintf("%.4f", x) for x in r.θ), ","))
        end
    end
    return (a, b)
end

"""
    write_parent_surface(path, ps::ParentSurface) -> path

Write the parent's tiled surface (#249) to the NetCDF file `path`: the
class fractions (`fraction`, x × y × class; 1 street, 2 roof, 3 park,
4 water, 5 wall), the morphometry window's plan-area fraction `lambda_p`,
the frontal-area fraction `lambda_f`, the building height `h`, the
Macdonald et al. (1998) roughness length `z0` and displacement `d`, and
the child's footprint (`footprint`, 1 where the parent's fractions are
the child's own class map), the land-use table's provenance lines
(`landuse_notes`). What examples/urban/plot_parent_surface.jl draws.
"""
function write_parent_surface(path::AbstractString, ps)
    nx, ny, nc = size(ps.fractions)
    dx = ps.landuse.dx
    ncf = nc_create(path; attributes = Dict("title" => "Aeolus.jl urban nest parent: tiled surface",
                                            "references" => "src/urban/parent/surface.jl",
                                            "landuse_notes" => join(ps.landuse.header, "\n")))
    try
        ncf.dim["x"] = nx; ncf.dim["y"] = ny; ncf.dim["class"] = nc
        nc_defvar!(ncf, "x", Float64, ("x",); attributes = Dict("units" => "m", "long_name" => "cell centre, east"))
        nc_defvar!(ncf, "y", Float64, ("y",); attributes = Dict("units" => "m", "long_name" => "cell centre, north"))
        nc_defvar!(ncf, "fraction", Float64, ("x", "y", "class");
                   attributes = Dict("units" => "1", "long_name" => "class fraction (1 street 2 roof 3 park 4 water 5 wall)"))
        for (v, u, ln) in (("lambda_p", "1", "plan-area fraction of buildings, morphometry window"),
                           ("lambda_f", "1", "frontal-area fraction of buildings"),
                           ("h", "m", "building height"), ("z0", "m", "roughness length, Macdonald et al. (1998)"),
                           ("d", "m", "displacement height, Macdonald et al. (1998)"),
                           ("footprint", "1", "1: under the child's footprint"))
            nc_defvar!(ncf, v, Float64, ("x", "y"); attributes = Dict("units" => u, "long_name" => ln))
        end
        ncf["x"][:] = [ps.landuse.x0 + (i - 0.5) * dx for i in 1:nx]
        ncf["y"][:] = [ps.landuse.y0 + (j - 0.5) * dx for j in 1:ny]
        ncf["fraction"][:, :, :] = ps.fractions
        ncf["lambda_p"][:, :] = ps.λp; ncf["lambda_f"][:, :] = ps.λf; ncf["h"][:, :] = ps.h
        ncf["z0"][:, :] = ps.z0; ncf["d"][:, :] = ps.d
        fp = zeros(nx, ny)
        ps.footprint === nothing || (fp[ps.footprint...] .= 1.0)
        ncf["footprint"][:, :] = fp
    finally
        close(ncf)
    end
    return path
end

"A heat budget as W m⁻² per term (`\"stage.name\"`), the storage and the closure."
function _heat_wm2(hb)
    e = hb.elapsed
    d = Dict{String,Any}("$(k[1]).$(k[2])" => (e > 0 ? v / e : 0.0) for (k, v) in hb.terms)
    d["storage"] = e > 0 ? hb.storage / e : 0.0
    d["closure"] = e > 0 ? hb.closure / e : 0.0
    d["elapsed_s"] = e
    return d
end

"""
    parent_results(p; wall = NaN, failure = nothing) -> Dict{String,Any}

The results of a parent run (what the `urban_parent` driver writes to
`results.toml`): `grid` (Nx, Ny, Nz), `steps`, `retries`, the dt
history's minimum, maximum and mean (s), the wall seconds, `finite`
(θ, and qv with moisture), the air-heat budget of the column
([`heat_budget`](@ref)) over the whole run (`[heat]`), the night
(`[heat_night]`, 20–05 LT) and the day (`[heat_day]`, 10–16 LT), each term
in W m⁻² (the night nudging against the daytime surface flux of Heinze et
al. 2017, Atmos. Chem. Phys. 17, 7083-7109), the largest |closure| of the
whole-run budget relative to its largest term (`closure_rel`), the
largest parcel z_i of the records, the wall seconds per stage
(`wall_s_<stage>`), and `failure` when the run stopped.
"""
function parent_results(p; wall = NaN, failure = nothing)
    m, sim = p.mesh, p.sim
    dts = p.dt_hist
    hb = heat_budget(p.heat)
    big = maximum(abs, values(hb.terms); init = 0.0)
    out = Dict{String,Any}(
        "grid" => [m.Nx, m.Ny, m.Nz], "steps" => p.nsteps, "retries" => p.retries,
        "dt_min" => isempty(dts) ? NaN : minimum(dts), "dt_max" => isempty(dts) ? NaN : maximum(dts),
        "dt_mean" => isempty(dts) ? NaN : sum(dts) / length(dts), "wall_s" => wall,
        "simulated_h" => sim.t / 3600,
        "finite" => all(isfinite, interior_view(sim.θ)) && (!p.pc.moisture || all(isfinite, interior_view(sim.qv))),
        "heat" => _heat_wm2(hb), "heat_night" => _heat_wm2(heat_budget(p.heat; window = (20, 5))),
        "heat_day" => _heat_wm2(heat_budget(p.heat; window = (10, 16))),
        "closure_rel" => big > 0 ? abs(hb.closure) / big : 0.0,
        "zi_parcel_max" => isempty(p.diag) ? NaN : maximum(r.zi_parcel for r in p.diag))
    for (k, v) in p.timing
        out["wall_s_$k"] = v           # wall seconds per stage ("wall" in the key: not compared by replays)
    end
    failure === nothing || (out["failure"] = failure)
    return out
end

function Aeolus.Simulations.run!(::UrbanParentDriver, r::UrbanParentRun, rf::RunFolder, ck::CheckpointSchedule;
                                 callback = nothing)
    callback === nothing || throw(ArgumentError("the urban_parent driver takes no step callback"))
    p = r.parent
    dir = output_dir(rf)
    p.surface === nothing || write_parent_surface(joinpath(dir, "parent_surface.nc"), p.surface)
    t0 = time()
    finish(failure) = begin
        write_parent_tables(dir, p; failure = failure)
        Config.write_results(rf, parent_results(p; wall = time() - t0, failure = failure))
        @info("parent air-heat ledger\n" * heat_report(p.heat; layers = (:column, (0.0, 100.0), (0.0, 1000.0))),
              subsystem = :urban, domain = current_domain(), event = :parent_ledger)
    end
    try
        run_parent!(p)
    catch err
        err isa InterruptException || finish(sprint(showerror, err))
        rethrow()
    end
    finish(nothing)
    return r
end

"""
    state_fields(r::UrbanParentRun) -> NamedTuple

The parent's prognostic state, interior cells: `u, v, w, p, θ` (and
`qv` with moisture).
"""
function Config.state_fields(r::UrbanParentRun)
    s = r.parent.sim
    out = Pair{Symbol,Array{Float64}}[f => Array{Float64}(interior_view(getproperty(s, f))) for f in (:u, :v, :w, :p, :θ)]
    r.parent.pc.moisture && push!(out, :qv => Array{Float64}(interior_view(s.qv)))
    return (; out...)
end

register_driver!(UrbanParentDriver())
