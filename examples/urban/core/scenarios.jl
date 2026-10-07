# Named scenarios of the Berlin case: the shipped TOML files of
# examples/urban/configs/ (CFG-c, #358). A scenario only sets
# configuration values; the physics defaults live with the couplings
# (the `[physics]` defaults of the schema come from the registry).

"""
    SCENARIO_FILES :: Dict{Symbol,String}

The shipped scenario files of the Berlin case, in
`examples/urban/configs/` ([`CaseConfig`](@ref)`(scenario)` loads them):

* `:summer` (alias `:alexanderplatz`), `berlin_summer.toml` — the
  flagship, with every option written out: 1 km × 1 km × 400 m around
  Alexanderplatz on 64 × 64 × 64 cells, nested in the 6 km × 6 km × 3 km
  Aeolus parent, 29 June 2019 from 06:00 CEST, 24 h at dt = 2 s, from the
  Lindenberg sounding;
* `:winter`, `berlin_winter.toml` — extends the summer file: 15 January
  2024 from 06:00 CET, an idealised cold profile (no winter sounding
  yet), cold surfaces and soil; the surface physics is the summer set;
* `:smoke`, `berlin_smoke.toml` — extends the summer file: a 250 m ×
  250 m × 100 m window on 16 × 16 × 16 cells at local noon in a small
  parent (1 km × 1 km × 500 m, refinement (4, 4, 8)), 12 steps, no
  NetCDF; the run of the smoke test;
* `:mitte_sample`, `berlin_mitte_sample.toml` — extends the summer
  file: the synthetic OSM sample (`data/sample_berlin_mitte.osm`, not
  real buildings), buildings of 15 m and more.

Each file's header says more.
"""
const SCENARIO_FILES = Dict{Symbol,String}(
    :summer => "berlin_summer.toml", :alexanderplatz => "berlin_summer.toml",
    :winter => "berlin_winter.toml", :smoke => "berlin_smoke.toml",
    :mitte_sample => "berlin_mitte_sample.toml")

"The path of the shipped file of `scenario`."
function scenario_file(scenario::Symbol)
    haskey(SCENARIO_FILES, scenario) ||
        error("unknown scenario :$scenario (known: $(join(sort!(string.(collect(keys(SCENARIO_FILES)))), ", ")))")
    return joinpath(CONFIG_DIR, SCENARIO_FILES[scenario])
end

function CaseConfig(scenario::Symbol; kwargs...)
    # the alias keeps its own name (the run folder's and the output's label)
    ovr = scenario === :alexanderplatz ? ["run.name=\"alexanderplatz\""] : String[]
    c = CaseConfig(load_run_config(scenario_file(scenario); overrides = ovr, schema = berlin_schema()))
    for (k, v) in kwargs
        hasfield(CaseConfig, k) || throw(ArgumentError("CaseConfig has no field $k"))
        if k === :options
            # per coupling over the file's options (the smoke file's nest
            # parent survives a caller's options for another coupling)
            o = copy(c.options)
            for (name, nt) in v
                o[name] = haskey(o, name) ? merge(o[name], nt) : nt
            end
            c.options = o
        elseif k === :physics
            c.physics = merge(c.physics, Dict{Symbol,Any}(v))
        else
            setfield!(c, k, convert(fieldtype(CaseConfig, k), v))
        end
    end
    return c
end
