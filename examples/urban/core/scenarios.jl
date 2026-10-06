# Named scenario presets for CaseConfig.
#
# A scenario only sets CaseConfig fields; it does not select physics. The
# physics defaults live with the couplings, so a scenario plus the
# registry defaults is the whole case.

"""
    SCENARIOS :: Dict{Symbol, NamedTuple}

Named presets for [`CaseConfig`](@ref):

* `:summer` (alias `:alexanderplatz`) — the flagship: 1 km × 1 km × 400 m
  around Alexanderplatz on the 64 × 64 × 64 grid (dx = 15.6 m, dz =
  6.25 m), nested in the 6 km × 6 km × 3 km Aeolus parent of #259
  (refinement (2, 2, 4); `options[:nesting]`, #265), 29 June
  2019 (a clear day of the late-June 2019 heat wave; #247) from 06:00
  CEST, 24 h at dt = 2 s, initialised from the Lindenberg sounding
  (`cfg.sounding`).
* `:winter` — the same domain on 15 January 2024 from 06:00 CET, with
  cold initial air, surfaces and soil, from an explicitly **idealised**
  profile (`sounding = nothing`: no winter sounding has been added yet).
  Initial conditions only: the surface
  physics is the summer set (the snowpack mosaic, river ice and snow
  optics arrive with the winter-surface issue) — so this is not yet a
  physical winter run.
* `:smoke` — a 250 m × 250 m × 100 m window of the real OSM file (centred
  312.5 m south of the anchor: 69 building and 50 park columns, tallest
  roof 31 m, well clear of the towers that reach 200 m in the OSM
  heights) on a
  16 × 16 × 16 grid (dx = 15.6 m) at local noon, nested in a small
  parent (1 km × 1 km × 500 m on 16 × 16 × 10 cells of 62.5 m × 50 m,
  refinement (4, 4, 8), the land use a crop of the 6 km table; #265),
  `options[:nesting]`), 12 steps of 2 s with
  radiation every 8 s, the surfaces (soil included) every 4 s and output
  every 12 s. Used for
  quick checks of every code path, not for physics. **Idealised** initial
  profile (`sounding = nothing`, capping inversion at 60 m): the smoke
  checks of the turbulence diagnostics and the AMD/Vreman/σ switch-off
  need a stable layer above an inversion inside the 100 m box, which the
  observed noon profile (mixed to 1.2 km) does not have. The sounding
  path itself runs in the smoke test's "summer sounding" testset.
* `:mitte_sample` — the bundled synthetic OSM sample
  (`data/sample_berlin_mitte.osm`, hand-generated, *not* real
  buildings) with buildings of 15 m and more.
"""
const SCENARIOS = Dict{Symbol,NamedTuple}(
    :summer => (;),
    :winter => (date = (2024, 1, 15), utc_offset = 1.0, sounding = nothing,
                θ_surface = 271.0, θ₀ = 271.0,
                qv0 = 0.0025, Ts0 = 272.0, Ts_water_offset = 5.0, T_soil0 = 276.0),
    :smoke  => (size = (16, 16, 16), extent = (250.0, 250.0, 100.0),
                offset = (0.0, -312.5), start_h = 12.0, nsteps = 12,
                sounding = nothing, θ_inversion_z = 60.0,
                cadence = Dict(:radiation => 8.0, :soil => 4.0, :output => 12.0),
                output_path = nothing,
                # the small parent of the smoke nest (#265): 1 km × 1 km × 500 m,
                # 16 × 16 × 10 cells of 62.5 m × 50 m
                options = Dict{Symbol,Any}(:nesting => (parent = (extent = (1000.0, 1000.0, 500.0),
                                                                  refinement = (4, 4, 8),
                                                                  vertical = :uniform),))),
    :mitte_sample => (osm_path = joinpath(DATA_DIR, "sample_berlin_mitte.osm"),
                      min_height = 15.0),
)
SCENARIOS[:alexanderplatz] = SCENARIOS[:summer]

function CaseConfig(scenario::Symbol; kwargs...)
    haskey(SCENARIOS, scenario) ||
        error("unknown scenario :$scenario (known: $(join(sort!(collect(keys(SCENARIOS))), ", ")))")
    preset = SCENARIOS[scenario]
    # copy mutable preset values so two configs never share a Dict
    fields = Dict{Symbol,Any}(k => (v isa AbstractDict ? copy(v) : v) for (k, v) in pairs(preset))
    for (k, v) in kwargs
        if k === :options && haskey(fields, :options)
            # options merge over the preset's, per coupling and keyword
            # (the smoke preset's nest parent survives a user's options)
            o = copy(fields[:options])
            for (name, nt) in v
                o[name] = haskey(o, name) ? merge(o[name], nt) : nt
            end
            fields[k] = o
        else
            fields[k] = v
        end
    end
    haskey(fields, :scenario) || (fields[:scenario] = scenario === :alexanderplatz ? :summer : scenario)
    haskey(fields, :name) || (fields[:name] = scenario)
    return CaseConfig(; fields...)
end
