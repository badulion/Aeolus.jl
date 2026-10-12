# Captures test/data/input_static/digests.toml (CFG-f, #361). Run ONCE on
# the commit before the Berlin geometry moved to the static driver (the
# OSM file still parsed by build_geometry at every start):
#
#   julia --project=. test/data/input_static/capture_digests.jl
#
# It records the fnv1a64 digests of what the OSM-direct path built:
#   * per config (smoke, mitte_sample on 32 × 32 × 16, the same periodic (no nest: no building-free frame, the wall tagging wraps)):
#     build_geometry's SolidMask, surface classes, heightmap, view factors
#     (F, F_sky, area) and sky-view factor;
#   * summer (64 × 64 × 64): the rasterisation only (mask, classes,
#     heightmap), the radiosity scene being too costly for a test;
#   * the nest parent's land use (cropped and coarsened to the parent grid)
#     for the smoke and summer parents;
#   * the :smoke case's state after 12 steps (FFTW "estimate").
# Kept for the record; it no longer runs after the switch (and its OSM
# reader, load_city_osm, is AeolusPrep.jl's since ONE-h, #457).

using Aeolus, TOML
const BC = Aeolus.UrbanCase
include(joinpath(@__DIR__, "..", "..", "support", "berlin_configs.jl"))
include(joinpath(@__DIR__, "..", "..", "support", "berlin_fixture.jl"))

out = Dict{String,Any}("commit" => readchomp(`git -C $(@__DIR__) rev-parse HEAD`))
meshof(cfg) = Aeolus.Grids.RegularGrid(; size = cfg.size, extent = cfg.extent, halo = cfg.halo)
function geomdig(g)
    Dict{String,Any}("mask" => digest(g.sm.mask), "sfc" => digest(g.sfc), "heightmap" => digest(g.heightmap),
                     "F" => digest(g.vfac.F), "F_sky" => digest(g.vfac.F_sky), "area" => digest(g.vfac.area),
                     "svf" => digest(g.svf), "buildings" => length(g.city.buildings),
                     "patches" => length(g.city.patches), "nfacets" => length(g.vfac.area))
end
for (label, cfg) in ("smoke" => site_config(berlin_toml("smoke")),
                     "mitte_periodic" => site_config(berlin_toml("mitte_sample"); size = (32, 32, 16), lateral = :periodic,
                                                       overrides = ["nesting.enabled=false"]),
                     "mitte_sample" => site_config(berlin_toml("mitte_sample"); size = (32, 32, 16)))
    out[label] = geomdig(BC.build_geometry(cfg, meshof(cfg)))
end
let cfg = site_config(berlin_toml("summer")), mesh = meshof(cfg)
    city = BC.load_city_osm(cfg.osm_path; lat0 = cfg.lat0, lon0 = cfg.lon0,
                            x_shift = mesh.Lx / 2 - cfg.offset[1], y_shift = mesh.Ly / 2 - cfg.offset[2],
                            min_height = cfg.min_height)
    w_f = BC.lateral_open(cfg) ? cfg.boundary_frame : 0
    sm, sfc = BC.surface_class_map(city, mesh; tag_walls = true, frame = (w_f, w_f), bcs = BC.case_scalar_bcs(cfg))
    hm = Aeolus.Radiosity.heightmap_from_solid_mask(sm)
    out["summer"] = Dict{String,Any}("mask" => digest(sm.mask), "sfc" => digest(sfc), "heightmap" => digest(hm),
                                     "buildings" => length(city.buildings), "patches" => length(city.patches))
end
for s in (:smoke, :summer)
    pc = BC.nest_parent_config(site_config(berlin_toml(s)))
    Nx = BC.parent_size(pc)[1]
    pl = BC.crop_landuse(BC.read_parent_landuse(pc.child.landuse), pc)
    pl = BC.coarsen_landuse(pl, pl.nx ÷ Nx)
    out["parent_" * string(s)] = Dict{String,Any}("counts" => digest(pl.counts), "lambda_f" => digest(pl.λf),
        "h_mean" => digest(pl.hmean), "h_max" => digest(pl.hmax), "nx" => pl.nx, "ny" => pl.ny,
        "dx" => pl.dx, "x0" => pl.x0, "y0" => pl.y0, "r" => pl.r)
end
Aeolus.Operators.set_fftw_policy!(; planner = :estimate)
case = BC.build_case(site_config(berlin_toml("smoke")))
out["smoke_case"] = Dict{String,Any}("mask" => digest(case.geom.sm.mask), "sfc" => digest(case.geom.sfc))
BC.run_case!(case)
out["smoke_state"] = Dict{String,Any}(string(k) => digest(v) for (k, v) in pairs(berlin_state(case)))
out["threads"] = Threads.nthreads()
open(joinpath(@__DIR__, "digests.toml"), "w") do io
    TOML.print(io, out; sorted = true)
end
println("wrote digests at ", Threads.nthreads(), " threads")
