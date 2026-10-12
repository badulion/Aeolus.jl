# The static driver (CFG-f, #361): Aeolus.InputDrivers — the conventions,
# the schema check, the rasterisation and the Berlin case reading its
# ground from one file. The OpenStreetMap and GeoJSON readers that make
# the drivers, and the check that every committed driver is reproduced,
# are AeolusPrep.jl's (ONE-h, #457).
#
# Acceptance tests of the issue covered here:
#   * geometry is recovered: three hand-written buildings of known size
#     rasterised at dx = 5 m give the exact cell counts, at dx = 7 m an
#     area within one cell perimeter; the class code table round-trips;
#   * polygons round-trip exactly (three_buildings, the Berlin drivers' cities);
#   * errors: grid mismatch (both grids named), a missing variable, a
#     wrong schema version, a stale source;
#   * bit-identical to the OSM-direct path: the Berlin smoke geometry from
#     its static driver (masks, classes, heightmap, view factors) and the
#     summer rasterisation equal the digests captured before the switch
#     (test/data/input_static/digests.toml), and so do the Mitte sample at
#     32 × 32 × 16 (its polygons rasterised there; nested and periodic) and
#     the nest parent's land use; the 12-step smoke state is checked
#     against the same capture in test/config_berlin.jl (berlin_smoke.toml
#     now runs from its static driver), as is the round trip, whose
#     [inputs] now list the static drivers (asserted here);
#   * a committed driver's rasters are its own polygons rasterised.

using Test
using TOML
using Aeolus
using Aeolus.InputDrivers
using Aeolus.InputDrivers: check_grid
using Aeolus.Config
using Aeolus.Grids: xnodes, znodes
using Aeolus.NCIO: NCDataset
import NCDatasets

const BCI = Aeolus.UrbanCase
include(joinpath(@__DIR__, "support", "berlin_configs.jl"))
include(joinpath(@__DIR__, "support", "berlin_fixture.jl"))

const IS_DATA = joinpath(@__DIR__, "data", "input_static")
const IS_DIGESTS = TOML.parsefile(joinpath(IS_DATA, "digests.toml"))

"A copy of the driver `src` at `dst` without the variables `skip`, with global attributes `attrs` changed."
function copy_driver(src, dst; skip = String[], attrs = Dict{String,Any}())
    NCDataset(src) do a
        NCDataset(dst, "c"; format = :netcdf4) do b
            for (k, v) in a.attrib
                b.attrib[k] = get(attrs, k, v)
            end
            for (d, n) in a.dim
                b.dim[d] = n
            end
            for name in keys(a)
                name in skip && continue
                v = a[name]
                nv = NCDatasets.defVar(b, name, eltype(v.var), NCDatasets.dimnames(v);
                                       attrib = collect(v.attrib))
                nv[ntuple(_ -> Colon(), ndims(v))...] = v.var[ntuple(_ -> Colon(), ndims(v))...]
            end
        end
    end
    return dst
end

"""
    three_buildings() -> UrbanCity

Three hand-written buildings and a park in mesh metres (what
test/data/input_static/three_buildings.geojson held before the GeoJSON
reader moved to AeolusPrep.jl, ONE-h #457): a 40 × 20 m block of 20 m, a
30 × 30 m block of 30 m, an L of 15 m with a 10 × 10 m courtyard (two
rings, each followed by a `NaN` vertex) and a 40 × 50 m park.
"""
function three_buildings()
    ring(v...) = NTuple{2,Float64}[Float64.(p) for p in v]
    L = vcat(ring((10, 50), (60, 50), (60, 70), (30, 70), (30, 100), (10, 100), (10, 50)), [(NaN, NaN)],
             ring((15, 75), (25, 75), (25, 85), (15, 85), (15, 75)), [(NaN, NaN)])
    return UrbanCity([Building(ring((10, 10), (50, 10), (50, 30), (10, 30), (10, 10)), 20.0),
                      Building(ring((70, 10), (100, 10), (100, 40), (70, 40), (70, 10)), 30.0),
                      Building(L, 15.0)],
                     [Building(ring((70, 60), (110, 60), (110, 110), (70, 110), (70, 60)), 0.0, SURFACE_PARK)])
end

meshof(cfg) = RegularGrid(; size = cfg.size, extent = cfg.extent, halo = cfg.halo)
geo_mesh(L, dx) = RegularGrid(; size = (round(Int, L / dx), round(Int, L / dx), 4), extent = (L, L, 40.0),
                              halo = (3, 3, 3))
const ANCHOR = (origin_lat = 52.5, origin_lon = 13.4, origin_x = 0.0, origin_y = 0.0)

@testset "InputDrivers: geometry of three known buildings is recovered" begin
    city = three_buildings()
    @test length(city.buildings) == 3 && length(city.patches) == 1
    # the L-shape with a courtyard: two rings, each closed and NaN-terminated
    L = city.buildings[3].polygon
    @test count(p -> isnan(p[1]), L) == 2 && isnan(L[end][1])
    area = (800.0, 900.0, 50 * 20 + 20 * 30 - 10 * 10)     # 40×20, 30×30, L (1600) − courtyard (100)
    perim = (120.0, 120.0, 200.0 + 40.0)
    # dx = 5 m: every edge on a cell face, so the counts are exact
    st = static_inputs(city, geo_mesh(120.0, 5.0); ANCHOR..., class_lai = Dict(SURFACE_PARK => 3.0))
    n5 = [count(==(Int32(b)), st.building_id) for b in 1:3]
    @test n5 == [32, 36, 60]
    @test n5[3] == area[3] / 25
    @test count(>(0), st.building_height) == sum(n5)
    @test all(st.building_height[st.building_id .== 2] .== 30.0)
    @test all(st.surface_class[st.building_id .>= 0] .== SURFACE_ROOF)
    @test all(st.ground_class[st.building_id .>= 0] .== SURFACE_STREET)   # no patch under them
    # the park: 40 × 50 m, its leaf area on park cells only
    @test count(==(SURFACE_PARK), st.surface_class) == 80
    @test all(st.lai[st.surface_class .== SURFACE_PARK] .== 3.0) && sum(st.lai) == 80 * 3.0
    # the courtyard is open ground
    xs = collect(xnodes(geo_mesh(120.0, 5.0), CellCenter()))
    i, j = findfirst(==(17.5), xs), findfirst(==(77.5), xs)
    @test st.building_id[i, j] == -1 && st.surface_class[i, j] == SURFACE_STREET
    # the solid mask: every cell centre at or below the roof
    m5 = geo_mesh(120.0, 5.0)
    sm, sfc = solid_and_classes(st, m5; tag_walls = true)
    zc = znodes(m5, CellCenter())                        # 5, 15, 25, 35 m
    @test count(sm.mask) == 32 * count(<=(20), zc) + 36 * count(<=(30), zc) + 60 * count(<=(15), zc)
    @test count(==(SURFACE_WALL), sfc) > 0 && all(sfc[st.building_id .>= 0] .== SURFACE_ROOF)
    # dx = 7 m: the area within one cell perimeter of the analytic area
    st7 = static_inputs(city, geo_mesh(126.0, 7.0); ANCHOR...)
    for b in 1:3
        err = abs(count(==(Int32(b)), st7.building_id) * 49.0 - area[b])
        @test err <= perim[b] * 7.0 / 2                 # half a cell along the perimeter (measured: 82, 80, 166 m²)
    end
    # the code table round-trips through the file
    p = tempname() * ".nc"
    write_static_driver(p, st)
    NCDataset(p) do ds
        @test collect(ds["surface_class"].attrib["flag_values"]) == UInt8[first(c) for c in SURFACE_CLASSES]
        @test split(ds["surface_class"].attrib["flag_meanings"]) == [last(c) for c in SURFACE_CLASSES]
        @test ds.attrib["projection"] == "equirectangular" && ds.attrib["aeolus_schema_version"] == 1
        @test ds["lai"].attrib["standard_name"] == "leaf_area_index"
    end
    st2 = read_static_driver(p; mesh = m5, ANCHOR...)
    @test st2.surface_class == st.surface_class && st2.building_id == st.building_id &&
          st2.building_height == st.building_height && st2.lai == st.lai && st2.ground_class == st.ground_class
    rm(p)
end

@testset "InputDrivers: polygons round-trip exactly" begin
    m = geo_mesh(160.0, 10.0)
    # the cities of the committed Berlin drivers (OpenStreetMap, NaN-separated rings and the
    # OSM attributes: levels, roof shapes, materials) and the hand-written one
    berlin(s) = (c = site_config(berlin_toml(s)); BCI.case_static_inputs(c, meshof(c)).city)
    for (label, city) in ("summer" => berlin("summer"), "mitte_sample" => berlin("mitte_sample"),
                          "three_buildings" => three_buildings())
        p = tempname() * ".nc"
        write_static_driver(p, city, m; ANCHOR...)
        back = read_static_driver(p; mesh = m, ANCHOR...).city
        @test isequal(back, city)
        @test length(back.buildings) == length(city.buildings) > 0
        # vertices bit for bit, rings (NaN separators) and attributes
        @test all(isequal(a.polygon, b.polygon) for (a, b) in zip(back.buildings, city.buildings))
        @test sum(b -> count(v -> isnan(v[1]), b.polygon), back.buildings) ==
              sum(b -> count(v -> isnan(v[1]), b.polygon), city.buildings)
        rm(p)
    end
    c = berlin("summer")
    @test count(b -> !isnan(b.levels), c.buildings) > 0
    @test count(b -> !isempty(b.roof), c.buildings) > 0
end

@testset "InputDrivers: errors" begin
    city = three_buildings()
    m = geo_mesh(120.0, 5.0)
    src = tempname() * ".geojson"
    write(src, "{\"type\": \"FeatureCollection\", \"features\": []}\n")   # any source file: its hash
    p = tempname() * ".nc"
    write_static_driver(p, city, m; ANCHOR..., source = src)
    @test isempty(check_driver(p, STATIC_SCHEMA))
    NCDataset(p) do ds
        @test ds.attrib["source_fnv1a64"] == file_fnv1a64(src)
        @test all(k -> haskey(ds.attrib, k), DRIVER_ATTRIBUTES)
    end
    # a grid mismatch names both grids
    e = try
        read_static_driver(p; mesh = geo_mesh(120.0, 10.0), ANCHOR...); nothing
    catch err
        sprint(showerror, err)
    end
    @test e !== nothing && occursin("24 × 24 cells of 5.0 × 5.0 m", e) && occursin("12 × 12 cells of 10.0 × 10.0 m", e)
    # … and so does an anchor elsewhere
    e = try
        read_static_driver(p; mesh = m, ANCHOR..., origin_x = 10.0); nothing
    catch err
        sprint(showerror, err)
    end
    @test e !== nothing && occursin("at (0.0, 0.0) m", e) && occursin("at (10.0, 0.0) m", e)
    # a missing required variable is named
    q = copy_driver(p, tempname() * ".nc"; skip = ["lai"])
    probs = check_driver(q, STATIC_SCHEMA)
    @test any(s -> occursin("missing required variable lai", s), probs)
    @test_throws ErrorException read_static_driver(q; mesh = m, ANCHOR...)
    # a wrong schema version
    q2 = copy_driver(p, tempname() * ".nc"; attrs = Dict{String,Any}("aeolus_schema_version" => 2))
    @test any(s -> occursin("aeolus_schema_version = 2", s), check_driver(q2, STATIC_SCHEMA))
    # the tiles schema refuses a resolved file
    @test any(s -> occursin("aeolus_form", s), check_driver(p, STATIC_TILES_SCHEMA))
    # a value out of range
    q3 = copy_driver(p, tempname() * ".nc")
    NCDataset(q3, "a") do ds
        ds["building_height"].var[1, 1] = -5.0
    end
    @test any(s -> occursin("building_height has 1 value(s) outside", s), check_driver(q3, STATIC_SCHEMA))
    # a source that changed since: stale
    open(src, "a") do io
        println(io, " ")
    end
    probs = check_driver(p, STATIC_SCHEMA)
    @test length(probs) == 1 && occursin("stale", probs[1]) && occursin(file_fnv1a64(src), probs[1])
    @test isempty(check_driver(p, STATIC_SCHEMA; source = false))
    foreach(rm, (p, q, q2, q3, src))
    # the schema renders for the docs
    md = schema_markdown(STATIC_SCHEMA)
    @test occursin("| `surface_class` | x, y | `UInt8` |", md) && count(==('\n'), md) == length(STATIC_SCHEMA.variables) + 2
end

# ---------------------------------------------------------------------------
# the Berlin case
# ---------------------------------------------------------------------------

function geom_digests_equal(label, g)
    d = IS_DIGESTS[label]
    got = Dict("mask" => digest(g.sm.mask), "sfc" => digest(g.sfc), "heightmap" => digest(g.heightmap),
               "F" => digest(g.vfac.F), "F_sky" => digest(g.vfac.F_sky), "area" => digest(g.vfac.area),
               "svf" => digest(g.svf))
    bad = [k for (k, v) in got if v != d[k]]
    isempty(bad) || @info "geometry digests differing from the OSM-direct path" label bad
    return isempty(bad) && length(g.city.buildings) == d["buildings"] && length(g.city.patches) == d["patches"]
end

@testset "InputDrivers: the Berlin geometry from its static driver is the OSM path's, bit for bit" begin
    # smoke: from the committed static driver
    cs = site_config(berlin_toml("smoke"))
    @test endswith(cs.static, "urban/input/berlin_smoke_static.nc")
    @test geom_digests_equal("smoke", BCI.build_geometry(cs, meshof(cs)))
    # the Mitte sample at 32 × 32 × 16, nested (frame) and periodic (no frame, wrapped
    # walls): its committed driver's polygons rasterised on that grid give the
    # digests the OSM file did in memory (before ONE-h, #457)
    full = site_config(berlin_toml("mitte_sample"))
    mitte = BCI.case_static_inputs(full, meshof(full)).city
    for (label, c) in ("mitte_sample" => site_config(berlin_toml("mitte_sample"); size = (32, 32, 16)),
                       "mitte_periodic" => site_config(berlin_toml("mitte_sample"); size = (32, 32, 16),
                                                          lateral = :periodic,
                                                          overrides = ["nesting.enabled=false"]))
        m = meshof(c)
        ox, oy = BCI.anchor_xy(c, m)
        c = merge(c, (; static = write_static_driver(tempname() * ".nc", mitte, m; origin_lat = c.lat0,
                                                     origin_lon = c.lon0, origin_x = ox, origin_y = oy)))
        @test geom_digests_equal(label, BCI.build_geometry(c, m))
        rm(c.static)
    end
    # summer, 64 × 64 × 64: masks, classes and heightmap at setup
    c = site_config(berlin_toml("summer"))
    mesh = meshof(c)
    st = BCI.case_static_inputs(c, mesh)
    @test !isempty(st.attributes)                        # read from the file
    sm, sfc = solid_and_classes(st, mesh; frame = (c.boundary_frame, c.boundary_frame), bcs = BCI.case_scalar_bcs(c))
    d = IS_DIGESTS["summer"]
    @test digest(sm.mask) == d["mask"] && digest(sfc) == d["sfc"]
    @test digest(Aeolus.Radiosity.heightmap_from_solid_mask(sm)) == d["heightmap"]
    @test length(st.city.buildings) == d["buildings"] && length(st.city.patches) == d["patches"]
    # the file's rasters are its own polygons rasterised (park LAI 2.5, the summer source's)
    ox, oy = BCI.anchor_xy(c, mesh)
    mem = static_inputs(st.city, mesh; origin_lat = c.lat0, origin_lon = c.lon0, origin_x = ox, origin_y = oy,
                        class_lai = Dict(SURFACE_PARK => 2.5))
    @test all(f -> isequal(getfield(st, f), getfield(mem, f)), filter(!=(:attributes), fieldnames(StaticInputs)))
    # the nest parents' land use, from their static drivers and in memory
    for s in (:smoke, :summer)
        pc = BCI.nest_parent_config(site_config(berlin_toml(s)))
        pm = BCI.parent_mesh(pc)
        d = IS_DIGESTS["parent_$s"]
        for pl in (BCI.parent_landuse(pc, pm), BCI.parent_landuse_source(pc, pm))
            @test digest(pl.counts) == d["counts"] && digest(pl.λf) == d["lambda_f"] &&
                  digest(pl.hmean) == d["h_mean"] && digest(pl.hmax) == d["h_max"]
            @test (pl.nx, pl.ny, pl.dx, pl.x0, pl.y0, pl.r) == (d["nx"], d["ny"], d["dx"], d["x0"], d["y0"], d["r"])
        end
    end
    # the round trip hashes the static drivers under [inputs]
    inp = config_inputs(load_run_config(joinpath(berlin_configs_dir(), "berlin_smoke.toml")))
    @test inp["domain.main.static_driver"]["fnv1a64"] == content_hash(cs.static)
    @test inp["domain.main.urban.parent_static"]["fnv1a64"] == content_hash(cs.parent_static)
    # a grid the file was not made for is refused, naming both grids and the command
    e = try
        BCI.build_geometry(site_config(berlin_toml("smoke"); size = (8, 8, 16)), meshof(site_config(berlin_toml("smoke"); size = (8, 8, 16))))
        nothing
    catch err
        sprint(showerror, err)
    end
    @test e !== nothing && occursin("16 × 16 cells", e) && occursin("8 × 8 cells", e) && occursin("aeolusprep static", e)
    # a missing driver: a config error before any run folder, with the exact command
    cfg = load_run_config(joinpath(berlin_configs_dir(), "berlin_smoke.toml"); overrides = ["static_driver=\"nope.nc\""])
    e = try
        Aeolus.Config.build(Aeolus.SimulationDriver.SimulationDriverType(), cfg); nothing
    catch err
        err
    end
    @test e isa ConfigError && occursin("AeolusPrep static $(joinpath(berlin_configs_dir(), "berlin_smoke.toml"))", sprint(showerror, e))
    # no static driver at all: no ground, an error naming AeolusPrep, not a flat city
    e = try
        BCI.build_geometry(site_config(berlin_toml("smoke"); static = nothing), meshof(cs)); nothing
    catch err
        sprint(showerror, err)
    end
    @test e !== nothing && occursin("static is \"none\"", e) && occursin("AeolusPrep", e)
end

@testset "InputDrivers: the committed Berlin static drivers" begin
    # AeolusPrep.jl reproduces them (its test/recipes.jl); here: small, valid
    for s in ("summer", "smoke", "mitte_sample")
        cc = site_config(berlin_toml(s))
        for (f, sch) in ((cc.static, STATIC_SCHEMA), (cc.parent_static, STATIC_TILES_SCHEMA))
            @test filesize(f) <= 2 * 1024^2                 # committed: 2 MB or smaller
            @test isempty(check_driver(f, sch))
            # made by AeolusPrep (ONE-h, #457): the OSM source in its package, the
            # land-use table in Aeolus (relative, hashed: a changed table makes it stale)
            NCDataset(f) do ds
                @test ds.attrib["preprocessor"] == "aeolusprep static"
                @test sch === STATIC_SCHEMA ? startswith(ds.attrib["source"], "AeolusPrep:data/berlin/") :
                                              ds.attrib["source"] == "../../data/berlin_landuse_6km.csv"
            end
        end
    end
    smoke = joinpath(berlin_configs_dir(), "input", "berlin_mitte_sample_smoke_static.nc")
    @test filesize(smoke) <= 2 * 1024^2 && isempty(check_driver(smoke, STATIC_SCHEMA))
end

@testset "InputDrivers: the source is recorded per package (ONE-h, #457)" begin
    ref = Aeolus.InputDrivers._source_ref
    root = mktempdir()
    mk(p, name) = (mkpath(p); name === nothing || write(joinpath(p, "Project.toml"), "name = \"$name\"\n"); p)
    model = mk(joinpath(root, "Model.jl"), "Model")
    prep = mk(joinpath(root, "deep", "Prep.jl"), "Prep")
    mk(joinpath(model, "docs"), nothing)                        # a Project.toml without a name is not a package
    src_here = joinpath(model, "examples", "data", "s.csv"); mkpath(dirname(src_here)); write(src_here, "x")
    src_prep = joinpath(prep, "data", "berlin", "a.osm"); mkpath(dirname(src_prep)); write(src_prep, "x")
    out = joinpath(model, "examples", "case", "input", "d.nc")
    # the same package: relative to the driver's directory, as before
    @test ref(src_here, out) == "../../data/s.csv"
    # another package: its name and the path in it, wherever it is checked out
    @test ref(src_prep, out) == "Prep:data/berlin/a.osm"
    @test ref(src_prep, joinpath(model, "docs", "d.nc")) == "Prep:data/berlin/a.osm"
    # outside any package: absolute
    loose = joinpath(root, "loose.osm"); write(loose, "x")
    @test ref(loose, out) == loose
    # a source in another package is not resolved here (no stale check)
    p = tempname() * ".nc"
    write_static_driver(p, static_inputs(three_buildings(), geo_mesh(120.0, 5.0); ANCHOR...); source = src_prep, preprocessor = "test")
    NCDataset(p) do ds
        @test ds.attrib["source"] == "Prep:data/berlin/a.osm" && ds.attrib["preprocessor"] == "test"
        @test Aeolus.InputDrivers.source_path(p, ds) === nothing
    end
    @test isempty(check_driver(p, STATIC_SCHEMA))
    rm(p)
end
