# The static driver (CFG-f, #361): Aeolus.InputDrivers — the conventions,
# the schema check, the OSM/GeoJSON preprocessors and the Berlin case
# reading its ground from one file.
#
# Acceptance tests of the issue covered here:
#   * geometry is recovered: three hand-written buildings of known size
#     rasterised at dx = 5 m give the exact cell counts, at dx = 7 m an
#     area within one cell perimeter; the class code table round-trips;
#   * polygons round-trip exactly (both OSM files, city.geojson);
#   * errors: grid mismatch (both grids named), a missing variable, a
#     wrong schema version, a stale source;
#   * bit-identical to the OSM-direct path: the Berlin smoke geometry from
#     its static driver (masks, classes, heightmap, view factors) and the
#     summer rasterisation equal the digests captured before the switch
#     (test/data/input_static/digests.toml), and so do the in-memory path
#     and the nest parent's land use; the 12-step smoke state is checked
#     against the same capture in test/config_berlin.jl (berlin_smoke.toml
#     now runs from its static driver), as is the round trip, whose
#     [inputs] now list the static drivers (asserted here);
#   * generated files are reproducible: re-running the preprocessor gives
#     the committed files' data variables byte for byte.

using Test
using TOML
using Aeolus
using Aeolus.InputDrivers
using Aeolus.InputDrivers: check_grid, VOLATILE_ATTRIBUTES
using Aeolus.Config
using Aeolus.Grids: xnodes, znodes
using Aeolus.NCIO: NCDataset
import NCDatasets

const BCI = Aeolus.UrbanCase
include(joinpath(@__DIR__, "support", "berlin_configs.jl"))
include(joinpath(@__DIR__, "support", "berlin_fixture.jl"))

const IS_DATA = joinpath(@__DIR__, "data", "input_static")
const IS_DIGESTS = TOML.parsefile(joinpath(IS_DATA, "digests.toml"))
const URBAN_DATA = joinpath(@__DIR__, "..", "examples", "urban", "data")

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

geo_mesh(L, dx) = RegularGrid(; size = (round(Int, L / dx), round(Int, L / dx), 4), extent = (L, L, 40.0),
                              halo = (3, 3, 3))
const ANCHOR = (origin_lat = 52.5, origin_lon = 13.4, origin_x = 0.0, origin_y = 0.0)

@testset "InputDrivers: geometry of three known buildings is recovered" begin
    city = load_city_geojson(joinpath(IS_DATA, "three_buildings.geojson"))
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
    for (label, city) in (
            "alexanderplatz_real.osm" => load_city_osm(joinpath(URBAN_DATA, "alexanderplatz_real.osm");
                                                       lat0 = 52.522, lon0 = 13.41, x_shift = 500, y_shift = 500),
            "sample_berlin_mitte.osm" => load_city_osm(joinpath(URBAN_DATA, "sample_berlin_mitte.osm");
                                                       lat0 = 52.522, lon0 = 13.41, x_shift = 500, y_shift = 500),
            "city.geojson" => load_city_geojson(joinpath(URBAN_DATA, "city.geojson")),
            "three_buildings.geojson" => load_city_geojson(joinpath(IS_DATA, "three_buildings.geojson")))
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
    # the OSM attributes are kept (alexanderplatz_real.osm has levels, roof shapes, materials)
    c = load_city_osm(joinpath(URBAN_DATA, "alexanderplatz_real.osm"); lat0 = 52.522, lon0 = 13.41)
    @test count(b -> !isnan(b.levels), c.buildings) > 0
    @test count(b -> !isempty(b.roof), c.buildings) > 0
end

@testset "InputDrivers: errors" begin
    city = load_city_geojson(joinpath(IS_DATA, "three_buildings.geojson"))
    m = geo_mesh(120.0, 5.0)
    src = tempname() * ".geojson"
    cp(joinpath(IS_DATA, "three_buildings.geojson"), src)
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

# Moved from test/berlin_parent_surface.jl (#320); the preprocessor is
# src/preprocess/osm.jl (multipolygon relations, #249).
@testset "InputDrivers: OSM multipolygon relations — rings joined, holes kept" begin
    # A 100 m water square (outer ring split over two ways, one reversed)
    # with a 40 m island (inner ring), around the anchor; a park way.
    lat0, lon0 = 52.5, 13.4
    R = 6_371_000.0
    dlat(y) = lat0 + rad2deg(y / R)
    dlon(x) = lon0 + rad2deg(x / (R * cosd(lat0)))
    pts = Dict(1 => (-50, -50), 2 => (50, -50), 3 => (50, 50), 4 => (-50, 50),
               5 => (-20, -20), 6 => (20, -20), 7 => (20, 20), 8 => (-20, 20),
               9 => (60, 60), 10 => (90, 60), 11 => (90, 90))
    nodes = join(("<node id=\"$k\" lat=\"$(dlat(y))\" lon=\"$(dlon(x))\"/>" for (k, (x, y)) in pts), "\n")
    nd(ids) = join(("<nd ref=\"$i\"/>" for i in ids), "")
    xml = """<?xml version="1.0"?><osm>
    $nodes
    <way id="100">$(nd([1, 2, 3]))</way>
    <way id="101">$(nd([1, 4, 3]))</way>
    <way id="102">$(nd([5, 6, 7, 8, 5]))</way>
    <way id="103">$(nd([9, 10, 11, 9]))<tag k="leisure" v="park"/></way>
    <relation id="200"><member type="way" ref="100" role="outer"/><member type="way" ref="101" role="outer"/><member type="way" ref="102" role="inner"/><tag k="type" v="multipolygon"/><tag k="natural" v="water"/></relation>
    <relation id="201"><member type="way" ref="102" role="outer"/><tag k="type" v="route"/><tag k="natural" v="water"/></relation>
    </osm>"""
    path = tempname() * ".osm"
    write(path, xml)
    city = load_city_osm(path; lat0 = lat0, lon0 = lon0, x_shift = 100, y_shift = 100)
    rm(path)
    @test length(city.patches) == 2                    # the park way + one multipolygon (the route is ignored)
    water = only(filter(p -> p.class == SURFACE_WATER, city.patches))
    @test count(p -> isnan(p[1]), water.polygon) == 2  # two closed rings, NaN-terminated
    mesh = RegularGrid(; size = (20, 20, 1), extent = (200.0, 200.0, 6.25))
    sm, sfc = surface_class_map(city, mesh; tag_walls = false)
    xc(i) = (i - 0.5) * 10 - 100                        # cell centre relative to the anchor
    inwater(x, y) = max(abs(x), abs(y)) < 50 && max(abs(x), abs(y)) > 20
    @test all(sfc[i, j] == (inwater(xc(i), xc(j)) ? SURFACE_WATER :
                            (60 < xc(i) < 90 && 60 < xc(j) < xc(i) ? SURFACE_PARK : SURFACE_STREET))
              for i in 1:20, j in 1:20 if !(60 < xc(i) < 90 && abs(xc(j) - xc(i)) < 6))
    @test sfc[10, 10] == SURFACE_STREET                 # the island is a hole
    @test count(==(SURFACE_WATER), sfc) == 100 - 16     # 10×10 cells of 10 m minus the 4×4 island
    @test polygon_bbox(water.polygon) == polygon_bbox(filter(p -> !isnan(p[1]), water.polygon))
    # a file without relations (the Alexanderplatz extract) takes the way-only path
    @test !occursin("<relation", read(joinpath(URBAN_DATA, "alexanderplatz_real.osm"), String))
end

# ---------------------------------------------------------------------------
# the Berlin case
# ---------------------------------------------------------------------------

meshof(cfg) = RegularGrid(; size = cfg.size, extent = cfg.extent, halo = cfg.halo)
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
    cs = BCI.CaseConfig(berlin_toml("smoke"))
    @test endswith(cs.static, "configs/input/berlin_smoke_static.nc")
    @test geom_digests_equal("smoke", BCI.build_geometry(cs, meshof(cs)))
    # the in-memory path (static = none), nested (frame) and periodic (no frame, wrapped walls)
    for (label, c) in ("mitte_sample" => BCI.CaseConfig(berlin_toml("mitte_sample"); size = (32, 32, 16), static = nothing),
                       "mitte_periodic" => BCI.CaseConfig(berlin_toml("mitte_sample"); size = (32, 32, 16), static = nothing,
                                                          lateral = :periodic,
                                                          physics = Dict{Symbol,Any}(:nesting => false)))
        @test geom_digests_equal(label, BCI.build_geometry(c, meshof(c)))
    end
    # summer, 64 × 64 × 64: masks, classes and heightmap at setup
    c = BCI.CaseConfig(berlin_toml("summer"))
    mesh = meshof(c)
    st = BCI.case_static_inputs(c, mesh)
    @test !isempty(st.attributes)                        # read from the file
    sm, sfc = solid_and_classes(st, mesh; frame = (c.boundary_frame, c.boundary_frame), bcs = BCI.case_scalar_bcs(c))
    d = IS_DIGESTS["summer"]
    @test digest(sm.mask) == d["mask"] && digest(sfc) == d["sfc"]
    @test digest(Aeolus.Radiosity.heightmap_from_solid_mask(sm)) == d["heightmap"]
    @test length(st.city.buildings) == d["buildings"] && length(st.city.patches) == d["patches"]
    # the file and the in-memory path give the same inputs
    mem = BCI.urban_static_inputs(c, mesh)
    @test all(f -> isequal(getfield(st, f), getfield(mem, f)), filter(!=(:attributes), fieldnames(StaticInputs)))
    # the nest parents' land use, from their static drivers and in memory
    for s in (:smoke, :summer)
        pc = BCI.nest_parent_config(BCI.CaseConfig(berlin_toml(s)))
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
    @test inp["geometry.static"]["fnv1a64"] == content_hash(cs.static)
    @test inp["geometry.parent_static"]["fnv1a64"] == content_hash(cs.parent_static)
    # a grid the file was not made for is refused, naming both grids and the command
    e = try
        BCI.build_geometry(BCI.CaseConfig(berlin_toml("smoke"); size = (8, 8, 16)), meshof(BCI.CaseConfig(berlin_toml("smoke"); size = (8, 8, 16))))
        nothing
    catch err
        sprint(showerror, err)
    end
    @test e !== nothing && occursin("16 × 16 cells", e) && occursin("8 × 8 cells", e) && occursin("aeolus preprocess static", e)
    # a missing driver: a config error before any run folder, with the exact command
    cfg = load_run_config(joinpath(berlin_configs_dir(), "berlin_smoke.toml"); overrides = ["geometry.static=\"nope.nc\""])
    e = try
        Aeolus.Config.build(BCI.UrbanDriver(), cfg); nothing
    catch err
        err
    end
    @test e isa ConfigError && occursin("preprocess static $(joinpath(berlin_configs_dir(), "berlin_smoke.toml"))", sprint(showerror, e))
end

"The data variables of a driver (name => raw array) and its global attributes but the volatile ones."
function driver_content(path)
    NCDataset(path) do ds
        vars = Dict(name => ds[name].var[ntuple(_ -> Colon(), ndims(ds[name]))...] for name in keys(ds))
        attrs = Dict(k => v for (k, v) in ds.attrib if !(k in VOLATILE_ATTRIBUTES))
        (vars, attrs)
    end
end

@testset "InputDrivers: the committed Berlin static drivers are reproducible" begin
    dir = mktempdir()
    for s in ("summer", "smoke", "mitte_sample")
        toml = joinpath(berlin_configs_dir(), "berlin_$s.toml")
        cfg = load_run_config(toml)
        cc = BCI.CaseConfig(cfg)
        out, pout = joinpath(berlin_configs_dir(), "input", "_repro_$(s).nc"), joinpath(berlin_configs_dir(), "input", "_repro_$(s)_parent.nc")
        try
            files = Aeolus.Config.preprocess_static(BCI.UrbanDriver(), cfg; out = out, parent_out = pout)
            @test files == [out, pout]
            for (new, old) in ((out, cc.static), (pout, cc.parent_static))
                @test filesize(old) <= 2 * 1024^2                 # committed: 2 MB or smaller
                @test isempty(check_driver(old, occursin("parent", old) ? STATIC_TILES_SCHEMA : STATIC_SCHEMA))
                vn, an = driver_content(new); vo, ao = driver_content(old)
                @test keys(vn) == keys(vo)
                @test all(k -> isequal(vn[k], vo[k]), keys(vn))   # byte for byte (NaN separators included)
                @test an == ao                                     # same source, same hash
            end
        finally
            rm(out; force = true); rm(pout; force = true)
        end
    end
end

@testset "InputDrivers: aeolus preprocess static (the CLI)" begin
    io, err = IOBuffer(), IOBuffer()
    @test Aeolus.Config.main(["preprocess"]; io = io, err = err) == 0
    @test occursin("static <config.toml>", String(take!(io)))
    dir = mktempdir()
    out, pout = joinpath(dir, "c.nc"), joinpath(dir, "p.nc")
    toml = joinpath(berlin_configs_dir(), "berlin_smoke.toml")
    rc = redirect_stdout(devnull) do
        Aeolus.Config.main(["preprocess", "static", toml, "--out", out, "--parent-out", pout]; io = io, err = err)
    end
    @test rc == 0 && isfile(out) && isfile(pout)
    @test isempty(check_driver(out, STATIC_SCHEMA)) && isempty(check_driver(pout, STATIC_TILES_SCHEMA))
    NCDataset(out) do ds
        # the source is recorded absolute outside the checkout, with its hash
        @test ds.attrib["source"] == BCI.CaseConfig(berlin_toml("smoke")).osm_path
        @test ds.attrib["source_fnv1a64"] == file_fnv1a64(BCI.CaseConfig(berlin_toml("smoke")).osm_path)
        @test ds.attrib["preprocessor"] == "aeolus preprocess static"
    end
    # a config whose static is "none" needs --out
    rc = Aeolus.Config.main(["preprocess", "static", toml, "--set", "geometry.static=\"none\""]; io = io, err = err)
    @test rc == 2
end

