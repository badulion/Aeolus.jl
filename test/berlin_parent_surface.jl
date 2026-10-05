# The Berlin parent's tiled surface (N-parent-d, #249): the unresolved
# buildings' roughness (Macdonald et al. 1998), the OSM multipolygon
# ingest of the 6 km extract, the class-to-surface factory shared with the
# child, the per-column mosaic, and its consistency with the child under
# the child's footprint.

using Test
using Aeolus
using Aeolus.SurfaceTiles: macdonald_roughness

isdefined(Main, :BerlinCase) ||
    include(joinpath(@__DIR__, "..", "examples", "urban", "BerlinCase.jl"))
const BCP = Main.BerlinCase

@testset "Berlin parent surface (#249)" begin

@testset "Macdonald et al. (1998) roughness of unresolved buildings" begin
    # d/h = 1 + A^(-λp)(λp - 1): at λp = 0.3, A = 4.43 → 1 - 0.7·4.43^(-0.3) = 0.5521
    r = macdonald_roughness(0.3, 0.3, 1.0)
    @test isapprox(r.d, 0.552, atol = 1e-3)
    @test r.d ≈ 1 - 0.7 * 4.43^(-0.3) rtol = 1e-14
    # z0 and d scale with h
    r20 = macdonald_roughness(0.3, 0.2, 20.0)
    r1 = macdonald_roughness(0.3, 0.2, 1.0)
    @test r20.z0 ≈ 20r1.z0 rtol = 1e-14
    @test r20.d ≈ 20r1.d rtol = 1e-14
    # the maximum of z0/h over cube arrays (λf = λp) lies at intermediate
    # density, λf ≈ 0.15-0.2 (Grimmond & Oke 1999, JAM 38, 1262)
    λs = 0.01:0.001:0.99
    z0s = [macdonald_roughness(λ, λ, 1.0).z0 for λ in λs]
    λpk = λs[argmax(z0s)]
    @test 0.15 <= λpk <= 0.2
    @test 0.1 < maximum(z0s) < 0.15          # z0/h ≈ 0.135 at the peak (measured)
    # limits: no buildings → d, z0 → 0; a closed roof → d → h, z0 → 0
    r0 = macdonald_roughness(1e-9, 1e-9, 25.0)
    @test r0.d < 1e-7 && r0.z0 < 1e-12
    @test macdonald_roughness(0.0, 0.0, 25.0) == (z0 = 0.0, d = 0.0)
    r1 = macdonald_roughness(1 - 1e-9, 0.5, 25.0)
    @test abs(r1.d - 25.0) < 1e-6 && r1.z0 < 1e-6
    @test macdonald_roughness(1.0, 0.5, 25.0).d == 25.0
    # d increases monotonically with λp (sheltering), 0 ≤ z0 < h - d
    ds = [macdonald_roughness(λ, 0.2, 1.0).d for λ in λs]
    @test all(diff(ds) .> 0)
    @test all(λ -> (local q = macdonald_roughness(λ, λ, 1.0); 0 <= q.z0 < 1 - q.d), λs)
    # square (aligned) arrays, A = 3.59, are less displaced than staggered ones
    @test macdonald_roughness(0.3, 0.3, 1.0; A = 3.59, β = 0.55).d < r.d
    @test_throws ArgumentError macdonald_roughness(1.2, 0.1, 10.0)
    @test_throws ArgumentError macdonald_roughness(0.2, -0.1, 10.0)
end

@testset "OSM multipolygon relations: rings joined, holes kept" begin
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
    city = BCP.load_city_osm(path; lat0 = lat0, lon0 = lon0, x_shift = 100, y_shift = 100)
    @test length(city.patches) == 2                    # the park way + one multipolygon (the route is ignored)
    water = only(filter(p -> p.class == BCP.SURFACE_WATER, city.patches))
    @test count(p -> isnan(p[1]), water.polygon) == 2  # two closed rings, NaN-terminated
    mesh = Aeolus.Grids.RegularGrid(; size = (20, 20, 1), extent = (200.0, 200.0, 6.25))
    sm, sfc = BCP.surface_class_map(city, mesh; tag_walls = false)
    xc(i) = (i - 0.5) * 10 - 100                        # cell centre relative to the anchor
    inwater(x, y) = max(abs(x), abs(y)) < 50 && max(abs(x), abs(y)) > 20
    @test all(sfc[i, j] == (inwater(xc(i), xc(j)) ? BCP.SURFACE_WATER :
                            (60 < xc(i) < 90 && 60 < xc(j) < xc(i) ? BCP.SURFACE_PARK : BCP.SURFACE_STREET))
              for i in 1:20, j in 1:20 if !(60 < xc(i) < 90 && abs(xc(j) - xc(i)) < 6))
    @test sfc[10, 10] == BCP.SURFACE_STREET             # the island is a hole
    @test count(==(BCP.SURFACE_WATER), sfc) == 100 - 16 # 10×10 cells of 10 m minus the 4×4 island
    @test BCP.polygon_bbox(water.polygon) == BCP.polygon_bbox(filter(p -> !isnan(p[1]), water.polygon))
    # the child's own OSM file has no relations: its city is unchanged
    @test !occursin("<relation", read(BCP.CaseConfig().osm_path, String))
end

end # testset
