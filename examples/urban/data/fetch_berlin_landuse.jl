# Land use of the Berlin parent domain (#249): the 6 km × 6 km box centred
# on the child's anchor (CaseConfig().lat0/lon0, Alexanderplatz), from
# OpenStreetMap, coarse-grained to the parent grid.
#
#   julia --project=. examples/urban/data/fetch_berlin_landuse.jl            # download + process
#   julia --project=. examples/urban/data/fetch_berlin_landuse.jl --cached   # reuse the raw extract
#
# 1. Overpass query (public API, no auth; `curl`) for buildings, parks,
#    green land use and water in the box — ways AND multipolygon relations
#    (the Spree, the parks and many courtyard blocks are relations). The
#    raw OSM goes to data/raw/ (gitignored; it is ~35 MB and is NOT
#    committed). Highways are not queried: the classifier's ground class
#    is street, so a highway carries no information for it.
# 2. The extract is classified with the Berlin case's own `load_city_osm`
#    / `osm_class` (one classifier) and rasterised with its own
#    `surface_class_map` (building-adjacent cells tagged SURFACE_WALL) on
#    the child's 15.625 m grid, 384 × 384 cells, cell centres coinciding
#    with the child's under its footprint; building heights per cell come
#    from `osm_height_map` (the tallest footprint).
# 3. Coarse-grained by r = 2 to the 192 × 192 parent grid (31.25 m): per
#    parent cell the class counts of its 2 × 2 child cells (fractions
#    n/4, exact) and the building statistics λ_p, λ_f, mean and maximum
#    height, written with their provenance to
#    data/berlin_landuse_6km.csv.
#
# Data © OpenStreetMap contributors, ODbL 1.0 (https://www.openstreetmap.org/copyright).

include(joinpath(@__DIR__, "..", "BerlinCase.jl"))
using .BerlinCase
using Aeolus
using Printf
using Dates

const BC = BerlinCase
const CFG = CaseConfig(; verbose = false)
const LAT0, LON0 = CFG.lat0, CFG.lon0
const L_PARENT = 6000.0                       # m, the parent box (centred on the anchor)
const DX_CHILD = CFG.extent[1] / CFG.size[1]  # 15.625 m
const R = 2                                   # child cells per parent cell, per axis
const NF = round(Int, L_PARENT / DX_CHILD)    # 384 fine cells
const NP = NF ÷ R                             # 192 parent cells
const MARGIN = 100.0                          # m of extra OSM around the box
const RAW = joinpath(@__DIR__, "raw", "berlin_landuse_6km.osm")
const CSV = joinpath(@__DIR__, "berlin_landuse_6km.csv")
const ENDPOINTS = ("https://overpass-api.de/api/interpreter",
                   "https://overpass.kumi.systems/api/interpreter")

"The query box (south, west, north, east) in degrees: the parent box plus a margin."
function query_box()
    h = L_PARENT / 2 + MARGIN
    dlat = rad2deg(h / BC._EARTH_R)
    dlon = rad2deg(h / (BC._EARTH_R * cosd(LAT0)))
    return round.((LAT0 - dlat, LON0 - dlon, LAT0 + dlat, LON0 + dlon); digits = 4)
end

function overpass_query(box)
    b = join(box, ",")
    sel = ("[\"building\"]", "[\"leisure\"=\"park\"]",
           "[\"landuse\"~\"^(grass|forest|meadow|recreation_ground)\$\"]",
           "[\"natural\"=\"water\"]", "[\"waterway\"=\"riverbank\"]")
    body = join(("  $(t)$(s)($b);" for s in sel for t in ("way", "relation")), "\n")
    return "[out:xml][timeout:300];\n(\n$body\n);\n(._;>;);\nout body;\n"
end

function download(box)
    mkpath(dirname(RAW))
    q = overpass_query(box)
    for attempt in 1:3, url in ENDPOINTS
        @printf("Overpass %s (attempt %d)…\n", url, attempt)
        ok = success(pipeline(`curl -sS --max-time 900 -A "Aeolus.jl OSM ingest" $url
                               --data-urlencode $("data=" * q)`; stdout = RAW))
        head = ok ? String(read(RAW, min(filesize(RAW), 400))) : ""
        ok && occursin("<osm", head) && endswith(rstrip(String(read(RAW))), "</osm>") && return RAW
        sleep(20)
    end
    error("Overpass download failed; retry later or pass --cached with a raw extract in $RAW")
end

"Counts of the height information of the building ways and relations of the raw extract."
function height_provenance(text)
    n = Dict(:height => 0, :levels => 0, :none => 0)
    for m in eachmatch(r"<(way|relation)\s+id=\"\d+\"[^>]*>(.*?)</\1>"s, text)
        body = m.captures[2]
        occursin("k=\"building\"", body) || continue
        k = occursin("k=\"height\"", body) ? :height :
            occursin("k=\"building:levels\"", body) ? :levels : :none
        n[k] += 1
    end
    return n
end

function main(; cached = false)
    box = query_box()
    cached && isfile(RAW) || download(box)
    text = read(RAW, String)
    osm_base = something(match(r"osm_base=\"([^\"]+)\"", text), (captures = ["unknown"],)).captures[1]
    hp = height_provenance(text)
    nrel = count("<relation", text)

    # the child's mesh, widened to the parent box (one level of the child's
    # Δz: footprints are solid where the child's first level is)
    dz = CFG.extent[3] / CFG.size[3]
    mesh = Aeolus.Grids.RegularGrid(; size = (NF, NF, 1), extent = (L_PARENT, L_PARENT, dz))
    t = @elapsed city = BC.load_city_osm(RAW; lat0 = LAT0, lon0 = LON0,
                                         x_shift = L_PARENT / 2, y_shift = L_PARENT / 2,
                                         min_height = CFG.min_height)
    @printf("%d buildings, %d park/water patches (%.1f s)\n", length(city.buildings),
            length(city.patches), t)
    t = @elapsed (sm_sfc = BC.surface_class_map(city, mesh; tag_walls = true))
    sfc = sm_sfc[2]
    hf = BC.osm_height_map(city, mesh, 1)
    h = [sfc[I] == BC.SURFACE_ROOF ? hf[I] : 0.0 for I in CartesianIndices(sfc)]
    @printf("rasterised %d × %d in %.1f s; classes %s\n", NF, NF, t,
            join((count(==(c), sfc) for c in 0x01:0x05), "/"))

    # λ_f: windward facade area per unit ground area, for the four cardinal
    # wind directions (each step up in the height map along the wind is
    # the windward face of the cell it rises into), then averaged
    # (no face at the edge of the box: the height outside is the edge's)
    hpad(i, j) = h[clamp(i, 1, NF), clamp(j, 1, NF)]
    face = zeros(NF, NF)
    for j in 1:NF, i in 1:NF
        face[i, j] = (max(h[i, j] - hpad(i - 1, j), 0.0) + max(h[i, j] - hpad(i + 1, j), 0.0) +
                      max(h[i, j] - hpad(i, j - 1), 0.0) + max(h[i, j] - hpad(i, j + 1), 0.0)) * DX_CHILD / 4
    end
    Ap = (R * DX_CHILD)^2
    open(CSV, "w") do io
        println(io, "# Berlin parent land use (#249): per parent cell, class counts of its child cells and building statistics.")
        println(io, "# source: OpenStreetMap via the Overpass API, osm_base = $osm_base, fetched $(Dates.today()) by examples/urban/data/fetch_berlin_landuse.jl")
        println(io, "# licence: (c) OpenStreetMap contributors, ODbL 1.0, https://www.openstreetmap.org/copyright")
        println(io, "# query_box_deg (S,W,N,E) = $(join(box, ","))  (parent box + $(MARGIN) m); ways and type=multipolygon relations ($nrel relations in the extract)")
        println(io, "# anchor lat0 = $LAT0, lon0 = $LON0 (CaseConfig); projection: equirectangular about the anchor (osm_ingest.jl latlon_to_meters)")
        println(io, "# parent grid: nx = $NP, ny = $NP, dx = $(R * DX_CHILD) m, x0 = y0 = $(-L_PARENT / 2) m relative to the anchor; refinement r = $R child cells of $(DX_CHILD) m per axis")
        println(io, "# classifier: BerlinCase.osm_class (load_city_osm), min_height = $(CFG.min_height) m, untagged height 12 m, building:levels x 3 m; rasteriser: surface_class_map (tag_walls)")
        println(io, "# building height information (ways + relations): height tag $(hp[:height]), building:levels only $(hp[:levels]), neither (12 m default) $(hp[:none])")
        println(io, "# classes: 1 street, 2 roof, 3 park, 4 water, 5 wall (building-adjacent street); n_c = child cells of class c (out of $(R^2))")
        println(io, "# lambda_p = roof fraction; lambda_f = windward facade area / ground area, mean over the four cardinal wind directions, from the child-resolution height map")
        println(io, "# h_mean = roof-area-weighted mean building height (m, 0 without buildings); h_max = tallest building (m)")
        println(io, "I,J,n1,n2,n3,n4,n5,lambda_p,lambda_f,h_mean,h_max")
        for J in 1:NP, I in 1:NP
            is, js = (I - 1) * R + 1:I * R, (J - 1) * R + 1:J * R
            n = [count(==(UInt8(c)), view(sfc, is, js)) for c in 1:5]
            roof = [h[i, j] for i in is, j in js if sfc[i, j] == BC.SURFACE_ROOF]
            λp = n[2] / R^2
            λf = sum(view(face, is, js)) / Ap
            hm = isempty(roof) ? 0.0 : sum(roof) / length(roof)
            hx = isempty(roof) ? 0.0 : maximum(roof)
            @printf(io, "%d,%d,%d,%d,%d,%d,%d,%.4f,%.5f,%.2f,%.2f\n", I, J, n..., λp, λf, hm, hx)
        end
    end
    @printf("wrote %s (%.0f kB)\n", CSV, filesize(CSV) / 1024)
    return sfc
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(; cached = "--cached" in ARGS)
end
