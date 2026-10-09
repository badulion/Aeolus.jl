# Input data of the urban run type (EX-a, #438): the `aeolus preprocess`
# subcommands that fetch and make the sources its static and dynamic
# drivers are built from. They were scripts in examples/urban/data/:
#
#   fetch-osm       OpenStreetMap buildings, parks and water of a box (Overpass API)
#   fetch-landuse   the nest parent's land-use table (Overpass API, classified and coarse-grained)
#   fetch-sounding  radiosonde launches from IGRA2 (NOAA NCEI) as the sounding CSV
#   osm-sample      a synthetic OSM file (perimeter blocks, offices, park, water)
#
# The downloads use `curl` (and `unzip`, or Python's zipfile module, for
# IGRA2): offline tooling, never called by a run.

using ..Aeolus.Config: register_preprocessor!

"""
    _pp_args(args, valued, flags) -> (positionals, options, flags)

The arguments of a preprocessor (`--key value` / `--key=value`, flags);
an unknown option is an `ArgumentError` (exit code 2).
"""
function _pp_args(args, valued, flags)
    pos, sets, opts, fl = Config._parse_args(args, valued, flags)
    isempty(sets) || throw(ArgumentError("this preprocessor takes no --set"))
    return pos, opts, fl
end

function _pp_box(s::AbstractString)
    v = tryparse.(Float64, split(s, ','))
    (length(v) == 4 && all(!isnothing, v)) ||
        throw(ArgumentError("--box: expected S,W,N,E in degrees, got $s"))
    return Tuple(Float64.(v))
end

"Run `curl` with `args`, writing to `out`; `true` when curl succeeded."
_curl(args, out) = success(pipeline(`curl $args`; stdout = out))

# ---------------------------------------------------------------------------
# fetch-osm
# ---------------------------------------------------------------------------

"""
    overpass_query(box; relations = false, timeout = 60) -> String

The Overpass QL query of the urban case's sources in `box = (S, W, N, E)`
(degrees): buildings, parks, green land use (grass, forest, meadow,
recreation ground) and water, as ways (and, with `relations`, as
multipolygon relations, which the parks, rivers and courtyard blocks of a
large box often are), with their nodes. Highways are not queried: the
ground class of the classifier (`InputDrivers.osm_class`) is street.
"""
function overpass_query(box; relations::Bool = false, timeout::Integer = 60)
    b = join(box, ",")
    if !relations
        return "[out:xml][timeout:$timeout];\n(\n" *
               "  way[\"building\"]($b);\n  way[\"leisure\"=\"park\"]($b);\n" *
               "  way[\"landuse\"~\"grass|forest|meadow|recreation_ground\"]($b);\n" *
               "  way[\"natural\"=\"water\"]($b);\n);\n(._;>;);\nout body;\n"
    end
    sel = ("[\"building\"]", "[\"leisure\"=\"park\"]",
           "[\"landuse\"~\"^(grass|forest|meadow|recreation_ground)\$\"]",
           "[\"natural\"=\"water\"]", "[\"waterway\"=\"riverbank\"]")
    body = join(("  $(t)$(s)($b);" for s in sel for t in ("way", "relation")), "\n")
    return "[out:xml][timeout:$timeout];\n(\n$body\n);\n(._;>;);\nout body;\n"
end

const OVERPASS_ENDPOINTS = ("https://overpass-api.de/api/interpreter",
                            "https://overpass.kumi.systems/api/interpreter")

"""
    fetch_overpass(query, out; attempts = 3, max_time = 900) -> out

Download the answer of the Overpass `query` to `out` (`curl`, the public
endpoints in turn, `attempts` rounds); an error when no complete OSM
document arrives.
"""
function fetch_overpass(query::AbstractString, out::AbstractString; attempts::Integer = 3, max_time::Integer = 900)
    mkpath(dirname(abspath(out)))
    for attempt in 1:attempts, url in OVERPASS_ENDPOINTS
        @info "Overpass $url (attempt $attempt)"
        ok = _curl(`-sS --max-time $max_time -A "Aeolus.jl OSM ingest" $url --data-urlencode $("data=" * query)`, out)
        if ok && isfile(out)
            text = read(out, String)
            occursin("<osm", first(text, 400)) && endswith(rstrip(text), "</osm>") && return out
        end
        attempt == attempts || sleep(20)
    end
    error("the Overpass download failed; retry later")
end

"""
`aeolus preprocess fetch-osm --box S,W,N,E --out file.osm`: download the
OpenStreetMap buildings, parks, green land use and water of the box
(degrees) as OSM XML, the `[preprocess] osm_path` of an urban run. A new
file makes the static drivers made from the old one stale: re-run
`aeolus preprocess static` for the configs that use it.
"""
function _fetch_osm(args)
    _, opts, _ = _pp_args(args, ("box", "out"), ())
    (haskey(opts, "box") && haskey(opts, "out")) || throw(ArgumentError("fetch-osm: give --box S,W,N,E and --out file.osm"))
    out = abspath(opts["out"])
    fetch_overpass(overpass_query(_pp_box(opts["box"])), out; attempts = 1, max_time = 90)
    text = read(out, String)
    println("wrote ", out, @sprintf(" (%.1f kB, %d ways, %d building tags)", filesize(out) / 1024,
                                    count("<way ", text), count("k=\"building\"", text)))
    return 0
end

# ---------------------------------------------------------------------------
# fetch-landuse
# ---------------------------------------------------------------------------

"Counts of the height information of the building ways and relations of an OSM extract."
function height_provenance(text::AbstractString)
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

"""
    write_parent_landuse(csv, raw, cc::CaseConfig; extent = nest parent's, refinement = nest parent's,
                         margin = 100.0, box = nothing, fetched = today) -> csv

The nest parent's land-use table (#249) from the OSM extract `raw`: the
`extent` square (m) centred on the case's anchor `cc.lat0/lon0`,
classified with `InputDrivers.load_city_osm` / `osm_class`, rasterised
with `surface_class_map` (building-adjacent cells tagged wall) on the
child's cells (`cc.extent[1] / cc.size[1]`), heights per cell from
[`osm_height_map`](@ref), then coarse-grained by `refinement` to the
parent's cells: per parent cell the class counts of its child cells and
the building statistics λ_p, λ_f (windward facade area per ground area,
mean over the four cardinal directions), mean and maximum height,
with the provenance in `#` lines. What [`read_parent_landuse`](@ref)
reads.
"""
function write_parent_landuse(csv::AbstractString, raw::AbstractString, cc;
                              extent = nothing, refinement = nothing, margin = 100.0, box = nothing,
                              fetched = string(Dates.today()))
    pc = nest_parent_config(cc)
    L = extent === nothing ? pc.extent[1] : Float64(extent)
    R = refinement === nothing ? pc.refinement[1] : Int(refinement)
    lat0, lon0 = cc.lat0, cc.lon0
    dxc = cc.extent[1] / cc.size[1]
    NF = round(Int, L / dxc)
    NF % R == 0 || throw(ArgumentError("the parent box ($L m) is not a whole number of parent cells"))
    NP = NF ÷ R
    text = read(raw, String)
    osm_base = something(match(r"osm_base=\"([^\"]+)\"", text), (captures = ["unknown"],)).captures[1]
    hp = height_provenance(text)
    nrel = count("<relation", text)
    dz = cc.extent[3] / cc.size[3]
    mesh = RegularGrid(; size = (NF, NF, 1), extent = (L, L, dz))
    city = load_city_osm(raw; lat0 = lat0, lon0 = lon0, x_shift = L / 2, y_shift = L / 2,
                         min_height = cc.min_height)
    sfc = surface_class_map(city, mesh; tag_walls = true)[2]
    hf = osm_height_map(city, mesh, 1)
    h = [sfc[I] == SURFACE_ROOF ? hf[I] : 0.0 for I in CartesianIndices(sfc)]
    hpad(i, j) = h[clamp(i, 1, NF), clamp(j, 1, NF)]
    face = zeros(NF, NF)
    for j in 1:NF, i in 1:NF
        face[i, j] = (max(h[i, j] - hpad(i - 1, j), 0.0) + max(h[i, j] - hpad(i + 1, j), 0.0) +
                      max(h[i, j] - hpad(i, j - 1), 0.0) + max(h[i, j] - hpad(i, j + 1), 0.0)) * dxc / 4
    end
    Ap = (R * dxc)^2
    qbox = box === nothing ? "" : "# query_box_deg (S,W,N,E) = $(join(box, ","))  (parent box + $(margin) m); ways and type=multipolygon relations ($nrel relations in the extract)\n"
    mkpath(dirname(abspath(csv)))
    open(csv, "w") do io
        println(io, "# Parent land use (#249): per parent cell, class counts of its child cells and building statistics.")
        println(io, "# source: OpenStreetMap via the Overpass API, osm_base = $osm_base, fetched $fetched by `aeolus preprocess fetch-landuse`")
        println(io, "# licence: (c) OpenStreetMap contributors, ODbL 1.0, https://www.openstreetmap.org/copyright")
        print(io, qbox)
        println(io, "# anchor lat0 = $lat0, lon0 = $lon0 (CaseConfig); projection: equirectangular about the anchor (latlon_to_meters)")
        println(io, "# parent grid: nx = $NP, ny = $NP, dx = $(R * dxc) m, x0 = y0 = $(-L / 2) m relative to the anchor; refinement r = $R child cells of $(dxc) m per axis")
        println(io, "# classifier: osm_class (load_city_osm), min_height = $(cc.min_height) m, untagged height 12 m, building:levels x 3 m; rasteriser: surface_class_map (tag_walls)")
        println(io, "# building height information (ways + relations): height tag $(hp[:height]), building:levels only $(hp[:levels]), neither (12 m default) $(hp[:none])")
        println(io, "# classes: 1 street, 2 roof, 3 park, 4 water, 5 wall (building-adjacent street); n_c = child cells of class c (out of $(R^2))")
        println(io, "# lambda_p = roof fraction; lambda_f = windward facade area / ground area, mean over the four cardinal wind directions, from the child-resolution height map")
        println(io, "# h_mean = roof-area-weighted mean building height (m, 0 without buildings); h_max = tallest building (m)")
        println(io, "I,J,n1,n2,n3,n4,n5,lambda_p,lambda_f,h_mean,h_max")
        for J in 1:NP, I in 1:NP
            is, js = (I - 1) * R + 1:I * R, (J - 1) * R + 1:J * R
            n = [count(==(UInt8(c)), view(sfc, is, js)) for c in 1:5]
            roof = [h[i, j] for i in is, j in js if sfc[i, j] == SURFACE_ROOF]
            λp = n[2] / R^2
            λf = sum(view(face, is, js)) / Ap
            hm = isempty(roof) ? 0.0 : sum(roof) / length(roof)
            hx = isempty(roof) ? 0.0 : maximum(roof)
            @printf(io, "%d,%d,%d,%d,%d,%d,%d,%.4f,%.5f,%.2f,%.2f\n", I, J, n..., λp, λf, hm, hx)
        end
    end
    return csv
end

"""
`aeolus preprocess fetch-landuse <config.toml> [--out file.csv] [--raw file.osm] [--cached]
[--set key=value]...`: the nest parent's land-use table of an urban run
([`write_parent_landuse`](@ref)): the OSM extract of the parent box
(`[physics.options.nesting.parent] extent` around `[geometry] lat0/lon0`,
plus 100 m) is downloaded to `--raw` (default `raw/<name>.osm` next to the
table; `--cached` reuses it), classified, rasterised on the child's cells
and coarse-grained by the parent's refinement into `--out` (default
`[preprocess] landuse`). Re-run `aeolus preprocess static` afterwards.
"""
function _fetch_landuse(args)
    pos, sets, opts, fl = Config._parse_args(args, ("set", "out", "raw"), ("cached",))
    path = Config._one_positional(pos, "config file")
    cc = CaseConfig(path; overrides = sets)
    csv = haskey(opts, "out") ? abspath(opts["out"]) : cc.landuse
    csv === nothing && throw(ArgumentError("[preprocess] landuse is \"none\": pass --out file.csv"))
    raw = haskey(opts, "raw") ? abspath(opts["raw"]) :
          joinpath(dirname(csv), "raw", splitext(basename(csv))[1] * ".osm")
    pc = nest_parent_config(cc)
    margin = 100.0
    h = pc.extent[1] / 2 + margin
    R_E = Aeolus.InputDrivers._EARTH_R
    dlat, dlon = rad2deg(h / R_E), rad2deg(h / (R_E * cosd(cc.lat0)))
    box = round.((cc.lat0 - dlat, cc.lon0 - dlon, cc.lat0 + dlat, cc.lon0 + dlon); digits = 4)
    ("cached" in fl && isfile(raw)) || fetch_overpass(overpass_query(box; relations = true, timeout = 300), raw)
    write_parent_landuse(csv, raw, cc; margin = margin, box = box)
    println("wrote ", csv)
    return 0
end

# ---------------------------------------------------------------------------
# fetch-sounding
# ---------------------------------------------------------------------------

_days_from_civil(y, m, d) = Dates.value(Dates.Date(y, m, d) - Dates.Date(1970, 1, 1))

_igra_field(line, a, b) = strip(line[a:min(b, length(line))])
_igra_int(s) = parse(Int, s)

"One data record of an IGRA2 sounding (igra2-data-format.txt)."
function _igra_level(line)
    f(a, b) = _igra_field(line, a, b)
    return (; lvl = string(line[1], line[2]), press = _igra_int(f(10, 15)), pflag = f(16, 16),
            gph = _igra_int(f(17, 21)), zflag = f(22, 22), temp = _igra_int(f(23, 27)), tflag = f(28, 28),
            dpdp = _igra_int(f(35, 39)), wdir = _igra_int(f(41, 45)), wspd = _igra_int(f(47, 51)))
end

"""
    igra_sounding_csv(txt, out; launches, epoch, station, elevation, p_min = 10000,
                      source = "", notes = String[]) -> (levels, dropped)

Write the IGRA2 launches `launches` (`(y, m, d, hour UTC)` nominal
times) of the station record `txt` (the unzipped period-of-record file)
as the sounding CSV that `LargeScaleForcing.read_sounding` reads: one row
per level (release time in seconds from `epoch = (y, m, d)` 00 UTC,
pressure, geopotential height, T, Td = T − dew-point depression, wind),
levels above `p_min` Pa and below the surface left out, the `#!`
header lines `epoch` and `station_elevation_m = elevation` (m; the
launch site's height, which the IGRA station list may give differently),
the provenance in `#` lines (`notes` appended). Durre et al. (2018),
J. Atmos. Oceanic Technol. 35, 1753-1770.
"""
function igra_sounding_csv(txt::AbstractString, out::AbstractString; launches, epoch, station::AbstractString,
                           elevation::Real, p_min::Integer = 10000, source::AbstractString = "",
                           notes = String[])
    want = Set(launches)
    rows = String[]
    dropped = Dict{String,Int}()
    drop!(why) = (dropped[why] = get(dropped, why, 0) + 1)
    found = Tuple{Int,Int,Int,Int}[]
    cur = nothing
    t_epoch = _days_from_civil(epoch...) * 86400
    for line in eachline(txt)
        if startswith(line, "#")
            y = _igra_int(_igra_field(line, 14, 17)); m = _igra_int(_igra_field(line, 19, 20))
            d = _igra_int(_igra_field(line, 22, 23)); h = _igra_int(_igra_field(line, 25, 26))
            rel = _igra_int(_igra_field(line, 28, 31))
            if (y, m, d, h) in want
                rh, rm = rel == 9999 ? (h, 0) : (div(rel, 100), rem(rel, 100))
                tday = _days_from_civil(y, m, d) * 86400
                trel = tday + 3600rh + 60rm
                trel > tday + 3600h + 3 * 3600 && (trel -= 86400)   # released the evening before
                cur = (; tag = @sprintf("%04d-%02d-%02dT%02dZ", y, m, d, h),
                         rel = @sprintf("%04d", rel), t = trel - t_epoch, psfc = Ref(typemax(Int)))
                push!(found, (y, m, d, h))
            else
                cur = nothing
            end
            continue
        end
        cur === nothing && continue
        L = _igra_level(line)
        L.press == -9999 && (drop!("no pressure"); continue)
        L.press < p_min && continue
        L.lvl[2] == '1' && (cur.psfc[] = L.press)
        L.press > cur.psfc[] && (drop!("below the surface (p > p_sfc)"); continue)
        T  = L.temp in (-9999, -8888) ? NaN : L.temp / 10 + 273.15
        Td = (isnan(T) || L.dpdp in (-9999, -8888)) ? NaN : T - L.dpdp / 10
        ws = L.wspd in (-9999, -8888) ? NaN : L.wspd / 10
        wd = (isnan(ws) || L.wdir in (-9999, -8888)) ? NaN : float(L.wdir)
        L.temp == -8888 && drop!("temperature removed by IGRA QC (-8888)")
        (isnan(T) && isnan(ws)) && (drop!("neither temperature nor wind"); continue)
        gph = L.gph in (-9999, -8888) ? NaN : float(L.gph)
        push!(rows, @sprintf("%s,%s,%d,%s,%d,%s,%s,%s,%s,%s,%s,%s,%s",
                             cur.tag, cur.rel, cur.t, L.lvl, L.press, L.pflag == "" ? "-" : L.pflag,
                             isnan(gph) ? "NaN" : @sprintf("%.0f", gph), L.zflag == "" ? "-" : L.zflag,
                             isnan(T) ? "NaN" : @sprintf("%.2f", T), L.tflag == "" ? "-" : L.tflag,
                             isnan(Td) ? "NaN" : @sprintf("%.2f", Td),
                             isnan(wd) ? "NaN" : @sprintf("%.0f", wd),
                             isnan(ws) ? "NaN" : @sprintf("%.1f", ws)))
    end
    Set(found) == want || error("launches missing from the IGRA record: $(setdiff(want, Set(found)))")
    mkpath(dirname(abspath(out)))
    open(out, "w") do io
        println(io, "# IGRA2 station $station radiosonde launches $(join((@sprintf("%04d-%02d-%02dT%02dZ", l...) for l in sort!(collect(want))), ", ")).")
        println(io, "# Source: NOAA NCEI, Integrated Global Radiosonde Archive v2 (IGRA2), period-of-record file $(station)-data.txt.zip,")
        isempty(source) || println(io, "#   $source")
        println(io, "#   downloaded $(Dates.today()) by `aeolus preprocess fetch-sounding`. Public domain (NOAA).")
        println(io, "# Reference: Durre, I., Yin, X., Vose, R. S., Applequist, S. & Arnfield, J. (2018), J. Atmos. Oceanic Technol. 35, 1753-1770.")
        println(io, "# Columns: launch = nominal synoptic time (UTC); release_utc = actual release HHMM (UTC); t_s = release time in seconds from the")
        println(io, "#   epoch below; lvltyp = IGRA level type; p_Pa; pflag/zflag/tflag = IGRA QC flags; gph_m = reported geopotential height")
        println(io, "#   above sea level; T_K; Td_K = T - dew-point depression; wdir_deg (from), wspd_ms. NaN = missing.")
        println(io, "#! epoch = $(epoch[1])-$(lpad(epoch[2], 2, '0'))-$(lpad(epoch[3], 2, '0'))T00:00Z")
        println(io, "#! station_elevation_m = $(Float64(elevation))")
        println(io, "# Levels dropped: $(isempty(dropped) ? "none" : join(["$(v) ($(k))" for (k, v) in sort!(collect(dropped))], "; ")).")
        println(io, "#   Levels above $(p_min ÷ 100) hPa are not kept. Wind-only levels (T = NaN) are kept for the wind profile.")
        foreach(n -> println(io, "# ", n), notes)
        println(io, "launch,release_utc,t_s,lvltyp,p_Pa,pflag,gph_m,zflag,T_K,tflag,Td_K,wdir_deg,wspd_ms")
        foreach(r -> println(io, r), rows)
    end
    return (length(rows), dropped)
end

function _pp_time(s)
    m = match(r"^(\d{4})-(\d{2})-(\d{2})T(\d{2})Z?$", s)
    m === nothing && throw(ArgumentError("expected a launch time YYYY-MM-DDTHH (UTC), got $s"))
    return Tuple(parse.(Int, m.captures))
end

"""
`aeolus preprocess fetch-sounding --station ID --launches T1,T2,… --elevation m --out file.csv
[--epoch YYYY-MM-DD]`: download the IGRA2 period-of-record file of the
station (NOAA NCEI; `curl`, then `unzip` or `python3 -m zipfile`) and
write the launches `Ti` (`YYYY-MM-DDTHH`, nominal UTC) as the sounding
CSV of `[initial] sounding` ([`igra_sounding_csv`](@ref)); `--elevation`:
the launch site's height (m), `--epoch`: t = 0 (default: the first
launch's day, 00 UTC).
"""
function _fetch_sounding(args)
    _, opts, _ = _pp_args(args, ("station", "launches", "elevation", "out", "epoch"), ())
    for k in ("station", "launches", "elevation", "out")
        haskey(opts, k) || throw(ArgumentError("fetch-sounding: give --$k"))
    end
    station = opts["station"]
    launches = [_pp_time(String(s)) for s in split(opts["launches"], ',')]
    elev = tryparse(Float64, opts["elevation"])
    elev === nothing && throw(ArgumentError("--elevation: expected metres, got $(opts["elevation"])"))
    epoch = haskey(opts, "epoch") ? _pp_time(opts["epoch"] * "T00")[1:3] : minimum(launches)[1:3]
    url = "https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/access/data-por/$(station)-data.txt.zip"
    out = abspath(opts["out"])
    mktempdir() do dir
        zip = joinpath(dir, "$(station)-data.txt.zip")
        txt = joinpath(dir, "$(station)-data.txt")
        @info "downloading $url"
        run(`curl -sS --fail --max-time 900 -o $zip $url`)
        if Sys.which("unzip") !== nothing
            run(pipeline(`unzip -p $zip`; stdout = txt))
        elseif Sys.which("python3") !== nothing
            run(`python3 -m zipfile -e $zip $dir`)
        else
            error("neither unzip nor python3 is available to extract $zip")
        end
        n, dropped = igra_sounding_csv(txt, out; launches = launches, epoch = epoch, station = station,
                                       elevation = elev, source = url)
        println("wrote ", out, ": ", n, " levels from ", length(launches), " launches")
    end
    return 0
end

# ---------------------------------------------------------------------------
# osm-sample
# ---------------------------------------------------------------------------

"""
    synthetic_osm(io; lat0, lon0, seed = 1)

Write a synthetic OpenStreetMap XML city of 1 km × 1 km about
`lat0/lon0`: perimeter blocks (four buildings of 45 m around a courtyard
per 130 m block, 30 m streets; heights 18 m + 6 m·N(0,1) + 0.02·(x+500 m),
clamped to 10–60 m, from `Random.Xoshiro(seed)`), three office towers
(55, 75, 90 m), a park with a pond, a green space and a water body.
Real OSM XML: the same reader (`InputDrivers.load_city_osm`) takes it.
"""
function synthetic_osm(io::Base.IO; lat0::Real, lon0::Real, seed::Integer = 1)
    R_E = 6_371_000.0
    coslat = cos(deg2rad(lat0))
    node_id = Ref(0); way_id = Ref(0)
    function pushnode!(dx, dy)
        node_id[] += 1
        @printf(io, "  <node id=\"%d\" lat=\"%.7f\" lon=\"%.7f\"/>\n", node_id[],
                lat0 + rad2deg(dy / R_E), lon0 + rad2deg(dx / (R_E * coslat)))
        return node_id[]
    end
    function pushway!(ids, tags)
        way_id[] += 1
        println(io, "  <way id=\"$(way_id[])\">")
        foreach(id -> println(io, "    <nd ref=\"$id\"/>"), ids)
        foreach(t -> println(io, t), tags)
        println(io, "  </way>")
    end
    function pushbuilding!(cx, cy, w, h, H; class = "yes")
        xs = (cx - w / 2, cx + w / 2, cx + w / 2, cx - w / 2, cx - w / 2)
        ys = (cy - h / 2, cy - h / 2, cy + h / 2, cy + h / 2, cy - h / 2)
        ids = [pushnode!(xs[i], ys[i]) for i in 1:5]
        pushway!(ids, ("    <tag k=\"building\" v=\"$class\"/>", @sprintf("    <tag k=\"height\" v=\"%.1f\"/>", H)))
    end
    function pushpatch!(verts, k, v)
        ids = [pushnode!(p[1], p[2]) for p in verts]
        push!(ids, ids[1])
        pushway!(ids, ("    <tag k=\"$k\" v=\"$v\"/>",))
    end
    println(io, """<?xml version="1.0" encoding="UTF-8"?>""")
    println(io, """<osm version="0.6" generator="Aeolus.urban_sample">""")
    rng = Random.Xoshiro(seed)
    block, step, bw = 130.0, 160.0, 45.0
    for cy in -400:160.0:400, cx in -400:160.0:400
        (abs(cx) < step / 2 && abs(cy - step) < step / 2) && continue      # the park block
        (cy > 300 && abs(cx - step) < step / 2) && continue                 # the water block
        for (dx, dy) in ((-block / 2 + bw / 2, 0), (block / 2 - bw / 2, 0),
                         (0, -block / 2 + bw / 2), (0, block / 2 - bw / 2))
            H = clamp(18.0 + 6 * randn(rng) + 0.02 * (cx + 500), 10.0, 60.0)
            pushbuilding!(cx + dx, cy + dy, bw, bw, H; class = "residential")
        end
    end
    for (cx, cy, w, h, H) in ((-320.0, -320.0, 60.0, 60.0, 55.0), (0.0, -320.0, 80.0, 40.0, 75.0),
                              (300.0, 100.0, 70.0, 50.0, 90.0))
        pushbuilding!(cx, cy, w, h, H; class = "commercial")
    end
    pushpatch!([(-60.0, 100.0), (60.0, 100.0), (60.0, 220.0), (-60.0, 220.0)], "leisure", "park")
    pushpatch!([(-20.0, 140.0), (20.0, 140.0), (20.0, 180.0), (-20.0, 180.0)], "natural", "water")
    pushpatch!([(220.0, -420.0), (400.0, -420.0), (400.0, -260.0), (220.0, -260.0)], "landuse", "grass")
    pushpatch!([(240.0, 340.0), (400.0, 340.0), (400.0, 500.0), (240.0, 500.0)], "natural", "water")
    println(io, "</osm>")
    return (nodes = node_id[], ways = way_id[])
end

"""
`aeolus preprocess osm-sample --lat0 deg --lon0 deg --out file.osm [--seed n]`: write the
synthetic city of [`synthetic_osm`](@ref) about the anchor.
"""
function _osm_sample(args)
    _, opts, _ = _pp_args(args, ("lat0", "lon0", "out", "seed"), ())
    for k in ("lat0", "lon0", "out")
        haskey(opts, k) || throw(ArgumentError("osm-sample: give --$k"))
    end
    function num(k)
        v = tryparse(Float64, opts[k])
        v === nothing && throw(ArgumentError("--$k: expected a number, got $(opts[k])"))
        return v
    end
    lat0, lon0 = num("lat0"), num("lon0")
    seed = tryparse(Int, get(opts, "seed", "1"))
    seed === nothing && throw(ArgumentError("--seed: expected an integer, got $(opts["seed"])"))
    out = abspath(opts["out"])
    mkpath(dirname(out))
    n = open(io -> synthetic_osm(io; lat0 = lat0, lon0 = lon0, seed = seed), out, "w")
    println("wrote ", out, " (", n.nodes, " nodes, ", n.ways, " ways)")
    return 0
end

register_preprocessor!("fetch-osm", _fetch_osm,
                       "fetch-osm --box S,W,N,E --out file.osm  download the OpenStreetMap buildings, parks and water of a box")
register_preprocessor!("fetch-landuse", _fetch_landuse,
                       "fetch-landuse <config.toml> [--out file.csv] [--raw file.osm] [--cached] [--set key=value]...  " *
                       "the urban nest parent's land-use table")
register_preprocessor!("fetch-sounding", _fetch_sounding,
                       "fetch-sounding --station ID --launches YYYY-MM-DDTHH,... --elevation m --out file.csv " *
                       "[--epoch YYYY-MM-DD]  IGRA2 radiosonde launches as a sounding CSV")
register_preprocessor!("osm-sample", _osm_sample,
                       "osm-sample --lat0 deg --lon0 deg --out file.osm [--seed n]  a synthetic OSM city")
