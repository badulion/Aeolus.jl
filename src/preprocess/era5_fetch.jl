# `aeolus preprocess fetch-era5` (EX-e, #442): download ERA5 from the
# Copernicus Climate Data Store for the ERA5 preprocessor of the dynamic
# driver (CFG-h, #363). Formerly the script examples/data/fetch_era5.jl.
#
# The CDS "retrieve v1" REST API (OGC API Processes) through the
# `Downloads` stdlib only: submit the request, poll the job, fetch the
# result. Each file gets the request (JSON), the licence statement and the
# attribution as global attributes. The tests never reach the network:
# they check the requests against the committed Lindenberg sample.
#
# Licence: Copernicus Climate Change Service (C3S) data, "Licence to use
# Copernicus products": free use and redistribution with attribution.
# Attribution: Hersbach, H. et al. (2023): ERA5 hourly data on pressure
# levels from 1940 to present. C3S Climate Data Store (CDS),
# doi:10.24381/cds.bd0915c6; ... on single levels, doi:10.24381/cds.adbb2d47.

import Downloads

"The CDS API root used when `~/.cdsapirc` names no `url:`."
const CDS_API = "https://cds.climate.copernicus.eu/api"

const CDS_LICENCE = "Generated using Copernicus Climate Change Service information 2019. Neither the European " *
                    "Commission nor ECMWF is responsible for any use that may be made of the Copernicus " *
                    "information or data it contains. Licence to use Copernicus products " *
                    "(https://cds.climate.copernicus.eu/licences): free use and redistribution with attribution."

const CDS_ATTRIBUTION = Dict(
    "reanalysis-era5-pressure-levels" =>
        "Hersbach, H. et al. (2023): ERA5 hourly data on pressure levels from 1940 to present. " *
        "Copernicus Climate Change Service (C3S) Climate Data Store (CDS), doi:10.24381/cds.bd0915c6",
    "reanalysis-era5-single-levels" =>
        "Hersbach, H. et al. (2023): ERA5 hourly data on single levels from 1940 to present. " *
        "Copernicus Climate Change Service (C3S) Climate Data Store (CDS), doi:10.24381/cds.adbb2d47")

"The pressure levels (hPa) fetched: 1000–500 hPa, the ERA5 levels between."
const ERA5_PL_LEVELS = ["500", "550", "600", "650", "700", "750", "775", "800", "825", "850", "875", "900",
                        "925", "950", "975", "1000"]

"""
    cds_credentials(path = "~/.cdsapirc") -> (url, key)

The CDS API root and personal access token from a `.cdsapirc` file
(`url: …` and `key: …` lines; `url` defaults to [`CDS_API`](@ref)).
"""
function cds_credentials(path = joinpath(homedir(), ".cdsapirc"))
    isfile(path) || throw(ArgumentError("no $path: create a CDS account (https://cds.climate.copernicus.eu, " *
                                        "free) and put its `url:` and `key:` lines in it"))
    d = Dict{String,String}()
    for l in eachline(path)
        m = match(r"^\s*(\w+)\s*:\s*(.+?)\s*$", l)
        m === nothing || (d[m[1]] = m[2])
    end
    haskey(d, "key") || throw(ArgumentError("$path has no `key:` line"))
    return get(d, "url", CDS_API), d["key"]
end

# minimal JSON for a request: strings, numbers, vectors and dicts of them
cds_json(x::AbstractString) = "\"" * escape_string(x) * "\""
cds_json(x::Real) = string(x)
cds_json(x::AbstractVector) = "[" * join(cds_json.(x), ",") * "]"
cds_json(d::AbstractDict) = "{" * join(("$(cds_json(String(k))):$(cds_json(v))" for (k, v) in d), ",") * "}"

"""
    era5_requests(lat, lon, date, h0, h1; half = 1.0) -> Vector{(dataset, request, suffix)}

The two CDS requests of an ERA5 sample: the pressure levels
([`ERA5_PL_LEVELS`](@ref); T, q, u, v, ω, Φ) and the single levels
(surface pressure, skin and 2 m temperature, surface Φ), hours `h0:h1`
UTC of `date`, in the box `lat ± half`, `lon ± half` degrees. `suffix`
is `"pl"` or `"sfc"`, the file name ending the ERA5 preprocessor
expects (`--from-era5 <name>_pl.nc <name>_sfc.nc`).
"""
function era5_requests(lat::Real, lon::Real, date::Dates.Date, h0::Integer, h1::Integer; half::Real = 1.0)
    (0 <= h0 <= h1 <= 23) || throw(ArgumentError("hours: expected 0 ≤ h0 ≤ h1 ≤ 23, got $h0-$h1"))
    half > 0 || throw(ArgumentError("box: expected a half-width > 0 degrees, got $half"))
    area = [lat + half, lon - half, lat - half, lon + half]             # N, W, S, E
    common = Dict{String,Any}("product_type" => ["reanalysis"], "year" => [Dates.format(date, "yyyy")],
                              "month" => [Dates.format(date, "mm")], "day" => [Dates.format(date, "dd")],
                              "time" => [@sprintf("%02d:00", h) for h in h0:h1], "area" => area,
                              "data_format" => "netcdf", "download_format" => "unarchived")
    pl = merge(common, Dict{String,Any}("variable" => ["temperature", "specific_humidity", "u_component_of_wind",
                                                        "v_component_of_wind", "vertical_velocity", "geopotential"],
                                        "pressure_level" => ERA5_PL_LEVELS))
    sfc = merge(common, Dict{String,Any}("variable" => ["surface_pressure", "skin_temperature",
                                                         "2m_temperature", "geopotential"]))
    return [("reanalysis-era5-pressure-levels", pl, "pl"), ("reanalysis-era5-single-levels", sfc, "sfc")]
end

function _cds_http(method, url, key; body = nothing)
    out = IOBuffer()
    hdr = ["PRIVATE-TOKEN" => key, "Accept" => "application/json", "Content-Type" => "application/json"]
    r = Downloads.request(url; method, headers = hdr, output = out,
                          input = body === nothing ? nothing : IOBuffer(body), throw = false)
    s = String(take!(out))
    r isa Downloads.Response && r.status < 300 ||
        error("CDS $method $url failed: $(r isa Downloads.Response ? r.status : r)\n$s")
    return s
end

_cds_field(s, key) = (m = match(Regex("\"$key\"\\s*:\\s*\"([^\"]*)\""), s); m === nothing ? nothing : String(m[1]))

"""
    cds_retrieve(dataset, request, out; credentials = cds_credentials()) -> out

Submit `request` (a Dict) for `dataset` to the CDS, wait for the job
(polling every 10 s) and download its result to `out`.
"""
function cds_retrieve(dataset, request, out; credentials = cds_credentials())
    url, key = credentials
    s = _cds_http("POST", "$url/retrieve/v1/processes/$dataset/execution", key;
                  body = cds_json(Dict("inputs" => request)))
    job = _cds_field(s, "jobID")
    job === nothing && error("CDS: no jobID in the reply\n$s")
    println(dataset, ": job ", job, " submitted")
    status = ""
    while true
        s = _cds_http("GET", "$url/retrieve/v1/jobs/$job", key)
        status = something(_cds_field(s, "status"), "?")
        status in ("successful", "failed", "rejected", "dismissed") && break
        println("  ", Dates.format(now(), "HH:MM:SS"), " ", status)
        sleep(10)
    end
    status == "successful" || error("CDS job $job: $status\n$s")
    s = _cds_http("GET", "$url/retrieve/v1/jobs/$job/results", key)
    href = _cds_field(s, "href")
    href === nothing && error("CDS: no result href\n$s")
    Downloads.download(href, out)
    return out
end

"""
    cds_stamp!(path, dataset, request) -> path

Write the provenance of a downloaded ERA5 file into its global
attributes: `cds_dataset`, `cds_request` (JSON), `licence`,
`attribution` and `fetched_by`.
"""
function cds_stamp!(path, dataset, request)
    NCDatasets.NCDataset(path, "a") do ds
        ds.attrib["cds_dataset"] = dataset
        ds.attrib["cds_request"] = cds_json(request)
        ds.attrib["licence"] = CDS_LICENCE
        ds.attrib["attribution"] = CDS_ATTRIBUTION[dataset]
        ds.attrib["fetched_by"] = "aeolus preprocess fetch-era5 on $(Dates.format(now(UTC), "yyyy-mm-ddTHH:MMZ"))"
    end
    return path
end

"""
    fetch_era5(lat, lon, date, h0, h1, outdir; half = 1.0, name = "era5") -> [pl, sfc]

Download the two files of [`era5_requests`](@ref) to
`outdir/<name>_pl.nc` and `outdir/<name>_sfc.nc` and stamp their
provenance ([`cds_stamp!`](@ref)). Needs a CDS account and its token in
`~/.cdsapirc` ([`cds_credentials`](@ref)), and the ERA5 licence accepted
once on the dataset's web page. The committed Lindenberg sample
(`examples/data/era5_lindenberg_20190629_{pl,sfc}.nc`) is
`fetch_era5(52.21, 14.12, Date(2019, 6, 29), 0, 12, "examples/data"; name = "era5_lindenberg_20190629")`.
"""
function fetch_era5(lat, lon, date::Dates.Date, h0, h1, outdir; half = 1.0, name = "era5",
                    credentials = cds_credentials())
    reqs = era5_requests(lat, lon, date, h0, h1; half = half)
    mkpath(outdir)
    out = String[]
    for (ds, req, suf) in reqs
        p = joinpath(outdir, "$(name)_$(suf).nc")
        cds_retrieve(ds, req, p; credentials = credentials)
        cds_stamp!(p, ds, req)
        println("wrote ", p, @sprintf(" (%.1f kB)", filesize(p) / 1024))
        push!(out, p)
    end
    return out
end

"""
    era5_fetch_args(args) -> (lat, lon, date, h0, h1, outdir, half, name)

The arguments of `aeolus preprocess fetch-era5 --site lat,lon --date
YYYY-MM-DD --hours h0-h1 --out dir [--box deg] [--name prefix]`, checked
(an `ArgumentError`, exit code 2, names the bad option).
"""
function era5_fetch_args(args)
    pos, sets, opts, _ = Config._parse_args(args, ("site", "date", "hours", "out", "box", "name"), ())
    isempty(sets) || throw(ArgumentError("fetch-era5 takes no --set"))
    isempty(pos) || throw(ArgumentError("fetch-era5 takes no positional arguments, got $(join(pos, " "))"))
    for k in ("site", "date", "hours", "out")
        haskey(opts, k) || throw(ArgumentError("fetch-era5: give --$k"))
    end
    site = tryparse.(Float64, split(opts["site"], ','))
    (length(site) == 2 && all(!isnothing, site)) ||
        throw(ArgumentError("--site: expected lat,lon in degrees, got $(opts["site"])"))
    date = tryparse(Dates.Date, opts["date"])
    date === nothing && throw(ArgumentError("--date: expected YYYY-MM-DD, got $(opts["date"])"))
    hs = tryparse.(Int, split(opts["hours"], '-'))
    (length(hs) == 2 && all(!isnothing, hs) && 0 <= hs[1] <= hs[2] <= 23) ||
        throw(ArgumentError("--hours: expected h0-h1 (UTC, 0 ≤ h0 ≤ h1 ≤ 23), got $(opts["hours"])"))
    half = tryparse(Float64, get(opts, "box", "1.0"))
    (half === nothing || !(half > 0)) && throw(ArgumentError("--box: expected a half-width > 0 degrees, got $(opts["box"])"))
    return (site[1], site[2], date, hs[1], hs[2], abspath(opts["out"]), half, get(opts, "name", "era5"))
end

function _fetch_era5_cli(args)
    lat, lon, date, h0, h1, out, half, name = era5_fetch_args(args)
    fetch_era5(lat, lon, date, h0, h1, out; half = half, name = name)
    return 0
end

Config.register_preprocessor!("fetch-era5", _fetch_era5_cli,
    "fetch-era5 --site lat,lon --date YYYY-MM-DD --hours h0-h1 --out dir [--box deg] [--name prefix]  " *
    "ERA5 pressure and single levels from the Copernicus CDS (~/.cdsapirc)")
