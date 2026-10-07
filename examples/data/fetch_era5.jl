# Download ERA5 from the Copernicus Climate Data Store (CDS) for the ERA5
# preprocessor of the dynamic driver (CFG-h, #363):
#
#   julia examples/data/fetch_era5.jl                     # the committed Lindenberg sample
#   julia examples/data/fetch_era5.jl <lat> <lon> <yyyy-mm-dd> <h0> <h1> <outdir> [half_box_deg]
#
# The first form re-creates `era5_lindenberg_20190629_pl.nc` and
# `era5_lindenberg_20190629_sfc.nc` next to this script: ERA5 around
# Lindenberg (52.21 N, 14.12 E), 2019-06-29 00-12 UTC hourly, pressure
# levels 1000-500 hPa, a ±1° box. The second form fetches the same
# variables for another site, day, hour range and box.
#
# Credentials: a CDS account (https://cds.climate.copernicus.eu, free) and
# its personal access token in `~/.cdsapirc`:
#
#   url: https://cds.climate.copernicus.eu/api
#   key: <your personal access token>
#
# and the ERA5 licence accepted once on the dataset's web page ("Licence
# to use Copernicus products"). Requests are queued by the CDS: a small
# request like the sample takes one to a few minutes.
#
# The script uses the CDS "retrieve v1" REST API (OGC API Processes) through
# Julia's `Downloads` stdlib only: submit the request, poll the job, fetch
# the result. It writes the request (JSON) and the licence statement into
# the file's global attributes (`cds_request`, `licence`, `attribution`).
# The tests never call it: they read the committed sample.
#
# Licence: Copernicus Climate Change Service (C3S) data, "Licence to use
# Copernicus products": free use and redistribution with attribution.
# Attribution: Hersbach, H. et al. (2023): ERA5 hourly data on pressure
# levels from 1940 to present. C3S Climate Data Store (CDS),
# doi:10.24381/cds.bd0915c6; ... on single levels, doi:10.24381/cds.adbb2d47.

using Downloads, Printf, Dates
import NCDatasets

const API = "https://cds.climate.copernicus.eu/api"
const LICENCE = "Generated using Copernicus Climate Change Service information 2019. Neither the European " *
                "Commission nor ECMWF is responsible for any use that may be made of the Copernicus " *
                "information or data it contains. Licence to use Copernicus products " *
                "(https://cds.climate.copernicus.eu/licences): free use and redistribution with attribution."
const ATTRIBUTION = Dict("reanalysis-era5-pressure-levels" =>
                             "Hersbach, H. et al. (2023): ERA5 hourly data on pressure levels from 1940 to present. " *
                             "Copernicus Climate Change Service (C3S) Climate Data Store (CDS), doi:10.24381/cds.bd0915c6",
                         "reanalysis-era5-single-levels" =>
                             "Hersbach, H. et al. (2023): ERA5 hourly data on single levels from 1940 to present. " *
                             "Copernicus Climate Change Service (C3S) Climate Data Store (CDS), doi:10.24381/cds.adbb2d47")

function cds_credentials(path = joinpath(homedir(), ".cdsapirc"))
    isfile(path) || error("no $path: create a CDS account and put `url:` and `key:` lines in it (see the header)")
    d = Dict{String,String}()
    for l in eachline(path)
        m = match(r"^\s*(\w+)\s*:\s*(.+?)\s*$", l)
        m === nothing || (d[m[1]] = m[2])
    end
    haskey(d, "key") || error("$path has no `key:` line")
    return get(d, "url", API), d["key"]
end

# minimal JSON for the request (strings, numbers, vectors of them)
_json(x::AbstractString) = "\"" * escape_string(x) * "\""
_json(x::Real) = string(x)
_json(x::AbstractVector) = "[" * join(_json.(x), ",") * "]"
_json(d::AbstractDict) = "{" * join(("$(_json(String(k))):$(_json(v))" for (k, v) in d), ",") * "}"

function _http(method, url, key; body = nothing)
    out = IOBuffer()
    hdr = ["PRIVATE-TOKEN" => key, "Accept" => "application/json", "Content-Type" => "application/json"]
    r = Downloads.request(url; method, headers = hdr, output = out,
                          input = body === nothing ? nothing : IOBuffer(body), throw = false)
    s = String(take!(out))
    r isa Downloads.Response && r.status < 300 || error("CDS $method $url failed: $(r isa Downloads.Response ? r.status : r)\n$s")
    return s
end

_field(s, key) = (m = match(Regex("\"$key\"\\s*:\\s*\"([^\"]*)\""), s); m === nothing ? nothing : String(m[1]))

"""
    cds_retrieve(dataset, request, out) -> out

Submit `request` (a Dict) for `dataset` to the CDS, wait for the job and
download its result to `out`.
"""
function cds_retrieve(dataset, request, out)
    url, key = cds_credentials()
    s = _http("POST", "$url/retrieve/v1/processes/$dataset/execution", key;
              body = _json(Dict("inputs" => request)))
    job = _field(s, "jobID")
    job === nothing && error("CDS: no jobID in the reply\n$s")
    @printf("%s: job %s submitted\n", dataset, job)
    status = ""
    while true
        s = _http("GET", "$url/retrieve/v1/jobs/$job", key)
        status = something(_field(s, "status"), "?")
        status in ("successful", "failed", "rejected", "dismissed") && break
        @printf("  %s %s\n", Dates.format(now(), "HH:MM:SS"), status)
        sleep(10)
    end
    status == "successful" || error("CDS job $job: $status\n$s")
    s = _http("GET", "$url/retrieve/v1/jobs/$job/results", key)
    href = _field(s, "href")
    href === nothing && error("CDS: no result href\n$s")
    Downloads.download(href, out)
    return out
end

function stamp!(path, dataset, request)
    NCDatasets.NCDataset(path, "a") do ds
        ds.attrib["cds_dataset"] = dataset
        ds.attrib["cds_request"] = _json(request)
        ds.attrib["licence"] = LICENCE
        ds.attrib["attribution"] = ATTRIBUTION[dataset]
        ds.attrib["fetched_by"] = "examples/data/fetch_era5.jl on $(Dates.format(now(UTC), "yyyy-mm-ddTHH:MMZ"))"
    end
    return path
end

const PL_LEVELS = ["500", "550", "600", "650", "700", "750", "775", "800", "825", "850", "875", "900", "925", "950",
                   "975", "1000"]

function fetch_era5(lat, lon, date::Date, h0, h1, outdir; half = 1.0, name = "era5")
    area = [lat + half, lon - half, lat - half, lon + half]             # N, W, S, E
    common = Dict{String,Any}("product_type" => ["reanalysis"], "year" => [Dates.format(date, "yyyy")],
                              "month" => [Dates.format(date, "mm")], "day" => [Dates.format(date, "dd")],
                              "time" => [@sprintf("%02d:00", h) for h in h0:h1], "area" => area,
                              "data_format" => "netcdf", "download_format" => "unarchived")
    pl = merge(common, Dict{String,Any}("variable" => ["temperature", "specific_humidity", "u_component_of_wind",
                                                        "v_component_of_wind", "vertical_velocity", "geopotential"],
                                        "pressure_level" => PL_LEVELS))
    sfc = merge(common, Dict{String,Any}("variable" => ["surface_pressure", "skin_temperature",
                                                         "2m_temperature", "geopotential"]))
    mkpath(outdir)
    out = String[]
    for (ds, req, suf) in (("reanalysis-era5-pressure-levels", pl, "pl"), ("reanalysis-era5-single-levels", sfc, "sfc"))
        p = joinpath(outdir, "$(name)_$(suf).nc")
        cds_retrieve(ds, req, p)
        stamp!(p, ds, req)
        @printf("wrote %s (%.1f kB)\n", p, filesize(p) / 1024)
        push!(out, p)
    end
    return out
end

if abspath(PROGRAM_FILE) == @__FILE__
    if isempty(ARGS)
        fetch_era5(52.21, 14.12, Date(2019, 6, 29), 0, 12, @__DIR__; half = 1.0, name = "era5_lindenberg_20190629")
    else
        length(ARGS) in (6, 7) || error("usage: fetch_era5.jl <lat> <lon> <yyyy-mm-dd> <h0> <h1> <outdir> [half_box_deg]")
        fetch_era5(parse(Float64, ARGS[1]), parse(Float64, ARGS[2]), Date(ARGS[3]), parse(Int, ARGS[4]),
                   parse(Int, ARGS[5]), ARGS[6]; half = length(ARGS) == 7 ? parse(Float64, ARGS[7]) : 1.0)
    end
end
