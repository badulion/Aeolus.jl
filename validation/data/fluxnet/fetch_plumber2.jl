# Regenerate the PLUMBER2 subsets in this directory (not run by the
# validation suite; needs network access). Usage, from the repository root:
#
#     julia --project=. validation/data/fluxnet/fetch_plumber2.jl
#
# Source: Ukkola, A. & Abramowitz, G. (2020). PLUMBER2: forcing and
# evaluation datasets for a model intercomparison project for land surface
# models, v1.0. NCI National Research Data Collection,
# doi:10.25914/5fdb0902607e1 — derived from FLUXNET2015 (Tier 1),
# licence CC BY 4.0. Read over OPeNDAP from the NCI THREDDS server.
# FLUXNET2015 itself: Pastorello, G. et al. (2020). The FLUXNET2015
# dataset and the ONEFlux processing pipeline for eddy covariance data.
# Sci. Data, 7, 225. The five tower records are FLUXNET2015 Tier 1 (CC BY
# 4.0): DE-Hai (Hainich), FI-Hyy (Hyytiälä), GF-Guy (Guyaflux), DE-Gri
# (Grillenburg), FR-Pue (Puéchabon). Changes made here: hourly means of
# the half-hourly values, unit conversions (Qair g/kg, Psurf hPa, Precip
# mm/h), CO2air reduced to its window mean, and the time windows below.
#
# Each file holds hourly means of the half-hourly PLUMBER2 records for
# one window, local standard time as in FLUXNET2015. Flux quality flags
# are the worse (larger) of the two half hours; 0 = measured.

using Downloads, Dates, Printf

const BASE = "https://thredds.nci.org.au/thredds/dodsC/ks32/CLEX_Data/PLUMBER2/v1-0"

# site => (file stem, first day, last day)
const WINDOWS = [
    ("DE-Hai", "DE-Hai_2000-2012", Date(2008, 6, 10), Date(2008, 7, 14)),
    ("FI-Hyy", "FI-Hyy_1996-2014", Date(2008, 6, 26), Date(2008, 7, 30)),
    ("GF-Guy", "GF-Guy_2004-2014", Date(2007, 9, 10), Date(2007, 10, 14)),
    ("DE-Gri", "DE-Gri_2004-2014", Date(2008, 6, 10), Date(2008, 7, 14)),
    ("FR-Pue", "FR-Pue_2000-2014", Date(2006, 5, 1), Date(2006, 8, 31)),
]

const MET = ["SWdown", "LWdown", "Tair", "Qair", "Wind", "Psurf", "Precip", "CO2air", "LAI"]  # CO2air: window mean only
const FLUX = ["Rnet", "Qh", "Qle", "Qg", "Qh_qc", "Qle_qc", "Qh_cor", "Qle_cor"]

fetch_text(url) = String(take!(Downloads.download(url, IOBuffer())))

function parse_block(txt, var)
    i = findfirst(Regex("^" * var * "\\[", "m"), txt)
    i === nothing && error("no $var in response")
    vals = Float64[]
    for line in eachline(IOBuffer(txt[first(i):end]))
        startswith(line, "[") || (isempty(vals) ? continue : break)
        push!(vals, parse(Float64, strip(split(line, ",")[end])))
    end
    return vals
end

function scalar(txt, var)
    i = findfirst(Regex("^" * var * "\\[", "m"), txt)
    line = split(txt[first(i):end], '\n')[2]
    return parse(Float64, strip(split(line, ",")[end]))
end

function fetch_site(site, stem, d0, d1)
    y0 = parse(Int, stem[8:11])
    i0 = Int(Dates.value(DateTime(d0) - DateTime(y0)) ÷ 1_800_000)
    i1 = Int(Dates.value(DateTime(d1 + Day(1)) - DateTime(y0)) ÷ 1_800_000) - 1
    sub(v) = "$v.$v%5B$i0:1:$i1%5D%5B0%5D%5B0%5D"
    met = fetch_text("$BASE/Met/$(stem)_FLUXNET2015_Met.nc.ascii?" * join(sub.(MET), ",") *
                     ",reference_height.reference_height,canopy_height.canopy_height,IGBP_veg_short," *
                     "latitude.latitude,longitude.longitude,elevation.elevation")
    dds = fetch_text("$BASE/Flux/$(stem)_FLUXNET2015_Flux.nc.dds")
    have = [v for v in FLUX if occursin(Regex("\\b" * v * "\\["), dds)]
    flx = fetch_text("$BASE/Flux/$(stem)_FLUXNET2015_Flux.nc.ascii?" * join(sub.(have), ","))
    cols = Dict(v => parse_block(met, v) for v in MET)
    merge!(cols, Dict(v => parse_block(flx, v) for v in have))
    n0 = length(cols["Tair"])
    for v in setdiff(FLUX, have)              # e.g. no ground heat flux at the site
        cols[v] = fill(-9999.0, n0)
    end
    n = i1 - i0 + 1
    all(length(c) == n for c in values(cols)) || error("$site: length mismatch")
    igbp = strip(match(r"IGBP_veg_short, \"([^\"]*)\"", met)[1])
    meta = (lat = scalar(met, "latitude"), lon = scalar(met, "longitude"),
            zref = scalar(met, "reference_height"), hc = scalar(met, "canopy_height"),
            elev = scalar(met, "elevation"), igbp = igbp)
    hourly(v; worst = false) = [worst ? max(v[2k-1], v[2k]) : (v[2k-1] + v[2k]) / 2 for k in 1:n÷2]
    path = joinpath(@__DIR__, "$site.csv")
    open(path, "w") do io
        println(io, "# PLUMBER2 v1.0 (Ukkola & Abramowitz 2020, doi:10.25914/5fdb0902607e1), from FLUXNET2015 Tier 1; CC BY 4.0")
        println(io, "# site=$site file=$(stem)_FLUXNET2015_{Met,Flux}.nc igbp=$(meta.igbp)")
        @printf(io, "# lat=%.5f lon=%.5f elevation=%.1f reference_height=%.2f canopy_height=%.2f\n",
                meta.lat, meta.lon, meta.elev, meta.zref, meta.hc)
        println(io, "# start=$(d0)T00:00 local standard time, hourly means of the half-hourly records")
        co2 = filter(>(0), cols["CO2air"])
        @printf(io, "# co2_mean=%.1f ppm (window mean of CO2air)\n", sum(co2) / length(co2))
        println(io, "# units: SWdown LWdown Rnet Qh Qle Qg Qh_cor Qle_cor W/m2; Tair K; Qair g/kg; Wind m/s; Psurf hPa; Precip mm/h; LAI m2/m2 (Copernicus); *_qc 0=measured, 9=missing")
        println(io, "hour,SWdown,LWdown,Tair,Qair,Wind,Psurf,Precip,LAI,Rnet,Qh,Qle,Qg,Qh_qc,Qle_qc,Qh_cor,Qle_cor")
        h = Dict(v => hourly(cols[v]; worst = endswith(v, "_qc")) for v in keys(cols))
        for k in 1:n÷2
            f(x) = x < -9000 ? "NaN" : @sprintf("%.1f", x)
            @printf(io, "%d,%s,%s,%.2f,%.3f,%.2f,%.1f,%.2f,%.2f,%s,%s,%s,%s,%d,%d,%s,%s\n",
                    k - 1, f(h["SWdown"][k]), f(h["LWdown"][k]), h["Tair"][k], 1000h["Qair"][k],
                    h["Wind"][k], h["Psurf"][k] / 100, 3600h["Precip"][k], h["LAI"][k],
                    f(h["Rnet"][k]), f(h["Qh"][k]), f(h["Qle"][k]), f(h["Qg"][k]),
                    h["Qh_qc"][k] < -9000 ? 9 : Int(h["Qh_qc"][k]),
                    h["Qle_qc"][k] < -9000 ? 9 : Int(h["Qle_qc"][k]),
                    f(h["Qh_cor"][k]), f(h["Qle_cor"][k]))
        end
    end
    println("wrote $path ($(n ÷ 2) hours)")
end

for (site, stem, d0, d1) in WINDOWS
    fetch_site(site, stem, d0, d1)
end
