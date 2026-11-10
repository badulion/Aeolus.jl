# ERA5 for the Berlin 100 m run: 22 UTC 23 Aug - 22 UTC 24 Aug 2025 (00-24 h CEST),
# pressure levels and single levels (with the soil: stl1-4, swvl1-4, slt, lsm),
# a ±1° box around the domain centre. Uses AeolusPrep.jl's CDS client
# (data/era5/fetch_era5.jl; credentials in ~/.cdsapirc).
#
#   julia --project=env fetch_era5.jl
#
# The CDS gives the cartesian product of days × hours, so both days are
# fetched whole (data/era5_raw_*.nc) and cut to the 25 hourly records of the
# run (data/era5_berlin_20250824_{pl,sfc}.nc): the ERA5 preprocessor takes
# the first record as the model start.

import AeolusPrep
include(joinpath(pkgdir(AeolusPrep), "data", "era5", "fetch_era5.jl"))
using NCDatasets

const LAT, LON, HALF = 52.515, 13.40, 1.0
const T0, T1 = DateTime(2025, 8, 23, 22), DateTime(2025, 8, 24, 22)
const DIR = joinpath(@__DIR__, "data")

function request(vars; levels = nothing)
    r = Dict{String,Any}("product_type" => ["reanalysis"], "year" => ["2025"], "month" => ["08"], "day" => ["23", "24"],
                         "time" => [@sprintf("%02d:00", h) for h in 0:23],
                         "area" => [LAT + HALF, LON - HALF, LAT - HALF, LON + HALF],
                         "data_format" => "netcdf", "download_format" => "unarchived", "variable" => vars)
    levels === nothing || (r["pressure_level"] = levels)
    return r
end

# the records of `src` in [T0, T1] written to `dst` (every variable over the time dimension sliced)
function cut(src, dst)
    NCDataset(src) do s
        tdim = haskey(s.dim, "valid_time") ? "valid_time" : "time"
        t = s[tdim][:]
        keep = findall(x -> T0 <= x <= T1, t)
        length(keep) == 25 || error("$src: $(length(keep)) records in [$T0, $T1], expected 25")
        isfile(dst) && rm(dst)
        NCDataset(dst, "c"; attrib = collect(s.attrib)) do d
            for (n, v) in s.dim
                d.dim[n] = n == tdim ? length(keep) : v
            end
            for n in keys(s)
                v = s[n]
                dn = NCDatasets.dimnames(v)
                A = v.var[ntuple(_ -> Colon(), length(dn))...]
                if tdim in dn
                    k = findfirst(==(tdim), collect(dn))
                    A = selectdim(A, k, keep)
                end
                atts = [a for a in v.attrib if a[1] != "_FillValue"]
                fv = get(v.attrib, "_FillValue", nothing)
                w = fv === nothing ? defVar(d, n, eltype(A), dn; attrib = atts) :
                                     defVar(d, n, eltype(A), dn; attrib = atts, fillvalue = fv)
                w.var[ntuple(_ -> Colon(), length(dn))...] = A
            end
        end
    end
    @printf("wrote %s (%.1f MB)\n", dst, filesize(dst) / 2^20)
end

mkpath(DIR)
pl = request(["temperature", "specific_humidity", "u_component_of_wind", "v_component_of_wind",
              "vertical_velocity", "geopotential"]; levels = PL_LEVELS)
sfc = request(vcat(["surface_pressure", "skin_temperature", "2m_temperature", "geopotential"], SOIL_VARIABLES))
for (ds, req, suf) in (("reanalysis-era5-pressure-levels", pl, "pl"), ("reanalysis-era5-single-levels", sfc, "sfc"))
    raw = joinpath(DIR, "era5_raw_$(suf).nc")
    isfile(raw) || (cds_retrieve(ds, req, raw); stamp!(raw, ds, req))
    cut(raw, joinpath(DIR, "era5_berlin_20250824_$(suf).nc"))
end
