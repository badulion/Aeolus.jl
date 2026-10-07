# Writes the GRIB2 fixture of the ERA5 preprocessor tests (CFG-h, #363):
# the analytic ERA5-format atmosphere of test/support/era5_analytic.jl as
# `analytic_pl.grib2` (t, q, u, v, w, z on isobaricInhPa) and
# `analytic_sfc.grib2` (sp, skt, 2t, z on the surface), the layout of the
# CDS's GRIB delivery, written once and committed:
#
#   julia --project=. test/data/era5/make_grib_fixture.jl
#
# Messages are cloned from ecCodes' regular_ll GRIB2 samples (shipped with
# eccodes_jll) through GRIB.jl (a dependency of GRIBDatasets). The values
# are packed as 64-bit IEEE floats (`packingType = grid_ieee`,
# `precision = 2`), so the GRIB path reads the same Float64 numbers the
# NetCDF fixture holds and the two drivers can be compared bitwise. (ERA5
# itself is simple-packed at 16 bits; the reader does not care.)

using Aeolus
import GRIBDatasets
const GRIB = GRIBDatasets.GRIB
include(joinpath(@__DIR__, "..", "..", "support", "era5_analytic.jl"))

const SAMPLES = joinpath(GRIB.eccodes_jll.artifact_dir, "share", "eccodes", "samples")

function template(kind)
    GRIB.GribFile(joinpath(SAMPLES, "regular_ll_$(kind)_grib2.tmpl")) do f
        return GRIB.Message(f)
    end
end

function message(tmpl, short, level_type, level, k, values)
    m = deepcopy(tmpl)
    m["Ni"] = length(EA_LON); m["Nj"] = length(EA_LAT)
    m["iScansNegatively"] = 0; m["jScansPositively"] = 0
    m["latitudeOfFirstGridPointInDegrees"] = EA_LAT[end]; m["latitudeOfLastGridPointInDegrees"] = EA_LAT[1]
    m["longitudeOfFirstGridPointInDegrees"] = EA_LON[1]; m["longitudeOfLastGridPointInDegrees"] = EA_LON[end]
    m["iDirectionIncrementInDegrees"] = EA_LON[2] - EA_LON[1]; m["jDirectionIncrementInDegrees"] = EA_LAT[2] - EA_LAT[1]
    t = EA_TIMES[k]
    m["dataDate"] = parse(Int, Dates.format(t, "yyyymmdd")); m["dataTime"] = 100 * Dates.hour(t)
    m["typeOfLevel"] = level_type
    level === nothing || (m["level"] = level)
    m["shortName"] = short
    m["packingType"] = "grid_ieee"; m["precision"] = 2
    m["values"] = vec(values[:, end:-1:1])          # west to east fastest, rows north to south
    return m
end

function write_fixture(dir = @__DIR__)
    lev, sfc = ea_fields()
    pl = joinpath(dir, "analytic_pl.grib2"); sf = joinpath(dir, "analytic_sfc.grib2")
    rm(pl; force = true); rm(sf; force = true)
    tp = template("pl"); ts = template("sfc")
    for k in eachindex(EA_TIMES), (l, p) in enumerate(EA_PLEV), n in ("t", "q", "u", "v", "w", "z")
        GRIB.writemessage(message(tp, n, "isobaricInhPa", Int(p), k, lev[n][:, :, l, k]), pl; mode = "a")
    end
    for k in eachindex(EA_TIMES), (n, g, lt) in (("sp", "sp", "surface"), ("skt", "skt", "surface"),
                                                 ("t2m", "2t", "heightAboveGround"), ("z", "z", "surface"))
        GRIB.writemessage(message(ts, g, lt, lt == "heightAboveGround" ? 2 : nothing, k, sfc[n][:, :, k]), sf;
                          mode = "a")
    end
    return pl, sf
end

if abspath(PROGRAM_FILE) == @__FILE__
    for f in write_fixture()
        println("wrote ", f, " (", filesize(f), " bytes)")
    end
end
