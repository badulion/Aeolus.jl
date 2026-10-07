# An analytic ERA5-format atmosphere (CFG-h, #363): isothermal and
# hydrostatic at the site, every field linear in the local (x, y) of the
# projection about the site, so that bilinear interpolation and centred
# differences are exact. The ground (the surface geopotential) is the
# 1000 hPa surface, Φ_s = a x + b y, and the surface pressure 1000 hPa. Written as NetCDF in the CDS layout by
# `ea_write_netcdf` (test/preprocess_era5.jl) and as GRIB2 by
# test/data/era5/make_grib_fixture.jl (the committed GRIB fixture).

using Dates
import NCDatasets
using Aeolus.InputDrivers: latlon_to_meters
using Aeolus.MoistThermodynamics: DEFAULT_CONSTANTS

const EA_LAT = collect(51.25:0.25:53.25)          # multiples of 1/4: exact in binary and in GRIB micro-degrees
const EA_LON = collect(13.0:0.25:15.0)
const EA_PLEV = [1000.0, 925.0, 850.0, 700.0, 600.0, 500.0]   # hPa
const EA_TIMES = [DateTime(2019, 6, 29, h) for h in 0:2]
const EA_SITE = (52.21, 14.12)
const EA_T0 = 280.0                                # K at the site, every level: isothermal
const EA_P0 = 1.0e5                                # Pa, the surface pressure (and the θ reference)
const EA_G0 = DEFAULT_CONSTANTS.grav
const EA_RD = DEFAULT_CONSTANTS.gas_constant / DEFAULT_CONSTANTS.molmass_dry
# the linear slopes (per metre of the local x east, y north)
const EA_PHI_X, EA_PHI_Y = 5.0e-4, -1.2e-3        # m s⁻² : Φ = g₀ z(p) + a x + b y
const EA_T_X, EA_T_Y = 1.0e-5, -2.0e-5            # K m⁻¹
const EA_Q_X, EA_Q_Y = 1.0e-8, 2.0e-8             # m⁻¹
const EA_U, EA_V = 5.0, -3.0                       # m s⁻¹ (u grows by 1 m s⁻¹ per record)
const EA_U_Z, EA_U_X = 2.0e-3, 1.0e-4              # u = EA_U + (k − 1) + EA_U_Z z + EA_U_X x: linear in height and x

ea_xy(lat, lon) = latlon_to_meters(lat, lon, EA_SITE...)
"The isothermal hydrostatic height (m) of pressure `p` (Pa) above the ground at EA_P0."
ea_height(p) = EA_RD * EA_T0 / EA_G0 * log(EA_P0 / p)
ea_phi(p, lat, lon) = (xy = ea_xy(lat, lon); EA_G0 * ea_height(p) + EA_PHI_X * xy[1] + EA_PHI_Y * xy[2])
ea_T(p, lat, lon) = (xy = ea_xy(lat, lon); EA_T0 + EA_T_X * xy[1] + EA_T_Y * xy[2])
ea_q(p, lat, lon) = (xy = ea_xy(lat, lon); 0.008 * (p / EA_P0)^3 + EA_Q_X * xy[1] + EA_Q_Y * xy[2])
"u: linear in the level's height above sea level (Φ/g₀) and in x, so the planes are exact."
ea_u(p, lat, lon, k) = EA_U + (k - 1) + EA_U_Z * ea_phi(p, lat, lon) / EA_G0 + EA_U_X * ea_xy(lat, lon)[1]
ea_v(p, lat, lon, k) = EA_V
ea_omega(p, lat, lon, k) = -0.2 * p / EA_P0        # Pa s⁻¹: rising air
ea_skt(lat, lon, k) = 290.0 + (k - 1)

"`f(p, lat, lon, k)` on the (lon, lat, level, time) grid; levels in Pa."
ea_grid4(f) = [f(100 * p, la, lo, k) for lo in EA_LON, la in EA_LAT, p in EA_PLEV, k in eachindex(EA_TIMES)]
ea_grid3(f) = [f(la, lo, k) for lo in EA_LON, la in EA_LAT, k in eachindex(EA_TIMES)]

"The analytic fields by ERA5 short name: level fields (lon, lat, level, time), surface (lon, lat, time)."
function ea_fields()
    lev = Dict("t" => ea_grid4((p, la, lo, k) -> ea_T(p, la, lo)), "q" => ea_grid4((p, la, lo, k) -> ea_q(p, la, lo)),
               "u" => ea_grid4(ea_u), "v" => ea_grid4(ea_v), "w" => ea_grid4(ea_omega),
               "z" => ea_grid4((p, la, lo, k) -> ea_phi(p, la, lo)))
    sfc = Dict("sp" => ea_grid3((la, lo, k) -> EA_P0), "skt" => ea_grid3(ea_skt),
               "t2m" => ea_grid3((la, lo, k) -> 285.0), "z" => ea_grid3((la, lo, k) -> ea_phi(EA_P0, la, lo)))
    return lev, sfc
end

"""
    ea_write_netcdf(dir; T = Float64) -> (pl, sfc)

The analytic fields as two NetCDF files in the CDS layout (`era5_pl.nc`
over longitude, latitude (decreasing, as ERA5), pressure_level (hPa) and
valid_time (seconds since 1970); `era5_sfc.nc` without the levels), in
element type `T`.
"""
function ea_write_netcdf(dir; T = Float64)
    lev, sfc = ea_fields()
    ilat = reverse(eachindex(EA_LAT))                 # ERA5: north to south
    secs = [Dates.value(t - DateTime(1970)) ÷ 1000 for t in EA_TIMES]
    files = String[]
    for (name, vars) in (("era5_pl.nc", lev), ("era5_sfc.nc", sfc))
        path = joinpath(dir, name)
        NCDatasets.NCDataset(path, "c") do ds
            ds.attrib["Conventions"] = "CF-1.7"
            ds.attrib["institution"] = "analytic ERA5-format fixture (test/support/era5_analytic.jl)"
            NCDatasets.defVar(ds, "longitude", EA_LON, ("longitude",); attrib = ["units" => "degrees_east"])
            NCDatasets.defVar(ds, "latitude", EA_LAT[ilat], ("latitude",); attrib = ["units" => "degrees_north"])
            dims = ("longitude", "latitude", "valid_time")
            if name == "era5_pl.nc"
                NCDatasets.defVar(ds, "pressure_level", EA_PLEV, ("pressure_level",); attrib = ["units" => "hPa"])
                dims = ("longitude", "latitude", "pressure_level", "valid_time")
            end
            NCDatasets.defVar(ds, "valid_time", secs, ("valid_time",);
                              attrib = ["units" => "seconds since 1970-01-01", "calendar" => "proleptic_gregorian"])
            for (k, A) in vars
                B = ndims(A) == 4 ? A[:, ilat, :, :] : A[:, ilat, :]
                NCDatasets.defVar(ds, k, T.(B), dims)
            end
        end
        push!(files, path)
    end
    return files
end
