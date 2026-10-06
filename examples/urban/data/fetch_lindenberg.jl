# Download the Lindenberg (WMO 10393, IGRA2 GMM00010393) radiosonde
# launches of the Berlin case day from NOAA NCEI and write them, with a
# provenance block, to `lindenberg_10393_20190629.csv`.
#
#   julia examples/urban/data/fetch_lindenberg.jl
#
# Source: the Integrated Global Radiosonde Archive, version 2 (IGRA2),
# Durre, Yin, Vose, Applequist & Arnfield (2018), J. Atmos. Oceanic
# Technol. 35, 1753-1770, doi:10.1175/JTECH-D-17-0223.1; data set
# doi:10.7289/V5X63K0Q. NOAA data, public domain. Format:
# https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/doc/igra2-data-format.txt
#
# Offline tooling only: `curl` downloads the period-of-record zip
# (~77 MB) into a temporary directory and `unzip -p` (or, where unzip is
# not installed, Python's stdlib `python3 -m zipfile`) extracts it. The
# package itself never runs this and has no dependency on either tool.
#
# Case day (see the provenance block written below): 2024-06-20, the
# date the epic first named, was overcast (DWD hourly total cloud cover
# 7-8 oktas most hours at Lindenberg and Berlin-Tempelhof) and its 12 UTC
# sounding is saturated at 907 hPa (dew-point depression 0.0 K, about
# 1 km), so by the maintainer decision of 2026-10-05 (epic #246) the case
# moved to the clear heat-wave day 2019-06-29 (Lindenberg 0-2 oktas from
# 04 UTC on the 29th to 16 UTC on the 30th).

using Printf

const STATION    = "GMM00010393"
const URL        = "https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/access/data-por/$(STATION)-data.txt.zip"
const OUT_PATH   = joinpath(@__DIR__, "lindenberg_10393_20190629.csv")
const EPOCH      = (2019, 6, 29)                 # t = 0 at 00 UTC of this day
# nominal (y, m, d, hour UTC) of the launches kept
const LAUNCHES   = [(2019, 6, 29, 0), (2019, 6, 29, 6), (2019, 6, 29, 12), (2019, 6, 29, 18),
                    (2019, 6, 30, 0), (2019, 6, 30, 6)]
const P_MIN      = 10000                          # Pa: levels above 100 hPa are not kept
# Launch-site elevation. IGRA2's station list gives 97.7 m (the elevation
# of the observatory's synop site, DWD 03015: 98 m); the surface record of
# the same launches in the University of Wyoming archive gives 112 m, and
# the IGRA 1000 hPa geopotential heights imply 112 ± 3 m (hypsometric
# integration from the surface record, test/era6_sounding.jl). 112 m is used.
const STATION_ELEVATION = 112.0

days_from_civil(y, m, d) = begin                 # days since 1970-01-01 (Hinnant)
    y -= m <= 2
    era = fld(y, 400); yoe = y - era * 400
    doy = fld(153 * (m + (m > 2 ? -3 : 9)) + 2, 5) + d - 1
    doe = yoe * 365 + fld(yoe, 4) - fld(yoe, 100) + doy
    era * 146097 + doe - 719468
end

function download_igra(dir)
    zip = joinpath(dir, "$(STATION)-data.txt.zip")
    txt = joinpath(dir, "$(STATION)-data.txt")
    @printf("Downloading %s …\n", URL)
    run(`curl -sS --fail --max-time 900 -o $zip $URL`)
    if Sys.which("unzip") !== nothing
        run(pipeline(`unzip -p $zip`; stdout = txt))
    elseif Sys.which("python3") !== nothing
        run(`python3 -m zipfile -e $zip $dir`)
    else
        error("neither unzip nor python3 is available to extract $zip")
    end
    return txt
end

field(line, a, b) = strip(line[a:min(b, length(line))])
ival(s) = parse(Int, s)

# IGRA2 data record columns (igra2-data-format.txt)
function parse_level(line)
    lvl1 = line[1]; lvl2 = line[2]
    press = ival(field(line, 10, 15)); pflag = field(line, 16, 16)
    gph   = ival(field(line, 17, 21)); zflag = field(line, 22, 22)
    temp  = ival(field(line, 23, 27)); tflag = field(line, 28, 28)
    dpdp  = ival(field(line, 35, 39))
    wdir  = ival(field(line, 41, 45)); wspd = ival(field(line, 47, 51))
    return (; lvl = string(lvl1, lvl2), press, pflag, gph, zflag, temp, tflag, dpdp, wdir, wspd)
end

function main()
    dir = mktempdir()
    txt = download_igra(dir)
    want = Set(LAUNCHES)
    rows = String[]
    dropped = Dict{String,Int}()
    drop!(why) = (dropped[why] = get(dropped, why, 0) + 1)
    found = Tuple{Int,Int,Int,Int}[]
    cur = nothing
    t_epoch = days_from_civil(EPOCH...) * 86400
    for line in eachline(txt)
        if startswith(line, "#")
            y = ival(field(line, 14, 17)); m = ival(field(line, 19, 20))
            d = ival(field(line, 22, 23)); h = ival(field(line, 25, 26))
            rel = ival(field(line, 28, 31))
            if (y, m, d, h) in want
                # release time HHMM (UTC); 9999 = missing → nominal hour
                rh, rm = rel == 9999 ? (h, 0) : (div(rel, 100), rem(rel, 100))
                tday = days_from_civil(y, m, d) * 86400
                trel = tday + 3600rh + 60rm
                trel > tday + 3600h + 3 * 3600 && (trel -= 86400)  # 22:45 of the previous day
                cur = (; tag = @sprintf("%04d-%02d-%02dT%02dZ", y, m, d, h),
                         rel = @sprintf("%04d", rel), t = trel - t_epoch, psfc = Ref(typemax(Int)))
                push!(found, (y, m, d, h))
            else
                cur = nothing
            end
            continue
        end
        cur === nothing && continue
        L = parse_level(line)
        L.press == -9999 && (drop!("no pressure"); continue)
        L.press < P_MIN && continue
        L.lvl[2] == '1' && (cur.psfc[] = L.press)
        if L.press > cur.psfc[]
            drop!("below the surface (p > p_sfc)"); continue
        end
        T  = L.temp in (-9999, -8888) ? NaN : L.temp / 10 + 273.15
        Td = (isnan(T) || L.dpdp in (-9999, -8888)) ? NaN : T - L.dpdp / 10
        ws = L.wspd in (-9999, -8888) ? NaN : L.wspd / 10
        wd = (isnan(ws) || L.wdir in (-9999, -8888)) ? NaN : float(L.wdir)
        L.temp == -8888 && drop!("temperature removed by IGRA QC (-8888)")
        if isnan(T) && isnan(ws)
            drop!("neither temperature nor wind"); continue
        end
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
    today = string(Libc.strftime("%Y-%m-%d", time()))
    open(OUT_PATH, "w") do io
        println(io, """
# Lindenberg (WMO 10393; IGRA2 station $STATION; 52.2085 N, 14.1180 E) radiosonde launches, 2019-06-29 00 UTC to 2019-06-30 06 UTC.
# Source: NOAA NCEI, Integrated Global Radiosonde Archive v2 (IGRA2), period-of-record file $(STATION)-data.txt.zip,
#   $URL
#   downloaded $today by examples/urban/data/fetch_lindenberg.jl. Public domain (NOAA). Data source in the IGRA headers: ncdc-gts
#   (GTS TEMP messages: standard and significant levels, not the full-resolution BUFR profile).
# Reference: Durre, I., Yin, X., Vose, R. S., Applequist, S. & Arnfield, J. (2018), J. Atmos. Oceanic Technol. 35, 1753-1770.
# Columns: launch = nominal synoptic time (UTC); release_utc = actual release HHMM (UTC); t_s = release time in seconds from the
#   epoch below (the time stamp of the profile); lvltyp = IGRA level type (first digit 1 standard, 2 other; second digit 1 surface,
#   2 tropopause, 0 other); p_Pa; pflag/zflag/tflag = IGRA QC flags (A tier-1 climatological check passed, B tiers 1 and 2 passed,
#   - not checked); gph_m = reported geopotential height above sea level (standard levels only); T_K; Td_K = T - dew-point
#   depression; wdir_deg (from), wspd_ms. NaN = missing.
#! epoch = $(EPOCH[1])-$(lpad(EPOCH[2], 2, '0'))-$(lpad(EPOCH[3], 2, '0'))T00:00Z
#! station_elevation_m = $STATION_ELEVATION
# Station elevation: IGRA2 station list 97.7 m (the observatory synop site, DWD 03015: 98 m); the surface record of these launches
#   in the University of Wyoming archive and the reported 1000 hPa geopotential heights both put the launch site at 112 m. 112 m used.
# Levels dropped: $(isempty(dropped) ? "none" : join(["$(v) ($(k))" for (k, v) in sort!(collect(dropped))], "; ")).
#   Levels above $(P_MIN ÷ 100) hPa are not kept (not needed for a <= 3 km domain). Wind-only levels (T = NaN) are kept for the wind profile.
# Case-day check (DWD Climate Data Center, hourly total cloud cover, stundenwerte_N_<id>_..._hist.zip, read 2026-10-05):
#   2024-06-20 (the date first proposed): 7-8 oktas most hours at Lindenberg (03015) and Berlin-Tempelhof (00433); the 12 UTC
#   sounding has dew-point depression 0.0 K at 907 hPa (saturated near 1 km). Not clear-sky: rejected.
#   2019-06-29 (used): Lindenberg 0-2 oktas from 00 UTC 29 June to 16 UTC 30 June, Berlin Brandenburg (00427) 0-2 oktas over the
#   same interval; minimum dew-point depression below 700 hPa in the 12 UTC sounding 6.0 K. Berlin-Alexanderplatz (00399) has no
#   cloud observations after 1992, so Tempelhof/Brandenburg stand in for it. Part of the late-June 2019 European heat wave.
launch,release_utc,t_s,lvltyp,p_Pa,pflag,gph_m,zflag,T_K,tflag,Td_K,wdir_deg,wspd_ms""")
        foreach(r -> println(io, r), rows)
    end
    @printf("Wrote %s: %d levels from %d launches; dropped %s\n", OUT_PATH, length(rows),
            length(found), isempty(dropped) ? "none" : string(dropped))
end

main()
