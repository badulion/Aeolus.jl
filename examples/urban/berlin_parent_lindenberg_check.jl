# Lindenberg, 29/30 June 2019: the checks of the Berlin parent alone against
# the soundings — checks 4 and 6 of the parent validation (N-parent-e,
# #259). Analysis only (EX-a, #438): it reads the output of a finished run
# of examples/urban/configs/berlin_parent_coarse.toml (24 h on the coarse
# parent, 24 × 24 × 38, Δx = 250 m, ~6 min wall) and the sounding driver,
# and runs nothing:
#
#     julia --project -m Aeolus run examples/urban/configs/berlin_parent_coarse.toml
#     julia --project examples/urban/berlin_parent_lindenberg_check.jl <run folder>
#
#   4. z_i at 14 LT against the 12 UTC Lindenberg sounding, θ at 500 m and
#      1 km, the residual layer at 03 LT, the night nudging (Heinze et al.
#      2017, Atmos. Chem. Phys. 17, 7083-7109);
#   6. robustness: 24 h without NaN, no retries, the adaptive dt history.
#
# The "measured" numbers next to the checks were measured with the epic's
# original subsidence D = 3e-6 s^-1, which then completed on this coarse
# grid. The parent default is now D = 0 (#312); re-measuring them at the
# final D is part of issue #315.

using Test
using Printf
using TOML
using Dates: DateTime, Date, value
using Aeolus
using Aeolus.InputDrivers: read_dynamic_driver, shift_table

length(ARGS) == 1 || error("usage: berlin_parent_lindenberg_check.jl <run folder of berlin_parent_coarse.toml>")
const RUN = ARGS[1]
const OUT = joinpath(RUN, "output")
const RES = TOML.parsefile(joinpath(OUT, "results.toml"))
const CFG = load_run_config(RUN)            # the run's resolved config (run.toml)

"A comma table with `#` comment lines and a header: (header, rows of Float64)."
function read_table(path)
    lines = [l for l in readlines(path) if !startswith(l, "#")]
    return split(lines[1], ','), [parse.(Float64, split(l, ',')) for l in lines[2:end]]
end

hd, rows = read_table(joinpath(OUT, "parent.csv"))
col(name) = [r[findfirst(==(name), hd)] for r in rows]
thd, trows = read_table(joinpath(OUT, "parent_theta.csv"))
const ZC = [parse(Float64, s[2:end]) for s in thd[2:end]]
hours_lt = col("hour_LT")
rec(h) = findfirst(x -> isapprox(x, h; atol = 1e-6), hours_lt)
θprof(i) = trows[i][2:end]
θat(θ, h) = (k = searchsortedlast(ZC, h); θ[k] + (θ[k+1] - θ[k]) * (h - ZC[k]) / (ZC[k+1] - ZC[k]))

# the sounding the run started from: its dynamic driver (the tables' time
# axis shifted to seconds from 00 UTC of the case date)
const DYN = CFG.initial.dynamic

@testset "4 + 6. Lindenberg 29/30 June 2019, 24 h (coarse parent)" begin
    @test RES["grid"] == [24, 24, 38]
    @test RES["finite"] && !haskey(RES, "failure")
    @test RES["retries"] == 0
    @info @sprintf("Lindenberg 24 h: %d steps, dt %.2f–%.2f s (mean %.2f), %d retries",
                   RES["steps"], RES["dt_min"], RES["dt_max"], RES["dt_mean"], RES["retries"])
    @test RES["dt_min"] >= 0.01 && RES["dt_max"] <= 10.0

    d = read_dynamic_driver(DYN; planes = false)
    off = value(DateTime(rstrip(d.init_time, 'Z')) - DateTime(Date(DateTime(rstrip(d.init_time, 'Z'))))) / 1000
    P = (θ = shift_table(d.tables[:profile_theta], off), qv = shift_table(d.tables[:profile_qv], off))
    MT = Aeolus.MoistThermodynamics
    t12 = 12 * 3600.0                     # 14 LT = 12 UTC (s from 00 UTC; released 10:45)
    z = P.θ.z
    θv_obs = [MT.virtual_temperature(forcing_value(P.θ, h, t12), MT.PhasePartition(forcing_value(P.qv, h, t12)))
              for h in z]
    zi_obs = boundary_layer_height(z, θv_obs; method = :parcel)
    i14 = rec(14.0)
    zi14, zif14 = col("zi_parcel_m")[i14], col("zi_min_flux_m")[i14]
    err = zi14 / zi_obs - 1
    @info @sprintf("z_i(14 LT) parcel %.0f m (min flux %.0f m), 12 UTC sounding %.0f m: %+.1f %%",
                   zi14, zif14, zi_obs, 100err)
    @test abs(err) < 0.25           # measured +12.6 % (1332 m against 1183 m)
    for h in (500.0, 1000.0)
        dθ = θat(θprof(i14), h) - forcing_value(P.θ, h, t12)
        @info @sprintf("⟨θ⟩(%.0f m, 14 LT) = %.2f K, sounding %.2f K: %+.2f K", h, θat(θprof(i14), h),
                       forcing_value(P.θ, h, t12), dθ)
        @test abs(dθ) < 1.5          # measured +0.12 K (500 m), -0.17 K (1000 m)
    end
    # residual layer at 03 LT: |dθ/dz| < 1.5 K/km between 300 m and 0.8 z_i,max
    zimax = RES["zi_parcel_max"]
    θ03 = θprof(rec(3.0))
    bulk = (θat(θ03, 0.8zimax) - θat(θ03, 300.0)) / (0.8zimax - 300)
    @info @sprintf("residual layer 03 LT: bulk dθ/dz %.2f K/km over 300-%.0f m; z_i,max %.0f m",
                   1e3bulk, 0.8zimax, zimax)
    @test abs(bulk) < 1.5e-3        # measured 0.26 K/km (z_i,max 1529 m)
    # the night nudging against the daytime surface flux (Heinze et al. 2017)
    nn = RES["heat_night"]["dynamics.nudging"]
    hs = RES["heat_day"]["surface_flux.tiles"]
    h24 = RES["heat"]
    @info @sprintf("nudging 20-05 LT %.2f W/m², surface 10-16 LT %.1f W/m² (ratio %.3f); 24 h: surface %.1f, radiation %.1f, subsidence %.1f, nudging %.1f, damping %.1f, top %.3f W/m²",
                   nn, hs, nn / hs, (get(h24, k, 0.0) for k in ("surface_flux.tiles", "column_tendency.column_radiation",
                    "dynamics.subsidence", "dynamics.nudging", "dynamics.damping", "dynamics.top_flux"))...)
    # measured: 8.5 W/m² at night against 299 W/m² by day (ratio 0.028). The
    # observed free troposphere warmed by 3.7-6.5 K between the 04:45 UTC
    # launches of the 29th and the 30th (+6.0 K at 2 km, the warm-air
    # advection of the heat wave): over 24 h the nudging (12 W/m²) and the
    # damping (23 W/m²) carry part of that into the box, above 1.2 z_i.
    @test abs(nn) < 0.05 * hs
    @test RES["closure_rel"] < 1e-10
end
