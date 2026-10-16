# Lindenberg, 29/30 June 2019: the checks of the Berlin parent alone against
# the soundings — checks 4 and 6 of the parent validation (N-parent-e,
# #259). Analysis only (EX-a, #438): it reads the output of a finished run
# of examples/urban/berlin_parent_coarse.toml (24 h on the coarse
# parent, 24 × 24 × 38, Δx = 250 m, ~6 min wall; NEST-p3 #483: a domain of
# the model's core, its output parent.nc, parent_output.jl) and the
# sounding driver, and runs nothing:
#
#     julia --project -m Aeolus run examples/urban/berlin_parent_coarse.toml
#     julia --project examples/urban/berlin_parent_lindenberg_check.jl <run folder>
#
#   4. z_i at 14 LT against the 12 UTC Lindenberg sounding, θ at 500 m and
#      1 km, the residual layer at 03 LT, the night nudging (Heinze et al.
#      2017, Atmos. Chem. Phys. 17, 7083-7109);
#   6. robustness: 24 h without NaN, no retries, the adaptive dt history.
#
# The "measured" numbers next to the checks were measured with the epic's
# original subsidence D = 3e-6 s^-1 on the ParentCase parent, which then
# completed on this coarse grid. The parent default is now D = 0 (#312);
# re-measuring them at the final D, on the parent [[domain]], is part of
# issue #315. The night forcing is now the large-scale forcing's total
# (nudging, damping and the geostrophic pressure gradient's work, one term
# of the domain's budget; the ParentCase ledger had the nudging apart).

using Test
using Printf
using Dates: DateTime, Date, value
using Aeolus
using Aeolus.InputDrivers: read_dynamic_driver, shift_table
include(joinpath(@__DIR__, "parent_output.jl"))

length(ARGS) == 1 || error("usage: berlin_parent_lindenberg_check.jl <run folder of berlin_parent_coarse.toml>")
const RUN = ARGS[1]
const P = read_parent_output(RUN)

function rec(h)
    i = findfirst(x -> isapprox(x, h; atol = 1e-6), P.hour_lt)
    i === nothing && error("$RUN has no record at $h LT: the checks need the 24 h run of berlin_parent_coarse.toml")
    return i
end
θat(θ, h) = (k = searchsortedlast(P.z, h); θ[k] + (θ[k+1] - θ[k]) * (h - P.z[k]) / (P.z[k+1] - P.z[k]))

# the sounding the run started from: its dynamic driver (the tables' time
# axis shifted to seconds from 00 UTC of the case date)
const DYN = P.cfg.domain.dynamic_driver

@testset "4 + 6. Lindenberg 29/30 June 2019, 24 h (coarse parent)" begin
    @test P.grid == [24, 24, 38]
    @test all(isfinite, P.θ) && P.t[end] ≈ 24 * 3600
    @test sum(P.retries) == 0
    @info @sprintf("Lindenberg 24 h: %d steps, dt %.2f–%.2f s (mean %.2f), %d retries",
                   length(P.dts), minimum(P.dts), maximum(P.dts), sum(P.dts) / length(P.dts), sum(P.retries))
    @test minimum(P.dts[1:end-1]) >= 0.01 && maximum(P.dts) <= 10.0      # (the last step lands on the end)

    d = read_dynamic_driver(DYN; planes = false)
    off = value(DateTime(rstrip(d.init_time, 'Z')) - DateTime(Date(DateTime(rstrip(d.init_time, 'Z'))))) / 1000
    S = (θ = shift_table(d.tables[:profile_theta], off), qv = shift_table(d.tables[:profile_qv], off))
    MT = Aeolus.MoistThermodynamics
    t12 = 12 * 3600.0                     # 14 LT = 12 UTC (s from 00 UTC; released 10:45)
    z = S.θ.z
    θv_obs = [MT.virtual_temperature(forcing_value(S.θ, h, t12), MT.PhasePartition(forcing_value(S.qv, h, t12)))
              for h in z]
    zi_obs = boundary_layer_height(z, θv_obs; method = :parcel)
    i14 = rec(14.0)
    zi14, zif14 = P.zi_parcel[i14], P.zi_min_flux[i14]
    err = zi14 / zi_obs - 1
    @info @sprintf("z_i(14 LT) parcel %.0f m (min flux %.0f m), 12 UTC sounding %.0f m: %+.1f %%",
                   zi14, zif14, zi_obs, 100err)
    @test abs(err) < 0.25           # measured +12.6 % (1332 m against 1183 m)
    for h in (500.0, 1000.0)
        dθ = θat(P.θ[:, i14], h) - forcing_value(S.θ, h, t12)
        @info @sprintf("⟨θ⟩(%.0f m, 14 LT) = %.2f K, sounding %.2f K: %+.2f K", h, θat(P.θ[:, i14], h),
                       forcing_value(S.θ, h, t12), dθ)
        @test abs(dθ) < 1.5          # measured +0.12 K (500 m), -0.17 K (1000 m)
    end
    # residual layer at 03 LT: |dθ/dz| < 1.5 K/km between 300 m and 0.8 z_i,max
    zimax = maximum(P.zi_parcel)
    θ03 = P.θ[:, rec(3.0)]
    bulk = (θat(θ03, 0.8zimax) - θat(θ03, 300.0)) / (0.8zimax - 300)
    @info @sprintf("residual layer 03 LT: bulk dθ/dz %.2f K/km over 300-%.0f m; z_i,max %.0f m",
                   1e3bulk, 0.8zimax, zimax)
    @test abs(bulk) < 1.5e-3        # measured 0.26 K/km (z_i,max 1529 m)
    # the night forcing against the daytime surface flux (Heinze et al. 2017):
    # 20-05 LT and 10-16 LT (14-23 h and 4-10 h from the 06 LT start)
    E = P.energy
    nn = budget_rate(P, E["body_force__large_scale_forcing"], 14 * 3600.0, 23 * 3600.0)
    hs = budget_rate(P, E["surface_flux__tiled_surface"], 4 * 3600.0, 10 * 3600.0)
    @info @sprintf("large-scale forcing 20-05 LT %.2f W/m², surface 10-16 LT %.1f W/m² (ratio %.3f); 24 h: %s",
                   nn, hs, nn / hs, join((@sprintf("%s %.1f", k, budget_rate(P, v, 0.0, P.ts[end]))
                                          for (k, v) in sort!(collect(E)) if k != "closure"), ", "))
    # measured (ParentCase, the nudging alone): 8.5 W/m² at night against
    # 299 W/m² by day (ratio 0.028). The observed free troposphere warmed by
    # 3.7-6.5 K between the 04:45 UTC launches of the 29th and the 30th (+6.0 K
    # at 2 km, the warm-air advection of the heat wave): over 24 h the
    # nudging (12 W/m²) and the damping (23 W/m²) carry part of that into the
    # box, above 1.2 z_i.
    @test abs(nn) < 0.05 * hs
    scale = maximum(maximum(abs, v) for (k, v) in E if k != "closure")
    @test maximum(abs, E["closure"]) < 1e-10 * scale
end
