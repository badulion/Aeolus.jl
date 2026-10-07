# Lindenberg, 29/30 June 2019: the Berlin parent alone, 24 h on the coarse
# parent (24 × 24 × 38, Δx = 250 m) — checks 4 and 6 of the parent
# validation (N-parent-e, #259), moved out of test/berlin_parent.jl because
# they need a 24 h simulation (~6 min wall on the coarse grid, half of the old
# test file's 11 min 45 s). Run by hand:
#
#     julia --project=. examples/urban/berlin_parent_lindenberg_check.jl
#
#   4. z_i at 14 LT against the 12 UTC Lindenberg sounding, θ at 500 m and
#      1 km, the residual layer at 03 LT, the night nudging (Heinze et al.
#      2017, Atmos. Chem. Phys. 17, 7083-7109);
#   6. robustness: 24 h without NaN, no retries, the adaptive dt history.
#
# The "measured" numbers next to the assertions were measured with the
# epic's original subsidence D = 3e-6 s^-1, which then completed on this
# coarse grid. The parent default is now D = 0 (#312); re-measuring these
# numbers at the final D is part of issue #315.

using Test
using Aeolus
using Aeolus.Fields: interior_view
using Printf

isdefined(Main, :BerlinCase) || include(joinpath(@__DIR__, "BerlinCase.jl"))
const BP = Main.BerlinCase

@testset "4 + 6. Lindenberg 29/30 June 2019, 24 h (coarse parent)" begin
    child = BP.CaseConfig(; output_path = nothing)
    pc = BP.ParentConfig(child; refinement = (16, 16, 8), output_period = 1800.0)
    p = BP.build_parent(pc)
    @test (p.mesh.Nx, p.mesh.Ny, p.mesh.Nz) == (24, 24, 38)
    BP.run_parent!(p)
    @test all(isfinite, interior_view(p.sim.θ)) && all(isfinite, interior_view(p.sim.qv))
    @test p.retries == 0
    dts = p.dt_hist
    @info @sprintf("Lindenberg 24 h: %d steps, dt %.2f–%.2f s (mean %.2f), %d retries, %.0f s wall",
                   length(dts), minimum(dts), maximum(dts), sum(dts) / length(dts), p.retries, p.timing[:total])
    @test minimum(dts) >= 0.01 && maximum(dts) <= 10.0

    S = BP.case_sounding(child)
    P = S.profiles
    MT = Aeolus.MoistThermodynamics
    t12 = 12 * 3600.0                     # 14 LT = 12 UTC on the tables' axis (s from 00 UTC; released 10:45)
    θv_obs = [MT.virtual_temperature(BP.forcing_value(P.θ, h, t12), MT.PhasePartition(BP.forcing_value(P.qv, h, t12)))
              for h in P.z]
    zi_obs = boundary_layer_height(P.z, θv_obs; method = :parcel)
    rec(hlt) = p.diag[findfirst(r -> isapprox(r.hour, hlt; atol = 1e-6), p.diag)]
    r14 = rec(14.0)
    err = r14.zi_parcel / zi_obs - 1
    @info @sprintf("z_i(14 LT) parcel %.0f m (min flux %.0f m), 12 UTC sounding %.0f m: %+.1f %%",
                   r14.zi_parcel, r14.zi, zi_obs, 100err)
    @test abs(err) < 0.25           # measured +12.6 % (1332 m against 1183 m)
    zc = BP.level_centres(p.mesh)
    θat(r, h) = (k = searchsortedlast(zc, h); r.θ[k] + (r.θ[k+1] - r.θ[k]) * (h - zc[k]) / (zc[k+1] - zc[k]))
    for h in (500.0, 1000.0)
        d = θat(r14, h) - BP.forcing_value(P.θ, h, t12)
        @info @sprintf("⟨θ⟩(%.0f m, 14 LT) = %.2f K, sounding %.2f K: %+.2f K", h, θat(r14, h),
                       BP.forcing_value(P.θ, h, t12), d)
        @test abs(d) < 1.5          # measured +0.12 K (500 m), -0.17 K (1000 m)
    end
    # residual layer at 03 LT: |dθ/dz| < 1.5 K/km between 300 m and 0.8 z_i,max
    zimax = maximum(r.zi_parcel for r in p.diag)
    r03 = rec(3.0)
    bulk = (θat(r03, 0.8zimax) - θat(r03, 300.0)) / (0.8zimax - 300)
    @info @sprintf("residual layer 03 LT: bulk dθ/dz %.2f K/km over 300-%.0f m; z_i,max %.0f m",
                   1e3bulk, 0.8zimax, zimax)
    @test abs(bulk) < 1.5e-3        # measured 0.26 K/km (z_i,max 1529 m)
    # the night nudging against the daytime surface flux (Heinze et al. 2017)
    night = BP.heat_budget(p.heat; window = (20, 5))
    day = BP.heat_budget(p.heat; window = (10, 16))
    nn = night.terms[(:dynamics, :nudging)] / night.elapsed
    hs = day.terms[(:surface_flux, :tiles)] / day.elapsed
    @info @sprintf("nudging 20-05 LT %.2f W/m², surface 10-16 LT %.1f W/m² (ratio %.3f); 24 h: surface %.1f, radiation %.1f, subsidence %.1f, nudging %.1f, damping %.1f, top %.3f W/m²",
                   nn, hs, nn / hs,
                   (BP.heat_budget(p.heat).terms[k] / 86400 for k in ((:surface_flux, :tiles), (:column_tendency, :column_radiation),
                    (:dynamics, :subsidence), (:dynamics, :nudging), (:dynamics, :damping), (:dynamics, :top_flux)))...)
    # measured: 8.5 W/m² at night against 299 W/m² by day (ratio 0.028). The
    # observed free troposphere warmed by 3.7-6.5 K between the 04:45 UTC
    # launches of the 29th and the 30th (+6.0 K at 2 km, the warm-air
    # advection of the heat wave): over 24 h the nudging (12 W/m²) and the
    # damping (23 W/m²) carry part of that into the box, above 1.2 z_i.
    @test abs(nn) < 0.05 * hs
    hb = BP.heat_budget(p.heat)
    @test abs(hb.closure) < 1e-10 * maximum(abs, values(hb.terms))
end
