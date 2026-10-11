# The inflow statistics of the convective parent of
# cbl_inflow_statistics.toml, as a table at 50, 100, 200, 300 and
# 400 m (the nearest levels): the parent's resolved stress against what the
# generator must add (the band k_c < k ≤ 2k_c plus the SGS part).
# Reads output/results.toml only.
#
#   julia --project -m Aeolus run examples/cbl_inflow_statistics/cbl_inflow_statistics.toml
#   julia --project=. examples/cbl_inflow_statistics/report_cbl_inflow_statistics.jl runs/<date>_cbl_inflow_statistics_<id>

using Printf
include(joinpath(@__DIR__, "..", "viz", "run_output.jl"))

folder = run_folder_arg()
res = read_results(folder)
s = res["inflow"]
N = recorded_config(folder)["grid"]["n"]
@printf("ParentResolvedStatistics: %d updates, %.1f ms each (%d × %d plane, %d levels, lag_z = %d)\n",
        s["updates"], s["wall_ms_per_update"], N, N, length(s["z"]), s["lag_z"])
@printf("w* = %.2f m/s, z_i = %.0f m; band: wavelengths %.0f–%.0f m\n", res["w_star"], res["zi"],
        s["band_min"], s["band_max"])
println("   z    σ_u res  σ_u miss (band+SGS)   σ_w res  σ_w miss   u'w' res   u'w' miss   e_SGS   L_z res  L_z band  L_s band")
for z in (50.0, 100.0, 200.0, 300.0, 400.0)
    k = findmin(abs.(s["z"] .- z))[2]
    @printf("%5.0f   %6.3f    %6.3f (%5.3f+%5.3f)    %6.3f    %6.3f   %8.4f   %8.4f   %6.4f   %6.1f   %6.1f   %6.1f\n",
            s["z"][k], sqrt(s["uu_resolved"][k]), sqrt(s["uu_generated"][k]), s["uu_band"][k], s["uu_sgs"][k],
            sqrt(s["ww_resolved"][k]), sqrt(s["ww_generated"][k]), s["uw_resolved"][k], s["uw_generated"][k],
            s["e_sgs"][k], s["L_z_full_u"][k], s["L_z_u"][k], s["L_s_u"][k])
end
println("(σ in m/s; the band and SGS columns of σ_u are variances, m²/s²; lengths of u in m)")
