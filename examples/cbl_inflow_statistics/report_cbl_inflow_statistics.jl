# The inflow statistics of the convective parent of
# cbl_inflow_statistics.toml, as a table at 50, 100, 200, 300 and
# 400 m (the nearest levels): the parent's resolved stress against what the
# generator must add (the band k_c < k ≤ 2k_c plus the SGS part).
# Reads the last record of the run's output.nc only.
#
#   julia --project -m Aeolus run examples/cbl_inflow_statistics/cbl_inflow_statistics.toml
#   julia --project=. examples/cbl_inflow_statistics/report_cbl_inflow_statistics.jl runs/<date>_cbl_inflow_statistics_<id>

using Printf
include(joinpath(@__DIR__, "..", "viz", "run_output.jl"))

folder = run_folder_arg()
dom = recorded_domain(folder)
path = output_file(folder)
s, t = read_last_record(path)
z = NCDataset(path) do ds; Array(ds["z"][:]); end
g = dom["grid"]; Δh = max(g["extent"][1] / g["size"][1], g["extent"][2] / g["size"][2])
st = dom["inflow_statistics"]
r = min(st["refine"][1], st["refine"][2])
@printf("ParentResolvedStatistics at t = %.0f s (from %.0f s, τ_avg = %.0f s), %d × %d plane; band: wavelengths %.0f–%.0f m\n",
        t, st["start"], st["tau_avg"], g["size"][1], g["size"][2], 4Δh / r, 4Δh)
println("   z    σ_u res  σ_u miss (band+SGS)   σ_w res  σ_w miss   u'w' res   u'w' miss   e_SGS   L_z res  L_z band  L_s band")
for zq in (50.0, 100.0, 200.0, 300.0, 400.0)
    k = findmin(abs.(z .- zq))[2]
    isfinite(s["inflow_uu_resolved"][k]) || continue          # above the statistics' depth
    @printf("%5.0f   %6.3f    %6.3f (%5.3f+%5.3f)    %6.3f    %6.3f   %8.4f   %8.4f   %6.4f   %6.1f   %6.1f   %6.1f\n",
            z[k], sqrt(s["inflow_uu_resolved"][k]), sqrt(s["inflow_uu_generated"][k]), s["inflow_uu_band"][k],
            s["inflow_uu_sgs"][k], sqrt(s["inflow_ww_resolved"][k]), sqrt(s["inflow_ww_generated"][k]),
            s["inflow_uw_resolved"][k], s["inflow_uw_generated"][k], s["inflow_e_sgs"][k], s["inflow_L_z_full_u"][k],
            s["inflow_L_z_u"][k], s["inflow_L_s_u"][k])
end
println("(σ in m/s; the band and SGS columns of σ_u are variances, m²/s²; lengths of u in m)")
