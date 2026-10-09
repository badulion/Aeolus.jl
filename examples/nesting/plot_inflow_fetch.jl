# The development length behind a synthetic inflow, the tables and the
# figures of a run of configs/inflow_fetch.toml. Reads output/results.toml only.
#
#   julia --project -m Aeolus run examples/nesting/configs/inflow_fetch.toml
#   julia --project=. examples/nesting/plot_inflow_fetch.jl runs/<date>_inflow_fetch_<id>
#
# Prints the precursor/reference profiles, the development lengths (δ) of
# every run against its own source and against the child-grid reference,
# and the ratios to the reference at the sample positions; writes
# output/inflow_fetch_{tke,uw}_z<h>.svg (the copies in
# docs/src/assets/nesting/ are these files).

using Printf
include(joinpath(@__DIR__, "..", "viz", "plot_utils.jl"))
include(joinpath(@__DIR__, "..", "viz", "run_output.jl"))

folder = run_folder_arg()
res = read_results(folder)
cfg = recorded_config(folder)["child"]
P = res["profiles"]
@printf("precursor %.0f s, reference %.0f s\n", res["precursor_wall_seconds"], res["reference_wall_seconds"])
println("  z/δ   U_pre  U_ref   TKE_pre TKE_ref  -u'w'_pre -u'w'_ref   τ_pre(z_f)  1 - z_f")
for k in 2:2:length(P["z"])
    @printf("%5.2f  %6.2f %6.2f   %6.3f  %6.3f    %6.3f   %6.3f      %6.3f   %6.3f\n", P["z"][k], P["U_pre"][k],
            P["U_ref"][k], P["tke_pre"][k], P["tke_ref"][k], P["uw_pre"][k], P["uw_ref"][k], P["stress_pre"][k],
            1 - P["z_faces"][k])
end
names = cfg["runs"]
runs = res["runs"]
for n in names
    r = runs[n]
    @printf("%-8s %.0f s: bulk/U_b %.6f, dp/dx bias %+.4f G, KE flux %.1f, div %.1e\n", n, r["wall_seconds"],
            r["bulk_ratio"], r["dpdx_bias"], r["ke_flux"], r["max_div"])
end
nlev = length(cfg["heights"])
println("\nDevelopment lengths (δ, x ≤ Lx - margin) against the run's own source / the child-grid reference:")
println("run       z/δ   TKE 10%        TKE 20%        -u'w' 10%      -u'w' 20%")
for n in names, j in 1:nlev
    l = runs[n]["level_$j"]
    @printf("%-8s %.2f  %5.2f / %5.2f  %5.2f / %5.2f  %5.2f / %5.2f  %5.2f / %5.2f\n", n, l["z"],
            l["L10_tke_source"], l["L10_tke_reference"], l["L20_tke_source"], l["L20_tke_reference"],
            l["L10_uw_source"], l["L10_uw_reference"], l["L20_uw_source"], l["L20_uw_reference"])
end
println("\nValues at x = ", join(res["sample_x"], ", "), "δ (TKE / reference, -u'w' / reference):")
for n in names, j in 1:nlev
    l = runs[n]["level_$j"]
    @printf("%-8s %.2f  TKE %s   -u'w' %s\n", n, l["z"], join((@sprintf("%.2f", v) for v in l["tke_at_sample_x"]), " "),
            join((@sprintf("%.2f", v) for v in l["uw_at_sample_x"]), " "))
end
# the figures: every run at one height, normalised by the child-grid reference
lat = cfg["lateral"] == "open" ? "_open" : ""
for j in 1:nlev, (f, lab) in (("tke", "TKE"), ("uw", "-u'w'"))
    z = runs[names[1]]["level_$j"]["z"]
    x = runs[names[1]]["x"]
    ys = [runs[n]["level_$j"]["$(f)_over_reference"] for n in names]
    push!(ys, ones(length(x)))
    file = plot_path(folder, @sprintf("inflow_fetch_%s_z%02d%s.svg", f, round(Int, 10cfg["heights"][j]), lat))
    save_multiline(file, x, ys, vcat(replace.(names, "df_same" => "DF-same", "df" => "DF", "fsd" => "FSD"), "reference");
                   xlabel = "x / δ (from the inflow plane)", ylabel = "$lab / child-grid periodic reference",
                   title = @sprintf("%s behind the inflow, z = %.2fδ", lab, z), width = 820, height = 460,
                   legend = :bottomright)
    println("wrote ", file)
end
