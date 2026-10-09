# The T3A-like plate of configs/flat_plate_t3a.toml: the numbers and the
# figures. Reads output/results.toml only.
#
#   JULIA_NUM_THREADS=8 julia --project -m Aeolus run examples/transition/configs/flat_plate_t3a.toml
#   julia --project=. examples/transition/plot_flat_plate_t3a.jl runs/<date>_flat_plate_t3a_<id>
#
# Writes output/t3a_cf.svg, t3a_shape_factor.svg and t3a_tu.svg (the copies
# in docs/src/assets/transition/ are these files).

using Printf
include(joinpath(@__DIR__, "..", "viz", "plot_utils.jl"))
include(joinpath(@__DIR__, "..", "viz", "run_output.jl"))

folder = run_folder_arg()
res = read_results(folder)
p = res["plate"]
@printf("%d steps, wall %.0f s (%.2f s/step)\n", res["steps"], res["wall_seconds"], res["wall_seconds_per_step"])
@printf("free-stream Tu: %.2f %% at the inflow, %.2f %% at Re_x = 1e5, %.2f %% at the plate end\n",
        100res["tu_inflow"], 100res["tu_re_x_1e5"], 100res["tu_plate_end"])
@printf("onset: x = %.1f δ*₀, Re_x = %.0f, Re_θ = %.0f (θ measured; Blasius at that Re_x: %.0f)\n",
        res["onset_x"], res["onset_re_x"], res["onset_re_theta"], res["onset_re_theta_blasius"])
Reθ, Mayle, AGS = res["onset_re_theta"], res["mayle"], res["abu_ghannam_shaw"]
@printf("       Mayle (1991) %.0f  → %+.0f %%;  Abu-Ghannam & Shaw (1980) %.0f → %+.0f %%\n",
        Mayle, 100(Reθ / Mayle - 1), AGS, 100(Reθ / AGS - 1))
if haskey(res, "downstream")
    d = res["downstream"]
    @printf("downstream (Re_x %.0f–%.0f): c_f / 0.0576 Re_x^-1/5 ∈ [%.3f, %.3f], mean %.3f; H ∈ [%.2f, %.2f]\n",
            d["re_x"]..., d["cf_ratio_min"], d["cf_ratio_max"], d["cf_ratio_mean"], d["shape_factor_min"],
            d["shape_factor_max"])
end
R = p["re_x"]
Tu = recorded_config(folder)["disturbance"]["tu"]
save_multiline(plot_path(folder, "t3a_cf.svg"), R ./ 1e5, [1e3 .* p["cf"], 1e3 .* p["cf_laminar"], 1e3 .* p["cf_turbulent"]],
               [@sprintf("Tu = %.1f %%", 100Tu), "laminar", "turbulent"];
               xlabel = "Re_x / 1e5 (laminar 0.664 Re_x^-1/2, turbulent 0.0576 Re_x^-1/5)", ylabel = "1000 c_f",
               title = @sprintf("T3A-like plate: onset Re_θ = %.0f (Mayle %.0f, AGS %.0f)", Reθ, Mayle, AGS))
save_multiline(plot_path(folder, "t3a_shape_factor.svg"), R ./ 1e5, [p["shape_factor"]], ["H = δ*/θ"];
               xlabel = "Re_x / 1e5", ylabel = "shape factor H",
               title = "Shape factor H = δ*/θ (integrated to the δ99.5 edge)")
save_multiline(plot_path(folder, "t3a_tu.svg"), R ./ 1e5, [100 .* p["tu"]], ["u_rms / U∞"];
               xlabel = "Re_x / 1e5", ylabel = "Tu (%)", title = "Free-stream Tu (u_rms above the δ99.5 edge)")
println("figures in ", joinpath(folder, "output"))
