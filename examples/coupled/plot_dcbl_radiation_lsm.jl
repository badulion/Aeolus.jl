# Coupled example 01, the plots: the slab surface temperature, the
# boundary-layer height (the height of the most negative resolved heat
# flux ⟨w'θ'⟩, the entrainment flux; meaningful once the floor heats the air) and the
# volume-mean kinetic energy of a run of configs/dcbl_radiation_lsm.toml.
# Reads diagnostics.csv and diagnostics.nc only.
#
#   julia --project -m Aeolus run examples/coupled/configs/dcbl_radiation_lsm.toml
#   julia --project=. examples/coupled/plot_dcbl_radiation_lsm.jl runs/<date>_dcbl_radiation_lsm_<id>
#
# Writes output/dcbl_Tsoil.svg, dcbl_zi.svg and dcbl_KE.svg in the run folder.

using Printf
include(joinpath(@__DIR__, "..", "viz", "plot_utils.jl"))
include(joinpath(@__DIR__, "..", "viz", "run_output.jl"))

folder = run_folder_arg()
d = read_diagnostics_csv(folder)
t, Ts = diagnostic_series(d, "surface.T_s"); _, H = diagnostic_series(d, "surface.H")
_, ke = diagnostic_series(d, "ke_mean"); _, div = diagnostic_series(d, "max_div")
z, WT, tc = read_profile(folder, "w_theta")
zi = [z[argmin(WT[:, n])] for n in axes(WT, 2)]
a = save_multiline(plot_path(folder, "dcbl_Tsoil.svg"), t ./ 3600, [Ts], ["T_s (K)"];
                   xlabel = "time (h)", ylabel = "T (K)", title = "DCBL: slab surface temperature")
b = save_multiline(plot_path(folder, "dcbl_zi.svg"), tc ./ 3600, [zi], ["z_i (m)"];
                   xlabel = "time (h)", ylabel = "height (m)", title = "DCBL: height of the minimum heat flux")
c = save_multiline(plot_path(folder, "dcbl_KE.svg"), t ./ 3600, [ke], ["KE"];
                   xlabel = "time (h)", ylabel = "KE (m² s⁻²)", title = "DCBL: mean resolved kinetic energy")
@printf("t = %.2f h: T_s = %.2f K, H = %.1f W/m², z_i ≈ %.0f m, KE = %.2e m²/s², max|div| = %.1e\n",
        t[end] / 3600, Ts[end], H[end], zi[end], ke[end], div[end])
println.((a, b, c))
