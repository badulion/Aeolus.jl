# The T3A-like plate of flat_plate_t3a.toml: the numbers and the
# figures. Reads the run's output.nc only: the records of the last 600 δ*₀/U
# (each the mean over its interval) give the mean c_f(x), the mean u(x, z)
# and its rms, and from them (Aeolus.Transition's boundary-layer integrals)
# the onset, the shape factor and the free-stream Tu.
#
#   JULIA_NUM_THREADS=8 julia --project -m Aeolus run examples/flat_plate_t3a/flat_plate_t3a.toml
#   julia --project=. examples/flat_plate_t3a/plot_flat_plate_t3a.jl runs/<date>_flat_plate_t3a_<id> [average]
#
# Writes output/t3a_cf.svg, t3a_shape_factor.svg and t3a_tu.svg (the copies
# in docs/src/assets/transition/ are these files).

using Printf
using Aeolus.Transition: boundary_layer_edge, momentum_thickness, shape_factor, transition_onset,
                         blasius_skin_friction, turbulent_skin_friction, blasius_solution, default_blasius,
                         onset_reynolds_theta_mayle, onset_reynolds_theta_abu_ghannam_shaw
include(joinpath(@__DIR__, "..", "viz", "plot_utils.jl"))
include(joinpath(@__DIR__, "..", "viz", "run_output.jl"))

folder = run_folder_arg()
average = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 600.0    # δ*₀/U: the statistics' window
dom = recorded_domain(folder)
fr = dom["fringe"]; ν = dom["dynamics"]["nu"]; U∞ = fr["velocity"]
Tu = fr["tu"]; Lx = dom["grid"]["extent"][1]
path = output_file(folder)
ds = NCDataset(path)
t = Array(ds["time"][:]); x0 = Array(ds["x"][:]); dz = ds["z"][2] - ds["z"][1]
recs = findall(>(t[end] - average * (1 + 1e-9)), t)
mean_of(name) = sum(Float64.(ds[name][ntuple(_ -> Colon(), ndims(ds[name]) - 1)..., r]) for r in recs) ./ length(recs)
cfmap = mean_of("skin_friction"); um = mean_of("u"); u2 = mean_of("u_squared")
close(ds)
cf = vec(sum(cfmap; dims = 2)) ./ size(cfmap, 2)
ū = dropdims(sum(um; dims = 2); dims = 2) ./ size(um, 2)
urms = sqrt.(max.(dropdims(sum(u2; dims = 2); dims = 2) ./ size(u2, 2) .- ū .^ 2, 0.0))
@printf("statistics over t = %.0f–%.0f (%d records)\n", t[recs[1]] - (t[2] - t[1]), t[end], length(recs))

# the analysis of the averaged layer (the plate starts at the inflow x = 0,
# the Blasius layer at x_in from the virtual leading edge)
x_in = (fr["inflow_reynolds"] / default_blasius().delta_star)^2 * ν / U∞
x = x0 .- 0.5 * (x0[2] - x0[1])                         # the u faces
Re_x = (x .+ x_in) .* U∞ ./ ν
Nz = size(ū, 2)
plate = findall(<(Lx - fr["length"] - 20.0), x)          # 20 δ*₀ clear of the fringe rise
ke = boundary_layer_edge(ū; fraction = 0.995)
Ue = vec(maximum(ū; dims = 2))
θ = momentum_thickness(ū, dz; U_e = Ue, kmax = ke)
H = shape_factor(ū, dz; U_e = Ue, kmax = ke)
fs(i) = (r = (ke[i] + 1):(Nz - 4); isempty(r) ? ((Nz - 9):(Nz - 4)) : r)
Tu_x = [sum(urms[i, k] for k in fs(i)) / length(fs(i)) for i in eachindex(x)]
xo, io = transition_onset(x[plate], cf[plate], blasius_skin_friction.(Re_x[plate]))
Reθ = io == 0 ? NaN : θ[io] * U∞ / ν
Mayle = onset_reynolds_theta_mayle(100Tu); AGS = onset_reynolds_theta_abu_ghannam_shaw(100Tu)
i5 = argmin(abs.(Re_x .- 1e5))
@printf("free-stream Tu: %.2f %% at the inflow, %.2f %% at Re_x = 1e5, %.2f %% at the plate end\n",
        100Tu_x[1], 100Tu_x[i5], 100Tu_x[plate[end]])
@printf("onset: x = %.1f δ*₀, Re_x = %.0f, Re_θ = %.0f (θ measured; Blasius at that Re_x: %.0f)\n",
        xo, io == 0 ? NaN : Re_x[io], Reθ, io == 0 ? NaN : 2 * blasius_solution().fpp0 * sqrt(Re_x[io]))
@printf("       Mayle (1991) %.0f  → %+.0f %%;  Abu-Ghannam & Shaw (1980) %.0f → %+.0f %%\n",
        Mayle, 100(Reθ / Mayle - 1), AGS, 100(Reθ / AGS - 1))
down = [i for i in plate if Re_x[i] >= 1.2e5]
if !isempty(down)
    ratio = cf[down] ./ turbulent_skin_friction.(Re_x[down])
    @printf("downstream (Re_x %.0f–%.0f): c_f / 0.0576 Re_x^-1/5 ∈ [%.3f, %.3f], mean %.3f; H ∈ [%.2f, %.2f]\n",
            Re_x[down[1]], Re_x[down[end]], minimum(ratio), maximum(ratio), sum(ratio) / length(ratio),
            minimum(H[down]), maximum(H[down]))
end
R = Re_x[plate]
save_multiline(plot_path(folder, "t3a_cf.svg"), R ./ 1e5,
               [1e3 .* cf[plate], 1e3 .* blasius_skin_friction.(R), 1e3 .* turbulent_skin_friction.(R)],
               [@sprintf("Tu = %.1f %%", 100Tu), "laminar", "turbulent"];
               xlabel = "Re_x / 1e5 (laminar 0.664 Re_x^-1/2, turbulent 0.0576 Re_x^-1/5)", ylabel = "1000 c_f",
               title = @sprintf("T3A-like plate: onset Re_θ = %.0f (Mayle %.0f, AGS %.0f)", Reθ, Mayle, AGS))
save_multiline(plot_path(folder, "t3a_shape_factor.svg"), R ./ 1e5, [H[plate]], ["H = δ*/θ"];
               xlabel = "Re_x / 1e5", ylabel = "shape factor H",
               title = "Shape factor H = δ*/θ (integrated to the δ99.5 edge)")
save_multiline(plot_path(folder, "t3a_tu.svg"), R ./ 1e5, [100 .* Tu_x[plate]], ["u_rms / U∞"];
               xlabel = "Re_x / 1e5", ylabel = "Tu (%)", title = "Free-stream Tu (u_rms above the δ99.5 edge)")
println("figures in ", joinpath(folder, "output"))
