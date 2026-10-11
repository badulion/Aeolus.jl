# The Kelvin–Helmholtz billow of kelvin_helmholtz.toml: the numbers
# and the figures. Reads output/results.toml, output/vorticity.nc and
# run.toml only.
#
#   julia --project -m Aeolus run examples/kelvin_helmholtz/kelvin_helmholtz.toml
#   julia --project=. examples/kelvin_helmholtz/plot_kelvin_helmholtz.jl runs/<date>_kelvin_helmholtz_<id>
#
# Writes output/kh_amplitude.svg, kh_gamma.svg, kh_rif.svg and
# kh_vorticity.png (the copies in docs/src/assets/transition/ are these files).

using Printf
include(joinpath(@__DIR__, "..", "viz", "plot_utils.jl"))
include(joinpath(@__DIR__, "..", "viz", "apng.jl"))
include(joinpath(@__DIR__, "..", "viz", "run_output.jl"))

folder = run_folder_arg()
res = read_results(folder)
lay = recorded_config(folder)["layer"]
J, Re, δ = lay["richardson"], lay["re"], lay["delta"]
s = res["series"]
ts, As, γm, Rif = s["t"], s["A"], s["gamma"], s["rif"]
@printf("KH billow J = %.2f, Re = %.0f: %d steps\n", J, Re, res["steps"])
@printf("linear growth rate σ = %.4f U/δ (R² = %.6f over t = %.0f-%.0f δ/U)\n", res["growth_rate"],
        res["growth_rate_r2"], res["growth_window"]...)
@printf("saturation: A_max = %.3f U at t = %.0f δ/U\n", res["A_max"], res["t_saturation"])
for t_ in (40.0, 60.0, 80.0, 100.0, 140.0, 160.0)
    i = argmin(abs.(ts .- t_))
    @printf("t = %5.0f: A = %.2e, γ(|z|<4δ) = %.3f, Ri_f = %.3f\n", ts[i], As[i], γm[i], Rif[i])
end
rc = res["rif_cumulative"]
@printf("cumulative Ri_f from t = %.0f: %.3f to saturation (t = %.0f), %.3f to t = %.0f\n", recorded_config(folder)["diagnostics"]["rif_from"],
        res["rif_cumulative_saturation"], res["t_saturation"], res["rif_cumulative_end"], ts[end])

save_multiline(plot_path(folder, "kh_amplitude.svg"), ts, [As], ["rms w / U"];
               xlabel = "t U / δ", ylabel = "rms w / U (log10)", logy = true,
               title = @sprintf("KH J = %.2f, Re = %.0f: σ = %.3f U/δ", J, Re, res["growth_rate"]))
save_multiline(plot_path(folder, "kh_gamma.svg"), ts, [γm], ["γ, |z| ≤ 4δ"];
               xlabel = "t U / δ", ylabel = "turbulent fraction γ",
               title = "Turbulent fraction of the shear layer")
save_multiline(plot_path(folder, "kh_rif.svg"), rc["t"], [rc["value"]], ["−∫B dt / ∫P dt"];
               xlabel = "t U / δ", ylabel = "cumulative flux Richardson number",
               title = "Cumulative flux Richardson number of the layer")

# spanwise vorticity, the snapshots stacked (|z| < 6δ), blue-white-red
function rgb_panel(ω, kk; lim)
    Z = ω[reverse(kk), :]
    up = 2
    H, W = size(Z, 1) * up, size(Z, 2) * up
    img = Array{UInt8}(undef, H, W, 3)
    for j in 1:H, i in 1:W
        v = clamp(Z[cld(j, up), cld(i, up)] / lim, -1, 1)
        r, gg, b = v >= 0 ? (1.0, 1 - v, 1 - v) : (1 + v, 1 + v, 1.0)
        img[j, i, 1] = round(UInt8, 255r); img[j, i, 2] = round(UInt8, 255gg)
        img[j, i, 3] = round(UInt8, 255b)
    end
    return img
end
ds = NCDataset(joinpath(folder, "output", "vorticity.nc"))
z = Array(ds["z"][:]); tsnap = Array(ds["t"][:]); ω = Array(ds["omega_y"][:, :, :])
close(ds)
kk = findall(zz -> abs(zz) < 6δ, z)
panels = [rgb_panel(permutedims(ω[:, :, s]), kk; lim = 1.0) for s in axes(ω, 3)]
sep = fill(UInt8(0), 4, size(panels[1], 2), 3)
out = plot_path(folder, "kh_vorticity.png")
write_png(out, reduce((a, b) -> vcat(a, sep, b), panels))
println("figures in ", dirname(out), " (vorticity panels at t = ", join(round.(tsnap; digits = 1), ", "), ")")
