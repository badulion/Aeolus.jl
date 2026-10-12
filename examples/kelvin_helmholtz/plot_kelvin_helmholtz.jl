# The Kelvin–Helmholtz billow of kelvin_helmholtz.toml: the numbers
# and the figures. Reads the run's output.nc and run.toml only: the rms of
# w every δ/U (the amplitude A), and per record the turbulent fraction γ
# and the shear production P and buoyancy flux B of the layer
# |z − z_c| < 4δ, whose ratio −B/P is the flux Richardson number, and ω_y.
#
#   julia --project -m Aeolus run examples/kelvin_helmholtz/kelvin_helmholtz.toml
#   julia --project=. examples/kelvin_helmholtz/plot_kelvin_helmholtz.jl runs/<date>_kelvin_helmholtz_<id>
#
# Writes output/kh_amplitude.svg, kh_gamma.svg, kh_rif.svg and
# kh_vorticity.png (the copies in docs/src/assets/transition/ are these files).

using Printf
using Aeolus.Transition: growth_rate
include(joinpath(@__DIR__, "..", "viz", "plot_utils.jl"))
include(joinpath(@__DIR__, "..", "viz", "apng.jl"))
include(joinpath(@__DIR__, "..", "viz", "run_output.jl"))

folder = run_folder_arg()
dom = recorded_domain(folder)
u0 = dom["initial"]["u"]; dyn = dom["dynamics"]
δ, U, zc = u0["thickness"], u0["amplitude"], u0["center"]
Δθ = dom["initial"]["theta"]["amplitude"]
J = dyn["g"] * Δθ * δ / (dyn["theta0"] * U^2); Re = U * δ / dyn["nu"]
const LAYER = 4.0          # the layer |z − z_c| < LAYER δ
const RIF_FROM = 10.0      # the cumulative Ri_f from this time (after the seed's adjustment)
const SNAPSHOTS = (60.0, 90.0, 140.0)
ds = NCDataset(output_file(folder))
z = Array(ds["z"][:]); x = Array(ds["x"][:])
tser = Array(ds["time_series"][:]); As = Array(ds["w_rms"][:])
trec = Array(ds["time"][:])
layer = findall(zz -> abs(zz - zc) < LAYER * δ, z)
γ = Array(ds["turbulent_fraction_profile"][:, :]); P = Array(ds["shear_production_profile"][:, :])
B = Array(ds["buoyancy_production_profile"][:, :])
snap = [argmin(abs.(trec .- s)) for s in SNAPSHOTS if s <= trec[end]]
isempty(snap) && (snap = [length(trec)])          # a short run: its last record
ω = cat((Array(ds["vorticity_y"][:, 1, :, r]) for r in snap)...; dims = 3); tsnap = trec[snap]
close(ds)
ts = trec
γm = [sum(γ[layer, r]) / length(layer) for r in eachindex(trec)]
Bs = [sum(B[layer, r]) for r in eachindex(trec)]; Ps = [sum(P[layer, r]) for r in eachindex(trec)]
Rif = [p == 0 ? NaN : -b / p for (b, p) in zip(Bs, Ps)]
fit = try                  # a short run has no exponential stage: NaN
    growth_rate(tser, As)
catch err
    (err isa ErrorException && startswith(err.msg, "growth_rate: no window")) || rethrow()
    (rate = NaN, r2 = NaN, window = (NaN, NaN))
end
ipk = argmax(As)
i10 = findfirst(>=(RIF_FROM), trec)
Rif_run = -cumsum(Bs[i10:end]) ./ cumsum(Ps[i10:end])
isat = argmin(abs.(trec .- tser[ipk]))
@printf("KH billow J = %.2f, Re = %.0f\n", J, Re)
@printf("linear growth rate σ = %.4f U/δ (R² = %.6f over t = %.0f-%.0f δ/U)\n", fit.rate, fit.r2, fit.window...)
@printf("saturation: A_max = %.3f U at t = %.0f δ/U\n", As[ipk], tser[ipk])
for t_ in (40.0, 60.0, 80.0, 100.0, 140.0, 160.0)
    i = argmin(abs.(trec .- t_)); j = argmin(abs.(tser .- t_))
    @printf("t = %5.0f: A = %.2e, γ(|z|<4δ) = %.3f, Ri_f = %.3f\n", trec[i], As[j], γm[i], Rif[i])
end
@printf("cumulative Ri_f from t = %.0f: %.3f to saturation (t = %.0f), %.3f to t = %.0f\n", RIF_FROM,
        Rif_run[max(isat - i10 + 1, 1)], tser[ipk], Rif_run[end], trec[end])
rc = Dict("t" => trec[i10:end], "value" => Rif_run)
res = Dict("growth_rate" => fit.rate)

save_multiline(plot_path(folder, "kh_amplitude.svg"), tser, [As], ["rms w / U"];
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
kk = findall(zz -> abs(zz - zc) < 6δ, z)
panels = [rgb_panel(permutedims(ω[:, :, s]), kk; lim = 1.0) for s in axes(ω, 3)]
sep = fill(UInt8(0), 4, size(panels[1], 2), 3)
out = plot_path(folder, "kh_vorticity.png")
write_png(out, reduce((a, b) -> vcat(a, sep, b), panels))
println("figures in ", dirname(out), " (vorticity panels at t = ", join(round.(tsnap; digits = 1), ", "), ")")
