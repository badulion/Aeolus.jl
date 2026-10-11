# =========================================================================
# Decaying isotropic turbulence: LES against DNS, a posteriori (ONE-g, #456)
#
# The runs are the configurations of examples/dns_vs_les/: the
# 256³ DNS (dns.toml) and the 64³ LES of the SGS models (les_<model>.toml),
# all started from the same isotropic field ([initial] field: the LES
# start is the DNS start truncated to its modes). This script only reads
# their output (output/output.nc: the resolved kinetic energy and the
# dissipation every 0.05 s, the energy spectrum E(k) every second) and
#
#   * fits the decay exponent n of K(t) ∝ t^(-n) over t ∈ [T0, T1]
#     (after the dissipation peak; default 4-8 s) for every run,
#   * compares the LES spectra with the DNS spectrum at t = 2, 4 and 8 s:
#     the RMS of log10(E_LES / E_DNS) over the shells 1 ≤ k ≤ k_c / 2 (the
#     well-resolved LES range) and 1 ≤ k ≤ k_c (k_c = 32, the LES cut-off),
#   * compares K and the dissipation ε (molecular + SGS) with the DNS: the
#     largest relative K difference over the run, the peak of ε and its time,
#
# prints the table and writes examples/dns_vs_les/output/dns_vs_les.svg
# and dns_vs_les.csv.
#
#   julia --project -m Aeolus run examples/dns_vs_les/dns.toml
#   for m in smagorinsky dynamic_smagorinsky amd vreman sigma; do
#       julia --project -m Aeolus run examples/dns_vs_les/les_$m.toml; done
#   julia --project examples/dns_vs_les/compare.jl runs/*_dns_* runs/*_les_*
#
# The first folder whose run is named "dns" is the reference.
# =========================================================================

using TOML
using Printf
using Aeolus.NCIO: NCDataset
include(joinpath(@__DIR__, "..", "sbl", "svgplot.jl"))

const FIT = (4.0, 8.0)                 # the window of the decay-exponent fit (s)
const TIMES = (2.0, 4.0, 8.0)          # the times of the spectra comparison (s)

"What a run folder holds: its name, the series t, K, ε, ε_sgs and the spectra (k, the record times, E[k, record])."
function read_run(folder)
    cfg = TOML.parsefile(joinpath(folder, "run.toml"))["config"]
    NCDataset(joinpath(folder, "output", "output.nc")) do ds
        (name = cfg["run"]["name"], n = ds.dim["x"], t = Float64.(ds["time_series"][:]),
         K = Float64.(ds["kinetic_energy"][:]), ε = Float64.(ds["dissipation"][:]),
         εs = Float64.(ds["dissipation_sgs"][:]), k = Float64.(ds["k"][:]), tr = Float64.(ds["time"][:]),
         E = Float64.(ds["energy_spectrum"][:, :]))
    end
end

"The decay exponent n of K ∝ t^(-n) over the window (least squares in log-log)."
function decay_exponent(r, (t0, t1))
    sel = findall(t -> t0 <= t <= t1 * (1 + 1e-9) && t > 0, r.t)
    x = log.(r.t[sel]); y = log.(r.K[sel])
    xm = sum(x) / length(x); ym = sum(y) / length(y)
    return -sum((x .- xm) .* (y .- ym)) / sum(abs2, x .- xm)
end

"The record of `r` at time `t` (the nearest)."
record(r, t) = argmin(abs.(r.tr .- t))

"RMS of log10(E_les / E_dns) over the shells 1 ≤ k ≤ kmax at time t."
function spectrum_mismatch(les, dns, t, kmax)
    a = les.E[:, record(les, t)]; b = dns.E[:, record(dns, t)]
    ns = [n for n in 2:min(length(a), length(b)) if les.k[n] <= kmax && a[n] > 0 && b[n] > 0]
    return sqrt(sum(abs2, log10.(a[ns] ./ b[ns])) / length(ns))
end

runs = [read_run(a) for a in ARGS]
isempty(runs) && error("usage: julia --project examples/dns_vs_les/compare.jl <dns run folder> <les run folders>…")
iref = findfirst(r -> r.name == "dns", runs)
iref === nothing && error("no run named \"dns\" among the folders")
dns = runs[iref]
les = [r for (i, r) in enumerate(runs) if i != iref]
kc = les[1].n / 2 * les[1].k[2]          # the LES cut-off π/Δ (the shells up to the corner of its cube go beyond it)

"The K of `r` interpolated at the series times of the DNS."
K_at(r, t) = (i = clamp(searchsortedlast(r.t, t), 1, length(r.t) - 1);
              r.K[i] + (r.K[i+1] - r.K[i]) * (t - r.t[i]) / (r.t[i+1] - r.t[i]))

rows = []
for r in runs
    n = decay_exponent(r, FIT)
    ip = argmax(r.ε)
    dK = r === dns ? 0.0 : maximum(abs(K_at(r, t) / K_at(dns, t) - 1) for t in dns.t if t <= min(r.t[end], dns.t[end]))
    mm = r === dns ? fill(NaN, 2 * length(TIMES)) :
         vcat([[spectrum_mismatch(r, dns, t, kc / 2), spectrum_mismatch(r, dns, t, kc)] for t in TIMES]...)
    push!(rows, (name = r.name, n = n, εpeak = r.ε[ip], tpeak = r.t[ip], dK = dK, mm = mm,
                 εs_share = r.ε[ip] > 0 ? r.εs[ip] / r.ε[ip] : 0.0))
end

@printf("\nDecaying isotropic turbulence, LES (%d³) against the DNS (%d³); decay exponent over t = %g-%g s\n",
        les[1].n, dns.n, FIT...)
@printf("%-24s %6s %9s %7s %8s %9s   %s\n", "run", "n", "ε_peak", "t_peak", "SGS/ε", "max|ΔK|/K",
        join(("E t=$(Int(t)) (k≤$(Int(kc ÷ 2)), k≤$(Int(kc)))" for t in TIMES), "  "))
for r in rows
    @printf("%-24s %6.3f %9.4f %7.2f %8.3f %9.3f   %s\n", r.name, r.n, r.εpeak, r.tpeak, r.εs_share, r.dK,
            join((@sprintf("%.3f, %.3f", r.mm[2i-1], r.mm[2i]) for i in eachindex(TIMES)), "        "))
end
println("(spectra: RMS of log10(E_LES / E_DNS) over the shells; 0.1 is a factor 1.26)")

outdir = mkpath(joinpath(@__DIR__, "output"))
open(joinpath(outdir, "dns_vs_les.csv"), "w") do io
    println(io, "# DNS against LES of decaying isotropic turbulence (examples/dns_vs_les). Model output, not reference data.")
    println(io, "run,grid,decay_exponent,eps_peak,t_peak,sgs_share_at_peak,max_rel_K_difference," *
                join(("rms_log10_E_t$(Int(t))_khalf,rms_log10_E_t$(Int(t))_kc" for t in TIMES), ","))
    for (r, x) in zip(rows, runs)
        @printf(io, "%s,%d,%.4f,%.5f,%.3f,%.4f,%.4f,%s\n", r.name, x.n, r.n, r.εpeak, r.tpeak, r.εs_share, r.dK,
                join((@sprintf("%.4f", v) for v in r.mm), ","))
    end
end

# ---- the figure: K(t), ε(t), and the spectra at the three times ----------------
fig = svg_figure(1180, 820, "Decaying isotropic turbulence: LES ($(les[1].n)³) against DNS ($(dns.n)³)")
leg = [(r.name, i, "") for (i, r) in enumerate(runs)]
tmax = dns.t[end]
svg_panel(fig, 80, 50, 300, 280; xlim = (0.0, tmax), ylim = (0.0, 2.0), xticks = 0:2:tmax, yticks = 0:0.5:2,
          xlabel = "t (s)", ylabel = "resolved K (m²/s²)", tag = "(a)",
          curves = [(r.t, r.K, i) for (i, r) in enumerate(runs)], legend = leg, legend_x = 120)
svg_panel(fig, 460, 50, 300, 280; xlim = (0.0, tmax), ylim = (0.0, 0.8), xticks = 0:2:tmax, yticks = 0:0.2:0.8,
          xlabel = "t (s)", ylabel = "ε, molecular + SGS (m²/s³)", tag = "(b)",
          curves = [(r.t, r.ε, i) for (i, r) in enumerate(runs)])
svg_panel(fig, 840, 50, 300, 280; xlim = (log10(0.5), log10(tmax)), ylim = (-1.0, 0.5), xticks = [0.0, 0.25, 0.5, 0.75],
          yticks = -1:0.5:0.5, xlabel = "log10 t", ylabel = "log10 K", tag = "(c)",
          curves = [(log10.(r.t[2:end]), log10.(r.K[2:end]), i) for (i, r) in enumerate(runs)],
          vbands = [(log10(FIT[1]), log10(FIT[2]))])
for (j, t) in enumerate(TIMES)
    svg_panel(fig, 80 + (j - 1) * 380, 440, 300, 280; xlim = (0.0, log10(dns.k[end])), ylim = (-8.0, 1.0),
              xticks = 0:0.5:2.5, yticks = -8:2:0, xlabel = "log10 k (rad/m)", ylabel = "log10 E(k)",
              tag = "($('d' + j - 1)) t = $(Int(t)) s",
              curves = [(log10.(r.k[2:end]), log10.(max.(r.E[2:end, record(r, t)], 1e-12)), i) for (i, r) in enumerate(runs)],
              vbands = [(log10(kc), log10(kc) + 0.01)])
end
svg_text(fig, 80, 790, "(c) grey: the window of the decay-exponent fit; (d-f) the thin line: the LES cut-off k_c = $(Int(kc)).", size = 11)
println("wrote ", svg_write(fig, joinpath(outdir, "dns_vs_les.svg")))
