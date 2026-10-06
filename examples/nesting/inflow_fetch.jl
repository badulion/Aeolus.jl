# =========================================================================
# The development length behind a synthetic inflow — N-inflow-d (#266)
#
# How far into an open-boundary child is the inflow turbulence still out of
# equilibrium, and how much does the generator shorten that distance?
#
#   precursor    the periodic, neutral, wall-modelled half channel of P54-d
#                (`half_channel_case`, δ = u* = 1, z0 = 1e-4 δ,
#                StaticSmagorinsky(0.1)) on 5δ × 2.5δ × δ at the parent
#                resolution Δx_p = 0.208δ, Δz = δ/16 (24 × 12 × 16): the
#                statistics source (ParentResolvedStatistics, :full) and the
#                mean inflow profile
#   reference    the same periodic channel on the child grid
#                (48 × 24 × 16): the child grid's own equilibrium
#   child        the channel opened in x, refine 2 horizontally
#                (Δx = 0.104δ), `Lx` long (default 15δ, beyond the 4–6δ of
#                the CI case, because the fetch is longer than that), with
#                  off       no generator (the precursor's mean profile only)
#                  DF        the digital filter (Xie & Castro 2008), the
#                            parent's statistics
#                  FSD       the random-mode FreeStreamInflow provider
#                  DF-same   the digital filter with the reference's
#                            (child-grid) statistics and mean: the classical
#                            set-up of Xie & Castro (2008) and Kim et al. (2013)
#
# Writes TKE(x) and −u'w'(x) at 0.2δ and 0.5δ (normalised by the
# precursor and by the reference) to docs/src/assets/nesting/ and prints
# the development lengths (10 % and 20 %).
#
#   julia --project=. examples/nesting/inflow_fetch.jl [Lx] [t_avg]
#
# Cost (24 cores shared, one thread): about 3 min per 15δ child run with
# t_avg = 10 δ/u*, 1 min for the precursors.
# =========================================================================

using Aeolus
using Printf
include(joinpath(@__DIR__, "..", "..", "test", "inflow_fetch_harness.jl"))
include(joinpath(@__DIR__, "..", "viz", "plot_utils.jl"))

Lx = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 15.0
t_avg = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 10.0
Ly = 2.5
Np, Nr = (24, 12, 16), (48, 24, 16)
Nc = (round(Int, 48 * Lx / 5), 24, 16)
outdir = joinpath(@__DIR__, "..", "..", "docs", "src", "assets", "nesting")
mkpath(outdir)

pre = fetch_precursor(; N = Np, extent = (5.0, Ly), t_spin = 20.0, t_avg = 10.0)
ref = fetch_precursor(; N = Nr, extent = (5.0, Ly), t_spin = 20.0, t_avg = 10.0, seed = 2)
@printf("precursor %.0f s, reference %.0f s\n", pre.wall_time, ref.wall_time)
println("  z/δ   U_pre  U_ref   TKE_pre TKE_ref  -u'w'_pre -u'w'_ref   τ_pre(z_f)  1 - z_f")
for k in 2:2:16
    @printf("%5.2f  %6.2f %6.2f   %6.3f  %6.3f    %6.3f   %6.3f      %6.3f   %6.3f\n", pre.z[k],
            pre.U[k], ref.U[k], pre.tke[k], ref.tke[k], pre.uw[k], ref.uw[k], pre.stress[k],
            1 - pre.z_faces[k])
end

runs = [(:off, "off", pre), (:df, "DF", pre), (:fsd, "FSD", pre), (:df, "DF-same", ref)]
res = Dict{String,Any}()
for (g, name, src) in runs
    r = fetch_child(src; generator = g, N = Nc, extent = (Lx, Ly), t_spin = 2.0,
                    t_avg = t_avg, sample_every = 4)
    res[name] = (r, src)
    @printf("%-8s %.0f s: bulk/U_b %.6f, dp/dx bias %+.4f G, KE flux %.1f, div %.1e\n", name,
            r.wall_time, r.bulk[1] / (sum(src.U) / 16), pressure_gradient_bias(r, src.case.forcing),
            r.ke_flux, r.maxdiv)
end

ks = [nearest_level(pre.z, h) for h in (0.2, 0.5)]
# the last 0.5δ is the convective outflow layer (the columns next to the
# east face lose variance to the radiation condition): not part of the fetch
keep = findall(x -> x <= Lx - 0.5, res["DF"][1].p.x)
println("\nDevelopment lengths (δ, x ≤ Lx - 0.5δ) against the run's own source / the child-grid reference:")
println("run       z/δ   TKE 10%        TKE 20%        -u'w' 10%      -u'w' 20%")
for (_, name, src) in runs
    r, s = res[name]
    for k in ks
        L(f, T, fr) = development_length((r.p.x[keep], getfield(r.p, f)[keep, k]), T; x0 = 0.0,
                                         frac = fr)
        @printf("%-8s %.2f  %5.2f / %5.2f  %5.2f / %5.2f  %5.2f / %5.2f  %5.2f / %5.2f\n", name, pre.z[k],
                L(:tke, s.tke[k], 0.9), L(:tke, ref.tke[k], 0.9), L(:tke, s.tke[k], 0.8),
                L(:tke, ref.tke[k], 0.8), L(:uw, s.uw[k], 0.9), L(:uw, ref.uw[k], 0.9),
                L(:uw, s.uw[k], 0.8), L(:uw, ref.uw[k], 0.8))
    end
end

println("\nValues at x = 1, 3, 5, 10, 14δ (TKE / reference, -u'w' / reference):")
for (_, name, _) in runs, k in ks
    r = res[name][1]
    @printf("%-8s %.2f  TKE %s   -u'w' %s\n", name, pre.z[k],
            join((@sprintf("%.2f", r.p.tke[nearest_level(r.p.x, x), k] / ref.tke[k]) for x in (1, 3, 5, 10, 14) if x < Lx), " "),
            join((@sprintf("%.2f", r.p.uw[nearest_level(r.p.x, x), k] / ref.uw[k]) for x in (1, 3, 5, 10, 14) if x < Lx), " "))
end

# figures: the four runs at one height, normalised by the child-grid reference
for (k, h) in zip(ks, (0.2, 0.5)), (f, lab) in ((:tke, "TKE"), (:uw, "-u'w'"))
    x = res["DF"][1].p.x
    ys = [getfield(res[n][1].p, f)[:, k] ./ getfield(ref, f)[k] for (_, n, _) in runs]
    push!(ys, ones(length(x)))
    names = [n for (_, n, _) in runs]
    push!(names, "reference")
    file = joinpath(outdir, @sprintf("inflow_fetch_%s_z%02d.svg", f === :tke ? "tke" : "uw", round(Int, 10h)))
    save_multiline(file, x, ys, names; xlabel = "x / δ (from the inflow plane)",
                   ylabel = "$lab / child-grid periodic reference",
                   title = @sprintf("%s behind the inflow, z = %.2fδ", lab, pre.z[k]),
                   width = 820, height = 460, legend = :bottomright)
    println("wrote ", file)
end
