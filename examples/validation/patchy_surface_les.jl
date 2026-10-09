# =========================================================================
# Patchy surface heating — resolved LES against tiled aggregation
#
# Phase 53 (#43). A convective boundary layer over strips of hot
# (310 K) and cool (300 K) ground, 50 % each, z0 = 0.1 m, z0h = z0/10,
# initial mixed layer 300 K to 600 m. Every surface column computes its
# own Monin–Obukhov fluxes through the SurfaceExchange contract (the
# `patchy_surface` run type, src/validation_runs/patchy_surface.jl). Two
# sets of runs:
#
#  A. 50 m resolution, 3.2 × 1.6 × 1.6 km, 2 h (average over the second):
#     resolved strips (λ = 0.8 and 3.2 km), an LES with the tiled mosaic
#     in every column, and the uniform mean-temperature surface (the
#     parameter aggregate) — calm, and with a 5 m/s geostrophic wind
#     across the strips (f = 1e-4 s⁻¹).
#  B. 100 m resolution, 12.8 × 1.6 × 1.6 km, 3 h (average over the last
#     1.5 h): λ = 0.8 … 12.8 km, i.e. λ/z_i ≈ 0.8 … 11, for the
#     organised-motion comparison with Patton, Sullivan & Moeng (2005).
#
# The thirteen runs are the sweep examples/validation/configs/
# patchy_surface.toml (run them one after the other, or one process per
# member, `--member A_tiled`, in parallel). This script reads their
# output/results.toml, prints the tables and writes SVG figures to
# examples/validation/output/patchy_surface/:
#
#   julia --project -m Aeolus run examples/validation/configs/patchy_surface.toml
#   julia --project examples/validation/patchy_surface_les.jl [runs-root]
#
# References
#   Patton, E. G., Sullivan, P. P. & Moeng, C.-H. (2005). The influence of
#     idealized heterogeneity on wet and dry planetary boundary layers
#     coupled to the land surface. J. Atmos. Sci., 62, 2078-2097.
#     (Conference abstract with the 4 < λ/z_i < 9 statement: NASA NTRS
#     20030068489.)
#   Wanner, L., Calaf, M. & Mauder, M. (2022). Incorporating the effect of
#     heterogeneous surface heating into a semi-empirical model of the
#     surface energy balance closure. PLOS ONE, 17, e0268097.
#   Beljaars, A. C. M. (1995). The parametrization of surface fluxes in
#     large-scale models under free convection. Q. J. R. Meteorol. Soc.,
#     121, 255-270.
# =========================================================================

using Printf
include(joinpath(@__DIR__, "read_runs.jl"))
include(joinpath(@__DIR__, "..", "viz", "plot_utils.jl"))

const OUT = joinpath(@__DIR__, "output", "patchy_surface")

"Root-mean-square about the mean of a phase-averaged profile."
patchy_amplitude(x) = sqrt(sum(abs2, x .- sum(x) / length(x)) / length(x))
mkpath(OUT)

# member label => the name the tables print
A_cases = [("A_uniform", "uniform 305 K (= parameter aggregate)"),
           ("A_strips_800", "strips λ = 0.8 km"),
           ("A_strips_3200", "strips λ = 3.2 km"),
           ("A_tiled", "tiled mosaic in every column"),
           ("A_windy_uniform", "windy: uniform 305 K"),
           ("A_windy_strips_3200", "windy: strips λ = 3.2 km"),
           ("A_windy_tiled", "windy: tiled mosaic")]
B_λ = [800.0, 1600.0, 3200.0, 6400.0, 12800.0]
B_cases = [[("B_strips_$(Int(λ))", "sweep λ = $(Int(λ)) m") for λ in B_λ]; ("B_uniform", "sweep uniform 305 K")]

runs = member_runs(runs_root(), validation_config("patchy_surface"))
name_of = Dict(vcat(A_cases, B_cases))
"The results of a run as a NamedTuple (vectors as Float64 vectors; wθ_p and θ_profile under their Julia names)."
function as_result(d)
    ren = Dict("wtheta_p" => :wθ_p, "theta_profile" => :θ_profile)
    ks = [get(ren, k, Symbol(k)) for k in keys(d)]
    vs = [v isa AbstractVector ? Float64.(v) : v for v in values(d)]
    return NamedTuple{Tuple(ks)}(Tuple(vs))
end
res = Dict(name_of[l] => as_result(r.res) for (l, r) in runs)
cfgs = Dict(name_of[l] => (λ = r.cfg["surface"]["wavelength"], T_hot = r.cfg["surface"]["t_hot"],
                           T_cool = r.cfg["surface"]["t_cool"], mode = Symbol(r.cfg["surface"]["mode"]))
            for (l, r) in runs)
B_grid = (Lz = runs[end][2].cfg["grid"]["extent"][3], Nz = runs[end][2].cfg["grid"]["size"][3])
A_cases = [(n, nothing) for (_, n) in A_cases]
rel(a, b) = 100 * (a - b) / b
Au(r) = patchy_amplitude(r.u_p) / r.wstar

println("\nA. Domain-mean surface kinematic heat flux w'θ'₀ (K m/s), time mean of the second hour")
@printf("%-40s %7s %7s | %8s %8s %8s %8s %8s %8s | %6s %5s %6s\n", "case", "LES", "λ/zi",
        "tile:pat", "tile:spd", "tile:vec", "tile:gst", "blended", "param", "w*", "zi", "A_u/w*")
for (name, _) in A_cases
    r = res[name]
    c = cfgs[name]
    λzi = (c.T_hot == c.T_cool || c.mode !== :resolved) ? "—" : @sprintf("%.2f", c.λ / r.zi)
    @printf("%-40s %7.4f %7s | %+7.1f%% %+7.1f%% %+7.1f%% %+7.1f%% %+7.1f%% %+7.1f%% | %6.2f %5.0f %6.3f\n",
            name, r.les, λzi, rel(r.tile_patch, r.les), rel(r.tile_speed, r.les),
            rel(r.tile_vec, r.les), rel(r.tile_gust, r.les), rel(r.blended, r.les), rel(r.param, r.les),
            r.wstar, r.zi, Au(r))
end
println("  tile:pat  tiles forced by the mean air over their own patches")
println("  tile:spd  tiles forced by the domain-mean air, mean wind *speed*")
println("  tile:vec  … with the speed of the domain-mean wind *vector* (U_min = 0.1 m/s)")
println("  tile:gst  tile:vec plus Beljaars (1995) gustiness 1.2 w*")
println("  blended   BlendedMosaic, l_b = first level, forced by the mean air at 125 m")
println("  param     one surface at the mean skin temperature, forced like tile:spd")

r1, r4, rt, rh = (res[n] for n in ("strips λ = 0.8 km", "strips λ = 3.2 km",
                                   "tiled mosaic in every column", "uniform 305 K (= parameter aggregate)"))
rw, rtw = res["windy: strips λ = 3.2 km"], res["windy: tiled mosaic"]
@printf("\nA-posteriori: tiled LES %+.1f%% vs strips λ = 0.8 km, %+.1f%% vs λ = 3.2 km; windy %+.1f%%;\n",
        rel(rt.les, r1.les), rel(rt.les, r4.les), rel(rtw.les, rw.les))
@printf("              uniform mean-temperature LES %+.1f%% / %+.1f%%;  z_i: strips %.0f / %.0f m, tiled %.0f m, uniform %.0f m\n",
        rel(rh.les, r1.les), rel(rh.les, r4.les), r1.zi, r4.zi, rt.zi, rh.zi)
θdiff(a, b) = maximum(abs, a.θ_profile[a.z .< 0.8 * a.zi] .- b.θ_profile[a.z .< 0.8 * a.zi])
@printf("              mixed-layer θ: max |tiled − strips| %.2f K (λ 0.8 km), %.2f K (3.2 km); |uniform − strips| %.2f K\n",
        θdiff(rt, r1), θdiff(rt, r4), θdiff(rh, r1))

println("\nB. Organised motion against λ/z_i (100 m resolution, hours 1.5-3)")
@printf("%-12s %6s %8s %8s %8s %10s %10s\n", "λ (m)", "λ/zi", "A_u/w*", "A_w/w*", "LES", "tile:pat", "mid/edge")
sweep = [(λ, res["sweep λ = $(Int(λ)) m"]) for λ in B_λ]
function mid_edge(r)
    n = length(r.wθ_p) ÷ 2
    h = r.wθ_p[1:n]
    m = max(1, n ÷ 8)
    return (sum(h[n÷2:n÷2+1]) / 2) / (sum([h[1:m]; h[end-m+1:end]]) / 2m)
end
for (λ, r) in sweep
    @printf("%-12.0f %6.2f %8.3f %8.3f %8.4f %+9.1f%% %10.3f\n", λ, λ / r.zi, Au(r),
            patchy_amplitude(r.w_p) / r.wstar, r.les, rel(r.tile_patch, r.les), mid_edge(r))
end
ru = res["sweep uniform 305 K"]
@printf("%-12s %6s %8.3f %8.3f %8.4f  (phase-average noise floor)\n", "uniform", "—", Au(ru),
        patchy_amplitude(ru.w_p) / ru.wstar, ru.les)

# ---- figures -------------------------------------------------------------
# Resample a phase profile (one wavelength) onto x/λ ∈ (0, 1).
function on_unit(r, v; n = 64)
    np = length(v)
    return [v[clamp(floor(Int, (k - 0.5) / n * np) + 1, 1, np)] for k in 1:n]
end
xu = ((1:64) .- 0.5) ./ 64
labels = ["λ/zi = $(round(λ / r.zi, digits = 1))" for (λ, r) in sweep]
save_multiline(joinpath(OUT, "phase_flux.svg"), xu,
               [on_unit(r, r.wθ_p ./ r.les) for (_, r) in sweep], labels;
               xlabel = "x / λ  (hot strip 0-0.5, cool 0.5-1)",
               ylabel = "w'θ'0 / domain mean", title = "Phase-averaged surface heat flux")
save_multiline(joinpath(OUT, "phase_u.svg"), xu,
               [on_unit(r, r.u_p ./ r.wstar) for (_, r) in sweep], labels;
               xlabel = "x / λ  (hot strip 0-0.5, cool 0.5-1)",
               ylabel = "u (first level) / w*", title = "Phase-averaged near-surface wind")
save_multiline(joinpath(OUT, "amplitude.svg"), [λ / r.zi for (λ, r) in sweep],
               [[Au(r) for (_, r) in sweep], [patchy_amplitude(r.w_p) / r.wstar for (_, r) in sweep],
                fill(Au(ru), length(sweep))],
               ["u_p rms", "w_p rms", "uniform u_p"];
               xlabel = "λ / zi", ylabel = "rms of phase average / w*  (u: $(round(Int, B_grid.Lz / B_grid.Nz / 2)) m, w: $(round(Int, ru.z_mid)) m)",
               title = "Patch-locked motion against heterogeneity scale", logx = true)
save_multiline(joinpath(OUT, "theta_profiles.svg"), r1.z,
               [r1.θ_profile, r4.θ_profile, rt.θ_profile, rh.θ_profile],
               ["strips 0.8 km", "strips 3.2 km", "tiled", "uniform 305 K"];
               xlabel = "z (m)", ylabel = "mean θ (K)", title = "Mean θ after 2 h, calm")
println("\nFigures in ", normpath(OUT))
