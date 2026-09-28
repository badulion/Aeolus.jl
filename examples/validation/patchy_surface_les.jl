# =========================================================================
# Patchy surface heating — resolved LES against tiled aggregation
#
# Phase 53 (#43). A convective boundary layer over strips of hot
# (310 K) and cool (300 K) ground, 50 % each, z0 = 0.1 m, z0h = z0/10,
# initial mixed layer 300 K to 600 m. Every surface column computes its
# own Monin–Obukhov fluxes through the SurfaceExchange contract (see
# patchy_surface/driver.jl). Two sets of runs:
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
# Writes SVG figures to output/patchy_surface/. The runs are independent;
# start Julia with threads to run them in parallel:
#
#   julia --project=. -t 8 examples/validation/patchy_surface_les.jl
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
include(joinpath(@__DIR__, "patchy_surface", "driver.jl"))
include(joinpath(@__DIR__, "..", "viz", "plot_utils.jl"))

const OUT = joinpath(@__DIR__, "..", "..", "output", "patchy_surface")
mkpath(OUT)

hot, cool, mean_T = 310.0, 300.0, 305.0
windy = (Ug = 5.0, f_cor = 1e-4)
A_cases = [
    ("uniform 305 K (= parameter aggregate)", (λ = 800.0, T_hot = mean_T, T_cool = mean_T)),
    ("strips λ = 0.8 km",                     (λ = 800.0, T_hot = hot, T_cool = cool)),
    ("strips λ = 3.2 km",                     (λ = 3200.0, T_hot = hot, T_cool = cool)),
    ("tiled mosaic in every column",          (λ = 800.0, T_hot = hot, T_cool = cool, mode = :tiled)),
    ("windy: uniform 305 K",                  (λ = 3200.0, T_hot = mean_T, T_cool = mean_T, windy...)),
    ("windy: strips λ = 3.2 km",              (λ = 3200.0, T_hot = hot, T_cool = cool, windy...)),
    ("windy: tiled mosaic",                   (λ = 3200.0, T_hot = hot, T_cool = cool, mode = :tiled, windy...)),
]
B_λ = [800.0, 1600.0, 3200.0, 6400.0, 12800.0]
B_cases = [[("sweep λ = $(Int(λ)) m", (λ = λ, T_hot = hot, T_cool = cool)) for λ in B_λ];
           ("sweep uniform 305 K", (λ = 800.0, T_hot = mean_T, T_cool = mean_T))]

A_grid = (Nx = 64, Ny = 32, Nz = 32, Lx = 3200.0, Ly = 1600.0, Lz = 1600.0)
B_grid = (Nx = 128, Ny = 16, Nz = 16, Lx = 12800.0, Ly = 1600.0, Lz = 1600.0)
A_run = (t_end = 7200.0, dt = 3.0, t_avg = 3600.0, every = 5)
B_run = (t_end = 10800.0, dt = 5.0, t_avg = 5400.0, every = 3)

# Build serially (FFTW plans), run in parallel.
jobs = [[(name, patchy_les(; A_grid..., cfg...), A_run) for (name, cfg) in A_cases];
        [(name, patchy_les(; B_grid..., cfg...), B_run) for (name, cfg) in B_cases]]
results = Vector{Any}(undef, length(jobs))
println("Running $(length(jobs)) LES on $(Threads.nthreads()) thread(s) …")
Threads.@threads :dynamic for k in eachindex(jobs)
    name, pl, run = jobs[k]
    t = @elapsed results[k] = patchy_run!(pl; run...)
    @printf("  done: %-40s %6.0f s\n", name, t)
end
res = Dict(jobs[k][1] => results[k] for k in eachindex(jobs))
cfgs = Dict(vcat(A_cases, B_cases))
rel(a, b) = 100 * (a - b) / b
Au(r) = patchy_amplitude(r.u_p) / r.wstar

println("\nA. Domain-mean surface kinematic heat flux w'θ'₀ (K m/s), time mean of the second hour")
@printf("%-40s %7s %7s | %8s %8s %8s %8s %8s %8s | %6s %5s %6s\n", "case", "LES", "λ/zi",
        "tile:pat", "tile:spd", "tile:vec", "tile:gst", "blended", "param", "w*", "zi", "A_u/w*")
for (name, _) in A_cases
    r = res[name]
    c = cfgs[name]
    λzi = (c.T_hot == c.T_cool || get(c, :mode, :resolved) !== :resolved) ? "—" : @sprintf("%.2f", c.λ / r.zi)
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
