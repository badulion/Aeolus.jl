# The Berlin nesting parent alone (N-parent-e, #259): a 24 h diurnal run,
# 06 to 06 LT, of the 6 × 6 × 3 km Aeolus parent around Alexanderplatz on
# 29/30 June 2019 (the Lindenberg sounding of #247), with every parent
# forcing on: the tiled urban surface (#249), the child's column radiation,
# the geostrophic pressure gradient, θ/qv nudging, subsidence, θ damping
# and the fixed-gradient top (#248, #254), the TKE closure on the
# stretched grid. See docs/src/design/berlin_parent.md.
#
#   julia --project=. -t 3 examples/urban/berlin_parent_diurnal.jl [grid] [hours] [D]
#
# grid  `default`  (2, 2, 4): 192 × 192 × 53, Δx = 31.25 m, Δz = 25 m → 100 m
#       `fallback` (4, 4, 8): 96 × 96 × 60, Δx = 62.5 m, Δz = 50 m uniform
#       `coarse`   (16, 16, 8): 24 × 24 × 38 (the test configuration)
# hours simulated hours (default 24)
# D     large-scale divergence of the subsidence, s⁻¹ (default 0, the
#       parent default; 3e-6 for the sensitivity run, which currently
#       diverges during the day, issue #312: the run then stops, reports the
#       failure and still writes its tables up to that point)
#
# Writes examples/urban/output/berlin_parent_<grid>[_D<D>].csv (one row per
# output: z_i, surface fluxes, the ledger terms), the θ profiles as
# berlin_parent_<grid>_theta.csv, and the figures
# docs/src/assets/berlin/parent_theta_zi.svg (θ(z, t) with z_i(t)) and
# parent_ledger.svg (the air-heat ledger terms) for the default D = 0
# (drawn by berlin_parent_figures.jl from those tables).

using Aeolus, Printf
isdefined(Main, :BerlinCase) || include(joinpath(@__DIR__, "BerlinCase.jl"))
const BC = Main.BerlinCase

grid = length(ARGS) >= 1 ? ARGS[1] : "default"
hours = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 24.0
D = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.0
refinement, vertical = grid == "default" ? ((2, 2, 4), :stretched) :
                       grid == "fallback" ? ((4, 4, 8), :uniform) :
                       grid == "coarse" ? ((16, 16, 8), :stretched) :
                       error("grid must be default, fallback or coarse")
tag = grid * (D == 0 ? "" : "_D$D")

child = BC.CaseConfig(; verbose = false, output_path = nothing)
pc = BC.ParentConfig(child; refinement = refinement, vertical = vertical, hours = hours,
                     divergence = D, output_period = 1800.0)
tb = @elapsed p = BC.build_parent(pc)
@printf("build %.1f s\n", tb)
failure = nothing
t = @elapsed try
    BC.run_parent!(p)
catch err
    global failure = sprint(showerror, err)
end
m = p.mesh
done_h = p.sim.t / 3600
failure === nothing ||
    @printf("%s FAILED at %.2f h (%05.2f LT): %s\n", tag, done_h, mod(6 + done_h, 24), first(split(failure, '\n')))
@printf("%s: %d × %d × %d, %d steps in %.0f s wall = %.1f s per simulated hour; dt %.2f–%.2f s (mean %.2f), %d retries\n",
        tag, m.Nx, m.Ny, m.Nz, p.nsteps, t, t / max(done_h, 1e-9), minimum(p.dt_hist), maximum(p.dt_hist),
        sum(p.dt_hist) / p.nsteps, p.retries)
println("timing: ", join((@sprintf("%s %.0f s", k, v) for (k, v) in p.timing), ", "))
print(BC.heat_report(p.heat; layers = (:column, (0.0, 100.0), (0.0, 1000.0))))
print(BC.heat_report(p.heat; window = (20, 5), layers = (:column, (0.0, 100.0))))

# --- tables -------------------------------------------------------------------
outdir = joinpath(@__DIR__, "output")
mkpath(outdir)
keys_ = [(:surface_flux, :tiles), (:column_tendency, :column_radiation), (:dynamics, :subsidence),
         (:dynamics, :nudging), (:dynamics, :damping), (:dynamics, :top_flux),
         (:dynamics, :transport), (:dynamics, :redistribution)]
open(joinpath(outdir, "berlin_parent_$tag.csv"), "w") do io
    println(io, "# Berlin parent ($tag, D = $D s^-1): one row per output; ledger terms cumulative J/m2 (column)",
            failure === nothing ? "" : "; FAILED at $(round(done_h; digits = 2)) h")
    println(io, "t_s,hour_LT,zi_min_flux_m,zi_parcel_m,H_cum_J_m2,LE_cum_J_m2,sw_down,lw_down,storage_J_m2,",
            join(("$(a)_$(b)" for (a, b) in keys_), ","), ",closure_J_m2")
    for r in p.diag
        @printf(io, "%.0f,%.3f,%.1f,%.1f,%.6e,%.6e,%.2f,%.2f,%.6e", r.t, r.hour, r.zi, r.zi_parcel,
                r.sfc[:H], r.sfc[:LE], r.sw_down, r.lw_down, r.heat.storage)
        for k in keys_
            @printf(io, ",%.6e", get(r.heat.terms, k, 0.0))
        end
        @printf(io, ",%.3e\n", r.heat.closure)
    end
end
zc = BC.level_centres(m)
open(joinpath(outdir, "berlin_parent_$(tag)_theta.csv"), "w") do io
    println(io, "hour_LT,", join((@sprintf("z%.1f", z) for z in zc), ","))
    for r in p.diag
        println(io, @sprintf("%.3f", r.hour), ",", join((@sprintf("%.4f", x) for x in r.θ), ","))
    end
end

# --- figures (default subsidence only, completed runs) -------------------------
(D == 0 && failure === nothing) || exit()
include(joinpath(@__DIR__, "berlin_parent_figures.jl"))
parent_figures(tag; outdir = outdir, label = @sprintf("%d × %d × %d, D = 0", m.Nx, m.Ny, m.Nz))
