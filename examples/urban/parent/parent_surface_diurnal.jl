# Offline diurnal run of three Berlin parent columns (N-parent-d, #249):
# the densest built column of the child footprint, a pure park column and
# a pure water (Spree) column nearest the anchor, under the synthetic
# clear-sky forcing `parent_offline_forcing` (cfg.date, 0-24 h local,
# dt = 300 s). Prints H, LE, G and the skin temperature at 13:00.
#
#   julia --project=. examples/urban/parent/parent_surface_diurnal.jl
#
# The footprint mean against the child's area mean is in
# test/berlin_parent_surface.jl (its @info lines).

using Aeolus, Printf
isdefined(Main, :BerlinCase) || include(joinpath(@__DIR__, "..", "BerlinCase.jl"))
const BC = Main.BerlinCase

cfg = BC.CaseConfig(; verbose = false, output_path = nothing)
case = BC.build_case(cfg; io = devnull)
pl = BC.read_parent_landuse()
ps0 = BC.build_parent_surface(case, pl; columns = CartesianIndex{2}[])
Is, Js = ps0.footprint
near(I) = hypot(I[1] - (pl.nx + 1) / 2, I[2] - (pl.ny + 1) / 2)
pick(c) = argmin(I -> (ps0.fractions[I, c] < 1) * 1e6 + near(I), CartesianIndices((pl.nx, pl.ny)))
cols = ["core" => argmax(I -> ps0.λp[I], CartesianIndices((Is, Js))),
        "park" => pick(BC.SURFACE_PARK), "water" => pick(BC.SURFACE_WATER)]
ps = BC.build_parent_surface(case, pl; columns = last.(cols))

dt = 300.0
for (name, I) in cols
    col = ps.columns[I]
    local fl13 = nothing
    for n in 1:round(Int, 86400 / dt)
        hour = (n - 1) * dt / 3600
        fl = BC.step_parent_column!(ps, col, BC.parent_offline_forcing(cfg, hour), dt; U_min = 0.1)
        hour == 13.0 && (fl13 = (fl, surface_temperature(col.surface)))
    end
    fl, Ts = fl13
    @printf("%-5s column %-10s classes %-14s λp %.2f h %5.1f m built-tile z0 %.2f m d %5.2f m, forcing level of the first tile %5.1f m | 13:00 Rn %6.1f H %6.1f LE %6.1f G %6.1f W m⁻², T_skin %.1f K\n",
            name, string(Tuple(I)), string(Int.(col.classes)), ps.λp[I], ps.h[I], ps.z0[I], ps.d[I],
            ps.z_centres[col.ktile[1]], fl.Rn, fl.H, fl.LE, fl.G, Ts)
end
