# UrbanFacetScene step on the Berlin (Alexanderplatz) radiosity scene:
# the matrix-free GMRES solve against the dense LU it replaced (#138).
#
# Run: julia --project=. benchmarks/bench_urban_facets.jl [wall_subdivisions] [blas_threads]
#
# Builds the flagship geometry (CaseConfig defaults: 64×64×32 cells over
# 1000 m × 1000 m × 200 m, the real OSM file), gives every radiosity facet
# its own UrbanFacet column, and advances 6 steps of 600 s (the radiation
# cadence) around local noon on 20 June with the Bird clear-sky sun, once
# with each solver. Prints the facet count, the one-off cost of the
# longwave response, the per-step cost of each phase and the largest
# skin-temperature difference between the two solvers.

using Aeolus, Printf, LinearAlgebra
using Aeolus.UrbanSurface, Aeolus.SurfaceExchange, Aeolus.Radiosity
using Aeolus.SolarGeometry: BirdClearSky, clear_sky_irradiance
isdefined(Main, :BerlinCase) ||
    include(joinpath(@__DIR__, "..", "examples", "urban", "BerlinCase.jl"))
const BC = Main.BerlinCase
const US = Aeolus.UrbanSurface

ws  = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1
nth = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
BLAS.set_num_threads(nth)

cfg  = BC.CaseConfig(; verbose = false, wall_subdivisions = ws)
mesh = RegularGrid(; size = cfg.size, extent = cfg.extent, halo = cfg.halo)
geom = BC.build_geometry(cfg, mesh; io = devnull)
sc, vf = geom.scene, geom.vfac
n = nfacets(sc)
make() = UrbanFacetScene(deepcopy(sc), vf; T = 293.0)
t_resp = @elapsed (u_i = make())
u_d = make()
air = SurfaceForcing(lw_down = 330.0, T_air = 293.0, U = 2.0, z_ref = 30.0)
sky = BirdClearSky(; aod380 = 0.24, aod500 = 0.16, precipitable_water = 2.0,
                   ozone = 0.33, albedo = 0.15)

function timed_step!(u, solver, sw)
    t_p = @elapsed s = US._prepare_scene_step(u, air, 600.0; sw = sw, solver = solver)
    t_s = @elapsed (x, _, _) = US._solve_scene_step(s)
    t_a = @elapsed US._apply_scene_step!(u, s, x)
    return t_p, t_s, t_a
end

@printf("Berlin scene, wall_subdivisions = %d: %d facets (%d roof, %d wall, %d ground), BLAS threads %d\n",
        ws, n, count(f -> f.kind === :roof, sc.facets), count(f -> f.kind === :wall, sc.facets),
        count(f -> f.kind === :ground, sc.facets), nth)
@printf("  one-off: UrbanFacetScene incl. longwave_response (one LU) %.2f s; J holds %.0f MB\n",
        t_resp, 8n^2 / 2^20)
ti = zeros(3); td = zeros(3); worst = 0.0; its = Int[]
for (k, hour) in enumerate(11.5:600/3600:12.5 - 1e-9)
    sp = BC.solar_position_at(cfg, hour)
    irr = clear_sky_irradiance(sp, sky)
    swsol = solve_shortwave(u_i.scene, vf, solar_unit_vector(sp); dni = irr.dni, dhi = irr.dhi)
    a = timed_step!(u_i, :iterative, swsol)
    b = timed_step!(u_d, :direct, swsol)
    if k > 1                                   # the first step compiles
        ti .+= a; td .+= b
    end
    push!(its, u_i.solve_iterations)
    global worst = max(worst, maximum(abs, facet_temperatures(u_i) .- facet_temperatures(u_d)))
end
ti ./= 5; td ./= 5
@printf("  per step (mean of steps 2–6): prepare / solve / apply\n")
@printf("    GMRES     %.3f / %.3f / %.3f s  = %.3f s  (%d–%d iterations)\n",
        ti..., sum(ti), extrema(its)...)
@printf("    dense LU  %.3f / %.3f / %.3f s  = %.3f s\n", td..., sum(td))
@printf("  max |T_skin(GMRES) - T_skin(LU)| over 6 steps: %.2e K\n", worst)
Ts = facet_temperatures(u_i)
for kind in (:roof, :wall, :ground)
    idx = [i for (i, f) in enumerate(sc.facets) if f.kind === kind]
    @printf("  %-6s skin after 1 h from 293 K at noon: mean %.2f K, range %.2f–%.2f K\n",
            kind, sum(Ts[idx]) / length(idx), extrema(Ts[idx])...)
end
