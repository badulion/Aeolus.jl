# Real-geometry urban simulation — Alexanderplatz, Berlin.
#
# Uses `data/alexanderplatz_real.osm`, an actual Overpass-API download
# of the 1 km × 1 km box (52.518°N–52.526°N, 13.402°E–13.418°E).
# Contains 373 buildings (including the 368-m Fernsehturm — clipped to
# the 200 m domain top), 132 land-use patches, real streets and blocks.
#
# Re-fetch with `julia --project=examples data/fetch_alexanderplatz.jl`.
#
# All the physics we have runs simultaneously — see `urban_diurnal.jl`
# for the full list.  Nothing in this file is hand-tuned for the
# geometry; the same script drops in on any Overpass download.

using Aeolus
using Aeolus.Fields: set!, interior_view
using Aeolus.Radiation: apply_radiative_cooling!
using Aeolus.Operators: advect_centered!
using Printf
using Statistics: mean

include(joinpath(@__DIR__, "osm_ingest.jl"))
include(joinpath(@__DIR__, "urban_physics.jl"))
include(joinpath(@__DIR__, "..", "viz", "plot_utils.jl"))

const OUT = joinpath(@__DIR__, "output"); mkpath(OUT)

# ---------------------------------------------------------------------------
# 1. Mesh + real OSM city
# ---------------------------------------------------------------------------

Lx = Ly = 1000.0; Lz = 200.0
Nx = Ny = 64;    Nz = 32
mesh = RegularGrid(; size=(Nx,Ny,Nz), extent=(Lx,Ly,Lz), halo=(2,2,2))

const OSM_PATH = joinpath(@__DIR__, "data", "alexanderplatz_real.osm")
isfile(OSM_PATH) || error("Missing $(OSM_PATH). Run data/fetch_alexanderplatz.jl first.")

city = load_city_osm(OSM_PATH;
                     lat0 = 52.5220, lon0 = 13.4100,
                     x_shift = Lx/2, y_shift = Ly/2,
                     min_height = 8.0)     # skip 0-m artefacts
@printf("Alexanderplatz OSM: %d buildings, %d patches loaded\n",
        length(city.buildings), length(city.patches))

sm, sfc = surface_class_map(city, mesh; tag_walls = true)
params  = default_surface_params()
@printf("Solid cells: %d,  wall-tagged: %d,  park: %d,  water: %d\n",
        count(sm.mask),
        count(==(SURFACE_WALL), sfc),
        count(==(SURFACE_PARK), sfc),
        count(==(SURFACE_WATER), sfc))

# ---------------------------------------------------------------------------
# 1b. Era III radiation: full urban radiosity scene
# ---------------------------------------------------------------------------
# The geometry is static, so the facet decomposition and the view-factor
# matrix are built ONCE. Only the radiosity solve runs inside the time
# loop, and only every `rad_every` steps.
#
# This replaces four separate approximations the earlier version used:
#   * diffuse sky light scaled by a ray-cast sky-view factor, with no
#     inter-reflection between surfaces at all;
#   * a hand-tuned `boost = 0.5(1-μ)` fudge to get some light onto walls;
#   * longwave down as `320·SVF + 380·(1-SVF)`, two magic numbers;
#   * a binary shadow mask.
# All four are now consequences of the view-factor matrix rather than
# parameters.
@printf("Building radiosity scene... ")
t_rad = time()
heightmap = heightmap_from_solid_mask(sm)
scene = build_scene(mesh, heightmap; wall_subdivisions = 1)
vfac  = view_factors(scene)
svf   = vfac.F_sky                       # per-FACET sky-view factor
@printf("%d facets, %.1fs\n", nfacets(scene), time() - t_rad)
@printf("  reciprocity error %.2e, closure error %.2e\n",
        reciprocity_error(vfac), closure_error(vfac))
@printf("  sky-view factor: min %.3f, mean %.3f, max %.3f\n",
        minimum(svf), sum(svf)/length(svf), maximum(svf))

# Map each facet to the LES column it belongs to, so surface temperatures
# can be pushed into the scene and fluxes pulled back out.
facet_col = [f.col for f in scene.facets]

# Radiative trapping on this geometry, measured once for the record.
let sun_noon = solar_unit_vector((elevation = 60.9, azimuth = 180.0))
    α_eff = effective_albedo(scene, vfac, sun_noon; dni = 800.0, dhi = 120.0)
    α_mat = sum(facet_areas(scene) .* facet_albedos(scene)) / sum(facet_areas(scene))
    @printf("  effective albedo %.4f vs area-weighted material %.4f (trapping %.1f%%)\n",
            α_eff, α_mat, 100 * (1 - α_eff / α_mat))
end

# 2-D geometry snapshots
heights_2d = zeros(Nx, Ny)
for j in 1:Ny, i in 1:Nx, k in 1:Nz
    if sm.mask[mesh.Hx+i, mesh.Hy+j, mesh.Hz+k]
        heights_2d[i, j] = k * mesh.dz
    end
end
save_heatmap(joinpath(OUT, "alexanderplatz_heights.ppm"),
             Array(heights_2d); upscale=6)
save_heatmap(joinpath(OUT, "alexanderplatz_class.ppm"),
             Array(Float64.(sfc)); upscale=6)

# ---------------------------------------------------------------------------
# 2. Simulation setup
# ---------------------------------------------------------------------------

sim = BoussinesqSimulation(mesh;
    ν              = 2.0,
    θ₀             = 293.0,
    g_accel        = 9.81,
    f_coriolis     = 1e-4,
    C_s            = 0.1,
    Pr_t           = 1/3,
    moisture       = false,          # qv treated passively for stability
    sponge_α_max   = 0.05,
    sponge_z_start = 0.75 * Lz,
    bcs   = FieldBoundaryConditions(:periodic, :periodic, :wall),
    θ_bcs = FieldBoundaryConditions(:periodic, :periodic, :noflux))

set!(sim.θ, (x,y,z) -> 293.0 + max(0, (z - 100.0)) * 3e-3)
set!(sim.u, (x,y,z) -> 1.0)

# Passive moisture + Kessler
qv_p = Field(CellCenter(), mesh); set!(qv_p, (x,y,z) -> 0.010 * exp(-z / 800))
ql_p = Field(CellCenter(), mesh)
qr_p = Field(CellCenter(), mesh)
tqv  = Field(CellCenter(), mesh)

Ts = init_surface_state(sfc, 285.0)
@. Ts = 285.0 + 5.0 * (sfc == SURFACE_WATER)

# ---------------------------------------------------------------------------
# 3. Pollutants
# ---------------------------------------------------------------------------

pollutant = Field(CellCenter(), mesh); set!(pollutant, (x,y,z) -> 0.0)
tp        = Field(CellCenter(), mesh)
# Chimney locations — near the Fernsehturm plaza, S-Bahn station, and Kaufhof
chimneys = [(500.0, 500.0, 60.0),
            (350.0, 650.0, 25.0),
            (700.0, 400.0, 40.0)]
chimney_ijk = [(round(Int, cx / mesh.dx),
                round(Int, cy / mesh.dy),
                clamp(round(Int, cz / mesh.dz), 1, Nz)) for (cx, cy, cz) in chimneys]

particles = ParticleSet(mesh)

# ---------------------------------------------------------------------------
# 4. Diurnal loop
# ---------------------------------------------------------------------------

const LAT     = 52.5220
const LON     = 13.4100
const DOY     = 172
const T_DAY   = 24 * 3600.0
const dt      = 2.0
# Set AEOLUS_NSTEPS to shorten the run for a smoke test.
const nsteps  = parse(Int, get(ENV, "AEOLUS_NSTEPS", string(Int(T_DAY / dt))))
const start_h = 6.0

cool_atm = fill(-2.315e-5, Nz)

ts_h     = Float64[start_h]
T_street = Float64[mean_surface_temperature(Ts, sfc, SURFACE_STREET)]
T_roof   = Float64[mean_surface_temperature(Ts, sfc, SURFACE_ROOF)]
T_park   = Float64[mean_surface_temperature(Ts, sfc, SURFACE_PARK)]
T_water  = Float64[mean_surface_temperature(Ts, sfc, SURFACE_WATER)]
T_wall   = Float64[mean_surface_temperature(Ts, sfc, SURFACE_WALL)]
sw_hist  = Float64[0.0]

force!         = (s, Δt) -> add_geostrophic_wind!(s, 1.0, 0.0, 1800.0)
post_predictor = (s, Δt) -> implicit_ibm_correction!(s, sm; τ = 0.0, dt = Δt)

sensible = zeros(Float64, Nx, Ny)
sw_arr   = zeros(Float64, Nx, Ny)
lw_arr   = fill(320.0, Nx, Ny)
snap_hours = [10.0, 14.0, 18.0, 22.0, 4.0]
Hx, Hy, Hz = mesh.Hx, mesh.Hy, mesh.Hz
const rad_every = 300        # radiosity solve every 10 sim-minutes
sw_top_now   = Ref(0.0)
elev_now_deg = Ref(0.0)

# Bird clear-sky model with a continental-urban aerosol load. The AOD is
# taken from the Phase 37 urban mixture at a typical summer Berlin burden
# rather than guessed.
const sky_model = BirdClearSky(; aod380 = 0.24, aod500 = 0.16,
                                 precipitable_water = 2.0, ozone = 0.33,
                                 albedo = 0.15)

# Downwelling longwave from a correlated-k + two-stream solve on a
# mid-latitude summer column, as a function of near-surface air
# temperature. Built once and interpolated, because a full spectral
# solve every radiation step would dominate the run time for a quantity
# that varies smoothly over ~15 K.
const _LW_SKY_TABLE = let
    lw   = longwave_bands(Float64; ngpt_per_band = 2)
    gas  = correlated_k_optics(lw)
    slv  = two_stream_solver(lw; lw_variant = Quadrature())
    Ts_probe = collect(283.0:3.0:308.0)
    Ls = map(Ts_probe) do Tsfc
        col = us_standard_column(Float64; nz = 30, T_sfc = Tsfc,
                                 humid = :midlatitude_summer)
        o = allocate_optics(Float64, (30,), lw); compute_optics!(o, gas, col)
        f = allocate_fluxes(Float64, (), 31)
        solve_lw!(f, slv, o, col; sfc_emissivity = 0.95)
        f.flux_dn[1]
    end
    (Ts_probe, Ls)
end

"Downwelling longwave at the surface for near-surface air temperature `Ta` (K)."
function sky_longwave(Ta::Real)
    xs, ys = _LW_SKY_TABLE
    Ta <= xs[1]   && return ys[1]
    Ta >= xs[end] && return ys[end]
    k = searchsortedlast(xs, Ta)
    t = (Ta - xs[k]) / (xs[k+1] - xs[k])
    return ys[k] * (1 - t) + ys[k+1] * t
end

@printf("Alexanderplatz diurnal: %d steps × %.0f s = %.0f h\n",
        nsteps, dt, T_DAY / 3600)
t_wall_start = time()

blew_up = false
for n in 1:nsteps
    hour_of_day = (start_h + sim.t / 3600) % 24.0

    # ----- Era III radiation, refreshed every ~10 sim-min ------------------
    if n % rad_every == 0 || n == 1
        # (a) Sun position: Meeus, accurate to <0.01 deg, replacing the
        #     Era I sinusoid that was good to about a degree.
        hh  = floor(Int, hour_of_day)
        mm  = floor(Int, (hour_of_day - hh) * 60)
        ss  = (hour_of_day - hh - mm/60) * 3600
        st  = solar_time(2024, 6, 20, hh, mm, ss;
                         latitude = LAT, longitude = LON, elevation = 34.0)
        sunpos = solar_position(st)
        μ = max(sunpos.μ₀, 0.0)

        # (b) Clear-sky irradiance: Bird model with a realistic urban
        #     aerosol load, replacing Beer's law with a fixed optical
        #     depth and the hardcoded 85/15 direct/diffuse split.
        irr = clear_sky_irradiance(sunpos, sky_model)
        dni, dhi = irr.dni, irr.dhi

        # (c) Downwelling longwave from a real two-stream solve on a
        #     correlated-k column, replacing the constant 320 W/m².
        #     The near-surface air temperature is the diurnally varying
        #     part, so the sky flux now breathes with the boundary layer.
        θ1 = mean(view(sim.θ.data, Hx+1:Hx+Nx, Hy+1:Hy+Ny, Hz+1))
        L_sky = sky_longwave(θ1)

        # (d) Materials follow the sun: water is a Fresnel interface and
        #     its albedo runs 0.03 overhead to 0.26 at low sun.
        mats = default_urban_materials(; μ₀ = max(μ, 0.05))
        scene.materials.albedo     .= mats.albedo
        scene.materials.emissivity .= mats.emissivity

        # (e) Push current surface temperatures onto the facets, then
        #     solve both bands on the real geometry. Shadowing,
        #     inter-reflection and canyon longwave trapping all fall out
        #     of the view-factor matrix.
        for (fi, (ci, cj)) in enumerate(facet_col)
            set_temperature!(scene, fi, Ts[ci, cj])
        end
        sunv = solar_unit_vector(sunpos)
        swsol = μ > 0 ? solve_shortwave(scene, vfac, sunv; dni = dni, dhi = dhi) :
                        nothing
        lwsol = solve_longwave(scene, vfac; L_sky = L_sky)
        forcing = aggregate_to_grid(scene, vfac; sw = swsol, lw = lwsol,
                                    T_surface = Ts)

        # `sw_effective` / `lw_down_effective` invert `step_surfaces!`'s own
        # algebra, so the surface scheme reproduces the true facet fluxes
        # without being modified.
        # `aggregate_to_grid` returns fluxes per unit PLAN area, so a column
        # holding a 200 m tower reports ~13x the incident flux — its wall
        # area is 13x its footprint. That is the correct column ENERGY, but
        # `step_surfaces!` carries one temperature per column and wants an
        # irradiance per unit of actual SURFACE. Divide by the area density
        # to get the area-weighted mean over the column's facets.
        #
        # This is an explicit approximation, and worth naming: collapsing a
        # whole facade to a single temperature is a limitation of the
        # surface scheme, not of the radiosity solve, which resolves every
        # facet individually. A multi-layer urban canopy (Era V, Phase 49)
        # is what removes it.
        ad = max.(forcing.area_density, 1.0)
        σSB = 5.670374419e-8
        sw_arr .= forcing.sw_effective ./ ad
        lw_arr .= σSB .* Ts.^4 .+
                  (forcing.lw_down_effective .- σSB .* Ts.^4) ./ ad
        sw_top_now[] = irr.ghi
        elev_now_deg[] = sunpos.elevation
    end
    sw = sw_top_now[]

    # `local` is required: a bare assignment inside a `for` at top level
    # shadows the outer `sensible` and Julia warns about the ambiguity.
    local sensible = apply_surface_flux!(sim, Ts, sfc, params;
                                    dt = dt, H_cap = 250.0, C_h = 1e-3)

    # Passive moisture + Kessler
    fill_halos!(qv_p, FieldBoundaryConditions(:periodic))
    fill!(tqv.data, 0)
    advect_centered!(tqv, sim.u, sim.v, sim.w, qv_p)
    interior_view(qv_p) .+= dt .*
        view(tqv.data, Hx+1:Hx+Nx, Hy+1:Hy+Ny, Hz+1:Hz+Nz)
    apply_kessler_step!(qv_p, ql_p, qr_p, dt; Vt = 3.0)
    clamp!(interior_view(qv_p), 0.0, 0.03)

    # Pollutants
    fill_halos!(pollutant, FieldBoundaryConditions(:periodic))
    fill!(tp.data, 0)
    advect_centered!(tp, sim.u, sim.v, sim.w, pollutant)
    apply_linear_chemistry!(tp, pollutant, 3600.0)
    daytime_boost = (hour_of_day > 7 && hour_of_day < 20) ? 1.0 : 0.3
    for (ci, cj, ck) in chimney_ijk
        tp.data[Hx+ci, Hy+cj, Hz+ck] += 0.5 * daytime_boost
    end
    interior_view(pollutant) .+= dt .*
        view(tp.data, Hx+1:Hx+Nx, Hy+1:Hy+Ny, Hz+1:Hz+Nz)
    clamp!(interior_view(pollutant), 0.0, 100.0)

    step!(sim, dt; body_force! = force!, post_predictor! = post_predictor)
    step_surfaces!(Ts, sfc, params, sw_arr, lw_arr, sensible, dt)
    apply_radiative_cooling!(sim.θ, cool_atm .* dt)
    clamp!(Ts, 270.0, 340.0)

    if n % 150 == 0
        for (cx, cy, cz) in chimneys
            add_particles!(particles, [(cx + 10*(rand()-0.5),
                                         cy + 10*(rand()-0.5),
                                         cz)])
        end
    end
    if length(positions(particles)) > 0
        fill_halos!(sim.u, sim.bcs); fill_halos!(sim.v, sim.bcs); fill_halos!(sim.w, sim.bcs)
        advance_particles!(particles, sim, dt)
    end

    if any(!isfinite, interior_view(sim.u)) || any(!isfinite, Ts)
        @printf("!! blowup at step %d (t=%.1fh)\n", n, hour_of_day)
        global blew_up = true; break
    end

    if n % 60 == 0
        push!(ts_h,     hour_of_day)
        push!(T_street, mean_surface_temperature(Ts, sfc, SURFACE_STREET))
        push!(T_roof,   mean_surface_temperature(Ts, sfc, SURFACE_ROOF))
        push!(T_park,   mean_surface_temperature(Ts, sfc, SURFACE_PARK))
        push!(T_water,  mean_surface_temperature(Ts, sfc, SURFACE_WATER))
        push!(T_wall,   mean_surface_temperature(Ts, sfc, SURFACE_WALL))
        push!(sw_hist,  sw)
    end

    for h_snap in snap_hours
        if abs(hour_of_day - h_snap) < dt/3600
            fname = @sprintf("alexanderplatz_surfT_%02.0fh.ppm", h_snap)
            save_heatmap(joinpath(OUT, fname), Array(Ts); upscale=6,
                         vmin=280, vmax=325)
            pname = @sprintf("alexanderplatz_pollutant_%02.0fh.ppm", h_snap)
            k_slice = round(Int, 30 / mesh.dz)
            slice_p = view(interior_view(pollutant), :, :, k_slice)
            save_heatmap(joinpath(OUT, pname), Array(slice_p); upscale=6)

            @printf("  t=%5.1fh  wall=%5.0fs  SW=%5.0f  T_str=%5.1f T_roof=%5.1f T_wall=%5.1f T_park=%5.1f  N=%d\n",
                    hour_of_day, time() - t_wall_start, sw,
                    T_street[end], T_roof[end], T_wall[end], T_park[end],
                    length(positions(particles)))
        end
    end
end

# ---------------------------------------------------------------------------
# 5. Final plots
# ---------------------------------------------------------------------------

sim_hours = collect(range(0.0, T_DAY / 3600; length = length(ts_h)))
save_multiline(joinpath(OUT, "alexanderplatz_class_surfT.svg"),
               sim_hours, [T_street, T_roof, T_wall, T_park, T_water],
               ["street", "roof", "wall", "park", "water"];
               xlabel="hours since $(round(Int, start_h)):00",
               ylabel="mean surface T (K)",
               title="Alexanderplatz — diurnal surface T by class")

save_multiline(joinpath(OUT, "alexanderplatz_sw_forcing.svg"),
               sim_hours, [sw_hist], ["SW (W/m²)"];
               xlabel="hours since $(round(Int, start_h)):00",
               ylabel="SW (W/m²)",
               title="Astronomical SW at Alexanderplatz")

zs   = collect(Aeolus.Grids.znodes(mesh, Aeolus.Grids.CellCenter()))
θbar = mean_θ_profile(sim)
save_multiline(joinpath(OUT, "alexanderplatz_final_theta.svg"),
               zs, [θbar], ["⟨θ⟩(z)"];
               xlabel="z (m)", ylabel="θ (K)",
               title="⟨θ⟩(z) after 24 h — Alexanderplatz")

Ts_safe   = map(x -> isfinite(x) ? x : 293.0, Ts)
final_map = Ts_safe .- mean(Ts_safe)
save_heatmap(joinpath(OUT, "alexanderplatz_surfT_final.ppm"),
             Array(final_map); upscale=6, vmin=-8, vmax=8)

pos_map = zeros(Nx, Ny)
for p in positions(particles)
    i = clamp(round(Int, p[1] / mesh.dx), 1, Nx)
    j = clamp(round(Int, p[2] / mesh.dy), 1, Ny)
    pos_map[i, j] += 1.0
end
save_heatmap(joinpath(OUT, "alexanderplatz_particles_final.ppm"),
             Array(pos_map); upscale=6)

t_wall = time() - t_wall_start
if blew_up
    @printf("\nRun ended early. Peak street T so far: %.1f K\n", maximum(T_street))
else
    @printf("\nCompleted 24 h at Alexanderplatz in %.1f s wall time.\n", t_wall)
end
@printf("Peak ΔT (heat island): %.1f K   Streets vs parks: %.1f K   Walls vs parks: %.1f K\n",
        maximum(Ts) - minimum(Ts),
        maximum(T_street) - maximum(T_park),
        maximum(T_wall)  - maximum(T_park))
@printf("max|∇·u| = %.2e   Particles = %d   Outputs → %s\n",
        max_divergence(sim), length(positions(particles)), OUT)
