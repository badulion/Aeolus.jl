# Animated urban run — Alexanderplatz, Berlin.  Now with per-cell
# ray-marched building shadows and sky-view-factor-weighted radiation.
#
# Captures an 8-panel mosaic every 30 sim-minutes during a 24-h run
# and emits a single self-contained APNG.
#
# Panels (2 rows × 4 columns):
#   1. Surface temperature Ts
#   2. Direct-SW shadow mask (bright = sunlit, dark = shaded)
#   3. Potential-temp anomaly θ' @ z ≈ 30 m
#   4. Vertical velocity   w      @ z ≈ 30 m
#   5. Horizontal wind |uh|       @ z ≈ 30 m
#   6. Pressure anomaly p'        @ z ≈ 30 m
#   7. Pollutant tracer  c        @ z ≈ 30 m
#   8. Sky-view factor (static)   — reference for how much sky each cell sees

using Aeolus
using Aeolus.Fields: set!, interior_view
using Aeolus.Radiation: apply_radiative_cooling!
using Aeolus.Operators: advect_centered!
using Printf
using Statistics: mean

include(joinpath(@__DIR__, "osm_ingest.jl"))
include(joinpath(@__DIR__, "urban_physics.jl"))
include(joinpath(@__DIR__, "land_surface_addons.jl"))       # L2/L3/L5
include(joinpath(@__DIR__, "turbulence_addons.jl"))          # F1/F2/F3
include(joinpath(@__DIR__, "moisture_addons.jl"))            # M1/M2/M3/M4
include(joinpath(@__DIR__, "aq_addons.jl"))                  # A1..A4
include(joinpath(@__DIR__, "glass_material.jl"))             # R6
include(joinpath(@__DIR__, "..", "viz", "plot_utils.jl"))
include(joinpath(@__DIR__, "..", "viz", "apng.jl"))

const OUT = joinpath(@__DIR__, "output"); mkpath(OUT)

# ---------------------------------------------------------------------------
# 1. Geometry — real Alexanderplatz OSM data
# ---------------------------------------------------------------------------

Lx = Ly = 1000.0; Lz = 200.0
Nx = Ny = 64;    Nz = 32
mesh = RegularGrid(; size=(Nx,Ny,Nz), extent=(Lx,Ly,Lz), halo=(2,2,2))

const OSM_PATH = joinpath(@__DIR__, "data", "alexanderplatz_real.osm")
isfile(OSM_PATH) || error("Missing $(OSM_PATH). Run data/fetch_alexanderplatz.jl first.")

city = load_city_osm(OSM_PATH;
                     lat0 = 52.5220, lon0 = 13.4100,
                     x_shift = Lx/2, y_shift = Ly/2,
                     min_height = 8.0)
sm, sfc = surface_class_map(city, mesh; tag_walls = true)
# R6: promote a few south-facing roof cells to glass to demo the material
# (real OSM has no glass tag; we synthesize a mask here for illustration)
glass_mask = falses(Nx, Ny)
for j in 1:Ny, i in 1:Nx
    # Glass on the sunny (south) face of tall roofs
    if sfc[i, j] == SURFACE_ROOF && (i % 5 == 0) && (j % 4 == 0)
        glass_mask[i, j] = true
    end
end
promote_to_glass!(sfc, glass_mask)
params = glass_surface_params()
@printf("Alexanderplatz OSM: %d buildings; solid=%d wall=%d glass=%d\n",
        length(city.buildings), count(sm.mask),
        count(==(SURFACE_WALL), sfc),
        count(==(SURFACE_GLASS), sfc))

# L2/L3/L5 state
veg_mask   = (sfc .== SURFACE_PARK)
LAI_map    = fill(2.5, Nx, Ny)   # dense grassy park; open water/streets → 0
LAI_map .*= veg_mask
soil       = SoilGrid(mesh; T0 = 288.0)
snow       = SnowState(mesh)
q_dew      = zeros(Float64, Nx, Ny)

# Precompute the sky-view factor once (geometry doesn't change).
@printf("Computing sky-view factor... ")
t_svf = time()
# Era III: a real radiosity scene replaces the ray-cast sky-view factor.
# Geometry is static, so facets and view factors are built once.
heightmap = heightmap_from_solid_mask(sm)
scene = build_scene(mesh, heightmap; wall_subdivisions = 1)
vfac  = view_factors(scene)
facet_col = [f.col for f in scene.facets]
# Per-COLUMN sky-view factor, for the display panel and diagnostics.
svf = aggregate_to_grid(scene, vfac).svf
@printf("radiosity scene: %d facets in %.1fs, recip %.1e, closure %.1e\n",
        nfacets(scene), time() - t_svf, reciprocity_error(vfac), closure_error(vfac))
@printf("SVF per column: min %.3f mean %.3f max %.3f\n",
        minimum(svf), mean(svf), maximum(svf))

# ---------------------------------------------------------------------------
# 2. Simulation
# ---------------------------------------------------------------------------

sim = BoussinesqSimulation(mesh;
    ν              = 2.0,
    θ₀             = 293.0,
    g_accel        = 9.81,
    f_coriolis     = 1e-4,
    C_s            = 0.1,
    Pr_t           = 1/3,
    moisture       = false,
    sponge_α_max   = 0.05,
    sponge_z_start = 0.75 * Lz,
    bcs   = FieldBoundaryConditions(:periodic, :periodic, :wall),
    θ_bcs = FieldBoundaryConditions(:periodic, :periodic, :noflux))

set!(sim.θ, (x,y,z) -> 293.0 + max(0, (z - 100.0)) * 3e-3)
set!(sim.u, (x,y,z) -> 1.0)

qv_p = Field(CellCenter(), mesh); set!(qv_p, (x,y,z) -> 0.010 * exp(-z / 800))
ql_p = Field(CellCenter(), mesh)
qr_p = Field(CellCenter(), mesh)
tqv  = Field(CellCenter(), mesh)

Ts = init_surface_state(sfc, 285.0); @. Ts = 285.0 + 5.0 * (sfc == SURFACE_WATER)

pollutant = Field(CellCenter(), mesh); set!(pollutant, (x,y,z) -> 0.0)
tp        = Field(CellCenter(), mesh)
chimneys  = [(500.0, 500.0, 60.0), (350.0, 650.0, 25.0), (700.0, 400.0, 40.0)]
chimney_ijk = [(round(Int, cx / mesh.dx),
                round(Int, cy / mesh.dy),
                clamp(round(Int, cz / mesh.dz), 1, Nz)) for (cx, cy, cz) in chimneys]

# ---------------------------------------------------------------------------
# 3. Run + frame capture
# ---------------------------------------------------------------------------

const LAT     = 52.5220
const DOY     = 172
const T_DAY   = 24 * 3600.0
const dt      = 2.0
const nsteps  = parse(Int, get(ENV, "AEOLUS_NSTEPS", string(Int(T_DAY / dt))))
const start_h = 6.0
const frame_every = 900
const shadow_every = 300     # recompute shadow every 300 steps ≈ 10 sim-min

k_slice = clamp(round(Int, 30 / mesh.dz), 1, Nz)
upscale = 4

# Direct/diffuse split of the top-of-atmosphere solar constant.
# ~85% direct beam, ~15% isotropic diffuse — clear-sky approximation.
const DIRECT_FRAC  = 0.85   # retained only for the legacy frame-0 call
const LON          = 13.4100
const sky_model_anim = BirdClearSky(; aod380 = 0.24, aod500 = 0.16,
                                      precipitable_water = 2.0,
                                      ozone = 0.33, albedo = 0.15)
sun_now = Ref(solar_position(solar_time(2024, 6, 20, 6, 0, 0.0;
                                        latitude = LAT, longitude = LON,
                                        elevation = 34.0)))
const DIFFUSE_FRAC = 0.15

lim_Ts    = (280.0, 335.0)
lim_θ     = (-4.0, 4.0)
lim_w     = (-2.0, 2.0)
lim_uh    = (0.0, 4.0)
lim_p     = (-3.0, 3.0)
lim_c     = (0.0, 5.0)

force! = (s, Δt) -> begin
    add_geostrophic_wind!(s, 1.0, 0.0, 1800.0)
    # vegetation drag disabled — even at cd=0.03 it destabilizes at dt=2s;
    # a proper implicit-Euler treatment is a follow-up (see the plan).
    # apply_vegetation_drag!(s, veg_mask, LAI_map; cd = 0.01, k_height = 3)
end
post_predictor = (s, Δt) -> implicit_ibm_correction!(s, sm; τ = 0.0, dt = Δt)
sensible = zeros(Float64, Nx, Ny)
sw_arr   = zeros(Float64, Nx, Ny)
lw_arr   = fill(320.0, Nx, Ny)
shade    = falses(Nx, Ny)   # display only
sw_beam_now = Ref(0.0)
Hx, Hy, Hz = mesh.Hx, mesh.Hy, mesh.Hz

# SVF panel is static; render once, reused each frame.
svf_panel = scalar_to_rgb(Array(svf), 0.0, 1.0; upscale = upscale)
add_label!(svf_panel, "SVF"; row = 4, col = 4)

# Batch D: two-moment moisture + fog dew + aerosol + chemistry state
chem   = ChemistryState(mesh)
aerosol = AerosolState(mesh)
# The 2-moment state is separate from qv_p/ql_p/qr_p used above; we
# advance it in parallel as the "real" microphysics path.

frames = Vector{Array{UInt8, 3}}()

function capture_frame(hour_of_day, sw_direct_beam)
    ui = interior_view(sim.u); vi = interior_view(sim.v); wi = interior_view(sim.w)
    θi = interior_view(sim.θ); pi_= interior_view(sim.p); ci = interior_view(pollutant)

    u_slice  = view(ui, :, :, k_slice)
    v_slice  = view(vi, :, :, k_slice)
    w_slice  = view(wi, :, :, k_slice)
    θ_slice  = view(θi, :, :, k_slice)
    p_slice  = view(pi_, :, :, k_slice)
    c_slice  = view(ci, :, :, k_slice)
    uh_slice = sqrt.(u_slice .^ 2 .+ v_slice .^ 2)
    θ_anom   = θ_slice .- mean(θ_slice)
    p_anom   = p_slice .- mean(p_slice)

    # Shortwave reaching the surface, straight from the radiosity solve.
    # This is strictly more informative than the old binary shadow mask:
    # a shaded street still receives diffuse and inter-reflected light,
    # and the panel now shows that rather than pretending it is zero.
    direct_sw = copy(sw_arr)

    Ts_panel   = scalar_to_rgb(Array(Ts),         lim_Ts...; upscale = upscale)
    sun_panel  = scalar_to_rgb(direct_sw,          0.0, 1200.0; upscale = upscale)
    θ_panel    = scalar_to_rgb(Array(θ_anom),     lim_θ...; upscale = upscale)
    w_panel    = scalar_to_rgb(Array(w_slice),    lim_w...; upscale = upscale)
    u_panel    = scalar_to_rgb(Array(uh_slice),   lim_uh...; upscale = upscale)
    p_panel    = scalar_to_rgb(Array(p_anom),     lim_p...; upscale = upscale)
    c_panel    = scalar_to_rgb(Array(c_slice),    lim_c...; upscale = upscale)

    for (panel, title) in ((Ts_panel,  "Ts K"),
                           (sun_panel, "direct SW"),
                           (θ_panel,   "theta anom K"),
                           (w_panel,   "w m/s"),
                           (u_panel,   "uh m/s"),
                           (p_panel,   "p anom"),
                           (c_panel,   "pollutant"))
        add_label!(panel, title; row = 4, col = 4)
    end

    mosaic = tile_frames(reshape([Ts_panel, sun_panel, θ_panel,  w_panel,
                                  u_panel,  p_panel,   c_panel, svf_panel],
                                 (2, 4)))
    add_label!(mosaic, @sprintf("t %04.1fh", hour_of_day);
               row = size(mosaic, 1) - 12, col = 4)
    return mosaic
end

# Initial shadow at t=start_h


@printf("Animation: %d steps, %d frames, dt=%.0fs\n",
        nsteps, cld(nsteps, frame_every), dt)
t_wall_start = time()
capture_frame(start_h, 0.0) |> f -> push!(frames, f)

for n in 1:nsteps
    hour_of_day = (start_h + sim.t / 3600) % 24.0

    # Recompute shadows every ~10 sim-min
    # ----- Era III radiation ------------------------------------------------
    # Radiosity subsumes what used to be four separate approximations here:
    # the binary shadow mask, the SVF-weighted diffuse, the R3 wall
    # directional factor, and the R4 canyon-longwave proxy. Shadowing,
    # wall orientation, inter-reflection and longwave trapping are all
    # consequences of the view-factor matrix now.
    if n % shadow_every == 0 || n == 1
        hh = floor(Int, hour_of_day)
        mi = floor(Int, (hour_of_day - hh) * 60)
        se = (hour_of_day - hh - mi/60) * 3600
        sun_now[] = solar_position(solar_time(2024, 6, 20, hh, mi, se;
                                              latitude = LAT, longitude = LON,
                                              elevation = 34.0))
        sp  = sun_now[]
        irr = clear_sky_irradiance(sp, sky_model_anim)
        μ   = max(sp.μ₀, 0.0)

        mats = default_urban_materials(; μ₀ = max(μ, 0.05))
        scene.materials.albedo     .= mats.albedo
        scene.materials.emissivity .= mats.emissivity
        for (fi, (ci, cj)) in enumerate(facet_col)
            set_temperature!(scene, fi, Ts[ci, cj])
        end
        sunv  = solar_unit_vector(sp)
        swsol = μ > 0 ? solve_shortwave(scene, vfac, sunv;
                                        dni = irr.dni, dhi = irr.dhi) : nothing
        lwsol = solve_longwave(scene, vfac; L_sky = lw_from_walls(mean(Ts)))
        forcing = aggregate_to_grid(scene, vfac; sw = swsol, lw = lwsol,
                                    T_surface = Ts)
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
        sw_beam_now[] = irr.dni * μ
    end
    sw_direct_beam = sw_beam_now[]
    sw_top = sw_direct_beam
    # L2: tree shade attenuates SW under vegetation
    apply_tree_shade!(sw_arr, veg_mask, LAI_map)
    # R5: cloud shadow from column-integrated ql
    ql_col = dropdims(sum(interior_view(ql_p); dims = 3) .* mesh.dz;
                      dims = 3)
    apply_cloud_shadow!(sw_arr, ql_col, sw_direct_beam)

    local sensible = apply_surface_flux!(sim, Ts, sfc, params;
                                    dt = dt, H_cap = 250.0, C_h = 1e-3)
    # L4: anthropogenic heat (traffic, AC exhaust, interior heat loss)
    add_anthropogenic_heat!(sensible, sfc, hour_of_day)
    # R6: glass materials let some SW pass through; reduce sensible
    apply_glass_sw_transmission!(sensible, sfc, sw_arr)
    # L1: evaporative cooling
    LE = apply_latent_flux!(sim, Ts, sfc; dt = dt, qv_field = qv_p,
                            qv_air = 0.010)
    @. sensible += LE
    # L5: dew condensation on cold surfaces (disabled — warming term
    # amplifies before proper coupling with atmosphere qv is wired in)
    # apply_dew_condensation!(Ts, 0.010, LE; ε_warm = 3e-3)
    # A3: emissions (traffic + biogenic); A1: photochemistry; A2: dep
    add_emissions!(chem, sfc, hour_of_day, dt; dz = mesh.dz)
    apply_photochem_step!(chem, hour_of_day, dt)
    for spec in (:NO, :NO2, :O3)
        apply_dry_deposition!(getproperty(chem, spec), sfc, spec, dt;
                              dz = mesh.dz)
    end
    apply_dry_deposition!(aerosol.q_a, sfc, :PM, dt; dz = mesh.dz)
    apply_aerosol_step!(aerosol, dt; V_sed = 0.005, sfc = sfc, dz = mesh.dz)

    fill_halos!(qv_p, FieldBoundaryConditions(:periodic))
    fill!(tqv.data, 0)
    advect_centered!(tqv, sim.u, sim.v, sim.w, qv_p)
    interior_view(qv_p) .+= dt .* view(tqv.data, Hx+1:Hx+Nx, Hy+1:Hy+Ny, Hz+1:Hz+Nz)
    apply_kessler_step!(qv_p, ql_p, qr_p, dt; Vt = 3.0)
    clamp!(interior_view(qv_p), 0.0, 0.03)

    fill_halos!(pollutant, FieldBoundaryConditions(:periodic))
    fill!(tp.data, 0)
    advect_centered!(tp, sim.u, sim.v, sim.w, pollutant)
    apply_linear_chemistry!(tp, pollutant, 3600.0)
    dayboost = (hour_of_day > 7 && hour_of_day < 20) ? 1.0 : 0.3
    for (ci, cj, ck) in chimney_ijk
        tp.data[Hx+ci, Hy+cj, Hz+ck] += 0.5 * dayboost
    end
    interior_view(pollutant) .+= dt .* view(tp.data, Hx+1:Hx+Nx, Hy+1:Hy+Ny, Hz+1:Hz+Nz)
    clamp!(interior_view(pollutant), 0.0, 100.0)

    step!(sim, dt; body_force! = force!, post_predictor! = post_predictor)
    step_surfaces!(Ts, sfc, params, sw_arr, lw_arr, sensible, dt)
    # L3: multi-layer soil couples the ground reservoir to Ts
    step_soil!(soil, Ts, sfc, dt)
    # L5: snow state (dormant in a summer run, but the machinery ships)
    update_snow_state!(snow, Ts, 0)
    # M4: fog / dew — disabled by default; enable once atmosphere θ
    # is proven robust to per-cell condensation warming (small
    # ε_warm needed). API ships in moisture_addons.jl.
    # apply_fog_dew!(sim.θ, qv_p, q_dew, Ts, dt; dz = mesh.dz, rate = 1e-4)
    apply_radiative_cooling!(sim.θ, fill(-2.315e-5, Nz) .* dt)
    clamp!(Ts, 270.0, 340.0)

    if any(!isfinite, interior_view(sim.u)) || any(!isfinite, Ts)
        @printf("!! blowup at step %d\n", n); break
    end

    if n % frame_every == 0
        push!(frames, capture_frame(hour_of_day, sw_direct_beam))
        @printf("  frame %2d @ t=%5.1fh  wall=%5.0fs  Ts=%5.1f/%5.1f  SW=%4.0f/%4.0f W/m²  elev=%5.1f°\n",
                length(frames), hour_of_day, time() - t_wall_start,
                minimum(Ts), maximum(Ts),
                minimum(sw_arr), maximum(sw_arr), sun_now[].elevation)
    end
end

t_wall = time() - t_wall_start
@printf("Ran %d steps in %.1f s. Captured %d frames.\n", nsteps, t_wall, length(frames))

out_path = joinpath(OUT, "alexanderplatz_animation.png")
write_apng(out_path, frames; delay_ms = 250)
@printf("APNG written: %s  (%.1f MB, %d frames, %dx%d)\n",
        out_path, filesize(out_path)/1e6,
        length(frames), size(frames[1], 2), size(frames[1], 1))
