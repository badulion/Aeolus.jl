# Alexanderplatz, Berlin — THE simulation.
#
# Real OSM geometry, full Era III radiation, and every physics add-on the
# stack ships. This is the single run; nothing else in examples/urban
# integrates the equations.
#
# It writes a small frame file every `frame_every` steps into
# `output/frames/`. `alexanderplatz_animation.jl` is a pure renderer over
# those files — it does no physics at all.
#
# WHY THIS IS ONE SCRIPT AND NOT TWO
#
# There used to be two: this one and an "animation" script that ran its
# own independent copy of the simulation in order to capture frames from
# live state. That was never a design decision, just the path of least
# resistance, and it cost real money:
#
#   * The physics had to be maintained twice. When the Era III radiation
#     landed it had to be wired into both, separately.
#   * The two copies DRIFTED. This one had soil conduction, latent heat,
#     chemistry and aerosol; the other had none of them, while the docs
#     claimed both ran "the same physics".
#   * That drift is what hid an energy-conservation bug in `step_soil!`
#     for as long as it survived: the script that exercised it was not
#     the script whose diagnostics were being read.
#   * Every animation cost a second full 24 h integration (~50 min), and
#     re-rendering a panel meant re-running the physics.
#
# One run, saved state, separate rendering.

using Aeolus
using Aeolus.Fields: set!, interior_view
using Aeolus.Radiation: apply_radiative_cooling!
using Aeolus.Operators: advect_centered!
using Aeolus.AdvectionSchemes: advect_positive!, FluxLimiterWorkspace
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

const OUT = joinpath(@__DIR__, "output"); mkpath(OUT)

# ---------------------------------------------------------------------------
# 1. Geometry — real Alexanderplatz OSM data
# ---------------------------------------------------------------------------

Lx = Ly = 1000.0; Lz = 200.0
Nx = Ny = 64;    Nz = 32
mesh = RegularGrid(; size=(Nx,Ny,Nz), extent=(Lx,Ly,Lz), halo=(3,3,3))   # 3: WENO-Z5 moisture transport

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
# L3: the production soil column (Aeolus.SoilColumn, Phase 46) under
# every non-water cell — 10 CLM layers to 3.4 m, moist loam, base 288 K.
soil       = UrbanSoil(sfc; T0 = 288.0, T_deep = 288.0)
snow       = SnowState(mesh)
q_dew      = zeros(Float64, Nx, Ny)      # surface dew reservoir, kg m⁻²
# Layer pressures for fog/dew thermodynamics: isothermal 8 km scale height
# from a standard surface pressure (the Boussinesq core carries no p).
p_col      = [1.01325e5 * exp(-(k - 0.5) * (Lz / Nz) / 8000.0) for k in 1:Nz]

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
moist_bcs = FieldBoundaryConditions(:periodic, :periodic, :noflux)
adv_ws    = FluxLimiterWorkspace(mesh)

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
const soil_every   = 15      # soil exchange every 15 steps = 30 s (implicit, see step_soil!)

k_slice = clamp(round(Int, 30 / mesh.dz), 1, Nz)

# Direct/diffuse split of the top-of-atmosphere solar constant.
# ~85% direct beam, ~15% isotropic diffuse — clear-sky approximation.
const LON          = 13.4100
const sky_model = BirdClearSky(; aod380 = 0.24, aod500 = 0.16,
                                      precipitable_water = 2.0,
                                      ozone = 0.33, albedo = 0.15)
sun_now = Ref(solar_position(solar_time(2024, 6, 20, 6, 0, 0.0;
                                        latitude = LAT, longitude = LON,
                                        elevation = 34.0)))

force! = (s, Δt) -> add_geostrophic_wind!(s, 1.0, 0.0, 1800.0)
# Park canopy drag: a uniform crown over the lowest 3 layers (18.75 m) with
# a leaf drag coefficient of 0.2. Applied implicitly after the predictor
# (Aeolus.PlantCanopy, issue #22), so it is stable at any dt. The old
# explicit tendency shipped disabled after an instability at dt = 2 s that
# #22 could not reproduce (dt·cd·LAI·|U| <= 0.05 here); cause undiagnosed.
veg_drag = vegetation_canopy_drag(mesh, veg_mask, LAI_map; cd = 0.2, k_height = 3)
post_predictor = (s, Δt) -> begin
    implicit_ibm_correction!(s, sm; τ = 0.0, dt = Δt)
    apply_vegetation_drag!(s, veg_drag, Δt)
end
sensible = zeros(Float64, Nx, Ny)
sw_arr   = zeros(Float64, Nx, Ny)
lw_arr   = fill(320.0, Nx, Ny)
sw_beam_now = Ref(0.0)
Hx, Hy, Hz = mesh.Hx, mesh.Hy, mesh.Hz

# ---------------------------------------------------------------------------
# Frame output — everything the renderer needs, and nothing more.
#
# Only 2-D slices are stored: the renderer draws horizontal panels at one
# height, so writing the full 3-D state would be ~30x larger for no gain.
# Static geometry (sky-view factor, heights, surface class) goes into a
# single `static.jls`; per-frame fields go into `frame_NNN.jls`.
# ---------------------------------------------------------------------------
const NCPATH = joinpath(OUT, "alexanderplatz.nc")
nframe = 0

# Full 3-D model state in NetCDF-3, written with the dependency-free
# writer in `src/NetCDFIO.jl`. This is the default output format because
# it is the one every other tool in the field can read — ncview, xarray,
# Paraview, CDO, Panoply — without Aeolus being installed.
#
# NCDatasets/NetCDF.jl/HDF5.jl all fail to precompile on this host, so
# the container format is written directly from the Unidata specification
# rather than through the C library. The files are ordinary `.nc` files;
# nothing downstream can tell the difference.
#
# Size: 64x64x32 Float32 is 512 kB per 3-D field, nine fields per record,
# 48 records — about 220 MB for a full day. Float32 rather than Float64
# because that is the usual precision for model output and it halves the
# file for no loss that matters at single-precision measurement accuracy.
const ncf = nc_create(NCPATH; attributes = Dict(
    "title"       => "Aeolus.jl LES — Alexanderplatz, Berlin",
    "institution" => "Aeolus.jl",
    "source"      => "Boussinesq LES, cut-cell IBM, Era III radiation",
    "comment"     => "Real OSM geometry; 24 h diurnal cycle from 06:00 local",
    "references"  => "examples/urban/alexanderplatz_diurnal.jl"))

nc_defdim!(ncf, "x", Nx); nc_defdim!(ncf, "y", Ny); nc_defdim!(ncf, "z", Nz)
nc_defdim!(ncf, "time", nothing)          # unlimited record dimension

for (d, n, L) in (("x", Nx, Lx), ("y", Ny, Ly), ("z", Nz, Lz))
    nc_defvar!(ncf, d, Float32, (d,);
               attributes = Dict("units" => "m", "axis" => uppercase(d),
                                 "long_name" => "$d cell centre"))
end
nc_defvar!(ncf, "time", Float64, ("time",);
           attributes = Dict("units" => "seconds since simulation start",
                             "axis" => "T", "long_name" => "simulation time"))
nc_defvar!(ncf, "hour", Float64, ("time",);
           attributes = Dict("units" => "h", "long_name" => "local hour of day"))
nc_defvar!(ncf, "solar_elevation", Float64, ("time",);
           attributes = Dict("units" => "degree", "long_name" => "solar elevation angle"))

# --- static 2-D geometry ------------------------------------------------
nc_defvar!(ncf, "sky_view_factor", Float32, ("x", "y");
           attributes = Dict("units" => "1",
               "long_name" => "sky view factor, area-weighted per column"))
nc_defvar!(ncf, "building_height", Float32, ("x", "y");
           attributes = Dict("units" => "m", "long_name" => "building height"))
nc_defvar!(ncf, "surface_class", Float32, ("x", "y");
           attributes = Dict("units" => "1",
               "long_name" => "surface class (1 street 2 roof 3 park 4 water 5 wall)"))

# --- prognostic 3-D fields ---------------------------------------------
const NC3D = (("u",         "m s-1", "eastward wind"),
              ("v",         "m s-1", "northward wind"),
              ("w",         "m s-1", "upward wind"),
              ("theta",     "K",     "potential temperature"),
              ("p",         "m2 s-2", "kinematic pressure"),
              ("qv",        "kg kg-1", "water vapour specific humidity"),
              ("ql",        "kg kg-1", "cloud liquid water"),
              ("qr",        "kg kg-1", "rain water"),
              ("pollutant", "1",     "passive pollutant tracer"))
for (n, u, ln) in NC3D
    nc_defvar!(ncf, n, Float32, ("x", "y", "z", "time");
               attributes = Dict("units" => u, "long_name" => ln,
                                 "coordinates" => "x y z"))
end

# --- diagnostic 2-D fields ---------------------------------------------
const NC2D = (("Ts",    "K",     "surface temperature"),
              ("sw_in", "W m-2", "shortwave irradiance at the surface"),
              ("lw_in", "W m-2", "downwelling longwave at the surface"),
              ("T_soil1", "K",   "soil temperature, top layer (node 7 mm; NaN under water)"),
              ("G_soil", "W m-2", "ground heat flux from the surface slab into the soil (NaN under water)"))
for (n, u, ln) in NC2D
    nc_defvar!(ncf, n, Float32, ("x", "y", "time");
               attributes = Dict("units" => u, "long_name" => ln,
                                 "coordinates" => "x y"))
end

nc_endef!(ncf)
nc_write!(ncf, "x", Float32[(i - 0.5) * mesh.dx for i in 1:Nx])
nc_write!(ncf, "y", Float32[(j - 0.5) * mesh.dy for j in 1:Ny])
nc_write!(ncf, "z", Float32[(k - 0.5) * mesh.dz for k in 1:Nz])
nc_write!(ncf, "sky_view_factor", Float32.(svf))
nc_write!(ncf, "building_height",
          Float32[maximum(k for k in 1:Nz
                          if sm.mask[mesh.Hx+i, mesh.Hy+j, mesh.Hz+k]; init = 0) * mesh.dz
                  for i in 1:Nx, j in 1:Ny])
nc_write!(ncf, "surface_class", Float32.(sfc))

"Ground heat flux of the last soil exchange on the (Nx, Ny) grid, NaN under water."
function G_map()
    out = fill(NaN, Nx, Ny)
    for (n, I) in enumerate(soil.cells)
        out[I] = soil.G[n]
    end
    return out
end

"Append one time level of the full 3-D state to the NetCDF file."
function save_frame(idx::Int, hour_of_day::Real)
    iv(f) = Float32.(interior_view(f))
    rec = Dict{String,Any}(
        "time"            => [Float64(sim.t)],
        "hour"            => [Float64(hour_of_day)],
        "solar_elevation" => [Float64(sun_now[].elevation)],
        "u" => iv(sim.u), "v" => iv(sim.v), "w" => iv(sim.w),
        "theta" => iv(sim.θ), "p" => iv(sim.p), "qv" => iv(qv_p),
        "ql" => iv(ql_p), "qr" => iv(qr_p), "pollutant" => iv(pollutant),
        "Ts" => Float32.(Ts), "sw_in" => Float32.(sw_arr),
        "lw_in" => Float32.(lw_arr),
        "T_soil1" => Float32.(soil_layer_temperature(soil, (Nx, Ny), 1)),
        "G_soil"  => Float32.(G_map()))
    nc_append_record!(ncf, rec)
    return nothing
end


# Batch D: two-moment moisture + fog dew + aerosol + chemistry state
chem   = ChemistryState(mesh)
aerosol = AerosolState(mesh)
# The 2-moment state is separate from qv_p/ql_p/qr_p used above; we
# advance it in parallel as the "real" microphysics path.



# Initial shadow at t=start_h


@printf("Animation: %d steps, %d frames, dt=%.0fs\n",
        nsteps, cld(nsteps, frame_every), dt)
t_wall_start = time()

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
        irr = clear_sky_irradiance(sp, sky_model)
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
        # facet individually. `UrbanFacetScene` (Era V, Phase 49) removes
        # it: one conduction column and skin temperature per facet.
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
    # R6: glazing transmits part of the SW into the buildings; that part
    # leaves the glass cells' surface budget (UrbanSurface glazing optics)
    apply_glass_sw_transmission!(sensible, sfc, sw_arr)
    # L1: evaporative cooling
    LE = apply_latent_flux!(sim, Ts, sfc; dt = dt, qv_field = qv_p,
                            qv_air = 0.010)
    @. sensible += LE
    # L5: dew on cold surfaces is handled by M4 `apply_fog_dew!` below,
    # which removes the vapour it deposits (the old L5 proxy did not).
    # A3: emissions (traffic + biogenic); A1: photochemistry; A2: dep
    add_emissions!(chem, sfc, hour_of_day, dt; dz = mesh.dz)
    apply_photochem_step!(chem, hour_of_day, dt)
    for spec in (:NO, :NO2, :O3)
        apply_dry_deposition!(getproperty(chem, spec), sfc, spec, dt;
                              dz = mesh.dz)
    end
    apply_dry_deposition!(aerosol.q_a, sfc, :PM, dt; dz = mesh.dz)
    apply_aerosol_step!(aerosol, dt; V_sed = 0.005, sfc = sfc, dz = mesh.dz)

    # Moisture transport. qv and ql are ACTIVE now (fog/dew feed latent
    # heat back into θ and Ts), so they need a stable, non-oscillatory
    # scheme: WENO-Z5 with the positivity limiter (Phase 26). The former
    # forward-Euler + 2nd-order centered update is unconditionally
    # unstable; as a passive field its grid-scale noise was simply
    # clipped at the 0.03 clamp (RH reached 2.07 in 8,395 cells within
    # 1000 steps), but as soon as condensation was coupled that noise
    # became latent heat and the run blew up within 500 steps.
    # Cloud / fog water must move with the air too, or condensate stays
    # put while its vapour is carried away — heating one air mass and
    # evaporatively cooling the next.
    # Known residual: a few wall-nook cells (e.g. (64,58,1)) still gain
    # vapour, because the IBM correction applied after the projection
    # leaves the velocity slightly convergent there. Fog now caps those
    # cells at saturation instead of letting RH run to the clamp.
    for q in (qv_p, ql_p)
        fill_halos!(q, moist_bcs)
        fill!(tqv.data, 0)
        advect_positive!(tqv, sim.u, sim.v, sim.w, q, dt;
                         scheme = :weno_z5, workspace = adv_ws, bcs = moist_bcs)
        interior_view(q) .+= dt .* view(tqv.data, Hx+1:Hx+Nx, Hy+1:Hy+Ny, Hz+1:Hz+Nz)
    end
    apply_kessler_step!(qv_p, ql_p, qr_p, dt; Vt = :spectrum)
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

    # M4: dew onto cold surfaces + fog in the air (Aeolus.RadiationFog).
    # Dew's latent heat goes to the SURFACE budget: `sensible` is a loss
    # term there, so subtracting H_dew warms the surface. Fog condenses
    # into ql_p with enthalpy-conserving warming of θ.
    H_dew = apply_fog_dew!(sim.θ, qv_p, ql_p, q_dew, Ts, dt;
                           p = p_col, dz = mesh.dz, ρ = 1.0)
    @. sensible -= H_dew

    step!(sim, dt; body_force! = force!, post_predictor! = post_predictor)
    step_surfaces!(Ts, sfc, params, sw_arr, lw_arr, sensible, dt)
    # L3: the production soil column exchanges heat with the slab Ts
    # (implicit, energy-conserving; sub-cycled every `soil_every` steps)
    n % soil_every == 0 && step_soil!(soil, Ts, sfc, soil_every * dt, params)
    # L5: snow state (dormant in a summer run, but the machinery ships)
    update_snow_state!(snow, Ts, 0)
    apply_radiative_cooling!(sim.θ, fill(-2.315e-5, Nz) .* dt)
    clamp!(Ts, 270.0, 340.0)

    if any(!isfinite, interior_view(sim.u)) || any(!isfinite, Ts)
        @printf("!! blowup at step %d\n", n); break
    end

    if n % frame_every == 0
        global nframe += 1
        save_frame(nframe, hour_of_day)
        @printf("  frame %2d @ t=%5.1fh  wall=%5.0fs  Ts=%5.1f/%5.1f  SW=%4.0f/%4.0f W/m²  elev=%5.1f°  θ1=%5.1f/%5.1f  qv1min=%.4f  dew=%.3f mm  fog ql1max=%.2e\n",
                nframe, hour_of_day, time() - t_wall_start,
                minimum(Ts), maximum(Ts),
                minimum(sw_arr), maximum(sw_arr), sun_now[].elevation,
                minimum(view(interior_view(sim.θ), :, :, 1)),
                maximum(view(interior_view(sim.θ), :, :, 1)),
                minimum(view(interior_view(qv_p), :, :, 1)), maximum(q_dew), maximum(view(interior_view(ql_p), :, :, 1)))
    end
end

t_wall = time() - t_wall_start
nc_close!(ncf)
@printf("Ran %d steps in %.1f s. Wrote %d time levels to %s (%.1f MB)\n",
        nsteps, t_wall, nframe, NCPATH, filesize(NCPATH) / 1e6)
@printf("Peak ΔT (heat island): %.1f K   max|∇·u| = %.2e\n",
        maximum(Ts) - minimum(Ts), max_divergence(sim))
println("Render with:  julia --project=. examples/urban/alexanderplatz_animation.jl")
