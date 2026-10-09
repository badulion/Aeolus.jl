# Examples battery

Aeolus ships a three-tier battery of examples that together exercise
every physics module in the stack. An example is a TOML config (plus its
input drivers) under `examples/<tier>/configs/`, run with `aeolus run`;
the `plot_<case>.jl` script beside it reads the finished run folder
(`diagnostics.csv`, `diagnostics.nc`, `output/`) and writes its figures
into the run's `output/` (EX-d, #441). No example script builds or runs
a model: the examples are configuration, and a `.jl` under `examples/`
only reads a run's output files (enforced by
`test/examples_config_only.jl`, EX-e #442):

```bash
$ julia --project -m Aeolus run examples/<tier>/configs/<case>.toml
$ julia --project=. examples/<tier>/plot_<case>.jl runs/<date>_<case>_<id>
```

The plotting stack is pure Julia (no C-library dependencies) — see
`examples/viz/plot_utils.jl` and the run-folder readers
`examples/viz/run_output.jl`. Convert the `.ppm` files to `.png` with
ImageMagick if you want to view them in most browsers:
`convert file.ppm file.png`.

Each config reproduces the script it replaced bit for bit
(`test/config_run_drivers.jl`), except `coupled/dcbl_radiation_lsm`, see
below. The couplings are sections of the `simulation` driver
(`[forcing]`, `[surface]`, `[microphysics]`, `[tracer]`, `[particles]`,
`[obstacles]`; `aeolus options simulation`).

## Tier 1 — Kernel demos

Seconds-to-run cases, each highlighting one mechanism.

| Config (`kernels/configs/`) | Physics                                          |
| :-------------------------- | :----------------------------------------------- |
| `tgv_energy_decay.toml`     | Taylor-Green vortex + analytic `exp(-6νt)` overlay |
| `decaying_hit.toml`         | Random-init decaying isotropic turbulence (`[initial] velocity_perturbation`) |
| `channel_flow.toml`         | Wall-bounded channel + `⟨u⟩(z)` mean profile      |
| `cloud_spinup.toml`         | Kessler warm-rain: `ql → qr → fallout` (the `column` driver) |
| `pollution_plume.toml`      | Point-source tracer with linear-decay chemistry (`[tracer]`) |
| `particle_trail.toml`       | Lagrangian particles in a Taylor-Green vortex (`[particles]`) |

The tracer of `pollution_plume` and `flow_past_cube` is advected with
centred differences and a forward-Euler step, as the scripts did; that
scheme is dispersive, and the tracer slices show its 2Δx oscillations
next to the sources (negative values included).

## Tier 2 — Coupled examples

Combine several physics modules in one run.

| Config (`coupled/configs/`) | Combines                                             |
| :-------------------------- | :--------------------------------------------------- |
| `dcbl_radiation_lsm.toml`   | DCBL + radiative cooling + `SlabLand` (3 h; `[forcing]`, `[surface] model = "slab"`) |
| `moist_convection.toml`     | Moist LES with initial cloud puff + Kessler (`[microphysics]`) |
| `heated_plume.toml`         | Hot-patch surface flux → rising plume, θ' slices (`[surface] model = "flux_map"`) |
| `flow_past_cube.toml`       | Wall-mounted cube with IBM penalty + tracer wake (the cube in the static driver `input/flow_past_cube_static.nc`, made by `aeolus preprocess static` from `input/flow_past_cube.geojson`) |

`dcbl_radiation_lsm` differs from its script on purpose. The script set
`sim.θ_flux_bottom` every step on a simulation built with a zero flux,
whose surface-flux flag is frozen at construction, so the land never
heated the air: its θ was bitwise that of a run without land (asserted in
the test). The run type applies the flux; measured over the 3 h, the
slab warms from 297 K to 315 K and heats the air with up to 128 W m⁻²,
and the boundary layer (the height of the most negative heat flux) grows
from 800 m to 1275 m.

## Tier 3 — Flagship urban simulation

!!! note "Historical section"
    This section describes `examples/urban/urban_diurnal.jl`, a second
    script that integrated the equations over the synthetic sample city.
    It was removed in #126: there is now one simulation, the Berlin case
    (see [Real Berlin — Alexanderplatz OSM run](@ref) and the
    [Berlin case design page](../design/berlin_case.md)). The synthetic
    sample city runs through the same case with
    `examples/urban/configs/berlin_mitte_sample.toml`. The setup and numbers below are the
    old script's and are kept as a record.

`examples/urban/urban_diurnal.jl` (removed) — the "everything on"
showcase over a real-format OSM city.

**Setup**
- 1000 × 1000 × 200 m domain, 64 × 64 × 32 grid (dx ≈ 15.6 m)
- **Real-format OSM** city geometry loaded from
  `data/sample_berlin_mitte.osm` — Berlin-Mitte-style perimeter block
  housing + several taller commercial buildings, at real geographic
  coordinates (52.522 N, 13.410 E). The OSM XML parser reads the same
  format as an Overpass-API download; the sample is bundled so the
  example runs without an internet fetch.
- 136 buildings (height ≥ 15 m), 4 land-use patches (parks + water)
- Latitude 52.5 N, day 172 (summer solstice), 24 h starting at 6:00 am
- Physics on:
  * `SolidMask` + **implicit** IBM correction (`implicit_ibm_correction!`)
    in a `post_predictor!` hook — unconditionally stable for the dense
    obstacle field
  * 5-class LSM: roof / street / park / water / **wall**
    (wall = building-adjacent street cell with its own thermal
    parameters representing the effective vertical-face heat load)
  * Astronomical solar zenith + Beer's-law direct SW; walls get
    extra SW at low sun angles
  * Prescribed downward LW; free-atmosphere radiative cooling (−2 K/day)
  * Smagorinsky-Lilly SGS closure
  * Rayleigh sponge in the top 25 % of the domain
  * f-plane Coriolis (`f = 1e-4 s⁻¹`)
  * Geostrophic forcing (`LargeScaleForcing` since #171: a pressure
    gradient by default, Rayleigh nudging of the mean wind as an option)
  * Cosine-FFT pressure solver (walls in z, periodic in xy)
  * **Moisture**: prognostic `qv` as a passive tracer (advected by the
    same staggered scheme as θ), with `Kessler` microphysics processing
    any generated cloud / rain water
  * **Pollutants**: three chimney stacks with per-step injection and
    1-h photolysis decay
  * **Lagrangian particles**: seeded near each stack every 5 minutes,
    tracked with trilinear velocity interpolation

**Physical result** (one representative run):

```
OSM city: 136 buildings, 4 patches loaded
Solid cells: 6089,  wall-tagged cells: 1123
Urban diurnal (OSM city, moist + pollutants): 43200 steps × 2 s = 24 h
  t=10.0h  SW= 847 W/m²  T_str=304.7 T_roof=311.6 T_wall=301.6 T_park=295.7  N=144
  t=14.0h  SW= 847 W/m²  T_str=329.0 T_roof=338.7 T_wall=321.4 T_park=310.9  N=288
  t=18.0h  SW= 228 W/m²  T_str=331.7 T_roof=333.6 T_wall=327.5 T_park=315.7  N=432
  t=22.0h  SW=   0 W/m²  T_str=316.8 T_roof=311.4 T_wall=317.4 T_park=309.7  N=576
  t= 4.0h  SW=   0 W/m²  T_str=302.8 T_roof=296.2 T_wall=305.9 T_park=302.4  N=792
Completed 24-h urban simulation in 677 s wall time.
Peak ΔT (heat island): 12.0 K   Streets vs parks: 18.0 K   Walls vs parks: 12.0 K
max|∇·u| = 2.86e-15
```

**Physics interpretation**

  * **Roofs peak first** (14 h, 339 K) — low heat capacity + direct sun.
  * **Streets peak later** (18 h, 332 K) — thermal lag from higher `C_s`.
    By the next morning at 4 am they're still the warmest surface class,
    illustrating the classic *nocturnal urban heat island*.
  * **Walls track streets closely** (327 K peak at 18 h) — even more
    thermal lag because concrete/masonry stores more heat per unit area.
  * **Parks stay cool all day** (peak 316 K) — high heat capacity plus
    higher albedo.
  * **Water is essentially flat** (huge thermal inertia, low albedo).
  * At the **peak of insolation**, the street-vs-park contrast is **18 K**
    — consistent with observed urban heat islands on clear summer days.

**Numerical result**

Divergence stays at machine ε (10⁻¹⁵) for all 43,200 time steps even
with:
- 136 IBM obstacles enforced implicitly,
- surface heat flux injected per cell,
- passive moisture undergoing Kessler processing,
- 864 Lagrangian particles integrated to end of run,
- a pollutant tracer with 1-h linear decay,
- Coriolis + geostrophic wind + sponge damping + Smagorinsky SGS
  simultaneously active.

**Output files** (all in `examples/urban/output/`)

| File                          | Description                                      |
| :---------------------------- | :----------------------------------------------- |
| `urban_heights.ppm`           | Building height map (0–90 m)                     |
| `urban_class.ppm`             | Surface class map (5-class)                      |
| `urban_surfT_10h.ppm`         | Surface T at 10:00 (rising heat island)          |
| `urban_surfT_14h.ppm`         | Surface T at 14:00 (peak solar)                  |
| `urban_surfT_18h.ppm`         | Surface T at 18:00                               |
| `urban_surfT_22h.ppm`         | Surface T at 22:00 (evening cooldown)            |
| `urban_surfT_04h.ppm`         | Surface T at 04:00 (pre-dawn minimum)            |
| `urban_surfT_final.ppm`       | Final surface-T anomaly (Ts − ⟨Ts⟩)              |
| `urban_pollutant_XXh.ppm`     | Pollutant near-ground slice at each snapshot     |
| `urban_particles_final.ppm`   | 2-D density of Lagrangian particles after 24 h   |
| `urban_class_surfT.svg`       | Time series: class-averaged T over the day       |
| `urban_sw_forcing.svg`        | Astronomical SW at 52.5 N solstice               |
| `urban_final_theta.svg`       | ⟨θ⟩(z) profile after 24 h                        |

**Design notes / current limitations**

  * `moisture = true` on the simulation activates *virtual-θ* buoyancy —
    a small (~0.6 %) modification. On this coarse grid with dense
    obstacles it was mildly destabilizing, so the flagship keeps `qv`
    as a *passive* tracer instead (`moisture = false` on the sim) but
    still runs the Kessler microphysics on the qv/ql/qr fields. Users
    with finer grids or gentler forcing can flip this on.
  * `ν = 2.0` is far above physical (~10⁻²). The bulk viscosity substitutes
    for a proper implicit-diffusion wall-model near obstacles. Real
    high-Re LES would need cut-cell IBM or a wall function.
  * `dt = 2 s` is CFL-limited by the ~15 m grid + ~2 m/s peak flow;
    finer grids need shorter dt (proportional to dx).
  * Snapshots are five static PPMs at fixed times, not an animated
    sequence. Convert to APNG or GIF externally if desired.

## Animated multi-field playback

`examples/urban/alexanderplatz_animation.jl` is a **pure renderer**. It
runs no physics at all: it reads the NetCDF output of the flagship run
(`examples/urban/configs/berlin_summer.toml`) and stitches an
8-panel mosaic into a **single self-contained APNG** that plays in any
modern browser (Chrome, Firefox, Safari) or in macOS Preview.

```bash
$ julia --project -m Aeolus run examples/urban/configs/berlin_summer.toml --set output.cadence=600
$ julia --project=. examples/urban/alexanderplatz_animation.jl # ~1.5 s
```

!!! note "Why rendering is separate from running"
    These used to be two scripts that each ran their own independent
    24-hour simulation, because capturing a frame needs the full state
    at that instant and it was easier to duplicate the loop than to save
    state. That was never a design decision, and it cost:

      * the physics had to be maintained twice;
      * the two copies **drifted** — one had soil conduction, latent
        heat, chemistry and aerosol, the other had none of them, while
        this page claimed both ran "the same physics";
      * that drift hid an energy-conservation bug in `step_soil!`, since
        the script exercising the soil model was not the script whose
        diagnostics were read;
      * every animation cost a second full integration, and re-colouring
        a panel meant re-running the physics.

    Now there is one simulation. It writes 2-D slices (~295 kB per
    frame — only what a horizontal panel needs), and re-rendering takes
    **1.5 s instead of 50 minutes**.

Panels (2 rows × 4 columns):

| top row              | bottom row                 |
| :------------------- | :------------------------- |
| Surface temperature  | Horizontal wind speed `\|uh\|` |
| Direct-SW shadow mask (bright = sunlit, dark = shaded) | Pressure anomaly `p'` (z ≈ 30 m) |
| Potential-temp anomaly `θ'` (z ≈ 30 m) | Pollutant tracer `c` (z ≈ 30 m) |
| Vertical velocity `w` (z ≈ 30 m) | Sky-view factor (static geometric reference) |

Each panel is colour-clamped to fixed limits so amplitudes stay visually
comparable across frames. The elapsed simulation time (`t 06.0h` …
`t 06.0h` wrapping through midnight) is printed on the mosaic.

**Encoder details.** No `ffmpeg` / `ImageMagick` / `libpng` dependency
— `examples/viz/apng.jl` is a ~200-line pure-Julia encoder that writes
uncompressed-DEFLATE PNGs and stitches them into APNG animation
chunks (acTL / fcTL / fdAT). File sizes are ~2× a normally compressed
PNG but the encoder has zero external deps and runs identically on
NixOS, macOS, Windows.

**One-shot run**:

```bash
$ julia --project=examples examples/urban/alexanderplatz_animation.jl
```

Representative result:

```
Alexanderplatz OSM: 322 buildings; solid=3111 wall=978
Animation run: 43200 steps, 48 frames expected
  frame  1 @ t= 6.5h  wall=    8s  Ts=285.0/305.0  max|u|=1.10
  frame  8 @ t=10.0h  wall=   64s  Ts=285.0/315.4  max|u|=1.29
  frame 16 @ t=14.0h  wall=  128s  Ts=290.5/329.5  max|u|=1.56
  frame 24 @ t=18.0h  wall=  191s  Ts=294.6/331.9  max|u|=1.72
  frame 32 @ t=22.0h  wall=  253s  Ts=294.4/318.4  max|u|=1.65
  frame 49 @ t= 6.0h  wall=  374s  Ts=292.9/304.9  max|u|=1.48
Ran 43200 steps in 374.5 s. Captured 49 frames.
APNG written: examples/urban/output/alexanderplatz_animation.png
              (57.8 MB, 49 frames, 768×512)
```

**Viewing.** Open the file in any modern browser — the APNG loops
automatically. To convert to a smaller file, use `apngasm` (once
installed): `apngasm output.gif input.png` (auto-detects APNG →
falls back to a GIF).

**Building your own animation.** The `apng.jl` module exposes three
public entry points you can compose:

  * `write_apng(path, frames; delay_ms, loop)` — write RGB frames
    (`Array{UInt8, 3}` each) to disk.
  * `scalar_to_rgb(Z, vmin, vmax; upscale)` — turn a 2-D scalar array
    into a viridis-mapped RGB panel.
  * `tile_frames(panels)` — combine a matrix of panels into one bigger
    RGB frame.
  * `add_label!(rgb, text; row, col)` — a minimal 5×7 bitmap font for
    time-stamps and titles (ASCII 0-9 A-Z a-z `.-:_ /`).

## Real Berlin — Alexanderplatz OSM run

`examples/urban/configs/berlin_summer.toml` is **the** simulation, run by
`julia --project -m Aeolus run`: no script in `examples/urban` integrates
the equations (#438). The run type is `Aeolus.UrbanCase` (the `urban`
driver; `src/UrbanCase.jl`), with every physics coupling in its own file
under `src/urban/couplings/`. The [Berlin case design page](../design/berlin_case.md)
describes the registry, the stage loop and how to add a coupling.
Its configuration is `examples/urban/configs/berlin_summer.toml`;
`--set clock.nsteps=…` sets the number of steps and `--set output.cadence=…`
the seconds between NetCDF records. It runs
the full physics against `data/alexanderplatz_real.osm` — an actual Overpass-API
download of a 1 km × 1 km box around Alexanderplatz, Berlin
(52.518–52.526 N, 13.402–13.418 E). To (re-)fetch:

```bash
$ julia --project -m Aeolus preprocess fetch-osm --box 52.518,13.402,52.526,13.418 --out examples/urban/data/alexanderplatz_real.osm
$ julia --project -m Aeolus preprocess static examples/urban/configs/berlin_summer.toml
```

(Uses the public `overpass-api.de/api/interpreter` endpoint. Retry if
the server is busy — the fetch script does not currently auto-retry.)

Representative run summary — genuine OSM geometry, all physics on:

```
Alexanderplatz OSM: 322 buildings, 132 patches loaded
Solid cells: 3111,  wall-tagged: 978,  park: 511,  water: 2
Computing sky-view factor... done in 0.5s, SVF ∈ [0.368, 1.000]
Alexanderplatz diurnal: 43200 steps × 2 s = 24 h
  t=10.0h  SW= 847 W/m²  T_str=305.4 T_roof=313.4 T_wall=299.7 T_park=296.2
  t=14.0h  SW= 847 W/m²  T_str=329.6 T_roof=338.9 T_wall=318.0 T_park=311.4
  t=18.0h  SW= 228 W/m²  T_str=332.9 T_roof=334.4 T_wall=323.5 T_park=316.7
  t=22.0h  SW=   0 W/m²  T_str=318.0 T_roof=312.5 T_wall=314.7 T_park=310.8
  t= 4.0h  SW=   0 W/m²  T_str=303.5 T_roof=296.6 T_wall=304.6 T_park=303.1
Completed 24 h at Alexanderplatz in 1006 s wall time.
Peak ΔT (heat island): 15.9 K   Streets vs parks: 17.9 K   Walls vs parks: 6.9 K
max|∇·u| = 3.25e-15
```

**Now with ray-marched building shadows and sky-view-factor-weighted
diffuse SW and LW** (post-Phase-20 items **R1** and **R2** in the
refinement plan):

  * At each step the per-cell shadow mask is refreshed by casting a ray
    from each cell's top-of-stack toward the sun. Ray marching uses the
    same `SolidMask` the IBM uses, so buildings cast physically-correct
    shadows.
  * Diffuse SW is scaled by the pre-computed sky-view factor (SVF); LW
    down blends `320 W/m² · svf + 380 W/m² · (1 - svf)` to model the
    warm LW emitted by surrounding walls in urban canyons.
  * The physical pattern now reads correctly:
    - Roofs (open exposure) peak highest at noon (339 K)
    - Streets have thermal lag → peak at 18h (333 K)
    - Walls track streets but with more radiative trapping at night
    - Parks stay cool all day (albedo + `C_s`)
    - **Nocturnal urban heat island**: at 4 am streets are the warmest
      class (303.5 K) while roofs are the coldest (296.6 K, they radiate
      to open sky).

Notable geometry:

  * 322 real Berlin building footprints (out of 373 in the raw OSM
    extract; the example filters `min_height ≥ 8 m` to skip 0-m
    artefacts).
  * Median building height ~12 m (typical residential); tallest is
    the Fernsehturm at 368 m — clipped to the 200 m domain top.
  * 511 park-tagged cells and 2 water cells (Alexanderplatz has
    generous open plazas + surrounding green space).
  * Same qualitative physical result as the synthetic sample: 18 K
    street-vs-park contrast; divergence at machine precision throughout.

Outputs land in `examples/urban/output/alexanderplatz_*`:
`heights.ppm` / `class.ppm` (geometry), `surfT_XXh.ppm` (5 snapshots),
`pollutant_XXh.ppm`, `particles_final.ppm`, `class_surfT.svg`,
`sw_forcing.svg`, `final_theta.svg`, `surfT_final.ppm`.

### Ground: the production soil column

**Since #129** the soil column is no longer under a slab: each street,
building-adjacent and park column's surface *is* a `SoilThermalColumn`,
stepped by the Berlin case's surface exchange with Monin–Obukhov fluxes
from the first air cell, with its soil water (`hydrology = true`), so β
follows the top layer. Roofs, water and glass are `Radiation.SlabLand`
surfaces. `UrbanSoil` / `step_soil!` and `couplings/soil_urban.jl` are
gone; `couplings/ground_soil.jl` registers the soil columns. See
`docs/src/design/berlin_surface_radiation.md` for the design and the measured
effect. The rest of this section describes the #18 arrangement and its
measurements.

Below each non-water cell's surface slab the run carried one
`SoilThermalColumn` from `Aeolus.SoilColumn` (Phase 46, #18). It replaces
a three-layer explicit soil that was written into the example. The
example's `UrbanSoil` / `step_soil!` (in `couplings/soil_urban.jl`) only
couples that column to the slab, and does no soil physics of its own:

- **The column.** It uses the 10 CLM2–4.5 exponential layers down to
  3.43 m, with the top node at 7 mm. The soil is a moist loam at
  θ = 0.20 (k = 1.64 W m⁻¹ K⁻¹, C = 2.10 MJ m⁻³ K⁻¹), and the base is
  held at 288 K. Soil water is not stepped.
- **The coupling.** The slab (Ts, `C_s`) reaches the soil surface through
  a contact conductance h = 6 W m⁻² K⁻¹, the skin coupling of the old
  scheme. Folding the slab into
  `LinearizedSkinBalance(0, -1/(1/h + Δt/C_s), Ts)` makes the exchange
  implicit, and the slab loses exactly the ground heat flux the soil
  gains.
- **Sub-cycling.** The exchange runs every 15 dynamics steps (30 s). The
  24 h run takes about 9 % longer.
- **New output.** The NetCDF output gains `T_soil1` (top-layer soil
  temperature) and `G_soil` (ground heat flux).

`alexanderplatz_surface_diagnostics.jl` summarises a run, or compares two
runs, from the NetCDF alone. It reports per-class means for sunset at
19:30 local:

| class-mean Ts, 24 h from 06:00 | three-layer soil | production column |
| :-- | --: | --: |
| street max (15:30) / night min (04:00–04:30) | 324.86 / 301.92 K | 324.54 / 301.40 K |
| roof max (15:00) / night min (04:00) | 330.21 / 300.07 K | 329.71 / 299.47 K |
| wall max (17:30) / night min (04:00) | 305.66 / 300.09 K | 305.48 / 299.72 K |
| street cooling, sunset → night min | 14.23 K (1.67 K h⁻¹) | 14.27 K (1.59 K h⁻¹) |
| roof cooling, sunset → night min | 16.25 K | 16.16 K |

The diurnal cycle is essentially unchanged. The slab–soil exchange is
limited by the contact conductance in both schemes, and the production
column differs mainly in resolving the upper centimetres of the soil:

- The street top-layer soil temperature swings through 13.5 K.
- Street G peaks at 148 W m⁻² into the ground and reaches −5 W m⁻² at
  night.

What changes is what the ground can represent: moisture-dependent
conductivity, frozen ground with latent heat, and a column that the
land-surface model (#21) can share.

## Adding your own city

Two ingest formats are supported.

### GeoJSON

The minimal schema is:

```json
{ "type": "FeatureCollection",
  "features": [
    { "type": "Feature",
      "properties": {"height": 60, "class": "roof"},
      "geometry": {"type": "Polygon",
                   "coordinates": [[ [x, y], [x, y], ... ]]}},
    ...
  ]
}
```

Coordinates are meters relative to the mesh origin. Recognised `class`
values: `roof`, `street`, `park`, `water`. See
`examples/urban/data/city.geojson` for a working example, and
`src/preprocess/geojson.jl` for the loader and `src/preprocess/city.jl`
for the rasteriser (`src/preprocess/osm.jl` for OSM XML), all in
`Aeolus.InputDrivers`. A polygon with holes (a courtyard) lists its
rings after the outer one. `aeolus preprocess static <config.toml>`
writes the rasterised city to a static driver (NetCDF-4) that the run
then reads (see the [input drivers](../api/input_drivers.md) page).

### OSM XML

The `load_city_osm(path; lat0, lon0, x_shift, y_shift, min_height,
default_height)` function reads a subset of standard OSM XML:

  * `<node id lat lon />` for coordinates,
  * `<way id>…<nd ref/>…<tag k=… v=…/></way>` for polygons.

Ways tagged `building=…` become buildings with height taken from a
`height` or `building:levels` tag (fallback `default_height`).
Ways tagged `leisure=park`, `landuse=grass|forest|meadow`,
`natural=water`, `water=lake|pond|river` become surface patches.

Latitude/longitude coordinates are projected to local meters via
equirectangular projection anchored at `(lat0, lon0)`; `x_shift` /
`y_shift` re-center the extract inside the mesh; `min_height` filters
short buildings for stability tuning.

The bundled `data/sample_berlin_mitte.osm` was produced by
`aeolus preprocess osm-sample --lat0 52.5220 --lon0 13.4100` — a plausible European perimeter-block
neighbourhood at Alexanderplatz coordinates. Genuine Overpass-API
downloads are consumed by the same parser.
