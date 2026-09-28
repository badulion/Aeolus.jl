# The Berlin case

The Berlin (Alexanderplatz) run is the flagship of Aeolus: every
physics module in `src/` is meant to be exercised by it (the rule of
#123, in `CLAUDE.md`). Since #126 it is not a script but a small
framework, the `BerlinCase` module in `examples/urban/`, built so that
many couplings can be added independently and at the same time.

```julia
include("examples/urban/BerlinCase.jl")
using .BerlinCase

cfg  = CaseConfig(:alexanderplatz; nsteps = 1800)   # 1 h at dt = 2 s
case = build_case(cfg)
print(describe(case))                               # couplings and schedule
run_case!(case)                                     # NetCDF + log + timing table
```

`examples/urban/alexanderplatz_diurnal.jl` is exactly this, reading
`AEOLUS_NSTEPS` (default 43 200 = 24 h), `AEOLUS_OUTPUT_DT` (seconds
between records, default 1800) and `AEOLUS_SCENARIO`.

## Layout

```
examples/urban/
  BerlinCase.jl        module: includes core/*, then every couplings/*.jl in sorted order
  core/
    config.jl          CaseConfig — grid, window, clock, cadences (s), physics/options
    scenarios.jl       :summer (= :alexanderplatz), :winter, :smoke, :mitte_sample
    registry.jl        AbstractCoupling, the contract, Cadence, register!, selection
    geometry.jl        surface classes, rasterisation, build_geometry (scene, view factors)
    osm_ingest.jl      OSM XML reader
    bus.jl             SurfaceBus, SurfaceParams (per-class slab parameters)
    tracers.jl         tracer registry (ensure_tracer!)
    ledger.jl          energy/water budget ledger (stub, filled by #127)
    output.jl          NetCDF output driven by the registry
    case.jl            Case, build_case
    stages.jl          the stage loop: step_case!, run_case!, timing_report
  couplings/           one file per coupling, each ending in register!(Type)
  alexanderplatz_diurnal.jl              the run
  alexanderplatz_animation.jl            renderer (reads the NetCDF only)
  alexanderplatz_surface_diagnostics.jl  per-class diurnal summary (NetCDF only)
```

Adding a coupling means adding one file under `couplings/`. No shared
file is edited: the file is picked up by the directory scan, registers
itself, and the registry decides from the config whether it runs.

## The coupling contract

A coupling is a struct `<: AbstractCoupling` plus methods of the
contract below for its type. Only `coupling_name` and the `run_stage!`
methods of the declared stages are required; everything else has a
default.

| method | meaning |
| :-- | :-- |
| `coupling_name(::Type{C})` | unique `Symbol`; key in `cfg.physics` / `cfg.options` |
| `slot(::Type{C})` | `:addon` (additive, on/off) or an exclusive slot such as `:microphysics` |
| `provider(::Type{C})` | the value of `cfg.physics[slot]` that selects `C` |
| `is_default(::Type{C})` | default provider of its slot; for an addon, on by default |
| `modules_used(::Type{C})` | Aeolus submodules it runs (for the coverage test, #127) |
| `default_options(::Type{C})` | `NamedTuple`, merged with `cfg.options[name]` and passed to `build` |
| `incompatible(::Type{C})` | `slot => provider` pairs or coupling names it cannot run with |
| `stages(::Type{C})` | `((stage, order, Cadence), …)`, at most one entry per stage |
| `build(::Type{C}, case, opts)` | construct the instance (default `C()`) |
| `init!(c, case)` | after every coupling is built, in registry order |
| `run_stage!(c, case, ::Val{S}, dt)` | the work of stage `S` |
| `output_fields(c, case)` | `OutputField`s added to the NetCDF file |
| `budget!(c, case, ledger)` | budget terms (#127) |
| `report(c, case)` | an extra log line with each output record |
| `smoke_variants(::Type{C})`, `smoke_check(c, case)` | for the smoke test (#127) |

`selected_couplings(cfg)` checks the configuration: an unknown key, an
unknown provider, an additive switch used on a slot coupling, a slot
without exactly one default, and a declared incompatibility are all
errors rather than silently ignored.

**Cadences are in seconds.** `Cadence(600.0)` fires on the step that
crosses a multiple of 600 s, `floor(t_new/P) > floor(t_old/P)`;
`Cadence(:radiation; first = true)` names a period in `cfg.cadence` and
also fires on step 1. With `dt = 2 s` this is exactly the old
`n % 300 == 0 || n == 1` (checked over a whole day in
`test/berlin_case.jl`), and it keeps working with an adaptive `dt`.

## The stage loop

One step is a fixed sequence of stages; within a stage the entries run
in the integer `order` their couplings declare.

| stage | entries (order) in the default case |
| :-- | :-- |
| `:clock` | local hour (core) |
| `:radiation` (600 s, and step 1) | `radiosity_sun` (10), `buildings_citymean` (20) |
| `:sw_modifiers` | `sw_modifiers` (10) |
| `:surface_flux` | `surface_slab` (10) |
| `:anthropogenic` | `buildings_citymean` (10), `traffic_heat` (20), `glass` (30), `surface_slab` latent flux (90) |
| `:chemistry` | `aq_chem` (10), `aq_aerosol` (20) |
| `:transport` | `transport_weno` (10) |
| `:microphysics` | `micro_kessler` (10) |
| `:tracers` | `pollutant` (10) |
| `:fog_dew` | `fog_dew` (10) |
| dynamics: `step!` with `:body_force` / `:post_predictor` | `geostrophic` (10) / `ibm` (10), `veg_drag` (20) |
| `:surface_update` | `surface_slab` (10), `soil_urban` (20, every 30 s), `snow_legacy` (30) |
| `:column_tendency` | `cooling_prescribed` (10) |
| `:sanity` | `surface_slab` Ts clamp (10), then the core finiteness check |
| `:output` (1800 s) | NetCDF record and log line (core), then coupling entries |

The order is the order of the pre-refactor loop, floating-point order
included. That is why the slab's latent flux sits at the end of
`:anthropogenic`: the old loop added it to the surface loss after the
building, traffic and glazing terms. The surface-exchange issue (#129)
replaces the slab and folds it into `:surface_flux`.

`run_case!` prints the wall time of every entry (`timing_report`). For
the 1 h flagship run the dynamics dominate; see the table in the #126 PR.

## Shared state

Couplings never keep a private copy of a quantity another coupling
needs.

* **`SurfaceBus`** (`case.bus`, per column): `Ts`, `sw`, `lw`, the
  combined surface loss `sensible`, the dew reservoir `dew`, the clear-sky
  beam `sw_beam`, the per-class `params`, and the last radiosity
  solutions `swsol`/`lwsol`. #129 extends it with the MOST fields.
* **Tracers** (`case.tracers`): `ensure_tracer!(case, :qi; advect = true,
  positive = true, units = …, long_name = …)` creates or returns a
  cell-centred field; flags are OR-ed, so two couplings share one `ql`.
  The `:transport` stage advects every tracer with `advect = true` in
  registration order, and every tracer with `output = true` is written to
  the NetCDF file. The core registers `qv`, `ql` (advected) and `qr`.
* **Geometry** (`case.geom`): mask, surface classes, scene, view factors.
  Couplings may add classes at `init!` (the glass coupling does).

## Validation of the refactor (#126)

The refactor had to be bit-for-bit. On a 1 h run (`AEOLUS_NSTEPS=1800`,
two records) every NetCDF variable of the refactored case is identical
(`==`) to the pre-refactor script, and the printed frame lines match
apart from the wall-clock column.

**The pre-refactor script is not reproducible run to run.** Two 1 h runs
of the *same* old script, one with a freshly measured FFTW plan and one
with an imported plan, differed by up to 7.5e-9 m/s in `w`, 4.7e-10 m/s
in `v` and 6.0e-8 in the pollutant (Float32 output; θ, moisture and the
surface fields happened to agree). The cause is `FFTW.MEASURE` in the cosine-FFT
pressure solver: the planner times candidate algorithms and may pick a
different one in each process, with different round-off, which the flow
then amplifies. With the same FFTW wisdom imported into both processes
(`FFTW.import_wisdom`) the planner is deterministic, and old and new
cases then agree exactly in every variable. Bit-for-bit comparisons of
the Berlin case have to fix the FFTW plan this way.

## Known defects kept by the refactor

These are deliberately unchanged so the refactor could be checked
bit-for-bit; each is labelled `KNOWN` in its file and belongs to a
later issue.

| defect | file | fixed by |
| :-- | :-- | :-- |
| tree shade and cloud shadow multiply `bus.sw` in place every step, so they compound between radiation updates | `sw_modifiers.jl` | #128 |
| no `class_map` in the radiosity scene (park/water facets get street material) | `core/geometry.jl` | #128 |
| slab: constant C_h and U, flux clamps, roof fluxes into solid k = 1, no surface drag | `surface_slab.jl`, `fog_dew.jl` | #129 |
| sky longwave from the surface temperature, prescribed −2 K/day cooling | `radiosity_sun.jl`, `cooling_prescribed.jl` | #130 |
| isothermal pressure profile for fog/dew, no virtual buoyancy | `fog_dew.jl` | #131 |
| `qr` sedimented but not advected; qv clamp | `micro_kessler.jl` | #143 |
| glass is a synthetic roof mask | `glass.jl` | #144 |
| chemistry/aerosol species not advected or output; toy mechanism | `aq_chem.jl`, `aq_aerosol.jl` | #135, #147 |
| OSM `&&`/`||` precedence; patches overwrite roofs | `core/osm_ingest.jl`, `core/geometry.jl` | #141 |

**Found during #126: stale velocity halos in the scalar transport.**
`step!` corrects the interior velocity after the pressure solve but
leaves the halos at the predictor values. The Berlin case advects its
scalars (qv, ql with WENO-Z5, the pollutant with centred differences)
at the start of the next step, on those halos, so the velocity it sees
is not divergence-free across the periodic seam and the flux form stops
conserving: in the `:smoke` window after four steps the transport stage
changes the domain total of qv by −4.4e-5 and of ql by −3.5e-5 *per
step*. With the halos refreshed (`fill_halos!` on u, v, w) the same
stage conserves both to 1.5e-15 and 3.8e-15. The fix changes the
answer, so it is not in #126; `test/berlin_case.jl` records the leak as
`@test_broken` and asserts the conservation with fresh halos. The
scalar-advection issue (#132) owns it.
