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
    bus.jl             SurfaceBus (incl. the per-column surfaces), SurfaceParams, surface factories
    tracers.jl         tracer registry (ensure_tracer!)
    ledger.jl          energy/water budget ledger (surface energy per class, water in the air)
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
| `modules_used(::Type{C})` | Aeolus submodules it runs (checked by the coverage test) |
| `default_options(::Type{C})` | `NamedTuple`, merged with `cfg.options[name]` and passed to `build` |
| `incompatible(::Type{C})` | `slot => provider` pairs or coupling names it cannot run with |
| `stages(::Type{C})` | `((stage, order, Cadence), …)`, at most one entry per stage |
| `build(::Type{C}, case, opts)` | construct the instance (default `C()`) |
| `init!(c, case)` | after every coupling is built, in registry order |
| `run_stage!(c, case, ::Val{S}, dt)` | the work of stage `S` |
| `output_fields(c, case)` | `OutputField`s added to the NetCDF file |
| `budget!(c, case, ledger)` | this step's surface-energy and water terms (see [the ledger](#The-budget-ledger)) |
| `report(c, case)` | an extra log line with each output record |
| `smoke_variants(::Type{C})` | extra smoke-test configurations, a tuple of `NamedTuple`s (see [the smoke test](#Smoke-test-and-coverage-test)) |
| `smoke_check(c, case)` | `Vector{Pair{String,Bool}}` of physics checks on the state after a smoke run |

`selected_couplings(cfg)` checks the configuration: an unknown key, an
unknown provider, an additive switch used on a slot coupling, a slot
without exactly one default, and a declared incompatibility are all
errors rather than silently ignored.

**Cadences are in seconds.** `Cadence(600.0)` fires on the step that
crosses a multiple of 600 s, `floor(t_new/P) > floor(t_old/P)`;
`Cadence(:radiation; first = true)` names a period in `cfg.cadence` and
also fires on step 1. With `dt = 2 s` this is exactly the old
`n % 300 == 0 || n == 1` (checked over a whole day in
`test/berlin_case.jl`). With an adaptive `dt` the entry still fires
once per period crossed, but the couplings that integrate over their
interval, `buildings_citymean` and the surface exchange, advance by the
nominal period `P` rather than the time elapsed since they last ran.
That is exact only when `dt` divides `P`; the adaptive-dt issue (#133)
has to pass the elapsed interval instead.

## The stage loop

One step is a fixed sequence of stages; within a stage the entries run
in the integer `order` their couplings declare.

| stage | entries (order) in the default case |
| :-- | :-- |
| `:clock` | local hour (core) |
| `:radiation` (600 s, and step 1) | `radiosity_sun` (10), `buildings_citymean` (20) |
| `:sw_modifiers` | `sw_modifiers` (10) |
| `:surface_flux` | `surface_exchange` (10): the held H, E into the first air cell (step 1: the first surface step) |
| `:anthropogenic` | `buildings_citymean` (10), `traffic_heat` (20), `glass` (30) |
| `:chemistry` | `aq_chem` (10), `aq_aerosol` (20) |
| `:transport` | `transport_weno` (10) |
| `:microphysics` | `micro_kessler` (10) |
| `:tracers` | `pollutant` (10) |
| `:fog_dew` | `fog_dew` (10) |
| dynamics: `step!` with `:body_force` / `:post_predictor` | `geostrophic` (10) / `ibm` (10), `surface_exchange` drag (15), `veg_drag` (20) |
| `:surface_update` | `surface_exchange` (10: collects `bus.sensible`; every surface period steps all surfaces, soil included), `snow_legacy` (30) |
| `:column_tendency` | `cooling_prescribed` (10) |
| `:sanity` | the core finiteness check |
| (ledger) | every coupling's `budget!`, and the change of water in the air (core) |
| `:output` (1800 s) | NetCDF record and log line (core), then coupling entries |

The order was the order of the pre-refactor loop, floating-point order
included (#126). The surface exchange (#129) replaced the slab, whose
latent flux sat at the end of `:anthropogenic` and whose Ts clamp ran in
`:sanity` only to keep that order; see [Surface exchange](#Surface-exchange-(#129)).

`run_case!` prints the wall time of every entry (`timing_report`). In
the 1 h flagship run (1800 steps, first-call compilation included) the
largest entry is the WENO-Z5 scalar transport, about 17 ms/step, ahead
of `step!` itself (about 12.5 ms/step), then `fog_dew` (6.1),
`radiosity_sun` (5.5 averaged over all steps) and `soil_urban` (2.8);
everything else is below 1 ms/step. See the table in the #126 PR.

## Shared state

Couplings never keep a private copy of a quantity another coupling
needs.

* **`SurfaceBus`** (`case.bus`, per column): `Ts`, `sw`, `lw`, the
  extra surface loss `sensible` (energy other couplings take from or give
  to the surfaces this step — window sun, glazing transmission, dew heat;
  not the turbulent sensible flux), the dew reservoir `dew`, the clear-sky
  beam `sw_beam`, the per-class `params`, the last radiosity solutions
  `swsol`/`lwsol`, and (#129) the per-column `surface`
  (`AbstractSurface`), the roof-aware air level `k_air`, the forcing each
  surface last saw (`T_air`, `q_air`, `U`, `z_ref`), the fluxes it
  returned (`Rn`, `H`, `LE`, `G`, `E`, `u_star`) and the `dew_mask`.
* **Surface factories.** A coupling that provides the surface of a
  class registers `register_surface_factory!(case, class, name, (case, I) -> surface)`
  in its `build`; the surface exchange builds every column's surface
  from these in its `init!` (a `SlabLand` for a class nobody claims).
  This is how the park (#140), water (#141) and facet (#139) couplings
  replace a class's surface without touching the surface exchange.
* **Tracers** (`case.tracers`): `ensure_tracer!(case, :qi; advect = true,
  positive = true, units = …, long_name = …)` creates or returns a
  cell-centred field; flags are OR-ed, so two couplings share one `ql`.
  The `:transport` stage advects every tracer with `advect = true` in
  registration order, and every tracer with `output = true` is written to
  the NetCDF file. The core registers `qv`, `ql` (advected) and `qr`.
* **Geometry** (`case.geom`): mask, surface classes, scene, view factors.
  Couplings may add classes at `init!` (the glass coupling does).

## Adding a coupling

This is the whole procedure. It touches one new file, one line of the
coverage test and the PR body; no shared file of the case is edited.

**1. Put the physics in `src/`.** A coupling only wires a src
implementation into the 3-D run. If the physics is missing, add it to
the right src module with its own tests first. Never copy it into the
example.

**2. Write `examples/urban/couplings/<name>.jl`.** One struct, the
contract methods it needs, and `register!` as the last line:

```julia
# P3 microphysics (Aeolus.P3Microphysics) in the :microphysics slot.

struct MicroP3{W} <: AbstractCoupling
    work::W
end

coupling_name(::Type{<:MicroP3}) = :micro_p3             # unique; key in cfg.options
slot(::Type{<:MicroP3})          = :microphysics         # or :addon
provider(::Type{<:MicroP3})      = :p3                   # cfg.physics[:microphysics] = :p3
is_default(::Type{<:MicroP3})    = false                 # an alternative: off by default
modules_used(::Type{<:MicroP3})  = (:P3Microphysics,)    # what the coverage test checks
default_options(::Type{<:MicroP3}) = (skip_dry = true,)
stages(::Type{<:MicroP3}) = ((:microphysics, 10, EVERY_STEP),)

function build(::Type{<:MicroP3}, case, o)
    ensure_tracer!(case, :qi; advect = true, positive = true, units = "kg kg-1",
                   long_name = "ice mass")               # transport picks it up
    return MicroP3(...)
end
init!(c::MicroP3, case) = water_tracer!(case.ledger, :qi) # count it as water

function run_stage!(c::MicroP3, case, ::Val{:microphysics}, dt)
    ...                                                  # calls into Aeolus.P3Microphysics
end

budget!(c::MicroP3, case, l) = add_water!(l, :precipitation, -c.surface_precip_step)

smoke_check(c::MicroP3, case) =
    ["ice ≥ 0" => minimum(interior_view(tracer(case, :qi))) >= 0,
     "precipitation ≥ 0" => c.surface_precip_step >= 0]

register!(MicroP3)
```

The file is picked up by the directory scan; include order is the
sorted file name, which is also the registry order (and the tie-break
between entries of the same stage and order).

* **On by default.** An additive coupling is on unless `is_default`
  says otherwise; a slot provider is the slot's default only if it is
  the one with `is_default = true`. The rule of #123: new physics is on
  by default unless it is a genuine alternative or too expensive, and
  then its cost per step is quoted from `timing_report` in the PR.
* **Cadence.** An expensive stage can run every `P` seconds:
  `Cadence(600.0)`, or `Cadence(:mykey)` with a default in
  `cfg.cadence` (add the key through the scenario or your option; a
  missing key is an error). Stages inside `step!` (`:body_force`,
  `:post_predictor`) must run every step.
* **Shared state** goes through the bus, the tracer registry and the
  geometry, never a private copy (see above). If you need a bus field
  that does not exist, add it to `core/bus.jl` additively and say so
  in the PR.
* **Replacing a coupling.** Delete the old file (the issue says which)
  and move its `budget!` terms and checks over. The ledger closure in
  the smoke test fails if a term that changes a surface temperature or
  the water in the air is lost.

**3. Report to the ledger.** If the coupling changes a surface
temperature, adds or removes water from the air, or clips anything,
book it in `budget!` (next section). The smoke test asserts the
closure, so a missing term fails the suite.

**4. Give it smoke coverage.** The smoke test already runs every
additive coupling and one configuration per non-default provider, so
most couplings need no `smoke_variants`. Add one for a configuration
that is not reached that way, for example an option that changes the
code path:

```julia
smoke_variants(::Type{<:MicroP3}) =
    ((label = "sgs cloud on", physics = Dict(:microphysics => :p3),
      options = Dict(:micro_p3 => (sgs_cloud = true,))),)
```

Each variant is a `NamedTuple`: `label`, then `physics` and `options`
(merged into the everything-on configuration), then any `CaseConfig`
field (`nsteps = 30`, `size = …`). Write `smoke_check`s that assert
physics: a conservation or a budget, a bound that follows from the
equations (a maximum principle, positivity, a range set by the
thermostat), a sign that follows from the physics. "It ran" is already
asserted for every coupling.

**5. Remove the module from the pending list.** Delete its line from
`BERLIN_PENDING` in `test/berlin_coverage.jl`. The coverage test fails
until you do (the ratchet), and it also checks that the coupling's file
really calls into every module in its `modules_used` (by an exported
name or a `Module.` qualification). An indirect use (you call a src
function that runs the module every time) goes in `BERLIN_INDIRECT`
with the explanation.

**6. Run and quote.** `julia --project=. test/berlin_smoke.jl` and
`test/berlin_coverage.jl` locally (under a minute), then the full suite.
In the PR body quote one Berlin diagnostic — a ledger term is often the
natural one — the configuration it was measured in, and the cost from
`timing_report`.

## The budget ledger

`case.ledger` integrates, step by step, the surface energy balance of
every surface class and the water budget of the air. The core calls
`ledger_begin_step!` before `:clock` (it snapshots `bus.Ts` into
`ledger.Ts_start` and the water in the air) and `ledger_end_step!` after
`:sanity` (it calls every coupling's `budget!` and books the change of
the water in the air). A coupling reports only what it did during that
step:

| call | meaning |
| :-- | :-- |
| `add_surface!(l, term, F, sfc, dt)` | a W m⁻² map `F` on the columns, over `dt` s, sorted by surface class |
| `add_water!(l, term, m)` | `m` kg m⁻² of domain added to the air (negative: removed) |
| `water_tracer!(l, :qi)` | count a new tracer as water in the air (not one that is part of another, like P3's rime mass) |
| `ledger_water(case, q)` | ``Σ_k ρ_k Σ_{ij} q Δz / (N_x N_y)``, for measuring a change of a tracer |

**Surface energy.** `:Rn` (net radiation into the surface), `:H`,
`:LE`, `:G` (sensible, latent and ground heat flux out of it) and `:dS`
(storage change of the surface layer) are the core terms. Every other
term name is an extra flux *into* the surface. `surface_budget(ledger)`
returns per class, and for the whole domain,

    residual = Rn − H − LE − G − ΔS        (what the four-term balance misses)
    closure  = residual + Σ extras          (round-off if nothing is missed)

The residual is a physical baseline: the terms this model has that the
conventional balance does not. The closure is bookkeeping and the smoke
test asserts it to 1e-6 W m⁻² (measured ≤ 3.5e-10).

Who books what in the default case: `surface_exchange` books, on each
surface step, the whole interval's `Rn`, `H`, `LE` and `G` (the heat
conducted into a soil column; a slab's storage `C_s ΔT` goes to `ΔS`
instead), and every step the change of the extra energy it holds for
the next surface step (in `ΔS`); `fog_dew` the dew's latent heat
(`dew_heat`); `buildings_citymean` the window sun that leaves the wall
cells (`window_sun`); `glass` the shortwave transmitted into the
building (`glass_transmission`). Before #129 the slab booked the same
terms per step and its Ts clamp as `clamp`; nothing is clamped now.

**Water.** `water_budget(ledger)` gives the change of water in the air
`ΔW` (measured by the ledger over `qv`, `ql`, `qr` and every
`water_tracer!`), the booked sources and `residual = ΔW − Σ sources`.
Sources in the default case: `evaporation` (surface exchange; negative
where the surface takes up dew), `building_vapour`,
`dew` (deposition, negative), `precipitation` (Kessler rain out of the
bottom), `qv_clamp` (the Kessler coupling's clamp) and `transport` (the
change of water the transport stage makes; zero for a conservative
scheme). The ledger integrates the water with `ledger.ρ`, one density
per level, `1` for the Boussinesq case. A coupling that puts a flux into
the air with a different density than the ledger's shows up as a water
residual: #131, which introduces a reference density profile, sets
`ledger.ρ` and the couplings must agree with it.

The water integral runs over every grid cell, solid ones included.
Before #129 the slab's evaporation and the dew exchange acted on level 1
of every column, which under a roof is inside the building; the ledger
closed either way, so a closed water ledger does not show where the
water went. Since #129 both act on the first air cell (`bus.k_air`), and
the surface exchange's smoke check counts writes into solid cells
(zero).

`run_case!` prints `ledger_report(case)` at the end of a verbose run.
The cost is about 1 ms/step on the flagship grid (the two water
integrals), plus about 0.4 ms/step for the Kessler coupling's own
bookkeeping.

### Baseline (measured, #127)

Flagship configuration, 1 h (`AEOLUS_NSTEPS=1800`, 06:00–07:00 local,
20 June, 64 × 64 × 32, dt = 2 s), mean W m⁻² over the hour per class:

| class | cols | Rn | H | LE | G | ΔS | Rn − H − LE − G − ΔS | extras | closure |
| :-- | --: | --: | --: | --: | --: | --: | --: | :-- | --: |
| street | 1690 | 327.47 | −18.45 | 0.88 | −4.72 | 350.04 | −0.27 | dew heat 0.27 | 1.7e-11 |
| roof | 874 | 309.18 | −15.83 | 0.44 | 1.51 | 323.21 | −0.16 | dew heat 0.16 | 2.3e-11 |
| park | 511 | −9.95 | −23.51 | 0.04 | −15.07 | 33.13 | −4.55 | dew heat 4.55 | 1.1e-12 |
| water | 2 | 389.85 | −9.36 | 18.23 | 0 | 380.98 | 0.00 | — | −3.5e-10 |
| wall | 978 | 172.33 | −24.03 | 0.05 | −12.26 | 145.08 | 63.49 | dew heat 3.39, window sun −66.89 | 1.4e-10 |
| glass | 41 | 309.22 | −4.53 | 4.31 | 21.27 | 43.87 | 244.29 | glass transmission −244.29 | 9.6e-12 |
| domain | 4096 | 244.28 | −19.71 | 0.53 | −6.22 | 252.79 | **16.89** | dew 1.52, glass −2.45, window sun −15.97 | 4.6e-11 |

Water over the same hour (kg m⁻² of domain): ΔW = −7.75e-4 =
evaporation +7.72e-4, building vapour +9.38e-4, dew −2.21e-3,
precipitation 0, clamp 0, transport **−2.70e-4**; residual −1.4e-14.

What the numbers say about the current (pre-#128/#129) physics:

* **The park has negative net radiation in the morning sun** (−9.95
  W m⁻² while the streets get +327). This is the compounding tree shade
  of #128: `bus.sw` over the park is multiplied by exp(−0.5·LAI) every
  step between radiation updates, so it is effectively zero.
* **H is negative everywhere** (the air heats the surfaces): the case
  starts with `Ts0 = 285 K` under air at θ = 293 K, and the slab's
  constant-coefficient flux (#129) is weak. The surfaces warm by storage
  (ΔS ≈ Rn) instead.
* **The transport is not conservative**: over the hour it removes
  2.7e-4 kg m⁻² of water, 35 % of what the surfaces evaporate. This is
  the stale-halo leak below (#132).
* The domain residual of 16.9 W m⁻² is the window sun (−16.0), the
  glass transmission (−2.4) and the dew's latent heat (+1.5): energy
  that goes into the buildings or comes from condensation, which the
  four-term balance does not see.

In the `:smoke` configuration of the smoke test (seeded fog, 12 steps
at noon, 16 × 16 × 16) the domain residual is −34.4 W m⁻² in summer
(dew heat +44.4 from the saturated seeded layer, window sun −6.9, glass
−3.0) and +6.1 W m⁻² in winter.

## Smoke test and coverage test

**`test/berlin_smoke.jl`** runs the case in `CaseConfig(:smoke)`: a
250 m × 250 m × 100 m window of the real OSM file (16 × 16 × 16,
dx = 15.6 m, 312.5 m south of the anchor, 69 building and 50 park
columns, tallest roof 31 m) at local noon, 12 steps of 2 s, radiation
every 8 s, soil every 4 s. Before the first step it seeds a fog layer
from a fixed RNG seed (`Xoshiro(127)`): saturated air with ql = 0.2 g/kg
± 10 % in the lowest three levels and rain qr < 5e-5 in levels 2–3, so
dew, fog, sedimentation and precipitation all do real work. The
flagship's pollutant stacks are outside this window, so the smoke runs
put one stack in it, 40 m up.

The run matrix is derived from the registry (`smoke_matrix()` in the
test file):

1. **summer, everything on** — every additive coupling on, including
   those off by default, and the default provider of every slot;
2. **winter, everything on** — the same with the `:winter` initial
   state;
3. one run per **non-default provider** of every exclusive slot;
4. one run per entry of every coupling's **`smoke_variants`**.

A run that selects a coupling declared `incompatible` with an additive
coupling switches that additive coupling off. Every run must finish
without a blow-up, leave every field and tracer finite, pass every
`smoke_check` of its couplings, and close the surface-energy ledger per
class (1e-6 W m⁻²) and the water ledger (1e-12 kg m⁻²). The first run
also asserts that the domain gains radiative energy at noon (Rn > 0,
ΔS > 0) and records the transport's water non-conservation as
`@test_broken` (#132 turns it into a pass). Finally every registered
coupling must have run in at least one configuration, and the whole
test must take less than 300 s.

Today the matrix is 2 runs (no slot has a second provider yet). Run on
its own, the test takes 26 s, almost all of it first-call compilation
(the two runs themselves take 0.3 s each once compiled).

**`test/berlin_coverage.jl`** checks that every submodule of `Aeolus`
is either listed in some registered coupling's `modules_used`, or in
`BERLIN_EXEMPT` with a reason, or in `BERLIN_PENDING` with the issue
that wires it in. It ratchets: a new src module fails the suite until
it is coupled or exempted, and a pending module that some coupling now
uses fails it until it is removed from `BERLIN_PENDING`, so that list
only shrinks. It also checks that each coupling's file calls into every
module it lists (an exported name, or a `Module.` qualification, in the
code with comments and docstrings removed); reviewed indirect uses go in
`BERLIN_INDIRECT`. At #127: 55 modules, 15 coupled, 11 exempt
(the core's infrastructure and the Parallel / Backends / SpectralElement
scaffolds), 30 pending.

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
| slab: constant C_h and U, flux clamps, roof fluxes into solid k = 1, no surface drag | `surface_slab.jl`, `fog_dew.jl` | #129 (fixed: `surface_exchange.jl`, `ground_soil.jl`) |
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

The ledger (#127) measures the same leak in water. On **step 1** it is
much larger: the velocity halos have never been filled when the
transport first runs (`set!` fills only the interior), and in the
`:smoke` window the first transport step *adds* 7.5e-3 kg m⁻² of water,
0.8 % of the water in the air (9.4e-1 kg m⁻²); every later step removes
about 3.5e-5 kg m⁻². With the halos refreshed the same steps change the
water by at most 1.3e-15 kg m⁻². Over the 1 h flagship run the
transport term is −2.70e-4 kg m⁻². `test/berlin_smoke.jl` records it as
`@test_broken`.
