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
    ledger.jl          energy/water budget ledger (surface energy per class, water in the air)
    reference_state.jl hydrostatic reference p̄, ρ̄, T̄, Π̄ (CaseReference)
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
interval, `soil_urban` and `buildings_citymean`, pass the nominal period
`P` rather than the time elapsed since they last ran. That is exact only
when `dt` divides `P`; the adaptive-dt issue (#133) has to pass the
elapsed interval instead.

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
| dynamics: `step!` with `:body_force` / `:post_predictor` | `geostrophic` (10), `moist_buoyancy` (20) / `ibm` (10), `veg_drag` (20) |
| `:surface_update` | `surface_slab` (10), `soil_urban` (20, every 30 s), `snow_legacy` (30) |
| `:column_tendency` | `cooling_prescribed` (10) |
| `:sanity` | `surface_slab` Ts clamp (10), then the core finiteness check |
| (ledger) | every coupling's `budget!`, and the change of water in the air (core) |
| `:output` (1800 s) | NetCDF record and log line (core), then coupling entries |

The order is the order of the pre-refactor loop, floating-point order
included. That is why the slab's latent flux sits at the end of
`:anthropogenic`: the old loop added it to the surface loss after the
building, traffic and glazing terms. The surface-exchange issue (#129)
replaces the slab and folds it into `:surface_flux`.

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
  Couplings may add classes at `init!` (the glass coupling does). The
  scene is built with `class_map = scene_class_map(sfc)`, so the ground
  facet of every open column has the material of its class (park, water,
  street; wall-tagged cells are street ground).
* **Reference state** (`case.ref`, see [below](#Reference-state-and-moist-buoyancy)):
  the hydrostatic `p̄`, `ρ̄`, `T̄`, `Π̄` per level. Every coupling that
  needs a pressure, converts a flux into a mixing-ratio or θ increment,
  or turns θ into T takes it from here (`ref_pressure`, `ref_density`,
  `ref_exner`, `surface_pressure`, `ref_ρcp`), never from a constant of
  its own.

**Modifying `bus.sw`.** The radiation provider writes `bus.sw` only at
the radiation cadence; the `:sw_modifiers` stage runs every step. A
modifier must therefore act on the unmodified field, not on its own
previous output. `sw_modifiers` keeps that field (`sw_base`), takes a
new one whenever `bus.sw` differs from what it wrote on its previous
step, and rebuilds `bus.sw = cloud(shade(sw_base))` each step. A second
modifier added later belongs in the same pattern. Before #128 the tree
shade and the cloud shadow multiplied `bus.sw` in place, so they
compounded between radiation updates.

## Reference state and moist buoyancy

The Boussinesq core carries neither pressure nor density, and before
#131 each coupling assumed its own: fog and dew an isothermal 8 km
scale height from 1013.25 hPa, the surface, building and traffic
fluxes and the ledger `ρ = 1` (`ρcp = 1005`). Since #131
`core/reference_state.jl` builds one hydrostatic profile per case,
`Aeolus.Anelastic.reference_from_profile` of the initial θ(z) from
`cfg.p_surface` (101 325 Pa at the ground), with the dry-air constants
of `MoistThermodynamics` (so `T = θ Π̄` is the temperature the fog
adjustment and the saturation functions assume) and the `g` of the
dynamics. It is exactly (1e-12 Pa m⁻¹) in the discrete balance of the
anelastic core and 5e-8 from an RK4 integration of the continuous
hydrostatic equation. The isothermal profile it replaces was 0.17 % off
at the flagship's top level; the density is 1.20 kg m⁻³ at the ground
in summer, 1.30 in winter.

| consumer | uses |
| :-- | :-- |
| `fog_dew` | fog adjusted at `p̄_k` (`T = θ Π̄_k`); dew at `p_surface` with `ρ̄_1` |
| `surface_slab` | `ρ̄_1 c_pd` and `ρ̄_1` in the bulk H and LE and their conversion to θ, qv; q_sat at `p_surface` |
| `buildings_citymean`, `traffic_heat` | `ρ̄_k c_pd`, `ρ̄_k` for the release into level k |
| the ledger | `ledger.ρ = ρ̄`: the water in the air is integrated with the density the couplings use, so the water ledger still closes |
| `moist_buoyancy` | `Π̄_1`, `p̄_1` for the RH and dew point it reports |

**Moist buoyancy** (`couplings/moist_buoyancy.jl`, on by default). The
core adds only the dry buoyancy `g (θ − θ₀)/θ₀` (`sim.moisture = false`:
the water lives in the tracer registry). In `:body_force` the coupling
adds the rest of the density-potential-temperature buoyancy,

    g (θ_ρ − θ)′ / θ₀,     θ_ρ = θ (1 + q_v/ε − q_t) ≈ θ (1 + 0.608 q_v − q_c),

from `MoistThermodynamics.density_temperature`, at every interior w face
(the two adjacent centres averaged, as the core does for θ). The prime
is the deviation from the mean over the fluid cells of the level: a
horizontally uniform force is a pure pressure gradient in the
incompressible core, so subtracting it only changes the diagnosed
pressure. The condensate `q_c` is every tracer the ledger counts as
water other than `qv` (`ql` and `qr` today), so a microphysics provider
that calls `water_tracer!` has its species loaded without touching this
coupling. It writes `b_moist` (m s⁻²) to the NetCDF file and a report
line (rms and max of b′, max RH and min dew-point depression of the
lowest layer).

In the flagship configuration (1 h from 06:00, no fog) it costs 0.86
ms/step. At the end of the hour the rms moist buoyancy is 8.8e-4 m s⁻²
(max 1.8e-2, the buoyancy of 0.5 K), a quarter of the rms dry buoyancy
perturbation (3.6e-3). In the same hour run without the coupling, the
moist buoyancy the state would have is twice as large (rms 1.8e-3,
max 3.4e-2): with the term on, the light, humid air the buildings and
surfaces release rises and mixes instead of accumulating as anomalies.
The two runs also differ in the minimum level-1 humidity (4.8 g/kg with,
3.3 without) and in dew deposition (1.6e-3 against 2.9e-3 kg m⁻²), but
after an hour of turbulence part of that is the divergence of two
chaotic runs, not the term alone.

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

Who books what in the default case: `surface_slab` books `Rn` (computed
from `Ts_start` exactly as `step_surfaces!` does), `H`, `LE`, the
storage `C_s ΔTs` over the whole step (the soil exchange and the clamp
included) and the energy its Ts clamp adds (`clamp`); `soil_urban`
books `G·P` on the step it exchanges; `fog_dew` the dew's latent heat
(`dew_heat`); `buildings_citymean` the window sun that leaves the wall
cells (`window_sun`); `glass` the shortwave transmitted into the
building (`glass_transmission`).

**Water.** `water_budget(ledger)` gives the change of water in the air
`ΔW` (measured by the ledger over `qv`, `ql`, `qr` and every
`water_tracer!`), the booked sources and `residual = ΔW − Σ sources`.
Sources in the default case: `evaporation` (slab), `building_vapour`,
`dew` (deposition, negative), `precipitation` (Kessler rain out of the
bottom), `qv_clamp` (the Kessler coupling's clamp) and `transport` (the
change of water the transport stage makes; zero for a conservative
scheme). The ledger integrates the water with `ledger.ρ`, one density
per level, the reference density `case.ref.ρ̄` (#131). A coupling that
puts a flux into the air with a different density than the ledger's
shows up as a water residual, so a coupling converts every flux with
`ref_density(case, k)`.

The incompressible core does not conserve that ρ̄-weighted water: ∇·u = 0
conserves Σ q, so vapour carried up into thinner air loses ρ̄-weighted
mass (an anelastic core, #133, would conserve Σ ρ̄ q). The transport
stage therefore books its change of water in two parts
(`ledger_water_split`): `transport`, the column-mean ρ̄ times its change
of Σ q (zero for a conservative scheme; today the #132 halo leak), and
`density_redistribution`, the rest. Over the 1 h flagship run
(06:00–07:00) with moist buoyancy on, `density_redistribution` is
−8.1e-4 kg m⁻² (72 % of the slab's evaporation, 1.13e-3) and
`transport` −7.0e-4; with it off, −6.3e-4 and +1.5e-5. Without the
split both would be booked as `transport` (−1.5e-3, against −2.7e-4 at
ρ = 1 in the #127 baseline), and fixing the #132 halo leak could not
bring it to zero. Kessler
sedimentation has the same effect inside its `precipitation` term
(rain falling into denser levels), which #143 owns.

The water integral runs over every grid cell, solid ones included.
Today the slab's evaporation and the dew exchange act on level 1 of
every column, which under a roof is inside the building (#129 moves
them to the first air cell); the ledger closes either way, so a closed
water ledger does not show that water sits in solid cells.

`run_case!` prints `ledger_report(case)` at the end of a verbose run.
The cost is about 1 ms/step on the flagship grid (the two water
integrals), plus about 0.4 ms/step for the Kessler coupling's own
bookkeeping.

### Baseline (measured, #127)

(Before the reference state of #131. With it the slab's fluxes carry
ρ̄_1 ≈ 1.20 instead of 1, so in the same hour the domain H is −23.2
instead of −19.7 W m⁻², and the moist buoyancy changes the flow; see
the #131 PR for its ledger.)

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
  step between radiation updates, so it is effectively zero. **Fixed in
  #128**: see "After #128" below.
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

**Correction (#131).** That seeded layer was not saturated for the fog
scheme: the seed set `q_v = q_s(θ, p)`, treating θ as the temperature,
while the fog adjustment uses `T = θ Π(p)`, about 1.1 K warmer at
1013 hPa. The air was at 93.5 % RH and the fog evaporated completely on
the first step (ql after one `:fog_dew` call: 0 of the seeded amount;
after 12 steps 1.6 %). Seeded at `q_s(θ Π̄_k, p̄_k)` it persists (99.7 %
after the first call, 90 % after 12 steps). With a fog that stays, the
compounding cloud shadow (#128) darkens the smoke window: domain Rn
34.8 W m⁻² instead of 388.7, residual −53.2 W m⁻² (dew heat +60.5,
window sun −6.9, glass −0.4) in summer and +5.5 in winter.

### After #128 (shade applied once, radiosity class map)

Same 1 h flagship configuration. The park shortwave (mean over the 511
park columns) is 116.05 W m⁻² averaged over the hour, against 0.62
before; the park / open-street ratio is 0.288–0.298 on every step
(before: 0.298 on a radiation step, then 0.086, 0.025, 0.007, …, 0 to
three decimals after 180 steps). The ratio is a little above
exp(−0.5·2.5) = 0.2865 because the unshaded effective irradiance of the
park columns is 0.4–4 % above that of the open streets (the park is less
shaded by buildings in the low morning sun). Per class, W m⁻² over the
hour:

| class | Rn before | Rn after | ΔS after | dew heat after |
| :-- | --: | --: | --: | --: |
| park | −9.95 | 80.47 | 119.74 | 3.43 |
| street | 327.47 | 327.63 | 350.21 | 0.28 |
| water | 389.85 | 389.95 | 381.12 | — |
| domain | 244.28 | 255.71 | 263.72 | 1.39 |

The domain residual Rn − H − LE − G − ΔS is 17.07 W m⁻² (16.89 before);
it is still the window sun, glass transmission and dew heat. The park and
water ground facets now carry their own material in the radiosity solve
(511 park and 2 water facets; before, all 3181 ground facets were
street). For flat ground this changes the absorbed shortwave but not the
effective irradiance `absorbed / (1 − α)` the slab receives, so the park
shortwave at the first radiation step moves only from 99.409 to 99.427
W m⁻²; the brighter park raises the reflected light on its neighbours
(street 333.190 → 333.212).

Around noon (900 steps, 11:45–12:15, otherwise the same flagship
configuration) the park / open-street ratio is 0.2840–0.2860 on every
step against exp(−1.25) = 0.2865, and the park shortwave is 234.83
W m⁻² averaged over the half hour (1.36 before). Park Rn goes from
−10.12 to +174.01 W m⁻²; domain Rn from 495.61 to 518.76.

In the `:smoke` configuration the summer domain residual is −33.9 W m⁻²
after #128.

## Smoke test and coverage test

**`test/berlin_smoke.jl`** runs the case in `CaseConfig(:smoke)`: a
250 m × 250 m × 100 m window of the real OSM file (16 × 16 × 16,
dx = 15.6 m, 312.5 m south of the anchor, 69 building and 50 park
columns, tallest roof 31 m) at local noon, 12 steps of 2 s, radiation
every 8 s, soil every 4 s. Before the first step it seeds a fog layer
from a fixed RNG seed (`Xoshiro(127)`): air saturated at the reference
state (`q_v = q_s(θ Π̄_k, p̄_k)`) with ql = 0.2 g/kg
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

## Cost of per-facet surfaces (#138)

The flagship radiosity scene has this many facets:

| `wall_subdivisions` | facets | roof | wall | ground |
|--:|--:|--:|--:|--:|
| 1 | 5810 | 915 | 1714 | 3181 |
| 2 | 7524 | 915 | 3428 | 3181 |
| 4 | 10952 | 915 | 6856 | 3181 |

The `:smoke` window has 314 facets.

A coupling that gives each facet its own `UrbanFacetScene` column (#139)
pays two costs:

* **Once, at `init!`:** the longwave response ``J``. This is one LU,
  9 s at 5810 facets and 18 s at 7524 on one BLAS thread (3 s at 5810 on
  8 threads). ``J`` then stays resident, at 258 MB and 432 MB.
* **Per step:** `step_urban_facets!` is matrix-free (GMRES, one ``O(n^2)``
  product with ``J`` per iteration). It takes 0.31 s at 5810 facets and
  0.47 s at 7524, on one thread, with 7 iterations. The dense LU it
  replaced took 4.5 s and 9.3 s.

Pass the radiation coupling's `bus.swsol` as `sw`, so the shortwave is
not solved a second time.

The response is rebuilt whenever a facet emissivity changes, for
example under snow (#146). To avoid that, change emissivities only at
the radiation cadence.

Reproduce these numbers with `julia --project=. benchmarks/bench_urban_facets.jl [wall_subdivisions] [blas_threads]`.

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
| tree shade and cloud shadow multiply `bus.sw` in place every step, so they compound between radiation updates | `sw_modifiers.jl` | #128 (fixed) |
| no `class_map` in the radiosity scene (park/water facets get street material) | `core/geometry.jl` | #128 (fixed) |
| slab: constant C_h and U, flux clamps, roof fluxes into solid k = 1, no surface drag | `surface_slab.jl`, `fog_dew.jl` | #129 |
| sky longwave from the surface temperature, prescribed −2 K/day cooling | `radiosity_sun.jl`, `cooling_prescribed.jl` | #130 |
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
