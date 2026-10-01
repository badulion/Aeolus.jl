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
| `sim_options(::Type{C}, cfg, mesh, opts)` | keywords of the dynamical core's constructor (the `solver` of `BoussinesqSimulation`; `tol`, `maxit` of a sound-proof core), before the core is built (see [numerics slots](#Numerics-slots)) |
| `step_options(c, case)` | keywords added to every dynamics step, `step!` or `step_imex!` (e.g. `momentum_advection`, `eddy_viscosity!`) |
| `incompatible_reason(::Type{C})` | why the `incompatible` pairs fail, appended to the selection error |

Two slots drive the dynamics itself and have four more methods
(#133, see [Dynamical core and time integration](#Dynamical-core-and-time-integration)):

| method | meaning |
| :-- | :-- |
| `build_core(::Type{C}, cfg, mesh, ref, simkw)` | slot `:core`: construct the dynamical-core simulation (`case.sim`) on the case's reference state `ref` with the merged `sim_options` keywords `simkw`; a keyword it does not take is an error |
| `next_dt(c, case)` | slot `:timestepper`: the length of the next step (default `cfg.dt`) |
| `advance_dynamics!(c, case, dt)` | slot `:timestepper`: advance `case.sim` by `dt` with the composed `:body_force` / `:post_predictor` hooks and the numerics slots' `case.step_kwargs` (default `dynamics_step!`: one `step!`) |
| `variable_dt(::Type{C})` | slot `:timestepper`: `true` if `next_dt` varies; `run_case!` then runs to the simulated time `nsteps · cfg.dt` |

A keyword from `sim_options` or `step_options` is never dropped: every
time stepper passes `case.step_kwargs` to the step it takes, and a core
or stepper that cannot take a provider's keyword declares that provider
`incompatible` (with `incompatible_reason`), so the selection fails
with the reason instead.

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
once per period crossed, and the adaptive stepper caps `dt` at the
shortest `cfg.cadence` period so that no firing is skipped. The surface
exchange steps its surfaces (the soil columns of `ground_soil`
included) forward by its period `P` and gives the held fluxes to the
air; a step that runs past the end of a surface interval delivers them
only up to that end and carries the rest of the step into the next
interval (`air_delivery_time!`), so the air receives every interval's
fluxes for exactly `P` (smoke check, 1e-9 P; with a fixed `dt` dividing
`P` that is every step's `dt`, bit for bit). `buildings_citymean`
advances its zone over the *next* radiation interval, which it cannot
know under an adaptive `dt`; it uses the nominal `P`, so the zone and
the air it heats disagree by less than one step per interval (≤ 10 s of
600 s) and the difference does not accumulate. Every `budget!` books
with `ledger.dt`, the current step's length, never `cfg.dt`.

## The stage loop

One step is a fixed sequence of stages; within a stage the entries run
in the integer `order` their couplings declare.

| stage | entries (order) in the default case |
| :-- | :-- |
| `:clock` | local hour (core) |
| `:radiation` (600 s, and step 1) | `column_radiation` (5), `radiosity_sun` (10), `cloud_shadow` (15), `buildings_citymean` (20) |
| `:sw_modifiers` | `sw_modifiers` (10, tree shade) |
| `:surface_flux` | `surface_exchange` (10): the held H, E into the first air cell (step 1: the first surface step) |
| `:anthropogenic` | `buildings_citymean` (10), `traffic_heat` (20), `glass` (30) |
| `:chemistry` | `aq_chem` (10), `aq_aerosol` (20) |
| `:transport` | `scalar_advection` (10) |
| `:microphysics` | `micro_kessler` (10) |
| `:tracers` | `pollutant` (10) |
| `:fog_dew` | `fog_dew` (10) |
| dynamics: `advance_dynamics!` of the `:timestepper` provider (default one `step!`) with `:body_force` / `:post_predictor` | `geostrophic` (10), `moist_buoyancy` (20) / `ibm` (10), `surface_exchange` drag (15), `veg_drag` (20); the `step_options` of the numerics slots |
| `:surface_update` | `surface_exchange` (10: collects `bus.sensible`; every surface period steps all surfaces, soil included), `snow_legacy` (30), `aq_chem` r_a / v_d refresh (90, every 30 s) |
| `:column_tendency` | `column_radiation` (10) |
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
  `swsol`/`lwsol`, the sky longwave `L_sky` and the direct/diffuse
  shortwave `dni`/`dhi` the `:column_radiation` slot hands the radiosity
  solve (#130; `NaN` shortwave means "Bird"), and (#129) the per-column `surface`
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
| the sound-proof cores (#133) | `case.ref.state` itself as the base state: the projection, buoyancy and advection weights are this ρ̄, θ̄ (`build_case` builds the reference before the core) |
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

## Numerics slots

Four exclusive slots swap the numerics of the dynamical core and the
transport (#132). Select with `physics = Dict(:slot => :provider)`;
options go under the coupling name in `options`.

| slot | providers (default first) | how |
| :-- | :-- | :-- |
| `:pressure_solver` | `:cosine_fft` (`pressure_fft`), `:multigrid` (`pressure_multigrid`), `:mg_cg` (`pressure_mg_cg`), `:density_weighted_cg` (`pressure_density_weighted`, sound-proof cores only, #133) | `sim_options`: the `solver` of `BoussinesqSimulation`; `tol`, `maxit` of the sound-proof core's own CG |
| `:sgs` | `:smagorinsky` (`sgs_smagorinsky`, the built-in closure with `cfg.C_s`), `:dynamic` (`sgs_dynamic`), `:mlsgs` (`sgs_mlsgs`) | `step_options`: `eddy_viscosity!` |
| `:momentum_advection` | `:centered` (`momentum_centered`), `:weno_z` (`momentum_weno_z`, option `order = 5`/7) | `step_options`: `momentum_advection` |
| `:scalar_advection` | `:advection_schemes` (`scalar_advection`, option `scheme = :weno_z5`) | the `:transport` stage |

`sim_options` and `step_options` are contract hooks: a coupling returns a
`NamedTuple` of keywords for the core's constructor or for every
dynamics step (`step!`, or `step_imex!` under the IMEX steppers, which
takes the same `momentum_advection` and `eddy_viscosity!`); two
couplings may not set the same keyword. Which providers combine with the
`:core` and `:timestepper` slots is in [Dynamical core and time
integration](#Dynamical-core-and-time-integration).

* **Pressure solvers.** All three solve the same 7-point Laplacian with
  the same boundary conditions (periodic, periodic, Neumann); the
  multigrid ones iterate to a relative residual of 1e-10. In the
  `:smoke` window after 100 steps, MG-CG differs from the FFT by at most
  6.1e-10 m s⁻¹ in u, v, w and 2.1e-9 K in θ, with max|∇·u| = 1.9e-13
  s⁻¹ (`test/berlin_numerics.jl`). They are here for the stretched-grid
  and cut-cell solver work (#151, #152), not for speed.
* **SGS.** `:dynamic` is the Germano–Lilly coefficient of
  `Operators.compute_dynamic_smagorinsky_νt!`, plane-averaged, test-filter
  ratio 3 (see [the SGS theory page](../theory/sgs.md)); `report` prints
  the mean C_s in the canopy and in the free stream (the levels between
  the highest roof + 1 and the sponge). `:mlsgs` wraps `ML.MLSGS`; with
  the default `fn` (the Smagorinsky closure with `cfg.C_s`) the run is
  bit for bit the default run (tested over 100 smoke steps). There is no
  trained model.
* **Scalar advection.** Every tracer registered with `advect = true`;
  `positive` tracers go through the flux limiter (`advect_positive!`).
  `:weno_z7` needs `CaseConfig(halo = (4, 4, 4))`, `:teno8` halo 5; the
  build errors otherwise. The coupling refreshes the velocity halos
  first (see [the stale-halo defect](#Known-defects-kept-by-the-refactor)).

Measured in the `:smoke` window over 1 h (1800 steps, 12:00–13:00, the
free stream is levels 7–12, 37.5–75 m, above the 31 m roofs and below
the sponge), the dynamic mean C_s is 0.077 in the free stream and 0.076
in the canopy with ratio 3 (0.128 and 0.115 with ratio 2). That is below
the inertial-range 0.17: the window's flow is a 1 m s⁻¹ geostrophic wind
over 250 m with a capping inversion at 60 m, far from developed
isotropic turbulence, and the dynamic coefficient is lower in stratified
and transitional flow (it reaches 0.13–0.15 in the decaying-turbulence
test of `test/era2_sgs_numerics.jl`).

**Cost** on the flagship grid (64 × 64 × 32), per call, minimum of 15
repetitions on a loaded 24-core machine (each operator runs once per
step, the scalar scheme once per advected tracer — qv and ql):

| option | operator | ms/call |
| :-- | :-- | --: |
| pressure solve | cosine FFT (default) | 2.0 |
| | multigrid (20 V-cycles to 1e-10) | 72.6 |
| | MG-CG (11 iterations) | 45.8 |
| SGS ν_t | Smagorinsky (default; also MLSGS default) | 0.8 |
| | dynamic | 10.4 |
| momentum advection | centred (default) | 0.3 |
| | WENO-Z5 | 36.9 |
| scalar advection, one positive tracer | WENO-Z5 (default) | 7.4 |
| | TENO5 | 10.4 |
| | WENO-Z7 (halo 4) | 12.4 |
| velocity halo refresh (the #132 fix) | | 0.3 |

For scale, a whole default step costs about 40–60 ms under the same load
(`timing_report`, radiation excluded), of which `step!` is 12–17 ms. The
alternatives stay options: none is more accurate for this case on this
grid, and multigrid, MG-CG and the WENO-Z momentum advection each at
least triple the cost of `step!` (107, 64 and 48 ms per step) and add
roughly 50–150 % to a whole step.

## Dynamical core and time integration

Two exclusive slots select the dynamics (#133), both in
`couplings/core_select.jl` and `couplings/timestepper.jl`:

| slot | provider | src | what it is |
| :-- | :-- | :-- | :-- |
| `:core` | `:boussinesq` (default) | `Simulations.BoussinesqSimulation` | ∇·u = 0, buoyancy g(θ − θ₀)/θ₀ about the constant `cfg.θ₀` |
| `:core` | `:anelastic` | `Anelastic.AnelasticSimulation` | ∇·(ρ̄u) = 0, buoyancy g(θ − θ̄)/θ̄, ρ̄-weighted advection and projection (Lipps & Hemler 1982) |
| `:core` | `:pseudo_incompressible` | `Anelastic.PseudoIncompressibleSimulation` | ∇·(ρ̄θ̄u) = 0 (Durran 1989) |
| `:timestepper` | `:fixed` | `step!` | one forward-Euler projection step of `cfg.dt`; the stepper of the sound-proof cores |
| `:timestepper` | `:adaptive` (default) | `TimeIntegration.adaptive_dt`, `step_adaptive!` | CFL / diffusion / buoyancy limited step, growth ≤ 10 %, rollback and halving on failure; option `imex = true` steps with ARK2 and only the explicit limits bind |
| `:timestepper` | `:imex` | `TimeIntegration.step_imex!(…, ARK2())` | implicit diffusion, explicit advection and hooks, a projection per stage, at `cfg.dt` |

```julia
cfg = CaseConfig(:alexanderplatz; nsteps = 1800,          # a sound-proof core brings its solver
                 physics = Dict{Symbol,Any}(:core => :anelastic,
                                            :pressure_solver => :density_weighted_cg,
                                            :timestepper => :fixed))
cfg = CaseConfig(:alexanderplatz; nsteps = 1800,          # 1 h of simulated time
                 options = Dict{Symbol,Any}(:timestep_adaptive => (C_advective = 0.5, dt_max = 10.0)))
```

**The sound-proof cores** take the case's reference state as their
base state: `case.ref.state`, built by `build_reference` (θ̄(z) =
`initial_θ(cfg)` integrated hydrostatically from `cfg.p_surface` with
the MoistThermodynamics constants, see [Reference
state](#Reference-state-and-moist-buoyancy)), so there is one ρ̄(z) per
case — the one the couplings convert fluxes with and the ledger
integrates water with — the initial state has no buoyancy, and the flow
starts as in the Boussinesq core. The dynamics hooks (IBM, canopy drag,
geostrophic forcing, moist buoyancy, surface drag) touch only
`sim.u/v/w/tu/tv/θ` and run unchanged. The projection is the core's own
Jacobi-preconditioned CG solve of ∇·(W∇φ) = ∇·(W u*) (the
`:pressure_solver` provider `:density_weighted_cg`, options `tol =
1e-10`, `maxit = 500`), which is what makes them expensive: see the
numbers below.

**Tracers under a sound-proof core.** ∇·(W u) = 0 does not make ∇·u
vanish (∇·u = −w ∂ ln W/∂z), so `scalar_advection` advects with the mass
flux: ∂(W q)/∂t = −∇·(W u q), the scheme applied to (W/W_min) u and the
increment scaled by W_min/W_k (`Anelastic.constraint_weights` gives the
core's W). Σ W q is conserved exactly, the positivity limiter stays
valid (the scaling is ≤ 1) and a uniform tracer stays uniform: in the
smoke window, 24 s, max|q − 1| = 1.1e-11, against 5.8e-4 with the plain
velocity (option `weighted = false`). On the anelastic core W = ρ̄ is
the ledger's density, so the transport books no
`density_redistribution` (exactly 0.0); on the pseudo-incompressible
core W = ρ̄θ̄ and the θ̄ part of it is booked there (−1.4e-10 kg m⁻² in
the smoke window).

**The adaptive step** is chosen at the *start* of a case step
(`next_dt`), so every stage — surface, transport, microphysics — uses
the dt the dynamics covers. If `step_adaptive!` has to roll back and
halve, the dynamics covers the rest of that dt in further pieces, so
the case clock and the dynamics clock never disagree. `dt_max`
(10 s) is capped at the shortest `cfg.cadence` period. The ledgers book
with `ledger.dt`, the current step's length, not `cfg.dt`.

**Numerics keywords under every stepper.** The `:fixed` stepper is
`dynamics_step!` (`step!` with the hooks and `case.step_kwargs`). The
`:adaptive` stepper hands `step_adaptive!` a stepper that calls `step!`
(or `step_imex!` with `imex = true`) with `case.step_kwargs`, so every
retried piece carries them; with an `eddy_viscosity!` closure (`sgs =
:dynamic`, `:mlsgs`) its diffusive limit uses that closure's ν_t of the
current flow (`sgs_cfl_diagnostics`), also with `C_s = 0`, since
`step!` applies the closure whatever `C_s` is. `:imex` passes them to
`step_imex!`, whose `momentum_advection` is the explicit operator and
whose `eddy_viscosity!` supplies the frozen ν_eff. Tested with an MLSGS
closure that is Smagorinsky plus a call counter: called once per step
(fixed, IMEX) or twice (adaptive: the dt limit and the step), and the
flow is bit for bit the plain run of each stepper.

**Incompatible pairs** (declared, so selecting one is an error that
says why, and the smoke matrix steers around them):

| provider | incompatible with | why |
| :-- | :-- | :-- |
| `timestepper = :adaptive`, `:imex` | `core = :anelastic`, `:pseudo_incompressible` | `adaptive_dt`, `step_adaptive!`, `step_imex!` are typed on `BoussinesqSimulation` |
| `core = :anelastic`, `:pseudo_incompressible` | `pressure_solver = :cosine_fft`, `:multigrid`, `:mg_cg` | those solve the constant-coefficient ∇²φ = ∇·u*, not the weighted projection; select `:density_weighted_cg` |
| | `momentum_advection = :weno_z` | an unweighted operator; the core advects momentum in its W-weighted centred flux form (`:centered` passes no keyword to it) |
| | `sgs = :dynamic`, `:mlsgs` | they act through `eddy_viscosity!`, which the sound-proof step does not take; the core's closure is the built-in Smagorinsky |
| `pressure_solver = :density_weighted_cg` | `core = :boussinesq` | it solves only the weighted projection of the sound-proof cores |

Behind the declarations, `build_core` of a sound-proof core rejects any
`sim_options` keyword other than `tol` and `maxit`.

**Measured** (flagship 64 × 64 × 32, 1000 m × 1000 m × 200 m,
20 June 06:00–07:00, 1 h, first-call compilation included; #133,
measured on the branch before it was merged with #129–#132 and #135,
which changed the surface exchange, the radiation and the numerics; the
table was not re-measured after the merge):

| configuration | steps | wall (s) | dynamics (ms/step) | max\|Δ⟨θ⟩(z)\| vs Boussinesq/fixed |
| :-- | --: | --: | --: | --: |
| Boussinesq, fixed 2 s | 1800 | 165 | 19.6 | — |
| anelastic, fixed 2 s | 1800 | 367 | 139.6 (CG ≈ 333 it.) | 0.0020 K |
| pseudo-incompressible, fixed 2 s | 1800 | 364 | 136.9 | 0.0019 K |
| Boussinesq, adaptive | 1480 | 151 | 25.1 | 0.0018 K |
| Boussinesq, adaptive + ARK2 | 1460 | 349 | 163.9 | 0.0010 K |
| Boussinesq, IMEX ARK2 at 2 s | 1800 | 334 | 124.7 | 0.0020 K |
| *12:00–13:00:* Boussinesq, fixed 2 s | 1800 | 171 | 22.8 | — |
| anelastic, fixed 2 s | 1800 | 371 | 142.8 (CG ≈ 331 it.) | 0.0036 K |
| Boussinesq, adaptive | 1694 | 198 | — | 0.0004 K |

Wall times are from runs sharing the machine with five others, so they
compare only roughly; the step counts and profiles are exact.

* The adaptive step saves 18 % of the steps in the morning hour and 6 %
  at noon, with no retry in either; the step is 1.6–3.5 s (mean 2.43 s
  in the morning, 2.13 s at noon, where the flow is stronger), bound by
  advection (C = 0.5) in 98 % / 99.9 % of the steps and by the explicit
  diffusion (ν = 2 m² s⁻¹ on dz = 6.25 m: 3.70 s) in the rest. The CFL
  diagnostics cost about as much as the steps saved, so the wall time
  is roughly unchanged.
* IMEX buys no longer step here: at dx = 15.6 m, dz = 6.25 m the
  diffusive limit is only 4 % above the advective one, so ARK2 is
  bound by advection at nearly the same dt and costs twice as much per
  step. It is the option for configurations where diffusion binds
  (finer dz, larger ν); the smoke test shows it stable at dt = 12 s,
  where the explicit step amplifies the 2Δ mode of θ.
* In a 200 m deep domain the density changes by 2 % (Hρ ≈ 10.7 km), and
  the sound-proof cores give the Boussinesq mean profile to 0.002 K
  (morning) and 0.004 K (noon) after 1 h, at 2.2× the cost. They are there for deeper domains, not
  for the flagship's physics.

**The adaptive stepper is the default.** After the merge with #129–#132
and #135 the fixed 2 s step is no longer safe. Measured on master
(a630153, flagship grid, 1 h from 12:00, default couplings, the two
steppers otherwise identical):

| run | stepper | steps | dt min / mean / max (s) | wall (s) | outcome |
| :-- | :-- | --: | :-- | --: | :-- |
| summer, 20 June | fixed 2 s | 1800 | 2 / 2 / 2 | 259 | completed |
| summer, 20 June | adaptive | 2157 | 0.76 / 1.67 / 3.23 | 312 | completed |
| winter, 15 Jan | fixed 2 s | 141 | 2 / 2 / 2 | — | **blows up**: θ ∈ [−95, 587] K at step 141, `fog_dew` stops on a negative temperature |
| winter, 15 Jan | adaptive | 3378 | 0.70 / 1.07 / 2.69 | 426 | completed, θ ∈ [273.0, 277.3] K |

In summer the two agree at 13:00 to under 1 % (domain H 80.0 / 79.5,
LE 10.3 / 10.3 W m⁻², Ts 300.08 / 300.09 K, the same interval-mean Rn
466.17 W m⁻²) and both close the energy (≤ 1e-10 W m⁻²) and water
(≤ 2e-14 kg m⁻²) ledgers; the adaptive step costs 20 % more wall time,
because at noon the flow now needs a step below 2 s (0.76 s at 13:00).
In winter the fixed step is unstable within 5 minutes; this is the
winter-flagship blow-up recorded under #130, and it is a time-step
problem, not one of the physics. The fixed stepper stays for the
sound-proof cores, which require it (the adaptive and IMEX steppers are
typed on BoussinesqSimulation).

**Known limitations.**
* The sound-proof cores take only the built-in Smagorinsky closure and
  their own centred momentum advection (see the table above): their
  step has no operator keywords.
* `buildings_citymean` uses the nominal radiation period under an
  adaptive `dt` (see [Cadences](#The-coupling-contract)).
* `moist_buoyancy` adds `g (θ_ρ − θ)′/θ₀` with the constant `θ₀` on every
  core; on a sound-proof core the dry buoyancy is about θ̄(z), a
  difference of θ̄/θ₀ − 1 (< 1 % over the flagship depth).
* Resolved since the PR was opened: the plain-velocity tracer transport
  under a sound-proof core (now W-weighted, above), the stale-halo leak
  that the adaptive stepper's CFL diagnostics used to hide (#132 refreshes
  the halos in the transport itself), and the ledger density (ρ̄ since
  #131).

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
| `l.dt` | the length of the current step (s); use it, not `cfg.dt`, to turn the step's energy into a flux — they differ under the adaptive stepper |

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
scheme, and measured at ~1e-19 kg m⁻² at ρ = 1 since #132). The ledger
integrates the water with `ledger.ρ`, one density
per level, the reference density `case.ref.ρ̄` (#131). A coupling that
puts a flux into the air with a different density than the ledger's
shows up as a water residual, so a coupling converts every flux with
`ref_density(case, k)`.

The incompressible core does not conserve that ρ̄-weighted water: ∇·u = 0
conserves Σ q, so vapour carried up into thinner air loses ρ̄-weighted
mass. The transport stage therefore books its change of water in two
parts (`ledger_water_split`): `transport`, the column-mean ρ̄ times its
change of Σ q (zero for a conservative scheme; before #132 the halo
leak), and `density_redistribution`, the rest. Under a sound-proof core
(#133) the transport is W-weighted and conserves Σ W q, and the split is
by that weight: `transport` is c Σ W dq (c = ρ̄_m/W_m), zero for a
conservative scheme, and `density_redistribution` Σ(ρ̄_k − c W_k) dq,
exactly 0 on the anelastic core (W = ρ̄). Over the 1 h flagship run
(06:00–07:00, before #132's halo fix) with moist buoyancy on,
`density_redistribution` is
−8.1e-4 kg m⁻² (72 % of the slab's evaporation, 1.13e-3) and
`transport` −7.0e-4; with it off, −6.3e-4 and +1.5e-5. Without the
split both would be booked as `transport` (−1.5e-3, against −2.7e-4 at
ρ = 1 in the #127 baseline), and fixing the #132 halo leak could not
bring it to zero. Kessler
sedimentation has the same effect inside its `precipitation` term
(rain falling into denser levels), which #143 owns.

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
After the stale-halo fix of #132 the same hour gives ΔW = −5.72e-4 =
evaporation +7.57e-4, building vapour +9.12e-4, dew −2.24e-3,
transport **+6.9e-19**; residual −1.3e-14 (the surface-energy table
changes by at most 0.1 W m⁻² in any entry).

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
* **The transport was not conservative**: over the hour it removed
  2.7e-4 kg m⁻² of water, 35 % of what the surfaces evaporate. This was
  the stale-halo leak below, fixed by #132.
* The domain residual of 16.9 W m⁻² is the window sun (−16.0), the
  glass transmission (−2.4) and the dew's latent heat (+1.5): energy
  that goes into the buildings or comes from condensation, which the
  four-term balance does not see.

In the `:smoke` configuration of the smoke test (seeded fog, 12 steps
at noon, 16 × 16 × 16) the domain residual was −34.4 W m⁻² in summer
(dew heat +44.4 from the saturated seeded layer, window sun −6.9, glass
−3.0) and +6.1 W m⁻² in winter; with the transport conserving water
(#132) the summer value is −33.9 W m⁻².

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

## Surface exchange (#129)

`couplings/surface_exchange.jl` (slot `:surface_exchange`, provider
`:most`) and `couplings/ground_soil.jl` (slot `:ground`, provider
`:soil_column`) replaced the slab scheme of #126 (`surface_slab.jl`,
`soil_urban.jl`).

* **One surface per column.** Every column with air above it has an
  `AbstractSurface`: a `SoilColumn.SoilThermalColumn` with hydrology for
  streets, building-adjacent cells and the park (from `ground_soil`'s
  factory), a `Radiation.SlabLand` for roofs, glass and water (β = 1)
  until #139 / #144 / #141 register their own. Albedo and emissivity come
  from `bus.params` (the radiosity materials), so the radiation and the
  surface balance agree.
* **Roof-aware air level.** The forcing is the first non-solid cell of
  the column (`bus.k_air`), `z_ref = Δz/2` above the surface; the air
  temperature is `θ Π̄_k` and the pressure `p̄_k` of the reference state
  (#131; roofs sit up to 190 m up). H goes into θ, E into qv of that
  cell only: the smoke check counts writes into solid cells (0 in every
  run), and `test/berlin_surface_exchange.jl` checks that the stage
  changes nothing else, and that ``Σρc_pΔzΔθ = ΣHΔt`` (2e-11).
* **Cadence.** The surfaces are stepped every `cfg.cadence[:soil]`
  seconds (30 s in the flagship, 4 s in `:smoke`; a `:surface` key
  overrides it) and their fluxes are held as tendencies until the next
  surface step, so the air receives exactly what the surface gave.
* **Momentum.** The surface stress ``ρu_*^2`` is applied as an implicit
  drag on the first-air-cell velocity faces, after the IBM. The momentum
  removed equals ``Σ u_*^2\,\mathbf{u}/U\,Δt`` to ``O(aΔt)`` (1e-4 in
  the test). The `:wall` ground used to be free-slip.
* **Extra surface energy.** `bus.sensible` (window sun, glazing
  transmission, dew heat) is collected every step and delivered to the
  surfaces as `Q_extra` at their next step (`step_surface!` of `SlabLand`
  and `SoilThermalColumn` take it since #129). The ledger books the
  pending energy in ``ΔS``, so the per-class closure stays at round-off
  (≤ 1.5e-10 W m⁻² in 1 h of the flagship).
* **Dew.** One vapour exchange per surface: the soil and water surfaces
  take up dew as negative evaporation (into the soil water); the
  RadiationFog dew reservoir of `fog_dew` acts only over surfaces with
  β = 0 (roofs, glass), above the roof.
* **No clamps.** H, LE and Ts are not clipped (the Kessler qv clamp
  stays for #143).

### Measured (flagship, 64 × 64 × 32, dt = 2 s, 1 h)

These runs predate the merge with #131. They used `T_air = θ − Γ_d z`
(about 1.1 K below `θ Π̄_k` at the ground), an 8 km scale-height
pressure and ρ = 1 for the air-side increments; the numbers have not
been remeasured with the reference state.

Two 1 h runs, from 06:00 and from 12:00 local on 20 June, each against
the same run on the #127 branch (slab scheme). Domain means over
4096 columns, W m⁻², averaged over the interval:

| interval | slab H | MOST H | MOST LE | MOST H street+wall | MOST H roof | u* street | u* roof |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| 06:00–06:05 | −24.1 | −0.5 | 0.6 | −0.6 | −0.1 | 0.010 | 0.002 |
| 06:45–07:00 | −15.9 | 22.7 | 46.2 | 34.5 | 0.0 | 0.054 | 0.001 |
| 12:00–12:05 | −23.7 | 6.7 | 22.6 | 10.4 | −0.1 | 0.045 | 0.002 |
| 12:45–13:00 | −8.7 | 86.5 | 15.2 | 127.9 | 10.5 | 0.085 | 0.040 |

Why H differs: the slab scheme exchanged at a fixed `C_h = 1e-3` and
`U = 3 m/s`, i.e. 3.0 W m⁻² K⁻¹ whatever the flow, from a slab that
starts 8 K colder than the air (Ts0 = 285 K under θ = 293 K), so its H
was about −24 W m⁻² at the start and still negative after an hour at
noon. With Monin–Obukhov and the resolved wind at the first air cell
(0.55 m/s over streets, 0.21 m/s over roofs at 07:00) the cold,
stably stratified start is almost decoupled (H ≈ 0), and the soil
columns' massless skins heat within minutes, so the streets turn
unstable and give 30–130 W m⁻² after half an hour. The cold-started roof
slabs (C_s = 1.8e5 J m⁻² K⁻¹) stay colder than the air for most of the
hour, stable and decoupled.

**u\* over roofs.** The plan expected 0.2–0.6 m/s at midday. Measured:
0.045 m/s mean over roofs at 13:00 (0.0022 at 07:00); streets 0.116.
Two reasons, both in the case rather than the scheme: the wind at the
first air cell above the roofs is only 0.28 m/s (the case is driven by a
1 m/s geostrophic wind, u0 = 1 m/s), which gives a *neutral* u\* of
0.027 m/s for z0m = 0.05 m; and the roofs start at 285 K, so they are
stably stratified for most of the first hour. At 13:00, with the roofs
2.4 K warmer than the air, u\* is 1.7 × its neutral value (the
convective enhancement). The expectation assumed a few m/s of wind.

**Street β.** The street soil starts at θ = 0.10 (β = 0.44). In the
noon hour its mean β falls to 0.37 (0.35–0.44) as the top layer dries,
and the evaporation the soil can deliver collapses: the domain-mean LE
is 71 W m⁻² between 12:15 and 12:30 and 16 W m⁻² after 12:30.

**Bug found (src).** `SoilThermalColumn.step_surface!` returned the
evaporation the skin balance charged even when the soil, at its dry
potential, delivered less; the difference was only recorded as
`budget.latent_mismatch`. Coupled to the air this created water: in the
first noon run a third of the street LE (≈ 70 of 184 W m⁻²) was never
lost by the soil. It now returns the delivered E and LE and keeps the
unspent latent heat in the top soil layer (test in
`era5_soil_thermal.jl`). LandSurface's vegetated column has the same
pattern (`latent_mismatch` there); #140 should check it.

**Cost.** The surface step (4094 surfaces: 3179 soil columns with
hydrology, 915 slabs) is 10–17 ms per dynamics step averaged at the
30 s cadence (18–31 s per simulated hour, the range is the machine's
load), the air-side update 0.9 ms and the drag 0.15 ms. The soil column
with hydrology costs about 56 µs per call, a slab 16 µs.

**Known limitations.** Streets are bare loam: SoilColumn has no sealed
surface, so a street evaporates until its top layer dries (a sealed
option belongs in src). No rain reaches the soil (the microphysics books
precipitation as a domain total). The roughness lengths are per class
options (street/roof 0.05 m, park 0.1 m, water 1e-4 m, glass 0.01 m,
z0h = z0m/10), not a survey. The more vigorous noon convection makes the
stale-halo transport leak of #132 larger: the transport created
3.4e-3 kg m⁻² of water in the noon hour and drove qv negative aloft,
which the Kessler clamp then filled (+8.9e-5 kg m⁻²).

## Radiation (#130)

Two couplings put the Era III radiation stack into the case. Both run at
the `:radiation` cadence, around the radiosity solve.

**`column_radiation`** (slot `:column_radiation`, provider `:twostream`,
the only one; it replaced `cooling_prescribed`). At order 5, before
the radiosity:

1. the mean over the fluid cells of θ, qv and ql on every LES level
   becomes a hydrostatic column (`column_from_profiles`, surface
   pressure `surface_pressure(case)` of the reference state, #131), stitched with `stitch_columns` under
   `us_standard_column` (60 layers to 30 km, its surface temperature
   set so that it continues the LES top temperature at 6.5 K/km,
   mid-latitude-summer or sub-arctic-winter humidity by scenario);
2. the surface of the column is the whole domain: the radiative
   temperature ``(\sum ε T_s^4/\sum ε)^{1/4}``, the mean emissivity and the
   effective albedo of the last radiosity solve;
3. correlated-k gas optics plus Mie cloud optics, `solve_lw!` and
   `solve_sw!` (scaled by the Earth–Sun distance factor);
4. `bus.L_sky` is the downwelling longwave at the surface of that
   column (the fog emits into it); `bus.dni`/`bus.dhi` are the surface
   direct and diffuse shortwave of the **same column with the LES
   condensate removed**, because the 3-D cloud shadow below applies the
   resolved cloud to the direct beam, and doing both would count it
   twice; option `surface_sw = :bird` leaves the shortwave to the
   radiosity coupling's Bird clear sky;
5. the heating rate of the LES layers (LW + SW, with the condensate) is
   applied to θ on every step at `:column_tendency`, uniform over the
   fluid cells of each level, through `Radiation.apply_radiative_cooling!`
   with the solid mask (the buildings' cells are not air; heating them
   leaks back into the air next to the walls — in review, a 30 min night
   run of the `:smoke` window cooled the fluid next to buildings at
   level 1 by −0.314 K with the solid cells heated, −0.236 K without).

**`cloud_shadow`** (additive). At order 15, after the radiosity: the
per-cell cloud optical depth with the `RadiationCore` mass extinction
(the same as the column's Mie cloud optics; zero in solid cells),
delta-scaled with the visible ``ω``, ``g`` of that cloud optics
(``τ' = 0.26\,τ``), the slant-path transmittance
`cloud_shadow_transmittance!` from the top of each column's solid stack,
and `bus.sw ← bus.sw·(1 − f_dir(1 − T))` with ``f_{dir}`` the share of the
column's absorbed radiosity shortwave that arrived as direct beam. It
replaced `apply_cloud_shadow!` (vertical ql integral with an ad-hoc
700 m² kg⁻¹, 15 % of the removed light kept, no sun angle, multiplied
into `bus.sw` every step, so compounding). Limitations: the
interreflections of the shaded beam and the buildings' window sun (read
from the radiosity solution) are not attenuated, and the light the cloud
scatters sideways is not added to the diffuse.

Measured before the merge with #131, with the column's surface pressure
from `cfg.elevation` (100.92 kPa against the reference state's 101.325)
(1 h runs of the flagship, 64 × 64 × 32, dt = 2 s, radiation
every 600 s; values at the end of the hour; winter on the `:smoke`
window, because the winter flagship blows up, see below):

| | 00–01 h | 06–07 h | 12–13 h | winter 12–12:30 (smoke window) |
| :-- | --: | --: | --: | --: |
| `L_sky` (W m⁻²) | 331.9 | 333.5 | 335.4 | 237.9 |
| old `0.95σT̄s⁴` at the same T̄s | 356.2 | 372.4 | 389.0 | 297.9 |
| GHI, column (W m⁻²) | 0 | 559.5 | 831.4 | 234.3 |
| Bird (the radiosity's sky: PW 2 cm, O₃ 0.33), aerosol-free / AOD₅₀₀ = 0.16 | 0 / 0 | 574.4 / 542.3 | 834.2 / 797.1 | 235.1 / 211.7 |
| LES-layer heating, LW / SW (K day⁻¹) | −12.9 / 0 | −9.8 / +3.5 | −6.7 / +4.6 | −4.0 / +1.4 |
| the same above 50 m, LW / total | −5.3 / −5.3 | −4.5 / −1.0 | −3.8 / +0.8 | −2.5 / −1.1 |
| lowest level (0–6 m), LW | −156 | −108 | −59 | −16 |
| precipitable water (kg m⁻²) | 12.3 | 12.3 | 12.2 | 3.5 |

* **Sky longwave** is 25–54 W m⁻² below the old surface-based value in
  summer. It is almost constant over the day (it depends on the air,
  not on the surfaces); in winter it is 238 W m⁻², below the
  300–380 W m⁻² the issue expected — that range was a summer
  expectation, and a clear sky over 3.5 kg m⁻² of water at 274 K gives
  about 0.7 σT⁴.
* **Clear-sky GHI.** Against Bird with no aerosol and the column's own
  precipitable water and ozone (440 DU in summer, 365 in winter), the column is −0.4 % at
  μ₀ = 0.86, −3.1 % at μ₀ = 0.58 and −4.6 % at the winter noon sun,
  μ₀ = 0.27 (first step of the `:smoke` window): TwoStream falls below
  Bird as the sun gets low. The smoke check asserts 5 % for μ₀ ≥ 0.5 (the
  issue's noon criterion) and 10 % below. Against Bird with the case's
  urban AOD the column is 4.3 % higher at noon and 10.7 % at the low
  winter sun: the TwoStream column has no aerosol until the
  aerosol-optics coupling (#142).
* **Heating.** The lowest LES levels cool at tens to 156 K day⁻¹: the
  case starts its surfaces at 285 K under 293 K air, and the air near
  the ground radiates into the colder surface. Above 50 m the longwave
  cooling is −3.8 to −5.3 K day⁻¹ in summer, beyond the −1…−3 K day⁻¹
  the issue expected. Both the cold surface and the humid initial
  profile (qv = 10 g kg⁻¹ at the ground) cause it: in the initial state,
  −5.0 K day⁻¹ above 50 m; −3.1 with the surfaces at 293 K; −1.4 with
  that and half the humidity.
* **Energy.** ``\int ρ c_p\,\dot T\,dz`` over the LES layers equals the
  net-flux convergence to round-off (a smoke check, 1e-10), and the θ
  increment is that heating times dt in every fluid cell and zero in
  every solid one (a smoke check on the field). The air therefore takes
  up the column's convergence times the fluid fraction of each level.
* **Ledger.** In the 06–07 h hour the domain Rn falls from 244.28 W m⁻²
  (#127 baseline, same configuration) to 228.59 W m⁻²: the lower sky
  longwave outweighs the stronger TwoStream shortwave (GHI 559.5 against
  Bird's 542.3).
* **Cloud shadow** (`:smoke`, seeded fog of 0.2 g kg⁻¹ in the lowest
  18.75 m, noon): on step 1 the direct-beam transmittance is 0.80–0.85
  (mean 0.849; ``e^{-0.2}``, the delta-scaled 0.17 over μ₀ = 0.86), and
  the shadow takes 117.6 W m⁻² (domain mean) off the surface shortwave;
  the old formula would have kept 21 % of the shortwave under the same
  fog, and then again on every step. Roofs above the fog are not
  shaded.
* **Cost** per radiation call, measured after warm-up on the flagship:
  column radiation 38 ms by day, 23 ms at night; cloud shadow 13 ms;
  (the radiosity solve itself: 1–3 s). At the 600 s cadence that is
  0.2 ms/step, plus 0.3–0.4 ms/step for applying the heating. The
  radiation cadence (`cfg.cadence[:radiation]`) is shared with the
  radiosity: the sky longwave and the beam must be the ones the
  radiosity solve uses.

**The winter flagship blows up** (pre-existing, not a radiation
problem): `CaseConfig(:winter)` from 12:00 on the 64 × 64 × 32 grid
reaches θ ∈ [10, 353] K by step 118 and −700 K in the dynamics of step
119, after which `fog_dew` stops on a negative temperature. The #127
branch, without these couplings, does exactly the same (|u| grows from
1.9 to 12 m s⁻¹ between steps 100 and 118). It was the fixed 2 s step:
with the adaptive stepper, the default since then, the winter flagship
runs (see "The adaptive stepper is the default" above).

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
coupling switches that additive coupling off; one incompatible with the
default provider of another slot switches that slot to its first
compatible provider (so `core = anelastic` runs with `pressure_solver =
density_weighted_cg`, and that solver's own run with the anelastic
core). Every run must finish
without a blow-up, leave every field and tracer finite, pass every
`smoke_check` of its couplings, and close the surface-energy ledger per
class (1e-6 W m⁻²) and the water ledger (1e-12 kg m⁻²). The first run
also asserts that the domain gains radiative energy at noon (Rn > 0,
ΔS > 0) and that the transport conserves water (|transport| <
1e-12 kg m⁻²; measured ~1e-19 since #132). Finally every registered
coupling must have run in at least one configuration, and the whole
test must take less than 300 s.

At #127 the matrix was 2 runs (no slot had a second provider yet); #130
added the column radiation's `surface_sw = :bird` variant, and #132 the
non-default providers of the numerics slots (`momentum_advection =
weno_z`, `pressure_solver = multigrid` and `mg_cg`, `sgs = dynamic` and
`mlsgs`) and the scalar-advection variants (TENO5; WENO-Z7 with halo 4).
Run on its own the test took about 66 s at #132, almost all of it
first-call compilation (each run takes 0.3–1 s once compiled, the
multigrid runs a few seconds). The `:core` and `:timestepper` slots of #133
add six (anelastic, pseudo-incompressible, `pressure_solver =
density_weighted_cg`, adaptive, IMEX, adaptive + ARK2): 18 runs.

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
scaffolds), 30 pending; #130 coupled RadiationCore, GasOptics and
TwoStream (27 pending).

**The flagship guardrails.** The Berlin case is the flagship: it runs
every component, on by default, and new physics lands in it in the PR
that adds the physics. `test/berlin_coverage.jl` enforces that with four
reviewed lists, each frozen against a literal snapshot in the same file
(`*_FROZEN`; the test requires list = snapshot, so an entry can only be
deleted, from both; adding one is a snapshot edit a reviewer sees and
only the maintainer agrees to):

| list | holds | at #160 |
| :-- | :-- | :-- |
| `BERLIN_EXEMPT` | non-physics modules; each reason starts `infrastructure:` or `scaffold:` | 11 |
| `BERLIN_PENDING` | modules not wired in yet, with the owning issue | 24 |
| `BERLIN_DEFAULT_OFF` | addon couplings off by default, with the measured cost or conflict | 0 |
| `BERLIN_OPTION_ONLY` | modules only a non-default slot provider runs, with why it is an alternative | 2 (Multigrid: `pressure_solver = :multigrid / :mg_cg`; ML: `sgs = :mlsgs`) |

and with these testsets:

* **frozen lists … may only shrink** — each list equals its snapshot;
  exemption reasons start with `infrastructure:` / `scaffold:`.
* **every addon on by default, every slot's default provider in the
  default run** — every `slot = :addon` coupling has `is_default = true`
  unless it is in `BERLIN_DEFAULT_OFF`; every exclusive slot has exactly
  one default provider and it is not an `:off` / `:none` / `:disabled`
  switch (`chem_off` is a non-default provider of `:chemistry`); and
  `selected_couplings` of `CaseConfig()` and of every `SCENARIOS` preset
  contains every such addon and default provider.
* **a module only a non-default provider runs is a reviewed exception** —
  a module some coupling lists counts as coupled (the smoke test runs
  every provider), but if the default `CaseConfig()` run does not use
  it, it must be in `BERLIN_OPTION_ONLY`; an entry the default run now
  uses is stale and fails too.

`test/berlin_smoke.jl` adds one more: the "summer / winter, everything
on" configurations of `smoke_matrix()` must select every addon and every
default provider, so an `incompatible` declaration cannot silently drop
a coupling from the runs that are meant to have everything.

Each guardrail was checked against a deliberately broken tree (not
committed): a new `src/DummyPhysics.jl` added to `BERLIN_PENDING` or
`BERLIN_EXEMPT` fails "BERLIN_PENDING has grown: DummyPhysics" (resp.
EXEMPT, plus the `infrastructure:` / `scaffold:` check);
`is_default = false` on `veg_drag` fails "Addon couplings off by
default: veg_drag" and "CaseConfig() does not run: veg_drag"; `veg_drag`
moved into the `:sgs` slot as a non-default provider fails "Modules the
default Berlin run does not use (only a non-default provider does):
PlantCanopy (by veg_drag)"; making `chem_off` the `:chemistry` default
fails "slot :chemistry defaults to an off switch (chem_off)"; and
declaring `cloud_shadow` incompatible with `fog_dew` fails 'smoke run
"summer, everything on" does not run: cloud_shadow, fog_dew'.

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

## Air quality: chemistry and deposition (#135)

The `:chemistry` slot's default provider `:nox_o3_voc` (`aq_chem.jl`)
runs the `Aeolus.Chemistry` mechanism ([theory](../theory/urban_chemistry.md))
on five tracers registered with `advect = true`: NO, NO2, O3, VOC and
HNO3, in ppb. The `:transport` stage moves them. The example's toy
mechanism, its deposition table and its dead `apply_wet_deposition!`
are gone. What stays in the coupling is case data: the emission
inventory, the class → surface-kind map and the roughness per class.

Every step in `:chemistry`, in this order:

1. Emissions go into the first air cell `k_air` of each column (above the
   roof on building columns). Traffic NOx is emitted on street and
   building-adjacent columns: 2 µg m⁻² s⁻¹ as NO2 at the morning peak,
   10 % primary NO2, with an assumed diurnal shape (`traffic_profile`).
   Isoprene is emitted on park columns: 0.3 µg m⁻² s⁻¹ in full sun,
   scaled by J(NO2). Both are order-of-magnitude estimates, not a
   Berlin inventory.
2. `apply_photochemistry!` with the clear-sky `j_NO2(μ₀)` of the current
   sun, OH prescribed as 5 × 10⁶ cm⁻³ × J/J_max, T = θ̄ Π̄ (θ̄ the
   level mean over air cells) and the reference p̄ per level (`case.ref`,
   #131). The numbers below were measured before the switch from the 8 km
   scale-height pressure, which differed from p̄ by at most 0.17 %.
3. Dry deposition of NO, NO2, O3 and HNO3 from `k_air` with
   ``v_d = 1/(r_a + r_b + r_c)``.
4. Rain scavenging of HNO3 by `qr`.

In `:surface_update`, on the surface cadence, `r_a = 1/(C_H U)` and `u*`
are recomputed per column. They come from
`SurfaceExchange.solve_surface_layer` between the skin `bus.Ts` and the
first air cell (cell-centre wind, θ, qv at `z = Δz/2`), with
`z0h = z0m/10` and the per-class `z0m` that #129 gives its surfaces. Once
#129 (PR #158) is merged, the bus carries u* and the roughness per
column, and this coupling should read them rather than solve again.

`physics[:chemistry] = :off` (provider of `NoChem`) removes the species.
It is not the default. The smoke matrix also runs a `night titration`
variant (start 00:30, J = OH = 0).

### Measured (flagship grid 64 × 64 × 32, 20 June, 12:00–12:30, 900 steps)

| quantity | default (`Ts0` = 285 K) | warm surfaces (`Ts0` = 300 K) |
| :-- | --: | --: |
| Leighton ratio φ, median over air cells with NO, O3 > 0.01 ppb, at 12:30 | 1.0019 | 1.0014 |
| φ / φ_ext (peroxy-radical steady state), median | 0.9968 | 0.9965 |
| r_a street / roof / park, median (s m⁻¹) | 1.7e5 / 4.7e6 / 6.1e8 | 85 / 82 / 77 |
| u* street / park, median (m s⁻¹) | 0.003 / 0.000 | 0.087 / 0.065 |
| v_d O3 park / street (mm s⁻¹) | ≈ 0 | 3.17 / 0.46 |
| v_d NO2, HNO3 street (mm s⁻¹) | ≈ 0 | 1.52, 5.72 |
| mean dry deposition flux O3 / NO2 / HNO3 (nmol m⁻² s⁻¹) | 0.019 / 0.008 / 0.001 | 1.19 / 0.63 / 0.14 |
| mean emission NO / NO2 / isoprene (nmol m⁻² s⁻¹) | 17.9 / 2.0 / 0.50 | same |
| surface-layer NO / NO2 / O3 at 12:30 (ppb) | 13.5 / 16.3 / 31.6 | 8.7 / 11.5 / 33.6 |

The default flagship starts its slab surfaces 8 K below the air. At noon
the surface layer is then so stable that `solve_surface_layer` clamps ζ
at its limit, `r_a` reaches 1e5–1e9 s m⁻¹, and dry deposition is
effectively off for the first hour. This is the MOST answer for that
state, not a defect of the deposition. With #129's soil skins the
streets turn unstable within minutes. The periodic domain has no inflow
of background air, so emitted NOx accumulates: surface-layer NO rises
from 1.7 to 8.7 ppb in 30 min.

Wet deposition, measured in the smoke window with `qr = 1e-4` seeded in
levels 2–4: the HNO3 removed per step equals ``s[\mathrm{HNO_3}]\,(Σ q_r Δz)\,Δt``
to 0.1 %. The rain sediments out in about four steps.

**Cost.** On the flagship grid the step goes from 43.1 to 101.1 ms (300
steps from 12:00, loaded machine). The WENO-Z5 transport of the five
species adds 44 ms, 8.8 ms per tracer; the photochemistry adds 11.2 ms
(85 ns per cell); the r_a refresh adds 2–3 ms averaged over steps. The
chemistry stays on by default because the flagship is meant to exercise
all the physics, and `:off` exists for runs that do not need air
quality.

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
| slab: constant C_h and U, flux clamps, roof fluxes into solid k = 1, no surface drag | `surface_slab.jl`, `fog_dew.jl` | #129 (fixed: `surface_exchange.jl`, `ground_soil.jl`) |
| `qr` sedimented but not advected; qv clamp | `micro_kessler.jl` | #143 |
| glass is a synthetic roof mask | `glass.jl` | #144 |
| toy aerosol: not advected or output, zero with no source; its PM deposition table | `aq_aerosol.jl` (table in `aq_chem.jl`) | #147 |
| OSM `&&`/`||` precedence; patches overwrite roofs | `core/osm_ingest.jl`, `core/geometry.jl` | #141 |

**Found during #126, fixed in #132: stale velocity halos in the scalar
transport.**
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
transport term is −2.70e-4 kg m⁻². **Fixed in #132**: the
`scalar_advection` coupling refreshes the u, v, w halos before it
advects (`refresh_halos = true`), which also gives the pollutant, which
advects later in the step, fresh halos. The water change of the
transport stage is now ~1e-19 kg m⁻² per run (smoke) and +6.9e-19 kg m⁻²
over the flagship hour; both `@test_broken` are now `@test`, and
`test/berlin_case.jl` keeps the mechanism tested with
`refresh_halos = false`.
