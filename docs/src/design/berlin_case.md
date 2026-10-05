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

## Status of the epic (#125)

The work is tracked in the milestone "Berlin — full-physics flagship"
under the epic #125. On master `915bd49` (after PRs #161 and #166) 10 of
its 29 issues are closed and 19 are open:

* **Wave 0, done:** the refactor (#126) and the smoke / coverage /
  ledger tests (#127).
* **Wave 1, 7 of 10 merged:** shade / class map (#128), surface exchange
  and soil (#129), column radiation and cloud shadow (#130), reference
  state and moist buoyancy (#131), numerics slots (#132), time
  integration and core options (#133, PR #161; the adaptive stepper
  became the default in PR #166), chemistry and deposition (#135).
  Open: Era I add-ons (#134), the Monte-Carlo RT diagnostic (#136),
  visibility and fog feedback (#137).
* **Wave 2:** the GMRES facet solve (#138) is merged; the per-facet
  walls (#139), park land column (#140), water surfaces (#141), aerosol
  optics (#142) and P3 slot (#143) are open.
* **Waves 3–4 and the solver issues** (#144–#153) are open.

None of this is a roadmap phase: the Era VI and Era VII groundwork it
delivered (the dynamic Smagorinsky in `Operators`, the reduced
chemistry in `src/Chemistry.jl`) is recorded under those eras in the
long-range roadmap, without marking a phase shipped.

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
| `step_options(c, case)` | keywords added to every dynamics step, `step!` or `step_imex!` (e.g. `momentum_advection`, `sgs`) |
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
| dynamics: `advance_dynamics!` of the `:timestepper` provider (default one `step!`) with `:body_force` / `:post_predictor` | `geostrophic` (10), `moist_buoyancy` (20), `wall_model_facades` stencil-flux removal (40) / `ibm` (10), `wall_model_facades` facade stress (12), `surface_exchange` drag (15), `veg_drag` (20); the `step_options` of the numerics slots |
| `:surface_update` | `surface_exchange` (10: collects `bus.sensible`; every surface period steps all surfaces, soil included), `snow_legacy` (30), `aq_chem` r_a / v_d refresh (90, every 30 s) |
| `:column_tendency` | `column_radiation` (10) |
| `:sanity` | the core finiteness check |
| (ledger) | every coupling's `budget!`, and the change of water in the air (core) |
| `:output` (1800 s) | NetCDF record and log line (core), then coupling entries: `turbulence_diag` (10) |

The order was the order of the pre-refactor loop, floating-point order
included (#126). The surface exchange (#129) replaced the slab, whose
latent flux sat at the end of `:anthropogenic` and whose Ts clamp ran in
`:sanity` only to keep that order; see [Surface exchange](#Surface-exchange-(#129)).

`run_case!` prints the wall time of every entry (`timing_report`). At
#126, in the 1 h flagship run (1800 steps, first-call compilation
included), the largest entry was the WENO-Z5 scalar transport, about
17 ms/step, ahead of `step!` itself (about 12.5 ms/step), then `fog_dew`
(6.1), `radiosity_sun` (5.5 averaged over all steps) and `soil_urban`
(2.8); everything else was below 1 ms/step (table in the #126 PR).
Since then `soil_urban` has been replaced by the surface exchange
(#129; its surface step costs 13.6–16.3 ms per step of the adaptive
stepper, remeasured on master `915bd49`) and the chemistry (#135) adds five advected species; see the
cost paragraphs of those sections.

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
  A tracer diffuses with the closure's `κt` unless it names its own
  diffusivity (`ensure_tracer!(…; diffusivity = case -> field)`): the SGS
  TKE `e` of the `:tke` provider diffuses with `2ν_t` (#182). Passive
  tracers (qv, ql, the chemistry species, the pollutant) never name one:
  they share θ's κ_t under every `:sgs` provider, AMD included (#234, see
  below).
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
| `surface_exchange` | forcing `T_air = θ Π̄_k`, `p̄_k`, `ρ̄_k` at the first air cell `k_air`; `ρ̄_k c_pd`, `ρ̄_k` in the conversion of H and E to θ, qv |
| `aq_chem` | `T = θ̄ Π̄_k` and `p̄_k` for the rate constants; its r_a solve uses `θ_s = Ts/Π̄_k` at the first air cell |
| `column_radiation` | surface pressure `surface_pressure(case)` (101.325 kPa) |
| `cloud_shadow` | `ρ̄_k` for the cloud water path when no density is given |
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
| `:sgs` | `:tke` (`sgs_tke`, `PrognosticSGS.TKEClosure`, the Deardorff/Moeng prognostic SGS TKE, #182; Laplacian momentum form, see below), `:smagorinsky` (`sgs_smagorinsky`, `StaticSmagorinsky(cfg.C_s)`, the coefficients of the cores' built-in closure bit for bit; the default before #182), `:dynamic` (`sgs_dynamic`: `LagrangianDynamicSmagorinsky` by default, `DynamicSmagorinsky` for `averaging = :plane` / `:local`; with the buildings' mask), `:mlsgs` (`sgs_mlsgs`, `ML.MLSGS`), `:amd` (`sgs_amd`, `AMDClosure`, zero in the building cells), `:vreman` (`sgs_vreman`, `VremanClosure(cfg.C_s)`), `:sigma` (`sgs_sigma`, `SigmaModel`, #184), `:reynolds_stress` (`sgs_reynolds_stress`, `PrognosticSGS.ReynoldsStressClosure`, the differential Reynolds-stress closure, #188; not with `timestepper = :imex`); all but `:tke` and `:reynolds_stress` with option `stress = :symmetric` (`:laplacian`); `:smagorinsky`, `:dynamic` and `:mlsgs` are each wrapped in `StabilityCorrected` (P55-e, #187; `cfg.sgs_stability = :brown94`; `:none` gives the bare closure, for `:smagorinsky` the coefficients of the cores' built-in closure bit for bit) and also take `scalar_tensor = :diagonal` (`:isentropic`, with `diapycnal`, `slope_max`); `:tke` and `:amd` carry their own stability treatment (TKE length scale, AMD buoyancy term) and are not wrapped, nor are `:vreman`, `:sigma` and `:reynolds_stress` (its stresses carry the buoyancy production directly) | `step_options`: `sgs`, an `AbstractSGSClosure` (the Era VI contract, #167), and `sgs_stress` (the stress form, #175); every core and stepper takes both |
| `:momentum_advection` | `:centered` (`momentum_centered`), `:weno_z` (`momentum_weno_z`, option `order = 5`/7) | `step_options`: `momentum_advection` |
| `:scalar_advection` | `:advection_schemes` (`scalar_advection`, options `scheme = :weno_z5`, `sgs_diffusion = true`) | the `:transport` stage: advection, then SGS diffusion with `κt` (#177; with a stability-corrected `:sgs` closure κh on the x, y faces and its κv on the z faces, as for θ) |

`sim_options` and `step_options` are contract hooks: a coupling returns a
`NamedTuple` of keywords for the core's constructor or for every
dynamics step (`step!`, or `step_imex!` under the IMEX steppers, which
takes the same `momentum_advection` and `sgs`); two
couplings may not set the same keyword.

**SGS providers** are closure objects of the Era VI contract
([Closure contract](../theory/sgs.md#Closure-contract)): a provider
builds its `AbstractSGSClosure` once and `step_options` returns
`sgs_step_options(closure, stress) = (sgs = closure, sgs_stress = stress)`.
Every provider runs the **stress-form** momentum divergence
`∂ⱼ(2νᵢⱼSᵢⱼ)` (`sgs_stress = :symmetric`, P55-b #175; see
[Stress form vs Laplacian form](../theory/sgs.md#Stress-form-vs-Laplacian-form)),
not the cores' default Laplacian form `∂ⱼ(νₜ∂ⱼuᵢ)`; option
`stress = :laplacian` restores that (a smoke variant), and each provider's
`smoke_check` asserts that the stress-form dissipation Σ2νSᵢⱼSᵢⱼ is
non-negative. Under the IMEX steppers the diagonal part is implicit and
the cross terms explicit.

**Stability correction** (P55-e, #187; on by default for the
`:smagorinsky`, `:dynamic` and `:mlsgs` providers; the default provider
`:tke` has its own stability length and is not wrapped). The one builder
`sgs_provider_closure` wraps those providers' eddy-viscosity closures in
`StabilityCorrected` ([Stability correction](../theory/sgs.md#Stability-correction)):
ν_h = ν ℓ_h, ν_v = ν ℓ_v f_m(Ri_g), κ_h = ν ℓ_h/Pr_t,
κ_v = ν ℓ_v f_h(Ri_g)/Pr_t, with the TurbulenceStability functions of
`CaseConfig.sgs_stability` (`:brown94` default, `:lilly`, `:long_tail`,
`:short_tail`, `:most`; `:none` = the bare closure, the pre-P55-e run),
the anisotropic length scales Δ_h = √(dx dy), Δ_v = dz (ℓ_h = 1.842,
ℓ_v = 0.2947 at dx/dz = 2.5), Ri_g from the θ_v of the `qv` tracer and
the core's own reference θ (θ₀, or θ̄(z) on the sound-proof cores, so it
runs with `core = :anelastic`). `:dynamic` gets no f_m on the momentum
(only Pr_t(Ri)). Each provider's `smoke_check` asserts κ_v, ν_v ≥ 0;
κ_v ≤ κ_h and ν_v ≤ ν_h ℓ_v/ℓ_h where Ri ≥ 0; ν_v ≥ ν_h ℓ_v/ℓ_h where
Ri ≤ 0 (the convective enhancement of Brown et al.; not for `:dynamic`);
κ_v = 0 where Ri ≥ Ri_c; its `report` prints the supercritical fraction
and the mean κ_v/κ_h. Smoke variants: `sgs_stability = :none`, `:lilly`.
Its validation on GABLS1 and the stably stratified channel (P55-f,
#191) adds no Berlin coupling — no new physics, a stated exception — and
is in [Stable boundary layers](../theory/stable_boundary_layer.md).
The scalar (θ) flux is the contract's diagonal operator
`add_eddy_diffusion_scalar!(tϕ, ϕ, κh, κv)` (P55-c, #181) with these
κ_h ≠ κ_v. Option
`scalar_tensor = :isentropic` wraps the closure in
`IsentropicDiffusion`: the rotated (Redi) tensor of θ in the triad form,
with κ_v = `diapycnal`·κ_h across the isentropes (`nothing`: the
stability-corrected closure's own κ_v, < κ_h in stable air), tapered above
`slope_max` and isotropic where
N² ≤ 0 ([Tensorial scalar diffusivity](../theory/sgs.md#Tensorial-scalar-diffusivity)).
It is option-only, with smoke variants (`diapycnal = 0.1`, and the
corrected κ_v) that check K is positive semi-definite in every cell. The
flagship is a daytime convective case in which N² ≤ 0 over most of the
domain. A stateful closure (Lagrangian dynamic, prognostic
TKE) keeps its memory in that object; the cores commit it once per
accepted step and `step_adaptive!` rolls it back with the state on a
rejected step. A new provider goes in its own file,
`couplings/sgs_<name>.jl`: the couplings are included in sorted order,
so it loads after `sgs.jl`, which defines the slot. Which providers combine with the
`:core` and `:timestepper` slots is in [Dynamical core and time
integration](#Dynamical-core-and-time-integration).

**SGS diffusion of the tracers (#177).** The closure's eddy diffusivity
reaches every scalar the case transports. θ gets it inside `step!`; the
tracers get it in the `:transport` stage: after its advective update each
tracer with `advect = true` (qv, ql, the chemistry species) is advanced by
`dt · ∇·(κt ∇q)` with `sim.κt` of the last dynamics step
(`TracerSGSDiffusion`, `sgs_diffuse!`). The pollutant, which has its own
`:tracers` stage, calls the same helper. The flux divergence is the src
kernel `add_eddy_diffusion_scalar!` with the case's solid mask (periodic
lateral halos): no flux through a face next to a building and solid cells
untouched, so vapour neither leaks into the building interiors nor draws
on the stale value they hold. On a sound-proof core it passes the
constraint weights, giving `(1/W)∇·(W κt ∇q)`, which conserves `Σ W q`;
on the anelastic core that is the ledger's `Σ ρ̄ q`. The diffusion books
its change of water as `sgs_diffusion` (zero to round-off) plus
`density_redistribution`. Its `smoke_check` asserts that the diffusion
alone conserves `Σ q V` (`Σ W q V`) to 1e-12 and that `κt ≥ 0`.
The core runs with `moisture = false`, so qv and ql are case tracers and
are mixed here, with the same κ as every other tracer. Before
#177 no tracer was diffused at all: they saw only the implicit
dissipation of WENO, a missing-physics bug. `sgs_diffusion = false`
reproduces that transport for comparison. Measured over the first hour of the summer flagship (06:00–07:00,
adaptive steps, 64×64×32), with vs without the diffusion: the mean qv of
the lowest fluid level 9.496e-3 vs 9.572e-3 kg/kg and its horizontal
standard deviation 4.18e-4 vs 4.96e-4 kg/kg (−16 %); the ground-level
pollutant maximum 5.01 vs 6.88 (−27 %, domain maximum 13.6 vs 21.5); in
the 1669 wall-nook cells (fluid with two or more solid lateral
neighbours) the largest vapour excess over the level mean 1.42e-3 vs
1.86e-3 kg/kg, so the diffusion relieves the IBM nook residual noted in
`scalar_advection.jl` but does not remove it; water residual −1.8e-14
kg/m², `sgs_diffusion` −6.1e-19 kg/m². Cost: +1.0 ms per advected tracer
and step (7 tracers: `scalar_advection` 57.3 → 64.3 ms/step), +0.9 ms for
the pollutant; the masked kernel takes 0.58 ms per call against 0.10 ms
unmasked.

*One shared κ, also under `:amd` (#234).* AMD offers a per-scalar
diffusivity from each scalar's own gradient
(`Operators.sgs_tracer_diffusivity!`, forwarded by `SolidMaskedSGS`); the
stage does not use it, by design. Real high-Re turbulence mixes all
passive scalars alike (one turbulent Schmidt number). Per-scalar κ would
mix NO, NO₂ and O₃ differently and move them off the photostationary
state, and mix qv and ql differently, so `q_t` and `θ_l` would no longer
be mixed as conserved variables. Its zero-gradient fallback
(`κ = ν_t/Pr_t = 3ν_t`) gives a uniform or flat field the largest κ of all
and over-mixes plume edges, and the adaptive step (`sgs_cfl_diagnostics`)
sees θ's κ_t, not the tracers'. An earlier probe of the `:amd` run measured
cell-wise κ differences against θ's of 54–64 % for qv and 66–92 % for the
chemistry, 28 % between NO and NO₂, 196 % between qv and ql, and a
fallback maximum of 5.9 against 2.6 m²/s. The design test
(`test/berlin_numerics.jl`, *one shared κ*) pins it: under `:amd` the
stage is bit for bit the stage with every tracer forced onto `sim.κt`, and
on its 12-step state AMD's own κ would differ from θ's by 77–80 %
(`Σ|Δκ|/Σκ`) for qv and the chemistry, 24 % between NO and NO₂, with ql
(zero there) at the fallback 3ν_t everywhere, maximum 5.08 against
2.71 m²/s. Rationale in full: [the SGS theory page](../theory/sgs.md#Closure-contract).

* **Pressure solvers.** All three solve the same 7-point Laplacian with
  the same boundary conditions (periodic, periodic, Neumann); the
  multigrid ones iterate to a relative residual of 1e-10. In the
  `:smoke` window after 100 steps, MG-CG differs from the FFT by at most
  6.1e-10 m s⁻¹ in u, v, w and 2.1e-9 K in θ, with max|∇·u| = 1.9e-13
  s⁻¹ (`test/berlin_numerics.jl`). They are here for the stretched-grid
  and cut-cell solver work (#151, #152), not for speed.
* **SGS.** `:tke` (default since #182) is described in
  [Prognostic SGS TKE (#182)](#Prognostic-SGS-TKE-(#182)) below.
  `:dynamic` is the Germano–Lilly coefficient, test-filter
  ratio 3, with the buildings' solid mask (`case.geom.sm.mask`): ν_t = 0
  in buildings, and the test filter and the averages see air cells only
  (#173). The default `averaging = :lagrangian` is
  `Operators.LagrangianDynamicSmagorinsky` (Meneveau, Lund & Cabot 1996:
  averaged along pathlines, no homogeneous planes needed), and by default
  in its scale-dependent form (`scale_dependent = true`, LASD, Bou-Zeid,
  Meneveau & Parlange 2005; #186) with `cadence = 4` (the Germano
  identities every 4th step, the memories transported every step).
  `scale_dependent = false` (LASI) and `averaging = :plane` are smoke
  variants; `:local` runs `Operators.DynamicSmagorinsky` too. See
  [the SGS theory page](../theory/sgs.md). `report` prints the mean C_s
  over air cells in the canopy and in the free stream (the levels
  between the highest roof + 1 and the sponge), and with LASD the
  geometric-mean β there. `smoke_check` asserts ν_t = 0 in buildings and
  β ∈ [1/8, 8] in the air. `CaseConfig(:smoke)`, 100 fixed steps:

  | averaging | canopy C_s | free-stream C_s | β canopy / free |
  | :-- | --: | --: | :-- |
  | Lagrangian, scale dependent (default, cadence 4) | 0.095 | 0.089 | 0.61 / 0.48 |
  | Lagrangian, scale dependent, cadence 1 | 0.091 | 0.086 | 0.62 / 0.48 |
  | Lagrangian, scale invariant (LASI) | 0.037 | 0.033 | — |
  | plane | 0.032 | 0.018 | — |

  Over the last 30 min of a 1 h smoke run, LASD gives C_s 0.090 in the
  canopy and 0.084 in the free stream (LASI: 0.053 and 0.104), β 0.76
  and 1.47. The time-mean wind above the roofs is 0.19 m/s with a
  maximum at 45 m, so Φ_m there is not a log-law diagnostic, and it is
  the same within 0.02 for LASD, LASI and Smagorinsky: the resolved
  flow, not the closure, sets the profile in this window.

  Cost on the 64 × 64 × 32 flagship mesh (loaded machine; coefficient
  plus commit per Euler step): LASI 31-40 ms, LASD 58-72 ms at
  `cadence = 1` (≈ 28 % of a step) and 25-33 ms at `cadence = 4`
  (≈ 15 %). The memory time scale ``T`` is 50-100 s (10th percentile to
  median, smoke case), so a 4-step cadence of 2 s steps is well inside it.
  Under RK3 the Lagrangian model is cheaper than the
  plane model, because it runs the Germano kernel once per step instead
  of once per substage. `:mlsgs` wraps `ML.MLSGS`; with
  the default `fn` (the Smagorinsky closure with `cfg.C_s`) the run is
  bit for bit the `:smagorinsky` run (tested over 100 smoke steps). There is no
  trained model.
  `:amd` and `:vreman` (`couplings/sgs_amd_vreman.jl`, #176) are the
  anisotropic minimum-dissipation and Vreman closures (see [the SGS
  theory page](../theory/sgs.md#Anisotropic-minimum-dissipation-and-Vreman-closures)).
  AMD runs with C = 0.3, its own κ_t and the buoyancy term with the
  core's g/θ₀; the case's tracers are diffused with that κ_t (θ's), not
  with AMD's per-scalar diffusivity (see *One shared κ* above). It is set to zero in the building cells: there the
  resolved velocity is ≈ 0, so the denominator ∂_l u_m ∂_l u_m vanishes
  while θ still varies, and the buoyancy term gave ν_t up to 19 m²/s
  inside buildings, against a fluid maximum of 1.6 m²/s. That only
  shortened the adaptive step: 20 steps instead of 12 for the 24 s smoke
  run. In the smoke matrix's everything-on run the two maxima are 8.6
  (solid) and 8.9 m²/s (fluid). The large values are the published
  model's: ν_t grows like 1/|∇u| wherever |∇u| → 0 while θ varies, and
  that can happen in fluid too. The wrapper (`SolidMaskedSGS`) forwards
  every contract generic to the closure and zeroes νh, νv, κh, κv and the
  tracer diffusivity in solid cells only. Its `smoke_check` asserts that
  the fluid cells match the bare closure bit for bit. Vreman uses c = 2.5 C_s² with the case's C_s. Their `smoke_check`
  evaluates the closure on the laminar version of the current flow (u and
  v replaced by their level means, w = 0). It requires ν_t = 0 exactly in
  the free stream above the inversion, where Smagorinsky gives up to
  0.013 m²/s. `report` prints the canopy and free-stream mean ν_t against
  Smagorinsky with `cfg.C_s`. Neither is the default; flipping it is the
  maintainer's call.
  Measured in `:smoke` after 24 s (adaptive stepper), mean ν_t over the
  fluid cells, canopy / free stream (the levels from the highest roof + 2
  to the sponge, which include the convective layer below the 60 m
  inversion):

  | closure | canopy | free stream | steps |
  | :-- | --: | --: | --: |
  | AMD | 0.082 | 0.033 | 12 |
  | Vreman | 0.017 | 0.0033 | 9 |
  | Smagorinsky, C_s = 0.1 | 0.030 | 0.0063 | 9 |

  All values are in m²/s. AMD is above Smagorinsky in the heated canopy
  and in the mixed layer because of the buoyancy term, and it is zero in
  half of the fluid cells. From 72 m up its level means fall to
  4–25 × 10⁻⁴ m²/s, against 1.2–4.1 × 10⁻² m²/s for Smagorinsky
  (whose top levels include the sponge).
  `:sigma` (`couplings/sgs_sigma.jl`, #184) is the σ-model of
  Nicoud et al. (2011), option `C_σ = 1.35`. Its ``ν_t`` vanishes in
  laminar shear and 2-D flow; `report` prints the canopy and free-stream
  mean ``ν_t`` next to Smagorinsky's on the same flow. After 100 smoke
  steps that is 3.1e-2 (Smagorinsky 2.5e-2) m²/s in the canopy and 4.0e-3
  (2.3e-3) above the sponge start, where the sponge now holds a nearly
  uniform geostrophic stream with weak 3-D fluctuations (γ = 0.96); with
  `sponge_to_geostrophic = false` that region is a laminar sheared layer
  (γ = 0) and σ gives 5.3e-4 against 2.0e-2. Its `smoke_check` is the
  laminar-reference check AMD and Vreman share (``ν_t = 0`` exactly on
  the level-mean flow). It runs the stress form (option `stress`) and
  costs 17.8 against 16.5 ms per smoke step. Since P59-e (#195) its
  `smoke_check` also asserts σ's ``ν_t < 10^{-2} ν`` above the sponge
  start (the relaminarisation criterion). It is off by default, and
  switching the default is the maintainer's call. `Operators` is exempt,
  so no coverage list changes.
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

For scale, a whole default step cost about 40–60 ms under the same load
at #132 (`timing_report`, radiation excluded), of which `step!` was
12–17 ms; the surface exchange (#129) and the chemistry (#135) have
since added to it (see their cost paragraphs). The
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
retried piece carries them, and it passes the `sgs` closure to
`step_adaptive!` too, which snapshots the closure's memory with the state
and restores both on a rejected step. The diffusive limit uses the
selected closure's ν_t and κ_t of the current flow (`sgs_cfl_diagnostics`,
the pure `compute_sgs!`: it commits nothing), also with `C_s = 0`, since
`step!` applies the closure whatever `C_s` is. `:imex` passes them to
`step_imex!`, whose `momentum_advection` is the explicit operator and
whose `sgs` supplies the frozen ν_eff and κ_eff. Tested with a stateful
closure that is Smagorinsky plus a commit counter in its memory: one
commit per accepted step under every stepper (12 fixed, 9 adaptive,
12 IMEX in the smoke window), the flow bit for bit the plain run of each
stepper; and with a forced retry (one rejected try at 10⁶ × ν_t) the
rejected commit is rolled back, so the commits equal the accepted pieces
(6 after 5 steps, one of them halved).

**Incompatible pairs** (declared, so selecting one is an error that
says why, and the smoke matrix steers around them):

| provider | incompatible with | why |
| :-- | :-- | :-- |
| `timestepper = :adaptive`, `:imex` | `core = :anelastic`, `:pseudo_incompressible` | `adaptive_dt`, `step_adaptive!`, `step_imex!` are typed on `BoussinesqSimulation` |
| `core = :anelastic`, `:pseudo_incompressible` | `pressure_solver = :cosine_fft`, `:multigrid`, `:mg_cg` | those solve the constant-coefficient ∇²φ = ∇·u*, not the weighted projection; select `:density_weighted_cg` |
| | `momentum_advection = :weno_z` | an unweighted operator; the core advects momentum in its W-weighted centred flux form (`:centered` passes no keyword to it) |
| `pressure_solver = :density_weighted_cg` | `core = :boussinesq` | it solves only the weighted projection of the sound-proof cores |

Behind the declarations, `build_core` of a sound-proof core rejects any
`sim_options` keyword other than `tol` and `maxit`. Every `:sgs` provider
combines with every core: the sound-proof step takes the contract's
`sgs` closure like the Boussinesq one (#167; before it, `:dynamic` and
`:mlsgs` were incompatible with the sound-proof cores).

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
bottom), `qv_clamp` (the Kessler coupling's clamp), `transport` (the
change of water the transport stage makes; zero for a conservative
scheme, and measured at ~1e-19 kg m⁻² at ρ = 1 since #132) and
`sgs_diffusion` (the same for the tracers' SGS diffusion, #177; ~1e-19). The ledger
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

On master (`915bd49`, after #128–#133, adaptive step) the same hour
gives a domain Rn of 222.28 W m⁻² and a residual of 19.02 W m⁻²: window
sun −16.57, glass −2.50, dew heat +0.06. The dew term has almost gone because since #129
the RadiationFog dew acts only over the β = 0 surfaces (roofs, glass);
the soil and water surfaces take up dew as negative evaporation. This
was measured with a reconstruction of the method above, not the
original script, so it is indicative.

**The `:smoke` fog seed (#127, corrected in #131).** The #127 seed set
`q_v = q_s(θ, p)`, treating θ as the temperature, while the fog
adjustment uses `T = θ Π(p)`, about 1.1 K warmer at 1013 hPa. The air
was at 93.5 % RH and the fog evaporated completely on the first
`:fog_dew` call. Since #131 the seed is `q_s(θ Π̄_k, p̄_k)` and the fog
persists. Remeasured on master (`915bd49`, `test/berlin_smoke.jl`
configuration, 16 × 16 × 16, every addon on, 24 s at noon: with the
adaptive default that is 9 steps, the first of 2.2 s, mean 2.67 s, not
12 of 2 s):

| | summer | winter |
| :-- | --: | --: |
| ql left after the first `:fog_dew` call, called directly on the seeded state (dt of the first step, 2.2 s) / within step 1 | 99.91 % / 99.13 % | 100.0 % / 98.1 % |
| ql left after 24 s (9 steps) | 84.1 % | 75.1 % |
| domain Rn (W m⁻²) | 568.41 | 70.10 |
| domain residual Rn − H − LE − G − ΔS (W m⁻²) | −4.87 | +7.86 |
| of which dew heat / window sun / glass (W m⁻²) | +14.99 / −6.31 / −3.81 | +0.09 / −6.92 / −1.03 |

The other summer terms are H −1.30, LE −1.24, G 363.36, ΔS 212.46
W m⁻², closure −2.9e-13. The dew heat now sits only on the β = 0
surfaces: a class mean of +55.6 W m⁻² on roofs and +56.3 on glass. The
same model without fog (the pre-#131 seed, whose fog evaporates at
once) has a summer Rn of 602.30, so the fog darkens the smoke window by
about 34 W m⁻².

Earlier values of this paragraph (residual −34.4 / −33.9 W m⁻² with
dew heat +44.4 at #127–#132; Rn 34.8 instead of 388.7 and residual
−53.2 at #131) do not reproduce on master. #128–#132 changed the
shade, the cloud shadow, the surfaces and the seed in between; which of
these changes accounts for the difference was not isolated. Even the
pre-#131 seed now gives a summer residual of −3.14 W m⁻² (dew heat
+13.34, window sun −6.31, glass −3.88), winter +8.0, with 0 % of the
fog left after the first call and after 24 s (that reproduction uses p̄_k, since the 8 km scale-height pressure
it originally used is gone).

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

In the `:smoke` configuration the summer domain residual was −33.9
W m⁻² after #128; on master it is −4.87 W m⁻² (see the table under the
#127 baseline).

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
* **Momentum.** The ground is a wall of `Aeolus.WallModels` (#169):
  every surface step hands each column a `MOSTWallLaw` with its `z0m`
  and the Obukhov length of that step (`matching_obukhov_length`, the
  `L` for which the law gives back the step's ``u_*``), and every step
  `apply_wall_stress!` evaluates it at the instantaneous first-air-cell
  wind and applies the stress as an implicit drag on the first-air-cell
  velocity faces (over the `k_air` level mask, after the IBM);
  `bus.u_star` is that step's ``u_τ``. The momentum removed equals
  ``Σ τ_w(U)\,\mathbf{u}/U\,Δt`` to ``O(aΔt)`` (1e-4 in the test). With
  `L` and the wind frozen the operator reproduced the drag of #129
  (``u_*^2/U`` held between surface steps) exactly (0.0 relative
  difference). Option `wall_formulation` selects the local form
  (default), Schumann's or Moeng's plane-mean form (smoke variants). In
  `:smoke` (surfaces every 4 s) per-step evaluation changes the ground
  mean ``u_*`` after 1 h from 0.05647 to 0.05644 m s⁻¹ and the street
  wind from 0.47784 to 0.47796 m s⁻¹; the operator costs 1.2 ms per call
  on the 64 × 64 flagship plane. The `:wall` ground used to be
  free-slip. It stays a free-slip `:wall` carrying the wall-model drag,
  not a resolved `NoSlipWall` (#172): at Δx = 15.6 m a resolved no-slip
  wall would be unphysical
  ([Transition and wall-bounded flow](../theory/transition.md#The-Berlin-case)).
* **Wall-model input** (P57-b, #183). The wind the ground law is
  evaluated at is passed through the exponential time filter of Yang,
  Park & Moin (2017) (`WallModels.MatchingInput`), on by default with
  `input_T_f = 10 s` and advanced once per accepted step in
  `:surface_update`; the stress keeps the direction of the instantaneous
  first-cell wind. The matching height stays at the first air cell
  (`h_wm_cells = 1`): three cells above a street (18.75 m) lie in or
  above the canyon, where no log law holds. `h_wm_cells = 3` samples the
  third air cell of the park and water columns (`h_wm_classes`) — a
  smoke variant. The smoke checks require `bus.u_star` to be the wall
  law at the step's wind and matching height (1e-12), the matching cells
  to be air and the filter state to be finite. Measured in `:smoke` (1 h,
  last 30 min): street ``τ_w`` 7.77e-3 vs 7.61e-3 m² s⁻² without the
  filter, temporal RMS 1.054e-3 vs 1.029e-3, sub-60 s RMS 2.24e-6 vs
  2.18e-6 (the smoke flow has almost no fast fluctuation to filter);
  ground ``u_*`` 0.05260 vs 0.05225 m s⁻¹.
* **Non-equilibrium ground** (P57-c, #189). `wall_model =
  :nonequilibrium` (default `:equilibrium`) makes the ground stress that
  of a `WallModels.TBLEWallModel`: the thin-boundary-layer equations on a
  30-point column under every ground and roof column (rough, the
  surface's `z0m`), between the surface and the same filtered matching
  wind, with the resolved pressure gradient at the first air level
  (one-sided next to a building) and `tble_terms` (default
  `:pressure_gradient`). Applied by the same `apply_wall_stress!` face
  loop in `:post_predictor`, advanced once per step in
  `:surface_update`; `bus.u_star` is the columns' ``u_τ``; the surface
  heat and moisture fluxes stay MOST. Neutral (no stability correction in
  the TBLE). A smoke variant runs it, with checks that every surface
  column has a finite TBLE column with its surface's roughness and that
  `bus.u_star` is ``|τ_w|^{1/2}`` of the columns. Flagship (64 × 64 × 32,
  100 steps after 300): +15.6 ms per step in the coupling's stages (9.4 %;
  not proposed as default), and on the street columns the TBLE and
  equilibrium stresses differ in sign on 11.1 % and in sign or by > 30 %
  on 77.3 %: the canyon pressure gradient (median 3.3e-3 m s⁻²) times the
  3.1 m matching height is as large as the stress.
* **Dynamic roughness** (P57-e, #190; option `z0m = :dynamic`, a smoke
  variant, off by default). The ground law's ``z_{0m}`` becomes
  `WallModels.DynamicRoughness`, ``\sqrt{z_{0,f}^2 + α^2σ_Δ^2}``:
  ``σ_Δ`` is the RMS of the OSM building heights (`osm_height_map`,
  `dynamic_refine = 4` times finer, true heights) about the IBM
  staircase, ``z_{0,f}`` is the column's table value (still the
  roughness of the surface-layer solve), and α is solved every surface
  step from the first-air wind. The smoke checks require the law's
  ``z_0`` to equal the model's (1e-12), and it must be finite, at least
  the table value and below the matching height. Measured in `:smoke`
  over 30 min:
  * street ``z_0`` 0.0555 m against 0.05 m (range 0.05-0.37 m);
  * street ``u_*`` +1.5 %, ground ``u_*`` +5.7 %;
  * cost 1.1 ms per surface step.
* **Extra surface energy.** `bus.sensible` (window sun, glazing
  transmission, dew heat) is collected every step and delivered to the
  surfaces as `Q_extra` at their next step (`step_surface!` of `SlabLand`
  and `SoilThermalColumn` take it since #129). The ledger books the
  pending energy in ``ΔS``, so the per-class closure stays at round-off
  (≤ 3.1e-11 W m⁻² in 1 h of the flagship, wall class at noon).
* **Dew.** One vapour exchange per surface: the soil and water surfaces
  take up dew as negative evaporation (into the soil water); the
  RadiationFog dew reservoir of `fog_dew` acts only over surfaces with
  β = 0 (roofs, glass), above the roof.
* **No clamps.** H, LE and Ts are not clipped (the Kessler qv clamp
  stays for #143).
* **Maximum sustainable heat flux (P58-e, #193).** Every surface step also
  evaluates each column's MSHF (Van de Wiel et al. 2012;
  `SurfaceExchange.surface_mshf_ratio`): one maximisation of the
  Monin–Obukhov flux curve at the column's wind ``\max(U, U_{min})``,
  `z_ref`, `z0m` and ``θ_{va}``. It stores `H_max` (W m⁻², output
  `H_max_sfc`) and the ratio of the downward buoyancy flux to it
  (`mshf_ratio`, negative where unstable). The report prints the mean
  ratio over the stable street, roof and park columns and the share of
  them beyond the MSHF point (``z_{ref}/L`` past the peak of the flux
  curve, output `mshf_beyond`). The smoke check
  asserts ratio ≤ 1 + 1e-8 for every column on the MOST fluxes of the
  step's start. The fluxes a soil column returns are linearised in
  ``T_{skin}`` and may overshoot by O(ΔT²): measured ≤ 3.1 % in the smoke
  run, checked < 10 %. Over the 21 h flagship the delivered ratio reaches
  2684. This happens on calm soil columns where ``H_{max}`` is below
  0.1 W m⁻² and a linearised step turns an upward buoyancy flux into dew,
  so the excess is a few W m⁻². At 03:00 the report reads street 0.02,
  roof 0.04, park 0.00, with 100 % of the stable columns beyond the MSHF
  point. The MSHF costs 48–63 ms per surface step (every 30 s), about 2 % of
  the run's wall time. See
  [Stability functions → MSHF in the Berlin case](../theory/turbulence_stability.md#MSHF-in-the-Berlin-case).

### Measured (flagship, 64 × 64 × 32, adaptive step, 1 h)

Remeasured on master (`915bd49`, with the reference state of #131, the
halo fix of #132 and the adaptive stepper, the default since #166). Two
1 h runs, from 06:00 and from 12:00 local on 20 June
(`CaseConfig(:alexanderplatz; nsteps = 1800, start_h = 6 or 12)`,
default couplings), each run to 3600 s of simulated time: 1648 steps of
1.45–3.36 s (mean 2.18 s) from 06:00 and 2157 steps of 0.76–3.23 s
(mean 1.67 s) from 12:00. The bus fields are averaged over simulated
time, each step weighted by its dt, over each interval. "Slab H" is the same run on the #127 branch (slab scheme),
kept from the #129 measurement. Domain means over 4096 columns (the two
columns without a surface count as zero), W m⁻²; "street+wall" is the
mean over the 1690 street and 978 wall-adjacent columns, "roof" over the
873 roof columns, u\* (m/s) over street and roof columns:

| interval | slab H | MOST H | MOST LE | MOST H street+wall | MOST H roof | u* street | u* roof |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| 06:00–06:05 | −24.1 | −0.4 | 0.3 | −0.5 | −0.1 | 0.011 | 0.002 |
| 06:45–07:00 | −15.9 | 16.5 | 42.9 | 25.2 | −0.0 | 0.066 | 0.0007 |
| 12:00–12:05 | −23.7 | 4.5 | 22.1 | 7.0 | −0.1 | 0.063 | 0.002 |
| 12:45–13:00 | −8.7 | 74.1 | 13.4 | 111.9 | 3.0 | 0.107 | 0.027 |

The street columns alone give H = −0.7, 35.6, 11.2 and 168.4 W m⁻² in
the same four intervals. (The scripts of the #129 measurement no longer
exist; the class averages are reconstructed from their column labels.
Before #131, with `T_air = θ − Γ_d z`, an 8 km scale-height pressure and
ρ = 1, the MOST columns read 22.7 / 46.2 / 34.5 at 06:45–07:00 and
86.5 / 15.2 / 127.9 / 10.5 at 12:45–13:00.)

Why H differs: the slab scheme exchanged at a fixed `C_h = 1e-3` and
`U = 3 m/s`, i.e. 3.0 W m⁻² K⁻¹ whatever the flow, from a slab that
starts 8 K colder than the air (Ts0 = 285 K under θ = 293 K), so its H
was about −24 W m⁻² at the start and still negative after an hour at
noon. With Monin–Obukhov and the resolved wind at the first air cell
(0.48 m/s over streets, 0.20 m/s over roofs at 07:00) the cold,
stably stratified start is almost decoupled (H ≈ 0), and the soil
columns' massless skins heat within minutes, so the streets turn
unstable: after half an hour street+wall H is 25 W m⁻² in the morning
and 112 W m⁻² at noon (street columns alone 36–168). The cold-started
roof slabs (C_s = 1.8e5 J m⁻² K⁻¹) stay colder than the air for most of
the hour, stable and decoupled.

**u\* over roofs.** The plan expected 0.2–0.6 m/s at midday. Measured:
0.038 m/s mean over roofs at 13:00 (0.0010 at 07:00); streets 0.109.
Two reasons, both in the case rather than the scheme: the wind at the
first air cell above the roofs is only 0.25 m/s (the case is driven by a
1 m/s geostrophic wind, u0 = 1 m/s), which gives a *neutral* u\* of
0.0246 m/s for z0m = 0.05 m (the mean of κ max(U, 0.1)/ln(z_ref/z0m));
and the roofs start at 285 K, so they are stably stratified for most of
the first hour (roof skin − air −5.07 K averaged over 12:15–12:30). At
13:00, with the roofs 1.1 K warmer than the air, u\* is 1.5 × its
neutral value (the convective enhancement). The expectation assumed a
few m/s of wind.

**Street β.** The street soil starts at θ = 0.10 (β = 0.44). In the
noon hour its mean β falls as the top layer dries, to 0.373 at 13:00
(hour mean 0.394, range 0.373–0.441), and the evaporation the soil can
deliver collapses: the domain-mean LE is 70.1 W m⁻² between 12:15 and
12:30 and 14.2 W m⁻² after 12:30. Averaged over the noon hour (the
ledger), the street columns have Rn 617.0, H 109.5, LE 68.9 and G 438.6
W m⁻², the roofs Rn 587.35, H 0.81 and ΔS 586.6 W m⁻² (before #131:
637 / 125 / 69 / 443 and 614 / 2.8 / 611).

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
hydrology, 915 slabs) is 16.3 ms (06:00 run, 1648 steps) and 13.6 ms
(12:00 run, 2157 steps) per dynamics step averaged at the 30 s cadence,
i.e. 26.9 and 29.2 s per simulated hour; the air-side update is
1.30 / 1.08 ms and the drag 0.26 / 0.24 ms per step (remeasured on
master `915bd49` with the adaptive step, on a 24-core machine shared
with five other runs; at #129, with the fixed 2 s step, 10–17, 0.9 and
0.15 ms, 18–31 s per simulated hour). The soil column with
hydrology costs about 56 µs per call, a slab 16 µs (measured at #129,
not remeasured).

**Known limitations.** Streets are bare loam: SoilColumn has no sealed
surface, so a street evaporates until its top layer dries (a sealed
option belongs in src). No rain reaches the soil (the microphysics books
precipitation as a domain total). The roughness lengths are per class
options (street/roof 0.05 m, park 0.1 m, water 1e-4 m, glass 0.01 m,
z0h = z0m/10), not a survey. Before the halo fix of #132 the more
vigorous noon convection made the stale-halo transport leak larger (it
created 3.4e-3 kg m⁻² of water in the noon hour, and the Kessler clamp
filled the negative qv it left aloft, +8.9e-5 kg m⁻²). On master both
are zero to round-off in both runs: transport ≤ 3e-18 kg m⁻² in
magnitude, qv clamp 0; the water ledger closes to −1.7e-14 kg m⁻².

## Facade wall model (#179)

The building facades are walls of `Aeolus.WallModels`
(`couplings/wall_model_facades.jl`, addon, on by default). Before #179
the facades had no wall model: the direct-forcing IBM zeroes the
velocity inside the buildings, and the facade momentum flux was the
viscous and eddy-viscosity stencils' ``(ν + ν_t)(u - 0)/Δ`` — an
unresolved no-slip at `cfg.ν` = 2 m² s⁻¹ and Δ = 15.6 m.

| | |
| :-- | :-- |
| walls | `immersed_walls` on `case.geom.sm` at `init!`: u faces across ±y, v across ±x, w across ±x/±y (the ground and roofs stay with `surface_exchange`, #169) |
| law | `RoughLogLaw(z0m)` at ``h = Δ/2``; `z0m` from `bus.z0m_class` (the surface exchange's per-class table, put on the bus in its `build`): `SURFACE_GLASS` for a glass-tagged building column, `SURFACE_WALL` otherwise (0.01 m / 0.05 m by default) |
| `:body_force` (40) | `remove_stencil_wall_flux!`: the stencil's facade flux out of the tendency (with ``ν_t`` when the step has an eddy viscosity) |
| `:post_predictor` (12) | `apply_immersed_wall_stress!`, implicit, after the `ibm` direct forcing (10) and before the ground's wall stress (15) |
| checks | facade ``τ_w ≥ 0``; no stress written into a solid cell's face; the total facade drag and the removed stencil drag oppose the mean canopy wind |

Measured in `:smoke` after 1 h (3600 s, adaptive step), on vs off
(`physics = Dict(:wall_model_facades => false)`):

| | facades off | facades on |
| :-- | --: | --: |
| facade ``Σ τ_w ΔA`` (m⁴ s⁻²) | — | 25.7 (⟨τ_w⟩ = 8.3e-4 m² s⁻² over 30 859 m²) |
| ground + roof ``Σ u_*^2 ΔxΔy`` (m⁴ s⁻²) | | 250.5 |
| facade share of the surface drag | | 9.3 % (26.3 % after 12 steps) |
| stencil facade drag removed (m⁴ s⁻²) | | 504 (the modelled drag is 15.5: the stencil was 33× it) |
| pedestrian-level (k = 1) mean wind, 106 street columns (m s⁻¹) | 0.4780 | 0.4793 (+0.3 %) |
| k = 1 mean wind, all 187 open columns (m s⁻¹) | 0.3791 | 0.3854 (+1.7 %) |
| ground mean ``u_*`` (m s⁻¹) | 0.0564 | 0.0571 |

The coupling took 2.2 s of a 5567 s run (0.04 %; machine heavily
loaded). The cut-cell method of #179 is not wired: the case has no
cut-cell provider (CutCell is pending on #152), which picks it up.

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

Remeasured on master (`915bd49`: the reference-state surface pressure
of 101.325 kPa, the MOST / soil surfaces of #129, the adaptive step of
#166). 1 h runs of the flagship, 64 × 64 × 32, each run to 3600 s of
simulated time (1429 / 1648 / 2157 adaptive steps from 00:00 / 06:00 /
12:00), radiation every 600 s, default couplings; values from the last
radiation call of the hour (at 3600 s less one step; at #130 this was
step 1800). Winter on the `:smoke` window (smoke cadences, no fog
seeded; 690 adaptive steps to 1800 s), which the smoke check uses, and,
since the adaptive stepper lets it run, on the winter flagship itself
(`CaseConfig(:winter; nsteps = 1800, start_h = 12)`, 3378 adaptive steps
of mean 1.07 s to 3600 s). "Above 50 m" is the
ρΔz-weighted mean over the levels whose centre is above 50 m; the old
`0.95σT̄s⁴` uses the arithmetic mean of `bus.Ts`, as the deleted
`lw_from_walls(mean(Ts))` did.

| | 00–01 h | 06–07 h | 12–13 h | winter 12–12:30 (smoke window) | winter 12–13 h (flagship) |
| :-- | --: | --: | --: | --: | --: |
| `L_sky` (W m⁻²) | 335.3 | 337.9 | 342.6 | 238.6 | 240.9 |
| old `0.95σT̄s⁴` at the same T̄s | 360.1 (T̄s 285.94 K) | 402.2 (293.95 K) | 436.8 (300.09 K) | 310.3 (275.49 K) | 309.2 (275.25 K) |
| GHI, column (W m⁻²) | 0 | 559.0 | 830.6 | 234.2 | 200.8 |
| Bird (the radiosity's sky: PW 2 cm, O₃ 0.33), aerosol-free / AOD₅₀₀ = 0.16 | 0 / 0 | 574.5 / 542.3 | 834.1 / 797.0 | 235.1 / 211.7 | 204.0 / 181.8 |
| LES-layer heating, LW / SW (K day⁻¹) | −13.0 / 0 | −4.3 / +3.6 | +2.45 / +4.7 | −0.2 / +1.4 | −1.4 / +1.2 |
| the same above 50 m, LW / total | −5.2 / −5.2 | −3.3 / +0.4 | −1.7 / +3.1 | −1.3 / +0.1 | −1.3 / −0.1 |
| lowest level (0–6 m), LW | −158 | −24 | +77 | +9.7 | −2.7 |
| precipitable water (kg m⁻²) | 12.3 | 12.3 | 12.3 | 3.5 | 3.6 |

(Before #129 and #131, with the slab surfaces and 100.92 kPa: L_sky
331.9 / 333.5 / 335.4 / 237.9, LES-layer LW −12.9 / −9.8 / −6.7 / −4.0,
lowest level −156 / −108 / −59 / −16 K day⁻¹.)

* **Sky longwave** is 25–94 W m⁻² below the old surface-based value in
  summer; the gap grows over the day because the MOST / soil surfaces
  warm to a mean of 300 K by 13:00. `L_sky` itself is almost constant
  (335–343 W m⁻²; it depends on the air, not on the surfaces). In
  winter it is 239 W m⁻², below the 300–380 W m⁻² the issue expected —
  that range was a summer expectation, and a clear sky over 3.5 kg m⁻²
  of water at 274 K gives about 0.7 σT⁴.
* **Clear-sky GHI.** Against Bird with no aerosol and the column's own
  precipitable water and ozone (about 442 DU in summer, 367 in winter),
  the column is −0.90 % at μ₀ = 0.86 (12:00), −1.20 % at μ₀ = 0.81
  (13:00), −3.45 % at μ₀ = 0.58 (07:00) and −4.91 % at the winter noon
  sun, μ₀ = 0.27 (first step of the `:smoke` window; −5.39 % at the end
  of the window, μ₀ = 0.25; −6.31 % at 13:00 on the winter flagship,
  μ₀ = 0.22): TwoStream falls below Bird as the sun gets low. The smoke check asserts 5 % for μ₀ ≥ 0.5 (the issue's noon
  criterion) and 10 % below. Against Bird with the case's urban AOD the
  column is 4.2 % higher at 13:00 and 10.6 % at the low winter sun: the
  TwoStream column has no aerosol until the aerosol-optics coupling
  (#142).
* **Heating.** At night the lowest LES level cools at 158 K day⁻¹: the
  case starts its surfaces at 285 K under 293 K air, and the air near
  the ground radiates into the colder surface. By day the surfaces
  overtake the air (mean Ts 293.9 K at 07:00, 300.1 K at 13:00), and the
  lowest level's longwave goes from −24 K day⁻¹ at 07:00 to **+77 K
  day⁻¹** at 13:00; the winter smoke window heats it too (+9.7), while on
  the winter flagship at 13:00 it cools slightly (−2.7). Above 50 m
  the longwave cooling is −1.7 to −5.2 K day⁻¹ in summer; the noon hour
  is inside the −1…−3 K day⁻¹ the issue expected, the night and morning
  are beyond it. Both the cold surface and the humid initial profile
  (qv = 10 g kg⁻¹ at the ground) cause the excess: in the initial state
  (12:00, one column call) the mean above 50 m is −5.11 K day⁻¹; −3.18
  with the surfaces at 293 K; −1.54 with that and `qv0` halved.
* **Energy.** ``\int ρ c_p\,\dot T\,dz`` over the LES layers equals the
  net-flux convergence to round-off (a smoke check, 1e-10), and the θ
  increment is that heating times dt in every fluid cell and zero in
  every solid one (a smoke check on the field). The air therefore takes
  up the column's convergence times the fluid fraction of each level.
* **Ledger.** In the 06–07 h hour the domain Rn is 222.28 W m⁻²
  (closure 5.2e-12; GHI 559.0 against Bird's 542.3). At #130 it was
  228.59, against 244.28 in the #127 baseline; that baseline cannot be
  rerun on master, and Rn now also carries the #129 surfaces, so the
  share of the drop due to the lower sky longwave alone is no longer
  measured.
* **Cloud shadow** (`:smoke`, default couplings, the smoke test's seeded
  fog of 0.2 g kg⁻¹ in the lowest 18.75 m, noon, μ₀ = 0.860): on step 1
  the direct-beam transmittance has a mean of 0.848 (``e^{-0.2}``, the
  delta-scaled τ′ = 0.26 τ ≈ 0.17 over μ₀) and a minimum of 0.803; shaded
  non-roof columns span 0.803–0.937. The shadow takes 117.7 W m⁻²
  (domain mean) off the surface shortwave; the old formula would have
  kept 21 % of the shortwave under the same fog, and then again on every
  step. Most roofs above the fog are unshaded, but the lowest roof
  transmittance is 0.928 (not checked: probably roofs below the 18.75 m
  fog top).
* **Cost** per radiation call, measured warm (mean of 10 calls after a
  600 s run, 230 adaptive steps) on the flagship on master `915bd49`:
  column radiation 45.0 ms by day, 27.9 ms at night; cloud shadow
  13.3 ms (at #130: 38 / 23 / 13; at `2895fed`: 33.4 / 20.8 / 11.1, on a
  less loaded machine); the radiosity solve itself takes 1–3 s. At the
  600 s cadence that is about 0.1 ms per simulated second (0.2 ms per
  step of 2 s), plus
  0.36–0.59 ms per step for applying the heating (all on a shared
  24-core machine).
  The radiation cadence (`cfg.cadence[:radiation]`) is shared with the
  radiosity: the sky longwave and the beam must be the ones the
  radiosity solve uses.

**The winter flagship blows up with the fixed 2 s step, and runs with
the adaptive one** (not a radiation problem). With `timestepper =
:fixed`, `CaseConfig(:winter; start_h = 12)` on the 64 × 64 × 32 grid
had θ ∈ [270.0, 276.8] K and |u| 1.95 m s⁻¹ at step 125 on `2895fed`,
then θ ∈ [44.8, 428.4] K at step 140 and [−95, 587] K with |u| 15 m s⁻¹
at step 141, after which a saturation call stopped on a negative
temperature (`DomainError` in `saturation_vapor_pressure`); the fixed
stepper still blows up at step 141 on `a630153` (table above). At #130
it blew up earlier (θ ∈ [10, 353] K by step 118, −700 K at step 119),
and the #127 branch, without the radiation couplings, did the same (|u|
grew from 1.9 to 12 m s⁻¹ between steps 100 and 118). With the adaptive
stepper, the default since #166, the winter flagship runs. Measured on
master `915bd49`, 3 h from 12:00: 8606 steps (dt 0.70–2.69 s, mean
1.07 s in the first hour and 1.25 s over the three), θ within
[270.2, 278.4] K throughout, max |u| 2.46 m s⁻¹ at 12:10 falling to
1.17 m s⁻¹ at 15:00, Ts ∈ [262.0, 283.8] K at 15:00; the surface energy
ledger closes to −3.8e-11 W m⁻² (domain) and the water ledger to
−4.5e-15 kg m⁻². Its radiation is in the last column of the table above.

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

## Turbulence diagnostics (#170)

`turbulence_diag` (`couplings/turbulence_diag.jl`, addon, on by default)
is the Berlin coupling of the Era VI stability contract
`TurbulenceStability` (see [the theory page](../theory/turbulence_stability.md)).
At the output cadence it evaluates, on copies of `u, v, w, θ, q_v` whose
halos it fills with the core's boundary conditions (the model state is
not touched):

* the pointwise ``N^2`` of ``θ_v = θ(1 + 0.61 q_v)``
  (`buoyancy_frequency_squared!`, the z-interpolation of the core's
  buoyancy), ``|S|^2`` (`strain_rate_squared!`, the Smagorinsky stencil)
  and ``Ri_g`` (`gradient_richardson!`);
* per level, over the fluid cells: the mean ``N^2`` (`N2_mean`), the
  Richardson number of the means ``⟨N^2⟩/⟨|S|^2⟩`` (`Ri_g_mean`; the mean
  of the pointwise ratio is dominated by shear-free cells, where the
  ``S^2`` floor makes it huge) and the mean ``f_m(Ri_g)`` (`f_m_mean`).
  These are written as `(z, time)` profiles (`OutputField` dims `:z`)
  and summarised in the log line.

Option `stability`: `:brown` (`BrownEtAl1994`, the LEM / MONC functions,
default), `:lilly`, `:short_tail`, `:long_tail`, `:most`
(`MOSTConsistent` with the surface exchange's BDBH functions; the smoke
variant). `budget!` is a no-op: the coupling changes no state. P59-c
(#180) extends this addon with the transition diagnostics (below).

**Measured** (`:smoke`, after its 12 steps, noon, capping inversion at
60 m, `Γ = 3 K km⁻¹` above). In the first two levels above the
inversion (z = 65.6 m and 71.9 m) `Ri_g` of the means is 0.273 and
0.196 and the mean `f_m` (Brown et al.) 0.049 and 0.0057; with
`stability = :most` the same cells give 0.101 and 0.069 (no critical
Richardson number). Below the inversion θ is uniform, and the vapour
lapse of the initial state (``q_v ∝ e^{-z/800\,{\rm m}}``) makes
``θ_v`` decrease with height: ``N^2 ≈ -7 × 10^{-5}`` s⁻² at step 0, and
−2 to −16 × 10⁻⁵ s⁻² after 12 steps of noon heating, so `Ri_g` < 0 and
`f_m` = 2.5–10. The same vapour lapse removes about 2/3 of the dry
``gΓ/θ_0 = 1.0 × 10^{-4}`` s⁻² above the inversion (3.3 × 10⁻⁵ at step 0).
Cost per evaluation (compiled): 0.15 ms on the 16³ smoke grid
(`:most`: 0.89 ms, a root find per cell), 3.6 ms on the 64 × 64 × 32
flagship grid, once per output record (every 1800 s).

### Transition diagnostics (#180)

P59-c extends the same addon (there is no second diagnostic addon;
`modules_used = (:TurbulenceStability, :Transition)`) with the
laminar/turbulent diagnostics of `Aeolus.Transition` (see
[the theory page](../theory/transition.md)), written as `(z, time)`
profiles at the output cadence:

* `gamma_turb` — the turbulent fraction ``γ(z)`` of the fluid cells,
  `turbulent_indicator!` with ``|ω'|/⟨|ω|⟩ >`` `gamma_threshold`
  (option, default 0.2) and the plane normalisation
  (`gamma_normalization = :plane`, or `:volume`); solid cells are never
  turbulent;
* `Ri_supercritical_fraction` — the fraction of fluid cells with
  ``Ri_g > 1/4`` (`supercritical_fraction` of the `gradient_richardson!`
  field above: Transition does not define a Richardson number);
* `Re_grid` — ``|U_h| Δ/(ν + ⟨ν_t⟩)`` (`grid_reynolds_number`) with the
  plane-mean horizontal wind speed, ``Δ = (Δx Δy Δz)^{1/3}`` and `cfg.ν`.

The log line adds canopy (levels with buildings) and free-stream (above
the highest roof + 1, below the sponge) means
(`transition_summary`). `smoke_check` adds ``0 ≤ γ ≤ 1``, no turbulent
solid cell, the supercritical fraction in [0, 1] and a finite positive
grid Reynolds number.

**Measured** (`:smoke`, after 12 steps). Summer noon: ``γ`` = 0.74–0.99
on the nine canopy levels (mean 0.871), 0.02 at z = 59 m and 0 above the
60 m inversion, where the initial state is still laminar (free-stream
mean 0.438 because it includes the 53 m and 59 m levels); the
supercritical fraction is 0 in the canopy and 0.61 / 0.10 in the two
levels above the inversion (domain mean 0.044). Winter (`:winter`
initial state) at 06:00: domain-mean supercritical fraction 0.244, with
0.88–1.0 in the four levels from 53 to 72 m — drier air, so the vapour
lapse removes less of the inversion's ``N^2``. Grid Reynolds number at
``ν = 2`` m² s⁻¹: 4.0–5.7 on the smoke grid (Δ = 15.6 m × 15.6 m × 6.25 m);
on the 64 × 64 × 32 flagship grid after 60 steps it peaks at 5.7 (2.4 at
the ground, 0.14 at the top) — below the "about 8" estimated in the
issue. Cost per evaluation of the whole addon: 0.28–0.55 ms on the smoke
grid and 7.0 ms on the flagship grid (3.6 ms before #180), once per
output record.

### Flat-plate transition (#185): stated exception

The spatially developing boundary layer of P59-d (fringe region,
`FreeStreamBoundary` top, Blasius inflow, `FreeStreamDisturbance`) is not
coupled: the Berlin domain is laterally periodic with geostrophic forcing
and has no inflow direction, so a fringe and an inflow generator have no
place in it. Inflow boundaries arrive with nesting / ERA5 driving
(Phase 68). Transition stays coupled through `turbulence_diag` above. The
one shared piece that does run in every Berlin step is the damping
profile: the Rayleigh sponge (`sponge_α_max` = 0.05 s⁻¹ from 75 m in
`:smoke`) now takes its rate from `Operators.DampingProfile`, the
implementation the fringe also uses. The smoke run's final state is
bit-identical with the verbatim Phase-7 kernel reinstalled; the sponge's
column kinetic-energy sink ``∫α|\mathbf u_h|^2dz`` after the 12 smoke
steps is 0.139 m³ s⁻³. No Berlin list changed (`BERLIN_DEFAULT_OFF` stays
empty).

## Prognostic SGS TKE (#182)

`sgs_tke` (`couplings/sgs_tke.jl`) is the `:tke` provider of the `:sgs`
slot and its default: `PrognosticSGS.TKEClosure`, the Deardorff (1980) /
Moeng (1984) closure (see [the theory page](../theory/sgs_tke.md)). It is
PALM's urban-LES default and the only closure here whose mixing responds
prognostically to stratification (its ``0.76\sqrt e/N`` length scale; no
stability function is applied on top of it).

* The closure is the `sgs` of every dynamics step, so the cores evaluate
  ``ν_t = c_k l\sqrt e``, ``κ_t = (1 + 2l/Δ)ν_t`` from the committed ``e``
  and commit the sources (shear production, the exact dual of the SGS
  momentum operator; buoyancy production with the ``θ_v`` of the `qv`
  tracer; the implicit sinks) once per accepted step; `step_adaptive!`
  rolls ``e`` back with the state.
* ``e`` is the tracer `:e` (`advect = true`, `positive = true`, m² s⁻²,
  written to the NetCDF file). The `:transport` stage advects it with the
  case's scheme and limiter and diffuses it with ``2ν_t`` (its
  `diffusivity`), mask-aware; the closure runs with `transport = false`.
* ``l`` is capped at ``κ`` × `IBM.wall_distance` of the building mask and
  the floor (option `wall_cap = true`), ``c_ε`` gets Moeng's 3.9 above the
  floor and the roofs (`IBM.first_level_cells`), ``e = 0`` inside the
  buildings.
* The first air cell of each column (`bus.k_air`) holds
  ``e_s = u_*^2/\sqrt{c_k c_ε}`` with the friction velocity of the MOST
  surface exchange (`bus.u_star`, option `surface = true`).
* `init!` seeds ``e`` with the local equilibrium of the initial flow, at
  least `e_min = 1e-4` m² s⁻² (``e = 0`` is a fixed point).
* `budget!` books nothing: ``e`` carries no surface energy or water (its
  dissipation heating is neglected, as with every closure here).
* `smoke_check`: ``e ≥ 0``, ``ν_t ≥ 0``, ``e = 0`` in the buildings,
  ``κ_t ≥ ν_t``, and (Boussinesq core) ``Σ P_s V`` equal to the SGS sink of
  `kinetic_energy_budget(sim; sgs = closure)` to 1e-8. The smoke variant
  runs it without the wall cap and the surface value.

The default changed from `:smagorinsky` to `:tke` because the coverage
ratchet requires a new src module to run by default and no frozen list
may grow; whether it stays the default is decided at review (#46).
Berlin tests that pinned the Smagorinsky default now select
`:smagorinsky` explicitly where they test its bit-for-bit equivalences
(MLSGS, the commit counter, the stale-halo demonstration).

**Measured** (summer flagship, 64 × 64 × 32, adaptive step, 30 min from
12:00 and from 02:00 local, `:tke` vs `:smagorinsky` with `cfg.C_s = 0.1`;
"street canopy" is the fluid cells of levels 1–3, 0–18.75 m):

| | noon `:tke` | noon `:smagorinsky` | 02:00 `:tke` | 02:00 `:smagorinsky` |
| :-- | --: | --: | --: | --: |
| street-canopy mean ``e`` (m² s⁻²) | 0.0110 | — | 0.0011 | — |
| street-canopy mean ``ν_t`` (m² s⁻¹) | 0.0275 | 0.0326 | 0.0086 | 0.0174 |
| domain mean ``ν_t`` (m² s⁻¹) | 0.0756 | 0.0220 | 0.0412 | 0.0118 |
| whole step (ms) | 161.6 | 131.9 | 131.9 | 116.1 |

At night the street-canopy ``ν_t`` is 51 % below Smagorinsky's (16 % at
noon) and the canopy ``e`` is a tenth of its noon value: the stability
length and the wall cap act where the static closure only sees the
strain. Higher up the TKE closure mixes more: its neutral equilibrium is
Smagorinsky with ``C_s = 0.194``, against the case's ``C_s = 0.1``
(``(0.194/0.1)^2 = 3.8``). No clip was booked. Cost per call on this grid:
`advance_sgs!` 7.2–7.4 ms, the TKE coefficients 0.9 ms (Smagorinsky
1.0 ms); the `:e` tracer adds 8–11 ms to `scalar_advection` (80.1 vs
69.1, 71.8 vs 63.7 ms/step); whole steps +30 ms (+23 %) at noon and
+16 ms (+14 %) at night (loaded machine, sequential runs).

## Differential Reynolds stress (#188)

`sgs_reynolds_stress` (`couplings/sgs_reynolds_stress.jl`) is the
`:reynolds_stress` provider of the `:sgs` slot. It is an option, a genuine
alternative to the default `:tke`, and it uses the same module
(`PrognosticSGS`), so no coverage list changes. The closure is
`ReynoldsStressClosure` (see the Reynolds-stress section of
[the theory page](../theory/sgs_tke.md)).

* The closure is the `sgs` of every dynamics step. The cores take three
  things from the committed stresses: ``ν_t``, the Daly–Harlow tensor
  ``K = c_θτ`` (θ goes through the P55-c tensor operator), and the
  momentum ``−∂_jτ^{dev}_{ij}`` through the P55-b stress operator.
  `advance_sgs!` is the source stage, run once per accepted step: implicit
  sources (sub-stepped for growing modes), exact Coriolis, the
  realizability projection and the booking. `step_adaptive!` rolls it back
  with the state. A separate case stage could not be rolled back.
* The six stresses are the tracers `:tau11 … :tau23` (m² s⁻², written to
  the NetCDF file). The `:transport` stage advects them, with the
  diagonals `positive` and therefore flux-limited, and diffuses them with
  ``2ν_t`` (their `diffusivity`), mask-aware. The closure runs with
  `transport = false`.
* The wall cap ``κ d_{wall}`` (option `wall_cap`) and Moeng's 3.9 come from
  the TKE closure's functions. ``τ = 0`` in the buildings, and an edge that
  touches a building or the ground carries no stress.
* The first air cell of each column (`bus.k_air`, option `surface`) holds
  the LRR-IP constant-stress layer:
  * ``τ_{13}, τ_{23} = −u_*^2(\cos α, \sin α)`` against the cell's wind,
    with `bus.u_star` from the MOST exchange;
  * ``e_s = u_*^2/0.3395``.
* `budget!` books nothing, because the stresses carry no surface energy or
  water.
* **Incompatible** with `timestepper = :imex`: `step_imex!` represents the
  SGS momentum by an implicit eddy viscosity, and it refuses the closure
  (`Operators.sgs_explicit_stress`).
* **Stated exception.** The other tracers (qv, ql, the species, the
  pollutant) are diffused by the `:transport` stage with the isotropic
  ``κ_t = \operatorname{tr}K/3`` (`sim.κt`), not with the full tensor.
* `smoke_check`:
  * ``τ`` is positive semi-definite in every cell and ``τ = 0`` in the
    buildings;
  * the realizability corrections are below 1e-3 of the SGS energy;
  * ``Σ e V > 0`` and ``ν_t ≥ 0``;
  * the closure commits once per accepted step;
  * on the Boussinesq core, ``Σ P V`` equals the SGS sink of
    `kinetic_energy_budget(sim; sgs = closure)` to 1e-6.

  The smoke variant runs the closure without the wall cap and without the
  surface value. That variant caught a code bug: a single backward-Euler
  source step inverted a growing mode at a street corner, and the
  projection then added more SGS energy than the domain held. The fix is
  the sub-stepping.

**Measured.** Summer flagship, 64 × 64 × 32, adaptive step, 30 min from
12:00 and from 02:00 local time. "Canopy" means the fluid cells of the
levels that are more than 5 % solid, as for `:tke`.

| | noon `:reynolds_stress` | noon `:tke` | 02:00 `:reynolds_stress` | 02:00 `:tke` |
| :-- | --: | --: | --: | --: |
| canopy mean ``e`` (m² s⁻²) | 0.0133 | 0.0099 | 0.0025 | 0.0013 |
| canopy anisotropy ``II = b_{ij}b_{ji}`` | 0.068 | — | 0.066 | — |
| canopy ``τ_{33}/τ_{11}`` | 1.35 | — | 1.35 | — |
| canopy mean ``ν_t`` (m² s⁻¹) | 0.043 | 0.034 | 0.017 | 0.012 |
| domain mean ``ν_t`` (m² s⁻¹) | 0.096 | 0.074 | 0.056 | 0.040 |
| realizability corrections (30 min) | 203 383 | — | 502 282 | — |
| `step!` (ms/step) | 141.5 | 69.7 | 118.0 | 35.8 |
| `scalar_advection` (ms/step) | 196.7 | 140.9 | 171.1 | 94.6 |
| whole step (ms) | 462.4 | 344.1 | 388.8 | 193.4 |

The canopy stresses are anisotropic, with ``II ≈ 0.07``; in the neutral
shear equilibrium of the LRR-IP model ``II = 0.091``. The vertical
component dominates: ``τ_{33}/τ_{11} = 1.35`` at noon, where the canopy is
convective and buoyancy produces ``τ_{33}``.

The 02:00 value is the same, and that is physical in this case, not a
defect. The canopy at night is still unstably stratified: the city
surfaces stay warmer than the air, and a 5-minute diagnostic run from
02:00 has θ falling by 0.22 K over the lowest 75 m. Above the roofs, where
the night air is stable, ``τ_{33}/τ_{11} = 0.43`` (level 24).

The realizability corrections come from the component-wise transport. In
the last step, 127 cells reached the source update already unrealizable
at noon and 126 were corrected; at night the numbers were 632 and 616, or
0.1–0.5 % of the 131 072 cells. Over 30 min the corrections added
2.1e3 m⁵ s⁻² at noon and 1.5e3 m⁵ s⁻² at night. That is 0.09 % and
0.16 % of the final SGS energy ``Σ e V`` (2.38e6 and 8.85e5 m⁵ s⁻²).

The closure is a cost option. Its `advance_sgs!` adds 72–82 ms per step
to `step!`, from a 6×6 solve in every cell. The five extra tracers add
56–76 ms to `scalar_advection`. The whole step is 1.3× the `:tke` cost at
noon and 2.0× at night. These were sequential runs on a heavily loaded
machine, so the ratios are rough.

### A-priori metrics, outputs and a 1 h comparison (P56-e, #192)

No new physics. Both prognostic providers now add an a-priori line to
their `report`. It comes from `sgs_test_filter_metrics` in
`couplings/sgs.jl`, which compares the closure's subfilter stress over the
canopy fluid cells with the resolved (Germano) stress at the test-filter
scale. That stress is ``L_{ij} = \widehat{\bar u_i\bar u_j} − \hat{\bar u}_i\hat{\bar u}_j``,
computed by `subfilter_stress!` of the LES velocity under the 3-cell
masked box filter (P56-c). The line reports:

* ``k_{test} = ½⟨L_{kk}⟩``;
* the ratio ``e/k_{test}``;
* the off-diagonal correlation ``ρ(τ, L)``;
* the median II of ``L`` and of the model stress;
* the fraction of cells whose model stress lies outside the Lumley
  triangle.

For `:tke` the model stress is the implied ``\tfrac23 e δ − 2ν_t\bar S``;
for `:reynolds_stress` it is ``τ``.

There are new `output_fields`. `:tke` writes `II_sgs`; its ``e`` is
already the tracer `e`. `:reynolds_stress` writes `e` (``½ τ_{kk}``) and
`II_sgs`. The smoke checks now also require ``k_{test} > 0``, a finite
ratio, TKE II ≥ 0, and RSM II in ``[0, 2/3]``.

**Measured.** 1 h runs of the summer flagship (64 × 64 × 32, adaptive
step) from 02:00 and from 12:00. The four runs ran at the same time on a
shared machine, so the costs compare with each other.

| | 02:00 `:tke` | 02:00 `:reynolds_stress` | 12:00 `:tke` | 12:00 `:reynolds_stress` |
| :-- | --: | --: | --: | --: |
| canopy mean ``e`` (m² s⁻²) | 0.001 | 0.0014 | 0.012 | 0.016 |
| canopy mean ``ν_t`` (m² s⁻¹) | 0.008 | 0.011 | 0.036 | — |
| ``k_{test}`` (m² s⁻²) | 0.0043 | 0.0042 | 0.0203 | 0.0202 |
| ``e/k_{test}`` | 0.18 | 0.33 | 0.58 | 0.78 |
| ``ρ(τ, L)`` | 0.07 | 0.25 | 0.21 | 0.13 |
| median II: ``L`` / model | 0.294 / 0.008 | 0.295 / 0.079 | 0.252 / 0.006 | 0.255 / 0.077 |
| model cells outside the Lumley triangle | 10 % | 0 % | 2 % | 0 % |
| realizability corrections (1 h) | — | 917 442 | — | 361 796 |
| steps | 1397 | 1379 | 2097 | 2094 |
| wall time (s) | 261 | 510 | 382 | 748 |
| `step!` (ms/step) | 33.1 | 112.5 | 32.4 | 108.9 |
| `scalar_advection` (ms/step) | 89.3 | 157.5 | 89.2 | 159.0 |

The canopy ``ν_t`` of the noon Reynolds-stress run was not recorded; the
night value is the one the issue asks for. The `:tke` values are the
report's three-decimal ones.

* Under both closures the resolved stress at the test filter is far more
  anisotropic (median II ≈ 0.25–0.30) than either closure's stress. The
  Reynolds-stress closure is ten times closer (0.08 against 0.006–0.008).
  In 2–10 % of the canopy cells the TKE closure's implied stress is not
  realizable.
* At night the Reynolds-stress stress correlates better with ``L``
  (0.25 against 0.07). At noon TKE's does (0.21 against 0.13).
* The Reynolds-stress closure costs 2.0× the whole step. It also makes
  ~10⁶ realizability corrections per hour at night.

**Recommendation for the maintainer.** Keep `:tke` as the default. The
Reynolds-stress closure wins clearly only where its physics is
structural. It reproduces the stress anisotropy in stratified decay
(``⟨τ_{33}⟩/⟨τ_{11}⟩`` 0.76 against DNS 0.70; TKE 1.00) and has the
smallest KE error under rotation. It does not win on resolved KE in
isotropic decay, and its heat flux is not closer to the DNS (see the
[validation](../theory/sgs_tke.md)). At 2× the cost that is not a clear
win, so this PR does not change the default.

### Relaminarisation of the stable layer (#195)

P59-e extends the same addon once more (no new module, no list change):
`relam_fraction` — per level, and `relam_frac` over the whole stable
layer, the fraction of the fluid cells with ``Ri_g > 1/4`` that are
relaminarised, i.e. γ-laminar **and** carry ``ν_t/ν < 10^{-2}``
(`Transition.relaminarised_fraction`; ``ν_t`` the larger of the active
closure's ``ν_h`` and ``ν_v``, `Operators.sgs_fields`; ``ν`` = `cfg.ν`).
The laminar mask uses the volume normalisation (option
`relam_normalization = :volume`): a quiet, shear-free free stream is
laminar against the domain-mean ``|ω|``, while the plane-normalised
indicator, being scale-free, marks its weak disturbances turbulent. The
log line adds the percentage and the number of stable cells. For a
laminar-consistent closure (`Operators.laminar_consistent`: `:sigma`)
`smoke_check` asserts that ``ν_t ≡ 0`` (bitwise) in the parallel flow of
the current plane-mean wind on the Berlin mesh (a `deepcopy` of the
closure on a scratch core, so the case's closure is untouched). The
issue asked for ``ν_t = 0`` "in laminar cells" of the running flow; the
γ-laminar cells of the smoke run are not exactly laminar (weakly 3-D),
so σ's ``ν_t`` there is small but not zero (12 steps: max 0.117 ν, mean
0.007 ν over the 251 plane-γ laminar cells, the same as Smagorinsky's
0.086 ν / 0.010 ν) — the exact statement is the parallel-flow one. AMD
and Vreman (`:amd`, `:vreman`, P55-d) do not declare
`laminar_consistent` yet; their own `smoke_check` asserts the
laminar-reference switch-off.

**Measured** — `:winter` from 05:00 to 06:00 (1800 steps of 2 s, before
the 08:15 sunrise), 64 × 64 × 32: the lowest 50 m are convective (the
warm city under 271 K air: ``N^2 = -7 × 10^{-4}`` s⁻² at 9 m) and an
elevated stable layer lies above (supercritical fraction 0.48–0.69
between 60 and 140 m). Of the 49 056 stable fluid cells 4.0 % are
relaminarised with the default Smagorinsky closure and 4.1 % (48 750
cells) with `:sigma` — mostly at the ground level (0.42) and in the
quiet top levels inside the sponge (0.61 and 0.91 at z = 184 and 197 m);
the stable layer between them is γ-turbulent. The background viscosity
``ν = 2`` m² s⁻¹ dominates the result: with ``ν = 0.2`` m² s⁻¹ (not the
default; same hour) the stable cells drop to 15 392 and only 0.1 % of
them are relaminarised (grid Reynolds number 28.7 against 3.8 in the
canopy). Cost of the added evaluation: one more `laminar_mask!` per
output record (the addon measured 34.6 ms per call with Smagorinsky,
43.2 ms with `:sigma`, on the flagship grid under a shared load).

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
   #131).
3. Dry deposition of NO, NO2, O3 and HNO3 from `k_air` with
   ``v_d = 1/(r_a + r_b + r_c)``.
4. Rain scavenging of HNO3 by `qr`.

In `:surface_update`, on the surface cadence, `r_a = 1/(C_H U)` and `u*`
are recomputed per column. They come from
`SurfaceExchange.solve_surface_layer` between the skin `bus.Ts` and the
first air cell (cell-centre wind, θ, qv at `z = Δz/2`), with
`z0h = z0m/10` and the per-class `z0m` that #129 gives its surfaces.
Since #129 (PR #158) the bus carries u* (`bus.u_star`) and every
column's surface carries its roughness, but this coupling still runs
its own surface-layer solve; reading them from the bus is open.

`physics[:chemistry] = :off` (provider of `NoChem`) removes the species.
It is not the default. The smoke matrix also runs a `night titration`
variant (start 00:30, J = OH = 0).

### Measured (flagship grid 64 × 64 × 32, 20 June, 12:00–12:30)

Remeasured on master (`915bd49`: reference state, MOST / soil surfaces
of #129, adaptive step of #166), `CaseConfig(:alexanderplatz; nsteps =
900, start_h = 12)` with the default couplings, with and without
`Ts0 = 300.0`, run to 1800 s of simulated time (962 / 1079 adaptive
steps; at #135 and in the first remeasurement 900 steps of 2 s). Street / roof /
park are surface classes 1 / 2 / 3; r_a, u* and v_d are medians over
the class; fluxes and emissions are means over the ledger's elapsed time.

| quantity | default (`Ts0` = 285 K) | warm surfaces (`Ts0` = 300 K) |
| :-- | --: | --: |
| Leighton ratio φ, median over air cells with NO, O3 > 0.01 ppb, at 12:30 | 1.0013 | 1.0013 |
| φ / φ_ext (peroxy-radical steady state), median | 0.9964 | 0.9963 |
| r_a street / roof / park, median (s m⁻¹) | 89.5 / 5.6e6 / 5.76e5 | 89.6 / 88.4 / 8.48e5 |
| u* street / park, median (m s⁻¹) | 0.094 / 0.0016 | 0.093 / 0.0013 |
| v_d O3 park / street (mm s⁻¹) | 0.002 / 0.464 | 0.001 / 0.464 |
| v_d NO2, HNO3 street (mm s⁻¹) | 1.53, 5.78 | 1.52, 5.74 |
| mean dry deposition flux O3 / NO2 / HNO3 (nmol m⁻² s⁻¹) | 0.316 / 0.320 / 0.063 | 0.507 / 0.382 / 0.093 |
| mean emission NO / NO2 / isoprene (nmol m⁻² s⁻¹) | 17.84 / 1.98 / 0.504 | same |
| surface-layer NO / NO2 / O3 at 12:30 (ppb) | 7.74 / 10.66 / 35.55 | 7.50 / 10.45 / 35.55 |

The default flagship starts its surfaces (the soil columns and slabs of
#129) 8 K below the air (`Ts0` = 285 K). At step 0 every class is stably
stratified, r_a 6.2e3–7.9e3 s m⁻¹ and u* ≈ 0.015 m s⁻¹, short of the
clamp. The street soil skins heat within minutes, and by 12:30 the
streets are unstable (r_a 89 s m⁻¹, v_d O3 0.46 mm s⁻¹): the domain-mean
O3 deposition is 0.32 nmol m⁻² s⁻¹, 17 × its value with the pre-#129
slabs, when r_a reached 1e5–1e9 s m⁻¹ and dry deposition was
effectively off. Roofs and park stay decoupled (r_a 5.6e6 and 5.8e5
s m⁻¹; mean Ts at 12:30 291.2 K on roofs and 291.05 K in the park, under
first-cell air of about 294 K; streets 302.8 K). Warm surfaces do not
change that for the park: its skin cools from 300 to about 291 K within
the half hour and decouples (r_a 1.0e3–1.05e9 s m⁻¹), so the PR #162
expectation that the default case would look like the right-hand column
holds for the streets but not for the park. The periodic domain has no
inflow of background air, so emitted NOx accumulates: surface-layer NO
rises from 1.68 to 7.74 ppb in 30 min.

Wet deposition, measured in the smoke window with `qr = 1e-4` seeded in
levels 2–4: the HNO3 removed per step equals ``s[\mathrm{HNO_3}]\,(Σ q_r Δz)\,Δt``
to 0.1 %. The rain sediments out in about four steps.

**Cost.** On the flagship grid on master `915bd49` (600 s from 12:00,
with and without `physics = Dict(:chemistry => :off)`, run one after the
other on a loaded 24-core machine, first-call compilation included; both
runs take the same 230 adaptive steps, mean dt 2.61 s, since the
chemistry does not feed back on the flow) the step goes from 233.7 to
289.1 ms (wall time over steps, i.e. 89.6 → 110.8 ms per simulated
second; 190.5 → 248.8 ms summed over the stage timings), so the
chemistry adds about 24–31 %. The WENO-Z5 transport of the five species
adds 50.0 ms (10.0 ms per tracer: `scalar_advection` 33.7 → 83.7 ms per
step); the `aq_chem` stage costs 12.1 ms (92 ns per cell) including
emissions and deposition; the r_a refresh 3.3 ms averaged over steps.
The absolute numbers depend on the load: a first pair run alongside
five other measurements gave 325.7 → 509.0 ms per step. (Remeasured at
`2895fed` with 300 steps of 2 s: 135.5 → 169.2 ms. At #135, before
#129, the step went from 43.1 to 101.1 ms.) The chemistry stays on by default because
the flagship is meant to exercise all the physics, and `:off` exists for
runs that do not need air quality.

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
