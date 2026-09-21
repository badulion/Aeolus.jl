# Aeolus.jl

A Julia-native large-eddy simulation framework. Work-in-progress.

## Goals

A general-purpose LES/DNS framework for atmospheric, oceanic, and
engineering boundary-layer flows. Single-node CPU first; MPI and GPU
come later. Inspired by PALM (finite-difference + FFT pressure) and
Oceananigans.jl (staggered finite volume, kernel abstractions).

## Design

Spatial discretization lives behind an `AbstractMesh` interface:

- **Phase 1–2**: finite-volume on a staggered Cartesian grid
  (`RegularGrid`). This is what PALM, Oceananigans, WRF-LES use.
- **Future**: spectral-element on unstructured meshes for engineering
  flows with complex geometry. New backend = new `AbstractMesh` subtype
  + new methods on the internal operator kernels. No user-code changes.

The pressure solver is pluggable in the same way: FFT-direct for
periodic, cosine-FFT for channel-in-z (shipped in Phase 5), conjugate
gradient / multigrid for arbitrary BCs (Phase 14).

## Roadmap — 20 phases

Each phase is a 2–4 week chunk of focused work that ends with a
validation, a benchmark, or a concrete new capability. See
`docs/src/design/roadmap.md` for the full description of each phase.

### Shipped

- [x] **Phase 1** — Foundations (grid, fields, BCs, operators, RK3)
- [x] **Phase 2** — Dry incompressible core (staggered momentum,
      FFT-Poisson, Euler-projection, TGV validation)
- [x] **Phase 3** — SGS + Boussinesq physics (Smagorinsky-Lilly,
      buoyancy, Coriolis, θ transport, convection example)
- [x] **Phase 4** — Documentation & benchmarks (Documenter site,
      BenchmarkTools harness)
- [x] **Phase 5** — Channel-BC pressure solver (`CosineFFTPoissonSolver`
      for periodic-xy + Neumann-z)

### Planned

- [x] **Phase 6** — Wall BCs on velocity + simulation integration
      (`ImpermeableWall` + auto-select cosine-FFT solver)
- [x] **Phase 7** — Surface fluxes for scalars + sponge damping layer
- [x] **Phase 8** — DCBL example + diagnostics (reference-quality
      validation deferred as separate benchmark task)
- [x] **Phase 9** — I/O infrastructure (Serialization snapshots +
      restart + TOML namelists; NetCDF deferred as optional add-on)
- [x] **Phase 10** — Numerical upgrades (WENO-5, projected RK3)
- [x] **Phase 11** — Moist thermodynamics (qv prognostic + virtual θ;
      saturation adjustment moves to Phase 12)
- [x] **Phase 12** — Bulk cloud microphysics (Kessler kernel; BOMEX
      validation deferred)
- [x] **Phase 13** — Immersed boundary method (direct-forcing);
      stretched grid deferred
- [x] **Phase 14** — Iterative CG pressure solver
- [x] **Phase 15** — MPI infrastructure scaffolding (interface types;
      actual MPI wiring lives in a future weak-dep extension)
- [x] **Phase 16** — Nesting scaffold (`NestSpec`, one-way forcing +
      two-way feedback; higher-order interp deferred)
- [x] **Phase 17** — GPU backend scaffold (backend selector; full
      KernelAbstractions port deferred to weak-dep extension)
- [x] **Phase 18** — Spectral element mesh scaffold (`SpectralElementMesh`
      type + GLL nodes/weights; operator retrofit deferred)
- [x] **Phase 19** — Radiation (prescribed profile) + slab LSM
      (RRTMGP + full vegetation LSM deferred)
- [x] **Phase 20** — Advanced physics scaffolds (linear chemistry,
      Lagrangian particles, pluggable ML SGS, nudging-DA)

Rough milestones: **Phase 8** — dry ABL LES comparable to PALM (subset).
**Phase 12** — moist ABL with shallow clouds. **Phase 17** — MPI+GPU
production HPC. **Phase 18+** — multi-year territory needing domain
expertise across met/ocean/CFD.

## Post-Phase-20 refinement plan — physics gaps to close

The 20 phases wired up the *scaffolding* for every major LES capability
we intend to ship. The flagship urban simulation exposed a set of
physical mechanisms that are still absent — the current run captures
class-based thermal inertia contrasts but is missing several first-order
effects that shape a real urban heat island. In roughly the order they
would visibly change the animation:

### Radiation / lighting
- [x] **R1. Direct-beam building shadows.** Per-cell ray-marched
      shadow mask (`compute_shadow_mask!`). Wired into Alexanderplatz run.
- [x] **R2. Sky-view factor.** Hemispheric occlusion per cell
      (`compute_sky_view_factor`). Scales diffuse SW + outgoing LW.
- [x] **R3. Wall directional SW.** `wall_directional_factor` gives
      each wall cell a cos(sun·normal) SW factor. Full per-face
      `SlabLand` on vertical building walls remains a further refinement.
- [x] **R4. LW canyon trapping.** `lw_from_walls` refines the
      SVF-weighted LW blend with a temperature-dependent canyon term.
      Full 1-2 pass radiosity between building faces still to add.
- [x] **R5. Cloud shadows.** `apply_cloud_shadow!` attenuates direct
      SW by the column-integrated `ql` via Beer's law.
- [x] **R6. Window / glass materials.** `SURFACE_GLASS` class with
      high specular albedo + `apply_glass_sw_transmission!` removes
      the transmitted portion from the exterior surface energy budget.

### Land surface
- [x] **L1. Latent heat flux.** `apply_latent_flux!` bulk formula
      `LE = ρ Lᵥ Cₑ U β Δq` with class-dependent β (water=1, park=0.6,
      street=0.15, roof/wall=0.05). Feeds into `sim.qv` too.
- [x] **L2. Vegetation drag + tree shade.** `apply_vegetation_drag!`
      + `apply_tree_shade!`. **Note:** explicit-Euler drag is CFL-
      limited; enabled cd needs to satisfy `cd·LAI·|u|·dt < 1`.
      Currently disabled in the Alexanderplatz flagship run — needs
      implicit treatment before it can safely run at `dt = 2 s`.
- [x] **L3. Multi-layer soil.** `SoilGrid`/`step_soil!` — 3-layer
      conduction below each cell, deep BC pinned at 288 K.
- [x] **L4. Anthropogenic heat.** `anthropogenic_heat` +
      `add_anthropogenic_heat!` — traffic/AC W/m² with morning + evening
      rush-hour peaks, class-dependent baseline.
- [x] **L5. Snow / ice / dew state.** `SnowState`,
      `update_snow_state!`, `snow_modify_params!` for albedo/emis flip;
      `apply_dew_condensation!` for below-dew warming (disabled until
      it can be coupled with saturated-air feedback).

### Surface flux + turbulence
- [x] **F1. MOST wall model.** `apply_surface_flux_most!` uses Louis-
      1979 stability-corrected bulk formula with per-class `z₀`.
      Scaffold shipped; flagship run still uses the bulk-`Cₕ` version
      because MOST at the current Δt / dx is more delicate.
- [x] **F2. Per-class roughness `z₀`.** `roughness_z0(cls)` lookup
      (street 3 cm, roof 10 cm, park 15 cm, water 0.1 mm, wall 5 cm).
- [x] **F3. Dynamic Smagorinsky.** `compute_dynamic_smagorinsky_cs!`
      Germano-Lilly per-cell `Cₛ²` via a 3×3×3 test filter, clipped to
      `[0, 0.25]`. `compute_smagorinsky_νt_dynamic!` reads it.

### Moisture & clouds
- [x] **M1. Full saturation adjustment.** `saturation_adjustment!`
      Bolton-1980 coupled qv⇄ql via Tetens with latent-heat feedback.
- [x] **M2. Two-moment microphysics scaffold.** `TwoMomentState`
      (qv, ql, qr, qi, Nc, Nr) + `apply_two_moment_micro!` (Kessler
      auto/accretion coupled to number-conc + Marshall-Palmer sed).
- [x] **M3. Ice / mixed-phase scaffold.** `apply_ice_processes!` —
      homogeneous freezing < 235 K and melting > 273 K with latent-heat
      back-reaction.
- [x] **M4. Fog / dew.** `apply_fog_dew!` — first-layer condensation
      below the dew point. **Disabled in flagship** until coupled with
      full saturated-air feedback.

### Air-quality
- [x] **A1. Multi-species chemistry.** `ChemistryState` (NO/NO₂/O₃/VOC)
      + `apply_photochem_step!` with photolysis + coupling reactions.
      A proper KPP-generated mechanism replaces the toy 3-reaction
      scheme in a future item.
- [x] **A2. Wet + dry deposition.** `deposition_velocity` per species
      × surface-class table; `apply_dry_deposition!` at the first
      layer; `apply_wet_deposition!` scavenging by `qr`.
- [x] **A3. Emissions maps.** `emission_rate` + `add_emissions!` —
      bimodal traffic on street/wall for NO/NO₂; Gaussian biogenic on
      park cells for VOC.
- [x] **A4. Aerosol prognostics.** `AerosolState` (qₐ, Nₐ) +
      `apply_aerosol_step!` — semi-implicit coagulation + upwind
      sedimentation + PM dry deposition.

### Numerical infrastructure
- [x] **N1. Bulk-viscosity map (scaffold).** `bulk_viscosity_map`
      returns a per-cell ν so a future implicit-IBM solver can lower ν
      away from obstacles while keeping it high near walls. The full
      cut-cell / ghost-cell IBM refactor still to come.
- [x] **N2. Stretched vertical grid metric (scaffold).**
      `ZStretchMap` returns a non-uniform `dz(k)` table. Full operator
      retrofit to use it stays a follow-up.
- [x] **N3. Adaptive Δt.** `cfl_time_step` computes a CFL/viscous-
      bounded dt from the current velocity field.
- [x] **N4. WENO-5 tendency hook.** `add_weno_scalar_tendency!` wraps
      `advect_weno5!` so users can add WENO advection to a scalar
      tendency inside a `body_force!` callback.
- [x] **N5. Lateral damper (non-periodic BC proxy).**
      `apply_lateral_damper!` Rayleigh-damps toward an ambient wind
      near the horizontal boundaries. Full characteristic inflow /
      outflow BCs remain a follow-up.

### Regional / large-scale coupling
- [x] **G1. Mesoscale nudging (scaffold).** `MesoForcing` bundle +
      `apply_mesoscale_nudging!` relaxes horizontal means toward
      externally supplied θ / qv / U / V profiles.
- [x] **G2. Sea / lake coupling (scaffold).** `WaterMask` +
      `apply_lake_exchange!` pins water-cell Ts and injects sea-breeze
      sensible flux.
- [x] **G3. Terrain (scaffold).** `add_terrain_to_mask!` adds a 2-D
      height map to the `SolidMask`. Simple staircase terrain; smooth
      terrain-following coordinates remain a follow-up.

Each bullet is a shippable 1–3 week task in its own right. All 27 are
now complete — either as full implementations (radiation, latent heat,
anthropogenic heat, soil, chemistry, aerosol) or as validated-API
scaffolds that the long-range roadmap below replaces with production
versions.

## Long-range roadmap — Eras II–XI (the next decade)

Era I (Phases 1–20 plus the 27 refinement items above) built the LES
core and wired every interface. The next decade replaces every
remaining approximation with a validated, production-quality
implementation. **Guiding rule from here on: no scaffolds — a phase is
only done when it is validated against reference data and stable at
production resolution.**

See **[`docs/src/design/longterm_roadmap.md`](docs/src/design/longterm_roadmap.md)**
for the full plan: 10 eras, 80 phases (21–100), each 2–6 weeks with an
explicit validation target.

| Era | Phases | Theme |
| :-- | :----- | :---- |
| **II** | 21–31 | **Numerical foundations for fidelity** — cut-cell IBM, ghost-cell walls, adaptive Δt, terrain-following coordinates, IMEX integration, WENO-Z/TENO, anelastic dynamics, energy-conserving operators, multigrid, production nesting, AMR |
| **III** | 32–38 | **High-fidelity radiation** — RRTMGP gas optics, two-stream with cloud optics, 3-D Monte-Carlo RT, full urban radiosity, spectral BRDF albedo, aerosol radiative feedback, NREL solar position |
| **IV** | 39–45 | **High-fidelity clouds** — exact moist thermodynamics, P3 microphysics, superdroplet Lagrangian reference, aerosol-cloud interaction, ice nucleation, fog/visibility, turbulence-microphysics coupling |
| **V** | 46–53 | **High-fidelity land & urban** — Richards-equation soil, Farquhar photosynthesis, resolved canopy, production urban-canopy model, building energy coupling, multi-layer snow, wave-coupled water, surface heterogeneity |
| **VI** | 54–59 | **High-fidelity turbulence** — scale-dependent dynamic SGS, anisotropic/tensorial closures, prognostic SGS stress, high-Re wall models, stable-boundary-layer regime, laminar-turbulent transition |
| **VII** | 60–66 | **High-fidelity chemistry** — KPP-generated mechanism with analytic Jacobian, online photolysis, aqueous/heterogeneous chemistry, modal+sectional aerosol, SOA, real emission inventories, deposition & pollen |
| **VIII** | 67–73 | **Real-world data & validation** — ERA5 driver, mesoscale nesting with synthetic-turbulence inflow, production OSM + DEM ingestion, observation operators, the canonical benchmark library (dry + moist), reproducibility & provenance |
| **IX** | 74–79 | **Data assimilation & inverse problems** — Enzyme adjoint, 4D-Var, LETKF, hybrid & coupled DA, parameter and source inversion, calibrate-emulate-sample UQ |
| **X** | 80–87 | **Performance & scale** *(deliberately after physics)* — production MPI to 10⁴ cores, full GPU port, multi-GPU, mixed precision, performance engineering, async/in-situ I/O, SEM completion, DG option |
| **XI** | 88–100 | **ML, applications & ecosystem** — neural SGS & emulators, differentiable-physics design optimisation, foundation-model coupling, wildfire, wind energy, urban climate services, air-quality nowcasting, emergency response, contrails, Earth-system coupling, companion packages, community & papers |

**Milestones**: numerically sound real-Reynolds urban LES after Era II;
research-standard cloud physics after Era IV; validated urban/land model
after Era V; real chemistry-transport model after Era VII; HPC
production after Era X.

**Honest scope**: this is a decade of sustained work assuming domain
expertise across meteorology, cloud physics, chemistry, turbulence and
numerical analysis, plus reference data and HPC access. A solo
contributor should plan on reaching the end of Era IV in that window,
not Era XI. The *ordering* is the load-bearing part: fidelity before
speed, because a fast model of the wrong physics is worthless and
optimising early freezes interfaces that still need to change.

## Quick start

```julia
using Pkg; Pkg.activate("."); Pkg.instantiate()
using Aeolus
include("examples/taylor_green_vortex.jl")
```

Run tests:

```julia
Pkg.test()
```

## Directory layout

```
src/
  Aeolus.jl              top-level module
  Grids.jl               AbstractMesh, RegularGrid, Location types
  Fields.jl              Field type (mesh-generic)
  BoundaryConditions.jl  Periodic, NoFlux, Value, Flux BCs; fill_halos!
  Operators.jl           public API — dispatches on mesh type
  FiniteVolume.jl        FV kernels for RegularGrid
  TimeSteppers.jl        Shu–Osher SSP-RK3
  PressureSolvers.jl     AbstractPressureSolver, FFT + CosineFFT solvers
  Simulations.jl         BoussinesqSimulation + fractional-step driver
examples/
  scalar_advection_2d.jl
  poisson_manufactured.jl
  poisson_neumann_manufactured.jl
  taylor_green_vortex.jl
  buoyant_convection.jl
test/
  runtests.jl            39 unit tests
benchmarks/
  bench_operators.jl
  bench_poisson.jl
  bench_step.jl
docs/
  src/                   Documenter source (~20 pages)
  make.jl                docs build
```
