# Roadmap

Aeolus is under active development. This roadmap divides the work into
**20 phases of roughly equal size** — each phase is a 2–4 week chunk of
focused work that ends with either a validation, a benchmark, or a
concrete new user-visible capability. Phases 1–5 are already shipped;
Phases 6–20 are planned.

The point of equal-sized phases is that you can stop at any phase
boundary and have a coherent, useful system.

## Shipped

### Phase 1 — Foundations

  * `AbstractMesh`, `RegularGrid`, `Field{Location}`, boundary
    conditions, halo fill.
  * FV first-derivative and Laplacian stencils.
  * Shu-Osher SSP-RK3 stepper (generic, scalar-oriented).
  * 2D passive scalar advection example + tests.

### Phase 2 — Dry incompressible core

  * Face-centered `u, v, w`; staggered Arakawa-C operators.
  * Staggered momentum advection (2nd-order centered flux form).
  * `AbstractPressureSolver` + `FFTPoissonSolver` for triply-periodic BCs.
  * `BoussinesqSimulation` with fractional-step / Euler-projection driver.
  * Validated against 3D Taylor-Green vortex (analytic decay).

### Phase 3 — SGS + Boussinesq physics

  * Smagorinsky-Lilly SGS closure (full strain-rate tensor at cell
    centers; eddy-viscosity conservative flux divergence).
  * Buoyancy on `w`-momentum; f-plane Coriolis on horizontal momenta.
  * Prognostic potential temperature θ with staggered scalar advection.
  * Buoyant-convection coupled example.

### Phase 4 — Documentation & benchmarks

  * Full Documenter.jl site: theory (7 pages), design (4), API (7),
    examples (5), benchmarks (1).
  * Extensive docstrings on every exported symbol.
  * BenchmarkTools.jl harness for kernels, Poisson scaling, end-to-end
    `step!`.

### Phase 5 — Channel-BC pressure solver

  * `CosineFFTPoissonSolver`: periodic in `x, y`, Neumann in `z` via
    DCT-II along `z` + rfft in `(x, y)`.
  * Manufactured-Poisson error at machine precision.
  * Enabling piece for wall-bounded flows (channels, DCBL, engineering
    CFD).

## Planned

### Phase 6 — Wall BCs on velocity + simulation integration *(shipped)*

  * `ImpermeableWall` BC type: `w = 0` enforced at wall faces every
    halo fill, anti-symmetric halo extension.
  * Free-slip mirror for `u, v` in `z`; Neumann mirror for `p, θ`.
  * `BoussinesqSimulation` auto-selects `CosineFFTPoissonSolver` when
    the vertical BCs are non-periodic.
  * Tests: wall zeroing + antisymmetric halo, free-slip mirror, and a
    full simulation with wall BCs returning `∇·u ≈ 10⁻¹⁴`.

### Phase 7 — Surface fluxes for scalars + sponge damping *(shipped)*

  * `add_surface_flux!` kernel injects prescribed diffusive fluxes at
    the bottom / top walls directly into the scalar tendency.
  * `add_sponge!` kernel: Rayleigh damping toward a reference value in
    a smooth `sin²`-tapered top-of-domain region.
  * Both plumbed into `BoussinesqSimulation` via `θ_flux_bottom`,
    `θ_flux_top`, `sponge_α_max`, `sponge_z_start` kwargs. Zero → off.
  * Tests: surface flux warms bottom + cools top; sponge damps top-zone
    velocity while leaving the bottom untouched.

### Phase 8 — DCBL example + diagnostics *(shipped)*

  * `examples/dry_convective_bl.jl`: Sullivan & Patton-flavoured DCBL
    setup (walls + surface flux + capping inversion + sponge + SGS).
  * `horizontal_mean`, `horizontal_variance`, `resolved_flux_wθ`
    diagnostics on `BoussinesqSimulation`.
  * Regression test: coupled system runs stably with `max ∇·u < 10⁻¹²`,
    `KE > 0`, and `νₜ > 0` after 20 steps.
  * Reference-quality validation against Sullivan / Moeng data (requires
    ≥ 64³ grid + longer run + spectrum diagnostics) is deferred as a
    stand-alone future benchmark task.

### Phase 9 — I/O infrastructure *(shipped)*

  * Julia-native `Serialization` snapshots: `write_snapshot`,
    `read_snapshot`, `checkpoint!`, `restart!`.
  * `TOML`-based namelists via `load_config`.
  * NetCDF binding deferred as an optional add-on (HDF5 stack conflicts
    with system MPI on the dev machine; Julia-native snapshots cover
    restart + inspection now).

### Phase 10 — Numerical accuracy upgrades *(shipped)*

  * `advect_weno5!`: WENO-5 upwind reconstruction (Jiang & Shu 1996).
    Non-oscillatory near sharp gradients. Requires halo ≥ 3.
  * `step_rk3!`: Chorin-projected Shu-Osher SSP-RK3 with a Poisson
    solve at each substage. Third-order-accurate in time for
    divergence-free flows.
  * Tests: WENO-5 conserves mass on divergence-free flow, errors on
    small halos; step_rk3! keeps divergence at machine precision.

### Phase 11 — Moist thermodynamics *(shipped)*

  * Optional `qv` (water vapor mixing ratio) prognostic on
    `BoussinesqSimulation` (kwarg `moisture = true`).
  * `qv` transported by staggered centered advection alongside θ.
  * `add_buoyancy!` gains a 5-arg method with virtual potential
    temperature `θᵥ = θ (1 + 0.61 qv)`.
  * Snapshots include `qv`; restart handles both dry and moist runs.
  * Saturation adjustment / condensation is Phase 12 (microphysics).

### Phase 12 — Bulk cloud microphysics *(shipped)*

  * `Microphysics.apply_kessler_step!` — Kessler autoconversion +
    accretion + 1st-order upwind rain fallout.
  * Full integration into `BoussinesqSimulation` and evaporation of
    falling rain / detailed saturation adjustment planned as
    refinements (need θ-p coupling).
  * BOMEX validation deferred as a stand-alone benchmark.

### Phase 13 — Non-uniform grid + immersed boundary *(IBM shipped)*

  * `IBM.SolidMask` + `mark_solid!` + `apply_solid_mask!`: direct-
    forcing IBM zeroes velocity on faces of solid cells.
  * Stretched vertical grid `dz(k)` deferred as a broader refactor
    (requires k-indexed spacing throughout the operator kernels;
    single follow-up phase in itself).
  * Validation (flow past a cube, backward-facing step) deferred as
    stand-alone benchmark task.

### Phase 14 — Iterative pressure solver *(shipped)*

  * `CGPoissonSolver`: matrix-free preconditioned conjugate gradient
    on the 7-point discrete Laplacian; supports periodic and Neumann
    BCs (Dirichlet needs a separate kernel).
  * Preconditioner: identity for now (matrix has small condition on
    the unit-cube grid); multigrid preconditioner planned as a later
    refinement.
  * Regression: matches `FFTPoissonSolver` on periodic manufactured mode
    to `< 10⁻⁸`.

### Phase 15 — MPI infrastructure *(scaffold shipped)*

  * `Parallel.AbstractPartition` type hierarchy: `SerialPartition`
    (default, no comm) and `PencilPartition(nx, ny)` (2-D layout stub).
  * `exchange_halos_distributed!` interface: no-op on serial, errors
    with a helpful message on `PencilPartition` until the MPI extension
    is loaded.
  * The actual MPI wiring is deferred to an `AeolusMPI` weak-dep
    extension (`ext/AeolusMPI.jl`) so Aeolus stays MPI-independent for
    serial users. `PencilFFTs.jl` integration follows the same pattern.

### Phase 16 — Nesting *(scaffold shipped)*

  * `NestSpec(parent, child)` records a nested `BoussinesqSimulation`
    pair with integer refinement + offset.
  * `apply_one_way_forcing!` copies parent boundary values into the
    child halo (refine = 1 direct copy today).
  * `apply_two_way_feedback!` copies child interior back into parent
    cells the child overlaps.
  * High-order interpolation, non-trivial refinement, and interior-
    domain-boundary treatment for turbulent flow deferred as follow-up.

### Phase 17 — GPU backend *(scaffold shipped)*

  * `Backends.AbstractBackend` selector; `CPUBackend` (default) and
    `GPUBackend(kind)` stub for `:cuda / :rocm / :oneapi / :metal`.
  * `allocate(backend, T, dims)` and `to_host(x)` interface.
  * Full `KernelAbstractions.jl` port + GPU FFT plans deferred to a
    weak-dep extension (`AeolusCUDA` / `AeolusAMDGPU`) — needs a GPU
    to develop against.

### Phase 18 — Spectral element backend *(mesh scaffold shipped)*

  * `SpectralElementMesh <: AbstractMesh` — brick-of-elements layout
    with polynomial order `N` and per-element GLL nodes.
  * `gll_nodes(N)` / `gll_weights(N)` — Newton iteration on Lobatto
    polynomials for reference-element quadrature.
  * `AbstractMesh` interface (`totalsize`, `interior`,
    `xnodes/ynodes/znodes`) implemented — proves the two-backend
    promise from the design page.
  * Deferred: tensor-product differentiation operators, element-local
    mass matrix, matrix-free CG for pressure on SEM, and
    `BoussinesqSimulation` awareness of the new storage layout. Each is
    a substantial follow-up in its own right.

### Phase 19 — Radiation + land surface *(shipped)*

  * `apply_radiative_cooling!(tθ, rate)` — scalar or profile forcing
    on the potential-temperature tendency.
  * `SlabLand(T, C_s, albedo)` — single-layer soil-temperature
    reservoir with SW / LW balance and outgoing sensible flux.
    `step_land!` advances it under any explicit forcing.
  * Full `RRTMGP.jl` two-stream radiation and vegetation / canopy
    LSMs deferred as follow-up refinements; the interface here plumbs
    the essential physics.

### Phase 20 — Advanced physics & extensions *(scaffolds shipped)*

  * **Chemistry** — `apply_linear_chemistry!(tc, c, τ; source)`: first-
    order chemical relaxation on a passive tracer with optional
    spatially-varying source. Full KPP integrator deferred.
  * **Particles** — `ParticleSet` + `add_particles!` +
    `advance_particles!`: Lagrangian tracker with trilinear velocity
    interpolation and periodic wrap. Inertial particles and
    back-reaction deferred.
  * **ML SGS** — `MLSGS(fn)` + `apply_mlsgs!(sim, mlsgs)`: pluggable
    callable that writes `sim.νt` from state. Any Julia framework
    (`Lux.jl`, `Flux.jl`) supplies the callable; the existing
    eddy-viscosity kernels consume it unchanged.
  * **Data assimilation** — `Observation` records and
    `assimilate_nudging!` Newtonian relaxation seam. External DA
    drivers (4D-Var, EnKF) can be layered on the same observation
    interface.

## Post-Phase-20 physics-refinement plan

The 20 phases wired up the scaffolding for every major LES capability.
The flagship Alexanderplatz simulation surfaced a set of physical
mechanisms that are still absent. In visibility-through-the-animation
order:

### Radiation / lighting
  * **R1. Direct-beam building shadows** — per-cell ray-marched shadow
    mask; shaded cells receive only diffuse SW.  *(next)*
  * **R2. Sky-view factor** — hemispheric occlusion per cell; scales
    diffuse SW and outgoing LW.
  * **R3. Wall-face temperatures** — per-façade `SlabLand` on vertical
    building walls, replacing the 2-D "wall class" proxy.
  * **R4. Longwave inter-reflection** — building faces + ground
    exchange LW in a 1–2 pass radiosity.
  * **R5. Cloud radiative effect + cloud shadows.**
  * **R6. Window / glass materials.**

### Land surface
  * **L1. Latent heat / evapotranspiration** — the real reason parks
    are cool. Bulk formula on wet surfaces.
  * **L2. Vegetation drag + tree shade.**
  * **L3. Multi-layer soil.**
  * **L4. Anthropogenic heat maps** (AC, traffic).
  * **L5. Snow / ice / dew surface state.**

### Surface flux + turbulence
  * **F1. MOST wall model** — stability-corrected surface flux.
  * **F2. Per-class roughness `z₀`.**
  * **F3. Anisotropic / dynamic Smagorinsky.**

### Moisture & clouds
  * **M1. Full saturation adjustment** with Tetens curve, replacing
    the current qv-clamp trick.
  * **M2. Multi-moment microphysics (Morrison / SBM).**
  * **M3. Ice / mixed-phase.**
  * **M4. Fog / dew condensation.**

### Air-quality
  * **A1. KPP-generated multi-species chemistry.**
  * **A2. Wet + dry deposition per surface class.**
  * **A3. Biogenic + diurnal traffic emissions.**
  * **A4. Aerosol prognostics (PM₂.₅ / PM₁₀).**

### Numerical infrastructure
  * **N1. True LES viscosity regime** (`ν ≈ 10⁻⁵`) — cut-cell IBM +
    fine near-wall grid.
  * **N2. Stretched vertical grid** (`dz(k)`).
  * **N3. Adaptive time stepping.**
  * **N4. Wire WENO-5 into `BoussinesqSimulation.step!`.**
  * **N5. Non-periodic horizontal BCs.**

### Regional / large-scale coupling
  * **G1. Nested mesoscale forcing.**
  * **G2. Sea / lake coupling.**
  * **G3. Terrain.**

Each bullet is a shippable 1–3 week task; together they form ~1–2
years of physics work beyond the current scaffold. The plan is to
knock them off in visibility order, starting with **R1 (shadows)**.

## Beyond the refinement plan

The 27 items above are complete. The next decade of development is
planned in **[Long-range roadmap (Eras II–XI)](longterm_roadmap.md)** —
80 further phases organised into 10 eras, ordered physics-first, with
the standing rule that every phase ships a *validated production
capability* rather than a scaffold.

## Timeline realism

Each phase is 2–4 weeks of focused work for one person. Two rough
milestones:

  * **End of Phase 8** (~6 months of work from now): a validated
    atmospheric-boundary-layer LES for dry convective cases. Directly
    comparable to PALM at reduced feature set.
  * **End of Phase 12** (~1 year of work from now): moist ABL with
    shallow clouds, I/O ready for production runs.
  * **End of Phase 17** (~1.5–2 years): MPI + GPU parallelization,
    ready for production HPC.
  * **Phase 18+ enters multi-year territory** and starts to require
    domain expertise across meteorology, physical oceanography, and
    engineering CFD.

**Validation against published benchmarks is the gating criterion at
every phase — not code volume.**
