# Roadmap

The plan of Aeolus, shipped and open, on one page: a status checklist,
what each milestone makes Aeolus, then one section per milestone in phase
order. A shipped section keeps its record (what was built, the numbers it
was measured at, the limitations carried forward); an open section is the
plan. Milestones are named by content and match the GitHub milestones;
issues for open phases are generated from this page
(`.github/scripts/generate_roadmap_issues.py`).

## Guiding principle for everything below

> **No scaffolds. Every phase ships a production-quality, validated
> capability.**

The LES core deliberately traded depth for breadth — it wired interfaces so
later work would have somewhere to plug in. That phase is over. From
Phase 21 onward, a phase is only "done" when:

1. The physics or numerics is implemented to the standard of the
   reference literature, not a toy approximation.
2. It is **validated** against either an analytic solution, a published
   benchmark with reference data, or observations.
3. It is **stable** in the flagship urban run and in the relevant
   canonical case at production resolution — not just at a tamed
   `Δt` or an inflated viscosity.
4. It has docstrings, a docs page explaining the theory, a regression
   test, and a benchmark entry.

Items that cannot meet that bar get **deferred**, not shipped as
scaffolds. Where Phases 1–20 left a scaffold, the corresponding phase
among Phases 21–45 has the job to *replace* it with the real thing.

## Scope

Roughly **5× the work completed so far**, organised as 10 milestones of 8–12
phases each — about **100 phases**, each 2–6 weeks. A small team could
do this in 4–5 years; a solo contributor should plan on 8–10.

The ordering is deliberately physics-first: Phases 21–45 (numerics,
radiation, clouds) close every approximation in the current physics before the project chases scale,
ML, or applications. A high-fidelity single-node model is more useful
than a fast wrong one.

Phases 101–112 are a **future direction** beyond that
decade: kilometre-scale regional and climate modelling. It is listed
so the architecture does not close the door on it, not as a commitment
within the 100-phase plan.

## Status

✅ shipped · ☐ open. A phase is checked when all of it has shipped; a milestone when all its phases have.
Partly delivered phases are split (67a/67b, 81a/81b, 84a/84b). The sections below
follow the same order.

- ✅ **LES core and physics scaffolds** *(Phases 1–20)*
    - ✅ Phase 1 — Foundations: mesh, fields, boundary conditions, finite-volume stencils, SSP-RK3
    - ✅ Phase 2 — Dry incompressible core: Arakawa-C operators, FFT pressure projection, Taylor–Green validation
    - ✅ Phase 3 — Smagorinsky–Lilly SGS, Boussinesq buoyancy and Coriolis
    - ✅ Phase 4 — Documentation site, docstrings and benchmarks
    - ✅ Phase 5 — Channel pressure solver (cosine FFT, Neumann in z)
    - ✅ Phase 6 — Wall boundary conditions on velocity, simulation integration
    - ✅ Phase 7 — Scalar surface fluxes and sponge damping
    - ✅ Phase 8 — Dry convective boundary layer example and diagnostics
    - ✅ Phase 9 — I/O: snapshots and TOML namelists (NetCDF came later)
    - ✅ Phase 10 — Numerical accuracy: WENO-5 advection and the SSP-RK3 projection step
    - ✅ Phase 11 — Moist thermodynamics (water vapour prognostic)
    - ✅ Phase 12 — Bulk cloud microphysics (Kessler); replaced by Phases 39–40
    - ✅ Phase 13 — Immersed boundary (solid mask); replaced by Phases 21–22
    - ✅ Phase 14 — Iterative pressure solver (preconditioned CG)
    - ✅ Phase 15 — MPI partition scaffold; Phase 80 builds the real one
    - ✅ Phase 16 — Nesting scaffold; replaced by Phase 30
    - ✅ Phase 17 — GPU backend selector scaffold; Phases 81a/81b
    - ✅ Phase 18 — Spectral-element mesh scaffold; Phase 86 completes it
    - ✅ Phase 19 — Radiation and land-surface scaffolds; replaced by Phases 32–38 and 46
    - ✅ Phase 20 — Chemistry, particle, ML-SGS and data-assimilation scaffolds; Phases 60–66, 74–79 and 88
- ✅ **Numerical foundations** *(Phases 21–31)*
    - ✅ Phase 21 — Cut-cell immersed boundary method
    - ✅ Phase 22 — Ghost-cell wall reconstruction
    - ✅ Phase 23 — Adaptive time stepping with substepping
    - ✅ Phase 24 — Stretched and terrain-following vertical coordinate
    - ✅ Phase 25 — Semi-implicit IMEX time integration
    - ✅ Phase 26 — WENO-Z and TENO advection for all prognostics
    - ✅ Phase 27 — Anelastic and pseudo-incompressible dynamics; plus the direct pressure solver (#309)
    - ✅ Phase 28 — Energy- and enstrophy-conserving discrete operators
    - ✅ Phase 29 — Multigrid pressure solver
    - ✅ Phase 30 — Grid nesting with proper two-way feedback; hardening and hierarchies shipped (#327–#329, #349); the nested Berlin epic #246 is open
    - ✅ Phase 31 — Adaptive mesh refinement
- ✅ **Radiation** *(Phases 32–38)*
    - ✅ Phase 32 — Correlated-k gas optics
    - ✅ Phase 33 — Two-stream solver with cloud optics
    - ✅ Phase 34 — 3-D Monte-Carlo radiative transfer
    - ✅ Phase 35 — Full urban radiosity
    - ✅ Phase 36 — Spectral surface albedo and BRDF
    - ✅ Phase 37 — Aerosol and gas radiative feedback
    - ✅ Phase 38 — Solar-position and cloud-shadow accuracy
- ✅ **Clouds and microphysics** *(Phases 39–45)*
    - ✅ Phase 39 — Production saturation adjustment and thermodynamics
    - ✅ Phase 40 — P3 microphysics
    - ✅ Phase 41 — Superdroplet Lagrangian microphysics; the audit follow-up #69 (Technical debt) is open
    - ✅ Phase 42 — Aerosol-cloud interaction
    - ✅ Phase 43 — Ice nucleation
    - ✅ Phase 44 — Precipitation, fog and visibility
    - ✅ Phase 45 — Turbulence-microphysics interaction
- ✅ **Land, urban and surface exchange** *(Phases 46–53)*
    - ✅ Phase 46 — Multi-layer soil and hydrology
    - ✅ Phase 47 — Vegetation photosynthesis and stomatal conductance
    - ✅ Phase 48 — Resolved plant canopy
    - ✅ Phase 49 — Production urban canopy model
    - ✅ Phase 50 — Building energy model coupling
    - ✅ Phase 51 — Snow, ice and frozen surfaces
    - ✅ Phase 52 — Water surfaces and wave coupling
    - ✅ Phase 53 — Surface heterogeneity and blending
- ✅ **Turbulence closure** *(Phases 54–59)*
    - ✅ Phase 54 — Production dynamic and scale-dependent SGS models
    - ✅ Phase 55 — Anisotropic and tensorial eddy viscosity
    - ✅ Phase 56 — Prognostic SGS energy and stress
    - ✅ Phase 57 — Wall models for high Reynolds number
    - ✅ Phase 58 — Stable-boundary-layer and intermittent turbulence
    - ✅ Phase 59 — Transition and laminar-turbulent coexistence
- ✅ **Model infrastructure**
    - ✅ Run configuration and input drivers (#311)
    - ✅ One model: core code, configuration and input drivers (#459)
    - ✅ Structured logging, run folders, diagnostics and provenance (#293, #514)
    - ✅ Tests, CI and documentation deployment (#316, #317, #320, #354)
    - ✅ Speed over bit-identity (#493)
- ☐ **Chemistry and aerosol** *(Phases 60–66)*
    - ☐ Phase 60 — KPP-generated gas-phase mechanism
    - ☐ Phase 61 — Photolysis rates
    - ☐ Phase 62 — Aqueous and heterogeneous chemistry
    - ☐ Phase 63 — Modal and sectional aerosol microphysics
    - ☐ Phase 64 — Secondary organic aerosol
    - ☐ Phase 65 — Real emission inventories
    - ☐ Phase 66 — Deposition and pollen
- ☐ **Real-world data, forcing and validation** *(Phases 67–73)*
    - ✅ Phase 67a — ERA5 and reanalysis driver: ERA5 dynamic driver (#363; the reader moved to AeolusPrep.jl, #457)
    - ☐ Phase 67b — ERA5 and reanalysis driver: other reanalyses and the production driver (#57)
    - ☐ Phase 68 — Mesoscale model nesting
    - ☐ Phase 69 — Production OSM and geospatial ingestion
    - ☐ Phase 70 — Observation operators and evaluation framework
    - ☐ Phase 71 — Canonical benchmark library, part 1 (dry)
    - ☐ Phase 72 — Canonical benchmark library, part 2 (moist and reactive)
    - ☐ Phase 73 — Reproducibility and provenance
- ☐ **Data assimilation and inverse problems** *(Phases 74–79)*
    - ☐ Phase 74 — Adjoint model via automatic differentiation
    - ☐ Phase 75 — 4D-Var
    - ☐ Phase 76 — Ensemble Kalman filtering
    - ☐ Phase 77 — Hybrid and coupled assimilation
    - ☐ Phase 78 — Parameter and source inversion
    - ☐ Phase 79 — Uncertainty quantification
- ☐ **Performance, scale and portability** *(Phases 80–87)*
    - ☐ Phase 80 — Production MPI
    - ✅ Phase 81a — GPU port: CPU port to KernelAbstractions (#291)
    - ☐ Phase 81b — GPU port: GPU backends
    - ☐ Phase 82 — Multi-GPU and heterogeneous execution
    - ☐ Phase 83 — Mixed and reduced precision
    - ✅ Phase 84a — Performance engineering: multithreaded scaling (#406)
    - ☐ Phase 84b — Performance engineering: profiling, kernel optimisation and the remaining targets
    - ☐ Phase 85 — Asynchronous and in-situ workflows
    - ☐ Phase 86 — Spectral-element backend completion
    - ☐ Phase 87 — Discontinuous Galerkin option
- ☐ **Machine learning, applications, ecosystem** *(Phases 88–100)*
    - ☐ Phase 88 — Neural SGS closures
    - ☐ Phase 89 — Neural emulators for expensive physics
    - ☐ Phase 90 — Differentiable-physics applications
    - ☐ Phase 91 — Foundation-model coupling
    - ☐ Phase 92 — Wildfire coupling
    - ☐ Phase 93 — Wind energy
    - ☐ Phase 94 — Urban climate services
    - ☐ Phase 95 — Air-quality nowcasting
    - ☐ Phase 96 — Emergency response
    - ☐ Phase 97 — Aviation and contrails
    - ☐ Phase 98 — Coupled Earth-system components
    - ☐ Phase 99 — Ecosystem packages
    - ☐ Phase 100 — Community, documentation and science
- ☐ **Kilometre-scale regional and climate modelling (future direction)** *(Phases 101–112)*
    - ☐ Phase 101 — Compressible non-hydrostatic dynamical core
    - ☐ Phase 102 — Orography in the core
    - ☐ Phase 103 — Grids on the sphere: one abstraction, several grid types
    - ☐ Phase 104 — Lateral and top boundary driving
    - ☐ Phase 105 — Grey-zone boundary-layer turbulence
    - ☐ Phase 106 — Convection and clouds at kilometre scale
    - ☐ Phase 107 — Land surface and hydrology for climate time scales
    - ☐ Phase 108 — Radiative forcing for climate
    - ☐ Phase 109 — Long-run conservation and production output
    - ☐ Phase 110 — Convection-permitting evaluation
    - ☐ Phase 111 — Scale-seamless nesting
    - ☐ Phase 112 — Climate-change downscaling

## Milestones

| After | Aeolus is… |
| :---- | :--------- |
| Numerical foundations (Phase 31) | numerically sound enough to run real-Reynolds urban LES without inflated viscosity |
| Radiation (Phase 38) | radiatively accurate — 3-D urban radiation, real gas optics |
| Clouds and microphysics (Phase 45) | cloud-physics-capable at research standard, superdroplet reference included |
| Land, urban and surface exchange (Phase 53) | a credible urban and land-surface model, validated against field campaigns |
| Turbulence closure (Phase 59) | defensible in the turbulence-modelling literature, including the stable boundary layer |
| Chemistry and aerosol (Phase 66) | a real chemistry-transport model, driven by real inventories |
| Real-world data, forcing and validation (Phase 73) | drivable from reanalysis, validated across the canonical benchmark suite, fully reproducible |
| Data assimilation and inverse problems (Phase 79) | an inverse-problem and uncertainty-quantification platform |
| Performance, scale and portability (Phase 87) | HPC-production: MPI + GPU at scale, high-order backends |
| Machine learning, applications, ecosystem (Phase 100) | a community Earth-system-scale tool with an ML-native workflow and real applications |
| Kilometre-scale regional and climate modelling (Phase 112), future direction | a convection-permitting regional climate model (~3 km) that nests seamlessly down to its own urban LES |

---

# LES core and physics scaffolds

*Phases 1–20, before the long-range plan.* A dry and moist LES core with
pluggable pressure solvers, the first urban example, and scaffolds for
everything else (MPI, GPU, spectral elements, nesting, radiation, land,
chemistry). The rule that followed, "no scaffolds", made each of those
scaffolds a later phase's job to replace; the checklist above says which.

After Phase 20 a refinement pass driven by the Alexanderplatz run (27
items: shadows and sky-view factors, wall temperatures, latent and
anthropogenic heat, multi-layer soil, MOST, dynamic Smagorinsky,
two-moment microphysics, four-species chemistry, aerosol, nesting and
terrain scaffolds) closed the most visible gaps; the phases from 21 on
replace its approximations with validated schemes. The original phase
descriptions are in git history (`docs/src/design/roadmap.md` before
2026-11-06).

# Numerical foundations

*Replace every numerical compromise that currently forces artificially
high viscosity or small time steps. Without this milestone, none of the
physics work downstream can run at realistic parameters.*

**Status: complete.** 2,094 validation tests, all green. Every phase
ships a production implementation with measured validation against an
analytic solution, a manufactured solution, or a convergence study —
numbers recorded per phase below.

### Phase 21 — Cut-cell immersed boundary method
`src/CutCell.jl`. Conservative cut-cell geometry from a level set by
sub-cell sampling: wetted volume fractions, per-face open-area
apertures, surface normals from `∇φ`, wetted areas. Aperture-weighted
divergence / gradient / Laplacian, small-cell volume floor plus optional
flux-redistribution merging, exact no-penetration enforcement.
**Validated:** discrete adjointness `⟨p,Du⟩ = −⟨Gp,u⟩` to **1.45e-15
relative** on a sloping wall with an embedded sphere (this is what makes
the projection exact on cut geometry); `L = D·G` to 2.2e-17 relative;
half-cut cell volume fraction and area exact to 1e-14; tilted-plane
normals to 1e-12; uniform-flow divergence past a cut wall < 1e-12.
**Unlocks** `ν = 10⁻⁵` in urban geometry — the whole point of the milestone.

### Phase 22 — Ghost-cell wall reconstruction
`src/CutCell.jl`. Image/foot-point stencils on the *true* surface with
trilinear donor weights (solid donors dropped, inverse-distance
fallback), Dirichlet and Neumann reconstruction, wall shear stress on
the true normal.
**Validated:** linear profiles reproduced **exactly** (max error 0.0);
quadratic-profile errors 1.06e-3 → 2.66e-4 → 6.64e-5 at Nz = 16/32/64,
i.e. convergence ratios **4.0000 and 4.0** — clean second order;
analytic wall stress `νᾱ = 0.0375` recovered as 0.0375 with zero spread.

### Phase 23 — Adaptive time stepping with substepping
`src/TimeIntegration.jl`. Per-process CFL diagnostics reporting each
limiting `dt` separately (advective via the true per-cell sum form,
diffusive, Brunt-Väisälä, Coriolis, SGS-scalar) plus which one binds;
growth-limited adaptive `dt`; process substepping; and retry-on-failure
with bit-exact preallocated state rollback.
**Validated:** hand-computed advective `dt` to 1e-12 with the right
`limiter` symbol; growth limiting never exceeded; deliberate
over-large-`dt` blow-up triggers rollback and recovers a finite state.

### Phase 24 — Stretched and terrain-following vertical coordinate
`src/StretchedGrids.jl`. `StretchedGrid` (uniform / geometric / tanh /
two-wall-tanh / custom) and `TerrainFollowingGrid` (Gal-Chen–Somerville
σ) as `AbstractMesh` subtypes, with metric-correct z-stencils added as
new methods on the `FiniteVolume` internals so the public `Operators`
façade picks them up by dispatch with no edit to it. Two distinct
metrics (`dzc` cell thickness, `dzf` centre spacing) so `div(grad)` stays
self-adjoint. Volume-weighted vertical interpolation.
**Validated:** `:uniform` reduces to `RegularGrid` to **1.44e-15**;
Laplacian convergence ratios **3.976, 3.995**; centre→face `∂z` exact on
a linear profile to 6.2e-17; volume-weighted interpolation **5.7e-14**
where naive ½-averaging gives **0.270**; flat terrain reduces to the
stretched case to 1.7e-18 with `J ≡ 1`; `∂x` of physical height over a
bell hill **3.5e-15** with the metric cross-term vs **0.158** without;
hydrostatic residual 3.75e-4 relative and *unchanged* over the hill,
converging at ratios 3.99, 4.00 — and 86.5× worse if the wrong metric
is used, which the suite asserts.

### Phase 25 — Semi-implicit IMEX time integration
`src/TimeIntegration.jl`. Matrix-free implicit diffusion `(I − dtν∇²)`
via Jacobi-preconditioned CG (scalar or per-cell ν, wall rows
eliminated), plus `ARK2` (ARS(2,2,2)) and `Rosenbrock2` (ROS2 W-method)
drivers that project at every stage so each stage value is
divergence-free.
**Validated:** manufactured implicit solve in **3 CG iterations**, max
error 9.1e-15 (34 iterations for variable ν); **unconditionally stable
at 100× the explicit limit** — 20 backward-Euler steps, monotone decay,
non-negative, mass drift **5.2e-12**; measured convergence orders
**2.0004 / 2.0002** (ARK2) and 1.986 / 1.993 (Rosenbrock2); divergence
4.3e-16 after stepping with walls + SGS + body force; **18.4× dt
speed-up** over the diffusive constraint.

### Phase 26 — WENO-Z and TENO advection for all prognostics
`src/AdvectionSchemes.jl`. WENO-Z5/Z7 (τ-based Z weights), TENO5/TENO8
(ENO cut-off then optimal linear weights), corrected WENO-JS5 as an
accuracy reference, high-order staggered momentum advection, and a
positivity-preserving flux limiter. All coefficients derived in exact
rational arithmetic; smoothness indicators evaluated in the `a`-basis so
they keep relative accuracy as `h → 0`.
**Validated:** measured orders WENO-Z5 **5.00**, WENO-Z7 **6.86–6.98**,
TENO5 **4.99**, TENO8 **6.95–7.19** (polynomial exactness is the sharper
statement: TENO8 reproduces a degree-7 polynomial to 4.4e-16);
**critical-point accuracy** — WENO-Z 1.19e-9 vs WENO-JS 8.73e-9 (7.3×
better), the defect Z-weights exist to fix; square-wave total-variation
ratios 1.0004 (Z5) / 0.99992 (Z7) / 1.00029 (TENO5) vs **4.11 for
centred**; positivity — without the limiter a high-order scheme reaches
**−1.84** (Z5) and **−19.6** (TENO8), with it **−1.7e-18** and −2.0e-28
at mass drift ≤ 3e-15.

### Phase 27 — Anelastic and pseudo-incompressible dynamics
`src/Anelastic.jl`. `ReferenceState` base states (isothermal,
isentropic, constant-`N²`, arbitrary profile) constructed in **discrete**
hydrostatic balance on the model's own grid, with a logarithmic-mean
face density so the exponential case is simultaneously analytic and
discretely balanced. `DensityWeightedPoissonSolver` (deflated-Jacobi PCG
for `∇·(w(z)∇φ) = f`) and one generic mass-weighted core aliased as
`AnelasticSimulation` (`w = ρ̄`) and `PseudoIncompressibleSimulation`
(`w = ρ̄θ̄`).
**Validated:** hydrostatic residual **3.0e-13 … 6.8e-14 Pa/m** across
all four constructors (≤ 4.3e-14 relative); isothermal `ρ̄` matches the
analytic exponential to 2.2e-16; constant-`N²` recovered to 2e-17;
variable-coefficient Poisson with a **density ratio of 42.5** recovered
to **1.25e-13** in 71 iterations, operator exactly self-adjoint
(`max|A − Aᵀ| = 0.0`); constraint after a step `(1/ρ̄)∇·(ρ̄u) = 1.1e-17`
while plain `∇·u = 3.1e-5` — confirming the leftover is exactly
`−w ∂log ρ̄/∂z`; Boussinesq-limit error grows **linearly** with density
variation across three decades (9.96, 10.25 per decade); warm bubble
rises in all three systems with anelastic ascent 3.1% stronger aloft.

#### Direct anelastic pressure solver (#309)

**✅ SHIPPED** (`PressureSolvers.ColumnTridiagonalSolver`,
`Anelastic.DirectWeightedPoissonSolver`; #309 / PR #386). The
density-weighted operator div(w(z) grad) is separable, so the
sound-proof cores now project with a horizontal FFT/DCT plus one
tridiagonal solve per wavenumber. It is the default
(`pressure_solver = :direct`); the Jacobi-preconditioned CG stays as
`:cg`. `StretchedPoissonSolver` delegates to the same column solver,
so the FFT/DCT + Thomas code has one source.

**Measured**: manufactured solution recovered to 1.2e-15–3.9e-15
relative; agreement with the CG 8.7e-14–2.7e-13; a single solve at one
thread 176 ms (367 CG iterations) → **1.05 ms** at 64³ (166×) and
1868 → 8.1 ms at 128×128×64 (231×). Berlin, 64³, nest off,
`core = :anelastic`: **391 → 200 ms per step**, weighted divergence
6.4e-13 → 2.9e-17 s⁻¹.

**Expectation that was wrong** (merge train):
`logging_step_diagnostics` assumed the old CG default of the
sound-proof cores (an iterative solve with iteration counts) after PS-a
made the direct solve the default.

**Not built**: a periodic vertical (the sound-proof cores reject one;
the CG covers it).

### Phase 28 — Energy- and enstrophy-conserving discrete operators
`src/AdvectionSchemes.jl`. Morinishi skew-symmetric momentum advection
(exact discrete KE conservation for *any* velocity field), a
variance-conserving scalar form, and the KE / scalar-variance budget and
enstrophy diagnostics that serve as the validation instruments.
**Validated:** KE production on a non-divergence-free field — skew
**8.3e-17** vs flux form **0.176**, a 2e15 separation; Taylor-Green
advection-only drift over 2000 steps — skew **−3.19e-11** (pure RK3 time
error) vs flux form **+0.223**, ~7e9× worse; variance-conserving form
**−2.2e-16** vs centred **−2.77e-2** when `∇·u ≠ 0`; KE-budget closure
residual **5.4e-16 relative**; `enstrophy` satisfies
`νΣu·∇²u = −2ν·enstrophy` to 1e-8, an independent check of the staggered
curl.
**Bug found and fixed:** the Phase-10 WENO-5 in `src/FiniteVolume.jl`
wrote `2f0` / `3f0` in its smoothness indicators. In Julia those are
*Float32 literals*, not `2*f0` / `3*f0`. The scheme was therefore not
translation-invariant (violation 0.0236), measured ~3rd order instead of
5th, and *produced* scalar variance where every other upwind scheme
dissipates. The centre argument is now named `fc` so the mistake is
unrepresentable; translation-invariance violation is now **0.0**.

### Phase 29 — Multigrid pressure solver
`src/Multigrid.jl`. Cell-centred geometric multigrid: V/W-cycle plus
full-multigrid start, red-black Gauss-Seidel smoother, full-weighting
restriction and trilinear prolongation, variable-coefficient support
with coefficient coarsening, null-space projection at every level for
singular (all-periodic / all-Neumann) systems, and a
multigrid-preconditioned CG variant.
**Validated:** cycle count **flat at 7 from 16³ to 64³** — a 64× cell
increase with no growth, which is the defining multigrid property and
exactly what CG cannot do; per-cycle convergence factor **0.07–0.12**;
manufactured solution to 1e-9; agreement with the exact cosine-FFT
solver to 1e-8 on channel BCs; variable-coefficient recovery 6.4e-11;
**MG-PCG converges in 8 iterations**; divergence 3.7e-13 when used as
the projection solver inside a running simulation; non-power-of-two
grids degrade gracefully via semi-coarsening.

### Phase 30 — Grid nesting with proper two-way feedback
`src/AdaptiveMesh.jl`. Prolongation via the primitive function (parent
cell averages fix the cumulative integral exactly at parent faces), so
conservation is *structural* rather than approximate; staggering handled
per-direction so a face velocity is nodal along its own axis and
conservative across the others; volume/area-weighted restriction;
Berger-Colella flux registers and refluxing; a tapered nest sponge; and
subcycled stepping with time-interpolated boundary data.
**Validated:** prolongation orders **1.97 / 3.04 / 3.97** for requested
2 / 3 / 4; restriction conservation **bit-exact** at r=2 and 4.0e-16 at
r=3; flux-preserving prolongation interface mismatch **1.5e-33**;
**mass error 0.0 with flux correction vs 1.06e-3 without** — the
correction doing real, measurable work; nest sponge reduces reflected
amplitude **12.6×** (3.71e-2 → 2.94e-3).
*Since then* the parent–child operators moved to `src/Nesting.jl`
(#250) and gained multi-level hierarchies, sibling, non-aligned and
stretched children (#328); see the two records below.

#### Nesting hardening: hierarchies, siblings, non-aligned and stretched children (#327–#329, #349)

**✅ SHIPPED** (`src/Nesting.jl`, `src/nesting/`; follow-ups of the
Berlin nesting epic #246, whose N-* PRs #272–#292, #323, #330 and #343
are counted with Turbulence closure; #327 / PR #339, #329 / PR #348, #349 / PR #371,
#328 / PR #394). What was built:
- **Hardening M1–M6 (#327).** One mixed-core two-way guard,
  `_check_two_way_pair`. One per-face advective-plus-diffusive kernel,
  `scalar_face_flux`, replacing two copies. Flux registers that book the
  flux the transport actually applied (`book_nest_flux!`,
  `applied_by = :transport`), with qv allowed as a tracer. One
  solid-at-open-face path (`mark_open_boundary_solids!`). `_try_advance!`
  retries only numerical failures (`NumericalFailure`) and rethrows the
  rest.
- **Minor findings (#329).** Outflow and halo passed through `nest_bcs`.
  Refluxing respects the parent's boundary conditions. `NestSponge` acts
  before the projection. Halo-only (shell) field sets. Typed nesting
  state. Conservative prolongation limiters `:positive` (Zhang & Shu
  2010) and `:monotone` (Barth & Jespersen 1989). One two-snapshot time
  blend, `WindowedBoundaryData`. A public `fill_nest_halos!`.
- **Follow-ups (#349).** A sponged `NestedRun` refuses to be driven
  externally. The `periodic` keyword of the flux correction and of
  refluxing is required.
- **Hierarchies (#328).** `NestAxis` overlap geometry (rational
  refinement ratios, off-face origins), overlap prolongation and
  restriction, stretched children, open boundaries on a z-stretched
  mesh, and `NestHierarchy`: nest levels and sibling nests with
  recursive Berger–Oliger subcycling. `NestedRun` is now a facade over
  it, golden-tested bit-identical to the old pair. Theory:
  `docs/src/theory/nesting_hierarchy.md`.

**Measured**:
- Flux registers (M3), 100 steps with a WENO-Z5 tracer slot: composite
  tracer drift **1.2e-14 per step** booked by the transport, against
  2.9e-7 booked by the core hook (the pre-#327 way) and 2.5e-6 unbooked.
- Two-way circulation (M5), 4–6 seeds per configuration: two-way minus
  one-way child-mean w at mid-depth −0.163 ± 0.057 m/s (6 of 6 seeds),
  and none at r = 1 (+0.034 ± 0.062 m/s). It is thermally direct: a
  standalone CBL at the child's spacing is 0.080 K cooler below 800 m.
- Sponge before the projection: child max|∇·u| **2.8e-14 s⁻¹**, against
  1.6 s⁻¹ with the same relaxation after the step.
- Memory and allocation: a 32³ child's field sets 5144 → 2144 KiB; per
  nested step one-way 5371 → 57.7 KiB and two-way 9009 → 78.0 KiB; a
  32³ Boussinesq `step!` 1641 KiB → 0.
- Limiters: an unlimited oblique 0/1 front gives [−0.250, 1.250]
  (order 3); `:monotone` keeps [0, 1] and the parent mean.
- Hierarchies (three levels, two siblings at r = 2 and 3, r = 5/2 with
  off-face edges, a stretched child; one- and two-way; `step!` and
  `step_rk3!`): free stream max|u − U|/|U| ≤ **3.8e-15** and
  |∇·u|Δ/|U| ≤ 7.5e-16; composite θ with refluxing ≤ **5.2e-15 per
  step** (3.5e-8 to 2.7e-7 without); prolongation order 1.99–4.00 for
  r ∈ {5/2, 3/2, 7/3}, and 1.98 / 2.99 / 3.98 on a stretched child.
  RMS error over the nest against an exact solution: coarse 0.235,
  child 0.106, grandchild 0.070, uniform 25 m reference 0.031.

**Bugs the tests caught**:
- #287 had deleted the mixed-core two-way guard (M1): a Boussinesq
  parent with an anelastic child ran two-way, silently wrong.
- The merged M2 flux kernel inferred as `Tuple{Any,Any,Any}`: 313 kB per
  call and about 43× slower. Now 0 B, with an `@inferred` test.
- A vertically periodic open box never initialised its radiating
  outflow, so a `:convective` nest face stayed Dirichlet (residual 1.0
  instead of 1.15e-7).
- `boundary_time` and the two-way feedback did not invalidate the cached
  sponge target: two sponged RK3 runs differed by 1.4e-3.
- Refluxing wrapped every axis with `mod1`, so a top-plane correction
  landed in the parent's lowest layer.
- A sponged child driven through `begin_child_step!` got no velocity
  relaxation and no warning (max|Δu| 4.02e-3 in one step).
- A first version made `:positive` the default of the low-level
  interpolators; the prolongation-order tests on products of sines
  failed. The code default was wrong, not the tests.

**Expectations that were wrong**:
- The mass-measure refluxing test passed only because the triply
  periodic default wrapped a flux through the parent's ground into its
  top layer. Its registers are now physical (drift 3.86e-15 with
  refluxing, 1.71e-7 without).
- "The two-way circulation has the opposite sign to the literature"
  (M5): the descent is a resolution effect, not a coupling bug.

**Not built**: overlapping siblings (refused; no priority rule); two-way
feedback on a non-aligned child without a common-face conservation box
(refused); escalating rollback; validation in the Berlin geometry
(idealised cases only, Berlin runs paused). `AdaptiveMesh` keeps its own
index-space proper-nesting check.

#### Nested Berlin: synthetic inflow on by default, checkpoint/restart (epic #246, open)

**✅ SHIPPED** (#268 / PR #415, #269 / PR #416). What was built:
- **Synthetic inflow on the Berlin nest (#268).** The `:inflow_turbulence`
  addon puts `Transition.InflowTurbulence` on the child's lateral nest
  faces, on by default: the digital filter (Xie & Castro 2008) by
  default, FSD (`free_stream_generator`, now the single FSD factory) as
  an option; statistics from the parent's resolved field by default
  (`ParentResolvedStatistics`, which now accepts a `StretchedGrid`
  parent, bit-identical to a `RegularGrid` on a uniform grid; SGS TKE
  from each level's thickness), or parametrized. They are folded from a
  new nest window hook (`fold_inflow_statistics!`). `budget!` books the
  injected KE flux as `:inflow_ke`. Two-way feedback leaves out an
  inflow band of 2 max L_s (`exclude_inflow_feedback!`). A new
  `CaseConfig` field `[boundaries] analysis_buffer` (default 140 m, the
  canopy-top estimate of the N-inflow-d fetch rule) and
  `analysis_columns(case)` cut the columns near lateral faces out of the
  observation-compared u* diagnostics.
- **Checkpoint/restart of the nested Berlin case (#269).**
  `Aeolus.IO.structural_state` / `restore_structural_state!` record and
  refill the whole mutable state of an object graph, aliasing included.
  `nest_state` / `restore_nest_state!` cover a whole `NestedRun`: both
  domains with halos, closure memory, open-boundary buffers, the two
  parent time levels and the window, step-size memory, flux registers,
  statistics, boundary data and the inflow generator, with a parent step
  running ahead settled first. The Berlin driver checkpoints the whole
  case on the `[checkpoint]` schedule (on in `berlin_summer.toml`, every
  1800 s of wall clock, 2 kept; Boussinesq core only) and `run!` resumes
  the newest checkpoint into a fresh build, NetCDF output continuing at
  the checkpoint's record (`nc_append_record!(...; at)`).

**Measured**:
- Idealised nest (`inflow_nest_longrun.jl`, Hellsten-type CBL,
  parent 100 m, r = 3, 2 h): one-way, resolved TKE within 200 m of the
  upwind face at 50–150 m is 3.165 m²/s² with the generator against
  2.278 without (+39 %). KE flux 0.786 W/m²; largest added mass flux
  3.7e-17 of the mean-flow flux; child max|∇·u| 1.1e-16.
- Two-way without the generator, against one-way: child KE ×1.04,
  parent ×0.995, overlap θ −0.002 K (passes).
- Berlin smoke (12 steps of 2 s, noon), every variant: added mass flux
  1.3e-18 to 2.7e-18 of the mean-flow flux; KE flux 0.0007–0.0017 W/m²;
  child heat ledger ≤ 7e-13 W/m², water ≤ 4.6e-16 kg/m².
- Berlin flagship, 80 s of model time (coupling check, not a fetch
  measurement): the generator adds +3.9 % TKE at canopy top within
  200 m of the upwind face at 03:00 and +2.5 % at 12:00; 163 against
  161 ms/step.
- Nested restart (`nest_restart.jl`, TKE parent, Lagrangian-dynamic
  child, r = 3, generator on, adaptive steps, checkpoint mid-window):
  bitwise equal to the uninterrupted run in every field, closure memory,
  open-boundary buffer, generator plane and clock, one-way, one-way with
  the parent running ahead, and two-way, at 1 and 12 threads. Restoring
  only the two simulations differs in 11, 16 and 28 of 30 items. Record
  8.8–9.0 MiB, write 0.03–0.8 s, restore 0.03–1.1 s.
- Berlin smoke resume (`config_roundtrip.jl`, every coupling on, 15
  adaptive steps, checkpoint every 6, interrupted after step 9): state
  bitwise equal, parent included, and all 51 NetCDF variables equal.
  Flagship checkpoint 2754.5 MiB in 9.5 s (child alone 20.9 MiB,
  0.02 s); a fresh process restored it and stayed bitwise for 30 child
  steps.

**Two-way nesting with the generator: stable since #418 / PR #419.**
#268 measured it unstable on the idealised nest over 2 h, against
one-way with the generator: child KE ×12.4 (×11.8 in a second
realisation), parent ×2.6 (×2.4), overlap θ +0.26 K (+0.23 K). The
cause was a loop through the two-way interface flux correction: it set
each parent interface face to the child's flux, which included the
synthetic perturbation, so the next window's boundary data carried the
perturbation back and the parent statistics measured the grown eddies.
PR #419 subtracts the synthetic normal velocity (a new contract generic
`synthetic_normal_velocity(data, side, t)` in BoundaryConditions,
implemented by `InflowTurbulence`) from the child's open faces for the
correction and its ledger; runs without a synthetic part stay
bit-identical. Measured, two-way against one-way with the generator,
2 h sheared CBL, r = 3: child KE ×0.937 / ×1.091, parent ×1.015 /
×1.020, overlap θ −0.0031 / −0.0085 K (seeds 1 / 2), within #268's
1.5× / 0.2 K criteria. The former `@test_broken` in
`inflow_nest_longrun.jl` are real tests for both realisations
(18/18 pass). Decaying refine-3 nest, 2 h: child ×0.985, parent
×0.999, θ +0.0009 K (×2.26, ×1.21, +0.31 K before). Composite θ ledger
drift ≤ 3.5e-14 per parent step; interface imbalance before the
correction 1.2e-16 (1.4e-2 before). Footprint-excluded parent
statistics were measured and dropped (child ×1.76 alone, no gain on top
of the fix). The coupling's two-way warning is removed; the Berlin
two-way nest itself stays unvalidated (#331).

**Bugs the tests caught**:
- A shallow copy in the structural walk restored the run's own plane
  arrays into the fresh build; `nest_restart.jl` caught it.
- An example driver's types could not be deserialized: Serialization
  looks a type up by its module path, so driver modules are now bound
  in `Main`.

**Expectations that were wrong**:
- The inflow smoke check expected exactly zero perturbation on outflow
  levels; the applied perturbation blends two knots in time, so a level
  just turned to outflow keeps the older knot's share. The check is now
  on the newest knot.
- The config_berlin check for the anelastic core with checkpoints first
  hit the adaptive-timestepper incompatibility and never ran its own
  assertion; it now sets `timestepper = "fixed"`.
- At merge, the smoke matrix's anelastic run inherited `[checkpoint]`
  from `berlin_summer.toml` and was refused (correctly) by the config
  check; the smoke matrix now turns checkpoints off on a non-Boussinesq
  core.

**Not built**: the Berlin fetch measurement (hours of model time on the
nested child; long runs are paused), deferred to the #270 24 h
validation, which should replace the 140 m `analysis_buffer` default.
Checkpoints of the anelastic core; resume of `child_boundaries` runs
(refused with a message).

### Phase 31 — Adaptive mesh refinement
`src/AdaptiveMesh.jl`. Block-structured AMR: Löhner gradient, vorticity
and Richardson-extrapolation refinement criteria with union /
intersection / majority combination; Berger-Rigoutsos signature-based
clustering; conservative regridding; proper-nesting enforcement by flag
erosion; and recursive subcycled level advance with refluxing.
**Validated:** criteria flag 896 cells at a tanh front and **0** in the
smooth region (vorticity: 512 at a shear layer, 0 for uniform flow);
Berger-Rigoutsos turns 344 flagged cells into **27 blocks at 0.787
efficiency**; mass conserved to **1.13e-16 across two regrid events**
(vs 4.49e-3 with refluxing disabled); and AMR reaches **near-fine-grid
accuracy at 56% of the cell count** (L1 error 1.206e-2 on 2,296 cells
vs 1.247e-2 on 4,096 uniform-fine cells, against 2.019e-2 coarse).

### Numerical foundations — known limitations carried forward

Recorded honestly rather than papered over; each is a candidate
follow-up phase:

  * **Solver/geometry coupling.** The FFT and cosine-FFT pressure
    solvers are aperture-unaware, so cut-cell runs must re-apply the
    velocity BC after the legacy projection. Wiring `cutcell_laplacian!`
    into the Phase-29 multigrid (which already supports variable
    coefficients) closes this.
  * **Stretched-grid operator coverage.** Advection, momentum advection,
    Coriolis and the Smagorinsky/eddy-viscosity kernels are not yet
    retrofitted for `StretchedGrid` — they `MethodError` rather than
    silently using the wrong metric, which is the safe failure mode.
  * **Cut-cell geometry is sampling-quantised**; marching-cubes exact
    intersection is the refinement.
  * **IMEX SGS term** is a co-located approximation of the Smagorinsky
    stress divergence, and `ν_eff` is lagged at `uⁿ` (standard).
  * **Nested pressure projection** is single-level per grid; a
    multi-level MAC sync projection is out of scope, so refluxing leaves
    `O(mismatch)` divergence in the parent shell.
  * **Sound-proof cores** reuse the Boussinesq viscous/SGS kernels
    rather than density-weighted flux divergences, and do not carry
    moisture.
  * **AMR block advance** uses a prescribed velocity field with
    flux-form upwind; per-block momentum and pressure solves are future
    work.

# Radiation

**Status: all seven phases (32–38) and the shared contract are
complete** (Phases 34–37 landed together in `ac38822`; the radiation test set
is 4,912 tests on master at `2895fed`). `src/RadiationCore.jl` fixes the interface every component in
the radiation stack speaks — `SpectralGrid` (RRTMG-layout bands with correlated-k
g-points and weights), `OpticalProperties` (τ/ssa/g with the correct
optical-depth- and scattering-weighted `combine_optics!` rule),
`RadiativeFluxes` (level-based, bottom-up, `net = dn - up` positive
downward), `AtmosphericColumn`, and the `AbstractGasOptics` /
`AbstractCloudOptics` / `AbstractAerosolOptics` / `AbstractRTSolver` /
`AbstractSurfaceOptics` interfaces. Validated: the shortwave band solar
source sums to exactly 1361.0 W/m², and π·Σ_bands `planck_band` recovers
σT⁴ to 0.008% (the residual lying genuinely outside 10-3250 cm⁻¹).

*Replace the prescribed-profile and Beer's-law approximations with a
real radiative-transfer solver, including 3-D effects that matter in
urban geometry.*

### Phase 32 — Correlated-k gas optics
`src/GasOptics.jl`. Full spectral treatment of H₂O, CO₂, O₃, CH₄, N₂O
and O₂ across the 16 RRTMG longwave and 14 shortwave bands, on the
model's own vertical grid, replacing the prescribed LW-down constant.

**Design deviation from the original plan, and why.** The plan said
"bind `RRTMGP.jl`". That is not possible on this host: RRTMGP.jl pulls
in NetCDF/HDF5 for its k-distribution files, and those binary
dependencies do not load here (the same `libmpifort.so` failure that
forced the Serialization+TOML I/O layer in Phases 1–20). Rather than ship a
module that cannot be loaded, Phase 32 builds the k-distribution
**from first principles** instead of reading it from a file:

  * A **Malkmus statistical narrow-band model** with embedded line
    parameters (mean line strength and mean line spacing per band per
    species) gives band transmission in closed form.
  * **Domoto (1974) inversion** converts that band transmission into a
    cumulative k-distribution, from which per-g-point absorption
    coefficients follow — the standard route from a band model to
    correlated-k when line-by-line data is unavailable.
  * Pressure and temperature scaling of the absorber amount follows the
    usual Curtis-Godson-style path scaling.

This is less accurate than reading real RRTMGP k-tables — it is a
band-model reconstruction, not a line-by-line fit — but it is
self-contained, loads everywhere, and lands the broadband fluxes within
a few percent of observations (below). A future phase can add an
optional RRTMGP.jl backend behind the same `AbstractGasOptics`
interface on hosts where it loads.

Also implemented: Rayleigh scattering optics, Planck-weighted longwave
layer and surface sources, and `us_standard_column` for a reference
atmosphere. `erf_series`/`erfc_series` are hand-written because
`SpecialFunctions` does not load either.

**Measured validation** (US Standard Atmosphere, 40 layers):

| Quantity | Aeolus | Reference |
|---|---|---|
| Clear-sky OLR | 240.2 W/m² | ~240 (global mean) |
| Surface LW downwelling | 306.7 W/m² | 288 (Brutsaert, this column) |
| Surface LW upwelling | 389.2 W/m² | — |
| Greenhouse effect Δ | 149.0 W/m² | ~150 |
| LW cooling rate (mean) | −0.92 K/day | −1 to −2 |
| CO₂ doubling band-4 response | +52.6 % absorption | — |
| Window transmission, dry column | 0.81 | window stays open when dry |

(These are the numbers *after* the window recalibration described below.
Before it they were OLR 229.1, surface downwelling 345.8 and greenhouse
effect 160.9 — the first and third visibly too low and too high
respectively, and the second too high by ~58 W/m² against the Brutsaert
empirical relation for a column of this humidity, though it looked
plausible against the global-mean 333 W/m² that belongs to a much
moister atmosphere.)

**Bug found and fixed during validation**: at large absorber amounts
the band transmission underflows to exactly 0, so `log(T)` returned
`-Inf` and the optical depth became `NaN`. Added
`malkmus_log_transmission`, which computes `log T` directly without
ever forming `T` — a genuine robustness fix, not a test workaround.

**Test expectation corrected**: an early test asserted that
transmission *rises* with pressure. That is backwards — pressure
broadening widens lines and fills the gaps between them, so higher
pressure means **more** absorption and **lower** transmission. The
module was right and the test was wrong.

**Accuracy bug found later by the DYCOMS-II RF01 case, and fixed.**
The version of Phase 32 that shipped first had *two compensating
errors* in the 8-12 μm atmospheric window: water-vapour line strengths
roughly 50× too large, and a water-vapour continuum roughly 15× too
weak. Those two cancel almost exactly for one particular column, so the
US Standard Atmosphere broadband fluxes in the table above looked
fine — which is precisely why a single broadband number is not an
adequate test of a band model.

They do not cancel in general, because the two terms scale differently:
line absorption grows as ``\sqrt{ku}`` in the strong-line limit, while
the continuum grows as ``u\,(e + \gamma p_{dry})`` — linear in amount
*and* in water partial pressure. Applied to the dry 4.2 kg/m² column
above the DYCOMS RF01 cloud top, the window emissivity came out at 0.38
where it should be about 0.12, adding ~27 W/m² of spurious downwelling
and cutting the cloud-top radiative cooling — the single term that
drives the entire RF01 boundary layer — nearly in half.

The fix replaces the ad-hoc continuum with the **Roberts, Selby &
Biberman (1976)** parameterisation, which is calibrated against
measurement and has the correct self/foreign split (γ = 0.002, so the
self-continuum dominates wherever there is appreciable water), and
rebalances the window line strengths against it. See
`examples/validation/dycoms_rf01_radiation.jl`.

The lesson is recorded here because it generalises: **a band model must
be validated across columns that differ in absorber amount and
pressure, not against a single atmosphere.** Compensating errors are
invisible to the latter and obvious to the former.

### Phase 33 — Two-stream solver with cloud optics
`src/TwoStream.jl`. Three two-stream closures behind one interface —
**delta-Eddington**, **PIFM** (Zdunkowski) and **Quadrature** — using
the Meador & Weaver (1980) layer reflectance/transmittance
coefficients, an adding/doubling column solver, maximum-random cloud
overlap, and parameterised Mie cloud optics in terms of liquid/ice
water path and effective radius. Full cloud-radiative feedback on the
thermodynamics via `radiative_heating!`.

**Numerical care that mattered.** Meador & Weaver eq. 14 as written
contains `e^{kτ}` factors that overflow for optically thick layers.
The implementation divides numerator and denominator through by
`e^{kτ}` analytically, so the expression is evaluated entirely in terms
of `e^{-kτ} ≤ 1` and nothing overflows even at τ = 10⁴.

The longwave layer source is **linear in τ** rather than constant.
Measured against a sub-layered reference of 180.66 W/m², the linear
form gives an error of **0.55** where the constant form gives **3.30**
— a 6× accuracy gain for no extra cost.

**Measured validation**:

| Check | Result |
|---|---|
| Conservative scattering, white surface: (out/in) | 0.99999999997 |
| Cloud τ at LWP 100 g/m², r_eff 10 μm | 15.0 (textbook 15) |
| …its single-scattering albedo | 0.99999 |
| …its asymmetry parameter | 0.86 |
| Cloud albedo at LWP 10/50/200/1000 g/m² | 0.110 / 0.407 / … monotone, < 1 |
| LW heating rate (min/mean/max) | −2.48 / −1.00 / −0.089 K/day |

**Two real implementation bugs were found by the tests** — worth
recording because both produced plausible-looking output:

  1. A spurious trailing `* k` on the direct-beam reflectance and
     transmittance. Conservative scattering returned 1.5 W/m² instead
     of 800, and a thick cloud had an albedo of 0.022 instead of > 0.5.
     Diagnosed by re-deriving against Meador & Weaver eq. 14 term by
     term.
  2. `S_up` and `S_dn` swapped, with the wrong integral, in the
     linear-in-τ longwave source. This made the "improved" linear
     source *worse* than the constant one (error 6.03 vs 3.28), which
     is what gave it away. Fixed by deriving the moment integral
     properly: `∫₀^x s e^{-s} ds = 1 − e^{-x}(1+x)`.

**Test expectations corrected** (implementation was right): a
reflectance floor of `1e-6` at τ = 10⁴ with ω = 0.999 ignores that the
diffusion regime falls off only as 1/τ; and a 96 K/day heating rate
came from an artificial grey optical depth, not from the solver — it
drops below 30 K/day with real gas optics.

**DYCOMS-II RF01 intercomparison**: now implemented, in
`examples/validation/dycoms_rf01_radiation.jl`. RF01 is nocturnal marine
stratocumulus in which longwave cloud-top cooling is the *only* term
driving the boundary layer, which makes it the sharpest available
single-column test of a longwave scheme. The intercomparison prescribes
the answer analytically — a net upward flux of F₀ = 70 W/m² at cloud
top and F₁ = 22 W/m² at cloud base — so it is a genuine external check.

| Quantity | Aeolus | DYCOMS spec |
|---|---|---|
| Net upward LW at cloud base | 24.8 W/m² | 22.2 |
| Net upward LW at cloud top | **70.3 W/m²** | 70.1 |
| Jump across cloud | 45.6 W/m² | 47.9 (−4.8 %) |
| Peak cooling rate | −296 K/day, 2.5 m below cloud top | — |
| Mean cooling, top 50 m | −83.9 K/day | ΔF/(ρc_pΔz) ⇒ −85 |

The sounding itself reproduces the specification: cloud top lands exactly
on the 840 m inversion and peak liquid water is 0.472 g/kg against the
RF01 reference of ≈0.45.

Running this case is what exposed the Phase 32 window-opacity bug
recorded above — the first version of it disagreed with the prescribed
cloud-top flux by 54 %.

### Phase 34 — 3-D Monte-Carlo radiative transfer
Photon-tracing solver that handles the genuinely 3-D radiative
environment of an urban canyon or a broken cloud field: side
illumination, inter-reflection, and shadowing without the
one-dimensional plane-parallel assumption. Runs as a periodic
correction to the two-stream solution.
**Validated by**: I3RC intercomparison cases; comparison of canyon
absorption against the two-stream + SVF approximation.

**Built** (`src/MonteCarloRT.jl`): Amanatides–Woo voxel traversal,
Henyey–Greenstein and Rayleigh phase sampling with closed-form inverse
CDFs, implicit capture, Russian roulette and next-event estimation;
every routine takes an `rng`, so runs are reproducible.
**Measured**: against delta-Eddington over 16 cases the albedo differs
by a median 0.0059 (worst 0.0223 at τ = 1, g = 0.85, μ₀ = 0.5, where the
Eddington closure is weakest; tolerances from the computed standard
error); the Beer–Lambert direct beam is exact to < 1e-12; energy closes
to < 1e-10 per realisation with buildings; SE(N)/SE(4N) = 2.09. A cubic
cloud absorbs 11.5 % less than the independent-column answer at
overhead sun (side leakage) and 70.6 % more at μ₀ = 0.5 (illuminated
side). **Not done**: the I3RC cases were not run.

### Phase 35 — Full urban radiosity
Replace the sky-view-factor approximation with a view-factor matrix
between every pair of urban surface patches (roofs, walls, ground),
solved by radiosity iteration for both shortwave inter-reflection and
longwave exchange. Includes specular treatment for glass.
**Validated by**: energy-balance closure; comparison against measured
façade temperatures from an urban field campaign (e.g. Basel BUBBLE).

**Built** (`src/Radiosity.jl`): facet decomposition, occlusion-tested
view factors, direct and iterative radiosity solves for shortwave and
longwave. **Measured**: view factors against closed forms 9.2e-15
(parallel) and 4.4e-15 (perpendicular); reciprocity 5.1e-17, closure
4.4e-16; canyon sky-view factor within 0.04–1.4 % of
√((H/W)²+1) − H/W. Radiative trapping: material albedo 0.2 gives an
effective canyon albedo of 0.152 / 0.127 / 0.106 / 0.095 at
H/W = 0.5 / 1 / 2 / 4, and the canyon floor's longwave loss falls to
0.63 / 0.42 / 0.25 / 0.14 of a flat plate. The façade-temperature
comparison against BUBBLE / CAPITOUL was done in Phase 49 (#30).
Glass is Lambertian here; the angle-dependent glazing is Phase 49 (#27).

### Phase 36 — Spectral surface albedo and BRDF
Wavelength-dependent albedo per material with a bidirectional
reflectance distribution function, so low-sun-angle reflection off
asphalt, glass, and vegetation is correct.
**Validated by**: comparison against spectral-albedo measurements for
common urban materials.

**Built** (`src/SurfaceOptics.jl`): 16 materials as reflectance curves
integrated per band, four BRDF models, Cox–Munk water, Wiscombe–Warren
snow. **Measured**: MODIS Ross-thick white-sky albedo 0.189186 against
the published 0.189184, Li-sparse-reciprocal −1.377660 against
−1.377622; reciprocity 8.3e-13 over 17 models × 2000 geometries;
Lambertian black- and white-sky albedo = ρ to 1e-12; water albedo 0.030
overhead, 0.294 at 80°. **Single source of truth** (same commit): the
three copies of the urban class materials (SurfaceOptics, Radiosity, the
urban example) became one derived mosaic; the derived class albedos
differ from the old hand-entered ones (street 0.126 vs 0.10, wall 0.227
vs 0.18). **Expectations corrected**: non-water classes are not
zenith-independent (vegetation brightens 1.53× toward grazing), and the
old magic-number assertions were replaced by tests of the derivation.

### Phase 37 — Aerosol and gas radiative feedback
Couple the aerosol and chemistry state into the optics: aerosol direct
effect, absorption by black carbon, and the radiative impact of ozone.
**Validated by**: aerosol-optical-depth-to-radiative-forcing comparison
against AERONET-constrained cases.

**Built** (`src/AerosolOptics.jl`): full Bohren–Huffman Mie series with
Wiscombe downward recursion, log-normal size integration, κ-Köhler
growth, OPAC mixtures. **Measured**: Mie Q_ext 2.881999 at x = 10,
m = 1.5 (Wiscombe 2.8820) and 8.033538e-6 at x = 0.101, m = 0.75
(8.033542e-6); black carbon ω 0.209 and MAC 7.33 m² g⁻¹ (Bond &
Bergstrom 0.2–0.3, 7.5 ± 1.2); Ångström exponent urban 1.44, desert
dust 0.069; critical surface albedo 0.1338 against 0.1398 from the
Chylek–Coakley formula. **Not done**: the AERONET forcing comparison,
and coupling to a prognostic chemistry state (the Berlin column
radiation has no aerosol until #142).

### Phase 38 — Solar-position and cloud-shadow accuracy
`src/SolarGeometry.jl`. Apparent solar position from the standard
astronomical series (Meeus ch. 25/22/12) including the equation of time,
nutation in longitude, apparent obliquity, aberration and topocentric
parallax; Bennett refraction scaled by station pressure and temperature;
Kasten-Young air mass; Bird clear-sky irradiance; Erbs and Reindl
direct/diffuse partitioning; and true 3-D cloud-shadow ray marching with
combined cloud-plus-building attenuation in a single march.
**Validated:** against the published NREL SPA reference case
(2003-10-17, Golden CO) to **Δzenith 0.0083°, Δazimuth 0.0036°**;
declination at both solstices and the March equinox to < 0.02°; equation
of time spanning −14.2 to +16.4 min with the correct zero crossings;
refraction 0.564° at the horizon and 1.0 arcmin at 45°; air mass exactly
1.0 at the zenith, 2.0 at 30° and 37.9 at the horizon (finite, where the
plane-parallel 1/μ₀ diverges); TOA irradiance 1316.7 W/m² at aphelion and
1408.0 at perihelion; Berlin summer-solstice maximum elevation within
0.5° of the analytic 90 − φ + ε and solar noon at the correct UTC hour.
The earlier sinusoidal declination it replaces is in error by up to
**0.79°** near the equinoxes.
**Accuracy is stated honestly**: this is Meeus-grade (~0.01°), not full
NREL SPA (±0.0003°). The full SPA needs ≈ 350 published VSOP87
coefficients; transcribing those from memory would risk silent errors,
which is worse than a documented 0.01°. `AbstractSolarPosition` is the
extension point for a table-carrying `SPAPosition`. For radiative
transfer 0.01° is not limiting — the solar disc itself subtends 0.53°.

---

# Clouds and microphysics

*Move from the Kessler/two-moment scaffold to schemes that reproduce
observed cloud microstructure and precipitation.*

**Status: all seven phases (39–45) are complete.** Phases 39, 41 and 42
landed before the issue tracker (`622184e`, `63fc020`); Phases 40 and
43–45 are issues #1–#14 (PRs #72–#89), plus the contract fix #82
(PR #91) and the P3 size-limit deduplication (PR #87). Milestone 4 has
15 closed issues and none open. The microphysics test set is 47,392 tests on
master at `2895fed`; about 29,850 of them are the 14,880-state enthalpy
sweep of #91. Shared contract: `src/MoistThermodynamics.jl`, which
every cloud scheme speaks and which also owns Köhler theory
(after three drifted copies were merged, see Phase 42) and, since #91,
the one enthalpy-conserving process adjustment.

### Phase 39 — Production saturation adjustment and thermodynamics
Exact moist thermodynamics: liquid-water potential temperature as a
prognostic, iterative saturation adjustment with the full latent-heat
temperature dependence, and consistent treatment of moist buoyancy
including condensate loading.
**Validated by**: moist-adiabatic ascent against analytic parcel theory;
conservation of moist static energy.

**✅ SHIPPED** (`src/MoistThermodynamics.jl`, `622184e`; enthalpy
adjustment #82 / PR #91). Temperature-dependent L(T) by Kirchhoff,
e_s as the analytic Clausius–Clapeyron integral of that same L(T),
c_pm with vapour and condensate, exponential θ_li, density temperature,
Bolton θ_e/θ_w, LCL, CAPE/CIN. **Measured**: Clausius–Clapeyron residual
5.2e-12 (liquid), 4.2e-12 (ice); e_s 611.21 Pa at 273.15 K, 2336.9 at
293.15 (ref. 2339), ice 103.28 at 253.15 (103.3); liquid–ice gap
27.00 Pa at 261.5 K; θ_li conserved to 9.9e-11 K over a 6 km saturated
ascent; moist lapse rate 3.78 K/km at 300 K, 9.53 at 220 K. The boiling
point comes out at 99,754 Pa (−1.55 %, the constant-c assumption,
recorded rather than tuned). **#91** added
`isobaric_saturation_adjustment` and `temperature_from_moist_enthalpy`:
over 14,880 states |Δh|/h ≤ 4.4e-16; the P3 heating ratio
c_pmΔT/(L_vΔq_c) went 0.99848 / 1.0121 / 1.06442 → 1.00082 / 1.00085 /
1.00097 at 1000 / 900 / 600 hPa.
**Bugs the tests caught**: `saturation_specific_humidity` returned
values > 1 past boiling (15.19 at 400 K), which made θ_e non-monotone
and broke the wet-bulb bisection; and (#82, found by the P3 tests of
#81) the contract's adjustment conserves θ_li, not enthalpy, adding
1.2 % latent heating at 900 hPa and 6.4 % at 600 hPa to every process
that used it. **Expectations that were wrong**: a 2nd-order finite
difference was measuring its own truncation (1.2e-8), not the physics;
moist static energy is not an invariant of isobaric condensation
(2.8 % drift over 6 km, 24 % of L·Δq per #85); #85's two `@test_broken`
compared fog with the θ_li *diagnosis*, which should not conserve
enthalpy; and the first #91 assertion had the sign of the
diagnosis/process gap wrong (the diagnosis condenses 1.75 % *less*).

### Phase 40 — P3 microphysics
Implement the Morrison-Milbrandt P3 scheme: predicted particle
properties, a single prognostic ice category with predicted rime mass
and density, four-moment rain.
**Validated by**: comparison against P3 reference implementation on
idealised cases; radar-reflectivity comparison for a squall line.

**✅ SHIPPED** (`src/P3Microphysics.jl`, #1–#4 / PRs #72, #76, #81, #79,
dedup PR #87). In-package Lanczos Γ and regularised incomplete Γ
(`gamma_inc_pq`, renamed to avoid a SpecialFunctions collision);
four-regime m(D)/A(D) with D_th, D_gr, D_cr solved in closed form from
F_rim and ρ_rim; Mitchell–Heymsfield fall speeds; KK2000 warm rain,
ventilated deposition/evaporation, Cober–List rime density, melting and
shedding; per-moment flux-form sedimentation with sub-stepping; rain and
ice reflectivity. **Measured**: Γ against MPFR 2.0e-14 (x ∈ (0, 30)),
incomplete Γ against MPFR 1.0e-13 (1 ≤ a ≤ 100); closed-form ρ_g against
a 256-bit solve 5.1e-16; mass continuous across the regime thresholds
(4.9e-12); total water over 3 regimes × 120 steps 7.8e-16; V(1 mm rain)
3.983 m/s (Gunn–Kinzer 4.03); Marshall–Palmer DSD at 5 mm/h 33.93 dBZ
against MP Z = 200R^1.6 34.19 dBZ; sedimentation mass error 5.7e-16 at
Courant 162. Rain mean-size limits are P3 v5.5.2's 10 μm–2 mm (#87),
held in one place. **Bugs the tests caught**: `gamma_psd_partial_moment`
went negative in narrow intervals (262 of 33,600 cases); underflow of
the small-L series at F_rim ≤ 1e-200; the MH05 turbulence correction
applied to every particle (graupel peaked at 11 m/s, then 0); N_c set
with the pre-adjustment density; a 0/0 in the limiter; sublimation and
rain evaporation limited separately (q_v/q_si overshot to 1.193); rain
riming number not cut with its mass; two-moment sedimentation left the
leading edge with mass and no number (D_m 4e4 m) until the λ_r limiter
was added; and (#87) two copies of the rain size limits that disagreed
(0.8 mm, P3's *old* value, vs 5 mm). **Expectations that were wrong**:
P(a, 1e-6a) < 1e-5 is false for a < 1; the issue's 35–45 dBZ at 5 mm/h
is too high (MP gives 34.2); a 1 % enthalpy expectation for the
θ_li adjustment (the cause of #82); a hand-computed median-volume
constant wrong in the 7th digit. **Not done**: a comparison against the
P3 reference implementation and a squall-line radar comparison.

### Phase 41 — Superdroplet Lagrangian microphysics
Monte-Carlo superdroplet method as an alternative to bulk schemes:
individual particle attributes, stochastic collision-coalescence,
explicit condensation growth. The reference against which bulk schemes
are evaluated.
**Validated by**: Golovin analytic collision-coalescence solution;
comparison against bin-microphysics reference for a warm cumulus.

**✅ SHIPPED** (`src/Superdroplet.jl`, `63fc020`). Shima et al. (2009)
all-or-nothing stochastic coalescence, Golovin / hydrodynamic /
gravitational kernels, implicit diffusional growth with Kelvin and Raoult
terms, Eulerian coupling. **Measured**: against Golovin's solution
(n₀ = 2²³ m⁻³, r̄ = 30.531 μm, 2¹⁵ superdroplets) L¹ error 0.037 at
1200 s, 0.080 at 2400 s, 0.118 at 3600 s; mass conserved to 5e-15; the
error falls with N_sd. **Bugs the tests caught**: a growing pure-water
droplet collapsed to the radius floor in one step (the backward-Euler
residual is U-shaped, and the bracket was chosen at the wrong end; 5 μm
at 13 % supersaturation went to 1e-10 m) — and the *evaporating* branch
of the same solver had the same collapse, found only in Phase 45 (#13:
Var(r²) was 61× the analytic value). **Not done**: the bin-microphysics
warm-cumulus comparison. A finite-N_sd coalescence bias of ~2.5 % at
N_sd = 2¹² remains (README limitations).

### Phase 42 — Aerosol-cloud interaction
Prognostic aerosol with size distribution, Köhler-theory activation to
cloud droplets, in-cloud processing, and regeneration on evaporation.
The first and second indirect effects done properly.
**Validated by**: activation-spectrum comparison against parcel-model
reference; Twomey-effect sensitivity against observations.

**✅ SHIPPED** (`src/AerosolCloud.jl`, `63fc020`). Abdul-Razzak & Ghan
activation, an explicit adiabatic parcel reference, prognostic aerosol
with activation / regeneration / in-cloud processing, Twomey and Albrecht
diagnostics. **Measured**: over w = 0.1–5 m/s and N_a = 100–1000 cm⁻³
ARG under-predicts S_max by 8–24 % (its known bias) and the activated
fraction agrees with the parcel to 1–25 % (mean 7 %). **One Köhler**:
AerosolOptics, Superdroplet and AerosolCloud each carried their own
Köhler theory; they had drifted (σ = 0.0728 N/m at 298 K, where 0.0720
is measured, 1.2 % in s_crit) and now delegate to MoistThermodynamics
(agreement 1.6e-12). **Bugs the tests caught**: the parcel model
reported zero activation always (it tested growth past r_crit at the
instant of S_max instead of s_crit < S_max); the Köhler critical-point
bracket inverted for nearly insoluble particles. **Not done**: the
Twomey-sensitivity comparison against observations.

### Phase 43 — Ice nucleation
Heterogeneous ice nucleation with prognostic ice-nucleating-particle
species (dust, biological), immersion and deposition freezing,
secondary ice production (Hallett-Mossop).
**Validated by**: comparison against cloud-chamber measurements;
mixed-phase-cloud persistence in an Arctic stratus case.

**✅ SHIPPED** (`src/IceNucleation.jl`, #5–#7 / PRs #73, #77, #88).
Prognostic dust and biological INP (available / consumed, closed
budget, Stokes–Cunningham settling); Niemand 2012, Ullrich 2017
(immersion, deposition), DeMott 2015 and Tobo 2013 nucleation; Hallett–
Mossop splintering in P3 (350 splinters per mg of rime, 10 μm
splinters, P3's rimer conditions); nucleation coupled into the P3 ice
category; a 1-D ISDAC-like mixed-phase stratus column. **Measured**:
INP budget closure 9.0e-16 over 1500 random steps; settling 3.49e-3 /
0.304 cm/s at 1 / 10 μm (Seinfeld & Pandis 3.5e-3 / 0.31); Niemand n_s
2.347e8 / 4.13e10 m⁻² at −20 / −30 °C; Ullrich/Niemand = e^0.424 at
every T; Hallett–Mossop raises N_i ×115 in a 20 min −5 °C riming
parcel; stratus initial LWP 14.38 g/m² (Paukert & Hoose ~14); the cloud
persists 8 h with ×1 and ×10 INP and glaciates at 2.17 h with ×100
(after #91 the 8 h LWP is 35.59 g/m² with ×1 INP). **Bugs the tests
caught**: singular nucleation moved 16,117 m⁻³ particles with zero mass;
dust took ρ_p = 1770 from the ARG default instead of 2600; below
T_icenuc deposition ran on INP immersed in drops (2.75e5 m⁻³ spurious
crystals); splinter mass was 2 μm instead of P3's 10 μm (125×); column
ice number exceeded the INP consumed by 3e-4 (density mismatch).
**Expectations that were wrong**: a reviewer's 85–90 % largest-first
mass fraction (94.0 %); the 25 μm drop restriction cutting splinters
below 0.1 % (0.59 %, broad spectrum); a 1e-14 tolerance below the
round-off of 253.15 − 273.15. **Not done**: cloud-chamber comparison;
the stratus IWP was not compared with the ISDAC LES (paper not
accessible), and the stratus longwave constants are not sourced.

### Phase 44 — Precipitation, fog and visibility
Sedimentation with a proper terminal-velocity spectrum, sub-grid
precipitation fraction, radiation fog with the correct droplet-
radiative feedback, and a visibility diagnostic for transport
applications.
**Validated by**: fog onset/dissipation timing against a field campaign;
drop-size-distribution comparison against disdrometer data.

**✅ SHIPPED** (#8–#11 / PRs #86, #89, #75, #85). Kessler rain now falls
at the mass-weighted speed of its Marshall–Palmer spectrum through the
conservative P3 sedimentation; sub-grid precipitation fraction with
maximum-random overlap and four in-cell regions (`src/p3/precip_fraction.jl`);
`src/Visibility.jl` (Koschmieder over Mie droplet, κ-Köhler haze and
Rayleigh extinction); `src/RadiationFog.jl` (enthalpy-conserving fog
adjustment, implicit dew into the surface budget, a 1-D `FogColumn`
with droplet-radiative feedback through the radiation stack).
**Measured**: Kessler fall speed within 0.1 % of Klemp & Wilhelmson at
1–3 g m⁻³; the old urban rain kernel created mass (×1.000678 in 20 s),
the new one conserves exactly; fog visibility 171 / 108 / 83 m at LWC
0.1 / 0.2 / 0.3 g m⁻³ (Kunkel 205 / 111 / 78); clear air 307 km
(~340 from the Bucholtz tables: the GasOptics λ⁻⁴ fit is ~10 % high);
ParisFog 2011: max LWP 92 g/m² (obs ~70), fog reaches the ground at
05:20 (obs 02:30), clears at 08:00 (obs ~10:00), C_LW 30–35 g m⁻² h⁻¹
(Waersted 40–70); with the droplet feedback off the fog never reaches
the ground. The old fog scheme heated a box model from 288 to 363 K in
24 h; the new one holds 288.0000 K. **Bugs the tests caught**: the
vapour heat-capacity term counted twice in the isenthalpic solve;
`fog_equilibrium` returned the bracket midpoint when Newton hit g = 0
exactly (82 of 200,000 states); gamma-spectrum Mie extinction biased
high for broad spectra (Q_eff 6.567 where ~2.12 is physical); the
gamma quadrature aliased Mie resonances; the per-region limiter applied
to summed rates (rain on 10 % of a cell removed 64.5 % of q_c); the fog
column's default dt = 10 s was not converged (onset moved by 55 min).
**Expectations that were wrong**: conserving moist static energy (24 %
error; `moist_enthalpy` added instead); |Q − 2| < 1e-3 at r = 5 mm; a
pointwise x^(−2/3) Mie slope; the bare f^0.32 autoconversion factor;
random overlap of a contiguous deck; the issue premise that the old
fog scheme blew up the urban run (it drained qv; the blow-up came from
the unstable moisture advection, replaced by WENO-Z5). **Not done**:
disdrometer DSD comparison; onset is 2.8 h late and dissipation ~2 h
early against ParisFog.

### Phase 45 — Turbulence-microphysics interaction
Sub-grid supersaturation fluctuations and their effect on droplet
growth and activation — the mechanism by which entrainment broadens
droplet spectra.
**Validated by**: spectral-broadening comparison against DNS and
superdroplet reference.

**✅ SHIPPED** (`src/TurbulentMicrophysics.jl`, #12–#14 / PRs #74, #78,
#84). Sub-grid saturation-deficit PDF (Gaussian and Larson double
Gaussian) from the local-equilibrium SGS variance; activation averaged
over that PDF; per-superdroplet Ornstein–Uhlenbeck supersaturation
fluctuations; a bulk spectral-moment scheme validated against the
superdroplet reference. **Measured**: σ_s → 0 reproduces the contract
adjustment exactly (`==`); condensate +0.16 / +0.48 / +1.58 % against
2-D quadrature of the exact adjustment at σ_qt = 0.1 / 0.3 / 1 g/kg;
N_d at w = 0.5 m/s 182.31 → 171.36 cm⁻³ as σ_S goes 0 → 0.1 %;
Var(r²) / analytic OU 0.991–1.017; bulk vs superdroplet ε within 1 %
for σ_S ≤ 0.1 % and 3.5–4.1 % low at 0.2 % (the gamma closure); width
growth σ_r² ∝ t^0.5142 (bulk) and t^0.5155 (superdroplet) against the
OU 0.5142 and the DNS t^½ law. **Bugs the tests caught**: Mellor's a
used the bare L/c_pm (≈1 % bias that did not vanish as σ → 0);
the evaporating superdroplet collapse of Phase 41; per-cell stepping
aged every OU fluctuation once per cell (τ became τ/K); the double
Gaussian returned NaN for |Sk| ≳ 1e8. **Expectations that were wrong**:
cloud fraction rising monotonically with skewness; O(Sk) rather than
O(Sk^{1/3}) convergence to the Gaussian; N_CCN = N at S = 10 %; ε
narrowing for 100 s without fluctuations (S̄ is used up); a guessed
ε ratio of 1.5 (measured 1.33); five more in #84, none of them code.
**Not done**: digitised DNS data (the DNS target is the analytic OU
broadening law).

---

# Land, urban and surface exchange

*Make the surface as good as the atmosphere. Currently the biggest
source of error in the urban runs.*

**Status: all eight phases (46–53) and the shared contract are
complete** — issues #15–#43 and #92, PRs #93–#122. Milestone 5 has 30
closed issues and none open. The surface test set is 19,326 tests (6 `@test_broken`) on
master at `2895fed`. Shared contract (`src/SurfaceExchange.jl`, #92 /
PR #93): material constants that read latent heats, c_p and g from
MoistThermodynamics rather than restating them; Monin–Obukhov
similarity as a dispatch hierarchy (Businger–Dyer / Paulson unstable,
Beljaars–Holtslag stable) solved for ζ by a bracketed Illinois
iteration; and the `AbstractSurface` interface (`surface_temperature`,
`roughness_lengths`, `step_surface!`, the RadiationCore albedo and
emissivity generics) that every land and urban surface implements. Measured:
neutral C_D 2.7e-16 from (κ/ln(z/z0))²; ψ against Paulson and Beljaars–
Holtslag to 6 digits; returned L against the Obukhov length of its own
fluxes 9.1e-15; a very stable case (ΔT = −10 K, U = 2 m/s) converges in
8 iterations. Expectation that was wrong: Ri_B = ζ C_D/C_H (the identity
is ζΦ_h/Φ_m²). Since the milestone closed, the Berlin case runs this contract
in every column (MOST + `SoilThermalColumn`, #129 / PR #158) — which
found a src bug recorded under Phase 46 — and the urban facet system
got a matrix-free GMRES solve (#138 / PR #163, Phase 49).

### Phase 46 — Multi-layer soil and hydrology
CLM-style soil column: 10+ layers with Richards-equation water
transport, temperature-dependent hydraulic conductivity, freeze-thaw,
and a water table.
**Validated by**: soil-temperature and moisture profiles against FLUXNET
site observations.

**✅ SHIPPED** (`src/SoilColumn.jl`, #15–#18 / PRs #98, #109, #107,
#117). CLM5 / CLM-exponential layering; Clapp–Hornberger, van
Genuchten–Mualem (with the Vogel/Ippisch air entry) and Gardner
hydraulics; mixed-form Richards solved by modified Picard with water
conserved by construction; temperature-dependent K via water viscosity;
Johansen thermal properties, enthalpy-form heat with freeze–thaw and
ice impedance; `SoilThermalColumn <: AbstractSurface`; a SIMTOP water
table / unconfined aquifer; the production column under the urban
example (replacing its three-layer soil). **Measured**: exact Gardner
infiltration 1.36e-4 → 3.86e-5 (50 → 100 layers, order 1.8); 30-day
mixed budget 1.3e-15 m per step; Neumann freezing front −1.15e-3
relative at 10 mm (first order); diurnal damping-depth amplitude 0.3693
vs 0.3703; energy residual over 40 freeze–thaw days ≤ 6.0e-8 J m⁻²;
water-table recession 1 step vs 2000 steps identical to 1e-12; soil +
aquifer drift 4.9e-14 m over 6 years; in the 24 h Berlin run the
production soil lowered street and roof Ts extremes by 0.3–0.6 K
(heat island 18.4 → 17.9 K). **Bugs the tests caught**: Picard chatter
near saturation (4,136 sub-steps per 5 min; fixed with air entry,
HYDRUS criteria and under-relaxation, now 5); the arithmetic interface
mean is first order (logarithmic now default); a 3-term viscosity
correlation 1.2 % off at 80 °C; example `SoilGrid` type piracy once
`SoilGrid` was exported. **Found later, in the Berlin integration
(#129)**: `SoilThermalColumn.step_surface!` returned the evaporation the
skin balance charged even when the soil delivered less (≈ 70 of
184 W m⁻² of street LE was water the soil never lost); fixed in src.
**Expectations that were wrong**: the Parlange sorptivity integral
stopping at the air-entry potential (12 % off); "Parlange within 1–2 %"
(8.5 % for constant D); an upward flux beyond Gardner's maximum
capillary rise; hydrostatic equilibrium of a dry column in 2 years (6
needed); sine forcing from t = 0 biasing a harmonic projection.
**Not done**: soil temperature / moisture profiles against FLUXNET
(PLUMBER2 carries no soil water; see Phase 47 for the flux comparison).

### Phase 47 — Vegetation photosynthesis and stomatal conductance
Farquhar photosynthesis coupled to Ball-Berry / Medlyn stomatal
conductance, so transpiration responds to light, CO₂, humidity and soil
water stress rather than a fixed availability factor.
**Validated by**: latent- and sensible-heat-flux comparison against
FLUXNET eddy-covariance towers across several biomes.

**✅ SHIPPED** (`src/Photosynthesis.jl`, `src/LandSurface.jl`, #19–#21 /
PRs #95, #105, #116). Farquhar C3 / Collatz C4 with Bernacchi and
Kattge–Knorr temperature responses and the CLM4.5 PFT table;
Ball–Berry and Medlyn stomata solved exactly for the leaf-surface
humidity, coupled with A_n in one bracketed residual; a two-big-leaf
vegetated column over `SoilThermalColumn` with Zeng root fractions and
CLM water stress. **Measured**: Kc/Ko/Γ* temperature shape against
Bernacchi 1.3e-4; A_n = g_c(c_a − c_i) over 3,456 cases 2.4e-11, all
converged; C3 c_i/c_a mean 0.719; against five FLUXNET2015 towers
(PLUMBER2, no calibration) hourly LE r = 0.83–0.94 and Rn r ≥ 0.98, LE
bias +7.9 (GF-Guy) to +87 W m⁻² (DE-Gri); the column does not beat the
PLUMBER "1lin" benchmark in RMSE at any site (as for the LSMs of Best et
al. 2015). **Bugs the tests caught**: the Medlyn deficit bound on air
rather than leaf-surface deficit ran away to g_s ≈ 10³ mol m⁻² s⁻¹
(now max(D_s, D_min)); β_t > 1 by one ulp; Newton cycling on the dew
switch (leaf residuals up to 40 W m⁻²); an oscillating outer canopy
iteration; NaN at LAI = 0; K_n transcribed as 0.11 instead of CLM4.5's
0.30 (daytime EF at DE-Hai 0.862 → 0.786). **Expectations that were
wrong**: peaked V_cmax agreeing with Bernacchi within 2 % (the Bonan-2011
deactivation removes 13 % at 25 °C); TPU limitation with CLM defaults
(impossible: J_max/4 < 3T_p); C4 c_i/c_a stable when c_a doubles; a
0.01 K outer-iteration bound set from one site. **Not done**: GPP and
soil moisture against observations; the FR-Pue drought response has the
right sign but the midday Bowen ratio is 3–5× too low.

### Phase 48 — Resolved plant canopy
Resolved-canopy drag with a leaf-area-density profile, implicit
treatment so it is unconditionally stable, radiation attenuation
through the canopy, and leaf-energy-balance for leaf temperature.
**Validated by**: within-canopy wind and temperature profiles against
forest-tower observations; wind-tunnel canopy-flow data.

**✅ SHIPPED** (`src/PlantCanopy.jl`, #22–#25 / PRs #96, #106, #113,
#119). LAD profiles integrated per layer; the exact implicit drag factor
1/(1 + Δt C_d a|U|) applied to the LES velocity (re-enabling park drag
in the urban example); Sellers two-stream canopy shortwave through
`TwoStream`, sunlit/shaded split, longwave; leaf energy balance with
Campbell–Norman or CLM boundary layers and coupled physiology; a 1-D
canopy column with a Harman–Finnigan mixing length capped at κz.
**Measured**: implicit drag against U₀/(1 + C_d a U₀ t) 7.7e-15 (explicit
Euler reaches 2.4e26 m/s at dt·C_d·a·U = 3.2); face fluxes against RK4
of Sellers' equations 3.4e-14 of the incident; canopy energy closure
4.7e-11; wind-tunnel U/U_h in the canopy within 0.021 (Brunet et al.
1994); CHATS crown U/U_h within 0.069 (leaves) / 0.093 (no leaves),
displacement height 8.36 / 6.64 m against 7.8 / 5.8; Hyytiälä U/u*
+26 % / +30 % above the canopy and ×0.49–0.55 in the trunk space; leaf
ΔT 1.9–3.4 K view-weighted against 1.7–4.8 K observed (Leuzinger &
Körner 2007, assumed forcing). **Bugs the tests caught**: Gauss–Legendre
integrated a Beta LAD only to 2e-5 (endpoint grading added); a TwoStream
resonance where 1 − k²μ₀² = 0 put the canopy two-stream 3–14 % off the
incident flux (a radiation fix); β₀ = 0/0 near χ_L = −0.4; a lagged-K
column two-cycling; a mixing length not capped at the wall (13 % of the
stress on the floor). The retired explicit drag used LAI for LAD — but
its recorded instability did not reproduce in the urban example.
**Expectations that were wrong**: half the Lalic–Mihailovic LAI above
z_m (47 %); Λ = 30 as a deep canopy at σ = 0.95; CHATS above-canopy
within 10 % (12.2 %); Hyytiälä within 30 % (29.7 %, too close to be a
guard). **Not done**: a 3-D LES with resolved drag against the
profiles; turbulence statistics; the CHATS leafless L_s/h (`@test_broken`).

### Phase 49 — Production urban canopy model
Full urban-canopy model matching or exceeding PALM's USM: per-facet
(roof / wall / ground) multi-layer wall conduction, window transmission,
anthropogenic heat with a real building-energy model, and
street-canyon-resolved radiation coupled to the radiation stack's radiosity.
**Validated by**: façade and surface temperatures against BUBBLE /
CAPITOUL urban field-campaign data.

**✅ SHIPPED** (`src/UrbanSurface.jl` + `BuildingEnergy` coupling,
#26–#30 / PRs #94, #102, #103, #121, #122; GMRES #138 / PR #163). A cited
16-material thermal table (optics derived from the SurfaceOptics library,
not restated); a θ-scheme conduction core; `UrbanFacet <: AbstractSurface`
with the skin balance linearised into a Robin condition; Fresnel /
Beer–Lambert glazing, multi-pane units with ISO 15099 gaps,
`WindowFacet`; implicit facet–facet longwave through the
radiosity response J; building heat release from the Phase 50 model
replacing the prescribed anthropogenic heat; TEB canyon wind and Rowley
convection for the validation driver. **Measured**: ISO 6946 composite
wall to 1e-10; asphalt admittance 1204.04 (Heisler & Brazel 1204);
3 mm clear float T/R/A 0.833/0.076/0.091 and SHGC 0.857 (NFRC 0.83/0.08/
0.09, 0.86); 6-13-6 U 2.707 / 2.547 W m⁻² K⁻¹ (air / argon; Vitro 2.669 /
2.555); implicit longwave coupling conserves to 7e-15 W/m² where the
lagged one leaks 14.7 W/m²; CAPITOUL road and wall mean temperature
+0.06 / −0.03 K over 2,304 hourly samples (TEB RMSE 3.09 / 1.22);
BUBBLE Q* 170.0 vs 179 W m⁻² (TEB 170); CAPITOUL winter building energy
use 65.7 W m⁻² vs the inventory's 58.6 (+12 %; TEB-BEM −15 %). GMRES
(#138): 1.4e-12 K against dense LU in 6–7 iterations; the Berlin scene
costs 0.31 s per step at 5,810 facets (LU 4.54 s). Each iteration is
O(n²): from 5,810 to 7,524 facets the solve time grew 0.112 → 0.196 s,
a ratio of 1.75 against (7524/5810)² = 1.68, at 7 iterations each (an
implied exponent of ln 1.75 / ln 1.295 ≈ 2.2 from these two sizes).
**Bugs the tests caught**: θ < ½ accepted (Ts ran away to 1e8 K);
the `h ≥ 0` check bypassed by the default constructor; a singular
inter-pane system at grazing sun; the example heated glass with the
light passing through it; the example's prescribed anthropogenic heat
never reached the air — it *cooled* the surfaces by up to 40 W/m².
**Expectations that were wrong**: 8 cm EPS carrying > 80 % of the ΔT
(76.9 %); ISO 13786 within 1 % at 2 cm cells (spatial truncation,
second order); the first step returning fluxes at the initial skin; a
6 mm pane held to the 3 mm angular tolerance; Fresnel reflectance
monotone from 0°; a linearised NFRC reference (+3.2 % → +1.4 %).
Incropera's asphalt k = 0.062 was rejected as a source. **Failures
recorded as `@test_broken`, not loosened**: CAPITOUL roof +3.93 K (TEB
+1.40; the observed roof is pitched and its construction unknown);
BUBBLE S↑ misses TEB's tolerance by 0.3 W m⁻² after the forcing was
corrected; BUBBLE daytime L↑ +39 W m⁻² (no vegetation in the source
area). **Not done**: per-sample RMSE (campaign series not public),
per-wall CAPITOUL values (figures only), a canyon-air model.

### Phase 50 — Building energy model coupling
Two-way coupled building interior: indoor air temperature, HVAC
demand and its exhaust heat, thermal mass, and occupancy schedules.
The AC-feedback loop that amplifies urban heat islands.
**Validated by**: measured building energy demand vs simulated for an
instrumented building.

**✅ SHIPPED** (`src/BuildingEnergy.jl`, #31–#33 / PRs #104, #111,
#118). One thermal zone with a prognostic air node solved implicitly
with every `ConductionColumn` / `UrbanFacet` of its envelope; the ISO
13790 5R1C reference; ideal-loads HVAC with the EnergyPlus DX
biquadratic COP and capacity curves, condenser exhaust (air-cooled,
evaporative, boiler, heat pump), dehumidification; DOE medium-office
schedules and internal gains. **Measured**: concrete-slab first mode
τ₁ 42.43 h against the plane-wall 42.42 h; zone energy over 10 days
3.3e-14; AC–canyon feedback amplification 1/(1 − g) = 1.3621 exactly;
ANSI/ASHRAE 140-2020 (BESTEST, Denver TMY3): incident solar 5/5 inside
the reference range with Perez (2/5 isotropic), annual loads 19 of 20
inside the six-program range (Case 395 heating 0.6 % low), peaks 16 of
20. **Bugs the tests caught**: a driver albedo cap that made walls absorb
50 % of the sun; a COP check dividing by the wrong quantity.
**Expectations that were wrong**: a 40-day spin-up for a light box
(slowest mode τ ≈ 18 d); monotone DX capacity over the whole fitted range
(the packaged curve peaks at 19.6 °C); BESTEST peak count ≥ 36 of 40
(there are 20); the Standard's combined exterior coefficients being
adequate (they put 6 of 10 heating loads below range). **Not met**: the
phase target — no measured demand of an instrumented building was
available, and BESTEST is program-to-program.

### Phase 51 — Snow, ice and frozen surfaces
Multi-layer snowpack with compaction, metamorphism, albedo ageing,
liquid-water percolation and refreezing; lake and sea-ice
thermodynamics.
**Validated by**: snow-depth and albedo evolution against SnowMIP
reference sites.

**✅ SHIPPED** (`src/Snowpack.jl`, `src/IceThermodynamics.jl`, #34–#37 /
PRs #108, #114, #115, #99). Layered snowpack with Anderson / Vionnet
compaction and four conductivity laws, mass/depth/enthalpy-conserving
relayering, `AbstractSurface`; Flanner–Zender dry ageing from CLM's own
SNICAR table (read with the in-package NetCDF reader), Brun wet ageing,
albedo through SurfaceOptics' Wiscombe–Warren snow; Coléou–Lesaffre
retention, bucket and Colbeck kinematic-wave percolation, enthalpy-exact
refreezing; Bitz–Lipscomb (1999) brine-pocket sea and lake ice.
**Measured**: 12-day storm/melt run mass 1.6e-13 kg m⁻², enthalpy
1.4e-7 J m⁻²; seasonal bulk density within 25 % of Sturm et al. (2010)
alpine climatology (−17 / −19 % by late March); grain growth within
0.1 % of a literal port of CTSM `SnowAge_grain`; Summit SSA after 37 d
23.8 m² kg⁻¹ (observed 23 ± 3, assumed temperature range); kinematic
front arrival 7.35 h vs 7.37 h analytic; Stefan ice growth 0.035 %;
MU71 sea ice first annual cycle within 0.3 cm of the BL99 reference
code, 100-yr equilibrium 264 cm vs BL99 Table 1's 281 (the 6 % offset
traced to one boundary stencil in the reference). **Bugs the tests
caught**: a layer merge at T_f lost enthalpy; a pack sublimating away
returned more vapour than it held (0.29 vs 0.20 kg m⁻²); trilinear
interpolation of the jointly fitted SNICAR parameters (484 μm where the
nodes give 123–181); gravity drainage counting the first refreeze
twice; 0/0 brine heat capacity for fresh ice at 0 °C; an albedo jump
inside the surface Newton. **Expectations that were wrong**: fresh-snow
depth exactly the fall depth (it settles 0.75 % within the step); a
hand-computed wet-growth radius; the Neumann small-St expansion
(1 − St/6, not /12); a basal-melt rate formula that holds only for
accretion. **Not done**: SnowMIP sites; albedo of old melting snow
(needs impurity deposition).

### Phase 52 — Water surfaces and wave coupling
Lake and coastal-sea surface with a mixed-layer model, wave-dependent
roughness (Charnock and beyond), sea-spray heat flux, and a diurnal
warm-layer scheme.
**Validated by**: sea-surface-temperature diurnal cycle against buoy
observations; air-sea flux comparison against COARE.

**✅ SHIPPED** (`src/OceanMixedLayer.jl`, `src/AirSeaFlux.jl`, #38–#40 /
PRs #97, #100, #112). Layered water column with EOS-80, Jerlov
shortwave, Kraus–Turner entrainment, Fairall cool skin and Fairall /
Zeng–Beljaars warm layers; Charnock, COARE 3.0/3.5, wave-age and
sea-state roughness with COARE stability and gustiness; Andreas (2015)
sea spray with the microphysics delegated to MoistThermodynamics and
Superdroplet; rain heat flux; the full COARE loop. **Measured**: cool
skin against the COARE 3.6 code on its ship data 9.2e-5 K (653 records);
fresh-water density maximum 3.9817 °C; Kato–Phillips deepening −0.017 %;
COARE 3.0 C_D10N at 20 m/s 2.071 × 10⁻³ (Fairall 2003: 2.06); against
the NOAA COARE 3.5 reference file u* 0.27 %, LE 0.25 %, H 1.7 % (the
moist c_pm; 0.23 % with dry c_p); full algorithm LE within 1.51 W m⁻²;
TOGA COARE peak afternoon warming (Fairall 1996a, 42 days) r = 0.953,
RMSE 0.246 K with the Fairall scheme, 0.50 K with Zeng–Beljaars.
**Bugs the tests caught**: the u* bracket of `neutral_transfer_coefficients`
(negative COARE 3.5 α below 2.9 m/s, a second unstable Oost root); a
0/0 in the gustiness convergence check. **Expectations that were
wrong**: COARE's own shipped output file (it disagrees with COARE's code
by 0.24 K); a 1e-12 energy tolerance against a 4e10 J m⁻² store; free
convection at U = 0.5 m/s (not ≪ w_g); spray tolerances against
digitised curves. **Open**: the default fully developed sea makes the
spray latent flux 1.76× Andreas's figure at 25 m/s — pass a `SeaState`.

### Phase 53 — Surface heterogeneity and blending
Sub-grid tile approach with a blending-height formulation so a single
grid cell can contain multiple surface types and still produce the
right aggregate flux and the right internal boundary layers.
**Validated by**: heterogeneous-surface LES against homogeneous-tile
aggregation; observations from patchy terrain campaigns.

**✅ SHIPPED** (`src/SurfaceTiles.jl`, #41–#43 / PRs #101, #110, #120).
`SurfaceMosaic` (flux aggregation over any `AbstractSurface` tiles,
nestable), parameter aggregation for comparison, blending-height
effective roughness with Mason / Claussen / Bou-Zeid / IBL closures and
`BlendedMosaic`; a patchy-strip LES driver. **Measured**: a single tile
is bit-identical to the bare surface; parameter aggregation gets the
sign of H wrong for a stable + unstable pair (+43.6 vs −24.1 W m⁻²);
effective z0 against the Bou-Zeid et al. (2007) LES within 15 % in all 8
cases (rms ln error 0.074; Mason 0.052; log mean 0.193); a priori, tiles
with patch-mean air are within 2.5 % of the resolved strip-LES flux and
parameter aggregation 22–37 % low; a posteriori the tiled LES is +4.2 %
/ −8.0 % (calm) and +1.4 % (Ug = 5 m/s). **Bugs the tests caught**: an
export collision with SurfaceOptics' `TiledSurface` (renamed
`SurfaceMosaic`); a signed zero breaking bit-identity; the interrupted
`blended_forcing` draft's closure overwriting its own residual (every
Jacobian column zero). **Expectations that were wrong**: fractions summing
exactly to 1; Claussen blending higher than Bou-Zeid; single-tile split
exact under stability; the domain-mean wind vector as grid-cell forcing
(+16–34 % in free convection, uniform surface included). **Not done**:
field observations (LITFASS); the published strip LES was compared only
qualitatively.

---

# Turbulence closure

*The SGS model is the heart of LES. Make it defensible.*

**Status: all six phases (54–59) are merged.** That is the 29 issues
#167–#195, merged as PRs #203–#231 on 2026-10-04/05, plus six follow-ups
from the Technical-debt milestone (#232–#237 / PRs #240–#244, #271).
Milestone 6 is still open: its six phase tracking issues #44–#49 have
no open work left, but agents do not close issues, so closing them and
the milestone is left to the maintainer. The turbulence-closure work adds five src modules:
`TurbulenceStability`, `WallModels`, `LargeScaleForcing`, `PrognosticSGS`
and `Transition`. The closures themselves live in `Operators`. The groundwork from before it (#132 /
PR #160) moved the local Germano–Lilly scaffold into `Operators` after
its M_ij sign was found wrong (C_s = 0 everywhere).

**Berlin defaults after the milestone.** The `:sgs` slot runs the Deardorff
TKE closure (`:tke`), with the stress-form divergence, the diagonal
scalar diffusivity, and qv and every tracer diffused with κ_t. Every
eddy-viscosity provider is wrapped in `StabilityCorrected` (`:brown94`).
TKE handles stratification itself, so it is not wrapped, by design
(#233). The ground wall model takes a time-filtered input. The facade
wall model is on, and TKE shear production at the facades matches the
facade stress (#232). The `:geostrophic` coupling uses the geostrophic
pressure gradient, with the sponge retargeted at U_g. `turbulence_diag`
writes N², Ri_g, f_m, γ, the supercritical fraction and the
relaminarised fraction. The surface exchange evaluates the maximum
sustainable heat flux in every column. The following are options, not
defaults: LASD (the default of the `:dynamic` provider), AMD, Vreman,
σ, the Reynolds-stress closure, the isentropic tensor, the TBLE wall
model and the dynamic z0.

**Open after the milestone** (Technical debt, not the Turbulence closure milestone):
#238, Berlin nights are implausible (37 °C at 03:00, heat island
27.7 K, u* ≈ 0.001 m/s); #239, N² ignores condensate loading and
saturated stability.

### Phase 54 — Production dynamic and scale-dependent SGS models
Lagrangian-averaged scale-dependent dynamic Smagorinsky (Bou-Zeid,
Meneveau & Parlange), fully validated, replacing the local
Germano-Lilly scaffold. Includes the correct near-wall behaviour
without ad-hoc damping.
**Validated by**: log-law recovery with no wall-damping function;
resolved-scale spectra against DNS.

**✅ SHIPPED** (`src/operators/sgs_closures.jl` and the Lagrangian
model in `Operators`, #167, #168, #173, #174, #186 / PRs #207, #205,
#212, #216, #224). What was built:
- **SGS closure contract.** `AbstractSGSClosure`, a κ_t field and
  θ-aware closures. A closure commits once per accepted step on every
  core (Euler, RK3, IMEX, the anelastic core, and adaptive stepping
  with rollback on a rejected try).
- **LES toolbox.** Box, Gaussian and spectral filters, the isotropic
  energy spectrum, a solenoidal initialiser and the Comte-Bellot &
  Corrsin (CBC) benchmark.
- **Lagrangian dynamic Smagorinsky** (Meneveau, Lund & Cabot 1996),
  aware of the solid mask.
- **Scale-dependent LASD** (Bou-Zeid et al. 2005), with a 5-cell second
  filter and a coefficient cadence.
- **Channel and half-channel case builders**, with Φ_m, total-stress
  and E11 diagnostics.

**Measured**:
- The contract reproduces the old Smagorinsky paths bit for bit over
  50 steps of three steppers, at 0.9 % overhead.
- CBC at 32³, dynamic: K98/K42 +13.2 % and K171/K42 −3.6 %. At 64³:
  −4.9 % and −17.6 %.
- In isotropic turbulence the Lagrangian C_s is 0.1500, against the
  plane-averaged 0.1539, at t = 3.
- Half channel with Smagorinsky 0.17: the near-wall Φ_m overshoot
  peaks at 1.629 (1.45–1.65 over 12 seeds).
- LASD at 24³ with the 2Δ input filter, without wall damping: Φ_m stays
  in 0.934–1.151 over 0.05–0.3 H (0.961–1.094 at 48³), and ⟨τ_w⟩ is
  1.028 u*². Smagorinsky 0.1 at 32³ gives 0.965–1.589.
- Berlin `:dynamic` with LASD at cadence 4: C_s 0.095 in the canopy
  and 0.089 in the free stream (LASI: 0.037 / 0.033), at 25–33 ms per
  step on the 64×64×32 mesh.

**Bugs the tests caught**:
- The spectrum measured the centre average of the staggered velocities.
  As a result, fresh CBC fields carried 22.0 % too much kinetic energy
  (13.7 % at 64³).
- The masked Germano filter allocated 125 MB per call.
- The 5-cell LASD filter reached into the mirrored halo at a free-slip
  wall. That gave a spurious β ≈ 1 at the second level.

**Expectations that were wrong**:
- The 32³ CBC pass of K98 within ±10 % depended on the spectrum defect.
  The measured value is now +13.2 %, kept as `@test_broken`.
- β was tested as an arithmetic mean. The right statistics are the
  geometric mean (1.03–1.06) and the median.
- "LASD C_s within 10 % of LASI" fails on the domain mean (+10.5 to
  +14.2 %), because C²_Δ is convex in a noisy φ.
- "β → 1 above z/H ≈ 0.2": β crosses 1 about 10 cells from the wall,
  wherever that falls in z/H.
- Exactly zero ν_t in laminar shear for the Lagrangian model: it decays
  geometrically, to 1.4e-8 of the start.

**Not met**:
- Resolved spectra were compared only with CBC, which is an experiment,
  not DNS.
- The k⁻⁵ᐟ³ range of E11 is not reproduced at 32³–48³ (fitted slopes
  −4.27 and −3.48).
- K171 within ±10 % fails for every closure but AMD at 32³.
- The −17.6 % K171 deficit at 64³ is unexplained.
- LASD is the default of a non-default provider.

### Phase 55 — Anisotropic and tensorial eddy viscosity
Anisotropic minimum-dissipation (AMD) and Vreman models, plus a full
tensorial eddy-diffusivity formulation for strongly stratified flow
where vertical and horizontal mixing differ by orders of magnitude.
**Validated by**: stably stratified channel against DNS; GABLS1
intercomparison.

**✅ SHIPPED** (`Operators`, `WallModels` channel cases,
`examples/sbl/driver.jl`; #175, #181, #176, #187, #191 / PRs #214, #218,
#215, #225, #228). What was built:
- **Stress-form SGS divergence** ∂_j(2ν_ij S_ij) with separate (ν_h, ν_v),
  a generic `add_stress_divergence!`, an `sgs_stress` keyword on every
  core, the IMEX split, and the stress-form diffusive CFL limit.
- **Tensorial scalar diffusion**: a diagonal (κ_h, κ_v) operator, a
  general symmetric K_ij in triad form, and a rotated isentropic tensor
  with a slope taper.
- **AMD** (with its scalar and buoyancy terms) and **Vreman** closures.
- **`StabilityCorrected`**, which wraps any eddy-viscosity closure with
  Ri_g stability functions from `TurbulenceStability`, anisotropic
  length scales and Pr_t(Ri).
- **A shared stable-boundary-layer (SBL) driver**, used by GABLS1 and the
  stratified channel. `BusingerDyerLogLinear` moved into SurfaceExchange
  so that there is one copy.

**Measured**:
- Stress form: solid-body rotation gives zero tendency (the Laplacian
  form gives 2.33), and the discrete dissipation matches to 2.8e-16.
- The stress form's spectral radius is 8νΣΔ⁻², twice the Laplacian's.
  The corrected CFL limit costs one extra step in 1,647 over a Berlin
  hour.
- The isentropic tensor spreads a blob at 0.9977 / 1.0067 of 2κt along /
  across the isentropes.
- AMD is exactly zero in laminar shear and reaches zero at the critical
  N²*. In CBC, AMD gives +9.9 % / −4.7 %, the first closure to meet
  ±10 % on K171. Vreman gives +4.9 % / −12.7 %.
- `StabilityCorrected`:
  - Neutral case bit for bit with the base closure; analytic
    stability functions to 6.1e-14.
  - Internal wave: ω within 0.46 % of N k_x/|k|, with ν_v ≡ 0.
  - Breaking wave: 23.4 % of the energy lost in 3 periods, against
    5.5 % resolved only.
  - Berlin: ν_v in the capping inversion is 2.3e-5 m²/s, against
    8.1e-3 uncorrected. The flagship night mean κ_v/κ_h is 0.016.
    Cost: +8 % per step, +5 % over the 21 h run.
- GABLS1 at 64³, hours 8–9, Brown 1994:
  - h 196.4 m (reference about 200).
  - u* 0.284 m/s, 0.4 % above the range's upper end 0.283.
  - w'θ'₀ −0.0134 K m/s, inside −0.0148 to −0.0094.
  - Low-level jet 9.49 m/s at 191 m.
  - At 32³, u* is 4–7 % above the range.
- Stably stratified channel at Re_τ = 180, 32×32×48:
  - U_b⁺ within 4.4 % of García-Villalba & del Álamo neutral, and
    within 1.4 % / 2.7 % at Ri_τ = 18.
  - Nu is over-predicted by 15 % and 9 % / 14 %.
  - Nu(18)/Nu(0) is 0.63 / 0.67, against the DNS 0.667.

**Bugs the tests caught**:
- Vreman's B_β written as a sum of minors left 1e-10 m²/s on a
  two-component shear. It is now in Cauchy–Binet form.
- IMEX froze θ's κ_t for qv under AMD.
- `step!` diffused θ with κ_h only while `step_imex!` used (κ_h, κ_v),
  a relative error of 1.0.
- The Laplacian-form CFL limit was 2× too permissive for the unprojected
  stress form.

**Expectations that were wrong**:
- "AMD ν_t falls monotonically with N²" holds only where ∂w/∂z < 0.
- "Breaking wave loses more than 30 %": measured 23.4 %.
- "ν_v = ν_h where Ri ≤ 0" ignored the anisotropic length scales.
- "The corrected closure gives a lower GABLS1 BL": it is deeper, 196.4 m
  against 191.1 m.
- "The corrected channel is closer to the DNS": it is not, at CI
  resolution.
- A Berlin dt_diff check used the Laplacian-form limit for the stress
  form.

**Not met**:
- At Ri_τ = 60 the channel relaminarises (Nu 1.23, against the DNS
  2.80). We attribute this to resolution.
- The DNS at Re_τ = 550 is unaffordable wall-resolved.
- AMD's CBC K98 is asserted to ±13 % (the reference spectrum is
  uncertain by about 3 %).

### Phase 56 — Prognostic SGS energy and stress
Deardorff prognostic TKE and, beyond it, a full differential
Reynolds-stress SGS model for the cases where eddy viscosity fails
(rapid rotation, strong stratification, flow around sharp obstacles).
**Validated by**: SGS-stress comparison against filtered DNS fields.

**✅ SHIPPED** (`src/PrognosticSGS.jl`, the a-priori harness in
`Operators`; #177, #182, #178, #188, #192 / PRs #213, #219, #208, #223,
#227; follow-ups #232 / PR #242, #235 / PR #244). What was built:
- **SGS diffusion of qv and of every Berlin tracer** with κ_t, aware of
  the solid mask. Before this, that physics was missing.
- **Deardorff / Moeng TKE closure**, plus `wall_distance`. It is the
  Berlin `:sgs` default.
- **A-priori harness**: exact subfilter stress; Smagorinsky, Bardina and
  gradient model stresses; metrics. It has a 96³ DNS fixture in the
  suite and an offline 256³ record.
- **Differential SGS Reynolds-stress closure** (Deardorff 1973; LRR-IP;
  Schumann realizability), as an option.
- **Validation against filtered DNS** (isotropic, rotating and
  stratified decay) and a surface-mounted cube.

**Measured**:
- qv diffusion matches the discrete factor to 1.8e-15. The old code
  gave a relative error of 2.3.
- Berlin with tracer diffusion: ground-level pollutant maximum −27 %,
  cost +4.5 %.
- TKE closure:
  - Neutral equilibrium C_s is 0.194413 = c_k^{3/4} c_ε^{-1/4}.
  - Ri_c is 0.23313.
  - Energy exchange closes to 2e-16.
  - 64³ CBL: σ_w² peaks at 0.459 w*² at 0.351 z_i, against
    Lenschow et al.'s 0.466 at 0.313 (RMS difference 0.035).
- A priori at 256³ (Re_λ 78.6): Smagorinsky ρ_Π(4h) 0.832, matched C_s
  0.099, Bardina ρ_Π 0.956, backscatter 21.1 %.
- RSM: the shear equilibrium is exact to 1e-10, and the rotation
  invariants hold to 1e-13.
- Against filtered DNS (48³ DNS, 12³ LES):
  - Rotating decay, KE error: RSM +6.4 %, Smagorinsky −11.9 %.
    At 96³ the RSM error is 1.4 %.
  - Stratified decay, τ₃₃/τ₁₁: DNS 0.700, RSM 0.762, TKE 1.002.
- Surface-mounted cube, x_R/H: Smagorinsky 1.82, TKE 1.80, RSM 1.97,
  against 1.612 (Martinuzzi & Tropea 1993).
- In Berlin, RSM costs about 2× TKE, so TKE stays the default.
- With the facade fix (#232): canopy shear production +71 %,
  near-facade ν_t +5.3 %.

**Bugs the tests caught**:
- `step!` and `step_rk3!` never diffused qv, and moist `step_rk3!`
  advanced qv with three Euler steps.
- A backward-Euler RSM source flipped the sign of growing modes. The
  realizability projection then added 3.96e4 m⁵/s² of energy.
- The literal `2e0` in the constructor made every RSM closure built
  with `e0` start at e = 1 m²/s².
- `germano_lilly_cs` raised a DomainError.
- A Berlin `sgs_dissipation` shadowed an export.
- TKE production at wall-modelled facades booked a stencil flux the step
  does not apply: 27.3 % of canopy production (#232).
- The 256³ record failed Parseval (0.9967) because of the pre-#205
  spectrum (#235).

**Expectations that were wrong**:
- "ρ_Π < 0.85 at 48³": Re_λ 32 is too low.
- "ρ_Π falls monotonically with Re_λ": dropped on the maintainer's
  decision.
- "TKE and RSM isotropic KE within 10 %": measured −11.6 % / −11.7 %.
- "TKE e within 25 % of ½τ_kk": measured 2.8×.
- "RSM heat flux closer to the DNS than TKE's": the reverse holds.
- "ρ(τ₃₃/τ₁₁) > 0.5": holds only up to T/4.
- "Σ P_s V = −stencil sink with the facade model on" is physically
  wrong.

**Not met**:
- TKE over-dissipates the CBC decay (−18.5 % / −34.4 %).
- The cube's x_F is far too short (0.21–0.51, against 1.04).
- RSM cannot run under IMEX.
- RSM facade production has the inconsistency #232 fixed for TKE.

### Phase 57 — Wall models for high Reynolds number
Equilibrium and non-equilibrium wall models, dynamic wall models, and a
proper treatment of the log-layer mismatch problem. Required for
wall-modelled LES of realistic urban flow.
**Validated by**: Kim-Moin-Moser channel at several Reynolds numbers;
comparison against wall-resolved reference.

**✅ SHIPPED** (`src/WallModels.jl`, #169, #183, #179, #189, #190 / PRs
#204, #220, #211, #221, #222). What was built:
- **Equilibrium wall laws** (Spalding, rough log law, MOST), delegating
  to SurfaceExchange, and an implicit `apply_wall_stress!` (local,
  Schumann, Moeng). The Berlin ground stress is evaluated every step.
- **Matching height, time filter and 2Δ input filter**
  (`MatchingInput`). The Berlin ground takes the time-filtered input.
- **Wall model on IBM and cut-cell walls.** Berlin
  `wall_model_facades` is on by default.
- **TBLE non-equilibrium wall model** with dynamic κ, as an option.
- **Anderson–Meneveau dynamic z0**, as an option.

**Measured**:
- Spalding round trip 8.7e-15. Momentum is conserved to 1.3e-15, and
  the operator is stable at 100× the explicit limit.
- Log-layer mismatch against Lee & Moser, U_b⁺ error (first-cell
  matching → matching height plus filters):

  | Re_τ | 182 | 544 | 1000 | 5186 |
  | :-- | :-- | :-- | :-- | :-- |
  | U_b⁺ error | +5.58 → −1.24 % | −0.50 → −1.19 % | −0.59 → −1.40 % | −1.41 → −1.45 % |

  Φ_m(Δz) goes from 1.70 to 1.47 at Re_τ 182 and from 1.24 to 0.93 at
  5186.
- IBM walls match z-walls to 3e-15, at U_b⁺ 19.18 (−4.0 % against
  Lee & Moser). Without the model it is 13.86.
- The 30° cut-cell channel gives ⟨τ_w⟩ 0.990.
- Berlin facades carry 9.3 % of the surface drag. The stencil drag
  they replace was 33× the modelled drag.
- TBLE:
  - Stokes I converges at second order.
  - After a sudden tripling of dp/dx, τ_w gets half-way in
    5.965 δ/u_τ, against the resolved reference 5.960 (equilibrium:
    6.255).
  - In Berlin, the street stresses differ from the equilibrium ones
    by more than 30 % in 77 % of columns, at +9.4 % stage cost.
- Dynamic z0: u* is grid-invariant to within 4.5 % from 32² to 16²,
  against +9 % to +10.6 % with a fixed α.

**Bugs the tests caught**:
- With `wind` given, the wall-stress operator scaled the stress by
  U1/U_in. The #169 test had encoded the wrong formula.
- Matching the dynamic κ on the stress is degenerate.
- TBLE `:full` with centred convection was unstable.

**Expectations that were wrong**:
- "|u⁺ − log law| < 1e-3 at y⁺ = 1e4": Spalding itself differs by
  2.506e-3 there.
- "2× lower U_b⁺ error at Re_τ 1000": first-cell matching is already
  within 1 %.
- "The matching height lowers the rough-wall Φ_m peak": the peak is the
  Smagorinsky overshoot, which the wall model does not control.
- "TBLE within 1 % of Spalding at h⁺ = 30": −3.0 %, because the buffer
  layer differs.
- "Stencil drag more than 10× the modelled drag": measured 3.8×.

**Not met**:
- The 2× reduction at Re_τ 1000.
- Dynamic z0 of a cube array is 5.8× Macdonald (1998).
- Berlin keeps first-cell matching, because 3 cells is in the canyon.
- The cut-cell wall model is not in Berlin.

### Phase 58 — Stable-boundary-layer and intermittent turbulence
The hardest regime in boundary-layer meteorology: very stable
stratification, intermittent turbulence, gravity-wave-turbulence
interaction, and the correct behaviour as the flow relaminarises.
**Validated by**: GABLS2, GABLS3, GABLS4 intercomparison cases;
CASES-99 observations.

**✅ SHIPPED** (`src/TurbulenceStability.jl`, `src/LargeScaleForcing.jl`,
the MSHF in `SurfaceExchange`, `examples/sbl/`; #170, #171, #193, #194 /
PRs #203, #210, #229, #231; follow-up #237 / PR #243). What was built:
- **Stability functions of Ri_g** (Lilly, Brown et al. 1994, short and
  long tail, MOST-consistent) and N² / Ri_g fields. The fog column and
  both cores delegate to them.
- **`LargeScaleForcing`**: geostrophic pressure gradient, nudging,
  advective tendencies and a T_s series. It replaces both example
  copies.
- **Maximum sustainable heat flux** (Van de Wiel et al. 2012) and a LES
  collapse test.
- **GABLS2, GABLS3, GABLS4 and CASES-99** on the SBL driver.

**Measured**:
- MOST round trip of K_m 5.4e-11. The log-linear critical Ri is
  0.199996.
- Ekman veer 44.74°. Inertial oscillation 6.5e-8 of the
  ageostrophic speed.
- MSHF: the closed form matches to 1e-8.
- Collapse LES:
  - At H = 2 H_max, the lowest third keeps 2.8 % of the neutral TKE,
    with γ = 0.
  - At H = 0.5 H_max it keeps 83.2 %.
  - The first-level wind is 1.253–1.257 U₀, against the predicted
    2^{1/3} = 1.260.
- GABLS3:
  - CI grid (25 m): jet 9.07 m/s at 94.9 m. Envelope 7.8–9.3 m/s at
    100–130 m.
  - 6.25 m: jet 7.70 m/s at 126.8 m; U(10 m) 2.62 m/s, inside
    2.41–2.83.
  - 3.125 m: profiles on the LES median.
- GABLS4 exp. 3 at 2 m, against the six 1 m LES:
  - u* 0.092 m/s and H −11.0 W/m² are inside.
  - h 30.7 m is outside (14.5–19.5).
  - L 4.6 m is outside (6.5–9.9).
  - θ* 0.125 K is outside (0.07–0.12).
- GABLS2 at 30 h, corrected: H −23.0 W/m², u* 0.192 m/s and h 100 m
  are inside the SCM ranges. Uncorrected, H and h are outside.
- CASES-99 hockey stick, at 30 m only: V_s 7.56 m/s and slope 0.17,
  against 6.5 m/s and about 0.25 (Sun et al. 2012).
- Berlin at 03:00: every stable column is beyond the MSHF point. The
  night has no jet and u* 0.001–0.002 m/s (the very stable regime 1).

**Bugs the tests caught**:
- Random initial perturbations were not divergence-free, which gave θ
  swings of ±20 K.
- Under the pressure gradient, the Berlin sponge drained the mean wind
  to 0.12 m/s. Nudging had hidden this.
- The GABLS4 reference counted CSIRO's 3.9 / 1.17 m run as 1 m (#237).
- Couvreux et al. (2020) Table 5 is shifted by one hour.

**Expectations that were wrong**:
- "N² within 2× of the dry value": the vapour term takes about 2/3 off.
- "A nocturnal low-level jet in Berlin": the 200 m domain is too
  shallow.
- "One H_max separates both regimes": the bulk value is 0.032 × the
  wall value.
- "Whole-domain TKE below 5 % at 20 H/u*": it gets there at 40 H/u*.
- "GABLS3 jet height at CI resolution": 5–7 m low, kept as
  `@test_broken`.
- The GABLS4 h range of 14.5–25 m.

**Not met**:
- GABLS4 h, L and θ*.
- The CASES-99 shape at 10 m and 50 m.
- GABLS4 was not run at 1 m, and GABLS3 not on the 800 m cube.
- All the runs are dry.
- Calm soil columns deliver more than H_max (linearised skin).

### Phase 59 — Transition and laminar-turbulent coexistence
Handle flows that are partly laminar: nocturnal boundary layers,
low-Reynolds regions in a cut-cell wake, and the transition process
itself.
**Validated by**: transition location in a flat-plate boundary layer;
relaminarisation in a strongly stable case.

**✅ SHIPPED** (`src/Transition.jl`, `NoSlipWall` in BoundaryConditions,
the σ-model in `Operators`; #172, #184, #180, #185, #195 / PRs #209,
#217, #206, #226, #230; follow-up #236 / PR #271). What was built:
- **No-slip and moving walls**, with IMEX support.
- **σ-model** (Nicoud et al. 2011), with closed-form singular values,
  as the Berlin `:sigma` option.
- **`Transition` diagnostics**: γ, intermittency, Ri_f, Re_b, V_TKE and
  growth-rate fits, in the Berlin `turbulence_diag`.
- **Flat-plate boundary layer**: a fringe that shares `add_sponge!`'s
  profile, Blasius inflow and free-stream disturbances.
- **Relaminarisation**: criteria, the `laminar_consistent` trait, and
  the Berlin relaminarised fraction.

**Measured**:
- Couette 2.7e-15. Poiseuille converges at order 2.0000.
- σ-model:
  - Exactly zero for shear, rotation and 2-D flow. Near-wall slope
    2.955 (z³).
  - The Kelvin–Helmholtz growth rate with σ equals the no-model 0.1820.
  - CBC: +6.6 % / −11.1 %.
- Kelvin–Helmholtz growth rate 0.1864, against Michalke's 0.1897
  (−1.76 %). At Miles–Howard J = 0.15 it grows (σ 0.0391); at
  J = 0.30 it decays.
- T3A-like plate at 576×64×96:
  - Downstream c_f 1.057–1.135 × the turbulent correlation (target:
    within 20 %).
  - Onset at Re_θ 198 (Mayle 190, Abu-Ghannam & Shaw 200).
  - A 3-D control at Tu 0.1 % stays laminar.
- Relaminarisation:
  - Laminar flow: the σ closure stays exactly zero.
  - Cooled open channel at Re_τ 360: survives at h/L 0.45 and collapses
    at 0.50, on 48³ and on 96³. The critical value is about 0.474,
    against Nieuwstadt's 0.492 in the κ-including convention (−4 %).
- Berlin winter, before dawn: 4.0 % of stable cells relaminarised
  (4.1 % with σ). The background ν = 2 m²/s dominates this.

**Bugs the tests caught**:
- `growth_rate` windows took in the seed transient and the start of
  saturation (0.1859).
- Mixed z boundary pairs were built, then failed in `fill_halos!`.
- The example's critical-h/L bracket always returned NaN.

**Expectations that were wrong**:
- "Smagorinsky is nonzero for solid-body rotation".
- "σ gives exactly zero in Berlin's free stream", which is weakly 3-D.
- The original factor-3 miss against Nieuwstadt: Aeolus puts κ in L,
  so his 1.23 is 0.492 in our convention.
- The coarse T3A grid (1.26–1.39) was the grid, not the code.
- A single-seed L11 assertion.

**Not met**:
- At Re_τ 180, survival at h/L 0.3 fails (critical 0.12).
- Flores & Riley's L⁺ ≈ 100 cannot be reconciled with Nieuwstadt.
- h/L 0.45 was not re-run on 96³.
- σ is off by default.

### Turbulence closure — known limitations carried forward

- **The Berlin flagship night is not physical** (#238): 37 °C at 03:00,
  a 27.7 K heat island, u* ≈ 0.001 m/s. Its constant ν = 2 m²/s is
  about 100× the night-time SGS κ, so it hides most of the closure
  differences measured above.
- **N²** uses dry or vapour-only θ_v. It has no condensate loading and
  no moist-adiabatic stability (#239). AMD's buoyancy term uses dry θ.
- **The SBL validation runs are dry.** The Boussinesq `step_rk3!` does
  not advance qv.
- **Resolution-limited misses**: the K171 CBC deficit, the stratified
  channel at Ri_τ = 60, collapse at Re_τ 180, and GABLS4 at 2 m.
- **Option-only closures** wait on a maintainer decision to become
  defaults, measured at their cost: LASD, AMD, Vreman, σ, RSM, TBLE and
  the dynamic z0.

---

# Model infrastructure

*The model around the physics: how a run is configured, driven, logged
and tested. It has no phase numbers; the work was pulled forward by the
Berlin nesting epic and the Technical-debt milestone. Its GitHub
milestone is "Model infrastructure".*

### Run configuration and input drivers (epic #311)

**✅ SHIPPED** (`src/Config.jl`, `src/config/`, `src/InputDrivers.jl`,
`src/input_drivers/`; CFG-a to CFG-i, #356–#364 / PRs #366, #372, #376,
#375, #374, #380, #389, #393, #396; `docs/src/design/run_configuration.md`).
What was built:
- **`Aeolus.Config`**: a declarative schema with strict validation (every
  problem in one `ConfigError`, with suggestions), `extends` chains (at
  most 8 links), dotted overrides, a canonical resolved `run.toml` with
  `config_hash` and input hashes, and `load_run_config` to replay it.
- **One entry point**: `julia -m Aeolus run | check | resolve | options |
  preprocess`, a run-driver contract and drivers for bare simulations,
  Berlin, SBL, DNS, channels, columns, CBC decay, patchy surfaces and
  DNS decay; `[sweep]` tables; every example script runs a shipped
  config.
- **Static driver** (NetCDF-4): terrain, buildings, surface and ground
  classes, LAI and polygons, written by the OSM/GeoJSON preprocessor;
  Berlin reads its geometry from one committed file.
- **Dynamic driver** (NetCDF-4): initial state, boundary planes and
  forcing, from sounding, Aeolus-parent and ERA5 (GRIB2 or NetCDF)
  preprocessors; the Berlin offline nest. New dependency GRIBDatasets.
- **Docs and coverage**: generated option and input-format references
  and a round-trip suite over every shipped config.

**Measured**:
- 400 random configs (NaN, ±Inf, −0.0, control characters, emoji) are
  a canonical fixed point; Float64 bits survive TOML → config → TOML.
- **25 of 25 shipped configs** replay bitwise from their own `run.toml`
  (4190 assertions); a resumed run is bitwise equal to an uninterrupted
  one for the simulation and SBL drivers.
- Berlin: the four TOML scenarios equal the Julia presets field for
  field, and the 22 state digests after 12 steps are bit-identical;
  geometry from the static driver is bit-identical to the OSM path.
- ERA5 preprocessor on an analytic atmosphere: level heights 2.2e-16,
  θ 1.9e-16, geostrophic wind 1.2e-13 relative; GRIB2 and NetCDF paths
  bitwise identical. Real ERA5 against the Lindenberg sounding
  (2019-06-29 00 UTC, 200–3000 m): θ RMSE **0.787 K**, qv RMSE
  **0.718 g/kg** (bound 1.5 each).

**Bugs the tests caught**:
- `sbl_surface!` read stale periodic halos at the end of an RK3 step,
  so the surface layer of the last row and column used an old wind
  (GABLS1 32³: max|Δu| 1.9e-4 m/s; the reduced GABLS2 inversion moves
  856.0 → 851.7 m). Caught by CFG-d's bit-identity digests.
- The case-section check compared value origins, so a `run.toml`
  (which records every default) did not replay (CFG-e).

**Expectations that were wrong**:
- The DYCOMS RF01 prescribed flux jump "between 50 and 75 W/m²": the
  measured 47.9 W/m² is the exact (F₀ − F₁)(1 − e^{−κ LWP}), now asserted
  to 1e-12.
- "At least 20 CBC steps": the decay is 16 steps at 16³.
- CFG-a's "a driver section may not share a common section's name"
  became a rule change in CFG-b (a same-named driver section extends the
  common one).
- In the merge train, CFG-h's `lindenberg_era5.toml` needed a `[test]`
  table and an ERA5 dynamic driver made once at test time before CFG-i's
  round-trip suite could cover it.

**Not met / not built**: `config_roundtrip.jl` takes 154–176 s against
the 120 s budget (first-call compilation); Berlin resume (#269, since
built: PR #416);
vegetation, soil and water type rasters in the static driver; terrain
(`zt` is written as 0).

### One model: core code, configuration and input drivers (epic #459)

The target is `docs/src/design/single_model.md` (PR #449, agreed with the
maintainer on 2026-10-09): one model, a run is a TOML configuration plus
its static and dynamic drivers, physics reaches a run as a component of a
fixed stage pipeline, and examples are configuration. It supersedes epic
#443 (EX-a … EX-e) and PRs #447/#448 (closed 2026-10-16). Shipped
2026-10-16 with NEST-p3 (#497): one model, one nest engine, one
assembly path (`aeolus run` → `SimulationDriver` → `Model`); the urban
site, its nest parent and the former validation run types are
configurations of it.

**Sizes, before → after**: urban code 16,600 lines (`src/UrbanCase.jl` +
`src/urban/`, before ONE-e) → 12,961 (`src/components/urban.jl` +
`src/components/urban/`); `src/sbl/` 2,277 → 0 (GABLS/CASES-99 are
configurations); `Aeolus.RunDrivers` and the `ParentCase`/`ParentConfig`/
`urban_parent` parent gone; three assembly paths (the model, the urban
case, the run types) → one; `src/config/simulation_driver.jl` 630 →
1,401 lines, the one builder (nesting, the urban site and its parent
included). All of `src/`: 127,813 lines. Fast tier on master after the
last merge: 149 jobs, 120,748 pass, 0 fail.

What was built, in the epic's order:
- **ONE-a (#450 / PR #462)**: `Aeolus.Components`: the component
  interface, the fixed stage pipeline (stages 6–8 inside the core through
  concretely typed hooks), `Domain` and `Model`; advection, SGS,
  Coriolis, surface θ flux and sponge as components. 28 config variants
  reproduce master's state digests bit for bit through `Model`.
- **ONE-b (#451 / PR #466)**: schema version 2: no run types for the
  model, one section per component, `[[domain]]` with per-domain
  overrides, slot/requires/incompatible validation, a bare config is a
  minimal dry LES, no `extends`.
- **ONE-c (#452 / PR #467)**: one budget system (`Aeolus.BudgetSystem`)
  and one output writer selecting from component catalogues.
- **ONE-d (#453 / PR #468)**: `[checks]`, `results.toml` and exit code 3.
  Dry CBL at 16³: energy closure 4.5e-14 W m⁻², momentum 6.2e-25 Pa.
- **ONE-j (#460 / PR #464)**: every component on both cores, with
  checkpointing. Anelastic θ budget of the dry CBL, 32³, 30 min:
  −1.36e-3 → −2.5e-14 of the surface input.
- **ONE-h (#457 / PR #463)**: preprocessing moved to AeolusPrep.jl; it
  regenerates all 11 committed drivers bit for bit in their data.
- **ONE-e part 1 (#454 / PR #469)**: every equation out of the urban glue
  into its module (54-run smoke matrix, 1245 digests, 0 differ).
- **ONE-e part 2 (#454 / PR #472)**: the urban couplings are components of
  an urban site of the model (`[urban] enabled`, one section per
  component); `CaseConfig`'s physics/options, the coupling registry, the
  urban stage loop and the `urban` run type are gone; the Berlin configs
  are self-contained TOML in `examples/urban/`. The open faces' θ
  relaxation was booked twice in the air-heat budget (−119.015 J m⁻² as
  urban tendencies, +119.607 J m⁻² as transport over the 12-step smoke run);
  booked once since.
- **ONE-f (#455 / PR #470)**: nesting by `[[domain]]`, two-way self-nesting
  by default; the blow-up guard checks every domain.
- **ONE-g (#456 / PR #471)**: GABLS1–4 and CASES-99 as configurations of
  the model; the column, channel and DNS a-priori cases dropped; the
  a-posteriori DNS-vs-LES example (dynamic Smagorinsky closest: max
  |ΔK|/K 0.079).
- **ONE-i (#458)**: examples are case directories,
  `examples/<case>/<case>.toml` + `input/` + `data/` + scripts that read
  output; `test/examples_enforcement.jl` fails on any `.jl` under
  `examples/` that builds or steps a model, with no exemptions (the
  forbidden names are read from the package: 46 assertions, each rule
  also fired on synthetic files); the `column` and `dns_decay` legacy
  cases that lived in `examples/` are gone (the Kessler box is a model
  configuration; `sgs_decay` loads the P56-c fixture itself) with
  `Config.DRIVER_FILES` and `run_configs`. The dry CBL of the README,
  32³, 60 steps of 5 s: energy closure 3.4e-13 W m⁻² of 279.98 W m⁻²
  stored, momentum closure 2.4e-21 kg m⁻¹ s⁻².
- **ONE-k (#474 / PR #478)**: the urban site is a component of its
  domain (`UrbanSite`), stepped by `step!(::Model)` through the model's
  run loop; `UrbanCase`'s run loop, `CaseConfig`, its NetCDF writer and
  restart are gone, and `Aeolus.UrbanCase` is `Aeolus.Urban` under
  `src/components/urban/` (14,083 → 13,889 lines). 54 smoke runs, 1137
  digests, unchanged.
- **ONE-l (#475 / PR #477)**: the seven `Aeolus.RunDrivers` run types
  ported to configurations or dropped (`inflow_fetch`, `obstacle_channel`:
  channel cases), `Aeolus.RunDrivers` deleted; the channel harness moved
  to `test/fixtures/channel_cases.jl`. KH, flat plate and CBL-TKE match
  the old scripts; `scalar_advection_2d` is a stated physics change
  (centred flux form on the staggered grid), mean θ constant to every
  digit over 1006 steps.
- **ONE-m (#476 / PR #479)**: one key per exclusive slot (`[core] model`,
  `[timestep] scheme`, `[sgs] model`, `[surface] model`, `[microphysics]
  scheme`, `[advection] scheme`); no enable/disable pairs.
- **NEST-p2 (#482 / PR #485)**: one nest engine (`ModelNest`): windows,
  adaptive root steps with the smallest stable whole number of child
  substeps, a failed window retried from a snapshot, siblings in
  parallel. Adaptive windows 6.1× cheaper per simulated second than
  fixed ones on the 64×64×32 test nest (8 s windows, n = 3, 0.065 s).
- **NEST-p1 (#481 / PR #486)**: the Berlin parent's physics as
  components (tiled surface, prescribed flux, mean-column radiation,
  forcing); the 96×96×38 tiled parent steps in 0.0711 s through the
  pipeline against 0.0742 s before.
- **NEST-p3 (#483 / PR #497)**: complete per-domain configs
  (`--set domain.<id>.<section>.<key>`); the Berlin parent is a
  `[[domain]]` with its own static driver (AeolusPrep.jl writes its
  tiles, the site's classes under the site's footprint); the site is a
  child of `ModelNest`; `ParentCase`, `ParentConfig`, `urban_parent`
  and NestedRun's window API deleted (`src/components/urban` 14,204 →
  12,950 lines). Berlin smoke, 48 s on one thread: 2,175 → 1,947 s
  wall per simulated hour, 0.97 → 0.79 s per site step; a nest with an
  urban site no longer snapshots its windows (3.5–4.4 s per 1.5 s
  window measured).
- **PREP-fix (#487 / PR #488)**: AeolusPrep.jl's suite green against
  Aeolus again (132/17/10 → 243/0/0), and `bin/test-against <checkout>`
  in AeolusPrep runs it against any Aeolus branch.

**Expectations that were wrong** (recorded in the PRs): ONE-g's GABLS3
jet-height bound was narrower than the scatter between realisations
(five seeds: 89.5–99.3 m, σ = 0.5 Δz), widened to 1.5 Δz; a contract-test
tolerance of 1e-9 was below the round-off of the measured change
(1.7e-7); ONE-i's cloud_spinup test expected a diagnostics row at
t = 0, which the run monitor never writes (rows at 60 … 480 s). **Bugs the tests caught**: a nested run's guard snapshot held
the root at t = 4 s and the child at t = 0 (ONE-f); a `mean` writer's
open averaging intervals were not checkpointed (ONE-g: u* 0.5027 vs
0.5083 after a resume). Later: `[sgs] model = "tke"` could not be built
on a stretched grid (NEST-p1); an earlier NEST-p3 commit silently
dropped the site's footprint from the parent's tiles (restored as
driver data); the Berlin smoke digests depended on the BLAS thread
count, not on load (#489: the radiosity LU). Expectations that proved
wrong in NEST-p3: the smoke checks' Bird tolerance at μ₀ = 0.148
(−15.2 %, outside the calibrated μ₀ ≥ 0.27) and the stability check
on a profile with no capping inversion (#490).

**Left for #498**: the run-driver registry and `[run] legacy` hold one
entry (the model) and go; the core's step keywords (`body_force!`,
`post_predictor!`, `sgs`) become private once the wall-model channel
harness, the flat plate, the immersed-boundary path and the urban time
stepper pass the pipeline's hooks. #484 (attributed budget exchange
across nests) and #461 (external LES validation) are separate.

---

### Structured logging, run folders, diagnostics and provenance (epic #293)

**✅ SHIPPED** (`src/logging/RunLog.jl`, `src/io/run_folder.jl`; LOG-a
to LOG-g, #333–#338 and #347 / PRs #340, #344, #346, #370, #373, #388,
#390; `docs/src/api/logging.md`). What was built: structured events
through stdlib `Logging` (the library never prints; a quiet test
logger); one run folder per run with `run.toml` provenance (commit,
versions, Manifest hash, FFTW policy, seeds, one `[[session]]` per
restart) and a dirty-tree policy; a diagnostics registry writing
`diagnostics.csv` and `diagnostics.nc` (NetCDF-4, per-step rows and
costly rows with profiles); a progress line; a blowup guard that
diagnoses, writes a crash checkpoint and stops; a `StepRecord` filled
by every step core (dt and its limiter, CFL, solver iterations and
residuals, wall time per stage) with the run monitor attached to the
cores; the `verbose` keywords removed; the Berlin console output
through the logger.

**Measured**: one six-key record to `run.log` 7.9 µs with the 5 s flush
policy; the default monitor's parts **0.084 ms per step, 1.56 %** of a
64×64×32 step (budget 2 %, asserted by the benchmark); `step!` still
allocates 0 B with a hook installed; a run folder costs 11.6 ms per run.

**Bugs the tests caught**: `latest_run` compared Float64 timestamps as
strings, so `resume = :latest` could restart the wrong run; the
`diagnostics.nc` provenance of a restart that changed its config
recorded the run's first config.

**Budget restored (#514 / PR #515, 2026-11-06).** By November the same
parts were **2.27 %** of a 3.71 ms step and the benchmark failed: the
step had become 30 % faster (#493), the monitor had not, and nothing
reran the benchmark. Without checking less often:
- the attached guard repeats nothing `step_adaptive!` has done: it
  sweeps only `νt`, `κt` (and a dry core's `qv`) and keeps the step's
  rollback snapshot as the last good state instead of copying the state;
- the remaining full checks are one fused, threaded sweep;
- one pressure value is read between checks;
- the costly `diagnostics.nc` rows go in the 5 s block write (each one
  was 13.5 ms written at once, 0.085 ms of it the profiles).

Measured on 64×64×32:
- the default (attached) monitor's parts are **0.57 %** of a step on 1
  thread and 0.46–0.55 % on 4–8; a due check costs 0.068 ms against
  0.44 ms before;
- timed step by step, what a run pays is **+1.1 to +1.5 %** on 1 thread
  (master +2.8 to +3.1 %), about 2.5× the parts (cache effects). The
  benchmark now prints both and asserts the parts.

Against master, 13 blowup scenarios caught the same blowups at the
same step in the same fields; under an attached monitor the crash
checkpoint is one step older, and fixed-step runs are stopped 3–7 steps
earlier. `les_tke` (32³, fixed FFTW planner) was bit-for-bit identical
in field output, `diagnostics.nc` and `results.toml`.

**Not met / loose ends**:
- The benchmark was not run by anything, so the overrun went unnoticed
  for a month; it is still a manual benchmark.
- The fused finiteness test in `save_state!` was rejected: it was slower
  serially (0.54 against 0.45 ms).
- The per-step CSV row costs ~23 µs in a run against ~2 µs timed alone
  (the largest remaining in-run cost).
- `bitwise_reproducible` is still recorded in the provenance, although
  #493 dropped bit-identity as a goal.
- Asynchronous field-output and checkpoint writes are #516 (Performance
  milestone).

**Expectation that was wrong**: `test/logging_diagnostics_nc.jl` asserted
that a costly row is on disk at once. That was the old policy, which #514
changed on purpose, so the code was right and the expectation was stale.

### Tests, CI and documentation deployment (#316, #317, #320, #354)

**✅ SHIPPED** (#316 / PR #325, #354 / PR #355, #320 / PR #387, #317 /
PR #395; `docs/src/testing.md`). What was built:
- **Runner**: `test/run.jl` runs the files in parallel. *Update
  2026-11-05:* the fast/full tiers were replaced by two suites, the tests
  (the PR gate, cut to their crucial checks) and the validation runs
  (`validation/`, not a gate); see `docs/src/testing.md` and
  `docs/src/validation.md`.
- **CI** (GitHub Actions): the tests on every PR and push to `master`,
  the validation runs weekly and on demand; Julia 1.13. The Documenter
  workflow builds the docs on every PR and deploys `dev` from `master`
  and `stable` from tags to `gh-pages`.
- **Berlin test suite removed (#320)**: the 13 `test/berlin_*.jl` files
  (4162 assertions, 1260 s of job time) and the coverage lists are gone.
  Every `src` physics check moved into a module test with a synthetic
  setup; the example's end-to-end runs are a by-hand script,
  `examples/urban/berlin_checks.jl` (897 assertions).
- **Pruning (#317)**: `large_scale_forcing.jl` 78.3 → 12.6 s (a
  turbulent box instead of 100 Berlin smoke steps); one shared
  k^(−5/3) fixture for five SGS files.


**Shipped with these blocks**: FLAG-a (#392 / PR #398): the default
flagship failed near step 12 because a timing script forced 2 s steps
past the stable dt (0.47 → 0.098 s within 9 steps; |v| 7.8 → 49 m/s,
θ 229–360 K at step 15); `step_case!(case, dt)` now refuses a dt above
the adaptive stepper's `stable_dt`, and the default case runs 20 steps
without a retry (max CFL 0.46). The issue's premise proved wrong: it
read as an instability of the default case, but the case is stable when
stepped at its own dt; the failure was the forced step.

**The guard's snapshot of the tiled surface (#526, 2026-11-06).** Profiling
a 24 h Berlin run showed where the time went. The model domain is 100 m,
200 × 200 × 60, with land-use tiles and ERA5 open faces, on 24 threads.
The run loop took 1.75 s per step against 0.26 s for the model step
alone.

87 % of the main thread was the blow-up guard. Twice every 10 steps it
snapshots the components, and the tiled surface's `checkpoint_state`
was the generic structural walk: 4,538,305 objects and 346 MB, 7.2 s.
Most of those bytes were solver scratch (35 workspace vectors per soil
tile) and constants.

Built: a packed checkpoint state.
- `SurfaceExchange.pack_state!` / `unpack_state!` are implemented by
  `SoilWaterColumn`, `SoilThermalColumn`, `SlabLand` and `SurfaceMosaic`.
- `TiledSurfaceState` holds every tile's state in one vector, packed
  over the columns in parallel. The walk stays for other surfaces and
  for old checkpoints.

Measured on that configuration (the real run loop, (240 s run − 40 s
run) / 100 steps, 24 threads):

| | surface `checkpoint_state` | serialized | run loop |
| :-- | --: | --: | --: |
| before | 6.8–7.9 s | 523 MB | 1.753 s/step |
| after | 0.043 s | 50.6 MB | 0.357 s/step |

That is 4.9× faster, a 24 h run in about 4.3 h instead of 21 h.

The numbers vary between sessions. Two more sessions of the new code gave
0.430 and 0.361 s/step. A free snapshot (the ceiling) gave 0.294, 0.306,
0.325 and 0.345 s/step. Measured back to back, the new code was 0.361
against 0.345: 0.016 s/step for the snapshot. Most of that is garbage
collection (0.034 against 0.014 s/step), from the snapshot's 50 MB
allocations.

The real checkpoint/resume path was also checked: run 240 s, resume from
the 120 s checkpoint, compare the 240 s states. The old and the new code
give the same continuation to within run-to-run round-off (2.6e-9 after
the resume, 2.9e-9 between two uninterrupted runs).

That check also showed a resume of an open-box domain does not continue
the uninterrupted run. The first step after it differs by up to 0.012 m/s
at the east open face, the same with the old and the new surface state.
That is pre-existing and not part of this change; its numbers are in
the description of #526's PR, for an issue of its own.

Tests: a column (with an aquifer), a mosaic and the whole component are
restored into another history with the scratch poisoned (NaN), and must
continue as the original (rtol 1e-12). A test expectation that proved too
weak: the first component test poisoned the tiles through `unpack_state!`
itself, so a dropped `T_skin` restore went unnoticed (identical
histories). It now gives the surface a different history (three extra
tile steps) and checks that a dropped `T_skin` fails it.

**Open after these blocks**: #397 (FFTW MEASURE segfault);
#238 (implausible Berlin nights).

### Speed over bit-identity (epic #493)

Results need not be bit-identical across thread counts, machines,
restarts or refactors; tests assert physics within stated tolerances
(maintainer decision, 2026-10-16). See `docs/src/testing.md`.

**FFTW threads in every model (#524, 2026-11-06).** Since KA-h (#298)
`init_threads!` puts FFTW on `Threads.nthreads()`, but only the urban
site's `build_case` called it: every other model domain, nested or not,
kept FFTW at its start-up count of 1, so the pressure solver's FFTs ran
on one core while the kernels ran on all of them. `Config.build` now
calls `init_threads!` for every run before any plan is made; an explicit
count (`[fftw] threads`, `AEOLUS_FFTW_THREADS`, `set_fftw_policy!`) is
kept. Measured on a 200 × 200 × 60 open-box domain (Berlin, 100 m, 40
steps, 24 Julia threads on 12 cores): the pressure solve went from 3.48 s
(1 FFTW thread) to 0.89 s (12 threads), 8 % → 2 % of the step; the
step's total did not change, so the remaining time is elsewhere (being
profiled). Test: `config_model.jl` builds a non-urban run from a policy
at 3 threads and expects `Threads.nthreads()`, and an explicit 2 kept.

**The open faces' time blend reads the slabs directly (#529, 2026-11-06).**
A dynamic driver's boundary record is a `SlabArray`, a storage-indexed view
whose `getindex` searched the field's slabs and tested three ranges at
every point. The blend of two records into each open plane read two such
points per output point. Most planes are below the threading threshold
(26 112 points against 16 384 per task; 18 of the 20 run serially), so
this was a serial loop paying the search at every point. `blend_window!`
now asks `blend_source(A, ranges)` for the dense array and offset behind
the plane's box. A `SlabArray` returns the one slab holding the whole
box, found once per plane, and a box no single slab holds keeps the
point-by-point view.

Measured on the Berlin 100 m domain (200 × 200 × 60, ERA5 open faces, 24
threads), all 20 planes and 548 352 points:
- one `boundary_values!` takes 2.69 ms before and 0.42 ms after
  (2.55 and 0.36 ms in a second pair of runs);
- per step, counted inside `step!`: 3.0 calls, 8.1 ms before and
  1.8 ms after, out of a 0.26 s step. That is a 2.4 % gain.

A deviation from the plan: the issue estimated the blend at 12 % of the
step, from its share (95 of 785 samples) of the main thread's profile.
The direct timing gives 3 %. The real run loop can't resolve 2.4 %: its
s/step varied between 0.35 and 0.59 between sessions, for both versions.

Test: `input_dynamic.jl` records a driver whose values differ at every
point, field and time. Every plane takes the slab path, reads exactly what
`getindex` reads (copied data) and blends to the same values; a box
spanning more than one slab keeps the view. Shifting the slab offset by
one point makes the test fail (43 errors).

**A checkpoint of an open box continues the run (#528, 2026-11-06).**
Resuming the Berlin 100 m run (ERA5 open faces) did not continue it. One
step after `--resume`, u differed by up to 0.012 m/s at the east open face;
after 60 steps, by 0.14 m/s in u and 0.10 K in θ. Two uninterrupted runs
agree to 3e-9.

The open boundaries belong to the simulation's boundary conditions, not to
a component, and the checkpoint stored only the fields and the components.
A restart rebuilt each `OpenBoundaryState` with `initialized = false`. Its
first step refilled every plane buffer from the boundary data, so the
values carried over were lost:
- the radiated values at outflow points;
- the Runge–Kutta stage combination of the buffers;
- the mass-flux correction.

The Orlanski history was lost as well.

The checkpoint now stores `open_boundary_checkpoint(sim)`: every buffer,
`initialized`, the correction's ledger and the radiation planes' history.
`restore_checkpoint!` copies these back. A checkpoint from before this
change restores as before, with a warning.

A bug the experiment caught in the first version of the fix: an
`initialized` state skips the first update, which is where θ is bound
(by identity) to its planes. The unbound θ got the mirror halo for one
step, so the restored run still diverged (2e-4 m/s in u, 1.6e-2 K in θ
after one step, with Dirichlet faces). Restoring now binds the fields.
Finding this took a walk over the whole object graph: the restored and
the continuous simulation were identical except for scratch, and
stepping a fresh build once before restoring made it exact.

Measured on a 16 × 12 × 8 open box (RK3, dt 0.5 s, time-varying data,
checkpoint after 20 steps, compared with the uninterrupted run over the
next 10 steps):

| outflow | restored with the open-boundary state | old restart, after 10 steps |
| :-- | --: | --: |
| `dirichlet` | 0 | 9.9e-5 m/s, 1.6e-3 K |
| `convective` | 0 | 0.42 m/s, 1.5 K |
| `orlanski` | 0 | 0.41 m/s, 1.5 K |

Test (`open_boundaries.jl`), Orlanski outflow:
- a restored fresh build continues to 1e-12;
- an old payload warns and diverges by more than 1e-2;
- a differently built open box is refused.

**The real run, through the CLI.** This is the Berlin 100 m configuration (200 × 200 × 60, ERA5 open faces), at 24 Julia threads, dt 2 s, with the `estimate` FFTW planner (a deterministic plan, and no #397 crash):
1. `run` for 240 s, with checkpoints every 60 steps;
2. set the 240 s checkpoint aside;
3. `--resume`, which restores the 120 s checkpoint and runs to 240 s.

The resumed state at 240 s equals the uninterrupted one in every field (u, v, w, θ, qv, p, νt, κt): max|Δ| = 0. Before the fix the same check gave 0.14 m/s in u after 60 steps.

**FFTW's threads on its own pool: the MEASURE planner no longer crashes
(#397, 2026-11-06).** Planning a threaded transform under MEASURE crashed
the process (signal 11 in `spawn_apply`) at 10–14 threads.

FFTW.jl registers `spawnloop` as FFTW's thread callback. It spawns one
Julia task per job, over a job array on the C stack of the FFTW call
waiting in `@sync`, and that caller may yield while the jobs run. This is
upstream JuliaMath/FFTW.jl#306, and the fix there (#344) is unmerged.
FFTW.jl 1.10.0, the latest release, with FFTW_jll 3.3.12, still has it.

Since #525 every model plans threaded, so this hit every run with an
unlucky thread count. The FFTW policy (`_fftw_policy!`, the one place
every plan's flags come from) now clears the callback when it first
resolves. FFTW then runs its threaded transforms on its own threads.

Measured on a Ryzen AI 9 HX 370 (12 cores), at 12 Julia and 12 FFTW
threads, under MEASURE:
- planning a 256³ irfft crashed 4 of 4 processes before, 0 of 4 after;
- the Berlin r2r plus the 256³ irfft crashed 8 of 8 before, 0 of 8 after.

The cost: the 200 × 200 × 60 r2r under ESTIMATE takes 1.6–1.7 ms on Julia
tasks against 1.9–2.3 ms on FFTW's threads at `-t 12`, and both run
1.7–2.5 ms at `-t 24`. That is about 2 ms of a 0.24 s Berlin step.

Test: `fftw_threads.jl` (exclusive) plans the 256³ irfft under MEASURE
in a `-t 12` subprocess and checks that the round trip recovers the input
to 1e-12. With the fix disabled, the subprocess crashes and the test
fails. Remove the workaround once FFTW.jl ships the fix.

# Chemistry and aerosol

*From four toy species to a mechanism a chemist would accept.*

**Status: not started** (milestone 7: seven issues, all open). The
Berlin epic (#135 / PR #162) moved a reduced NOx–O₃–VOC mechanism with an
exact NO–NO₂–O₃ triad into `src/Chemistry.jl` (triad against RK4 1e-9,
NOx and Ox conserved to 1e-14, noon Leighton ratio 1.0013 in the
flagship), an MCM clear-sky J(NO₂), Wesely-type r_a + r_b + r_c dry
deposition and HNO₃ rain scavenging. That is groundwork, not Phases 60,
61 or 66: no KPP mechanism, no online actinic flux, r_c is
order-of-magnitude, there is no pollen and nothing is validated against
flux data.

### Phase 60 — KPP-generated gas-phase mechanism
Real KPP code generation with a full mechanism (MOZART-T1, CB05 or
MCM subset), analytic Jacobian, and a Rosenbrock stiff solver.
Replaces the three-reaction toy scheme.
**Validated by**: box-model comparison against the reference KPP
implementation; ozone-production comparison for a photochemical smog
case.

### Phase 61 — Photolysis rates
Online photolysis using the computed actinic flux from the
radiation solver, including the effect of clouds and aerosol on
photolysis rates.
**Validated by**: J-value comparison against TUV reference model.

### Phase 62 — Aqueous and heterogeneous chemistry
In-cloud aqueous-phase chemistry (sulphate production), heterogeneous
reactions on aerosol and building surfaces, and the urban-specific
chemistry of NO₂ on surfaces.
**Validated by**: sulphate-production comparison against cloud-chamber
data.

### Phase 63 — Modal and sectional aerosol microphysics
Full aerosol dynamics: nucleation, condensational growth, coagulation,
and a thermodynamic-equilibrium module (ISORROPIA-style) for
inorganic partitioning, with a modal and a sectional option.
**Validated by**: size-distribution evolution against chamber
experiments; comparison against reference aerosol-model output.

### Phase 64 — Secondary organic aerosol
Volatility-basis-set treatment of SOA formation from biogenic and
anthropogenic VOCs, including the urban-relevant cooking and
traffic sources.
**Validated by**: organic-aerosol mass and O:C ratio against AMS
measurements.

### Phase 65 — Real emission inventories
Ingest gridded emission inventories (TNO-MACC, EDGAR, CAMS), traffic
emission models keyed to OSM road geometry and diurnal traffic counts,
biogenic emissions from MEGAN driven by the resolved vegetation and
temperature, and point sources from facility registries.
**Validated by**: mass-balance audit against inventory totals;
concentration comparison against urban monitoring stations.

### Phase 66 — Deposition and pollen
Resistance-network dry deposition (Wesely / Zhang) with real surface
properties, size-resolved wet deposition, and a pollen model for
allergy applications.
**Validated by**: deposition-velocity comparison against flux
measurements; pollen-concentration comparison against counting
stations.

---

# Real-world data, forcing and validation

*A model is only as good as what drives it and what it is checked
against.*

### Phase 67 — ERA5 and reanalysis driver
Read ERA5 / MERRA-2 to build initial and boundary conditions:
vertical profiles, geostrophic forcing, advective tendencies, surface
conditions, and their time evolution over a multi-day run.
**Validated by**: reproduction of an observed boundary-layer evolution
for several real dates.

**Phase 67a, shipped** (CFG-h, #363 / PR #393): an ERA5 reader (GRIB2 and
NetCDF) writes the dynamic driver, compared with the Lindenberg sounding
for one date (θ RMSE 0.787 K, qv RMSE 0.718 g/kg). The reader now lives
in AeolusPrep.jl (#457). **Phase 67b, open** (#57): MERRA-2, multi-day
runs and the validation across several dates.

**Initial soil profiles, shipped** (#521; AeolusPrep#6 / its PR #7 writes them).

- **The driver.** It reads PALM's `init_soil_t` / `init_soil_m` over `zsoil`, as a profile or a volume (`alt_dims`, a new field of `DriverVariable`).
- **Where they go.** The urban site's ground (`[ground] soil_init`: auto, uniform, driver, annual_wave) and the nest parent's soil tiles start from them. The base is held at the deepest level.
- **The fallback without data.** It is the analytic annual wave (`SoilColumn.annual_soil_temperature`, `soil_damping_depth`).

Measured:

- **The wave start removes the slow mode.** On a 15 m loam column under the annual surface wave, 30 d of hourly steps from day 120: started on the wave it stays within 0.0045 K of the analytic solution over the top 3 m; started uniform at the mean it is 1.51 K off (d = 2.84 m).
- **FLUXNET land column, uniform start vs annual wave** (a script, not a validation case: the climatology is assumed, MAT 8.3 °C / 7.8 °C, A = 9 K, maximum on day 200). Bias of G against the observed `Qg`:

  | site | days 1–5 | days 6–35 |
  | :-- | :-- | :-- |
  | DE-Hai | −21.6 → −2.8 W m⁻² | −6.1 → +2.7 W m⁻² |
  | DE-Gri | −13.6 → +5.4 W m⁻² | −3.5 → +5.9 W m⁻² |

  RMSE: days 1–5 31.0 → 25.6 (DE-Hai) and 24.9 → 27.3 (DE-Gri); days 6–35 26.5 → 27.5 and 23.7 → 26.7 W m⁻². The 1 m soil temperatures of the two starts still differ by 6.0 K and 5.3 K after 35 d. The start is not a detail at the daily scale, and an assumed climatology is no substitute for ERA5's profile.

Test expectations that were wrong:

- Two CLM nodes below ERA5's deepest level (1.945 m): there is one, because the 9th node is at 1.73 m.
- The water clipped after the interpolation: the code clips the driver's levels before, which is what it should do.

A soil volume read without a mesh is no longer an error. The sounding reader opens the same file without one; the soil's consumer reads it with the mesh and checks x and y.

**Deviations:**

- `"auto"` without soil warns only when a driver exists; an idealised case logs info.
- A nested child does not yet take the parent's mean profile (PALM's rule).
- Per-column texture from data (SoilGrids) is the extension noted on #521.

### Phase 68 — Mesoscale model nesting
Offline and online nesting into WRF, ICON and IFS output, with the
interface handling different vertical coordinates, different
microphysics assumptions, and the spin-up of turbulence from smooth
mesoscale fields (synthetic-turbulence inflow generation).
**Validated by**: nested-LES comparison against the driving mesoscale
model's own boundary layer; turbulence spin-up distance diagnostics.

### Phase 69 — Production OSM and geospatial ingestion
Full OSM support: multipolygon relations, building parts, inner
courtyards, roof shapes, material and level tags, street widths,
tree registries, and land-use polygons. Plus DEM terrain (SRTM /
Copernicus), CORINE land cover, and lidar-derived building and canopy
heights.
**Validated by**: geometric comparison against reference city models
(CityGML) for several European cities.

### Phase 70 — Observation operators and evaluation framework
Forward operators mapping model state to what instruments measure:
sonic anemometer, sodar, wind lidar, ceilometer, radar reflectivity,
radiometer brightness temperature, air-quality monitor, thermal
infrared imagery. Plus a statistical-evaluation framework.
**Validated by**: end-to-end comparison for an instrumented site.

### Phase 71 — Canonical benchmark library, part 1 (dry)
Automated, versioned, reference-data-backed implementations of: neutral
channel (Kim-Moin-Moser, Hoyas-Jiménez), dry CBL (Sullivan-Patton),
GABLS1-4, Wangara, CASES-99, flow past a cube (Silsoe), backward-facing
step, urban canyon (Uehara wind-tunnel), Michelstadt and MUST urban
dispersion campaigns.
**Deliverable**: a CI-runnable suite that reports skill scores per case.

### Phase 72 — Canonical benchmark library, part 2 (moist and reactive)
BOMEX, ATEX, RICO, DYCOMS-II RF01/RF02, ARM SGP diurnal cumulus,
ASTEX transition, plus photochemical-smog and urban-dispersion
reactive cases.
**Deliverable**: the same CI suite extended, with the intercomparison
ensembles as reference envelopes.

### Phase 73 — Reproducibility and provenance
Full run provenance: input hashes, library versions, compiler flags,
a deterministic-mode guarantee, and archived reference output so that
any result in the docs can be regenerated years later.
**Deliverable**: a `provenance.toml` per run and a `reproduce` command.

*Partly built early* (LOG-b #334, CFG-a/b #356–#357, the KA port #291):
every run folder's `run.toml` records the commit, versions, Manifest
hash, FFTW policy, seeds and input hashes; `julia -m Aeolus run
run.toml` replays it (within round-off: since #493 results need not be
bit-identical). Archived reference output and
compiler-flag provenance remain.

### Spin-up (#519)

**Surface spin-up, shipped** (#519, part B of the issue): PALM's `spinup_time` as a run-level `[spinup]` section. See [Run configuration](run_configuration.md#Surface-spin-up-([spinup],-#519)).

- **How it runs.** The urban site's surfaces, radiation and building advance alone at negative times. The air above the surfaces is forced by PALM's solar sinusoid or by the dynamic driver's `sfc_temperature`. The soil water is frozen by default. Afterwards the atmosphere, the clock and the ledgers are exactly as at the start.
- **The initial envelope.** The city-mean building now starts in steady conduction between outdoor air and room (`steady_conduction!`, `steady_envelope!`).
- **Measured.** These runs are on the synthetic small city (radiation every 600 s, amplitude 4 K). The figure is the maximum over surface classes of the day-over-day drift of the daily-mean surface temperature:
  - 2 d: roof 5.26 K/d with `dt = 300`, 2.91 K/d with `dt = 60`;
  - 3 d: roof 0.144 / 0.026 K/d, street 0.25 / 0.21 K/d;
  - 4 d (`dt = 300`): roof −0.007 K/d, street 0.13 K/d;
  - water stays at 2–2.7 K/d, because its slab (C = 8 × 10⁶ J m⁻³ K⁻¹) is slow;
  - cost: 4 d at `dt = 300` is 1152 steps in 31 s; 3 d at `dt = 60` is 4320 steps in 67 s.
- **Time scales.** These are the slowest eigenmodes: brick wall 9.3 h, roof 26 h, ground stack 19 d, soil (3.43 m) 30 d. Deep soil is for #521 (profiles from data), not for the spin-up.
- **Validation.** `validation/run.jl --only urban_checks.jl`: 33 pass, 0 fail (292 s).
- **Test expectations that were wrong:**
  - "the roof concrete is within 1 K of the room" was wrong: the analytic drop is 1.07 K. The test now asserts the analytic q·(R_si + R_plaster + R_concrete).
  - The XPS assertion is the exact q·(d − Δz)/k.
  - `config_model.jl`'s inventory of run-level sections gains `spinup` by design.
- **Deviations from the issue's plan:**
  - The forcing is θ only. Wind and humidity stay at the initial state, as in PALM, rather than coming from the driver.
  - The solar forcing uses the case date's sun on every day.
  - There is no separate `surface_time` / `surface_dt`: the section is `[spinup] time` / `dt`.
- **Open:**
  - part A (`[time] spinup`: output, statistics and checks skipping the turbulence spin-up);
  - saving and reusing the spun-up state (PALM's `read_spinup_data`);
  - moving the validation cases' own spin-up loops (`land_surface_fluxnet`, `anthropogenic_heat`, `building_energy`) onto it.

---

# Data assimilation and inverse problems

*Once forward fidelity is real, make the model answer inverse
questions.*

### Phase 74 — Adjoint model via automatic differentiation
Full reverse-mode AD through the solver with `Enzyme.jl`: adjoint of
dynamics, physics and I/O, with checkpointing for long windows.
**Validated by**: gradient verification against finite differences for
every prognostic and parameter.

### Phase 75 — 4D-Var
Strong- and weak-constraint 4D-Var with the observation
operators, background-error covariance modelling, and a quasi-Newton
minimiser.
**Validated by**: identical-twin experiments; assimilation of real
tower and lidar data.

### Phase 76 — Ensemble Kalman filtering
LETKF and ensemble-square-root filters with covariance localisation
and inflation, run over an ensemble of LES members.
**Validated by**: identical-twin experiments; comparison of EnKF and
4D-Var analyses.

### Phase 77 — Hybrid and coupled assimilation
Hybrid EnVar combining ensemble and static covariances; coupled
assimilation across atmosphere, land and urban surfaces so soil
moisture and wall temperatures are constrained too.
**Validated by**: joint-state analysis skill against withheld
observations.

### Phase 78 — Parameter and source inversion
Estimate uncertain physics parameters (SGS coefficients, roughness,
soil hydraulic properties) and unknown emission sources (location and
strength of a chemical release) from observations.
**Validated by**: recovery of known parameters and known source terms
in synthetic experiments; a real tracer-release campaign.

### Phase 79 — Uncertainty quantification
Calibrate-emulate-sample workflow, Gaussian-process emulators of the
model, Sobol sensitivity analysis, and calibrated predictive
uncertainty on model output.
**Deliverable**: uncertainty bands on the flagship urban-heat-island
prediction.

---

# Performance, scale and portability

*Deliberately placed after physics: a fast model of the wrong physics
is worthless, and premature optimisation would have frozen interfaces
too early. By this point the physics is settled, so it is safe to
optimise hard.*

### Phase 80 — Production MPI
Real domain decomposition (3-D pencil), non-blocking halo exchange
overlapped with computation, distributed FFT via `PencilFFTs.jl`,
distributed multigrid, parallel I/O, and dynamic load balancing for
the spatially varying cost of chemistry and radiation.
**Validated by**: bit-reproducibility against serial for the benchmark
suite; strong and weak scaling to 10⁴ cores.

### Phase 81 — GPU port
Full `KernelAbstractions.jl` port of every kernel, GPU-resident FFT and
multigrid, GPU microphysics and chemistry, and a single-source
CPU/GPU code path.
**Validated by**: bit-comparable results within tolerance; 10–50×
speed-up over single-CPU-core on realistic cases.

**Phase 81a, the CPU half, has shipped** (epic #291; its record is the
next section). The KernelAbstractions port landed early
(epic #291) for CPU multithreading; the GPU backend, GPU-resident FFT/MG
and SoA surfaces remain. Measured (KA-o #308, Ryzen 9 7900): Berlin
flagship 155 -> 88 ms/step at 12 threads (1.77x), 256³ DNS 1.15 h -> 48
min (1.43x; step_rk3! 1.87x); thread-invariant at 1/4/12/24 (3746
hashes). The scaling targets of #308 (≥ 2.5× at 4 threads, ≥ 4× at 12)
were **missed**: the stencils are memory-bound at the 50–75 GB/s DRAM
bandwidth of that machine, while compute-bound kernels such as WENO-Z5
reach 8.6×. The scaling epic #406 (shipped) took the 256³ DNS
`step_rk3!` to 3.03× at 12 threads (0.509 s; `dns_step!` 2.25 → 0.573 s)
and the flagship to 2.02× (2.29× with the parent prefetched).

#### Phase 81a — CPU port to KernelAbstractions (epic #291)

Phase 81's port, pulled forward for single-node multithreading through
KernelAbstractions' CPU backend, with one code path that the GPU
backend will reuse.

**✅ SHIPPED** (`src/Backends.jl` and every hot loop in `src/`; KA-a to
KA-n, #294–#307 / PRs #342, #367, #341, #377, #353, #378, #352, #384,
#381, #382, #391, #383, #379, #385; scaling report KA-o #308 / PR #399;
design `docs/src/design/ka_port.md`). KernelAbstractions port landed
early (epic #291) for CPU multithreading; the GPU backend, GPU-resident
FFT/MG and SoA surfaces remain. Measured (KA-o #308, Ryzen 9 7900):
Berlin flagship 155 -> 88 ms/step at 12 threads (1.77x), 256³ DNS
1.15 h -> 48 min (1.43x; step_rk3! 1.87x); thread-invariant at 1/4/12/24
(3746 hashes). What was built:
- **Primitives (KA-a).** A real `CPUBackend` on `KernelAbstractions.CPU()`;
  `launch!` over storage-index ranges (one i-row per work item, serial
  below 32,768 points, at least 16,384 points per task); canonical
  eight-lane reductions; `init_threads!` (BLAS pinned to one thread when
  Julia runs threaded); the opt-in thread-invariance harness
  (`AEOLUS_THREAD_TESTS=1`, `test/thread_invariance.jl`).
- **Serial baseline (KA-c).** Compile-time integer powers in WENO-Z/TENO,
  allocation-free halos and solvers, an RK3 workspace: flagship
  allocation 30.4 → 1.0 MB per step.
- **Deterministic sums (KA-b).** Every order-dependent sum on the step
  path goes through `plane_reduce!`, `plane_map!` or `range_max`, so its
  order does not depend on the thread count.
- **Kernels.** Finite volume on a `GridMetrics` abstraction that folds
  about twenty duplicated StretchedGrid kernels into one source (KA-d);
  halos, open boundaries and nesting operators as owner-computes gathers
  (KA-f); one matrix-free `pcg!` for all four CG solvers (KA-g);
  threaded FFTW by default and the stretched Thomas solve over
  wavenumbers (KA-h); colour-indexed multigrid (KA-i); the step cores
  with one fused CFL sweep (5 → 1) and one fused step guard (11 → 1)
  and per-simulation workspaces (KA-j); IBM, wall stress, canopy drag
  and large-scale forcing (KA-k); every SGS closure (KA-l); column
  physics one work item per column, with prefilled or locked caches
  (KA-m); radiosity LU caching, threaded view factors, multi-column
  two-stream and Philox4x32-10 counter-based random streams for
  Monte-Carlo and superdroplets (KA-n); WENO/TENO scalar transport and
  the positivity limiter (KA-e).
- **Dependency policy (2026-10-06).** "Minimal dependencies" replaces
  "no binary dependencies". KernelAbstractions and StaticArrays are
  direct dependencies.

**Measured** (KA-o; Ryzen 9 7900, 12 cores / 24 threads, dual-channel
DDR5, Julia 1.13.1; speed-up = new code at 1 thread / at N threads):
- Flagship `CaseConfig(:summer)` (64×64×64 child in its 192×192×53
  parent): `step_case!` **155.2 → 90.7 / 87.5 / 97.0 ms** at 4 / 12 / 24
  threads (1.71× / 1.77× / 1.60×). The child alone (nest off): 100.3 →
  46.5 ms at 12 threads (2.16×). Per entry at 12 threads: aq_chem 4.98×,
  fog_dew 5.19×, scalar_advection 2.08×, dynamics 1.76×, the nest
  parent 1.48×.
- 256³ DNS: `dns_step!` **3.28 → 2.29 s** (1.43×); `step_rk3!` alone
  2.21 → 1.18 s (1.87×); the 1264-step run 1.15 h → 48 min. The
  unported per-step history diagnostics cost 0.88 s of the 2.29 s.
- Kernels at 256³, 12 threads: WENO-Z5 **8.57×** and `set!` sin·exp
  8.07× (compute-bound); axpy 1.06×, `laplacian` 1.65×,
  `momentum_advect!` 1.80× (memory-bound).
- Launch overhead: 263 threaded and 569 serial launches per flagship
  step, under 3 % of the step at 12 threads. SMT: 24 threads gain on
  compute-bound work and lose on memory-bound work (−18 %); 12 threads
  is the better setting on this machine.
- Thread invariance: **3746 hashes identical at 1/4/12/24 threads**
  (bitwise invariance is no longer required since #493).
- Serial wins on the way (new code at 1 thread vs the pre-port loops):
  CG-family solves 0.41–0.61×; multigrid 0.43–0.59×; the radiosity day
  call 3.78 → 1.80 s and the night call 1.89 s → 17 ms (LU cache);
  `Subsidence(:local)` 15.1 → 0.15 ms; canopy drag 1.01 → 0.10 ms;
  `fog_adjust!` 11.4 → 5.0 ms.

**Not met** — the targets of #308 were **MISSED** and are not changed
after the fact: ≥ 2.5× at 4 threads and ≥ 4× at 12, for the flagship
(1.71× / 1.77×) and for the DNS (1.43× / 1.43×; 48 min against the ≈ 30
min aimed for). The stencils are memory-bound: streaming kernels reach
50–75 GB/s, the DRAM bandwidth of this dual-channel desktop, while
compute-bound kernels such as WENO-Z5 reach 8.6×. An Amdahl fit gives a
serial fraction of 45–53 %, against the 10–15 % estimated; most of it is
parallel code that stops at 1.5–2× on bandwidth or on launches too
small to split (the 64³ flagship). Also open:
- FFTW MEASURE plans are not thread-invariant (3 of 13 solvers differ
  at 64×64×32): production keeps MEASURE, the tests use ESTIMATE.
  MEASURE planning of a threaded 256³ c2r transform segfaults at 10–14
  threads (#397).
- The radiosity ≥ 10× target is met at night only; by day the SW
  operator is refactorised every call (μ₀-dependent albedos).
- Sparse lists (immersed walls, canopy faces) do not thread at the
  flagship size: `launch_weighted!` weights by points, not cost.
- Remaining for Phase 81: the GPU backend, GPU-resident FFT and
  multigrid, SoA surfaces.
- Follow-up: the scaling epic #406, shipped (next block): 256³
  `step_rk3!` 3.03× at 12 threads over its own 1 thread, `dns_step!`
  2.25 → 0.573 s; flagship 2.02× (2.29× with the prefetched parent),
  short of its 3–4× target.

**Bugs the tests caught**:
- A rejected adaptive step left the blown-up NaN `νt`/`κt` in the
  simulation, where diagnostics and crash checkpoints read them; the
  rollback now restores them (KA-j, #346).
- The process-global `WeakKeyDict` step workspaces raced when two
  domains were stepped from different tasks (KA-j).
- The Mie log-derivative buffer was one global vector: two tasks
  calling `mie_sphere` at once corrupted each other's recursion (KA-m's
  first-use test).
- `CGPoissonSolver` divided by a vanishing p·Ap at breakdown and kept
  going (KA-g).
- A KA-d commit deleted three launch kernels; the suite at `-t 1` missed
  it (every launch is serial there), the threaded benchmarks caught the
  `UndefVarError`.
- The 1-thread gate (new ≤ 1.10× old) caught performance bugs the tests
  could not: widened `Vararg` arguments (axpy 375× the old loop), an
  in-place aliasing path (1.8×), a boxed accumulator in `column_sums`
  (roughness 97–101 ms against 8.3 ms), loads not hoisted from the
  argument tuple (WENO-5 6.3 against 2.1 ms).
- The harness worker at `-t 1` left OpenBLAS at its default thread
  count, and LAPACK's bits depend on it (2216 differing smoke hashes);
  the worker now pins BLAS to one thread.

**Expectations that were wrong**:
- `interior_dot` against Base `sum(abs2, view)` at rtol 1e-13: Base's
  sequential sum was the less accurate side (2.9e-13 against 3.1e-16
  for the eight-lane order, both measured against BigFloat).
- A `backends.jl` test field was identically ±0 instead of spanning
  four decades (KA-b).
- `stretched` asked which module defines the StretchedGrid method;
  with one kernel per operator that question is obsolete (KA-d).
- `multigrid` asserted that MG-CG's dot product equals
  `sum(x .* y)`, untrue after the planned reduction-order change; it now
  checks `interior_dot`. Iteration counts are unchanged in all 28
  measured CG cases and 22 multigrid cases (KA-g, KA-i).
- "`init_threads!` keeps the FFTW policy": changed by the maintainer's
  threaded-FFTW decision (KA-h).
- `logging_blowup` expected `[:v, :νt, :κt]` non-finite after a
  rollback: it described the missing restore (KA-j).
- Monte-Carlo shadow test `argmin(profile) == i_expect` assumed one
  darkest cell; the τ = 20 shadow is a six-cell plateau within the
  noise. Three other Monte-Carlo assertions held only for the old
  random realisation; their photon counts were raised, tolerances
  unchanged (KA-n).
- Bit-pattern references re-recorded after named last-bits changes:
  the P3 precipitation-fraction golden values and the CFG Berlin
  fixture digests after KA-m's fused saturation vapour pressure (≤ 8.3e-16
  relative in e_s over 230–320 K); 13 Berlin fixture digests after
  KA-l's dual-strain gather (Berlin e moves 4.6e-10 m² s⁻² after 12
  steps); and, in the merge train, the `input_dynamic` bitwise qv
  digests, which went stale after KA-m's fused e_s.

### Phase 82 — Multi-GPU and heterogeneous execution
Multi-GPU per node and multi-node GPU with CUDA-aware MPI, plus
heterogeneous splitting where radiation or chemistry runs on the CPU
while dynamics runs on the GPU.
**Validated by**: scaling on a GPU cluster; efficiency metrics per
component.

### Phase 83 — Mixed and reduced precision
Float32 and mixed-precision execution with error analysis per
component, plus stochastic rounding where it helps.
**Validated by**: benchmark-suite skill unchanged within tolerance;
memory and bandwidth savings quantified.

### Phase 84 — Performance engineering
Cache-blocking, fused kernels, SIMD verification, memory-layout
optimisation, roofline analysis, and elimination of every remaining
allocation on the hot path.
**Deliverable**: documented roofline position per kernel and a
regression-tested performance budget.

#### Phase 84a — Multithreaded scaling beyond the KA port (epic #406)

The follow-up to KA-o (#308): the stencils are bound by memory bandwidth
and launch overhead, so move fewer bytes and launch fewer, larger
kernels.

**✅ SHIPPED** (SC-a #401 / PR #410, SC-b #402 / PR #409, SC-c #403 /
PR #411, SC-d #404 / PR #412, SC-e #405 / PR #413, SC-f #400 / PR #408;
design `docs/src/design/ka_port.md`, sections SC-a to SC-e; nesting
`docs/src/theory/nesting_time.md`). What was built:
- **Roofline profile and pinning (SC-a).** `benchmarks/bench_roofline.jl`
  times every launch of a step and books its compulsory bytes against
  STREAM: the 256³ DNS step was 98 % bandwidth-bound in its launches.
  Pinning (`JULIA_EXCLUSIVE`, `taskset`, one CCD) is within 5 % of
  unpinned at 12 threads and 18 % slower at 4: the launch setting stays
  `julia -t 12`, no pinning.
- **Threaded halo fills (SC-b).** `fill_halos!` of a field set (tuple or
  vector) fills every field in one threaded launch per phase (x/y phases
  by k-planes, z by j-rows). A set of 4 fields at 192×192×53, 12
  threads: 458 → 156 µs.
- **Concurrent parent and child (SC-c).** One-way nests prefetch the
  parent's next window on its own task (`prefetch_parent_step!`,
  `step_nested!(run, n; concurrent = true)`); the Berlin nest does so
  under `run_case!`. Two-way nests and `:shared_min_dt` stay sequential
  (the chain is serial). Bit-identical to sequential stepping.
- **Kernel fusion (SC-d).** Tendencies written once (advection + ν∇²,
  no memset or scratch), the divergence straight into the spectral and
  column solvers' source buffer (`source_buffer`,
  `solve_source_buffer!`), one correction pass that also writes p and
  the RK3 combination. DNS bytes per step 29.1 → 14.9 GB, threaded
  launches 107 → 55. Bit-identical to the pre-fusion cores.
- **Temporal blocking (SC-e).** After SC-d every remaining kernel ran at
  88–120 % of the DRAM triad, so spatial blocking had nothing to win.
  `FiniteVolume.blocked_predictor!` sweeps the tendencies and the
  predictor slab by slab, row by row, in cache, for the hook-free core
  (the DNS): those kernels' compulsory bytes 7.92 → 2.26 GB per step.
- **DNS statistics on the reduction primitives (SC-f, #400).** Skewness,
  ε and the CFL maximum moved off serial broadcasts (0.661 + 0.223 +
  0.161 s → 0.064 s per step at 12 threads).

**Measured** (Ryzen 9 7900, 12 cores; each figure the minimum over 3
processes started at 1-minute load < 4, from the PRs; base and new code
run alternately):

| | before the epic | after | speed-up |
| :-- | --: | --: | --: |
| 256³ `step_rk3!`, 1 thread | 2.13 s | 1.545 s | 1.38× |
| 256³ `step_rk3!`, 12 threads | 1.16 s | 0.509 s | 2.28× |
| 256³ `step_rk3!`, 12 t over 1 t of the same code | 1.83× | 3.03× | |
| 256³ `dns_step!`, 12 threads | 2.25 s | 0.573 s | 3.93× |
| 256³ DNS run, 1264 steps at 12 threads | 47 min | 12 min | |
| Flagship `step_case!`, 12 threads, sequential parent | 92.1 ms | 81.1 ms | 1.14× |
| Flagship, 12 threads, prefetched parent (SC-c) | 95.2 ms | 69.8 ms | 1.36× |
| Flagship, 12 t over 1 t | 1.81× | 2.02× sequential, 2.29× prefetched | |

Before: master `c2a579a5` (SC-b's base). After: SC-e's measurement
(DNS, sequential flagship) and SC-c's (prefetched flagship, measured
before SC-d and SC-e merged; the flagship cores do not take SC-e's
blocked path, and SC-d moved the sequential 12-thread flagship
83.5 → 76.6 ms). Prefetch and fusion together were not timed on one
master: when the merge train finished the machine was running a 12-thread
Berlin run (1-minute load ≈ 12). The DNS run time is 1264 × `dns_step!`.

**Target missed**: the flagship's 3–4× at 12 threads. It reaches 2.0×
(2.3× with the prefetched parent): its 64³ child and the nest exchange
are small launches, and SGS, buoyancy and the hooks keep it off SC-e's
blocked path. The DNS, which is one large hook-free core, reaches 3.0×.
Next for the DNS: the FFTs of the three Poisson solves (about 0.29 s of
the 0.51 s step, outside launches).

**Bugs the tests caught**:
- `NestedRun.ahead` was an `Any` field; the #329 typed-state test caught
  it (SC-c).
- `step_rk3!` allocated 272 bytes per step after the fusion (a mixed
  Field/Array tuple in `interior_map!`); the allocation tests caught it,
  now 16 bytes (SC-d).
- Writing all five tendencies in one launch stopped vectorising (39.2
  against 18.9 ms at 1 thread); the 1-thread gate caught it (SC-d).
- The sweep's row loops, inlined, did not vectorise (2× slower at 1
  thread); split into `@noinline` `@simd ivdep` functions (SC-e).

**Expectations that were wrong**:
- The thread-invariance worker stepped a variable-step case for nsteps
  steps, not to its horizon, so threaded runs prefetched a parent window
  the loop never reached (90 of 3825 hashes differed, parent only); the
  harness loop was wrong, not the code (SC-c).
- max|∇·u| ≈ 0 after an RK3 step from a non-solenoidal start: RK3
  combines with the step-start state, so the result is solenoidal only
  if that state is (SC-d's own new test, twice).
- `ka_fusion` compared the tendency fields on the hook-free path, where
  SC-e made them unused scratch; it now compares the state (SC-e).
- No reference digest was re-captured for SC-b to SC-e: each is
  bit-identical to the code it replaces (SC-f re-captured the 256³
  record, whose S and ε moved in their last bits with the new summation
  order).

Open: #397 (FFTW MEASURE planning of a threaded 256³ c2r transform
segfaults at 10–14 threads; upstream; workaround
`AEOLUS_FFTW_PLANNER=estimate`, documented in
`benchmarks/bench_ka_scaling.jl`).

### Phase 85 — Asynchronous and in-situ workflows
Asynchronous I/O, in-situ analysis and visualisation, in-transit
compression, and streaming output so that a large run does not stall
on disk.
**Deliverable**: I/O no longer visible in the timing breakdown.

### Phase 86 — Spectral-element backend completion
Finish the SEM backend: tensor-product operators, element-local mass
matrices, matrix-free CG, curved elements, and `p`-adaptivity — giving
high-order accuracy on genuinely complex geometry.
**Validated by**: exponential convergence on smooth problems; the FV
benchmark suite reproduced on aligned SEM meshes.

### Phase 87 — Discontinuous Galerkin option
A DG variant for problems needing shock capturing or extreme
`hp`-adaptivity, sharing the same physics modules.
**Validated by**: shock-tube and compressible-turbulence cases.

---

# Machine learning, applications, ecosystem

*With a differentiable, validated, fast model in hand, the payoff
milestone.*

### Phase 88 — Neural SGS closures
Train SGS models on filtered DNS and superdroplet reference data,
with online (solver-in-the-loop) training via the Phase 74 adjoint,
plus stability guarantees and physical-constraint enforcement.
**Validated by**: a-posteriori skill on unseen flow regimes; stability
over long integrations.

### Phase 89 — Neural emulators for expensive physics
Emulate radiation, chemistry and microphysics with neural networks at
10²–10³× speed-up, with conservation constraints and a fallback to the
reference solver when the emulator leaves its training distribution.
**Validated by**: benchmark-suite skill preserved; measured speed-up.

### Phase 90 — Differentiable-physics applications
End-to-end optimisation problems: optimal sensor placement, optimal
urban-geometry design for ventilation and cooling, control of building
systems, all via gradients through the solver.
**Deliverable**: worked design-optimisation case studies.

### Phase 91 — Foundation-model coupling
Use AI weather models (GraphCast, Pangu, Aurora, AIFS) as boundary
forcing and as a background for assimilation, and study hybrid
AI-physics forecasting at the urban scale.
**Validated by**: forecast-skill comparison against reanalysis-driven
runs.

### Phase 92 — Wildfire coupling
Coupled fire-atmosphere modelling: fuel model, spread-rate
parameterisation, fire-induced buoyancy and winds, smoke emission and
its radiative feedback.
**Validated by**: FireFlux experimental data; historical fire cases.

### Phase 93 — Wind energy
Actuator-line and actuator-disc turbine models, wake interaction,
wind-farm control, and terrain-induced flow effects.
**Validated by**: wake profiles against wind-tunnel and field
measurements; comparison against SOWFA.

### Phase 94 — Urban climate services
Operational-grade workflows: heat-island mitigation scenario
comparison, thermal-comfort indices (UTCI, PET, MRT), pedestrian wind
comfort, ventilation-corridor analysis, and climate-change-scenario
downscaling.
**Deliverable**: a reproducible scenario-comparison report for a real
city district.

### Phase 95 — Air-quality nowcasting
Operational dispersion and air-quality forecasting with data
assimilation of monitoring-station data, source attribution, and
exposure mapping.
**Deliverable**: a running nowcast for a test city, evaluated against
withheld stations.

### Phase 96 — Emergency response
Hazardous-release modelling: dense-gas dispersion, indoor-outdoor
exchange, evacuation-relevant exposure fields, and backward source
attribution from sensor detections.
**Validated by**: the Prairie Grass, Kit Fox and Jack Rabbit release
datasets.

### Phase 97 — Aviation and contrails
Contrail formation and persistence, wake-vortex evolution, airport
microclimate, and the contrail-avoidance optimisation problem.
**Validated by**: contrail-persistence comparison against satellite
observations.

### Phase 98 — Coupled Earth-system components
Two-way coupling to `Oceananigans.jl` for air-sea interaction, to
hydrological models for urban flooding, and to `SpeedyWeather.jl` /
`ClimaAtmos.jl` for large-scale context — via a proper coupler with
conservative flux exchange.
**Validated by**: conservation across the coupling interface;
sea-breeze and coastal cases.

### Phase 99 — Ecosystem packages
`AeolusAnalysis.jl` (post-processing, spectra, budgets, conditional
sampling), `AeolusViz.jl` (GPU volume rendering, web export),
`AeolusDA.jl`, `AeolusCases.jl` (the benchmark library as a package),
and `AeolusCoupler.jl`.
**Deliverable**: registered packages with their own docs and tests.

### Phase 100 — Community, documentation and science
A full theory monograph, API-stability guarantees with semantic
versioning, contributor guide, tutorial series, JOSS and Geoscientific
Model Development papers, participation in model-intercomparison
projects, and a governance model.
**Deliverable**: Aeolus as a citable, community-governed model.

---

---

# Kilometre-scale regional and climate modelling

*A future direction, not part of the decade plan above (tracking
issue #289). Through Phase 100
Aeolus is a large-eddy model: grid spacings of metres to tens of
metres over a few kilometres, with the mesoscale entering from outside
(Phase 68). Phases 101–112 would make Aeolus itself a convection-permitting
regional model at about 3 km, in the class of ICON-LAM, COSMO-CLM and
REMO-NH, so that one code and one physics library cover the scales
from a European domain down to a Berlin street canyon.*

**Why it sits here.** Every piece of it depends on something earlier:
- the cost needs MPI and GPU (Phases 80–87);
- the driving data needs the reanalysis readers (Phase 67) and
  the nesting interface of Phase 68 (#58, and the Berlin nesting epic
  #246);
- long-run conservation needs the full-state checkpointing of #245;
- the grey-zone physics needs the turbulence closures (Phases 54–59) to blend into.

A 3 km domain the size of EURO-CORDEX is roughly 1500 × 1500 × 60
≈ 1.4 × 10⁸ cells. On the present single-threaded solver one time step
would take of order 10 s, and a 30-year run of order a decade of
single-core wall time before any physics. This is an HPC-only
capability.

**What changes in kind, not degree.** At 3 km the model no longer
resolves the energetic boundary-layer eddies, so the LES closures stop
being valid. The full troposphere and the orography enter the
dynamics, and the domain is a piece of a sphere driven at its edges.
The phases below address each of these. A phase is done by the same
standard as every other milestone: implemented to the reference literature,
validated, stable at production resolution.

### Phase 101 — Compressible non-hydrostatic dynamical core
A fully compressible core for a deep atmosphere (about 30 km):
- split-explicit acoustic substepping with a horizontally explicit,
  vertically implicit treatment (Wicker & Skamarock 2002; Klemp,
  Skamarock & Dudhia 2007);
- the same physics interface as the Boussinesq and anelastic cores, so
  every physics module runs on all three.

**Validated by**: density current (Straka et al. 1993); inertia-gravity
waves in a channel (Skamarock & Klemp 1994); warm bubble; a
baroclinic-wave test in a channel against the published reference.

### Phase 102 — Orography in the core
A terrain-following core on the existing `TerrainFollowingGrid`
(Phase 24), using a smooth-level coordinate (SLEVE; Schär et al. 2002)
with metric terms and controlled pressure-gradient errors over steep
slopes. Cut cells (Phase 21) are the alternative for very steep relief.

**Validated by**: Schär et al. (2002) mountain-wave test; a resting
atmosphere over steep idealised terrain (spurious wind below a stated
bound); a flow-over-hill case against linear theory.

### Phase 103 — Grids on the sphere: one abstraction, several grid types
Several grid types behind **one grid abstraction**, so that the
dynamical core and every physics module run unchanged on any of them.
The grids chosen for regional work must also be usable, later, for
global models.

- **The abstraction.** Operators are written against general
  curvilinear metric terms: cell areas and volumes, edge lengths, the
  local basis and its rotation, map factors, latitude for Coriolis,
  and deep-atmosphere radius factors. The topology is a set of
  logically rectangular **panels** with explicit connectivity, so that
  halo exchange crosses panel edges (with the vector-component rotation
  done there) as it later crosses MPI ranks (Phase 80). The existing
  Cartesian box becomes one panel with trivial metrics, so the urban
  LES is the special case, not a separate code path.
- **Rotated-pole latitude–longitude** (one panel), the REMO and
  COSMO-CLM choice for regional climate. It is the primary regional
  grid, because it keeps the staggered C-grid stencils and the
  KernelAbstractions kernels of #291.
- **Conformal map projections** (Lambert conformal, polar
  stereographic, Mercator; one panel each), as in WRF, through map
  factors.
- **Cubed sphere** (six panels, equiangular gnomonic). **Global.** Its
  panels are structured, so the same stencils and kernels apply, with
  rotated halos at the cube edges. A regional domain can be a subset of
  panels or a single stretched panel (Schmidt transform), so regional
  and global runs share one grid family. The spectral-element backend
  (Phase 86) on curved elements gives a high-order path on the same
  grid; ClimaAtmos and HOMME use that approach.
- **Unstructured icosahedral or Voronoi grids** (ICON, MPAS): an
  explicit later option, not part of this phase. It needs
  unstructured operators (no i/j/k stencils) and would most naturally
  come through the DG and SEM backends (Phases 86–87). The abstraction
  keeps the door open by not baking i/j/k indexing into the physics
  interfaces (column physics already works per column).

**Validated by**:
- the Williamson et al. (1992) shallow-water test suite on the cubed
  sphere, which is global: solid-body advection over the poles and
  across panel edges, and steady geostrophic flow, with errors and
  convergence order quoted;
- solid-body advection across the rotated pole, and geostrophic
  balance preserved on the rotated and projected grids;
- a baroclinic wave (DCMIP-style) on the cubed sphere, with the
  regional subset against the global run;
- the Cartesian box unchanged bit for bit when run through the
  abstraction.

### Phase 104 — Lateral and top boundary driving
- Davies (1976) relaxation zones at the lateral boundaries.
- Upper-boundary damping toward the driver.
- Optional spectral nudging of the large scales (von Storch et al.
  2000).
- Driving from ERA5 and from CMIP-class global model output, built on
  the boundary-data contract of the nesting epic (#246) and the
  real-world data readers (Phases 67–73).
- Mass and energy consistency of the boundary fluxes.

**Validated by**: a perfect-model "big brother" experiment (Denis et
al. 2002); driving drift and boundary noise quantified.

### Phase 105 — Grey-zone boundary-layer turbulence
A scale-aware boundary-layer scheme, for example an eddy-diffusivity
mass-flux (EDMF) TKE closure (Siebesma et al. 2007), blended by grid
spacing with the LES turbulence closures (Honnert, Masson & Couvreux 2011;
Boutle et al. 2014). The same model then runs continuously from LES to
3 km, without switching physics by hand.

**Validated by**: the resolution dependence of a convective boundary
layer from 100 m to 3 km against coarse-grained LES; GABLS1 and a
shallow-cumulus case (BOMEX) at mesoscale resolution.

### Phase 106 — Convection and clouds at kilometre scale
- Scale-aware shallow convection, and an optional deep-convection
  scheme for grids coarser than about 5 km.
- Subgrid cloud cover from a statistical cloud scheme.
- The cloud microphysics (two-moment, P3) and the aerosol–cloud
  coupling, at kilometre scale.

**Validated by**: an idealised supercell; radiative–convective
equilibrium; the diurnal cycle of convective precipitation (timing and
intensity) against radar.

### Phase 107 — Land surface and hydrology for climate time scales
Multi-year soil hydrology with groundwater, frozen soil and snow,
river routing, lakes, vegetation phenology, and land-use tiles at
kilometre scale (`SurfaceTiles`), built on the land-surface modules.

**Validated by**: multi-year eddy-covariance sites (FLUXNET); river
discharge at gauged catchments; snow-cover duration.

### Phase 108 — Radiative forcing for climate
- Greenhouse-gas concentration scenarios.
- Aerosol climatologies, for example MACv2-SP (Stevens et al. 2017).
- Ozone and solar variability.
- Sea-surface temperature and sea ice from the driver or from the
  ocean mixed layer.
- A radiation call frequency with a stated error.

**Validated by**: offline radiative-forcing comparison against
line-by-line reference cases; top-of-atmosphere and surface energy
budgets against satellite and surface-radiation-network climatologies.

### Phase 109 — Long-run conservation and production output
- Domain mass, water and energy budgets closed and monitored over
  decades.
- Drift detection.
- Bit-reproducible restarts (building on #245).
- Output in the CORDEX data-request conventions, written by the
  in-package NetCDF writer.

**Validated by**: a 10-year run with stated budget closure and no
unexplained drift.

### Phase 110 — Convection-permitting evaluation
A multi-year ERA5-driven hindcast at about 3 km over Central Europe and
the Alps, in the protocol of the CORDEX Flagship Pilot Study on
convective phenomena (Coppola et al. 2020; Ban et al. 2021). It is
evaluated against gridded observations, radar and hourly station
precipitation, with emphasis on extremes, and set beside the published
convection-permitting ensemble (COSMO-CLM, ICON, REMO, WRF).

**Validated by**: skill within the spread of that ensemble for the
diurnal cycle, the seasonal precipitation climatology and hourly
extremes.

### Phase 111 — Scale-seamless nesting
Online multi-level nesting inside one Aeolus run, from the 3 km
regional domain through about 1 km to the urban LES of the Berlin case.
Physics changes consistently with grid spacing through Phase 105, and
there is two-way coupling with the urban scale. The Cartesian urban
child sits in the curved parent as a local tangent plane, with the
interpolation between the two geometries (metric terms, vector
rotation) done in the nesting operators. At about 1 km the
tangent-plane error is negligible, but it is designed in and tested,
not assumed.

**Validated by**: the urban-scale result against the
offline-driven Berlin case (#246); conservation across all levels.

### Phase 112 — Climate-change downscaling
- Scenario runs driven by CMIP-class global models.
- The pseudo-global-warming method (Schär et al. 1996).
- Ensembles with the uncertainty machinery of Phases 74–79.
- Urban climate projections, for example Berlin heat-wave statistics,
  from the full chain down to street scale.

**Validated by**: historical-period skill (Phase 110) and a documented
uncertainty budget for the projected change.

**Milestone.** Aeolus is a convection-permitting regional
climate model, evaluated against the published ensemble, that nests
seamlessly down to its own urban LES.

## Cross-cutting commitments

These apply to *every* phase above, not to any one of them:

  * **Validation first.** A phase starts by identifying its reference
    data and ends by reporting skill against it. No phase ships on
    "it runs and looks plausible".
  * **No silent approximations.** Every parameterisation documents its
    assumptions, its validity range, and what it omits. Where a cheaper
    option exists it is selectable, never the silent default.
  * **Regression-tested.** The benchmark suite (Phases 71–72) runs in
    CI from Phase 67 onward; earlier milestones add their validation cases to
    it as they land.
  * **Performance budget.** Each phase records its cost. A phase that
    makes the flagship run more than ~20 % slower needs either an
    optimisation plan or an opt-in flag.
  * **Docs as first-class.** Theory page, API reference, and a worked
    example per phase.

## Honest assessment

This is a decade of work and it assumes sustained effort, domain
expertise across meteorology, cloud physics, chemistry, turbulence and
numerical analysis, and access to reference data and HPC resources. A
solo contributor should expect to reach the end of Clouds and microphysics (Phase 45) in that
window, not the end of Phase 100.

The ordering is the important part. Physics and numerical fidelity come
first because everything downstream — assimilation, ML, applications —
inherits their errors. Performance comes late deliberately: there is no
value in efficiently computing the wrong answer, and optimising before
the physics settles freezes interfaces that still need to change.

Kilometre-scale modelling changes what kind of model Aeolus is: from a large-eddy model
with the mesoscale entering at its boundaries to a regional model that
contains its own mesoscale. Projects of that kind (ICON, COSMO-CLM,
REMO) are built and maintained by institutional teams over many
years. It belongs on the roadmap because the payoff, one physics
library from the regional domain down to the street canyon, is
distinctive. It should be started only once real-world data (Phases 67–73) and
performance (Phases 80–87) are shipped, and broken into issues only when it comes into view.

The kilometre-scale model is deliberately **not** a global climate model. But its grid
abstraction (Phase 103) is designed so a global one stays possible: the
cubed sphere is global from the start, and the panel and metric
abstraction leaves room for unstructured grids later. Global context in
the meantime comes from coupling to existing global models (Phase 98)
and from lateral driving (Phase 104).
