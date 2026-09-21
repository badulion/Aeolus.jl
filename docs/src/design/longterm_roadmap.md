# Long-range roadmap — Eras II through XI

This document plans the next **decade** of Aeolus development. It sits
downstream of two things already delivered:

  * **Era I — Foundations** (Phases 1–20): the LES core, staggered
    Arakawa-C grid, pluggable pressure solvers, SGS closure,
    fractional-step projection, IBM, I/O, and the scaffolding for MPI /
    GPU / SEM / nesting.
  * **The post-Phase-20 physics-gap pass** (27 items R1–R6, L1–L5,
    F1–F3, M1–M4, A1–A4, N1–N5, G1–G3): radiation with shadows and
    sky-view factors, latent heat, anthropogenic heat, multi-layer
    soil, MOST, dynamic Smagorinsky, two-moment microphysics,
    four-species chemistry, aerosol, and regional-coupling scaffolds.

## Guiding principle for everything below

> **No scaffolds. Every phase ships a production-quality, validated
> capability.**

Era I deliberately traded depth for breadth — it wired interfaces so
later work would have somewhere to plug in. That phase is over. From
Era II onward, a phase is only "done" when:

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
scaffolds. Where Era I left a scaffold, the corresponding Era II–IV
phase's job is to *replace* it with the real thing.

## Scope

Roughly **5× the work completed so far**, organised as 10 eras of 8–12
phases each — about **100 phases**, each 2–6 weeks. A small team could
do this in 4–5 years; a solo contributor should plan on 8–10.

The ordering is deliberately physics-first: Eras II–IV close every
approximation in the current physics before the project chases scale,
ML, or applications. A high-fidelity single-node model is more useful
than a fast wrong one.

---

# Era II — Numerical foundations for fidelity ✅ SHIPPED

*Replace every numerical compromise that currently forces artificially
high viscosity or small time steps. Without this era, none of the
physics work downstream can run at realistic parameters.*

**Status: complete.** 2,094 validation tests, all green. Every phase
ships a production implementation with measured validation against an
analytic solution, a manufactured solution, or a convergence study —
numbers recorded per phase below.

### Phase 21 — Cut-cell immersed boundary method ✅
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
**Unlocks** `ν = 10⁻⁵` in urban geometry — the whole point of the era.

### Phase 22 — Ghost-cell wall reconstruction ✅
`src/CutCell.jl`. Image/foot-point stencils on the *true* surface with
trilinear donor weights (solid donors dropped, inverse-distance
fallback), Dirichlet and Neumann reconstruction, wall shear stress on
the true normal.
**Validated:** linear profiles reproduced **exactly** (max error 0.0);
quadratic-profile errors 1.06e-3 → 2.66e-4 → 6.64e-5 at Nz = 16/32/64,
i.e. convergence ratios **4.0000 and 4.0** — clean second order;
analytic wall stress `νᾱ = 0.0375` recovered as 0.0375 with zero spread.

### Phase 23 — Adaptive time stepping with substepping ✅
`src/TimeIntegration.jl`. Per-process CFL diagnostics reporting each
limiting `dt` separately (advective via the true per-cell sum form,
diffusive, Brunt-Väisälä, Coriolis, SGS-scalar) plus which one binds;
growth-limited adaptive `dt`; process substepping; and retry-on-failure
with bit-exact preallocated state rollback.
**Validated:** hand-computed advective `dt` to 1e-12 with the right
`limiter` symbol; growth limiting never exceeded; deliberate
over-large-`dt` blow-up triggers rollback and recovers a finite state.

### Phase 24 — Stretched and terrain-following vertical coordinate ✅
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

### Phase 25 — Semi-implicit IMEX time integration ✅
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

### Phase 26 — WENO-Z and TENO advection for all prognostics ✅
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

### Phase 27 — Anelastic and pseudo-incompressible dynamics ✅
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

### Phase 28 — Energy- and enstrophy-conserving discrete operators ✅
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

### Phase 29 — Multigrid pressure solver ✅
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

### Phase 30 — Grid nesting with proper two-way feedback ✅
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

### Phase 31 — Adaptive mesh refinement ✅
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

### Era II — known limitations carried forward

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

# Era III — High-fidelity radiation

*Replace the prescribed-profile and Beer's-law approximations with a
real radiative-transfer solver, including 3-D effects that matter in
urban geometry.*

### Phase 32 — RRTMGP correlated-k gas optics
Bind `RRTMGP.jl` for shortwave and longwave gas optics: full spectral
treatment of H₂O, CO₂, O₃, CH₄ absorption on the model's own vertical
grid, replacing the prescribed LW-down constant.
**Validated by**: clear-sky flux profiles against RFMIP reference
line-by-line calculations.

### Phase 33 — Two-stream solver with cloud optics
Two-stream (delta-Eddington) solution with cloud liquid/ice optical
properties derived from the microphysics state. Full cloud-radiative
feedback on the thermodynamics.
**Validated by**: DYCOMS-II RF01 radiative-flux profiles; comparison
with published LES intercomparison ensemble.

### Phase 34 — 3-D Monte-Carlo radiative transfer
Photon-tracing solver that handles the genuinely 3-D radiative
environment of an urban canyon or a broken cloud field: side
illumination, inter-reflection, and shadowing without the
one-dimensional plane-parallel assumption. Runs as a periodic
correction to the two-stream solution.
**Validated by**: I3RC intercomparison cases; comparison of canyon
absorption against the two-stream + SVF approximation.

### Phase 35 — Full urban radiosity
Replace the sky-view-factor approximation with a view-factor matrix
between every pair of urban surface patches (roofs, walls, ground),
solved by radiosity iteration for both shortwave inter-reflection and
longwave exchange. Includes specular treatment for glass.
**Validated by**: energy-balance closure; comparison against measured
façade temperatures from an urban field campaign (e.g. Basel BUBBLE).

### Phase 36 — Spectral surface albedo and BRDF
Wavelength-dependent albedo per material with a bidirectional
reflectance distribution function, so low-sun-angle reflection off
asphalt, glass, and vegetation is correct.
**Validated by**: comparison against spectral-albedo measurements for
common urban materials.

### Phase 37 — Aerosol and gas radiative feedback
Couple the aerosol and chemistry state into the optics: aerosol direct
effect, absorption by black carbon, and the radiative impact of ozone.
**Validated by**: aerosol-optical-depth-to-radiative-forcing comparison
against AERONET-constrained cases.

### Phase 38 — Solar-position and cloud-shadow accuracy
Full astronomical solar position (NREL SPA algorithm), atmospheric
refraction, proper direct/diffuse partitioning with a clear-sky index,
and true 3-D cloud shadows projected from the resolved cloud field.
**Validated by**: irradiance comparison against pyranometer time series
for a real site and date.

---

# Era IV — High-fidelity clouds and microphysics

*Move from the Kessler/two-moment scaffold to schemes that reproduce
observed cloud microstructure and precipitation.*

### Phase 39 — Production saturation adjustment and thermodynamics
Exact moist thermodynamics: liquid-water potential temperature as a
prognostic, iterative saturation adjustment with the full latent-heat
temperature dependence, and consistent treatment of moist buoyancy
including condensate loading.
**Validated by**: moist-adiabatic ascent against analytic parcel theory;
conservation of moist static energy.

### Phase 40 — P3 microphysics
Implement the Morrison-Milbrandt P3 scheme: predicted particle
properties, a single prognostic ice category with predicted rime mass
and density, four-moment rain.
**Validated by**: comparison against P3 reference implementation on
idealised cases; radar-reflectivity comparison for a squall line.

### Phase 41 — Superdroplet Lagrangian microphysics
Monte-Carlo superdroplet method as an alternative to bulk schemes:
individual particle attributes, stochastic collision-coalescence,
explicit condensation growth. The reference against which bulk schemes
are evaluated.
**Validated by**: Golovin analytic collision-coalescence solution;
comparison against bin-microphysics reference for a warm cumulus.

### Phase 42 — Aerosol-cloud interaction
Prognostic aerosol with size distribution, Köhler-theory activation to
cloud droplets, in-cloud processing, and regeneration on evaporation.
The first and second indirect effects done properly.
**Validated by**: activation-spectrum comparison against parcel-model
reference; Twomey-effect sensitivity against observations.

### Phase 43 — Ice nucleation
Heterogeneous ice nucleation with prognostic ice-nucleating-particle
species (dust, biological), immersion and deposition freezing,
secondary ice production (Hallett-Mossop).
**Validated by**: comparison against cloud-chamber measurements;
mixed-phase-cloud persistence in an Arctic stratus case.

### Phase 44 — Precipitation, fog and visibility
Sedimentation with a proper terminal-velocity spectrum, sub-grid
precipitation fraction, radiation fog with the correct droplet-
radiative feedback, and a visibility diagnostic for transport
applications.
**Validated by**: fog onset/dissipation timing against a field campaign;
drop-size-distribution comparison against disdrometer data.

### Phase 45 — Turbulence-microphysics interaction
Sub-grid supersaturation fluctuations and their effect on droplet
growth and activation — the mechanism by which entrainment broadens
droplet spectra.
**Validated by**: spectral-broadening comparison against DNS and
superdroplet reference.

---

# Era V — High-fidelity land, urban and surface exchange

*Make the surface as good as the atmosphere. Currently the biggest
source of error in the urban runs.*

### Phase 46 — Multi-layer soil and hydrology
CLM-style soil column: 10+ layers with Richards-equation water
transport, temperature-dependent hydraulic conductivity, freeze-thaw,
and a water table.
**Validated by**: soil-temperature and moisture profiles against FLUXNET
site observations.

### Phase 47 — Vegetation photosynthesis and stomatal conductance
Farquhar photosynthesis coupled to Ball-Berry / Medlyn stomatal
conductance, so transpiration responds to light, CO₂, humidity and soil
water stress rather than a fixed availability factor.
**Validated by**: latent- and sensible-heat-flux comparison against
FLUXNET eddy-covariance towers across several biomes.

### Phase 48 — Resolved plant canopy
Resolved-canopy drag with a leaf-area-density profile, implicit
treatment so it is unconditionally stable, radiation attenuation
through the canopy, and leaf-energy-balance for leaf temperature.
**Validated by**: within-canopy wind and temperature profiles against
forest-tower observations; wind-tunnel canopy-flow data.

### Phase 49 — Production urban canopy model
Full urban-canopy model matching or exceeding PALM's USM: per-facet
(roof / wall / ground) multi-layer wall conduction, window transmission,
anthropogenic heat with a real building-energy model, and
street-canyon-resolved radiation coupled to Era III's radiosity.
**Validated by**: façade and surface temperatures against BUBBLE /
CAPITOUL urban field-campaign data.

### Phase 50 — Building energy model coupling
Two-way coupled building interior: indoor air temperature, HVAC
demand and its exhaust heat, thermal mass, and occupancy schedules.
The AC-feedback loop that amplifies urban heat islands.
**Validated by**: measured building energy demand vs simulated for an
instrumented building.

### Phase 51 — Snow, ice and frozen surfaces
Multi-layer snowpack with compaction, metamorphism, albedo ageing,
liquid-water percolation and refreezing; lake and sea-ice
thermodynamics.
**Validated by**: snow-depth and albedo evolution against SnowMIP
reference sites.

### Phase 52 — Water surfaces and wave coupling
Lake and coastal-sea surface with a mixed-layer model, wave-dependent
roughness (Charnock and beyond), sea-spray heat flux, and a diurnal
warm-layer scheme.
**Validated by**: sea-surface-temperature diurnal cycle against buoy
observations; air-sea flux comparison against COARE.

### Phase 53 — Surface heterogeneity and blending
Sub-grid tile approach with a blending-height formulation so a single
grid cell can contain multiple surface types and still produce the
right aggregate flux and the right internal boundary layers.
**Validated by**: heterogeneous-surface LES against homogeneous-tile
aggregation; observations from patchy terrain campaigns.

---

# Era VI — High-fidelity turbulence closure

*The SGS model is the heart of LES. Make it defensible.*

### Phase 54 — Production dynamic and scale-dependent SGS models
Lagrangian-averaged scale-dependent dynamic Smagorinsky (Bou-Zeid,
Meneveau & Parlange), fully validated, replacing the local
Germano-Lilly scaffold. Includes the correct near-wall behaviour
without ad-hoc damping.
**Validated by**: log-law recovery with no wall-damping function;
resolved-scale spectra against DNS.

### Phase 55 — Anisotropic and tensorial eddy viscosity
Anisotropic minimum-dissipation (AMD) and Vreman models, plus a full
tensorial eddy-diffusivity formulation for strongly stratified flow
where vertical and horizontal mixing differ by orders of magnitude.
**Validated by**: stably stratified channel against DNS; GABLS1
intercomparison.

### Phase 56 — Prognostic SGS energy and stress
Deardorff prognostic TKE and, beyond it, a full differential
Reynolds-stress SGS model for the cases where eddy viscosity fails
(rapid rotation, strong stratification, flow around sharp obstacles).
**Validated by**: SGS-stress comparison against filtered DNS fields.

### Phase 57 — Wall models for high Reynolds number
Equilibrium and non-equilibrium wall models, dynamic wall models, and a
proper treatment of the log-layer mismatch problem. Required for
wall-modelled LES of realistic urban flow.
**Validated by**: Kim-Moin-Moser channel at several Reynolds numbers;
comparison against wall-resolved reference.

### Phase 58 — Stable-boundary-layer and intermittent turbulence
The hardest regime in boundary-layer meteorology: very stable
stratification, intermittent turbulence, gravity-wave-turbulence
interaction, and the correct behaviour as the flow relaminarises.
**Validated by**: GABLS2, GABLS3, GABLS4 intercomparison cases;
CASES-99 observations.

### Phase 59 — Transition and laminar-turbulent coexistence
Handle flows that are partly laminar: nocturnal boundary layers,
low-Reynolds regions in a cut-cell wake, and the transition process
itself.
**Validated by**: transition location in a flat-plate boundary layer;
relaminarisation in a strongly stable case.

---

# Era VII — High-fidelity chemistry and aerosol

*From four toy species to a mechanism a chemist would accept.*

### Phase 60 — KPP-generated gas-phase mechanism
Real KPP code generation with a full mechanism (MOZART-T1, CB05 or
MCM subset), analytic Jacobian, and a Rosenbrock stiff solver.
Replaces the three-reaction toy scheme.
**Validated by**: box-model comparison against the reference KPP
implementation; ozone-production comparison for a photochemical smog
case.

### Phase 61 — Photolysis rates
Online photolysis using the computed actinic flux from Era III's
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

# Era VIII — Real-world data, forcing and validation

*A model is only as good as what drives it and what it is checked
against.*

### Phase 67 — ERA5 and reanalysis driver
Read ERA5 / MERRA-2 to build initial and boundary conditions:
vertical profiles, geostrophic forcing, advective tendencies, surface
conditions, and their time evolution over a multi-day run.
**Validated by**: reproduction of an observed boundary-layer evolution
for several real dates.

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

---

# Era IX — Data assimilation and inverse problems

*Once forward fidelity is real, make the model answer inverse
questions.*

### Phase 74 — Adjoint model via automatic differentiation
Full reverse-mode AD through the solver with `Enzyme.jl`: adjoint of
dynamics, physics and I/O, with checkpointing for long windows.
**Validated by**: gradient verification against finite differences for
every prognostic and parameter.

### Phase 75 — 4D-Var
Strong- and weak-constraint 4D-Var with the Era-VII observation
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

# Era X — Performance, scale and portability

*Deliberately placed after physics: a fast model of the wrong physics
is worthless, and premature optimisation would have frozen interfaces
too early. By this era the physics is settled, so it is safe to
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

# Era XI — Machine learning, applications and ecosystem

*With a differentiable, validated, fast model in hand, the payoff
era.*

### Phase 88 — Neural SGS closures
Train SGS models on filtered DNS and superdroplet reference data,
with online (solver-in-the-loop) training via the Era-IX adjoint,
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

## Cross-cutting commitments

These apply to *every* phase above, not to any one of them:

  * **Validation first.** A phase starts by identifying its reference
    data and ends by reporting skill against it. No phase ships on
    "it runs and looks plausible".
  * **No silent approximations.** Every parameterisation documents its
    assumptions, its validity range, and what it omits. Where a cheaper
    option exists it is selectable, never the silent default.
  * **Regression-tested.** The benchmark suite (Phases 71–72) runs in
    CI from Era VIII onward; earlier eras add their validation cases to
    it as they land.
  * **Performance budget.** Each phase records its cost. A phase that
    makes the flagship run more than ~20 % slower needs either an
    optimisation plan or an opt-in flag.
  * **Docs as first-class.** Theory page, API reference, and a worked
    example per phase.

## Milestones

| After | Aeolus is… |
| :---- | :--------- |
| Era II (Phase 31) | numerically sound enough to run real-Reynolds urban LES without inflated viscosity |
| Era III (Phase 38) | radiatively accurate — 3-D urban radiation, real gas optics |
| Era IV (Phase 45) | cloud-physics-capable at research standard, superdroplet reference included |
| Era V (Phase 53) | a credible urban and land-surface model, validated against field campaigns |
| Era VI (Phase 59) | defensible in the turbulence-modelling literature, including the stable boundary layer |
| Era VII (Phase 66) | a real chemistry-transport model, driven by real inventories |
| Era VIII (Phase 73) | drivable from reanalysis, validated across the canonical benchmark suite, fully reproducible |
| Era IX (Phase 79) | an inverse-problem and uncertainty-quantification platform |
| Era X (Phase 87) | HPC-production: MPI + GPU at scale, high-order backends |
| Era XI (Phase 100) | a community Earth-system-scale tool with an ML-native workflow and real applications |

## Honest assessment

This is a decade of work and it assumes sustained effort, domain
expertise across meteorology, cloud physics, chemistry, turbulence and
numerical analysis, and access to reference data and HPC resources. A
solo contributor should expect to reach the end of Era IV in that
window, not the end of Era XI.

The ordering is the important part. Physics and numerical fidelity come
first because everything downstream — assimilation, ML, applications —
inherits their errors. Performance comes late deliberately: there is no
value in efficiently computing the wrong answer, and optimising before
the physics settles freezes interfaces that still need to change.
