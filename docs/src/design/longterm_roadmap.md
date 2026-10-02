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

# Era III — High-fidelity radiation ✅ SHIPPED

**Status: all seven phases (32–38) and the shared contract are
complete** (Phases 34–37 landed together in `ac38822`; the era test set
is 4,912 tests on master at `2895fed`). `src/RadiationCore.jl` fixes the interface every component in
this era speaks — `SpectralGrid` (RRTMG-layout bands with correlated-k
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

### Phase 32 — Correlated-k gas optics ✅ SHIPPED
`src/GasOptics.jl`. Full spectral treatment of H₂O, CO₂, O₃, CH₄, N₂O
and O₂ across the 16 RRTMG longwave and 14 shortwave bands, on the
model's own vertical grid, replacing the prescribed LW-down constant.

**Design deviation from the original plan, and why.** The plan said
"bind `RRTMGP.jl`". That is not possible on this host: RRTMGP.jl pulls
in NetCDF/HDF5 for its k-distribution files, and those binary
dependencies do not load here (the same `libmpifort.so` failure that
forced the Serialization+TOML I/O layer in Era I). Rather than ship a
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

### Phase 33 — Two-stream solver with cloud optics ✅ SHIPPED
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

### Phase 34 — 3-D Monte-Carlo radiative transfer ✅ SHIPPED
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

### Phase 35 — Full urban radiosity ✅ SHIPPED
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
comparison against BUBBLE / CAPITOUL was done in Era V (Phase 49, #30).
Glass is Lambertian here; the angle-dependent glazing is Phase 49 (#27).

### Phase 36 — Spectral surface albedo and BRDF ✅ SHIPPED
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

### Phase 37 — Aerosol and gas radiative feedback ✅ SHIPPED
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

### Phase 38 — Solar-position and cloud-shadow accuracy ✅
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
The Era-I sinusoidal declination it replaces is in error by up to
**0.79°** near the equinoxes.
**Accuracy is stated honestly**: this is Meeus-grade (~0.01°), not full
NREL SPA (±0.0003°). The full SPA needs ≈ 350 published VSOP87
coefficients; transcribing those from memory would risk silent errors,
which is worse than a documented 0.01°. `AbstractSolarPosition` is the
extension point for a table-carrying `SPAPosition`. For radiative
transfer 0.01° is not limiting — the solar disc itself subtends 0.53°.

---

# Era IV — High-fidelity clouds and microphysics ✅ SHIPPED

*Move from the Kessler/two-moment scaffold to schemes that reproduce
observed cloud microstructure and precipitation.*

**Status: all seven phases (39–45) are complete.** Phases 39, 41 and 42
landed before the issue tracker (`622184e`, `63fc020`); Phases 40 and
43–45 are issues #1–#14 (PRs #72–#89), plus the contract fix #82
(PR #91) and the P3 size-limit deduplication (PR #87). Milestone 4 has
15 closed issues and none open. The era test set is 47,392 tests on
master at `2895fed`; about 29,850 of them are the 14,880-state enthalpy
sweep of #91. Shared contract: `src/MoistThermodynamics.jl`, which
every scheme of the era speaks and which also owns Köhler theory
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
with droplet-radiative feedback through the Era III stack).
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

# Era V — High-fidelity land, urban and surface exchange ✅ SHIPPED

*Make the surface as good as the atmosphere. Currently the biggest
source of error in the urban runs.*

**Status: all eight phases (46–53) and the shared contract are
complete** — issues #15–#43 and #92, PRs #93–#122. Milestone 5 has 30
closed issues and none open. The era test set is 19,326 tests (6 `@test_broken`) on
master at `2895fed`. Shared contract (`src/SurfaceExchange.jl`, #92 /
PR #93): material constants that read latent heats, c_p and g from
MoistThermodynamics rather than restating them; Monin–Obukhov
similarity as a dispatch hierarchy (Businger–Dyer / Paulson unstable,
Beljaars–Holtslag stable) solved for ζ by a bracketed Illinois
iteration; and the `AbstractSurface` interface (`surface_temperature`,
`roughness_lengths`, `step_surface!`, the RadiationCore albedo and
emissivity generics) that every Era V surface implements. Measured:
neutral C_D 2.7e-16 from (κ/ln(z/z0))²; ψ against Paulson and Beljaars–
Holtslag to 6 digits; returned L against the Obukhov length of its own
fluxes 9.1e-15; a very stable case (ΔT = −10 K, U = 2 m/s) converges in
8 iterations. Expectation that was wrong: Ri_B = ζ C_D/C_H (the identity
is ζΦ_h/Φ_m²). Since the era closed, the Berlin case runs this contract
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
incident flux (an Era III fix); β₀ = 0/0 near χ_L = −0.4; a lagged-K
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
street-canyon-resolved radiation coupled to Era III's radiosity.
**Validated by**: façade and surface temperatures against BUBBLE /
CAPITOUL urban field-campaign data.

**✅ SHIPPED** (`src/UrbanSurface.jl` + `BuildingEnergy` coupling,
#26–#30 / PRs #94, #102, #103, #121, #122; GMRES #138 / PR #163). A cited
16-material thermal table (optics derived from the SurfaceOptics library,
not restated); a θ-scheme conduction core; `UrbanFacet <: AbstractSurface`
with the skin balance linearised into a Robin condition; Fresnel /
Beer–Lambert glazing, multi-pane units with ISO 15099 gaps,
`WindowFacet`; implicit facet–facet longwave through the Era III
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

# Era VI — High-fidelity turbulence closure

*The SGS model is the heart of LES. Make it defensible.*

**Status: not started** (milestone 6: six issues, all open). Groundwork
from the Berlin epic (#125), which does not satisfy Phase 54: the local
Germano–Lilly scaffold moved into `Operators` as
`compute_dynamic_smagorinsky_νt!` with Lilly plane averaging, after
its M_ij sign was found wrong (C_s = 0 everywhere); in decaying
turbulence it gives C_s 0.129 at t = 1 and 0.153 at t = 3 (Lilly 0.17),
in the Berlin `:smoke` free stream only 0.077 (#132 / PR #160). There is
no scale-dependent or Lagrangian-averaged model yet.

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
