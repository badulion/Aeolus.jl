# The Berlin case: numerics and the dynamical core

## Numerics slots

Four exclusive slots swap the numerics of the dynamical core and the
transport (#132). Each slot is one section whose selector key names the
provider (ONE-m, #476; the component name of each provider in
parentheses below): `[pressure] solver = "multigrid"`, `[sgs] model =
"smagorinsky"`, `[momentum_advection] scheme = "weno_z"`,
`[scalar_advection] model = "advection_schemes"`. The section holds the
chosen provider's options and no other's (`--set
pressure.solver="multigrid" --set pressure.tol=1e-8` on the command
line: a selector set by `--set` drops the file's options of the old
provider).

| slot | providers (default first) | how |
| :-- | :-- | :-- |
| `:pressure_solver` | `:cosine_fft` (`pressure_fft`), `:multigrid` (`pressure_multigrid`), `:mg_cg` (`pressure_mg_cg`), `:density_weighted_direct` (`pressure_density_direct`, sound-proof cores only, the direct FFT/DCT + tridiagonal solve, #309), `:density_weighted_cg` (`pressure_density_weighted`, sound-proof cores only, #133) | `sim_options`: the `solver` of `BoussinesqSimulation`, built with the per-axis Poisson kinds of the case's velocity `bcs` (`case_poisson_axes(cfg, mesh)`, #251: periodic x, y and Neumann z by default, all-Neumann for `lateral = :open` or a nested child; smoke-checked); for a sound-proof core `pressure_solver = :direct` or `:cg` (with the CG's `tol`, `maxit`), the core building the solver with axes derived from the same `bcs` |
| `:sgs` | `:tke` (`sgs_tke`, `PrognosticSGS.TKEClosure`, the Deardorff/Moeng prognostic SGS TKE, #182; Laplacian momentum form, see below), `:smagorinsky` (`sgs_smagorinsky`, `StaticSmagorinsky(cfg.C_s)`, the coefficients of the cores' built-in closure bit for bit; the default before #182), `:dynamic` (`sgs_dynamic`: `LagrangianDynamicSmagorinsky` by default, `DynamicSmagorinsky` for `averaging = :plane` / `:local`; with the buildings' mask), `:mlsgs` (`sgs_mlsgs`, `ML.MLSGS`), `:amd` (`sgs_amd`, `AMDClosure`, zero in the building cells), `:vreman` (`sgs_vreman`, `VremanClosure(cfg.C_s)`), `:sigma` (`sgs_sigma`, `SigmaModel`, #184), `:reynolds_stress` (`sgs_reynolds_stress`, `PrognosticSGS.ReynoldsStressClosure`, the differential Reynolds-stress closure, #188; not with `timestepper = :imex`); all but `:tke` and `:reynolds_stress` with option `stress = :symmetric` (`:laplacian`); `:smagorinsky`, `:dynamic` and `:mlsgs` are each wrapped in `StabilityCorrected` (P55-e, #187; `cfg.sgs_stability = :brown94`; `:none` gives the bare closure, for `:smagorinsky` the coefficients of the cores' built-in closure bit for bit) and also take `scalar_tensor = :diagonal` (`:isentropic`, with `diapycnal`, `slope_max`); `:tke` and `:amd` carry their own stability treatment (TKE length scale, AMD buoyancy term) and are not wrapped, nor are `:vreman`, `:sigma` and `:reynolds_stress` (its stresses carry the buoyancy production directly) | `step_options`: `sgs`, an `AbstractSGSClosure` (the SGS closure contract, #167), and `sgs_stress` (the stress form, #175); every core and stepper takes both |
| `:momentum_advection` | `:centered` (`momentum_centered`), `:weno_z` (`momentum_weno_z`, option `order = 5`/7) | `step_options`: `momentum_advection` |
| `:scalar_advection` | `:advection_schemes` (`scalar_advection`, options `scheme = :weno_z5`, `sgs_diffusion = true`) | the `:transport` stage: advection, then SGS diffusion with `κt` (#177; with a stability-corrected `:sgs` closure κh on the x, y faces and its κv on the z faces, as for θ) |

**Lateral boundaries.** The case is periodic in `x` and `y` by default
(`cfg.lateral = :periodic`, bit for bit the case before #262). Open
(Dirichlet-from-boundary-data) faces exist since N-open-b (#255:
`OpenBoundary`, `open_boundary_conditions`, the `AbstractBoundaryData`
contract and the global mass-flux correction, called from every core
through `update_open_boundaries!`), with the radiation outflow of
N-open-c (#260) and the one boundary-flux function
`boundary_scalar_flux`. `cfg.lateral = :open` (N-open-d, #262) opens the
four sides and the top of the case: see
[Open boundaries](berlin_ledgers.md#Open-boundaries). Since N-berlin-nest (#265)
the default case is nested: the same open faces are fed by the parent run
([Nesting](berlin_nesting.md#Nesting-(#265))), and `lateral` (`:periodic`, `:open`) chooses
the faces only with the nest switched off.

`sim_options` and `step_options` are contract hooks: a coupling returns a
`NamedTuple` of keywords for the core's constructor or for every
dynamics step (`step!`, or `step_imex!` under the IMEX steppers, which
takes the same `momentum_advection` and `sgs`); two
couplings may not set the same keyword.

**SGS providers** are closure objects of the SGS closure contract
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
`[urban] sgs_stability` (`:brown94` default, `:lilly`, `:long_tail`,
`:short_tail`, `:most`; `:none` = the bare closure, the pre-P55-e run),
the anisotropic length scales Δ_h = √(dx dy), Δ_v = dz (ℓ_h = 1.842,
ℓ_v = 0.2947 at dx/dz = 2.5), Ri_g from the θ_v of the `qv` tracer and
the core's own reference θ (θ₀, or θ̄(z) on the sound-proof cores, so it
runs with `core = :anelastic`). `:dynamic` gets no f_m on the momentum
(only Pr_t(Ri)). Each provider's `smoke_check` asserts κ_v, ν_v ≥ 0;
κ_v ≤ κ_h and ν_v ≤ ν_h ℓ_v/ℓ_h where Ri ≥ 0; ν_v ≥ ν_h ℓ_v/ℓ_h where
Ri ≤ 0 (the convective enhancement of Brown et al.; not for `:dynamic`);
κ_v = 0 where Ri ≥ Ri_c; its diagnostics give the supercritical fraction
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
(*one shared κ*, in `test/berlin_numerics.jl` until #320 removed the
Berlin suite) pinned it: under `:amd` the
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
  s⁻¹ (measured by the former `test/berlin_numerics.jl`; the same
  agreement on a synthetic 16³ box is `test/multigrid.jl`). They are here for the stretched-grid
  and cut-cell solver work (#151, #152), not for speed.
* **SGS.** `:tke` (default since #182) is described in
  [Prognostic SGS TKE (#182)](berlin_turbulence.md#Prognostic-SGS-TKE-(#182)) below.
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
  [the SGS theory page](../theory/sgs.md). the diagnostics give the mean C_s
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
  `:amd` and `:vreman` (`components/urban/sgs_amd_vreman.jl`, #176) are the
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
  0.013 m²/s. the diagnostics give the canopy and free-stream mean ν_t against
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
  `:sigma` (`components/urban/sgs_sigma.jl`, #184) is the σ-model of
  Nicoud et al. (2011), option `C_σ = 1.35`. Its ``ν_t`` vanishes in
  laminar shear and 2-D flow; the diagnostics give the canopy and free-stream
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
  `:weno_z7` needs `[grid] halo = [4, 4, 4]`, `:teno8` halo 5; the
  build errors otherwise. The coupling refreshes the velocity halos
  first (see [the stale-halo defect](berlin_case.md#Known-defects-kept-by-the-refactor)).

Measured in the `:smoke` window over 1 h (1800 steps, 12:00–13:00, the
free stream is levels 7–12, 37.5–75 m, above the 31 m roofs and below
the sponge), the dynamic mean C_s is 0.077 in the free stream and 0.076
in the canopy with ratio 3 (0.128 and 0.115 with ratio 2). That is below
the inertial-range 0.17: the window's flow is a 1 m s⁻¹ geostrophic wind
over 250 m with a capping inversion at 60 m, far from developed
isotropic turbulence, and the dynamic coefficient is lower in stratified
and transitional flow (it reaches 0.13–0.15 in the decaying-turbulence
run of `validation/sgs_numerics.jl`).

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

The table is from before #295 (KA-c), which removed the runtime integer
power from the WENO-Z / TENO weights (Base's non-inlined `pow_body`, one
call per candidate weight) and the plane copies from the halo fills.
Remeasured per call on the same grid (1 thread, minimum of 2 × 15 calls,
1-min load ≈ 10 on 24 cores; `advect_positive!` with its workspace):

| operator | option | before (ms/call) | after (ms/call) |
| :-- | :-- | --: | --: |
| scalar advection, one positive tracer | WENO-Z5 (default) | 7.16 | 1.65 |
| | TENO5 | 9.88 | 2.02 |
| | WENO-Z7 (halo 4) | 12.06 | 3.47 |
| momentum advection | WENO-Z5 | 34.96 | 6.62 |
| pressure solve | multigrid | 60.9 | 55.3 |
| | MG-CG | 41.9 | 39.0 |

The numerics are unchanged: p = 2 is evaluated as `x*x`, which is
Base's `x^2` bit for bit; TENO's q = 6 as `x³·x³`, which differs from
Base's compensated `x^6` by at most 4 ulp (3.02 measured) and changed no
value of the Berlin smoke matrix nor of 10⁵ random TENO5 stencils
(`test/advection.jl`).

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
`components/urban/core_select.jl` and `components/urban/timestepper.jl`:
`[core] model` and `[timestep] scheme` (ONE-m, #476; on an urban site
these sections offer the providers below, on a domain of the model's
core `boussinesq` / `anelastic` and `rk3` / `euler`):

| slot | provider | src | what it is |
| :-- | :-- | :-- | :-- |
| `:core` | `:boussinesq` (default) | `Simulations.BoussinesqSimulation` | ∇·u = 0, buoyancy g(θ − θ₀)/θ₀ about the constant `cfg.θ₀` |
| `:core` | `:anelastic` | `Anelastic.AnelasticSimulation` | ∇·(ρ̄u) = 0, buoyancy g(θ − θ̄)/θ̄, ρ̄-weighted advection and projection (Lipps & Hemler 1982) |
| `:core` | `:pseudo_incompressible` | `Anelastic.PseudoIncompressibleSimulation` | ∇·(ρ̄θ̄u) = 0 (Durran 1989) |
| `:timestepper` | `:fixed` | `step!` | one forward-Euler projection step of `cfg.dt`; the stepper of the sound-proof cores |
| `:timestepper` | `:adaptive` (default) | `TimeIntegration.adaptive_dt`, `step_adaptive!` | CFL / diffusion / buoyancy limited step, growth ≤ 10 %, rollback and halving on failure; option `imex = true` steps with ARK2 and only the explicit limits bind |
| `:timestepper` | `:imex` | `TimeIntegration.step_imex!(…, ARK2())` | implicit diffusion, explicit advection and hooks, a projection per stage, at `cfg.dt` |

```sh
# a sound-proof core brings its solver (and has no checkpoints yet)
julia --project -m Aeolus run examples/urban/berlin_summer.toml --set time.nsteps=1800 \
    --set checkpoint.every_seconds=inf \
    --set core.model=\"anelastic\" --set pressure.solver=\"density_weighted_direct\" \
    --set timestep.scheme=\"fixed\"
# 1 h of simulated time with the adaptive step's limits set
julia --project -m Aeolus run examples/urban/berlin_summer.toml --set time.nsteps=1800 \
    --set timestep.c_advective=0.5 --set timestep.dt_max=10.0
```

**Stretched vertical grid (#151): a stated exception.** The
`BoussinesqSimulation` core runs on a `StretchedGrids.StretchedGrid`. See
[Stretched vertical grid](@ref) for the operators, the direct
stretched-z Poisson solve and the measured accuracy and cost. The Berlin
child keeps its uniform `RegularGrid`, and there is no `:stretched` case
option. StretchedGrids is instead coupled by default through the nested
parent in N-berlin-nest (#265). The parent's vertical layout is
`surface_refined_faces(dz_surface = 25, z_uniform = 500, z_top = 3000,
ratio = 1.08, dz_max = 100)`: 53 levels, 20 of them uniform. One step of
192 × 192 × 53 costs 0.225 s on one thread, against 0.526 s for
192 × 192 × 120 at a uniform 25 m.

**The sound-proof cores** take the case's reference state as their
base state: `case.ref.state`, built by `build_reference` (θ̄(z) =
`initial_θ(cfg)` integrated hydrostatically from `case_surface_pressure(cfg)` with
the MoistThermodynamics constants, see [Reference
state](berlin_case.md#Reference-state-and-moist-buoyancy)), so there is one ρ̄(z) per
case — the one the couplings convert fluxes with and the ledger
integrates water with — the initial state has no buoyancy, and the flow
starts as in the Boussinesq core. The dynamics hooks (IBM, canopy drag,
geostrophic forcing, moist buoyancy, surface drag) touch only
`sim.u/v/w/tu/tv/θ` and run unchanged. The projection solves
∇·(W∇φ) = ∇·(W u*) with the core's own solver: directly (FFT/DCT +
tridiagonal, the `:pressure_solver` provider `:density_weighted_direct`,
#309) or by Jacobi-preconditioned CG (`:density_weighted_cg`, options
`tol = 1e-10`, `maxit = 500`). The CG is what made the sound-proof cores
expensive in the #133 table below; the direct solve removes that cost
(#309 numbers after the table).

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
| `core = :anelastic`, `:pseudo_incompressible` | `pressure_solver = :cosine_fft`, `:multigrid`, `:mg_cg` | those solve the constant-coefficient ∇²φ = ∇·u*, not the weighted projection; select `:density_weighted_direct` or `:density_weighted_cg` |
| | `momentum_advection = :weno_z` | an unweighted operator; the core advects momentum in its W-weighted centred flux form (`:centered` passes no keyword to it) |
| `pressure_solver = :density_weighted_direct`, `:density_weighted_cg` | `core = :boussinesq` | they solve only the weighted projection of the sound-proof cores |

Behind the declarations, `build_core` of a sound-proof core rejects any
`sim_options` keyword other than `pressure_solver`, `tol` and `maxit` (and the nest faces). Every `:sgs` provider
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

**The direct projection (#309).** Measured on the branch of #309
(flagship 64 × 64 × 64, nest off, `core = :anelastic`, fixed step, 30
steps from 06:00, one thread, second run of each so compilation is
excluded): the whole case step costs 200 ms with
`:density_weighted_direct` against 391 ms with `:density_weighted_cg`,
and the anelastic constraint max|(1/ρ̄)∇·(ρ̄u)| after the run is
2.9e-17 s⁻¹ against 6.4e-13 s⁻¹ (the plain ∇·u, 9.88e-5 s⁻¹, is the
same in both: the flow is the same).

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
  adaptive `dt` (see [Cadences](berlin_case.md#The-coupling-contract)).
* `moist_buoyancy` adds `g (θ_ρ − θ)′/θ₀` with the constant `θ₀` on every
  core; on a sound-proof core the dry buoyancy is about θ̄(z), a
  difference of θ̄/θ₀ − 1 (< 1 % over the flagship depth).
* Resolved since the PR was opened: the plain-velocity tracer transport
  under a sound-proof core (now W-weighted, above), the stale-halo leak
  that the adaptive stepper's CFL diagnostics used to hide (#132 refreshes
  the halos in the transport itself), and the ledger density (ρ̄ since
  #131).
