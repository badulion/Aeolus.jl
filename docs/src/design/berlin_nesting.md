# The Berlin case: nesting

## Nesting (#265)

The default case is the child of a **one-way nest** inside an Aeolus
parent run (N-berlin-nest, #265; the Berlin half of #153). The parent is
the deep diurnal run of [The Berlin parent](berlin_parent.md) (#259); the
coupling is `couplings/nesting.jl` (`:nesting`, an addon, on by default),
the src machinery `Aeolus.Nesting` (N-nest-a…d) and the open faces of
`BoundaryConditions` (N-open-a…d).

**Parent.** `options[:nesting].parent` holds `ParentConfig` keywords:
empty for `:summer` (6 km × 6 km × 3 km, refinement (2, 2, 4): Δx =
31.25 m, Δz = 25 m to 500 m, 192 × 192 × 53), the `:smoke` scenario
sets a 1 km × 1 km × 500 m parent at (4, 4, 8) (16 × 16 × 10 cells of
62.5 m × 50 m, the land use a crop of the 6 km table). The parent keeps
all of its own physics — the tiled surface (`SurfaceTiles`), column
radiation, the geostrophic pressure gradient, θ/qv nudging, θ damping,
the fixed-gradient θ top, the TKE closure, SSP-RK3 under
`step_adaptive!` — and its two ledgers; `step_parent!(p, h)` is the
`NestedRun`'s `advance_parent` closure (`domain_advance`). The parent is
centred on the child and placed in the child mesh's coordinates, so the
child (64 × 64 × 64, 1 km × 1 km × 400 m, deepened for the nest) nests at
its own origin: parent cells 81–112 in x and y, layers 1–16.

**Pluggable parent.** `options[:nesting].parent_driver` is `:aeolus` (the
parent run) or any `AbstractBoundaryData` — the interface a mesoscale
reader (#58) or reanalysis (#57) implements — which then feeds the same
nest faces with no parent run (`:profile`, the analytic
`case_boundary_data`, is the smoke variant of that path).

**Faces.** The coupling's `sim_options` give the core its boundary
conditions (`nest_case_bcs`: a ground wall, the four sides and the top
`OpenBoundary` nest faces with the outflow `cfg.outflow`, Dirichlet by
default): `case_boundary_conditions` defines none for a nested case, so
the periodic box and the θ `:initial_gradient` top are not built and
nothing runs alongside. The default `:pressure_fft` provider reads the
all-Neumann axes from them (`case_poisson_axes`). The `NestBoundaryData`
of the run fills the faces at every stage time of every core
(`update_open_boundaries!`) with the parent's field set prolongated at the
two ends of the parent step and blended linearly in time, and the single
mass-flux correction of N-open-b. Building faces inside solids keep
`u·n = 0` (the child's mask). The core's velocity sponge and the θ
relaxation over the same layer (`cfg.θ_relaxation`) relax toward the
parent's mean profile over the child (`_covered_profile!`, refreshed at
every window).

**Field set.** θ, and every advected tracer, ql and qr. qv and the TKE
`e` (`:sgs => :tke`) take the parent's values; the tracers the parent does
not carry (ql, qr, the chemistry species, the pollutant if advected) get
the `:inflow` halo — zero where the face's wind enters, their own value
where it leaves (`add_open_scalars!`, after the couplings registered
them). The tracers' halos are the core's open set (`tracer_bcs`). The
child's initial wind is the parent's, prolongated (a wind that differs
from the faces' makes ``∇·u ≠ 0`` in the boundary cells, and the first
flux-form θ step then moves θ by ``-θ∇·uΔt``).

**Time.** The child steps are the case's own (the `:adaptive` stepper),
landing on the parent's windows:

| where | what |
| :-- | :-- |
| `step_dt` → `limit_dt` | the child's step: `subcycle_dt(R, dt)`, `R` the time left in the open window, or the parent's next step `parent_dt` when none is open |
| `:nest_parent` (after `:clock`) | no window open: `begin_parent_step!` — the parent takes its step (a fixed child step: a whole number of child steps), prolongation at both ends, the faces point at the window; the relaxation targets are refreshed |
| dynamics (`advance_dynamics!`) | every core reads the blended parent at its stage times |
| `:nest_feedback` (after `:column_tendency`) | the child reached the window's end: `end_parent_step!` (one-way: the clock and the statistics; the two-way feedback is N-berlin-twoway, #267) |

The nest is built in `init!` before the other couplings
(`init_priority = -1`): they initialise from the faces' data and the
child's wind.

**Ledgers.** The heat and water the faces carry are N-open-d's terms:
`(:dynamics, :boundary_heat)`, `(:dynamics, :mass_correction)` and
`(:dynamics, :relaxation)` in the air-heat ledger, `:boundary` in the
water ledger (booked with the face fluxes the transport used, the
positivity-limited ones included). The parent's own heat and water
ledgers close separately. The nesting smoke checks assert both parent
closures (≤ 1e-10 of the largest term), the child's net boundary volume
flux after the correction (≤ 1e-12 ``UA``) and ``|∇·u|``, one geostrophic
wind in both domains (the child's `:geostrophic` coupling takes the
parent's ``U_g(z,t)``, ``V_g(z,t)``), the tile mosaic and the field set;
every smoke run closes the child's surface, water and air-heat ledgers.
The individual side and lid terms carry the through-flow of ≈ 293 K air
and largely cancel; the heat the box exchanges is their sum.

**Measured** (`examples/urban/berlin_nest_check.jl`, commit `e4b34e7`,
one thread each on the shared 24-core machine, 29 June 2019 from the
Lindenberg sounding).

*Cost*, 12:00–12:10 LT, the first step (compilation) excluded:

| run | child | s per simulated hour | child steps (mean dt) |
| :-- | :-- | --: | --: |
| periodic, 200 m (the child before the nest; #238 quoted 312 s/h) | 64 × 64 × 32 | 339 | 338 (1.78 s) |
| periodic, 400 m | 64 × 64 × 64 | 671 | 382 (1.57 s) |
| periodic, 400 m, driven by the nest's geostrophic wind | 64 × 64 × 64 | 2343 | 1490 (0.40 s) |
| **nested** (6 km parent, 192 × 192 × 53, 239 windows) | 64 × 64 × 64 | **5189** | 2215 (0.27 s) |

The nested child is driven by the sounding's wind (the parent's
geostrophic wind, ≈ 3.4 m/s at the start) instead of the periodic case's
1 m/s, and its advective step shrinks with it: at the same wind the
periodic box already costs 2343 s/h. The nest itself adds the parent's
steps (270 s of 865 s wall, 31 %) and 1.5× more child steps (they land on
the windows; the inflow carries the parent's structure); a child step
costs the same (0.27 s nested, 0.26 s periodic at that wind).

*Evening*, 18:00–19:00 LT (both runs start from the sounding at 18:00,
no spin-up; the nest with the uniform 62.5 m × 50 m fallback parent of
#259, refinement (4, 4, 8), to keep the run at 47 min wall):

| | periodic (nest off) | nested |
| :-- | --: | --: |
| ⟨θ⟩ 0–100 m, 18:00 → 19:00 | 301.000 → 300.090 K (−0.91 K) | 301.002 → 299.091 K (−1.91 K) |
| heat through the lid | 0.77 W m⁻² (fixed-gradient top) | −5949 W m⁻² |
| heat through the sides | — | +6291 W m⁻² |
| net boundary heat (`boundary_heat` + `mass_correction`) | — | +342 W m⁻² |
| surface sensible heat | −0.06 W m⁻² | −19.3 W m⁻² |

The side and lid terms are the through-flow of ≈ 300 K air (in through the
sides, out through the lid: a mean 1.8 cm s⁻¹ upward at the lid); their
sum, +342 W m⁻², is dominated by the weights of the ledger's heat
``ρ̄c_pΠ̄θ``: ``ρ̄Π̄`` is ≈ 6 % smaller at 400 m than at the ground
(ρ̄ −4.6 %, Π̄ −1.3 %), so air that enters low and leaves high books
≈ 6 % of the ≈ 6000 W m⁻² through-flow (≈ 350 W m⁻²) as a gain even at
uniform θ — the open-box counterpart of the closed box's
`(:dynamics, :redistribution)`. The lowest 100 m cool by 1.0 K more in the
nested run in this first evening hour: the realistic wind exchanges more
heat with the cooling surfaces (−19.3 against −0.06 W m⁻² sensible heat),
and the box no longer keeps its own air. The full evening and night are
the 24 h validation of #270.

**Tests.** The physics of the nest is tested on synthetic boxes in
`test/era2_nest_external.jl` (rule of #321): the split of `step_nested!`
is bit for bit; uniform state, Galilean invariance, mass, the warm-bubble
export and the hooks in both domains (numbers on the [nested time-coupling
page](../theory/nesting_time.md)). The Berlin wiring is checked by the
coupling's smoke checks in every smoke run (47 runs, 188 s with the
nest). Checks derived for the periodic box take the nest off
(`:nesting => false`) and keep their assertions (berlin_air_heat,
berlin_dynamics, berlin_numerics, berlin_open_boundaries,
berlin_surface_exchange); the momentum-conservation and Reynolds-stress
identities run in closed-box smoke variants.

**Stated exceptions.** The two-way option is N-berlin-twoway (#267)
(`mode = :two_way` is an error here). The parent's 24 h validation is
#315, the nested 24 h validation #270. `:AdaptiveMesh` stays in
`BERLIN_PENDING` (AMR blocks, not the nest).
