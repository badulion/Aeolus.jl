# The Berlin case: nesting

## Nesting (#265)

The default case is the child of a **one-way nest** inside an Aeolus
parent run (N-berlin-nest, #265; the Berlin half of #153). The parent is
the deep diurnal run of [The Berlin parent](berlin_parent.md) (#259); the
coupling is `couplings/nesting.jl` (`:nesting`, an addon, on by default),
the src machinery `Aeolus.Nesting` (N-nest-a…d) and the open faces of
`BoundaryConditions` (N-open-a…d). Two-way nesting is an option,
`options[:nesting].mode = :two_way` (#267), **not validated**: see
[Two-way option](#Two-way-option-(#267)) below.

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
| `:nest_feedback` (after `:column_tendency`) | the child reached the window's end: `end_parent_step!` (one-way: the clock and the statistics; two-way: also the feedback onto the parent, see below) |

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

**Stated exceptions.** The two-way option (#267, below) is not
validated. The parent's 24 h validation is #315, the nested 24 h
validation #270. `:AdaptiveMesh` stays in `BERLIN_PENDING` (AMR blocks,
not the nest).

## Two-way option (#267)

!!! warning "Unvalidated"
    Two-way nesting of the Berlin case is **unvalidated**. It runs, and its
    ledgers close in the 12-step smoke test, but nothing longer has been
    run. The long-run stability campaign (multi-hour to 24 h runs against
    one-way, composite drift, buffer and flux-register stability with
    buildings, the secondary circulation of #327 M5, cost) is #331, after
    the KernelAbstractions port. **One-way stays the default** until #331
    validates two-way; that issue then makes two-way the default for
    self-nesting (Aeolus parent → Aeolus child), as in PALM. Driving from
    a mesoscale model or reanalysis (#58) stays one-way.

`options[:nesting].mode = :two_way` feeds the child back onto the parent
at the end of every parent window, with [`TwoWayFeedback`](@ref) (its
physics, and the synthetic tests it is validated by, are on the
[two-way nesting page](../theory/nesting_twoway.md)):

1. **Anterpolation** of the field set (u, v, w, θ, qv, the tracers, the
   TKE `e`) onto the covered parent cells, solid-aware with the child's
   building mask; optionally relaxed (`τ`, default `nothing`: direct
   replacement, as in PALM);
2. not in the **buffer** of `buffer` parent cells next to every open face
   of the child (default 2, PALM's) and not below **`z_anterp`**, the
   canopy restriction (`:canopy`, default: the mean height of the
   child's built columns, `nest_z_anterp`; or a height in m, `-Inf` for
   none);
3. the **interface flux correction**: the parent's velocity on the
   interface planes takes the child's flux;
4. **refluxing** of θ and qv (`reflux = true`, default) from
   Berger–Colella flux registers;
5. the **re-projection** of the parent. At the next window the child is
   projected onto its new boundary data (`begin_parent_step!`).

| option | default | |
| :-- | :-- | :-- |
| `mode` | `:one_way` | `:two_way` turns the feedback on |
| `buffer` | 2 | parent cells next to the open faces left out |
| `z_anterp` | `:canopy` | the canopy restriction (m) |
| `τ` | `nothing` | relaxation time of the anterpolation (s) |
| `reflux` | `true` | refluxing of θ and qv |

**Bookings.** Neither core is stepped by the `NestedRun` (the parent by
its own `step_parent!`, the child by the stage loop), so the run's hooks
cannot book the flux registers. Every refluxed scalar is booked by the
code that transports it, with [`book_nest_flux!`](@ref) and
`NestScalarFlux(applied_by = :transport)`:

| scalar | domain | booked by | flux |
| :-- | :-- | :-- | :-- |
| θ | parent | `step_parent!`'s `body_force!`, every dynamics stage × the stage weight (SSP-RK3: 1/6, 1/6, 2/3) | centred advection + molecular and SGS diffusion (`diffusivity = :theta`) |
| qv | parent | the same | centred advection + the core's tracer diffusion (`diffusivity = :tracer`) |
| θ | child | the coupling's `:body_force` entry (order 5; a no-op one-way), every stage of `step!` | centred advection + the closure's θ diffusion |
| qv | child | the `:scalar_advection` slot, before its update (#327 M3) | the slot's scheme (WENO-Z5) with the velocity it uses, then the SGS diffusion with the tracer's κ |

Each child step is bracketed by `begin_child_step!` (in `:nest_parent`)
and `end_child_step!` (in `:nest_feedback`), so the slot's and the
dynamics' bookings of the step land in its registers. The other tracers
are anterpolated but not refluxed; a tracer the parent does not carry
gets a parent field of its own in a two-way run (the anterpolation
writes into it; it is not transported in the parent).

**Ledgers.** The feedback is a zero-length step of the parent's two
ledgers: its heat is booked as `(:nest_feedback, :feedback_heat)` in the
parent's air-heat ledger and its water as `(:nest_feedback,
:feedback_water)` in the parent's water ledger, so both still close. The
child's ledgers are untouched (the feedback changes only the parent).
The composite (the parent outside the child plus the child) is **not**
conserved exactly in the Berlin case, and no test asserts it: the child's
sponge and θ relaxation toward the parent's mean profile, the parent's
nudging and θ damping and both surfaces are sources, and see the
residuals below. The synthetic conservation tests are in
`test/era2_nest_twoway.jl` and `test/era2_nest_external.jl`.

**Smoke test.** The nest's smoke variant "two-way" (buffer 1 and
`z_anterp = 0`: the smoke child covers only 4 × 4 × 2 parent cells, so
the default buffer of 2 would leave nothing to anterpolate) runs 12
steps with every coupling on. Besides every run's closed child ledgers it
checks that the feedback ran and named a non-zero `feedback_heat`, that
both parent ledgers close (≤ 1e-10 of their largest term), the interface
mass-flux mismatch after the flux correction (≤ 1e-12 of ``UA``) and the
re-projected parent's ``\max|∇·u|`` (≤ 1e-8 s⁻¹).

*Measured* (the smoke variant, 12 child steps = 24 s, 8 parent windows,
7 feedbacks): parent heat-ledger closure 0.0 and water 1.6e-16 of the
largest term; child air-heat closure 3.4e-13 W m⁻², water residual
1.9e-15 kg m⁻²; the feedback put +96.7 J m⁻² of heat and
−4.3e-5 kg m⁻² of water into the parent; the parent's ⟨θ⟩ over the
footprint at 0–100 m (two levels) was 293.0066 K two-way against
292.9905 K one-way (the domain mean 293.0231 against 293.0222 K). This
says the coupling acts, in the expected direction (the warmer urban child
warms the parent above it); it says nothing about stability.

**Refused.** `mode = :two_way` with a `parent_driver` other than
`:aeolus` (boundary data cannot take a feedback); refluxing with an IMEX
stepper (its θ diffusion is implicit, the bookings are explicit; use
`reflux = false`); a sound-proof child core (a Boussinesq parent and a
sound-proof child have no common measure, #327 M1).

**Known limitations of the slim option** (to be examined by #331):

* the parent's surface flux over the child's footprint is not replaced
  by the child's area-mean flux (the original #267 scope item 3); with
  the canopy restriction, the covered parent canopy keeps its own
  surface exchange;
* where the slot's positivity limiter scales an interface flux, the
  booking is the unlimited one;
* a rejected attempt inside a core's own `step_adaptive!` (a retry) keeps
  its bookings;
* the cost of two-way per parent step has not been measured on the
  flagship (the smoke run is dominated by compilation).

