# The Berlin case: nesting

## Nesting (#265; NEST-p3, #483)

The default case is the child of a **one-way nest** inside an Aeolus
parent (N-berlin-nest, #265; the Berlin half of #153). Since NEST-p3
(#483) the parent is a `[[domain]]` of the model's config, a domain of
the model's core, and the site is the next `[[domain]]` block, nested in
it at its `origin` and `refinement`; one engine steps both (the model's
`Components.ModelNest`, NEST-p2 #482). The coupling of the site is
`components/urban/nesting.jl` (`:nesting`, an addon), the src machinery
`Aeolus.Nesting` (N-nest-a…d) and the open faces of `BoundaryConditions`
(N-open-a…d). Two-way nesting is an option (`feedback = "two_way"` in the
site's block, #267), **not validated**: see
[Two-way option](#Two-way-option-(#267)) below.

**Parent.** The parent's block configures its whole physics ([The Berlin
parent](berlin_parent.md), #259): `berlin_summer.toml`'s is 6 km × 6 km ×
3 km on 192 × 192 × 53 cells (Δx = 31.25 m, Δz = 25 m to 500 m, then ×
1.08 per level up to 100 m: `[grid] stretching = "surface_refined"`), the
site at refinement (2, 2, 4); `berlin_smoke.toml`'s 1 km × 1 km × 500 m
on 16 × 16 × 10 cells of 62.5 m × 50 m at refinement (4, 4, 8). Its
components: the tiled urban surface of its static driver (`[surface]
model = "tiles"`; the site's own classes under its footprint, AeolusPrep.jl),
the column radiation of the plane-mean column (`[radiation] scheme =
"mean_column"`), the geostrophic pressure gradient, θ/qv nudging and θ
damping from the sounding driver (`[large_scale_forcing]`), the Coriolis
force and velocity sponge, the TKE closure (`[sgs] model = "tke"`), the
fixed-gradient θ top, SSP-RK3 at the root's adaptive step (`[time] cfl`,
`dt_max`). The parent's corner is at minus the site's origin, so the
site (64 × 64 × 64, 1 km × 1 km × 400 m) stands at its own corner (0, 0):
parent cells 81–112 in x and y, layers 1–16.

**Pluggable parent.** A site alone (no parent `[[domain]]`) takes its
faces from boundary data instead (`[nesting] parent_driver`): `"profile"`,
the analytic `case_boundary_data`, `"offline"`, the parent-made dynamic
driver of an online run (`[urban] child_boundaries`), or from Julia any
`AbstractBoundaryData` — the interface a mesoscale reader (#58) or
reanalysis (#57) implements.

**Faces.** The coupling's `sim_options` give the core its boundary
conditions (`nest_case_bcs`: a ground wall, the four sides and the top
`OpenBoundary` nest faces with the outflow `cfg.outflow`, Dirichlet by
default): `case_boundary_conditions` defines none for a nested case, so
the periodic box and the θ `:initial_gradient` top are not built and
nothing runs alongside. The default `:pressure_fft` provider reads the
all-Neumann axes from them (`case_poisson_axes`). The `NestBoundaryData`
of the link fills the faces at every stage time of every core
(`update_open_boundaries!`) with the parent's field set prolongated at the
two ends of the parent step and blended linearly in time, and the single
mass-flux correction of N-open-b. Building faces inside solids keep
`u·n = 0` (the site's mask). The core's velocity sponge and the θ
relaxation over the same layer (`cfg.θ_relaxation`) relax toward the
parent's mean profile over the site (`_covered_profile!`, refreshed at
every window).

**Field set.** θ, and every advected tracer, ql and qr. qv and the TKE
`e` (the parent's `[sgs] model = "tke"`) take the parent's values; the
tracers the parent does not carry (ql, qr, the chemistry species, the
pollutant if advected) get the `:inflow` halo — zero where the face's
wind enters, their own value where it leaves (`add_open_scalars!`, after
the couplings registered them). This is the site's rule, not the model's
PALM rule (a nested model domain carries exactly its parent's fields): the
site's extra tracers are not its dynamic driver's either; the nest checks
leave an urban child's fields to its `:nesting` component. The tracers'
halos are the core's open set (`tracer_bcs`). The site's initial wind is
the parent's, prolongated (a wind that differs from the faces' makes
``∇·u ≠ 0`` in the boundary cells, and the first flux-form θ step then
moves θ by ``-θ∇·uΔt``).

**Time.** One window of the nest engine is one step of the parent; the
site's steps are its own (the `:adaptive` stepper, retrying its own
pieces), landing on the window's end:

| where | what |
| :-- | :-- |
| the engine | the parent takes its adaptive step (the growth from the site's `[time] dt` at the first; with a fixed site step, a whole number of the site's steps, `Components.window_dt`), the window opens (prolongation at both ends, the faces point at it), the site steps to its end (`step!` of the site's domain, its whole pipeline), the window closes (one-way: the clock and the statistics; two-way: also the feedback onto the parent, see below); the last window lands on the run's end |
| `step_dt` → `limit_dt` | the site's step: `subcycle_dt(R, dt)`, `R` the time left in the window (`Nesting.nest_window_end`) |
| `:nest_sync` (after `:clock`) | the site's first step in a new window: the relaxation targets are refreshed, the boundary recorder writes, the window hooks run (the parent is at rest) |
| dynamics (`advance_dynamics!`) | every core reads the blended parent at its stage times |
| `:nest_feedback` (after `:column_tendency`) | two-way: the step's bookings are committed |

The site joins the nest in its `init!`, before the other couplings
(`init_priority = -1`): they initialise from the faces' data and the
site's wind. The nest does not save and retry its windows (the site
retries its own pieces; the structural snapshot of the parent alone cost
the summer nest 3.5–4.4 s per window of 1.5 s): a failed step is the
run's blow-up.

**No parent ahead.** NestedRun's prefetch of the parent's next window
while the child stepped (SC-c, #403, one-way under several threads) went
with NEST-p3: the engine prefetches a one-way parent below the root
only (the root's next step is the model's next window). The flagship
measured 1.17–1.36× from it at 4–24 threads (the history is on the
[nested time-coupling page](../theory/nesting_time.md)).

**Ledgers.** The heat and water the faces carry are N-open-d's terms:
`(:dynamics, :boundary_heat)`, `(:dynamics, :mass_correction)` and
`(:dynamics, :relaxation)` in the air-heat ledger, `:boundary` in the
water ledger (booked with the face fluxes the transport used, the
positivity-limited ones included; the site books what its faces carry
itself, no nest-exchange term). The parent's energy and water budgets
(its domain's, when it measures them) close separately, a two-way
feedback booked as its nest exchange. The nesting smoke checks assert
both parent closures (≤ 1e-10 of the largest term) when it measures
them, the child's net boundary volume
flux after the correction (≤ 1e-12 ``UA``) and ``|∇·u|``, one geostrophic
wind in both domains (the child's `:geostrophic` coupling takes the
parent's ``U_g(z,t)``, ``V_g(z,t)``), the tile mosaic and the field set;
every smoke run closes the child's surface, water and air-heat ledgers.
The individual side and lid terms carry the through-flow of ≈ 293 K air
and largely cancel; the heat the box exchanges is their sum.

**Measured** (with the ParentCase parent before NEST-p3; `examples/urban/berlin_nest_check.jl`, commit `e4b34e7`; since #438
these runs are `aeolus run examples/urban/berlin_summer.toml` with
`--set` overrides — the start hour, `nesting.enabled=false`, the parent's
refinement — and the numbers below are in each run's `output/results.toml`:
`[timing] wall_s_per_simulated_h`, `[open_boundary_heat]`, `[air_heat]`,
`theta_mean_below_100m_start`/`_end`;
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
`test/era2_nest_external.jl` (rule of #321): a window driven from outside
is `step_nested!`'s bit for bit; uniform state, Galilean invariance, mass, the warm-bubble
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
validation #270. `AdaptiveMesh` is not coupled to the Berlin run (AMR
blocks, not the nest).

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

`feedback = "two_way"` (the site's block) feeds the site back onto the parent
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
   projected onto its new boundary data (when the window opens).

| option | default | |
| :-- | :-- | :-- |
| `feedback` (the site's `[[domain]]` block) | `"one_way"` | `"two_way"` turns the feedback on |
| `buffer` | 2 | parent cells next to the open faces left out |
| `z_anterp` | `:canopy` | the canopy restriction (m) |
| `τ` | `nothing` | relaxation time of the anterpolation (s) |
| `reflux` | `true` | refluxing of θ and qv |

**Bookings.** The parent is a domain of the model's core: the link's
register hook books its θ and qv in every stage of its core's step with
the stage weights (SSP-RK3: 1/6, 1/6, 2/3). The site moves its θ in its
core's step pieces and qv in its `:scalar_advection` slot, and books both
itself with [`book_nest_flux!`](@ref): the link's
`NestScalarFlux(child_applied_by = :transport)` (NEST-p3, #483; before,
the parent's bookings were explicit too):

| scalar | domain | booked by | flux |
| :-- | :-- | :-- | :-- |
| θ | parent | the link's register hook, every stage of the core × the stage weight | centred advection + molecular and SGS diffusion |
| qv | parent | the same | centred advection + the core's tracer diffusion |
| θ | child | the coupling's `:body_force` entry (order 5; a no-op one-way), every stage of `step!` | centred advection + the closure's θ diffusion |
| qv | child | the `:scalar_advection` slot, before its update (#327 M3) | the slot's scheme (WENO-Z5) with the velocity it uses, then the SGS diffusion with the tracer's κ |

Each site step is bracketed by the link's `_begin!` (in `:nest_sync`)
and `_commit!` (in `:nest_feedback`), so the slot's and the
dynamics' bookings of the step land in its registers. The other tracers
are anterpolated but not refluxed; a tracer the parent does not carry
gets a parent field of its own in a two-way run (the anterpolation
writes into it; it is not transported in the parent).

**Ledgers.** The feedback is booked in the parent's budgets (when it
measures them) as its nest exchange, `(:nest_feedback, :nest_exchange)`
(`Components.NEST_EXCHANGE`), so they still close (before NEST-p3 the
ParentCase ledgers named it `(:nest_feedback, :feedback_heat)` and
`:feedback_water`). The
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
checks that the feedback ran, that the parent's budgets close (≤ 1e-10 of
their largest term) when it measures them, the interface
mass-flux mismatch after the flux correction (≤ 1e-12 of ``UA``) and the
re-projected parent's ``\max|∇·u|`` (≤ 1e-8 s⁻¹).

*Measured* (the ParentCase parent before NEST-p3; the smoke variant, 12 child steps = 24 s, 8 parent windows,
7 feedbacks): parent heat-ledger closure 0.0 and water 1.6e-16 of the
largest term; child air-heat closure 3.4e-13 W m⁻², water residual
1.9e-15 kg m⁻²; the feedback put +96.7 J m⁻² of heat and
−4.3e-5 kg m⁻² of water into the parent; the parent's ⟨θ⟩ over the
footprint at 0–100 m (two levels) was 293.0066 K two-way against
292.9905 K one-way (the domain mean 293.0231 against 293.0222 K). This
says the coupling acts, in the expected direction (the warmer urban child
warms the parent above it); it says nothing about stability.

**Refused.** A site alone takes no feedback (boundary data cannot take
one); refluxing with an IMEX
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


## Synthetic inflow turbulence (#268)

The child's lateral faces take the parent's field at 2–4× the child's
spacing (flagship: Δx_p = 31.25 m against 15.6 m). The parent's eddies
arrive; the eddies between the two resolutions do not, and an inflow fed
only with the parent's interpolated field enters the child too smooth
(N-inflow-d, #266: with no generator at all, a laminar inflow stayed
laminar for 15 boundary-layer depths). The addon `:inflow_turbulence`
(`components/urban/inflow_turbulence.jl`, on by default) adds the missing part
on the **inflow** parts of the four lateral faces with
[`InflowTurbulence`](@ref Aeolus.Transition.InflowTurbulence) (N-inflow-c,
#264; theory on the [transition page](../theory/transition.md)):
velocity only, ramped on per level by the face-mean inflow
(``\min(1, -U_n/u_{\min})``), its per-level mean removed (no mass added),
linear in time over a parent window like the parent data, and inserted
inside `update_open_boundaries!` before the mass-flux correction, so no
step core is touched.

| option | default | |
| :-- | :-- | :-- |
| `generator` | `:digital_filter` | `DigitalFilterGenerator` (Xie & Castro 2008); `:fsd`: the random-mode `FreeStreamInflow` (`free_stream_generator`, swept at the parent's mean wind over the child) |
| `statistics` | `:parent_resolved` | the parent's resolved stress in the scale gap of the refinement plus its SGS stress (`ParentResolvedStatistics`, on the parent's `StretchedGrid`); `:parametrized`: `ParametrizedStatistics` with the child's u* and Obukhov length and the parent's ``z_i`` (the path of a mesoscale parent, #58) |
| `u_min` | 0.5 m s⁻¹ | the inflow ramp |
| `seed` | 1 | generator seeds (face q: seed + q − 1) |
| `statistics_interval` | 60 s | parent time between two folds of the parent statistics |
| `length_interval` | 3600 s | rebuild of the digital-filter kernels from the current lengths |
| `length_scale` | `NaN` | `:parametrized` only: the single length (NaN: 8 child spacings) |

**The parent is read only at rest.** The parent statistics are folded
from a hook of the nesting coupling (`window_hooks`, called at the
site's first step in a window, after the parent's step), with [`fold_inflow_statistics!`](@ref
Aeolus.Transition.fold_inflow_statistics!); the child's boundary refresh
never reads the parent.

**Ledger and checks.** `budget!` books the perturbation kinetic energy
the mean inflow carries through the perturbed faces,
``ρ⟨\tfrac12|u'|^2\max(-U_n,0)⟩`` (W m⁻² of perturbed face,
`injected_ke_flux`), as the ledger term `:inflow_ke` (J m⁻²); the
diagnostics `inflow_ke_flux`, `inflow_ke` and `inflow_added_flux_max`
report it. The added mass flux is zero by construction: `smoke_check`
asserts it ≤ 1e-12 of the face's mean-flow flux on every face at every
step, the booked term, a non-zero perturbation on an inflow face and a
newest knot that is exactly zero on every outflow level (the applied
perturbation blends two knots in time, so a level that has just turned
to outflow still carries the older knot's share). Its smoke variants are the FSD
generator, the parametrized statistics (with both generators) and the
parametrized statistics on the analytic `parent_driver = :profile` data;
the nest's "two-way" variant runs it with the feedback.

**No running parent.** Faces fed by a driver instead of the running
parent (`parent_driver = :profile` or `:offline`, or the open box without
the nest) have no resolved statistics: the generator is installed there
only with `statistics = :parametrized`. The offline child replays the
parent's faces, not the online child's perturbations, which are drawn
from the running parent; `test/input_dynamic.jl`'s bit-for-bit
online/offline round trip therefore runs with the addon off.

**Two-way.** With `feedback = "two_way"` the feedback skips an inflow band of
``⌈2\max L_s/Δx_p⌉`` parent cells (`exclude_inflow_feedback!`). The
interface flux correction gives the parent the child's flux minus the
generator's normal perturbation (`synthetic_normal_velocity`, #418).
Before #418 it copied the perturbation into the parent's interface faces.
The next window's boundary data returned it to the child, and the
generator added the next one on top. On the 2 h sheared-CBL nest of
`validation/era6_inflow_nest_longrun.jl` (Hellsten et al. 2021's two-way CBL
with a 3 m s⁻¹ wind, r = 3), that made the child's kinetic energy ×12
and the parent's ×2.5, with an overlap θ drift of +0.25 K against one-way
with the generator. Two-way with the generator now meets the #268
criteria (1.5×, 0.2 K) in two realisations: child ×0.94 and ×1.09,
parent ×1.02, overlap θ −0.003 K and −0.009 K. Two-way without the
generator gives ×1.04, ×0.995 and −0.002 K. Analysis and the choice of
fix: [Two-way with the generator](../theory/transition.md#Two-way-with-the-generator-(N-inflow-e)).
The Berlin two-way nest itself is unvalidated; its campaign is #331.

**Analysis buffer.** Behind an inflow face the turbulence is still
developing, so the diagnostics compared with observations leave out the
columns within `cfg.analysis_buffer` of the lateral open faces
(`analysis_columns`, `site/open_boundary.jl`); today these are `ustar_roof` and
`ustar_ground` of the surface exchange (#270 adds canopy T and the heat
island). The default, 140 m, is an **estimate**: the 1.2-turnover rule
N-inflow-d measured, ``L = 1.2\,Uk/ε``, with neutral surface-layer
values at the mean roof height of the flagship scene
(`ANALYSIS_BUFFER_DEFAULT`). The Berlin fetch itself — on the
child with buildings, inflow from the parent at 12:00 and 03:00,
generator on and off, to 90 % of the canopy-top TKE plateau and at
100 m — needs hours of model time and is left to the 24 h validation
(#270), which replaces the default with it; long Berlin runs were
paused when this was written.

## Checkpoint / restart (#269)

The nest is checkpointed whole, at the end of a window (where the run
loop checkpoints): the model's nest (`Components.ModelNest`) records its
hierarchy with the structural walk, which reaches every domain — the
parent's core and components (its surface tiles, radiation, forcing,
closure), the site (`checkpoint_state(case)`: its core, every coupling,
the surfaces, the ledgers) — and every link: both parent time levels,
the step sizes the adaptive controllers continue from, the flux
registers of a two-way run, the statistics, the site's boundary data and
the inflow generator (planes, RNG, folded parent statistics).

`[checkpoint]` is on in `berlin_summer.toml` (every 30 min of wall clock,
the newest 2 kept); `aeolus run <folder> --resume` continues the run.
Module-level proof: `test/nest_restart.jl` (bitwise, one-way and
two-way, a checkpoint at a window's end); the resume of the nested
`berlin_smoke.toml` (a checkpoint every 3 windows, interrupted after 5,
the site, its tracers and the parent domain bitwise) is in
`test/config_roundtrip.jl`. The 24 h flagship
restart (22:00 resumed to 23:00) waits for the long Berlin runs, paused
when this was written. Design and limits: [Checkpoint /
restart](checkpoint_restart.md#The-Berlin-case).
