# Nested time coupling: subcycling and temporal interpolation

The [nesting operators](nesting.md) move a field set between a parent and a
child grid. This page covers how the two grids are advanced in time
together: [`NestedRun`](@ref) and [`step_nested!`](@ref) in
`src/nesting/nested_run.jl` (N-nest-c, #257).

## The pieces

A [`NestedRun`](@ref) holds the parent, the child, the
[`NestedGrid`](@ref), three [`NestInterpState`](@ref) buffers (the parent
prolongated at the start and the end of the parent step, and a scratch
blend), one advance closure per domain, the schedule, the feedback mode and
the statistics.

Each domain is advanced by a [`DomainAdvance`](@ref), built with
[`domain_advance`](@ref)`(core, args...; kwargs...)`. It carries the hooks
and keywords that domain's steps need (`body_force!`, `post_predictor!`,
`momentum_advection`, `sgs`, `sgs_stress`, …) and calls the core through
dispatch: `step!` (Boussinesq, or the anelastic / pseudo-incompressible step
of a `MassWeightedSimulation`), `step_rk3!`, `step_imex!` with `ARK2()` or
`Rosenbrock2()`, or a user function. No step logic is copied. The run adds one
hook of its own, chained after the domain's `post_predictor!`, which
refreshes the child's halos (see below).

## Scheduling

**Parent step.** ``Δt_p`` is the parent's [`adaptive_dt`](@ref) (per-cell
advective, diffusive, buoyancy and Coriolis limits, the growth limiter and
the clamps of its `CFLConstraints`). It can also be passed as a fixed `dt`.

**Child substeps (`sync = :subcycle`, the default).** The child lands
exactly on ``t_p + Δt_p``. It starts with

```math
n = \left\lceil \frac{Δt_p}{Δt_c^{\max}} \right\rceil ,
```

where ``Δt_c^{\max}`` is the child's own `adaptive_dt`. Before every piece
the count is re-evaluated against the current child limit,
``n_{\rm rem} \leftarrow \max\!\big(n_{\rm rem}, \lceil R/Δt_c^{\max}\rceil\big)``,
and the piece is ``Δt_c = R/n_{\rm rem}``, where ``R`` is the time left. The
last piece takes exactly ``R``. The ceiling uses a relative tolerance of
``10^{-9}``, so a ratio that is an integer up to round-off is not rounded up:
a child whose limit is exactly ``Δt_p/r`` takes ``r`` substeps, not
``r + 1``. The test shows this for ``r = 2, 3, 4, 5``. At the end the child
clock is set to the parent's. The discrepancy before that is kept in
`stats.max_clock_error`, and it was 0 over 1000 adaptive parent steps.

**Shared step (`sync = :shared_min_dt`).** Both domains take one step of
``\min(Δt_p^{\max}, Δt_c^{\max})``. This is PALM's scheme (Hellsten et
al. 2021, §3.1): every domain uses the smallest time step of all domains.
It is simpler, but the parent then runs at the child's step.

**Rejections.** A step is rejected when it throws, leaves a non-finite
prognostic, a non-finite bound scalar or a speed above `max_velocity`, or
(parent only) exceeds `divergence_tol`. A rejected step is rolled back with
the TimeIntegration `StateSnapshot` contract, plus the bound scalars and the
SGS closure's memory. The parent is stepped and validated **before** any
child substep, so a parent rejection never costs child work. A rejected
child piece rolls back only the child. That piece is redone at half its
size and the remaining time is re-divided. The parent is never advanced
twice.

## Temporal interpolation

The child's boundary data at a time ``t`` inside the parent step is the
linear blend of the parent field set prolongated at the two ends of the
step:

```math
B(t) = (1-τ)\,\mathcal P\,φ_p(t_p) + τ\,\mathcal P\,φ_p(t_p+Δt_p),
\qquad τ = \frac{t-t_p}{Δt_p}.
```

[`boundary_time`](@ref)`(run, t)` evaluates it over the whole field set
(`u, v, w, θ`, `qv`, extra scalars), at the points the run stores (the
halos and boundary face planes only, without a sponge: #329). Linear interpolation of a smooth
``φ(t)`` has the error

```math
\lvert B(t) - \mathcal Pφ(t)\rvert \le \frac{Δt_p^2}{8}\max\lvert \ddot φ\rvert ,
```

which is second order. With ``φ = 300 + A\sin ωt`` and ``A = 1.5``,
``ω = 2π``, the measured maximum error over the child's stage times was
``1.824\times10^{-2}`` at ``Δt_p = 1/20``. The bound is
``1.851\times10^{-2}``. Halving ``Δt_p`` divided the error by 3.957, and
halving it again divided it by 3.989.

### Boundary data at the stage times of every core

A multi-stage core reads its halos once per stage. Imposing the data at the
step start for all stages loses accuracy (Carpenter et al. 1995). The
halos must hold ``B(t_0 + c_i Δt)`` when stage ``i`` computes its
tendencies. The ``c_i`` are given by [`stage_times`](@ref). Since N-nest-d
(#261) the child's nest faces are `OpenBoundary` faces fed by a
[`NestBoundaryData`](@ref), and every core refreshes them itself, once per
projected stage, before the projection (`update_open_boundaries!`, N-open-b
#255), with the time in the last column ([nested boundary
conditions](nesting_boundaries.md)). Before, the run wrote the halos
through a `post_predictor!` hook with the same times.

| core | ``c_i`` | boundary data refreshed at the end of stage ``k`` |
| :-- | :-- | :-- |
| `step!` (Boussinesq or sound-proof) | ``0`` | ``B(t_0 + Δt)`` |
| `step_imex!`, ARK2 (Ascher et al. 1997) | ``0, γ`` | ``B(t_0 + h_k)``, with ``h_k = γΔt, Δt`` the stage increments |
| `step_imex!`, Rosenbrock2 | ``0, a_{21}`` | ``B(t_0 + h_k)``, with ``h_k = a_{21}Δt, Δt`` |
| `step_rk3!`, SSP-RK3 (Shu & Osher 1988) | ``0, 1, \tfrac12`` | ``B(t_0+Δt),\ B(t_0+2Δt),\ B(t_0+\tfrac32Δt)`` |

The SSP-RK3 row needs explaining. `step_rk3!` combines whole arrays, halos
included, and the open-boundary buffers with the same weights:
``u^{(2)} = \tfrac34 u^n + \tfrac14 \tilde u`` and
``u^{n+1} = \tfrac13 u^n + \tfrac23 \tilde u``. Because ``B`` is linear in
time, the values in the last column make the combined halos

```math
B(t_0+Δt),\quad \tfrac34 B(t_0) + \tfrac14 B(t_0+2Δt) = B(t_0+\tfrac12Δt),
\quad \tfrac13 B(t_0) + \tfrac23 B(t_0+\tfrac32Δt) = B(t_0+Δt) .
```

These are exactly the stage times and the step end. A write past
``t_p + Δt_p`` is a linear extrapolation of the same line.

The boundary face of the normal velocity on a west, south or bottom nest
face is the child's interior face index ``H+1``; it belongs to the
`OpenBoundary` and is written with the halos (N-nest-c had to re-impose it
with an `impose_interface` option, now removed). A periodic child direction
has no open face and keeps its own interior face.

**Measured.** The parent was advanced analytically, linear in ``x, y, z``
and ``t``, and the child's halos were checked at every substep and every
stage against the analytic value. The maximum relative error was
``4.4\times10^{-16}`` for `step!` and `step_rk3!`, ``4.6\times10^{-16}`` for
ARK2 and ``4.4\times10^{-16}`` for Rosenbrock2. `boundary_time` reached
``3.8\times10^{-16}``.

## A child driven by its caller (N-berlin-nest, #265)

The Berlin case advances its child through a whole stage loop (surfaces,
radiation, chemistry, transport, then the dynamics), so it cannot hand
the child to the run's own subcycling. `step_nested!(run)` is therefore
three calls, and a caller can take the middle one itself:

1. [`begin_parent_step!`](@ref)`(run; dt)` — the parent step ``Δt_p`` (its
   own pick, or `dt`), the prolongation at ``t_p`` (reused in a one-way
   run) and at ``t_p + Δt_p``, and the child's boundary window
   ``[t_p, t_p + Δt_p]``; it returns `window`, with `window.t_child_end`;
2. the caller's child steps, which must land on `window.t_child_end`:
   [`subcycle_dt`](@ref)`(R, limit)` is the rule above (``R/n``,
   ``n = \lceil R/\mathrm{limit}\rceil`` with the ``10^{-9}`` tolerance), and
   every core reads the blended parent at its own stage times;
3. [`end_parent_step!`](@ref)`(run, window)` — the clock check and snap,
   the two-way feedback, the statistics.

`step_nested!(run)` is exactly these with the run's own subcycling in
between; `test/era2_nest_external.jl` checks the two are bit-identical
(one- and two-way, four windows of three substeps). The same file runs
the physics of an externally driven child on synthetic boxes, with
adaptive child steps that do not divide the window (limit
``0.37\,Δt_p``): a parent at rest with a hydrostatic θ(z) leaves the child
at rest (max ``|u|`` = 1.0e-14 m/s after 12 steps, θ unchanged); a parent
at (2, 1) m/s leaves the child at (2, 1) m/s exactly; the net boundary
volume flux after the correction stays at 8.1e-18 ``UA`` and
max ``|∇·u|`` at 5.7e-14 s⁻¹; a +1 K bubble near the lid of a calm child
loses 5.0 % of its heat through the faces in 80 s and the time integral of
`boundary_scalar_flux` books it to 8.9e-12 of the anomaly's heat (the
periodic box keeps it to 6.8e-12); the parent's geostrophic tendency is
exactly ``(-fV_g, fU_g)`` and the child's IBM hook runs in every child step.

A run with a [`NestSponge`](@ref) cannot be driven this way (#349): the
sponge relaxes the child inside the run's own subcycling (the velocity on
every stage's predicted velocity, the scalars after each substep), so an
externally driven child would get none, and nothing would say so.
`begin_parent_step!` and `begin_child_step!` refuse such a run with an
`ArgumentError` before either domain is touched; an externally driven
child is relaxed by its core's own sponge instead (the Berlin case's top
sponge toward the parent's mean profile).

## The parent stepped concurrently (SC-c, #403)

In a one-way run the data dependencies between the two domains are

```
parent step n ──► B(t_n+1) ──► child substeps of window n
      │
      └──► parent step n+1 ──► B(t_n+2) ──► child substeps of window n+1
```

The parent step of window ``n+1`` reads nothing but the parent at the end
of window ``n``: it does not wait for the child. So it can run on its own
task while the child subcycles through window ``n``:
[`prefetch_parent_step!`](@ref)`(run, window; dt)` starts it (after
[`begin_parent_step!`](@ref) has opened window ``n``), and the next
`begin_parent_step!` takes it instead of stepping the parent.
[`step_nested!`](@ref)`(run, n)` steps ``n`` windows this way, prefetching
every window but the first.

The exchange stays at the existing synchronisation points. The task
prolongates the parent at the end of its step into a spare state; when
the next window opens, its start state ``B(t_p)`` is the current window's
end state (the parent has not changed since: the `reuse_boundary` path)
and its end state is the task's. While the task runs, the child reads only
the current window's two states, which nothing writes. The step size is
the sequential one (the parent's `adaptive_dt` from its state at the end
of window ``n`` with ``Δt_n`` as the previous step, or the `dt` the caller
will ask for, which `begin_parent_step!` checks). Every operation is the
sequential run's, on the same data, so the result is **bit-identical**:
`test/era2_nest_concurrent.jl` checks child, parent, bound scalars,
clocks, statistics and step records against window-by-window stepping
over 5 windows with fixed and adaptive parent steps, SSP-RK3 and the
projection step, a moist pair with a tracer, a sponged child and a
z-stretched parent, and the externally driven child of the Berlin
pattern.

The threads are shared between the two tasks: each domain's launches
spread over Julia's thread pool, and the scheduler divides it between
them. What can be prefetched, and what cannot:

* **one-way, `sync = :subcycle`** (or a fixed `dt`): prefetched;
* **two-way**: not — the parent step of window ``n+1`` needs the
  feedback of window ``n``, and the child's window ``n`` needs
  ``B(t_{n+1})`` from the parent step ``n``: the chain is strictly serial.
  `prefetch_parent_step!` refuses it (`ArgumentError`) and
  `step_nested!(run, n)` steps it sequentially;
* **`sync = :shared_min_dt`**: not — the step size needs the child's
  limit as well.

Until the next `begin_parent_step!` (or [`wait_parent_step!`](@ref)) the
parent belongs to the task: the caller must not read or change it, the
parent's hooks run on the task, and `end_parent_step!` takes the parent's
clock and limiter from the window instead of reading the parent. A run
that stops after a prefetch has its parent one window ahead. The Berlin
case prefetches only under `run_case!`, which knows the run's end
([berlin_nesting.md](../design/berlin_nesting.md)).

**Measured** (the Berlin flagship, `CaseConfig(:summer)`, 64 × 64 × 64
child in its 192 × 192 × 53 parent): see the SC-c section of the KA port
design page.

A child tracer the parent does not carry (in Berlin: ql, qr, the chemistry
species) still needs a face condition: its plane is added after the core
is built ([`add_open_scalars!`](@ref Aeolus.BoundaryConditions.add_open_scalars!))
with the `:inflow` halo — the parent's value (here zero) where the face's
normal velocity enters, the zero-gradient mirror where it leaves.

## Feedback

* **`:one_way`** (default, the Berlin default). The parent never sees the
  child. The parent of a one-way run is bit-identical to a standalone parent
  run with the same ``Δt`` sequence; the test checks this over 20 adaptive
  steps. The boundary data at the start of a step is reused from the end of
  the previous one.
* **`:two_way`** (or a [`TwoWayFeedback`](@ref)). After the child reaches
  ``t_p + Δt_p``, its field set is anterpolated onto the covered parent
  cells outside a buffer zone, the interface flux of the open faces is
  corrected, the scalars are refluxed from flux registers booked at every
  stage of both domains, and the parent is re-projected; see the
  [two-way nesting page](nesting_twoway.md). The conservation ledger is
  read with [`nest_conservation_error`](@ref).

The historical `step_nested!(parent, child, nest, dt; …)` is a thin wrapper.
It is a two-way `NestedRun` with a fixed ``Δt``, a fixed `substeps`
(default ``\max r``) and no step validation.

## Measured behaviour (test/era2_nested_run.jl)

| property | result | bound |
| :-- | :-- | :-- |
| stage-time halo error, parent linear in ``t``, all four cores | ``\le 4.6\times10^{-16}`` (relative) | ``10^{-14}`` |
| boundary error ratio when ``Δt_p`` is halved, ``\sin ωt`` | 3.957, 3.989 | ``\ge 3.7`` |
| substeps when the child limit is ``Δt_p/r`` | ``n = r`` for ``r = 2..5`` | ``= r`` |
| linear drag ``-cu`` through the hooks, RK3 | ``\lvert\bar u/e^{-ct}-1\rvert = 4.2\times10^{-8}`` | ``10^{-6}`` |
| the same, anelastic child, drag integrated exactly | ``2.4\times10^{-14}`` | ``10^{-6}`` |
| the other domain under one-way | unchanged (0) | ``10^{-14}`` |
| 1000 adaptive parent steps, one-way / two-way: clock error | 0 / 0 | ``10^{-12}\,t`` |
| the same: maximum child CFL | 0.4911 / 0.4877 | ``C_{\rm adv} = 0.5`` |
| the same: ``KE_{\rm child}/KE_{\rm parent}`` over the covered box | [0.954, 1.102] / [0.954, 1.108] | [0.85, 1.15] |
| two-way interface mass imbalance after the correction | 0 | ``10^{-10}`` |

The last three rows are for the open-faced child of N-nest-d (#261), with
no sponge; N-nest-c measured 0.4898 / 0.4992, [0.959, 1.096] /
[0.960, 1.102] and ``3\times10^{-19}`` with a periodic child projection, a
re-imposed normal velocity and a 3-cell sponge.
| injected child NaN: child state vs. the uninjected run | ``1.1\times10^{-16}``; 10 parent calls, not 11 | ``10^{-12}`` |

## Cost for the Berlin child

`benchmarks/bench_nested_run.jl` runs a 192 × 192 × 120 parent (6 × 6 × 3
km, Δx = 31.25 m, uniform Δz = 25 m) and a 1 × 1 km child at
``(2, 2, 4)``. Both use Smagorinsky SGS and a 5 m/s westerly with noise.
The run takes 12 adaptive parent steps of ``Δt_p = 1.44 - 1.80`` s on one
core. Since N-nest-d (#261) the child's sides and top are open nest faces
(`OpenBoundary`) over a ground wall, with no sponge; the boundary column
includes the per-stage refresh of the open-boundary buffers:

| child | ``n`` | ``Δt_c`` (s) | max child CFL | parent / child / boundary / feedback per parent step (s) |
| :-- | :-- | :-- | :-- | :-- |
| 64 × 64 × 32, one-way | 4–5 | 0.34–0.45 | 0.494 | 0.386 / 0.061 / 0.012 / — |
| 64 × 64 × 64, one-way | 4–8 | 0.22–0.45 | 0.497 | 0.365 / 0.147 / 0.014 / — |
| 64 × 64 × 64, two-way (with flux correction) | 4–6 | 0.24–0.45 | 0.488 | 0.349 / 0.116 / 0.025 / 0.098 |

The child costs 0.16× the parent at 32 levels and 0.40× at 64 levels
(N-nest-c, with a wall lid, a sponge and the periodic-projection child:
0.24× and 0.50×). In this case ``n`` exceeds the horizontal ratio 2
because the child's vertical spacing is 4× finer, so the vertical term of
the per-cell CFL sum weighs more. The two-way case now runs with the
interface flux correction: N-nest-c had to switch it off, because it
forced the parent's flux through the child's wall lid to ``w = 0``.

## Current limits

These are owned by other issues of the nesting epic (#246).

* **Resolved by N-nest-d (#261).** The child's projection now sees the
  parent's boundary flux (open faces, all-Neumann solve, the single
  mass-flux correction), and a wall below a nest face above is a supported
  child (the Berlin shape). With N-nest-c's periodic child projection a
  Taylor-Green child blew up within about 10 steps unless the normal
  velocity was re-imposed and a sponge added; it now runs the 1000-step
  tests with neither, and the benchmark child has an open top.
* **Resolved by N-nest-e (#263): two-way with walls.** The interface flux
  correction and the refluxing act on the child's open faces only; a child
  wall that is not the parent's wall is reported (the composite integral
  cannot be conserved across it).
* **Resolved by #329: the sponge and ∇·u.** A `NestSponge` relaxes the
  child velocity on the predicted velocity of every stage, before the
  pressure solve (`post_predictor!`), so the child stays divergence-free
  (``2.8\times10^{-14}`` s⁻¹, against 1.6 s⁻¹ when relaxed after the
  step); the scalars are relaxed after the substep. A user core must call
  `post_predictor!` for the velocity part to act.
* **Resolved by #329: memory and allocations.** Without a sponge the
  run's three field sets are halo-only (`storage = :shell`, 42 % of the
  child-sized sets for a ``32^3`` child, 24 % for ``64^3``); the rollback
  state of each domain and the two-way scratch are allocated once. Per
  nested step (``32^3`` parent, ``32^3`` child at ``r = 2``, `step!`), the
  nesting layer allocates 24 KiB one-way and 44 KiB two-way on top of the
  cores, against 2.9 MiB and 6.6 MiB before (the composite momentum
  integral alone allocated 858 KiB per call, twice per two-way step); the
  periodic halo fill no longer allocates, so a 32³ `step!` went from
  1.6 MiB to 0. The state is concretely typed (no `Any`, `Vector{Any}`,
  `Vector{Field}` or `Dict{Any,Any}`), as the KernelAbstractions port
  (#291) needs.
* **Resolved by #329: refluxing at the parent boundary.** The correction
  of a child face lying on the parent's own boundary goes to the wrapped
  cell across a periodic seam and nowhere across a non-periodic boundary
  (it used to wrap on every axis, so a child at the parent top would have
  corrected the parent's lowest layer); `apply_flux_correction!` takes the
  parent's periodic axes ([`nest_periodic_axes`](@ref)). Since #349 the
  `periodic` keyword of the cell-centred `apply_flux_correction!` and of
  `reflux!` is required: its triply periodic default was wrong for every
  parent with a wall or an open face.
* **Resolved by #328: hierarchies, siblings, non-aligned and stretched
  children.** A [`NestHierarchy`](@ref) advances a tree of domains by
  recursive subcycling, every link a `NestedRun`; children may have a
  rational ratio, edges off the parent faces or a stretched z. See
  [nest hierarchies, non-aligned and stretched children](nesting_hierarchy.md).

## References

- Ascher, U. M., Ruuth, S. J. & Spiteri, R. J. (1997). Implicit-explicit
  Runge-Kutta methods for time-dependent partial differential equations.
  *Appl. Numer. Math.* 25, 151–167.
- Berger, M. J. & Colella, P. (1989). Local adaptive mesh refinement for
  shock hydrodynamics. *J. Comput. Phys.* 82, 64–84.
- Carpenter, M. H., Gottlieb, D., Abarbanel, S. & Don, W.-S. (1995). The
  theoretical accuracy of Runge-Kutta time discretizations for the initial
  boundary value problem: a study of the boundary error. *SIAM J. Sci.
  Comput.* 16, 1241–1252.
- Hellsten, A. et al. (2021). A nested multi-scale system implemented in
  the large-eddy simulation model PALM model system 6.0. *Geosci. Model
  Dev.* 14, 3185–3214.
- Shu, C.-W. & Osher, S. (1988). Efficient implementation of essentially
  non-oscillatory shock-capturing schemes. *J. Comput. Phys.* 77, 439–471.
