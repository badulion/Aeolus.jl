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
(`u, v, w, θ`, `qv`, extra scalars). Linear interpolation of a smooth
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

## Feedback

* **`:one_way`** (default, the Berlin default). The parent never sees the
  child. The parent of a one-way run is bit-identical to a standalone parent
  run with the same ``Δt`` sequence; the test checks this over 20 adaptive
  steps. The boundary data at the start of a step is reused from the end of
  the previous one.
* **`:two_way`**. After the child reaches ``t_p + Δt_p``, its field set is
  restricted conservatively onto the covered parent cells, and the interface
  mass-flux mismatch is corrected (Berger & Colella 1989). The conservation
  ledger is read with [`nest_conservation_error`](@ref).

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
* **Two-way with walls.** The interface flux correction treats all six
  faces as interfaces. With an open child top that is consistent (the
  child's top flux is the parent's), but a child face on a wall that is not
  the parent's wall would still be forced. Per-face two-way feedback is
  N-nest-e (#263).

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
