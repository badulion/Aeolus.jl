# Nested boundary conditions: one-way parent → child data

A nested child is a limited-area LES whose lateral faces and top are not
walls and not periodic: the flow enters and leaves through them, and what
enters is what the parent says. This page describes how the parent's
solution becomes the child's boundary condition (N-nest-d, #261), with the
pieces it is built from:

| piece | where | page |
| :-- | :-- | :-- |
| open faces, the boundary-data contract, the mass-flux correction | `OpenBoundary`, `AbstractBoundaryData`, `correct_boundary_mass_flux!` (N-open-b, #255) | [fractional step](fractional_step.md) |
| the projection with Neumann faces | `CosineFFTPoissonSolver` from `axis_poisson_bc` (N-open-a, #251) | [Poisson](poisson.md) |
| parent → child prolongation of the field set | `interpolate_parent_state!` (N-nest-a, #250) | [nesting](nesting.md) |
| time blend over the parent step, subcycling | `NestedRun`, `boundary_time` (N-nest-c, #257) | [nested time coupling](nesting_time.md) |
| **the parent as boundary data** | [`NestBoundaryData`](@ref), [`nest_bcs`](@ref) (this page) | |

Nothing here is new physics or new numerics: `NestBoundaryData` is the
glue that feeds the existing prolongation and time blend into the existing
open-boundary buffers, and it is the Aeolus-parent implementer of the
same contract a mesoscale parent (#58) and reanalysis driving (#57) will
implement.

## What the child's faces carry

The child is built with [`nest_bcs`](@ref)`(child_mesh, x, y; bottom, top,
moisture, scalars)`. Every `:nest` face is an `OpenBoundary`; the Berlin
child has open sides and top and a wall at the ground, a child inside its
parent (not touching the ground) has an open bottom too. On an open face
(PALM's Dirichlet child boundary, Hellsten et al. 2021 §3.2):

* the **normal velocity** is prescribed on the boundary face and on the
  ``H`` halo layers beyond it. On a west, south or bottom face the boundary
  face is the child's interior face index ``H+1``; on an east, north or top
  face it is the first halo index ``H+N+1``;
* the **tangential velocities and every scalar** of the field set (`θ`,
  `qv`, extra tracers) get Dirichlet halos: the parent's value at the halo
  cell;
* the **pressure correction** gets a Neumann (mirror) halo,
  ``∂φ/∂n = 0``, so the projection
  ``u^{n+1}·n = u^*·n − Δt\,∂φ/∂n`` leaves the prescribed normal velocity
  unchanged (Kim & Moin 1985). The child's default solver is therefore the
  all-Neumann `CosineFFTPoissonSolver`. Before N-nest-d the child kept a
  periodic FFT projection that never saw the parent's flux, and a
  Taylor–Green child needed the normal velocity re-imposed after every
  stage plus a sponge to stay stable; it now runs 1000 adaptive parent
  steps without either.

## The boundary values

At time ``t`` of the current parent step ``[t_p, t_p+Δt_p]`` every buffer
receives

```math
B(t) = (1-τ)\,\mathcal P φ_p(t_p) + τ\,\mathcal P φ_p(t_p+Δt_p),
\qquad τ = \frac{t - t_p}{Δt_p},
```

at the buffer's own storage points, for every member ``φ`` of the field
set. ``\mathcal P`` is the conservative staggered prolongation of the
[nesting page](nesting.md): nodal (Lagrange) along a velocity component's
own direction, conservative cell-average reconstruction across it and for
scalars. Each dynamical core asks for the data itself, once per projected
stage, at that stage's end time (`update_open_boundaries!`): ``t_0+Δt``
for `step!`, ``t_0+γΔt`` and ``t_0+Δt`` for ARK2, and for SSP-RK3
``t_0+Δt,\ t_0+2Δt,\ t_0+\tfrac32Δt`` with the buffers combined with the
fields' Shu–Osher weights, which places ``B`` at the stage times
``(0, 1, \tfrac12)`` exactly because ``B`` is linear in ``t``
([nested time coupling](nesting_time.md)). The run refreshes the buffers
once more at the start of each parent step, where a two-way parent may
have changed.

## Mass

With every face's normal velocity prescribed, the Neumann problem
``∇^2φ = ∇·u^*`` is solvable only if the net boundary flux vanishes;
otherwise the solver leaves a uniform divergence ``Q/V``. The single
global correction of N-open-b subtracts ``δ = Q/\sum A`` from the outward
normal velocity of every open fluid face (Hellsten et al. 2021, §3.3,
Eq. 2). It is not reimplemented for nesting.

For a nest the correction has almost nothing to do. The child's faces lie
on parent faces. Along the face normal the prolongation is nodal, so it
reproduces the parent's face value at a coincident face; across the face
it is conservative, so the ``r_1 r_2`` child faces covering a parent face
carry exactly the parent's flux. Summed over the closed boundary this is
the parent's ``\int ∇·u\,dV`` over the covered box, zero for a projected
parent, and the time blend of two zero-flux states has zero flux. Measured
for a random divergence-free parent: ``|Q|/(UA) \le 5.7\times10^{-17}``
before the correction, ``|δ|/U \le 1.4\times10^{-17}``. Hellsten et al.
(2021) report ``δ`` three to four orders of magnitude below ``U`` for
PALM's (non-conservative) interpolation; here it is round-off, and ``δ``
does real work only where the child's geometry differs from the parent's.

## Faces inside solids

`NestedRun(...; solid = mask)` (or [`nest_boundary_data`](@ref)`(...;
solid)`) passes the child's solid mask. An open face whose adjacent child
cell is solid gets ``u·n = 0`` on the face and the layers beyond it, and is
flagged non-fluid in its `OpenPlane`; the mass-flux correction neither
counts nor adjusts it. A building cut by the east face of a test child
(16 solid faces of 128) keeps ``u·n = 0.0`` exactly, and
``δ = Q / A_{\rm fluid}`` holds bit for bit. The parent does not see the
building, so the solid faces create a genuine imbalance
(``Q = -2.97\times10^4`` m³/s, ``δ = -0.063`` m/s, 2 % of ``U``) that the
correction spreads over the fluid faces.

## Sponge

A [`NestSponge`](@ref) (relaxation zone, Davies 1976) is optional and
**off by default**: Kadasch et al. (2021, §2.2–2.3) and Hellsten et al.
(2021) use none at the child boundary. When given, it relaxes every member
of the field set toward the time-blended parent after each child substep,
on the open faces only (`nest_open_sides`). Its measured effect is in the
next section.

## Validation (`test/era2_nest_boundary.jl`)

| test | result |
| :-- | :-- |
| divergence-free random parent, children inside (r = 2) and on the ground (r = (2,2,4), 3): net boundary flux after the correction | ``\le 3.2\times10^{-17}\,UA`` (bound ``10^{-13}``) |
| the same: ``\lvert δ\rvert/U``; child ``\max\lvert∇·u\rvert`` | ``\le 1.4\times10^{-17}``; ``\le 5.7\times10^{-14}`` s⁻¹ |
| Gaussian θ anomaly, U = 5 m/s, centred θ (the core's scheme): child / parent L2 error over the overlap while inside | 0.593 (r = 2), 0.530 (r = 3) |
| the same, WENO-Z5 θ with an order-5 prolongation | 0.933, 0.916 (order 3: 3.54, 3.46) |
| residual max ``\lvert θ'\rvert`` after the anomaly left, centred θ, no sponge / `NestSponge(6, 0.2)` | ``3.1\times10^{-2}`` / ``7.6\times10^{-4}`` (r = 2); ``4.1\times10^{-2}`` / ``2.3\times10^{-3}`` (r = 3) |
| the same, WENO-Z5 θ | ``7.2\times10^{-5}`` / ``7.6\times10^{-5}`` (r = 2); ``7.7\times10^{-5}`` / ``7.9\times10^{-5}`` (r = 3) |
| convective boundary layer, overlap-mean θ(z), u(z) of child and parent after spin-up | differences ``\le`` 0.55 and 0.40 of the 4-seed run-to-run spread |
| parent θ trend ``a = 10^{-3}`` K/s: child top halo | follows it to ``1.1\times10^{-13}`` K |
| the same: child ``Δ\int θ`` vs. the diffusive flux through the top | equal to ``1.7\times10^{-12}`` (relative); no change at all with ``κ = 0`` |
| building cut by the east face | ``u·n = 0`` exactly; ``δ = Q/A_{\rm fluid}`` exactly |
| perfect parent, r = 1, 100 steps: relative L2 child − parent | ``\approx 3\times10^{-13}`` (`step!`), ``2.1\times10^{-5}`` (`step_rk3!`) |

Three findings:

* **The prolongation order must match the transport scheme.** With the
  core's 2nd-order centred θ the default 3rd-order prolongation is ample.
  With WENO-Z5 the parent's own error is ``O(Δx^5)`` and the 3rd-order
  interface error dominates: the child was 3.5× *worse* than the parent
  over the overlap; with `order = 5` it is better (0.93).
* **Centred advection and a Dirichlet outflow.** The 2nd-order centred
  scheme has an undamped ``2Δx`` mode with group velocity ``-U``. The
  Dirichlet outflow face excites it where the child's solution and the
  parent's data differ, and the Dirichlet inflow face reflects it: 3–4 % of
  the amplitude stays trapped in the child after the anomaly has left,
  above the 1 % the issue asks for (kept as a `@test_broken`). The sponge
  removes it (0.08–0.2 %), and an upwind-biased scheme (WENO-Z5, the
  scheme of the Berlin tracers) never creates it (0.007 %). The upwind
  scalar outflow halo is N-open-c (#260).
* **Refinement 1 is exact for the projection step.** The predictor sees
  the same halo values as the parent, and the Neumann solve with the
  parent's (end-of-step) face velocities reproduces the parent's interior
  pressure gradient, so the child stays on the parent to round-off. With
  SSP-RK3 the intermediate stages see the blended parent instead of the
  parent's own stage values, an ``O(Δt^2)`` difference per step.

## Limits

* No Berlin coupling yet: `Nesting` stays in `BERLIN_PENDING` until
  N-berlin-nest (#265) makes the nested run the default.
* Scalar outflow is Dirichlet. The upwind / convective outflow option and
  `boundary_scalar_flux` are N-open-c (#260).
* Two-way feedback keeps N-nest-c's restriction and six-plane flux
  correction; per-face two-way feedback (walls excluded), Clark–Farley
  anterpolation and the parent re-projection are N-nest-e (#263).

## References

- Davies, H. C. (1976). A lateral boundary formulation for multi-level
  prediction models. *Q. J. R. Meteorol. Soc.* 102, 405–418.
- Hellsten, A., Ketelsen, K., Sühring, M., Auvinen, M., Maronga, B.,
  Knigge, C., Barmpas, F., Tsegas, G., Moussiopoulos, N. & Raasch, S.
  (2021). A nested multi-scale system implemented in the large-eddy
  simulation model PALM model system 6.0. *Geosci. Model Dev.* 14,
  3185–3214.
- Kadasch, E., Sühring, M., Gronemeier, T. & Raasch, S. (2021). Mesoscale
  nesting interface of the PALM model system 6.0. *Geosci. Model Dev.* 14,
  5435–5465.
- Kim, J. & Moin, P. (1985). Application of a fractional-step method to
  incompressible Navier–Stokes equations. *J. Comput. Phys.* 59, 308–323.
- Shu, C.-W. & Osher, S. (1988). Efficient implementation of essentially
  non-oscillatory shock-capturing schemes. *J. Comput. Phys.* 77, 439–471.
