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
moisture, scalars, outflow, halo)`. Every `:nest` face is an `OpenBoundary`
(`outflow = :convective` or `:orlanski` makes its outflow points radiate
instead of taking the parent's value, `halo` is checked against the
advection scheme's stencil, #329); the Berlin
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

`NestedRun(...; solid = mask)` (or [`apply_nest_boundary!`](@ref)`(...;
solid)`) passes the child's solid mask, and the run marks the open faces
with the one open-boundary path,
[`mark_open_boundary_solids!`](@ref Aeolus.BoundaryConditions.mark_open_boundary_solids!)
(N-open-d #262). A boundary point is solid when an interior cell it
borders is solid (the cell next to the face for the normal velocity and the
scalars, either of the two cells for a staggered tangential component).
Every velocity buffer — the normal velocity on the face and the layers
beyond it, **and** the tangential velocities — is zero there, and the
point is flagged non-fluid in its `OpenPlane`, so the mass-flux correction
neither counts nor adjusts it. (Until #327 the nest had a second
implementation, `NestBoundaryData.solid`, that zeroed only the normal
velocity; it is retired, and `nest_boundary_data` no longer takes a
`solid` keyword.) A building cut by the east face of a test child
(16 solid faces of 128, 40 tangential solid points) keeps ``u·n = 0.0``
and zero tangential velocity exactly, and
``δ = Q / A_{\rm fluid}`` holds bit for bit. The parent does not see the
building, so the solid faces create a genuine imbalance
(``Q = -2.97\times10^4`` m³/s, ``δ = -0.063`` m/s, 2 % of ``U``) that the
correction spreads over the fluid faces.

## Sponge

A [`NestSponge`](@ref) (relaxation zone, Davies 1976) is optional and
**off by default**: Kadasch et al. (2021, §2.2–2.3) and Hellsten et al.
(2021) use none at the child boundary. When given, it relaxes every member
of the field set toward the parent blended to the end of the child
substep, on the open faces only (`nest_open_sides`). The velocities are
relaxed on the predicted velocity of every stage, in the core's
`post_predictor!` hook, *before* the pressure solve; the scalars after the
substep (#329). The prolongated parent velocity is flux-consistent on
parent cells, not divergence-free on child cells, so the former relaxation
after the step left the child divergent: ``\max\lvert∇·u\rvert = 1.6``
s⁻¹ for a random parent (width 3, ``α_{\max} = 20`` s⁻¹), against
``2.8\times10^{-14}`` s⁻¹ now, with `step!` and `step_rk3!`. Its effect on
an outgoing anomaly is in the next section.

## Storage and cost of the boundary data

Without a sponge the run reads the prolongated parent only at the child's
open-boundary points, so its three field sets (the parent at both ends of
the parent step and their blend) are halo-only (`storage = :shell`, #329):
six dense halo slabs per member plus the boundary-normal face plane of the
velocities, ``(N+2H)^3 - N^3`` instead of ``(N+2H)^3`` points. For a
``32^3`` child with ``H = 3`` the three sets of `u, v, w, θ` hold 2144 KiB
instead of 5144 KiB (42 %); for ``64^3`` a shell is 24 % of the full
array. The results are bit-identical. The prolongation itself loops over
the six halo slabs (one kernel, `_interp_box!`), not over the whole child
array.

## Validation (`test/nest_boundary.jl`; the convective boundary layer in `validation/nest_boundary.jl`)

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
  scheme of the Berlin tracers) never creates it (0.007 %). A radiating
  outflow face removes it too: `nest_bcs(…; outflow = :convective)` (#329)
  lets a θ pulse that only the child carries leave with a residual
  ``\int θ'^2 / \int θ'^2_0 = 1.2\times10^{-7}`` after two crossings,
  where the Dirichlet parent face keeps all of it (1.00).
* **Refinement 1 is exact for the projection step.** The predictor sees
  the same halo values as the parent, and the Neumann solve with the
  parent's (end-of-step) face velocities reproduces the parent's interior
  pressure gradient, so the child stays on the parent to round-off. With
  SSP-RK3 the intermediate stages see the blended parent instead of the
  parent's own stage values, an ``O(Δt^2)`` difference per step.

## Limits

* No Berlin coupling on `master` yet: the Berlin multi-domain case is
  N-berlin-nest (#265, PR #330).
* Since #328 the boundary data of a [`NestHierarchy`](@ref) flow down
  level by level through the same contract (a grandchild's data come from
  the child's state at the child's substep ends), and a stretched child's
  open faces use the per-level face areas of its mesh in the mass-flux
  correction; see [nest hierarchies, non-aligned and stretched
  children](nesting_hierarchy.md). A one-way child with Dirichlet outflow
  faces keeps a reflected residual after a sharp feature has left it
  (measured there); a radiation outflow (`outflow = :convective`) does
  not.

Resolved since N-nest-d:

* **Outflow and halo (#329).** `nest_bcs` passes `outflow`
  (`:dirichlet`, the default, `:convective`, `:orlanski` or per side) and
  the advection `halo` to `open_boundary_conditions` (N-open-c, #260). A
  vertically periodic open box used to skip the initialisation of its
  radiating faces, which silently stayed Dirichlet; fixed with it.
* **Two-way feedback** is N-nest-e (#263): per-face flux correction and
  refluxing on the open faces only, Clark–Farley anterpolation with a
  buffer zone, re-projection of the parent; see the
  [two-way nesting page](nesting_twoway.md).
* **Prolongation limiter (#329).** The 3rd- and 5th-order prolongation
  overshoots at a sharp front (``[-0.25, 1.25]`` at order 3,
  ``[-0.30, 1.30]`` at order 5 for a 0/1 tracer). A conservative limiter
  scales the children of each parent cell about the parent value:
  `:positive` (Zhang & Shu 2010; the default for `qv` and the tracers of
  the field set) keeps them non-negative, `:monotone` (Barth & Jespersen
  1989) inside the parent neighbourhood's range; the children's mean stays
  the parent value. θ and the velocities are not limited by default.
* **One windowed boundary-data type (#329).**
  [`NestBoundaryData`](@ref) is the generic two-snapshot
  [`WindowedBoundaryData`](@ref Aeolus.BoundaryConditions.WindowedBoundaryData)
  with the prolongated field set as the snapshot; a mesoscale parent
  (#58) implements `snapshot_values` and reuses the time blend.
* **Externally driven children** refresh their halos with the public
  [`fill_nest_halos!`](@ref) (#329).

## References

- Barth, T. J. & Jespersen, D. C. (1989). The design and application of
  upwind schemes on unstructured meshes. AIAA paper 89-0366.
- Zhang, X. & Shu, C.-W. (2010). On maximum-principle-satisfying high
  order schemes for scalar conservation laws. *J. Comput. Phys.* 229,
  3091–3120 (the scaling limiter about the cell mean).

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
