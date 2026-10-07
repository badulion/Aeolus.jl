# Nest hierarchies, non-aligned and stretched children

Until #328 a nested run was one parent and one child, the child aligned
with its parent's cells by an integer ratio and uniform in z. This page
describes the generalisation along both lines:

* **geometry** — a child whose refinement is rational (five child cells
  in two parent cells), whose origin and extent are off the parent faces,
  or whose vertical grid is stretched ([`NestAxis`](@ref),
  `NestedGrid(parent, child)`);
* **topology** — a tree of domains: children, grandchildren and siblings,
  advanced by recursive subcycling ([`NestHierarchy`](@ref),
  [`add_nest!`](@ref), [`step_nested!`](@ref)).

Both reuse the operators of the [nesting page](nesting.md), the time
coupling of the [time-coupling page](nesting_time.md), the boundary data
of the [boundary page](nesting_boundaries.md) and the feedback of the
[two-way page](nesting_twoway.md); nothing is forked. The tests are in
`test/era2_nest_hierarchy.jl` (fast tier) and
`test/era2_nest_hierarchy_longrun.jl` (full tier).

## Geometry: one map per axis

A [`NestAxis`](@ref) describes the parent–child pair along one axis by
coordinates rather than by an integer ratio:

* the parent and child face coordinates, storage-indexed. A child face
  within ``10^{-9}`` of a child spacing of a parent face is replaced by
  that parent face's bits, so the Lagrange bases of the prolongation are
  exactly 0 and 1 there;
* the **common faces**: parent faces inside the child that are also child
  faces;
* the **conservation box**: the parent cells between the first and the
  last common face inside the child, and the child cells of the same box;
* the overlap length ``\omega(c, P)`` of every box parent cell ``P`` with
  every child cell ``c`` (compressed rows);
* for every box parent face, the coincident child face, or the two child
  faces that bracket it and the linear weight between them.

`NestedGrid(parent, child)` builds the three axes from the meshes and
checks that the child interior lies inside the parent interior, that the
child is at least as fine as the parent everywhere, and that the child's
storage (halos included) lies inside the parent's. `nested_grid` accepts
rational ratios (`(5//2, 2, 2)`) and extents off the parent faces.

| Case | Conservation box |
| :-- | :-- |
| integer ratio, edges on parent faces | the whole child (the N-nest-a geometry) |
| rational ratio, edges on parent faces (Δ_p = 100 m, Δ_c = 40 m) | the whole child; common faces every 2 parent cells |
| edges off the parent faces | shrunk to the first and last common faces inside the child |
| no two common faces along an axis | none: the child nests one-way only (two-way is refused) |

An axis that is aligned with an integer ratio and a uniform child keeps
the integer plans and kernels of N-nest-a; when all three are, the nest is
`aligned` and every operator is bit-identical to before #328.
`NestedGrid(parent, child; plan = :overlap)` forces the general path on an
aligned geometry, which is how the tests prove that the integer path is
its special case.

## Operators on overlap pieces

**Prolongation.** A child cell ``c`` that straddles parent faces is
integrated piecewise,

```math
\bar\phi_c = \frac{1}{|c|}\sum_{P \cap c \neq \emptyset}
             \int_{c \cap P} \mathcal{P}'_{s(P)}(\xi)\,d\xi ,
```

where ``\mathcal{P}_{s(P)}`` is the degree-``n_s`` polynomial interpolating
the primitive at the faces of the stencil centred on ``P`` (the
coordinate rule of `_cons_weights_nodes`, applied once per piece; the
pieces' weights are summed into one stencil row, one parent cell wider).
Each stencil polynomial matches the primitive at its own cell's faces, so
the child cells of every block of parent cells bounded by common faces
carry the block's integral exactly: conservation is structural per block.
``R\circ P = \mathrm{id}`` holds per parent cell only where every parent
face is common (the aligned and the stretched-child cases); with a
rational ratio it holds per block. The normal velocity at a child face is
the nodal Lagrange interpolant at the face's coordinate, as before.

**Restriction.** A box parent cell is the overlap-weighted mean

```math
\phi_P = \frac{\sum_c \omega_x\,\omega_y\,\omega_z\,\chi_c\,\phi_c}
              {\sum_c \omega_x\,\omega_y\,\omega_z\,\chi_c},
```

with ``\chi`` the fluid mask and ``\omega_z`` in the measure coordinate
(the reference mass ``\mu`` for a sound-proof pair), so
``\sum_P V_P\phi_P = \sum_c V_c\phi_c`` over the box exactly. A parent
face on a common plane takes the overlap-area mean of the child faces on
it (the child's flux); a box face between child face planes (inside the
box, rational ratio) the linear interpolation of the two bracketing
planes' means.

**Flux correction, registers, composite integrals.** The interface
planes are the box planes, which are common faces, so the interface flux
correction and the refluxing remain exact: the fine flux of a parent face
is the overlap-area mean of the child faces on it, the registers are
allocated over the box and book the child faces on the box planes
(interior child faces when the box is shrunk), the refluxed parent cells
are those just outside the box, and the composite integral counts the
parent outside the box and the child inside it. The strip between a
shrunk box and the child's edge is computed by the child but owned by the
parent (an extended buffer).

## A stretched child

A child on a z-[`StretchedGrid`](@ref) has its own face heights in its
vertical axis. The vertical measure takes the child faces from it (the
mass measure maps box faces to the child's ``\mu``), restriction weights
are the layer overlaps, and the volume elements of the composite integral
are the child's layer thicknesses. The child's open faces are
`OpenBoundary` faces on the stretched mesh: plane coordinates from the
mesh's face and centre heights, lateral face areas ``\Delta y\,\Delta z_k``
in the single mass-flux correction (`correct_boundary_mass_flux!`); its
projection is the stretched Poisson solver with Neumann sides, which
`_default_solver(::StretchedGrid)` already picks. A radiation outflow on
the bottom or top of a stretched mesh is refused (lateral faces only).

## The nest tree and its time stepping

A [`NestHierarchy`](@ref) holds the domains and the links between them.
Each link is a [`NestedRun`](@ref) of that parent–child pair: its
boundary data, flux registers, feedback (one-way and two-way links mix
freely), sponge, solid mask, ledger and statistics. One root step
([`step_nested!`](@ref)`(h)`) is Berger & Oliger's recursion (as in PALM,
Hellsten et al. 2021 §3.1, and WRF):

```
advance(d, step h):     # domain d takes one step of size h (with retries)
    for each child link L of d: boundary data B(t_d) → L.s_old
    step d (register hooks of its uplink and of all its child links)
      rejected → restore d only, retry at h/2 (no descendant has moved)
    for each L: B(t_d + h) → L.s_new, open L's window [t_d, t_d + h]
    for each L: the child subcycles to t_d + h, each accepted
                substep calling advance(child, substep) — the recursion
    for each L: close the window (clock snap, two-way feedback without
                the re-projection, ledger)
    one re-projection of d if any child is two-way
```

Each domain uses its own `CFLConstraints`, SGS closure and core; the
number of substeps comes from the CFL limit (re-evaluated before every
substep) or from the largest refinement ratio, rounded up. Under
`sync = :shared_min_dt` every domain takes one step of the smallest limit
over the tree. The boundary data flow down level by level through the
#58 contract (a grandchild's data come from the child's state at the
child's substep ends, linearly blended); the feedback flows up in post
order: a grandchild feeds back at the end of each of its parent's
substeps, before the parent's own feedback to the root. That ordering is
what makes the composite integral exact at every level — the child's
integral changes only by its boundary fluxes, which the root–child
registers booked, plus the grandchild's refluxing inside the child, which
conserves child plus grandchild.

**Siblings.** Several children of one domain, each with its own ratio,
geometry and feedback, must not overlap (the parent cells their interiors
cover are disjoint); two two-way siblings' boxes must be at least one
parent cell apart, or a cell refluxed by one would be anterpolated by the
other. Overlapping siblings are refused: a priority rule needs the
higher-priority sibling as boundary data of the other, a separate feature.
Within a step the siblings advance in link order; they read only the
parent's state at the window ends, write disjoint parent cells, and the
parent is re-projected once after all their feedback, so the result does
not depend on the order (bit for bit, tested).

**Proper nesting.** [`add_nest!`](@ref) refuses a child whose open faces'
halo stencils read parent cells outside the parent's interior, unless that
side of the parent is periodic ([`proper_nesting_violations`](@ref)): in a
hierarchy those would be the parent's own boundary halos.

**Rollback.** A rejected step (non-finite state, solver failure,
`max_velocity`; the root also `divergence_tol`) restores that domain only
— its state, bound scalars of all its links, SGS memory, open-boundary
buffers and pending register bookings — and is redone at half its size;
no descendant has moved within it. Only numerical failures are rejected;
a `MethodError` propagates. Escalation to the parent level (restoring a
subtree and halving the coarser step) is not implemented: a domain that
exhausts its retries raises an error, as a `NestedRun` child does.

**NestedRun** keeps its interface and its numbers: its parent step is
split into the pieces the hierarchy reuses (`_prepare_window!`, the
parent advance, `_open_window!`), and without a subtree the code path is
the former one. Golden checksums of 20 steps of six configurations
(one- and two-way, `step!` and `step_rk3!`, fixed and adaptive, moist,
sponge, solid), recorded before the change (`test/nest_golden_gen.jl`),
are reproduced bit for bit.

## Measured (`test/era2_nest_hierarchy.jl`)

Configurations: **H3** root ``16^3`` (Δ = 100 m, periodic), child
``r = 2`` over the middle ``8^3`` parent cells, grandchild ``r = 2`` over
the child's middle; **S2** two siblings, ``r = 2`` and ``r = 3``, two
parent cells apart; **NA** ``r = 5/2`` in x with the child's edges 0.4 and
0.8 parent cells off the faces (box shrunk to 6 parent cells); **ST** a
stretched child, 3, 3, 3, 2, 2, 2, 1, 1 layers per 100 m parent layer.

| Check | Measured | Target |
| :-- | :-- | :-- |
| free stream U = (5, 2, 0) m/s, 20 root steps, every config, one-/two-way, `step!`/`step_rk3!` | ``\max|u-U|/|U| \le 3.8\times10^{-15}``, ``|θ-θ_0|/θ_0 \le 7.4\times10^{-15}``, ``|∇·u|Δ/|U| \le 7.5\times10^{-16}`` (H3: exactly 0) | ``10^{-12}``, ``10^{-13}``, ``10^{-12}`` |
| composite ``∫θ``, two-way with refluxing, 40 root steps (H3, S2, NA, ST) | per step ``\le 5.2\times10^{-15}``, total ``\le 6.6\times10^{-15}`` | ``10^{-13}``, ``10^{-12}`` |
| same without refluxing (negative control) | per step ``3.5\times10^{-8}`` … ``2.7\times10^{-7}`` | — |
| H3 moist, WENO tracer slot booking into both links: ``∫θ, ∫q_v, ∫\mathrm{co_2}`` | per step ``\le 4.5\times10^{-15}`` | ``10^{-13}`` |
| anelastic pair, ``r = 5/2`` horizontally (mass measure), 20 steps: ``∫\bar\rho θ`` | ``1.0\times10^{-14}`` | ``10^{-12}`` |
| sibling order permuted (S2, 15 steps) | bit-identical | bit-identical |
| box integral of prolongation / restriction (r = 5/2 × 3/2, off-face, 7/3, stretched) | ``\le 2.2\times10^{-15}`` | ``10^{-14}`` |
| stretched child: ``\max|RPφ-φ|/\max|φ|`` | ``4.1\times10^{-16}`` | ``10^{-14}`` |
| prolongation order, r ∈ {5/2, 3/2, 7/3} × order ∈ {2, 3, 4}, origin off the faces | 1.99–4.00 | ≥ order − 0.2 |
| prolongation order, stretched child, order 2, 3, 4 | 1.98, 2.99, 3.98 | ≥ order − 0.2 |
| overlap path vs integer path (prolongation, restriction; uniform `StretchedGrid` child) | ``2.7\times10^{-16}`` | ``10^{-14}`` |
| 20 two-way RK3 steps, overlap vs integer path | ``1.3\times10^{-15}`` | ``10^{-12}`` |
| stretched child, 10 steps: net open-boundary flux, ``\max|∇·u|`` | ``9.8\times10^{-17}|U|A``, ``3.2\times10^{-17}`` s⁻¹ | ``10^{-12}|U|A``, ``10^{-11}`` |

**Against a fine reference.** A quasi-1-D channel (6.4 km at Δ = 100 m,
U = 5 m/s, ν = 2 m²/s, SSP-RK3), nests spanning the parent in y and z:
child ``r = 2`` over [2400, 4000] m, grandchild ``r = 2`` over
R = [2800, 3600] m; RMS errors over R against the exact advected-diffused
Gaussian:

| Case | coarse only | child | H3 grandchild | reference |
| :-- | :-- | :-- | :-- | :-- |
| σ = 100 m (1 root cell), t = 100 s, one-way | 0.235 | 0.106 | 0.070 | uniform Δ = 25 m: 0.031 |
| same, two-way | 0.235 | 0.106 (H3 child 0.065) | 0.062 | |
| blob starting in the grandchild, t = 60 s, one-way / two-way | 0.202 | | 0.0193 / 0.0190 | two-level child at r = 4: 0.0190 |

The grandchild's error is within 2 % of a two-level run whose child has
the grandchild's resolution over the child's region: the extra interface
costs little. Where the blob first crosses the r = 2 child (σ = 100 m case)
the grandchild inherits the child's error (0.062–0.070 against 0.031 for a
uniform 25 m grid). Convergence with all domains refined together
(N = 32, 64, 128, σ = 200 m): grandchild errors 4.2e-2, 1.35e-2, 3.6e-3,
observed order 1.63 and 1.91.

After the blob has left (σ = 300 m, t = 1000 s) the nests hold at most
8.7e-3 K (two-way) and 9.5e-3 K (one-way with a `:convective` outflow),
the size of the root's own dispersive wake there (8.4e-3 K in a
coarse-only run). A one-way child with **Dirichlet** outflow faces keeps
4.6e-2 K: at its outflow face it receives the parent's coarse blob while
its own fine blob leaves, and the centred scheme reflects the mismatch.
The 1e-2 K target of the design holds two-way and with a radiation
outflow, not one-way with Dirichlet outflow.

**Rollback.** A NaN injected into one grandchild substep (one-way
grandchild) rolls back the grandchild only: the root and the child are
bit-identical to a clean run, the grandchild link records one retry and
one extra substep.

**Adaptive.** In H3 with `CFLConstraints(C_advective = 0.5)` on every
level each domain takes its own CFL step (root 7.14 s, two substeps per
level at CFL 0.50) and lands on its parent's step ends; the composite
``∫θ`` drifts by ``2.8\times10^{-15}`` in 10 root steps.

**Long run** (`test/era2_nest_hierarchy_longrun.jl`, full tier). A
forced neutral (Kolmogorov) flow in a ``16^3`` periodic box with
Smagorinsky closure, child ``r = 2`` over the middle half and grandchild
``r = 2`` over the child's middle half, both links two-way with refluxing,
adaptive steps on every level (CFL 0.5), 300 time units (62 eddy
turnovers): 7496 root steps, 37 373 child and 74 746 grandchild substeps,
no rejected step; ``\max|∇·u|`` per level ``3.6\times10^{-15}``,
``7.1\times10^{-15}``, ``1.6\times10^{-14}``; composite ``∫θ`` of a passive
blob per root step ``\le 6.6\times10^{-15}``. The root's kinetic energy
over the last 20 turnovers is 0.915 of the two-level run's; its 20-turnover
means differ by a ratio 0.82 between the last two windows (seeds 1 and 3:
1.08, 1.05; the two-level run: 1.01). Cost (the recorded full-tier run): 0.39 s per time unit against
0.09 s for the two-level run (×4.5; the grandchild takes twice the child's
substeps). Wall-clock cost varies with machine load: an earlier run on a
less loaded machine measured ×3.3.

## Limits

* Overlapping siblings are refused (no priority rule).
* No escalating rollback across levels (a domain that exhausts its retries
  raises an error).
* Siblings advance one after the other (threading them is a separate
  change).
* `AdaptiveMesh` keeps its own index-space proper-nesting check (block
  margins in its level index space), which is a different criterion from
  the stencil reach checked here.
* The horizontal axes must be uniform (only z may be stretched), and a
  rational ratio or an extent off the parent faces in `nested_grid` needs
  a uniform parent (on a stretched parent build the child mesh and call
  `NestedGrid(parent, child)`).
* `InflowTurbulence`'s parent-resolved statistics need an integer
  horizontal ratio.

## References

* Berger, M. J. & Oliger, J. (1984), Adaptive mesh refinement for
  hyperbolic partial differential equations, *J. Comput. Phys.* 53,
  484–512.
* Berger, M. J. & Colella, P. (1989), Local adaptive mesh refinement for
  shock hydrodynamics, *J. Comput. Phys.* 82, 64–84.
* Hellsten, A. et al. (2021), A nested multi-scale system implemented in
  the large-eddy simulation model PALM model system 6.0, *Geosci. Model
  Dev.* 14, 3185–3214.
