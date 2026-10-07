# Grid nesting: conservative staggered prolongation

A nested child grid refines a box of a parent grid by integer ratios
``(r_x, r_y, r_z)`` (each ``\ge 1``, odd, even or mixed, e.g. the Berlin
``(2, 2, 4)``). Every operator between the two grids lives in
`Aeolus.Nesting` ([API](../api/era2/nesting.md)); this page derives the
three on which everything else rests — prolongation, restriction and
refluxing — and quotes the numbers they are tested at
(`test/era2_nesting.jl`).

## Geometry

This page treats the aligned geometry: the child origin sits on a parent
face, ``N_\text{child} = r\,n`` for an integer number ``n`` of covered
parent cells, and ``\Delta_\text{child} = \Delta_\text{parent}/r`` in each
direction (`NestedGrid(parent, child; refinement, offset)` checks all
three and refuses a misaligned child with an error that names the
direction). On this geometry child face ``q\,r + 1`` (counted from the
nest's low face) coincides with parent face ``q``, for every
``q = 0, \dots, n``. Rational ratios, children off the parent faces and
z-stretched children (#328) use the same constructions on overlap pieces;
see [nest hierarchies, non-aligned and stretched children](nesting_hierarchy.md).

## Prolongation: the primitive-function construction

On a finite-volume grid a cell-centred value is a **cell average**
``\bar f_I = \frac{1}{\Delta}\int_{x_{I-1/2}}^{x_{I+1/2}} f\,dx``. A child
value must be an average over the child cell, and the ``r`` child
averages inside a parent cell must average back to the parent value —
otherwise prolongation creates or destroys the conserved quantity.

Both requirements are met by interpolating the **primitive function**
(Colella & Woodward 1984, §2),

```math
F(x) = \int_{x_{s-1/2}}^{x} f\,dx',
\qquad
F(x_{s-1/2+q}) = \Delta \sum_{q'=1}^{q} \bar f_{s-1+q'} ,
```

which the parent averages give **exactly** at the parent faces. For a
stencil of ``p`` parent cells, a degree-``p`` polynomial ``P`` through the
``p+1`` face values of ``F`` is built, and the average over child cell
``[a, b]`` is the difference quotient

```math
\bar f^{\,c}_{[a,b]} = \frac{P(b) - P(a)}{b - a}.
```

Because ``P = F`` at the parent faces, the ``r`` child averages of one
parent cell telescope to ``(F(x_{I+1/2}) - F(x_{I-1/2}))/\Delta = \bar f_I``:
conservation is **structural**, independent of ``p`` and of ``r``, not a
property of the accuracy. ``P'`` reproduces polynomials of degree
``p-1``, so the reconstruction is formally order ``p``. In 3-D the
weights are a tensor product of 1-D weights, precomputed once per
(direction, rule, order) into a plan cached on the `NestedGrid`.

## Staggering: nodal in the normal direction, conservative across it

A C-grid velocity is not a cell average in every direction. ``u`` on a
`FaceX` is a **point value in x** (it sits on the face) and a
**face-area average in y and z** (it is the volume flux through the face
divided by its area). The prolongation therefore uses, per direction,

| location | x | y | z |
| :-- | :-- | :-- | :-- |
| `CellCenter` (θ, `qv`, tracers) | conservative | conservative | conservative |
| `FaceX` (u) | nodal | conservative | conservative |
| `FaceY` (v) | conservative | nodal | conservative |
| `FaceZ` (w) | conservative | conservative | nodal |

where *nodal* is ``p``-point Lagrange interpolation of the face values.
Two exact properties follow.

* **Coincident faces.** The Lagrange basis evaluated at a node is
  exactly 0 or 1 in floating point, so a child face lying on a parent
  face takes the parent value in the normal direction without round-off.
  With refinement only along the normal (e.g. ``(3, 1, 1)`` for ``u``)
  the coincident child faces equal the parent faces bit for bit; this is
  tested for ``r = 3`` and ``r = 5``.
* **Flux preservation.** On a coincident face the tangential rule is
  conservative, so the ``r_t \times r_{t'}`` child faces covering a parent
  face carry, summed, exactly the parent volume flux. The parent and the
  child agree on the mass entering the nest before any correction.

A purely conservative (cell-average) treatment of ``u`` would get the
tangential flux sums right but smear the normal position; a purely nodal
one would get the point values right but not the fluxes. The mixed rule
is what makes a staggered nest both accurate and conservative (the same
split underlies the interactive nesting of Clark & Farley 1984 and the
PALM nest of Hellsten et al. 2021).

## Restriction

The child → parent operator is the exact adjoint in the conservative
directions: a volume-weighted mean of the ``r_x r_y r_z`` child cells of
each covered parent cell for cell-centred fields, and an area-weighted
mean of the ``r_t r_{t'}`` child faces on each aligned parent face for
velocities. With prolongation it satisfies

```math
\mathcal R\,\mathcal P = \mathcal I
```

on the covered parent cells and faces, for any ``r`` and ``p``.

## Refluxing

In a two-way nest the parent cells just outside the child were advanced
with the coarse flux through the interface, the child with the fine
fluxes. The composite integral (uncovered parent cells plus all child
cells) then drifts by the difference. With time-integrated flux
registers ``\Phi = \int F\,dt`` on both grids, the Berger & Colella (1989)
correction books the mismatch into the outside parent cell,

```math
\delta\Phi = \Phi_\text{coarse} - \frac{1}{r_t r_{t'}}\sum_\text{fine}\Phi,
\qquad
\bar f_\text{outside} \mathrel{+}= \pm\frac{\delta\Phi\,A_p}{V_p},
```

(``+`` on the nest's low face, ``-`` on the high face), after which the
composite integral changes only by what crossed the parent's own
boundary. `reflux!` does this for one scalar and `reflux_scalars!` for
every scalar of a `NestInterpState` field set.

## Field sets

The operators above are per-field. A `NestInterpState` names the set
they act on: ``u, v, w``, ``θ``, ``q_v`` when the simulations are moist,
and any extra cell-centred scalar passed as
`scalars = (:co2 => (child_field, parent_field), …)`. Prolongation, the
halo write, the sponge, restriction, the composite integral and
refluxing iterate over the set through one code path per member, so a
tracer initialised equal to ``θ`` stays bit-identical to ``θ`` through
all of them (tested).

## Measured accuracy and conservation

From `test/era2_nesting.jl` (smooth separable
``f = \prod_d \sin(2\pi x_d + \varphi_d)``, parent ``8^3 \to 16^3 \to 32^3``,
halo error of the child, rate from the finest pair):

| ``p`` | measured order, ``r \in \{2,3,4,5,(2,2,4)\}``, all four locations |
| :-: | :-- |
| 2 | 1.96 – 2.12 |
| 3 | 3.06 – 3.18 |
| 4 | 3.97 – 4.14 |

| property (random data) | measured | bound tested |
| :-- | :-- | :-- |
| ``\lvert\mathcal R\mathcal P f - f\rvert / \max\lvert f\rvert``, all ``r``, ``p``, locations | ``6.6\times10^{-16}`` | ``10^{-14}`` |
| summed child flux − parent flux, per interface face, ``/(\max\lvert u\rvert A)`` | ``3.5\times10^{-16}`` | ``10^{-13}`` |
| coincident child face, normal-only refinement, ``r = 3, 5`` | 0 (bitwise) | ``==`` |
| ``r\times r`` mean of coincident child faces, ``r = 3, 5`` | ``5.7\times10^{-16}`` | ``10^{-15}`` |
| composite ``\int θ``, ``\int q_v``, ``\int`` tracer after refluxing, ``r = (2,3,4)`` | ``0.9 - 3.5\times10^{-15}`` (drift without: ``2.6\times10^{-5} - 4.3\times10^{-4}``) | ``10^{-13}`` |

## Stretched parent, reference-mass measure, solid-aware restriction

The operators above were written for a uniform parent and a volume
average. Three generalisations (N-nest-b, #256) make them usable for the
Berlin nest, whose parent is z-stretched, whose sound-proof variant
weights by the reference density, and whose child contains buildings.

### A z-stretched parent

`NestedGrid` accepts a `StretchedGrid` parent (`child` stays a uniform
`RegularGrid`). The child bottom and top must lie on parent faces, and
every covered parent layer must hold exactly ``r_z`` child layers: the
overlap sits in a uniform part of the parent, which may stretch freely
above it. The face heights and layer thicknesses come from
`StretchedGrids.zfaces` / `cell_thickness` (one source of truth).

The primitive-function construction carries over unchanged once it is
written in a coordinate instead of in "parent cell numbers": the
primitive ``F(z_j) = \sum_{q \le j} \bar f_q\,\Delta z_q`` is known
exactly at the actual parent faces ``z_j``, a degree-``p`` Lagrange
polynomial on those (non-uniform) nodes interpolates it, and a child
value is the difference quotient across the child cell. The weight of
parent cell ``q`` for the child cell ``[a, b]`` is

```math
w_q = \frac{\Delta z_q}{b - a} \sum_{j \ge q} \big(L_j(b) - L_j(a)\big),
```

which reduces to the uniform weights for ``\Delta z_q \equiv 1``. Child
faces inside the overlap are generated from the parent faces, so a child
face lying on a parent face *is* that face bitwise, the basis is exactly
0/1 there, and ``\mathcal R \mathcal P = \mathcal I`` holds to round-off.
Nodal (``w``) weights are Lagrange weights on the actual face heights.
On a uniform `RegularGrid` parent the integer plans of the previous
sections are used, bit for bit.

### The reference-mass measure

For the sound-proof systems (`MassWeightedSimulation`: anelastic, with
``W = \bar\rho``; pseudo-incompressible, ``W = \bar\rho\bar\theta``) the
conserved quantity is ``\int W\phi\,dV``, not ``\int\phi\,dV``. The same
construction in the mass coordinate ``\mu(z) = \int W\,dz`` gives

* restriction ``\phi_P = \sum W_c \phi_c \Delta V_c / \sum W_c \Delta V_c``
  for scalars and ``u_P = \sum W_c u_c A_c / \sum W_c A_c`` for the
  horizontal-normal faces (the parent face carries the children's summed
  mass flux); on ``w`` faces ``W`` is common to all children and cancels;
* prolongation conserving ``\int W\phi\,dV`` exactly and keeping a
  constant constant (``F`` is linear in ``\mu``).

`nest_measure(nest, parent_sim, child_sim)` builds it from the sims'
`constraint_weights`, after asserting that parent and child share one
`ReferenceState` profile: `nest_reference_mismatch` compares the parent's
``\bar\rho\,\Delta z_p`` with the sum over its ``r_z`` children. One
smooth profile sampled on both grids differs only by the midpoint rule,
``(\Delta z_p^2 - \Delta z_c^2)/(24 H^2)`` (measured ``8.64\times10^{-6}``
against a predicted ``8.63\times10^{-6}`` for 125 m layers, ``H = 8.4``
km); a child surface 1 K warmer gives ``3.5\times10^{-3}`` and is refused
(default tolerance ``10^{-4}``). With ``W`` constant the measure *is* the
volume measure and the operators are the uniform ones bit for bit.

Two parent cell masses exist. The coordinate gives a covered parent
layer the summed mass of its children (what makes prolongation and
restriction exact); the parent simulation conserves its own
``W_p \Delta z_p``. Uncovered parent cells — the ones the composite
integral counts and refluxing corrects — use the parent's own mass, so
the composite ``\int W\phi\,dV`` is exactly what the two simulations
conserve. (The first implementation used the summed mass there too and
refluxing left a ``2.9\times10^{-12}`` drift: the midpoint-rule
difference times the interface flux. It is now ``3.6\times10^{-15}``.)

The sound-proof core has no ``q_v`` field, so a moist anelastic nest
carries ``q_v`` as a tracer of the field set
(`scalars = (:qv => (child_qv, parent_qv),)`); it is transferred in the
mass measure like every other member.

### Solid-aware restriction

A child containing IBM buildings must not hand the zero velocity inside
a building to the parent (Hellsten et al. 2021, §3.5). With `solid` (the
child's `SolidMask` or a Boolean `solid` array, such as the Berlin
case's) restriction averages fluid child cells only, and fluid child
faces only (both neighbouring cells fluid). A parent cell or face with no
fluid child keeps its value. The preserved quantity is the
fluid-weighted integral ``\sum_\text{fluid} W\phi\,\Delta V``. N-nest-e
(#263) builds its canopy-restricted anterpolation on this primitive.

### Measured (`test/era2_nesting_conservative.jl`)

| property | measured | bound tested |
| :-- | :-- | :-- |
| halo order, stretched parent (ratio 1.08 above the overlap), ``r \in \{2,3,4\}``, all locations, ``p = 2`` / ``3`` | 1.99 – 2.11 / 3.02 – 3.17 | ``> p - 0.45`` |
| child-top halo (stencil reaches the stretched layers), ``p = 2`` / ``3`` | 2.06 – 2.19 / 2.80 – 3.12 | ``> p - 0.6`` |
| ``\lvert\mathcal R\mathcal P f - f\rvert/\max\lvert f\rvert``, stretched parent | ``4.3\times10^{-16}`` | ``10^{-14}`` |
| overlap ``\int f`` / face-flux mismatch, stretched parent | ``1.5\times10^{-15}`` / ``3.1\times10^{-16}`` | ``10^{-13}`` |
| ``\int\bar\rho\theta\,dV`` / face mass flux, ``\bar\rho`` falling 30 % over 3 km | ``2.1\times10^{-15}`` / ``3.6\times10^{-16}`` | ``10^{-13}`` |
| negative control: volume-weighted restriction of the same data | ``3.8\times10^{-7}`` / ``1.1\times10^{-5}`` | ``> 10^3\times`` the above |
| ``\bar\rho \equiv`` const vs uniform operators | bitwise | ``==`` |
| composite ``\int\bar\rho\theta\,dV`` after mass-measure refluxing | ``3.6\times10^{-15}`` (drift without: ``1.4\times10^{-7}``) | ``10^{-13}`` |
| solid-aware: fluid-weighted ``\int\theta`` | ``7.7\times10^{-16}`` | ``10^{-13}`` |
| free stream through a nest on a stretched parent, 30 two-way steps | ``\lvert u - 2\rvert \le 1.3\times10^{-15}``, ``\lvert w\rvert \le 8.7\times10^{-16}`` | ``10^{-12}`` |

On the real Berlin `solid` field (OpenStreetMap buildings around
Alexanderplatz on the 1 km × 1 km × 400 m child, 64 × 64 × 64 cells,
refinement ``(2, 2, 4)``): 3165 of 262 144 child cells are solid; a
plain area mean would average solid faces into 577 ``u``, 560 ``v`` and
227 ``w`` parent faces, the solid-aware restriction into none; 24 parent
cells are entirely built over and keep their value.

For the Berlin parent of N-parent-e (#259) — uniform 25 m to 500 m, then
stretched by 1.08 (capped at 100 m) to 3 km, 53 levels — the 400 m child
top is the parent face at 400 m, the layers around it are 25, 25, 25 m,
``r_z = 4``, and the order-3 stencils of the child-top halo reach
450 m, inside the uniform region (the first stretched layer is
26.88 m thick).

## Cost for the Berlin child

Measured with a 192 × 192 × 120 parent (6 × 6 × 3 km, uniform 25 m in
the vertical, the region the child sits in) and a 1 × 1 km child at
refinement ``(2, 2, 4)``, order 3, moist with one extra tracer, on one
core of the development machine (minimum of 5 calls):

| child | plan memory (6 plans) | one scalar, halo / whole field | field set ``u,v,w,θ,q_v,``tracer, halo / whole | restriction of the set |
| :-- | :-- | :-- | :-- | :-- |
| 64 × 64 × 32 | 28.7 KiB | 0.92 / 2.86 ms | 21.9 / 33.3 ms | 0.48 ms |
| 64 × 64 × 64 (~400 m deep) | 33.7 KiB | 1.40 / 5.26 ms | 19.0 / 39.9 ms | 1.02 ms |

The field-set halo cost is dominated by filling the six parent fields'
halos (192 × 192 × 120 each, 2.49 ms per field, about 15 ms of the
~20 ms), not by the interpolation; a bare child
`step!` without physics costs 4.4 ms (32 levels) and 9.0 ms (64 levels)
for comparison. The plans are built once per (location, axis, order) and
cached on the `NestedGrid`.

## References

- Berger, M. J. & Colella, P. (1989). Local adaptive mesh refinement for
  shock hydrodynamics. *J. Comput. Phys.* 82, 64–84.
- Clark, T. L. & Farley, R. D. (1984). Severe downslope windstorm
  calculations in two and three spatial dimensions using anelastic
  interactive grid nesting. *J. Atmos. Sci.* 41, 329–350.
- Colella, P. & Woodward, P. R. (1984). The piecewise parabolic method
  (PPM) for gas-dynamical simulations. *J. Comput. Phys.* 54, 174–201.
- Hellsten, A. et al. (2021). A nested multi-scale system implemented in
  the large-eddy simulation model PALM model system 6.0. *Geosci. Model
  Dev.* 14, 3185–3214.
