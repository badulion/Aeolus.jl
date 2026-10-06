# Stretched vertical grid

`BoussinesqSimulation` runs on a
[`StretchedGrid`](@ref Aeolus.StretchedGrids.StretchedGrid) as well as on a
uniform `RegularGrid` (#151). The grid is uniform in `x` and `y` and has an
arbitrary face distribution `z_f[k]` in `z`. Its first user is the parent
domain of the Berlin nesting epic (#246). The parent is 6 × 6 × 3 km, with
Δz = 25 m up to 500 m so that the ~400 m deep child lies in its uniform
part, and stretched above that.

## Metrics

All vertical metrics come from the grid, through
[`cell_thickness`](@ref Aeolus.StretchedGrids.cell_thickness) and
[`center_spacing`](@ref Aeolus.StretchedGrids.center_spacing):

| quantity | symbol | where it is used |
| :-- | :-- | :-- |
| cell thickness `z_f[k+1] − z_f[k]` | ``Δz_c[k]`` | height of a `u`, `v`, `θ`, `p` control volume; the face → centre difference (divergence) |
| centre spacing `z_c[k] − z_c[k−1]` | ``Δz_f[k]`` | height of a `w` control volume; the centre → face gradient |

## Operators

Every operator of the core is the finite-volume form on these control
volumes. On a `:uniform` StretchedGrid each one reduces algebraically to
its `RegularGrid` stencil. The test suite checks that the two meshes give
the same 20-step run to 1e-12.

* **Projection.** The pressure equation is ``DGφ = D\mathbf{u}^*``, with
  the grid's own divergence ``D`` (metric ``Δz_c``) and gradient ``G``
  (metric ``Δz_f``):

  ```math
  \frac{δ_x^2φ}{Δx^2} + \frac{δ_y^2φ}{Δy^2}
  + \frac{1}{Δz_c[k]}\left[\frac{φ_{k+1}-φ_k}{Δz_f[k+1]}
  - \frac{φ_k-φ_{k-1}}{Δz_f[k]}\right] = (D\mathbf u^*)_k .
  ```

  Because `x` and `y` are uniform, the operator is diagonalised
  horizontally by the real FFT (periodic sides) or the DCT-II (Neumann
  sides), using the exact eigenvalues of the 3-point stencil. What is
  left is one tridiagonal system per wavenumber, solved by the Thomas
  algorithm (Schumann & Sweet 1988, *J. Comput. Phys.* 75, 123). This is
  [`StretchedPoissonSolver`](@ref Aeolus.StretchedGrids.StretchedPoissonSolver),
  the default solver on a stretched grid. It is direct, so the projected
  velocity is solenoidal to round-off: max|∇·u| = 4e-17 s⁻¹ on the
  32 × 32 × 53 parent-like grid.

  The same operator is available to the multigrid solvers:
  `MultigridPoissonSolver(g::StretchedGrid)` and
  `MGPreconditionedCG(g::StretchedGrid)` multiply row `k` by
  ``s_k = Δz_c[k]/Δ``, which gives the symmetric variable-coefficient form
  `coeff_x = coeff_y = s_k`, `coeff_z = Δ/Δz_f[k]`, `rhs_scale = s_k` on a
  uniform grid of spacing `Δ`. The two solvers agree to 1e-8 relative.
  Multigrid is **not** the default, because the hierarchy coarsens only
  when all three dimensions halve, and 53 levels do not halve. Measured
  on the parent levels with a tolerance of 1e-12: the V-cycle stalls at a
  relative residual of 1.9e-6 after 500 cycles (32 × 32 × 53), and MG-CG
  needs 64 iterations and 5.5 s at 64 × 64 × 53, against 1.2 ms for the
  direct solve.

* **Momentum advection.** This is the divergence form on the staggered
  control volumes. The transported velocity is the arithmetic mean at the
  control-volume face. The transporting velocity is the volume flux
  through that face. For a `w` volume the side faces span half of the
  `u` cell below and half of the one above, so the transporting `u` there
  is ``(Δz_c[k-1]u_{k-1} + Δz_c[k]u_k)/(Δz_c[k-1]+Δz_c[k])``. With these
  fluxes, the volume-flux divergence of every momentum control volume is
  a weighted mean of the divergences of the pressure cells it overlaps.
  The operator is therefore skew-symmetric in the volume-weighted inner
  product, and conserves kinetic energy for a discretely solenoidal field
  (Morinishi et al. 1998, *J. Comput. Phys.* 143, 90; Vasilyev 2000,
  *J. Comput. Phys.* 157, 746). Measured: ``|dE/dt|/E < 10^{-12}`` on the
  parent grid.
* **Scalar advection and diffusion** are in flux form, with flux
  differences divided by ``Δz_c`` and gradients taken over ``Δz_f``. So
  ``Σϕ\,ΔV`` changes only through the boundary fluxes, and ``Σϕ^2ΔV`` is
  invariant under a solenoidal flow.
* **Molecular viscosity** on `w` uses the `w` control volume: height
  ``Δz_f[k]``, with neighbours ``Δz_c[k-1]`` and ``Δz_c[k]`` away.
  Applying the centre metrics to `w` instead is a first-order error. The
  manufactured-solution test caught it. The same fix applies to the
  terrain-following Laplacian of a z-face field.
* **Smagorinsky.** ``ν_t = (C_sΔ_k)^2|S|`` with the local Deardorff
  (1970) width ``Δ_k = (ΔxΔyΔz_c[k])^{1/3}``. The strain uses the one
  strain stencil (`Operators._strain_at`): ``∂w/∂z`` across the cell, and
  ``∂u/∂z``, ``∂v/∂z`` across the two neighbouring centres. On the parent
  grid, the 100 m cells get ``(100/25)^{2/3} = 2.52`` times the ``ν_t`` of
  the 25 m cells for the same shear.
* **Buoyancy** interpolates ``θ`` (and ``q_v``) to the faces with
  thickness weights, so a horizontally uniform stratification at rest
  stays at rest (|w| < 1e-12 after 20 steps).
* **Time step.** The vertical CFL number of a cell is ``|w|/Δz_c[k]``, so
  the thin near-surface cells set the limit, not the mean spacing. The
  diffusive and stress-form limits use the thinnest cell.

## Accuracy

Two manufactured solutions are used. In both, the geometric map holds
``r^N`` fixed, so the grid refines smoothly, and the stretching ratio is
1.08 on the coarsest grid.

* **Poisson.** ``φ = \cos(2πx/L)\cos(πz/H)`` converges at orders 1.93 and
  1.97 with periodic sides. ``φ = \cos(πx/L)\cos(πz/H)`` converges at 1.90
  and 1.95 with Neumann sides.
* **Navier–Stokes.** A forced two-mode cellular flow between free-slip
  walls is run through the full projected RK3 step to ``t = 0.1``. It
  converges at orders 2.01 and 1.99 (N = 16, 32, 64).

## Cost

One `step!` on one thread, with Smagorinsky, buoyancy and no-slip
bottom:

| grid | s/step | Poisson solve |
| :-- | :-- | :-- |
| 192 × 192 × 53, stretched (parent) | 0.225 s | 0.066 s |
| 192 × 192 × 120, uniform 25 m | 0.526 s | 0.197 s |

## Not covered

The following paths stay `RegularGrid`-only:

* the SGS **stress form** (`sgs_stress = :symmetric`) and **implicit
  diffusion** (`step_imex!`), which raise an `ArgumentError` that names
  the mesh;
* the dynamic Smagorinsky closures, whose constructors take a
  `RegularGrid`.

The prognostic TKE closure of `PrognosticSGS` runs on a stretched grid
since #259 (the SGS closure of the Berlin nesting parent; see the
[SGS TKE theory page](sgs_tke.md)), and with it the positive flux-limited
scalar advection `advect_positive!` with the centred reconstruction. The
other Era VI closures, which have their own stencils, have not been
tested on a stretched grid. Non-periodic lateral velocity boundaries are
not solenoidal on either mesh; that is the open-boundary work of #251.
