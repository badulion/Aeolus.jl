# Smagorinsky-Lilly SGS closure

Aeolus closes the filtered momentum equation with the eddy-viscosity
Smagorinsky-Lilly model (see [LES filtering & closure](les_filtering.md)
for motivation). This page walks through the discrete implementation.

## The model

Local eddy viscosity is determined by the resolved strain rate:

```math
\nu_t = (C_s\,\Delta)^2\, |\bar S|, \qquad
|\bar S| = \sqrt{2\,\bar S_{ij}\,\bar S_{ij}}
```

where the strain-rate tensor is

```math
\bar S_{ij} = \frac12\!\left(\frac{\partial \bar u_i}{\partial x_j}
                           + \frac{\partial \bar u_j}{\partial x_i}\right)
```

and the filter width is

```math
\Delta = (\Delta x\,\Delta y\,\Delta z)^{1/3}.
```

For the scalar (potential temperature) the SGS diffusivity is
`ν_t / Prₜ`.

## Discretization on the C-grid

The six independent components of the symmetric tensor `S_{ij}` live at
different places on the staggered mesh:

  * **Diagonal** `S₁₁ = ∂u/∂x`, `S₂₂ = ∂v/∂y`, `S₃₃ = ∂w/∂z` live
    naturally at cell centers (they are 2-point face → cell differences
    of the corresponding velocity components — the same stencils used
    by the divergence operator).
  * **Off-diagonal** `S_{12}`, `S_{13}`, `S_{23}` live at cell *edges*
    (x-y, x-z, y-z edges respectively).

Aeolus computes `ν_t` at cell centers. This requires interpolating the
off-diagonal components to the cell center — a standard 4-point edge
average.

The kernel `_compute_smagorinsky_νt!` reads:

```julia
S11 = (u[i+1,j,k] - u[i,j,k]) / dx           # cell center
S22 = (v[i,j+1,k] - v[i,j,k]) / dy           # cell center
S33 = (w[i,j,k+1] - w[i,j,k]) / dz           # cell center

# ∂u/∂y at cell center: average two face-to-face differences in y,
# each interpolated in x to the cell center.
du_dy = ((u[i,j+1,k] + u[i+1,j+1,k]) -
         (u[i,j-1,k] + u[i+1,j-1,k])) / (4 dy)
dv_dx = ((v[i+1,j,k] + v[i+1,j+1,k]) -
         (v[i-1,j,k] + v[i-1,j+1,k])) / (4 dx)
S12 = 0.5 * (du_dy + dv_dx)

# ... similarly for S13 and S23 ...

|S|² = 2*(S11² + S22² + S33²) + 4*(S12² + S13² + S23²)
νt[i,j,k] = (C_s * Δ)^2 * sqrt(|S|²)
```

## Applying the SGS tendency to momentum

Once `νt` is known at cell centers, the SGS contribution to the
momentum equation is

```math
\frac{\partial u_i}{\partial t} \supset
\frac{\partial}{\partial x_j}\!\left(\nu_t \frac{\partial u_i}{\partial x_j}\right).
```

Note this is the **diagonal-only** or **eddy-viscosity flux form** — we
retain the term with `∂uᵢ/∂xⱼ` and drop the transposed piece
`∂uⱼ/∂xᵢ` from the full stress divergence. For a well-resolved LES with
modest `νₜ` variation, the omitted piece is small (and it disappears
exactly for incompressible flow when `νₜ` is constant). PALM makes the
same choice.

For u-momentum at `FaceX`:

```math
\frac{\partial}{\partial x}\!\left(\nu_t \frac{\partial u}{\partial x}\right)\bigg|_{i-1/2}
= \frac{\nu_{t,i}\,(u_{i+1}-u_i)/\Delta x - \nu_{t,i-1}\,(u_i-u_{i-1})/\Delta x}{\Delta x}
```

with `νₜ` already at cell centers. The y- and z-flux terms need `νₜ`
at x-y and x-z edges respectively — again a 4-point average.

## Applying the SGS tendency to a scalar

The scalar tendency is

```math
\frac{\partial \theta}{\partial t} \supset
\frac{\partial}{\partial x_j}\!\left(\frac{\nu_t}{Pr_t}
                                     \frac{\partial \theta}{\partial x_j}\right).
```

Because `θ` is cell-centered, its diffusive flux lives at faces. The
kernel `_add_eddy_diffusion_scalar!` interpolates `νₜ` linearly from
cell centers to each face.

## Choice of coefficients

  * `Cₛ = 0.17` for isotropic free turbulence (Lilly 1967). Reduced to
    ~0.1 near walls — but Aeolus currently has only periodic BCs, so
    the constant-`Cₛ` form is used.
  * `Prₜ = 1/3` is standard for atmospheric ABL (Deardorff 1972).
    Engineering flows often use `Prₜ ≈ 0.7 – 1`.

Both are constructor parameters on
[`BoussinesqSimulation`](@ref Aeolus.Simulations.BoussinesqSimulation).

## Limitations and planned upgrades

The current implementation is deliberately the *simplest* working
Smagorinsky-Lilly. Known deficiencies (all fixable in later phases):

  * No wall damping (Van Driest / Piomelli). Needed only when walls
    exist — Phase 1b.
  * No dynamic coefficient (Germano-Lilly procedure) — the model is
    known to be too dissipative in transitional / stably stratified
    regimes.
  * The **stability correction** for stably stratified flow
    (Deardorff / Lilly closure with a Richardson-number reduction) is
    not applied — it will matter when we do stably stratified boundary
    layers.
  * No **near-surface roughness** treatment (MOST) — Phase 1b.

## References

  * Smagorinsky, J. (1963), Mon. Wea. Rev. 91, 99.
  * Lilly, D. K. (1967), *The representation of small-scale turbulence
    in numerical simulation experiments*, IBM Sci. Comp. Symp.
  * Deardorff, J. W. (1972), *Numerical investigation of neutral and
    unstable planetary boundary layers*, J. Atmos. Sci. 29, 91.
  * Moin, P. & Kim, J. (1982), *Numerical investigation of turbulent
    channel flow*, JFM 118, 341.
  * Pope, S. B. (2000), *Turbulent Flows*, ch. 13.
