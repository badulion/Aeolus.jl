# Staggered discretization: the Arakawa C-grid

Aeolus discretizes the Boussinesq equations on a **staggered
Arakawa-C grid**. Pressure and scalars live at cell centers; each
velocity component lives at the corresponding face:

```text
        ┌───── FaceX(u) ─────┬───── FaceX(u) ─────┐
        │                     │                     │
        │                     │                     │
   FaceY(v)   CellCenter    FaceY(v)   CellCenter   FaceY(v)
        │      p, θ           │      p, θ           │
        │                     │                     │
        └─── FaceX(u) ──────┴──── FaceX(u) ──────┘

  (2D slice; the 3rd axis has w on FaceZ)
```

The reason this is the *right* choice for incompressible LES is deep,
worth spelling out.

## Why staggered?

Consider a *collocated* layout where `u, v, w, p` all live at the same
cell center. Discretize the pressure gradient with centered differences:

```math
\left.\frac{\partial p}{\partial x}\right|_i \approx \frac{p_{i+1} - p_{i-1}}{2\Delta x}.
```

The stencil skips cell `i` — the pressure at the point where we're
computing the momentum tendency plays no role. The result is a *checker-
board* null mode: alternating pressures `p = (+1, -1, +1, -1, …)` give
zero discrete gradient. Momentum cannot see them, but they exist and
grow. Collocated schemes need either *Rhie-Chow interpolation* or
*artificial dissipation* to kill them.

The staggered layout eliminates the problem structurally:

```math
\left.\frac{\partial p}{\partial x}\right|_{i-1/2} = \frac{p_i - p_{i-1}}{\Delta x}
```

is a compact 2-point stencil that has *no* null mode. Discrete gradient
and discrete divergence become each other's transposes:

```math
G^\top = -D
```

which is the discrete analogue of the identity `⟨p, ∇·u⟩ = -⟨∇p, u⟩`
that underlies the continuum incompressibility. This exact adjointness
is what makes the [FFT Poisson solver](poisson.md) recover a truly
divergence-free velocity to machine precision.

## Coordinate convention in Aeolus

For a [`RegularGrid`](@ref Aeolus.Grids.RegularGrid) with `Nx × Ny ×
Nz` interior cells and halo width `(Hx, Hy, Hz)`:

  * Cell `(i, j, k)` occupies `[(i-1)Δx, iΔx] × [(j-1)Δy, jΔy] × [(k-1)Δz, kΔz]`.
  * Its center is at `((i-½)Δx, (j-½)Δy, (k-½)Δz)`.
  * Its west face (index `i` on `FaceX`) is at `x = (i-1)Δx`.
  * All storage arrays have shape `(Nx+2Hx, Ny+2Hy, Nz+2Hz)`. Interior
    indices are `Hx+1 : Hx+Nx`, and similarly for the other axes.

The east face of cell `Nx` is identified with the west face of cell 1
under periodic wrap. So a `FaceX` field stores `Nx` values, one per
cell's west face; no extra storage for the wrap-around face.

## Discrete stencils

Multiple dispatch picks the correct stencil from the [`Location`](@ref
Aeolus.Grids.Location) parameters. There are three families of
first-derivative stencils:

### Cell-center → face (gradient)

Used for pressure gradient, buoyancy flux, etc.

```math
\left.\frac{\partial \phi}{\partial x}\right|_{i-1/2,j,k}
    = \frac{\phi_{i,j,k} - \phi_{i-1,j,k}}{\Delta x}
```

Compact, 2-point. Second-order accurate.

Aeolus method: `∂x(dϕdx::Field{FaceX}, ϕ::Field{CellCenter})`.

### Face → cell-center (divergence contribution)

```math
\left.\frac{\partial u}{\partial x}\right|_{i,j,k}
    = \frac{u_{i+1,j,k} - u_{i,j,k}}{\Delta x}
```

Also 2-point. When combined with the analogous `v`- and `w`- differences
we get the discrete divergence at cell centers:

```math
(\nabla\cdot u)_{i,j,k} = \frac{u_{i+1}-u_i}{\Delta x} + \frac{v_{j+1}-v_j}{\Delta y} + \frac{w_{k+1}-w_k}{\Delta z}
```

Aeolus method: `∂x(divu::Field{CellCenter}, u::Field{FaceX})` — and
`divergence(out, u, v, w)` combines them.

### Cell → cell (2nd-order centered)

Kept for diagnostics on cell-centered fields; not used in the Boussinesq
core:

```math
\left.\frac{\partial \phi}{\partial x}\right|_{i,j,k}
    = \frac{\phi_{i+1,j,k} - \phi_{i-1,j,k}}{2\Delta x}
```

Aeolus method: `∂x(dϕdx::Field{CellCenter}, ϕ::Field{CellCenter})`.

## The discrete Laplacian

For any field at *any* single location the 2nd-order centered Laplacian
is the same 7-point stencil:

```math
(\nabla^2 \phi)_{i,j,k}
= \frac{\phi_{i+1}-2\phi_i+\phi_{i-1}}{\Delta x^2}
+ \frac{\phi_{j+1}-2\phi_j+\phi_{j-1}}{\Delta y^2}
+ \frac{\phi_{k+1}-2\phi_k+\phi_{k-1}}{\Delta z^2}.
```

It appears with three different physical meanings:

  * `∇²u_i` — viscous diffusion of the `i`-th velocity component (in
    the constant-`ν` limit).
  * `∇²θ` — molecular scalar diffusion.
  * `∇²p` — the discrete pressure Poisson operator.

The Aeolus kernel `_laplacian!` is generic over `Location`.

## Discrete divergence of discrete gradient

A crucial identity: the divergence of the gradient (both discrete) is
*exactly* the 7-point Laplacian. Starting from

```math
G_x p := (p_i - p_{i-1})/\Delta x  \text{ at west face of cell }i,
```

```math
D_x u := (u_{i+1} - u_i)/\Delta x  \text{ at cell }i,
```

```math
D_x G_x p \;=\; \frac{(p_{i+1}-p_i)/\Delta x - (p_i-p_{i-1})/\Delta x}{\Delta x}
           \;=\; \frac{p_{i+1}-2p_i+p_{i-1}}{\Delta x^2}.
```

This means: if we solve `∇²φ = ∇·u*` using the *same* discrete
Laplacian, the corrected velocity `u = u* - ∇φ` satisfies `∇·u = 0`
exactly. Not "to leading order" — *exactly*, to machine precision.
See [Fractional-step projection](fractional_step.md).

## References

  * Arakawa, A. & Lamb, V. R. (1977), *Computational design of the basic
    dynamical processes of the UCLA general circulation model*, Meth.
    Comput. Phys. 17.
  * Harlow, F. H. & Welch, J. E. (1965), *Numerical calculation of
    time-dependent viscous incompressible flow of fluid with free
    surface*, Phys. Fluids 8, 2182 — the original staggered scheme.
  * Ferziger, J. H. & Perić, M. (2020), *Computational Methods for
    Fluid Dynamics*, ch. 7.
