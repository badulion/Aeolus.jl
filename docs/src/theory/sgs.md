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
`κ_t = ν_t / Prₜ`, kept in its own field `sim.κt` (see [Closure
contract](#Closure-contract): other closures set `κ_t` independently).

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
same choice. It is the default of every core (`sgs_stress = :laplacian`);
the full stress form is `sgs_stress = :symmetric`, below, and the Berlin
case runs it.

For u-momentum at `FaceX`:

```math
\frac{\partial}{\partial x}\!\left(\nu_t \frac{\partial u}{\partial x}\right)\bigg|_{i-1/2}
= \frac{\nu_{t,i}\,(u_{i+1}-u_i)/\Delta x - \nu_{t,i-1}\,(u_i-u_{i-1})/\Delta x}{\Delta x}
```

with `νₜ` already at cell centers. The y- and z-flux terms need `νₜ`
at x-y and x-z edges respectively — again a 4-point average.

## Stress form vs Laplacian form

The SGS stress of an eddy-viscosity closure is `τᵢⱼ = −2νᵢⱼSᵢⱼ`, and its
divergence

```math
-\frac{\partial \tau_{ij}}{\partial x_j}
= \frac{\partial}{\partial x_j}\left[\nu_{ij}\left(\frac{\partial u_i}{\partial x_j}
  + \frac{\partial u_j}{\partial x_i}\right)\right]
```

differs from the Laplacian form above by the transposed term
`∂ⱼ(νᵢⱼ ∂uⱼ/∂xᵢ)`. For constant ν that term is `ν ∂ᵢ(∇·u) = 0`; where ν
varies (the canopy, the surface layer, the edge of a wake) it does not
vanish, and the Laplacian form then has three defects: it is not the
divergence of a symmetric stress (so it does not conserve angular
momentum), it gives a tendency for a solid-body rotation, and its
kinetic-energy tendency is not a negative-definite dissipation. The stress
form fixes all three. `add_sgs_stress_divergence!` (in `Aeolus.Operators`,
`src/operators/sgs_stress.jl`) implements it, and
`add_stress_divergence!(tu, tv, tw, τ)` is the generic operator for any
symmetric stress — P56-d's Reynolds-stress model passes its prognostic τ
there.

**Placement.** The stresses sit where the eddy-viscosity fluxes of the
Laplacian kernels already live: `τ11, τ22, τ33` at cell centres, `τ12` on
the xy-edges (where u-face `i` meets v-face `j`), `τ13` on the xz-edges,
`τ23` on the yz-edges. Each stress enters the two faces it separates with
opposite signs, so the operator conserves momentum exactly for any τ.

**Strain.** At each location the compact face difference: `Sᵢᵢ` is the
diagonal of `_velocity_gradient_at` (face to face through the cell), and
on the xy-edge `S12 = ½((u_{i,j} − u_{i,j−1})/Δy + (v_{i,j} − v_{i−1,j})/Δx)`
from the four velocities adjacent to it. The cell-centred off-diagonal
`Sᵢⱼ` of `_velocity_gradient_at` (two-cell centred differences, used for
`|S|` in the closures) is *not* used for the fluxes: interpolated to the
edges it would not commute with the discrete divergence, and the
constant-ν identity below would fail at truncation order.

**Edge interpolation.** ν is the cell value for the diagonals and the
4-cell average onto the edge for the off-diagonals, exactly as in
`add_eddy_viscosity_u!`, so the stress form is the Laplacian kernel plus
the transposed term, and the two agree to round-off when ν is constant.

**Anisotropy.** `ν11 = ν22 = ν12 = ν_h` and `ν33 = ν13 = ν23 = ν_v`: ν_h
for the horizontal pairs, ν_v for every pair involving z. This is the
convention of CM1 (Bryan & Fritsch 2002, `turbnmod`), which also matches
the split of horizontal and vertical mixing in Kosović (1997)'s
anisotropic model. The isotropic closure is `ν_h = ν_v`. Through the
closure contract the two come from `SGSFields.νh` and `SGSFields.νv`
(aliased for every closure so far; P55-e supplies the split).

**Discrete dissipation.** Summation by parts on a periodic domain gives the
kinetic-energy tendency exactly:

```math
\sum_{\rm faces} u_i\, \frac{\partial}{\partial x_j}(2\nu_{ij}S_{ij})\, \Delta V
= -\sum 2\nu_{ij} S_{ij} S_{ij}\, \Delta V \le 0,
```

with the diagonal terms summed over cells and each off-diagonal over its
edges, counted for `ij` and `ji` (`sgs_stress_dissipation`).

**IMEX split.** `step_imex!` with `sgs_stress = :symmetric` treats the
diagonal part `∂ⱼ(νᵢⱼ ∂uᵢ/∂xⱼ)` implicitly and the transposed part
explicitly. The implicit operator takes a per-direction coefficient per
velocity component (`ν + ν_h` on the faces of the 11, 22, 12 fluxes,
`ν + ν_v` on those of 33, 13, 23) with the explicit kernel's edge
interpolation, so it is identical to the explicit `part = :diagonal` and
stays symmetric (`I − γΔt L` is SPD for CG). The cross terms carry no
second derivative of a component along its own direction, so they are not
stiff; they vanish for constant ν on the projected stage values, which is
why the split is exact in that limit. With `:laplacian` the implicit
operator is unchanged (the co-located Phase-25 approximation). A pair
`(ν_h, ν_v)` is also accepted as the coefficient of
`solve_implicit_diffusion!` (ν_h on the x/y faces, ν_v on the z faces),
and a closure with a separate `κv` gets it for the scalars.

Measured (`test/era6_sgs_stress_form.jl`; spacing (15.625, 15.625, 6.25) m):

| test | result |
| :-- | :-- |
| projected random field, constant ν = 0.37 m²/s: `:symmetric` vs `:laplacian` | 2.2e-16 of max\|tendency\| |
| solid-body rotation Ω = 0.01 s⁻¹, ν = 0.5–5 m²/s: max\|t\| / (Ω ν_max/Δ²) | 0 (`:symmetric`), 2.33 (`:laplacian`) |
| random u, random ν_h ∈ [0,2], ν_v ∈ [0,0.2]: KE tendency vs −Σ2νSS ΔV | 2.8e-16 relative, ≤ 0 (4 seeds) |
| shear modes, ν_h/ν_v = 100: tendency vs −(ν_h k̂² + ν_v k̂_z²) | 2.4e-16, 2.4e-15 |
| same, time-stepped: Euler hλ = 0.05 × 40 / ARK2 hλ = 5 × 4 | 3.3e-15 / 9.5e-14 from the discrete amplification factor |
| Σu, Σv over 20 steps with Smagorinsky ν | 5.6e-16, 2.8e-15 relative |
| `add_stress_divergence!(sgs_stress_tensor!)` vs fused | bitwise equal |
| cost on 64 × 64 × 32 (momentum kernels) | 0.62 ms vs 0.44 ms Laplacian |

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

## Tensorial scalar diffusivity

P55-c (#181) generalises the scalar flux to a diffusivity tensor,
``\partial_t\phi \supset \partial_i(K_{ij}\,\partial_j\phi)``, through one
operator family in `Aeolus.Operators` (`src/operators/scalar_tensor.jl`).
P55-e (κ_h ≠ κ_v under stratification) and P56-d (the Daly–Harlow heat
flux ``K_{ij} = C\,(e/\varepsilon)\,\tau_{ij}``) reuse it.

**Diagonal.** `add_eddy_diffusion_scalar!(tϕ, ϕ, κh, κv)` puts κ_h on the
x and y faces and κ_v on the z faces (each the 2-point average of the
cell values), the tensor ``\mathrm{diag}(\kappa_h, \kappa_h, \kappa_v)``.
The isotropic three-argument form is the same kernel with κ_h ≡ κ_v, bit
for bit. The contract's default `add_sgs_scalar!` calls it with the
closure's `f.κh`, `f.κv`; before P55-c the explicit cores used `f.κh`
alone while `step_imex!` already used the pair, so a closure with a
separate κ_v was diffused differently by the two paths.

**General symmetric tensor: the triad form.** `add_tensor_diffusion_scalar!(tϕ, ϕ, K)`
takes the six components of K at cell centres
([`DiffusivityTensor`](@ref Aeolus.Operators.DiffusivityTensor)). The cross
terms need gradients that are not co-located with the flux they enter.
The discretisation follows Griffies et al. (1998). Each cell is split into
eight octants. Octant ``(a,b,c)`` pairs one x-face, one y-face and one
z-face gradient of the cell, ``g = (g_x, g_y, g_z)``. The operator is
defined as the gradient of a discrete functional:

```math
F[\phi] = -\tfrac12 \sum_{\text{cells}} \frac{\Delta V}{8} \sum_{\text{octants}}
          g^{\mathsf T} K_{\text{cell}}\, g,
\qquad \Delta V\, \partial_t \phi_n = \frac{\partial F}{\partial \phi_n}.
```

In flux form, the flux through a face is the mean of ``(Kg)_n`` over the
eight octants that touch it. Because K is per cell, the four octant terms
of a cell collapse to the average of its two cross gradients:

```math
F_x^{\,i+1/2} = \tfrac12\left(K_{11}^{i} + K_{11}^{i+1}\right)\delta_x\phi
 + \tfrac14 \sum_{c\in\{i,i+1\}}\left[K_{12}^{c}\,(g_{y,S}+g_{y,N})_c
                                     + K_{13}^{c}\,(g_{z,B}+g_{z,T})_c\right].
```

``F_y`` and ``F_z`` follow the same pattern. The tendency is
``\delta_x F_x + \delta_y F_y + \delta_z F_z``. Its diagonal part is exactly
the face-averaged diagonal kernel. Three properties follow from the
functional, hold to round-off for any K, and do not depend on resolution:

* **conservation**: ``\sum \Delta V\,\partial_t\phi = 0``, because F is
  invariant under a constant shift of ϕ;
* **down-gradient**: ``\sum \Delta V\,\phi\,\partial_t\phi = 2F \le 0`` when
  every cell's K is positive semi-definite (Griffies et al. 1998, Sec. 3);
* **symmetry** of the discrete operator.

For constant K, a plane wave ``\cos(\mathbf k\cdot\mathbf x)`` is an exact
eigenfunction. Its rate is
``K_{11}\hat k_x^2 + K_{22}\hat k_y^2 + K_{33}\hat k_z^2 + 2K_{12}s_x s_y + 2K_{13}s_x s_z + 2K_{23}s_y s_z``,
where ``\hat k = 2\sin(k\Delta/2)/\Delta`` and ``s = \sin(k\Delta)/\Delta``:
the diagonal terms see the compact second difference and the cross terms
the centred one.

The `bounded` keyword (from the scalar boundary conditions,
`bounded_directions(bcs)`) handles walls. A wall face's gradient is left
out of the cross terms. The wall flux keeps only its diagonal part from
the halo, as in the diagonal operator, which is zero for a no-flux halo.
With no-flux halos the three properties hold with walls as well.
Prescribed surface fluxes are added separately.

**Rotated isentropic tensor.** `isentropic_tensor!(K, θ, κh, κv; slope_max, grad_min)`
builds the Redi (1982) tensor, with κ_h along the isentropes and κ_v
across them. With ``\mathbf n = \nabla\theta/|\nabla\theta|``:

```math
K = \kappa_h(I - \mathbf n\mathbf n^{\mathsf T}) + \kappa_v\,\mathbf n\mathbf n^{\mathsf T}
  \;\to\; K = \kappa_h I - f\,(\kappa_h-\kappa_v)\,\mathbf n\mathbf n^{\mathsf T}.
```

Here ``f\in[0,1]`` blends the rotated tensor toward the isotropic κ_h:

* the small-slope taper (the slope clipping of Gerdes, Köberle &
  Willebrand 1991; see Danabasoglu & McWilliams 1995): with
  ``S = |\nabla_h\theta|/\partial_z\theta``, ``f = 1`` for ``S \le S_{max}``
  and ``f = (S_{max}/S)^2`` above it;
* the isotropic fallback ``f = 0`` where ``N^2 \propto \partial_z\theta \le 0``.
  This is the limit of the taper as S → ∞, so K is continuous across
  N² = 0. f also ramps linearly to 0 as ``|\nabla\theta|`` drops below
  `grad_min`.

∇θ is the centred gradient at the cell centre. The eigenvalues are
``\kappa_h, \kappa_h, \kappa_h - f(\kappa_h-\kappa_v)``, so K is positive
semi-definite and the triad operator dissipates variance. Where ∇θ is
uniform, the triad gradients of θ are the cell's ∇θ, which K annihilates
when κ_v = 0. Isentropic diffusion therefore leaves θ unchanged (the
"triad property"). With ``\mathbf n = \hat z``, K is
``\mathrm{diag}(\kappa_h,\kappa_h,\kappa_v)``.

**In the closure contract.** `sgs_scalar_tensor(c, f, sim)` (default
`nothing`) exposes a closure's full tensor. `IsentropicDiffusion(inner, mesh; slope_max, grad_min, diapycnal)`
runs `inner` for the coefficients, the momentum and the memory, builds K
from `sim.θ` in `sgs_coefficients!`, and applies it in `add_sgs_scalar!`.
κ_v is the closure's own `f.κv`, or `diapycnal·f.κh` when `diapycnal` is
given. **IMEX:** `step_imex!` solves the diagonal
``(\nu+K_{11}, \nu+K_{22}, \nu+K_{33})`` implicitly, each component on the
faces of its own direction (exactly the `part = :diagonal` of the explicit
operator). In a stable boundary layer κ_v sits there, which is the stiff
term. The off-diagonal part (`part = :offdiagonal`) is added explicitly on
θ and qv. K is frozen at the start of the step, the same lagged
linearisation as ν_t. The explicit cross terms are bounded by
``|K_{13}| \le \sqrt{K_{11}K_{33}}``, which the taper keeps at or below
``\kappa_h S_{max}/(1+S_{max}^2)``. They therefore do not tighten the
explicit advective limit the IMEX step runs at.

**Validation** (`test/era6_tensor_diffusion.jl`, Berlin spacing
15.625 × 15.625 × 6.25 m):

| check | result |
| :-- | :-- |
| diagonal, κ_h/κ_v = 10³: mode decay vs κ_h(k̂ₓ²+k̂_y²)+κ_v k̂_z² | operator 1.1e-15 (vertical mode alone 9.7e-16); Euler hλ = 0.05 × 40 2.2e-15; ARK2 hλ = 5 × 4 2.9e-14 |
| constant general K, plane-wave eigenvalue (cross terms 11 % of λ) | 4.9e-15 |
| random PSD K: ``|\sum t\,\Delta V| / \sum|t|\Delta V`` (periodic / no-flux walls) | 1.4e-17 / 7.5e-18; symmetry 3.9e-18 / 8.0e-18 |
| random PSD K, 200 explicit steps, no-flux walls | Σϕ² non-increasing every step (1.47e6 → 1.37e5); Σϕ dV drift 2.1e-17 |
| isentropic, slope 0.05, κ_v = 0, κ_h varying 10×: max ``|t_\theta|/(\kappa_h|\nabla\theta|/\Delta)`` | 2.2e-13 (isotropic κ_h: 0.43) |
| blob in the tilted stratification, κ_h/κ_v = 100: ΔM/(2κt) along / across | 0.9977 / 1.0067 (2 % tolerance) |
| random tilted θ (stable, tapered and unstable cells), 200 steps | Σϕ² non-increasing; Σϕ dV drift 2.5e-16 |
| n = ẑ vs diagonal operator | K 4.7e-16 m²/s, tendency 4.5e-16 |
| tensor closure under `step!` / `step_imex!` (ARK2, diag implicit + cross explicit) | 3.2e-15 / 8.2e-15 from the exact split amplification factor (dropping the cross part would give 0.00129 instead of 0.04702) |

## Stability correction

P55-e (#187) is the one stability-corrected closure of Era VI (the
"stratification-dependent anisotropic" closure of P55 merged with the
former "buoyancy-aware Smagorinsky" of P58). The constant-``C_s`` closure
mixes at the same rate whatever the stratification: it keeps mixing where
``Ri \gg 1/4``, damps internal gravity waves and keeps a stable boundary
layer artificially turbulent. `StabilityCorrected(base, mesh; scheme,
length_scales)` (`src/operators/sgs_stability.jl`) wraps any
eddy-viscosity closure (`StaticSmagorinsky`, `DynamicSmagorinsky`,
`FunctionSGS`, `ML.MLSGS`, …) and returns four coefficients:

```math
\nu_h = \nu_{base}\,\ell_h, \qquad
\nu_v = \nu_{base}\,\ell_v\,f_m(Ri), \qquad
\kappa_h = \frac{\nu_{base}\,\ell_h}{Pr_t}, \qquad
\kappa_v = \frac{\nu_{base}\,\ell_v\,f_h(Ri)}{Pr_t}.
```

* ``Ri = N^2/\max(|S|^2, S^2_{min})`` is the gradient Richardson number of
  `gradient_richardson!` with the strain ``|S|^2 = 2S_{ij}S_{ij}`` the
  Smagorinsky closure sees, and ``N^2`` from `buoyancy_frequency_squared!`
  (of ``\theta_v`` when a humidity is known; about the core's own
  reference, ``\theta_0`` on the Boussinesq core and ``\bar\theta(z)`` on
  the anelastic / pseudo-incompressible cores).
* ``f_m, f_h`` are the stability functions of the TurbulenceStability
  contract (P58-a; [theory](turbulence_stability.md)); nothing is redefined
  here. The default is Brown, Derbyshire & Mason (1994) (the LEM/MONC
  functions: ``(1 - Ri/Ri_c)^4`` with ``Ri_c = 1/4`` in stable air,
  enhanced ``\sqrt{1 - cRi}`` mixing in unstable air); `LillyCutoff()` is
  Lilly (1962) / Mason (1989). Both are 1 at ``Ri = 0``, so the correction
  leaves a neutral flow alone, and the neutral ``Pr_t`` is the base
  closure's.
* **Anisotropic length scales** (`length_scales = :anisotropic`, the
  default). The filter width of the horizontal coefficients is
  ``\Delta_h = \sqrt{\Delta x\,\Delta y}`` and of the vertical ones
  ``\Delta_v = \Delta z`` (Bryan & Fritsch 2002, CM1) instead of
  ``\Delta = (\Delta x\Delta y\Delta z)^{1/3}``. For ``\nu \propto \Delta^2``
  this is the factor ``\ell_h = (\Delta_h/\Delta)^2``,
  ``\ell_v = (\Delta_v/\Delta)^2`` — on the Berlin grid
  (``\Delta x/\Delta z = 2.5``) 1.842 and 0.2947. On cubic cells both are
  exactly 1.
* **Horizontal coefficients keep no stability factor.** ``\nu_h`` and
  ``\kappa_h`` are the base values (with ``\ell_h``): the correction
  targets the vertical exchange that the stratification suppresses. The
  horizontal part of the SGS damping of a gravity wave therefore remains
  (see the numbers below).
* **Dynamic bases.** A coefficient from the Germano identity already sees
  the damping of the resolved motion by the stratification, so a second
  ``f_m`` would count it twice. For `dynamic_sgs_coefficient(base) = true`
  the momentum gets no factor (``\nu_v = \nu_{base}\ell_v``) and only the
  scalar is corrected, through ``Pr_t(Ri) = Pr_t f_m/f_h``:
  ``\kappa_v = \nu_{base}\ell_v (f_h/f_m)/Pr_t`` (0 where ``f_m = 0``).
* **Not every closure is wrapped.** A closure with its own buoyancy term
  (the anisotropic minimum-dissipation model, P55-d) or its own length
  scale (prognostic TKE, P56-b) already accounts for the stratification;
  the constructor accepts eddy-viscosity closures only.

The four fields reach every core through the contract defaults: the
stress form ``\partial_j(2\nu_{ij}S_{ij})`` with ``\nu_h, \nu_v`` (P55-b,
`sgs_stress = :symmetric`; the Laplacian form uses ``\nu_h`` only) and the
diagonal scalar operator with ``\kappa_h, \kappa_v`` (P55-c), explicit or —
under `step_imex!` — in the implicit operator. The scalar tendency keeps
the base closure's order of operations,
``(1/Pr_t)\,\partial_j(a_j\,\partial_j\phi)``, so with ``N^2 = 0`` and
``\ell_h = \ell_v = 1`` the wrapper is the base closure **bit for bit**.
`IsentropicDiffusion(StabilityCorrected(base, mesh), mesh)` gives the
rotated tensor the corrected ``\kappa_v`` across the isentropes.

Measured (`test/era6_stratified_sgs.jl`):

| test | result |
| :-- | :-- |
| ``N^2 = 0``: coefficients, tendencies (both stress forms), `step!`, `step_rk3!`, `step_imex!`, `step_adaptive!`, anelastic `step!` | `==` the base closure (Static and Dynamic Smagorinsky) |
| ``u = Sz``, ``\theta = \theta_0 + \Gamma z``, ``Ri \in \{-0.5, 0, 0.1, 0.2, 0.25, 0.5, 5\}`` (Brown) | worst relative error of ``\nu_h, \nu_v, \kappa_h, \kappa_v`` 6.1e-14; ``\nu_v = \kappa_v = 0`` for ``Ri > Ri_c``; ``\nu_h`` independent of ``Ri`` to ≤ 1e-14 |
| same, `LillyCutoff` | to 1e-12; at ``Ri = Ri_c`` exactly the discrete ``Ri`` is 0.25 − O(1e-13) and ``f = \sqrt{\delta/Ri_c}`` ≈ 1e-6 (the √ singularity of P58-a) |
| anelastic core, ``N^2 = (g/\bar\theta(z))\,\Gamma`` | to 1e-12 |
| standing internal wave, ``N = 0.01`` s⁻¹, 32 × 4 × 32, ``\Delta x/\Delta z = 2``, steepness 0.05, min ``Ri`` = 256 | ``\omega`` within 0.46 % of ``N k_x/|k|``; energy decay over 5 periods 0.62 % more than ``C_s = 0`` (uncorrected: 0.82 %; ``\nu_v \equiv 0``, the rest is the retained ``\nu_h, \kappa_h``) |
| breaking wave, steepness 0.3 / 1.5 | ``\nu_v \equiv 0`` for 3 periods / ``\nu_v > 0`` from the first step; wave energy −23.4 % in 3 periods (−5.5 % without SGS) |
| Σθ dV, no-flux walls, 200 steps of the corrected scalar flux | drift 4.4e-15; θ variance non-increasing at every step |
| surface-cooled column, ``Ri \gg Ri_c`` | θ-flux divergence exactly 0 in the 22 interior levels (uncorrected: 1.1e-3 K/s) |
| the correction's θ change, `step_imex!` vs `step!` | relative mismatch 3.8e-3 (dt = 0.1 s), 1.5e-3 (dt = 0.05 s) |

Two expectations of the issue proved wrong, not the code: the breaking
wave loses 23.4 %, not "more than 30 %", of its energy in 3 periods at
this resolution with ``C_s = 0.17`` (the test asserts > 20 % and more than
3× the resolved loss); and the Berlin smoke check "``\nu_v = \nu_h`` where
``Ri \le 0``" ignores both the anisotropic length scales and the
unstable-side enhancement of Brown et al. (``f_m > 1``), so the check is
``\nu_v \ge \nu_h\,\ell_v/\ell_h`` there (and ``\le`` where ``Ri \ge 0``).

**Cost** (64 × 64 × 32, one `compute_sgs!`, single thread, machine under
load): 4.5 ms against 1.0 ms for `StaticSmagorinsky` on a random field;
4.8 ms (minimum; median 8.5 ms) against 1.1 ms on the Berlin flagship state
after 300 s, where a dynamics step takes about 98 ms and the closure is
evaluated twice per step (CFL estimate and step), so about +8 %. The extra
work is the strain of ``Ri`` (`gradient_richardson!` evaluates it a second
time, 1.2 ms), the stability functions per cell (1.4–1.8 ms; the Float64
``x^4`` and the branches of the Brown functions) and two more halo fills.
The bare closure (`sgs_stability = :none` in the Berlin case) reproduces the
pre-P55-e run bit for bit (100 nominal smoke steps, 59 adaptive steps:
every prognostic field, the tracers and the surface temperatures `==`).

## Choice of coefficients
## Choice of coefficients

  * `Cₛ = 0.17` for isotropic free turbulence (Lilly 1967). Reduced to
    ~0.1 near walls; the constant-`Cₛ` form is used everywhere (no
    near-wall damping), with the wall stress from a
    [wall model](wall_models.md).
  * `Prₜ = 1/3` is standard for atmospheric ABL (Deardorff 1972).
    Engineering flows often use `Prₜ ≈ 0.7 – 1`.

Both are constructor parameters on
[`BoussinesqSimulation`](@ref Aeolus.Simulations.BoussinesqSimulation).

## Dynamic coefficient (Germano–Lilly)

[`compute_dynamic_smagorinsky_νt!`](@ref Aeolus.Operators.compute_dynamic_smagorinsky_νt!)
computes ``C_s`` from the resolved flow instead of prescribing it
(Germano et al. 1991, Lilly 1992). The resolved stress at a coarser
*test* filter ``\widetilde{\cdot}`` (a 3 × 3 × 3 box of cell-centred
values) is known exactly, and the Germano identity equates it to the
difference of the model at the two filter levels:

```math
L_{ij} = \widetilde{u_i u_j} - \tilde u_i \tilde u_j = C_s^2 M_{ij}, \qquad
M_{ij} = 2Δ^2\left(\widetilde{|S| S_{ij}} - α^2 |\tilde S|\, \tilde S_{ij}\right),
```

with ``α = \hat Δ/Δ``. The least-squares solution, averaged over each
horizontal level (the homogeneous directions), is
``C_s^2(z) = ⟨L^d_{ij} M_{ij}⟩ / ⟨M_{ij} M_{ij}⟩``, clipped to
``[0, 0.5^2]`` (no backscatter). `averaging = :local` uses the pointwise
ratio instead.

**Filter-width ratio.** The discrete box of three cells has second moment
``2Δ^2/3``; combined with the implicit grid filter (a top hat of width
``Δ``, second moment ``Δ^2/12``) the effective test-level width is
``\sqrt{Δ^2 + 12 \cdot 2Δ^2/3} = 3Δ``, so `filter_ratio = 3` is the
default. With 2 (the value often quoted for the trapezoidal filter) the
coefficient is too large for this filter:

| field (32³, triply periodic, ``u_{rms} = 1``) | ``α = 3`` | ``α = 2`` |
| :-- | --: | --: |
| random phases, ``E(k) ∝ k^{-5/3}`` | 0.012 | — |
| same field after ``t = 1`` (≈ 1 turnover) of LES | 0.129 | 0.220 |
| after ``t = 3`` | 0.154 | 0.267 |

The initial field is drawn by
[`isotropic_random_field!`](@ref Aeolus.Operators.isotropic_random_field!)
(Phase 54). Up to #168 it came from a test-local generator (continuous
projection, amplitude ``∝ k^{-5/3}`` per mode); that field gave 0.018,
0.129 / 0.222 and 0.153 / 0.265. The difference is the random
realisation, not the closure: with the old field the refactored
dynamic model reproduces the old numbers bit for bit.

Lilly's inertial-range value is 0.17. The random-phase field has the
right spectrum but no energy transfer to small scales, and the dynamic
procedure — which measures that transfer — correctly gives
``C_s ≈ 0``; it switches off in laminar shear too (``L_{ij}M_{ij} = 0``
for a linear velocity profile), where the constant-coefficient model
still dissipates. These are the tests in `test/era2_sgs_numerics.jl`.

**Using it.** `step!(sim, dt; sgs = DynamicSmagorinsky(ws))` replaces
the built-in closure (the eddy-viscosity terms are added where the
built-in ones are); it is bit for bit the pre-contract
`eddy_viscosity! = (νt, u, v, w) -> compute_dynamic_smagorinsky_νt!(νt,
u, v, w; workspace = ws)`, which still works. `ML.MLSGS` closures plug in
the same way. The Berlin case selects it with `physics[:sgs] = :dynamic`,
on any dynamical core.

## Closure contract

Era VI (#167) defines the interface between an SGS closure and the
dynamical cores **once**, in `src/operators/sgs_closures.jl`, so that
closures that need more than the velocity — buoyancy-aware mixing, a
prognostic TKE, Reynolds-stress transport, a Lagrangian-averaged dynamic
coefficient — plug into every core the same way. A closure is an
[`AbstractSGSClosure`](@ref Aeolus.Operators.AbstractSGSClosure) passed as
`sgs = closure` to `step!`, `step_rk3!`, `step_imex!`, `step_adaptive!`
and the anelastic / pseudo-incompressible `step!`.

| generic | contract |
| :-- | :-- |
| `sgs_coefficients!(f, c, sim)` | fill the eddy viscosity and diffusivity `f = SGSFields(νh, νv, κh, κv)` (isotropic: `νv === νh`, `κv === κh`; by default `sim.νt`, `sim.κt`) from the current state — `u, v, w`, `θ`, `qv` with filled halos, the mesh, `θ₀`, `g` — and the closure's committed memory. **Pure.** |
| `advance_sgs!(c, sim, dt)` | commit the memory over an accepted step of length `dt`, after the projection. Default no-op. |
| `sgs_state(c)`, `restore_sgs_state!(c, s)` | snapshot / restore the memory around a step that may be rejected. Default no-op. |
| `add_sgs_momentum!(tu, tv, tw, c, f, sim)` | the stress divergence; default the Laplacian eddy-viscosity form with `νh` |
| `add_sgs_scalar!(tϕ, ϕ, c, f, sim)` | the scalar flux divergence; default `∂ⱼ(κh ∂ⱼϕ)` |

**Why `sgs_coefficients!` must be pure.** The cores evaluate the
coefficients more often than once per step: once per RK3 substage, once
for the adaptive stepper's CFL limit, and again on every retried try of
a rejected step. Before the contract the closure hook `f(νt, u, v, w)`
was called at each of these, so a closure with memory would have
advanced it two to four times per step and never rolled it back. Under
the contract the memory changes only in `advance_sgs!`, which every core
calls exactly once per accepted step (RK3: once with `dt` after the
third substage, not three times), and `step_adaptive!` brackets each try
with `sgs_state` / `restore_sgs_state!`, so after a rejection the memory
is bit for bit that of a run that took the shorter step directly.

**κ_t is a field.** The simulations carry `κt` next to `νt`. The θ
tendency is formed with it, the CFL limit is
``\Delta t_{sgs} = C_{diff}\, Δ^2_{harm} / \max(ν_{t,\max}, κ_{t,\max})``
(rather than ``ν_{t,\max} \max(1, 1/Pr_t)``), and the IMEX implicit
operator freezes ``ν + κ_t`` for the scalars. The Smagorinsky-type
closures ([`AbstractEddyViscosityClosure`](@ref
Aeolus.Operators.AbstractEddyViscosityClosure): `StaticSmagorinsky`,
`DynamicSmagorinsky`, `ML.MLSGS`, and `FunctionSGS`, which wraps a
legacy `f(νt, u, v, w)`) set ``κ_t = ν_t/Pr_t``; they apply it as
``(1/Pr_t)\,∂_j(ν_t ∂_jθ)`` through the same kernel, the order of
operations of the pre-contract code, so the default run is unchanged bit
for bit (face averages of ``ν_t/Pr_t`` are not bitwise ``1/Pr_t`` times
face averages of ``ν_t``). A closure with its own ``κ_t`` (a stability
function, a TKE closure) gets the three-argument
[`add_eddy_diffusion_scalar!`](@ref Aeolus.Operators.add_eddy_diffusion_scalar!)
by default. qv and the passive tracers keep their current treatment
(P56-a, #177).

**One gradient stencil.** `Operators._velocity_gradient_at` returns the
full resolved gradient ``g_{ij} = ∂u_i/∂x_j`` at a cell centre (diagonal:
the face-to-face difference; off-diagonal: the centred difference over
two cells of the component averaged to the centre); the strain rate of
both Smagorinsky closures is built from it, and the Era VI closures that
need the gradient (AMD / Vreman, σ, the gradient Richardson number, TKE
shear production) read the same tensor.

**What was measured** (`test/era6_sgs_closure.jl`):

| property | result |
| :-- | :-- |
| `StaticSmagorinsky(C_s)`, a legacy `f(νt, u, v, w)` vs the built-in `C_s` run, 50 steps of 16³ decaying turbulence, `step!` / `step_rk3!` / `step_imex!` | identical `u, v, w, θ` (`==`) |
| `DynamicSmagorinsky(ws)` vs `eddy_viscosity! = (…) -> compute_dynamic_smagorinsky_νt!(…)` | identical (`==`), same `cs2` |
| commits over 6 steps: Euler / RK3 / IMEX / adaptive | 6 / 6 (18 coefficient evaluations) / 6 / 6 |
| adaptive, one forced retry | memory `(1, [dt/2])` = the run that took `dt/2` directly, state bitwise equal |
| `θ = sin(2πz/L_z)`, constant `κ₀ ≠ ν_t/Pr_t`: amplitude vs the discrete amplification factor, explicit (100 steps) / ARK2 (20 steps, 25× the explicit limit) | 1.8e-15 / 4.4e-16 relative |
| `κ_t = 5ν_t`: `dt_sgs` | `C_diff Δ²_harm / κ_{t,max}` exactly |
| random positive `κ_t`, no-flux walls, 100 steps: ``Σθ`` | conserved to 0 (Euler), 6e-14 (ARK2) relative |
| linear velocity field on (15.625, 15.625, 6.25) m: `_velocity_gradient_at` | 5.6e-17 s⁻¹ from the prescribed tensor |
| anelastic vs Boussinesq core, `DynamicSmagorinsky`, same velocity | identical `ν_t`, `κ_t` (`==`) |

In the Berlin case the SGS slot's providers are closure objects passed as
`sgs`, so `:dynamic` and `:mlsgs` now also run on the sound-proof cores
(see [The Berlin case](../design/berlin_case.md#Numerics-slots)).

## Decaying grid turbulence (Comte-Bellot and Corrsin)

The Era VI benchmark for SGS closures (Phase 54) is the wind-tunnel decay
of Comte-Bellot & Corrsin (1971, CBC): grid mesh ``M = 5.08`` cm, mean
speed ``U_0 = 10`` m/s, spectra measured at ``tU_0/M = 42, 98, 171``
(their table 3, in `test/data/cbc1971_table3.txt`; the tabulated spectra
integrate to ``\tfrac32 u'^2`` of their table 4 within 4.2, 1.3 and
0.3 %). Following Meneveau, Lund & Cabot (1996) and Kang, Chester &
Meneveau (2003) the flow is simulated in a frame moving with ``U_0``: a
triply periodic box of side ``11M`` at ``ν = 1.5×10^{-5}`` m²/s,
initialised with the station-42 spectrum
([`isotropic_random_field!`](@ref Aeolus.Operators.isotropic_random_field!),
then two cycles of evolve-and-[`rescale`](@ref Aeolus.Operators.rescale_to_spectrum!)
so the cascade is established), run to the later stations with SSP-RK3.
The LES is compared with CBC *filtered to the grid cut-off*: TKE is the
shell sum over ``k ≤ k_c = (N/2)Δk``. CBC is an experiment, not DNS; the
harness (`cbc_benchmark(closure; N)` in `test/cbc_benchmark_driver.jl`)
accepts any spectrum function, so a tabulated DNS spectrum can replace it.

Measured (three random seeds; `test/era6_cbc_decay.jl` at 32³,
`examples/validation/cbc_decay.jl` at 64³):

| ``N^3`` | closure | ``K_{98}/K_{42}`` vs CBC | ``K_{171}/K_{42}`` vs CBC | RMS ``\log_{10}E``, ``k ≤ \tfrac23 k_c`` (98 / 171) | ``E(k_c)``/CBC (98 / 171) |
| :-- | :-- | --: | --: | --: | --: |
| 32³ | dynamic (plane)     | +1.3 % | −15.7 % | 0.085 / 0.099 | 1.23 / 1.00 |
| 32³ | Smagorinsky 0.17    | +2.5 % | −14.2 % | 0.088 / 0.097 | 1.27 / 1.08 |
| 32³ | no model            | +78.3 % | +121.1 % | 0.216 / 0.316 | 5.25 / 10.4 |
| 32³ | σ-model 1.35 (#184)  | +6.6 % | −11.1 % | 0.096 / 0.105 | 1.43 / 1.31 |
| 64³ | dynamic (plane)     | −13.4 % | −23.5 % | 0.062 / 0.118 | 0.72 / 0.61 |
| 64³ | Smagorinsky 0.17    | −14.2 % | −24.3 % | 0.065 / 0.124 | 0.67 / 0.52 |
| 64³ | no model            | +32.7 % | +15.7 % | 0.226 / 0.346 | 5.66 / 8.02 |

CI cost at 32³: 14 s for the three closures × three seeds.

Without a model the energy piles up at the cut-off (5–10 × CBC), which
is what the test uses to show it can fail. With either eddy-viscosity
closure the first interval is right at 32³ but the second decays too
fast: the deficit is in the energy-containing shells (LES/CBC 0.65–0.71
at ``k ≈ 0.2–0.3`` cm⁻¹ at station 171), where the measured large scales
barely decay and the ``11M`` box holds only a few dozen modes. At 64³
the closures are also too dissipative at the small scales. The
proposed ±10 % on both TKE ratios (a figure from spectral-code LES) is
therefore not met; the test asserts ±20 % at station 171 and keeps ±10 %
as a broken test, so a closure that reaches it (Lagrangian dynamic,
AMD, …: one line each in the test) shows up. Starting from the random-
phase field without the rescale cycles instead gives +21 % / −3 % at
32³ (no transfer during the first turnover).

## σ-model

The σ-model of Nicoud, Baya Toda, Cabrit, Bose & Lee (2011) is a
closure of the [contract](#Closure-contract),
[`SigmaModel`](@ref Aeolus.Operators.SigmaModel)`(C_σ = 1.35)`, P59-b
(#184). It builds the eddy viscosity from the singular values
``σ_1 ≥ σ_2 ≥ σ_3 ≥ 0`` of the resolved velocity gradient
``g_{ij} = ∂u_i/∂x_j``:

```math
ν_t = (C_σ Δ)^2\, D_σ, \qquad
D_σ = \frac{σ_3\,(σ_1 - σ_2)\,(σ_2 - σ_3)}{σ_1^2}, \qquad
Δ = (Δx\,Δy\,Δz)^{1/3},
```

with ``κ_t = ν_t/Pr_t``. The design goal is an operator that is zero
wherever the flow cannot be turbulent. Smagorinsky's ``|S|`` sees only
the strain, so it is on in any laminar shear. ``D_σ`` vanishes in these
cases (Nicoud et al. 2011, Sec. II):

* ``σ_3 = 0`` whenever ``g`` has rank ≤ 2: any two-dimensional flow, pure
  shear, solid-body rotation;
* ``σ_1 = σ_2`` or ``σ_2 = σ_3``: isotropic expansion or contraction,
  axisymmetric strain.

At a no-slip wall ``u, v = O(z)`` and ``w = O(z^2)``, so ``σ_1 = O(1)``,
``σ_2 = O(z)``, ``σ_3 = O(z^2)`` and ``D_σ = O(z^3)``. This is the
near-wall behaviour of the true SGS dissipation, and it comes without a
damping function. ``C_σ = 1.35`` is Nicoud et al.'s value from isotropic
turbulence.

**Computing the singular values.** ``σ_i^2`` are the eigenvalues of
``G = g^Tg``. They are found in closed form, with no LAPACK call per
cell, from the trigonometric solution of the characteristic cubic (Smith
1961; the form Nicoud et al. take from Hasan, Basri & Kopp 2010). The
cubic is written in the deviator ``G' = G - \tfrac13 \mathrm{tr}G\,I``,
with ``α_1 = \mathrm{tr}(G'^2)/6`` and ``α_2 = \det G'/2``, so neither is
formed by cancellation. ``D_σ`` is a product of eigenvalue *gaps*, and
near a double eigenvalue the cubic gives a gap only to ``\sqrt ε``,
because the arccos argument is near ±1. A plain implementation therefore
returns ``ν_t ≈ 10^{-8}(C_σΔ)^2|g|`` for axisymmetric strain instead of
zero. Two steps fix this:

1. Where the smaller gap is below 1 % of the spread, the simple
   eigenvalue is deflated. Its eigenvector is the largest cross product
   of two rows of ``G − λI``, and the close pair is recomputed from the
   2 × 2 compression of ``G`` on the orthogonal complement, whose gap is
   accurate to ``ε\|G\|`` (Kopp 2008).
2. The smallest singular value is ``σ_3 = |\det g|/(σ_1σ_2)``, which is
   exactly zero for a rank-2 ``g`` rather than the square root of a
   round-off eigenvalue.

``σ_1 − σ_2`` is formed as ``(λ_1 − λ_2)/(σ_1 + σ_2)``. Round-off can
still leave ``σ_3`` a few ulp above ``σ_2`` when the two coincide; the
operator is clipped at zero there. The gradient is
`_velocity_gradient_at`, the one stencil of the contract, so there is no
second stencil.

**What was measured** (`test/era6_sigma_sgs.jl`; KH in
`test/era6_transition_kh.jl`; CBC in `test/era6_cbc_decay.jl`):

| property | σ-model | Smagorinsky |
| :-- | :-- | :-- |
| 16³ linear fields, ``\max ν_t/(Δ^2\|g\|)``: uniform shear / solid-body rotation / isotropic expansion / axisymmetric strain (also rotated off the grid axes) | 0 / 0 / 1.7e-30 / 5.0e-16 (rotated: 3.3e-18 / 1.5e-30 / 1.1e-15) | 0.0289 / 0 / 0.0409 / 0.0409 (``C_s = 0.17``) |
| 2-D flow (16³, ``w = 0``, no ``z`` dependence) | exactly 0 | max 1.44e-3 |
| closed form vs `svdvals`, 1e4 random ``g`` over six decades | ``\max\|Δσ\|/σ_1`` = 2.5e-14 | — |
| near-degenerate ``Q\,\mathrm{diag}(a, a(1+δ), b)\,Q^T``, δ = 1e-1 … 1e-8 | ``\max\|ΔD_σ\|/σ_1 ≤`` 2.1e-15 | — |
| clip: degenerate ``g`` (two equal singular values), 1e4 | raw ``D_σ < 0`` in 431, ``\|D_σ\| ≤`` 3.8e-16 ``σ_1``, clipped to 0 | — |
| ``d\log ν_t/d\log z``, first 8 of 64 cells, Taylor wall field | 2.955 | 0.0025 |
| laminar Couette, Re = 100, ``N_z = 32``, from rest | wall stress ``νU/h`` to 2.8e-14, ``ν_t ≡ 0`` | stress excess ``(C_sΔ)^2(U/h)/ν`` = 4.5156e-2, measured 4.5156e-2 |
| decaying HIT 32³ (``ν = 10^{-4}``), exponent ``n`` of ``K ∝ t^{-n}`` on ``t ∈ [1, 3]``, seeds 1 / 2 | 0.5823 / 0.4746 | 0.5840 / 0.4848 |
| CBC 32³ (3 seeds): ``K_{98}/K_{42}``, ``K_{171}/K_{42}`` vs CBC; RMS ``\log_{10}E`` (98 / 171); ``E(k_c)``/CBC | +6.6 %, −11.1 %; 0.096 / 0.105; 1.43 / 1.31 | +2.5 %, −14.2 %; 0.088 / 0.097; 1.27 / 1.08 |
| KH 64×1×128 growth rate (no SGS 0.1820) | 0.1820 (``ν_t ≡ 0``: the layer is 2-D) | 0.1769 (−2.82 %, ``C_s = 0.1``) |
| kernel cost, 64×64×32 | 6.7 ms | 0.85 ms |

The issue expected Smagorinsky to be nonzero in all five laminar cases.
That expectation is wrong for solid-body rotation, which has no strain,
so Smagorinsky is zero there too. The test asserts that.

**Berlin.** The `:sgs` slot's provider `:sigma`
(`examples/urban/couplings/sgs_sigma.jl`) is off by default; flipping
the default from Smagorinsky is the maintainer's call. After 100 steps
of the `:smoke` window, the mean ``ν_t`` is 2.7e-2 m²/s in the canopy,
against 2.4e-2 for Smagorinsky on the same flow. Above the sponge start
it is 5.3e-4 against 2.0e-2 m²/s. The Transition indicator finds that
free stream laminar (γ = 0). It is not *exactly* parallel, though: weak
3-D disturbances from the canopy reach it, below the γ threshold, so
``σ_3 ≠ 0`` and ``ν_t`` is small rather than zero. For an exactly
parallel profile on the Berlin mesh it is zero to the bit
(`test/berlin_numerics.jl`). The smoke step costs 16.4 ms against
15.8 ms with Smagorinsky.

Those free-stream numbers were measured with a sponge that relaxed the
wind towards rest, which sheared the free stream (U fell from 0.52 to
0.07 U_g over the top four levels). With the geostrophic sponge of the
large-scale forcing (#171) the free stream above the sponge start is the
geostrophic wind and shear free (U = 0.98–1.00 U_g). Measured on that
stack (P59-e, #195): σ's mean ``ν_t`` there is 4.2e-3 m²/s against
2.3e-3 for Smagorinsky, which now has no mean shear to see; both are
below 1e-2 ν (``ν`` = 2 m²/s). The plane-normalised γ marks the weak
disturbances turbulent (0.96: the indicator is scale-free), the
volume-normalised γ laminar (0.02). The test asserts the latter and
``ν_t/ν < 10^{-2}`` there.

## Limitations and planned upgrades

The current implementation is deliberately the *simplest* working
Smagorinsky-Lilly. Known deficiencies (all fixable in later phases):

  * No wall damping (Van Driest / Piomelli) for Smagorinsky. Needed only
    when walls exist. The [σ-model](#σ-model) needs none: its ``ν_t``
    falls off as ``z^3`` at a no-slip wall (measured slope 2.955).
  * The constant coefficient is too dissipative in transitional / stably
    stratified regimes. The dynamic coefficient and the σ-model, which is
    zero in laminar shear and in 2-D flow, are the alternatives; in
    stable stratification the Richardson-number reduction of
    [Stability correction](#Stability-correction) (P55-e) applies, on by
    default in the Berlin case; its validation on GABLS1 and the
    stratified channel (P55-f) is in
    [Stable boundary layers](stable_boundary_layer.md).
  * The σ-model costs about 8 times the Smagorinsky kernel (6.7 against
    0.85 ms on 64×64×32). It is zero only in *exactly* laminar
    (parallel, 2-D, pure-strain) flow; a laminar but weakly 3-D free
    stream gets a small ``ν_t`` (Berlin, shear-free free stream above the
    sponge start: 2e-3 ν). On its own it is not stability-aware; wrap it
    in `StabilityCorrected` for that.
  * **Relaminarised stable regions** (P59-e, #195; see
    [Transition → Relaminarisation](transition.md#Relaminarisation-in-strongly-stable-stratification)).
    In decaying turbulence on a held laminar shear at ``Ri = 1`` the
    collapsed state keeps Smagorinsky at max ``ν_t/ν`` = 0.118 (its
    ``(C_sΔ)^2|S|`` sees the laminar shear), the σ-model's horizontal
    viscosity at 0.017 ν (the residual internal waves are 3-D, so
    ``σ_3 ≠ 0``), and the σ-model + stability correction's vertical
    viscosity at exactly 0 (``Ri_g > Ri_c``, ``f_m = 0`` in Brown et al.
    1994). Only the vertical viscosity is stability-corrected; the
    horizontal one is not. Coarse wall-resolved LES of the cooled open
    channel relaminarises at a much smaller ``h/L`` than DNS, with any of
    the closures and without one (see the transition page): the critical
    stability of a resolved near-wall cycle is resolution-limited.
  * The SGS closure itself has no wall treatment: at a free-slip z-wall
    its flux through the wall face is zero. The wall stress — smooth
    (Spalding), rough (log law) or Monin–Obukhov — is supplied by the
    wall model, `apply_wall_stress!` of `Aeolus.WallModels` (Phase 57,
    #169; see [Wall models](wall_models.md)); the Berlin ground uses it.

## References

  * Smagorinsky, J. (1963), Mon. Wea. Rev. 91, 99.
  * Lilly, D. K. (1967), *The representation of small-scale turbulence
    in numerical simulation experiments*, IBM Sci. Comp. Symp.
  * Deardorff, J. W. (1972), *Numerical investigation of neutral and
    unstable planetary boundary layers*, J. Atmos. Sci. 29, 91.
  * Moin, P. & Kim, J. (1982), *Numerical investigation of turbulent
    channel flow*, JFM 118, 341.
  * Pope, S. B. (2000), *Turbulent Flows*, ch. 13.
  * Germano, M., Piomelli, U., Moin, P. & Cabot, W. H. (1991), *A dynamic
    subgrid-scale eddy viscosity model*, Phys. Fluids A 3, 1760.
  * Lilly, D. K. (1992), *A proposed modification of the Germano subgrid-scale
    closure method*, Phys. Fluids A 4, 633.
  * Lund, T. S. (1997), *On the use of discrete filters for large eddy
    simulation*, CTR Annual Research Briefs, 83.
  * Comte-Bellot, G. & Corrsin, S. (1971), *Simple Eulerian time
    correlation of full- and narrow-band velocity signals in
    grid-generated, 'isotropic' turbulence*, JFM 48, 273.
  * Kang, H. S., Chester, S. & Meneveau, C. (2003), *Decaying turbulence
    in an active-grid-generated flow and comparisons with large-eddy
    simulation*, JFM 480, 129.
  * Meneveau, C., Lund, T. S. & Cabot, W. H. (1996), *A Lagrangian dynamic
    subgrid-scale model of turbulence*, J. Fluid Mech. 319, 353 (the
    stateful closure the commit contract is designed for).
  * Bryan, G. H. & Fritsch, J. M. (2002), *A benchmark simulation for moist
    nonhydrostatic numerical models*, Mon. Wea. Rev. 130, 2917 (CM1; the
    ν_h / ν_v convention of the stress form and the anisotropic length
    scales Δh = √(dx dy), Δv = dz of `StabilityCorrected`).
  * Lilly, D. K. (1962), *On the numerical simulation of buoyant
    convection*, Tellus 14, 148 (the Richardson-number reduction).
  * Mason, P. J. (1989), *Large-eddy simulation of the convective
    atmospheric boundary layer*, J. Atmos. Sci. 46, 1492.
  * Brown, A. R., Derbyshire, S. H. & Mason, P. J. (1994), *Large-eddy
    simulation of stable atmospheric boundary layers with a revised
    stochastic subgrid model*, Q. J. R. Meteorol. Soc. 120, 1485 (the
    default stability functions of `StabilityCorrected`).
  * Kosović, B. (1997), *Subgrid-scale modelling for the large-eddy
    simulation of high-Reynolds-number boundary layers*, J. Fluid Mech.
    336, 151.
  * Redi, M. H. (1982), *Oceanic isopycnal mixing by coordinate rotation*,
    J. Phys. Oceanogr. 12, 1154 (the rotated diffusivity tensor).
  * Gerdes, R., Köberle, C. & Willebrand, J. (1991), *The influence of
    numerical advection schemes on the results of ocean general circulation
    models*, Clim. Dyn. 5, 211 (slope clipping).
  * Danabasoglu, G. & McWilliams, J. C. (1995), *Sensitivity of the global
    ocean circulation to parameterizations of mesoscale tracer transports*,
    J. Climate 8, 2967 (tapering of the rotated tensor).
  * Griffies, S. M., Gnanadesikan, A., Pacanowski, R. C., Larichev, V. D.,
    Dukowicz, J. K. & Smith, R. D. (1998), *Isoneutral diffusion in a
    z-coordinate ocean model*, J. Phys. Oceanogr. 28, 805 (the triad
    discretisation and its variance property).
  * Deardorff, J. W. (1970), *A numerical study of three-dimensional
    turbulent channel flow at large Reynolds numbers*, J. Fluid Mech. 41,
    453 (the stress form on a staggered grid).
  * Deardorff, J. W. (1980), *Stratocumulus-capped mixed layers derived
    from a three-dimensional model*, Boundary-Layer Meteorol. 18, 495
    (prognostic SGS TKE).
  * Nicoud, F., Baya Toda, H., Cabrit, O., Bose, S. & Lee, J. (2011),
    *Using singular values to build a subgrid-scale model for large eddy
    simulations*, Phys. Fluids 23, 085106 (the σ-model).
  * Hasan, K. M., Basri, A. H. & Kopp, J. (2010), as cited by Nicoud et
    al. (2011) for the closed-form eigenvalues; Smith, O. K. (1961),
    *Eigenvalues of a symmetric 3 × 3 matrix*, Commun. ACM 4, 168.
  * Kopp, J. (2008), *Efficient numerical diagonalization of Hermitian
    3 × 3 matrices*, Int. J. Mod. Phys. C 19, 523.
