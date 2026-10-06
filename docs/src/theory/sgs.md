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

**Explicit stability of the stress form.** The explicit operator `L` is
symmetric and negative semi-definite, with `−⟨u, Lu⟩ = Σ 2νᵢⱼSᵢⱼSᵢⱼ ΔV`.
Forward Euler is therefore stable for `Δt ρ(L) ≤ 2`, where `ρ(L)` is the
largest Rayleigh quotient. To bound it, write each diagonal strain as one
difference, `(u_{i+1} − u_i)²/Δx² ≤ 2(u_{i+1}² + u_i²)/Δx²`, and split each
off-diagonal pair with `(a + b)² ≤ 2a² + 2b²`. Every face value enters two
differences per direction, which gives

```math
\rho(L) \le 8\max\Bigl(\nu_h^{\max}(\Delta x^{-2} + \Delta y^{-2}) + \nu_v^{\max}\Delta z^{-2},\;
              \nu_v^{\max}(\Delta x^{-2} + \Delta y^{-2} + \Delta z^{-2})\Bigr).
```

The first entry is the u and v rows. The second is the w row, which sees only ν_v.
For `ν_h = ν_v = ν` this is `8ν ΣΔ⁻²`, twice the radius `4ν ΣΔ⁻²` of the
7-point Laplacian. The factor 2 is the 2 in the normal stress
`τᵢᵢ = −2ν∂uᵢ/∂xᵢ`, and the bound is sharp. With constant ν, `L = ν(∇² + ∇∇·)`
on the grid, and on the irrotational checkerboard mode `u = ∇φ` both terms
give `−4ν ΣΔ⁻²`. Power iteration measures `ρ = 8ν ΣΔ⁻²` to 1e-6.

The factor applies to the unprojected operator. The cores project every
predictor, `uⁿ⁺¹ = P(uⁿ + Δt Luⁿ) = (I + Δt PLP)uⁿ`. For solenoidal periodic
`u`, `Σ 2S:S = Σ|∇u|²`, so `ρ(PLP) ≤ 4ν_max ΣΔ⁻²`, the old Laplacian bound.
Power iteration confirms this for every ν field tested (table below). Even so,
`cfl_diagnostics` and `adaptive_dt` use the unprojected bound with
`sgs_stress = :symmetric`: `νt_cfl = stress_form_cfl_viscosity(ν_h^max, ν_v^max, mesh)`
in place of `νt_max` in `C_diffusive Δ²_harm / (ν + νt_cfl)` and in the
SGS-only limit. The reason is that the projected bound needs an exact,
orthogonal projection of a periodic field. Several parts of the code do not
have one: the immersed-boundary masking of the Berlin run, an iterative
pressure solve stopped at a tolerance, wall boundaries, and the anelastic
ρ-weighted projection. Only the unprojected bound holds for every velocity
field. `:laplacian` keeps `νt_cfl = νt_max`, so its limits are unchanged bit
for bit.

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
| ρ(L) / 4ν_max ΣΔ⁻², 12 × 12 × 8, power iteration: ν constant / log-uniform 0.01–1 / checkerboard 0.01‖1 / Gaussian blob 100× | 2.000 / 1.159 / 1.010 / 1.426 (old limit at C = ½: Δt ρ = 4.00 / 2.32 / 2.02 / 2.85 > 2, unstable; new limit: Δt ρ ≤ 2) |
| ρ(PLP) / 4ν_max ΣΔ⁻², same fields | 1.000 / 0.555 / 0.505 / 0.706 (stable at the old limit) |
| 20 Euler steps at C = ½ from the leading mode, constant ν | growth 1 (new limit), 3.5e9 (old limit), 1 (old limit, projected) |
| anisotropic ν_h ∈ [0,1], ν_v ∈ [0,0.05]: ρ(L) vs bound | 0.0488 ≤ 0.0757 s⁻¹ (ν_cfl = 0.56 vs 2 max ν = 2.00) |

## Applying the SGS tendency to a scalar

The scalar tendency is

```math
\frac{\partial \theta}{\partial t} \supset
\frac{\partial}{\partial x_j}\!\left(\frac{\nu_t}{Pr_t}
                                     \frac{\partial \theta}{\partial x_j}\right).
```

Because `θ` is cell-centered, its diffusive flux lives at faces. The
kernel `_add_eddy_diffusion_scalar!` interpolates `νₜ` linearly from
cell centers to each face. The same tendency, with the same coefficient,
is applied to `qv` and to every tracer the Berlin case transports (see
*Every transported scalar uses κ_t* below).

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
  the constructor accepts eddy-viscosity closures only. This is by
  design, not a gap: the TKE closure also has no ``\ell_h, \ell_v``
  anisotropy (one ``\Delta = (\Delta x\Delta y\Delta z)^{1/3}``, as in
  Deardorff 1980 and PALM), so on the Berlin grid its neutral equilibrium
  mixes ``1/\ell_v = 3.39\times`` more vertically and
  ``\ell_h = 1.84\times`` less horizontally than this wrapper around a
  Smagorinsky closure of the same ``C_s``
  ([details](sgs_tke.md#The-model)); the rotated tensor of
  `IsentropicDiffusion` is not offered for it either.

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
| same field after ``t = 1`` (≈ 1 turnover) of LES | 0.128 | 0.219 |
| after ``t = 3`` | 0.155 | 0.268 |

The initial field is drawn by
[`isotropic_random_field!`](@ref Aeolus.Operators.isotropic_random_field!)
(Phase 54). Up to #168 it came from a test-local generator (continuous
projection, amplitude ``∝ k^{-5/3}`` per mode); that field gave 0.018,
0.129 / 0.222 and 0.153 / 0.265. The difference is the random
realisation, not the closure: with the old field the refactored
dynamic model reproduces the old numbers bit for bit. (With the
initialiser's first, centre-averaged normalisation, before the fix
described under the CBC benchmark below, the same seed gave 0.012,
0.129 / 0.220 and 0.154 / 0.267.)

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

## Lagrangian dynamic averaging (Meneveau, Lund & Cabot 1996)

The plane average of the Germano identity needs horizontally homogeneous
statistics, which an urban canopy does not have. Before P54-b (#173)
the plane average also ran over the solid cells of the immersed
buildings. [`LagrangianDynamicSmagorinsky`](@ref Aeolus.Operators.LagrangianDynamicSmagorinsky)
averages the identity backwards in time along fluid pathlines instead
(MLC96), so the coefficient is local and needs no homogeneous direction.
The averages obey

```math
\mathcal I_{LM}^{n+1}(\mathbf x) = ε\,[L^d_{ij}M_{ij}]^{n+1}(\mathbf x)
  + (1-ε)\,\mathcal I_{LM}^{n}(\mathbf x - \mathbf u^n Δt),
\qquad
\mathcal I_{MM}^{n+1}(\mathbf x) = ε\,[M_{ij}M_{ij}]^{n+1}(\mathbf x)
  + (1-ε)\,\mathcal I_{MM}^{n}(\mathbf x - \mathbf u^n Δt),
```

```math
ε = \frac{Δt/T}{1+Δt/T}, \qquad
T = θ\,Δ\,(\mathcal I_{LM}\mathcal I_{MM})^{-1/8},\ θ = 1.5, \qquad
C_s^2 = \mathcal I_{LM}/\mathcal I_{MM}.
```

This is the exponentially weighted pathline average of MLC96 eqs.
2.10–2.13, in the time discretisation of their eqs. 3.1–3.3. The details:

* The upstream values come from trilinear interpolation. Periodic
  directions wrap, and in a bounded direction the departure point is
  clamped to the boundary cell.
* ``\mathcal I_{LM}`` is clipped at 0.
* ``T`` uses the upstream memories. Where ``\mathcal I_{LM}`` has been
  clipped to 0, ``T`` is evaluated with ``\mathcal I_{LM} ≥
  10^{-8}\mathcal I_{MM}``, so the point can recover; with ``T = ∞`` it
  would freeze. Where ``\mathcal I_{MM} = 0``, ``ε = 1``.
* Initialisation (MLC96 §3): ``\mathcal I_{MM} = M_{ij}M_{ij}`` and
  ``\mathcal I_{LM} = C_0^2\mathcal I_{MM}`` with ``C_0 = 0.16``.
* ``C_s^2`` is clipped to ``[0, 0.5^2]``. No damping function is applied
  anywhere.

**One Germano kernel.** ``L^d_{ij}M_{ij}`` and ``M_{ij}M_{ij}`` come from
`_germano_identity!`. That kernel also computes the plane and local
averages of
[`compute_dynamic_smagorinsky_νt!`](@ref Aeolus.Operators.compute_dynamic_smagorinsky_νt!),
and its test filter is the stencil of
[`box_filter!`](@ref Aeolus.Operators.box_filter!). The refactor is bit
for bit: LM, MM, ``C_s^2`` and ``ν_t`` on the evolved field below are
`==` the pre-refactor values for both averagings and both filter ratios.

**Contract.** The memory advances only in `advance_sgs!`, once per
accepted step, from the state after the step. The departure point uses
the cell-centred velocity committed with the previous memory,
``\mathbf u^n``. `sgs_coefficients!` is pure: it takes ``C_s^2`` from the
committed memory and ``|S|`` from the current state. So the three RK3
substages and the CFL estimate see one coefficient, and between commits a
coefficient call costs only a strain evaluation. `sgs_state` copies the
memory, and a rejected adaptive step is rolled back bit for bit.

**Solids.** `mask` is a cell mask, e.g. `IBM.SolidMask.mask`. Its halos
are filled from the boundary conditions. The plane / local
`DynamicSmagorinsky` take the same mask.

* ``ν_t = 0`` and the memories are 0 in solid cells.
* The dynamic procedure reads the velocity with the no-slip value 0 on
  every face that touches a solid.
* The test filter runs over fluid neighbours only, renormalised by their
  number.
* A departure point whose nearest cell is solid takes the value of the
  cell itself.
* Solid corners of the trilinear stencil are dropped and the remaining
  weights renormalised.

Nothing stored inside an obstacle reaches a fluid coefficient: test 6
below sets every solid face to NaN and to 1e30 and gets the same bits.

Measured (`test/era6_lagrangian_sgs.jl`):

| test | result |
| :-- | :-- |
| relaxation to a frozen source, ``u = 0``, fixed ``T`` (40 steps) | ``\big\lvert\lvert I^n - S\rvert - (1-ε)^n\lvert I^0-S\rvert\big\rvert ≤ 1.4×10^{-15}`` |
| pathline transport, ``ε = 0``, 32³ periodic, CFL 0.4, 50 steps | domain sum to 1.4e-14, centroid moves ``\mathbf U t`` to 3.7e-13 m, max 0.911 → 0.171 (never grows) |
| laminar shear ``u = 3z`` | ``L^d_{ij}M_{ij} = 0`` exactly; ``C_s^2/C_0^2`` decays monotonically to 1.4e-8 after ``St = 45`` |
| isotropic, 32³ (field of the table above), ``t = 1`` | mean ``C_s`` 0.1228 against the plane 0.1291 (−4.8 %) |
| same, ``t = 3`` | 0.1500 against 0.1539 (−2.5 %) |
| 200 steps with the closure driving the flow | ``\mathcal I_{LM} ≥ 0``, ``\min \mathcal I_{MM} = 1.33``, ``ν_t ≥ 0``, clip fraction 0, ``\mathcal I_{LM} = 0`` in 0.59 % of cells |
| wall-mounted cube, 32×32×16 periodic channel, 200 steps | ``ν_t = 0`` in solids, no NaN/Inf, ``C_s^2 ∈ [0, 0.25]`` next to the cube (mean ``C_s`` 0.098 there, 0.152 over all fluid) |

As in MLC96 fig. 6, the Lagrangian coefficient follows the plane value in
isotropic turbulence. In laminar shear it does not drop to 0 at once, as
the plane value does. It relaxes there, with ``T`` growing as
``\mathcal I_{LM} → 0``. ``C_s^2`` and ``ν_t`` therefore reach 0 only as
``t → ∞``; the test asserts ``< 10^{-6}`` of the initial value. In the
CBC decay below the Lagrangian model is slightly closer to the
measurements than the plane model.

In the Berlin case it is the default averaging of the `:dynamic` SGS
provider. There, `CaseConfig(:smoke)` after 100 fixed steps of 2 s gives
these mean ``C_s`` over air cells:

| averaging | canopy | free stream | max ``ν_t`` (m²/s) |
| :-- | --: | --: | --: |
| Lagrangian | 0.037 | 0.033 | 0.38 |
| plane (solids now excluded) | 0.032 | 0.018 | 0.037 |

There is no clipping in either run.

## Scale-dependent Lagrangian dynamic model (Bou-Zeid, Meneveau and Parlange 2005)

The scale-invariant dynamic procedure assumes that the coefficient is the
same at the grid scale ``Δ`` and at the test-filter scale ``αΔ``. Near a
wall, where the filter widths are comparable to the distance to it, the
coefficient falls toward the grid scale, so the procedure underestimates
``C_s`` there and the mean shear comes out too large or too small
depending on the closure (Porté-Agel, Meneveau & Parlange 2000). Static
Smagorinsky overshoots (``Φ_m`` up to 1.45-1.65 at 24³ with ``C_s =
0.17``, P54-d), and in the half channel below LASI undershoots (``Φ_m ≈
0.7`` in the log layer, mean ``C_s`` 0.04-0.06 near the wall). The
classical fix is a damping function; Era VI uses none.

`LagrangianDynamicSmagorinsky(mesh; scale_dependent = true)` (P54-e,
#186) is the scale-dependent Lagrangian model (LASD) of Bou-Zeid,
Meneveau & Parlange (2005), as an option of the same type. It adds a
second test filter, the 5 × 5 × 5 box. Its second moment is ``2Δ^2``, so
with the convention that makes the 3-cell box ``α = 3`` (implicit grid
filter plus the discrete box, ``\sqrt{Δ^2 + 12σ^2}``) its width is
``α_2 = 5``. It also adds two memories transported along the same
pathlines, with the same ``ε`` and ``T``:

```math
Q_{ij} = \widehat{u_i u_j} - \hat u_i \hat u_j, \qquad
N_{ij} = 2Δ^2\left(\widehat{|S| S_{ij}} - α_2^2 |\hat S| \hat S_{ij}\right),
```

```math
\mathcal I_{QN}^{n+1} = ε\,[Q^d_{ij}N_{ij}]^{n+1}
  + (1-ε)\,\mathcal I_{QN}^{n}(\mathbf x - \mathbf u^n Δt),
\qquad
\mathcal I_{NN}^{n+1} = ε\,[N_{ij}N_{ij}]^{n+1}
  + (1-ε)\,\mathcal I_{NN}^{n}(\mathbf x - \mathbf u^n Δt).
```

``C^2_{αΔ} = \mathcal I_{LM}/\mathcal I_{MM}`` and ``C^2_{α_2Δ} =
\mathcal I_{QN}/\mathcal I_{NN}`` are the coefficients at the two
test-filter scales. With a power law ``C^2(aΔ) = C^2_Δ a^φ`` between them,

```math
φ = \frac{\ln(C^2_{α_2Δ}/C^2_{αΔ})}{\ln(α_2/α)}, \qquad
β = 2^φ \in [1/8,\ 8], \qquad
C^2_Δ = C^2_{αΔ}\,α^{-φ}.
```

For Bou-Zeid's filters (``α = 2``, ``α_2 = 4``) this is their ``β =
C^2_{4Δ}/C^2_{2Δ}`` and ``C^2_Δ = C^2_{2Δ}/β``. In general ``β`` is the
coefficient ratio per doubling of the filter width. The lower clip
``β ≥ 1/8`` is Bou-Zeid's. The upper clip, 8, is ours: it is symmetric
in ``φ`` and keeps the diagnostic finite. The details:

* The box is separable and summed plane by plane. In a periodic direction
  it wraps. In a bounded direction it stays in the interior, and the
  points beyond are dropped (renormalised). The 3-cell first filter reads
  the halo cell, but at a free-slip wall that cell holds the mirrored,
  reversed shear, and the 5-cell box would reach it two cells from the
  wall. The first version did read it; the second level then had
  ``β ≈ 1`` and a ``C_s`` dip (0.066 against 0.13-0.16 around it).
* With a solid mask the box runs over fluid cells only, and the memories
  are 0 in solids.
* Where ``C^2_{αΔ} = 0`` the coefficient is 0 and ``β`` is stored as 1.
  Where only ``C^2_{α_2Δ} = 0``, ``β = 1/8``.
* Initialisation: ``\mathcal I_{QN} = C_0^2 \mathcal I_{NN}``, so ``β = 1``.
* `beta = 1` (a fixed ``β``) reproduces LASI bit for bit (test 2).
* `cadence = N`: the Germano identities, which are the expensive part,
  are evaluated every ``N`` commits. In between, the memories are only
  transported (``ε = 0``), and the next evaluation relaxes over the
  elapsed time. This is common practice in Lagrangian dynamic codes.
  ``N = 1`` is the exact model, and the counters are part of
  `sgs_state`.

**No wall damping.** No damping function is applied anywhere. The
near-wall coefficient comes from the scale dependence, and the wall
stress from the wall model (`WallModels`, #169, with the P57-b matching
input, #183).

Measured (`test/era6_lasd.jl`):

| test | result |
| :-- | :-- |
| separable 5-cell filter against a direct 5³ sum (periodic and bounded, with and without mask) | relative difference < 1e-12 |
| laminar shear ``u = 3z`` | ``Q^d_{ij}N_{ij} = 0`` to 1e-12 of ``N_{ij}N_{ij}`` |
| prescribed power law ``C^2(aΔ) = C^2_Δ a^φ``, ``\lvert φ\rvert < 3`` | ``C^2_Δ`` and ``β = 2^φ`` recovered to 1e-12 (``α = 3, 5`` and ``2, 4``) |
| isotropic 32³, ``t = 3`` (the P54-b protocol) | ``β`` geometric mean 1.058 (seeds 1-3: 1.058 / 1.028 / 1.030), median 1.107, arithmetic mean 1.335 |
| same, ``C_s`` against LASI | median pointwise ratio 0.921; domain mean 0.1657 against 0.1500 (+10.5 %; seeds 2, 3: +13.6, +14.2 %) |
| ``β ≡ 1`` | LASI bit for bit (u, ν_t, ``C_s^2``, ``\mathcal I_{LM}``) over 30 RK3 steps |
| half channel 24³, 10-20 ``H/u_*`` (below) | ``Φ_m ∈ [0.934, 1.151]`` over ``0.05 ≤ z/H ≤ 0.3`` |
| 500 steps, isotropic and wall-mounted cube | ``ν_t ≥ 0``, ``β ≥ 1/8``, no NaN, ``ν_t = 0`` in solids |

Isotropic turbulence is scale invariant, and the mean exponent agrees:
``⟨φ⟩ = 0.04-0.08``, ``2^{⟨φ⟩} = 1.03-1.06``. Pointwise, though, ``φ`` is a
log-ratio of two noisy averages, with a standard deviation of about 1.2.
The **arithmetic** mean of ``β`` is therefore biased, ``E[2^φ] ≈ 2^{⟨φ⟩}
e^{(σ\ln 2)^2/2}``, which gives 1.3. That reading of the issue's "domain-mean
β within 1 ± 0.15" was the wrong statistic, not a code error; it is
kept as a broken test. For the same reason the domain-mean ``C_s`` is
10-14 % above LASI while the median pointwise ratio is 0.92-0.93, the
``3^{-⟨φ⟩}`` the mean exponent predicts. ``C^2_Δ = C^2_{αΔ}α^{-φ}`` is
convex in ``φ``, and the 2-3 % of cells at the ``β = 1/8`` clip carry
``27 C^2_{αΔ}``. The issue's ±10 % on the mean is not met at 32³ and is
kept broken; ±15 % is asserted.

**Log-law recovery** (the headline test). This is the P54-d half channel
at 24³, with the P57-b 2Δ input filter of the wall model (`filter =
:spatial`, the 3 × 3 wall-parallel box of Bou-Zeid et al. 2005). LASD
starts from the noise and is averaged over 10-20 ``H/u_*``:

| ``z/H`` | 0.042 | 0.083 | 0.125 | 0.167 | 0.208 | 0.250 | 0.292 | 0.333 | 0.417 | 0.500 |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| ``Φ_m``, LASD 24³ | 0.878 | 0.934 | 1.033 | 0.989 | 1.140 | 1.114 | 1.151 | 1.116 | 1.072 | 1.011 |
| ``Φ_m``, LASI 24³, same set-up | 1.505 | 1.003 | 1.031 | 0.741 | 0.795 | 0.711 | 0.723 | 0.690 | 0.666 | 0.632 |

At 32³ (same set-up, seed 1) ``Φ_m`` at ``z/H = k/32``, ``k = 1 … 8`` is
0.890, 0.943, 1.042, 0.993, 1.107, 1.119, 1.137, 1.104.

* ``Φ_m`` stays in [0.8, 1.2] over ``0.05 ≤ z/H ≤ 0.3`` (Bou-Zeid et al.
  2005, Figs. 5-6): [0.934, 1.151] at 24³, [0.943, 1.137] at 32³, and
  [1.042, 1.141] for seed 2 at 24³.
* max ``\lvert Φ_m - 1\rvert`` is 0.151 over the band and 0.122 over
  the first three faces. Static Smagorinsky 0.17 gives max ``Φ_m``
  1.45-1.65 on those faces (P54-d, 12 seeds), so LASD is within a third
  of it; the test asserts less than half, < 0.225.
* The momentum budget closes: ``⟨τ_w⟩ = 1.028 u_*^2`` and the total
  stress stays within 0.032 ``u_*^2`` of ``u_*^2(1 - z/H)``.
* The geometric plane mean of ``β`` is 0.36-0.73 below ``z/H = 0.3``.
  Away from the wall it rises through 1 and above 1 toward the free-slip
  top. Mean ``C_s`` is 0.13-0.16 in the log layer, against 0.04-0.08 for
  LASI.
* The issue's "β → 1 above ``z/H ≈ 0.2``" is not met at these
  resolutions. ``β`` crosses 1 near ``z/H = 0.45`` at 24³ and near 0.32
  at 32³. That is a fixed height in grid cells (10.8 and 10.3), where the
  wider test filter (``5Δ``) is about half the distance to the wall. With
  these filters, "above 0.2 H" needs ``N ≳ 50``. The expectation assumed
  a resolution-independent height, and it is kept as a broken test.

**Spectra** (`examples/validation/neutral_abl.jl`, seed 1, averaged over
18-30 ``H/u_*``; slopes are least-squares fits in log-log):

| run | ``Φ_m``, ``0.05 ≤ z/H ≤ 0.3`` | ``E_{11}`` slope near the wall, ``k_1H ∈ [1, 6]`` | slope at ``z/H ≈ 0.24``, ``k_1H ∈ [4, N/3]`` |
| :-- | :-- | :-- | --: |
| Smagorinsky 0.1, 32³ | [0.965, 1.589] | −2.01 (``z/H`` 0.047), −1.28 (0.109) | −5.14 |
| LASD, 32³ | [0.935, 1.133] | −0.66 (0.047), −0.79 (0.109) | −4.27 |
| LASD, 48³ | [0.961, 1.094] | −0.43 (0.052), −0.61 (0.094) | −3.48 |

``Φ_m`` approaches 1 as the grid is refined. Near the wall LASD gives the
shallow production-range spectrum, where Bou-Zeid et al. (2005) expect
``k_1^{-1}``; the Smagorinsky overshoot steepens it. The
``k_1^{-5/3}`` range at ``z/H ≈ 0.25`` (``k_1 z > 1``, so ``k_1H > 4``)
is **not** reproduced at 32³-48³. Those wavelengths are below 8-12 ``Δx``,
where the second-order scheme and the closure damp. The fitted slope
steepens on coarser grids (−3.5 at 48³, −4.3 at 32³, −5.1 for
Smagorinsky at 32³), which points to numerical dissipation rather than
to the closure.

The CBC decay with LASD is in the CBC table below. On the corrected
(#205) set-up it gives +0.2 % / −10.1 % at 32³, the only closure there
within the proposed ±10 % at station 98; it misses ±10 % at station 171
by 0.1 points. Its errors sit about 12.5 points below LASI's at both
stations because its isotropic ``C_s`` is higher (above). On the
pre-#205 set-up it read −9.6 % / −19.7 %; the exact start moved it by the
same ~10 points as every other closure.

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
| `sgs_explicit_stress(c)` | `true` for a closure whose momentum flux is its own stress (the Reynolds-stress closure, P56-d #188); `step_imex!`, whose implicit operator is an eddy viscosity, then refuses it. Default `false`. |

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
by default.

**Every transported scalar uses κ_t** (P56-a, #177). The scalar flux of
the closure is applied, with the same coefficients, to every scalar a core
or the Berlin case transports, not only to θ:

  * `step!` and `step_rk3!` add it to `qv` when `moisture = true`
    (`add_sgs_scalar!(tqv, qv, closure, f, sim)`, or eddy diffusion with
    `qv`'s own diffusivity for a closure that has one — AMD,
    [`sgs_tracer_diffusivity!`](@ref Aeolus.Operators.sgs_tracer_diffusivity!),
    #176 — the same rule as `step_imex!`'s implicit `qv` coefficient),
    together with the molecular `ν∇²qv` that θ gets; `step_imex!` already
    diffused `qv` implicitly with `ν + κ_t`. Before #177 `step!` advected `qv` but
    never diffused it — a code bug: a moist run's vapour saw no SGS
    mixing at all — and `step_rk3!` did not even combine `qv` over its
    substages. Dry runs are unchanged bit for bit. (The anelastic and
    pseudo-incompressible cores carry no `qv`.)
  * The Berlin case diffuses every tracer its `:transport` stage advects
    (qv, ql, the chemistry species) and the pollutant with `sim.κt` of the
    last dynamics step: one shared κ, θ's κ_t, under every `:sgs` provider
    including `:amd` (see [The Berlin case](../design/berlin_case.md#Numerics-slots)).
    The Berlin core runs with `moisture = false`, so qv and ql are case
    tracers, mixed in the `:transport` stage with the same shared κ, not by
    the core's `qv` path above. Under `:amd` this is a design choice, not a
    missing feature; the reasons follow (#234).

**Why the Berlin tracers share one κ** (#234). AMD can give every scalar
its own diffusivity from its own gradient (see
[below](#Anisotropic-minimum-dissipation-and-Vreman-closures)); the
`:transport` stage deliberately does not use it.

  * *Physics.* At high Reynolds number the subfilter eddies that mix one
    passive scalar mix every other one the same way: the turbulent
    Schmidt number is a property of the flow, not of the species, and one
    `Sc_t` (here `Pr_t`) for every passive scalar is the standard LES
    assumption. Per-scalar AMD κ makes the mixing depend on the shape of
    each field, which is a property of the closure's gradient model, not
    of the turbulence.
  * *Chemistry.* NO, NO₂ and O₃ are chemically linked: away from sources
    they sit near the photostationary state
    ``[\mathrm{NO}][\mathrm{O_3}]/[\mathrm{NO_2}] = j_{NO_2}/k``.
    Mixing them with different κ separates them artificially and moves the
    mixture off that state, i.e. the transport itself does chemistry.
  * *Water.* With different κ for qv and ql, total water ``q_t = q_v + q_l``
    and ``θ_l`` are no longer mixed as conserved variables: a cloud edge
    that mixes vapour faster than liquid creates or destroys
    supersaturation in the mixing step instead of in the microphysics.
  * *Fallback.* Where a scalar's gradient vanishes AMD falls back to
    ``κ = ν_t/Pr_t = 3ν_t``. A tracer that is uniform (a clean background,
    ql = 0 outside clouds) or flat at a plume edge then gets the largest κ
    of all, three times ν_t, and over-mixes plume edges.
  * *Time step.* The adaptive step limits by ``\max(ν_{t,\max},
    κ_{t,\max})`` of θ (`sgs_cfl_diagnostics`); per-tracer κ, which can be
    larger (the fallback), is not seen by it, so an explicit tracer
    diffusion could run past its own stability limit.

Measured. An earlier probe of the Berlin `:amd` run (2026-10-05) gave
cell-wise differences between a tracer's own AMD κ and θ's κ_t of
54–64 % for qv and 66–92 % for the chemistry species, 28 % between NO
and NO₂, 196 % between qv and ql, and a fallback maximum of 5.9 m²/s
against 2.6 m²/s for θ. The design test (`test/berlin_numerics.jl`, *one
shared κ*) recomputes them on the 12-step `:amd` smoke state (fluid cells,
``Σ|Δκ|/Σκ_θ`` against θ's AMD κ_t on the same state): qv 78 % (maximum
0.26 against θ's 2.71 m²/s), NO / NO₂ / O₃ 80 / 80 / 77 %, NO vs NO₂ 24 %,
and ql, which is zero everywhere there, exactly the fallback ``3ν_t`` in
every cell, maximum 5.08 against 2.71 m²/s. (The probe's and the test's
numbers come from different states and normalisations; both show
differences of order one.) The
test pins the choice: the default `:transport` stage is bit for bit the
stage with every tracer forced onto `sim.κt`, while giving qv its own AMD
κ changes qv (and nothing else). The core's moist path (`moisture = true`
in `step!`, `step_rk3!`, `step_imex!`) still gives `qv` AMD's own κ; the
Berlin case does not run it.

The three-argument kernel takes two keywords for this. `mask` (a Bool
solid mask, e.g. `IBM.SolidMask(…).mask`) closes every face with a solid
cell on either side and leaves the solid cells' tendency untouched, so a
tracer neither diffuses into a building nor draws on the stale value held
inside it; the fluid ``Σϕ\,dV`` is conserved because every open face
carries the unmasked flux. `weights = (W_c, W_f)` gives the sound-proof
form ``(1/W)\,∂_j(W κ_t ∂_jϕ)`` with the constraint weight (``ρ̄`` or
``ρ̄θ̄``) at cell centres and ``z`` faces; it conserves ``Σ W ϕ\,dV``.
Without either keyword the kernel is the unmasked one, bit for bit.

| property (`test/era6_scalar_sgs_diffusion.jl`) | result |
| :-- | :-- |
| `qv = sin(2πz/L_z)`, constant `κ₀`, `moisture = true`: amplitude vs the discrete amplification factor, `step!` / `step_rk3!` (100 steps) / ARK2 (20 steps) | 1.8e-15 / 2.6e-15 / 4.4e-16 relative (before #177: `step!` and `step_rk3!` left `qv` at amplitude 1, error 2.3) |
| random positive `κ_t`, no-flux walls, 100 `step!`s: ``Σ qv``; ``Σ qv^2`` | conserved to 3.5e-15 relative; non-increasing at every step |
| steady state of ``∂_z(κ ∂_zϕ) = 0``, ``κ = κ_0(1+z/H)``, flux below, value above, vs ``ϕ_{top} + (FH/κ_0)\ln\frac{1+L/H}{1+z/H}`` | max error 1.42e-5 (``N_z = 32``), 3.64e-6 (64): ratio 3.90, 2nd order |
| blob next to a solid block (and one across the periodic seam), masked, 100 steps | fluid ``Σϕ`` to 2.5e-16 relative, solid cells unchanged (`==`), all-false mask `==` unmasked kernel |
| mass-weighted form | ``Σ W ϕ`` to round-off; a uniform field stays uniform (`== 0` tendency) |
| `moisture = true` vs `false` with ``θ_0 = 0`` (no route back), built-in Smagorinsky | ``u, v, w, θ, ν_t, κ_t`` identical (`==`) |

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

## Anisotropic minimum-dissipation and Vreman closures

Era VI, P55-d (#176), `src/operators/sgs_amd_vreman.jl`. Two closures on
the contract that use the spacing of each direction instead of one
``Δ = (ΔxΔyΔz)^{1/3}``, which matters on the Berlin grid (15.6 × 15.6 ×
6.25 m), and that vanish in laminar flow, where Smagorinsky's
``(C_sΔ)^2|S|`` does not. Both read the one gradient stencil,
`_velocity_gradient_at`; the θ gradient is the centred difference over
two cells (exact for a linear field, like the off-diagonal velocity
gradients).

**AMD** ([`AMDClosure`](@ref Aeolus.Operators.AMDClosure),
[`compute_amd_coefficients!`](@ref Aeolus.Operators.compute_amd_coefficients!)).
The anisotropic minimum-dissipation model of Rozema, Bae, Moin &
Verstappen (2015) bounds the energy of the subfilter scales in a grid box
with the Poincaré inequality and takes the smallest eddy viscosity that
keeps it from growing. With the scalar model of Abkar, Bae & Moin (2016)
and the buoyancy term of Abkar & Moin (2017):

```math
ν_t = C\,\frac{\max\!\left(0,\; -(δ_k∂_k u_i)(δ_k∂_k u_j)S_{ij}
      + \frac{g}{θ_0}(δ_k∂_k w)(δ_k∂_k θ)\right)}{∂_l u_m\,∂_l u_m},
\qquad
κ_t = C\,\frac{\max\!\left(0,\; -(δ_k∂_k u_i)(δ_k∂_k θ)\,∂_i θ\right)}{∂_l θ\,∂_l θ},
```

``δ_k = (Δx, Δy, Δz)``, summed over ``k`` inside each product. The
numerators are the gradient-model estimates of the subfilter shear
production and of the subfilter buoyancy flux. ``κ_t`` is AMD's own
(not ``ν_t/Pr_t``); where ``∂_lθ∂_lθ = 0`` it falls back to
``ν_t/Pr_t``. It is per scalar: another scalar ``ϕ`` gets the same formula
with its own gradient
([`compute_amd_scalar_diffusivity!`](@ref Aeolus.Operators.compute_amd_scalar_diffusivity!),
reached through the contract generic
[`sgs_tracer_diffusivity!`](@ref Aeolus.Operators.sgs_tracer_diffusivity!)).
This is how `step_imex!` diffuses ``q_v``. The Berlin `:transport` stage does not use it
for its tracers: they share θ's κ_t by design (see [*Why the Berlin tracers
share one κ*](#Closure-contract) above, #234). The formula is unchanged under
``ϕ → aϕ + b``. ``C = 0.3`` is the modified Poincaré constant for a
second-order central discretisation from Rozema et al. (2015) (their
decaying-turbulence tests; 1/12 for a spectral method, 0.212 for fourth
order), as restated by Gadde, Stieren & Stevens (2021, §2.3). Abkar & Moin
(2017) used 1/3 on their mixed spectral / second-order grid. The
buoyancy parameter is the core's: ``g/θ_0`` on the Boussinesq core and
``g/\bar θ(z)`` on the sound-proof cores
([`sgs_buoyancy_parameter`](@ref Aeolus.Operators.sgs_buoyancy_parameter)).
No separate Richardson-number correction is applied: the buoyancy term is
AMD's stratification response, and a stability wrapper (P55-e, #187) is
not stacked on it by default.

What the buoyancy term does is set by the sign of ``∂w/∂z``, not by the
stratification alone: on ``θ = θ_s + Γz`` it is ``(g/θ_0)Δz^2(∂w/∂z)Γ``,
which lowers ``ν_t`` where the resolved flow is compressed vertically and
raises it where it is stretched. On a random velocity field with
``Γ = 3`` K/km it lowered ``ν_t`` in 1341 cells and raised it in 450 (of
3584). The monotone decrease with ``N^2`` that the issue asked for holds
for ``∂w/∂z < 0`` (the test uses ``u = ax + Sz``, ``v = ay``,
``w = -2az``).

**Vreman** ([`VremanClosure`](@ref Aeolus.Operators.VremanClosure),
[`compute_vreman_νt!`](@ref Aeolus.Operators.compute_vreman_νt!)).
Vreman (2004): ``ν_t = c\sqrt{B_β/(α_{ij}α_{ij})}`` with
``α_{ij} = ∂_i u_j``, ``β_{ij} = Δ_m^2 α_{mi}α_{mj}``,
``B_β = β_{11}β_{22} - β_{12}^2 + β_{11}β_{33} - β_{13}^2 + β_{22}β_{33} - β_{23}^2``
and ``c = 2.5C_s^2`` (0.0723 for ``C_s = 0.17``); ``κ_t = ν_t/Pr_t``.
``B_β`` is the sum of the principal 2 × 2 minors of ``β`` and is computed
in the equivalent Cauchy–Binet form
``Σ_{m<n} Δ_m^2Δ_n^2 |∂_m\mathbf u × ∂_n\mathbf u|^2``. That form is
non-negative by construction and exactly zero for any flow that varies in
one direction only. The minor form leaves round-off on a two-component
shear ``u(z), v(z)``, and the square root turned that into
``ν_t ≈ 10^{-10}`` m²/s in the Berlin laminar check.

**What was measured** (`test/era6_amd_vreman.jl`, `test/era6_cbc_decay.jl`):

| property | AMD | Vreman | Smagorinsky 0.17 |
| :-- | :-- | :-- | :-- |
| linear field on (15.625, 15.625, 6.25) m vs closed form | ``ν_t``: 9.9e-15, ``κ_t``: 7.2e-14 relative | 4.1e-16 relative | |
| uniform flow / ``u = Sz`` (S = 0.02 s⁻¹) / ``u(z), v(z)`` | 0 / 0 / 0 (exactly) | 0 / 0 / 0 (exactly) | 0 / 0.0766 m²/s / ≠ 0 |
| solid-body rotation, Ω = 0.01 s⁻¹ | 0 (exactly) | 0.125 m²/s ``= c\sqrt{Δx^2Δy^2Ω^2/2}`` | ≈ 0 (6.6e-18) |
| ``u = ax + Sz``, ``v = ay``, ``w = -2az`` (a = 0.01, S = 0.05 s⁻¹), ``N^2/N^2_* = 0, ¼, ½, ¾, 1, 1.5`` | 7.75e-2, 5.81e-2, 3.87e-2, 1.94e-2, 2.3e-15, 0 m²/s; ``N^2_* = 1.025×10^{-3}`` s⁻² analytic | | |
| uniform shift (7, −4, 1.5) m/s, random field | ``ν_t`` 4.8e-16, ``κ_t`` 3.9e-16 relative | 3.5e-16 | |
| decaying turbulence 32³, t = 1: ``⟨2ν_tS_{ij}S_{ij}⟩`` / Smagorinsky | 1.27 (``ν_t = 0`` in 27 % of cells) | 0.82 | 1 |
| CBC ``K_{98}/K_{42}``, ``K_{171}/K_{42}`` vs CBC, 32³ / 48³ / 64³ | +20.0, +7.7 / +5.0, −8.7 / −3.0, −14.2 % | +16.1, −0.3 / +2.7, −13.9 / −5.1, −18.5 % | +13.2, −2.6 / +0.8, −14.4 / −5.9, −17.6 % |
| ``compute_sgs!`` on 64 × 64 × 32 (one thread; a `step!` is 6.2 ms) | 0.71 ms | 2.13 ms | 1.52 ms |

The CBC numbers are those of the corrected benchmark (the #205 spectrum
fix; see [below](#Decaying-grid-turbulence-(Comte-Bellot-and-Corrsin))),
three seeds. At 32³ both closures leave more energy at the cut-off than
Smagorinsky (``E(k_c)``/CBC 1.86 for AMD and 1.77 for Vreman, against
1.62), AMD because ``ν_t = 0`` wherever the resolved strain does not
produce subfilter energy, so the 32³ station-98 excess that every closure
shows is larger for them. That excess is the coarse grid's: at 48³ AMD is
the only closure within the proposed ±10 % at both stations (RMS log
error 0.079 / 0.065), and at 64³ it is the closest to CBC at both.

* Rozema et al. (2015, §II.C) set ``δx_i = Δx_i`` and ``C = 0.300`` for
  second-order central schemes, tuned on this CBC case at 64³. Their
  ``C = 1/3`` is the QR model's constant, not AMD's. The CBC grid is
  isotropic, so the anisotropic ``δ_i`` cannot enter.
* At 32³ the station-98 error is set by the constant: K98 / K171 errors
  +26.9 / +14.6 % for C = 0.25, +20.0 / +7.7 % for 0.3 and +9.2 / −2.4 %
  for 0.4. The gap between the two stations is 11.5–12.3 points for every
  C and 15.6–16.8 for Smagorinsky, Vreman and dynamic: it belongs to the
  set-up (see below), not to AMD.

The test asserts Vreman like the other closures, and AMD at 32³ to what
the grid allows (station 171 within ±10 %, station 98 within ±25 % with
its ±10 % kept as a broken test, more energy at ``k_c`` than
Smagorinsky) and at 48³ to the proposed ±10 % at both stations. It also
asserts the C sweep and the shared station spread.

**An expectation that proved wrong.** Before the #205 fix the 32³ run
gave AMD +9.9 % / −4.7 % (Vreman +4.9 % / −12.7 %), and AMD was taken to be
the first closure that meets ±10 % at station 171. That came from the
over-energetic start, whose extra small-scale energy the closures
drained faster. With the exact start the AMD errors moved by +10 / +12
points. The closure and its constant did not change.

In the Berlin case both are providers of the
`:sgs` slot (`:amd`, `:vreman`; see
[The Berlin case](../design/berlin_case.md#Numerics-slots)).

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
| 32³ | dynamic (plane)     | +13.2 % | −3.6 % | 0.100 / 0.084 | 1.64 / 1.47 |
| 32³ | dynamic (Lagrangian, #173) | +12.8 % | −2.8 % | 0.095 / 0.077 | 1.61 / 1.43 |
| 32³ | LASD (scale-dependent Lagrangian, #186) | +0.2 % | −10.1 % | 0.063 / 0.064 | 1.27 / 1.15 |
| 32³ | Smagorinsky 0.17    | +13.2 % | −2.6 % | 0.102 / 0.085 | 1.62 / 1.51 |
| 32³ | no model            | +98.2 % | +160.6 % | 0.224 / 0.315 | 6.67 / 15.2 |
| 32³ | AMD, C = 0.3 (#176) | +20.0 % | +7.7 % | 0.113 / 0.095 | 1.86 / 1.94 |
| 32³ | Vreman, c = 0.0723 (#176) | +16.1 % | −0.3 % | 0.109 / 0.093 | 1.77 / 1.72 |
| 32³ | σ-model 1.35 (#184) | +17.8 % | +1.5 % | 0.112 / 0.097 | 1.83 / 1.81 |
| 48³ | AMD, C = 0.3 (#176) | +5.0 % | −8.7 % | 0.079 / 0.065 | 1.21 / 1.19 |
| 48³ | Vreman, c = 0.0723 (#176) | +2.7 % | −13.9 % | 0.084 / 0.083 | 1.28 / 1.17 |
| 48³ | Smagorinsky 0.17    | +0.8 % | −14.4 % | 0.076 / 0.078 | 1.14 / 1.02 |
| 64³ | dynamic (plane)     | −4.9 % | −17.6 % | 0.055 / 0.083 | 1.06 / 0.93 |
| 64³ | Smagorinsky 0.17    | −5.9 % | −17.6 % | 0.052 / 0.085 | 0.94 / 0.80 |
| 64³ | AMD, C = 0.3 (#176) | −3.0 % | −14.2 % | 0.052 / 0.074 | 0.94 / 0.80 |
| 64³ | Vreman, c = 0.0723 (#176) | −5.1 % | −18.5 % | 0.055 / 0.088 | 1.08 / 0.91 |
| 64³ | no model            | +54.5 % | +38.6 % | 0.236 / 0.353 | 7.85 / 12.0 |

CI cost at 32³: about 10 s (20 s under load) for the three closures ×
three seeds, plus 2.2 s for the Lagrangian model, which runs through
`step_rk3!(sim, dt; sgs)` so that its memory is committed once per step;
it was not run at 64³. AMD and Vreman add about 2 s each, and the AMD
checks a further 48³ run (about 10 s) and the two C-sweep runs. The
σ-model (#184) adds about 6.5 s, LASD (#186) about 4–13 s.

Without a model the energy piles up at the cut-off (7–15 × CBC). The
test uses this to show that it can fail.

**A defect found in review (#205).** The first version of this table read
+1.3 % / −15.7 % (32³, dynamic) and −13.4 % / −23.5 % (64³). The spectrum
diagnostic then measured the centre average of the staggered velocity,
and the initialiser compensated by dividing by ``\cos(k_jΔ_j/2)`` (see
[filtering](les_filtering.md)). The run was therefore started with more
kinetic energy than CBC in the energy the solver carries: +12.8 % at
32³ and +8.7 % at 64³ after the rescale cycles, and ×2.1 in the cut-off
shell. Measured in that energy, the old 32³ run was +12.8 / +13.8 /
−6.3 % against CBC at stations 42 / 98 / 171. With the face spectrum
the start is exact, and the table above is the corrected run.

**What remains.** The proposed ±10 % on both ratios (a figure from
spectral-code LES) still does not hold across resolutions. Dynamic
model, three seeds:

| ``N`` | 32 | 40 | 48 | 64 |
| :-- | --: | --: | --: | --: |
| ``K_{98}/K_{42}`` error | +13.2 % | +7.5 % | +1.8 % | −4.9 % |
| ``K_{171}/K_{42}`` error | −3.6 % | −9.5 % | −13.9 % | −17.6 % |
| ``E(k_c)``/CBC at 98 | 1.64 | 1.42 | 1.21 | 1.06 |

So the 32³ CI case meets ±10 % at station 171 only because two errors
cancel there. 40³ meets it at both stations, but only because it falls
between the two error trends. Neither result is evidence that the
target is reached.

- **Station 98 at 32³.** The excess is the coarse grid. Here
  ``k_c = 1.8`` cm⁻¹ is only 3.6 × the station-42 peak, and energy
  piles up at the cut-off. The pile-up is gone at 64³.
- **Station 171.** The deficit grows with ``N``. Each of the following
  was varied alone and is ruled out as the cause:

  | Varied | Values tried | ``K_{171}/K_{42}`` error |
  | :-- | :-- | :-- |
  | time step (cfl) | 0.5, 0.25 | equal to 0.1 % |
  | spin-up cycles | 1, 2, 4 | −4.0 / −3.6 / −4.2 % at 32³ |
  | seeds | 3, 8 | −3.6 / −3.4 % |
  | box side, same ``Δ`` | 22M at 64³, 11M at 32³ | −1.5 / −3.6 % |
  | ``C_s`` at 64³ | 0.12, 0.17, 0.22 | −17.5 / −17.6 / −16.8 % |

The spectrum shows where the energy is missing. Shells 2–3
(``k = 0.22``–``0.34`` cm⁻¹) end at 0.62–0.77 of CBC at station 171.
This holds for every closure tried, including none, at every resolution
and in both boxes, and on its own it costs about 6 % of ``K_{171}``.
The rest is a 10–20 % deficit in shells 4–12 at 48³–64³.

The table data are not the cause: they were re-checked against the
printed table 3. CBC (§6, appendix E) do state that their large-eddy
data are the least certain, because of a.c. coupling and the Taylor
approximation at long times. The low-``k`` part of ``E(k)`` also comes
from graphical differentiation of the one-dimensional spectra.

Two candidates in the numerics remain open:

- the second-order truncation and aliasing errors of an implicitly
  filtered finite-difference LES, which are comparable to the SGS term
  near the cut-off (Ghosal 1996; Kravchenko & Moin 1997);
- the eddy-viscosity term in the Laplacian form ``∂_j(ν_t ∂_j u_i)``
  instead of ``∂_j(2ν_t S_{ij})`` (stress-form SGS, #175).

The test therefore asserts ±10 % at station 171 (32³) and ±20 % at
station 98, and keeps ±10 % at station 98 as a broken test. A closure
or numerics change that moves the 64³ numbers is checked with
`examples/validation/cbc_decay.jl`. Starting from the random-phase field
without the rescale cycles gives +33.5 % / +10.5 % at 32³, because the
field transfers no energy during the first turnover.

## Prognostic TKE closure (P56-b)

`PrognosticSGS.TKEClosure` (#182) is the first stateful closure of the
contract: the Deardorff (1980) / Moeng (1984) 1.5-order model with the SGS
TKE ``e`` as its memory, ``ν_t = c_k l\sqrt e`` and ``κ_t = (1 + 2l/Δ)ν_t``.
Its shear production is the exact discrete dual of the momentum operator
above, so the resolved kinetic energy the closure removes is the energy
``e`` gains; its neutral equilibrium is this page's model with
``C_s = 0.194``; its stability length ``0.76\sqrt e/N`` gives a critical
Richardson number of 0.233 (in equilibrium; ``ν_t`` stays positive while
``e`` decays in a transient), so no stability function is applied to it
and `StabilityCorrected` does not wrap it — by design, not a gap. It has
no ``κ_h ≠ κ_v`` anisotropy either (one filter width ``Δ``, as in PALM):
on the Berlin grid it mixes about 3.4× more vertically and 1.84× less
horizontally than `StabilityCorrected` Smagorinsky of equal ``C_s``. Its
``N^2`` misses the moist-adiabatic stability of saturated air (#239).
In the CBC benchmark above it decays too fast (−9.4 % / −26.9 % at 32³).
Everything else is on [Prognostic SGS TKE](sgs_tke.md).

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
| CBC 32³ (3 seeds): ``K_{98}/K_{42}``, ``K_{171}/K_{42}`` vs CBC; RMS ``\log_{10}E`` (98 / 171); ``E(k_c)``/CBC | +17.8 %, +1.5 %; 0.112 / 0.097; 1.83 / 1.81 (on the #205-corrected set-up; +6.6 %, −11.1 % before it) | +13.2 %, −2.6 %; 0.102 / 0.085; 1.62 / 1.51 |
| KH 64×1×128 growth rate (no SGS 0.1820) | 0.1820 (``ν_t ≡ 0``: the layer is 2-D) | 0.1769 (−2.82 %, ``C_s = 0.1``) |
| kernel cost, 64×64×32 | 6.7 ms | 0.85 ms |

The issue expected Smagorinsky to be nonzero in all five laminar cases.
That expectation is wrong for solid-body rotation, which has no strain,
so Smagorinsky is zero there too. The test asserts that.

**Berlin.** The `:sgs` slot's provider `:sigma`
(`examples/urban/couplings/sgs_sigma.jl`) is off by default; flipping
the default (`:tke` since #182) is the maintainer's call. Like every
eddy-viscosity provider it runs the stress-form momentum divergence
(option `stress`, #175). After 100 steps of the `:smoke` window, the mean
``ν_t`` is 3.1e-2 m²/s in the canopy, against 2.5e-2 for Smagorinsky on
the same flow. Above the sponge start it is 4.0e-3 against 2.3e-3 m²/s:
since #171 the sponge relaxes toward the geostrophic wind, so that region
is a nearly uniform stream (ū = 1.0 m/s) with weak 3-D residual
fluctuations (γ = 0.96), not a laminar shear layer, and σ has no laminar
gradient to discount. With the sponge relaxing to zero
(`sponge_to_geostrophic = false`, the pre-#171 flow) the same region is a
laminar sheared layer (γ = 0), and σ gives 5.3e-4 against 2.0e-2 m²/s
(2.7 %). On the laminar version of the current flow (level means of u, v;
``w = 0``) σ's ``ν_t`` is zero to the bit where Smagorinsky is on (the
smoke check AMD and Vreman share), and for an exactly parallel turning
profile on the Berlin mesh too (`test/berlin_numerics.jl`). The smoke step
costs 17.8 ms against 16.5 ms with Smagorinsky.

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

  * No wall damping function (Van Driest / Piomelli) for Smagorinsky, by
    design. The static closure has none; near walls it overshoots
    (``Φ_m`` up to 1.45-1.65 at ``C_s = 0.17``, P54-d). The dynamic
    closures replace damping with the dynamic procedure: the
    scale-dependent Lagrangian model (LASD, #186, above) recovers the log
    law (``Φ_m ∈ [0.93, 1.15]`` in the half channel at 24³) without a
    damping function, and the wall stress comes from the wall model. The
    plane-averaged `DynamicSmagorinsky` is kept as an option of the same
    Germano kernel; it is not scale dependent. The [σ-model](#σ-model)
    needs none either: its ``ν_t`` falls off as ``z^3`` at a no-slip wall
    (measured slope 2.955).
  * The constant coefficient is too dissipative in transitional / stably
    stratified regimes. The dynamic coefficient and the σ-model, which is
    zero in laminar shear and in 2-D flow, are the alternatives; in stable
    stratification the Richardson-number reduction of
    [Stability correction](#Stability-correction) (P55-e) applies to the
    Smagorinsky, dynamic and MLSGS closures, on by default for those
    providers in the Berlin case; its validation on GABLS1 and the
    stratified channel (P55-f) is in
    [Stable boundary layers](stable_boundary_layer.md).
  * The σ-model costs about 8 times the Smagorinsky kernel (6.7 against
    0.85 ms on 64×64×32). It is zero only in *exactly* laminar
    (parallel, 2-D, pure-strain) flow; a laminar but weakly 3-D free
    stream gets a small ``ν_t`` (Berlin, sponge relaxing to zero: 2.7 % of
    Smagorinsky's above the sponge start), and in a weakly turbulent,
    nearly unsheared stream it can exceed Smagorinsky's (Berlin default:
    1.7 ×, both small, 2e-3 ν). It is not stability-aware by itself; it can be
    wrapped in `StabilityCorrected` like any eddy-viscosity closure, but
    the Berlin `:sigma` provider runs it unwrapped. The prognostic TKE
    closure (P56-b, above) has its own stability length and is not
    wrapped.
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
  * Porté-Agel, F., Meneveau, C. & Parlange, M. B. (2000), *A
    scale-dependent dynamic model for large-eddy simulation: application
    to a neutral atmospheric boundary layer*, JFM 415, 261.
  * Bou-Zeid, E., Meneveau, C. & Parlange, M. B. (2005), *A
    scale-dependent Lagrangian dynamic model for large eddy simulation of
    complex turbulent flows*, Phys. Fluids 17, 025105.
  * Lilly, D. K. (1992), *A proposed modification of the Germano subgrid-scale
    closure method*, Phys. Fluids A 4, 633.
  * Lund, T. S. (1997), *On the use of discrete filters for large eddy
    simulation*, CTR Annual Research Briefs, 83.
  * Comte-Bellot, G. & Corrsin, S. (1971), *Simple Eulerian time
    correlation of full- and narrow-band velocity signals in
    grid-generated, 'isotropic' turbulence*, JFM 48, 273.
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
  * Kang, H. S., Chester, S. & Meneveau, C. (2003), *Decaying turbulence
    in an active-grid-generated flow and comparisons with large-eddy
    simulation*, JFM 480, 129.
  * Ghosal, S. (1996), *An analysis of numerical errors in large-eddy
    simulations of turbulence*, J. Comput. Phys. 125, 187.
  * Kravchenko, A. G. & Moin, P. (1997), *On the effect of numerical
    errors in large eddy simulations of turbulent flows*, J. Comput.
    Phys. 131, 310.
  * Deardorff, J. W. (1980), *Stratocumulus-capped mixed layers derived
    from a three-dimensional model*, Boundary-Layer Meteorol. 18, 495
    (prognostic SGS TKE).
  * Rozema, W., Bae, H. J., Moin, P. & Verstappen, R. (2015),
    *Minimum-dissipation models for large-eddy simulation*, Phys. Fluids
    27, 085107.
  * Abkar, M., Bae, H. J. & Moin, P. (2016), *Minimum-dissipation scalar
    transport model for large-eddy simulation of turbulent flows*, Phys.
    Rev. Fluids 1, 041701(R).
  * Abkar, M. & Moin, P. (2017), *Large-eddy simulation of thermally
    stratified atmospheric boundary-layer flow using a minimum
    dissipation model*, Boundary-Layer Meteorol. 165, 405.
  * Gadde, S. N., Stieren, A. & Stevens, R. J. A. M. (2021), *Large-eddy
    simulations of stratified atmospheric boundary layers: comparison of
    different subgrid models*, Boundary-Layer Meteorol. 178, 363.
  * Vreman, A. W. (2004), *An eddy-viscosity subgrid-scale model for
    turbulent shear flow: algebraic theory and applications*, Phys.
    Fluids 16, 3670.
  * Nicoud, F., Baya Toda, H., Cabrit, O., Bose, S. & Lee, J. (2011),
    *Using singular values to build a subgrid-scale model for large eddy
    simulations*, Phys. Fluids 23, 085106 (the σ-model).
  * Hasan, K. M., Basri, A. H. & Kopp, J. (2010), as cited by Nicoud et
    al. (2011) for the closed-form eigenvalues; Smith, O. K. (1961),
    *Eigenvalues of a symmetric 3 × 3 matrix*, Commun. ACM 4, 168.
  * Kopp, J. (2008), *Efficient numerical diagonalization of Hermitian
    3 × 3 matrices*, Int. J. Mod. Phys. C 19, 523.
