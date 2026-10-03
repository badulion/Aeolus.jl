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

**Using it.** `step!(sim, dt; eddy_viscosity! = (νt, u, v, w) ->
compute_dynamic_smagorinsky_νt!(νt, u, v, w; workspace = ws))` replaces
the built-in closure (the eddy-viscosity terms are added where the
built-in ones are); `ML.MLSGS` closures plug in the same way. The
Berlin case selects it with `physics[:sgs] = :dynamic`.

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
| 32³ | Smagorinsky 0.17    | +13.2 % | −2.6 % | 0.102 / 0.085 | 1.62 / 1.51 |
| 32³ | no model            | +98.2 % | +160.6 % | 0.224 / 0.315 | 6.67 / 15.2 |
| 64³ | dynamic (plane)     | −4.9 % | −17.6 % | 0.055 / 0.083 | 1.06 / 0.93 |
| 64³ | Smagorinsky 0.17    | −5.9 % | −17.6 % | 0.052 / 0.085 | 0.94 / 0.80 |
| 64³ | no model            | +54.5 % | +38.6 % | 0.236 / 0.353 | 7.85 / 12.0 |

CI cost at 32³: about 10 s (20 s under load) for the three closures ×
three seeds.

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

## Limitations and planned upgrades

The current implementation is deliberately the *simplest* working
Smagorinsky-Lilly. Known deficiencies (all fixable in later phases):

  * No wall damping (Van Driest / Piomelli). Needed only when walls
    exist — Phase 1b.
  * The constant coefficient is too dissipative in transitional / stably
    stratified regimes. The dynamic coefficient below is the alternative.
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
  * Germano, M., Piomelli, U., Moin, P. & Cabot, W. H. (1991), *A dynamic
    subgrid-scale eddy viscosity model*, Phys. Fluids A 3, 1760.
  * Lilly, D. K. (1992), *A proposed modification of the Germano subgrid-scale
    closure method*, Phys. Fluids A 4, 633.
  * Lund, T. S. (1997), *On the use of discrete filters for large eddy
    simulation*, CTR Annual Research Briefs, 83.
  * Comte-Bellot, G. & Corrsin, S. (1971), *Simple Eulerian time
    correlation of full- and narrow-band velocity signals in
    grid-generated, 'isotropic' turbulence*, JFM 48, 273.
  * Meneveau, C., Lund, T. S. & Cabot, W. H. (1996), *A Lagrangian
    dynamic subgrid-scale model of turbulence*, JFM 319, 353.
  * Kang, H. S., Chester, S. & Meneveau, C. (2003), *Decaying turbulence
    in an active-grid-generated flow and comparisons with large-eddy
    simulation*, JFM 480, 129.
  * Ghosal, S. (1996), *An analysis of numerical errors in large-eddy
    simulations of turbulence*, J. Comput. Phys. 125, 187.
  * Kravchenko, A. G. & Moin, P. (1997), *On the effect of numerical
    errors in large eddy simulations of turbulent flows*, J. Comput.
    Phys. 131, 310.
