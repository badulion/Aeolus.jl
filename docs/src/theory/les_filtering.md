# LES filtering & subgrid closure

## Why filter?

Real turbulent flows have energy across a huge range of scales — in the
atmospheric boundary layer, from the depth of the layer (~1 km) down to
the Kolmogorov microscale (~1 mm). *Direct* numerical simulation would
need `10¹⁸` grid points to resolve them all at meteorological Reynolds
numbers. That is not going to happen.

**Large-eddy simulation** side-steps this by *filtering* the
equations. We separate every field into a resolved (large-scale) part
and a subgrid-scale (SGS) part:

```math
u_i = \bar u_i + u_i''
```

where the overbar is a spatial filter of width `Δ` (in practice, the
grid spacing). Substituting into the Boussinesq momentum equation and
filtering yields:

```math
\frac{\partial \bar u_i}{\partial t}
  + \frac{\partial}{\partial x_j}(\bar u_j \bar u_i)
  = -\frac{1}{\rho_0}\frac{\partial \bar p}{\partial x_i}
     + \bar b\,\delta_{i3}
     - \frac{\partial \tau_{ij}}{\partial x_j}
     + \nu \nabla^2 \bar u_i
```

The new term

```math
\tau_{ij} = \overline{u_i u_j} - \bar u_i \bar u_j
```

is the **subgrid stress tensor** — the effect of unresolved scales on
resolved momentum. It is not closed by the resolved variables alone;
it has to be modelled.

## The Boussinesq eddy-viscosity closure

The oldest and simplest closure, and the one Aeolus implements, is due
to Boussinesq (1877 — a different Boussinesq than the buoyancy one):

```math
\tau_{ij} - \tfrac13\,\tau_{kk}\,\delta_{ij}
    = -2\,\nu_t\,\bar S_{ij}
```

where the resolved rate-of-strain tensor is

```math
\bar S_{ij} = \tfrac12\!\left(\frac{\partial \bar u_i}{\partial x_j}
                            + \frac{\partial \bar u_j}{\partial x_i}\right).
```

The trace (`τₖₖ`) is absorbed into a modified pressure and is not
needed explicitly. The *eddy viscosity* `νₜ` must be specified.

## Smagorinsky-Lilly

The choice of `νₜ` in Aeolus is the classical Smagorinsky-Lilly form:

```math
\nu_t = (C_s\,\Delta)^2\, |\bar S|,
\qquad |\bar S| = \sqrt{2\,\bar S_{ij}\,\bar S_{ij}}.
```

  * `Cₛ` is the Smagorinsky coefficient. In free turbulence `Cₛ ≈
    0.17`. Near walls it is often reduced; wall-model refinements come
    later.
  * `Δ = (Δx·Δy·Δz)^{1/3}` is the geometric-mean filter width — the
    standard choice for anisotropic grids in this closure.

For the scalar transport equation the SGS scalar flux is closed with
an analogous **gradient-diffusion** hypothesis:

```math
\overline{u_j'\theta'} = -\frac{\nu_t}{Pr_t}\frac{\partial \bar \theta}{\partial x_j}
```

with a turbulent Prandtl number `Prₜ`. For dry ABL work `Prₜ ≈ 1/3` is a
common starting value; for engineering `Prₜ ≈ 0.7 – 1` is typical.

See the [Smagorinsky-Lilly](sgs.md) page for the discrete implementation.

## Explicit filters and spectra

The LES filter above is implicit: the grid and the discrete operators
decide what "resolved" means. Analysis, the dynamic procedure and
filtered-DNS comparisons need *explicit* filters and a spectrum, which
`Aeolus.Operators` provides (Phase 54; API in
[Operators](../api/operators.md#LES-analysis-toolbox-(Phase-54))).

**Box (top-hat) filter.** [`box_filter!`](@ref Aeolus.Operators.box_filter!)
averages over ``n`` cells per direction (``n`` odd). On a Fourier mode
``e^{ikx}`` the three-point box acts as

```math
\hat G_{\text{box}}(k) = \frac{1 + 2\cos kΔ}{3},
```

and its second moment ``(n^2-1)Δ^2/12`` is what sets the test-to-grid
width ratio of the dynamic model (``\sqrt{1 + 12 \cdot 2/3} = 3`` for
``n = 3``, see [the SGS page](sgs.md)). It is the dynamic model's test
filter: `compute_dynamic_smagorinsky_νt!` sums the same stencil, so the
two cannot drift (the refactor is bit for bit). With a solid mask the
average runs over fluid neighbours only and is renormalised by their
number, so values inside buildings never leak into the filtered field;
[`staggered_solid_mask`](@ref Aeolus.Operators.staggered_solid_mask)
builds the face masks (a face is solid if either cell is).

**Gaussian and sharp spectral filters** (periodic, FFTW). The Gaussian of
width ``Δ`` has ``\hat G = \exp(-k^2Δ^2/24)``, the same second moment as
a top hat of width ``Δ`` (Pope 2000, §13.2). The sharp cut-off keeps
``|\mathbf k| ≤ k_c`` and is a projection (idempotent).

**Energy spectrum.** [`isotropic_energy_spectrum`](@ref Aeolus.Operators.isotropic_energy_spectrum)
Fourier transforms each velocity component on its own staggered grid and
sums ``\tfrac12|\hat{\mathbf u}|^2`` over shells of unit width
``Δk = 2π/L``, normalised so that ``\sum E Δk = \tfrac12⟨u_iu_i⟩``
exactly (Parseval; shell 0 is the mean flow). The average is over the
stored face values, so the shell sum is the kinetic energy the solver
carries and conserves (`kinetic_energy(sim)` per unit volume). A
half-cell shift only changes the phase of a mode, so no interpolation is
needed. Interpolating to cell centres first, as the first version did,
damps mode ``\mathbf k`` by ``\prod_j\cos^2(k_jΔ_j/2)``, which is 1/2
along an axis at ``k_c/2`` and 0 at ``k_c``. That is a filter, not a
spectrum of the model state.

**Initial fields.** [`isotropic_random_field!`](@ref Aeolus.Operators.isotropic_random_field!)
draws random phases and projects each mode so that the *discrete* C-grid
divergence vanishes. Face amplitudes ``a_j = \hat b_j e^{-ik_jΔ_j/2}``
have divergence ``2i\sum_j \hat b_j \sin(k_jΔ_j/2)/Δ_j``, so ``\hat b``
is projected normal to ``\sin(k_jΔ_j/2)/Δ_j`` (the continuous projection
normal to ``\mathbf k`` is recovered as ``kΔ → 0``). Each shell is then
scaled to hold exactly ``E(k)Δk``: the field is divergence-free to
``10^{-15}`` of its gradients, its spectrum equals the request to
round-off, and so does its kinetic energy. The first version matched the
request in the centre-averaged spectrum and divided the face amplitudes
by ``\cos(k_jΔ_j/2)``. In the solver's energy it started the CBC case
22 % (32³) and 14 % (64³) too energetic, ×2.1 in the cut-off shell (see
[the SGS page](sgs.md#Decaying-grid-turbulence-(Comte-Bellot-and-Corrsin))).
A random-phase field has the right spectrum but no energy transfer.
[`rescale_to_spectrum!`](@ref Aeolus.Operators.rescale_to_spectrum!)
supports the usual "evolve, then rescale the shells back" preparation
(Rogallo 1981), which keeps the phases a run has built.

The benchmark built on these tools — decaying grid turbulence of
Comte-Bellot & Corrsin (1971) — is described on the
[SGS page](sgs.md#Decaying-grid-turbulence-(Comte-Bellot-and-Corrsin)).

## What LES buys us — and what it doesn't

**Buys us:** the ability to run turbulence at meteorological Reynolds
numbers on machines that exist. The largest eddies (which carry most of
the energy and most of the fluxes) are resolved directly; only the
inertial-range and dissipation-range physics is modelled.

**Doesn't buy us:** correct near-wall physics without help. The
resolved velocity gradients near a solid boundary are severely
under-resolved unless we place many grid points inside the surface
layer, which for atmospheric applications is impractical. This is why
LES codes couple to *wall models* (MOST — Monin–Obukhov Similarity
Theory in the atmosphere; equilibrium wall models in engineering).
Aeolus has them in `Aeolus.WallModels` (Phase 57; see
[Wall models](wall_models.md)).

## Reading list

  * Smagorinsky, J. (1963), *General circulation experiments with the
    primitive equations*, Mon. Wea. Rev. 91, 99.
  * Lilly, D. K. (1967), *The representation of small-scale turbulence
    in numerical simulation experiments*, IBM Sci. Comp. Symp.
  * Sagaut, P. (2006), *Large Eddy Simulation for Incompressible Flows*,
    ch. 4 (SGS models).
  * Pope, S. B. (2000), *Turbulent Flows*, ch. 6 (spectra) and 13
    (filters).
  * Rogallo, R. S. (1981), *Numerical experiments in homogeneous
    turbulence*, NASA TM 81315.
