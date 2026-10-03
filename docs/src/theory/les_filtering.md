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
  * Pope, S. B. (2000), *Turbulent Flows*, ch. 13.
