# Theory overview

Aeolus solves the **filtered incompressible Boussinesq equations** for a
turbulent boundary-layer flow. This overview names the pieces and links
to the detailed pages.

## What we are solving

We integrate the equations of motion for a resolved velocity field
`u = (u, v, w)`, a resolved potential temperature `θ`, and a pressure
`p`, on a Cartesian domain. The system is:

```math
\begin{aligned}
\frac{\partial u_i}{\partial t} + \frac{\partial}{\partial x_j}(u_j u_i)
    &= -\frac{1}{\rho_0}\frac{\partial p}{\partial x_i}
       + g\,\frac{\theta - \theta_0}{\theta_0}\,\delta_{i3}
       - f\,\epsilon_{ij3}\, u_j
       + \frac{\partial}{\partial x_j}\!\left[(\nu + \nu_t)\frac{\partial u_i}{\partial x_j}\right] \\[4pt]
\frac{\partial u_j}{\partial x_j} &= 0 \\[4pt]
\frac{\partial \theta}{\partial t} + \frac{\partial}{\partial x_j}(u_j \theta)
    &= \frac{\partial}{\partial x_j}\!\left[\!\left(\frac{\nu}{Pr} + \frac{\nu_t}{Pr_t}\right)\!\frac{\partial \theta}{\partial x_j}\right]
\end{aligned}
```

Here:

  * `ρ₀` is a reference density (Boussinesq approximation — density
    variations enter only through buoyancy).
  * `g` is gravitational acceleration; `θ₀` is a reference potential
    temperature.
  * `f` is the Coriolis parameter (f-plane approximation).
  * `ν` is the molecular kinematic viscosity.
  * `νₜ` is the subgrid-scale (SGS) eddy viscosity, closed by the
    Smagorinsky-Lilly model (see [SGS](sgs.md)).
  * `Pr, Prₜ` are molecular and turbulent Prandtl numbers.

The three big pieces of physics — advection, pressure/incompressibility,
and buoyancy — each get their own pages below.

## Where each piece lives in code

| Physics                                        | Kernel                                              | Public API                                                |
| :--------------------------------------------- | :-------------------------------------------------- | :-------------------------------------------------------- |
| Momentum advection `∂(uⱼuᵢ)/∂xⱼ`               | `FiniteVolume._momentum_advect!`                    | [`momentum_advect!`](@ref Aeolus.Operators.momentum_advect!) |
| Scalar advection `∂(uⱼϕ)/∂xⱼ`                   | `FiniteVolume._advect_centered!` (staggered method) | [`advect_centered!`](@ref Aeolus.Operators.advect_centered!) |
| Buoyancy `g(θ-θ₀)/θ₀`                          | `FiniteVolume._add_buoyancy!`                       | [`add_buoyancy!`](@ref Aeolus.Operators.add_buoyancy!)    |
| Coriolis `f × u`                               | `FiniteVolume._add_coriolis!`                       | [`add_coriolis!`](@ref Aeolus.Operators.add_coriolis!)    |
| Smagorinsky-Lilly `νₜ = (CₛΔ)²|S|`              | `FiniteVolume._compute_smagorinsky_νt!`             | [`compute_smagorinsky_νt!`](@ref Aeolus.Operators.compute_smagorinsky_νt!) |
| Eddy viscosity `∂/∂xⱼ(νₜ ∂uᵢ/∂xⱼ)`             | `FiniteVolume._add_eddy_viscosity_*!`               | [`add_eddy_viscosity_u!`](@ref Aeolus.Operators.add_eddy_viscosity_u!) etc.  |
| Pressure Poisson `∇²φ = ∇·u*`                  | `PressureSolvers.FFTPoissonSolver`                  | [`solve!`](@ref Aeolus.PressureSolvers.solve!)             |
| Time integration                               | `Simulations.step!` (Euler-projection)              | [`step!`](@ref)                                            |

## Reading path

1. [Boussinesq equations](boussinesq.md) — where the equations come from
   and what "incompressible" costs us.
2. [LES filtering & closure](les_filtering.md) — why we solve for
   *filtered* fields and where `νₜ` enters.
3. [Staggered discretization](staggered_grid.md) — the Arakawa C-grid,
   why we use it, and what each stencil looks like.
4. [Fractional-step projection](fractional_step.md) — how we enforce
   `∇·u = 0` on the discrete grid.
5. [Smagorinsky-Lilly SGS](sgs.md) — the closure in detail.
6. [Discrete Poisson & FFT](poisson.md) — why the FFT solver recovers
   the pressure to machine precision.
