# Fractional-step projection

The Boussinesq equations couple velocity and pressure through the
incompressibility constraint. Because there is no evolution equation
for `p` (it is a Lagrange multiplier), we cannot advance `u` and `p`
by a straightforward explicit step. The classical remedy is the
**fractional-step / pressure-projection method** of Chorin (1968)
and Temam (1969).

## The idea

Split each time step into two sub-problems:

1. **Predictor.** Advance the momentum equation *ignoring the pressure
   gradient*. The result `u*` is an intermediate velocity that is
   generally not divergence-free.
2. **Corrector.** Find a pressure-like field `φ` whose gradient projects
   `u*` back onto the space of divergence-free fields.

Concretely, one step of size `Δt` is:

```math
\frac{u^* - u^n}{\Delta t} = F(u^n)                          \tag{1}
```

```math
\frac{u^{n+1} - u^*}{\Delta t} = -\frac{1}{\rho_0}\nabla \phi \tag{2}
```

with `F(u)` collecting advection, viscous, SGS, buoyancy, Coriolis, and
any body forces — everything *except* the pressure gradient. Taking
divergence of (2) and applying incompressibility `∇·uⁿ⁺¹ = 0`:

```math
\nabla^2 \phi = \frac{\rho_0}{\Delta t}\,\nabla\!\cdot\! u^*.   \tag{3}
```

Solve (3) for `φ`, then apply (2) to get `uⁿ⁺¹`. The physical pressure
after the step is `p = φ/Δt`.

## The Aeolus recipe

In code (`Simulations.step!`), one Euler-projection step reads:

```julia
1.  fill_halos!(u, v, w, θ)
2.  zero!(tu, tv, tw, tθ)
3.  momentum_advect!(tu, tv, tw, u, v, w)
    advect_centered!(tθ, u, v, w, θ)
4.  if ν > 0
        tu += ν ∇²u; tv += ν ∇²v; tw += ν ∇²w; tθ += ν ∇²θ
5.  if C_s > 0
        compute_smagorinsky_νt!(νt, u, v, w)
        add_eddy_viscosity_[u,v,w]!(t_, _, νt)
        add_eddy_diffusion_scalar!(tθ, θ, νt, Pr_t)
6.  if θ₀ > 0
        add_buoyancy!(tw, θ, θ₀, g)
7.  if f ≠ 0
        add_coriolis!(tu, tv, u, v, f)
8.  u* = u + Δt · tu     (in place)
    v* = v + Δt · tv
    w* = w + Δt · tw
9.  fill_halos!(u*, v*, w*)
    div_buf = ∇·u*
10. solve!(φ, div_buf, solver)      # solves ∇²φ = div_buf
11. u   = u* - ∂x φ                 (staggered gradient at faces)
    v   = v* - ∂y φ
    w   = w* - ∂z φ
12. θ  = θ + Δt · tθ
    p  = φ / Δt
    t += Δt
```

Steps 1–7 build the tendency `F(uⁿ)`; steps 8–9 form `u*`; steps 10–11
project. All buffers are pre-allocated in the constructor — no heap
traffic in the hot loop.

## Why the projection is *exact* on the discrete grid

As shown on the [staggered grid page](staggered_grid.md), the discrete
divergence of the discrete gradient equals the standard 7-point
Laplacian. If the Poisson solve in step (10) uses that same discrete
Laplacian (as our FFT solver does — see [Discrete Poisson](poisson.md)),
then:

```math
\nabla \!\cdot\! u^{n+1} = \nabla\!\cdot\! u^* - \nabla\!\cdot\!\nabla\phi
                        = \nabla\!\cdot\! u^* - \nabla^2\phi = 0.
```

The residual divergence is *zero to machine precision*, not to some
truncation order. The `max_divergence` diagnostic in our TGV and
convection examples sits at `10⁻¹⁵`–`10⁻¹⁶` throughout the run,
confirming this.

## Order of accuracy

The Chorin-style splitting used above is:

  * **First-order** in time (Euler predictor).
  * **Second-order** in space (from the finite-difference stencils).

There is a well-known **splitting error** on the pressure of `O(Δt)`
that biases the pressure boundary layer but does not degrade the
velocity accuracy globally. For higher-order timestepping we can
apply the projection at each stage of a Runge-Kutta scheme — this is
**projected RK3**, planned for a later step.

## Splitting error and its remedies

The Chorin scheme applies `∇p` only *after* the predictor. This is
equivalent to imposing a boundary condition `∂φ/∂n = 0` on `φ` (for
periodic BCs this is trivial). For walls, `∂φ/∂n = 0` is inconsistent
with `∂p/∂n ≠ 0` at solid boundaries, giving the well-known "pressure
boundary layer". The **incremental projection** and **rotational
projection** schemes of Timmermans, Minev & Van De Vosse and Guermond
et al. fix this by projecting `pⁿ⁺¹ - pⁿ` instead of `pⁿ⁺¹`. Aeolus is
periodic-only for now, so this refinement is deferred to Phase 1b when
walls arrive.

## Alternative: pressure-Poisson-per-substage

For RK3 (or higher-order Runge-Kutta), the divergence-free constraint
must be enforced at *each* Runge-Kutta stage, not just once per step.
Each stage becomes a mini-projection. Cost: 3× the Poisson solves per
time step. Benefit: 3rd-order in time on `u`, and no splitting-error
accumulation. This is what Oceananigans.jl and NCAR-LES do.

## References

  * Chorin, A. J. (1968), *Numerical solution of the Navier-Stokes
    equations*, Math. Comput. 22, 745.
  * Temam, R. (1969), *Sur l'approximation de la solution des équations
    de Navier-Stokes par la méthode des pas fractionnaires*, Arch.
    Rational Mech. Anal. 32, 135.
  * Guermond, J.-L., Minev, P. & Shen, J. (2006), *An overview of
    projection methods for incompressible flows*, CMAME 195, 6011.
  * Ferziger, J. H. & Perić, M. (2020), *Computational Methods for
    Fluid Dynamics*, ch. 8.
