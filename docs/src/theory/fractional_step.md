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

## Prescribed-flux faces (open boundaries)

A nested child, or any domain driven by outside data, prescribes the
velocity on every face (N-open-b, #255; `OpenBoundary`). The normal
component is written into the boundary-face slot of `u*` *before* the
divergence is formed, and `φ` obeys the homogeneous Neumann condition
`∂φ/∂n = 0` (Kim & Moin 1985). Then

```math
u^{n+1}\cdot n = u^*\cdot n - Δt\,\frac{∂φ}{∂n} = g^{n+1}\cdot n ,
```

so the projection leaves the prescribed face value exactly unchanged (the
code asserts that `∂φ/∂n` is exactly zero at the west and south faces,
whose face slots lie inside the interior index range). This is not an
approximation: with ``u^* = u(t+Δt) + Δt∇p`` in the interior and the face
value ``g`` on the boundary, the discrete Neumann problem is solved by
``φ = Δt\,p`` — the face's missing ``Δt\,∂p/∂n`` is exactly the term the
mirror halo drops from the boundary cell's Laplacian.

### Compatibility and the mass-flux correction

With every normal velocity prescribed, the Neumann problem is solvable
only if ``\sum ∇·u^*\,ΔV = \oint u·n\,dA = 0``
([compatibility](poisson.md#The-discrete-compatibility-condition)).
Interpolated or analytic data never balance to round-off, and the
solvers would leave the uniform residual ``∇·u = Q/V`` in every cell.
`update_open_boundaries!` therefore applies, once per projected
(sub)step and before the projection, the only boundary mass correction
in the code base (`correct_boundary_mass_flux!`; Hellsten et al. 2021,
§3.3, Eq. 2):

```math
Q = \sum_{\text{open faces}} W\,u·n\,A, \qquad
δ = \frac{Q}{\sum W A}, \qquad u·n \leftarrow u·n - δ ,
```

with ``W = 1`` for the Boussinesq core and the constraint weight
(``ρ̄`` anelastic, ``ρ̄θ̄`` pseudo-incompressible) for the sound-proof
cores, taken exactly as the projection's divergence takes it (``W_c(k)``
on lateral faces, ``W_f`` on the top). `correction = :outflow_only`
adjusts only the outflow faces. Measured (`test/era6_open_boundaries.jl`,
16 × 12 × 8 box, ``Q = 4.8 × 10^4`` m³ s⁻¹ of net outflow): without the
correction ``\max|∇·u| = Q/V = 2.5 × 10^{-4}`` s⁻¹ in every core, to
1e-12; with it ``\max|∇·u| < 10^{-12}`` s⁻¹ in `step!`, `step_rk3!`,
`step_imex!` (ARK2, ROS2) and `step_adaptive!`, and
``\max|(1/W)∇·(Wu)| < 10^{-9}`` s⁻¹ in the anelastic and
pseudo-incompressible cores; the boundary faces equal the corrected data
bit for bit.

### Where the boundary data enter a step

The buffers belong to the state: the halo fill at the start of a
(sub)step reads them, so they must hold the boundary values of the
(sub)step's start state. `update_open_boundaries!(sim, t, dt)` is
called after the predictor with the time `t` the projected velocity
belongs to — `t + Δt` for an Euler step; `t + Δt`, `t + 2Δt` and
`t + 3Δt/2` for the three Shu–Osher substages of `step_rk3!`, whose
buffers are then combined with the fields' own weights (3/4, 1/4 and
1/3, 2/3); the stage times of ARK2 and ROS2. `step_adaptive!`
restores the buffers with the fields when it rejects a step.

### Implicit diffusion: the intermediate-velocity condition

With an implicit viscous term (`step_imex!`) the diffusion solve needs
boundary values for the *intermediate* velocity ``u^* = u^{n+1} + ∇φ``,
``φ ≈ τ p`` (``τ`` the stage's projection step). Imposing the data
``g`` itself on the tangential components leaves, after the projection, a
viscous boundary layer of amplitude ``τ\,∂p/∂s`` — a first-order
splitting error that shows up as order reduction. Kim & Moin's (1985)
remedy is the condition ``u^*_s = g_s + τ\,∂p/∂s``, with the latest
pressure; `open_intermediate_velocity!` applies it around each ARK2
stage (the projection itself sees ``g``). Measured on the
Dirichlet-everywhere Taylor–Green window (``ν Δt/h^2 = 0.2``,
``N = 16, 32, 64``): with ``g`` alone the observed order was 1.60 and
1.65, the time error growing with ``ν Δt/h^2`` (at 0.4 the ``N = 32``
error was 2.2 times the spatial one); with the Kim & Moin condition it is
2.005 and 2.006 and independent of ``ν Δt/h^2`` up to 0.4. The
explicit drivers need no such condition (their tendencies use the
step-start halos): `step_rk3!` measures 2.035 and 2.022.

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
  * Kim, J. & Moin, P. (1985), *Application of a fractional-step method
    to incompressible Navier–Stokes equations*, J. Comput. Phys. 59,
    308–323.
  * Hellsten, A., Ketelsen, K., Sühring, M. et al. (2021), *A nested
    multi-scale system implemented in the large-eddy simulation model
    PALM model system 6.0*, Geosci. Model Dev. 14, 3185–3214.
  * Kadasch, E., Sühring, M., Gronemeier, T. & Raasch, S. (2021),
    *Mesoscale nesting interface of the PALM model system 6.0*, Geosci.
    Model Dev. 14, 5435–5465.
