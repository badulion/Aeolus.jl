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

### Radiation outflow

Structures generated inside a nested child — building wakes, plumes,
eddies — reach the outflow face, where the parent's data do not contain
them; a Dirichlet face then reflects them (N-open-c, #260). The
alternative is a radiation condition on the outflow points,

```math
\frac{∂φ}{∂t} + c\,\frac{∂φ}{∂n} = 0 ,
```

selected per face by `open_boundary_conditions(…; outflow = …)`:
`:dirichlet` (the data everywhere, the nested default), `:convective` or
`:orlanski`. A point is an outflow point when the outward normal velocity
of its boundary face is positive at the start of the step; inflow points
keep the Dirichlet data. `prepare_open_boundaries!` fixes the phase speed
once per step:

* `:convective` — ``c = \max(u·n, 0)`` (Pauley, Moin & Reynolds 1990);
* `:orlanski` — for the tangential velocities and the scalars, the speed
  diagnosed from the interior layer next to the face at this and the
  previous step start, ``c = -(∂φ/∂t)/(∂φ/∂n)`` (Orlanski 1976), clipped
  to ``0 ≤ c ≤ Δn/Δt`` (Miller & Thorpe 1981); the normal velocity uses
  the convective speed (below).

Every halo layer ``b`` of every field (u, v, w and each scalar with
boundary data) is then advanced, inside `update_open_boundaries!`, by the
explicit second-order upstream update

```math
φ_b \leftarrow φ_b^0 - r\,\frac{3φ_b^0 - 4φ_{b-1}^0 + φ_{b-2}^0}{2},
\qquad r = \min\!\left(\frac{c\,Δt}{Δn},\,1\right),
```

in values at the start of the (sub)stage (the buffers, and the two
interior layers recorded by `mark_open_boundary_stage!` after the
stage's halo fill). Two details matter:

* **Stage-start values.** At the update the velocity is already the
  predicted ``u^*``, which differs from the projected velocity by
  ``Δt∇φ``. Radiating from ``u^*`` put that pressure-gradient error into
  the outflow; radiating from the stage-start values makes the boundary
  update an explicit stage that SSP-RK3 combines with its own weights. A
  first version that read the halos before the stage's fill picked up
  stale corner values (a uniform stream through the pseudo-incompressible
  core with SGS on drifted, |w| = 6.5e-4 m/s after 100 steps); the
  layers are now recorded after the fill.
* **Second-order upstream.** With the centred interior schemes the
  first-order upstream difference reflected a scalar pulse (σ = 4Δx)
  with a residual ``∫θ'^2/∫θ'^2_0 = 1.9 × 10^{-4}``; the second-order
  one, ``7.9 × 10^{-6}``.

**The normal velocity is not a radiating wave.** In a projected core
continuity ties ``u·n`` at the face to the interior, and the pressure
responds at once to the boundary value. An Orlanski speed diagnosed from
``u·n`` and fed back through the pressure drifts: in the outgoing-vortex
test the diagnosed speeds of u and w settled at 0.79 U and 1.23 U (the
diagnostic evaluated on a run with an accurate boundary gives 0.96 U for
both) and R reached 0.15. With the convective speed for ``u·n`` and
Orlanski for the rest, R = 4.9e-3. Both radiating modes therefore advance
the normal velocity convectively. The mass-flux correction of the
previous section balances the radiated normal velocities exactly as it
balances data.

Measured (`test/era6_outflow_conditions.jl`):

| test | result |
| :-- | :-- |
| outgoing Lamb–Oseen vortex (r_c = 8Δx, U = 1, v_max = 0.5, 192 × 48 child, R when the centre is 3 r_c past the face) | Dirichlet, perfect parent 4.0e-5; Dirichlet, uniform-U parent 1.02; convective 6.5e-3; Orlanski 4.9e-3 |
| scalar pulse (σ = 8Δx), ``∫θ'^2`` after two crossings | convective 1.2e-7, Orlanski 1.7e-6, Dirichlet 1.00 (trapped); interior L2 error of the centred scheme 2.8 % |
| plane Poiseuille, Re = 50, open x | ``\max|u - u_{disc}|`` 1.3e-15; dp/dx relative error 2.6e-14 (middle 80 %); last 5 cells 2.0e-15 |
| 2000 steps, parent stored every 5 steps and interpolated in time | ``\max|∫|u|^2_{child}/∫|u|^2_{parent} - 1|``: Dirichlet 3.0e-5, convective 2.7e-3, Orlanski 2.6e-3 |

The vortex is carried by a uniform stream between free-slip walls (a
2-D x–z flow), whose images make its far field decay as
``\exp(-π|x|/L_z)``; the truth is the same flow in a periodic domain
three times longer. A single vortex in a periodic row would instead carry
the uniform jump ``±Γ/(2L)`` of a vortex sheet, which no uniform parent
matches. The Poiseuille reference is the discrete steady state, the
parabola shifted by ``U_{max}Δz^2/H^2``: the no-slip halo is linear
(``u_0 = -u_1``), and the parabola itself differs from the scheme's
steady solution by exactly that constant.

### The advective flux through the boundary

`boundary_scalar_flux(sim, ϕ; scheme)` (in `AdvectionSchemes`) returns
the flux ``\oint ϕ\,u·n\,dA`` out of each face, formed from the face
velocity and the face value of ``ϕ`` that the scheme reconstructs
(`:centered`, `:upwind`, WENO-Z5/7, TENO5/8, WENO-JS5). The flux-form
schemes telescope, so

```math
\int ϕ\,dV\Big|^{n+1} - \int ϕ\,dV\Big|^{n} = -Δt \oint ϕ\,u·n\,dA
```

per forward-Euler step to round-off: measured ``|·|/\int|ϕ|dV`` = 2.6e-18
(centred), 1.5e-18 (WENO-Z5), 4.0e-18 (TENO5) and 5.5e-18 through
`step!`. It is the only boundary-flux function of the code base: the
nesting and Berlin budgets (N-nest-d, N-nest-e, N-open-d, N-berlin-nest)
use it. The inflow faces of a scalar carry Dirichlet data in all ``H``
halo layers, so the wide upwind stencils read data, not extrapolation;
`open_boundary_conditions(…; halo = required_halo(scheme))` and
`boundary_scalar_flux` refuse a mesh whose halo along an open axis is
shallower than the scheme needs.

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
  * Orlanski, I. (1976), *A simple boundary condition for unbounded
    hyperbolic flows*, J. Comput. Phys. 21, 251–269.
  * Miller, M. J. & Thorpe, A. J. (1981), *Radiation conditions for the
    lateral boundaries of limited-area numerical models*, Q. J. R.
    Meteorol. Soc. 107, 615–628.
  * Pauley, L. L., Moin, P. & Reynolds, W. C. (1990), *The structure of
    two-dimensional separation*, J. Fluid Mech. 220, 397–411.
  * Colonius, T. (2004), *Modeling artificial boundary conditions for
    compressible flow*, Annu. Rev. Fluid Mech. 36, 315–345.
  * Kadasch, E., Sühring, M., Gronemeier, T. & Raasch, S. (2021),
    *Mesoscale nesting interface of the PALM model system 6.0*, Geosci.
    Model Dev. 14, 5435–5465.
