# Boussinesq equations

The **Boussinesq approximation** is a controlled simplification of the
compressible Navier–Stokes system that is standard for boundary-layer
flows in the atmosphere, ocean, and much of engineering CFD. It rests on
two observations:

1. Density variations across the flow are *small* compared to a mean
   `ρ₀`. Formally, `Δρ/ρ₀ ≪ 1`.
2. Those small density variations matter *only where they multiply
   gravity* — i.e. in the buoyancy term. Everywhere else we can treat
   the fluid as incompressible with density `ρ₀`.

Applying these two rules to the compressible momentum equation yields:

```math
\frac{\partial u_i}{\partial t} + u_j\frac{\partial u_i}{\partial x_j}
    = -\frac{1}{\rho_0}\frac{\partial p}{\partial x_i}
      - \frac{\rho'}{\rho_0}\,g\,\delta_{i3}
      + \nu \nabla^2 u_i
```

paired with the incompressibility constraint

```math
\frac{\partial u_j}{\partial x_j} = 0.
```

## Why Boussinesq for boundary layers

For atmospheric turbulence and boundary-layer meteorology:

  * Vertical scales are typically ≤ 2–3 km, well within the *shallow*
    limit where the anelastic corrections are small.
  * Mach numbers are tiny; acoustic modes are irrelevant.
  * Buoyancy is the dominant density effect and is the only one that
    survives to drive convection.

For the ocean, the Boussinesq approximation is even better justified —
seawater density variations are ~0.1 %.

For engineering flows (heated channels, chimneys, indoor ventilation),
the same approximation applies whenever `βΔT ≪ 1` with `β` the thermal
expansion coefficient.

## Buoyancy from potential temperature

For a dry atmosphere the density anomaly `ρ'/ρ₀` is related to the
*potential temperature* anomaly by:

```math
\frac{\rho'}{\rho_0} \approx -\frac{\theta - \theta_0}{\theta_0}
```

(θ is the temperature a fluid parcel would have if brought
adiabatically to a reference pressure). This gives the buoyancy term
we implement in the w-momentum equation:

```math
b = g\,\frac{\theta - \theta_0}{\theta_0}.
```

Concretely: [`add_buoyancy!`](@ref Aeolus.Operators.add_buoyancy!)
interpolates `θ` from cell centers to the z-face and adds `b` to the
`w`-tendency.

## Rotation — the f-plane

For flows over horizontal scales of tens of km or more, the Coriolis
force is not negligible. Aeolus uses the *f-plane* approximation:

```math
\text{Coriolis} = -f\,\hat z \times u
= (+f\,v,\; -f\,u,\; 0)
```

with a constant `f = 2Ω sin(φ)` where `Ω` is Earth's rotation rate and
`φ` is latitude. This is standard for LES boxes small enough that
curvature can be ignored (typically < 100 km).

Implemented by [`add_coriolis!`](@ref Aeolus.Operators.add_coriolis!),
which uses a 4-point horizontal average to interpolate `v` to the
`u`-face and vice versa — preserving the discrete rotational symmetry.

## What "incompressible" costs us

Because there is no evolution equation for `p`, the pressure is a
*Lagrange multiplier* that enforces `∇·u = 0` at every instant. On the
continuum this is elegant; on a discrete grid it means we must solve a
Poisson equation every time step (see
[Fractional-step projection](fractional_step.md)). The FFT solver we
use is exact on the discrete operator, so this is not an approximation
— just a computational cost.

## What we *don't* include (yet)

  * **Non-hydrostatic compressibility corrections** (anelastic /
    pseudo-incompressible systems). Aeolus is purely Boussinesq — deep
    convection with strong density variation is out of scope.
  * **Moisture** — the water-vapor and cloud-microphysics equations are
    a Phase 2 target.
  * **Radiation** — Phase 5.

## References

  * Spiegel, E. A. & Veronis, G. (1960), *On the Boussinesq approximation
    for a compressible fluid*, ApJ 131, 442.
  * Wyngaard, J. C. (2010), *Turbulence in the Atmosphere*, ch. 3–4.
  * Pope, S. B. (2000), *Turbulent Flows*, §1.3.
