# Boundary conditions

Per-face boundary conditions, grouped into a
[`FieldBoundaryConditions`](@ref Aeolus.BoundaryConditions.FieldBoundaryConditions)
container. `fill_halos!` applies `Periodic` and `NoFlux` on every face
and the two impermeable z-walls, free-slip `ImpermeableWall` and no-slip /
moving `NoSlipWall` (P59-a, #172; theory in
[Transition and wall-bounded flow](../theory/transition.md#Walls)).
`ValueBC` is declared but has no halo fill. `FluxBC(Γ)` is the
fixed-gradient top of a cell-centred scalar (N-parent-b, #254; below).
Open lateral and top faces fed by boundary data are `OpenBoundary`
(N-open-b, #255), optionally with a convective or Orlanski radiation
outflow (N-open-c, #260); see the last sections.

## BC types

```@docs
Aeolus.BoundaryConditions.AbstractBC
Aeolus.BoundaryConditions.Periodic
Aeolus.BoundaryConditions.NoFlux
Aeolus.BoundaryConditions.ValueBC
Aeolus.BoundaryConditions.FluxBC
```

## Container

```@docs
Aeolus.BoundaryConditions.FieldBoundaryConditions
```

## Halo fill

```@docs
Aeolus.BoundaryConditions.fill_halos!
```

## Wall conditions

```@docs
Aeolus.BoundaryConditions.ImpermeableWall
Aeolus.BoundaryConditions.NoSlipWall
Aeolus.BoundaryConditions.is_wall
Aeolus.BoundaryConditions.homogeneous_bcs
```

## Free-stream top (P59-d, #185)

The top face of a spatially developing boundary layer: zero shear and a
prescribed transpiration velocity (theory in
[Transition and wall-bounded flow](../theory/transition.md#Flat-plate-transition)).

```@docs
Aeolus.BoundaryConditions.FreeStreamBoundary
```

## Pressure-Poisson kind of a face (N-open-a, #251)

The boundary condition of the projection's φ that a velocity face implies
(`:periodic`, or `:neumann` wherever the normal velocity is prescribed),
read by the default pressure solver and by
[`poisson_axis_bcs`](@ref Aeolus.PressureSolvers.poisson_axis_bcs).

```@docs
Aeolus.BoundaryConditions.axis_poisson_bc
```

## Open boundaries (N-open-b, #255)

Lateral and top faces whose values come from outside — a parent run, a
mesoscale model, reanalysis, or an analytic field — as PALM's
Dirichlet child boundary. The normal velocity is written to the
boundary-face slot, tangential velocities and scalars get Dirichlet
halos from the boundary data, and `φ` gets the mirror (Neumann) halo, so
the projection leaves the prescribed face values unchanged. Once per
projected (sub)step every core calls
[`update_open_boundaries!`](@ref Aeolus.BoundaryConditions.update_open_boundaries!),
which refreshes the buffers from the data and applies the code base's
single boundary mass-flux correction. The free-stream top above is the
special case with a fixed normal-velocity profile and mirror halos.
Theory: [Prescribed-flux faces](../theory/fractional_step.md#Prescribed-flux-faces-(open-boundaries)).

```julia
data = ProfileBoundaryData(u = 5.0, v = 2.0, θ = z -> 300 + 0.003z)
bcs  = open_boundary_conditions(mesh, data)          # open W, E, S, N and top; wall bottom
sim  = BoussinesqSimulation(mesh; bcs = bcs, θ₀ = 300.0)
step!(sim, dt)                                        # all-Neumann DCT solver chosen
open_boundary_ledger(sim)                             # (Q, residual, δ, t) of the correction
```

### Faces and buffers

```@docs
Aeolus.BoundaryConditions.AbstractOpenBoundary
Aeolus.BoundaryConditions.OpenBoundary
Aeolus.BoundaryConditions.OpenPlane
Aeolus.BoundaryConditions.OpenBoundaryState
Aeolus.BoundaryConditions.open_boundary_conditions
Aeolus.BoundaryConditions.set_boundary_data!
Aeolus.BoundaryConditions.has_open_boundaries
Aeolus.BoundaryConditions.open_boundary_states
Aeolus.BoundaryConditions.pinned_face_indices
```

### The boundary-data contract

```@docs
Aeolus.BoundaryConditions.AbstractBoundaryData
Aeolus.BoundaryConditions.boundary_values!
Aeolus.BoundaryConditions.ProfileBoundaryData
Aeolus.BoundaryConditions.FunctionBoundaryData
```

### Update, mass-flux correction and multi-stage support

```@docs
Aeolus.BoundaryConditions.update_open_boundaries!
Aeolus.BoundaryConditions.prepare_open_boundaries!
Aeolus.BoundaryConditions.correct_boundary_mass_flux!
Aeolus.BoundaryConditions.open_boundary_flux
Aeolus.BoundaryConditions.boundary_flux_weights
Aeolus.BoundaryConditions.open_boundary_ledger
Aeolus.BoundaryConditions.open_boundary_snapshot
Aeolus.BoundaryConditions.restore_open_boundaries!
Aeolus.BoundaryConditions.combine_open_boundaries!
Aeolus.BoundaryConditions.assert_open_face_correction
Aeolus.BoundaryConditions.open_intermediate_velocity!
```

## Radiation outflow (N-open-c, #260)

`open_boundary_conditions(mesh, data; outflow = :convective)` (or
`:orlanski`, or a `NamedTuple` per side) advances the outflow points of
the open faces by the radiation equation instead of taking the data; the
inflow points keep the data. The advective flux of a scalar through the
boundary, from the scheme's own face fluxes, is
[`boundary_scalar_flux`](@ref Aeolus.AdvectionSchemes.boundary_scalar_flux)
(`AdvectionSchemes`). Theory:
[Radiation outflow](../theory/fractional_step.md#Radiation-outflow).

```julia
bcs = open_boundary_conditions(mesh, data; outflow = (east = :convective,), halo = 3)
sim = BoussinesqSimulation(mesh; bcs = bcs)
step_rk3!(sim, dt)
F = boundary_scalar_flux(sim, sim.θ; scheme = :centered)   # (west, east, …, total)
```

```@docs
Aeolus.BoundaryConditions.RadiationPlane
Aeolus.BoundaryConditions.OUTFLOW_MODES
Aeolus.BoundaryConditions.has_radiating_outflow
Aeolus.BoundaryConditions.mark_open_boundary_stage!
Aeolus.BoundaryConditions.rewind_open_boundaries!
```

## Fixed-gradient scalar top (N-parent-b, #254)

`FluxBC(Γ)` as the `top` face, above a wall or a no-flux bottom: the
ghost cells continue the profile with slope `Γ`
(`φ(N+h) = φ(N+1−h) + Γ·(z_c(N+h) − z_c(N+1−h))`, the cell-centre
heights of a `StretchedGrid` or `dz` steps on a `RegularGrid`), so a
diffusion with diffusivity `κ` carries `κΓ` through the top face. Its
homogeneous part ([`homogeneous_bcs`](@ref
Aeolus.BoundaryConditions.homogeneous_bcs)) is `FluxBC(0)`, which the
implicit diffusion of `step_imex!` uses. Theory and measured convergence:
[Large-scale forcing → The fixed-gradient top](../theory/large_scale_forcing.md#The-fixed-gradient-top).

```julia
θ_bcs = FieldBoundaryConditions(:periodic, :periodic; bottom = :noflux, top = FluxBC(3e-3))
```

A scalar that shares θ's faces but not its prescribed top gradient
(`qv`, the SGS TKE) takes `gradient_free_bcs(θ_bcs)`, which keeps
open-boundary data and replaces only the gradient top by `FluxBC(0)`.

```@docs
Aeolus.BoundaryConditions.gradient_free_bcs
```

## Near the open faces (N-open-d, #262)

The one BC-aware neighbour index (it replaces the periodic `mod1` wraps
of code that must also run in an open domain) and the solid points of an
open face a building touches. Theory:
[Near the open faces](../theory/fractional_step.md#Near-the-open-faces-(N-open-d)).

```@docs
Aeolus.BoundaryConditions.neighbor_index
Aeolus.BoundaryConditions.mark_open_boundary_solids!
```
