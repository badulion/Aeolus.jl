# Boundary conditions

Per-face boundary conditions, grouped into a
[`FieldBoundaryConditions`](@ref Aeolus.BoundaryConditions.FieldBoundaryConditions)
container. `fill_halos!` applies `Periodic` and `NoFlux` on every face
and the two impermeable z-walls, free-slip `ImpermeableWall` and no-slip /
moving `NoSlipWall` (P59-a, #172; theory in
[Transition and wall-bounded flow](../theory/transition.md#Walls)).
`FluxBC(Γ)` is the fixed-gradient top of a cell-centred scalar
(N-parent-b, #254; below). `ValueBC` is declared but has no halo fill.

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

## Pressure-Poisson kind of a face (N-open-a, #251)

The boundary condition of the projection's φ that a velocity face implies
(`:periodic`, or `:neumann` wherever the normal velocity is prescribed),
read by the default pressure solver and by
[`poisson_axis_bcs`](@ref Aeolus.PressureSolvers.poisson_axis_bcs).

```@docs
Aeolus.BoundaryConditions.axis_poisson_bc
```
