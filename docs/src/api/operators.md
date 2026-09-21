# Operators

The `Aeolus.Operators` module is a thin dispatch façade. Each function
here forwards to a location-and-mesh-specialized kernel in the
`FiniteVolume` submodule (see [Multiple dispatch](../design/dispatch.md)).

## First derivatives

```@docs
Aeolus.Operators.∂x
Aeolus.Operators.∂y
Aeolus.Operators.∂z
```

## Second-order operators

```@docs
Aeolus.Operators.laplacian
Aeolus.Operators.divergence
```

## Advection

```@docs
Aeolus.Operators.advect_centered!
Aeolus.Operators.advect_upwind!
Aeolus.Operators.momentum_advect!
```

## Interpolation between staggered locations

```@docs
Aeolus.Operators.interpolate_to_face!
Aeolus.Operators.interpolate_to_center!
```

## Physics forcings

```@docs
Aeolus.Operators.add_buoyancy!
Aeolus.Operators.add_coriolis!
```

## Subgrid-scale closure

```@docs
Aeolus.Operators.compute_smagorinsky_νt!
Aeolus.Operators.add_eddy_viscosity_u!
Aeolus.Operators.add_eddy_viscosity_v!
Aeolus.Operators.add_eddy_viscosity_w!
Aeolus.Operators.add_eddy_diffusion_scalar!
```
