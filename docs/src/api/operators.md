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
Aeolus.Operators.compute_dynamic_smagorinsky_νt!
Aeolus.Operators.DynamicSmagorinskyWorkspace
Aeolus.Operators.add_eddy_viscosity_u!
Aeolus.Operators.add_eddy_viscosity_v!
Aeolus.Operators.add_eddy_viscosity_w!
Aeolus.Operators.add_eddy_diffusion_scalar!
```

## Stratification (Era VI, P58-a)

Pointwise fields of the [`TurbulenceStability`](era6/turbulence_stability.md)
contract.

```@docs
Aeolus.Operators.buoyancy_frequency_squared!
Aeolus.Operators.strain_rate_squared!
Aeolus.Operators.gradient_richardson!
```

## Advection schemes

```@docs
Aeolus.Operators.advect_weno5!
```

## Forcing terms

```@docs
Aeolus.Operators.add_surface_flux!
Aeolus.Operators.add_sponge!
```

## Damping profiles

The one implementation of relaxation rates, shared by the Rayleigh sponge
(every mesh type, and the `:cos2` nest sponge) and the fringe region of
`Transition` (P59-d, #185).

```@docs
Aeolus.Operators.AbstractRamp
Aeolus.Operators.SinSquaredRamp
Aeolus.Operators.NordstromRamp
Aeolus.Operators.smooth_step
Aeolus.Operators.DampingProfile
Aeolus.Operators.sponge_profile
Aeolus.Operators.damping_rate
Aeolus.Operators.FiniteVolume.in_damping
```
