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

## LES analysis toolbox (Phase 54)

Explicit filters, the isotropic energy spectrum and the isotropic
random-field initialiser (see [LES filtering & closure](../theory/les_filtering.md#Explicit-filters-and-spectra)).
The box filter is the test filter of the dynamic Smagorinsky model; the
Fourier tools treat the interior as triply periodic.

```@docs
Aeolus.Operators.box_filter!
Aeolus.Operators.staggered_solid_mask
Aeolus.Operators.gaussian_filter!
Aeolus.Operators.spectral_cutoff_filter!
Aeolus.Operators.isotropic_energy_spectrum
Aeolus.Operators.isotropic_random_field!
Aeolus.Operators.rescale_to_spectrum!
```
