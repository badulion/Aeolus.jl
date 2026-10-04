# Shared contract — SurfaceExchange

`SurfaceExchange` is to Era V what `RadiationCore` is to Era III and
`MoistThermodynamics` to Era IV: every surface model of the era — soil
column, canopy, urban facets, building energy, snowpack, sea and lake
ice, ocean mixed layer, sub-grid tiles — speaks it, so they agree on
what a roughness length, a stability correction and a turbulent flux
mean. It holds three things:

- **Material constants** (`SurfaceConstants`). Latent heats, the heat
  capacities of water and ice, and `g` are *delegated* to
  `MoistThermodynamics.ThermodynamicConstants` through accessors
  (`LH_v0`, `LH_s0`, `LH_f0`, `grav`, `cp_water`, `cp_ice`, qualified as
  `Aeolus.SurfaceExchange.grav` etc.); the Stefan–Boltzmann constant is
  `RadiationCore.STEFAN_BOLTZMANN`.
- **Monin–Obukhov similarity**: stability functions as a dispatch
  hierarchy (`AbstractStabilityFunctions`, default
  `BusingerDyerBeljaarsHoltslag`) and the surface-layer solver
  `solve_surface_layer`. A new family only needs `psi_m`/`psi_h` (and
  `phi_m`/`phi_h` for diagnostics) to plug into the solver. The
  **maximum sustainable heat flux** (Era VI, P58-e, #193:
  `max_sustainable_heat_flux`, `mshf_solution`, `flux_friction_velocity`,
  `surface_mshf_ratio`; closed form where `log_linear_slope` is defined)
  is the largest downward buoyancy flux MOST can carry at a given wind
  (Van de Wiel et al. 2012). See
  [the derivation](../../theory/turbulence_stability.md#Maximum-sustainable-heat-flux-(P58-e)).
- **The surface interface**: `AbstractSurface` and the generics an
  implementer extends. `surface_albedo` and `surface_emissivity` are the
  `RadiationCore` generics; the one-argument methods are broadband.
  `Radiation.SlabLand` is the first implementer.

Fluxes are positive upward (`H`, `LE`, `E`, `τ`); `Rn` is positive into
the surface and `G` into the substrate, so ``R_n = H + LE + G``.

## Implementing a surface

```julia
using Aeolus
import Aeolus.SurfaceExchange: surface_temperature, roughness_lengths,
    surface_albedo, surface_emissivity, moisture_availability, step_surface!

mutable struct Pond <: AbstractSurface
    T::Float64
end
surface_temperature(p::Pond) = p.T
roughness_lengths(::Pond)   = (z0m = 1e-4, z0h = 1e-5, z0q = 1e-5)
surface_albedo(::Pond)      = 0.06
surface_emissivity(::Pond)  = 0.97
moisture_availability(::Pond) = 1.0          # open water: saturated skin

function step_surface!(p::Pond, f::SurfaceForcing, dt)
    fl = turbulent_fluxes(p, f)              # H, LE, τ, Rn and G = Rn - H - LE
    p.T += fl.G * dt / (1000 * 4181 * 1.0)   # 1 m of water
    return fl
end
```

```@autodocs
Modules = [Aeolus.SurfaceExchange]
```
