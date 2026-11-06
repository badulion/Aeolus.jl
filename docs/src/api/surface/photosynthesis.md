# Photosynthesis — Farquhar C3 / Collatz C4

`Photosynthesis` is the leaf biochemistry (Phase 47, #19): given the
intercellular CO₂ `ci`, the photon flux a leaf absorbs and its
temperature, how fast does it fix carbon? It implements the Farquhar,
von Caemmerer & Berry (1980) model for C3 leaves and Collatz et al.
(1992) for C4 leaves, in the form used by CLM4.5/CLM5, with the
Bernacchi et al. (2001) Rubisco kinetics and either the non-acclimating
Bonan et al. (2011) or the acclimating Kattge & Knorr (2007) temperature
responses. The theory is in
[Leaf photosynthesis](../../theory/photosynthesis.md).

It is the single owner in Aeolus of:

- the temperature responses (`ArrheniusResponse`,
  `PeakedArrheniusResponse`, `Q10Response`, `temperature_factor`);
- the PAR → photon-flux conversion (`PAR_PHOTONS_PER_JOULE`,
  `par_to_photon_flux`);
- the smooth co-limitation `colimited_rate`;
- the CLM4.5 per-PFT photosynthetic table (`CLM45_PFT_PHOTOSYNTHESIS`).

Stomatal conductance (#20), the vegetated land column (#21) and the
canopy leaf energy balance (#24) delegate here rather than restating
any of these. Leaf-level photosynthesis is not itself a surface, so it
does not implement `AbstractSurface`; the land column that uses it will.

Units: CO₂ in µmol mol⁻¹, O₂ in mmol mol⁻¹, rates in µmol m⁻² s⁻¹,
absorbed photon flux in µmol m⁻² s⁻¹, temperature in K. Mole fractions
make the CLM equations (written in Pa) pressure-independent.

```julia
using Aeolus
p  = pft_photosynthesis_parameters(:c3_grass)          # Vcmax25 = 78.2
Q  = 0.85 * par_to_photon_flux(400.0)                  # absorbed, from 400 W m⁻² PAR
r  = leaf_assimilation(p, 280.0, Q, 298.15)            # ci = 280 µmol/mol, 25 °C
r.An, r.limitation                                     # net µmol m⁻² s⁻¹, :rubisco/:light/:export
co2_compensation_point(p, Q, 298.15)                   # Γ ≈ 54 µmol/mol
```

## Stomatal conductance and leaf gas exchange (#20)

The same module couples the biochemistry to stomata: Ball, Woodrow &
Berry (1987) (`BallBerryStomata`, CLM4.5) and Medlyn et al. (2011)
(`MedlynStomata`, CLM5) closures, solved exactly for the leaf-surface
humidity; the leaf boundary layer (`leaf_boundary_layer_conductance`,
CLM5 eq. 2.5.122); transpiration (`leaf_transpiration`); and the coupled
``A_n``–``g_s``–``c_i`` solution (`leaf_gas_exchange`). Soil-water
stress enters as a factor β through `apply_water_stress`; computing β
from the soil column is #21. Per-PFT parameters: `pft_stomatal_model`,
with the CLM5 Medlyn slopes in `CLM5_MEDLYN_G1`. Theory:
[Stomatal conductance](../../theory/stomatal_conductance.md).

It is the single owner of the stomatal closures, the CO₂/H₂O
diffusivity ratios (`STOMATAL_DIFFUSIVITY_RATIO` = 1.6,
`BOUNDARY_LAYER_DIFFUSIVITY_RATIO` = 1.4) and the leaf boundary-layer
conductance; the land column (#21) and the canopy leaf energy balance
(#24) delegate here. Leaf temperature is an input.

```julia
using Aeolus
using Aeolus.MoistThermodynamics: saturation_vapor_pressure
p  = pft_photosynthesis_parameters(:bdt_temperate)
m  = pft_stomatal_model(:bdt_temperate)                 # Medlyn, g1 = 4.45 kPa^½
T  = 298.15
gb = leaf_boundary_layer_conductance(1.0, T, 101325.0)  # U = 1 m/s → 2.04 mol m⁻² s⁻¹
r  = leaf_gas_exchange(p, m; ca = 400.0, Q = 0.85 * par_to_photon_flux(400.0),
                       T_leaf = T, e_air = saturation_vapor_pressure(T) - 1500.0, gb = gb)
r.An, r.ci / 400, r.gs, r.LE     # ≈ 12.7 µmol m⁻² s⁻¹, 0.78, 0.25 mol m⁻² s⁻¹, 146 W m⁻²
leaf_gas_exchange(p, m; ca = 400.0, Q = 800.0, T_leaf = T, e_air = 1500.0, β = 0.5)  # stressed
```

```@autodocs
Modules = [Aeolus.Photosynthesis]
```
