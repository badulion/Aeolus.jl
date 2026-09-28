# Photosynthesis — Farquhar C3 / Collatz C4

`Photosynthesis` is Era V's leaf biochemistry (Phase 47, #19): given the
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

```@autodocs
Modules = [Aeolus.Photosynthesis]
```
