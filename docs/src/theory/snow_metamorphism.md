# Snow metamorphism and albedo ageing

Phase 51 (#35) gives the [layered snowpack](snowpack.md) a prognostic
optical grain radius and lets its albedo follow it. The albedo physics
itself is not new: it is the Wiscombe–Warren snow albedo that
`SurfaceOptics.SnowSurface` already implements. This phase supplies the
grain radius that model needs, and hands it over.

## 1. The optical grain radius

Snow is a porous ice matrix whose shortwave properties are fixed, to a
good approximation, by one number: the specific surface area (SSA,
m² kg⁻¹), or equivalently the radius of ice spheres with the same
surface-to-mass ratio,

```math
r = \frac{3}{ρ_{ice}\,\mathrm{SSA}}
```

(`snow_specific_surface_area`, `snow_optical_radius`; ``ρ_{ice}`` is the
contract's 917 kg m⁻³). Fresh snow has SSA ≈ 60 m² kg⁻¹, r ≈ 55 μm; old
melt forms reach SSA ≈ 5 m² kg⁻¹, r > 600 μm (Carmagnola et al. 2014
measured 5–80 m² kg⁻¹ at Col de Porte). Each layer of `LayeredSnowpack`
carries `grain[i]` in μm. New snow enters with
`fresh_grain_radius(scheme, pack, T_air)` and is mass-weighted into the
top layer; merging layers mass-weights the radius; refrozen melt enters
through `mix_refrozen_grain!` (the refreezing itself belongs to #36).

## 2. Dry metamorphism: Flanner & Zender (2006)

Vapour moves from high- to low-curvature ice and, under a temperature
gradient, from warm to cold grains. Flanner & Zender (2006) simulated
this for ensembles of ice spheres and fitted the growth of the effective
radius, for each temperature, temperature gradient and density, to

```math
\frac{dr}{dt} = \left(\frac{dr}{dt}\right)_0
                \left(\frac{τ}{(r - r_0) + τ}\right)^{1/κ},
\qquad r_0 = 54.5\ \mathrm{μm}.
```

The fits are CLM's look-up table `snicar_drdt_bst_fit_60_c070416.nc`
(11 temperatures 223–273 K × 31 gradients 0–300 K m⁻¹ × 8 densities
50–400 kg m⁻³), shipped unchanged in `src/data/` and read with the
in-package `NetCDFIO` reader (`snicar_aging_table`).

**Exact integration.** At fixed parameters the law integrates in closed
form. With ``x = r - r_0 + τ`` and ``e = 1 + 1/κ``,

```math
x(t)^{e} = x(0)^{e} + e\,(dr/dt)_0\,τ^{1/κ}\,t ,
```

(`flanner_grain_radius`). This composes exactly — ageing for ``t_1`` then
``t_2`` is ageing for ``t_1 + t_2`` — so at fixed state the result does
not depend on the time step. CLM takes one forward-Euler step per model
step instead; over 30 days of 30-minute steps the two differ by at most
0.1 % (338.37 vs 338.05 μm at −5 °C, 20 K m⁻¹, 100 kg m⁻³).

**Interpolation.** CLM uses the nearest table node (`interpolation =
:nearest`). The three parameters are fitted jointly and trade off against
one another, so interpolating them separately is wrong: between the
300 and 350 kg m⁻³ nodes at −25 °C and 50 K m⁻¹, where the nodes grow
an 88 μm grain to 181 and 123 μm in 37 days, interpolated parameters give
484 μm. The default `:linear` therefore evaluates the exact solution at
the eight surrounding nodes and interpolates the *radii*
(`snicar_grain_radius`). The result always lies between the node
outcomes, and the 15 K m⁻¹ cell edge that makes nearest-node growth jump
by more than 1 μm over 10 days changes it by less than 1 % of that jump.

**Temperature gradient.** As in CLM (CLM5 Technical Note, Eq. 2.3.76), from
interface temperatures that are thickness-weighted means of neighbouring
layers, with the top interface taken at the top-layer temperature. The
bottom interface is the ground temperature (`SnowGroundTemperature`), or
the temperature that carries the prescribed flux across half the bottom
layer (`SnowBottomFlux`) (`snow_temperature_gradient`).

**Trends.** Flanner & Zender report that warm snow, a strong temperature
gradient and low density age fastest. Over the table the 30-day growth
rises with temperature in 98.1 % of adjacent cells, with gradient in
93.0 % and falls with density in 94.2 %. The exceptions come from the
fits. At 100 kg m⁻³ and no gradient, 30-day radii run from 69 μm at
−50 °C to 133 μm at 0 °C. At −10 °C, 20 K m⁻¹ takes the radius to
296 μm.

## 3. Wet metamorphism: Brun (1989)

Liquid water coarsens grains much faster. Brun (1989) measured growth
rates against liquid-water content. CLM uses them without the dry offset:

```math
\frac{dr}{dt} = \frac{10^{18}\,C_1\,f_{liq}^3}{4π r^2}\ \mathrm{μm\,s^{-1}},
\qquad C_1 = 4.22·10^{-13}\ \mathrm{m^3\,s^{-1}},
```

with ``f_{liq}`` the liquid mass fraction, capped at 0.1 as in CTSM. In
``r^3`` this is linear in time, so it too is integrated exactly. At
``f_{liq} = 0.1`` a 54.5 μm grain reaches 443 μm in ten days. In the
same ten days, dry ageing at 0 °C, 100 kg m⁻³ and no gradient takes it to
108 μm.

With the default `InstantSnowRunoff` liquid-water scheme of #34, no
liquid stays in the pack, so wet growth only acts once #36 retains water.

## 4. Albedo: delegation to `SurfaceOptics.SnowSurface`

`SnowOpticsAlbedo` computes the thickness-weighted grain radius of the
top `sensing_depth` (2 cm) with `snowpack_optical_radius`. It builds
`SnowSurface(spec; grain_radius = r, soot_ppmw)` and returns
`broadband_albedo`, diffuse by default or direct-beam at a given
``μ_0``. `SnowSurface` holds the asymptotic Wiscombe–Warren
parameterisation, ``α = \exp(-\sqrt{\ln^2 α_{clean} + s})`` with
``α_{clean}(r) = α(r_0)^{\sqrt{r/r_0}}``. The snowpack module does not
restate it. `snowpack_optics` returns that `SnowSurface`, so band-resolved
radiation code sees the same surface. Because the albedo enters
`SurfaceExchange.net_radiation` through `surface_albedo`, grain growth
feeds back on the energy balance. At 600 W m⁻², growing the grains from
54.5 to 500 μm adds 64 W m⁻² of absorbed sunshine.

Grain growth darkens the near-infrared, where ice absorbs. From 54.5 to
300 μm the VIS (< 0.7 μm) albedo falls by 1.8 % and the NIR albedo by
22.6 %, a ratio of 12. In the visible the albedo is set by impurities
(`soot_ppmw`).

## 5. Validation

| check | measured | reference |
|:--- |:--- |:--- |
| closed form vs ODE | rel. 1e-6 (finite difference) | ``dr/dt`` law |
| exact vs CLM forward Euler, 30 d, 3 states | ≤ 0.1 % | CTSM `SnowAge_grain`, same table |
| wet growth vs 10⁴ Euler steps, 1 day | rel. 1e-4 | Brun (1989) law |
| fresh snow (54.5 μm) broadband diffuse albedo | 0.847 | CLASS 0.84; USACE/VIC 0.85; ECMWF 0.85 |
| fresh snow VIS / NIR | 0.986 / 0.679 | BATS 0.95 / 0.65 (BATS VIS includes impurities) |
| Summit SSA, 37 d at 325 kg m⁻³ from 37 m² kg⁻¹ | 15.3–28.3 over −20…−10 °C, 10–30 K m⁻¹; 23.8 at −15 °C, 20 K m⁻¹ | observed 23 ± 3 (Carmagnola et al. 2014) |
| same, no gradient, −15 °C | 26.4 m² kg⁻¹ | observed 23 ± 3 |
| 30-day clear-sky pack, −10 °C, clean | 0.851 → 0.770 | USACE 0.545, ECMWF 0.61, CLASS 0.55, BATS ≈ 0.64 |
| same, 0.05 ppmw BC | 0.834 → 0.730 | as above |

What these comparisons are, stated plainly:

* **Summit** is the only comparison with observations. The paper reports
  the SSA at 2–15 cm depth, 37 ± 3 m² kg⁻¹ over 5–19 May 2011 and
  23 ± 3 m² kg⁻¹ over 11–25 June, at 315–335 kg m⁻³. It does not report
  the snow temperature or temperature gradient over that interval, so the
  test scans an *assumed* cold ice-sheet range and asks only that the
  observation fall inside the model's spread. It does, and near the
  centre of the range. Without a gradient the model is at the edge of
  the observed range.
* **Albedo decay** is compared with reference *models*, not data: the US
  Army Corps of Engineers (1956) curves as coded in VIC, ECMWF
  (Douville et al. 1995), CLASS (Verseghy 1991) and BATS (Dickinson et
  al. 1993) as coded in Noah-MP. These schemes were tuned to real, dirty
  snow. BATS's cold-snow ageing, for example, is dominated by its constant
  "dirt and soot" term. Clean-snow grain growth darkens more slowly than
  every one of them: at day 30 the pack is at 0.770 against 0.55–0.64. The
  test asserts this as a bound, not as agreement. With 0.05 ppmw of black
  carbon the pack reaches 0.730. Observed melting snow at Col de Porte goes
  below 0.5 (Carmagnola et al. 2014). Getting there needs refrozen grains
  (#36) and impurity deposition and melt enrichment, which are not
  modelled: `soot_ppmw` is a fixed parameter.
* No SnowMIP site forcing is used.

## References

- Brun, E. (1989). Investigation on wet-snow metamorphism in respect of liquid-water content. *Ann. Glaciol.*, 13, 22–26.
- Carmagnola, C. M. et al. (2014). Implementation and evaluation of prognostic representations of the optical diameter of snow in the SURFEX/ISBA-Crocus detailed snowpack model. *The Cryosphere*, 8, 417–437.
- Dickinson, R. E., Henderson-Sellers, A. & Kennedy, P. J. (1993). *Biosphere–Atmosphere Transfer Scheme (BATS) version 1e.* NCAR/TN-387+STR.
- Douville, H., Royer, J.-F. & Mahfouf, J.-F. (1995). A new snow parameterization for the Météo-France climate model. *Clim. Dyn.*, 12, 21–35.
- Flanner, M. G. & Zender, C. S. (2006). Linking snowpack microphysics and albedo evolution. *J. Geophys. Res.*, 111, D12208.
- Lawrence, D. et al. (2018). *Technical description of version 5.0 of the Community Land Model (CLM).* NCAR, Sect. 2.3.2.3.
- US Army Corps of Engineers (1956). *Snow Hydrology.* North Pacific Division, Portland, OR.
- Verseghy, D. L. (1991). CLASS — a Canadian land surface scheme for GCMs. I. Soil model. *Int. J. Climatol.*, 11, 111–133.
- Wiscombe, W. J. & Warren, S. G. (1980). A model for the spectral albedo of snow. I: Pure snow. *J. Atmos. Sci.*, 37, 2712–2733.
- Yang, Z.-L., Dickinson, R. E., Robock, A. & Vinnikov, K. Y. (1997). Validation of the snow submodel of the Biosphere–Atmosphere Transfer Scheme with Russian snow cover and meteorological observational data. *J. Climate*, 10, 353–373.
