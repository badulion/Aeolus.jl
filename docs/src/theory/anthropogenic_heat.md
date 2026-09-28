# Anthropogenic heat from the building energy model

Buildings are the largest source of anthropogenic heat in dense city
centres (Sailor 2011). Until issue #29 the urban example prescribed it:
a two-peak diurnal profile of 2–40 W m⁻² per surface class, with no
source for the numbers. Now the building part of ``Q_F`` is what the
Phase 50 zone model ([Building energy: indoor air and thermal
mass](building_energy.md)) and its HVAC plant ([HVAC demand and exhaust
heat](hvac_exhaust.md)) actually give to the outdoors, step by step.

## 1. Where the energy a building uses goes

Everything a zone exchanges with the outdoors appears in one of its
balances: the air node and the columns (`ZoneBudget`), the vapour in the
air, the plant (`HVACBudget`) and the split of the internal gains
(`InternalGainRates`). `building_heat_release` collects the terms that
leave the building into a `BuildingHeatRelease`, positive outwards:

| term | what it is | where it enters the atmosphere |
|:--|:--|:--|
| `Q_exhaust_sensible`, `Q_exhaust_latent` | condenser / cooling tower rejection, boiler flue | directly (roofs) |
| `Q_ventilation_sensible` | ``H_{ve}(T_{air} - T_{out})`` | directly (canyon) |
| `Q_ventilation_latent` | ``-L_v W_{ventilation}``, net vapour carried out | directly (canyon) |
| `Q_windows` | ``U_w A_w (T_{air} - T_{out})`` | directly (canyon) |
| `Q_lost`, `Q_latent_vented` | gains removed by return/exhaust air | directly (canyon) |
| `Q_envelope_air` | outer faces of sol-air `EnvelopeElement`s | directly |
| `Q_envelope_facets` | ``-Σ A G`` through `FacetElement` outer faces | **through the facet's own surface energy balance** |
| `Q_envelope_ground` | into the ground | the soil |

Summing the zone balance, the vapour balance, the plant's first law and
the gain split gives, for every step,

```math
Q_{released} + \frac{ΔE}{Δt} + L_v ρV \frac{Δq}{Δt}
 = P_{electric} + P_{fuel} + Q_{metabolic} + Q_{solar} + Q_{prescribed},
```

where ``E`` is `zone_heat_content` (air and every column) and ``Q_{solar}``
the transmitted sunshine. Over a period after which the building is back
in its initial state, **the heat released equals the energy consumed
plus the transmitted sun**. That is the premise of the inventory method
of estimating ``Q_F`` (Sailor 2011). A building model adds the timing:
thermal storage delays the release, so a building heated at night gives
part of the heat back the next day.

Three consequences hold exactly and are tested in closed form:

- Steady **heating**: the flue loss plus the conduction and air-exchange
  loss ``HΔT`` add up to the fuel ``Q/η``.
- **Heat-pump** heating: the negative exhaust, ``-Q(1 - 1/COP)``, taken
  from the outdoor air, cancels part of the loss, so the city receives
  ``Q/COP``, the electricity.
- Steady **cooling**: the air exchange brings ``HΔT`` in and the
  condenser rejects ``HΔT(1 + 1/COP)``. The net is ``P_{cool}``: an air
  conditioner adds its power to the city, nothing more, but moves the
  rest from the canyon's air to the condenser's air.

**Direct versus through the facets.** A facet element's outer face is
closed by the Era V surface energy balance. Heat conducted out of the
building warms the skin and leaves as the facet's ``H``, ``LE`` and
emitted long-wave. It is already in the facet fluxes, and adding
`Q_envelope_facets` to the air again would count it twice. So
`anthropogenic_sensible_heat` and `anthropogenic_latent_heat` are only
the direct terms, and `anthropogenic_heat_maps` maps only those: the
exhaust at the building's roof facets (rooftop condensers and towers),
the air exchange, glazing and lost gains at its wall facets, into the
canyon each wall faces. The map sums to the direct release exactly. The
atmosphere above a column receives `aggregate_facet_fluxes(u, fl).H .+
anthropogenic_heat_maps(...).H`.

## 2. Zones behind the facets of a street

`UrbanFacetScene` (#28) gives every roof, wall and street facet its own
conduction column and couples their long-wave exchange implicitly.
`scene_building_zone` builds a zone whose envelope **is** a building's
roof and wall facets: the same `UrbanFacet` objects, wrapped in
`FacetElement`s of the facets' areas. Its floor area is the volume over
the floor height, and its intermediate floors are internal mass. One
zone per building is the TEB-BEM assumption (Bueno et al. 2012).
`step_buildings!` then advances the scene and every zone together.

Each facet column step is affine in two things. One is an extra flux
imposed on its outer face; the facets' long-wave exchange acts through
it. The other is the temperature of its interior environment; the zone
acts through it. Three trial steps per facet (base, +Δq, +1 K) measure
the skin increment ``x_i = a_i + b_i c_i + d_i y_k`` and the flux into
the column at the inner face ``F_i = F^0_i + e_i c_i + g_i y_k``. Here
``c_i = Σ_{j≠i} G_{ij} x_j`` is the long-wave coupling of #28, and
``y_k`` is the change of the air temperature of zone ``k``. The zone's
own elements (other envelope, internal mass, glazing, air exchange) give
its air balance as ``\mathcal{L}_k y_k = R_k + Φ_k`` through the #31
affine assembly. The facets add ``Σ_{i∈k} A_i (q_i - F_i)``, with
``q_i`` the radiative gains on the inner face. Facets and zones form
one dense linear system in ``(x, y)``.

The air node's time constant, ``C_a / Σ hA``, is minutes. The coupling
is implicit, so it is stable at canopy time steps. A lagged coupling,
where the columns step against the old air temperature and the air
against the columns' fluxes, would be unstable there.

**Thermostats by active set.** A zone whose free-floating temperature
leaves its dead band is *pinned* on the violated setpoint. Its unknown
becomes the load ``Φ_k`` instead of ``y_k``. A load beyond capacity is
clamped and the zone freed with that load. A pinned zone that, after a
neighbour changed, needs the opposite action floats again. For one zone
this is exactly `step_zone!(zone, forcing, dt, hvac)`. With several
zones the set settles in a few passes; the loop is bounded.

## 3. Validation

All numbers are measured by `test/era5_anthropogenic_heat.jl`.

| check | reference | measured |
|:--|:--|:--|
| First law every step, 8 days (winter and summer, every element kind, evaporative condenser, dehumidification) | storage from `zone_heat_content` and ``Δq``, independent of the budget | 6.5e-12 (relative to the gross flux) |
| Periodic steady state, day 30 | released = consumed + sun | winter (boiler) 64.541 = 62.403 + 2.138 kWh, imbalance 2e-11; summer (air-cooled AC) 29.766 = 27.628 + 2.138 kWh, imbalance 2e-10 |
| Steady heating, boiler / heat pump / resistance | release = ``Q/η`` | rtol 1e-12 |
| Steady cooling, ΔT = 10 K, COP 3.2 | net = ``P_{cool} = HΔT/COP`` | exhaust 2621.4 W, air exchange + glazing −1997.2 W, net 624.1 W = ``P_{cool}`` |
| `step_buildings!` with no scene facet | `step_zone!` + `step_urban_facets!` | < 1e-10 K, < 1e-8 W |
| Isolated roof (F_sky = 1) over a zone, 24 h with thermostat | `step_zone!` with a `FacetElement` | skin 5.9e-5 K (the analytic vs finite-difference dRn/dT of #28), air 1.7e-8 K, load 2.4e-5 W |
| H/W = 2 canyon, two zones, 12 h | zone budgets / ground columns / first law / map sum | 5.0e-12 / 1.6e-11 / 5.2e-12 / 8.3e-16; setpoint held to 0 K |
| Free-floating zone, 6 h clear night | dt = 60 s | 0.0146 K at 900 s, 0.0302 K at 1800 s, monotone cooling |

**Magnitude: Toulouse, CAPITOUL.** Bueno et al. (2012), Table 3,
describe the dense residential centre of Toulouse: 20 m buildings,
building density 0.68, wall-to-plan ratio 1.05, 30 % glazing, brick with
3 cm insulation, 0.5 ACH, internal gains 5.8 W m⁻² of floor, set points
19–24 °C, 2/3 electric heating. The test builds a 3 × 3 array of
50 m blocks with 10 m streets (density 0.694, wall ratio 1.11). Every
block is a zone behind its facets. The run uses the PVGIS typical-year
weather for the site (ERA5 temperature, SARAH2 irradiance), with 14 days
of spin-up, and averages over 19 Dec–17 Feb. The comparison is the
CAPITOUL inventory of building electricity and gas use (Pigeon et al.
2007), as reported in Bueno et al. Table 2: model means 30.8 and
19.0 W m⁻² with mean bias errors −2.8 and −6.0 W m⁻², so the inventory
is 33.6 and 25.0 W m⁻².

| per m² of urban area | this model | CAPITOUL inventory | TEB-BEM (Bueno et al.) |
|:--|--:|--:|--:|
| electricity | 43.8 | 33.6 | 30.8 |
| gas | 21.9 | 25.0 | 19.0 |
| **total building energy use** | **65.7** (+12 %) | **58.6** | 49.8 (−15 %) |
| heat released | 69.9, of which 49.9 directly to the air, 17.8 through the facets | | |

The test asserts the total within 20 %. Bueno et al. call a 20 % error
in building energy use acceptable for urban canopy models. The
electricity/gas split depends on how the "2/3" heating ratio and the
"electric fraction 0.7" of the gains are read, so it is not asserted.
Table 3's wording, "fraction of electric heating systems over gas
heating systems: 2/3", can also mean electric : gas = 2 : 3 (40 %
electric). With the same run, that reading gives electricity 33.8
(inventory 33.6) and gas 33.1 (inventory 25.0), total 66.8 W m⁻²
(+14 %). The simulated plant heats electrically (η = 1): the released
heat omits the flue loss of the part costed as gas, 1.4 W m⁻² for the
2/3 reading.
The released heat, 69.9 W m⁻², falls inside the 5–100 W m⁻² typical of
cities (Chrysoulakis & Grimmond 2016, after Christen & Vogt 2004 and
Allen et al. 2011). Dense cores exceed 100 W m⁻² (up to 550 W m⁻² in
London, Iamarino et al. 2012).

Changing wall subdivisions from 2 to 1 and spin-up from 14 to 7 days
moves the total by 1.4 % (66.65 W m⁻²).

A whole typical year of the same configuration, not run by the test
suite, was also computed. It used ideal cooling at 24 °C, which
Toulouse's residential centre largely lacks (Bueno et al.'s first
hypothesis). It released 51.4 W m⁻² on the annual mean (winter DJF
72.3, summer JJA 46.0), 41.0 of them directly to the air. The largest
hourly direct release was 104 W m⁻².

**What is not validated.** The weather is a typical year, not the
2004–05 campaign. The inventory is city-scale and the model is a
regular array. Floor height (3 m), window U (3.1 W m⁻² K⁻¹) and SHGC
(0.7), the insulation material, the internal-mass floors (10 cm
concrete) and the boiler efficiency (0.9) are assumptions, labelled in
`test/toulouse_bem_driver.jl`. Sailor (2011) itself is paywalled and
its tables were not accessible here. The ranges above are quoted from
open reviews that cite it.

## 4. The flagship example

`examples/urban/alexanderplatz_diurnal.jl` still runs the slab surface
scheme, which has no building interior. Its anthropogenic heat now
comes from one city-mean `BuildingZone`. The zone takes the scene's
total volume, footprint and wall area, the DOE medium-office schedules,
thermostat and gains (#33), and a packaged DX unit with a gas boiler.
It steps every radiation update (10 min) against the canyon air. The
release is mapped onto the grid:

- exhaust (by floors) and the roof envelope at the roof cells;
- air exchange, glazing, lost gains and the wall envelope at the cells
  next to the buildings.

The heat goes into the first air layer above each column. The
window-transmitted sun is removed from the wall cells' surface budget,
which would otherwise absorb it too.

The walls and roof exchange with the canyon air through the ISO 6946
outside coefficient but **absorb no sunshine**, because the slabs
already absorb it. Two other designs were tried in one- and two-hour
runs and dropped:

- **Sol-air walls** counted the sun twice. The envelope returned
  −8.7 W m⁻² to the air.
- **Outer face held at the slab skin temperature**, with the envelope
  heat sent to the slabs. That stiff interface, exchanged explicitly
  every 10 min, swung the slab budget from −64 to +27 W m⁻² (domain
  mean) between two updates.

Over 06:00–08:00 on 20 June the domain-mean release to the air rises
from 27 to 35 W m⁻². The exhaust rises from 4.6 to 18.9 W m⁻² as the
offices fill and cooling starts.

Traffic is a separate, constant 8 W m⁻² of urban area over the road
cells, the value Bueno et al. use for Toulouse, and is not from the
building model.

**A bug this replaced.** The old code added ``Q_F`` to the `sensible`
matrix. That matrix is a *loss* in the surface budget of
`step_surfaces!`, and the air had already received its sensible heat in
`apply_surface_flux!`. So the prescribed anthropogenic heat never reached
the air and instead cooled the surfaces by up to 40 W m⁻².

## References

- Allen, L., Lindberg, F. & Grimmond, C. S. B. (2011). Global to city
  scale urban anthropogenic heat flux: model and variability. *Int. J.
  Climatol.*, 31, 1990–2005.
- Bueno, B., Pigeon, G., Norford, L. K., Zibouche, K. & Marchadier, C.
  (2012). Development and evaluation of a building energy model
  integrated in the TEB scheme. *Geosci. Model Dev.*, 5, 433–448.
- Christen, A. & Vogt, R. (2004). Energy and radiation balance of a
  central European city. *Int. J. Climatol.*, 24, 1395–1421.
- Chrysoulakis, N. & Grimmond, C. S. B. (2016). Understanding and
  reducing the anthropogenic heat emission. In Santamouris, M. &
  Kolokotsa, D. (eds.), *Urban Climate Mitigation Techniques*,
  Routledge, 27–40.
- Iamarino, M., Beevers, S. & Grimmond, C. S. B. (2012).
  High-resolution (space, time) anthropogenic heat emissions: London
  1970–2025. *Int. J. Climatol.*, 32, 1754–1767.
- Pigeon, G., Legain, D., Durand, P. & Masson, V. (2007). Anthropogenic
  heat release in an old European agglomeration (Toulouse, France).
  *Int. J. Climatol.*, 27, 1969–1981.
- Sailor, D. J. (2011). A review of methods for estimating anthropogenic
  heat and moisture emissions in the urban environment. *Int. J.
  Climatol.*, 31, 189–199.
- PVGIS 5.2, European Commission Joint Research Centre: typical
  meteorological year, 43.604 N 1.444 E.
