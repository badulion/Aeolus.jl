# BUBBLE / CAPITOUL validation of the urban canopy

This page is the acceptance test of the Phase 49 urban canopy (#30). The
pieces were validated one at a time against analytic solutions and
standards: conduction ([Urban facet conduction](urban_conduction.md)),
radiosity coupling ([Facet–radiosity coupling](urban_radiosity_coupling.md)),
the building zone ([Building energy](building_energy.md)) and glazing
([Glazing](urban_glazing.md)). Here all of them run together in street
canyons taken from two urban field campaigns, and the results are
compared with published observations:

- **CAPITOUL** (Toulouse, 2004–05): the surface temperatures of roads,
  walls and a roof.
- **BUBBLE** (Basel, June 2002): the surface energy balance.

The test is `validation/urban_validation.jl`. The case is built in
`validation/support/urban_validation_driver.jl`, which labels every input with its
source, or as an assumption.

## The offline canyon case

Every piece of physics comes from the package:

- **Scene.** A street canyon between two building rows is a `Radiosity`
  scene (`build_scene`, `view_factors`). Shading is ray-cast and view
  factors are exact.
- **Rotation.** The scene's street axis is its +y axis. Real street
  azimuths are handled by rotating the sun into the scene frame.
- **Facets.** Every facet is an `UrbanFacet` conduction column. The
  facets are coupled through the implicit longwave radiosity system
  (`UrbanFacetScene`).
- **Buildings.** Each building row is a `BuildingZone` behind its own
  facets (`scene_building_zone`), solved with them in one implicit system
  (`step_buildings!`). Following TEB in the CAPITOUL runs (Pigeon et al.
  2009), zones are heated to at least 19 °C and not cooled.
- **Air-side exchange.** This uses the TEB closures, added to
  `UrbanSurface` for this issue:
  - **Roofs:** Monin–Obukhov similarity (`SurfaceExchange`) in the
    forcing wind, with TEB's roughness lengths
    ``z_{0m} = 0.15`` m and ``z_{0h} = 7.5\times10^{-4}`` m.
  - **Street:** Monin–Obukhov similarity in the canyon wind at ``H/2``,
    with ``z_{0m} = 0.05`` m and ``z_{0h} = 2.5\times10^{-4}`` m (Masson
    et al. 2002; Lemonsu et al. 2004).
  - **Walls:** ``h = 11.8 + 4.2\,U_{can}``, from
    [`rowley_convection_coefficient`](@ref).
  - **Canyon wind** ([`canyon_wind_speed`](@ref), Masson 2000; Lemonsu
    et al. 2004 eq. B1):

```math
U_{can} = \frac{2}{\pi}\exp\!\left(-\frac{H}{4W}\right)
          \frac{\ln\big((H/3)/z_{0,town}\big)}{\ln\big((z_{ref}+H/3)/z_{0,town}\big)}\,U,
          \qquad z_{0,town} = H/10 .
```

- **Sampling.** Temperatures are averaged over the facets of the two
  central rows of a canyon about 5 H long. The street there sees the sky
  within 0.8–1.4 % of an infinitely long canyon (`canyon_sky_view_factor`,
  asserted < 2 %).
- **Time step.** One hour, with the sun at mid-hour.

### Forcing — and why it is not the tower's

The campaign tower records are not public. The forcing is NASA POWER
hourly data for the campaign dates:

- **Radiation:** CERES SYN1deg, giving ``S↓``, direct normal, diffuse and
  ``L↓``.
- **Meteorology:** MERRA-2, giving 2-m temperature and humidity, the
  50-m wind and pressure.

The files are `test/data/nasa_power_*_hourly.csv`, with provenance
headers.

The beam is applied as ``(S↓ - S_{dif})/\sin h_\odot`` at the mid-hour
sun. This way the scene receives exactly the hourly global irradiance;
below 3° elevation the beam is counted as diffuse.

The satellite radiation is checked against the tower where the papers
allow it:

- **Seven clear BUBBLE days, all hours.** CERES
  ``S↓ + L↓ = 681.5`` W m⁻². The tower value, from Hamdi & Masson 2008
  Table 4 as ``Q^* + S↑ + L↑``, is 680 W m⁻² (asserted within 2 %).
- **Hamdi & Schayes (2005) windows.** CERES ``L↓`` is 366 / 338 W m⁻²
  (day / night) against the tower's 360 / 340. CERES ``S↓`` in the day
  window is 766 W m⁻² against 741 (+3.4 %).

The MERRA-2 temperature is a grid-box value **without the urban heat
island**:

- **BUBBLE.** On the seven clear days it is 2.55 K below the tower's
  18-m mean (Hamdi & Masson 2008, Table 3). BUBBLE therefore adds this
  one constant.
- **CAPITOUL.** No published tower mean was available for the three
  periods, so the grid value is used unchanged. The canyon air is taken
  equal to the forcing air, because there is no canyon-air model.

These two approximations act in opposite directions. The sensitivity
of the CAPITOUL means to the air temperature is measured below.

## CAPITOUL — road, wall and roof temperatures

**Canyons.** Three canyons were instrumented with infrared thermometers:

| canyon | H (m) | W (m) | H/W | street azimuth | walls facing |
|:--|--:|--:|--:|--:|:--|
| rue d'Alsace-Lorraine | 20.5 | 14.0 | 1.46 | 0° | 90°, 270° |
| rue de la Pomme | 20.5 | 10.0 | 2.05 | 120° | 210°, 30° |
| rue Rémusat | 16.8 | 10.0 | 1.68 | 60° | 150°, 330° |

H and W are from Moscicki & Voogt (2007), Table 2.1. The azimuths and
facing directions are from Pigeon et al. (2009), Table 3.

**Constructions and optics.** These are the TEB inputs of Pigeon et al.
(2009), Table 2:

- **Walls:** 30 cm red brick (``C = 1.58`` MJ m⁻³ K⁻¹,
  ``k = 1.15`` W m⁻¹ K⁻¹), α = 0.25, ε = 0.92.
- **Roofs:** 6 cm clay tile over 3 cm wood, α = 0.15, ε = 0.90.
- **Road:** 5 cm asphalt, 20 cm aggregate and 1 m gravel/soil, α = 0.08,
  ε = 0.95.

The deep-road temperature and the initial temperature are the mean air
temperature of each 46-day window. This is an assumption.

**Periods.** There are three independent runs. Each has 14 days of
spin-up, then the 32-day periods of Moscicki & Voogt (2007):

- 15 Jul–15 Aug 2004
- 15 Oct–15 Nov 2004
- 15 Jan–15 Feb 2005

**Observations.** The observations are from Moscicki & Voogt (2007),
Table 4.1, typed from the table text into
`validation/data/capitoul_surface_temperature_statistics.csv`. That table gives
the observed and TEB-modelled mean and standard deviation over all three
periods (about 4500 half-hourly samples), with TEB forced by the tower
itself. The observed series are:

- **Road:** the mean of the three roads.
- **Walls:** a weighted mean of the six walls.
- **Roof:** a single pitched tile roof on the Alsace canyon.

The model uses equal weights for the walls. Pigeon et al. state that
their weighting changes the statistics by at most 0.3 K.

**Results** (measured, 2304 hourly samples):

| °C | model mean ± sd | observed | TEB (tower-forced) | model − obs | TEB RMSE |
|:--|:--|:--|:--|--:|--:|
| roads | 15.84 ± 11.98 | 15.78 ± 9.38 | 15.14 ± 12.09 | +0.06 | 3.09 |
| walls | 15.27 ± 10.12 | 15.30 ± 9.11 | 15.98 ± 8.88 | −0.03 | 1.22 |
| roof  | 16.81 ± 13.53 | 12.88 ± 11.28 | 14.28 ± 13.20 | **+3.93** | 3.50 |

**Assertions.** The yardstick is the error of the reference model
forced by the campaign tower:

- **Means.** Each mean must be within TEB's RMSE. Roads and walls pass.
- **Roof.** The roof fails, at +3.93 K against 3.50 K. It is recorded
  with `@test_broken`.
- **Amplitude.** Each standard deviation must be within the largest
  model/observation ratio that TEB itself shows (1.29, roads). The
  measured ratios are 1.28, 1.11 and 1.20.
- **Ordering.** The observed ordering of the variability, roof > roads >
  walls, is reproduced.

**Orientation** (no per-wall observations are published, so these are
physics checks):

- **Alsace (north–south).** The east and west walls have the same mean
  (difference 0.00 K). In the summer composite the east wall peaks at
  10–11 UTC and the west wall at 14–15 UTC.
- **Pomme and Rémusat.** The equator-facing walls are warmer than the
  pole-facing ones: SSW − NNE = +0.62 K and SSE − NNW = +0.74 K.

**The match of the road and wall means is partly luck.** The means move
by +0.82 K (road), +0.85 K (walls) and +0.76 K (roof) for every +1 K of
forcing air temperature. The grid-box temperature lacks the urban heat
island: in Basel, on the seven clear summer days, it was 2.55 K below the
tower's 18-m mean (that difference mixes the heat island with any
reanalysis bias; the two cannot be separated here). There is also no
canyon-air model. The Hamdi & Masson (2008) Table 3 observations show
canyon air about 1 K warmer than the air above the roofs by day. An
uncorrected cold forcing and the missing canyon warming both bias the
same way: towards colder surfaces. The agreement to 0.1 K therefore
cannot be read as a skill of 0.1 K. What the comparison does establish:

- the mean is within the reference model's per-sample error;
- the variability is within the reference model's amplitude error;
- the orientation behaviour is physical.

**The roof is too warm, and why.** The TEB input roof (6 cm tile over
3 cm wood) sits directly on the building zone. Its U-value is
2.9 W m⁻² K⁻¹ with the ISO 6946 surface resistances. Over the summer nights (22–04 UTC) the modelled roof
stays at the air temperature (19.95 °C against 19.89 °C), fed by
15.8 W m⁻² of heat conducted up from the rooms. The observed roof "typically
dropped below 15 °C at night and above 40 °C during mid-afternoon" in
summer (Moscicki & Voogt 2007, §4.1.3). It is a pitched tile roof,
whereas TEB (and this case) assume a flat one; what lies under it is not
described in the sources used here. TEB with the same inputs also runs
warm (+1.40 K), and an independent SUM analysis in the same paper
suggests that the observed roof is not unusually cold.

A flat, thin roof straight onto a heated zone is the likely cause.
Replacing it with some other construction (e.g. a ventilated roof space)
would be a guess that no paper here supports, so it was not done.

## BUBBLE — the surface energy balance of Basel-Sperrstrasse

**Geometry and materials** (Hamdi & Schayes 2005, Tables 1 and 3):

- Buildings 14.6 m, street 11.23 m (H/W = 1.30), roofs 15 m wide
  (modelled 14.97 m; roof plan fraction 0.571).
- Two street directions, 67° and 157°, averaged with equal weights.
- Albedo 0.14 / 0.14 / 0.08 and emissivity 0.90 / 0.90 / 0.94 for wall /
  roof / road.
- Thermal diffusivity and heat capacity 6.7×10⁻⁷ m² s⁻¹ and
  1.4 MJ m⁻³ K⁻¹ for walls and roofs, 2.8×10⁻⁷ and 1.8 for the road.

The thicknesses are not published. The case uses 0.30 m for walls and
roofs and 0.50 m for the road. These are at least 2.2 diurnal damping
depths, so the diurnal surface response does not depend on them.

**Fluxes.** The fluxes are those of one repeat unit of the canyon array
per unit plan area:

- ``Q^*``, ``Q_H`` and ``ΔQ_S`` are the area-weighted facet ``R_n``,
  ``H`` and ``G``. ``G`` includes the heat conducted into the rooms.
- ``S↑ = S↓ - Σ A\,S_{abs}/A_{plan}`` and
  ``L↑ = L↓ - Σ A\,L_{net}/A_{plan}``.

**Observations**, in `validation/data/bubble_sperrstrasse_energy_balance.csv`:

- **Hamdi & Masson (2008), Table 4.** Means over the clear days 17, 18,
  19, 22, 23, 29 and 30 June 2002, with TEB's skill at the same site. The
  paper does not define its day/night split, so only its all-hours means
  are compared with. Its day (D) and night (N) RMSEs are, however, borrowed
  as the error scale for the Hamdi & Schayes windows below.
- **Hamdi & Schayes (2005), Table 2.** Clear days 17, 18, 23, 26 and
  30 June. Day is 08–16 UTC and night is 22–03 UTC.

**Results.** Measured over all hours (Hamdi & Masson 2008). The
tolerance is the RMSE of TEB-SBL forced by the tower:

| W m⁻² | model | observed | TEB-SBL | tolerance | |
|:--|--:|--:|--:|--:|:--|
| ``S↑`` | 29.7 | 34 | 31 | 4 | **fails by 0.3** (`@test_broken`) |
| ``L↑`` | 481.9 | 467 | 453 | 16 | pass |
| ``Q^*`` | 170.0 | 179 | 170 | 29 | pass |
| ``Q_H`` vs obs ``Q_H + Q_E`` | 148.6 | 155 | — | 49 | pass |
| ``ΔQ_S`` | 21.4 | 24 | 47 | 70 | pass |

Measured over the day and night windows (Hamdi & Schayes 2005). The
tolerance is TEB-SBL's D / N RMSE from Hamdi & Masson (2008), Table 4.
That is a proxy, not a like-for-like pass mark: those RMSEs belong to
different days and to day/night windows the paper does not define (its
observed daytime ``Q^*`` is 373 W m⁻², against 510 in the 08–16 UTC
window). Asserted: ``Q^*`` and ``S↑`` by day and night, ``L↑`` at night.

| | ``S↑`` | ``L↑`` | ``Q^*`` | ``Q_H/Q^*`` (obs: ``(Q_H+Q_E)/Q^*``) | ``ΔQ_S/Q^*`` |
|:--|:--|:--|:--|:--|:--|
| day, model / obs | 69.6 / 72 | **559.8 / 521** | 502.7 / 510 | 0.57 / 0.64 | 0.43 / 0.32 |
| night, model / obs | 0.0 / 3 | 417.5 / 420 | −79.3 / −82 | −0.58 / −0.40 | 1.58 / 1.30 |

**Storage hysteresis.** Hamdi & Masson (2008) observe two things: the
storage flux rises faster than ``Q_H`` in the morning and peaks earlier,
and it turns negative 1–2 h before ``Q^*``. In the model's clear-day
composite:

- ``ΔQ_S`` peaks at 09 UTC, ``Q^*`` at 11 UTC and ``Q_H`` at 12 UTC;
- ``ΔQ_S`` turns negative at 16 UTC and ``Q^*`` at 18 UTC.

At night the storage release exceeds the radiative loss
(``ΔQ_S/Q^* = 1.58``, observed 1.30), and it keeps ``Q_H`` positive, as
observed. All of these are asserted.

**What fails, and the estimates that go with it.** These are estimates,
not fits:

- **Daytime ``L↑`` (+39 W m⁻² against TEB's 15).** The modelled canopy is
  radiometrically too warm by day. The source area is 16 % vegetation,
  which is not modelled. At the air temperature of the window (28.7 °C)
  that vegetation would emit about 469 W m⁻² instead of the urban 560,
  which lowers ``L↑`` by about 15 W m⁻². The satellite ``S↓`` in the
  window is also 3.4 % above the tower's.
- **``S↑`` (−4.3 against ±4).** The canyon array reflects 9.2 % of
  ``S↓``. The vegetation at the package's park-class albedo (0.205) would
  add about 6 W m⁻².
- **Turbulent flux.** Without vegetation there is no ``Q_E``, so the
  model's ``Q_H`` is compared with the observed ``Q_H + Q_E`` (155), using
  the RMSE of ``Q_H`` (49) as the tolerance. Against the observed ``Q_H``
  alone (112) the model is +36.6 W m⁻², which is also inside 49 but is a
  real sensible-heat excess: the energy the missing vegetation would
  evaporate goes into ``Q_H`` instead. At night the model's
  ``Q_H/Q^* = -0.58`` against an observed ``Q_H/Q^* = -0.21``, i.e.
  about 46 against 17 W m⁻² upward; only its sign is asserted.

**Sensitivity to the temperature correction.** With separate day and
night offsets instead of one (+1.91 / +4.90 K, which assumes that "day"
in Hamdi & Masson Table 3 means sun up):

- night ``Q_H/Q^* = -0.28`` and ``ΔQ_S/Q^* = 1.28``, closer to the
  observed −0.21 and 1.30;
- all-hours ``L↑`` 482.6 and ``Q^*`` 169.3 W m⁻²;
- daytime ``L↑`` 557.7 W m⁻².

The single documented offset is what the test uses.

## Numerical sensitivity (CAPITOUL means and standard deviations)

| change | roads | walls | roof |
|:--|:--|:--|:--|
| baseline (1 h step) | 15.84 ± 11.98 | 15.27 ± 10.12 | 16.81 ± 13.53 |
| 30-min step | 15.81 ± 11.95 | 15.28 ± 10.11 | 16.79 ± 13.48 |
| 7 instead of 14 days of spin-up | 15.82 ± 11.97 | 15.27 ± 10.12 | 16.81 ± 13.53 |
| canyon 1.5 × longer | 15.82 ± 11.98 | 15.27 ± 10.11 | 16.81 ± 13.53 |
| 4 instead of 3 wall strips | 15.84 ± 11.98 | 15.28 ± 10.13 | 16.82 ± 13.53 |
| forcing air +1 K | 16.66 ± 11.98 | 16.12 ± 10.15 | 17.57 ± 13.54 |

The discretisation is converged far below the observation–model
differences. The forcing air temperature is not.

## What was not validated

- **Time series.** The campaign time series are not public. Hence there
  is no per-sample RMSE, bias by hour, or scatter against observations,
  only period statistics. The model's RMSE against observations is
  unknown.
- **Per-season and per-wall observations.** Moscicki & Voogt give these
  only as figures, and figures were not digitised. The seasonal
  statements in their text (TEB's winter road bias −2.53 K, the summer
  road over-estimate at noon) were not tested.
- **The CAPITOUL tower forcing, the canyon air temperature, and the
  anthropogenic heat flux against Pigeon et al. (2009).** The last was
  compared with the building energy inventory in #29.
- **Vegetation and latent heat** (BUBBLE's ``Q_E``).
- **Windows.** The walls here are opaque brick, as in TEB.
- **Hand-held and airborne infrared data.** The hand-held infrared
  samples in Qu (2011, Tables 6.3–6.5) and the airborne infrared images
  (Lagouarde et al. 2010; Hénon et al.) are single-instant brightness
  temperatures of individual sunlit and shaded patches. The canyon
  model's facet averages are not comparable with them.
- **Other papers.** Lemonsu et al. (2004), Marseille, is not BUBBLE or
  CAPITOUL and was not used. Rotach et al. (2005) and Christen & Vogt
  (2004) were not accessible, so nothing is quoted from them.

## Wrong expectations found while writing the test

- **The hand value of the canyon-wind check.** I wrote 0.28158 for
  ``\frac{2}{π}e^{-1/4}\ln(10/3)/\ln(25/3)``; the correct value is
  0.28154. This was an arithmetic slip in the expectation, not the code.
- **``S↑``.** It passed TEB's tolerance by 0.2 W m⁻² with the first
  forcing split, which gave the scene 1.2 % more shortwave than the
  hourly global irradiance. After that was corrected, it fails by
  0.3 W m⁻². The assertion was kept and is recorded as broken.

## References

- Hamdi, R. & Masson, V. (2008). Inclusion of a drag approach in the
  Town Energy Balance (TEB) scheme: offline 1-D validation in a street
  canyon. *J. Appl. Meteorol. Climatol.*, 47, 2627–2644.
- Hamdi, R. & Schayes, G. (2005). Validation of the Martilli's urban
  boundary layer scheme with measurements from two mid-latitude European
  cities. *Atmos. Chem. Phys. Discuss.*, 5, 4257–4289.
- Lemonsu, A., Grimmond, C. S. B. & Masson, V. (2004). Modeling the
  surface energy balance of the core of an old Mediterranean city:
  Marseille. *J. Appl. Meteorol.*, 43, 312–327.
- Masson, V. (2000). A physically-based scheme for the urban energy
  budget in atmospheric models. *Boundary-Layer Meteorol.*, 94, 357–397.
- Masson, V., Grimmond, C. S. B. & Oke, T. R. (2002). Evaluation of the
  Town Energy Balance (TEB) scheme with direct measurements from dry
  districts in two cities. *J. Appl. Meteorol.*, 41, 1011–1026.
- Moscicki, M. & Voogt, J. A. (2007). A comparison of modeled and
  observed urban surface temperatures in Toulouse, France. *AMS 7th
  Symposium on the Urban Environment*, paper J2.6.
- Pigeon, G., Moscicki, M. A., Voogt, J. A. & Masson, V. (2008).
  Simulation of fall and winter surface energy balance over a dense
  urban area using the TEB scheme. *Meteorol. Atmos. Phys.*, 102,
  159–171.
- Pigeon, G., Moscicki, M., Voogt, J. A. & Masson, V. (2009). Simulation
  of fall and winter surface energy balance over a dense urban area
  using the TEB scheme. *AMS 8th Symposium on the Urban Environment*,
  Phoenix, paper 3.6 (extended abstract of the study above; Tables 2–4
  are used here).
- Qu, Y. (2011). *Three-dimensional modeling of radiative and convective
  exchanges in the urban atmosphere.* PhD thesis, Université Paris-Est,
  §6.2.
- Rowley, F. B., Algren, A. B. & Blackshaw, J. L. (1930). Surface
  conductances as affected by air velocity, temperature and character of
  surface. *ASHVE Trans.*, 36, 429–446.
- NASA POWER hourly data (CERES SYN1deg, MERRA-2), NASA Langley Research
  Center POWER Project.
