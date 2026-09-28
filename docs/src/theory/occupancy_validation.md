# Occupancy schedules and BESTEST validation

A building's energy demand depends on how it is used as much as on how
it is built. People, lights and office equipment release heat on a
weekly rhythm, and the thermostat follows the same rhythm. Issue #33
adds these schedules to `src/BuildingEnergy.jl` and checks the whole
zone model (conduction, air node, infiltration, internal gains, ideal
loads) against the building-energy benchmark ANSI/ASHRAE Standard
140-2020 (BESTEST).

## 1. Schedules

A [`DayProfile`](@ref) is a piecewise-constant day in the form of an
EnergyPlus `Schedule:Compact` day: value ``v_i`` holds from breakpoint
``u_{i-1}`` to ``u_i`` hours (``u_0 = 0``, last ``u_n = 24``). A
[`WeeklySchedule`](@ref) holds a weekday, a Saturday and a
Sunday/holiday profile. Time is counted in seconds from 00:00 of day 1,
and `start_weekday` fixes the weekday of day 1.

A model step of length ``Δt`` should receive the *mean* of the
schedule over the step, not its value at one instant.
[`schedule_mean`](@ref) integrates the step function exactly:

```math
\bar v(t, t + Δt) = \frac{1}{Δt}\int_t^{t+Δt} v\,dt' ,
```

so a scheduled gain delivers exactly its scheduled energy for any step
length. The daily integral is the *equivalent full-load hours*
([`full_load_hours`](@ref)).

`DOE_MEDIUM_OFFICE_SCHEDULES` is transcribed from the U.S. DOE
commercial reference building *Medium office, new construction, ASHRAE
90.1-2004* (Deru et al. 2011), EnergyPlus file
`RefBldgMediumOfficeNew2004_Chicago.idf`: occupancy, lighting, plug
loads, heating and cooling setpoints, HVAC operation. Full-load hours per
week, summed by hand from the IDF and checked in the tests: occupancy
54.50 h, lighting 62.00 h, equipment 90.95 h.

## 2. Internal gains

[`InternalGains`](@ref) follows the EnergyPlus `People`, `Lights` and
`ElectricEquipment` objects. At time ``t`` (or averaged over a step):

| source | power | split |
|:--|:--|:--|
| people | ``n\,f_{occ}\,M`` | sensible ``S(M, T_{air})``, radiant fraction ``f_r`` of it; latent ``M - S`` |
| lighting | ``P_L A\,f_{light}`` | radiant + visible → surfaces; return air → lost; rest → air |
| equipment | ``P_E A\,f_{equip}`` | radiant → surfaces; latent; lost; rest → air |

The occupant sensible heat is the EnergyPlus correlation
([`occupant_sensible_heat`](@ref)),

```math
S = C_0 + C_1 M + C_2 M^2 + T(C_3 + C_4 M + C_5 M^2) + T^2(C_6 + C_7 M + C_8 M^2),
```

with ``T`` in °C, limited to ``[0, M]``. It is a fit to Carrier's (1965)
sensible gains at 350–1450 Btu h⁻¹ and 70–82 °F. The fit was
constrained to ``S = 0`` at 96 °F and ``S = M`` at 30 °F. At those
anchors the implementation gives ``S/M ≤ 0.022`` and
``|S - M|/M ≤ 0.004`` over the fitted range. At 120 W and 24 °C,
``S = 69.5`` W. The coefficients come from the EnergyPlus source
(`InternalHeatGains.cc`). The Engineering Reference prints two of them
slightly differently (``C_0``, ``C_5``); the difference is below
0.01 W.

The split conserves energy exactly: people + lighting + equipment =
convective + radiative + latent + lost ([`InternalGainRates`](@ref)).
`ZoneForcing(r::InternalGainRates; …)` passes the convective and
radiative parts to the zone. Because the mix of people, lights and
equipment changes over the day, the radiative fraction changes every
step. `ZoneForcing` now accepts a per-step `radiative_fraction` that
overrides the zone's.

A [`ThermostatSchedule`](@ref) holds heating and cooling setpoint
schedules and an optional availability schedule.
[`scheduled_hvac`](@ref) returns the `IdealLoadsHVAC` for time ``t``;
while the system is unavailable its capacities are zero and the zone
floats.

### The periodic-mean theorem

With the outdoor temperature constant and the occupant split frozen, the
zone is a linear time-invariant system, ``x_{n+1} = A x_n + B u_n``. Its
periodic response to weekly gains has the same mean as its steady
response to the week-mean gains. This holds exactly for the discrete
θ-scheme too: averaging the recurrence over one period gives
``\bar x = A \bar x + B \bar u``. The test drives the massive test box
with the DOE office gains. The week-mean air temperature of the periodic
response matches the steady temperature under the mean gains to
1.5 × 10⁻¹² K. A scheduled office week conserves the zone's heat to a
relative 9 × 10⁻¹³.

## 3. Validation against ASHRAE Standard 140-2020

**What was compared.** The reference is the example results of six
whole-building programs (BSIMAC, CSE, DeST, EnergyPlus, ESP-r, TRNSYS)
in Informative Annex B8 of Standard 140-2020. The Standard's weather is
Denver TMY3. These are **program-to-program comparisons, not
measurements**. No measured data from an instrumented building was
available, so the phase target "measured demand vs simulated" is not
met by this validation. The Standard itself says the min–max ranges are
not acceptance criteria. They are used here because they span what
established tools compute for the same fully specified building.

**Cases.** The in-depth cases of §5.2.3 that have no transparent
windows (window optics are issue #27):

| case | what it adds |
|:--|:--|
| 220 | base: lightweight 8 × 6 × 2.7 m cell, 12 m² high-conductance south element, "20,20" thermostat |
| 230 | infiltration 1 ach |
| 240 | internal gains 200 W (60 % radiative) |
| 250 | exterior solar absorptance 0.9 |
| 400 | 220 with the "20,27" thermostat |
| 395 | 400 without the high-conductance element (solid conduction) |
| 410 | 400 + infiltration 0.5 ach |
| 420 | 410 + internal gains 200 W |
| 430 | 420 + exterior absorptance 0.6 |
| 800 | 430 in high-mass construction |

**Model inputs.** The inputs follow the Standard's normative tables:
materials, the 140-2020 surface coefficients, and infiltration as the
Table 5-3 mass flow (1.0156 kg m⁻³ × 1015 J kg⁻¹ K⁻¹). The zone model
lumps interior long-wave exchange into the surface-to-air coefficient,
so it uses the Standard's interior *combined* coefficients (Table 5-9:
walls 1.8, ceiling 1.7, floor 3.7, high-conductance element 4.5
W m⁻² K⁻¹). For the exterior, two options the Standard allows are
compared:

* **explicit long-wave** (`FacetElement`, the reported result): an
  `UrbanFacet` outer face with the Standard's convective coefficients
  (walls 11.9, roof 14.4, raised floor 0.8, element 8.0), emissivity
  0.9, and long-wave irradiance ``F_{sky} L_{sky} + (1 - F_{sky})σ
  T_{out}^4``. Here ``L_{sky}`` is the weather file's horizontal
  infrared and the ground is at the outdoor air temperature
  (§5.2.1.8.2.2.1);
* **constant combined** (`EnvelopeElement`): the Standard's combined
  exterior coefficients (21.6 / 21.8 / 5.2 / 17.8) with a sol-air
  source.

Plane-of-array irradiance comes from the Perez et al. (1990) model with
ground reflectance 0.2, in the test driver `test/ashrae140_driver.jl`.
The zone is stepped six times per hour. Going to 12 steps per hour
changes every annual heating load and every peak by less than 0.04 %,
and the annual cooling loads by at most 0.12 % (Case 400, whose annual
cooling is only 0.025 MWh). Each case gets a 31-day warm-up
before the year.

**Results (explicit long-wave).**

| check | inside the reference range |
|:--|:--|
| incident solar, 5 planes (Table B8-13) | 5 / 5 (isotropic sky: 2 / 5) |
| annual heating + cooling (Tables B8-1, B8-2) | 19 / 20; the miss, Case 395 heating 4.538 MWh vs 4.565–5.145, is 0.6 % of the mean below the range |
| annual peaks (Tables B8-3, B8-4) | 16 / 20; worst 7.9 % of the mean outside (Case 250 peak cooling) |
| internal-gain sensitivity 240−220, 420−410 (4 quantities each) | 8 / 8 |
| other sensitivities (230−220, 250−220, 400−395, 410−400, 430−420, 800−430) | 13 / 24 |

The internal-gain differences are the direct check of this issue's
physics:

| Δ | quantity | model | range | EnergyPlus |
|:--|:--|--:|:--|--:|
| 420−410 | annual heating (MWh) | −1.211 | −1.231 … −1.175 | −1.175 |
| 420−410 | annual cooling (MWh) | +0.093 | +0.074 … +0.132 | +0.095 |
| 420−410 | peak heating (kW) | −0.180 | −0.200 … −0.178 | −0.185 |
| 420−410 | peak cooling (kW) | +0.205 | +0.197 … +0.212 | +0.205 |
| 240−220 | annual cooling (MWh) | +0.357 | +0.326 … +0.484 | +0.369 |

**Where it misses, and why.**

The 11 misses among the other sensitivities are, in full: 250−220
annual heating and peak cooling, 430−420 peak cooling (solar); 400−395
annual and peak heating (high-conductance element); 800−430 annual
cooling and both peaks (thermal mass); and 230−220 annual and peak
cooling and 410−400 peak cooling (infiltration: +0.408 MWh vs
0.379–0.399, +0.731 kW vs 0.620–0.726, +0.259 kW vs 0.168–0.252 — the
infiltration adds slightly too much cooling). The causes below are
likely, not demonstrated.

1. *Solar-driven peaks.* Case 250 peak cooling and the 430−420 and
   250−220 peak differences are low, and so is the heating saved by
   the dark exterior (250−220: −1.476 MWh vs −1.805 … −1.556). Constant exterior convective
   coefficients represent the annual mean wind (4.02 m s⁻¹, per the
   Standard). The reference programs compute wind- and
   temperature-dependent coefficients, which are lower on calm, sunny
   afternoons.
2. *The high-conductance element.* The 400−395 heating difference is
   2.02 MWh against 1.34–1.90, so the element loses too much heat. The
   likely cause is the lumped interior: the element's long-wave exchange
   with the room surfaces goes through the air node.
3. *Thermal mass.* The 800−430 reductions in cooling and in both peaks
   are weaker than the reference. The Standard's interior combined
   coefficient for walls (1.8 W m⁻² K⁻¹) couples the concrete block to
   the air weakly. The reference programs also couple it radiatively to
   the other surfaces.

With **constant combined exterior coefficients** only 28 of 72 checks
fall inside the range. That variant misses the sky's long-wave deficit:
the sky is on average about 16 K colder than the air. It therefore
gives less heating in every case and more cooling in every
low-absorptance case, and the tests assert this. The Standard's own
Case 450 (constant coefficients) is likewise below Case 600 in heating,
for all six reference programs (Table B8-9: −0.002 to −0.672 MWh).

## 4. An office cell with the DOE schedules (illustration)

This is an illustration, not a validation. It uses the Case 430 cell in
the Denver weather, the DOE office gains (1 occupant per 18.58 m²,
10.76 W m⁻² lighting and plugs, return-air fraction 0) and the same
week-mean gain (532 W) in three runs:

| gains | thermostat | heating (MWh) | sensible cooling (MWh) |
|:--|:--|--:|--:|
| constant at the week mean | 21 / 24 °C always | 4.72 | 2.29 |
| DOE schedules | 21 / 24 °C always | 5.27 | 2.87 |
| DOE schedules | DOE setpoints + HVAC availability | 2.74 | 2.55 |

The first two runs deliver the same energy as internal gains, but
scheduling changes the demand. The daytime peak gain is partly removed
by cooling, and the unoccupied nights and weekends then need heating
(+12 % heating, +25 % cooling). This shows why a constant-gain
building model misjudges demand. Setback and HVAC availability then
halve the heating.

## References

- ANSI/ASHRAE (2020). Addendum a to ANSI/ASHRAE Standard 140-2017,
  *Standard Method of Test for the Evaluation of Building Energy
  Analysis Computer Programs*: §5.2.1–5.2.3, Tables 5-2, 5-7, 5-9,
  5-27, 5-36 to 5-39, A1-1; Informative Annex B8 (example results,
  `RESULTS5-2A.xlsx`).
- NREL BESTEST-GSR (github.com/NatLabRockies/BESTEST-GSR): the weather
  file `725650TYCST.epw` and the results workbook.
- Deru, M. et al. (2011). *U.S. Department of Energy Commercial
  Reference Building Models of the National Building Stock.*
  NREL/TP-5500-46861.
- EnergyPlus Engineering Reference, *Zone internal gains — heat gain
  from people*; `src/EnergyPlus/InternalHeatGains.cc`.
- Perez, R., Ineichen, P., Seals, R., Michalsky, J. & Stewart, R.
  (1990). Modeling daylight availability and irradiance components from
  direct and global irradiance. *Solar Energy*, 44, 271–289.
- Kasten, F. & Young, A. T. (1989). Revised optical air mass tables and
  approximation formula. *Applied Optics*, 28, 4735–4738.
