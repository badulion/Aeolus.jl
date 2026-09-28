# HVAC demand and exhaust heat

An air-conditioned building does not remove heat from the city. It
moves the heat outdoors and adds the compressor work: every joule of
cooling leaves the condenser as ``1 + 1/COP`` joules. The hotter the
street, the larger the cooling load and the lower the COP. Both push
more heat into the street. This is the air-conditioning feedback that
amplifies urban heat islands (Salamanca et al. 2014; de Munck et al.
2013). `src/BuildingEnergy.jl` (issue #32) closes this loop for the
zone model of [Building energy: indoor air and thermal mass](building_energy.md).

## 1. Ideal loads from the affine zone step

`step_zone!` solves the zone implicitly. Every column is stepped twice,
at the old air temperature ``x_0`` and at ``x_0 + 1`` K, so the air
balance is one linear equation

```math
\mathcal{L}\,(x - x_0) = R + Φ_{hvac},
```

where ``\mathcal{L}`` collects ``C_a/Δt``, the surface, glazing and
ventilation conductances, and ``R`` collects everything else. The
free-floating end-of-step temperature is ``x_f = x_0 + R/\mathcal{L}``.
An `IdealLoadsHVAC` then supplies

```math
Φ_{req} = \begin{cases}
\mathcal{L}(T_{heat} - x_0) - R & x_f < T_{heat},\\
\mathcal{L}(T_{cool} - x_0) - R & x_f > T_{cool},\\
0 & \text{otherwise},
\end{cases}
\qquad Φ_{hvac} = \mathrm{clamp}(Φ_{req}, -Q_{cap}, Q_{heat,cap}).
```

This is exact, not iterated. When the capacity suffices, the air
temperature ends the step on the setpoint to round-off. The shortfall
``Φ_{req} - Φ_{hvac}`` is reported as `Q_unmet`. ISO 13790 §C.4.2 does
the same for the 5R1C model by interpolating between a free step and a
10 W m⁻² step (`iso13790_ideal_load!`). Both are linear, so the
interpolation is exact there too.

**Humidity.** When `ZoneForcing` carries the outdoor specific humidity,
the zone vapour balance is advanced implicitly after the temperature:

```math
M\frac{q^{n+1}-q^n}{Δt} = M n (q_{out} - q^{n+1}) + \frac{Φ_{lat}}{L_v} + W_{hvac},
```

with ``M = ρV`` and ``n`` the total outdoor-air change rate. Heat
recovery is sensible only, and hygroscopic buffering is not modelled.
With `dehumidify = true`, ``W_{hvac} < 0`` removes the vapour that would
raise the relative humidity above `RH_max` at the new air temperature.
The coil load is then ``Q_{cool} = Q_s + L_v |W_{hvac}|``. Capacity goes
to the sensible load first. Re-heat is not modelled.

## 2. Equipment performance

Cooling is a direct-expansion unit in the DOE-2 / EnergyPlus
`Coil:Cooling:DX:SingleSpeed` form. Two biquadratic modifiers,
normalised to 1 at the AHRI 210/240 "A" point (26.7 °C dry bulb /
19.4 °C wet bulb indoors, 35 °C outdoors), scale the rated values:

```math
COP = \frac{COP_{rated}}{EIR_{FT}(T_{wb,i}, T_{c,i})},\qquad
Q_{cap} = Q_{rated}\,CAP_{FT}(T_{wb,i}, T_{c,i}).
```

Here ``T_{wb,i}`` is the thermodynamic wet bulb of the zone air, from
`MoistThermodynamics.wet_bulb_temperature`. ``T_{c,i}`` is the air
entering the condenser. For an air-cooled condenser it is the outdoor
dry bulb. For an evaporatively cooled condenser it is (EnergyPlus
Engineering Reference)

```math
T_{c,i} = T_{wb,o} + (1-ε)(T_{db,o} - T_{wb,o}),\qquad ε = 0.9 \text{ by default}.
```

Two published curve sets are shipped, both copied verbatim from the
EnergyPlus test files. `DX_CURVES_WINDOW_AC` comes from
`WindACAuto.idf` and `DX_CURVES_PACKAGED_AC` from `HeatPumpAuto.idf`.
The operating point is taken from the start-of-step state. Part-load
degradation and fan power are not modelled.

Heating can be a boiler (fuel ``Q/η``, ``η = 0.9`` by default),
electric resistance, or a heat pump with a constant ``COP_h``.

## 3. Exhaust heat

The plant obeys the first law:

```math
Q_{exh,s} + Q_{exh,l} = P_{heat} + P_{cool} + Q_{cool} - Q_{heat}.
```

- **Cooling.** The condenser rejects ``Q_{cool}(1 + 1/COP)``. An
  air-cooled condenser rejects all of it as sensible heat. An
  evaporative condenser or cooling tower rejects a fraction
  `latent_fraction` (default 0.85) as evaporated water,
  ``W_{evap} = Q_{exh,l}/L_v``. The 0.85 is the cooling-tower rule of
  thumb: evaporation of 0.00085 of the circulating flow per °F of range,
  against 0.001 if every joule went into evaporation. Published values
  span 60–95 %.
  This split is not the one EnergyPlus uses. The COP follows the
  EnergyPlus pre-cooling-pad condenser, whose water use is
  ``\dot m_{cond}\,Δw`` across the pad and scales with the condenser air
  flow, not with the rejected heat. The latent/sensible split follows the
  wet-tower rule. So ``W_{evap}`` is an estimate within that band. The
  total ``Q_{exh,s} + Q_{exh,l}`` does not depend on the split.
- **Heating.** A boiler's flue loss ``Q(1/η - 1)`` is sensible exhaust.
  A heat pump's ``-Q(1 - 1/COP_h)`` is negative: it cools the outdoor
  air.

`HVACBudget` reports these terms separately so that the canopy coupling
(#29, [Anthropogenic heat from the building energy model](anthropogenic_heat.md))
can place them: `anthropogenic_heat_maps` puts them at the building's roofs.

## 4. The feedback loop

Take a canyon air volume exchanging heat with the rural air ``T_r``
through a conductance ``K``, with background heating ``Q_0``. Surround
it with a zone of heat-loss coefficient ``H`` and internal gain ``G``,
cooled to ``T_{set}`` by an air-cooled unit at constant COP. The steady
canyon is

```math
K(T_c - T_r) = Q_0 + (1 + 1/COP)\bigl(H(T_c - T_{set}) + G\bigr)
\;\Rightarrow\;
T_c = \frac{T_r + [Q_0 + (1+1/COP)(G - H T_{set})]/K}{1 - g},\quad
g = \frac{(1 + 1/COP) H}{K}.
```

The exhaust-induced warming is therefore amplified by ``1/(1-g)`` over
the open-loop estimate. A temperature-dependent COP adds
``Q_{cool}\,\partial(1/COP)/\partial T_c`` to the loop gain.

## 5. Validation (measured)

All numbers are from `test/era5_hvac.jl`.

| Check | Reference | Measured |
|:--|:--|:--|
| DX curves at the AHRI A point | 1 by construction of the published fits | window: CAP 0.999998, EIR 1.00015; packaged: CAP 0.99948, EIR 1.00085 |
| COP(46.1 °C)/COP(35 °C) | two independently fitted EnergyPlus units | 0.7736 and 0.7728 |
| COP(27.8 °C)/COP(35 °C) (AHRI B/A) | same | 1.1524 and 1.1459 |
| Ideal load, massless box | ``H ΔT + Φ`` | rel. error < 1e-12 |
| Ideal load, massive box, 25 days constant | ``H ΔT`` = 1246.136 W | 1246.136 W (rtol 1e-6) |
| Setpoint holding, 6 hot/cold days, 600 s steps | exact | max \|T − T_set\| = 0 K while on; 215 heating / 351 cooling / 298 dead-band steps |
| Zone energy conservation with HVAC | ``ΔE = ∫ \text{sources}`` | 3.0e-4 J of 3.85e9 J gross |
| Air-cooled exhaust | ``Q_{cool}(1 + 1/COP)`` | 1e-14; 1.337 ``Q_{cool}`` at 35 °C |
| Plant first law, mixed 2-day run | exact | < 1e-13 relative |
| Dehumidified steady state | ``M n (q_{out} - q_{max}) + Φ_{lat}/L_v`` | rtol 1e-9; RH = 0.600, SHR 0.502 |
| Humidity relaxation | implicit factor ``(1 + nΔt)^{-k}`` | rtol 1e-12 |
| AC–canyon loop, constant COP | analytic ``T_c`` above | 312.3173 K, rtol 1e-10; amplification 1.3621 = 1/(1−g), g = 0.2659 |
| Same, with the window-AC COP curve | vs fixed COP 3.098 | +5.1 % more exhaust warming (6.41 vs 6.10 K) |
| Resolved zone vs ISO 13790 5R1C cooling need (day 5, heavy box) | same construction and forcing | 35.09 vs 30.32 kWh (ratio 1.157); peak 2298 vs 1900 W |

Test expectations that proved wrong: the capacity modifier is not
monotone in condenser temperature over the whole fitted range. The
packaged-unit fit peaks at 19.6 °C, where
``c_4 + 2c_5 y + c_6 x = 0``. That is a property of the published fit,
so the monotonicity check now starts at 20 °C.

The 5R1C bound (ratio in 0.8–1.25) is an envelope set after measuring,
not a physics expectation. A likely cause of the higher resolved load,
not isolated by a test, is where the gains attach. The resolved model
puts all transmitted solar gain on interior surfaces that release it to
the air within hours. 5R1C sends part of it into its single, slow mass
node.

**Not validated against observations.** No metered building or
measured COP data were used. The equipment behaviour is that of the
published EnergyPlus curves, and the feedback is checked against its
own linear theory. The phase target, simulated against measured demand
for an instrumented building, belongs to #33.

## References

- EnergyPlus Engineering Reference, *Coils*: single-speed electric DX air cooling coil (EIR and capacity modifiers; condenser options). EnergyPlus test files `WindACAuto.idf`, `HeatPumpAuto.idf` (NREL/EnergyPlus).
- ANSI/AHRI Standard 210/240, rating conditions A and B.
- ASHRAE *Fundamentals* (2017), ch. 1 (psychrometrics, thermodynamic wet bulb).
- ISO 13790:2008, §C.4.2 (heating and cooling need of the simple hourly method).
- Bueno, B., Pigeon, G., Norford, L. K., Zibouche, K. & Marchadier, C. (2012). Development and evaluation of a building energy model integrated in the TEB scheme. *Geosci. Model Dev.* 5, 433-448.
- Salamanca, F., Krpo, A., Martilli, A. & Clappier, A. (2010). A new building energy model coupled with an urban canopy parameterization for urban climate simulations — part I. *Theor. Appl. Climatol.* 99, 331-344.
- Salamanca, F., Georgescu, M., Mahalov, A., Moustaoui, M. & Wang, M. (2014). Anthropogenic heating of the urban environment due to air conditioning. *J. Geophys. Res. Atmos.* 119, 5949-5965.
- de Munck, C., Pigeon, G., Masson, V., Meunier, F., Bousquet, P., Tréméac, B., Merchat, M., Poeuf, P. & Marchadier, C. (2013). How much can air conditioning increase air temperatures for a city like Paris, France? *Int. J. Climatol.* 33, 210-227.
- Primo, J. *Cooling Towers — Basic Calculations*, PDHonline course M374 (evaporation rule of thumb; 60–95 % of heat removed by evaporation, after Perry's *Chemical Engineers' Handbook*).
