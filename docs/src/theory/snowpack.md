# Layered snowpack

A seasonal snowpack is a porous ice matrix that is, at once, a thermal
insulator, a slowly flowing viscous material and a reservoir of latent
heat. `src/Snowpack.jl` represents it the way CLM, SNOWPACK and Crocus
do: a column of layers, each with ice mass ``w_{ice}``, liquid mass
``w_{liq}`` (kg m⁻²), thickness ``Δz`` and temperature ``T``, numbered
from the surface down. `LayeredSnowpack` implements the Era V
`AbstractSurface` interface, so its turbulent fluxes come from
`SurfaceExchange` like every other surface.

## 1. Settling (compaction)

Mass is conserved while a layer settles, so the density follows from the
fractional compaction rate

```math
C_R = \frac{1}{Δz}\frac{\partial Δz}{\partial t} = -\frac{\partial \ln ρ}{\partial t} .
```

**Overburden.** A layer under a load creeps like a very viscous fluid,
``C_R = -σ/η``. The load is the mass above plus half the layer's own,
``P_s = w_i/2 + Σ_{j<i} w_j``. Two viscosity laws are provided:

| law | viscosity | load |
|:--- |:--- |:--- |
| Anderson (1976), CLM4 | ``η = η_0 \exp[c_5 (T_f - T) + c_6 ρ_{ice}]``, ``η_0 = 9·10^5`` kg s m⁻², ``c_5 = 0.08`` K⁻¹, ``c_6 = 0.023`` m³ kg⁻¹ | ``P_s`` (kg m⁻²) |
| Vionnet et al. (2012), Crocus | ``η = f_1 f_2 η_0 (ρ/c_η) \exp[a_η (T_f - T) + b_η ρ]``, ``η_0 = 7.62237·10^6`` Pa s, ``a_η = 0.1`` K⁻¹, ``b_η = 0.023`` m³ kg⁻¹, ``c_η = 250`` kg m⁻³ | ``σ = g P_s`` (Pa) |

Both make cold and dense snow stiff: ten kelvin of cooling slows
Crocus settling by ``e^{-1}``, and every 100 kg m⁻³ of density by
``e^{-2.3}``. In Crocus ``f_1 = 1/(1 + 60\, w_{liq}/(ρ_w Δz))`` softens wet
snow, and ``f_2`` stiffens angular (depth-hoar) grains; ``f_2`` needs the
grain size, so it is a hook of the metamorphism scheme (default 1).

The Crocus law has a closed-form solution under constant load and
temperature, which the tests use:

```math
\frac{dρ}{dt} = K e^{-b_η ρ},\quad K = \frac{σ c_η}{f_1 f_2 η_0 e^{a_η (T_f-T)}}
\;\Longrightarrow\;
e^{b_η ρ(t)} = e^{b_η ρ_0} + b_η K t .
```

**Destructive metamorphism.** New snow also settles as its dendrites
break down, independently of the load (Anderson 1976; CLM4 Eq. 7.40):

```math
C_{R1} = -c_3 c_1 c_2 \exp[-c_4 (T_f - T)],
```

``c_3 = 2.777·10^{-6}`` s⁻¹, ``c_4 = 0.04`` K⁻¹, ``c_1 = 1`` below
``ρ_{ice} = 100`` kg m⁻³ and ``e^{-0.046(ρ_{ice}-100)}`` above, ``c_2 = 2``
for wet snow. At the melting point it alone halves fresh snow in
``\ln 2/c_3 = 2.9`` days. It is part of the Anderson law by default and
can be added to the Crocus law (`Vionnet2012Compaction(destructive =
DestructiveMetamorphism())`).

**Melt** thins a layer in proportion to the ice lost, at constant density
— CLM's ``C_{R3}`` applied exactly. No layer is compacted beyond the
density of ice.

The layer ODE is integrated in ``\ln Δz`` with the explicit midpoint rule,
sub-stepped so no layer changes by more than 1 % per sub-step.

## 2. Fresh snow

New snow arrives at the density of Anderson (1976),
``50 + 1.7 (T_a - T_f + 15)^{3/2}`` kg m⁻³ between ``T_f - 15`` and
``T_f + 2``, or of Crocus (Pahaut 1976; Vionnet et al. 2012 Eq. 1),
``109 + 6 (T_a - T_f) + 26 U^{1/2}`` (at least 50), which adds wind
packing. Its temperature is ``\min(T_a, T_f)``; it is mixed into the top
layer conserving enthalpy.

## 3. Heat conduction and the surface energy balance

```math
C_i \frac{dT_i}{dt} = F_{i-1/2} - F_{i+1/2},\qquad
F_{i+1/2} = \frac{T_i - T_{i+1}}{Δz_i/2k_i + Δz_{i+1}/2k_{i+1}},
```

with ``C_i = w_{ice} c_{ice} + w_{liq} c_w`` and harmonic-mean interface
conductances, solved θ-implicitly (tridiagonal; θ = 1 by default, ½ for
Crank–Nicolson). The effective conductivity ``k(ρ)`` is a choice of

| law | ``k`` (W m⁻¹ K⁻¹), ρ in kg m⁻³ | ``k(300)`` |
|:--- |:--- |:--- |
| Yen (1981), Crocus | ``k_{ice} (ρ/ρ_w)^{1.88}`` | 0.23 |
| Sturm et al. (1997) | ``0.138 - 1.01ρ' + 3.233ρ'^2``, ``ρ' = ρ/1000`` (``0.023 + 0.234ρ'`` below 0.156) | 0.13 |
| Jordan (1991), CLM | ``k_{air} + (7.75·10^{-5}ρ + 1.105·10^{-6}ρ^2)(k_{ice} - k_{air})`` | 0.29 |
| Calonne et al. (2011) | ``2.5·10^{-6}ρ^2 - 1.23·10^{-4}ρ + 0.024`` | 0.21 |

The factor-two spread at a given density is physical — conduction
depends on the bond network of the ice, not on ρ alone — and is one of
the main sources of spread between snow models.

**Upper boundary.** The skin temperature is the temperature of the (thin,
2 cm) top layer. The surface energy balance
``G = R_n - H - LE`` is evaluated by `SurfaceExchange.turbulent_fluxes`
at ``T_1`` and ``T_1 + δT`` and linearised,
``G(T_1) ≈ G_0 + (∂G/∂T)(T_1 - T_1^n)``; the linearisation enters the
implicit matrix, so the stiff coupling of a thin top layer to radiation
and turbulence is stable at hourly steps. Every returned flux is updated
to the same time-weighted skin temperature, so the fluxes handed to the
atmosphere satisfy ``R_n = H + LE + G`` exactly and ``G`` is exactly the
heat that entered the pack. Latent heat is that of sublimation and the
skin humidity is saturation over ice. If the pack sublimates away
within a step, ``E`` and ``LE`` are cut to the mass actually removed and
the unspent latent heat stays in ``G`` and passes to the substrate, so the
atmosphere never receives vapour the pack did not have.

**Lower boundary.** Either a fixed snow/ground interface temperature
(`SnowGroundTemperature`) or a prescribed flux (`SnowBottomFlux`).

**Melt.** A layer left above ``T_f`` melts
``\min(C(T - T_f)/L_f,\, w_{ice})`` and returns to ``T_f``; a surplus
passes downward, and from the bottom layer into the ground.

## 4. Layers

Target thicknesses grow geometrically from the surface,
``Δz_i = 0.02 \cdot 1.6^{i-1}`` m, up to 10 layers. Layers thinner than
half their target (or lighter than 0.01 kg m⁻²) merge with their thinner
neighbour, layers thicker than twice their target split, and the
thinnest pair merges when there are too many. Merging conserves ice,
water, depth and enthalpy exactly (the merged temperature is the
heat-capacity-weighted mean).

## 5. Conservation

With enthalpy referred to liquid water at the melting point,
``h = w_{ice}[c_{ice}(T - T_f) - L_f] + w_{liq} c_w (T - T_f)``, the
`SnowBudget` of a pack closes

```math
ΔM = \text{snowfall} + \text{rain} - \text{sublimation} - \text{runoff},\qquad
ΔH = ∫G\,dt - ∫F_{base}\,dt + \text{advected} - \text{runoff enthalpy}
```

to round-off over any sequence of steps (tested over 12 days of storms,
radiative cooling, melt, rain and hoar deposition: residuals
``2·10^{-13}`` kg m⁻² and ``1.4·10^{-7}`` J m⁻² on ``4.6·10^7`` J m⁻²).

## 6. Hooks for later phases

| hook | default | extended by |
|:--- |:--- |:--- |
| `AbstractSnowMetamorphism`: `snow_metamorphism!`, `viscosity_grain_factor` | layers age, grains fixed, ``f_2 = 1`` | #35 grain metamorphism |
| `AbstractSnowAlbedo`: `snowpack_albedo` | constant 0.8 | #35 albedo ageing (delegating to `SurfaceOptics.SnowSurface`) |
| `AbstractSnowLiquidWater`: `snow_liquid_water!` | all liquid runs off at once | #36: `SnowBucketPercolation`, `SnowGravityDrainage` (Sect. 8) |

The pack books the enthalpy carried off by whatever runoff a
liquid-water scheme returns, so conservation holds for any scheme that
conserves enthalpy internally.

## 7. Validation

| check | measured | reference |
|:--- |:--- |:--- |
| Destructive settling, 3 temperatures, 2 d | max rel. error 2.2e-16 | ``ρ_0 e^{c_3 e^{-c_4ΔT} t}`` |
| Crocus settling, constant load, 5 d | max rel. error 5e-5 (223.23 vs 223.24 kg m⁻³) | closed form above |
| Anderson overburden, 10 d | implied time 9.99978 d | quadrature of ``∫e^{c_6ρ}/ρ\,dρ`` |
| Insulated cosine mode, 1 e-fold | amplitude error 3.3e-4 | ``e^{-κπ^2 t/H^2}`` |
| Surface step on deep pack, 2 d | max error 0.0009 K of 10 K | erf solution |
| Two-density steady state | flux to 1e-15 | series resistance |
| Seasonal bulk density (synthetic forcing) | −3 to −17 % (Anderson), −5 to −19 % (Vionnet) | Sturm et al. (2010) alpine climatology |

The last line is the only comparison with observations, and it is a
comparison with a climatology: Sturm et al. (2010, Eq. 6, Table 4) fit
bulk density against depth and day of year for 25 688 depth–density–SWE
measurements from the US, Canada and Switzerland; the local natural
variation of bulk density they quote (after Jonas et al. 2009) is 7–23 %. A
synthetic dry season (500 mm SWE evenly from 1 November to 31 March, air
−5 ± 3 °C) gives bulk densities 3-19 % below the alpine curve with
either compaction law, the gap growing through the season. No SnowMIP
site forcing is used here; site validation needs the albedo and
liquid-water physics of #35 and #36.

## 8. Liquid water: retention, refreezing, percolation

Meltwater and rain enter the top layer as liquid. `InstantSnowRunoff`
(the default, kept so that the dry-season results above are unchanged)
drains it all within the step. Two schemes keep it in the pack.

**Irreducible water.** Snow holds liquid against gravity up to an
irreducible volumetric content ``θ_r(θ_i)``, with
``θ_i = w_{ice}/(ρ_{ice}Δz)``. `ColeouLesaffre1998Retention` is the
laboratory drainage result of Coléou & Lesaffre (1998) in the form used
by SNOWPACK (Wever et al. 2014, Eq. 1):

```math
θ_r = \begin{cases}
0.0264 + 0.0099\,(1-θ_i)/θ_i, & θ_i ≤ 0.23\\
0.08 - 0.1023\,(θ_i - 0.03), & 0.23 < θ_i ≤ 0.812\\
0, & θ_i > 0.812
\end{cases}
```

(0.062 at 200 kg m⁻³, 0.050 at 300, 0.038 at 400).
`PoreFractionRetention(fraction = f)` is ``θ_r = f(1-θ_i)``: Crocus's
5 % of the pore volume (Vionnet et al. 2012, Eq. 19) or CLM's
``S_r = 0.033``. For 0.1 m of 300 kg m⁻³ snow they hold 4.96 and
3.36 kg m⁻² respectively.

**Refreezing.** Liquid water in a layer colder than ``T_f`` refreezes at
constant enthalpy. With cold content ``C(T_f - T)``
(``C = w_{ice}c_{ice} + w_{liq}c_w``), everything freezes if
``C(T_f-T) ≥ w_{liq}L_f`` and the layer warms to
``T_f - (C(T_f-T) - w_{liq}L_f)/C'``; otherwise ``C(T_f-T)/L_f`` freezes
and the layer ends at ``T_f``. The ice fills pores: the thickness does
not change (Vionnet et al. 2012, Sect. 3.10). Water that arrives in a
layer is first mixed in at constant enthalpy, then refrozen. The
refrozen mass is booked in `SnowBudget.refreeze`, so that
``Δ\sum w_{liq} = \text{melt} + \text{rain} - \text{refreeze} - \text{runoff}``.

**Bucket** (`SnowBucketPercolation`, Crocus and CLM). From the top down,
each layer refreezes what its cold content allows, retains
``ρ_wΔzθ_r``, and passes the excess to the layer below within the step.
Water leaving the bottom layer is runoff. If refrozen water would fill a
layer's pores, the water that does not fit passes on; capillary barriers
and ponding are not represented.

**Gravity drainage** (`SnowGravityDrainage`). This is Richards' equation
without the capillary-pressure term. It is the kinematic-wave model of
Colbeck (1972), used in SNTHERM (Jordan 1991) and early Crocus:

```math
q = ρ_w K_s S_e^{3},\qquad S_e = \frac{θ_w-θ_r}{θ_s-θ_r},\qquad θ_s = 1-θ_i,
```

with the saturated hydraulic conductivity from the permeability of
Calonne et al. (2012),
``K_s = 0.75\,r_{es}^2 e^{-0.013ρ_{dry}}\,ρ_w g/μ``. The layer's optical
grain radius (`grain`, which #35 evolves) is ``r_{es}``, and
``μ = 1.792·10^{-3}`` Pa s (Wever et al. 2014). The layer fluxes are
upwinded explicitly. Sub-steps keep the Courant number of the kinematic
wave, ``dq/dθ = 3K_sS_e^2/(θ_s-θ_r)``, at 0.5 or below. No layer drains
below ``θ_r`` or fills beyond ``θ_s``, and water arriving in a cold layer
refreezes. Under a constant input ``q_0`` on dry snow the wetting front
is a shock. Behind it the water content is ``θ_1``, set by
``ρ_wK_sS_e(θ_1)^3 = q_0``, and it advances at
``v = q_0/(ρ_wθ_1)``. Capillary suction and the van Genuchten retention
curve of full Richards models (Wever et al. 2014) are not included, so
neither are capillary barriers.

| check | measured | reference |
|:--- |:--- |:--- |
| Retention laws | to 1e-12 | Wever et al. (2014) Eq. 1; Vionnet et al. (2012) Eq. 19 |
| Refreezing, 1 layer, 3 water amounts | enthalpy to 1e-14, refrozen mass and temperature to 1e-12 | exact cold-content solution |
| Bucket on a cold 5-layer pack, 3 rain amounts | refrozen / retained / runoff to 1e-10 (e.g. 3.159 / 30.251 / 7.000 kg m⁻²) | closed-form partition |
| Kinematic wave, 1.5 m, 18 mm h⁻¹ | arrival 9.52 h vs 9.54 h (−0.22 %); θ behind front 0.1145 (to 1e-3); steady runoff 1.00000 q₀ (tested to 1e-3) | Colbeck (1972) shock |
| Gravity drainage into −8 °C pack | refrozen 15.165 = cold content / L_f (to 1e-9); first runoff after 103 kg m⁻² (> 15.2 + 47.9) | cold content + retention |
| 8 diurnal cycles + 21.6 mm rain, 3 schemes | mass residual ≤ 5e-14 kg m⁻², enthalpy residual ≤ 1e-7 J m⁻² of 6.2e7 | budget closure |

In the diurnal run, the retaining schemes refreeze 24.5 kg m⁻² (bucket)
and 28.8 kg m⁻² (gravity drainage) of meltwater and rain, and let none
run off. Instant runoff loses 29.8 kg m⁻². Each afternoon's meltwater has
refrozen by dawn until the rain day. None of this is compared with
observations: SnowMIP lysimeter runoff is not used here.

## References

- Anderson, E. A. (1976). *A point energy and mass balance model of a snow cover.* NOAA Tech. Rep. NWS 19.
- Calonne, N. et al. (2011). *Geophys. Res. Lett.*, 38, L23501.
- Jordan, R. (1991). *A one-dimensional temperature model for a snow cover.* CRREL Spec. Rep. 91-16.
- Oleson, K. W. et al. (2010). *Technical description of version 4.0 of the Community Land Model.* NCAR/TN-478+STR.
- Sturm, M., Holmgren, J., König, M. & Morris, K. (1997). *J. Glaciol.*, 43, 26-41.
- Sturm, M., Taras, B., Liston, G. E., Derksen, C., Jonas, T. & Lea, J. (2010). *J. Hydrometeorol.*, 11, 1380-1394.
- Vionnet, V. et al. (2012). *Geosci. Model Dev.*, 5, 773-791.
- Yen, Y.-C. (1981). *Review of thermal properties of snow, ice and sea ice.* CRREL Rep. 81-10.
- Calonne, N. et al. (2012). *The Cryosphere*, 6, 939-951.
- Colbeck, S. C. (1972). A theory of water percolation in snow. *J. Glaciol.*, 11, 369-385.
- Coléou, C. & Lesaffre, B. (1998). Irreducible water saturation in snow: experimental results in a cold laboratory. *Ann. Glaciol.*, 26, 64-68.
- Wever, N. et al. (2014). Solving Richards Equation for snow improves snowpack meltwater runoff estimations. *The Cryosphere*, 8, 257-274.
