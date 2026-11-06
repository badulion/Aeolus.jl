# Water surfaces: mixed layer, cool skin and warm layer

A lake or a coastal sea is the one surface whose temperature a
model cannot simply prescribe: the heat it exchanges with the air is
stored in, and redistributed through, metres to tens of metres of water.
`src/OceanMixedLayer.jl` makes that temperature prognostic with three
nested pieces, each at the scale on which it acts:

| scale        | process                          | component                          |
|:------------ |:-------------------------------- |:---------------------------------- |
| 1–100 m      | mixed layer and thermocline      | `WaterColumn`, `KrausTurnerMixing` |
| 0.1–10 m     | diurnal warm layer               | `FairallWarmLayer`, `ZengBeljaarsWarmLayer` |
| ≲ 1 mm       | conductive cool skin             | `cool_skin`                        |

The skin temperature seen by the atmosphere is

```math
T_s = T_1 + \Delta T_{warm}(\Delta z_1/2) - \Delta T_{cool},
```

the top layer of the column, plus the warm layer's warming between the
surface and the top layer's centre, minus the cool skin. Air–sea fluxes
are *not* computed here: they come from the surface shared core (`SurfaceExchange`)
(`SurfaceExchange.turbulent_fluxes`) or, through the `flux` keyword of
`step_surface!`, from a dedicated air–sea algorithm (`AirSeaFlux`, #39).
Whatever the flux function returns is exactly what the column absorbs.

## 1. The column

Layers of thickness ``\Delta z_k`` carry temperatures ``T_k`` at a
constant salinity ``S``. One step:

1. **Heating.** The non-solar flux ``Q_{ns} = G - SW_{net}`` enters the top
   layer; the net shortwave is absorbed layer by layer from a
   sum-of-exponentials profile ``I(z)/I_0 = \sum_i a_i e^{-z/\zeta_i}``;
   what reaches the bottom is absorbed by a lake bed or leaves through
   deep water.
2. **Convective adjustment.** Any group of layers with denser water above
   lighter is mixed. Density comes from the equation of state, so the
   adjustment knows about the 3.98 °C maximum of fresh water.
3. **Entrainment.** A bulk (Kraus–Turner / Niiler–Kraus) energy budget
   deepens the mixed layer from the surface.
4. **Skin corrections** — warm layer and cool skin, which hold no heat of
   their own.

Heat is conserved exactly: over a step the column's heat content changes
by ``(G - SW_{bottom} - F_{bottom})\,\Delta t``, to round-off (measured:
≤ 2.3 × 10⁻¹⁴ of the stored heat over three coupled days).

### Equation of state

The one-atmosphere International Equation of State of Seawater (EOS-80),

```math
\rho(t,S) = \rho_w(t) + A(t)\,S + B(t)\,S^{3/2} + C\,S^2 ,
```

covers fresh and sea water alike; ``\alpha = -\rho^{-1}\partial\rho/\partial T``
and ``\beta = \rho^{-1}\partial\rho/\partial S`` are its analytic
derivatives. It reproduces the published check values to 5 × 10⁻⁶ kg m⁻³
and puts the fresh-water maximum at 3.982 °C. For lakes this is the whole
story of autumn overturn: surface cooling drives convection until the
lake reaches 3.98 °C throughout, after which colder water is *lighter*
and stays at the top — the inverse stratification under which lakes
freeze from the surface.

### Shortwave penetration

Paulson & Simpson (1977) fitted two exponentials to Jerlov's water types,

| type | R    | ζ₁ (m) | ζ₂ (m) |
|:---- |:---- |:------ |:------ |
| I    | 0.58 | 0.35   | 23     |
| IA   | 0.62 | 0.60   | 20     |
| IB   | 0.67 | 1.0    | 17     |
| II   | 0.77 | 1.5    | 14     |
| III  | 0.78 | 1.4    | 7.9    |

The warm-layer schemes use Soloviev's three-band profile with fractions
(0.28, 0.27, 0.45): e-folding depths (0.014, 0.357, 12.82) m in COARE and
(0.014, 0.357, 16.7) m in Zeng & Beljaars (2005). About 60 % of the net
shortwave is absorbed in the top 3 m — which is why a calm sunny day can
warm the surface by a kelvin while the water a few metres down barely
moves.

### Mixed-layer energetics

Mixing a stably stratified column raises its potential energy. The bulk
model supplies, per step,

```math
E = \rho_0\, m\, u_{*w}^3\, \Delta t\; e^{-h/h_d} \;+\; n\,\Delta PE_{conv},
```

the wind work (``u_{*w} = \sqrt{\tau/\rho_0}``) plus a fraction ``n`` of
the potential energy released by convective overturning, and the mixed
layer deepens until the potential-energy cost of mixing equals ``E``.
The base layer is entrained *partially* (a fraction found by bisection),
so the depth is not quantised to the grid.

Two analytic limits check the energetics:

* **Kato–Phillips** (wind into a constant ``N^2``, no heat flux): mixing
  to depth ``h`` costs ``\rho_0 N^2 h^3/12``, so
  ``h = (12\,m\,u_{*w}^3 t / N^2)^{1/3}``. Measured after a day: −0.57 % at
  ``\Delta z`` = 0.2 m and −0.017 % at 0.1 m.
* **Free convection** (surface buoyancy loss ``B_0`` into ``N^2``, no
  wind): the budget gives ``h^2 = 2(1 + 2n)\,B_0 t/N^2`` — encroachment
  for ``n = 0``, and the classical entrainment-flux ratio 0.2 for
  ``n = 0.2``. Measured after two days: −0.04 % and +0.14 %.

The defaults ``m = 0.5``, ``n = 0.2`` lie in the range of the bulk-model
literature (Niiler & Kraus 1977) and are not tuned. Not represented:
shear instability at the mixed-layer base (currents are not carried),
background diffusion below the mixed layer, and ice (below freezing the
column hands over to `IceThermodynamics`, #37).

## 2. The cool skin

The top millimetre of water loses heat by conduction only, so it is
cooler than the water beneath by (Saunders 1967; Fairall et al. 1996a)

```math
\Delta T_{cool} = \frac{\delta}{k_w}\,\big(Q_{out} - f_s(\delta)\,SW_{net}\big),
\qquad
\delta = \frac{\lambda\,\nu_w}{u_{*w}},\quad
\lambda = 6\Big[1 + \Big(\frac{16\,g\,c_w\rho_w\nu_w^3 B}{k_w^2 u_{*w}^4}\Big)^{3/4}\Big]^{-1/3},
```

with ``Q_{out}`` the non-solar loss (net longwave + sensible + latent),
``f_s(\delta) = 0.065 + 11\delta - (6.6\times10^{-5}/\delta)(1 - e^{-\delta/8\times10^{-4}})``
the shortwave absorbed within the skin, and the buoyancy flux
``B = \alpha q_c + \beta S\, c_w LE/L_v`` including the salt left behind by
evaporation. It is typically 0.2–0.4 K. On the COARE 3.6 ship test data
the implementation reproduces the reference code to 9 × 10⁻⁵ K.

## 3. The diurnal warm layer

Under weak wind the absorbed sunshine builds a stable layer a few metres
deep; the skin can be a kelvin or more warmer than the water at 5 m by
early afternoon. Two schemes are provided.

**Fairall et al. (1996b)**, as in COARE. From the first moment in a day
the absorbed heat exceeds 50 W m⁻², the heat ``Q_{ac} = \int q\,dt`` and
momentum ``I_{ac} = \int \tau\,dt`` put into the layer are accumulated. A
layer with linear temperature and current profiles at critical
Richardson number 0.65 then has

```math
D = I_{ac}\sqrt{\frac{2 Ri_c c_w}{\alpha g \rho_w Q_{ac}}},\qquad
\Delta T = \frac{2 Q_{ac}}{\rho_w c_w D}.
```

It needs a daily reset (at local midnight). On the COARE 3.6 test data it
reproduces the reference code to 4 × 10⁻⁷ K (peak warming 0.436 K).

**Zeng & Beljaars (2005)**, as in the IFS: a prognostic equation for
``\Delta T = T_{-\delta} - T_{-d}`` over a fixed ``d = 3`` m with the
profile ``\Delta T[1 - (z/d)^\nu]``, ``\nu = 0.3``,

```math
\frac{\partial \Delta T}{\partial t} =
\frac{Q + R_s - R(-d)}{d\rho_w c_w \nu/(\nu+1)}
- \frac{(\nu+1)\kappa u_{*w}}{d\,\phi_t(d/L)}\,\Delta T ,
```

with the log-linear water-side ``\phi_t``. Its neutral steady state is
``\Delta T = (Q + R_s - R(-d))/(\rho_w c_w \nu\kappa u_{*w})``. Being
prognostic it needs no reset and lets a residual layer survive a calm
night.

**Which to use.** Only when the column's top layer is thicker than the
warm layer (a slab of several metres). A column with top layers of
≲ 0.5 m resolves the warm layer itself — on a clear day it gives a 1.25 K
diurnal skin range at 1.5 m s⁻¹ wind and 0.32 K at 10 m s⁻¹ — and adding a
sub-grid warm layer would count it twice.

**Against observations.** On the COARE ship data (days 19–25, trade winds
4–11 m s⁻¹) the observed difference between a sea snake at 5 cm and the
thermosalinograph at 5.3 m never exceeds 0.14 K. Both schemes put the
largest warming on the calmest day, as observed, but both overestimate
it (Fairall 0.33 K, Zeng–Beljaars 0.85 K on that day; day-time RMS
differences 0.09 K and 0.25 K). Takaya et al. (2010) refined the
Zeng–Beljaars scheme for this regime (Langmuir mixing, a modified stable
``\phi_t``); those refinements are not implemented.

## References

* Fairall, C. W. et al. (1996a). Cool-skin and warm-layer effects on sea
  surface temperature. *J. Geophys. Res.*, 101, 1295–1308.
* Fairall, C. W. et al. (1996b). Bulk parameterization of air-sea fluxes
  for TOGA COARE. *J. Geophys. Res.*, 101, 3747–3764.
* Fairall, C. W. et al. (2003). Bulk parameterization of air–sea fluxes:
  updates and verification for the COARE algorithm. *J. Climate*, 16,
  571–591.
* Zeng, X. & Beljaars, A. (2005). A prognostic scheme of sea surface skin
  temperature for modeling and data assimilation. *Geophys. Res. Lett.*,
  32, L14605.
* Takaya, Y., Bidlot, J.-R., Beljaars, A. C. M. & Janssen, P. A. E. M.
  (2010). Refinements to a prognostic scheme of skin sea surface
  temperature. *J. Geophys. Res.*, 115, C06009.
* Paulson, C. A. & Simpson, J. J. (1977). Irradiance measurements in the
  upper ocean. *J. Phys. Oceanogr.*, 7, 952–956.
* Kraus, E. B. & Turner, J. S. (1967). A one-dimensional model of the
  seasonal thermocline II. *Tellus*, 19, 98–106.
* Niiler, P. P. & Kraus, E. B. (1977). One-dimensional models of the
  upper ocean. In *Modelling and Prediction of the Upper Layers of the
  Ocean*, Pergamon, 143–172.
* Kato, H. & Phillips, O. M. (1969). On the penetration of a turbulent
  layer into stratified fluid. *J. Fluid Mech.*, 37, 643–655.
* Saunders, P. M. (1967). The temperature at the ocean-air interface.
  *J. Atmos. Sci.*, 24, 269–273.
* UNESCO (1981). *Tech. Pap. Mar. Sci.* 36/38; Fofonoff, N. P. & Millard,
  R. C. (1983). *Tech. Pap. Mar. Sci.* 44.
* ECMWF (2017). IFS Documentation Cy43r1, Part IV, §8.10.
