# Soil heat and freeze–thaw

Phase 46 (#16) gives the soil column of [soil water](soil_water.md) a
temperature. Heat is conducted through layers whose conductivity and heat
capacity depend on how much liquid water and ice they hold; water freezes
and thaws with its latent heat; and a massless skin on top closes the
surface energy balance with the atmosphere. The result, `SoilThermalColumn`
in `src/SoilColumn.jl`, is an `AbstractSurface` of the
[Era V contract](surface_exchange.md).

Why it matters: without a ground heat flux the skin is decoupled from
the heat stored below it. The single-slab `step_soil!` of the urban
example once computed the conduction flux into the soil and never took
it out of the skin; for the hot urban surfaces of that run the omission
removed roughly half of the evening cooling (nearly all of it in deep
canyons) — commit 2b31389. The flux works both ways: it drains a skin
warmer than the soil beneath and, on a clear night over cooler air,
feeds heat back to a radiatively cooling skin (section 5).

Since #18 the urban example has no soil model of its own. Since #129 the
`SoilThermalColumn` *is* the surface of every street, building-adjacent
and park column of the Alexanderplatz run: its massless skin exchanges
Monin–Obukhov fluxes with the first air cell through `step_surface!`,
with the soil water stepped (`hydrology = true`) so the moisture
availability follows the top layer (see
[the Alexanderplatz run](../examples/battery.md#Ground:-the-production-soil-column)
and `docs/src/design/berlin_surface_radiation.md`). Before that (#18) it sat under a
surface slab, coupled through a contact conductance.

## 1. Thermal properties

**Heat capacity** is the sum over constituents (de Vries 1963),

```math
C = c_s(1-\theta_s) + \rho_w c_w\,\theta_{liq} + \rho_{ice} c_{ice}\,\theta_{ice},
```

air neglected, with the solids' ``c_s`` either from sand and clay
(``c_s = (2.128\,\%sand + 2.385\,\%clay)/(\%sand+\%clay)\times10^6``
J m⁻³ K⁻¹; Farouki 1981, as in CLM) or the mineral value
``2.0\times10^6`` J m⁻³ K⁻¹ of de Vries. ``\rho_w``, ``\rho_{ice}`` are the
contract's `SurfaceConstants`; ``c_w``, ``c_{ice}`` are delegated through
it to `MoistThermodynamics`.

**Conductivity** follows Johansen (1975) as used by CLM:

```math
k = K_e\,k_{sat} + (1-K_e)\,k_{dry},\qquad
k_{sat} = k_s^{1-\theta_s}\,k_w^{\theta_s f_{liq}}\,k_{ice}^{\theta_s(1-f_{liq})},\qquad
k_{dry} = \frac{0.135\,\rho_d + 64.7}{\rho_p - 0.947\,\rho_d},
```

with ``\rho_d = \rho_p(1-\theta_s)``, ``f_{liq}`` the liquid fraction of
the pore water, and the Kersten number ``K_e`` a function of the degree
of saturation ``S_r = (\theta_{liq}+\theta_{ice})/\theta_s``:
``\log_{10}S_r + 1`` (fine soils) or ``0.7\log_{10}S_r + 1`` (coarse),
floored at zero, when unfrozen, and ``S_r`` when frozen. A partly frozen
layer takes the liquid-weighted mean of the two — continuous, where CLM
switches at 0 °C. Solids: ``k_s = (8.80\,\%sand + 2.92\,\%clay)/(\%sand+\%clay)``
(Farouki; `clm_soil_thermal_properties`) or ``7.7^q\,k_o^{1-q}`` from the
quartz fraction (Johansen; Peters-Lidard et al. 1998;
`johansen_soil_thermal_properties`).

For a 40 % sand / 20 % clay soil of porosity 0.45: ``k_{dry}`` = 0.205,
``k_{sat}`` = 2.24 unfrozen and 4.12 W m⁻¹ K⁻¹ frozen (ice conducts
3.9 times better than water).

## 2. Freezing

**Clapeyron curve.** Liquid water held in pores next to ice has the
matric potential (Koopmans & Miller 1966)

```math
\psi_{fr}(T) = \frac{L_f\,(T - T_f)}{g\,T},
```

about −125 m (−1.22 MPa) per kelvin of undercooling. The largest liquid
content that can coexist with ice is the retention curve evaluated there,
``\theta_{liq,max}(T) = \theta(\psi_{fr}(T))`` — the "freezing = drying"
analogy used by CLM5 (Niu & Yang 2006) and Dall'Amico et al. (2011). A
layer at potential ``\psi`` starts to freeze at
``T^* = L_f T_f/(L_f - g\psi)``, below ``T_f``; drier soil starts
later. For the Carsel–Parrish loam the unfrozen water is 0.226, 0.120,
0.090 and 0.083 m³ m⁻³ at −0.01, −0.1, −1 and −5 °C.

`freezing_point_depression = false` gives instead a **sharp** phase
change at ``T_f`` — the classical Stefan problem. In both modes a small
fraction (`unfrozen_fraction`, default 10⁻³) of the pore space above the
residual content stays liquid, so the water solver always has a liquid
phase to move.

**Ice in the water solver.** The retention curve of the water column is
applied to the *total* water ``W = \theta_{liq} + (\rho_{ice}/\rho_w)\theta_{ice}``
(after CLM4.5 onward, Swenson et al. 2012, which use the liquid-plus-ice
volume; see the end of this paragraph): freezing moves water between
phases at constant ``\psi``, and ice acts through the impedance factor
``10^{-\Omega\,\theta_{ice}/\theta_s}`` with ``\Omega = 6``
(`PowerLawIceImpedance`) and the viscosity of cold water, rather than
through a suction. The alternative — ψ from the liquid alone — makes a
frozen layer look like very dry soil (−1250 m at −10 °C) next to wet
unfrozen soil, and with the column's logarithmic interface conductivity
that drives an unbounded "cryosuction" flux on coarse layers. Ice lensing
and frost heave are not modelled; the liquid-equivalent ``W`` is used
rather than ``\theta_{liq}+\theta_{ice}`` so that the 9 % expansion of
freezing water does not register as over-saturation.

## 3. Enthalpy formulation

Each layer carries its volumetric enthalpy relative to liquid water at
``T_f``,

```math
H = C(\theta_{liq},\theta_{ice})\,(T - T_f) - \rho_{ice} L_f\,\theta_{ice},
```

which — with ``L_f`` from `MoistThermodynamics` and constant specific
heats — is Kirchhoff-consistent: the latent heat at ``T`` is
``L_f + (c_w - c_{ice})(T - T_f)``. Given ``H`` and ``W``, the temperature
and the phase partition are the unique equilibrium: all liquid if
``H \ge C_{liq}(T^*-T_f)``; otherwise the monotone equation
``H_{eq}(T) = H`` is solved below ``T^*`` by a safeguarded Newton
iteration converged in enthalpy (to 10⁻¹³ of the latent heat content); in
sharp mode the layer sits at ``T_f`` while ``H`` crosses the latent
interval.

**Discretisation.** Finite volumes on the soil grid, with the interface
conductance of the two half layers in series, and backward Euler:

```math
\Delta z_i \frac{H_i^{n+1} - H_i^n}{\Delta t} = F_{i-1/2} - F_{i+1/2},\qquad
F_{i+1/2} = \frac{T_i - T_{i+1}}{(z_{h,i+1}-z_i)/k_i + (z_{i+1}-z_{h,i+1})/k_{i+1}} .
```

The nonlinear system is solved by Newton's method *in H* with the
conductivity lagged: ``T \approx T^m + s^m(H - H^m)``, ``s = dT/dH``. In a
sharp phase change ``s = 0``, so no apparent-heat-capacity smoothing is
needed and the iteration cannot overshoot the latent plateau (the
enthalpy method; Voller et al. 1990, Nedjar 2002). The matrix
``D + AS`` (``A`` the conduction operator, ``S = \mathrm{diag}(s)``) is
column diagonally dominant, so the tridiagonal solve is stable. After
convergence the new enthalpy is formed from the divergence of the
converged fluxes, so ``\Delta\sum\Delta z H = (G - F_{bottom})\Delta t``
holds to round-off. Sub-steps halve on non-convergence and grow after
fast convergence.

**Boundaries.** Every upper condition is reduced to a ground flux linear
in the top-node temperature, ``G = \alpha + \beta T_1``: a prescribed flux
(`GroundHeatFluxTop`), a prescribed skin temperature (`SkinTemperatureTop`,
through the half layer ``z_1``), or a linearised skin energy balance
(`LinearizedSkinBalance`, below). The base is insulated (CLM's default),
has a geothermal flux, or a fixed temperature.

**Water.** A Richards step keeps each layer's temperature while its
liquid changes, so water carries the enthalpy of the layer it enters or
leaves (booked as `budget.water`); each layer then re-equilibrates at
constant enthalpy — water that infiltrates frozen soil refreezes and
warms it. Rain is assumed to arrive at the temperature of the top layer.

## 4. The skin energy balance

`step_surface!(col, forcing, dt)`:

1. evaluates `SurfaceExchange.turbulent_fluxes` (Monin–Obukhov) at the
   current skin temperature ``T_0`` and at ``T_0 + 0.01`` K, giving
   ``R_n``, ``H``, ``LE``, ``E`` and their derivatives; evaporation is
   capped by the liquid in the top layer;
2. solves the soil together with the skin, implicitly:
   ``G_{atm}(T_s) = G_0 + (dG/dT_s)(T_s - T_0) = (k_1/z_1)(T_s - T_1)``
   eliminates ``T_s`` and becomes the linear upper condition of the heat
   solve;
3. returns the fluxes at the step-mean skin temperature, so
   ``R_n = H + LE + G`` exactly and ``G\,\Delta t`` is exactly what the
   soil received;
4. runs the Richards step with the rain of the forcing and the ``E`` just
   returned.

The bare-soil evaporation efficiency is the Lee & Pielke (1992) β of the
top layer's liquid water; roughness defaults to CLM4.5's bare-soil
0.01 m, emissivity to the dry-soil value of `SurfaceOptics`. The
linearisation is done once per step, as in CLM.

## 5. Validation

All numbers are from `test/era5_soil_thermal.jl`.

**Neumann's freezing front** (Carslaw & Jaeger 1959, §11.2). Soil with
``W = 0.3`` at +2 °C, surface held at −10 °C, sharp freezing; the phase
properties are the model's own for the fully frozen and thawed soil
(``\lambda = 0.2897``). After 30 days the front, measured as the
frozen-depth equivalent of the column's ice, is at

| Δz, Δt | numerical X | Neumann X | error | max ``|ΔT|`` off the front cell |
| :-- | :-- | :-- | :-- | :-- |
| 40 mm, 3600 s | 1.2595 m | 1.2653 m | −4.6e-3 | 0.027 K |
| 20 mm, 900 s | 1.2625 m | 1.2653 m | −2.3e-3 | 0.058 K |
| 10 mm, 225 s | 1.2639 m | 1.2653 m | −1.2e-3 | 0.012 K |

— first-order convergence of the front, as expected for a fixed-grid
enthalpy method, and at most one partly frozen cell. The node of that
cell is pinned at ``T_f`` (the method's well-known staircase), which is
why the temperature comparison leaves it out.

**Periodic surface temperature** (Carslaw & Jaeger §2.6). A sinusoidal
skin temperature penetrates as ``A e^{-z/d}\sin(\omega t - z/d)`` with
damping depth ``d = \sqrt{2k/(C\omega)}``, and the ground heat flux leads
the surface temperature by π/4 with amplitude ``A\sqrt{kC\omega}``. For
the loam at ``\theta = 0.3`` (``k`` = 2.02 W m⁻¹ K⁻¹, ``C`` = 2.52 MJ m⁻³ K⁻¹):

| wave | d | amplitude at d, 2d | exact | lag at d, 2d (rad) | G lead |
| :-- | :-- | :-- | :-- | :-- | :-- |
| diurnal (Δz 5 mm, Δt 150 s) | 0.149 m | 0.3693, 0.1339 | 0.3703, 0.1348 | 0.996, 2.002 | 0.7825 |
| annual (Δz 0.1 m, Δt 12 h) | 2.84 m | 0.3654, 0.1356 | 0.3661, 0.1364 | 1.007, 1.989 | 0.7830 |

(π/4 = 0.7854; the ground-flux amplitude matches ``A\sqrt{kC\omega}`` to
1e-4). The residual amplitude error has two parts: the numerical damping
of backward Euler, and the start-up transient left after the six cycles
the test runs. At 3d it is −4.2 %, −1.8 % and −1.2 % after 6 cycles, and
−3.4 %, −1.0 % and −0.36 % after 16, for (Δz, Δt) = (10 mm, 600 s),
(5 mm, 150 s), (2.5 mm, 37.5 s).

**Energy conservation.** Forty days of freezing and thawing a layered
CLM5 column under ±80 W m⁻² with a geothermal base, both freezing modes:
worst per-step residual 6e-8 J m⁻², cumulative 5e-8 J m⁻², water
conserved to 1e-14. A single sharp-freezing layer holds exactly at
``T_f`` (a zero curtain) for ``\rho_{ice}L_f\theta_{ice,max}\Delta z/|G|``
and then cools. The full coupled surface–heat–water runs close
``\Delta H = \int G - \int F_{bottom} + \text{water}`` to round-off.

**Skin balance.** ``R_n - H - LE - G`` = 0 to 1e-10 W m⁻² every step.
The linearisation error — the full Monin–Obukhov balance at the new skin
temperature minus ``G`` — is 13.8 W m⁻² on the first step of a night
(the skin drops 3 K from an unbalanced start) and at most 0.21 W m⁻²
from the fourth 10-minute step on. After 12 h of a clear night the
conducting soil keeps the skin at 280.2 K, where a surface with ``G = 0``
would sit at 272.5 K. Under constant forcing over a fixed-temperature
base the column reaches the steady state ``G = k\,(T_s - T_b)/D`` with
the full nonlinear balance closed to 0.01 W m⁻².

**Frozen ground.** 30 mm of rain in 6 h: 0 mm runoff on thawed loam,
30 mm on frozen loam (ice impedance). Water infiltrating a frozen layer
refreezes and warms it from −2.00 to −0.004 °C. A 60-day winter–spring
cycle for every Carsel–Parrish texture builds and thaws frost with
energy and water closed.

**Not validated against observations.** The phase target — soil
temperature and moisture profiles at FLUXNET sites — needs the vegetated
land column (#21); this issue is validated against the analytic
solutions and conservation laws above only.

## References

- Carslaw, H. S. & Jaeger, J. C. (1959). *Conduction of Heat in Solids*, 2nd ed. Oxford.
- Dall'Amico, M. et al. (2011). *The Cryosphere*, 5, 469–484.
- de Vries, D. A. (1963). In *Physics of Plant Environment*, 210–235. North-Holland.
- Farouki, O. T. (1981). *Thermal properties of soils*. CRREL Monograph 81-1.
- Johansen, O. (1975). *Thermal conductivity of soils*. PhD thesis, Trondheim.
- Koopmans, R. W. R. & Miller, R. D. (1966). *Soil Sci. Soc. Am. Proc.*, 30, 680–685.
- Lee, T. J. & Pielke, R. A. (1992). *J. Appl. Meteorol.*, 31, 480–484.
- Nedjar, B. (2002). *Comput. Struct.*, 80, 9–21.
- Niu, G.-Y. & Yang, Z.-L. (2006). *J. Hydrometeorol.*, 7, 937–952.
- Peters-Lidard, C. D. et al. (1998). *J. Atmos. Sci.*, 55, 1209–1224.
- Pollack, H. N., Hurter, S. J. & Johnson, J. R. (1993). *Rev. Geophys.*, 31, 267–280.
- Swenson, S. C., Lawrence, D. M. & Lee, H. (2012). *J. Adv. Model. Earth Syst.*, 4, M08002.
- Voller, V. R., Swaminathan, C. R. & Thomas, B. G. (1990). *Int. J. Numer. Methods Eng.*, 30, 875–898.
