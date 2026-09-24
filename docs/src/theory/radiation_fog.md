# Radiation fog

This page covers the radiative half of radiation fog in Aeolus (Era IV,
Phase 44, issue #9): how fog droplets enter the Era III radiation, why
that coupling is what thickens a fog, and the 1-D column used to test
it against the ParisFog case of 14–15 November 2011. The thermodynamic
half (condensation, dew, energy conservation) is in
[Fog and dew](fog.md). The code is in `src/radiation_fog/` inside the
`RadiationFog` module.

## 1. The feedback

A clear night cools the ground by about 60 W m⁻² of net longwave
(Wærsted et al. 2017 measured this at SIRTA before fog formed). Once
fog forms and its liquid water path passes about 30 g m⁻², it becomes
opaque in the thermal infrared. The ground then sees the fog's own
emission instead of the cold sky, and its net longwave drops toward
zero. The cooling moves to the fog top, which now radiates to the sky
with a net loss of 70–100 W m⁻². That cooling is concentrated in the
uppermost 50–100 m of the fog. Saturated air cooled there condenses
more water, and cooling from the top destabilises the fog layer. Mixing
then carries the cold, wet air down and the fog deepens. Without the
droplets in the radiation none of this happens: the fog stays a thin
layer that cools only from the ground.

## 2. Coupling to the Era III radiation

Nothing in `RadiationFog` computes an optical property. For each layer
the column writes

* the fog water `q_l` into `AtmosphericColumn.ql`,
* the effective radius
  ``r_\mathrm{eff} = \bigl(3\,\mathrm{LWC}/(4\pi\rho_w N_d k)\bigr)^{1/3}``
  (from `AerosolCloud.droplet_effective_radius`, Martin et al. 1994,
  ``k = 0.8``) into `r_eff_liq`,
* the vapour into `vmr[:h2o]`.

It then calls `GasOptics` correlated-k, `TwoStream.MieCloudOptics`,
`RadiationCore.combine_optics!` and `solve_lw!` / `solve_sw!`. Above
the 1 km model top sits a fixed `us_standard_column` upper atmosphere,
made continuous in temperature, pressure and humidity with the model
top, so that the sky seen from the fog top has realistic emission.

`droplet_feedback = false` passes `ql = 0` to the radiation. The fog
still exists thermodynamically (its water, its latent heat, its
stability) but is transparent. Switching it off is the controlled
experiment behind the feedback tests.

### Radiation-driven condensation

Wærsted et al. (2017, Eqs. 1–5) turn a radiative heating rate into the
condensation it forces in saturated air:

```math
C_\mathrm{rad} = -\frac{\frac{d\rho_s}{dT}\left(\frac{dT}{dt}\right)_\mathrm{rad}}
                       {1 + \frac{L_v}{\rho c_p}\frac{d\rho_s}{dT}},\qquad
\frac{d\rho_s}{dT} = \frac{e_s}{R_v T^2}\left(\frac{L_v}{R_v T} - 1\right).
```

`radiative_condensation_rate` evaluates it with the contract's
``e_s`` and ``L_v``. It gives 0.525 g m⁻² h⁻¹ per W m⁻² at −2 °C and
0.897 at 15 °C. Wærsted et al. quote 0.55 and 0.90. It also agrees
within 2.6 % with the exact outcome of `fog_condensation` applied to a
parcel cooled by the same energy.

## 3. The column

`FogColumn` is a single-column model with 1 m layers at the ground,
stretched by 8 % per layer to at most 25 m, up to 1000 m. Each step of
10 s does the following:

1. Radiation, every 60 s, with gas optics refreshed every 15 min. The
   heating is applied as ``\Delta h = Q\,\Delta t/\rho`` at fixed water
   partition.
2. Surface. A force-restore ground temperature (Bhumralkar 1975;
   Deardorff 1978) with admittance 800 J m⁻² K⁻¹ s⁻½, receiving the net
   longwave (upwelling re-evaluated from the current ``T_s``), the
   absorbed shortwave, a bulk sensible heat flux and the latent heat of
   `dew_deposition` or of β-limited soil evaporation.
3. Turbulence. A first-order K closure,
   ``K = l^2 |\partial U/\partial z| f(Ri) + K_\min``, with the
   Blackadar length (``\lambda`` = 30 m) and
   ``f = (1+5Ri)^{-2}`` for ``Ri > 0``, ``\sqrt{1-16Ri}`` for ``Ri < 0``.
   ``Ri`` uses a *moist* ``N^2``, measured by displacing a parcel
   across each interface with conserved ``s = h + gz`` and total water,
   saturation-adjusting it with `fog_equilibrium`, and comparing
   ``\theta_\rho``. In an 8 K km⁻¹ layer this gives
   ``N^2 = +5.4\times10^{-5}`` s⁻² dry and ``-6.4\times10^{-6}`` s⁻²
   saturated. That sign change is what lets fog-top cooling overturn
   the fog. Mixing is implicit and in flux form for ``s``, ``q_v``,
   ``q_l`` and momentum, followed by Coriolis turning toward an
   8 m s⁻¹ geostrophic wind.
4. Settling at one bulk speed, `Superdroplet.terminal_velocity`
   evaluated at ``r_\mathrm{eff}``: 1.35 cm s⁻¹ at 0.2 g m⁻³ and
   50 cm⁻³. A fall-speed *spectrum* belongs to issue #8.
5. Saturation adjustment with `fog_condensation`.

**Conservation.** Mixing and settling move static energy in flux form,
and the phase change conserves moist enthalpy exactly. The column
integral ``\sum \rho\Delta z\,(h + gz)`` therefore changes only through
the radiative net flux at the model top and surface, the sensible heat
flux, and the enthalpy of the vapour exchanged with the ground and of
the droplets that settle out. `FogBudget` accumulates all four. Over
6 h of the case the residual is 1.8×10⁻⁶ J m⁻² on a 1.14×10⁶ J m⁻²
change, and over the full 14 h it stays below 10⁻⁹ of the fluxes. Water
closes to 10⁻¹⁵.

## 4. Validation: ParisFog, 14–15 November 2011

Mazoyer et al. (2017) document a radiation fog at SIRTA (48.71 °N,
2.21 °E). Liquid water appeared at 150 m at 02:00 UTC, and fog reached
the ground at 02:30. The fog was optically thick by ≈ 03:00 at ≈ 80 m
depth, its LWP peaked at ≈ 70 g m⁻² at 07:30 and its top at ≤ 300 m
(sodar). It cleared at the ground at ≈ 10:00, three hours after the
06:59 sunrise. Their LES starts at 23:20 UTC with a 276 K surface and an
8 m s⁻¹ geostrophic wind, and so does `parisfog_2011_column`.

**What is not reproduced.** The initial sounding is only shown as a
figure, so the column's profile is *constructed*, not digitised. It has
a 2 K inversion over 60 m, a weakly stable layer to 400 m, relative
humidity 97 % at the ground falling to 88 % at 400 m and 50 % above,
and 4 K km⁻¹ above. The profile was set before the first run and not
tuned afterwards. Tree drag and droplet deposition on vegetation, which
Mazoyer et al. found to matter most for their LES, are absent. Onset and
depth are therefore compared to the observations as a life cycle, not
fitted to them.

Measured (`examples/validation/parisfog_2011_radiation_fog.jl`):

| | model | observed |
| :-- | :-- | :-- |
| fog first appears | aloft, top 120 m, base > 10 m | aloft at 150 m |
| fog at the ground (3 m) | 04:35 UTC | 02:30 UTC |
| max LWP | 101 g m⁻² at 08:30 | ≈ 70 g m⁻² at 07:30 |
| max fog top before sunrise | ≈ 330 m | ≤ 300 m |
| ground clear | 08:50 UTC (sunrise + 1.85 h) | ≈ 10:00 (+3.0 h) |
| fog layer gone | 12:50 UTC | — |

The fog forms two hours late and clears at the ground 1.2 h early, 0.5 h
before the lower end of the +2.3 to +7.6 h range that Wærsted et al.
(2017, Table 2) list for SIRTA radiation fogs that cleared after sunrise.
After sunrise the fog lifts into a stratus whose top keeps rising, to
600 m by midday, so the column fog is too deep by then. Mazoyer et al.'s
own LES also cleared an hour early and also lifted into stratus.

**Opaque fog at night (LWP > 30 g m⁻², 25 samples)** against the SIRTA
radiation climatology of Wærsted et al. (2017):

| | model | Wærsted et al. |
| :-- | :-- | :-- |
| net LW at the ground | −13.4 … −10.8 W m⁻² | → 0, "frequently −10" |
| net LW at the fog top | −67.8 … −63.6 W m⁻² | −73 … −100 W m⁻² |
| LW loss in the top 100 m (static fog, 53 g m⁻²) | 99 % | "uppermost 50–100 m" |
| fog-top LW cooling rate | 65–98 K day⁻¹ | — (LES: > 5.5 K h⁻¹ = 132 K day⁻¹ at 1 m) |
| ``C_\mathrm{LW}`` | 31–35 g m⁻² h⁻¹ | 40–70 g m⁻² h⁻¹ |
| max ``E_\mathrm{SW}`` in daytime fog | 5.8 g m⁻² h⁻¹ | 0–15 g m⁻² h⁻¹ |

``C_\mathrm{LW}`` falls below the observed range. The conversion factor
is right (0.59–0.61 g m⁻² h⁻¹ per W m⁻² at 275 K, where Wærsted et al.
give 0.55 to 0.90 between −2 °C and 15 °C), so the shortfall is in the
radiative loss itself. The fog top loses 64–68 W m⁻² rather than 73–100.
Two causes are identifiable. First, the constructed air above the fog is
humid (88 % to 400 m), which strengthens the downwelling longwave.
Second, `MieCloudOptics` gives fog a longwave single-scattering albedo of
0.45, which makes its emissivity ≈ 0.95 rather than ≈ 1: the downwelling
at the ground under the fog is 313 W m⁻² against 326 W m⁻² for a black
body at the fog temperature, and Mazoyer et al. observed 325 W m⁻². The
result sits inside Nakanishi's (2000) LES estimate of ≈ 30–40
g m⁻² h⁻¹, as quoted by Wærsted et al. The peak cooling rate depends on
resolution, because the fog top here sits in 10–25 m layers. The
resolution-independent quantity is the flux jump.

### The feedback, isolated

The same night with the droplets transparent to radiation:

| | feedback on | feedback off |
| :-- | :-- | :-- |
| fog at the ground | 04:35 | 05:40 |
| LWP at 06:00 | 61.2 g m⁻² | 7.0 g m⁻² |
| fog top at 06:00 | 239 m | 203 m |
| max LWP | 101 g m⁻² | 20.6 g m⁻² |
| net LW at the ground in fog | ≈ −12 W m⁻² | ≈ −59 W m⁻² |

With the feedback off the ground keeps cooling at the clear-sky rate
and the fog remains a weak, radiatively passive layer.

### Dissipation and the sun

With the sun switched off, the fog still leaves the ground at 09:10, 20
minutes later than with the sun. The ground heat flux, restoring the
surface toward the deeper soil, and top-driven mixing lift its base.
The layer then persists, with 68 g m⁻² at 13:20. With the sun on, the
surface absorbs up to ≈ 135 W m⁻² under the thinning fog (Wærsted et
al.: 40–120 W m⁻²), and the layer is gone by 12:50. In this setup, then,
the sun removes the fog *layer*. Clearing at the ground is set mostly by
the surface heat balance, and the tests assert only what the model
shows.

## References

* Bhumralkar, C. M. (1975). Numerical experiments on the computation of
  ground surface temperature. *J. Appl. Meteorol.*, 14, 1246–1258.
* Deardorff, J. W. (1978). Efficient prediction of ground surface
  temperature and moisture. *J. Geophys. Res.*, 83, 1889–1903.
* Martin, G. M., Johnson, D. W. & Spice, A. (1994). The measurement and
  parameterization of effective radius of droplets in warm stratocumulus
  clouds. *J. Atmos. Sci.*, 51, 1823–1842.
* Mazoyer, M., Lac, C., Thouron, O., Bergot, T., Masson, V. &
  Musson-Genon, L. (2017). Large eddy simulation of radiation fog:
  impact of dynamics on the fog life cycle. *Atmos. Chem. Phys.*, 17,
  13017–13035. (Numbers quoted from the discussion version,
  acp-2016-900.)
* Nakanishi, M. (2000). Large-eddy simulation of radiation fog.
  *Bound.-Layer Meteorol.*, 94, 461–493.
* Wærsted, E. G., Haeffelin, M., Dupont, J.-C., Delanoë, J. & Dubuisson,
  P. (2017). Radiation in fog: quantification of the impact on fog
  liquid water based on ground-based remote sensing. *Atmos. Chem.
  Phys.*, 17, 10811–10835.
