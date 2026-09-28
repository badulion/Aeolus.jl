# Sea spray and the full COARE algorithm

Phase 52, issue #40. This page covers three additions to `AirSeaFlux`:
the spray-mediated heat fluxes of Andreas et al. (2015), the complete
COARE algorithm (bulk fluxes with cool skin, warm layer and rain), and
the coupling of COARE fluxes to the prognostic `OceanMixedLayer` water
column. The page ends with what these were validated against. The bulk
algorithm itself (roughness, stability, gustiness) is described in
[Air–sea fluxes and wave-dependent roughness](air_sea_flux.md). The cool
skin and warm layer are described in
[Water surfaces (mixed layer)](ocean_mixed_layer.md).

## 1. Spray-mediated heat transfer

Heat and moisture leave the sea by two routes (Andreas et al. 2008,
2015):

- The *interfacial* route is controlled by molecular diffusion at the
  interface. It is the route every bulk algorithm computes.
- The *spray* route runs through droplets torn from wave crests. It
  becomes significant, meaning at least 10 % of the interfacial flux,
  above about 10–12 m s⁻¹.

A spume droplet cools to its equilibrium temperature ``T_{eq}`` within
seconds, and the heat it gives up comes from the ocean. It evaporates
more slowly, on the time scale ``τ_r``, and the latent heat for that
evaporation comes from the air. The droplet falls back into the sea
after a residence time

```math
τ_f = \frac{H_{1/3}}{2\,u_f(r_0)},
```

where ``u_f`` is its terminal speed. During that flight its radius
relaxes towards the equilibrium radius:

```math
r(t) = r_{eq} + (r_0 - r_{eq})\, e^{-t/τ_r}.
```

The fast algorithm represents all droplet sizes by two *bellwether*
droplets: 50 µm for the latent flux and 100 µm for the sensible flux.
Their wind functions ``V_L`` and ``V_S`` were fitted to about 4000
eddy-covariance flux measurements:

```math
H_{L,sp} = ρ_w L_v \left[1 - \left(\frac{r(τ_{f,50})}{50\,\mu m}\right)^3\right] V_L(u_*),
\qquad
H_{s,sp} = ρ_w c_w (Θ_s - T_{eq,100})\, V_S(u_*),
```

```math
V_L = 2.08\times10^{-7} u_*^{2.39},\qquad V_S = 5.02\times10^{-6} u_*^{2.54}
\quad (\text{constant below } u_* ≈ 0.14\ \text{m s}^{-1}).
```

``H_{s,sp}`` already includes the cooling of the air by droplet
evaporation, the ``-(α-γ)Q_L`` term of the full model. The total fluxes
that leave the droplet evaporation layer are the interfacial fluxes plus
the spray fluxes. As in Andreas et al., the spray does not feed back on
the stability.

**Microphysics comes from what the package already has.** Andreas used
his own approximation formulas (Andreas 2005); those are not
reproduced here. Instead:

- **``T_{eq}``** is the saline wet-bulb temperature:
  ``k_a(T_a - T) = L_v D_v [a_w e_s(T)/(R_vT) - RH\, e_s(T_a)/(R_vT_a)]``.
  - The water activity ``a_w`` includes solute and curvature and comes
    from the κ-Köhler curve in `MoistThermodynamics`, with the sea-salt
    κ of `Superdroplet`.
  - ``k_a`` and ``D_v`` come from `Superdroplet`.
- **``r_{eq}``** is the stable κ-Köhler equilibrium radius from
  `MoistThermodynamics.kohler_equilibrium_radius`.
- **``τ_r``** follows Andreas's (1990) definition: the time at which
  ``(r - r_{eq})/(r_0 - r_{eq}) = 1/e``. It is found by integrating
  ``r\,dr/dt = (RH - 1 - s_{eq}(r))/(F_k + F_d)`` with RK4, using
  `Superdroplet`'s ``F_k`` and ``F_d``.
- **``u_f``** is the Beard terminal velocity from `Superdroplet`.
- **``ρ_w``** is the EOS-80 seawater density from `OceanMixedLayer`.

**Wave height.** ``H_{1/3}`` comes from a `SeaState` when one is given.
Without one, a fully developed sea is assumed (Taylor & Yelland 2001:
``H_s = 0.0248\,U_{10N}^2``). Andreas et al. did not use that relation:
they used measured wave heights where available, and otherwise the
depth-dependent coastal fit of Andreas & Wang (2007). That fit is not
implemented, because its coefficients could not be obtained. The latent
spray flux is sensitive to this choice (see §4).

**Drag.** Andreas et al. tuned ``V_L`` and ``V_S`` against the ``u_*``
of their own drag relation (Andreas et al. 2012), which
[`andreas_friction_velocity`](@ref) provides. [`coare_fluxes`](@ref)
drives the spray with the COARE ``u_*`` instead. That ``u_*`` is 1–8 %
larger between 5 and 25 m s⁻¹ (11 % at 30 m s⁻¹), which raises the spray
fluxes by roughly 2.5 times as much, since ``V_L, V_S ∝ u_*^{2.4–2.5}``.

## 2. Rain heat flux

Rain falls at roughly the wet-bulb temperature (Gosnell et al. 1995;
Fairall et al. 1996b, eq. 23):

```math
H_{rain} = R\,c_w\,α\left[(T_s - T_a) + (q_s - q_a)\frac{L_v}{c_p}\right],\qquad
α = \left[1 + \frac{L_v}{c_p}\frac{dq_{sat}}{dT}\frac{D_v}{κ_T}\right]^{-1}.
```

Here ``R`` is the rain mass flux and ``κ_T`` the thermal diffusivity of
air. The COARE reference codes evaluate ``dq/dT`` with the air's actual
humidity. The wet-bulb derivation calls for ``q_{sat}``, which is what
is used here. On the COARE 3.6 test data (same c_w) this makes the flux 11–22 %
smaller than the reference.

## 3. The full COARE algorithm

[`coare_fluxes`](@ref) is COARE as the reference codes run it. Every
piece is delegated to its owner:

- **Interface temperature:**
  ``T_{skin} = T_{sea} + ΔT_{warm} - ΔT_{cool}``. ``T_{sea}`` is the
  bulk water temperature at the sensor depth.
- **Seawater humidity:** ``q_s = (1 - 0.02S/34)\,q_{sat}(T_{skin})``,
  from `OceanMixedLayer.salinity_humidity_factor`.
- **Interfacial fluxes:** [`air_sea_fluxes`](@ref).
- **Cool skin:** `OceanMixedLayer.cool_skin`, iterated with the fluxes
  until ``ΔT_{cool}`` changes by less than ``10^{-6}`` K. The cool skin
  sees only the interfacial fluxes, because spray heat leaves with the
  droplets.
- **Rain and spray** are added to the fluxes.

[`coare_warm_layer_series`](@ref) is the warm-layer time loop of
`coare36vnWarm_et`:

- The Fairall warm layer is stepped with the current net shortwave and
  the previous record's heat loss and stress.
- It starts on the first local morning and is reset at local midnight.
- The warming between the surface and the sensor depth is passed to
  `coare_fluxes`.

## 4. Coupling to the water column

`OceanMixedLayer.WaterColumn` takes a `flux` hook. Given the column's
skin temperature (top layer + warm layer − cool skin),
[`coare_flux_hook`](@ref):

- computes COARE fluxes with wave-dependent roughness, and stores that
  roughness in the column so a tile aggregator sees it;
- adds spray and rain heat;
- returns ``G = R_n - H - LE - H_{rain}``, which the column absorbs
  exactly. The column's heat budget therefore stays closed.

One limitation: the column computes its own cool skin from the `H` and
`LE` it is given, so with spray the skin also "sees" the spray heat.
`coare_fluxes` keeps the two apart.

## 5. Validation (measured)

**Full algorithm vs the COARE 3.6 reference code.** Test set: 653 ship
records (the `test_36_data` of NOAA-PSL, 7 days), run with cool skin
and warm layer. Configuration: ε = 0.97, c_w = 4000, reference net
shortwave.

| quantity | max difference | range in the data |
| :-- | :-- | :-- |
| u* | 0.12 % | |
| τ | 0.53 % | |
| H | 0.68 W m⁻² | 3–40 W m⁻² |
| LE | 1.5 W m⁻² (rms 1.1, bias −0.65 %) | 90–250 W m⁻² |
| cool skin ΔT | 0.017 K | 0.18–0.37 K |
| warm-layer warming at 5.3 m | 0.0031 K | up to 0.33 K |

With the package's default fresh-water c_w (4181), the warm-layer
difference becomes 0.021 K. The warm-layer warming scales as
``c_w^{-3/2}``.

**Diurnal warming vs observations.** The observations are 42 days of
peak afternoon warming measured by the floating sensor of R/V *Moana
Wave* during TOGA COARE (Fairall et al. 1996a, Table 2). The schemes
are forced with the 7-hour means from the table.

| model | r | RMSE (K) | bias (K) |
| :-- | --: | --: | --: |
| Fairall et al. eq. (25), as printed in the paper | 0.956 | 0.194 | −0.005 |
| `FairallWarmLayer` (this code) | 0.953 | 0.246 | −0.149 |
| `ZengBeljaarsWarmLayer` | 0.897 | 0.50 | +0.34 |

The calm days (U < 2 m s⁻¹) warm by 1.73 K in the observations and by
1.40 K in the Fairall scheme. These are ship observations, not buoy
observations. Water temperature, salinity and air density for the
forcing are assumed warm-pool values (29.5 °C, 34 psu, 1.15 kg m⁻³).

**Spray vs Andreas et al. (2015).** The comparison values were
digitised from Figs. 9–11 of the extended abstract (AMS 21st Symp. on
Boundary Layers and Turbulence, 2B.3), the conference version of the
QJRMS paper; the figure numbers refer to that document. They are the authors' model
curves, not data.

- **Sensible spray flux** (independent of wave height): within 8.7 % at
  all 14 points above 50 W m⁻². The model is systematically about 8 %
  high, partly from the fresh-water c_w.
- **Latent spray flux:** needs ``H_{1/3}``, which the figures do not
  give. With one ``H_{1/3}`` (7.3 m at 25 m s⁻¹) calibrated at RH = 90 %,
  the other six RH points agree to within 23 W m⁻². With a fully
  developed sea (15.5 m), the latent flux is 1.76 times Fig. 11.
- **Sign change of the latent spray flux:** at RH = 98.24 %, against
  "about 98.1 %" in the paper. Here a_w comes from the sea-salt κ = 1.10.
- **Near saturation** (RH > 99 %) the exponential radius law with the
  e-folding ``τ_r`` breaks down, because ``r_{eq}`` grows to millimetres:
  ``H_{L,sp}`` returns to 0 at RH = 100 % (−12 W m⁻² at 99.5 %), while
  Fig. 11 stays near −20 W m⁻². Supersaturated air is treated as
  RH = 100 %.
- **Crossover winds**, where spray equals the interfacial flux, with
  COARE interfacial fluxes: 21.0 m s⁻¹ for sensible heat and
  27.75 m s⁻¹ for latent heat, against the paper's ≈ 19 and ≈ 26.

## References

- Andreas, E. L (1990). Time constants for the evolution of sea spray
  droplets. *Tellus*, 42B, 481–497.
- Andreas, E. L (1995). The temperature of evaporating sea spray
  droplets. *J. Atmos. Sci.*, 52, 852–862.
- Andreas, E. L, Persson, P. O. G. & Hare, J. E. (2008). *J. Phys.
  Oceanogr.*, 38, 1581–1596.
- Andreas, E. L, Mahrt, L. & Vickers, D. (2012). *J. Atmos. Sci.*, 69,
  2520–2537.
- Andreas, E. L, Mahrt, L. & Vickers, D. (2015). An improved bulk
  air–sea surface flux algorithm, including spray-mediated transfer.
  *Q. J. R. Meteorol. Soc.*, 141, 642–654 (extended abstract: AMS 21st
  Symp. on Boundary Layers and Turbulence, 2B.3).
- Fairall, C. W. et al. (1996a). Cool-skin and warm-layer effects on sea
  surface temperature. *J. Geophys. Res.*, 101, 1295–1308.
- Fairall, C. W. et al. (1996b). Bulk parameterization of air–sea fluxes
  for TOGA COARE. *J. Geophys. Res.*, 101, 3747–3764.
- Gosnell, R., Fairall, C. W. & Webster, P. J. (1995). The sensible heat
  of rainfall in the tropical ocean. *J. Geophys. Res.*, 100,
  18437–18442.
