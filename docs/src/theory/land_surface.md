# The vegetated land column

The [stomatal conductance](stomatal_conductance.md) page ends at a single
leaf, with its temperature, the air around it and the soil-water stress
factor β all given as inputs. This page closes those three loops for a
whole vegetated column. The code is `src/LandSurface.jl` (Phase 47, #21):
a two-big-leaf canopy over the [soil column](soil_water.md) with its
[heat and freeze–thaw](soil_heat.md), coupled to the atmosphere through
[Monin–Obukhov similarity](surface_exchange.md). The formulation is the
one of CLM4.5 (Oleson et al. 2013, §4, §5.3, §7.4, §8), and each piece is
delegated to the Aeolus module that owns it.

## 1. Soil-water stress

Roots are distributed with depth after Zeng (2001). The fraction of
roots above depth ``z`` is

```math
Y(z) = 1 - \tfrac12\left(e^{-r_a z} + e^{-r_b z}\right),
```

so layer ``i`` holds ``r_i = Y(z_{h,i}) - Y(z_{h,i-1})``, and the bottom
layer also takes the roots below the column (CLM4.5 eq. 8.30, Table 8.3).
Each layer gets a wilting factor that falls linearly in the matric
potential, from 1 at ``ψ_o`` (stomata fully open) to 0 at ``ψ_c`` (fully
closed), and is scaled by the unfrozen pore fraction (eq. 8.27):

```math
w_i = \frac{ψ_c - ψ_i}{ψ_c - ψ_o}\,\frac{θ_{sat} - θ_{ice,i}}{θ_{sat}} ∈ [0, 1],
\qquad β_t = \sum_i r_i w_i .
```

A layer colder than ``T_f - 2`` K, or with no liquid water, contributes
nothing. ``ψ_o`` and ``ψ_c`` come from CLM4.5 Table 8.1, converted from mm
to m: −66/−255 m for needleleaf and broadleaf evergreen trees, −35/−224 m
for broadleaf deciduous trees, −83/−428 m for shrubs and −74/−275 m for
grasses and crops. β_t is the factor that `Photosynthesis.leaf_gas_exchange`
applies to ``V_{cmax}``, ``R_d`` and ``g_0``.

The transpiration ``E_t`` leaves the soil through the roots, layer by
layer, with the effective root fraction ``r_i w_i/β_t`` (eqs. 7.104,
7.107). Wetter layers with more roots supply more, and the sink passed to
the Richards solver sums exactly to ``E_t/ρ_w``. When β_t = 0 no layer can
supply water, and the column shuts transpiration off.

## 2. Radiation of a two-big-leaf canopy

Leaves are spherically distributed (projection ``G = 1/2``), so a beam at
``μ``, the cosine of the solar zenith angle, is extinguished with ``k_b = Ω/(2μ)``. The sunlit leaf
area is ``L_{sun} = (1 - e^{-k_b L})/k_b`` and the rest is shaded
(Goudriaan 1977; de Pury & Farquhar 1997). In each waveband (visible and
near-infrared, with leaf scattering ``σ = ρ_{leaf} + τ_{leaf}`` from CLM4.5
Table 3.1):

- scattering is folded into ``k' = k\sqrt{1-σ}``, after a canopy reflection
  ``ρ_{cb} = 1 - \exp[-2ρ_h k_b/(1 + k_b)]``, with
  ``ρ_h = (1-\sqrt{1-σ})/(1+\sqrt{1-σ})``;
- diffuse light from an isotropic sky is the hemispheric integral of the
  beam. Its effective ``k_d'`` is chosen so that ``e^{-k_d' L}`` equals
  the exact transmittance ``2∫_0^1 e^{-k_b'(μ)L} μ\,dμ``. Its reflection
  ``ρ_{cd}`` is the hemispheric mean of ``ρ_{cb}``;
- sunlit leaves absorb the unscattered beam, and the diffuse and scattered
  light weighted by the sunlit fraction ``e^{-k_b l}``. The integrals of
  de Pury & Farquhar (1997) are evaluated in closed form. Shaded leaves
  absorb the rest of the canopy absorption;
- the ground reflects its albedo's share of what reaches it. Of that, the
  fraction ``e^{-k_d' L}`` escapes and the shaded leaves absorb the rest.
  There is one ground reflection.

Shortwave is conserved exactly: sunlit + shaded + ground + reflected equals
the incident flux. The tests measure a residual of 5.7 × 10⁻¹⁴ W m⁻², and
the sunlit/shaded closed forms agree with quadrature in depth to 10⁻⁶.

Longwave follows CLM4.5 eqs. 4.16-4.20. The canopy emissivity is
``ε_v = 1 - e^{-(L+S)}``, and there is one reflection at the ground.
Canopy, ground and upward fluxes sum to ``L↓``, and an isothermal
enclosure exchanges nothing. The column's effective emissivity is
``1 - (1-ε_g)(1-ε_v)^2``. The model has no stems in the shortwave and does
not use CLM's leaf-orientation index χ_L.

Leaf capacity declines through the canopy as ``e^{-K_n x}``. It is
integrated separately over the sunlit and the shaded leaf area (CLM4.5
eqs. 8.19-8.23), with ``K_n = 0.30``: the value CLM4.5 adopted for the
sunlit/shaded canopy, which offsets the excess absorption of scattered
light by the shaded big leaf (Bonan et al. 2012). Each big leaf then calls
`leaf_gas_exchange` with its absorbed PAR per unit leaf area.

## 3. The canopy air space

Leaves, ground and atmosphere exchange heat and vapour through a canopy
air space with no storage. Flux continuity gives its temperature and
humidity (CLM4.5 eqs. 5.90, 5.104):

```math
T_{ac} = \frac{c_a θ_a + c_g T_g + c_v T_v}{c_a + c_g + c_v},\qquad
q_{ac} = \frac{c_a^w q_a + c_g^w q_{sat}(T_g) + c_v^w q_{sat}(T_v)}{c_a^w + c_g^w + c_v^w}.
```

The conductances (m s⁻¹) are:

| conductance | from | owner |
| :-- | :-- | :-- |
| ``c_a = C_H U``, ``c_a^w = C_E U`` | MOST between the canopy air at ``d + z_0`` and the reference height ``z - d`` | `SurfaceExchange.solve_surface_layer` |
| ``c_v = (L+S) g_b`` | leaf boundary layer at ``U = u_*`` | `Photosynthesis.leaf_boundary_layer_conductance` |
| ``c_v^w = Σ_{sun,sha} L_x g_b g_{s,x}/(g_b + g_{s,x})`` | boundary layer and stomata in series (eqs. 5.106, 5.110) | `Photosynthesis.leaf_gas_exchange` |
| ``c_g = C_s u_*``, ``c_g^w = β_{soil} c_g`` | Zeng et al. (2005) / Sakaguchi & Zeng (2009), eqs. 5.115-5.119; Lee & Pielke β | here; `SoilColumn.moisture_availability` |

Roughness and displacement are ``R_{z0m} z_{top}`` and ``R_d z_{top}`` for
a full canopy, blended towards the bare ground for ``L + S < 2`` (eqs.
5.126-5.128, Table 5.1).

**Leaf energy balance.** The canopy stores no heat, so
``R_{n,v}(T_v) = H_v(T_v) + λE_v(T_v)``. Here ``H_v`` and ``E_v`` are
CLM4.5 eqs. 5.94 and 5.111, and ``R_{n,v}`` is the absorbed shortwave plus
the canopy longwave. It is solved by Newton's method with the conductances
fixed. An outer iteration updates the surface layer, the stomata and the
canopy air until ``T_v`` and ``T_{ac}`` settle to 10⁻⁴ K. The canopy-air
update is under-relaxed after 10 iterations, and the relaxation halves
whenever the change grows. The transpiration/dew regime is fixed for each
Newton solve. When the leaves are colder than the canopy air's dew point,
dew forms on all leaf and stem area and goes to the soil surface, because
the column has no interception store.

**Ground.** The ground skin is the soil column's own. With the leaf
temperature held, as CLM does after its canopy iteration, the ground's
``H_g``, ``E_g`` and ``R_{n,g}`` are linearised in ``T_g`` and handed to
`step_soil_heat!` as a `LinearizedSkinBalance`. That solves the skin and
soil temperatures implicitly. The ground fluxes are then evaluated at the
step-mean skin temperature, so

```math
R_n - H - LE - G = R_{n,v} - H_v - λ E_v ,
```

which is the leaf residual alone: ≤ 4 × 10⁻¹¹ W m⁻² in every test.
Finally `step_soil_water!` takes rain and leaf dew as input, ground
evaporation as the surface demand and transpiration as the root sink. Over
four idealised days, water is conserved to 3 × 10⁻¹⁶ m.

## 4. Drought: the physics tests

The column has the drought response the stress function is meant to give.
The test forces one clear day at noon with a uniform soil potential, using
a broadleaf deciduous tree with ψ_c = −224 m:

| ψ (m) | β_t | LE (W m⁻²) | H (W m⁻²) | Bowen ratio | GPP (µmol m⁻² s⁻¹) | T_v − T_a (K) |
| --: | --: | --: | --: | --: | --: | --: |
| −3.4 | 1.000 | 366 | 90 | 0.25 | 27.4 | 1.8 |
| −50 | 0.921 | 347 | 106 | 0.31 | 26.2 | 2.0 |
| −100 | 0.656 | 292 | 155 | 0.53 | 20.6 | 2.6 |
| −150 | 0.392 | 197 | 241 | 1.22 | 12.4 | 3.5 |
| −200 | 0.127 | 77 | 351 | 4.56 | 3.9 | 4.6 |
| −260 | 0 | 14 (soil only) | 409 | 29.0 | 0 | 5.2 |

β_t, LE and GPP fall monotonically as the soil dries. H, the Bowen ratio
and the leaf-to-air temperature difference rise monotonically. Beyond ψ_c
the transpiration is exactly zero. In a 30-day rain-free dry-down of a C3
grass from field capacity, β_t never increases from one day to the next,
the midday Bowen ratio rises from 0.11 to 0.17 as β_t falls to 0.82 by
day 30 (132 mm of evapotranspiration), and the soil loses exactly the
evaporated water plus drainage, less dew.

## 5. Validation against FLUXNET

The column is forced with the measured meteorology of five FLUXNET2015
towers, taken from the PLUMBER2 v1.0 files (Ukkola & Abramowitz 2020,
CC BY 4.0). The inputs are hourly means of the gap-filled half-hourly
forcing and the Copernicus LAI of those files. The sun comes from
`SolarGeometry`, and the diffuse fraction from the Reindl correlation.
The model uses the CLM4.5 PFT parameters without site calibration. The
soil is initialised at field capacity. Soil texture comes from the site
metadata where it is given, and is otherwise assumed to be a loam. After
a 5-day spin-up, hourly LE, H and Rn are compared with the hours whose
two half-hours were both *measured* (quality flag 0).

| site | biome, PFT | window | Rn r / bias | LE r / bias | daytime EF model / obs |
| :-- | :-- | :-- | :-- | :-- | :-- |
| DE-Hai | DBF, `:bdt_temperate` | 15 Jun–14 Jul 2008 | 0.998 / +10.1 | 0.938 / +47.4 | 0.786 / 0.569 |
| FI-Hyy | ENF, `:net_boreal` | 1–30 Jul 2008 | 0.980 / −10.2 | 0.924 / +10.4 | 0.579 / 0.602 |
| GF-Guy | tropical EBF, `:bet_tropical` | 15 Sep–14 Oct 2007 | 0.998 / −1.5 | 0.939 / +7.9 | 0.655 / 0.687 |
| DE-Gri | GRA, `:c3_grass` | 15 Jun–14 Jul 2008 | 0.994 / +8.7 | 0.911 / +87.0 | 1.017 / 0.725 |
| FR-Pue | Mediterranean EBF, `:bet_temperate` | 6 May–31 Aug 2006 | 0.999 / −6.0 | 0.826 / +68.1 | 0.568 / 0.221 |

(Biases are in W m⁻². EF is ``ΣLE/Σ(H+LE)`` over daytime hours. The
Bowen-ratio-preserving closure correction of PLUMBER2 leaves it unchanged,
so it compares directly with a model that closes its energy budget.)

**What it gets right.** Net radiation is reproduced everywhere, and the
diurnal cycle of LE is well correlated. The evaporative fraction is within
0.03 of the observations at the boreal pine (FI-Hyy) and the tropical
rainforest (GF-Guy). At Puéchabon the soil dries through the summer of
2006, and the midday Bowen ratio rises from May to June to July in both
the observations (2.78, 3.78, 5.49) and the model (0.52, 1.29, 1.61).

**What it gets wrong.** LE is too high, and so is EF, at the beech forest
(DE-Hai), the grassland (DE-Gri) and Puéchabon. At Puéchabon the size of
the error follows from an assumption, not from the physics. The model
starts at field capacity in a 2.7 m loam, but the site is a shallow,
rocky karst soil that is already water-limited in May: the observed
Bowen ratio is already 2.8. The model's drought response therefore
starts too late. At DE-Gri the Copernicus LAI of 4.8 and the CLM5 C3
grass slope ``g_1 = 5.25`` give a canopy conductance of about
35 mm s⁻¹, and LE exceeds the available energy in the afternoons.

With the CLM4.5 Ball-Berry closure instead of Medlyn, the EF at DE-Hai is
0.584 and at DE-Gri 0.833. Measured values for all five sites are printed
by the test.

Like the land-surface models in PLUMBER (Best et al. 2015), the column does
not beat the in-sample linear regression of the flux on shortwave ("1lin")
in RMSE for LE or H at any of these sites.

## References

  * Oleson, K. W. et al. (2013). *Technical description of version 4.5 of
    the Community Land Model (CLM)*. NCAR/TN-503+STR.
  * Zeng, X. (2001). Global vegetation root distribution for land
    modeling. *J. Hydrometeorol.*, 2, 525-530.
  * Zeng, X., Barlage, M., Castro, C. & Fling, K. (2005). *J.
    Hydrometeorol.*, 6, 979-989.
  * Sakaguchi, K. & Zeng, X. (2009). *J. Geophys. Res.*, 114, D01107.
  * Goudriaan, J. (1977). *Crop Micrometeorology: A Simulation Study*.
    Pudoc.
  * de Pury, D. G. G. & Farquhar, G. D. (1997). *Plant Cell Environ.*,
    20, 537-557.
  * Bonan, G. B. et al. (2011). *J. Geophys. Res.*, 116, G02014.
  * Bonan, G. B. et al. (2012). *J. Geophys. Res.*, 117, G02026.
  * Ukkola, A. & Abramowitz, G. (2020). PLUMBER2 v1.0, NCI,
    doi:10.25914/5fdb0902607e1; Ukkola, A. M. et al. (2022). *Earth Syst.
    Sci. Data*, 14, 449-461.
  * Pastorello, G. et al. (2020). The FLUXNET2015 dataset and the ONEFlux
    processing pipeline for eddy covariance data. *Sci. Data*, 7, 225.
  * Best, M. J. et al. (2015). The plumbing of land surface models:
    benchmarking model performance. *J. Hydrometeorol.*, 16, 1425-1442.
