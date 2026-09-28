# Resolved plant canopy: leaf area, drag, radiation and leaf temperature

A resolved canopy is vegetation that occupies several model layers, so
that the LES sees the leaves as a distributed momentum sink rather than
as a roughness length. This page covers what `src/PlantCanopy.jl`
implements for Phase 48: the vertical distribution of leaf area, the
drag it exerts, treated implicitly so it is stable at any time step, and
the attenuation of shortwave and longwave radiation through it
(section 4), and the temperature of the leaves themselves from their
energy balance (section 5).

## 1. Leaf-area density

The leaf-area density ``a(z)`` [m² m⁻³] is one-sided leaf area per unit
volume; its column integral is the leaf-area index,
``\mathrm{LAI} = \int_0^h a\,dz``. Three profiles are provided, each
normalised to its LAI by construction:

| Profile | ``a(z)`` | Use |
| :-- | :-- | :-- |
| `UniformLAD(LAI, h; z_base)` | ``\mathrm{LAI}/(h - z_{base})`` on the crown | analytic theory, idealised cases |
| `LalicMihailovicLAD(LAI, h; z_m)` | ``L_m r^n e^{n(1-r)}``, ``r = \frac{h-z_m}{h-z}``, ``n = 6`` below ``z_m``, ``1/2`` above | forests (Lalic & Mihailovic 2004) |
| `BetaLAD(LAI, h; p, q)` | ``\frac{\mathrm{LAI}}{h}\frac{x^{p-1}(1-x)^{q-1}}{B(p,q)}``, ``x = z/h`` | flexible crown shapes (Markkanen et al. 2003) |

For Lalic–Mihailovic the peak ``L_m = a(z_m)`` is solved for so that the
profile integrates to the given LAI.

On a model grid the LAD of a layer is its **layer mean**
``a_k = Δz_k^{-1}\int_{z_k}^{z_{k+1}} a\,dz`` (`layer_leaf_area_density`),
not a point sample. With the urban example's ``Δz = 6.25`` m and a
20 m canopy the top layer spans 18.75–25 m; centre sampling puts no
leaves there and loses 6 % of the LAI, the layer integral loses none.
The integrals use Gauss–Legendre quadrature graded towards the ends of
each smooth piece, which keeps the ``(1-x)^{q-1}`` end of a beta profile
with non-integer ``q`` at round-off accuracy.

The leaf area above a height, ``Λ(z) = \int_z^h a\,dz'``
(`cumulative_leaf_area`), is the optical-depth coordinate for canopy
radiative transfer.

## 2. Canopy drag, and why it must be implicit

Form drag on the leaves is (Shaw & Schumann 1992)

```math
\frac{\partial U_i}{\partial t} = -C_d\,a\,|U|\,U_i ,
```

with ``C_d ≈ 0.15``–``0.3``. Taken alone, the speed obeys
``d|U|/dt = -C_d a |U|^2`` and the direction is fixed, so the exact
solution over a step ``Δt`` from ``U^*`` is

```math
U^{n+1} = \frac{U^*}{1 + Δt\,C_d\,a\,|U^*|}.
```

This is also backward Euler with the drag coefficient ``C_d a |U|``
lagged, but here the lagging costs nothing: for the pure drag equation
the factor is exact at any ``Δt``. It lies in ``(0, 1]``, so the step is
unconditionally stable and can never reverse the flow.

Explicit Euler, ``U^{n+1} = U^n (1 - Δt\,C_d a |U^n|)``, reverses the flow
once ``Δt\,C_d a|U| > 1`` and amplifies it once that exceeds 2. For a
dense forest (``a ≈ 1`` m⁻¹, ``C_d = 0.2``) in a 5 m s⁻¹ wind the limit is
``Δt = 2`` s — the urban example's step. The old explicit
`apply_vegetation_drag!` shipped disabled after an instability at that
step was reported, but in the urban example's own configuration
``Δt\,C_d\,\mathrm{LAI}\,|U| ≤ 0.05`` and re-running it for #22 did not
reproduce the instability; its cause is undiagnosed. The implicit form
removes the stiffness limit either way.

### In the LES

`apply_canopy_drag!(sim, drag, Δt)` runs in the `post_predictor!` hook,
on the predicted velocity ``u^*`` before the pressure projection. The
three velocity components live on different faces of the Arakawa-C
grid. At each face the LAD is the mean of the two adjacent cells, the
speed ``|U|`` includes the other two components averaged from their
four neighbouring faces, and all factors are computed before any is
applied. The call returns the momentum removed,
``\sum (U^{n+1} - U^*)ΔV/Δt``, which closes the momentum budget exactly.

## 3. In-canopy wind: the exponential profile

In a deep uniform canopy with a constant mixing length ``l`` the steady
mean momentum balance

```math
\frac{d}{dz}\left(l^2 \left|\frac{dU}{dz}\right|\frac{dU}{dz}\right) = C_d\,a\,U^2
```

has the solution ``U = U_h e^{γ(z-h)}`` with ``γ = (C_d a / 2l^2)^{1/3}``
(Inoue 1963). Harman & Finnigan (2007) express the same profile through
``β = u_*/U_h`` and the canopy-drag length ``L_c = 1/(C_d a)``: the
mixing length is ``l = 2β^3 L_c`` and ``U = U_h e^{β(z-h)/l}``, so
``γ = β/l``. Substituting shows that the stress at the canopy top
``l^2 (dU/dz)^2 = β^2 U_h^2``, i.e. the friction velocity comes back as
``β U_h``.

`solve_canopy_wind_column` solves the balance above on a 1-D column,
with ``K = l^2|∂U/∂z| + ν``, the wind prescribed at the column top, a
free-slip or no-slip ground, and the implicit drag folded into a
tridiagonal pseudo-time step. At convergence the discrete stress at the
top equals the drag on the leaves plus the ground stress.

Measured against the analytic profile (``h = 20`` m, LAI = 10,
``C_d = 0.25``, ``β = 0.3``, so ``l = 0.432`` m and ``γh = 13.9``),
compared down to ``U = e^{-5} U_h``:

| Layers | max rel. error of ``U(z)`` | ``u_*/U_h`` |
| --: | --: | --: |
| 100 | ``5.5\times10^{-3}`` | 0.29904 |
| 200 | ``1.4\times10^{-3}`` | 0.29976 |
| 400 | ``3.5\times10^{-4}`` | 0.29994 |

Convergence is second order and ``u_*/U_h → β``. The comparison needs
``ν`` far below ``l^2|∂U/∂z|``: deep in a dense canopy that viscosity
decays with ``U``, and a molecular ``ν = 1.5\times10^{-5}`` m² s⁻¹
already shifts the profile by ≈ 0.15 % at depth ``5/γ``.

What this does **not** yet establish: agreement with forest-tower or
wind-tunnel observations. That is the job of the canopy-profile
validation issue (#25), once the radiation and leaf energy balance are
in place.

## 4. Radiation in the canopy

The optical depth of a canopy is its cumulative leaf area above a height,
``Λ(z) = \int_z^h a\,dz'`` (`cumulative_leaf_area`). How much of a beam
unit leaf area intercepts depends on leaf orientation through Ross's
projection function ``G(μ)``; `RossGoudriaanLeaves(χ_L)` uses Goudriaan's
fit ``G = φ_1 + φ_2 μ`` (valid for ``-0.4 ≤ χ_L ≤ 0.6``; ``χ_L = 0`` is the
spherical distribution with ``G = 1/2``), and `HorizontalLeaves()` has
``G = μ`` exactly.

**Beer–Lambert.** The unscattered beam from a sun at zenith cosine ``μ``
reaches depth ``Λ`` with ``e^{-KΛ}``, ``K = G(μ)/μ``, and that is also the
sunlit fraction of the leaves there. For black leaves the exact diffuse
transmittance is the hemispheric integral
``T_d = 2\int_0^1 e^{-GΛ/μ} μ\,dμ`` (`canopy_diffuse_transmittance`;
``2E_3(Λ/2)`` for spherical leaves).

**Two-stream (Sellers 1985).** With leaf reflectance ``α``, transmittance
``τ_l`` and ``ω = α + τ_l``,

```math
-\bar μ \frac{dI^\uparrow}{dΛ} + [1-(1-β)ω]I^\uparrow - ωβ I^\downarrow = ω\bar μ K β_0 e^{-KΛ},\qquad
\bar μ \frac{dI^\downarrow}{dΛ} + [1-(1-β)ω]I^\downarrow - ωβ I^\uparrow = ω\bar μ K (1-β_0) e^{-KΛ},
```

with ``\bar μ = \int_0^1 μ/G\,dμ`` and the up-scatter fractions ``β``,
``β_0`` of the CLM5.0 Technical Note (§2.3.1). Dividing by ``\bar μ``
puts this in exactly the Meador–Weaver form solved by the Era III
`TwoStream` module, with optical depth ``Λ``, beam cosine ``1/K`` and

```math
γ_1 = \frac{1-(1-β)ω}{\bar μ},\quad γ_2 = \frac{ωβ}{\bar μ},\quad γ_3 = β_0,\quad γ_4 = 1-β_0 .
```

`canopy_shortwave` therefore does not re-implement the two-stream: it
supplies these coefficients as a `TwoStreamVariant` and calls
`layer_reflectance_transmittance_sw` and `adding_solver!` (which gained
a `F_dif_top` keyword for the sky diffuse entering at the canopy top).
The ground enters as the lower boundary albedo. Because the equations
depend on height only through ``Λ``, the solution does not depend on how
the leaf area is split into layers.

**Sunlit and shaded leaves.** A layer between ``Λ_t`` and ``Λ_b`` holds
``(e^{-KΛ_t} - e^{-KΛ_b})/K`` of sunlit leaf area. Sunlit leaves absorb
the fraction ``1-ω`` of the beam they intercept; the diffuse absorption
of the layer (sky diffuse plus scattered beam) is shared in proportion to
leaf area (de Pury & Farquhar 1997).

**Longwave.** `canopy_longwave` treats each layer as a slab with leaf
reflectance ``1 - ε_{leaf}``, takes ``R`` and ``T`` from the same
two-stream layer formulae, emits ``(1-R-T)σT_{leaf}^4`` in each direction
(Kirchhoff) and solves the coupled layer–ground system exactly.

`canopy_shortwave_field!` applies the column solution to each grid
column of a resolved LES canopy (independent-column approximation) and
returns the absorbed shortwave per cell in W m⁻³.

### What the tests measure

| Check | Reference | Measured |
| :-- | :-- | --: |
| ``\bar μ`` closed form (incl. series near ``χ_L = 0``) | quadrature of ``\int μ/G`` | ``6.7\times10^{-16}`` rel. |
| ``β_0`` closed form for ``a_s`` | quadrature of its integral | ``< 10^{-8}`` rel. |
| ``β_0`` at the removable singularity ``μφ_2 + G = 0`` (``χ_L = -0.4``, ``μ ≈ 0.996``) | quadrature of its integral | ``8.9\times10^{-16}`` rel. |
| Beam flux at every face | ``e^{-KΛ(z)}`` | ``< 10^{-9}`` rel. |
| Black-leaf diffuse | ``2E_3(Λ/2)`` via series ``E_1`` | ``< 10^{-10}`` rel. |
| Energy closure (576 cases; a bookkeeping identity, since absorption is the flux divergence) | incident = reflected + canopy + ground | ``3.2\times10^{-16}`` |
| Face fluxes | RK4 integration of the Sellers ODEs | ``3.4\times10^{-14}`` of incident |
| Face fluxes at the resonance ``K = k`` (``μ ≈ 0.54`` for spherical PAR leaves) | RK4 integration of the Sellers ODEs | ``1.9\times10^{-9}`` of incident |
| Deep horizontal-leaf canopy albedo | Goudriaan ``ρ_h = \frac{1-\sqrt{1-σ}}{1+\sqrt{1-σ}}`` | ``< 10^{-9}`` abs. |
| Isothermal longwave | zero net exchange | ``< 10^{-9}`` W m⁻² |

Two limitations are worth stating with numbers.

*Angular closure.* The two-stream extinguishes diffuse light at
``1/\bar μ`` per unit leaf area, which for spherical leaves is 1. The
exact black-leaf diffuse transmittance is larger: at ``Λ = 1`` it is
0.443 against the two-stream's 0.368, and at ``Λ = 3`` 0.114 against
0.050. The two-stream therefore puts more diffuse absorption in the
upper crown than an exact angular treatment. For horizontal leaves the
two coincide, since every direction sees ``K = 1``.

*Spherical-leaf canopy albedo.* For spherical leaves with ``σ = 0.15``
(PAR) over a deep canopy the two-stream albedo is 0.0454, 0.0338, 0.0254
at ``μ = 0.3, 0.6, 1``; the widely used approximation
``ρ_{cb} = 1 - e^{-2ρ_h K/(1+K)}`` gives 0.0495, 0.0362, 0.0267, i.e. the
two differ by 5–8 %.

Not validated here: comparison with measured within-canopy radiation
profiles or with published tables of another canopy two-stream code.

## 5. Leaf energy balance and leaf temperature

Leaves are not at air temperature. A sunlit leaf absorbs several hundred
W m⁻² and can only lose it by emission, by conduction through its
boundary layer, and by transpiration; the balance sets how far it sits
above the air. Per unit (one-sided) leaf area:

```math
C\,\frac{dT_\ell}{dt} = R_{abs} - ε_e σ T_\ell^4 - c_p g_H (T_\ell - T_a) - λ E,
\qquad E = \frac{g_s g_v}{g_s + g_v}\,\frac{e_s(T_\ell) - e_a}{p}.
```

``R_{abs}`` is the absorbed short- and longwave radiation, ``ε_e`` the
emission coefficient (``2ε`` for a flat leaf, which emits from both
sides), ``c_p`` the molar heat capacity of air, ``g_H`` the boundary-layer
conductance to heat of both leaf sides and ``g_v`` that to water vapour
of the sides with stomata, ``g_s`` the stomatal conductance, all in
mol m⁻² s⁻¹ (Campbell & Norman 1998, ch. 14). ``e_s`` and ``λ`` come from
`MoistThermodynamics`, the transpiration and the coupled
photosynthesis–stomata solution from `Photosynthesis`: with
`CoupledLeafPhysiology` the stomata, the biochemistry and the energy
balance all run at the same leaf temperature and see the same boundary
layer.

**Boundary layer.** `CampbellNormanBoundaryLayer` uses Campbell &
Norman's flat-plate conductances per leaf side, forced
``g_{Ha} = 0.135\sqrt{u/d}``, ``g_{va} = 0.147\sqrt{u/d}`` (× 1.4 for outdoor
turbulence, ``d = 0.72 ×`` leaf width), and free convection
``g_{Ha} = 0.05\,(ΔT/d)^{1/4}``, taking the larger. These coefficients are
the laminar flat-plate correlations (``\overline{Nu} = 0.664\,Re^{1/2}Pr^{1/3}``
forced, ``0.54\,Ra^{1/4}`` free; Incropera et al. 2007) evaluated for air
at 20 °C: the test re-derives them from tabulated air properties to within
1.2 % (forced) and 2.8 % (free). `CLMLeafBoundaryLayer`
is CLM5's ``C_v\sqrt{U/d}`` (delegated to `Photosynthesis`). For a 4 cm
leaf in 1 m s⁻¹ wind the two give 1.89 and 2.08 mol m⁻² s⁻¹ for heat.

**Solution.** The equation is scalar in ``T_\ell``; it is solved on a
bracket with the Illinois method of `SurfaceExchange`. With a heat
capacity ``C`` and a step ``Δt`` it is a backward-Euler step, stable at any
``Δt``; ``Δt → ∞`` is the diagnostic (steady) balance. For a small
departure the leaf relaxes with ``τ = C/[c_p(g_H + g_r) + λ g_w s/p]``,
about 14 s for ``C = 1.3`` kJ m⁻² K⁻¹ at 1 m s⁻¹, so over an LES time
step of seconds a leaf is close to, but not at, equilibrium.

**Penman–Monteith limit.** Linearising ``σT^4`` (radiative conductance
``g_r = 4ε_eσT_a^3/c_p``) and ``e_s`` (slope ``s``) gives

```math
T_\ell - T_a = \frac{R_{ni} - λ g_w D/p}{c_p (g_H + g_r) + λ g_w s/p},
\qquad R_{ni} = R_{abs} - ε_eσT_a^4,
```

which is `leaf_penman_monteith`. The exact balance approaches it
quadratically: at ``R_{ni} = 300`` W m⁻², ``D = 1`` kPa the difference is
0.028 K in 2.24 K, and it drops by 4.00 ± 0.01 per halving of the
forcing.

**In the canopy.** `canopy_leaf_energy_balance` solves a sunlit and a
shaded leaf class in every layer. The shortwave is `canopy_shortwave`'s
sunlit/shaded absorption (any number of bands), the PAR band is turned
into photons for photosynthesis, and the longwave is `canopy_longwave`
with each layer at the leaf-area-weighted ``T^4`` of its two classes.
Because the leaves exchange longwave radiation with each other, the
layers are coupled; an outer iteration holds each layer's *absorbed*
longwave fixed while the classes solve their own balances with their own
emission ``ε_e = 2(1 - R_k - T_k)/ΔΛ_k`` (the two-stream layer
emissivity, ``2ε_{leaf}`` for thin layers of spherical leaves), and
repeats until no temperature moves by more than 1e-9 K (about 20
iterations). The sensible heat and transpiration of each layer, divided
by its depth, are the heat (W m⁻³) and vapour (kg m⁻³ s⁻¹) sources for
the air.

### What the tests measure

`test/era5_leaf_energy.jl`:

| Check | Reference | Measured |
| :-- | :-- | --: |
| C&N boundary-layer coefficients 0.135/0.147 (forced), 0.050/0.055 (free) | laminar flat-plate correlations, air at 20 °C | 0.1359/0.1488, 0.0486/0.0539 (≤ 2.8 %) |
| Single-leaf energy closure (300 cases: fixed and coupled stomata, both boundary layers, wind, light, humidity, T) | ``R_{abs} = εσT^4 + H + LE`` | ``1.7\times10^{-14}`` rel. |
| Isothermal net radiation zero in saturated air | ``T_\ell = T_a`` | ``< 10^{-9}`` K |
| No radiation, ``g_s → ∞``, ``g_H = g_v`` | psychrometer equation by bisection | ``< 10^{-6}`` K (``ΔT = -8.74`` K at 40 % RH, 25 °C) |
| Linearised limit, three regimes, forcing halved 3× | Penman–Monteith | error ratio 3.99–4.01 per halving; 0.028 K at full forcing |
| Backward-Euler step from ``T^* + δ`` | ``1/(1 + Δt/τ)`` | ``< 10^{-3}`` rel. for ``Δt/τ = 0.1, 1, 10`` |
| Coupled physiology | `leaf_gas_exchange` at the returned ``T_\ell`` | ``g_s``, ``E``, ``A_n`` to ``10^{-12}`` rel. |
| Canopy column closure (24 cases, day and night) | SW + LW in = ``H + LE`` + storage | ``4.7\times10^{-11}`` rel.; worst layer ``1.8\times10^{-9}`` W m⁻² |
| Each canopy leaf class | single-leaf balance with the final longwave | ``4.3\times10^{-10}`` K |
| Isothermal night (black sky and ground at ``T_a``, saturated) | all leaves at ``T_a`` | ``< 10^{-9}`` K |
| Prognostic march (80 × 120 s from 290 K) | stored energy = integrated imbalance; diagnostic steady state | ``10^{-8}`` rel.; ``2.4\times10^{-10}`` K |

### Comparison with observations

*Leuzinger & Körner (2007, Table 1)* measured canopy leaf temperatures
with a thermal camera from the Swiss Canopy Crane (35 m deciduous forest,
47.5° N) on 18 July 2004, 11:30 true local time, clear sky, ample soil
water, 25.0 °C, VPD 19.7 hPa. The broad-leaved species were on average
1.7 (Prunus), 2.4 (Quercus), 4.5 (Carpinus), 4.6 (Fagus) and 4.8 K (Tilia)
above the air, with crown maxima 6.4–12.0 K above it. The paper does not
report the radiation components or the wind, so the model forcing is an
**assumption**: the solar geometry of the date and time (cos θ = 0.888),
780 + 120 W m⁻² direct + diffuse, LAI 5 (the paper's own assumption),
45 mm leaves, CLM5 broadleaf-deciduous Medlyn stomata and a canopy-top
wind of 0.5–2 m s⁻¹. The leaf temperature seen from above (layers
weighted by their visible leaf area) is then 3.4, 2.6 and 1.9 K above
the air for ``U_h`` = 0.5, 1 and 2 m s⁻¹, inside the observed range of
species means, and the warmest sunlit leaves are 8.7, 8.2 and 7.5 K above
it, inside the observed range of crown maxima. At 4 m s⁻¹ the view mean
falls to 1.4 K, below the range. This is a check of magnitude, not a
validation: the wind and radiation were not measured with the
temperatures, and the ranges span several species.

*Ansari & Loomis (1959)*, as summarised by Leuzinger & Körner, found
sunlit thin leaves 6–10 K above the air in still air and 3–5 K in
moderate wind (5 m s⁻¹). For a horizontal 5 cm leaf under a 1000 W m⁻²
overhead sun (absorptance 0.5, assumed), VPD 2 kPa and
``g_s = 0.2``–``0.4`` mol m⁻² s⁻¹, the model gives 9.1–7.7 K in still air
(free convection only), inside the first range. At 5 m s⁻¹ it gives
1.05 K, **below** the second range. The boundary-layer conductance at
5 m s⁻¹ is 4.5 mol m⁻² s⁻¹, and 3–5 K would need either a much smaller
conductance or far less transpiration than assumed. The test does not
assert this second figure.

### Limitations

- The Campbell & Norman coefficients are for 20 °C and 101.3 kPa; their
  weak dependence on temperature and pressure is neglected.
- Free and forced convection are not combined; the larger one is used.
- Dew and intercepted water on leaves are not modelled; a negative ``E``
  goes through the stomatal path only. Wet leaves belong to the land
  column (#21).
- In a prognostic step each class keeps its own previous temperature.
  When the sun moves, the leaves that change class carry their old
  class's temperature, which is not tracked. The column's energy is
  conserved exactly only while the class leaf areas do not change.
- Air temperature, humidity and wind are inputs per layer; the feedback
  of the leaf sources on the in-canopy air is the LES's (or #25's) job.

## References

- Ansari, A. Q. & Loomis, W. E. (1959). Leaf temperatures. *Am. J.
  Bot.*, 46, 713–717 (cited from Leuzinger & Körner 2007).
- Campbell, G. S. & Norman, J. M. (1998). *An Introduction to
  Environmental Biophysics*, 2nd ed. Springer, ch. 7 and 14.
- CLM5.0 Technical Note (Lawrence et al., NCAR), §2.3.1, Canopy
  Radiative Transfer.
- de Pury, D. G. G. & Farquhar, G. D. (1997). Simple scaling of
  photosynthesis from leaves to canopies without the errors of big-leaf
  models. *Plant Cell Environ.*, 20, 537–557.
- Goudriaan, J. (1977). *Crop Micrometeorology: A Simulation Study.*
  Pudoc, Wageningen.
- Harman, I. N. & Finnigan, J. J. (2007). A simple unified theory for
  flow in the canopy and roughness sublayer. *Boundary-Layer Meteorol.*,
  123, 339–363.
- Incropera, F. P., DeWitt, D. P., Bergman, T. L. & Lavine, A. S.
  (2007). *Fundamentals of Heat and Mass Transfer*, 6th ed. Wiley.
- Inoue, E. (1963). On the turbulent structure of airflow within crop
  canopies. *J. Meteorol. Soc. Japan*, 41, 317–326.
- Leuzinger, S. & Körner, C. (2007). Tree species diversity affects
  canopy leaf temperatures in a mature temperate forest. *Agric. For.
  Meteorol.*, 146, 29–37.
- Lalic, B. & Mihailovic, D. T. (2004). An empirical relation describing
  leaf-area density inside the forest for environmental modeling.
  *J. Appl. Meteorol.*, 43, 641–645.
- Markkanen, T., Rannik, Ü., Marcolla, B., Cescatti, A. & Vesala, T.
  (2003). Footprints and fetches for fluxes over forest canopies with
  varying structure and density. *Boundary-Layer Meteorol.*, 106, 437–459.
- Monteith, J. L. & Unsworth, M. H. (2013). *Principles of Environmental
  Physics*, 4th ed. Academic Press.
- Ross, J. (1981). *The Radiation Regime and Architecture of Plant
  Stands.* Junk, The Hague.
- Sellers, P. J. (1985). Canopy reflectance, photosynthesis and
  transpiration. *Int. J. Remote Sens.*, 6, 1335–1372.
- Shaw, R. H. & Schumann, U. (1992). Large-eddy simulation of turbulent
  flow above and within a forest. *Boundary-Layer Meteorol.*, 61, 47–64.
