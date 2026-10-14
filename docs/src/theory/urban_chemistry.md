# Urban chemistry and deposition

`Aeolus.Chemistry` carries a reduced NOx–O3–VOC mechanism for urban
boundary-layer air quality, together with resistance-model dry
deposition and below-cloud wet deposition. All mixing ratios are in ppb.
The Berlin case runs it on five advected tracers
(`src/components/urban/aq_chem.jl`, #135).

## Mechanism

| reaction | rate | source |
| :-- | :-- | :-- |
| R1 NO2 + hν → NO + O3 | ``J = l μ_0^m e^{-n/μ_0}``, ``l = 1.165×10^{-2}`` s⁻¹, ``m = 0.244``, ``n = 0.267`` | MCM v3 J4 (Saunders et al. 2003) |
| R2 NO + O3 → NO2 + O2 | ``3.0×10^{-12} e^{-1500/T}`` | JPL 19-5 (Burkholder et al. 2019) |
| R3 VOC + O3 → products | ``1.03×10^{-14} e^{-1995/T}`` (isoprene) | IUPAC (Atkinson et al. 2006) |
| R4 VOC + OH → ν NO→NO2 | ``2.7×10^{-11} e^{390/T}`` (isoprene), ν = 2 | IUPAC |
| R5 NO2 + OH (+M) → HNO3 | Troe form, ``k_0 = 1.8×10^{-30}(T/300)^{-3}``, ``k_∞ = 2.8×10^{-11}``, ``F_c = 0.6`` | JPL 19-5 |

Bimolecular constants (cm³ molecule⁻¹ s⁻¹) become ppb⁻¹ s⁻¹ through the
air number density, ``k_{ppb} = k\,M\,10^{-9}``, ``M = p/(k_B T)``. OH is
prescribed: the mechanism has no HOx budget. In R4 the peroxy radical
and the HO2 it yields each convert one NO to NO2, so each VOC oxidised by
OH makes two Ox. This is the mechanism's ozone production.

### The triad, exactly

R1–R2 conserve ``NO_x = NO + NO_2`` and ``O_x = O_3 + NO_2``. With
``y = [NO_2]`` they reduce to a Riccati equation

```math
\dot y = k (NO_x - y)(O_x - y) - J y = k (y - y_1)(y - y_2),
```

with roots ``y_1 ≤ \min(NO_x, O_x) ≤ y_2``. Its solution over a step is

```math
y(t) = y_1 + \frac{(y_0 - y_1) E}{1 - (y_0 - y_1)(1 - E)/r},
\qquad r = y_2 - y_1,\; E = e^{-k r t}.
```

`nox_triad` evaluates it with the cancellation-free root
``y_1 = 2 k NO_x O_x/(B + D)`` and takes the degenerate limit ``r → 0``
(night titration with ``NO_x = O_x``) through `expm1`. The integrator is
unconditionally stable and positive, and conserves NOx and Ox to
round-off for any time step. The test compares it with a 200 000-step RK4
reference to 1e-9. The photostationary state ``y_1`` is where the
Leighton ratio

```math
φ = \frac{J [NO_2]}{k [NO][O_3]}
```

equals 1. The relaxation time is ``1/(k r)``, about 40 s at noon. With
peroxy radicals the steady state is instead
``φ_{ext} = 1 + ν k_4[OH][VOC]/(k[NO][O_3])``.

R3–R5 are applied first, as pseudo-first-order losses over the step with
their partners frozen (exact exponentials), then the triad. This is
first-order operator splitting. `photochemistry_cell` returns the Ox
production and loss, so that ``Δ O_x = P - L`` and
``Δ NO_x = -Δ HNO_3`` hold exactly per cell.

## Dry deposition

The deposition velocity is three resistances in series (Wesely & Hicks
1977; Seinfeld & Pandis 2016, ch. 19):

```math
v_d = \frac{1}{r_a + r_b + r_c}.
```

* ``r_a = 1/(C_H U)`` is the scalar aerodynamic resistance between the
  surface and the first air cell. The caller supplies it; the Berlin
  case takes it from `SurfaceExchange.solve_surface_layer`.
* ``r_b = \frac{2}{κ u_*}(Sc/Pr)^{2/3}`` is the quasi-laminar resistance.
  ``Sc = ν/D`` uses the water-vapour diffusivity of Massman (1998) and
  the ``D_{H_2O}/D_x`` ratios of Wesely (1989).
* ``r_c`` is the surface resistance per species and surface kind
  (`surface_resistance`). For O3 and NOx the values are the reciprocals
  of the total deposition velocities the Berlin example used to carry,
  now read as the surface conductance. They are of the order of Wesely's
  midsummer values but are not a fit. HNO3 uses Wesely's floor of
  10 s m⁻¹. The lumped VOC does not deposit.

`apply_dry_deposition!` removes ``c(1 - e^{-v_d Δt/Δz})`` from the first
air cell of each column. This is the exact solution of the surface-flux
sink over the step. The first air cell is a per-column level, so the
sink sits above the roof on building columns.

## Wet deposition

Below-cloud scavenging by rain is ``dc/dt = -Λ c`` with ``Λ = s\,q_r``,
integrated exactly. For HNO3, ``s = 1`` (kg/kg)⁻¹ s⁻¹ gives
``Λ ≈ 10^{-4}`` s⁻¹ at ``q_r = 10^{-4}`` (about 2 mm h⁻¹). That is the
order of magnitude of below-cloud scavenging for a highly soluble gas,
not a fit. For NO, NO2 and O3 the Henry's-law solubility is so small
that ``s = 0``.

## Emissions

`emission_ppb_flux(F, M)` converts a mass flux (kg m⁻² s⁻¹) to a
mixing-ratio flux, ``10^9 (F/M)/(p/RT)`` ppb m s⁻¹. Spread over a first
cell of depth ``Δz``, this raises the mixing ratio by ``F_{ppb}Δt/Δz``.
The emission inventory itself (traffic NOx, park isoprene) is case data
and lives in the Berlin coupling.

## References

* Atkinson, R. et al. (2006), Evaluated kinetic and photochemical data
  for atmospheric chemistry: Volume II, *Atmos. Chem. Phys.* 6, 3625–4055.
* Burkholder, J. B. et al. (2019), *Chemical Kinetics and Photochemical
  Data for Use in Atmospheric Studies*, Evaluation No. 19, JPL
  Publication 19-5.
* Massman, W. J. (1998), A review of the molecular diffusivities of H2O,
  CO2, CH4, CO, O3, SO2, NH3, N2O, NO, and NO2 in air, O2 and N2 near
  STP, *Atmos. Environ.* 32, 1111–1127.
* Saunders, S. M. et al. (2003), Protocol for the development of the
  Master Chemical Mechanism, MCM v3 (Part A), *Atmos. Chem. Phys.* 3,
  161–180.
* Seinfeld, J. H. & Pandis, S. N. (2016), *Atmospheric Chemistry and
  Physics*, 3rd ed., Wiley.
* Wesely, M. L. (1989), Parameterization of surface resistances to
  gaseous dry deposition in regional-scale numerical models, *Atmos.
  Environ.* 23, 1293–1304.
* Wesely, M. L. & Hicks, B. B. (1977), Some factors that affect the
  deposition rates of sulfur dioxide and similar gases on vegetation,
  *J. Air Pollut. Control Assoc.* 27, 1110–1116.
