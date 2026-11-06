# Air–sea fluxes and wave-dependent roughness

Over land the roughness length is a property of the ground. Over water
it is not: the roughness elements are the waves, which the wind itself
raises, so ``z_0`` depends on the friction velocity it helps determine,
and on how developed the sea is. This page describes `src/AirSeaFlux.jl`
(Phase 52, issue #39), which supplies that roughness and the COARE bulk
algorithm around it. The Monin–Obukhov solve itself is the Era V
contract's (see [Surface exchange](surface_exchange.md)); nothing here
restates it.

## 1. Momentum roughness

### Charnock and the smooth-flow limit

Dimensional analysis for a fully rough, wave-covered surface gives
Charnock's (1955) ``z_0 = \alpha u_*^2/g``. At low wind the surface is
aerodynamically smooth and the viscous sublayer sets ``z_0 = 0.11\nu/u_*``.
Smith (1988) adds the two:

```math
z_0 = \alpha\frac{u_*^2}{g} + 0.11\frac{\nu}{u_*},
\qquad R_r = \frac{z_0 u_*}{\nu}.
```

The drag coefficient ``C_{D10N} = [\kappa/\ln(10/z_0)]^2`` therefore has
a minimum near 3 m s⁻¹ — rising toward calm (smooth flow) and toward
storms (Charnock). Because ``z_0`` depends on ``u_*``, even the neutral
problem ``u_* = \kappa U/\ln(z/z_0(u_*))`` is a fixed point;
`neutral_transfer_coefficients` solves it by a bracketed secant, taking
the lower root (for steep wave laws, ``z_0 \propto u_*^{4.5}``, a second,
unstable root appears where ``z_0 \to z``).

### Wind-dependent Charnock: COARE 3.0 and 3.5

| law | α | source |
|:-- |:-- |:-- |
| `CharnockRoughness` | 0.011 (0.018 in ECMWF) | Smith 1988; COARE 2.5 |
| `COARE30Charnock` | 0.011 → 0.018 linearly, ``U_{10N}`` = 10 → 18 m s⁻¹ | Fairall et al. 2003 |
| `COARE35Charnock` | ``0.0017\,U_{10N} - 0.005``, constant above 19 m s⁻¹ (≈ 0.028) | Edson et al. 2013 (+ corrigendum) |

### Wave age and wave steepness

A wind sea at fetch or duration limit is *young*: short, slow, steep
waves that are rougher than a mature sea at the same wind. The wave age
``c_p/u_*`` (phase speed of the peak waves over friction velocity) is
the usual measure; a fully developed sea has ``c_p/u_* \approx 32``
(Donelan 1990), equivalently ``c_p/U_{10N} \approx 1.14``–1.2.

| law | ``z_0`` (rough part) | source |
|:-- |:-- |:-- |
| `WaveAgeCharnock` | ``0.114\,(u_*/c_p)^{0.622}\;u_*^2/g`` | Edson et al. 2013 (COARE 3.5) |
| `OostRoughness` | ``\frac{50}{2\pi}\lambda_p (u_*/c_p)^{4.5}`` ⇔ ``\alpha = 50 (c_p/u_*)^{-2.5}`` | Oost et al. 2002 |
| `TaylorYellandRoughness` | ``1200\,H_s (H_s/\lambda_p)^{4.5}`` | Taylor & Yelland 2001 |
| `SeaStateRoughness` | ``0.091\,H_s (u_*/c_p)^2`` | Edson et al. 2013 (COARE 3.5) |

each plus the smooth-flow term. The Oost form's equivalence of Charnock
parameter and roughness length rests on the deep-water dispersion
relation ``g\lambda_p/2\pi = c_p^2``, which the tests check exactly.
Wave information enters as a `SeaState(c_p, H_s, λ_p)`, built from a
peak period by `deep_water_sea_state` or, for a fully developed sea,
from ``H_s = 0.0248\,U_{10N}^2``, ``T_p = 0.729\,U_{10N}``
(`fully_developed_sea_state`; Taylor & Yelland 2001).

Edson et al. (2013) found the wind-speed and wave-age COARE 3.5 forms
give similar open-ocean drag because inverse wave age grows nearly
linearly with wind there; the wave forms matter in coastal and
fetch-limited water, lakes, and swell.

## 2. Scalar roughness

Heat and water vapour cross the last millimetre by molecular diffusion
into surface-renewing eddies, not by form drag, so ``z_{0t}`` and
``z_{0q}`` are not ``z_0``. Liu, Katsaros & Businger (1979) expressed
them as functions of ``R_r``; COARE 3.0 replaced their tables by a fit
(Fairall et al. 2003, eq. 28),

```math
z_{0q} = z_{0t} = \min\!\left(1.1\times10^{-4},\; 5.5\times10^{-5} R_r^{-0.6}\right),
```

and COARE 3.5 refits it (``1.6\times10^{-4}``, ``5.8\times10^{-5}``,
``-0.72``) to keep the Dalton number of 3.0 with the rougher 3.5 ``z_0``.
The Brutsaert–Garratt form ``z_{0q} = z_0\exp(2 - 2.28R_r^{1/4})`` is
also provided. The scalar roughness *falls* as the surface roughens —
breaking and spray aside, a rougher sea does not transfer heat
proportionally faster.

## 3. Convective stability functions and gustiness

In free convection (``\zeta \to -\infty``) local similarity requires
``\phi \propto (-\zeta)^{-1/3}``, which the Kansas forms
(``-1/4``, ``-1/2``) violate. COARE blends them into a convective form
(Fairall et al. 1996; Grachev & Fairall 1997):

```math
\psi = (1-f)\,\psi_K + f\,\psi_C,\qquad f = \frac{\zeta^2}{1+\zeta^2},
\qquad
\psi_C = \tfrac32\ln\frac{y^2+y+1}{3} - \sqrt3\arctan\frac{2y+1}{\sqrt3} + \frac{\pi}{\sqrt3},
```

``y = (1-\gamma_c\zeta)^{1/3}``, ``\gamma_c = 10.15`` (momentum) and
34.15 (scalars). `COAREStability` implements this as a further
`AbstractStabilityFunctions`; its Kansas and stable (Beljaars–Holtslag)
parts are the contract's `BusingerDyerBeljaarsHoltslag` with COARE's
coefficients. The analytic ``\phi`` of the blend,
``\phi = (1-f)\phi_K + f\phi_C - \zeta f'(\psi_C-\psi_K)``, is checked
against ``d\psi/d\zeta = (1-\phi)/\zeta`` to ``2\times10^{-10}``.

As ``U \to 0`` the mean-wind shear vanishes but boundary-layer eddies
still sweep air across the surface. The gustiness
``w_g = \beta\,(B z_i)^{1/3}`` (β = 1.2, ``B`` the surface buoyancy flux,
``z_i`` the mixed-layer depth) enters the effective wind
``S = \sqrt{U^2 + w_g^2}``, keeping the fluxes finite; in stable
conditions ``w_g = 0.2`` m s⁻¹. Stress is ``\tau = \rho u_*^2 U/S``: the
gust part has no mean direction.

## 4. The bulk algorithm

`air_sea_fluxes` iterates: roughness from ``u_*`` (and ``U_{10N}`` or the
sea state) → exact Monin–Obukhov solve (`solve_surface_layer`, with
`COAREStability` and gustiness) → buoyancy flux, new gustiness and
``U_{10N}``, until ``u_*`` and ``w_g`` change by less than 10⁻⁹. Since the
stability is solved exactly inside each pass, the outer loop only
converges the roughness–gustiness feedback (10–18 passes).

`SeaSurface` wraps this as an `AbstractSurface` (seawater humidity
``e = 0.98\,e_{sat}``, water emissivity from `SurfaceOptics`) so the tile
framework can aggregate open water with the other Era V surfaces. Its
skin temperature is prescribed; the ocean mixed layer, warm layer and
cool skin that evolve it belong to `OceanMixedLayer`.

## 5. Validation

Measured with `test/air_sea_flux.jl`:

| check | model | reference |
|:-- |:-- |:-- |
| COARE 3.0 ``10^3 C_{D10N}`` at 20 m s⁻¹ | 2.071 | 2.06 (Fairall et al. 2003) |
| COARE 3.0 ``10^3 C_{E10N}`` at 5 / 18 m s⁻¹ | 1.107 / 1.210 | ≈ 1.08 / 1.2 measured (Fairall et al. 2003) |
| COARE 3.5 ``u_*``, 10–25 m s⁻¹ | within 6.9 % | fit ``u_* = 0.062U_{10N} - 0.28`` to data (Edson et al. 2013, eq. 22) |
| wave-age α at ``c_p/u_* = 32``, 43, 20 | 0.0132, 0.0110, 0.0177 | 0.013; 0.011–0.018 (Edson et al. 2013) |
| NOAA COARE 3.5 reference test file, 116 records | ``u_*`` 0.27 %, τ 0.58 %, LE 0.25 %, ``C_{D10N}`` 0.66 %, L 2.0 % (max) | reference output |
| NOAA COARE 3.5 reference, 60 synthetic stable/unstable/wave cases | ``u_*, θ_*, q_*`` < 0.3 %, L 1.2 % | reference output |

Sensible heat is 1.5 % above the reference file throughout: COARE uses
the dry-air ``c_p = 1004.67`` J kg⁻¹ K⁻¹, the model the moist ``c_{pm}``
of `MoistThermodynamics`; with the dry value the difference is 0.23 %.

What this does **not** validate: the reference-model comparisons show
that COARE 3.5 is reproduced, not that COARE 3.5 is right; the only
comparisons with measurements are the published bin-averaged transfer
coefficients above. No buoy time series was used. The Liu–Katsaros–
Businger (1979) lookup tables are not implemented (their coefficients
could not be checked against the primary source); COARE 3.0's eq. 28
replaces them.

## References

  * Charnock, H. (1955), *Q. J. R. Meteorol. Soc.* 81, 639–640.
  * Smith, S. D. (1988), *J. Geophys. Res.* 93, 15467–15472.
  * Liu, W. T., Katsaros, K. B. & Businger, J. A. (1979), *J. Atmos. Sci.* 36, 1722–1735.
  * Fairall, C. W. et al. (1996), *J. Geophys. Res.* 101, 3747–3764.
  * Grachev, A. A. & Fairall, C. W. (1997), *J. Appl. Meteorol.* 36, 406–414.
  * Taylor, P. K. & Yelland, M. J. (2001), *J. Phys. Oceanogr.* 31, 572–590.
  * Oost, W. A. et al. (2002), *Boundary-Layer Meteorol.* 103, 409–438.
  * Fairall, C. W. et al. (2003), *J. Climate* 16, 571–591.
  * Edson, J. B. et al. (2013), *J. Phys. Oceanogr.* 43, 1589–1610; corrigendum (2014) 44, 2590.
  * Beljaars, A. C. M. (1995), *Q. J. R. Meteorol. Soc.* 121, 255–270.
  * Donelan, M. A. (1990), Air–sea interaction, in *The Sea*, vol. 9.
