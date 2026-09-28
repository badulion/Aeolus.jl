# Facet temperatures coupled to radiosity

The Era III radiosity solver (`src/Radiosity.jl`) resolves every roof,
wall strip and street patch of a city block. The Era I urban example
still averaged its solution back onto one surface temperature per grid
column. A column with a street, the lower half of one facade and the
upper half of another therefore had one temperature for all of them.
`UrbanFacetScene` (`src/UrbanSurface.jl`) removes that approximation.
Facet ``i`` of the radiosity scene is also facet ``i`` of the surface
energy balance. It has its own conduction column
([Urban facet conduction](urban_conduction.md)) and its own skin
temperature, and it exchanges longwave with every other facet directly.

## 1. Longwave exchange is linear in the emission

For grey Lambertian facets with view factors ``F_{ij}``, sky-view factors
``F_{sky,i}`` and sky irradiance ``L_{sky}``, the incident longwave
``Q`` and the radiosity ``B = E + (1-\varepsilon)Q`` satisfy
``Q = FB + F_{sky}L_{sky}``. Eliminating ``B``,

```math
Q = \bigl(I - F\,\mathrm{diag}(1-\varepsilon)\bigr)^{-1}\bigl(F E + F_{sky} L_{sky}\bigr),
```

so the net longwave a facet absorbs, ``N = \mathrm{diag}(\varepsilon)Q - E``,
is **exactly linear** in the emission ``E_j = \varepsilon_j\sigma T_j^4``:

```math
N_i = \sum_j J_{ij} E_j + j^{sky}_i L_{sky},\qquad
J = \mathrm{diag}(\varepsilon)\bigl(I - F\,\mathrm{diag}(1-\varepsilon)\bigr)^{-1}F - I .
```

`Radiosity.longwave_response` returns ``J`` and ``j^{sky}``. They depend
on geometry and emissivity only, so they are computed once and reused
until the emissivities change. ``J`` inherits two properties from the
view factors:

* **Reciprocity of total exchange areas** (Hottel & Sarofim 1967):
  ``A_i J_{ij}\varepsilon_j = A_j J_{ji}\varepsilon_i`` for ``i \ne j``.
* **Closure**: ``\sum_i A_i J_{ij} = -A_j\,f^{esc}_j``, where
  ``f^{esc}_j`` is the fraction of ``j``'s emission that escapes to the
  sky. In an enclosure it is zero: extra emission by one facet is
  absorbed, in full, by the others.

## 2. The implicit coupled step

The only nonlinearity in the exchange is ``T^4``. Linearising it about
the current skin temperatures, ``E_j \approx E_j^n + 4\varepsilon_j\sigma
(T_j^n)^3 x_j``, the energy balance of facet ``i`` becomes

```math
R_{n,i} - H_i - LE_i \approx F_{0,i} + F'_{ii}\,x_i + \sum_{j\ne i} G_{ij}\,x_j,
\qquad G_{ij} = J_{ij}\,4\varepsilon_j\sigma (T_j^n)^3 .
```

Here ``x`` are the θ-weighted skin increments of the step. ``F'_{ii}``
combines the radiative ``G_{ii}`` with the finite-difference turbulent
derivatives, exactly as for an isolated facet. The diagonal part is the
Robin condition that closes each facet column. The off-diagonal part
couples all the columns. Each column step is affine in an extra flux
``c_i`` imposed on its outer face, ``x_i = a_i + b_i c_i``, so two
trial steps per facet give ``a`` and ``b``. The coupled increments then
solve one dense system,

```math
\bigl(I - \mathrm{diag}(b)\,G_{\text{off}}\bigr)\,x = a ,
```

and the columns are advanced with ``c_i = \sum_{j\ne i}G_{ij}x_j``. The
fluxes returned are those applied: per facet, ``R_n - H - LE = G``, and
the stored heat changes by ``(G - Q_\text{interior})\Delta t`` to
round-off. The longwave actually applied, `lw_net`, equals the radiosity
solution for the applied emission `lw_emission`. Shortwave does not
depend on temperature and comes from `Radiosity.solve_shortwave`.

Albedo and emissivity have one source: the scene's `UrbanMaterials`.
They are copied into the facets at construction and again at every
step.

## 3. Why a lagged coupling is not enough

The simple alternative steps each facet on its own against the incident
longwave frozen at the start of the step (`coupling = :lagged`, which is
kept only as a reference). A facet that warms during the step emits more,
and that extra emission leaves it. But the other facets never receive it,
because their incident longwave was frozen. The exchange then creates or
destroys energy in proportion to ``\Delta t`` and the temperature change.
Two adiabatic plates facing each other, at 320 K and 270 K, 5 cm of
concrete each, stepped at 30 min:

| coupling | max ``\lvert\sum_i A_i N_i\rvert`` per step | stored-heat drift over 100 h |
|:---------|:-------------------------------------------|:-----------------------------|
| implicit | 7·10⁻¹⁵ W m⁻² | −6·10⁻⁵ J m⁻² (1·10⁻¹² relative) |
| lagged   | 14.7 W m⁻² | 1.2·10⁵ J m⁻² |

## 4. Validation

All numbers are from `test/era5_urban_radiosity.jl`.

| check | reference | measured |
|:------|:----------|:---------|
| `longwave_response` | `solve_longwave`, any temperatures | 2.7·10⁻¹³ W m⁻²; Hottel reciprocity 6·10⁻¹⁶; ``J = F - I`` for a black scene to 1·10⁻¹⁴ |
| Infinite grey parallel plates (ε 0.9 / 0.6), interiors held at 330 / 280 K through 5 cm concrete | ``q = \sigma(T_1^4-T_2^4)/(1/\varepsilon_1+1/\varepsilon_2-1) = (T_a-T_1)/R`` | skins 324.846666 / 285.153334 K, < 10⁻⁶ K; ``q = 144.2934`` W m⁻², 10⁻⁸ relative |
| Adiabatic plates | energy conservation; equilibrium at the heat-weighted mean | see §3; both plates reach 295.000000 K |
| Infinite canyon, adiabatic walls, black floor (crossed-string view factors) | ``\sigma T_w^4 = (\sigma T_g^4 + L)/2`` for any ``H/W`` and wall ε; floor loss ``\tfrac{1+\psi}{2}(\sigma T_g^4 - L)`` | ``H/W`` 0.5 / 1 / 2 with ε_w 0.9 / 0.7 / 0.95: ``T_w`` 279.667618 / 280.100011 / 280.501963 K, all < 10⁻⁶ K |
| Isolated flat facet | `step_surface!(::UrbanFacet)` | 5·10⁻⁵ K over 12 h (analytic vs finite-difference ``dR_n/dT``) |
| 3-D ``H/W = 2`` canyon, 336 facets | radiosity re-solved at the applied emission | 8·10⁻¹³ W m⁻²; per-facet SEB 4·10⁻¹⁵ W m⁻²; storage 2·10⁻¹⁰ relative; shortwave and plan-area aggregation to 10⁻¹² |
| Same, 6 h night, vs ``\Delta t = 30`` s | time-step convergence | implicit 0.0043 K at 900 s, 0.0004 K at 450 s; lagged 0.0128 K at 900 s |

The last two rows are the capability this adds. After a 12-hour night
in the 3-D canyon, the lowest wall strips (sky-view factor 0.083) are
at 291.24 K and the top strips (0.402) are at 289.61 K. A single column
temperature cannot hold that difference. A step of the 336-facet scene
costs about 13 ms; one `solve_longwave` alone costs 1.6 ms.

**Steady-state tests use θ = 1.** A steady state does not depend on θ.
But at ``\Delta t \gg \Delta z^2/\kappa``, Crank–Nicolson's amplification
factor for the stiffest cell modes tends to −1, so those modes ring for
hundreds of steps. With θ = ½ the plate test was still 1·10⁻⁶ K off after
400 steps.

**Validation against observations** is on
[BUBBLE / CAPITOUL validation](urban_validation.md). The air state near
each facet is an input here. The canyon air temperature is not
modelled. The canyon wind and the wall convection coefficient can come
from the TEB closures `canyon_wind_speed` and
`rowley_convection_coefficient` (#30).

**Buildings behind the facets (#29).** The implicit step is built from
three pieces: trial column steps, one dense solve, and the final column
steps. A facet can also be bound to an *interior node*, the air of the
building zone behind it. A third trial step then measures the column's
response to that node's temperature, and the zones' air temperatures
join the dense solve. `BuildingEnergy.step_buildings!` uses this; see
[Anthropogenic heat from the building energy model](anthropogenic_heat.md).
Without nodes the system is the one above, and every number in §4 is
unchanged.

## References

- Hottel, H. C. & Sarofim, A. F. (1967). *Radiative Transfer*.
  McGraw-Hill, ch. 3 (total exchange areas, crossed-string method).
- Modest, M. F. (2013). *Radiative Heat Transfer*, 3rd ed. Academic
  Press, §5 (radiative exchange between grey diffuse surfaces; parallel
  plates).
- Harman, I. N., Best, M. J. & Belcher, S. E. (2004). Radiative exchange
  in an urban street canyon. *Boundary-Layer Meteorol.*, 110, 301-316.
- Resler, J. et al. (2017). PALM-USM v1.0. *Geosci. Model Dev.*, 10,
  3635-3659.
