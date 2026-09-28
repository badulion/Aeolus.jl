# Sub-grid surface tiles

A grid cell of tens to hundreds of metres is rarely one surface. An
urban cell is roof, street, park and water; a rural one is field, wood
and pond. `src/SurfaceTiles.jl` represents such a cell as a **mosaic of
tiles** (Avissar & Pielke 1989; Koster & Suarez 1992): each tile is an
ordinary Era V surface model — any
[`AbstractSurface`](@ref Aeolus.SurfaceExchange.AbstractSurface) — with
an area fraction ``f_i``, ``\sum_i f_i = 1``.

## 1. Flux aggregation

Every tile keeps its own prognostic state and solves its own energy
balance

```math
R_{n,i} = H_i + LE_i + G_i
```

against the grid-cell forcing, through the `SurfaceExchange` contract
(its own skin temperature, humidity, roughness and Monin–Obukhov
stability). The atmosphere sees the area-weighted flux

```math
\overline{F} = \sum_i f_i\, F_i, \qquad F \in \{R_n, H, LE, G, \tau, E\}.
```

Because every component is weighted identically, the tile balances sum
to the cell balance ``\overline{R_n} = \overline{H} + \overline{LE} +
\overline{G}`` exactly, and if tile ``i`` stores ``G_i\,Δt`` per unit
area the cell stores ``\overline{G}\,Δt``. Nothing is conserved "on
average": the fraction-weighted sum is the conservation statement. The
test suite measures it at a relative error of ``1.7\times10^{-13}`` over
a 24 h diurnal cycle of three slab tiles.

The alternative, *parameter aggregation*, averages the surface
parameters (``\bar T_s``, ``\bar α``, ``\bar z_0``, ``\bar β``) and
computes one flux ``F(\bar p)``. The two agree only where ``F`` is linear
in the parameters that differ between tiles (Mahrt 1987); the
difference ``\sum f_i F(p_i) - F(\bar p)`` is the aggregation error that
tiling removes. `parameter_aggregate` builds the parameter-averaged
surface so the two can be compared directly.

## 2. Exact radiative aggregates

For a uniform downwelling field the reflected shortwave and the absorbed
longwave are linear in the albedo and emissivity, so area weighting is
exact:

```math
\bar α = \sum_i f_i α_i, \qquad \bar ε = \sum_i f_i ε_i .
```

Emission is not linear in temperature. The **effective radiative
temperature** is defined by the emitted longwave,

```math
\bar ε\, σ\, T_{rad}^4 = \sum_i f_i\, ε_i\, σ\, T_i^4 ,
```

so a radiation scheme that sees a single surface ``(\bar α, \bar ε,
T_{rad})`` gets ``R_n = \sum f_i R_{n,i}`` identically. Because ``T^4`` is
convex, ``T_{rad} \ge \sum f_i T_i``: two black tiles at 280 K and 320 K
radiate like one at 302.0 K, not 300 K. A mosaic reports ``T_{rad}`` as
its `surface_temperature`.

These optical rules are the same area weighting as the Era III optical
mosaic `SurfaceOptics.TiledSurface`; the difference is that a
`SurfaceMosaic` asks each tile for its (broadband) albedo rather than
holding spectral optics itself.

## 3. Where the aggregation error comes from

All numbers below are measured by `test/era5_surface_tiles.jl` with the
forcing ``T_a = 295`` K, ``U = 3`` m s⁻¹ at 10 m, two tiles of equal
area. They are model-internal comparisons, not observations.

| heterogeneity | flux aggregation | parameter aggregation | why |
|:--|--:|--:|:--|
| albedo only (same ``T_s``, ``z_0``) | equal to ``10^{-13}`` | | ``F`` linear in ``α`` |
| ``T_s`` = 285 / 310 K, **neutral** ψ ≡ 0 | ``H`` = 89.10 | 89.10 W m⁻² | ``H`` linear in ``T_s`` when ``C_H`` is fixed |
| ``T_s`` = 296 / 306 K (both unstable) | ``H`` = 186.7 | 166.3 W m⁻² | ``C_H`` grows with instability: ``H(T_s)`` convex |
| ``T_s`` = 285 / 299 K (one stable, one unstable) | ``H`` = +43.6 | −24.1 W m⁻² | mean skin below the air: wrong **sign** |
| ``z_0`` = 1 mm / 1 m, neutral | ``τ`` = 0.172 | 0.052 N m⁻² | ``1/\ln^2(z/z_0)`` convex; rough tile dominates |
| wet (β=1) / dry (β=0) slabs, 24 h mean | ``H`` 46, ``LE`` 89, ``R_n`` 123 | ``H`` 1, ``LE`` 146, ``R_n`` 132 W m⁻² | dry tile runs hot |

The neutral roughness case has a closed form,
``τ_i = ρ\,[κU/\ln(z/z_{0,i})]^2``, which the mosaic reproduces to
``10^{-12}``. The stress it produces corresponds to an effective
roughness of 0.42 m, more than ten times the logarithmic mean 0.032 m —
the result Mason (1988) formalised with the blending height (see
[Blending height](blending_height.md)). In the stable/unstable case a cell whose *average* skin is colder than
the air still sends heat upward, because the warm tile's unstable
exchange is far more efficient than the cool tile's stable one. Mahrt
(1987) showed that when stability varies within a grid area the
grid-averaged exchange coefficient is a complicated function of the
grid-averaged stability; a sign reversal of ``H`` is the extreme case of
that nonlinearity. The wet/dry case is the land-surface situation
Avissar & Pielke (1989) introduced the mosaic for: the dry half has no
evaporative cooling, runs hot, loses more longwave (lower ``R_n``) and
converts the rest into sensible heat, so the cell's Bowen ratio is 0.52
against 0.007 for the uniform half-wet surface.

## 4. Class maps

The urban example stores a surface class per LES cell (street, roof,
park, water, wall). That is the one-tile-per-cell limit of a mosaic.
`class_fractions` and `tiles_from_class_map` coarse-grain such a map:
each coarse cell gets one tile per class that occurs, with its area
fraction. Because cells of one class under one forcing evolve
identically, the coarse mosaic reproduces the block mean of the
fine-resolution per-cell computation — measured at ``5\times10^{-16}``
relative over six hours for an 8×8 five-class map in 4×4 blocks.

## 5. What is not here

A plain `SurfaceMosaic` forces all tiles with the grid-cell forcing (or
a forcing per tile supplied by the caller). The blending-height
formulation, in which tiles are forced at the height where their
internal boundary layers merge, is `BlendedMosaic`; see
[Blending height](blending_height.md). Changing tile fractions (`set_mosaic_fractions!`) leaves tile
states untouched; moving heat and water between tiles when a fraction
shrinks is left to the caller.

## References

- Avissar, R. & Pielke, R. A. (1989). A parameterization of heterogeneous
  land surfaces for atmospheric numerical models and its impact on
  regional meteorology. *Mon. Wea. Rev.*, 117, 2113–2136.
- Koster, R. D. & Suarez, M. J. (1992). Modeling the land surface
  boundary in climate models as a composite of independent vegetation
  stands. *J. Geophys. Res.*, 97(D3), 2697–2715.
- Mahrt, L. (1987). Grid-averaged surface fluxes. *Mon. Wea. Rev.*, 115,
  1550–1560.
- Mason, P. J. (1988). The formation of areally-averaged roughness
  lengths. *Q. J. R. Meteorol. Soc.*, 114, 399–420.
- Giorgi, F. & Avissar, R. (1997). Representation of heterogeneity
  effects in Earth system modeling: experience from land surface
  modeling. *Rev. Geophys.*, 35, 413–437.
