# Blending height

A tiled grid cell ([Sub-grid surface tiles](surface_tiles.md)) still has
to decide which air each tile exchanges with. Over one patch, the flow
near the ground adjusts to that patch inside an **internal boundary
layer** (IBL) that deepens with fetch. Higher up, the IBLs of
neighbouring patches merge. Above the **blending height** ``l_b`` the
flow is horizontally uniform and has one Monin–Obukhov profile, which
carries the area-averaged flux (Wieringa 1986; Mason 1988; Claussen
1991). The blending-height formulation in `src/SurfaceTiles.jl` has
three parts:

1. an effective roughness at ``l_b``;
2. closures for ``l_b`` and IBL growth laws;
3. `BlendedMosaic`, whose tiles are forced with the state at ``l_b``.

## 1. Effective roughness

Below ``l_b`` each tile has its own neutral log profile. At ``l_b`` the
wind ``U_b`` is common to all tiles. Tile ``i`` therefore feels the
stress ``ρ[κU_b/\ln(l_b/z_{0,i})]^2``. The cell stress equals that of
one surface of roughness ``z_{0,eff}`` if

```math
\frac{1}{\ln^2(l_b/z_{0,eff})} = \sum_i \frac{f_i}{\ln^2(l_b/z_{0,i})} .
```

Because ``1/\ln^2`` is convex, ``z_{0,eff}`` lies above the logarithmic
mean ``\exp\overline{\ln z_0}`` and below ``\max z_{0,i}``. It rises as
``l_b`` falls, so smaller patches make a rougher cell. As
``l_b → ∞`` it tends to the log average of Taylor (1987), and does so
only logarithmically:

```math
\ln z_{0,eff} = \overline{\ln z_0} + \tfrac32\,\frac{\mathrm{Var}(\ln z_0)}{\ln(l_b/z_0)} + O(\ln^{-2}),
```

which follows from expanding ``(a - x_i)^{-2}`` for ``a = \ln l_b`` large.
The tests measure the ratio of the actual gap to this leading term at
0.917, 0.949, 0.971 and 0.984 for ``l_b = 10^2, 10^4, 10^8, 10^{16}`` m
(``z_0`` = 0.01 / 0.1 m).

For heat and moisture the same construction is applied to the neutral
transfer coefficient ``κ^2/[\ln(l_b/z_{0m})\ln(l_b/z_{0h})]``. This gives
`effective_scalar_roughness`, which is exact when every tile has the
same surface-to-``l_b`` scalar difference.

## 2. Where is ``l_b``?

| closure | equation for ``l_b`` (solved jointly with ``z_{0,eff}(l_b)``) | source |
|:--|:--|:--|
| `MasonBlendingHeight(L)` | ``(l_b/L)\ln^2(l_b/z_{0,eff}) = 2κ^2`` | Mason 1988, as in Kramm & Dlugi 2009 Eq. 5.4 |
| `ClaussenBlendingHeight(L)` | ``(l_b/L)\ln(l_b/z_{0,eff}) = 1.75κ`` | Claussen 1991: form as described by Bou-Zeid et al. 2007 Sect. 6, constant from Kramm & Dlugi 2009 Eq. 5.5 (see note) |
| `BouZeidBlendingHeight(L)` | ``l_b[\ln(l_b/z_{0,eff}) - 1] = 1.7κL`` | Bou-Zeid et al. 2004; 2007 Eqs. 2–3 |
| `IBLBlendingHeight(model, L)` | deepest pairwise IBL at fetch ``2L`` | premise of Bou-Zeid et al. 2004 |
| `FixedBlendingHeight(l_b)` | prescribed, e.g. diagnosed from an LES | — |

Mason's closure asks where ``(u_*/U(l_b))^2`` equals the aspect ratio
``l_b/(2L)``. It gives ``l_b ≈ L/100``. Bou-Zeid's closure follows
diffusion-analogy IBLs over about twice the patch length and gives
heights ten times larger. Claussen's closure is Bou-Zeid's without the
``-1``.

*Note on the Claussen closure.* Claussen (1991) was not consulted
directly. Bou-Zeid et al. (2007, Sect. 6) state that Claussen's
``z_{0,e}`` equation differs from theirs only by the ``-1``, which fixes
the linear-in-``\ln`` form used here. Kramm & Dlugi (2009, Eq. 5.5),
the source of ``C_b ≅ 1.75``, print the logarithm squared instead,
``(l_b/L)\ln^2(l_b/z_0) ≅ C_b κ``; read literally, that gives a
Mason-like ``l_b`` (50 m instead of 265 m for case S25R75 below). The
constant should be treated as uncertain.

The IBL growth laws are the neutral short-fetch formulae compiled by
Savelyev & Taylor (2005, Table I):

- `ElliottIBL`: Elliott (1958), ``δ/z_{0D} = (0.75 - 0.03\ln(z_{0D}/z_{0U}))(x/z_{0D})^{0.8}``.
- `PowerLawIBL`: Wood (1982), ``δ/z_{0r} = 0.28(x/z_{0r})^{0.8}`` with ``z_{0r} = \max(z_{0U}, z_{0D})``.
- `DiffusionIBL`: the Panofsky & Dutton (1984) diffusion analogy,
  ``1.25κx/z_{0D} = (δ/z_{0D})(\ln(δ/z_{0D}) - 1) + 1``. It is solved by
  Newton's method in ``\ln δ`` and has the growth rate
  ``dδ/dx = 1.25κ/\ln(δ/z_{0D})``.

## 3. Tiles forced at the blending height

`blended_forcing(mosaic, forcing, l_b)` finds the wind ``U_b``, potential
temperature ``θ_b`` and humidity ``q_b`` at ``l_b`` that satisfy three
conditions together:

- each tile exchanges with ``(U_b, θ_b, q_b)`` through its own surface
  layer of depth ``l_b``, using the tile's own `turbulent_fluxes`, so its
  own ``z_0``, skin state and stability;
- the aggregate ``\sum f_i F_i`` defines ``u_*``, ``θ_*`` and ``q_*`` and
  an Obukhov length;
- one Monin–Obukhov layer between ``l_b`` and the forcing height ``z``
  carries that flux:
  ``U(z) - U_b = (u_*/κ)[\ln(z/l_b) - ψ_m(z/L) + ψ_m(l_b/L)]``, and
  likewise for ``θ`` and ``q`` with ``ψ_h``.

The three residuals are solved by a damped Newton iteration with a
finite-difference Jacobian. `BlendedMosaic` wraps a `SurfaceMosaic`
with a closure for ``l_b``. It is an `AbstractSurface`: each
`step_surface!` first solves for the blended state and then steps every
tile with it. Its `roughness_lengths` are the blending-height effective
lengths.

**Exact limits (tested).**

- With ``ψ ≡ 0`` and a common surface-to-``l_b`` difference, the blended
  cell reproduces one surface of ``z_{0m,eff}``, ``z_{0h,eff}``,
  ``z_{0q,eff}`` forced at ``z``. Measured: ``5.4\times10^{-11}``.
- A single tile split at ``l_b`` gives the same kinematic fluxes as the
  tile forced at ``z``. Measured: ``3.8\times10^{-11}`` with ``ψ ≡ 0``.
- For ``l_b ≥ z`` there is no blending, and `blended_forcing` returns
  the plain forcing.
- As ``l_b → z^-`` the result tends continuously to the plain mosaic.

**Where the split is not exact.** The `SurfaceExchange` solve refers
buoyancy to ``θ_v`` at its forcing level, which is ``l_b`` for a tile
and ``z`` for the direct solve. That ``O(Δθ/θ)`` ambiguity in ``L``
leaves a single-tile split error, and stability amplifies it:

| case | single-tile split error |
|:--|:--|
| unstable and near-neutral (BDBH) | ``5.5\times10^{-4}`` |
| strongly stable (skin 285 K, air 293 K) | ``2.5\times10^{-2}`` |

Fixing ``θ_{v,ref}`` to a constant in both solves, as a temporary
experiment, reduced these errors to ``3\times10^{-11}`` and
``10^{-13}``. The ambiguity is therefore the whole of the difference.
Separately, the default β hook forms ``q_s`` from ``q_a`` at the
forcing level, so a tile with ``0 < β < 1`` evaporates differently when
it is forced at ``l_b``. A tile with a resistance formulation does not
have this problem.

**Conservation.** A `BlendedMosaic` inherits the conservation of the
mosaic. Over a 24 h cycle of two slab tiles with 288 steps, all 288
blended solves converged. ``\sum f_i C_i ΔT_i`` matched
``\sum \overline{G}\,Δt`` to ``4.1\times10^{-14}``.

## 4. Validation

**Bou-Zeid, Parlange & Meneveau (2007, Table 2).** LES of two
periodic stripes, ``z_0`` = 0.01 / 0.1 m, with patch length scale
``L_p`` = 3140 m. The LES ``z_{0,e}`` is a log-law fit to the mean
profile, and ``h_b`` is where the velocity quartiles merge.

| case (smooth %) | LES ``z_{0,e}`` | Bou-Zeid | Claussen | Mason | log mean | LES ``h_b`` | Bou-Zeid ``l_b`` |
|:--|--:|--:|--:|--:|--:|--:|--:|
| S07R93 | 0.090 | 0.0893 | 0.0893 | 0.0904 | 0.0851 | 225 | 299.9 |
| S12R88 | 0.080 | 0.0821 | 0.0821 | 0.0839 | 0.0759 | 230 | 296.8 |
| S25R75 | 0.063 | 0.0651 | 0.0652 | 0.0680 | 0.0562 | 240 | 288.7 |
| S33R67 | 0.058 | 0.0559 | 0.0560 | 0.0591 | 0.0468 | 240 | 283.5 |
| S67R33 | 0.028 | 0.0263 | 0.0264 | 0.0286 | 0.0214 | 260 | 260.4 |
| S75R25 | 0.023 | 0.0214 | 0.0214 | 0.0231 | 0.0178 | 265 | 254.6 |
| S88R12 | 0.017 | 0.0147 | 0.0147 | 0.0155 | 0.0132 | 305 | 244.9 |
| S93R07 | 0.014 | 0.0126 | 0.0126 | 0.0131 | 0.0117 | 310 | 241.0 |
| rms ``\ln`` error | | 0.074 | 0.073 | 0.052 | 0.193 | | max 33 % |

- **Effective roughness.** The blending-height models reproduce the LES
  within 13.4 %. The log average is low in every case, by up to 21 %.
  In this two-stripe set Mason's closure fits ``z_{0,e}`` best, even
  though its ``l_b`` (19–30 m) is ten times below the LES blending
  height. ``z_{0,eff}`` depends on ``l_b`` only logarithmically.
- **Blending height.** The Bou-Zeid ``l_b`` is within 33 % of the LES
  ``h_b`` but has the opposite trend: the LES ``h_b`` rises as the
  smooth fraction grows, and the model's falls.

**IBL formulae (Savelyev & Taylor 2005, Table II).** At 90 m fetch after
a sea-to-land change of roughness from ``3\times10^{-6}`` to
``3\times10^{-2}`` m, the implementations reproduce the published values
of the formulae to within 0.05 m. The measured daytime IBL height there
was 7.2 m (Echols & Wagner 1972, via Table II).

| law | Aeolus | Table II | vs observed 7.2 m |
|:--|--:|--:|--:|
| Elliott | 8.60 m | 8.6 m | +19 % |
| Wood | 5.08 m | 5.1 m | −29 % |
| Panofsky–Dutton | 9.46 m | 9.5 m | +31 % |

**IBL spread (Savelyev & Taylor 2005, Fig. 3 and text).** The two cases
use the roughness changes of the Jegede & Foken (1999) site. At 300 m
fetch, the published formulae span 8–34 m, and Elliott's is the
highest.

| roughness change | Elliott | Wood | Panofsky–Dutton |
|:--|--:|--:|--:|
| 0.008 → 0.032 m | 34.1 m | 13.5 m | 26.3 m |
| 0.053 → 0.014 m | 32.2 m | 14.9 m | 23.4 m |

**Not validated against observations.** Apart from the single IBL
height quoted in Table II, no field data are used, and none from patchy
terrain. The other comparisons are with a published LES and a published
compilation of formulae. Tiles in stable and unstable
conditions are checked only for self-consistency; there is no reference
flux for them.
A resolved-LES test of tiles and `BlendedMosaic` over thermal strips is
in [Heterogeneous-surface validation](heterogeneous_surface.md).

## References

- Bou-Zeid, E., Meneveau, C. & Parlange, M. B. (2004). Large-eddy
  simulation of neutral atmospheric boundary layer flow over
  heterogeneous surfaces: blending height and effective surface
  roughness. *Water Resour. Res.*, 40, W02505.
- Bou-Zeid, E., Parlange, M. B. & Meneveau, C. (2007). On the
  parameterization of surface roughness at regional scales. *J. Atmos.
  Sci.*, 64, 216–227.
- Claussen, M. (1991). Estimation of areally-averaged surface fluxes.
  *Boundary-Layer Meteorol.*, 54, 387–410.
- Echols, W. T. & Wagner, N. K. (1972), as cited in Savelyev & Taylor
  (2005, Table II).
- Elliott, W. P. (1958). The growth of the atmospheric internal boundary
  layer. *Trans. Amer. Geophys. Union*, 39, 1048–1054.
- Kramm, G. & Dlugi, R. (2009). On the accuracy with which the lower
  boundary conditions can be determined in numerical models of the
  atmosphere. arXiv:0911.2286.
- Mason, P. J. (1988). The formation of areally-averaged roughness
  lengths. *Q. J. R. Meteorol. Soc.*, 114, 399–420.
- Panofsky, H. A. & Dutton, J. A. (1984). *Atmospheric Turbulence*.
  Wiley.
- Savelyev, S. A. & Taylor, P. A. (2005). Internal boundary layers: I.
  Height formulae for neutral and diabatic flows. *Boundary-Layer
  Meteorol.*, 115, 1–25.
- Taylor, P. A. (1987). Comments and further analysis on effective
  roughness lengths for use in numerical 3-dimensional models.
  *Boundary-Layer Meteorol.*, 39, 403–418.
- Wieringa, J. (1986). Roughness-dependent geographical interpolation of
  surface wind speed averages. *Q. J. R. Meteorol. Soc.*, 112, 867–889.
- Wood, D. H. (1982). Internal boundary-layer growth following a change
  in surface roughness. *Boundary-Layer Meteorol.*, 22, 241–244.
