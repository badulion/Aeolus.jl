# Heterogeneous-surface validation

The tile mosaic ([Sub-grid surface tiles](surface_tiles.md)) and the
blending height ([Blending height](blending_height.md)) claim that a grid
cell holding several surfaces gets its fluxes right if each surface
exchanges with the cell's air on its own and the fluxes are
area-averaged. This page tests that claim against the only reference
that has every flux at every point: an LES in which the heterogeneity is
resolved. It then compares the resolved flow with published LES of the
same kind of surface.

## Set-up

A dry convective boundary layer over strips of hot (310 K) and cool
(300 K) ground, 50 % each, normal to ``x``. Both surfaces have
``z_0 = 0.1`` m and ``z_{0h} = z_0/10``. The initial mixed layer is at
300 K up to 600 m with 6 K km⁻¹ above. This is close to the
heterogeneous cases of Margairaz et al. (2020), as described by Wanner,
Calaf & Mauder (2022). The roughness lengths are the same, and so is
``f = 10^{-4}`` s⁻¹ in the windy runs. They used random patches whose
temperatures have a ±5 K standard deviation about the mean; here there
are two strip temperatures at ±5 K.

The skin temperatures are **prescribed**, and the fluxes are not.
Avissar & Schmidt (1998) prescribed the heat flux. With a prescribed
flux, tiled aggregation of the heat flux would be exact by
construction. Here each column's flux depends on its own first-level
air through Monin–Obukhov similarity, as in a coupled land surface
(Patton et al. 2005).

The harness is `test/support/patchy_surface.jl` (the example driver
`examples/validation/patchy_surface/driver.jl` until ONE-g, #456). It is a
Boussinesq LES with Smagorinsky SGS and SSP-RK3 with one projection per
stage. Every surface column is a `SurfaceExchange.AbstractSurface`, and
its fluxes come from the contract's `turbulent_fluxes`. The heat flux
goes into the first level, and the stress acts along the local wind.
Three surface modes use that one code path:

| mode | what a column holds | meaning |
|:--|:--|:--|
| `:resolved` | the hot or the cool surface | heterogeneity resolved |
| `:tiled` | a `SurfaceMosaic` of both (0.5/0.5) | heterogeneity sub-grid, tiled |
| `:parameter` | `parameter_aggregate(mosaic)`: one 305 K surface | parameter aggregation |

The coupling conserves heat. The column heat content changes by the
applied flux to ``10^{-10}`` relative, and the flux is accumulated with
the RK3 stage weights 1/6, 1/6, 2/3. Over a uniform surface the three
modes agree to ``8×10^{-13}`` K after 200 s.

## A-priori: tiles forced by the LES's own air

The resolved domain-mean flux ``\overline{w'θ'_0}`` is compared with the
mosaic's flux under several choices of forcing air. 50 m grid,
3.2 × 1.6 × 1.6 km, mean over the second hour
(the former `examples/validation/patchy_surface_les.jl`, (removed by ONE-g, #456)):

| case | LES (K m s⁻¹) | tiles, patch air | tiles, mean air & speed | `BlendedMosaic` | parameter |
|:--|--:|--:|--:|--:|--:|
| strips λ = 0.8 km, calm (λ/z_i = 0.86) | 0.0692 | −2.5 % | −1.0 % | +1.2 % | −36.8 % |
| strips λ = 3.2 km, calm (λ/z_i = 3.2) | 0.0784 | −0.4 % | −5.8 % | −12.4 % | −35.0 % |
| strips λ = 3.2 km, U_g = 5 m s⁻¹ | 0.1094 | −0.1 % | +2.2 % | −1.1 % | −22.4 % |
| uniform 305 K, calm (reference) | 0.0485 | −2.5 % | −2.5 % | −6.6 % | −2.5 % |

- **Patch air.** Each tile is forced by the mean air over its own
  patches. The tiles then reproduce the resolved flux to 0.1–2.5 %,
  which is the uniform-surface level (−2.5 %). The mosaic therefore adds
  no error of its own. What remains is the one-point Monin–Obukhov
  closure applied to averaged air.
- **Domain-mean air.** This is what one grid cell of this size would
  see. The error grows with λ/z_i and reaches −5.8 % at λ/z_i = 3.2. A
  patch-locked circulation makes the air over the hot strips warmer
  than the domain mean, and the mean air does not know this.
- **Parameter aggregation** misses 22–37 %. The flux is strongly convex
  in the skin temperature: the cool strips sit below the mixed-layer
  temperature and are stably stratified, with almost no flux. A single
  surface at the mean temperature averages that nonlinearity away.
- **`BlendedMosaic`**, with ``l_b`` at the first level and forced by the
  mean air at 125 m, is within 1.2 % with wind or at λ/z_i < 1. At
  λ/z_i = 3.2 in calm conditions it is −12 %, because one
  Monin–Obukhov layer does not describe the mean profile under a
  patch-locked circulation.

The wind used for the domain-mean forcing matters more than the tiles
do. In free convection the mean wind *vector* is ≈ 0.03 m s⁻¹. Tiles
forced by it sit on the solver's ``U_{min}`` floor and are +16 to +34 %
off (uniform surface included). The Beljaars (1995) gustiness
``1.2 w_*`` gives +8 to +21 %. Forcing with the mean *speed* is −1 to
−6 %. This is a statement about the grid-cell wind that couples to a
surface scheme, not about tiling.

## A-posteriori: an LES with the tiled surface

This test replaces the surface itself, not only the diagnosis. Every
column holds the mosaic (`:tiled`), or every column holds the 305 K
surface (`:parameter`, which is the uniform run).

| | tiled vs strips 0.8 km | tiled vs strips 3.2 km | uniform 305 K vs strips 0.8 / 3.2 km |
|:--|--:|--:|--:|
| domain-mean flux, calm | +4.2 % | −8.0 % | −29.9 % / −38.1 % |
| domain-mean flux, U_g = 5 m s⁻¹ | — | +1.4 % | — |
| mixed-layer θ, max difference | 0.10 K | 0.22 K | 0.29 K (0.8 km) |
| z_i | 909 vs 935 m | 909 vs 1001 m | 868 m |

The tiled surface carries the heterogeneous flux, and the
parameter-aggregated surface does not. The tiled run falls short of the
3.2 km strips for the same reason as the a-priori test: at λ/z_i ≳ 3 the
resolved circulation moves warm air over the hot strips, and
horizontally uniform tiles cannot represent that.

## Comparison with published strip LES

Patton, Sullivan & Moeng (2005) ran LES coupled to a land-surface model
over wet/dry strips with 2–30 km wavelength (1 < λ/z_i < 18). The
conference abstract of that work (NASA NTRS 20030068489) states that
4 < λ/z_i < 9 "induces the strongest organized flow fields (u_p, w_p)
while heterogeneity with smaller or larger λ/z_i induces little
organized motion". The paper's abstract adds that "surface fluxes in the
middle of the patches are small compared to the patch edges".

Here the sweep was run at 100 m resolution, 12.8 × 1.6 × 1.6 km, mean
over 1.5–3 h. ``A_u``, ``A_w`` are the rms of the phase-averaged
first-level ``u`` and of ``w`` at 350 m, over ``w_*``:

| λ (km) | λ/z_i | A_u/w_* | A_w/w_* | hot-strip flux mid/edge | tiles, patch air |
|--:|--:|--:|--:|--:|--:|
| 0.8 | 0.76 | 0.28 | 0.41 | 0.98 | −2.2 % |
| 1.6 | 1.46 | 0.45 | 0.42 | 0.94 | −1.3 % |
| 3.2 | 2.79 | 0.66 | 0.39 | 0.88 | −0.6 % |
| 6.4 | 5.46 | 0.97 | 0.38 | 0.84 | −0.1 % |
| 12.8 | 10.6 | 1.37 | 0.33 | 0.74 | +0.1 % |
| uniform | — | 0.02 | 0.02 | — | — |

- **Reproduced.** The near-surface circulation converges on the hot
  strip from both sides, with the updraft above it. It strengthens from
  λ/z_i ≈ 0.8 to 5.5, by 3.5×. The flux is smallest mid-patch, where
  the converging air is warmest and the wind weakest. The mid-patch dip
  deepens with λ, from 2 % to 26 %.
- **Not reproduced.** The *decline* above λ/z_i ≈ 9. Here ``A_u`` keeps
  growing to λ/z_i = 10.6, and ``A_w`` at 350 m is nearly flat (0.33–0.42)
  rather than peaked. Possible reasons, none tested: the prescribed skin
  temperature (Patton et al. coupled soil moisture, which weakens the
  forcing where the circulation dries or cools the surface); runs of 3 h
  that are short against the λ/U adjustment time of the largest strips;
  a 1.6 km lateral domain and 100 m cells.
- **Background wind.** Wanner, Calaf & Mauder (2022, Sect. 3 and Fig. 5)
  describe the LES of Margairaz et al. (2020): the large cellular
  circulations at U_g = 1 m s⁻¹ "disappear with increasing wind speed".
  Here a 5 m s⁻¹ geostrophic wind across the strips reduces ``A_u/w_*``
  from 0.76 to 0.024, a factor 30. That is below the phase-average
  noise of the calm uniform surface (0.045) but above that of the windy
  uniform surface (0.009), so a weak patch-locked signal remains.

`validation/era5_heterogeneous_surface.jl` repeats the comparison on a 100 m
grid (32 × 16 × 16, 1.25 h), with three seeds checked. It asserts:

- conservation (the identity of the three code paths is in
  `test/era5_heterogeneous_surface.jl`);
- tiles with patch air within 3 %, and with mean air within 10 %;
- parameter aggregation below 80 % of the resolved flux;
- the tiled LES within 8 % of the λ = 0.8 km strips;
- the windy mosaic and `BlendedMosaic` within 3 %;
- ``A_u`` at λ/z_i ≈ 3 more than 2× that at ≈ 0.9, which is itself more
  than 3× the uniform-surface noise;
- convergence on the hot strip, mean ascent over it and descent over
  the cool strip, and a mid-patch flux below 0.95 of the
  edge flux;
- the windy ``A_u`` below 0.1 of the calm one.

## What is not validated

- **No field observations.** The patchy-terrain campaigns (LITFASS-98,
  LITFASS-2003) publish area-averaged fluxes for a real landscape of
  forest, lake, farmland and villages. Reproducing them needs the
  vegetated land surface of Era V (#21), which does not exist yet, and
  the campaign's land-use and forcing data. The comparison above is
  with published LES, and only qualitatively: the numbers of Patton et
  al. were not available to compare against.
- Moisture: the surfaces are dry, so the scalar dissimilarity that
  Patton et al. report for wet/dry strips is not tested.
- Roughness heterogeneity in a resolved LES. The effective roughness is
  compared with the published LES of Bou-Zeid et al. (2007) in
  [Blending height](blending_height.md), not with an Aeolus LES.

## References

- Avissar, R. & Schmidt, T. (1998). An evaluation of the scale at which
  ground-surface heat flux patchiness affects the convective boundary
  layer using large-eddy simulations. *J. Atmos. Sci.*, 55, 2666–2689.
- Beljaars, A. C. M. (1995). The parametrization of surface fluxes in
  large-scale models under free convection. *Q. J. R. Meteorol. Soc.*,
  121, 255–270.
- Margairaz, F., Pardyjak, E. R. & Calaf, M. (2020). Surface thermal
  heterogeneities and the atmospheric boundary layer: the thermal
  heterogeneity parameter. *Boundary-Layer Meteorol.*, 177, 49–68.
- Patton, E. G., Sullivan, P. P. & Moeng, C.-H. (2005). The influence of
  idealized heterogeneity on wet and dry planetary boundary layers
  coupled to the land surface. *J. Atmos. Sci.*, 62, 2078–2097.
  Conference abstract: NASA NTRS 20030068489.
- Wanner, L., Calaf, M. & Mauder, M. (2022). Incorporating the effect of
  heterogeneous surface heating into a semi-empirical model of the
  surface energy balance closure. *PLOS ONE*, 17, e0268097.
