# Radiative transfer

Radiation is the only term in the LES energy budget that is genuinely
*non-local*: every other process — advection, diffusion, pressure —
couples a cell to its neighbours, but a photon emitted at the surface
can be absorbed ten kilometres above it, and a building can shade a
street two hundred metres away. This page develops the theory behind
Aeolus' radiation stack (Era III, `src/RadiationCore.jl` and the
modules built on it) from the transfer equation down to the specific
approximations each module makes, and states honestly where each
approximation breaks.

## 1. The radiative transfer equation

Let ``I_\nu(\mathbf{x}, \hat{\Omega})`` be the *spectral radiance* at
position ``\mathbf{x}`` travelling in direction ``\hat{\Omega}`` —
energy per unit time, per unit area normal to ``\hat{\Omega}``, per
unit solid angle, per unit wavenumber. Along a path parameterised by
``s``, radiance is depleted by extinction and replenished by emission
and by scattering from other directions:

```math
\hat{\Omega} \cdot \nabla I_\nu
= -\beta_{e,\nu} I_\nu
+ \beta_{a,\nu} B_\nu(T)
+ \frac{\beta_{s,\nu}}{4\pi} \int_{4\pi} p_\nu(\hat{\Omega}', \hat{\Omega})\, I_\nu(\hat{\Omega}')\, d\Omega' .
```

Here ``\beta_a``, ``\beta_s`` and ``\beta_e = \beta_a + \beta_s`` are
the absorption, scattering and extinction coefficients (units m⁻¹),
``B_\nu(T)`` is the Planck function, and ``p_\nu`` is the *phase
function*, normalised so that ``\frac{1}{4\pi}\int p \, d\Omega = 1``.

It is conventional to work in terms of three dimensionless quantities
instead, and these are exactly the three fields carried by Aeolus'
`OpticalProperties` type:

```math
\tau = \int \beta_e \, ds, \qquad
\omega = \frac{\beta_s}{\beta_e}, \qquad
g = \frac{1}{4\pi}\int_{4\pi} p(\cos\Theta)\, \cos\Theta \, d\Omega .
```

``\tau`` is the **optical depth** (how opaque the layer is), ``\omega``
the **single-scattering albedo** (what fraction of extinction is
scattering rather than absorption — ``\omega = 1`` is a purely
scattering medium such as a cloud in the visible, ``\omega = 0`` a
purely absorbing one such as a gas line in the infrared), and ``g`` the
**asymmetry parameter** (the mean cosine of the scattering angle;
``g = 0`` is isotropic or Rayleigh, ``g \approx 0.85`` is forward-peaked
scattering by cloud droplets, ``g < 0`` is backscattering).

### Combining optical properties

When several constituents — gas, cloud, aerosol — occupy the same
layer, their optical depths add, but ``\omega`` and ``g`` must be
combined as *weighted means*, because each is defined as a ratio. The
correct rule, implemented in `combine_optics!`, is

```math
\tau = \sum_i \tau_i, \qquad
\omega = \frac{\sum_i \tau_i \omega_i}{\sum_i \tau_i}, \qquad
g = \frac{\sum_i \tau_i \omega_i g_i}{\sum_i \tau_i \omega_i} .
```

``\omega`` is weighted by optical depth because it is a per-extinction
ratio; ``g`` is weighted by the *scattering* optical depth
``\tau_i \omega_i`` because only scattered photons have a scattering
angle at all. Getting this weighting wrong is a classic error that
produces subtly wrong cloud albedos — a thin absorbing aerosol mixed
with a thick scattering cloud will drag ``g`` far too low if weighted
by ``\tau`` alone.

## 2. Gas optics: the spectral problem

The transfer equation above is *monochromatic*. Real gas absorption
varies by six orders of magnitude between the centre of a rotational
line and the window a fraction of a wavenumber away. Solving the
transfer equation at line resolution — a *line-by-line* calculation —
needs ``\sim 10^6`` spectral points and is far too expensive inside an
LES that calls radiation every few minutes.

### Why band-averaging fails

The obvious shortcut — average the absorption coefficient over a
spectral band and solve once — is wrong, and wrong in a specific
direction. Transmission is exponential in ``k``, and by Jensen's
inequality

```math
\overline{e^{-ku}} \;\ge\; e^{-\bar{k}u},
```

so the true band transmission always *exceeds* the transmission of the
band-mean absorption coefficient. Physically: photons escape through
the gaps between lines. Band-averaging systematically makes the
atmosphere too opaque.

### The correlated-k method

The correlated-k method fixes this by reordering. Within a band, the
*value* of ``k`` at a particular wavenumber does not matter for the
band-integrated flux — only the *distribution* of ``k`` values does.
So define the cumulative distribution ``g(k)``: the fraction of the
band with absorption coefficient below ``k``. Inverting gives a
monotone ``k(g)`` on ``g \in [0,1]``, and

```math
\overline{T} = \int_{\Delta\nu} e^{-k_\nu u}\, \frac{d\nu}{\Delta\nu}
= \int_0^1 e^{-k(g) u} \, dg
\approx \sum_{j} w_j \, e^{-k_j u}.
```

The last step is quadrature: a handful of *g-points* (Aeolus uses 2 per
band by default, configurable) reproduces to a fraction of a percent
what a million spectral points would give. The "correlated" assumption
is that the *ordering* of ``k`` values by wavenumber stays the same at
different pressures and temperatures, so a given ``g`` refers to the
same spectral regions throughout the column. This is excellent within
a band dominated by one species and degrades where bands overlap.

### How Aeolus builds ``k(g)``

Operational models (RRTMG, RRTMGP) tabulate ``k(g, p, T)`` from
line-by-line calculations and ship it as a data file. Aeolus does not
(yet): when `src/GasOptics.jl` was written the NetCDF/HDF5 stack those
files need did not load on the development machine, so it *constructs*
the distribution analytically. Since #310 NCDatasets is a dependency and
reading an RRTMGP table would be possible:

1. A **Malkmus statistical narrow-band model** treats the band as a
   random array of Lorentz lines with an exponential-tailed inverse
   distribution of line strengths. Its band transmission has the
   closed form
   ```math
   -\ln \overline{T} = \frac{\pi \gamma}{2\delta}
   \left[ \sqrt{1 + \frac{4 S u}{\pi \gamma}} - 1 \right],
   ```
   with ``S/\delta`` the mean line strength per unit spacing and
   ``\gamma`` the collision half-width — two parameters per species per
   band, which Aeolus embeds as tables.
2. **Domoto (1974) inversion** recovers ``k(g)`` from that
   transmission function by inverse Laplace transform, which for the
   Malkmus form is analytic and involves the error function.
3. Pressure and temperature scaling is applied to the absorber path in
   the usual Curtis-Godson manner.

This is a *reconstruction*, not a fit to line-by-line data, and is
correspondingly less accurate — but it is self-contained, it loads
anywhere, and it lands the broadband fluxes within a few percent of
observation (clear-sky OLR 229 W/m², surface downwelling 346 W/m²,
greenhouse effect 161 W/m²). An RRTMGP backend can be added later
behind the same `AbstractGasOptics` interface wherever it loads.

!!! note "Numerical hazard"
    At large absorber amounts ``\overline{T}`` underflows to exactly
    zero in floating point, and ``\ln \overline{T}`` becomes ``-\infty``.
    Aeolus provides `malkmus_log_transmission`, which evaluates
    ``\ln \overline{T}`` directly from the closed form above without
    ever forming ``\overline{T}``. This is not a cosmetic fix — the
    unguarded version produces `NaN` optical depths in the lower
    troposphere.

## 3. The two-stream approximation

Even monochromatically, the transfer equation is an integro-differential
equation in five dimensions. The two-stream approximation collapses the
angular dimension by assuming the diffuse radiance field can be
represented by just two hemispheric fluxes, ``F^\uparrow`` and
``F^\downarrow``. Substituting a two-term expansion into the transfer
equation gives the coupled pair

```math
\frac{dF^\uparrow}{d\tau} = \gamma_1 F^\uparrow - \gamma_2 F^\downarrow - \text{(source)},
\qquad
\frac{dF^\downarrow}{d\tau} = \gamma_2 F^\uparrow - \gamma_1 F^\downarrow + \text{(source)} .
```

The different **closures** differ only in how ``\gamma_1 \dots \gamma_4``
are defined. Aeolus ships three, all behind one interface:

| Closure | Best for | Character |
| :-- | :-- | :-- |
| **Delta-Eddington** | shortwave, thick cloud | truncates the forward-scattering peak into the direct beam |
| **PIFM** (Zdunkowski) | shortwave, general | practical improved flux method; well-behaved at small ``\omega`` |
| **Quadrature** | longwave | Gaussian angular quadrature; the standard LW choice |

### Delta-scaling

Cloud droplets scatter overwhelmingly forward: ``g \approx 0.86`` means
most "scattered" photons barely change direction. Treating them as
scattered wastes the two-stream approximation's limited angular
resolution. Delta-scaling removes a Dirac forward spike of weight
``f = g^2`` and rescales the remainder:

```math
\tau' = (1 - \omega f)\,\tau, \qquad
\omega' = \frac{(1-f)\,\omega}{1 - \omega f}, \qquad
g' = \frac{g - f}{1 - f}.
```

The scaled medium has a smaller optical depth and a much less
forward-peaked phase function, and the two-stream solution of the
scaled problem is far closer to the truth than that of the unscaled
one.

### The layer solution, and an overflow trap

For a homogeneous layer, the Meador & Weaver (1980) solution gives
closed-form reflectance and transmittance in terms of
``k = \sqrt{\gamma_1^2 - \gamma_2^2}``. As written in the paper, the
expressions contain ``e^{+k\tau}`` in both numerator and denominator.
For ``\tau = 10^4`` — an entirely reachable value in a correlated-k
line centre — ``e^{k\tau}`` overflows to `Inf`, and `Inf/Inf` is `NaN`.

Aeolus divides numerator and denominator through by ``e^{k\tau}``
analytically, so every exponential that appears in the code is of the
form ``e^{-k\tau} \le 1``. The result is algebraically identical and
numerically bounded at any optical depth.

### Linear-in-``\tau`` longwave source

Within a layer the Planck source is not constant — the temperature
varies across it, and in an optically thick layer the emission that
escapes comes disproportionately from the near edge. Assuming a
constant source therefore misplaces the emission. Writing
``B(\tau) = B_{\text{top}} + \Delta \cdot \tau/\tau_d`` and integrating
the layer emission properly, using

```math
\int_0^x s\, e^{-s} \, ds = 1 - e^{-x}(1+x),
```

gives layer sources that are measurably more accurate: against a
sub-layered reference flux of 180.66 W/m², the linear form errs by
0.55 W/m² where the constant form errs by 3.30 — a sixfold
improvement at no extra cost.

### Column solution: adding/doubling

Individual layer reflectances and transmittances are combined into a
full column by the **adding** method, which accounts for the infinite
series of internal reflections between two stacked layers in closed
form:

```math
R_{12} = R_1 + \frac{T_1 R_2 T_1}{1 - R_1 R_2}, \qquad
T_{12} = \frac{T_1 T_2}{1 - R_1 R_2} .
```

The denominator ``(1 - R_1 R_2)^{-1}`` is exactly the geometric sum of
photons bouncing between the two interfaces — the same structure that
reappears in §6 as the urban radiosity matrix inverse.

The adding recurrence is sequential in the vertical, but the g-points and
the columns are independent. The multi-column solves (`solve_sw!` /
`solve_lw!` on optics `τ[k, g, column]`, #305) therefore run one work
item per (g-point, column), with the recurrence over ``k`` inside the
item. The broadband flux is the sum over g-points, taken in ascending
order. A single column is the one-column case of the same code, so it
gets the same bits at every thread count.

## 4. From fluxes to heating rates

The quantity the LES actually needs is a temperature tendency. Net
flux is defined positive downward, ``F^{net} = F^\downarrow - F^\uparrow``,
stored on *levels* (cell faces), and the heating rate of the layer
between two levels follows from energy conservation:

```math
\left(\frac{\partial T}{\partial t}\right)_{\text{rad}}
= -\frac{1}{\rho c_p} \frac{\partial F^{net}}{\partial z} .
```

A layer that receives more flux than it passes on heats. Clear-sky
tropospheric air cools radiatively at roughly 1–2 K/day — the measured
Aeolus mean is **−1.00 K/day**, with the profile ranging from −2.48 to
−0.089 K/day. This cooling is not incidental: it is the destabilising
term that drives much of the convection an LES exists to resolve.

## 5. Solar geometry

The shortwave boundary condition requires the sun's position to
sub-degree accuracy — an error of one degree in solar elevation near
sunrise changes surface irradiance by tens of W/m². Aeolus implements
the **Meeus** algorithm (`src/SolarGeometry.jl`): Julian day, geometric
mean longitude and anomaly, equation of centre, apparent longitude with
nutation, obliquity, declination and hour angle. Accuracy is better
than 0.01° over the modern era, against roughly 1° for the sinusoidal
approximation Era I used.

Three corrections matter near the horizon, and all three are
implemented:

* **Atmospheric refraction** (Bennett's formula) lifts the apparent sun
  by about 34′ at the horizon — more than its own diameter, which is
  why the sun is already geometrically below the horizon at the moment
  it appears to set.
* **Air mass** (Kasten & Young) departs sharply from ``1/\cos\theta_z``
  near grazing incidence; the secant formula diverges at the horizon
  while the true air mass is finite (≈ 38). Note that Kasten-Young is
  valid *at* the horizon, so the domain guard must be
  ``h_{\text{apparent}} < 0``, not ``\le 0``.
* **Earth-sun distance** varies by ±1.7 % over the year, so
  extraterrestrial irradiance ranges from 1317 W/m² at aphelion to
  1413 W/m² at perihelion about the 1361 W/m² mean.

Global irradiance is split into direct and diffuse components by the
**Erbs** or **Reindl** correlations, keyed on the clearness index
``k_t = \text{GHI}/\text{ETR}``. The split matters enormously in urban
geometry: direct beam casts sharp shadows, diffuse does not, so the
same total irradiance produces a completely different surface
temperature pattern depending on how it is partitioned.

## 6. Beyond plane-parallel

Everything above assumes a **plane-parallel** atmosphere: horizontally
infinite, uniform layers. That assumption is very good for the free
troposphere and very bad for the two cases an urban LES cares most
about — broken cloud fields, and street canyons.

Aeolus addresses each with a dedicated solver, documented in their own
modules:

* **3-D Monte-Carlo** (`src/MonteCarloRT.jl`) traces individual photon
  packets through the voxel grid, capturing side illumination, shadow
  displacement and inter-cloud scattering that the independent-column
  approximation cannot represent. It is the reference solution against
  which the two-stream result is checked, not a production-speed
  solver. Every photon draws from its own counter-based random stream,
  keyed by the seed, the g-point and the photon index (#305). A solve is
  therefore reproducible from its seed and identical at every thread
  count, and there is no unseeded default.
* **Urban radiosity** (`src/Radiosity.jl`) treats the built environment
  as a set of finite facets exchanging radiation through view factors,
  solving the multiple-reflection problem exactly rather than through a
  sky-view-factor approximation. Radiative *trapping* in canyons —
  where the effective albedo of a street falls well below the material
  albedo of its surfaces because photons get several chances to be
  absorbed — is a first-order driver of the urban heat island and
  cannot be captured plane-parallel.
* **Spectral surface optics** (`src/SurfaceOptics.jl`) supplies the
  wavelength- and angle-dependent reflectance these solvers need, in
  place of a single grey albedo.
* **Aerosol optics** (`src/AerosolOptics.jl`) supplies Mie-derived
  ``\tau``, ``\omega`` and ``g`` for the aerosol burden, including
  hygroscopic growth, which dominates visibility and the direct
  radiative effect at high relative humidity.

### 3-D cloud shadows: the slant path

`SolarGeometry.cloud_shadow_transmittance!` gives the direct-beam
transmittance at each surface cell by tracing the ray toward the sun
through the 3-D cloud field. With ``\tau_c`` the vertical optical depth
of cell ``c`` (extinction coefficient ``\tau_c/\Delta z``) and
``\ell_c`` the exact length of the ray inside it (a grid traversal,
Amanatides & Woo 1987),

```math
T_{ij} = \exp\Big(-\sum_{c} \frac{\tau_c}{\Delta z}\,\ell_c\Big),
```

which for a horizontally uniform layer is exactly
``\exp(-\tau/\mu_0)``: the ``1/\mu_0`` of the slant path is already in
the path lengths. (Until #130 the march applied it a second time and
returned ``\exp(-\tau/\mu_0^2)``, 0.20 instead of 0.45 at 30° elevation
for ``\tau = 0.4``, and counted the lowest cell only half.)

The cloud optical depth is ``\tau = k\,\rho q\,\Delta z`` with the
mass extinction of `RadiationCore`,
``k_{liq} = 1.5/r_{eff}`` m² g⁻¹ (``r_{eff}`` in μm; 0.15 m² g⁻¹ at
10 μm) and ``k_{ice} = 1/r_{eff}`` — the same coefficients
`TwoStream.MieCloudOptics` uses, so a column scheme and the shadow see
the same cloud. A forward-peaked droplet phase function sends most of
the light it "removes" on in almost the same direction; for the beam
that reaches the ground the Berlin case therefore uses the
delta-scaled depth ``\tau' = (1 - \omega g^2)\tau``, the same scaling
the two-stream solve applies (``\approx 0.26\,\tau`` for liquid in the
visible).

### An LES column under a standard atmosphere

An LES domain is a few hundred metres deep; its sky longwave and the
sunlight at its top come from the ten kilometres above it.
`RadiationCore.column_from_profiles` turns LES profiles into a
hydrostatic column and `stitch_columns` puts it under the part of a
deeper column (`GasOptics.us_standard_column`) above the LES top, with
the pressure made continuous. A lower column that follows the standard
column's own temperature and humidity gives the same surface ``L^\downarrow``
(to 0.02 W/m²), outgoing longwave and global shortwave (to 0.12 W/m²)
as the standard column alone.

## References

* Amanatides, J. & Woo, A. (1987). A fast voxel traversal algorithm for
  ray tracing. *Eurographics '87*, 3–10.
* Meador, W. E. & Weaver, W. R. (1980). Two-stream approximations to
  radiative transfer in planetary atmospheres. *J. Atmos. Sci.*, 37,
  630–643.
* Joseph, J. H., Wiscombe, W. J. & Weinman, J. A. (1976). The
  delta-Eddington approximation for radiative flux transfer.
  *J. Atmos. Sci.*, 33, 2452–2459.
* Zdunkowski, W. G., Welch, R. M. & Korb, G. (1980). An investigation
  of the structure of typical two-stream methods. *Beitr. Phys. Atmos.*,
  53, 147–166.
* Goody, R. M. & Yung, Y. L. (1989). *Atmospheric Radiation:
  Theoretical Basis*, 2nd ed. Oxford University Press.
* Domoto, G. A. (1974). Frequency integration for radiative transfer
  problems involving homogeneous non-grey gases. *JQSRT*, 14, 935–942.
* Malkmus, W. (1967). Random Lorentz band model with exponential-tailed
  S⁻¹ line-intensity distribution. *J. Opt. Soc. Am.*, 57, 323–329.
* Mlawer, E. J. et al. (1997). Radiative transfer for inhomogeneous
  atmospheres: RRTM. *J. Geophys. Res.*, 102, 16663–16682.
* Meeus, J. (1998). *Astronomical Algorithms*, 2nd ed. Willmann-Bell.
* Kasten, F. & Young, A. T. (1989). Revised optical air mass tables and
  approximation formula. *Appl. Opt.*, 28, 4735–4738.
* Erbs, D. G., Klein, S. A. & Duffie, J. A. (1982). Estimation of the
  diffuse radiation fraction. *Solar Energy*, 28, 293–302.
