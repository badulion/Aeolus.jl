# Visibility

Visibility is the diagnostic that transport, aviation and air-quality
users read first, and it is almost entirely an optics problem: given the
particles in a volume of air, how far can an observer see? This page
develops the diagnostic in `src/Visibility.jl` (Phase 44), from
Koschmieder's law to the extinction of fog, haze and clear air, and
states where each approximation breaks.

## 1. Koschmieder's law

An observer looks at a black object against the horizon sky through a
homogeneous atmosphere with extinction coefficient ``\beta_{ext}``
(m⁻¹). Light from the object is attenuated as ``e^{-\beta x}``; airlight
scattered into the line of sight adds a brightness that approaches the
horizon-sky brightness as ``1-e^{-\beta x}``. The apparent contrast
against the sky therefore decays as

```math
C(x) = C_0\,e^{-\beta_{ext} x},\qquad C_0 = -1 \text{ for a black object}.
```

Visibility is the distance at which ``|C|`` falls to the observer's
contrast threshold ``\varepsilon`` (Koschmieder 1924):

```math
V = \frac{-\ln\varepsilon}{\beta_{ext}}.
```

Two thresholds are in use:

| threshold | ``-\ln\varepsilon`` | name |
|:--|:--|:--|
| ``\varepsilon = 0.02`` | 3.912 | Koschmieder's value; assumed by the Kunkel (1984) fit. **Default** (`KOSCHMIEDER_THRESHOLD`). |
| ``\varepsilon = 0.05`` | 2.996 | WMO meteorological optical range, MOR (`MOR_THRESHOLD`). |

The MOR is shorter by ``\ln 20/\ln 50 = 0.766``. Every visibility quoted
by this module states its ``\varepsilon``.

Koschmieder assumes a uniform path, a black target, a uniform horizon
sky and daylight. It does not describe night-time visibility of lights
(Allard's law), nor slant visibility through a fog top.

## 2. Extinction

Extinction coefficients of independent scatterers add, so

```math
\beta_{ext} = \beta_{drop} + \beta_{haze} + \beta_{Rayleigh}
```

evaluated at 550 nm, the peak of photopic sensitivity.

### 2.1 Droplets

For a droplet number distribution ``n(r)`` (m⁻⁴),

```math
\beta_{drop} = \int_0^\infty \pi r^2\,Q_{ext}\!\left(\frac{2\pi r}{\lambda}, m_w\right) n(r)\,dr.
```

``Q_{ext}`` is the Mie extinction efficiency from Phase 37
(`AerosolOptics.mie_sphere`) with the Hale & Querry refractive index of
water, ``m_w = 1.333 + 2\times10^{-9} i`` at 550 nm. For large drops
``Q_{ext}\to 2`` — the *extinction paradox*: a large sphere removes twice
its geometric cross-section, half by absorption and reflection and half
by diffraction into a narrow forward lobe. The approach is slow; the
Nussenzveig & Wiscombe (1980) edge term

```math
Q_{ext} \simeq 2 + 1.9923861\,x^{-2/3}
```

still leaves ``Q_{ext}-2 = 1.3\times10^{-3}`` for a 5 mm raindrop
(``x = 5.7\times10^4``). The module uses Mie up to ``x = 10^4`` and this
asymptote above it. Pointwise, Mie oscillates about the asymptote by up
to ``3\times10^{-3}`` (interference ripple); averaged over ±10 % in
``x`` the two agree to 0.03 % (``x = 10^3``) – 0.14 % (``x = 9\times10^3``).

Three spectra are provided: monodisperse, the Hansen (1971) gamma
distribution

```math
n(r) \propto r^{(1-3b)/b}\,e^{-r/(ab)},\qquad a = r_{eff},\; b = v_{eff},
```

and a binned spectrum (also built from a Phase 41 superdroplet
ensemble). For the gamma the shape factor
``k = (r_v/r_{eff})^3 = (1-b)(1-2b)`` is exact, so a spectrum built from
a bulk ``(\mathrm{LWC}, N_d)`` pair takes its effective radius from
`AerosolCloud.droplet_effective_radius` with that ``k`` — the same droplet
geometry the Phase 42 indirect-effect diagnostics use. ``v_{eff} = 0.1``
gives ``k = 0.72``, inside the Martin et al. (1994) range.

In the geometric limit ``Q_{ext}=2``,

```math
\beta_{drop} = 2\pi N\langle r^2\rangle = \frac{3\,\mathrm{LWC}}{2\rho_w r_{eff}}
\;\propto\; N_d^{1/3}\,\mathrm{LWC}^{2/3},
```

which is exactly the per-metre form of the cloud optical depth
``\tau = 3\,\mathrm{LWP}/(2\rho_w r_{eff})`` used by `TwoStream` and
`AerosolCloud`; the test suite checks the two agree to ``10^{-12}``. At
fixed water, more droplets mean lower visibility — the aerosol indirect
effect acting on fog. For fog spectra (``r_{eff}`` = 2–15 μm) Mie is 3–14 %
above the geometric limit.

**Quadrature.** Water barely absorbs at 550 nm, so ``Q_{ext}(x)`` carries
narrow resonance spikes. The gamma integral uses Simpson in ``\ln r``
with at least 10 points per unit of size parameter; against a
60 000-interval reference the error is typically ``3\times10^{-4}`` and
at worst ``1.9\times10^{-3}`` (``r_{eff}=2`` μm). A coarser 5-points-per-π
rule, as used for the aerosol modes in Phase 37, aliased the resonances
into errors of up to ``5\times10^{-3}`` for droplets and was rejected.
The quadrature returns ``\beta = \pi N\langle r^2\rangle\,\bar Q``: it
integrates only the area-weighted mean efficiency ``\bar Q`` and takes
``\langle r^2\rangle`` analytically. Normalising by the numerically
integrated *number* instead (the first implementation) let the small-drop
number cut off at the lower integration bound — most of ``N`` when
``v_{eff}\to 0.5`` — inflate ``\beta``, to an effective
``Q_{ext} = 6.6`` at ``v_{eff} = 0.49``; the area-weighted form gives 2.12.

### 2.2 Haze

Unactivated aerosol grows hygroscopically as the humidity rises and
dominates visibility between clear air and fog. This term is Phase 37's
`AerosolOptics.bulk_optics`: κ-Köhler wet radius and Mie for each
log-normal mode of an OPAC mixture. Measured at 550 nm, with Rayleigh
included, ``\varepsilon = 0.02``:

| mixture | RH 0 % | 50 % | 80 % | 95 % |
|:--|:--|:--|:--|:--|
| continental clean | 170 km | 151 km | 112 km | 50 km |
| urban (RH 80 %) | | | 12.0 km | |

### 2.3 Clear air

Molecular scattering sets the upper bound. With
``\beta_R = n\,\sigma_R(\lambda)``, ``n = p/(k_B T)`` and
``\sigma_R`` from `GasOptics.rayleigh_cross_section` (the same
cross-section the radiation code uses), 1013.25 hPa, 288.15 K and 550 nm
give ``\beta_R = 1.27\times10^{-5}`` m⁻¹ and ``V = 307`` km. That
cross-section is a pure ``\lambda^{-4}`` fit about 10 % above the
Bucholtz (1995) / Bodhaine et al. (1999) tables, so the true
Rayleigh-limited visibility is closer to 340 km; the bias is inherited,
not duplicated.

## 3. Empirical fog parameterisations

Operational models diagnose fog visibility from bulk water alone:

* **Kunkel (1984):** ``\beta = 144.7\,\mathrm{LWC}^{0.88}`` km⁻¹ (LWC in
  g m⁻³), i.e. ``V = 27\,\mathrm{m}\times\mathrm{LWC}^{-0.88}`` at
  ``\varepsilon = 0.02``. No droplet-number dependence.
* **Gultepe et al. (2006):** ``V = 1.002\,\mathrm{km}/(\mathrm{LWC}\,N_d)^{0.6473}``
  (LWC in g m⁻³, ``N_d`` in cm⁻³), fitted to FRAM-C observations.

Comparison with the explicit calculation (gamma, ``v_{eff}=0.1``,
``N_d = 100`` cm⁻³, Mie, 550 nm):

| LWC [g m⁻³] | ``r_{eff}`` [μm] | ``\beta_{Mie}/\beta_{geo}`` | V (ε=0.02) | V (MOR) | Kunkel | Gultepe |
|:--|:--|:--|:--|:--|:--|:--|
| 0.10 | 6.92 | 1.058 | 171 m | 131 m | 205 m | 226 m |
| 0.20 | 8.72 | 1.049 | 108 m | 83 m | 111 m | 144 m |
| 0.30 | 9.98 | 1.045 | 83 m | 64 m | 78 m | 111 m |

Koschmieder/Kunkel ranges 0.83–1.07 and Koschmieder/Gultepe 0.75–0.76.
Dense fog of 0.1–0.3 g m⁻³ gives visibilities of order 100 m, an order
of magnitude below the 1 km fog definition. The residual differences are
what one expects: Kunkel carries no ``N_d`` dependence, and Gultepe was
fitted to observed (often broader, bimodal) spectra and to measured
visibility rather than an ideal black target.

## References

* Koschmieder, H. (1924). *Beitr. Phys. freien Atm.*, 12, 33–53, 171–181.
* Hansen, J. E. (1971). *J. Atmos. Sci.*, 28, 1400–1426.
* Nussenzveig, H. M. & Wiscombe, W. J. (1980). *Phys. Rev. Lett.*, 45, 1490–1494.
* Kunkel, B. A. (1984). *J. Climate Appl. Meteor.*, 23, 34–41.
* Martin, G. M., Johnson, D. W. & Spice, A. (1994). *J. Atmos. Sci.*, 51, 1823–1842.
* Bucholtz, A. (1995). *Appl. Opt.*, 34, 2765–2773.
* Bodhaine, B. A. et al. (1999). *J. Atmos. Oceanic Technol.*, 16, 1854–1861.
* Gultepe, I., Müller, M. D. & Boybeyi, Z. (2006). *J. Appl. Meteor. Climatol.*, 45, 1469–1480.
* WMO (2018). *Guide to Instruments and Methods of Observation*, WMO-No. 8, Vol. I, Ch. 9.
