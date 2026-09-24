# Ice nucleation

Between 0 °C and about −38 °C liquid water is metastable: a supercooled
droplet does not freeze on its own on any atmospheric timescale. The
first ice in a mixed-phase cloud forms *heterogeneously*, on a solid
surface that templates the ice lattice — an **ice-nucleating particle**
(INP). INP are rare, typically one ambient aerosol particle in
``10^5``–``10^6`` at −20 °C, and dominated by two classes: mineral
dust (feldspars and clays) and primary biological particles
(bacteria, fungal spores, pollen fragments), which are the most active
INP known at temperatures warmer than −15 °C (Hoose & Möhler 2012;
Kanji et al. 2017).

This page develops the theory behind `src/IceNucleation.jl` (Phase 43).
Section 1–4 cover the prognostic INP population (issue #5); the
freezing parameterisations that draw on it (issue #6) and the coupling
into the P3 ice category (issue #7) build on the interface described in
§5.

## 1. Why INP must be prognostic

Most operational freezing parameterisations are *singular*: they state
how many INP are active at temperature ``T``, independently of time.
DeMott et al. (2010) write it in terms of the number of aerosol
particles larger than 0.5 μm diameter,

```math
N_\mathrm{INP}(T) = a\,(273.16 - T)^b\,N_{>0.5\,\mu\mathrm{m}}^{\,c(273.16-T)+d},
```

and the ice-nucleation-active-site (INAS) formulations (Niemand et al.
2012; Ullrich et al. 2017) in terms of the particle surface area ``A``
and a site density ``n_s(T)``:

```math
N_\mathrm{INP}(T) = N\left[1 - \exp\!\left(-n_s(T)\,\frac{A}{N}\right)\right]
\;\xrightarrow{\;n_s A/N \ll 1\;}\; n_s(T)\,A .
```

Both describe the *cumulative* number frozen by the time the parcel has
cooled to ``T``. Applied every time step to a diagnostic aerosol field,
they re-freeze the same particles over and over and turn a one-off
nucleation burst into a steady ice source. In Arctic mixed-phase
stratus that is the difference between a cloud that persists for days —
as observed — and one that glaciates and precipitates itself out in
hours (Harrington & Olsson 2001; Morrison et al. 2012). Solomon et al.
(2015) showed that realistic ice number in such clouds requires
tracking INP depletion *and* the return of INP when ice sublimates.

The INP population is therefore split into two prognostic populations
per mode:

| population | meaning |
| :-- | :-- |
| **available** | has not nucleated ice; interstitial or immersed in a droplet |
| **consumed**  | inside an ice crystal; cannot nucleate again until released |

## 2. Modal representation

Each INP mode is a log-normal in dry radius with fixed geometric
standard deviation ``\sigma_g``, carried as two moments — number ``N``
and dry mass ``M`` — per population. The median radius is diagnosed,

```math
r_g = \left[\frac{M}{N\,\rho_p\,\tfrac{4\pi}{3}\exp(4.5\ln^2\sigma_g)}\right]^{1/3},
```

exactly as in the Phase 42 `AerosolReservoir`; the mode shapes and
species are the Phase 37 `LogNormalMode`/`AerosolSpecies` records
(`:mineral_dust`, `:biological`), converted to SI with
`AerosolCloud.arg_mode`. The species record is the only source of the
particle density: a population built from SI `ARGMode`s whose `ρ_p`
disagrees with its species is rejected. Size-selective sinks (sedimentation of the
coarse tail, nucleation on the largest particles) therefore shrink the
diagnosed ``r_g`` of what remains, which is the memory a two-moment
scheme can hold.

### Analytic moments above a cut-off

The ``k``-th moment of a log-normal is log-normal with the same
``\sigma_g`` and median ``r_g e^{k\ln^2\sigma_g}``, so the part above a
cut-off radius ``r_c`` is

```math
\mathcal{M}_k(>r_c) = \frac{\mathcal{M}_k}{2}\,
  \mathrm{erfc}\!\left(\frac{\ln(r_c/r_g)}{\sqrt2\ln\sigma_g}
  - \frac{k\ln\sigma_g}{\sqrt2}\right).
```

``k = 0`` (number, DeMott's ``N_{>0.5\,\mu\mathrm m}`` with
``r_c = 0.25`` μm) and ``k = 3`` (mass) are the Phase 42
`number_above_radius`/`mass_above_radius`; ``k = 2`` (surface area,
for INAS) is `surface_area_above_radius`, with total
``A = 4\pi N r_g^2 e^{2\ln^2\sigma_g}``. The test suite checks all three
against an independent 200 000-point quadrature in ``\ln r``: worst
relative difference ``2.1\times10^{-9}`` over ``\sigma_g`` = 1.4–2.8.

### Size-dependent per-particle probabilities

Every sink in the module acts particle by particle with a
size-dependent probability ``p(r)``: stochastic freezing
``1 - e^{-J(T)4\pi r^2\Delta t}``, settling out of a layer
``1 - e^{-v(r)\Delta t/\Delta z}``, size-dependent scavenging. The
removed fractions of number and mass are

```math
\frac{\Delta N}{N} = \int p\!\left(r_g e^{\ln\sigma_g z}\right)\varphi(z)\,dz,\qquad
\frac{\Delta M}{M} = \int p\!\left(r_g e^{\ln\sigma_g (z + 3\ln\sigma_g)}\right)\varphi(z)\,dz,
```

with ``\varphi`` the standard normal density. Integrating each moment
in its own shifted variable makes ``p \equiv c`` remove exactly ``c``
of every moment, so a size-independent sink leaves ``r_g`` unchanged
to rounding. The trapezoid rule on a Gaussian weight is spectrally
accurate for smooth ``p``; for the INAS curve it agrees with an
independent quadrature to ``3\times10^{-15}``. A discontinuous ``p`` (a
hard size threshold) converges only as the node spacing: 0.3 % absolute
error in the frozen fraction, which is why the threshold case has its
own analytic path (`selection = :largest`).

## 3. The budget

Each operator books what it moves, so two identities hold exactly:

```math
N_\mathrm{avail} + N_\mathrm{cons} = N_\mathrm{init} + N_\mathrm{src}
  - N_\mathrm{wet} - N_\mathrm{sed},
\qquad
N_\mathrm{cons} = N_\mathrm{nuc} - N_\mathrm{rel} - N_\mathrm{cons,lost},
```

and the same for mass. `inp_budget_residual` evaluates them; over 1500
random steps of every source and sink operator the worst relative
residual is ``9\times10^{-16}``.

| operator | moves | budget term |
| :-- | :-- | :-- |
| `inp_emit!` | source → available | ``N_\mathrm{src}`` |
| `inp_entrain!` | environment ↔ available (linear mixing) | ``N_\mathrm{src}`` (may be negative) |
| `inp_consume!` | available → consumed (nucleation) | ``N_\mathrm{nuc}`` |
| `inp_nucleate_singular!` | available → consumed, increment only | ``N_\mathrm{nuc}`` |
| `inp_release!` | consumed → available (ice sublimated) | ``N_\mathrm{rel}`` |
| `inp_scavenge!` | available or consumed → removed | ``N_\mathrm{wet}`` |
| `inp_sediment!` | available → removed (settling) | ``N_\mathrm{sed}`` |

All removals use the exact exponential solution for the step, so they
are positive for any ``\Delta t``: one 3600 s scavenging step equals 36
steps of 100 s to ``10^{-12}``.

### Singular nucleation without double counting

A singular scheme's statement applies to the **total** population —
particles that froze earlier would freeze again at this temperature —
so the nucleation increment is

```math
\Delta N = \min\!\left[N_\mathrm{avail},\;
  \max\!\left(0,\; N_\mathrm{frozen}(T;\,N_\mathrm{avail}+N_\mathrm{cons})
  - N_\mathrm{cons}\right)\right].
```

The mass increment ``M_\mathrm{frozen} - M_\mathrm{cons}`` is computed
the same way, but it is a difference of two separately computed totals
and need not describe a set of particles that exists. If earlier
nucleation took the largest particles (the DeMott number form uses
`selection = :largest`), the consumed population already holds more
mass than an INAS ``p(r)`` assigns to that many particles, and the
difference is negative. The increment is therefore required to lie
between the mass of the ``\Delta N`` smallest and the ``\Delta N``
largest available particles. Outside that range, ``\Delta N`` moves
with the mean mass of the available particles that ``p`` selects, so
no particle is ever moved without its mass.

Holding ``T`` fixed nucleates nothing after the first call (tested to
be exactly zero); cooling further nucleates only the difference;
warming nucleates nothing. INP returned by `inp_release!` re-freeze
once the parcel is again as cold as when they first froze — the ice
recycling of Solomon et al. (2015).

### Which particles nucleate

For a nucleated number ``\Delta N`` the mass that moves with it depends
on which particles froze:

* `selection = :largest` — ``\Delta N`` is converted to the radius
  ``r_c`` with ``N(>r_c) = \Delta N`` and the mass above ``r_c`` moves.
  Right for every singular scheme, whose activity rises monotonically
  with size.
* `selection = :random` — ``\Delta M = M\,\Delta N/N``, ``r_g``
  unchanged.
* a probability function ``p(r)`` — the integrals above.

## 4. Sedimentation and scavenging

Dry INP settle at the Stokes speed with Cunningham slip, the regime-1
law of the droplet terminal velocity (`Superdroplet.stokes_settling_velocity`,
shared so that there is one drag law in the package):

```math
v(r) = \left(1 + 2.51\frac{\lambda}{d}\right)\frac{(\rho_p-\rho_a)\,g\,d^2}{18\mu}.
```

It reproduces the Seinfeld & Pandis (2016) Table 9.4 speeds for
unit-density spheres, 3.49×10⁻³ cm s⁻¹ at 1 μm and 0.304 cm s⁻¹ at
10 μm (tabulated 3.5×10⁻³ and 0.31). Without slip the moment-weighted
speeds are closed-form,

```math
\langle v\rangle_N = k r_g^2 e^{2\ln^2\sigma_g},\qquad
\langle v\rangle_M = k r_g^2 e^{8\ln^2\sigma_g},
```

and the quadrature matches them to ``4\times10^{-15}``. Mass settles
``e^{6\ln^2\sigma_g}`` times faster than number — 17.8× for
``\sigma_g = 2`` — so a settling dust layer loses its mass long before
its number, and its INP (which sit in the coarse tail) faster still.
For a DeMott-type dust mode with ``r_g = 0.4`` μm, ``\sigma_g = 2``,
``\langle v\rangle_N = 0.14`` mm s⁻¹ and ``\langle v\rangle_M = 2.4``
mm s⁻¹.

Scavenging takes a coefficient ``\Lambda`` (s⁻¹), constant or a
function ``\Lambda(r)``; consumed INP are removed with their ice by
passing the host's ice removal rate with `population = :consumed`.

## 5. Interface for the freezing and P3 coupling

`IceNucleation` never touches a hydrometeor state. Freezing routines
(issue #6) take an `INPPopulation`, temperature and supersaturation,
compute ice-number and ice-mass **source terms**, and draw the
matching INP sink through `inp_consume!` or `inp_nucleate_singular!`.
The P3 coupling (issue #7) adds those rates to its ice category and
removes the corresponding liquid or vapour; when ice sublimates it
calls `inp_release!`, and when ice precipitates out of a cell it calls
`inp_scavenge!(…; population = :consumed)`. Nucleation can therefore be
tested in isolation and is independent of any one bulk scheme.

## References

* DeMott, P. J. et al. (2010), *Predicting global atmospheric ice nuclei
  distributions and their impacts on climate*, PNAS 107, 11217.
* DeMott, P. J. et al. (2015), *Integrating laboratory and field data to
  quantify the immersion freezing ice nucleation activity of mineral
  dust particles*, ACP 15, 393.
* Harrington, J. Y. & Olsson, P. Q. (2001), *On the potential influence
  of ice nuclei on surface-forced marine stratocumulus cloud dynamics*,
  JGR 106, 27473.
* Hoose, C. & Möhler, O. (2012), *Heterogeneous ice nucleation on
  atmospheric aerosols: a review of results from laboratory
  experiments*, ACP 12, 9817.
* Kanji, Z. A. et al. (2017), *Overview of ice nucleating particles*,
  Meteorol. Monogr. 58, 1.1.
* Morrison, H. et al. (2012), *Resilience of persistent Arctic
  mixed-phase clouds*, Nature Geosci. 5, 11.
* Niemand, M. et al. (2012), *A particle-surface-area-based
  parameterization of immersion freezing on desert dust particles*,
  JAS 69, 3077.
* Seinfeld, J. H. & Pandis, S. N. (2016), *Atmospheric Chemistry and
  Physics*, 3rd ed., Wiley — ch. 9, gravitational settling.
* Solomon, A., Feingold, G. & Shupe, M. D. (2015), *The role of ice
  nuclei recycling in the maintenance of cloud ice in Arctic
  mixed-phase stratocumulus*, ACP 15, 10631.
* Ullrich, R. et al. (2017), *A new ice nucleation active site
  parameterization for desert dust and soot*, JAS 74, 699.
* Whitby, E. R. & McMurry, P. H. (1997), *Modal aerosol dynamics
  modeling*, Aerosol Sci. Technol. 27, 673.
