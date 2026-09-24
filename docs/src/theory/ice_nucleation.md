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
Sections 1–4 cover the prognostic INP population (issue #5); §6–7 the
immersion and deposition freezing that draws on it and its consistency
with the Phase 39 mixed-phase ramp (issue #6). The coupling into the P3
ice category (issue #7) builds on the interface described in §5 and §6.4.

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

## 6. Immersion and deposition freezing

Freezing is **singular** throughout: each scheme states how many of a
mode's INP are frozen at ``(T, S_i)``, and `inp_nucleate_singular!`
applies it to the *total* (available + consumed) population so that
only the increment is nucleated (§3). ``\Delta t`` only converts
increments into rates.

### 6.1 Schemes

Two families, dispatched on type (`FreezingScheme`):

| scheme | mechanism | form | fitted range |
| :-- | :-- | :-- | :-- |
| `Niemand2012` | immersion, dust | ``n_s = \exp[-0.517(T-273.15) + 8.934]`` m⁻² | −36 … −12 °C |
| `Ullrich2017Immersion` | immersion, dust | ``n_s = \exp(150.577 - 0.517T)`` m⁻² | −30 … −14 °C |
| `Ullrich2017Deposition` | deposition, dust | ``n_s(T, S_i)``, below | 206 … 240 K |
| `DeMott2015` | immersion, dust | ``c_f N_{>0.5}^{1.25} e^{0.46(273.16-T) - 11.6}`` | −35 … −20 °C |
| `Tobo2013` | immersion, biological | ``N_{>0.5}^{-0.074\Delta T + 3.8} e^{0.414\Delta T - 9.671}`` | −34 … −9 °C |

The INAS schemes freeze a particle of radius ``r`` with the Poisson
probability ``1 - \exp(-n_s 4\pi r^2)``, integrated over the
log-normal with the quadrature of §2; the aerosol-count schemes are
written in std cm⁻³ / std L⁻¹ (273.15 K, 1013.25 hPa) and converted
with the ideal-gas factor ``(p/p_0)(T_0/T)``. Defaults per species:
dust — `Niemand2012` immersion + `Ullrich2017Deposition`; biological —
`Tobo2013` immersion, no deposition.

The Ullrich et al. (2017) deposition density for dust is

```math
n_s(T, S_i) = \exp\!\left\{\alpha (S_i - 1)^{1/4}
  \cos^2[\beta(T-\gamma)]\,\frac{\mathrm{arccot}[\kappa(T-\lambda)]}{\pi}\right\} - 1,
```

``\alpha = 285.692``, ``\beta = 0.017``, ``\gamma = 256.692``,
``\kappa = 0.080``, ``\lambda = 200.745``, with ``S_i`` clamped at
water saturation ``e_s/e_{si}`` (both from `MoistThermodynamics`). The
``-1`` is ours: the published ``\exp\{\cdot\}`` gives 1 m⁻² at
``S_i = 1``; subtracting it makes deposition vanish continuously at ice
saturation and changes ``n_s`` by less than 10⁻⁶ wherever it exceeds
10⁶ m⁻². Meyers et al. (1992) was not used for deposition because it
predicts ice from ``S_i`` alone and cannot consume a prognostic INP.

**Fitted ranges.** No scheme is extrapolated: warmer than its range it
predicts no ice (an exponential fit carried to 0 °C gives spurious ice
where no data constrain it), colder it is held at the cold-edge value;
at ``T \ge T_\mathrm{freeze}`` every scheme is zero regardless.

### 6.2 Measured values

| quantity | value |
| :-- | :-- |
| Niemand ``n_s(-20\,°\mathrm{C})`` | 2.347×10⁸ m⁻² |
| Niemand ``n_s(-30\,°\mathrm{C})`` | 4.13×10¹⁰ m⁻² |
| Ullrich / Niemand immersion | ``e^{0.424}`` = 1.528 at every ``T`` (same slope) |
| Ullrich deposition, 220 K, ``S_i`` = 1.1 | 2.66×10⁸ m⁻² |
| Ullrich deposition, 230 K, ``S_i`` = 1.2 | 4.13×10⁸ m⁻² |
| DeMott 2015 (``c_f`` = 3), −20 °C, ``N_{>0.5}`` = 1 std cm⁻³ | 0.273 std L⁻¹ |
| Tobo 2013, −20 °C, ``N_{>0.5}`` = 1 std cm⁻³ | 0.250 std L⁻¹ |
| DeMott (``c_f`` = 3) / Niemand, Saharan-like mode, −35 … −20 °C | 0.24 – 0.60 |
| mode frozen fraction vs independent quadrature | 4×10⁻¹⁴ relative |

The Saharan-like mode is ``N`` = 15 cm⁻³, ``r_g`` = 0.4 μm,
``\sigma_g`` = 2 (``N_{>0.5}`` = 11.3 cm⁻³, ``A`` = 7.9×10⁻⁵ m² m⁻³,
cf. DeMott et al. 2015 §3.3). DeMott et al. report excellent agreement
with Niemand for their *measured* Saharan-layer size distribution; for
this idealised log-normal the two independent fits agree within a
factor of 4.2, DeMott lower.

### 6.3 Source terms

`ice_nucleation!(inp, T, S_i, Δt; p, N_liq, M_liq, immersed_fraction)`
applies, per mode, the immersion scheme to the immersed fraction ``f``
of the INP (default 1 in cloud, 0 out of cloud) and the deposition
scheme to the rest. When both are INAS schemes the per-particle
probabilities are combined, ``p(r) = f\,p_\mathrm{imm}(r) + (1-f)\,p_\mathrm{dep}(r)``,
so the consumed mass stays size-resolved; otherwise the frozen numbers
are combined. The increment is split between mechanisms in proportion
to their contributions. Then:

* an immersion-frozen INP freezes one droplet of the mean droplet mass
  ``M_\mathrm{liq}/N_\mathrm{liq}``; at most ``N_\mathrm{liq}``
  droplets freeze;
* a deposition-nucleated INP makes one crystal of mass
  ``m_{i0} = \tfrac{4\pi}{3}\,900\,(1\,\mu\mathrm{m})^3`` =
  3.8×10⁻¹⁵ kg (the P3 nucleated-ice mass), drawn from vapour but never
  more than the excess over ice saturation,
  ``(S_i - 1)e_{si}(T)/(R_v T)``;
* the latent heat is ``L_f(T) M_\mathrm{liq} + L_s(T) M_\mathrm{vap}``
  with the Kirchhoff latent heats of `MoistThermodynamics`.

The result is an `IceNucleationTendency` of increments (all
non-negative); `nucleation_rates` divides by ``\Delta t``. The tests
check, for every call, INP consumed = crystals formed (to 10⁻¹²),
``M_\mathrm{ice} = M_\mathrm{liq} + M_\mathrm{vap}`` exactly, heat per
kilogram = ``L_f`` or ``L_s``, closure of the INP budget (§3) to
10⁻¹⁴, a second call at the same ``(T, S_i)`` nucleating exactly zero,
and two cooling steps summing to one step straight to the colder
temperature (to 10⁻¹⁰). For the reference parcel of the tests (1 cm⁻³
dust, 0.01 cm⁻³ biological, 100 cm⁻³ droplets, 0.2 g m⁻³) immersion
freezing at −25 °C forms 15.6 L⁻¹ crystals on dust and 3×10⁻⁴ L⁻¹ on
biological particles; at −30 °C 150 L⁻¹; deposition at 225 K and
``S_i`` = 1.25 (out of cloud) forms 30 L⁻¹.

### 6.4 Coupling contract for P3 (issue #7)

The host passes ``T``, ``S_i`` (`relative_humidity_ice`), ``p`` and
its droplet number and liquid content, adds
``N_\mathrm{ice,imm} + N_\mathrm{ice,dep} + N_\mathrm{ice,hom}`` and
``M_\mathrm{ice}`` to its ice category, removes ``N_\mathrm{liq}``,
``M_\mathrm{liq}`` from cloud liquid and ``M_\mathrm{vap}`` from vapour,
and heats the air by ``\mathrm{heat}/(\rho c_{pm})``. When its ice
sublimates or melts it calls `inp_release!`; when ice precipitates out,
`inp_scavenge!(…; population = :consumed)`.

### 6.5 The P3 host side (issue #7)

`p3_ice_nucleation(s, inp, Δt)` is that contract for P3's single ice
category. It evaluates ``S_i`` with `relative_humidity_ice`, passes
``N_\mathrm{liq} = N_c\rho`` and ``M_\mathrm{liq} = q_c\rho`` (``\rho``
from `air_density` of the cell) and applies the increments per kg of air:
droplets frozen leave ``q_c, N_c``; vapour deposited leaves ``q_v``;
``q_i`` gains ``M_\mathrm{ice}/\rho``; ``N_i`` gains one particle per
crystal; frozen drops enter the rime mass and volume at
``\rho_\mathrm{rim,max}`` = 900 kg m⁻³ (the P3 code's treatment of
frozen cloud) while vapour-nucleated crystals are unrimed; and
``c_{pm}\Delta T = \mathrm{heat}/\rho``, which is exactly the P3
enthalpy form ``c_{pm}\,dT = -L_v\,dq_v + L_f\,dq_i``. Only cloud
droplets freeze — P3 rain has no immersion-freezing process here.

`p3_inp_release!(inp, s, r, Δt)` returns the consumed INP of ice that
sublimated or melted: the fraction ``\Delta t\,(n_\mathrm{sub} +
n_\mathrm{melt})/N_i`` of each mode, using the limited rates that
`p3_step_with_rates` applied.

The two nucleated-crystal masses are one: `NUCLEATED_ICE_MASS` is
defined as `P3_NUCLEATED_ICE_MASS` (``m_{i0}`` of P3), the mass also
given to Hallett–Mossop splinters.

| Check (tests) | Measured |
| :-- | :-- |
| P3 ice particles gained vs INP consumed, −25 °C cloud | 1e-12 rel. (15.59 L⁻¹) |
| total water across the coupling | 1e-15 rel. |
| heating vs ``L_f\Delta q_\mathrm{liq} + L_s\Delta q_\mathrm{vap}`` | 1e-10 rel. (``\Delta T`` itself to rounding of ``T``) |
| rime density of frozen drops | 900 kg m⁻³ to 1e-10 |
| INP released vs fraction of ice sublimated | 1e-12 rel.; budget residual < 1e-14 |

## 7. Consistency with the mixed-phase ramp

`MoistThermodynamics.liquid_fraction` (Phase 39) partitions condensate
between liquid and ice with a smoothstep between
``T_\mathrm{icenuc}`` = 233.15 K and ``T_\mathrm{freeze}`` = 273.15 K.
It and heterogeneous nucleation answer different questions — ``\lambda``
is an equilibrium partition for schemes *without* prognostic ice, while
nucleation sets the number of crystals that depositional growth
(Wegener–Bergeron–Findeisen, the host's business) then feeds. They must
nevertheless agree about where ice is impossible and where it is
compulsory. The rule is that `freezing_regime` is bounded by the **same
constants**:

| regime | ``T`` | ``\lambda(T)`` | nucleation |
| :-- | :-- | :-- | :-- |
| `:liquid` | ``T \ge T_\mathrm{freeze}`` | 1 | none; no INP consumed |
| `:heterogeneous` | ``T_\mathrm{icenuc} < T < T_\mathrm{freeze}`` | (0, 1) | immersion + deposition on INP |
| `:homogeneous` | ``T \le T_\mathrm{icenuc}`` | 0 | all liquid freezes (immersed INP inside it); deposition continues on the interstitial INP only |

Tested: the regime and ``\lambda`` agree on a 0.05 K grid from 220 to
280 K; at ``T_\mathrm{freeze}`` nothing freezes even for 100 cm⁻³ of
dust at ``S_i`` = 1.5; at ``T_\mathrm{icenuc}`` the frozen liquid
fraction is exactly ``1 - \lambda = 1`` and the immersed INP are
consumed inside the frozen drops; a custom ``T_\mathrm{icenuc}`` moves
both together. Inside the ramp heterogeneous freezing never outruns the
ramp's ice fraction: for 10 cm⁻³ of immersed dust the frozen liquid
fraction is at most 0.075 of ``1 - \lambda(T)``.

## 8. Validation: mixed-phase Arctic stratus

The test a mixed-phase ice scheme must pass is that an Arctic
stratocumulus deck *persists*: observed decks keep supercooled liquid
for days with 0.1–1 L⁻¹ of ice, and a scheme that makes too much ice
glaciates them through the Wegener–Bergeron–Findeisen process within
hours (Harrington et al. 1999; Morrison et al. 2011, 2012; Ovchinnikov
et al. 2014).

### Set-up and what is simplified

`run_arctic_stratus(ArcticStratusColumn(...))` integrates a **1-D
column**, not an LES, in the set-up of the ISDAC intercomparison
(Ovchinnikov et al. 2014, flight 31): the ISDAC initial ``\theta_l`` and
``q_t`` profiles (`isdac_initial_profile`; elevated mixed layer
400–825 m under an inversion), 1020 hPa at the surface, no surface
fluxes (sea ice), 200 cm⁻³ droplets, the ISDAC longwave
``F = F_0 e^{-\kappa\,\mathrm{LWP}_\downarrow} + F_1 e^{-\kappa\,\mathrm{LWP}_\uparrow}``
(``F_0`` = 72 W m⁻², ``F_1`` = 15 W m⁻², ``\kappa`` = 170 m² kg⁻¹) and
the ISDAC subsidence ``w = -Dz``, ``D`` = 5×10⁻⁶ s⁻¹ (constant above
825 m). The profile and forcing constants are quoted from the case
specification as the author of this page recalled it; the paper itself
was not accessible when this was written, so treat them as
"ISDAC-like".

Simplified, and why it matters:

* **Turbulence** is a prescribed eddy diffusivity, 30 m² s⁻¹ inside the
  400–825 m mixed layer and zero across the inversion (implicit
  diffusion of ``s_l``, the water species, the ice moments and the
  INP). Free-tropospheric air enters the mixed layer only through the
  subsidence, i.e. at the entrainment rate ``w_e = D z_i`` ≈ 4 mm s⁻¹
  that keeps the inversion stationary (Lilly 1968), not at the rate
  resolved turbulence would produce. The cloud-top cooling is therefore
  not balanced by entrainment warming as in the LES, and LWP grows with
  time even with ice (14 → 39 g m⁻² in 8 h without ice).
* **Ice initiation** is prognostic — Niemand et al. (2012) immersion
  freezing on a dust mode (5 cm⁻³, ``r_g`` = 0.4 μm, ``\sigma_g`` = 2) with
  INP consumed on freezing, entrained with free-tropospheric air,
  falling with the ice number and released where ice sublimates —
  instead of ISDAC's prescribed ice number. The dust number is chosen
  so the ice falls in the ISDAC range (0.25–4 L⁻¹).
* **Microphysics** is the full P3 of this package: process rates with
  Hallett–Mossop (inactive: the cloud is at −12 to −15 °C, as in ISDAC
  and M-PACE), saturation adjustment (which conserves ``\theta_{li}``,
  not ``s_l`` — issue #82), and ice sedimentation with P3's mass- and
  number-weighted fall speeds.

Water (column + precipitation − subsidence source) and INP (column +
surface deposit − subsidence source) budgets close to 10⁻¹². A test
that every P3 ice particle has a consumed INP (Hallett–Mossop being
inactive) caught a bug during development: the column carried INP per
m³ at a fixed transport density while P3 converts with the cell's
current density, so each nucleated INP made ``\rho_\mathrm{cell}/\rho``
crystals (up to 10⁻⁴ too many). INP are now rescaled to the cell
density around each nucleation call.

### Results

| INP | run | LWP start → end (g m⁻²) | IWP end (g m⁻²) | ice in liquid layer (L⁻¹) | glaciation (LWP < 10 %) |
| :-- | :-- | :-- | :-- | :-- | :-- |
| none | 8 h | 14.4 → 39.3 | 0 | 0 | never |
| ×1 (5 cm⁻³ dust) | 8 h | 14.4 → 35.4 | 0.36 | 0.27 → 0.48 | never |
| ×10 | 8 h | 14.4 → 14.7 | 2.28 | 2.6 → 4.8 | never |
| ×100 | 4 h | 14.4 → 0.0 | 0.35 | 27 → 30 | 2.2 h |
| ×300 | 2 h | 14.4 → 0.0 | 2.37 | 77 → 83 | 0.7 h |

With realistic INP the deck is mixed-phase and persistent; ice lowers
LWP monotonically with ice number, as in the ISDAC LES; a 100-fold
increase of INP glaciates it in about two hours, and the glaciation time
shortens monotonically as INP rise. The IWP of the base case
(0.36 g m⁻² at 0.48 L⁻¹) is, as far as the author recalls the
intercomparison, below the several g m⁻² the ISDAC LES produced at
1 L⁻¹ — the ice number here is half that, and bulk (single-moment-shape)
ice PSDs were found by Ovchinnikov et al. to underpredict IWP by a factor
of two — but the IWP comparison is not a verified match and the tests
do not claim one.

## References

* DeMott, P. J. et al. (2010), *Predicting global atmospheric ice nuclei
  distributions and their impacts on climate*, PNAS 107, 11217.
* DeMott, P. J. et al. (2015), *Integrating laboratory and field data to
  quantify the immersion freezing ice nucleation activity of mineral
  dust particles*, ACP 15, 393.
* Harrington, J. Y., Reisin, T., Cotton, W. R. & Kreidenweis, S. M.
  (1999), *Cloud resolving simulations of Arctic stratus. Part II:
  Transition-season clouds*, Atmos. Res. 51, 45.
* Harrington, J. Y. & Olsson, P. Q. (2001), *On the potential influence
  of ice nuclei on surface-forced marine stratocumulus cloud dynamics*,
  JGR 106, 27473.
* Hoose, C. & Möhler, O. (2012), *Heterogeneous ice nucleation on
  atmospheric aerosols: a review of results from laboratory
  experiments*, ACP 12, 9817.
* Kanji, Z. A. et al. (2017), *Overview of ice nucleating particles*,
  Meteorol. Monogr. 58, 1.1.
* Lilly, D. K. (1968), *Models of cloud-topped mixed layers under a
  strong inversion*, QJRMS 94, 292.
* Marinou, E. et al. (2019), *Retrieval of ice-nucleating particle
  concentrations from lidar observations and comparison with UAV in situ
  measurements*, ACP 19, 11315.
* Meyers, M. P., DeMott, P. J. & Cotton, W. R. (1992), *New primary
  ice-nucleation parameterizations in an explicit cloud model*, JAM 31,
  708.
* Morrison, H. & Milbrandt, J. A. (2015), *Parameterization of
  cloud microphysics based on the prediction of bulk ice particle
  properties. Part I*, JAS 72, 287.
* Morrison, H. et al. (2011), *Intercomparison of cloud model
  simulations of Arctic mixed-phase boundary layer clouds observed
  during SHEBA/FIRE-ACE*, JAMES 3, M06003.
* Morrison, H. et al. (2012), *Resilience of persistent Arctic
  mixed-phase clouds*, Nature Geosci. 5, 11.
* Niemand, M. et al. (2012), *A particle-surface-area-based
  parameterization of immersion freezing on desert dust particles*,
  JAS 69, 3077.
* Ovchinnikov, M. et al. (2014), *Intercomparison of large-eddy
  simulations of Arctic mixed-phase clouds: Importance of ice size
  distribution assumptions*, JAMES 6, 223.
* Seinfeld, J. H. & Pandis, S. N. (2016), *Atmospheric Chemistry and
  Physics*, 3rd ed., Wiley — ch. 9, gravitational settling.
* Solomon, A., Feingold, G. & Shupe, M. D. (2015), *The role of ice
  nuclei recycling in the maintenance of cloud ice in Arctic
  mixed-phase stratocumulus*, ACP 15, 10631.
* Tobo, Y. et al. (2013), *Biological aerosol particles as a key
  determinant of ice nuclei populations in a forest ecosystem*, JGR 118,
  10100.
* Ullrich, R. et al. (2017), *A new ice nucleation active site
  parameterization for desert dust and soot*, JAS 74, 699.
* Whitby, E. R. & McMurry, P. H. (1997), *Modal aerosol dynamics
  modeling*, Aerosol Sci. Technol. 27, 673.
