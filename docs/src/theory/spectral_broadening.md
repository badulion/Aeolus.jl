# Spectral broadening: bulk closure and validation

*Phase 45 (issue #14) — module `TurbulentMicrophysics`.*

The [previous page](sgs_activation_growth.md) lets every superdroplet grow
in its own supersaturation ``\bar S + \sigma_S\xi_i(t)``, where ``\xi_i``
is a unit Ornstein-Uhlenbeck (OU) process with correlation time ``\tau``.
A bulk scheme has no droplets to give a supersaturation to. This page
describes what it can carry instead: the moments of the spectrum those
droplets would have. It then checks the result against the superdroplet
reference and against the broadening law found in DNS.

## Moment equations

Diffusional growth is linear in ``w = r^2``:

```math
\frac{dw_i}{dt} = 2G\left(\bar S + \sigma_S\xi_i - \frac{a}{\sqrt{w_i}}\right),
```

where ``G = 1/(F_k + F_d)`` and ``a`` is the Kelvin coefficient. Both
are taken from Phase 41 (`fk_thermal`, `fd_vapor`, `kelvin_a`), so the
bulk scheme and the superdroplets share the same droplet physics. The
Raoult term is dropped. For an activated droplet with 10⁻¹⁸ kg of NaCl
it is ``5.2\times10^{-6}`` at 3 μm and ``1.1\times10^{-6}`` at 5 μm,
two to three orders of magnitude below ``\bar S``.

The bulk state ([`SpectralMoments`](@ref)) consists of the droplet number
``N``, the mean ``\mu = \langle w\rangle``, the variance
``V = \mathrm{Var}(w)``, and the covariance ``C = \mathrm{Cov}(w, \xi)``.
The covariance is needed because a droplet that has been in a moist
sub-volume is both larger and, for about one correlation time, still in
moist air. Averaging over droplets with ``\langle\xi\rangle = 0`` and
``\langle\xi^2\rangle = 1`` gives

```math
\begin{aligned}
\dot\mu &= 2G\left(\bar S - a\langle w^{-1/2}\rangle\right),\\
\dot V  &= 4G\sigma_S C - 4Ga\,\mathrm{Cov}(w, w^{-1/2}),\\
\dot C  &= 2G\sigma_S - C/\tau + Ga\,\mu^{-3/2}C .
\end{aligned}
```

``w`` itself carries no white noise, because ``\xi`` is continuous, so
there is no Itô term. Without the Kelvin term the three equations are
**exact and closed**. For an uncorrelated start they give
``V - V_0 = (2G)^2\,2\sigma_S^2\tau[t - \tau(1 - e^{-t/\tau})]``, which is
the analytic result of issue #13 (`supersaturation_integral_variance`).
The test reproduces it to ``10^{-9}``.

`bulk_condensation_step!` integrates ``C`` exactly over each sub-step,
since it is linear. Within a sub-step it holds ``\bar S``, ``T`` and ``p``
fixed, which is the same operator split the superdroplet step uses. It
sub-steps to ``0.2\,\tau_\mathrm{phase}`` and ``\tau/10``, and passes the
condensed mass ``\tfrac43\pi\rho_w N\Delta\langle r^3\rangle`` to Phase
41's `apply_condensate_tendency!`. Total water is therefore conserved to
round-off (measured ``|\Delta W/W| \le 3\times10^{-15}``).

## The one assumption: a gamma closure in ``r^2``

``\langle r^2\rangle = \mu`` is exact. Every other radius moment
(``\langle r\rangle``, ``\langle r^3\rangle``, ``\varepsilon``, and the
Kelvin moments ``\langle w^{-1/2}\rangle``) requires a shape. The scheme
assumes that ``w`` follows a gamma distribution with shape
``k = \mu^2/V``, so that
``\langle w^p\rangle = \mu^p\,\Gamma(k+p)/(\Gamma(k)k^p)``
([`gamma_moment_ratio`](@ref)). The ratio is evaluated in log form using
Stirling's series after shifting ``k \ge 16``. Its accuracy is
``10^{-14}``, including for narrow spectra where ``k \sim 10^6``. No
SpecialFunctions dependency is needed.

The gamma shape is right-skewed, with skewness ``2/\sqrt k``. This fits an
activated spectrum well: for a log-normal spectrum with
``\varepsilon = 0.1`` the closure gives a relative dispersion 0.75–1.2 %
too high. The fluctuation-generated part of the spectrum is different. It
is a sum of Gaussian increments, so it is **symmetric** in ``r^2``. When
that part dominates, the closure underestimates ``\varepsilon``, as shown
below.

## Validation case

A [`BroadeningCase`](@ref) is a fixed-mass parcel (`parcel_ascent!`
applies dry-adiabatic cooling and hydrostatic expansion, with latent heat
added by the condensation step). The case is run by
[`run_broadening_case`](@ref) in two ways from the **same initial
droplets**:

* **Superdroplet reference.** This is issue #13's `sgs_condensation_step!`,
  in which every superdroplet has its own OU supersaturation and growth is
  Phase 41's `condensation_growth!`.
* **Bulk scheme.** This is `bulk_condensation_step!` applied to
  [`spectral_moments`](@ref) of those droplets.

`parcel_ascent!` with no droplets produces ``dS/dz`` equal to Phase 42's
``\alpha(T, p)`` to within 0.12–0.24 %. The remaining difference is the
dry-air molar mass in ``\alpha``'s pressure term.

### Ascending parcel: bulk vs superdroplet

The parcel starts at cloud base with 100 cm⁻³ of 5 μm droplets
(log-normal, ``\varepsilon_0 = 0.1``). It rises at ``w = 0.5`` m s⁻¹
for 300 s with ``\tau = 5`` s, and the reference uses 8000 superdroplets.
The superdroplet columns are seed 1. Since #305 the superdroplet radii
and fluctuations come from counter-based streams (one per droplet), so the
realisations, and these numbers, changed once. The table and the seed
lists below were re-measured then. The ranges quoted for the monodisperse
and stationary variants are from the earlier streams.

| ``\sigma_S`` | ``\varepsilon`` SD | ``\varepsilon`` bulk | diff | ``\sigma_{r^2}`` SD / bulk (μm²) | ``\bar r`` diff |
| :-: | :-: | :-: | :-: | :-: | :-: |
| 0     | 0.0301 | 0.0305 | +1.0 % | 5.57 / 5.59   | −1.1e-5 |
| 0.1 % | 0.0642 | 0.0646 | +0.6 % | 11.66 / 11.82 | −9.8e-5 |
| 0.2 % | 0.1229 | 0.1194 | **−2.9 %** | 21.35 / 21.57 | +3.8e-4 |

Across seeds 1–3, the ``\varepsilon`` difference is within 1.6 % for
``\sigma_S \le 0.1\,\%``. That is inside the stated tolerance: three
standard errors of a sample standard deviation, ``3/\sqrt{2n_{sd}} = 2.4\,\%``,
plus the closure error of the initial spectrum. The width
``\sigma_{r^2}``, which depends on no closure, agrees within 1.6 % in
every case.

At ``\sigma_S = 0.2\,\%`` the bulk ``\varepsilon`` is **2.9–5.1 % low**
for every seed. This is a systematic error, not sampling. Here the
fluctuations have produced most of the variance, so the ``r^2``
distribution is nearly symmetric, and the gamma closure puts too little
weight in the small-``r`` tail. The error grows with the fluctuation
share. In the same ascending parcel with monodisperse droplets, frozen
sub-volumes (``\tau = \infty``) and 60 s, where ``\varepsilon \approx 0.15``,
the bulk value is 5–7 % low over four seeds. With stationary
fluctuations, ``\tau = 20`` s, 600 s and ``\varepsilon \approx 0.165``, it
is 7–9 % low. In all of these cases ``\sigma_{r^2}`` still agrees to
sampling error. The test suite keeps this bias as a known limitation (the
test asserts a 2–6 % deficit) and does not widen the tolerance to hide it.
The natural fix is to carry the third cumulant of ``r^2``, which the
linear fluctuation forcing leaves unchanged. That would let the closure
tell the symmetric fluctuation-generated part from the skewed activated
part.

Without fluctuations the spectrum narrows in both schemes, as in
Phase 41, from ``\varepsilon`` = 0.10 to 0.030. One exception: during
the first second ``\bar S`` is still below the Kelvin equilibrium
``a/r \approx 0.024\,\%`` of a 5 μm droplet, and ``\varepsilon`` rises by
0.06 % before narrowing starts.

### Stationary fluctuations: ``\sigma_{r^2} \propto t^{1/2}``

The DNS configuration has no mean updraught and statistically stationary
``S'``. Sardina et al. (2015) show analytically and in DNS that under
these conditions the width of the squared-radius distribution grows
without bound as ``t^{1/2}``. The Lanotte et al. (2009) DNS used the same
``\sigma_{r^2}`` diagnostic and found that the broadening increases with
the Reynolds number. That is, it increases with the scale of the
fluctuations, which corresponds to larger ``\sigma_S\sqrt\tau`` here. The
same OU closure gives
``\sigma_{r^2} \to 2G\sigma_S\sqrt{2\tau t}`` for ``t \gg \tau``. The
comparison is therefore the **exponent**. The DNS prefactors depend on
their Reynolds numbers and cannot be transferred directly.

The test case uses monodisperse 10 μm droplets, ``w = 0``,
``\sigma_S = 0.1\,\%``, ``\tau = 5`` s and 600 s, with the fit taken over
``t \ge 10\tau``:

| | exponent ``p`` | ``\sigma_{r^2}(600\,\mathrm{s})`` / OU |
| :-- | :-: | :-: |
| OU analytic, same window | 0.5142 | 1 |
| bulk, Kelvin off | 0.5142 | 1.0000 |
| superdroplet, Kelvin off | 0.5265 | — |
| bulk, full physics | 0.5276 | 1.0334 |
| superdroplet, full physics | 0.5401 (seeds 1–3: 0.540, 0.529, 0.526) | 1.039 |

The analytic OU curve has exponent 0.514 in this window rather than 0.5
because of its ``-\tau`` offset. The local exponent is
``\tfrac12\,t/(t-\tau)``, which tends to ``\tfrac12``. The Kelvin term adds
real broadening, because small droplets grow more slowly:
``d\ln\mathrm{Var}/dt = 2Ga\mu^{-3/2} \approx 2\times10^{-4}`` s⁻¹ at 10 μm.
This raises the width by 3.3 % and the exponent by 0.013, and the bulk
scheme and the superdroplets agree on both. In the frozen limit
(``\tau = \infty``) both schemes give ``p = 1.0009``. This is the
``t \ll \tau`` limit of the same OU result, ``\sigma_{r^2} = 2G\sigma_S t``,
in which each droplet keeps its supersaturation for the whole run (cf. the
stochastic model of Paoli & Shariff 2009).

Monodisperse droplets with ``\sigma_S = 0`` stay monodisperse in both
representations: ``\sigma_{r^2}`` is at round-off level
(``< 10^{-10}\mu``) and ``\varepsilon < 10^{-6}``.

## Worked example

The former `examples/validation/spectral_broadening.jl` (a parcel case,
(removed by ONE-g, #456)) ran both cases, printed the tables above and wrote two SVG figures:
``\varepsilon(t)`` in the ascending parcel, and ``\sigma_{r^2}(t)`` on
log-log axes against the OU curve and ``t^{1/2}``.

## Limitations

* The fluctuation is **prescribed**. It does not relax on the
  phase-relaxation time as the droplets consume it. The in-cloud caveat
  on the [previous page](sgs_activation_growth.md) applies to both
  schemes equally.
* The scheme handles activated droplets only, with no activation and no
  complete evaporation (`bulk_condensation_step!` throws if ``\mu \le 0``).
  Activation is `sgs_activation`.
* ``\varepsilon`` depends on the gamma closure, and its error grows with
  the share of fluctuation-generated variance (see above).
* There is no fall-speed spectrum or sub-grid precipitation fraction.
  Those are Phase 44 #8, deferred until P3 lands.

## References

* Grabowski, W. W. & Wang, L.-P. (2013). Growth of cloud droplets in a
  turbulent environment. *Annu. Rev. Fluid Mech.*, 45, 293-324.
* Lanotte, A. S., Seminara, A. & Toschi, F. (2009). Cloud droplet growth
  by condensation in homogeneous isotropic turbulence. *J. Atmos. Sci.*,
  66, 1685-1697.
* Paoli, R. & Shariff, K. (2009). Turbulent condensation of droplets:
  direct simulation and a stochastic model. *J. Atmos. Sci.*, 66, 723-740.
* Sardina, G., Picano, F., Brandt, L. & Caballero, R. (2015). Continuous
  growth of droplet size variance due to condensation in turbulent
  clouds. *Phys. Rev. Lett.*, 115, 184501.
