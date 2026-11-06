# Sub-grid supersaturation in activation and droplet growth

*Phase 45 (issue #13) — module `TurbulentMicrophysics`.*

The [sub-grid cloud PDF](subgrid_cloud.md) says how a cell's saturation
state is distributed. This page covers what that distribution does to
the droplets: which aerosol particles activate (Phase 42,
`AerosolCloud`), and how the droplets then grow by condensation
(Phase 41, `Superdroplet`). Both are **delegated**. The CCN spectrum is
`AerosolCloud.ccn_fraction`, and growth is `Superdroplet.grow_droplet`
through `condensation_growth!`. This module only supplies the
distribution of supersaturation that they see.

## Activation: the PDF-mean of the CCN spectrum

Phase 42 (Abdul-Razzak & Ghan 2000) computes one peak supersaturation
``S_\max`` per cell from the mean updraught and activates every particle
whose critical supersaturation is below it. For a log-normal mode with
median critical supersaturation ``S_m`` that fraction is

```math
f_\mathrm{CCN}(S) = \tfrac12\,\mathrm{erfc}\!\left(
   \frac{2\ln(S_m/S)}{3\sqrt2\,\ln\sigma_g}\right),
```

which is `ccn_fraction`. `arg_activation` evaluates this same function
at ``S_\max``.

With unresolved turbulence, sub-volumes reach ``S_\max + S'``, with
``S'`` distributed with standard deviation ``\sigma_S`` (the
[`SubgridCloud`](@ref)`.σ_S` of issue #12, or given directly) and any of
the issue #12 shapes. Each sub-volume activates its own particles, so

```math
\langle f\rangle = \int f_\mathrm{CCN}(S)\,P(S)\,dS
                 = \int \varphi(u)\,P\bigl(S > s_c(u)\bigr)\,du,
```

where ``u = \ln(r_d/r_g)/\ln\sigma_g`` is the standardised dry radius and
``s_c(u) = S_m e^{-3u\ln\sigma_g/2}``. The two forms are the same 2-D
Gaussian probability integrated in opposite orders. `sgs_ccn_fraction`
uses whichever has the smoother integrand: the radius form when the
activation threshold is wider than one standard deviation in ``u``,
``\sigma_c/(1.5\ln\sigma_g\max(\bar S,\sigma_c)) \ge 1``, and the
supersaturation form otherwise. With 2001 trapezoid nodes the result
agrees with an independent Simpson integral to ``1.6\times10^{-11}``.

### Jensen: the sign follows the curvature

``f_\mathrm{CCN}`` is a sigmoid in ``\ln S``. In ``S`` it is convex on the
steep low-``S`` flank, where few particles have activated, and concave
once most of the mode has. Expanding about the mean,

```math
\langle N\rangle - N(\bar S) = \tfrac12 N''(\bar S)\,\sigma_S^2 + O(\sigma_S^4),
```

so sub-grid fluctuations **add** droplets where the spectrum is convex
and **remove** them where it is concave. For the two-mode test population
(300 cm⁻³ at 40 nm and 50 cm⁻³ at 150 nm, ``\kappa = 0.6``, 283 K), the
measured ratio ``\Delta N/(\tfrac12N''\sigma^2)`` at ``\sigma = 0.05\bar S``
is 0.999, 1.013, 1.003, 1.003, 0.999, 1.002 and 1.007 for
``\bar S`` = 0.01, 0.02, 0.05, 0.1, 0.2, 0.4 and 0.8 %. The first two
values are in the convex regime and the rest are concave. At
``w = 0.5`` m s⁻¹ (``S_\max = 0.21\,\%``, concave) ``N_d`` falls from
182.3 to 181.5, 179.1 and 171.4 cm⁻³ for ``\sigma_S`` = 0.025, 0.05 and
0.1 %.

``\sigma_S = 0`` returns `arg_activation` bit for bit.

**Limitations.** The fluctuation is superimposed on ``S_\max``. It does
not feed back on the competition for vapour that sets ``S_\max`` (ARG's
``\eta``), and with ``w \le 0`` nothing activates, as in Phase 42. The
common alternative integrates over a sub-grid *updraught* PDF
(Ghan et al. 1997; Morrison & Gettelman 2008). This scheme fluctuates
the supersaturation directly, as diagnosed from the scalar variances.

## Growth: a random walk in ``r^2``

Diffusional growth obeys ``r\,dr/dt = G(S - a/r + b/r^3)``. At a single
``S`` every droplet gains the same ``r^2``, so the radius spectrum
**narrows**. This is the Phase 41 result, and the reason condensation
alone cannot make rain. In a turbulent cloud each droplet takes its own
path through the supersaturation field and gains ``2G\int S\,dt`` along
it (Cooper 1989). Here droplet ``i`` sees ``\bar S + \sigma_S\xi_i(t)``,
where ``\xi_i`` is a unit Ornstein-Uhlenbeck process with correlation
time ``\tau``, updated exactly, with ``\tau = \infty`` freezing each
droplet in its sub-volume. For an initially monodisperse population
this gives

```math
\mathrm{Var}(r^2) = (2G)^2\,2\sigma_S^2\tau\left[t - \tau\left(1 - e^{-t/\tau}\right)\right]
```

(`supersaturation_integral_variance`; Paoli & Shariff 2009; Sardina et
al. 2015). The width of this variance grows as ``\sigma_S t`` while
``t \ll \tau``, and as ``\sqrt{2\sigma_S^2\tau t}`` without bound for
``t \gg \tau``.

Measured with 4000 superdroplets (sampling error 2.2 %), the ratio of
``\mathrm{Var}(r^2)`` to the analytic value is **0.991** (τ = 5 s) and
**0.992** (τ = ∞).

`sgs_condensation_step!` is Phase 41's two-way-coupled
`condensation_step!` with a per-droplet ``S``. Vapour removal and latent
heating go through `apply_condensate_tendency!` using the total condensed
mass, so the cell's total water is conserved to round-off for any
``\sigma_S``. The fluctuations only decide which droplets receive the
condensate. The step is sub-stepped to at most ``\tau/10``, which puts
the error in ``\mathrm{Var}\int S\,dt`` at ``(h/\tau)^2/12 \approx 0.08\,\%``.
With ``\sigma_S = 0`` it is `condensation_step!` bit for bit.
Stepping a grid cell by cell (`index = k`) advances only that cell's
droplets' fluctuations, so each ``\xi_i`` ages once per time step
however many cells there are; `fl.σ_S` is one value and should be set to
each cell's ``\sigma_S`` before its call.

Starting from a 5 μm population with ``\varepsilon_0 = 0.115`` in a
cell at 12 % supersaturation (1000 superdroplets, τ = 10 s, 25 s), the
relative dispersion ``\varepsilon = \sigma_r/\bar r`` depends on
``\sigma_S`` as follows:

| ``\sigma_S`` | 0 | 0.05 % | 0.1 % | 0.2 % | 0.4 % |
| :-- | :-: | :-: | :-: | :-: | :-: |
| ``\varepsilon`` | 0.0232 | 0.0240 | 0.0267 | 0.0358 | 0.0603 |

In every run ``|\Delta W/W| \le 5.4\times10^{-16}``, and the mean radius
differs by less than 0.5 % between runs. After 100 s the mean
supersaturation has been consumed. The ``\sigma_S = 0`` spectrum has
stopped narrowing, with ``\varepsilon`` = 0.0233, while the
``\sigma_S = 0.2\,\%`` spectrum has broadened to 0.0714.

### Entrainment

At a cloud edge the entrained air is warm **and** dry, so
``\overline{q_t'\theta_{li}'} < 0``. In the Mellor mapping
``\sigma_s^2 = a_q^2\sigma_{q}^2 - 2a_qb\,\overline{q_t'\theta_{li}'} + b^2\sigma_\theta^2``
the covariance term then *adds* to the variance. Surface thermals are
moist and warm, and partly cancel it. With ``\sigma_q = 3\times10^{-5}``
and ``\sigma_\theta = 0.03`` K the correlations ``\rho = -0.8, 0, +0.8``
give ``\sigma_S = 0.63, 0.50, 0.32\,\%`` and ``\varepsilon`` = 0.092,
0.073, 0.049. Entrainment broadens the spectrum the most, and total
water is conserved in each case.

**Caveat on ``\sigma_S`` inside cloud.** `SubgridCloud.σ_S` is the
*clear-air* spread, ``\sigma_s/(a\,q_{sl})``. Inside a cloud, droplets
absorb supersaturation fluctuations on the phase-relaxation time
``\tau_\mathrm{phase}`` (a few seconds), so the sustained in-cloud
``S'`` is smaller, and its correlation time is at most
``\min(\tau_\mathrm{eddy}, \tau_\mathrm{phase})``. The clear-air value is
appropriate where droplets meet freshly entrained, unmixed air. Deeper
inside a cloud it is an upper bound.

## A bug this found

Before the Phase 41 fix in this change, a pure-water droplet (no solute)
in **any** subsaturation evaporated to the 10⁻¹² m radius floor in a
single step: a 10 μm droplet at ``S = -0.1\,\%`` over 0.5 s came out at
1e-12 m instead of 9.995 μm. Without a Raoult term, the Kelvin term makes
the backward-Euler residual positive at the floor, and that alone was
taken to mean "no root". In the stochastic benchmark droplets briefly
see ``S < 0``, and ``\mathrm{Var}(r^2)`` came out 61 times the analytic
value. The code was wrong, not the expectation. It now brackets the
root from the residual's minimum, and a regression test is in
`superdroplet.jl`.

## References

* Abdul-Razzak, H. & Ghan, S. J. (2000). *J. Geophys. Res.*, 105, 6837-6844.
* Cooper, W. A. (1989). Effects of variable droplet growth histories on
  droplet size distributions. Part I: Theory. *J. Atmos. Sci.*, 46, 1301-1311.
* Ghan, S. J., Leung, L. R., Easter, R. C. & Abdul-Razzak, H. (1997).
  *J. Geophys. Res.*, 102, 21777-21794.
* Morrison, H. & Gettelman, A. (2008). *J. Climate*, 21, 3642-3659.
* Paoli, R. & Shariff, K. (2009). Turbulent condensation of droplets:
  direct simulation and a stochastic model. *J. Atmos. Sci.*, 66, 723-740.
* Sardina, G., Picano, F., Brandt, L. & Caballero, R. (2015). Continuous
  growth of droplet size variance due to condensation in turbulent
  clouds. *Phys. Rev. Lett.*, 115, 184501.
