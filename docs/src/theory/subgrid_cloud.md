# Sub-grid cloud and supersaturation PDF

*Era IV, Phase 45 — module `TurbulentMicrophysics`.*

An LES cell of tens of metres is not uniformly saturated. Unresolved
fluctuations of total water ``q_t`` and liquid-ice potential temperature
``\theta_{li}`` spread it over a distribution of saturation states, so
the cell can be partly cloudy. The all-or-nothing saturation adjustment
of the cell mean (Phase 39, `saturation_adjustment`) cannot represent
this: it clouds the whole cell or none of it. Because the condensate
``\max(s, 0)`` is convex in the saturation state, Jensen's inequality
also says the mean-state adjustment *underestimates* the condensate of
a variable cell.

## The saturation deficit

Following Sommeria & Deardorff (1977) and Mellor (1977), define the
liquid-water temperature ``T_l`` — the temperature with all condensate
evaporated — and the extended condensate

```math
s = a\,\bigl(q_t - q_s(T_l)\bigr), \qquad
a = \frac{1}{1 + \Gamma\,\partial q_s/\partial T},
```

where ``\Gamma = \partial T/\partial q_c`` at fixed ``\theta_{li}``, ``p``
and ``q_t``. ``s > 0`` is the condensate a sub-volume holds after
adjustment; ``s < 0`` is its (scaled) saturation deficit.

``\Gamma`` is usually written ``L/c_p``. Here it is differentiated from
the contract's own definition
``\theta_{li} = (T/\Pi)\exp[-(L_v q_l + L_s q_i)/(c_{pm} T)]``, which
adds the change of the moist Exner exponent ``R_m/c_{pm}`` as vapour
becomes condensate:

```math
\Gamma = \frac{L}{c_{pm}} + T_l \ln\frac{p}{p_0}\,
         \frac{\partial (R_m/c_{pm})}{\partial q_c}.
```

At 900 hPa this is 1.7 % larger than ``L/c_{pm}``. With bare
``L/c_{pm}`` the linearised slope ``\partial s/\partial q_t`` at the
saturation boundary is 0.32569 against the exact adjustment's 0.32208;
with ``\Gamma`` it is 0.32208 (relative difference 8 × 10⁻⁷).

A fluctuation ``(q_t', \theta_{li}')`` gives
``s' = a_q q_t' - b\,\theta_{li}'`` with ``b = a\,(\partial q_s/\partial T)\,\Pi``
and ``a_q = a\,(1 - \partial q_s/\partial T\;\partial T_l/\partial q_t)``,
so

```math
\sigma_s^2 = a_q^2\,\overline{q_t'^2}
           - 2 a_q b\,\overline{q_t'\theta_{li}'}
           + b^2\,\overline{\theta_{li}'^2}.
```

### Only the fluctuations are linearised

The cell mean ``\bar s`` is anchored on the exact, nonlinear adjustment:
where `saturation_adjustment` condenses water, ``\bar s`` *is* that
condensate; elsewhere ``\bar s = a(q_t - q_s(T_l)) < 0``. Consequently
the ``\sigma_s \to 0`` limit of the scheme is the Phase 39 adjustment
bit-for-bit (not a linearisation of it), which the tests check over 15
warm, mixed-phase and cold states.

## Cloud fraction and condensate

With ``Q = \bar s/\sigma_s`` and a Gaussian PDF,

```math
C = \tfrac12\bigl[1 + \operatorname{erf}(Q/\sqrt2)\bigr], \qquad
\bar q_c = \sigma_s\Bigl[Q\,C + \tfrac{1}{\sqrt{2\pi}}e^{-Q^2/2}\Bigr].
```

Cumulus fields are positively skewed in ``s`` (Bechtold et al. 1995), so
a binormal family in the form of Larson, Golaz & Cotton (2002) is also
provided, with both components sharing ``\sigma_c = \sigma_s\sqrt{1-\gamma}``
and the width closed by ``\gamma^{3/2} = |Sk|/(|Sk| + c)`` (a choice made
here so that it reduces exactly to the Gaussian at zero skewness).

Condensate is split into liquid and ice by `liquid_fraction` at the
resulting mean temperature, which is recovered from ``\theta_{li}`` —
so ``q_t`` and ``\theta_{li}``, the conserved variables, are untouched.

## Variance from the SGS closure

The LES closes SGS fluxes with the Smagorinsky-Lilly eddy viscosity
``\nu_t = (C_s\Delta)^2|S|`` and a scalar diffusivity ``\nu_t/\mathrm{Pr}_t``
(see [Smagorinsky-Lilly SGS](sgs.md)). The SGS scalar-variance budget in
local equilibrium (Deardorff 1980; Bechtold et al. 1995; Chaboureau &
Bechtold 2002) balances gradient production against dissipation over
the SGS turnover time ``\Delta/\sqrt e``:

```math
2\,\frac{\nu_t}{\mathrm{Pr}_t}\,|\nabla\bar\phi|^2
  = C_\chi\,\frac{\sqrt e}{\Delta}\,\overline{\phi'^2}.
```

Writing ``\nu_t = C_k\Delta\sqrt e``, ``\nu_t`` cancels:

```math
\overline{\phi'^2} = c_\mathrm{var}\,\Delta^2\,|\nabla\bar\phi|^2,
\qquad c_\mathrm{var} = \frac{2C_k}{C_\chi\,\mathrm{Pr}_t},
```

and likewise ``\overline{\phi'\psi'} = c_\mathrm{var}\Delta^2\,\nabla\bar\phi\cdot\nabla\bar\psi``.
In equilibrium the variance therefore does not depend on ``C_s``; ``\nu_t``
only decides whether SGS turbulence is present (``\nu_t = 0`` ⇒ zero
variance ⇒ all-or-nothing adjustment). The default
``c_\mathrm{var} = 1/12`` is the variance of the resolved linear profile
across a cell of width ``\Delta``; `equilibrium_variance_coefficient`
gives the turbulent value for explicit ``(C_k, C_\chi, \mathrm{Pr}_t)``.

A known limitation of any gradient closure: a well-mixed layer has no
resolved gradient and hence no sub-grid variance, so cloud *base* in a
mixed layer stays all-or-nothing; the partial cloud appears at the
inversion, where the gradients are.

## Validation

| Check | Result |
|:--|:--|
| PDF normalisation, all families | ``|\int P - 1| \le 1.4\times10^{-14}`` |
| ``\sigma_s = 0`` vs `saturation_adjustment` (15 states) | identical (`==`) |
| ``\sigma_s = 10^{-12}`` vs `saturation_adjustment` | ``|\Delta T| \le 1.1\times10^{-10}`` K |
| Boundary slope ``\partial q_c/\partial q_t``, exact vs linearised | 0.322085 vs 0.322084 |
| Total-water residual / ``\theta_{li}`` residual | ``4\times10^{-19}`` / ``1\times10^{-13}`` (relative) |

Against direct quadrature of the *exact* adjustment over a bivariate
normal in ``(q_t, \theta_{li})`` at 900 hPa, 290 K:

| ``q_t/q_s`` | ``\sigma_{q_t}`` | ``\sigma_\theta`` | ``\rho`` | ``C`` exact | ``C`` PDF | ``\bar q_c`` rel. error |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 1.00 | 0.1 g/kg | 0 | 0 | 0.5000 | 0.5000 | +0.16 % |
| 1.00 | 0.3 g/kg | 0 | 0 | 0.5000 | 0.5000 | +0.48 % |
| 1.00 | 1.0 g/kg | 0 | 0 | 0.5000 | 0.5000 | +1.58 % |
| 1.00 | 0.3 g/kg | 0.2 K | +0.5 | 0.4986 | 0.5000 | +0.80 % |
| 0.97 | 0.3 g/kg | 0.2 K | −0.3 | 0.1477 | 0.1483 | +0.27 % |
| 1.03 | 0.5 g/kg | 0.3 K | 0 | 0.7625 | 0.7624 | +0.69 % |

The condensate error grows linearly with ``\sigma``, as expected of a
first-order linearisation of the fluctuations.

## References

- Sommeria, G. & Deardorff, J. W. (1977). *J. Atmos. Sci.*, 34, 344-355.
- Mellor, G. L. (1977). *J. Atmos. Sci.*, 34, 356-358.
- Deardorff, J. W. (1980). *Bound.-Layer Meteor.*, 18, 495-527.
- Bechtold, P., Cuijpers, J. W. M., Mascart, P. & Trouilhet, P. (1995).
  *J. Atmos. Sci.*, 52, 455-463.
- Chaboureau, J.-P. & Bechtold, P. (2002). *J. Atmos. Sci.*, 59, 2362-2372.
- Larson, V. E., Golaz, J.-C. & Cotton, W. R. (2002). *J. Atmos. Sci.*,
  59, 3519-3539.
