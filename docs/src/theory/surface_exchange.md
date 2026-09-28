# Surface exchange and Monin–Obukhov similarity

Every Era V surface model — soil, canopy, urban facet, snow, ice, ocean,
sub-grid tile — ends in the same question: given the skin state and the
air at the lowest model level, what are the momentum, heat and moisture
fluxes between them? This page develops the answer implemented once, in
`src/SurfaceExchange.jl`, which the surface models delegate to.

## 1. Similarity and sign convention

In the surface layer the fluxes are nearly constant with height and set
the scales

```math
u_*^2 = -\overline{u'w'},\qquad
\overline{w'\theta'} = -u_*\theta_*,\qquad
\overline{w'q'} = -u_* q_* .
```

Fluxes are **positive upward**: a surface warmer than the air has
``\theta_* < 0`` and a sensible heat flux ``H = -\rho c_p u_*\theta_* > 0``.
Net radiation ``R_n`` is positive into the surface and the substrate flux
``G`` positive downward, so the energy balance is
``R_n = H + LE + G``.

Monin–Obukhov theory (1954) states that the dimensionless gradients are
universal functions of ``\zeta = z/L``, with the Obukhov length

```math
L = \frac{u_*^2\,\theta_v}{\kappa\, g\, \theta_{v*}}
  = -\frac{u_*^3\,\theta_v}{\kappa\, g\,\overline{w'\theta_v'}},
\qquad \theta_{v*} = \theta_*(1+\varepsilon' q) + \varepsilon'\theta q_*,
```

``\varepsilon' = R_v/R_d - 1 \approx 0.608``. Buoyancy is carried by the
*virtual* potential temperature, so evaporation alone can destabilise
the surface layer. ``\zeta > 0`` is stable, ``\zeta < 0`` unstable.

```math
\phi_m(\zeta) = \frac{\kappa z}{u_*}\frac{\partial u}{\partial z},\qquad
\phi_h(\zeta) = \frac{\kappa z}{\theta_*}\frac{\partial \theta}{\partial z}.
```

Integrating from the roughness length ``z_0`` to ``z`` with
``\psi(\zeta) = \int_0^\zeta [1-\phi(x)]/x\,dx`` gives the profiles

```math
U(z) = \frac{u_*}{\kappa}\Big[\ln\frac{z}{z_{0m}} - \psi_m\!\Big(\frac{z}{L}\Big) + \psi_m\!\Big(\frac{z_{0m}}{L}\Big)\Big]
\equiv \frac{u_*}{\kappa}\,\Phi_m ,
```

and likewise ``\theta_a - \theta_s = (\theta_*/\kappa)\,\Phi_h`` with
``\psi_h`` and ``z_{0h}``, ``q_a - q_s = (q_*/\kappa)\,\Phi_q`` with
``z_{0q}``. Separate roughness lengths for momentum, heat and moisture
matter: over land ``z_{0h}`` is typically an order of magnitude below
``z_{0m}``.

## 2. Stability functions

The default family, `BusingerDyerBeljaarsHoltslag`, joins two standard
forms at ``\zeta = 0``.

**Unstable** — Businger et al. (1971) as recast by Dyer (1974) for
``\kappa = 0.40``:

```math
\phi_m = (1-16\zeta)^{-1/4},\qquad \phi_h = (1-16\zeta)^{-1/2},
```

integrated exactly by Paulson (1970), with ``x = (1-16\zeta)^{1/4}``:

```math
\psi_m = 2\ln\frac{1+x}{2} + \ln\frac{1+x^2}{2} - 2\arctan x + \frac{\pi}{2},\qquad
\psi_h = 2\ln\frac{1+x^2}{2}.
```

**Stable** — Beljaars & Holtslag (1991), with ``a = 1``, ``b = 2/3``,
``c = 5``, ``d = 0.35``:

```math
\psi_m = -\Big[a\zeta + b\Big(\zeta - \frac{c}{d}\Big)e^{-d\zeta} + \frac{bc}{d}\Big],\qquad
\psi_h = -\Big[\Big(1 + \frac{2a\zeta}{3}\Big)^{3/2} + b\Big(\zeta - \frac{c}{d}\Big)e^{-d\zeta} + \frac{bc}{d} - 1\Big].
```

Unlike the log-linear ``\phi = 1 + 5\zeta``, these do not produce a
critical Richardson number: ``\phi_m`` grows only linearly for large
``\zeta``, so a very stable surface layer keeps a small but finite
exchange instead of decoupling. The corresponding ``\phi`` follow from
``\phi = 1 - \zeta\,d\psi/d\zeta``.

| ``\zeta`` | ``\psi_m`` | ``\psi_h`` | source |
|:--|:--|:--|:--|
| −1 | 1.1162 | 1.8812 | Paulson (1970), γ = 16 |
| −1/2 | ``2\ln\frac{1+\sqrt3}{2} + \ln 2 - \frac{2\pi}{3} + \frac{\pi}{2}`` | ``2\ln 2`` | exact |
| +1 | −4.2823 | −4.4339 | Beljaars & Holtslag (1991) |

The tests check these values, ``\psi(0) = 0``, ``\phi(0) = 1``,
continuity across ``\zeta = 0``, and ``d\psi/d\zeta = (1-\phi)/\zeta``
numerically (max error 1.3 × 10⁻¹¹ over ``-5 \le \zeta \le 10``).

Other families (e.g. the COARE convective form of Grachev & Fairall)
subtype `AbstractStabilityFunctions` and add methods to `psi_m`,
`psi_h`; the solver needs nothing else.

## 3. Solving for the Obukhov length

Given ``U``, ``\theta_a``, ``q_a`` at the forcing level and ``\theta_s``,
``q_s`` at the skin, ``u_*``, ``\theta_*``, ``q_*`` are explicit functions
of ``\zeta``; substituting them into the definition of ``L`` leaves one
scalar equation

```math
F(\zeta) = \zeta - \frac{z_m\,\kappa\, g\,\theta_{v*}(\zeta)}{u_*(\zeta)^2\,\theta_{va}} = 0 .
```

`solve_surface_layer` brackets the root in ``[\zeta_{min}, 0]``
(unstable) or ``[0, \zeta_{max}]`` (stable) — the sign of ``F(0)`` tells
which — and applies the Illinois variant of regula falsi. The textbook
fixed-point iteration ``L_{n+1} = L(u_*(L_n), \theta_*(L_n))`` is not
used: it oscillates or diverges in strongly stable conditions. A root
beyond the bracket is clamped to its end and reported with
`converged = false`. For the dry case with ``z_m = z_h`` the equation is
equivalent to the classical

```math
Ri_B = \frac{g z (\theta_{va} - \theta_{vs})}{\theta_{va} U^2}
     = \zeta\,\frac{\Phi_h}{\Phi_m^2},
```

which the tests verify to 8 × 10⁻¹⁶.

**Calm winds.** As ``U \to 0`` over a heated surface ``u_* \to 0`` and
``L \to 0``; physically, convective eddies keep mixing. The solver uses
``U_e = \max(\sqrt{U^2 + w_g^2}, U_{min})`` with an optional gustiness
``w_g`` (Beljaars 1995 suggests ``w_g = 1.2\,w_*``) and a floor
``U_{min} = 0.1`` m s⁻¹.

**Transfer coefficients.** Referred to ``U_e``,

```math
C_D = \frac{\kappa^2}{\Phi_m^2},\quad C_H = \frac{\kappa^2}{\Phi_m\Phi_h},\quad
C_E = \frac{\kappa^2}{\Phi_m\Phi_q},
```

so ``\tau = \rho C_D U_e^2``, ``H = \rho c_p C_H U_e(\theta_s - \theta_a)``,
``E = \rho C_E U_e (q_s - q_a)``. In the neutral limit
``C_D = [\kappa/\ln(z/z_0)]^2``, which the solver reproduces to
2.7 × 10⁻¹⁶.

## 4. From a surface model to fluxes

`turbulent_fluxes(s, forcing)` works for any `AbstractSurface`. The air
temperature is referred to the surface along the dry adiabat,
``\theta_a = T_a + \Gamma_d z_{ref}``, ``\theta_s = T_s``; density and
``c_{pm}`` come from `MoistThermodynamics`. The skin humidity uses the
"β" moisture availability,

```math
q_s = q_a + \beta\,[q_{sat}(T_s, p) - q_a]
\;\Longrightarrow\; E = \beta\,\rho\, C_E U_e\,[q_{sat}(T_s) - q_a],
```

so a dry surface (``\beta = 0``, the default) has zero moisture flux —
setting ``q_s = 0`` instead would turn it into a vapour sink — and open
water has ``\beta = 1``. Models with stomatal or soil resistances, dew
physics or ice skins override `surface_specific_humidity` and
`surface_latent_heat`.

## References

- Beljaars, A. C. M. (1995). The parametrization of surface fluxes in
  large-scale models under free convection. *Q. J. R. Meteorol. Soc.*,
  121, 255–270.
- Beljaars, A. C. M. & Holtslag, A. A. M. (1991). Flux parameterization
  over land surfaces for atmospheric models. *J. Appl. Meteorol.*, 30,
  327–341.
- Businger, J. A., Wyngaard, J. C., Izumi, Y. & Bradley, E. F. (1971).
  Flux-profile relationships in the atmospheric surface layer.
  *J. Atmos. Sci.*, 28, 181–189.
- Dyer, A. J. (1974). A review of flux-profile relationships.
  *Boundary-Layer Meteorol.*, 7, 363–372.
- Monin, A. S. & Obukhov, A. M. (1954). Basic laws of turbulent mixing
  in the surface layer of the atmosphere. *Tr. Akad. Nauk SSSR Geophiz.
  Inst.*, 24, 163–187.
- Paulson, C. A. (1970). The mathematical representation of wind speed
  and temperature profiles in the unstable atmospheric surface layer.
  *J. Appl. Meteorol.*, 9, 857–861.
