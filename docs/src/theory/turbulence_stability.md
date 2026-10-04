# Stability functions and the gradient Richardson number

Era VI (Phase 58, #170) puts every Richardson-number-dependent piece of
turbulence physics behind one contract, `TurbulenceStability`. Before
it the repository had two stability families that did not share code:
the Monin–Obukhov ``φ/ψ`` functions of `SurfaceExchange` and a private
`_fstab(Ri)` in the radiation-fog column. It also had two private
mean-profile ``N^2`` helpers (in `TimeIntegration` and `Anelastic`) and
no pointwise ``N^2`` or ``Ri`` field. The contract is pure and is
included before `Operators`, so the LES closure can call it.

## Local first-order closure

A local closure writes the eddy viscosity and diffusivity as

```math
K_m = l^2 |S|\, f_m(Ri), \qquad
K_h = \frac{l^2 |S|}{Pr_n}\, f_h(Ri), \qquad
Ri = \frac{N^2}{S^2}, \qquad N^2 = \frac{g}{θ_0}\frac{∂θ_v}{∂z},
```

with ``l`` a mixing length (``C_sΔ`` in an LES, ``κz`` near a wall),
``|S|`` the shear, and ``Pr_n`` the neutral turbulent Prandtl number. In
the contract both factors are normalised to ``f(0) = 1``, and ``Pr_n`` is
returned separately by `neutral_prandtl`, so different families can be
compared factor for factor. Every implementer has

* ``f(0) = 1``,
* ``f ≥ 0``,
* ``f`` non-increasing for ``Ri > 0``,
* ``f ≥ 1`` for ``Ri < 0``.

The test suite asserts all four on ``Ri ∈ [-5, 10] ∪ \{10^2, 10^4\}``
for every implementer.

## The families

| family | stable ``f_m`` | stable ``f_h`` | unstable ``f_m``, ``f_h`` | ``Pr_n`` | ``Ri_c`` |
| :-- | :-- | :-- | :-- | :-- | :-- |
| `LillyCutoff` | ``\sqrt{1 - Ri/Ri_c}`` | same | same expression | 1 | 0.25 |
| `BrownEtAl1994` | ``(1 - Ri/Ri_c)^4`` | ``(1 - Ri/Ri_c)^4 (1 - 1.2Ri)`` | ``\sqrt{1-16Ri}``, ``\sqrt{1-40Ri}`` | 0.7 | 0.25 |
| `ShortTail` | ``(1 + 5Ri)^{-2}`` | same | ``\sqrt{1 - 16Ri}`` | 1 | none |
| `LongTail` | ``[1 + 10Ri/\sqrt{1+5Ri}]^{-1}`` | ``[1 + 15Ri\sqrt{1+5Ri}]^{-1}`` | ``1 - 10Ri``, ``1 - 15Ri`` | 0.74 | none |
| `MOSTConsistent` | ``φ_m^{-2}`` | ``(φ_mφ_h)^{-1}`` | same | ``φ_h(0)/φ_m(0)`` | ``1/α`` (log-linear) or none (BDBH) |

* **Lilly (1962)** obtained his correction from the equilibrium TKE
  budget ``K_m S^2 - K_h N^2 = ε`` with ``ε ∝ K^3/l^4``, which gives
  ``K = l^2 (S^2 - N^2/Ri_c)^{1/2}``. Turbulence switches off at the
  critical Richardson number. Mason (1989) uses the same expression on
  the unstable side.
* **Brown, Derbyshire & Mason (1994)** are the functions of the Met
  Office LEM and of MONC, with the constants of that code.
* **ShortTail** is the radiation-fog column's function, moved into the
  contract unchanged (bit for bit). Its ``Ri^{-2}`` tail lets turbulence
  collapse in a strong inversion, which radiation fog needs.
* **Louis (1979)** has a long tail, ``f_m ∝ Ri^{-1/2}``, with no critical
  value. On the unstable side Louis's coefficient depends on the
  roughness length. Away from the surface no roughness length exists, so
  the free-atmosphere limit (coefficient → 0) is used.

## Consistency with Monin–Obukhov similarity

In the surface layer, similarity theory gives the gradients

```math
\frac{∂u}{∂z} = \frac{u_*}{κz}φ_m(ζ), \qquad
\frac{∂θ}{∂z} = \frac{θ_*}{κz}φ_h(ζ), \qquad
ζ = \frac{z}{L}, \qquad L = \frac{u_*^2 θ_0}{κ g θ_*}.
```

Dividing ``(g/θ_0)∂θ/∂z`` by ``(∂u/∂z)^2`` gives the gradient Richardson
number as a function of ``ζ``:

```math
Ri = \frac{g}{θ_0}\frac{θ_* φ_h/(κz)}{u_*^2 φ_m^2/(κz)^2}
   = \frac{κ z g θ_*}{θ_0 u_*^2}\,\frac{φ_h}{φ_m^2}
   = ζ\,\frac{φ_h(ζ)}{φ_m(ζ)^2}.
```

Next, ask what ``f_m`` and ``f_h`` make the local closure with ``l = κz``
reproduce the surface fluxes ``u_*^2`` and ``u_*θ_*``:

```math
K_m\frac{∂u}{∂z} = (κz)^2\Big(\frac{∂u}{∂z}\Big)^2 f_m
                 = u_*^2 φ_m^2 f_m = u_*^2
\;\Longrightarrow\; f_m = φ_m^{-2},
```

```math
K_h\frac{∂θ}{∂z} = (κz)^2 \frac{∂u}{∂z}\frac{∂θ}{∂z} f_h
                 = u_*θ_* φ_m φ_h f_h = u_*θ_*
\;\Longrightarrow\; f_h = (φ_m φ_h)^{-1}.
```

`MOSTConsistent(sf)` evaluates these at the ``ζ`` that solves
``Ri = ζφ_h/φ_m^2``. It finds that ``ζ`` with the Illinois solver of
`solve_surface_layer`, on ``[ζ_{\min}, 0]`` or ``[0, ζ_{\max}]``.

* **Log-linear functions,** ``φ = 1 + αζ``. Here
  ``Ri = ζ/(1+αζ) → 1/α`` as ``ζ → ∞``. That is a critical Richardson
  number: 0.2 for ``α = 5``, and ``Ri(ζ = 10^4) = 0.199996``. Beyond it
  no ``ζ`` exists and ``f = 0``.
* **Beljaars–Holtslag functions.** ``φ_h`` grows faster than ``φ_m``,
  so ``Ri`` is unbounded and there is no critical value:
  ``Ri(ζ = 100) = 8.07``.
* **Unstable side, Businger–Dyer with ``γ_m = γ_h``.**
  ``φ_h = φ_m^2``, so ``Ri = ζ`` exactly.

**Round trip (test 2).** The test builds the analytic profiles ``u(z)``,
``θ(z)`` for ``ζ ∈ [-2, 10]`` (49 values) with BDBH and with log-linear
``α = 5``. It differentiates them at 10 m with a fourth-order stencil
and forms ``K_m = (κz)^2 S f_m``. The result reproduces ``u_*^2/S`` to
a maximum relative error of ``5.4 × 10^{-11}``, and ``K_h θ_z``
reproduces ``u_*θ_*`` to ``2.0 × 10^{-10}``. The requirement is
``10^{-6}``.

## N² and Ri on the grid

`buoyancy_frequency_squared(θ_below, θ_above, Δz, g, θ_ref)` is the one
``N^2`` of a θ difference, ``(g/θ_{\rm ref})Δθ/Δz``. Without ``θ_{\rm ref}``
it is the logarithmic form ``gΔ\ln θ/Δz``, which is exact for the
constant-``N^2`` reference profile of the anelastic core. Three callers
use it:

* the pointwise field (`Operators.buoyancy_frequency_squared!`);
* the CFL buoyancy limit of `TimeIntegration`, applied to plane means;
* the reference ``N^2`` of `Anelastic`.

Both mean-profile results are bit for bit the same as before the
delegation (test 6).

**The pointwise fields** live at cell centres.

* ``N^2_k`` is the difference of the buoyancy that `add_buoyancy!`
  applies at the cell's two w faces. θ and ``q_v`` are interpolated to
  the faces as there, with ``θ_v = θ(1 + 0.61 q_v)``. On a uniform grid
  this is the centred difference ``(g/θ_0)(θ_{k+1} - θ_{k-1})/(2Δz)``.
* `gradient_richardson!` divides by ``|S|^2 = 2S_{ij}S_{ij}`` from the
  strain stencil of the Smagorinsky closure. For a pure vertical shear,
  ``|S|^2 = (∂u/∂z)^2``.
* ``S^2`` is floored at `RI_SHEAR2_FLOOR` = ``10^{-10}`` s⁻², a shear
  of ``10^{-5}`` s⁻¹, so ``Ri`` stays finite in shear-free cells.

**Measured accuracy (test 4).**

* Linear profiles ``u = Sz``, ``θ = θ_0 + Γz`` on a 16³ grid: ``Ri_g``
  equals ``gΓ/(θ_0 S^2)`` to a relative ``2.3 × 10^{-13}``.
* The shear layer ``u = U\tanh(z/δ)`` with constant ``N^2``: the
  discrete ``Ri`` converges to ``J\cosh^4(z/δ)`` with orders 1.92 and
  1.98 (``N_z`` = 32 → 64 → 128, over ``|z| ≤ 2δ``).

## In the Berlin case

The `turbulence_diag` addon writes the plane-mean ``N^2``, ``Ri_g`` and
``f_m`` profiles at every output record. See
[The Berlin case](../design/berlin_case.md#Turbulence-diagnostics-(#170)),
which also gives the measured values in the 60 m capping inversion of
the smoke case.

## References

* Brown, A. R., Derbyshire, S. H. & Mason, P. J. (1994). Large-eddy
  simulation of stable atmospheric boundary layers with a revised
  stochastic subgrid model. *Q. J. R. Meteorol. Soc.*, 120, 1485-1512.
* Businger, J. A., Wyngaard, J. C., Izumi, Y. & Bradley, E. F. (1971).
  Flux-profile relationships in the atmospheric surface layer.
  *J. Atmos. Sci.*, 28, 181-189.
* Beljaars, A. C. M. & Holtslag, A. A. M. (1991). Flux parameterization
  over land surfaces for atmospheric models. *J. Appl. Meteorol.*, 30,
  327-341.
* Lilly, D. K. (1962). On the numerical simulation of buoyant
  convection. *Tellus*, 14, 148-172.
* Louis, J.-F. (1979). A parametric model of vertical eddy fluxes in
  the atmosphere. *Boundary-Layer Meteorol.*, 17, 187-202.
* Mason, P. J. (1989). Large-eddy simulation of the convective
  atmospheric boundary layer. *J. Atmos. Sci.*, 46, 1492-1516.
