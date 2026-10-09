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
* **Cloudy air** (#239). Given a condensate field ``q_c`` (cloud or fog
  water) and the pressure, the buoyancy variable is
  ``θ_ρ = θ(1 + 0.61 q_v - q_c)`` (condensate loading) and, in a saturated
  cell (``q_c > 0``), ``N^2`` is that of a saturated parcel lifted on the
  moist adiabat, `MoistThermodynamics.saturated_buoyancy_frequency_squared`
  (see below).
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

## Saturated air

Without the condensate, a moist-adiabatic cloud or fog layer — neutral for
the saturated parcels that make up its turbulence — has the ``θ_v``
gradient of a stable layer: about ``1.4 × 10^{-4}`` s⁻² for the 1 km
column of test 7, enough for `StabilityCorrected` (Brown et al.) to shut
the vertical mixing off at a shear of 0.02 s⁻¹ (``Ri = 0.36``) and for the
Deardorff length scale ``0.76\sqrt e/N`` to fall to 6 m at
``e = 0.01`` m² s⁻². Deardorff (1980) formulated the closure for
stratocumulus with the saturated buoyancy.

The saturated ``N^2`` linearises the ``θ_ρ`` of a parcel displaced on the
(reversible) moist adiabat against the environment:

```math
N^2 = \frac{g}{θ_0}\Bigl[(1 + 0.61 q_v - q_c)
      \Bigl(\frac{∂θ}{∂z} - \frac{Γ_d - Γ_m}{Π}\Bigr)
      + θ\Bigl(1.61\Bigl(\frac{∂q_v}{∂z} - \frac{dq_s}{dz}\Big|_{\rm ad}\Bigr)
      - \frac{∂q_w}{∂z}\Bigr)\Bigr].
```

``(Γ_d - Γ_m)/Π`` is the parcel's θ gradient (``T = θΠ``), so the pressure
enters only pointwise; θ, ``q_v`` and ``q_w = q_v + q_c`` are differenced
between the faces as for the dry ``N^2``. In a saturated environment the
expression is the saturated Brunt–Väisälä frequency of Durran & Klemp
(1982, eq. 36) to ``O(q)``. At a cloud edge it differs on purpose:
Durran & Klemp assume a saturated environment and so drop the virtual
effect of the humidity jump. A first version used their form. In the
seeded Berlin smoke state it gave the fog-top level
``⟨N^2⟩ = +3.1 × 10^{-3}`` s⁻², while ``θ_v`` gives
``-2.2 × 10^{-3}``. A saturated parcel lifted into drier air of the same θ
is buoyant: its vapour outweighs its water.

**Measured (test 7).**

* A saturated moist-adiabatic column with constant total water gives
  ``|N^2| = 1.3 × 10^{-9}``, ``3.1 × 10^{-10}`` and ``7.8 × 10^{-11}`` s⁻² at
  ``Δz`` = 50, 25 and 12.5 m (second order). The ``θ_v`` form gives
  ``1.4 × 10^{-4}``.
* In a saturated environment the result matches eq. 36 of Durran & Klemp,
  written out by hand, to 0.9 %. It matches a parcel lifted 1 m by
  ``8.5 × 10^{-9}`` s⁻², including at a cloud top.
* It is within 1.4 %, 3.1 % and 1.3 % of the fog column's parcel ``N^2``
  (`RadiationFog.moist_stability`, an enthalpy-conserving finite
  displacement) on saturated layers of 8, 5 and 3 K km⁻¹.
* Without condensate it reproduces the dry and the ``q_v`` forms bit for
  bit.

Rain is not loaded into the closures' ``N^2`` in the Berlin case: there
``q_c`` is the cloud water ``q_l``, which also marks a cell as saturated.

## Maximum sustainable heat flux (P58-e)

Monin–Obukhov similarity, read with the **wind held fixed**, limits how
much heat a stable surface layer can carry downward (Van de Wiel et al.
2012). With the wind ``U`` at height ``z`` over roughness ``z_{0m}``,

```math
u_* = \frac{κU}{Φ_m(ζ)}, \qquad
Φ_m(ζ) = \ln\frac{z}{z_{0m}} - ψ_m(ζ) + ψ_m\!\left(ζ\frac{z_{0m}}{z}\right),
\qquad ζ = \frac{z}{L} = \frac{zκgQ}{u_*^3θ_{ref}},
```

for the downward kinematic buoyancy flux ``Q = -\overline{w'θ_v'} > 0``.
Eliminating ``u_*`` gives the flux as a function of the stability alone:

```math
Q(ζ) = \frac{κ^2U^3θ_{ref}}{g z}\,\frac{ζ}{Φ_m(ζ)^3}.
```

``Q = 0`` at neutral (``ζ = 0``). As ``ζ`` grows, ``Q`` rises, because a
stronger temperature contrast is mixed. ``Q`` then falls, because the
stratification suppresses ``u_*`` faster than the contrast grows. The
maximum is the **maximum sustainable heat flux** (MSHF). A surface that
must lose more heat than this by turbulence, for example a skin under
strong radiative cooling with little heat from below, has no turbulent
equilibrium at that wind. Its surface layer collapses.

**Log-linear closed form.** For ``φ_m = 1 + αζ`` the integrated function
is ``ψ_m = -αζ``, so ``Φ_m = Λ + α(z - z_{0m})ζ/z`` with ``Λ = \ln(z/z_{0m})``.
In terms of ``u_*`` (use ``κU/u_* = Φ_m``):

```math
Q(u_*) = \frac{θ_{ref}}{α g (z - z_{0m}) κ}\left(κU\,u_*^2 - Λ\,u_*^3\right).
```

This is a cubic in ``u_*``. Setting ``dQ/du_* = 2κUu_* - 3Λu_*^2 = 0``
gives the maximum at

```math
u_* = \frac{2κU}{3Λ}, \qquad
Q_{max} = \frac{4}{27}\,\frac{κ^2U^3θ_{ref}}{α\,g\,(z - z_{0m})\,Λ^2},
\qquad ζ = \frac{Λz}{2α(z - z_{0m})}.
```

At the maximum ``Φ_m = 3Λ/2``. For ``z_{0m} ≪ z`` this is the form of
Van de Wiel et al. (2012), with ``z`` in place of ``z - z_{0m}``. Here
``ψ_m(z_{0m}/L)`` is kept exactly. Note that ``Q_{max} ∝ U^3``: read the
other way round, ``U_{min} = (Q/Q_{max}(1))^{1/3}`` is the *minimum wind
for sustainable turbulence* of the paper's title.

**Other families.** `max_sustainable_heat_flux` maximises
``ζ/Φ_m(ζ)^3`` numerically. A 48-point scan in ``\ln ζ`` over
``[10^{-6}, 10^3]`` brackets the maximum, then golden-section search
narrows it to ``|Δ\ln ζ| ≤ 10^{-12}``. This takes about 120 evaluations
of ``ψ_m``, about 3 µs. For BDBH (Beljaars–Holtslag stable branch) the
curve is unimodal with an interior maximum. Its MSHF lies between those
of log-linear ``α = 5`` and ``α = 1``, because ``φ_m`` starts like
``1 + 5ζ`` and tends to ``1 + ζ``.

The heat roughness ``z_{0h}`` does not enter. With the wind held fixed,
the flux–stability relation is the momentum half of MOST. Every
`solve_surface_layer` solution with effective wind ``U`` lies on the
curve, so its buoyancy flux ``u_*θ_{v*}`` never exceeds ``Q_{max}``.

**Measured (`test/era6_collapse.jl`).**

* *Log-linear ``α = 5``.* The numerical maximum agrees with the closed
  form to ``5 × 10^{-16}`` (requirement ``10^{-8}``) at four
  ``(U, z, z_{0m})``. The numerical argmax ``u_*`` matches ``2κU/(3Λ)``
  to ``7 × 10^{-9}``.
* *Solver consistency.* Over ``ΔT ∈ [0, 20]`` K, ``U ∈ [0.5, 10]`` m/s
  (41 × 39 points, ``z = 10`` m, ``z_{0m} = 0.1`` m), the largest ratio
  ``u_*θ_{v*}/Q_{max}`` is 0.99999976 for BDBH and 0.9999995 for
  log-linear. The surface flux, as a function of ``ΔT``, comes up to the
  bound and stops there.

**Flux-prescribed wall law.** `flux_friction_velocity` solves
``u_* = κU/Φ_m(ζ)`` for a *prescribed* flux. It takes the weakly stable
root between neutral and the MSHF point, which is the root continuously
connected to the neutral state. When ``Q > Q_{max}`` it returns 0.
`WallModels.MOSTFluxWallLaw` is this relation as a wall law. It is the
wall of `half_channel_case(surface_heat_flux = Q)`, and it exerts **no
stress** in a column whose wind is too weak for the flux.

### The collapse regime in LES

The flux-cooled half channel is built from three pieces:

* the P54-d case, 16 × 16 × 32 cells on ``2πH × 2πH × H``, with
  ``z_0/H = 10^{-4}`` and ``-∂p/∂x = u_*^2/H``;
* the wall law above;
* `StabilityCorrected(StaticSmagorinsky(0.1); length_scales = :isotropic)`
  (P55-e).

The anisotropic default length scales give ``Δ_h = 12.6Δz`` on this grid.
With them even the neutral run relaminarises (resolved TKE falls from
0.18 to 0.0035 ``u_*^2`` over 20 ``H/u_*``), so the collapse runs use the
isotropic scales.

Each run starts from a neutral run spun up for 10 ``H/u_*``, and the
cooling is switched on at that point. The case has **two MSHFs**, both
taken for the neutral state:

* ``Q_{wall}`` at the wall model's height ``Δz/2 = H/64``, where
  ``U_0 = 12.63\,u_*``;
* ``Q_{bulk}`` at ``z = H``, the channel-scale criterion of Donda et al.
  (2015), who normalise by the half height.

Their ratio is ``Q_{bulk}/Q_{wall} = 0.032``. The table is
`examples/sbl/collapse_channel.jl` over the runs of
`configs/collapse_channel_sweep.toml`, averaged over 15–20 ``H/u_*``, with
γ at 20 ``H/u_*``:

| ``H/H_{max}^{wall}`` | ``H/H_{max}^{bulk}`` | TKE/TKE₀ | TKE/TKE₀ lowest third | levels with γ < 0.2 (of 11) | ``U_1/U_0`` | ``τ_w/u_*^2`` |
| --: | --: | --: | --: | --: | --: | --: |
| 0 | 0 | 1 | 1 | 0 | 0.993 | 1.02 |
| 0.0159 | 0.5 | 0.830 | 0.782 | 0 | 0.982 | 0.99 |
| 0.0636 | 2 | 0.292 | 0.207 | 5 | 0.841 | 0.69 |
| 0.25 | 7.9 | 0.123 | 0.062 | 11 | 0.859 | 0.64 |
| 0.5 | 15.7 | 0.114 | 0.071 | 11 | 0.916 | 0.63 |
| 1 | 31.4 | 0.055 | 0.020 | 11 | 1.027 | 0.60 |
| 2 | 62.9 | 0.136 | 0.040 | 11 | 1.254 | 0.50 |

TKE₀ = 2.98 ``u_*^2``. The table shows three regimes.

* **Below ``Q_{bulk}``** the channel stays turbulent. At 0.5 ``H_{max}^{bulk}``
  the TKE is 83 % of neutral.
* **Between ``Q_{bulk}`` and ``Q_{wall}``** the lowest third is laminar
  (γ < 0.2 on all 11 levels from 8 ``H_{max}^{bulk}``), but the wall
  model still exerts stress. The TKE is intermittent: global bursts
  reach about 2× the neutral TKE (at 8 ``H_{max}^{bulk}``, 6.9 ``u_*^2``
  near 22 ``H/u_*`` in a 40 ``H/u_*`` run) and decay again.
  This is the transient collapse of Donda et al. (2015).
* **Above ``Q_{wall}``** the wall layer itself collapses. The first-level
  wind accelerates until it is the minimum wind for sustainable
  turbulence, ``U_1 = (H/H_{max}^{wall})^{1/3}U_0``. At 2
  ``H_{max}^{wall}`` it is held at 1.253–1.257 ``U_0`` against the
  predicted ``2^{1/3} = 1.260``, from 5 to 40 ``H/u_*``.

So a single ``H_{max}`` cannot separate the regimes of the issue in this
case. The test asserts each threshold at the MSHF it belongs to:

* ``H = 0.5H_{max}^{bulk}``: TKE ≥ 30 % of neutral. Measured: 83 %.
* ``H = 2H_{max}^{wall}``:
  * γ < 0.2 on all 11 levels of the lowest third;
  * lowest-third TKE 2.8 % of neutral at 20 ``H/u_*``;
  * whole-domain TKE 3.6 % at 40 ``H/u_*``.

The whole-domain TKE at 20 ``H/u_*`` is 13 %, not below 5 %. The upper
two thirds, cut off from the wall shear, decay on their own turnover
time. So the issue's "< 5 % at 20 turnovers" was the wrong time for the
whole domain, and the test checks it at 40.

### MSHF in the Berlin case

The `:surface_exchange` coupling evaluates the MSHF of every column at
every surface step: one maximisation for the column's wind, forcing
height, ``z_{0m}`` and ``θ_{va}``, through
`SurfaceExchange.surface_mshf_ratio`. It writes the fields `H_max_sfc`
and `mshf_ratio`. Its report gives the mean ratio of downward buoyancy
flux to MSHF over the stable columns of each class (street, roof, park),
and the share of those columns *beyond* the MSHF point: ``z_{ref}/L``
past the peak of the flux curve, on the very stable branch where a
stronger temperature contrast carries less heat (output `mshf_beyond`).
A small ratio alone does not mean a column is far from collapse: a
collapsed column also has a small ratio, on the other side of the peak.

Its smoke check asserts ``|H| ≤ H_{max}(1 + 10^{-8})`` for every column
and every surface step, on the Monin–Obukhov fluxes of the skin at the
start of the step. The fluxes a soil column *returns* are its linear
extrapolation in ``T_{skin}`` to the step's mean skin temperature. Near
the maximum of the concave ``H(T_{skin})`` this extrapolation overshoots
by ``O(ΔT^2)``: up to 3.1 % in the smoke run, on shaded street columns
with ``U ≈ 0.6`` m/s and ``H_{max} ≈ 1`` W m⁻². Those fluxes are checked
against 10 % in the 12-step smoke run, and the measured worst case is
reported.

That bound does not hold over a full day. In the 06:00→03:00 flagship
run the delivered ratio reached 402 in the first half hour and 2684 by
the evening. Instrumenting the worst columns showed why: soil columns at
near-calm wind (``U ≈ 0.1``–0.3 m/s), where ``H_{max} ∝ U^3`` is only
0.004–0.07 W m⁻². At the start of the step MOST gives an *upward*
buoyancy flux (ratio −15 to −72, evaporation). The linearised step then
delivers dew (``LE ≈ −10`` W m⁻²) and ``H ≈ −1.5`` W m⁻², a small
downward buoyancy flux that is many times the tiny ``H_{max}``. The
excess is a few W m⁻² in absolute terms. It is a property of the soil
skin's linearisation in ``T_{skin}`` and ``q_s``, not of MOST, whose own
fluxes stay at or below ``H_{max}`` (worst 1.000) all day. The
diagnostic exposes it; it does not fix it.

## The boundary-layer height

`boundary_layer_height(z, profile; method)` is the only ``z_i``
diagnostic in the package (#247). It is applied in the same way to
radiosonde profiles (`LargeScaleForcing.sounding_profiles`) and to LES
output: the CBL harness `test/tke_cbl_driver.jl` delegates to it, and
so do the diagnostics of the nesting epic (#246). It offers four
methods, following the review of Seibert et al. (2000):

* `:parcel`, the parcel method (Holzworth 1964). ``z_i`` is the first
  height at which ``θ_v(z) = θ_v(z_1) + \text{excess}``, linearly
  interpolated between levels. On a profile that is stable from the
  ground it returns ``z_1``.
* `:bulk_richardson`, the bulk Richardson number from the lowest level,
  ``Ri_b(z) = (g/θ_{\rm ref})\,Δθ_v\,Δz / (Δu^2 + Δv^2 + b u_*^2)``,
  with ``Ri_c = 0.25`` and ``b = 100`` (Vogelezang & Holtslag 1996).
  It is evaluated as `gradient_richardson(N², S²)` of the layer
  ``[z_1, z]``, so the shear floor of the previous section also covers
  a calm profile.
* `:max_gradient`, the midpoint of the layer with the largest
  ``∂θ/∂z``.
* `:min_flux`, the height of the minimum of the total heat flux
  ``\overline{w'θ'}`` (Sullivan et al. 1998).

**Measured.**

* On a mixed layer with a 3 K jump at ``h`` = 800 m, sampled every
  25 m, every profile method returns ``h`` to within one level.
* On a linear stable profile `:parcel` returns the first level.
* On the 24³ CBL of the TKE harness (1 h), `:min_flux` reproduces the
  former local `zf[argmin(flux)]` bit for bit (900 m), and `:parcel` on
  the mean θ gives 847 m (−5.9 %).
* At Lindenberg at 12 UTC on 29 June 2019 (released 10:45 UTC),
  `:parcel` and `:bulk_richardson` both give 1171 m and `:max_gradient`
  gives 1149 m. Beyrich & Leps (2012) give a monthly-mean 12 UTC mixing
  height of 1200–1400 m for early summer at Lindenberg.
* The 00/06 UTC launches show surface-based inversions 79–329 m deep.
  The GTS significant levels stored in IGRA do not resolve a nocturnal
  boundary layer, so the test compares this inversion depth, not a
  night ``z_i``.

## In the Berlin case

The `turbulence_diag` addon writes the plane-mean ``N^2``, ``Ri_g`` and
``f_m`` profiles at every output record. See
[The Berlin case](../design/berlin_turbulence.md#Turbulence-diagnostics-(#170)),
which also gives the measured values in the 60 m capping inversion of
the smoke case.

## References

* Beyrich, F. & Leps, J.-P. (2012). An operational mixing height data
  set from routine radiosoundings at Lindenberg: methodology.
  *Meteorol. Z.*, 21, 337-348.
* Holzworth, G. C. (1964). Estimates of mean maximum mixing depths in
  the contiguous United States. *Mon. Wea. Rev.*, 92, 235-242.
* Seibert, P., Beyrich, F., Gryning, S.-E., Joffre, S., Rasmussen, A. &
  Tercier, P. (2000). Review and intercomparison of operational methods
  for the determination of the mixing height. *Atmos. Environ.*, 34,
  1001-1027.
* Sullivan, P. P., Moeng, C.-H., Stevens, B., Lenschow, D. H. & Mayor,
  S. D. (1998). Structure of the entrainment zone capping the convective
  atmospheric boundary layer. *J. Atmos. Sci.*, 55, 3042-3064.
* Vogelezang, D. H. P. & Holtslag, A. A. M. (1996). Evaluation and
  model impacts of alternative boundary-layer height formulations.
  *Boundary-Layer Meteorol.*, 81, 245-269.
* Donda, J. M. M., van Hooijdonk, I. G. S., Moene, A. F., Jonker,
  H. J. J., van Heijst, G. J. F., Clercx, H. J. H. & van de Wiel,
  B. J. H. (2015). Collapse of turbulence in stably stratified channel
  flow: a transient phenomenon. *Q. J. R. Meteorol. Soc.*, 141,
  2137-2147.
* Van de Wiel, B. J. H., Moene, A. F., Jonker, H. J. J., Baas, P.,
  Basu, S., Donda, J. M. M., Sun, J. & Holtslag, A. A. M. (2012). The
  minimum wind speed for sustainable turbulence in the nocturnal
  boundary layer. *J. Atmos. Sci.*, 69, 3116-3127.
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
