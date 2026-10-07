# Large-scale forcing

An LES domain of a few kilometres does not contain the synoptic
pressure field, the large-scale advection or the surface that drives a
boundary-layer case. They are prescribed. `Aeolus.LargeScaleForcing`
(Phase 58, #171) is the one place where that forcing lives. The GABLS
and CASES-99 cases are built on it, Phase 67 will fill it from
reanalysis, and the Berlin flagship uses it for its background wind.
Before #171 there were two copies of the geostrophic forcing, and they
disagreed: the Berlin case nudged the mean wind, while the patchy-surface
driver applied a pressure gradient.

## Forcing quantities

Each forcing quantity ``F`` is one of:

* a number;
* a height profile, `ForcingTable(z, values)`;
* a height-time table, `ForcingTable(z, t, values)`;
* a function `(z, t) -> value`.

A table is interpolated linearly in ``z`` and in ``t``. Outside the
tabulated range it is held at its end values, as the GABLS and DICE
forcing files are used, and it is never extrapolated.

A forcing is applied over a whole step ``[t, t+Δt]``, so the tendency
used is the **step mean**

```math
\bar F(z) = \frac{1}{Δt}\int_t^{t+Δt} F(z, t')\,dt' .
```

For a table this mean is exact, because ``F`` is piecewise linear in
``t`` and the antiderivative is evaluated segment by segment. A forward
step therefore adds ``Δt\,\bar F``, and the steps sum to
``\int F\,dt`` to round-off whatever the step size and wherever the
table nodes fall. A function is averaged with Simpson's rule. A number
is its own mean.

All forcings act on the cell-centre heights ``z_k``, read from the
mesh: ``z_0 + (k - \tfrac12)Δz`` on a `RegularGrid`, the stored centres
and cell thicknesses on a `StretchedGrid` (#248). On the C-grid these
are the heights of ``u``, ``v`` and the scalars.

## Geostrophic pressure gradient

When the large-scale pressure field is in geostrophic balance with the
wind ``(U_g, V_g)``, the horizontal momentum equations on an f-plane are

```math
\frac{\partial u}{\partial t} = \dots + f v - f V_g, \qquad
\frac{\partial v}{\partial t} = \dots - f u + f U_g .
```

The core already adds the Coriolis terms ``+fv`` and ``-fu``.
`GeostrophicPressureGradient(U_g, V_g)` adds the pressure-gradient terms
``-f V_g`` and ``+f U_g`` (Holton & Hakim 2013, §3.2). In this form the
geostrophic wind is a **pressure gradient**, not a target state. Two
consequences follow, and both are tested.

**Inertial oscillation** (Blackadar 1957). For frictionless, horizontally
uniform flow, the ageostrophic wind
``W = (u - U_g) + i(v - V_g)`` obeys ``dW/dt = -ifW``. It turns clockwise
in the northern hemisphere at the rate ``f``, returns after the period
``2π/f``, and the speed peaks at ``|U_g| + |W_0|``. At night the surface
friction decouples from the residual layer, the daytime subgeostrophic
wind is left as an ageostrophic departure, and that departure
oscillates. This is the mechanism of the nocturnal low-level jet.

*Test* (`f = 1e-4`, ``U_g = 5``, ``W_0 = 2 + i``, SSP-RK3 with
``Δt = 2π/(1000f)``):

| Quantity | Measured | Required |
| :-- | :-- | :-- |
| Error after one period | 6.5 × 10⁻⁸ ``\|W_0\|`` | < 10⁻³ ``\|W_0\|`` |
| Maximum speed vs ``U_g + \|W_0\|`` | −1.8 × 10⁻⁷ (relative) | 0.1 % |
| ``W`` after a quarter period | ``(1, -2) = -iW_0`` (clockwise) | — |

**Ekman spiral** (Ekman 1905). With constant viscosity ``ν`` and a
no-slip wall, the steady state is

```math
u = U_g\,(1 - e^{-z/δ}\cos z/δ), \qquad v = U_g\,e^{-z/δ}\sin z/δ, \qquad δ = \sqrt{2ν/f} .
```

Its surface stress points 45° to the left of ``U_g``.

*Test* (``ν = 1`` m² s⁻¹, ``f = 10^{-4}``, so ``δ = 141.4`` m;
``N_z = 64``, ``Δz = 20`` m, a domain of 9 δ, an impulsive start from
``u = U_g``, run for ``ft = 60``):

| Quantity | Measured | Required |
| :-- | :-- | :-- |
| Relative L2 error of the profile | 5.75 × 10⁻³ | < 1 % |
| Relative L2 error of the ageostrophic part | 2.4 × 10⁻² | — |
| Maximum ``u`` (supergeostrophic) | 10.677 m/s (analytic 10.670) | — |
| Maximum ``v`` | 3.242 m/s (analytic 3.224) | — |
| Surface veer (direction of the wall stress) | 44.74° | 45° ± 0.5° |

Neither the `NoSlipWall` of P59-a (#172) nor the `SpaldingLaw` of P57-a
(#169) was on master when this was written. The test therefore applies
the no-slip bottom as the laminar wall law ``τ/ρ = ν u_1/(Δz/2)``, which
is the ``u^+ = y^+`` limit of Spalding's law and is equivalent to a
no-slip ghost ``u_0 = -u_1``.

That wall stress is first order in ``Δz``. The surface veer is the
direction of the model's wall stress, which is the direction of the
first-level wind. At ``z_1 = 10`` m, the analytic wind points at 43.0°,
while the model's first level points at 44.7°. A quadratic extrapolation
of the discrete shear to ``z = 0`` gives 47.0°. Both differences come
from the half-cell wall closure. They are not an error of the forcing.

## Profile nudging

`ProfileNudging(targets, τ; weight)` relaxes the **plane mean** of any
of ``u``, ``v``, ``θ``, ``q_v`` toward a reference (#248):

```math
\left.\frac{\partial φ}{\partial t}\right|_n = -\,w(z)\,\frac{\langle φ\rangle_{xy}(z) - φ_{ref}(z, t)}{τ} .
```

The tendency is one number per level, added to every point of it, so
the fluctuations ``φ - \langle φ\rangle`` are not damped. The reference
is any forcing quantity and enters as its step mean. This is the
nudging of DALES (Heus et al. 2010, Eq. 92), of the PALM HOPE runs
(Heinze et al. 2017, Eq. 4, τ = 6 h for θ and q) and of MicroHH (van
Heerwaarden et al. 2017). The weight ``w(z)`` is `nothing` (``w ≡ 1``),
a `NudgingWeight(z_n; factor, depth)` (zero below
``z_0 = \mathrm{factor}\cdot z_n``, one above ``z_0 + \mathrm{depth}``, a
``\sin^2`` ramp between) or any callable `z -> w`. ``z_n`` may be a
callable on the state, e.g. a diagnosed boundary-layer depth, so that
only the free troposphere is nudged. `apply_forcing!` returns the volume
integrals ``\int (∂φ/∂t)_n\,dV`` it added (also `tendency_integrals(F)`),
which a ledger books.

`GeostrophicNudging(U_g, V_g, τ)` is `ProfileNudging((u = U_g, v = V_g),
τ)`: a constructor, not a second implementation. It is the Berlin
case's behaviour before #171. Over 100 Berlin smoke steps it reproduces
the former `add_geostrophic_wind!` bit for bit (`==` on ``u, v, w, θ, p``
and every tracer); the plane mean and the subtraction are computed in
the same order as before.

Nudging is not a pressure gradient. It has no Coriolis partner, so it
damps the inertial oscillation on the time scale ``τ`` and pulls the
nocturnal jet back toward ``U_g``.

*Tests* (`test/era6_nudging_subsidence.jl`):

| Check | Measured | Required |
| :-- | :-- | :-- |
| Forward Euler, Δt = τ/100, with a `NudgingWeight(300 m; depth = 200 m)`: level residual vs ``\prod (1 - wΔt/τ)`` | 6.1 × 10⁻¹⁴ | 10⁻¹² |
| Level below ``z_n`` | untouched (302 K exactly) | `==` |
| SSP-RK3, Δt = τ/1000: residual after ``t = τ`` vs ``e^{-1}`` (θ / u) | 2.1 × 10⁻¹¹ / 1.5 × 10⁻¹¹ | 10⁻¹⁰ |
| Added tendency identical on every point of a level (u, v, θ) | yes | `==` |
| Fluctuations after one nudged step vs un-nudged (θ′ / u′) | 2.8 × 10⁻¹³ K / 1.3 × 10⁻¹⁵ m/s | 10⁻¹² |

The forward-Euler residual after ``t = τ`` is ``(1 - 1/N)^N = 0.36603``
for ``N = 100``, not ``e^{-1} = 0.36788``: that difference,
``≈ e^{-1}/2N``, is the stepper's, and the test checks the residual
against the exact Euler factor. Only a third-order step (or
``N ≳ 10^9``) reaches ``e^{-1}`` to 10⁻¹⁰. The fluctuations agree to
round-off, not bit for bit: ``(a + c) - (\bar a + c)`` and ``a - \bar a``
round differently in floating point when the turbulent tendency is not
a dyadic number. Bitwise identity holds for the added tendency itself.

## Large-scale subsidence

A large-scale vertical velocity ``w_s(z, t)`` (negative downward)
advects the resolved fields. `Subsidence(w_s; fields = (:θ, :qv),
form = :mean_gradient)` adds (#248)

```math
\left.\frac{\partial φ}{\partial t}\right|_s = -\,w_s(z, t)\,
\frac{\langle φ\rangle_{k+1} - \langle φ\rangle_{k-1}}{z_{k+1} - z_{k-1}}
```

uniformly per level: the centred gradient of the plane mean, as DALES
(Heus et al. 2010, Eq. 91) and MicroHH do. It moves the mean profile and
leaves the fluctuations alone, and it is exact for a linear profile on
any grid, stretched included. `form = :local` (Heinze et al. 2017,
Eq. 3) uses each column's own gradient, first-order upwind with respect
to ``w_s``. On the bottom and top levels the gradient is one-sided
toward the interior. Subsidence caps the growth of a convective
boundary layer: it lowers the inversion at ``w_s(z_i)`` while
entrainment raises it.

The centred form needs a time scheme that is stable for centred
advection. SSP-RK3 is (``|w_s|Δt/Δz ≤ \sqrt 3``); forward Euler is
weakly unstable for it, so with Euler use `:local` or a small Courant
number. `qv` is subsided by default only when the simulation carries
moisture; a field named explicitly that the simulation lacks is an
error. Unlike the advective tendencies, subsidence depends on the state
and does not conserve ``\int φ\,dV``. `apply_forcing!` returns the
volume integrals it added, and a ledger books them: with SSP-RK3 the
three stages are booked with the weights 1/6, 1/6, 2/3.

*Tests*:

| Check | Measured | Required |
| :-- | :-- | :-- |
| ``θ = θ_0 + Γz`` with fluctuations, constant ``w_s = -0.01``: tendency vs ``-w_sΓ`` at every level (mean-gradient / local) | 2.4 × 10⁻¹² / 6.1 × 10⁻¹³ (relative) | 10⁻¹⁰ |
| The same on a geometric `StretchedGrid` (Δz 44.5-179.9 m), ``w_s = -10^{-5} z`` | 1.2 × 10⁻¹³ / 5.6 × 10⁻¹⁴ | 10⁻¹⁰ |
| A 4 K jump at 1000 m, ``w_s = -0.01`` m/s, 6 h, Δz = 25 m, no turbulence: mid-jump height vs ``z_1 + w_s t = 784`` m (mean-gradient / local) | 800.2 m (0.65 Δz) / 788.2 m (0.17 Δz) | 1 Δz |
| Mixed layer (constant ⟨θ⟩, ±0.05 K fluctuations) | 0 K/s | round-off |
| Closed turbulent box, nudging + subsidence, 600 s: ``Δ\int θ\,dV`` vs the booked integrals (Euler / RK3, both forms) | 1.3-5.8 × 10⁻¹² | 10⁻¹⁰ |
| The same for ``q_v`` (Euler) | 2.6-9.1 × 10⁻¹² | 10⁻¹⁰ |

The centred form over-shoots the jump by 1.1 K of 4 K (dispersion of a
centred scheme on a discontinuity); the upwind form smears it but is
monotone. Real inversions in an LES are resolved over several levels,
where the centred form is the accurate one.

### GABLS2

The GABLS2 case applies ``w_s = -0.005\,z/1000`` m s⁻¹ below 1 km
(−0.005 above) from 40 h local time (16 LT 23 October), from the CLUBB
participant set-up (`examples/sbl/data/gabls2_case.csv`; since CFG-d
#359 the `[forcing.subsidence]` table of `examples/sbl/configs/gabls2.toml`).
That config applies it to θ (`--set forcing.subsidence={}` for the
sensitivity run without it), and the SBL driver books it in its θ ledger. In the reduced test
(16 × 16 × 40, Δx = 150 m, Δz = 40 m), the residual-layer inversion
(the largest ``∂⟨θ⟩/∂z`` above 300 m) at 06 LT 24 October is 856.0 m
with subsidence and 1039.7 m without (−183.6 m), 0.80 Δz above a passive
surface descending from the 18 LT inversion. The θ ledger closes to
8.1 × 10⁻¹³ of the column content.

At 48 × 48 × 80 (Δx = 50 m, Δz = 20 m, `examples/sbl/gabls2.jl`,
Brown et al. 1994 closure, 16-54 h, 3607 s / 3691 s wall), the inversion
at 06 LT 24 October is 842.1 m with subsidence and 1059.2 m without
(−217.1 m). A passive surface leaving the 18 LT inversion (1033.8 m)
reaches 833.4 m, so the modelled inversion descends as a passive surface
to within 8.7 m (0.43 Δz). The booked subsidence heating to 54 h is
2329 K m (1.46 K column mean).

Night 2 against the digitised screen-level envelopes
(`gabls2_reference.csv`) at 48 h:

| | with subsidence | without | obs | LES |
| :-- | :-- | :-- | :-- | :-- |
| T(2 m) (°C) | 2.40 | 2.52 | 1.2-2.6 | 1.9-3.3 |
| U(10 m) (m s⁻¹) | 3.58 | 3.78 | 2.1-2.9 | 4.2-5.0 |

Subsidence keeps T(2 m) inside both envelopes and does not change which
envelopes the run is inside. U(10 m) is outside both, with and without
subsidence. That is not the forcing: the observed and LES envelopes do
not overlap (2.9 vs 4.2 m s⁻¹), and the model lies between them. In the
reduced test (Δz = 40 m) all four 48 h screen values are outside, with
and without subsidence; it asserts only that subsidence moves them by
less than the digitisation half-widths and changes no in/out status.

The two 48 × 48 × 80 runs already differ slightly in the first night
(H = −22.8 vs −22.3 W m⁻² at 30 h), before the subsidence starts. The
subsidence tendency is exactly zero before 40 h: at 16 × 16 × 40 the
two runs are bit for bit identical at 30 h. The difference is
run-to-run nondeterminism of the 48 × 48 × 80 run, not the forcing.

## Advective tendencies

`AdvectiveTendencies(z, t, ∂θ/∂t, ∂q/∂t, ∂u/∂t, ∂v/∂t)` adds the
prescribed large-scale advection uniformly over each level, as the
GABLS2 and DICE protocols specify it (Svensson et al. 2011). Each entry
may be absent. The humidity tendency goes to the simulation's `tqv`, or
to the field passed as `tq`. A humidity tendency with no field to
receive it is an error, not a silent no-op. Subsidence depends on the
model state and is `Subsidence` (below).

*Test*: a turbulent closed box, with periodic sides, impermeable walls
and no surface flux, so flux-form advection and diffusion conserve
``\int θ\,dV``. The ``θ`` table varies with height and is piecewise
linear in time, with a kink at 250 s that falls inside a step and an
end at 600 s that is passed after 602 s. The change of ``\int θ\,dV``
matches ``\iint (∂θ/∂t)_{adv}\,dV\,dt`` (trapezoid rule on the table
nodes) to a relative 9.9 × 10⁻¹³.

## Surface-temperature series

`SurfaceTemperatureSeries(t, T_s)` is a prescribed skin temperature.
GABLS1, for example, uses 265 K cooling at 0.25 K/h. It is read with
`surface_temperature(series, t)`, a method of the `SurfaceExchange`
generic. It is not a surface model. It is the boundary value handed to a
wall law (P57-a) or a surface step.

## Radiosonde soundings

`Sounding` holds the launches of one station. `read_sounding(path)`
reads one from the CSV that `examples/urban/data/fetch_lindenberg.jl`
writes, and `sounding_profiles(s; z_top)` converts it to θ, ``q_v``, u,
v and p `ForcingTable`s on heights above the launch site and launch
times in seconds from an epoch (`sounding_time`). These tables are the
form that initial states, nudging targets and boundary data take
(#247, epic #246).

* **Heights** come from hypsometric integration of the virtual
  temperature, ``z_{k+1} = z_k + (R_d/g_0)\,\bar T_v\,\ln(p_k/p_{k+1})``
  (Wallace & Hobbs 2006, eq. 3.29). ``T_v`` comes from
  `MoistThermodynamics.virtual_temperature`. The standard gravity
  ``g_0`` defines the geopotential metre, so ``z`` plus the station
  elevation can be compared directly with the reported geopotential
  height, which is used only as a check.
* **θ** is ``T/Π(p)`` with the dry `exner_function`.
* **``q_v``** is the specific humidity whose vapour pressure is
  ``e_s(T_d)`` (`specific_humidity_from_relative_humidity` at RH = 1).
* **Interpolation.** The default height grid is the union of every
  launch's levels, so each launch's piecewise-linear profile is held
  exactly. `forcing_value` is linear in time between launches.

Measured on the six Lindenberg launches of 29/30 June 2019: the
computed heights of the standard levels below 3 km match IGRA's
geopotential heights to RMS 0.8–4.3 m. The bound is 4.7–4.9 m, set by
the whole-hPa surface pressure of the GTS message. θ equals
``T(p_0/p)^{R_d/c_{pd}}`` to ``10^{-12}``. The table returns each
launch's θ, ``q_v``, p, u and v exactly at its own levels.

## The one entry point

`apply_forcing!(sim, forcing, t, Δt)` (also accepts a tuple of forcings)
is called from the `body_force!` hook of `step!`, for example

```julia
step!(sim, dt; body_force! = (s, Δt) -> apply_forcing!(s, forcing, s.t, Δt))
```

It is duck-typed on `mesh`, `u`, `v`, `tu`, `tv`, `tθ` and `f_coriolis`
(and `qv`, `tqv` for humidity), so it works with every dynamical core,
the anelastic ones included. For the state-dependent forcings
(`ProfileNudging`, `Subsidence`) it returns the tendency volume
integrals; for the others, and for a tuple, it returns `sim`, and
`tendency_integrals(tuple, :θ)` sums the integrals of the members.

## Free-atmosphere top: the damping layer and the fixed-gradient top

A deep domain (the 3 km parent of the nesting epic, #246) ends in the
free atmosphere, not at a lid. Two things are needed there (N-parent-b,
#254): a layer that absorbs the gravity waves the boundary layer
radiates upward, and a θ condition at the top face that keeps the
stratification of the air above.

### Damping toward a reference

`ReferenceDamping(field, profile, target)` adds

```math
\left.\frac{\partial φ}{\partial t}\right|_{d} = -λ(z)\,\bigl(φ - φ_{ref}\bigr)
```

to `θ`, `qv` or a velocity component, with the rate ``λ(z)`` of a
`DampingProfile` (the sin² ramp of `sponge_profile(α_max, z_start, z_top)`,
Klemp & Lilly 1978, or the Nordström ramp of the fringe). It is an
`AbstractLargeScaleForcing`, applied with `apply_forcing!` from the
body-force hook, so the core's time integrator treats it like every other
tendency. The cores' own sponge damps the velocity only; θ and qv are
damped here and nowhere else.

The target is a number, a **level vector** (one value per interior level,
e.g. the initial ``\bar θ(z)``), a `Field` or 3-D array (the parent state
a nest is damped toward, N-open-d), a `ForcingTable` or ``(z, t)``
function (evaluated at the stage time), or `:mean`. With `:mean` only the
deviations from the slab mean are damped and the mean profile evolves
freely (DALES, Heus et al. 2010, Eqs. 40–41): every level's integral is
zero to round-off. The arithmetic is the one kernel
`FiniteVolume._add_damping!` that `add_sponge!` (whose `f_ref` takes the
same targets) and the fringe region use, on a `RegularGrid` and a
`StretchedGrid` alike; a constant level vector gives the number target's
tendency bit for bit. After each call `damping_tendency(F)` returns the
tendency per level and its volume integral, the term a heat ledger books.

**Exact relaxation.** A uniform θ anomaly with no flow, integrated by
SSP-RK3 with the forcing in every stage, decays per level by exactly
``R = 1 - a + a^2/2 - a^3/6`` (``a = λΔt``) per step: after 100 steps of
10 s the anomaly matches ``A R^{100}`` to 1.8 × 10⁻¹² K at every level,
and the levels below `z_start` keep their tendency at zero bit for bit.

**Gravity-wave absorption** (`test/era6_deep_top.jl`, test 3). A 2-D
packet with ``N = 0.01`` s⁻¹, ``λ_x = λ_z = 1`` km (``ω = N/\sqrt2``,
``c_{gz} = 0.56`` m s⁻¹, initialised from a C-grid stream function so its
discrete divergence is zero) propagates upward into a sin² layer of
``α_{max} = 10^{-3}`` s⁻¹ (``0.1N``). The reflected energy fraction is the
time-integrated energy at ``m > 0`` over that at ``m < 0`` in the vertical
spectrum of ``w`` below the layer (3–5.5 km, Hann window):

| layer | ``u, v, w`` damped | ``+ θ`` damped |
| :-- | --: | --: |
| none (rigid lid) | 0.985 | — |
| 1 ``λ_z`` deep | 0.1745 | 0.0338 |
| 2 ``λ_z`` deep | 0.0330 | 0.0036 |

Velocity-only damping leaves the buoyancy half of the wave energy
undamped, so at this weak rate the layer absorbs much less; damping θ as
well (PALM, WRF §4.4.3, MicroHH) brings the reflection below 5 % from one
vertical wavelength. The advantage is a weak-damping one: in a scan of
``α_{max}`` (not asserted) the θ damping helps up to ``α_{max} ≈ 2 × 10^{-3}``
s⁻¹ and, for a 1 ``λ_z`` layer, reflects slightly *more* above it
(0.0157 against 0.0049 at ``α_{max} = 10^{-2}``), where the onset of the
damping itself begins to reflect (Klemp & Lilly 1978); the noise floor of
the measurement is about 0.003.

### The fixed-gradient top

`FluxBC(Γ)` as the top face of a scalar's boundary conditions
(`FieldBoundaryConditions(:periodic, :periodic; bottom = :noflux, top =
FluxBC(Γ))`) holds ``∂θ/∂z = Γ`` at the top, usually the initial lapse
rate (PALM; DALES, Heus et al. 2010, Eq. 39). `fill_halos!` sets

```math
θ_{N+h} = θ_{N+1-h} + Γ\,\bigl(z_{N+h} - z_{N+1-h}\bigr), \qquad h = 1, …, H_z,
```

with the cell-centre heights (on a `StretchedGrid` the first gap is the
face spacing ``Δz_f`` across the top face), so a linear profile of slope
Γ is continued exactly into every halo layer and a diffusion with
diffusivity ``κ`` carries exactly ``κΓ`` (per unit area) into the domain.
The condition is affine in θ: its homogeneous part is `FluxBC(0)` (the
mirror), which is what `homogeneous_bcs` returns, so the implicit
diffusion of `step_imex!` moves ``κΓ`` to the right-hand side as it does
for a moving wall. `qv` and the SGS TKE share θ's boundary conditions
but not θ's prescribed gradient (they are filled with
`homogeneous_bcs(θ_bcs)`).

Measured (`test/era6_deep_top.jl`, test 4):

* ``Δ∫θ\,dz = κΓΔt`` per step to 4.4 × 10⁻¹¹ (relative) with `step!`, and
  to 1.4 × 10⁻¹² with ARK2 at 6.8 times the forward-Euler diffusion limit;
* a linear profile of slope Γ (with the surface flux ``-κΓ`` below) is
  steady to 0 K over 100 steps;
* against the analytic solution of ``θ_t = κθ_{zz}``,
  ``θ_z(0) = 0``, ``θ_z(L) = Γ``,
  ``θ = Γ(z^2/2L + κt/L) + a\cos(πz/L)e^{-κπ^2t/L^2}``, the maximum error
  falls by 3.98 and 4.00 under ``Δz`` halving on a `RegularGrid`
  (9.6 × 10⁻⁵ → 6.0 × 10⁻⁶ from 16 to 64 levels), and by 3.55 and 3.78 on a
  geometric `StretchedGrid` refined as a smooth map (ratio ``1.1^{16/N_z}``),
  still approaching 4 from below.

**Budget.** With flow, buoyancy, a surface flux ``F_s``, the gradient top
and θ damping toward the initial profile, every `step!` changes
``∫ρc_pθ\,dV`` by ``ρc_pΔt\,(D + κΓA + F_sA)`` (``D`` the damping
integral) to 5.4 × 10⁻¹² relative (test 5).

### In the Berlin case

The 200 m child's θ top is `cfg.θ_top = :initial_gradient`
(`FluxBC(3 × 10⁻³ K/m)`, the initial lapse rate above the 100 m
inversion) instead of the closed lid; the dynamics books the heat it
carries as `(:dynamics, :top_flux)` in the air-heat ledger. θ is **not**
damped in the child: its sponge starts at 150 m, so a constant-target
damping would clamp the top 50 m (#238, approach B), and in the nested
run the child top comes from the parent. The θ damping layer runs in the
parent (N-parent-e). See [The Berlin case](../design/berlin_ledgers.md#The-θ-top).

## Users

* **Berlin case**: the `:geostrophic` coupling, with
  `mode = :pressure_gradient` by default and `mode = :nudging` as an
  option (a `ProfileNudging` since #248). See
  [The Berlin case](../design/berlin_case.md). θ/qv nudging and
  subsidence belong to the deep parent run of the nesting epic (#246,
  N-parent-e), not to the nested Berlin child: the child takes its top
  and lateral θ and q from the parent, and nudging it as well would apply
  the large scale twice.
* **GABLS2** (`examples/sbl/cases.jl`): `Subsidence` from 40 h.

### The Berlin case and the sponge

The cores' Rayleigh sponge (top 25 % of the domain,
``α_{max} = 0.05`` s⁻¹) relaxes ``u, v`` toward zero. Nudging hid this,
because it pulls the mean back toward ``U_g`` at every level. A pressure
gradient does not. The sponge is then a momentum sink of ``α⟨u⟩``, and
the forcing ``fU_g = 10^{-4}`` m s⁻² cannot balance it. In the 24 h run
with the sponge still targeting zero, the domain-mean wind collapsed to
0.12 m/s, and the top level fell to 0.003 m/s.

So in the pressure-gradient mode the coupling retargets the sponge at
``(U_g, V_g)``. This is option `sponge_to_geostrophic`, and it is on by
default. The coupling adds ``+α(z)U_g`` using the core's own sponge
profile (`add_sponge!` on a zero field with `f_ref = U_g`). The sum is
then ``-α(z)(u - U_g)``, the usual sponge of a pressure-gradient-forced
LES. The nudging mode is left exactly as it was.

*24 h flagship run* (`alexanderplatz`, 06–06 h local, ``U_g = 1`` m/s,
32 levels of 6.25 m, sponge above 150 m). The values are domain means,
building cells included:

| Mode | Night (20–06 h) wind maximum below the sponge | ``(ū, v̄)`` at 100 m, 06 h | top level |
| :-- | :-- | :-- | :-- |
| pressure gradient, sponge → ``U_g`` (default) | 0.81 m/s at 147 m (22.5 h) | (0.446, +0.054) | 0.997 m/s |
| pressure gradient, sponge → 0 | 0.12 m/s at 91 m (21.8 h) | (0.012, +0.115) | 0.003 m/s |
| nudging (before #171) | 0.47 m/s at 91 m (06 h) | (0.464, −0.044) | 0.014 m/s |

With the pressure gradient, the mean wind crosses the isobars toward low
pressure (``v̄ > 0``), as the Ekman balance requires. Under nudging
there is no pressure gradient to oppose Coriolis, so the mean wind turns
the wrong way (``v̄ < 0``).

**No low-level jet forms in this domain.** In every mode the night
profile increases monotonically up to the sponge. The domain is 200 m
deep, its top 50 m are sponge, and the city canopy keeps the whole
column coupled to the surface. That leaves no decoupled residual layer
in which an inertial oscillation could run. To resolve the jet, the
domain would need to be deeper than the nocturnal boundary layer.
* **Patchy-surface validation driver**
  (`examples/validation/patchy_surface/driver.jl`): it now delegates its
  ``+fU_g`` to `GeostrophicPressureGradient`. After 60 steps of the
  ``U_g = 5`` m/s, ``f = 10^{-4}`` setup, the state is unchanged bit for
  bit, and `test/era5_heterogeneous_surface.jl` prints identical numbers.

## References

* Heinze, R. et al. (2017). Evaluation of large-eddy simulations forced
  with mesoscale model output for a multi-week period during a
  measurement campaign. *Atmos. Chem. Phys.* 17, 7083–7109.
* Heus, T. et al. (2010). Formulation of the Dutch Atmospheric
  Large-Eddy Simulation (DALES) and overview of its applications.
  *Geosci. Model Dev.* 3, 415–444.
* van Heerwaarden, C. C. et al. (2017). MicroHH 1.0: a computational
  fluid dynamics code for direct numerical simulation and large-eddy
  simulation of atmospheric boundary layer flows. *Geosci. Model Dev.*
  10, 3145–3165.

* Blackadar, A. K. (1957). Boundary layer wind maxima and their
  significance for the growth of nocturnal inversions. *Bull. Amer.
  Meteor. Soc.* 38, 283–290.
* Ekman, V. W. (1905). On the influence of the earth's rotation on
  ocean-currents. *Ark. Mat. Astron. Fys.* 2(11), 1–52.
* Durran, D. R. (2010). *Numerical Methods for Fluid Dynamics*, 2nd ed.,
  Springer, §8.4 (wave-absorbing layers).
* Heus, T. et al. (2010). Formulation of the Dutch Atmospheric
  Large-Eddy Simulation (DALES) and overview of its applications.
  *Geosci. Model Dev.* 3, 415–444.
* Holton, J. R. & Hakim, G. J. (2013). *An Introduction to Dynamic
  Meteorology*, 5th ed., Academic Press, §3.2 and §8.4.
* Klemp, J. B. & Lilly, D. K. (1978). Numerical simulation of hydrostatic
  mountain waves. *J. Atmos. Sci.* 35, 78–107.
* Maronga, B. et al. (2015). The Parallelized Large-Eddy Simulation Model
  (PALM) version 4.0. *Geosci. Model Dev.* 8, 2515–2551.
* Skamarock, W. C. et al. (2019). *A Description of the Advanced Research
  WRF Model Version 4*. NCAR/TN-556+STR, §4.4.3.
* van Heerwaarden, C. C. et al. (2017). MicroHH 1.0: a computational
  fluid dynamics code for direct numerical simulation and large-eddy
  simulation of atmospheric boundary layer flows. *Geosci. Model Dev.*
  10, 3145–3165.
* Svensson, G. et al. (2011). Evaluation of the diurnal cycle in the
  atmospheric boundary layer over land as represented by a variety of
  single-column models: the second GABLS experiment. *Bound.-Layer
  Meteor.* 140, 177–206.
* Durre, I., Yin, X., Vose, R. S., Applequist, S. & Arnfield, J.
  (2018). Enhancing the data coverage in the Integrated Global
  Radiosonde Archive. *J. Atmos. Oceanic Technol.*, 35, 1753-1770.
* Wallace, J. M. & Hobbs, P. V. (2006). *Atmospheric Science: An
  Introductory Survey*, 2nd ed., Academic Press, §3.2.2.
