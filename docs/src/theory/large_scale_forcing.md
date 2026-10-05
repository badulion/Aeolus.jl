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

All forcings act on the cell-centre heights
``z_k = z_0 + (k - \tfrac12)Δz``. On the C-grid these are the heights of
``u``, ``v`` and the scalars.

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

## Geostrophic nudging

`GeostrophicNudging(U_g, V_g, τ)` relaxes the **horizontal mean** of
each level toward the target:

```math
\frac{\partial u}{\partial t} = \dots - \frac{\langle u\rangle_{xy} - U_g}{τ} .
```

Turbulent fluctuations are not damped. This is the Berlin case's
behaviour before #171, moved here unchanged. Over 100 Berlin smoke steps
it reproduces the former `add_geostrophic_wind!` bit for bit (`==` on
``u, v, w, θ, p`` and every tracer).

Nudging is not a pressure gradient. It has no Coriolis partner, so it
damps the inertial oscillation on the time scale ``τ`` and pulls the
nocturnal jet back toward ``U_g``.

## Advective tendencies

`AdvectiveTendencies(z, t, ∂θ/∂t, ∂q/∂t, ∂u/∂t, ∂v/∂t)` adds the
prescribed large-scale advection uniformly over each level, as the
GABLS2 and DICE protocols specify it (Svensson et al. 2011). Each entry
may be absent. The humidity tendency goes to the simulation's `tqv`, or
to the field passed as `tq`. A humidity tendency with no field to
receive it is an error, not a silent no-op. Subsidence depends on the
model state and is not included.

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

It is duck-typed on `mesh`, `u`, `v`, `tu`, `tv`, `tθ` and `f_coriolis`,
so it works with every dynamical core, the anelastic ones included.

## Users

* **Berlin case**: the `:geostrophic` coupling, with
  `mode = :pressure_gradient` by default and `mode = :nudging` as an
  option. See [The Berlin case](../design/berlin_case.md).

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

* Blackadar, A. K. (1957). Boundary layer wind maxima and their
  significance for the growth of nocturnal inversions. *Bull. Amer.
  Meteor. Soc.* 38, 283–290.
* Ekman, V. W. (1905). On the influence of the earth's rotation on
  ocean-currents. *Ark. Mat. Astron. Fys.* 2(11), 1–52.
* Holton, J. R. & Hakim, G. J. (2013). *An Introduction to Dynamic
  Meteorology*, 5th ed., Academic Press, §3.2 and §8.4.
* Svensson, G. et al. (2011). Evaluation of the diurnal cycle in the
  atmospheric boundary layer over land as represented by a variety of
  single-column models: the second GABLS experiment. *Bound.-Layer
  Meteor.* 140, 177–206.
* Durre, I., Yin, X., Vose, R. S., Applequist, S. & Arnfield, J.
  (2018). Enhancing the data coverage in the Integrated Global
  Radiosonde Archive. *J. Atmos. Oceanic Technol.*, 35, 1753-1770.
* Wallace, J. M. & Hobbs, P. V. (2006). *Atmospheric Science: An
  Introductory Survey*, 2nd ed., Academic Press, §3.2.2.
