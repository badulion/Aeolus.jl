# Fog and dew

This page covers the thermodynamic half of radiation fog in Aeolus
(Era IV, Phase 44, `src/RadiationFog.jl`): how much vapour condenses
in the air or onto a surface, where the latent heat goes, and why the
update cannot overshoot saturation. The droplet-radiative feedback that
drives radiation fog — fog cooling itself by emitting longwave from its
top — builds on these functions and is documented in
[Radiation fog](radiation_fog.md).

## 1. What went wrong before

The urban example shipped an `apply_fog_dew!` that was switched off
because "its warming term amplifies". Measured in a one-cell box model
with the example's own settings (surface 280 K, air 288 K,
``q_v`` = 10 g/kg held by resupply, ``\Delta t`` = 2 s, rate
``10^{-4}`` s⁻¹), it had two defects:

1. **Condensation in subsaturated air.** When the *surface* was below
   the dew point, vapour was removed from the *air* and its latent heat
   added to the *air*, whatever the air's own humidity. Nothing tied the
   heating to air-side saturation, so there was no brake: the air warmed
   from 288 K to 363 K in 24 h while its relative humidity fell from
   0.95 to 0.016.
2. **Explicit-rate overshoot.** In supersaturated air it removed
   ``r\,\Delta t\,(q_v - q_s)`` without the latent-heat feedback on
   ``q_s``, overshooting below saturation for ``r\,\Delta t \gtrsim 0.4``
   (``q_v - q_s`` = −6.3×10⁻⁴ at ``r\Delta t = 1``, −2.9×10⁻³ at 3).

In the full urban run (06–09 h, same settings) the first defect did not
blow the run up within 3 h; it showed instead as the scheme draining the
lowest layer of vapour entirely — ``q_v`` = 0 in some cells after
~35 minutes — which is condensation from air at zero relative humidity.

## 2. The conserved quantity

For a reversible isobaric phase change the invariant is the moist
enthalpy, referenced to liquid water (Emanuel 1994, ch. 4):

```math
h = (c_{pd}\, q_d + c_l\, q_t)\, T + L_v(T)\, q_v - L_f(T)\, q_i ,
```

with the Kirchhoff latent heats of the contract. This is
`MoistThermodynamics.moist_enthalpy`. The familiar moist static energy
``c_{pm} T + L_v q_v`` is *not* invariant here: it double-counts the
vapour heat capacity against ``L_v(T)`` and changes by
``(c_{pv} - c_l)\,T\,\Delta q``, about 24 % of ``L_v \Delta q`` at 285 K.

Because ``h`` is linear in ``T`` with slope ``c_{pm}``, condensing
``\Delta q`` at fixed ``h`` gives exactly

```math
c_{pm}(q_\text{final})\, \Delta T = L_v(T_\text{initial})\, \Delta q ,
```

which the tests check to 10⁻¹⁰.

## 3. Fog: saturation-limited condensation

The target state solves ``q_v - \Delta = q_s(T(\Delta), p)`` with
``T(\Delta)`` from the enthalpy constraint. The residual is strictly
decreasing in ``\Delta`` — condensing removes vapour *and* warms the
parcel, raising ``q_s`` — and ``\Delta \in [-q_l, q_v]``, so a
safeguarded Newton iteration inside that bracket always converges
(`fog_equilibrium`). Subsaturated air without fog is returned untouched
after one saturation evaluation.

A finite condensation timescale ``\tau`` relaxes toward that target by
the exact exponential factor

```math
\Delta q = \bigl(1 - e^{-\Delta t/\tau}\bigr)\, \Delta_\text{eq} ,
```

not by the explicit ``\Delta t/\tau``, which exceeds 1 once
``\Delta t > \tau``. Since the residual is monotone, any fraction in
``[0, 1]`` leaves the parcel on the same side of saturation, for every
``\Delta t``. The default ``\tau = 0`` is full saturation adjustment:
the phase-relaxation time of fog droplets at 100 cm⁻³ is a few seconds.

### Relation to the contract's `saturation_adjustment`

`MoistThermodynamics.saturation_adjustment` already solves a saturation
adjustment, but it conserves the liquid-ice potential temperature
``\theta_{li}`` rather than ``h``. The two are tested against each other.
At ``p = p_\text{ref}`` they agree to ``|\Delta q_l|/q_l \le 3.2\times10^{-3}``
and ``|\Delta T| \le 10^{-3}`` K for 275–310 K and ``S`` = 1.01–1.10. The
remaining difference comes from the exponential ``\theta_{li}``
approximation. Away from ``p_\text{ref}`` they disagree more, and the
contract is the one in error. Its ``\theta_{li}`` uses the moist Exner
exponent ``R_m/c_{pm}`` of the *current* partition, which changes when
vapour condenses. At 800 hPa, 285 K and ``S`` = 1.05 its adjustment
violates moist enthalpy by 2.8 % of ``L_v \Delta q``, and its condensate
differs from the exact value here by 1.75 %. This is a pre-existing
defect of the Phase 39 contract, recorded as `@test_broken` in
`test/era4_fog.jl`. Fog uses the enthalpy form because it is exact for
the isobaric step.

## 4. Dew: deposition with surface-side latent heat

Dew exchanges vapour between the lowest air layer (thickness
``\Delta z``) and a surface at ``T_s`` by the bulk law

```math
\rho\, \Delta z\, \frac{dq_v}{dt} = -\rho\, C_e U\, \bigl(q_v - q_s(T_s)\bigr),
```

integrated exactly over the step with ``T_s`` fixed. The air can at
most be dried to ``q_s(T_s)``, whatever ``\Delta t`` is. The deposited
mass ``m = -\rho \Delta z \Delta q_v`` goes to a surface reservoir and
its latent heat ``L_v(T_s)\, m`` to the **surface** energy budget; the
air temperature is unchanged, because the vapour that left carried its
own sensible enthalpy. When the surface warms, dew re-evaporates,
limited by the amount present.

With the urban example's coefficients (``C_e = 10^{-3}``,
``U`` = 3 m/s) a surface 7 K below the dew point collects
0.049 mm/h, under Monteith's (1957) observed ceiling of about
0.07–0.08 mm/h.

Saturation is over liquid water. On a sub-freezing surface the
deposit is really frost, which saturates over ice and releases
``L_s``. That regime is not modelled; it is treated as supercooled dew.

## 5. In the urban example

`apply_fog_dew!` (examples/urban/moisture_addons.jl) now delegates to
these functions and is switched on. Enabling it exposed that the
example's moisture transport, forward-Euler with centered advection,
is unconditionally unstable: with condensation off, relative humidity
reached 2.07 in 8,395 cells within 1000 steps, which the 0.03 clamp
had hidden. With condensation on, that noise turned into latent heat
and the run blew up. Moisture (``q_v`` and the fog water ``q_l``, which
must move with the air) now uses WENO-Z5 with the positivity limiter,
after which the run stays bounded with fog and dew active.

Measured over the full 24 h run (06 h to 06 h, ``\Delta t`` = 2 s): the
lowest-layer ``\theta`` stayed in 292.1–301.5 K and ``\max|\nabla\cdot u|``
= 1.70×10⁻¹⁵ (re-measured after the `fog_equilibrium` convergence fix; 1784 s wall). Dew formed on the initially 285 K surfaces in the morning,
peaked at 0.185 mm at 09:30 and had evaporated by 22:00. Fog water
reached at most 3.4 g/kg, confined to wall nooks, and was gone by 15:30.
The example's surfaces never cool below the dew point at night (the
minimum ``T_s`` rises from 285 K to 292 K over the day), so it contains no
nocturnal radiation fog. Two known residuals come from the example's IBM
velocity correction, which is applied after the projection and is not
divergence-free in cells next to buildings. First, a few wall-nook cells
gain vapour: fog now caps them at saturation. Second, the lowest-layer
``q_v`` drops to 0 in some cells between 18:30 and 22:30, while neither
dew nor fog is active.

A third residual belongs to the example's surface coupling, not to this
module. The example's bulk evaporation `apply_latent_flux!` (``\beta``-limited)
and dew re-evaporation are computed independently. On a surface that
still carries dew and is above the dew point, the two evaporation rates
therefore add, reaching up to ``(1+\beta)`` times the potential rate.
Physically a dew-wetted surface evaporates at the potential rate
(``\beta = 1``). The excess lasts only while dew remains, and at most
0.185 mm was present in the run. It is left for the land-surface
coupling to resolve.

## References

* Monteith, J. L. (1957). Dew. *Q. J. R. Meteorol. Soc.*, 83, 322–341.
* Garratt, J. R. & Segal, M. (1988). On the contribution of atmospheric
  moisture to dew formation. *Bound.-Layer Meteorol.*, 45, 209–236.
* Bott, A., Sievers, U. & Zdunkowski, W. (1990). A radiation fog model
  with a detailed treatment of the interaction between radiative
  transfer and fog microphysics. *J. Atmos. Sci.*, 47, 2153–2166.
* Emanuel, K. A. (1994). *Atmospheric Convection*. Oxford.
