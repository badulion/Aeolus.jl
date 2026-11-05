# Building energy: indoor air and thermal mass

A heated or cooled building feeds heat into the urban canopy through its
walls, windows and exhaust. How much it feeds, and when, depends on the
indoor temperature. The indoor temperature in turn depends on how much
heat the building's fabric has stored. A concrete office and a timber
house under the same weather keep very different indoor temperatures,
and they release their stored heat at different hours.
`src/BuildingEnergy.jl` models a single thermal zone. Its capacitances
are resolved: every massive element is a multi-layer conduction column,
and the zone air is one prognostic node. This follows the building
energy model that TEB couples to its facets (TEB-BEM; Bueno et al. 2011).

## 1. The network

```
 outdoor air / sun / sky ─┐          ┌─ glazing  U_w A_w ─┐
                          │ columns  │                    │
  envelope elements  ═════╪══════════╪══ h_j A_j ──┐      │
  (EnvelopeElement,       │          │             │      │
   FacetElement)          │          │          T_air ────┴── outdoor air
                          │          │   C_a       │      H_ve (infiltration,
  internal mass      ═════╧══════════╧══ h_j A_j ──┘            ventilation)
  (InternalMass)
```

**The air node.** The zone air has heat capacity ``C_a = ρ c_p V``, with
``ρ = p/(R_d T)`` and ``c_p`` taken from `MoistThermodynamics`. It is
coupled to:

- each interior surface ``j``, through the ISO 6946 combined surface
  coefficient ``h_j = 1/R_{si}``;
- the outdoors, through the glazing ``U_w A_w`` and the air exchange
  (ISO 13790 §9.3)

```math
H_{ve} = ρ c_p V \frac{n_{inf} + (1-η)\,n_{vent}}{3600},
```

  where ``n`` are air-change rates (h⁻¹) and ``η`` is the heat-recovery
  efficiency;
- the convective part ``(1-f_r)`` of the internal gains, and any HVAC
  heat.

**The massive elements.** Each is an `UrbanSurface` conduction column.
BuildingEnergy uses that column as it is and keeps no copy of the
solver.

- An `EnvelopeElement` is closed on the outside by outdoor air through
  ``1/R_{se}`` plus an absorbed irradiance (the sol-air form), by a
  fixed ground temperature, or adiabatically.
- A `FacetElement` is an `UrbanFacet`, so its outside is the full Era V
  surface energy balance.
- An `InternalMass` has one or both faces in the room.

Every interior face receives the imposed flux

```math
q = \frac{f_r Φ_{int} + Φ_{sol}}{\sum_j A_j},
```

which is the radiative part of the internal gains plus the transmitted
solar gain, spread over the exposed area.

**One node for air and radiant temperature.** Surface-to-surface
long-wave exchange is folded into ``h_j``, as in the ISO 6946
coefficients. The node is therefore strictly a blend of the air
temperature and the mean radiant temperature. The operative temperature
``(T_{air} + T_{mrt})/2`` is available as a diagnostic.

## 2. The coupled implicit step

The air node's time constant, ``C_a / \sum h_j A_j``, is a few minutes.
Any useful step is longer than that, so the air balance

```math
C_a \frac{T^{n+1}-T^n}{Δt} = \sum_j A_j\bigl(q - F_{in,j}(T^{n+1})\bigr)
  + (U_w A_w + H_{ve})(T_{out} - T^{n+1}) + (1-f_r)Φ_{int} + Φ_{hvac}
```

is solved implicitly and simultaneously with every column. Each column
takes a θ-scheme step with ``T^{n+1}`` as its interior environment.
That step is *affine* in ``T^{n+1}``: the new cell temperatures and both
face fluxes have the form ``a + b\,T^{n+1}``. This holds for a facet
too. Its outer linearisation depends only on its skin temperature at
the start of the step, and that value is fixed.

Each element is therefore stepped twice from the same initial state, at
``T^n`` and at ``T^n + 1`` K. The two solves give the base solution and
its exact derivative. The air balance then reduces to one linear
equation. Once it is solved, every element is set to the combination
that matches the new air temperature.

The result is identical, to round-off, to stepping each column directly
with the final air temperature (tested to 1e-9 K). The step is
unconditionally stable and conserves the zone's heat content exactly.
It costs two tridiagonal solves per column per step.

## 3. The ISO 13790 reference model

ISO 13790's simple hourly method (§7.2.2 and Annex C) represents a zone
by five conductances and one capacitance ``C_m``. The conductances link
the outdoor, supply-air, air (``θ_{air}``), central surface (``θ_s``) and
mass (``θ_m``) nodes, and ``θ_m`` is advanced by Crank–Nicolson.
`iso13790_zone` derives its parameters from the same construction as a
`BuildingZone`:

- ``H_{tr,op} = \sum U_j A_j``, and ``H_{tr,em} = (1/H_{tr,op} - 1/H_{tr,ms})^{-1}``;
- ``C_m = \sum κ_j A_j`` and ``A_m = C_m^2/\sum A_j κ_j^2`` (§12.3.1).

Here ``κ_j`` is the ISO 13786 Annex A simplified internal areal heat
capacity. It is the ``ρcd`` of the layers from the room side up to the
smallest of three depths: half the element thickness, the first
insulating layer, and 10 cm.

The 5R1C model uses the same steady conductances as the resolved zone
but a single capacitance, so it is a meaningful check. It is not a
substitute for data.

## 4. Validation

All numbers below are measured by `test/era5_building_energy.jl` (the resolved
zone vs 5R1C: `validation/era5_building_energy.jl`).

| Check | Reference | Result |
|:------|:----------|:-------|
| Single air node, ventilation only | ``τ = C_a/H_{ve}`` = 1800 s (2 h⁻¹) | discrete solution ``(1+Δt/τ)^{-n}`` to 1e-10; fitted τ 1802.2 s (the backward-Euler bias ``Δt/2τ`` = 0.125 %) |
| Air + 5 mm steel sheet (Bi = 6e-4) | two-node lumped eigen-solution, τ = 404 s and 39 444 s | max error 1.1e-3 K on a 10 K step over 2 days, Δt = 5 s |
| Air + 20 cm concrete slab, ventilated | plane-wall first mode ``β\tanβ = Bi`` (Incropera §5.5), Bi = 0.435, τ₁ = 42.42 h | fitted τ₁ 42.43 h (−0.009 %) |
| Steady state: walls, roof, ground slab, glazing, ventilation with heat recovery, radiative and convective gains, HVAC | independent series–parallel network | agreement to 1e-8 K |
| Energy conservation, 10 days: facets (MOST and fixed-h), sol-air wall, ground, internal mass, HVAC | ``ΔE = \int \sum Q\,dt`` | relative error 3.3e-14 |
| Reuse exactness | the facet and the column stepped directly at the solved ``T_{air}`` | ≤ 1e-9 K |
| 5R1C implementation | independent three-node Kirchhoff solve; CN decay factor | 1e-8 K; exact ratio |

For the resolved zone against 5R1C, the case is an 8 × 6 × 2.7 m box
with an outdoor temperature of 15 ± 5 °C, office-hours gains and a
noon solar peak. The heavy box has brick walls and a brick partition;
the light box has timber-frame walls. Both are run to a periodic regime
over 150 days.

The "operative" temperatures compared are each model's own definition:
``(T_{air} + T_{mrt})/2`` for the resolved zone and ISO 13790 eq. C.12,
``0.3 θ_{air} + 0.7 θ_s``, for 5R1C.

| Box | Operative mean, resolved / 5R1C | Amplitude, resolved / 5R1C | Peak time |
|:----|:----|:----|:----|
| Heavy | 18.541 / 18.552 °C | 0.605 / 0.848 K | 16.2 / 18.5 h |
| Light | 22.598 / 22.445 °C | 1.357 / 1.135 K | 15.4 / 16.5 h |

**Means.** The means agree to 0.3 % (heavy) and 2.1 % (light) of the
indoor–outdoor excess. The two networks share their steady
conductances; they differ only in where the gains and the glazing
attach.

**Amplitudes.** The amplitudes differ by −29 % and +20 %. With one
lumped capacitance, 5R1C under-damps the heavy box and over-damps the
light one. Neither sign is a bug in either model. It is the known limit
of a one-capacitance representation of distributed mass.

**Peak times.** The single capacitance also puts the heavy box's peak
about 2 h later.

The test bounds on this comparison are 3 % of the excess for the mean,
40 % for the amplitude and 3 h for the peak. They are envelopes set
after measurement, not physics expectations.

**Not validated against observations.** No building measurements were
used. The phase target, simulated against measured energy demand for an
instrumented building, needs occupancy and HVAC (issues #32 and #33).
ASHRAE 140 (BESTEST) cases need its weather file and were not run.

## 5. Scope

This page covers issue #31 only. HVAC control and its exhaust heat
(#32) are on [HVAC demand and exhaust heat](hvac_exhaust.md). Occupancy schedules and the validation against a measured
building (#33), anthropogenic heat coupling to the canopy (#29, see
[Anthropogenic heat from the building energy model](anthropogenic_heat.md)),
and glazing optics (#27) are all separate issues. `Q_hvac`, `Q_solar` and
`FacetElement` are the hooks they plug into.

## References

- Bueno, B., Norford, L., Pigeon, G. & Britter, R. (2011). Combining a detailed building energy model with a physically-based urban canopy model. *Boundary-Layer Meteorol.* 140, 471-489, doi:10.1007/s10546-011-9620-6.
- ISO 13790:2008, §7.2.2, §9.3, §12.3 and Annex C.
- ISO 13786:2007, Annex A.
- ISO 6946:2017, Table 7.
- Bergman, T. L. et al. (2011). *Fundamentals of Heat and Mass Transfer*, 7th ed., §5.2 and §5.5.
