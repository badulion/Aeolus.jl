# Stability contract — TurbulenceStability

`TurbulenceStability` is the stability contract: the
gradient-Richardson stability functions ``f_m(Ri)``, ``f_h(Ri)`` as a
dispatch hierarchy (`AbstractRiStability`), the one definition of the
squared buoyancy frequency from a θ difference
(`buoyancy_frequency_squared`) and of the gradient Richardson number
(`gradient_richardson`, with its ``S^2`` floor `RI_SHEAR2_FLOOR`). The
theory, including the derivation of the MOST-consistent factors, is on
[Stability functions and the gradient Richardson number](../../theory/turbulence_stability.md).

Implementers: `LillyCutoff` (Lilly 1962; Mason 1989), `BrownEtAl1994`
(the LEM / MONC functions), `ShortTail` (the radiation-fog column's
function), `LongTail` (Louis 1979), and in `SurfaceExchange`
`MOSTConsistent`, which inverts ``Ri = ζφ_h/φ_m^2`` for any
`AbstractStabilityFunctions`.

The pointwise fields live in `Operators`:
`buoyancy_frequency_squared!`, `strain_rate_squared!` and
`gradient_richardson!` (see [Operators](../operators.md)).

```julia
using Aeolus
sf = BrownEtAl1994()
Ri = gradient_richardson(2e-5, 1e-4)          # N², S² → 0.2
K  = (0.2 * 10.0)^2 * sqrt(1e-4) * stability_factor_m(sf, Ri)
stability_factor_m(MOSTConsistent(), Ri)      # the Businger–Dyer–Beljaars–Holtslag value
```

`boundary_layer_height(z, profile; method)` is the only ``z_i``
diagnostic in the package: the parcel method, the bulk Richardson
number, the maximum θ gradient and the minimum heat flux (#247; see
[The boundary-layer height](../../theory/turbulence_stability.md#The-boundary-layer-height)).

```@autodocs
Modules = [Aeolus.TurbulenceStability]
```

## The SurfaceExchange implementer

`MOSTConsistent`, `most_gradient_richardson` and
`most_stability_parameter` belong to `Aeolus.SurfaceExchange` and are
documented with it, on [Shared contract — SurfaceExchange](../surface/surface_exchange.md).
