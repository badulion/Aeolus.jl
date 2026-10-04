# Transition diagnostics — Transition

`Transition` owns the laminar/turbulent and intermittency diagnostics of
Era VI: the turbulent fraction ``γ(z)`` of the vorticity-fluctuation
indicator (`turbulent_fraction`, `turbulent_indicator!`,
`laminar_mask!`), the time-series `intermittency_factor`, the flux
Richardson number with resolved and SGS fluxes (`flux_richardson`), the
buoyancy Reynolds number, the Sun et al. (2012) ``V_{TKE}``, the grid
Reynolds number, the supercritical fraction of a ``Ri_g`` field, and a
log-linear `growth_rate` fit. It defines no Richardson number: ``Ri_g``
comes from [TurbulenceStability](turbulence_stability.md). The theory
and the Kelvin–Helmholtz / Miles–Howard validation are on
[Laminar–turbulent transition](../../theory/transition.md#Laminar–turbulent-transition).

```julia
using Aeolus
γ  = turbulent_fraction(sim)                    # γ(z), plane-normalised |ω′|/⟨|ω|⟩ > 0.2
r  = flux_richardson(sim)                       # r.Ri_f, r.B, r.P (resolved + SGS)
fit = growth_rate(ts, A)                        # fit.rate, fit.r2, fit.window
intermittency_factor(series; threshold = 0.1)
```

```@autodocs
Modules = [Aeolus.Transition]
```
