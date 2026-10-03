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
[Transition and wall-bounded flow](../../theory/transition.md).

```julia
using Aeolus
γ  = turbulent_fraction(sim)                    # γ(z), plane-normalised |ω′|/⟨|ω|⟩ > 0.2
r  = flux_richardson(sim)                       # r.Ri_f, r.B, r.P (resolved + SGS)
fit = growth_rate(ts, A)                        # fit.rate, fit.r2, fit.window
intermittency_factor(series; threshold = 0.1)
```

It also owns the spatially developing flat-plate boundary layer (P59-d,
#185): the Blasius solution (`blasius_solution`, `blasius_profile`,
`blasius_velocity`), the fringe region (`FringeRegion`, `add_fringe!`,
`fringe_forcing`, `blasius_target`, `blasius_freestream`), free-stream
turbulence (`FreeStreamDisturbance`, `disturbance_velocity!`,
`disturbance_fringe_update`), the integral diagnostics as functions of
`x` (`skin_friction`, `displacement_thickness`, `momentum_thickness`,
`shape_factor`, `boundary_layer_edge`) and the onset finder
(`transition_onset`, with the Mayle and Abu-Ghannam & Shaw
correlations); theory in
[Flat-plate transition](../../theory/transition.md#Flat-plate-transition).

```julia
c = flat_plate_setup()                          # 2-D Blasius plate, Re_x 1.35e4 → 1e5
c.sim.u.data .= 1
for n in 1:1200; step_rk3!(c.sim, 1.0; body_force! = c.forcing!); end
cf = skin_friction(c.sim; U_e = 1.0); H = shape_factor(c.sim)
```

```@autodocs
Modules = [Aeolus.Transition]
```
