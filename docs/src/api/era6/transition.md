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

Synthetic inflow turbulence (N-inflow-a, #252): the generator contract
(`AbstractInflowGenerator`, `unit_fluctuations!`, `generator_state`,
`restore_generator_state!`), the Lund transform (`lund_coefficients`,
`apply_lund!`), the Xie & Castro digital filter (`DigitalFilterGenerator`,
`digital_filter_kernel`) and free-stream turbulence as a second provider
(`FreeStreamInflow`, `disturbance_velocity`, `disturbance_covariance`);
theory in [Synthetic inflow turbulence](../../theory/transition.md#Synthetic-inflow-turbulence).

```julia
gen = DigitalFilterGenerator(64, 40, 15.625, 6.25; L_s = 62.5, L_z = 25.0, T = 30.0)
a   = lund_coefficients([0.514 0 -0.09; 0 0.332 0; -0.09 0 0.141])   # R_ij, m² s⁻²
u, v, w = zeros(64, 40), zeros(64, 40), zeros(64, 40)
unit_fluctuations!(u, v, w, gen, Δt); apply_lund!(u, v, w, a)
```

```@autodocs
Modules = [Aeolus.Transition]
```
