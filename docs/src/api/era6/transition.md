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

Inflow statistics (N-inflow-b, #258): the contract (`InflowStatistics`,
`inflow_stresses`, `inflow_lengths`, `update_inflow_statistics!`,
`inflow_projection_log`, `realizable_stress`, `inflow_lund_coefficients`),
the statistics of an Aeolus LES parent (`ParentResolvedStatistics`,
`inflow_stress_parts`: resolved, scale-gap band, SGS from
`Operators.sgs_tke!`) and the similarity profiles for a parent without
resolved turbulence (`ParametrizedStatistics`, `parametrized_scales`);
theory in [Inflow statistics](../../theory/transition.md#Inflow-statistics-(N-inflow-b)).

```julia
src = ParentResolvedStatistics(parent; closure = tke, τ_avg = 600.0, refine = (2, 2, 4),
                               levels = 1:20)
update_inflow_statistics!(src, Δt)              # after every parent step (or at a cadence)
z   = [(k - 0.5) * 6.25 for k in 1:64]          # the child's levels
gen = DigitalFilterGenerator(64, 64, 15.625, 6.25; inflow_lengths(src, z, t)...)
a   = inflow_lund_coefficients(src, z, t)       # realizable, projections booked
unit_fluctuations!(u, v, w, gen, Δt); apply_lund!(u, v, w, a)
```

```@autodocs
Modules = [Aeolus.Transition]
```
