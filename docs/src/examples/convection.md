# Buoyant convection

End-to-end exercise of the full Phase 1a system: staggered momentum
advection + fractional-step projection + Smagorinsky-Lilly SGS +
buoyancy + scalar transport, all coupled.

Location: `examples/configs/buoyant_convection.toml` (the `simulation`
driver; `examples/buoyant_convection.jl` is a three-line runner for it).

```bash
$ julia --project -m Aeolus run examples/configs/buoyant_convection.toml
```

The θ profile is the `mixed_layer_lapse` family with `z_i = 0`
(`300.5 − 1.0 z`), and the 1 mK noise comes from `Xoshiro([run] seed)`,
so the trajectory is not bit for bit the former script's (which used
`θ₀ + Δθ (½ − z/L)` and the global RNG); the dt (5/97 s) and every
parameter are the same. The run writes `diagnostics.csv`, every 0.5 s:

## Setup

Unstably stratified θ profile plus tiny random noise, triply periodic
(we do not yet have walls — Phase 1b):

  * `θ(z) = θ₀ + Δθ (½ - z/L) + noise`
  * `θ₀ = 300 K`, `Δθ = 1 K`, `L = 1 m`
  * `32³` grid
  * SGS on: `Cₛ = 0.17`, `Prₜ = 1/3`
  * Convective velocity scale `w* = √(g Δθ / θ₀ · L) ≈ 0.18 m/s`
  * CFL 0.3, `T_end = 5 s`

## Expected output

The former script's printout (the config's run gives
`ke_mean` 5.64e-12 m² s⁻² at t = 0.567 s, 4.25e-10 at t = 4.69 s, with
`max_div` ≈ 5e-16 throughout):

```text
Convection: N = 32³   w* = 0.181 m/s   dt = 5.15e-02   nsteps = 97
t = 0.000   KE = 0.000e+00   max νₜ = 0.000e+00   max|div| = 0.00e+00
t = 0.464   KE = 3.77e-12    max νₜ = 8.67e-09    max|div| = 5.44e-16
t = 0.928   KE = 1.51e-11    max νₜ = 1.85e-08    max|div| = 4.98e-16
t = 2.320   KE = 9.50e-11    max νₜ = 4.91e-08    max|div| = 5.49e-16
t = 4.639   KE = 4.15e-10    max νₜ = 1.14e-07    max|div| = 5.56e-16
```

## What to look for

  * **Divergence at machine precision throughout the coupled run.** All
    five physics blocks (advection, SGS, buoyancy, viscous, Coriolis-
    absent-here) are active; the projection still kills divergence.
  * **KE grows from noise.** Buoyancy converts stratification into
    motion; the linear-instability phase amplifies perturbations
    exponentially.
  * **`νₜ` responds to shear.** `max νₜ` grows from 0 as velocity
    gradients develop — evidence the SGS model is dynamic.

This is not (yet) a full DCBL validation — that needs walls and MOST
(Phase 1b). It is a coupled-system correctness check.
