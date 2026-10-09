# Passive scalar advection in 2D

Advects a Gaussian blob in a prescribed rigid-body-rotation flow.
This is the Phase 0 end-to-end smoke test — it exercises the mesh,
halo fill, upwind advection kernel, and the SSP-RK3 stepper.

Location: `examples/configs/scalar_advection_2d.toml` (the `advection`
driver, [`Aeolus.RunDrivers.AdvectionDriver`](@ref); formerly the script
`examples/scalar_advection_2d.jl`, which it reproduces bit for bit,
`test/config_run_drivers.jl`).

Run with:

```bash
$ julia --project -m Aeolus run examples/configs/scalar_advection_2d.toml
```

The numbers below are in the run folder's `output/results.toml` (`t`,
`mass`, `max`, `peak_x`, `peak_y`); this run gives them to the printed
digits.

## Setup

  * Domain `[-½, ½]²`, single cell in `z`.
  * `Nₓ × Nᵧ = 128 × 128`.
  * Prescribed velocity `u = -ω y`, `v = ω x` (rigid rotation).
  * Initial `ϕ`: off-center Gaussian at `(0.25, 0)` with `σ = 0.08`.
  * `ω = 2π` → one full revolution in `t = 1`.
  * Time step from Courant condition, CFL = 0.4.

## Expected output

```text
2D advection: N = 128 × 128, dt = 9.94e-04, nsteps = 1006
t = 0.000    mass = 4.018e-02    max = 0.998
t = 0.124    mass = 4.018e-02    max = 0.869
t = 0.249    mass = 4.018e-02    max = 0.770
...
t = 0.994    mass = 4.018e-02    max = 0.463
peak location after 1 rev: (0.230, -0.004)   expected: (0.250, 0.000)
```

## What to look for

  * **Mass conservation** — exact to floating-point precision (upwind
    is *conservative*, and periodic BCs make surface fluxes cancel).
  * **Peak location** — close to the starting point (small phase error
    from 1st-order upwind numerical dispersion).
  * **Peak amplitude decay** — 1.0 → 0.46 over one revolution. This is
    the numerical diffusion of upwind. Once the WENO-5 kernel lands
    (Phase 2) the amplitude drop will shrink dramatically.
