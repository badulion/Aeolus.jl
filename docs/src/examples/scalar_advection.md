# Passive scalar advection in 2D

Advects a Gaussian column of θ in a prescribed rigid-body-rotation flow.
This is the Phase 0 end-to-end smoke test — it exercises the mesh, the
halo fill, the core's scalar advection and the SSP-RK3 stepper, with the
velocity held fixed by the `[kinematic_flow]` component (ONE-l, #475).

Location: `examples/scalar_advection_2d/scalar_advection_2d.toml` (a
configuration of the model; formerly the script
`examples/scalar_advection_2d.jl`, then the `advection` run type).
`test/config_cases.jl` runs it at 32² for a quarter revolution.

Run with:

```bash
$ julia --project -m Aeolus run examples/scalar_advection_2d/scalar_advection_2d.toml
```

The run writes `output/output.nc`: θ every 1/8 revolution and the time
series of its domain mean.

## Setup

  * Domain `[0, 1]²`, single cell in `z`, periodic.
  * `Nₓ × Nᵧ = 128 × 128`.
  * Prescribed velocity `u = -ω (y − ½)`, `v = ω (x − ½)` (rigid
    rotation about the centre; on the staggered grid `u` depends on `y`
    only and `v` on `x` only, so it is discretely divergence-free).
  * Initial θ: off-centre Gaussian at `(0.75, 0.5)` with `σ = 0.08`
    (`[initial] theta = {kind = "gaussian", …}`), no buoyancy
    (`dynamics.theta0 = 0`).
  * `ω = 2π` → one full revolution in `t = 1`.
  * `dt = 0.4 Δx / (ω/2)` (CFL 0.4 at the box edge): 1006 steps.

The scheme is the core's second-order centred flux form, not the
first-order upwind kernel of the former run type (a stated change of the
port): the integral is conserved, the peak is not diffused away, and the
dispersive undershoot is a few thousandths.

## Expected output

Measured (2026-10-12, the 128² configuration):

```text
t = 0.000    mean θ = 4.017679e-02    max = 0.998   min =  0.0000
t = 0.250    mean θ = 4.017679e-02    max = 0.998   min = -0.0019
t = 0.500    mean θ = 4.017679e-02    max = 0.998   min = -0.0020
t = 0.750    mean θ = 4.017679e-02    max = 0.998   min = -0.0024
t = 1.000    mean θ = 4.017679e-02    max = 0.996   min = -0.0033
peak after 1 rev: (0.746, 0.496)   expected: (0.750, 0.500)
```


## What to look for

  * **Conservation** — the mean of θ is constant to round-off (the flux
    form on a periodic box; `test/config_cases.jl` asserts 1e-13).
  * **Peak location** — back at its start after one revolution, within a
    cell (0.746, 0.496 against 0.75, 0.5).
  * **Peak amplitude** — 0.998 → 0.996: the centred scheme has no
    numerical diffusion to speak of (the former upwind kernel took it to
    0.46); the price is the dispersive undershoot behind the blob (min
    −0.003).
