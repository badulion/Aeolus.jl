# 3D Taylor–Green vortex

The canonical benchmark for incompressible LES cores. The initial
condition is analytically divergence-free, and (in the linearized
limit) the kinetic energy decays as `exp(-6νt)`.

Location: `examples/taylor_green_vortex.jl`.

```bash
$ julia --project=. examples/taylor_green_vortex.jl
```

## Setup

```julia
u(x, y, z, 0) =  sin(x) cos(y) cos(z)
v(x, y, z, 0) = -cos(x) sin(y) cos(z)
w(x, y, z, 0) =  0
```

on `32³` over `[0, 2π]³`, triply-periodic, molecular viscosity
`ν = 10⁻²` (Re ≈ 100), for `T_end = 2`. Time step from `min(convective,
viscous)` CFL with factor 0.4.

## Expected output

```text
TGV N = 32³   ν = 1.00e-02   dt = 7.69e-02   nsteps = 26
t = 0.000   KE = 3.10063e+01   max|div| = 5.33e-15
t = 0.154   KE = 3.07439e+01   KE/KE₀ = 0.9915   max|div| = 1.71e-15
t = 0.462   KE = 3.02204e+01   KE/KE₀ = 0.9747   max|div| = 1.71e-15
t = 0.923   KE = 2.94203e+01   KE/KE₀ = 0.9488   max|div| = 1.50e-15
t = 1.385   KE = 2.85746e+01   KE/KE₀ = 0.9216   max|div| = 1.55e-15
t = 1.846   KE = 2.76550e+01   KE/KE₀ = 0.8919   max|div| = 1.67e-15
t = 2.000   KE = 2.73280e+01   KE/KE₀ = 0.8814   max|div| = 1.55e-15
```

## What to look for

  * **Divergence** at `10⁻¹⁵` throughout — the projection is working
    correctly.
  * **Energy decay**: at `t = 2`, `KE/KE₀ = 0.881`. Linearized-Stokes
    analytic value is `exp(-6ν t) = exp(-0.12) = 0.887`. The `~0.7%`
    discrepancy is real physics — nonlinear energy transfer to
    smaller scales.
  * **Stability**: no blow-up, energy strictly monotonically
    decreasing.

For a full validation you would push to a much later time and compare
against the enstrophy peak at `t ≈ 9` from Brachet et al. (1983). Not
yet in Aeolus's benchmark suite.
