# Quick start

Aeolus is **one model** ([One model](design/single_model.md)). A run is a
configuration — a TOML file that says which modules run and how they are
set up — plus its static and dynamic input drivers (NetCDF). `aeolus run`
always runs the same model; the sections of the file switch its
components on.

## Run an example

Every example is a case directory, `examples/<case>/`, holding its
configuration `<case>.toml` (and the drivers in `input/` it needs):

```bash
julia --project -m Aeolus run examples/dry_convective_bl/dry_convective_bl.toml
```

The configuration, abridged (the file comments every choice):

```toml
schema_version = 2

[run]
name = "dry_convective_bl"
seed = 1

[grid]
size = [32, 32, 32]
extent = [3200.0, 3200.0, 1600.0]

[boundaries]
z = "free_slip"

[sgs]
closure = "smagorinsky"
c_s = 0.17

[surface_theta_flux]
bottom = 0.24                   # K m/s

[sponge]
alpha_max = 0.01
z_start = 1200.0

[initial]
theta = { kind = "mixed_layer_lapse", value = 300.0, z_i = 800.0, lapse = 0.003 }
perturbation_amplitude = 0.1
perturbation_depth = 100.0

[time]
scheme = "euler"
dt = 5.0
duration = 300.0

[checks]
finite = { required = true }
energy_closure = { required = true }
momentum_closure = { required = true }
```

RUN_FOLDER_PLACEHOLDER

## Write your own

A file with no sections at all is a minimal dry LES (32³ on 3.2 km ×
3.2 km × 1.6 km, Smagorinsky, adaptive RK3, one hour); every other module
is off until its section enables it. Unknown keys are errors, with a
suggestion:

```bash
julia --project -m Aeolus check my_case.toml       # validate only
julia --project -m Aeolus resolve my_case.toml     # print the resolved file: every option of every component
julia --project -m Aeolus options sgs              # the option reference of one section
julia --project -m Aeolus run my_case.toml --set grid.size=[16,16,16]
```

Geometry and forcing that TOML cannot hold (buildings, land use, boundary
planes, large-scale forcing tables) come from the static and dynamic
drivers, which the companion package AeolusPrep.jl writes; see
[Run configuration](design/run_configuration.md). A script that analyses
a run reads its folder (`run.toml`, `diagnostics.csv`/`.nc`,
`output/output.nc`, `output/results.toml`); it never builds or steps the
model (`test/examples_enforcement.jl`).

## Under the hood: the dynamical core

A run's `Model` holds one `Domain` per grid; each domain holds its core,
a [`BoussinesqSimulation`](@ref Aeolus.Simulations.BoussinesqSimulation)
or an `AnelasticSimulation`, and steps it through the fixed stage
pipeline ([Components and the stage pipeline](design/components.md)). The
physics modules' tests drive the core directly; the rest of this page
describes it for that use, not as the way to set up a run.

## Anatomy of a `BoussinesqSimulation`

The struct owns:

| Group           | Fields                                        |
| :-------------- | :-------------------------------------------- |
| Prognostic      | `u @ FaceX`, `v @ FaceY`, `w @ FaceZ`, `p, θ @ CellCenter` |
| Tendencies      | `tu, tv, tw @ Face…`, `tθ @ CellCenter`       |
| Scratch         | `φ, div_buf @ CellCenter`, `dφdx/dφdy/dφdz`, `lap_*`, `νt` |
| Parameters      | `ν, θ₀, g_accel, f_coriolis, C_s, Pr_t`        |
| Flags           | `has_viscosity, has_buoyancy, has_coriolis, has_sgs` |

Each physics is enabled by setting the corresponding parameter to a
nonzero value in the constructor:

```julia
sim = BoussinesqSimulation(mesh;
    ν          = 1e-2,      # molecular viscosity
    θ₀         = 300.0,     # → buoyancy on
    g_accel    = 9.81,
    f_coriolis = 1e-4,      # → Coriolis on
    C_s        = 0.17,      # → Smagorinsky-Lilly SGS on
    Pr_t       = 1/3,
)
```

A parameter of `0` disables that physics. All allocation happens in the
constructor; `step!` does no heap allocation on the hot path.

## The `step!` algorithm

One `step!(sim, dt)` executes an Euler-predictor / pressure-projection
scheme:

1. Fill halos of `u, v, w, θ`.
2. Zero all tendency buffers.
3. Add momentum advection ([`momentum_advect!`](@ref
   Aeolus.Operators.momentum_advect!)) and scalar advection
   ([`advect_centered!`](@ref Aeolus.Operators.advect_centered!)).
4. Optionally add molecular viscosity `ν ∇²`.
5. Optionally add SGS: compute `νₜ`, then eddy viscosity + eddy diffusion.
6. Optionally add buoyancy `g(θ-θ₀)/θ₀` to `w`-tendency.
7. Optionally add Coriolis `+f v` on `u`, `-f u` on `v`.
8. Predictor step `u* = u + Δt · F`.
9. `div_buf = ∇·u*` at cell centers.
10. Pressure solve `∇²φ = div_buf` via `sim.solver`.
11. Correct `u = u* - ∇φ` at faces.
12. Advance θ, record physical pressure `p = φ/Δt`, tick the clock.

See [Theory → Fractional step](theory/fractional_step.md) for the maths.

## Diagnostics

```julia
kinetic_energy(sim)      # ½ ∫ (u² + v² + w²) dV
max_divergence(sim)      # ‖∇·u‖∞ over interior
max_νt(sim)              # max SGS eddy viscosity
mean_θ_profile(sim)      # horizontally averaged θ(z)
```

## Next steps

  * Read the [Theory → Overview](theory/overview.md).
  * Skim the [Design → Architecture](design/architecture.md) if you want
    to add a new physics module or backend.
  * Browse the [API reference](api/grids.md).
