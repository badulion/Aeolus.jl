# Quick start

Aeolus.jl gives you a small, composable set of building blocks. The
canonical entry point is [`BoussinesqSimulation`](@ref
Aeolus.Simulations.BoussinesqSimulation), which owns a mesh, boundary
conditions, a pressure solver, and the prognostic state
`(u, v, w, p, θ)`.

## The 20-line Taylor–Green vortex

```julia
using Aeolus

# 1. Mesh
mesh = RegularGrid(; size=(32,32,32), extent=(2π,2π,2π), halo=(2,2,2))

# 2. Simulation (triply periodic, molecular viscosity only)
sim = BoussinesqSimulation(mesh; ν = 1e-2)

# 3. Initial condition — the 3D Taylor–Green vortex
Aeolus.Fields.set!(sim.u, (x,y,z) ->  sin(x)*cos(y)*cos(z))
Aeolus.Fields.set!(sim.v, (x,y,z) -> -cos(x)*sin(y)*cos(z))
Aeolus.Fields.set!(sim.w, (x,y,z) ->  0.0)

# 4. Choose a stable dt (viscous + convective CFL)
dx = mesh.dx
dt = min(0.4dx / 1.0, 0.4dx^2 / (2sim.ν))

# 5. Advance
for n in 1:100
    step!(sim, dt)
end

@show sim.t, kinetic_energy(sim), max_divergence(sim)
```

The full example is the config `examples/taylor_green_vortex/taylor_green_vortex.toml`:
`julia --project -m Aeolus run examples/taylor_green_vortex/taylor_green_vortex.toml`
writes the diagnostics every 0.1 s to the run folder.

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
