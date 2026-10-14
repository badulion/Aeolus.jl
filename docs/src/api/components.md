# Components

`Aeolus.Components` (ONE-a, #450) is the component interface and the
fixed stage pipeline of the single model. A [`Model`](@ref
Aeolus.Components.Model) holds [`Domain`](@ref Aeolus.Components.Domain)s;
a domain holds a dynamical core, its time scheme, its drivers and a
concretely typed tuple of components, and `step!(model, dt)` runs the
pipeline. The design is [One model](../design/single_model.md); what is
built, and how stages 6–8 run inside the core, is on
[Components and the stage pipeline](@ref).

```julia
using Aeolus, Aeolus.Components
mesh = RegularGrid(; size = (32, 32, 32), extent = (3200.0, 3200.0, 1600.0))
sim  = BoussinesqSimulation(mesh; θ₀ = 300.0, bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
d0   = Domain(sim; validate = false)                 # the components need the domain to build
comps = Components.build_components((CentredAdvection => (; scheme = :centred),
                                     SGS => (model = :smagorinsky, c_s = 0.17, pr_t = 1/3, stability = :none),
                                     Coriolis => (; f = 1e-4)), d0)
model = Model(Domain(d0; components = comps))
step!(model, 2.0)
```

Several domains are a nest (ONE-f, #455): [`nested_model`](@ref
Aeolus.Components.nested_model) couples them with one
[`NestLink`](@ref Aeolus.Components.NestLink) per child (its parent, its
`Nesting.NestedGrid`, its feedback, two-way by default, and optional
synthetic inflow turbulence), and `step!(model, dt)` (or
`step_adaptive!(model)`) takes one window of the nest engine
([`nest_window!`](@ref Aeolus.Components.nest_window!), NEST-p2 #482):
the root steps, every child subcycles through its own pipeline, the
two-way children feed back, and a failed step retries the window. A config builds it from its `[[domain]]` blocks; see
[Run configuration](../design/run_configuration.md).

```@autodocs
Modules = [Aeolus.Components]
```
