module ML

# Phase 20 (component) — Pluggable SGS closure via a user-supplied
# function. This is the abstraction hook for ML-augmented turbulence
# closures (Kochkov 2021, Frezat 2022): swap in any callable that
# writes ν_t as a function of (u, v, w). The callable can be a small
# Julia neural net (`Lux.jl` / `Flux.jl`) or a hand-crafted alternative
# closure.

using ..Grids: RegularGrid, interior
using ..Fields: Field
using ..Simulations: BoussinesqSimulation
using ..Operators: AbstractEddyViscosityClosure
import ..Operators: sgs_viscosity!

export MLSGS, apply_mlsgs!

"""
    MLSGS(fn)

Wraps a user function `fn(νt, u, v, w, mesh)` that populates `νt::Field`
in-place from the current velocity state. Signature is intentionally
minimal — any framework can provide the callable. Register on a
`BoussinesqSimulation` by:

```julia
mlsgs = MLSGS((νt, u, v, w, m) -> my_neural_net(νt, u, v, w, m))
apply_mlsgs!(sim, mlsgs)      # populates sim.νt from the callable
```

Aeolus then feeds `sim.νt` into the existing `add_eddy_viscosity_*!`
and `add_eddy_diffusion_scalar!` kernels — the ML kernel is a *drop-in
replacement* for `compute_smagorinsky_νt!`.

An `MLSGS` is an [`AbstractEddyViscosityClosure`](@ref
Aeolus.Operators.AbstractEddyViscosityClosure) of the Era VI SGS contract, so
it is passed to any core directly, `step!(sim, dt; sgs = mlsgs)`, with
`κt = νt / sim.Pr_t`. It is stateless.
"""
struct MLSGS{F} <: AbstractEddyViscosityClosure
    fn::F
end

sgs_viscosity!(νt, m::MLSGS, sim) = (m.fn(νt, sim.u, sim.v, sim.w, sim.mesh); νt)

"""
    apply_mlsgs!(sim, mlsgs)

Invoke the wrapped ML function on the current state, populating
`sim.νt`. No side-effects on velocity; the caller then applies the
usual eddy-viscosity kernels.
"""
function apply_mlsgs!(sim::BoussinesqSimulation, m::MLSGS)
    m.fn(sim.νt, sim.u, sim.v, sim.w, sim.mesh)
    return sim.νt
end

end # module
