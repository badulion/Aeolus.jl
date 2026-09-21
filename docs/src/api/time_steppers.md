# TimeSteppers

Generic time-integration schemes. Currently only Shu-Osher SSP-RK3 is
provided; a projected-RK3 variant is planned in Phase 2. The `step!`
function is a generic function shared with [`Simulations`](@ref
Aeolus.Simulations) — the Boussinesq driver adds its own method to it.

```@docs
Aeolus.TimeSteppers.RK3
Aeolus.TimeSteppers.step!
```
