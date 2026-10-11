# Budgets

`Aeolus.BudgetSystem` (ONE-c, #452) is the one budget system of the
model: energy (the surface classes and the heat of the air), water,
momentum and tracers, in one [`Budgets`](@ref
Aeolus.BudgetSystem.Budgets) container. It replaces three systems that
grew apart: the urban case's energy and water ledger, its air-heat ledger
(#253) and the SBL run type's mean-momentum budget (P58-d, #194).

- **Measured budgets.** A [`LevelMeter`](@ref Aeolus.BudgetSystem.LevelMeter)
  measures the per-level content of a field in the air. A
  [`StageBudget`](@ref Aeolus.BudgetSystem.StageBudget) attributes every
  change of it to the stage entries of a step and the dynamics, with no
  reporting by the components. This is the stage-context measurement of
  the old air-heat ledger, and it is now the core mechanism for every
  measured quantity. The model's domains use it
  ([`with_budgets`](@ref Aeolus.Components.with_budgets)), and so do the
  urban case and its nest parent.
- **Reported terms.** Components report surface energy terms per surface
  class ([`add_surface!`](@ref Aeolus.BudgetSystem.add_surface!)), water
  sources ([`add_water!`](@ref Aeolus.BudgetSystem.add_water!)) and named
  heat ([`add_heat!`](@ref Aeolus.BudgetSystem.add_heat!),
  [`book!`](@ref Aeolus.BudgetSystem.book!)).

Each budget closes to round-off when nothing escapes: `storage − Σ terms`
([`stage_budget`](@ref Aeolus.BudgetSystem.stage_budget),
[`surface_budget`](@ref Aeolus.BudgetSystem.surface_budget),
[`water_budget`](@ref Aeolus.BudgetSystem.water_budget)). The
mechanism is described on
[Components and the stage pipeline](@ref "Budgets: the stage-context measurement").

```julia
using Aeolus, Aeolus.BudgetSystem
mesh = RegularGrid(; size = (16, 16, 16), extent = (1600.0, 1600.0, 800.0))
θ = Field(CellCenter(), mesh)
b = StageBudget(; bins = 1)
budget_start!(b, LevelMeter(mesh, fill(1.2, 16), ones(16), 1004.6))   # ρ̄, Π̄, c_p
budget_begin_step!(b, θ)
budget_open!(b, :radiation, :sun, θ); interior_view(θ) .+= 0.1; budget_close!(b, θ)
budget_end_step!(b, θ, 60.0)
stage_budget(b).terms[(:radiation, :sun)]       # 1.2 · 1004.6 · 0.1 · 800 J m⁻²
```

```@autodocs
Modules = [Aeolus.BudgetSystem]
```
