"""
    Aeolus.BudgetSystem

The one budget system of the model (ONE-c, #452; the design is
`docs/src/design/single_model.md`, "Budgets and checks"): energy (the
surface classes and the heat of the air), water, momentum and tracers,
in one [`Budgets`](@ref) container. It replaces the urban case's energy
and water ledger, its air-heat ledger and the SBL run type's momentum
budget (since ONE-g #456 the SBL cases measure momentum like every other
quantity), which were three separate systems.

* [`LevelMeter`](@ref) measures the per-level content of a field in the
  air (the heat of the air, the mass of water or a tracer, momentum);
* [`StageBudget`](@ref) attributes every change of one measured quantity
  to the stage entries of a step and the dynamics, automatically: the
  stage-context measurement of the urban air-heat ledger (#253), now the
  core mechanism of every measured budget. The model's domains attribute
  their stages through it (`Components`), and so does the urban case;
* [`Budgets`](@ref) holds the measured budgets, the surface energy
  balance per class and the water of the air from reported sources
  ([`add_surface!`](@ref), [`add_water!`](@ref), [`add_heat!`](@ref)).

Components book their terms through `Components.budget!`; the closures
are reported by [`stage_budget`](@ref), [`surface_budget`](@ref) and
[`water_budget`](@ref).
"""
module BudgetSystem

using Printf: @printf, @sprintf
using ..Backends: backend, plane_reduce!, plane_partials!
using ..Anelastic: MassWeightedSimulation, constraint_weights

export LevelMeter, measure_levels!, solid_levels!, measure, air_heat, layer_fraction, level_thickness,
       split_transport, transport_conserved_weight,
       StageBudget, budget_start!, budget_begin_step!, budget_end_step!, budget_open!, budget_close!, budget_skip!,
       budget_discard!, book!, stage_budget, stage_report, report_order,
       Budgets, energy, measured_quantities, add_term!, water_tracer!, add_surface!, add_water!, add_heat!,
       heat_budget, level_sums, water_content, water_split, water_split_levels, surface_budget,
       water_budget, budget_report, SURFACE_CLASS_NAMES, SURFACE_CORE_TERMS

include("budgets/meter.jl")
include("budgets/stage_budget.jl")
include("budgets/budgets.jl")

end # module
