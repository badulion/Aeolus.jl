"""
    Aeolus.Components

The component interface and the fixed stage pipeline of the single model
(ONE-a, #450; the design is `docs/src/design/single_model.md`).

A **component** ([`AbstractComponent`](@ref)) is a reusable module
instance configured by one TOML section: a thin adapter from a physics
module (the equations live there) to the pipeline. A [`Domain`](@ref)
holds a dynamical core (`BoussinesqSimulation` or `AnelasticSimulation`),
its time scheme, its drivers and its concretely typed tuple of components;
a [`Model`](@ref) holds the domains. `step!(model, dt)` runs the fixed
stage pipeline ([`PIPELINE`](@ref)): the stages before the core, the core
step with the components of stages 6–8 called from inside it (every
substage of RK3), then the stages after it.

The core's own pieces are the first components: [`CentredAdvection`](@ref)
(slot `:advection`), [`SGS`](@ref) (slot `:sgs`), [`Coriolis`](@ref),
[`SurfaceThetaFlux`](@ref) and [`Sponge`](@ref).

A domain's budgets ([`with_budgets`](@ref), ONE-c #452) attribute every
change of the measured quantities to the stage entries automatically
(`Aeolus.BudgetSystem`); its output catalogue ([`domain_catalogue`](@ref))
is the core's variables and every component's [`outputs`](@ref), which
`Aeolus.Output` writes; its checks ([`domain_checks`](@ref), ONE-d #453)
are the core's (the sanity check and the budget closures) and every
component's [`checks`](@ref), which `Aeolus.Checks` evaluates.
"""
module Components

using ..Operators: Operators, momentum_advect!, add_coriolis!, add_surface_flux!,
                   AbstractSGSClosure, StaticSmagorinsky, DynamicSmagorinsky, VremanClosure,
                   SigmaModel, AMDClosure, StabilityCorrected
using ..TurbulenceStability: BrownEtAl1994, LillyCutoff, ShortTail, LongTail
using ..Simulations: Simulations, BoussinesqSimulation, step_rk3!, add_velocity_sponge!
using ..Anelastic: MassWeightedSimulation
using ..TimeIntegration: TimeIntegration, CFLConstraints, adaptive_dt
import ..TimeIntegration: step_adaptive!
using ..InputDrivers: advance_feed!
using ..Config: Config, Section, ConfigError, ConfigProblem
using ..Grids: interior
using ..Fields: interior_view
using ..Backends: plane_partials!
using ..MoistThermodynamics: DEFAULT_CONSTANTS, R_d
using ..BudgetSystem: Budgets, StageBudget, LevelMeter, transport_conserved_weight, budget_start!,
                      budget_begin_step!, budget_end_step!, budget_open!, budget_close!, budget_discard!,
                      stage_budget
using ..Output: OutputVariable, check_catalogue, cell_centres
using ..Checks: Check, ReferenceProfile, reference_check
using ..IO: nonfinite_fields
import ..IO: checkpoint_state, restore_checkpoint_state!
import ..TimeSteppers: step!

export AbstractComponent, Stage, PIPELINE, SLOTS, Domain, Model,
       CentredAdvection, SGS, Coriolis, Sponge, SurfaceThetaFlux,
       DomainBudgets, with_budgets, BUDGET_QUANTITIES, CORE_OUTPUTS, DEFAULT_OUTPUTS, domain_catalogue,
       CORE_CHECKS, domain_checks

include("components/interface.jl")
include("components/pipeline.jl")
include("components/domain.jl")
include("components/budgets.jl")
include("components/outputs.jl")
include("components/checks.jl")
include("components/advection.jl")
include("components/sgs.jl")
include("components/coriolis.jl")
include("components/surface_theta_flux.jl")
include("components/sponge.jl")

end # module
