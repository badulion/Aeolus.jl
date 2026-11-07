"""
    Aeolus.Components

The component interface and the fixed stage pipeline of the single model
(ONE-a, #450; the design is `docs/src/design/single_model.md`).

A **component** ([`AbstractComponent`](@ref)) is a reusable module
instance configured by one TOML section: a thin adapter from a physics
module (the equations live there) to the pipeline. A [`Domain`](@ref)
holds a dynamical core (`BoussinesqSimulation` or `AnelasticSimulation`),
its time scheme, its drivers and its concretely typed tuple of components;
a [`Model`](@ref) holds the domains and, for several, their nest coupling
([`nested_model`](@ref), ONE-f #455). `step!(model, dt)` runs the fixed
stage pipeline ([`PIPELINE`](@ref)): the stages before the core, the core
step with the components of stages 6–8 called from inside it (every
substage of RK3), then the stages after it.

The core's own pieces are the first components: [`CentredAdvection`](@ref)
(slot `:advection`), [`SGS`](@ref) (slot `:sgs`), [`Coriolis`](@ref),
[`SurfaceThetaFlux`](@ref) and [`Sponge`](@ref). The boundary-layer
physics of the stable-boundary-layer cases (ONE-g, #456) are the
[`LargeScale`](@ref) forcing, the Monin–Obukhov [`SurfaceLayer`](@ref)
(slot `:surface`) and the virtual [`Towers`](@ref).

A domain's budgets ([`with_budgets`](@ref), ONE-c #452) attribute every
change of the measured quantities to the stage entries automatically
(`Aeolus.BudgetSystem`); its output catalogue ([`domain_catalogue`](@ref))
is the core's variables and every component's [`outputs`](@ref), which
`Aeolus.Output` writes; its checks ([`domain_checks`](@ref), ONE-d #453)
are the core's (the sanity check and the budget closures) and every
component's [`checks`](@ref), which `Aeolus.Checks` evaluates.
"""
module Components

using ..Operators: Operators, momentum_advect!, add_coriolis!, coriolis_parameter, add_surface_flux!,
                   total_stress_profile, total_heat_flux_profile, stress_bl_height, inversion_height,
                   sgs_fields, damping_coordinates, w_variance_profile, resolved_tke_profile, smooth_step, isotropic_energy_spectrum, dissipation_rate,
                   AbstractSGSClosure, StaticSmagorinsky, DynamicSmagorinsky, VremanClosure,
                   SigmaModel, AMDClosure, StabilityCorrected
using ..TurbulenceStability: BrownEtAl1994, LillyCutoff, ShortTail, LongTail, boundary_layer_height
using ..Simulations: Simulations, BoussinesqSimulation, step_rk3!, add_velocity_sponge!
using ..PrognosticSGS: TKEClosure, initialize_tke!
using ..IBM: SolidMask, first_level_cells, wall_distance
using ..BoundaryConditions: Periodic, NoSlipWall
using ..Grids: CellCenter
using ..StretchedGrids: StretchedGrid
using ..BoundaryConditions: fill_halos!
using ..SurfaceExchange: AbstractStabilityFunctions, BusingerDyerLogLinear, BusingerDyerBeljaarsHoltslag,
                         solve_surface_layer, kinematic_heat_flux, similarity_theta, similarity_wind, obukhov_length
using ..WallModels: MOSTWallLaw, WallModel, apply_wall_stress!
using ..LargeScaleForcing: GeostrophicPressureGradient, AdvectiveTendencies, Subsidence, ProfileNudging,
                           SurfaceTemperatureSeries, apply_forcing!, forcing_mean, surface_temperature, _zc,
                           forcing_value, NudgingWeight, ReferenceDamping
using ..Operators: sponge_profile
using ..Operators: FiniteVolume                                      # the vertical metrics of ω_y (#505)
using ..InputDrivers: shift_table
using ..Transition: TowerSampler, tower_sampler, sample_towers!, reset_segment!, tower_thresholds,
                    ParentResolvedStatistics, update_inflow_statistics!, inflow_stress_parts,
                    turbulent_fraction, flux_richardson, FringeRegion, blasius_target, blasius_freestream,
                    FreeStreamDisturbance, blasius_solution, disturbance_fringe_update, set_blasius!, add_fringe!,
                    skin_friction, default_blasius, InflowTurbulence, DigitalFilterGenerator,
                    free_stream_generator, exclude_inflow_feedback!, fold_inflow_statistics!
using ..Anelastic: MassWeightedSimulation
using ..TimeIntegration: TimeIntegration, CFLConstraints, adaptive_dt
import ..TimeIntegration: step_adaptive!
using ..InputDrivers: advance_feed!, dynamic_forcing
using ..Config: Config, Section, ConfigError, ConfigProblem
using ..Grids: interior
using ..Fields: interior_view, set!
using ..Backends: plane_partials!
using ..MoistThermodynamics: DEFAULT_CONSTANTS, R_d, virtual_temperature, PhasePartition
using ..BudgetSystem: Budgets, StageBudget, LevelMeter, transport_conserved_weight, budget_start!,
                      budget_begin_step!, budget_end_step!, budget_open!, budget_close!, budget_discard!, budget_skip!,
                      stage_budget
using ..Output: OutputVariable, check_catalogue, cell_centres
using ..Checks: Check, ReferenceProfile, reference_check
using ..IO: NonFiniteField, nonfinite_fields, structural_state, restore_structural_state!, refresh_structural_state!
using ..Nesting: Nesting, NestHierarchy, add_nest!, step_nested!, domain_advance, DomainAdvance
import ..IO: checkpoint_state, restore_checkpoint_state!, component_nonfinite_fields
import ..TimeSteppers: step!

export AbstractComponent, Stage, PIPELINE, SLOTS, Domain, Model,
       CentredAdvection, SGS, Coriolis, Sponge, SurfaceThetaFlux, LargeScale, SurfaceLayer, Towers, ParentStatistics, KinematicFlow, Fringe,
       DomainBudgets, with_budgets, BUDGET_QUANTITIES, CORE_OUTPUTS, DEFAULT_OUTPUTS, domain_catalogue,
       CORE_CHECKS, domain_checks, NestLink, ModelNest, nested_model, nest_window!, NestStepFailure,
       NEST_EXCHANGE, nest_exchange, NEST_INFLOW_DEFAULTS,
       BoundaryLayerTracker, nudging_height, parcel_zi, surface_friction_velocity

include("components/interface.jl")
include("components/pipeline.jl")
include("components/domain.jl")
include("components/budgets.jl")
include("components/outputs.jl")
include("components/checks.jl")
include("components/nesting.jl")
include("components/nest_engine.jl")
include("components/advection.jl")
include("components/sgs.jl")
include("components/coriolis.jl")
include("components/surface_theta_flux.jl")
include("components/large_scale_forcing.jl")
include("components/sponge.jl")
include("components/surface_layer.jl")
include("components/towers.jl")
include("components/inflow_statistics.jl")
include("components/kinematic_flow.jl")
include("components/fringe.jl")

end # module
