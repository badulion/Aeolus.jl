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
import ..IO: checkpoint_state, restore_checkpoint_state!
import ..TimeSteppers: step!

export AbstractComponent, Stage, PIPELINE, SLOTS, Domain, Model,
       CentredAdvection, SGS, Coriolis, Sponge, SurfaceThetaFlux

include("components/interface.jl")
include("components/pipeline.jl")
include("components/domain.jl")
include("components/advection.jl")
include("components/sgs.jl")
include("components/coriolis.jl")
include("components/surface_theta_flux.jl")
include("components/sponge.jl")

end # module
