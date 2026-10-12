# UrbanCase — the urban site of the model (ONE-e, #454; formerly the urban
# LES run type, EX-a #438).
#
# A 3-D LES of a city window: buildings and land use from a static input
# driver, the initial state and the boundary forcing from a sounding, a
# dynamic input driver or an online Aeolus nest parent, and the urban
# physics as components of the domain's stage pipeline. A model config is
# an urban site with
#
#   [urban]
#   enabled = true
#
# and one section per urban component it enables (`aeolus options` lists
# them). The Berlin (Alexanderplatz) case is one configuration of it:
#
#   julia --project -m Aeolus run examples/urban/berlin_summer.toml
#
# `aeolus run` builds it as a `Model` of one domain ([`build_model`]) and
# runs it with the model's run loop; the site is the domain's first
# component, and its `:timestepper` component takes the core step
# (`step!(model)`, ONE-k #474).
#
# Layout (src/urban/):
#
#   core/        the case machinery — config, the component interface,
#                geometry, the surface bus, the tracer registry, the
#                budgets, the UrbanSite (the site's component) and its Domain on the core's stage
#                pipeline, the output, and the site's sections of the
#                model's configuration schema.
#   couplings/   one file per urban component. Each defines a struct
#                `<: UrbanComponent` (an `Aeolus.Components.AbstractComponent`)
#                and its methods (core/components.jl); `URBAN_COMPONENTS`
#                lists them.
#   parent/      the nest parent: its tiled land-use surface, its config,
#                the parent case.
#
# The physics itself lives in the other modules of `src/`. A coupling only
# wires an implementation into the 3-D run; it never carries its own copy.
# See docs/src/design/berlin_case.md. Moved from examples/urban/ (EX-a,
# #438).

module UrbanCase

using ..Aeolus
using ..Aeolus.Fields: set!, interior_view
using ..Aeolus.Radiation: apply_radiative_cooling!
using ..Aeolus.Operators: advect_centered!
using ..Aeolus.AdvectionSchemes: advect_positive!, FluxLimiterWorkspace
using ..Aeolus.IBM: SolidMask, mark_solid!
# the ground: surface classes, footprints and the static driver (CFG-f, #361)
using ..Aeolus.InputDrivers: SURFACE_STREET, SURFACE_ROOF, SURFACE_PARK, SURFACE_WATER, SURFACE_WALL,
                           Building, UrbanCity, polygon_bbox, point_in_polygon, rasterize_city,
                           solid_and_classes, tag_building_adjacent!,
                           StaticInputs, StaticTiles, read_static_driver, read_static_tiles,
                           DynamicInputs, read_dynamic_driver, sounding_dynamic_inputs,
                           init_time_string, init_day_seconds, surface_pressure_at, shift_table, file_fnv1a64,
                           FileBoundaryFeed, feed_window!, feed_steady!, check_planes, check_coverage,
                           NestBoundaryRecorder, record_boundary!, plane_snapshot, write_dynamic_driver
using ..Aeolus.StretchedGrids: surface_refined_faces
# the budgets (ONE-c, #452): the case books into the core's Budgets
using ..Aeolus.BudgetSystem
# the one output writer (ONE-c, #452): the case and its couplings publish catalogues
# the core's stage pipeline (ONE-a #450, ONE-e #454): the case is a Domain of it
using ..Aeolus.Components: Components, AbstractComponent, Stage, Domain, Model, DomainBudgets
# the urban components' names, slots and incompatibilities are the component interface's
import ..Aeolus.Components: component_name, slot, incompatible
using ..Aeolus.Checks: Check, CheckRunner, enabled_checks, evaluate_checks, results_table, required_failures,
                       advisory_failures, ChecksFailed
using ..Aeolus.Output: Output, OutputVariable, check_catalogue
# the diagnostics registry (LOG-g, #338): the couplings extend these two
import ..Aeolus.RunLog: diagnostics, profile_diagnostics
using Printf
using Dates: Dates, DateTime, Millisecond
using Statistics: mean
import Random

for f in ("config", "components", "geometry", "bus",
          "tracers", "budgets", "reference_state", "top_boundary",
          "open_boundary",
          "case", "stages", "output", "diagnostics")
    include(joinpath(@__DIR__, "urban", "core", f * ".jl"))
end

"Urban component source files, in the order they are included."
const COUPLING_FILES = sort!(filter(endswith(".jl"), readdir(joinpath(@__DIR__, "urban", "couplings"))))
for f in COUPLING_FILES
    include(joinpath(@__DIR__, "urban", "couplings", f))
end

"""
    URBAN_COMPONENTS

The urban components of the site (ONE-e, #454; formerly the coupling
registry), in the order `build_case` builds them (and of the `init!`
calls of equal `init_priority`); within a stage the case's domain orders
them by their declared stage order, then by name. Per slot the
alternatives, in the order the smoke matrix tries them when it moves a
slot to its first compatible component. Each has a section of the model's schema; a config enables it
there.
"""
const URBAN_COMPONENTS = Type[
    AQChem, NoChem, CityMeanBuildings, CloudShadow, ColumnRadiation,
    BoussinesqCore, AnelasticCore, PseudoIncompressibleCore, FogDew, GeostrophicCoupling,
    GlassClass, GroundSoil, IBMCoupling, InflowTurbulenceCoupling, MicroKessler, MoistBuoyancy,
    MomentumCentered, MomentumWENOZ, NestingCoupling, PollutantCoupling,
    PressureCosineFFT, PressureMultigrid, PressureMGCG, PressureDensityDirect, PressureDensityWeighted,
    RadiositySun, ScalarAdvection,
    SGSSmagorinsky, SGSDynamic, SGSMLSGS, SGSAMD, SGSVreman, SGSReynoldsStress, SGSSigma, SGSTKE,
    SurfaceExchangeMOST, SWModifiers, FixedStep, AdaptiveStep, IMEXStep,
    TrafficHeat, TurbulenceDiag, VegDrag, WallModelFacades]
foreach(check_urban_component, URBAN_COMPONENTS)
allunique(map(component_name, URBAN_COMPONENTS)) || error("two urban components share a name")

"""
    URBAN_DEFAULTS

The reference selection of the urban site: one component per urban slot
and every additive one, what the Berlin configurations enable (each
`examples/urban/berlin_*.toml` states its selection in full) and what
`CaseConfig(; kw...)` starts from. The smoke matrix varies one slot of it
at a time.
"""
const URBAN_DEFAULTS = Type[
    AQChem, CityMeanBuildings, CloudShadow, ColumnRadiation, BoussinesqCore, FogDew, GeostrophicCoupling,
    GlassClass, GroundSoil, IBMCoupling, InflowTurbulenceCoupling, MicroKessler, MoistBuoyancy,
    MomentumCentered, NestingCoupling, PollutantCoupling, PressureCosineFFT, RadiositySun, ScalarAdvection,
    SGSTKE, SurfaceExchangeMOST, SWModifiers, AdaptiveStep, TrafficHeat, TurbulenceDiag, VegDrag,
    WallModelFacades]
all(s -> count(C -> slot(C) === s, URBAN_DEFAULTS) == 1, URBAN_SLOTS) ||
    error("URBAN_DEFAULTS: one component per urban slot")

# The parent domain of the nest (#246): its tiled surface (#249), the
# parent case (#259)
include(joinpath(@__DIR__, "urban", "parent", "surface.jl"))
include(joinpath(@__DIR__, "urban", "parent", "config.jl"))
include(joinpath(@__DIR__, "urban", "parent", "case.jl"))
include(joinpath(@__DIR__, "urban", "core", "restart.jl"))      # checkpoint / restart of the case (#269)

# The site's sections of the model's schema and its run under `aeolus run`
# (CFG-c #358, EX-a #438, ONE-e #454): built from URBAN_COMPONENTS, so it comes last
include(joinpath(@__DIR__, "urban", "core", "config_schema.jl"))
include(joinpath(@__DIR__, "urban", "core", "parent_driver.jl"))   # the parent alone (`urban_parent`)

export CaseConfig, UrbanParentDriver, build_model, run_results, parent_results, case_state, build_case, describe,
       UrbanComponent, Cadence, URBAN_COMPONENTS, URBAN_SLOTS, URBAN_DEFAULTS, selected_components, component_options,
       ensure_tracer!, tracer, SurfaceBus, SurfaceParams, find_coupling,
       case_diagnostics, case_profiles,
       register_surface_factory!,
       SurfaceSite, class_surface, class_roughness,
       ParentLanduse, read_parent_landuse, child_footprint, parent_fractions,
       parent_morphometry, parent_forcing_level, ParentColumn, ParentSurface,
       build_parent_surface, parent_tile_forcings, step_parent_column!, parent_offline_forcing,
       CaseReference, ref_pressure, ref_pressure_storage, ref_density, ref_exner, surface_pressure,
       initial_θ, initial_qv, case_sounding, case_start_time, case_surface_pressure,
       initial_inversion_height, case_weekday,
       θ_top_gradient, θ_top_bc, TopHeatFlux,
       case_boundary_conditions, case_boundary_data, case_scalar_bcs, OpenBoundaryHeat,
       ParentConfig, ParentCase, build_parent, step_parent!, run_parent!, parent_record!,
       parent_zi, parent_heat_flux, mean_heat_flux, parent_mesh, parent_faces, parent_size, coarsen_landuse

end # module UrbanCase
