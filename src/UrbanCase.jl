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
# From Julia: `build_case(CaseConfig(load_run_config(path)))`, then
# `run_case!(case)`.
#
# Layout (src/urban/):
#
#   core/        the case machinery — config, the coupling registry,
#                geometry, the surface bus, the tracer registry, the
#                budgets, the Case and its Domain on the core's stage
#                pipeline, the output, and the site's sections of the
#                model's configuration schema.
#   couplings/   one file per physics coupling. Each defines a struct
#                `<: AbstractCoupling`, extends the registry contract, and
#                ends with `register!(...)`. Files are included in sorted
#                order; adding a coupling means adding one file.
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
using ..Aeolus.Components: Components, AbstractComponent, Stage, Domain, DomainBudgets
using ..Aeolus.Checks: Check, CheckRunner, enabled_checks, evaluate_checks, results_table, required_failures,
                       advisory_failures, ChecksFailed
using ..Aeolus.Output: Output, OutputVariable, OutputWriter, check_catalogue, cell_centres, write_record!
# the diagnostics registry (LOG-g, #338): the couplings extend these two
import ..Aeolus.RunLog: diagnostics, profile_diagnostics
using Printf
using Dates: Dates, DateTime, Millisecond
using Statistics: mean
import Random

for f in ("config", "registry", "geometry", "bus",
          "tracers", "budgets", "reference_state", "top_boundary",
          "open_boundary",
          "output", "case", "stages", "diagnostics")
    include(joinpath(@__DIR__, "urban", "core", f * ".jl"))
end

"Coupling source files, in the order they are included (and registered)."
const COUPLING_FILES = sort!(filter(endswith(".jl"), readdir(joinpath(@__DIR__, "urban", "couplings"))))
for f in COUPLING_FILES
    include(joinpath(@__DIR__, "urban", "couplings", f))
end

# The parent domain of the nest (#246): its tiled surface (#249), the
# parent case (#259)
include(joinpath(@__DIR__, "urban", "parent", "surface.jl"))
include(joinpath(@__DIR__, "urban", "parent", "config.jl"))
include(joinpath(@__DIR__, "urban", "parent", "case.jl"))
include(joinpath(@__DIR__, "urban", "core", "restart.jl"))      # checkpoint / restart of the case (#269)

# The site's sections of the model's schema and its run under `aeolus run`
# (CFG-c #358, EX-a #438, ONE-e #454): built from the registry, so it comes last
include(joinpath(@__DIR__, "urban", "core", "config_schema.jl"))
include(joinpath(@__DIR__, "urban", "core", "parent_driver.jl"))   # the parent alone (`urban_parent`)

export CaseConfig, UrbanParentDriver, run_urban!, run_results, parent_results, case_state, build_case, run_case!, step_case!, describe,
       AbstractCoupling, Cadence, REGISTRY, register!, selected_couplings,
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
