# BerlinCase — the Berlin (Alexanderplatz) flagship case as a module.
#
#   include("examples/urban/BerlinCase.jl")
#   using .BerlinCase
#   case = build_case(CaseConfig(:alexanderplatz; nsteps = 1800))
#   run_case!(case)
#
# Layout:
#
#   core/        the case machinery — config, scenarios, the coupling
#                registry, geometry, the surface bus, the tracer registry,
#                the budget ledger, the Case, the stage loop, the output.
#   couplings/   one file per physics coupling. Each defines a struct
#                `<: AbstractCoupling`, extends the registry contract, and
#                ends with `register!(...)`. Files are included in sorted
#                order; adding a coupling means adding one file.
#
# The physics itself lives in `src/` (Aeolus). A coupling only wires a
# src implementation into the 3-D run; it never carries its own copy.
# See docs/src/design/berlin_case.md.

module BerlinCase

using Aeolus
using Aeolus.Fields: set!, interior_view
using Aeolus.Radiation: apply_radiative_cooling!
using Aeolus.Operators: advect_centered!
using Aeolus.AdvectionSchemes: advect_positive!, FluxLimiterWorkspace
using Aeolus.IBM: SolidMask, mark_solid!
using Printf
using Statistics: mean

"Directory of the Berlin case (examples/urban)."
const URBAN_DIR = @__DIR__
"Input data directory (OSM files)."
const DATA_DIR = joinpath(URBAN_DIR, "data")

for f in ("config", "scenarios", "registry", "geometry", "osm_ingest", "bus",
          "tracers", "air_heat", "air_heat_ledger", "ledger", "reference_state", "top_boundary",
          "output", "case", "stages")
    include(joinpath(@__DIR__, "core", f * ".jl"))
end

"Coupling source files, in the order they are included (and registered)."
const COUPLING_FILES = sort!(filter(endswith(".jl"), readdir(joinpath(@__DIR__, "couplings"))))
for f in COUPLING_FILES
    include(joinpath(@__DIR__, "couplings", f))
end

# The parent domain of the nest (#246): its tiled surface (#249)
include(joinpath(@__DIR__, "parent", "surface.jl"))

export CaseConfig, SCENARIOS, build_case, run_case!, step_case!, describe,
       AbstractCoupling, Cadence, REGISTRY, register!, selected_couplings,
       ensure_tracer!, tracer, SurfaceBus, SurfaceParams, find_coupling,
       Ledger, ledger_report, surface_budget, water_budget, heat_budget, add_heat!,
       AirHeatMeter, air_heat, register_surface_factory!,
       SurfaceSite, class_surface, class_roughness,
       ParentLanduse, read_parent_landuse, child_footprint, parent_fractions,
       parent_morphometry, parent_forcing_level, ParentColumn, ParentSurface,
       build_parent_surface, parent_tile_forcings, step_parent_column!, parent_offline_forcing,
       CaseReference, ref_pressure, ref_density, ref_exner, surface_pressure,
       initial_θ, initial_qv, case_sounding, case_start_time, case_surface_pressure,
       initial_inversion_height, case_weekday,
       θ_top_gradient, θ_top_bc, TopHeatFlux

end # module BerlinCase
