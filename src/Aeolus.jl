module Aeolus

# Aeolus.jl — a Julia-native large-eddy simulation framework.
#
# Design intent: the spatial discretization is behind an `AbstractMesh`
# interface. Phase 0/1 ships a finite-volume backend on a staggered
# Cartesian grid (RegularGrid). A future spectral-element backend can be
# added without changing user code — it plugs in via multiple dispatch on
# the mesh type inside the Operators layer.

include("Grids.jl")
include("Fields.jl")
include("BoundaryConditions.jl")
include("Operators.jl")
include("TimeSteppers.jl")
include("PressureSolvers.jl")
include("Simulations.jl")
include("IO.jl")
include("Microphysics.jl")
include("IBM.jl")
include("Parallel.jl")
include("Nesting.jl")
include("Backends.jl")
include("SpectralElement.jl")
include("Radiation.jl")
include("Chemistry.jl")
include("Particles.jl")
include("ML.jl")
include("DataAssimilation.jl")

# --- Era II: numerical foundations for fidelity (Phases 21-31) -----------
include("StretchedGrids.jl")    # Phase 24
include("CutCell.jl")           # Phases 21-22
include("AdvectionSchemes.jl")  # Phases 26, 28
include("TimeIntegration.jl")   # Phases 23, 25
include("Anelastic.jl")         # Phase 27
include("Multigrid.jl")         # Phase 29
include("AdaptiveMesh.jl")      # Phases 30-31

# --- Era III: high-fidelity radiation (Phases 32-38) ---------------------
include("RadiationCore.jl")     # shared spectral / optics / flux contract
include("GasOptics.jl")         # Phase 32
include("SurfaceOptics.jl")     # Phase 36
include("AerosolOptics.jl")     # Phase 37
include("TwoStream.jl")         # Phase 33
include("MonteCarloRT.jl")      # Phase 34
include("Radiosity.jl")         # Phase 35
include("SolarGeometry.jl")     # Phase 38

using .Grids
using .Fields
using .BoundaryConditions
using .Operators
using .TimeSteppers
using .PressureSolvers
using .Simulations
using .IO
using .Microphysics
using .IBM
using .Parallel
using .Nesting
using .Backends
using .SpectralElement
using .Radiation
using .Chemistry
using .Particles
using .ML
using .DataAssimilation

using .StretchedGrids
using .CutCell
using .AdvectionSchemes
using .TimeIntegration
using .Anelastic
using .Multigrid
using .AdaptiveMesh

using .RadiationCore
using .GasOptics
using .SurfaceOptics
using .AerosolOptics
using .TwoStream
using .MonteCarloRT
using .Radiosity
using .SolarGeometry

# Re-exports for the user-facing API.
export AbstractMesh, RegularGrid
export Field, CellCenter, FaceX, FaceY, FaceZ
export Periodic, NoFlux, ValueBC, FluxBC, FieldBoundaryConditions, fill_halos!
export ∂x, ∂y, ∂z, divergence, laplacian, advect_centered!, advect_upwind!,
       interpolate_to_face!, interpolate_to_center!, momentum_advect!,
       add_buoyancy!, add_coriolis!, compute_smagorinsky_νt!,
       add_eddy_viscosity_u!, add_eddy_viscosity_v!, add_eddy_viscosity_w!,
       add_eddy_diffusion_scalar!,
       add_surface_flux!, add_sponge!,
       advect_weno5!
export RK3, step!
export AbstractPressureSolver, FFTPoissonSolver, CosineFFTPoissonSolver,
       CGPoissonSolver, solve!
export ImpermeableWall
export BoussinesqSimulation, run!, step_rk3!, kinetic_energy, max_divergence,
       max_νt, mean_θ_profile,
       horizontal_mean, horizontal_variance, resolved_flux_wθ
export write_snapshot, read_snapshot, checkpoint!, restart!, load_config
export apply_kessler_step!
export SolidMask, mark_solid!, apply_solid_mask!, add_ibm_penalty!,
       implicit_ibm_correction!
export AbstractPartition, SerialPartition, PencilPartition,
       nranks, is_serial, exchange_halos_distributed!
export NestSpec, apply_one_way_forcing!, apply_two_way_feedback!
export AbstractBackend, CPUBackend, GPUBackend, allocate, to_host, backend_name
export SpectralElementMesh, gll_nodes, gll_weights
export apply_radiative_cooling!, SlabLand, step_land!, surface_temperature
export apply_linear_chemistry!
export ParticleSet, add_particles!, advance_particles!, positions
export MLSGS, apply_mlsgs!
export Observation, assimilate_nudging!

# =========================================================================
# Era II — numerical foundations for fidelity (Phases 21-31)
# =========================================================================

# Phases 21-22 — cut-cell IBM + ghost-cell wall reconstruction
export CutCellGeometry, cut_cell_geometry, cut_cell_geometry_from_heightmap,
       solid_mask, fluid_volume, wetted_area, cut_cell_count,
       cutcell_divergence!, cutcell_gradient_x!, cutcell_gradient_y!,
       cutcell_gradient_z!, cutcell_laplacian!,
       cutcell_cell_inner, cutcell_face_inner,
       apply_cutcell_velocity_bc!,
       GhostCellStencil, build_ghost_stencils,
       reconstruct_wall_velocity!, reconstruct_wall_scalar!,
       wall_shear_stress

# Phase 24 — stretched & terrain-following vertical coordinates
export StretchedGrid, TerrainFollowingGrid, terrain_following_grid,
       grid_stretching_ratio,
       zfaces, zcenters, cell_thickness, center_spacing,
       metric_jacobian, terrain_height, physical_heights,
       set_physical!, hydrostatic_balance_error

# Phases 23, 25 — adaptive time stepping + IMEX integration
export CFLConstraints, cfl_diagnostics, adaptive_dt,
       SubsteppingSchedule, substeps_for,
       StateSnapshot, state_snapshot, save_state!, restore_state!,
       step_adaptive!,
       DiffusionWorkspace, solve_implicit_diffusion!,
       IMEXScheme, ARK2, Rosenbrock2,
       IMEXWorkspace, imex_workspace, step_imex!, imex_stability_limit

# Phases 26, 28 — WENO-Z / TENO advection + conservative operators
export weno_z5_left, weno_z5_right, weno_z7_left, weno_z7_right,
       teno5_left, teno5_right, teno8_left, teno8_right,
       weno_js5_left, weno_js5_right,
       advect_weno_z!, advect_teno!, momentum_advect_weno_z!,
       apply_positivity_limiter!, advect_positive!, FluxLimiterWorkspace,
       AdvectionScheme, advection_scheme, required_halo,
       momentum_advect_skew!, advect_variance_conserving!,
       kinetic_energy_budget, scalar_variance_budget, enstrophy

# Phase 27 — anelastic & pseudo-incompressible dynamics
export ReferenceState,
       isothermal_reference, isentropic_reference,
       constant_stability_reference, reference_from_profile,
       hydrostatic_residual, max_hydrostatic_residual,
       reference_scale_height,
       DensityWeightedPoissonSolver, apply_weighted_laplacian!,
       solver_iterations, solver_residual,
       MassWeighting, AnelasticWeighting, PseudoIncompressibleWeighting,
       MassWeightedSimulation, AnelasticSimulation,
       PseudoIncompressibleSimulation,
       step_anelastic!, step_pseudo_incompressible!,
       anelastic_divergence!, pseudo_incompressible_divergence!,
       mass_flux_divergence!, weighted_divergence!,
       max_anelastic_divergence, max_mass_flux_divergence,
       max_pseudo_incompressible_divergence, constraint_residual,
       add_anelastic_buoyancy!,
       weighted_momentum_advect!, weighted_scalar_advect!,
       set_reference_state!, reference_state,
       θ_variance, max_w, bubble_top_height, unweighted_kinetic_energy,
       system_comparison_diagnostics

# Phase 29 — geometric multigrid pressure solver
export MultigridPoissonSolver, MGPreconditionedCG,
       mg_vcycle!, mg_solve!, mg_residual_norm,
       mg_level_count, mg_iteration_history

# Phases 30-31 — production nesting + block-structured AMR
export NestedGrid, nested_grid, refinement_ratio, nest_offset,
       nest_covered_range, nest_child_mesh, nest_parent_mesh,
       NestBC, nest_bcs,
       interpolate_to_nest_boundary!, interpolate_parent_to_child!,
       restrict_to_parent!,
       NestFluxMismatch, compute_nest_flux_mismatch, apply_flux_correction!,
       mismatch_l1, mismatch_area,
       NestSponge, apply_nest_sponge!,
       NestInterpState, nest_interp_state, interpolate_parent_state!,
       step_nested!, nest_conservation_error, composite_integral,
       RefinementBlock, AMRHierarchy, amr_hierarchy,
       refinement_criterion_gradient, refinement_criterion_vorticity,
       refinement_criterion_richardson, combine_criteria,
       cluster_flagged_cells, box_efficiency,
       regrid!, check_proper_nesting, step_amr!,
       amr_total_mass, amr_error, total_cells, level_cells, block_bounds

# =========================================================================
# Era III — high-fidelity radiation (Phases 32-38)
# =========================================================================

# Shared radiation contract (spectral grids, optics, fluxes, interfaces)
export SpectralGrid, shortwave_bands, longwave_bands,
       nbands, ngpoints, band_of_gpoint, band_limits, gpoint_range,
       OpticalProperties, allocate_optics, zero_optics!,
       combine_optics!, scale_optics!, total_optical_depth,
       RadiativeFluxes, allocate_fluxes, zero_fluxes!,
       net_flux!, flux_divergence!, heating_rate!,
       AbstractGasOptics, AbstractCloudOptics, AbstractAerosolOptics,
       AbstractRTSolver, AbstractSurfaceOptics,
       compute_optics!, solve_rt!, surface_albedo, surface_emissivity,
       AtmosphericColumn, column_from_simulation,
       STEFAN_BOLTZMANN, SOLAR_CONSTANT, planck_band, planck_integrated

# Phase 38 — solar position, refraction, irradiance partitioning, shadows
export AbstractSolarPosition, MeeusPosition,
       SolarTime, solar_time, julian_day, julian_century,
       SolarPosition, solar_position,
       equation_of_time, solar_declination_accurate,
       sun_direction_enu, air_mass,
       refraction_bennett, apply_refraction,
       earth_sun_distance_factor, extraterrestrial_irradiance,
       ClearSkyModel, BirdClearSky, clear_sky_irradiance,
       clearness_index, diffuse_fraction_erbs, diffuse_fraction_reindl,
       partition_global_irradiance,
       cloud_optical_depth_column, cloud_shadow_transmittance!,
       cloud_shadow_map, combined_shadow_transmittance!

end # module
