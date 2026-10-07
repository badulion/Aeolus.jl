module Aeolus

# Aeolus.jl — a Julia-native large-eddy simulation framework.
#
# Design intent: the spatial discretization is behind an `AbstractMesh`
# interface. Phase 0/1 ships a finite-volume backend on a staggered
# Cartesian grid (RegularGrid). A future spectral-element backend can be
# added without changing user code — it plugs in via multiple dispatch on
# the mesh type inside the Operators layer.

# Structured logging (LOG-a, #333): first, so every module can tag its events.
include("logging/RunLog.jl")
include("Grids.jl")
include("Backends.jl")     # KA kernels + launch/reduction primitives (KA-a, #294); before Fields
include("Fields.jl")
include("BoundaryConditions.jl")
# Era VI stability contract (Phase 58): pure, and needed by Operators.
include("TurbulenceStability.jl")
# MoistThermodynamics (Era IV, Phase 39) is dependency-free. It is included
# ahead of Operators, whose N² uses its saturated buoyancy frequency (#239),
# and so also ahead of AerosolOptics (Kelvin coefficient, Köhler theory).
include("MoistThermodynamics.jl")  # Phase 39
include("Operators.jl")
include("TimeSteppers.jl")
include("PressureSolvers.jl")
include("Simulations.jl")
include("NCIO.jl")            # NetCDF-4 output policy over NCDatasets (#310); before IO (diagnostics.nc, #337)
include("IO.jl")
include("Config.jl")         # run configuration schema, loader, canonical run.toml (CFG-a, #356)
include("IBM.jl")
include("InputDrivers.jl")   # static driver, OSM/GeoJSON preprocessors (CFG-f, #361); needs IBM, NCIO, IO
include("Parallel.jl")
include("SpectralElement.jl")
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
include("Nesting.jl")          # Phase 30 parent-child operators (N-nest-a, #250)
include("AdaptiveMesh.jl")      # Phase 31 AMR (delegates to Nesting)

# --- Era III: high-fidelity radiation (Phases 32-38) ---------------------
include("RadiationCore.jl")     # shared spectral / optics / flux contract
# Era V contract (surface exchange) needs MoistThermodynamics and the
# RadiationCore albedo/emissivity generics, and SlabLand (Radiation.jl)
# implements its surface interface, so it sits here.
include("SurfaceExchange.jl")
include("Radiation.jl")         # SlabLand: first AbstractSurface implementer
include("GasOptics.jl")         # Phase 32
include("SurfaceOptics.jl")     # Phase 36
include("AerosolOptics.jl")     # Phase 37
include("TwoStream.jl")         # Phase 33
include("MonteCarloRT.jl")      # Phase 34
include("Radiosity.jl")         # Phase 35
include("SolarGeometry.jl")     # Phase 38
# Era V, Phase 52 — lake / coastal-sea water column (needs SurfaceOptics).
include("OceanMixedLayer.jl")

include("AerosolCloud.jl")         # Phase 42
include("Superdroplet.jl")         # Phase 41
include("P3Microphysics.jl")       # Phase 40
include("Microphysics.jl")         # Phase 12 Kessler; delegates rain fall speeds to P3 (Phase 44)
include("IceNucleation.jl")        # Phase 43
include("TurbulentMicrophysics.jl") # Phase 45
include("Visibility.jl")           # Phase 44 (visibility diagnostic)
include("RadiationFog.jl")         # Phase 44 (fog/dew coupling)

# --- Era V: high-fidelity land, urban and surface exchange (Phases 46-53)
include("SoilColumn.jl")           # Phase 46 (#15): soil column, Richards equation
include("Photosynthesis.jl")       # Phase 47 (Farquhar C3 / Collatz C4)
include("LandSurface.jl")          # Phase 47 (#21): vegetated land column
include("PlantCanopy.jl")          # Phase 48 (resolved plant canopy)
include("UrbanSurface.jl")         # Phase 49 (urban facets; needs SurfaceExchange + SurfaceOptics)
include("BuildingEnergy.jl")       # Phase 50 (indoor air + thermal mass; reuses UrbanSurface columns)
include("Snowpack.jl")             # Phase 51 (#34): layered snowpack, Era V surface
include("IceThermodynamics.jl")    # Era V, Phase 51: lake and sea ice (#37)
include("AirSeaFlux.jl")           # Phase 52 wave-dependent roughness (#39)
include("SurfaceTiles.jl")         # Phase 53 sub-grid tiles (#41)

# Era VI — high-fidelity turbulence closure (Phases 54-60)
# (TurbulenceStability is included before Operators, above.)
include("WallModels.jl")           # Phase 57 (#169): equilibrium wall laws, wall-stress operator
include("LargeScaleForcing.jl")    # Phase 58 (#171): geostrophic, nudging, advective, T_s(t)
include("Transition.jl")           # Phase 59 (#180): laminar/turbulent + intermittency diagnostics
include("PrognosticSGS.jl")        # Phase 56 Deardorff prognostic SGS TKE (#182)

# CFG-b (#357): the bare-simulation run driver (needs the dynamical cores,
# so after them; the driver contract and the CLI are in Config).
include("config/simulation_driver.jl")

using .RunLog
using .Grids
using .Fields
using .BoundaryConditions
using .Operators
using .TimeSteppers
using .PressureSolvers
using .Simulations
using .NCIO
using .IO
using .Config
using .Microphysics
using .IBM
using .InputDrivers
using .Parallel
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
using .Nesting
using .AdaptiveMesh

using .RadiationCore
using .GasOptics
using .SurfaceOptics
using .AerosolOptics
using .TwoStream
using .MonteCarloRT
using .Radiosity
using .SolarGeometry

using .MoistThermodynamics
using .AerosolCloud
using .Superdroplet
using .P3Microphysics
using .IceNucleation
using .TurbulentMicrophysics
using .Visibility                  # Phase 44
using .RadiationFog                # Phase 44

using .SurfaceExchange             # Era V contract
using .SoilColumn                  # Phase 46
using .Photosynthesis              # Phase 47
using .LandSurface                 # Phase 47 (#21)
using .PlantCanopy                 # Phase 48
using .UrbanSurface                # Phase 49
using .BuildingEnergy              # Phase 50
using .Snowpack                    # Phase 51 (#34)
using .IceThermodynamics           # Era V, Phase 51 (#37)
using .OceanMixedLayer             # Phase 52 (#38)
using .AirSeaFlux                  # Phase 52 (#39)
using .SurfaceTiles                # Phase 53 sub-grid tiles (#41)

# --- Era VI: high-fidelity turbulence closure (Phases 54-60) -------------
using .TurbulenceStability         # Phase 58 stability contract (#170)
using .Transition                  # Phase 59 transition diagnostics (#180)
using .WallModels                  # Phase 57 (#169)
using .LargeScaleForcing           # Era VI, Phase 58 (#171)
using .PrognosticSGS               # Phase 56 prognostic SGS TKE (#182)

# LOG-c (#335): the diagnostics registry's methods for the library's
# components (simulations, closures, wall models, nesting); needs them all.
include("logging/component_diagnostics.jl")

# Re-exports for the user-facing API.
# Structured logging (LOG-a, #333)
export AeolusLogger, TeeLogger, open_run_log, with_run_log, test_logger,
       current_domain, with_domain, parse_log_level
# Diagnostics registry, CSV writer, progress line (LOG-c, #335)
export Diagnostic, diagnostics, collect_diagnostics, prefixed, reduce_diagnostic, evaluate_diagnostic,
       ProfileDiagnostic, profile_diagnostics, collect_profiles, evaluate_profile,
       DiagnosticsWriter, record_diagnostics!, last_diagnostics, costly_due,
       ProgressLine, progress!, WarningTap, with_progress, note_warning!
export AbstractMesh, RegularGrid
export Field, CellCenter, FaceX, FaceY, FaceZ
export Periodic, NoFlux, ValueBC, FluxBC, FieldBoundaryConditions, fill_halos!
export axis_poisson_bc                # N-open-a (#251)
export ∂x, ∂y, ∂z, divergence, laplacian, advect_centered!, advect_upwind!,
       interpolate_to_face!, interpolate_to_center!, momentum_advect!,
       add_buoyancy!, add_coriolis!, compute_smagorinsky_νt!,
       compute_dynamic_smagorinsky_νt!, DynamicSmagorinskyWorkspace,
       add_eddy_viscosity_u!, add_eddy_viscosity_v!, add_eddy_viscosity_w!,
       add_eddy_diffusion_scalar!,
       add_surface_flux!, add_sponge!,
       advect_weno5!
# Era VI (P54-a, #167): the SGS closure contract
export AbstractSGSClosure, SGSFields, sgs_fields, sgs_coefficients!, advance_sgs!,
       sgs_state, restore_sgs_state!, add_sgs_momentum!, add_sgs_scalar!,
       sgs_tracer_diffusivity!,
       compute_sgs!, sgs_closure,
       AbstractEddyViscosityClosure, sgs_viscosity!, turbulent_prandtl,
       StaticSmagorinsky, DynamicSmagorinsky, FunctionSGS,
       sgs_tke!                 # N-inflow-b (#258)
# Era VI (P54-b, #173): Lagrangian-averaged dynamic Smagorinsky (operators/lagrangian_sgs.jl)
export LagrangianDynamicSmagorinsky
# Era VI (P55-b, #175): the stress-form SGS momentum divergence
export StressTensor, add_stress_divergence!, sgs_stress_tensor!,
       add_sgs_stress_divergence!, sgs_stress_dissipation
# Era VI (P55-c, #181): the tensorial scalar eddy diffusivity
export DiffusivityTensor, diagonal_tensor!, bounded_directions,
       add_tensor_diffusion_scalar!, isentropic_tensor!,
       sgs_scalar_tensor, IsentropicDiffusion
# Era VI, Phase 54 (P54-c): LES analysis toolbox
export box_filter!, gaussian_filter!, spectral_cutoff_filter!, staggered_solid_mask,
       isotropic_energy_spectrum, isotropic_random_field!, rescale_to_spectrum!
# Era VI (P54-d, #174): wall-bounded flow diagnostics
export phi_m_profile, total_stress_profile, streamwise_spectrum
# Era VI (P55-d, #176): AMD and Vreman closures
export AMDClosure, VremanClosure, compute_amd_coefficients!, compute_amd_scalar_diffusivity!,
       compute_vreman_νt!,
       sgs_buoyancy_parameter
# Era VI, Phase 56 (P56-c): a-priori SGS-stress analysis
export SymmetricTensorField, subfilter_stress!, bardina_stress!, filtered_strain, filtered_strain!,
       eddy_viscosity_stress!, smagorinsky_stress!, sgs_dissipation, stress_eigvals,
       lumley_invariants, apriori_metrics, dissipation_matched_cs
# Era VI (P59-b, #184): the σ-model closure
export SigmaModel, compute_sigma_νt!
# Era VI (P55-e, #187): the stability-corrected anisotropic closure
export StabilityCorrected, dynamic_sgs_coefficient, anisotropic_length_factors,
       sgs_stratification!
# P58-e (#193): resolved TKE profile (turbulence-collapse runs)
export resolved_tke_profile
# #245: the FFTW planning policy
export set_fftw_policy!, fftw_policy, fftw_planner_flags, save_fftw_wisdom, load_fftw_wisdom!
export RK3, step!
export AbstractPressureSolver, FFTPoissonSolver, CosineFFTPoissonSolver,
       CGPoissonSolver, solve!, poisson_axis_bcs
export ImpermeableWall
export BoussinesqSimulation, run!, step_rk3!, kinetic_energy, max_divergence,
       max_νt, mean_θ_profile,
       horizontal_mean, horizontal_variance, resolved_flux_wθ
export write_snapshot, read_snapshot, checkpoint!, restart!, load_config
# #245: full-state checkpoints with bitwise restart, periodic checkpoints
export CHECKPOINT_VERSION, read_checkpoint, checkpoint_state, restore_checkpoint_state!,
       fieldwise_checkpoint_state, restore_fieldwise!, checkpoint_fields,
       StepperState, record_step!, previous_dt,
       CheckpointSchedule, checkpoint_schedule, resume!, maybe_checkpoint!, checkpoint_due,
       latest_checkpoint, checkpoint_path, wisdom_path, apply_fftw_config!,
       driver_checkpoints, finish_checkpoints!
# CFG-a (#356): run configuration (schema building: Aeolus.Config)
export RunConfig, ConfigError, load_run_config, with, to_toml, write_config, config_text,
       config_hash, register_driver!
# CFG-b (#357): the driver contract, run_config, the bare-simulation driver.
# `Aeolus.main` (the `julia -m Aeolus` entry point) is not exported.
export AbstractRunDriver, run_config, state_fields
# CFG-e (#360): sweeps; the example scripts run their configs with run_configs
export run_configs, load_sweep
# #334: run folder and provenance
export RunFolder, start_run, finish_run!, with_run, run_checkpoints, UncommittedCodeError,
       provenance, git_state, read_run_toml, latest_run, run_log_path, run_toml_path,
       diagnostics_path, output_dir, checkpoint_dir, refresh_fftw_provenance!
# #335: blowup guard (diagnose, dump, stop) and the run monitor
export checkpoint_payload, write_checkpoint,
       BlowupGuard, BlowupError, NonFiniteField, nonfinite_fields, blowup_fields,
       check_blowup!, guarded_step!, blowup!, snapshot_good!, crash_checkpoint_path,
       RunMonitor, monitor!, monitored_step!, with_monitor, crash_dir
# #337: diagnostics.nc
export DiagnosticsNCWriter, diagnostics_nc_path, nc_provenance_attributes
export NCDataset, nc_create, nc_defvar!, nc_append_record!, nc_chunksizes,
       cf_time_units, NC_DEFLATE_LEVEL                              # #310
export apply_kessler_step!, KESSLER_N0, kessler_rain_number, kessler_fall_speed
export SolidMask, mark_solid!, apply_solid_mask!, add_ibm_penalty!,
       implicit_ibm_correction!, wall_distance, first_level_cells   # + P56-b (#182)
export AbstractPartition, SerialPartition, PencilPartition,
       nranks, is_serial, exchange_halos_distributed!
export AbstractBackend, CPUBackend, GPUBackend, allocate, to_host, backend_name,
       backend, default_backend, set_default_backend!, with_backend, init_threads!,
       runs_serially, launch!, launch_weighted!, range_map!,
       plane_partials!, ordered_sum, interior_sum, interior_dot, plane_mean!,
       column_partials, interior_max, interior_all_finite, interior_map!,
       plane_reduce!, range_max, plane_map!, apply_transform!,
       pcg!, PCGWorkspace, PointOperator, PointPreconditioner, IdentityPreconditioner,
       CounterRNG, counter_rng, counter_xoshiro, set_stream!
export SpectralElementMesh, gll_nodes, gll_weights
export apply_radiative_cooling!, SlabLand, step_land!, surface_temperature
export apply_linear_chemistry!,
       air_number_density, ppb_rate_constant, emission_ppb_flux,
       k_NO_O3, k_NO2_OH, k_VOC_O3, k_VOC_OH, j_NO2,
       nox_triad, photostationary_NO2, leighton_ratio,
       photochemistry_cell, apply_photochemistry!,
       schmidt_number, quasi_laminar_resistance, surface_resistance,
       dry_deposition_velocity, apply_dry_deposition!,
       rain_scavenging_coefficient, apply_wet_deposition!
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
export CFLConstraints, cfl_diagnostics, adaptive_dt, stress_form_cfl_viscosity,
       SubsteppingSchedule, substeps_for,
       StateSnapshot, state_snapshot, save_state!, restore_state!,
       step_adaptive!, NumericalFailure, is_numerical_failure,
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
export boundary_scalar_flux           # N-open-c (#260): the one boundary-flux function
export scalar_face_flux, face_diffusivity, face_inv_spacing   # #327 M2: the one per-face scalar flux kernel

# Phase 27 — anelastic & pseudo-incompressible dynamics
export ReferenceState,
       isothermal_reference, isentropic_reference,
       constant_stability_reference, reference_from_profile,
       hydrostatic_residual, max_hydrostatic_residual,
       reference_scale_height,
       DensityWeightedPoissonSolver, DirectWeightedPoissonSolver, apply_weighted_laplacian!,
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

# Phase 30 — parent-child nesting (Aeolus.Nesting); Phase 31 — block-structured AMR
export NestedGrid, nested_grid, refinement_ratio, nest_offset,
       nest_covered_range, nest_child_mesh, nest_parent_mesh,
       nest_bcs, NestBoundaryData, nest_boundary_data, nest_open_sides,
       interpolate_to_nest_boundary!, interpolate_parent_to_child!,
       restrict_to_parent!,
       NestFluxMismatch, compute_nest_flux_mismatch, apply_flux_correction!, nest_periodic_axes,
       mismatch_l1, mismatch_area,
       NestSponge, apply_nest_sponge!,
       NestInterpState, nest_interp_state, interpolate_parent_state!,
       nest_scalar_names, apply_nest_boundary!, nest_storage, fill_nest_halos!,
       restrict_state_to_parent!, reflux!, reflux_scalars!,
       step_nested!, nest_conservation_error, composite_integral,
       NestMeasure, nest_measure, nest_reference_mismatch, nest_fluid_mask,
       DomainAdvance, domain_advance, stage_times,
       NestedRun, NestRunStats, boundary_time, begin_parent_step!, end_parent_step!, subcycle_dt,
       begin_child_step!, end_child_step!,
       TwoWayFeedback, NestScalarFlux, NestFluxRegisters, two_way_feedback!,
       anterpolate!, feedback_mask, book_nest_flux!, nest_stage_weights,
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
       allocate_column_optics,
       RadiativeFluxes, allocate_fluxes, zero_fluxes!,
       net_flux!, flux_divergence!, heating_rate!,
       AbstractGasOptics, AbstractCloudOptics, AbstractAerosolOptics,
       AbstractRTSolver, AbstractSurfaceOptics,
       compute_optics!, solve_rt!, surface_albedo, surface_emissivity,
       AtmosphericColumn, column_from_simulation, column_from_profiles,
       stitch_columns, liquid_mass_extinction, ice_mass_extinction,
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

# Phase 32 — correlated-k gas optics
export MalkmusParams, malkmus_transmission, malkmus_log_transmission,
       malkmus_params, erf_series, erfc_series,
       malkmus_cumulative_k, kdist_from_band_model,
       CorrelatedKGasOptics, correlated_k_optics,
       rayleigh_optical_depth, rayleigh_optics!,
       lw_source!, lw_surface_source,
       column_absorber_amount, transmission_band, optical_depth_profile,
       us_standard_column

# Phase 33 — two-stream RT and cloud optics
export TwoStreamSolver, two_stream_solver,
       DeltaEddington, PIFM, Quadrature,
       delta_scale, two_stream_coefficients,
       layer_reflectance_transmittance_sw,
       layer_reflectance_transmittance_lw,
       adding_solver!, solve_sw!, solve_lw!, TwoStreamWorkspace,
       MieCloudOptics, mie_cloud_optics,
       liquid_water_path, ice_water_path, cloud_optical_depth,
       maximum_random_overlap, radiative_heating!

# Phase 36 — spectral albedo and BRDF
# `material_albedo` / `material_emissivity` are methods of the
# RadiationCore generics, exported there, not here.
export AbstractBRDF, BRDFSurface, CookTorranceBRDF, FresnelWater,
       LambertianBRDF, RPVBRDF, RossLiBRDF, SNOW_GROWTH_RATE,
       SNOW_REFERENCE_RADIUS, SNOW_SOOT_CONSTANT, SnowSurface,
       SpectralAlbedo, TiledSurface, band_average_reflectance,
       band_center_wavelength, black_sky_albedo, blue_sky_albedo,
       brdf, broadband_albedo, broadband_emissivity,
       cook_torrance_preset, cox_munk_slope_variance,
       diffuse_albedo_vector, direct_albedo_vector, emissivity_vector,
       fresnel_reflectance, fresnel_polarized_reflectance,
       hemisphere_quadrature_albedo,
       li_sparse_kernel, material_names, ross_thick_kernel,
       rossli_preset, rpv_preset, snow_albedo, snow_albedo_spectrum,
       snow_grain_radius, spectral_emissivity, spectral_reflectance,
       tile_components, tile_fractions, tiled_surface,
       water_albedo_diffuse, water_albedo_direct, white_sky_albedo,
       whitecap_fraction

# Phase 37 — Mie aerosol optics and radiative feedback
# The named mixture constructors (urban, desert, maritime_clean, ...)
# are deliberately NOT re-exported: bare names like `urban` and
# `desert` in every user's namespace are a collision waiting to
# happen. Use `aerosol_mixture(:urban)`, or the qualified form
# `Aeolus.AerosolOptics.urban()`.
export AEROSOL_MIXTURE_NAMES, AerosolBandOptics, AerosolMode,
       AerosolProfile, AerosolSpecies, HygroscopicTable, KELVIN_A_298,
       LogNormalMode, MieResult, aerosol_band_optics,
       aerosol_extinction_profile, aerosol_heating_rate,
       aerosol_mixture, aerosol_optical_depth,
       aerosol_optical_depth_550, aerosol_profile,
       aerosol_radiative_forcing, aerosol_species,
       aerosol_species_names, angstrom_exponent, bulk_optics,
       clear_optics_cache!, critical_surface_albedo, effective_radius,
       growth_factor, hygroscopic_table, interpolate_rh,
       kohler_saturation, lognormal_mode, mass_absorption_coefficient,
       mass_extinction_coefficient, mie_efficiencies, mie_sphere,
       mode_mass, mode_number, mode_optics, number_pdf,
       rayleigh_efficiencies, rebuild_bands, refractive_index,
       relative_humidity_profile, size_parameter,
       water_refractive_index, wet_radius, wet_refractive_index

# Phase 34 — 3-D Monte-Carlo radiative transfer
export MCDomain, MCMedium, MCResult, MonteCarloSolver,
       add_box_building!, domain_extent, domain_top,
       henyey_greenstein_phase, homogeneous_medium,
       mc_closure_residual, mc_domain, mc_energy_budget,
       mc_heating_rate, mc_medium, mc_solve, mc_sun_direction,
       mc_surface_irradiance, medium_from_optics, monte_carlo_solver,
       optical_depth_column, rayleigh_phase, rotate_scatter,
       sample_free_path, sample_henyey_greenstein, sample_lambertian,
       sample_rayleigh, scatter_direction, set_solid!,
       solid_from_mask!

# Phase 35 — urban radiosity
# Urban class greys are DERIVED from the Phase 36 spectral library
# via URBAN_CLASS_MIXTURES, so the two cannot drift apart.
export Facet, LongwaveSolution, RadiosityScene, ShortwaveSolution,
       URBAN_CLASS_MIXTURES, URBAN_PARK, URBAN_ROOF, URBAN_STREET,
       URBAN_WALL, URBAN_WATER, UrbanMaterials, UrbanRadiationForcing,
       ViewFactors, aggregate_to_grid, build_facets, build_scene,
       canyon_centreline_sky_view_factor, canyon_sky_view_factor,
       canyon_wallside_sky_view_factor, closure_error,
       default_urban_materials, diffuse_sky_irradiance,
       direct_beam_irradiance, effective_albedo, exchange_area,
       facet_albedos, facet_areas, facet_corners, facet_edges,
       facet_emissivities, facet_sample_points, facet_sunlit_fraction,
       facet_temperatures, heightmap_from_solid_mask,
       longwave_trapping_factor, LongwaveResponse, longwave_response,
       nfacets, ray_blocked,
       reciprocity_error, segment_blocked, set_temperature!,
       set_temperatures!, sky_view_factor_raycast,
       sky_view_factors_raycast, solar_unit_vector, solve_longwave,
       solve_radiosity_direct, solve_radiosity_iterative, RadiosityFactorCache,
       solve_shortwave, sunlit_fractions, total_sky_exchange_area,
       uniform_materials, urban_class_mixtures, view_factor_contour,
       view_factor_parallel_rectangles,
       view_factor_perpendicular_rectangles, view_factors

# =========================================================================
# Era IV — high-fidelity clouds and microphysics (Phases 39-45)
# =========================================================================

# Phase 39 — exact moist thermodynamics (the Era IV shared contract)
export ThermodynamicConstants, DEFAULT_CONSTANTS,
       latent_heat_vapor, latent_heat_sublim, latent_heat_fusion,
       saturation_vapor_pressure, saturation_vapor_pressure_ice,
       saturation_specific_humidity, saturation_mixing_ratio,
       saturation_specific_humidity_ice,
       vapor_pressure, relative_humidity, relative_humidity_ice,
       specific_humidity_from_relative_humidity,
       dewpoint_temperature, frost_point_temperature, wet_bulb_temperature,
       gas_constant_moist, heat_capacity_moist, exner_function,
       potential_temperature, temperature_from_theta,
       liquid_ice_potential_temperature, temperature_from_theta_li,
       virtual_temperature, density_temperature,
       virtual_potential_temperature, density_potential_temperature,
       equivalent_potential_temperature, wet_bulb_potential_temperature,
       moist_static_energy, liquid_water_static_energy,
       air_density, specific_volume,
       saturation_adjustment, saturation_adjustment_ice,
       PhasePartition, phase_partition, liquid_fraction,
       moist_adiabatic_lapse_rate, dry_adiabatic_lapse_rate,
       lifting_condensation_level, level_of_free_convection,
       cape_cin, buoyancy, brunt_vaisala_frequency_moist,
       saturated_buoyancy_frequency_squared,
       supersaturation, supersaturation_ice,
       kelvin_coefficient, kohler_supersaturation, kohler_critical_point,
       kohler_equilibrium_radius, surface_tension_water,
       kappa_from_solute, solute_from_kappa,
       moist_enthalpy, temperature_from_moist_enthalpy,
       isobaric_saturation_adjustment

# Phase 42 — aerosol-cloud interaction (Kohler activation)
# Kohler theory itself lives in the Phase 39 contract; the names here
# are the kappa/dry-radius entry points in micrometres, which is the
# unit AerosolOptics works in.
export ARGMode, ActivationCoefficients, ActivationResult,
       AerosolReservoir, KAPPA_MIN, ParcelResult, ParcelSettings,
       RHO_WATER, activate!, activated_fraction, activated_number,
       activation_coefficients, activation_susceptibility,
       aerosol_reservoir, aerosol_total_mass, aerosol_total_number,
       albedo_susceptibility, albrecht_lifetime_factor,
       arg_activation, arg_maximum_supersaturation, arg_mode,
       arg_modes, autoconversion_rate, autoconversion_suppression,
       cloud_albedo, cloud_droplet_number, cloud_optical_depth_lwp,
       cloud_water_path, critical_radius, critical_supersaturation,
       droplet_effective_radius, droplet_number_from_effective_radius,
       droplet_volume_radius, equilibrium_radius, growth_coefficient,
       interstitial_modes, interstitial_number, kelvin_A,
       kohler_critical, kohler_equilibrium_supersaturation,
       kohler_slope, kohler_stable, mass_above_radius,
       mode_critical_supersaturation, mode_dry_mass,
       number_above_radius, parcel_activation, process_collisions!,
       regenerate!, reservoir_dry_radius, supersaturation_sink,
       supersaturation_source, thermal_conductivity_air,
       twomey_albedo_change, twomey_radius_exponent,
       water_vapor_diffusivity

# Phase 41 — superdroplet Lagrangian microphysics
# The `droplet_*` Kohler names take SOLUTE MASS, which is what a
# superdroplet carries; they delegate to the same contract.
export AMMONIUM_SULFATE, AerosolProperties, CollisionKernel,
       DEFAULT_MICRO, EulerianCell, GolovinKernel,
       GravitationalKernel, HydrodynamicKernel, LongEfficiency,
       LongKernel, MicrophysicalConstants, NACL, SEASALT_KAPPA,
       SuperdropletSet, UnitEfficiency, add_superdroplet!,
       advance_positions!, aerosol_from_vant_hoff,
       apply_condensate_tendency!, assign_cells!, bessel_i1,
       bessel_i1_scaled, cell_air_mass, cell_from_state,
       cell_populations, cell_pressure, cell_supersaturation,
       cell_total_water, coalescence_probability, coalescence_step!,
       collision_coalescence!, collision_efficiency, collision_kernel,
       compact!, condensation_growth!, condensation_step!,
       deposit_liquid_water!, diffusional_growth_rate,
       droplet_critical_radius, droplet_critical_supersaturation,
       droplet_equilibrium_radius, droplet_kohler_supersaturation,
       droplet_mass, droplet_volume, dry_mass, dry_radius,
       effective_radius, fd_vapor, fk_thermal, golovin_T,
       golovin_mass_density, golovin_number_density, grow_droplet,
       init_exponential_superdroplets,
       init_monodisperse_superdroplets, kappa_kohler_supersaturation,
       kelvin_a, liquid_water_content, liquid_water_mixing_ratio,
       mass_distribution, mean_radius, nsuperdroplets,
       number_concentration, radius_moment, radius_percentile,
       raoult_b, relative_dispersion, size_distribution,
       surface_tension, terminal_velocity, thermal_conductivity,
       total_number, total_water_mass, vapor_diffusivity

# Phase 40 — P3 (predicted particle properties) bulk microphysics
# Gamma-function names carry `_lanczos` / `gamma_inc_*` spellings so they
# cannot collide with SpecialFunctions in user code that loads both.
export lgamma_lanczos, gamma_lanczos,
       gamma_inc_lower, gamma_inc_upper, gamma_inc_pq,
       gamma_psd_density, gamma_psd_moment, gamma_psd_partial_moment,
       gamma_psd_intercept, gamma_psd_slope
# P3 mass-dimension relation and rime density: `P3`/`p3_` prefixed, since
# generic names (`particle_mass`, `fall_speed`) are taken or too ambiguous.
export P3IceParameters, P3_DEFAULT_ICE, P3MassDimension,
       p3_rime_state, p3_mass_dimension, p3_regime,
       p3_particle_mass, p3_particle_area,
       p3_mean_mass, p3_ice_mass, p3_bulk_density,
       p3_slope, p3_size_distribution,
       p3_air_viscosity, p3_fall_speed, p3_bulk_fall_speed
# P3 process rates (issue #3)
export P3ProcessParameters, P3_DEFAULT_PROCESS, P3ParcelState,
       p3_total_water,
       p3_autoconversion, p3_accretion, p3_rain_self_collection,
       p3_cober_list_density, p3_rime_density,
       p3_condensation, p3_ventilation,
       p3_process_rates, p3_tendencies, p3_limit_rates, p3_step

# P3 sedimentation (per-moment fall speeds, sub-stepped column) and radar
# reflectivity; `p3_` prefixed next to Superdroplet's `terminal_velocity`.
export p3_rain_fall_speed, p3_rain_size_distribution, p3_rain_limit_number,
       p3_rain_bulk_fall_speed, p3_rain_rate, p3_bulk_fall_speed_z,
       P3_RAIN_D_MEAN_MIN, P3_RAIN_D_MEAN_MAX,
       P3_K2_WATER, P3_K2_ICE, p3_rain_reflectivity, p3_ice_reflectivity, p3_dbz,
       p3_sediment!, p3_rain_sedimentation_velocities, p3_sediment_rain!,
       p3_ice_sedimentation_velocities, p3_sediment_ice!
# P3 sub-grid precipitation fraction and overlap (Phase 44, issue #8)
export p3_precipitation_overlap, p3_precipitation_fraction,
       p3_fractional_process_rates, p3_step_column

# P3 secondary ice (issue #7): Hallett-Mossop rime splintering
export P3_NUCLEATED_ICE_MASS, P3_HM_SPLINTER_MASS, P3HallettMossop, P3_DEFAULT_HM,
       p3_hm_temperature_factor, p3_hm_large_drop_fraction, p3_hm_rimer_diameter,
       p3_hallett_mossop, p3_hm_tendencies, p3_step_with_rates

# Phase 43 — ice nucleation: prognostic INP species (dust, biological)
# and their available/consumed budget. All names are `inp_`-prefixed or
# otherwise specific so they cannot collide with the P3 ice category.
export DEMOTT_CUTOFF_RADIUS, INPPopulation, inp_budget,
       inp_budget_residual, inp_consume!, inp_emit!, inp_entrain!,
       inp_frozen_fraction, inp_mass, inp_median_radius, inp_modes,
       inp_nmodes, inp_nucleate_singular!, inp_number,
       inp_number_above_radius, inp_population, inp_release!,
       inp_scavenge!, inp_sediment!, inp_settling_velocity,
       inp_species, inp_surface_area, inp_surface_area_above_radius,
       mode_surface_area, surface_area_above_radius
# Phase 43 — immersion and deposition freezing: schemes and source terms.
export AerosolCountScheme, DeMott2015, FreezingScheme, IceNucleationTendency,
       NUCLEATED_ICE_MASS, Niemand2012, STANDARD_PRESSURE,
       STANDARD_TEMPERATURE, SiteDensityScheme, Tobo2013,
       Ullrich2017Deposition, Ullrich2017Immersion,
       default_freezing_scheme, default_freezing_schemes, freezing_mechanism,
       freezing_regime, ice_active_site_density, ice_nucleation!,
       ice_number_formed, inas_frozen_fraction, nucleation_rates,
       scheme_temperature_range, singular_inp_concentration
# Phase 43 (issue #7) — nucleation sources into the P3 ice category.
export p3_ice_nucleation, p3_inp_release!,
       ArcticStratusColumn, isdac_initial_profile, run_arctic_stratus,
       glaciation_time

# Phase 45 — turbulence-microphysics interaction
# Sub-grid saturation-deficit PDF, partial cloud fraction and condensate
# (issue #12). Thermodynamics is the Phase 39 contract's.
export AbstractSaturationPDF, GaussianSaturationPDF,
       DoubleGaussianSaturationPDF,
       saturation_pdf_density, exceedance_fraction,
       pdf_cloud_fraction, pdf_condensate,
       MellorCoefficients, mellor_coefficients, saturation_deficit_variance,
       extended_condensate, SubgridCloud, subgrid_cloud,
       GEOMETRIC_VARIANCE_COEFF, equilibrium_variance_coefficient,
       filter_width, sgs_scalar_covariance!, sgs_scalar_variance!,
       subgrid_cloud!
# Sub-grid supersaturation in activation (Phase 42 CCN spectrum) and
# condensational growth (Phase 41 superdroplets) (issue #13).
export ccn_fraction, ccn_spectrum,                  # defined in AerosolCloud
       pdf_expectation, sgs_ccn_fraction, sgs_ccn_spectrum,
       SGSActivationResult, sgs_activation,
       SupersaturationFluctuations, supersaturation_fluctuations,
       advance_fluctuations!, local_supersaturation,
       supersaturation_integral_variance,
       sgs_condensation_growth!, sgs_condensation_step!
# Spectral broadening: bulk moment closure of the sub-grid broadening,
# validated against the superdroplet reference and DNS (issue #14).
# mean_radius / relative_dispersion / effective_radius gain methods for
# the bulk state; they remain the Superdroplet / MoistThermodynamics
# functions.
export SpectralMoments, spectral_moments, gamma_moment_ratio,
       closure_radius_moment, r2_standard_deviation,
       bulk_condensation_step!, parcel_ascent!,
       BroadeningCase, BroadeningHistory, run_broadening_case,
       broadening_initial_population, fit_power_law

# Phase 44 — visibility diagnostic (Koschmieder)
# Mie, haze and Rayleigh are delegated to AerosolOptics / GasOptics;
# `rayleigh_cross_section` is the GasOptics definition, re-exported here.
export KOSCHMIEDER_THRESHOLD, MOR_THRESHOLD, VISIBILITY_WAVELENGTH,
       koschmieder_visibility, extinction_from_visibility,
       DropletSpectrum, MonodisperseDroplets, GammaDroplets, BinnedDroplets,
       gamma_droplets, droplet_spectrum, spectrum_number, spectrum_moment,
       spectrum_water_content, water_extinction_efficiency,
       droplet_extinction, fog_extinction, aerosol_extinction,
       rayleigh_extinction, rayleigh_cross_section, total_extinction,
       visibility, kunkel_visibility, gultepe_visibility

# Phase 44 — fog / dew condensation coupled to the vapour field
export relaxation_fraction, fog_equilibrium, fog_condensation, fog_adjust!,
       dew_deposition, dew_exchange!
# Phase 44 (#9) — droplet-radiative feedback: fog droplets in the Era III
# radiation (delegated to GasOptics/TwoStream/RadiationCore)
export fog_effective_radius, FogRadiation, fog_radiation,
       fog_atmospheric_column, fog_radiative_transfer!,
       fog_radiative_heating!, fog_top_flux_jump,
       radiative_condensation_rate,
       FogColumnSettings, FogBudget, FogColumn, stretched_levels,
       parisfog_2011_column, column_energy, column_water,
       fog_liquid_water_path, update_radiation!, moist_stability,
       eddy_diffusivity, surface_exchange, settling_velocity,
       fog_diagnostics, fog_life_cycle
# (the fog column adds methods to the generic `step!` and `run!`)

# =========================================================================
# Era V — high-fidelity land, urban and surface exchange (Phases 46-53)
# =========================================================================

# Era V contract — SurfaceExchange: material constants, Monin–Obukhov
# similarity, and the AbstractSurface interface. `surface_temperature`
# (exported above with SlabLand) is its generic; `surface_albedo` /
# `surface_emissivity` are the RadiationCore generics (exported above).
# Constant accessors (LH_v0, grav, cp_water, …) stay qualified:
# `Aeolus.SurfaceExchange.grav(c)`.
export SurfaceConstants, DEFAULT_SURFACE_CONSTANTS,
       AbstractStabilityFunctions, BusingerDyerBeljaarsHoltslag, BusingerDyerLogLinear,
       phi_m, phi_h, psi_m, psi_h,
       SurfaceLayerSolution, solve_surface_layer,
       bulk_richardson, obukhov_length,
       AbstractSurface, SurfaceForcing, SurfaceFluxes,
       roughness_lengths, moisture_availability,
       surface_specific_humidity, surface_latent_heat,
       net_radiation, turbulent_fluxes, step_surface!

# Phase 46 (#15) — SoilColumn: CLM-style soil column, Richards equation.
# Boundary-condition types (FreeDrainage, HeadTop, AtmosphericTop, …),
# hooks and the generic-sounding accessors (saturated_conductivity,
# air_entry_potential, …) stay qualified: `Aeolus.SoilColumn.FreeDrainage`,
# or `using Aeolus.SoilColumn`.
export SoilGrid, clm5_soil_grid, clm_exponential_soil_grid, uniform_soil_grid,
       AbstractHydraulicModel, ClappHornberger, VanGenuchtenMualem, GardnerExponential,
       clapp_hornberger_texture, carsel_parrish_texture, cosby_clapp_hornberger,
       soil_water_content, soil_matric_potential, soil_hydraulic_conductivity,
       soil_water_capacity, soil_water_diffusivity, soil_porosity,
       parlange_sorptivity, water_dynamic_viscosity,
       SoilWaterColumn, step_soil_water!, set_water_content!, set_matric_potential!,
       soil_water_storage, soil_layer_conductivity, soil_field_capacity

# Phase 46 (#16) — SoilColumn: soil heat, freeze–thaw, and the column as an
# AbstractSurface. Heat boundary types (GroundHeatFluxTop, BottomTemperature,
# …) and the Johansen/Farouki helpers stay qualified, as for #15.
export SoilThermalProperties, clm_soil_thermal_properties, johansen_soil_thermal_properties,
       soil_thermal_conductivity, soil_heat_capacity,
       freezing_matric_potential, supercooled_liquid_content, PowerLawIceImpedance,
       SoilThermalColumn, step_soil_heat!, set_soil_temperature!,
       soil_enthalpy, soil_ice_mass

# Phase 46 (#17) — SoilColumn: water table / unconfined aquifer (SIMTOP).
# The helpers aquifer_impermeable_fraction, gardner_two_point_flux and
# bottom_water_storage stay qualified.
export UnconfinedAquifer, water_table_depth, set_water_table!,
       aquifer_baseflow_rate, total_water_storage

# Phase 47 — Photosynthesis: Farquhar C3 / Collatz C4 leaf biochemistry,
# Bernacchi / Kattge-Knorr temperature responses, CLM4.5 PFT parameters.
export AbstractTemperatureResponse, ArrheniusResponse, PeakedArrheniusResponse,
       Q10Response, temperature_factor, optimum_temperature,
       C3TemperatureResponses, bonan2011_c3_temperature,
       kattge_knorr_c3_temperature, kattge_knorr_jmax_ratio,
       PAR_PHOTONS_PER_JOULE, par_to_photon_flux, colimited_rate,
       electron_transport_rate,
       AbstractPhotosynthesisParameters, C3Parameters, C4Parameters, collatz1992_c4,
       leaf_capacities, photorespiratory_compensation_point,
       LeafAssimilation, leaf_assimilation, net_assimilation, co2_compensation_point,
       PFTPhotosynthesis, CLM45_PFT_PHOTOSYNTHESIS, vcmax25_from_leaf_nitrogen,
       pft_photosynthesis_parameters

# Phase 47 — stomatal conductance (#20): Ball-Berry / Medlyn closures,
# leaf boundary layer, transpiration, coupled A–gs–ci solution.
export AbstractStomatalModel, BallBerryStomata, MedlynStomata, stomatal_conductance,
       leaf_boundary_layer_conductance, molar_conductance, leaf_transpiration,
       STOMATAL_DIFFUSIVITY_RATIO, BOUNDARY_LAYER_DIFFUSIVITY_RATIO,
       LeafGasExchange, leaf_gas_exchange, apply_water_stress,
       CLM5_MEDLYN_G1, pft_stomatal_model

# Phase 47 (#21) — LandSurface: vegetated land column (two-big-leaf canopy
# over a SoilThermalColumn), root-zone soil-water stress, big-leaf radiation.
export VegetationParameters, pft_vegetation,
       zeng_root_fractions, AbstractSoilWaterStress, CLMWaterStress,
       wilting_factor, soil_water_stress, root_water_uptake,
       BigLeafShortwave, bigleaf_shortwave, BigLeafLongwave, bigleaf_longwave,
       canopy_roughness, undercanopy_transfer_coefficient,
       VegetatedLandColumn, LandSurfaceDiagnostics, LandSurfaceBudget,
       set_leaf_area!, radiative_temperature

# Phase 48 — PlantCanopy: leaf-area-density profiles and implicit
# (unconditionally stable) canopy drag.
export AbstractLeafAreaDensity, UniformLAD, LalicMihailovicLAD, BetaLAD, TabulatedLAD,
       leaf_area_density, leaf_area_index, canopy_height, cumulative_leaf_area,
       layer_leaf_area_density,
       canopy_drag_factor, canopy_drag_force, implicit_canopy_drag,
       ResolvedCanopyDrag, canopy_lad_field, apply_canopy_drag!,
       CanopyWindSolution, solve_canopy_wind_column,
       harman_finnigan_mixing_length, canopy_attenuation_coefficient,
       canopy_exponential_wind,
       AbstractLeafAngleDistribution, RossGoudriaanLeaves, SphericalLeaves,
       HorizontalLeaves, LeafOptics, leaf_projection_function,
       beam_extinction_coefficient, average_inverse_optical_depth,
       canopy_scattering_parameters, canopy_beam_transmittance,
       sunlit_leaf_fraction, canopy_diffuse_transmittance,
       CanopyShortwaveSolution, canopy_shortwave,
       CanopyLongwaveSolution, canopy_longwave, canopy_shortwave_field!

# Phase 48 — leaf energy balance (#24): leaf temperature, boundary layer,
# transpiration and sensible heat of resolved-canopy leaves.
export AbstractLeafBoundaryLayer, CampbellNormanBoundaryLayer, CLMLeafBoundaryLayer,
       leaf_boundary_layer_conductances,
       AbstractLeafStomata, FixedStomatalConductance, CoupledLeafPhysiology,
       LeafEnergyBalance, leaf_energy_balance, leaf_penman_monteith,
       CanopyLeafEnergySolution, canopy_leaf_energy_balance

# Phase 48 — within-canopy profiles (#25): scalar K-theory column,
# displacement height, tabulated (measured) LAD profiles.
export CanopyScalarSolution, solve_canopy_scalar_column, canopy_displacement_height,
       canopy_mixing_length, harman_finnigan_scalar_diffusivity

# Phase 49 — UrbanSurface: per-facet multi-layer wall conduction. The
# material-property accessors (volumetric_heat_capacity,
# thermal_diffusivity, thermal_admittance) are generic names other Era V
# modules may want, so they stay qualified: `Aeolus.UrbanSurface.thermal_admittance`.
export ThermalMaterial, thermal_material, thermal_material_names,
       WallLayer, thermal_resistance, thermal_transmittance,
       ISO6946_SURFACE_RESISTANCE,
       ColumnBoundary, fixed_temperature_boundary, heat_flux_boundary,
       convective_boundary,
       ConductionColumn, step_conduction!, face_temperature,
       column_heat_content, cell_centers,
       UrbanFacet, interior_heat_flux, set_interior_boundary!,
       set_interior_temperature!, facet_layer_temperatures,
       example_roof_layers, example_wall_layers, example_ground_layers,
       UrbanFacetScene, step_urban_facets!, aggregate_facet_fluxes,
       example_facet_layers

# Phase 49 — UrbanSurface glazing (#27): angle-dependent pane / unit
# optics, gap heat transfer, U-value, SHGC and the window facet. The
# fill-gas property fits (gas_conductivity, …) stay qualified.
export CLEAR_FLOAT_REFRACTIVE_INDEX, CLEAR_FLOAT_EXTINCTION, UNCOATED_GLASS_EMISSIVITY,
       GlassPane, glass_pane_from_normal, pane_optics,
       FillGas, GLAZING_AIR, GLAZING_ARGON, GlazingGap, gap_nusselt, gap_conductance,
       Glazing, single_glazing, double_glazing, triple_glazing,
       glazing_optics, glazing_optics_hemispherical,
       glazing_u_value, inward_flowing_fractions, glazing_shgc,
       WindowFacet, set_solar_incidence!, window_optics, transmitted_shortwave,
       pane_temperatures, window_heat_content

# Phase 50 — BuildingEnergy: prognostic indoor air temperature, thermal
# mass (UrbanSurface conduction columns), ventilation, and the ISO 13790
# 5R1C reference model.
export AbstractZoneElement, EnvelopeElement, FacetElement, InternalMass,
       BuildingZone, ZoneForcing, ZoneBudget, step_zone!, facet_fluxes,
       zone_air_temperature, zone_heat_content, zone_heat_loss_coefficient,
       zone_mean_radiant_temperature, zone_operative_temperature,
       air_heat_capacity, ventilation_conductance, element_transmittance,
       iso13786_areal_heat_capacity,
       ISO13790Zone, iso13790_zone, step_iso13790!, iso13790_ideal_load!,
       zone_specific_humidity, zone_relative_humidity,
       BiquadraticCurve, DXCoolingCurves, DX_RATING_POINT,
       DX_CURVES_WINDOW_AC, DX_CURVES_PACKAGED_AC,
       IdealLoadsHVAC, HVACBudget, condenser_inlet_temperature,
       cooling_cop, available_cooling_capacity

# Phase 50 — BuildingEnergy: occupancy, lighting and equipment schedules and
# internal gains (#33).
export DayProfile, WeeklySchedule, schedule_value, schedule_mean, full_load_hours,
       occupant_sensible_heat, InternalGains, InternalGainRates, internal_gains,
       ThermostatSchedule, scheduled_hvac, DOE_MEDIUM_OFFICE_SCHEDULES,
       office_internal_gains, office_thermostat

# Phase 49 — anthropogenic heat from the building energy model (#29).
export BuildingHeatRelease, building_heat_release, energy_consumption, total_heat_release,
       anthropogenic_sensible_heat, anthropogenic_latent_heat,
       step_buildings!, building_facets, scene_building_zone, anthropogenic_heat_maps

# Phase 49 — TEB canyon-exchange closures used by the BUBBLE / CAPITOUL
# validation (#30).
export canyon_wind_speed, rowley_convection_coefficient

# Phase 51 (#34) — Snowpack: layered snowpack with compaction, heat
# conduction and melt; `LayeredSnowpack <: AbstractSurface`.
export AbstractSnowConductivity, Yen1981Conductivity, Sturm1997Conductivity,
       Jordan1991Conductivity, Calonne2011Conductivity, snow_thermal_conductivity,
       AbstractFreshSnowDensity, Anderson1976FreshSnow, Vionnet2012FreshSnow,
       fresh_snow_density,
       AbstractSnowCompaction, DestructiveMetamorphism, Anderson1976Compaction,
       Vionnet2012Compaction, snow_compaction_rate, snow_viscosity,
       AbstractSnowMetamorphism, NoSnowMetamorphism, snow_metamorphism!,
       viscosity_grain_factor,
       AbstractSnowAlbedo, ConstantSnowAlbedo, snowpack_albedo,
       AbstractSnowLiquidWater, InstantSnowRunoff, snow_liquid_water!,
       AbstractSnowBottom, SnowGroundTemperature, SnowBottomFlux,
       SnowLayering, LayeredSnowpack, SnowBudget,
       snow_nlayers, snow_depth, snow_water_equivalent, snow_bulk_density,
       snow_layer_density, snow_layer_depths, snowpack_enthalpy, snow_load,
       snow_compaction!, snow_heat_conduction!, snow_phase_change!,
       add_snowfall!, add_rainfall!, snow_sublimation!, relayer_snowpack!

# Phase 51 (#36) — Snowpack: liquid-water retention, percolation, refreezing.
export AbstractSnowWaterRetention, ColeouLesaffre1998Retention, PoreFractionRetention,
       snow_irreducible_water, snow_water_holding_capacity, snow_liquid_water_content,
       snow_saturated_hydraulic_conductivity, snow_refreeze!,
       SnowBucketPercolation, SnowGravityDrainage

# Phase 51 (#35) — Snowpack: grain metamorphism (Flanner & Zender 2006 /
# Brun 1989) and albedo ageing through SurfaceOptics.SnowSurface.
export fresh_grain_radius, refrozen_grain_radius, mix_refrozen_grain!,
       SnicarAgingTable, snicar_aging_table, snicar_aging_parameters,
       snicar_grain_radius, flanner_grain_radius,
       Flanner2006Metamorphism, ParabolicGrainGrowth,
       snow_dry_grain_growth_rate, snow_wet_grain_growth_rate,
       snow_temperature_gradient, snow_specific_surface_area, snow_optical_radius,
       SnowOpticsAlbedo, snowpack_optics, snowpack_optical_radius

# Phase 51 — IceThermodynamics (#37): Bitz & Lipscomb (1999) energy-
# conserving lake and sea ice with brine pockets, snow, melt and growth.
export IceParameters, IceColumn, sea_ice_column, lake_ice_column,
       seawater_freezing_point, bl99_salinity_profile,
       ice_melting_temperature, ice_energy_of_melting, ice_enthalpy,
       ice_temperature_from_enthalpy, ice_heat_capacity, ice_conductivity,
       ice_snow_enthalpy,
       PrescribedSurfaceTemperature, PrescribedSurfaceFluxes, IceStepBudget,
       step_ice!, ice_thickness, ice_snow_depth, ice_layer_temperatures,
       ice_column_enthalpy, ice_freeboard,
       stefan_ice_thickness, neumann_lambda, neumann_ice_thickness

# Phase 52 (#38) — OceanMixedLayer: lake / coastal-sea water column with a
# Kraus–Turner mixed layer, shortwave penetration, cool skin and warm layer.
export AbstractWaterEOS, UNESCOWaterEOS, LinearWaterEOS,
       water_density, water_thermal_expansion, water_haline_contraction,
       SolarBands, jerlov_water, soloviev_solar,
       solar_transmission, solar_absorbed_between, solar_mean_transmission,
       CoolSkin, cool_skin,
       AbstractWarmLayer, NoWarmLayer, FairallWarmLayer, ZengBeljaarsWarmLayer,
       step_warm_layer!, reset_warm_layer!, warm_layer_warming, warm_layer_depth,
       KrausTurnerMixing, WaterColumn, layer_depths,
       absorb_heat!, convective_adjustment!, wind_mixing!, mix_column!,
       column_heat_content, column_potential_energy, mixed_layer_depth,
       salinity_humidity_factor

# Phase 52 — AirSeaFlux (#39): wave-dependent sea-surface roughness, COARE
# stability / gustiness, COARE-style bulk fluxes, SeaSurface. The viscosity
# helper stays qualified: `Aeolus.AirSeaFlux.air_kinematic_viscosity`.
export COAREStability, ConvectiveGustiness, gustiness_velocity,
       convective_velocity_scale,
       SeaState, deep_water_sea_state, fully_developed_sea_state, wave_age,
       AbstractSeaRoughness, AbstractCharnockRoughness, CharnockRoughness,
       COARE30Charnock, COARE35Charnock, WaveAgeCharnock, OostRoughness,
       TaylorYellandRoughness, SeaStateRoughness,
       smooth_flow_roughness, charnock_parameter, sea_roughness_length,
       AbstractScalarRoughness, COARE30ScalarRoughness, COARE35ScalarRoughness,
       BrutsaertGarrattRoughness, sea_scalar_roughness, roughness_reynolds,
       neutral_transfer_coefficients,
       BulkAirSea, coare30_bulk, coare35_bulk, AirSeaSolution, air_sea_fluxes,
       SeaSurface, sea_surface_solution

# Phase 52 — AirSeaFlux (#40): Andreas sea spray, rain heat flux, the full
# COARE algorithm (cool skin + warm layer) and its WaterColumn coupling.
export andreas_friction_velocity, AndreasSpray, spray_wind_functions,
       spray_dry_radius, spray_equilibrium_temperature, spray_radius_evolution,
       spray_residence_time, sea_spray_fluxes, rain_heat_flux,
       COARESolution, coare_fluxes, coare_warm_layer_series, coare_flux_hook

# Phase 53 — sub-grid surface tiles (#41): flux aggregation over a
# mosaic of AbstractSurface tiles.
export SurfaceMosaic, ntiles, mosaic_fractions, set_mosaic_fractions!,
       aggregate_fluxes, tile_fluxes, last_tile_fluxes,
       effective_radiative_temperature, tile_mean_temperature,
       ParameterAggregatedSurface, parameter_aggregate,
       class_fractions, tiles_from_class_map

# Phase 53 — blending height (#42): effective roughness, blending-height
# closures, internal-boundary-layer growth, tiles forced at l_b.
export effective_roughness, effective_scalar_roughness,
       AbstractBlendingHeight, FixedBlendingHeight, MasonBlendingHeight,
       ClaussenBlendingHeight, BouZeidBlendingHeight, IBLBlendingHeight,
       blending_height,
       AbstractIBLModel, ElliottIBL, PowerLawIBL, DiffusionIBL, ibl_height,
       BlendingHeightSolution, blended_forcing, BlendedMosaic, last_blending

# =========================================================================
# Era VI — high-fidelity turbulence closure (Phases 54-60)
# =========================================================================

# Phase 58 — TurbulenceStability contract (#170): gradient-Richardson
# stability functions, N² and Ri_g (pointwise fields live in Operators).
export AbstractRiStability, stability_factor_m, stability_factor_h, neutral_prandtl,
       LillyCutoff, BrownEtAl1994, ShortTail, LongTail,
       gradient_richardson, buoyancy_frequency_squared, RI_SHEAR2_FLOOR
# N-parent-a (#247): the one boundary-layer-height diagnostic
export boundary_layer_height
# SurfaceExchange's implementer of the contract (MOST-consistent factors).
export MOSTConsistent, most_gradient_richardson, most_stability_parameter
# P58-e (#193): the maximum sustainable heat flux (Van de Wiel et al. 2012)
# and the flux-prescribed MOST friction velocity, in SurfaceExchange.
export log_linear_slope, MSHFSolution, mshf_solution, max_sustainable_heat_flux,
       flux_friction_velocity, surface_mshf_ratio
# Operators' pointwise fields of the contract.
export buoyancy_frequency_squared!, gradient_richardson!, strain_rate_squared!

# Phase 56 — PrognosticSGS (#182): the Deardorff 1.5-order prognostic SGS TKE
# closure (an AbstractSGSClosure) and its pointwise physics.
export TKEClosure, tke_length_scale, tke_dissipation_coefficient, tke_surface_value,
       tke_critical_richardson, tke_equilibrium, dual_strain!, shear_production!,
       buoyancy_production!, tke_budget, initialize_tke!, tke_statistics,
       tke_eddy_coefficients,
       attach_immersed_walls!   # #232: facade shear production from the wall-model stress
# P56-d (#188): the differential SGS Reynolds-stress closure.
export ReynoldsStressClosure, reynolds_stress_terms, reynolds_stress_rhs, reynolds_stress_step,
       daly_harlow_coefficient, coriolis_rotation, rotate_stress, is_realizable,
       project_realizable, stress_anisotropy, stress_edges!, dual_velocity_gradient,
       reynolds_stress_statistics, reynolds_stress_budget, initialize_reynolds_stress!,
       reynolds_stress_shear_equilibrium

# Phase 59 — Transition (#180): turbulent fraction γ(z), intermittency,
# Ri_f / Re_b / V_TKE diagnostics, growth-rate fit. Ri_g is P58-a's.
export turbulent_fraction, turbulent_indicator!, laminar_mask!,
       intermittency_factor, flux_richardson, buoyancy_reynolds,
       turbulence_velocity_scale, growth_rate, supercritical_fraction,
       grid_reynolds_number
# P59-e (#195): relaminarisation criteria (L is SurfaceExchange.obukhov_length)
export friction_reynolds, relaminarisation_criteria, relaminarised_fraction

# P59-d (#185) — spatially developing flat-plate boundary layer: Blasius,
# fringe region, free-stream disturbances, BL integral diagnostics.
export BlasiusSolution, blasius_solution, blasius_profile, blasius_velocity,
       blasius_skin_friction, turbulent_skin_friction,
       FringeRegion, fringe_rate, add_fringe!, fringe_forcing,
       blasius_target, set_blasius!, blasius_freestream, fringe_distance, flat_plate_setup,
       streamwise_profile, edge_velocity, boundary_layer_edge, skin_friction, displacement_thickness,
       momentum_thickness, shape_factor, transition_onset,
       onset_reynolds_theta_mayle, onset_reynolds_theta_abu_ghannam_shaw,
       FreeStreamDisturbance, von_karman_spectrum, disturbance_velocity!,
       disturbance_fringe_update
# N-inflow-a (#252) — synthetic inflow turbulence: the generator contract,
# the Xie & Castro (2008) digital filter, the Lund (1998) transform, and
# FreeStreamDisturbance as a second provider.
export AbstractInflowGenerator, unit_fluctuations!, generator_state, restore_generator_state!,
       inflow_plane_size, DigitalFilterGenerator, digital_filter_kernel, lund_coefficients,
       apply_lund!, FreeStreamInflow, disturbance_velocity, disturbance_covariance
# N-inflow-b (#258) — statistics for the synthetic inflow: parent-resolved
# (resolved + SGS, scale gap) and parametrized (Rotach et al. 1996 blend).
export InflowStatistics, inflow_stresses, inflow_lengths, update_inflow_statistics!,
       inflow_projection_log, realizable_stress, inflow_lund_coefficients,
       ParentResolvedStatistics, inflow_stress_parts, ParametrizedStatistics, parametrized_scales
# N-inflow-c (#264) — the synthetic turbulence applied on the inflow faces of
# an open-boundary child, with zero added mass flux.
export InflowTurbulence, inflow_normal_length, set_integral_time!, remove_inflow_turbulence!,
       refresh_inflow_lengths!, added_mass_flux, injected_ke_flux, inflow_perturbation,
       feedback_exclude_width
# N-inflow-d (#266) — the development length (fetch) behind a synthetic inflow.
export FetchStatistics, accumulate_fetch!, fetch_profiles, development_length
# Phase 57 — wall models (#169): equilibrium wall laws and the implicit
# wall-stress operator. `friction_velocity` / `wall_stress` do not collide
# with `andreas_friction_velocity` (AirSeaFlux) or `wall_shear_stress`
# (CutCell, the resolved stress).
export AbstractWallLaw, SpaldingLaw, RoughLogLaw, MOSTWallLaw, LEE_MOSER_2015,
       friction_velocity, wall_stress, wall_drag,
       spalding_yplus, spalding_uplus, matching_obukhov_length,
       WallModel, apply_wall_stress!
# P58-e (#193): the flux-prescribed MOST wall law (collapses beyond the MSHF)
export MOSTFluxWallLaw
# Phase 57-d (#179): the wall model on immersed (IBM mask) and cut-cell
# walls. The cut-cell stress is a method of CutCell.wall_shear_stress, not
# a new name.
export ImmersedWalls, immersed_walls, remove_stencil_wall_flux!,
       apply_immersed_wall_stress!, immersed_wall_count, immersed_wall_drag!,
       wall_slip_velocity, slip_wall_value
# P54-d (#174): the wall-bounded benchmark harness (half channel, channel).
# `advance_channel!` / `run_channel!` do not collide with the Berlin case's
# `step_case!` / `run_case!` (examples/urban/BerlinCase.jl).
export WallBoundedCase, half_channel_case, channel_case,
       advance_channel!, run_channel!, channel_dt,
       channel_richardson, stratified_channel_walls   # P55-f (#191): stratified channel
# P57-b (#183): the wall-model input — matching height, temporal and
# wall-parallel filters (log-layer mismatch).
export MatchingInput, matching_distance, matching_input!, update_input_filter!,
       exponential_filter_weight, wall_parallel_filter!
# P57-c (#189): the non-equilibrium (TBLE) wall model with dynamic κ_wm.
# `update_wall_model!` / `tble_*` collide with nothing.
export AbstractWallModel, AbstractTBLEEddyViscosity, DampedMixingLength, NoEddyViscosity,
       tble_eddy_viscosity, TBLEColumn, tble_grid, tble_solve!, tble_column_momentum,
       TBLEWallModel, update_wall_model!, tble_kappa, tble_columns

# P59-a (#172) — no-slip / moving walls; wall_shear_stress(sim; side) is a
# method of the CutCell generic (no new name).
export NoSlipWall
include("wall_shear_stress.jl")
# P59-d (#185) — free-stream (transpiration) top of a spatial boundary layer.
export FreeStreamBoundary
# N-open-b (#255) — open (Dirichlet-from-boundary-data) faces, the
# boundary-data contract and the global boundary mass-flux correction.
export AbstractOpenBoundary, OpenBoundary, OpenPlane, OpenBoundaryState,
       AbstractBoundaryData, ProfileBoundaryData, FunctionBoundaryData,
       boundary_values!, open_boundary_conditions, set_boundary_data!, add_open_scalars!,
       has_open_boundaries, open_boundary_states, update_open_boundaries!,
       prepare_open_boundaries!, correct_boundary_mass_flux!, open_boundary_flux,
       boundary_flux_weights, open_boundary_ledger, pinned_face_indices,
       open_boundary_snapshot, restore_open_boundaries!, combine_open_boundaries!,
       assert_open_face_correction, open_intermediate_velocity!
# #329 — the two-snapshot (windowed) boundary data: nest parents, mesoscale (#58).
export WindowedBoundaryData, snapshot_values, window_fraction, blend_window!
# N-open-c (#260) — convective / Orlanski radiation outflow on open faces.
export RadiationPlane, OUTFLOW_MODES, has_radiating_outflow, rewind_open_boundaries!,
       mark_open_boundary_stage!
# N-parent-b (#254) / N-open-d (#262) — θ's faces without its top gradient
# (qv, the SGS TKE); the one BC-aware neighbour index (replaces the periodic
# `mod1` wraps); solid obstacles at open faces; the sponge target of an open
# top (`Simulations`).
export passive_scalar_bcs, neighbor_index, mark_open_boundary_solids!, sponge_target

# Phase 58 — LargeScaleForcing (#171): geostrophic pressure gradient,
# nudging, prescribed advective tendencies, surface-temperature series.
# `surface_temperature(series, t)` is a method of the SurfaceExchange
# generic, already exported.
export ForcingTable, forcing_value, forcing_mean,
       AbstractLargeScaleForcing, GeostrophicPressureGradient, GeostrophicNudging,
       AdvectiveTendencies, SurfaceTemperatureSeries, apply_forcing!,
       ReferenceDamping, damping_tendency
# #248 (N-parent-c): plane-mean nudging of u, v, θ, qv and large-scale subsidence.
export ProfileNudging, NudgingWeight, nudging_weight, Subsidence, tendency_integrals
# N-parent-a (#247): radiosonde soundings → (z, t) profiles
export Sounding, SoundingLaunch, read_sounding, sounding_profiles, sounding_time

# P57-e (#190): the dynamic roughness of partially resolved relief
# (Anderson & Meneveau 2011). `surface_gradient_drag` does not collide
# with `CutCell.wall_shear_stress` or the PlantCanopy `canopy_drag_*` names.
export SubfilterHeight, subfilter_height, DynamicRoughness,
       surface_gradient_drag, roughness_surface_drag, roughness_drag_residual, dynamic_alpha,
       update_dynamic_roughness!, with_roughness, dynamic_wall_laws, dynamic_wall_laws!

# CFG-b (#357): the command line. `julia --project -m Aeolus <command> …`
# (Julia ≥ 1.12) calls `Aeolus.main(ARGS)`; `bin/aeolus` does the same on
# older Julia. The return value is the exit code.
"""
    Aeolus.main(args::Vector{String}; io = stdout, err = stderr) -> Cint

The `aeolus` command line ([`Aeolus.Config.main`](@ref)): `julia --project
-m Aeolus run config.toml`, or `bin/aeolus run config.toml`.
"""
main(args::AbstractVector{<:AbstractString} = ARGS; kwargs...) = Config.main(args; kwargs...)
@static if VERSION >= v"1.11"
    @main
end

end # module
