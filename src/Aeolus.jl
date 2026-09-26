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
include("NetCDFIO.jl")        # dependency-free NetCDF-3 reader/writer
include("IBM.jl")
include("Parallel.jl")
include("Nesting.jl")
include("Backends.jl")
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
include("AdaptiveMesh.jl")      # Phases 30-31

# --- Era III: high-fidelity radiation (Phases 32-38) ---------------------
# MoistThermodynamics (Era IV, Phase 39) is dependency-free and defines the
# canonical Kelvin coefficient and Köhler theory, which AerosolOptics needs,
# so it is included ahead of the radiation stack rather than with its own era.
include("MoistThermodynamics.jl")  # Phase 39
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

include("AerosolCloud.jl")         # Phase 42
include("Superdroplet.jl")         # Phase 41
include("P3Microphysics.jl")       # Phase 40
include("Microphysics.jl")         # Phase 12 Kessler; delegates rain fall speeds to P3 (Phase 44)
include("IceNucleation.jl")        # Phase 43
include("TurbulentMicrophysics.jl") # Phase 45
include("Visibility.jl")           # Phase 44 (visibility diagnostic)
include("RadiationFog.jl")         # Phase 44 (fog/dew coupling)

# --- Era V: high-fidelity land, urban and surface exchange (Phases 46-53)
include("Photosynthesis.jl")       # Phase 47 (Farquhar C3 / Collatz C4)

using .Grids
using .Fields
using .BoundaryConditions
using .Operators
using .TimeSteppers
using .PressureSolvers
using .Simulations
using .IO
using .NetCDFIO
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

using .MoistThermodynamics
using .AerosolCloud
using .Superdroplet
using .P3Microphysics
using .IceNucleation
using .TurbulentMicrophysics
using .Visibility                  # Phase 44
using .RadiationFog                # Phase 44

using .SurfaceExchange             # Era V contract
using .Photosynthesis              # Phase 47

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
export NCWriter, NCFile, nc_create, nc_defdim!, nc_defvar!, nc_endef!,
       nc_write!, nc_append_record!, nc_close!, nc_write_fields,
       nc_open, nc_varnames, nc_dimnames, nc_read, nc_read_record,
       nc_attributes, nc_nrecords
export apply_kessler_step!, KESSLER_N0, kessler_rain_number, kessler_fall_speed
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
       adding_solver!, solve_sw!, solve_lw!,
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
       fresnel_reflectance, hemisphere_quadrature_albedo,
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
       longwave_trapping_factor, nfacets, ray_blocked,
       reciprocity_error, segment_blocked, set_temperature!,
       set_temperatures!, sky_view_factor_raycast,
       sky_view_factors_raycast, solar_unit_vector, solve_longwave,
       solve_radiosity_direct, solve_radiosity_iterative,
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
       dewpoint_temperature, frost_point_temperature,
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
       AbstractStabilityFunctions, BusingerDyerBeljaarsHoltslag,
       phi_m, phi_h, psi_m, psi_h,
       SurfaceLayerSolution, solve_surface_layer,
       bulk_richardson, obukhov_length,
       AbstractSurface, SurfaceForcing, SurfaceFluxes,
       roughness_lengths, moisture_availability,
       surface_specific_humidity, surface_latent_heat,
       net_radiation, turbulent_fluxes, step_surface!

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

end # module
