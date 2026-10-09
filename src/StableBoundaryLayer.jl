"""
    Aeolus.StableBoundaryLayer

The `sbl` run type (`[run] driver = "sbl"`): the stable-boundary-layer
cases (GABLS1-4, CASES-99, the stratified, relaminarising and collapsing
channels) configured from TOML only. See [`SBLDriver`](@ref) and the
Stable boundary layer theory page; the shipped configs are in
`examples/sbl/configs/`, the case tables in `examples/sbl/data/`.

    julia --project -m Aeolus run examples/sbl/configs/gabls1.toml

A run writes `output/statistics.toml` (the averaged profiles and scalars),
`output/series.csv` (`[averaging] series_interval`) and
`output/towers.csv` (`[towers]`) into its run folder; the scripts of
`examples/sbl/` only read those.
"""
module StableBoundaryLayer

# The run type of the SBL cases (CFG-d #359), moved from examples/sbl/ into
# src by EX-b (#439): examples are configuration only. Its own module
# because it needs the dynamical cores, the surface layer, the wall models
# and the large-scale forcing, all included after Aeolus.Config; included
# at the end of Aeolus.jl. Not re-exported by Aeolus: `using
# Aeolus.StableBoundaryLayer` for the builders.
#
#   sbl/case.jl     the case builders and diagnostics (SBLCase, sbl_step!,
#                   sbl_run!, towers, the channels: stratified,
#                   relaminarisation, collapse)
#   sbl/tables.jl   the case tables (`[case] data`) and the GABLS2 formula
#   sbl/driver.jl   the schema, the config -> builder arguments, run!,
#                   the run's output files

using ..Aeolus
using ..Fields: set!
using ..Operators: add_sponge!, total_stress_profile, interior_view, sgs_fields
using ..Config
using ..Config: Config, AbstractRunDriver, RunConfig, ConfigError, Section, OpenTable, OneOf,
                Optional, Path, section, option!, register_driver!, config_schema
using ..IO: IO, RunFolder, CheckpointSchedule, with_monitor, monitor!, output_dir
using ..Simulations: Simulations
using ..InputDrivers: read_dynamic_driver, dynamic_forcing, forcing_dynamic_inputs, write_dynamic_driver
using Random
using Statistics: mean
import TOML

export SBLDriver, SBLRun, SBL_CONFIG_DIR, SBL_KINDS, sbl_build, sbl_config,
       # the case builders and their step (case.jl)
       SBLCase, SBL_GABLS1_STABILITY, sbl_case, sbl_surface!, sbl_dt, sbl_step!, sbl_run!,
       sbl_column_heat, sbl_profiles, sbl_bl_height, sbl_read_csv, sbl_csv_value,
       SBLMomentumBudget, sbl_budget_start!, sbl_budget, sbl_screen,
       SBLTowers, sbl_towers, sbl_towers_sample!, sbl_towers_bins, sbl_hockey_threshold,
       stratified_channel_case, stratified_channel_spinup, stratified_channel_run,
       SBL_RELAM_θ₀, relam_heat_flux, relam_case, relam_perturb!, relam_spinup, relam_run,
       collapse_heat_flux_scales, collapse_case, collapse_run,
       # the case tables (tables.jl)
       SBL_OMEGA, sbl_coriolis, sbl_case_tables, gabls2_surface_temperature_C, sbl_inversion_height,
       # config -> objects (driver.jl)
       sbl_quantity, sbl_profile, sbl_perturbation, sbl_surface_temperature, sbl_stability,
       sbl_base_closure, sbl_extra_forcing, sbl_forcing, sbl_channel_case, sbl_channel_spinup,
       sbl_channel_run

include("sbl/case.jl")
include("sbl/tables.jl")
include("sbl/driver.jl")

end # module
