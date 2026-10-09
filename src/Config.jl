module Config

# Run configuration (CFG-a, #356): one in-package schema system, a loader
# with `extends` and overrides, strict validation that reports every
# problem at once, and the canonical resolved form `run.toml` stores.
#
#   config/schema.jl   option types, Option / Section / ConfigSchema,
#                      value checks, the driver registry, "did you mean"
#   config/load.jl     load_run_config: extends chain, merge, overrides,
#                      validation, reading a run folder's [config]
#   config/resolve.jl  RunConfig, with, to_toml / write_config /
#                      config_hash, [inputs] hashes, the run.toml hooks
#   config/common.jl   the common sections ([run], [fftw], [logging],
#                      [checkpoint], [output]); checkpoint_schedule and
#                      apply_fftw_config! on them
#   config/run_record.jl  the run.toml record of a RunConfig: the hooks
#                      IO.start_run calls, start_run(cfg)
#   config/drivers.jl  the run-driver contract (AbstractRunDriver, build,
#                      run!, state_fields), runnable drivers incl. the
#                      example drivers loaded on first use, run_config (CFG-b)
#   config/sweep.jl    [sweep] tables expanded into override sets, one run
#                      per member; run_configs for the example scripts (CFG-e)
#   ../cli/main.jl     the aeolus command line: main(args) (CFG-b)
#
# The bare-simulation driver ("simulation") needs the dynamical cores, so
# it lives in its own module, Aeolus.SimulationDriver
# (config/simulation_driver.jl), included at the end of Aeolus.jl.
#
# Included after IO: it extends IO.start_run and the run.toml hooks.
# Dependencies: the stdlibs TOML and Dates only.

using TOML
using Logging
using Dates: Dates, Date, Time, DateTime
using ..RunLog: parse_log_level, current_domain
using ..Simulations: run!
using ..Operators: set_fftw_policy!, fftw_policy, load_fftw_wisdom!, FFTW_DEFAULT_PLANNER,
                   _FFTW_PLANNERS
using ..IO: IO, CheckpointSchedule, RunFolder, fnv1a64

export RunConfig, ConfigSection, ConfigError, ConfigProblem, load_run_config, with,
       to_toml, write_config, config_text, config_hash, config_inputs, content_hash,
       config_origin, config_sources, config_schema, nonreplayable_keys,
       ConfigSchema, Section, Option, Path, Optional, OpenTable, OneOf, section, option!, check!,
       register_driver!, driver_schema, registered_drivers, shipped_configs,
       damerau_levenshtein, CONFIG_SCHEMA_VERSION,
       checkpoint_schedule, apply_fftw_config!,
       AbstractRunDriver, driver_name, driver_sections, driver_checks, driver_defaults, config_dir, build,
       state_fields, preprocess_static, preprocess_dynamic, run_driver, run_config, RunResult, DRIVER_FILES, package_drivers,
       register_preprocessor!, option_reference, reference_sections, undocumented_keys, reference_option,
       write_results, no_checkpoints,
       kwdef_section, kwdef_value, field_key, selected_case_sections,
       SweepMember, sweep_members, load_sweep, run_configs, config_test_table

include("config/schema.jl")
include("config/load.jl")
include("config/resolve.jl")
include("config/common.jl")
include("config/run_record.jl")
include("config/drivers.jl")
include("config/sweep.jl")
include("cli/main.jl")

# The common driver: the common sections only.
register_driver!("common")

end # module
