# The run-driver contract (CFG-b, #357): what a driver implements so that
# `run_config` (the Julia API behind `aeolus run`) can build and run it
# from a RunConfig, and the registry of runnable drivers, including the
# drivers that live under `examples/` and are loaded on first use.

"""
    AbstractRunDriver

A driver: a kind of run that can be built from a [`RunConfig`](@ref) and
run into a run folder. A driver type `D` implements

  * `driver_name(::D) -> String`: the `[run] driver` value that selects it;
  * `driver_sections(::D) -> Vector{Section}`: its sections, on top of the
    common ones (a section named like a common one adds options to it);
  * `driver_checks(::D)`: cross-field checks ([`check!`](@ref); default none);
  * `config_dir(::D)`: where `extends = "base.toml"` looks for bare names
    (default `nothing`);
  * `driver_defaults(::D)`: the driver's own defaults of common options,
    `"section.key" => value` pairs (default none);
  * `build(::D, cfg::RunConfig)`: the runnable object (a simulation and
    what goes with it);
  * `run!(::D, obj, rf::RunFolder, ck::CheckpointSchedule; callback = nothing)`:
    run `obj`, writing into the run folder `rf`, checkpointing with `ck`
    (and, on a resumed run, continuing from its newest checkpoint with
    `resume!`); `callback(obj, step)`, when given, is called after every
    step;
  * `state_fields(obj) -> NamedTuple`: the prognostic arrays, the ones the
    round-trip test (`test/config_roundtrip_harness.jl`) compares bit for bit.

and is registered with [`register_driver!`](@ref)`(D())`.
"""
abstract type AbstractRunDriver end

"""
    driver_name(d::AbstractRunDriver) -> String

The `[run] driver` value that selects the driver `d`.
"""
function driver_name end

"""
    driver_sections(d::AbstractRunDriver) -> Vector{Section}

The configuration sections of the driver `d`, on top of the common ones.
"""
driver_sections(::AbstractRunDriver) = Section[]

"""
    driver_checks(d::AbstractRunDriver) -> Vector

The cross-field checks of the driver `d` (functions `cfg -> Vector{ConfigError}`,
see [`check!`](@ref)).
"""
driver_checks(::AbstractRunDriver) = Any[]

"""
    driver_defaults(d::AbstractRunDriver)

The driver's own defaults of common options, `"section.key" => value`
pairs (see [`ConfigSchema`](@ref)); default none.
"""
driver_defaults(::AbstractRunDriver) = ()

"""
    config_dir(d::AbstractRunDriver) -> Union{Nothing,String}

The directory of the driver's shipped configs: where `extends` looks for a
bare file name that is not next to the extending file.
"""
config_dir(::AbstractRunDriver) = nothing

"""
    build(d::AbstractRunDriver, cfg::RunConfig) -> obj

Build the runnable object of the driver `d` from the resolved config.
"""
function build end

"""
    state_fields(obj) -> NamedTuple

The prognostic arrays of a built run: what the bitwise round-trip test
compares.
"""
function state_fields end

# Simulations.run!(::AbstractRunDriver, obj, rf, ck; callback) is the
# contract's run method (it adds a method to the generic run!).

const _RUN_DRIVERS = Dict{String,AbstractRunDriver}()

"""
    register_driver!(d::AbstractRunDriver) -> d

Register the runnable driver `d`: its schema (the common sections plus
`driver_sections(d)`, with `driver_checks(d)`, `config_dir(d)` and
`driver_defaults(d)`) under
`driver_name(d)`, and `d` itself for [`run_config`](@ref). Registering
a name again replaces it.
"""
function register_driver!(d::AbstractRunDriver)
    name = String(driver_name(d))
    register_driver!(name; sections = driver_sections(d), checks = driver_checks(d),
                     config_dir = config_dir(d), defaults = driver_defaults(d))
    _RUN_DRIVERS[name] = d
    return d
end

"""
    DRIVER_FILES

Drivers that live under `examples/`, loaded the first time a config names
them: `name => path` relative to the package directory. The file is
included into a fresh module and must call [`register_driver!`](@ref)
for `name`. Physics stays in `examples/`; nothing is moved into `src/`
for the CLI's sake. The migrations of the example drivers (#311) add
their entries here.
"""
const DRIVER_FILES = Dict{String,String}()

const _PKG_DIR = normpath(joinpath(@__DIR__, "..", ".."))

"Load the example driver `name` from [`DRIVER_FILES`](@ref) if it is listed and not yet registered."
function _ensure_driver(name::AbstractString)
    (haskey(_DRIVERS, name) || !haskey(DRIVER_FILES, name)) && return nothing
    path = joinpath(_PKG_DIR, DRIVER_FILES[name])
    isfile(path) || throw(ConfigError("run.driver", "the driver \"$name\" is listed in " *
                                      "DRIVER_FILES but $path does not exist"))
    m = Module(Symbol("AeolusDriver_", name))
    Base.include(m, path)
    haskey(_DRIVERS, name) ||
        throw(ConfigError("run.driver", "$path did not register the driver \"$name\""))
    return nothing
end

"""
    run_driver(name) -> AbstractRunDriver

The runnable driver registered as `name`, loading it from
[`DRIVER_FILES`](@ref) on first use. A [`ConfigError`](@ref) for a name
that has a schema but no driver (`"common"`), or none at all.
"""
function run_driver(name::AbstractString)
    _ensure_driver(name)
    haskey(_RUN_DRIVERS, name) && return _RUN_DRIVERS[name]
    runnable = sort!(vcat(collect(keys(_RUN_DRIVERS)), collect(keys(DRIVER_FILES))))
    throw(ConfigError("run.driver", (haskey(_DRIVERS, name) ?
        "the driver \"$name\" has a schema but cannot be run" :
        "unknown driver \"$name\"" * _suggest(name, runnable)) *
        "; runnable drivers: " * join(repr.(unique!(runnable)), ", ")))
end

"""
    RunResult

What [`run_config`](@ref) returns: `folder`, the [`RunFolder`](@ref
Aeolus.IO.RunFolder) (finished), and `obj`, the object the driver built
and ran (for [`state_fields`](@ref)).
"""
struct RunResult
    folder::RunFolder
    obj::Any
end

"""
    run_config(cfg::RunConfig; root = nothing, resume = nothing,
               allow_uncommitted = nothing, console = stderr, repo = …,
               callback = nothing) -> RunResult

Run the configuration `cfg`: the Julia API behind `aeolus run`. The chain:

1. [`apply_fftw_config!`](@ref)`(cfg)`, so the planner is set before any
   plan (and the session's provenance records it);
2. [`start_run`](@ref Aeolus.IO.start_run)`(cfg; …)`: a new run folder
   (or, with `resume = folder`, that run's next session), `run.toml`
   with the canonical config;
3. [`run_checkpoints`](@ref Aeolus.IO.run_checkpoints) from `[checkpoint]` (before the build: it
   imports the folder's FFTW wisdom, so a resumed run plans the same
   transforms);
4. `build(driver, cfg)`, then `run!(driver, obj, rf, ck; callback)`;
5. [`finish_run!`](@ref Aeolus.IO.finish_run!): `completed`, or `failed` / `interrupted` with
   the error, which is then rethrown.

`root` and `allow_uncommitted` override `[run]` without changing the
config (so the recorded config stays the one given). The driver is
`cfg.run.driver` ([`run_driver`](@ref)).
"""
function run_config(cfg::RunConfig; root = nothing, resume = nothing,
                    allow_uncommitted::Union{Nothing,Bool} = nothing,
                    console = stderr, repo::AbstractString = IO.aeolus_source_dir(),
                    callback = nothing)
    d = run_driver(cfg.run.driver)
    apply_fftw_config!(cfg)
    obj = nothing
    rf = IO.start_run(cfg; root = root, resume = resume, allow_uncommitted = allow_uncommitted,
                      console = console, repo = repo)
    try
        Logging.with_logger(rf.logger) do
            c = cfg.checkpoint
            ck = IO.run_checkpoints(rf; every_steps = c.every_steps, every_seconds = c.every_seconds,
                                    keep = c.keep, prefix = c.prefix)
            obj = Base.invokelatest(build, d, cfg)
            Base.invokelatest(run!, d, obj, rf, ck; callback = callback)
        end
    catch err
        interrupted = err isa InterruptException
        bt = catch_backtrace()
        Logging.with_logger(rf.logger) do
            interrupted || @error("run stopped by an error", subsystem = :run,
                                  domain = current_domain(), exception = (err, bt))
        end
        IO.finish_run!(rf; status = interrupted ? :interrupted : :failed, error = err)
        rethrow()
    end
    IO.finish_run!(rf)
    return RunResult(rf, obj)
end
