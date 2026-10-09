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
  * `driver_checks(::D)`: cross-field checks, `"what it checks" => f` pairs
    ([`check!`](@ref); default none);
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

The cross-field checks of the driver `d`: `"what it checks, in words" => f`
pairs with `f(cfg) -> Vector{ConfigError}` (see [`check!`](@ref)). The
text goes into the option reference, one bullet per line.
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

"""
    preprocess_static(d::AbstractRunDriver, cfg::RunConfig; out = nothing, parent_out = nothing)
        -> Vector{String}

Write the static input driver(s) of the run `cfg` (CFG-f, #361:
`Aeolus.InputDrivers`) from the sources its config names, and return the
paths written. `out` replaces the path the config gives the run's own
driver, `parent_out` that of a nest parent's. `aeolus preprocess static
<config.toml> [--out file.nc] [--parent-out file.nc]` calls it. The
default: the driver has no static preprocessor (an `ArgumentError`).
"""
preprocess_static(d::AbstractRunDriver, cfg::RunConfig; out = nothing, parent_out = nothing) =
    throw(ArgumentError("the driver \"$(driver_name(d))\" has no static preprocessor"))

"""
    preprocess_dynamic(d::AbstractRunDriver, cfg::RunConfig; from, out = nothing, child = nothing)
        -> Vector{String}

Write the dynamic input driver of the run `cfg` (CFG-g, #362:
`Aeolus.InputDrivers`) and return the paths written: `from = :sounding`
from the sounding or case tables the config names, `from = :parent` by
running the config's nest parent with the child's boundary slabs recorded
at every parent step (`child`, a `RunConfig`, when the child is not `cfg`
itself). `out` replaces the path the config gives the driver. `aeolus
preprocess dynamic (--from-sounding | --from-parent) <config.toml>` calls
it. The default: the driver has no dynamic preprocessor (an
`ArgumentError`).

`from = :era5` (CFG-h, #363; `aeolus preprocess dynamic --from-era5
<files…> --config <config.toml> [--box deg]`) passes two more keywords,
`files` (the ERA5 files) and `box` (degrees, or `nothing`: the config's);
a driver with an ERA5 preprocessor accepts them (the `simulation` driver).
"""
preprocess_dynamic(d::AbstractRunDriver, cfg::RunConfig; from::Symbol, out = nothing, child = nothing) =
    throw(ArgumentError("the driver \"$(driver_name(d))\" has no dynamic preprocessor"))

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
their entries here. (The urban run type, formerly the `berlin` entry, is
part of the package since EX-a #438: `Aeolus.UrbanCase`.)
"""
const DRIVER_FILES = Dict{String,String}(
    # CFG-d (#359): the filtered-DNS record (the SBL cases are the `sbl` run
    # type in src since EX-b #439: Aeolus.StableBoundaryLayer)
    "dns" => "examples/validation/dns_apriori_driver.jl",
    # CFG-e (#360): the validation drivers and the remaining example scripts
    "channel" => "examples/validation/channel_driver.jl",
    "cbc_decay" => "examples/validation/cbc_driver.jl",
    "patchy_surface" => "examples/validation/patchy_surface_driver.jl",
    "column" => "examples/validation/column_driver.jl",
    "dns_decay" => "examples/validation/dns_driver.jl",
)

const _PKG_DIR = normpath(joinpath(@__DIR__, "..", ".."))

"Load the example driver `name` from [`DRIVER_FILES`](@ref) if it is listed and not yet registered."
function _ensure_driver(name::AbstractString)
    (haskey(_DRIVERS, name) || !haskey(DRIVER_FILES, name)) && return nothing
    path = joinpath(_PKG_DIR, DRIVER_FILES[name])
    isfile(path) || throw(ConfigError("run.driver", "the driver \"$name\" is listed in " *
                                      "DRIVER_FILES but $path does not exist"))
    sym = Symbol("AeolusDriver_", name)
    m = Module(sym)
    # bound in Main, where its name says it is: Serialization finds a type
    # by its module path, and a checkpoint holds the driver's own types
    # (the Berlin case's structural state, #269)
    isdefined(Main, sym) || Core.eval(Main, :(const $sym = $m))
    # what `module` defines for its body: include (relative to the file) and eval
    Core.eval(m, :(include(p::AbstractString) = Base.include($m, p)))
    Core.eval(m, :(eval(x) = Core.eval($m, x)))
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

# ---------------------------------------------------------------------------
# helpers for drivers (CFG-e, #360)
# ---------------------------------------------------------------------------

"""
    write_results(rf::RunFolder, data::AbstractDict; name = "results.toml") -> String

Write a driver's results (numbers, strings, vectors of numbers, nested
tables) as TOML into the run folder's `output/` (or `[output] path`) and
return the file's path. The example drivers record what their scripts
analyse this way, so a finished run folder holds its numbers.
"""
function write_results(rf::RunFolder, data::AbstractDict; name::AbstractString = "results.toml",
                       dir = nothing)
    d = dir === nothing ? IO.output_dir(rf) : mkpath(dir)
    path = joinpath(d, name)
    open(path, "w") do io
        _emit_table(io, Dict{String,Any}(String(string(k)) => _toml_out(v) for (k, v) in data), String[])
    end
    return path
end

"""
    no_checkpoints(cfg) -> Vector{ConfigError}

A cross-field check for drivers whose runs cannot be checkpointed (a
harness that runs to the end in one call): `[checkpoint]` must be off, so
that a requested checkpoint is an error rather than silently ignored.
"""
function no_checkpoints(cfg)
    c = cfg.checkpoint
    (c.every_steps == 0 && c.every_seconds == Inf) && return ConfigError[]
    return [ConfigError("checkpoint.every_steps",
                        "the \"$(cfg.run.driver)\" driver runs to the end in one call and writes no checkpoints; " *
                        "leave [checkpoint] off")]
end

const _GREEK = Dict('α' => "alpha", 'β' => "beta", 'γ' => "gamma", 'δ' => "delta", 'ε' => "epsilon",
                    'κ' => "kappa", 'λ' => "lambda", 'μ' => "mu", 'ν' => "nu", 'ρ' => "rho",
                    'σ' => "sigma", 'τ' => "tau", 'θ' => "theta", 'ω' => "omega", 'Γ' => "gamma")

"The ASCII snake_case option key of a Julia field name (`σ_S` → `sigma_s`, `T_deep` → `t_deep`)."
function field_key(f::Symbol)
    s = join(get(_GREEK, ch, string(ch)) for ch in String(f))
    return lowercase(s)
end

"""
    kwdef_section(name, T; doc = "", exclude = (), docs = Dict(), units = Dict()) -> Section

A section whose options are the fields of the keyword struct `T`
(`Base.@kwdef`), with the defaults of `T()`: one option per field of type
`Float64`, `Int`, `Bool`, `Symbol` or an `NTuple` of those, keyed by
[`field_key`](@ref) (`σ_S` is `sigma_s`). `exclude` names fields left out;
`docs` and `units` map field names to the option's doc and units. Rebuild
the struct with [`kwdef_value`](@ref). One source of truth: the struct's
defaults are the schema's defaults.
"""
function kwdef_section(name::AbstractString, ::Type{T}; doc::AbstractString = "", exclude = (),
                       docs = Dict{Symbol,String}(), units = Dict{Symbol,String}()) where {T}
    d = T()
    return section(name; doc = doc) do s
        for f in fieldnames(T)
            f in exclude && continue
            v = getfield(d, f); FT = fieldtype(T, f)
            FT in (Float64, Int, Bool, Symbol) || (FT <: Tuple && fieldcount(FT) > 0) ||
                throw(ArgumentError("kwdef_section: field $f of $T has type $FT"))
            option!(s, field_key(f), FT; default = v,
                    doc = get(docs, f, "$(nameof(T)).$f"), units = get(units, f, ""))
        end
    end
end

"""
    kwdef_value(T, sec::ConfigSection; exclude = (), kw...) -> T

The keyword struct `T` from the section made by [`kwdef_section`](@ref):
`T(; field = sec[field_key(field)], …, kw...)`.
"""
function kwdef_value(::Type{T}, sec::ConfigSection; exclude = (), kw...) where {T}
    vals = Dict{Symbol,Any}(f => sec[field_key(f)] for f in fieldnames(T) if !(f in exclude))
    return T(; vals..., kw...)
end

"""
    selected_case_sections(cfg, case_key, sections) -> Vector{ConfigError}

A cross-field check for drivers with a `kind`/`case` switch and one
section per case (`sections`: case => section name pairs; a case may
appear more than once): any key of the section of a case that is not
selected whose value differs from its default is an error, so a key that
would be ignored does not pass silently. (Values, not origins, are
compared: a `run.toml` records every key, defaults included.)
"""
function selected_case_sections(cfg::RunConfig, case_key::AbstractString, sections)
    e = ConfigError[]
    chosen = cfg[case_key]
    schema = getfield(cfg, :schema)
    for (case, sec) in sections
        string(case) == string(chosen) && continue
        spec = only(s for s in schema.sections if s.name == sec)
        for o in spec.options
            isequal(getproperty(cfg, Symbol(sec))[o.key], o.default) && continue
            push!(e, ConfigError("$sec.$(o.key)",
                                 "[$sec] belongs to $case_key = \"$case\", not \"$chosen\": it would be ignored"))
        end
    end
    return e
end
