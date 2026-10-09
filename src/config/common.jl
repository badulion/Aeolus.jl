# The common sections every run has (CFG-a, #356), and the two consumers
# that existed before the schema: checkpoint_schedule and apply_fftw_config!.
# Their defaults live here, in the schema, and nowhere else.

_env_bool(name, default::Bool) = (s = lowercase(strip(get(ENV, name, ""))); isempty(s) ? default : s in ("1", "true", "yes"))

function _env_log_level(default::String)
    s = lowercase(strip(get(ENV, "AEOLUS_LOG", "")))
    s == "warning" && return "warn"
    s == "none" && return "off"
    return isempty(s) ? default : s
end

const _LOG_LEVELS = ("debug", "info", "warn", "error", "off")

"""
The common sections, `[run]` with its `model` pinned to `driver` (the
run type; `driver` is accepted as its deprecated alias, EX-e #442):

  * `[run]`: `model`, `name`, `seed`, `root = "runs"`, `allow_uncommitted`
    (default: `AEOLUS_ALLOW_UNCOMMITTED == "1"`, else `false`);
  * `[fftw]`: `planner` (default `AEOLUS_FFTW_PLANNER`, else `measure`),
    `threads` (default `AEOLUS_FFTW_THREADS`, else `"none"`: automatic,
    one per Julia thread), `wisdom` (a file to import before planning);
  * `[logging]`: `console_level` (default `AEOLUS_LOG`, else `info`),
    `file_level`, `flush_interval`, `diagnostics_cadence`;
  * `[checkpoint]`: `every_steps`, `every_seconds`, `keep`, `prefix`;
  * `[output]`: `path` (`"none"`: the run folder's `output/`), `cadence`.
"""
function _common_sections(driver::String)
    run = section("run"; doc = "What runs and where its folder goes.") do s
        option!(s, "model", String; default = driver, allowed = [driver],
                doc = "the run type (model) that runs this config and whose schema validates it; " *
                      "the deprecated alias `driver` is still read")
        option!(s, "name", String; default = driver,
                doc = "the case name: part of the run folder name runs/<date>_<name>_<id>")
        option!(s, "seed", Int; default = 0, range = (0, typemax(Int)),
                doc = "the seed of the run's random numbers")
        option!(s, "root", String; default = "runs",
                doc = "the directory the run folders go in (relative to the working directory)")
        option!(s, "allow_uncommitted", Bool;
                default = () -> _env_bool("AEOLUS_ALLOW_UNCOMMITTED", false),
                doc = "start on a dirty git tree, saving the diff (default: AEOLUS_ALLOW_UNCOMMITTED=1)")
    end
    fftw = section("fftw"; doc = "The FFTW planning policy (set_fftw_policy!).") do s
        option!(s, "planner", Symbol; allowed = [_FFTW_PLANNERS...],
                default = () -> Symbol(lowercase(strip(get(ENV, "AEOLUS_FFTW_PLANNER",
                                                           String(FFTW_DEFAULT_PLANNER))))),
                doc = "estimate | measure | patient | exhaustive (default: AEOLUS_FFTW_PLANNER, else measure)")
        option!(s, "threads", Optional{Int}; range = (1, 4096),
                default = () -> (t = get(ENV, "AEOLUS_FFTW_THREADS", ""); isempty(t) ? nothing : something(tryparse(Int, t), t)),
                doc = "FFTW threads; \"none\": one per Julia thread (default: AEOLUS_FFTW_THREADS)")
        option!(s, "wisdom", Optional{Path}; input = false,
                doc = "FFTW wisdom file imported before planning, if it exists")
    end
    logging = section("logging"; doc = "The run log and the diagnostics.") do s
        option!(s, "console_level", Symbol; allowed = [_LOG_LEVELS...],
                default = () -> Symbol(_env_log_level("info")),
                doc = "lowest level shown on the console (default: AEOLUS_LOG, else info)")
        option!(s, "file_level", Symbol; allowed = [_LOG_LEVELS...], default = :debug,
                doc = "lowest level written to run.log")
        option!(s, "flush_interval", Float64; default = 5.0, range = (0, Inf), units = "s",
                doc = "run.log is flushed at most this often (warnings at once); 0: every record")
        option!(s, "diagnostics_cadence", Float64; default = 60.0, range = (0, Inf),
                open_lower = true, units = "s",
                doc = "simulated time between the costly diagnostics (RunMonitor cadence)")
    end
    checkpoint = section("checkpoint"; doc = "Periodic checkpoints (CheckpointSchedule).") do s
        option!(s, "every_steps", Int; default = 0, range = (0, typemax(Int)),
                doc = "steps between checkpoints; 0: off")
        option!(s, "every_seconds", Float64; default = Inf, range = (0, Inf), open_lower = true,
                units = "s", doc = "wall-clock seconds between checkpoints; inf: off")
        option!(s, "keep", Int; default = 2, range = (1, typemax(Int)),
                doc = "how many of the newest checkpoints are kept")
        option!(s, "prefix", String; default = "checkpoint",
                doc = "checkpoint file names: <prefix>_<step>.jls")
    end
    output = section("output"; doc = "Output files.") do s
        option!(s, "path", Optional{Path}; input = false,
                doc = "output directory; \"none\": the run folder's output/")
        option!(s, "cadence", Float64; default = 3600.0, range = (0, Inf), open_lower = true,
                units = "s", doc = "simulated time between output files")
    end
    return Section[run, fftw, logging, checkpoint, output]
end

# ---------------------------------------------------------------------------
# checkpoint_schedule / apply_fftw_config!
# ---------------------------------------------------------------------------

"The defaults of a common section, from the schema (the one place they live)."
function _section_defaults(name::String)
    s = _section(driver_schema("common"), name)
    return Dict{String,Any}(o.key => _default(o, "$name.$(o.key)")[2] for o in s.options)
end

"""
    checkpoint_schedule(cfg::RunConfig, dir = "checkpoints") -> CheckpointSchedule
    checkpoint_schedule(cfg::AbstractDict, dir = cfg["checkpoint"]["dir"]) -> CheckpointSchedule

A [`CheckpointSchedule`](@ref) from the `[checkpoint]` section
(`every_steps`, `every_seconds`, `keep`, `prefix`). Applies the `[fftw]`
section first ([`apply_fftw_config!`](@ref)), so the planner is set
before any plan. The `Dict` method (a namelist from
[`load_config`](@ref)) also reads `dir`; its missing keys take the
schema defaults.
"""
function checkpoint_schedule(cfg::RunConfig, dir = nothing)
    apply_fftw_config!(cfg)
    c = cfg.checkpoint
    return CheckpointSchedule(dir === nothing ? "checkpoints" : dir; every_steps = c.every_steps,
                              every_seconds = c.every_seconds, keep = c.keep, prefix = c.prefix)
end

function checkpoint_schedule(cfg::AbstractDict, dir = nothing)
    apply_fftw_config!(cfg)
    c = get(cfg, "checkpoint", Dict{String,Any}())
    d0 = _section_defaults("checkpoint")
    d = dir === nothing ? get(c, "dir", "checkpoints") : dir
    return CheckpointSchedule(d; every_steps = get(c, "every_steps", d0["every_steps"]),
                              every_seconds = get(c, "every_seconds", d0["every_seconds"]),
                              keep = get(c, "keep", d0["keep"]), prefix = get(c, "prefix", d0["prefix"]))
end

"""
    apply_fftw_config!(cfg::RunConfig) -> NamedTuple
    apply_fftw_config!(cfg::AbstractDict) -> NamedTuple

Apply the `[fftw]` section to the FFTW planning policy
([`set_fftw_policy!`](@ref Aeolus.Operators.set_fftw_policy!)), before
the simulation is built:

```toml
[fftw]
planner = "measure"   # estimate | measure (default) | patient | exhaustive
threads = 4           # "none": one per Julia thread
wisdom  = "run/fftw_wisdom"   # optional: import before planning
```

From a `RunConfig` every option is resolved, so the planner is always
set (`threads = "none"` leaves the thread count automatic). From a `Dict`
(a namelist from [`load_config`](@ref)) only the keys present change
anything, and without the table nothing changes. Returns the resulting
[`fftw_policy`](@ref Aeolus.Operators.fftw_policy).
"""
apply_fftw_config!(cfg::RunConfig) =
    _apply_fftw(cfg.fftw.planner, cfg.fftw.threads, cfg.fftw.wisdom)

function apply_fftw_config!(cfg::AbstractDict)
    f = get(cfg, "fftw", nothing)
    f === nothing && return fftw_policy()
    return _apply_fftw(get(f, "planner", nothing), get(f, "threads", nothing), get(f, "wisdom", nothing))
end

function _apply_fftw(planner, threads, wisdom)
    set_fftw_policy!(; planner = planner, threads = threads)
    wisdom === nothing || load_fftw_wisdom!(wisdom)
    return fftw_policy()
end
