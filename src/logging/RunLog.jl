"""
    Aeolus.RunLog

Structured logging for Aeolus (LOG-a, #333; design in epic #293).

The library emits events through stdlib `Logging` (`@debug`, `@info`,
`@warn`, `@error`) with key-value pairs and never prints. Every event of
the library carries two tags:

* `subsystem`: a `Symbol` naming the emitting part of the code
  (`:pressure`, `:timestepping`, `:diffusion`, `:checkpoint`, …);
* `domain`: the domain of a multi-domain run the event belongs to,
  [`current_domain`](@ref)`()` (`:main` outside [`with_domain`](@ref)).

What is shown, and where, is the caller's choice: they install a logger.
This module provides the loggers a run needs, with no dependency beyond
the standard library: [`AeolusLogger`](@ref) (one formatted, timestamped
line per record), [`TeeLogger`](@ref) (several sinks, each with its own
level), [`open_run_log`](@ref) / [`with_run_log`](@ref) (info and up to
the console, debug and up to `run.log`) and [`test_logger`](@ref) (the
test suite's: warnings and up, or the level of `AEOLUS_LOG`).
"""
module RunLog

using Logging
using Logging: AbstractLogger, LogLevel, Debug, Info, Warn, Error, BelowMinLevel, AboveMaxLevel
import Logging: min_enabled_level, shouldlog, handle_message, catch_exceptions

export AeolusLogger, TeeLogger, open_run_log, with_run_log, test_logger,
       current_domain, with_domain, parse_log_level
# LOG-c (#335): the diagnostics registry, the CSV writer, the progress line
export Diagnostic, diagnostics, collect_diagnostics, prefixed, reduce_diagnostic, evaluate_diagnostic,
       ProfileDiagnostic, profile_diagnostics, collect_profiles, evaluate_profile,
       DiagnosticsWriter, record_diagnostics!, last_diagnostics, costly_due,
       ProgressLine, progress!, WarningTap, with_progress, note_warning!

# ---------------------------------------------------------------------------
# domain tag
# ---------------------------------------------------------------------------

const _DOMAIN_KEY = :aeolus_log_domain

"""
    current_domain() -> Symbol

The domain tag of the events emitted by the current task: the innermost
[`with_domain`](@ref), or `:main`. Library events pass it as
`domain = current_domain()`, so in a nested run every record names the
domain it came from (epic #293: "every record carries a domain tag").
"""
current_domain() = get(task_local_storage(), _DOMAIN_KEY, :main)::Symbol

"""
    with_domain(f, domain::Symbol)

Run `f()` with [`current_domain`](@ref) returning `domain`, then restore
the previous tag. A multi-domain driver wraps each domain's step in it.
The tag is task-local: tasks spawned inside `f` start at `:main` again
(only the main task logs; epic #293).
"""
with_domain(f, domain::Symbol) = task_local_storage(f, _DOMAIN_KEY, domain)

# ---------------------------------------------------------------------------
# levels
# ---------------------------------------------------------------------------

"""
    parse_log_level(s) -> LogLevel

A log level from its name, case-insensitive: `"debug"`, `"info"`,
`"warn"`/`"warning"`, `"error"`, `"off"`/`"none"` (nothing), `"all"`
(everything); an integer string is taken as the numeric level. Errors on
anything else, so a misspelt `AEOLUS_LOG` is not silently ignored.
"""
function parse_log_level(s::AbstractString)
    t = lowercase(strip(s))
    t == "debug" && return Debug
    t == "info" && return Info
    (t == "warn" || t == "warning") && return Warn
    t == "error" && return Error
    (t == "off" || t == "none") && return AboveMaxLevel
    t == "all" && return BelowMinLevel
    n = tryparse(Int, t)
    n === nothing || return LogLevel(n)
    throw(ArgumentError("unknown log level \"$s\"; use debug, info, warn, error or off"))
end
parse_log_level(l::LogLevel) = l
parse_log_level(s::Symbol) = parse_log_level(String(s))

"`ENV[\"AEOLUS_LOG\"]` as a level, or `default` when unset or empty."
function _env_level(default::LogLevel)
    s = get(ENV, "AEOLUS_LOG", "")
    return isempty(strip(s)) ? default : parse_log_level(s)
end

_level_name(l::LogLevel) =
    l == Debug ? "DEBUG" : l == Info ? "INFO " : l == Warn ? "WARN " : l == Error ? "ERROR" :
    l < Info ? "DEBUG" : l < Warn ? "INFO " : l < Error ? "WARN " : "ERROR"

# ---------------------------------------------------------------------------
# AeolusLogger: one formatted line per record
# ---------------------------------------------------------------------------

"""
    AeolusLogger(stream = nothing; min_level = Info, timestamps = :datetime,
                 source = false, flush = true, flush_interval = 5.0, owns_stream = false)

A logger that writes one line per record:

    2026-10-06T14:03:12.481 WARN  pressure[main] MultigridPoissonSolver did not converge  cycles=50 residual=2.1e-5 tol=1.0e-8  @ Multigrid.jl:685

* `stream`: an `IO`, or `nothing` for whatever `stderr` is when the
  record is written (so `redirect_stderr` captures it).
* `min_level`: records below it are dropped before their message or
  key-values are evaluated.
* `timestamps`: `:datetime` (local `yyyy-mm-ddTHH:MM:SS.sss`), `:time`
  (`HH:MM:SS.sss`) or `:none`.
* `source`: append ` @ file:line` to every record; warnings and errors
  always carry it.
* `flush` and `flush_interval` (seconds): the flush policy (maintainer
  decision of 2026-10-06, PR #346). With `flush = true` a record at
  `Warn` or above is flushed at once; a `Debug` or `Info` record is
  flushed only when more than `flush_interval` seconds have passed since
  the last flush. Anything still buffered is flushed by `close(logger)`,
  which [`with_run_log`](@ref) and `with_run` call on every exit,
  including an error, and by an `atexit` hook if the process exits with
  the logger still open. So only a hard kill (`SIGKILL`, power loss) can
  lose the last few seconds of debug and info lines. There is no timer
  task: the clock is read when a record arrives, so a debug line written
  just before a long quiet spell stays buffered until the next record or
  the close. `flush_interval = 0` flushes every record (the LOG-a
  behaviour); `flush = false` never flushes before `close`.
* `owns_stream`: `close(logger)` closes `stream` (the run log's file);
  otherwise the stream belongs to the caller and is left open.

The tags `subsystem` and `domain` (see [`RunLog`](@ref Aeolus.RunLog))
are printed as `subsystem[domain]` in front of the message; the other
key-values follow it as `key=value`. An `exception = err` or
`exception = (err, backtrace)` value is printed with `showerror` on
indented continuation lines, as are multi-line values. `maxlog = n` is
honoured per record id, as by the stdlib loggers. Writes are serialised
by a lock, so the logger can be shared between tasks.
"""
mutable struct AeolusLogger <: AbstractLogger
    stream::Union{Nothing,IO}
    min_level::LogLevel
    timestamps::Symbol
    source::Bool
    flush::Bool
    flush_interval::Float64
    last_flush::Float64          # time() of the last flush (or of construction)
    owns_stream::Bool
    lock::ReentrantLock
    message_limits::Dict{Any,Int}
end

function AeolusLogger(stream::Union{Nothing,IO} = nothing; min_level = Info,
                      timestamps::Symbol = :datetime, source::Bool = false, flush::Bool = true,
                      flush_interval::Real = 5.0, owns_stream::Bool = false)
    timestamps in (:datetime, :time, :none) ||
        throw(ArgumentError("timestamps must be :datetime, :time or :none, got :$timestamps"))
    flush_interval >= 0 || throw(ArgumentError("flush_interval must be ≥ 0, got $flush_interval"))
    l = AeolusLogger(stream, parse_log_level(min_level), timestamps, source, flush,
                     Float64(flush_interval), time(), owns_stream, ReentrantLock(), Dict{Any,Int}())
    stream === nothing || _register_open!(l)
    return l
end

# Loggers with a stream of their own, flushed by an atexit hook if the
# process ends while they are open (an uncaught error outside with_run_log).
const _OPEN_LOGGERS = WeakKeyDict{AeolusLogger,Nothing}()
const _OPEN_LOCK = ReentrantLock()
_register_open!(l::AeolusLogger) = lock(() -> (_OPEN_LOGGERS[l] = nothing), _OPEN_LOCK)
_unregister_open!(l::AeolusLogger) = lock(() -> delete!(_OPEN_LOGGERS, l), _OPEN_LOCK)

"Flush every open logger's stream; registered with `atexit`."
function _flush_open_loggers()
    for l in lock(() -> collect(keys(_OPEN_LOGGERS)), _OPEN_LOCK)
        try
            flush(l)
        catch
        end
    end
    return nothing
end

function __init__()
    atexit(_flush_open_loggers)
end

"""
    flush(logger::AeolusLogger)

Flush the logger's stream now, whatever its flush policy.
"""
function Base.flush(l::AeolusLogger)
    s = l.stream === nothing ? stderr : l.stream
    lock(l.lock) do
        isopen(s) && flush(s)
        l.last_flush = time()
    end
    return nothing
end

min_enabled_level(l::AeolusLogger) = l.min_level
shouldlog(l::AeolusLogger, level, _module, group, id) = get(l.message_limits, id, 1) > 0
catch_exceptions(::AeolusLogger) = true

"The local wall-clock time of `t` (seconds since the epoch), as text."
function _timestamp(kind::Symbol, t::Float64 = time())
    kind === :none && return ""
    s = floor(t)
    ms = min(999, floor(Int, 1000 * (t - s)))
    fmt = kind === :datetime ? "%Y-%m-%dT%H:%M:%S" : "%H:%M:%S"
    return string(Libc.strftime(fmt, s), '.', lpad(ms, 3, '0'))
end

_showvalue(io::IO, v) = show(IOContext(io, :compact => true, :limit => true), v)
_showvalue(io::IO, v::AbstractString) = print(io, v)
_showvalue(io::IO, v::Symbol) = print(io, v)

"""
    format_record(level, message, _module, file, line, kwargs; timestamps, source) -> String

The text [`AeolusLogger`](@ref) writes for one record (ending in a
newline). Exposed so other sinks (a future diagnostics stream) format
records the same way.
"""
function format_record(level::LogLevel, message, _module, file, line, kwargs;
                       timestamps::Symbol = :datetime, source::Bool = false,
                       time::Float64 = Base.time())
    buf = IOBuffer()
    ts = _timestamp(timestamps, time)
    isempty(ts) || print(buf, ts, ' ')
    print(buf, _level_name(level), ' ')
    subsystem = nothing; domain = nothing
    for (k, v) in kwargs
        k === :subsystem && (subsystem = v)
        k === :domain && (domain = v)
    end
    if subsystem !== nothing || domain !== nothing
        print(buf, subsystem === nothing ? "" : subsystem, '[', domain === nothing ? :main : domain, "] ")
    end
    msg = message isa AbstractString ? message : sprint(_showvalue, message)
    lines = split(chomp(String(msg)), '\n')
    print(buf, lines[1])
    extra = String[]                       # indented continuation lines
    append!(extra, lines[2:end])
    sep = "  "
    for (k, v) in kwargs
        (k === :subsystem || k === :domain || k === :maxlog) && continue
        if k === :exception
            text = v isa Tuple && length(v) == 2 ?
                   sprint(Base.display_error, v[1], v[2]) :
                   sprint(showerror, v)
            push!(extra, "exception =")
            append!(extra, ("  " * l for l in split(chomp(text), '\n')))
            continue
        end
        s = sprint(_showvalue, v)
        if occursin('\n', s)
            push!(extra, string(k, " ="))
            append!(extra, ("  " * l for l in split(chomp(s), '\n')))
        else
            print(buf, sep, k, '=', s)
            sep = " "
        end
    end
    if (source || level >= Warn) && file !== nothing
        print(buf, "  @ ", basename(string(file)), ':', line === nothing ? "?" : line)
    end
    println(buf)
    for l in extra
        println(buf, "    │ ", l)
    end
    return String(take!(buf))
end

function handle_message(l::AeolusLogger, level::LogLevel, message, _module, group, id,
                        file, line; kwargs...)
    maxlog = get(kwargs, :maxlog, nothing)
    if maxlog isa Integer
        remaining = lock(l.lock) do
            r = get(l.message_limits, id, Int(maxlog))
            l.message_limits[id] = r - 1
            r
        end
        remaining > 0 || return nothing
    end
    t = time()
    text = format_record(level, message, _module, file, line, kwargs;
                         timestamps = l.timestamps, source = l.source, time = t)
    io = l.stream === nothing ? stderr : l.stream
    lock(l.lock) do
        write(io, text)
        # the flush policy: Warn and up at once, the rest every flush_interval s
        if l.flush && (level >= Warn || t - l.last_flush >= l.flush_interval)
            flush(io)
            l.last_flush = t
        end
    end
    return nothing
end

"Close the logger's stream if the logger owns it (`owns_stream`); else just flush it."
function Base.close(l::AeolusLogger)
    s = l.stream
    s === nothing && return nothing
    _unregister_open!(l)
    if !l.owns_stream
        flush(l)
        return nothing
    end
    lock(l.lock) do
        isopen(s) && (flush(s); close(s))
    end
    return nothing
end

# ---------------------------------------------------------------------------
# TeeLogger: several sinks, each with its own level
# ---------------------------------------------------------------------------

"""
    TeeLogger(loggers...)

Send every record to each of `loggers` whose level admits it. The tee's
own minimum level is the lowest of its children's, so a record nobody
wants is dropped before it is built. `close(tee)` closes every child.
"""
struct TeeLogger{L<:Tuple} <: AbstractLogger
    loggers::L
end
TeeLogger(loggers::AbstractLogger...) = TeeLogger(loggers)

min_enabled_level(t::TeeLogger) = minimum(min_enabled_level, t.loggers)
shouldlog(t::TeeLogger, level, _module, group, id) =
    any(l -> level >= min_enabled_level(l) && shouldlog(l, level, _module, group, id), t.loggers)
catch_exceptions(t::TeeLogger) = any(catch_exceptions, t.loggers)

function handle_message(t::TeeLogger, level::LogLevel, message, _module, group, id,
                        file, line; kwargs...)
    for l in t.loggers
        if level >= min_enabled_level(l) && shouldlog(l, level, _module, group, id)
            handle_message(l, level, message, _module, group, id, file, line; kwargs...)
        end
    end
    return nothing
end

"Flush every child of the tee that can be flushed."
Base.flush(t::TeeLogger) = (foreach(l -> applicable(flush, l) && flush(l), t.loggers); nothing)

function Base.close(t::TeeLogger)
    for l in t.loggers
        applicable(close, l) && close(l)
    end
    return nothing
end

# ---------------------------------------------------------------------------
# the run logger and the test logger
# ---------------------------------------------------------------------------

"""
    open_run_log(path; console = stderr, console_level = Info,
                 file_level = Debug, append = true, flush_interval = 5.0) -> TeeLogger

The logger of a run (epic #293): records at `console_level` and up go to
`console` with short `HH:MM:SS.sss` timestamps, records at `file_level`
and up go to the file `path` with full timestamps and source locations.
`console = nothing` writes to whatever `stderr` is at the time;
`console = false` turns the console off. If `AEOLUS_LOG` is set it
overrides the default `console_level`. `append = true` continues an
existing file (a restart reuses its run log). The caller closes it with
`close`, or uses [`with_run_log`](@ref).

The file follows the flush policy of [`AeolusLogger`](@ref): warnings
and errors are on disk at once, debug and info lines at the latest
`flush_interval` seconds later (at the next record), at `close`, or at
process exit. `flush_interval = 0` flushes every record. The console
flushes every record.
"""
function open_run_log(path::AbstractString; console::Union{Bool,Nothing,IO} = stderr,
                      console_level = _env_level(Info), file_level = Debug,
                      append::Bool = true, flush_interval::Real = 5.0)
    mkpath(dirname(abspath(path)))
    io = open(path, append ? "a" : "w")
    file = AeolusLogger(io; min_level = file_level, timestamps = :datetime, source = true,
                        flush_interval = flush_interval, owns_stream = true)
    console === false && return TeeLogger(file)
    con = AeolusLogger(console === true ? stderr : console; min_level = console_level,
                       timestamps = :time, source = false, flush_interval = 0.0)
    return TeeLogger(con, file)
end

"""
    with_run_log(f, path; kwargs...)

Run `f()` with [`open_run_log`](@ref)`(path; kwargs...)` as the current
logger and close the log afterwards, also when `f` throws (the error is
logged at `Error` level first, so `run.log` records why the run stopped).
Closing flushes every record still buffered by the flush policy, so on
any exit short of a hard kill the whole run is on disk. Returns `f()`.
"""
function with_run_log(f, path::AbstractString; kwargs...)
    logger = open_run_log(path; kwargs...)
    try
        return Logging.with_logger(logger) do
            try
                f()
            catch err
                err isa InterruptException ||
                    @error "run stopped by an error" subsystem = :runlog domain = current_domain() exception = (err, catch_backtrace())
                rethrow()
            end
        end
    finally
        close(logger)
    end
end

"""
    test_logger(; level = AEOLUS_LOG or Warn, stream = nothing) -> AeolusLogger

The test suite's logger (installed by `test/runtests.jl`, and so by every
worker of `test/run.jl`): warnings and errors only, with short timestamps,
to the current `stderr`. `AEOLUS_LOG=debug` (or `info`) opts in to more.
Tests that assert on events capture them with their own logger
(`Test.@test_logs`, `Test.TestLogger`), which this does not affect.
"""
test_logger(; level = _env_level(Warn), stream::Union{Nothing,IO} = nothing) =
    AeolusLogger(stream; min_level = level, timestamps = :time, source = false, flush_interval = 0.0)

# ---------------------------------------------------------------------------
# the diagnostics registry, CSV writer and progress line (LOG-c, #335)
# ---------------------------------------------------------------------------

include("diagnostics.jl")

end # module
