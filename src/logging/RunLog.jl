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
                 source = false, flush = true)

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
* `flush`: flush the stream after each record, so a crash loses nothing.
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
    owns_stream::Bool
    lock::ReentrantLock
    message_limits::Dict{Any,Int}
end

function AeolusLogger(stream::Union{Nothing,IO} = nothing; min_level = Info,
                      timestamps::Symbol = :datetime, source::Bool = false, flush::Bool = true,
                      owns_stream::Bool = false)
    timestamps in (:datetime, :time, :none) ||
        throw(ArgumentError("timestamps must be :datetime, :time or :none, got :$timestamps"))
    return AeolusLogger(stream, parse_log_level(min_level), timestamps, source, flush,
                        owns_stream, ReentrantLock(), Dict{Any,Int}())
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
    text = format_record(level, message, _module, file, line, kwargs;
                         timestamps = l.timestamps, source = l.source)
    io = l.stream === nothing ? stderr : l.stream
    lock(l.lock) do
        write(io, text)
        l.flush && flush(io)
    end
    return nothing
end

"Close the logger's stream if the logger owns it (`owns_stream`); else just flush it."
function Base.close(l::AeolusLogger)
    s = l.stream
    s === nothing && return nothing
    if !l.owns_stream
        lock(() -> isopen(s) && flush(s), l.lock)
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
                 file_level = Debug, append = true) -> TeeLogger

The logger of a run (epic #293): records at `console_level` and up go to
`console` with short `HH:MM:SS.sss` timestamps, records at `file_level`
and up go to the file `path` with full timestamps and source locations.
`console = nothing` writes to whatever `stderr` is at the time;
`console = false` turns the console off. If `AEOLUS_LOG` is set it
overrides the default `console_level`. `append = true` continues an
existing file (a restart reuses its run log). The caller closes it with
`close`, or uses [`with_run_log`](@ref).
"""
function open_run_log(path::AbstractString; console::Union{Bool,Nothing,IO} = stderr,
                      console_level = _env_level(Info), file_level = Debug,
                      append::Bool = true)
    mkpath(dirname(abspath(path)))
    io = open(path, append ? "a" : "w")
    file = AeolusLogger(io; min_level = file_level, timestamps = :datetime, source = true,
                        owns_stream = true)
    console === false && return TeeLogger(file)
    con = AeolusLogger(console === true ? stderr : console; min_level = console_level,
                       timestamps = :time, source = false)
    return TeeLogger(con, file)
end

"""
    with_run_log(f, path; kwargs...)

Run `f()` with [`open_run_log`](@ref)`(path; kwargs...)` as the current
logger and close the log afterwards, also when `f` throws (the error is
logged at `Error` level first, so `run.log` records why the run stopped).
Returns `f()`.
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
    AeolusLogger(stream; min_level = level, timestamps = :time, source = false)

end # module
