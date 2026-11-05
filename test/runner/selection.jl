# Job selection for the Aeolus.jl suites (#316, validation split).
#
# There are two suites, run by the same machinery:
#
#   * "test":       test/runtests.jl, tags in test/test_tags.toml. Tests of
#                   code: unit, analytic, conservation, interface, short
#                   code-path runs. The PR gate; what `Pkg.test()` runs.
#   * "validation": validation/runvalidation.jl, tags in
#                   validation/validation_tags.toml. Simulated cases that
#                   confirm the physics works and stays stable (GABLS,
#                   channels, decaying turbulence, campaigns). Not a gate.
#
# A suite's entry file is the single list of its files. This module reads
# that list from the file's syntax tree, joins it with the suite's tag table
# (weight, exclusivity, measured seconds) and evaluates a pruned copy of the
# entry file that keeps only the selected files, each inside the @testset
# blocks that enclose it there. It is used by the parallel runner
# (`runner.jl`, via test/run.jl and validation/run.jl) and its workers
# (`worker.jl`).
#
# Stdlib only. See docs/src/testing.md and docs/src/validation.md.

module AeolusTestSelection

using TOML

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))

"""
    Suite(name, dir, entry, tagfile)

One suite: its directory, its entry file (the include list) and its tag
table.
"""
struct Suite
    name::String
    dir::String
    entry::String
    tagfile::String
end

const SUITES = Dict(
    "test"       => Suite("test", joinpath(ROOT, "test"),
                          joinpath(ROOT, "test", "runtests.jl"),
                          joinpath(ROOT, "test", "test_tags.toml")),
    "validation" => Suite("validation", joinpath(ROOT, "validation"),
                          joinpath(ROOT, "validation", "runvalidation.jl"),
                          joinpath(ROOT, "validation", "validation_tags.toml")),
)

"Pseudo-job name for the inline `@testset`s written directly in a suite's entry file: its file name."
core(s::Suite) = basename(s.entry)

function suite(name::AbstractString)
    haskey(SUITES, name) || error("unknown suite \"$name\" (one of $(join(sort(collect(keys(SUITES))), ", ")))")
    return SUITES[name]
end

"""
    Tag(weight, exclusive, seconds, reason)

Tag-table entry for one file. Untagged files get `Tag()`: light, not
exclusive.

* `weight`: `"heavy"` files are started first by the parallel runner.
* `exclusive`: run alone after the parallel batch (timing-sensitive files).
* `seconds`: measured wall time, used to order the parallel batch.
* `reason`: free text, e.g. why a file is exclusive or what a validation
  case checks.
"""
struct Tag
    weight::String
    exclusive::Bool
    seconds::Float64
    reason::String
end
Tag() = Tag("light", false, 0.0, "")

"""
    read_tags(path) -> Dict{String,Tag}

Read a tag table. Errors on unknown keys and bad values.
"""
function read_tags(path::AbstractString)
    isfile(path) || return Dict{String,Tag}()
    raw = TOML.parsefile(path)
    tags = Dict{String,Tag}()
    for (file, d) in raw
        d isa AbstractDict || error("$path: entry \"$file\" must be a table")
        for k in keys(d)
            k == "tier" && error("$path: \"$file\" has a `tier`: the fast/full tiers were replaced by " *
                                 "the validation suite; move a validation run to validation/ " *
                                 "(docs/src/validation.md)")
            k in ("weight", "exclusive", "seconds", "reason") ||
                error("$path: unknown key \"$k\" in \"$file\"")
        end
        t = Tag(String(get(d, "weight", "light")), Bool(get(d, "exclusive", false)),
                Float64(get(d, "seconds", 0.0)), String(get(d, "reason", "")))
        t.weight in ("light", "heavy") || error("$path: \"$file\": weight must be light or heavy")
        tags[file] = t
    end
    return tags
end
read_tags(s::Suite) = read_tags(s.tagfile)

tag(tags::AbstractDict, file) = get(tags, file, Tag())

# ---------------------------------------------------------------------------
# Syntax-tree helpers
# ---------------------------------------------------------------------------

"The file name of a literal `include(\"file.jl\")` call, else `nothing`."
function include_target(ex)
    ex isa Expr && ex.head === :call && length(ex.args) == 2 &&
        ex.args[1] === :include && ex.args[2] isa String || return nothing
    return ex.args[2]::String
end

is_testset(ex) = ex isa Expr && ex.head === :macrocall && ex.args[1] === Symbol("@testset")

function has_include(ex)
    include_target(ex) === nothing || return true
    ex isa Expr || return false
    return any(has_include, ex.args)
end

"""
    listed_files(entry) -> Vector{String}

The files that an entry file includes with a literal `include("file.jl")`,
in order. (`include(joinpath(...))` lines are setup, not list entries.)
"""
function listed_files(path::AbstractString)
    files = String[]
    walk(ex) = begin
        f = include_target(ex)
        f === nothing || (push!(files, f); return)
        ex isa Expr && foreach(walk, ex.args)
    end
    foreach(walk, Meta.parseall(read(path, String); filename = path).args)
    return files
end
listed_files(s::Suite) = listed_files(s.entry)

"""
    prune(ex, keep, core) -> Expr or nothing

Prune one top-level expression of an entry file.

* `include("f")` survives iff `keep(f)`.
* An `@testset` that contains includes survives iff something in it does.
* An `@testset` without includes (an inline test) survives iff `core`.
* Every other statement (`using`, constants, setup includes) survives.
"""
function prune(ex, keep, core::Bool)
    f = include_target(ex)
    f === nothing || return keep(f) ? ex : nothing
    if is_testset(ex)
        has_include(ex) || return core ? ex : nothing
        body = prune(ex.args[end], keep, core)
        body === nothing && return nothing
        return Expr(ex.head, ex.args[1:end-1]..., body)
    end
    if ex isa Expr && ex.head === :block
        kept = Any[]
        live = false
        for a in ex.args
            a isa LineNumberNode && (push!(kept, a); continue)
            p = prune(a, keep, core)
            p === nothing && continue
            push!(kept, p)
            live |= has_include(p) || is_testset(p)
        end
        return live ? Expr(:block, kept...) : nothing
    end
    return ex
end

contains_testset(ex) = is_testset(ex) || (ex isa Expr && any(contains_testset, ex.args))

"""
    has_inline(entry) -> Bool

Does the entry file hold `@testset`s of its own (outside the listed files)?
The runner gives them a job only if it does.
"""
has_inline(path::AbstractString) =
    any(ex -> contains_testset(something(prune(ex, f -> false, true), :nothing)),
        Meta.parseall(read(path, String); filename = path).args)
has_inline(s::Suite) = has_inline(s.entry)

"""
    run_selected(keep, core, entry; mod = Main)

Evaluate the entry file with every top-level expression pruned by
[`prune`](@ref). Line numbers and relative include paths are those of the
original file.
"""
run_selected(keep, core::Bool, entry::AbstractString; mod::Module = Main) =
    Base.include(ex -> prune(ex, keep, core), mod, entry)

end # module
