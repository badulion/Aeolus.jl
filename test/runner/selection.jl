# Test selection for Aeolus.jl (#316).
#
# `test/runtests.jl`'s include list is the single list of test files. This
# module reads that list from the file's syntax tree, joins it with the tag
# table `test/test_tags.toml` (tier, weight, exclusive) and evaluates a
# pruned copy of `runtests.jl` that keeps only the selected files, each
# inside the `@testset` blocks that enclose it there. It is used by
#
#   * the serial run, `julia --project=. test/runtests.jl`, which honours
#     `AEOLUS_TEST_TIER=fast|full` through the header of runtests.jl;
#   * the parallel runner `test/run.jl`, whose workers (`worker.jl`) each
#     evaluate one file this way.
#
# Stdlib only. See docs/src/testing.md.

module AeolusTestSelection

using TOML

"Pseudo-file name for the inline `@testset`s written directly in runtests.jl."
const CORE = "runtests.jl"

const TESTDIR  = normpath(joinpath(@__DIR__, ".."))
const RUNTESTS = joinpath(TESTDIR, "runtests.jl")
const TAGFILE  = joinpath(TESTDIR, "test_tags.toml")

const TIERS = ("fast", "full")

"""
    Tag(tier, weight, exclusive, seconds, reason)

Tag-table entry for one test file. Untagged files get `Tag()`: fast tier,
light, not exclusive.

* `tier`: `"fast"` runs in both tiers, `"full"` only in the full tier.
* `weight`: `"heavy"` files are started first by the parallel runner.
* `exclusive`: run alone after the parallel batch (timing-sensitive files).
* `seconds`: measured wall time, used to order the parallel batch.
* `reason`: why a file is full-only (required for `tier = "full"`).
"""
struct Tag
    tier::String
    weight::String
    exclusive::Bool
    seconds::Float64
    reason::String
end
Tag() = Tag("fast", "light", false, 0.0, "")

"""
    tier() -> String

The selected tier: `ENV["AEOLUS_TEST_TIER"]`, `"fast"` when unset.
"""
function tier()
    t = lowercase(strip(get(ENV, "AEOLUS_TEST_TIER", "fast")))
    isempty(t) && return "fast"
    t in TIERS || error("AEOLUS_TEST_TIER must be one of $(join(TIERS, ", ")), got \"$t\"")
    return t
end

"""
    read_tags(path = TAGFILE) -> Dict{String,Tag}

Read the tag table. Errors on unknown keys, bad values or a full-only file
without a reason.
"""
function read_tags(path::AbstractString = TAGFILE)
    isfile(path) || return Dict{String,Tag}()
    raw = TOML.parsefile(path)
    tags = Dict{String,Tag}()
    for (file, d) in raw
        d isa AbstractDict || error("$path: entry \"$file\" must be a table")
        for k in keys(d)
            k in ("tier", "weight", "exclusive", "seconds", "reason") ||
                error("$path: unknown key \"$k\" in \"$file\"")
        end
        t = Tag(String(get(d, "tier", "fast")), String(get(d, "weight", "light")),
                Bool(get(d, "exclusive", false)), Float64(get(d, "seconds", 0.0)),
                String(get(d, "reason", "")))
        t.tier in TIERS || error("$path: \"$file\": tier must be fast or full")
        t.weight in ("light", "heavy") || error("$path: \"$file\": weight must be light or heavy")
        t.tier == "full" && isempty(strip(t.reason)) &&
            error("$path: \"$file\" is full-only and needs a one-line `reason`")
        tags[file] = t
    end
    return tags
end

tag(tags::AbstractDict, file) = get(tags, file, Tag())

"Does `tier` run a file tagged `t`?"
selected(t::Tag, tier::AbstractString) = tier == "full" || t.tier == "fast"

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

"Is `ex` the selection header of runtests.jl (anything naming this module)?"
function is_header(ex)
    ex === :AeolusTestSelection && return true
    ex isa QuoteNode && return is_header(ex.value)
    ex isa Expr || return false
    return any(is_header, ex.args)
end

function has_include(ex)
    include_target(ex) === nothing || return true
    ex isa Expr || return false
    return any(has_include, ex.args)
end

"""
    listed_files(path = RUNTESTS) -> Vector{String}

The test files that runtests.jl includes, in order (the header excluded).
"""
function listed_files(path::AbstractString = RUNTESTS)
    files = String[]
    walk(ex) = begin
        f = include_target(ex)
        f === nothing || (push!(files, f); return)
        ex isa Expr && foreach(walk, ex.args)
    end
    for ex in Meta.parseall(read(path, String); filename = path).args
        is_header(ex) || walk(ex)
    end
    return files
end

"""
    prune(ex, keep, core) -> Expr or nothing

Prune one top-level expression of runtests.jl.

* `include("f")` survives iff `keep(f)`.
* An `@testset` that contains includes survives iff something in it does.
* An `@testset` without includes (an inline test in runtests.jl) survives
  iff `core`.
* Every other statement (`using`, constants) survives.
"""
function prune(ex, keep, core::Bool)
    is_header(ex) && return nothing
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

"""
    run_selected(keep, core; mod = Main, path = RUNTESTS)

Evaluate runtests.jl with every top-level expression pruned by
[`prune`](@ref). Line numbers and relative include paths are those of the
original file.
"""
run_selected(keep, core::Bool; mod::Module = Main, path::AbstractString = RUNTESTS) =
    Base.include(ex -> prune(ex, keep, core), mod, path)

"""
    serial_entry(path) -> Bool

Called by the header of runtests.jl. In the full tier it returns `false`
and runtests.jl runs unchanged. Otherwise it runs the selected files and
returns `true` (the header then exits, so the unpruned file is skipped).
"""
function serial_entry(path::AbstractString)
    t = tier()
    t == "full" && return false
    if isinteractive()
        @warn "AEOLUS_TEST_TIER=$t is ignored in an interactive session; running the full tier. " *
              "Use `julia --project=. test/runtests.jl` or test/run.jl to select a tier."
        return false
    end
    tags = read_tags()
    files = listed_files(path)
    skipped = [f for f in files if !selected(tag(tags, f), t)]
    println("AEOLUS_TEST_TIER=$t: running $(length(files) - length(skipped)) of $(length(files)) ",
            "test files; full-only files skipped: ", isempty(skipped) ? "none" : join(skipped, ", "))
    println("Run the full tier with AEOLUS_TEST_TIER=full (see docs/src/testing.md).")
    run_selected(f -> selected(tag(tags, f), t), true; path = path)
    return true
end

end # module
