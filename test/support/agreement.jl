# Comparing simulated results without bit-identity (epic #493, #495).
#
# Results need not be bit-identical across thread counts (Julia, BLAS,
# FFTW), machines, restarts or refactors (maintainer decision 2026-10-16).
# Two runs of the same computation that only round differently are compared
# with `agree` (in a test, both arrays at hand) or through a stored
# `fingerprint` (a regression fixture), both relative to the field's size.
#
# The default tolerance, ROUNDOFF_RTOL = 1e-10 of the field's largest
# magnitude, was measured: the Berlin smoke (12 nested steps, turbulence,
# radiosity, chemistry) at 1 and at 12 BLAS threads differs by at most
# 7.1e-14 of the field's size (the SGS TKE e; u 4.7e-15, θ 3.9e-16), so
# 1e-10 leaves a factor of ~1000 for round-off and catches any change of a
# formula, a coefficient or a stencil, which moves a field by far more.
# A comparison over many steps of a chaotic flow passes its own,
# larger tolerance with its reason.
#
# Exact equality stays where data is copied, not computed (a checkpoint
# read back, a config parsed, a driver file regenerated from the same
# input) and for integer and categorical data (masks, class maps, counts).

using TOML: TOML

"Relative tolerance of `agree` and `fingerprint_agree`: measured round-off times ~1000 (see above)."
const ROUNDOFF_RTOL = 1e-10

"""
    field_rel_diff(a, b) -> Float64

`max |a - b|` over the entries finite in both, divided by `max |b|` (the
field's size; 1 for an all-zero `b`). `Inf` when the sizes differ or the
non-finite entries are not in the same places with the same values.
"""
function field_rel_diff(a, b)
    size(a) == size(b) || return Inf
    d = 0.0
    m = 0.0
    for (x, y) in zip(a, b)
        fx, fy = isfinite(x), isfinite(y)
        fx == fy || return Inf
        if fx
            d = max(d, abs(Float64(x) - Float64(y)))
            m = max(m, abs(Float64(y)))
        else
            isequal(x, y) || (isnan(x) && isnan(y)) || return Inf
        end
    end
    return m == 0 ? d : d / m
end

"""
    agree(a, b; rtol = ROUNDOFF_RTOL, atol = 0.0) -> Bool

Whether two simulated fields (arrays, or numbers) are the same computation
up to round-off: `max |a - b| ≤ rtol · max |b| + atol` over the finite
entries, with NaN/Inf in the same places. Use in place of `==` on
simulated floats.
"""
agree(a::Number, b::Number; rtol::Real = ROUNDOFF_RTOL, atol::Real = 0.0) = agree([a], [b]; rtol, atol)
function agree(a, b; rtol::Real = ROUNDOFF_RTOL, atol::Real = 0.0)
    size(a) == size(b) || return false
    m = 0.0
    for y in b
        isfinite(y) && (m = max(m, abs(Float64(y))))
    end
    r = field_rel_diff(a, b)
    isfinite(r) || return false
    return (m == 0 ? r : r * m) <= rtol * m + atol
end

"""
    agree_all(a, b; rtol, atol) -> Vector{String}

The keys of the collections of fields `a` and `b` (Dicts, NamedTuples,
vectors of pairs) whose fields do not `agree`, or that only one has.
Empty when every field agrees.
"""
function agree_all(a, b; rtol::Real = ROUNDOFF_RTOL, atol::Real = 0.0)
    da, db = Dict(string(k) => v for (k, v) in pairs_of(a)), Dict(string(k) => v for (k, v) in pairs_of(b))
    bad = String[]
    for k in sort!(collect(union(keys(da), keys(db))))
        (haskey(da, k) && haskey(db, k) && agree(da[k], db[k]; rtol, atol)) || push!(bad, k)
    end
    return bad
end
pairs_of(x::AbstractVector{<:Pair}) = x
pairs_of(x) = pairs(x)

const _FP_SAMPLES = 16

"""
    fingerprint(a) -> Dict{String,Any}

A regression fixture of a simulated field, stored in TOML in place of a
digest of its bits: its length `n`, the number of non-finite entries, the
mean, mean |·|, RMS, min and max of the finite entries, and the values at
16 evenly spaced linear indices. [`fingerprint_agree`](@ref) compares two
within a tolerance of the field's size.
"""
function fingerprint(a)
    x = vec(collect(a))
    n = length(x)
    fin = Float64[Float64(v) for v in x if isfinite(v)]
    d = Dict{String,Any}("n" => n, "nonfinite" => n - length(fin))
    if !isempty(fin)
        nf = length(fin)
        d["mean"] = sum(fin) / nf
        d["mean_abs"] = sum(abs, fin) / nf
        d["rms"] = sqrt(sum(abs2, fin) / nf)
        d["min"] = minimum(fin)
        d["max"] = maximum(fin)
    end
    idx = n == 0 ? Int[] : unique(round.(Int, range(1, n; length = min(n, _FP_SAMPLES))))
    d["samples"] = Float64[Float64(x[i]) for i in idx]
    return d
end

"""
    fingerprint_agree(got, ref; rtol = ROUNDOFF_RTOL, atol = 0.0) -> Bool

Whether two fingerprints (dicts from [`fingerprint`](@ref), `ref` read
from a fixture) describe the same field up to round-off: equal lengths and
non-finite counts, and every statistic and sample within `rtol` of the
reference field's size (`max(|min|, |max|)`) plus `atol`.
"""
function fingerprint_agree(got, ref; rtol::Real = ROUNDOFF_RTOL, atol::Real = 0.0)
    (got["n"] == ref["n"] && got["nonfinite"] == ref["nonfinite"]) || return false
    haskey(ref, "max") == haskey(got, "max") || return false
    scale = haskey(ref, "max") ? max(abs(ref["min"]), abs(ref["max"])) : 0.0
    tol = rtol * scale + atol
    close(x, y) = (isnan(x) && isnan(y)) || (isinf(x) || isinf(y) ? x == y : abs(x - y) <= tol)
    for k in ("mean", "mean_abs", "rms", "min", "max")
        haskey(ref, k) && (close(Float64(got[k]), Float64(ref[k])) || return false)
    end
    gs, rs = got["samples"], ref["samples"]
    length(gs) == length(rs) || return false
    return all(close(Float64(x), Float64(y)) for (x, y) in zip(gs, rs))
end

"""
    fingerprints_differing(got, ref; rtol, atol) -> Vector{String}

The names whose fingerprint in `got` (name => field, or name => fingerprint
dict) does not agree with the fixture table `ref` (name => fingerprint),
or that only one side has.
"""
function fingerprints_differing(got, ref; rtol::Real = ROUNDOFF_RTOL, atol::Real = 0.0)
    g = Dict(string(k) => (v isa AbstractDict ? v : fingerprint(v)) for (k, v) in pairs_of(got))
    bad = String[]
    for k in sort!(collect(union(keys(g), keys(ref))))
        (haskey(g, k) && haskey(ref, k) && fingerprint_agree(g[k], ref[k]; rtol, atol)) || push!(bad, k)
    end
    return bad
end

"""
    diagnostics_columns(rows; skip = ("wall",)) -> Dict{String,Vector{Float64}}

The numeric columns of a `diagnostics.csv` (its lines, header first), for
`fingerprint`ing: every column whose header does not start with one of
`skip` (wall-clock columns differ between runs) and that parses as numbers.
"""
function diagnostics_columns(rows; skip = ("wall",))
    hdr = split(rows[1], ',')
    out = Dict{String,Vector{Float64}}()
    for (j, h) in enumerate(hdr)
        any(s -> startswith(h, s), skip) && continue
        vals = [tryparse(Float64, split(r, ',')[j]) for r in rows[2:end]]
        all(!isnothing, vals) && (out[String(h)] = Float64.(vals))
    end
    return out
end

"""
    diagnostics_differing(got, ref; rtol, atol) -> Vector{String}

[`fingerprints_differing`](@ref) for the columns of a diagnostics.csv
(`got`, `ref`: column => fingerprint).
"""
diagnostics_differing(got, ref; rtol::Real = ROUNDOFF_RTOL, atol::Real = 0.0) =
    fingerprints_differing(got, ref; rtol, atol)

"""
    values_agree(a, b; rtol = ROUNDOFF_RTOL, atol = 0.0) -> Bool

Recursive agreement of two parsed outputs (a `results.toml` table, a
NetCDF file's variables, a diagnostics row): dictionaries with the same
keys, collections of the same length, floating-point numbers and arrays
within [`agree`](@ref), everything else (strings, integers, booleans,
symbols) equal.
"""
function values_agree(a, b; rtol::Real = ROUNDOFF_RTOL, atol::Real = 0.0)
    if a isa AbstractDict && b isa AbstractDict
        keys(a) == keys(b) || Set(keys(a)) == Set(keys(b)) || return false
        return all(k -> values_agree(a[k], b[k]; rtol, atol), keys(a))
    elseif a isa AbstractArray{<:AbstractFloat} && b isa AbstractArray{<:AbstractFloat}
        return agree(a, b; rtol, atol)
    elseif a isa AbstractFloat && b isa AbstractFloat
        return agree(a, b; rtol, atol)
    elseif a isa AbstractArray{<:Union{Missing,AbstractFloat}} && b isa AbstractArray{<:Union{Missing,AbstractFloat}}
        size(a) == size(b) || return false
        ma, mb = ismissing.(a), ismissing.(b)
        ma == mb || return false
        return agree(Float64[x for x in a if !ismissing(x)], Float64[x for x in b if !ismissing(x)]; rtol, atol)
    elseif (a isa Union{Tuple,AbstractArray}) && (b isa Union{Tuple,AbstractArray})
        length(a) == length(b) || return false
        return all(values_agree(x, y; rtol, atol) for (x, y) in zip(a, b))
    elseif a isa NamedTuple && b isa NamedTuple
        keys(a) == keys(b) || return false
        return all(k -> values_agree(a[k], b[k]; rtol, atol), keys(a))
    elseif typeof(a) === typeof(b) && !isbitstype(typeof(a)) && isstructtype(typeof(a)) &&
           !(a isa Union{AbstractString,Symbol,Function,Module,Type}) && fieldcount(typeof(a)) > 0
        # a snapshot struct (a closure's memory, a generator's state): field by field
        return all(f -> values_agree(getfield(a, f), getfield(b, f); rtol, atol), fieldnames(typeof(a)))
    end
    return isequal(a, b)
end

"""
    csv_agree(rows_a, rows_b; rtol, atol) -> Bool

Two CSV files' lines (header first) agree: the same header and number of
rows, numeric cells within [`agree`](@ref) column by column (relative to
the column's size), every other cell equal.
"""
function csv_agree(ra, rb; rtol::Real = ROUNDOFF_RTOL, atol::Real = 0.0)
    length(ra) == length(rb) || return false
    isempty(ra) && return true
    ra[1] == rb[1] || return false
    ca = [split(r, ',') for r in ra[2:end]]; cb = [split(r, ',') for r in rb[2:end]]
    all(length(x) == length(y) for (x, y) in zip(ca, cb)) || return false
    ncol = isempty(ca) ? 0 : length(ca[1])
    for j in 1:ncol
        xa = [tryparse(Float64, r[j]) for r in ca]; xb = [tryparse(Float64, r[j]) for r in cb]
        if all(!isnothing, xa) && all(!isnothing, xb)
            agree(Float64.(xa), Float64.(xb); rtol, atol) || return false
        else
            all(r -> ca[r][j] == cb[r][j], eachindex(ca)) || return false
        end
    end
    return true
end

"""
    capturing_fingerprints() -> Bool

`AEOLUS_CAPTURE_FINGERPRINTS=1`: the tests that keep a fingerprint fixture
record what they compute ([`fixture_agree!`](@ref)) and write their
fixture file ([`write_fixture`](@ref)) instead of comparing. Re-capture
only for a change that moves the results beyond round-off, with the reason
in the commit.
"""
capturing_fingerprints() = get(ENV, "AEOLUS_CAPTURE_FINGERPRINTS", "") == "1"

"""
    fixture_agree!(fixture, label, key, value; rtol, atol) -> Bool

Compare `value`'s fingerprint with `fixture[label][key]` (a table read from
a fingerprint fixture), or, when [`capturing_fingerprints`](@ref), record it
there and return `true`.
"""
function fixture_agree!(fixture::AbstractDict, label, key, value; rtol::Real = ROUNDOFF_RTOL, atol::Real = 0.0)
    fp = fingerprint(value)
    if capturing_fingerprints()
        get!(fixture, string(label), Dict{String,Any}())[string(key)] = fp
        return true
    end
    ref = get(get(fixture, string(label), Dict()), string(key), nothing)
    ref === nothing && return false
    ok = fingerprint_agree(fp, ref; rtol, atol)
    ok || @info "fingerprint differing from the fixture" label key
    return ok
end

"Write a fingerprint fixture (when capturing): `header` lines as TOML comments, then the table."
function write_fixture(path, fixture; header = String[])
    capturing_fingerprints() || return nothing
    open(path, "w") do io
        foreach(l -> println(io, "# ", l), header)
        TOML.print(io, fixture; sorted = true)
    end
    println("wrote the fingerprint fixture ", path)
    return path
end

"""
    summary_of(x) -> Any

A compact, comparable summary of a computed quantity, for results that
travel between processes (the thread-invariance worker): a float array as
its [`fingerprint`](@ref), a float as itself, an integer or Boolean array
(exact data: masks, counts) as its `hash`, a collection or struct field by
field, anything else as itself. Compared by [`summaries_agree`](@ref).
"""
summary_of(x::AbstractArray{<:AbstractFloat}) = fingerprint(x)
summary_of(x::AbstractArray{<:Union{Integer,Bool}}) = Dict{String,Any}("hash" => string(hash(collect(x)); base = 16), "size" => collect(size(x)))
summary_of(x::AbstractArray) = Any[summary_of(v) for v in x]
summary_of(x::AbstractFloat) = Float64(x)
summary_of(x::Union{Integer,Bool,AbstractString,Symbol,Nothing,Missing}) = x
summary_of(x::Tuple) = Any[summary_of(v) for v in x]
summary_of(x::NamedTuple) = Dict{String,Any}(string(k) => summary_of(v) for (k, v) in pairs(x))
summary_of(x::AbstractDict) = Dict{String,Any}(string(k) => summary_of(v) for (k, v) in x)
function summary_of(x)
    T = typeof(x)
    (isstructtype(T) && fieldcount(T) > 0 && !(x isa Union{Function,Module,Type})) || return repr(x)
    return Dict{String,Any}(string(f) => summary_of(getfield(x, f)) for f in fieldnames(T))
end

_is_fingerprint(d) = d isa AbstractDict && haskey(d, "samples") && haskey(d, "nonfinite")

"Whether two [`summary_of`](@ref) results agree: fingerprints within `rtol` of the field's size, floats by [`agree`](@ref), the rest equal."
function summaries_agree(a, b; rtol::Real = ROUNDOFF_RTOL, atol::Real = 0.0)
    if _is_fingerprint(a) && _is_fingerprint(b)
        return fingerprint_agree(a, b; rtol, atol)
    elseif a isa AbstractDict && b isa AbstractDict
        Set(keys(a)) == Set(keys(b)) || return false
        return all(k -> summaries_agree(a[k], b[k]; rtol, atol), keys(a))
    elseif a isa AbstractVector && b isa AbstractVector
        length(a) == length(b) || return false
        return all(summaries_agree(x, y; rtol, atol) for (x, y) in zip(a, b))
    elseif a isa AbstractFloat && b isa AbstractFloat
        return agree(a, b; rtol, atol)
    end
    return isequal(a, b)
end
