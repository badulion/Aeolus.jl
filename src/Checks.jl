"""
    Aeolus.Checks

The in-model checks (ONE-d, #453; the design is
`docs/src/design/single_model.md`, "Budgets and checks"). The model and
its components **publish** named checks ([`Check`](@ref)); the
configuration's `[checks]` section **enables** them, each as required or
advisory and with a tolerance; the run evaluates them and writes the
results to `results.toml` ([`results_table`](@ref)); a failed required
check stops `aeolus run` with a non-zero exit code ([`ChecksFailed`](@ref)).

A check measures one non-negative **error** `value` and passes when
`value ≤ tolerance`:

| check | value |
| :-- | :-- |
| a budget closure | the closure and the unattributed change, per second of the run |
| `finite` (the sanity check) | the number of non-finite values in the state |
| a reference comparison ([`reference_check`](@ref)) | the distance outside the reference band, or the RMS difference of a profile |

A reference comparison reads a reference data file (CSV or NetCDF,
[`read_reference`](@ref)) named in the configuration, so a comparison
with an intercomparison ensemble (GABLS-style) is configuration plus data,
not code.

The checks a component publishes are `Components.checks(c)`; the core's
(the sanity check and the budget closures) are `Components.core_checks`.
This module holds what is common to all of them: the check, its settings
and evaluation, the reference data and the results table.
"""
module Checks

using Printf: @sprintf
using NCDatasets: NCDataset, dimnames
import ..IO: checkpoint_state, restore_checkpoint_state!

export Check, CheckSetting, CheckRunner, CheckResult, ChecksFailed, check_entry_problems,
       reference_entry_problems, enabled_checks, sample_checks!, evaluate_checks, stopped_checks,
       results_table, required_failures, advisory_failures, ReferenceValue, ReferenceProfile,
       read_reference, reference_check, WindowMean

# ---------------------------------------------------------------------------
# a check
# ---------------------------------------------------------------------------

"""
    Check(name; doc, units = "1", tolerance, evaluate, sample = nothing, acc = nothing)

One published check:

  * `name` — its key in `[checks]` and in `results.toml`;
  * `doc` — what it measures, in words (written to `results.toml`);
  * `units` — the units of its value and tolerance;
  * `tolerance` — its default tolerance (`[checks] name = {tolerance}`
    overrides it);
  * `evaluate(acc) -> (value, info)` — the error measure `value ≥ 0` at
    the end of the run, and a `Dict{String,Any}` of numbers worth
    recording (the model's value, the reference band, …);
  * `sample(acc, t, dt)` — called after every step (stage 13, `:checks`)
    with the time `t` at its end and its length `dt`, for a check that
    accumulates over the run (`nothing`: none);
  * `acc` — that accumulator (checkpointed with the run, so a resumed run
    continues it).

It passes when `value ≤ tolerance`; a `NaN` value fails.
"""
struct Check
    name::String
    doc::String
    units::String
    tolerance::Float64
    evaluate::Function
    sample::Union{Nothing,Function}
    acc::Any
end
Check(name::AbstractString; doc::AbstractString, units::AbstractString = "1", tolerance::Real,
      evaluate::Function, sample = nothing, acc = nothing) =
    Check(String(name), String(doc), String(units), Float64(tolerance), evaluate, sample, acc)

"""
    CheckSetting(required, tolerance)

What `[checks]` sets for one check: `required` (a failure stops the run
with a non-zero exit code) or advisory (a failure warns), and the
`tolerance` (`nothing`: the check's default).
"""
struct CheckSetting
    required::Bool
    tolerance::Union{Nothing,Float64}
end

_isnum(x) = x isa Real && !(x isa Bool)

"""
    check_entry_problems(tbl) -> Vector{String}

The problems of one `[checks]` entry `name = {required, tolerance}`: an
empty table is "off"; otherwise `required` (a Bool) must be given and
`tolerance`, if given, is a number ≥ 0.
"""
function check_entry_problems(tbl; extra = ())
    isempty(tbl) && return String[]
    probs = String[]
    keys_ = ("required", "tolerance", extra...)
    for k in sort!(collect(keys(tbl)))
        k in keys_ || push!(probs, "$k: unknown key (keys: $(join(keys_, ", ")))")
    end
    haskey(tbl, "required") || push!(probs, "required: give required = true (a failure stops the run) or false (it warns)")
    haskey(tbl, "required") && !(tbl["required"] isa Bool) && push!(probs, "required: expected true or false")
    if haskey(tbl, "tolerance")
        t = tbl["tolerance"]
        (_isnum(t) && t >= 0) || push!(probs, "tolerance: expected a number ≥ 0, got $(repr(t))")
    end
    return probs
end

"""
    reference_entry_problems(tbl) -> Vector{String}

The problems of one reference comparison `[checks.reference.<quantity>]`:
those of [`check_entry_problems`](@ref) and its own keys, `variable` (a
string: the output-catalogue variable compared; default the quantity's
name) and `window = [t0, t1]` (s, t0 < t1: the time mean over the steps
ending in it; default the value at the end of the run). Unlike a
published check, a reference comparison is always on: `required` is
mandatory.
"""
function reference_entry_problems(tbl)
    tbl isa AbstractDict || return ["expected a table {required, tolerance, variable, window}"]
    isempty(tbl) && return ["required: give required = true or false"]
    probs = check_entry_problems(tbl; extra = ("variable", "window"))
    haskey(tbl, "variable") && !(tbl["variable"] isa AbstractString) && push!(probs, "variable: expected a string")
    if haskey(tbl, "window")
        w = tbl["window"]
        (w isa AbstractVector && length(w) == 2 && all(_isnum, w) && w[1] < w[2]) ||
            push!(probs, "window: expected [t0, t1] with t0 < t1 (s)")
    end
    return probs
end

# ---------------------------------------------------------------------------
# the enabled checks of a run, their sampling and evaluation
# ---------------------------------------------------------------------------

"""
    CheckResult

The outcome of one enabled check: `name`, `required`, `evaluated`
(`false` when the run stopped before the end), `passed`, `value`,
`tolerance`, `units`, `doc` and `info` (the check's recorded numbers, or
the reason it was not evaluated).
"""
struct CheckResult
    name::String
    required::Bool
    evaluated::Bool
    passed::Bool
    value::Float64
    tolerance::Float64
    units::String
    doc::String
    info::Dict{String,Any}
end

"""
    CheckRunner(checks, settings)

The enabled checks of a run: each of `checks` that `settings` (a
`Dict{String,CheckSetting}`, from `[checks]`) names, with its setting. An
`ArgumentError` names a setting no check publishes. Its accumulators are
checkpointed with the run (`checkpoint_state`).
"""
struct CheckRunner
    checks::Vector{Check}
    settings::Vector{CheckSetting}
end

function CheckRunner(checks, settings::AbstractDict)
    by = Dict{String,Check}()
    for c in checks
        haskey(by, c.name) && throw(ArgumentError("two published checks are named $(c.name)"))
        by[c.name] = c
    end
    missing_ = sort!([n for n in keys(settings) if !haskey(by, n)])
    isempty(missing_) || throw(ArgumentError("checks enabled but not published: $(join(missing_, ", ")) " *
                                             "(published: $(join(sort!(collect(keys(by))), ", ")))"))
    names = sort!(collect(String, keys(settings)))
    return CheckRunner(Check[by[n] for n in names], CheckSetting[settings[n] for n in names])
end

# a resumed run continues the accumulators (#269): one per enabled check
checkpoint_state(r::CheckRunner) = Any[deepcopy(c.acc) for c in r.checks]
function restore_checkpoint_state!(r::CheckRunner, s)
    length(s) == length(r.checks) || throw(ArgumentError("the checkpoint holds $(length(s)) check accumulators, the run has $(length(r.checks))"))
    for (c, a) in zip(r.checks, s)
        c.acc === nothing && continue
        for f in fieldnames(typeof(c.acc))
            setfield!(c.acc, f, deepcopy(getfield(a, f)))
        end
    end
    return r
end

"""
    enabled_checks(tbl) -> Dict{String,CheckSetting}

The checks a resolved `[checks]` table enables: every entry that is a
non-empty table (`{required, tolerance}`), and every reference
comparison (`reference`, named `reference_<quantity>`).
"""
function enabled_checks(tbl)
    out = Dict{String,CheckSetting}()
    for (k, v) in tbl
        v isa AbstractDict || continue
        if k == "reference"
            for (q, e) in v
                out["reference_" * q] = _setting(e)
            end
        elseif !isempty(v)
            out[k] = _setting(v)
        end
    end
    return out
end
_setting(e) = CheckSetting(e["required"], haskey(e, "tolerance") ? Float64(e["tolerance"]) : nothing)

"""
    sample_checks!(r::CheckRunner, t, dt)

Let every enabled check that accumulates over the run sample the step
that ended at `t` and was `dt` long (stage 13, `:checks`).
"""
function sample_checks!(r::CheckRunner, t, dt)
    for c in r.checks
        c.sample === nothing || c.sample(c.acc, Float64(t), Float64(dt))
    end
    return r
end

_tol(c::Check, s::CheckSetting) = something(s.tolerance, c.tolerance)

"""
    evaluate_checks(r::CheckRunner) -> Vector{CheckResult}

Evaluate every enabled check at the end of the run. A check whose
evaluation throws fails, with the error as its `info["error"]`.
"""
function evaluate_checks(r::CheckRunner)
    out = CheckResult[]
    for (c, s) in zip(r.checks, r.settings)
        tol = _tol(c, s)
        value, info = try
            v, i = c.evaluate(c.acc)
            Float64(v), Dict{String,Any}(i)
        catch e
            NaN, Dict{String,Any}("error" => first(sprint(showerror, e), 500))
        end
        push!(out, CheckResult(c.name, s.required, true, value <= tol, value, tol, c.units, c.doc, info))
    end
    return out
end

"""
    stopped_checks(r::CheckRunner, err; step, t) -> Vector{CheckResult}

The results of a run that stopped with the error `err` at `step`, time
`t`: the sanity check `finite` fails (a blown-up run records it whether
or not it was enabled; required) with the error as its reason, every
other enabled check is recorded as not evaluated, and failed.
"""
function stopped_checks(r::CheckRunner, err; step::Integer, t::Real)
    why = first(sprint(showerror, err), 500)
    info = Dict{String,Any}("stopped_at_step" => Int(step), "stopped_at_t" => Float64(t), "error" => why)
    out = CheckResult[]
    have_finite = false
    for (c, s) in zip(r.checks, r.settings)
        fin = c.name == "finite"
        have_finite |= fin
        push!(out, CheckResult(c.name, s.required, fin, false, NaN, _tol(c, s), c.units, c.doc, copy(info)))
    end
    have_finite || pushfirst!(out, CheckResult("finite", true, true, false, NaN, 0.0, "1",
                                                "the run stopped with an error (a blow-up)", info))
    return out
end

"The required checks of `results` that failed."
required_failures(results) = filter(x -> x.required && !x.passed, results)
"The advisory checks of `results` that failed."
advisory_failures(results) = filter(x -> !x.required && !x.passed, results)

"""
    results_table(results) -> Dict{String,Any}

The `results.toml` table of a run's check results: `passed` (every
required check passed), `failed_required` and `failed_advisory` (names),
and per check `[checks.<name>]` with `required`, `evaluated`, `passed`,
`value`, `tolerance`, `units`, `description` and its recorded numbers.
"""
function results_table(results)
    checks = Dict{String,Any}()
    for x in results
        t = Dict{String,Any}("required" => x.required, "evaluated" => x.evaluated, "passed" => x.passed,
                             "value" => x.value, "tolerance" => x.tolerance, "units" => x.units,
                             "description" => x.doc)
        for (k, v) in x.info
            t[k] = v
        end
        checks[x.name] = t
    end
    return Dict{String,Any}("passed" => isempty(required_failures(results)),
                            "failed_required" => String[x.name for x in required_failures(results)],
                            "failed_advisory" => String[x.name for x in advisory_failures(results)],
                            "checks" => checks)
end

"""
    ChecksFailed(failed)

The error a run ends with when a required check failed (`aeolus run`
exits with code 3): `failed` are the [`CheckResult`](@ref)s. The run's
fields and `results.toml` are complete; the session is recorded as
`failed`.
"""
struct ChecksFailed <: Exception
    failed::Vector{CheckResult}
end

function Base.showerror(io::IO, e::ChecksFailed)
    print(io, "ChecksFailed: ", length(e.failed), length(e.failed) == 1 ? " required check" : " required checks",
          " failed (results.toml)")
    for x in e.failed
        print(io, "\n  - ", x.name, ": ", x.evaluated ? @sprintf("%.6g > tolerance %.6g %s", x.value, x.tolerance, x.units) :
                                                         "not evaluated: " * string(get(x.info, "error", "")))
    end
end

# ---------------------------------------------------------------------------
# reference data and the reference comparison
# ---------------------------------------------------------------------------

"""
    ReferenceValue(low, high)

A scalar reference: the band `[low, high]` (an ensemble range; `low ==
high` for a single value).
"""
struct ReferenceValue
    low::Float64
    high::Float64
end

"""
    ReferenceProfile(z, values)

A vertical reference profile: `values` at the heights `z` (m, increasing).
"""
struct ReferenceProfile
    z::Vector{Float64}
    values::Vector{Float64}
end

_fields(line) = String[strip(s) for s in split(line, ',')]

"""
    read_reference(path) -> Dict{String,Union{ReferenceValue,ReferenceProfile}}

The reference data of a file, by quantity name. Two formats:

  * **CSV** (`#` lines are comments; the first other line is the header):
    a header with a `quantity` column and `low` and `high` columns (or one
    `value` column) gives a [`ReferenceValue`](@ref) per row (other
    columns, such as `unit` and `source`, are ignored; the format of
    `examples/sbl/data/*_reference.csv`); a header whose first column is
    `z` (or `z_m`, heights in m) gives a [`ReferenceProfile`](@ref) per
    other column, named by its (lowercased) header;
  * **NetCDF** (`.nc`): a variable with one value (or attributes
    `low` and `high`) is a `ReferenceValue`; a one-dimensional variable
    along `z`, with the coordinate variable `z`, a `ReferenceProfile`.
"""
function read_reference(path::AbstractString)
    isfile(path) || throw(ArgumentError("reference file $path does not exist"))
    return endswith(lowercase(path), ".nc") ? _read_reference_nc(path) : _read_reference_csv(path)
end

function _read_reference_csv(path)
    lines = [l for l in readlines(path) if !isempty(strip(l)) && !startswith(lstrip(l), '#')]
    isempty(lines) && throw(ArgumentError("reference file $path has no header"))
    head = lowercase.(_fields(lines[1]))
    rows = [_fields(l) for l in lines[2:end]]
    num(s, what) = (x = tryparse(Float64, s); x === nothing ?
                    throw(ArgumentError("reference file $path: $what: expected a number, got $(repr(s))")) : x)
    out = Dict{String,Union{ReferenceValue,ReferenceProfile}}()
    if "quantity" in head
        iq = findfirst(==("quantity"), head)
        il = findfirst(==("low"), head); ih = findfirst(==("high"), head); iv = findfirst(==("value"), head)
        (il !== nothing && ih !== nothing) || iv !== nothing ||
            throw(ArgumentError("reference file $path: a quantity table needs low and high columns, or a value column"))
        for r in rows
            q = r[iq]
            lo, hi = il !== nothing ? (num(r[il], "$q low"), num(r[ih], "$q high")) :
                                      (v = num(r[iv], "$q value"); (v, v))
            lo <= hi || throw(ArgumentError("reference file $path: $q: low $lo > high $hi"))
            out[q] = ReferenceValue(lo, hi)
        end
    elseif head[1] in ("z", "z_m")
        z = [num(r[1], "z") for r in rows]
        issorted(z; lt = <=) || throw(ArgumentError("reference file $path: z must increase strictly"))
        for (j, q) in enumerate(head)
            j == 1 && continue
            out[q] = ReferenceProfile(z, [num(r[j], "$q at z = $(r[1])") for r in rows])
        end
    else
        throw(ArgumentError("reference file $path: the header needs a quantity column (scalar references) " *
                            "or z (z_m) as its first column (profiles)"))
    end
    return out
end

function _read_reference_nc(path)
    out = Dict{String,Union{ReferenceValue,ReferenceProfile}}()
    NCDataset(path, "r") do ds
        z = haskey(ds, "z") ? Float64.(collect(ds["z"][:])) : nothing
        for name in keys(ds)
            name == "z" && continue
            v = ds[name]
            a = v.attrib
            if haskey(a, "low") && haskey(a, "high")
                out[name] = ReferenceValue(Float64(a["low"]), Float64(a["high"]))
            elseif length(v) == 1
                x = Float64(only(collect(v[:])))
                out[name] = ReferenceValue(x, x)
            elseif ndims(v) == 1 && z !== nothing && length(v) == length(z) && only(dimnames(v)) == "z"
                out[name] = ReferenceProfile(z, Float64.(collect(v[:])))
            end
        end
    end
    return out
end

"""
    WindowMean(window)

The accumulator of a reference comparison: the step-length-weighted sum
of the compared value over the steps that end in `window = (t0, t1)`
(`nothing`: none, the value at the end is compared), the summed length
and the number of steps.
"""
mutable struct WindowMean
    window::Union{Nothing,Tuple{Float64,Float64}}
    sum::Any
    weight::Float64
    n::Int
end
WindowMean(window) = WindowMean(window === nothing ? nothing : (Float64(window[1]), Float64(window[2])), nothing, 0.0, 0)

# linear interpolation of (zs, vs) at z, constant beyond the ends
function _interp(zs, vs, z)
    z <= zs[1] && return vs[1]
    z >= zs[end] && return vs[end]
    i = searchsortedlast(zs, z)
    w = (z - zs[i]) / (zs[i+1] - zs[i])
    return (1 - w) * vs[i] + w * vs[i+1]
end

"""
    reference_check(quantity, ref, value_of; variable, z = nothing, units = "", window = nothing,
                    tolerance = 0.0) -> Check

The reference comparison `reference_<quantity>` of the model's
`variable` against the reference data `ref` of `quantity`
([`read_reference`](@ref)). `value_of()` is the model's current value: a
number (compared with a [`ReferenceValue`](@ref)) or a profile at the
heights `z` (compared with a [`ReferenceProfile`](@ref)). With `window =
(t0, t1)` the compared value is its time mean over the steps that end in
the window (weighted by their length); otherwise the value at the end of
the run. The error measure:

  * a scalar: the distance outside the band, `max(low − x, x − high, 0)`;
  * a profile: the RMS of model − reference at the reference heights in
    the model's column (the model profile interpolated linearly; a `NaN`
    reference value is a gap in the data and is skipped, and so is a height
    where the model profile is undefined, `NaN`, such as above the
    virtual towers; a profile undefined everywhere fails).

`tolerance` (in the variable's `units`) is the default; it fails when
no step ended in the window.
"""
function reference_check(quantity::AbstractString, ref, value_of::Function; variable::AbstractString,
                         z = nothing, units::AbstractString = "", window = nothing, tolerance::Real = 0.0)
    profile = ref isa ReferenceProfile
    profile && z === nothing && throw(ArgumentError("reference_check $quantity: a profile reference needs the model heights z"))
    acc = WindowMean(window)
    sample = window === nothing ? nothing : function (a, t, dt)
        a.window[1] <= t <= a.window[2] * (1 + 1e-12) || return nothing
        x = value_of()
        a.sum = a.sum === nothing ? x .* dt : a.sum .+ x .* dt
        a.weight += dt; a.n += 1
        return nothing
    end
    function evaluate(a)
        x = if a.window === nothing
            value_of()
        else
            a.n == 0 && return NaN, Dict{String,Any}("error" => "no step ended in the window $(a.window) s")
            a.sum ./ a.weight
        end
        info = Dict{String,Any}("variable" => String(variable))
        a.window === nothing || (info["window"] = collect(a.window); info["steps_in_window"] = a.n)
        if profile
            zr, vr = ref.z, ref.values
            zm = collect(Float64, z)
            inside = [k for k in eachindex(zr) if zm[1] <= zr[k] <= zm[end] && !isnan(vr[k])]
            isempty(inside) && return NaN, merge(info, Dict{String,Any}("error" => "no reference height inside the model column"))
            d = filter(isfinite, [_interp(zm, x, zr[k]) - vr[k] for k in inside])
            isempty(d) && return NaN, merge(info, Dict{String,Any}("error" => "the model profile is undefined (NaN) at every reference height"))
            info["levels"] = length(d)
            info["undefined_levels"] = length(inside) - length(d)
            info["max_abs_difference"] = maximum(abs, d)
            return sqrt(sum(abs2, d) / length(d)), info
        end
        xv = Float64(x)
        info["model"] = xv; info["low"] = ref.low; info["high"] = ref.high
        return isnan(xv) ? NaN : max(ref.low - xv, xv - ref.high, 0.0), info
    end
    doc = profile ? "RMS difference of the $variable profile from the reference $quantity" :
                    "distance of $variable outside the reference band of $quantity"
    window === nothing || (doc *= ", time mean over $(window[1])–$(window[2]) s")
    return Check("reference_" * quantity; doc = doc, units = units, tolerance = tolerance,
                 evaluate = evaluate, sample = sample, acc = acc)
end

end # module
