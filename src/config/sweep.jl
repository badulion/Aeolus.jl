# Sweeps (CFG-e, #360): one base file plus a `[sweep]` table that the
# runner expands into override sets, one run (one run folder) per member.
# The core loader has no templating: a member is the base config plus
# ordinary `dotted.key=<TOML value>` overrides, and its run.toml records
# those overrides and no [sweep], so it replays as a single run.
#
#   [sweep]
#   axis = [ { key = "sgs.model", values = ["smagorinsky", "vreman"] },
#            { key = "grid.n",      values = [32, 48] } ]          # product
#
#   [[sweep.member]]                                  # explicit members
#   label = "fix"
#   wall_model = { h_wm_cells = 3, filter = "temporal" }

"""
    SweepMember

One member of a sweep: `index` (1-based, in the fixed expansion order),
`label` (part of its run name) and `overrides`, the `dotted.key=<TOML
value>` strings that make it from the base file.
"""
struct SweepMember
    index::Int
    label::String
    overrides::Vector{String}
end

_sweep_error(file, n) = ConfigError("sweep", "this file is a sweep of $n members: run it with " *
                                    "`aeolus run` (every member, or --member k) or load it with " *
                                    "load_sweep"; file = file)

_label_text(v) = v isa AbstractString ? String(v) : v isa Bool ? string(v) :
                 v isa Integer ? string(v) : v isa AbstractFloat ? _emit_float(Float64(v)) :
                 v isa AbstractVector ? join(_label_text.(v), "x") : "table"
_clean_label(s) = replace(s, r"[^A-Za-z0-9._-]+" => "-")

"The `dotted.key=<TOML value>` override of `key` set to the TOML value `v`."
_override_string(key, v) = string(key, "=", _emit_inline(v))

function _emit_inline(v)
    v isa AbstractDict || return _emit_value(v)
    return "{ " * join((string(_emit_key(String(k)), " = ", _emit_inline(x))
                        for (k, x) in sort!(collect(v); by = first)), ", ") * " }"
end

_isdotted(k) = k isa AbstractString && occursin(r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$", k)

"""
    sweep_members(sweep::AbstractDict, file) -> Vector{SweepMember}

Expand a `[sweep]` table: the explicit members (`[[sweep.member]]`, each a
table of `section = { option = value }` plus an optional `label`; none
means one empty member) times the axes (`axis = [{ key = "section.option",
values = [...] }, ...]`), members outermost, then the axes in the order
written, the last axis fastest. A member's label is its `label` (else its
index) followed by the axis values.
"""
function sweep_members(sw::AbstractDict, file::AbstractString)
    probs = ConfigProblem[]
    for k in sort!(collect(keys(sw)))
        k in ("axis", "member") ||
            push!(probs, ConfigProblem("sweep.$k", file, "unknown key in [sweep] (keys: axis, member)" *
                                       _suggest(k, ["axis", "member"])))
    end
    axes = Tuple{String,Vector{Any}}[]
    for (i, a) in enumerate(get(sw, "axis", Any[]))
        key = "sweep.axis[$i]"
        if !(a isa AbstractDict) || sort!(collect(keys(a))) != ["key", "values"]
            push!(probs, ConfigProblem(key, file, "an axis is { key = \"section.option\", values = [...] }"))
            continue
        end
        _isdotted(a["key"]) ||
            push!(probs, ConfigProblem(key, file, "key: expected \"section.option\", got $(_show_raw(a["key"]))"))
        (a["values"] isa AbstractVector && !isempty(a["values"])) ||
            push!(probs, ConfigProblem(key, file, "values: expected a non-empty array"))
        isempty(probs) && push!(axes, (String(a["key"]), Any[a["values"]...]))
    end
    members = Tuple{String,Vector{String}}[]
    for (i, m) in enumerate(get(sw, "member", Any[]))
        key = "sweep.member[$i]"
        m isa AbstractDict || (push!(probs, ConfigProblem(key, file, "a member is a table")); continue)
        ovr = String[]; label = string(i)
        for k in sort!(collect(keys(m)))
            v = m[k]
            if k == "label"
                v isa AbstractString && !isempty(v) ? (label = String(v)) :
                    push!(probs, ConfigProblem("$key.label", file, "expected a non-empty string"))
            elseif v isa AbstractDict
                for o in sort!(collect(keys(v)))
                    push!(ovr, _override_string("$k.$o", v[o]))
                end
            else
                push!(probs, ConfigProblem("$key.$k", file,
                      "expected a table of options (section = { option = value }) or label"))
            end
        end
        push!(members, (label, ovr))
    end
    isempty(probs) || throw(ConfigError(_sorted(probs)))
    explicit = !isempty(members)
    explicit || push!(members, ("", String[]))
    out = SweepMember[]
    combos = isempty(axes) ? [()] : vec(collect(Iterators.product(reverse([a[2] for a in axes])...)))
    # Iterators.product varies its first argument fastest: reversed, the last axis is fastest
    combos = isempty(axes) ? combos : [reverse(c) for c in combos]
    for (label, movr) in members, c in combos
        parts = String[]
        isempty(label) || push!(parts, label)
        ovr = copy(movr)
        for (j, v) in enumerate(c)
            push!(ovr, _override_string(axes[j][1], v)); push!(parts, _label_text(v))
        end
        push!(out, SweepMember(length(out) + 1, _clean_label(join(parts, "_")), ovr))
    end
    (explicit || !isempty(axes)) ||
        throw(ConfigError("sweep", "an empty [sweep]: give axis = [...] or [[sweep.member]]"; file = file))
    return out
end

"""
    sweep_members(path; overrides = String[]) -> Union{Nothing,Vector{SweepMember}}

The members of the sweep defined by the config file `path` (its
`[sweep]`), or `nothing` when it defines none.
"""
function sweep_members(path::AbstractString; overrides = String[])
    p = abspath(path)
    (isfile(p) && !isdir(p)) || return nothing
    d = _parse_toml_file(p)
    (haskey(d, "run_id") && get(d, "config", nothing) isa AbstractDict) && return nothing
    sw = _load_file(p, d, overrides, nothing; sweep = :table)
    sw isa AbstractDict || return nothing
    return sweep_members(sw, p)
end

"""
    load_sweep(path; overrides = String[], member = nothing) -> Vector{Pair{SweepMember,RunConfig}}

Load every member of the sweep in `path` (or only `member`: an index or a
label): the base config plus the member's overrides, then `overrides`
(precedence base < member < `overrides`). Unless the member or
`overrides` set `run.name`, a member's run name is the base name plus
`_<label>`. A file without a `[sweep]` gives one member (label `""`).
"""
function load_sweep(path::AbstractString; overrides = String[], member = nothing)
    ms = sweep_members(path; overrides = overrides)
    if ms === nothing
        member === nothing || member == 1 || member == "" ||
            throw(ConfigError("sweep", "--member: the file defines no sweep"; file = abspath(path)))
        return [SweepMember(1, "", String[]) => load_run_config(path; overrides = overrides)]
    end
    if member !== nothing
        sel = filter(m -> m.index == member || m.label == string(member), ms)
        isempty(sel) && throw(ConfigError("sweep", "no member $(repr(member)) (members: " *
                                          join(("$(m.index) ($(m.label))" for m in ms), ", ") * ")";
                                          file = abspath(path)))
        ms = sel
    end
    out = Pair{SweepMember,RunConfig}[]
    for m in ms
        ovr = vcat(m.overrides, String[overrides...])
        named = any(s -> startswith(strip(s), "run.name"), ovr)
        cfg = _load_member(path, ovr)
        if !named && !isempty(m.label)
            cfg = _load_member(path, vcat(m.overrides, [_override_string("run.name", cfg.run.name * "_" * m.label)],
                                          String[overrides...]))
        end
        push!(out, m => cfg)
    end
    return out
end

function _load_member(path, ovr)
    p = abspath(path)
    return _load_file(p, _parse_toml_file(p), ovr, nothing; sweep = :ignore)
end

_member_arg(opts) = haskey(opts, "member") ? something(tryparse(Int, opts["member"]), opts["member"]) : nothing
