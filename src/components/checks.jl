# The checks of a domain (ONE-d, #453): the core's (the sanity check and
# the budget closures), every component's `checks`, and the reference
# comparisons of `[checks.reference]`; `Aeolus.Checks` evaluates them.

"""
    CORE_CHECKS

The checks the core publishes ([`core_checks`](@ref)):

| name | value | default tolerance |
| :-- | :-- | :-- |
| `finite` | the number of non-finite interior values of the prognostic fields (the sanity check) | 0 |
| `energy_closure` | (\\|closure\\| + \\|unattributed\\|) / elapsed of the heat of the air (W m⁻²) | 1e-6 |
| `water_closure` | the same for the water of the air (kg m⁻² s⁻¹) | 1e-12 |
| `momentum_closure` | the larger of x and y momentum (kg m⁻¹ s⁻², i.e. Pa) | 1e-9 |
| `tracer_closure` | the same for the components' tracer (kg m⁻² s⁻¹ per unit mixing ratio) | 1e-12 |

A closure check measures its budget ([`with_budgets`](@ref)): enabling
it turns the budget on, as selecting its `budget_*` output does, and has
the same requirements.
"""
const CORE_CHECKS = ("finite", "energy_closure", "water_closure", "momentum_closure", "tracer_closure")

"The budget quantities, units, default tolerance and output variable of each closure check."
const CLOSURE_CHECKS = Dict(
    "energy_closure"   => (qs = (:energy,), units = "W m-2", tolerance = 1e-6, output = "budget_energy"),
    "water_closure"    => (qs = (:water,), units = "kg m-2 s-1", tolerance = 1e-12, output = "budget_water"),
    "momentum_closure" => (qs = (:momentum_x, :momentum_y), units = "kg m-1 s-2", tolerance = 1e-9,
                           output = "budget_momentum"),
    "tracer_closure"   => (qs = (:tracer,), units = "kg m-2 s-1", tolerance = 1e-12, output = "budget_tracer"))

"""
    check_names(::Type{C}) -> Tuple{Vararg{String}}

The names of the checks a component of type `C` can publish: the
configuration's `[checks]` section has one entry per name (and the core's,
[`CORE_CHECKS`](@ref)), and [`checks`](@ref) of the built component
returns checks of these names. Default: none.
"""
check_names(::Type{<:AbstractComponent}) = ()

"""
    check_problems(names, types_opts; moisture, theta0, core, published = ()) -> Vector{Pair{String,String}}

The problems of the checks `names` that `[checks]` enables, for a domain
whose enabled components are the pairs `types_opts` (`C => opts`): `name => message` for a
check no enabled component publishes ([`check_names`](@ref)), and for a
closure check whose budget cannot be measured (the requirements of its
`budget_*` output, [`output_problems`](@ref)). `published` names the
output variables published outside `types` (the model's couplings).
"""
function check_problems(names, types_opts; moisture::Bool, theta0::Real, core::Symbol, published = ())
    probs = Pair{String,String}[]
    own = Dict{String,Any}()
    for (C, _) in types_opts, n in check_names(C)
        own[n] = C
    end
    for n in names
        if haskey(CLOSURE_CHECKS, n)
            out = CLOSURE_CHECKS[n].output
            for p in output_problems([out], types_opts; moisture = moisture, theta0 = theta0, core = core,
                                     published = published)
                push!(probs, n => replace(p, out => n; count = 1))
            end
        elseif !(n in CORE_CHECKS) && !haskey(own, n)
            push!(probs, n => "no enabled component publishes the check $n")
        end
    end
    return probs
end

"The budget quantities the checks `names` measure."
check_budgets(names) = Symbol[q for n in names if haskey(CLOSURE_CHECKS, n) for q in CLOSURE_CHECKS[n].qs]

function _closure_check(d::Domain, name::String)
    spec = CLOSURE_CHECKS[name]
    function evaluate(_)
        b = d.budgets
        b === nothing && return NaN, Dict{String,Any}("error" => "the domain measures no budget")
        worst = 0.0
        info = Dict{String,Any}()
        for q in spec.qs
            r = stage_budget(b.budgets.measured[q])
            r.elapsed > 0 || return NaN, Dict{String,Any}("error" => "no step was measured")
            un = get(r.terms, (:core, :unattributed), 0.0) + get(r.terms, (:output, :between_steps), 0.0)
            v = (abs(r.closure) + abs(un)) / r.elapsed
            sfx = length(spec.qs) == 1 ? "" : "_" * last(split(string(q), '_'))
            info["closure" * sfx] = r.closure / r.elapsed
            info["unattributed" * sfx] = un / r.elapsed
            info["storage" * sfx] = r.storage / r.elapsed
            info["elapsed"] = r.elapsed
            worst = max(worst, v)
        end
        return worst, info
    end
    return Check(name; doc = "closure of the $(join(spec.qs, " and ")) budget of the air and the change no " *
                             "stage entry booked, per second of the run", units = spec.units,
                 tolerance = spec.tolerance, evaluate = evaluate)
end

function _finite_check(d::Domain)
    function evaluate(_)
        bad = nonfinite_fields(d.state)
        info = Dict{String,Any}("fields" => String[string(f.name) for f in bad])
        return Float64(sum(f -> f.count, bad; init = 0)), info
    end
    return Check("finite"; doc = "non-finite interior values of the prognostic fields (the sanity check: a blow-up)",
                 tolerance = 0.0, evaluate = evaluate)
end

"""
    core_checks(d::Domain) -> Vector{Check}

The core's checks of domain `d` ([`CORE_CHECKS`](@ref)): the sanity
check `finite`, and the closure check of every budget `d` measures.
"""
function core_checks(d::Domain)
    cs = Check[_finite_check(d)]
    b = d.budgets
    for n in CORE_CHECKS
        haskey(CLOSURE_CHECKS, n) || continue
        b !== nothing && all(q -> q in b.quantities, CLOSURE_CHECKS[n].qs) && push!(cs, _closure_check(d, n))
    end
    return cs
end

"""
    domain_checks(d::Domain) -> Vector{Check}

The checks domain `d` publishes: the core's ([`core_checks`](@ref)) and
every component's ([`checks`](@ref)).
"""
domain_checks(d::Domain) = vcat(core_checks(d), Check[c for comp in d.components for c in checks(comp)])

"""
    reference_checks(d::Domain, ref, entries) -> Vector{Check}

The reference comparisons of `[checks.reference]`: for every entry
`quantity => {variable, window, …}` of `entries`, a
[`reference_check`](@ref Aeolus.Checks.reference_check) of the domain's
output-catalogue variable `variable` (default `quantity`; a `:scalar`
variable against a scalar reference, a `:z` profile against a profile)
against `ref[quantity]` (`Checks.read_reference`). Throws a
`ConfigError` (key `checks.reference.<quantity>`) for a variable or a
quantity that does not fit.
"""
function reference_checks(d::Domain, ref::AbstractDict, entries::AbstractDict)
    cat = Dict(v.name => v for v in domain_catalogue(d))
    z = last(cell_centres(grid(d)))
    out = Check[]
    probs = ConfigProblem[]
    for q in sort!(collect(keys(entries)))
        e = entries[q]
        key = "checks.reference.$q"
        var = get(e, "variable", q)
        v = get(cat, var, nothing)
        r = get(ref, q, nothing)
        if v === nothing || r === nothing
            push!(probs, ConfigProblem(key, "", v === nothing ? "$var is not in the domain's output catalogue" :
                                                                "the reference file has no quantity $q"))
            continue
        end
        ok = r isa ReferenceProfile ? v.dims === :z : (v.dims === :scalar && !v.group)
        if !ok
            push!(probs, ConfigProblem(key, "", "$var ($(v.dims)$(v.group ? ", a group" : "")) cannot be compared with " *
                                                (r isa ReferenceProfile ? "a profile (a :z variable)" : "a scalar (a :scalar variable)")))
            continue
        end
        w = get(e, "window", nothing)
        push!(out, reference_check(q, r, () -> v.getter(d); variable = var, z = z, units = v.units,
                                   window = w === nothing ? nothing : (Float64(w[1]), Float64(w[2]))))
    end
    isempty(probs) || throw(ConfigError(probs))
    return out
end
