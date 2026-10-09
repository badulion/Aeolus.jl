# The output catalogue of a domain (ONE-c, #452): the core's variables
# and every component's `outputs`, from which `[output] variables`
# selects; `Aeolus.Output` writes them.

"""
    CORE_OUTPUTS

The names in the core's output catalogue ([`core_outputs`](@ref)):

| name | dims | |
| :-- | :-- | :-- |
| `u`, `v`, `w`, `p`, `theta`, `qv` | 3-D | the prognostic fields (`qv` with moisture), at their own staggered positions |
| `u_profile`, `v_profile`, `theta_profile`, `qv_profile` | z | horizontal means per level |
| `u_mean`, `v_mean`, `w_mean`, `theta_mean`, `qv_mean` | scalar | domain means (time series) |
| `budget_energy`, `budget_water`, `budget_momentum`, `budget_tracer` | scalar group | the measured budgets ([`with_budgets`](@ref)): cumulative storage, every attributed term and the closure (time series) |

`qv*` need moisture; `budget_water` needs moisture, `budget_tracer` a
component that carries a tracer, `budget_energy` on the Boussinesq core
`θ₀ > 0`.
"""
const CORE_OUTPUTS = ("u", "v", "w", "p", "theta", "qv",
                      "u_profile", "v_profile", "theta_profile", "qv_profile",
                      "u_mean", "v_mean", "w_mean", "theta_mean", "qv_mean",
                      "budget_energy", "budget_water", "budget_momentum", "budget_tracer")

"""
    DEFAULT_OUTPUTS

The variables a run writes when `[output] variables` is not given: `u`,
`v`, `w`, `theta` and the time series of the domain means of u, v, w and
θ (the design's default set).
"""
const DEFAULT_OUTPUTS = ("u", "v", "w", "theta", "u_mean", "v_mean", "w_mean", "theta_mean")

"The budget quantities each `budget_*` output variable turns on."
const BUDGET_OUTPUTS = Dict("budget_energy" => (:energy,), "budget_water" => (:water,),
                            "budget_momentum" => (:momentum_x, :momentum_y), "budget_tracer" => (:tracer,))

"""
    output_names(::Type{C}, opts) -> Tuple{Vararg{String}}

The names in the output catalogue that a component of type `C` with the
resolved options `opts` publishes: [`outputs`](@ref) of the built
component has exactly these (the contract tests check it). The
configuration is validated against them before anything is built, so
selecting a variable whose component is off is an error. Default: none.
"""
output_names(::Type{<:AbstractComponent}, opts) = ()

# plane sums per level of a field's interior, in the canonical order
_level_sums(f) = plane_partials!(zeros(f.mesh.Nz), identity, interior(f.mesh), f.data)

function _profile(f)
    m = f.mesh
    return _level_sums(f) ./ (m.Nx * m.Ny)
end

# the domain mean: level means weighted by the level thickness
function _domain_mean(f)
    m = f.mesh
    dz = hasproperty(m, :dzc) ? [m.dzc[m.Hz+k] for k in 1:m.Nz] : fill(Float64(m.dz), m.Nz)
    S = _level_sums(f)
    acc = 0.0
    @inbounds for k in eachindex(S)
        acc += S[k] * dz[k]
    end
    return acc / (m.Nx * m.Ny * sum(dz))
end

# the cumulative terms of a measured budget, by "<stage>__<name>", with the
# storage and the closure (one StageBudget, or the sum of two: momentum x, y
# are written apart)
function _budget_group(sb::StageBudget)
    b = stage_budget(sb)
    g = Dict{String,Float64}("storage" => b.storage, "closure" => b.closure)
    for ((st, n), v) in b.terms
        g[string(st, "__", n)] = v
    end
    return g
end

"""
    core_outputs(d::Domain) -> Vector{OutputVariable}

The core's output catalogue of domain `d` ([`CORE_OUTPUTS`](@ref)): the
variables its state has (the moisture ones only with moisture, the budget
ones only for the quantities its budgets measure).
"""
function core_outputs(d::Domain)
    sim = d.state
    moist = hasproperty(sim, :has_moisture) && sim.has_moisture
    iv(f) = interior_view(f)
    vs = OutputVariable[
        OutputVariable("u", "m s-1", "x wind (at the x faces)", :xyz, d -> iv(d.state.u)),
        OutputVariable("v", "m s-1", "y wind (at the y faces)", :xyz, d -> iv(d.state.v)),
        OutputVariable("w", "m s-1", "upward wind (at the z faces)", :xyz, d -> iv(d.state.w)),
        OutputVariable("p", "m2 s-2", "kinematic pressure", :xyz, d -> iv(d.state.p)),
        OutputVariable("theta", "K", "potential temperature", :xyz, d -> iv(d.state.θ)),
        OutputVariable("u_profile", "m s-1", "horizontal mean of u", :z, d -> _profile(d.state.u)),
        OutputVariable("v_profile", "m s-1", "horizontal mean of v", :z, d -> _profile(d.state.v)),
        OutputVariable("theta_profile", "K", "horizontal mean of theta", :z, d -> _profile(d.state.θ)),
        OutputVariable("u_mean", "m s-1", "domain mean of u", :scalar, d -> _domain_mean(d.state.u)),
        OutputVariable("v_mean", "m s-1", "domain mean of v", :scalar, d -> _domain_mean(d.state.v)),
        OutputVariable("w_mean", "m s-1", "domain mean of w", :scalar, d -> _domain_mean(d.state.w)),
        OutputVariable("theta_mean", "K", "domain mean of theta", :scalar, d -> _domain_mean(d.state.θ))]
    if moist
        append!(vs, OutputVariable[
            OutputVariable("qv", "kg kg-1", "water vapour mixing ratio", :xyz, d -> iv(d.state.qv)),
            OutputVariable("qv_profile", "kg kg-1", "horizontal mean of qv", :z, d -> _profile(d.state.qv)),
            OutputVariable("qv_mean", "kg kg-1", "domain mean of qv", :scalar, d -> _domain_mean(d.state.qv))])
    end
    b = d.budgets
    if b !== nothing
        units = Dict(:energy => "J m-2", :water => "kg m-2", :tracer => "kg m-2", :momentum_x => "kg m-1 s-1",
                     :momentum_y => "kg m-1 s-1")
        for (name, qs) in BUDGET_OUTPUTS
            all(q -> q in b.quantities, qs) || continue
            if length(qs) == 1
                q = only(qs)
                push!(vs, OutputVariable(name, units[q], "$(q) budget of the air, cumulative since the start",
                                         :scalar, d -> _budget_group(d.budgets.budgets.measured[q]); group = true))
            else
                push!(vs, OutputVariable(name, units[first(qs)], "momentum budget of the air (x, y), cumulative since the start",
                                         :scalar, d -> merge(
                                             Dict("x__" * k => v for (k, v) in _budget_group(d.budgets.budgets.measured[:momentum_x])),
                                             Dict("y__" * k => v for (k, v) in _budget_group(d.budgets.budgets.measured[:momentum_y])));
                                         group = true))
            end
        end
    end
    return vs
end

"""
    domain_catalogue(d::Domain) -> Vector{OutputVariable}

The output catalogue of domain `d`: the core's ([`core_outputs`](@ref))
and every component's ([`outputs`](@ref)), checked for duplicate names.
"""
domain_catalogue(d::Domain) =
    check_catalogue(vcat(core_outputs(d), OutputVariable[v for c in d.components for v in outputs(c)]))

"""
    output_problems(names, types_opts; moisture, theta0, core, published = ()) -> Vector{String}

The problems of the `[output] variables` selection `names` against the
catalogue of a domain whose enabled components are the pairs
`types_opts` (`C => opts`): a name no catalogue has, a variable whose
component is off (or that needs moisture), and the requirements of the
budget variables (`budget_tracer`: a component publishing `tracer`).
`published` names what components built outside `types_opts` publish
(the model's couplings).
"""
function output_problems(names, types_opts; moisture::Bool, theta0::Real, core::Symbol, published = ())
    probs = String[]
    comp = Dict{String,Any}()
    for (C, opts) in types_opts, n in output_names(C, opts)
        comp[n] = C
    end
    for n in published
        comp[n] = nothing
    end
    for n in names
        if n in CORE_OUTPUTS
            startswith(n, "qv") && !moisture && push!(probs, "$n is written only with dynamics.moisture = true")
            n == "budget_water" && !moisture && push!(probs, "budget_water needs dynamics.moisture = true")
            n == "budget_tracer" && !haskey(comp, "tracer") &&
                push!(probs, "budget_tracer needs a component that carries a tracer (one that publishes \"tracer\")")
            n == "budget_energy" && core === :boussinesq && theta0 <= 0 &&
                push!(probs, "budget_energy needs dynamics.theta0 > 0 (the Boussinesq reference density p_ref / (R_d theta0))")
        elseif !haskey(comp, n)
            push!(probs, "$n: no enabled component publishes it (the core's: $(join(CORE_OUTPUTS, ", ")))")
        end
    end
    return probs
end
