# Nesting by domains (ONE-f, #455): a config with several [[domain]]
# blocks is one model. The first domain is the root; every later domain
# names its `parent` (an earlier one), stands on the parent's ground at
# `origin`, refines the parent's spacing by `refinement`, and feeds back
# two-way unless it says `feedback = "one_way"`. Its lateral faces (and its
# top, below the parent's) are nest faces fed by the parent; the coupling
# is Components.nested_model (Nesting.NestHierarchy).

const _NEST_CHECKS_DOC = """
the first domain is the root (no `parent`); every later domain names an earlier one as its `parent`
a root has no `origin` or `refinement`, and `feedback` stays two_way (it has no parent); a nested domain gives both
a nested model takes a fixed `time.dt` (the children subcycle max(refinement) substeps per parent step) and measures no budgets (no `budget_*` output, no closure check)
every domain of a nested model runs the boussinesq core on a uniform grid between z walls
a nested domain lies inside its parent (origin ≥ 0, origin + grid.extent within the parent's extent, its height at most the parent's), with `grid.size` = `grid.extent` / parent spacing × `refinement`
a nested domain keeps its parent's `[boundaries]` and `dynamics.moisture`, has no `dynamic_driver` and none of the couplings ([forcing], [surface], [microphysics], [tracer], [particles], [obstacles])"""

"Whether `cfg` is a nested model (several domains)."
_nested(cfg) = length(config_domains(cfg)) > 1

# The checks of one domain's view that concern nesting.
function _nest_domain_checks(cfg)
    e = ConfigError[]
    dom = cfg.domain
    if dom.parent === nothing
        dom.origin === nothing ||
            push!(e, ConfigError("domain.origin", "the root domain (no parent) has no origin in a parent"))
        dom.refinement === nothing ||
            push!(e, ConfigError("domain.refinement", "the root domain (no parent) has no refinement"))
        dom.feedback === :two_way ||
            push!(e, ConfigError("domain.feedback", "the root domain has no parent to feed back to"))
        return e
    end
    dom.origin === nothing &&
        push!(e, ConfigError("domain.origin", "a nested domain needs its origin (x, y) in its parent"))
    dom.refinement === nothing &&
        push!(e, ConfigError("domain.refinement", "a nested domain needs its refinement (rx, ry, rz)"))
    dom.dynamic_driver === nothing ||
        push!(e, ConfigError("domain.dynamic_driver", "a nested domain is driven by its parent: its faces are nest faces; leave dynamic_driver none"))
    _any_coupling(cfg) &&
        push!(e, ConfigError("domain.parent", "the couplings ([forcing], [surface], [microphysics], [tracer], [particles], " *
                                              "[obstacles]) are not nested: their fields have no nest faces; switch them off in a nested domain"))
    return e
end

# The checks of the [[domain]] graph (a Config.ModelCheck: once, on all domains).
function _nest_checks(cfg)
    e = ConfigError[]
    ds = config_domains(cfg)
    n = length(ds)
    key(i, k) = n == 1 ? k : "domain[$i].$k"
    ids = [d.domain.id for d in ds]
    parent_of = zeros(Int, n)
    for (i, d) in enumerate(ds)
        p = d.domain.parent
        if i == 1
            p === nothing || push!(e, ConfigError(key(1, "domain.parent"),
                                                  "the first domain is the root of the nest: it has no parent (got \"$p\")"))
        elseif p === nothing
            push!(e, ConfigError(key(i, "domain.parent"), "every domain after the first is nested: name its parent " *
                                                          "(one of $(join(repr.(ids[1:i-1]), ", ")))"))
        else
            j = findfirst(==(p), ids[1:i-1])
            j === nothing ? push!(e, ConfigError(key(i, "domain.parent"), "\"$p\" is not an earlier domain " *
                                                                          "(the earlier ones: $(join(repr.(ids[1:i-1]), ", ")))")) :
                            (parent_of[i] = j)
        end
    end
    n == 1 && return e
    cfg.time.dt === :adaptive &&
        push!(e, ConfigError("time.dt", "a nested model takes a fixed dt: each child subcycles max(refinement) substeps per step of its parent"))
    for v in cfg.output.variables
        haskey(Components.BUDGET_OUTPUTS, v) &&
            push!(e, ConfigError("output.variables", "$v: a nested model measures no budgets (the two-way feedback is outside every stage)"))
    end
    for c in Components.check_budgets(_published_enabled(cfg)) |> unique
        push!(e, ConfigError("checks", "the closure checks measure budgets, which a nested model does not (:$c)"))
    end
    for (i, d) in enumerate(ds)
        d.core.model === :boussinesq ||
            push!(e, ConfigError(key(i, "core.model"), "a nested model runs the boussinesq core (the anelastic children would share their parent's reference profile)"))
        d.grid.stretching === :uniform ||
            push!(e, ConfigError(key(i, "grid.stretching"), "a nested model takes uniform grids"))
        d.boundaries.z === :periodic &&
            push!(e, ConfigError(key(i, "boundaries.z"), "a nested model has z walls (a child stands on its parent's ground)"))
        j = parent_of[i]
        (j == 0 || d.domain.origin === nothing || d.domain.refinement === nothing) && continue
        p = ds[j]; g = d.grid; pg = p.grid
        o = d.domain.origin; r = d.domain.refinement
        Δp = pg.extent ./ pg.size
        want = ntuple(a -> g.extent[a] / Δp[a] * r[a], 3)
        if any(a -> abs(want[a] - round(want[a])) > 1e-9 * max(1.0, want[a]), 1:3)
            push!(e, ConfigError(key(i, "grid.extent"), "$(g.extent) is not a whole number of child cells " *
                                                        "(parent spacing $Δp / refinement $r)"))
        elseif Tuple(round.(Int, want)) != g.size
            push!(e, ConfigError(key(i, "grid.size"), "$(g.size): the extent $(g.extent) at the parent spacing $Δp " *
                                                      "refined by $r is $(Tuple(round.(Int, want))) cells"))
        end
        for a in 1:2
            (o[a] >= 0 && o[a] + g.extent[a] <= pg.extent[a] * (1 + 1e-12)) ||
                push!(e, ConfigError(key(i, "domain.origin"), "the domain spans $(("x", "y")[a]) = $(o[a]) … " *
                                                              "$(o[a] + g.extent[a]) m, outside its parent's 0 … $(pg.extent[a]) m"))
        end
        g.extent[3] <= pg.extent[3] * (1 + 1e-12) ||
            push!(e, ConfigError(key(i, "grid.extent"), "the domain is $(g.extent[3]) m high, its parent $(pg.extent[3]) m"))
        for ax in ("x", "y", "z")
            d.boundaries[ax] === p.boundaries[ax] ||
                push!(e, ConfigError(key(i, "boundaries.$ax"), "a nested domain keeps its parent's faces ($(p.boundaries[ax])): " *
                                                              "its lateral faces are nest faces, its ground the parent's"))
        end
        d.dynamics.moisture == p.dynamics.moisture ||
            push!(e, ConfigError(key(i, "dynamics.moisture"), "a nested domain carries qv exactly when its parent does"))
    end
    return e
end

# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

# The NestedGrid of the nested domain `v` (a domain view) in the parent mesh
# `pm`: the child stands on the parent's ground.
function _nest_grid(v, pm)
    g = v.grid; o = v.domain.origin
    ext = ((pm.x0 + o[1], pm.x0 + o[1] + g.extent[1]), (pm.y0 + o[2], pm.y0 + o[2] + g.extent[2]),
           (pm.z0, pm.z0 + g.extent[3]))
    nest = nested_grid(pm, ext, v.domain.refinement; halo = Tuple(g.halo))
    m = nest.child_mesh
    (m.Nx, m.Ny, m.Nz) == Tuple(g.size) ||
        throw(ConfigError("grid.size", "the nest's child mesh has $((m.Nx, m.Ny, m.Nz)) cells, the domain $(g.size)"))
    return nest
end

# The faces of a nested domain: nest faces at the sides (periodic along an
# axis it spans entirely, if its parent is periodic there), its parent's
# wall below, a nest face on top (its parent's wall when it reaches the
# parent's top).
function _nest_child_bcs(v, pv, nest)
    m = nest.child_mesh; pm = nest.parent_mesh
    b = pv.boundaries
    full(a) = a == 1 ? abs(m.Lx - pm.Lx) <= 1e-9 * pm.Lx : abs(m.Ly - pm.Ly) <= 1e-9 * pm.Ly
    side(a, k) = k === :periodic && full(a) ? :periodic : :nest
    wall = b.z === :noslip ? :noslip : :wall
    top = m.Lz < pm.Lz * (1 - 1e-12) ? :nest : wall
    return nest_bcs(m, side(1, b.x), side(2, b.y); bottom = wall, top = top, moisture = v.dynamics.moisture)
end

# The Model of a nested config: every domain built as a single domain would
# be (its own [initial], components and couplings), the children on their
# nests' child meshes with nest faces, coupled by Components.nested_model.
function _build_nested(cfg)
    ds = config_domains(cfg)
    n = length(ds)
    ids = [d.domain.id for d in ds]
    doms = Any[]
    links = Components.NestLink[]
    for i in 1:n
        v = domain_view(cfg, i)
        if i == 1
            push!(doms, _build_domain(v))
            continue
        end
        j = findfirst(==(v.domain.parent), ids)
        nest = _nest_grid(v, Components.grid(doms[j]))
        bcs = _nest_child_bcs(v, domain_view(cfg, j), nest)
        push!(doms, _build_domain(v; mesh = nest.child_mesh, bcs = bcs))
        push!(links, Components.NestLink(i, j, nest; feedback = v.domain.feedback))
    end
    return try
        Components.nested_model(Tuple(doms), links)
    catch err
        err isa ArgumentError || rethrow()
        throw(ConfigError("domain", err.msg))
    end
end

# The published checks of a nested model: each check name once, its value
# the worst over the domains that publish it, with each domain's value
# (and its information) in the result's info under the domain's id.
function _nest_checks_merged(doms)
    per = [Components.domain_checks(d) for d in doms]
    names = unique!([c.name for cs in per for c in cs])
    out = Checks.Check[]
    for nm in names
        group = [(d.id, c) for (d, cs) in zip(doms, per) for c in cs if c.name == nm]
        all(g -> last(g).acc === nothing, group) ||
            throw(ArgumentError("the check $nm accumulates over the run; a nested model evaluates such checks on one domain only"))
        c1 = last(first(group))
        evaluate = function (_)
            worst = -Inf
            info = Dict{String,Any}()
            for (id, c) in group
                v, i = c.evaluate(c.acc)
                worst = isnan(v) || isnan(worst) ? NaN : max(worst, Float64(v))
                info[id] = merge(Dict{String,Any}("value" => Float64(v)), Dict{String,Any}(i))
            end
            return worst, info
        end
        push!(out, Checks.Check(nm; doc = c1.doc * " (the worst over the domains)", units = c1.units,
                                tolerance = c1.tolerance, evaluate = evaluate))
    end
    return out
end
