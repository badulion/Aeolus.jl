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
a nested model steps every domain with `time.dt`: a fixed dt (each child takes max(refinement) substeps per step of its parent) or adaptive (the root picks its step, each child the smallest whole number of equal substeps that keeps it stable; a failed step retries the window); with `[nesting] budget_exchange = "off"` it measures no budgets (no `budget_*` output, no closure check); with `"exchange_term"` (the default) each domain closes its own, with one nest-exchange term per quantity
every domain of a nested model runs the boussinesq or the anelastic core (mixed freely, two-way included: a conservative density conversion in the boundary data and the feedback) on a uniform grid between z walls; an anelastic nested domain needs [timestep] scheme = "euler" (its SSP-RK3 step takes no open faces), and one nested in an anelastic parent the same dynamics.theta0 and g (one reference profile)
a nested domain lies inside its parent (origin ≥ 0, origin + grid.extent within the parent's extent, its height at most the parent's), with `grid.size` = `grid.extent` / parent spacing × `refinement`
a nested domain may enable `[inflow_turbulence]` (synthetic turbulence on its lateral nest faces, with its parent's resolved statistics: `statistics = "parent_resolved"`, no `length_scale`); the root may not
every field the nest exchanges (`qv`, the Kessler `ql`, `qr`, the tracer) exists in a nested domain exactly when it exists in its parent (the PALM rule; a nested domain could carry an extra field only with its own dynamic driver giving its boundary values, and has none yet)
a nested domain keeps its parent's `[boundaries]`, has no `dynamic_driver` and none of the couplings ([forcing], [surface], [microphysics], [tracer], [particles], [obstacles])"""

"Whether `cfg` is a nested model (several domains)."
_nested(cfg) = length(config_domains(cfg)) > 1

# The checks of one domain's view that concern nesting.
function _nest_domain_checks(cfg)
    e = ConfigError[]
    dom = cfg.domain
    it = cfg.inflow_turbulence
    if it.enabled
        dom.parent === nothing &&
            push!(e, ConfigError("inflow_turbulence.enabled", "synthetic inflow turbulence perturbs a nested domain's lateral nest faces; the root has none"))
        it.statistics === :parent_resolved ||
            push!(e, ConfigError("inflow_turbulence.statistics", "a nested domain's inflow statistics are its parent's resolved ones (parent_resolved)"))
        isnan(it.length_scale) ||
            push!(e, ConfigError("inflow_turbulence.length_scale", "belongs to the parametrized statistics; a nested domain measures its lengths on its parent"))
    end
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
    # (an urban site reads its initial profile from its sounding driver)
    dom.dynamic_driver === nothing || Urban.urban_enabled(cfg) ||
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
    if ds[1].nesting.budget_exchange === :off
        for v in cfg.output.variables
            haskey(Components.BUDGET_OUTPUTS, v) &&
                push!(e, ConfigError("output.variables", "$v: with [nesting] budget_exchange = \"off\" a nested model measures no budgets"))
        end
        for c in Components.check_budgets(_published_enabled(cfg)) |> unique
            push!(e, ConfigError("checks", "the closure checks measure budgets, which a nested model with [nesting] budget_exchange = \"off\" does not (:$c)"))
        end
    end
    # an urban site (NEST-p3 #483) is a child of a domain of the model's core,
    # with no children of its own; its parent may have a stretched grid
    urban = [Urban.urban_enabled(domain_view(cfg, i)) for i in 1:n]
    for (i, d) in enumerate(ds)
        if urban[i]
            i == 1 && push!(e, ConfigError(key(1, "urban.enabled"), "an urban site in a nest is a child of a domain of the model's core, not the root"))
            j = parent_of[i]
            j > 0 && urban[j] && push!(e, ConfigError(key(i, "domain.parent"), "an urban site nests in a domain of the model's core, not in the urban site \"$(ids[j])\""))
            (d.nesting.enabled && d.nesting.parent_driver === :aeolus) ||
                push!(e, ConfigError(key(i, "nesting.enabled"), "an urban site in a nest takes its faces from its parent: [nesting] enabled = true, parent_driver = \"aeolus\""))
            if j > 0 && d.domain.origin !== nothing
                po = ds[j].grid.origin
                all(a -> isapprox(po[a], -d.domain.origin[a]; atol = 1e-9 * max(1.0, abs(po[a]))), 1:2) ||
                    push!(e, ConfigError(key(j, "grid.origin"), "the urban site \"$(ids[i])\" stands at its own corner (0, 0) at origin " *
                                                               "$(d.domain.origin) in \"$(ids[j])\": its parent's [grid] origin is $(Tuple(-x for x in d.domain.origin)), not $(Tuple(po))"))
            end
        end
        if !urban[i]
            d.core.model in (:boussinesq, :anelastic) ||
                push!(e, ConfigError(key(i, "core.model"), "a nested model runs the boussinesq or the anelastic core"))
            i > 1 && d.core.model === :anelastic && d.timestep.scheme === :rk3 &&
                push!(e, ConfigError(key(i, "core.model"), "an anelastic nested domain has open (nest) faces, which the anelastic SSP-RK3 step does not take: step the nest with [timestep] scheme = \"euler\""))
            kids = [k for k in 1:n if parent_of[k] == i]
            d.grid.stretching === :uniform || (!isempty(kids) && all(k -> urban[k], kids)) ||
                push!(e, ConfigError(key(i, "grid.stretching"), "a nested model takes uniform grids (a stretched grid only in the parent of urban sites)"))
            d.boundaries.z === :periodic &&
                push!(e, ConfigError(key(i, "boundaries.z"), "a nested model has z walls (a child stands on its parent's ground)"))
        end
        j = parent_of[i]
        (j == 0 || d.domain.origin === nothing || d.domain.refinement === nothing) && continue
        p = ds[j]; g = d.grid; pg = p.grid
        o = d.domain.origin; r = d.domain.refinement
        Δp = pg.extent ./ pg.size
        # (a stretched parent: the spacing of its lowest levels, where the site stands)
        pg.stretching === :uniform || (Δp = (Δp[1], Δp[2], pg.dz_surface))
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
        urban[i] && continue           # the site's faces and field set are its :nesting component's
        # two anelastic domains share one reference profile (their transfers use its mass measure)
        (d.core.model === :anelastic && p.core.model === :anelastic &&
         (d.dynamics.theta0 != p.dynamics.theta0 || d.dynamics.g != p.dynamics.g)) &&
            push!(e, ConfigError(key(i, "dynamics.theta0"), "an anelastic domain nested in an anelastic parent shares its reference profile: " *
                                                           "the same dynamics.theta0 and dynamics.g (parent $(p.dynamics.theta0), $(p.dynamics.g))"))
        for ax in ("x", "y", "z")
            d.boundaries[ax] === p.boundaries[ax] ||
                push!(e, ConfigError(key(i, "boundaries.$ax"), "a nested domain keeps its parent's faces ($(p.boundaries[ax])): " *
                                                              "its lateral faces are nest faces, its ground the parent's"))
        end
        # the PALM rule (Hellsten et al. 2021, §3.1), with the driver exception
        fc = _exchanged_fields(domain_view(cfg, i)); fp = _exchanged_fields(domain_view(cfg, j))
        for (f, k) in fp
            haskey(fc, f) ||
                push!(e, ConfigError(key(i, k), "domain \"$(ids[i])\" carries no $f, its parent \"$(ids[j])\" does: every field " *
                                                "the nest exchanges exists in the parent and the child (the PALM rule)"))
        end
        for (f, k) in fc
            haskey(fp, f) ||
                push!(e, ConfigError(key(i, k), "domain \"$(ids[i])\" carries $f, its parent \"$(ids[j])\" does not: a nested domain " *
                                                "carries a field its parent lacks only when its own dynamic driver gives that " *
                                                "field's boundary values (a nested domain has no dynamic driver yet)"))
        end
    end
    return e
end

"""
    _exchanged_fields(v) -> Dict{String,String}

The fields the nest exchanges besides `u, v, w, θ` for the domain view
`v`, each with the config key that switches it on: `qv`
(`dynamics.moisture`), the Kessler water `ql`, `qr` (`[microphysics]`)
and the passive tracer (`[tracer]`). The nest checks them against the
parent's (the PALM rule).
"""
function _exchanged_fields(v)
    out = Dict{String,String}()
    v.dynamics.moisture && (out["qv"] = "dynamics.moisture")
    if _microphysics(v) !== :none
        out["ql"] = "microphysics.model"; out["qr"] = "microphysics.model"
    end
    v.tracer.enabled && (out["tracer"] = "tracer.enabled")
    return out
end

"""
    NEST_SMOOTH_RATIO

The horizontal refinement from which a nested domain's parent counts as
much smoother than the domain (3): the parent resolves none of the
eddies between its spacing and three of the child's, so without synthetic
inflow turbulence the child's inflow enters laminar and needs a long fetch
to develop them (Kadasch et al. 2021, §2.4; Hellsten et al. 2021, §3.3).
"""
const NEST_SMOOTH_RATIO = 3

"""
    nest_handover_warnings(cfg) -> Vector{String}

The subgrid handover warnings of a nested config (NEST-p2, #482): a
nested domain with a lateral nest face that refines its parent
horizontally by [`NEST_SMOOTH_RATIO`](@ref) or more and has no
`[inflow_turbulence]`. The build logs each with `@warn`. (The nest passes
resolved fields only; the SGS model may differ per domain: a TKE field on
one side only is not exchanged — the child's `e` takes a zero-gradient
value at its nest faces, a parent's `e` under the child is not fed back.)
"""
function nest_handover_warnings(cfg)
    ds = config_domains(cfg)
    out = String[]
    for i in 2:length(ds)
        d = domain_view(cfg, i)
        r = d.domain.refinement
        r === nothing && continue
        pv = domain_view(cfg, findfirst(x -> x.domain.id == d.domain.parent, ds))
        # a lateral nest face: an axis the domain does not span (or one its parent does not wrap)
        lateral = any(a -> d.grid.extent[a] < pv.grid.extent[a] * (1 - 1e-12) ||
                           pv.boundaries[("x", "y")[a]] !== :periodic, 1:2)
        rh = max(r[1], r[2])
        if lateral && rh >= NEST_SMOOTH_RATIO && !d.inflow_turbulence.enabled
            push!(out, "domain \"$(d.domain.id)\" refines its parent \"$(d.domain.parent)\" by $rh horizontally " *
                       "(its sgs model $(d.sgs.model), the parent's $(pv.sgs.model)) " *
                       "and has no [inflow_turbulence]: the parent is much smoother, and the inflow enters without " *
                       "the eddies between the two spacings (enable [inflow_turbulence] in its [[domain]] block)")
        end
    end
    return out
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
    foreach(w -> @warn(w), nest_handover_warnings(cfg))
    ds = config_domains(cfg)
    n = length(ds)
    ids = [d.domain.id for d in ds]
    urban = _urban_nest(cfg)
    doms = Any[]
    links = Components.NestLink[]
    h = nothing
    for i in 1:n
        v = domain_view(cfg, i)
        if i == 1
            # the root of an urban site's nest steps adaptively ([time] cfl,
            # dt_max): the site's own [time] dt is its step
            push!(doms, urban ? _build_domain(v; cfl = _root_cfl(v)) : _build_domain(v))
            # an urban site joins the nest while it is built (NEST-p3 #483):
            # the hierarchy exists first, and every domain is linked in order
            urban && (h = Components.nest_hierarchy(doms[1]))
            continue
        end
        j = findfirst(==(v.domain.parent), ids)
        if Urban.urban_enabled(v)
            nx = Urban.UrbanNest(doms[j], h, Symbol(v.domain.id), Symbol(ids[j]), Tuple(Float64.(v.domain.origin)),
                                 Tuple(Int.(v.domain.refinement)), v.domain.feedback)
            child = only(Urban.build_model(v; nest = nx).domains)
            Components.set_pipeline_domain!(h.advance[end], child)
            push!(doms, child)
            push!(links, Components.NestLink(i, j, h.links[end].nest; feedback = v.domain.feedback, prebuilt = true))
            continue
        end
        nest = _nest_grid(v, Components.grid(doms[j]))
        bcs = _nest_child_bcs(v, domain_view(cfg, j), nest)
        push!(doms, _build_domain(v; mesh = nest.child_mesh, bcs = bcs))
        it = v.inflow_turbulence
        inflow = it.enabled ? (generator = it.generator, u_min = it.u_min, seed = it.seed,
                               statistics_interval = it.statistics_interval, length_interval = it.length_interval) :
                 nothing
        l = Components.NestLink(i, j, nest; feedback = v.domain.feedback, inflow = inflow, prebuilt = urban)
        if urban
            try
                Components.link_domain!(h, Tuple(doms), l)
            catch err
                err isa ArgumentError || rethrow()
                throw(ConfigError("domain", err.msg))
            end
        end
        push!(links, l)
    end
    model = try
        Components.nested_model(Tuple(doms), links; budget_exchange = ds[1].nesting.budget_exchange, hierarchy = h)
    catch err
        err isa ArgumentError || rethrow()
        throw(ConfigError("domain", err.msg))
    end
    # the root's first step grows from the site's step, as the site's own does
    urban && (model.nest.dt_prev[1] = Float64(cfg.time.dt))
    return model
end

"""
    _urban_nest(cfg) -> Bool

Whether the nested config `cfg` has an urban site among its domains (a
child; NEST-p3 #483): its root then steps adaptively with `[time] cfl`
and `dt_max` while `[time] dt` and `nsteps` are the site's.
"""
_urban_nest(cfg) = _nested(cfg) && any(i -> Urban.urban_enabled(domain_view(cfg, i)), 2:length(config_domains(cfg)))

"The CFL constraints of the root of an urban site's nest: `[time] cfl` and `dt_max`."
_root_cfl(cfg) = CFLConstraints(; C_advective = cfg.time.cfl, dt_max = cfg.time.dt_max)

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
