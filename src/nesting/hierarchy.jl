# Nesting: a tree of nested domains (#328) — several nest levels and
# sibling nests, advanced by recursive Berger–Oliger subcycling.
#
#   * NestHierarchy — the domains (a root and any number of nested
#     children, grandchildren, siblings) and the links between them. Each
#     link is a NestedRun (boundary data, registers, two-way feedback,
#     statistics of that parent–child pair), so every operator, the
#     boundary-data contract (#58) and the time blend are those of the
#     two-domain run; NestedRun's own stepping, extended by a domain's
#     "subtree" (`_HDomain`), is the recursion.
#   * add_nest!     — attach a child to a domain, with the proper-nesting,
#     sibling and two-way checks.
#   * step_nested!  — one root step: every domain subcycles to land on its
#     parent's step ends, the boundary data flowing down and the two-way
#     feedback flowing up level by level.
#
# References: Berger, M. J. & Oliger, J. (1984), J. Comput. Phys. 53,
# 484–512 (recursive subcycling of a grid hierarchy); Berger, M. J. &
# Colella, P. (1989), J. Comput. Phys. 82, 64–84 (refluxing across every
# coarse–fine interface); Hellsten, A. et al. (2021), Geosci. Model Dev.
# 14, 3185–3214, §3.1 (PALM's nest tree: siblings, levels, one-/two-way).

# The register hooks of a domain that is the fine side of one link and the
# coarse side of others, called in turn at every stage.
struct _HookChain{T<:Tuple} <: Function
    hooks::T
end
function (h::_HookChain)(sim, Δt)
    for f in h.hooks
        f(sim, Δt)
    end
    return nothing
end
_chain_hooks(hs) = (hs = filter(!isnothing, hs); isempty(hs) ? nothing :
                    length(hs) == 1 ? hs[1] : _HookChain(Tuple(hs)))

# A domain's subtree: what its steps carry beyond the link that drives them
# (the uplink of a child; the first link of the root).
mutable struct _HDomain
    id::Symbol
    sim::Any
    up::Int                          # uplink index (0: the root)
    links::Vector{Int}               # links where the domain is the coarse side
    hook::Any                        # the composed register hooks of all its links
    fields::Vector{Any}              # the bound scalars of all its links (rollback, acceptance)
    rb::Any                          # its rollback (_Rollback), or nothing before the first link
    # the current step of the domain (filled by _sub_prepare!)
    t0::Float64
    m0::Vector{Float64}
    h::Any                           # the NestHierarchy (for the recursion)
end

"""
    NestHierarchy(root_sim; id = :root, advance = domain_advance(),
                  cfl = CFLConstraints(), sync = :subcycle, max_retries = 3,
                  divergence_tol = Inf, max_velocity = Inf, validate = true)

A tree of nested domains (#328): the root `root_sim` and the children,
grandchildren and siblings attached with [`add_nest!`](@ref), advanced
together one root step per [`step_nested!`](@ref)`(h)`.

Every link is a [`NestedRun`](@ref) between a domain and one of its
children, with its own [`NestedGrid`](@ref) (any refinement, integer,
rational or stretched), boundary data, feedback mode (`:one_way` or
two-way, mixed freely in one tree), sponge and solid mask. The time
stepping is Berger & Oliger's recursion (as in PALM and WRF): a domain
takes a step (adaptive with its own `cfl`, or `dt` for the root), its
children's boundary data are prolongated at both ends of that step, every
child subcycles to land exactly on its end (each with its own adaptive
step, re-evaluated before every substep), recursively, and then the
two-way children feed back. A grandchild's feedback reaches its parent at
the end of each of the parent's substeps, before the parent's own
feedback, so the composite integrals ([`composite_integral`](@ref)`(h, q)`)
of the hierarchy are conserved at every level.

`sync = :shared_min_dt` gives every domain one step of the smallest limit
over all domains instead (PALM's scheme). A rejected step (non-finite
state, solver failure, `max_velocity`; the root also `divergence_tol`)
rolls back that domain only — no descendant has moved within it — and is
redone at half its size; `max_retries` failures raise an error.
Within one step the siblings advance in link order; they read only their
common parent's state at the window ends and write disjoint parent cells,
and the parent is re-projected once after all their feedback, so the
result does not depend on that order.

`h.t` is the common time; `nest_link(h, id)` the link (a `NestedRun`,
with its `stats` and ledgers) whose child is domain `id`; `nest_domain(h, id)`
the simulation.
"""
mutable struct NestHierarchy
    domains::Vector{_HDomain}
    links::Vector{NestedRun}
    coarse::Vector{Int}              # per link: the coarse (parent) domain
    fine::Vector{Int}                # per link: the fine (child) domain
    advance::Vector{DomainAdvance}   # per domain
    cfl::Vector{Any}                 # per domain
    sync::Symbol
    max_retries::Int
    divergence_tol::Float64
    max_velocity::Float64
    validate::Bool
    t::Float64
    dt_prev_root::Union{Nothing,Float64}
    root_steps::Int
end

function NestHierarchy(root::_NestSim; id::Symbol = :root, advance::DomainAdvance = domain_advance(),
                       cfl = CFLConstraints(), sync::Symbol = :subcycle, max_retries::Integer = 3,
                       divergence_tol::Real = Inf, max_velocity::Real = Inf, validate::Bool = true)
    sync in (:subcycle, :shared_min_dt) ||
        throw(ArgumentError("NestHierarchy: sync must be :subcycle or :shared_min_dt, got :$sync"))
    d = _HDomain(id, root, 0, Int[], nothing, Any[], nothing, Float64(root.t), Float64[], nothing)
    h = NestHierarchy([d], NestedRun[], Int[], Int[], DomainAdvance[advance], Any[cfl], sync,
                      Int(max_retries), Float64(divergence_tol), Float64(max_velocity), validate,
                      Float64(root.t), nothing, 0)
    d.h = h
    return h
end

function Base.show(io::IO, h::NestHierarchy)
    print(io, "NestHierarchy(", length(h.domains), " domains, ", length(h.links), " links, t=",
          h.t, ", ", h.root_steps, " root steps)")
end

function _domain_index(h::NestHierarchy, id::Symbol)
    i = findfirst(d -> d.id === id, h.domains)
    i === nothing && throw(ArgumentError("NestHierarchy: no domain :$id (domains: " *
                                         "$(join((d.id for d in h.domains), ", ")))"))
    return i
end
_domain_index(h::NestHierarchy, i::Integer) =
    1 <= i <= length(h.domains) ? Int(i) :
    throw(ArgumentError("NestHierarchy: no domain $i"))

"The simulation of domain `id` (a `Symbol` or the domain number; 1 is the root)."
nest_domain(h::NestHierarchy, id) = h.domains[_domain_index(h, id)].sim

"The link ([`NestedRun`](@ref)) whose child is domain `id`."
function nest_link(h::NestHierarchy, id)
    d = _domain_index(h, id)
    d == 1 && throw(ArgumentError("nest_link: the root has no parent link"))
    return h.links[h.domains[d].up]
end

"""
    add_nest!(h::NestHierarchy, parent => child_sim, nest; id, advance = domain_advance(),
              cfl = <the parent's>, feedback = :two_way, kwargs...) -> id

Attach `child_sim` (on `nest`'s child mesh) as a child of domain
`parent` (its id, a `Symbol`, or number) of the [`NestHierarchy`](@ref)
`h`, coupled through the [`NestedGrid`](@ref) `nest` (whose parent mesh is
the parent domain's). `advance` and `cfl` are the child's
[`DomainAdvance`](@ref) and `CFLConstraints` (`cfl = nothing`: a fixed
number of `substeps` per parent step); `feedback` (two-way by default, as
for every Aeolus parent driving an Aeolus child, ONE-f #455; `:one_way`
per child) and the other keywords
(`sponge`, `solid`, `order`, `scalars`, `flux_correction`, limiters, …)
are those of the link's [`NestedRun`](@ref). Returns the new domain's id
(default `Symbol(:d, n)`).

Refused with an `ArgumentError` that names the domains involved:

  * a child whose open faces' halo stencils reach outside the parent
    domain's interior, unless that side of the parent is periodic
    ([`proper_nesting_violations`](@ref)): the stencils would read the
    parent's own boundary halos;
  * a child overlapping a sibling (the parent cells their interiors
    cover intersect); two two-way siblings whose conservation boxes are
    less than one parent cell apart (a cell refluxed by one would be
    anterpolated by the other);
  * two-way feedback of a mixed core pair or without a conservation box
    (as for a `NestedRun`).

Overlapping siblings are refused rather than ranked: a priority rule needs
the higher-priority sibling as boundary data of the other, which is a
separate feature.
"""
function add_nest!(h::NestHierarchy, pr::Pair, nest::NestedGrid; id::Symbol = Symbol(:d, length(h.domains) + 1),
                   advance::DomainAdvance = domain_advance(), cfl = :parent,
                   feedback = :two_way, kwargs...)
    pidx = _domain_index(h, first(pr))
    child = last(pr)
    child isa _NestSim || throw(ArgumentError("add_nest!: the child must be a simulation"))
    any(d -> d.id === id, h.domains) && throw(ArgumentError("add_nest!: domain :$id exists"))
    any(d -> d.sim === child, h.domains) &&
        throw(ArgumentError("add_nest!: this simulation is already a domain of the hierarchy"))
    psim = h.domains[pidx].sim
    size(psim.θ.data) == totalsize(nest.parent_mesh) ||
        throw(ArgumentError("add_nest!: domain :$(h.domains[pidx].id) is not on the nest's parent mesh"))
    ccfl = cfl === :parent ? h.cfl[pidx] : cfl
    viol = proper_nesting_violations(nest, psim, child; order = get(kwargs, :order, 3))
    isempty(viol) ||
        throw(ArgumentError("add_nest!: :$id is not properly nested in :$(h.domains[pidx].id): " *
                            join(viol, "; ")))
    twoway = feedback !== :one_way
    for l in h.domains[pidx].links
        _check_siblings(h.links[l], nest, twoway, h.domains[h.fine[l]].id, id)
    end
    run = NestedRun(psim, child, nest; advance_parent = h.advance[pidx], advance_child = advance,
                    cfl_parent = h.cfl[pidx], cfl_child = ccfl, feedback,
                    sync = h.sync, max_retries = h.max_retries,
                    divergence_tol = h.divergence_tol, max_velocity = h.max_velocity,
                    validate = h.validate, kwargs...)
    push!(h.links, run); push!(h.coarse, pidx)
    d = _HDomain(id, child, length(h.links), Int[], nothing, Any[], nothing, Float64(child.t),
                 Float64[], h)
    push!(h.domains, d); push!(h.fine, length(h.domains))
    push!(h.advance, advance); push!(h.cfl, ccfl)
    push!(h.domains[pidx].links, length(h.links))
    _rebuild!(h, pidx); _rebuild!(h, length(h.domains))
    return id
end

# The register hooks, bound scalars and rollback of a domain, after its
# links changed.
function _rebuild!(h::NestHierarchy, di::Int)
    d = h.domains[di]
    up = d.up == 0 ? nothing : h.links[d.up]
    down = [h.links[l] for l in d.links]
    # the root is stepped through its first link, which books itself
    d.hook = up === nothing ? _chain_hooks(Any[_hook(L, :parent) for L in down]) :
                              _chain_hooks(Any[_hook(up, :child); [_hook(L, :parent) for L in down]])
    fs = Any[]
    up === nothing || append!(fs, _bound_fields(up, :child))
    for L in down, f in _bound_fields(L, :parent)
        any(g -> g === f, fs) || push!(fs, f)
    end
    d.fields = fs
    d.rb = _Rollback(d.sim, h.advance[di], fs)
    return d
end

# The subtree a domain's steps carry: for the root, its links beyond the
# first (which drives the step); for any other domain, all its links.
_sub_links(h::NestHierarchy, d::_HDomain) = d.up == 0 ? d.links[2:end] : d.links

function _begin_sub!(d::_HDomain)
    h = d.h
    for l in _sub_links(h, d)
        _begin!(h.links[l], :parent)
    end
    return nothing
end
function _commit_sub!(d::_HDomain)
    h = d.h
    for l in _sub_links(h, d)
        _commit!(h.links[l], :parent)
    end
    return nothing
end
_begin_sub!(::Nothing) = nothing
_commit_sub!(::Nothing) = nothing

# Before a step of domain d at time t: the starting ledger and the boundary
# data at t of every link where d is the coarse side.
function _sub_prepare!(d::_HDomain, t::Float64)
    h = d.h
    d.t0 = t
    resize!(d.m0, length(d.links))
    for (n, l) in enumerate(d.links)
        d.m0[n] = _prepare_window!(h.links[l], t)
    end
    return nothing
end

# After an accepted step of domain d of size Δt: open every child's window
# on it, advance every child to its end (recursively), close the windows
# (clock snap, two-way feedback, ledgers) and re-project d once after all
# two-way feedback.
function _sub_after_step!(d::_HDomain, Δt::Float64)
    h = d.h
    isempty(d.links) && return nothing
    t_end = Float64(d.sim.t)
    ws = [_open_window!(h.links[l], d.t0, t_end, Δt, d.m0[n]) for (n, l) in enumerate(d.links)]
    for (n, l) in enumerate(d.links)
        L = h.links[l]
        tw = time_ns()
        _subcycle_child!(L, ws[n].t_child_end, _child_sub(h, l))
        L.stats.wall_child += (time_ns() - tw) / 1e9
    end
    reproject = false
    for (n, l) in enumerate(d.links)
        L = h.links[l]
        _end_parent_step!(L, ws[n], false)
        L.twoway !== nothing && L.twoway.reproject && (reproject = true)
    end
    if any(l -> h.links[l].feedback === :two_way, d.links)
        reproject && _reproject!(d.sim)
        for l in d.links                       # d changed after every link's end data
            h.links[l].new_valid_at = NaN
        end
    end
    return nothing
end

# The subtree of a link's child, or nothing for a leaf (NestedRun's own path).
function _child_sub(h::NestHierarchy, l::Int)
    c = h.domains[h.fine[l]]
    return isempty(c.links) ? nothing : c
end

"""
    step_nested!(h::NestHierarchy; dt = nothing) -> h

One root step of the hierarchy: the root's step (adaptive from its `cfl`,
or `dt`; under `sync = :shared_min_dt` the smallest limit over all
domains), with the recursive subcycling of every descendant described at
[`NestHierarchy`](@ref).
"""
function step_nested!(h::NestHierarchy; dt = nothing)
    isempty(h.links) && throw(ArgumentError("step_nested!: the hierarchy has no nest; add_nest! first"))
    root = h.domains[1]
    L1 = h.links[root.links[1]]
    t_p = Float64(root.sim.t)
    Δt = if dt !== nothing
        Float64(dt)
    elseif h.cfl[1] === nothing
        error("step_nested!: the root has no cfl; pass dt")
    elseif h.sync === :shared_min_dt
        _shared_limit(h)
    else
        _dt_limit(root.sim, h.advance[1], h.cfl[1], h.dt_prev_root)
    end
    Δt > 0 || error("step_nested!: root step must be > 0, got $Δt")
    _sub_prepare!(root, t_p)
    tw = time_ns()
    # the root steps through its first link; `root` adds the hooks, bookings
    # and rollback of all its links
    Δt = _advance_parent!(L1, Δt, root)
    L1.stats.wall_parent += (time_ns() - tw) / 1e9
    _sub_after_step!(root, Δt)
    h.dt_prev_root = Δt
    h.root_steps += 1
    h.t = Float64(root.sim.t)
    return h
end

# The smallest step limit over all domains (sync = :shared_min_dt).
function _shared_limit(h::NestHierarchy)
    lim = Inf
    for (i, d) in enumerate(h.domains)
        c = h.cfl[i]
        c === nothing && throw(ArgumentError("step_nested!: sync = :shared_min_dt needs a cfl " *
                                             "for every domain (:$(d.id) has none)"))
        dtp = i == 1 ? h.dt_prev_root : h.links[d.up].dt_prev_child
        lim = min(lim, _dt_limit(d.sim, h.advance[i], c, dtp))
    end
    return lim
end

# =========================================================================
# Proper nesting and siblings
# =========================================================================

"""
    proper_nesting_violations(nest, parent_sim, child_sim; order = 3) -> Vector{String}

Why `child_sim` is not properly nested in `parent_sim` through `nest`
(empty when it is): along every open (nest) face of the child, the parent
cells that the order-`order` prolongation stencils of the child's halo
cells read must be parent *interior* cells, unless that side of the parent
is periodic. Otherwise the child's boundary data would be interpolated
from the parent's own boundary halos — in a nest hierarchy, from data the
parent itself only received from its parent.
"""
function proper_nesting_violations(nest::NestedGrid, psim, csim; order::Int = 3)
    out = String[]
    open = nest_open_sides(csim)
    per = nest_periodic_axes(psim)
    names = ("west", "east", "south", "north", "bottom", "top")
    ns = max(order, 1)
    for s in 1:6
        open[s] || continue
        ax = _side_axis(s)
        per[ax] && continue
        a = nest.axes[ax]
        # the child storage cells beyond the face, and their parent stencils
        cells = _side_high(s) ? (a.Hc + a.Nc + 1:a.Nc + 2a.Hc) : (1:a.Hc)
        lo = typemax(Int); hi = typemin(Int)
        for c in cells
            x = (a.cf[c] + a.cf[c+1]) / 2
            P = clamp(searchsortedlast(a.pf, x), 1, length(a.pf) - 1)
            s0 = P - ((ns - 1) ÷ 2)
            lo = min(lo, s0); hi = max(hi, s0 + ns - 1)
        end
        (lo >= a.Hp + 1 && hi <= a.Hp + a.Np) ||
            push!(out, "the $(names[s]) face's halo stencils read parent cells " *
                       "$(lo - a.Hp):$(hi - a.Hp), outside the parent interior 1:$(a.Np)")
    end
    return out
end

# The parent interior cells a child's interior covers, per axis.
function _footprint(nest::NestedGrid)
    return ntuple(3) do d
        a = nest.axes[d]
        x0 = a.cf[a.Hc+1]; x1 = a.cf[a.Hc+a.Nc+1]
        P0 = searchsortedlast(a.pf, x0); P1 = searchsortedfirst(a.pf, x1) - 1
        (P0 - a.Hp):(P1 - a.Hp)
    end
end

_boxes_overlap(a::NTuple{3,UnitRange{Int}}, b::NTuple{3,UnitRange{Int}}) =
    all(d -> !isempty(intersect(a[d], b[d])), 1:3)
_grow(b::NTuple{3,UnitRange{Int}}, g::Int) = map(r -> first(r)-g:last(r)+g, b)

function _check_siblings(L::NestedRun, nest::NestedGrid, twoway::Bool, other::Symbol, id::Symbol)
    fa = _footprint(L.nest); fb = _footprint(nest)
    _boxes_overlap(fa, fb) &&
        throw(ArgumentError("add_nest!: :$id overlaps its sibling :$other (parent cells " *
                            "$(map((x, y) -> intersect(x, y), fa, fb)) are covered by both). " *
                            "Overlapping siblings are not supported"))
    if twoway && L.feedback === :two_way
        ba = nest_covered_range(L.nest); bb = nest_covered_range(nest)
        _boxes_overlap(_grow(ba, 1), bb) &&
            throw(ArgumentError("add_nest!: the two-way siblings :$other and :$id have " *
                                "conservation boxes $ba and $bb less than one parent cell apart: " *
                                "a cell refluxed by one would be anterpolated by the other"))
    end
    return nothing
end

# =========================================================================
# Composite integrals of the hierarchy
# =========================================================================

"""
    composite_integral(h::NestHierarchy, quantity) -> Float64

The integral of a cell-centred scalar over the composite grid of the
hierarchy: every domain's cells that are not inside the conservation box
of one of its children, the root's over its whole interior and every other
domain's over the conservation box of its own link (the strip between a
child's edge and its box belongs to the parent). `quantity` is `:θ`
(`:theta`), `:qv`, or a scalar of the links' field sets (`:co2`, …),
weighted by each domain's reference mass for the sound-proof cores
(`∫ W φ dV`). The quantity a two-way hierarchy with refluxing conserves to
round-off.
"""
function composite_integral(h::NestHierarchy, quantity::Symbol)
    total = 0.0
    for (i, d) in enumerate(h.domains)
        f = _hfield(h, i, quantity)
        m = d.sim.mesh
        own = d.up == 0 ? (1:m.Nx, 1:m.Ny, 1:m.Nz) :
              map(a -> a.cbox, h.links[d.up].nest.axes)
        excl = NTuple{3,UnitRange{Int}}[nest_covered_range(h.links[l].nest) for l in d.links]
        total += _own_sum(f.data, m, own, excl, _level_mass(d.sim))
    end
    return total
end

# Σ φ V over the cells `own` of a domain outside the boxes `excl` (a
# function barrier: concrete array and mesh types in the loop).
function _own_sum(fd::AbstractArray, m, own, excl::Vector{NTuple{3,UnitRange{Int}}},
                  mass::Vector{Float64})
    part = 0.0
    @inbounds for k in own[3]
        V = m.dx * m.dy * mass[k]
        for j in own[2], i in own[1]
            _in_boxes(excl, i, j, k) && continue
            part += fd[m.Hx + i, m.Hy + j, m.Hz + k] * V
        end
    end
    return part
end

function _in_boxes(excl::Vector{NTuple{3,UnitRange{Int}}}, i::Int, j::Int, k::Int)
    for b in excl
        (i in b[1]) && (j in b[2]) && (k in b[3]) && return true
    end
    return false
end

# Per-level cell thickness (or reference mass W Δz) of a domain, interior-indexed.
function _level_mass(sim)
    m = sim.mesh
    dz = [_dzp(m, k) for k in 1:m.Nz]
    sim isa MassWeightedSimulation || return dz
    W = constraint_weights(sim)[1]
    return [W[m.Hz + k] * dz[k] for k in 1:m.Nz]
end

function _hfield(h::NestHierarchy, i::Int, q::Symbol)
    sim = h.domains[i].sim
    (q === :θ || q === :theta) && return sim.θ
    links = h.domains[i].up == 0 ? Int[] : [h.domains[i].up]
    for l in links
        s = h.links[l].s_bc
        n = findfirst(==(q), s.names)
        n === nothing || return s.child[n]
    end
    for l in h.domains[i].links
        s = h.links[l].s_bc
        n = findfirst(==(q), s.names)
        n === nothing || return s.parent[n]
    end
    q === :qv && hasproperty(sim, :qv) && return sim.qv
    throw(ArgumentError("composite_integral: no field :$q on domain :$(h.domains[i].id)"))
end
