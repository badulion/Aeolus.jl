# The urban site as a nested domain (N-berlin-nest, #265; NEST-p3, #483).
#
# The site is the child of its parent `[[domain]]` of the model: a domain of
# the model's core with its own components (the deep boundary-layer domain
# of #259: a stretched grid, the tiled urban surface, the column radiation,
# the large-scale forcing, TKE) and its own static and dynamic driver. One
# nest engine steps both (`Components.ModelNest`): the parent takes its
# adaptive step, the site steps through its own pipeline to the end of the
# parent's step, each step of the length it picks (the window), and a
# two-way site is fed back onto the parent (#267). This component wires the
# site into that nest:
#
#   faces     the site's four sides and its top are `OpenBoundary` faces fed
#             by the link's `Nesting.NestBoundaryData` (N-nest-d #261):
#             `sim_options` returns them (`nest_bcs`). The outflow condition
#             is `cfg.outflow` (Dirichlet from the parent by default).
#   link      `init!` links the site into the parent's nest hierarchy
#             (`Nesting.add_nest!`, before every other component's `init!`:
#             they initialise from the faces' data and the site's wind,
#             which come from the parent): the `Nesting.NestedGrid` at the
#             block's `origin` and `refinement`, the field set (θ, every
#             advected tracer, ql and qr), the solid mask on the faces, and
#             the feedback of the block's `feedback` key.
#   time      `limit_dt` shortens the site's step to land on the end of the
#             parent's step (`Nesting.nest_window_end`, `subcycle_dt`);
#             `:nest_sync` notes a new window (the relaxation targets, the
#             boundary recorder, the window hooks: the parent is at rest).
#   two-way   (#267) the engine feeds the site back at the end of the
#             parent's step (anterpolation outside a buffer of `buffer`
#             parent cells next to the open faces and above `z_anterp`,
#             `:canopy`: the mean height of the built columns; the interface
#             flux correction; Berger–Colella refluxing; the re-projection),
#             booked in the parent's budgets as its nest exchange. The
#             parent's θ and qv fluxes through the interface are booked by
#             its core's stages, the site's by its transports
#             (`book_nest_flux!`): θ from a `:body_force` entry, qv by the
#             `:scalar_advection` slot; each site step is bracketed by
#             `:nest_sync` / `:nest_feedback`.
#   scalars   qv (and the TKE `e` of the `:tke` SGS provider) take the
#             parent's values; every other advected tracer, and ql and qr,
#             is zero on inflow and leaves with its own value
#             (`add_open_scalars!` with the `:inflow` halo). The water and
#             heat the faces carry are booked by N-open-d's terms (#262):
#             `(:dynamics, :boundary_heat)` / `:mass_correction` /
#             `:relaxation` in the air-heat ledger, `:boundary` in the water
#             ledger.
#   targets   the core's velocity sponge (top 25 %) and the θ relaxation over
#             the same layer relax toward the parent's mean profile over the
#             site, refreshed every window.
#
# Without a parent domain the faces take boundary data instead
# (`parent_driver`): `:profile`, the analytic data of the site (the
# geostrophic wind and the initial θ); `:offline`, the parent-made dynamic
# driver an online run wrote (`[urban] child_boundaries`, CFG-g #362); or
# any `AbstractBoundaryData` (the interface a mesoscale reader of #58 or a
# reanalysis of #57 implements).

"""
    NestingCoupling

Additive coupling `:nesting`: the site is a nested domain (see the file
header and `docs/src/design/berlin_nesting.md`). Options:

* `parent_driver` — `:aeolus` (default: the site's parent `[[domain]]`,
  stepped by the model's nest engine), `:profile` (the analytic
  `case_boundary_data` of the site), `:offline` (the parent-made dynamic
  driver of `[initial] dynamic`), or any `AbstractBoundaryData`;
* `buffer` — two-way: parent cells next to every open face of the site
  that are not anterpolated (default 2, PALM's);
* `z_anterp` — two-way: parent cells whose lower face lies below it are
  not anterpolated (the canopy restriction); `:canopy` (default) is the
  mean height of the site's built columns, or a height in m (`-Inf`:
  none);
* `τ` — two-way: relaxation time of the anterpolation (`nothing`,
  default: direct replacement, as in PALM);
* `reflux` — two-way: Berger–Colella refluxing of θ and qv (default
  `true`);
* `budget_exchange` — the nest's budgets (`[nesting] budget_exchange` of
  the root domain, `Components.nested_model`).

Whether the site feeds back is its `[[domain]]` block's `feedback`
(`"one_way"` or `"two_way"`; the Berlin nest is one-way until the
two-way campaign #331 validates it).

Fields: `mode` (`:one_way` or `:two_way`), the options, the faces'
`driver`, the parent's `Components.Domain` `parent` and the hierarchy's
link `run` (a `Nesting.NestedRun`; built by `init!`), the end of the
parent window the site steps in (`window_end`), the tracer names of the
field set (`from_parent`: carried by the parent; `inflow`: zero on
inflow), the windows seen, the relaxation `targets`, the offline replay
(`feed`, `inputs`, `k`, `set`, `window`), the boundary `recorder`, and
`window_hooks`: functions `f(c, case)` called at the site's first step in
every parent window — the parent at rest — the one point where another
component may read the parent's fields (the synthetic inflow's
statistics, `components/urban/inflow_turbulence.jl`, #268).
"""
mutable struct NestingCoupling <: UrbanComponent
    mode::Symbol
    opts::Any                     # the coupling's options (the two-way keywords)
    driver::Any                   # :aeolus, :offline, or the AbstractBoundaryData that replaces the parent
    parent::Any                   # the parent's Components.Domain (online), or nothing
    run::Any                      # the hierarchy's link (online), the driver or the feed, or nothing
    window_end::Float64           # online: the end of the parent window the site steps in (NaN: none yet)
    window::Any                   # offline: the open record window, or nothing
    from_parent::Vector{Symbol}
    inflow::Vector{Symbol}
    windows::Int
    wall_build::Float64           # s building the link
    targets::Any                  # (u, v, θ) level vectors the relaxations use, or nothing
    feed::Any                     # offline: the FileBoundaryFeed of cfg.dynamic, or nothing
    inputs::Any                   # offline: the DynamicInputs of cfg.dynamic
    k::Int                        # offline: the record the next window starts from
    set::Vector{Any}              # offline: the child fields of the field set's tracers, in order
    recorder::Any                 # online with cfg.child_boundaries: the NestBoundaryRecorder
    window_hooks::Vector{Any}     # f(c, case) at every window's first site step, the parent at rest (#268)
end

component_name(::Type{<:NestingCoupling}) = :nesting
modules_used(::Type{<:NestingCoupling}) = (:Nesting,)
default_options(::Type{<:NestingCoupling}) =
    (parent_driver = :aeolus, buffer = 2, z_anterp = :canopy, τ = nothing, reflux = true,
     budget_exchange = :exchange_term)
# the child's θ booking of the two-way registers runs in every dynamics stage
stages(::Type{<:NestingCoupling}) = ((:nest_sync, 10, EVERY_STEP), (:nest_feedback, 10, EVERY_STEP),
                                     (:body_force, 5, EVERY_STEP))

"""
    nest_geostrophic_wind(case) -> (U_g, V_g)

The geostrophic wind of the parent domain of the nested site `case`: the
`U_g`, `V_g` of its `[large_scale_forcing]` component (numbers, tables or
functions `(z, t)`, as `LargeScaleForcing.forcing_value` reads them).
The `:geostrophic` coupling of a nested site is driven by it, so the
large-scale pressure gradient is one field across the nest.
"""
function nest_geostrophic_wind(case)
    p = case.nest.parent
    i = findfirst(c -> c isa Components.LargeScale, p.components)
    i === nothing && error("nesting: the parent domain $(p.id) has no [large_scale_forcing]: the nested " *
                           "site's [geostrophic] mode = \"pressure_gradient\" takes the parent's geostrophic wind")
    ls = p.components[i]
    return ls.U_g, ls.V_g
end

"""
    nest_case_bcs(cfg, mesh) -> FieldBoundaryConditions

The boundary conditions of the nested child: a ground wall, the four sides
and the top nest faces (`Nesting.nest_bcs`, `OpenBoundary` faces whose data
the parent's `NestBoundaryData` supplies once the nest is built), with
the outflow condition `cfg.outflow` and the halo the tracer advection
scheme needs at a face. The one definition of the nested case's faces:
the coupling's `sim_options` pass it to the core, the pressure-solver
couplings derive their axes from it.
"""
function nest_case_bcs(cfg::NamedTuple, mesh)
    sa = component_options(ScalarAdvection, cfg)
    H = Aeolus.AdvectionSchemes.required_halo(Aeolus.AdvectionSchemes.advection_scheme(sa.scheme))
    return nest_bcs(mesh; bottom = :wall, top = :nest, outflow = cfg.outflow, halo = H)
end

function sim_options(::Type{<:NestingCoupling}, cfg, mesh, o)
    b = nest_case_bcs(cfg, mesh)
    # the sponge below the open top relaxes toward the parent's mean wind
    # over the child: level vectors, filled at every parent window
    cfg.sponge_α_max > 0 || return (bcs = b, θ_bcs = b)
    return (bcs = b, θ_bcs = b, sponge_target = (u = zeros(mesh.Nz), v = zeros(mesh.Nz), w = 0.0))
end

function build(::Type{<:NestingCoupling}, case, o)
    d = o.parent_driver
    nx = case.nest
    if d === :aeolus
        nx === nothing &&
            error("nesting: parent_driver = \"aeolus\" nests the site in its parent [[domain]] of the model; " *
                  "this site has none (give its [[domain]] block a parent, or set [nesting] parent_driver " *
                  "to \"profile\" or \"offline\")")
    else
        nx === nothing ||
            error("nesting: the site is nested in the domain $(nx.parent_id): its faces take the parent's data " *
                  "(parent_driver = \"aeolus\"), not $(repr(d))")
    end
    driver = d in (:aeolus, :offline) ? d : d === :profile ? case_boundary_data(case.cfg) :
             d isa AbstractBoundaryData ? d :
             error("nesting: parent_driver must be :aeolus, :profile, :offline or an AbstractBoundaryData, got $(repr(d))")
    mode = nx === nothing ? :one_way : nx.feedback
    return NestingCoupling(mode, o, driver, nx === nothing ? nothing : nx.parent, nothing, NaN, nothing,
                           Symbol[], Symbol[], 0, 0.0, nothing, nothing, nothing, 1, Any[], nothing, Any[])
end

"Whether the nest's faces are fed by boundary data other than the parent domain or its file (`parent_driver`)."
external_driver(c::NestingCoupling) = !(c.driver in (:aeolus, :offline))

"Whether the nest's faces replay a parent-made dynamic driver (`parent_driver = :offline`, CFG-g #362)."
offline(c::NestingCoupling) = c.driver === :offline

"""
    nest_offline(cfg) -> Bool

Whether the nested case `cfg` replays its parent from the parent-made
dynamic driver `cfg.dynamic` (`options[:nesting].parent_driver =
:offline`, CFG-g #362) instead of running it.
"""
nest_offline(cfg::NamedTuple) = case_nested(cfg) && component_options(NestingCoupling, cfg).parent_driver === :offline

"Whether the nest feeds the child back onto the parent (the `[[domain]]` block's `feedback = \"two_way\"`, #267)."
two_way(c::NestingCoupling) = c.mode === :two_way

"""
    nest_z_anterp(case, z_anterp) -> Float64

The canopy restriction of the two-way feedback: `z_anterp` in m, or for
`:canopy` the mean height of the child's built columns (the urban canopy
height; `-Inf` for a scene without buildings). Parent cells whose lower
face lies below it are not anterpolated (`Nesting.TwoWayFeedback`): the
coarse parent cannot represent the street canyons the child resolves.
"""
function nest_z_anterp(case, z::Union{Symbol,Real})
    z isa Real && return Float64(z)
    z === :canopy || error("nesting: z_anterp must be :canopy or a height in m, got :$z")
    h = filter(>(0), vec(case.geom.heightmap))
    return isempty(h) ? -Inf : sum(h) / length(h)
end

# The two-way feedback of the site. The parent's θ and qv are transported
# inside its core's stages, so the run's register hook books them there
# (`applied_by = :core`); the site moves θ in its core's step pieces and qv
# by its :scalar_advection slot, and books both itself (`child_applied_by =
# :transport`, book_nest_flux!).
function _berlin_feedback(c::NestingCoupling, case, p::Domain)
    o = c.opts
    if o.reflux
        ts = timestepper(case)
        (ts isa IMEXStep || (ts isa AdaptiveStep && ts.scheme !== nothing)) &&
            error("nesting: two-way refluxing books the explicit θ diffusion of the child's " *
                  "step!; the IMEX stepper's is implicit. Use reflux = false with it")
    end
    tf = (_moist(p.state) && :qv in c.from_parent) ?
         (:qv => NestScalarFlux(; molecular = true, sgs = :tracer, child_applied_by = :transport),) : ()
    return TwoWayFeedback(; buffer = o.buffer, z_anterp = nest_z_anterp(case, o.z_anterp), τ = o.τ,
                          reflux = o.reflux,
                          theta_flux = NestScalarFlux(; molecular = true, sgs = :theta, child_applied_by = :transport),
                          tracer_fluxes = tf)
end

init!(c::NestingCoupling, case) = (ensure_nest!(c, case); nothing)
# before every other coupling's init!: they initialise from the faces' data
# and the child's wind, which come from the parent
init_priority(::Type{<:NestingCoupling}) = -1

# The parent field a child tracer takes its boundary values from, or nothing.
function _parent_scalar(p::Domain, name::Symbol)
    s = p.state
    name === :qv && return _moist(s) ? s.qv : nothing
    if name === :e
        cl = Components.sgs_closure(p)
        return cl isa TKEClosure ? cl.e : nothing
    end
    return nothing
end

"""
    ensure_nest!(c::NestingCoupling, case) -> c

Link the site into its parent's nest (once; `init!`): the
`Nesting.NestedGrid` (the site at its own origin, the parent's corner at
its `[grid] origin`, the site at the block's `origin` in it), the field
set (θ and every advected tracer, ql and qr; the planes of the tracers
added to the nest faces with `add_open_scalars!`), and the link
(`Nesting.add_nest!` on the model's hierarchy: the site's solid mask on
the faces, the feedback of its block), which attaches the parent's
`NestBoundaryData` to the site's faces and fills their halos. The site's
initial wind is the parent's, prolongated. Without a parent domain:
the offline replay or the external driver's faces.
"""
function ensure_nest!(c::NestingCoupling, case)
    c.run === nothing || return c
    t0 = time_ns()
    sim = case.sim
    if offline(c)
        _ensure_offline!(c, case)
        c.wall_build += (time_ns() - t0) * 1e-9
        return c
    end
    if external_driver(c)
        # a boundary-data driver in place of the parent (the #58 interface)
        set_boundary_data!(sim.bcs, c.driver)
        prepare_open_boundaries!(sim)
        fill_halos!((sim.u, sim.v, sim.w, sim.θ), (sim.bcs, sim.bcs, sim.bcs, sim.θ_bcs))
        c.run = c.driver
        st = sim.sponge_target
        if st.u isa AbstractVector && c.driver isa ProfileBoundaryData
            tgt = Aeolus.Simulations.sponge_target(c.driver, case.mesh)
            st.u .= tgt.u; st.v .= tgt.v
        end
        c.wall_build += (time_ns() - t0) * 1e-9
        return c
    end
    nx = case.nest
    p = nx.parent
    pm, cm = Components.grid(p), case.mesh
    # the site's mesh starts at its own corner (0, 0); its parent's corner
    # (its [grid] origin) is then at -origin
    (isapprox(pm.x0, cm.x0 - nx.origin[1]; atol = 1e-9 * pm.dx) &&
     isapprox(pm.y0, cm.y0 - nx.origin[2]; atol = 1e-9 * pm.dy)) ||
        error("nesting: the site's corner is at ($(cm.x0), $(cm.y0)) m and its origin in $(nx.parent_id) is " *
              "$(nx.origin) m, so the parent's [grid] origin is $((cm.x0 - nx.origin[1], cm.y0 - nx.origin[2])), " *
              "not ($(pm.x0), $(pm.y0))")
    off = (round(Int, nx.origin[1] / pm.dx) + 1, round(Int, nx.origin[2] / pm.dy) + 1, 1)
    nest = NestedGrid(pm, cm; refinement = nx.refinement, offset = off)
    # the field set: θ (always), and the tracers
    zp = Field(CellCenter(), pm)                     # the value of a tracer the parent lacks
    pairs = Pair{Symbol,Any}[]
    for t in case.tracers
        (t.advect || t.name in (:ql, :qr)) || continue
        pf = _parent_scalar(p, t.name)
        # two-way anterpolates every member of the field set into its parent
        # field: a tracer the parent lacks gets a parent field of its own
        # (inert in the parent; its child faces keep the :inflow halo)
        push!(pairs, t.name => (t.field, pf !== nothing ? pf : two_way(c) ? Field(CellCenter(), pm) : zp))
        push!(pf === nothing ? c.inflow : c.from_parent, t.name)
    end
    isempty(c.from_parent) || add_open_scalars!(sim.bcs, c.from_parent; mode = :value)
    isempty(c.inflow) || add_open_scalars!(sim.bcs, c.inflow; mode = :inflow)
    state = ntuple(_ -> nest_interp_state(nest; scalars = pairs), 3)
    fb = two_way(c) ? _berlin_feedback(c, case, p) : :one_way
    h = nx.hierarchy
    Aeolus.Nesting.add_nest!(h, nx.parent_id => sim, nest; id = nx.id,
                             advance = Aeolus.Nesting.domain_advance(Components.PipelineCore()),
                             cfl = nothing, feedback = fb, solid = case.geom.sm, state = state)
    run = h.links[h.domains[end].up]
    c.run = run
    # the child's initial wind is the parent's, prolongated over the whole
    # child: an initial wind that differs from the faces' (the case's u0
    # against a sounding's geostrophic wind) makes ∇·u ≠ 0 in the boundary
    # cells, and the first step's flux-form θ advection then changes θ there
    # by −θ ∇·u Δt (≈ 290 K in a corner of the smoke window with the summer
    # sounding: 1 m/s inside, 3.4 m/s on the faces). At t = 0 the parent's
    # wind is horizontally uniform, so the prolongated wind is divergence-free.
    s = run.s_new
    interpolate_parent_state!(s, p.state, nest; order = run.order, region = :all)
    for (cf, sf) in ((sim.u, s.u), (sim.v, s.v), (sim.w, s.w))
        interior_view(cf) .= interior_view(sf)
    end
    fill_nest_halos!(sim, run.s_old)
    st = sim.sponge_target
    c.targets = (u = st.u isa AbstractVector ? st.u : nothing,
                 v = st.v isa AbstractVector ? st.v : nothing,
                 θ = case.open === nothing || case.open.damping === nothing ? nothing :
                     case.open.damping.target)
    _update_targets!(c, case)
    case.cfg.child_boundaries === nothing || _start_recorder!(c, case)
    c.wall_build += (time_ns() - t0) * 1e-9
    return c
end

"""
    mean_heat_flux(sim, κ) -> (zf, flux)

The plane-mean total kinematic heat flux (K m s⁻¹) of any core `sim` at
its interior w-faces (heights `zf` above the bottom): resolved
``⟨w'θ'⟩`` (θ averaged to the face) plus the SGS ``−⟨κ_v ∂θ/∂z⟩`` with the
cell-centred diffusivity `κ` (the centre spacing of a stretched grid in
the gradient) — the profile whose minimum is the `:min_flux` z_i
(Sullivan et al. 1998). Works on a `RegularGrid` and a `StretchedGrid`.
"""
function mean_heat_flux(sim, κ)
    m = sim.mesh
    Hx, Hy, Hz, Nx, Ny, Nz = m.Hx, m.Hy, m.Hz, m.Nx, m.Ny, m.Nz
    θd, wd, κd = sim.θ.data, sim.w.data, κ.data
    zfk(K) = hasproperty(m, :zf) ? m.zf[K] - m.z0 : (K - Hz - 1) * m.dz
    dzfk(K) = hasproperty(m, :dzf) ? m.dzf[K] : m.dz
    zf = Float64[zfk(Hz + k) for k in 2:Nz]
    flux = zeros(Nz - 1)
    n = Nx * Ny
    @inbounds for k in 2:Nz
        K = Hz + k
        sw = 0.0; sθ = 0.0; swθ = 0.0; sg = 0.0
        idz = 1 / dzfk(K)
        for j in Hy+1:Hy+Ny, i in Hx+1:Hx+Nx
            θf = 0.5 * (θd[i, j, K-1] + θd[i, j, K])
            w = wd[i, j, K]
            sw += w; sθ += θf; swθ += w * θf
            sg -= 0.5 * (κd[i, j, K-1] + κd[i, j, K]) * (θd[i, j, K] - θd[i, j, K-1]) * idz
        end
        flux[k-1] = swθ / n - (sw / n) * (sθ / n) + sg / n
    end
    return zf, flux
end

"""
    parent_zi(p::Domain; method = :min_flux) -> m

The boundary-layer height of the domain `p` (a nested site's parent) now:
`:min_flux`, the height of the minimum of [`mean_heat_flux`](@ref) with
the vertical κ of its SGS closure (the CBL definition), or `:parcel` (on
the plane-mean θ_v), both `boundary_layer_height`.
"""
function parent_zi(p::Domain; method::Symbol = :min_flux)
    s = p.state
    if method === :min_flux
        cl = Components.sgs_closure(p)
        zf, fl = mean_heat_flux(s, cl === nothing ? s.κt : sgs_fields(cl, s).κv)
        return boundary_layer_height(zf, fl; method = :min_flux)
    end
    method === :parcel || throw(ArgumentError("parent_zi: method is :min_flux or :parcel, got :$method"))
    return Components.parcel_zi(s)
end

# --- the parent-made dynamic driver (CFG-g, #362) ----------------------------

# The mean-profile targets as the recorder's `profiles` (only those the case relaxes toward).
_target_profiles(c::NestingCoupling) =
    (; (n => v for (n, v) in ((:u, c.targets.u), (:v, c.targets.v), (:theta, c.targets.θ)) if v !== nothing)...)

# Online, with cfg.child_boundaries: the in-run writer of the child's
# boundary slabs (Aeolus.InputDrivers.NestBoundaryRecorder), started with
# the child's initial wind and the first record, B(0), the data the child
# starts from.
function _start_recorder!(c::NestingCoupling, case)
    two_way(c) && error("nesting: [output] child_boundaries records a one-way nest (the offline child " *
                        "cannot feed back); feedback = \"two_way\"")
    cfg, sim, m = case.cfg, case.sim, case.mesh
    planes = [p for st in open_boundary_states(sim) for p in st.planes]
    ox, oy = anchor_xy(cfg, m)
    fields = Dict{Symbol,Array{Float64,3}}(n => Array{Float64}(interior_view(getproperty(sim, n))) for n in (:u, :v, :w))
    c.recorder = NestBoundaryRecorder(cfg.child_boundaries, planes, m; init_time = case_init_time(cfg),
        fields = fields, profiles = collect(keys(_target_profiles(c))),
        attributes = Dict{String,Any}("nest_from_parent" => join(String.(c.from_parent), " "),
                                      "nest_inflow" => join(String.(c.inflow), " "),
                                      "nest_refinement" => collect(Int, case.nest.refinement)),
        preprocessor = "aeolus run ([output] child_boundaries)",
        origin_lat = cfg.lat0, origin_lon = cfg.lon0, origin_x = ox, origin_y = oy)
    record_boundary!(c.recorder, sim.t, NaN, c.run.data.a; profiles = _target_profiles(c))
    return c
end

# Offline: the faces, the field set, the initial wind and the targets from
# the parent-made driver, in the order the online ensure_nest! sets them.
function _ensure_offline!(c::NestingCoupling, case)
    cfg, sim, m = case.cfg, case.sim, case.mesh
    path = cfg.dynamic
    path === nothing && error("nesting: parent_driver = :offline reads the nest faces from [initial] dynamic, " *
                              "which is \"none\" (the online run writes one: [output] child_boundaries)")
    ox, oy = anchor_xy(cfg, m)
    d = read_dynamic_driver(path; mesh = m, origin_lat = cfg.lat0, origin_lon = cfg.lon0, origin_x = ox, origin_y = oy)
    src = get(d.attributes, "dynamic_source", "")
    src == "parent" || error("nesting: parent_driver = :offline needs a parent-made dynamic driver; $path has " *
                             "dynamic_source = $(repr(src))")
    d.boundary === nothing && error("nesting: the dynamic driver $path has no boundary planes")
    d.init_time == case_init_time(cfg) ||
        error("nesting: the dynamic driver $path starts at $(d.init_time), the case at $(case_init_time(cfg))")
    want_p = Set(Symbol.(split(get(d.attributes, "nest_from_parent", ""))))
    want_i = Set(Symbol.(split(get(d.attributes, "nest_inflow", ""))))
    for t in case.tracers
        (t.advect || t.name in (:ql, :qr)) || continue
        push!(t.name in want_p ? c.from_parent : c.inflow, t.name)
        push!(c.set, t.field)
    end
    (Set(c.from_parent) == want_p && Set(c.inflow) == want_i) ||
        error("nesting: the dynamic driver $path was written for the tracers $(sort!(collect(want_p))) " *
              "(from the parent) and $(sort!(collect(want_i))) (zero inflow); this case advects " *
              "$(sort!(vcat(c.from_parent, c.inflow)))")
    isempty(c.from_parent) || add_open_scalars!(sim.bcs, c.from_parent; mode = :value)
    isempty(c.inflow) || add_open_scalars!(sim.bcs, c.inflow; mode = :inflow)
    feed = d.boundary
    for st in open_boundary_states(sim)
        check_planes(feed, st.planes)
    end
    check_coverage(feed, sim.t, sim.t + cfg.nsteps * cfg.dt)
    for (f, n) in ((sim.u, :u), (sim.v, :v), (sim.w, :w))
        haskey(d.fields, n) || error("nesting: the dynamic driver $path has no 3-D initial $n")
    end
    Aeolus.Nesting._mark_nest_solids!(sim, case.geom.sm)
    feed_steady!(feed, 1)
    for st in open_boundary_states(sim)
        st.data = feed.data
        st.initialized = false
        for t in case.tracers
            (t.advect || t.name in (:ql, :qr)) && (st.bound[t.field.data] = t.name)
        end
    end
    _fill_offline_halos!(c, case)
    for (f, n) in ((sim.u, :u), (sim.v, :v), (sim.w, :w))
        interior_view(f) .= d.fields[n]
    end
    _fill_offline_halos!(c, case)
    c.feed, c.inputs, c.run, c.k = feed, d, feed, 1
    st = sim.sponge_target
    c.targets = (u = st.u isa AbstractVector ? st.u : nothing,
                 v = st.v isa AbstractVector ? st.v : nothing,
                 θ = case.open === nothing || case.open.damping === nothing ? nothing : case.open.damping.target)
    _offline_targets!(c, 1)
    return c
end

# The open-boundary buffers at the child's time and the halos of the field
# set (Nesting.fill_nest_halos!, member for member).
function _fill_offline_halos!(c::NestingCoupling, case)
    sim = case.sim
    update_open_boundaries!(sim, sim.t, zero(sim.t))
    fill_halos!((sim.u, sim.v, sim.w, sim.θ), (sim.bcs, sim.bcs, sim.bcs, sim.θ_bcs))
    foreach(f -> fill_halos!(f, sim.θ_bcs), c.set)
    return c
end

# The relaxation targets of record k (the parent's mean profiles at time_bc[k]).
function _offline_targets!(c::NestingCoupling, k::Int)
    tg = c.targets
    tg === nothing && return c
    for (v, n) in ((tg.u, :profile_u), (tg.v, :profile_v), (tg.θ, :profile_theta))
        v === nothing && continue
        tab = get(c.inputs.tables, n, nothing)
        tab === nothing && error("nesting: the dynamic driver $(c.feed.path) has no ls_$n, a target this case relaxes toward")
        v .= view(tab.values, :, k)
    end
    return c
end

# The parent's mean of `f` over the columns the child covers, per parent
# level, interpolated linearly in height to the child's level centres `zc`
# (constant below the parent's first centre).
function _covered_profile!(out::AbstractVector, f, nest, zc)
    pm = nest.parent_mesh
    cx, cy, _ = nest_covered_range(nest)
    a = interior_view(f)
    n = length(cx) * length(cy)
    prof = [sum(view(a, cx, cy, K)) / n for K in 1:pm.Nz]
    zp = level_centres(pm)
    @inbounds for k in eachindex(out)
        z = zc[k]
        K = searchsortedlast(zp, z)
        out[k] = K < 1 ? prof[1] : K >= pm.Nz ? prof[end] :
                 prof[K] + (prof[K+1] - prof[K]) * (z - zp[K]) / (zp[K+1] - zp[K])
    end
    return out
end

# Refresh the relaxation targets (the core's velocity sponge, the θ
# relaxation of the open faces) from the parent.
function _update_targets!(c::NestingCoupling, case)
    tg = c.targets
    tg === nothing && return c
    m = case.mesh
    zc = Float64[m.z0 + (k - 0.5) * m.dz for k in 1:m.Nz]
    ps, nest = c.parent.state, c.run.nest
    tg.u === nothing || _covered_profile!(tg.u, ps.u, nest, zc)
    tg.v === nothing || _covered_profile!(tg.v, ps.v, nest, zc)
    tg.θ === nothing || _covered_profile!(tg.θ, ps.θ, nest, zc)
    return c
end

# Online: the site's step lands on the end of the parent window it is in
# (the nest engine opened it; a fixed step lands there by itself, the
# window being a whole number of its steps: `Components.window_dt`).
function limit_dt(c::NestingCoupling, case, dt)
    ensure_nest!(c, case)
    external_driver(c) && return dt
    if offline(c)
        if c.window === nothing
            c.k < length(c.feed.time) || return dt          # past the records: run_stage! reports it
            R = c.feed.step[c.k + 1]
        else
            R = c.window.t_child_end - case.sim.t
        end
    else
        R = Aeolus.Nesting.nest_window_end(c.run) - case.sim.t
    end
    R > 0 || return dt
    return subcycle_dt(R, dt)
end

function run_stage!(c::NestingCoupling, case, ::Val{:nest_sync}, dt)
    ensure_nest!(c, case)
    external_driver(c) && return nothing
    t = case.sim.t
    if offline(c)
        c.window === nothing && _open_offline_window!(c, case)
        w = c.window
        t + dt <= w.t_child_end + 1e-9 * w.Δt ||
            error("nesting: the child step [$t, $(t + dt)] crosses the end of the record window $(w.t_child_end)")
        return nothing
    end
    L = c.run
    we = Aeolus.Nesting.nest_window_end(L)
    isfinite(we) ||
        error("nesting: the site steps outside a window of its parent $(c.parent.id); a nested site is " *
              "stepped by the model's nest engine (step_adaptive!(model))")
    if we != c.window_end
        # the first step of the site in a new parent window: the parent is at rest
        c.window_end = we
        c.windows += 1
        _update_targets!(c, case)
        c.recorder === nothing ||
            record_boundary!(c.recorder, we, L.Δt, L.data.b; profiles = _target_profiles(c))
        # what reads the parent between its steps (the inflow statistics, #268)
        for f in c.window_hooks
            f(c, case)
        end
    end
    t + dt <= we + 1e-9 * L.Δt ||
        error("nesting: the site's step [$t, $(t + dt)] crosses the end of the parent window $we; " *
              "with a fixed step, dt must divide the window (the root's window is a whole number of the " *
              "site's fixed steps)")
    # two-way: the child step's interface fluxes are booked between here and
    # :nest_feedback (the :scalar_advection slot, the :body_force entry)
    if two_way(c)
        L.sub_t0 = Float64(t)
        Aeolus.Nesting._begin!(L, :child)
    end
    return nothing
end

# Two-way: the child's θ flux through the interface planes, at every
# dynamics stage (the child's step! has one stage of weight 1): centred
# advection and the core's diffusion with the case's SGS closure.
function run_stage!(c::NestingCoupling, case, ::Val{:body_force}, dt)
    (two_way(c) && c.opts.reflux && !external_driver(c)) || return nothing
    sim = case.sim
    book_nest_flux!(sim, :θ, sim.θ, dt; scheme = :centered, diffusivity = :theta,
                    sgs = get(case.step_kwargs, :sgs, nothing))
    return nothing
end

function run_stage!(c::NestingCoupling, case, ::Val{:nest_feedback}, dt)
    external_driver(c) && return nothing
    if offline(c)
        w = c.window
        w === nothing && return nothing
        if abs(case.sim.t - w.t_child_end) <= 1e-9 * w.Δt
            case.sim.t = w.t_child_end              # the child lands on the record
            c.k += 1
            c.window = nothing
        end
        return nothing
    end
    # the engine closes the window (and feeds back) when the site reaches its end
    two_way(c) && Aeolus.Nesting._commit!(c.run, :child)
    return nothing
end

# Offline: open the window from record k to k + 1 (t0 the child's clock,
# Δt = time_bc[k+1] − time_bc[k]), refresh the faces at its start and the
# targets to its end, as the online nest does.
function _open_offline_window!(c::NestingCoupling, case)
    f, k, sim = c.feed, c.k, case.sim
    k < length(f.time) || check_coverage(f, sim.t, sim.t + case.cfg.dt)
    tol = 1e-9 * max(f.time[k+1] - f.time[k], 1.0)
    abs(sim.t - f.time[k]) <= tol ||
        error("nesting: the child is at t = $(sim.t), the window of record $k of $(f.path) starts at $(f.time[k])")
    feed_window!(f, k)
    f.data.t0 = Float64(sim.t)
    c.window = (t_child_end = f.time[k+1], Δt = f.step[k+1])
    c.windows += 1
    _fill_offline_halos!(c, case)
    _offline_targets!(c, k + 1)
    return c
end

"""
    diagnostics(c::NestingCoupling, case)

Nested in a parent domain: `parent_windows` (parent windows the site
stepped in), `dt_parent_min` / `dt_parent_max` (s); the heat the open
faces carried, `boundary_heat_sides` / `boundary_heat_top` (W m⁻², mean
over the run) and the largest normal-velocity correction `delta_max`
(m s⁻¹); a two-way nest adds `feedback_count`. An offline nest
(`parent_driver = :offline`, #362): `parent_windows` (windows replayed)
and `boundary_records` (records read from the dynamic driver). None for
an external `parent_driver`.
"""
function diagnostics(c::NestingCoupling, case)
    offline(c) && return Diagnostic[
        Diagnostic("parent_windows", "1", () -> c.windows; cadence = :step,
                   description = "parent windows replayed from the dynamic driver"),
        Diagnostic("boundary_records", "1", () -> c.feed === nothing ? 0 : c.feed.reads;
                   description = "boundary records read from the dynamic driver")]
    (c.run === nothing || external_driver(c)) && return Diagnostic[]
    st() = c.run.stats
    ob = case.open
    perrun(x) = () -> ob === nothing ? NaN : x(ob) / max(case.sim.t, eps())
    ds = Diagnostic[
        Diagnostic("parent_windows", "1", () -> c.windows; cadence = :step),
        Diagnostic("dt_parent_min", "s", () -> st().dt_parent_min),
        Diagnostic("dt_parent_max", "s", () -> st().dt_parent_max),
        Diagnostic("boundary_heat_sides", "W m-2", perrun(o -> o.lateral);
                   description = "heat through the lateral faces, mean over the run"),
        Diagnostic("boundary_heat_top", "W m-2", perrun(o -> o.top);
                   description = "heat through the top face, mean over the run"),
        Diagnostic("delta_max", "m s-1", () -> ob === nothing ? NaN : ob.δmax;
                   description = "largest normal-velocity correction of the open faces")]
    two_way(c) && push!(ds, Diagnostic("feedback_count", "1", () -> st().parent_steps;
                                       description = "feedbacks onto the parent"))
    return ds
end

# --- smoke checks ------------------------------------------------------------

function smoke_check(c::NestingCoupling, case)
    c.run === nothing && return ["the nest was built" => false]
    if external_driver(c) || offline(c)
        sim = case.sim; m = case.mesh
        A = 2 * (m.Lx + m.Ly) * m.Lz + m.Lx * m.Ly
        U = max(maximum(abs, interior_view(sim.u)), maximum(abs, interior_view(sim.v)), 1e-3)
        data = offline(c) ? c.feed.data : c.driver
        # (wrapped by the synthetic inflow when it is installed, #268)
        return ["parent_driver: the nest faces carry the driver's data" =>
                    all(st -> st.data === data || (st.data isa InflowTurbulence && st.data.data === data),
                        open_boundary_states(sim)),
                "child: net boundary volume flux after the correction ≤ 1e-12 U A" =>
                    abs(open_boundary_ledger(sim).residual) <= 1e-12 * U * A]
    end
    sim, p, run = case.sim, c.parent, c.run
    m = case.mesh
    A = 2 * (m.Lx + m.Ly) * m.Lz + m.Lx * m.Ly
    U = max(maximum(abs, interior_view(sim.u)), maximum(abs, interior_view(sim.v)), 1e-3)
    led = open_boundary_ledger(sim)
    geo = find_coupling(case, :geostrophic)
    zc = [m.z0 + (k - 0.5) * m.dz for k in 1:m.Nz]
    tw = sim.t
    Ug, Vg = nest_geostrophic_wind(case)
    geo_same = geo === nothing || !(geo.forcing isa GeostrophicPressureGradient) ||
               all(z -> forcing_value(geo.forcing.U_g, z, tw) == forcing_value(Ug, z, tw) &&
                        forcing_value(geo.forcing.V_g, z, tw) == forcing_value(Vg, z, tw), zc)
    sfc = Components.slot_component(p, :surface)
    tiles = !(sfc isa TiledUrbanSurface) ||
            all(col -> col === nothing || isapprox(sum(mosaic_fractions(col.surface)), 1; atol = 1e-12),
                sfc.surface.columns)
    we = c.window_end
    checks = two_way(c) ? _two_way_checks(c, case) : Pair{String,Bool}[]
    out = [checks...,
           "the parent ran: ≥ 1 window, the site at the end of the last one" =>
               c.windows >= 1 && abs(sim.t - we) <= 1e-9 * max(run.Δt, 1.0),
           "the parent is the domain $(p.id), nested at refinement $(case.nest.refinement)" =>
               refinement_ratio(run.nest) == case.nest.refinement && run.parent === p.state,
           "every advected tracer (and ql, qr) is in the nest's field set" =>
               Set(vcat(c.from_parent, c.inflow)) ==
               Set(t.name for t in case.tracers if t.advect || t.name in (:ql, :qr)),
           "qv takes the parent's values" => !_moist(p.state) || :qv in c.from_parent,
           "child: net boundary volume flux after the correction ≤ 1e-12 U A" =>
               abs(led.residual) <= 1e-12 * U * A,
           # (the sound-proof cores constrain ∇·(W u); their solver coupling checks it)
           "child: max|∇·u| ≤ 1e-8 s⁻¹ (Boussinesq core)" =>
               !(sim isa BoussinesqSimulation) || max_divergence(sim) <= 1e-8,
           "child and parent are driven by one geostrophic wind" => geo_same,
           "the parent's tiled surface: fractions sum to 1 in every column" => tiles,
           "the sponge targets the parent's mean wind over the child" =>
               c.targets === nothing || c.targets.u === nothing ||
               (sim.sponge_target.u === c.targets.u && any(!iszero, c.targets.u))]
    # the parent's budgets close with the nest exchange named (when it measures them)
    b = p.budgets
    if b !== nothing
        for q in intersect(b.quantities, (:energy, :water))
            sb = stage_budget(b.budgets.measured[q])
            scale = maximum(abs, values(sb.terms); init = 0.0)
            push!(out, "the parent's $q budget closes (≤ 1e-10 of its largest term)" =>
                       scale == 0 || abs(sb.closure) <= 1e-10 * scale)
        end
    end
    return out
end

# The two-way checks (#267): the feedback ran, the interface mass flux
# matches after the flux correction, and the re-projected parent is
# divergence-free.
function _two_way_checks(c::NestingCoupling, case)
    p, run = c.parent, c.run
    return ["two-way: the run is two-way, with the registers of θ and qv" =>
                run.feedback === :two_way && (!c.opts.reflux || (run.regs !== nothing &&
                    run.regs.names == (_moist(p.state) ? [:θ, :qv] : [:θ]))),
            "two-way: ≥ 1 feedback" => run.stats.parent_steps >= 1,
            "two-way: interface mass-flux mismatch after the flux correction ≤ 1e-12 (of U A)" =>
                get(run.nest.ledger, :mass, Inf) <= 1e-12,
            "two-way: the re-projected parent's max|∇·u| ≤ 1e-8 s⁻¹" =>
                max_divergence(p.state) <= 1e-8]
end

# The outflow options of the nest faces (N-open-c, #260), and the two-way
# option (#267; buffer 1 and no canopy restriction, so the smoke child,
# which covers only 4 × 4 × 2 parent cells, is anterpolated at all); every
# coupling on.
smoke_variants(::Type{<:NestingCoupling}) =
    ((label = "two-way", feedback = :two_way, options = Dict(:nesting => (buffer = 1, z_anterp = 0.0))),
     (label = "convective outflow", outflow = :convective),
     (label = "parent_driver: analytic boundary data in place of the parent (the #58 interface)",
      options = Dict(:nesting => (parent_driver = :profile,)), parent = false),
     (label = "nest off: the periodic box", physics = Dict(:nesting => false), parent = false))
