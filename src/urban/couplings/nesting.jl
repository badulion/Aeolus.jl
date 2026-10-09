# The Berlin nest (N-berlin-nest, #265): the case runs inside an Aeolus
# parent, one-way by default; two-way is an option (N-berlin-twoway, #267,
# unvalidated until the stability campaign #331).
#
#   parent    the deep diurnal Aeolus run of N-parent-e (#259,
#             src/urban/parent/): Boussinesq on a StretchedGrid, the
#             tiled urban surface (SurfaceTiles), column radiation, the
#             geostrophic pressure gradient, θ/qv nudging, θ damping, the
#             fixed-gradient θ top and the TKE closure — its own hooks,
#             passed to the nest as the parent's `advance!` closure
#             (`Nesting.domain_advance`). `:summer`: the 6 km × 6 km × 3 km
#             default, refinement (2, 2, 4); `:smoke`: 1 km × 1 km × 500 m on
#             16 × 16 × 10 cells, refinement (4, 4, 8) (the scenario's
#             `options[:nesting].parent`, keywords of `ParentConfig`).
#   faces     the child's four sides and its top are `OpenBoundary` faces fed
#             by a `Nesting.NestBoundaryData` (N-nest-d #261): `sim_options`
#             returns them (`nest_bcs`), so `case.jl` keeps one definition
#             of the core and the periodic box and the θ `:initial_gradient`
#             top are not built at all. The outflow condition is
#             `cfg.outflow` (Dirichlet from the parent by default).
#   time      `:nest_parent` (after `:clock`) opens a parent window
#             `[t_p, t_p + Δt_p]` (`Nesting.begin_parent_step!`): the parent
#             takes its own adaptive step, its field set is prolongated at
#             both ends; the child's stepper lands on the window end
#             (`limit_dt` → `Nesting.subcycle_dt`) and every core refreshes
#             the nest faces at its stage times (`update_open_boundaries!`,
#             the parent blended linearly in time). `:nest_feedback` (after
#             `:column_tendency`) closes the window when the child reaches it
#             (`Nesting.end_parent_step!`; one-way: clock and statistics;
#             two-way, `mode = :two_way`: the feedback onto the parent).
#   ahead     (SC-c, #403) one-way, under `run_case!`: when a window opens
#             and the run will step past its end, the parent's step of the
#             next window starts on its own task (`Nesting.prefetch_parent_step!`)
#             and runs while the child steps through this window; the next
#             `:nest_parent` takes it. Its step size is the one the
#             sequential run picks there (`parent_dt` of the parent at the
#             window end; a fixed child step: the whole number of child
#             steps), and the result is bit-identical. `wall.parent` is then
#             the time the child still waited for the parent.
#   two-way   (#267) the child's field set is anterpolated onto the covered
#             parent cells outside a buffer of `buffer` parent cells next to
#             the open faces and above `z_anterp` (`:canopy`: the mean
#             height of the built columns), the parent's interface faces
#             take the child's flux, θ and qv are refluxed from flux
#             registers and the parent is re-projected
#             (`Nesting.TwoWayFeedback`). Both cores are driven by the case
#             (the parent's `step_parent!`, the child's stage loop), so every
#             refluxed scalar is booked by its transport
#             (`book_nest_flux!`): the parent's θ and qv from its dynamics
#             stages, the child's θ from a `:body_force` entry, the child's
#             qv by the `:scalar_advection` slot; each child step is bracketed
#             by `begin_child_step!` / `end_child_step!`. The parent ledgers
#             book the feedback as `(:nest_feedback, :feedback_heat)` and
#             `(:nest_feedback, :feedback_water)`.
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
#             child, refreshed every window.
#
# The parent and the nest are built in `init!`, after every coupling is
# built, so every tracer the couplings register is in the nest's field set,
# and the built case is the whole initial state (the child's wind is the
# parent's). Measured: 2.9 s and ≈ 1.5 GB for the 6 km parent of the
# flagship (after compilation), 0.2–0.4 s for the smoke parent.

"""
    NestingCoupling

Additive coupling `:nesting` (on by default): the case is the child of a
nest inside the Berlin parent run, one-way by default (see the file
header and `docs/src/design/berlin_nesting.md`). Options:

* `mode` — `:one_way` (default) or `:two_way` (#267; **unvalidated**: the
  long-run stability campaign is #331, and only after it validates
  two-way does it become the default): the child is fed back onto the
  parent once per parent window (`Nesting.TwoWayFeedback`);
* `buffer` — two-way: parent cells next to every open face of the child
  that are not anterpolated (default 2, PALM's);
* `z_anterp` — two-way: parent cells whose lower face lies below it are
  not anterpolated (the canopy restriction); `:canopy` (default) is the
  mean height of the child's built columns, or a height in m (`-Inf`:
  none);
* `τ` — two-way: relaxation time of the anterpolation (`nothing`,
  default: direct replacement, as in PALM);
* `reflux` — two-way: Berger–Colella refluxing of θ and qv (default
  `true`);
* `parent` — a `NamedTuple` of [`ParentConfig`](@ref) keywords (empty:
  the #259 default, 6 km × 6 km × 3 km at refinement (2, 2, 4); the
  `:smoke` scenario sets the 1 km × 1 km × 500 m parent at (4, 4, 8));
* `parent_driver` — `:aeolus` (default: the Aeolus parent run), or any
  `AbstractBoundaryData` (the interface a mesoscale reader of #58 or a
  reanalysis of #57 implements), which then feeds the same nest faces
  instead of a parent run (no windows; the tracers keep their mirror
  halos); `:profile` is the analytic `case_boundary_data` of the case
  (the geostrophic wind and the initial θ), the smoke variant of this
  path.

Fields: the parent configuration `pc`, the [`ParentCase`](@ref) `parent`
and the `NestedRun` `run` (built by `init!`), the open
parent `window` (`begin_parent_step!`'s result, or `nothing`), the parent
step `pending` that `limit_dt` picked for the next window, the tracer
names of the field set (`from_parent`: carried by the parent; `inflow`:
zero on inflow), statistics, and `window_hooks`: functions `f(c, case)`
called whenever a parent window opens, after the parent's step and
before its next one starts ahead on another task — the one point where
another coupling may read the parent's fields (the synthetic inflow's
statistics, `couplings/inflow_turbulence.jl`, #268).
"""
mutable struct NestingCoupling <: AbstractCoupling
    mode::Symbol
    opts::Any                     # the coupling's options (the two-way keywords)
    driver::Any                   # :aeolus, or the AbstractBoundaryData that replaces the parent
    pc::Any                       # ParentConfig (src/urban/parent/config.jl)
    parent::Any                   # ParentCase (built by init!)
    run::Any                      # Nesting.NestedRun, or nothing
    window::Any                   # the open parent window, or nothing
    pending::Float64              # Δt_p picked for the next window (NaN: none)
    from_parent::Vector{Symbol}
    inflow::Vector{Symbol}
    windows::Int
    wall_parent::Float64          # s in the parent's steps
    wall_build::Float64           # s building the parent and the nest
    targets::Any                  # (u, v, θ) level vectors the relaxations use, or nothing
    feedback::Dict{Symbol,Float64}   # two-way: what the feedbacks did (J m⁻², kg m⁻², counts)
    feed::Any                     # offline: the FileBoundaryFeed of cfg.dynamic, or nothing
    inputs::Any                   # offline: the DynamicInputs of cfg.dynamic
    k::Int                        # offline: the record the next window starts from
    set::Vector{Any}              # offline: the child fields of the field set's tracers, in order
    recorder::Any                 # online with cfg.child_boundaries: the NestBoundaryRecorder
    ahead_step::Float64           # the child step a prefetched fixed-step window was sized for (NaN: none)
    ahead_window::Float64         # that window's parent step
    window_hooks::Vector{Any}     # f(c, case) at every window opening, the parent at rest (#268)
end

coupling_name(::Type{<:NestingCoupling}) = :nesting
modules_used(::Type{<:NestingCoupling}) = (:Nesting, :StretchedGrids, :SurfaceTiles, :LargeScaleForcing)
default_options(::Type{<:NestingCoupling}) =
    (mode = :one_way, parent = (;), parent_driver = :aeolus,
     buffer = 2, z_anterp = :canopy, τ = nothing, reflux = true)
# the child's θ booking of the two-way registers runs in every dynamics stage
stages(::Type{<:NestingCoupling}) = ((:nest_parent, 10, EVERY_STEP), (:nest_feedback, 10, EVERY_STEP),
                                     (:body_force, 5, EVERY_STEP))

"""
    nest_parent_config(cfg) -> ParentConfig

The parent of the nested case `cfg`: [`ParentConfig`](@ref)`(cfg)` with
`options[:nesting].parent` (the `:smoke` scenario's small parent, or
the #259 default), centred on the child.
"""
function nest_parent_config(cfg::CaseConfig)
    o = coupling_options(NestingCoupling, cfg)
    return ParentConfig(cfg; o.parent...)
end

"""
    nest_geostrophic_wind(cfg) -> (U_g, V_g)

The geostrophic wind of the parent of the nested case `cfg`, functions
`(z, t)` ([`parent_geostrophic`](@ref)): the `:geostrophic` coupling of a
nested child is driven by it, so the large-scale pressure gradient is one
field across the nest.
"""
nest_geostrophic_wind(cfg::CaseConfig) = parent_geostrophic(nest_parent_config(cfg))

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
function nest_case_bcs(cfg::CaseConfig, mesh)
    sa = coupling_options(ScalarAdvection, cfg)
    H = Aeolus.AdvectionSchemes.required_halo(Aeolus.AdvectionSchemes.advection_scheme(sa.scheme))
    return nest_bcs(mesh; bottom = :wall, top = :nest, outflow = cfg.outflow, halo = H)
end

function sim_options(::Type{<:NestingCoupling}, cfg, mesh, o)
    o.mode in (:one_way, :two_way) ||
        error("nesting: mode = :$(o.mode); use :one_way (default) or :two_way")
    o.mode === :two_way && o.parent_driver !== :aeolus &&
        error("nesting: mode = :two_way needs the Aeolus parent (parent_driver = :aeolus): " *
              "boundary data from $(repr(o.parent_driver)) cannot take a feedback")
    b = nest_case_bcs(cfg, mesh)
    # the sponge below the open top relaxes toward the parent's mean wind
    # over the child: level vectors, filled at every parent window
    cfg.sponge_α_max > 0 || return (bcs = b, θ_bcs = b)
    return (bcs = b, θ_bcs = b, sponge_target = (u = zeros(mesh.Nz), v = zeros(mesh.Nz), w = 0.0))
end

function build(::Type{<:NestingCoupling}, case, o)
    d = o.parent_driver
    driver = d in (:aeolus, :offline) ? d : d === :profile ? case_boundary_data(case.cfg) :
             d isa AbstractBoundaryData ? d :
             error("nesting: parent_driver must be :aeolus, :profile, :offline or an AbstractBoundaryData, got $(repr(d))")
    return NestingCoupling(o.mode, o, driver, nest_parent_config(case.cfg), nothing, nothing, nothing, NaN,
                           Symbol[], Symbol[], 0, 0.0, 0.0, nothing, Dict{Symbol,Float64}(),
                           nothing, nothing, 1, Any[], nothing, NaN, NaN, Any[])
end

"Whether the nest's faces are fed by boundary data other than the Aeolus parent or its file (`parent_driver`)."
external_driver(c::NestingCoupling) = !(c.driver in (:aeolus, :offline))

"Whether the nest's faces replay a parent-made dynamic driver (`parent_driver = :offline`, CFG-g #362)."
offline(c::NestingCoupling) = c.driver === :offline

"""
    nest_offline(cfg) -> Bool

Whether the nested case `cfg` replays its parent from the parent-made
dynamic driver `cfg.dynamic` (`options[:nesting].parent_driver =
:offline`, CFG-g #362) instead of running it.
"""
nest_offline(cfg::CaseConfig) = case_nested(cfg) && coupling_options(NestingCoupling, cfg).parent_driver === :offline

"Whether the nest feeds the child back onto the parent (`mode = :two_way`, #267)."
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

# The two-way feedback of the Berlin nest. Both cores are driven by the
# case, so every refluxed scalar is booked by its transport: θ (and the
# moist parent's qv) by the dynamics stages, the child's qv by the
# :scalar_advection slot.
function _berlin_feedback(c::NestingCoupling, case, p)
    o = c.opts
    if o.reflux
        ts = timestepper(case)
        (ts isa IMEXStep || (ts isa AdaptiveStep && ts.scheme !== nothing)) &&
            error("nesting: two-way refluxing books the explicit θ diffusion of the child's " *
                  "step!; the IMEX stepper's is implicit. Use reflux = false with it")
    end
    tf = (p.pc.moisture && :qv in c.from_parent) ? (:qv => NestScalarFlux(applied_by = :transport),) : ()
    return TwoWayFeedback(; buffer = o.buffer, z_anterp = nest_z_anterp(case, o.z_anterp), τ = o.τ,
                          reflux = o.reflux, theta_flux = NestScalarFlux(applied_by = :transport),
                          tracer_fluxes = tf)
end

# The parent's share of the two-way bookings, at every dynamics stage of
# step_parent! with the stage's weight: its θ and qv, centred advection and
# the core's diffusion with the parent's closure (a no-op outside a child
# window of a refluxing run).
function _book_parent!(p, s, Δt, w)
    book_nest_flux!(s, :θ, s.θ, w * Δt; scheme = :centered, diffusivity = :theta, sgs = p.closure)
    p.pc.moisture &&
        book_nest_flux!(s, :qv, s.qv, w * Δt; scheme = :centered, diffusivity = :tracer, sgs = p.closure)
    return nothing
end

init!(c::NestingCoupling, case) = (ensure_nest!(c, case); nothing)
# before every other coupling's init!: they initialise from the faces' data
# and the child's wind, which come from the parent
init_priority(::Type{<:NestingCoupling}) = -1

# The parent field a child tracer takes its boundary values from, or nothing.
function _parent_scalar(p, name::Symbol)
    name === :qv && return p.pc.moisture ? p.sim.qv : nothing
    name === :e && return p.closure isa TKEClosure ? p.closure.e : nothing
    return nothing
end

# The child's dynamics over h (the :timestepper provider's advance_dynamics!).
_child_advance!(case, h) =
    (ts = timestepper(case); ts === nothing ? dynamics_step!(case, h) : advance_dynamics!(ts, case, h))

"""
    ensure_nest!(c::NestingCoupling, case) -> c

Build the parent and the nest (once; `init!`): the [`ParentCase`](@ref)
(`build_parent`, with this case as the child of its tiled surface), the
`Nesting.NestedGrid` (the child at its own origin, the parent placed by
`parent_origin`), the field set (θ and every advected tracer, ql and qr;
the planes of the tracers added to the nest faces with
`add_open_scalars!`), and the `Nesting.NestedRun` (one-way, no step
validation — the parent's `step_parent!` retries its own pieces — the
child's solid mask on the faces), which attaches the parent's
`NestBoundaryData` to the child's faces and fills their halos.
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
    p = build_parent(c.pc; child = case)
    pm, cm = p.mesh, case.mesh
    off = (round(Int, (cm.x0 - pm.x0) / pm.dx) + 1, round(Int, (cm.y0 - pm.y0) / pm.dy) + 1, 1)
    nest = NestedGrid(pm, cm; refinement = c.pc.refinement, offset = off)
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
    bookp = two_way(c) && c.opts.reflux ? (s, Δt, w) -> _book_parent!(p, s, Δt, w) : nothing
    run = NestedRun(p.sim, sim, nest;
                    advance_parent = domain_advance((s, h; kw...) -> (step_parent!(p, h; body_force! = bookp); s)),
                    advance_child = domain_advance((s, h; kw...) -> (_child_advance!(case, h); s)),
                    cfl_parent = nothing, cfl_child = nothing, feedback = fb,
                    solid = case.geom.sm, state = state, validate = false)
    c.parent, c.run = p, run
    # the child's initial wind is the parent's, prolongated over the whole
    # child: an initial wind that differs from the faces' (the case's u0
    # against a sounding's geostrophic wind) makes ∇·u ≠ 0 in the boundary
    # cells, and the first step's flux-form θ advection then changes θ there
    # by −θ ∇·u Δt (≈ 290 K in a corner of the smoke window with the summer
    # sounding: 1 m/s inside, 3.4 m/s on the faces). At t = 0 the parent's
    # wind is horizontally uniform, so the prolongated wind is divergence-free.
    s = run.s_new
    interpolate_parent_state!(s, p.sim, nest; order = run.order, region = :all)
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
                        "cannot feed back); mode = :two_way")
    cfg, sim, m = case.cfg, case.sim, case.mesh
    planes = [p for st in open_boundary_states(sim) for p in st.planes]
    ox, oy = anchor_xy(cfg, m)
    fields = Dict{Symbol,Array{Float64,3}}(n => Array{Float64}(interior_view(getproperty(sim, n))) for n in (:u, :v, :w))
    c.recorder = NestBoundaryRecorder(cfg.child_boundaries, planes, m; init_time = case_init_time(cfg),
        fields = fields, profiles = collect(keys(_target_profiles(c))),
        attributes = Dict{String,Any}("nest_from_parent" => join(String.(c.from_parent), " "),
                                      "nest_inflow" => join(String.(c.inflow), " "),
                                      "nest_refinement" => collect(Int, c.pc.refinement)),
        preprocessor = "aeolus run ([output] child_boundaries)",
        origin_lat = cfg.lat0, origin_lon = cfg.lon0, origin_x = ox, origin_y = oy)
    record_boundary!(c.recorder, sim.t, NaN, c.run.data.a; profiles = _target_profiles(c))
    return c
end

# Offline: the faces, the field set, the initial wind and the targets from
# the parent-made driver, in the order the online ensure_nest! sets them.
function _ensure_offline!(c::NestingCoupling, case)
    cfg, sim, m = case.cfg, case.sim, case.mesh
    two_way(c) && error("nesting: parent_driver = :offline replays a one-way nest; mode = :two_way needs the parent")
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
    zp = Float64[pm.zc[pm.Hz+K] for K in 1:pm.Nz]
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
    p, nest = c.parent, c.run.nest
    tg.u === nothing || _covered_profile!(tg.u, p.sim.u, nest, zc)
    tg.v === nothing || _covered_profile!(tg.v, p.sim.v, nest, zc)
    tg.θ === nothing || _covered_profile!(tg.θ, p.sim.θ, nest, zc)
    return c
end

# A fixed child step: the window is a whole number of child steps, at most
# the parent's own limit (at least one child step).
_fixed_window(c::NestingCoupling, dt) =
    dt * max(1, floor(Int, parent_dt(c.parent) / dt * (1 + 1e-9)))

function limit_dt(c::NestingCoupling, case, dt)
    ensure_nest!(c, case)
    external_driver(c) && return dt
    if c.window === nothing
        if offline(c)
            c.k < length(c.feed.time) || return dt          # past the records: run_stage! reports it
            R = c.feed.step[c.k + 1]
        else
            isnan(c.pending) && (c.pending = parent_dt(c.parent))
            R = c.pending
        end
    else
        R = c.window.t_child_end - case.sim.t
    end
    R > 0 || return dt
    return subcycle_dt(R, dt)
end

function run_stage!(c::NestingCoupling, case, ::Val{:nest_parent}, dt)
    ensure_nest!(c, case)
    external_driver(c) && return nothing
    t = case.sim.t
    if c.window === nothing && offline(c)
        _open_offline_window!(c, case)
    elseif c.window === nothing
        Δt_p = _window_dt(c, dt)
        tw = time_ns()
        c.window = begin_parent_step!(c.run; dt = Δt_p)
        c.wall_parent += (time_ns() - tw) * 1e-9
        c.pending = NaN
        c.windows += 1
        _update_targets!(c, case)
        c.recorder === nothing ||
            record_boundary!(c.recorder, c.window.t_child_end, c.window.Δt, c.run.data.b;
                             profiles = _target_profiles(c))
        # what reads the parent between its steps (the inflow statistics, #268)
        for f in c.window_hooks
            f(c, case)
        end
        # the last reads of the parent of this window are done: its next step may start
        _prefetch_parent!(c, case, dt)
    end
    w = c.window
    t + dt <= w.t_child_end + 1e-9 * w.Δt ||
        error("nesting: the child step [$t, $(t + dt)] crosses the end of the parent window " *
              "$(w.t_child_end); with a fixed step, dt must divide the window (it does when " *
              "the windows start on a child step: use the adaptive stepper otherwise)")
    # two-way: the child step's interface fluxes are booked between here and
    # :nest_feedback (the :scalar_advection slot, the :body_force entry)
    two_way(c) && begin_child_step!(c.run)
    return nothing
end

# The parent step of the window that opens now, as the sequential run
# picks it: `parent_dt` of the parent at the window start (`c.pending`, set
# by `limit_dt` under a variable step), or a whole number of fixed child
# steps. A prefetched step was sized by the same rule when it started
# (the parent has moved on since), and `begin_parent_step!` checks the two
# agree.
function _window_dt(c::NestingCoupling, dt)
    if c.run.ahead !== nothing && isnan(c.pending)
        dt == c.ahead_step ||
            error("nesting: the parent step of this window was prefetched for a fixed child step " *
                  "of $(c.ahead_step) s; this step is $dt s")
        return c.ahead_window
    end
    return isnan(c.pending) ? _fixed_window(c, dt) : max(c.pending, dt)
end

# SC-c (#403): start the parent's step of the next window on its own task,
# when it is certain to be taken (one-way, the Aeolus parent, a run whose
# loop goes past this window's end), sized as _window_dt will size it:
# under a variable step limit_dt sets `pending` from the parent at the
# window start and the child's step is never longer (subcycle_dt ≤ R), so
# the window is `pending`; under a fixed step it is _fixed_window of the
# fixed step.
function _prefetch_parent!(c::NestingCoupling, case, dt)
    two_way(c) && return nothing
    Threads.nthreads() > 1 || return nothing
    w = c.window
    step_will_run(case.clock, w.t_child_end) || return nothing
    ts = timestepper(case)
    if ts !== nothing && variable_dt(typeof(ts))
        c.pending = parent_dt(c.parent)
        Δt_next = c.pending
        c.ahead_step = NaN
    else
        Δt_next = _fixed_window(c, dt)
        c.ahead_step = Float64(dt); c.ahead_window = Δt_next
    end
    prefetch_parent_step!(c.run, w; dt = Δt_next)
    return nothing
end

"""
    end_run!(c::NestingCoupling, case)

Wait for a parent step prefetched for a window the run did not reach (a
run stopped early by a blow-up): the parent is then one window ahead.
"""
function end_run!(c::NestingCoupling, case)
    c.run isa NestedRun && wait_parent_step!(c.run)
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
    w = c.window
    w === nothing && return nothing
    two_way(c) && end_child_step!(c.run)
    if abs(case.sim.t - w.t_child_end) <= 1e-9 * w.Δt
        if offline(c)                       # end_parent_step!'s clock: the child lands on the record
            case.sim.t = w.t_child_end
            c.k += 1
        else
            two_way(c) ? _feedback!(c, case, w) : end_parent_step!(c.run, w)
        end
        c.window = nothing
    end
    return nothing
end

# Offline: open the window from record k to k + 1 (begin_parent_step!'s
# window: t0 the child's clock, Δt = time_bc[k+1] − time_bc[k]), refresh the
# faces at its start and the targets to its end, as the online nest does.
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

# Close a two-way window: the feedback onto the parent, its heat and water
# booked in the parent's ledgers as (:nest_feedback, :feedback_heat) and
# (:nest_feedback, :feedback_water) — a zero-length ledger step around it,
# so the parent ledgers close with the feedback named.
function _feedback!(c::NestingCoupling, case, w)
    p = c.parent
    sim = p.sim
    moist = p.pc.moisture
    hour = hour_of_day(p.pc.child, sim.t)
    heat_begin_step!(p.heat, sim.θ, hour)
    moist && heat_begin_step!(p.water, sim.qv, hour)
    heat_open!(p.heat, :nest_feedback, :feedback_heat, sim.θ)
    moist && heat_open!(p.water, :nest_feedback, :feedback_water, sim.qv)
    H0 = sum(p.heat.H_now)
    W0 = moist ? sum(p.water.H_now) : 0.0
    tw = time_ns()
    end_parent_step!(c.run, w)
    c.feedback[:wall] = get(c.feedback, :wall, 0.0) + (time_ns() - tw) * 1e-9
    heat_close!(p.heat, sim.θ)
    moist && heat_close!(p.water, sim.qv)
    heat_end_step!(p.heat, sim.θ, 0.0)
    moist && heat_end_step!(p.water, sim.qv, 0.0)
    c.feedback[:heat] = get(c.feedback, :heat, 0.0) + (sum(p.heat.H_now) - H0)
    c.feedback[:water] = get(c.feedback, :water, 0.0) + (moist ? sum(p.water.H_now) - W0 : 0.0)
    c.feedback[:n] = get(c.feedback, :n, 0.0) + 1
    led = c.run.nest.ledger
    c.feedback[:mass] = max(get(c.feedback, :mass, 0.0), get(led, :mass, 0.0))
    return c
end

"""
    diagnostics(c::NestingCoupling, case)

With the Aeolus parent: `parent_windows` (parent steps begun),
`dt_parent_min` / `dt_parent_max` (s); the heat the open faces carried,
`boundary_heat_sides` / `boundary_heat_top` (W m⁻², mean over the run)
and the largest normal-velocity correction `delta_max` (m s⁻¹); a
two-way nest adds `feedback_count`, `feedback_heat` (J m⁻², into the
parent) and `feedback_water` (kg m⁻²). The wall time of the parent run
`wall.parent` and of the feedback `wall.feedback` (s, cumulative). An
offline nest (`parent_driver = :offline`, #362): `parent_windows`
(windows replayed) and `boundary_records` (records read from the
dynamic driver). None for an external `parent_driver`.
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
        Diagnostic("parent_windows", "1", () -> st().parent_steps + (c.window === nothing ? 0 : 1);
                   cadence = :step),
        Diagnostic("dt_parent_min", "s", () -> st().dt_parent_min),
        Diagnostic("dt_parent_max", "s", () -> st().dt_parent_max),
        Diagnostic("boundary_heat_sides", "W m-2", perrun(o -> o.lateral);
                   description = "heat through the lateral faces, mean over the run"),
        Diagnostic("boundary_heat_top", "W m-2", perrun(o -> o.top);
                   description = "heat through the top face, mean over the run"),
        Diagnostic("delta_max", "m s-1", () -> ob === nothing ? NaN : ob.δmax;
                   description = "largest normal-velocity correction of the open faces"),
        Diagnostic("wall.parent", "s", () -> c.wall_parent; description = "wall time of the parent run")]
    if two_way(c)
        append!(ds, [Diagnostic("feedback_count", "1", () -> get(c.feedback, :n, 0.0)),
                     Diagnostic("feedback_heat", "J m-2", () -> get(c.feedback, :heat, 0.0);
                                description = "heat the feedback put into the parent"),
                     Diagnostic("feedback_water", "kg m-2", () -> get(c.feedback, :water, 0.0);
                                description = "water the feedback put into the parent"),
                     Diagnostic("wall.feedback", "s", () -> get(c.feedback, :wall, 0.0);
                                description = "wall time of the feedback")])
    end
    return ds
end

# --- smoke checks ------------------------------------------------------------

"Relative closure of an `AirHeatLedger` (|closure| / largest term)."
function _ledger_closure(hl)
    b = heat_budget(hl)
    scale = maximum(abs, values(b.terms); init = 0.0)
    return scale == 0 ? 0.0 : abs(b.closure) / scale
end

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
    geo_same = geo === nothing || !(geo.forcing isa GeostrophicPressureGradient) ||
               p.forcing.geostrophic === nothing ||
               all(z -> forcing_value(geo.forcing.U_g, z, tw) == forcing_value(p.forcing.geostrophic.U_g, z, tw) &&
                        forcing_value(geo.forcing.V_g, z, tw) == forcing_value(p.forcing.geostrophic.V_g, z, tw), zc)
    tiles = p.surface === nothing ||
            all(col -> col === nothing || isapprox(sum(mosaic_fractions(col.surface)), 1; atol = 1e-12),
                p.surface.columns)
    w = c.window
    checks = two_way(c) ? _two_way_checks(c, case) : Pair{String,Bool}[]
    return [checks...,
            "the parent ran: ≥ 1 window, the child inside the last one" =>
                c.windows >= 1 && (w === nothing ? abs(sim.t - run.t) <= 1e-9 * max(run.Δt, 1.0) :
                                   w.t_child_end - w.Δt - 1e-9 <= sim.t <= w.t_child_end + 1e-9),
            "the parent is the StretchedGrid run of #259, nested at refinement $(c.pc.refinement)" =>
                p.mesh isa StretchedGrid && refinement_ratio(run.nest) == c.pc.refinement,
            "every advected tracer (and ql, qr) is in the nest's field set" =>
                Set(vcat(c.from_parent, c.inflow)) ==
                Set(t.name for t in case.tracers if t.advect || t.name in (:ql, :qr)),
            "qv takes the parent's values" => :qv in c.from_parent,
            "parent heat ledger closes (≤ 1e-10 of its largest term)" => _ledger_closure(p.heat) <= 1e-10,
            "parent water ledger closes (≤ 1e-10 of its largest term)" =>
                !p.pc.moisture || _ledger_closure(p.water) <= 1e-10,
            "child: net boundary volume flux after the correction ≤ 1e-12 U A" =>
                abs(led.residual) <= 1e-12 * U * A,
            # (the sound-proof cores constrain ∇·(W u); their solver coupling checks it)
            "child: max|∇·u| ≤ 1e-8 s⁻¹ (Boussinesq core)" =>
                !(sim isa BoussinesqSimulation) || max_divergence(sim) <= 1e-8,
            "child and parent are driven by one geostrophic wind" => geo_same,
            "the parent's surface is the tile mosaic (fractions sum to 1)" => tiles,
            "the sponge targets the parent's mean wind over the child" =>
                c.targets === nothing || c.targets.u === nothing ||
                (sim.sponge_target.u === c.targets.u && any(!iszero, c.targets.u))]
end

# The two-way checks (#267): the feedback ran and changed the parent, its
# heat and water are named terms of the parent ledgers (whose closure the
# checks above assert), the interface mass flux matches after the flux
# correction, and the re-projected parent is divergence-free.
function _two_way_checks(c::NestingCoupling, case)
    p, run = c.parent, c.run
    hb = heat_budget(p.heat)
    wb = p.pc.moisture ? heat_budget(p.water) : nothing
    return ["two-way: the run is two-way, with the transport-booked registers of θ and qv" =>
                run.feedback === :two_way && (!c.opts.reflux || (run.regs !== nothing &&
                    run.regs.names == (p.pc.moisture ? [:θ, :qv] : [:θ]))),
            "two-way: ≥ 1 feedback, which changed the parent's θ" =>
                get(c.feedback, :n, 0.0) >= 1 && haskey(hb.terms, (:nest_feedback, :feedback_heat)) &&
                hb.terms[(:nest_feedback, :feedback_heat)] != 0,
            "two-way: the parent water ledger names the feedback" =>
                wb === nothing || haskey(wb.terms, (:nest_feedback, :feedback_water)),
            "two-way: interface mass-flux mismatch after the flux correction ≤ 1e-12 (of U A)" =>
                get(c.feedback, :mass, Inf) <= 1e-12,
            "two-way: the re-projected parent's max|∇·u| ≤ 1e-8 s⁻¹" =>
                max_divergence(p.sim) <= 1e-8]
end

# The outflow options of the nest faces (N-open-c, #260), and the two-way
# option (#267; buffer 1 and no canopy restriction, so the smoke child,
# which covers only 4 × 4 × 2 parent cells, is anterpolated at all); every
# coupling on.
smoke_variants(::Type{<:NestingCoupling}) =
    ((label = "two-way", options = Dict(:nesting => (mode = :two_way, buffer = 1, z_anterp = 0.0))),
     (label = "convective outflow", outflow = :convective),
     (label = "parent_driver: analytic boundary data in place of the parent (the #58 interface)",
      options = Dict(:nesting => (parent_driver = :profile,))),
     (label = "nest off: the periodic box", physics = Dict(:nesting => false)))

register!(NestingCoupling)
