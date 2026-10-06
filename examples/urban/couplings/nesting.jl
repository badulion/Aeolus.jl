# The Berlin nest (N-berlin-nest, #265): the case runs inside an Aeolus
# parent, one-way, by default.
#
#   parent    the deep diurnal Aeolus run of N-parent-e (#259,
#             examples/urban/parent/): Boussinesq on a StretchedGrid, the
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
#             (`Nesting.end_parent_step!`; one-way: clock and statistics —
#             the two-way feedback is N-berlin-twoway, #267).
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
one-way nest inside the Berlin parent run (see the file header and
`docs/src/design/berlin_case.md`, "Nesting"). Options:

* `mode` — `:one_way` (the only mode here; `:two_way` is N-berlin-twoway, #267);
* `parent` — a `NamedTuple` of [`ParentConfig`](@ref) keywords (empty:
  the #259 default, 6 km × 6 km × 3 km at refinement (2, 2, 4); the
  `:smoke` scenario sets the 1 km × 1 km × 500 m parent at (4, 4, 8)).

Fields: the parent configuration `pc`, the [`ParentCase`](@ref) `parent`
and the `NestedRun` `run` (built by `init!`), the open
parent `window` (`begin_parent_step!`'s result, or `nothing`), the parent
step `pending` that `limit_dt` picked for the next window, the tracer
names of the field set (`from_parent`: carried by the parent; `inflow`:
zero on inflow), and statistics.
"""
mutable struct NestingCoupling <: AbstractCoupling
    mode::Symbol
    pc::Any                       # ParentConfig (examples/urban/parent/config.jl)
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
end

coupling_name(::Type{<:NestingCoupling}) = :nesting
modules_used(::Type{<:NestingCoupling}) = (:Nesting, :StretchedGrids, :SurfaceTiles, :LargeScaleForcing)
default_options(::Type{<:NestingCoupling}) = (mode = :one_way, parent = (;))
stages(::Type{<:NestingCoupling}) = ((:nest_parent, 10, EVERY_STEP), (:nest_feedback, 10, EVERY_STEP))

"""
    nest_parent_config(cfg) -> ParentConfig

The parent of the nested case `cfg`: [`ParentConfig`](@ref)`(cfg)` with
`options[:nesting].parent` (the `:smoke` scenario's small parent, or
the #259 default), centred on the child.
"""
function nest_parent_config(cfg::CaseConfig)
    o = coupling_options(NestingCoupling, cfg)
    return ParentConfig(cfg; verbose = false, o.parent...)
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
    o.mode === :one_way ||
        error("nesting: mode = :$(o.mode) — the Berlin nest is one-way here; the two-way " *
              "option is N-berlin-twoway (#267)")
    b = nest_case_bcs(cfg, mesh)
    # the sponge below the open top relaxes toward the parent's mean wind
    # over the child: level vectors, filled at every parent window
    cfg.sponge_α_max > 0 || return (bcs = b, θ_bcs = b)
    return (bcs = b, θ_bcs = b, sponge_target = (u = zeros(mesh.Nz), v = zeros(mesh.Nz), w = 0.0))
end

build(::Type{<:NestingCoupling}, case, o) =
    NestingCoupling(o.mode, nest_parent_config(case.cfg), nothing, nothing, nothing, NaN,
                    Symbol[], Symbol[], 0, 0.0, 0.0, nothing)

init!(c::NestingCoupling, case) = (ensure_nest!(c, case); nothing)

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
    p = build_parent(c.pc; child = case, io = devnull)
    pm, cm = p.mesh, case.mesh
    off = (round(Int, (cm.x0 - pm.x0) / pm.dx) + 1, round(Int, (cm.y0 - pm.y0) / pm.dy) + 1, 1)
    nest = NestedGrid(pm, cm; refinement = c.pc.refinement, offset = off)
    # the field set: θ (always), and the tracers
    zp = Field(CellCenter(), pm)                     # the value of a tracer the parent lacks
    pairs = Pair{Symbol,Any}[]
    for t in case.tracers
        (t.advect || t.name in (:ql, :qr)) || continue
        pf = _parent_scalar(p, t.name)
        push!(pairs, t.name => (t.field, pf === nothing ? zp : pf))
        push!(pf === nothing ? c.inflow : c.from_parent, t.name)
    end
    isempty(c.from_parent) || add_open_scalars!(sim.bcs, c.from_parent; mode = :value)
    isempty(c.inflow) || add_open_scalars!(sim.bcs, c.inflow; mode = :inflow)
    state = ntuple(_ -> nest_interp_state(nest; scalars = pairs), 3)
    run = NestedRun(p.sim, sim, nest;
                    advance_parent = domain_advance((s, h; kw...) -> (step_parent!(p, h); s)),
                    advance_child = domain_advance((s, h; kw...) -> (_child_advance!(case, h); s)),
                    cfl_parent = nothing, cfl_child = nothing, feedback = :one_way,
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
    Aeolus.Nesting._fill_nest_halos!(sim, run.s_old)
    st = sim.sponge_target
    c.targets = (u = st.u isa AbstractVector ? st.u : nothing,
                 v = st.v isa AbstractVector ? st.v : nothing,
                 θ = case.open === nothing || case.open.damping === nothing ? nothing :
                     case.open.damping.target)
    _update_targets!(c, case)
    c.wall_build += (time_ns() - t0) * 1e-9
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
    if c.window === nothing
        isnan(c.pending) && (c.pending = parent_dt(c.parent))
        R = c.pending
    else
        R = c.window.t_child_end - case.sim.t
    end
    R > 0 || return dt
    return subcycle_dt(R, dt)
end

function run_stage!(c::NestingCoupling, case, ::Val{:nest_parent}, dt)
    ensure_nest!(c, case)
    t = case.sim.t
    if c.window === nothing
        Δt_p = isnan(c.pending) ? _fixed_window(c, dt) : max(c.pending, dt)
        tw = time_ns()
        c.window = begin_parent_step!(c.run; dt = Δt_p)
        c.wall_parent += (time_ns() - tw) * 1e-9
        c.pending = NaN
        c.windows += 1
        _update_targets!(c, case)
    end
    w = c.window
    t + dt <= w.t_child_end + 1e-9 * w.Δt ||
        error("nesting: the child step [$t, $(t + dt)] crosses the end of the parent window " *
              "$(w.t_child_end); with a fixed step, dt must divide the window (it does when " *
              "the windows start on a child step: use the adaptive stepper otherwise)")
    return nothing
end

function run_stage!(c::NestingCoupling, case, ::Val{:nest_feedback}, dt)
    w = c.window
    w === nothing && return nothing
    if abs(case.sim.t - w.t_child_end) <= 1e-9 * w.Δt
        end_parent_step!(c.run, w)
        c.window = nothing
    end
    return nothing
end

function report(c::NestingCoupling, case)
    c.run === nothing && return "  nesting: parent not built"
    st, ob, t = c.run.stats, case.open, max(case.sim.t, eps())
    return @sprintf("  nesting (one-way): %d parent windows, dt_p %.2f–%.2f s, parent %.1f s wall; boundary heat sides %.2f, top %.2f W/m² (mean), max|δ| %.1e m/s",
                    st.parent_steps + (c.window === nothing ? 0 : 1), st.dt_parent_min, st.dt_parent_max,
                    c.wall_parent, ob === nothing ? NaN : ob.lateral / t, ob === nothing ? NaN : ob.top / t,
                    ob === nothing ? NaN : ob.δmax)
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
    return ["the parent ran: ≥ 1 window, the child inside the last one" =>
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

# The outflow options of the nest faces (N-open-c, #260); every coupling on.
smoke_variants(::Type{<:NestingCoupling}) =
    ((label = "convective outflow", outflow = :convective),
     (label = "nest off: the periodic box", physics = Dict(:nesting => false)))

register!(NestingCoupling)
