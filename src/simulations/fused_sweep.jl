# The fused sweep of the Boussinesq step core: which terms a substage adds
# to its tendencies, whether all of them run inside the sweep, and the
# report of the substages that fall back to one loop per term (#533).
# Included into `module Simulations`, after `_euler_projection!`.
#
# A substage takes the fused sweep (`FiniteVolume.blocked_predictor!`)
# when every term between the base tendencies and the predictor is a
# `SweepTerm` ([`FiniteVolume.sweep_term`](@ref Aeolus.Operators.FiniteVolume.sweep_term)):
# the built-in ones of the core and the components of stages 6 and 7
# (through the body-force hook, `Components._ForceHook`). Otherwise it runs
# separate loops, one per term (the reference implementation), and the simulation's
# `SweepReport` records which terms caused it, once, with the documented
# reason of `SWEEP_FALLBACKS`.

"""
    SWEEP_FALLBACKS

The documented reasons why a term of the step core does not run inside
the fused sweep yet (#533): `"<module>.<type>" => reason` for a
component (or an SGS closure, a momentum-advection operator) without a
[`sweep_term`](@ref Aeolus.Operators.FiniteVolume.sweep_term), and
`"builtin.<term>" => reason` for a built-in term of the Boussinesq core.
This is the explicit allowlist of the completeness guard
(`test/fused_sweep.jl`): every component that attaches to `:tendencies`,
`:body_force` or `:post_predictor` and every built-in term has a
`sweep_term` or an entry here, and no entry is stale. A run reports the
reasons of the terms that kept it with one loop per term in its
`results.toml` (`[loop_fusion]`, [`sweep_report`](@ref)).
"""
const SWEEP_FALLBACKS = Dict{String,String}(
    # --- built-in terms of `_euler_projection!` ------------------------------
    "builtin.sgs" => "the SGS closure (compute_sgs!, add_sgs_momentum!, add_sgs_scalar!): " *
                     "νt and κh precomputed as inputs of the sweep, plan step 4",
    "builtin.momentum_advection" => "a momentum-advection operator other than the built-in centred one " *
                                    "is a whole-field function; only momentum_advect! is the sweep's base term",
    "builtin.body_force" => "a body_force! hook passed to the core as a function is a whole-field operator; " *
                            "only the pipeline's components can be expressed as sweep terms",
    "builtin.post_predictor" => "post-predictor operators (the post_predictor! hook) run on the predicted " *
                                "field; the delayed-row scheme of the sweep for them is plan step 6",
    "builtin.open_boundaries" => "open lateral faces: the radiating outflow and the boundary data's mass " *
                                 "correction around the predictor are not yet verified inside the sweep",
    "builtin.backend" => "the ring-buffer sweep runs on the CPU backend; the per-point form for a GPU is plan step 8",
    "builtin.core" => "this step core (step_imex!, the anelastic core) has no fused sweep yet, plan step 8",
    "builtin.budgets" => "the measured budgets count the substages and attribute the in-core components " *
                         "on whole tendency fields (budget_open!/budget_close!); booked inside the sweep " *
                         "from plan step 7",
    # --- components of :tendencies / :body_force -----------------------------
    "Components.LargeScale" => "geostrophic forcing, nudging and advective tendencies need horizontal-mean " *
                               "profiles of the stage-start state (prepare!): plan step 3",
    "Components.SurfaceLayer" => "the MOST wall stress and surface fluxes on the first level, its " *
                                 "post-predictor correction: plan steps 5 and 6",
    "Components.KinematicFlow" => "overwrites the predicted velocity with a prescribed flow (post-predictor): " *
                                  "plan step 6",
    "SimulationDriver.SimulationCouplings" => "the obstacles' IBM penalty (a mask, :body_force): plan step 5; " *
                                              "without obstacles the slab's floor flux runs in the sweep",
    # --- the urban site ------------------------------------------------------
    "Urban.UrbanSite" => "captures the top and open-face heat fluxes of the stage-start state, a reduction " *
                         "(prepare!): plan step 5",
    "Urban.GeostrophicCoupling" => "mode = :nudging relaxes the plane means of the stage-start state " *
                                   "(LargeScaleForcing.ProfileNudging): plan step 3; the pressure-gradient " *
                                   "mode runs in the sweep",
    "Urban.WallModelFacades" => "the facade wall model's stress on cells next to walls (masks as inputs) and " *
                                "its post-predictor correction: plan steps 5 and 6",
    "Urban.NestingCoupling" => "the nest's flux registers and boundary relaxation of the child: plan step 5",
    "Urban.VegDrag" => "implicit vegetation drag on the predicted velocity (post-predictor): plan step 6",
    "Urban.IBMCoupling" => "immersed-boundary forcing on the predicted velocity (post-predictor): plan step 6",
    "Urban.SurfaceExchangeMOST" => "surface exchange on the predicted near-wall cells (post-predictor): plan step 6",
    "Urban.TiledUrbanSurface" => "the tiled surface's fluxes into the predicted first level (post-predictor): " *
                                 "plan step 6",
)

"The key of type `T` in [`SWEEP_FALLBACKS`](@ref): `\"<module>.<type>\"`."
sweep_key(T::Type) = string(nameof(parentmodule(T)), ".", nameof(Base.unwrap_unionall(T)))

"""
    sweep_reason(key::String) -> String
    sweep_reason(T::Type) -> String

The documented reason of [`SWEEP_FALLBACKS`](@ref) why the built-in term
`key` (`"builtin.<term>"`) or the type `T` runs outside the fused sweep.
"""
sweep_reason(key::AbstractString) =
    get(SWEEP_FALLBACKS, key, "no sweep_term and no documented reason in Simulations.SWEEP_FALLBACKS")
sweep_reason(T::Type) = sweep_reason(sweep_key(T))

# one blocker of the fused sweep: what, and why (only built when reporting)
_blocker(key::AbstractString) = (key => sweep_reason(key))
_blocker(T::Type) = (sweep_key(T) => sweep_reason(T))

"""
    hook_terms(hook, sim) -> Union{Tuple,Nothing}

The sweep terms that do what the core's `body_force!` hook adds to the
tendencies of `sim`, in its order, or `nothing` when one of them is not a
[`SweepTerm`](@ref Aeolus.Operators.FiniteVolume.SweepTerm). A plain
function (a user hook) is `nothing`; the pipeline's hook
(`Components._ForceHook`) asks each of its components'
[`sweep_term`](@ref Aeolus.Operators.FiniteVolume.sweep_term).
"""
hook_terms(hook, sim) = nothing

"""
    hook_blockers(hook, sim) -> Vector{Pair{String,String}}

What keeps the core's hook `hook` out of the fused sweep, `key => reason`
([`SWEEP_FALLBACKS`](@ref)); empty when [`hook_terms`](@ref) has its
terms. Called once per simulation, when reporting.
"""
hook_blockers(hook, sim) = hook_terms(hook, sim) === nothing ? [_blocker("builtin.body_force")] : Pair{String,String}[]
"""
    post_hook_blockers(hook, sim) -> Vector{Pair{String,String}}

What keeps the core's `post_predictor!` hook out of the fused sweep (no
post-predictor operator runs inside it before plan step 6).
"""
post_hook_blockers(hook, sim) = [_blocker("builtin.post_predictor")]

# The terms of one substage of `_euler_projection!` if every one of them
# is a sweep term (the base tendencies first, with viscosity `ν` or
# `nothing`), else `nothing`. Type-stable per configuration: a tuple type
# or Nothing.
# (The hooks' types are parameters: an argument of a Function type that is
# only passed on is otherwise not specialised on, and the plan's type
# would be left to a dynamic call; review of #533 step 2.)
function _sweep_terms(sim::BoussinesqSimulation, closure, body_force!::B, post_predictor!::P,
                      momentum_advection, ν) where {B, P}
    Backends.backend(sim.u.data) isa Backends.CPUBackend || return nothing
    has_open_boundaries(sim.bcs) && return nothing
    post_predictor! === nothing || return nothing
    momentum_advection === momentum_advect! || return nothing
    closure === nothing || return nothing          # SGS terms in the sweep: plan step 4
    core = (BaseTendencies(ν), _builtin_terms(sim)...)
    body_force! === nothing && return core
    ht = hook_terms(body_force!, sim)
    ht === nothing && return nothing
    return (core..., ht...)
end

# The built-in pointwise terms of steps 5-6b, in the order of the passes,
# each switched by its flag (one tuple type per simulation type, #533 plan
# step 2).
function _builtin_terms(sim::BoussinesqSimulation)
    g = sim.mesh
    return (OptionalTerm(sim.has_buoyancy, BuoyancyTerm(sim.θ₀, sim.g_accel / sim.θ₀)),
            OptionalTerm(sim.has_coriolis, CoriolisTerm(sim.f_coriolis)),
            OptionalTerm(sim.has_surface_flux, SurfaceFluxTerm(sim.θ_flux_bottom, sim.θ_flux_top, g)),
            OptionalTerm(sim.has_sponge, sponge_term(sim.sponge_α_max, sim.sponge_z_start, _sponge_targets(sim), g)))
end

"""
    SpongeTargets{T}

The targets `u, v, w` of the core's sponge (`sim.sponge_target`) in the
concrete form the fused sweep's damping term reads
([`DampingTargets`](@ref Aeolus.Operators.FiniteVolume.DampingTargets):
each a number, a level vector or a 3-D array of eltype `T`, a `Field` by
its data). Made once per simulation and target (`sim.workspaces.sponge`):
the stored `sponge_target` is a NamedTuple of abstract fields, read here
once instead of every substage. A level vector or array of eltype `T` is
the stored one, so an in-place update of it (the nest's sponge target) is
seen; another eltype is converted once.
"""
struct SpongeTargets{T}
    targets::DampingTargets{T}
    src::NamedTuple
end
function _sponge_targets(sim::BoussinesqSimulation{T}) where {T}
    c = sim.workspaces.sponge
    c isa SpongeTargets{T} && c.src === sim.sponge_target && return c.targets
    return _new_sponge_targets!(sim)
end
@noinline function _new_sponge_targets!(sim::BoussinesqSimulation{T}) where {T}
    tg = sim.sponge_target
    c = SpongeTargets{T}(DampingTargets{T}(tg.u, tg.v, tg.w), tg)
    sim.workspaces.sponge = c
    return c.targets
end

# Every term that keeps a substage off the fused sweep, `key => reason`
# (only when reporting; allocates).
function _sweep_blockers(sim::BoussinesqSimulation, closure, body_force!, post_predictor!, momentum_advection)
    out = Pair{String,String}[]
    Backends.backend(sim.u.data) isa Backends.CPUBackend || push!(out, _blocker("builtin.backend"))
    has_open_boundaries(sim.bcs) && push!(out, _blocker("builtin.open_boundaries"))
    momentum_advection === momentum_advect! || push!(out, _blocker("builtin.momentum_advection"))
    closure === nothing || push!(out, _blocker("builtin.sgs"))
    body_force! === nothing || append!(out, hook_blockers(body_force!, sim))
    post_predictor! === nothing || append!(out, post_hook_blockers(post_predictor!, sim))
    return out
end


# an unfused substage: counted; its blockers recorded and logged once
function _note_unfused!(sim, closure, body_force!, post_predictor!, momentum_advection)
    r = sim.workspaces.fused
    r.unfused += 1
    r.reported || _report_unfused!(r, sim, closure, body_force!, post_predictor!, momentum_advection)
    return nothing
end
# the blocker sets logged so far: each is logged once per process (a run's
# domains share one line when they share the reason; every domain records
# its own in results.toml)
const _LOGGED_FALLBACKS = Set{String}()
const _LOGGED_LOCK = ReentrantLock()
@noinline function _report_unfused!(r::SweepReport, sim, closure, body_force!, post_predictor!, momentum_advection)
    r.fallback = _sweep_blockers(sim, closure, body_force!, post_predictor!, momentum_advection)
    r.reported = true
    who = join((first(p) for p in r.fallback), ", ")
    fresh = lock(() -> who in _LOGGED_FALLBACKS ? false : (push!(_LOGGED_FALLBACKS, who); true), _LOGGED_LOCK)
    fresh && @info("the step core runs one loop per term (not the fused sweep): " * who *
                   " (no fused form; results.toml [loop_fusion])",
                   subsystem = :timestepping, event = :not_fused,
                   reasons = join((string(first(p), ": ", last(p)) for p in r.fallback), "; "))
    return nothing
end

const LOOP_FUSION_SETTINGS = (:if_possible, :always, :never)

"""
    set_loop_fusion!(sim, setting) -> sim

Whether the substages of `sim`'s Boussinesq step core compute all their
tendency terms and the predictor in one fused, cache-blocked pass over
the grid (loop fusion, #533; PALM's `loop_optimization = 'cache'`) or in
one loop per term (PALM's `'vector'`, the reference implementation):

  * `:if_possible` (the default): fused when every term of the substage
    has a fused form ([`sweep_term`](@ref Aeolus.Operators.FiniteVolume.sweep_term)),
    otherwise one loop per term; the terms without a fused form are
    logged once and written to `results.toml` (`[loop_fusion]`);
  * `:always`: fused, or an `ArgumentError` at the first substage that
    names every term without a fused form (PALM stops the same way when a
    scheme is not implemented for its `loop_optimization`);
  * `:never`: one loop per term.

The results agree within round-off either way. Only `step!` and
`step_rk3!` of the Boussinesq core have the fused form; `step_imex!` and
the anelastic core always run one loop per term, so `:always` is refused
for an anelastic core. The TOML key is
`[timestep] loop_fusion` (`"if_possible"`, `"always"`, `"never"`).
"""
function set_loop_fusion!(sim, setting)
    m = Symbol(setting)
    m in LOOP_FUSION_SETTINGS ||
        throw(ArgumentError("loop_fusion must be one of $(join(LOOP_FUSION_SETTINGS, ", ")), got $(repr(setting))"))
    m === :always && !(sim isa BoussinesqSimulation) &&
        throw(ArgumentError("loop_fusion = \"always\": the $(nameof(typeof(sim))) core has no fused form yet " *
                            "(#533 plan step 8); use \"if_possible\" or \"never\""))
    sim.workspaces.fused.loop_fusion = m
    return sim
end

"""
    loop_fusion(sim) -> Symbol

The loop-fusion setting of `sim` ([`set_loop_fusion!`](@ref)).
"""
loop_fusion(sim) = sim.workspaces.fused.loop_fusion

# loop_fusion = :always and a term without a fused form: stop, naming them
@noinline function _refuse_unfused(sim, closure, body_force!, post_predictor!, momentum_advection)
    bl = _sweep_blockers(sim, closure, body_force!, post_predictor!, momentum_advection)
    throw(ArgumentError("loop_fusion = \"always\", but these terms have no fused form yet: " *
                        join((string(first(p), " (", last(p), ")") for p in bl), "; ") *
                        ". Use loop_fusion = \"if_possible\" or \"never\"."))
end

"""
    sweep_report(sim) -> Dict{String,Any}

The `results.toml` table (`[loop_fusion.<domain>]`) of whether the
substages of `sim`'s step core ran fused (#533): `setting` (the
[`set_loop_fusion!`](@ref) setting), `substages_fused` (`"all"`, `"some"` or
`"none"`; `"none"` with the core's reason when the step core has no fused
sweep) and `not_fusable`, the `"term: reason"` of every term that kept a
substage on separate loops (one per term) ([`SWEEP_FALLBACKS`](@ref)). The substage
counts stay in the simulation's [`SweepReport`](@ref)
(`sim.workspaces.fused`): they restart with a resumed session, and a
resumed run's `results.toml` is that of an uninterrupted run.
"""
function sweep_report(sim)
    r = sim.workspaces.fused
    fb = r.fallback
    if r.fused == 0 && r.unfused == 0
        fb = [_blocker("builtin.core")]
    end
    sub = r.unfused == 0 && r.fused > 0 ? "all" : r.fused == 0 ? "none" : "some"
    return Dict{String,Any}("setting" => string(r.loop_fusion), "substages_fused" => sub,
                            "not_fusable" => String[string(first(p), ": ", last(p)) for p in fb])
end

# Steps 2-12 of `_euler_projection!` in the fused sweep when every term of
# the substage is a sweep term; `false` (and the substage counted as
# unfused) otherwise.
function _try_fused!(sim::BoussinesqSimulation{T}, Δt::T, t_bc, stage, closure, body_force!::B,
                     post_predictor!::P, momentum_advection, ν) where {T, B, P}
    setting = sim.workspaces.fused.loop_fusion
    if setting === :never                  # one loop per term, by choice: nothing to report
        sim.workspaces.fused.unfused += 1
        return false
    end
    terms = _sweep_terms(sim, closure, body_force!, post_predictor!, momentum_advection, ν)
    if terms === nothing
        setting === :always && _refuse_unfused(sim, closure, body_force!, post_predictor!, momentum_advection)
        _note_unfused!(sim, closure, body_force!, post_predictor!, momentum_advection)
        return false
    end
    sim.workspaces.fused.fused += 1
    _blocked_projection!(sim, Δt, t_bc, stage, terms)
    return true
end

@inline _prepare_terms!(::Tuple{}, sim, Δt) = nothing
@inline _prepare_terms!(ts::Tuple, sim, Δt) =
    (prepare!(first(ts), sim, Δt); _prepare_terms!(Base.tail(ts), sim, Δt))

# Steps 2-12 of `_euler_projection!` in the fused sweep over `terms`: the
# tendencies and the predictors of u, v, w and of θ (qv) in one sweep
# (`FiniteVolume.blocked_predictor!`), then the projection. The scalars'
# update of step 11 moves before the projection, which does not read them;
# the tendency fields are not used.
function _blocked_projection!(sim::BoussinesqSimulation{T}, Δt::T, t_bc, stage, terms::Tuple) where {T}
    moist = sim.has_moisture
    ii, jj, kk = interior(sim.mesh)
    lag, lz = sweep_lags(terms)
    ns = sweep_slabs(Backends.backend(sim.u.data), length(ii) * length(jj) * length(kk), length(kk); lz = lz)
    bufs = sweep_buffers(sim.workspaces.sweep, sim.u.data, moist ? 5 : 4, ns, length(kk); lag = lag, lz = lz)
    sim.workspaces.sweep = bufs
    _prepare_terms!(terms, sim, Δt)
    _blocked_sweep!(bufs, terms, sim, Δt, stage)
    update_open_boundaries!(sim, t_bc, Δt)
    _project_velocity!(sim, _DivBy(Δt), stage)
    if stage !== nothing                    # the halos of the RK3 combination (step 11)
        _shell_map!(stage.c, (sim.θ.data,), (stage.ws.θ0,), sim.mesh)
        moist && _shell_map!(stage.c, (sim.qv.data,), (stage.ws.q0,), sim.mesh)
    end
    sim.t += Δt
    return sim
end
# the sweep with concrete arguments: the scalars' update is the axpy of
# step 11 (`step!`), or with the RK3 combination and the step-start θ, qv
_blocked_sweep!(bufs, terms, sim, Δt, ::Nothing) =
    _blocked_sweep!(bufs, terms, sim, Δt, _Axpy(Δt), nothing, nothing)
_blocked_sweep!(bufs, terms, sim, Δt, stage) =
    _blocked_sweep!(bufs, terms, sim, Δt, _AxpyStage(_Axpy(Δt), stage.c), stage.ws.θ0, stage.ws.q0)
function _blocked_sweep!(bufs, terms, sim, Δt, ps, θ0, q0)
    s1 = (sim.θ, ps, θ0)
    if sim.has_moisture
        blocked_predictor!(bufs, terms, sim.u, sim.v, sim.w, _Axpy(Δt), s1, (sim.qv, ps, q0), sim.mesh)
    else
        blocked_predictor!(bufs, terms, sim.u, sim.v, sim.w, _Axpy(Δt), s1, nothing, sim.mesh)
    end
    return nothing
end
