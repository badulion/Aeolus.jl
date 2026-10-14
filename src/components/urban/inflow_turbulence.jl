# Synthetic inflow turbulence on the nest faces (N-inflow-e, #268).
#
# The child of the Berlin nest takes its lateral faces from a parent at
# 4-8× its spacing: the parent's eddies arrive, its sub-filter eddies do
# not, and an inflow face fed only with the parent's field enters the
# child under-resolved (N-inflow-d, #266: with no generator a laminar
# inflow stays laminar over 15 boundary-layer depths). This addon
# superposes the synthetic turbulence of `Aeolus.Transition` on the
# inflow parts of the four lateral faces:
#
#   faces      Transition.InflowTurbulence (N-inflow-c, #264) wraps the
#              faces' boundary data (the parent's NestBoundaryData, or the
#              driver that replaces the parent) and adds its perturbations
#              inside update_open_boundaries!, before the mass-flux
#              correction, at every stage time: velocity only, ramped on by
#              the face-mean inflow, its per-level mean removed (zero added
#              mass flux), linear in time over a parent window. Nothing is
#              copied into the step cores.
#   generator  `generator`: :digital_filter (default; Xie & Castro 2008,
#              DigitalFilterGenerator) or :fsd (the random-mode field of
#              FreeStreamDisturbance, Transition.free_stream_generator,
#              swept at the parent's mean wind over the child).
#   statistics `statistics`: :parent_resolved (default; the parent's
#              resolved + SGS stresses and lengths in the scale gap of the
#              nest refinement, ParentResolvedStatistics, N-inflow-b #258)
#              or :parametrized (the similarity profiles of a parent that
#              resolves no turbulence, ParametrizedStatistics — the path a
#              mesoscale parent of #58 takes — with u* and the Obukhov
#              length of the child's surface and the parent's z_i).
#   no parent  faces fed by a driver instead of a running parent
#              (parent_driver = :profile or :offline, or nesting off with
#              lateral = :open) have no resolved statistics: the generator
#              is installed there only with statistics = :parametrized; with
#              the default it is not, and smoke_check says so. (The offline
#              child replays the parent's faces, not the online child's
#              perturbations, which are drawn from the running parent.)
#   parent     ParentResolvedStatistics reads the parent's fields. The
#              nest steps the parent of the next window on another task
#              while the child steps (SC-c, #403), so the statistics are
#              folded from the nest's window hook (the parent at rest),
#              never from the child's refresh.
#   two-way    the feedback skips an inflow band of
#              ⌈2 max L_s / Δx_p⌉ parent cells (exclude_inflow_feedback!,
#              N-nest-e #263), and the interface flux correction leaves
#              the generator's normal perturbation out of the parent's
#              interface faces (synthetic_normal_velocity, #418). On the
#              idealised 2 h sheared-CBL nest (test/era6_inflow_nest_longrun.jl)
#              two-way with the generator is within 10 % of one-way in KE
#              and 0.01 K in overlap θ (child KE ×12 before #418). In
#              Berlin the two-way nest is unvalidated (nesting.jl); the
#              default nest is one-way; the two-way campaign is #331.
#   ledger     budget! books the perturbation kinetic energy the mean
#              inflow carries in, ρ⟨½|u'|² max(−U_n, 0)⟩ (W m⁻² of
#              perturbed face, injected_ke_flux), as the ledger term
#              :inflow_ke (J m⁻²). The added mass flux is zero by
#              construction; smoke_check asserts it every step.
#
# No lateral open face (nest off, periodic box): nothing is installed.

"""
    InflowTurbulenceCoupling

Additive coupling `:inflow_turbulence` (on by default): synthetic
turbulence on the inflow parts of the child's lateral open faces (see
the file header and `docs/src/design/berlin_nesting.md`). Options:

* `generator` — `:digital_filter` (default, `DigitalFilterGenerator`) or
  `:fsd` (`FreeStreamInflow`, `free_stream_generator`);
* `statistics` — `:parent_resolved` (default, `ParentResolvedStatistics`
  of the Aeolus parent) or `:parametrized` (`ParametrizedStatistics`);
  faces without a running Aeolus parent (a `parent_driver` other than
  `:aeolus`, or the open box without the nest) get the generator only
  with `:parametrized`;
* `u_min` — the inflow ramp (m s⁻¹): perturbations scale with
  ``\\min(1, -U_n/u_{\\min})`` per level and face (default 0.5);
* `seed` — the generators' seed (face `q` takes `seed + q - 1`);
* `statistics_interval` — parent seconds between two folds of the
  parent statistics (default 60);
* `length_interval` — seconds between two rebuilds of the
  digital-filter kernels from the current lengths (default 3600);
* `length_scale` — `:parametrized`: the single integral length (m);
  `NaN` (default) takes 8 child spacings, the shortest scale the child
  grid carries without dissipating it in a few steps (Kadasch et al.
  2021).

Fields: the options, the installed `Transition.InflowTurbulence` `it`
(`nothing` without a lateral open face, or without a parent to measure
with the default statistics), the source the statistics came from
(`stats`: `:parent_resolved`, `:parametrized` or `:none`), the two-way `exclude` band (parent
cells), the running ledgers (`ke` J m⁻², `qmax` the largest relative
added flux of any face and step, `nsteps`).
"""
mutable struct InflowTurbulenceCoupling <: UrbanComponent
    opts::Any
    it::Any                    # Transition.InflowTurbulence, or nothing
    stats::Symbol              # :parent_resolved / :parametrized / :none
    exclude::Int               # two-way: the inflow band (parent cells), 0 one-way
    ke::Float64                # ∫ injected KE flux dt (J m⁻²)
    ke_last::Float64           # the last step's KE flux (W m⁻²)
    qmax::Float64              # max over faces and steps of |added flux| / mean-flow flux
    nsteps::Int
end

component_name(::Type{<:InflowTurbulenceCoupling}) = :inflow_turbulence
modules_used(::Type{<:InflowTurbulenceCoupling}) = (:Transition, :Nesting)
default_options(::Type{<:InflowTurbulenceCoupling}) =
    (generator = :digital_filter, statistics = :parent_resolved, u_min = 0.5, seed = 1,
     statistics_interval = 60.0, length_interval = 3600.0, length_scale = NaN)
# after the nest (init_priority -1), which builds the faces' data
init_priority(::Type{<:InflowTurbulenceCoupling}) = 0

function build(::Type{<:InflowTurbulenceCoupling}, case, o)
    o.generator in (:digital_filter, :fsd) ||
        error("inflow_turbulence: generator = :$(o.generator); use :digital_filter (default) or :fsd")
    o.statistics in (:parent_resolved, :parametrized) ||
        error("inflow_turbulence: statistics = :$(o.statistics); use :parent_resolved (default) or :parametrized")
    return InflowTurbulenceCoupling(o, nothing, :none, 0, 0.0, 0.0, 0.0, 0)
end

# The lateral faces of the child that are open (west, east, south, north).
_open_lateral(sim) = [s for s in (:west, :east, :south, :north)
                      if getproperty(sim.bcs, s) isa Aeolus.BoundaryConditions.OpenBoundary]

# The child's surface scales for the parametrized statistics: the mean u*
# and kinematic heat flux of the columns with a surface (the bus of the
# last surface step).
function _surface_scales(case)
    bus = case.bus
    have = [I for I in eachindex(bus.surface) if bus.surface[I] !== nothing]
    isempty(have) && return (u_star = 0.0, θ_flux = 0.0)
    return (u_star = mean(bus.u_star[have]), θ_flux = mean(bus.H[have]) / ref_ρcp(case, 1))
end

"""
    inflow_parametrized_statistics(case, o, parent) -> ParametrizedStatistics

The `:parametrized` statistics of the Berlin child: `u_star` and the
Obukhov length (`SurfaceExchange.obukhov_length`, θ_v reference
`cfg.θ₀`) from the child's surface at the time of each evaluation, `z_i`
the parent's (`parent_zi`, minimum heat flux) when there is an Aeolus
parent and it is positive, else the initial inversion height
([`initial_inversion_height`](@ref)), and the length `o.length_scale`
(`NaN`: 8 child spacings).
"""
function inflow_parametrized_statistics(case, o, parent)
    m, cfg = case.mesh, case.cfg
    ℓ = isnan(o.length_scale) ? 8 * max(m.dx, m.dy) : Float64(o.length_scale)
    zi0 = initial_inversion_height(cfg)
    zi(t) = (z = parent === nothing ? NaN : parent_zi(parent); isfinite(z) && z > 0 ? z : zi0)
    us(t) = _surface_scales(case).u_star
    function L(t)
        s = _surface_scales(case)
        (s.u_star > 0 && s.θ_flux != 0) || return Inf
        return Aeolus.SurfaceExchange.obukhov_length(s.u_star, -s.θ_flux / s.u_star, cfg.θ₀)
    end
    return ParametrizedStatistics(; u_star = us, z_i = zi, L_Ob = L, length_scale = ℓ)
end

# The generator argument of InflowTurbulence.
function _inflow_generator(case, o)
    o.generator === :digital_filter && return DigitalFilterGenerator
    sim = case.sim
    U = hypot(sum(interior_view(sim.u)) / length(interior_view(sim.u)),
              sum(interior_view(sim.v)) / length(interior_view(sim.v)))
    return free_stream_generator(; U∞ = max(U, o.u_min))
end

function init!(c::InflowTurbulenceCoupling, case)
    o, sim = c.opts, case.sim
    isempty(_open_lateral(sim)) && return nothing
    nc = find_coupling(case, :nesting)
    run = nc === nothing ? nothing : nc.run
    nested = run isa NestedRun
    # no running parent: nothing to measure the default statistics on
    (o.statistics === :parent_resolved && !nested) && return nothing
    gen = _inflow_generator(case, o)
    common = (generator = gen, u_min = o.u_min, seed = o.seed, length_interval = o.length_interval)
    if o.statistics === :parent_resolved
        p = nc.parent
        # folded from the nest's window hook (the parent at rest), never by the refresh
        c.it = InflowTurbulence(run; common..., statistics_interval = Inf,
                                statistics_kwargs = (closure = Components.sgs_closure(p),))
        it, iv = c.it, Float64(o.statistics_interval)
        push!(nc.window_hooks, (n, cs) -> fold_inflow_statistics!(it; interval = iv))
    else
        st = inflow_parametrized_statistics(case, o, nested ? nc.parent : nothing)
        c.it = InflowTurbulence(nested ? run : sim; common..., statistics = st)
    end
    c.stats = o.statistics
    c.exclude = nested ? exclude_inflow_feedback!(run, c.it) : 0
    return nothing
end

function budget!(c::InflowTurbulenceCoupling, case, l)
    c.it === nothing && return nothing
    ke = injected_ke_flux(c.it).total
    c.ke_last = ke
    c.ke += ke * l.dt
    add_term!(l, :inflow_ke, ke * l.dt)
    c.qmax = max(c.qmax, maximum(a.relative for a in added_mass_flux(c.it)))
    c.nsteps += 1
    return nothing
end

"""
    diagnostics(c::InflowTurbulenceCoupling, case)

`inflow_ke_flux` (W m⁻², the perturbation kinetic energy the mean inflow
carries through the perturbed faces at the last refresh, area mean over
the four faces), `inflow_ke` (J m⁻², its integral over the run, the
ledger term `:inflow_ke`) and `inflow_added_flux_max` (the largest
added volume flux of a face over its mean-flow flux, any step: zero to
round-off by construction). None without a lateral open face.
"""
function diagnostics(c::InflowTurbulenceCoupling, case)
    c.it === nothing && return Diagnostic[]
    return Diagnostic[
        Diagnostic("inflow_ke_flux", "W m-2", () -> c.ke_last;
                   description = "perturbation KE flux through the inflow faces"),
        Diagnostic("inflow_ke", "J m-2", () -> c.ke; description = "perturbation KE carried in, run total"),
        Diagnostic("inflow_added_flux_max", "1", () -> c.qmax;
                   description = "largest added volume flux of a face / its mean-flow flux")]
end

function smoke_check(c::InflowTurbulenceCoupling, case)
    sim = case.sim
    if isempty(_open_lateral(sim))
        return ["no lateral open face (periodic box): no generator installed" => c.it === nothing]
    end
    nc = find_coupling(case, :nesting)
    run = nc === nothing ? nothing : nc.run
    if c.opts.statistics === :parent_resolved && !(run isa NestedRun)
        return ["no running parent: no resolved statistics, no generator (statistics = :parametrized installs it)" =>
                    c.it === nothing && all(st -> !(st.data isa InflowTurbulence), open_boundary_states(sim))]
    end
    it = c.it
    it === nothing && return ["the generator was installed on the open faces" => false]
    # per face: the newest knot is zero on every level its ramp left off
    # (outflow), and the applied perturbation is nonzero on some inflow
    # face (the applied one blends two knots linearly in time, so a level
    # that just turned to outflow still carries the older knot's share)
    var_in = 0.0; off_zero = true
    for f in it.faces
        P = inflow_perturbation(it, f.side)
        for k in axes(P[1], 2)
            if f.ramp[k] > 0
                var_in += sum(abs2, view(P[1], :, k)) + sum(abs2, view(P[2], :, k)) + sum(abs2, view(P[3], :, k))
            else
                off_zero &= all(c -> all(iszero, view(f.Lc[c], :, k)), 1:3)
            end
        end
    end
    l = case.budgets
    checks = [
        "the open faces' data is the generator (wrapping the nest's or driver's data)" =>
            all(st -> st.data === it, open_boundary_states(sim)),
        "statistics: $(c.stats), generator: $(c.opts.generator)" =>
            (c.stats === :parent_resolved) == (first(it.faces).stats isa ParentResolvedStatistics) &&
            (c.opts.generator === :fsd) == (first(it.faces).gen isa FreeStreamInflow),
        "added mass flux ≤ 1e-12 of the mean-flow flux, every face and step" =>
            c.nsteps >= 1 && c.qmax <= 1e-12,
        "the injected KE flux is booked: ledger term :inflow_ke = ∫ flux dt > 0" =>
            get(l.terms, :inflow_ke, NaN) == c.ke && c.ke > 0,
        "nonzero perturbation variance on an inflow face" => var_in > 0,
        "the newest perturbation knot is zero on every outflow level of every face" => off_zero]
    if run isa NestedRun && run.twoway !== nothing
        pm = run.nest.parent_mesh
        need = ceil(Int, feedback_exclude_width(it) / min(pm.dx, pm.dy) - 1e-9)
        push!(checks, "two-way: the feedback skips the inflow band, ⌈2 max L_s/Δx_p⌉ = $need parent cells" =>
                          run.twoway.exclude_width >= need && c.exclude == run.twoway.exclude_width)
    end
    return checks
end

# The generator × statistics options (the default, digital filter with the
# parent's statistics, is the base run); the two-way nest with the generator
# is the nesting coupling's "two-way" variant (every addon on).
smoke_variants(::Type{<:InflowTurbulenceCoupling}) =
    ((label = "inflow: FSD generator, parent statistics",
      options = Dict(:inflow_turbulence => (generator = :fsd,))),
     (label = "inflow: digital filter, parametrized statistics",
      options = Dict(:inflow_turbulence => (statistics = :parametrized,))),
     (label = "inflow: FSD generator, parametrized statistics",
      options = Dict(:inflow_turbulence => (generator = :fsd, statistics = :parametrized))),
     (label = "inflow: parametrized statistics on the analytic boundary data (no parent)",
      options = Dict(:nesting => (parent_driver = :profile,),
                     :inflow_turbulence => (statistics = :parametrized,))))

