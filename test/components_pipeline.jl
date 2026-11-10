# The component interface and the fixed stage pipeline (ONE-a, #450):
# src/Components.jl, src/components/*.jl.
#
#   * the pipeline: stage order (euler and the three substages of RK3),
#     cadence, the order within a stage, validation, the hook-free fast path;
#   * the core's components against the built-in terms, on both
#     cores, and their physics: inertial oscillation (Coriolis), the θ
#     budget of the prescribed fluxes, the decay rate of the sponge;
#   * component contracts: build from a section, a step on both cores, the
# checkpoint round trip (a restart is the uninterrupted run).

using Test
using TOML
using Random: Xoshiro
using Aeolus
using Aeolus.Components
const CPC = Aeolus.Components          # the interface's generic names stay qualified (serial runtests share Main)
using Aeolus.Config: ConfigError
using Aeolus.Fields: set!, interior_view
using Aeolus.Operators: sponge_profile, damping_rate, damping_coordinates
using Aeolus.Anelastic: AnelasticSimulation, isentropic_reference
using Aeolus.IO: checkpoint!, restart!, checkpoint_state, restore_checkpoint_state!


# ---------------------------------------------------------------------------
# test components
# ---------------------------------------------------------------------------

const CP_ATTACHABLE = (Aeolus.Components.PRE_CORE..., Aeolus.Components.IN_CORE...,
                       Aeolus.Components.POST_CORE...)

"Records the stage of every call, and the dt it got."
struct CPRecorder <: AbstractComponent
    log::Vector{Symbol}
    dts::Vector{Float64}
    st::Tuple
end
CPRecorder(st = map(Stage, CP_ATTACHABLE)) = CPRecorder(Symbol[], Float64[], st)
Aeolus.Components.component_name(::Type{CPRecorder}) = :recorder
Aeolus.Components.section(::Type{CPRecorder}) = "recorder"
Aeolus.Components.stages(c::CPRecorder) = c.st
Aeolus.Components.run_stage!(c::CPRecorder, d, ::Val{S}, dt) where {S} = (push!(c.log, S); push!(c.dts, dt); nothing)

"A component with a slot and a name of choice (validation and order)."
struct CPSlotted{S,N} <: AbstractComponent end
Aeolus.Components.component_name(::Type{CPSlotted{S,N}}) where {S,N} = N
Aeolus.Components.section(::Type{CPSlotted{S,N}}) where {S,N} = string(N)
Aeolus.Components.slot(::Type{CPSlotted{S,N}}) where {S,N} = S
struct CPNeedsCoriolis <: AbstractComponent end
Aeolus.Components.component_name(::Type{CPNeedsCoriolis}) = :needs_coriolis
Aeolus.Components.section(::Type{CPNeedsCoriolis}) = "needs_coriolis"
Aeolus.Components.requires(::Type{CPNeedsCoriolis}) = (Coriolis,)
Aeolus.Components.incompatible(::Type{CPNeedsCoriolis}) = (Sponge,)
"A body force that adds a constant to u (to compare the hook with step!'s keyword)."
struct CPPush <: AbstractComponent
    a::Float64
end
Aeolus.Components.component_name(::Type{CPPush}) = :push
Aeolus.Components.section(::Type{CPPush}) = "push"
Aeolus.Components.stages(::CPPush) = (Stage(:body_force),)
Aeolus.Components.run_stage!(c::CPPush, d, ::Val{:body_force}, dt) = (interior_view(d.state.tu) .+= c.a; nothing)

# ---------------------------------------------------------------------------
# small domains
# ---------------------------------------------------------------------------

const CP_MESH = RegularGrid(; size = (8, 8, 8), extent = (800.0, 800.0, 400.0), halo = (2, 2, 2))
cp_bcs() = (bcs = FieldBoundaryConditions(:periodic, :periodic, :wall),
            θ_bcs = FieldBoundaryConditions(:periodic, :periodic, :noflux))

"A Boussinesq or an anelastic core on CP_MESH (free-slip z walls), kwargs to the core."
function cp_core(core::Symbol; kw...)
    core === :boussinesq && return BoussinesqSimulation(CP_MESH; cp_bcs()..., kw...)
    return AnelasticSimulation(CP_MESH; ref = isentropic_reference(CP_MESH; θ_const = 300.0), cp_bcs()..., kw...)
end

"A domain on `sim` with the components built from `specs` (`C => opts`), plus the advection."
function cp_domain(sim, specs...; scheme = :euler, extra = ())
    d0 = Domain(sim; scheme = scheme, validate = false)
    return Domain(d0; components = (CentredAdvection(), CPC.build_components(specs, d0)..., extra...))
end

"A random, divergent-free-enough initial state (the projection of the first step fixes it)."
function cp_state!(sim; seed = 3)
    rng = Xoshiro(seed)
    for f in (sim.u, sim.v, sim.w)
        interior_view(f) .= 0.5 .* (rand(rng, size(interior_view(f))...) .- 0.5)
    end
    interior_view(sim.θ) .+= 0.2 .* rand(rng, size(interior_view(sim.θ))...)
    return sim
end

cp_bits(sim) = [copy(f.data) for f in (sim.u, sim.v, sim.w, sim.θ, sim.p)]   # compared within ROUNDOFF_RTOL (#495)

const CP_COR = (; f = 1.0e-3)                                  # [coriolis]
const CP_SPO = (alpha_max = 0.02, z_start = 250.0)             # [sponge]
const CP_BND = (bottom = 0.1, top = -0.03)                     # [surface_theta_flux]
const CP_SGS = (model = :smagorinsky, c_s = 0.17, pr_t = 1 / 3, stability = :none)

# ---------------------------------------------------------------------------

@testset "pipeline: the stages, the slots and Stage" begin
    names = map(last, PIPELINE)
    @test length(unique(names)) == length(names)
    @test issorted(map(first, PIPELINE))                      # the design's numbering, 1 … 13
    @test map(first, PIPELINE)[[1, end]] == (1, 13)
    @test (Aeolus.Components.PRE_CORE..., Aeolus.Components.IN_CORE..., :open_boundaries, :projection,
           Aeolus.Components.POST_CORE...) == names           # the three groups cover the pipeline in order
    @test SLOTS == (:advection, :sgs, :microphysics, :radiation, :surface)
    @test Stage(:radiation; every = 60).every == 60.0
    @test_throws ArgumentError Stage(:no_such_stage)
    @test_throws ArgumentError Stage(:radiation; every = -1)
end

@testset "pipeline: one Euler step runs every stage once, RK3 the core stages three times" begin
    for (core_, scheme) in ((:boussinesq, :euler), (:anelastic, :rk3))
        rec = CPRecorder()
        d = cp_domain(cp_core(core_; ν = 1.0), extra = (rec,); scheme = scheme)
        step!(Model(d), 0.5)
        incore = scheme === :rk3 ? repeat(collect(Aeolus.Components.IN_CORE), 3) :
                                   collect(Aeolus.Components.IN_CORE)
        @test rec.log == vcat(collect(Aeolus.Components.PRE_CORE), incore, collect(Aeolus.Components.POST_CORE))
        @test all(==(0.5), rec.dts)                       # SSP-RK3 substages are Euler steps of dt
        @test CPC.clock(d) == 0.5
    end
end

@testset "pipeline: cadence" begin
    rec = CPRecorder((Stage(:radiation; every = 0.25), Stage(:output)))
    d = cp_domain(cp_core(:boussinesq; ν = 1.0), extra = (rec,))
    for _ in 1:8
        step!(d, 0.125)
    end
    # every 0.25 s: the steps that reach t = 0.25, 0.5, 0.75, 1.0
    @test count(==(:radiation), rec.log) == 4 && count(==(:output), rec.log) == 8
    @test rec.log[1:3] == [:output, :radiation, :output]
end

@testset "pipeline: the order within a stage — slot first, then name" begin
    cs = (CPSlotted{nothing,:zeta}(), CPSlotted{nothing,:alpha}(), CPSlotted{:sgs,:a_sgs}(),
          CPSlotted{:advection,:z_adv}(), CPSlotted{:surface,:b_surf}())
    @test map(CPC.component_name, CPC.order_components(cs)) == (:z_adv, :a_sgs, :b_surf, :alpha, :zeta)
    # the domain's tuple is concretely typed and ordered
    d = cp_domain(cp_core(:boussinesq), Sponge => CP_SPO, Coriolis => CP_COR, SGS => CP_SGS,
                  SurfaceThetaFlux => CP_BND)
    @test map(CPC.component_name, d.components) == (:advection, :sgs, :coriolis, :sponge, :surface_theta_flux)
    @test isconcretetype(typeof(d.components))
end

"A component with a name and stages of choice (the order within a stage)."
struct CPOrdered{N} <: AbstractComponent
    st::Tuple
    log::Vector{Symbol}
end
cpo(N::Symbol, st::Stage...; log = Symbol[]) = CPOrdered{N}(st, log)
Aeolus.Components.component_name(::Type{CPOrdered{N}}) where {N} = N
Aeolus.Components.section(::Type{CPOrdered{N}}) where {N} = string(N)
Aeolus.Components.stages(c::CPOrdered) = c.st
Aeolus.Components.run_stage!(c::CPOrdered{N}, d, ::Val{S}, dt) where {N,S} = (push!(c.log, N); nothing)

@testset "pipeline: a stage's declared order, then slot and name (ONE-e)" begin
    # within :radiation the declared order puts c before b before a, in
    # :sources c before a; the slot-and-name order (a, b, c) bends to both
    a = cpo(:a, Stage(:radiation; order = 20), Stage(:sources; order = 30))
    b = cpo(:b, Stage(:radiation; order = 10))
    c = cpo(:c, Stage(:radiation; order = 5), Stage(:sources; order = 10))
    o = CPC.order_components((a, b, c))
    @test map(CPC.component_name, o) == (:c, :b, :a)
    # no declared order: the slot-and-name order is unchanged
    @test map(CPC.component_name, CPC.order_components((cpo(:z, Stage(:radiation)),
                                                         cpo(:y, Stage(:radiation))))) == (:y, :z)
    # orders that contradict each other across stages are refused
    x = cpo(:x, Stage(:radiation; order = 1), Stage(:sources; order = 2))
    y = cpo(:y, Stage(:radiation; order = 2), Stage(:sources; order = 1))
    @test_throws ArgumentError CPC.order_components((x, y))
    # the pipeline runs a stage in that order
    log = Symbol[]
    a2 = cpo(:a, Stage(:radiation; order = 20); log = log)
    c2 = cpo(:c, Stage(:radiation; order = 5); log = log)
    d = cp_domain(cp_core(:boussinesq; ν = 1.0), extra = (a2, c2))
    step!(d, 0.1)
    @test log == [:c, :a]
end

@testset "pipeline: the first step of a cadenced stage, and the :transport stage" begin
    rec = CPRecorder((Stage(:radiation; every = 1.0, first = true), Stage(:transport)))
    d = cp_domain(cp_core(:boussinesq; ν = 1.0), extra = (rec,))
    for _ in 1:4
        step!(d, 0.4)                               # t = 0.4, 0.8, 1.2, 1.6
    end
    # radiation on the step from t = 0 (first) and on the one that crosses t = 1
    @test count(==(:radiation), rec.log) == 2
    @test count(==(:transport), rec.log) == 4
    # :transport runs after :chemistry and before :microphysics
    names = map(last, PIPELINE)
    @test findfirst(==(:chemistry), names) < findfirst(==(:transport), names) < findfirst(==(:microphysics), names)
    @test Stage(:radiation; every = 8, first = true, order = 5).order == 5
end

@testset "pipeline: validation names the sections" begin
    err(f) = try f(); nothing catch e; e isa ConfigError ? e : rethrow() end
    adv = CentredAdvection()
    e = err(() -> CPC.validate_components((adv, CPSlotted{:sgs,:one}(), CPSlotted{:sgs,:two}())))
    @test e !== nothing && any(p -> p.key == "two" && occursin(":sgs slot", p.message), e.problems)
    e = err(() -> CPC.validate_components((adv, CPSlotted{:turbulence,:x}())))
    @test any(p -> occursin("unknown slot", p.message), e.problems)
    e = err(() -> CPC.validate_components((CPSlotted{nothing,:x}(),)))
    @test any(p -> p.key == "advection", e.problems)                     # no :advection provider
    e = err(() -> CPC.validate_components((adv, CPNeedsCoriolis(), Sponge(0.1, 1.0, (u = 0, v = 0, w = 0)))))
    msgs = [p.message for p in e.problems]
    @test any(m -> occursin("requires Coriolis ([coriolis])", m), msgs)
    @test any(m -> occursin("incompatible with sponge ([sponge])", m), msgs)
    @test CPC.validate_components((adv, CPNeedsCoriolis(), Coriolis(1e-4))) === nothing
    e = err(() -> CPC.validate_components((adv, CPRecorder((Stage(:projection),)))))
    @test any(p -> occursin(":projection", p.message), e.problems)
    e = err(() -> CPC.validate_components((adv, CPRecorder((Stage(:body_force; every = 10),)))))
    @test any(p -> occursin("takes no cadence", p.message), e.problems)
    e = err(() -> CPC.validate_components((adv, CPRecorder(), CPRecorder())))
    @test any(p -> occursin("two components named", p.message), e.problems)
    @test_throws ArgumentError Model((cp_domain(cp_core(:boussinesq)), cp_domain(cp_core(:boussinesq))))
    # both schemes and the adaptive step on both cores (ONE-j, #460; the
    # anelastic core was Euler with a fixed dt before)
    @test Domain(cp_core(:anelastic); scheme = :rk3, components = (adv,)).scheme === :rk3
    @test Domain(cp_core(:anelastic); cfl = Aeolus.TimeIntegration.CFLConstraints(), components = (adv,)).cfl !== nothing
    @test_throws ArgumentError Domain(cp_core(:anelastic); scheme = :ab2, components = (adv,))
end

@testset "pipeline: the hook-free fast path when nothing attaches to stages 6-10" begin
    # a DNS domain: no hooks, the blocked sweep of SC-e runs (it allocates its buffers)
    sim = cp_core(:boussinesq; ν = 1.0); cp_state!(sim)
    d = cp_domain(sim)
    h = CPC.core_hooks(d)
    @test h.body_force! === nothing && h.post_predictor! === nothing && h.sgs === nothing
    @test sim.workspaces.sweep === nothing
    step!(d, 0.1)
    @test sim.workspaces.sweep !== nothing
    # a stage-6 component without a fused form turns it off; one with a
    # fused form (Coriolis, since #533 step 2) and a stage-11 one do not
    sim = cp_core(:boussinesq; ν = 1.0); cp_state!(sim)
    d = cp_domain(sim; extra = (CPRecorder((Stage(:tendencies),)),))
    @test CPC.core_hooks(d).body_force! !== nothing
    step!(d, 0.1)
    @test sim.workspaces.sweep === nothing
    sim = cp_core(:boussinesq; ν = 1.0); cp_state!(sim)
    d = cp_domain(sim, Coriolis => CP_COR)
    @test CPC.core_hooks(d).body_force! !== nothing
    step!(d, 0.1)
    @test sim.workspaces.sweep !== nothing
    sim = cp_core(:boussinesq; ν = 1.0); cp_state!(sim)
    d = cp_domain(sim; extra = (CPRecorder((Stage(:surface_update),)),))
    step!(d, 0.1)
    @test sim.workspaces.sweep !== nothing
end

@testset "components = the core's built-in terms ($c, $sch)" for (c, sch) in ((:anelastic, :rk3),)
    # the built-in path: the core's own Coriolis, θ fluxes, sponge and Smagorinsky
    ref = cp_state!(cp_core(c; ν = 0.5, f_coriolis = CP_COR.f, C_s = 0.17,
                            θ_flux_bottom = CP_BND.bottom, θ_flux_top = CP_BND.top,
                            sponge_α_max = CP_SPO.alpha_max, sponge_z_start = CP_SPO.z_start,
                            (c === :boussinesq ? (; θ₀ = 300.0) : (;))...))
    sim = cp_state!(cp_core(c; ν = 0.5, (c === :boussinesq ? (; θ₀ = 300.0) : (;))...))
    d = cp_domain(sim, SGS => CP_SGS, Coriolis => CP_COR, Sponge => CP_SPO, SurfaceThetaFlux => CP_BND;
                  scheme = sch)
    @test Aeolus.Components.core(d) === c
    for _ in 1:4
        sch === :rk3 ? step_rk3!(ref, 2.0) : step!(ref, 2.0)
        step!(Model(d), 2.0)
    end
    @test values_agree(cp_bits(sim), cp_bits(ref))
    @test sim.t === ref.t
    @test maximum(abs, interior_view(sim.w)) > 0           # it moved
    # a body-force component is step!'s body_force! keyword
    ref = cp_state!(cp_core(c; ν = 0.5))
    sim = cp_state!(cp_core(c; ν = 0.5))
    d = cp_domain(sim; scheme = sch, extra = (CPPush(1e-3),))
    kw = (; body_force! = (s, Δt) -> (interior_view(s.tu) .+= 1e-3; nothing))
    for _ in 1:3
        sch === :rk3 ? step_rk3!(ref, 2.0; kw...) : step!(ref, 2.0; kw...)
        step!(d, 2.0)
    end
    @test values_agree(cp_bits(sim), cp_bits(ref))
end

@testset "Coriolis: the inertial oscillation ($c)" for c in (:boussinesq,)
    # a uniform (u, v) under f only: forward Euler maps u + iv to
    # (1 − i f Δt)(u + iv) exactly (a clockwise rotation at f, amplitude
    # √(1 + (fΔt)²) per step); the projection of a uniform field is zero
    f = 1.0e-3; Δt = 50.0; n = 40; U0 = 3.0
    sim = cp_core(c); set!(sim.u, (x, y, z) -> U0)
    d = cp_domain(sim, Coriolis => (; f = f))
    for _ in 1:n
        step!(d, Δt)
    end
    w = (1 - im * f * Δt)^n * U0
    @test all(x -> isapprox(x, real(w); rtol = 1e-12), interior_view(sim.u))
    @test all(x -> isapprox(x, imag(w); rtol = 1e-12, atol = 1e-12), interior_view(sim.v))
    @test angle(complex(interior_view(sim.u)[1], interior_view(sim.v)[1])) ≈ -n * atan(f * Δt)
    begin
        # SSP-RK3 over one inertial period 2π/f: back to (U0, 0), amplitude
        # error O((fΔt)⁴) per step (the scheme's |R(iz)|² = 1 − z⁴/12 + z⁶/36);
        # on both cores (the anelastic SSP-RK3 is ONE-j's, #460)
        sim = cp_core(c); set!(sim.u, (x, y, z) -> U0)
        d = cp_domain(sim, Coriolis => (; f = f); scheme = :rk3)
        m = 126; Δ = 2π / f / m
        for _ in 1:m
            step!(d, Δ)
        end
        u, v = interior_view(sim.u)[1], interior_view(sim.v)[1]
        z = f * Δ
        @test hypot(u, v) / U0 ≈ (1 - z^4 / 12 + z^6 / 36)^(m / 2) rtol = 1e-10
        @test abs(atan(v, u)) < 1e-3
    end
    # the adaptive limit (on both cores: the anelastic adaptive step is ONE-j's)
    dc = cp_domain(cp_core(c), Coriolis => (; f = -4.0e-3))
    @test CPC.limit_dt(only(filter(x -> x isa Coriolis, collect(dc.components))), dc) === nothing  # fixed step
    da = Domain(cp_core(c); cfl = Aeolus.TimeIntegration.CFLConstraints(; C_advective = 0.5, dt_max = 1e3),
                components = (CentredAdvection(), Coriolis(-4.0e-3)))
    @test CPC.limit_dt(da.components[2], da) == (0.5 / 4.0e-3, :coriolis)
    res = CPC.step_adaptive!(Model(da))
    @test res[2] == 0.5 / 4.0e-3                                   # a fluid at rest: Coriolis binds
    @test Aeolus.Simulations.step_record(da.state).limiter === :coriolis
end

@testset "SurfaceThetaFlux: Σ W θ ΔV changes by (W_bot F_bottom − W_top F_top) A t ($c)" for c in (:anelastic,)
    # at rest: only the fluxes act; the anelastic core is buoyant (θ̄ is its
    # reference), so it takes one step from rest, the Boussinesq one ten.
    # W = 1 on the Boussinesq core: the domain-mean θ changes by
    # (F_bottom − F_top) t / Lz. On the anelastic core W = ρ̄ (ONE-j, #460),
    # and the wall faces carry the weight of the adjacent cell, so F/Δz is
    # the weighted flux form
    sim = cp_core(c); c === :boussinesq && set!(sim.θ, (x, y, z) -> 300.0)
    θ0 = copy(interior_view(sim.θ))
    d = cp_domain(sim, SurfaceThetaFlux => CP_BND)
    n = c === :boussinesq ? 10 : 1
    for _ in 1:n
        step!(d, 5.0)
    end
    H = CP_MESH.Hz
    Wc, Wf = c === :boussinesq ? (ones(8 + 2H), ones(9 + 2H)) : Aeolus.Anelastic.constraint_weights(sim)
    # the sum of the change (the difference of two sums of θ ≈ 300 K loses
    # five digits to cancellation)
    dθ = interior_view(sim.θ) .- θ0
    got = sum(Wc[H+k] * sum(dθ[:, :, k]) for k in 1:8) * 50.0 / 8^2         # Σ W δθ Δz per column
    @test isapprox(got, (Wf[H+1] * CP_BND.bottom - Wf[H+9] * CP_BND.top) * 5.0n; rtol = 1e-10)
    θ = interior_view(sim.θ)
    @test θ[:, :, 1] != θ[:, :, 4] && θ[:, :, 8] != θ[:, :, 4]   # through the bottom and the top levels
    @test !CPC.enabled(SurfaceThetaFlux, (bottom = 0.0, top = 0.0))
    @test CPC.enabled(SurfaceThetaFlux, (bottom = 0.0, top = -0.01))
end

@testset "Sponge: u decays as (1 − α(z)Δt)ⁿ above z_start, nothing below ($c)" for c in (:boussinesq,)
    U0 = 2.0; Δt = 5.0; n = 12
    sim = cp_core(c); set!(sim.u, (x, y, z) -> U0)
    d = cp_domain(sim, Sponge => CP_SPO)
    for _ in 1:n
        step!(d, Δt)
    end
    zs = damping_coordinates(CP_MESH, FaceX())
    p = sponge_profile(CP_SPO.alpha_max, CP_SPO.z_start, 400.0)
    u = interior_view(sim.u)
    @test all(k -> all(x -> isapprox(x, U0 * (1 - Δt * damping_rate(p, zs[k]))^n; rtol = 1e-12), u[:, :, k]), 1:8)
    @test all(==(U0), u[:, :, findall(<(CP_SPO.z_start), zs)])
    @test u[1, 1, 8] < 0.6U0                                     # the top level: α ≈ α_max
    # the sponge is off when α = 0 or z_start is "none"; a z_start at or above
    # the top is refused (the model's configuration reports it, ONE-b)
    @test !CPC.enabled(Sponge, (alpha_max = 0.0, z_start = 100.0))
    @test !CPC.enabled(Sponge, (alpha_max = 0.02, z_start = nothing))
    @test CPC.enabled(Sponge, (alpha_max = 0.02, z_start = 400.0))
    @test_throws ArgumentError CPC.build(Sponge, (alpha_max = 0.02, z_start = 400.0), d)
end

@testset "component contracts ($c)" for c in (:boussinesq,)
    sim = cp_state!(cp_core(c; ν = 0.5))
    d = cp_domain(sim, SGS => CP_SGS, Coriolis => CP_COR, Sponge => CP_SPO, SurfaceThetaFlux => CP_BND)
    contract(comp, T = typeof(comp)) =
        CPC.section(T) isa String && !isempty(CPC.section(T)) &&
        Aeolus.Components.schema(T) isa Aeolus.Config.Section && Aeolus.Components.schema(T).name == CPC.section(T) &&
        all(s -> s isa Stage, CPC.stages(comp)) &&
        isempty(CPC.outputs(comp)) && isempty(CPC.checks(comp)) && CPC.check_names(T) == () &&
        CPC.budget!(comp, d, nothing) === nothing &&
        # stateless or not, the checkpoint state restores into the component
        restore_checkpoint_state!(comp, checkpoint_state(comp)) === comp
    @test all(contract, d.components)
    @test keys(CPC.checkpoint_components(d)) == (:advection, :sgs, :coriolis, :sponge, :surface_theta_flux)
    step!(d, 2.0)
    @test all(f -> all(isfinite, interior_view(f)), (sim.u, sim.v, sim.w, sim.θ))
    @test maximum(interior_view(sim.νt)) > 0                      # the :sgs component ran
    # the closures of [sgs]
    bc(o) = CPC.sgs_closure(cp_domain(cp_core(c), SGS => merge(CP_SGS, o)))
    @test bc((; model = :vreman, c_s = nothing)) isa VremanClosure
    @test bc((; model = :amd, c_s = nothing)).C == 0.3
    @test bc((; stability = :brown_et_al_1994)) isa StabilityCorrected
    @test bc((; model = :none)) === nothing
end

@testset "checkpoint round trip: a restart through the components is the uninterrupted run ($c)" for c in (:anelastic,)
    # the Lagrangian dynamic closure carries memory (its averaged LM, MM);
    # on both cores (the anelastic checkpoint is ONE-j's, #460)
    mk() = (s = cp_state!(cp_core(c; ν = 0.5, (c === :boussinesq ? (; θ₀ = 300.0) : (;))...));
            d0 = Domain(s; validate = false);
            Domain(d0; components = (CentredAdvection(), SGS(LagrangianDynamicSmagorinsky(CP_MESH)),
                                     CPC.build_components((Coriolis => CP_COR, Sponge => CP_SPO), d0)...)))
    d = mk()
    for _ in 1:3; step!(d, 2.0); end
    path = joinpath(mktempdir(), "ck.jls")
    checkpoint!(d.state, path; components = CPC.checkpoint_components(d))
    for _ in 1:3; step!(d, 2.0); end
    d2 = mk()
    restart!(d2.state, path; components = CPC.checkpoint_components(d2))
    for _ in 1:3; step!(d2, 2.0); end
    @test values_agree(cp_bits(d2.state), cp_bits(d.state))
    # a restart into another closure is refused
    d3 = Domain(Domain(cp_core(c; ν = 0.5, (c === :boussinesq ? (; θ₀ = 300.0) : (;))...); validate = false);
                components = (CentredAdvection(), SGS(StaticSmagorinsky(0.17)), Coriolis(1e-3), Sponge(0.02, 250.0, (u = 0, v = 0, w = 0))))
    @test_throws ArgumentError restart!(d3.state, path; components = CPC.checkpoint_components(d3))
end
