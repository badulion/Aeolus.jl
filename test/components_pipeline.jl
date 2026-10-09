# The component interface and the fixed stage pipeline (ONE-a, #450):
# src/Components.jl, src/components/*.jl.
#
#   * acceptance: every `simulation`-config variant of
#     test/support/simulation_digests.jl (the four shipped configs with
#     every SGS closure, Coriolis, the sponge, the surface θ flux, both
#     cores, euler / rk3 / adaptive) reproduces, through Model, the state
#     digests captured on master before the pipeline (bit for bit);
#   * the pipeline: stage order (euler and the three substages of RK3),
#     cadence, the order within a stage, validation, the hook-free fast path;
#   * the core's components against the built-in terms, bit for bit, on both
#     cores, and their physics: inertial oscillation (Coriolis), the θ
#     budget of the prescribed fluxes, the decay rate of the sponge;
#   * component contracts: build from a section, a step on both cores, the
#     checkpoint round trip (a restart is bitwise the uninterrupted run).

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

include(joinpath(@__DIR__, "support", "simulation_digests.jl"))

const CP_DIGESTS = TOML.parsefile(joinpath(@__DIR__, "data", "components", "simulation_digests.toml"))
# Runs whose closure was the core's built-in Smagorinsky (C_s) before ONE-a
# and is the :sgs component's StaticSmagorinsky now: their diagnostics.csv
# gains the three `sgs.*` columns every other closure already had; the other
# columns are the master bytes.
const CP_NEW_SGS_COLUMNS = Set(l for (l, (n, o)) in SD_VARIANTS
                               if n != "taylor_green_vortex" && !any(x -> startswith(x, "sgs.closure"), o) &&
                                  !any(x -> startswith(x, "sgs.stability"), o))

@testset "pipeline acceptance: every simulation-config variant reproduces master's digests ($(length(SD_VARIANTS)) runs, 16³)" begin
    ref = CP_DIGESTS["digests"]
    @test CP_DIGESTS["commit"] == "ead7b986323b519c75681d38399c989ffde964b4"
    @test Set(keys(ref)) == Set(first.(SD_VARIANTS))
    inputs = sd_made_inputs()
    for (label, (name, ovr)) in SD_VARIANTS
        cfg = load_run_config(joinpath(SD_CONFIGS, name * ".toml");
                              overrides = vcat(SD_SMALL, ovr, get(inputs, name, String[])))
        r = run_config(cfg; root = mktempdir(), console = false, repo = mktempdir())
        @test r.obj.model isa Model
        s = state_fields(r.obj)
        for (k, v) in pairs(s)
            @test (label, k, sd_digest(v)) == (label, k, ref[label][string(k)])
        end
        @test (label, sd_digest([CPC.clock(r.obj.model)])) == (label, ref[label]["t"])
        rows = sd_diagnostics(r.folder)
        if label in CP_NEW_SGS_COLUMNS
            hdr = split(rows[1], ',')
            keep = findall(h -> !startswith(h, "sgs."), hdr)
            @test length(hdr) - length(keep) == 3
            rows = [join(split(l, ',')[keep], ',') for l in rows]
        end
        @test (label, sd_digest(codeunits(join(rows, "\n")))) == (label, ref[label]["diagnostics_csv"])
    end
end

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

cp_bits(sim) = [reinterpret(UInt64, vec(copy(f.data))) for f in (sim.u, sim.v, sim.w, sim.θ, sim.p)]

const CP_DYN = (f_coriolis = 1.0e-3, sponge_alpha_max = 0.02, sponge_z_start = 250.0)
const CP_BND = (theta_flux_bottom = 0.1, theta_flux_top = -0.03)
const CP_SGS = (closure = :smagorinsky, c_s = 0.17, pr_t = 1 / 3, stability = :none)

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
    for (core_, scheme) in ((:boussinesq, :euler), (:boussinesq, :rk3), (:anelastic, :euler))
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
    d = cp_domain(cp_core(:boussinesq), Sponge => CP_DYN, Coriolis => CP_DYN, SGS => CP_SGS,
                  SurfaceThetaFlux => CP_BND)
    @test map(CPC.component_name, d.components) == (:advection, :sgs, :coriolis, :sponge, :surface_theta_flux)
    @test isconcretetype(typeof(d.components))
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
    @test any(m -> occursin("requires Coriolis ([dynamics])", m), msgs)
    @test any(m -> occursin("incompatible with sponge ([dynamics])", m), msgs)
    @test CPC.validate_components((adv, CPNeedsCoriolis(), Coriolis(1e-4))) === nothing
    e = err(() -> CPC.validate_components((adv, CPRecorder((Stage(:projection),)))))
    @test any(p -> occursin(":projection", p.message), e.problems)
    e = err(() -> CPC.validate_components((adv, CPRecorder((Stage(:body_force; every = 10),)))))
    @test any(p -> occursin("takes no cadence", p.message), e.problems)
    e = err(() -> CPC.validate_components((adv, CPRecorder(), CPRecorder())))
    @test any(p -> occursin("two components named", p.message), e.problems)
    @test_throws ArgumentError Model((cp_domain(cp_core(:boussinesq)), cp_domain(cp_core(:boussinesq))))
    @test_throws ArgumentError Domain(cp_core(:anelastic); scheme = :rk3, components = (adv,))
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
    # a stage-6 component turns it off; a stage-11 one does not
    sim = cp_core(:boussinesq; ν = 1.0); cp_state!(sim)
    d = cp_domain(sim, Coriolis => CP_DYN)
    @test CPC.core_hooks(d).body_force! !== nothing
    step!(d, 0.1)
    @test sim.workspaces.sweep === nothing
    sim = cp_core(:boussinesq; ν = 1.0); cp_state!(sim)
    d = cp_domain(sim; extra = (CPRecorder((Stage(:surface_update),)),))
    step!(d, 0.1)
    @test sim.workspaces.sweep !== nothing
end

@testset "components = the core's built-in terms, bit for bit ($c, $sch)" for (c, sch) in
        ((:boussinesq, :euler), (:boussinesq, :rk3), (:anelastic, :euler))
    # the built-in path: the core's own Coriolis, θ fluxes, sponge and Smagorinsky
    ref = cp_state!(cp_core(c; ν = 0.5, f_coriolis = CP_DYN.f_coriolis, C_s = 0.17,
                            θ_flux_bottom = CP_BND.theta_flux_bottom, θ_flux_top = CP_BND.theta_flux_top,
                            sponge_α_max = CP_DYN.sponge_alpha_max, sponge_z_start = CP_DYN.sponge_z_start,
                            (c === :boussinesq ? (; θ₀ = 300.0) : (;))...))
    sim = cp_state!(cp_core(c; ν = 0.5, (c === :boussinesq ? (; θ₀ = 300.0) : (;))...))
    d = cp_domain(sim, SGS => CP_SGS, Coriolis => CP_DYN, Sponge => CP_DYN, SurfaceThetaFlux => CP_BND;
                  scheme = sch)
    @test Aeolus.Components.core(d) === c
    for _ in 1:4
        sch === :rk3 ? step_rk3!(ref, 2.0) : step!(ref, 2.0)
        step!(Model(d), 2.0)
    end
    @test cp_bits(sim) == cp_bits(ref)
    @test sim.t === ref.t
    @test maximum(abs, interior_view(sim.w)) > 0           # it moved
    # a body-force component is step!'s body_force! keyword, bit for bit
    ref = cp_state!(cp_core(c; ν = 0.5))
    sim = cp_state!(cp_core(c; ν = 0.5))
    d = cp_domain(sim; scheme = sch, extra = (CPPush(1e-3),))
    kw = (; body_force! = (s, Δt) -> (interior_view(s.tu) .+= 1e-3; nothing))
    for _ in 1:3
        sch === :rk3 ? step_rk3!(ref, 2.0; kw...) : step!(ref, 2.0; kw...)
        step!(d, 2.0)
    end
    @test cp_bits(sim) == cp_bits(ref)
end

@testset "Coriolis: the inertial oscillation ($c)" for c in (:boussinesq, :anelastic)
    # a uniform (u, v) under f only: forward Euler maps u + iv to
    # (1 − i f Δt)(u + iv) exactly (a clockwise rotation at f, amplitude
    # √(1 + (fΔt)²) per step); the projection of a uniform field is zero
    f = 1.0e-3; Δt = 50.0; n = 40; U0 = 3.0
    sim = cp_core(c); set!(sim.u, (x, y, z) -> U0)
    d = cp_domain(sim, Coriolis => (; f_coriolis = f))
    for _ in 1:n
        step!(d, Δt)
    end
    w = (1 - im * f * Δt)^n * U0
    @test all(x -> isapprox(x, real(w); rtol = 1e-12), interior_view(sim.u))
    @test all(x -> isapprox(x, imag(w); rtol = 1e-12, atol = 1e-12), interior_view(sim.v))
    @test angle(complex(interior_view(sim.u)[1], interior_view(sim.v)[1])) ≈ -n * atan(f * Δt)
    if c === :boussinesq
        # SSP-RK3 over one inertial period 2π/f: back to (U0, 0), amplitude
        # error O((fΔt)⁴) per step (the scheme's |R(iz)|² = 1 − z⁴/12 + z⁶/36)
        sim = cp_core(c); set!(sim.u, (x, y, z) -> U0)
        d = cp_domain(sim, Coriolis => (; f_coriolis = f); scheme = :rk3)
        m = 126; Δ = 2π / f / m
        for _ in 1:m
            step!(d, Δ)
        end
        u, v = interior_view(sim.u)[1], interior_view(sim.v)[1]
        z = f * Δ
        @test hypot(u, v) / U0 ≈ (1 - z^4 / 12 + z^6 / 36)^(m / 2) rtol = 1e-10
        @test abs(atan(v, u)) < 1e-3
        @info "Coriolis RK3, one inertial period in $m steps (fΔt = $(round(z; sigdigits = 3))): " *
              "|u|/U0 = $(hypot(u, v) / U0), phase error $(atan(v, u)) rad"
    end
    # the adaptive limit
    dc = cp_domain(cp_core(:boussinesq), Coriolis => (; f_coriolis = -4.0e-3))
    @test CPC.limit_dt(only(filter(x -> x isa Coriolis, collect(dc.components))), dc) === nothing  # fixed step
    da = Domain(cp_core(:boussinesq); cfl = Aeolus.TimeIntegration.CFLConstraints(; C_advective = 0.5, dt_max = 1e3),
                components = (CentredAdvection(), Coriolis(-4.0e-3)))
    @test CPC.limit_dt(da.components[2], da) == (0.5 / 4.0e-3, :coriolis)
    res = CPC.step_adaptive!(Model(da))
    @test res[2] == 0.5 / 4.0e-3                                   # a fluid at rest: Coriolis binds
    @test Aeolus.Simulations.step_record(da.state).limiter === :coriolis
end

@testset "SurfaceThetaFlux: the domain-mean θ changes by (F_bottom − F_top) t / Lz ($c)" for c in (:boussinesq, :anelastic)
    # at rest: only the fluxes act; the anelastic core is buoyant (θ̄ is its
    # reference), so it takes one step from rest, the Boussinesq one ten
    sim = cp_core(c); c === :boussinesq && set!(sim.θ, (x, y, z) -> 300.0)
    θ0 = copy(interior_view(sim.θ))
    d = cp_domain(sim, SurfaceThetaFlux => CP_BND)
    n = c === :boussinesq ? 10 : 1
    for _ in 1:n
        step!(d, 5.0)
    end
    # the mean of the change (the difference of two means of θ ≈ 300 K loses
    # five digits to cancellation)
    @test isapprox(sum(interior_view(sim.θ) .- θ0) / 8^3, (CP_BND.theta_flux_bottom - CP_BND.theta_flux_top) * 5.0n / 400.0; rtol = 1e-10)
    θ = interior_view(sim.θ)
    @test θ[:, :, 1] != θ[:, :, 4] && θ[:, :, 8] != θ[:, :, 4]   # through the bottom and the top levels
    @test !CPC.enabled(SurfaceThetaFlux, (theta_flux_bottom = 0.0, theta_flux_top = 0.0), d)
end

@testset "Sponge: u decays as (1 − α(z)Δt)ⁿ above z_start, nothing below ($c)" for c in (:boussinesq, :anelastic)
    U0 = 2.0; Δt = 5.0; n = 12
    sim = cp_core(c); set!(sim.u, (x, y, z) -> U0)
    d = cp_domain(sim, Sponge => CP_DYN)
    for _ in 1:n
        step!(d, Δt)
    end
    zs = damping_coordinates(CP_MESH, FaceX())
    p = sponge_profile(CP_DYN.sponge_alpha_max, CP_DYN.sponge_z_start, 400.0)
    u = interior_view(sim.u)
    for k in 1:8
        expected = U0 * (1 - Δt * damping_rate(p, zs[k]))^n
        @test all(x -> isapprox(x, expected; rtol = 1e-12), u[:, :, k])
    end
    @test all(==(U0), u[:, :, findall(<(CP_DYN.sponge_z_start), zs)])
    @test u[1, 1, 8] < 0.6U0                                     # the top level: α ≈ α_max
    # the sponge is off when z_start is at or above the top, or α = 0
    @test !CPC.enabled(Sponge, (sponge_alpha_max = 0.02, sponge_z_start = 400.0), d)
    @test !CPC.enabled(Sponge, (sponge_alpha_max = 0.0, sponge_z_start = 100.0), d)
    @test !CPC.enabled(Sponge, (sponge_alpha_max = 0.02, sponge_z_start = nothing), d)
end

@testset "component contracts ($c)" for c in (:boussinesq, :anelastic)
    sim = cp_state!(cp_core(c; ν = 0.5))
    d = cp_domain(sim, SGS => CP_SGS, Coriolis => CP_DYN, Sponge => CP_DYN, SurfaceThetaFlux => CP_BND)
    for comp in d.components
        T = typeof(comp)
        @test CPC.section(T) isa String && !isempty(CPC.section(T))
        @test Aeolus.Components.schema(T) isa Aeolus.Config.Section
        @test Aeolus.Components.schema(T).name == CPC.section(T)
        @test all(s -> s isa Stage, CPC.stages(comp))
        @test CPC.outputs(comp) == () && CPC.checks(comp) == ()
        @test CPC.budget!(comp, d, nothing) === nothing
        # stateless or not, the checkpoint state restores into the component
        @test restore_checkpoint_state!(comp, checkpoint_state(comp)) === comp
    end
    @test keys(CPC.checkpoint_components(d)) == (:advection, :sgs, :coriolis, :sponge, :surface_theta_flux)
    step!(d, 2.0)
    @test all(f -> all(isfinite, interior_view(f)), (sim.u, sim.v, sim.w, sim.θ))
    @test maximum(interior_view(sim.νt)) > 0                      # the :sgs component ran
    # the closures of [sgs]
    bc(o) = CPC.sgs_closure(cp_domain(cp_core(c), SGS => merge(CP_SGS, o)))
    @test bc((; closure = :vreman, c_s = nothing)) isa VremanClosure
    @test bc((; closure = :amd, c_s = nothing)).C == 0.3
    @test bc((; stability = :brown_et_al_1994)) isa StabilityCorrected
    @test bc((; closure = :none)) === nothing
end

@testset "checkpoint round trip: a restart through the components is the uninterrupted run" begin
    # the Lagrangian dynamic closure carries memory (its averaged LM, MM)
    mk() = (s = cp_state!(cp_core(:boussinesq; ν = 0.5, θ₀ = 300.0));
            d0 = Domain(s; validate = false);
            Domain(d0; components = (CentredAdvection(), SGS(LagrangianDynamicSmagorinsky(CP_MESH)),
                                     CPC.build_components((Coriolis => CP_DYN, Sponge => CP_DYN), d0)...)))
    d = mk()
    for _ in 1:3; step!(d, 2.0); end
    path = joinpath(mktempdir(), "ck.jls")
    checkpoint!(d.state, path; components = CPC.checkpoint_components(d))
    for _ in 1:3; step!(d, 2.0); end
    d2 = mk()
    restart!(d2.state, path; components = CPC.checkpoint_components(d2))
    for _ in 1:3; step!(d2, 2.0); end
    @test cp_bits(d2.state) == cp_bits(d.state)
    # a restart into another closure is refused
    d3 = Domain(Domain(cp_core(:boussinesq; ν = 0.5, θ₀ = 300.0); validate = false);
                components = (CentredAdvection(), SGS(StaticSmagorinsky(0.17)), Coriolis(1e-3), Sponge(0.02, 250.0, (u = 0, v = 0, w = 0))))
    @test_throws ArgumentError restart!(d3.state, path; components = CPC.checkpoint_components(d3))
end
