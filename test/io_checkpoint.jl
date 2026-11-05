# Full-state checkpoint / restart (#245): src/io/checkpoint.jl, the FFTW
# planning policy (src/operators/fftw_planning.jl).
#
# The physics assertion: a run checkpointed at step N and restarted (into
# freshly built objects whose scratch buffers hold NaN) agrees with the
# continuous run at step 2N within ROUNDOFF_RTOL (test/support/agreement.jl;
# no bit-identity since #495) — fields with
# halos, closure memory, clock — for a closure with memory under RK3, and
# the adaptive stepper's dt memory.

using Test
using Aeolus
using Aeolus.Fields: interior_view, set!
using Aeolus.BoundaryConditions: FieldBoundaryConditions
using Aeolus.TimeIntegration: step_adaptive!, step_imex!, ARK2, CFLConstraints
using Random
using Serialization
include(joinpath(@__DIR__, "support", "agreement.jl"))

const CK_N = (12, 10, 8)

function ck_sim(; seed = 3)
    m = RegularGrid(; size = CK_N, extent = (1200.0, 1000.0, 400.0), halo = (3, 3, 3))
    bcs = FieldBoundaryConditions(:periodic, :periodic, :wall)
    sim = BoussinesqSimulation(m; θ₀ = 300.0, f_coriolis = 1e-4, bcs = bcs,
                               θ_flux_bottom = 0.05)
    rng = Xoshiro(seed)
    set!(sim.u, (x, y, z) -> 2.0 + 0.6 * randn(rng))
    set!(sim.v, (x, y, z) -> 0.6 * randn(rng))
    set!(sim.w, (x, y, z) -> 0.0)
    set!(sim.θ, (x, y, z) -> 300.0 + 0.004 * z + 0.2 * randn(rng))
    Aeolus.TimeIntegration._project!(sim, 1.0)
    return sim
end

const CK_CLOSURES = (
    smagorinsky = sim -> StaticSmagorinsky(0.17),
    tke         = sim -> (c = TKEClosure(sim.mesh); initialize_tke!(c, sim); c),
    lagrangian  = sim -> LagrangianDynamicSmagorinsky(sim.mesh),
    lasd        = sim -> LagrangianDynamicSmagorinsky(sim.mesh; scale_dependent = true),
    rsm         = sim -> (c = ReynoldsStressClosure(sim.mesh);
                          initialize_reynolds_stress!(c, sim; e_min = 1e-3); c),
)

const CK_DT = 2.0
const CK_CFL = CFLConstraints(; dt_max = 4.0, dt_min = 1e-3)

function ck_step!(sim, c, st, scheme)
    if scheme === :fixed
        step!(sim, CK_DT; sgs = c); record_step!(st, CK_DT)
    elseif scheme === :rk3
        step_rk3!(sim, CK_DT; sgs = c); record_step!(st, CK_DT)
    elseif scheme === :imex
        step_imex!(sim, CK_DT, ARK2(); sgs = c); record_step!(st, CK_DT)
    else
        _, dt, nr = step_adaptive!(sim, CK_CFL; dt_prev = previous_dt(st), sgs = c)
        record_step!(st, dt; retries = nr)
    end
    return sim
end

# NaN in every buffer that is not state: a restart that read one of them
# before writing it would not agree (it would not even be finite).
function ck_poison!(sim)
    for f in (sim.tu, sim.tv, sim.tw, sim.tθ, sim.tqv, sim.div_buf, sim.φ, sim.dφdx,
              sim.dφdy, sim.dφdz, sim.lap_u, sim.lap_v, sim.lap_w, sim.lap_θ)
        fill!(f.data, NaN)
    end
    return sim
end

ck_state(sim) = map(f -> copy(f.data), checkpoint_fields(sim))

const CK_SCHEMES = (:fixed, :rk3, :imex, :adaptive)

# Agreement of two (nested) checkpoint states: floats within ROUNDOFF_RTOL,
# the rest (counts, symbols, structure) equal.
ck_same(a, b) = values_agree(a, b)

# The structural state of an object graph (#269): what the nested run and
# the Berlin case are checkpointed with.
mutable struct _SSNode
    a::Vector{Float64}
    b::Vector{Float64}                     # shares `a` in the run, not in a fresh build
    lazy::Union{Nothing,Vector{Float64}}   # allocated during the run
    planes::Vector{NamedTuple{(:name, :data), Tuple{Symbol, Vector{Float64}}}}  # inline structs with references
    bound::IdDict{Any,Symbol}              # binds arrays by identity
    d::Dict{Symbol,Any}
    lock::ReentrantLock                    # not state
    rng::Xoshiro
    n::Int
    const tag::Symbol
end
function _ss_node()
    x = [1.0, 2.0]
    _SSNode(x, copy(x), nothing, [(name = :u, data = x)], IdDict{Any,Symbol}(), Dict{Symbol,Any}(),
            ReentrantLock(), Xoshiro(5), 0, :node)
end


@testset "Checkpoint / restart (#245)" begin

# a closure with memory (TKE) under RK3; the stepper's memory (the adaptive
# dt history) is the next testset
@testset "restart: $name × $scheme" for (name, scheme) in ((:tke, :rk3),)
    mk = CK_CLOSURES[name]
    Nhalf = 2
    # the RSM's explicit stress cannot be split by step_imex!: refused by design
    if name === :rsm && scheme === :imex
        sim = ck_sim(); c = mk(sim)
        @test_throws ArgumentError step_imex!(sim, CK_DT, ARK2(); sgs = c)
        continue
    end
    # continuous run, 2N steps
    simA = ck_sim(); cA = mk(simA); stA = StepperState(scheme)
    for _ in 1:2Nhalf; ck_step!(simA, cA, stA, scheme); end
    # N steps, checkpoint
    simB = ck_sim(); cB = mk(simB); stB = StepperState(scheme)
    for _ in 1:Nhalf; ck_step!(simB, cB, stB, scheme); end
    path = tempname() * ".jls"
    checkpoint!(simB, path; sgs = cB, stepper = stB, extra = Dict(:note => "half"))
    # restart into fresh objects (other initial state, NaN scratch), N more steps
    simC = ck_sim(; seed = 99); cC = mk(simC); stC = StepperState(scheme)
    ck_poison!(simC)
    restart!(simC, path; sgs = cC, stepper = stC)
    @test stC.step == Nhalf && agree(simC.t, simB.t)
    for _ in 1:Nhalf; ck_step!(simC, cC, stC, scheme); end
    @test agree(simC.t, simA.t)
    sA, sC = ck_state(simA), ck_state(simC)
    @test all(k -> agree(sA[k], sC[k]), keys(sA))            # halos included
    @test values_agree(checkpoint_state(cC), checkpoint_state(cA))       # closure memory
    @test values_agree(checkpoint_state(stC), checkpoint_state(stA))     # step count, dt history
    @test all(isfinite, interior_view(simC.u))
    # the run did something: the state moved, and the memory is not the initial one
    @test sA.u != ck_state(ck_sim()).u
    rm(path)
end

@testset "restart: adaptive dt controller resumes" begin
    # The first step is forced short (dt = 0.5 s); afterwards the controller
    # grows dt by at most growth_max = 1.1 per step from dt_prev, far below
    # the CFL limit (≈ 9 s), so every dt is set by the memory dt_prev. A
    # restart that forgot it would jump to the CFL limit.
    c = CFLConstraints(; dt_max = 50.0, dt_min = 1e-3, growth_max = 1.1)
    sgs = StaticSmagorinsky(0.17)
    function adv!(sim, st, n)
        for _ in 1:n
            _, dt, nr = st.step == 0 ? step_adaptive!(sim, c; dt = 0.5, sgs = sgs) :
                                       step_adaptive!(sim, c; dt_prev = previous_dt(st), sgs = sgs)
            record_step!(st, dt; retries = nr)
        end
    end
    simA = ck_sim(); stA = StepperState(:adaptive); adv!(simA, stA, 6)
    simB = ck_sim(); stB = StepperState(:adaptive); adv!(simB, stB, 3)
    path = tempname()
    checkpoint!(simB, path; sgs = sgs, stepper = stB)
    simC = ck_sim(); stC = StepperState(:adaptive)
    restart!(simC, path; sgs = sgs, stepper = stC)
    adv!(simC, stC, 3)
    @test agree(stC.dt_history, stA.dt_history)
    @test all(i -> agree(stA.dt_history[i+1], 1.1 * stA.dt_history[i]), 1:5)   # growth-limited
    @test agree(ck_state(simC).u, ck_state(simA).u)
    # without the stepper state the first restarted dt is the CFL limit
    simD = ck_sim(); restart!(simD, path; sgs = sgs, stepper = StepperState(:adaptive))
    _, dtD, _ = step_adaptive!(simD, c; sgs = sgs)
    @test dtD > 5 * stA.dt_history[4]
    rm(path)
end

# ---------------------------------------------------------------------------
# Format: versioned, old snapshots refused, mismatches caught
# ---------------------------------------------------------------------------
@testset "format: version, provenance, refusals" begin
    sim = ck_sim()
    c = TKEClosure(sim.mesh); initialize_tke!(c, sim)
    step!(sim, CK_DT; sgs = c)
    path = tempname()
    checkpoint!(sim, path; sgs = c, extra = Dict("tag" => 7))
    d = read_checkpoint(path)
    @test d[:format] == "Aeolus.checkpoint" && d[:version] == CHECKPOINT_VERSION == 1
    @test d[:julia_version] == string(VERSION)
    @test d[:extra][:tag] == 7
    @test d[:closure_type] == "TKEClosure"
    @test size(d[:fields][:u]) == size(sim.u.data)            # halos stored
    # FFTW provenance: the suite plans with ESTIMATE, so reproducible
    @test d[:fftw].planner === :estimate && d[:fftw].bitwise_reproducible
    @test d[:fftw].threads == fftw_policy().threads
    @test !isfile(path * ".tmp")                               # atomic write
    # unversioned snapshots (write_snapshot, pre-#245 checkpoint!) are refused
    old = tempname()
    write_snapshot(old, sim)
    err = try restart!(ck_sim(), old); nothing catch e; e end
    @test err isa ArgumentError && occursin("unversioned snapshot", err.msg)
    # a newer format is refused
    newer = tempname(); dd = copy(d); dd[:version] = CHECKPOINT_VERSION + 1
    open(io -> serialize(io, dd), newer, "w")
    @test_throws ArgumentError restart!(ck_sim(), newer)
    # wrong closure, missing closure, missing components, wrong mesh
    @test_throws ArgumentError restart!(ck_sim(), path; sgs = StaticSmagorinsky(0.1))
    @test_throws ArgumentError restart!(ck_sim(), path)
    @test_throws ArgumentError restart!(ck_sim(), path; sgs = TKEClosure(sim.mesh),
                                        components = (x = Ref(1),))
    @test_throws ArgumentError restart!(ck_sim(), path; sgs = TKEClosure(sim.mesh),
                                        stepper = StepperState())
    small = BoussinesqSimulation(RegularGrid(; size = (8, 8, 8), extent = (1.0, 1.0, 1.0), halo = (3, 3, 3)))
    @test_throws ArgumentError restart!(small, path; sgs = TKEClosure(small.mesh))
    # run parameters (maintainer decision on #318): a mismatch is an error that
    # lists every difference, unless allow_mismatch = true, which logs them
    other = ck_sim(); other.f_coriolis = 2e-4; other.ν = 1.5e-5
    err = try restart!(other, path; sgs = TKEClosure(other.mesh)); nothing catch e; e end
    @test err isa ArgumentError
    @test occursin("f_coriolis = 0.0002 (checkpoint: 0.0001)", err.msg)
    @test occursin("ν = 1.5e-5 (checkpoint: 0.0)", err.msg)
    @test occursin("allow_mismatch = true", err.msg)
    @test other.t == 0.0 && other.u.data != sim.u.data          # nothing restored
    cO = TKEClosure(other.mesh)
    @test_logs (:warn, r"allow_mismatch = true.*f_coriolis = 0\.0002 \(checkpoint: 0\.0001\)") match_mode = :any restart!(
        other, path; sgs = cO, allow_mismatch = true)
    @test other.t == sim.t && other.u.data == sim.u.data         # restored with the override
    @test other.f_coriolis == 2e-4                               # the run keeps its own parameters
    # the schedule's resume! takes the same switch
    ckdir = mktempdir(); ck = CheckpointSchedule(ckdir; wisdom = false)
    cp(path, joinpath(ckdir, "checkpoint_000000001.jls"))
    sR = ck_sim(); sR.f_coriolis = 2e-4
    @test_throws ArgumentError resume!(ck, sR; sgs = TKEClosure(sR.mesh))
    @test resume!(ck, sR; sgs = TKEClosure(sR.mesh), allow_mismatch = true) isa AbstractDict
    @test sR.t == sim.t && sR.u.data == sim.u.data
    rm(ckdir; recursive = true)
    # a component without a method is named
    err = try checkpoint!(sim, tempname(); components = (x = :sym,)); nothing catch e; e end
    @test err isa ArgumentError && occursin("checkpoint_state", err.msg)
    rm(path); rm(old); rm(newer)
end

@testset "components: RNG, accumulators, mutable structs" begin
    sim = ck_sim()
    rng = Xoshiro(17); rand(rng, 5)
    series = [1.0, 2.0]; acc = Dict(:n => 3, :sum => [1.0, 2.0]); r = Ref(0.5)
    path = tempname()
    checkpoint!(sim, path; rng = rng, components = (series = series, acc = acc, r = r))
    expected = rand(copy(rng), 4)
    rand(rng, 100); push!(series, 3.0); acc[:n] = 99; r[] = -1.0
    restart!(sim, path; rng = rng, components = (series = series, acc = acc, r = r))
    @test rand(rng, 4) == expected
    @test series == [1.0, 2.0] && acc == Dict(:n => 3, :sum => [1.0, 2.0]) && r[] == 0.5
    # the task-local default RNG
    Random.seed!(5); rand(3)
    checkpoint!(sim, path; rng = Random.default_rng())
    e2 = rand(4)
    restart!(sim, path; rng = Random.default_rng())
    @test rand(4) == e2
    rm(path)
end

# ---------------------------------------------------------------------------
# Periodic checkpoints with automatic resume (the long drivers' helper)
# ---------------------------------------------------------------------------
@testset "structural state: aliasing, lazy members, identity maps, opaque objects" begin
    r = _ss_node()
    r.b = r.a                                  # aliasing made during the run
    r.a .= (3.0, 4.0); r.lazy = [9.0]
    push!(r.planes, (name = :v, data = [5.0]))
    r.bound[r.a] = :a; r.bound[r.planes[2].data] = :v
    r.d[:k] = [7.0]; r.d[:j] = 1
    rand(r.rng); r.n = 4
    s = structural_state(r)
    @test s isa StructuralState && s.objects > 0
    s = deserialize(seekstart((io = IOBuffer(); serialize(io, s); io)))   # self-contained
    f = _ss_node(); lk = f.lock
    restore_structural_state!(f, s)
    @test f.a == [3.0, 4.0] && f.b === f.a                       # aliasing restored
    @test f.lazy == [9.0] && f.n == 4 && f.d[:k] == [7.0] && f.d[:j] == 1
    @test f.planes[1].data === f.a && f.planes[2].data == [5.0] # not the record's arrays
    @test f.planes[2].data !== s.root.fields[4].elems[2].fields[2].a
    @test f.bound[f.a] === :a && f.bound[f.planes[2].data] === :v && length(f.bound) == 2
    @test f.lock === lk                                          # opaque: kept
    @test rand(f.rng) == rand(copy(r.rng))                       # RNG state
    # refusals: another root type; a const field that differs
    @test_throws ArgumentError restore_structural_state!(Ref(1), s)
    g = _SSNode([0.0, 0.0], [0.0, 0.0], nothing, [(name = :u, data = [0.0, 0.0])], IdDict{Any,Symbol}(),
                Dict{Symbol,Any}(), ReentrantLock(), Xoshiro(5), 0, :other)
    err = try restore_structural_state!(g, s); nothing catch e; e end
    @test err isa ArgumentError && occursin(".tag", sprint(showerror, err))
end

@testset "structural state refreshed in place: the record of now, in the arrays of then (#482)" begin
    r = _ss_node()
    r.b = r.a
    r.bound[r.a] = :a
    r.d[:k] = [7.0]
    s = structural_state(r)
    rec_a = s.root.fields[1].a
    # the run moves on; the record is refreshed into its own arrays
    r.a .= (5.0, 6.0); r.n = 11; r.d[:k][1] = 8.0; rand(r.rng)
    want = rand(copy(r.rng))
    s2 = Aeolus.IO.refresh_structural_state!(s, r)
    @test s2.root === s.root && s2.root.fields[1].a === rec_a      # nothing re-allocated
    # and restores exactly as a fresh record of now does
    r.a .= 0.0; r.n = -1; r.d[:k][1] = 0.0; rand(r.rng, 3)
    restore_structural_state!(r, s2)
    @test r.a == [5.0, 6.0] && r.b === r.a && r.n == 11 && r.d[:k] == [8.0]
    @test r.bound[r.a] === :a && rand(r.rng) == want
    # a changed graph (a dictionary with another key) falls back to a fresh record
    r.d[:new] = 1
    s3 = Aeolus.IO.refresh_structural_state!(s2, r)
    @test s3.root !== s2.root
    r.d[:new] = 2
    restore_structural_state!(r, s3)
    @test r.d[:new] == 1
end

@testset "CheckpointSchedule: keep-K, resume, wisdom" begin
    dir = mktempdir()
    function run!(stop_at; nsteps = 10)
        ck = CheckpointSchedule(dir; every_steps = 3, keep = 2)
        sim = ck_sim(); c = TKEClosure(sim.mesh); initialize_tke!(c, sim)
        st = StepperState(:rk3); series = Float64[]
        d = resume!(ck, sim; sgs = c, stepper = st, components = (series = series,))
        while st.step < nsteps
            step_rk3!(sim, CK_DT; sgs = c); record_step!(st, CK_DT)
            push!(series, kinetic_energy(sim))
            maybe_checkpoint!(ck, sim; step = st.step, sgs = c, stepper = st,
                              components = (series = series,))
            st.step == stop_at && return (sim = sim, c = c, series = series, resumed = d)
        end
        return (sim = sim, c = c, series = series, resumed = d)
    end
    run!(7)                                            # "killed" after step 7
    files = filter(f -> endswith(f, ".jls"), readdir(dir))
    @test files == ["checkpoint_000000003.jls", "checkpoint_000000006.jls"]   # keep = 2
    @test isfile(joinpath(dir, "fftw_wisdom"))         # wisdom saved at run start
    second = run!(-1)                                  # resumes from step 6
    @test second.resumed !== nothing && second.resumed[:extra][:step] == 6
    @test basename(second.resumed[:path]) == "checkpoint_000000006.jls"
    @test fftw_policy().wisdom_loaded                  # the restart imported the wisdom
    cont = let
        sim = ck_sim(); c = TKEClosure(sim.mesh); initialize_tke!(c, sim); s = Float64[]
        for _ in 1:10; step_rk3!(sim, CK_DT; sgs = c); push!(s, kinetic_energy(sim)); end
        (sim = sim, c = c, series = s)
    end
    @test agree(ck_state(second.sim).u, ck_state(cont.sim).u)
    @test values_agree(second.series, cont.series)     # the driver's series continues
    @test values_agree(checkpoint_state(second.c), checkpoint_state(cont.c))
    @test filter(f -> endswith(f, ".jls"), readdir(dir)) == ["checkpoint_000000006.jls", "checkpoint_000000009.jls"]
    rm(dir; recursive = true)
end

# ---------------------------------------------------------------------------
# The FFTW planning policy
# ---------------------------------------------------------------------------
@testset "FFTW policy: configurable, provenance, config table" begin
    p0 = fftw_policy()
    @test p0.planner === :estimate                     # forced by runtests.jl
    @test fftw_planner_flags() == Aeolus.Operators.FFTW.ESTIMATE
    # the default outside the suite is MEASURE, PATIENT is opt-in
    # (maintainer decision on #318)
    @test Aeolus.Operators.FFTW_DEFAULT_PLANNER === :measure
    try
        for (p, f) in ((:measure, Aeolus.Operators.FFTW.MEASURE), (:patient, Aeolus.Operators.FFTW.PATIENT),
                       (:exhaustive, Aeolus.Operators.FFTW.EXHAUSTIVE))
            set_fftw_policy!(; planner = p)
            @test fftw_planner_flags() == f
        end
        set_fftw_policy!(; planner = "Patient")
        @test fftw_policy().planner === :patient
        @test_throws ArgumentError set_fftw_policy!(; planner = :fastest)
        @test_throws ArgumentError set_fftw_policy!(; threads = 0)
        # namelist table
        cfg = Dict("fftw" => Dict("planner" => "measure", "threads" => 1))
        @test apply_fftw_config!(cfg).planner === :measure
        @test apply_fftw_config!(Dict{String,Any}()).planner === :measure   # no table: unchanged
    finally
        set_fftw_policy!(; planner = p0.planner, threads = p0.threads)
    end
    @test fftw_policy().planner === :estimate
end

end # testset
