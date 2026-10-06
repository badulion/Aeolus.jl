# Full-state checkpoint / restart (#245): src/io/checkpoint.jl,
# src/wall_models/checkpoint.jl, the FFTW planning policy
# (src/operators/fftw_planning.jl).
#
# The physics assertion is **bitwise**: a run checkpointed at step N and
# restarted (into freshly built objects whose scratch buffers hold NaN) is
# identical, bit for bit, to the continuous run at step 2N — fields with
# halos, closure memory, wall-model state, clock — for every closure with
# memory, under every stepper, and with wall models whose input filter and
# TBLE columns carry memory.

using Test
using Aeolus
using Aeolus.Fields: interior_view, set!
using Aeolus.BoundaryConditions: FieldBoundaryConditions
using Aeolus.TimeIntegration: step_adaptive!, step_imex!, ARK2, CFLConstraints
using Random
using Serialization

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
# before writing it would not be bitwise (it would not even be finite).
function ck_poison!(sim)
    for f in (sim.tu, sim.tv, sim.tw, sim.tθ, sim.tqv, sim.div_buf, sim.φ, sim.dφdx,
              sim.dφdy, sim.dφdz, sim.lap_u, sim.lap_v, sim.lap_w, sim.lap_θ)
        fill!(f.data, NaN)
    end
    return sim
end

ck_state(sim) = map(f -> copy(f.data), checkpoint_fields(sim))

const CK_SCHEMES = (:fixed, :rk3, :imex, :adaptive)

# Two wall-modelled cases whose wall model carries memory: the half channel
# with the P57-b matching input filtered in time and along the wall, run
# with the scale-dependent Lagrangian closure; and the channel with the
# P57-c TBLE columns behind a temporally filtered input.
function ck_case(kind)
    if kind === :filtered_lasd
        m = RegularGrid(; size = (12, 10, 8), extent = (2π, 2π, 1.0))
        half_channel_case(; N = (12, 10, 8), seed = 4, noise = 1.0, h_wm = 2 / 8,
                          filter = :both, T_f = 0.2,
                          closure = LagrangianDynamicSmagorinsky(m; scale_dependent = true))
    else
        dz = 2.0 / 8
        channel_case(; Re_τ = 1000, N = (12, 8, 8), seed = 4, wall_law = SpaldingLaw(),
                     h_wm = 2dz, filter = :temporal, wall_model = :nonequilibrium)
    end
end

# Bitwise equality of two (nested) checkpoint states: their serialised bytes.
ck_bytes(x) = (io = IOBuffer(); serialize(io, x); take!(io))
ck_same(a, b) = ck_bytes(a) == ck_bytes(b)

@testset "Checkpoint / restart (#245)" begin

@testset "bitwise restart: $name × $scheme" for name in keys(CK_CLOSURES), scheme in CK_SCHEMES
    mk = CK_CLOSURES[name]
    Nhalf = 3
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
    @test stC.step == Nhalf && simC.t == simB.t
    for _ in 1:Nhalf; ck_step!(simC, cC, stC, scheme); end
    @test simC.t === simA.t
    sA, sC = ck_state(simA), ck_state(simC)
    @test all(k -> sA[k] == sC[k], keys(sA))                 # bitwise, halos included
    @test checkpoint_state(cC) == checkpoint_state(cA)       # closure memory, bitwise
    @test checkpoint_state(stC) == checkpoint_state(stA)     # step count, dt history
    @test all(isfinite, interior_view(simC.u))
    # the run did something: the state moved, and the memory is not the initial one
    @test sA.u != ck_state(ck_sim()).u
    rm(path)
end

@testset "bitwise restart: adaptive dt controller resumes" begin
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
    @test stC.dt_history == stA.dt_history
    @test all(i -> stA.dt_history[i+1] == 1.1 * stA.dt_history[i], 1:5)   # growth-limited
    @test ck_state(simC).u == ck_state(simA).u
    # without the stepper state the first restarted dt is the CFL limit
    simD = ck_sim(); restart!(simD, path; sgs = sgs, stepper = StepperState(:adaptive))
    _, dtD, _ = step_adaptive!(simD, c; sgs = sgs)
    @test dtD > 5 * stA.dt_history[4]
    rm(path)
end

# ---------------------------------------------------------------------------
# Wall models with memory (P57-b input filter, P57-c TBLE columns) and the
# case harness (WallBoundedCase): checkpointed as one component.
# ---------------------------------------------------------------------------
@testset "bitwise restart: wall model with memory ($kind)" for kind in (:filtered_lasd, :tble)
    Nhalf = 4; dt = 0.004
    A = ck_case(kind)
    for _ in 1:2Nhalf; advance_channel!(A, dt); end
    B = ck_case(kind)
    for _ in 1:Nhalf; advance_channel!(B, dt); end
    path = tempname()
    checkpoint!(B.sim, path; components = (case = B,))
    C = ck_case(kind)
    ck_poison!(C.sim)
    restart!(C.sim, path; components = (case = C,))
    @test C.nsteps == Nhalf
    for _ in 1:Nhalf; advance_channel!(C, dt); end
    sA, sC = ck_state(A.sim), ck_state(C.sim)
    @test all(k -> sA[k] == sC[k], keys(sA))
    @test ck_same(checkpoint_state(C), checkpoint_state(A))
    @test C.τ_wall_bottom === A.τ_wall_bottom
    if kind === :filtered_lasd
        # the filter state is memory: primed, and not the instantaneous sample
        mi = A.inputs.bottom
        @test mi.primed && mi.T_f > 0
        @test checkpoint_state(C.inputs.bottom) == checkpoint_state(mi)
        # a restart that dropped the filter memory diverges: the memory matters
        D = ck_case(kind)
        restart!(D.sim, path; components = (case = D,))
        D.inputs.bottom.primed = false
        for _ in 1:Nhalf; advance_channel!(D, dt); end
        @test ck_state(D.sim).u != sA.u
    else
        w = A.wall.walls[:bottom]
        @test w.nupdates > 0
        @test C.wall.walls[:bottom].cols[3, 2].u == w.cols[3, 2].u
    end
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
    # FFTW provenance: the suite plans with ESTIMATE, so bitwise reproducible
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
@testset "CheckpointSchedule: keep-K, resume, wisdom" begin
    dir = mktempdir()
    function run!(stop_at; nsteps = 10)
        ck = CheckpointSchedule(dir; every_steps = 3, keep = 2, verbose = false)
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
    @test ck_state(second.sim).u == ck_state(cont.sim).u
    @test second.series == cont.series                 # the driver's series continues
    @test checkpoint_state(second.c) == checkpoint_state(cont.c)
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
        # a PATIENT plan of the pressure solver solves the same problem
        # (round-off apart): the planner changes speed, not the answer
        set_fftw_policy!(; planner = :estimate)
        sE = ck_sim(); step!(sE, CK_DT)
        set_fftw_policy!(; planner = :patient)
        sP = ck_sim(); step!(sP, CK_DT)
        @test maximum(abs, interior_view(sP.u) .- interior_view(sE.u)) < 1e-12
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
