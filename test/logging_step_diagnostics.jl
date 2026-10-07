# Per-step diagnostics in the step cores (#336, epic #293):
# src/simulations/step_record.jl, the brackets in every step core, the
# step-record and Field-reduction diagnostics of
# src/logging/component_diagnostics.jl, the attached run monitor of
# src/io/run_monitor.jl.
#
# The acceptance tests:
#   * every step core (step!, step_rk3!, step_imex!, step_adaptive!, the
#     sound-proof core, both domains and the run of step_nested!) calls the
#     simulation's hook exactly once per step: a core inside another and a
#     rejected attempt are not steps, a step that throws is none;
#   * the recorded values equal independently computed ones: dt and its
#     limiter, retries, the advective and diffusive CFL numbers recomputed
#     by hand from the pre-step state, the pressure-solver iterations and
#     residuals of every solve;
#   * the Field reductions of the registry equal serial reductions;
#   * an attached monitor writes one row per step, with these values, and
#     a driver that reports itself gets the steps since its last row.

using Test
using Logging
using Random
using Aeolus
using Aeolus.Fields: interior_view, set!
using Aeolus.Nesting
using Aeolus.TimeIntegration: step_adaptive!, CFLConstraints, cfl_diagnostics, step_imex!, ARK2,
                              imex_workspace, NumericalFailure
using Aeolus.PressureSolvers: AbstractPressureSolver, solve!

function sd_sim(; N = (12, 10, 8), C_s = 0.17, seed = 3, kw...)
    m = RegularGrid(; size = N, extent = (600.0, 500.0, 400.0), halo = (3, 3, 3))
    sim = BoussinesqSimulation(m; θ₀ = 300.0, C_s = C_s,
                               bcs = FieldBoundaryConditions(:periodic, :periodic, :wall), kw...)
    rng = Xoshiro(seed)
    set!(sim.u, (x, y, z) -> 2.0 + 0.4 * randn(rng))
    set!(sim.v, (x, y, z) -> -0.5 + 0.4 * randn(rng))
    set!(sim.θ, (x, y, z) -> 300.0 + 0.003 * z + 0.05 * randn(rng))
    Aeolus.TimeIntegration._project!(sim, 1.0)
    # that projection's solve is outside any step: it would count in the
    # first interval (as a two-way nest's re-projection counts in its window)
    reset_interval!(step_record(sim))
    return sim
end

sd_raw(f) = (m = f.mesh; f.data[m.Hx+1:m.Hx+m.Nx, m.Hy+1:m.Hy+m.Ny, m.Hz+1:m.Hz+m.Nz])

# max over the cells of Σ max(|u_face|) / Δx, by hand on the halo-filled data
function sd_adv_rate(sim)
    m = sim.mesh
    fill_halos!(sim.u, sim.bcs); fill_halos!(sim.v, sim.bcs); fill_halos!(sim.w, sim.bcs)
    U, V, W = sim.u.data, sim.v.data, sim.w.data
    r = 0.0
    for k in 1:m.Nz, j in 1:m.Ny, i in 1:m.Nx
        I, J, K = m.Hx + i, m.Hy + j, m.Hz + k
        r = max(r, max(abs(U[I, J, K]), abs(U[I+1, J, K])) / m.dx +
                   max(abs(V[I, J, K]), abs(V[I, J+1, K])) / m.dy +
                   max(abs(W[I, J, K]), abs(W[I, J, K+1])) / m.dz)
    end
    return r
end

# A pressure solver that delegates and reports fixed statistics: the
# iterations and residual the record must sum and maximise.
struct CountingSolver{S} <: AbstractPressureSolver
    inner::S
    calls::Base.RefValue{Int}
end
function Aeolus.PressureSolvers.solve!(p, src, s::CountingSolver)
    s.calls[] += 1
    return solve!(p, src, s.inner)
end
Aeolus.PressureSolvers.solver_stats(s::CountingSolver) = (7, 1e-3 * s.calls[])
Aeolus.PressureSolvers.poisson_axis_bcs(s::CountingSolver) =
    Aeolus.PressureSolvers.poisson_axis_bcs(s.inner)

function hooked(sim)
    n = Ref(0)
    set_step_hook!(sim, (s, r) -> (n[] += 1))
    return n
end

@testset "step record: one step per outermost core call" begin
    sim = sd_sim()
    calls = hooked(sim)
    r = step_record(sim)
    @test r === sim.workspaces.rec && step_hook(sim) !== nothing
    step!(sim, 0.4)
    @test calls[] == 1 && r.steps == 1 && r.n == 1 && r.depth == 0
    @test r.dt == 0.4 && r.dt_sum == 0.4 && r.limiter === :fixed && r.retries == 0
    @test r.solves == 1 && r.solver_iters == 0 && isnan(r.solver_residual)   # FFT: direct
    @test isnan(r.cfl_adv)                     # no dt pick: a fixed step has no CFL number
    step_rk3!(sim, 0.4)
    @test calls[] == 2 && r.n == 2 && r.solves == 4 && r.dt_sum ≈ 0.8
    step_imex!(sim, 0.4, ARK2())
    ws = imex_workspace(sim)
    @test calls[] == 3 && ws.total_iterations > 0 && r.implicit_iters == ws.total_iterations
    @test r.implicit_residual == ws.max_residual
    # the interval restarts; the cumulative wall times do not
    w = r.wall_step
    reset_interval!(r)
    @test r.n == 0 && r.solves == 0 && r.dt_sum == 0 && r.steps == 3 && r.wall_step == w > 0
    @test r.wall_pressure > 0 && r.wall_sgs > 0 && r.wall_pressure + r.wall_sgs < r.wall_step
    # a step that throws is no step, and the bracket is closed
    @test_throws ErrorException step!(sim, 0.4; body_force! = (s, h) -> error("boom"))
    @test calls[] == 3 && r.steps == 3 && r.depth == 0
    step!(sim, 0.4)
    @test calls[] == 4
    set_step_hook!(sim, nothing)
    @test step_hook(sim) === nothing
    step!(sim, 0.4)
    @test calls[] == 4 && r.steps == 5
    # no hook: the core still allocates nothing
    @test (@allocated step!(sim, 0.4)) == 0
    @test step_record(42) === nothing && step_hook(42) === nothing
    @test_throws ArgumentError set_step_hook!(42, identity)
end

@testset "step record: pressure-solver iterations and residuals of every solve" begin
    m0 = sd_sim().mesh
    cs = CountingSolver(FFTPoissonSolver(m0), Ref(0))
    sim = sd_sim(; solver = cs)
    cs.calls[] = 0                             # sd_sim's own projection solved once
    r = step_record(sim)
    step!(sim, 0.3)
    @test r.solves == 1 && r.solver_iters == 7 && r.solver_residual == 1e-3
    reset_interval!(r)
    step_rk3!(sim, 0.3)                        # three projections
    @test r.solves == 3 && r.solver_iters == 21 && r.solver_residual == 4e-3   # the largest
    # an iterative solver reports its own count and residual
    m = sd_sim().mesh
    cg = CGPoissonSolver(m; tol = 1e-10, maxit = 500)
    sc = sd_sim(; solver = cg)
    rc = step_record(sc)
    step!(sc, 0.3)
    @test rc.solver_iters == cg.last_iter[] > 0 && rc.solver_residual == cg.last_residual[]
    @test solver_stats(cg) == (cg.last_iter[], cg.last_residual[])
    @test solver_stats(FFTPoissonSolver(m))[1] == 0 && isnan(solver_stats(FFTPoissonSolver(m))[2])
end

@testset "step_adaptive!: CFL numbers and limiter of the dt pick, retries in one step" begin
    sim = sd_sim()
    calls = hooked(sim)
    r = step_record(sim)
    c = CFLConstraints(; C_advective = 0.5, dt_max = 100.0, growth_max = 10.0)
    # the pick's rates, by hand from the pre-step state
    rate = sd_adv_rate(sim)
    d = cfl_diagnostics(sim, c)                # fills νt, κt of this state
    m = sim.mesh
    νt, κt = maximum(sd_raw(sim.νt)), maximum(sd_raw(sim.κt))
    inv2 = 1 / m.dx^2 + 1 / m.dy^2 + 1 / m.dz^2
    @test d.advective_rate ≈ rate rtol = 1e-14
    _, dt, nr = step_adaptive!(sim, c)
    @test calls[] == 1 && r.steps == 1 && nr == 0      # one step, not one per inner step!
    @test dt == d.dt_limiting && r.dt == dt && r.limiter === d.limiter
    @test r.cfl_adv ≈ dt * rate rtol = 1e-14
    @test r.cfl_diff ≈ dt * max(νt, κt) * inv2 rtol = 1e-12          # ν = 0
    @test r.cfl_adv ≤ c.C_advective * (1 + 1e-12)                   # what the pick enforces
    @test r.solves == 1
    # a clamp that binds is the limiter
    _, dt2, _ = step_adaptive!(sim, CFLConstraints(; dt_max = 0.05))
    @test dt2 == 0.05 && r.limiter === :dt_max && calls[] == 2
    # a rejected attempt: one step with its retry, half the picked dt
    tries = Ref(0)
    flaky(s, h; kw...) = (tries[] += 1; tries[] == 1 && throw(NumericalFailure("first")); step!(s, h; kw...))
    reset_interval!(r)
    dpick = cfl_diagnostics(sim, c).dt_limiting
    _, dt3, nr3 = with_logger(NullLogger()) do
        step_adaptive!(sim, c; stepper = flaky)
    end
    @test nr3 == 1 && tries[] == 2 && dt3 == dpick / 2
    @test calls[] == 3 && r.n == 1 && r.retries == 1 && r.retries_sum == 1 && r.limiter === :retry
    @test r.depth == 0 && r.solves == 1          # the throwing attempt never reached its solve
    # a prescribed dt with no pick of this state: :fixed; a driver's pick at another dt: :prescribed
    step_adaptive!(sim, c; dt = 0.1)
    @test r.limiter === :fixed && isnan(r.pick_t)
    cfl_diagnostics(sim, c)
    step_adaptive!(sim, c; dt = 0.1)
    @test r.limiter === :prescribed
    # retries exhausted: the error propagates, no step is recorded, the bracket closes
    s0 = r.steps
    @test_throws ErrorException with_logger(NullLogger()) do
        step_adaptive!(sim, c; stepper = (s, h; kw...) -> throw(NumericalFailure("always")),
                       max_retries = 2)
    end
    @test r.steps == s0 && r.depth == 0
end

@testset "sound-proof core: record, dt pick and solver statistics" begin
    m = RegularGrid(; size = (8, 8, 8), extent = (800.0, 800.0, 800.0), halo = (3, 3, 3))
    sim = AnelasticSimulation(m; ref = isothermal_reference(m; T_surf = 300.0),
                              bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
    rng = Xoshiro(2)
    set!(sim.u, (x, y, z) -> 3.0 + 0.5 * randn(rng))
    calls = hooked(sim)
    r = step_record(sim)
    c = CFLConstraints(; dt_max = 1000.0)
    dt = adaptive_dt(sim, c)
    step!(sim, dt)
    @test calls[] == 1 && r.dt == dt && r.limiter === :advective
    @test r.solves == 1 && r.solver_iters == solver_iterations(sim.solver) > 0
    @test r.solver_residual == solver_residual(sim.solver)
    @test r.cfl_adv ≈ c.C_advective rtol = 1e-12      # the advective limit binds
end

@testset "nested run: the run's step, each domain's accepted steps" begin
    pm = RegularGrid(Float64; size = (12, 12, 12), extent = (1200.0, 1200.0, 1200.0), halo = (3, 3, 3))
    nest = nested_grid(pm, ((300.0, 900.0), (300.0, 900.0), (0.0, 600.0)), 2)
    cm = nest_child_mesh(nest)
    p = BoussinesqSimulation(pm; bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
    c = BoussinesqSimulation(cm; bcs = nest_bcs(cm, :nest, :nest; bottom = :wall, top = :nest))
    fill!(p.u.data, 4.0)
    run = NestedRun(p, c, nest; feedback = :one_way, cfl_parent = CFLConstraints(; dt_max = 20.0),
                    cfl_child = CFLConstraints(; dt_max = 20.0))
    cr, cp, cc = hooked(run), hooked(p), hooked(c)
    rr = step_record(run)
    for _ in 1:3
        step_nested!(run)
    end
    @test cr[] == 3 && cp[] == 3 && rr.steps == 3
    @test cc[] == run.stats.child_substeps ≥ 6       # refinement 2: at least two substeps each
    @test step_record(p).steps == 3 && step_record(c).steps == run.stats.child_substeps
    @test rr.dt == step_record(p).dt == run.stats.dt_parent_max
    @test rr.limiter === step_record(p).limiter === :advective
    @test rr.cfl_adv == run.stats.max_child_cfl > 0    # the window's largest child-substep CFL
    @test step_record(p).cfl_adv ≈ 0.5 rtol = 1e-12     # the parent's pick: C_advective
    @test rr.retries == 0 && rr.wall_step > 0
end

@testset "Field reductions: the port's reductions equal serial ones" begin
    sim = sd_sim()
    u = sd_raw(sim.u)
    @test reduce_diagnostic(:max, sim.u) == maximum(u)
    @test reduce_diagnostic(:min, sim.u) == minimum(u)
    @test reduce_diagnostic(:maxabs, sim.w) == maximum(abs, sd_raw(sim.w))
    @test reduce_diagnostic(:sum, sim.u) ≈ sum(u) rtol = 1e-14
    @test reduce_diagnostic(:mean, sim.θ) ≈ sum(sd_raw(sim.θ)) / length(sd_raw(sim.θ)) rtol = 1e-14
    @test reduce_diagnostic(:rms, sim.u) ≈ sqrt(sum(abs2, u) / length(u)) rtol = 1e-14
    @test reduce_diagnostic(:sum, sim.u) ==
          Aeolus.Backends.interior_sum(sim.u.data, Aeolus.Grids.interior(sim.mesh))
    # a non-finite value propagates
    sim.u.data[5, 5, 5] = NaN
    @test isnan(reduce_diagnostic(:max, sim.u)) && isnan(reduce_diagnostic(:mean, sim.u))
    @test_throws ArgumentError reduce_diagnostic(:median, sim.u)
end

@testset "registry: step-record, state-CFL and process diagnostics" begin
    sim = sd_sim()
    ds = diagnostics(sim)
    names = [d.name for d in ds]
    for n in ("cfl_adv", "cfl_diff", "solver_iters", "solver_residual", "implicit_iters")
        @test n in names && ds[findfirst(==(n), names)].cadence === :step
    end
    for n in ("wall_step", "wall_pressure", "wall_sgs", "wall_cfl", "wall_guard",
              "cfl_adv_state", "cfl_diff_state")
        @test n in names && ds[findfirst(==(n), names)].cadence === :costly
    end
    ev(n) = evaluate_diagnostic(ds[findfirst(==(n), names)])
    step!(sim, 0.6)
    @test ev("cfl_adv_state") ≈ 0.6 * sd_adv_rate(sim) rtol = 1e-14
    m = sim.mesh
    @test ev("cfl_diff_state") ≈ 0.6 * max(maximum(sd_raw(sim.νt)), maximum(sd_raw(sim.κt))) *
                                  (1 / m.dx^2 + 1 / m.dy^2 + 1 / m.dz^2) rtol = 1e-12
    @test ev("solver_iters") == 0 && isnan(ev("cfl_adv"))
    pd = process_diagnostics()
    @test [d.name for d in pd] == ["mem_live_mb", "mem_maxrss_mb", "gc_time"]
    v = evaluate_diagnostic.(pd)
    @test all(isfinite, v) && v[1] > 0 && v[2] ≥ v[1] && v[3] ≥ 0
end

function sd_csv(path)
    lines = readlines(path)
    hdr = split(lines[1], ',')
    rows = [split(l, ',') for l in lines[2:end]]
    col(name) = [r[findfirst(h -> h == name || startswith(h, name * " ["), hdr)] for r in rows]
    return (; hdr, rows, col)
end

@testset "attached monitor: one row per step, with the step's values" begin
    sim = sd_sim()
    c = CFLConstraints(; dt_max = 100.0)
    dir = mktempdir()
    path = joinpath(dir, "d.csv")
    expected_dt = Float64[]; expected_cfl = Float64[]; expected_lim = String[]
    with_logger(NullLogger()) do
        with_monitor(sim; csv = path, cadence = 3.0, t_end = 1e6, progress_every = 1e9) do m
            @test m.attached && step_hook(sim) isa MonitorHook
            @test_throws ArgumentError attach_monitor!(RunMonitor(sim))   # one hook per sim
            for n in 1:5
                d = cfl_diagnostics(sim, c)
                _, dt, _ = step_adaptive!(sim, c)
                push!(expected_dt, dt); push!(expected_cfl, dt * d.advective_rate)
                push!(expected_lim, string(d.limiter))
            end
            step!(sim, 0.25)                                  # a fixed step: no pick
            push!(expected_dt, 0.25); push!(expected_cfl, NaN); push!(expected_lim, "fixed")
        end
    end
    @test step_hook(sim) === nothing                          # closing removes the hook
    t = sd_csv(path)
    @test length(t.rows) == 6
    @test parse.(Int, t.col("step")) == 1:6
    @test parse.(Float64, t.col("dt")) == expected_dt
    @test t.col("dt_limiter") == expected_lim
    cfl = parse.(Float64, t.col("cfl_adv"))
    @test cfl[1:5] == expected_cfl[1:5] && isnan(cfl[6])
    @test all(==(0.0), parse.(Float64, t.col("solver_iters")))
    # costly cadence: the first row and every 3 s simulated
    costly = .!isempty.(t.col("wall_step"))
    @test costly[1] && 1 < count(costly) < 6
    @test any(h -> startswith(h, "proc.mem_maxrss_mb"), t.hdr)
    # step0: a resumed run continues the numbering; monitored_step! adds no row
    path2 = joinpath(dir, "d2.csv")
    with_logger(NullLogger()) do
        with_monitor(sim; csv = path2, step0 = 40) do m
            for n in 41:43
                monitored_step!(m, sim; step = n) do
                    step_adaptive!(sim, c)
                end
            end
        end
    end
    @test parse.(Int, sd_csv(path2).col("step")) == 41:43
end

@testset "driver-reported rows: the steps since the previous row" begin
    sim = sd_sim()
    path = joinpath(mktempdir(), "e.csv")
    with_logger(NullLogger()) do
        with_monitor(sim; csv = path, attach = false, progress_every = 1e9) do m
            @test !m.attached && step_hook(sim) === nothing
            for n in 1:3                                     # a driver step = two core steps
                step!(sim, 0.2); step_rk3!(sim, 0.3)
                monitor!(m, sim; step = n)
            end
        end
    end
    t = sd_csv(path)
    @test length(t.rows) == 3
    @test parse.(Float64, t.col("dt")) ≈ fill(0.5, 3)        # the time the row covers
    @test t.col("dt_limiter") == fill("fixed", 3)
end

@testset "attached monitor on a nested run: one row per parent step" begin
    pm = RegularGrid(Float64; size = (12, 12, 12), extent = (1200.0, 1200.0, 1200.0), halo = (3, 3, 3))
    nest = nested_grid(pm, ((300.0, 900.0), (300.0, 900.0), (0.0, 600.0)), 2)
    cm = nest_child_mesh(nest)
    p = BoussinesqSimulation(pm; bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
    c = BoussinesqSimulation(cm; bcs = nest_bcs(cm, :nest, :nest; bottom = :wall, top = :nest))
    fill!(p.u.data, 4.0)
    run = NestedRun(p, c, nest; feedback = :one_way, cfl_parent = nothing, cfl_child = nothing)
    path = joinpath(mktempdir(), "n.csv")
    with_logger(NullLogger()) do
        with_monitor(run; csv = path, cadence = 0.0) do m
            for _ in 1:4
                step_nested!(run; dt = 5.0)
            end
        end
    end
    t = sd_csv(path)
    @test length(t.rows) == 4 && parse.(Float64, t.col("dt")) == fill(5.0, 4)
    @test parse.(Float64, t.col("child.solver_iters")) == fill(0, 4)
    @test parse.(Float64, t.col("cfl_adv")) ≈ fill(run.stats.max_child_cfl, 4) rtol = 1e-12
    # each row is one parent step of two child substeps (refinement 2)
    @test parse.(Float64, t.col("child_substeps")) == [2, 4, 6, 8]
end

@testset "two-way nested run: the conservation ledger as diagnostics" begin
    pm = RegularGrid(Float64; size = (12, 12, 12), extent = (1200.0, 1200.0, 1200.0), halo = (3, 3, 3))
    nest = nested_grid(pm, ((300.0, 900.0), (300.0, 900.0), (0.0, 600.0)), 2)
    cm = nest_child_mesh(nest)
    p = BoussinesqSimulation(pm; θ₀ = 300.0, bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
    c = BoussinesqSimulation(cm; θ₀ = 300.0, bcs = nest_bcs(cm, :nest, :nest; bottom = :wall, top = :nest))
    fill!(p.u.data, 4.0); fill!(p.θ.data, 300.0); fill!(c.θ.data, 300.0)
    run = NestedRun(p, c, nest; feedback = :two_way, cfl_parent = nothing, cfl_child = nothing)
    step_nested!(run; dt = 5.0)
    ds = collect_diagnostics(run)
    led = [d for d in ds if startswith(d.name, "ledger_")]
    @test [d.name for d in led] == ["ledger_" * string(k) for k in vcat(run.work.keys, [:momentum_x, :mass])]
    for d in led
        key = Symbol(d.name[8:end])
        v = evaluate_diagnostic(d)
        @test haskey(nest.ledger, key) ? v == nest.ledger[key] : isnan(v)
    end
    @test haskey(nest.ledger, :theta) && evaluate_diagnostic(led[1]) == nest.ledger[:theta]
end
