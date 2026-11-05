# Diagnostics registry, CSV writer and progress line (LOG-c, #335; design
# in epic #293): src/logging/diagnostics.jl and
# src/logging/component_diagnostics.jl.
#
# The acceptance tests of the issue:
#   * the registry collects the components' declarations (simulations,
#     closures, wall models, nesting, a user component);
#   * the CSV values equal independently computed ones: each diagnostic is
#     recomputed here from the raw arrays (own loops, own divergence, own
#     friction velocity), and the CSV cells read back are the Float64s
# the writer evaluated;
#   * the costly diagnostics follow their cadence, the cheap ones every row;
#   * the progress line carries %, steps/s, ETA, dt, limiter and the last
#     warning.

using Test
using Logging
using Random
using Aeolus
# the wall-bounded benchmark harness (half channel, channel) is a test fixture (ONE-l, #475)
isdefined(@__MODULE__, :ChannelCases) || include(joinpath(@__DIR__, "fixtures", "channel_cases.jl"))
using .ChannelCases
using Aeolus.Fields: interior_view, set!
using Aeolus.Nesting
using Aeolus.TimeIntegration: step_adaptive!, CFLConstraints, cfl_diagnostics
using Aeolus.RunLog: format_duration, format_simtime

function dg_sim(; N = (12, 10, 8), C_s = 0.17, seed = 5)
    m = RegularGrid(; size = N, extent = (600.0, 500.0, 400.0), halo = (3, 3, 3))
    sim = BoussinesqSimulation(m; θ₀ = 300.0, C_s = C_s,
                               bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
    rng = Xoshiro(seed)
    set!(sim.u, (x, y, z) -> 2.0 + 0.4 * randn(rng))
    set!(sim.v, (x, y, z) -> -0.5 + 0.4 * randn(rng))
    set!(sim.θ, (x, y, z) -> 300.0 + 0.003 * z + 0.05 * randn(rng))
    Aeolus.TimeIntegration._project!(sim, 1.0)
    return sim
end

# raw interior block of a field's storage (no interior_view)
dg_raw(f) = (m = f.mesh; f.data[m.Hx+1:m.Hx+m.Nx, m.Hy+1:m.Hy+m.Ny, m.Hz+1:m.Hz+m.Nz])

"The diagnostics of a BoussinesqSimulation, computed independently."
function dg_independent(sim)
    m = sim.mesh
    u, v, w, θ = dg_raw(sim.u), dg_raw(sim.v), dg_raw(sim.w), dg_raw(sim.θ)
    ke = 0.0
    for I in eachindex(u)
        ke += u[I]^2 + v[I]^2 + w[I]^2
    end
    ke_mean = 0.5 * ke * m.dx * m.dy * m.dz / (m.Lx * m.Ly * m.Lz)
    # the divergence of the staggered velocity, by hand, on the halo-filled data
    fill_halos!(sim.u, sim.bcs); fill_halos!(sim.v, sim.bcs); fill_halos!(sim.w, sim.bcs)
    U, V, W = sim.u.data, sim.v.data, sim.w.data
    dmax = 0.0
    for k in 1:m.Nz, j in 1:m.Ny, i in 1:m.Nx
        I, J, K = m.Hx + i, m.Hy + j, m.Hz + k
        d = (U[I+1, J, K] - U[I, J, K]) / m.dx + (V[I, J+1, K] - V[I, J, K]) / m.dy +
            (W[I, J, K+1] - W[I, J, K]) / m.dz
        dmax = max(dmax, abs(d))
    end
    return Dict("ke_mean" => ke_mean, "max_abs_u" => maximum(abs, u),
                "max_abs_v" => maximum(abs, v), "max_abs_w" => maximum(abs, w),
                "theta_min" => minimum(θ), "theta_max" => maximum(θ), "max_div" => dmax,
                "nu_t_max" => maximum(dg_raw(sim.νt)))
end

"Friction velocities of the bottom wall by hand: the law at the first-cell centre speed."
function dg_ustar(sim, law)
    m = sim.mesh
    fill_halos!(sim.u, sim.bcs); fill_halos!(sim.v, sim.bcs)
    U, V = sim.u.data, sim.v.data
    us = Float64[]
    K = m.Hz + 1
    for j in 1:m.Ny, i in 1:m.Nx
        I, J = m.Hx + i, m.Hy + j
        uc = (U[I, J, K] + U[I+1, J, K]) / 2
        vc = (V[I, J, K] + V[I, J+1, K]) / 2
        push!(us, friction_velocity(law, hypot(uc, vc), m.dz / 2))
    end
    return us
end

"Parse a diagnostics CSV: header names and rows of cells (strings)."
function dg_csv(path)
    lines = readlines(path)
    return split(lines[1], ','), [split(l, ',') for l in lines[2:end]]
end

struct DGComponent
    scale::Float64
end
Aeolus.RunLog.diagnostics(c::DGComponent, sim) = Diagnostic[
    Diagnostic("scaled_t", "s", () -> c.scale * sim.t; cadence = :step),
    Diagnostic("u_sum", "m s-1", () -> interior_view(sim.u); reduction = :sum),
]
struct DGDuplicate end
Aeolus.RunLog.diagnostics(::DGDuplicate, sim) =
    [Diagnostic("a", "1", () -> 1.0), Diagnostic("a", "1", () -> 2.0)]

@testset "Diagnostics registry, CSV writer, progress line (LOG-c, #335)" begin

@testset "Diagnostic: declaration and validation" begin
    d = Diagnostic("max_abs_w", "m s-1", () -> [1.0, -3.0]; reduction = :maxabs)
    @test d.cadence === :costly && d.reduction === :maxabs && d.units == "m s-1"
    @test evaluate_diagnostic(d) == 3.0
    @test_throws ArgumentError Diagnostic("bad name", "1", () -> 0.0)
    @test_throws ArgumentError Diagnostic("a,b", "1", () -> 0.0)
    @test_throws ArgumentError Diagnostic("x", "1", () -> 0.0; cadence = :sometimes)
    @test_throws ArgumentError Diagnostic("x", "1", () -> 0.0; reduction = :median)
    # :value needs a number
    @test_throws ArgumentError evaluate_diagnostic(Diagnostic("x", "1", () -> [1.0]))
    # the default: no diagnostics
    @test diagnostics(:anything, nothing) == Diagnostic[]
    @test diagnostics(42) == Diagnostic[]
end

@testset "reduce_diagnostic: every reduction against its definition" begin
    A = [3.0 -7.5; 0.25 2.0]
    @test reduce_diagnostic(:max, A) == 3.0
    @test reduce_diagnostic(:min, A) == -7.5
    @test reduce_diagnostic(:maxabs, A) == 7.5
    @test reduce_diagnostic(:sum, A) == 3.0 - 7.5 + 0.25 + 2.0
    @test reduce_diagnostic(:mean, A) == (3.0 - 7.5 + 0.25 + 2.0) / 4
    @test reduce_diagnostic(:rms, A) ≈ sqrt((9 + 56.25 + 0.0625 + 4) / 4) rtol = 1e-15
    # a non-finite value propagates, so a blowup shows in the diagnostics
    B = [1.0, NaN, 5.0]
    for r in (:max, :min, :maxabs, :sum, :mean, :rms)
        @test isnan(reduce_diagnostic(r, B))
    end
    @test reduce_diagnostic(:max, [1.0, Inf]) == Inf
    @test isnan(reduce_diagnostic(:max, Float64[]))
end

@testset "a source that throws gives NaN and one warning" begin
    d = Diagnostic("broken", "1", () -> error("no data"))
    lg = Test.TestLogger(; min_level = Logging.Warn)
    v = with_logger(lg) do
        (evaluate_diagnostic(d), evaluate_diagnostic(d))
    end
    @test all(isnan, v)
    @test length(lg.logs) == 1                       # maxlog = 1
    r = lg.logs[1]
    @test r.level == Logging.Warn && Dict(r.kwargs)[:subsystem] === :diagnostics
    @test Dict(r.kwargs)[:diagnostic] == "broken"
end

@testset "values equal independently computed ones; CSV round trip; cadence" begin
    sim = dg_sim()
    law = RoughLogLaw(0.05)
    wm = WallModel(law)
    closure = StaticSmagorinsky(0.17)
    # the step cores' record diagnostics (#336) are tested against
    # independent values in logging_step_diagnostics.jl; the wall times are
    # not reproducible
    ds = filter(d -> !(d.name in ("cfl_adv", "cfl_diff", "solver_iters", "solver_residual",
                                  "implicit_iters", "cfl_adv_state", "cfl_diff_state") ||
                       startswith(d.name, "wall_")),
                collect_diagnostics(sim, (; sgs = closure, wall = wm, user = DGComponent(3.0))))
    path = joinpath(mktempdir(), "diagnostics.csv")
    w = DiagnosticsWriter(path, ds; cadence = 5.0)
    cfl = CFLConstraints(dt_max = 10.0)
    expected = Dict{Int,Dict{String,Float64}}()
    costly_steps = Int[]
    bad = String[]
    for n in 1:12
        step_adaptive!(sim, cfl; dt = 1.0, sgs = closure)
        # an independent copy of the state, before any diagnostic touches it
        ind = dg_independent(sim)
        us = dg_ustar(sim, law)
        ind["sgs.nu_h_max"] = maximum(dg_raw(sim.νt))
        ind["sgs.nu_v_max"] = maximum(dg_raw(sim.νt))
        ind["sgs.kappa_h_max"] = maximum(dg_raw(sim.κt))
        ind["wall.ustar_mean"] = sum(us) / length(us)
        ind["wall.ustar_max"] = maximum(us)
        ind["user.scaled_t"] = 3.0 * sim.t
        ind["user.u_sum"] = sum(dg_raw(sim.u))
        expected[n] = ind
        costly = record_diagnostics!(w, sim.t; step = n, dt = 1.0, limiter = "advective", retries = 0)
        costly && push!(costly_steps, n)
        # what the writer evaluated equals the independent value
        for d in ds
            (d.cadence === :step || costly) || continue
            isapprox(w.last[d.name], ind[d.name]; rtol = 1e-12, atol = 1e-15) || push!(bad, "$n $(d.name)")
        end
    end
    @test isempty(bad)
    close(w)
    # cadence 5 s at dt = 1 s: the first row, then every 5 s
    @test costly_steps == [1, 6, 11]
    header, rows = dg_csv(path)
    @test header[1:6] == ["step [1]", "t [s]", "wall [s]", "dt [s]", "dt_limiter", "retries [1]"]
    @test header[7:end] == ["$(d.name) [$(d.units)]" for d in ds]
    @test length(rows) == 12
    ok = true
    for (n, r) in enumerate(rows)
        ok &= parse(Int, r[1]) == n
        ok &= parse(Float64, r[2]) == n * 1.0          # the clock, exactly
        ok &= r[5] == "advective" && r[6] == "0"
        for (c, d) in enumerate(ds)
            cell = r[6+c]
            if d.cadence === :step || n in costly_steps
                # the cell is the Float64 that was evaluated, and that is the independent value
                ok &= isapprox(parse(Float64, cell), expected[n][d.name]; rtol = 1e-12, atol = 1e-15)
            else
                ok &= cell == ""
            end
        end
    end
    @test ok
    # the cells round-trip exactly: w.last is the last costly row (row 11)
    # for the costly diagnostics, the last row (12) for the :step one
    @test all(parse(Float64, (d.cadence === :step ? rows[12] : rows[11])[6+c]) === w.last[d.name]
              for (c, d) in enumerate(ds))
    # last_diagnostics: in column order, with when
    ld = last_diagnostics(w)
    @test [x.name for x in ld] == [d.name for d in ds]
    @test ld[1].step == 11 && ld[1].t == 11.0
    @test ld[end-1].step == 12 && ld[end].step == 11   # user.scaled_t is :step: the latest row
end

@testset "progress line: %, steps/s, ETA, dt, limiter, last warning" begin
    @test format_duration(3725) == "1:02:05"
    @test format_duration(59.6) == "0:01:00"
    @test format_duration(NaN) == "--:--:--"
    @test format_simtime(30.0) == "30.0 s" && format_simtime(1800.0) == "30.0 min"
    @test format_simtime(9000.0) == "2.5 h" && format_simtime(2 * 86400.0) == "2.0 d"
    @test_throws ArgumentError ProgressLine(10.0; t_start = 10.0)

    p = ProgressLine(1000.0; t_start = 100.0, every = 0.0)
    lg = Test.TestLogger(; min_level = Logging.Debug)
    with_logger(lg) do
        with_progress(p) do
            sleep(0.05)
            @test progress!(p, 280.0; step = 30, dt = 2.5, limiter = :advective)
            @warn "solver did not converge" subsystem = :pressure
            sleep(0.05)
            @test progress!(p, 400.0; step = 70, dt = 2.0, limiter = :diffusive)
        end
    end
    recs = [r for r in lg.logs if get(Dict(r.kwargs), :subsystem, nothing) === :progress]
    @test length(recs) == 2
    k1 = Dict(recs[1].kwargs); k2 = Dict(recs[2].kwargs)
    @test recs[1].level == Logging.Info
    @test k1[:percent] == 20.0
    @test k2[:percent] ≈ 100 * 300 / 900 atol = 0.01
    @test k1[:step] == 30 && k1[:dt] == 2.5 && k1[:limiter] == "advective"
    # ETA = elapsed · (1 − f) / f
    @test isfinite(k1[:eta_s]) && k1[:eta_s] > 0
    @test k2[:eta_s] < k1[:eta_s] * 1.5
    @test k1[:steps_per_s] > 0 && k2[:steps_per_s] > 0
    @test k1[:last_warning] == ""
    @test k2[:last_warning] == "solver did not converge"     # noted by the WarningTap
    @test occursin("ETA", recs[2].message) && occursin("(diffusive)", recs[2].message)
    @test occursin("last warning", recs[2].message) && occursin("33.3 %", recs[2].message)
    @test p.lines == 2
    # the tap passes the warning on unchanged
    @test any(r -> r.level == Logging.Warn && r.message == "solver did not converge", lg.logs)
    # throttled: not again within `every`
    q = ProgressLine(10.0; every = 1e9)
    @test with_logger(() -> progress!(q, 1.0; step = 1), NullLogger()) == false
    @test with_logger(() -> progress!(q, 1.0; step = 1, force = true), NullLogger()) == true
end

end
