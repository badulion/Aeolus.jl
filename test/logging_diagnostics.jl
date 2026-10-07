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
#     the writer evaluated, bit for bit;
#   * the costly diagnostics follow their cadence, the cheap ones every row;
#   * the progress line carries %, steps/s, ETA, dt, limiter and the last
#     warning.

using Test
using Logging
using Random
using Aeolus
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

@testset "the registry collects the components' declarations" begin
    sim = dg_sim()
    law = RoughLogLaw(0.05)
    closure = StaticSmagorinsky(0.17)
    ds = collect_diagnostics(sim, (; sgs = closure, wall = WallModel(law), user = DGComponent(2.0)))
    names = [d.name for d in ds]
    # the core's own, then the step cores' record (#336, every row except the
    # wall times and the CFL of the state), then the components'
    rec = ["cfl_adv", "cfl_diff", "solver_iters", "solver_residual",
           "wall_step", "wall_pressure", "wall_sgs", "wall_cfl", "wall_guard",
           "implicit_iters", "cfl_adv_state", "cfl_diff_state"]
    @test names == ["ke_mean", "max_abs_u", "max_abs_v", "max_abs_w", "theta_min", "theta_max",
                    "max_div", "nu_t_max", rec...,
                    "sgs.nu_h_max", "sgs.nu_v_max", "sgs.kappa_h_max",
                    "wall.ustar_mean", "wall.ustar_max",
                    "user.scaled_t", "user.u_sum"]
    every = ("cfl_adv", "cfl_diff", "solver_iters", "solver_residual", "implicit_iters")
    @test all(d -> d.cadence === (d.name in every ? :step : :costly), ds[1:end-2])
    @test ds[end-1].cadence === :step
    @test [d.units for d in ds[1:8]] ==
          ["m2 s-2", "m s-1", "m s-1", "m s-1", "K", "K", "s-1", "m2 s-1"]
    # without the sim, and with moisture
    @test [d.name for d in collect_diagnostics(sim, (; user = DGComponent(1.0)); include_sim = false)] ==
          ["user.scaled_t", "user.u_sum"]
    mm = RegularGrid(; size = (4, 4, 4), extent = (1.0, 1.0, 1.0), halo = (3, 3, 3))
    @test "qv_max" in [d.name for d in diagnostics(BoussinesqSimulation(mm; moisture = true))]
    # a name twice is an error
    @test_throws ArgumentError collect_diagnostics(sim, (; d = DGDuplicate()))
    @test [d.name for d in prefixed("a.b", [Diagnostic("x", "1", () -> 0.0)])] == ["a.b.x"]

    # TKE closure: the generic closure diagnostics plus its energy
    tke = TKEClosure(sim.mesh)
    interior_view(tke.e) .= reshape(collect(1.0:length(interior_view(tke.e))), size(interior_view(tke.e))) ./ 100
    td = diagnostics(tke, sim)
    @test [d.name for d in td] == ["nu_h_max", "nu_v_max", "kappa_h_max", "e_mean", "e_max"]
    n = length(interior_view(tke.e))
    @test evaluate_diagnostic(td[4]) ≈ (n + 1) / 2 / 100 rtol = 1e-14
    @test evaluate_diagnostic(td[5]) == n / 100

    # TBLE wall model: one pair per wall it has run on
    g = RegularGrid(; size = (8, 6, 8), extent = (4.0, 3.0, 2.0), halo = (2, 2, 2))
    ws = BoussinesqSimulation(g; bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
    interior_view(ws.u) .= 2.0; interior_view(ws.v) .= 1.0
    wm = TBLEWallModel(; ν = 1e-4, terms = :pressure_gradient, pressure = false)
    @test diagnostics(wm, ws) == Diagnostic[]          # no wall used yet
    apply_wall_stress!(ws, wm, 1e-5; side = :both)
    wd = diagnostics(wm, ws)
    @test [d.name for d in wd] == ["bottom.utau_mean", "bottom.kappa_wm", "top.utau_mean", "top.kappa_wm"]
    cols, act = tble_columns(wm, :bottom)
    @test evaluate_diagnostic(wd[1]) ≈ sum(c.u_τ for c in cols) / length(cols) rtol = 1e-14
    @test evaluate_diagnostic(wd[2]) == wm.walls[:bottom].κ_dyn

    # WallBoundedCase: the stresses of its last step, every step
    case = half_channel_case(; N = 8)
    advance_channel!(case, 1e-3)
    cd = diagnostics(case, case.sim)
    @test [d.name for d in cd] == ["tau_wall_bottom", "tau_wall_top"]
    @test all(d -> d.cadence === :step, cd)
    @test evaluate_diagnostic(cd[1]) == case.τ_wall_bottom
    @test evaluate_diagnostic(cd[2]) == case.τ_wall_top
end

@testset "nested run and sound-proof domains" begin
    pm = RegularGrid(Float64; size = (12, 12, 12), extent = (1200.0, 1200.0, 1200.0),
                     halo = (3, 3, 3))
    nest = nested_grid(pm, ((300.0, 900.0), (300.0, 900.0), (0.0, 600.0)), 2)
    cm = nest_child_mesh(nest)
    rp = isothermal_reference(pm; T_surf = 300.0); rc = isothermal_reference(cm; T_surf = 300.0)
    pA = AnelasticSimulation(pm; ref = rp, bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
    cA = AnelasticSimulation(cm; ref = rc,
                             bcs = nest_bcs(cm, :nest, :nest; bottom = :wall, top = :nest))
    fill!(pA.u.data, 4.0)
    rng = Xoshiro(3)
    interior_view(pA.v) .+= 0.3 .* randn(rng, size(interior_view(pA.v)))
    Aeolus.Anelastic._project_mass_weighted!(pA)
    run = NestedRun(pA, cA, nest; feedback = :one_way, cfl_parent = nothing, cfl_child = nothing)
    for n in 1:2
        step_nested!(run; dt = 5.0)
    end
    ds = collect_diagnostics(run)
    names = [d.name for d in ds]
    @test names[1:11] == ["parent_steps", "child_substeps", "parent_retries", "child_retries",
                          "substeps_last", "max_child_cfl", "max_clock_error",
                          "wall_parent", "wall_child", "wall_boundary", "wall_feedback"]
    @test "parent.ke" in names && "child.max_div" in names && "parent.theta_max" in names
    v = Dict(d.name => evaluate_diagnostic(d) for d in ds)
    @test v["parent_steps"] == run.stats.parent_steps == 2
    @test v["child_substeps"] == run.stats.child_substeps
    @test v["wall_child"] == run.stats.wall_child
    # the sound-proof core's ke is its density-weighted energy, by hand
    ρc, ρf = rp.ρ̄c_pad, rp.ρ̄f_pad
    ke = 0.0
    for k in 1:pm.Nz, j in 1:pm.Ny, i in 1:pm.Nx
        I, J, K = pm.Hx + i, pm.Hy + j, pm.Hz + k
        ke += ρc[K] * (pA.u.data[I, J, K]^2 + pA.v.data[I, J, K]^2) + ρf[K] * pA.w.data[I, J, K]^2
    end
    @test v["parent.ke"] ≈ 0.5 * ke * pm.dx * pm.dy * pm.dz rtol = 1e-13
    # and max_div is the anelastic residual, at the solver tolerance
    @test v["parent.max_div"] == Aeolus.Anelastic.max_anelastic_divergence(pA)
    @test v["parent.max_div"] < 1e-8
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
            @test w.last[d.name] ≈ ind[d.name] rtol = 1e-12 atol = 1e-15
        end
    end
    close(w)
    # cadence 5 s at dt = 1 s: the first row, then every 5 s
    @test costly_steps == [1, 6, 11]
    header, rows = dg_csv(path)
    @test header[1:6] == ["step [1]", "t [s]", "wall [s]", "dt [s]", "dt_limiter", "retries [1]"]
    @test header[7:end] == ["$(d.name) [$(d.units)]" for d in ds]
    @test length(rows) == 12
    for (n, r) in enumerate(rows)
        @test parse(Int, r[1]) == n
        @test parse(Float64, r[2]) == n * 1.0          # the clock, exactly
        @test r[5] == "advective" && r[6] == "0"
        for (c, d) in enumerate(ds)
            cell = r[6+c]
            if d.cadence === :step || n in costly_steps
                # the cell is the Float64 that was evaluated, and that is the independent value
                @test parse(Float64, cell) ≈ expected[n][d.name] rtol = 1e-12 atol = 1e-15
            else
                @test cell == ""
            end
        end
    end
    # the cells round-trip exactly: w.last is the last costly row (row 11)
    # for the costly diagnostics, the last row (12) for the :step one
    for (c, d) in enumerate(ds)
        r = d.cadence === :step ? rows[12] : rows[11]
        @test parse(Float64, r[6+c]) === w.last[d.name]
    end
    # last_diagnostics: in column order, with when
    ld = last_diagnostics(w)
    @test [x.name for x in ld] == [d.name for d in ds]
    @test ld[1].step == 11 && ld[1].t == 11.0
    @test ld[end-1].step == 12 && ld[end].step == 11   # user.scaled_t is :step: the latest row
end

@testset "CSV: non-finite values, append on restart, other columns go elsewhere" begin
    dir = mktempdir(); path = joinpath(dir, "d.csv")
    x = Ref(1.0)
    ds = [Diagnostic("x", "1", () -> x[]; cadence = :step)]
    w = DiagnosticsWriter(path, ds)
    for (n, v) in enumerate((1.5, NaN, Inf, -Inf, 1e-300))
        x[] = v
        record_diagnostics!(w, Float64(n); step = n)
    end
    close(w)
    _, rows = dg_csv(path)
    @test [r[7] for r in rows] == ["1.5", "NaN", "Inf", "-Inf", "1.0e-300"]
    @test parse(Float64, rows[5][7]) === 1e-300
    @test rows[1][4] == "NaN" && rows[1][5] == ""     # dt not given
    # a restart appends below the same header
    w2 = DiagnosticsWriter(path, ds)
    record_diagnostics!(w2, 6.0; step = 6)
    close(w2)
    lines = readlines(path)
    @test length(lines) == 7 && count(l -> startswith(l, "step"), lines) == 1
    @test startswith(lines[end], "6,6.0,")
    # other columns: the old file is left alone, the rows go to d-2.csv
    ds3 = [Diagnostic("y", "K", () -> 2.0)]
    w3 = @test_logs (:warn, r"other columns") DiagnosticsWriter(path, ds3)
    record_diagnostics!(w3, 7.0; step = 7)
    close(w3)
    @test w3.path == joinpath(dir, "d-2.csv")
    @test readlines(path) == lines
    @test readlines(w3.path)[1] == "step [1],t [s],wall [s],dt [s],dt_limiter,retries [1],y [K]"
    # append = false starts over
    w4 = DiagnosticsWriter(path, ds; append = false); close(w4)
    @test length(readlines(path)) == 1
    @test_throws ArgumentError DiagnosticsWriter(path, [Diagnostic("t", "s", () -> 0.0)])
    @test_throws ArgumentError DiagnosticsWriter(path, [ds; ds])
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

@testset "the writer and the progress line never print" begin
    path = tempname()
    out = open(path, "w") do io
        redirect_stdout(io) do
            redirect_stderr(io) do
                with_logger(NullLogger()) do
                    sim = dg_sim(; N = (6, 6, 6))
                    w = DiagnosticsWriter(tempname(), collect_diagnostics(sim); cadence = 0.0)
                    p = ProgressLine(10.0; every = 0.0)
                    for n in 1:3
                        step_adaptive!(sim, CFLConstraints(dt_max = 1.0))
                        record_diagnostics!(w, sim.t; step = n)
                        progress!(p, sim.t; step = n)
                    end
                    close(w)
                end
            end
        end
    end
    @test read(path, String) == ""
end

end
