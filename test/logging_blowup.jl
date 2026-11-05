# Blowup: diagnose, dump, stop; the run monitor (LOG-c, #335; design in
# epic #293): src/io/blowup.jl, src/io/run_monitor.jl.
#
# The acceptance test of the issue: a blowup produces a diagnosis plus a
# crash checkpoint.
#   * the diagnosis names the non-finite field, how many values, where
#     (index and coordinates, checked against the injected cell) and when,
#     with the last diagnostics;
#   * the crash checkpoint holds the last good state, and a
#     run restarts from it;
#   * the run stops with a BlowupError; inside with_run the session is
#     recorded as failed and run.log keeps the diagnosis.

using Test
using Logging
using Random
using Aeolus
using Aeolus.Fields: interior_view, set!
using Aeolus.Grids: xnodes, ynodes, znodes
using Aeolus.TimeIntegration: step_adaptive!, CFLConstraints

function bu_sim(; seed = 3)
    m = RegularGrid(; size = (10, 8, 8), extent = (500.0, 400.0, 400.0), halo = (3, 3, 3))
    sim = BoussinesqSimulation(m; θ₀ = 300.0, C_s = 0.17,
                               bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
    rng = Xoshiro(seed)
    set!(sim.u, (x, y, z) -> 1.5 + 0.3 * randn(rng))
    set!(sim.θ, (x, y, z) -> 300.0 + 0.003 * z + 0.05 * randn(rng))
    Aeolus.TimeIntegration._project!(sim, 1.0)
    return sim
end

const BU_GIT = `git -c user.name=aeolus-test -c user.email=test@example.org -c commit.gpgsign=false`
function bu_repo()
    d = mktempdir()
    run(`git -C $d init -q`)
    write(joinpath(d, "model.jl"), "x = 1\n")
    run(`$BU_GIT -C $d add model.jl`)
    run(`$BU_GIT -C $d commit -q -m initial`)
    return d
end

blowup_kwargs(r) = Dict{Symbol,Any}(r.kwargs)

@testset "Blowup: diagnose, dump, stop; run monitor (LOG-c, #335)" begin

@testset "allfinite: the branch-free test agrees with all(isfinite, ·)" begin
    a = randn(Xoshiro(1), 10, 9, 8); v = view(a, 2:9, 2:8, 2:7)
    @test Aeolus.IO.allfinite(v)
    for (I, x) in (((5, 5, 5), NaN), ((2, 2, 2), Inf), ((9, 8, 7), -Inf))
        b = copy(a); b[I...] = x
        @test !Aeolus.IO.allfinite(view(b, 2:9, 2:8, 2:7))
        @test Aeolus.IO.allfinite(view(b, 2:9, 2:8, 2:7)) == all(isfinite, view(b, 2:9, 2:8, 2:7))
    end
    b = copy(a); b[1, 1, 1] = NaN                       # outside the view
    @test Aeolus.IO.allfinite(view(b, 2:9, 2:8, 2:7))
    @test Aeolus.IO.allfinite(fill(floatmax(Float64), 3, 3, 3))   # huge but finite
    @test Aeolus.IO.allfinite(Float32[1 2; 3 4]) && !Aeolus.IO.allfinite([1.0, NaN])
end

@testset "nonfinite_fields: which field, how many, where" begin
    sim = bu_sim()
    @test isempty(nonfinite_fields(sim))
    m = sim.mesh
    sim.w.data[m.Hx+4, m.Hy+2, m.Hz+7] = Inf
    sim.θ.data[m.Hx+9, m.Hy+5, m.Hz+3] = NaN
    sim.θ.data[m.Hx+1, m.Hy+6, m.Hz+3] = NaN
    nf = nonfinite_fields(sim)
    @test [f.name for f in nf] == [:w, :θ]
    w, θ = nf
    @test w.count == 1 && w.inf == 1 && w.nan == 0 && w.index == (4, 2, 7)
    # the coordinates are those of the field's own (staggered) location
    @test w.position == (xnodes(m, FaceZ())[4], ynodes(m, FaceZ())[2], znodes(m, FaceZ())[7])
    @test w.position == (175.0, 75.0, 300.0)          # cell-centre x, y; the face below cell 7
    @test θ.count == 2 && θ.nan == 2
    @test θ.index == (9, 5, 3)                           # first in memory order (k, then j, then i)
    @test θ.position == (425.0, 225.0, 125.0)
    @test θ.size == (10, 8, 8)
    @test occursin("θ: 2 of 640 non-finite (2 NaN, 0 Inf)", sprint(show, θ))
    # a halo value does not count
    sim2 = bu_sim(); sim2.u.data[1, 1, 1] = NaN
    @test isempty(nonfinite_fields(sim2))
end

@testset "a blowup produces a diagnosis plus a crash checkpoint of the last good state" begin
    sim = bu_sim()
    closure = StaticSmagorinsky(0.17)
    st = StepperState(:rk3)
    dir = mktempdir()
    ds = collect_diagnostics(sim, (; sgs = closure))
    w = DiagnosticsWriter(joinpath(dir, "diagnostics.csv"), ds; cadence = 0.0)
    guard = BlowupGuard(sim; every_steps = 5, dir = joinpath(dir, "crash"), sgs = closure,
                        stepper = st, diagnostics = w)
    cfl = CFLConstraints(dt_max = 2.0)
    good = nothing
    lg = Test.TestLogger(; min_level = Logging.Warn)
    err = with_logger(lg) do
        try
            for n in 1:20
                _, dt, nr = step_adaptive!(sim, cfl; dt = 1.0, sgs = closure)
                record_step!(st, dt; retries = nr)
                # one bad value, right before the check of step 11 (step_adaptive!
                # would reject a step that starts from it)
                n == 11 && (sim.θ.data[sim.mesh.Hx+3, sim.mesh.Hy+4, sim.mesh.Hz+5] = NaN)
                record_diagnostics!(w, sim.t; step = n, dt)
                check_blowup!(guard, sim; step = n)
                # the state the guard keeps after its check at step 6
                n == 6 && (good = (t = sim.t, u = copy(sim.u.data), θ = copy(sim.θ.data),
                                   νt = copy(sim.νt.data), steps = st.step))
            end
            nothing
        catch e
            e
        end
    end
    # stop: a BlowupError at the check of step 11 (checks at 1, 6, 11)
    @test err isa BlowupError
    @test guard.checks == 3
    @test err.step == 11 && err.t == 11.0
    @test [f.name for f in err.fields] == [:θ]
    @test err.fields[1].index == (3, 4, 5) && err.fields[1].count == 1
    @test err.cause === nothing
    # dump: the crash checkpoint holds the state of the last passing check (step 6)
    @test err.checkpoint == joinpath(dir, "crash", "crash_000000011.jls")
    @test isfile(err.checkpoint)
    @test err.checkpoint_step == 6 && err.checkpoint_t == 6.0
    d = read_checkpoint(err.checkpoint)
    @test d[:t] == good.t
    @test d[:fields][:u] == good.u && d[:fields][:θ] == good.θ && d[:fields][:νt] == good.νt
    @test all(isfinite, d[:fields][:θ])
    @test d[:extra][:crash] == true && d[:extra][:blowup_step] == 11 && d[:extra][:step] == 6
    @test d[:stepper].step == good.steps
    # ... and a run restarts from it
    sim2 = bu_sim(; seed = 99)
    st2 = StepperState(:rk3)
    restart!(sim2, err.checkpoint; sgs = StaticSmagorinsky(0.17), stepper = st2)
    @test sim2.t == good.t && sim2.u.data == good.u && sim2.θ.data == good.θ
    @test st2.step == good.steps
    # diagnose: one Error record with the field, where, when and the last diagnostics
    recs = [r for r in lg.logs if get(blowup_kwargs(r), :subsystem, nothing) === :blowup]
    @test length(recs) == 1
    r = recs[1]; k = blowup_kwargs(r)
    @test r.level == Logging.Error
    @test k[:event] === :blowup && k[:step] == 11 && k[:t] == 11.0
    @test k[:fields] == (:θ,) && k[:checkpoint] == err.checkpoint && k[:checkpoint_step] == 6
    @test occursin("non-finite values in θ", r.message)
    @test occursin("θ: 1 of 640 non-finite (1 NaN, 0 Inf), first at (3, 4, 5)", k[:details])
    # the last diagnostics: the row of step 11 (cadence 0), where θ already is NaN
    @test occursin("last theta_max = NaN K  (step 11", k[:details])
    @test occursin("last ke_mean = ", k[:details]) && occursin("last sgs.nu_h_max = ", k[:details])
    @test length(err.diagnostics) == length(ds)
    @test isnan(only(x.value for x in err.diagnostics if x.name == "theta_max"))
    # the message of the error says it all again
    msg = sprint(showerror, err)
    @test occursin("blew up at step 11", msg) && occursin("crash checkpoint (step 6", msg)
    close(w)
end

end
