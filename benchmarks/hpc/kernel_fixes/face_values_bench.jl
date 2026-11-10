# Starting material from the laptop measurements (#554-#560), see benchmarks/hpc/HANDOFF.md.
# Paths below point at the laptop's scratch and worktrees: adapt them before use.
# #557: branch-free face bodies (new) against the branching ones (old,
# `Branching` of test/support/branching_faces.jl), interleaved in one
# process: the flux-limited scalar advection (advect_positive!) and the
# TKE update (blocked pass and separate loops), synthetic 200x200x60, and
# the Berlin TKE closure (centred). Min / median ms.
using Aeolus, Statistics
using Aeolus.Operators: advance_sgs!
const SIM = Aeolus.Simulations
const TKB = Aeolus.PrognosticSGS
W = "/home/andi/coding/work/Aeolus.jl/.worktrees/issue-557"
include("$W/test/support/branching_faces.jl")
include("$W/test/support/tke_blocked_cases.jl")
S = "/tmp/claude-1000/-home-andi-coding-work-Aeolus-jl/3e8f9011-18a5-4b37-8ab9-483314fc7365/scratchpad"
println("threads ", Threads.nthreads())

function with_recon(c, r)
    vals = [f === :recon ? r : (f === :blocked ? Ref{Any}(nothing) : getfield(c, f)) for f in fieldnames(typeof(c))]
    return TKEClosure(vals...)
end

function measure(fs; rounds = 15, inner = 3)
    for (k, f) in fs; f(); f(); end
    res = Dict(k => Float64[] for (k, _) in fs)
    for rep in 1:rounds, (k, f) in (isodd(rep) ? fs : reverse(fs))
        for _ in 1:inner; t0 = time_ns(); f(); push!(res[k], (time_ns() - t0) / 1e6); end
    end
    for (k, _) in fs
        println("  ", rpad(k, 26), " min ", lpad(round(minimum(res[k]), digits = 2), 8), " ms  median ",
                lpad(round(median(res[k]), digits = 2), 8), " ms")
    end
end
rel(a, b) = maximum(abs.(a .- b)) / maximum(abs, b)

# --- A: advect_positive!, 200x200x60 ---------------------------------------
g = RegularGrid(; size = (200, 200, 60), extent = (20000.0, 20000.0, 3000.0), halo = (3, 3, 3))
sim0, c0 = tkb_case(g; surface = true)      # a random flow and e (the limiter acts)
u, v, w, ϕ = sim0.u, sim0.v, sim0.w, c0.e
ws = FluxLimiterWorkspace(g); tend = Field(CellCenter(), g)
for (name, r) in (("WENO-Z5", _AS.WENOZ(5, 2)), ("centred", _AS.Centered2()))
    t1 = Field(CellCenter(), g); t2 = Field(CellCenter(), g)
    advect_positive!(t1, u, v, w, ϕ, 2.0; scheme = r, workspace = ws)
    advect_positive!(t2, u, v, w, ϕ, 2.0; scheme = Branching(r), workspace = ws)
    println("advect_positive! $name 200x200x60: rel max |Δtend| new-old = ", rel(t1.data, t2.data))
    measure(("old (branching)" => () -> advect_positive!(tend, u, v, w, ϕ, 2.0; scheme = Branching(r), workspace = ws),
             "new (branch-free)" => () -> advect_positive!(tend, u, v, w, ϕ, 2.0; scheme = r, workspace = ws)))
end

# --- B: the TKE update, 200x200x60 WENO-Z5 ----------------------------------
let c = c0, sim = sim0, co = with_recon(c0, Branching(c0.recon)), e0 = copy(c0.e.data)
    run(f, cc) = (copyto!(cc.e.data, e0); f(cc, sim, 2.0))
    run(TKB._advance_blocked!, c); en = copy(c.e.data)
    run(TKB._advance_blocked!, co); eo = copy(co.e.data)
    run(TKB._advance_separate!, co); es = copy(co.e.data)
    println("TKE WENO-Z5 200x200x60: rel max|Δe| new-old ", rel(en, eo), "  new pass - old separate ", rel(en, es))
    measure(("pass old" => () -> run(TKB._advance_blocked!, co), "pass new" => () -> run(TKB._advance_blocked!, c),
             "separate old" => () -> run(TKB._advance_separate!, co), "separate new" => () -> run(TKB._advance_separate!, c),
             "copy e" => () -> copyto!(c.e.data, e0)))
    copyto!(c.e.data, e0)
end

# --- C: the Berlin TKE closure (centred) ------------------------------------
cfg = load_run_config("$S/i533int/berlin_100m_20250824.toml"; overrides = ["fftw.planner=\"estimate\""])
model = Aeolus.Config.build(Aeolus.SimulationDriver.SimulationDriverType(), cfg).model
for _ in 1:3; step!(model, 2.0); end
d = model.domains[1]; sim = d.state; c = Aeolus.Components.sgs_closure(d)
co = with_recon(c, Branching(c.recon))
println("Berlin closure recon ", c.recon, " grid ", size(c.e.data))
e0 = copy(c.e.data); st0 = copy(c.stats)
runb(cc, set) = (SIM.set_loop_fusion!(sim, set); copyto!(cc.e.data, e0); advance_sgs!(cc, sim, 2.0))
runb(c, :if_possible); en = copy(c.e.data); runb(co, :if_possible); eo = copy(co.e.data)
println("Berlin: rel max|Δe| new-old ", rel(en, eo))
measure(("pass old" => () -> runb(co, :if_possible), "pass new" => () -> runb(c, :if_possible),
         "separate old" => () -> runb(co, :never), "separate new" => () -> runb(c, :never),
         "copy e" => () -> copyto!(c.e.data, e0)))
copyto!(c.e.data, e0); copyto!(c.stats, st0)
