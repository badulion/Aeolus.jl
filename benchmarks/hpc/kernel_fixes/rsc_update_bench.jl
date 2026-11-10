# Starting material from the laptop measurements (#554-#560), see benchmarks/hpc/HANDOFF.md.
# Paths below point at the laptop's scratch and worktrees: adapt them before use.
# never (separate loops) vs if_possible (blocked pass), interleaved in one process
include(joinpath(@__DIR__, "rsc_case.jl"))
const SIM = Aeolus.Simulations
nrep = parse(Int, get(ENV, "NREP", "15"))
for (name, h, adv, mask) in (("WENO-Z5", 3, :weno_z5, false), ("WENO-Z5 mask", 3, :weno_z5, true), ("centred", 2, :centered, false))
    g = RegularGrid(; size = (200, 200, 60), extent = (2000.0, 2000.0, 600.0), halo = (h, h, h))
    sim, c = rsc_case(g; mask = mask, advection = adv, smooth = true)
    τ0 = map(f -> copy(f.data), c.τ); st0 = copy(c.stats)
    reset() = foreach((f, a) -> copyto!(f.data, a), c.τ, τ0)
    run(set) = (SIM.set_loop_fusion!(sim, set); reset(); advance_sgs!(c, sim, 0.5))
    fs = ("separate" => () -> run(:never), "blocked" => () -> run(:if_possible), "copy_τ" => reset)
    for (k, f) in fs; f(); f(); end
    run(:never); ts = map(f -> copy(f.data), c.τ); run(:if_possible); tb = map(f -> copy(f.data), c.τ)
    d = maximum(maximum(abs.(x .- y)) for (x, y) in zip(ts, tb))
    res = Dict(k => Float64[] for (k, _) in fs)
    for rep in 1:nrep, (k, f) in (isodd(rep) ? fs : reverse(fs))
        for _ in 1:3; t0 = time_ns(); f(); push!(res[k], (time_ns() - t0) / 1e6); end
    end
    st = PS.reynolds_stress_statistics(c)
    println("CASE ", name, " threads ", Threads.nthreads(), " max|Δτ| ", d, " clips ", st.step_clips, " t ", round(Int, time()))
    for (k, _) in fs
        println("  ", rpad(k, 10), " min ", round(minimum(res[k]), digits = 2), " ms  median ", round(sort(res[k])[cld(end, 2)], digits = 2), " ms")
    end
end
