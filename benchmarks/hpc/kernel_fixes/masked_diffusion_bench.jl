# Starting material from the laptop measurements (#554-#560), see benchmarks/hpc/HANDOFF.md.
# Paths below point at the laptop's scratch and worktrees: adapt them before use.
# one process of the alternating before/after: sgs_diffuse! with this checkout's default setting
using Aeolus, Statistics
which = ARGS[1]
S = "/tmp/claude-1000/-home-andi-coding-work-Aeolus-jl/3e8f9011-18a5-4b37-8ab9-483314fc7365/scratchpad"
include(which == "base" ? "$S/i559/cases_base.jl" :
        "/home/andi/coding/work/Aeolus.jl/.worktrees/issue-559/test/support/tracer_diffusion_cases.jl")
core = Symbol(get(ENV, "CORE", "boussinesq")); aniso = get(ENV, "ANISO", "0") == "1"
g = RegularGrid(; size = (200, 200, 60), extent = (400.0, 400.0, 120.0), halo = (3, 3, 3))
case, d, q, κv = tdc_case(g; core = core, aniso = aniso)
q0 = copy(q.data); κ = case.sim.κt; kv = κv === nothing ? κ : κv
f() = (copyto!(q.data, q0); TDU.sgs_diffuse!(d, case, q, 0.8; water = true, κ = κ, κv = kv))
f(); f()
r = Float64[]
for _ in 1:60; t0 = time_ns(); f(); push!(r, (time_ns() - t0) / 1e6); end
open("$S/i559/q_$(which)_$(core)_t$(Threads.nthreads()).bin", "w") do io; write(io, q.data); end
println("ab ", which, " ", core, " aniso ", aniso, " threads ", Threads.nthreads(), " min ", round(minimum(r), digits = 2),
        " median ", round(median(r), digits = 2), " (incl. copy of q)")
