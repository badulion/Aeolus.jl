# Starting material from the laptop measurements (#554-#560), see benchmarks/hpc/HANDOFF.md.
# Paths below point at the laptop's scratch and worktrees: adapt them before use.
include(joinpath(@__DIR__, "setup.jl"))
println("threads ", Threads.nthreads())
function tm(f, n)
    ts = Float64[]
    for _ in 1:n
        t0 = time_ns(); f(); push!(ts, (time_ns() - t0) / 1e6)
    end
    ts
end
for stretched in (false, true)
    s = mksetup(; stretched)
    c, sim, g = s.c, s.sim, s.g
    D2 = similar(c.D.data); e0 = copy(c.e.data)
    fs = Dict(
      "strain_old" => () -> P._dual_strain_old!(c.D, sim.u, sim.v, sim.w, sim.bcs, g),
      "strain_new" => () -> P._dual_strain!(c.D, sim.u, sim.v, sim.w, sim.bcs, g),
      "update_old" => () -> (copyto!(c.e.data, e0); P._update_old!(c, 0.5, (0.0, 0.0))),
      "update_new" => () -> (copyto!(c.e.data, e0); P._update!(c, 0.5, (0.0, 0.0))),
      "copy_e" => () -> copyto!(c.e.data, e0))
    for f in values(fs); f(); f(); end
    copyto!(c.e.data, e0); fill!(c.stats, 0); P._update_old!(c, 0.5, (0.0, 0.0)); eo = copy(c.e.data); so = copy(c.stats)
    copyto!(c.e.data, e0); fill!(c.stats, 0); P._update!(c, 0.5, (0.0, 0.0)); en = copy(c.e.data); sn = copy(c.stats)
    println("update: max|Δe| = ", maximum(abs.(en .- eo)), "  max rel Δ stats = ", maximum(abs.(sn .- so) ./ max.(abs.(so), 1e-300)))
    P._dual_strain_old!(c.D, sim.u, sim.v, sim.w, sim.bcs, g); Do = copy(c.D.data)
    P._dual_strain!(c.D, sim.u, sim.v, sim.w, sim.bcs, g)
    println("strain: max|ΔD|/max|D| = ", maximum(abs.(c.D.data .- Do)) / maximum(abs.(Do)))
    res = Dict(k => Float64[] for k in keys(fs))
    for rep in 1:15, k in sort(collect(keys(fs)))
        append!(res[k], tm(fs[k], 3))
    end
    for k in sort(collect(keys(res)))
        println(stretched ? "stretched " : "regular   ", rpad(k, 12), " min ", round(minimum(res[k]), digits=3), " ms  median ", round(median(res[k]), digits=3), " ms")
    end
end
