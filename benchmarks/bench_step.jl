# End-to-end BoussinesqSimulation step! — full physics on / grid scaling.
#
# Run: julia --project=benchmarks benchmarks/bench_step.jl

using Aeolus
using Aeolus.Fields: set!
using BenchmarkTools
using Printf

function build_sim(N; sgs=true, buoy=true)
    mesh = RegularGrid(; size=(N,N,N), extent=(2π,2π,2π), halo=(2,2,2))
    sim = BoussinesqSimulation(mesh;
        ν       = 1e-3,
        θ₀      = buoy ? 300.0 : 0.0,
        g_accel = 9.81,
        C_s     = sgs ? 0.17 : 0.0,
        Pr_t    = 1/3,
    )
    set!(sim.u, (x,y,z) ->  sin(x)*cos(y)*cos(z))
    set!(sim.v, (x,y,z) -> -cos(x)*sin(y)*cos(z))
    set!(sim.θ, (x,y,z) -> 300.0 + 0.1*sin(y))
    return sim
end

function bench_one(N, dt; sgs, buoy)
    sim = build_sim(N; sgs=sgs, buoy=buoy)
    step!(sim, dt)   # warm-up
    b = @benchmark step!($sim, $dt) samples = 20 evals = 1
    return b
end

println("BoussinesqSimulation step! — scaling with grid size")
println("─"^92)
@printf("  %5s   %10s   %-30s   %12s   %10s\n",
        "N", "cells", "physics", "median (ms)", "ns/cell")
println("─"^92)

for (N, dt) in ((16, 0.01), (32, 0.005), (64, 0.002), (96, 0.001))
    for (label, opts) in (("advect + project only",       (sgs=false, buoy=false)),
                          ("+ SGS",                        (sgs=true,  buoy=false)),
                          ("+ SGS + buoyancy",             (sgs=true,  buoy=true)))
        b = bench_one(N, dt; opts...)
        med = median(b.times) / 1e6
        ns  = median(b.times) / N^3
        @printf("  %5d   %10d   %-30s   %12.2f   %10.1f\n", N, N^3, label, med, ns)
    end
end
println("─"^92)
