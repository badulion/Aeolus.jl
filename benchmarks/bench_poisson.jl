# FFT pressure solver — scaling with grid size.
#
# Run: julia --project=benchmarks benchmarks/bench_poisson.jl

using Aeolus
using Aeolus.PressureSolvers: FFTPoissonSolver, solve!
using Aeolus.Fields: set!, interior_view
using BenchmarkTools
using Printf

function bench_size(N)
    mesh   = RegularGrid(; size=(N,N,N), extent=(2π,2π,2π), halo=(1,1,1))
    src    = Field(CellCenter(), mesh)
    p      = Field(CellCenter(), mesh)
    set!(src, (x,y,z) -> sin(x)*cos(y)*sin(z))
    # ensure zero mean (compatibility)
    iv = interior_view(src)
    iv .-= sum(iv) / length(iv)
    solver = FFTPoissonSolver(mesh)
    b      = @benchmark solve!($p, $src, $solver)
    return b
end

println("FFTPoissonSolver — solve! scaling (triply periodic)")
println("─"^70)
@printf("  %5s   %10s   %10s   %10s\n", "N", "cells", "median (ms)", "ns/cell")
println("─"^70)

for N in (16, 32, 64, 96, 128)
    b = bench_size(N)
    med = median(b.times) / 1e6
    ns_per = median(b.times) / N^3
    @printf("  %5d   %10d   %10.3f   %10.1f\n", N, N^3, med, ns_per)
end
println("─"^70)
