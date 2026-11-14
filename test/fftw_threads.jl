# #575: FFTW plans are single-threaded and the spectral pressure solvers run
# them on Julia's threads, by blocks of levels (x-y transforms) and rows
# (z transforms), as PALM does. FFTW starts no threads of its own: its
# former pool was created from thread 1 and inherited its CPU affinity, so
# with JULIA_EXCLUSIVE=1 every FFTW thread ran on core 0 (Berlin 20 m, 72
# threads: 28.4 s per step pinned against ~6 s unpinned, #566).
#
# A subprocess at -t 8, with the threshold of the default backend at 0 so
# that even these small grids are cut into blocks, checks for every spectral
# solver and axis-BC combination, at an even and an odd size (odd Nx·Ny: the
# levels alternate between two memory alignments mod 16):
#   * the blocked solve agrees with the same solver built for a serial
#     backend (one block: one plan over the whole array);
#   * no OS thread is created (Linux /proc/self/task) by planning and
#     solving under MEASURE, the planner whose threaded form crashed in
#     FFTW.jl's task callback (#397) and whose own pool caused #575.

using Test

const _FT_SCRIPT = raw"""
using Aeolus, Random
using Aeolus.Grids: RegularGrid
using Aeolus.Fields: Field, interior_view
using Aeolus.StretchedGrids: StretchedGrid, StretchedPoissonSolver
const PS = Aeolus.PressureSolvers
const B = Aeolus.Backends
Aeolus.Operators.set_fftw_policy!(; planner = :measure)
B.init_threads!(B.CPUBackend())      # as every run does (before #575: FFTW on 8 threads)
ntask() = isdir("/proc/self/task") ? length(readdir("/proc/self/task")) : -1

function solvers(N)
    m  = RegularGrid(; size = N, extent = (1.0, 1.3, 0.7), halo = (1, 1, 1))
    sm = StretchedGrid(; size = N, extent = (800.0, 800.0, 3000.0), halo = (1, 1, 1),
                       stretching = (:geometric, 1.04))
    dm = RegularGrid(; size = N, extent = (1000.0, 1000.0, 3000.0), halo = (1, 1, 1))
    dref = Aeolus.Anelastic.isothermal_reference(dm; T_surf = 300.0)
    s = Pair{String,Any}["fft" => (m, () -> PS.FFTPoissonSolver(m))]
    for bx in (:periodic, :neumann), by in (:periodic, :neumann), bz in (:periodic, :neumann)
        push!(s, "cosine($bx,$by,$bz)" => (m, () -> PS.CosineFFTPoissonSolver(m; bc_x = bx, bc_y = by, bc_z = bz)))
        push!(s, "transforms($bx,$by,$bz)" => (m, () -> PS._CosineTransformSolver(m; bc_x = bx, bc_y = by, bc_z = bz)))
    end
    for bx in (:periodic, :neumann), by in (:periodic, :neumann)
        push!(s, "stretched($bx,$by)" => (sm, () -> StretchedPoissonSolver(sm; bc_x = bx, bc_y = by)))
    end
    for bx in (:periodic, :neumann)
        push!(s, "direct_weighted($bx)" => (dm, () ->
              Aeolus.Anelastic.DirectWeightedPoissonSolver(dm, dref; bc_x = bx, bc_y = bx)))
    end
    return s
end

n0 = ntask()
worst = 0.0
for N in ((32, 24, 16), (27, 21, 13))
    for (name, (m, mk)) in solvers(N)
        B.set_default_backend!(B.CPUBackend(; threshold = 0))     # blocks on every thread
        sb = mk()
        B.set_default_backend!(B.CPUBackend(; serial = true))     # one block
        s1 = mk()
        B.set_default_backend!(B.CPUBackend(; threshold = 0))
        src = Field(m); interior_view(src) .= randn(Xoshiro(575), size(interior_view(src)))
        p1 = Field(m); pb = Field(m)
        PS.solve!(p1, src, s1); PS.solve!(pb, src, sb)
        a, b = interior_view(pb), interior_view(p1)
        d = maximum(abs.(a .- b)) / maximum(abs.(b))
        global worst = max(worst, d)
        d < 1e-12 || println("DIFF $name $N $d")
    end
end
println("worst ", worst)
println("tasks ", n0, " ", ntask(), " julia ", Threads.nthreads(), " fftw ", Aeolus.fftw_policy().threads)
"""

@testset "FFTW: serial plans on Julia's threads, no FFTW threads (#575)" begin
    project = dirname(Base.active_project())
    cmd = `$(Base.julia_cmd()) --startup-file=no -t 8 --project=$project -e $_FT_SCRIPT`
    out = IOBuffer(); err = IOBuffer()
    p = run(pipeline(ignorestatus(cmd); stdout = out, stderr = err))
    msg = String(take!(out)); e = String(take!(err))
    @test success(p)
    success(p) || @info "fftw_threads.jl: the subprocess failed" msg e
    # Blocked and one-block solves are the same arithmetic up to the order
    # FFTW's codelets take for a block of another size: they agree to
    # round-off (a few ulp of the transform, O(1e-15) relative measured).
    m = match(r"worst ([0-9.eE+-]+)", msg)
    @test m !== nothing && parse(Float64, m[1]) < 1e-12
    @test !occursin("DIFF", msg)
    # No OS thread is created by planning and solving: FFTW has no pool.
    t = match(r"tasks (-?\d+) (-?\d+) julia (\d+) fftw (\d+)", msg)
    @test t !== nothing
    if t !== nothing
        @test parse(Int, t[4]) == 1
        if parse(Int, t[1]) >= 0          # Linux: /proc/self/task
            @test parse(Int, t[2]) == parse(Int, t[1])
        end
    end
end
