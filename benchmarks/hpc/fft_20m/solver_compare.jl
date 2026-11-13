# #566 REPORT_fft_20m.md: the 3-D DCT solver against 2-D DCT per level + tridiagonal in z.
# usage: julia -t J,0 solver_compare.jl Nx Ny Nz fftw_threads planner [reps]
# Prints the solve time of each, their agreement on a random zero-mean source,
# and the 2-D transform pair done PALM-style: serial plans, Julia threads over levels.
using Aeolus, FFTW, LinearAlgebra, Random
const PS = Aeolus.PressureSolvers
using Aeolus.Fields: interior_view, Field
using Aeolus.Grids: RegularGrid, CellCenter
Nx, Ny, Nz, nt = parse.(Int, ARGS[1:4]); planner = Symbol(ARGS[5])
reps = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 3
Aeolus.Operators.set_fftw_policy!(planner = planner, threads = nt)

g = RegularGrid(; size = (Nx, Ny, Nz), extent = (20.0Nx, 20.0Ny, 20.0Nz), halo = (1, 1, 1))
T = Float64
tc = @elapsed cos3 = PS.CosineFFTPoissonSolver(g; bc_x = :neumann, bc_y = :neumann, bc_z = :neumann)
idz2 = 1 / g.dz^2
lower = T[k > 1 ? idz2 : 0.0 for k in 1:Nz]; upper = T[k < Nz ? idz2 : 0.0 for k in 1:Nz]
tt = @elapsed col = PS.ColumnTridiagonalSolver(T, (Nx, Ny, Nz), (g.dx, g.dy);
        lower = lower, upper = upper, hweight = ones(T, Nz), measure = fill(T(g.dz^2), Nz),
        inv_coupling = ones(T, Nz + 1), gauge = ones(T, Nz), bc_x = :neumann, bc_y = :neumann)

src = Field(CellCenter(), g); p1 = Field(CellCenter(), g); p2 = Field(CellCenter(), g)
raw = randn(Xoshiro(566), Nx, Ny, Nz); raw .-= sum(raw) / length(raw)
interior_view(src) .= raw

function timeit(f, reps)
    f(); ts = Float64[]
    for _ in 1:reps; push!(ts, @elapsed f()); end
    minimum(ts)
end
t3 = timeit(() -> PS.solve!(p1, src, cos3), reps)
tz = timeit(() -> PS.solve!(p2, src, col), reps)
a = interior_view(p1); b = interior_view(p2)
ma = sum(a) / length(a); mb = sum(b) / length(b)
rel = maximum(abs.((a .- ma) .- (b .- mb))) / maximum(abs.(a .- ma))

# PALM-style: one serial 2-D DCT plan per level, Julia threads over levels
buf = copy(raw); out = similar(buf)
FFTW.set_num_threads(1)
pf = FFTW.plan_r2r(buf[:, :, 1], FFTW.REDFT10; flags = Aeolus.Operators.fftw_planner_flags())
pi_ = FFTW.plan_r2r(buf[:, :, 1], FFTW.REDFT01; flags = Aeolus.Operators.fftw_planner_flags())
FFTW.set_num_threads(nt)
function levels!(out, p, inp)
    Threads.@threads :static for k in axes(inp, 3)
        mul!(view(out, :, :, k), p, view(inp, :, :, k))
    end
end
tl = timeit(() -> (levels!(out, pf, buf); levels!(buf, pi_, out)), reps)
# the FFTW-threaded 2-D pair of the column solver, for the same comparison
tf = timeit(() -> (mul!(col.buf2, col.plan_fwd, col.buf); mul!(col.buf, col.plan_inv, col.buf2)), reps)

np = Nx * Ny * Nz
println("CMP Nx=$Nx Ny=$Ny Nz=$Nz jt=$(Threads.nthreads()) ft=$nt planner=$planner ",
        "build3d_s=$(round(tc, digits=2)) buildcol_s=$(round(tt, digits=2)) ",
        "solve3d_s=$(round(t3, digits=4)) solvecol_s=$(round(tz, digits=4)) ",
        "pair2d_fftwthreads_s=$(round(tf, digits=4)) pair2d_juliathreads_s=$(round(tl, digits=4)) ",
        "ns_pt_3d=$(round(1e9t3/np, digits=2)) ns_pt_col=$(round(1e9tz/np, digits=2)) rel_diff=$(round(rel, sigdigits=3))")
