# The post-predictor operators inside the fused sweep (#533 plan step 6):
# one Euler substage's tendencies + predictor + the urban chain of
# post-predictor operators (implicit IBM correction, facade wall stress,
# ground wall stress on the roofs and the street, an elevated canopy drag),
# three ways:
#
#   loops    one loop per term: the fused tendency launch, the predictor
#            passes, then each operator's whole-field method (the reference);
#   after    the fused sweep without post terms, then each operator's
#            whole-field method after it (every operator an AfterSweep);
#   rows     the fused sweep with the operators on its delayed rows (and
#            their deferred points after it).
#
#   julia --project=. -t 12 benchmarks/bench_post_sweep.jl [N] [Nz] [calls]
#
# Prints ms per substage (median of `calls`) for each way, interleaved
# call by call so that the power cap's boost and sustained phases hit all
# three alike. The domain: N × N × Nz cells of 2 m × 2 m × 2 m, 16
# buildings of 10 % of N square and up to Nz/3 tall, a canopy over a
# quarter of the columns between Nz/3 and Nz/2.

using Aeolus
using Aeolus.Fields: interior_view
using Aeolus.Simulations: BoussinesqSimulation
using Aeolus.IBM: SolidMask, mark_solid!, implicit_ibm_correction!, ibm_correction_term
using Aeolus.PlantCanopy: ResolvedCanopyDrag, apply_canopy_drag!, canopy_drag_term
using Aeolus.WallModels: RoughLogLaw, WallModel, immersed_walls, apply_immersed_wall_stress!, immersed_wall_term,
                         apply_wall_stress!, wall_stress_term
using Random: Xoshiro, randn
using Statistics: median

const FV = Aeolus.Operators.FiniteVolume
const SIM = Aeolus.Simulations

N = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 128
Nz = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 64
ncalls = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 20
Δ = 2.0
g = RegularGrid(; size = (N, N, Nz), extent = (N * Δ, N * Δ, Nz * Δ), halo = (3, 3, 3))
s = BoussinesqSimulation(g; ν = 1e-2)
rng = Xoshiro(533)
for f in (s.u, s.v, s.w)
    interior_view(f) .= 2.0 .* randn(rng, size(interior_view(f)))
end
interior_view(s.θ) .= 300.0
SIM.fill_state_halos!(s)
save = deepcopy((s.u.data, s.v.data, s.w.data, s.θ.data))
restore!() = foreach((x, y) -> copyto!(x, y), (s.u.data, s.v.data, s.w.data, s.θ.data), save)

L = N * Δ; b = 0.1L; H = Nz * Δ
sm = SolidMask(g)
mark_solid!(sm, (x, y, z) -> begin
    ib, jb = floor(Int, x / (L / 4)), floor(Int, y / (L / 4))
    xc, yc = (ib + 0.5) * L / 4, (jb + 0.5) * L / 4
    abs(x - xc) < b / 2 && abs(y - yc) < b / 2 && z < H / 3 * (0.5 + 0.5 * ((ib + jb) % 2))
end)
lev = [something(findfirst(k -> !sm.mask[g.Hx+i, g.Hy+j, g.Hz+k], 1:Nz), Nz + 1) for i in 1:N, j in 1:N]
lad = zeros(N, N, Nz)
for k in Nz÷3+1:Nz÷2, j in 1:N, i in 1:N
    (i ÷ 8 + j ÷ 8) % 4 == 0 && (lad[i, j, k] = 0.5)
end
iw = immersed_walls(s, sm, RoughLogLaw(0.05))
wm = WallModel(RoughLogLaw(0.1))
dr = ResolvedCanopyDrag(g, lad; C_d = 0.2)
Δt = 0.1

whole!() = (implicit_ibm_correction!(s, sm; dt = Δt); apply_immersed_wall_stress!(s, iw, Δt);
            apply_wall_stress!(s, wm, Δt; mask = lev); apply_canopy_drag!(s, dr, Δt); nothing)

function loops!()
    FV._fused_tendencies!(s.tu, s.tv, s.tw, s.u, s.v, s.w, s.ν, (s.tθ, s.θ), nothing, g)
    for (f, t) in ((s.u, s.tu), (s.v, s.tv), (s.w, s.tw))
        SIM.axpy!(f, Δt, t)
    end
    whole!()
    SIM.axpy!(s.θ, Δt, s.tθ)
    return nothing
end
terms = (FV.BaseTendencies(s.ν),)
ns = FV.sweep_slabs(Aeolus.Backends.backend(s.u.data), N * N * Nz, Nz)
bufs = FV.sweep_buffers(nothing, s.u.data, 4, ns, Nz)
s1 = (s.θ, SIM._Axpy(Δt), nothing)
h = SIM._VelocityHalos(s)
after!() = (FV.blocked_predictor!(bufs, terms, (FV.AfterSweep(whole!),), h, s.u, s.v, s.w, SIM._Axpy(Δt), s1, nothing, g);
            nothing)
function rows!()
    post = (ibm_correction_term(sm; dt = Δt), immersed_wall_term(iw, Δt), wall_stress_term(s, wm, Δt; mask = lev),
            canopy_drag_term(dr, s, Δt))
    foreach(t -> FV.prepare!(t, s, Δt), post)
    FV.blocked_predictor!(bufs, terms, post, h, s.u, s.v, s.w, SIM._Axpy(Δt), s1, nothing, g)
    foreach(t -> FV.finish!(t, s, Δt), post)
    return nothing
end

ways = (("loops", loops!), ("after", after!), ("rows", rows!))
for (_, f) in ways
    restore!(); f(); restore!(); f()
end
times = Dict(n => Float64[] for (n, _) in ways)
for _ in 1:ncalls, (n, f) in ways
    restore!()
    push!(times[n], @elapsed f())
end
println("post sweep N=$N Nz=$Nz threads=$(Threads.nthreads()) slabs=$ns pairs=$(length(iw.faces)) " *
        "canopy faces=$(length(dr.act)) solid=$(count(sm.mask))")
for (n, _) in ways
    println(rpad(n, 6), " ", round(1e3 * median(times[n]); digits = 2), " ms/substage (median of $ncalls)")
end
