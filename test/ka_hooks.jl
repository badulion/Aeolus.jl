# KA-k (#307): the dynamics hooks of the Berlin run and the WallModels
# operators on launch!.
#
#   1. the immersed-wall operators (apply_immersed_wall_stress!,
#      remove_stencil_wall_flux!, immersed_wall_drag!) as a per-face gather
#      over the face index of ImmersedWalls, bit for bit against the former
#      scatter into full work arrays (test/support/legacy_hooks.jl): on a
#      solid mask with corners and one-cell gaps (faces with two walls), with
#      and without eddy viscosity, also for a shuffled pair order; serially,
#      threaded with every launch on the KA path (`threshold = 0`), and on the
#      default backend;
#   2. the face index groups every pair of a face, once, in pair order.

using Test
using Random: Xoshiro, randn, rand, shuffle
using Aeolus
using Aeolus.Fields: interior_view
using Aeolus.IBM: SolidMask, mark_solid!, implicit_ibm_correction!
using Aeolus.WallModels: ImmersedWalls, ImmersedWallFace

isdefined(@__MODULE__, :LegacyHooks) || include(joinpath(@__DIR__, "support", "legacy_hooks.jl"))

const HK_BACKENDS = ("serial" => CPUBackend(serial = true),
                     "threaded, threshold 0" => CPUBackend(threshold = 0),
                     "default" => CPUBackend())
hk_bits(A::AbstractArray) = reinterpret(UInt64, vec(collect(Float64, A)))
hk_bits(x::Real) = reinterpret(UInt64, Float64(x))
hk_bits(x::Tuple) = map(hk_bits, x)

# a 3-D city: blocks with a one-cell street between two of them (faces with
# a wall on both sides), an L (inner corners) and a tower; periodic in x, y
function hk_city(; seed = 307, sgs = true)
    g = RegularGrid(; size = (24, 20, 10), extent = (24.0, 20.0, 10.0), halo = (3, 3, 3))
    sim = BoussinesqSimulation(g; ν = 0.05, C_s = sgs ? 0.17 : 0.0,
                               bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
    sm = SolidMask(g)
    mark_solid!(sm, (x, y, z) -> (3 < x < 8 && 2 < y < 9 && z < 6) ||       # block A
                                 (9 < x < 13 && 2 < y < 9 && z < 4) ||      # block B, 1-cell street to A? (x 8..9)
                                 (14 < x < 20 && 11 < y < 13 && z < 7) ||   # L: one arm
                                 (14 < x < 16 && 11 < y < 18 && z < 7) ||   # L: other arm
                                 (20 < x < 22 && 15 < y < 17 && z < 9))     # tower
    rng = Xoshiro(seed)
    for f in (sim.u, sim.v, sim.w, sim.tu, sim.tv, sim.tw)
        interior_view(f) .= randn(rng, size(interior_view(f)))
    end
    implicit_ibm_correction!(sim, sm)
    sim.νt.data .= 0.1 .* abs.(randn(rng, size(sim.νt.data)))
    return sim, sm
end

hk_state(sim) = map(copy, (sim.u.data, sim.v.data, sim.w.data, sim.tu.data, sim.tv.data, sim.tw.data))
function hk_restore!(sim, st)
    for (f, s) in zip((sim.u, sim.v, sim.w, sim.tu, sim.tv, sim.tw), st)
        f.data .= s
    end
end

@testset "KA-k: dynamics hooks and WallModels on launch! (#307)" begin

@testset "immersed walls: the face index" begin
    sim, sm = hk_city()
    iw = immersed_walls(sim, sm, RoughLogLaw(0.05))
    nf = length(iw.face_ptr) - 1
    @test length(iw.face_entry) == length(iw.faces) && sort(iw.face_entry) == 1:length(iw.faces)
    keyof(f) = (f.c, f.I, f.J, f.K)
    seen = Set{Any}()
    multi = 0
    for p in 1:nf
        es = iw.face_entry[iw.face_ptr[p]:iw.face_ptr[p+1]-1]
        @test !isempty(es) && issorted(es)                     # pairs of a face in pair order
        @test allequal(keyof(iw.faces[n]) for n in es)          # one face per row
        k = keyof(iw.faces[es[1]])
        @test !(k in seen); push!(seen, k)                      # every face once
        multi += length(es) > 1
    end
    @test multi > 0                                             # corners / one-cell gaps exist
    @test length(iw.dq) == nf && length(iw.a) == length(iw.faces)
    @info "KA-k city: $(length(iw.faces)) pairs on $nf faces, $multi faces with two walls"
end

@testset "immersed walls: gather = former scatter, bit for bit ($(sgs ? "with" : "no") ν_t, $(shuf ? "shuffled" : "built") order)" for sgs in (true, false), shuf in (false, true)
    sim, sm = hk_city(; sgs = sgs)
    law = (i, j, k) -> isodd(i + j) ? RoughLogLaw(0.05) : RoughLogLaw(0.2)
    iw0 = immersed_walls(sim, sm, law)
    faces = shuf ? shuffle(Xoshiro(1), iw0.faces) : iw0.faces
    iw = ImmersedWalls(faces, iw0.ν, iw0.Δ, iw0.dims)
    st = hk_state(sim)
    # the reference: the former loops
    A = (zeros(size(sim.u.data)), zeros(size(sim.v.data)), zeros(size(sim.w.data)))
    rs = Dict{Symbol,Any}()
    LegacyHooks.remove_stencil_wall_flux!(sim, faces, iw.Δ, rs; ν = sim.ν, νt = sim.νt, sgs = sgs)
    LegacyHooks.apply_immersed_wall_stress!(sim, faces, iw.Δ, iw.ν, A, rs, 0.7)
    aref = LegacyHooks.immersed_wall_drag!(zeros(length(faces)), faces, iw.Δ, iw.ν,
                                           sim.u.data, sim.v.data, sim.w.data)
    ref = hk_state(sim)
    @test all(iszero, A[1]) && all(iszero, A[2]) && all(iszero, A[3])
    for (bname, b) in HK_BACKENDS
        hk_restore!(sim, st)
        got = with_backend(b) do
            remove_stencil_wall_flux!(sim, iw; ν = sim.ν, sgs = sgs)
            apply_immersed_wall_stress!(sim, iw, 0.7)
            immersed_wall_drag!(zeros(length(faces)), iw, sim.u.data, sim.v.data, sim.w.data)
        end
        ok = all(hk_bits(x) == hk_bits(y) for (x, y) in zip(hk_state(sim), ref)) &&
             hk_bits(got) == hk_bits(aref) &&
             all(hk_bits(iw.stats[k]) == hk_bits(rs[k]) for k in (:F, :τA, :area, :τ_min, :F_stencil))
        ok || @info "KA-k bit mismatch (immersed walls)" bname sgs shuf
        @test ok
    end
    # the physics is untouched: the walls take momentum out, τ_w ≥ 0
    @test iw.stats[:τ_min] >= 0 && iw.stats[:τA] > 0
end

end # testset
