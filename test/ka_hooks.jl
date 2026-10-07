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
#   2. the face index groups every pair of a face, once, in pair order;
#   3. the canopy drag (PlantCanopy.apply_canopy_drag!) over its list of
#      faces with leaves, the direct forcing of the IBM
#      (implicit_ibm_correction! with τ = 0 and τ > 0, add_ibm_penalty!) as
#      launches over the faces, the moist buoyancy of the Berlin example
#      (examples/urban/couplings/moist_buoyancy.jl) and the large-scale
#      forcings (geostrophic pressure gradient, profile nudging with a
#      height weight, both subsidence forms, advective tendencies, the
#      `:mean` target of ReferenceDamping) one work item per level, on a
#      RegularGrid and a StretchedGrid: bit for bit against the former loops,
#      on every backend.

using Test
using Random: Xoshiro, randn, rand, shuffle
using Aeolus
using Aeolus.Fields: interior_view
using Aeolus.IBM: SolidMask, mark_solid!, implicit_ibm_correction!
using Aeolus.WallModels: ImmersedWalls, ImmersedWallFace
using Aeolus.PlantCanopy: ResolvedCanopyDrag, apply_canopy_drag!, UniformLAD, canopy_lad_field
using Aeolus.IBM: add_ibm_penalty!
using Aeolus.StretchedGrids: StretchedGrid
const LSF = Aeolus.LargeScaleForcing

isdefined(Main, :BerlinCase) ||
    include(joinpath(@__DIR__, "..", "examples", "urban", "BerlinCase.jl"))

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

@testset "canopy drag over the faces with leaves = former full-grid loops" begin
    sim, sm = hk_city(; seed = 11)
    g = sim.mesh
    prof = [(3 < i < 12 && 12 < j < 18) || (i == 20 && j == 3) ? UniformLAD(2.5 + 0.1i, 4.0) : nothing
            for i in 1:g.Nx, j in 1:g.Ny]
    drag = ResolvedCanopyDrag(g, canopy_lad_field(g, prof); C_d = 0.25)
    @test length(drag.act) == count(>(0), drag.a_u) + count(>(0), drag.a_v) + count(>(0), drag.a_w)
    @test issorted(drag.act; by = x -> (x[1], x[4], x[3], x[2]))       # u, v, w; each k, j, i
    st = hk_state(sim)
    fs = ntuple(_ -> ones(g.Nx, g.Ny, g.Nz), 3)
    Fref = LegacyHooks.apply_canopy_drag!(sim, drag, fs..., 2.0)
    ref = hk_state(sim)
    for (bname, b) in HK_BACKENDS
        hk_restore!(sim, st)
        F = with_backend(() -> apply_canopy_drag!(sim, drag, 2.0), b)
        ok = all(hk_bits(x) == hk_bits(y) for (x, y) in zip(hk_state(sim), ref)) &&
             hk_bits(Tuple(F)) == hk_bits(Tuple(Fref))
        ok || @info "KA-k bit mismatch (canopy drag)" bname
        @test ok
    end
    @test Fref.Fx != 0 && Fref.Fy != 0 && Fref.Fz != 0      # the canopy is on all three components
end

@testset "IBM direct forcing over the faces = former loop over the solid cells" begin
    sim, sm = hk_city(; seed = 12)
    rng = Xoshiro(5)
    for f in (sim.u, sim.v, sim.w)          # undo the correction of hk_city: velocity inside the solid
        interior_view(f) .= randn(rng, size(interior_view(f)))
    end
    st = hk_state(sim)
    cases = (("implicit τ = 0", s -> implicit_ibm_correction!(s, sm), s -> LegacyHooks.implicit_ibm_correction!(s, sm)),
             ("implicit τ > 0", s -> implicit_ibm_correction!(s, sm; τ = 3.0, dt = 0.7),
                                s -> LegacyHooks.implicit_ibm_correction!(s, sm; τ = 3.0, dt = 0.7)),
             ("penalty", s -> add_ibm_penalty!(s, sm; τ = 0.4), s -> LegacyHooks.add_ibm_penalty!(s, sm; τ = 0.4)))
    for (cname, new!, old!) in cases
        hk_restore!(sim, st); old!(sim); ref = hk_state(sim)
        @test any(hk_state(sim)[i] != st[i] for i in 1:6)
        for (bname, b) in HK_BACKENDS
            hk_restore!(sim, st)
            with_backend(() -> new!(sim), b)
            ok = all(hk_bits(x) == hk_bits(y) for (x, y) in zip(hk_state(sim), ref))
            ok || @info "KA-k bit mismatch (IBM)" cname bname
            @test ok
        end
    end
end

@testset "moist buoyancy of the Berlin example = former loops" begin
    BCm = Main.BerlinCase
    g = RegularGrid(; size = (24, 20, 10), extent = (24.0, 20.0, 10.0), halo = (3, 3, 3))
    rng = Xoshiro(13)
    θ = 300 .+ randn(rng, 24, 20, 10); qv = 8e-3 .+ 1e-3 .* randn(rng, 24, 20, 10)
    qc = [abs.(1e-4 .* randn(rng, 24, 20, 10)), abs.(1e-5 .* randn(rng, 24, 20, 10))]
    fluid = rand(rng, 24, 20, 10) .< 0.8
    fluid[:, :, 10] .= false                                    # a level with no fluid
    tw = Aeolus.Fields.Field(Aeolus.Grids.FaceZ(), g); tw.data .= randn(rng, size(tw.data))
    tw0 = copy(tw.data)
    # the reference: the same plane_reduce! level sums, then the former loops
    Δr = zeros(24, 20, 10); S, n = zeros(10), zeros(10)
    plane_reduce!((S, n), CPUBackend(serial = true), BCm._moist_Δ_at, (1:24, 1:20, 1:10), Δr, θ, qv, qc, fluid)
    LegacyHooks.moist_deviation!(Δr, S, n, fluid)
    LegacyHooks.add_moist_buoyancy!(tw, Δr, 300.0, 9.81); twr = copy(tw.data)
    for (bname, b) in HK_BACKENDS
        Δ = zeros(24, 20, 10); tw.data .= tw0
        with_backend(b) do
            BCm.moist_buoyancy_perturbation!(Δ, θ, qv, qc, fluid)
            BCm.add_moist_buoyancy!(tw, Δ, 300.0, 9.81)
        end
        ok = hk_bits(Δ) == hk_bits(Δr) && hk_bits(tw.data) == hk_bits(twr)
        ok || @info "KA-k bit mismatch (moist buoyancy)" bname
        @test ok
    end
end

function hk_lsf_sim(gname; seed = 14)
    sz, ex = (20, 16, 24), (2000.0, 1600.0, 2400.0)
    g = gname == "regular" ? RegularGrid(; size = sz, extent = ex, halo = (3, 3, 3)) :
                             StretchedGrid(; size = sz, extent = ex, halo = (3, 3, 3), stretching = (:geometric, 1.06))
    sim = BoussinesqSimulation(g; ν = 0.0, f_coriolis = 1e-4,
                               bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
    rng = Xoshiro(seed)
    for f in (sim.u, sim.v, sim.θ, sim.tu, sim.tv, sim.tθ)
        f.data .= randn(rng, size(f.data))
    end
    sim.θ.data .+= 300
    return sim
end
hk_lsf_state(sim) = map(copy, (sim.tu.data, sim.tv.data, sim.tθ.data))
function hk_lsf_restore!(sim, st)
    for (f, x) in zip((sim.tu, sim.tv, sim.tθ), st)
        f.data .= x
    end
end

@testset "large-scale forcing one level per work item = former loops ($gname)" for gname in ("regular", "stretched")
    sim = hk_lsf_sim(gname)
    m = sim.mesh
    st = hk_lsf_state(sim)
    Ug = (z, t) -> 5 + 0.002z + 1e-4t
    tab = LSF.ForcingTable([0.0, 800.0, 2400.0], [299.0, 301.0, 306.0])
    ws = (z, t) -> -3e-3 * z / 1000
    W = LSF.NudgingWeight(600.0; depth = 400.0)
    t, Δt = 120.0, 2.5
    cases = (
      ("geostrophic", s -> (apply_forcing!(s, LSF.GeostrophicPressureGradient(Ug, -1.5), t, Δt); ()),
                      s -> (LegacyHooks.apply_gpg!(s, LSF.GeostrophicPressureGradient(Ug, -1.5), t, Δt); ())),
      ("nudging (weighted)", s -> Tuple(apply_forcing!(s, LSF.ProfileNudging((θ = tab, u = 4.0), 900.0; weight = W), t, Δt)),
                      s -> (wf = LSF._weight_fn(W, s);
                            (LegacyHooks._nudge_levels!(s.θ.data, s.tθ.data, m, tab, 1 / 900.0, wf, t, t + Δt),
                             LegacyHooks._nudge_levels!(s.u.data, s.tu.data, m, 4.0, 1 / 900.0, wf, t, t + Δt)))),
      ("nudging (unweighted)", s -> Tuple(apply_forcing!(s, LSF.GeostrophicNudging(3.0, 1.0, 600.0), t, Δt)),
                      s -> (LegacyHooks._nudge_levels!(s.u.data, s.tu.data, m, 3.0, 1 / 600.0, nothing, t, t + Δt),
                            LegacyHooks._nudge_levels!(s.v.data, s.tv.data, m, 1.0, 1 / 600.0, nothing, t, t + Δt))),
      ("subsidence, mean gradient", s -> Tuple(apply_forcing!(s, LSF.Subsidence(ws; fields = (:θ, :u)), t, Δt)),
                      s -> (LegacyHooks._subside_mean!(s.θ.data, s.tθ.data, m, ws, t, t + Δt, Float64[]),
                            LegacyHooks._subside_mean!(s.u.data, s.tu.data, m, ws, t, t + Δt, Float64[]))),
      ("subsidence, local", s -> Tuple(apply_forcing!(s, LSF.Subsidence(ws; fields = (:θ, :v), form = :local), t, Δt)),
                      s -> (LegacyHooks._subside_local!(s.θ.data, s.tθ.data, m, ws, t, t + Δt),
                            LegacyHooks._subside_local!(s.v.data, s.tv.data, m, ws, t, t + Δt))),
      ("subsidence, local, ascent", s -> Tuple(apply_forcing!(s, LSF.Subsidence(2e-3; fields = (:θ,), form = :local), t, Δt)),
                      s -> (LegacyHooks._subside_local!(s.θ.data, s.tθ.data, m, 2e-3, t, t + Δt),)),
      ("advective tendencies", s -> (apply_forcing!(s, LSF.AdvectiveTendencies(; θ = (z, t) -> 1e-5 * sin(z / 300), u = tab), t, Δt); ()),
                      s -> (LegacyHooks._add_level_tendency!(s.tθ.data, m, (z, t) -> 1e-5 * sin(z / 300), t, Δt);
                            LegacyHooks._add_level_tendency!(s.tu.data, m, tab, t, Δt); ())))
    for (cname, new!, old!) in cases
        hk_lsf_restore!(sim, st); Iref = old!(sim); ref = hk_lsf_state(sim)
        @test any(ref[i] != st[i] for i in 1:3)
        for (bname, b) in HK_BACKENDS
            hk_lsf_restore!(sim, st)
            I = with_backend(() -> new!(sim), b)
            ok = all(hk_bits(x) == hk_bits(y) for (x, y) in zip(hk_lsf_state(sim), ref)) &&
                 hk_bits(Float64.(I)) == hk_bits(Float64.(Iref))
            ok || @info "KA-k bit mismatch (large-scale forcing)" gname cname bname
            @test ok
        end
    end
    # ReferenceDamping toward the slab mean: the level means of the target
    F = LSF.ReferenceDamping(:θ, Aeolus.Operators.sponge_profile(0.01, 1200.0, 2400.0), :mean)
    bref = LegacyHooks.resolve_plane_means!(zeros(m.Nz), sim.θ)
    for (bname, b) in HK_BACKENDS
        resize!(F.buf, m.Nz)
        got = with_backend(() -> copy(LSF._resolve_target(F, :mean, sim.θ, nothing, t)), b)
        @test hk_bits(got) == hk_bits(bref)
    end
    # the whole apply_forcing! of the damping is thread-invariant
    outs = map(HK_BACKENDS) do (bname, b)
        hk_lsf_restore!(sim, st)
        with_backend(() -> apply_forcing!(sim, F, t, Δt), b)
        (hk_bits(sim.tθ.data), hk_bits(F.levels), hk_bits(F.integral[]))
    end
    @test allequal(outs)
end

end # testset
