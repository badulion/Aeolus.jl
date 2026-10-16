# KA-d (#299): the finite-volume kernels on launch! and the grid-metric
# abstraction (Operators.FiniteVolume, StretchedGrids).
#
#   1. against the pre-port serial loops (a transcript of the
#      old src/FiniteVolume.jl, the StretchedGrid / TerrainFollowingGrid
#      stencils of src/StretchedGrids.jl and the #151 / #276 copies of
#      src/stretched/boussinesq.jl: test/support/legacy_fv.jl) for every
#      ported kernel, on a RegularGrid, on a geometric and a `:uniform`
#      StretchedGrid and on a TerrainFollowingGrid over a hill, on random
#      fields with random halos; serially, threaded with every launch on
#      the KA path (`threshold = 0`), and on the default backend. The folded
#      StretchedGrid kernels are compared with the copies they replace;
#   2. the damping kernel with `levels` (the #288 budget path) runs one
#      plane per work item: the per-level sums and the tendencies equal the
#      serial loop on every backend (at `-t 24` in the thread-invariance
#      harness, `fv` section);
#   3. the metric contract: `grid_metrics` is uniform on a RegularGrid and
#      per level on a StretchedGrid, and the accessors are the old
#      per-level expressions;
#   4. a kernel launch allocates nothing on the serial path.

using Test
using Random: Xoshiro, randn, rand
using Aeolus
using Aeolus.Grids: FaceX, FaceY, FaceZ, CellCenter, Location, interior
using Aeolus.Fields: Field
using Aeolus.StretchedGrids: StretchedGrid, TerrainFollowingGrid, terrain_following_grid
const FVK = Aeolus.Operators.FiniteVolume

isdefined(@__MODULE__, :LegacyFV) || include(joinpath(@__DIR__, "support", "legacy_fv.jl"))

const FV_BACKENDS = ("serial" => CPUBackend(serial = true),
                     "threaded, threshold 0" => CPUBackend(threshold = 0),
                     "default" => CPUBackend())
fv_bits(A) = collect(Float64, A)      # a copy, compared within ROUNDOFF_RTOL (#495)

const FV_LOCS = (CellCenter(), FaceX(), FaceY(), FaceZ())

function fv_field(rng, loc, g; scale = 1.0, offset = 0.0)
    f = Field(loc, g)
    f.data .= offset .+ scale .* randn(rng, size(f.data))
    return f
end
fv_copy(f) = (h = Field(Aeolus.Fields.location(f), f.mesh); h.data .= f.data; h)

# Run `new!(outs...)` on every backend and `old!(outs...)` once, each on its
# own copy of the output fields/arrays `outs`; every output must agree bit
# for bit (halos included).
function fv_check(name, new!, old!, outs...)
    cp(x::Field) = fv_copy(x)
    cp(x::AbstractArray) = copy(x)
    data(x::Field) = x.data
    data(x::AbstractArray) = x
    ref = map(cp, outs)
    old!(ref...)
    for (bname, b) in FV_BACKENDS
        got = map(cp, outs)
        with_backend(b) do
            new!(got...)
        end
        ok = all(agree(data(a), data(r)) for (a, r) in zip(got, ref))
        ok || @info "KA-d bit mismatch" name bname
        @test ok
    end
end

function fv_grids()
    rg = RegularGrid(Float64; size = (13, 11, 9), extent = (1.3, 1.7, 0.8), halo = (3, 3, 3))
    sg = StretchedGrid(Float64; size = (13, 11, 9), extent = (1.3, 1.7, 0.8), halo = (3, 3, 3),
                       stretching = (:geometric, 1.08))
    su = StretchedGrid(Float64; size = (13, 11, 9), extent = (1.3, 1.7, 0.8), halo = (3, 3, 3))
    return (("RegularGrid", rg), ("StretchedGrid geometric", sg), ("StretchedGrid uniform", su))
end

# the kernels both a RegularGrid and a StretchedGrid have (the folded ones)
function fv_common_kernels(rng, gname, g)
    u = fv_field(rng, FaceX(), g); v = fv_field(rng, FaceY(), g); w = fv_field(rng, FaceZ(), g)
    θ = fv_field(rng, CellCenter(), g; scale = 2.0, offset = 300.0)
    qv = fv_field(rng, CellCenter(), g; scale = 1e-3, offset = 8e-3)
    K = fv_field(rng, CellCenter(), g; scale = 0.1, offset = 1.0)
    Kv = fv_field(rng, CellCenter(), g; scale = 0.1, offset = 0.5)
    c = fv_field(rng, CellCenter(), g)
    ucc = fv_field(rng, CellCenter(), g); vcc = fv_field(rng, CellCenter(), g); wcc = fv_field(rng, CellCenter(), g)
    o(loc) = fv_field(rng, loc, g)
    t = "$gname: "

    # first derivatives (the six x/y stencils and the three z stencils)
    for (nm, out, inp, fnew, fold) in (
            ("∂x cc", o(CellCenter()), c, FVK._∂x!, LegacyFV._∂x!),
            ("∂y cc", o(CellCenter()), c, FVK._∂y!, LegacyFV._∂y!),
            ("∂z cc", o(CellCenter()), c, FVK._∂z!, LegacyFV._∂z!),
            ("∂x c→f", o(FaceX()), c, FVK._∂x!, LegacyFV._∂x!),
            ("∂y c→f", o(FaceY()), c, FVK._∂y!, LegacyFV._∂y!),
            ("∂z c→f", o(FaceZ()), c, FVK._∂z!, LegacyFV._∂z!),
            ("∂x f→c", o(CellCenter()), u, FVK._∂x!, LegacyFV._∂x!),
            ("∂y f→c", o(CellCenter()), v, FVK._∂y!, LegacyFV._∂y!),
            ("∂z f→c", o(CellCenter()), w, FVK._∂z!, LegacyFV._∂z!),
            ("interp c→fx", o(FaceX()), c, FVK._interp_to_face!, LegacyFV._interp_to_face!),
            ("interp c→fy", o(FaceY()), c, FVK._interp_to_face!, LegacyFV._interp_to_face!),
            ("interp c→fz", o(FaceZ()), c, FVK._interp_to_face!, LegacyFV._interp_to_face!),
            ("interp fx→c", o(CellCenter()), u, FVK._interp_to_center!, LegacyFV._interp_to_center!),
            ("interp fy→c", o(CellCenter()), v, FVK._interp_to_center!, LegacyFV._interp_to_center!),
            ("interp fz→c", o(CellCenter()), w, FVK._interp_to_center!, LegacyFV._interp_to_center!))
        fv_check(t * nm, x -> fnew(x, inp, g), x -> fold(x, inp, g), out)
    end
    for loc in FV_LOCS
        f = fv_field(rng, loc, g)
        fv_check(t * "laplacian $(nameof(typeof(loc)))", x -> FVK._laplacian!(x, f, g),
                 x -> LegacyFV._laplacian!(x, f, g), o(loc))
    end
    fv_check(t * "divergence staggered", x -> FVK._divergence!(x, u, v, w, g),
             x -> LegacyFV._divergence!(x, u, v, w, g), o(CellCenter()))
    fv_check(t * "divergence collocated", x -> FVK._divergence!(x, ucc, vcc, wcc, g),
             x -> LegacyFV._divergence!(x, ucc, vcc, wcc, g), o(CellCenter()))
    fv_check(t * "surface flux", x -> FVK._add_surface_flux!(x, 0.13, -0.07, g),
             x -> LegacyFV._add_surface_flux!(x, 0.13, -0.07, g), o(CellCenter()))

    # the Boussinesq core
    fv_check(t * "momentum advection", (a, b, d) -> FVK._momentum_advect!(a, b, d, u, v, w, g),
             (a, b, d) -> LegacyFV._momentum_advect!(a, b, d, u, v, w, g), o(FaceX()), o(FaceY()), o(FaceZ()))
    fv_check(t * "scalar advection (staggered)", x -> FVK._advect_centered!(x, u, v, w, θ, g),
             x -> LegacyFV._advect_centered!(x, u, v, w, θ, g), o(CellCenter()))
    fv_check(t * "buoyancy", x -> FVK._add_buoyancy!(x, θ, 300.0, 9.81, g),
             x -> LegacyFV._add_buoyancy!(x, θ, 300.0, 9.81, g), o(FaceZ()))
    fv_check(t * "moist buoyancy", x -> FVK._add_buoyancy_moist!(x, θ, qv, 300.0, 9.81, g),
             x -> LegacyFV._add_buoyancy_moist!(x, θ, qv, 300.0, 9.81, g), o(FaceZ()))
    fv_check(t * "coriolis", (a, b) -> FVK._add_coriolis!(a, b, u, v, 1e-4, g),
             (a, b) -> LegacyFV._add_coriolis!(a, b, u, v, 1e-4, g), o(FaceX()), o(FaceY()))
    fv_check(t * "smagorinsky νt", x -> FVK._compute_smagorinsky_νt!(x, u, v, w, 0.17, g),
             x -> LegacyFV._compute_smagorinsky_νt!(x, u, v, w, 0.17, g), o(CellCenter()))
    fv_check(t * "eddy viscosity u", x -> FVK._add_eddy_viscosity_u!(x, u, K, g),
             x -> LegacyFV._add_eddy_viscosity_u!(x, u, K, g), o(FaceX()))
    fv_check(t * "eddy viscosity v", x -> FVK._add_eddy_viscosity_v!(x, v, K, g),
             x -> LegacyFV._add_eddy_viscosity_v!(x, v, K, g), o(FaceY()))
    fv_check(t * "eddy viscosity w", x -> FVK._add_eddy_viscosity_w!(x, w, K, g),
             x -> LegacyFV._add_eddy_viscosity_w!(x, w, K, g), o(FaceZ()))
    fv_check(t * "eddy diffusion (K, s)", x -> FVK._add_eddy_diffusion_scalar!(x, θ, K, 1 / 0.7, g),
             x -> LegacyFV._add_eddy_diffusion_scalar!(x, θ, K, 1 / 0.7, g), o(CellCenter()))
    fv_check(t * "eddy diffusion (Kh, Kv, s)", x -> FVK._add_eddy_diffusion_scalar!(x, θ, K, Kv, 1.0, g),
             x -> LegacyFV._add_eddy_diffusion_scalar!(x, θ, K, Kv, 1.0, g), o(CellCenter()))

    # the sponge (one damping kernel): constant, level-vector, field targets
    zt = g.z0 + g.Lz
    for (tn, target) in (("constant", 300.0), ("levels", collect(range(299.0, 301.0; length = g.Nz))),
                         ("field", θ))
        fv_check(t * "sponge, $tn target", x -> FVK._add_sponge!(x, θ, 0.01, g.z0 + 0.4g.Lz, zt, target, g),
                 x -> LegacyFV._add_damping!(x, θ, FVK.sponge_profile(0.01, g.z0 + 0.4g.Lz, zt),
                                             FVK.damping_coordinates(g, CellCenter()), 3, target, g),
                 o(CellCenter()))
    end
    # set_physical! (the whole storage, analytically)
    fun(x, y, z) = sin(3x) * cos(2y) + z^2
    for loc in FV_LOCS
        fv_check(t * "set_physical! $(nameof(typeof(loc)))",
                 x -> Aeolus.StretchedGrids._set_physical!(x, fun, g),
                 x -> LegacyFV._set_physical!(x, fun, g), o(loc))
    end
    return nothing
end

# the kernels that exist on a RegularGrid only
function fv_regular_kernels(rng, g)
    u = fv_field(rng, FaceX(), g); v = fv_field(rng, FaceY(), g); w = fv_field(rng, FaceZ(), g)
    θ = fv_field(rng, CellCenter(), g; scale = 2.0, offset = 300.0)
    ucc = fv_field(rng, CellCenter(), g); vcc = fv_field(rng, CellCenter(), g); wcc = fv_field(rng, CellCenter(), g)
    nh = abs.(randn(rng, size(θ.data))); nv = abs.(randn(rng, size(θ.data)))
    o(loc) = fv_field(rng, loc, g)
    arr() = randn(rng, size(θ.data))
    t = "RegularGrid: "

    fv_check(t * "scalar advection (collocated)", x -> FVK._advect_centered!(x, ucc, vcc, wcc, θ, g),
             x -> LegacyFV._advect_centered!(x, ucc, vcc, wcc, θ, g), o(CellCenter()))
    fv_check(t * "upwind advection", x -> FVK._advect_upwind!(x, ucc, vcc, wcc, θ, g),
             x -> LegacyFV._advect_upwind!(x, ucc, vcc, wcc, θ, g), o(CellCenter()))
    fv_check(t * "WENO-5 advection", x -> FVK._advect_weno5!(x, u, v, w, θ, g),
             x -> LegacyFV._advect_weno5!(x, u, v, w, θ, g), o(CellCenter()))
    for (a, b) in ((1, 1), (1, 0), (0, 1))
        fv_check(t * "SGS stress divergence (a, b) = ($a, $b)",
                 (x, y, z) -> FVK._add_sgs_stress_divergence!(x, y, z, u, v, w, nh, nv, a, b, g),
                 (x, y, z) -> LegacyFV._add_sgs_stress_divergence!(x, y, z, u, v, w, nh, nv, a, b, g),
                 o(FaceX()), o(FaceY()), o(FaceZ()))
    end
    fv_check(t * "SGS stress tensor",
             (τs...) -> FVK._sgs_stress_tensor!(τs..., u, v, w, nh, nv, g),
             (τs...) -> LegacyFV._sgs_stress_tensor!(τs..., u, v, w, nh, nv, g),
             arr(), arr(), arr(), arr(), arr(), arr())
    τ = ntuple(_ -> arr(), 6)
    fv_check(t * "stored-stress divergence",
             (x, y, z) -> FVK._add_stress_divergence!(x, y, z, τ..., g),
             (x, y, z) -> LegacyFV._add_stress_divergence!(x, y, z, τ..., g),
             o(FaceX()), o(FaceY()), o(FaceZ()))
    Ks = ntuple(_ -> arr(), 6)
    for bounded in ((false, false, false), (false, false, true), (true, true, true)),
        (dodiag, dooff) in ((true, true), (true, false), (false, true))
        fv_check(t * "tensor diffusion bounded=$bounded diag=$dodiag off=$dooff",
                 (x, Fx, Fy, Fz) -> FVK._add_tensor_diffusion_scalar!(x, θ, Ks..., Fx, Fy, Fz,
                                                                       bounded, dodiag, dooff, g),
                 (x, Fx, Fy, Fz) -> LegacyFV._add_tensor_diffusion_scalar!(x, θ, Ks..., Fx, Fy, Fz,
                                                                            bounded, dodiag, dooff, g),
                 o(CellCenter()), arr(), arr(), arr())
    end
    K = fv_field(rng, CellCenter(), g; scale = 0.1, offset = 1.0)
    Kv = fv_field(rng, CellCenter(), g; scale = 0.1, offset = 0.5)
    mask = rand(rng, size(θ.data)...) .< 0.2
    nz = size(θ.data, 3)
    wc = 1 .+ rand(rng, nz); wf = 1 .+ rand(rng, nz + 1)
    for (mn, mk) in (("no mask", nothing), ("mask", mask)), (wn, ws) in (("unweighted", (nothing, nothing)),
                                                                      ("weighted", (wc, wf)))
        fv_check(t * "masked eddy diffusion, $mn, $wn",
                 x -> FVK._add_eddy_diffusion_scalar_masked!(x, θ, K, Kv, g, mk, ws...),
                 x -> LegacyFV._add_eddy_diffusion_scalar_masked!(x, θ, K, Kv, g, mk, ws...),
                 o(CellCenter()))
    end
    fv_check(t * "masked eddy diffusion, one coefficient",
             x -> FVK._add_eddy_diffusion_scalar_masked!(x, θ, K, g, mask, nothing, nothing),
             x -> LegacyFV._add_eddy_diffusion_scalar_masked!(x, θ, K, g, mask, nothing, nothing),
             o(CellCenter()))
    θs = fv_field(rng, CellCenter(), g; scale = 0.5, offset = 300.0)
    θs.data .+= reshape(1:size(θ.data, 3), 1, 1, :) .* 0.3   # stable stratification
    for (κn, κh, κv) in (("numbers", 2.0, 0.3), ("arrays", abs.(arr()), abs.(arr())))
        fv_check(t * "isentropic tensor, κ $κn",
                 (Ks...) -> FVK._isentropic_tensor!(Ks..., θs, κh, κv, 0.3, 1e-3, g),
                 (Ks...) -> LegacyFV._isentropic_tensor!(Ks..., θs, κh, κv, 0.3, 1e-3, g),
                 arr(), arr(), arr(), arr(), arr(), arr())
    end
    # the damping kernel along x (the fringe) and on 3-D coordinates
    fr = FVK.DampingProfile(FVK.NordstromRamp(), 0.05, 0.6, 0.2; stop = 1.2, fall = 0.1)
    xs = collect(range(0.05, 1.25; length = g.Nx))
    fv_check(t * "fringe (axis 1)", x -> FVK._add_damping!(x, θ, fr, xs, 1, 300.0, g),
             x -> LegacyFV._add_damping!(x, θ, fr, xs, 1, 300.0, g), o(CellCenter()))
    z3 = 0.8 .* rand(rng, size(θ.data)...)
    sp = FVK.sponge_profile(0.02, 0.3, 0.8)
    fv_check(t * "sponge on 3-D heights", x -> FVK._add_damping!(x, θ, sp, z3, 3, θs, g),
             x -> LegacyFV._add_damping!(x, θ, sp, z3, 3, θs, g), o(CellCenter()))
    return nothing
end

function fv_tfg_kernels(rng)
    Lx, Lz = 2000.0, 1500.0
    bell(x, y) = 150.0 / (1 + ((x - Lx / 2) / 300.0)^2) + 20.0 * sin(2π * y / 400.0)
    sg = StretchedGrid(Float64; size = (13, 11, 9), extent = (Lx, 400.0, Lz), halo = (2, 2, 2),
                       stretching = (:tanh, 1.4))
    g = terrain_following_grid(sg, bell)
    o(loc) = fv_field(rng, loc, g)
    u = o(FaceX()); v = o(FaceY()); w = o(FaceZ()); c = o(CellCenter())
    ucc = o(CellCenter()); vcc = o(CellCenter()); wcc = o(CellCenter())
    θ = fv_field(rng, CellCenter(), g; scale = 2.0, offset = 300.0)
    t = "TerrainFollowingGrid: "
    for (nm, out, inp, fnew, fold) in (
            ("∂x cc", o(CellCenter()), c, FVK._∂x!, LegacyFV._∂x!),
            ("∂y cc", o(CellCenter()), c, FVK._∂y!, LegacyFV._∂y!),
            ("∂z cc", o(CellCenter()), c, FVK._∂z!, LegacyFV._∂z!),
            ("∂x c→f", o(FaceX()), c, FVK._∂x!, LegacyFV._∂x!),
            ("∂y c→f", o(FaceY()), c, FVK._∂y!, LegacyFV._∂y!),
            ("∂z c→f", o(FaceZ()), c, FVK._∂z!, LegacyFV._∂z!),
            ("∂x f→c", o(CellCenter()), u, FVK._∂x!, LegacyFV._∂x!),
            ("∂y f→c", o(CellCenter()), v, FVK._∂y!, LegacyFV._∂y!),
            ("∂z f→c", o(CellCenter()), w, FVK._∂z!, LegacyFV._∂z!),
            ("interp c→fz", o(FaceZ()), c, FVK._interp_to_face!, LegacyFV._interp_to_face!),
            ("interp fz→c", o(CellCenter()), w, FVK._interp_to_center!, LegacyFV._interp_to_center!))
        fv_check(t * nm, x -> fnew(x, inp, g), x -> fold(x, inp, g), out)
    end
    for loc in FV_LOCS
        f = fv_field(rng, loc, g)
        fv_check(t * "laplacian $(nameof(typeof(loc)))", x -> FVK._laplacian!(x, f, g),
                 x -> LegacyFV._laplacian!(x, f, g), o(loc))
    end
    fv_check(t * "divergence staggered", x -> FVK._divergence!(x, u, v, w, g),
             x -> LegacyFV._divergence!(x, u, v, w, g), o(CellCenter()))
    fv_check(t * "divergence collocated", x -> FVK._divergence!(x, ucc, vcc, wcc, g),
             x -> LegacyFV._divergence!(x, ucc, vcc, wcc, g), o(CellCenter()))
    fv_check(t * "surface flux", x -> FVK._add_surface_flux!(x, 0.13, -0.07, g),
             x -> LegacyFV._add_surface_flux!(x, 0.13, -0.07, g), o(CellCenter()))
    fv_check(t * "sponge", x -> FVK._add_sponge!(x, θ, 0.01, 900.0, Lz, 300.0, g),
             x -> LegacyFV._add_sponge!(x, θ, 0.01, 900.0, Lz, 300.0, g), o(CellCenter()))
    fv_check(t * "buoyancy", x -> FVK._add_buoyancy!(x, θ, 300.0, 9.81, g),
             x -> LegacyFV._add_buoyancy!(x, θ, 300.0, 9.81, g), o(FaceZ()))
    fun(x, y, z) = 1e-3x + sin(y / 50) + z / 100
    fv_check(t * "set_physical!", x -> Aeolus.StretchedGrids._set_physical!(x, fun, g),
             x -> LegacyFV._set_physical!(x, fun, g), o(CellCenter()))
    return nothing
end

@testset "KA-d: finite-volume kernels on launch! " begin
    rng = Xoshiro(299)
    for (gname, g) in fv_grids()
        @testset "$gname" begin
            fv_common_kernels(rng, gname, g)
        end
    end
    @testset "RegularGrid only" begin
        fv_regular_kernels(rng, fv_grids()[1][2])
    end
    @testset "TerrainFollowingGrid" begin
        fv_tfg_kernels(rng)
    end
end

@testset "KA-d: damping with levels (one plane per work item)" begin
    rng = Xoshiro(288)
    for (gname, g) in fv_grids()
        θ = fv_field(rng, CellCenter(), g; scale = 2.0, offset = 300.0)
        tf0 = fv_field(rng, CellCenter(), g)
        p = FVK.sponge_profile(0.01, g.z0 + 0.3g.Lz, g.z0 + g.Lz)
        zs = FVK.damping_coordinates(g, CellCenter())
        target = collect(range(299.0, 301.0; length = g.Nz))
        ref = fv_copy(tf0); lref = zeros(g.Nz)
        LegacyFV._add_damping!(ref, θ, p, zs, 3, target, g; levels = lref)
        for (bname, b) in FV_BACKENDS
            tf = fv_copy(tf0); lv = zeros(g.Nz)
            with_backend(b) do
                FVK._add_damping!(tf, θ, p, zs, 3, target, g; levels = lv)
            end
            @test agree(tf.data, ref.data)
            @test agree(lv, lref)
        end
        # the levels hold what was added to each level (a budget's integral)
        tf = fv_copy(tf0); lv = zeros(g.Nz)
        FVK._add_damping!(tf, θ, p, zs, 3, target, g; levels = lv)
        ii, jj, kk = interior(g)
        added = [sum(tf.data[i, j, k] - tf0.data[i, j, k] for j in jj, i in ii) for k in kk]
        @test lv ≈ added rtol = 1e-12 atol = 1e-12
        @test any(!iszero, lv) && iszero(lv[1])
        @test_throws DimensionMismatch FVK._add_damping!(tf, θ, p, zs, 3, target, g; levels = zeros(g.Nz - 1))
    end
end

@testset "KA-d: the grid-metric contract" begin
    rg, sg, su = (x[2] for x in fv_grids())
    mr = FVK.grid_metrics(rg)
    @test mr.z isa FVK.UniformZ
    @test FVK._idz(mr, 5) === 1 / rg.dz && FVK._idzf(mr, 5) === 1 / rg.dz && FVK._idz_off(mr, 5) === 1 / rg.dz
    @test FVK._idx(mr, 3) === 1 / rg.dx && FVK._idy(mr, 3) === 1 / rg.dy
    @test FVK._zface(mr, 5, 1.0, 2.0) === 1.5
    @test FVK._filter_width(mr, 5) === (rg.dx * rg.dy * rg.dz)^(1/3)
    ms = FVK.grid_metrics(sg)
    @test ms.z isa FVK.LevelZ
    @test ms.z === sg.zm                     # cached with the grid, not rebuilt per call
    for k in sg.Hz:sg.Hz+sg.Nz+1
        @test FVK._idz(ms, k) === 1 / sg.dzc[k]
        @test FVK._idzf(ms, k) === 1 / sg.dzf[k]
        @test FVK._idz_off(ms, k) === 2 / (sg.dzf[k] + sg.dzf[k+1])
        @test FVK._filter_width(ms, k) === (sg.dx * sg.dy * sg.dzc[k])^(1/3)
        a, b = 0.3, -1.7
        @test FVK._zface(ms, k, a, b) === sg.dzc[k] / (sg.dzc[k-1] + sg.dzc[k]) * a +
                                           sg.dzc[k-1] / (sg.dzc[k-1] + sg.dzc[k]) * b
    end
    # the geometric weights are exact for a linear profile; ½ on the uniform grid
    k = sg.Hz + 4
    @test FVK._zface(ms, k, sg.zc[k-1], sg.zc[k]) ≈ sg.zf[k] rtol = 1e-14
    @test FVK._zcentre(ms, k, sg.zf[k], sg.zf[k+1]) ≈ sg.zc[k] rtol = 1e-14
    mu = FVK.grid_metrics(su)
    @test all(FVK._zface(mu, k, 1.0, 3.0) == 2.0 for k in su.Hz+1:su.Hz+su.Nz)
    # a mesh without metrics has its own kernels or none
    @test_throws MethodError FVK.grid_metrics(terrain_following_grid(sg, 0.0))
end

@testset "KA-d: kernels allocate nothing on the serial path" begin
    rg, sg, _ = (x[2] for x in fv_grids())
    for g in (rg, sg)
        u = Field(FaceX(), g); v = Field(FaceY(), g); w = Field(FaceZ(), g)
        θ = Field(CellCenter(), g); K = Field(CellCenter(), g); t = Field(CellCenter(), g)
        tu = Field(FaceX(), g); tv = Field(FaceY(), g); tw = Field(FaceZ(), g)
        with_backend(CPUBackend(serial = true)) do
            FVK._momentum_advect!(tu, tv, tw, u, v, w, g)
            FVK._add_eddy_diffusion_scalar!(t, θ, K, K, 1.0, g)
            FVK._laplacian!(tw, w, g)
            @test (@allocated FVK._momentum_advect!(tu, tv, tw, u, v, w, g)) == 0
            @test (@allocated FVK._add_eddy_diffusion_scalar!(t, θ, K, K, 1.0, g)) == 0
            @test (@allocated FVK._laplacian!(tw, w, g)) == 0
        end
    end
end
