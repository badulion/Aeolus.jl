# KA-f (#297): halo fills, open-boundary halos and the multigrid level halo
# as launches of the per-axis halo kernels (boundary_conditions/
# halo_kernels.jl).
#
#   1. bit for bit against the pre-port fill (a transcript of the plane
#      copies of master at 70d9f83, below) for every BC type and
#      combination, every location, on random fields with random halos
#      (edges and corners included), at a regular size, at uneven halos
#      and at a degenerate size (N < H, where the layers of a pencil depend
#      on each other); serially, threaded with every launch on the KA path
#      (`threshold = 0`), and on the default backend;
#   2. corner order: the axes are filled x, then y, then z, so an edge or
#      corner holds the value of the last axis that reaches it;
#   3. the multigrid level halo is the same kernel (one source);
#   4. halo fills allocate nothing (the open faces allocated 0.8-3.4 kB
#      per fill before the port);
#   5. the nesting transfer operators — prolongation (orders 1/3/5, the
#      :positive / :monotone limiters, halo / interface / whole child,
#      halo-only shells), the restrictions (volume, mass measure,
#      solid-aware), the flux mismatch, the face flux correction and the
#      refluxing — bit for bit against the pre-port operators
#      (test/support/legacy_nesting.jl) on every backend;
#   6. a short one-way and two-way (with a sponge) nested run: serial =
#      threaded, in process (across processes: the worker's `nested`
#      section of test/thread_invariance.jl);
#   7. SC-b (#402): the fill of a set of fields (two launches: x and y by
#      k-planes, then z by j-rows) is bit for bit the pre-port fill of each
#      field, for every BC type and every chunking of the planes and rows.

using Test
using Random: Xoshiro, randn
using Aeolus
using Aeolus.Grids: FaceX, FaceY, FaceZ, CellCenter, Location
using Aeolus.Fields: Field
using Aeolus.BoundaryConditions: Periodic, NoFlux, ImpermeableWall, NoSlipWall, FreeStreamBoundary,
                                 FluxBC, OpenBoundary, OpenPlane, _ZWall, _wall_velocity,
                                 _side_axis, _face_index, _face_local, _center_gap, homogeneous_bcs

const KH_BACKENDS = ("serial" => CPUBackend(serial = true),
                     "threaded, threshold 0" => CPUBackend(threshold = 0),
                     "default" => CPUBackend())
kh_bits(A) = reinterpret(UInt64, vec(collect(Float64, A)))

# The pre-port fill (master at 70d9f83), transcribed: test/support/legacy_halos.jl.
isdefined(@__MODULE__, :LegacyHalos) || include(joinpath(@__DIR__, "support", "legacy_halos.jl"))
isdefined(@__MODULE__, :LegacyNesting) || include(joinpath(@__DIR__, "support", "legacy_nesting.jl"))
using Aeolus.Nesting

# ---------------------------------------------------------------------------
const KH_LOCS = (CellCenter(), FaceX(), FaceY(), FaceZ())

function kh_closed_cases(m)
    return [("periodic", FieldBoundaryConditions(:periodic)),
            ("noflux", FieldBoundaryConditions(:noflux)),
            ("periodic + free-slip walls", FieldBoundaryConditions(:periodic, :periodic, :wall)),
            ("noflux x + walls", FieldBoundaryConditions(:noflux, :periodic, :wall)),
            ("no-slip / free-slip", FieldBoundaryConditions(:periodic, :periodic; bottom = :noslip, top = :wall)),
            ("free-slip / moving lid", FieldBoundaryConditions(:periodic, :periodic; bottom = :wall,
                                                               top = NoSlipWall(1.5, -0.5))),
            ("two moving walls", FieldBoundaryConditions(:periodic, :noflux; bottom = NoSlipWall(0.3, 0.7),
                                                         top = NoSlipWall(1.5, -0.5))),
            ("free-stream top", FieldBoundaryConditions(:periodic, :periodic; bottom = :noslip,
                top = FreeStreamBoundary(collect(range(-1.0, 1.0; length = m.Nx + 2m.Hx))))),
            ("gradient top / noflux", FieldBoundaryConditions(:periodic, :periodic; bottom = :noflux,
                                                              top = FluxBC(0.003))),
            ("gradient top / wall", FieldBoundaryConditions(:periodic, :noflux; bottom = :wall,
                                                            top = FluxBC(-0.01)))]
end

function kh_open_cases(m, rng)
    data = ProfileBoundaryData(u = 1.0, v = 0.5, w = 0.0, θ = 300.0)
    out = []
    for (lab, kw) in (("open, value halos", (;)), ("open, odd normal halo", (; normal_halo = :odd)),
                      ("open, mirror scalars", (; scalar_halo = :mirror)),
                      ("open x, periodic y", (; y = :periodic)), ("open, no-slip top", (; top = :noslip)),
                      ("open bottom and top", (; bottom = :open, top = :open, normal_halo = :odd)))
        bcs = open_boundary_conditions(m, data; kw...)
        st = only(open_boundary_states(bcs))
        for p in st.planes; p.data .= randn(rng, size(p.data)); end
        push!(out, (lab, bcs, st), (lab * " (homogeneous)", homogeneous_bcs(bcs), st))
    end
    bcs = open_boundary_conditions(m, data)
    add_open_scalars!(bcs, :qv; mode = :inflow)
    st = only(open_boundary_states(bcs))
    for p in st.planes; p.data .= randn(rng, size(p.data)); end
    push!(out, ("open, inflow scalar", bcs, st))
    return out
end

# Fill a copy with the legacy code and the field with fill_halos! on each backend.
function kh_compare(f, bcs, init)
    ref = Field(Aeolus.Fields.location(f), f.mesh); ref.data .= init
    LegacyHalos.legacy_fill!(ref, bcs; key = f.data)
    ok = true
    for (_, b) in KH_BACKENDS
        f.data .= init
        with_backend(b) do
            fill_halos!(f, bcs)
        end
        ok &= kh_bits(f.data) == kh_bits(ref.data)
    end
    return ok
end

@testset "KA-f halo kernels and nesting transfer operators (#297)" begin
    meshes = (("12×10×8, H = 3", RegularGrid(size = (12, 10, 8), extent = (1.2, 1.0, 0.8), halo = (3, 3, 3))),
              ("9×7×6, H = (2, 4, 3)", RegularGrid(size = (9, 7, 6), extent = (0.9, 0.7, 0.6), halo = (2, 4, 3))),
              ("5×1×2, H = 3 (N < H)", RegularGrid(size = (5, 1, 2), extent = (0.5, 0.1, 0.2), halo = (3, 3, 3))))

    @testset "bit for bit with the pre-port fill: $mlab" for (mlab, m) in meshes
        rng = Xoshiro(297)
        n = 0
        for (lab, bcs) in kh_closed_cases(m), loc in KH_LOCS
            (bcs.top isa FluxBC && !(loc isa CellCenter)) && continue
            f = Field(loc, m)
            ok = kh_compare(f, bcs, randn(rng, size(f.data)))
            ok || @info "differs: $lab $(nameof(typeof(loc)))"
            @test ok
            n += 1
        end
        if m.Ny >= 3
            for (lab, bcs, st) in kh_open_cases(m, rng), loc in KH_LOCS
                f = Field(loc, m)
                loc isa CellCenter && (st.bound[f.data] = endswith(lab, "inflow scalar") ? :qv : :θ)
                ok = kh_compare(f, bcs, randn(rng, size(f.data)))
                ok || @info "differs: $lab $(nameof(typeof(loc)))"
                @test ok
                if loc isa CellCenter           # a field without a plane: the mirror
                    g = Field(loc, m)
                    @test kh_compare(g, bcs, randn(rng, size(g.data)))
                    n += 1
                end
                n += 1
            end
        end
        @info "KA-f halos, $mlab: $n fills bit-identical to the pre-port loops on $(length(KH_BACKENDS)) backends"
    end

    @testset "gradient top on a stretched grid" begin
        sm = Aeolus.StretchedGrids.StretchedGrid(; size = (6, 5, 8), extent = (0.6, 0.5, 0.9),
                                                 stretching = (:geometric, 1.15), halo = (3, 3, 3))
        f = Field(CellCenter(), sm)
        @test kh_compare(f, FieldBoundaryConditions(:periodic, :periodic; bottom = :wall, top = FluxBC(0.004)),
                         randn(Xoshiro(5), size(f.data)))
    end

    @testset "corner order: x, then y, then z" begin
        m = RegularGrid(size = (8, 6, 5), extent = (0.8, 0.6, 0.5), halo = (2, 2, 2))
        H = 2; Nx, Ny, Nz = 8, 6, 5
        # open (Dirichlet) x faces, periodic y, free-slip walls in z
        bcs = open_boundary_conditions(m, ProfileBoundaryData(u = 1.0, v = 0.0, w = 0.0, θ = 300.0);
                                       y = :periodic, top = :wall)
        st = only(open_boundary_states(bcs))
        for p in st.planes; p.data .= randn(Xoshiro(1), size(p.data)); end
        θ = Field(CellCenter(), m); w = Field(FaceZ(), m)
        st.bound[θ.data] = :θ
        for (_, b) in KH_BACKENDS
            θ.data .= randn(Xoshiro(2), size(θ.data)); w.data .= randn(Xoshiro(3), size(w.data))
            with_backend(b) do
                fill_halos!(θ, bcs); fill_halos!(w, bcs)
            end
            Pθ = bcs.west.planes[:θ].data            # west θ buffer = storage x layers 1:H
            # an x-halo cell in the y halo takes the y-periodic image of the x buffer
            # (y fills after x), and in the z halo its z mirror (z fills last)
            ok = true
            for k in H+1:H+Nz, hy in 1:H, hx in 1:H
                ok &= θ.data[hx, hy, k] == Pθ[hx, Ny + hy, k]
                ok &= θ.data[hx, H + Ny + hy, k] == Pθ[hx, H + hy, k]
            end
            for hz in 1:H, hy in 1:H, hx in 1:H              # corner: all three
                ok &= θ.data[hx, hy, H + 1 - hz] == Pθ[hx, Ny + hy, H + hz]
            end
            @test ok
            # the wall face of w is zero across the x and y halos too (z last)
            @test all(iszero, view(w.data, :, :, H + 1)) && all(iszero, view(w.data, :, :, H + Nz + 1))
            @test w.data[1, 1, H] == -w.data[1, 1, H + 2]
        end
    end

    @testset "multigrid level halo: the same kernels" begin
        rng = Xoshiro(11)
        for bc in ((:periodic, :periodic, :periodic), (:neumann, :periodic, :neumann), (:neumann, :neumann, :neumann))
            φ = randn(rng, 10, 7, 6); ref = copy(φ)
            Aeolus.Multigrid._fill_halo!(φ, bc...)
            for (ax, kind) in enumerate(bc)            # the pre-port plane copies
                n = size(ref, ax) - 2
                kind === :periodic ? LegacyHalos.periodic!(ref, ax, 1, n) : LegacyHalos.mirror!(ref, ax, 1, n)
            end
            @test kh_bits(φ) == kh_bits(ref)
        end
    end

    @testset "halo fills allocate nothing" begin
        m = RegularGrid(size = (16, 12, 8), extent = (1.0, 1.0, 1.0), halo = (3, 3, 3))
        ob = open_boundary_conditions(m, ProfileBoundaryData(u = 1.0, v = 0.5, w = 0.0, θ = 300.0);
                                      normal_halo = :odd)
        add_open_scalars!(ob, :qv; mode = :inflow)
        st = only(open_boundary_states(ob))
        allocs = Int[]
        for bcs in (kh_closed_cases(m)[[1, 3, 7, 8]]..., ("open", ob)), loc in KH_LOCS
            f = Field(loc, m); loc isa CellCenter && (st.bound[f.data] = :qv)
            fill_halos!(f, bcs[2]); fill_halos!(f, bcs[2])
            push!(allocs, @allocated fill_halos!(f, bcs[2]))
        end
        @test all(iszero, allocs)
        φ = randn(10, 10, 10); Aeolus.Multigrid._fill_halo!(φ, :periodic, :neumann, :periodic)
        @test (@allocated Aeolus.Multigrid._fill_halo!(φ, :periodic, :neumann, :periodic)) == 0
    end

    @testset "nesting transfer operators: bit for bit with the pre-port loops" begin
        pm = RegularGrid(Float64; size = (16, 16, 16), extent = (1.0, 1.0, 1.0), halo = (3, 3, 3))
        nest = nested_grid(pm, ((0.25, 0.75), (0.25, 0.625), (0.0, 0.5)), 2; halo = (3, 3, 3))
        cm = nest_child_mesh(nest)
        rng = Xoshiro(42)
        L = LegacyNesting
        # run `new!(dst)` on every backend and `old!(ref)` once, from the same start
        function same(new!, old!, dst::Field, init)
            ref = Field(Aeolus.Fields.location(dst), dst.mesh); ref.data .= init
            old!(ref)
            ok = true
            for (_, b) in KH_BACKENDS
                dst.data .= init
                with_backend(() -> new!(dst), b)
                ok &= kh_bits(dst.data) == kh_bits(ref.data)
            end
            return ok
        end
        n = 0
        for loc in KH_LOCS
            pf = Field(loc, pm); pf.data .= randn(rng, size(pf.data))
            cf = Field(loc, cm); c0 = randn(rng, size(cf.data))
            for order in (1, 3, 5), lim in (:none, :positive, :monotone)
                (loc isa CellCenter || lim !== :positive) || continue
                for (f, g) in ((interpolate_to_nest_boundary!, L.interpolate_to_nest_boundary!),
                               (interpolate_parent_to_child!, L.interpolate_parent_to_child!))
                    @test same(c -> f(c, pf, nest; order, limiter = lim),
                               c -> g(c, pf, nest; order, limiter = lim), cf, c0)
                    n += 1
                end
                @test same(c -> interpolate_to_nest_boundary!(c, pf, nest; order, limiter = lim,
                                                              impose_interface = true),
                           c -> L.interpolate_to_nest_boundary!(c, pf, nest; order, limiter = lim,
                                                                impose_interface = true), cf, c0)
                n += 1
            end
            cf.data .= c0
            solid = rand(rng, size(cf.data)...) .< 0.2
            for ib in (false, true), sol in (nothing, solid)
                (loc isa CellCenter && ib) && continue
                kw = loc isa CellCenter ? (; solid = sol) : (; include_boundary = ib, solid = sol)
                @test same(p -> restrict_to_parent!(p, cf, nest; kw...),
                           p -> L.restrict_to_parent!(p, cf, nest; kw...), pf, copy(pf.data))
                n += 1
            end
            loc isa CellCenter && continue
            mm = compute_nest_flux_mismatch(pf, cf, nest)
            mo = L.compute_nest_flux_mismatch(pf, cf, nest)
            @test all(kh_bits(getfield(mm, s)) == kh_bits(getfield(mo, s)) for s in fieldnames(typeof(mm)))
            @test same(p -> apply_flux_correction!(p, mm, nest), p -> L.apply_flux_correction!(p, mm, nest),
                       pf, copy(pf.data))
            θp = Field(CellCenter(), pm); θ0 = randn(rng, size(θp.data))
            for per in ((true, true, true), (true, false, false), (false, true, false))
                @test same(t -> apply_flux_correction!(t, mm, nest; periodic = per),
                           t -> L.apply_flux_correction!(t, mm, nest; periodic = per), θp, θ0)
            end
            n += 5
        end
        # the mass measure of a sound-proof pair (weighted restriction and mismatch)
        using_anel = Aeolus.Anelastic
        rp = using_anel.isothermal_reference(pm; T_surf = 300.0)
        rc = using_anel.isothermal_reference(cm; T_surf = 300.0)
        pA = using_anel.AnelasticSimulation(pm; ref = rp, bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
        cA = using_anel.AnelasticSimulation(cm; ref = rc, bcs = nest_bcs(cm, :nest, :nest; bottom = :wall, top = :nest))
        meas = nest_measure(nest, pA, cA)
        for loc in KH_LOCS
            pf = Field(loc, pm); pf.data .= randn(rng, size(pf.data))
            cf = Field(loc, cm); cf.data .= randn(rng, size(cf.data))
            @test same(c -> interpolate_parent_to_child!(c, pf, nest; measure = meas),
                       c -> L.interpolate_parent_to_child!(c, pf, nest; measure = meas), cf, copy(cf.data))
            @test same(p -> restrict_to_parent!(p, cf, nest; measure = meas),
                       p -> L.restrict_to_parent!(p, cf, nest; measure = meas), pf, copy(pf.data))
            n += 2
            loc isa CellCenter && continue
            mm = compute_nest_flux_mismatch(pf, cf, nest; measure = meas)
            mo = L.compute_nest_flux_mismatch(pf, cf, nest; measure = meas)
            @test all(kh_bits(getfield(mm, s)) == kh_bits(getfield(mo, s)) for s in fieldnames(typeof(mm)))
            @test same(p -> apply_flux_correction!(p, mm, nest; measure = meas),
                       p -> L.apply_flux_correction!(p, mm, nest; measure = meas), pf, copy(pf.data))
            n += 2
        end
        @info "KA-f nesting: $n operator calls bit-identical to the pre-port loops on $(length(KH_BACKENDS)) backends"
    end

    @testset "nested runs: serial = threaded" begin
        function nested_pair(fb; sponge = nothing)
            pm = RegularGrid(Float64; size = (16, 16, 16), extent = (1.0, 1.0, 1.0), halo = (3, 3, 3))
            nest = nested_grid(pm, ((0.25, 0.75), (0.25, 0.75), (0.25, 0.75)), 2; halo = (3, 3, 3))
            cm = nest_child_mesh(nest)
            rng = Xoshiro(329)
            p = BoussinesqSimulation(pm; ν = 1e-3)
            for f in (p.u, p.v, p.w)
                Aeolus.Fields.interior_view(f) .= 0.3 .* randn(rng, size(Aeolus.Fields.interior_view(f)))
            end
            Aeolus.Fields.interior_view(p.θ) .= 300 .+ randn(rng, size(Aeolus.Fields.interior_view(p.θ)))
            step!(p, 1e-10); p.t = 0.0
            c = BoussinesqSimulation(cm; ν = 1e-3, bcs = nest_bcs(cm))
            for (cf, pf) in ((c.u, p.u), (c.v, p.v), (c.w, p.w), (c.θ, p.θ))
                interpolate_parent_to_child!(cf, pf, nest)
            end
            kw = sponge === nothing ? (;) : (; sponge)
            return NestedRun(p, c, nest; cfl_parent = nothing, cfl_child = nothing, feedback = fb, kw...)
        end
        for (lab, fb, kw) in (("one-way", :one_way, (;)), ("two-way + sponge", :two_way, (; sponge = NestSponge())))
            res = map((CPUBackend(serial = true), CPUBackend(threshold = 0))) do b
                with_backend(b) do
                    run = nested_pair(fb; kw...)
                    for _ in 1:2; step_nested!(run; dt = 2e-3); end
                    [kh_bits(getproperty(s, f).data) for s in (run.child, run.parent) for f in (:u, :v, :w, :θ, :p)]
                end
            end
            @test res[1] == res[2]
        end
    end

    @testset "SC-b: set fills, by planes then rows, bit for bit (#402)" begin
        BCm = Aeolus.BoundaryConditions
        for (mlab, m) in meshes
            rng = Xoshiro(402)
            cases = Any[(lab, bcs, nothing) for (lab, bcs) in kh_closed_cases(m)]
            m.Ny >= 3 && append!(cases, kh_open_cases(m, rng))
            n = 0
            for (lab, bcs, st) in cases
                fs = Tuple(Field(loc, m) for loc in KH_LOCS)        # θ, u, v, w
                # a gradient top is a scalar condition: the velocities take the wall pair
                vb = bcs.top isa FluxBC ? FieldBoundaryConditions(bcs.west, bcs.east, bcs.south, bcs.north,
                                                                  bcs.bottom, bcs.bottom) : bcs
                bt = (bcs, vb, vb, vb)
                st === nothing || (st.bound[fs[1].data] = endswith(lab, "inflow scalar") ? :qv : :θ)
                init = [randn(rng, size(f.data)) for f in fs]
                ref = map(fs, bt, init) do f, b, x
                    r = Field(Aeolus.Fields.location(f), m); r.data .= x
                    LegacyHalos.legacy_fill!(r, b; key = f.data)
                    kh_bits(r.data)
                end
                same() = all(kh_bits(f.data) == r for (f, r) in zip(fs, ref))
                # every chunking, including more chunks than planes or rows
                for nc in (1, 2, 5, 3 + size(fs[1].data, 3))
                    foreach((f, x) -> f.data .= x, fs, init)
                    BCm._fill_set_threaded!(CPUBackend(), fs, bt, map(f -> f.data, fs), nc)
                    ok = same()
                    ok || @info "set fill differs: $mlab, $lab, $nc chunks"
                    @test ok
                end
                for (_, b) in KH_BACKENDS                       # the public set fill
                    foreach((f, x) -> f.data .= x, fs, init)
                    with_backend(() -> fill_halos!(fs, bt), b)
                    @test same()
                end
                n += 1
            end
            @info "SC-b set fills, $mlab: $n BC cases × 4 locations bit-identical to the pre-port loops"
        end
        m = meshes[1][2]
        fs = (Field(FaceX(), m), Field(CellCenter(), m))
        bcs = FieldBoundaryConditions(:periodic, :periodic, :wall)
        @test_throws ArgumentError fill_halos!(fs, (bcs,))
        @test_throws ArgumentError fill_halos!(fs, bcs; keys = (fs[1].data,))
        @test fill_halos!((), bcs) === ()
        # a z phase restricted to k-planes would break the x, y, z corner order
        @test_throws ArgumentError BCm._fill_z!(BCm._Clip(fs[2].data, 3, 1:2), bcs.bottom, bcs.top,
                                                m.Hz, m.Nz, CellCenter(), m, fs[2].data)
        fill_halos!(fs, bcs); fill_halos!(fs, bcs)
        @test (@allocated fill_halos!(fs, bcs)) == 0      # below the threshold: the serial fills
    end

end
