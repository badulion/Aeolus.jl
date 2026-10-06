# KA-a (#294): Backends — the launch contract, the KA-ported set! /
# interior_map! kernels, and the deterministic reductions.
#
# Every check compares the serial reference (`CPUBackend(serial = true)`)
# with the threaded path forced on every launch (`threshold = 0`) and the
# default backend. Under `julia -t 1` the threaded path also runs
# serially, so the cross-thread comparison is the subprocess harness
# (test/thread_invariance.jl, AEOLUS_THREAD_TESTS=1); in-process these
# tests check the order and the bits against the old loops.

using Test
using Random: Xoshiro
using Aeolus
using Aeolus.Grids: interior, xnodes, ynodes, znodes
using Aeolus.Fields: set!, interior_view, location
using LinearAlgebra: BLAS

const BACKENDS = ("serial" => CPUBackend(serial = true),
                  "threaded, threshold 0" => CPUBackend(threshold = 0),
                  "default" => CPUBackend())

bits(A) = reinterpret(UInt64, vec(collect(A)))
same_bits(A, B) = size(A) == size(B) && bits(A) == bits(B)
same_bits(a::Real, b::Real) = reinterpret(UInt64, Float64(a)) == reinterpret(UInt64, Float64(b))

# --- the loops set! ran before the port (master at a5b78cd) -----------------
function old_set!(f, g::Function)
    loc = location(f); m = f.mesh
    xs = xnodes(m, loc); ys = ynodes(m, loc); zs = znodes(m, loc)
    ii, jj, kk = interior(m)
    @inbounds for (kk_, kv) in enumerate(kk), (jj_, jv) in enumerate(jj), (ii_, iv) in enumerate(ii)
        f.data[iv, jv, kv] = g(xs[ii_], ys[jj_], zs[kk_])
    end
    return f
end
function old_set!(f, value::Real)
    v = eltype(f)(value)
    ii, jj, kk = interior(f.mesh)
    @inbounds for k in kk, j in jj, i in ii
        f.data[i, j, k] = v
    end
    return f
end
function old_axpy!(y, a, x, (ii, jj, kk))
    @inbounds for k in kk, j in jj, i in ii
        y[i, j, k] = y[i, j, k] + a * x[i, j, k]
    end
    return y
end

# --- the canonical plane order, written out independently -------------------
function reference_plane_sum(A, ir, jr, k)
    l = zeros(8)
    nb = length(ir) ÷ 8
    for j in jr
        for b in 0:nb-1, r in 0:7
            l[r+1] += A[first(ir) + 8b + r, j, k]
        end
        for i in first(ir)+8nb:last(ir)
            l[1] += A[i, j, k]
        end
    end
    return ((l[1] + l[2]) + (l[3] + l[4])) + ((l[5] + l[6]) + (l[7] + l[8]))
end

@testset "Backends (KA-a, #294)" begin
    Hx = 3
    # odd sizes: rows that are not a multiple of the eight lanes
    mesh = RegularGrid(; size = (37, 29, 23), extent = (1.0, 2.0, 3.0), halo = (Hx, Hx, Hx))
    I3 = interior(mesh)

    @testset "set! is bit-identical to the old loops, halos untouched ($name)" for (name, b) in BACKENDS
        with_backend(b) do
            for loc in (CellCenter(), FaceX(), FaceY(), FaceZ())
                g(x, y, z) = sin(3x) * exp(-y) + z^2 / 7
                a = Field(loc, mesh); r = Field(loc, mesh)
                fill!(a.data, -1.0); fill!(r.data, -1.0)
                @test same_bits(set!(a, g).data, old_set!(r, g).data)
                @test same_bits(set!(a, π).data, old_set!(r, π).data)
                @test a.data[1, 1, 1] == -1.0            # halo untouched
            end
        end
    end

    @testset "interior_map! (axpy) is bit-identical to the old loop ($name)" for (name, b) in BACKENDS
        rng = Xoshiro(294)
        with_backend(b) do
            x = Field(mesh); y = Field(mesh)
            x.data .= randn(rng, size(x.data)); y.data .= randn(rng, size(y.data))
            yref = copy(y.data)
            a = 0.37
            interior_map!((yv, xv) -> yv + a * xv, y, y, x)
            @test same_bits(y.data, old_axpy!(yref, a, x.data, I3))
            # numbers are passed through, fields indexed
            z = Field(mesh)
            interior_map!(muladd, z, 2.0, x, 1.0)
            @test interior_view(z) == 2.0 .* interior_view(x) .+ 1.0
            @test z.data[1, 1, 1] == 0.0
        end
    end

    @testset "launch!: every storage-index range kind, each point once ($name)" for (name, b) in BACKENDS
        Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
        i2 = Hx + Nx
        ranges3 = ("interior" => I3,
                   "face (+1)" => (Hx+1:i2+1, Hx+1:Hx+Ny, Hx+1:Hx+Nz),
                   "extended (Hx:i2)" => (Hx:i2, Hx:Hx+Ny, Hx:Hx+Nz),
                   "shell (west halo)" => (1:Hx, 1:Ny+2Hx, 1:Nz+2Hx),
                   "empty" => (Hx+1:Hx, I3[2], I3[3]))
        count!(i, j, k, A) = (@inbounds A[i, j, k] += 1; nothing)
        for (what, r) in ranges3
            A = zeros(Int, size(Field(mesh).data))
            launch!(b, count!, r, A)
            ref = zeros(Int, size(A)); ref[r...] .= 1
            @test A == ref
        end
        # 2-D (i, j) surface range and 1-D ids
        S = zeros(Nx + 2Hx, Ny + 2Hx)
        launch!(b, (i, j, S) -> (@inbounds S[i, j] = 1000i + j; nothing), (Hx+1:i2, Hx+1:Hx+Ny), S)
        @test all(S[i, j] == 1000i + j for i in Hx+1:i2, j in Hx+1:Hx+Ny)
        @test sum(S .!= 0) == Nx * Ny
        v = zeros(5000)
        launch!(b, (n, v) -> (@inbounds v[n] = 2n; nothing), (17:4321,), v)
        @test v[17:4321] == 2 .* (17:4321) && count(!iszero, v) == 4321 - 16
    end

    @testset "launch_weighted!: column bodies, each index once, same bits ($name)" for (name, b) in BACKENDS
        # KA-h (#298): a body that loops over a column itself; the weight
        # only changes how the work is split, never the result
        Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
        rng = Xoshiro(298)
        X = randn(rng, Nx, Ny, Nz)
        col!(i, j, Y, X, Nz) = (@inbounds begin
            s = 0.0
            for k in 1:Nz
                s = 0.5 * s + X[i, j, k]; Y[i, j, k] = s
            end
        end; nothing)
        ref = similar(X)
        for j in 1:Ny, i in 1:Nx
            col!(i, j, ref, X, Nz)
        end
        for w in (1, Nz, 10^6)
            Y = zeros(Nx, Ny, Nz)
            launch_weighted!(b, w, col!, (1:Nx, 1:Ny), Y, X, Nz)
            @test same_bits(Y, ref)
            v = zeros(Int, Ny)
            launch_weighted!(b, w, (j, v) -> (@inbounds v[j] += 1; nothing), (1:Ny,), v)
            @test all(==(1), v)
            C = zeros(Int, Nx, Ny, 2)
            launch_weighted!(b, w, (i, j, k, C) -> (@inbounds C[i, j, k] += 1; nothing),
                             (1:Nx, 1:Ny, 1:2), C)
            @test all(==(1), C)
        end
        @test_throws ErrorException launch_weighted!(GPUBackend(), 4, (j,) -> nothing, (1:3,))
    end

    rng = Xoshiro(91)
    # magnitudes spread over four decades, so the order of the additions shows
    # (`rand(rng, dims...)`: `rand(rng, dims)` would draw one element of the
    # tuple, and 10^(4 n) overflows Int to 0, leaving A ≡ ±0)
    A = randn(rng, size(Field(mesh).data)) .* 10 .^ (4 .* rand(rng, size(Field(mesh).data)...))
    @assert count(!iszero, A) == length(A)
    B = randn(rng, size(A))
    ir, jr, kr = I3

    @testset "reductions: canonical order, identical on every path" begin
        parts = [reference_plane_sum(A, ir, jr, k) for k in kr]
        refsum = foldl(+, parts)
        for (name, b) in BACKENDS
            p = zeros(length(kr))
            plane_partials!(p, b, identity, I3, A)
            @test same_bits(p, parts)                       # the documented order, bit for bit
            @test same_bits(interior_sum(b, A, I3), refsum)
            @test same_bits(interior_dot(b, A, B, I3), interior_dot(BACKENDS[1][2], A, B, I3))
            @test same_bits(interior_max(b, A, I3), maximum(view(A, I3...)))
            @test interior_all_finite(b, A, I3)
            prof = zeros(length(kr)); plane_mean!(prof, b, A, I3)
            @test same_bits(prof, parts ./ (length(ir) * length(jr)))
            C = column_partials(b, A, I3)
            @test same_bits(C, [foldl(+, A[i, j, k] for k in kr) for i in ir, j in jr])
            @test same_bits(ordered_sum(C), foldl(+, vec(C)))
        end
        # ... and close to the pairwise sum (a different order: a named
        # one-time change where the port switches a call site over)
        @test interior_sum(A, I3) ≈ sum(view(A, I3...)) rtol = 1e-12
        @test interior_dot(A, B, I3) ≈ sum(view(A, I3...) .* view(B, I3...)) rtol = 1e-12
        @test ordered_sum(Float64[]) === 0.0
        @test interior_max(A, (1:0, jr, kr)) == -Inf
    end

    # KA-b (#296): the indexed reductions every call site of the port uses
    @testset "plane_reduce! / range_max: canonical order, fused writes ($name)" for (name, b) in BACKENDS
        parts = [reference_plane_sum(A, ir, jr, k) for k in kr]
        p = zeros(length(kr))
        @test same_bits(plane_reduce!(p, b, (i, j, k, A) -> (@inbounds A[i, j, k]), I3, A), parts)
        # a stencil body: the same order as plane_partials! on the stencil values
        D = similar(A); fill!(D, 0)
        for k in kr, j in jr, i in ir
            D[i, j, k] = A[i+1, j, k] - A[i-1, j, k]
        end
        q = zeros(length(kr)); plane_partials!(q, b, identity, I3, D)
        @test same_bits(plane_reduce!(p, b, (i, j, k, A) -> (@inbounds A[i+1, j, k] - A[i-1, j, k]), I3, A), q)
        # a tuple body: each component in the canonical order (a masked sum and its count)
        M = A .> 0
        s, n = zeros(length(kr)), zeros(length(kr))
        plane_reduce!((s, n), b, (i, j, k, A, M) -> (@inbounds M[i, j, k] ? (A[i, j, k], 1.0) : (0.0, 0.0)),
                      I3, A, M)
        q2 = zeros(length(kr)); plane_partials!(q2, b, (a, m) -> m ? a : 0.0, I3, A, M)
        @test same_bits(s, q2)
        @test n == [count(view(M, ir, jr, k)) for k in kr]
        # a fused update: the body writes its point and returns what it added
        C = copy(A); dC = zeros(length(kr))
        plane_reduce!(dC, b, (i, j, k, C, f) -> (@inbounds (d = f * C[i, j, k]; C[i, j, k] += d; d)), I3, C, 0.25)
        @test same_bits(view(C, I3...), view(A, I3...) .+ 0.25 .* view(A, I3...))
        q3 = zeros(length(kr)); plane_partials!(q3, b, a -> 0.25 * a, I3, A)
        @test same_bits(dC, q3)
        # interior-sized views with axes ranges: the same bits as the halo arrays
        V = view(A, I3...)
        @test same_bits(plane_reduce!(p, b, (i, j, k, V) -> (@inbounds V[i, j, k]), axes(V), V), parts)
        # range_max: exact, so equal to Base's maximum; NaN-propagating
        @test same_bits(range_max(b, (i, j, k, A) -> abs(A[i, j, k]), I3, A), maximum(abs, view(A, I3...)))
        C = copy(A); C[ir[3], jr[2], kr[4]] = NaN
        @test isnan(range_max(b, (i, j, k, A) -> abs(A[i, j, k]), I3, C))
        @test range_max(b, (i, j, k, A) -> A[i, j, k], (ir, 1:0, kr), A) == -Inf
        # the result type is inferred, never found by evaluating the body an extra time
        @test range_max(b, (i, j, k, A) -> Float32(A[i, j, k]), I3, A) isa Float32
        @test range_max(b, (i, j, k, A) -> (@inbounds A[i, j, k] > 0 ? 1 : 0), I3, A) === 1.0
        # plane_map!: a whole-plane function, one work item per plane, its own order;
        # it may write its plane (here a fused update in cell order, i fastest)
        function cell_order_plane(k, ir, jr, C, f)
            s = 0.0
            @inbounds for j in jr, i in ir
                d = f * C[i, j, k]; C[i, j, k] += d; s += d
            end
            return s
        end
        C = copy(A); dC = zeros(length(kr))
        @test plane_map!(dC, b, cell_order_plane, I3, C, 0.25) === dC
        @test same_bits(view(C, I3...), view(A, I3...) .+ 0.25 .* view(A, I3...))
        ref = similar(dC); Cr = copy(A)
        for (c, k) in enumerate(kr)
            ref[c] = cell_order_plane(k, ir, jr, Cr, 0.25)
        end
        @test same_bits(dC, ref)
        @test same_bits(plane_map!(zeros(length(kr)), b, (k, ir, jr, A) -> sum(view(A, ir, jr, k)), I3, A),
                        [sum(view(A, ir, jr, k)) for k in kr])
    end
    @testset "range_max evaluates the body once per point" begin
        calls = Ref(0)
        range_max(CPUBackend(serial = true), (i, j, k, A, n) -> (n[] += 1; A[i, j, k]), I3, A, calls)
        @test calls[] == prod(length, I3)
    end

    @testset "max propagates NaN, all_finite catches NaN and ±Inf ($name)" for (name, b) in BACKENDS
        for (bad, maxwant) in ((NaN, NaN), (Inf, Inf), (-Inf, maximum(view(A, I3...))))
            C = copy(A); C[ir[5], jr[7], kr[end]] = bad
            m = interior_max(b, C, I3)
            @test isequal(m, maxwant)
            @test !interior_all_finite(b, C, I3)
        end
        C = copy(A); C[1, 1, 1] = NaN                      # halo NaN is outside the region
        @test interior_all_finite(b, C, I3) && !isnan(interior_max(b, C, I3))
        @test interior_max(b, fill(-Inf, size(A)), I3) == -Inf
    end

    @testset "field-level reductions and a physical profile" begin
        θ = Field(mesh)
        set!(θ, (x, y, z) -> 300 + 0.01 * z)                # stable layer, 10 K/km
        prof = zeros(mesh.Nz)
        plane_mean!(prof, θ)
        zs = znodes(mesh, CellCenter())
        @test prof ≈ 300 .+ 0.01 .* zs rtol = 1e-14
        # against the exact (BigFloat) sums: the eight-lane order is accurate
        # to ~3e-16 here, Base's sum(abs2, view) only to 2.9e-13
        v = interior_view(θ)
        @test interior_sum(θ) ≈ Float64(sum(big.(v))) rtol = 1e-14
        @test interior_dot(θ, θ) ≈ Float64(sum(big.(v) .^ 2)) rtol = 1e-14
        @test interior_max(θ) == maximum(interior_view(θ))
        @test interior_all_finite(θ)
    end

    @testset "bounds are checked once per call" begin
        @test_throws BoundsError interior_sum(A, (ir, jr, 1:size(A, 3)+1))
        @test_throws BoundsError column_partials(A, (0:3, jr, kr))
        small = RegularGrid(; size = (4, 4, 4), extent = (1.0, 1.0, 1.0), halo = (1, 1, 1))
        @test_throws DimensionMismatch interior_map!(+, Field(mesh), Field(small))
    end

    @testset "backend selection, allocation, thread policy" begin
        @test default_backend() isa CPUBackend
        sb = CPUBackend(serial = true)
        @test runs_serially(sb) && runs_serially(CPUBackend(), 10)
        @test runs_serially(CPUBackend(threshold = 0)) == (Threads.nthreads() == 1)
        with_backend(sb) do
            @test default_backend() === sb
            f = Field(CellCenter(), mesh)
            @test backend(f) === sb && f.data isa Array{Float64,3}
        end
        @test default_backend() == CPUBackend()             # restored
        @test_throws ErrorException with_backend(() -> error("boom"), sb)
        @test default_backend() == CPUBackend()             # restored on error too
        # the CPU path is already concretely typed: a launch on host storage
        # finds its backend without dynamic dispatch or allocation (#294, #342)
        @test @inferred(backend(zeros(2, 2, 2))) isa CPUBackend
        @test @inferred(backend(view(zeros(2, 2, 2), :, :, 1))) isa CPUBackend
        @test @inferred(backend(Field(mesh))) isa CPUBackend
        f = Field(mesh; backend = CPUBackend())
        @test size(f.data) == (37 + 6, 29 + 6, 23 + 6) && all(iszero, f.data)
        @test_throws ErrorException launch!(GPUBackend(), (i, j, k) -> nothing, I3)
        @test_throws ErrorException Field(mesh; backend = GPUBackend(:rocm))

        fftw0 = Aeolus.fftw_policy().threads
        blas0 = BLAS.get_num_threads()
        st = Aeolus.Operators._FFTW_POLICY
        explicit0 = st.threads_explicit
        nfftw() = Aeolus.Operators.FFTW.get_num_threads()
        # KA-h (#298): with no explicit count, FFTW runs on Julia's threads.
        # (Until KA-h the expectation was "FFTW policy kept at 1"; changed by
        # the maintainer decision on #291 once the FFT solves were shown to be
        # thread-invariant.)
        st.threads_explicit = false
        r = init_threads!(CPUBackend())
        @test r.backend == CPUBackend() && r.julia_threads == Threads.nthreads()
        Threads.nthreads() > 1 && @test BLAS.get_num_threads() == 1   # pinned when threaded
        @test r.fftw_threads == Threads.nthreads() == Aeolus.fftw_policy().threads == nfftw()
        @test !st.threads_explicit                                    # still a default
        # an explicit count (set_fftw_policy!, AEOLUS_FFTW_THREADS) is kept
        Aeolus.set_fftw_policy!(threads = 3)
        @test init_threads!(CPUBackend()).fftw_threads == 3 == nfftw()
        r2 = init_threads!(CPUBackend(); fftw_threads = 2)
        @test r2.fftw_threads == 2 == Aeolus.fftw_policy().threads == nfftw()  # through the policy
        @test init_threads!(CPUBackend()).fftw_threads == 2
        init_threads!(CPUBackend(); fftw_threads = fftw0, blas_threads = blas0)
        st.threads_explicit = explicit0
        @test Aeolus.fftw_policy().threads == fftw0 && BLAS.get_num_threads() == blas0
    end
end
