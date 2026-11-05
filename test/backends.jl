# KA-a (#294) and the ports after it: Backends — the launch contract, the
# KA-ported set! / interior_map! kernels, the canonical reductions, the
# column primitives (KA-m #301), the one preconditioned-CG driver pcg!
# (KA-g #300) and the counter-based random streams (KA-n #305).
#
# The cross-thread comparison is the subprocess harness
# (test/thread_invariance.jl, AEOLUS_THREAD_TESTS=1); in-process these tests
# check the contracts, the documented summation order and the results
# against the old loops and against dense / known-answer references.

using Test
using Random: Xoshiro
using Aeolus
using Aeolus.Grids: interior, xnodes, ynodes, znodes
using Aeolus.Fields: set!, interior_view, location

const PSKb = Aeolus.PressureSolvers

# --- the loop set! ran before the port (master at a5b78cd) ------------------
function old_set!(f, g::Function)
    loc = location(f); m = f.mesh
    xs = xnodes(m, loc); ys = ynodes(m, loc); zs = znodes(m, loc)
    ii, jj, kk = interior(m)
    @inbounds for (kk_, kv) in enumerate(kk), (jj_, jv) in enumerate(jj), (ii_, iv) in enumerate(ii)
        f.data[iv, jv, kv] = g(xs[ii_], ys[jj_], zs[kk_])
    end
    return f
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

# (I·σ − ∇²) with a Neumann mirror on a unit-halo array, and its dense matrix
@inline _shifted_at(i, j, k, u, σ) =
    @inbounds σ * u[i, j, k] - (u[i+1, j, k] + u[i-1, j, k] + u[i, j+1, k] + u[i, j-1, k] +
                                u[i, j, k+1] + u[i, j, k-1] - 6u[i, j, k])
function dense_shifted(n, σ)
    A = zeros(n^3, n^3)
    lin(i, j, k) = i + n * (j - 1) + n^2 * (k - 1)
    for k in 1:n, j in 1:n, i in 1:n
        r = lin(i, j, k)
        A[r, r] += σ
        for (di, dj, dk) in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
            A[r, r] += 1                      # −(u_nb − u): the mirror ghost
            A[r, lin(clamp(i + di, 1, n), clamp(j + dj, 1, n), clamp(k + dk, 1, n))] -= 1
        end
    end
    return A
end

@testset "Backends (KA-a, #294)" begin
    Hx = 3
    # odd sizes: rows that are not a multiple of the eight lanes
    mesh = RegularGrid(; size = (37, 29, 23), extent = (1.0, 2.0, 3.0), halo = (Hx, Hx, Hx))
    I3 = interior(mesh)
    b = CPUBackend(threshold = 0)            # every launch on the KA path

    @testset "set! and interior_map! are the old loops; halos untouched" begin
        with_backend(b) do
            g(x, y, z) = sin(3x) * exp(-y) + z^2 / 7
            a = Field(FaceX(), mesh); r = Field(FaceX(), mesh)
            fill!(a.data, -1.0); fill!(r.data, -1.0)
            @test agree(set!(a, g).data, old_set!(r, g).data) && a.data[1, 1, 1] == -1.0
            x = Field(mesh); x.data .= randn(Xoshiro(294), size(x.data))
            z = Field(mesh)
            interior_map!(muladd, z, 2.0, x, 1.0)        # numbers passed through, fields indexed
            @test agree(interior_view(z), 2.0 .* interior_view(x) .+ 1.0) && z.data[1, 1, 1] == 0.0
        end
    end

    @testset "launch!: every storage-index range kind, each point once" begin
        Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
        i2 = Hx + Nx
        count!(i, j, k, A) = (@inbounds A[i, j, k] += 1; nothing)
        ok = all(("interior" => I3, "face (+1)" => (Hx+1:i2+1, Hx+1:Hx+Ny, Hx+1:Hx+Nz),
                  "shell (west halo)" => (1:Hx, 1:Ny+2Hx, 1:Nz+2Hx), "empty" => (Hx+1:Hx, I3[2], I3[3]))) do (_, r)
            A = zeros(Int, size(Field(mesh).data))
            launch!(b, count!, r, A)
            ref = zeros(Int, size(A)); ref[r...] .= 1
            A == ref
        end
        @test ok
        v = zeros(5000)
        launch!(b, (n, v) -> (@inbounds v[n] = 2n; nothing), (17:4321,), v)
        @test v[17:4321] == 2 .* (17:4321) && count(!iszero, v) == 4321 - 16
        @test_throws ErrorException launch!(GPUBackend(), (i, j, k) -> nothing, I3)
    end

    rng = Xoshiro(91)
    # magnitudes spread over four decades, so the order of the additions shows
    A = randn(rng, size(Field(mesh).data)) .* 10 .^ (4 .* rand(rng, size(Field(mesh).data)...))
    B = randn(rng, size(A))
    ir, jr, kr = I3

    @testset "reductions: the canonical order; close to the pairwise sum" begin
        parts = [reference_plane_sum(A, ir, jr, k) for k in kr]
        p = zeros(length(kr))
        plane_partials!(p, b, identity, I3, A)
        @test agree(p, parts)                                # the documented order
        @test agree(interior_sum(b, A, I3), foldl(+, parts))
        @test agree(interior_max(b, A, I3), maximum(view(A, I3...)))
        C = column_partials(b, A, I3)
        @test agree(C, [foldl(+, A[i, j, k] for k in kr) for i in ir, j in jr])
        @test interior_sum(A, I3) ≈ sum(view(A, I3...)) rtol = 1e-12
        @test interior_dot(A, B, I3) ≈ sum(view(A, I3...) .* view(B, I3...)) rtol = 1e-12
        # a fused update: the body writes its point and returns what it added
        Cf = copy(A); dC = zeros(length(kr))
        plane_reduce!(dC, b, (i, j, k, C, f) -> (@inbounds (d = f * C[i, j, k]; C[i, j, k] += d; d)), I3, Cf, 0.25)
        q3 = zeros(length(kr)); plane_partials!(q3, b, a -> 0.25 * a, I3, A)
        @test agree(view(Cf, I3...), 1.25 .* view(A, I3...)) && agree(dC, q3)
        # against the exact (BigFloat) sum: the eight-lane order is accurate
        θ = Field(mesh); set!(θ, (x, y, z) -> 300 + 0.01 * z)
        @test interior_sum(θ) ≈ Float64(sum(big.(interior_view(θ)))) rtol = 1e-14
    end

    @testset "max propagates NaN, all_finite catches NaN and ±Inf; halos outside" begin
        ok = all(((NaN, NaN), (Inf, Inf), (-Inf, maximum(view(A, I3...))))) do (bad, maxwant)
            C = copy(A); C[ir[5], jr[7], kr[end]] = bad
            isequal(interior_max(b, C, I3), maxwant) && !interior_all_finite(b, C, I3)
        end
        @test ok
        C = copy(A); C[1, 1, 1] = NaN
        @test interior_all_finite(b, C, I3) && !isnan(interior_max(b, C, I3))
        @test_throws BoundsError interior_sum(A, (ir, jr, 1:size(A, 3)+1))
    end

    @testset "foreach_column! / column_sum: each column once, column-order fold" begin
        X = randn(Xoshiro(301), 13, 11, 7)
        cr, crj = 2:12, 3:10
        colsum(i, j, A) = (s = 0.0; @inbounds(for k in axes(A, 3); s += A[i, j, k]; end); s)
        out = zeros(13, 11)
        foreach_column!(b, (i, j, out, A) -> (@inbounds out[i, j] = colsum(i, j, A); nothing), (cr, crj), out, X; weight = 7)
        @test out[cr, crj] == [colsum(i, j, X) for i in cr, j in crj] && all(iszero, out[1, :])
        ref = 0.0
        for j in crj, i in cr; ref += colsum(i, j, X); end
        @test agree(column_sum(b, colsum, (cr, crj), X; weight = 7), ref)
    end

    @testset "pcg! solves an SPD system (dense reference); breakdown and zero rhs" begin
        n, σ = 6, 0.3
        Ad = dense_shifted(n, σ)
        bi = randn(Xoshiro(1), n, n, n)
        xref = reshape(Ad \ vec(bi), n, n, n)
        rhs = zeros(n + 2, n + 2, n + 2); rhs[2:n+1, 2:n+1, 2:n+1] .= bi
        x = zero(rhs); R = (2:n+1, 2:n+1, 2:n+1)
        ws = PCGWorkspace(rhs; ranges = R)
        op = PointOperator(_shifted_at, σ; prepare = a -> PSKb.fill_unit_halo!(a, :neumann, :neumann, :neumann))
        st = pcg!(op, IdentityPreconditioner(), x, rhs, ws; ranges = R, tol = 1e-13, maxit = 500)
        @test st.status === :converged && maximum(abs, x[R...] .- xref) < 1e-11 * maximum(abs, xref)
        st0 = pcg!(op, IdentityPreconditioner(), x, zero(rhs), ws; ranges = R, tol = 1e-10, maxit = 10)
        @test st0.status === :zero_rhs && st0.iterations == 0 && all(==(0), x[R...])
        neg = PointOperator((i, j, k, u) -> @inbounds(-u[i, j, k]))
        stb = pcg!(neg, IdentityPreconditioner(), x, rhs, ws; ranges = R, tol = 1e-10, maxit = 10)
        @test stb.status === :breakdown && stb.curvature ≈ -sum(abs2, bi)   # p = b: p·Ap = −‖b‖²
    end

    @testset "counter-based random streams: Philox4x32-10 known answers, pure function of the id" begin
        φ = Aeolus.Backends.philox4x32
        z = UInt32(0); f = typemax(UInt32)
        # Random123 known-answer vectors (kat_vectors, philox4x32_10)
        @test φ((z, z, z, z), (z, z)) == (0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8)
        @test φ((f, f, f, f), (f, f)) == (0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd)
        @test φ((0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344), (0xa4093822, 0x299f31d0)) ==
              (0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1)
        a = [rand(counter_rng(42, 3, 7)) for _ in 1:1]
        r2 = counter_rng(1, 0, 0); set_stream!(r2, 42, 3, 7)
        @test rand(r2) == a[1] != rand(counter_rng(42, 3, 8))       # re-pointed = fresh; ids differ
        # first draws of 2·10^4 neighbouring streams: uniform mean (4σ)
        xs = [rand(counter_rng(2026, 0, i)) for i in 0:19_999]
        @test abs(sum(xs) / length(xs) - 0.5) < 4 * sqrt(1 / 12 / length(xs))
    end

    @testset "backend selection and restore" begin
        sb = CPUBackend(serial = true)
        with_backend(sb) do
            @test default_backend() === sb && backend(Field(CellCenter(), mesh)) === sb
        end
        @test default_backend() == CPUBackend()             # restored
        @test_throws ErrorException with_backend(() -> error("boom"), sb)
        @test default_backend() == CPUBackend()             # restored on error too
        @test @inferred(backend(zeros(2, 2, 2))) isa CPUBackend
    end

    @testset "init_threads!: FFTW on the policy's threads, an explicit count kept" begin
        fftw0 = Aeolus.fftw_policy().threads
        st = Aeolus.Operators._FFTW_POLICY
        explicit0 = st.threads_explicit
        nfftw() = Aeolus.Operators.FFTW.get_num_threads()
        try
            # with no explicit count FFTW runs on Julia's threads (maintainer decision on #291)
            st.threads_explicit = false
            r = init_threads!(CPUBackend())
            @test r.julia_threads == Threads.nthreads() && r.fftw_threads == Threads.nthreads() == nfftw()
            @test init_threads!(CPUBackend(); fftw_threads = 2).fftw_threads == 2 == nfftw()
            @test init_threads!(CPUBackend()).fftw_threads == 2         # the explicit count is kept
        finally
            init_threads!(CPUBackend(); fftw_threads = fftw0)
            st.threads_explicit = explicit0
        end
        @test Aeolus.fftw_policy().threads == fftw0
    end
end
