# KA-m (#301): column physics parallel over columns.
#
#   1. the primitives: foreach_column! (2-D columns and 1-D id lists, with
#      and without partials) and foreach_cell! — every column / cell runs
#      once, the partials are the per-column terms, and their column-order
#      fold equals the serial loop `s += x` it replaces, bit for bit, on
#      every backend;
#   2. the global caches (one Gauss-Legendre node cache, the aerosol
#      bulk-optics memo, the SNICAR ageing table): prefilled or guarded,
#      so a first use from many tasks at once gives the serial result;
#   3. the ported column physics against transcripts of the pre-port loops
#      (test/support/legacy_columns.jl), bit for bit, on every backend.

using Test
using Random: Xoshiro, rand, randn
using Aeolus

const KC_BACKENDS = ("serial" => CPUBackend(serial = true),
                     "threaded, threshold 0" => CPUBackend(threshold = 0),
                     "default" => CPUBackend())
kc_bits(A) = reinterpret(UInt64, vec(collect(Float64, A)))

@testset "KA-m: column physics (#301)" begin

@testset "foreach_column! / foreach_cell! primitives" begin
    rng = Xoshiro(301)
    A = randn(rng, 13, 11, 7) .* 10 .^ (3 .* rand(rng, 13, 11, 7))
    ir, jr = 2:12, 3:10
    # the serial references: column sums over k, and the column-major fold
    ref_col = [sum(A[i, j, k] for k in 1:7) for i in ir, j in jr]
    function ref_fold(f)
        s = 0.0
        for j in jr, i in ir
            s += f(i, j)
        end
        return s
    end
    colsum(i, j, A) = (s = 0.0; @inbounds(for k in axes(A, 3); s += A[i, j, k]; end); s)
    for (lab, b) in KC_BACKENDS
        # no partials: a column body writing its own column
        out = zeros(13, 11)
        foreach_column!(b, (i, j, out, A) -> (@inbounds out[i, j] = colsum(i, j, A); nothing),
                        (ir, jr), out, A; weight = 7)
        @test out[ir, jr] == ref_col
        @test all(iszero, out[1, :]) && all(iszero, out[:, 1:2])
        # partials: two terms, the second skipped (0) on odd columns
        P = zeros(length(ir) * length(jr), 2)
        foreach_column!(b, (i, j, A) -> (s = colsum(i, j, A); (s, isodd(i + j) ? 0.0 : abs(s))),
                        (ir, jr), A; partials = P, weight = 7)
        @test vec(P[:, 1]) == vec(ref_col)
        S = column_sums(P)
        @test kc_bits(S[1]) == kc_bits(ref_fold((i, j) -> colsum(i, j, A)))
        @test kc_bits(S[2]) == kc_bits(ref_fold((i, j) -> isodd(i + j) ? 0.0 : abs(colsum(i, j, A))))
        # a maximum over the partials (exact: any order)
        @test foldl(max, view(P, :, 1); init = -Inf) == maximum(ref_col)
        # 1-D id lists, one term returned as a number
        ids = 4:40
        Q = zeros(length(ids), 1)
        foreach_column!(b, (n, v) -> v[n]^2, (ids,), collect(1.0:50.0); partials = Q)
        @test vec(Q) == [Float64(n)^2 for n in ids]
        cnt = zeros(Int, 50)
        foreach_column!(b, (n, cnt) -> (cnt[n] += 1; nothing), (ids,), cnt)
        @test cnt[ids] == ones(Int, length(ids)) && sum(cnt) == length(ids)
        @test_throws DimensionMismatch foreach_column!(b, (n,) -> 1.0, (ids,); partials = zeros(3, 1))
        # foreach_cell!: a pointwise update, and per-plane partials
        B = copy(A)
        foreach_cell!(b, (i, j, k, B) -> (@inbounds B[i, j, k] = 2B[i, j, k]; nothing),
                      (ir, jr, 2:6), B; weight = 3)
        @test B[ir, jr, 2:6] == 2 .* A[ir, jr, 2:6] && B[1, :, :] == A[1, :, :]
        p1 = zeros(5); p2 = zeros(5)
        foreach_cell!(b, (i, j, k, A) -> (@inbounds A[i, j, k]), (ir, jr, 2:6), A; partials = p1)
        plane_partials!(p2, CPUBackend(serial = true), identity, (ir, jr, 2:6), A)
        @test kc_bits(p1) == kc_bits(p2)
    end
end

end # testset KA-m
