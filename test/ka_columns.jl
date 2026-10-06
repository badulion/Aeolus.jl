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
include(joinpath(@__DIR__, "support", "legacy_columns.jl"))
const LC = LegacyColumns

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

@testset "global caches: one Gauss-Legendre source, safe first use under threads" begin
    RC = Aeolus.RadiationCore
    # one source: SurfaceOptics and Radiosity return the very same nodes
    for n in (1, 2, 5, 8, 16, 32, 64, 65, 120)
        x, w = RC._gauss_legendre(n)
        @test Aeolus.SurfaceOptics._gauss_legendre(n) === (x, w)
        @test Aeolus.Radiosity._gauss_nodes(n) === RC._gauss_legendre(n)
        @test kc_bits(x) == kc_bits(RC._gauss_legendre_nodes(n)[1]) &&
              kc_bits(w) == kc_bits(RC._gauss_legendre_nodes(n)[2])
        @test sum(w) ≈ 2 && issorted(x; rev = true)
    end
    @test_throws ErrorException RC._gauss_legendre(0)
    # orders ≤ 64 are a table filled at load; higher ones are memoised under
    # a lock: a first use from many tasks at once gives the serial nodes
    empty!(RC._GL_CACHE)
    ref = Dict(n => RC._gauss_legendre_nodes(n) for n in 60:110)
    empty!(RC._GL_CACHE)
    got = Vector{Any}(undef, 400)
    @sync for t in 1:400
        Threads.@spawn (got[t] = RC._gauss_legendre(60 + (t % 51)))
    end
    @test all(t -> got[t] == ref[60 + (t % 51)], 1:400)
    @test all(t -> got[t] === RC._gauss_legendre(60 + (t % 51)), 1:400)
    # the aerosol bulk-optics memo (locked)
    AO = Aeolus.AerosolOptics
    modes = [lognormal_mode(s, 100.0, 0.05 + 0.01i, 1.6) for (i, s) in enumerate((:sulphate, :sea_salt, :dust))]
    AO.clear_optics_cache!()
    serial = [mode_optics(m, λ; rh = rh) for m in modes, λ in (0.4, 0.55, 1.0), rh in (0.0, 0.8)]
    AO.clear_optics_cache!()
    res = Array{Any}(undef, size(serial)..., 4)
    @sync for r in 1:4, I in CartesianIndices(serial)
        m, λ, rh = modes[I[1]], (0.4, 0.55, 1.0)[I[2]], (0.0, 0.8)[I[3]]
        Threads.@spawn (res[I, r] = mode_optics(m, λ; rh = rh))
    end
    @test all(r -> res[:, :, :, r] == serial, 1:4)
    # the SNICAR ageing table: read once, the same object for every caller
    SP = Aeolus.Snowpack
    SP._SNICAR_TABLE[] = nothing
    tabs = Vector{Any}(undef, 16)
    @sync for t in 1:16
        Threads.@spawn (tabs[t] = snicar_aging_table())
    end
    @test all(t -> tabs[t] === tabs[1], 1:16) && tabs[1] === snicar_aging_table()
end

@testset "RadiationFog: fog_adjust! per plane with the early exit" begin
    MT = Aeolus.MoistThermodynamics
    n = (24, 20, 12)
    rng = Xoshiro(2026)
    p = collect(range(1.0e5, 8.5e4, length = n[3]))
    Π = [MT.exner_function(pk, MT.PhasePartition(0.0)) for pk in p]
    θ0 = 285 .+ 6 .* rand(rng, n...)
    qs = [MT.saturation_specific_humidity(θ0[i, j, k] * Π[k], p[k]) for i in 1:n[1], j in 1:n[2], k in 1:n[3]]
    # a mix: subsaturated dry cells, cells just below and just above
    # saturation (the early exit's margin), supersaturated cells, fog cells
    # that evaporate, and a few negative values
    f = rand(rng, n...)
    qv0 = qs .* ifelse.(f .< 0.4, 0.7 .+ 0.25 .* rand(rng, n...),
                 ifelse.(f .< 0.5, 1 .- 1e-9 .* rand(rng, n...),
                 ifelse.(f .< 0.6, 1 .- 2e-8 .* rand(rng, n...), 1 .+ 0.02 .* rand(rng, n...))))
    ql0 = ifelse.(rand(rng, n...) .< 0.3, 2e-4 .* rand(rng, n...), 0.0)
    ql0[1:3, 1, 1] .= -1e-12; qv0[4, 1, 1] = -1e-9
    θr, qvr, qlr = copy(θ0), copy(qv0), copy(ql0)
    totr = LC.fog_adjust!(θr, qvr, qlr, p, 2.0)
    @test count(!=(0), qlr .- ql0) > 300          # the test exercises the adjustment
    for (lab, b) in KC_BACKENDS, τ in (0.0, 30.0)
        θa, qva, qla = copy(θ0), copy(qv0), copy(ql0)
        θb, qvb, qlb = copy(θ0), copy(qv0), copy(ql0)
        tb = LC.fog_adjust!(θb, qvb, qlb, p, 2.0; τ = τ)
        ta = fog_adjust!(θa, qva, qla, p, 2.0; τ = τ, backend = b)
        @test kc_bits(θa) == kc_bits(θb) && kc_bits(qva) == kc_bits(qvb) && kc_bits(qla) == kc_bits(qlb)
        # the total: plane sums in cell order, then ascending k (a named
        # one-time change of its last bits against the one running sum)
        @test isapprox(ta, tb; rtol = 1e-12)
        parts = [LC.fog_adjust!(copy(θ0), copy(qv0), copy(ql0), p, 2.0; τ = τ, levels = k:k) for k in 1:n[3]]
        @test kc_bits(ta) == kc_bits(ordered_sum(parts))
        # views (the Berlin coupling passes interior views), a scalar pressure
        H = 2
        big = [fill(NaN, n .+ 2H) for _ in 1:3]
        for (B, A) in zip(big, (θ0, qv0, ql0)); B[H+1:H+n[1], H+1:H+n[2], H+1:H+n[3]] .= A; end
        vs = [view(B, H+1:H+n[1], H+1:H+n[2], H+1:H+n[3]) for B in big]
        θc, qvc, qlc = copy(θ0), copy(qv0), copy(ql0)
        LC.fog_adjust!(θc, qvc, qlc, 9.5e4, 2.0; τ = τ)
        fog_adjust!(vs..., 9.5e4, 2.0; τ = τ, backend = b)
        @test kc_bits(vs[1]) == kc_bits(θc) && kc_bits(vs[2]) == kc_bits(qvc) && kc_bits(vs[3]) == kc_bits(qlc)
        @test all(isnan, big[1][1:H, :, :])
    end
end

end # testset KA-m
