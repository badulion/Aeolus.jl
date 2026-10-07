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
        # column_sum: the running sums serially, partials + fold threaded —
        # the same numbers in the same order
        cs = column_sum(b, (i, j, A) -> (s = colsum(i, j, A); (s, isodd(i + j) ? 0.0 : abs(s))),
                        (ir, jr), A; weight = 7)
        @test cs isa NTuple{2,Float64} && kc_bits(collect(cs)) == kc_bits(S)
        c1 = column_sum(b, colsum, (ir, jr), A; weight = 10^6)
        @test kc_bits(c1) == kc_bits(S[1])
        @test column_sum(b, (n, v) -> v[n], (3:9,), collect(1.0:10.0)) === sum(3.0:9.0)
        # a body whose type is not inferred still sums from zero (−0.0 + 0.0 = 0.0)
        anyv = Any[-0.0, -0.0, 1.5]
        @test kc_bits(column_sum(b, (n, v) -> v[n], (1:3,), anyv)) == kc_bits(0.0 + -0.0 + -0.0 + 1.5)
        @test kc_bits(column_sum(b, (n, v) -> v[n], (1:2,), anyv)) == kc_bits(0.0)
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
    # the fold of the partials is one register pass: no allocation per row
    # (a boxed accumulator allocated 72 bytes per row: 5.5 ms for 128² rows)
    P = randn(rng, 4096, 3)
    column_sums(P)
    @test (@allocated Aeolus.Backends._column_sums(P, Val(3))) < 1024
    @test kc_bits(collect(column_sums(P))) == kc_bits([foldl(+, P[:, t]; init = 0.0) for t in 1:3])
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

@testset "Microphysics: Kessler conversion per cell, sedimentation per column" begin
    mesh = RegularGrid(; size = (16, 12, 20), extent = (800.0, 600.0, 1000.0), halo = (2, 2, 2))
    rng = Xoshiro(44)
    mk() = (f = Aeolus.Fields.Field(mesh); f)
    qv0, ql0, qr0 = mk(), mk(), mk()
    I = Aeolus.Grids.interior(mesh)
    qv0.data .= 0.01
    ql0.data[I...] .= ifelse.(rand(rng, 16, 12, 20) .< 0.3, 2e-3 .* rand(rng, 16, 12, 20), 0.0)
    qr0.data[I...] .= ifelse.(rand(rng, 16, 12, 20) .< 0.2, 1e-3 .* rand(rng, 16, 12, 20), 0.0)
    qr0.data[I[1], I[2][1:3], I[3]] .= 0.0                # dry columns
    ρ = collect(range(1.2, 0.9, length = 20))
    for (lab, b) in KC_BACKENDS, Vt in (:spectrum, 3.0), ρa in (1.1, ρ)
        a = [copy(f.data) for f in (qv0, ql0, qr0)]
        fa = [mk() for _ in 1:3]; for (f, d) in zip(fa, a); f.data .= d; end
        fb = [mk() for _ in 1:3]; for (f, d) in zip(fb, a); f.data .= d; end
        LC.apply_kessler_step!(fb..., 20.0; Vt = Vt, ρ_air = ρa)
        apply_kessler_step!(fa..., 20.0; Vt = Vt, ρ_air = ρa, backend = b)
        @test all(n -> kc_bits(fa[n].data) == kc_bits(fb[n].data), 1:3)
        @test fa[3].data != qr0.data                       # rain was made and fell
    end
end

@testset "Chemistry: dry deposition per column, total in column order" begin
    rng = Xoshiro(9)
    c0 = 10 .+ rand(rng, 30, 20, 8); vd = 0.01 .* rand(rng, 30, 20); kk = rand(rng, 0:9, 30, 20)
    for (lab, b) in KC_BACKENDS, (v, k) in ((vd, kk), (0.004, 2), (vd, 1))
        ca, cb = copy(c0), copy(c0)
        rb = LC.apply_dry_deposition!(cb, v, 30.0; dz = 2.0, k = k)
        ra = Aeolus.Chemistry.apply_dry_deposition!(ca, v, 30.0; dz = 2.0, k = k, backend = b)
        @test kc_bits(ca) == kc_bits(cb) && kc_bits(ra) == kc_bits(rb)
    end
end

@testset "UrbanSurface: facet column steps, one work item per facet" begin
    hm = zeros(8, 6); hm[3:4, :] .= 12; hm[6, 2:5] .= 20
    sc = Aeolus.Radiosity.build_scene(hm, 4.0, 4.0; wall_subdivisions = 2)
    vf = Aeolus.Radiosity.view_factors(sc)
    air = SurfaceForcing(sw_down = 500.0, lw_down = 330.0, T_air = 295.0, U = 3.0, q_air = 0.008)
    function run(b, kw)
        u = UrbanFacetScene(sc, vf; T = 290.0, options = f -> (T_interior = 293.0,))
        fl = with_backend(b) do
            local fl
            for _ in 1:3
                fl = step_urban_facets!(u, air, 600.0; kw...)
            end
            fl
        end
        return vcat(facet_temperatures(u), [f.H for f in fl], [f.G for f in fl],
                    reduce(vcat, [f.column.T for f in u.facets]))
    end
    for kw in ((;), (solver = :direct,), (coupling = :lagged,))
        ref = run(CPUBackend(serial = true), kw)
        @test kc_bits(run(CPUBackend(threshold = 0), kw)) == kc_bits(ref)
        @test kc_bits(run(CPUBackend(), kw)) == kc_bits(ref)
    end
    # a facet listed twice would be stepped twice (and by two work items)
    u = UrbanFacetScene(sc, vf; T = 290.0)
    @test_throws ArgumentError UrbanFacetScene(sc, vf, vcat(u.facets[1:end-1], u.facets[1:1]))
end

@testset "WallModels: dynamic roughness per column" begin
    WM = Aeolus.WallModels
    Nx, Ny, r = 32, 8, 4
    dx = 2π / Nx
    xf = [(i - 0.5) * dx / r for i in 1:Nx*r]
    hf = [0.02 * sin(3x) + 0.006 * sin(10x) + 0.004 * cos(2y) for x in xf, y in xf[1:Ny*r]]
    st = WM.subfilter_height(hf, r; dx = dx, dy = dx, filter = :box)
    uc = 3 .+ 0.3 .* randn(Xoshiro(12), Nx, Ny); vc = 0.3 .* randn(Xoshiro(13), Nx, Ny)
    mask = rand(Xoshiro(14), Nx, Ny) .< 0.85
    res = map(KC_BACKENDS) do (lab, b)
        with_backend(b) do
            dr = WM.DynamicRoughness(st; z0_floor = 1e-5, C_d = 0.3, mask = mask)
            WM.update_dynamic_roughness!(dr, uc, vc, 0.05)
            (dr.α, dr.converged, copy(dr.z0), collect(WM.roughness_surface_drag(dr, uc, vc, 0.05)))
        end
    end
    @test res[1][2]                                     # a root was found
    for r in res[2:end]
        @test kc_bits(r[1]) == kc_bits(res[1][1]) && kc_bits(r[3]) == kc_bits(res[1][3]) &&
              kc_bits(r[4]) == kc_bits(res[1][4])
    end
end

@testset "RadiationFog: dew_exchange! per column" begin
    MT = Aeolus.MoistThermodynamics
    rng = Xoshiro(17)
    Ts = 280 .+ 8 .* rand(rng, 20, 14)
    q0 = [MT.saturation_specific_humidity(T, 1.0e5) * (0.9 + 0.2rand(rng)) for T in 282 .+ 4 .* rand(rng, 20, 14)]
    d0 = ifelse.(rand(rng, 20, 14) .< 0.5, 1e-3 .* rand(rng, 20, 14), 0.0)
    ref = nothing
    for (lab, b) in KC_BACKENDS, three in (false, true)
        q = three ? reshape(copy(q0), 20, 14, 1) : copy(q0); d = copy(d0)
        H = with_backend(() -> dew_exchange!(q, d, Ts, 1.0e5, 60.0; dz = 4.0, ρ = 1.2), b)
        r = (vec(q), d, H)
        ref === nothing && (ref = r)
        @test all(n -> kc_bits(r[n]) == kc_bits(ref[n]), 1:3)
    end
    @test count(!=(0), ref[3]) > 50                     # dew forms and evaporates
end

end # testset KA-m
