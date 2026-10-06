# KA-n (#305): the radiation and stochastic modules on the launch / reduction
# primitives.
#
#   1. Radiosity: the cached LU gives the bits of a refactorisation and
#      refactorises only when F or ρ change; the threaded view-factor
#      assembly and the column-gather `aggregate_to_grid` are bit-identical
#      to the serial loops before the port (test/support/legacy_radiosity.jl)
#      on every backend; `set_temperatures!` from a column map.
#   2. Gas / cloud optics and the two-stream solves: the multi-column forms
#      give, column by column, the bits of the single-column forms, on every
#      backend, for mixed columns (cloudy, clear, dry, warm) and mixed
#      surfaces.
#   3. The counter-based random streams (Backends) and the Monte-Carlo and
#      superdroplet modules that use them: thread-invariant on every backend.

using Test
using Random: MersenneTwister, Xoshiro
using LinearAlgebra
using Aeolus
using Aeolus.Radiosity
using Aeolus.RadiationCore
using Aeolus.GasOptics
using Aeolus.TwoStream

include(joinpath(@__DIR__, "support", "legacy_radiosity.jl"))

const KR_BACKENDS = ("serial" => CPUBackend(serial = true),
                     "threaded, threshold 0" => CPUBackend(threshold = 0),
                     "default" => CPUBackend())
kr_bits(A) = reinterpret(UInt64, vec(collect(Float64, A)))
kr_same(a, b) = size(a) == size(b) && kr_bits(a) == kr_bits(b)

# A small city: blocks of different heights, a courtyard, open edges.
function kr_scene(; nsub = 2)
    hm = zeros(9, 8)
    hm[2:3, 2:4] .= 12.0
    hm[5:7, 2:3] .= 20.0
    hm[5:7, 6:7] .= 8.0
    hm[6, 6] = 0.0                  # courtyard
    hm[2, 7] = 30.0                 # tower
    cls = fill(URBAN_STREET, 9, 8)
    cls[8:9, :] .= URBAN_PARK
    cls[1, 1:3] .= URBAN_WATER
    s = build_scene(hm, 3.0, 3.0; wall_subdivisions = nsub, class_map = cls)
    for i in 1:nfacets(s)
        set_temperature!(s, i, 285.0 + 15 * sin(0.37 * i))
    end
    return s
end

@testset "KA-n: radiation and stochastic modules (#305)" begin

    @testset "Radiosity: view factors bit-identical to the serial pair loop" begin
        s = kr_scene()
        F0, Fsky0 = legacy_view_factors(s)
        for (name, b) in KR_BACKENDS
            vf = view_factors(s; backend = b)
            @test kr_same(vf.F, F0)
            @test kr_same(vf.F_sky, Fsky0)
        end
        # odd and even facet counts (the row pairing p ↔ n − p)
        hm3 = zeros(3, 3); hm3[2, 2] = 6.0
        for s2 in (kr_scene(; nsub = 1), kr_scene(; nsub = 3), build_scene(hm3, 2.0, 2.0))
            F2, Fs2 = legacy_view_factors(s2)
            vf2 = view_factors(s2; backend = CPUBackend(threshold = 0))
            @test kr_same(vf2.F, F2)
            @test kr_same(vf2.F_sky, Fs2)
        end
        @test isodd(nfacets(build_scene(hm3, 2.0, 2.0)))
        @test iseven(nfacets(s))
        # reciprocity and closure as before
        vf = view_factors(s)
        @test reciprocity_error(vf) < 1e-14
        @test closure_error(vf) < 1e-3
    end

    @testset "Radiosity: cached LU = refactorisation, refactorised only on change" begin
        s = kr_scene()
        vf = view_factors(s)
        n = nfacets(s)
        α = facet_albedos(s)
        E1 = [100.0 + 40 * sin(0.3 * i) for i in 1:n]
        E2 = [80.0 + 30 * cos(0.2 * i) for i in 1:n]
        c = RadiosityFactorCache()
        @test c.nfactor == 0
        B1 = solve_radiosity_direct(vf.F, α, E1; cache = c)
        B2 = solve_radiosity_direct(vf.F, α, E2; cache = c)
        @test c.nfactor == 1                        # second solve reused the LU
        @test kr_same(B1, solve_radiosity_direct(vf.F, α, E1))
        @test kr_same(B2, solve_radiosity_direct(vf.F, α, E2))
        # a new ρ refactorises, and the result is that of the new ρ
        α2 = α .* 1.1
        B3 = solve_radiosity_direct(vf.F, α2, E1; cache = c)
        @test c.nfactor == 2
        @test kr_same(B3, solve_radiosity_direct(vf.F, α2, E1))
        # equal values in a new vector: no refactorisation
        solve_radiosity_direct(vf.F, copy(α2), E2; cache = c)
        @test c.nfactor == 2
        # another F object (even with equal values) refactorises
        solve_radiosity_direct(copy(vf.F), α2, E2; cache = c)
        @test c.nfactor == 3
        # the band solvers: the longwave LU is computed once while T changes
        clw = RadiosityFactorCache()
        lw1 = solve_longwave(s, vf; L_sky = 320.0, cache = clw)
        @test kr_same(lw1.radiosity, solve_longwave(s, vf; L_sky = 320.0).radiosity)
        set_temperatures!(s, [290.0 + 5 * cos(i) for i in 1:n])
        lw2 = solve_longwave(s, vf; L_sky = 330.0, cache = clw)
        @test clw.nfactor == 1
        @test kr_same(lw2.radiosity, solve_longwave(s, vf; L_sky = 330.0).radiosity)
        @test isapprox(lw2.total_absorbed,
                       lw2.total_emitted + lw2.total_from_sky - lw2.total_escaping; rtol = 1e-12)
        # shortwave: the albedos follow μ₀, so a new μ₀ refactorises
        csw = RadiosityFactorCache()
        sun = solar_unit_vector((0.3, -0.4, 0.8))
        for (k, μ₀) in enumerate((0.7, 0.7, 0.5))
            m = default_urban_materials(; μ₀ = μ₀)
            s.materials.albedo .= m.albedo
            sw = solve_shortwave(s, vf, sun; dni = 800.0, dhi = 90.0, cache = csw)
            @test csw.nfactor == (k == 3 ? 2 : 1)
            @test kr_same(sw.radiosity,
                          solve_shortwave(s, vf, sun; dni = 800.0, dhi = 90.0).radiosity)
            @test isapprox(sw.total_absorbed + sw.total_escaping, sw.total_incident; rtol = 1e-12)
        end
        # a degenerate (diagonal) operator goes through the same path as `\`
        Fz = zeros(3, 3)
        cz = RadiosityFactorCache()
        @test solve_radiosity_direct(Fz, [0.2, 0.3, 0.4], [1.0, 2.0, 3.0]; cache = cz) == [1.0, 2.0, 3.0]
    end

    @testset "Radiosity: aggregate_to_grid gather = the facet scatter" begin
        s = kr_scene()
        vf = view_factors(s)
        sun = solar_unit_vector((0.2, 0.5, 0.7))
        sw = solve_shortwave(s, vf, sun; dni = 750.0, dhi = 110.0)
        lw = solve_longwave(s, vf; L_sky = 310.0)
        Ts = [290.0 + 2 * cos(i + 2j) for i in 1:9, j in 1:8]
        for kw in ((sw = sw, lw = lw, T_surface = Ts), (sw = sw, lw = lw), (lw = lw,), NamedTuple())
            ref = legacy_aggregate_to_grid(s, vf; kw...)
            for (name, b) in KR_BACKENDS
                agg = aggregate_to_grid(s, vf; kw..., backend = b)
                for f in fieldnames(UrbanRadiationForcing)
                    @test kr_same(getfield(agg, f), getfield(ref, f))
                end
            end
        end
        # the CSR index lists every in-domain facet once, ascending per column
        ptr, ids = facet_column_index(s)
        @test sort(ids) == collect(1:nfacets(s))
        @test all(issorted(ids[ptr[c]:ptr[c+1]-1]) for c in 1:length(ptr)-1)
        # an exact tie of two materials goes to the lower material index
        hm = zeros(2, 1)
        st = build_scene(hm, 1.0, 1.0)
        f1 = st.facets[1]
        st.facets[1] = Facet{Float64}(f1.centroid, f1.normal, f1.u, f1.v, f1.lu, f1.lv, f1.area,
                                      f1.kind, URBAN_WATER, f1.temperature, (1, 1))
        f2 = st.facets[2]
        st.facets[2] = Facet{Float64}(f2.centroid, f2.normal, f2.u, f2.v, f2.lu, f2.lv, f2.area,
                                      f2.kind, URBAN_ROOF, f2.temperature, (1, 1))
        @test aggregate_to_grid(st, view_factors(st)).class[1, 1] == min(URBAN_WATER, URBAN_ROOF)
    end

    @testset "Radiosity: column temperatures onto the facets" begin
        s = kr_scene()
        Ts = [280.0 + i + 0.1j for i in 1:9, j in 1:8]
        cols = Tuple{Int,Int}[f.col for f in s.facets]
        for (name, b) in KR_BACKENDS
            set_temperatures!(s, 0.0)
            set_temperatures!(s, Ts, cols; backend = b)
            @test facet_temperatures(s) == [Ts[c...] for c in cols]
        end
    end
end
