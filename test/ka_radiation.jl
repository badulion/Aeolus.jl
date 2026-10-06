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
using Random: MersenneTwister, Xoshiro, shuffle!
using LinearAlgebra
using Aeolus
using Aeolus.Radiosity
using Aeolus.RadiationCore
using Aeolus.GasOptics
using Aeolus.TwoStream
using Aeolus.MonteCarloRT

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

    @testset "Gas and cloud optics: multi-column = single column, every backend" begin
        sw, lw = shortwave_bands(), longwave_bands()
        cols = [us_standard_column(; nz = 24, T_sfc = 288.15),
                us_standard_column(; nz = 24, T_sfc = 300.0, humid = :midlatitude_summer),
                us_standard_column(; nz = 24, T_sfc = 265.0, humid = :subarctic_winter),
                us_standard_column(; nz = 24, T_sfc = 293.0)]
        cols[2].ql[2:4] .= 3e-4                    # a low cloud
        cols[4].qi[10:11] .= 5e-5                  # an ice layer
        for (spec, go, cl) in ((sw, correlated_k_optics(sw), mie_cloud_optics(sw)),
                               (lw, correlated_k_optics(lw), mie_cloud_optics(lw)))
            ref = map(cols) do col
                o = allocate_optics(Float64, (24,), spec)
                compute_optics!(o, go, col; backend = CPUBackend(serial = true))
                oc = allocate_optics(Float64, (24,), spec)
                compute_optics!(oc, cl, col; backend = CPUBackend(serial = true))
                (o, oc)
            end
            for (name, b) in KR_BACKENDS
                o3 = allocate_column_optics(Float64, 24, length(cols), spec)
                compute_optics!(o3, go, cols; backend = b)
                oc3 = allocate_column_optics(Float64, 24, length(cols), spec)
                compute_optics!(oc3, cl, cols; backend = b)
                for c in eachindex(cols)
                    @test kr_same(o3.τ[:, :, c], ref[c][1].τ)
                    @test kr_same(o3.ssa[:, :, c], ref[c][1].ssa)
                    @test kr_same(o3.g[:, :, c], ref[c][1].g)
                    @test kr_same(oc3.τ[:, :, c], ref[c][2].τ)
                    @test kr_same(oc3.ssa[:, :, c], ref[c][2].ssa)
                    @test kr_same(oc3.g[:, :, c], ref[c][2].g)
                end
                # the single-column form on this backend: the same bits too
                o1 = allocate_optics(Float64, (24,), spec)
                compute_optics!(o1, go, cols[3]; backend = b)
                @test kr_same(o1.τ, ref[3][1].τ)
            end
            # optics are physical: τ ≥ 0, the cloud only where there is condensate
            @test all(>=(0), ref[1][1].τ)
            @test all(==(0), ref[1][2].τ) && any(>(0), ref[2][2].τ)
        end
    end

    @testset "Two-stream: multi-column solves = single-column solves, every backend" begin
        sw, lw = shortwave_bands(), longwave_bands()
        nz = 24
        cols = [us_standard_column(; nz = nz, T_sfc = 288.15 + 4c) for c in 1:3]
        cols[2].ql[3:5] .= 4e-4
        osw = allocate_column_optics(Float64, nz, 3, sw)
        olw = allocate_column_optics(Float64, nz, 3, lw)
        compute_optics!(osw, correlated_k_optics(sw), cols)
        compute_optics!(olw, correlated_k_optics(lw), cols)
        ocs = allocate_column_optics(Float64, nz, 3, sw); compute_optics!(ocs, mie_cloud_optics(sw), cols)
        ocl = allocate_column_optics(Float64, nz, 3, lw); compute_optics!(ocl, mie_cloud_optics(lw), cols)
        combine_optics!(osw, ocs); combine_optics!(olw, ocl)
        column(o, c) = OpticalProperties{Float64,2}(o.τ[:, :, c], o.ssa[:, :, c], o.g[:, :, c], o.spec)
        μ = [0.8, 0.45, 0.1]
        αdir = [0.10 + 0.002b + 0.05c for b in 1:nbands(sw), c in 1:3]   # (band, column)
        αdif = [0.12, 0.2, 0.3]                                          # per column
        ε = [0.90 + 0.003b + 0.01c for b in 1:nbands(lw), c in 1:3]
        ssw = two_stream_solver(sw)
        slw = two_stream_solver(lw)
        refsw = map(1:3) do c
            f = allocate_fluxes(Float64, (), nz + 1)
            solve_sw!(f, ssw, column(osw, c); μ₀ = μ[c], sfc_albedo_direct = αdir[:, c],
                      sfc_albedo_diffuse = αdif[c], backend = CPUBackend(serial = true))
        end
        reflw = map(1:3) do c
            f = allocate_fluxes(Float64, (), nz + 1)
            solve_lw!(f, slw, column(olw, c), cols[c]; sfc_emissivity = ε[:, c],
                      backend = CPUBackend(serial = true))
        end
        ws = TwoStreamWorkspace{Float64}()
        for (name, b) in KR_BACKENDS, rep in 1:2           # rep 2 reuses the workspace
            fsw = allocate_fluxes(Float64, (3,), nz + 1)
            solve_sw!(fsw, ssw, osw; μ₀ = μ, sfc_albedo_direct = αdir, sfc_albedo_diffuse = αdif,
                      workspace = ws, backend = b)
            flw = allocate_fluxes(Float64, (3,), nz + 1)
            solve_lw!(flw, slw, olw, cols; sfc_emissivity = ε, workspace = ws, backend = b)
            for c in 1:3
                for f in (:flux_up, :flux_dn, :flux_dn_dir, :flux_net)
                    @test kr_same(getfield(fsw, f)[c, :], getfield(refsw[c], f))
                    @test kr_same(getfield(flw, f)[c, :], getfield(reflw[c], f))
                end
            end
        end
        # physics of the solves: the top boundary is the incident beam F₀μ₀;
        # the column absorbs (net down at TOA ≥ net at the surface ≥ 0); the
        # cloudy column sends more longwave down to the surface
        for c in 1:3
            @test refsw[c].flux_dn[end] ≈ sum(sw.solar_src) * μ[c] rtol = 1e-12
            @test refsw[c].flux_net[end] >= refsw[c].flux_net[1] >= 0
        end
        @test reflw[2].flux_dn[1] > reflw[1].flux_dn[1]
        # more columns than one launch chunk: every column still matches
        nc = Aeolus.TwoStream.TWOSTREAM_CHUNK + 5
        big = allocate_column_optics(Float64, nz, nc, lw)
        for c in 1:nc
            big.τ[:, :, c] .= olw.τ[:, :, 1 + c % 3] .* (1 + 0.01c)
        end
        bigcols = [cols[1 + c % 3] for c in 1:nc]
        fbig = allocate_fluxes(Float64, (nc,), nz + 1)
        solve_lw!(fbig, slw, big, bigcols; sfc_emissivity = 0.97, workspace = ws,
                  backend = CPUBackend(threshold = 0))
        for c in (1, 64, 65, nc)
            f = allocate_fluxes(Float64, (), nz + 1)
            solve_lw!(f, slw, column(big, c), bigcols[c]; sfc_emissivity = 0.97)
            @test kr_same(fbig.flux_up[c, :], f.flux_up)
            @test kr_same(fbig.flux_dn[c, :], f.flux_dn)
        end
    end

    @testset "Cloud shadow: optical depth and rays per column, every backend" begin
        mesh = RegularGrid(; size = (12, 10, 8), extent = (240.0, 200.0, 160.0), halo = (2, 2, 2))
        rng = Xoshiro(305)
        ql = zeros(12, 10, 8); ql[3:7, 2:6, 4:6] .= 1e-4 .* rand(rng, 5, 5, 3)
        qi = zeros(12, 10, 8); qi[8:10, 5:9, 6:7] .= 2e-5
        ρ = [1.2 - 0.01k for k in 1:8]
        # transcript of the cell loop before KA-n
        τref = similar(ql)
        for k in 1:8, j in 1:10, i in 1:12
            lwc = ql[i, j, k] * ρ[k] * 1000
            iwc = qi[i, j, k] * ρ[k] * 1000
            τref[i, j, k] = (0.15 * lwc + 0.033 * iwc) * 20.0
        end
        mask = falses(16, 14, 12); mask[2 + 5, 2 + 4, 3:2 + 3] .= true     # a 3-cell tower
        kst = [Int(count(k -> mask[2 + i, 2 + j, 2 + k], 1:8)) for i in 1:12, j in 1:10]
        sd = (0.35, -0.2, 0.6) ./ sqrt(0.35^2 + 0.2^2 + 0.6^2)
        ref = nothing
        for (name, b) in KR_BACKENDS
            τ = cloud_optical_depth_column(ql, qi, ρ, 20.0; k_liq = 0.15, k_ice = 0.033, backend = b)
            @test kr_same(τ, τref)
            t1 = zeros(12, 10); cloud_shadow_transmittance!(t1, τ, mesh, sd; k_start = kst, backend = b)
            t2 = zeros(12, 10); combined_shadow_transmittance!(t2, τ, mask, mesh, sd; backend = b)
            t3 = zeros(12, 10); cloud_shadow_transmittance!(t3, τ, mesh, sd; backend = b)
            ref === nothing && (ref = (t1, t2, t3))
            @test kr_same(t1, ref[1]) && kr_same(t2, ref[2]) && kr_same(t3, ref[3])
        end
        t1, t2, t3 = ref
        @test all(t -> 0 < t <= 1, t1) && minimum(t1) < 1
        @test count(==(0), t2) > 0                 # the tower's building shadow
        # a uniform layer of vertical optical depth τ transmits exactly exp(−τ/μ₀)
        τu = fill(0.05, 12, 10, 8)
        tu = zeros(12, 10); cloud_shadow_transmittance!(tu, τu, mesh, sd; backend = CPUBackend(threshold = 0))
        @test all(isapprox.(tu, exp(-0.4 / sd[3]); rtol = 1e-12))
    end

    @testset "Counter-based random streams (Philox4x32-10)" begin
        φ = Aeolus.Backends.philox4x32
        z = UInt32(0); f = typemax(UInt32)
        # Random123 known-answer vectors (kat_vectors, philox4x32_10)
        @test φ((z, z, z, z), (z, z)) == (0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8)
        @test φ((f, f, f, f), (f, f)) == (0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd)
        @test φ((0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344), (0xa4093822, 0x299f31d0)) ==
              (0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1)
        r = counter_rng(42, 3, 7)
        a = [rand(r) for _ in 1:9]
        @test a[1] == rand(counter_rng(42, 3, 7))         # a pure function of (seed, stream, id)
        r2 = counter_rng(1, 0, 0)
        set_stream!(r2, 42, 3, 7)
        @test [rand(r2) for _ in 1:9] == a                 # re-pointed = fresh stream
        @test rand(counter_rng(42, 3, 8)) != a[1]          # neighbouring ids differ
        @test rand(counter_rng(42, 4, 7)) != a[1]
        @test rand(counter_rng(43, 3, 7)) != a[1]
        @test all(x -> 0 <= x < 1, a)
        # first draws of 10^5 neighbouring streams: uniform mean and variance
        x = [rand(counter_rng(2026, 0, i)) for i in 0:99_999]
        @test abs(sum(x) / length(x) - 0.5) < 4 * sqrt(1 / 12 / length(x))
        @test abs(sum(abs2, x .- 0.5) / length(x) - 1 / 12) < 0.002
        # and the Random API on top: randn, ranges, shuffle!
        g = [randn(counter_rng(5, 1, i)) for i in 0:19_999]
        @test abs(sum(g) / length(g)) < 4 / sqrt(length(g))
        @test abs(sum(abs2, g) / length(g) - 1) < 0.05
        @test sort(shuffle!(counter_rng(1, 1, 1), collect(1:10))) == 1:10
        @test_throws ArgumentError counter_rng(1, -1, 0)
        @test_throws ArgumentError counter_rng(1, 0, -1)
    end

    @testset "Monte-Carlo solve: thread-invariant photon streams" begin
        dom = mc_domain(Float64; size = (6, 5, 8), spacing = (40.0, 40.0, 40.0))
        kext = zeros(6, 5, 8); ssa = zeros(6, 5, 8); gg = zeros(6, 5, 8)
        kext[2:4, 2:4, 3:5] .= 0.05; ssa[2:4, 2:4, 3:5] .= 0.98; gg[2:4, 2:4, 3:5] .= 0.8
        med = mc_medium(dom; kext = kext, ssa = ssa, g = gg, sfc_albedo = 0.2)
        solver = monte_carlo_solver(; nphoton = 8_000, nbatch = 8)
        sd = mc_sun_direction(0.7; azimuth = 120.0)
        ref = nothing
        for (name, b) in KR_BACKENDS
            r = mc_solve(med, solver; seed = 305, sun_dir = sd, solar_flux = 900.0, backend = b)
            vals = (r.absorbed, r.absorbed_err, r.sfc_direct, r.sfc_diffuse, r.flux_up, r.flux_dn,
                    [r.albedo, r.albedo_err, r.transmittance, r.absorptance_atm])
            ref === nothing && (ref = vals)
            @test all(kr_same(v, w) for (v, w) in zip(vals, ref))
            @test abs(mc_closure_residual(r)) < 1e-10
        end
        # `rng` keys the streams by one draw: the same seed, the same solve
        r1 = mc_solve(med, solver; rng = MersenneTwister(9), sun_dir = sd)
        r2 = mc_solve(med, solver; rng = MersenneTwister(9), sun_dir = sd)
        @test kr_same(r1.absorbed, r2.absorbed)
        @test_throws ArgumentError mc_solve(med, solver; sun_dir = sd)    # no global stream
    end
end
