using Test
using LinearAlgebra
using Aeolus
using Aeolus.Radiosity
using Aeolus.RadiationCore: STEFAN_BOLTZMANN

# =========================================================================
# Era III — Phase 35: full urban radiosity
# =========================================================================
#
# The validation strategy, in order of hardness:
#
#   1. *Exact identities.* Reciprocity and closure must hold to machine
#      precision. They are engineered into the assembly, so a failure here
#      means the assembly itself is broken, not the physics.
#   2. *External closed forms.* Two classical rectangle view factors (parallel
#      coaxial, perpendicular sharing an edge) and the infinite-street-canyon
#      sky-view factor. These come from the literature and are never adjusted
#      to match the code — they are the only checks that the quadrature is
#      actually right rather than merely self-consistent.
#   3. *Limits.* A scene with no buildings must reproduce a flat plate exactly;
#      an isolated wall on a growing ground plane must approach F_sky = 1/2.
#   4. *Physics signatures.* Effective canyon albedo below the material albedo
#      and monotone in H/W; longwave cooling suppressed in a canyon.

# Canyon test scene: two building rows of height `H` running the full length
# `L` of the domain, separated by a street `W` cells wide. Cell size 1 m, so
# H/W = H/W_cells. The buildings sit on the domain edge, so the street is the
# only canyon and the outward-facing perimeter walls see nothing.
function _canyon_scene(Wcells::Int, H::Real, L::Int;
                       nsub::Int = 2, albedo::Real = 0.2,
                       emissivity::Real = 0.95, dx::Real = 1.0)
    nx = Wcells + 2
    hm = zeros(Float64, nx, L)
    hm[1, :]  .= H
    hm[nx, :] .= H
    return build_scene(hm, dx, dx; wall_subdivisions = nsub,
                       materials = uniform_materials(albedo, emissivity))
end

function _flat_scene(nx::Int, ny::Int; albedo::Real = 0.2,
                     emissivity::Real = 0.95, dx::Real = 1.0)
    return build_scene(zeros(Float64, nx, ny), dx, dx;
                       materials = uniform_materials(albedo, emissivity))
end

# Facet-level helpers for the analytic benchmarks: two rectangles in the
# canonical configurations of the closed-form catalogue.
const _EX = (1.0, 0.0, 0.0)
const _EY = (0.0, 1.0, 0.0)
const _EZ = (0.0, 0.0, 1.0)
const _MEZ = (0.0, 0.0, -1.0)

# Two identical a×b rectangles, coaxial, separated by c.
function _parallel_pair(a, b, c)
    f1 = Facet((0.0, 0.0, 0.0), _EZ, _EX, _EY, a, b, :ground, 1, 300.0, (1, 1))
    f2 = Facet((0.0, 0.0, float(c)), _MEZ, _EY, _EX, b, a, :roof, 1, 300.0, (1, 1))
    return f1, f2
end

# Floor l×w in z = 0 and wall l×h in y = 0, sharing the x-axis edge of length l.
function _perpendicular_pair(l, w, h)
    f1 = Facet((l / 2, w / 2, 0.0), _EZ, _EX, _EY, l, w, :ground, 1, 300.0, (1, 1))
    f2 = Facet((l / 2, 0.0, h / 2), _EY, _EZ, _EX, h, l, :wall, 5, 300.0, (1, 1))
    return f1, f2
end

@testset "Era III — Urban radiosity (Phase 35)" begin

    @testset "Facet decomposition" begin
        # one 10 m building in a 5×5 field of 2 m cells
        hm = zeros(5, 5); hm[3, 3] = 10.0
        facets = build_facets(hm, 2.0, 2.0; wall_subdivisions = 5)
        wall = filter(f -> f.kind === :wall, facets)
        @test count(f -> f.kind === :ground, facets) == 24 && count(f -> f.kind === :roof, facets) == 1
        @test length(wall) == 20 && sum(f -> f.area, wall) ≈ 4 * 2.0 * 10.0
        # right-handed (u, v, n) frames — the contour integral flips sign otherwise
        @test all(facets) do f
            c = (f.u[2] * f.v[3] - f.u[3] * f.v[2], f.u[3] * f.v[1] - f.u[1] * f.v[3],
                 f.u[1] * f.v[2] - f.u[2] * f.v[1])
            all(abs.(c .- f.normal) .< 1e-14) && f.area ≈ f.lu * f.lv
        end
        # only the exposed part of a stepped wall becomes a facet (no double count)
        hm2 = zeros(3, 1); hm2[1, 1] = 10.0; hm2[2, 1] = 4.0
        w2 = filter(f -> f.kind === :wall, build_facets(hm2, 1.0, 1.0; wall_subdivisions = 1))
        @test sort(map(f -> f.area, w2)) ≈ [4.0, 4.0, 4.0, 6.0, 10.0, 10.0, 10.0]
        @test first(filter(f -> f.normal[1] > 0.5, wall)).col == (4, 3)   # attributed to the open column
    end

    # Closed forms (Howell C-11, C-14): F = 0.199825 for unit squares one unit
    # apart, 0.200044 for two faces of a unit cube; the double contour integral
    # is exact for grid-aligned rectangles and symmetric in its arguments.
    @testset "Analytic view-factor benchmarks" begin
        @test view_factor_parallel_rectangles(1, 1, 1) ≈ 0.19982489569838746 atol = 1e-12
        @test view_factor_perpendicular_rectangles(1, 1, 1) ≈ 0.20004377607540316 atol = 1e-12
        @test 4view_factor_perpendicular_rectangles(1, 1, 1) + view_factor_parallel_rectangles(1, 1, 1) ≈ 1 atol = 1e-12
        err = 0.0
        for (a, b, c) in ((2.0, 3.0, 1.5), (1.0, 1.0, 0.1), (0.5, 0.5, 3.0))
            f1, f2 = _parallel_pair(a, b, c)
            err = max(err, abs(view_factor_contour(f1, f2) / (a * b) - view_factor_parallel_rectangles(a, b, c)),
                      abs(view_factor_contour(f1, f2) - view_factor_contour(f2, f1)))
        end
        for (l, w, h) in ((2.0, 1.0, 3.0), (1.0, 1.0, 10.0), (5.0, 0.5, 0.5))
            f1, f2 = _perpendicular_pair(l, w, h)
            err = max(err, abs(view_factor_contour(f1, f2) / (l * w) - view_factor_perpendicular_rectangles(l, w, h)),
                      abs(view_factor_contour(f1, f2) - view_factor_contour(f2, f1)))
        end
        @test err < 1e-12
    end

    # Reciprocity A_i F_ij = A_j F_ji and closure Σ_j F_ij + F_sky = 1 to machine
    # precision on a multi-building scene; the sky-view factor agrees with an
    # independent ray cast to that estimator's own sampling error.
    @testset "Reciprocity and closure" begin
        hm = zeros(8, 8)
        hm[2:3, 2:3] .= 12.0; hm[6:7, 2:3] .= 6.0; hm[5:6, 6:7] .= 18.0
        scene = build_scene(hm, 3.0, 3.0; wall_subdivisions = 2)
        vf = view_factors(scene)
        A, F = vf.area, vf.F
        @test maximum(abs.(A .* F .- (A .* F)')) < 1e-12
        @test maximum(abs.(vec(sum(F; dims = 2)) .+ vf.F_sky .- 1)) < 1e-12
        @test all(F .>= 0) && all(diag(F) .== 0) && all(0 .<= vf.F_sky .<= 1)
        @test minimum(view_factors(scene; clamp_sky = false).F_sky) > -1e-10
        dif = abs.(sky_view_factors_raycast(scene; nθ = 16, nφ = 32) .- vf.F_sky)
        @test maximum(dif) < 0.06 && sum(dif) / length(dif) < 0.015
    end

    # No buildings: an exact flat plate in SW (α_eff = α) and LW (ε(σT⁴ − L_sky)).
    @testset "Flat-plate limit" begin
        scene = _flat_scene(5, 5; albedo = 0.27, dx = 2.0)
        vf = view_factors(scene)
        @test all(vf.F .== 0) && all(vf.F_sky .== 1)
        @test total_sky_exchange_area(vf) ≈ 100.0
        sw = solve_shortwave(scene, vf, solar_unit_vector((elevation = 55.0, azimuth = 165.0)); dni = 900.0, dhi = 110.0)
        @test sw.effective_albedo ≈ 0.27 atol = 1e-12
        set_temperatures!(scene, 295.0)
        lw = solve_longwave(scene, vf; L_sky = 330.0)
        @test maximum(abs.(lw.net_loss .- 0.95 * (STEFAN_BOLTZMANN * 295.0^4 - 330.0))) < 1e-9
    end

    # Floor sky-view factor of an (almost) infinite canyon: ψ = √((H/W)²+1) − H/W,
    # per strip as well. A finite canyon over-estimates the sky (end effects).
    @testset "Infinite-canyon sky-view factor" begin
        Wc = 5; hw = 1.0; H = hw * Wc; L = 48
        scene = _canyon_scene(Wc, H, L)
        vf = view_factors(scene)
        idx = [i for i in 1:nfacets(scene) if scene.facets[i].kind === :ground &&
                                             scene.facets[i].col[2] == div(L, 2)]
        @test length(idx) == Wc
        @test maximum(abs(vf.F_sky[i] / canyon_sky_view_factor(H, float(Wc), (scene.facets[i].col[1] - 2) * 1.0,
                                                               (scene.facets[i].col[1] - 1) * 1.0) - 1) for i in idx) < 0.03
        ψ = sum(vf.area[i] * vf.F_sky[i] for i in idx) / sum(vf.area[i] for i in idx)
        @test ψ > canyon_sky_view_factor(hw) && isapprox(ψ, canyon_sky_view_factor(hw); rtol = 0.03)
        # the strip form reduces to the centreline and wall-side point values
        @test all(abs(canyon_sky_view_factor(h, 1.0, 0.5 - 1e-6, 0.5 + 1e-6) - canyon_centreline_sky_view_factor(h)) < 1e-9 &&
                  abs(canyon_sky_view_factor(h, 1.0, 0.0, 1e-7) - canyon_wallside_sky_view_factor(h)) < 1e-7
                  for h in (0.25, 1.0, 4.0))
        @test canyon_wallside_sky_view_factor(0) == 0.5
    end

    # A wall on a ground plane sees half the sky; the finite plane biases F_sky
    # high, and the bias falls as the plane grows.
    @testset "Isolated wall on a ground plane" begin
        vals = map((11, 21)) do n
            hm = zeros(n, n); c = div(n + 1, 2); hm[c, c] = 2.0
            scene = build_scene(hm, 1.0, 1.0; wall_subdivisions = 1)
            vf = view_factors(scene)
            vf.F_sky[findfirst(k -> scene.facets[k].kind === :wall && scene.facets[k].normal[1] > 0.5, 1:nfacets(scene))]
        end
        @test 0.5 < vals[2] < vals[1]
    end

    @testset "Occlusion by intervening buildings" begin
        function trio(hB)
            hm = zeros(8, 3); hm[2, :] .= 10.0; hm[6, :] .= 10.0
            hB > 0 && (hm[4, :] .= hB)
            build_scene(hm, 1.0, 1.0; wall_subdivisions = 1)
        end
        Fs = map((0.0, 5.0, 20.0)) do hB
            sc = trio(hB); vf = view_factors(sc)
            @test reciprocity_error(vf) < 1e-12 && closure_error(vf) < 1e-12
            ia = findfirst(k -> sc.facets[k].kind === :wall && sc.facets[k].normal[1] > 0.5 &&
                                sc.facets[k].centroid[1] < 3 && sc.facets[k].col[2] == 2, 1:nfacets(sc))
            ib = findfirst(k -> sc.facets[k].kind === :wall && sc.facets[k].normal[1] < -0.5 &&
                                sc.facets[k].centroid[1] > 4 && sc.facets[k].col[2] == 2, 1:nfacets(sc))
            vf.F[ia, ib]
        end
        @test Fs[1] > Fs[2] > 0 && Fs[3] == 0
        sc = trio(9.0)
        @test segment_blocked(sc, (1.5, 1.5, 1.0), (6.5, 1.5, 1.0))
        @test !ray_blocked(sc, (3.5, 1.5, 12.0), (1.0, 0.0, 0.0), 100.0)
    end

    # Energy conservation in = absorbed + escaping, and the direct, Gauss-Seidel
    # and progressive solvers agree.
    @testset "Shortwave radiosity" begin
        hm = zeros(9, 9); hm[3:4, 3:5] .= 14.0; hm[6:7, 5:7] .= 8.0
        scene = build_scene(hm, 2.5, 2.5; wall_subdivisions = 2, materials = default_urban_materials())
        vf = view_factors(scene)
        sun = solar_unit_vector((elevation = 38.0, azimuth = 152.0))
        swd = solve_shortwave(scene, vf, sun; dni = 870.0, dhi = 130.0, solver = :direct)
        swg = solve_shortwave(scene, vf, sun; dni = 870.0, dhi = 130.0, solver = :gauss_seidel)
        swp = solve_shortwave(scene, vf, sun; dni = 870.0, dhi = 130.0, solver = :progressive, tol = 1e-14)
        @test abs(swd.total_incident - swd.total_absorbed - swd.total_escaping) < 1e-8 * swd.total_incident
        scale = maximum(swd.radiosity)
        @test maximum(abs, swd.radiosity .- swg.radiosity) < 1e-9scale
        @test maximum(abs, swd.radiosity .- swp.radiosity) < 1e-8scale
        @test any(swd.interreflected .> 1.0) && all(swd.interreflected .>= -1e-12)
        # zenith sun: every column presents one horizontal facet ⇒ DNI × plan area
        @test solve_shortwave(scene, vf, (0.0, 0.0, 1.0); dni = 1000.0, dhi = 0.0).total_incident ≈ 1000.0 * 81 * 2.5^2 rtol = 1e-12
    end

    # Radiative trapping: effective canyon albedo below the material albedo and
    # falling with H/W (zenith sun, so the perimeter walls receive nothing).
    @testset "Radiative trapping: effective canyon albedo" begin
        Wc = 6; L = 24
        flat = _flat_scene(Wc + 2, L; albedo = 0.2)
        @test effective_albedo(flat, view_factors(flat), (0.0, 0.0, 1.0); dni = 1000.0, dhi = 0.0) ≈ 0.2 atol = 1e-12
        αeff = map((0.5, 2.0)) do hw
            scene = _canyon_scene(Wc, hw * Wc, L; albedo = 0.2)
            sw = solve_shortwave(scene, view_factors(scene), (0.0, 0.0, 1.0); dni = 1000.0, dhi = 0.0)
            @test abs(sw.total_incident - sw.total_absorbed - sw.total_escaping) < 1e-8 * sw.total_incident
            sw.effective_albedo
        end
        @test αeff[2] < αeff[1] < 0.2
    end

    # LW: energy identity, canyon floors cool less than a flat plate and less
    # the deeper the canyon, roofs lose the flat-plate amount, and an
    # isothermal scene under an equal-temperature sky exchanges nothing.
    @testset "Longwave radiosity and canyon trapping" begin
        Tsfc = 300.0; Lsky = 280.0; ε = 0.95
        flat_loss = ε * (STEFAN_BOLTZMANN * Tsfc^4 - Lsky)
        Wc = 4; L = 32
        local scene, vf
        losses = map((1.0, 2.0)) do hw
            scene = _canyon_scene(Wc, hw * Wc, L; emissivity = ε)
            vf = view_factors(scene)
            set_temperatures!(scene, Tsfc)
            lw = solve_longwave(scene, vf; L_sky = Lsky)
            @test abs(lw.total_absorbed - (lw.total_emitted + lw.total_from_sky - lw.total_escaping)) < 1e-8 * lw.total_emitted
            ridx = [i for i in 1:nfacets(scene) if scene.facets[i].kind === :roof]
            @test sum(vf.area[ridx] .* lw.net_loss[ridx]) / sum(vf.area[ridx]) ≈ flat_loss rtol = 0.02
            idx = [i for i in 1:nfacets(scene) if scene.facets[i].kind === :ground && scene.facets[i].col[2] == div(L, 2)]
            sum(vf.area[idx] .* lw.net_loss[idx]) / sum(vf.area[idx])
        end
        @test 0 < losses[2] < losses[1] < flat_loss
        l1 = solve_longwave(scene, vf; L_sky = Lsky, solver = :direct)
        l2 = solve_longwave(scene, vf; L_sky = Lsky, solver = :gauss_seidel)
        @test maximum(abs, l1.radiosity .- l2.radiosity) < 1e-9 * maximum(l1.radiosity)
        l4 = solve_longwave(scene, vf; L_sky = STEFAN_BOLTZMANN * Tsfc^4)
        @test maximum(abs, l4.net_loss) < 1e-9
    end

    @testset "Direct beam and shadows" begin
        hm = zeros(11, 11); hm[6, 6] = 15.0
        scene = build_scene(hm, 2.0, 2.0; wall_subdivisions = 1)
        wi = Dict{Symbol, Int}()
        for i in 1:nfacets(scene)
            f = scene.facets[i]
            f.kind === :wall || continue
            wi[f.normal[1] > 0.5 ? :east : f.normal[1] < -0.5 ? :west : f.normal[2] > 0.5 ? :north : :south] = i
        end
        σm = sunlit_fractions(scene, solar_unit_vector((elevation = 20.0, azimuth = 110.0)); samples = 3)
        @test (σm[wi[:east]], σm[wi[:west]], σm[wi[:south]], σm[wi[:north]]) == (1, 0, 1, 0)
        @test all(sunlit_fractions(scene, solar_unit_vector((elevation = -5.0, azimuth = 110.0))) .== 0)
        σ = sunlit_fractions(scene, solar_unit_vector((elevation = 15.0, azimuth = 90.0 + 1e-3)); samples = 3)
        shaded = [scene.facets[i].col for i in 1:nfacets(scene) if scene.facets[i].kind === :ground && σ[i] < 1]
        @test !isempty(shaded) && all(c -> c[1] < 6, shaded)   # sun in the east ⇒ shadow west
        E = direct_beam_irradiance(scene, solar_unit_vector((elevation = 45.0, azimuth = 180.0)); dni = 800.0)
        gi = findfirst(i -> scene.facets[i].kind === :ground && scene.facets[i].col == (1, 1), 1:nfacets(scene))
        @test E[gi] ≈ 800sind(45.0) rtol = 1e-12
        @test E[wi[:south]] ≈ 800cosd(45.0) rtol = 1e-12
        # solar_unit_vector follows SolarGeometry's ENU convention
        p = Aeolus.SolarGeometry.solar_position(Aeolus.SolarGeometry.solar_time(2024, 6, 21, 10, 0, 0.0;
                                                latitude = 52.52, longitude = 13.40))
        @test maximum(abs.(solar_unit_vector(p) .- Aeolus.SolarGeometry.sun_direction_enu(p))) < 1e-15
    end

    # Aggregation onto the LES grid is exactly conservative, and the
    # *_effective fields invert step_surfaces!' algebra.
    @testset "Aggregation onto the LES grid" begin
        hm = zeros(8, 8); hm[3:4, 3:4] .= 10.0; hm[6, 6] = 20.0
        scene = build_scene(hm, 4.0, 4.0; wall_subdivisions = 2, materials = default_urban_materials())
        vf = view_factors(scene)
        sw = solve_shortwave(scene, vf, solar_unit_vector((elevation = 50.0, azimuth = 160.0)); dni = 880.0, dhi = 120.0)
        set_temperatures!(scene, 302.0)
        lw = solve_longwave(scene, vf; L_sky = 340.0)
        fc = aggregate_to_grid(scene, vf; sw = sw, lw = lw)
        @test sum(fc.sw_absorbed) * 16 ≈ sw.total_absorbed rtol = 1e-12
        @test sum(fc.lw_net) * 16 ≈ sum(vf.area .* lw.net_loss) rtol = 1e-12
        @test sum(fc.area_density) * 16 ≈ sum(vf.area) rtol = 1e-12
        α = [scene.materials.albedo[c] for c in fc.class]; εm = [scene.materials.emissivity[c] for c in fc.class]
        @test maximum(abs.((1 .- α) .* fc.sw_effective .- fc.sw_absorbed)) < 1e-9
        @test maximum(abs.(εm .* fc.lw_down_effective .- εm .* STEFAN_BOLTZMANN * 302.0^4 .+ fc.lw_net)) < 1e-9
        @test fc.class[3, 3] == URBAN_ROOF && fc.class[1, 1] == URBAN_STREET
        # bridge from the IBM solid mask
        mesh = RegularGrid(; size = (8, 8, 8), extent = (32.0, 32.0, 32.0), halo = (1, 1, 1))
        smask = Aeolus.IBM.SolidMask(mesh)
        for k in 1:3, j in 3:4, i in 3:4
            smask.mask[mesh.Hx + i, mesh.Hy + j, mesh.Hz + k] = true
        end
        hb = heightmap_from_solid_mask(smask)
        @test hb[3, 3] ≈ 3mesh.dz && hb[1, 1] == 0
    end

    @testset "Materials and API guards" begin
        m = default_urban_materials()
        # single source of truth: the class greys are DERIVED from the Phase 36
        # spectral library (rebuilt here independently from the exported API)
        sw, lw = shortwave_bands(), longwave_bands()
        @test all(enumerate((:street, :roof, :park, :water, :wall))) do (idx, cls)
            mix = Aeolus.Radiosity.URBAN_CLASS_MIXTURES[cls]
            surf = tiled_surface(sw, [(n === :water ? FresnelWater() : n) => f for (n, f) in mix]...; lw_spec = lw)
            isapprox(material_albedo(m, idx), broadband_albedo(surf, 0.7); rtol = 1e-12) &&
                isapprox(material_emissivity(m, idx), broadband_emissivity(surf, lw); rtol = 1e-12)
        end
        # water is a Fresnel interface: tenfold albedo swing towards grazing
        wa(μ) = default_urban_materials(; μ₀ = μ).albedo[Int(URBAN_WATER)]
        @test wa(0.17) > 8wa(1.0)
        @test material_emissivity(m, URBAN_ROOF) < material_emissivity(m, URBAN_STREET)  # metal fraction
        @test m isa Aeolus.RadiationCore.AbstractSurfaceOptics
        @test_throws ErrorException UrbanMaterials([0.1], [1.5])
        @test_throws ErrorException build_facets(zeros(3, 3), 1.0, 1.0; wall_subdivisions = 0)
        # a left-handed frame would flip the sign of every exchange area
        @test_throws ErrorException Facet((0.0, 0.0, 0.0), _EZ, _EY, _EX, 1.0, 1.0, :ground, 1, 300.0, (1, 1))
        hm = zeros(4, 4); hm[2, 2] = 5.0
        s2 = build_scene(hm, 2.0, 2.0; wall_subdivisions = 1)
        @test exchange_area(s2, 1, 1) == 0 && abs(exchange_area(s2, 1, 5) - exchange_area(s2, 5, 1)) < 1e-14
    end
end
