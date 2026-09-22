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
#
# Measured numbers are printed with @info so they can be quoted as evidence.

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

    # ---------------------------------------------------------------------
    # 1. Facet decomposition
    # ---------------------------------------------------------------------
    @testset "Facet decomposition" begin
        # A single 1-cell building of height 10 m in a 5×5 field of 2 m cells.
        hm = zeros(5, 5); hm[3, 3] = 10.0
        facets = build_facets(hm, 2.0, 2.0; wall_subdivisions = 5)

        ground = filter(f -> f.kind === :ground, facets)
        roof   = filter(f -> f.kind === :roof, facets)
        wall   = filter(f -> f.kind === :wall, facets)

        @test length(ground) == 24          # 25 columns minus the built one
        @test length(roof) == 1
        @test length(wall) == 4 * 5         # four sides × five vertical strips

        @test all(f -> f.area ≈ 4.0, ground)
        @test all(f -> f.normal == (0.0, 0.0, 1.0), ground)
        @test all(f -> f.centroid[3] == 0.0, ground)
        @test roof[1].centroid[3] ≈ 10.0
        @test roof[1].normal == (0.0, 0.0, 1.0)

        # Total wall area: perimeter (4 sides × 2 m) × height 10 m.
        @test sum(f -> f.area, wall) ≈ 4 * 2.0 * 10.0
        # Every wall normal is horizontal and outward-pointing.
        @test all(f -> abs(f.normal[3]) < 1e-14, wall)
        @test count(f -> f.normal[1] > 0.5, wall) == 5
        @test count(f -> f.normal[1] < -0.5, wall) == 5
        @test count(f -> f.normal[2] > 0.5, wall) == 5
        @test count(f -> f.normal[2] < -0.5, wall) == 5

        # The (u, v, n) frame must be right-handed — the contour integral
        # changes sign if it is not.
        for f in facets
            c = (f.u[2] * f.v[3] - f.u[3] * f.v[2],
                 f.u[3] * f.v[1] - f.u[1] * f.v[3],
                 f.u[1] * f.v[2] - f.u[2] * f.v[1])
            @test all(abs.(c .- f.normal) .< 1e-14)
            @test f.area ≈ f.lu * f.lv
        end

        # Corners must be coplanar with the facet and span its extents.
        f = wall[1]
        for p in facet_corners(f)
            @test abs(sum((p .- f.centroid) .* f.normal)) < 1e-12
        end

        # Stepped geometry: only the *exposed* part of a wall becomes a facet.
        # A 10 m column abutting a 4 m one, both 1 m square, with open ground
        # beyond. The tall column contributes three fully exposed 10 m faces
        # plus a 6 m step against its short neighbour; the short column
        # contributes three 4 m faces and *nothing* towards the tall one,
        # since that face is buried. Without the height-step rule the pair
        # would double-count 4 m of wall.
        hm2 = zeros(3, 1); hm2[1, 1] = 10.0; hm2[2, 1] = 4.0
        fs2 = build_facets(hm2, 1.0, 1.0; wall_subdivisions = 1)
        w2 = filter(f -> f.kind === :wall, fs2)
        @test length(w2) == 7
        @test sort(map(f -> f.area, w2)) ≈ [4.0, 4.0, 4.0, 6.0, 10.0, 10.0, 10.0]
        @test sum(f -> f.area, w2) ≈ 3 * 10.0 + 6.0 + 3 * 4.0
        # The 6 m step faces east, out of the tall column into the short one.
        step = only(filter(f -> f.area ≈ 6.0, w2))
        @test step.normal == (1.0, 0.0, 0.0)
        @test step.centroid[3] ≈ 7.0           # spans z ∈ [4, 10]
        @test step.centroid[1] ≈ 1.0           # on the shared column boundary

        # Walls are attributed to the open column they face.
        wpx = first(filter(f -> f.normal[1] > 0.5, wall))
        @test wpx.col == (4, 3)

        sc = build_scene(hm, 2.0, 2.0; wall_subdivisions = 5)
        @test nfacets(sc) == length(facets)
        @test sc.hmax ≈ 10.0
        @test length(facet_areas(sc)) == nfacets(sc)

        # Temperature mutation round-trips.
        set_temperatures!(sc, 305.0)
        @test all(facet_temperatures(sc) .== 305.0)
        set_temperature!(sc, 1, 280.0)
        @test facet_temperatures(sc)[1] == 280.0
        @test facet_temperatures(sc)[2] == 305.0
    end

    # ---------------------------------------------------------------------
    # 2. Analytic view-factor benchmarks
    #
    # These are the hard external checks. References are the standard
    # closed forms (Howell's catalogue C-11 and C-14); the two canonical
    # values are F = 0.199825 for unit squares one unit apart, and
    # F = 0.200044 for two faces of a unit cube.
    # ---------------------------------------------------------------------
    @testset "Analytic view-factor benchmarks" begin
        @test view_factor_parallel_rectangles(1, 1, 1) ≈ 0.19982489569838746 atol = 1e-12
        @test view_factor_perpendicular_rectangles(1, 1, 1) ≈ 0.20004377607540316 atol = 1e-12
        # Two faces of a cube plus the four remaining faces must close the box:
        # 4 perpendicular neighbours + 1 opposite + 0 self = 1.
        @test 4 * view_factor_perpendicular_rectangles(1, 1, 1) +
              view_factor_parallel_rectangles(1, 1, 1) ≈ 1 atol = 1e-12

        errs_par = Float64[]
        for (a, b, c) in ((1.0, 1.0, 1.0), (2.0, 3.0, 1.5), (4.0, 4.0, 1.0),
                          (1.0, 5.0, 2.0), (1.0, 1.0, 0.1), (0.5, 0.5, 3.0))
            f1, f2 = _parallel_pair(a, b, c)
            num = view_factor_contour(f1, f2) / (a * b)
            ref = view_factor_parallel_rectangles(a, b, c)
            push!(errs_par, abs(num - ref))
            @test num ≈ ref atol = 1e-12
            # The exchange area is symmetric — this is where reciprocity
            # comes from, so check it at the source.
            @test view_factor_contour(f1, f2) ≈ view_factor_contour(f2, f1) atol = 1e-13
        end

        errs_perp = Float64[]
        for (l, w, h) in ((1.0, 1.0, 1.0), (2.0, 1.0, 3.0), (1.0, 4.0, 2.0),
                          (3.0, 3.0, 0.5), (1.0, 1.0, 10.0), (5.0, 0.5, 0.5))
            f1, f2 = _perpendicular_pair(l, w, h)
            num = view_factor_contour(f1, f2) / (l * w)
            ref = view_factor_perpendicular_rectangles(l, w, h)
            push!(errs_perp, abs(num - ref))
            @test num ≈ ref atol = 1e-12
            @test view_factor_contour(f1, f2) ≈ view_factor_contour(f2, f1) atol = 1e-13
        end

        @info "View-factor benchmark errors (double contour integral vs closed form)" max_err_parallel=maximum(errs_par) max_err_perpendicular=maximum(errs_perp)

        # The contour integral is exact for grid-aligned rectangles because
        # every contributing edge pair is parallel and integrated in closed
        # form — so the Gauss order must not matter at all.
        f1, f2 = _perpendicular_pair(1.0, 1.0, 1.0)
        @test view_factor_contour(f1, f2; ngauss = 3) ≈
              view_factor_contour(f1, f2; ngauss = 16) atol = 1e-14
    end

    # ---------------------------------------------------------------------
    # 3. Reciprocity and closure on a non-trivial multi-building scene
    # ---------------------------------------------------------------------
    @testset "Reciprocity and closure" begin
        hm = zeros(10, 10)
        hm[3:4, 3:4] .= 12.0
        hm[7:8, 2:3] .= 6.0
        hm[6:7, 7:9] .= 18.0
        hm[2, 8]      = 9.0
        scene = build_scene(hm, 3.0, 3.0; wall_subdivisions = 2)
        vf = view_factors(scene)
        n = nfacets(scene)
        @test n > 100                       # genuinely multi-building

        # Reciprocity, to machine precision, entry by entry.
        worst_abs = 0.0
        for i in 1:n, j in 1:n
            worst_abs = max(worst_abs,
                            abs(vf.area[i] * vf.F[i, j] - vf.area[j] * vf.F[j, i]))
        end
        @test worst_abs < 1e-12
        @test reciprocity_error(vf) < 1e-13

        # Closure, facet by facet.
        for i in 1:n
            @test sum(@view vf.F[i, :]) + vf.F_sky[i] ≈ 1 atol = 1e-12
        end
        @test closure_error(vf) < 1e-12

        # Physical bounds.
        @test all(vf.F .>= 0)
        @test all(vf.F .<= 1)
        @test all(0 .<= vf.F_sky .<= 1)
        @test all(diag(vf.F) .== 0)         # a planar facet cannot see itself
        # Sky-view factor must be the *unclamped* residual here: if the
        # quadrature were pushing rows past unity we would see it.
        raw = view_factors(scene; clamp_sky = false)
        @test minimum(raw.F_sky) > -1e-10
        @test maximum(abs, raw.F_sky .- vf.F_sky) < 1e-10

        @info "Multi-building scene" facets=n reciprocity_error=reciprocity_error(vf) closure_error=closure_error(vf) min_sky_view=minimum(vf.F_sky)

        # Independent hemispherical ray cast must agree with the closure
        # residual. The ray cast is a stratified estimator, so the tolerance
        # is its own sampling error, not the view factors'.
        rc = sky_view_factors_raycast(scene; nθ = 16, nφ = 32)
        dif = abs.(rc .- vf.F_sky)
        @test maximum(dif) < 0.06
        @test sum(dif) / length(dif) < 0.015
        @info "Sky-view factor: closure residual vs independent ray cast" max_abs_diff=maximum(dif) mean_abs_diff=sum(dif)/length(dif)
    end

    # ---------------------------------------------------------------------
    # 4. Flat-plate limit
    # ---------------------------------------------------------------------
    @testset "Flat-plate limit" begin
        scene = _flat_scene(5, 5; albedo = 0.27, dx = 2.0)
        vf = view_factors(scene)
        @test nfacets(scene) == 25
        @test all(vf.F .== 0)               # coplanar facets never see each other
        @test all(vf.F_sky .== 1)
        @test closure_error(vf) == 0
        @test reciprocity_error(vf) == 0
        # With no obstruction the sky exchange area is exactly the plan area.
        @test total_sky_exchange_area(vf) ≈ 5 * 5 * 4.0

        sun = solar_unit_vector((elevation = 55.0, azimuth = 165.0))
        sw = solve_shortwave(scene, vf, sun; dni = 900.0, dhi = 110.0)
        @test sw.effective_albedo ≈ 0.27 atol = 1e-12
        @test all(sw.interreflected .== 0)
        @test sw.total_absorbed ≈ (1 - 0.27) * sw.total_incident atol = 1e-9
        @test sw.total_incident ≈ sw.total_absorbed + sw.total_escaping atol = 1e-8

        # Longwave: with F_sky = 1 the net loss is the textbook flat-plate value.
        set_temperatures!(scene, 295.0)
        lw = solve_longwave(scene, vf; L_sky = 330.0)
        expect = 0.95 * (STEFAN_BOLTZMANN * 295.0^4 - 330.0)
        @test all(isapprox.(lw.net_loss, expect; atol = 1e-9))
        @test longwave_trapping_factor(scene, vf, lw; L_sky = 330.0) ≈ 1 atol = 1e-12
        @info "Flat-plate limit" alpha_eff=sw.effective_albedo alpha_material=0.27 lw_net_loss=first(lw.net_loss) lw_flat_reference=expect
    end

    # ---------------------------------------------------------------------
    # 5. Infinite-canyon sky-view factor
    #
    # ψ = sqrt((H/W)² + 1) - H/W for the floor average; the per-strip form
    # must be matched by every individual floor facet. The modelled canyon is
    # finite, so the residual is the end-effect truncation and must shrink as
    # the canyon is lengthened — that convergence is itself part of the test.
    # ---------------------------------------------------------------------
    @testset "Infinite-canyon sky-view factor" begin
        Wc = 5
        for hw in (0.5, 1.0, 2.0)
            H = hw * Wc
            L = 48
            scene = _canyon_scene(Wc, H, L)
            vf = view_factors(scene)
            jmid = div(L, 2)
            idx = [i for i in 1:nfacets(scene)
                   if scene.facets[i].kind === :ground &&
                      scene.facets[i].col[2] == jmid]
            @test length(idx) == Wc

            # Per-facet strip average: the sharp test.
            relerrs = Float64[]
            for i in idx
                a1 = (scene.facets[i].col[1] - 2) * 1.0
                ref = canyon_sky_view_factor(H, float(Wc), a1, a1 + 1.0)
                push!(relerrs, abs(vf.F_sky[i] - ref) / ref)
                @test vf.F_sky[i] ≈ ref rtol = 0.03
            end

            ψ = sum(vf.area[i] * vf.F_sky[i] for i in idx) /
                sum(vf.area[i] for i in idx)
            ψref = canyon_sky_view_factor(hw)
            @test ψ ≈ ψref rtol = 0.03
            # Finite length always *over*-estimates the sky: the missing wall
            # beyond the canyon ends would have blocked more.
            @test ψ > ψref

            # Cross-street profile: the centreline sees the most sky, the
            # strip hard against a wall the least, and the average lies
            # between. Three independent closed forms, and the modelled
            # facets must bracket the same way.
            @test canyon_centreline_sky_view_factor(hw) > ψref
            @test canyon_wallside_sky_view_factor(hw) < ψref
            fsk = [vf.F_sky[i] for i in idx]
            @test argmax(fsk) == div(Wc + 1, 2)      # brightest in mid-street
            @test fsk[1] ≈ fsk[end] rtol = 1e-6      # symmetric canyon
            @test fsk[1] < fsk[div(Wc + 1, 2)]

            @info "Canyon sky-view factor (W = $Wc m, L = $L m)" H_over_W=hw psi_modelled=ψ psi_analytic=ψref rel_error=abs(ψ-ψref)/ψref max_per_facet_rel_error=maximum(relerrs)
        end

        # Convergence to the infinite-canyon limit as the canyon lengthens.
        hw = 2.0; H = hw * Wc
        errs = Float64[]
        for L in (16, 32, 64)
            scene = _canyon_scene(Wc, H, L)
            vf = view_factors(scene)
            jmid = div(L, 2)
            idx = [i for i in 1:nfacets(scene)
                   if scene.facets[i].kind === :ground &&
                      scene.facets[i].col[2] == jmid]
            ψ = sum(vf.area[i] * vf.F_sky[i] for i in idx) /
                sum(vf.area[i] for i in idx)
            push!(errs, abs(ψ - canyon_sky_view_factor(hw)))
        end
        @test errs[2] < errs[1]
        @test errs[3] < errs[2]
        @info "Canyon SVF convergence with canyon length (H/W = 2)" abs_err_L16=errs[1] abs_err_L32=errs[2] abs_err_L64=errs[3]

        # Analytic consistency of the two closed forms themselves.
        for hw in (0.25, 0.5, 1.0, 2.0, 4.0)
            @test canyon_sky_view_factor(1.0 * hw) ≈
                  canyon_sky_view_factor(hw, 1.0, 0.0, 1.0) atol = 1e-12
            @test canyon_wallside_sky_view_factor(hw) <
                  canyon_sky_view_factor(hw) <
                  canyon_centreline_sky_view_factor(hw)
            # The strip-average form must reduce to the point values as the
            # strip shrinks onto the centreline and onto the wall.
            @test canyon_sky_view_factor(hw, 1.0, 0.5 - 1e-6, 0.5 + 1e-6) ≈
                  canyon_centreline_sky_view_factor(hw) atol = 1e-9
            @test canyon_sky_view_factor(hw, 1.0, 0.0, 1e-7) ≈
                  canyon_wallside_sky_view_factor(hw) atol = 1e-7
        end
        @test canyon_sky_view_factor(0) == 1
        @test canyon_centreline_sky_view_factor(0) == 1
        @test canyon_wallside_sky_view_factor(0) == 0.5
    end

    # ---------------------------------------------------------------------
    # 6. Isolated-wall limit: F_sky → 1/2
    #
    # A vertical facet standing on an unbounded horizontal plane sees the sky
    # through exactly half its hemisphere. The modelled ground plane is finite,
    # so F_sky is biased high and must fall towards 1/2 as the plane grows.
    # ---------------------------------------------------------------------
    @testset "Isolated wall on a ground plane" begin
        vals = Float64[]
        for n in (11, 21, 41)
            hm = zeros(n, n); c = div(n + 1, 2); hm[c, c] = 2.0
            scene = build_scene(hm, 1.0, 1.0; wall_subdivisions = 1)
            vf = view_factors(scene)
            i = findfirst(k -> scene.facets[k].kind === :wall &&
                               scene.facets[k].normal[1] > 0.5, 1:nfacets(scene))
            push!(vals, vf.F_sky[i])
            @test vf.F_sky[i] > 0.5
            @test closure_error(vf) < 1e-12
        end
        @test vals[2] < vals[1]
        @test vals[3] < vals[2]
        @test vals[3] < 0.55
        @info "Isolated wall sky-view factor → 1/2 as the ground plane grows" n11=vals[1] n21=vals[2] n41=vals[3]
    end

    # ---------------------------------------------------------------------
    # 7. Occlusion
    # ---------------------------------------------------------------------
    @testset "Occlusion by intervening buildings" begin
        # Two facing walls with a blocker of growing height between them.
        function trio(hB)
            hm = zeros(8, 3); hm[2, :] .= 10.0; hm[6, :] .= 10.0
            hB > 0 && (hm[4, :] .= hB)
            return build_scene(hm, 1.0, 1.0; wall_subdivisions = 1)
        end
        Fs = Float64[]
        for hB in (0.0, 2.0, 5.0, 9.0, 20.0)
            scene = trio(hB)
            vf = view_factors(scene)
            ia = findfirst(k -> scene.facets[k].kind === :wall &&
                                scene.facets[k].normal[1] > 0.5 &&
                                scene.facets[k].centroid[1] < 3 &&
                                scene.facets[k].col[2] == 2, 1:nfacets(scene))
            ib = findfirst(k -> scene.facets[k].kind === :wall &&
                                scene.facets[k].normal[1] < -0.5 &&
                                scene.facets[k].centroid[1] > 4 &&
                                scene.facets[k].col[2] == 2, 1:nfacets(scene))
            push!(Fs, vf.F[ia, ib])
            # Occlusion must never break the exact identities.
            @test reciprocity_error(vf) < 1e-12
            @test closure_error(vf) < 1e-12
        end
        @test Fs[1] > 0
        @test Fs[2] < Fs[1]
        @test Fs[3] < Fs[2]
        @test Fs[4] == 0                    # fully blocked
        @test Fs[5] == 0
        @info "Wall-to-wall view factor across a growing blocker" F_no_blocker=Fs[1] F_h2=Fs[2] F_h5=Fs[3] F_h9=Fs[4] F_h20=Fs[5]

        # Direct segment tests against the heightmap.
        scene = trio(9.0)
        @test segment_blocked(scene, (1.5, 1.5, 1.0), (6.5, 1.5, 1.0))
        @test !segment_blocked(scene, (3.5, 1.5, 12.0), (4.5, 1.5, 12.0))
        # A ray leaving the footprint sideways is never blocked (open lateral
        # boundary), but one that dips below the ground plane is.
        @test !ray_blocked(scene, (3.5, 1.5, 12.0), (1.0, 0.0, 0.0), 100.0)
        @test ray_blocked(scene, (3.5, 1.5, 1.0), (0.0, 0.0, -1.0), 100.0)
    end

    # ---------------------------------------------------------------------
    # 8. Shortwave radiosity: conservation and solver agreement
    # ---------------------------------------------------------------------
    @testset "Shortwave radiosity" begin
        hm = zeros(9, 9)
        hm[3:4, 3:5] .= 14.0
        hm[6:7, 5:7] .= 8.0
        scene = build_scene(hm, 2.5, 2.5; wall_subdivisions = 2,
                            materials = default_urban_materials())
        vf = view_factors(scene)
        sun = solar_unit_vector((elevation = 38.0, azimuth = 152.0))

        swd = solve_shortwave(scene, vf, sun; dni = 870.0, dhi = 130.0,
                              solver = :direct)
        swg = solve_shortwave(scene, vf, sun; dni = 870.0, dhi = 130.0,
                              solver = :gauss_seidel)
        swp = solve_shortwave(scene, vf, sun; dni = 870.0, dhi = 130.0,
                              solver = :progressive, tol = 1e-14)

        # Energy conservation: in = absorbed + escaping, to round-off.
        resid = swd.total_incident - swd.total_absorbed - swd.total_escaping
        @test abs(resid) < 1e-8 * swd.total_incident
        @test swd.total_incident > 0
        @test swd.total_absorbed > 0
        @test swd.total_escaping > 0
        @info "Shortwave energy budget" total_incident_W=swd.total_incident total_absorbed_W=swd.total_absorbed total_escaping_W=swd.total_escaping relative_residual=resid/swd.total_incident

        # Direct dense LU vs iterative solvers.
        scale = maximum(swd.radiosity)
        @test maximum(abs, swd.radiosity .- swg.radiosity) < 1e-9 * scale
        @test maximum(abs, swd.radiosity .- swp.radiosity) < 1e-8 * scale
        @info "Solver agreement" max_abs_diff_gauss_seidel=maximum(abs, swd.radiosity .- swg.radiosity) max_abs_diff_progressive=maximum(abs, swd.radiosity .- swp.radiosity) radiosity_scale=scale

        α = facet_albedos(scene)
        E = α .* (swd.direct .+ swd.diffuse_sky)
        Bdir = solve_radiosity_direct(vf.F, α, E)
        Bgs, its, res = solve_radiosity_iterative(vf.F, α, E)
        Bpr, shots, resp = solve_radiosity_iterative(vf.F, α, E;
                                                     method = :progressive,
                                                     tol = 1e-14)
        @test maximum(abs, Bdir .- Bgs) < 1e-9 * scale
        @test maximum(abs, Bdir .- Bpr) < 1e-8 * scale
        @test res < 1e-8
        @test resp < 1e-6
        @test its < 60                      # geometric at rate ≈ max α
        @test shots > 0
        @info "Iterative solvers" gauss_seidel_sweeps=its gauss_seidel_residual=res progressive_shots=shots progressive_residual=resp

        # The Neumann series is the bounce expansion: Gauss-Seidel started at
        # B = E must under-estimate, since it has not yet counted all bounces.
        Bone, _, _ = solve_radiosity_iterative(vf.F, α, E; maxiter = 1)
        @test all(Bone .<= Bdir .+ 1e-12)

        # Inter-reflection is a real, positive contribution — the term the
        # Era-I urban scheme dropped.
        @test any(swd.interreflected .> 1.0)
        @test all(swd.interreflected .>= -1e-12)
        @test all(swd.absorbed .>= -1e-12)
        @test all(swd.radiosity .>= -1e-12)
        canyon_idx = findall(i -> scene.facets[i].kind === :wall,
                             1:nfacets(scene))
        @info "Inter-reflected shortwave on walls" mean_W_per_m2=sum(swd.interreflected[canyon_idx])/length(canyon_idx) max_W_per_m2=maximum(swd.interreflected[canyon_idx])

        # With the sun at the zenith every column presents exactly one
        # horizontal facet to the beam, so the intercepted power is exactly
        # DNI × plan area. A shadow test that double-counts or leaks fails here.
        swz = solve_shortwave(scene, vf, (0.0, 0.0, 1.0); dni = 1000.0, dhi = 0.0)
        @test swz.total_incident ≈ 1000.0 * 9 * 9 * 2.5^2 rtol = 1e-12
    end

    # ---------------------------------------------------------------------
    # 9. Radiative trapping — the headline validation
    # ---------------------------------------------------------------------
    @testset "Radiative trapping: effective canyon albedo" begin
        αmat = 0.2
        Wc = 6; L = 24
        flat = _flat_scene(Wc + 2, L; albedo = αmat)
        vff = view_factors(flat)
        # Zenith sun, no diffuse: a controlled experiment in which the
        # perimeter walls of the test scene receive nothing at all, so the
        # measured number is the canyon's and only the canyon's.
        αflat = effective_albedo(flat, vff, (0.0, 0.0, 1.0); dni = 1000.0,
                                 dhi = 0.0)
        @test αflat ≈ αmat atol = 1e-12

        hws = (0.5, 1.0, 2.0, 4.0)
        αeff = Float64[]
        for hw in hws
            scene = _canyon_scene(Wc, hw * Wc, L; albedo = αmat)
            vf = view_factors(scene)
            sw = solve_shortwave(scene, vf, (0.0, 0.0, 1.0); dni = 1000.0,
                                 dhi = 0.0)
            push!(αeff, sw.effective_albedo)
            # Every canyon solve must still conserve energy exactly.
            @test abs(sw.total_incident - sw.total_absorbed - sw.total_escaping) <
                  1e-8 * sw.total_incident
            # Strictly below the material albedo.
            @test sw.effective_albedo < αmat
        end
        # Monotone decrease with canyon depth.
        for k in 2:length(αeff)
            @test αeff[k] < αeff[k - 1]
        end
        @info "Effective canyon albedo vs H/W (material albedo 0.2, zenith sun)" flat=αflat hw_0p5=αeff[1] hw_1=αeff[2] hw_2=αeff[3] hw_4=αeff[4]

        # Same signature with an oblique sun and a diffuse component. Only
        # `α_eff < α` is asserted here, not monotonicity: with an oblique sun
        # the scene's *perimeter* walls (which face empty space beyond the
        # open lateral boundary) are lit, and they behave like flat surfaces
        # of albedo α. Their area grows with H, so they dilute the canyon
        # signal — an artefact of the finite test domain, not of the solver.
        # The zenith configuration above removes it exactly, because a
        # vertical wall receives nothing from a zenith beam.
        scene1 = _canyon_scene(Wc, 1.0 * Wc, L; albedo = αmat)
        scene2 = _canyon_scene(Wc, 3.0 * Wc, L; albedo = αmat)
        vf1 = view_factors(scene1); vf2 = view_factors(scene2)
        sun = solar_unit_vector((elevation = 60.0, azimuth = 175.0))
        a1 = effective_albedo(scene1, vf1, sun; dni = 800.0, dhi = 150.0)
        a2 = effective_albedo(scene2, vf2, sun; dni = 800.0, dhi = 150.0)
        @test a1 < αmat
        @test a2 < αmat
        @info "Effective albedo with an oblique sun (60° elevation, DHI 150)" hw_1=a1 hw_3=a2

        # Sensitivity to the material albedo. The *absolute* extra absorption
        # α - α_eff grows with α (there is simply more radiation bouncing
        # around to be re-absorbed), but the *ratio* α_eff/α rises with α,
        # because a brighter cavity loses less at each bounce and so a larger
        # share of the trapped radiation eventually finds its way out. The
        # canopy-scale approximation α_eff ≈ αψ/(1 - α(1-ψ)) makes this
        # explicit: the ratio is ψ/(1 - α(1-ψ)), increasing in α.
        sceneA = _canyon_scene(Wc, 2.0 * Wc, L; albedo = 0.1)
        sceneB = _canyon_scene(Wc, 2.0 * Wc, L; albedo = 0.5)
        vfA = view_factors(sceneA); vfB = view_factors(sceneB)
        rA = effective_albedo(sceneA, vfA, (0.0, 0.0, 1.0); dni = 1000.0,
                              dhi = 0.0) / 0.1
        rB = effective_albedo(sceneB, vfB, (0.0, 0.0, 1.0); dni = 1000.0,
                              dhi = 0.0) / 0.5
        @test rA < 1 && rB < 1
        @test rB > rA
        # Roughly half of each material's reflectance is re-absorbed by the
        # canyon at H/W = 2...
        @test 0.4 * 0.1 < 0.1 - rA * 0.1 < 0.6 * 0.1
        # ...and the absolute extra absorption is far larger for the bright
        # material, even though its *relative* trapping is weaker.
        @test 0.1 - rA * 0.1 < 0.5 - rB * 0.5
        @info "Trapping ratio α_eff/α at H/W = 2" albedo_0p1=rA albedo_0p5=rB extra_absorption_0p1=0.1-rA*0.1 extra_absorption_0p5=0.5-rB*0.5
    end

    # ---------------------------------------------------------------------
    # 10. Longwave radiosity and canyon trapping
    # ---------------------------------------------------------------------
    @testset "Longwave radiosity and canyon trapping" begin
        Tsfc = 300.0; Lsky = 280.0; ε = 0.95
        flat_loss = ε * (STEFAN_BOLTZMANN * Tsfc^4 - Lsky)

        Wc = 4; L = 32
        factors = Float64[]
        losses  = Float64[]
        for hw in (0.5, 1.0, 2.0, 4.0)
            scene = _canyon_scene(Wc, hw * Wc, L; emissivity = ε)
            vf = view_factors(scene)
            set_temperatures!(scene, Tsfc)
            lw = solve_longwave(scene, vf; L_sky = Lsky)

            # Energy identity: absorbed = emitted + sky in − sky out.
            resid = lw.total_absorbed -
                    (lw.total_emitted + lw.total_from_sky - lw.total_escaping)
            @test abs(resid) < 1e-8 * lw.total_emitted

            # Mid-canyon floor, away from the open ends.
            jm = div(L, 2)
            idx = [i for i in 1:nfacets(scene)
                   if scene.facets[i].kind === :ground &&
                      scene.facets[i].col[2] == jm]
            loss = sum(vf.area[i] * lw.net_loss[i] for i in idx) /
                   sum(vf.area[i] for i in idx)
            push!(losses, loss)
            push!(factors, loss / flat_loss)

            @test loss > 0                  # a 300 K surface under a 280 W/m²
                                            # sky still cools, just less
            @test loss < flat_loss          # ...substantially less
            # Roofs are exposed: they must lose the flat-plate amount.
            ridx = [i for i in 1:nfacets(scene)
                    if scene.facets[i].kind === :roof]
            @test sum(vf.area[i] * lw.net_loss[i] for i in ridx) /
                  sum(vf.area[i] for i in ridx) ≈ flat_loss rtol = 0.02
        end
        for k in 2:length(factors)
            @test factors[k] < factors[k - 1]
        end
        @info "Longwave canyon trapping: mid-canyon floor net loss (T = 300 K, L_sky = 280 W/m²)" flat_plate_W_per_m2=flat_loss hw_0p5=losses[1] hw_1=losses[2] hw_2=losses[3] hw_4=losses[4] reduction_factor_hw2=factors[3] reduction_factor_hw4=factors[4]

        # Solver agreement in the longwave too.
        scene = _canyon_scene(Wc, 2.0 * Wc, L; emissivity = ε)
        vf = view_factors(scene)
        set_temperatures!(scene, Tsfc)
        l1 = solve_longwave(scene, vf; L_sky = Lsky, solver = :direct)
        l2 = solve_longwave(scene, vf; L_sky = Lsky, solver = :gauss_seidel)
        @test maximum(abs, l1.radiosity .- l2.radiosity) <
              1e-9 * maximum(l1.radiosity)

        # A warm wall heats a cool floor: raising every wall by 10 K must
        # raise the floor's incident longwave and cut its net loss.
        base_net = copy(l1.net_loss)
        Tvec = fill(Tsfc, nfacets(scene))
        for i in 1:nfacets(scene)
            scene.facets[i].kind === :wall && (Tvec[i] = Tsfc + 10)
        end
        set_temperatures!(scene, Tvec)
        l3 = solve_longwave(scene, vf; L_sky = Lsky, solver = :direct)
        gidx = [i for i in 1:nfacets(scene) if scene.facets[i].kind === :ground]
        @test all(l3.net_loss[gidx] .< base_net[gidx])
        @test all(l3.incident[gidx] .> l1.incident[gidx])
        @info "Warming the walls by 10 K" mean_floor_net_loss_before=sum(base_net[gidx])/length(gidx) mean_floor_net_loss_after=sum(l3.net_loss[gidx])/length(gidx)

        # An isothermal closed-ish cavity with a sky at the same effective
        # temperature must have zero net exchange everywhere.
        set_temperatures!(scene, Tsfc)
        l4 = solve_longwave(scene, vf; L_sky = STEFAN_BOLTZMANN * Tsfc^4)
        @test maximum(abs, l4.net_loss) < 1e-9
        @test all(isapprox.(l4.radiosity, STEFAN_BOLTZMANN * Tsfc^4; rtol = 1e-10))
    end

    # ---------------------------------------------------------------------
    # 11. Direct-beam shadowing
    # ---------------------------------------------------------------------
    @testset "Direct beam and shadows" begin
        hm = zeros(11, 11); hm[6, 6] = 15.0
        scene = build_scene(hm, 2.0, 2.0; wall_subdivisions = 1)
        wallidx = Dict{Symbol, Int}()
        for i in 1:nfacets(scene)
            f = scene.facets[i]
            f.kind === :wall || continue
            k = f.normal[1] > 0.5 ? :east : f.normal[1] < -0.5 ? :west :
                f.normal[2] > 0.5 ? :north : :south
            wallidx[k] = i
        end
        @test length(wallidx) == 4

        # Morning: sun in the east-south-east. The east wall is lit, the west
        # wall is not.
        dmorn = solar_unit_vector((elevation = 20.0, azimuth = 110.0))
        σm = sunlit_fractions(scene, dmorn; samples = 3)
        @test σm[wallidx[:east]] == 1
        @test σm[wallidx[:west]] == 0
        @test σm[wallidx[:south]] == 1
        @test σm[wallidx[:north]] == 0

        # Afternoon: azimuth rotated past south. East goes dark, west lights up.
        daft = solar_unit_vector((elevation = 20.0, azimuth = 250.0))
        σa = sunlit_fractions(scene, daft; samples = 3)
        @test σa[wallidx[:east]] == 0
        @test σa[wallidx[:west]] == 1
        @test σa[wallidx[:south]] == 1
        @test σa[wallidx[:north]] == 0

        # Sun below the horizon: nothing is lit anywhere.
        dnight = solar_unit_vector((elevation = -5.0, azimuth = 110.0))
        @test all(sunlit_fractions(scene, dnight) .== 0)
        @test all(direct_beam_irradiance(scene, dnight; dni = 900.0) .== 0)

        # The ground shadow must sit on the side away from the sun and move
        # with the azimuth.
        function shaded_columns(d)
            σ = sunlit_fractions(scene, d; samples = 3)
            return Set(scene.facets[i].col for i in 1:nfacets(scene)
                       if scene.facets[i].kind === :ground && σ[i] < 1)
        end
        low_east = solar_unit_vector((elevation = 15.0, azimuth = 90.0 + 1e-3))
        low_west = solar_unit_vector((elevation = 15.0, azimuth = 270.0 - 1e-3))
        se = shaded_columns(low_east)
        sw = shaded_columns(low_west)
        @test !isempty(se) && !isempty(sw)
        @test all(c -> c[1] < 6, se)        # sun in the east ⇒ shadow to the west
        @test all(c -> c[1] > 6, sw)
        @test isempty(intersect(se, sw))
        @info "Ground shadow of a 15 m tower (2 m cells, 15° sun)" shaded_columns_sun_east=length(se) shaded_columns_sun_west=length(sw)

        # Direct-beam magnitude on an unshaded horizontal facet is DNI·μ₀.
        d45 = solar_unit_vector((elevation = 45.0, azimuth = 180.0))
        E = direct_beam_irradiance(scene, d45; dni = 800.0)
        gi = findfirst(i -> scene.facets[i].kind === :ground &&
                            scene.facets[i].col == (1, 1), 1:nfacets(scene))
        @test E[gi] ≈ 800.0 * sind(45.0) rtol = 1e-12
        # ...and on a wall it is DNI·cos of the angle to the wall normal.
        @test E[wallidx[:south]] ≈ 800.0 * cosd(45.0) rtol = 1e-12
        @test E[wallidx[:north]] == 0

        # `solar_unit_vector` must reproduce SolarGeometry's convention.
        st = Aeolus.SolarGeometry.solar_time(2024, 6, 21, 10, 0, 0.0;
                                             latitude = 52.52, longitude = 13.40)
        p = Aeolus.SolarGeometry.solar_position(st)
        ours = solar_unit_vector(p)
        theirs = Aeolus.SolarGeometry.sun_direction_enu(p)
        @test all(abs.(ours .- theirs) .< 1e-15)
        @test solar_unit_vector((0.0, 0.0, 3.0)) == (0.0, 0.0, 1.0)
        @test solar_unit_vector([1.0, 0.0, 0.0]) == (1.0, 0.0, 0.0)
    end

    # ---------------------------------------------------------------------
    # 12. Coupling back to the LES surface scheme
    # ---------------------------------------------------------------------
    @testset "Aggregation onto the LES grid" begin
        hm = zeros(8, 8)
        hm[3:4, 3:4] .= 10.0
        hm[6, 6] = 20.0
        scene = build_scene(hm, 4.0, 4.0; wall_subdivisions = 2,
                            materials = default_urban_materials())
        vf = view_factors(scene)
        sun = solar_unit_vector((elevation = 50.0, azimuth = 160.0))
        sw = solve_shortwave(scene, vf, sun; dni = 880.0, dhi = 120.0)
        set_temperatures!(scene, 302.0)
        lw = solve_longwave(scene, vf; L_sky = 340.0)
        fc = aggregate_to_grid(scene, vf; sw = sw, lw = lw)

        cell = 4.0 * 4.0
        @test size(fc.sw_absorbed) == (8, 8)
        @test size(fc.lw_net) == (8, 8)
        # Aggregation must be exactly conservative.
        @test sum(fc.sw_absorbed) * cell ≈ sw.total_absorbed rtol = 1e-12
        @test sum(fc.lw_net) * cell ≈ sum(vf.area .* lw.net_loss) rtol = 1e-12
        # Total facet area is recovered by the area density.
        @test sum(fc.area_density) * cell ≈ sum(vf.area) rtol = 1e-12
        @test all(fc.area_density .>= 1 - 1e-12)
        @test maximum(fc.area_density) > 1.5       # walls really do add area
        @test all(0 .<= fc.svf .<= 1 + 1e-12)
        @test all(fc.sw_absorbed .>= 0)

        # The two `*_effective` fields must invert `step_surfaces!`'s algebra
        # exactly, so the existing scheme reproduces the true fluxes.
        mats = scene.materials
        for j in 1:8, i in 1:8
            m = fc.class[i, j]
            α = mats.albedo[m]; ε = mats.emissivity[m]
            @test (1 - α) * fc.sw_effective[i, j] ≈ fc.sw_absorbed[i, j] atol = 1e-9
            @test ε * fc.lw_down_effective[i, j] -
                  ε * STEFAN_BOLTZMANN * 302.0^4 ≈ -fc.lw_net[i, j] atol = 1e-9
        end
        # Built columns are classed as roof, open ones as street/wall.
        @test fc.class[3, 3] == URBAN_ROOF
        @test fc.class[1, 1] == URBAN_STREET
        @info "Grid-aggregated forcing" mean_sw_absorbed_W_per_m2=sum(fc.sw_absorbed)/64 mean_lw_net_W_per_m2=sum(fc.lw_net)/64 max_area_density=maximum(fc.area_density) min_svf=minimum(fc.svf)

        # Bridge from the IBM solid-mask representation.
        mesh = RegularGrid(; size = (8, 8, 8), extent = (32.0, 32.0, 32.0),
                           halo = (1, 1, 1))
        smask = Aeolus.IBM.SolidMask(mesh)
        for k in 1:3, j in 3:4, i in 3:4
            smask.mask[mesh.Hx + i, mesh.Hy + j, mesh.Hz + k] = true
        end
        hb = heightmap_from_solid_mask(smask)
        @test size(hb) == (8, 8)
        @test hb[3, 3] ≈ 3 * mesh.dz
        @test hb[1, 1] == 0
        sc2 = build_scene(mesh, hb; wall_subdivisions = 1)
        @test nfacets(sc2) > 64
        @test sc2.dx ≈ mesh.dx
    end

    # ---------------------------------------------------------------------
    # 13. API guards
    # ---------------------------------------------------------------------
    @testset "API guards and materials" begin
        m = default_urban_materials()
        @test length(m) == 5

        # --- single source of truth --------------------------------------
        # The class greys are DERIVED from the Phase 36 spectral library,
        # not tabulated here. Asserting the derivation is a far stronger
        # property than asserting the five numbers it happens to produce:
        # it cannot silently drift, and it fails loudly if either module's
        # material data changes underneath the other.
        sw, lw = shortwave_bands(), longwave_bands()
        for (idx, cls) in enumerate((:street, :roof, :park, :water, :wall))
            mix = Aeolus.Radiosity.URBAN_CLASS_MIXTURES[cls]
            # rebuild the mosaic independently from the exported API
            comps = Pair[]
            for (name, f) in mix
                push!(comps, (name === :water ? FresnelWater() : name) => f)
            end
            surf = tiled_surface(sw, comps...; lw_spec = lw)
            @test material_albedo(m, idx) ≈ broadband_albedo(surf, 0.7) rtol=1e-12
            @test material_emissivity(m, idx) ≈ broadband_emissivity(surf, lw) rtol=1e-12
            # and the mixture is a genuine partition
            @test sum(f for (_, f) in mix) ≈ 1.0
        end

        # --- physically sensible ranges ----------------------------------
        @test 0.05 < material_albedo(m, URBAN_STREET) < 0.20   # asphalt-dominated
        @test 0.10 < material_albedo(m, URBAN_ROOF)   < 0.35
        @test 0.10 < material_albedo(m, URBAN_PARK)   < 0.30   # vegetation
        @test 0.15 < material_albedo(m, URBAN_WALL)   < 0.40   # render + brick
        @test material_albedo(m, URBAN_WATER) < material_albedo(m, URBAN_STREET)

        # --- capability the old grey table could not express -------------
        # Water is a Fresnel interface, so its albedo climbs steeply toward
        # grazing incidence. A single grey number cannot represent a
        # tenfold swing, and that swing dominates the energy balance of an
        # urban river at either end of the day.
        w_high = default_urban_materials(; μ₀ = 1.0).albedo[Int(URBAN_WATER)]
        w_mid  = default_urban_materials(; μ₀ = 0.5).albedo[Int(URBAN_WATER)]
        w_low  = default_urban_materials(; μ₀ = 0.17).albedo[Int(URBAN_WATER)]
        @test w_high < w_mid < w_low
        @test w_low > 8 * w_high
        @info "water albedo vs solar zenith" overhead=w_high mid=w_mid grazing=w_low

        # Every class brightens somewhat toward grazing incidence — that is
        # the Briegleb direct-beam shape function in the Phase 36 library,
        # and it is real: rough surfaces do get brighter at low sun. But
        # water brightens by a completely different MECHANISM (Fresnel
        # reflection at a smooth interface) and by a completely different
        # AMOUNT. The test is that separation of scale, not independence:
        #   street x1.15, wall x1.24, roof x1.30, park x1.53, water x8.60
        # Vegetation is the strongest of the Lambertian-ish set, as canopies
        # genuinely are.
        ratio(idx) = default_urban_materials(; μ₀ = 0.2).albedo[Int(idx)] /
                     default_urban_materials(; μ₀ = 1.0).albedo[Int(idx)]
        for idx in (URBAN_STREET, URBAN_ROOF, URBAN_PARK, URBAN_WALL)
            @test 1.0 < ratio(idx) < 2.0
        end
        @test ratio(URBAN_WATER) > 5.0
        @test ratio(URBAN_WATER) > 3 * maximum(ratio(i) for i in
                 (URBAN_STREET, URBAN_ROOF, URBAN_PARK, URBAN_WALL))
        @info "albedo sensitivity to solar zenith (mu0 0.2 / 1.0)" street=ratio(URBAN_STREET) roof=ratio(URBAN_ROOF) park=ratio(URBAN_PARK) wall=ratio(URBAN_WALL) water=ratio(URBAN_WATER)

        # --- the metal fraction must depress roof emissivity -------------
        # Metal roofs have eps ~ 0.25 against >= 0.90 for everything else.
        # This is why they reach extreme daytime temperatures: they absorb
        # shortwave like any other roof but cannot radiate it away.
        @test material_emissivity(m, URBAN_ROOF) <
              material_emissivity(m, URBAN_STREET)
        no_metal = merge(Aeolus.Radiosity.URBAN_CLASS_MIXTURES,
                         (roof = (:roof_tile_red => 0.50,
                                  :roof_tile_grey => 0.30,
                                  :asphalt => 0.20),))
        m_nm = default_urban_materials(; mixtures = no_metal)
        @test material_emissivity(m_nm, URBAN_ROOF) >
              material_emissivity(m, URBAN_ROOF) + 0.05
        @info "roof emissivity" with_metal=material_emissivity(m, URBAN_ROOF) without=material_emissivity(m_nm, URBAN_ROOF)

        # --- the table really is derived, not cached ---------------------
        bright = merge(Aeolus.Radiosity.URBAN_CLASS_MIXTURES,
                       (street = (:concrete => 1.0,),))
        @test material_albedo(default_urban_materials(; mixtures = bright),
                              URBAN_STREET) > 0.25

        # Plugs into the shared Era-III surface-optics contract.
        @test m isa Aeolus.RadiationCore.AbstractSurfaceOptics
        @test Aeolus.RadiationCore.surface_albedo(m, URBAN_PARK) ≈
              material_albedo(m, URBAN_PARK)
        @test Aeolus.RadiationCore.surface_emissivity(m, URBAN_PARK) ≈
              material_emissivity(m, URBAN_PARK)

        @test_throws ErrorException UrbanMaterials([1.0], [0.9])
        @test_throws ErrorException UrbanMaterials([0.1], [1.5])
        @test_throws ErrorException UrbanMaterials([0.1, 0.2], [0.9])
        @test_throws ErrorException build_facets(zeros(3, 3), -1.0, 1.0)
        @test_throws ErrorException build_facets(zeros(3, 3), 1.0, 1.0;
                                                 wall_subdivisions = 0)
        # A non-right-handed frame must be rejected: it would silently flip
        # the sign of every exchange area the facet takes part in.
        @test_throws ErrorException Facet((0.0, 0.0, 0.0), _EZ, _EY, _EX,
                                          1.0, 1.0, :ground, 1, 300.0, (1, 1))
        @test_throws ErrorException Facet((0.0, 0.0, 0.0), _EZ, _EX, _EY,
                                          1.0, 1.0, :wibble, 1, 300.0, (1, 1))

        scene = _flat_scene(3, 3)
        vf = view_factors(scene)
        @test_throws ErrorException set_temperatures!(scene, [1.0, 2.0])
        @test_throws ErrorException solve_radiosity_direct(vf.F, [0.1],
                                                           zeros(nfacets(scene)))
        @test_throws ErrorException solve_radiosity_iterative(
            vf.F, facet_albedos(scene), zeros(nfacets(scene));
            method = :nonsense)
        @test_throws ErrorException solar_unit_vector((1.0, 2.0))
        @test_throws ErrorException canyon_sky_view_factor(1.0, 1.0, 1.0, 0.0)
        @test_throws ErrorException view_factor_parallel_rectangles(1, 1, 0)

        # A class map lets parks and water in.
        cm = fill(URBAN_STREET, 3, 3); cm[2, 2] = URBAN_WATER
        fs = build_facets(zeros(3, 3), 1.0, 1.0; class_map = cm)
        @test count(f -> f.material == URBAN_WATER, fs) == 1

        # `exchange_area` is symmetric and self-exchange is zero.
        hm = zeros(4, 4); hm[2, 2] = 5.0
        s2 = build_scene(hm, 2.0, 2.0; wall_subdivisions = 1)
        @test exchange_area(s2, 1, 1) == 0
        @test exchange_area(s2, 1, 5) ≈ exchange_area(s2, 5, 1) atol = 1e-14

        # `show` methods stay printable (they are used in @info diagnostics).
        @test occursin("RadiosityScene", sprint(show, s2))
        @test occursin("ViewFactors", sprint(show, vf))
        @test occursin("UrbanMaterials", sprint(show, m))
        @test occursin("Facet", sprint(show, s2.facets[1]))
    end
end
