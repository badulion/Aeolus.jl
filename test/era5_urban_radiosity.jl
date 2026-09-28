# Phase 49 — facet temperatures coupled to the Era III radiosity solver (#28).
#
# Every radiosity facet carries its own UrbanFacet conduction column and
# skin temperature, and the longwave exchange between facets is implicit.
# Validation is against things the code does not contain: the closed-form
# net exchange between infinite parallel grey plates, the radiative
# equilibrium of re-radiating (adiabatic) walls in an infinite street
# canyon, Hottel's reciprocity of total exchange areas, and exact energy
# bookkeeping of the coupled exchange (in an enclosure and under a sky).

using Test, Printf, LinearAlgebra
using Aeolus
using Aeolus.UrbanSurface
using Aeolus.SurfaceExchange
using Aeolus.Radiosity

const σSB = Aeolus.RadiationCore.STEFAN_BOLTZMANN

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# A scene whose view factors are supplied analytically. Facet geometry is
# only metadata here (kind, material, area, attribution column); the
# exchange is entirely in `vf`.
function analytic_scene(kinds, materials::UrbanMaterials, mat_idx, areas)
    fs = [Facet((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                a, 1.0, k, m, 293.15, (1, 1)) for (k, m, a) in zip(kinds, mat_idx, areas)]
    return RadiosityScene{Float64}(fs, zeros(1, 1), 1.0, 1.0, 0.0, 0.0, 0.0, materials)
end

# Two infinite parallel plates: every ray from one reaches the other.
plates_vf() = ViewFactors{Float64}([0.0 1.0; 1.0 0.0], [0.0, 0.0], [1.0, 1.0])

# Infinite street canyon, per unit length (Hottel crossed strings):
# facet 1 floor (width W), 2 and 3 the walls (height H).
function canyon_vf(H, W)
    ψf = sqrt((H / W)^2 + 1) - H / W                 # floor → sky
    Ffw = (1 - ψf) / 2                                 # floor → one wall
    Fww = sqrt((W / H)^2 + 1) - W / H                  # wall → opposite wall
    Fwf = (1 + W / H - sqrt(1 + (W / H)^2)) / 2        # wall → floor
    ψw = 1 - Fww - Fwf                                  # wall → sky
    F = [0.0 Ffw Ffw; Fwf 0.0 Fww; Fwf Fww 0.0]
    return ViewFactors{Float64}(F, [ψf, ψw, ψw], [W, H, H]), ψf, ψw
end

total_heat(u) = sum(u.vf.area[i] * column_heat_content(f.column)
                    for (i, f) in enumerate(u.facets))

# Scalar Newton with a numerical derivative (test-side reference solver).
function newton(g, x; tol = 1e-12, maxit = 100)
    for _ in 1:maxit
        h = 1e-4
        dx = g(x) / ((g(x + h) - g(x - h)) / 2h)
        x -= dx
        abs(dx) < tol && return x
    end
    error("newton did not converge")
end

const still_air = SurfaceForcing(lw_down = 300.0, T_air = 290.0, U = 2.0)

@testset "Facet–radiosity coupling (#28)" begin

# ===========================================================================
@testset "longwave_response: the radiosity system as a linear map" begin
    hm = zeros(9, 5); hm[3, :] .= 10; hm[7, :] .= 10
    sc = build_scene(hm, 5.0, 5.0; wall_subdivisions = 3)
    vf = view_factors(sc)
    n = nfacets(sc)
    set_temperatures!(sc, 280 .+ 30 .* ((1:n) .% 7) ./ 7)
    lw = solve_longwave(sc, vf; L_sky = 300.0)
    r = longwave_response(sc, vf)
    N = r.dN_dE * lw.emitted .+ r.dN_dLsky .* 300.0
    err_lin = maximum(abs, N .+ lw.net_loss)
    A, ε, J = vf.area, r.emissivity, r.dN_dE
    S = (A .* J .* ε') .* (1 .- I(n))                 # A_i J_ij ε_j, off-diagonal
    err_hottel = maximum(abs, S - S') / maximum(abs, S)
    @printf("  response vs solve_longwave: %.2e W/m²; Hottel reciprocity %.2e (%d facets)\n",
            err_lin, err_hottel, n)
    @test err_lin < 1e-10
    @test err_hottel < 1e-12
    # Closure: extra emission by j is absorbed by the others or escapes;
    # never more than emitted.
    col = vec(sum(A .* J; dims = 1)) ./ A
    @test all(-1 - 1e-12 .<= col .<= 1e-12)

    # The response-based solve (one matvec, no factorisation, #138) gives
    # the same LongwaveSolution as the direct radiosity solve, and closes
    # the scene budget absorbed = emitted + sky in - sky out.
    lwr = solve_longwave(sc, vf, r; L_sky = 300.0)
    for fld in (:radiosity, :incident, :absorbed, :net_loss)
        @test maximum(abs, getfield(lwr, fld) .- getfield(lw, fld)) < 1e-10
    end
    @test lwr.emitted == lw.emitted && lwr.from_sky == lw.from_sky
    @test lwr.total_escaping ≈ lw.total_escaping rtol = 1e-12
    @test lwr.total_absorbed ≈ lwr.total_emitted + lwr.total_from_sky - lwr.total_escaping rtol = 1e-12
    sc_e = deepcopy(sc); sc_e.materials.emissivity .*= 0.9
    @test_throws ArgumentError solve_longwave(sc_e, vf, r; L_sky = 300.0)

    # Black scene: J = F - I exactly, so Σ_i A_i J_ij = -A_j F_sky,j.
    scb = build_scene(hm, 5.0, 5.0; wall_subdivisions = 3,
                      materials = uniform_materials(0.2, 1.0))
    rb = longwave_response(scb, vf)
    @test maximum(abs, rb.dN_dE - (vf.F - I)) < 1e-14
    @test maximum(abs, vec(sum(A .* rb.dN_dE; dims = 1)) .+ A .* vf.F_sky) < 1e-12

    # Infinite parallel grey plates, closed form: N_1 = -σ(T1⁴-T2⁴)/D,
    # D = 1/ε1 + 1/ε2 - 1, with σT⁴ = E/ε.
    ε1, ε2 = 0.9, 0.6
    D = 1 / ε1 + 1 / ε2 - 1
    s2 = analytic_scene((:wall, :wall), UrbanMaterials([0.2, 0.2], [ε1, ε2]), (1, 2), (1.0, 1.0))
    r2 = longwave_response(s2, plates_vf())
    @test r2.dN_dE ≈ [-1/(ε1*D) 1/(ε2*D); 1/(ε1*D) -1/(ε2*D)] rtol = 1e-14
    @test all(abs.(r2.dN_dLsky) .< 1e-15)
end

# ===========================================================================
@testset "construction: optics from the scene, one facet per radiosity facet" begin
    hm = zeros(6, 4); hm[3:4, :] .= 12
    sc = build_scene(hm, 4.0, 4.0; wall_subdivisions = 2)
    vf = view_factors(sc)
    u = UrbanFacetScene(sc, vf; T = 290.0,
                        options = f -> f.kind === :wall ? (h_convective = 10.0, albedo = 0.99) : (;))
    @test nfacets(u) == nfacets(sc)
    for (i, f) in enumerate(u.facets)
        m = sc.facets[i].material
        @test f.kind === sc.facets[i].kind
        @test surface_albedo(f) == sc.materials.albedo[m]        # `albedo = 0.99` overridden
        @test surface_emissivity(f) == sc.materials.emissivity[m]
    end
    @test all(f -> f.h_convective == 10.0, filter(f -> f.kind === :wall, u.facets))
    @test facet_temperatures(u) == fill(290.0, nfacets(u))
    @test_throws ArgumentError UrbanFacetScene(sc, vf, u.facets[1:end-1])
    @test_throws ArgumentError step_urban_facets!(u, still_air, 60.0; coupling = :explicit)
    # A change of the scene's material table reaches the facets and the response.
    sc.materials.emissivity .= 0.8
    step_urban_facets!(u, still_air, 60.0)
    @test all(f -> f.emissivity == 0.8, u.facets)
    @test all(u.response.emissivity .== 0.8)
end

# ===========================================================================
@testset "an isolated flat facet reduces to step_surface!" begin
    mats = UrbanMaterials([0.15], [0.95])
    s1 = analytic_scene((:roof,), mats, (1,), (1.0,))
    vf1 = ViewFactors{Float64}(zeros(1, 1), [1.0], [1.0])
    u = UrbanFacetScene(s1, vf1; T = 300.0, options = f -> (T_interior = 295.0,))
    ref = UrbanFacet(:roof, example_roof_layers(); T = 300.0, T_interior = 295.0,
                     albedo = 0.15, emissivity = 0.95)
    worst = 0.0
    for k in 1:48
        f = SurfaceForcing(sw_down = 0.0, lw_down = 310.0, T_air = 288.0, U = 3.0)
        a = step_urban_facets!(u, f, 900.0)[1]
        b = step_surface!(ref, f, 900.0)
        worst = max(worst, abs(surface_temperature(u.facets[1]) - surface_temperature(ref)))
        @test a.Rn - a.H - a.LE ≈ a.G atol = 1e-10
    end
    # Only difference: the coupled step takes dRn/dT = -4εσT³ analytically,
    # step_surface! by a one-sided difference of width δT = 0.01 K.
    @printf("  isolated facet vs step_surface!: max |ΔT_skin| = %.2e K over 12 h\n", worst)
    @test worst < 2e-4
end

# ===========================================================================
@testset "two infinite parallel plates" begin
    ε1, ε2 = 0.9, 0.6
    D = 1 / ε1 + 1 / ε2 - 1
    mats = UrbanMaterials([0.2, 0.2], [ε1, ε2])
    sc = analytic_scene((:wall, :wall), mats, (1, 2), (1.0, 1.0))
    layer = [WallLayer(:concrete, 0.05; ncells = 10)]
    Rl = thermal_resistance(layer)

    @testset "steady radiative exchange between held interiors" begin
        Ta, Tb = 330.0, 280.0
        # θ = 1: the steady state does not depend on θ, and backward Euler
        # damps the stiff thin-cell modes that Crank–Nicolson (amplification
        # → -1 at dt ≫ Δz²/κ) leaves ringing for hundreds of steps.
        u = UrbanFacetScene(sc, plates_vf(); T = [330.0, 280.0], layers = f -> layer, θ = 1.0,
                            options = f -> (h_convective = 0.0,
                                            interior = fixed_temperature_boundary(
                                                f === sc.facets[1] ? Ta : Tb)))
        for _ in 1:400
            step_urban_facets!(u, still_air, 1800.0)
        end
        # Test-side reference: (Ta-T1)/R = σ(T1⁴-T2⁴)/D = (T2-Tb)/R. With
        # T2 = Tb + (Ta - T1), one unknown.
        T1 = newton(t -> (Ta - t) / Rl - σSB * (t^4 - (Tb + Ta - t)^4) / D, 310.0)
        T2 = Tb + Ta - T1
        q = (Ta - T1) / Rl
        T1m, T2m = facet_temperatures(u)
        @printf("  plates: T1 %.6f (ref %.6f), T2 %.6f (ref %.6f), q %.4f (ref %.4f) W/m²\n",
                T1m, T1, T2m, T2, -u.lw_net[1], q)
        @test T1m ≈ T1 atol = 1e-6
        @test T2m ≈ T2 atol = 1e-6
        @test -u.lw_net[1] ≈ q rtol = 1e-8
        @test u.lw_net[2] ≈ q rtol = 1e-8
        @test interior_heat_flux(u.facets[1]) ≈ -q rtol = 1e-8
    end

    @testset "adiabatic plates: exact conservation, relaxation to the mean" begin
        make() = UrbanFacetScene(sc, plates_vf(); T = [320.0, 270.0], layers = f -> layer, θ = 1.0,
                                 options = f -> (h_convective = 0.0,
                                                 interior = heat_flux_boundary(0.0)))
        results = Dict{Symbol, Any}()
        for scheme in (:implicit, :lagged)
            u = make()
            E0 = total_heat(u)
            worst_sum = 0.0
            for _ in 1:200
                step_urban_facets!(u, still_air, 1800.0; coupling = scheme)
                worst_sum = max(worst_sum, abs(sum(u.vf.area .* u.lw_net)))
            end
            results[scheme] = (drift = total_heat(u) - E0, worst_sum = worst_sum,
                               T = facet_temperatures(u))
        end
        im, la = results[:implicit], results[:lagged]
        @printf("  adiabatic plates, 100 h at dt = 1800 s: Σ A·N_lw max %.2e (implicit) vs %.2e W/m² (lagged)\n",
                im.worst_sum, la.worst_sum)
        @printf("  stored-heat drift: %.2e (implicit) vs %.2e J/m² (lagged); T_final %.6f / %.6f K\n",
                im.drift, la.drift, im.T[1], im.T[2])
        # The exchange conserves energy to round-off in the enclosure...
        @test im.worst_sum < 1e-9
        @test abs(im.drift) < 1e-6 * total_heat(make())
        # ...and relaxes to the energy-weighted mean of the two plates.
        @test all(abs.(im.T .- 295.0) .< 1e-6)
        # The lagged (explicit-in-neighbours) coupling does not conserve.
        @test la.worst_sum > 1e3 * im.worst_sum
        @test abs(la.drift) > 1e3 * abs(im.drift)
    end
end

# ===========================================================================
@testset "infinite canyon: re-radiating walls" begin
    # Adiabatic walls with no convection must re-emit everything they
    # absorb. For a black floor at T_g and sky L, symmetry gives
    #   σT_w⁴ = (σT_g⁴ + L)/2         for ANY H/W and ANY wall emissivity
    # (F_wall→floor = F_wall→sky in a symmetric canyon, and a re-radiating
    # surface's radiosity is σT_w⁴ whatever its ε), and the floor loses
    #   (1 + ψ)/2 (σT_g⁴ - L).
    Td, L = 300.0, 280.0
    floor_layer = [WallLayer(:concrete, 0.10; ncells = 10)]
    wall_layer = [WallLayer(:concrete, 0.02; ncells = 4)]
    R = thermal_resistance(floor_layer)
    for (HW, εw) in ((0.5, 0.9), (1.0, 0.7), (2.0, 0.95))
        vf, ψf, _ = canyon_vf(HW * 10.0, 10.0)
        mats = UrbanMaterials([0.1, 0.3], [1.0, εw])
        sc = analytic_scene((:ground, :wall, :wall), mats, (1, 2, 2), vf.area)
        u = UrbanFacetScene(sc, vf; T = 290.0, θ = 1.0,
                            layers = f -> f.kind === :ground ? floor_layer : wall_layer,
                            options = f -> f.kind === :ground ?
                                (h_convective = 0.0, T_interior = Td) :
                                (h_convective = 0.0, interior = heat_flux_boundary(0.0)))
        for _ in 1:600
            step_urban_facets!(u, SurfaceForcing(lw_down = L), 3600.0)
        end
        Tg = newton(t -> (Td - t) / R - (1 + ψf) / 2 * (σSB * t^4 - L), 290.0)
        Tw = ((σSB * Tg^4 + L) / 2 / σSB)^0.25
        Ts = facet_temperatures(u)
        @printf("  H/W %.1f, ε_w %.2f: T_g %.6f (ref %.6f), T_w %.6f / %.6f (ref %.6f) K\n",
                HW, εw, Ts[1], Tg, Ts[2], Ts[3], Tw)
        @test Ts[1] ≈ Tg atol = 1e-6
        @test Ts[2] ≈ Tw atol = 1e-6
        @test Ts[3] ≈ Tw atol = 1e-6
        @test -u.lw_net[1] ≈ (1 + ψf) / 2 * (σSB * Tg^4 - L) rtol = 1e-8
        @test all(abs.(u.lw_net[2:3]) .< 1e-8)
    end
end

# ===========================================================================
@testset "3-D canyon: per-facet temperatures and the coupled energy budget" begin
    # Two 20 m slab blocks either side of a 10 m street (H/W = 2), 60 m long.
    hm = zeros(8, 12); hm[1:3, :] .= 20; hm[6:8, :] .= 20   # street = columns 4:5
    sc = build_scene(hm, 5.0, 5.0; wall_subdivisions = 4)
    vf = view_factors(sc)
    opts(f) = f.kind === :wall ? (h_convective = 10.0, T_interior = 295.0) :
              f.kind === :roof ? (T_interior = 295.0,) : (T_interior = 290.0,)
    newscene(T0) = UrbanFacetScene(sc, vf; T = T0, options = opts)
    air = SurfaceForcing(lw_down = 310.0, T_air = 291.0, U = 2.0, z_ref = 30.0)

    @testset "applied longwave equals the radiosity solution of the applied emission" begin
        u = newscene(293.0)
        worst_lin = 0.0; worst_seb = 0.0; worst_store = 0.0; worst_budget = 0.0
        worst_hwall = 0.0
        for k in 1:24
            H0 = [column_heat_content(f.column) for f in u.facets]
            Tn = facet_temperatures(u)
            fl = step_urban_facets!(u, air, 900.0)
            # Walls use H = h_c (T_s - T_air), linear in T_s, so the sensible
            # heat returned (the SEB residual Rn - LE - G) must equal h_c
            # times the θ-weighted skin temperature the step applied. This
            # is independent of how H is assembled (per-facet SEB closure
            # below is exact by construction and is only a bookkeeping check).
            for (i, f) in enumerate(u.facets)
                f.kind === :wall || continue
                worst_hwall = max(worst_hwall,
                    abs(fl[i].H - 10.0 * (Tn[i] + u.skin_increment[i] - 291.0)))
            end
            # Re-solve the radiosity system at the temperatures whose emission
            # the step applied: it must reproduce lw_net (the exchange is
            # linear in E, so this is exact up to round-off).
            Teff = (u.lw_emission ./ ([f.emissivity for f in u.facets] .* σSB)) .^ 0.25
            sc2 = deepcopy(sc); set_temperatures!(sc2, Teff)
            lw = solve_longwave(sc2, vf; L_sky = 310.0)
            worst_lin = max(worst_lin, maximum(abs, u.lw_net .+ lw.net_loss))
            # Scene budget: Σ A N = absorbed from sky - escaping to sky.
            A = vf.area
            worst_budget = max(worst_budget, abs(sum(A .* u.lw_net) -
                               (lw.total_absorbed - lw.total_emitted)))
            for (i, f) in enumerate(u.facets)
                worst_seb = max(worst_seb, abs(fl[i].Rn - fl[i].H - fl[i].LE - fl[i].G))
                ΔH = column_heat_content(f.column) - H0[i]
                worst_store = max(worst_store,
                                  abs(ΔH - (fl[i].G - interior_heat_flux(f)) * 900.0) / abs(ΔH + 1))
            end
        end
        @printf("  3-D canyon (%d facets): |lw_net - radiosity(E_applied)| ≤ %.2e W/m², scene LW budget %.2e W\n",
                nfacets(u), worst_lin, worst_budget)
        @printf("  per-facet SEB closure ≤ %.2e W/m², storage bookkeeping ≤ %.2e (rel.)\n",
                worst_seb, worst_store)
        @printf("  walls: |H - h_c(T_n + x - T_air)| ≤ %.2e W/m²\n", worst_hwall)
        @test worst_hwall < 1e-8
        @test worst_lin < 1e-9
        @test worst_budget < 1e-8 * sum(vf.area)
        @test worst_seb < 1e-9
        @test worst_store < 1e-9
    end

    @testset "shortwave: absorbed equals the radiosity solution; aggregation conserves" begin
        u = newscene(293.0)
        sun = (0.3, -0.2, 0.93)
        swsol = solve_shortwave(sc, vf, sun; dni = 800.0, dhi = 120.0)
        fl = step_urban_facets!(u, air, 600.0; sun = sun, dni = 800.0, dhi = 120.0)
        @test sum(vf.area .* u.sw_absorbed) ≈ swsol.total_absorbed rtol = 1e-12
        @test u.sw_absorbed ≈ swsol.absorbed rtol = 1e-12
        agg = aggregate_facet_fluxes(u, fl)
        cell = sc.dx * sc.dy
        for (name, fac) in ((:H, f -> f.H), (:Rn, f -> f.Rn), (:G, f -> f.G))
            @test sum(getfield(agg, name)) * cell ≈ sum(vf.area .* fac.(fl)) rtol = 1e-12
        end
        @test sum(agg.area_density) * cell ≈ sum(vf.area) rtol = 1e-12
    end

    @testset "entry points: per-facet forcing, precomputed and diffuse-only shortwave" begin
        sun = (0.3, -0.2, 0.93)
        u1 = newscene(293.0); u2 = newscene(293.0); u3 = newscene(293.0)
        fl1 = step_urban_facets!(u1, air, 600.0; sun = sun, dni = 800.0, dhi = 120.0)
        # One identical forcing per facet and a precomputed ShortwaveSolution
        # must give the same step.
        swsol = solve_shortwave(sc, vf, sun; dni = 800.0, dhi = 120.0)
        fl2 = step_urban_facets!(u2, fill(air, nfacets(u2)), 600.0; sw = swsol)
        @test facet_temperatures(u2) == facet_temperatures(u1)
        @test [f.H for f in fl2] == [f.H for f in fl1]
        # Per-facet forcings that disagree on the sky irradiance need L_sky.
        bad = fill(air, nfacets(u3)); bad[1] = SurfaceForcing(lw_down = 300.0, T_air = 291.0)
        @test_throws ArgumentError step_urban_facets!(u3, bad, 600.0)
        # Diffuse sky only (sun = nothing): absorbed = the radiosity solution
        # with dni = 0, and every facet with sky view receives some.
        u4 = newscene(293.0)
        step_urban_facets!(u4, air, 600.0; dhi = 150.0)
        swd = solve_shortwave(sc, vf, (0.0, 0.0, 1.0); dni = 0.0, dhi = 150.0)
        @test u4.sw_absorbed ≈ swd.absorbed rtol = 1e-12
        @test all(u4.sw_absorbed[vf.F_sky .> 0.01] .> 0)
        u5 = newscene(293.0)
        step_urban_facets!(u5, air, 600.0)
        @test all(u5.sw_absorbed .== 0)
    end

    @testset "night cooling: the walls do not share one temperature" begin
        u = newscene(295.0)
        for _ in 1:48                         # 12 h of clear-sky night
            step_urban_facets!(u, air, 900.0)
        end
        Ts = facet_temperatures(u)
        # Walls facing the street (x between the blocks), in the canyon's
        # middle third so the open ends do not matter.
        inner = [i for (i, f) in enumerate(sc.facets) if f.kind === :wall &&
                 (abs(f.centroid[1] - 15.0) < 1e-9 || abs(f.centroid[1] - 25.0) < 1e-9) &&
                 20.0 < f.centroid[2] < 40.0]
        z = [sc.facets[i].centroid[3] for i in inner]
        lo = Ts[inner[z .< 5.0]]; hi = Ts[inner[z .> 15.0]]
        street = [i for (i, f) in enumerate(sc.facets) if f.kind === :ground &&
                  20.0 < f.centroid[2] < 40.0]
        roofs = [i for (i, f) in enumerate(sc.facets) if f.kind === :roof]
        svf_lo = sum(vf.F_sky[inner[z .< 5.0]]) / count(z .< 5.0)
        svf_hi = sum(vf.F_sky[inner[z .> 15.0]]) / count(z .> 15.0)
        @printf("  after 12 h: wall base %.2f K (ψ %.3f), wall top %.2f K (ψ %.3f), street %.2f K, roof %.2f K\n",
                sum(lo) / length(lo), svf_lo, sum(hi) / length(hi), svf_hi,
                sum(Ts[street]) / length(street), sum(Ts[roofs]) / length(roofs))
        # The base of the wall sees less sky and stays warmer than its top.
        @test svf_lo < svf_hi
        @test minimum(lo) > maximum(hi)
        # All facets cooled under a sky colder than the canyon.
        @test all(Ts .< 295.0)
    end

    @testset "time-step convergence of the implicit coupling" begin
        function run(dt, scheme)
            u = newscene(295.0)
            for _ in 1:round(Int, 6 * 3600 / dt)
                step_urban_facets!(u, air, dt; coupling = scheme)
            end
            return facet_temperatures(u)
        end
        ref = run(30.0, :implicit)
        e_im = maximum(abs, run(900.0, :implicit) .- ref)
        e_la = maximum(abs, run(900.0, :lagged) .- ref)
        e_im2 = maximum(abs, run(450.0, :implicit) .- ref)
        @printf("  6 h night, max |T - T(dt=30 s)|: implicit %.4f K (900 s), %.4f K (450 s); lagged %.4f K (900 s)\n",
                e_im, e_im2, e_la)
        @test e_im < 0.01
        @test e_im2 < e_im
        # At the same step, the implicit coupling is more accurate than
        # the lagged one (measured 0.0043 vs 0.0128 K).
        @test e_im < e_la
    end
end

end # testset
