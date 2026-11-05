# Phase 49 — facet temperatures coupled to the Era III radiosity solver (#28).
#
# Every radiosity facet carries its own UrbanFacet conduction column and
# skin temperature, and the longwave exchange between facets is implicit.
# Validation is against things the code does not contain: the closed-form
# net exchange between infinite parallel grey plates, the radiative
# equilibrium of re-radiating (adiabatic) walls in an infinite street
# canyon, Hottel's reciprocity of total exchange areas, and exact energy
# bookkeeping of the coupled exchange (in an enclosure and under a sky).

using Test, LinearAlgebra
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
    @test maximum(abs, N .+ lw.net_loss) < 1e-10
    A, ε, J = vf.area, r.emissivity, r.dN_dE
    S = (A .* J .* ε') .* (1 .- I(n))                 # Hottel reciprocity
    @test maximum(abs, S - S') / maximum(abs, S) < 1e-12
    # the response-based solve (#138) closes the scene budget
    lwr = solve_longwave(sc, vf, r; L_sky = 300.0)
    @test maximum(abs, lwr.net_loss .- lw.net_loss) < 1e-10
    @test lwr.total_absorbed ≈ lwr.total_emitted + lwr.total_from_sky - lwr.total_escaping rtol = 1e-12
    # infinite parallel grey plates, closed form with D = 1/ε1 + 1/ε2 − 1
    ε1, ε2 = 0.9, 0.6
    D = 1 / ε1 + 1 / ε2 - 1
    s2 = analytic_scene((:wall, :wall), UrbanMaterials([0.2, 0.2], [ε1, ε2]), (1, 2), (1.0, 1.0))
    r2 = longwave_response(s2, plates_vf())
    @test r2.dN_dE ≈ [-1/(ε1*D) 1/(ε2*D); 1/(ε1*D) -1/(ε2*D)] rtol = 1e-14
end

# ===========================================================================
@testset "an isolated flat facet reduces to step_surface!" begin
    mats = UrbanMaterials([0.15], [0.95])
    s1 = analytic_scene((:roof,), mats, (1,), (1.0,))
    vf1 = ViewFactors{Float64}(zeros(1, 1), [1.0], [1.0])
    u = UrbanFacetScene(s1, vf1; T = 300.0, options = f -> (T_interior = 295.0,))
    ref = UrbanFacet(:roof, example_roof_layers(); T = 300.0, T_interior = 295.0,
                     albedo = 0.15, emissivity = 0.95)
    worst = 0.0; seb = 0.0
    for k in 1:12
        f = SurfaceForcing(sw_down = 0.0, lw_down = 310.0, T_air = 288.0, U = 3.0)
        a = step_urban_facets!(u, f, 900.0)[1]
        step_surface!(ref, f, 900.0)
        worst = max(worst, abs(surface_temperature(u.facets[1]) - surface_temperature(ref)))
        seb = max(seb, abs(a.Rn - a.H - a.LE - a.G))
    end
    @test seb < 1e-10
    # only difference: analytic dRn/dT vs a δT = 0.01 K one-sided difference
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
        u = UrbanFacetScene(sc, plates_vf(); T = [330.0, 280.0], layers = f -> layer, θ = 1.0,
                            options = f -> (h_convective = 0.0,
                                            interior = fixed_temperature_boundary(
                                                f === sc.facets[1] ? Ta : Tb)))
        for _ in 1:400
            step_urban_facets!(u, still_air, 1800.0)
        end
        # (Ta−T1)/R = σ(T1⁴−T2⁴)/D = (T2−Tb)/R
        T1 = newton(t -> (Ta - t) / Rl - σSB * (t^4 - (Tb + Ta - t)^4) / D, 310.0)
        T1m, T2m = facet_temperatures(u)
        @test T1m ≈ T1 atol = 1e-6
        @test T2m ≈ Tb + Ta - T1 atol = 1e-6
        @test -u.lw_net[1] ≈ (Ta - T1) / Rl rtol = 1e-8
    end

    @testset "adiabatic plates: exact conservation, relaxation to the mean" begin
        make() = UrbanFacetScene(sc, plates_vf(); T = [320.0, 270.0], layers = f -> layer, θ = 1.0,
                                 options = f -> (h_convective = 0.0,
                                                 interior = heat_flux_boundary(0.0)))
        u = make()
        E0 = total_heat(u)
        worst_sum = 0.0
        for _ in 1:200
            step_urban_facets!(u, still_air, 1800.0)
            worst_sum = max(worst_sum, abs(sum(u.vf.area .* u.lw_net)))
        end
        @test worst_sum < 1e-9
        @test abs(total_heat(u) - E0) < 1e-6 * E0
        @test all(abs.(facet_temperatures(u) .- 295.0) .< 1e-6)
    end
end

# ===========================================================================
@testset "infinite canyon: re-radiating walls" begin
    # Adiabatic walls, black floor at T_g, sky L: σT_w⁴ = (σT_g⁴ + L)/2 for
    # any H/W and wall emissivity; the floor loses (1 + ψ)/2 (σT_g⁴ − L).
    Td, L = 300.0, 280.0
    floor_layer = [WallLayer(:concrete, 0.10; ncells = 10)]
    wall_layer = [WallLayer(:concrete, 0.02; ncells = 4)]
    R = thermal_resistance(floor_layer)
    HW, εw = 1.0, 0.7
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
    @test maximum(abs.(Ts .- [Tg, Tw, Tw])) < 1e-6
    @test -u.lw_net[1] ≈ (1 + ψf) / 2 * (σSB * Tg^4 - L) rtol = 1e-8
end

# ===========================================================================
@testset "3-D canyon: per-facet temperatures and the coupled energy budget" begin
    # two 20 m slab blocks either side of a 10 m street (H/W = 2)
    hm = zeros(8, 12); hm[1:3, :] .= 20; hm[6:8, :] .= 20
    sc = build_scene(hm, 5.0, 5.0; wall_subdivisions = 4)
    vf = view_factors(sc)
    opts(f) = f.kind === :wall ? (h_convective = 10.0, T_interior = 295.0) :
              f.kind === :roof ? (T_interior = 295.0,) : (T_interior = 290.0,)
    newscene(T0) = UrbanFacetScene(sc, vf; T = T0, options = opts)
    air = SurfaceForcing(lw_down = 310.0, T_air = 291.0, U = 2.0, z_ref = 30.0)

    @testset "applied longwave equals the radiosity solution of the applied emission" begin
        u = newscene(293.0)
        worst_lin = 0.0; worst_seb = 0.0; worst_store = 0.0; worst_hwall = 0.0
        for k in 1:4
            H0 = [column_heat_content(f.column) for f in u.facets]
            Tn = facet_temperatures(u)
            fl = step_urban_facets!(u, air, 900.0)
            for (i, f) in enumerate(u.facets)
                f.kind === :wall || continue
                worst_hwall = max(worst_hwall,
                    abs(fl[i].H - 10.0 * (Tn[i] + u.skin_increment[i] - 291.0)))
            end
            Teff = (u.lw_emission ./ ([f.emissivity for f in u.facets] .* σSB)) .^ 0.25
            sc2 = deepcopy(sc); set_temperatures!(sc2, Teff)
            lw = solve_longwave(sc2, vf; L_sky = 310.0)
            worst_lin = max(worst_lin, maximum(abs, u.lw_net .+ lw.net_loss))
            for (i, f) in enumerate(u.facets)
                worst_seb = max(worst_seb, abs(fl[i].Rn - fl[i].H - fl[i].LE - fl[i].G))
                ΔH = column_heat_content(f.column) - H0[i]
                worst_store = max(worst_store,
                                  abs(ΔH - (fl[i].G - interior_heat_flux(f)) * 900.0) / abs(ΔH + 1))
            end
        end
        @test worst_hwall < 1e-8
        @test worst_lin < 1e-9
        @test worst_seb < 1e-9
        @test worst_store < 1e-9
    end

    @testset "shortwave: absorbed equals the radiosity solution; aggregation conserves" begin
        u = newscene(293.0)
        sun = (0.3, -0.2, 0.93)
        swsol = solve_shortwave(sc, vf, sun; dni = 800.0, dhi = 120.0)
        fl = step_urban_facets!(u, air, 600.0; sun = sun, dni = 800.0, dhi = 120.0)
        @test u.sw_absorbed ≈ swsol.absorbed rtol = 1e-12
        agg = aggregate_facet_fluxes(u, fl)
        @test sum(agg.H) * sc.dx * sc.dy ≈ sum(vf.area .* [f.H for f in fl]) rtol = 1e-12
    end

    @testset "matrix-free GMRES solve matches the dense LU (#138)" begin
        US = Aeolus.UrbanSurface
        sun = (0.3, -0.2, 0.93)
        ui = newscene(293.0); ud = newscene(293.0)
        worst_T = 0.0
        for k in 1:4
            kw = k <= 2 ? (sun = sun, dni = 800.0, dhi = 120.0) : (;)
            step_urban_facets!(ui, air, 900.0; solver = :iterative, kw...)
            step_urban_facets!(ud, air, 900.0; solver = :direct, kw...)
            worst_T = max(worst_T, maximum(abs, facet_temperatures(ui) .- facet_temperatures(ud)))
        end
        @test worst_T < 1e-10
        # few iterations suffice because the system is diagonally dominant
        s = US._prepare_scene_step(newscene(293.0), air, 900.0)
        G = s.J .* s.dEdT'
        @test maximum(s.b[i] * (sum(abs, view(G, i, :)) - abs(G[i, i])) for i in eachindex(s.b)) < 1
        # with interior nodes (one zone free, one pinned): same solution,
        # and each zone's energy balance holds on the GMRES solution
        un = newscene(293.0)
        blk(f) = f.kind === :ground ? 0 : f.centroid[1] < 20.0 ? 1 : 2
        node = [blk(f) for f in sc.facets]
        lhs = [2.0e5, 3.0e5]; R = [1.0e4, -5.0e3]
        sI = US._prepare_scene_step(un, air, 900.0; node, sun = sun, dni = 800.0, dhi = 120.0)
        sD = US._prepare_scene_step(un, air, 900.0; node, sun = sun, dni = 800.0, dhi = 120.0,
                                    solver = :direct)
        args = (; lhs, R, pinned = [false, true], y = [0.3, -0.2], Φ = [500.0, -200.0])
        xI, yI, ΦI = US._solve_scene_step(sI; args...)
        xD, yD, ΦD = US._solve_scene_step(sD; args...)
        @test maximum(abs, xI .- xD) < 1e-10
        @test maximum(abs, ΦI .- ΦD) < 1e-8 * max(1.0, maximum(abs, ΦD))
        GI = sI.J .* sI.dEdT'
        c = GI * xI .- diag(GI) .* xI
        zres = maximum(1:2) do k
            into = sum(sI.area[i] * (sI.q[i] - (sI.F0[i] + sI.e[i] * c[i] + sI.g[i] * yI[k]))
                       for i in eachindex(node) if node[i] == k)
            abs(lhs[k] * yI[k] - (R[k] + ΦI[k] + into)) / max(abs(lhs[k] * yI[k]), 1.0)
        end
        @test zres < 1e-10
    end
end

end # testset
