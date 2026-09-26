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

end # testset
