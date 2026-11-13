# Phase 56 (P56-b, #182): the Deardorff (1980) 1.5-order prognostic
# SGS TKE closure (src/PrognosticSGS.jl) with the Moeng (1984) coefficients,
#
# The 500-step positivity run and the reduced CBL are validation runs, in
# validation/tke.jl.

using Test
using Aeolus
using Aeolus.Fields: interior_view, set!
using Aeolus.Operators: advance_sgs!, sgs_state, restore_sgs_state!
using Random: Xoshiro, randn

@testset "prognostic SGS TKE (P56-b, #182)" begin

@testset "1. free decay at l = Δ (c_ε = 0.70)" begin
    m = RegularGrid(; size = (4, 4, 4), extent = (40.0, 40.0, 40.0))
    sim = BoussinesqSimulation(m)
    Δ, e0, T = 10.0, 1.0, 600.0
    @test tke_dissipation_coefficient(Δ, Δ) ≈ 0.70
    exact(t) = (e0^-0.5 + 0.70 * t / (2Δ))^-2           # de/dt = −c_ε e^{3/2}/l
    c = TKEClosure(m; e0 = e0, dissipation = :exact)
    for _ in 1:60
        advance_sgs!(c, sim, T / 60)
    end
    @test maximum(abs.(interior_view(c.e) .- exact(T))) / exact(T) < 1e-6   # measured ≤ 1.7e-14
    @test tke_statistics(c).clips == 0
end

@testset "2. shear equilibrium: C_s = c_k^{3/4} c_ε^{-1/4}; critical Ri" begin
    S, Δ = 0.05, 10.0
    eq = 0.1 / 0.7 * Δ^2 * S^2                            # e = (c_k/c_ε) l² S²
    @test tke_equilibrium(S^2, 0.0, Δ) ≈ eq rtol = 1e-12
    # ν_t = c_k Δ √e at that e is Smagorinsky's with C_s = c_k^{3/4} c_ε^{-1/4} = 0.19441
    @test sqrt(0.1 * Δ * sqrt(eq) / (Δ^2 * S)) ≈ 0.1^0.75 * 0.70^-0.25 rtol = 1e-12
    Ric = tke_critical_richardson()
    @test Ric ≈ 1 / (1 + 0.19 / (0.1 * 0.76^2)) rtol = 1e-14
    @test tke_equilibrium(S^2, 0.25 * S^2, Δ) == 0          # above Ri_c: no equilibrium
    # below l = Δ, P_s + P_b = ε is linear in x = l/Δ: the closed-form root
    Ri = 0.2; a = 0.1 * 0.76^2
    x = (1 / Ri - 1 - 0.19 / a) / (2 + 0.51 / a)
    @test tke_equilibrium(S^2, Ri * S^2, Δ) ≈ (x * Δ * sqrt(Ri) * S / 0.76)^2 rtol = 1e-10
end

@testset "contract: pure coefficients, exact rollback" begin
    m = RegularGrid(; size = (8, 8, 8), extent = (2π, 2π, 2π), halo = (3, 3, 3))
    sim = BoussinesqSimulation(m)
    rng = Xoshiro(5)
    for f in (sim.u, sim.v, sim.w)
        interior_view(f) .= 0.3 .* randn(rng, 8, 8, 8)
    end
    c = TKEClosure(m)
    initialize_tke!(c, sim)
    e0 = copy(c.e.data)
    compute_sgs!(c, sim); compute_sgs!(c, sim)
    @test c.e.data == e0                                   # coefficients change no memory
    νt = interior_view(sim.νt)
    @test minimum(νt) >= 0 && all(interior_view(sim.κt) .>= νt)   # κ_t = (1 + 2l/Δ) ν_t
    snap = sgs_state(c)
    advance_sgs!(c, sim, 0.01)
    @test c.e.data != e0
    restore_sgs_state!(c, snap)
    @test c.e.data == e0 && tke_statistics(c).steps == 0   # copied back, not computed
end

end

@testset "wall_distance: floor, lid, and a cube (Danielsson 1980; #567)" begin
    using Aeolus.IBM: SolidMask, wall_distance
    using Aeolus.Grids: RegularGrid
    # no obstacle: the height above the floor, d = (k − 1/2) Δz, and with
    # the lid the nearer of the two (the shortcut without sweeps)
    m = RegularGrid(size = (6, 5, 8), extent = (12.0, 10.0, 4.0))
    sm = SolidMask(m)
    d = wall_distance(sm)
    @test size(d) == (6, 5, 8)
    @test all(d[i, j, k] ≈ (k - 0.5) * 0.5 for i in 1:6, j in 1:5, k in 1:8)
    d = wall_distance(sm; top = true)
    @test all(d[i, j, k] ≈ min(k - 0.5, 8.5 - k) * 0.5 for i in 1:6, j in 1:5, k in 1:8)
    @test all(isinf, wall_distance(sm; bottom = false))
    # a cube of 4 × 4 × 2 cells (4 m) on a 2:2:1 grid standing on the floor:
    # the Euclidean distance from each cell centre to the nearer of the cube
    # and the floor, to round-off (the sweeps propagate exact nearest points)
    Δ = (2.0, 2.0, 1.0); N = (16, 16, 10)
    m = RegularGrid(size = N, extent = N .* Δ)
    sm = SolidMask(m)
    ci, ck = 7:10, 1:4
    sm.mask[m.Hx .+ ci, m.Hy .+ ci, m.Hz .+ ck] .= true
    lo = (first(ci) - 1) .* (Δ[1], Δ[2]); hi = last(ci) .* (Δ[1], Δ[2]); top = last(ck) * Δ[3]
    function exact(i, j, k)
        x, y, z = (i - 0.5) * Δ[1], (j - 0.5) * Δ[2], (k - 0.5) * Δ[3]
        dx = max(lo[1] - x, 0.0, x - hi[1]); dy = max(lo[2] - y, 0.0, y - hi[2]); dz = max(z - top, 0.0)
        return min(sqrt(dx^2 + dy^2 + dz^2), z)
    end
    d = wall_distance(sm)
    solid = [sm.mask[m.Hx + i, m.Hy + j, m.Hz + k] for i in 1:N[1], j in 1:N[2], k in 1:N[3]]
    @test all(d[I] == 0 for I in findall(solid))
    @test maximum(abs(d[i, j, k] - exact(i, j, k)) for i in 1:N[1], j in 1:N[2], k in 1:N[3]
                  if !solid[i, j, k]) < 1e-12
end
