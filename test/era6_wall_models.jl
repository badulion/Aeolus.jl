# Era VI, Phase 57 (#169): WallModels — equilibrium wall laws and the
# implicit wall-stress operator.
#
# Physics asserted: Spalding's law and its inversion (round trip, the
# viscous and logarithmic limits), the delegation of the rough laws to
# SurfaceExchange (κ, ψ_m, the surface-layer solver bit for bit), the
# viscous limit against the resolved cut-cell stress.

using Test
using Printf
using Aeolus
using Aeolus.WallModels
using Aeolus.SurfaceExchange: BusingerDyerBeljaarsHoltslag, solve_surface_layer, von_karman

@testset "Era VI — WallModels (#169)" begin

@testset "Spalding: round trip over y⁺ ∈ [1e-2, 1e6]" begin
    law = SpaldingLaw()
    ys = exp10.(range(-2, 6; length = 200))
    # y⁺ → u⁺ (Newton) → y⁺ (explicit formula)
    err_y = maximum(y -> abs(spalding_yplus(law, spalding_uplus(law, y)) - y) / y, ys)
    # and through friction_velocity: U = u⁺ u_τ at h = y⁺ ν/u_τ gives back u_τ
    ν, uτ = 1.5e-5, 0.3
    err_u = maximum(ys) do y
        up = spalding_uplus(law, y)
        abs(friction_velocity(law, up * uτ, y * ν / uτ; ν = ν) - uτ) / uτ
    end
    @info @sprintf("  Spalding round trip, 200 points: max rel error y⁺ %.1e, u_τ %.1e", err_y, err_u)
    @test err_y < 1e-10
    @test err_u < 1e-10
    # the Lee & Moser (2015) instance is the same law with other constants
    lm = LEE_MOSER_2015
    @test lm.κ == 0.384 && lm.B == 4.27
    @test maximum(y -> abs(spalding_yplus(lm, spalding_uplus(lm, y)) - y) / y, ys) < 1e-10
end

@testset "Spalding: viscous and logarithmic limits; monotone in U" begin
    law = SpaldingLaw()
    κ, B = law.κ, law.B
    up = spalding_uplus(law, 0.1)
    @test abs(up - 0.1) / 0.1 < 1e-3                  # measured 1.4e-7
    # The log law. The issue asked for |u⁺ − (ln y⁺/κ + B)| < 1e-3 at
    # y⁺ = 1e4; that EXPECTATION is wrong for Spalding's formula, not the
    # code: the formula truncates the exponential after the cubic, so at
    # finite y⁺ it lies below the log law in y⁺ by
    # δy⁺ = u⁺ − e^{−κB}(1 + x + x²/2 + x³/6), x = κu⁺, i.e. above it in u⁺
    # by δu⁺ ≈ −δy⁺/(dy⁺/du⁺). At y⁺ = 1e4 that is 2.5e-3 (measured
    # 2.506e-3). Asserted instead: the deviation equals this first-order
    # estimate, and it is below 1e-3 from y⁺ = 1e5 on (measured 7.2e-4,
    # 1.4e-4 at 1e6).
    loglaw(y) = log(y) / κ + B
    for y in (1e4, 1e5, 1e6)
        u = spalding_uplus(law, y)
        x = κ * u
        δy = u - exp(-κ * B) * (1 + x + x^2 / 2 + x^3 / 6)
        δu_est = -δy / (1 + κ * exp(-κ * B) * (exp(x) - 1 - x - x^2 / 2))
        δu = u - loglaw(y)
        @info @sprintf("  Spalding − log law at y⁺ = %.0e: %.3e (first-order estimate %.3e)", y, δu, δu_est)
        @test δu > 0 && abs(δu - δu_est) < 0.02 * δu
        y >= 1e5 && @test abs(δu) < 1e-3
    end
    # friction velocity is strictly increasing in U, through every regime
    ν, h = 1.5e-5, 0.01
    Us = exp10.(range(-6, 2; length = 400))
    uτ = [friction_velocity(law, U, h; ν = ν) for U in Us]
    @test all(diff(uτ) .> 0)
    @test friction_velocity(law, -2.0, h; ν = ν) == friction_velocity(law, 2.0, h; ν = ν)
    @test friction_velocity(law, 0.0, h; ν = ν) == 0
    @test_throws ArgumentError friction_velocity(law, 1.0, h)          # ν is required
end

@testset "Delegation: rough laws use SurfaceExchange's κ and ψ_m" begin
    U, h, z0 = 2.0, 2.5, 0.05
    κ = von_karman()
    ref = κ * U / log(h / z0)
    @test friction_velocity(RoughLogLaw(z0), U, h) == ref
    @test wall_stress(RoughLogLaw(z0), U, h) == ref^2
    # neutral MOST (L = ±Inf) is the rough log law, to 1e-12
    sf = BusingerDyerBeljaarsHoltslag()
    for L in (Inf, -Inf)
        @test abs(friction_velocity(MOSTWallLaw(sf, z0, L), U, h) - ref) / ref < 1e-12
    end
    # stable / unstable: the solver's u_* bit for bit, at ζ = ±0.5.
    # θ_a − θ_s is bisected until the solver's root is ζ = ±0.5 to 1e-6, and
    # the law is built from that solve's L. The law sees ζ' = h/L, which
    # need not be the solver's ζ in the last bit (h/(h/ζ) ≠ ζ for about half
    # of all ζ; L ↦ h/L skips representable values), so among the solves
    # within 1e-10 K of the bisected one the first with h/L == ζ is taken:
    # bit for bit is then the claim that the law computes exactly what the
    # solver computes at the same ζ. For any L it is round-off (1e-14).
    function solve_at(ζt)
        lo, hi = ζt > 0 ? (0.0, 30.0) : (-30.0, 0.0)
        mk(Δ) = solve_surface_layer(sf; U = U, z_m = h, θ_a = 290.0 + Δ, θ_s = 290.0,
                                    z0m = z0, z0h = z0 / 10)
        for _ in 1:80
            Δ = (lo + hi) / 2
            (mk(Δ).ζ < ζt) ? (lo = Δ) : (hi = Δ)
        end
        Δ0 = (lo + hi) / 2
        for n in 0:200
            sol = mk(Δ0 + n * 5e-13)
            h / sol.L == sol.ζ && return sol, mk(Δ0)
        end
        error("no solve with h/L == ζ near ζ = $ζt")
    end
    for ζt in (0.5, -0.5)
        sol, sol0 = solve_at(ζt)
        @test abs(sol.ζ - ζt) < 1e-6 && sol.converged
        @test abs(friction_velocity(MOSTWallLaw(sf, z0, sol0.L), U, h) - sol0.u_star) <
              1e-14 * sol0.u_star
        @test h / sol.L == sol.ζ                       # the law sees the solver's ζ
        law = MOSTWallLaw(sf, z0, sol.L)
        @test friction_velocity(law, U, h) === sol.u_star
        @info @sprintf("  MOST at ζ = %+.1f: u_* = %.15f (log law %.15f)", ζt, sol.u_star, ref)
        @test ζt > 0 ? sol.u_star < ref : sol.u_star > ref
        # and the Obukhov length that reproduces a u_* is that L
        Lm = matching_obukhov_length(sf, sol.u_star, U, h, z0)
        @test abs(Lm - sol.L) / abs(sol.L) < 1e-10
    end
    # the minimum wind of the surface-layer solver applies to the law as well
    sol = solve_surface_layer(sf; U = 0.02, z_m = h, θ_a = 290.0, θ_s = 290.0, z0m = z0, z0h = z0)
    @test friction_velocity(MOSTWallLaw(sf, z0, Inf), 0.02, h) ≈ sol.u_star rtol = 1e-12
    @test wall_drag(MOSTWallLaw(sf, z0, Inf), 0.02, h) ≈ sol.u_star^2 / 0.1 rtol = 1e-12
end

@testset "Laminar limit: Spalding at h⁺ < 1 is the viscous stress νU/h, as CutCell's" begin
    law = SpaldingLaw()
    ν = 1.5e-2
    # the resolved cut-cell stress of a linear profile u = α (z − z_w) at a
    # flat cut-cell floor (as in era2_cutcell.jl): ν α at every stencil
    g = RegularGrid(; size = (4, 4, 32), extent = (1.0, 1.0, 1.0), halo = (2, 2, 2))
    zw = (32 ÷ 4 + 0.7) * g.dz
    geom = Aeolus.CutCell.cut_cell_geometry(g, (x, y, z) -> z - zw)
    st = Aeolus.CutCell.build_ghost_stencils(geom, g)
    α = 2.5
    u = Field(FaceX(), g); v = Field(FaceY(), g); w = Field(FaceZ(), g)
    for k in axes(u.data, 3), j in axes(u.data, 2), i in axes(u.data, 1)
        u.data[i, j, k] = α * ((k - g.Hz - 0.5) * g.dz - zw)
    end
    τc = wall_shear_stress(u, v, w, st, geom, ν)
    # the wall model at each stencil's probe: U = α·probe at h = probe
    worst, hp_max = 0.0, 0.0
    for (m, s) in enumerate(st)
        s.valid || continue
        hprobe = s.probe
        U = α * hprobe
        τm = wall_stress(law, U, hprobe; ν = ν)
        hp_max = max(hp_max, hprobe * sqrt(τm) / ν)
        worst = max(worst, abs(τm - ν * U / hprobe) / (ν * U / hprobe), abs(τm - τc[m]) / τc[m])
    end
    @info @sprintf("  laminar: h⁺ ≤ %.2f, max |τ_Spalding − νU/h|, |τ_Spalding − τ_CutCell| rel %.1e", hp_max, worst)
    @test hp_max < 1
    @test worst < 1e-3
    # and the drag coefficient's U → 0 limit is ν/h
    @test wall_drag(law, 0.0, 0.01; ν = ν) == ν / 0.01
    @test wall_drag(law, 1e-9, 0.01; ν = ν) ≈ ν / 0.01 rtol = 1e-9
end

end
