# Era VI, Phase 57 (#169): WallModels — equilibrium wall laws and the
# implicit wall-stress operator.
#
# Physics asserted: Spalding's law and its inversion (round trip, the
# viscous limit, monotone u_τ(U)), the delegation of the rough laws to
# SurfaceExchange (κ, neutral MOST); the operator's momentum budget (the
# momentum removed is the wall stress at the end-of-step velocity times
# area and Δt) and its unconditional stability.

using Test
using Random
using Aeolus
using Aeolus.WallModels
using Aeolus.SurfaceExchange: BusingerDyerBeljaarsHoltslag, von_karman

# -------------------------------------------------------------------------
# The operator
# -------------------------------------------------------------------------

# A small box; the operator acts on the first and last level whatever the
# halos are, so the default (periodic) boundary conditions suffice.
function wall_sim(; N = (8, 6, 5), extent = (8.0, 6.0, 2.5))
    g = RegularGrid(; size = N, extent = extent, halo = (2, 2, 2))
    return BoussinesqSimulation(g)
end

function fill_random!(sim; seed = 1)
    rng = Random.MersenneTwister(seed)
    for f in (sim.u, sim.v, sim.w)
        iv = Aeolus.Fields.interior_view(f)
        iv .= 2 .+ randn(rng, size(iv))
    end
    fill_halos!(sim.u, sim.bcs); fill_halos!(sim.v, sim.bcs); fill_halos!(sim.w, sim.bcs)
    return sim
end

# The expected stress per face, computed independently of the operator:
# column speeds from the start-of-step velocity, the law's drag per
# column, the mean of the two columns' drags on each face, times the
# END-of-step face velocity (the implicit stress).
function expected_impulse(sim0, sim1, law, Δt, k; ν = 1.5e-5)
    g = sim0.mesh
    Nx, Ny, dz = g.Nx, g.Ny, g.dz
    u0, v0 = Aeolus.Fields.interior_view(sim0.u), Aeolus.Fields.interior_view(sim0.v)
    u1, v1 = Aeolus.Fields.interior_view(sim1.u), Aeolus.Fields.interior_view(sim1.v)
    D = [wall_drag(law, hypot((u0[i, j, k] + u0[mod1(i + 1, Nx), j, k]) / 2,
                              (v0[i, j, k] + v0[i, mod1(j + 1, Ny), k]) / 2), dz / 2; ν = ν)
         for i in 1:Nx, j in 1:Ny]
    dA = g.dx * g.dy
    Ix = sum((D[i, j] + D[mod1(i - 1, Nx), j]) / 2 * u1[i, j, k] for i in 1:Nx, j in 1:Ny) * dA * Δt
    Iy = sum((D[i, j] + D[i, mod1(j - 1, Ny)]) / 2 * v1[i, j, k] for i in 1:Nx, j in 1:Ny) * dA * Δt
    return Ix, Iy
end

@testset "Era VI — WallModels (#169)" begin

@testset "Spalding: round trip, viscous limit, monotone in U" begin
    law = SpaldingLaw()
    ys = exp10.(range(-2, 6; length = 50))
    # y⁺ → u⁺ (Newton) → y⁺ (explicit formula)
    @test maximum(y -> abs(spalding_yplus(law, spalding_uplus(law, y)) - y) / y, ys) < 1e-10
    # and through friction_velocity: U = u⁺ u_τ at h = y⁺ ν/u_τ gives back u_τ
    ν, uτ = 1.5e-5, 0.3
    err_u = maximum(ys) do y
        up = spalding_uplus(law, y)
        abs(friction_velocity(law, up * uτ, y * ν / uτ; ν = ν) - uτ) / uτ
    end
    @test err_u < 1e-10
    @test abs(spalding_uplus(law, 0.1) - 0.1) / 0.1 < 1e-3      # u⁺ = y⁺ (measured 1.4e-7)
    # the drag coefficient's U → 0 limit is ν/h
    @test wall_drag(law, 1e-9, 0.01; ν = 1.5e-2) ≈ 1.5e-2 / 0.01 rtol = 1e-9
    # friction velocity is strictly increasing in U, through every regime
    Us = exp10.(range(-6, 2; length = 100))
    @test all(diff([friction_velocity(law, U, 0.01; ν = ν) for U in Us]) .> 0)
    @test_throws ArgumentError friction_velocity(law, 1.0, 0.01)          # ν is required
end

@testset "Delegation: rough laws use SurfaceExchange's κ and ψ_m" begin
    U, h, z0 = 2.0, 2.5, 0.05
    ref = von_karman() * U / log(h / z0)
    @test friction_velocity(RoughLogLaw(z0), U, h) ≈ ref
    @test wall_stress(RoughLogLaw(z0), U, h) ≈ ref^2
    # neutral MOST (L = Inf) is the rough log law; stable air mixes less
    sf = BusingerDyerBeljaarsHoltslag()
    @test abs(friction_velocity(MOSTWallLaw(sf, z0, Inf), U, h) - ref) / ref < 1e-12
    @test friction_velocity(MOSTWallLaw(sf, z0, 5.0), U, h) < ref
end

@testset "Operator: momentum removed = Σ τ_w ΔA Δt with the end-of-step velocity" begin
    for law in (SpaldingLaw(), RoughLogLaw(0.01))
        sim = fill_random!(wall_sim())
        g = sim.mesh
        ref = deepcopy(sim)
        Δt = 0.7
        F = apply_wall_stress!(sim, WallModel(law; ν = 1.5e-5), Δt; side = :both)
        dV = g.dx * g.dy * g.dz
        u0, u1 = Aeolus.Fields.interior_view(ref.u), Aeolus.Fields.interior_view(sim.u)
        v0, v1 = Aeolus.Fields.interior_view(ref.v), Aeolus.Fields.interior_view(sim.v)
        ΔMx = sum(u1 .- u0) * dV
        ΔMy = sum(v1 .- v0) * dV
        Ix = sum(expected_impulse(ref, sim, law, Δt, k)[1] for k in (1, g.Nz))
        Iy = sum(expected_impulse(ref, sim, law, Δt, k)[2] for k in (1, g.Nz))
        @test abs(ΔMx + Ix) / abs(Ix) < 1e-12 && abs(ΔMy + Iy) / abs(Iy) < 1e-12
        @test F.Fx * Δt ≈ ΔMx rtol = 1e-12
        # levels without a wall are untouched
        @test values_agree(u1[:, :, 2:g.Nz-1], u0[:, :, 2:g.Nz-1])
    end
end

@testset "Operator: implicit, stable at 100× the explicit limit" begin
    law = RoughLogLaw(0.05)
    sim = wall_sim()
    g = sim.mesh
    u = Aeolus.Fields.interior_view(sim.u); v = Aeolus.Fields.interior_view(sim.v)
    u .= 3.0; v .= -1.0
    wm = WallModel(law; ν = 1.5e-5)
    # explicit Euler u ← u (1 − Δt D/Δz) is unstable for Δt D/Δz > 2
    Δt = 100 * g.dz / wall_drag(law, hypot(3.0, 1.0), g.dz / 2; ν = 1.5e-5)
    us, vs = [3.0], [-1.0]
    for n in 1:20
        apply_wall_stress!(sim, wm, Δt)
        push!(us, u[1, 1, 1]); push!(vs, v[1, 1, 1])
    end
    @test all(diff(us) .< 0) && all(>(0), us)      # monotone decay, no sign change
    @test all(i -> us[i] / vs[i] ≈ -3.0, eachindex(us))   # the stress follows the wind
end

end
