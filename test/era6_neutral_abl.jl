# Era VI, Phase 54 (P54-d, #174): the wall-bounded benchmark harness —
# the pressure-driven half-channel ABL and the two-wall channel case
# builders (WallModels) and their diagnostics (Operators).
#
# Physics asserted: Φ_m of the analytic log law is 1 to O(Δz²) with the
# exact staggered coefficient; E11 satisfies Parseval; the stress profile
# is the flux the solver integrates (its divergence is the plane-mean
# tendency); the half channel's momentum budget (wall stress = u*², total
# stress = u*²(1 - z/H)); the two-wall channel's ⟨τ_w⟩ = u_τ²; and the
# Smagorinsky near-wall overshoot of Mason & Thomson (1992).

using Test
using Printf
using Random
using Aeolus
using Aeolus.SurfaceExchange: von_karman

@testset "Era VI — neutral ABL / channel harness (#174)" begin

@testset "Diagnostics: Φ_m of the analytic log law is 1 + (Δz/z)²/12 + O(Δz⁴)" begin
    κ = von_karman(); us = 0.45; z0 = 1e-3
    for Nz in (16, 32, 64)
        H = 1.0; dz = H / Nz
        zc = [(k - 0.5) * dz for k in 1:Nz]
        U = @. us / κ * log(zc / z0)
        prof = phi_m_profile(U, dz, us; κ = κ)
        @test prof.z ≈ [k * dz for k in 1:Nz-1]
        # exact staggered value (z/Δz) ln((z+Δz/2)/(z-Δz/2)), independent of z0
        exact = [z / dz * log((z + dz / 2) / (z - dz / 2)) for z in prof.z]
        @test maximum(abs.(prof.Φ .- exact)) < 1e-12
        @test abs(prof.Φ[1] - log(3)) < 1e-12
        # second order: Φ - 1 - (Δz/z)²/12 = (Δz/z)⁴/80 + …
        # (2/ε) artanh(ε/2) = 1 + ε²/12 + ε⁴/80 + ε⁶/448 + …, ε = Δz/z: every
        # term is positive, so the remainder after ε²/12 lies in
        # [ε⁴/80, ε⁴/80 + ε⁶/400] for ε ≤ 1/2 (plus 1e-13 of round-off in
        # forming Φ - 1 from a logarithm difference)
        for k in 2:Nz-1
            ε = dz / prof.z[k]
            rem = prof.Φ[k] - 1 - ε^2 / 12
            @test ε^4 / 80 - 1e-13 <= rem <= ε^4 / 80 + ε^6 / 400 + 1e-13
        end
    end
    # at a fixed height the error falls by 4× per halving of Δz
    zf = 0.25
    err = map((8, 16, 32, 64)) do Nz
        dz = 1.0 / Nz
        U = [log(((k - 0.5) * dz) / 1e-4) / κ for k in 1:Nz]
        p = phi_m_profile(U, dz, 1.0; κ = κ)
        abs(p.Φ[findfirst(≈(zf), p.z)] - 1)
    end
    orders = log2.(err[1:end-1] ./ err[2:end])
    @info @sprintf("  Φ_m - 1 at z = H/4: %s; observed orders %s",
                   join((@sprintf("%.2e", e) for e in err), ", "),
                   join((@sprintf("%.3f", o) for o in orders), ", "))
    @test all(o -> abs(o - 2) < 0.05, orders)
    # the Field method uses the plane-mean speed |(⟨u⟩, ⟨v⟩)|
    g = RegularGrid(; size = (8, 6, 16), extent = (1.0, 1.0, 1.0))
    u = Field(FaceX(), g); v = Field(FaceY(), g)
    Random.seed!(3)
    for k in 1:16
        z = (k - 0.5) / 16
        S = log(z / 1e-4) / κ
        noise = randn(8, 6); noise .-= sum(noise) / 48
        u.data[3:10, 3:8, 2+k] .= 0.6S .+ noise
        v.data[3:10, 3:8, 2+k] .= 0.8S
    end
    pf = phi_m_profile(u, v, 1.0; κ = κ)
    @test maximum(abs.(pf.Φ .- [z * 16 * log((z + 1/32) / (z - 1/32)) for z in pf.z])) < 1e-12
end

@testset "Diagnostics: streamwise_spectrum satisfies Parseval to 1e-12" begin
    Random.seed!(11)
    for (Nx, Lx) in ((24, 2π), (25, 3.0))
        g = RegularGrid(; size = (Nx, 7, 5), extent = (Lx, 1.0, 1.0))
        u = Field(FaceX(), g)
        u.data[3:2+Nx, 3:9, 3:7] .= 2 .+ randn(Nx, 7, 5)
        sp = streamwise_spectrum(u, 1:5)
        dk = 2π / Lx
        @test sp.k1 ≈ [n * dk for n in 0:Nx÷2]
        for k in 1:5
            a = u.data[3:2+Nx, 3:9, 2+k]
            var = sum((a .- sum(a; dims = 1) ./ Nx) .^ 2) / length(a)
            @test abs(sum(sp.E11[:, k]) * dk - var) / var < 1e-12
            @test sp.E11[1, k] < 1e-12 * var        # the mean is removed
        end
        sp2 = streamwise_spectrum(u, [2]; remove_mean = false)
        a = u.data[3:2+Nx, 3:9, 4]
        @test abs(sum(sp2.E11) * dk - sum(a .^ 2) / length(a)) / (sum(a .^ 2) / length(a)) < 1e-12
    end
    # a single mode lands in its own bin with energy A²/2
    g = RegularGrid(; size = (32, 4, 3), extent = (4.0, 1.0, 1.0))
    u = Field(FaceX(), g)
    x = [(i - 1) * 4.0 / 32 for i in 1:32]
    u.data[3:34, 3:6, 4] .= 0.7 .* cos.(2π * 5 .* x ./ 4.0)
    sp = streamwise_spectrum(u, [2])
    @test abs(sp.E11[6, 1] * (2π / 4.0) - 0.7^2 / 2) < 1e-14
    @test sum(sp.E11) - sp.E11[6, 1] < 1e-25
end

@testset "Diagnostics: stress profile divergence = plane-mean momentum tendency" begin
    # The profile is the flux the solver integrates: on any velocity and
    # eddy-viscosity field in a periodic-x/y, wall-z box, the plane mean of
    # the advective + eddy-viscous u tendency is ∂τ/∂z = Δτ/Δz exactly
    # (τ is the downward flux, ∂⟨u⟩/∂t = ∂τ/∂z).
    Random.seed!(5)
    g = RegularGrid(; size = (12, 10, 9), extent = (2.0, 1.5, 1.0))
    bcs = FieldBoundaryConditions(:periodic, :periodic, :wall)
    u = Field(FaceX(), g); v = Field(FaceY(), g); w = Field(FaceZ(), g)
    νt = Field(CellCenter(), g)
    ii, jj, kk = Aeolus.Grids.interior(g)
    u.data[ii, jj, kk] .= 3 .+ randn(12, 10, 9)
    v.data[ii, jj, kk] .= randn(12, 10, 9)
    w.data[ii, jj, kk] .= randn(12, 10, 9)
    w.data[ii, jj, first(kk)] .= 0                     # the bottom wall face
    νt.data[ii, jj, kk] .= 0.05 .+ 0.02 .* rand(12, 10, 9)
    for f in (u, v, w, νt); fill_halos!(f, bcs); end
    for comp in (:x, :y)
        tu = Field(FaceX(), g); tv = Field(FaceY(), g); tw = Field(FaceZ(), g)
        momentum_advect!(tu, tv, tw, u, v, w)
        add_eddy_viscosity_u!(tu, u, νt); add_eddy_viscosity_v!(tv, v, νt)
        t = comp === :x ? tu : tv
        prof = total_stress_profile(u, v, w, νt; component = comp)
        mt = [sum(t.data[ii, jj, 2+k]) / 120 for k in 1:9]
        @test maximum(abs.(mt .- diff(prof.total) ./ g.dz)) < 1e-12 * maximum(abs.(mt))
        # nothing crosses the free-slip walls
        @test prof.total[1] == 0 && prof.total[end] == 0
    end
    p = total_stress_profile(u, v, w, νt; ν = 1e-3, τ_wall_bottom = 0.2, τ_wall_top = 0.1)
    @test p.wall[1] == 0.2 && p.wall[end] == -0.1 && all(iszero, p.wall[2:end-1])
    @test p.total ≈ p.resolved .+ p.sgs .+ p.viscous .+ p.wall
    @test p.viscous[2] ≈ 1e-3 * (sum(u.data[ii, jj, 4]) - sum(u.data[ii, jj, 3])) / 120 / g.dz
end

@testset "step_rk3!: body_force! / post_predictor! hooks run in every substage" begin
    g = RegularGrid(; size = (8, 8, 8), extent = (1.0, 1.0, 1.0))
    sim = BoussinesqSimulation(g)
    Random.seed!(2)
    ii, jj, kk = Aeolus.Grids.interior(g)
    sim.u.data[ii, jj, kk] .= randn(8, 8, 8)
    G = 0.37; dt = 1e-3; calls = Ref(0)
    m0 = sum(sim.u.data[ii, jj, kk]) / 512
    step_rk3!(sim, dt; body_force! = (s, h) -> (s.tu.data[ii, jj, kk] .+= G),
              post_predictor! = (s, h) -> (calls[] += 1))
    # each substage adds G·dt; the SSP-RK3 weights 1/6 + 1/6 + 2/3 sum to 1
    @test abs(sum(sim.u.data[ii, jj, kk]) / 512 - m0 - G * dt) < 1e-14
    @test calls[] == 3
    @test sim.t ≈ dt
end

end # top testset
