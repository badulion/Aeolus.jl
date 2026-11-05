# Era VI, Phase 54 (P54-d, #174): the wall-bounded benchmark harness —
# the pressure-driven half-channel ABL and the two-wall channel case
# builders (WallModels) and their diagnostics (Operators).
#
# Physics asserted: Φ_m of the analytic log law is 1 to O(Δz²) with the
# exact staggered coefficient; E11 satisfies Parseval; the stress profile
# is the flux the solver integrates (its divergence is the plane-mean
# tendency); the plane-mean momentum budget of every step is exact. The
# benchmark runs (the half channel's wall stress = u*² and τ(z) = u*²(1 -
# z/H), the Mason & Thomson (1992) overshoot, the two-wall channel's ⟨τ_w⟩
# = u_τ²) are validation runs, in validation/era6_neutral_abl.jl.

using Test
using Printf
using Random
using Aeolus
# the wall-bounded benchmark harness (half channel, channel) is a test fixture (ONE-l, #475)
isdefined(@__MODULE__, :ChannelCases) || include(joinpath(@__DIR__, "fixtures", "channel_cases.jl"))
using .ChannelCases
using Aeolus.SurfaceExchange: von_karman
using Aeolus.WallModels

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

@testset "Builders: initial profile, reserved hooks, options" begin
    κ = von_karman()
    c = half_channel_case(; N = (12, 10, 16), noise = 0.0)
    g = c.sim.mesh
    @test (g.Lx, g.Ly, g.Lz) == (2π, 2π, 1.0)
    @test c.forcing == 1.0 && c.sim.ν == 0 && c.walls === :bottom && c.kind === :half_channel
    @test c.wall.law isa RoughLogLaw && c.wall.law.z0 == 1e-4
    # the initial profile is the law's: U = (u*/κ) ln(z/z0) at the centres,
    # so the first cell carries τ_w = u*² exactly
    U = [c.sim.u.data[3, 3, 2+k] for k in 1:16]
    @test maximum(abs.(U .- [log((k - 0.5) / 16 / 1e-4) / κ for k in 1:16])) < 1e-10
    @test abs(friction_velocity(c.wall.law, U[1], g.dz / 2) - 1) < 1e-12
    ch = channel_case(; Re_τ = 2000, N = 8, noise = 0.0, δ = 0.5, u_τ = 0.2)
    @test ch.sim.ν ≈ 0.2 * 0.5 / 2000 && ch.wall.ν == ch.sim.ν
    @test ch.forcing ≈ 0.2^2 / 0.5 && ch.walls === :both
    Uc = [ch.sim.u.data[3, 3, 2+k] for k in 1:8]
    @test Uc ≈ reverse(Uc)                         # symmetric about the centreline
    @test abs(friction_velocity(ch.wall.law, Uc[1], ch.sim.mesh.dz / 2; ν = ch.sim.ν) - 0.2) < 1e-12
    # reserved hooks of later issues are documented but refuse to run
    # (surface_heat_flux was reserved here; P58-e (#193) implemented it — see era6_collapse.jl)
    @test_throws ArgumentError half_channel_case(; surface_heat_flux = :cooling)
    @test_throws ArgumentError half_channel_case(; bottom_bc = :noslip)
    @test_throws ArgumentError half_channel_case(; θ_walls = (300.0, 290.0))
    @test_throws ArgumentError half_channel_case(; filter = :box)
    # channel_case's bottom_bc = :noslip and θ_walls are implemented by
    # P55-f (#191; exercised by test/era6_relaminarisation.jl since the former
    # test/era6_stratified_channel.jl was removed by ONE-g, #456); only invalid values throw
    @test_throws ArgumentError channel_case(; bottom_bc = :bogus)
    @test_throws ArgumentError channel_case(; θ_walls = (0.0, 1.0))   # θ walls need :noslip
    # a resolved (:noslip) wall has no wall model, so the wall-model keywords
    # of P57-b / P57-c (#183, #189) do not combine with it
    @test_throws ArgumentError channel_case(; bottom_bc = :noslip, h_wm_cells = 2)
    @test_throws ArgumentError channel_case(; bottom_bc = :noslip, filter = :temporal)
    @test_throws ArgumentError channel_case(; bottom_bc = :noslip, wall_model = :nonequilibrium)
    @test_throws ArgumentError channel_case(; walls = :top)
    @test_throws ArgumentError half_channel_case(; formulation = :bogus)
    # closure and wall law are arguments: MOST and a Moeng wall run as given
    m = half_channel_case(; N = 8, wall_law = MOSTWallLaw(1e-3, Inf), formulation = :moeng,
                          closure = nothing)
    @test m.closure === nothing && m.wall.law isa MOSTWallLaw && m.formulation === :moeng
    advance_channel!(m, 1e-3)
    @test m.τ_wall_bottom > 0 && m.nsteps == 1
end

@testset "Harness: the plane-mean momentum budget of every step is exact" begin
    # Lz Δ⟨u⟩ = (G Lz - τ_b - τ_t) Δt per step: the projection, advection
    # and SGS fluxes move no net x-momentum, and the recorded wall stress is
    # the momentum the wall operator removed (SSP-RK3-weighted substages).
    for (mk, stepper) in ((() -> half_channel_case(; N = 12), :rk3),
                          (() -> half_channel_case(; N = 12, stepper = :euler), :euler),
                          (() -> channel_case(; N = (12, 8, 12)), :rk3),
                          (() -> half_channel_case(; N = 12, formulation = :schumann), :rk3))
        c = mk()
        s = c.sim; g = s.mesh
        ii, jj, kk = Aeolus.Grids.interior(g)
        bulk() = sum(view(s.u.data, ii, jj, kk)) / (g.Nx * g.Ny * g.Nz)
        lhs = 0.0; rhs = 0.0; scale = 0.0
        for n in 1:20
            b0 = bulk(); dt = channel_dt(c; cfl = 0.4)
            advance_channel!(c, dt)
            lhs += g.Lz * (bulk() - b0)
            rhs += (c.forcing * g.Lz - c.τ_wall_bottom - c.τ_wall_top) * dt
            scale += c.forcing * g.Lz * dt
        end
        @test abs(lhs - rhs) < 1e-11 * scale
        @test c.stepper === stepper
    end
end

end # top testset
