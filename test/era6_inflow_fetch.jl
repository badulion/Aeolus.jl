# N-inflow-d (#266, epic #246): how far behind a synthetic inflow the
# turbulence is still out of equilibrium.
#
#   1. development_length: the analytic recovery of an exponential approach,
#      an overshoot that returns, profiles in the band from the start or
#      never in it
#   2. FetchStatistics: the moments of a prescribed travelling-wave field
#      (variances a²/2 at the native points, the wall faces' half weight,
#      −u'w' at the centres); a steady field has zero TKE
#   3. total_stress_profile(irange): an x band of a homogeneous field is the
#      plane mean; a field confined to some columns is seen only there
#   4. the open-x half channel (half_channel_case(inflow = …)): the inflow
#      profile on the west face, volume flux in = out, divergence-free
#   5. the CI fetch case: a periodic precursor at Δ_p is the parent and the
#      statistics source; the inflow–outflow child (5δ, refine 2
#      horizontally) runs with the generator off, the digital filter and
#      the FSD provider
#
# References: Xie & Castro (2008), FTaC 81, 449; Kim, Castro & Xie (2013),
# Comput. Fluids 84, 56; Porté-Agel et al. (2000), JFM 415, 261.

using Test
using Printf
using Aeolus
using Aeolus.Fields: interior_view
using Aeolus.BoundaryConditions: fill_halos!

include(joinpath(@__DIR__, "inflow_fetch_harness.jl"))

@testset "N-inflow-d: inflow fetch (#266)" begin

# ---------------------------------------------------------------------------
@testset "development_length: analytic profiles" begin
    λ = 0.7; T = 2.0
    x = collect(range(0.0, 8.0; length = 1601))
    f = T .* (1 .- exp.(-x ./ λ))
    # f ≥ 0.9 T once e^{-x/λ} ≤ 0.1: x = λ ln 10
    L = development_length((x, f), T)
    @printf("  exponential approach: L = %.5f, λ ln 10 = %.5f\n", L, λ * log(10))
    @test L ≈ λ * log(10) rtol = 1e-4
    # 20 %: λ ln 5
    @test development_length((x, f), T; frac = 0.8) ≈ λ * log(5) rtol = 1e-4
    # the inflow plane at x0 = -0.5 shifts the distance
    @test development_length((x, f), T; x0 = -0.5) ≈ λ * log(10) + 0.5 rtol = 1e-4
    # an overshoot that decays back: the upper edge T/0.9, 0.5 e^{-x/λ} = 1/0.9 - 1
    g = T .* (1 .+ 0.5 .* exp.(-x ./ λ))
    @test development_length((x, g), T) ≈ λ * log(0.5 / (1 / 0.9 - 1)) rtol = 1e-4
    # in the band from the first sample, and never in it at the end
    @test development_length((x, fill(1.05T, length(x))), T) == 0.0
    @test development_length((x, fill(1.05T, length(x))), T; x0 = -1.0) == 1.0
    @test development_length((x, fill(0.5T, length(x))), T) == Inf
    @test development_length((x, f .* 1.3), T) == Inf              # overshoot that stays
    @test_throws ArgumentError development_length((x, f), -1.0)
    @test_throws ArgumentError development_length((x, f), T; frac = 1.0)
    @test_throws ArgumentError development_length((reverse(x), f), T)
    @test_throws DimensionMismatch development_length((x, f[1:end-1]), T)
end

# ---------------------------------------------------------------------------
@testset "FetchStatistics: moments of a prescribed wave field" begin
    c = half_channel_case(; N = (8, 6, 4), extent = (2.0, 1.5), noise = 0.0)
    sim = c.sim; g = sim.mesh
    Nx, Ny, Nz, Hx, Hy, Hz = g.Nx, g.Ny, g.Nz, g.Hx, g.Hy, g.Hz
    a, b, cv = 0.8, 0.3, 0.5
    U0(k) = 10.0 + k
    nph = 16
    # nph equally weighted phases: every point's time mean is its wave-free value
    fs = FetchStatistics(sim)
    for n in 0:nph-1
        φ = 2π * n / nph
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            ψ = 2π * (j - 1) / Ny + φ
            sim.u.data[Hx+i, Hy+j, Hz+k] = U0(k) + a * cos(ψ)
            sim.v.data[Hx+i, Hy+j, Hz+k] = cv * sin(ψ)
        end
        for kf in 2:Nz, j in 1:Ny, i in 1:Nx
            sim.w.data[Hx+i, Hy+j, Hz+kf] = b * cos(2π * (j - 1) / Ny + φ)
        end
        accumulate_fetch!(fs, sim, 0.25)
    end
    p = fetch_profiles(fs)
    @test fs.nsamples == nph && fs.W ≈ 0.25nph
    @test all(isapprox.(p.uu, a^2 / 2; atol = 1e-12))
    @test all(isapprox.(p.vv, cv^2 / 2; atol = 1e-12))
    # w: interior levels see two faces with b²/2, the wall levels one (w ≡ 0 on the wall)
    @test all(isapprox.(p.ww[:, 2:Nz-1], b^2 / 2; atol = 1e-12))
    @test all(isapprox.(p.ww[:, [1, Nz]], b^2 / 4; atol = 1e-12))
    # −u'w' at the centres: w_c = b cos ψ inside, b/2 cos ψ on the wall levels
    @test all(isapprox.(p.uw[:, 2:Nz-1], -a * b / 2; atol = 1e-12))
    @test all(isapprox.(p.uw[:, [1, Nz]], -a * b / 4; atol = 1e-12))
    @test all(isapprox.(p.U, [U0(k) for i in 1:Nx, k in 1:Nz]; atol = 1e-12))
    @test p.tke ≈ 0.5 .* (p.uu .+ p.vv .+ p.ww)
    @test p.x ≈ [(i - 0.5) * g.dx for i in 1:Nx] && p.z ≈ [(k - 0.5) * g.dz for k in 1:Nz]
    # a steady field: zero turbulence, whatever its spatial structure
    fs2 = FetchStatistics(sim)
    for n in 1:5
        accumulate_fetch!(fs2, sim, 0.1n)
    end
    p2 = fetch_profiles(fs2)
    @test maximum(p2.tke) < 1e-12 && maximum(abs, p2.uw) < 1e-12
    @test_throws ArgumentError fetch_profiles(FetchStatistics(sim))
    @test_throws ArgumentError accumulate_fetch!(fs2, sim, 0.0)
end

# ---------------------------------------------------------------------------
@testset "total_stress_profile: an x band (irange)" begin
    c = half_channel_case(; N = (12, 8, 6), extent = (3.0, 2.0), noise = 1.0, seed = 3)
    s = c.sim
    for _ in 1:5
        advance_channel!(c, 0.01)
    end
    fill_halos!(s.u, s.bcs); fill_halos!(s.v, s.bcs); fill_halos!(s.w, s.bcs)
    νt = Aeolus.Operators.compute_sgs!(c.closure, s).νv
    full = total_stress_profile(s.u, s.v, s.w, νt)
    # the plane mean is the column-weighted mean of disjoint bands
    b1 = total_stress_profile(s.u, s.v, s.w, νt; irange = 1:5)
    b2 = total_stress_profile(s.u, s.v, s.w, νt; irange = 6:12)
    @test (5 .* b1.total .+ 7 .* b2.total) ./ 12 ≈ full.total atol = 1e-12
    @test (5 .* b1.resolved .+ 7 .* b2.resolved) ./ 12 ≈ full.resolved atol = 1e-12
    @test total_stress_profile(s.u, s.v, s.w, νt; irange = 1:12).total == full.total
    @test b1.total != b2.total                        # a turbulent field differs by band
    @test_throws ArgumentError total_stress_profile(s.u, s.v, s.w, νt; irange = 0:3)
    @test_throws ArgumentError total_stress_profile(s.u, s.v, s.w, νt; irange = 5:13)
end

# ---------------------------------------------------------------------------
@testset "half_channel_case(inflow = …): the inflow–outflow half channel" begin
    N = (16, 8, 8)
    probe = Aeolus.Grids.RegularGrid(size = N, extent = (2.0, 1.0, 1.0))
    Uz(z) = (1 / 0.4) * log(z / 1e-4)
    prof = [Uz((clamp(k - probe.Hz, 1, 8) - 0.5) / 8) for k in 1:8+2probe.Hz]
    c = half_channel_case(; N = N, extent = (2.0, 1.0), noise = 0.5,
                          inflow = ProfileBoundaryData(u = prof, v = 0.0, w = 0.0, θ = 0.0))
    s = c.sim; g = s.mesh
    @test s.bcs.west isa Aeolus.BoundaryConditions.OpenBoundary
    @test s.bcs.east isa Aeolus.BoundaryConditions.OpenBoundary
    @test s.bcs.south isa Aeolus.BoundaryConditions.Periodic
    @test g.Lx ≈ 2.0 && g.Ly ≈ 1.0 && g.Lz ≈ 1.0
    flux(i) = sum(s.u.data[g.Hx+i, g.Hy+1:g.Hy+g.Ny, g.Hz+1:g.Hz+g.Nz]) * g.dy * g.dz
    Qin = sum(prof[g.Hz+1:g.Hz+g.Nz]) * g.Ly * g.dz
    for _ in 1:40
        advance_channel!(c, channel_dt(c))
    end
    # the seeded noise is not solenoidal; SSP-RK3 keeps 1/3 of a step-start
    # divergence (u^{n+1} = u^n/3 + …), so after 40 steps it is round-off
    div = Aeolus.Simulations.max_divergence(s)
    Q = [flux(i) for i in 1:g.Nx+1]
    spread = maximum(abs.(Q .- Q[1])) / Q[1]
    # the inflow face: the profile plus the uniform mass-flux correction δ of
    # N-open-b (the radiated outflow flux differs from the inflow's)
    δ = s.u.data[g.Hx+1, g.Hy+1:g.Hy+g.Ny, g.Hz+1:g.Hz+g.Nz] .- reshape(prof[g.Hz+1:g.Hz+g.Nz], 1, :)
    @printf("  open-x half channel, 40 steps: max|∇·u| = %.2e, Q(x) spread %.2e, (Q - Q_in)/Q_in = %.2e, δ = %.2e\n",
            div, spread, Q[1] / Qin - 1, δ[1])
    @test div < 1e-10
    @test spread < 1e-12                         # volume flux in = out at every x
    @test maximum(δ) - minimum(δ) < 1e-12        # the profile, up to one uniform δ
    @test abs(Q[1] / Qin - 1) < 1e-3             # δ is small against the inflow
    @test all(isfinite, interior_view(s.u))
    @test_throws ArgumentError half_channel_case(; N = N, inflow = :profile)
    @test_throws ArgumentError half_channel_case(; N = N, extent = (1.0,))
    @test_throws ArgumentError half_channel_case(; N = N, surface_heat_flux = -0.01,
                                                 inflow = ProfileBoundaryData(u = 1.0, θ = 0.0))
end

end
