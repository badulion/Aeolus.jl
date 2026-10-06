# N-inflow-d (#266, epic #246): how far behind a synthetic inflow the
# turbulence is still out of equilibrium.
#
#   1. development_length: the analytic recovery of an exponential approach,
#      an overshoot that returns, profiles in the band from the start or
#      never in it
#   2. FetchStatistics: the moments of a prescribed travelling-wave field
#      (variances a²/2 at the native points, the wall faces' half weight,
#      −u'w' at the centres); a steady field has zero TKE; a y band (jrange)
#   3. total_stress_profile(irange, jrange): an x or y band of a field is
#      seen only there, and disjoint bands average to the plane mean
#   4. the open-x half channel (half_channel_case(inflow = …)): the inflow
#      profile on the west face, volume flux in = out, divergence-free;
#      with open lateral faces (lateral = :open) the global balance closes
#   5. the CI fetch case: a periodic precursor at Δ_p is the parent and the
#      statistics source; the inflow–outflow child (5δ, refine 2
#      horizontally, open lateral faces) runs with the generator off, the
#      digital filter and the FSD provider: orderings DF < FSD < off
#   6. the physical fetch (the papers' set-up): DF-same — statistics and
#      mean from the periodic channel on the child grid — in a 10δ child
#      with open lateral faces; development within Kim et al.'s (2013) 10δ
#
# Maintainer decisions on #314 (2026-10-06): the fetch target is ≈ 10δ
# (Kim, Castro & Xie 2013), the issue's 3δ placeholder being a wrong
# expectation; total stress and drive bias get ordering assertions
# (DF better than FSD better than no generator); the CI fetch case has
# open lateral boundaries; the coarse-parent mean-profile bias is #268's.
# Runs are realisations until #245 puts the suite on FFTW.ESTIMATE, so
# every assertion below holds over the realisations measured.
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
    # jrange: an amplitude that differs by row is seen only on the rows
    # averaged; the y-faces of a band: interior faces whole, end faces half
    jr = 2:4
    fs3 = FetchStatistics(sim; jrange = jr)
    amp(j) = j in jr ? a : 3a
    vamp(jf) = jf in 3:4 ? cv : (jf in (2, 5) ? 2cv : 5cv)   # interior faces of 2:4 are 3, 4
    for n in 0:nph-1
        φ = 2π * n / nph
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            sim.u.data[Hx+i, Hy+j, Hz+k] = U0(k) + amp(j) * cos(φ)
            sim.v.data[Hx+i, Hy+j, Hz+k] = vamp(j) * sin(φ)
        end
        fill!(sim.w.data, 0.0)
        accumulate_fetch!(fs3, sim, 1.0)
    end
    p3 = fetch_profiles(fs3)
    @test all(isapprox.(p3.uu, a^2 / 2; atol = 1e-12))
    # faces 2 (half), 3, 4 (whole), 5 (half): (0.5·4 + 1 + 1 + 0.5·4) cv²/2 / 3
    @test all(isapprox.(p3.vv, (0.5 * 4 + 2 + 0.5 * 4) / 3 * cv^2 / 2; atol = 1e-12))
    @test maximum(p3.ww) < 1e-12 && maximum(abs, p3.uw) < 1e-12
    @test_throws ArgumentError FetchStatistics(sim; jrange = 0:2)
    @test_throws ArgumentError FetchStatistics(sim; jrange = 5:Ny+1)
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
    # jrange: rows, as irange does columns
    c1 = total_stress_profile(s.u, s.v, s.w, νt; jrange = 1:3)
    c2 = total_stress_profile(s.u, s.v, s.w, νt; jrange = 4:8)
    @test (3 .* c1.total .+ 5 .* c2.total) ./ 8 ≈ full.total atol = 1e-12
    q = total_stress_profile(s.u, s.v, s.w, νt; irange = 1:5, jrange = 1:3)
    q2 = total_stress_profile(s.u, s.v, s.w, νt; irange = 1:5, jrange = 4:8)
    @test (3 .* q.resolved .+ 5 .* q2.resolved) ./ 8 ≈ b1.resolved atol = 1e-12
    @test_throws ArgumentError total_stress_profile(s.u, s.v, s.w, νt; jrange = 0:3)
    @test_throws ArgumentError total_stress_profile(s.u, s.v, s.w, νt; jrange = 1:9)
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
    # open lateral faces (as a nest child): the same data on south and
    # north, radiating where the flow leaves. The global volume balance
    # (west in = east + south + north out) closes to round-off, and the
    # projection keeps the field solenoidal.
    cl = half_channel_case(; N = N, extent = (2.0, 1.0), noise = 0.5, lateral = :open,
                           inflow = ProfileBoundaryData(u = prof, v = 0.0, w = 0.0, θ = 0.0))
    sl = cl.sim
    @test sl.bcs.south isa Aeolus.BoundaryConditions.OpenBoundary
    @test sl.bcs.north isa Aeolus.BoundaryConditions.OpenBoundary
    for _ in 1:40
        advance_channel!(cl, channel_dt(cl))
    end
    Hx, Hy, Hz, Nx, Ny, Nz = g.Hx, g.Hy, g.Hz, g.Nx, g.Ny, g.Nz
    kk = Hz+1:Hz+Nz
    Qw = sum(sl.u.data[Hx+1, Hy+1:Hy+Ny, kk]) * g.dy * g.dz
    Qe = sum(sl.u.data[Hx+Nx+1, Hy+1:Hy+Ny, kk]) * g.dy * g.dz
    Qs = sum(sl.v.data[Hx+1:Hx+Nx, Hy+1, kk]) * g.dx * g.dz
    Qn = sum(sl.v.data[Hx+1:Hx+Nx, Hy+Ny+1, kk]) * g.dx * g.dz
    divl = Aeolus.Simulations.max_divergence(sl)
    @printf("  open-x, open-y half channel, 40 steps: max|∇·u| = %.2e, (Q_w + Q_s - Q_e - Q_n)/Q_w = %.2e, Q_n/Q_w = %.2e\n",
            divl, (Qw + Qs - Qe - Qn) / Qw, Qn / Qw)
    @test divl < 1e-10
    @test abs(Qw + Qs - Qe - Qn) / Qw < 1e-12
    @test all(isfinite, interior_view(sl.u))
    @test_throws ArgumentError half_channel_case(; N = N, lateral = :open)
    @test_throws ArgumentError half_channel_case(; N = N, lateral = :closed,
                                                 inflow = ProfileBoundaryData(u = prof, θ = 0.0))
    @test_throws ArgumentError half_channel_case(; N = N, inflow = :profile)
    @test_throws ArgumentError half_channel_case(; N = N, extent = (1.0,))
    @test_throws ArgumentError half_channel_case(; N = N, surface_heat_flux = -0.01,
                                                 inflow = ProfileBoundaryData(u = 1.0, θ = 0.0))
end


# ---------------------------------------------------------------------------
# The CI fetch case. Precursor (the parent, the statistics source): the
# periodic half channel on 5δ × 2.5δ × δ at 24 × 12 × 16 (Δx_p = 0.208δ,
# Δz = δ/16), z0 = 1e-4 δ, StaticSmagorinsky(0.1), spun up 20 δ/u* and
# averaged 10 δ/u*. Child: the same channel opened in x at refine 2
# horizontally (48 × 24 × 16, Δx = 0.104δ), 5δ long, spun up 1.5 δ/u*
# (6 flow-throughs) and averaged 5 δ/u*, with OPEN lateral faces (#314
# decision 3): the precursor's mean profile as Dirichlet data on south and
# north (the one-way nest default, PALM's child), so the side layers carry
# no turbulence; the statistics, pressure and stress are taken over the
# span centre, 0.5δ (5 cells) from each side. Targets are the precursor's.
#
# What the issue expected, the decisions and what was measured (theory
# page, "Development length"; five realisations, precursor seeds 1-4
# and a repeat of seed 1):
#   * fetch "within 10 % by 3δ": a wrong expectation (decision 1); the
#     literature number ≈ 10δ is asserted in the DF-same testset. Against
#     this coarse precursor no 10 % fetch exists at 0.22δ for any
#     generator: the child relaxes to the equilibrium of the parent's
#     11 %-too-fast mean profile (DF TKE/pre 0.90-0.99 at 0.5δ rising to
#     1.33-1.36 at 4.5δ), which #268 handles (decision 2). Asserted here:
#     the DF is closer to the precursor than the FSD (TKE and −u'w').
#   * total stress at 4δ: ordering DF < FSD < off in RMS error over
#     0.1-0.8δ (decision 1); measured DF 0.33-0.46, FSD 0.51-0.55,
#     off 1.23-1.32 u*².
#   * null comparison, bulk velocity: as issued (bulk within 1.9e-4).
#   * drive bias: ordering |DF| < |FSD| < |off| (decision 1); measured
#     DF −0.8 to −8.6 %, FSD +27 to +31 %, off +66 to +71 %. The issued
#     2 % is a wrong expectation for a developing flow: the extra drive
#     balances the streamwise change of the momentum flux and of the
#     wall stress over 1-4δ, a fetch quantity.
@testset "inflow fetch: precursor, generator off / digital filter / FSD" begin
    Np, Nc, ext = (24, 12, 16), (48, 24, 16), (5.0, 2.5)
    t_wall = time()
    pre = fetch_precursor(; N = Np, extent = ext, t_spin = 20.0, t_avg = 10.0)
    zf = pre.z_faces
    sel = findall(z -> 0.1 <= z <= 0.9, zf)
    dev_pre = maximum(abs.(pre.stress[sel] .- (1 .- zf[sel])))
    @printf("  precursor (%.1f s): total stress vs u*²(1 - z/δ), max dev over 0.1-0.9δ = %.3f u*²\n",
            pre.wall_time, dev_pre)
    # the precursor is in equilibrium: the total stress of the half channel
    # is u*²(1 - z/δ) (Porté-Agel et al. 2000, Fig. 3)
    @test dev_pre < 0.08
    G = pre.case.forcing
    Ub = sum(pre.U) / length(pre.U)
    runs = Dict(g => fetch_child(pre; generator = g, N = Nc, extent = ext, t_spin = 1.5,
                                 t_avg = 5.0, sample_every = 4, lateral = :open,
                                 y_margin = 0.5) for g in (:off, :df, :fsd))
    # the child is open on all four lateral faces
    for f in (:west, :east, :south, :north)
        @test getfield(runs[:df].case.sim.bcs, f) isa Aeolus.BoundaryConditions.OpenBoundary
    end
    @test runs[:df].jrange == 6:19
    hs = (0.2, 0.5)
    ks = [nearest_level(pre.z, h) for h in hs]
    near3 = findall(x -> x <= 3.0, runs[:df].p.x)
    ratio(r, f, k) = getfield(r.p, f)[:, k] ./ getfield(pre, f)[k]
    for g in (:off, :df, :fsd)
        r = runs[g]
        @printf("  %-4s (%.1f s): bulk/U_b %.6f (spread %.1e), dp/dx bias %+.4f G, added flux %.1e, KE flux %.2f, div %.1e\n",
                g, r.wall_time, r.bulk[1] / Ub, (maximum(r.bulk) - minimum(r.bulk)) / Ub,
                pressure_gradient_bias(r, G), r.added, r.ke_flux, r.maxdiv)
        for (h, k) in zip(hs, ks)
            t = ratio(r, :tke, k); u = ratio(r, :uw, k)
            @printf("       z = %.2fδ: TKE/pre at x = 0.5, 1, 2, 3, 4.5δ: %s; L10 %.2fδ, L20 %.2fδ | -u'w'/pre: %s; L10 %.2fδ\n",
                    pre.z[k], join((@sprintf("%.2f", t[nearest_level(r.p.x, x)]) for x in (0.5, 1, 2, 3, 4.5)), " "),
                    development_length((r.p.x, r.p.tke[:, k]), pre.tke[k]; x0 = 0.0),
                    development_length((r.p.x, r.p.tke[:, k]), pre.tke[k]; x0 = 0.0, frac = 0.8),
                    join((@sprintf("%.2f", u[nearest_level(r.p.x, x)]) for x in (0.5, 1, 2, 3, 4.5)), " "),
                    development_length((r.p.x, r.p.uw[:, k]), pre.uw[k]; x0 = 0.0))
        end
    end
    off, df, fsd = runs[:off], runs[:df], runs[:fsd]
    # null comparison (issued): generator off, TKE < 50 % of the precursor
    # at the same x — a steady, spanwise-uniform inflow stays laminar
    for k in ks
        @test maximum(ratio(off, :tke, k)[near3]) < 0.5
        @test maximum(ratio(off, :tke, k)) < 1e-8
    end
    # the digital filter supplies the precursor's turbulence from the inflow
    # plane on and is closer to it than the FSD provider, whose isotropic
    # single-scale field carries no u'w' structure. (An absolute band,
    # [0.7, 1.5] calibrated on two runs, was removed: it was a regression
    # bound, not a physical tolerance, and over seven precursor
    # realisations the DF TKE ratio spans 0.72-1.39 and −u'w' 0.44-1.53,
    # i.e. within 0.02 of the band's floor. The fetch physics is asserted
    # against the child-grid reference in the DF-same testset.)
    lndev(r, f, k) = sum(abs.(log.(ratio(r, f, k)[near3]))) / length(near3)
    # (mean |ratio - 1| for −u'w': the FSD −u'w' may start at or below zero)
    absdev(r, k) = sum(abs.(ratio(r, :uw, k)[near3] .- 1)) / length(near3)
    for k in ks
        @test lndev(df, :tke, k) < lndev(fsd, :tke, k)
        # −u'w': the digital filter imposes the precursor's u'w' correlation
        # (the Lund transform), the FSD field none, so the DF shear stress
        # is closer to the precursor's (the physical fetch of −u'w' is
        # asserted in the DF-same testset below)
        @printf("  -u'w' at %.2fδ, mean |ratio - 1| over x ≤ 3δ: DF %.3f, FSD %.3f\n",
                pre.z[k], absdev(df, k), absdev(fsd, k))
        @test absdev(df, k) < absdev(fsd, k)
    end
    # shear stress at x ≈ 4δ, 0.1δ < z < 0.8δ: DF < FSD < off in RMS error
    sz = findall(z -> 0.1 < z < 0.8, zf)
    rms(r) = sqrt(sum(abs2, r.stress[sz] .- pre.stress[sz]) / length(sz))
    @printf("  total stress at x = %.2fδ, RMS error over 0.1-0.8δ: DF %.3f, FSD %.3f, off %.3f u*²\n",
            df.x_stress, rms(df), rms(fsd), rms(off))
    @test rms(df) < rms(fsd) < rms(off)
    # no spurious mean (issued): bulk velocity within 0.5 % of the
    # precursor's at every x
    for r in (off, df, fsd)
        @test maximum(abs.(r.bulk ./ Ub .- 1)) < 0.005
        @test r.maxdiv < 1e-10
    end
    @test df.added < 1e-12 && fsd.added < 1e-12
    # the extra driving pressure gradient over 1-4δ (see the header: a
    # fetch quantity, −0.8 to −8.6 % for the DF over five realisations,
    # so the issued 2 % is not asserted): the better the inflow
    # turbulence, the smaller the extra drive
    @test abs(pressure_gradient_bias(df, G)) < abs(pressure_gradient_bias(fsd, G)) <
          abs(pressure_gradient_bias(off, G))
    @printf("  CI fetch case: %.1f s\n", time() - t_wall)
end


# ---------------------------------------------------------------------------
# The physical fetch: the set-up of the papers. Kim, Castro & Xie (2013,
# §3.3: plane channel, Re_τ = 395, U_b/u_τ = 18.3, spanwise periodic, the
# statistics and mean from the periodic LES on the same grid) found the
# wall shear stress and −u'v' recovered by x ≈ 10δ for the digital filter
# (XC, XCMC), 14δ to 5 % for their divergence-free variant counted from
# its plane. That is the fetch target the maintainer adopted (#314
# decision 1) in place of the 3δ placeholder. Here: the periodic channel
# on the child grid (48 × 24 × 16) is the reference, the statistics
# source and the inflow profile (DF-same); the child is 10δ long at the
# same Δx, with open lateral faces as in the CI case above (decision 3).
#
# Physics asserted: the fetch is about one local eddy-turnover distance
# U k/P (P = τ ∂U/∂z, local equilibrium P = ε), measured on the
# reference. At 0.22δ, U k/P ≈ 5.5δ: TKE and −u'w' within 10 % by
# x ≤ 10δ (the paper's number) and L10 between 0.6 and 1.6 turnovers
# (five realisations, seeds 2-5 and a repeat of 2: L10 TKE 5.6-6.6δ,
# −u'w' 5.6-7.9δ, 1.03-1.46 turnovers; the spanwise-periodic 15δ example: 6.6δ). At
# 0.47δ, U k/P ≈ 10δ (the stress falls as 1 - z/δ), about the child
# length: the 10 % fetch is 12.3δ in the 15δ example (1.2 turnovers),
# so "about 10δ" and not inside this child. Asserted: not recovered
# within half a turnover, still developing across the child, and within
# 30 % of the reference over 8-9.5δ (measured 0.77-0.87).
@testset "inflow fetch: DF-same against the child-grid reference (Kim et al. 2013)" begin
    t_wall = time()
    Nr, Lc = (48, 24, 16), 10.0
    ref = fetch_precursor(; N = Nr, extent = (5.0, 2.5), t_spin = 20.0, t_avg = 10.0, seed = 2)
    r = fetch_child(ref; generator = :df, N = (round(Int, Nr[1] * Lc / 5), Nr[2], Nr[3]),
                    extent = (Lc, 2.5), t_spin = 1.5, t_avg = 5.0, sample_every = 4,
                    lateral = :open, y_margin = 0.5)
    @test r.case.sim.bcs.south isa Aeolus.BoundaryConditions.OpenBoundary
    dz = ref.case.sim.mesh.dz
    # the last 0.5δ is the convective outflow layer, not part of the fetch
    keep = findall(x -> x <= Lc - 0.5, r.p.x)
    x = r.p.x[keep]
    function turnover(k)
        τ = 0.5 * (ref.stress[k] + ref.stress[k+1])            # total stress at the centre
        dUdz = (ref.U[k+1] - ref.U[k-1]) / (2dz)
        return ref.U[k] * ref.tke[k] / (τ * dUdz)
    end
    L10(f, k) = development_length((x, getfield(r.p, f)[keep, k]), getfield(ref, f)[k]; x0 = 0.0)
    rat(f, k) = getfield(r.p, f)[keep, k] ./ getfield(ref, f)[k]
    k2, k5 = nearest_level(ref.z, 0.2), nearest_level(ref.z, 0.5)
    for k in (k2, k5)
        @printf("  DF-same z = %.2fδ: U k/P = %.2fδ; L10 TKE %.2fδ, -u'w' %.2fδ; TKE/ref at x = 1, 3, 5, 7, 9δ: %s\n",
                ref.z[k], turnover(k), L10(:tke, k), L10(:uw, k),
                join((@sprintf("%.2f", rat(:tke, k)[nearest_level(x, xx)]) for xx in (1, 3, 5, 7, 9)), " "))
    end
    @printf("  (reference %.1f s, 10δ child %.1f s)\n", ref.wall_time, r.wall_time)
    # 0.22δ: recovered within the paper's 10δ, after about one turnover
    T2 = turnover(k2)
    for f in (:tke, :uw)
        L = L10(f, k2)
        @test L <= 10.0
        @test 0.6T2 <= L <= 1.6T2
    end
    # 0.47δ: the turnover distance is about the child length. Not
    # recovered within half a turnover (both moments below 90 %), and
    # still developing: the ratio grows by more than 0.1 from x ≈ 1δ to
    # x ≈ 9δ (measured 0.17-0.47 over five realisations), and it is within
    # 30 % of the reference over the last 1.5δ (0.77-0.87)
    T5 = turnover(k5)
    @test T5 > 1.5T2                                  # the fetch grows with height
    half = findall(xx -> xx <= 0.5T5, x)
    for f in (:tke, :uw)
        @test maximum(rat(f, k5)[half]) < 0.9
        @test rat(f, k5)[nearest_level(x, 9.0)] - rat(f, k5)[nearest_level(x, 1.0)] > 0.1
        tail = findall(xx -> xx >= 8.0, x)
        @printf("  DF-same z = %.2fδ: %s/ref over 8-9.5δ = %.3f\n", ref.z[k5], f,
                sum(rat(f, k5)[tail]) / length(tail))
        @test sum(rat(f, k5)[tail]) / length(tail) > 0.7
    end
    # the reference itself is in equilibrium (as the precursor above)
    sel = findall(z -> 0.1 <= z <= 0.9, ref.z_faces)
    @test maximum(abs.(ref.stress[sel] .- (1 .- ref.z_faces[sel]))) < 0.08
    @test r.maxdiv < 1e-10
    @printf("  DF-same fetch case: %.1f s\n", time() - t_wall)
end

end
