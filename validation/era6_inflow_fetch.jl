# N-inflow-d (#266, epic #246): how far behind a synthetic inflow the
# turbulence is still out of equilibrium.
#
#   5. the CI fetch case: a periodic precursor at Δ_p is the parent and the
#      statistics source; the inflow–outflow child (5δ, refine 2
#      horizontally, open lateral faces) runs with the generator off, the
#      digital filter and the FSD provider: orderings DF < FSD < off
#   6. the physical fetch (the papers' set-up): DF-same — statistics and
#      mean from the periodic channel on the child grid — in a 10δ child
#      with open lateral faces; development within Kim et al.'s (2013) 10δ
#
# The diagnostics (1. development_length, 2. FetchStatistics,
# 3. total_stress_profile bands) and 4. the open-x half channel stay tested
# in test/era6_inflow_fetch.jl.
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
# the wall-bounded benchmark harness (half channel, channel) is a test fixture (ONE-l, #475)
isdefined(@__MODULE__, :ChannelCases) || include(joinpath(@__DIR__, "..", "test", "fixtures", "channel_cases.jl"))
using .ChannelCases
using Aeolus.Fields: interior_view
using Aeolus.BoundaryConditions: fill_halos!

# the harness (a test fixture since ONE-l #475; formerly the inflow_fetch run type)
isdefined(@__MODULE__, :InflowFetch) || include(joinpath(@__DIR__, "fixtures", "inflow_fetch.jl"))
using .InflowFetch: fetch_precursor, fetch_child, pressure_gradient_bias, nearest_level

@testset "N-inflow-d: inflow fetch (#266)" begin

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
