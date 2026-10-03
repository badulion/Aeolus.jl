# Era VI, Phase 54 (P54-c, #168): decaying grid turbulence of
# Comte-Bellot & Corrsin (1971), the LES benchmark of Era VI.
#
# Set-up and harness: test/cbc_benchmark_driver.jl (`cbc_benchmark`).
# Reference: CBC table 3 (test/data/cbc1971_table3.txt). CBC is a wind-
# tunnel experiment, not DNS; the LES is compared with the measured
# spectra filtered to the grid cut-off.
#
# Measured at 32³, three random seeds (the numbers in docs/src/theory/sgs.md):
#
#   closure        K98/K42 err  K171/K42 err  rms log10 E (98 / 171)  E(k_c)/CBC (98 / 171)
#   dynamic           +1.3 %      −15.7 %       0.085 / 0.099           1.23 / 1.00
#   Smagorinsky .17   +2.5 %      −14.2 %       0.088 / 0.097           1.27 / 1.08
#   no model         +78.3 %     +121.1 %       0.216 / 0.316           5.25 / 10.4
#   AMD (C = 0.3)     +9.9 %       −4.7 %       0.097 / 0.091           1.50 / 1.47   (P55-d, #176)
#   Vreman (c=.0723)  +4.9 %      −12.7 %       0.094 / 0.104           1.38 / 1.22   (P55-d, #176)
#
# The proposed ±10 % on K(171)/K(42) is NOT met (≈ −15 %); the RMS
# log-error at station 171 sits on the 0.1-decade target (0.099 for seeds
# 1–3, 0.101 for 1–6). The code checks out on the exact properties: the
# solver conserves energy to 0.14 % over the run without viscosity, the
# initial spectrum is CBC's to 1e-10 in every shell, and K(98)/K(42) is
# within 2.5 %. The deficit sits in the energy-containing range: LES/CBC
# is 0.71 and 0.65 in shells 2–3 (k = 0.22–0.34 cm⁻¹, the spectral peak)
# and 0.78–0.87 in shells 4–8, where CBC's large scales barely decay
# from 98 to 171 and the 11M box holds only a few dozen modes and
# none of the k < 2π/11M range; it is also present at 64³, which is
# further low at all k (examples/validation/cbc_decay.jl: −13 % / −24 %,
# E(k_c)/CBC ≈ 0.6–0.7, consistent with eddy viscosity on this 2nd-order
# grid over-dissipating once the cut-off is in the inertial range).
# Without the evolve-and-rescale start the errors swap (32³: +21 % /
# −3 %), the random-phase field decaying too slowly for one turnover. So
# the ±10 % expectation (from spectral-code LES, Kang et al. 2003) does not
# transfer to this set-up; the cause is not pinned down further here.
# The target is kept below as @test_broken, the assertion is ±20 %.
# AMD (#176) is the first closure that meets ±10 % at station 171
# (−4.7 %), so for it the target is a plain @test; it does so with less
# small-scale dissipation (E(k_c)/CBC 1.5 against 1.1–1.3), which is
# consistent with ν_t = 0 in a quarter of the cells (where the resolved
# strain does not produce subgrid energy) leaving more energy in the
# large scales.
#
# AMD margin at station 98 (maintainer review of #215). AMD's +9.85 % on
# K(98)/K(42) sits on the proposed ±10 %. Investigated:
#   * Constant and filter width are Rozema et al. (2015)'s: §II.C, after
#     their eq. 23, "δx_i = Δx_i" and "C = 0.300 for a central second-order
#     accurate method" (0.212 fourth order), tuned on this same CBC case
#     at 64³. C = 1/3 in that paper is the QR model's corrected constant,
#     not AMD's; 1/12 is the spectral value (Abkar, Bae & Moin 2016). The
#     CBC grid is isotropic, so δ_i = Δ and the anisotropic definition
#     cannot enter. No bug, so C stays 0.3.
#   * The margin is set by C (32³, seeds 1–3, K98 / K171 error):
#       C = 0.212 +20.9/+5.9 %, 0.25 +15.7/+0.8, 0.30 +9.9/−4.7,
#       1/3 +6.5/−7.7, 0.40 +0.6/−12.5 %.
#     No C meets ±10 % at both stations with margin: K98err − K171err is
#     13–15 points for every C and 16.7 / 17.5 / 17.0 for Smagorinsky,
#     Vreman and dynamic. That spread is the set-up deficit described
#     above, common to all closures. C only slides along it, and C = 0.3
#     lands K98 near the +10 % end.
#   * Resolution: at 64³ (seeds 1–3) the sign flips for every closure:
#     AMD 0.3 −11.0/−21.3 %, AMD 1/3 −12.2/−20.8, Smagorinsky
#     −14.2/−24.3, Vreman −13.9/−25.9. The ±10 % at 32³ is therefore a
#     property of 32³ on this 2nd-order grid, not a converged quantity.
#   * Seeds: per-seed K98 error at 32³ for seeds 1–10 is +10.00 % mean,
#     sd 0.41, and 4 of the 10 seeds are above 10 %. The 10-seed ensemble
#     is +10.00 % (seeds 1–3: +9.85 %). So `< 0.10` would pass or fail
#     depending on which seeds are drawn.
#   * Reference: at 32³ the first shell, k = 0.112 cm⁻¹, lies below CBC's
#     first tabulated k (0.20 cm⁻¹ at stations 42 and 98). It is
#     extrapolated, and holding E flat there instead of log-linear
#     changes the filtered reference K(98)/K(42) by +3.2 % (K171 by
#     +0.4 %). A 10 % criterion at station 98 is not resolved to better
#     than about 3 %.
# The original expectation, AMD within ±10 % at station 98 with the
# chosen seeds, holds only by chance. It is not physically wrong, but it
# is not resolvable at this resolution. AMD is therefore asserted at
# ±13 %: the ±10 % target widened by the 3 % reference uncertainty. The
# other closures keep ±10 %. The C sweep and the shared station spread
# are asserted below.

using Test
using Aeolus
using Printf

include("cbc_benchmark_driver.jl")

@testset "Era VI — Comte-Bellot–Corrsin decay (P54-c, #168)" begin

    @testset "reference data: table 3 against table 4" begin
        k, E, urms = cbc_table()
        @test size(E) == (20, 3) && count(isnan, E) == 4
        @test k[1] ≈ 15.0 && k[end] ≈ 2000.0                       # 0.15 … 20 cm⁻¹
        for s in 1:3
            f = cbc_spectrum(s)
            ok = .!isnan.(E[:, s])
            @test f.(k[ok]) ≈ E[ok, s] rtol = 1e-12                 # passes through the table
            # isotropy: ∫E dk = (3/2) u'² with u' from table 4 (an
            # independent measurement); within 5 % (measured 4.2, 1.3, 0.3 %)
            kk = exp.(range(log(1.0), log(3000.0), length = 20001))
            I = sum((f.(kk[1:end-1]) .+ f.(kk[2:end])) ./ 2 .* diff(kk))
            @test abs(I / (1.5 * urms[s]^2) - 1) < 0.05
        end
    end

    res = Dict{Symbol,Any}()
    t_total = @elapsed for c in (:dynamic, :smagorinsky, :none, :amd, :vreman)
        res[c] = cbc_benchmark(c; N = 32)
        @info "  " * cbc_summary(String(c), res[c])
    end
    @info @sprintf("  CBC 32³ (5 closures × 3 seeds): %.1f s", t_total)

    @testset "set-up: station-1 spectrum exact in every resolved shell" begin
        r = res[:dynamic]
        @test r.E_les[1][2:17] ≈ r.E_ref[1][2:17] rtol = 1e-10
        @test r.k[2] ≈ 2π / (11 * 0.0508)
        # filtering to 32³ keeps more of the decay than the full K = 3/2 u'²
        _, _, urms = cbc_table()
        @test r.K_ratio_ref[1] > (urms[2] / urms[1])^2
        @test r.K_ratio_ref[2] > (urms[3] / urms[1])^2
    end

    @testset "$c: TKE decay and spectra against CBC" for c in (:dynamic, :smagorinsky, :amd, :vreman)
        r = res[c]
        if c === :amd                                              # +9.85 %; 10 seeds +10.00 %:
            @test abs(r.K_err[1]) < 0.13                           #   ±10 % + 3 % reference (header)
        else
            @test abs(r.K_err[1]) < 0.10                           # K(98)/K(42) within ±10 %
        end
        @test abs(r.K_err[2]) < 0.20                               # K(171)/K(42): see header
        if c === :amd                                              # the proposed target:
            @test abs(r.K_err[2]) < 0.10                           #   met by AMD (−4.7 %)
        else
            @test_broken abs(r.K_err[2]) < 0.10                    #   not by the others
        end
        @test r.rms_log_err[1] <= 0.10
        @test r.rms_log_err[2] <= 0.11                             # 0.099 (seeds 1–3), 0.101 (1–6)
        @test r.E_kc_ratio[1] < 2 && r.E_kc_ratio[2] < 2           # no pile-up
    end

    @testset "AMD: the station-98 margin is set by C, the station spread by the set-up" begin
        ra = Dict(C => cbc_benchmark(AMDClosure(C = C); N = 32) for C in (0.25, 0.4))
        for C in (0.25, 0.4)
            @info "  " * cbc_summary("AMD C=$C", ra[C])
        end
        # more eddy viscosity → more decay → lower K(98)/K(42): strictly
        # monotone (+15.7, +9.9, +0.6 %), so the margin is the constant's
        @test ra[0.25].K_err[1] > res[:amd].K_err[1] + 0.03 > ra[0.4].K_err[1] + 0.06
        # the gap between the two stations is the shared set-up deficit:
        # 13–15 points for AMD at every C, 16.7–17.5 for the others
        spread(r) = r.K_err[1] - r.K_err[2]
        for r in (ra[0.25], res[:amd], ra[0.4], res[:smagorinsky], res[:vreman], res[:dynamic])
            @test 0.12 < spread(r) < 0.19
        end
    end

    @testset "discrimination: without a model energy piles up at k_c" begin
        r = res[:none]
        @test r.E_kc_ratio[1] > 2 && r.E_kc_ratio[2] > 2
        @test r.K_err[1] > 0.5                                     # far too little decay
        # and the models are distinguishable from it
        @test res[:dynamic].E_kc_ratio[2] < r.E_kc_ratio[2] / 4
    end

    @test t_total < 60                                             # CI budget
end
