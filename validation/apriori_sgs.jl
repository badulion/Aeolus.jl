# Era VI, Phase 56 (P56-c, #178): validation of the a-priori SGS-stress
# harness on the filtered-DNS fixture (test/fixtures/dns_decay.jl, 96³
# decaying isotropic turbulence): the fixture checked as a DNS, then
# the a-priori numbers against the literature. The harness stays tested in
# test/apriori_sgs.jl (closed forms, PSD of positive filters, Galilean
# invariance, the history reductions, harness identities) on a cheap
# state. The fixture wall time (< 300 s) is asserted here, with the 96³ run.
#
#   4. the a-priori Smagorinsky numbers at Δ = 4h on the filtered-DNS
#      fixture (test/fixtures/dns_decay.jl, 96³) against the published ranges;
#   5. the ranking Bardina > Smagorinsky in stress correlation, and the
#      dynamic C_s against the dissipation-matched one;
#   (6. the offline 256³ filtered-DNS record of the `dns` run type went with
#      the DNS a-priori study, ONE-g #456: maintainer decision of
#      2026-10-09, LES validation is to be done externally, #461; the
#      a-posteriori comparison is examples/dns_vs_les/.)
# The fixture itself is checked as a DNS first (skewness, energy budget,
# k_max η, Re_λ).
#
# Measured (96³, Re_λ ≈ 64, seed 1, t = 2.0; quoted in
# docs/src/theory/les_filtering.md):
#
#   fixture: 227 RK3 steps, 88–138 s (3 threads, shared machine);
#   ν = 7.14e-3; Re_λ = 63.6; k_max η ≥ 1.61 over the run (1.598 at the
#   dissipation peak, t = 2.41, in a run continued to t = 3); skewness
#   −0.483 at t = 2; |dK/dt / (−ε) − 1| ≤ 1.7e-4 per step.
#
#   Gaussian Δ      4h (7.8η)       8h (15.7η)
#   min λ₁/tr τ     5.2e-4          4.0e-3     (PSD; box 3: 7.4e-5)
#   ρ(τ12,13,23)    .20 .15 .14     .22 .13 .12     Smagorinsky
#   ρ_Π (Smag.)     0.810           0.734
#   C_s matched     0.094           0.097
#   ρ (Bardina)     0.96–0.97       0.91–0.93
#   ρ12 (gradient)  0.980           0.949
#   C_s dynamic     0.100 (+6.3 %)  0.099 (+2.8 %)  (fine grid, α² = 1 + 8h²/Δ²)
#   backscatter     21 %            20 %       (points with Π_exact < 0)
#
# History: the 48³ fixture and the Π-correlation band. Until the
# maintainer's review of #208 the fixture was 48³ (Re_λ = 32, 99 steps,
# 5–7 s). There ρ_Π = 0.867 at 4h, above the literature band 0.4–0.85,
# and the assertion had been widened to 0.9 with the 0.85 bound kept as
# @test_broken. The code was not at fault: ρ_Π falls with Δ/η, and the
# 48³ value at 4h is above the band. At fixed Δ/η (k_max η is held fixed
# by ν ∝ N^{-4/3}):
#
#   Δ        48³ (Re_λ 32)     96³ (Re_λ 64)
#   4h       0.867 (8.0η)      0.810 (7.8η)
#   8h       0.814 (15.9η)     0.734 (15.7η)
#   16h      0.669 (31.9η)     0.669 (31.3η)   (Δ = L/3 at 48³: box-bound)
#
# At Δ ≈ 8η the filter sits in the near-dissipation range, where τ is
# given to leading order by its Taylor expansion, the gradient model
# τ ≈ (Δ²/12) ∂ₖūᵢ ∂ₖūⱼ (Clark et al. 1979; ρ₁₂ = 0.97 at 48³, 0.98 at
# 96³). Π of the gradient model is a cubic in the velocity gradient, as
# (C_sΔ)²|S̄|³ is. At Re_λ ≈ 64 the 4h value is inside the band, which is
# asserted again.
#
# Expectation that proved wrong with the 256³ record: "ρ_Π falls
# monotonically with Re_λ at fixed Δ/η". It was asserted (48³ > 96³ at 4h
# and 8h; 48³, 96³ > 256³ at the same Δ/η) and the 256³ record broke it
# (the ≈ 8η row is the 256³ value at 4h, 8.1η; the record starts at
# 8.05η, so the 48³ and 96³ points just below it are not interpolated):
#
#   Δ/η      48³ (Re_λ 32)   96³ (Re_λ 64)   256³ (Re_λ 78.6)
#   ≈ 8η     0.867           0.810           0.832
#   ≈ 16η    0.814           0.734           0.787
#   ≈ 32η    0.669           0.669           0.755
#
# (256³ column from the record regenerated after #205, #235; the pre-#205
# record gave 0.832, 0.787, 0.756 at Re_λ 78.7.)
#
# The EXPECTATION was wrong, not the code (maintainer decision on #208).
# It was an extrapolation from two single-seed points (48³ → 96³), and
# the physical argument does not hold at fixed Δ/η: the filter sits at
# the same distance from the dissipation range whatever the Re, so only
# Δ/L changes, and no monotone dependence is established at Re_λ 32–79.
# The generator runs (past the peak, ν with a 3.5 % margin; examples/
# validation/dns256_generate.jl header, since removed) already agreed within about
# ±0.01 at 4h from 64³ upward: 0.819 (32³, Re_λ 21, 7.6η), 0.869 (48³,
# 31, 7.8η), 0.811 (64³, 45, 7.5η), 0.822 (96³, 52, 7.7η), against 0.810
# for this fixture (Re_λ 64, 7.8η). Both trend assertions are dropped
# (maintainer decision on #208): 48³, 96³ > 256³ and 48³ > 96³, the
# latter although it still passes, since it rests on the same claim. The
# band assertions for the 96³ fixture and the 256³ record stay, and the
# 48³ / 96³ / 256³ comparison is logged as unasserted data.
#
# Expectation that proved wrong with the 96³ fixture: "t = 2 is past the
# dissipation peak". The peak moves later as Re_λ grows (measured
# t ≈ 0.99, 1.45, 1.85, 2.41 at 32³, 48³, 64³, 96³, ν ∝ N^{-4/3}): the
# cascade needs longer to fill the wider range of scales. At 96³, t = 2
# is 1.8 % below the peak ε. The fixture is kept at t = 2 (Re_λ ≈ 64, as
# the example), and the check is that ε is on its plateau: d ln ε/dt over
# the last 0.25 time units is 0.16, against 0.70 at t = 1–1.25.
#
# The dynamic C_s: `compute_dynamic_smagorinsky_νt!` is applied to the
# Gaussian-filtered field on the DNS grid (spacing h, where the
# dissipation-matched C_s also takes its gradients). Its test filter is
# the 3-cell box (second moment 2h²/3) on top of the Gaussian (Δ²/12),
# so the test-to-grid width ratio is α = √(1 + 8h²/Δ²), and C_s at width
# Δ is the routine's coefficient times h/Δ. With the routine's default
# α = 3 (right when the grid filter *is* the grid, Δ = h) C_s would be
# 0.026: the ratio is a property of the filters, not a tuning knob.
# On a 24³ grid of spacing 4h (Fourier-coarse-grained filtered field,
# the former `dynamic_cs_on_coarse_grid`) the dynamic C_s was 0.116 (+24 %; 0.133,
# +33 %, on the 12³ grid of the old 48³ fixture), the coarse second-order
# strain underestimating |S̄|; that number is reported, not asserted
# (the former examples/validation/apriori_sgs.jl).

using Test
using Aeolus
using Aeolus.Fields: interior_view
using Printf
using LinearAlgebra: eigvals, Symmetric

# the DNS fixture (decaying isotropic turbulence)
isdefined(@__MODULE__, :dns_decay) || include(joinpath(@__DIR__, "..", "test", "fixtures", "dns_decay.jl"))
# apriori_case: the a-priori quantities of one filter width (shared by the
# test and the validation run)
isdefined(@__MODULE__, :apriori_case) || include(joinpath(@__DIR__, "..", "test", "support", "apriori_cases.jl"))

@testset "Era VI — a-priori SGS stress on the 96³ DNS fixture (P56-c, #178)" begin

    # ---------------------------------------------------------------- fixture
    d = dns_decay(; N = 96)
    @info @sprintf("  DNS fixture 96³: %d steps, %.1f s; Re_λ %.1f; k_max η ≥ %.2f (%.2f at end); S %.3f; max |budget err| %.1e",
                   d.nsteps, d.wall, d.Re_λ, d.kmax_eta, d.kmax_eta_end, d.S[end], maximum(abs, d.budget_err))

    @testset "fixture is a DNS of decaying isotropic turbulence" begin
        @test d.kmax_eta >= 1.5                               # resolved over the whole run
        late = d.t .>= 1.0
        @test all(s -> -0.6 <= s <= -0.35, d.S[late])         # Batchelor & Townsend; Pope fig. 6.30
        last = d.t .>= d.t[end] - 0.5
        @test maximum(d.S[last]) - minimum(d.S[last]) < 0.05  # settled
        # at the dissipation plateau: the peak moves later with Re_λ (t ≈ 1.45
        # at 48³, 2.41 at 96³; header), so t = 2 is 1.8 % below it at 96³ and
        # d ln ε/dt has fallen from 0.70 (t = 1–1.25) to 0.16 (t = 1.75–2)
        i = argmin(abs.(d.t .- (d.t[end] - 0.25)))
        @test log(d.ε[end] / d.ε[i]) / (d.t[end] - d.t[i]) < 0.25
        @test maximum(abs, d.budget_err) <= 0.02              # dK/dt = −ε to ≤ 2 %
        @test 55 <= d.Re_λ <= 72                              # Re_λ ≈ 64: no inertial range yet
        @test d.K[end] < d.K[1] && all(diff(d.K) .< 0)        # monotone decay, no forcing
        @test d.wall < 300                                    # CI budget (88–138 s measured)
    end

    sim = d.sim; g = sim.mesh; h = g.dx
    res = Dict{Int,Any}()
    for r in (4, 8)
        res[r] = apriori_case(sim, r); cs = res[r].cs
        m = res[r].smag; b = res[r].bard
        @info @sprintf("  Gaussian Δ = %dh: C_s(Π-matched) %.3f; Smagorinsky ρ(12,13,23) %.3f %.3f %.3f, ρ_Π %.3f; Bardina ρ(12,13,23) %.3f %.3f %.3f, ρ_Π %.3f; gradient ρ_12 %.3f; backscatter %.1f%%; II %.3f III %.4f",
                       r, cs, m.ρ.xy, m.ρ.xz, m.ρ.yz, m.ρ_Π, b.ρ.xy, b.ρ.xz, b.ρ.yz, b.ρ_Π, res[r].grad.ρ.xy,
                       100m.backscatter_exact, m.II, m.III)
    end

    # ---------------------------------------------------------------- 4
    @testset "Smagorinsky a priori at Δ = 4h" begin
        m = res[4].smag
        # Clark, Ferziger & Reynolds (1979); Liu, Meneveau & Katz (1994)
        @test all(0.1 .< (m.ρ.xy, m.ρ.xz, m.ρ.yz) .< 0.45)
        @test 0.4 < m.ρ_Π < 0.85                      # see header: 0.810 (0.867 at 48³)
        @test 0.08 < res[4].cs < 0.22                 # Lilly 0.17 at high Re; lower at low Re_λ
        @test res[8].smag.ρ_Π < 0.85                  # inside the band once Δ ≈ 16η
        # consistency of the harness: the matched C_s matches the mean
        # transfer, and Smagorinsky never backscatters
        @test m.Π_ratio ≈ 1 atol = 1e-3
        @test m.backscatter_model == 0
        @test 0 < m.backscatter_exact < 0.5           # the exact stress does backscatter locally
        # header evidence: Δ = 4h is in the Taylor-expansion range
        @test res[4].grad.ρ.xy > 0.95 && res[4].grad.ρ.xy > res[8].grad.ρ.xy
    end

    # ---------------------------------------------------------------- 5
    @testset "ranking: Bardina above Smagorinsky; dynamic C_s" begin
        for r in (4, 8)
            s = res[r].smag.ρ; b = res[r].bard.ρ
            @test all(values(b) .> values(s))
            @test res[r].bard.ρ_Π > res[r].smag.ρ_Π
        end
        for r in (4, 8)
            Δ = res[r].Δ
            bcs = FieldBoundaryConditions(:periodic)
            ff = (Field(FaceX(), g), Field(FaceY(), g), Field(FaceZ(), g))
            for (c, f) in zip(ff, (sim.u, sim.v, sim.w))
                gaussian_filter!(c, f, Δ); fill_halos!(c, bcs)
            end
            ws = DynamicSmagorinskyWorkspace(g)
            compute_dynamic_smagorinsky_νt!(Field(CellCenter(), g), ff...; workspace = ws,
                                            filter_ratio = sqrt(1 + 8h^2 / Δ^2))
            cs_dyn = sqrt(sum(ws.LM) / sum(ws.MM)) * h / Δ
            @info @sprintf("    Δ = %dh: dynamic C_s %.3f, dissipation-matched %.3f (%+.1f%%)",
                           r, cs_dyn, res[r].cs, 100(cs_dyn / res[r].cs - 1))
            @test abs(cs_dyn / res[r].cs - 1) < 0.30
        end
    end
end
