# Era VI, Phase 56 (P56-c, #178): the a-priori SGS-stress harness.
#
# What is asserted:
#   1. the exact subfilter stress of a positive filter is positive
#      semi-definite (Vreman, Geurts & Kuerten 1994), and a filter with
#      negative lobes (sharp spectral) breaks it;
#   2. Galilean invariance of τ;
#   3. the closed form of τ₁₁ for u = A sin(kx) under the box and Gaussian
#      filters, in their transfer functions;
#   4. the a-priori Smagorinsky numbers at Δ = 4h on the filtered-DNS
#      fixture (test/fixtures/dns_decay.jl, 96³) against the published ranges;
#   5. the ranking Bardina > Smagorinsky in stress correlation, and the
#      dynamic C_s against the dissipation-matched one;
#   6. the offline filtered-DNS record (hybrid-DNS decision on #208): the
#      pipeline DNS → statistics → TOML → read-back at 32³ on every run;
#      the 256³ regeneration only with AEOLUS_DNS256=1 (measured 2.05 h
#      on one core, the solver is single-threaded; 9.9 GB max RSS as
#      reported by the run, not stored in the TOML;
#      examples/validation/dns256_generate.jl is the same call); and, when
#      test/fixtures/dns256_apriori.toml exists, the literature band at
#      4h. The ρ_Π comparison 48³ / 96³ / 256³ is logged, not asserted
#      ("Expectation that proved wrong with the 256³ record", below).
#      Until that file lands the last testset is a no-op (it says so).
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
#   Δ/η      48³ (Re_λ 32)   96³ (Re_λ 64)   256³ (Re_λ 78.7)
#   ≈ 8η     0.867           0.810           0.832
#   ≈ 16η    0.814           0.734           0.787
#   ≈ 32η    0.669           0.669           0.756
#
# The EXPECTATION was wrong, not the code (maintainer decision on #208).
# It was an extrapolation from two single-seed points (48³ → 96³), and
# the physical argument does not hold at fixed Δ/η: the filter sits at
# the same distance from the dissipation range whatever the Re, so only
# Δ/L changes, and no monotone dependence is established at Re_λ 32–79.
# The generator runs (past the peak, ν with a 3.5 % margin; examples/
# validation/dns256_generate.jl header) already agreed within about
# ±0.01 at 4h from 64³ upward: 0.819 (32³, Re_λ 21, 7.6η), 0.869 (48³,
# 31, 7.8η), 0.811 (64³, 45, 7.5η), 0.822 (96³, 52, 7.7η), against 0.810
# for this fixture (Re_λ 64, 7.8η). The 256³ trend assertion is dropped;
# the band assertions for the 96³ fixture and the 256³ record stay, and
# the 48³ / 96³ / 256³ comparison is logged as unasserted data. The
# 48³ > 96³ comparison at 4h and 8h (it passes: 0.810 < 0.867, 0.734 <
# 0.814) is still asserted, as a comparison of the two fixed single-seed
# fixtures and not as a trend; whether it is dropped too is the
# maintainer's call on #208.
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
# `dynamic_cs_on_coarse_grid`) the dynamic C_s is 0.116 (+24 %; 0.133,
# +33 %, on the 12³ grid of the old 48³ fixture), the coarse second-order
# strain underestimating |S̄|; that number is reported, not asserted
# (examples/validation/apriori_sgs.jl).

using Test
using Aeolus
using Aeolus.Fields: interior_view
using Printf
using LinearAlgebra: eigvals, Symmetric

include(joinpath(@__DIR__, "fixtures", "dns_apriori.jl"))   # includes dns_decay.jl

const DNS256_TOML = joinpath(@__DIR__, "fixtures", "dns256_apriori.toml")
# ρ_Π of the old 48³ fixture (Re_λ = 32), (r, Δ/η, ρ_Π): the history in the header
const RHO_PI_48 = ((4, 8.0, 0.867), (8, 15.9, 0.814))

# ρ_Π(Δ/η) of a TOML record, linear in log(Δ/η) between its widths;
# `nothing` outside the recorded range (not comparable).
_delta_eta_range(rec) = extrema(f["delta_over_eta"] for f in values(rec["filter"]))

function _rho_pi_at(rec, x)
    rows = sort([(f["delta_over_eta"], f["smagorinsky"]["rho_Pi"]) for f in values(rec["filter"])])
    xs = first.(rows); ys = last.(rows)
    (x < xs[1] || x > xs[end]) && return nothing
    i = min(searchsortedlast(xs, x), length(xs) - 1)
    w = log(x / xs[i]) / log(xs[i+1] / xs[i])
    return (1 - w) * ys[i] + w * ys[i+1]
end

@testset "Era VI — a-priori SGS-stress harness (P56-c, #178)" begin

    # ---------------------------------------------------------------- 3
    @testset "analytic: τ₁₁ of A sin(kx) in the transfer function" begin
        N = 32; L = 2π
        g = RegularGrid(; size = (N, 4, 4), extent = (L, L / 8, L / 8), halo = (3, 3, 3))
        h = g.dx
        A = 1.7
        x = [(i - 0.5) * h for i in 1:N]
        for m in (1, 3, 5)
            k = m * 2π / L
            u = Field(CellCenter(), g); v = Field(CellCenter(), g); w = Field(CellCenter(), g)
            interior_view(u) .= A .* sin.(k .* x)
            for (name, filt, G) in (
                    ("box 3", (o, f) -> box_filter!(o, f, (3, 1, 1)), q -> sin(3q * h / 2) / (3sin(q * h / 2))),
                    ("box 5", (o, f) -> box_filter!(o, f, (5, 1, 1)), q -> sin(5q * h / 2) / (5sin(q * h / 2))),
                    ("Gaussian 4h", (o, f) -> gaussian_filter!(o, f, (4h, 4h, 4h)), q -> exp(-q^2 * (4h)^2 / 24)))
                τ = SymmetricTensorField(g)
                subfilter_stress!(τ, u, v, w, filt)
                # u² = A²/2 (1 − cos 2kx): τ₁₁ = A²/2 [(1 − G(k)²) − (G(2k) − G(k)²) cos 2kx]
                exact = (A^2 / 2) .* ((1 - G(k)^2) .- (G(2k) - G(k)^2) .* cos.(2k .* x))
                @test maximum(abs, τ.xx[:, 2, 3] .- exact) < 1e-12
                @test maximum(abs, τ.xx .- reshape(exact, :, 1, 1)) < 1e-12
                @test maximum(abs, τ.yy) == 0 && maximum(abs, τ.xy) == 0
            end
        end
    end

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
        Δ = r * h
        filt = (o, f) -> gaussian_filter!(o, f, Δ)
        ub = ntuple(_ -> Field(CellCenter(), g), 3)
        τe = SymmetricTensorField(g)
        subfilter_stress!(τe, sim.u, sim.v, sim.w, filt; ubar = ub)
        S = filtered_strain(ub...)
        cs = dissipation_matched_cs(τe, S, Δ)
        τs = SymmetricTensorField(g); smagorinsky_stress!(τs, S, cs, Δ)
        τb = SymmetricTensorField(g); bardina_stress!(τb, ub..., filt)
        res[r] = (; Δ, filt, ub, τe, S, cs, smag = apriori_metrics(τs, τe, S), bard = apriori_metrics(τb, τe, S),
                    grad = apriori_metrics(clark_gradient_stress(ub, Δ, h), τe, S))
        m = res[r].smag; b = res[r].bard
        @info @sprintf("  Gaussian Δ = %dh: C_s(Π-matched) %.3f; Smagorinsky ρ(12,13,23) %.3f %.3f %.3f, ρ_Π %.3f; Bardina ρ(12,13,23) %.3f %.3f %.3f, ρ_Π %.3f; gradient ρ_12 %.3f; backscatter %.1f%%; II %.3f III %.4f",
                       r, cs, m.ρ.xy, m.ρ.xz, m.ρ.yz, m.ρ_Π, b.ρ.xy, b.ρ.xz, b.ρ.yz, b.ρ_Π, res[r].grad.ρ.xy,
                       100m.backscatter_exact, m.II, m.III)
    end

    # ---------------------------------------------------------------- 1
    @testset "positive filters give a PSD τ (Vreman et al. 1994)" begin
        r4 = res[4]
        filters = (("box 3", (o, f) -> box_filter!(o, f, 3)), ("box 5", (o, f) -> box_filter!(o, f, 5)),
                   ("Gaussian 4h", res[4].filt), ("Gaussian 8h", res[8].filt))
        for (name, filt) in filters
            τ = SymmetricTensorField(g)
            subfilter_stress!(τ, sim.u, sim.v, sim.w, filt)
            λ1, _, _ = stress_eigvals(τ)
            tr = τ.xx .+ τ.yy .+ τ.zz
            @test all(λ1 .>= -1e-12 .* tr)
            @info @sprintf("    %-12s min λ₁/tr τ = %.2e", name, minimum(λ1 ./ tr))
        end
        # the exact stress is realizable: every point in the Lumley triangle
        @test res[4].smag.realizable == 1 && res[8].smag.realizable == 1
        @test 0 < res[4].smag.II < 2 / 3
        # discrimination: the sharp spectral filter's kernel (a sinc) has
        # negative lobes, and τ is not PSD everywhere (Vreman et al. 1994)
        τc = SymmetricTensorField(g)
        subfilter_stress!(τc, sim.u, sim.v, sim.w, (o, f) -> spectral_cutoff_filter!(o, f, π / (4h)))
        λ1, _, _ = stress_eigvals(τc)
        tr = τc.xx .+ τc.yy .+ τc.zz
        @test count(λ1 .< -1e-6 .* abs.(tr)) > 0
        # the eigenvalue routine against LinearAlgebra on a few points
        for I in (CartesianIndex(1, 1, 1), CartesianIndex(17, 30, 5), CartesianIndex(48, 2, 41), CartesianIndex(96, 64, 77))
            M = [r4.τe[i, j][I] for i in 1:3, j in 1:3]
            @test collect(getindex.(stress_eigvals(r4.τe), Ref(I))) ≈ eigvals(Symmetric(M)) rtol = 1e-10
        end
    end

    # ---------------------------------------------------------------- 2
    @testset "Galilean invariance of τ" begin
        U = (3.7, -2.1, 1.3)
        for (name, filt) in (("box 3", (o, f) -> box_filter!(o, f, 3)), ("Gaussian 4h", res[4].filt))
            τ0 = SymmetricTensorField(g); τ1 = SymmetricTensorField(g)
            subfilter_stress!(τ0, sim.u, sim.v, sim.w, filt)
            us = map((f, c) -> (s = _field_like(f); s.data .= f.data .+ c; s), (sim.u, sim.v, sim.w), U)
            subfilter_stress!(τ1, us..., filt)
            err = maximum(maximum(abs, a .- b) for (a, b) in zip(Aeolus.Operators.components(τ0), Aeolus.Operators.components(τ1)))
            @info @sprintf("    %-12s max |τ(u+U) − τ(u)| = %.1e (max τ₁₁ %.2e)", name, err, maximum(τ0.xx))
            @test err < 1e-12
        end
    end

    # ---------------------------------------------------------------- 4
    @testset "Smagorinsky a priori at Δ = 4h" begin
        m = res[4].smag
        # Clark, Ferziger & Reynolds (1979); Liu, Meneveau & Katz (1994)
        @test all(0.1 .< (m.ρ.xy, m.ρ.xz, m.ρ.yz) .< 0.45)
        @test 0.4 < m.ρ_Π < 0.85                      # see header: 0.810 (0.867 at 48³)
        # 96³ fixture below the frozen 48³ values at 4h and 8h (header): a
        # comparison of these two single-seed fixtures, NOT a Re_λ trend.
        # Kept pending the maintainer's explicit call on #208.
        @test m.ρ_Π < RHO_PI_48[1][3] && res[8].smag.ρ_Π < RHO_PI_48[2][3]
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

    # ---------------------------------------------------------------- 6
    # The offline 256³ record (hybrid-DNS decision on #208). The pipeline
    # (DNS → statistics → TOML → read back) is proven here at 32³ on every
    # run; the 256³ run itself is offline (examples/validation/
    # dns256_generate.jl, or AEOLUS_DNS256=1 for the regeneration testset).
    @testset "offline a-priori record: pipeline at 32³" begin
        path = joinpath(mktempdir(), "dns32_apriori.toml")
        rec, d32 = generate_dns_apriori(; N = 32, out = path, ratios = (4, 8), verbose = false)
        back = read_dns_apriori(path)
        @test back["dns"] == rec["dns"] && back["checks"] == rec["checks"]
        @test back["filter"]["4h"]["smagorinsky"]["rho"] == rec["filter"]["4h"]["smagorinsky"]["rho"]
        @test back["spectrum"]["k"] == rec["spectrum"]["k"] && back["spectrum"]["E"] == rec["spectrum"]["E"]
        @test sort(collect(keys(back["filter"]))) == ["4h", "8h"]
        @test all(values(back["checks"]))
        @test back["dns"]["kmax_eta_min"] >= 1.5                           # the ν margin
        @test back["dns"]["t_end"] > back["dns"]["t_eps_peak"] + 0.2      # run on past the peak
        # Parseval: the recorded spectrum carries the energy of the centred
        # field it is defined on, which is the staggered K less the little
        # the two-point centring removes near the grid scale (a resolved DNS)
        Kspec = sum(back["spectrum"]["E"]) * (back["spectrum"]["k"][2] - back["spectrum"]["k"][1])
        Kc = 0.5 * sum(sum(abs2, Aeolus.Operators._to_center_periodic(f)) / 32^3 for f in (d32.sim.u, d32.sim.v, d32.sim.w))
        @test Kspec ≈ Kc rtol = 1e-10
        @test Kspec ≈ back["dns"]["K_end"] rtol = 0.05
        # one source of truth: the record's 4h numbers are the harness's
        f = back["filter"]["4h"]
        @test 0 < f["smagorinsky"]["rho_Pi"] < 1 && f["bardina"]["rho_Pi"] > f["smagorinsky"]["rho_Pi"]
        @test f["delta_over_eta"] ≈ 4 * (2π / 32) / d32.η
        # a failed DNS check keeps the record out of the suite's path
        bad = joinpath(dirname(path), "bad.toml")
        @test_throws ErrorException generate_dns_apriori(; N = 32, out = bad, ν = 0.02, ratios = (4,), verbose = false)
        @test !isfile(bad) && isfile(bad * ".failed")
        @test read_dns_apriori(bad * ".failed")["checks"]["kmax_eta_ge_1.5"] == false
    end

    @testset "offline a-priori record: 256³ regeneration (AEOLUS_DNS256=1)" begin
        if get(ENV, "AEOLUS_DNS256", "0") == "1"
            rec, _ = generate_dns_apriori(; N = 256, out = DNS256_TOML)
            @test isfile(DNS256_TOML)
            @test all(v for (k, v) in rec["checks"] if k ∉ ADVISORY_CHECKS)
        else
            @info "  256³ regeneration skipped (set AEOLUS_DNS256=1; ≈ 2 h on one core, ≈ 10 GB)"
        end
    end

    # No-op until test/fixtures/dns256_apriori.toml exists.
    @testset "offline a-priori record: 256³ against the literature" begin
        if isfile(DNS256_TOML)
            rec = read_dns_apriori(DNS256_TOML)
            dn = rec["dns"]
            @info @sprintf("  256³ record: Re_λ %.1f, k_max η ≥ %.2f, S %.3f, ρ_Π(4h) %.3f",
                           dn["Re_lambda"], dn["kmax_eta_min"], dn["skewness_end"],
                           rec["filter"]["4h"]["smagorinsky"]["rho_Pi"])
            @test dn["N"] == 256
            @test all(v for (k, v) in rec["checks"] if k ∉ ADVISORY_CHECKS)
            @test dn["kmax_eta_min"] >= 1.5 && -0.6 <= dn["skewness_end"] <= -0.35
            @test dn["budget_err_max"] <= 0.02
            @test dn["Re_lambda"] > d.Re_λ                                   # above the 96³ fixture
            f = rec["filter"]["4h"]; m = f["smagorinsky"]
            # Clark et al. (1979); Liu, Meneveau & Katz (1994)
            @test 0.4 < m["rho_Pi"] < 0.85
            @test all(0.1 .< (m["rho"]["xy"], m["rho"]["xz"], m["rho"]["yz"]) .< 0.45)
            @test 0.08 < f["cs_matched"] < 0.22
            @test 0 < f["backscatter"] < 0.5
            for (key, fr) in rec["filter"]
                fr["bardina"]["rho_Pi"] > fr["smagorinsky"]["rho_Pi"] || @info "  Bardina ≤ Smagorinsky at $key"
                @test fr["bardina"]["rho_Pi"] > fr["smagorinsky"]["rho_Pi"]
            end
            # ρ_Π against Re_λ at fixed Δ/η: logged, NOT asserted. The
            # monotone fall was an expectation that proved wrong (header).
            refs = [(48, x, ρ) for (_, x, ρ) in RHO_PI_48]
            append!(refs, [(96, res[r].Δ / d.η, res[r].smag.ρ_Π) for r in (4, 8)])
            for (Nref, x, ρref) in refs
                ρ256 = _rho_pi_at(rec, x)
                ρ256 === nothing && (@info @sprintf("    Δ/η = %.1f outside the 256³ record (%.2f–%.2f η; not extrapolated)",
                                                     x, _delta_eta_range(rec)...); continue)
                @info @sprintf("    Δ/η = %.1f: ρ_Π %d³ %.3f, 256³ %.3f (unasserted)", x, Nref, ρref, ρ256)
            end
        else
            @info "  test/fixtures/dns256_apriori.toml absent: 256³ literature testset is a no-op"
        end
    end
end
