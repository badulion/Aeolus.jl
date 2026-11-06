# Phase 45 — turbulence-microphysics interaction.
#
# Issue #12: the sub-grid distribution of the saturation deficit, the
# partial cloud fraction and condensate it implies, and the SGS variance
# that drives it. Headline checks: the σ → 0 limit is the exact
# all-or-nothing saturation adjustment of the Phase 39 contract, the
# Gaussian case matches the Sommeria-Deardorff / Mellor closed forms, and
# the linearised PDF agrees with direct quadrature of the *exact*
# adjustment over the joint (q_t, θ_li) distribution.

using Test
using Aeolus
using Aeolus.Grids: interior

# Composite Simpson on a uniform grid (n even) — the tests' independent
# quadrature, deliberately not sharing any code with the closed forms.
function _simpson(f, a, b, n)
    h = (b - a) / n
    s = f(a) + f(b)
    for i in 1:n-1
        s += (isodd(i) ? 4 : 2) * f(a + i * h)
    end
    return s * h / 3
end

@testset "Turbulent microphysics (Phase 45)" begin

    @testset "Saturation-deficit PDFs: moments, closed forms vs quadrature" begin
        s̄, σ = 2.0e-4, 5.0e-4
        for pdf in (GaussianSaturationPDF(), DoubleGaussianSaturationPDF(0.8))
            Sk = pdf isa GaussianSaturationPDF ? 0.0 : pdf.skewness
            lo, hi = s̄ - 12σ, s̄ + 12σ
            P(s) = saturation_pdf_density(pdf, s, s̄, σ)
            @test _simpson(P, lo, hi, 4000) ≈ 1 atol = 1e-10
            @test _simpson(s -> s * P(s), lo, hi, 4000) ≈ s̄ rtol = 1e-8
            @test _simpson(s -> (s - s̄)^2 * P(s), lo, hi, 4000) ≈ σ^2 rtol = 1e-8
            @test _simpson(s -> (s - s̄)^3 * P(s), lo, hi, 4000) / σ^3 ≈ Sk atol = 1e-7
            @test pdf_cloud_fraction(pdf, s̄, σ) ≈ _simpson(P, 0.0, hi, 4000) rtol = 1e-9
            @test pdf_condensate(pdf, s̄, σ) ≈ _simpson(s -> s * P(s), 0.0, hi, 4000) rtol = 1e-9
        end
        # Gaussian closed forms against tabulated Φ (Abramowitz & Stegun 26.2)
        g = GaussianSaturationPDF(); σ = 3.0e-4
        table = ((-2.0, 0.022750131948179), (-1.0, 0.158655253931457), (0.0, 0.5),
                 (0.5, 0.691462461274013), (2.0, 0.977249868051821))
        @test maximum(abs(pdf_cloud_fraction(g, Q * σ, σ) / Φ - 1) for (Q, Φ) in table) < 1e-12
        @test maximum(abs(pdf_condensate(g, Q * σ, σ) /
                          (σ * (Q * Φ + exp(-Q^2 / 2) / sqrt(2π))) - 1) for (Q, Φ) in table) < 1e-12
        # far tails stay non-negative and monotone; extreme skewness finite
        @test all(pdf_condensate(g, Q, 1.0) >= 0 for Q in -60:0.1:60)
        @test all(diff([pdf_cloud_fraction(g, Q, 1.0) for Q in -60:0.1:60]) .>= 0)
        @test all(let d = DoubleGaussianSaturationPDF(Sk), C = pdf_cloud_fraction(d, s, 1e-4)
                      isfinite(C) && 0 <= C <= 1 && pdf_condensate(d, s, 1e-4) >= max(s, 0.0) * (1 - 1e-12)
                  end for Sk in (1e3, 1e20, -1e20), s in (-1e-4, 0.0, 1e-4))
        @test_throws ArgumentError DoubleGaussianSaturationPDF(NaN)
        # double Gaussian with Sk = 0 is the Gaussian; O(Sk) slope is Edgeworth's
        s̄, σ = -1.0e-4, 4.0e-4
        @test pdf_cloud_fraction(DoubleGaussianSaturationPDF(0.0), s̄, σ) == pdf_cloud_fraction(g, s̄, σ)
        Q = s̄ / σ
        edgeworth = (Q^2 - 1) * exp(-Q^2 / 2) / sqrt(2π) / 6
        @test (pdf_cloud_fraction(DoubleGaussianSaturationPDF(1e-8), s̄, σ) -
               pdf_cloud_fraction(g, s̄, σ)) / 1e-8 ≈ edgeworth rtol = 1e-3
    end

    p = 9.0e4
    T0 = 290.0
    qs0 = saturation_specific_humidity(T0, p)
    θ0 = liquid_ice_potential_temperature(T0, p, PhasePartition(qs0))

    @testset "σ_s → 0 is the all-or-nothing saturation adjustment, exactly" begin
        bad = 0; worst_T = 0.0; worst_q = 0.0
        for (T, pr) in ((290.0, 9.0e4), (255.0, 6.0e4), (228.0, 3.0e4)), f in (0.7, 1.0005, 1.3)
            qt = f * saturation_specific_humidity(T, pr)
            θ = liquid_ice_potential_temperature(T, pr, PhasePartition(qt))
            T_adj, q_adj = saturation_adjustment(θ, pr, qt)
            r = subgrid_cloud(θ, pr, qt)                 # all variances zero
            (r.σ_s == 0 && r.T == T_adj && r.q == q_adj &&
             r.cloud_fraction == (q_adj.liq + q_adj.ice > 0 ? 1.0 : 0.0)) || (bad += 1)
            rε = subgrid_cloud(θ, pr, qt; σ_s = 1e-12)  # and continuously so
            worst_T = max(worst_T, abs(rε.T - T_adj))
            worst_q = max(worst_q, abs(rε.condensate - r.condensate))
        end
        @test bad == 0
        @test worst_T < 1e-6 && worst_q < 1e-11
    end

    @testset "Mellor coefficients agree with the exact adjustment's slope" begin
        m = mellor_coefficients(θ0, p, qs0)
        h = 1e-3
        @test m.dqsdT ≈ (saturation_specific_humidity(m.T_l + h, p) -
                         saturation_specific_humidity(m.T_l - h, p)) / 2h rtol = 1e-6
        qc(θ, qt) = (q = saturation_adjustment(θ, p, qt)[2]; q.liq + q.ice)
        @test (qc(θ0, qs0 + 1e-8) - qc(θ0, qs0)) / 1e-8 ≈ m.a_q rtol = 1e-5
        @test (qc(θ0 - 1e-5, qs0) - qc(θ0, qs0)) / 1e-5 ≈ m.b rtol = 1e-4
        @test 0.25 < m.a < 0.4        # ≈ 0.32 near 290 K, 900 hPa
        # the cell diagnosis is the Gaussian closed form in Q = s̄/σ_s
        r = subgrid_cloud(θ0, p, 0.99qs0; var_qt = (3e-4)^2)
        Φ = 0.5 * (1 + Aeolus.GasOptics.erf_series(r.Q / sqrt(2)))
        @test r.cloud_fraction ≈ Φ rtol = 1e-12
        @test r.condensate ≈ r.σ_s * (r.Q * Φ + exp(-r.Q^2 / 2) / sqrt(2π)) rtol = 1e-12
        # monotone in the mean deficit, clear to overcast
        kw = (var_qt = (2.5e-4)^2, var_θ = 0.15^2, cov_qθ = 0.4 * 2.5e-4 * 0.15)
        Cq = [subgrid_cloud(θ0, p, f * qs0; kw...).cloud_fraction for f in 0.9:0.01:1.1]
        @test all(diff(Cq) .> 0) && Cq[1] < 1e-3 && Cq[end] > 1 - 1e-3
    end

    @testset "Total water and θ_li are conserved" begin
        worst_θ = 0.0; bad = 0
        for (T, pr) in ((290.0, 9.0e4), (240.0, 4.0e4)), f in (0.9, 1.1), σq in (1e-5, 1e-3),
            pdf in (GaussianSaturationPDF(), DoubleGaussianSaturationPDF(1.0))
            qt = f * saturation_specific_humidity(T, pr)
            θ = liquid_ice_potential_temperature(T, pr, PhasePartition(qt))
            r = subgrid_cloud(θ, pr, qt; var_qt = σq^2, pdf = pdf)
            (r.q.tot == qt && isapprox(r.q.liq + r.q.ice, r.condensate; rtol = 1e-14) &&
             0 <= r.condensate <= qt) || (bad += 1)
            worst_θ = max(worst_θ, abs(liquid_ice_potential_temperature(r.T, pr, r.q) - θ) / θ)
        end
        @test bad == 0
        @test worst_θ < 1e-11
        # condensate is capped at the water that is there
        θt = liquid_ice_potential_temperature(290.0, p, PhasePartition(1e-5))
        @test subgrid_cloud(θt, p, 1e-5; σ_s = 1.0).condensate == 1e-5
    end

    @testset "Linearised PDF vs exact adjustment over the joint (q_t, θ_li) PDF" begin
        # Reference: integrate the EXACT saturation_adjustment over a
        # bivariate normal in (q_t, θ_li) by 2-D Simpson. The difference is
        # the Mellor linearisation of the fluctuations only. One correlated
        # case (the full set of six is in the git history; all < 1 %/2 %).
        φ(z) = exp(-z^2 / 2) / sqrt(2π)
        qc(θ, qt) = (q = saturation_adjustment(θ, p, qt)[2]; q.liq + q.ice)
        σq, σθ, ρ = 3e-4, 0.2, 0.5
        state(z1, z2) = (θ0 + σθ * (ρ * z1 + sqrt(1 - ρ^2) * z2), qs0 + σq * z1)
        n, L = 60, 6.0
        h = 2L / n
        w(i) = (i == 0 || i == n) ? 1 : (isodd(i) ? 4 : 2)
        qc_ref = sum(w(i) * w(j) * qc(state(-L + i * h, -L + j * h)...) * φ(-L + i * h) * φ(-L + j * h)
                     for i in 0:n, j in 0:n) * (h / 3)^2
        r = subgrid_cloud(θ0, p, qs0; var_qt = σq^2, var_θ = σθ^2, cov_qθ = ρ * σq * σθ)
        @test abs(r.condensate / qc_ref - 1) < 0.02
    end

    # Fill a cell-centred field INCLUDING its halos from f(x, y, z), so a
    # smooth profile has exact centred gradients up to the domain edge.
    function fill_all!(fld, f)
        g = fld.mesh
        for k in axes(fld.data, 3), j in axes(fld.data, 2), i in axes(fld.data, 1)
            fld.data[i, j, k] = f(g.x0 + (i - g.Hx - 0.5) * g.dx, g.y0 + (j - g.Hy - 0.5) * g.dy,
                                  g.z0 + (k - g.Hz - 0.5) * g.dz)
        end
        return fld
    end

    @testset "SGS variance: geometric meaning and the equilibrium budget" begin
        g = RegularGrid(; size = (4, 4, 8), extent = (400.0, 400.0, 800.0))
        Δ = filter_width(g)
        @test Δ ≈ 100.0
        G = 2.0e-5
        φ = fill_all!(Field(g), (x, y, z) -> 0.01 - G * z)
        var = Field(g)
        sgs_scalar_variance!(var, φ)
        # variance of a linear profile inside one cell: G²Δ²/12
        @test all(v -> isapprox(v, G^2 * Δ^2 / 12; rtol = 1e-12), view(var.data, interior(g)...))
        cov = Field(g)
        sgs_scalar_covariance!(cov, φ, fill_all!(Field(g), (x, y, z) -> 3e-3 * z))
        @test view(cov.data, interior(g)...)[1, 1, 1] ≈ -G * 3e-3 * Δ^2 / 12 rtol = 1e-12
        # equilibrium: production 2(ν_t/Pr_t)|∇φ|² = dissipation C_χ(√e/Δ)φ'²
        # with ν_t = C_k Δ √e, cell by cell, ν_t over three decades
        C_k, C_χ, Pr_t = 0.1, 2.0, 1 / 3
        c_eq = equilibrium_variance_coefficient(; C_k = C_k, C_χ = C_χ, Pr_t = Pr_t)
        @test c_eq ≈ 0.3
        ν = Field(g); fill!(ν.data, 1.0)
        iν = view(ν.data, interior(g)...)
        iν[:, :, 1:4] .= 0.0
        for (k, νk) in zip(5:8, (0.1, 1.0, 7.0, 50.0)); iν[:, :, k] .= νk; end
        sgs_scalar_variance!(var, φ; c_var = c_eq, νt = ν)
        iv = view(var.data, interior(g)...)
        @test all(iv[:, :, 1:4] .== 0)
        @test maximum(abs(2 * (iν[i, j, k] / Pr_t) * G^2 /
                          (C_χ * iν[i, j, k] / (C_k * Δ) / Δ * iv[i, j, k]) - 1)
                      for k in 5:8, j in 1:4, i in 1:4) < 1e-12
    end

    @testset "Gridded diagnosis: SGS off is saturation adjustment; water conserved" begin
        g = RegularGrid(; size = (1, 1, 40), extent = (30.0, 30.0, 1200.0))
        θf, qf = Field(g), Field(g)
        H(z) = 0.5 * (1 + tanh((z - 800.0) / 40.0))        # stratocumulus-top inversion
        fill_all!(θf, (x, y, z) -> 289.0 + 8.0 * H(z) + 0.004 * max(z - 800.0, 0.0))
        fill_all!(qf, (x, y, z) -> 0.0105 - 0.006 * H(z))
        pz = [1.0e5 * exp(-(g.z0 + (k - 0.5) * g.dz) / 8000) for k in 1:g.Nz]
        cf, qc, T, ql, qi = Field(g), Field(g), Field(g), Field(g), Field(g)
        ii, jj, kk = interior(g)
        subgrid_cloud!(cf, qc, θf, qf, pz; νt = Field(g), T = T, liquid = ql, ice = qi)
        exact = all(begin
                        Ta, qa = saturation_adjustment(θf.data[ii[1], jj[1], k], pz[kl], qf.data[ii[1], jj[1], k])
                        T.data[ii[1], jj[1], k] == Ta && ql.data[ii[1], jj[1], k] == qa.liq &&
                            cf.data[ii[1], jj[1], k] == (qa.liq + qa.ice > 0 ? 1.0 : 0.0)
                    end for (kl, k) in enumerate(kk))
        @test exact
        cf_off = copy(view(cf.data, ii, jj, kk))
        @test any(cf_off .== 1) && any(cf_off .== 0)
        subgrid_cloud!(cf, qc, θf, qf, pz; T = T, liquid = ql, ice = qi)
        cf_on = view(cf.data, ii, jj, kk); qc_on = view(qc.data, ii, jj, kk)
        @test count(c -> 1e-6 < c < 1 - 1e-6, cf_on) >= 1          # partial cloud at the gradient
        @test all(0 .<= qc_on .<= view(qf.data, ii, jj, kk))
        @test all(isapprox.(view(ql.data, ii, jj, kk) .+ view(qi.data, ii, jj, kk), qc_on; rtol = 1e-14))
        @test cf_on[1, 1, 5] == cf_off[1, 1, 5]                       # well-mixed: unchanged
    end
end
