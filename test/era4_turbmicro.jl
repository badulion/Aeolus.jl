# Era IV, Phase 45 — turbulence-microphysics interaction.
#
# Issue #12: the sub-grid distribution of the saturation deficit, the
# partial cloud fraction and condensate it implies, and the SGS variance
# that drives it. Headline checks: the σ → 0 limit is the exact
# all-or-nothing saturation adjustment of the Phase 39 contract, the
# Gaussian case matches the Sommeria-Deardorff / Mellor closed forms, and
# the linearised PDF agrees with direct quadrature of the *exact*
# adjustment over the joint (q_t, θ_li) distribution.

using Test, Printf
using Aeolus

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

@testset "Era IV — Turbulent microphysics (Phase 45)" begin

    @testset "Saturation-deficit PDFs: normalisation and moments" begin
        s̄, σ = 2.0e-4, 5.0e-4
        pdfs = (GaussianSaturationPDF(),
                DoubleGaussianSaturationPDF(0.8),
                DoubleGaussianSaturationPDF(-1.5; shape = 0.5),
                DoubleGaussianSaturationPDF(3.0))
        worst_norm = 0.0
        for pdf in pdfs
            Sk = pdf isa GaussianSaturationPDF ? 0.0 : pdf.skewness
            lo, hi = s̄ - 12σ, s̄ + 12σ
            P(s) = saturation_pdf_density(pdf, s, s̄, σ)
            m0 = _simpson(P, lo, hi, 40_000)
            m1 = _simpson(s -> s * P(s), lo, hi, 40_000)
            m2 = _simpson(s -> (s - s̄)^2 * P(s), lo, hi, 40_000)
            m3 = _simpson(s -> (s - s̄)^3 * P(s), lo, hi, 40_000)
            worst_norm = max(worst_norm, abs(m0 - 1))
            @test m0 ≈ 1 atol = 1e-10
            @test m1 ≈ s̄ rtol = 1e-8
            @test m2 ≈ σ^2 rtol = 1e-8
            @test m3 / σ^3 ≈ Sk atol = 1e-7
            # closed-form cloud fraction and condensate against quadrature
            C_q = _simpson(P, 0.0, hi, 40_000)
            qc_q = _simpson(s -> s * P(s), 0.0, hi, 40_000)
            @test pdf_cloud_fraction(pdf, s̄, σ) ≈ C_q rtol = 1e-9
            @test pdf_condensate(pdf, s̄, σ) ≈ qc_q rtol = 1e-9
        end
        @info "PDF normalisation" worst_abs_error = worst_norm
    end

    @testset "Gaussian closed forms (Sommeria-Deardorff / Mellor)" begin
        pdf = GaussianSaturationPDF()
        σ = 3.0e-4
        # Reference values of Φ(Q) = ½(1+erf(Q/√2)) and of the standard
        # normal density, tabulated (Abramowitz & Stegun 26.2).
        table = ((-2.0, 0.022750131948179), (-1.0, 0.158655253931457),
                 (0.0, 0.5), (0.5, 0.691462461274013),
                 (1.0, 0.841344746068543), (2.0, 0.977249868051821))
        for (Q, Φ) in table
            s̄ = Q * σ
            @test pdf_cloud_fraction(pdf, s̄, σ) ≈ Φ rtol = 1e-12
            φ = exp(-Q^2 / 2) / sqrt(2π)
            @test pdf_condensate(pdf, s̄, σ) ≈ σ * (Q * Φ + φ) rtol = 1e-12
        end
        # Q = 0: half the cell is cloudy and holds σ/√(2π) of condensate.
        @test pdf_condensate(pdf, 0.0, σ) ≈ σ / sqrt(2π) rtol = 1e-14
        # Jensen: sub-grid variability only ever adds condensate, and the
        # excess over max(s̄,0) decays as the cell moves away from Q = 0.
        for Q in -3:0.25:3
            @test pdf_condensate(pdf, Q * σ, σ) >= max(Q * σ, 0.0)
        end
        @test pdf_condensate(pdf, 5σ, σ) ≈ 5σ rtol = 1e-6
        @test pdf_condensate(pdf, -5σ, σ) < 1e-6 * σ
        # exceedance at a threshold is the cloud fraction of the shifted mean
        @test exceedance_fraction(pdf, 1e-4, 2e-4, σ) ≈
              pdf_cloud_fraction(pdf, 1e-4, σ) rtol = 1e-14
    end

    @testset "Double Gaussian reduces to Gaussian and is continuous in Sk" begin
        g = GaussianSaturationPDF()
        s̄, σ = -1.0e-4, 4.0e-4
        d0 = DoubleGaussianSaturationPDF(0.0)
        @test pdf_cloud_fraction(d0, s̄, σ) == pdf_cloud_fraction(g, s̄, σ)
        @test pdf_condensate(d0, s̄, σ) == pdf_condensate(g, s̄, σ)
        # continuity as Sk → 0 (the separation vanishes as |Sk|^{1/3})
        dC = abs(pdf_cloud_fraction(DoubleGaussianSaturationPDF(1e-9), s̄, σ) -
                 pdf_cloud_fraction(g, s̄, σ))
        @test dC < 1e-3
        # Positive skew with a subsaturated mean: a moist tail clouds more
        # of the cell than the symmetric distribution does; negative skew
        # (a dry tail, moist bulk) less. The ordering is NOT monotone in
        # Sk beyond that — at large Sk the bulk component narrows and
        # stops reaching saturation — so only the sign is asserted.
        Cs = [pdf_cloud_fraction(DoubleGaussianSaturationPDF(Sk), -1.5σ, σ)
              for Sk in (-1.0, 0.0, 0.5, 1.0, 2.0)]
        @test all(Cs[3:end] .> Cs[2])
        @test Cs[1] < Cs[2]
        @info "cloud fraction at Q = -1.5 vs skewness -1, 0, .5, 1, 2" Cs
    end

    # ---------------------------------------------------------------------
    # Mellor linearisation and the cell diagnosis, against the contract
    # ---------------------------------------------------------------------
    p = 9.0e4
    T0 = 290.0
    qs0 = saturation_specific_humidity(T0, p)
    θ0 = liquid_ice_potential_temperature(T0, p, PhasePartition(qs0))

    @testset "σ_s → 0 is the all-or-nothing saturation adjustment, exactly" begin
        # warm sub- and supersaturated, mixed-phase (255 K), cold (228 K)
        states = Tuple{Float64,Float64,Float64}[]
        for (T, pr) in ((290.0, 9.0e4), (255.0, 6.0e4), (228.0, 3.0e4))
            qs = saturation_specific_humidity(T, pr)
            for f in (0.7, 0.999, 1.0005, 1.05, 1.3)
                qt = f * qs
                θ = liquid_ice_potential_temperature(T, pr, PhasePartition(qt))
                push!(states, (θ, pr, qt))
            end
        end
        worst_T = 0.0; worst_q = 0.0
        for (θ, pr, qt) in states
            T_adj, q_adj = saturation_adjustment(θ, pr, qt)
            r = subgrid_cloud(θ, pr, qt)                 # all variances zero
            @test r.σ_s == 0
            @test r.T == T_adj
            @test r.q == q_adj
            @test r.condensate == q_adj.liq + q_adj.ice
            @test r.cloud_fraction == (q_adj.liq + q_adj.ice > 0 ? 1.0 : 0.0)
            # ... and the limit is continuous: σ_s = 1e-12 kg/kg
            rε = subgrid_cloud(θ, pr, qt; σ_s = 1e-12)
            worst_T = max(worst_T, abs(rε.T - T_adj))
            worst_q = max(worst_q, abs(rε.condensate - r.condensate))
            @test rε.T ≈ T_adj atol = 1e-6
            @test rε.condensate ≈ r.condensate atol = 1e-11
            @test rε.q.liq ≈ q_adj.liq atol = 1e-11
            @test rε.q.ice ≈ q_adj.ice atol = 1e-11
        end
        @info "σ_s = 1e-12 vs saturation_adjustment" max_dT = worst_T max_dqc = worst_q
    end

    @testset "Mellor coefficients agree with the contract's derivatives" begin
        m = mellor_coefficients(θ0, p, qs0)
        # ∂q_s/∂T against a central difference of the contract's q_sat
        h = 1e-3
        fd = (saturation_specific_humidity(m.T_l + h, p) -
              saturation_specific_humidity(m.T_l - h, p)) / 2h
        @test m.dqsdT ≈ fd rtol = 1e-6
        # At the saturation boundary the linearised s must have the slope
        # of the EXACT adjustment's condensate: one-sided differences of
        # `saturation_adjustment` into the cloudy side.
        qc(θ, qt) = (q = saturation_adjustment(θ, p, qt)[2]; q.liq + q.ice)
        dq = 1e-8
        slope_q = (qc(θ0, qs0 + dq) - qc(θ0, qs0)) / dq
        dθ = 1e-5
        slope_θ = (qc(θ0 - dθ, qs0) - qc(θ0, qs0)) / dθ
        @test slope_q ≈ m.a_q rtol = 1e-5
        @test slope_θ ≈ m.b rtol = 1e-4
        # and bare L/c_pm would not have been good enough
        cpm = heat_capacity_moist(PhasePartition(qs0))
        a_naive = 1 / (1 + latent_heat_vapor(m.T_l) / cpm * m.dqsdT)
        @test abs(a_naive - slope_q) / slope_q > 5e-3
        @info "boundary slope ∂q_c/∂q_t" exact = slope_q mellor = m.a_q naive_L_over_cp = a_naive
        # a < 1: latent heating eats part of any moisture excess
        @test 0 < m.a < 1
        @test 0.25 < m.a < 0.4        # ≈ 0.32 near 290 K, 900 hPa
    end

    @testset "Gaussian cell diagnosis matches the analytic expressions" begin
        σq = 3.0e-4
        for f in (0.96, 0.99, 1.0, 1.02, 1.06)
            qt = f * qs0
            r = subgrid_cloud(θ0, p, qt; var_qt = σq^2)
            Q = r.s̄ / r.σ_s
            @test r.Q ≈ Q
            @test r.σ_s ≈ r.coefficients.a_q * σq rtol = 1e-14
            Φ = 0.5 * (1 + Aeolus.GasOptics.erf_series(Q / sqrt(2)))
            @test r.cloud_fraction ≈ Φ rtol = 1e-12
            @test r.condensate ≈ r.σ_s * (Q * Φ + exp(-Q^2 / 2) / sqrt(2π)) rtol = 1e-12
        end
    end

    @testset "Cloud fraction monotone in the mean saturation deficit" begin
        σq, σθ, ρ = 2.5e-4, 0.15, 0.4
        kw = (var_qt = σq^2, var_θ = σθ^2, cov_qθ = ρ * σq * σθ)
        # moistening at fixed θ_li
        Cq = [subgrid_cloud(θ0, p, f * qs0; kw...).cloud_fraction for f in 0.9:0.005:1.1]
        @test all(diff(Cq) .> 0)
        # warming at fixed q_t
        Cθ = [subgrid_cloud(θ0 + dθ, p, qs0; kw...).cloud_fraction for dθ in -2:0.1:2]
        @test all(diff(Cθ) .< 0)
        # and the ends are clear and overcast
        @test Cq[1] < 1e-3 && Cq[end] > 1 - 1e-3
        # a skewed PDF too
        pdf = DoubleGaussianSaturationPDF(1.2)
        Cs = [subgrid_cloud(θ0, p, f * qs0; kw..., pdf = pdf).cloud_fraction
              for f in 0.9:0.005:1.1]
        # strictly increasing until it rounds to exactly 1.0 in Float64
        @test all(diff(Cs) .>= 0)
        inner = findall(c -> 1e-12 < c < 1 - 1e-12, Cs)
        @test all(diff(Cs[inner]) .> 0)
        @test length(inner) >= 30
    end

    @testset "Total water and θ_li are conserved" begin
        worst_q = 0.0; worst_θ = 0.0
        for (T, pr) in ((290.0, 9.0e4), (262.0, 7.0e4), (240.0, 4.0e4))
            qs = saturation_specific_humidity(T, pr)
            for f in (0.9, 1.0, 1.1), σq in (1e-5, 1e-4, 1e-3)
                qt = f * qs
                θ = liquid_ice_potential_temperature(T, pr, PhasePartition(qt))
                for pdf in (GaussianSaturationPDF(), DoubleGaussianSaturationPDF(1.0))
                    r = subgrid_cloud(θ, pr, qt; var_qt = σq^2, pdf = pdf)
                    @test r.q.tot == qt
                    vap = r.q.tot - r.q.liq - r.q.ice
                    worst_q = max(worst_q, abs(vap + r.condensate - qt))
                    @test vap + r.q.liq + r.q.ice == qt || abs(vap + r.q.liq + r.q.ice - qt) < 4eps(qt)
                    @test r.q.liq + r.q.ice ≈ r.condensate rtol = 1e-14
                    @test 0 <= r.condensate <= qt
                    @test vap >= 0
                    θr = liquid_ice_potential_temperature(r.T, pr, r.q)
                    worst_θ = max(worst_θ, abs(θr - θ) / θ)
                    @test θr ≈ θ rtol = 1e-11
                end
            end
        end
        @info "conservation" max_water_residual = worst_q max_rel_theta_li_residual = worst_θ
        # condensate is capped at the water that is there
        tiny = 1e-5
        θt = liquid_ice_potential_temperature(290.0, p, PhasePartition(tiny))
        r = subgrid_cloud(θt, p, tiny; σ_s = 1.0)
        @test r.condensate == tiny
    end

    @testset "Linearised PDF vs exact adjustment over the joint (q_t, θ_li) PDF" begin
        # Reference: integrate the EXACT, nonlinear saturation_adjustment
        # over a bivariate normal in (q_t, θ_li). Cloud fraction from the
        # boundary curve (bisection per z₂, then Φ in z₁); condensate by
        # 2-D Simpson. Differences are the Mellor linearisation of the
        # fluctuations only — they should shrink with σ.
        φ(z) = exp(-z^2 / 2) / sqrt(2π)
        Φ(z) = 0.5 * Aeolus.GasOptics.erfc_series(-z / sqrt(2))
        qc(θ, qt) = (q = saturation_adjustment(θ, p, qt)[2]; q.liq + q.ice)
        function reference(qt, σq, σθ, ρ)
            state(z1, z2) = (θ0 + σθ * (ρ * z1 + sqrt(1 - ρ^2) * z2), qt + σq * z1)
            function zstar(z2)            # cloud boundary in z₁ at this z₂
                lo, hi = -12.0, 12.0
                for _ in 1:60
                    mid = 0.5 * (lo + hi)
                    qc(state(mid, z2)...) > 0 ? (hi = mid) : (lo = mid)
                end
                return 0.5 * (lo + hi)
            end
            C = _simpson(z2 -> φ(z2) * Φ(-zstar(z2)), -7.0, 7.0, 200)
            n, L = 160, 7.0
            h = 2L / n
            w(i) = (i == 0 || i == n) ? 1 : (isodd(i) ? 4 : 2)
            Qc = 0.0
            for i in 0:n, j in 0:n
                z1 = -L + i * h; z2 = -L + j * h
                Qc += w(i) * w(j) * qc(state(z1, z2)...) * φ(z1) * φ(z2)
            end
            return C, Qc * (h / 3)^2
        end
        rows = []
        for (f, σq, σθ, ρ) in ((1.00, 1e-4, 0.0, 0.0),
                               (1.00, 3e-4, 0.0, 0.0),
                               (1.00, 1e-3, 0.0, 0.0),
                               (1.00, 3e-4, 0.2, 0.5),
                               (0.97, 3e-4, 0.2, -0.3),
                               (1.03, 5e-4, 0.3, 0.0))
            qt = f * qs0
            C_ref, qc_ref = reference(qt, σq, σθ, ρ)
            r = subgrid_cloud(θ0, p, qt; var_qt = σq^2, var_θ = σθ^2,
                              cov_qθ = ρ * σq * σθ)
            dC = r.cloud_fraction - C_ref
            eq = (r.condensate - qc_ref) / qc_ref
            push!(rows, (f, σq, σθ, ρ, C_ref, r.cloud_fraction, dC, qc_ref, r.condensate, eq))
            @test abs(dC) < 0.01
            @test abs(eq) < 0.02
        end
        # error is second order in the fluctuation: ~linear growth in σ
        e1, e3, e10 = abs(rows[1][10]), abs(rows[2][10]), abs(rows[3][10])
        @test e1 < e3 < e10
        @test e1 < 3e-3
        for r in rows
            @info @sprintf("q_t/q_s=%.2f σq=%.0e σθ=%.2f ρ=%+.1f | C ref %.4f pdf %.4f (Δ %+.4f) | q̄c ref %.4e pdf %.4e (rel %+.2f%%)",
                           r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], 100 * r[10])
        end
    end
end
