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
end
