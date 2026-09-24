# P3 part-file tests: gamma function, regularised incomplete gamma,
# gamma-PSD moments (src/p3/gamma.jl, issue #1).
#
# References are independent of the code under test wherever possible:
# exact factorials in BigInt, closed forms (P(1,x), the Poisson sum for
# integer a, erf for a = 1/2 from GasOptics), Legendre's duplication
# formula, and direct numerical quadrature of the size distribution.

relerr(x, ref) = abs(x - ref) / abs(ref)

@testset "Gamma functions and PSD moments" begin

    @testset "Γ at known values" begin
        known = [(1.0, 1.0), (0.5, sqrt(π)), (5.0, 24.0), (1.5, sqrt(π) / 2),
                 (1/3, 2.678938534707747633655692940974677644128689377957),
                 (1/4, 3.625609908221908311930685155867672002995167682880),
                 (2.5, 3sqrt(π) / 4), (-0.5, -2sqrt(π)), (-1.5, 4sqrt(π) / 3),
                 (10.0, 362880.0)]
        maxe = 0.0
        for (x, ref) in known
            e = relerr(gamma_lanczos(x), ref)
            maxe = max(maxe, e)
            @test e < 5e-15
        end
        @printf("    Γ known values: max rel err %.2e\n", maxe)
        @test gamma_lanczos(1) == 1.0
        @test gamma_lanczos(5) == 24.0
        @test_throws DomainError gamma_lanczos(0)
        @test_throws DomainError gamma_lanczos(-3)
        @test lgamma_lanczos(0) == Inf
        @test lgamma_lanczos(-2.0) == Inf
        # Float32 in → Float32 out
        @test gamma_lanczos(0.5f0) isa Float32
        @test gamma_lanczos(0.5f0) ≈ Float32(sqrt(π)) rtol = 2e-6
        @test lgamma_lanczos(2.5f0) isa Float32
    end

    @testset "log Γ against exact factorials" begin
        # log Γ(n) = log((n-1)!) computed in BigFloat from exact BigInt
        # factorials. Near the zeros of log Γ (x = 1, 2) test absolute error.
        @test abs(lgamma_lanczos(1.0)) < 1e-15
        @test abs(lgamma_lanczos(2.0)) < 1e-15
        maxe = 0.0
        for n in [3:30; 50; 100; 171; 500; 1000; 10_000]
            ref = Float64(log(big(factorial(big(n - 1)))))
            e = relerr(lgamma_lanczos(Float64(n)), ref)
            maxe = max(maxe, e)
            @test e < 3e-15
        end
        @printf("    log Γ(n) vs log((n-1)!) for n ∈ 3…10⁴: max rel err %.2e\n", maxe)
        # Direct factorial path agrees with the exact integer factorials
        for n in 1:25
            @test gamma_lanczos(n) ≈ Float64(factorial(big(n - 1))) rtol = 1e-15
        end
    end

    @testset "Recurrence, reflection and duplication" begin
        # Γ(x+1) = x Γ(x) over a sweep that crosses the reflection branch
        # (x < 1/2) and negative non-integers.
        xs = [collect(0.013:0.0731:30.0); collect(-4.93:0.137:-0.05)]
        filter!(x -> !isinteger(x), xs)
        maxrec = 0.0
        for x in xs
            maxrec = max(maxrec, relerr(gamma_lanczos(x + 1), x * gamma_lanczos(x)))
        end
        @printf("    recurrence Γ(x+1)=xΓ(x), %d pts: max rel err %.2e\n",
                length(xs), maxrec)
        @test maxrec < 2e-13

        # Euler reflection Γ(x)Γ(1-x) = π / sin(πx)
        maxref = 0.0
        for x in 0.0123:0.0371:0.99
            maxref = max(maxref,
                         relerr(gamma_lanczos(x) * gamma_lanczos(1 - x), π / sinpi(x)))
        end
        @printf("    reflection Γ(x)Γ(1-x)=π/sin πx: max rel err %.2e\n", maxref)
        @test maxref < 1e-14

        # Legendre duplication Γ(x)Γ(x+½) = 2^{1-2x} √π Γ(2x) — an identity
        # the implementation does not use, so it is an independent check.
        maxdup = 0.0
        for x in 0.07:0.113:20.0
            lhs = lgamma_lanczos(x) + lgamma_lanczos(x + 0.5)
            rhs = (1 - 2x) * log(2.0) + 0.5 * log(π) + lgamma_lanczos(2x)
            maxdup = max(maxdup, abs(lhs - rhs))
        end
        @printf("    duplication formula (log form): max abs err %.2e\n", maxdup)
        @test maxdup < 1e-13

        # Stirling series at large argument
        x = 2345.6
        stirling = (x - 0.5) * log(x) - x + 0.5 * log(2π) + 1 / (12x) - 1 / (360x^3)
        @test relerr(lgamma_lanczos(x), stirling) < 1e-15
    end

    @testset "Incomplete gamma: limits and complement" begin
        for a in (0.1, 0.5, 1.0, 2.7, 10.0, 37.5, 100.0)
            @test gamma_inc_lower(a, 0.0) == 0.0
            @test gamma_inc_upper(a, 0.0) == 1.0
            @test gamma_inc_lower(a, Inf) == 1.0
            @test gamma_inc_upper(a, Inf) == 0.0
            # far tails
            @test gamma_inc_lower(a, 50a + 500) ≈ 1.0 atol = 1e-15
            # near zero P(a,x) → x^a/Γ(a+1) · (1 - a x/(a+1) + …); note this
            # is NOT small for small a (P(0.1, 1e-7) ≈ 0.21): the Gamma(a,1)
            # density is integrably singular at 0 when a < 1.
            # (two-term expansion: truncation error O(x²) ≈ 1e-12 at x = 1e-6)
            x0 = 1e-6
            lead = exp(a * log(x0) - lgamma_lanczos(a + 1)) * (1 - a * x0 / (a + 1))
            if lead > 1e-290
                @test relerr(gamma_inc_lower(a, x0), lead) < 1e-11
            else                     # x^a underflows (a = 100): P must too
                @test gamma_inc_lower(a, x0) < 1e-290
            end
            for x in (1e-3, 0.3, 0.9a, a, a + 1, 1.3a + 2, 5a + 10)
                P, Q = gamma_inc(a, x)
                @test abs(P + Q - 1) <= 2eps()
                @test 0 <= P <= 1 && 0 <= Q <= 1
            end
        end
        @test_throws DomainError gamma_inc(0.0, 1.0)
        @test_throws DomainError gamma_inc(1.0, -1.0)
    end

    @testset "Incomplete gamma: closed forms" begin
        # a = 1: P(1,x) = 1 - e^{-x}; the upper tail Q = e^{-x} relative.
        maxe = 0.0
        for x in [10.0 .^ (-8:0.25:0); 1.5:0.5:40]
            e = relerr(gamma_inc_lower(1.0, x), -expm1(-x))
            eq = relerr(gamma_inc_upper(1.0, x), exp(-x))
            maxe = max(maxe, e, eq)
        end
        @printf("    P(1,x)=1-e^{-x}, Q(1,x)=e^{-x}: max rel err %.2e\n", maxe)
        @test maxe < 1e-13

        # a = 1/2: P(½,x) = erf(√x), with erf from GasOptics (the package's
        # one error-function implementation). The upper tail Q(½,x) =
        # erfc(√x) is checked against a 256-bit Taylor series for erf
        # instead: GasOptics.erfc_series forms erfc as 1 - erf for z ≤ 3 and
        # so carries up to ~1e-9 relative error there (measured at z = 3),
        # which would mask the accuracy of Q.
        maxe = 0.0; maxq = 0.0
        setprecision(BigFloat, 256) do
            berf(z) = 2 / sqrt(big(π)) *
                      sum((-1)^n * z^(2n + 1) / (factorial(big(n)) * (2n + 1)) for n in 0:400)
            for x in [10.0 .^ (-6:0.25:0); 1.25:0.25:30]
                maxe = max(maxe, relerr(gamma_inc_lower(0.5, x), erf_series(sqrt(x))))
                C = 1 - berf(sqrt(big(x)))
                maxq = max(maxq, Float64(relerr(big(gamma_inc_upper(0.5, x)), C)))
            end
        end
        @printf("    P(½,x) vs GasOptics erf(√x): max rel err %.2e\n", maxe)
        @printf("    Q(½,x) vs 256-bit erfc(√x), x ≤ 30: max rel err %.2e\n", maxq)
        @test maxe < 1e-13
        @test maxq < 1e-13

        # Integer a: Q(n,x) = e^{-x} Σ_{k<n} x^k/k! (the Poisson CDF),
        # evaluated in BigFloat as the reference. Relative error in the
        # SMALLER of P and Q — the tail that matters.
        maxe = 0.0
        setprecision(BigFloat, 256) do
            for n in (2, 3, 7, 15, 30, 50), x in (0.05, 0.5, 1.0, 0.5n, n - 0.5,
                                                  n + 1.0, 1.5n, 2n + 3, 4n + 20)
                xb = big(x)
                # both tails summed directly (1 - qb would lose a tiny P)
                qb = exp(-xb) * sum(xb^k / factorial(big(k)) for k in 0:n-1)
                pb = exp(-xb) * sum(xb^k / factorial(big(k)) for k in n:n+800)
                P, Q = gamma_inc(n, x)
                e = pb < qb ? relerr(P, Float64(pb)) : relerr(Q, Float64(qb))
                maxe = max(maxe, e)
            end
        end
        @printf("    Q(n,x) Poisson sum, n ≤ 50, smaller tail: max rel err %.2e\n", maxe)
        @test maxe < 1e-12
    end

    @testset "Incomplete gamma: continuity and derivative" begin
        # Continuity across the series / continued-fraction switch at x = a+1
        for a in (0.3, 2.0, 8.5, 40.0)
            xm = prevfloat(a + 1.0, 4); xp = a + 1.0
            @test abs(gamma_inc_lower(a, xm) - gamma_inc_lower(a, xp)) < 1e-13
        end
        # dP/dx = x^{a-1} e^{-x} / Γ(a) (the Gamma(a,1) pdf), central differences
        maxe = 0.0
        for a in (0.7, 3.0, 12.0), x in (0.4, 1.0, a, a + 1, 2a + 1)
            h = 1e-5 * max(x, 1.0)
            dP = (gamma_inc_lower(a, x + h) - gamma_inc_lower(a, x - h)) / 2h
            pdf = exp((a - 1) * log(x) - x - lgamma_lanczos(a))
            maxe = max(maxe, relerr(dP, pdf))
        end
        @printf("    dP/dx vs gamma pdf (FD h=1e-5x): max rel err %.2e\n", maxe)
        @test maxe < 1e-7
        # monotone in x
        ps = [gamma_inc_lower(4.3, x) for x in 0:0.1:30]
        @test all(diff(ps) .>= 0)
    end

    @testset "Gamma PSD moments" begin
        # Composite Simpson quadrature of the PSD, the independent reference
        function simpson(f, a, b, n)
            h = (b - a) / n
            s = f(a) + f(b)
            for i in 1:n-1
                s += (isodd(i) ? 4 : 2) * f(a + i * h)
            end
            return s * h / 3
        end

        maxe = 0.0
        for (N0, μ, λ) in ((8e6, 0.0, 2000.0), (1e12, 2.0, 1e4), (3e20, 5.5, 5e4)),
            k in (0, 1, 2, 3, 6)
            Dmax = (μ + k + 60) / λ
            ref = simpson(D -> D^k * gamma_psd_density(D, N0, μ, λ), 0.0, Dmax, 20_000)
            maxe = max(maxe, relerr(gamma_psd_moment(N0, μ, λ, k), ref))
        end
        @printf("    full moments vs Simpson quadrature: max rel err %.2e\n", maxe)
        @test maxe < 1e-10

        # Truncated moments vs quadrature, and summing to the full moment
        maxq = 0.0; maxsum = 0.0
        for (N0, μ, λ) in ((8e6, 0.0, 2000.0), (1e12, 2.0, 1e4), (3e20, 5.5, 5e4)),
            k in (0, 1.9, 3, 6)
            Dm = (μ + k + 1) / λ                     # near the mode of D^k N(D)
            edges = (0.0, 0.2Dm, Dm, 3Dm, 10Dm, Inf)
            parts = [gamma_psd_partial_moment(N0, μ, λ, k, edges[i], edges[i+1])
                     for i in 1:5]
            full = gamma_psd_moment(N0, μ, λ, k)
            maxsum = max(maxsum, relerr(sum(parts), full))
            for i in 1:4
                ref = simpson(D -> D^k * gamma_psd_density(D, N0, μ, λ),
                              edges[i], edges[i+1], 20_000)
                maxq = max(maxq, relerr(parts[i], ref))
            end
            @test all(parts .>= 0)
            @test gamma_psd_partial_moment(N0, μ, λ, k, Dm, Dm) == 0
            @test gamma_psd_partial_moment(N0, μ, λ, k, 0.0, Inf) ≈ full rtol = 1e-14
        end
        @printf("    truncated moments vs Simpson: max rel err %.2e\n", maxq)
        @printf("    Σ truncated moments vs full moment: max rel err %.2e\n", maxsum)
        @test maxq < 1e-9
        @test maxsum < 1e-14

        # Far upper tail keeps relative accuracy (Q branch, no cancellation)
        N0, μ, λ = 1e8, 1.0, 5000.0
        tail = gamma_psd_partial_moment(N0, μ, λ, 3, 40 / λ, Inf)
        # ∫_{x₀}^∞ x^4 e^{-x} dx = e^{-x₀}(x₀⁴+4x₀³+12x₀²+24x₀+24), x₀ = 40
        x0 = 40.0
        exact = N0 / λ^5 * exp(-x0) * (x0^4 + 4x0^3 + 12x0^2 + 24x0 + 24)
        @test relerr(tail, exact) < 1e-12

        # Physics: Marshall-Palmer exponential rain (μ = 0, N₀ = 8e6 m⁻⁴)
        N0, λ, ρw = 8e6, 2000.0, 1000.0
        @test gamma_psd_moment(N0, 0, λ, 0) ≈ N0 / λ rtol = 1e-14
        lwc = π / 6 * ρw * gamma_psd_moment(N0, 0, λ, 3)
        @test lwc ≈ π * ρw * N0 / λ^4 rtol = 1e-14
        # mean diameter (μ+1)/λ
        @test gamma_psd_moment(1.0, 2.5, λ, 1) / gamma_psd_moment(1.0, 2.5, λ, 0) ≈
              3.5 / λ rtol = 1e-14
        # median-volume diameter of the exponential: λ D₀ = 3.67 (Ulbrich
        # 1983), i.e. P(4, λD₀) = ½; root 3.672060748850896 found by 256-bit
        # bisection of the closed form 1 - e^{-x}(1+x+x²/2+x³/6) = ½.
        D0 = 3.672060748850896 / λ
        half = gamma_psd_partial_moment(N0, 0, λ, 3, 0, D0) / gamma_psd_moment(N0, 0, λ, 3)
        @test half ≈ 0.5 atol = 1e-14

        # Intercept and slope closures round-trip
        for (Nt, μ, λ) in ((1e5, 0.0, 3e3), (2e8, 4.0, 4e4), (50.0, 12.0, 800.0))
            N0 = gamma_psd_intercept(Nt, μ, λ)
            @test gamma_psd_moment(N0, μ, λ, 0) ≈ Nt rtol = 1e-13
            M3 = gamma_psd_moment(N0, μ, λ, 3)
            @test gamma_psd_slope(Nt, 0, M3, 3, μ) ≈ λ rtol = 1e-13
            # non-integer order (aggregate mass exponent b = 1.9) and i > j
            Mb = gamma_psd_moment(N0, μ, λ, 1.9)
            @test gamma_psd_slope(Mb, 1.9, Nt, 0, μ) ≈ λ rtol = 1e-13
        end
        @test_throws DomainError gamma_psd_moment(1.0, -2.0, 1.0, 0)
        @test_throws ArgumentError gamma_psd_slope(1.0, 3, 1.0, 3, 0.0)
        # μ + i + 1 ≤ 0: M_0 does not exist, so no λ can be diagnosed (the
        # unchecked formula silently returned λ = 0.572 from log|Γ(-0.5)|)
        @test_throws DomainError gamma_psd_slope(1.0, 0, 2.0, 3, -1.5)
        @test_throws DomainError gamma_psd_intercept(1.0, 2.0, 0.0)
        @test_throws DomainError gamma_psd_intercept(1.0, 2.0, -3.0)
        @test gamma_lanczos(Inf) == Inf

        # Truncated moments are never negative, even for intervals only a
        # few ulp wide where P(a,x₂) - P(a,x₁) is pure rounding. Before the
        # clamp, 262 of these 33 600 intervals returned negative values
        # (down to -1.5e-14 of the full moment).
        nneg = 0
        for μ in 0.3:0.37:40, k in 0:3, j in 1:200
            x2 = μ + k + 1
            x1 = x2 - j * 1e-15 * x2
            nneg += gamma_psd_partial_moment(1.0, μ, 1.0, k, x1, x2) < 0
        end
        @test nneg == 0
    end
end
