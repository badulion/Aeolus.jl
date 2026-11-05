# P3 part-file tests: gamma function, regularised incomplete gamma,
# gamma-PSD moments (src/p3/gamma.jl, issue #1).
#
# References are independent of the code under test: exact factorials in
# BigInt, closed forms (P(1,x), the Poisson sum for integer a), Legendre's
# duplication formula, and direct quadrature of the size distribution.

@testset "Gamma functions and PSD moments" begin
    # Local to this testset: all test/p3/*.jl files share one module.
    relerr(x, ref) = abs(x - ref) / abs(ref)

    @testset "Γ and log Γ: known values and identities" begin
        known = [(1.0, 1.0), (0.5, sqrt(π)), (5.0, 24.0),
                 (1/3, 2.678938534707747633655692940974677644128689377957),
                 (-0.5, -2sqrt(π)), (10.0, 362880.0)]
        @test maximum(relerr(gamma_lanczos(x), r) for (x, r) in known) < 5e-15
        # log Γ(n) = log((n-1)!) from exact Int64 factorials (n ≤ 21)
        @test maximum(relerr(lgamma_lanczos(Float64(n)), log(Float64(factorial(n - 1))))
                      for n in (3, 12, 21)) < 3e-15
        @test_throws DomainError gamma_lanczos(-3)
        # Legendre duplication, an identity the implementation does not use
        @test maximum(abs(lgamma_lanczos(x) + lgamma_lanczos(x + 0.5) -
                          ((1 - 2x) * log(2.0) + 0.5 * log(π) + lgamma_lanczos(2x)))
                      for x in 0.07:0.113:20.0) < 1e-13
    end

    @testset "Incomplete gamma: closed forms and complement" begin
        # a = 1: P = 1 - e^{-x}, Q = e^{-x} (relative in the tail)
        xs = [10.0 .^ (-8:1:0); 2.0:4:40]
        @test maximum(max(relerr(gamma_inc_lower(1.0, x), -expm1(-x)),
                          relerr(gamma_inc_upper(1.0, x), exp(-x))) for x in xs) < 1e-13
        # integer a = 7: Q(n,x) = e^{-x} Σ_{k<n} x^k/k! (Poisson CDF); both
        # tails summed directly (positive terms), relative error of the smaller
        function poisson_tails(n, x)
            t = exp(-x); q = 0.0; p = 0.0
            for k in 0:n+300
                k < n ? (q += t) : (p += t)
                t *= x / (k + 1)
            end
            return p, q
        end
        maxe = maximum(begin
                           pb, qb = poisson_tails(7, x)
                           P, Q = gamma_inc_pq(7, x)
                           pb < qb ? relerr(P, pb) : relerr(Q, qb)
                       end for x in (0.5, 6.5, 8.0, 48.0))
        @test maxe < 1e-12   # Float64 reference sum: ~1e-15 rounding
        @test all(abs(sum(gamma_inc_pq(a, x)) - 1) <= 2eps()
                  for a in (0.1, 2.7, 100.0) for x in (1e-3, a, a + 1, 5a + 10))
        # continuity across the series / continued-fraction switch at x = a+1
        @test maximum(abs(gamma_inc_lower(a, prevfloat(a + 1.0, 4)) - gamma_inc_lower(a, a + 1.0))
                      for a in (0.3, 8.5, 40.0)) < 1e-13
    end

    @testset "Gamma PSD moments" begin
        function simpson(f, a, b, n)
            h = (b - a) / n
            s = f(a) + f(b)
            for i in 1:n-1
                s += (isodd(i) ? 4 : 2) * f(a + i * h)
            end
            return s * h / 3
        end
        maxe = 0.0; maxsum = 0.0
        for (N0, μ, λ) in ((8e6, 0.0, 2000.0), (3e20, 5.5, 5e4)), k in (0, 3, 6)
            ref = simpson(D -> D^k * gamma_psd_density(D, N0, μ, λ), 0.0, (μ + k + 60) / λ, 20_000)
            full = gamma_psd_moment(N0, μ, λ, k)
            maxe = max(maxe, relerr(full, ref))
            Dm = (μ + k + 1) / λ
            parts = [gamma_psd_partial_moment(N0, μ, λ, k, a, b)
                     for (a, b) in ((0.0, 0.2Dm), (0.2Dm, Dm), (Dm, 3Dm), (3Dm, Inf))]
            maxsum = max(maxsum, relerr(sum(parts), full))
        end
        @test maxe < 1e-10
        @test maxsum < 1e-14
        # Marshall-Palmer: LWC = π ρ_w N0/λ⁴; median-volume diameter λD₀ = 3.672
        # (Ulbrich 1983)
        N0, λ = 8e6, 2000.0
        @test π / 6 * 1000 * gamma_psd_moment(N0, 0, λ, 3) ≈ π * 1000 * N0 / λ^4 rtol = 1e-14
        @test gamma_psd_partial_moment(N0, 0, λ, 3, 0, 3.672060748850896 / λ) /
              gamma_psd_moment(N0, 0, λ, 3) ≈ 0.5 atol = 1e-14
        # intercept / slope closures round-trip (non-integer order b = 1.9)
        Nt, μ, λ = 2e8, 4.0, 4e4
        N0 = gamma_psd_intercept(Nt, μ, λ)
        @test gamma_psd_slope(gamma_psd_moment(N0, μ, λ, 1.9), 1.9, Nt, 0, μ) ≈ λ rtol = 1e-13
        # μ + i + 1 ≤ 0: M_0 does not exist (the unchecked formula returned 0.572)
        @test_throws DomainError gamma_psd_slope(1.0, 0, 2.0, 3, -1.5)
        # truncated moments over ulp-wide intervals are never negative
        # (262 of 33 600 were before the clamp)
        @test all(gamma_psd_partial_moment(1.0, μ, 1.0, k, (μ + k + 1) * (1 - j * 1e-15), μ + k + 1) >= 0
                  for μ in 0.3:1.37:40, k in 0:3, j in 1:20)
    end
end
