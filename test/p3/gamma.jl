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
end
