# N-inflow-a (#252): synthetic inflow turbulence — the generator contract,
# the Xie & Castro (2008) digital filter, the Lund et al. (1998)
# transform, and FreeStreamDisturbance as a second provider.

using Test, Printf, Random
using Aeolus

# Panofsky & Dutton (1984, §7) neutral surface layer: σ_u, σ_v, σ_w =
# 2.39, 1.92, 1.25 u*, u'w' = -u*² (components normal, along-face, vertical).
function _pd_tensor(us)
    su, sv, sw = 2.39us, 1.92us, 1.25us
    return [su^2 0.0 -us^2; 0.0 sv^2 0.0; -us^2 0.0 sw^2]
end

@testset "N-inflow-a Lund transform" begin
    R = _pd_tensor(0.3)
    a = lund_coefficients(R)
    @test a[1, 2] == a[1, 3] == a[2, 3] == 0
    @test maximum(abs, a * a' .- R) < 1e-12
    @test maximum(abs, lund_coefficients((R[1, 1], R[2, 2], R[3, 3], R[1, 2], R[1, 3], R[2, 3])) .- a) == 0
    # Lund Eq. 13 by hand: a11 = σ_u, a31 = u'w'/σ_u, a33 = √(σ_w² - a31²)
    @test a[1, 1] ≈ 2.39 * 0.3 rtol = 1e-14
    @test a[3, 1] ≈ -0.09 / (2.39 * 0.3) rtol = 1e-14
    @test a[3, 3] ≈ sqrt((1.25 * 0.3)^2 - (0.09 / (2.39 * 0.3))^2) rtol = 1e-14
    # Non-realizable (Schumann 1977): |u'w'| > σ_u σ_w, a negative variance,
    # a negative determinant with admissible 2 × 2 minors.
    @test_throws ArgumentError lund_coefficients([1.0 0 1.2; 0 1 0; 1.2 0 1])
    @test_throws ArgumentError lund_coefficients([-0.1 0 0; 0 1 0; 0 0 1])
    @test_throws ArgumentError lund_coefficients([1.0 0.9 0.9; 0.9 1 0.0; 0.9 0.0 1])
    @test_throws ArgumentError lund_coefficients([1.0 0.1 0; 0.2 1 0; 0 0 1])   # asymmetric
    # Semi-definite limits are realizable: w' = 0 (a wall), and v' ≡ u'.
    a0 = lund_coefficients([1.0 0 0; 0 2 0; 0 0 0])
    @test a0[3, :] == [0.0, 0.0, 0.0] && a0 * a0' ≈ [1.0 0 0; 0 2 0; 0 0 0]
    a1 = lund_coefficients([1.0 1 0; 1 1 0; 0 0 1])
    @test maximum(abs, a1 * a1' .- [1.0 1 0; 1 1 0; 0 0 1]) < 1e-12
    @test_throws ArgumentError lund_coefficients([0.0 0 0.1; 0 1 0; 0.1 0 1])  # u'w' with u' = 0
    # The Cholesky test agrees with Schumann's minors (PrognosticSGS.is_realizable)
    # on random symmetric tensors either side of the boundary.
    rng = Random.Xoshiro(7)
    agree = 0
    for _ in 1:2000
        τ = (rand(rng), rand(rng), rand(rng), 0.9 * randn(rng), 0.9 * randn(rng), 0.9 * randn(rng))
        ok = try
            lund_coefficients(τ); true
        catch e
            e isa ArgumentError || rethrow(); false
        end
        agree += ok == is_realizable(τ)
    end
    @test agree == 2000
    # apply_lund!: uniform and height-dependent tensors
    u = fill(1.0, 2, 3); v = fill(2.0, 2, 3); w = fill(3.0, 2, 3)
    apply_lund!(u, v, w, a)
    @test u[1, 1] ≈ a[1, 1] && v[1, 1] ≈ a[2, 1] + 2a[2, 2] && w[1, 1] ≈ a[3, 1] + 2a[3, 2] + 3a[3, 3]
    u = fill(1.0, 2, 3); v = fill(0.0, 2, 3); w = fill(0.0, 2, 3)
    apply_lund!(u, v, w, [k * a for k in 1:3])
    @test u[:, 3] ≈ [3a[1, 1], 3a[1, 1]] && w[:, 2] ≈ [2a[3, 1], 2a[3, 1]]
    @test_throws DimensionMismatch apply_lund!(u, v, w, [a, a])
end
