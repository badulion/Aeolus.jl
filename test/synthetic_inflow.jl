# N-inflow-a (#252): synthetic inflow turbulence — the Xie & Castro (2008)
# digital filter and the Lund et al. (1998) transform.

using Test, Random
using Aeolus
using Aeolus.Transition: _df_autocorrelation

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
    for _ in 1:200
        τ = (rand(rng), rand(rng), rand(rng), 0.9 * randn(rng), 0.9 * randn(rng), 0.9 * randn(rng))
        ok = try
            lund_coefficients(τ); true
        catch e
            e isa ArgumentError || rethrow(); false
        end
        agree += ok == is_realizable(τ)
    end
    @test agree == 200
    # apply_lund!: u_i ← a_ij ψ_j
    u = fill(1.0, 2, 3); v = fill(2.0, 2, 3); w = fill(3.0, 2, 3)
    apply_lund!(u, v, w, a)
    @test u[1, 1] ≈ a[1, 1] && v[1, 1] ≈ a[2, 1] + 2a[2, 2] && w[1, 1] ≈ a[3, 1] + 2a[3, 2] + 3a[3, 3]
end

@testset "N-inflow-a digital-filter kernel (length convention)" begin
    for LΔ in (4.0,)
        b, n, N = digital_filter_kernel(LΔ, 1.0)
        R = _df_autocorrelation(b)
        @test sum(abs2, b) ≈ 1 rtol = 1e-14                     # unit variance
        @test R[1] ≈ 1 rtol = 1e-14
        @test sum(R) - 0.5 * R[1] ≈ LΔ rtol = 1e-10            # ∫R dr = L exactly
        # X&C nominal n = L/Δ (N = 2n) overestimates the integral length
        bn = Aeolus.Transition._df_kernel(LΔ, ceil(Int, 2LΔ))
        Inom = sum(_df_autocorrelation(bn)) - 0.5
        @test 1.10 < Inom / LΔ < 1.18
        # documented shape (1 + 2r/L) e^{-2r/L}, r ≤ 2L
        rs = 0:floor(Int, 2LΔ)
        rmsd = sqrt(sum(abs2, R[rs .+ 1] .- [(1 + 2r / LΔ) * exp(-2r / LΔ) for r in rs]) / length(rs))
        @test rmsd < 0.05
    end
    @test_throws ArgumentError digital_filter_kernel(0.5, 1.0)
end

# Plane statistics accumulated over time for the digital-filter tests.
function _df_moments(gen, nsteps, Δt; a = nothing)
    ns, nz = inflow_plane_size(gen)
    u = zeros(ns, nz); v = zeros(ns, nz); w = zeros(ns, nz)
    S = zeros(3); SS = zeros(3, 3); n = 0
    for _ in 1:nsteps
        unit_fluctuations!(u, v, w, gen, Δt)
        a === nothing || apply_lund!(u, v, w, a)
        @inbounds for k in 1:nz, j in 1:ns
            q = (u[j, k], v[j, k], w[j, k])
            for c in 1:3
                S[c] += q[c]
                for d in 1:3
                    SS[c, d] += q[c] * q[d]
                end
            end
        end
        n += ns * nz
    end
    m = S ./ n
    return m, SS ./ n .- m * m', n
end

@testset "N-inflow-a digital filter: Reynolds stresses, mean, decorrelation" begin
    us = 0.3; ns, nz = 64, 48; Δ = 1.0; Δt = 1.0
    LΔ, TΔt, nsteps = 2.0, 5.0, 2000
    R = _pd_tensor(us); a = lund_coefficients(R)
    # effective sample sizes from the generator's own correlations (Bartlett):
    # variance estimates use Σ_r R(r)², means Σ_r R(r)
    b, _, _ = digital_filter_kernel(LΔ * Δ, Δ)
    Rs = _df_autocorrelation(b)
    S2 = 2sum(abs2, Rs) - 1; S1 = 2sum(Rs) - 1
    ρ = exp(-Δt / TΔt); S2t = (1 + ρ^2) / (1 - ρ^2); S1t = (1 + ρ) / (1 - ρ)
    Ntot = ns * nz * nsteps
    Neff2 = Ntot / (S2 * S2 * S2t); Neff1 = Ntot / (S1 * S1 * S1t)

    gen = DigitalFilterGenerator(ns, nz, Δ, Δ; L_s = LΔ * Δ, L_z = LΔ * Δ, T = TΔt * Δt, seed = 11)
    m, C, _ = _df_moments(gen, nsteps, Δt)
    se_mean = 1 / sqrt(Neff1)
    offd = maximum(abs(C[i, j]) for (i, j) in ((1, 2), (1, 3), (2, 3)))
    @test all(abs.(m) .< 3se_mean)
    @test offd < 0.03
    @test all(abs(C[c, c] - 1) < 3sqrt(2 / Neff2) for c in 1:3)

    gen = DigitalFilterGenerator(ns, nz, Δ, Δ; L_s = LΔ * Δ, L_z = LΔ * Δ, T = TΔt * Δt, seed = 12)
    _, C, _ = _df_moments(gen, nsteps, Δt; a = a)
    err_d = sqrt(2 / Neff2)                                 # relative s.e. of a variance
    r13 = R[1, 3] / sqrt(R[1, 1] * R[3, 3])
    err_uw = sqrt((1 + r13^2) / r13^2 / Neff2)              # relative s.e. of a covariance
    dev = [C[c, c] / R[c, c] - 1 for c in 1:3]
    dev_uw = C[1, 3] / R[1, 3] - 1
    @test 0.05 >= 3err_d && 0.07 >= 3err_uw                 # tolerances ≥ 3 × sampling error
    @test all(abs.(dev) .< 0.05)
    @test abs(dev_uw) < 0.07
    @test abs(C[1, 2]) < 0.05 * sqrt(R[1, 1] * R[2, 2]) && abs(C[2, 3]) < 0.05 * sqrt(R[2, 2] * R[3, 3])
end

