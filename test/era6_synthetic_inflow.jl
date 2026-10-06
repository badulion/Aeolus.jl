# N-inflow-a (#252): synthetic inflow turbulence — the generator contract,
# the Xie & Castro (2008) digital filter, the Lund et al. (1998)
# transform, and FreeStreamDisturbance as a second provider.

using Test, Printf, Random
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

@testset "N-inflow-a digital-filter kernel (length convention)" begin
    for LΔ in (2.0, 4.0, 8.0)
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
        @printf("    DF kernel L/Δ = %g: n = %.4f (πL/4Δ = %.4f), N = %d, nominal-n integral %.3f L, shape RMS %.4f\n",
                LΔ, n, π * LΔ / 4, N, Inom / LΔ, rmsd)
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
    @printf("    DF unit planes: mean (%.4f, %.4f, %.4f) [3σ/√N_eff = %.4f], var (%.4f, %.4f, %.4f), max |cross-corr| %.4f, N_eff = %.0f (means), %.0f (variances)\n",
            m..., 3se_mean, C[1, 1], C[2, 2], C[3, 3], offd, Neff1, Neff2)
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
    @printf("    DF + Lund (u* = 0.3): σ_u,v,w/u* = %.3f, %.3f, %.3f (2.39, 1.92, 1.25); u'w'/u*² = %.4f (-1); deviations %.2f/%.2f/%.2f %%, u'w' %.2f %%; expected s.e. %.2f %% (diag), %.2f %% (u'w')\n",
            sqrt(C[1, 1]) / us, sqrt(C[2, 2]) / us, sqrt(C[3, 3]) / us, C[1, 3] / us^2,
            100 .* dev..., 100dev_uw, 100err_d, 100err_uw)
    @test 0.05 >= 3err_d && 0.07 >= 3err_uw                 # tolerances ≥ 3 × sampling error
    @test all(abs.(dev) .< 0.05)
    @test abs(dev_uw) < 0.07
    @test abs(C[1, 2]) < 0.05 * sqrt(R[1, 1] * R[2, 2]) && abs(C[2, 3]) < 0.05 * sqrt(R[2, 2] * R[3, 3])
end

# Measured integral length along s (dim = 1) or z (dim = 2) of component c
# at levels `ks` (dim = 1) or around reference level `k0` (dim = 2),
# integrated (trapezoid) over the exact support 0..rmax of the correlation.
function _df_measure_length(gen, c, dim, rmax, nsteps, Δt; ks = 1:inflow_plane_size(gen)[2], k0 = 0)
    ns, nz = inflow_plane_size(gen)
    p = (zeros(ns, nz), zeros(ns, nz), zeros(ns, nz))
    acc = zeros(rmax + 1); cnt = zeros(rmax + 1)
    for _ in 1:nsteps
        unit_fluctuations!(p..., gen, Δt)
        ψ = p[c]
        @inbounds if dim == 1
            for k in ks, r in 0:rmax, j in 1:(ns-r)
                acc[r+1] += ψ[j, k] * ψ[j+r, k]; cnt[r+1] += 1
            end
        else    # vertical, both directions from k0
            for r in 0:rmax, j in 1:ns
                for kk in (k0 - r, k0 + r)
                    1 <= kk <= nz || continue
                    acc[r+1] += ψ[j, k0] * ψ[j, kk]; cnt[r+1] += 1
                end
            end
        end
    end
    Rm = (acc ./ cnt) ./ (acc[1] / cnt[1])
    return sum(Rm) - 0.5, Rm
end

@testset "N-inflow-a digital filter: length and time scales" begin
    ns, nz = 96, 96; Δs, Δz = 2.0, 1.0
    for LΔ in (2.0, 4.0, 8.0)
        gen = DigitalFilterGenerator(ns, nz, Δs, Δz; L_s = LΔ * Δs, L_z = LΔ * Δz, T = 1.0,
                                     seed = round(Int, 10LΔ))
        N = ceil(Int, 2LΔ)                    # kernel support; correlation is zero beyond 2N
        Ls, Rm = _df_measure_length(gen, 1, 1, 2N, 400, 2.0)
        Lz, _ = _df_measure_length(gen, 3, 2, 2N, 400, 2.0; k0 = nz ÷ 2)
        rs = 0:floor(Int, 2LΔ)
        shape = sqrt(sum(abs2, Rm[rs .+ 1] .- [(1 + 2r / LΔ) * exp(-2r / LΔ) for r in rs]) / length(rs))
        @printf("    DF scales L/Δ = %g: ∫R ds = %.3f L_s, ∫R dz = %.3f L_z, shape RMS vs (1+2r/L)e^{-2r/L} %.4f\n",
                LΔ, Ls / LΔ, Lz / LΔ, shape)
        @test abs(Ls / LΔ - 1) < 0.10
        @test abs(Lz / LΔ - 1) < 0.10
        @test shape < 0.05
    end
    # integral time: R(τ) = e^{-τ/T} for T/Δt ∈ {2, 4, 8}
    for TΔt in (2.0, 4.0, 8.0)
        gen = DigitalFilterGenerator(16, 16, 1.0, 1.0; L_s = 1.0, L_z = 1.0, T = TΔt * 0.5, seed = 3)
        M = ceil(Int, 10TΔt); nsteps = 4000
        p = (zeros(16, 16), zeros(16, 16), zeros(16, 16))
        hist = [zeros(16, 16) for _ in 1:(M+1)]
        acc = zeros(M + 1); cnt = 0
        for n in 1:nsteps
            unit_fluctuations!(p..., gen, 0.5)
            copyto!(hist[mod1(n, M + 1)], p[2])
            n > M || continue
            for m in 0:M
                acc[m+1] += sum(p[2] .* hist[mod1(n - m, M + 1)])
            end
            cnt += 1
        end
        Rt = acc ./ acc[1]
        Tm = 0.5 * (sum(Rt) - 0.5)
        @printf("    DF time: T = %.2f s requested (%g Δt), measured ∫R dτ = %.3f T, R(T) = %.3f (e^{-1} = 0.368)\n",
                TΔt * 0.5, TΔt, Tm / (TΔt * 0.5), Rt[round(Int, TΔt)+1])
        @test abs(Tm / (TΔt * 0.5) - 1) < 0.10
    end
    # Height- and component-dependent scales: L_z 4× larger in the upper
    # half; each region within 15 %. L_s differs per component.
    nz2 = 128; Lz = [k <= 48 ? 2.0 : 8.0 for k in 1:nz2]
    gen = DigitalFilterGenerator(48, nz2, 1.0, 1.0; L_s = (2.0, 3.0, 4.0), L_z = Lz, T = 1.0, seed = 5)
    @test gen.L_s[:, 1] == [2.0, 3.0, 4.0] && gen.L_z[1, 100] == 8.0
    Llo, _ = _df_measure_length(gen, 1, 2, 8, 1500, 2.0; k0 = 16)
    Lhi, _ = _df_measure_length(gen, 1, 2, 32, 1500, 2.0; k0 = 96)
    gen = DigitalFilterGenerator(48, nz2, 1.0, 1.0; L_s = (2.0, 3.0, 4.0), L_z = Lz, T = 1.0, seed = 6)
    Lw, _ = _df_measure_length(gen, 3, 1, 16, 300, 2.0)
    @printf("    DF height-dependent L_z: %.3f (2.0) at k = 16, %.3f (8.0) at k = 96; L_s(w) = %.3f (4.0)\n",
            Llo, Lhi, Lw)
    @test abs(Llo / 2 - 1) < 0.15
    @test abs(Lhi / 8 - 1) < 0.15
    @test abs(Lw / 4 - 1) < 0.15
end

@testset "N-inflow-a determinism and restart" begin
    mk() = DigitalFilterGenerator(20, 12, 1.0, 1.0; L_s = 2.0, L_z = (1.5, 2.0, 3.0),
                                  T = [0.5 + 0.1k for k in 1:12], seed = 42)
    g1, g2 = mk(), mk()
    P() = (zeros(20, 12), zeros(20, 12), zeros(20, 12))
    p1, p2 = P(), P()
    for _ in 1:5
        unit_fluctuations!(p1..., g1, 0.3); unit_fluctuations!(p2..., g2, 0.3)
    end
    @test all(p1[c] == p2[c] for c in 1:3)                   # same seed, same planes
    g3 = DigitalFilterGenerator(20, 12, 1.0, 1.0; L_s = 2.0, L_z = (1.5, 2.0, 3.0),
                                T = [0.5 + 0.1k for k in 1:12], seed = 43)
    p3 = P(); unit_fluctuations!(p3..., g3, 0.3)
    @test p3[1] != p1[1]
    # snapshot, advance (as a rejected adaptive step would), restore, continue
    s = generator_state(g1)
    unit_fluctuations!(p1..., g1, 0.7); unit_fluctuations!(p1..., g1, 0.2)
    restore_generator_state!(g1, s)
    for dt in (0.3, 0.1, 0.4)
        unit_fluctuations!(p1..., g1, dt); unit_fluctuations!(p2..., g2, dt)
    end
    @test all(p1[c] == p2[c] for c in 1:3)                   # bit-identical
    # a deep-copied state survives later snapshots
    keep = deepcopy(generator_state(g1))
    unit_fluctuations!(p1..., g1, 0.3); generator_state(g1)
    restore_generator_state!(g1, keep)
    unit_fluctuations!(p1..., g1, 0.3); unit_fluctuations!(p2..., g2, 0.3)
    @test all(p1[c] == p2[c] for c in 1:3)
    # Δt = 0 returns the current planes unchanged
    q = P(); unit_fluctuations!(q..., g1, 0.0)
    @test all(q[c] == p1[c] for c in 1:3)
    @test_throws DimensionMismatch unit_fluctuations!(zeros(3, 3), zeros(3, 3), zeros(3, 3), g1, 0.1)
    @test_throws ArgumentError DigitalFilterGenerator(8, 8, 1.0, 1.0; L_s = 2.0, L_z = 2.0, T = 0.0)

    # FreeStreamInflow: the clock is the state
    fi = FreeStreamInflow(FreeStreamDisturbance(0.05, 1.0, 100; seed = 3), 8, 6, 0.25, 0.25)
    fj = FreeStreamInflow(FreeStreamDisturbance(0.05, 1.0, 100; seed = 3), 8, 6, 0.25, 0.25)
    a1, a2 = (zeros(8, 6), zeros(8, 6), zeros(8, 6)), (zeros(8, 6), zeros(8, 6), zeros(8, 6))
    unit_fluctuations!(a1..., fi, 0.1); unit_fluctuations!(a2..., fj, 0.1)
    s = generator_state(fi); unit_fluctuations!(a1..., fi, 0.5); restore_generator_state!(fi, s)
    unit_fluctuations!(a1..., fi, 0.2); unit_fluctuations!(a2..., fj, 0.2)
    @test all(a1[c] == a2[c] for c in 1:3)

    # cost of one generator step on a 64 × 40 face (quoted, not asserted)
    gc = DigitalFilterGenerator(64, 40, 15.625, 6.25; L_s = 62.5, L_z = 25.0, T = 30.0, seed = 1)
    pc = (zeros(64, 40), zeros(64, 40), zeros(64, 40))
    unit_fluctuations!(pc..., gc, 1.0)
    t = @elapsed for _ in 1:200; unit_fluctuations!(pc..., gc, 1.0); end
    @printf("    DF cost: %.3f ms per step on a 64 × 40 face (L_s = 4Δs, L_z = 4Δz)\n", 1e3t / 200)
end

@testset "N-inflow-a FreeStreamDisturbance as a provider" begin
    Tu, L = 0.05, 1.0
    # the continuous mode sum agrees with the staggered grid curl to O((κΔ)²)
    fst = FreeStreamDisturbance(Tu, L, 200; seed = 4)
    g = RegularGrid(size = (4, 4, 4), extent = (4 / 64, 4 / 64, 4 / 64), halo = (1, 1, 1))
    u = Field(FaceX(), g); v = Field(FaceY(), g); w = Field(FaceZ(), g)
    disturbance_velocity!(u, v, w, fst)
    i = j = k = 3
    xu = g.x0 + (i - g.Hx - 1) * g.dx; yc = g.y0 + (j - g.Hy - 0.5) * g.dy; zc = g.z0 + (k - g.Hz - 0.5) * g.dz
    @test abs(u.data[i, j, k] - disturbance_velocity(fst, xu, yc, zc)[1]) < 0.02 * Tu
    C = disturbance_covariance(fst)
    @test 0.5 * (C[1, 1] + C[2, 2] + C[3, 3]) ≈ 1.5 * Tu^2 rtol = 1e-12        # ½ tr C = q

    # one-point statistics over a long sweep: whitened planes, then the
    # Panofsky-Dutton tensor through apply_lund!
    ns = nz = 32; Δ = 0.25; Δt = 0.25; nsteps = 2000; nb = 10
    fst = FreeStreamDisturbance(Tu, L, 400; seed = 1)
    gen = FreeStreamInflow(fst, ns, nz, Δ, Δ)
    R = _pd_tensor(0.3); a = lund_coefficients(R)
    p = (zeros(ns, nz), zeros(ns, nz), zeros(ns, nz))
    Cb = [zeros(3, 3) for _ in 1:nb]; Cu = zeros(3, 3); mu = zeros(3)
    for n in 1:nsteps
        unit_fluctuations!(p..., gen, Δt)
        for c in 1:3
            mu[c] += sum(p[c])
            for d in 1:3
                Cu[c, d] += sum(p[c] .* p[d])
            end
        end
        apply_lund!(p..., a)
        bb = cld(n * nb, nsteps)
        for c in 1:3, d in 1:3
            Cb[bb][c, d] += sum(p[c] .* p[d]) / (ns * nz * nsteps / nb)
        end
    end
    Nt = ns * nz * nsteps
    mu ./= Nt; Cu ./= Nt
    Cr = sum(Cb) ./ nb
    rel = [Cr[1, 1] / R[1, 1], Cr[2, 2] / R[2, 2], Cr[3, 3] / R[3, 3], Cr[1, 3] / R[1, 3]] .- 1
    # batch-means standard error of each ratio (10 consecutive blocks)
    se = map(((1, 1), (2, 2), (3, 3), (1, 3))) do (c, d)
        x = [Cb[b][c, d] / R[c, d] for b in 1:nb]
        sqrt(sum(abs2, x .- sum(x) / nb) / (nb - 1) / nb)
    end
    @printf("    FSD provider: unit-plane mean (%.4f, %.4f, %.4f), var (%.4f, %.4f, %.4f), cross (%.4f, %.4f, %.4f)\n",
            mu..., Cu[1, 1], Cu[2, 2], Cu[3, 3], Cu[1, 2], Cu[1, 3], Cu[2, 3])
    @printf("    FSD provider + Lund: R_11, R_22, R_33, u'w' deviations %.2f/%.2f/%.2f/%.2f %% (batch s.e. %.2f/%.2f/%.2f/%.2f %%)\n",
            100 .* rel..., 100 .* se...)
    @test all(abs.(rel) .< 0.05)
    @test all(0.05 .>= 3 .* se)
    @test maximum(abs, Cu .- [1.0 0 0; 0 1 0; 0 0 1]) < 0.05

    # L11 from the plane time series (Taylor: r = U∞ τ), as an ensemble over
    # mode sets. One realisation's L11 scatters with its random directions
    # (cf. P59-d: 0.82-1.43 L over 16 seeds with 400 modes), and the
    # correlation of the κ-band-limited von Kármán set first crosses zero
    # only at r ≈ 8.5 L (shell-averaged f(r)), so a short record or a
    # per-seed zero crossing is dominated by noise. Hence: 2000 modes,
    # records of 1600 L, 80 seeds; the seed-averaged correlation is
    # integrated to its first zero crossing, and the standard error is that
    # of the per-seed integrals to the same cutoff.
    nseed = 80; dt = 0.25; nst = 6400; M = 48           # lags to 12 L
    F = map(1:nseed) do s
        gi = FreeStreamInflow(FreeStreamDisturbance(Tu, L, 2000; seed = s), 2, 2, 0.25, 0.25)
        q = (zeros(2, 2), zeros(2, 2), zeros(2, 2))
        ser = zeros(4, nst)
        for n in 1:nst
            unit_fluctuations!(q..., gi, dt); ser[:, n] .= vec(q[1])
        end
        [sum(view(ser, :, 1:(nst-m)) .* view(ser, :, (1+m):nst)) / (4 * (nst - m)) for m in 0:M]
    end
    fbar = sum(F) ./ nseed; fbar ./= fbar[1]
    iz = something(findfirst(<(0), fbar), M + 2)
    L11m = fst.U∞ * dt * (sum(fbar[1:iz-1]) - 0.5)
    Is = [fst.U∞ * dt * (sum(f[1:iz-1]) / f[1] - 0.5) for f in F]
    sem = sqrt(sum(abs2, Is .- sum(Is) / nseed) / (nseed - 1) / nseed)
    @printf("    FSD provider: L11 (seed-averaged correlation, 80 × 2000 modes, zero crossing at %.2f L) = %.3f ± %.3f (requested %.3f); per-seed %.3f-%.3f\n",
            (iz - 1) * dt, L11m, sem, L, minimum(Is), maximum(Is))
    @test 0.05 >= 3sem
    @test abs(L11m / L - 1) < 0.05
end
