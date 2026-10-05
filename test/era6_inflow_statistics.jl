# N-inflow-b (#258): Reynolds stresses, lengths and times for the synthetic
# inflow of the child boundaries — the `InflowStatistics` contract, the
# parent-resolved statistics (resolved + SGS, scale-gap mode) and the
# parametrized statistics of a mesoscale parent (Rotach et al. 1996 blend).

using Test, Printf, Random
using Aeolus
using Aeolus.Fields: interior_view

@testset "N-inflow-b sgs_tke!: every closure's subgrid TKE" begin
    m = RegularGrid(; size = (8, 8, 8), extent = (80.0, 80.0, 40.0), halo = (3, 3, 3))
    sim = BoussinesqSimulation(m; C_s = 0.17,
                               bcs = FieldBoundaryConditions(:periodic, :periodic, :periodic))
    # uniform shear u = s z: |S| = s, Smagorinsky ν_t = (C_s Δ)² s
    s = 0.02
    zc = [(k - 0.5) * m.dz for k in 1:m.Nz]
    for k in 1:m.Nz
        sim.u.data[:, :, m.Hz+k] .= s * zc[k]
    end
    Aeolus.BoundaryConditions.fill_halos!(sim.u, sim.bcs)
    c = StaticSmagorinsky(0.17)
    f = sgs_fields(c, sim)
    sgs_coefficients!(f, c, sim)
    e = Aeolus.Fields.Field(Aeolus.Grids.CellCenter(), m)
    sgs_tke!(e, c, sim)
    Δ = cbrt(m.dx * m.dy * m.dz)
    # interior levels (the periodic wrap of u = s z breaks the shear at k = 1, Nz)
    ei = interior_view(e)[:, :, 2:end-1]
    @test maximum(abs, ei ./ (0.17^2 * Δ * s / 0.1)^2 .- 1) < 1e-12
    # the built-in closure (c = nothing) reads sim.νt
    interior_view(sim.νt) .= interior_view(f.νh)
    e2 = Aeolus.Fields.Field(Aeolus.Grids.CellCenter(), m)
    sgs_tke!(e2, nothing, sim)
    @test interior_view(e2) == interior_view(e)
    # prognostic closures return their memory
    tk = TKEClosure(m; e0 = 0.37)
    sgs_tke!(e, tk, sim)
    @test all(==(0.37), interior_view(e))
    rs = ReynoldsStressClosure(m; e0 = 0.37)
    sgs_tke!(e, rs, sim)
    @test maximum(abs, interior_view(e) .- 0.37) < 1e-15
end

# a fully periodic parent on unit spacing
function _parent(Nx, Ny, Nz; Δ = 1.0, C_s = 0.0)
    m = RegularGrid(; size = (Nx, Ny, Nz), extent = (Nx * Δ, Ny * Δ, Nz * Δ), halo = (3, 3, 3))
    sim = BoussinesqSimulation(m; C_s = C_s,
                               bcs = FieldBoundaryConditions(:periodic, :periodic, :periodic))
    return m, sim
end
_yc(m) = [(j - 0.5) * m.dy for j in 1:m.Ny]
_zc(m) = [(k - 0.5) * m.dz for k in 1:m.Nz]

@testset "N-inflow-b analytic field: σ_u² = A²/4, u'w' = 0" begin
    m, sim = _parent(8, 64, 4)
    U, A, B = 5.0, 0.8, 0.3
    ky = 2π * 2 / m.Ly
    P = 120.0; ω = 2π / P; nper = 16
    src = ParentResolvedStatistics(sim; τ_avg = Inf, mode = :full)
    y = _yc(m)
    for n in 0:3nper-1
        t = n * P / nper
        for j in 1:m.Ny
            sim.u.data[:, m.Hy+j, :] .= U + A * sin(ky * y[j]) * cos(ω * t)
            sim.w.data[:, m.Hy+j, :] .= B * sin(ky * y[j]) * sin(ω * t)
        end
        update_inflow_statistics!(src, P / nper)
    end
    p = inflow_stress_parts(src)
    @test maximum(abs, p.resolved[1, :] ./ (A^2 / 4) .- 1) < 1e-10
    @test maximum(abs, p.resolved[5, :]) < 1e-12 * A * B          # u'w'
    @test maximum(abs, p.resolved[3, :] ./ (B^2 / 4) .- 1) < 1e-10
    @test maximum(abs, p.resolved[2, :]) == 0 && maximum(abs, p.resolved[4, :]) < 1e-14
    @test maximum(abs, p.U[1, :] .- U) < 1e-12
    @test all(==(0), p.sgs)                                       # ν_t = 0, e = 0
    # the face frame of a south face swaps the horizontal components
    srcS = ParentResolvedStatistics(sim; τ_avg = Inf, mode = :full, face = :south)
    update_inflow_statistics!(srcS, 1.0)
    pS = inflow_stress_parts(srcS); src1 = ParentResolvedStatistics(sim; τ_avg = Inf, mode = :full)
    update_inflow_statistics!(src1, 1.0); p1 = inflow_stress_parts(src1)
    @test pS.resolved[[2, 1, 3, 4, 6, 5], :] == p1.resolved
    # the length along y of a single sine is the integral of cos to its zero,
    # λ/(2π) (λ = 32Δ: trapezoid error 0.2 %)
    @test p1.L_s[1, 1] ≈ m.Ly / 2 / (2π) rtol = 0.005
end

@testset "N-inflow-b running mean: e-folding = τ_avg" begin
    m, sim = _parent(4, 8, 2)
    ky = 2π / m.Ly; y = _yc(m)
    τ = 600.0
    for adaptive in (false, true)
        rng = Random.Xoshiro(3)
        src = ParentResolvedStatistics(sim; τ_avg = τ, mode = :full)
        for j in 1:m.Ny
            sim.u.data[:, m.Hy+j, :] .= sin(ky * y[j])
        end
        t = 0.0
        while t < 40τ
            dt = adaptive ? 5 + 20rand(rng) : 10.0
            update_inflow_statistics!(src, dt); t += dt
        end
        R0 = inflow_stress_parts(src).resolved[1, 1]
        @test R0 ≈ 0.5 rtol = 1e-12
        sim.u.data .= 0
        ts = [0.0]; Rs = [R0]; t = 0.0
        while t < 3τ
            dt = adaptive ? 5 + 20rand(rng) : 10.0
            update_inflow_statistics!(src, dt); t += dt
            push!(ts, t); push!(Rs, inflow_stress_parts(src).resolved[1, 1])
        end
        n = findfirst(<(R0 / ℯ), Rs)
        # log-linear interpolation of the 1/e crossing
        a = (log(R0 / ℯ) - log(Rs[n-1])) / (log(Rs[n]) - log(Rs[n-1]))
        τm = ts[n-1] + a * (ts[n] - ts[n-1])
        @test abs(τm / τ - 1) < 0.02
        @test abs(τm / τ - 1) < 1e-9          # exact kernel weights: e-folding to round-off
    end
end

@testset "N-inflow-b SGS part: tr R^SGS = 2e, (2/3)e δ when S = 0" begin
    m, sim = _parent(8, 8, 8; Δ = 10.0)
    rng = Random.Xoshiro(11)
    for f in (sim.u, sim.v, sim.w)
        interior_view(f) .= randn(rng, size(interior_view(f)))
    end
    c = TKEClosure(m; e0 = 0.0)
    interior_view(c.e) .= 0.1 .+ rand(rng, m.Nx, m.Ny, m.Nz)
    Aeolus.BoundaryConditions.fill_halos!(sim.u, sim.bcs)
    Aeolus.BoundaryConditions.fill_halos!(sim.v, sim.bcs)
    Aeolus.BoundaryConditions.fill_halos!(sim.w, sim.bcs)
    sgs_coefficients!(sgs_fields(c, sim), c, sim)
    src = ParentResolvedStatistics(sim; closure = c, τ_avg = Inf)
    update_inflow_statistics!(src, 1.0)
    p = inflow_stress_parts(src)
    tr = p.sgs[1, :] .+ p.sgs[2, :] .+ p.sgs[3, :]
    @test maximum(abs, tr .- 2 .* p.e_sgs) < 1e-14 * maximum(p.e_sgs)
    @test maximum(abs, p.e_sgs .- vec(sum(interior_view(c.e); dims = (1, 2))) ./ 64) < 1e-14
    @test maximum(abs, p.sgs[4:6, :]) > 1e-3          # the strain contributes off-diagonals
    # S = 0: a uniform flow, any ν_t
    sim.u.data .= 3.0; sim.v.data .= -1.0; sim.w.data .= 0.0
    interior_view(sgs_fields(c, sim).νh) .= 7.0
    src0 = ParentResolvedStatistics(sim; closure = c, τ_avg = Inf)
    update_inflow_statistics!(src0, 1.0)
    p0 = inflow_stress_parts(src0)
    @test maximum(abs, p0.sgs[1:3, :] .- (2 / 3) .* p0.e_sgs') < 1e-15
    @test all(==(0), p0.sgs[4:6, :])
end

@testset "N-inflow-b scale gap: band, refine → 1, laminar parent" begin
    m, sim = _parent(8, 24, 4; Δ = 30.0)
    y = _yc(m)
    A, Bm = 0.6, 1.1
    # λ = 3Δ is inside the gap (c_gap Δ/r, c_gap Δ] = (2Δ, 4Δ] for r = 2; λ = 8Δ is not
    k3 = 2π / (3m.dy); k8 = 2π / (8m.dy)
    for j in 1:m.Ny
        sim.u.data[:, m.Hy+j, :] .= 2.0 + A * sin(k3 * y[j]) + Bm * sin(k8 * y[j])
    end
    src = ParentResolvedStatistics(sim; τ_avg = Inf, refine = 2, c_gap = 4)
    update_inflow_statistics!(src, 1.0)
    p = inflow_stress_parts(src)
    @test maximum(abs, p.band[1, :] ./ (A^2 / 2) .- 1) < 1e-12
    @test maximum(abs, p.resolved[1, :] ./ ((A^2 + Bm^2) / 2) .- 1) < 1e-12
    @test maximum(abs, p.generated .- p.band) < 1e-15      # no SGS: ν_t = 0
    # the band field's length: one sine of λ = 3Δ to its zero crossing, capped at c_len Δ = 2Δ
    @test all(p.L_s[1, :] .<= 2m.dy)
    # refine = 1: no gap, R^gen = R^SGS
    c = TKEClosure(m; e0 = 0.25)
    sgs_coefficients!(sgs_fields(c, sim), c, sim)
    src1 = ParentResolvedStatistics(sim; closure = c, τ_avg = Inf, refine = 1)
    update_inflow_statistics!(src1, 1.0)
    p1 = inflow_stress_parts(src1)
    @test maximum(abs, p1.generated .- p1.sgs) <= 1e-12
    @test maximum(abs, p1.band) == 0
    R = inflow_stresses(src1, p1.z, 0.0)
    @test maximum(maximum(abs, collect(R[k]) .- p1.sgs[:, k]) for k in eachindex(R)) <= 1e-12
    # a laminar parent with e = e₀ gives (2/3)e₀ δ_ij
    sim.u.data .= 4.0; sim.v.data .= 0.0; sim.w.data .= 0.0
    src2 = ParentResolvedStatistics(sim; closure = c, τ_avg = Inf)
    update_inflow_statistics!(src2, 1.0)
    for z in (15.0, 40.0, 100.0)
        R2 = inflow_stresses(src2, z, 0.0)
        @test maximum(abs, collect(R2) .- [2 / 3 * 0.25, 2 / 3 * 0.25, 2 / 3 * 0.25, 0, 0, 0]) < 1e-15
    end
    @test isempty(inflow_projection_log(src2))
end

@testset "N-inflow-b known synthetic parent: R within 3 %, L within 10 %" begin
    # The parent is filled by the N-inflow-a generator: each x-slice is one
    # generator step (Δt = 1 s, U = 1 m/s, so x is Taylor-frozen time and
    # L_x = U T), with the Panofsky & Dutton neutral tensor imposed by Lund.
    Nx, Ny, Nz = 240, 64, 40
    m, sim = _parent(Nx, Ny, Nz)
    us = 0.3
    R = [(2.39us)^2 0 -us^2; 0 (1.92us)^2 0; -us^2 0 (1.25us)^2]
    a = lund_coefficients(R)
    Ls, Lz, T = 4.0, 4.0, 6.0
    gen = DigitalFilterGenerator(Ny, Nz, 1.0, 1.0; L_s = Ls, L_z = Lz, T = T, seed = 5)
    src = ParentResolvedStatistics(sim; τ_avg = Inf, mode = :full, levels = 2:23, lag_z = 16)
    ũ = zeros(Ny, Nz); ṽ = similar(ũ); w̃ = similar(ũ)
    I, J, K = m.Hx .+ (1:Nx), m.Hy .+ (1:Ny), m.Hz .+ (1:Nz)
    for n in 1:30
        for i in 1:Nx
            unit_fluctuations!(ũ, ṽ, w̃, gen, 1.0)
            apply_lund!(ũ, ṽ, w̃, a)
            sim.u.data[I[i], J, K] .= 1.0 .+ ũ
            sim.v.data[I[i], J, K] .= ṽ
            sim.w.data[I[i], J, K] .= w̃
        end
        update_inflow_statistics!(src, 1.0)
    end
    p = inflow_stress_parts(src)
    # The tensor is measured at the cell centres, i.e. after the two-point
    # average of each staggered component along its own direction (u in x,
    # v in y, w in z). The exact attenuation of the generator's field is
    # (1 + ρ₁)/2 per averaged direction, ρ₁ the lag-1 correlation:
    # e^{-Δt/T} in x (the AR(1) blend), the filter's discrete R_1 in y, z.
    ρx = exp(-1 / T)
    Rs1 = Aeolus.Transition._df_autocorrelation(gen.bs[1, 1])[2]
    Rz1 = Aeolus.Transition._df_autocorrelation(gen.bz[1, 1])[2]
    fx, fs, fz = (1 + ρx) / 2, (1 + Rs1) / 2, (1 + Rz1) / 2
    target = [R[1, 1] * fx, R[2, 2] * fs, R[3, 3] * fz, 0, R[1, 3] * fx * fz, 0]
    mR = vec(sum(p.resolved; dims = 2)) ./ size(p.resolved, 2)
    for q in (1, 2, 3, 5)
        @test abs(mR[q] / target[q] - 1) < 0.03
    end
    @test abs(mR[4]) < 0.03 * sqrt(target[1] * target[2])
    @test abs(mR[6]) < 0.03 * sqrt(target[2] * target[3])
    # integral lengths (first zero crossing of the measured correlation)
    mean2(A) = vec(sum(A; dims = 2)) ./ size(A, 2)
    @test all(abs.(mean2(p.L_s) ./ Ls .- 1) .< 0.10)
    @test all(abs.(mean2(p.L_z) ./ Lz .- 1) .< 0.10)
    @test all(abs.(mean2(p.T) ./ T .- 1) .< 0.10)              # T = L_x/|U_n|, U_n = 1
    # mean normal velocity: sampling error σ_u/√N_eff ≈ 0.69/√(160 × 30) ≈ 0.01 per level
    @test maximum(abs, p.U[1, :] .- 1) < 0.04
end

@testset "N-inflow-b realizability: projection booked and logged" begin
    @test realizable_stress((1.0, 2.0, 0.5, 0.1, -0.2, 0.0)) == ((1.0, 2.0, 0.5, 0.1, -0.2, 0.0), 0.0)
    Rp, ch = realizable_stress([1.0 0 1.5; 0 1 0; 1.5 0 1])          # |u'w'| > σ_u σ_w
    @test ch > 0
    a = lund_coefficients(Rp)                                          # passes the N-inflow-a check
    @test maximum(abs, a * a' .- [Rp[1] Rp[4] Rp[5]; Rp[4] Rp[2] Rp[6]; Rp[5] Rp[6] Rp[3]]) < 1e-12
    # the nearest PSD tensor: eigenvalues 2.5, 1, -0.5 → 2.5, 1, ~0; change = 0.5
    @test ch ≈ 0.5 rtol = 1e-6
    # an SGS stress that is not realizable: strong laminar shear, tiny e
    m, sim = _parent(4, 4, 8; Δ = 10.0)
    zc = _zc(m)
    for k in 1:m.Nz
        sim.u.data[:, :, m.Hz+k] .= 0.05 * zc[k]
    end
    c = TKEClosure(m; e0 = 1e-4)
    interior_view(sgs_fields(c, sim).νh) .= 5.0          # ν_t S13 = 0.125 ≫ (2/3) e
    src = ParentResolvedStatistics(sim; closure = c, τ_avg = Inf, levels = 3:6)
    update_inflow_statistics!(src, 1.0)
    p = inflow_stress_parts(src)
    @test p.sgs[5, 2] ≈ -5.0 * 0.05 rtol = 1e-12              # -2 ν_t S13, S13 = ∂u/∂z / 2
    zs = p.z
    Rq = inflow_stresses(src, zs, 7.0)
    @test length(inflow_projection_log(src)) == length(zs)
    @test all(e -> e.t == 7.0 && e.change > 0.1, inflow_projection_log(src))
    a_s = inflow_lund_coefficients(src, zs, 7.0)              # every tensor now factorises
    @test length(a_s) == length(zs)
end
