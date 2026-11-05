# Era VI, Phase 56 (P56-c, #178): the a-priori SGS-stress harness.
#
# What is asserted:
#   1. the exact subfilter stress of a positive filter is positive
#      semi-definite (Vreman, Geurts & Kuerten 1994), and a filter with
#      negative lobes (sharp spectral) breaks it;
#   2. Galilean invariance of τ;
#   3. the closed form of τ₁₁ for u = A sin(kx) under the box and Gaussian
#      filters, in their transfer functions, and of the Bardina stress;
#   the history reductions of the DNS fixture (#400), and the harness
#   identities (the matched C_s matches the mean transfer; Smagorinsky
#   never backscatters).
# 1, 2 and the identities run on a cheap DNS state (32³ to t = 0.5); the
# 96³ fixture, its DNS checks and the a-priori numbers against the
# literature (Smagorinsky at Δ = 4h, the Bardina ranking, the dynamic C_s)
# are a validation run, in validation/era6_apriori_sgs.jl.

using Test
using Aeolus
using Aeolus.Fields: interior_view
using Printf
using LinearAlgebra: eigvals, Symmetric

# the DNS fixture (decaying isotropic turbulence)
isdefined(@__MODULE__, :dns_decay) || include(joinpath(@__DIR__, "fixtures", "dns_decay.jl"))
# apriori_case: the a-priori quantities of one filter width (shared by the
# test and the validation run)
isdefined(@__MODULE__, :apriori_case) || include(joinpath(@__DIR__, "support", "apriori_cases.jl"))

@testset "Era VI — a-priori SGS-stress harness (P56-c, #178)" begin

    # ---------------------------------------------------------------- 3
    @testset "analytic: τ₁₁ of A sin(kx) in the transfer function" begin
        N = 32; L = 2π
        g = RegularGrid(; size = (N, 4, 4), extent = (L, L / 8, L / 8), halo = (3, 3, 3))
        h = g.dx
        A = 1.7
        x = [(i - 0.5) * h for i in 1:N]
        for m in (1, 3, 5)
            k = m * 2π / L
            u = Field(CellCenter(), g); v = Field(CellCenter(), g); w = Field(CellCenter(), g)
            interior_view(u) .= A .* sin.(k .* x)
            for (name, filt, G) in (
                    ("box 3", (o, f) -> box_filter!(o, f, (3, 1, 1)), q -> sin(3q * h / 2) / (3sin(q * h / 2))),
                    ("box 5", (o, f) -> box_filter!(o, f, (5, 1, 1)), q -> sin(5q * h / 2) / (5sin(q * h / 2))),
                    ("Gaussian 4h", (o, f) -> gaussian_filter!(o, f, (4h, 4h, 4h)), q -> exp(-q^2 * (4h)^2 / 24)))
                τ = SymmetricTensorField(g)
                subfilter_stress!(τ, u, v, w, filt)
                # u² = A²/2 (1 − cos 2kx): τ₁₁ = A²/2 [(1 − G(k)²) − (G(2k) − G(k)²) cos 2kx]
                exact = (A^2 / 2) .* ((1 - G(k)^2) .- (G(2k) - G(k)^2) .* cos.(2k .* x))
                @test maximum(abs, τ.xx[:, 2, 3] .- exact) < 1e-12
                @test maximum(abs, τ.xx .- reshape(exact, :, 1, 1)) < 1e-12
                @test maximum(abs, τ.yy) == 0 && maximum(abs, τ.xy) == 0
            end
        end
    end

    @testset "analytic: Bardina stress of A sin(kx) is τ of the filtered field" begin
        # scale similarity: the Bardina stress is the subfilter stress of ū
        # = A G(k) sin(kx), so τ₁₁ = C_B (A G(k))²/2 [(1 − G(k)²) − (G(2k) − G(k)²) cos 2kx]
        N = 32; L = 2π
        g = RegularGrid(; size = (N, 4, 4), extent = (L, L / 8, L / 8), halo = (3, 3, 3))
        h = g.dx; A = 1.7; k = 3 * 2π / L
        x = [(i - 0.5) * h for i in 1:N]
        G = q -> exp(-q^2 * (4h)^2 / 24)
        filt = (o, f) -> gaussian_filter!(o, f, (4h, 4h, 4h))
        u = Field(CellCenter(), g); v = Field(CellCenter(), g); w = Field(CellCenter(), g)
        interior_view(u) .= A .* sin.(k .* x)
        ub = ntuple(_ -> Field(CellCenter(), g), 3)
        τe = SymmetricTensorField(g)
        subfilter_stress!(τe, u, v, w, filt; ubar = ub)
        Ab = A * G(k)
        exact = (Ab^2 / 2) .* ((1 - G(k)^2) .- (G(2k) - G(k)^2) .* cos.(2k .* x))
        for C_B in (1.0, 0.45)
            τb = SymmetricTensorField(g)
            bardina_stress!(τb, ub..., filt; C_B)
            @test maximum(abs, τb.xx .- C_B .* reshape(exact, :, 1, 1)) < 1e-12
            @test maximum(abs, τb.yy) == 0 && maximum(abs, τb.xy) == 0
        end
    end

    # ---------------------------------------------------------------- cheap DNS state
    # the fixture at 32³ to t = 0.5: the checks below hold on any state;
    # the 96³ run and its statistics are the validation run
    d = dns_decay(; N = 32, t_end = 0.5)

    # The history runs on the KA-b reductions (#400). On the final 96³
    # state: S is the moment ratio of the old serial formula (circshift and
    # `sum`, pairwise order) to rounding; ε = −ν⟨u_i ∇²u_i⟩ equals
    # ν⟨|D⁺u_i|²⟩ (summation by parts on the periodic grid: the discrete
    # Laplacian is −D⁻·D⁺) to rounding; the CFL max is the serial maximum
    # (max is exact); and all three are the same with
    # the serial backend and the threaded one (threshold 0).
    @testset "history reductions: the quantities they replace, thread-invariant (#400)" begin
        s = d.sim; n = length(interior_view(s.u))
        fwd(a, dir, h) = (circshift(a, ntuple(i -> i == dir ? -1 : 0, 3)) .- a) ./ h
        hs = (s.mesh.dx, s.mesh.dy, s.mesh.dz)
        num = 0.0; den = 0.0
        for (f, dir) in ((s.u, 1), (s.v, 2), (s.w, 3))
            gd = fwd(interior_view(f), dir, hs[dir])
            num += sum(gd .^ 3) / n; den += sum(gd .^ 2) / n
        end
        @test velocity_derivative_skewness(s) ≈ (num / 3) / (den / 3)^1.5 rtol = 1e-12
        lap = (Field(FaceX(), s.mesh), Field(FaceY(), s.mesh), Field(FaceZ(), s.mesh))
        ε = _dns_epsilon(s, lap)
        g2 = sum(sum(abs2, fwd(interior_view(f), dir, hs[dir])) for f in (s.u, s.v, s.w), dir in 1:3) / n
        @test ε ≈ s.ν * g2 rtol = 1e-10
        @test ε == d.ε[end]                                         # the recorded history value
        @test _dns_umax(s) === maximum(f -> maximum(abs, interior_view(f)), (s.u, s.v, s.w))
        ser, thr = CPUBackend(serial = true), CPUBackend(threshold = 0)
        for q in (velocity_derivative_skewness, x -> _dns_epsilon(x, lap), _dns_umax)
            @test with_backend(() -> q(s), ser) === with_backend(() -> q(s), thr)
        end
    end

    sim = d.sim; g = sim.mesh; h = g.dx
    res = Dict{Int,Any}(r => apriori_case(sim, r) for r in (4, 8))

    # ---------------------------------------------------------------- 1
    @testset "positive filters give a PSD τ (Vreman et al. 1994)" begin
        r4 = res[4]
        filters = (("box 3", (o, f) -> box_filter!(o, f, 3)), ("box 5", (o, f) -> box_filter!(o, f, 5)),
                   ("Gaussian 4h", res[4].filt), ("Gaussian 8h", res[8].filt))
        for (name, filt) in filters
            τ = SymmetricTensorField(g)
            subfilter_stress!(τ, sim.u, sim.v, sim.w, filt)
            λ1, _, _ = stress_eigvals(τ)
            tr = τ.xx .+ τ.yy .+ τ.zz
            @test all(λ1 .>= -1e-12 .* tr)
            @info @sprintf("    %-12s min λ₁/tr τ = %.2e", name, minimum(λ1 ./ tr))
        end
        # the exact stress is realizable: every point in the Lumley triangle
        @test res[4].smag.realizable == 1 && res[8].smag.realizable == 1
        @test 0 < res[4].smag.II < 2 / 3
        # discrimination: the sharp spectral filter's kernel (a sinc) has
        # negative lobes, and τ is not PSD everywhere (Vreman et al. 1994)
        τc = SymmetricTensorField(g)
        subfilter_stress!(τc, sim.u, sim.v, sim.w, (o, f) -> spectral_cutoff_filter!(o, f, π / (4h)))
        λ1, _, _ = stress_eigvals(τc)
        tr = τc.xx .+ τc.yy .+ τc.zz
        @test count(λ1 .< -1e-6 .* abs.(tr)) > 0
        # the eigenvalue routine against LinearAlgebra on a few points
        for I in (CartesianIndex(1, 1, 1), CartesianIndex(17, 30, 5), CartesianIndex(24, 2, 21), CartesianIndex(32, 16, 29))   # on the 32³ grid
            M = [r4.τe[i, j][I] for i in 1:3, j in 1:3]
            @test collect(getindex.(stress_eigvals(r4.τe), Ref(I))) ≈ eigvals(Symmetric(M)) rtol = 1e-10
        end
    end

    # ---------------------------------------------------------------- 2
    @testset "Galilean invariance of τ" begin
        U = (3.7, -2.1, 1.3)
        for (name, filt) in (("box 3", (o, f) -> box_filter!(o, f, 3)), ("Gaussian 4h", res[4].filt))
            τ0 = SymmetricTensorField(g); τ1 = SymmetricTensorField(g)
            subfilter_stress!(τ0, sim.u, sim.v, sim.w, filt)
            us = map((f, c) -> (s = _field_like(f); s.data .= f.data .+ c; s), (sim.u, sim.v, sim.w), U)
            subfilter_stress!(τ1, us..., filt)
            err = maximum(maximum(abs, a .- b) for (a, b) in zip(Aeolus.Operators.components(τ0), Aeolus.Operators.components(τ1)))
            @info @sprintf("    %-12s max |τ(u+U) − τ(u)| = %.1e (max τ₁₁ %.2e)", name, err, maximum(τ0.xx))
            @test err < 1e-12
        end
    end

    @testset "harness identities: matched C_s, no model backscatter" begin
        for r in (4, 8)
            m = res[r].smag
            # the matched C_s matches the mean transfer, and Smagorinsky never
            # backscatters (the a-priori numbers are the validation run)
            @test m.Π_ratio ≈ 1 atol = 1e-3
            @test m.backscatter_model == 0
            @test res[r].cs > 0
        end
    end
end
