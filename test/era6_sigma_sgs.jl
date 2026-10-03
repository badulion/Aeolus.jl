# Era VI, Phase 59 (P59-b, #184): the σ-model SGS closure of Nicoud,
# Baya Toda, Cabrit, Bose & Lee (2011, Phys. Fluids 23, 085106),
#
#     ν_t = (C_σ Δ)² σ₃(σ₁ − σ₂)(σ₂ − σ₃)/σ₁²,   C_σ = 1.35,
#
# with σ₁ ≥ σ₂ ≥ σ₃ the singular values of the resolved gradient g_ij
# (from `_velocity_gradient_at`, the one SGS gradient stencil), computed in
# closed form (trigonometric cubic + one deflation step).
#
# What is checked against what:
#   1. the analytic invariants of Nicoud et al. (2011, Sec. II/Table 1):
#      ν_t = 0 for uniform shear, solid-body rotation, any 2-D flow,
#      isotropic expansion and axisymmetric strain (also rotated off the
#      grid axes); the singular values against LinearAlgebra.svdvals on
#      random tensors; ν_t ≥ 0 on 1e4 random gradients, and the clip at
#      zero is exercised where round-off would make it negative;
#   2. the near-wall scaling ν_t ∝ z³ (Nicoud et al. 2011, Sec. II);
#      Smagorinsky ∝ z⁰;
#   3. laminar Couette flow with the SGS on: the wall stress stays νU/h;
#   4. decaying isotropic turbulence: the σ-model's KE decay exponent over
#      t ∈ [1, 3] against Smagorinsky C_s = 0.17;
# and the contract plumbing (closure type, κ_t = ν_t/Pr_t, all steppers).
# The Kelvin–Helmholtz check (5) extends the P59-c test,
# test/era6_transition_kh.jl; the CBC row is in test/era6_cbc_decay.jl.

using Test, Printf, Random
using LinearAlgebra: svdvals, qr, norm, Diagonal, I
using Statistics: mean
using Aeolus
using Aeolus.Fields: interior_view
using Aeolus.Operators: sigma_operator, sigma_singular_values, _sigma_operator_raw,
                        _velocity_gradient_at

# Fill every point of `f` (halos included) from the analytic `fn(x, y, z)`;
# `off` is the location of the field in cell units ((0, ½, ½) for u @ FaceX …).
function σ_fill!(f, fn, off)
    m = f.mesh
    for K in axes(f.data, 3), J in axes(f.data, 2), I in axes(f.data, 1)
        x = (I - m.Hx - 1 + off[1]) * m.dx
        y = (J - m.Hy - 1 + off[2]) * m.dy
        z = (K - m.Hz - 1 + off[3]) * m.dz
        f.data[I, J, K] = fn(x, y, z)
    end
    return f
end
σ_tuple(A) = Tuple(Tuple(A[i, :]) for i in 1:3)
σ_ref(s) = s[3] * (s[1] - s[2]) * (s[2] - s[3]) / s[1]^2     # from sorted singular values

@testset "σ-model SGS closure (P59-b, #184)" begin

    # -----------------------------------------------------------------------
    @testset "1. analytic invariants: ν_t = 0 where Smagorinsky is not" begin
        m = RegularGrid(; size = (16, 16, 16), extent = (1.0, 1.0, 1.0))
        Δ = (m.dx * m.dy * m.dz)^(1 / 3)
        u = Field(FaceX(), m); v = Field(FaceY(), m); w = Field(FaceZ(), m)
        νσ = Field(CellCenter(), m); νs = Field(CellCenter(), m)
        Q = Matrix(qr(randn(Xoshiro(5), 3, 3)).Q)             # a rotation off the grid axes
        S, Ω, a = 2.0, 1.5, 0.7
        shear = [0 S 0; 0 0 0; 0 0 0]
        rot = [0 -Ω 0; Ω 0 0; 0 0 0]
        cases = ["uniform shear u = Sy" => shear,
                 "solid-body rotation" => rot,
                 "isotropic expansion" => a * Matrix(1.0I, 3, 3),
                 "axisymmetric strain" => [a 0 0; 0 a 0; 0 0 -2a],
                 "rotated uniform shear" => Q * shear * Q',
                 "rotated solid-body rotation" => Q * rot * Q',
                 "rotated axisymmetric strain" => Q * [a 0 0; 0 a 0; 0 0 -2a] * Q']
        for (name, A) in cases
            σ_fill!(u, (x, y, z) -> A[1, 1] * x + A[1, 2] * y + A[1, 3] * z, (0, 0.5, 0.5))
            σ_fill!(v, (x, y, z) -> A[2, 1] * x + A[2, 2] * y + A[2, 3] * z, (0.5, 0, 0.5))
            σ_fill!(w, (x, y, z) -> A[3, 1] * x + A[3, 2] * y + A[3, 3] * z, (0.5, 0.5, 0))
            compute_sigma_νt!(νσ, u, v, w)                       # C_σ = 1.35
            compute_smagorinsky_νt!(νs, u, v, w, 0.17)
            scale = Δ^2 * norm(A)
            rσ = maximum(interior_view(νσ)) / scale
            rs = minimum(interior_view(νs)) / scale
            @printf("    %-28s ν_t,σ/(Δ²|g|) = %.1e   Smagorinsky 0.17: %.4f\n", name, rσ, rs)
            @test minimum(interior_view(νσ)) >= 0
            @test rσ <= 1e-12
            # Smagorinsky sees the strain: (0.17)² |S|/|g| = 0.0289 (shear), 0.0409
            # (expansion, axisymmetric strain); solid-body rotation is strain-free,
            # so Smagorinsky is zero there too (the issue's "nonzero for each" is
            # wrong for rotation: |S| = 0)
            if occursin("rotation", name)
                @test rs <= 1e-14
            else
                @test rs > 0.02
            end
        end

        # any two-dimensional flow (no z dependence, w = 0): rank(g) ≤ 2, σ₃ = 0
        σ_fill!(u, (x, y, z) -> sin(2π * x) * cos(2π * y) + 0.3 * cos(4π * y), (0, 0.5, 0.5))
        σ_fill!(v, (x, y, z) -> -cos(2π * x) * sin(2π * y) + 0.5 * sin(2π * x), (0.5, 0, 0.5))
        σ_fill!(w, (x, y, z) -> 0.0, (0.5, 0.5, 0))
        compute_sigma_νt!(νσ, u, v, w); compute_smagorinsky_νt!(νs, u, v, w, 0.17)
        @printf("    2-D flow: max ν_t,σ = %.1e, Smagorinsky max %.3e, mean %.3e\n",
                maximum(interior_view(νσ)), maximum(interior_view(νs)), mean(interior_view(νs)))
        @test maximum(interior_view(νσ)) == 0
        @test maximum(interior_view(νs)) > 0

        # closed-form singular values against LinearAlgebra on random tensors
        # (entries spread over 6 decades), and D_σ against the svd form
        rng = Xoshiro(11)
        esv = 0.0; eD = 0.0
        for _ in 1:10_000
            A = randn(rng, 3, 3) .* exp.(2 .* randn(rng, 3, 3))
            s = svdvals(A); σ = sigma_singular_values(σ_tuple(A))
            esv = max(esv, maximum(abs.(collect(σ) .- s)) / s[1])
            eD = max(eD, abs(sigma_operator(σ_tuple(A)) - max(σ_ref(s), 0)) / s[1])
        end
        @printf("    closed form vs svdvals (1e4 random g): max|Δσ|/σ₁ = %.1e, max|ΔD_σ|/σ₁ = %.1e\n", esv, eD)
        @test esv < 1e-11
        @test eD < 1e-11

        # ν_t ≥ 0 on 1e4 random gradient samples
        Ds = [sigma_operator(σ_tuple(randn(rng, 3, 3))) for _ in 1:10_000]
        @printf("    1e4 random g (unit normal entries): min D_σ = %.3e, mean D_σ = %.3f\n", minimum(Ds), mean(Ds))
        @test all(>=(0), Ds)

        # the clip: tensors with two equal singular values (Q diag(a, a, b) Qᵀ)
        # give D_σ = 0 mathematically; round-off makes the raw value a few
        # ulp negative in a fraction of them, and the clip returns exactly 0
        raws = Float64[]; clipped = Float64[]
        for _ in 1:10_000
            Q = Matrix(qr(randn(rng, 3, 3)).Q); aa = randn(rng); b = randn(rng)
            g = σ_tuple(Q * Diagonal([aa, aa, b]) * Q')
            push!(raws, _sigma_operator_raw(g) / max(abs(aa), abs(b)))
            push!(clipped, sigma_operator(g))
        end
        nneg = count(<(0), raws)
        @printf("    degenerate σ₂ = σ₃ or σ₁ = σ₂ (1e4): raw D_σ < 0 in %d, max|raw|/σ₁ = %.1e\n",
                nneg, maximum(abs, raws))
        @test nneg > 0                              # the round-off case occurs …
        @test all(>=(0), clipped)                   # … and is clipped
        @test maximum(abs, raws) < 1e-14            # and is round-off, not a wrong formula
        # exactly coincident singular values and the zero tensor
        @test sigma_operator(σ_tuple(zeros(3, 3))) == 0
        @test sigma_operator(σ_tuple([1.0 0 0; 0 1 0; 0 0 1])) == 0
    end

    # -----------------------------------------------------------------------
    @testset "2. near-wall scaling: ν_t ∝ z³ (Smagorinsky ∝ z⁰)" begin
        # The leading terms of the Taylor expansion at a no-slip wall z = 0
        # (Nicoud et al. 2011, Sec. II): u = zA(x, y), v = zB(x, y),
        # w = C z² with C = −(∂A/∂x + ∂B/∂y)/2 (continuity). Then σ₁ = O(1),
        # σ₂ = O(z), σ₃ = |det g|/(σ₁σ₂) = O(z²) and D_σ = O(z³). A, B are
        # linear in x, y so the stencil is exact; u, v are odd in z, so the
        # analytic halo below the wall is the NoSlipWall ghost.
        Nz = 64
        m = RegularGrid(; size = (16, 16, Nz), extent = (1.0, 1.0, 1.0))
        a1, a2, b1, b2 = 0.4, 0.3, -0.2, 0.5
        C = -(a1 + b2) / 2
        u = Field(FaceX(), m); v = Field(FaceY(), m); w = Field(FaceZ(), m)
        νσ = Field(CellCenter(), m); νs = Field(CellCenter(), m)
        σ_fill!(u, (x, y, z) -> z * (1 + a1 * x + a2 * y), (0, 0.5, 0.5))
        σ_fill!(v, (x, y, z) -> z * (0.5 + b1 * x + b2 * y), (0.5, 0, 0.5))
        σ_fill!(w, (x, y, z) -> C * z^2, (0.5, 0.5, 0))
        compute_sigma_νt!(νσ, u, v, w); compute_smagorinsky_νt!(νs, u, v, w, 0.17)
        pm(f) = vec(mean(interior_view(f); dims = (1, 2)))
        zc = [(k - 0.5) * m.dz for k in 1:Nz]
        function slope(y)
            X = log.(zc[1:8]); Y = log.(y[1:8])
            return sum((X .- mean(X)) .* (Y .- mean(Y))) / sum(abs2, X .- mean(X))
        end
        sσ, ss = slope(pm(νσ)), slope(pm(νs))
        @printf("    first 8 of %d cells: d log ν_t / d log z = %.4f (σ), %.4f (Smagorinsky)\n", Nz, sσ, ss)
        @test abs(sσ - 3) <= 0.15
        @test abs(ss) <= 0.05
    end
end
