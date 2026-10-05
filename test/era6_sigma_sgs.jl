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
import LinearAlgebra                     # qualified: no names leak into the runtests scope
const _LA = LinearAlgebra
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
        Q = Matrix(_LA.qr(randn(Xoshiro(5), 3, 3)).Q)             # a rotation off the grid axes
        S, Ω, a = 2.0, 1.5, 0.7
        shear = [0 S 0; 0 0 0; 0 0 0]
        rot = [0 -Ω 0; Ω 0 0; 0 0 0]
        cases = ["uniform shear u = Sy" => shear,
                 "solid-body rotation" => rot,
                 "isotropic expansion" => [a 0 0; 0 a 0; 0 0 a],
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
            scale = Δ^2 * _LA.norm(A)
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
            s = _LA.svdvals(A); σ = sigma_singular_values(σ_tuple(A))
            esv = max(esv, maximum(abs.(collect(σ) .- s)) / s[1])
            eD = max(eD, abs(sigma_operator(σ_tuple(A)) - max(σ_ref(s), 0)) / s[1])
        end
        @printf("    closed form vs svdvals (1e4 random g): max|Δσ|/σ₁ = %.1e, max|ΔD_σ|/σ₁ = %.1e\n", esv, eD)
        @test esv < 1e-11
        @test eD < 1e-11

        # near-degenerate pairs, both sides of the 1 % deflation switch:
        # g = Q diag(a, a(1 + δ), b) Qᵀ has the exactly known singular values
        # |a|, |a(1 + δ)|, |b| (g symmetric), so D_σ is known in closed form
        for δ in (1e-1, 1.1e-2, 0.9e-2, 1e-4, 1e-8)
            e = 0.0
            for _ in 1:2000
                Q = Matrix(_LA.qr(randn(rng, 3, 3)).Q); aa = 0.5 + rand(rng); b = 2.5 + rand(rng)
                d = sort([aa, aa * (1 + δ), b]; rev = true)
                g = σ_tuple(Q * _LA.Diagonal(d) * Q')
                e = max(e, abs(sigma_operator(g) - σ_ref(d)) / d[1])
            end
            @printf("    near-degenerate pair, relative splitting δ = %.1e: max|ΔD_σ|/σ₁ = %.1e\n", δ, e)
            @test e < 1e-13
        end

        # ν_t ≥ 0 on 1e4 random gradient samples
        Ds = [sigma_operator(σ_tuple(randn(rng, 3, 3))) for _ in 1:10_000]
        @printf("    1e4 random g (unit normal entries): min D_σ = %.3e, mean D_σ = %.3f\n", minimum(Ds), mean(Ds))
        @test all(>=(0), Ds)

        # the clip: tensors with two equal singular values (Q diag(a, a, b) Qᵀ)
        # give D_σ = 0 mathematically; round-off makes the raw value a few
        # ulp negative in a fraction of them, and the clip returns exactly 0
        raws = Float64[]; clipped = Float64[]
        for _ in 1:10_000
            Q = Matrix(_LA.qr(randn(rng, 3, 3)).Q); aa = randn(rng); b = randn(rng)
            g = σ_tuple(Q * _LA.Diagonal([aa, aa, b]) * Q')
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

    # -----------------------------------------------------------------------
    @testset "3. laminar Couette with the SGS on (Re = 100, Nz = 32)" begin
        # P59-a's Couette case (test/era6_walls.jl) run from rest with the
        # closure passed as `sgs`. The total wall stress is the molecular
        # one (wall_shear_stress) plus the SGS flux through the wall face,
        # ν_t,face ∂u/∂z with ν_t averaged to the face as the eddy-viscosity
        # operator does. In u = Uz/h the σ-model has ν_t ≡ 0 (uniform shear),
        # so the wall stress is νU/h; Smagorinsky's ν_t = (C_sΔ)² U/h raises
        # it by the factor 1 + (C_sΔ)²(U/h)/ν.
        U, h, Re = 1.0, 1.0, 100
        ν = U * h / Re
        m = RegularGrid(; size = (4, 4, 32), extent = (1.0, 1.0, h))
        Δ = (m.dx * m.dy * m.dz)^(1 / 3)
        zc = [(k - 0.5) * m.dz for k in 1:m.Nz]
        function sgs_wall_stress(sim)
            fill_halos!(sim.νt, sim.bcs); fill_halos!(sim.u, sim.bcs)
            H = m.Hz; s = 0.0
            for j in m.Hy+1:m.Hy+m.Ny, i in m.Hx+1:m.Hx+m.Nx
                νf = 0.25 * (sim.νt.data[i-1, j, H] + sim.νt.data[i, j, H] +
                             sim.νt.data[i-1, j, H+1] + sim.νt.data[i, j, H+1])
                s += νf * (sim.u.data[i, j, H+1] - sim.u.data[i, j, H]) / m.dz
            end
            return s / (m.Nx * m.Ny)
        end
        res = Dict{String,Any}()
        for (name, sgs, C_s) in (("σ-model", SigmaModel(), 0.0),
                                 ("Smagorinsky 0.17", StaticSmagorinsky(0.17), 0.17))
            bcs = FieldBoundaryConditions(:periodic, :periodic; bottom = :noslip,
                                          top = NoSlipWall(U, 0))
            sim = BoussinesqSimulation(m; ν = ν, bcs = bcs)
            νe = ν + (C_s * Δ)^2 * U / h
            dt = 0.3 * m.dz^2 / νe
            for _ in 1:ceil(Int, 4h^2 / νe / dt)          # slowest mode e^{-π²νt/h²} ≈ 1e-17
                step!(sim, dt; sgs = sgs)
            end
            err = maximum(abs.(vec(mean(interior_view(sim.u); dims = (1, 2))) .- U .* zc ./ h))
            τ = wall_shear_stress(sim) + sgs_wall_stress(sim)
            res[name] = (; err, τ, νtmax = maximum(interior_view(sim.νt)))
            @printf("    %-17s ‖u − Uz/h‖∞ = %.1e U, τ_wall/(νU/h) − 1 = %.4e, max ν_t = %.2e\n",
                    name, err, τ / (ν * U / h) - 1, res[name].νtmax)
        end
        rσ, rs = res["σ-model"], res["Smagorinsky 0.17"]
        @test rσ.νtmax == 0
        @test rσ.err < 1e-10 * U
        @test abs(rσ.τ - ν * U / h) <= 1e-10 * (ν * U / h)
        # Smagorinsky: the same linear profile, with the predicted excess
        excess = (0.17 * Δ)^2 * (U / h) / ν
        @printf("    Smagorinsky excess (C_sΔ)²(U/h)/ν = %.4e (measured %.4e)\n", excess, rs.τ / (ν * U / h) - 1)
        @test rs.err < 1e-10 * U
        @test rs.τ / (ν * U / h) - 1 ≈ excess rtol = 1e-8
    end

    # -----------------------------------------------------------------------
    @testset "4. decaying isotropic turbulence: KE decay vs Smagorinsky 0.17" begin
        # 32³ in a 2π box, ν = 1e-4: a k^{-5/3} solenoidal field up to N/4
        # with u_rms = 1 (the era2 dynamic-Smagorinsky set-up), SSP-RK3 at
        # dt = 0.01 to t = 3. The decay exponent n of K ∝ t^{-n} over
        # t ∈ [1, 3] is n = −ln(K(3)/K(1))/ln 3. C_σ = 1.35 is the value
        # Nicoud et al. (2011) obtained in isotropic turbulence, so the
        # σ-model should dissipate like Smagorinsky there; target ±15 %.
        mesh = RegularGrid(; size = (32, 32, 32), extent = (2π, 2π, 2π), halo = (3, 3, 3))
        ke(sim) = 0.5 * sum(f -> mean(abs2, interior_view(f)), (sim.u, sim.v, sim.w))
        function hit(sgs, seed)
            sim = BoussinesqSimulation(mesh; ν = 1e-4)
            k0 = 2π / mesh.Lx
            isotropic_random_field!(sim, k -> (k / k0)^(-5 / 3); rng = Xoshiro(seed), kmax = 8k0)
            sc = 1 / sqrt(mean(abs2, interior_view(sim.u)))
            for f in (sim.u, sim.v, sim.w)
                interior_view(f) .*= sc; fill_halos!(f, sim.bcs)
            end
            K = Float64[]
            for n in 1:300
                step_rk3!(sim, 0.01; sgs = sgs)
                n % 100 == 0 && push!(K, ke(sim))
            end
            return (; K1 = K[1], K3 = K[3], n = -log(K[3] / K[1]) / log(3),
                      νt = mean(interior_view(sim.νt)))
        end
        for seed in (1, 2)
            s = hit(StaticSmagorinsky(0.17), seed)
            σ = hit(SigmaModel(1.35), seed)
            z = hit(nothing, seed)
            @printf("    seed %d: n = %.4f (Smagorinsky 0.17), %.4f (σ 1.35; ratio %.4f), %.4f (no model); K(3)/K(1) = %.4f / %.4f / %.4f; ⟨ν_t⟩(t=3) %.2e / %.2e\n",
                    seed, s.n, σ.n, σ.n / s.n, z.n, s.K3 / s.K1, σ.K3 / σ.K1, z.K3 / z.K1, s.νt, σ.νt)
            @test abs(σ.n / s.n - 1) <= 0.15
            @test σ.n > 5 * z.n                 # the closure, not ν, does the decay
        end
    end

    # -----------------------------------------------------------------------
    @testset "contract: κ_t = ν_t/Pr_t, every stepper, Pr_t keyword" begin
        m = RegularGrid(; size = (16, 16, 16), extent = (1.0, 1.0, 1.0), halo = (3, 3, 3))
        sim = BoussinesqSimulation(m; ν = 1e-4, Pr_t = 0.7)
        isotropic_random_field!(sim, k -> k^(-5 / 3); rng = Xoshiro(3), kmax = 8 * 2π)
        c = SigmaModel()
        @test c isa AbstractEddyViscosityClosure && c.C_σ == 1.35
        f = compute_sgs!(c, sim)
        ref = Field(CellCenter(), m); compute_sigma_νt!(ref, sim.u, sim.v, sim.w, 1.35)
        @test interior_view(f.νh) == interior_view(ref)
        @test f.κh.data == f.νh.data ./ 0.7
        @test minimum(interior_view(f.νh)) >= 0 && maximum(interior_view(f.νh)) > 0
        f2 = compute_sgs!(SigmaModel(2.7; Pr_t = 1 / 3), sim)
        @test interior_view(f2.νh) ≈ 4 .* interior_view(ref) rtol = 1e-14   # ∝ C_σ²
        @test f2.κh.data == f2.νh.data ./ (1 / 3)
        for stepper in (s -> step!(s, 1e-3; sgs = c), s -> step_rk3!(s, 1e-3; sgs = c),
                        s -> step_imex!(s, 1e-3, ARK2(); sgs = c))
            s2 = BoussinesqSimulation(m; ν = 1e-4)
            isotropic_random_field!(s2, k -> k^(-5 / 3); rng = Xoshiro(3), kmax = 8 * 2π)
            stepper(s2)
            @test all(isfinite, interior_view(s2.u)) && maximum(interior_view(s2.νt)) > 0
        end
    end
end
