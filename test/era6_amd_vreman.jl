# Era VI, P55-d (#176): the anisotropic minimum-dissipation (AMD) and
# Vreman SGS closures (src/operators/sgs_amd_vreman.jl).
#
#   * exact closed forms on linear fields on the Berlin grid spacing
#     (15.625, 15.625, 6.25) m;
#   * laminar switch-off (uniform flow, pure shear; AMD: solid-body
#     rotation), where Smagorinsky is not zero;
#   * AMD's buoyancy term: ν_t falls linearly with N² and vanishes at the
#     analytic N² where buoyancy cancels the production;
#   * Galilean invariance;
#   * dissipation in developed turbulence relative to Smagorinsky 0.17;
#   * the closures on the contract: every core, own κ_t, conservation.
# The Comte-Bellot–Corrsin rows are in test/era6_cbc_decay.jl.

using Test
using Aeolus
using Aeolus.Fields: interior_view, Field
using Aeolus.Grids: interior
using Random: Xoshiro
using Statistics: mean
using Printf

const OPS = Aeolus.Operators

# Linear fields u_a = c_a + A[a, b] x_b, θ = θc + γ·x on every stored value
# (halos included) of the staggered fields of `sim`.
function set_linear!(sim, A; c = (0.7, -0.3, 0.1), γ = (0.0, 0.0, 0.0), θc = 300.0)
    g = sim.mesh
    Hx, Hy, Hz = g.Hx, g.Hy, g.Hz
    xf(i) = (i - Hx - 1) * g.dx; xc(i) = (i - Hx - 0.5) * g.dx
    yf(j) = (j - Hy - 1) * g.dy; yc(j) = (j - Hy - 0.5) * g.dy
    zf(k) = (k - Hz - 1) * g.dz; zc(k) = (k - Hz - 0.5) * g.dz
    lin(a, x, y, z) = c[a] + A[a, 1] * x + A[a, 2] * y + A[a, 3] * z
    for I in CartesianIndices(sim.u.data)
        i, j, k = Tuple(I)
        sim.u.data[I] = lin(1, xf(i), yc(j), zc(k))
        sim.v.data[I] = lin(2, xc(i), yf(j), zc(k))
        sim.w.data[I] = lin(3, xc(i), yc(j), zf(k))
        sim.θ.data[I] = θc + γ[1] * xc(i) + γ[2] * yc(j) + γ[3] * zc(k)
    end
    return sim
end

# Closed forms of the two models for a constant gradient tensor A
# (A[a, b] = ∂u_a/∂x_b), θ gradient γ, spacings δ.
function amd_exact(A, γ, δ; C = 0.3, βg = 0.0, Pr_t = 1/3)
    S = (A + A') / 2
    P = sum(δ[k]^2 * A[i, k] * A[j, k] * S[i, j] for i in 1:3, j in 1:3, k in 1:3)
    B = βg * sum(δ[k]^2 * A[3, k] * γ[k] for k in 1:3)
    ν = C * max(0, -P + B) / sum(abs2, A)
    Q = sum(δ[k]^2 * A[i, k] * γ[k] * γ[i] for i in 1:3, k in 1:3)
    κ = sum(abs2, γ) > 0 ? C * max(0, -Q) / sum(abs2, γ) : ν / Pr_t
    return ν, κ
end
function vreman_exact(A, δ; C_s = 0.17)
    α = permutedims(A)                                   # α_ij = ∂u_j/∂x_i
    β = [sum(δ[m]^2 * α[m, i] * α[m, j] for m in 1:3) for i in 1:3, j in 1:3]
    Bβ = β[1,1]*β[2,2] - β[1,2]^2 + β[1,1]*β[3,3] - β[1,3]^2 + β[2,2]*β[3,3] - β[2,3]^2
    return 2.5 * C_s^2 * sqrt(Bβ / sum(abs2, α))
end

function closures!(sim; βg = 0.0)
    νa = Field(CellCenter(), sim.mesh); κa = Field(CellCenter(), sim.mesh)
    νv = Field(CellCenter(), sim.mesh); νs = Field(CellCenter(), sim.mesh)
    compute_amd_coefficients!(νa, κa, sim.u, sim.v, sim.w, sim.θ; buoyancy = βg)
    compute_vreman_νt!(νv, sim.u, sim.v, sim.w)
    compute_smagorinsky_νt!(νs, sim.u, sim.v, sim.w, 0.17)
    return interior_view(νa), interior_view(κa), interior_view(νv), interior_view(νs)
end

relerr(x, y) = abs(x - y) / max(abs(y), floatmin())

@testset "Era VI — AMD and Vreman SGS closures (P55-d, #176)" begin

    δ = (15.625, 15.625, 6.25)
    grid = RegularGrid(; size = (6, 6, 6), extent = (6δ[1], 6δ[2], 6δ[3]), halo = (2, 2, 2))
    sim = BoussinesqSimulation(grid; θ₀ = 300.0)

    @testset "exact on linear fields, anisotropic grid $(δ) m" begin
        A = [ 0.013 -0.021  0.034;
              0.008  0.017 -0.052;
             -0.027  0.041 -0.030]                 # s⁻¹, trace-free
        γ = (0.002, -0.001, 0.004)                 # K m⁻¹
        βg = 9.81 / 300
        set_linear!(sim, A; γ)
        νa, κa, νv, _ = closures!(sim; βg)
        ν_ex, κ_ex = amd_exact(A, γ, δ; βg)
        νv_ex = vreman_exact(A, δ)
        e = (maximum(relerr.(νa, ν_ex)), maximum(relerr.(κa, κ_ex)), maximum(relerr.(νv, νv_ex)))
        @printf("  [exact] AMD ν_t %.6e (rel err %.1e), κ_t %.6e (%.1e); Vreman ν_t %.6e (%.1e) m²/s\n",
                ν_ex, e[1], κ_ex, e[2], νv_ex, e[3])
        @test ν_ex > 0 && κ_ex > 0 && νv_ex > 0    # the tensor is chosen so none is clipped
        @test all(<=(1e-12), e)
        # the same tensor with the scalar sign flipped: AMD's κ_t is clipped
        # at zero (no counter-gradient diffusion), not negative
        set_linear!(sim, A; γ = .-γ .* (1, 1, -1))
        _, κa2, _, _ = closures!(sim; βg)
        _, κ2_ex = amd_exact(A, .-γ .* (1, 1, -1), δ; βg)
        @test all(x -> relerr(x, κ2_ex) <= 1e-12 || (κ2_ex == 0 && x == 0), κa2)
    end

    @testset "laminar switch-off" begin
        Z = zeros(3, 3)
        set_linear!(sim, Z; c = (3.0, -2.0, 0.5))                        # uniform flow
        νa, κa, νv, νs = closures!(sim)
        @test all(==(0), νa) && all(==(0), νv) && all(==(0), νs) && all(==(0), κa)

        S = 0.02                                                           # u = S z
        set_linear!(sim, [0 0 S; 0 0 0; 0 0 0])
        νa, _, νv, νs = closures!(sim)
        @printf("  [laminar] u = Sz, S = %.2f s⁻¹: AMD %.1e, Vreman %.1e, Smagorinsky(0.17) %.4e m²/s\n",
                S, maximum(νa), maximum(νv), minimum(νs))
        @test all(==(0), νa) && all(==(0), νv)
        Δ = prod(δ)^(1/3)
        @test all(x -> relerr(x, (0.17Δ)^2 * S) < 1e-12, νs)               # (C_s Δ)² S ≠ 0

        # a two-component shear u(z), v(z): β is rank one, so B_β = 0; the
        # Cauchy–Binet form makes it exactly 0 (the minor form β₁₁β₂₂ − β₁₂²
        # leaves round-off, which √ turns into ν_t ~ 1e-10 m²/s)
        set_linear!(sim, [0 0 0.02; 0 0 -0.013; 0 0 0])
        νa, _, νv, _ = closures!(sim)
        @test all(==(0), νa) && all(==(0), νv)

        Ω = 0.01                                                           # solid-body rotation
        set_linear!(sim, [0 -Ω 0; Ω 0 0; 0 0 0])
        νa, _, νv, νs = closures!(sim)
        @printf("  [laminar] solid-body rotation Ω = %.2f s⁻¹: AMD %.1e, Vreman %.3e, Smagorinsky %.1e m²/s\n",
                Ω, maximum(νa), minimum(νv), maximum(νs))
        @test all(==(0), νa)
        # Vreman is not zero for rotation (Vreman 2004: B_β = Δx²Δy²Ω⁴)
        @test all(x -> relerr(x, 2.5 * 0.17^2 * sqrt(δ[1]^2 * δ[2]^2 * Ω^4 / (2Ω^2))) < 1e-12, νv)
    end

    @testset "AMD in stable stratification: buoyancy cancels production" begin
        # u = a x + S z, v = a y, w = −2a z: positive AMD production
        # P = a(δz² S² − (δx² + δy² − 8δz²) a²) with a compressive ∂w/∂z,
        # so the buoyancy term (g/θ₀) δz² (∂w/∂z)(∂θ/∂z) = −2a δz² N² < 0.
        a, S = 0.01, 0.05
        A = [a 0 S; 0 a 0; 0 0 -2a]
        P = a * (δ[3]^2 * S^2 - (δ[1]^2 + δ[2]^2 - 8δ[3]^2) * a^2)
        N2c = P / (2a * δ[3]^2)
        θ₀ = 300.0; g = 9.81
        νs = Float64[]; κs = Float64[]
        N2s = N2c .* (0.0, 0.25, 0.5, 0.75, 1.0, 1.5)
        for N2 in N2s
            set_linear!(sim, A; γ = (0.0, 0.0, N2 * θ₀ / g), θc = θ₀)
            νa, κa, _, _ = closures!(sim; βg = g / θ₀)
            push!(νs, mean(νa)); push!(κs, mean(κa))
            ν_ex, κ_ex = amd_exact(A, (0.0, 0.0, N2 * θ₀ / g), δ; βg = g / θ₀)
            @test maximum(abs, νa .- ν_ex) <= 1e-12 * max(νs[1], 1e-300)
            N2 > 0 && @test maximum(relerr.(κa, κ_ex)) <= 1e-12
        end
        @printf("  [stable] P = %.4e m²/s³, analytic N²* = %.4e s⁻²; ν_t(N²/N²* = 0, ¼, ½, ¾, 1, 1.5) = %s m²/s\n",
                P, N2c, join((@sprintf("%.4e", x) for x in νs), ", "))
        @test P > 0
        @test all(diff(collect(νs[1:5])) .< 0)                          # monotone decrease
        @test νs[2] ≈ 0.75νs[1] rtol = 1e-10                               # linear in N²
        @test νs[5] <= 1e-12 * νs[1]                                       # zero at N²*
        @test νs[6] == 0                                                   # clipped beyond
        # AMD's own κ_t for θ = Γz: C δz² (2a)
        # (independent of N²: Q/|∇θ|² = δz² ∂w/∂z)
        @test all(x -> isapprox(x, 0.3 * δ[3]^2 * 2a; rtol = 1e-12), κs[2:end])
    end

    @testset "Galilean invariance" begin
        rng = Xoshiro(176)
        m = RegularGrid(; size = (12, 12, 12), extent = (12δ[1], 12δ[2], 12δ[3]), halo = (2, 2, 2))
        s = BoussinesqSimulation(m; θ₀ = 300.0)
        for f in (s.u, s.v, s.w)
            interior_view(f) .= randn(rng, size(interior_view(f)))
        end
        interior_view(s.θ) .= 300 .+ randn(rng, size(interior_view(s.θ)))
        fill = () -> foreach(f -> fill_halos!(f, s.bcs), (s.u, s.v, s.w, s.θ))
        fill()
        νa0, κa0, νv0, _ = copy.(closures!(s; βg = 9.81 / 300))
        interior_view(s.u) .+= 7.0; interior_view(s.v) .-= 4.0; interior_view(s.w) .+= 1.5
        fill()
        νa1, κa1, νv1, _ = closures!(s; βg = 9.81 / 300)
        dν = (maximum(abs, νa1 .- νa0) / maximum(νa0), maximum(abs, κa1 .- κa0) / maximum(κa0),
              maximum(abs, νv1 .- νv0) / maximum(νv0))
        @printf("  [Galilean] shift (7, −4, 1.5) m/s: max|Δν_t|/max ν_t AMD %.1e, AMD κ_t %.1e, Vreman %.1e\n", dν...)
        @test all(<=(1e-14), dν)
    end

    @testset "developed turbulence: dissipation relative to Smagorinsky 0.17" begin
        # the decaying-turbulence set-up of test/era2_sgs_numerics.jl: a
        # k^(-5/3) random field evolved for t = 1 (≈ one turnover) with
        # static Smagorinsky 0.17
        mesh = RegularGrid(; size = (32, 32, 32), extent = (2π, 2π, 2π), halo = (3, 3, 3))
        s = BoussinesqSimulation(mesh; C_s = 0.17, ν = 1e-4)
        k0 = 1.0
        isotropic_random_field!(s, k -> (k / k0)^(-5/3); rng = Xoshiro(1), kmax = 8k0)
        sc = 1 / sqrt(mean(abs2, interior_view(s.u)))
        foreach(f -> (interior_view(f) .*= sc; fill_halos!(f, s.bcs)), (s.u, s.v, s.w))
        for _ in 1:100
            step!(s, 0.01)
        end
        foreach(f -> fill_halos!(f, s.bcs), (s.u, s.v, s.w, s.θ))
        νa, _, νv, νs = closures!(s)
        S2 = Field(CellCenter(), mesh)
        compute_smagorinsky_νt!(S2, s.u, s.v, s.w, 1.0)                 # Δ²|S|
        Δ2 = (mesh.dx * mesh.dy * mesh.dz)^(2/3)
        magS = interior_view(S2) ./ Δ2
        ε(ν) = mean(ν .* magS .^ 2)                                        # ⟨2 ν_t S_ij S_ij⟩
        rA, rV = ε(νa) / ε(νs), ε(νv) / ε(νs)
        @printf("  [turbulence 32³, t = 1] ⟨2ν_t S S⟩ / Smagorinsky(0.17): AMD %.3f, Vreman %.3f; mean ν_t AMD %.2e Vreman %.2e Smag %.2e; AMD zero in %.1f %% of cells\n",
                rA, rV, mean(νa), mean(νv), mean(νs), 100 * count(==(0), νa) / length(νa))
        @test 0.6 <= rA <= 1.4
        @test 0.6 <= rV <= 1.4
    end

    @testset "on the closure contract: every core, own κ_t, conservation" begin
        rng = Xoshiro(55)
        m = RegularGrid(; size = (16, 16, 16), extent = (16δ[1], 16δ[2], 16δ[3]), halo = (3, 3, 3))
        U = [0.5 .* randn(rng, 16, 16, 16) for _ in 1:3]
        T = 300 .+ 0.5 .* randn(rng, 16, 16, 16) .+ [0.003 * (k - 0.5) * δ[3] for i in 1:16, j in 1:16, k in 1:16]
        function fresh(; kw...)
            s = BoussinesqSimulation(m; θ₀ = 300.0, kw...)
            for (f, a) in zip((s.u, s.v, s.w), U)
                interior_view(f) .= a
            end
            interior_view(s.θ) .= T
            foreach(f -> fill_halos!(f, s.bcs), (s.u, s.v, s.w))
            fill_halos!(s.θ, s.θ_bcs)
            return s
        end
        # compute_sgs! = the kernels with the core's g/θ₀ and Pr_t
        s = fresh()
        f = compute_sgs!(AMDClosure(), s)
        νr = Field(CellCenter(), m); κr = Field(CellCenter(), m)
        compute_amd_coefficients!(νr, κr, s.u, s.v, s.w, s.θ; buoyancy = 9.81 / 300, Pr_t = s.Pr_t)
        @test interior_view(f.νh) == interior_view(νr) && interior_view(f.κh) == interior_view(κr)
        @test f.νh === s.νt && f.κh === s.κt
        @test sgs_buoyancy_parameter(s) == 9.81 / 300
        @test sgs_buoyancy_parameter(BoussinesqSimulation(m)) == 0           # θ₀ = 0: buoyancy off
        # AMD's κ_t is its own, not ν_t/Pr_t: they differ in most cells
        # (they agree only where both are clipped to 0)
        nκ = count(interior_view(s.κt) .!= interior_view(s.νt) ./ s.Pr_t)
        @test nκ > 0.5 * 16^3
        # the buoyancy term on a stably stratified profile θ = θ_s + Γz (no
        # noise): (g/θ₀) δz² (∂w/∂z) Γ. It LOWERS ν_t where ∂w/∂z < 0 and
        # RAISES it where ∂w/∂z > 0 (the gradient-model estimate of the
        # subfilter buoyancy flux, Abkar & Moin 2017), so a random velocity
        # field sees no net reduction; the monotone decrease with N² holds
        # for compressive ∂w/∂z (testset above).
        interior_view(s.θ) .= [300 + 0.003 * (k - 0.5) * δ[3] for i in 1:16, j in 1:16, k in 1:16]
        fill_halos!(s.θ, s.θ_bcs)
        ν0 = Field(CellCenter(), m)
        compute_amd_coefficients!(νr, κr, s.u, s.v, s.w, s.θ; buoyancy = 9.81 / 300)
        compute_amd_coefficients!(ν0, κr, s.u, s.v, s.w, s.θ; buoyancy = 0.0)
        H = (m.Hx, m.Hy, m.Hz)
        dwdz = [(s.w.data[i+H[1], j+H[2], k+H[3]+1] - s.w.data[i+H[1], j+H[2], k+H[3]]) / δ[3]
                for i in 1:16, j in 1:16, k in 1:16]
        # levels 2:15 (the periodic halos wrap the linear profile at 1 and 16)
        νb, νn = interior_view(νr)[:, :, 2:15], interior_view(ν0)[:, :, 2:15]
        dwdz = dwdz[:, :, 2:15]
        @test all(νb[dwdz .< 0] .<= νn[dwdz .< 0]) && all(νb[dwdz .> 0] .>= νn[dwdz .> 0])
        @test count(νb .< νn) > 0 && count(νb .> νn) > 0
        @printf("  [contract] κ_t ≠ ν_t/Pr_t in %d / %d cells; Γ = 3 K/km: mean ν_t %.4f (buoyancy on) vs %.4f (off) m²/s, lowered in %d, raised in %d cells\n",
                nκ, 16^3, mean(νb), mean(νn), count(νb .< νn), count(νb .> νn))
        fv = compute_sgs!(VremanClosure(), s)
        @test interior_view(fv.κh) == interior_view(fv.νh) ./ s.Pr_t

        # every stepper takes them; periodic box: Σθ and Σu conserved
        steppers = (("step!", (x, h; kw...) -> step!(x, h; kw...)),
                    ("step_rk3!", (x, h; kw...) -> step_rk3!(x, h; kw...)),
                    ("step_imex!", (x, h; kw...) -> step_imex!(x, h, ARK2(); kw...)))
        for (name, adv!) in steppers, c in (AMDClosure(), VremanClosure())
            x = fresh()
            S0 = sum(interior_view(x.θ)); M0 = sum(interior_view(x.u))
            V0 = sum(abs2, interior_view(x.θ) .- mean(interior_view(x.θ)))
            for _ in 1:10
                adv!(x, 0.5; sgs = c)
            end
            dS = abs(sum(interior_view(x.θ)) - S0) / S0
            dM = abs(sum(interior_view(x.u)) - M0) / sum(abs, U[1])
            @test all(isfinite, interior_view(x.u)) && all(isfinite, interior_view(x.θ))
            @test minimum(interior_view(x.νt)) >= 0 && minimum(interior_view(x.κt)) >= 0
            # IMEX: the implicit diffusion solve is iterative (tol 1e-10),
            # so the sums are conserved to that tolerance, not to round-off
            tolc = name == "step_imex!" ? 1e-10 : 1e-13
            @test dS <= tolc && dM <= 10tolc
            @printf("  [contract] %-10s %-13s 10 steps: |ΔΣθ|/Σθ %.1e, |ΔΣu| %.1e, max ν_t %.3f, max κ_t %.3f m²/s\n",
                    name, nameof(typeof(c)), dS, dM, maximum(interior_view(x.νt)), maximum(interior_view(x.κt)))
        end
        x = fresh()
        k = CFLConstraints(dt_max = 1.0)
        for _ in 1:3
            step_adaptive!(x, k; sgs = AMDClosure())
        end
        @test all(isfinite, interior_view(x.u))

        # the sound-proof cores: g/θ̄(z) of the reference state
        an = AnelasticSimulation(m)
        βk = sgs_buoyancy_parameter(an)
        @test βk(1) == an.g_accel / an.ref.θ̄[1] && βk(16) == an.g_accel / an.ref.θ̄[16]
        @test sgs_buoyancy_parameter(AnelasticSimulation(m; buoyancy = false)) == 0
        for (fl, a) in zip((an.u, an.v, an.w), U)
            interior_view(fl) .= a
        end
        interior_view(an.w)[:, :, 1] .= 0
        for _ in 1:3
            step!(an, 0.5; sgs = AMDClosure())
        end
        @test all(isfinite, interior_view(an.u)) && minimum(interior_view(an.νt)) >= 0
        @test maximum(interior_view(an.νt)) > 0

        # ---- qv gets its own AMD diffusivity, not θ's (PR #215 review) ----
        # The AMD scalar model bounds the subfilter variance of the scalar it
        # is evaluated for (Abkar, Bae & Moin 2016), so κ_t is per scalar.
        x = fresh(moisture = true)
        Q = 0.008 .+ 1e-4 .* randn(rng, 16, 16, 16)              # uncorrelated with θ
        interior_view(x.qv) .= Q; fill_halos!(x.qv, x.θ_bcs)
        f = compute_sgs!(AMDClosure(), x)
        κq = sgs_tracer_diffusivity!(Field(CellCenter(), m), AMDClosure(), f, x, x.qv)
        κref = Field(CellCenter(), m)
        compute_amd_scalar_diffusivity!(κref, x.νt, x.u, x.v, x.w, x.qv; Pr_t = x.Pr_t)
        @test interior_view(κq) == interior_view(κref)
        # the θ kernel and the generic scalar kernel are one formula
        compute_amd_scalar_diffusivity!(κref, x.νt, x.u, x.v, x.w, x.θ; Pr_t = x.Pr_t)
        @test interior_view(κref) == interior_view(x.κt)
        nq = count(interior_view(κq) .!= interior_view(x.κt))
        @test nq > 0.5 * 16^3                                     # differs from θ's
        # κ_t is invariant under ϕ → aϕ + b: qv = a θ + b diffuses with θ's κ_t
        interior_view(x.qv) .= 0.01 .- 2e-5 .* (interior_view(x.θ) .- 300); fill_halos!(x.qv, x.θ_bcs)
        κa = sgs_tracer_diffusivity!(Field(CellCenter(), m), AMDClosure(), f, x, x.qv)
        @test interior_view(κa) ≈ interior_view(x.κt) rtol = 1e-9
        # no resolved qv gradient → ν_t/Pr_t
        x.qv.data .= 0.008
        κu = sgs_tracer_diffusivity!(Field(CellCenter(), m), AMDClosure(), f, x, x.qv)
        @test interior_view(κu) == interior_view(x.νt) ./ x.Pr_t
        # a Prandtl-number closure shares one coefficient (returned as is)
        fv = compute_sgs!(VremanClosure(), x)
        @test sgs_tracer_diffusivity!(Field(CellCenter(), m), VremanClosure(), fv, x, x.qv) === fv.κh
        # step_imex! freezes qv's own coefficient for AMD, the shared one for Vreman
        ws = Aeolus.TimeIntegration.IMEXWorkspace(x)
        interior_view(x.qv) .= Q; fill_halos!(x.qv, x.θ_bcs)
        _, cs = Aeolus.TimeIntegration._freeze_viscosity!(x, ws, AMDClosure())
        @test cs isa NamedTuple && cs.qv.a == x.ν .+ κq.data && cs.θ.a == x.ν .+ x.κt.data
        _, cv = Aeolus.TimeIntegration._freeze_viscosity!(x, ws, VremanClosure())
        @test !(cv isa NamedTuple)
        # and conserves Σqv through the iterative solve (tol 1e-10)
        Sq = sum(interior_view(x.qv))
        for _ in 1:5
            step_imex!(x, 0.5, ARK2(); sgs = AMDClosure())
        end
        dq = abs(sum(interior_view(x.qv)) - Sq) / Sq
        @test dq <= 1e-10 && all(isfinite, interior_view(x.qv))
        @printf("  [contract] AMD κ_t(qv) ≠ κ_t(θ) in %d / %d cells; mean κ_t(qv) %.4f vs κ_t(θ) %.4f m²/s; IMEX 5 steps |ΔΣqv|/Σqv %.1e\n",
                nq, 16^3, mean(interior_view(κq)), mean(interior_view(f.κh)), dq)

        # step! / step_rk3! diffuse qv explicitly (#177) by the same rule
        # (merge train, #215): qv's own AMD diffusivity, f.κh (bit for bit
        # add_sgs_scalar!) for a Prandtl-number closure
        interior_view(x.qv) .= Q; fill_halos!(x.qv, x.θ_bcs)
        foreach(g -> fill_halos!(g, x.bcs), (x.u, x.v, x.w)); fill_halos!(x.θ, x.θ_bcs)
        fa = compute_sgs!(AMDClosure(), x)
        tq = Field(CellCenter(), m); tref = Field(CellCenter(), m); tθκ = Field(CellCenter(), m)
        Aeolus.Simulations._add_sgs_tracer!(tq, x.qv, AMDClosure(), fa, x, Field(CellCenter(), m))
        κe = sgs_tracer_diffusivity!(Field(CellCenter(), m), AMDClosure(), fa, x, x.qv)
        Aeolus.Operators.add_eddy_diffusion_scalar!(tref, x.qv, κe)
        @test interior_view(tq) == interior_view(tref)
        add_sgs_scalar!(tθκ, x.qv, AMDClosure(), fa, x)                # with θ's κ_t
        @test interior_view(tq) != interior_view(tθκ)
        fs = compute_sgs!(StaticSmagorinsky(0.17), x)
        ts = Field(CellCenter(), m); tsr = Field(CellCenter(), m)
        Aeolus.Simulations._add_sgs_tracer!(ts, x.qv, StaticSmagorinsky(0.17), fs, x, Field(CellCenter(), m))
        add_sgs_scalar!(tsr, x.qv, StaticSmagorinsky(0.17), fs, x)
        @test ts.data == tsr.data
        # Σqv conserved by the explicit steppers with qv's own κ_t
        Sq = sum(interior_view(x.qv))
        for _ in 1:5
            step!(x, 0.2; sgs = AMDClosure()); step_rk3!(x, 0.2; sgs = AMDClosure())
        end
        dqe = abs(sum(interior_view(x.qv)) - Sq) / Sq
        @test dqe <= 1e-12 && all(isfinite, interior_view(x.qv))
        @printf("  [contract] step!/step_rk3! with AMD: qv tendency ≠ θ-κ_t one in %d cells; 10 steps |ΔΣqv|/Σqv %.1e\n",
                count(interior_view(tq) .!= interior_view(tθκ)), dqe)
    end
end
