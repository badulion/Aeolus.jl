# Era VI, P55-d (#176): the anisotropic minimum-dissipation (AMD) and
# Vreman SGS closures (src/operators/sgs_amd_vreman.jl).
#
#   * exact closed forms on linear fields on the Berlin grid spacing
#     (15.625, 15.625, 6.25) m;
#   * laminar switch-off (uniform flow, pure shear; AMD: solid-body
#     rotation), where Smagorinsky is not zero;
#   * (dissipation in developed turbulence relative to Smagorinsky 0.17 is
#     a validation run, validation/era6_amd_vreman.jl)
# The Comte-Bellot–Corrsin rows are in validation/era6_cbc_decay.jl.

using Test
using Aeolus
using Aeolus.Fields: interior_view, Field
using Aeolus.Grids: interior

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
    D = sum(abs2, A)
    Ri = abs(βg) * sqrt(sum(abs2, γ)) / D               # the #504 weight min(1, 1/Ri_∇)
    ν = C * max(0, -P + B * min(1, 1 / Ri)) / D
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

# closures!: the AMD, Vreman and Smagorinsky 0.17 coefficients of a state (shared
# with validation/era6_amd_vreman.jl)
isdefined(@__MODULE__, :closures!) || include(joinpath(@__DIR__, "support", "amd_vreman_closures.jl"))

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
        @test all(==(0), νa)
        # Vreman is not zero for rotation (Vreman 2004: B_β = Δx²Δy²Ω⁴)
        @test all(x -> relerr(x, 2.5 * 0.17^2 * sqrt(δ[1]^2 * δ[2]^2 * Ω^4 / (2Ω^2))) < 1e-12, νv)
    end

end
