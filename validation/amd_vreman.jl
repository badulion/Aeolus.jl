# Era VI, P55-d (#176): validation of the AMD and Vreman SGS closures in
# developed turbulence — a k^(-5/3) field (32³) evolved for t = 1 with
# static Smagorinsky 0.17, where both closures dissipate ⟨2ν_t S S⟩ within
# ±40 % of Smagorinsky 0.17. The closures stay tested in
# test/amd_vreman.jl (closed forms, laminar switch-off, buoyancy,
# Galilean invariance, the #504 weight, the contract); the
# Comte-Bellot–Corrsin rows are in validation/cbc_decay.jl.

using Test
using Aeolus
using Aeolus.Fields: interior_view, Field
using Random: Xoshiro
using Statistics: mean
using Printf
isdefined(@__MODULE__, :kolmogorov_field!) || include(joinpath(@__DIR__, "..", "test", "support", "kolmogorov.jl"))
# closures!: the AMD, Vreman and Smagorinsky 0.17 coefficients of a state (shared
# with test/amd_vreman.jl)
isdefined(@__MODULE__, :closures!) || include(joinpath(@__DIR__, "..", "test", "support", "amd_vreman_closures.jl"))

@testset "Era VI — AMD and Vreman SGS closures in turbulence (P55-d, #176)" begin

    @testset "developed turbulence: dissipation relative to Smagorinsky 0.17" begin
        # the decaying-turbulence set-up of test/sgs_numerics.jl: a
        # k^(-5/3) random field evolved for t = 1 (≈ one turnover) with
        # static Smagorinsky 0.17
        mesh = RegularGrid(; size = (32, 32, 32), extent = (2π, 2π, 2π), halo = (3, 3, 3))
        s = BoussinesqSimulation(mesh; C_s = 0.17, ν = 1e-4)
        kolmogorov_field!(s, Xoshiro(1))            # k_c = 8 = N/4, u_rms = 1
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
end
