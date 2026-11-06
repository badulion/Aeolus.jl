# Era II — the dynamic Smagorinsky closure (Germano et al. 1991 / Lilly
# 1992) in developed turbulence: a k^(-5/3) random-phase field on 32³ is
# evolved ~1 eddy turnover (100 steps) of static-Smagorinsky LES, after
# which the dynamic C_s must approach Lilly's inertial-range value 0.17
# while the random-phase field gives C_s ≈ 0. The closure stays tested in
# test/sgs_numerics.jl (vanishing limits, |S| consistency, Galilean
# invariance, plane averaging, argument checks, the step! swap-ins).

using Test
using Aeolus
using Random: Xoshiro
using Printf

isdefined(@__MODULE__, :kolmogorov_field!) || include(joinpath(@__DIR__, "..", "test", "support", "kolmogorov.jl"))
isdefined(@__MODULE__, :_dynamic_cs) || include(joinpath(@__DIR__, "..", "test", "support", "dynamic_smagorinsky.jl"))

@testset "Era II — SGS and momentum-advection swap-ins (#132)" begin

    mesh = RegularGrid(; size = (32, 32, 32), extent = (2π, 2π, 2π), halo = (3, 3, 3))

    @testset "dynamic Smagorinsky: C_s ≈ Lilly's value in developed turbulence, ≈ 0 for random phases" begin
        # The coefficient measures the energy transfer to subfilter
        # scales. A random-phase field with a k^(-5/3) spectrum has the
        # right spectrum but no transfer, so C_s ≈ 0; after the same field
        # has evolved for ~1 eddy turnover under the resolved dynamics
        # (static Smagorinsky LES) the dynamic C_s approaches the
        # inertial-range value 0.17 of Lilly (1967).
        sim = BoussinesqSimulation(mesh; C_s = 0.17, ν = 1e-4)
        kolmogorov_field!(sim, Xoshiro(1))
        cs_random, = _dynamic_cs(sim)
        for _ in 1:100                                                    # t = 1, ~1 turnover
            step!(sim, 0.01)
        end
        cs3, ws3, = _dynamic_cs(sim; filter_ratio = 3)
        cs2, = _dynamic_cs(sim; filter_ratio = 2)
        @info @sprintf("  dynamic C_s: random phases %.4f; evolved (t = 1): α = 3 → %.4f, α = 2 → %.4f",
                       cs_random, cs3, cs2)
        @test 0.10 < cs3 < 0.20
        @test cs_random < 0.25 * cs3
        @test cs2 > cs3               # a narrower assumed test filter needs a larger C_s
        # plane averaging: one coefficient per level
        @test all(k -> allequal(view(ws3.cs2, :, :, k)), 1:mesh.Nz)
    end
end
