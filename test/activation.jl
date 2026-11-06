# Phase 42 — aerosol-cloud interaction.
#
# Köhler theory, the Abdul-Razzak & Ghan (ARG) parameterisation and the
# explicit adiabatic parcel it approximates. The headline validation, ARG
# against the parcel over a range of updraughts and aerosol loadings, is
# validation/activation.jl.

using Test
using Aeolus

@testset "Aerosol-cloud interaction (Phase 42)" begin

    @testset "Köhler theory agrees with the Phase 39 contract" begin
        # One implementation of Köhler (MoistThermodynamics); the closed
        # form may differ from it only by the dilute approximation.
        cases = [(rd, κ) for rd in (2e-8, 1e-7, 2e-7), κ in (0.1, 0.6, 1.28)]
        exact = [last(kohler_critical_point(rd, κ, 298.15)) for (rd, κ) in cases]
        closed = [critical_supersaturation(rd, κ) for (rd, κ) in cases]
        numerical = [kohler_critical(rd, κ).s_crit for (rd, κ) in cases]
        @test maximum(abs.(numerical ./ exact .- 1)) < 5e-3
        @test maximum(abs.(closed ./ exact .- 1)) < 3e-2

        rd, κ, T = 1e-7, 0.6, 293.15
        rc, sc = kohler_critical_point(rd, κ, T)
        # it really is a maximum
        @test kohler_supersaturation(rc * 0.7, rd, κ, T) < sc > kohler_supersaturation(rc * 1.5, rd, κ, T)
        # s_crit ∝ r_d^(-3/2) and κ^(-1/2)
        @test last(kohler_critical_point(5e-8, κ, T)) / sc ≈ 2^1.5 rtol = 0.05
        @test last(kohler_critical_point(rd, 0.3, T)) /
              last(kohler_critical_point(rd, 1.2, T)) ≈ 2.0 rtol = 0.05
        # stable haze equilibrium below s_crit, none above
        req = kohler_equilibrium_radius(0.5 * sc, rd, κ, T)
        @test rd < req < rc
        @test kohler_supersaturation(req, rd, κ, T) ≈ 0.5 * sc rtol = 1e-6
        @test isnan(kohler_equilibrium_radius(1.5 * sc, rd, κ, T))
    end

    @testset "ARG activation: updraught and competition" begin
        T, p = 283.0, 9.0e4
        co = activation_coefficients(T, p)
        mode(N) = [arg_mode(N, 0.05e-6, 2.0, 0.6)]
        # more updraught ⇒ higher S_max and more activation
        rs = [arg_activation(mode(500e6), w, co) for w in (0.1, 1.0, 3.0)]
        @test all(diff([r.fraction[1] for r in rs]) .> 0)
        @test all(diff([r.S_max for r in rs]) .> 0)
        # more aerosol ⇒ more droplets but a smaller fraction and lower S_max
        # (the particles compete for the same vapour)
        res = [arg_activation(mode(N * 1e6), 0.5, co) for N in (50.0, 1000.0, 5000.0)]
        @test all(diff([r.N_act_total for r in res]) .> 0)
        @test all(diff([r.fraction[1] for r in res]) .< 0)
        @test all(diff([r.S_max for r in res]) .< 0)
        # multiple modes: coarse sea salt activates most, Aitken least
        r = arg_activation([arg_mode(800e6, 0.02e-6, 1.6, 0.6),
                            arg_mode(200e6, 0.08e-6, 2.0, 0.6),
                            arg_mode( 10e6, 0.5e-6,  2.2, 1.1)], 0.5, co)
        @test r.fraction[3] > r.fraction[2] > r.fraction[1]
        @test r.N_act_total ≈ sum(r.N_act) rtol = 1e-12
        # (the explicit parcel, parcel_activation, is exercised and compared
        # with ARG in validation/activation.jl)
    end

    @testset "Prognostic aerosol conserves number" begin
        res = aerosol_reservoir([arg_mode(500e6, 0.05e-6, 2.0, 0.6),
                                 arg_mode(100e6, 0.2e-6,  1.8, 1.1)])
        n0 = aerosol_total_number(res)
        activate!(res, 0.003; T_K = 283.0)
        @test aerosol_total_number(res) ≈ n0 rtol = 1e-10
        @test 0 < activated_number(res) && interstitial_number(res) < n0
        regenerate!(res, 1.0)
        @test aerosol_total_number(res) ≈ n0 rtol = 1e-10
    end
end
