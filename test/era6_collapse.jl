# P58-e (#193): turbulence collapse — the maximum sustainable heat flux
# (MSHF, Van de Wiel et al. 2012) of SurfaceExchange, and the LES
# collapse regime of the flux-cooled half channel (Donda et al. 2015).

using Test
using Aeolus
using Aeolus.SurfaceExchange: _Φm, von_karman, grav

@testset "P58-e: maximum sustainable heat flux" begin
    κ = von_karman(); g = grav()
    LL = BusingerDyerLogLinear(β = 5.0)
    BDBH = BusingerDyerBeljaarsHoltslag()

    @testset "1. analytic MSHF: log-linear closed form, numerical maximum" begin
        @test log_linear_slope(LL) == 5.0
        @test log_linear_slope(BDBH) === nothing
        for (U, z, z0m, θ) in ((5.0, 10.0, 0.1, 290.0), (2.0, 2.0, 0.01, 280.0),
                               (0.5, 1.0, 0.05, 300.0), (12.6, 1 / 64, 1e-4, 300.0))
            Λ = log(z / z0m)
            cf = mshf_solution(LL, U, z, z0m, θ)
            nm = mshf_solution(LL, U, z, z0m, θ; method = :numerical)
            @test cf.method === :closed_form && nm.method === :numerical && nm.interior
            # the closed form of the issue / Van de Wiel et al. (2012), with
            # the exact ψ_m(z0m/L) term: z → z - z0m
            Qcf = 4 / 27 * θ * κ^2 * U^3 / (5.0 * g * (z - z0m) * Λ^2)
            @test cf.Q_max ≈ Qcf rtol = 1e-14
            @test abs(nm.Q_max - cf.Q_max) <= 1e-8 * cf.Q_max
            # argmax u* = 2κU/(3Λ) (the numerical maximum is flat there:
            # an error δ in ζ costs δ² in Q, δ in u*)
            @test cf.u_star ≈ 2κ * U / (3Λ) rtol = 1e-14
            @test abs(nm.u_star - 2κ * U / (3Λ)) <= 1e-6 * cf.u_star
            @test cf.ζ ≈ Λ * z / (2 * 5.0 * (z - z0m)) rtol = 1e-14
            # the MSHF point lies on the flux curve Q = κ²U³θ ζ/(g z Φm³)
            @test cf.Q_max ≈ κ^2 * U^3 * θ * cf.ζ / (g * z * _Φm(LL, cf.ζ, z, z0m)^3) rtol = 1e-12
            # z0h does not enter (the momentum half of MOST)
            @test max_sustainable_heat_flux(LL, U, z, z0m, z0m / 10, θ) ==
                  max_sustainable_heat_flux(LL, U, z, z0m, z0m / 1000, θ)
            # scaling: Q_max ∝ U³ θ
            @test max_sustainable_heat_flux(LL, 2U, z, z0m, z0m, 2θ) ≈ 16cf.Q_max rtol = 1e-13
        end
        # the closed form needs a log-linear family
        @test_throws ArgumentError mshf_solution(BDBH, 5.0, 10.0, 0.1, 290.0; method = :closed_form)
        @test_throws ArgumentError mshf_solution(BDBH, 5.0, 0.1, 0.1, 290.0)

        # BDBH: finite maximum, interior, and the flux curve is unimodal in u*
        for (U, z, z0m) in ((5.0, 10.0, 0.1), (1.0, 2.0, 0.05), (8.0, 1.5625, 0.05))
            m = mshf_solution(BDBH, U, z, z0m, 290.0)
            @test m.method === :numerical && m.interior
            @test isfinite(m.Q_max) && m.Q_max > 0
            # BDBH is less stable than log-linear α = 5 (φ_m < 1 + 5ζ for ζ > 0),
            # so it sustains more flux, but not more than log-linear α = a = 1
            @test m.Q_max > mshf_solution(LL, U, z, z0m, 290.0).Q_max
            @test m.Q_max < mshf_solution(BusingerDyerLogLinear(β = 1.0), U, z, z0m, 290.0).Q_max
            ζs = exp.(range(log(1e-4), log(1e3); length = 2000))
            us = κ * U ./ [_Φm(BDBH, ζ, z, z0m) for ζ in ζs]           # decreasing in ζ
            Qs = κ^2 * U^3 * 290.0 / (g * z) .* ζs ./ [_Φm(BDBH, ζ, z, z0m)^3 for ζ in ζs]
            @test all(diff(us) .< 0)
            ip = argmax(Qs)
            @test 1 < ip < length(Qs)
            @test all(diff(Qs[1:ip]) .> 0) && all(diff(Qs[ip:end]) .< 0)   # one maximum
            @test maximum(Qs) <= m.Q_max * (1 + 1e-12)
            @test maximum(Qs) >= m.Q_max * (1 - 1e-4)                     # grid resolution
        end
    end

    @testset "2. every solve_surface_layer solution is below the MSHF" begin
        θa = 290.0
        for sf in (BDBH, LL)
            worst = 0.0; n = 0
            for ΔT in range(0, 20; length = 41), U in range(0.5, 10; length = 39)
                sol = solve_surface_layer(sf; U = U, z_m = 10.0, θ_a = θa, θ_s = θa - ΔT,
                                          z0m = 0.1, z0h = 0.01)
                Qdown = sol.u_star * sol.θv_star          # = -w'θv' (stable: > 0)
                Qm = max_sustainable_heat_flux(sf, sol.U, 10.0, 0.1, 0.01, θa)
                worst = max(worst, Qdown / Qm)
                n += Qdown <= Qm * (1 + 1e-10)
            end
            @test n == 41 * 39
            # the solutions come right up to the bound: the surface flux as a
            # function of ΔT attains the MSHF (Van de Wiel et al. 2012, Fig. 2)
            @test 0.999 < worst <= 1 + 1e-10
        end
    end

    @testset "flux-prescribed friction velocity" begin
        U, z, z0m, θ = 5.0, 10.0, 0.1, 290.0
        Λ = log(z / z0m)
        @test flux_friction_velocity(BDBH, U, z, z0m, 0.0, θ) ≈ κ * U / Λ rtol = 1e-14
        for sf in (BDBH, LL)
            m = mshf_solution(sf, U, z, z0m, θ)
            for frac in (0.1, 0.5, 0.9, 0.999)
                us = flux_friction_velocity(sf, U, z, z0m, -frac * m.Q_max, θ)
                # weakly stable branch: between the MSHF point and neutral
                @test m.u_star * (1 - 1e-9) <= us < κ * U / Λ
                # and it is a solution: u* = κU/Φm(ζ), ζ = zκgQ/(u*³θ)
                ζ = z * κ * g * frac * m.Q_max / (us^3 * θ)
                @test us ≈ κ * U / _Φm(sf, ζ, z, z0m) rtol = 1e-9
            end
            @test flux_friction_velocity(sf, U, z, z0m, -1.01 * m.Q_max, θ) == 0   # collapse
            # unstable: u* above neutral, and a solution
            us = flux_friction_velocity(sf, U, z, z0m, 0.2, θ)
            ζ = -z * κ * g * 0.2 / (us^3 * θ)
            @test us > κ * U / Λ
            @test us ≈ κ * U / _Φm(sf, ζ, z, z0m) rtol = 1e-9
        end
        # log-linear weak branch in closed form: Q = (κU u*² − Λu*³)θ/(α g z′ κ)
        frac = 0.5; m = mshf_solution(LL, U, z, z0m, θ)
        us = flux_friction_velocity(LL, U, z, z0m, -frac * m.Q_max, θ)
        @test (κ * U * us^2 - Λ * us^3) * θ / (5.0 * g * (z - z0m) * κ) ≈ frac * m.Q_max rtol = 1e-9
    end
end
