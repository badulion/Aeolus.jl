# Phase 50 — BuildingEnergy validation (#32): the resolved zone with ideal
# loads against the ISO 13790 5R1C ideal cooling need (§C.4.2) for the same
# construction and the same hot 5-day forcing; day-5 cooling energy ratio.
# HVAC demand, exhaust heat and the 5R1C setpoint holding are unit-tested
# in test/hvac.jl.

using Test, Printf
using Aeolus
using Aeolus.BuildingEnergy
using Aeolus.UrbanSurface

isdefined(@__MODULE__, :hvac_mass_box) || include(joinpath(@__DIR__, "..", "test", "support", "building_box.jl"))

@testset "BuildingEnergy: HVAC, resolved zone vs ISO 13790 5R1C (#32)" begin
    @testset "Resolved zone vs ISO 13790 5R1C ideal cooling need" begin
        # Same construction, same hourly forcing, both with ideal loads; the
        # 5R1C need follows ISO 13790 §C.4.2. Agreement bounds are envelopes
        # set after measuring (as in #31), not physics expectations.
        z = hvac_mass_box(T = 299.15)
        iso = iso13790_zone(z)
        h = IdealLoadsHVAC(T_heat = 293.15, T_cool = 299.15, curves = nothing)
        Tout(t) = 302.15 + 6 * sin(2π * (t / 86400 - 0.375))
        Qi(t) = 300.0 + (8 <= mod(t / 3600, 24) < 18 ? 500.0 : 0.0)
        Qs(t) = max(0.0, 1500 * sin(2π * (t / 86400 - 0.25)))
        E_res = zeros(5); E_iso = zeros(5); P_res = zeros(5); P_iso = zeros(5)
        for k in 0:(5 * 24 - 1)
            t = (k + 0.5) * 3600; d = k ÷ 24 + 1
            _, hb = step_zone!(z, ZoneForcing(T_out = Tout(t), Q_internal = Qi(t), Q_solar = Qs(t)), 3600.0, h)
            Φ, _, _, _ = iso13790_ideal_load!(iso, Tout(t), 3600.0; θ_heat = 293.15, θ_cool = 299.15,
                                             Φ_int = Qi(t), Φ_sol = Qs(t))
            E_res[d] += hb.Q_cooling_sensible * 3600; E_iso[d] += max(-Φ, 0.0) * 3600
            P_res[d] = max(P_res[d], hb.Q_cooling_sensible); P_iso[d] = max(P_iso[d], -Φ)
        end
        r = E_res[5] / E_iso[5]
        @printf("  day-5 cooling need: resolved %.2f kWh, ISO 13790 5R1C %.2f kWh (ratio %.3f); peak %.0f / %.0f W\n",
                E_res[5] / 3.6e6, E_iso[5] / 3.6e6, r, P_res[5], P_iso[5])
        @test 0.8 < r < 1.25
    end
end
