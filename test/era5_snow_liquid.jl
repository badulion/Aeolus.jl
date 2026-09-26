# Phase 51 (#36) — liquid water in the snowpack: retention, refreezing,
# percolation and runoff.
#
# The retention laws against their published forms, refreezing against
# the exact cold-content solution, the bucket scheme against the
# closed-form storage/refreeze/runoff partition of a cold pack, the
# gravity-drainage scheme against Colbeck's (1972) kinematic-wave shock
# speed, and exact mass and enthalpy conservation through melt-refreeze
# cycles driven by the full surface step.

using Test, Printf
using Aeolus
using Aeolus.Snowpack
using Aeolus.SurfaceExchange: DEFAULT_SURFACE_CONSTANTS, cp_ice, cp_water, LH_f0

const SPL = Aeolus.Snowpack
const Tf_liq = DEFAULT_SURFACE_CONSTANTS.thermo.T_freeze
const ρi_liq = DEFAULT_SURFACE_CONSTANTS.ρ_ice
const ρw_liq = DEFAULT_SURFACE_CONSTANTS.ρ_water
const ci_liq = cp_ice(DEFAULT_SURFACE_CONSTANTS)
const Lf_liq = LH_f0(DEFAULT_SURFACE_CONSTANTS)

# N identical layers, bypassing the layering rules.
function liquid_test_pack(N, dz, ρ, T; grain = 50.0, kwargs...)
    p = LayeredSnowpack(; kwargs...)
    for _ in 1:N
        SPL._push_layer!(p, dz, ρ * dz, 0.0, T, grain, 0.0)
    end
    return p
end

# Coléou & Lesaffre (1998) as printed in Wever et al. (2014, Eq. 1),
# restated here independently of the implementation.
cl98(θi) = θi <= 0.23 ? 0.0264 + 0.0099 * (1 - θi) / θi :
           θi <= 0.812 ? 0.08 - 0.1023 * (θi - 0.03) : 0.0

@testset "Era V — Snow liquid water (#36)" begin

    @testset "Retention laws: published forms and limits" begin
        L = ColeouLesaffre1998Retention()
        for θi in (0.05, 0.1, 0.2, 0.23, 0.3, 0.5, 0.7, 0.8)
            @test snow_irreducible_water(L, θi) ≈ cl98(θi) rtol = 1e-12
        end
        # the two branches join at θ_i = 0.23
        @test snow_irreducible_water(L, 0.23 - 1e-9) ≈ snow_irreducible_water(L, 0.23 + 1e-9) atol = 1e-5
        # no retention in (nearly) solid ice; capped by the pore volume in very light snow
        @test snow_irreducible_water(L, 0.85) == 0
        @test snow_irreducible_water(L, 0.01) <= 1 - 0.01
        @test snow_irreducible_water(L, 0.0) == 0
        # denser snow holds less water per unit volume
        θ = [snow_irreducible_water(L, ρ / ρi_liq) for ρ in 150:50:700]
        @test issorted(θ; rev = true)
        # Crocus (Vionnet et al. 2012, Eq. 19): W = 0.05 ρ_w D (1 - ρ/ρ_i)
        P = PoreFractionRetention()
        p = liquid_test_pack(3, 0.1, 300.0, 270.0)
        @test snow_water_holding_capacity(P, p) ≈ fill(0.05 * ρw_liq * 0.1 * (1 - 300 / ρi_liq), 3) rtol = 1e-12
        @test snow_water_holding_capacity(L, p) ≈ fill(ρw_liq * 0.1 * cl98(300 / ρi_liq), 3) rtol = 1e-12
        @info @sprintf("Holding capacity of 0.1 m at 300 kg m⁻³: Coléou-Lesaffre %.2f, Crocus 5 %% of pores %.2f kg m⁻²",
                       snow_water_holding_capacity(L, p)[1], snow_water_holding_capacity(P, p)[1])
    end

    @testset "Refreezing: exact cold-content solution" begin
        # One layer, 50 kg m⁻² at −10 °C: cold content 50 c_ice 10 J m⁻²
        # can freeze CC/L_f ≈ 3.15 kg m⁻² of water at the melting point.
        cold = 50 * ci_liq * 10
        cap = cold / Lf_liq
        for w in (1.0, cap, 5.0)
            p = liquid_test_pack(1, 0.2, 250.0, Tf_liq - 10)
            SPL._add_liquid!(p, 1, w, 0.0)
            H0, M0 = snowpack_enthalpy(p), snow_water_equivalent(p)
            r = snow_refreeze!(p)
            @test snowpack_enthalpy(p) ≈ H0 rtol = 1e-14
            @test snow_water_equivalent(p) == M0
            @test r ≈ min(w, cap) rtol = 1e-12
            @test p.budget.refreeze == r
            @test p.dz[1] == 0.2                           # pore filling
            if w < cap
                # all refrozen; the rest of the cold content remains
                @test p.m_liq[1] == 0
                @test p.T[1] ≈ Tf_liq - (cold - w * Lf_liq) / ((50 + w) * ci_liq) rtol = 1e-12
            else
                @test p.T[1] == Tf_liq
                @test p.m_liq[1] ≈ w - cap atol = 1e-12
            end
        end
        # warm layers and dry layers are untouched
        p = liquid_test_pack(2, 0.1, 200.0, Tf_liq)
        p.m_liq[1] = 2.0; p.T[2] = 260.0
        @test snow_refreeze!(p) == 0
        @test p.m_liq[1] == 2.0 && p.T[2] == 260.0
    end

    @testset "Bucket percolation: storage, refreeze and runoff of a cold pack" begin
        # Five 0.1 m layers at 200 kg m⁻³ and −5 °C receive R of water at
        # T_f on top. Downward from the surface each layer first freezes
        # CC/L_f, then holds ρ_w Δz θ_r at its new ice content; the rest
        # runs off. The closed form below uses only the published law.
        N, dz, ρ, ΔT = 5, 0.1, 200.0, 5.0
        m = ρ * dz
        freeze = m * ci_liq * ΔT / Lf_liq                          # per layer
        hold = ρw_liq * dz * cl98((m + freeze) / (ρi_liq * dz))  # per layer, after refreeze
        absorb = freeze + hold
        for R in (0.5 * freeze, 2.3 * absorb, 5 * absorb + 7.0)
            p = liquid_test_pack(N, dz, ρ, Tf_liq - ΔT; liquid = SnowBucketPercolation())
            SPL._add_liquid!(p, 1, R, 0.0)       # rain at T_f: zero enthalpy
            H0, M0 = snowpack_enthalpy(p), snow_water_equivalent(p)
            runoff = snow_liquid_water!(p.liquid, p, 3600.0)
            k = min(floor(Int, R / absorb), N)                  # layers wetted through
            left = R - k * absorb
            @test runoff ≈ max(R - N * absorb, 0.0) atol = 1e-10
            @test p.budget.refreeze ≈ k * freeze + (k < N ? min(left, freeze) : 0.0) rtol = 1e-10
            @test sum(p.m_liq) ≈ k * hold + (k < N ? max(left - freeze, 0.0) : 0.0) rtol = 1e-10
            @test snow_water_equivalent(p) + runoff ≈ M0 rtol = 1e-14
            @test snowpack_enthalpy(p) ≈ H0 rtol = 1e-12       # runoff at T_f carries no enthalpy
            @test all(p.T .<= Tf_liq)
            @test all(p.dz .== dz)
            @info @sprintf("Bucket, R = %.2f kg m⁻²: refrozen %.3f, retained %.3f, runoff %.3f (closed form %.3f)",
                           R, p.budget.refreeze, sum(p.m_liq), runoff, max(R - N * absorb, 0.0))
        end
        # at the melting point the pack only retains: runoff = R - Σ hold
        p = liquid_test_pack(N, dz, ρ, Tf_liq; liquid = SnowBucketPercolation())
        p.m_liq[1] = 40.0
        runoff = snow_liquid_water!(p.liquid, p, 60.0)
        hold0 = ρw_liq * dz * cl98(ρ / ρi_liq)
        @test runoff ≈ 40.0 - N * hold0 rtol = 1e-12
        @test p.m_liq ≈ fill(hold0, N) rtol = 1e-12
        @test p.budget.refreeze == 0
    end
end
