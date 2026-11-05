# Phase 51 (#36) — liquid water in the snowpack: retention, refreezing,
# percolation and runoff.
#
# The retention laws against their published forms, refreezing against
# the exact cold-content solution, the bucket scheme against the
# closed-form storage/refreeze/runoff partition of a cold pack, the
# gravity-drainage scheme against Colbeck's (1972) kinematic-wave shock
# speed, and exact mass and enthalpy conservation through melt-refreeze
# cycles driven by the full surface step.

using Test
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

    @testset "Retention laws: published forms" begin
        L = ColeouLesaffre1998Retention()
        θis = (0.05, 0.2, 0.3, 0.7, 0.8)
        @test maximum(abs(snow_irreducible_water(L, θi) / cl98(θi) - 1) for θi in θis) < 1e-12
        @test snow_irreducible_water(L, 0.85) == 0
        # Crocus (Vionnet et al. 2012, Eq. 19): W = 0.05 ρ_w D (1 − ρ/ρ_i)
        p = liquid_test_pack(3, 0.1, 300.0, 270.0)
        @test snow_water_holding_capacity(PoreFractionRetention(), p) ≈
              fill(0.05 * ρw_liq * 0.1 * (1 - 300 / ρi_liq), 3) rtol = 1e-12
    end

    @testset "Refreezing: exact cold-content solution" begin
        # 50 kg m⁻² at −10 °C can freeze CC/L_f ≈ 3.15 kg m⁻² of water
        cold = 50 * ci_liq * 10
        cap = cold / Lf_liq
        for w in (1.0, 5.0)
            p = liquid_test_pack(1, 0.2, 250.0, Tf_liq - 10)
            SPL._add_liquid!(p, 1, w, 0.0)
            H0 = snowpack_enthalpy(p)
            r = snow_refreeze!(p)
            @test snowpack_enthalpy(p) ≈ H0 rtol = 1e-14
            @test r ≈ min(w, cap) rtol = 1e-12
            if w < cap
                @test p.T[1] ≈ Tf_liq - (cold - w * Lf_liq) / ((50 + w) * ci_liq) rtol = 1e-12
            else
                @test p.T[1] == Tf_liq
            end
        end
        # refreezing coarsens grains through the #35 hook (CLM Eq. 2.3.72):
        # 2 kg m⁻² refrozen in 32 kg m⁻²: f = 1/16 towards 1000 μm
        p = liquid_test_pack(1, 0.1, 300.0, Tf_liq - 20; grain = 100.0,
                             metamorphism = Flanner2006Metamorphism())
        p.m_liq[1] = 2.0
        snow_refreeze!(p)
        @test p.grain[1] ≈ 156.25
    end

    @testset "Bucket percolation: storage, refreeze and runoff of a cold pack" begin
        # Five layers at −5 °C receive R at T_f on top: each layer freezes
        # CC/L_f, then holds ρ_w Δz θ_r; the rest runs off.
        N, dz, ρ, ΔT = 5, 0.1, 200.0, 5.0
        m = ρ * dz
        freeze = m * ci_liq * ΔT / Lf_liq
        hold = ρw_liq * dz * cl98((m + freeze) / (ρi_liq * dz))
        absorb = freeze + hold
        R = 2.3 * absorb
        p = liquid_test_pack(N, dz, ρ, Tf_liq - ΔT; liquid = SnowBucketPercolation())
        SPL._add_liquid!(p, 1, R, 0.0)
        H0, M0 = snowpack_enthalpy(p), snow_water_equivalent(p)
        runoff = snow_liquid_water!(p.liquid, p, 3600.0)
        k = 2; left = R - k * absorb
        @test runoff == 0
        @test p.budget.refreeze ≈ k * freeze + min(left, freeze) rtol = 1e-10
        @test sum(p.m_liq) ≈ k * hold + max(left - freeze, 0.0) rtol = 1e-10
        @test snow_water_equivalent(p) ≈ M0 rtol = 1e-14
        @test snowpack_enthalpy(p) ≈ H0 rtol = 1e-12
        # at the melting point the pack only retains: runoff = R − Σ hold
        p = liquid_test_pack(N, dz, ρ, Tf_liq; liquid = SnowBucketPercolation())
        p.m_liq[1] = 40.0
        @test snow_liquid_water!(p.liquid, p, 60.0) ≈ 40.0 - N * ρw_liq * dz * cl98(ρ / ρi_liq) rtol = 1e-12
    end

    @testset "Gravity drainage: Calonne K_s and Colbeck (1972) shock speed" begin
        # Calonne et al. (2012): k = 3.0 r² exp(−0.0130 ρ), K = k ρ_w g / μ
        r = 0.5e-3
        @test snow_saturated_hydraulic_conductivity(400.0, r) ≈
              3.0 * r^2 * exp(-0.0130 * 400) * 1000 * 9.80665 / 1.792e-3 rtol = 1e-3
        # isothermal pack, constant q₀: a shock filling the pores to θ₁ with
        # ρ_w K_s S_e(θ₁)³ = q₀, moving at v = q₀/(ρ_w θ₁)
        N, dz, ρ, rg = 30, 0.05, 350.0, 250.0
        D = N * dz
        q0 = 5e-3
        θi = ρ / ρi_liq
        θr, θs = cl98(θi), 1 - θi
        Ks = snow_saturated_hydraulic_conductivity(ρ, rg * 1e-6)
        θ1 = θr + (θs - θr) * (q0 / (ρw_liq * Ks))^(1 / 3)
        t_arrival = D / (q0 / (ρw_liq * θ1))
        p = liquid_test_pack(N, dz, ρ, Tf_liq; grain = rg, liquid = SnowGravityDrainage())
        M0 = snow_water_equivalent(p)
        dt = 120.0
        t, t_half, runoff, input, rate = 0.0, NaN, 0.0, 0.0, 0.0
        while t < 2.5 * t_arrival
            SPL._add_liquid!(p, 1, q0 * dt, 0.0)
            input += q0 * dt
            r = snow_liquid_water!(p.liquid, p, dt)
            runoff += r; rate = r / dt; t += dt
            isnan(t_half) && rate >= q0 / 2 && (t_half = t - dt / 2)
        end
        # 5 cm layers (was 2.5 cm): the front is smeared over twice the
        # depth, but its arrival time is set by v, still within 3 %
        @test abs(t_half / t_arrival - 1) < 0.03
        @test rate ≈ q0 rtol = 1e-3
        @test snow_liquid_water_content(p)[N ÷ 2] ≈ θ1 rtol = 1e-3
        @test snow_water_equivalent(p) + runoff ≈ M0 + input rtol = 1e-13
    end

    @testset "Melt-refreeze cycles through the surface step: conservation" begin
        # four spring days: sunny days melt, clear nights refreeze, rain on day 3
        σ = DEFAULT_SURFACE_CONSTANTS.σ_SB
        forcing(d, hr) = begin
            sun = max(0.0, sin(π * (hr - 6) / 12))
            Ta = 271.0 + 5 * sin(π * (hr - 9) / 12)
            SurfaceForcing(sw_down = 650sun, lw_down = 0.8σ * Ta^4, T_air = Ta, q_air = 0.0035,
                           U = 3.0, p = 85000.0, z_ref = 10.0,
                           rain = d == 3 && 10 <= hr < 14 ? 1.5e-3 : 0.0, snow = 0.0)
        end
        res = Dict{Symbol,Any}()
        for (name, liq) in ((:instant, InstantSnowRunoff()), (:gravity, SnowGravityDrainage()))
            p = LayeredSnowpack(depth = 0.6, density = 300.0, T = 268.0, liquid = liq,
                                bottom = SnowGroundTemperature(Tf_liq))
            M0, H0 = snow_water_equivalent(p), snowpack_enthalpy(p)
            morn = Float64[]
            for d in 1:4, s in 1:24
                step_surface!(p, forcing(d, s), 3600.0)
                s == 6 && push!(morn, sum(p.m_liq))
            end
            b = p.budget
            mres = snow_water_equivalent(p) - M0 - (b.snowfall + b.rainfall - b.sublimation - b.runoff)
            hres = snowpack_enthalpy(p) - H0 -
                   (b.surface_heat - b.bottom_heat + b.advected_enthalpy - b.runoff_enthalpy)
            @test abs(mres) < 1e-12 * M0
            @test abs(hres) < 1e-12 * abs(H0)
            @test sum(p.m_liq) ≈ b.melt + b.rainfall - b.refreeze - b.runoff atol = 1e-9
            res[name] = (p = p, morn = morn)
        end
        g = res[:gravity]
        @test g.p.budget.refreeze > 0.5 * g.p.budget.melt         # most meltwater refreezes
        @test g.p.budget.runoff < res[:instant].p.budget.runoff   # rain is retained
        @test all(g.morn[2:3] .== 0)                              # none left at dawn
    end
end
