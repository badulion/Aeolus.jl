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
    @testset "Saturated hydraulic conductivity (Calonne et al. 2012)" begin
        c = DEFAULT_SURFACE_CONSTANTS
        # k = 0.75 r² exp(-0.013 ρ), K = k ρ_w g / μ
        K = snow_saturated_hydraulic_conductivity(400.0, 0.5e-3)
        @test K ≈ 0.75 * 0.5e-3^2 * exp(-0.013 * 400) * 1000 * 9.80665 / 1.792e-3 rtol = 1e-3
        @test 1e-3 < K < 1e-2                  # wet-snow values are of order mm s⁻¹
        # coarser grains and lighter snow drain faster; quadratic in r
        @test snow_saturated_hydraulic_conductivity(400.0, 1e-3) ≈ 4K rtol = 1e-12
        @test snow_saturated_hydraulic_conductivity(300.0, 0.5e-3) > K
    end

    @testset "Gravity drainage: Colbeck (1972) kinematic-wave shock speed" begin
        # A 1.5 m isothermal (0 °C) pack of dry 350 kg m⁻³ snow with
        # 250 μm grains receives a constant q₀ at the top. With the
        # capillary term dropped, q = ρ_w K_s S_e³, and a front advancing
        # into dry snow is a shock that fills the pores to θ₁ with
        # ρ_w K_s S_e(θ₁)³ = q₀, moving at v = q₀/(ρ_w θ₁). The base should
        # start delivering water after D/v, and deliver q₀ once the whole
        # pack carries the front's water content.
        N, dz, ρ, rg = 60, 0.025, 350.0, 250.0
        D = N * dz
        q0 = 5e-3                                             # kg m⁻² s⁻¹ = 18 mm h⁻¹
        θi = ρ / ρi_liq
        θr, θs = cl98(θi), 1 - θi
        Ks = snow_saturated_hydraulic_conductivity(ρ, rg * 1e-6)
        θ1 = θr + (θs - θr) * (q0 / (ρw_liq * Ks))^(1 / 3)
        v = q0 / (ρw_liq * θ1)
        t_arrival = D / v
        p = liquid_test_pack(N, dz, ρ, Tf_liq; grain = rg, liquid = SnowGravityDrainage())
        M0, H0 = snow_water_equivalent(p), snowpack_enthalpy(p)
        dt = 120.0
        t, t_half, runoff, input = 0.0, NaN, 0.0, 0.0
        rate = 0.0
        while t < 2.5 * t_arrival
            SPL._add_liquid!(p, 1, q0 * dt, 0.0)
            input += q0 * dt
            r = snow_liquid_water!(p.liquid, p, dt)
            runoff += r
            rate = r / dt
            t += dt
            isnan(t_half) && rate >= q0 / 2 && (t_half = t - dt / 2)
        end
        @test abs(t_half / t_arrival - 1) < 0.03
        @test rate ≈ q0 rtol = 1e-3                             # steady state
        θw = snow_liquid_water_content(p)
        @test θw[N ÷ 2] ≈ θ1 rtol = 1e-3                       # front water content
        @test snow_water_equivalent(p) + runoff ≈ M0 + input rtol = 1e-13
        @test snowpack_enthalpy(p) ≈ H0 atol = 1e-6            # all at T_f
        @test p.budget.refreeze == 0
        @info @sprintf("Kinematic wave: K_s = %.2e m s⁻¹, θ₁ = %.4f, v = %.2e m s⁻¹; arrival %.2f h (analytic %.2f h, error %.2f %%); steady runoff %.5f of q₀; θ_w mid-pack %.4f",
                       Ks, θ1, v, t_half / 3600, t_arrival / 3600, 100 * (t_half / t_arrival - 1),
                       rate / q0, θw[N ÷ 2])
    end

    @testset "Gravity drainage into a cold pack: the cold content delays runoff" begin
        # The same pack at −8 °C: no water can leave the base before the
        # pack has frozen its whole cold content, CC/L_f, and stored its
        # irreducible water.
        N, dz, ρ, rg = 20, 0.05, 300.0, 300.0
        p = liquid_test_pack(N, dz, ρ, Tf_liq - 8; grain = rg, liquid = SnowGravityDrainage())
        M0, H0 = snow_water_equivalent(p), snowpack_enthalpy(p)
        freeze_total = N * ρ * dz * ci_liq * 8 / Lf_liq
        q0, dt = 2e-3, 300.0
        input, runoff, first = 0.0, 0.0, NaN
        never_warm = true
        for _ in 1:600
            SPL._add_liquid!(p, 1, q0 * dt, 0.0)
            input += q0 * dt
            r = snow_liquid_water!(p.liquid, p, dt)
            runoff += r
            r > 0 && isnan(first) && (first = input)
            never_warm &= all(p.T .<= Tf_liq)
        end
        @test never_warm
        stored = sum(snow_water_holding_capacity(ColeouLesaffre1998Retention(), p))
        @test first > freeze_total + stored
        @test p.budget.refreeze ≈ freeze_total rtol = 1e-9      # whole cold content used
        @test all(p.T .≈ Tf_liq)
        @test snow_water_equivalent(p) + runoff ≈ M0 + input rtol = 1e-13
        @test snowpack_enthalpy(p) ≈ H0 rtol = 1e-12
        @info @sprintf("Cold pack: first runoff after %.2f kg m⁻² of input (cold content freezes %.2f, irreducible storage %.2f); refrozen %.3f",
                       first, freeze_total, stored, p.budget.refreeze)
    end
    @testset "Melt-refreeze cycles through the surface step: conservation" begin
        # Eight spring days on a 0.6 m pack: sunny days melt the surface,
        # clear nights refreeze it, and 21.6 mm of rain falls on day 5.
        # Mass and enthalpy budgets must close to round-off with every
        # scheme; liquid water obeys Δw_liq = melt + rain - refreeze - runoff;
        # diurnal meltwater held near the surface refreezes overnight.
        σ = DEFAULT_SURFACE_CONSTANTS.σ_SB
        forcing(d, hr) = begin
            sun = max(0.0, sin(π * (hr - 6) / 12))
            Ta = 271.0 + 5 * sin(π * (hr - 9) / 12)
            SurfaceForcing(sw_down = 650sun, lw_down = 0.8σ * Ta^4, T_air = Ta, q_air = 0.0035,
                           U = 3.0, p = 85000.0, z_ref = 10.0,
                           rain = d == 5 && 10 <= hr < 14 ? 1.5e-3 : 0.0, snow = 0.0)
        end
        res = Dict{Symbol,Any}()
        for (name, liq) in ((:instant, InstantSnowRunoff()), (:bucket, SnowBucketPercolation()),
                            (:gravity, SnowGravityDrainage()))
            p = LayeredSnowpack(depth = 0.6, density = 300.0, T = 268.0, liquid = liq,
                                bottom = SnowGroundTemperature(Tf_liq))
            M0, H0 = snow_water_equivalent(p), snowpack_enthalpy(p)
            eve, morn = Float64[], Float64[]
            ok_T = true
            for d in 1:8, s in 1:48
                step_surface!(p, forcing(d, s / 2), 1800.0)
                ok_T &= all(p.T .<= Tf_liq)
                s == 36 && push!(eve, sum(p.m_liq))     # 18:00
                s == 12 && push!(morn, sum(p.m_liq))    # 06:00
            end
            b = p.budget
            mres = snow_water_equivalent(p) - M0 - (b.snowfall + b.rainfall - b.sublimation - b.runoff)
            hres = snowpack_enthalpy(p) - H0 -
                   (b.surface_heat - b.bottom_heat + b.advected_enthalpy - b.runoff_enthalpy)
            @test ok_T
            @test abs(mres) < 1e-12 * M0
            @test abs(hres) < 1e-12 * abs(H0)
            @test sum(p.m_liq) ≈ b.melt + b.rainfall - b.refreeze - b.runoff atol = 1e-9
            res[name] = (p = p, eve = eve, morn = morn, mres = mres, hres = hres, H0 = H0)
            @info @sprintf("%-8s melt %.1f, refrozen %.1f, runoff %.1f, retained %.2f kg m⁻²; SWE %.1f → %.1f; residuals %.1e kg m⁻², %.1e J m⁻² (of %.2e)",
                           name, b.melt, b.refreeze, b.runoff, sum(p.m_liq), M0,
                           snow_water_equivalent(p), mres, hres, H0)
        end
        for name in (:bucket, :gravity)
            r = res[name]
            b = r.p.budget
            @test b.refreeze > 0.5 * b.melt                    # most meltwater refreezes
            @test b.runoff < res[:instant].p.budget.runoff - 20   # rain is retained, not lost
            @test snow_water_equivalent(r.p) > snow_water_equivalent(res[:instant].p)
            # before the rain: meltwater every afternoon, none left at dawn
            @test all(r.eve[1:4] .> 0.1)
            @test all(r.morn[2:5] .== 0)
        end
        @test res[:instant].p.budget.refreeze == 0
    end
end
