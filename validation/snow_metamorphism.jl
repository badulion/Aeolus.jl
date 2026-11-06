# Phase 51 (#35) — snow metamorphism validation: the 30-day albedo decay of
# a clear-sky pack (Flanner & Zender grain growth, SnowOpticsAlbedo, clean
# and with 0.05 ppmw black carbon) against the broadband albedo curves of
# four reference land-surface schemes (USACE/VIC, ECMWF, CLASS, BATS).
# Grain growth and the albedo delegation are unit-tested in
# test/snow_metamorphism.jl.

using Test, Printf
using Aeolus
using Aeolus.Snowpack
using Aeolus.SurfaceExchange: DEFAULT_SURFACE_CONSTANTS
using Aeolus.MoistThermodynamics: saturation_specific_humidity

# Reference-model broadband albedo curves (days since snowfall), used only
# as comparison data — none of them is part of the model.
#  * US Army Corps of Engineers (1956), accumulation season, as coded in
#    VIC 5 (`snow_albedo`, SNOW_NEW_SNOW_ALB = 0.85, ACCUM_A = 0.94, B = 0.58).
usace_accum(d) = 0.85 * 0.94^(d^0.58)
#  * ECMWF (Douville et al. 1995; IFS documentation Part IV): dry snow,
#    linear decrease 0.008 per day from 0.85.
ecmwf_dry(d) = max(0.85 - 0.008d, 0.5)
#  * CLASS (Verseghy 1991) as in Noah-MP (SnowAlbedoClassMod, CLASS_ALB_NEW
#    = 0.84, CLASS_ALB_REF = 0.55, e-folding 100 h).
class_alb(d) = 0.55 + (0.84 - 0.55) * exp(-0.01 * d * 24)
#  * BATS (Dickinson et al. 1993; Yang et al. 1997) as in Noah-MP
#    (SnowAgingBatsMod / SnowAlbedoBatsMod, NoahmpTable.TBL): non-dimensional
#    age at ground temperature T, diffuse VIS/NIR albedos, and their mean.
function bats_alb(d, T)
    r1 = exp(5000 * (1 / 273.16 - 1 / T))
    τ = d * 86400 / 1e6 * (r1 + min(r1^10, 1.0) + 0.3)
    F = τ / (1 + τ)
    return (vis = 0.95 * (1 - 0.2F), nir = 0.65 * (1 - 0.5F))
end

@testset "Snow metamorphism: 30-day albedo decay (#35)" begin
    @testset "30-day albedo decay of a clear-sky pack vs reference-model curves" begin
        # A 50 cm, 150 kg m⁻³ pack under a clear-sky diurnal cycle at −10 °C
        # (SW peak 500 W m⁻², LW 230 W m⁻², RH 80 %, U = 3 m s⁻¹), 30-min steps,
        # no new snow, clean snow and 0.05 ppmw of black carbon.
        function season(soot)
            p = LayeredSnowpack(depth = 0.5, density = 150.0, T = 263.15,
                                metamorphism = Flanner2006Metamorphism(),
                                albedo = SnowOpticsAlbedo(soot_ppmw = soot),
                                bottom = SnowGroundTemperature(272.0))
            αd = [surface_albedo(p)]
            dt = 1800.0
            for n in 1:(30 * 48)
                h = mod(n * dt / 3600, 24)
                Ta = 263.15 + 4 * sin(π * (h - 9) / 12)
                qa = 0.8 * saturation_specific_humidity(Ta, 85000.0, DEFAULT_SURFACE_CONSTANTS.thermo; over = :ice)
                f = SurfaceForcing(sw_down = max(0.0, 500 * sin(π * (h - 6) / 12)), lw_down = 230.0,
                                   T_air = Ta, q_air = qa, U = 3.0, p = 85000.0, z_ref = 2.0,
                                   rain = 0.0, snow = 0.0)
                step_surface!(p, f, dt)
                n % 48 == 0 && push!(αd, surface_albedo(p))
            end
            return αd, p
        end
        αc, pc = season(0.0)
        αs, _ = season(0.05)
        Tg = pc.T[1]
        println("  day   clean  BC0.05  USACE  ECMWF  CLASS  BATS(VIS,NIR)")
        for d in (0, 1, 5, 10, 20, 30)
            b = bats_alb(d, Tg)
            @printf("  %3d   %.3f  %.3f   %.3f  %.3f  %.3f  %.3f, %.3f\n", d, αc[d+1], αs[d+1],
                    usace_accum(d), ecmwf_dry(d), class_alb(d), b.vis, b.nir)
        end
        # Physics: albedo only falls without new snow, and falls more with soot.
        @test all(diff(αc) .<= 1e-12)
        @test all(αs .< αc)
        # Fresh start agrees with the reference fresh values (above).
        @test abs(αc[1] - 0.85) < 0.01
        # From day 5 on, clean-snow grain growth alone darkens *more slowly*
        # than every empirical curve, which are fitted to real snow and so
        # include impurity deposition (BATS's explicit "dirt and soot" term
        # 0.3 is most of its cold-snow ageing). This is a physical bound,
        # stated as one. It does NOT hold on day 1: the fresh 54.5 μm grains
        # coarsen fastest at first (0.851 → 0.835), below ECMWF's linear
        # 0.842, so day 1 is deliberately not asserted.
        for d in (5, 10, 30)
            b = bats_alb(d, Tg)
            @test αc[d+1] > max(usace_accum(d), ecmwf_dry(d), class_alb(d), (b.vis + b.nir) / 2)
        end
        # ... and it is not negligible: ≥ 0.06 in 30 days, from grains alone.
        @test αc[1] - αc[end] > 0.06
        # CLASS/USACE/ECMWF/BATS span 0.55-0.64 at day 30; with 0.05 ppmw BC
        # (mid-latitude seasonal snow) the pack reaches ≤ 0.74.
        @test αs[end] < 0.74
    end
end
