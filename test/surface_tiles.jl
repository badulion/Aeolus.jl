# Phase 53 — sub-grid surface tiles (issue #41).
#
# A SurfaceMosaic holds N AbstractSurface tiles with area fractions; each
# tile solves its own energy balance through the SurfaceExchange contract
# and the cell sees the fraction-weighted flux (flux aggregation). These
# tests check conservation, the exact identities of the aggregates, the
# single-tile limit, and the size and sign of the difference between flux
# aggregation and parameter aggregation (Avissar & Pielke 1989; Mahrt
# 1987; Mason 1988).

using Test
using Aeolus
using Aeolus.SurfaceTiles
import Aeolus.SurfaceExchange: surface_temperature, roughness_lengths, surface_albedo,
                               surface_emissivity, moisture_availability, step_surface!,
                               psi_m, psi_h, phi_m, phi_h
using Aeolus.MoistThermodynamics: PhasePartition, air_density

# ---- simple test surfaces ------------------------------------------------

# A slab with a bucket-style moisture availability β (SlabLand is dry).
mutable struct TileTestBetaSlab <: AbstractSurface
    T::Float64
    C::Float64
    α::Float64
    ε::Float64
    β::Float64
    z0m::Float64
    z0h::Float64
end
TileTestBetaSlab(; T = 293.0, C = 2e5, α = 0.2, ε = 0.98, β = 0.0, z0m = 0.1, z0h = 0.01) =
    TileTestBetaSlab(T, C, α, ε, β, z0m, z0h)
surface_temperature(s::TileTestBetaSlab) = s.T
roughness_lengths(s::TileTestBetaSlab) = (z0m = s.z0m, z0h = s.z0h, z0q = s.z0h)
surface_albedo(s::TileTestBetaSlab) = s.α
surface_emissivity(s::TileTestBetaSlab) = s.ε
moisture_availability(s::TileTestBetaSlab) = s.β
function step_surface!(s::TileTestBetaSlab, f::SurfaceForcing, dt; kwargs...)
    fl = turbulent_fluxes(s, f; kwargs...)
    s.T += fl.G * dt / s.C
    return fl
end

# A surface whose step_surface! takes no keywords and that prescribes its
# fluxes — checks the framework does not assume turbulent_fluxes / kwargs.
mutable struct TileTestFixedFlux <: AbstractSurface
    T::Float64
    stored::Float64
end
surface_temperature(s::TileTestFixedFlux) = s.T
roughness_lengths(::TileTestFixedFlux) = (z0m = 0.01, z0h = 0.001, z0q = 0.001)
surface_albedo(::TileTestFixedFlux) = 0.3
surface_emissivity(::TileTestFixedFlux) = 0.9
surface_albedo(::TileTestFixedFlux, band::Integer, μ0) = 0.1 * band
function step_surface!(s::TileTestFixedFlux, f::SurfaceForcing, dt)
    fl = SurfaceFluxes(100.0, 30.0, 50.0, 20.0, 0.1, 50.0 / 2.5e6)
    s.stored += fl.G * dt
    return fl
end

# Neutral similarity: ψ ≡ 0, so C_D, C_H do not depend on the skin state.
struct TileTestNeutral <: Aeolus.SurfaceExchange.AbstractStabilityFunctions end
psi_m(::TileTestNeutral, ζ) = zero(ζ)
psi_h(::TileTestNeutral, ζ) = zero(ζ)
phi_m(::TileTestNeutral, ζ) = one(ζ)
phi_h(::TileTestNeutral, ζ) = one(ζ)

_fluxtuple(fl) = (fl.Rn, fl.H, fl.LE, fl.G, fl.τ, fl.E)
_maxrel(a, b) = maximum(abs(x - y) / max(abs(x), abs(y), 1e-300)
                        for (x, y) in zip(_fluxtuple(a), _fluxtuple(b)) if max(abs(x), abs(y)) > 1e-12)

@testset "sub-grid surface tiles (#41)" begin

    forcing = SurfaceForcing(sw_down = 600.0, lw_down = 330.0, T_air = 295.0,
                             q_air = 0.008, U = 3.0)

    @testset "Construction and fraction checks" begin
        m = SurfaceMosaic([SlabLand(), SlabLand(T = 300.0)], [0.25, 0.75])
        @test m isa AbstractSurface && sum(mosaic_fractions(m)) == 1
        # renormalisation is exact only to rounding: never more than a few ulp
        worst_sum = maximum(1:200) do k
            x = [mod(0.6180339887 * k * j, 1.0) + 1e-3 for j in 1:(2 + k % 6)]
            abs(sum(mosaic_fractions(SurfaceMosaic([SlabLand() for _ in x], x ./ sum(x)))) - 1)
        end
        @test worst_sum <= 4eps()
        @test_throws ArgumentError SurfaceMosaic([SlabLand(), SlabLand()], [0.5, 0.6])
    end

    @testset "Single-tile identity" begin
        bare = SlabLand(T = 300.0, albedo = 0.25, z0m = 0.05)
        tiled = SurfaceMosaic([SlabLand(T = 300.0, albedo = 0.25, z0m = 0.05)], [1.0])
        worst = 0.0
        for n in 1:20
            f = SurfaceForcing(sw_down = max(0.0, 700 * sin(2π * n / 20)), lw_down = 320.0,
                               T_air = 293.0 + 3 * sin(2π * n / 20), q_air = 0.007, U = 4.0)
            worst = max(worst, maximum(abs.(_fluxtuple(step_surface!(bare, f, 600.0)) .-
                                            _fluxtuple(step_surface!(tiled, f, 600.0)))))
        end
        @test worst == 0     # one tile of weight 1: the same arithmetic, copied
        @test tiled.tiles[1].T === bare.T
    end

    @testset "Identical tiles, any split, equal the single surface" begin
        bare = TileTestBetaSlab(T = 298.0, β = 0.6)
        split = SurfaceMosaic([TileTestBetaSlab(T = 298.0, β = 0.6) for _ in 1:3],
                              [0.13, 0.52, 0.35])
        worst = 0.0
        for n in 1:20
            a = step_surface!(bare, forcing, 120.0)
            b = step_surface!(split, forcing, 120.0)
            worst = max(worst, _maxrel(a, b))
        end
        @test worst < 1e-14
        @test maximum(abs(t.T - bare.T) for t in split.tiles) < 1e-12
    end

    @testset "Fraction-weighted energy conservation" begin
        tiles = [SlabLand(T = 290.0, C_s = 1.5e5, albedo = 0.12, z0m = 0.5),
                 SlabLand(T = 305.0, C_s = 3.0e5, albedo = 0.35, z0m = 0.01),
                 SlabLand(T = 296.0, C_s = 8.0e5, albedo = 0.08, emissivity = 0.95, z0m = 1e-3)]
        fr = [0.2, 0.5, 0.3]
        m = SurfaceMosaic(tiles, fr)
        E0 = sum(f * t.C_s * t.T for (t, f) in zip(tiles, fr))
        ∫G = 0.0
        worst_balance = worst_agg = 0.0
        dt = 600.0
        for n in 0:143                     # one diurnal cycle
            t = n * dt
            f = SurfaceForcing(sw_down = max(0.0, 800 * sin(π * (t / 3600 - 6) / 12)),
                               lw_down = 330.0, T_air = 293 + 5 * sin(π * (t / 3600 - 9) / 12),
                               q_air = 0.008, U = 3.0)
            fl = step_surface!(m, f, dt)
            ∫G += fl.G * dt
            worst_balance = max(worst_balance, abs(fl.Rn - fl.H - fl.LE - fl.G))
            # The aggregate is exactly the fraction-weighted tile flux.
            tf = last_tile_fluxes(m)
            worst_agg = max(worst_agg, abs(fl.H - sum(fr[i] * tf[i].H for i in 1:3)))
        end
        E1 = sum(f * t.C_s * t.T for (t, f) in zip(tiles, fr))
        @test abs((E1 - E0) - ∫G) / abs(∫G) < 1e-10
        @test max(worst_agg, worst_balance) < 1e-10
    end

    @testset "Radiative aggregates are exact" begin
        # T_rad weights by ε, so ε̄σT_rad⁴ = Σ f ε σ T⁴
        m2 = SurfaceMosaic([SlabLand(T = 270.0, emissivity = 0.9),
                            SlabLand(T = 310.0, emissivity = 0.98),
                            TileTestBetaSlab(T = 300.0, ε = 0.95)], [0.3, 0.3, 0.4])
        σ = DEFAULT_SURFACE_CONSTANTS.σ_SB
        lw_up = sum(f * surface_emissivity(t) * σ * surface_temperature(t)^4
                    for (t, f) in zip(m2.tiles, mosaic_fractions(m2)))
        @test surface_emissivity(m2) * σ * surface_temperature(m2)^4 ≈ lw_up rtol = 1e-14
        # the one-surface formula with the aggregates reproduces Σ f Rn exactly
        @test net_radiation(parameter_aggregate(m2; temperature = :radiative), forcing) ≈
              net_radiation(m2, forcing) rtol = 1e-13
        # log-averaged roughness
        m4 = SurfaceMosaic([SlabLand(z0m = 1e-3, z0h = 1e-4), SlabLand(z0m = 1.0, z0h = 0.1)], [0.5, 0.5])
        @test roughness_lengths(m4).z0m ≈ sqrt(1e-3) rtol = 1e-12
    end

    @testset "Flux vs parameter aggregation: linear limits agree" begin
        # (a) Heterogeneous albedo only (same T, z0): every flux is linear in α.
        m = SurfaceMosaic([SlabLand(T = 300.0, albedo = 0.05), SlabLand(T = 300.0, albedo = 0.6)], [0.3, 0.7])
        @test _maxrel(turbulent_fluxes(m, forcing),
                      turbulent_fluxes(parameter_aggregate(m), forcing)) < 1e-13
        # (b) Heterogeneous temperature under *neutral* similarity: H is linear
        # in T_s, so flux- and parameter-aggregated H agree (with the mean T).
        neutral = TileTestNeutral()
        m = SurfaceMosaic([SlabLand(T = 285.0), SlabLand(T = 310.0)], [0.4, 0.6])
        Hf = turbulent_fluxes(m, forcing; sf = neutral).H
        Hp = turbulent_fluxes(parameter_aggregate(m), forcing; sf = neutral).H
        @test Hf ≈ Hp rtol = 1e-12
    end

    @testset "Flux vs parameter aggregation: nonlinear cases (Avissar & Pielke)" begin
        # (1) Stability convexity: both tiles unstable (θ_a ≈ 295.1 K). C_H grows
        # with instability, so H(T_s) is convex and Σ f H(T_i) > H(Σ f T_i).
        m = SurfaceMosaic([SlabLand(T = 296.0), SlabLand(T = 306.0)], [0.5, 0.5])
        Hf, Hp = turbulent_fluxes(m, forcing).H, turbulent_fluxes(parameter_aggregate(m), forcing).H
        @test (Hf - Hp) / Hf > 0.05

        # (2) Mixed stable/unstable cell: a cool tile below the air and a warm
        # tile above it. The mean skin is *below* the air, so parameter
        # aggregation says the cell is stable (H < 0), while the tiles
        # together transport heat upward: the sign of H is wrong. This is the
        # extreme case of the Richardson-number dependence of the
        # grid-averaged exchange coefficient analysed by Mahrt (1987).
        m = SurfaceMosaic([SlabLand(T = 285.0), SlabLand(T = 299.0)], [0.5, 0.5])
        Hf, Hp = turbulent_fluxes(m, forcing).H, turbulent_fluxes(parameter_aggregate(m), forcing).H
        tf = tile_fluxes(m, forcing)
        @test tf[1].H < 0 < tf[2].H
        @test Hp < 0 < Hf

        # (3) Roughness heterogeneity, neutral: analytic per-tile stress
        # τ_i = ρ (κU / ln(z/z0_i))². The mosaic reproduces the analytic
        # aggregate; the log-averaged z0 underestimates it (1/ln² is convex
        # in ln z0, and the rough tile dominates — Mason 1988).
        neutral = TileTestNeutral()
        z0s, fr = (1e-3, 1.0), (0.5, 0.5)
        m = SurfaceMosaic([SlabLand(T = 295.0, z0m = z0s[1]), SlabLand(T = 295.0, z0m = z0s[2])], collect(fr))
        ρ = air_density(forcing.T_air, forcing.p, PhasePartition(forcing.q_air))
        κ = DEFAULT_SURFACE_CONSTANTS.κ
        τ_analytic = sum(f * ρ * (κ * forcing.U / log(forcing.z_ref / z0))^2 for (f, z0) in zip(fr, z0s))
        τf = turbulent_fluxes(m, forcing; sf = neutral).τ
        τp = turbulent_fluxes(parameter_aggregate(m), forcing; sf = neutral).τ
        @test τf ≈ τ_analytic rtol = 1e-12
        @test τp ≈ ρ * (κ * forcing.U / log(forcing.z_ref / sqrt(prod(z0s))))^2 rtol = 1e-12
        @test τp < τf
    end

    @testset "Wet/dry mosaic over a diurnal cycle (Avissar & Pielke 1989)" begin
        # Half the cell open wet soil (β = 1), half dry (β = 0), vs one
        # surface with β̄ = 0.5 and the same heat capacity. Both conserve
        # energy; they partition it differently. The dry tile, with no
        # evaporative cooling, runs hot: more sensible heat and more emitted
        # longwave (lower Rn) than the averaged surface.
        function run_day!(s)
            dt = 600.0
            acc = zeros(4)                 # ∫Rn, ∫H, ∫LE, ∫G
            Tmax = 0.0
            for n in 0:143
                t = n * dt
                f = SurfaceForcing(sw_down = max(0.0, 800 * sin(π * (t / 3600 - 6) / 12)),
                                   lw_down = 340.0, T_air = 293 + 5 * sin(π * (t / 3600 - 9) / 12),
                                   q_air = 0.008, U = 3.0)
                fl = step_surface!(s, f, dt)
                acc .+= (fl.Rn, fl.H, fl.LE, fl.G) .* dt
                Tmax = max(Tmax, surface_temperature(s))
            end
            return acc ./ 86400, Tmax
        end
        wet, dry = TileTestBetaSlab(β = 1.0), TileTestBetaSlab(β = 0.0)
        mosaic = SurfaceMosaic([wet, dry], [0.5, 0.5])
        single = TileTestBetaSlab(β = 0.5)
        (Rn_m, H_m, LE_m, G_m), _ = run_day!(mosaic)
        (Rn_p, H_p, LE_p, G_p), _ = run_day!(single)
        @test Rn_m ≈ H_m + LE_m + G_m rtol = 1e-10
        @test Rn_p ≈ H_p + LE_p + G_p rtol = 1e-10
        @test Rn_m ≈ H_m + LE_m + G_m rtol = 1e-10
        @test Rn_m < Rn_p                              # hot dry tile emits more LW
        @test H_m > H_p + 20                           # measured ≈ 46 vs ≈ 1 W m⁻² (60 s steps)
        @test LE_m < LE_p - 20                         # measured ≈ 89 vs ≈ 146 W m⁻² (60 s steps)
    end

    @testset "Generic tiles: no-kwarg step, per-tile forcing, nesting" begin
        # a tile with prescribed fluxes and a keyword-less step_surface!
        m = SurfaceMosaic(AbstractSurface[TileTestFixedFlux(290.0, 0.0), SlabLand(T = 300.0)], [0.25, 0.75])
        fl = step_surface!(m, forcing, 10.0)
        @test fl.G ≈ 0.25 * 20.0 + 0.75 * last_tile_fluxes(m)[2].G
        # per-tile forcing: identical to stepping the tiles one by one
        f1 = SurfaceForcing(U = 2.0, T_air = 290.0, sw_down = 300.0)
        f2 = SurfaceForcing(U = 6.0, T_air = 292.0, sw_down = 300.0)
        x1, x2 = SlabLand(T = 295.0), SlabLand(T = 288.0)
        mf = SurfaceMosaic([SlabLand(T = 295.0), SlabLand(T = 288.0)], [0.4, 0.6])
        agg = step_surface!(mf, [f1, f2], 60.0)
        @test agg.H ≈ 0.4 * step_surface!(x1, f1, 60.0).H + 0.6 * step_surface!(x2, f2, 60.0).H rtol = 1e-14
        # nesting: a mosaic of mosaics equals the flattened mosaic
        mk() = (TileTestBetaSlab(T = 290.0, β = 1.0), TileTestBetaSlab(T = 310.0, z0m = 0.5),
                SlabLand(T = 300.0, albedo = 0.3))
        p, q, r = mk()
        nested = SurfaceMosaic(AbstractSurface[SurfaceMosaic(AbstractSurface[p, q], [0.5, 0.5]), r], [0.4, 0.6])
        flat = SurfaceMosaic(AbstractSurface[mk()...], [0.2, 0.2, 0.6])
        worst = 0.0
        for n in 1:10
            worst = max(worst, _maxrel(step_surface!(nested, forcing, 60.0), step_surface!(flat, forcing, 60.0)))
        end
        @test worst < 1e-12
    end

    @testset "Urban class map coarse-grained into tiles" begin
        # The five-class urban map (1 street, 2 roof, 3 park, 4 water, 5 wall)
        # at fine resolution, one class per cell as in examples/urban, versus
        # the same map coarse-grained into 4×4-cell tiled cells. Optical
        # properties come from Radiosity's urban material library (the one
        # source of truth); heat capacities and roughness lengths are
        # illustrative test inputs, not measured values.
        mats = Aeolus.Radiosity.default_urban_materials()
        C = [3e5, 1.8e5, 5e5, 8e6, 4e5]
        z0 = [0.05, 0.1, 0.1, 1e-3, 0.5]
        make(k) = SlabLand(T = 295.0, C_s = C[k], albedo = mats.albedo[k],
                           emissivity = mats.emissivity[k], z0m = z0[k])
        classmap = [UInt8(1 + (3i + 5j + (i * j) % 7) % 5) for i in 1:8, j in 1:8]
        classmap[1:4, 1:4] .= 2                              # one pure roof block
        cells = tiles_from_class_map(classmap, (4, 4), make)
        @test ntiles(cells[1, 1]) == 1                       # pure block → one tile

        fine = [make(classmap[i, j]) for i in 1:8, j in 1:8]
        worst = 0.0
        for n in 0:11                                       # 6 h with 30 min steps
            f = SurfaceForcing(sw_down = 700.0 * sin(π * n / 24), lw_down = 330.0,
                               T_air = 294.0, q_air = 0.008, U = 2.5)
            ffl = [step_surface!(s, f, 1800.0) for s in fine]
            for J in 1:2, I in 1:2
                blk = ffl[4I-3:4I, 4J-3:4J]
                mean_fine = SurfaceFluxes(sum(x.Rn for x in blk) / 16, sum(x.H for x in blk) / 16,
                                          sum(x.LE for x in blk) / 16, sum(x.G for x in blk) / 16,
                                          sum(x.τ for x in blk) / 16, sum(x.E for x in blk) / 16)
                worst = max(worst, _maxrel(step_surface!(cells[I, J], f, 1800.0), mean_fine))
            end
        end
        @test worst < 1e-12
    end

    # Moved from test/berlin_parent_surface.jl (#320); unchanged.
    @testset "Macdonald et al. (1998) roughness of unresolved buildings" begin
        # d/h = 1 + A^(-λp)(λp - 1) at λp = 0.3, A = 4.43
        r = macdonald_roughness(0.3, 0.3, 1.0)
        @test r.d ≈ 1 - 0.7 * 4.43^(-0.3) rtol = 1e-14
        # the maximum of z0/h over cube arrays (λf = λp) lies at λ ≈ 0.15-0.2
        # (Grimmond & Oke 1999, JAM 38, 1262)
        λs = 0.01:0.005:0.99
        @test 0.15 <= λs[argmax([macdonald_roughness(λ, λ, 1.0).z0 for λ in λs])] <= 0.2
        # limits: no buildings → d, z0 → 0; a closed roof → d → h
        @test macdonald_roughness(0.0, 0.0, 25.0) == (z0 = 0.0, d = 0.0)
        @test macdonald_roughness(1.0, 0.5, 25.0).d == 25.0
    end
end

using Aeolus.SurfaceExchange: has_packed_state, packed_state_length, pack_state!, unpack_state!
using Aeolus.Radiation: SlabLand
using Aeolus.SoilColumn

@testset "Packed state of a mosaic: fractions, last fluxes, every tile (#526)" begin
    h = carsel_parrish_texture(:sand)
    p = clm_soil_thermal_properties(20, 5)
    mk() = SurfaceMosaic(Aeolus.SurfaceExchange.AbstractSurface[SlabLand(T = 300.0, C_s = 3e5, β = 0.1),
                             SoilThermalColumn(SoilWaterColumn(clm_exponential_soil_grid(8), h; θ = 0.2), p; T_soil = 290.0)],
                         [0.4, 0.6])
    f = SurfaceForcing(sw_down = 500.0, lw_down = 330.0, T_air = 293.0, q_air = 0.007, U = 2.5)
    a, b = mk(), mk()
    @test has_packed_state(a)
    @test isempty(a.fluxes)
    n = packed_state_length(a)
    @test n == 1 + 7 * 2 + 1 + packed_state_length(a.tiles[2])
    buf0 = zeros(n)
    pack_state!(buf0, 1, a)
    @test buf0[3] == 0.0                          # no fluxes before the first step
    for _ in 1:20; step_surface!(a, f, 60.0); end
    buf = zeros(n)
    @test pack_state!(buf, 1, a) == n + 1
    @test unpack_state!(b, buf, 1) == n + 1
    @test length(b.fluxes) == 2 && b.fluxes[1].H == a.fluxes[1].H   # copied, not computed
    @test b.tiles[1].T == a.tiles[1].T && b.tiles[2].water.T_soil == a.tiles[2].water.T_soil
    for _ in 1:10
        step_surface!(a, f, 60.0); step_surface!(b, f, 60.0)
    end
    @test b.tiles[1].T ≈ a.tiles[1].T rtol = 1e-12
    @test b.tiles[2].H ≈ a.tiles[2].H rtol = 1e-12
end

@testset "last_tile_fluxes: a step's vector is not overwritten by the next step (#537)" begin
    # validation/urban_checks.jl keeps a step's tile fluxes and steps on;
    # writing them in place (tried in #537) silently changed what it held
    m = SurfaceMosaic([SlabLand(; T = 310.0), SlabLand(; T = 290.0)], [0.4, 0.6])
    f = SurfaceForcing(sw_down = 600.0, lw_down = 330.0, T_air = 295.0, q_air = 0.008, U = 3.0)
    fl1 = step_surface!(m, f, 600.0)
    tf1 = last_tile_fluxes(m)
    H1 = [t.H for t in tf1]                     # copied
    step_surface!(m, f, 600.0)
    @test [t.H for t in tf1] == H1              # the held vector is that step's
    @test fl1.H ≈ 0.4 * tf1[1].H + 0.6 * tf1[2].H rtol = 1e-14
    @test last_tile_fluxes(m)[1].H != H1[1]     # the new step's fluxes differ (the slabs moved)
end
