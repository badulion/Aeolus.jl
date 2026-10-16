# Phase 53 — sub-grid surface tiles (issue #41).
#
# A SurfaceMosaic holds N AbstractSurface tiles with area fractions; each
# tile solves its own energy balance through the SurfaceExchange contract
# and the cell sees the fraction-weighted flux (flux aggregation). These
# tests check conservation, the exact identities of the aggregates, the
# single-tile limit, and the size and sign of the difference between flux
# aggregation and parameter aggregation (Avissar & Pielke 1989; Mahrt
# 1987; Mason 1988).

using Test, Printf
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

@testset "Era V — sub-grid surface tiles (#41)" begin

    forcing = SurfaceForcing(sw_down = 600.0, lw_down = 330.0, T_air = 295.0,
                             q_air = 0.008, U = 3.0)

    @testset "Construction and fraction checks" begin
        m = SurfaceMosaic([SlabLand(), SlabLand(T = 300.0)], [0.25, 0.75])
        @test m isa AbstractSurface
        @test ntiles(m) == 2
        @test sum(mosaic_fractions(m)) == 1
        # Fractions within tolerance are renormalised to sum exactly to one.
        m2 = SurfaceMosaic((SlabLand(), SlabLand()), (0.3 + 1e-9, 0.7))
        @test sum(mosaic_fractions(m2)) ≈ 1 atol = 2eps()
        # Renormalisation is exact only to rounding: Σ f of x ./ sum(x) can be
        # off by an ulp, never more than a few.
        worst_sum = maximum(1:2000) do k
            x = [mod(0.6180339887 * k * j, 1.0) + 1e-3 for j in 1:(2 + k % 6)]
            abs(sum(mosaic_fractions(SurfaceMosaic([SlabLand() for _ in x], x ./ sum(x)))) - 1)
        end
        @test worst_sum <= 4eps()
        @test_throws ArgumentError SurfaceMosaic([SlabLand(), SlabLand()], [0.5, 0.6])
        @test_throws ArgumentError SurfaceMosaic([SlabLand(), SlabLand()], [-0.1, 1.1])
        @test_throws DimensionMismatch SurfaceMosaic([SlabLand()], [0.5, 0.5])
        @test_throws ArgumentError SurfaceMosaic([SlabLand(), 1.0], [0.5, 0.5])
        set_mosaic_fractions!(m, [0.6, 0.4])
        @test mosaic_fractions(m) == [0.6, 0.4]
        @test_throws ArgumentError set_mosaic_fractions!(m, [0.6, 0.6])
    end

    @testset "Single-tile identity" begin
        bare = SlabLand(T = 300.0, albedo = 0.25, z0m = 0.05)
        tiled = SurfaceMosaic([SlabLand(T = 300.0, albedo = 0.25, z0m = 0.05)], [1.0])
        @test values_agree(surface_temperature(tiled), surface_temperature(bare))
        @test values_agree(surface_albedo(tiled), surface_albedo(bare))
        # exp(Σ f ln z0) round-trips through log/exp: equal to rounding only.
        @test all(isapprox.(values(roughness_lengths(tiled)), values(roughness_lengths(bare)); rtol = 4eps()))
        @test values_agree(turbulent_fluxes(tiled, forcing), turbulent_fluxes(bare, forcing))
        worst = 0.0
        for n in 1:500
            f = SurfaceForcing(sw_down = max(0.0, 700 * sin(2π * n / 500)), lw_down = 320.0,
                               T_air = 293.0 + 3 * sin(2π * n / 500), q_air = 0.007, U = 4.0)
            a = step_surface!(bare, f, 60.0)
            b = step_surface!(tiled, f, 60.0)
            worst = max(worst, maximum(abs.(_fluxtuple(a) .- _fluxtuple(b))))
        end
        @test worst == 0
        @test tiled.tiles[1].T === bare.T
        @info "Single-tile identity" steps = 500 max_flux_difference = worst T_end = bare.T
    end

    @testset "Identical tiles, any split, equal the single surface" begin
        bare = TileTestBetaSlab(T = 298.0, β = 0.6)
        split = SurfaceMosaic([TileTestBetaSlab(T = 298.0, β = 0.6) for _ in 1:3],
                              [0.13, 0.52, 0.35])
        worst = 0.0
        for n in 1:200
            a = step_surface!(bare, forcing, 120.0)
            b = step_surface!(split, forcing, 120.0)
            worst = max(worst, _maxrel(a, b))
        end
        @test worst < 1e-14
        @test all(t.T == bare.T for t in split.tiles)
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
        dt = 60.0
        for n in 0:1439                    # one diurnal cycle
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
        @test worst_agg < 1e-12
        E1 = sum(f * t.C_s * t.T for (t, f) in zip(tiles, fr))
        rel = abs((E1 - E0) - ∫G) / abs(∫G)
        @test rel < 1e-10
        @test worst_balance < 1e-10
        @info "Tile energy conservation over 24 h" ΔE_tiles = E1 - E0 ∫Gdt = ∫G rel_error = rel max_SEB_residual = worst_balance
    end

    @testset "Radiative aggregates are exact" begin
        a = SlabLand(T = 280.0, albedo = 0.1, emissivity = 1.0)
        b = SlabLand(T = 320.0, albedo = 0.4, emissivity = 1.0)
        m = SurfaceMosaic([a, b], [0.5, 0.5])
        Trad = effective_radiative_temperature(m)
        @test Trad ≈ ((280.0^4 + 320.0^4) / 2)^(1 / 4) rtol = 1e-14
        @test Trad > tile_mean_temperature(m) == 300.0     # T⁴ is convex
        @test surface_temperature(m) == Trad
        # Mixed emissivities: T_rad weights by ε, so ε̄σT_rad⁴ = Σ f ε σ T⁴.
        m2 = SurfaceMosaic([SlabLand(T = 270.0, emissivity = 0.9),
                            SlabLand(T = 310.0, emissivity = 0.98),
                            TileTestBetaSlab(T = 300.0, ε = 0.95)], [0.3, 0.3, 0.4])
        σ = DEFAULT_SURFACE_CONSTANTS.σ_SB
        lw_up = sum(f * surface_emissivity(t) * σ * surface_temperature(t)^4
                    for (t, f) in zip(m2.tiles, mosaic_fractions(m2)))
        @test surface_emissivity(m2) * σ * surface_temperature(m2)^4 ≈ lw_up rtol = 1e-14
        # One-surface formula with the aggregates reproduces Σ f Rn exactly,
        # so a radiation scheme that sees only (α, ε, T_rad) is consistent.
        @test net_radiation(parameter_aggregate(m2; temperature = :radiative), forcing) ≈
              net_radiation(m2, forcing) rtol = 1e-13
        @test net_radiation(m2, forcing) ≈
              sum(f * net_radiation(t, forcing) for (t, f) in zip(m2.tiles, mosaic_fractions(m2))) rtol = 1e-14
        # ... while the mean temperature under-states emission.
        @test net_radiation(parameter_aggregate(m2; temperature = :mean), forcing) >
              net_radiation(m2, forcing)
        # Spectral albedo forwarding to tiles that implement it.
        m3 = SurfaceMosaic([TileTestFixedFlux(290.0, 0.0), TileTestFixedFlux(290.0, 0.0)], [0.5, 0.5])
        @test surface_albedo(m3, 3, 0.5) ≈ 0.3
        # Log-averaged roughness.
        m4 = SurfaceMosaic([SlabLand(z0m = 1e-3, z0h = 1e-4), SlabLand(z0m = 1.0, z0h = 0.1)], [0.5, 0.5])
        @test roughness_lengths(m4).z0m ≈ sqrt(1e-3) rtol = 1e-12
        @test roughness_lengths(m4).z0h ≈ sqrt(1e-5) rtol = 1e-12
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
        @info "Linear limits" H_flux = Hf H_param = Hp
    end

    @testset "Flux vs parameter aggregation: nonlinear cases (Avissar & Pielke)" begin
        # (1) Stability convexity: both tiles unstable (θ_a ≈ 295.1 K). C_H grows
        # with instability, so H(T_s) is convex and Σ f H(T_i) > H(Σ f T_i).
        m = SurfaceMosaic([SlabLand(T = 296.0), SlabLand(T = 306.0)], [0.5, 0.5])
        Hf, Hp = turbulent_fluxes(m, forcing).H, turbulent_fluxes(parameter_aggregate(m), forcing).H
        @test Hf > Hp
        @test (Hf - Hp) / Hf > 0.05
        @info "Unstable/unstable mosaic" H_flux = Hf H_param = Hp rel = (Hf - Hp) / Hf

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
        @info "Stable/unstable mosaic" H_cool = tf[1].H H_warm = tf[2].H H_flux = Hf H_param = Hp

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
        # Effective z0 that reproduces the mosaic stress vs the log average.
        z0_eff = forcing.z_ref * exp(-κ * forcing.U / sqrt(τf / ρ))
        @test z0_eff > sqrt(prod(z0s))
        @info "Roughness mosaic (neutral)" τ_flux = τf τ_param = τp τ_analytic z0_eff z0_logmean = sqrt(prod(z0s))
    end

    @testset "Wet/dry mosaic over a diurnal cycle (Avissar & Pielke 1989)" begin
        # Half the cell open wet soil (β = 1), half dry (β = 0), vs one
        # surface with β̄ = 0.5 and the same heat capacity. Both conserve
        # energy; they partition it differently. The dry tile, with no
        # evaporative cooling, runs hot: more sensible heat and more emitted
        # longwave (lower Rn) than the averaged surface.
        function run_day!(s)
            dt = 60.0
            acc = zeros(4)                 # ∫Rn, ∫H, ∫LE, ∫G
            Tmax = 0.0
            for n in 0:1439
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
        @test Rn_m < Rn_p                              # hot dry tile emits more LW
        @test H_m > H_p + 20                           # measured ≈ 46 vs ≈ 1 W m⁻²
        @test LE_m < LE_p - 20                         # measured ≈ 89 vs ≈ 146 W m⁻²
        @test H_m / LE_m > 10 * (H_p / LE_p)           # Bowen ratio
        @info @sprintf("Wet/dry day means (W m⁻²): mosaic Rn %.1f H %.1f LE %.1f G %.1f | β̄ surface Rn %.1f H %.1f LE %.1f G %.1f",
                       Rn_m, H_m, LE_m, G_m, Rn_p, H_p, LE_p, G_p)
    end

    @testset "Generic tiles: no-kwarg step, zero fraction, per-tile forcing, nesting" begin
        # A tile with prescribed fluxes and a keyword-less step_surface!.
        m = SurfaceMosaic(AbstractSurface[TileTestFixedFlux(290.0, 0.0), SlabLand(T = 300.0)], [0.25, 0.75])
        fl = step_surface!(m, forcing, 10.0)
        tf = last_tile_fluxes(m)
        @test fl.G ≈ 0.25 * 20.0 + 0.75 * tf[2].G
        @test m.tiles[1].stored == 200.0

        # A zero-fraction tile is stepped but carries no weight.
        a, b = SlabLand(T = 300.0), SlabLand(T = 300.0)
        m0 = SurfaceMosaic([a, SlabLand(T = 280.0, albedo = 0.8)], [1.0, 0.0])
        ref = step_surface!(b, forcing, 60.0)
        @test _maxrel(step_surface!(m0, forcing, 60.0), ref) < 1e-15
        @test m0.tiles[2].T != 280.0

        # Per-tile forcing: identical to stepping the tiles one by one.
        f1 = SurfaceForcing(U = 2.0, T_air = 290.0, sw_down = 300.0)
        f2 = SurfaceForcing(U = 6.0, T_air = 292.0, sw_down = 300.0)
        x1, x2 = SlabLand(T = 295.0), SlabLand(T = 288.0)
        mf = SurfaceMosaic([SlabLand(T = 295.0), SlabLand(T = 288.0)], [0.4, 0.6])
        agg = step_surface!(mf, [f1, f2], 60.0)
        r1, r2 = step_surface!(x1, f1, 60.0), step_surface!(x2, f2, 60.0)
        @test agg.H ≈ 0.4 * r1.H + 0.6 * r2.H rtol = 1e-14
        @test mf.tiles[1].T == x1.T && mf.tiles[2].T == x2.T
        @test_throws DimensionMismatch step_surface!(mf, [f1], 60.0)

        # Nesting: a mosaic of mosaics equals the flattened mosaic.
        mk() = (TileTestBetaSlab(T = 290.0, β = 1.0), TileTestBetaSlab(T = 310.0, z0m = 0.5),
                SlabLand(T = 300.0, albedo = 0.3))
        p, q, r = mk()
        nested = SurfaceMosaic(AbstractSurface[SurfaceMosaic(AbstractSurface[p, q], [0.5, 0.5]), r], [0.4, 0.6])
        flat = SurfaceMosaic(AbstractSurface[mk()...], [0.2, 0.2, 0.6])
        worst = 0.0
        for n in 1:100
            worst = max(worst, _maxrel(step_surface!(nested, forcing, 60.0), step_surface!(flat, forcing, 60.0)))
        end
        @test worst < 1e-12
        @test surface_temperature(nested) ≈ surface_temperature(flat) rtol = 1e-14
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
        fr = class_fractions(classmap, 5)
        @test sum(fr) ≈ 1 && all(fr .> 0)
        @test_throws ArgumentError class_fractions([1, 6], 5)
        cells = tiles_from_class_map(classmap, (4, 4), make)
        @test size(cells) == (2, 2)
        @test ntiles(cells[1, 1]) == 1                       # pure block → one tile
        @test all(sum(mosaic_fractions(c)) ≈ 1 for c in cells)
        @test_throws ArgumentError tiles_from_class_map(classmap, (3, 4), make)

        fine = [make(classmap[i, j]) for i in 1:8, j in 1:8]
        worst = 0.0
        for n in 0:359                                      # 6 h with 60 s steps
            f = SurfaceForcing(sw_down = 700.0 * sin(π * n / 720), lw_down = 330.0,
                               T_air = 294.0, q_air = 0.008, U = 2.5)
            ffl = [step_surface!(s, f, 60.0) for s in fine]
            for J in 1:2, I in 1:2
                blk = ffl[4I-3:4I, 4J-3:4J]
                mean_fine = SurfaceFluxes(sum(x.Rn for x in blk) / 16, sum(x.H for x in blk) / 16,
                                          sum(x.LE for x in blk) / 16, sum(x.G for x in blk) / 16,
                                          sum(x.τ for x in blk) / 16, sum(x.E for x in blk) / 16)
                worst = max(worst, _maxrel(step_surface!(cells[I, J], f, 60.0), mean_fine))
            end
        end
        @test worst < 1e-12
        @info "Urban class map: tiled 2×2 vs block-mean of 8×8 per-cell slabs" max_rel_diff = worst
    end

    # Moved from test/berlin_parent_surface.jl (#320); unchanged.
    @testset "Macdonald et al. (1998) roughness of unresolved buildings" begin
        # d/h = 1 + A^(-λp)(λp - 1): at λp = 0.3, A = 4.43 → 1 - 0.7·4.43^(-0.3) = 0.5521
        r = macdonald_roughness(0.3, 0.3, 1.0)
        @test isapprox(r.d, 0.552, atol = 1e-3)
        @test r.d ≈ 1 - 0.7 * 4.43^(-0.3) rtol = 1e-14
        # z0 and d scale with h
        r20 = macdonald_roughness(0.3, 0.2, 20.0)
        r1 = macdonald_roughness(0.3, 0.2, 1.0)
        @test r20.z0 ≈ 20r1.z0 rtol = 1e-14
        @test r20.d ≈ 20r1.d rtol = 1e-14
        # the maximum of z0/h over cube arrays (λf = λp) lies at intermediate
        # density, λf ≈ 0.15-0.2 (Grimmond & Oke 1999, JAM 38, 1262)
        λs = 0.01:0.001:0.99
        z0s = [macdonald_roughness(λ, λ, 1.0).z0 for λ in λs]
        λpk = λs[argmax(z0s)]
        @test 0.15 <= λpk <= 0.2
        @test 0.1 < maximum(z0s) < 0.15          # z0/h ≈ 0.135 at the peak (measured)
        # limits: no buildings → d, z0 → 0; a closed roof → d → h, z0 → 0
        r0 = macdonald_roughness(1e-9, 1e-9, 25.0)
        @test r0.d < 1e-7 && r0.z0 < 1e-12
        @test macdonald_roughness(0.0, 0.0, 25.0) == (z0 = 0.0, d = 0.0)
        rc = macdonald_roughness(1 - 1e-9, 0.5, 25.0)
        @test abs(rc.d - 25.0) < 1e-6 && rc.z0 < 1e-6
        @test macdonald_roughness(1.0, 0.5, 25.0).d == 25.0
        # d increases monotonically with λp (sheltering), 0 ≤ z0 < h - d
        ds = [macdonald_roughness(λ, 0.2, 1.0).d for λ in λs]
        @test all(diff(ds) .> 0)
        @test all(λ -> (local q = macdonald_roughness(λ, λ, 1.0); 0 <= q.z0 < 1 - q.d), λs)
        # square (aligned) arrays, A = 3.59, are less displaced than staggered ones
        @test macdonald_roughness(0.3, 0.3, 1.0; A = 3.59, β = 0.55).d < r.d
        @test_throws ArgumentError macdonald_roughness(1.2, 0.1, 10.0)
        @test_throws ArgumentError macdonald_roughness(0.2, -0.1, 10.0)
    end
end
