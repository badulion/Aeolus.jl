# Phase 51 (#34) — Snowpack validation: the bulk-density evolution of a
# synthetic 151-day alpine season (Anderson 1976 and Vionnet 2012
# compaction) against the observational climatology of Sturm et al.
# (2010). The material laws, compaction, conduction and conservation are
# unit-tested in test/snowpack.jl.

using Test, Printf
using Aeolus
using Aeolus.Snowpack
using Aeolus.SurfaceExchange: DEFAULT_SURFACE_CONSTANTS

@testset "Era V — Snowpack: density vs Sturm et al. (2010) (#34)" begin
    @testset "Density evolution vs Sturm et al. (2010) alpine climatology" begin
        # Sturm et al. (2010), J. Hydrometeorol. 11, 1380, Eq. 6 and Table 4:
        # ρ(h, DOY) = (ρmax - ρ0)[1 - exp(-k1 h - k2 DOY)] + ρ0, h in cm,
        # ρ in g cm⁻³, DOY from -92 (1 Oct) through 181 (30 Jun), 1 Jan = 1.
        # Alpine class: ρmax = 0.5975, ρ0 = 0.2237, k1 = 0.0012, k2 = 0.0038.
        # A regression on 25 688 observed depth-density-SWE measurements
        # (US, Canada, Switzerland) — the observed seasonal densification.
        # DOY has no 0: 31 Dec = -1, 1 Jan = 1 (the sampled days below
        # straddle this by at most one day).
        sturm(h_cm, doy) = 1000 * ((0.5975 - 0.2237) * (1 - exp(-0.0012h_cm - 0.0038doy)) + 0.2237)
        # Synthetic forcing (NOT observations): 500 mm SWE falling evenly
        # 1 Nov - 31 Mar, air -5 ± 3 °C diurnal, clear-sky-ish longwave,
        # 400 W m⁻² peak sun, ground at 0 °C.
        σ = DEFAULT_SURFACE_CONSTANTS.σ_SB
        results = Dict{String,Vector{Float64}}()
        for (name, comp) in (("Anderson1976", Anderson1976Compaction()),
                             ("Vionnet2012", Vionnet2012Compaction()))
            p = LayeredSnowpack(compaction = comp, bottom = SnowGroundTemperature(273.15))
            rate = 500.0 / (151 * 86400)
            errs = Float64[]
            ρs_model = Float64[]
            for d in 0:150, h in 0:23
                Ta = 268.15 + 3sin(π * (h - 9) / 12)
                sw = 6 <= h <= 18 ? 400sin(π * (h - 6) / 12) : 0.0
                f = SurfaceForcing(sw_down = sw, lw_down = 0.80σ * 268.15^4, T_air = Ta,
                                   q_air = 0.0025, U = 3.0, p = 85000.0, z_ref = 10.0,
                                   rain = 0.0, snow = rate)
                step_surface!(p, f, 3600.0)
                if h == 23 && (d + 1) % 30 == 0
                    doy = -61 + d + 1
                    ρs = sturm(100snow_depth(p), doy)
                    push!(errs, snow_bulk_density(p) / ρs - 1)
                    push!(ρs_model, snow_bulk_density(p))
                    @info @sprintf("%-12s DOY %4d: depth %.2f m, ρ̄ = %.0f kg m⁻³, Sturm alpine %.0f (%+.0f %%)",
                                   name, doy, snow_depth(p), snow_bulk_density(p), ρs, 100errs[end])
                end
            end
            @test p.budget.melt == 0                           # a dry-snow season
            @test snow_water_equivalent(p) ≈ 500.0 - p.budget.sublimation rtol = 1e-10
            results[name] = errs
            # Bulk density rises through the season, as observed, and stays
            # within 25 % of the climatology (Sturm et al. quote a local
            # natural variability of 7-23 % of the mean bulk density).
            @test issorted(ρs_model)
            @test all(abs.(errs) .< 0.25)
        end
    end
end
