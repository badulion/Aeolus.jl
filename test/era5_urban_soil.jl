# Phase 46 (#18): the production soil column in the urban example.
#
# examples/urban/land_surface_addons.jl couples the example's surface slab
# (temperature Ts, areal heat capacity C_s per class) to one
# `SoilThermalColumn` per non-water cell through a contact conductance h.
# These tests check that adapter against physics: exact energy exchange,
# the analytic periodic response of a slab on a semi-infinite conducting
# half-space, insensitivity to the sub-cycling interval, and freeze–thaw
# carried through the exchange.

using Test
using Aeolus
using Printf

module UrbanSoilExample
    include(joinpath(@__DIR__, "..", "examples", "urban", "land_surface_addons.jl"))
end
const USE = UrbanSoilExample

# Periodic forcing of one slab cell: net flux F(t) = F_a cos(ωt) enters the
# slab explicitly (as `step_surfaces!` does), then `step_soil!` exchanges
# with the soil over the same dt. Returns the complex amplitudes of the slab
# and soil-surface temperatures over the last period. Cosine, not sine: a
# sine started at t = 0 injects a net F_a/ω of energy whose diffusion into
# the half-space decays only algebraically and biases the projection.
function _periodic_slab(; h, dt, C_s = 3e5, F_a = 200.0, days = 8, T0 = 288.0,
                        grid = clm_exponential_soil_grid(10))
    sfc = fill(USE.SURFACE_STREET, 1, 1)
    soil = USE.UrbanSoil(sfc; T0 = T0, T_deep = T0, h_contact = h, grid = grid)
    Cs = fill(C_s, 5)
    Ts = fill(T0, 1, 1)
    ω = 2π / 86400
    n_per = round(Int, 86400 / dt)
    ts = zeros(ComplexF64, 1); tg = zeros(ComplexF64, 1)
    for n in 1:days * n_per
        t = (n - 0.5) * dt
        Ts[1] += F_a * cos(ω * t) * dt / C_s
        USE.step_soil!(soil, Ts, sfc, dt, Cs)
        if n > (days - 1) * n_per
            tn = n * dt
            # complex amplitude A of x(t) = Re(A e^{iωt})
            ts[1] += 2 / n_per * (Ts[1] - T0) * exp(-im * ω * tn)
            tg[1] += 2 / n_per * (soil.columns[1].T_skin - T0) * exp(-im * ω * tn)
        end
    end
    return ts[1], tg[1], soil
end

@testset "urban example: production soil column (#18)" begin
    SC = Aeolus.SoilColumn

    @testset "construction: one column per non-water cell" begin
        sfc = UInt8[1 4 2; 3 5 4]
        soil = USE.UrbanSoil(sfc)
        @test length(soil.columns) == count(!=(USE.SURFACE_WATER), sfc) == 4
        @test all(sfc[I] != USE.SURFACE_WATER for I in soil.cells)
        col = soil.columns[1]
        @test length(col.water.grid) == 10
        @test col.water.grid.zh[end] ≈ 3.433093 atol = 1e-6
        @test all(col.water.T_soil .== 288.0)
        # loam at θ = 0.20: the bulk values quoted in the docstring
        p, θs = col.thermal[1], soil_porosity(col.water.hydraulics[1])
        @test soil_thermal_conductivity(p, θs, 0.20, 0.0) ≈ 1.641 atol = 1e-3
        @test soil_heat_capacity(p, θs, 0.20, 0.0) ≈ 2.098e6 rtol = 1e-3
        @test_throws ArgumentError USE.UrbanSoil(sfc; h_contact = 0.0)
    end

    @testset "isothermal slab and soil stay put; water cells untouched" begin
        sfc = UInt8[1 4; 2 3]
        soil = USE.UrbanSoil(sfc; T0 = 288.0, T_deep = 288.0)
        Ts = fill(288.0, 2, 2); Ts[1, 2] = 295.0          # water cell, not soil
        params = USE.default_surface_params()
        for _ in 1:100
            USE.step_soil!(soil, Ts, sfc, 30.0, params)
        end
        @test Ts[1, 2] == 295.0
        @test maximum(abs.(Ts[[1, 2, 4]] .- 288.0)) < 1e-10
        @test maximum(abs.(soil.G)) < 1e-8
    end

    @testset "energy: slab loss = soil gain, to round-off" begin
        sfc = UInt8[1 2 3 5]
        params = USE.default_surface_params()
        soil = USE.UrbanSoil(sfc; T0 = 285.0, T_deep = 285.0)
        Ts = [315.0 300.0 275.0 330.0]                   # hot/cold slabs over 285 K soil
        E0 = USE.urban_soil_energy(soil, Ts, params, sfc)
        bottom0 = [c.budget.bottom for c in soil.columns]
        for n in 1:2880                                  # 1 day at 30 s
            USE.step_soil!(soil, Ts, sfc, 30.0, params)
        end
        E1 = USE.urban_soil_energy(soil, Ts, params, sfc)
        out = sum(c.budget.bottom for c in soil.columns) - sum(bottom0)
        resid = E1 - E0 + out
        scale = sum(params.C_s[sfc[I]] * 40 for I in soil.cells)   # ~energy moved
        @info @sprintf("  slab+soil energy residual after 1 d: %.2e J/m² (moved ~%.1e, base %.2e)",
                       resid, scale, out)
        @test abs(resid) < 1e-9 * scale
        # the slabs relax toward the soil: the hot street has cooled, the cold park warmed
        @test Ts[1] < 300 && Ts[3] > 278
    end

    @testset "periodic forcing: slab on a conducting half-space (analytic)" begin
        # Slab C_s dT_s/dt = F - G, G = h (T_s - T_g), soil surface admittance
        # G = Y T_g with Y = sqrt(iωCk) = k(1+i)/d, d = sqrt(2k/(Cω)):
        #     T̂_s = F̂ / (iωC_s + Y_e),   Y_e = 1/(1/h + 1/Y),   T̂_g = h T̂_s/(h + Y).
        C_s, F_a = 3e5, 200.0
        ω = 2π / 86400
        # The CLM 10-layer grid (top node 7 mm, d = 0.147 m) under-resolves
        # the soil-surface temperature by 1.6 %; the slab temperature, which
        # the atmosphere sees, is within 0.2 %. A uniform 80-layer grid
        # shows the adapter itself converges (0.12 %, 0.006 rad; 0.07 %,
        # 0.0016 rad on 160 layers).
        # With the stiffer contact h = 50 the slab inherits more of that
        # grid error (0.56 %).
        cases = ((6.0, 30.0, clm_exponential_soil_grid(10), 0.003, 0.02, 0.01),
                 (6.0, 300.0, clm_exponential_soil_grid(10), 0.003, 0.02, 0.01),
                 (50.0, 30.0, clm_exponential_soil_grid(10), 0.008, 0.02, 0.01),
                 (6.0, 30.0, uniform_soil_grid(2.0, 80), 0.001, 0.003, 0.008))
        for (h, dt, grid, rtol_s, rtol_g, φtol_g) in cases
            ts, tg, soil = _periodic_slab(; h = h, dt = dt, C_s = C_s, F_a = F_a, grid = grid)
            col = soil.columns[1]
            p, θs = col.thermal[1], soil_porosity(col.water.hydraulics[1])
            k = soil_thermal_conductivity(p, θs, col.water.θ_liq[1], 0.0)
            C = soil_heat_capacity(p, θs, col.water.θ_liq[1], 0.0)
            Y = sqrt(im * ω * C * k)
            Ye = 1 / (1 / h + 1 / Y)
            ts_ex = F_a / (im * ω * C_s + Ye)
            tg_ex = h * ts_ex / (h + Y)
            @info @sprintf("  h = %4.0f, dt = %3.0f s, %2d layers: |T_s| %.4f K (exact %.4f, phase err %+.4f rad); |T_g| %.4f K (exact %.4f, phase err %+.4f rad)",
                           h, dt, length(grid), abs(ts), abs(ts_ex), angle(ts / ts_ex),
                           abs(tg), abs(tg_ex), angle(tg / tg_ex))
            @test abs(ts) ≈ abs(ts_ex) rtol = rtol_s
            @test abs(angle(ts / ts_ex)) < 0.01
            @test abs(tg) ≈ abs(tg_ex) rtol = rtol_g
            @test abs(angle(tg / tg_ex)) < φtol_g
        end
    end

    @testset "freezing under a cold slab: latent heat, energy still exact" begin
        sfc = fill(USE.SURFACE_PARK, 1, 1)
        params = USE.default_surface_params()
        wet = USE.UrbanSoil(sfc; T0 = 276.0, T_deep = 276.0, θ = 0.30)
        dry = USE.UrbanSoil(sfc; T0 = 276.0, T_deep = 276.0, θ = 0.08)
        for soil in (wet, dry)
            Ts = fill(263.0, 1, 1)
            E0 = USE.urban_soil_energy(soil, Ts, params, sfc)
            b0 = soil.columns[1].budget.bottom
            Fin = 0.0
            for n in 1:2 * 2880                          # 2 days at 30 s, slab held at 263 K
                Fin += params.C_s[sfc[1]] * (263.0 - Ts[1])   # energy removed by holding the slab
                Ts[1] = 263.0
                USE.step_soil!(soil, Ts, sfc, 30.0, params)
            end
            E1 = USE.urban_soil_energy(soil, Ts, params, sfc)
            @test abs(E1 - E0 - Fin + (soil.columns[1].budget.bottom - b0)) < 1e-9 * abs(Fin)
        end
        cw, cd = wet.columns[1], dry.columns[1]
        @info @sprintf("  after 2 d under a 263 K slab: ice %.2f kg/m² (wet) %.3f (dry); T at 6 cm %.2f K (wet) vs %.2f K (dry)",
                       soil_ice_mass(cw), soil_ice_mass(cd), cw.water.T_soil[3], cd.water.T_soil[3])
        @test soil_ice_mass(cw) > 5.0                    # a frozen top layer of wet soil
        @test cw.water.θ_ice[1] > 0.2
        # latent heat holds the wet soil near the freezing point while the
        # (almost) water-free soil cools well below it
        @test cw.water.T_soil[3] > cd.water.T_soil[3] + 1.0
        @test cw.water.T_soil[3] > 271.0
    end
end
