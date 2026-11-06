# Phase 46 (#17) — SoilColumn: prognostic water table in an
# unconfined aquifer with SIMTOP baseflow (Niu et al. 2005, 2007; CLM4).
#
# Checked against closed-form results: the Gardner (1958) steady profile
# above a water table, hydrostatic equilibrium, the hyperbolic SIMTOP
# recession, the analytic equilibrium water table under steady recharge,
# CLM4's impermeable-fraction formula, and exact water conservation of
# soil + aquifer.

using Test
using Aeolus
using Aeolus.SoilColumn

const SCW = Aeolus.SoilColumn

# Independent reference integrator for dW/dt = R - q_drai(W) (classical RK4).
function _rk4_aquifer(aq, col, W, R, t, n)
    h = t / n
    Q(W) = aq.q_max * exp(-aq.f * SCW._aq_depth(aq, col, W))
    for _ in 1:n
        k1 = R - Q(W); k2 = R - Q(W + h / 2 * k1)
        k3 = R - Q(W + h / 2 * k2); k4 = R - Q(W + h * k3)
        W += h / 6 * (k1 + 2k2 + 2k3 + k4)
    end
    return W
end

@testset "water table and unconfined aquifer (#17)" begin

@testset "Gardner two-point flux" begin
    # Gardner (1958): steady flux through an exponential soil above a table
    # at distance L gives ψ₁ = ln(r + (1 − r) e^{−αL})/α, r = q/K_s
    Ks, α = 1e-5, 2.0
    K(ψ) = Ks * exp(α * min(ψ, 0.0))
    worst = 0.0
    for L in (0.1, 5.0), r in (-0.99, 0.0, 0.9)
        r < 0 && (r *= exp(-α * L) / (1 - exp(-α * L)))     # up to Gardner's maximum rise
        ψ1 = log(r + (1 - r) * exp(-α * L)) / α
        worst = max(worst, abs(gardner_two_point_flux(K(ψ1), Ks, ψ1, 0.0, L) - r * Ks) / Ks)
    end
    @test worst < 1e-10
    # dry case where a mean conductivity is 1100× too large
    @test gardner_two_point_flux(K(-10.0), Ks, -10.0, 0.0, 5.0) ≈
          -Ks * (exp(-10) - exp(-20)) / (1 - exp(-10)) rtol = 1e-12
    # hydrostatic pair carries no flux, whatever the soil
    h = carsel_parrish_texture(:loam)
    @test abs(gardner_two_point_flux(soil_hydraulic_conductivity(h, -4.0), soil_hydraulic_conductivity(h, 0.0),
                                     -4.0, 0.0, 4.0)) < 1e-12 * h.K_s
end

@testset "SIMTOP recession: exact in time" begin
    v = carsel_parrish_texture(:loam)
    g = uniform_soil_grid(2.0, 10)
    mk(z) = (c = SoilWaterColumn(g, v; bottom = UnconfinedAquifer()); set_water_table!(c, z); c)
    # zero recharge, table below the column: z = z₀ + ln(1 + t/τ)/f, τ = S_y/(f q₀)
    c1 = mk(3.0); c2 = mk(3.0)
    aq1, aq2 = c1.bottom, c2.bottom
    q0 = aquifer_baseflow_rate(c1)
    τ = aq1.S_y / (aq1.f * q0)
    t = 200 * 86400.0
    SCW.finalize_bottom!(aq1, c1, 0.0, t)
    for _ in 1:20
        SCW.finalize_bottom!(aq2, c2, 0.0, t / 20)
    end
    zex = 3.0 + log(1 + t / τ) / aq1.f
    @test water_table_depth(c1) ≈ zex rtol = 1e-12
    @test water_table_depth(c2) ≈ zex rtol = 1e-12        # step-size independent
    @test aq1.baseflow ≈ aq1.S_y * log(1 + t / τ) / aq1.f rtol = 1e-10
    # constant recharge R: against an independent RK4 of dW/dt = R − q(W);
    # the table relaxes to z = ln(q_max/R)/f
    R = 2e-8
    c3 = mk(3.0)
    W0 = c3.bottom.W
    SCW.finalize_bottom!(c3.bottom, c3, R, 400 * 86400.0)
    @test c3.bottom.W ≈ _rk4_aquifer(c3.bottom, c3, W0, R, 400 * 86400.0, 40000) rtol = 1e-9
    SCW.finalize_bottom!(c3.bottom, c3, R, 1e5 * 86400.0)
    @test water_table_depth(c3) ≈ log(5.5e-6 / R) / 2.5 rtol = 1e-10
end

@testset "Hydrostatic equilibrium above the water table" begin
    # closed system: relaxes to ψ(z) = z − z_∇, soil + aquifer water conserved
    g = clm5_soil_grid()
    hyd = [i <= 8 ? clapp_hornberger_texture(:loam) : clapp_hornberger_texture(:sandy_loam) for i in 1:20]
    col = SoilWaterColumn(g, hyd; θ = 0.40, bottom = UnconfinedAquifer(W = 5.0, q_max = 0.0))
    S0 = total_water_storage(col)
    for _ in 1:146
        step_soil_water!(col, 5 * 86400.0)
    end
    zwt = water_table_depth(col)
    @test maximum(abs.(col.ψ .- (col.grid.z .- zwt))) < 1e-6
    @test total_water_storage(col) ≈ S0 atol = 1e-12
    @test zwt < g.zh[end]
end

@testset "Steady recharge: equilibrium table and Gardner profile" begin
    # baseflow = P at steady state: z_∇ = ln(q_max/P)/f, and the unsaturated
    # profile is Gardner's ψ(z) = ln(r + (1 − r) e^{−α(z_∇ − z)})/α
    Ks, α, P = 1e-5, 1.5, 2e-8
    soil = GardnerExponential(θ_r = 0.05, θ_s = 0.40, α = α, K_s = Ks)
    zeq = log(5.5e-6 / P) / 2.5
    r = P / Ks
    ψex(z, zwt) = log(r + (1 - r) * exp(-α * (zwt - z))) / α
    errs = Float64[]; zerr = 0.0
    for n in (20, 40)
        col = SoilWaterColumn(uniform_soil_grid(2.0, n), soil; bottom = UnconfinedAquifer(),
                              temperature_dependent = false)
        set_water_table!(col, zeq - 0.4)
        for _ in 1:3000
            step_soil_water!(col, 86400.0; water_input = P)
        end
        zwt = water_table_depth(col)
        zerr = max(zerr, abs(zwt / zeq - 1))
        push!(errs, maximum(abs.(col.ψ .- ψex.(col.grid.z, zwt))))
    end
    @test zerr < 1e-6
    @test errs[2] < 1e-3
    @test errs[1] / errs[2] > 3                                     # ≈ second order
end

@testset "Closed water budget: soil + aquifer" begin
    # a year of storms, evaporation and root extraction: the table rises into
    # the column and falls back out; soil + aquifer change by exactly
    # infiltration − extraction − baseflow
    v = carsel_parrish_texture(:silt_loam)
    col = SoilWaterColumn(clm5_soil_grid(), v; bottom = UnconfinedAquifer())
    set_water_table!(col, 9.0)
    S0 = total_water_storage(col)
    sink = zeros(20); sink[3:8] .= 2e-9
    zmin, zmax = Inf, -Inf
    for day in 1:365
        rain = (day % 7 == 0 || 100 <= day <= 130) ? 5e-7 : 0.0
        step_soil_water!(col, 86400.0; water_input = rain,
                         evaporation = rain > 0 ? 0.0 : 3e-8, sink = day > 200 ? sink : nothing)
        zwt = water_table_depth(col)
        zmin = min(zmin, zwt); zmax = max(zmax, zwt)
    end
    b, aq = col.budget, col.bottom
    @test abs(total_water_storage(col) - S0 - (b.infiltration - b.extraction - aq.baseflow)) < 1e-10
    @test aq.recharge ≈ b.drainage rtol = 1e-12
    @test zmin < col.grid.zh[end] < zmax
end

@testset "Frozen-soil restriction (CLM4 eq. 7.160)" begin
    v = carsel_parrish_texture(:loam)
    col = SoilWaterColumn(uniform_soil_grid(2.0, 10), v; bottom = UnconfinedAquifer())
    set_water_table!(col, 3.0)
    aq = col.bottom
    # half the bottom layer's water as ice: f_imp = (e^{−1.5} − e^{−3})/(1 − e^{−3})
    col.θ_ice[10] = col.θ_liq[10] * col.constants.ρ_water / col.constants.ρ_ice
    @test aquifer_impermeable_fraction(aq, col) ≈ (exp(-1.5) - exp(-3)) / (1 - exp(-3)) rtol = 1e-12
    col.θ_liq[10] = 0.0
    @test aquifer_baseflow_rate(col) ≈ 0 atol = 1e-20
end

end # water-table testset
