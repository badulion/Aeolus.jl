# Era V, Phase 46 (#17) — SoilColumn: prognostic water table in an
# unconfined aquifer with SIMTOP baseflow (Niu et al. 2005, 2007; CLM4).
#
# Checked against closed-form results: the Gardner (1958) steady profile
# above a water table, hydrostatic equilibrium, the hyperbolic SIMTOP
# recession, the analytic equilibrium water table under steady recharge,
# CLM4's impermeable-fraction formula, and exact water conservation of
# soil + aquifer.

using Test, Printf
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

@testset "Era V — water table and unconfined aquifer (#17)" begin

@testset "Gardner two-point flux" begin
    # Gardner (1958): steady flux q through an exponential soil above a water
    # table (ψ = 0) at distance L gives ψ₁ = ln(r + (1 - r) e^{-αL})/α, r = q/K_s.
    Ks, α = 1e-5, 2.0
    K(ψ) = Ks * exp(α * min(ψ, 0.0))
    # (upward flux is bounded: r > -e^{-αL}/(1 - e^{-αL}), Gardner's maximum
    # capillary rise; the upward cases use half and 99 % of it)
    for L in (0.1, 1.0, 5.0), r in (-0.5, -0.99, 0.0, 0.3, 0.9)
        r < 0 && (r *= exp(-α * L) / (1 - exp(-α * L)))
        ψ1 = log(r + (1 - r) * exp(-α * L)) / α
        @test gardner_two_point_flux(K(ψ1), Ks, ψ1, 0.0, L) ≈ r * Ks rtol = 1e-10 atol = 1e-20
    end
    # the dry case where a mean conductivity fails: ψ₁ = -10 m, 5 m above
    # the table; exact capillary rise is K_s e^{-10}, the logarithmic mean
    # conductivity would give K_s/20 — 1100 times too much.
    q = gardner_two_point_flux(K(-10.0), Ks, -10.0, 0.0, 5.0)
    @test q ≈ -Ks * (exp(-10) - exp(-20)) / (1 - exp(-10)) rtol = 1e-12
    @printf("  capillary rise from a table 5 m down, ψ = -10 m: exact %.3e, log-mean Darcy %.3e m/s\n",
            q, -(Ks - K(-10.0)) / 20)
    # uniform K: Darcy's law
    @test gardner_two_point_flux(Ks, Ks, 0.3, 0.0, 1.0) ≈ Ks * 1.3
    # hydrostatic pair carries no flux, whatever the soil
    for h in (carsel_parrish_texture(:loam), clapp_hornberger_texture(:clay),
              carsel_parrish_texture(:sand; ψ_e = 0.0))
        for (ψ2, L) in ((0.0, 0.7), (0.0, 4.0), (0.5, 1.2))
            ψ1 = ψ2 - L
            K1, K2 = soil_hydraulic_conductivity(h, ψ1), soil_hydraulic_conductivity(h, ψ2)
            @test abs(gardner_two_point_flux(K1, K2, ψ1, ψ2, L)) < 1e-12 * K2
        end
    end
    # monotone: wetter upper point, more downward flux
    qs = [gardner_two_point_flux(K(ψ), Ks, ψ, 0.0, 1.0) for ψ in -3:0.25:0.5]
    @test all(diff(qs) .> 0)
end

@testset "SIMTOP recession: exact in time" begin
    v = carsel_parrish_texture(:loam)
    g = uniform_soil_grid(2.0, 10)
    mk(z) = (c = SoilWaterColumn(g, v; bottom = UnconfinedAquifer()); set_water_table!(c, z); c)
    # Zero recharge, table below the column: z(t) = z₀ + ln(1 + t/τ)/f,
    # q(t) = q₀/(1 + t/τ), τ = S_y/(f q₀) (hyperbolic recession).
    c1 = mk(3.0); c2 = mk(3.0)
    aq1, aq2 = c1.bottom, c2.bottom
    q0 = aquifer_baseflow_rate(c1)
    τ = aq1.S_y / (aq1.f * q0)
    @test q0 ≈ 5.5e-6 * exp(-2.5 * 3.0)
    t = 200 * 86400.0
    SCW.finalize_bottom!(aq1, c1, 0.0, t)
    for _ in 1:2000
        SCW.finalize_bottom!(aq2, c2, 0.0, t / 2000)
    end
    zex = 3.0 + log(1 + t / τ) / aq1.f
    @printf("  recession: τ = %.1f d; after 200 d table at %.6f m (exact %.6f), q/q₀ = %.5f (exact %.5f)\n",
            τ / 86400, water_table_depth(c1), zex, aquifer_baseflow_rate(c1) / q0, 1 / (1 + t / τ))
    @test water_table_depth(c1) ≈ zex rtol = 1e-12
    @test water_table_depth(c2) ≈ zex rtol = 1e-12        # step-size independent
    @test aquifer_baseflow_rate(c1) ≈ q0 / (1 + t / τ) rtol = 1e-10
    @test aq1.baseflow ≈ aq1.S_y * log(1 + t / τ) / aq1.f rtol = 1e-10
    # Constant recharge R: against an independent RK4 integration of
    # dW/dt = R - q_drai(W); the table relaxes to z = ln(q_max/R)/f.
    R = 2e-8
    c3 = mk(3.0)
    W0 = c3.bottom.W
    SCW.finalize_bottom!(c3.bottom, c3, R, 400 * 86400.0)
    Wrk = _rk4_aquifer(c3.bottom, c3, W0, R, 400 * 86400.0, 40000)
    @test c3.bottom.W ≈ Wrk rtol = 1e-9
    SCW.finalize_bottom!(c3.bottom, c3, R, 1e5 * 86400.0)
    @test water_table_depth(c3) ≈ log(5.5e-6 / R) / 2.5 rtol = 1e-10
    @test aquifer_baseflow_rate(c3) ≈ R rtol = 1e-10
    # Table inside the column (aquifer full, elastic): baseflow empties the
    # elastic store within minutes, the table crosses the base, and the
    # piecewise-exact path matches RK4 through the kink.
    c4 = mk(1.0)
    W0 = c4.bottom.W
    @test W0 > c4.bottom.S_y * c4.bottom.D
    tt = 30 * 86400.0
    R = 1e-8                                   # equilibrium table at 2.52 m
    SCW.finalize_bottom!(c4.bottom, c4, R, tt)
    Wrk = _rk4_aquifer(c4.bottom, c4, W0, R, tt, 2_000_000)
    @test 2.0 < water_table_depth(c4) < log(5.5e-6 / R) / 2.5
    @test c4.bottom.W ≈ Wrk rtol = 1e-7
    @test c4.bottom.baseflow ≈ R * tt - (c4.bottom.W - W0) rtol = 1e-12
end

@testset "Hydrostatic equilibrium above the water table" begin
    # Closed system (no baseflow, no surface flux): whatever the start, the
    # column relaxes to ψ(z) = z - z_∇ with the water table where soil +
    # aquifer water is conserved.
    g = clm5_soil_grid()
    hyd = [i <= 8 ? clapp_hornberger_texture(:loam) : clapp_hornberger_texture(:sandy_loam) for i in 1:20]
    # (the dry loam top layers relax slowly: the error falls ~20× per year,
    # so the dry case is run for 6 years)
    for (θ0, W0, yrs, label) in ((0.25, 4.8, 6, "table below the column"),
                                 (0.40, 5.0, 2, "table inside the column"))
        col = SoilWaterColumn(g, hyd; θ = θ0, bottom = UnconfinedAquifer(W = W0, q_max = 0.0))
        S0 = total_water_storage(col)
        for _ in 1:73yrs
            step_soil_water!(col, 5 * 86400.0)
        end
        zwt = water_table_depth(col)
        err = maximum(abs.(col.ψ .- (col.grid.z .- zwt)))
        @printf("  hydrostatic, %s: z_∇ = %.4f m, max |ψ - (z - z_∇)| = %.1e m, Δ(soil+aquifer) = %.1e m\n",
                label, zwt, err, total_water_storage(col) - S0)
        @test err < 1e-6
        @test total_water_storage(col) ≈ S0 atol = 1e-12
        label == "table inside the column" ? @test(zwt < g.zh[end]) : @test(zwt > g.zh[end])
    end
    # a column set to a water table is already in equilibrium and stays put
    col = SoilWaterColumn(g, hyd; bottom = UnconfinedAquifer(q_max = 0.0))
    for zwt in (1.3, 9.0)
        set_water_table!(col, zwt)
        @test water_table_depth(col) ≈ zwt
        ψ0 = copy(col.ψ)
        r = step_soil_water!(col, 30 * 86400.0)
        @test maximum(abs.(col.ψ .- ψ0)) < 1e-9
        @test abs(r.drainage) < 1e-9          # round-off: < 4e-16 m/s
    end
    # diagnosing the table from the column itself (non-aquifer bottoms)
    c = SoilWaterColumn(g, hyd; bottom = ZeroFluxBottom())
    set_water_table!(c, 2.1)
    @test water_table_depth(c) ≈ 2.1
    set_matric_potential!(c, -1.0)
    @test water_table_depth(c) === nothing
end

@testset "Steady recharge: equilibrium table and Gardner profile" begin
    # Rain P on a Gardner soil over the aquifer. At steady state baseflow = P,
    # so z_∇ = ln(q_max/P)/f, and the unsaturated profile is Gardner's (1958)
    # ψ(z) = ln(r + (1 - r) e^{-α(z_∇ - z)})/α, r = P/K_s.
    Ks, α, P = 1e-5, 1.5, 2e-8
    soil = GardnerExponential(θ_r = 0.05, θ_s = 0.40, α = α, K_s = Ks)
    zeq = log(5.5e-6 / P) / 2.5                                   # 2.247 m
    r = P / Ks
    ψex(z, zwt) = log(r + (1 - r) * exp(-α * (zwt - z))) / α
    errs, zwts = Float64[], Float64[]
    for n in (20, 40, 80)
        col = SoilWaterColumn(uniform_soil_grid(2.0, n), soil; bottom = UnconfinedAquifer(),
                              temperature_dependent = false)
        set_water_table!(col, zeq - 0.4)
        for _ in 1:3000
            step_soil_water!(col, 86400.0; water_input = P)
        end
        zwt = water_table_depth(col)
        push!(zwts, zwt)
        @test zwt ≈ zeq rtol = 1e-6
        @test aquifer_baseflow_rate(col) ≈ P rtol = 1e-5
        push!(errs, maximum(abs.(col.ψ .- ψex.(col.grid.z, zwt))))
    end
    @printf("  steady table %.6f m (exact %.6f); Gardner-profile error %.2e / %.2e / %.2e m (20/40/80 layers)\n",
            zwts[3], zeq, errs...)
    @test errs[3] < 1e-3
    @test errs[2] / errs[3] > 3                                     # ≈ second order
    # Table inside a deeper column: the saturated zone carries P with
    # gradient 1 - r below the table, the unsaturated zone is Gardner's.
    col = SoilWaterColumn(uniform_soil_grid(4.0, 80), soil; bottom = UnconfinedAquifer(),
                          temperature_dependent = false)
    set_water_table!(col, zeq + 0.3)
    for _ in 1:3000
        step_soil_water!(col, 86400.0; water_input = P)
    end
    @test water_table_depth(col) ≈ zeq rtol = 1e-6
    zc = SCW._column_water_table(col)            # the column's own ψ = 0 crossing
    @test zc ≈ zeq atol = 0.02
    sat = col.grid.z .> zc
    @test maximum(abs.(col.ψ[sat] .- (1 - r) .* (col.grid.z[sat] .- zc))) < 1e-3
    @test maximum(abs.(col.ψ[.!sat] .- ψex.(col.grid.z[.!sat], zc))) < 1e-3
end

@testset "Closed water budget: soil + aquifer" begin
    # A year of storms, evaporation and root extraction on the CLM5 column:
    # the table rises into the column and falls back out; soil + aquifer
    # water changes by exactly infiltration - extraction - baseflow.
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
    res = total_water_storage(col) - S0 - (b.infiltration - b.extraction - aq.baseflow)
    @printf("  1 yr CLM5 silt loam: infiltration %.3f, runoff %.3f, recharge %.3f, baseflow %.3f m; table %.2f–%.2f m; residual %.1e m\n",
            b.infiltration, b.runoff, aq.recharge, aq.baseflow, zmin, zmax, res)
    @test abs(res) < 1e-10
    @test aq.recharge ≈ b.drainage rtol = 1e-12
    @test zmin < col.grid.zh[end] < zmax                   # crossed the base both ways
    @test aq.baseflow > 0
end

@testset "Frozen-soil restriction (CLM4 eq. 7.160)" begin
    v = carsel_parrish_texture(:loam)
    col = SoilWaterColumn(uniform_soil_grid(2.0, 10), v; bottom = UnconfinedAquifer())
    set_water_table!(col, 3.0)
    aq = col.bottom
    q_free = aquifer_baseflow_rate(col)
    @test aquifer_impermeable_fraction(aq, col) == 0
    # table below the column: only the bottom layer counts. Half its water
    # mass as ice (F = 1/2) gives f_imp = (e^{-1.5} - e^{-3})/(1 - e^{-3}).
    ρi, ρw = col.constants.ρ_ice, col.constants.ρ_water
    col.θ_ice[10] = col.θ_liq[10] * ρw / ρi
    @test aquifer_impermeable_fraction(aq, col) ≈ (exp(-1.5) - exp(-3)) / (1 - exp(-3)) rtol = 1e-12
    @test aquifer_baseflow_rate(col) ≈ q_free * (1 - 0.182426) rtol = 1e-5
    col.θ_liq[10] = 0.0                              # all ice: impermeable
    @test aquifer_impermeable_fraction(aq, col) ≈ 1
    @test aquifer_baseflow_rate(col) ≈ 0 atol = 1e-20
    aq.frozen_restriction = false
    @test aquifer_baseflow_rate(col) ≈ q_free
end

end # water-table testset
