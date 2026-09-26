# Phase 50 — BuildingEnergy: indoor air temperature and thermal mass (#31).
#
# Validation is against solutions the code does not contain: the single-node
# RC exponential, the two-node lumped-capacitance eigen-solution, the
# plane-wall first eigenmode β tan β = Bi (Incropera §5.5), the steady
# composite-wall network with radiative gains, exact energy bookkeeping,
# exactness of the reuse of the UrbanSurface columns and facets, and the
# ISO 13790 5R1C model (its own steady network, its Crank–Nicolson decay
# factor, and a side-by-side comparison with the resolved model).

using Test, Printf, LinearAlgebra
using Aeolus
using Aeolus.BuildingEnergy
using Aeolus.UrbanSurface
using Aeolus.SurfaceExchange

const BE = Aeolus.BuildingEnergy
const US = Aeolus.UrbanSurface

# Largest root of β tan β = Bi on (0, π/2) by bisection (test-side reference).
function first_eigenvalue(Bi)
    lo, hi = 0.0, π / 2 - 1e-12
    for _ in 1:200
        mid = (lo + hi) / 2
        mid * tan(mid) < Bi ? (lo = mid) : (hi = mid)
    end
    return (lo + hi) / 2
end

# Least-squares slope of log|y| against t.
function log_slope(t, y)
    ly = log.(abs.(y))
    tm, lm = sum(t) / length(t), sum(ly) / length(ly)
    return sum((t .- tm) .* (ly .- lm)) / sum((t .- tm) .^ 2)
end

# An 8 m × 6 m × 2.7 m single-zone box (the plan dimensions of the ASHRAE 140
# case-600 shoebox, used only as a geometry; no BESTEST result is claimed).
function test_box(; heavy = true, θ = 0.5, T = 293.15, facets = false)
    wl = heavy ? [WallLayer(:brick, 0.24), WallLayer(:gypsum_plaster, 0.015)] :
                 [WallLayer(:plywood, 0.012), WallLayer(:glass_fiber_board, 0.10),
                  WallLayer(:gypsum_plaster, 0.012)]
    A_wall = 2 * (8 + 6) * 2.7 - 12
    walls = facets ?
        FacetElement(UrbanFacet(:wall, wl; T = T, T_interior = T, h_convective = 15.0, θ = θ), A_wall) :
        EnvelopeElement(wl, A_wall; absorptance = 0.6, T = T, θ = θ, name = :walls)
    roof = facets ?
        FacetElement(UrbanFacet(:roof, example_roof_layers(); T = T, T_interior = T, θ = θ), 48.0) :
        EnvelopeElement(example_roof_layers(), 48.0; h_inside = 10.0, T = T, θ = θ, name = :roof)
    floor = EnvelopeElement([WallLayer(:xps_insulation, 0.1), WallLayer(:concrete, 0.15)], 48.0;
                            exterior = :adiabatic, h_inside = 1 / 0.17, T = T, θ = θ, name = :floor)
    im = heavy ? [InternalMass([WallLayer(:brick, 0.12)], 20.0; sides = 2, T = T, θ = θ)] :
                 InternalMass{Float64}[]
    return BuildingZone(; volume = 8 * 6 * 2.7, floor_area = 48.0, envelope = [walls, roof, floor],
                        internal_mass = im, window_area = 12.0, window_U = 2.8,
                        infiltration_ach = 0.5, T_air = T)
end

@testset "Era V — BuildingEnergy: indoor air and thermal mass (#31)" begin

    @testset "Air node and conductances from first principles" begin
        z = BuildingZone(; volume = 100.0, floor_area = 40.0, infiltration_ach = 0.5,
                         ventilation_ach = 1.0, heat_recovery = 0.8, T_air = 293.15)
        # Dry air at 20 °C and 1 atm: ρ = 1.204 kg m⁻³, c_p ≈ 1005 J kg⁻¹ K⁻¹.
        ρc = air_heat_capacity(z) / 100
        @test ρc ≈ 1.2041 * 1004.6 rtol = 1e-3
        # H_ve = ρ c V (n_inf + (1-η) n_vent)/3600 = ρc·100·(0.5 + 0.2)/3600
        @test ventilation_conductance(z) ≈ ρc * 100 * 0.7 / 3600 rtol = 1e-12
        # A bare envelope element with ISO 6946 defaults has the ISO 6946 U-value.
        wl = example_wall_layers()
        el = EnvelopeElement(wl, 10.0)
        @test element_transmittance(el) ≈ thermal_transmittance(wl) rtol = 1e-12
        @test element_transmittance(EnvelopeElement(wl, 10.0; exterior = :adiabatic)) == 0
        # Ground contact: no outside surface resistance.
        @test element_transmittance(EnvelopeElement(wl, 10.0; exterior = :ground)) ≈
              1 / (0.13 + US.thermal_resistance(wl)) rtol = 1e-12
        # Heat-loss coefficient: H_ve + U_w A_w + Σ U A over :air elements only.
        zb = test_box()
        H = ventilation_conductance(zb) + 2.8 * 12 +
            element_transmittance(zb.envelope[1]) * zb.envelope[1].area +
            element_transmittance(zb.envelope[2]) * 48
        @test zone_heat_loss_coefficient(zb) ≈ H rtol = 1e-12
        # Input validation
        @test_throws ArgumentError BuildingZone(; volume = -1.0, floor_area = 1.0)
        @test_throws ArgumentError BuildingZone(; volume = 1.0, floor_area = 1.0, heat_recovery = 1.5)
        @test_throws ArgumentError EnvelopeElement(wl, 1.0; exterior = :space)
        @test_throws ArgumentError InternalMass(wl, 1.0; sides = 3)
        @test_throws ArgumentError FacetElement(UrbanFacet(:ground, example_ground_layers()), 1.0)
        @test_throws ArgumentError step_zone!(zb, ZoneForcing(T_out = 280.0), 0.0)
        zf = test_box(facets = true)
        @test_throws ArgumentError step_zone!(zf, ZoneForcing(T_out = 280.0), 60.0)  # no `surface`
    end

    @testset "Single RC node: exponential with τ = C_a / H_ve" begin
        z = BuildingZone(; volume = 250.0, floor_area = 100.0, infiltration_ach = 2.0, T_air = 293.15)
        C, H = air_heat_capacity(z), ventilation_conductance(z)
        τ = C / H
        @test τ ≈ 1800 rtol = 1e-12          # 1/n for pure air exchange: 2 h⁻¹ → 30 min
        T_out = 283.15
        dt = τ / 400
        ts, dev = Float64[], Float64[]
        disc = 0.0
        for n in 1:1200
            b = step_zone!(z, ZoneForcing(T_out = T_out), dt)
            # the implicit (backward Euler) air update has this discrete solution exactly
            disc = max(disc, abs((b.T_air - T_out) / (10 * (1 + dt / τ)^(-n)) - 1))
            push!(ts, n * dt); push!(dev, b.T_air - T_out)
        end
        @test disc < 1e-10
        # The fitted τ carries the backward-Euler bias ln(1 + dt/τ)·τ/dt ≈ 1 - dt/2τ,
        # i.e. +0.125 % at dt = τ/400.
        τ_fit = -1 / log_slope(ts, dev)
        @printf("  single node: τ analytic %.1f s, fitted %.1f s (dt = τ/400)\n", τ, τ_fit)
        @test τ_fit ≈ τ * (dt / τ) / log(1 + dt / τ) rtol = 1e-9
        @test τ_fit ≈ τ rtol = 2e-3
        # With a constant gain Φ the air settles at T_out + Φ/H.
        for _ in 1:4000
            step_zone!(z, ZoneForcing(T_out = T_out, Q_internal = 500.0), 60.0)
        end
        @test zone_air_temperature(z) ≈ T_out + 500 / H rtol = 1e-10
    end

    @testset "Two-node lumped capacitance: air + thin steel mass (Bi ≈ 6e-4)" begin
        # A 5 mm steel sheet, Bi = h d/k = 7.69·0.005/60.5 ≈ 6e-4: lumped to 0.02 %.
        mass = InternalMass([WallLayer(:steel, 0.005)], 30.0; sides = 1, T = 293.15)
        z = BuildingZone(; volume = 100.0, floor_area = 40.0, internal_mass = [mass],
                         infiltration_ach = 0.5, T_air = 293.15)
        m = thermal_material(:steel)
        Cm = US.volumetric_heat_capacity(m) * 0.005 * 30
        Ca, H, hA = air_heat_capacity(z), ventilation_conductance(z), 30 / 0.13
        M = [-(hA + H)/Ca hA/Ca; hA/Cm -hA/Cm]
        λ = eigvals(M)
        @printf("  two-node: τ_fast = %.0f s, τ_slow = %.0f s\n", -1 / λ[1], -1 / λ[2])
        T_out, dt = 283.15, 5.0
        err = 0.0
        for n in 1:Int(2 * 86400 / dt)
            step_zone!(z, ZoneForcing(T_out = T_out), dt)
            if n % 60 == 0
                x = exp(M * n * dt) * [10.0, 10.0]
                err = max(err, abs(z.T_air - T_out - x[1]))
            end
        end
        @printf("  two-node: max |T_air - analytic| over 2 days = %.2e K (of 10 K)\n", err)
        @test err < 1e-2
    end

    @testset "Distributed mass: first plane-wall eigenmode β tan β = Bi" begin
        # A 20 cm concrete slab exposed on one face (other adiabatic) to a zone
        # whose air node is negligible (C_a = 1e3 J/K vs 4e7 J/K of slab) and
        # which is ventilated to T_out. The air is then quasi-steady and the
        # slab face sees T_out through h_eff = (1/h + A/H_ve)⁻¹: the classical
        # plane wall with convection (Incropera §5.5). Late-time decay rate
        # is κ β₁² / L² with β₁ tan β₁ = h_eff L / k.
        L, A = 0.20, 10.0
        slab = InternalMass([WallLayer(:concrete, L; max_cell = 0.004)], A; sides = 1, T = 293.15)
        z = BuildingZone(; volume = 150.0, floor_area = A, internal_mass = [slab],
                         infiltration_ach = 1.0, air_capacity = 1e3, T_air = 293.15)
        H = ventilation_conductance(z)
        h_eff = 1 / (0.13 + A / H)
        c = thermal_material(:concrete)
        Bi = h_eff * L / c.conductivity
        β = first_eigenvalue(Bi)
        rate = US.thermal_diffusivity(c) * β^2 / L^2
        ts, dev = Float64[], Float64[]
        dt = 120.0
        for n in 1:Int(12 * 86400 / dt)
            step_zone!(z, ZoneForcing(T_out = 283.15), dt)
            n * dt > 6 * 86400 && (push!(ts, n * dt); push!(dev, z.T_air - 283.15))
        end
        fit = -log_slope(ts, dev)
        @printf("  slab mode: Bi = %.3f, β₁ = %.4f, τ₁ analytic %.2f h, fitted %.2f h (%.3f %%)\n",
                Bi, β, 1 / rate / 3600, 1 / fit / 3600, 100 * (fit / rate - 1))
        @test fit ≈ rate rtol = 5e-3
    end

    @testset "Steady state: composite walls, glazing, air exchange, radiative gains" begin
        # θ = 1 (backward Euler) so the stiff modes are damped on long steps.
        z = test_box(θ = 1.0)
        ground = EnvelopeElement([WallLayer(:concrete, 0.15), WallLayer(:xps_insulation, 0.05)], 30.0;
                                 exterior = :ground, T_ground = 285.15, h_inside = 1 / 0.17, θ = 1.0)
        push!(z.envelope, ground)
        z.ventilation_ach = 1.0; z.heat_recovery = 0.75
        T_out, Φ_int, Φ_sol, Φ_hvac = 278.15, 800.0, 300.0, 250.0
        for _ in 1:24 * 200
            step_zone!(z, ZoneForcing(T_out = T_out, Q_internal = Φ_int, Q_solar = Φ_sol,
                                      Q_hvac = Φ_hvac), 3600.0)
        end
        # Independent network solution. A radiative flux q on an inner face with
        # outward resistance R_o (to T_e) splits: h R_o/(1 + h R_o) of it reaches
        # the air; on an adiabatic face or internal mass, all of it.
        A_int = sum(el -> BE._exposed_area(el), Iterators.flatten((z.envelope, z.internal_mass)))
        q = (0.5Φ_int + Φ_sol) / A_int
        num = (2.8 * 12 + ventilation_conductance(z)) * T_out + 0.5Φ_int + Φ_hvac
        den = 2.8 * 12 + ventilation_conductance(z)
        for el in z.envelope
            R = US.thermal_resistance(el.column.layers)
            if el.exterior === :adiabatic
                num += el.area * q
                continue
            end
            R_o = R + (el.exterior === :air ? 1 / el.h_outside : 0.0)
            T_e = el.exterior === :air ? T_out : el.T_ground
            U = 1 / (1 / el.h_inside + R_o)
            num += el.area * (U * T_e + q * el.h_inside * R_o / (1 + el.h_inside * R_o))
            den += el.area * U
        end
        num += sum(el -> BE._exposed_area(el) * q, z.internal_mass)
        T_ss = num / den
        @printf("  steady state: T_air = %.6f K, analytic %.6f K\n", z.T_air, T_ss)
        @test z.T_air ≈ T_ss atol = 1e-8
        # Steady ⇒ nothing stored: the budget balances with no storage term.
        b = step_zone!(z, ZoneForcing(T_out = T_out, Q_internal = Φ_int, Q_solar = Φ_sol,
                                      Q_hvac = Φ_hvac), 3600.0)
        @test abs(b.Q_exterior + b.Q_windows + b.Q_ventilation + Φ_int + Φ_sol + Φ_hvac) < 1e-6
    end

    @testset "Energy conservation: facets, sol-air, ground, internal mass, HVAC" begin
        z = test_box(facets = true)
        push!(z.envelope, EnvelopeElement([WallLayer(:concrete, 0.2)], 25.0; exterior = :ground,
                                          T_ground = 284.0, name = :basement))
        push!(z.envelope, EnvelopeElement(example_wall_layers(), 15.0; absorptance = 0.7, name = :south))
        push!(z.internal_mass, InternalMass([WallLayer(:softwood, 0.03)], 12.0; sides = 1))
        z.ventilation_ach = 0.8; z.heat_recovery = 0.6
        dt = 600.0
        E0 = zone_heat_content(z)
        src = 0.0; gross = 0.0
        max_bal = 0.0
        for n in 1:Int(10 * 86400 / dt)
            t = (n - 0.5) * dt
            day = sin(2π * (t / 86400 - 0.25))
            sf = SurfaceForcing(sw_down = max(0, 800day), lw_down = 320.0,
                                T_air = 288.0 + 6day, U = 3.0)
            f = ZoneForcing(T_out = 288.0 + 6day, Q_internal = 400 + 300 * (day > 0),
                            Q_solar = max(0, 900day), Q_hvac = -150.0 * (z.T_air > 297),
                            irradiance = [0.0, 0.0, 0.0, 0.0, max(0, 500day)], surface = sf)
            b = step_zone!(z, f, dt)
            s = b.Q_exterior + b.Q_windows + b.Q_ventilation + b.Q_internal + b.Q_solar + b.Q_hvac
            src += s * dt
            gross += (abs(b.Q_exterior) + abs(b.Q_windows) + abs(b.Q_ventilation) +
                      b.Q_internal + b.Q_solar + abs(b.Q_hvac)) * dt
            # every facet still satisfies its own surface energy balance G = Rn - H - LE
            for el in z.envelope
                el isa FacetElement || continue
                fl = facet_fluxes(el)
                max_bal = max(max_bal, abs(fl.Rn - fl.H - fl.LE - fl.G))
            end
        end
        err = zone_heat_content(z) - E0 - src
        @printf("  conservation over 10 days: |ΔE - ∫sources| = %.3e J, relative to gross %.2e\n",
                abs(err), abs(err) / gross)
        @test abs(err) / gross < 1e-11
        @test max_bal < 1e-9
        # The air node sees what the budget says: C_a dT = Q_surfaces + windows + vent + convective gains + hvac
        z2 = test_box()
        T0 = z2.T_air
        b = step_zone!(z2, ZoneForcing(T_out = 280.0, Q_internal = 600.0, Q_hvac = 100.0), 300.0)
        @test air_heat_capacity(z2) * (b.T_air - T0) / 300 ≈
              b.Q_surfaces + b.Q_windows + b.Q_ventilation + 0.5 * 600 + 100 rtol = 1e-10
    end

    @testset "Reuse is exact: zone step ≡ stepping the columns at the solved T_air" begin
        z = test_box(facets = true)
        push!(z.envelope, EnvelopeElement(example_wall_layers(), 15.0; absorptance = 0.7))
        sf = SurfaceForcing(sw_down = 600.0, lw_down = 330.0, T_air = 300.0, U = 2.0)
        f = ZoneForcing(T_out = 300.0, Q_internal = 500.0, Q_solar = 400.0,
                        irradiance = 450.0, surface = sf)
        for _ in 1:10
            step_zone!(z, f, 900.0)
        end
        before = deepcopy(z)
        b = step_zone!(z, f, 900.0)
        x = b.T_air
        # facet: step an independent copy with interior at x through UrbanSurface alone
        el0, el = before.envelope[1], z.envelope[1]
        fac = el0.facet
        set_interior_boundary!(fac, ColumnBoundary(x, fac.interior.h, el.q_inside))
        fl = step_surface!(fac, sf, 900.0)
        @test maximum(abs.(fac.column.T .- el.facet.column.T)) < 1e-9
        @test fac.T_skin ≈ el.facet.T_skin atol = 1e-9
        @test fl.H ≈ facet_fluxes(el).H atol = 1e-8
        @test interior_heat_flux(fac) ≈ interior_heat_flux(el.facet) atol = 1e-8
        # bare envelope element: step_conduction! with the same boundaries
        e0, e = before.envelope[end], z.envelope[end]
        Fo, Fi = step_conduction!(e0.column, ColumnBoundary(300.0, e0.h_outside, 0.7 * 450.0),
                                  ColumnBoundary(x, e0.h_inside, e.q_inside), 900.0)
        @test maximum(abs.(e0.column.T .- e.column.T)) < 1e-9
        @test Fo ≈ e.F_outer atol = 1e-8
    end

    @testset "Thermal mass damps and delays the indoor diurnal cycle" begin
        # Same outdoor forcing; heavy (brick + partition) vs light (timber frame) walls.
        res = Dict{Bool, Tuple{Float64, Int}}()
        for heavy in (true, false)
            z = test_box(heavy = heavy)
            Ts = Float64[]
            for n in 1:(20 * 144)
                t = (n - 0.5) * 600
                step_zone!(z, ZoneForcing(T_out = 288.15 + 5sin(2π * t / 86400)), 600.0)
                n > 19 * 144 && push!(Ts, z.T_air)
            end
            res[heavy] = ((maximum(Ts) - minimum(Ts)) / 2, argmax(Ts))
        end
        @printf("  indoor amplitude for ±5 K outdoor: heavy %.3f K (max at step %d), light %.3f K (step %d)\n",
                res[true][1], res[true][2], res[false][1], res[false][2])
        @test res[true][1] < 0.7 * res[false][1]
        @test res[true][1] < 5 && res[false][1] < 5
    end
end
