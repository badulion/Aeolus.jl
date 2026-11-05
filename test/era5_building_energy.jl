# Phase 50 — BuildingEnergy: indoor air temperature and thermal mass (#31).
#
# Validation is against solutions the code does not contain: the single-node
# RC exponential, the two-node lumped-capacitance eigen-solution, the
# plane-wall first eigenmode β tan β = Bi (Incropera §5.5), the steady
# composite-wall network with radiative gains, exact energy bookkeeping,
# exactness of the reuse of the UrbanSurface columns and facets, and the
# ISO 13790 5R1C model (its own steady network, its Crank–Nicolson decay
# factor, its parameter mapping from a resolved zone). The 150-day
# side-by-side run of the resolved zone and the 5R1C model is a validation
# run: validation/era5_building_energy.jl.

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

isdefined(@__MODULE__, :test_box) || include(joinpath(@__DIR__, "support", "building_box.jl"))

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
        # measured 1.12e-3 K (lumping error at Bi = 6e-4 plus CN time error)
        @test err < 2e-3
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
        @test fit ≈ rate rtol = 5e-4          # measured -9e-5
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
        T_ss = steady_air_temperature(z, T_out, Φ_int, Φ_sol, Φ_hvac)
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
    @testset "ISO 13786 simplified internal areal heat capacity" begin
        ρc(m) = US.volumetric_heat_capacity(thermal_material(m))
        # 20 cm concrete: 10 cm cap (half thickness = 10 cm too)
        @test iso13786_areal_heat_capacity([WallLayer(:concrete, 0.2)]) ≈ ρc(:concrete) * 0.10
        # 8 cm concrete: half thickness wins
        @test iso13786_areal_heat_capacity([WallLayer(:concrete, 0.08)]) ≈ ρc(:concrete) * 0.04
        # brick wall from the room side: 1.5 cm plaster + 8.5 cm brick
        @test iso13786_areal_heat_capacity(example_wall_layers()) ≈
              ρc(:gypsum_plaster) * 0.015 + ρc(:brick) * 0.085
        # insulated roof from the room: plaster + 8.5 cm of the concrete slab
        @test iso13786_areal_heat_capacity(example_roof_layers()) ≈
              ρc(:gypsum_plaster) * 0.015 + ρc(:concrete) * 0.085
        # from the outside the asphalt is stopped by the XPS
        @test iso13786_areal_heat_capacity(example_roof_layers(); side = :outer) ≈ ρc(:asphalt) * 0.01
        # an insulation-faced wall stores nothing on that side
        @test iso13786_areal_heat_capacity([WallLayer(:brick, 0.2), WallLayer(:eps_insulation, 0.1)]) == 0
    end

    @testset "ISO 13790 5R1C: steady network and Crank–Nicolson decay" begin
        iso = ISO13790Zone(; A_f = 50.0, C_m = 165000 * 50.0, A_m = 2.5 * 50, H_ve = 30.0,
                           H_tr_w = 40.0, H_tr_op = 60.0, θ_m = 293.15)
        @test iso.H_tr_is ≈ 3.45 * 4.5 * 50
        @test iso.H_tr_ms ≈ 9.1 * 125
        @test 1 / (1 / iso.H_tr_em + 1 / iso.H_tr_ms) ≈ 60.0     # em + ms in series = H_tr,op
        θ_e, θ_sup, Φi, Φs, Φhc = 273.15, 280.15, 1000.0, 600.0, 400.0
        for _ in 1:24 * 200
            step_iso13790!(iso, θ_e; Φ_int = Φi, Φ_sol = Φs, Φ_HC = Φhc, θ_sup = θ_sup)
        end
        # Independent steady solution of the three-node network (Kirchhoff).
        At, Am, Hw = iso.A_t, iso.A_m, iso.H_tr_w
        Φia = 0.5Φi + Φhc
        Φm = Am / At * (0.5Φi + Φs)
        Φst = (1 - Am / At - Hw / (9.1At)) * (0.5Φi + Φs)
        Hve, His, Hms, Hem = iso.H_ve, iso.H_tr_is, iso.H_tr_ms, iso.H_tr_em
        Mx = [Hve+His  -His           0;
              -His     His+Hw+Hms    -Hms;
              0        -Hms          Hms+Hem]
        rhs = [Hve * θ_sup + Φia, Hw * θ_e + Φst, Hem * θ_e + Φm]
        θa, θs, θm = Mx \ rhs
        @test iso.θ_air ≈ θa atol = 1e-8
        @test iso.θ_s ≈ θs atol = 1e-8
        @test iso.θ_m ≈ θm atol = 1e-8
        # Free decay: θ_m - θ_∞ shrinks by r = (C/Δt - H/2)/(C/Δt + H/2) per step,
        # H = H_tr,3 + H_tr,em, the single time constant of the 5R1C model.
        H1 = 1 / (1 / Hve + 1 / His); H3 = 1 / (1 / (H1 + Hw) + 1 / Hms)
        r = (iso.C_m / 3600 - (H3 + Hem) / 2) / (iso.C_m / 3600 + (H3 + Hem) / 2)
        iso.θ_m = 303.15
        dev = Float64[]
        for _ in 1:5
            step_iso13790!(iso, θ_e; Φ_int = Φi, Φ_sol = Φs, Φ_HC = Φhc, θ_sup = θ_sup)
            push!(dev, iso.θ_m - θm)
        end
        @test all(dev[2:end] ./ dev[1:end-1] .≈ r)
        @printf("  5R1C: time constant C_m/(H_tr,3 + H_tr,em) = %.1f h\n", iso.C_m / (H3 + Hem) / 3600)
        @test_throws ArgumentError ISO13790Zone(; A_f = 1.0, C_m = 1.0, A_m = 1.0, H_ve = 1.0,
                                                H_tr_w = 0.0, H_tr_op = 100.0)
    end

    @testset "Resolved zone vs ISO 13790 5R1C: parameter mapping" begin
        # Parameter mapping by hand for the heavy box.
        z = test_box(heavy = true)
        iso = iso13790_zone(z)
        walls, roof, floor = z.envelope
        κ(l; side = :inner) = iso13786_areal_heat_capacity(l; side = side)
        Cm = walls.area * κ(walls.column.layers) + 48 * κ(roof.column.layers) +
             48 * κ(floor.column.layers) + 2 * 20 * κ(z.internal_mass[1].column.layers)
        @test iso.C_m ≈ Cm rtol = 1e-12
        @test 1 / (1 / iso.H_tr_em + 1 / iso.H_tr_ms) ≈
              element_transmittance(walls) * walls.area + element_transmittance(roof) * 48 rtol = 1e-12
        @test iso.H_ve == ventilation_conductance(z) && iso.H_tr_w ≈ 2.8 * 12
        zg = test_box(); push!(zg.envelope, EnvelopeElement(example_wall_layers(), 1.0; exterior = :ground))
        @test_throws ArgumentError iso13790_zone(zg)
    end
end
