# Phase 49 — UrbanSurface: per-facet multi-layer wall conduction (#26).
#
# Validation is against things the code does not contain: the steady
# composite-wall (ISO 6946) U-value, the Carslaw & Jaeger periodic
# solution for a semi-infinite solid, the ISO 13786 heat-transfer matrix
# for a periodically forced multilayer wall, the separable decaying mode
# of a slab (convergence order), and exact energy bookkeeping.

using Test
using Aeolus
using Aeolus.UrbanSurface
using Aeolus.SurfaceExchange
import Aeolus.SurfaceOptics

const US = Aeolus.UrbanSurface

# ---------------------------------------------------------------------------
# Helpers (test-side reference solutions, independent of the solver)
# ---------------------------------------------------------------------------

# Complex amplitude of the fundamental of a series sampled over whole periods:
# x(t) ≈ mean + Re[X e^{iωt}]  ⇒  X = (2/N) Σ x_j e^{-iωt_j}.
fundamental(x, t, ω) = 2 * sum(x .* exp.(-im .* ω .* t)) / length(x)

# ISO 13786 heat-transfer matrix of a homogeneous layer, derived from
# θ'' = (iω/κ) θ, q = -k θ' (q positive in +z, from outside to inside):
#   [θ(d); q(d)] = [cosh md   -sinh(md)/(k m); -k m sinh md   cosh md] [θ(0); q(0)]
function layer_matrix(l::WallLayer, ω)
    k = l.material.conductivity
    κ = US.thermal_diffusivity(l.material)
    m = sqrt(im * ω / κ)
    d = l.thickness
    return [cosh(m * d) -sinh(m * d) / (k * m); -k * m * sinh(m * d) cosh(m * d)]
end
wall_matrix(layers, ω) = foldl((M, l) -> layer_matrix(l, ω) * M, layers;
                               init = ComplexF64[1 0; 0 1])

@testset "UrbanSurface: multi-layer facet conduction (#26)" begin

    @testset "Thermal material table: cited values, optical links" begin
        @test all(n -> (m = thermal_material(n); m.optical === nothing || m.optical in SurfaceOptics.material_names()),
                  thermal_material_names())
        # Heisler & Brazel (2010) Table 2-3: μ = 1204 J m⁻² K⁻¹ s⁻¹ᐟ² for asphalt
        @test US.thermal_admittance(thermal_material(:asphalt)) ≈ 1204 rtol = 1e-3
        # optics derived from SurfaceOptics, not stored (one source of truth)
        f = UrbanFacet(:wall, example_wall_layers())
        @test f.emissivity == SurfaceOptics.material_emissivity(:brick)
        @test f.albedo ≈ SurfaceOptics.broadband_albedo(SurfaceOptics.material_albedo(:brick))
    end

    @testset "Steady composite wall: U-value (ISO 6946) reproduced by the column" begin
        layers = [WallLayer(:cement_mortar, 0.02), WallLayer(:brick, 0.24),
                  WallLayer(:eps_insulation, 0.08), WallLayer(:gypsum_plaster, 0.015)]
        R = 0.02 / 0.72 + 0.24 / 0.72 + 0.08 / 0.040 + 0.015 / 0.22
        U = thermal_transmittance(layers)
        @test U ≈ 1 / (0.13 + R + 0.04)
        T_out, T_in = 263.15, 293.15
        col = ConductionColumn(layers; T = 280.0, θ = 1.0)
        outer = convective_boundary(T_out, 1 / 0.04)
        inner = convective_boundary(T_in, 1 / 0.13)
        local Fo, Fi
        for _ in 1:400
            Fo, Fi = step_conduction!(col, outer, inner, 86400.0)
        end
        q = U * (T_in - T_out)
        @test -Fo ≈ q rtol = 1e-10
        @test Fi ≈ q rtol = 1e-10
        # exactly piecewise linear: T(z) = Ts_out + q ∫₀ᶻ dz'/k
        Ts_out = face_temperature(col, outer, :outer)
        err = 0.0
        for i in eachindex(col.T)
            z = col.z[i]; acc = 0.0; z0 = 0.0
            for l in layers
                acc += clamp(z - z0, 0, l.thickness) / l.material.conductivity
                z0 += l.thickness
            end
            err = max(err, abs(col.T[i] - (Ts_out + q * acc)))
        end
        @test err < 1e-9
    end

    @testset "Periodic forcing of a composite wall: ISO 13786 heat-transfer matrix" begin
        # outer face T0 + A sin ωt, inner face to room air through h_i = 1/R_si;
        # the matrix solution gives the complex flux amplitude into the room
        P = 86400.0; ω = 2π / P; A = 10.0
        hi = 1 / ISO6946_SURFACE_RESISTANCE.inside_horizontal
        layers = example_wall_layers()
        M = wall_matrix(layers, ω)
        q0 = A * (hi * M[1, 1] - M[2, 1]) / (M[2, 2] - hi * M[1, 2])
        qN = M[2, 1] * A + M[2, 2] * q0
        col = ConductionColumn(layers; T = 293.15, θ = 0.5)
        dt = 600.0; nstep = round(Int, P / dt)
        t = 0.0; ts = Float64[]; Fin = Float64[]
        ndays = 8
        for day in 1:ndays, _ in 1:nstep
            bc = fixed_temperature_boundary(293.15 + A * sin(ω * (t + dt / 2)))
            _, Fi = step_conduction!(col, bc, convective_boundary(293.15, hi), dt)
            day == ndays && (push!(ts, t + dt / 2); push!(Fin, -Fi))
            t += dt
        end
        X = fundamental(Fin, ts, ω)
        Xa = qN * (-im)                   # sin ωt ↔ −i
        lag(X) = mod(-angle(X / (-im)) / ω / 3600, 24)
        # default ≤ 2 cm cells: within 2 % in amplitude, 0.05 h in lag
        # (Δt = 600 s instead of 120 s: CN time error is far below the spatial one)
        @test abs(abs(X) / abs(Xa) - 1) < 0.02
        @test abs(lag(X) - lag(Xa)) < 0.05
        @test lag(Xa) > 5                 # the massive brick wall delays by hours
    end

    @testset "Convergence order: decaying mode of a slab (space O(Δz²), CN O(Δt²), BE O(Δt))" begin
        m = thermal_material(:concrete); κ = US.thermal_diffusivity(m)
        L = 0.2; tend = 0.25 * L^2 / (κ * π^2)
        function run(n, nt, θ)
            col = ConductionColumn([WallLayer(m, L; ncells = n)]; T = 0.0, θ = θ)
            h = L / n
            for i in 1:n
                col.T[i] = (cos(π * (i - 1) * h / L) - cos(π * i * h / L)) * L / (π * h)
            end
            bc = fixed_temperature_boundary(0.0)
            for _ in 1:nt
                step_conduction!(col, bc, bc, tend / nt)
            end
            return col
        end
        exact_avg(col) = [(cos(π * (col.z[i] - col.dz[i] / 2) / L) - cos(π * (col.z[i] + col.dz[i] / 2) / L)) *
                          L / (π * col.dz[i]) * exp(-κ * π^2 * tend / L^2) for i in eachindex(col.T)]
        es = [(col = run(n, 4000, 0.5); maximum(abs.(col.T .- exact_avg(col)))) for n in (20, 40)]
        @test 1.85 < log2(es[1] / es[2]) < 2.15
        ref = run(20, 20000, 0.5).T
        e(nt, θ) = maximum(abs.(run(20, nt, θ).T .- ref))
        @test 1.9 < log2(e(20, 0.5) / e(40, 0.5)) < 2.1
        @test 0.9 < log2(e(20, 1.0) / e(40, 1.0)) < 1.1
    end

    @testset "Energy conservation of the column under arbitrary boundary conditions" begin
        col = ConductionColumn(example_roof_layers(); T = 290.0, θ = 0.5)
        E0 = column_heat_content(col; T_ref = 290.0)
        acc = 0.0
        for j in 1:300
            outer = ColumnBoundary(290 + 20 * sin(j / 37), 5 + 20 * abs(cos(j / 11)), 100 * sin(j / 5))
            inner = j % 3 == 0 ? heat_flux_boundary(-15.0) :
                    j % 3 == 1 ? convective_boundary(295.0, 7.7) : fixed_temperature_boundary(293.0)
            dt = 30.0 + 600 * abs(sin(j))
            Fo, Fi = step_conduction!(col, outer, inner, dt)
            acc += (Fo + Fi) * dt
        end
        @test abs(column_heat_content(col; T_ref = 290.0) - E0 - acc) / sum(abs, col.C .* col.dz) / 20 < 1e-12
    end

    @testset "UrbanFacet: skin on the balance (quadratic), steady state meets the U-value" begin
        # tiny steps act as Newton iterations on Rn(Ts) − H(Ts) = K (Ts − T₁)
        f = UrbanFacet(:roof, example_roof_layers(); T = 300.0)
        @test f isa AbstractSurface
        fo = SurfaceForcing(sw_down = 600.0, lw_down = 350.0, T_air = 295.0, U = 4.0)
        K = f.column.K[1]
        seb_residual(f) = turbulent_fluxes(f, fo).G - K * (surface_temperature(f) - f.column.T[1])
        res = [abs(seb_residual(f))]
        for _ in 1:3
            step_surface!(f, fo, 1e-6)
            push!(res, abs(seb_residual(f)))
        end
        @test res[3] < 1e-4 * res[2]          # quadratic: 0.45 → 5e-6 W/m²
        @test res[4] < 1e-8
        # steady state: Rn(Ts) − h_c (Ts − T_a) = (Ts − T_in)/(R_wall + R_si),
        # found by bisection independently of the solver
        σ = DEFAULT_SURFACE_CONSTANTS.σ_SB
        fo = SurfaceForcing(sw_down = 500.0, lw_down = 320.0, T_air = 300.0, U = 3.0)
        T_in, hc, L = 295.0, 15.0, example_wall_layers()
        f = UrbanFacet(:wall, L; T = 300.0, T_interior = T_in, h_convective = hc, θ = 1.0)
        Ueff = 1 / (thermal_resistance(L) + 1 / f.interior.h)
        local fl
        for _ in 1:3000
            fl = step_surface!(f, fo, 3600.0)
        end
        Fres(Ts) = (1 - f.albedo) * fo.sw_down + f.emissivity * (fo.lw_down - σ * Ts^4) -
                   hc * (Ts - fo.T_air) - Ueff * (Ts - T_in)
        a, b = 250.0, 400.0
        for _ in 1:200
            m = (a + b) / 2; Fres(a) * Fres(m) <= 0 ? (b = m) : (a = m)
        end
        @test surface_temperature(f) ≈ (a + b) / 2 atol = 1e-6
        @test fl.G ≈ interior_heat_flux(f) rtol = 1e-8
    end

    @testset "Diurnal cycle: exact facet energy bookkeeping" begin
        forcing(t) = SurfaceForcing(sw_down = max(0.0, 800 * sin(2π * (t / 86400 - 0.25))),
                                    lw_down = 330.0 + 20 * sin(2π * (t / 86400 - 0.375)),
                                    T_air = 293.0 + 5 * sin(2π * (t / 86400 - 0.375)), U = 3.0)
        f = UrbanFacet(:roof, example_roof_layers(); T = 293.0, θ = 0.5)
        E0 = column_heat_content(f.column; T_ref = 293.0)
        acc, seb, dt = 0.0, 0.0, 600.0
        for j in 0:143
            fl = step_surface!(f, forcing(j * dt + dt / 2), dt)
            acc += (fl.G - interior_heat_flux(f)) * dt
            seb = max(seb, abs(fl.Rn - fl.H - fl.LE - fl.G))
        end
        @test seb < 1e-9
        @test abs(column_heat_content(f.column; T_ref = 293.0) - E0 - acc) <
              1e-9 * sum(abs, f.column.C .* f.column.dz)
    end

    # The TEB canyon closures of the BUBBLE / CAPITOUL canyons (#30)
    @testset "TEB canyon closures" begin
        # Masson (2000) / Lemonsu et al. (2004) eq. B1, hand-evaluated
        ref = 2 / π * exp(-0.25) * log(20 / 3 / 2) / log((10 + 20 / 3) / 2)
        @test canyon_wind_speed(1.0, 10.0, 20.0, 20.0; z0_town = 2.0) ≈ ref rtol = 1e-14
        # Rowley et al. (1930): 11.8 + 4.2 U
        @test rowley_convection_coefficient(2.5) ≈ 22.3
    end
end

# ---------------------------------------------------------------------------
# The steady state of a column (#519): the initial envelope without a
# relaxation transient
# ---------------------------------------------------------------------------

@testset "steady_conduction!: the composite-wall steady state (#519)" begin
    layers = [WallLayer(:brick, 0.24), WallLayer(:gypsum_plaster, 0.015)]
    R = sum(l -> l.thickness / l.material.conductivity, layers)        # independent of the code
    # Dirichlet both sides: F = ΔT / R through every face, profile linear in each layer
    col = ConductionColumn(layers; T = 300.0)
    Fo, Fi = steady_conduction!(col, fixed_temperature_boundary(273.15), fixed_temperature_boundary(293.15))
    @test Fo ≈ (273.15 - 293.15) / R rtol = 1e-12
    @test Fi ≈ -Fo rtol = 1e-12
    # the cell temperatures lie on the analytic piecewise-linear profile
    zc = col.z
    Texact(z) = z <= 0.24 ? 273.15 + 20 * (z / 0.72) / R :
                273.15 + 20 * (0.24 / 0.72 + (z - 0.24) / 0.22) / R
    @test maximum(abs.(col.T .- Texact.(zc))) < 1e-9
    # Robin both sides (outdoor and indoor air): the U-value flux
    h_o, h_i = 25.0, 1 / 0.13
    col = ConductionColumn(layers; T = 300.0)
    Fo, _ = steady_conduction!(col, convective_boundary(270.0, h_o), convective_boundary(294.0, h_i))
    @test Fo ≈ (270.0 - 294.0) / (1 / h_o + R + 1 / h_i) rtol = 1e-12
    # started there, the column does not move under the same conditions,
    # for Crank–Nicolson and backward Euler and any step
    T0 = copy(col.T)
    for (θ, dt) in ((0.5, 60.0), (1.0, 3600.0), (0.5, 86400.0))
        for _ in 1:24
            step_conduction!(col, convective_boundary(270.0, h_o), convective_boundary(294.0, h_i), dt; θ = θ)
        end
        @test maximum(abs.(col.T .- T0)) < 1e-9          # round-off of the tridiagonal solves
    end
    # a uniform start under the same conditions moves by kelvins in the first day
    u = ConductionColumn(layers; T = 294.0)
    for _ in 1:24
        step_conduction!(u, convective_boundary(270.0, h_o), convective_boundary(294.0, h_i), 3600.0)
    end
    @test maximum(abs.(u.T .- 294.0)) > 5.0
    # two flux boundaries have no steady state
    @test_throws ArgumentError steady_conduction!(col, heat_flux_boundary(0.0), heat_flux_boundary(10.0))
end
