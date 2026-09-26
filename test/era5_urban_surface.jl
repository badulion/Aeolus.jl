# Phase 49 — UrbanSurface: per-facet multi-layer wall conduction (#26).
#
# Validation is against things the code does not contain: the steady
# composite-wall (ISO 6946) U-value, the Carslaw & Jaeger periodic
# solution for a semi-infinite solid, the ISO 13786 heat-transfer matrix
# for a periodically forced multilayer wall, the separable decaying mode
# of a slab (convergence order), and exact energy bookkeeping.

using Test, Printf
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

@testset "Era V — UrbanSurface: multi-layer facet conduction (#26)" begin

    @testset "Thermal material table: cited values, derived quantities, optical links" begin
        names = thermal_material_names()
        @test length(names) == length(unique(names)) == 16
        for n in names
            m = thermal_material(n)
            @test m.conductivity > 0 && m.density > 0 && m.specific_heat > 0
            @test !isempty(m.source)
            # The optical link must name a real SurfaceOptics entry (one source of truth).
            m.optical === nothing || @test m.optical in SurfaceOptics.material_names()
        end
        # Derived quantities are the textbook combinations.
        c = thermal_material(:concrete)
        @test US.volumetric_heat_capacity(c) == 2300.0 * 880.0
        @test US.thermal_diffusivity(c) ≈ 1.4 / (2300 * 880)
        # Admittance check against a *published* column: Heisler & Brazel (2010)
        # Table 2-3 lists μ = 1204 J m⁻² K⁻¹ s⁻¹ᐟ² for asphalt, computed there from
        # the same k, ρ, c. Reproducing it checks the row was transcribed right.
        μa = US.thermal_admittance(thermal_material(:asphalt))
        @test μa ≈ 1204 rtol = 1e-3
        # Physical ordering: insulation conducts ~50x less than masonry,
        # steel ~40x more than concrete; insulation stores almost nothing.
        @test thermal_material(:eps_insulation).conductivity < thermal_material(:brick).conductivity / 15
        @test thermal_material(:steel).conductivity > 40 * thermal_material(:concrete).conductivity
        @test US.volumetric_heat_capacity(thermal_material(:xps_insulation)) <
              US.volumetric_heat_capacity(thermal_material(:concrete)) / 25
        @test_throws ArgumentError thermal_material(:unobtainium)
        @test_throws ArgumentError ThermalMaterial(:x, 1.0, 1.0, 1.0; optical = :not_a_material)
        @info "Thermal materials" μ_asphalt = μa μ_published = 1204 μ_concrete = US.thermal_admittance(c)

        # Optics are derived from SurfaceOptics, not stored: a facet whose outer
        # layer is brick has exactly the SurfaceOptics brick emissivity and the
        # white-sky broadband albedo of the brick spectral curve.
        f = UrbanFacet(:wall, example_wall_layers())
        @test f.emissivity == SurfaceOptics.material_emissivity(:brick)
        @test f.albedo ≈ SurfaceOptics.broadband_albedo(SurfaceOptics.material_albedo(:brick))
        @test surface_albedo(f) == f.albedo && surface_emissivity(f) == f.emissivity
        # Metal roof: the steel layer inherits the ε = 0.25 of `:metal_roof`.
        fm = UrbanFacet(:roof, [WallLayer(:steel, 0.001), WallLayer(:glass_fiber_board, 0.1)])
        @test fm.emissivity == SurfaceOptics.material_emissivity(:metal_roof)
        # A non-exposed outer material must be given optics explicitly.
        @test_throws ArgumentError UrbanFacet(:wall, [WallLayer(:eps_insulation, 0.1)])
        @test UrbanFacet(:wall, [WallLayer(:eps_insulation, 0.1)]; albedo = 0.3, emissivity = 0.9).albedo == 0.3
    end

    @testset "Steady composite wall: U-value (ISO 6946) reproduced by the column" begin
        # Render + brick + EPS + plaster, deliberately with very different k.
        layers = [WallLayer(:cement_mortar, 0.02), WallLayer(:brick, 0.24),
                  WallLayer(:eps_insulation, 0.08), WallLayer(:gypsum_plaster, 0.015)]
        R = 0.02 / 0.72 + 0.24 / 0.72 + 0.08 / 0.040 + 0.015 / 0.22
        @test thermal_resistance(layers) ≈ R
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
        q = U * (T_in - T_out)            # W m⁻², inside → outside
        @test -Fo ≈ q rtol = 1e-10        # leaves through the outer face
        @test Fi ≈ q rtol = 1e-10         # enters through the inner face
        # Face and interface temperatures follow the series-resistance drops.
        Ts_out = face_temperature(col, outer, :outer)
        Ts_in = face_temperature(col, inner, :inner)
        @test Ts_out ≈ T_out + q * 0.04 rtol = 1e-12
        @test Ts_in ≈ T_in - q * 0.13 rtol = 1e-12
        # Temperature is exactly piecewise linear: every cell centre lies on the
        # analytic profile T(z) = Ts_out + q ∫₀ᶻ dz'/k.
        err = 0.0
        for i in eachindex(col.T)
            z = col.z[i]; acc = 0.0; z0 = 0.0
            for l in layers
                dzl = clamp(z - z0, 0, l.thickness)
                acc += dzl / l.material.conductivity
                z0 += l.thickness
            end
            err = max(err, abs(col.T[i] - (Ts_out + q * acc)))
        end
        @test err < 1e-9
        # The 8 cm of insulation carries three quarters of the temperature drop
        # (R_eps / R_total = 2.0 / 2.60 = 0.769), read off the numerical profile:
        # the temperature difference between the first and last EPS cell centres.
        ieps = findall(==(3), col.layer)
        ΔT_eps = (col.T[ieps[1]] - col.T[ieps[end]]) * 0.08 / (col.z[ieps[end]] - col.z[ieps[1]])
        @test ΔT_eps / (T_out - T_in) ≈ 2.0 * U rtol = 1e-10
        @test ΔT_eps / (T_out - T_in) > 0.75
        @info @sprintf("Composite wall: U = %.4f W/m²K, flux %.4f (analytic %.4f), max profile error %.1e K, EPS carries %.1f%% of ΔT",
                       U, Fi, q, err, 100ΔT_eps / (T_out - T_in))
        # Resolution-independent: one cell per layer gives the same flux.
        col1 = ConductionColumn([WallLayer(l.material, l.thickness; ncells = 1) for l in layers];
                                T = 280.0, θ = 1.0)
        for _ in 1:400
            Fo, Fi = step_conduction!(col1, outer, inner, 86400.0)
        end
        @test Fi ≈ q rtol = 1e-10
    end

    @testset "Periodic surface temperature: Carslaw & Jaeger semi-infinite solution" begin
        # T(0,t) = T0 + A sin ωt on a thick concrete block. Analytic:
        # T(z,t) = T0 + A e^{-z/d} sin(ωt - z/d), d = √(2κ/ω), and the surface
        # flux into the solid is A μ √ω sin(ωt + π/4), μ = √(kρc).
        m = thermal_material(:concrete)
        P = 86400.0; ω = 2π / P
        κ = US.thermal_diffusivity(m); μ = US.thermal_admittance(m)
        d = sqrt(2κ / ω)
        col = ConductionColumn([WallLayer(m, 1.5; ncells = 300)]; T = 290.0, θ = 0.5)
        A, T0 = 10.0, 290.0
        dt = 300.0
        nstep = round(Int, P / dt)
        iz = [argmin(abs.(col.z .- zz)) for zz in (d, 2d)]
        ts, Fs = Float64[], Float64[]
        Tz = [Float64[] for _ in iz]
        t = 0.0
        for day in 1:6, _ in 1:nstep
            # Crank–Nicolson: the boundary value at the half step.
            bc = fixed_temperature_boundary(T0 + A * sin(ω * (t + dt / 2)))
            Fo, _ = step_conduction!(col, bc, heat_flux_boundary(0.0), dt)
            if day == 6
                push!(ts, t + dt / 2); push!(Fs, Fo)
                for (j, i) in enumerate(iz)
                    push!(Tz[j], col.T[i])  # at t + dt
                end
            end
            t += dt
        end
        tend = ts .+ dt / 2
        # Surface flux: amplitude and 45° phase lead.
        X = fundamental(Fs, ts, ω)            # Re[X e^{iωt}], sin ωt ↔ -i
        Xa = A * μ * sqrt(ω) * exp(im * π / 4) * (-im)
        amp_err = abs(abs(X) / abs(Xa) - 1)
        ph_err = abs(angle(X / Xa))
        @test amp_err < 0.01
        @test ph_err < 0.01
        # Damping depth: amplitude e^{-z/d} and phase lag z/d at z ≈ d, 2d.
        errs = Float64[]
        for (j, i) in enumerate(iz)
            z = col.z[i]
            Y = fundamental(Tz[j], tend, ω)
            Ya = A * exp(-z / d) * exp(-im * z / d) * (-im)
            push!(errs, abs(abs(Y) / abs(Ya) - 1), abs(angle(Y / Ya)))
            @test abs(Y) / abs(Ya) ≈ 1 atol = 0.01
            @test abs(angle(Y / Ya)) < 0.01
        end
        @info @sprintf("Semi-infinite periodic (concrete, d = %.3f m): surface-flux amplitude err %.2e, phase err %.2e rad; depth d, 2d amp/phase errs %s",
                       d, amp_err, ph_err, string(round.(errs; sigdigits = 2)))
    end

    @testset "Periodic forcing of a composite wall: ISO 13786 heat-transfer matrix" begin
        # Outer face T = T0 + A sin ωt, inner face to room air at constant T_in
        # through h_i = 1/R_si. The matrix solution gives the complex amplitude
        # of the flux into the room (the "periodic thermal transmittance" times
        # A); the numerical column must reproduce its amplitude and time lag.
        P = 86400.0; ω = 2π / P; A = 10.0
        hi = 1 / ISO6946_SURFACE_RESISTANCE.inside_horizontal
        cases = (
            ("brick wall", example_wall_layers()),
            ("insulated brick", [WallLayer(:brick, 0.24), WallLayer(:eps_insulation, 0.10),
                                 WallLayer(:gypsum_plaster, 0.015)]),
            ("concrete roof slab", [WallLayer(:concrete, 0.20)]),
        )
        results = []
        for (label, layers) in cases
            M = wall_matrix(layers, ω)
            # Unknown q0 (flux entering at the outer face), θ0 = A (as a complex
            # amplitude). Inner: q_N = h_i θ_N (room air amplitude 0).
            # [θN; qN] = M [A; q0]  ⇒  M21 A + M22 q0 = h_i (M11 A + M12 q0)
            q0 = A * (hi * M[1, 1] - M[2, 1]) / (M[2, 2] - hi * M[1, 2])
            qN = M[2, 1] * A + M[2, 2] * q0
            function periodic_run(max_cell)
                lz = [WallLayer(l.material, l.thickness; max_cell = max_cell) for l in layers]
                col = ConductionColumn(lz; T = 293.15, θ = 0.5)
                dt = 120.0; nstep = round(Int, P / dt)
                t = 0.0; ts = Float64[]; Fin = Float64[]
                ndays = 12
                for day in 1:ndays, _ in 1:nstep
                    bc = fixed_temperature_boundary(293.15 + A * sin(ω * (t + dt / 2)))
                    _, Fi = step_conduction!(col, bc, convective_boundary(293.15, hi), dt)
                    if day == ndays
                        push!(ts, t + dt / 2); push!(Fin, -Fi)   # flux into the room
                    end
                    t += dt
                end
                return fundamental(Fin, ts, ω), sum(Fin) / length(Fin)
            end
            Xa = qN * (-im)                   # sin ωt ↔ -i
            lag(X) = mod(-angle(X / (-im)) / ω / 3600, 24)   # hours behind the forcing
            X2, mean2 = periodic_run(0.02)    # default resolution (≤ 2 cm cells)
            X05, _ = periodic_run(0.005)      # 5 mm cells
            e2 = abs(abs(X2) / abs(Xa) - 1)
            e05 = abs(abs(X05) / abs(Xa) - 1)
            # Default resolution: within 2 % in amplitude, 0.05 h in lag; the
            # error is spatial truncation and falls ~16x for 4x finer cells.
            @test e2 < 0.02
            @test e05 < 0.002
            @test e2 / e05 > 10
            @test abs(lag(X2) - lag(Xa)) < 0.05
            @test abs(lag(X05) - lag(Xa)) < 0.005
            # Steady part: the mean flux is zero (mean ΔT = 0).
            @test abs(mean2) < 1e-3 * abs(Xa)
            push!(results, (label, abs(Xa), abs(X2), abs(X05), lag(Xa), lag(X2), lag(X05)))
        end
        # Physics: insulation cuts the transmitted amplitude; the massive brick
        # wall delays it by several hours.
        @test results[2][2] < results[1][2] / 2
        @test results[1][5] > 5
        for r in results
            @info @sprintf("ISO 13786 %-18s |q_in| analytic %.4f, numeric (2 cm) %.4f, (5 mm) %.4f W/m²; lag analytic %.3f h, numeric %.3f h, %.3f h",
                           r...)
        end
    end

    @testset "Convergence order: decaying mode of a slab (space O(Δz²), CN O(Δt²), BE O(Δt))" begin
        # Homogeneous slab 0 < z < L, T = 0 on both faces, T(z,0) = sin(πz/L):
        # T = e^{-κπ²t/L²} sin(πz/L).
        m = thermal_material(:concrete); κ = US.thermal_diffusivity(m)
        L = 0.2; tend = 0.25 * L^2 / (κ * π^2)   # one quarter e-folding
        function run(n, nt, θ)
            col = ConductionColumn([WallLayer(m, L; ncells = n)]; T = 0.0, θ = θ)
            # initialise with the exact cell averages
            h = L / n
            for i in 1:n
                col.T[i] = (cos(π * (i - 1) * h / L) - cos(π * i * h / L)) * L / (π * h)
            end
            bc = fixed_temperature_boundary(0.0)
            dt = tend / nt
            for _ in 1:nt
                step_conduction!(col, bc, bc, dt)
            end
            return col
        end
        exact_avg(col) = [(cos(π * (col.z[i] - col.dz[i] / 2) / L) - cos(π * (col.z[i] + col.dz[i] / 2) / L)) *
                          L / (π * col.dz[i]) * exp(-κ * π^2 * tend / L^2) for i in eachindex(col.T)]
        # Space: fine in time (CN, many steps), refine Δz.
        es = Float64[]
        for n in (10, 20, 40, 80)
            col = run(n, 4000, 0.5)
            push!(es, maximum(abs.(col.T .- exact_avg(col))))
        end
        ps = log2.(es[1:end-1] ./ es[2:end])
        @test all(1.85 .< ps .< 2.15)
        # Time: fixed grid, compare against a very fine time step on the same grid
        # (isolates the temporal error of the θ-scheme).
        n = 40
        ref = run(n, 20000, 0.5).T
        et_cn = [maximum(abs.(run(n, nt, 0.5).T .- ref)) for nt in (10, 20, 40, 80)]
        et_be = [maximum(abs.(run(n, nt, 1.0).T .- ref)) for nt in (10, 20, 40, 80)]
        p_cn = log2.(et_cn[1:end-1] ./ et_cn[2:end])
        p_be = log2.(et_be[1:end-1] ./ et_be[2:end])
        @test all(1.9 .< p_cn .< 2.1)
        @test all(0.9 .< p_be .< 1.1)
        @info "Convergence orders: space (CN) $(round.(ps; digits = 3)), time CN $(round.(p_cn; digits = 3)), time BE $(round.(p_be; digits = 3)); space errors $(round.(es; sigdigits = 3))"
    end

    @testset "Energy conservation of the column under arbitrary boundary conditions" begin
        layers = example_roof_layers()
        for θ in (0.5, 1.0)
            col = ConductionColumn(layers; T = 290.0, θ = θ)
            E0 = column_heat_content(col; T_ref = 290.0)
            acc = 0.0
            for j in 1:2000
                outer = ColumnBoundary(290 + 20 * sin(j / 37), 5 + 20 * abs(cos(j / 11)), 100 * sin(j / 5))
                inner = j % 3 == 0 ? heat_flux_boundary(-15.0) :
                        j % 3 == 1 ? convective_boundary(295.0, 7.7) : fixed_temperature_boundary(293.0)
                dt = 30.0 + 600 * abs(sin(j))
                Fo, Fi = step_conduction!(col, outer, inner, dt)
                acc += (Fo + Fi) * dt
            end
            E1 = column_heat_content(col; T_ref = 290.0)
            rel = abs(E1 - E0 - acc) / sum(abs, col.C .* col.dz) / 20
            @test rel < 1e-12
            @info @sprintf("Column conservation θ = %.1f: |ΔE - ΣF dt| = %.2e J/m² over ΣF dt = %.3e J/m²",
                           θ, abs(E1 - E0 - acc), acc)
        end
        # Adiabatic both sides: heat content invariant, temperatures relax to the
        # heat-capacity-weighted mean.
        col = ConductionColumn(example_wall_layers(); T = collect(range(280, 300; length = 13)))
        E0 = column_heat_content(col)
        Tbar = E0 / sum(col.C .* col.dz)
        for _ in 1:2000
            step_conduction!(col, heat_flux_boundary(0.0), heat_flux_boundary(0.0), 3600.0)
        end
        @test column_heat_content(col) ≈ E0 rtol = 1e-14
        @test maximum(abs.(col.T .- Tbar)) < 1e-8
        @test_throws ArgumentError ConductionColumn(layers; θ = 1.5)
        @test_throws ArgumentError ColumnBoundary(290.0, -1.0, 0.0)
    end
    @testset "UrbanFacet implements AbstractSurface" begin
        for (kind, L) in ((:roof, example_roof_layers()), (:wall, example_wall_layers()),
                          (:ground, example_ground_layers()))
            f = UrbanFacet(kind, L; T = 291.0)
            @test f isa AbstractSurface
            @test surface_temperature(f) == 291.0
            @test roughness_lengths(f) == (z0m = 0.01, z0h = 0.001, z0q = 0.001)
            @test moisture_availability(f) == 0.0
            @test length(facet_layer_temperatures(f)) == length(f.column)
        end
        # Default interior hooks: ISO 6946 R_si for roof / wall, deep-ground Dirichlet.
        @test UrbanFacet(:roof, example_roof_layers()).interior.h ≈ 10.0
        @test UrbanFacet(:wall, example_wall_layers()).interior.h ≈ 1 / 0.13
        @test isinf(UrbanFacet(:ground, example_ground_layers(); T_interior = 285.0).interior.h)
        @test UrbanFacet(:ground, example_ground_layers(); T_interior = 285.0).interior.T_env == 285.0
        @test_throws ArgumentError UrbanFacet(:window, example_wall_layers())
        f = UrbanFacet(:wall, example_wall_layers())
        set_interior_temperature!(f, 300.0)
        @test f.interior.T_env == 300.0 && f.interior.h ≈ 1 / 0.13
        set_interior_boundary!(f, heat_flux_boundary(0.0))
        @test f.interior.h == 0.0
        # The skin is the massless outer face: its temperature is set by the
        # energy balance against the half-cell conductance K = 2k/Δz, not a
        # prognostic variable. Tiny steps therefore act as Newton iterations on
        # Rn(Ts) - H(Ts) = K (Ts - T₁): an arbitrary initial skin is pulled onto
        # the balance, the residual falling quadratically.
        f = UrbanFacet(:roof, example_roof_layers(); T = 300.0)
        fo = SurfaceForcing(sw_down = 600.0, lw_down = 350.0, T_air = 295.0, U = 4.0)
        K = f.column.K[1]
        seb_residual(f) = turbulent_fluxes(f, fo).G - K * (surface_temperature(f) - f.column.T[1])
        res = [abs(seb_residual(f))]
        for _ in 1:3
            step_surface!(f, fo, 1e-6)
            push!(res, abs(seb_residual(f)))
        end
        jump = surface_temperature(f) - 300.0
        @test res[2] < 1e-2 * res[1]
        @test res[3] < 1e-4 * res[2]          # quadratic: 0.45 → 5e-6 W/m²
        @test res[4] < 1e-8
        # On the balance, a tiny step returns the contract's fluxes at the skin.
        ref = turbulent_fluxes(f, fo)
        fl = step_surface!(f, fo, 1e-6)
        @test fl.Rn ≈ ref.Rn rtol = 1e-8
        @test fl.H ≈ ref.H rtol = 1e-6
        @test fl.G ≈ ref.G rtol = 1e-6
        @test fl.τ == ref.τ
        @test fl.LE == 0.0 && fl.E == 0.0
        @info @sprintf("Skin equilibration: initial skin moved %.3f K onto the balance; SEB residuals %s W/m² over successive steps",
                       jump, string(round.(res; sigdigits = 2)))
    end

    @testset "Facet steady state: surface energy balance meets the U-value" begin
        # Constant forcing, long backward-Euler steps: the skin temperature must
        # solve  Rn(Ts) - H(Ts) = (Ts - T_in) / (R_wall + R_si)  — found here by
        # bisection, independently of the solver.
        σ = DEFAULT_SURFACE_CONSTANTS.σ_SB
        fo = SurfaceForcing(sw_down = 500.0, lw_down = 320.0, T_air = 300.0, U = 3.0)
        T_in = 295.0
        bisect(F, a, b) = (for _ in 1:200; m = (a + b) / 2; F(a) * F(m) <= 0 ? (b = m) : (a = m); end; (a + b) / 2)
        results = []
        for (label, L, hc) in (("wall, h_c = 15", example_wall_layers(), 15.0),
                               ("roof, h_c = 10", example_roof_layers(), 10.0),
                               ("roof, MOST", example_roof_layers(), nothing))
            kind = startswith(label, "wall") ? :wall : :roof
            f = UrbanFacet(kind, L; T = 300.0, T_interior = T_in, h_convective = hc, θ = 1.0)
            Rsi = 1 / f.interior.h
            Ueff = 1 / (thermal_resistance(L) + Rsi)
            local fl
            for _ in 1:3000
                fl = step_surface!(f, fo, 3600.0)
            end
            probe = UrbanFacet(kind, L; T = 300.0, h_convective = hc)
            function Fres(Ts)
                probe.T_skin = Ts
                if hc === nothing
                    return turbulent_fluxes(probe, fo).G - Ueff * (Ts - T_in)
                end
                Rn = (1 - probe.albedo) * fo.sw_down + probe.emissivity * (fo.lw_down - σ * Ts^4)
                return Rn - hc * (Ts - fo.T_air) - Ueff * (Ts - T_in)
            end
            Ts_ref = bisect(Fres, 250.0, 400.0)
            @test surface_temperature(f) ≈ Ts_ref atol = 1e-6
            @test fl.G ≈ interior_heat_flux(f) rtol = 1e-8
            @test fl.G ≈ Ueff * (Ts_ref - T_in) rtol = 1e-6
            @test fl.Rn - fl.H - fl.LE ≈ fl.G atol = 1e-9
            push!(results, (label, surface_temperature(f), Ts_ref, fl.G))
        end
        for r in results
            @info @sprintf("Steady facet %-15s Ts = %.6f K (bisection %.6f), G = interior flux = %.4f W/m²", r...)
        end
        # Insulation: the roof passes far less heat to the room than the brick wall.
        @test results[2][4] < results[1][4] / 3
    end

    @testset "Diurnal cycle: exact facet energy bookkeeping, time-step robustness" begin
        forcing(t) = SurfaceForcing(sw_down = max(0.0, 800 * sin(2π * (t / 86400 - 0.25))),
                                    lw_down = 330.0 + 20 * sin(2π * (t / 86400 - 0.375)),
                                    T_air = 293.0 + 5 * sin(2π * (t / 86400 - 0.375)), U = 3.0)
        function diurnal(kind, L, dt; days = 3, θ = 0.5)
            f = UrbanFacet(kind, L; T = 293.0, θ = θ)
            E0 = column_heat_content(f.column; T_ref = 293.0)
            acc, seb, Ts, Qi = 0.0, 0.0, Float64[], Float64[]
            n = round(Int, 86400 / dt)
            for j in 0:days*n-1
                t = j * dt
                fl = step_surface!(f, forcing(t + dt / 2), dt)
                acc += (fl.G - interior_heat_flux(f)) * dt
                seb = max(seb, abs(fl.Rn - fl.H - fl.LE - fl.G))
                if j >= (days - 1) * n && (j + 1) % round(Int, 3600 / dt) == 0
                    push!(Ts, surface_temperature(f)); push!(Qi, interior_heat_flux(f))
                end
            end
            cons = abs(column_heat_content(f.column; T_ref = 293.0) - E0 - acc)
            return (; f, cons, acc, seb, Ts, Qi)
        end
        out = Dict()
        for (kind, L) in ((:roof, example_roof_layers()), (:wall, example_wall_layers()),
                          (:ground, example_ground_layers()))
            r60 = diurnal(kind, L, 60.0)
            r600 = diurnal(kind, L, 600.0)
            @test r60.seb < 1e-9                     # Rn - H - LE = G every step
            @test r60.cons < 1e-9 * sum(abs, r60.f.column.C .* r60.f.column.dz)
            dTs = maximum(abs.(r60.Ts .- r600.Ts))   # hourly skin temperatures, last day
            @test dTs < 0.5
            out[kind] = (r60, dTs)
            @info @sprintf("Diurnal %-6s: Ts range %.2f–%.2f K, SEB closure %.1e W/m², conservation %.1e J/m² of %.2e, |Ts(60 s) - Ts(600 s)| ≤ %.3f K, interior flux range %.2f–%.2f W/m²",
                           kind, minimum(r60.Ts), maximum(r60.Ts), r60.seb, r60.cons, abs(r60.acc), dTs,
                           minimum(r60.Qi), maximum(r60.Qi))
        end
        # Physics: an insulated roof damps the interior load. Its interior flux
        # swing is a small fraction of the uninsulated brick wall's, although its
        # skin temperature swing (low-albedo asphalt, full sun) is larger.
        roof, wall = out[:roof][1], out[:wall][1]
        swing(x) = maximum(x) - minimum(x)
        @test swing(roof.Qi) < swing(wall.Qi) / 5
        @test swing(roof.Ts) > swing(wall.Ts)
        # The roof's asphalt skin peaks in the early afternoon; the brick wall's
        # interior flux peaks hours later (thermal mass time lag).
        @test argmax(wall.Qi) > argmax(wall.Ts)
    end
end
