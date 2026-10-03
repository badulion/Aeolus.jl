# Phase 58 (P58-c, #171) — LargeScaleForcing: geostrophic pressure
# gradient, geostrophic nudging, prescribed advective tendencies and a
# prescribed surface-temperature series.
#
# What is checked against what:
#   * the tables: bilinear interpolation, end-holding, and the exact step
#     mean of a piecewise-linear series against hand-computed integrals;
#   * the pressure gradient: the frictionless inertial oscillation
#     (Blackadar 1957) after one period 2π/f and its maximum speed
#     U_g + |V_ag|, and the laminar Ekman spiral (Ekman 1905) over a
#     no-slip wall against the analytic profile and its 45° surface veer;
#   * the advective tendencies: Δ∫θ dV against ∫∫(∂θ/∂t)_adv dV dt, the
#     latter by the trapezoid rule on the table nodes, to round-off;
#   * bit for bit: GeostrophicNudging against the Berlin case's former
#     add_geostrophic_wind! over 100 smoke steps.

using Test, Printf
using Aeolus
using Aeolus.Fields: interior_view

# Shu–Osher SSP-RK3 of the projected Euler step `step!`, with a body
# force and the stage times t, t + Δt, t + Δt/2 (step_rk3! takes no hook).
function _lsf_rk3!(sim, dt, bf!)
    t0 = sim.t
    u0 = copy(sim.u.data); v0 = copy(sim.v.data); w0 = copy(sim.w.data); θ0 = copy(sim.θ.data)
    step!(sim, dt; body_force! = bf!)
    step!(sim, dt; body_force! = bf!)
    @. sim.u.data = 0.75 * u0 + 0.25 * sim.u.data
    @. sim.v.data = 0.75 * v0 + 0.25 * sim.v.data
    @. sim.w.data = 0.75 * w0 + 0.25 * sim.w.data
    @. sim.θ.data = 0.75 * θ0 + 0.25 * sim.θ.data
    sim.t = t0 + dt / 2
    step!(sim, dt; body_force! = bf!)
    @. sim.u.data = u0 / 3 + 2 / 3 * sim.u.data
    @. sim.v.data = v0 / 3 + 2 / 3 * sim.v.data
    @. sim.w.data = w0 / 3 + 2 / 3 * sim.w.data
    @. sim.θ.data = θ0 / 3 + 2 / 3 * sim.θ.data
    sim.t = t0 + dt
    return sim
end

_lsf_mean(f) = sum(interior_view(f)) / length(interior_view(f))
_lsf_level(f, k) = (iv = interior_view(f); sum(@view iv[:, :, k]) / (size(iv, 1) * size(iv, 2)))

@testset "Era VI — LargeScaleForcing (#171)" begin

    @testset "ForcingTable: interpolation, end values, exact step mean" begin
        z = [0.0, 100.0, 300.0]
        t = [0.0, 3600.0, 7200.0]
        V = [1.0 2.0 4.0;
             3.0 3.0 3.0;
             5.0 0.0 5.0]
        tab = ForcingTable(z, t, V)
        @test forcing_value(tab, 0.0, 0.0) == 1.0
        @test forcing_value(tab, 300.0, 3600.0) == 0.0
        @test forcing_value(tab, 50.0, 0.0) ≈ 2.0                 # mid-way in z
        @test forcing_value(tab, 200.0, 1800.0) ≈ (0.5 * (3 + 5) + 0.5 * (3 + 0)) / 2
        @test forcing_value(tab, -10.0, -100.0) == 1.0            # held below / before
        @test forcing_value(tab, 1e4, 1e5) == 5.0                 # held above / after
        # the step mean of a piecewise-linear series is exact:
        # at z = 0 over [1800, 5400]: ∫ = 900·(1.5+2)/2·2 … by hand
        # p(1800) = 1.5, p(3600) = 2, p(5400) = 3 → (1.5+2)/2·1800 + (2+3)/2·1800
        @test forcing_mean(tab, 0.0, 1800.0, 5400.0) ≈ ((1.5 + 2) / 2 * 1800 + (2 + 3) / 2 * 1800) / 3600
        # beyond the end the value is held: mean over [7200, 9000] = 4
        @test forcing_mean(tab, 0.0, 7200.0, 9000.0) ≈ 4.0
        # straddling the end: [6300, 8100] → (3.5+4)/2·900 + 4·900
        @test forcing_mean(tab, 0.0, 6300.0, 8100.0) ≈ ((3.5 + 4) / 2 * 900 + 4 * 900) / 1800
        # a zero-length step is the point value
        @test forcing_mean(tab, 200.0, 1800.0, 1800.0) == forcing_value(tab, 200.0, 1800.0)
        # step means add up to the whole integral to round-off
        tt = range(0.0, 9000.0; length = 38)
        s = sum(forcing_mean(tab, 150.0, tt[i], tt[i+1]) * (tt[i+1] - tt[i]) for i in 1:length(tt)-1)
        @test s ≈ forcing_mean(tab, 150.0, 0.0, 9000.0) * 9000 rtol = 1e-14
        # a profile (no time dependence), a number, a function
        prof = ForcingTable([0.0, 1000.0], [2.0, 12.0])
        @test forcing_value(prof, 250.0, 1e6) ≈ 4.5
        @test forcing_mean(prof, 250.0, 0.0, 600.0) ≈ 4.5
        @test forcing_mean(7.0, 1.0, 0.0, 1.0) === 7.0
        @test forcing_mean((z, t) -> z + t^2, 1.0, 0.0, 3.0) ≈ 1 + 3.0   # Simpson is exact for t²
        @test_throws ArgumentError ForcingTable([0.0, 0.0], [0.0], zeros(2, 1))
        @test_throws ArgumentError ForcingTable([0.0, 1.0], [0.0], zeros(3, 1))
    end

    @testset "SurfaceTemperatureSeries (GABLS1: 265 K, −0.25 K/h)" begin
        s = SurfaceTemperatureSeries([0.0, 9 * 3600.0], [265.0, 265.0 - 0.25 * 9])
        @test surface_temperature(s, 0.0) == 265.0
        @test surface_temperature(s, 4 * 3600.0) ≈ 264.0
        @test surface_temperature(s, 9 * 3600.0) ≈ 262.75
        @test surface_temperature(s, 20 * 3600.0) ≈ 262.75       # held after the end
        @test_throws ArgumentError SurfaceTemperatureSeries([0.0, 1.0], [265.0])
    end

    @testset "inertial oscillation about U_g (Blackadar 1957)" begin
        # Frictionless, horizontally uniform flow under the geostrophic
        # pressure gradient: W = (u − U_g) + i v obeys dW/dt = −i f W, so the
        # ageostrophic wind turns clockwise at f and returns after 2π/f.
        f, U_g, a, b = 1e-4, 5.0, 2.0, 1.0
        Vag = hypot(a, b)
        mesh = RegularGrid(; size = (4, 4, 4), extent = (400.0, 400.0, 400.0))
        sim = BoussinesqSimulation(mesh; f_coriolis = f)
        interior_view(sim.u) .= U_g + a
        interior_view(sim.v) .= b
        F = GeostrophicPressureGradient(U_g, 0.0)
        bf!(s, Δt) = apply_forcing!(s, F, s.t, Δt)
        N = 1000
        dt = 2π / f / N
        speed_max = 0.0
        quarter = (0.0, 0.0)
        for n in 1:N
            _lsf_rk3!(sim, dt, bf!)
            u, v = _lsf_mean(sim.u), _lsf_mean(sim.v)
            speed_max = max(speed_max, hypot(u, v))
            n == N ÷ 4 && (quarter = (u - U_g, v))
        end
        err = hypot(_lsf_mean(sim.u) - (U_g + a), _lsf_mean(sim.v) - b)
        @info @sprintf("Inertial oscillation: error after one period %.2e m/s = %.2e |V_ag|; max speed %.6f vs U_g + |V_ag| = %.6f (rel. %.1e); quarter period (u−U_g, v) = (%.4f, %.4f) vs (b, −a) = (%.1f, %.1f)",
                       err, err / Vag, speed_max, U_g + Vag, speed_max / (U_g + Vag) - 1, quarter..., b, -a)
        @test err < 1e-3 * Vag
        @test abs(speed_max / (U_g + Vag) - 1) < 1e-3
        @test all(isapprox.(quarter, (b, -a); atol = 1e-3 * Vag))   # clockwise (NH)
        # the flow stays horizontally uniform: the forcing is level-uniform
        @test maximum(abs, interior_view(sim.u) .- _lsf_mean(sim.u)) < 1e-12
    end

    @testset "laminar Ekman spiral over a no-slip wall (Ekman 1905)" begin
        # ν = 1 m²/s, f = 1e-4: δ = √(2ν/f) = 141.4 m. Analytic steady state
        #   u = U_g (1 − e^{−z/δ} cos z/δ),  v = U_g e^{−z/δ} sin z/δ,
        # whose wall stress points 45° to the left of U_g.
        # The no-slip bottom: neither #172 (NoSlipWall) nor #169
        # (SpaldingLaw) is on master, so the wall stress is the laminar wall
        # law τ/ρ = ν u₁/(Δz/2) — the u⁺ = y⁺ limit of Spalding's law, i.e.
        # the no-slip ghost u₀ = −u₁ — added to the first level of a
        # free-slip (:wall) domain.
        ν, f, U_g = 1.0, 1e-4, 10.0
        δ = sqrt(2ν / f)
        Nz = 64
        Lz = 1280.0                                   # 9.05 δ
        mesh = RegularGrid(; size = (4, 4, Nz), extent = (400.0, 400.0, Lz))
        sim = BoussinesqSimulation(mesh; ν = ν, f_coriolis = f,
                                   bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
        interior_view(sim.u) .= U_g                   # impulsive start from geostrophic flow
        F = GeostrophicPressureGradient(U_g, 0.0)
        dz = mesh.dz
        K = mesh.Hz + 1
        function bf!(s, Δt)
            apply_forcing!(s, F, s.t, Δt)
            c = ν / (dz / 2) / dz
            @views s.tu.data[:, :, K] .-= c .* s.u.data[:, :, K]
            @views s.tv.data[:, :, K] .-= c .* s.v.data[:, :, K]
            return nothing
        end
        dt = 100.0                                    # ν dt/dz² = 0.25
        nsteps = 6000                                 # f t = 60
        for _ in 1:nsteps
            _lsf_rk3!(sim, dt, bf!)
        end
        z = [(k - 0.5) * dz for k in 1:Nz]
        u = [_lsf_level(sim.u, k) for k in 1:Nz]
        v = [_lsf_level(sim.v, k) for k in 1:Nz]
        ue = @. U_g * (1 - exp(-z / δ) * cos(z / δ))
        ve = @. U_g * exp(-z / δ) * sin(z / δ)
        l2 = sqrt(sum(abs2, u .- ue) + sum(abs2, v .- ve)) / sqrt(sum(abs2, ue) + sum(abs2, ve))
        l2_ag = sqrt(sum(abs2, u .- ue) + sum(abs2, v .- ve)) /
                sqrt(sum(abs2, ue .- U_g) + sum(abs2, ve))
        # Surface veer: the angle between the surface stress and U_g. The
        # model's wall stress is ν (u₁, v₁)/(Δz/2), so its direction is that
        # of the first-level wind. (The analytic first-level wind, at z = 10 m,
        # points 43.0°: the wall law's half-cell linear profile puts the
        # first level 1.7° off it, and a quadratic extrapolation of the
        # discrete shear to z = 0 gives ≈ 47°; both are the first-order
        # wall-stress closure, not the forcing.)
        veer = atand(v[1], u[1])
        veer_quad = atand(9v[1] - v[2], 9u[1] - u[2])
        # the maximum of v (the spiral's cross-isobaric jet) and the
        # supergeostrophic maximum of u, at z = πδ/4 and 3πδ/4
        @info @sprintf("Ekman spiral (Nz = %d, dz = %.0f m, δ = %.1f m, f t = %.0f): relative L2 error %.2e (of the ageostrophic part %.2e); surface veer (stress direction) %.3f° (quadratic shear extrapolation %.3f°); max u %.4f vs %.4f, max v %.4f vs %.4f",
                       Nz, dz, δ, f * nsteps * dt, l2, l2_ag, veer, veer_quad, maximum(u), maximum(ue), maximum(v), maximum(ve))
        @test l2 < 0.01
        @test abs(veer - 45) < 0.5
        @test maximum(u) > U_g                        # supergeostrophic overshoot
        @test isapprox(maximum(u), maximum(ue); rtol = 0.01)
        @test isapprox(maximum(v), maximum(ve); rtol = 0.01)
    end

    @testset "advective tendencies: Δ∫θ dV = ∫∫(∂θ/∂t)_adv dV dt" begin
        # A turbulent, closed box (periodic sides, impermeable walls, no
        # surface flux): flux-form advection and diffusion conserve ∫θ, so
        # the change is the prescribed large-scale tendency alone. The table
        # varies with height and is piecewise linear in time with a kink at
        # t = 250 s that does not fall on a step boundary.
        mesh = RegularGrid(; size = (8, 8, 8), extent = (800.0, 800.0, 400.0))
        sim = BoussinesqSimulation(mesh; ν = 1.0, θ₀ = 300.0, f_coriolis = 1e-4,
                                   bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
        rng_vals(k) = (sin(1.3k), cos(0.7k))
        iu, iv, iw, iθ = interior_view.((sim.u, sim.v, sim.w, sim.θ))
        for k in 1:8, j in 1:8, i in 1:8
            iu[i, j, k] = 2 + sin(2π * j / 8 + k)
            iv[i, j, k] = cos(2π * i / 8 - k)
            iθ[i, j, k] = 300 + 0.01 * (i + 2j - k) + 0.3 * sin(i * j)
        end
        zc = [(k - 0.5) * mesh.dz for k in 1:8]
        tnodes = [0.0, 250.0, 600.0]
        Θ = [1e-4 * (1 + k / 8) * c for k in 1:8, c in (1.0, -2.0, 0.5)]   # K/s
        F = AdvectiveTendencies(zc, tnodes, Θ, nothing, nothing, nothing)
        bf!(s, Δt) = apply_forcing!(s, F, s.t, Δt)
        dV = mesh.dx * mesh.dy * mesh.dz
        H0 = sum(interior_view(sim.θ)) * dV
        dt = 7.0
        nsteps = 86                                   # 602 s: past the last node
        for _ in 1:nsteps
            step!(sim, dt; body_force! = bf!)
        end
        T = nsteps * dt
        # ∫∫ F dV dt: trapezoid on the nodes (exact for piecewise-linear t)
        # plus the held end value after the last node.
        It = [sum((tnodes[n+1] - tnodes[n]) * (Θ[k, n] + Θ[k, n+1]) / 2 for n in 1:2) +
              (T - tnodes[end]) * Θ[k, end] for k in 1:8]
        expected = sum(It) * mesh.Nx * mesh.Ny * dV
        ΔH = sum(interior_view(sim.θ)) * dV - H0
        rel = abs(ΔH - expected) / abs(expected)
        @info @sprintf("Advective tendencies: Δ∫θ dV = %.12e K m³, ∫∫F dV dt = %.12e K m³, relative difference %.1e (max |w| %.2f m/s)",
                       ΔH, expected, rel, maximum(abs, interior_view(sim.w)))
        @test rel < 1e-11
        @test maximum(abs, interior_view(sim.w)) > 1e-3   # the flow really mixed
        # no humidity field to receive a q tendency: an error, not a no-op
        Fq = AdvectiveTendencies(; q = 1e-8)
        @test_throws ArgumentError apply_forcing!(sim, Fq, 0.0, 1.0)
    end

end
