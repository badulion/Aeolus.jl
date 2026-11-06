# Phase 58 (P58-c, #171) and N-parent-c (#248) — LargeScaleForcing:
# geostrophic pressure gradient, prescribed advective tendencies, plane-mean
# nudging (ProfileNudging, NudgingWeight), large-scale subsidence and a
# prescribed surface-temperature series.
#
#   * the tables: bilinear interpolation, end-holding, and the exact step
#     mean of a piecewise-linear series against hand-computed integrals;
#   * the pressure gradient: the frictionless inertial oscillation
#     (Blackadar 1957) after one period 2π/f;
#   * nudging: the level-mean residual against the exact forward-Euler
#     amplification factor 1 − wΔt/τ; the tendency is uniform per level;
#   * subsidence: −w_s Γ on a linear profile, zero in a mixed layer;
#   * the ledger: Δ∫θ dV = booked nudging + subsidence + ∫∫(∂θ/∂t)_adv dV dt
#     (the latter by the trapezoid rule on the table nodes), to 1e-10.
#
# GABLS2 reduced (subsidence from 40 h) is the validation run
# validation/gabls2.jl.

using Test
using Aeolus
using Aeolus.Fields: interior_view, set!
using Statistics: mean

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

_ns_level(f, k) = (iv = interior_view(f); sum(@view iv[:, :, k]) / (size(iv, 1) * size(iv, 2)))
_ns_quiet(Nz; Lz = 800.0, θ₀ = 300.0, N = 4) =
    BoussinesqSimulation(RegularGrid(; size = (N, N, Nz), extent = (400.0, 400.0, Lz)); θ₀ = θ₀)

# a turbulent-looking, closed (fully periodic) box; flux-form advection and
# diffusion conserve ∫θ. Buoyancy is off (θ₀ = 0): θ is a passive scalar,
# and the periodic seam of a stratified θ would otherwise drive the flow
# unstable within the run (measured: NaN after 63 steps with θ₀ = 300 K).
function _ns_box(; N = 8, Nz = 8, moisture = false)
    mesh = RegularGrid(; size = (N, N, Nz), extent = (800.0, 800.0, 400.0))
    sim = BoussinesqSimulation(mesh; ν = 1.0, θ₀ = 0.0, f_coriolis = 1e-4, moisture = moisture)
    iu, iv, iθ = interior_view.((sim.u, sim.v, sim.θ))
    for k in 1:Nz, j in 1:N, i in 1:N
        iu[i, j, k] = 2 + sin(2π * j / N + k)
        iv[i, j, k] = cos(2π * i / N - k)
        iθ[i, j, k] = 300 + 0.004 * (k - 0.5) * mesh.dz + 0.3 * sin(i * j + k)
    end
    if moisture
        interior_view(sim.qv) .= [8e-3 - 1e-5 * k + 1e-4 * cos(i + 2j) for i in 1:N, j in 1:N, k in 1:Nz]
    end
    return sim
end

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
        @test forcing_value(tab, 1e4, 1e5) == 5.0                 # held above / after
        # the step mean of a piecewise-linear series is exact:
        # at z = 0 over [1800, 5400]: ∫ = 900·(1.5+2)/2·2 … by hand
        # p(1800) = 1.5, p(3600) = 2, p(5400) = 3 → (1.5+2)/2·1800 + (2+3)/2·1800
        @test forcing_mean(tab, 0.0, 1800.0, 5400.0) ≈ ((1.5 + 2) / 2 * 1800 + (2 + 3) / 2 * 1800) / 3600
        # straddling the end: [6300, 8100] → (3.5+4)/2·900 + 4·900
        @test forcing_mean(tab, 0.0, 6300.0, 8100.0) ≈ ((3.5 + 4) / 2 * 900 + 4 * 900) / 1800
        # step means add up to the whole integral to round-off
        tt = range(0.0, 9000.0; length = 38)
        s = sum(forcing_mean(tab, 150.0, tt[i], tt[i+1]) * (tt[i+1] - tt[i]) for i in 1:length(tt)-1)
        @test s ≈ forcing_mean(tab, 150.0, 0.0, 9000.0) * 9000 rtol = 1e-14
        # a profile (no time dependence), a number, a function
        prof = ForcingTable([0.0, 1000.0], [2.0, 12.0])
        @test forcing_value(prof, 250.0, 1e6) ≈ 4.5
        @test forcing_mean(prof, 250.0, 0.0, 600.0) ≈ 4.5
        @test forcing_mean((z, t) -> z + t^2, 1.0, 0.0, 3.0) ≈ 1 + 3.0   # Simpson is exact for t²
        @test_throws ArgumentError ForcingTable([0.0, 0.0], [0.0], zeros(2, 1))
    end

    @testset "SurfaceTemperatureSeries (GABLS1: 265 K, −0.25 K/h)" begin
        s = SurfaceTemperatureSeries([0.0, 9 * 3600.0], [265.0, 265.0 - 0.25 * 9])
        @test surface_temperature(s, 4 * 3600.0) ≈ 264.0
        @test surface_temperature(s, 9 * 3600.0) ≈ 262.75
        @test surface_temperature(s, 20 * 3600.0) ≈ 262.75       # held after the end
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
        @test err < 1e-3 * Vag
        @test abs(speed_max / (U_g + Vag) - 1) < 1e-3
        @test all(isapprox.(quarter, (b, -a); atol = 1e-3 * Vag))   # clockwise (NH)
        # the flow stays horizontally uniform: the forcing is level-uniform
        @test maximum(abs, interior_view(sim.u) .- _lsf_mean(sim.u)) < 1e-12
    end

    @testset "1. nudging e-folding: exact stepper factor" begin
        τ = 3600.0
        # forward Euler (step!), with a height weight: per level 1 − w(z)Δt/τ
        W = NudgingWeight(300.0; depth = 200.0)
        F = ProfileNudging((θ = 300.0,), τ; weight = W)
        # buoyancy off (θ₀ = 0): a level-dependent θ would otherwise give the
        # pressure solve a hydrostatic field to cancel
        sim = _ns_quiet(8; θ₀ = 0.0)
        interior_view(sim.θ) .= 302.0
        N = 100; dt = τ / N
        for _ in 1:N
            step!(sim, dt; body_force! = (s, Δt) -> apply_forcing!(s, F, s.t, Δt))
        end
        zc = [(k - 0.5) * sim.mesh.dz for k in 1:8]
        err = maximum(1:8) do k
            w = nudging_weight(W, zc[k], sim)
            r = (_ns_level(sim.θ, k) - 300) / 2
            abs(r - (1 - w * dt / τ)^N)
        end
        @test err < 1e-12
        @test _ns_level(sim.θ, 1) == 302.0               # below z_n: not nudged at all
        @test nudging_weight(W, 400.0, sim) ≈ 0.5
        @test_throws ArgumentError ProfileNudging((w = 0.0,), τ)
    end

    @testset "2. turbulence untouched: uniform tendency per level" begin
        prof = ForcingTable([0.0, 400.0], [301.0, 299.0])
        F = ProfileNudging((u = 3.0, v = -1.0, θ = prof), 600.0; weight = z -> min(1.0, z / 200))
        a = _ns_box()
        for f in (a.tu, a.tv, a.tθ); fill!(f.data, 0); end
        I = apply_forcing!(a, F, 0.0, 10.0)
        level_uniform(iv) = all(all(iv[:, :, k] .== iv[1, 1, k]) for k in 1:size(iv, 3))
        @test all(level_uniform(interior_view(f)) for f in (a.tu, a.tv, a.tθ))
        @test all(any(!=(0), interior_view(f)) for f in (a.tu, a.tv, a.tθ))
        @test keys(I) == (:u, :v, :θ)
    end

    @testset "4. linear profile: −w_s Γ at every level" begin
        Γ, ws = 0.003, -0.01
        sim = _ns_box(Nz = 16)
        m = sim.mesh
        iθ = interior_view(sim.θ)
        for k in 1:16, j in 1:8, i in 1:8   # θ₀ + Γz plus fluctuations of zero plane mean
            iθ[i, j, k] = 290 + Γ * (k - 0.5) * m.dz + 0.2 * sinpi(2i / 8) * cospi(2j / 8)
        end
        for form in (:mean_gradient, :local)
            fill!(sim.tθ.data, 0)
            I = apply_forcing!(sim, Subsidence(ws; fields = (:θ,), form = form), 0.0, 1.0)
            t = [_ns_level(sim.tθ, k) for k in 1:16]
            @test maximum(abs, t .+ ws * Γ) / abs(ws * Γ) < 1e-10
            @test I.θ ≈ -ws * Γ * m.Lx * m.Ly * m.Lz rtol = 1e-10
        end
    end

    @testset "6. mixed layer: no subsidence tendency where ⟨θ⟩ is constant" begin
        sim = _ns_box(Nz = 16)
        m = sim.mesh
        iθ = interior_view(sim.θ)
        zi = 200.0
        for k in 1:16, j in 1:8, i in 1:8
            z = (k - 0.5) * m.dz
            iθ[i, j, k] = (z < zi ? 300.0 : 300.0 + 0.01 * (z - zi)) + 0.05 * sinpi(2i / 8 + j / 4)
        end
        S = Subsidence(-0.02; fields = (:θ,))
        fill!(sim.tθ.data, 0)
        apply_forcing!(sim, S, 0.0, 1.0)
        kml = [k for k in 2:16 if (k + 0.5) * m.dz < zi]      # both neighbours inside the layer
        tml = maximum(abs, [_ns_level(sim.tθ, k) for k in kml])
        tabove = _ns_level(sim.tθ, 14)
        @test tml <= 0.02 * 4eps(300.0) / m.dz        # round-off of the plane means at most
        @test tabove ≈ 0.02 * 0.01 rtol = 1e-10
    end

    @testset "ledger: Δ∫θ dV = booked nudging + subsidence + advective tendency, to 1e-10" begin
        # a closed box: flux-form advection and diffusion conserve ∫θ, so the
        # change is the forcing alone. The tables vary with height and are
        # piecewise linear in time with kinks off the step boundaries.
        zt = [0.0, 200.0, 400.0]
        tt = [0.0, 300.0, 900.0]
        wtab = ForcingTable(zt, tt, [0.0 0.0 0.0; -0.004 -0.01 -0.002; -0.006 -0.02 -0.004])
        θref = ForcingTable([0.0, 400.0], [299.0, 303.0])
        sim = _ns_box()
        mesh = sim.mesh
        zc = [(k - 0.5) * mesh.dz for k in 1:8]
        tnodes = [0.0, 250.0, 600.0]
        Θ = [1e-4 * (1 + k / 8) * c for k in 1:8, c in (1.0, -2.0, 0.5)]   # K/s
        F = (ProfileNudging((θ = θref,), 300.0; weight = NudgingWeight(100.0; depth = 100.0)),
             Subsidence(wtab; form = :mean_gradient),
             AdvectiveTendencies(zc, tnodes, Θ, nothing, nothing, nothing))
        dV = mesh.dx * mesh.dy * mesh.dz
        H0 = sum(interior_view(sim.θ)) * dV
        nudged = 0.0; subsided = 0.0
        dt = 7.0; nsteps = 86                          # 602 s: past the last node
        for _ in 1:nsteps
            step!(sim, dt; body_force! = function (s, Δt)
                apply_forcing!(s, F, s.t, Δt)
                nudged += Δt * tendency_integrals(F[1]).θ
                subsided += Δt * tendency_integrals(F[2]).θ
            end)
        end
        T = nsteps * dt
        # ∫∫ F_adv dV dt: trapezoid on the nodes (exact for piecewise-linear t)
        # plus the held end value after the last node
        It = [sum((tnodes[n+1] - tnodes[n]) * (Θ[k, n] + Θ[k, n+1]) / 2 for n in 1:2) +
              (T - tnodes[end]) * Θ[k, end] for k in 1:8]
        advected = sum(It) * mesh.Nx * mesh.Ny * dV
        booked = nudged + subsided + advected
        ΔH = sum(interior_view(sim.θ)) * dV - H0
        @test abs(ΔH - booked) / abs(booked) < 1e-10
        @test abs(nudged) > 0 && subsided > 0 && abs(advected) > 0   # every term acts
        @test maximum(abs, interior_view(sim.w)) > 1e-3          # the flow really mixed
        # no humidity field to receive a q tendency: an error, not a no-op
        @test_throws ArgumentError apply_forcing!(sim, AdvectiveTendencies(; q = 1e-8), 0.0, 1.0)
    end
end
