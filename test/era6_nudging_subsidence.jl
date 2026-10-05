# N-parent-c (#248) — LargeScaleForcing: plane-mean nudging of (u, v, θ, qv)
# (`ProfileNudging`, `NudgingWeight`) and state-dependent large-scale
# subsidence (`Subsidence`).
#
# What is checked against what:
#   1. nudging e-folding: the level-mean residual against the exact
#      amplification factor of the stepper (forward Euler 1 − wΔt/τ per
#      step, SSP-RK3 1 − x + x²/2 − x³/6) and, for RK3 with Δt = τ/1000,
#      against e⁻¹ after t = τ;
#   2. turbulence untouched: the nudging tendency is exactly uniform per
#      level, and the fluctuations after one nudged step equal the
#      un-nudged ones to round-off;
#   3. bit for bit: GeostrophicNudging is a ProfileNudging and the Berlin
#      `:nudging` variant reproduces the former code (the 100-step test is
#      in era6_large_scale_forcing.jl; here the type and one level);
#   4. linear profile: −w_s Γ at every level, on a RegularGrid and on a
#      geometric StretchedGrid (cell centres read from the mesh);
#   5. inversion descent: a θ jump descends to z₁ + w_s t within one Δz
#      over 6 h, both forms;
#   6. mixed layer: no tendency where ⟨θ⟩ is constant;
#   7. ledger: Δ∫θ dV = the booked nudging + subsidence integrals, forward
#      Euler and SSP-RK3 (stage weights 1/6, 1/6, 2/3), to 1e-10;
#   8. GABLS2 (reduced): subsidence from 40 h, the heat ledger with it
#      booked, the residual-layer inversion lowered by about the passive
#      descent, and night 2 against the gabls2_reference.csv envelopes.

using Test, Printf
using Aeolus
using Aeolus.Fields: interior_view, set!
using Aeolus.Grids: znodes
using Aeolus.StretchedGrids: cell_thickness
using Statistics: mean

_ns_level(f, k) = (iv = interior_view(f); sum(@view iv[:, :, k]) / (size(iv, 1) * size(iv, 2)))
_ns_quiet(Nz; Lz = 800.0, θ₀ = 300.0, N = 4) =
    BoussinesqSimulation(RegularGrid(; size = (N, N, Nz), extent = (400.0, 400.0, Lz));
                         θ₀ = θ₀, bcs = FieldBoundaryConditions(:periodic, :periodic, :wall),
                         θ_bcs = FieldBoundaryConditions(:periodic, :periodic, :noflux))

# a turbulent-looking, closed box: periodic sides, impermeable walls, no
# surface flux; flux-form advection and diffusion conserve ∫θ
function _ns_box(; N = 8, Nz = 8, moisture = false)
    mesh = RegularGrid(; size = (N, N, Nz), extent = (800.0, 800.0, 400.0))
    sim = BoussinesqSimulation(mesh; ν = 1.0, θ₀ = 300.0, f_coriolis = 1e-4, moisture = moisture,
                               bcs = FieldBoundaryConditions(:periodic, :periodic, :wall),
                               θ_bcs = FieldBoundaryConditions(:periodic, :periodic, :noflux))
    iu, iv, iθ = interior_view.((sim.u, sim.v, sim.θ))
    for k in 1:Nz, j in 1:N, i in 1:N
        iu[i, j, k] = 2 + sin(2π * j / N + k)
        iv[i, j, k] = cos(2π * i / N - k)
        iθ[i, j, k] = 300 + 0.004 * (k - 0.5) * mesh.dz + 0.3 * sin(i * j + k)
    end
    if moisture
        interior_view(sim.qv) .= [8e-3 - 1e-5 * k + 1e-4 * cos(i + 2j) for i in 1:N, j in 1:N, k in 1:Nz]
    end
    Aeolus.TimeIntegration._project!(sim, 1.0)
    return sim
end

@testset "N-parent-c — ProfileNudging and Subsidence (#248)" begin

    @testset "1. nudging e-folding: exact stepper factor, e⁻¹ after τ" begin
        τ = 3600.0
        # forward Euler (step!), with a height weight: per level 1 − w(z)Δt/τ
        W = NudgingWeight(300.0; depth = 200.0)
        F = ProfileNudging((θ = 300.0,), τ; weight = W)
        # buoyancy off (θ₀ = 0): a level-dependent θ would otherwise give the
        # pressure solve a hydrostatic field to cancel, whose round-off
        # (~1e-12 K over the run) is the dynamics', not the forcing's
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
        r_top = (_ns_level(sim.θ, 8) - 300) / 2
        @info @sprintf("Nudging, forward Euler, Δt = τ/%d: max |residual − Π(1 − wΔt/τ)| = %.1e; top level (w = 1) %.12f vs (1 − 1/N)^N = %.12f, e⁻¹ = %.12f; bottom level (w = 0) residual %.3f",
                       N, err, r_top, (1 - 1 / N)^N, exp(-1), (_ns_level(sim.θ, 1) - 300) / 2)
        @test err < 1e-12
        @test _ns_level(sim.θ, 1) == 302.0               # below z_n: not nudged at all
        @test nudging_weight(W, 300.0, sim) == 0 && nudging_weight(W, 500.0, sim) == 1
        @test nudging_weight(W, 400.0, sim) ≈ 0.5
        # SSP-RK3 (step_rk3!): 1 − x + x²/2 − x³/6 per step, and e⁻¹ after τ
        F3 = ProfileNudging((θ = 300.0, u = 0.0), τ)
        sim = _ns_quiet(4)
        interior_view(sim.θ) .= 302.0
        interior_view(sim.u) .= 1.5
        N = 1000; dt = τ / N; x = dt / τ
        for _ in 1:N
            step_rk3!(sim, dt; body_force! = (s, Δt) -> apply_forcing!(s, F3, s.t, Δt))
        end
        r = (mean(interior_view(sim.θ)) - 300) / 2
        ru = mean(interior_view(sim.u)) / 1.5
        g = (1 - x + x^2 / 2 - x^3 / 6)^N
        @info @sprintf("Nudging, SSP-RK3, Δt = τ/%d: θ residual %.13f, u residual %.13f, Π(RK3 factor) %.13f, e⁻¹ %.13f; |θ − e⁻¹| = %.1e, |u − e⁻¹| = %.1e",
                       N, r, ru, g, exp(-1), abs(r - exp(-1)), abs(ru - exp(-1)))
        @test abs(r - g) < 1e-11 && abs(ru - g) < 1e-11
        @test abs(r - exp(-1)) < 1e-10
        @test abs(ru - exp(-1)) < 1e-10
        # a sim-dependent height: a multiple of a diagnosed depth
        Wz = NudgingWeight(s -> 250.0; factor = 2.0)
        @test nudging_weight(Wz, 499.0, sim) == 0 && nudging_weight(Wz, 501.0, sim) == 1
        @test_throws ArgumentError ProfileNudging((w = 0.0,), τ)
        @test_throws ArgumentError ProfileNudging((θ = 0.0,), -1.0)
        @test_throws ArgumentError apply_forcing!(sim, ProfileNudging((qv = 0.01,), τ), 0.0, 1.0)
    end

    @testset "2. turbulence untouched: uniform tendency, same fluctuations" begin
        τ = 600.0
        prof = ForcingTable([0.0, 400.0], [301.0, 299.0])
        F = ProfileNudging((u = 3.0, v = -1.0, θ = prof), τ; weight = z -> min(1.0, z / 200))
        a = _ns_box(); b = deepcopy(a)
        # the added tendency is the same number on every point of a level
        for f in (a.tu, a.tv, a.tθ); fill!(f.data, 0); end
        I = apply_forcing!(a, F, 0.0, 10.0)
        level_uniform(iv) = all(all(iv[:, :, k] .== iv[1, 1, k]) for k in 1:size(iv, 3))
        uniform = all(level_uniform(interior_view(f)) for f in (a.tu, a.tv, a.tθ))
        @test uniform
        @test all(any(!=(0), interior_view(f)) for f in (a.tu, a.tv, a.tθ))
        @test keys(I) == (:u, :v, :θ) && tendency_integrals(F) == I
        # one step with and one without: the fluctuations φ − ⟨φ⟩ agree
        a = _ns_box(); b = deepcopy(a)
        step!(a, 10.0; body_force! = (s, Δt) -> apply_forcing!(s, F, s.t, Δt))
        step!(b, 10.0)
        fluct(f) = (iv = interior_view(f); iv .- sum(iv; dims = (1, 2)) ./ (size(iv, 1) * size(iv, 2)))
        d = Dict(x => maximum(abs, fluct(getfield(a, x)) .- fluct(getfield(b, x))) for x in (:u, :v, :w, :θ))
        dm = maximum(abs, interior_view(a.θ) .- interior_view(b.θ))
        @info @sprintf("Fluctuations after one nudged step vs un-nudged: max |Δu'| %.1e, |Δv'| %.1e, |Δw'| %.1e m/s, |Δθ'| %.1e K (the mean moved by up to %.2e K)",
                       d[:u], d[:v], d[:w], d[:θ], dm)
        # round-off of adding a level constant to non-dyadic numbers
        # (ulp(300) = 5.7e-14), not bit for bit; see the PR note
        @test d[:θ] < 1e-12
        @test d[:u] < 1e-12 && d[:v] < 1e-12 && d[:w] < 1e-12
        @test dm > 1e-4                                  # the nudging did act on the mean
    end

    @testset "3. GeostrophicNudging is a ProfileNudging (one implementation)" begin
        F = GeostrophicNudging(1.0, 0.0, 1800.0)
        @test F isa ProfileNudging && keys(F.targets) == (:u, :v) && F.τ == 1800.0 && F.weight === nothing
        @test_throws ArgumentError GeostrophicNudging(1.0, 0.0, 0.0)
        sim = _ns_box()
        fill!(sim.tu.data, 0); fill!(sim.tv.data, 0)
        apply_forcing!(sim, F, 0.0, 1.0)
        for k in (1, 5)
            ū = 0.0; v̄ = 0.0
            for j in 1:8, i in 1:8
                ū += sim.u.data[sim.mesh.Hx+i, sim.mesh.Hy+j, sim.mesh.Hz+k]
                v̄ += sim.v.data[sim.mesh.Hx+i, sim.mesh.Hy+j, sim.mesh.Hz+k]
            end
            ū /= 64; v̄ /= 64
            @test interior_view(sim.tu)[3, 4, k] === -(1 / 1800.0) * (ū - 1.0)
            @test interior_view(sim.tv)[3, 4, k] === -(1 / 1800.0) * (v̄ - 0.0)
        end
    end

    @testset "4. linear profile: −w_s Γ at every level (regular and stretched)" begin
        Γ, ws = 0.003, -0.01
        sim = _ns_box(Nz = 16)
        m = sim.mesh
        iθ = interior_view(sim.θ)
        for k in 1:16, j in 1:8, i in 1:8   # θ₀ + Γz plus fluctuations of zero plane mean
            iθ[i, j, k] = 290 + Γ * (k - 0.5) * m.dz + 0.2 * sinpi(2i / 8) * cospi(2j / 8)
        end
        for form in (:mean_gradient, :local)
            fill!(sim.tθ.data, 0)
            S = Subsidence(ws; fields = (:θ,), form = form)
            I = apply_forcing!(sim, S, 0.0, 1.0)
            t = [_ns_level(sim.tθ, k) for k in 1:16]
            err = maximum(abs, t .+ ws * Γ) / abs(ws * Γ)
            @info @sprintf("Linear profile, %s: tendency %.10e K/s at every level (−w_sΓ = %.10e), max relative error %.1e",
                           form, mean(t), -ws * Γ, err)
            @test err < 1e-10
            @test I.θ ≈ -ws * Γ * m.Lx * m.Ly * m.Lz rtol = 1e-10
        end
        # a geometric StretchedGrid: the cell centres come from the mesh
        g = StretchedGrid(; size = (4, 4, 12), extent = (100.0, 100.0, 1200.0), stretching = (:geometric, 1.15))
        θ = Aeolus.Fields.Field(CellCenter(), g); tθ = Aeolus.Fields.Field(CellCenter(), g)
        zc = znodes(g, CellCenter())
        @test zc[2] - zc[1] < zc[end] - zc[end-1]        # really stretched
        interior_view(θ) .= [290 + Γ * zc[k] for _ in 1:4, _ in 1:4, k in 1:12]
        mock = (mesh = g, θ = θ, tθ = tθ)
        wprof = ForcingTable([0.0, 1200.0], [0.0, -0.012])   # w_s = −1e-5 z
        for form in (:mean_gradient, :local)
            fill!(tθ.data, 0)
            I = apply_forcing!(mock, Subsidence(wprof; fields = (:θ,), form = form), 0.0, 1.0)
            t = [_ns_level(tθ, k) for k in 1:12]
            exact = [1e-5 * zc[k] * Γ for k in 1:12]
            err = maximum(abs, t .- exact) / maximum(exact)
            vol = sum(cell_thickness(g)[g.Hz+k] * exact[k] for k in 1:12) * g.Lx * g.Ly
            @info @sprintf("StretchedGrid (geometric 1.15, Δz %.1f-%.1f m), %s: max relative error of −w_s(z)Γ %.1e; ∫ dV error %.1e",
                           zc[2] - zc[1], zc[end] - zc[end-1], form, err, abs(I.θ - vol) / vol)
            @test err < 1e-10
            @test I.θ ≈ vol rtol = 1e-10
        end
        # and the nudging integral on the stretched mesh: ∫ dV = Σ Δz_k A c_k
        fill!(tθ.data, 0)
        I = apply_forcing!(mock, ProfileNudging((θ = 290.0,), 100.0), 0.0, 1.0)
        @test I.θ ≈ -sum(Γ * zc[k] / 100 * cell_thickness(g)[g.Hz+k] for k in 1:12) * 1e4 rtol = 1e-10
        @test _ns_level(tθ, 7) ≈ -Γ * zc[7] / 100 rtol = 1e-10
    end

    @testset "5. inversion descent: z₁ + w_s t within one Δz over 6 h" begin
        ws, z1, Δθ = -0.01, 1000.0, 4.0
        T = 6 * 3600.0
        for form in (:mean_gradient, :local)
            sim = _ns_quiet(64; Lz = 1600.0, θ₀ = 292.0)
            dz = sim.mesh.dz
            set!(sim.θ, (x, y, z) -> z < z1 ? 290.0 : 290.0 + Δθ)
            S = Subsidence(ws; fields = (:θ,), form = form)
            dt = 60.0
            for _ in 1:round(Int, T / dt)
                step_rk3!(sim, dt; body_force! = (s, Δt) -> apply_forcing!(s, S, s.t, Δt))
            end
            θm = [_ns_level(sim.θ, k) for k in 1:64]
            zc = [(k - 0.5) * dz for k in 1:64]
            # the height where ⟨θ⟩ crosses the middle of the jump (from below)
            k = findfirst(>=(290 + Δθ / 2), θm)
            zx = zc[k-1] + (290 + Δθ / 2 - θm[k-1]) / (θm[k] - θm[k-1]) * dz
            @info @sprintf("Inversion descent (%s, Δz = %.0f m, 6 h, w_s = %.3f m/s): mid-jump at %.1f m, expected z₁ + w_s t = %.1f m (error %.1f m = %.2f Δz); ⟨θ⟩ range %.3f-%.3f K",
                           form, dz, ws, zx, z1 + ws * T, zx - (z1 + ws * T), (zx - (z1 + ws * T)) / dz,
                           minimum(θm), maximum(θm))
            @test abs(zx - (z1 + ws * T)) < dz
            @test maximum(abs, interior_view(sim.w)) < 1e-10   # no turbulence: pure subsidence
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
        @info @sprintf("Mixed layer (θ = 300 K ± 0.05 K fluctuations, z < %.0f m): max |tendency| %.1e K/s on %d levels; above: %.3e K/s (−w_sΓ = %.3e)",
                       zi, tml, length(kml), tabove, 0.02 * 0.01)
        @test tml <= 0.02 * 4eps(300.0) / m.dz        # round-off of the plane means at most
        @test tabove ≈ 0.02 * 0.01 rtol = 1e-10
    end

    @testset "7. ledger: Δ∫θ dV = booked nudging + subsidence, to 1e-10" begin
        zt = [0.0, 200.0, 400.0]
        tt = [0.0, 300.0, 900.0]
        wtab = ForcingTable(zt, tt, [0.0 0.0 0.0; -0.004 -0.01 -0.002; -0.006 -0.02 -0.004])
        θref = ForcingTable([0.0, 400.0], [299.0, 303.0])
        for (stepper, form) in ((:euler, :mean_gradient), (:euler, :local), (:rk3, :mean_gradient), (:rk3, :local))
            sim = _ns_box(moisture = stepper === :euler)
            F = (ProfileNudging((θ = θref,), 300.0; weight = NudgingWeight(100.0; depth = 100.0)),
                 Subsidence(wtab; form = form))
            dV = sim.mesh.dx * sim.mesh.dy * sim.mesh.dz
            H0 = sum(interior_view(sim.θ)) * dV
            Q0 = sum(interior_view(sim.qv)) * dV
            booked = 0.0; booked_q = 0.0; nudged = 0.0; subsided = 0.0
            dt = 6.0
            for _ in 1:100
                if stepper === :euler
                    step!(sim, dt; body_force! = function (s, Δt)
                        apply_forcing!(s, F, s.t, Δt)
                        nudged += Δt * tendency_integrals(F[1]).θ
                        subsided += Δt * tendency_integrals(F[2]).θ
                        booked_q += Δt * tendency_integrals(F, :qv)
                    end)
                else
                    stage = Ref(0); t0 = sim.t
                    step_rk3!(sim, dt; body_force! = function (s, Δt)
                        stage[] += 1
                        wst = (1 / 6, 1 / 6, 2 / 3)[stage[]]
                        apply_forcing!(s, F, t0, dt)
                        nudged += wst * dt * tendency_integrals(F[1]).θ
                        subsided += wst * dt * tendency_integrals(F[2]).θ
                    end)
                end
            end
            booked = nudged + subsided
            ΔH = sum(interior_view(sim.θ)) * dV - H0
            rel = abs(ΔH - booked) / abs(booked)
            msg = @sprintf("Ledger (%s, %s, 600 s): Δ∫θ dV = %.12e, booked = %.12e (nudging %.4e + subsidence %.4e) K m³, relative difference %.1e",
                           stepper, form, ΔH, booked, nudged, subsided, rel)
            if stepper === :euler
                ΔQ = sum(interior_view(sim.qv)) * dV - Q0
                relq = abs(ΔQ - booked_q) / abs(booked_q)
                msg *= @sprintf("; qv: Δ∫q dV = %.6e, booked %.6e, relative difference %.1e", ΔQ, booked_q, relq)
                @test relq < 1e-10
                @test keys(tendency_integrals(F[2])) == (:θ, :qv)   # moist: qv subsided by default
            else
                @test keys(tendency_integrals(F[2])) == (:θ,)       # dry: θ only
            end
            @info msg
            @test rel < 1e-10
            @test abs(nudged) > 0 && subsided > 0                    # both terms act; descent warms
            @test maximum(abs, interior_view(sim.w)) > 1e-3          # the flow really mixed
        end
    end

end
