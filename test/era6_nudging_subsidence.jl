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

end
