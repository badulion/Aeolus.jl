# Berlin case: dynamical-core and time-integration options (#133).
#
#   slot :core         :boussinesq (default) | :anelastic | :pseudo_incompressible
#   slot :timestepper  :fixed (default) | :adaptive | :imex
#
# Every run here is CaseConfig(:smoke) (250 m × 250 m × 100 m of the real
# OSM scene, 16³, noon). The 1 h flagship comparisons (steps saved by the
# adaptive step, anelastic vs Boussinesq θ(z)) are measured in the PR.

using Test
using Aeolus
using Aeolus.Fields: interior_view
using Statistics: mean
using Printf

isdefined(Main, :BerlinCase) ||
    include(joinpath(@__DIR__, "..", "examples", "urban", "BerlinCase.jl"))
const BC = Main.BerlinCase

smoke(; physics = Dict{Symbol,Any}(), options = Dict{Symbol,Any}(), kw...) =
    BC.CaseConfig(:smoke; verbose = false, output_path = nothing,
                  physics = Dict{Symbol,Any}(physics), options = Dict{Symbol,Any}(options), kw...)

run_smoke(; kw...) = (case = BC.build_case(smoke(; kw...); io = devnull); BC.run_case!(case); case)

mean_θ(case) = vec(mean(interior_view(case.sim.θ); dims = (1, 2)))

"2Δ (checkerboard) amplitude of θ on level k: the mode an unstable explicit diffusion step amplifies."
checkerboard(case, k) = (θ = interior_view(case.sim.θ);
                         abs(mean(θ[i, j, k] * (-1)^(i + j) for i in axes(θ, 1), j in axes(θ, 2))))

@testset "Berlin dynamics options (#133)" begin

    @testset "slots, defaults, and the incompatible pairs are rejected with the reason" begin
        slots = BC.slot_providers()
        @test Set(slots[:core]) == Set([:boussinesq, :anelastic, :pseudo_incompressible])
        @test Set(slots[:timestepper]) == Set([:fixed, :adaptive, :imex])
        @test BC.default_provider(:core) === :boussinesq
        @test BC.default_provider(:timestepper) === :fixed
        for ts in (:adaptive, :imex), core in (:anelastic, :pseudo_incompressible)
            err = try
                BC.selected_couplings(smoke(physics = Dict(:timestepper => ts, :core => core)))
                nothing
            catch e
                sprint(showerror, e)
            end
            @test err !== nothing
            @test occursin("incompatible", err) && occursin("BoussinesqSimulation", err) &&
                  occursin("timestepper = :fixed", err)
        end
        # every other pair is allowed
        for ts in (:fixed, :adaptive, :imex)
            @test length(BC.selected_couplings(smoke(physics = Dict(:timestepper => ts)))) > 0
        end
        for core in (:anelastic, :pseudo_incompressible)
            @test length(BC.selected_couplings(smoke(physics = Dict(:core => core)))) > 0
        end
    end

    bous = run_smoke()
    @testset "default: Boussinesq core, fixed step" begin
        @test bous.sim isa BoussinesqSimulation
        @test bous.clock.n == 12 && bous.sim.t == 24.0
        @test max_divergence(bous.sim) < 1e-12
    end

    @testset "anelastic / pseudo-incompressible core: the weighted constraint, same flow" begin
        for core in (:anelastic, :pseudo_incompressible)
            case = run_smoke(physics = Dict(:core => core))
            sim, m = case.sim, case.mesh
            ref = Aeolus.Anelastic.reference_state(sim)
            @test sim isa Aeolus.Anelastic.MassWeightedSimulation
            @test case.clock.n == 12 && !case.clock.blowup
            # base state = the initial sounding, hydrostatic, from p_surf
            zc = [(k - 0.5) * m.dz for k in 1:m.Nz]
            @test maximum(abs.(ref.θ̄ .- BC.initial_θ(case.cfg).(0.0, 0.0, zc))) < 1e-10
            @test Aeolus.Anelastic.max_hydrostatic_residual(ref) < 1e-8
            # ideal gas at the first level: T = θ (p/p₀)^(R/cₚ), p ≈ p_surf
            T1 = ref.θ̄[1] * (101_325 / 1e5)^(287.0 / 1004.0)
            @test isapprox(ref.ρ̄[1], 101_325 / (287.0 * T1); rtol = 1e-3)
            # The projection enforces ∇·(W u) = 0: at every level the mean
            # mass flux ⟨ρ̄ w⟩ vanishes to the solver tolerance, while the
            # plain divergence is w·∂ln ρ̄/∂z — the density-scale-height
            # signature of the sound-proof system (Hρ ≈ 10.7 km here).
            w = interior_view(sim.w)
            wmax = maximum(abs, w)
            flux = [abs(mean(ref.ρ̄_f[k] .* view(w, :, :, k))) for k in 1:m.Nz]
            @test maximum(flux) < 1e-9 * ref.ρ̄[1] * wmax
            @test Aeolus.Anelastic.constraint_residual(sim) < 1e-9
            dlnρ = maximum(abs.(diff(log.(ref.ρ̄)))) / m.dz
            Hρ = 1 / dlnρ
            @test 8_000 < Hρ < 13_000
            @test 0.5 * wmax * dlnρ < max_divergence(sim) < 1.2 * wmax * dlnρ
            # a 100 m deep domain is 1 % of Hρ: the sound-proof core gives
            # the Boussinesq flow to well under 1 % after 24 s
            Δθ = maximum(abs.(mean_θ(case) .- mean_θ(bous)))
            Δw = maximum(abs.(w .- interior_view(bous.sim.w)))
            @test Δθ < 1e-4           # K; measured 4.1e-7
            @test Δw < 1e-2 * wmax    # measured 4e-4 × max|w|
            @printf("  [%s] 24 s: max|Δ⟨θ⟩| = %.1e K, max|Δw| = %.1e m/s (max|w| %.2f), ∇·(Wu) %.1e, ∇·u %.1e s⁻¹, Hρ = %.0f m\n",
                    core, Δθ, Δw, wmax, Aeolus.Anelastic.constraint_residual(sim), max_divergence(sim), Hρ)
        end
    end

    @testset "adaptive step: CFL-limited dt, fewer steps, ledgers close" begin
        case = run_smoke(physics = Dict(:timestepper => :adaptive))
        c = BC.find_coupling(case, :timestep_adaptive)
        s, k = c.stats, c.constraints
        m, cfg = case.mesh, case.cfg
        # explicit diffusive limit of ν = 2 m²/s at dx = 15.6 m, dz = 6.25 m
        Δ² = 1 / (1 / m.dx^2 + 1 / m.dy^2 + 1 / m.dz^2)
        dt_diff = k.C_diffusive * Δ² / cfg.ν
        @test isapprox(dt_diff, 3.70; atol = 0.01)
        @test case.sim.t ≈ 24.0 && s.dt_sum ≈ 24.0
        @test s.n < 12                          # measured 9 steps for 24 s
        @test s.retries == 0
        @test s.dt_max <= dt_diff + 1e-12       # never above the diffusive limit
        @test s.dt_max > cfg.dt                 # but longer than the fixed 2 s
        @test s.cfl_max <= k.C_advective
        @test get(s.limiters, :diffusive, 0) + get(s.limiters, :sgs, 0) == s.n   # diffusion binds
        # the ledgers close with a varying dt (they book with ledger.dt)
        @test case.ledger.elapsed ≈ 24.0
        for r in BC.surface_budget(case.ledger)
            @test abs(r.closure) < 1e-6
        end
        @test abs(BC.water_budget(case.ledger).residual) < 1e-12
        @printf("  [adaptive] 24 s in %d steps (fixed: 12), dt %.2f–%.2f s, diffusive limit %.2f s, max CFL %.2f\n",
                s.n, s.dt_min, s.dt_max, dt_diff, s.cfl_max)

        # A failed step is rolled back and the dt covered in pieces: a
        # 64 s step (17× the diffusive limit) overshoots a 1.2 m/s velocity
        # ceiling, so step_adaptive! halves it until a piece passes, and
        # the rest of the 64 s is covered by further pieces. Every
        # accepted piece respected the ceiling.
        c.max_velocity = 1.2
        t_old, pieces = case.sim.t, s.pieces
        BC.step_case!(case, 64.0)
        @test case.sim.t ≈ t_old + 64.0
        @test s.retries >= 1 && s.pieces >= pieces + 2
        @test c.dt_prev < 64.0                  # growth restarts from a piece that worked
        @test maximum(abs, interior_view(case.sim.u)) <= 1.2
        @printf("  [adaptive] a 64 s step: %d pieces, %d retries, next growth from %.1f s\n",
                s.pieces - pieces, s.retries, c.dt_prev)
        # a step that can never pass fails loudly and restores the dynamics
        c.max_velocity = 1e-3
        t_old = case.sim.t
        @test_throws ErrorException BC.step_case!(case, 2.0)
        @test case.sim.t == t_old
    end

    @testset "adaptive + IMEX: only the explicit limits bind" begin
        # Measured: at dx = 15.6 m, dz = 6.25 m, ν = 2 m²/s the explicit
        # diffusive limit (3.70 s) is only 4 % above the advective one at
        # C_advective = 0.5 (≈ 3.55 s once the flow spins up), so with
        # IMEX the advective CFL binds instead and the step is about the
        # same: IMEX buys robustness here, not a longer step.
        case = run_smoke(nsteps = 24, physics = Dict(:timestepper => :adaptive),
                         options = Dict(:timestep_adaptive => (imex = true,)))
        c = BC.find_coupling(case, :timestep_adaptive)
        s = c.stats
        @test case.sim.t ≈ 48.0
        @test s.retries == 0
        @test c.constraints.dt_max == 4.0       # capped at the shortest cadence (soil)
        @test s.cfl_max <= c.constraints.C_advective + 1e-9
        @test s.cfl_max > 0.9 * c.constraints.C_advective   # the advective limit is reached
        @test keys(s.limiters) ⊆ Set([:advective, :buoyancy, :coriolis])
        @test !haskey(s.limiters, :diffusive) && !haskey(s.limiters, :sgs)
        @test max_divergence(case.sim) < 1e-12
    end

    @testset "IMEX: stable beyond the explicit diffusive limit" begin
        # dt = 12 s is 1.6× the forward-Euler stability bound 2Δ²/(4ν)·…
        # (7.4 s here): the explicit step amplifies the 2Δ mode of θ and
        # the wind runs away; ARK2 with implicit diffusion does not.
        ex = run_smoke(dt = 12.0, nsteps = 8)
        im = run_smoke(dt = 12.0, nsteps = 8, physics = Dict(:timestepper => :imex))
        ui(c) = maximum(abs, interior_view(c.sim.u))
        @test ui(ex) > 3.0                       # measured 6.8 m/s from ~1 m/s
        @test ui(im) < 1.2                       # measured 1.02 m/s
        @test checkerboard(ex, 4) > 1e-2         # measured 0.35 K
        @test checkerboard(im, 4) < 1e-3         # measured 3.7e-5 K
        imc = BC.find_coupling(im, :timestep_imex)
        @test imc.stats.imex_residual <= imc.tol
        @test max_divergence(im.sim) < 1e-12
        # at the fixed 2 s the two agree to the truncation error
        im2 = run_smoke(physics = Dict(:timestepper => :imex))
        @test maximum(abs.(mean_θ(im2) .- mean_θ(bous))) < 1e-2    # K; measured 2.9e-3
        @printf("  [imex] dt = 12 s: max|u| explicit %.2f vs ARK2 %.2f m/s; 2Δ θ %.1e vs %.1e K\n",
                ui(ex), ui(im), checkerboard(ex, 4), checkerboard(im, 4))
    end
end
