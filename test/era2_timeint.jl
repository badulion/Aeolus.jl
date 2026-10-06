# =========================================================================
# Era II — Phases 23 & 25: adaptive time stepping and semi-implicit IMEX.
# =========================================================================
#
# Standalone: `julia --project=. -e 'using Test, Aeolus; include("test/era2_timeint.jl")'`
# Numbers that quantify the schemes (solver iteration counts, measured
# convergence orders, stability-limit ratios) are printed as they are
# measured — they are the point of several of these tests.

using Test
using Aeolus
using Aeolus.TimeIntegration
using Aeolus.Fields: set!, interior_view
using Aeolus.Grids: interior, xnodes, znodes

@testset "Era II — Adaptive dt & IMEX" begin

    # =====================================================================
    # Phase 23 — CFL diagnostics
    # =====================================================================
    @testset "CFL diagnostics — advective limit" begin
        g = RegularGrid(; size = (8, 8, 8), extent = (1.0, 2.0, 4.0), halo = (2, 2, 2))
        sim = BoussinesqSimulation(g)              # ν = 0, no SGS, no buoyancy

        U, V, W = 1.0, -2.0, 0.5
        set!(sim.u, U); set!(sim.v, V); set!(sim.w, W)

        c = CFLConstraints()                       # C_advective = 0.5
        d = cfl_diagnostics(sim, c)

        # Multidimensional CFL is the per-cell SUM of the directional rates.
        rate_hand = abs(U) / g.dx + abs(V) / g.dy + abs(W) / g.dz
        dt_hand   = c.C_advective / rate_hand

        @test d.advective_rate ≈ rate_hand atol = 1e-14
        @test abs(d.dt_adv - dt_hand) < 1e-12
        @test d.limiter == :advective
        @test d.dt_limiting == d.dt_adv
        # everything else is switched off ⇒ never limiting
        @test d.dt_diff == Inf && d.dt_buoy == Inf && d.dt_cor == Inf && d.dt_sgs == Inf
        # and the CFL number of the returned dt is exactly C_advective
        @test d.dt_adv * d.advective_rate ≈ c.C_advective

        # --- crank ν until diffusion binds -------------------------------
        simν = BoussinesqSimulation(g; ν = 10.0)
        set!(simν.u, 0.01)
        dν = cfl_diagnostics(simν, c)
        Δ²harm  = 1 / (1 / g.dx^2 + 1 / g.dy^2 + 1 / g.dz^2)
        dt_diff_hand = c.C_diffusive * Δ²harm / 10.0
        @test abs(dν.dt_diff - dt_diff_hand) < 1e-12
        @test dν.limiter == :diffusive
        @test dν.dt_limiting == dν.dt_diff
        @test dν.dt_diff < dν.dt_adv

        println("  [cfl] dt_adv = $(d.dt_adv) (hand $(dt_hand)), " *
                "dt_diff(ν=10) = $(dν.dt_diff)")
    end

    @testset "CFL diagnostics — buoyancy and Coriolis limits" begin
        g = RegularGrid(; size = (4, 4, 16), extent = (100.0, 100.0, 400.0), halo = (2, 2, 2))
        Γ = 0.01                                        # K/m, stable lapse rate
        sim = BoussinesqSimulation(g; θ₀ = 300.0, f_coriolis = 1e-4)
        set!(sim.θ, (x, y, z) -> 300.0 + Γ * z)
        set!(sim.u, 0.001)                              # advection irrelevant

        c = CFLConstraints()
        d = cfl_diagnostics(sim, c)

        N_hand = sqrt(9.81 / 300.0 * Γ)
        @test d.N ≈ N_hand rtol = 1e-12
        @test d.dt_buoy ≈ c.C_buoyancy / N_hand rtol = 1e-12
        @test d.dt_cor  ≈ c.C_coriolis / 1e-4  rtol = 1e-12
        @test d.limiter == :buoyancy                    # N ≫ f
        println("  [cfl] N = $(d.N) 1/s, dt_buoy = $(d.dt_buoy) s, dt_cor = $(d.dt_cor) s")

        # neutral profile ⇒ no buoyancy limit, Coriolis takes over
        set!(sim.θ, 300.0)
        d2 = cfl_diagnostics(sim, c)
        @test d2.N == 0.0 && d2.dt_buoy == Inf
        @test d2.limiter == :coriolis
    end

    @testset "CFL diagnostics — SGS limit is reported separately" begin
        g = RegularGrid(; size = (8, 8, 8), extent = (1.0, 1.0, 1.0), halo = (2, 2, 2))
        # C_s is deliberately large so that the SGS diffusivity, not advection,
        # is the binding constraint on this grid.
        sim = BoussinesqSimulation(g; C_s = 0.6, Pr_t = 1/3)
        set!(sim.u, (x, y, z) -> sin(2π * x) * cos(2π * y))
        set!(sim.v, (x, y, z) -> -cos(2π * x) * sin(2π * y))
        d = cfl_diagnostics(sim, CFLConstraints())
        @test d.νt_max > 0
        @test isfinite(d.dt_sgs) && isfinite(d.dt_diff)
        # Prₜ = 1/3 ⇒ scalar diffusivity is 3νt ⇒ the SGS limit is the tighter one
        @test d.dt_sgs < d.dt_diff
        @test d.dt_sgs ≈ d.dt_diff / 3 rtol = 1e-12     # ν = 0, so the ratio is Prₜ
        @test d.dt_sgs < d.dt_adv
        @test d.limiter == :sgs
        println("  [cfl] νt_max = $(d.νt_max), dt_diff = $(d.dt_diff), dt_sgs = $(d.dt_sgs)")
    end

    # =====================================================================
    # Phase 23 — adaptive_dt growth limiting
    # =====================================================================
    @testset "adaptive_dt growth limiting and clamping" begin
        g = RegularGrid(; size = (8, 8, 8), extent = (1.0, 1.0, 1.0), halo = (2, 2, 2))
        sim = BoussinesqSimulation(g)
        set!(sim.u, 0.1)                       # loose advective limit: dt_adv = 0.625

        c = CFLConstraints(growth_max = 1.1, dt_min = 1e-8, dt_max = 10.0)
        d = cfl_diagnostics(sim, c)

        dt = 1e-4
        history = Float64[dt]
        for _ in 1:40
            dt_new = adaptive_dt(sim, c; dt_prev = dt)
            @test dt_new <= c.growth_max * dt * (1 + 1e-12)   # never jumps
            @test c.dt_min <= dt_new <= c.dt_max
            push!(history, dt_new)
            dt = dt_new
        end
        # it does grow, and saturates at the CFL limit rather than overshooting
        @test history[end] > history[1]
        @test history[end] <= d.dt_limiting * (1 + 1e-12)
        @test maximum(history[i+1] / history[i] for i in 1:length(history)-1) <=
              c.growth_max * (1 + 1e-12)

        # no growth cap requested ⇒ straight to the CFL limit
        @test adaptive_dt(sim, c) ≈ min(d.dt_limiting, c.dt_max)

        # dt_max clamp binds when nothing else does
        c2 = CFLConstraints(dt_max = 0.01)
        @test adaptive_dt(sim, c2) == 0.01

        # a broken state must not produce a NaN dt
        set!(sim.u, NaN)
        @test adaptive_dt(sim, c) == c.dt_min
        println("  [adaptive_dt] ramp 1e-4 → $(history[end]) in $(length(history)-1) " *
                "calls (CFL limit $(d.dt_limiting))")
    end

    # =====================================================================
    # Phase 23 — substepping schedule
    # =====================================================================
    @testset "SubsteppingSchedule" begin
        s = SubsteppingSchedule(microphysics = 4, chemistry = 2)
        @test substeps_for(s, :microphysics) == 4
        @test substeps_for(s, :chemistry) == 2
        @test substeps_for(s, :radiation) == 1        # default
        @test substeps_for(nothing, :anything) == 1
        s2 = SubsteppingSchedule(:micro => 8; default = 3)
        @test substeps_for(s2, :micro) == 8
        @test substeps_for(s2, :other) == 3
        @test_throws ArgumentError SubsteppingSchedule(:bad => 0)
    end

    @testset "step_adaptive! runs fast processes at dt/n" begin
        g = RegularGrid(; size = (8, 8, 8), extent = (1.0, 1.0, 1.0), halo = (2, 2, 2))
        sim = BoussinesqSimulation(g)
        set!(sim.u, 1.0)

        micro_dts = Float64[]
        chem_dts  = Float64[]
        procs = (microphysics = (s, Δt) -> (push!(micro_dts, Δt); nothing),
                 chemistry    = (s, Δt) -> (push!(chem_dts, Δt); nothing))
        sched = SubsteppingSchedule(microphysics = 4, chemistry = 2)

        c = CFLConstraints(dt_max = 0.05)
        _, dt_used, nretry = step_adaptive!(sim, c; fast_processes = procs,
                                            schedule = sched)
        @test nretry == 0
        @test length(micro_dts) == 4 && length(chem_dts) == 2
        @test all(≈(dt_used / 4), micro_dts)
        @test all(≈(dt_used / 2), chem_dts)
        @test sum(micro_dts) ≈ dt_used
        @test sim.t ≈ dt_used
    end

    # =====================================================================
    # Phase 23 — retry logic
    # =====================================================================
    @testset "step_adaptive! retries and reduces dt (runaway growth)" begin
        g = RegularGrid(; size = (8, 8, 8), extent = (1.0, 1.0, 1.0), halo = (2, 2, 2))
        sim = BoussinesqSimulation(g)
        set!(sim.u, 1.0)                      # uniform ⇒ advection and ∇·u vanish

        # A stiff explicit growth term: one Euler step multiplies u by (1 + Δt·G).
        G = 1.0e4
        function blowup!(s, Δt)
            ii, jj, kk = interior(s.mesh)
            @views s.tu.data[ii, jj, kk] .+= G .* s.u.data[ii, jj, kk]
            return nothing
        end

        c = CFLConstraints(dt_min = 1e-8, dt_max = 1.0)
        # dt = 1e-3 ⇒ |u| = 11 (reject), 5e-4 ⇒ 6 (reject), 2.5e-4 ⇒ 3.5 (accept)
        sim2, dt_used, nretry = step_adaptive!(sim, c; dt = 1e-3, body_force! = blowup!,
                                              max_velocity = 4.0, max_retries = 3)
        @test nretry == 2
        @test dt_used ≈ 1e-3 / 4
        @test all(isfinite, interior_view(sim.u))
        @test all(isfinite, interior_view(sim.v))
        @test all(isfinite, interior_view(sim.w))
        @test all(isfinite, interior_view(sim.θ))
        @test maximum(abs, interior_view(sim.u)) ≈ 1 + dt_used * G
        @test sim.t ≈ dt_used          # the rejected attempts did not advance the clock
        println("  [retry] runaway growth: $(nretry) retries, dt 1e-3 → $(dt_used), " *
                "|u|max = $(maximum(abs, interior_view(sim.u)))")

        # --- the non-finite path ----------------------------------------
        simn = BoussinesqSimulation(g)
        set!(simn.u, 1.0); set!(simn.θ, 300.0)
        function nan_bomb!(s, Δt)
            Δt > 1.001e-3 && fill!(s.tu.data, NaN)
            return nothing
        end
        _, dtn, nretryn = step_adaptive!(simn, c; dt = 4e-3, body_force! = nan_bomb!,
                                         max_retries = 3)
        @test nretryn == 2
        @test dtn ≈ 1e-3
        @test all(isfinite, interior_view(simn.u))
        @test maximum(abs, interior_view(simn.u)) ≈ 1.0   # no forcing survived

        # --- exhausting the retries restores the state and throws --------
        simf = BoussinesqSimulation(g)
        set!(simf.u, 1.0)
        u_before = copy(simf.u.data)
        always_nan!(s, Δt) = (fill!(s.tu.data, NaN); nothing)
        @test_throws ErrorException step_adaptive!(simf, c; dt = 1e-3,
                                                   body_force! = always_nan!,
                                                   max_retries = 2)
        @test simf.u.data == u_before    # snapshot rollback is bit-exact
        @test simf.t == 0.0
    end

    # =====================================================================
    # Phase 25 — implicit diffusion solver
    # =====================================================================
    @testset "Implicit diffusion — manufactured solution" begin
        g = RegularGrid(; size = (16, 16, 16), extent = (2π, 2π, 2π), halo = (2, 2, 2))
        bcs = FieldBoundaryConditions(:periodic)
        ν, dt = 0.7, 0.3

        # Several modes so the Krylov space is genuinely multidimensional
        # (a single eigenmode would converge in one iteration).
        φ = Field(CellCenter(), g)
        set!(φ, (x, y, z) -> sin(x) * cos(2y) + 0.5sin(3x) * sin(z) - 0.25cos(y) * cos(4z))
        fill_halos!(φ, bcs)

        lap = Field(CellCenter(), g)
        laplacian(lap, φ)
        rhs = Field(CellCenter(), g)
        ii, jj, kk = interior(g)
        @views rhs.data[ii, jj, kk] .= φ.data[ii, jj, kk] .- dt * ν .* lap.data[ii, jj, kk]

        sol = Field(CellCenter(), g)          # zero initial guess
        stats = solve_implicit_diffusion!(sol, rhs, ν, dt, g, bcs; tol = 1e-13, maxit = 500)
        err = maximum(abs, interior_view(sol) .- interior_view(φ))

        @test stats.converged
        @test err < 1e-8
        @test stats.relative_residual <= stats.tol
        println("  [implicit] manufactured: $(stats.iterations) CG iterations, " *
                "‖r‖ = $(stats.residual), ‖r‖/‖b‖ = $(stats.relative_residual), " *
                "max error = $(err)")

        # per-cell ν (the SGS path) must reproduce the constant-ν answer
        νarr_full = fill(ν, size(φ.data))
        sol2 = Field(CellCenter(), g)
        st2 = solve_implicit_diffusion!(sol2, rhs, νarr_full, dt, g, bcs;
                                        tol = 1e-13, maxit = 500)
        @test maximum(abs, interior_view(sol2) .- interior_view(sol)) < 1e-12
        # interior-sized array is accepted too
        νarr_int = fill(ν, g.Nx, g.Ny, g.Nz)
        sol3 = Field(CellCenter(), g)
        solve_implicit_diffusion!(sol3, rhs, νarr_int, dt, g, bcs; tol = 1e-13, maxit = 500)
        @test maximum(abs, interior_view(sol3) .- interior_view(sol)) < 1e-12
        @test_throws ArgumentError solve_implicit_diffusion!(sol3, rhs, fill(ν, 3, 3, 3),
                                                             dt, g, bcs)
        println("  [implicit] per-cell ν: $(st2.iterations) iterations, matches scalar ν")

        # variable ν is a genuinely different operator: verify against it directly
        νvar = Field(CellCenter(), g)
        set!(νvar, (x, y, z) -> 0.3 + 0.2sin(x) + 0.1cos(z))
        fill_halos!(νvar, bcs)
        rhs_v = Field(CellCenter(), g)
        Aeolus.TimeIntegration._diffusion!(lap, φ, νvar.data, g)      # ∇·(ν∇φ)
        @views rhs_v.data[ii, jj, kk] .= φ.data[ii, jj, kk] .- dt .* lap.data[ii, jj, kk]
        sol4 = Field(CellCenter(), g)
        st4 = solve_implicit_diffusion!(sol4, rhs_v, νvar.data, dt, g, bcs;
                                        tol = 1e-13, maxit = 500)
        @test st4.converged
        @test maximum(abs, interior_view(sol4) .- interior_view(φ)) < 1e-8
        println("  [implicit] variable ν(x,z): $(st4.iterations) iterations, " *
                "max error = $(maximum(abs, interior_view(sol4) .- interior_view(φ)))")

        # inhomogeneous BC types are rejected with an explanation
        bad = FieldBoundaryConditions(Periodic(), Periodic(), Periodic(), Periodic(),
                                      ValueBC(1.0), ValueBC(1.0))
        @test_throws ArgumentError solve_implicit_diffusion!(sol, rhs, ν, dt, g, bad)
        @test_throws ArgumentError solve_implicit_diffusion!(sol, rhs, ν, dt, g, bcs;
                                                             precondition = :chebyshev)
    end

    @testset "Implicit diffusion — walls, and w pinned at the wall face" begin
        g = RegularGrid(; size = (8, 8, 12), extent = (1.0, 1.0, 1.0), halo = (2, 2, 2))
        wbcs = FieldBoundaryConditions(:periodic, :periodic, :wall)
        ν, dt = 0.4, 0.25
        ii, jj, kk = interior(g)

        # cell-centered (mirror / Neumann in z)
        φ = Field(CellCenter(), g)
        set!(φ, (x, y, z) -> sin(2π * x) * cos(2π * z) + 0.3cos(2π * y))
        fill_halos!(φ, wbcs)
        lap = Field(CellCenter(), g); laplacian(lap, φ)
        rhs = Field(CellCenter(), g)
        @views rhs.data[ii, jj, kk] .= φ.data[ii, jj, kk] .- dt * ν .* lap.data[ii, jj, kk]
        sol = Field(CellCenter(), g)
        st = solve_implicit_diffusion!(sol, rhs, ν, dt, g, wbcs; tol = 1e-13, maxit = 500)
        @test st.converged
        @test maximum(abs, interior_view(sol) .- interior_view(φ)) < 1e-8

        # w @ FaceZ: the wall faces are Dirichlet-zero rows, eliminated from the system
        w = Field(FaceZ(), g)
        set!(w, (x, y, z) -> sin(2π * z) * cos(2π * x))
        w.data[:, :, g.Hz + 1] .= 0                     # bottom wall face
        fill_halos!(w, wbcs)
        lapw = Field(FaceZ(), g); laplacian(lapw, w)
        rhsw = Field(FaceZ(), g)
        @views rhsw.data[ii, jj, kk] .= w.data[ii, jj, kk] .- dt * ν .* lapw.data[ii, jj, kk]
        rhsw.data[:, :, g.Hz + 1] .= 0
        solw = Field(FaceZ(), g)
        stw = solve_implicit_diffusion!(solw, rhsw, ν, dt, g, wbcs; tol = 1e-13, maxit = 500)
        @test stw.converged
        @test maximum(abs, interior_view(solw) .- interior_view(w)) < 1e-8
        @test all(iszero, solw.data[:, :, g.Hz + 1])    # wall face stays exactly zero
        println("  [implicit] walls: cc $(st.iterations) iterations, " *
                "w@FaceZ $(stw.iterations) iterations")
    end

    @testset "Implicit diffusion is unconditionally stable" begin
        g = RegularGrid(; size = (24, 24, 24), extent = (1.0, 1.0, 1.0), halo = (2, 2, 2))
        bcs = FieldBoundaryConditions(:periodic)
        ν = 0.1
        # forward-Euler stability limit of the 7-point Laplacian
        dt_explicit = 0.5 / (ν * (1 / g.dx^2 + 1 / g.dy^2 + 1 / g.dz^2))
        dt = 100 * dt_explicit

        φ = Field(CellCenter(), g)
        set!(φ, (x, y, z) -> exp(-((x - 0.5)^2 + (y - 0.5)^2 + (z - 0.5)^2) / 0.01))
        rhs = Field(CellCenter(), g)
        ws  = DiffusionWorkspace(φ)
        ii, jj, kk = interior(g)

        mass0 = sum(interior_view(φ))
        peak  = maximum(interior_view(φ))
        peak0 = peak
        iters = Int[]
        for n in 1:20                       # backward Euler at 100× the explicit limit
            @views rhs.data[ii, jj, kk] .= φ.data[ii, jj, kk]
            st = solve_implicit_diffusion!(φ, rhs, ν, dt, g, bcs;
                                           tol = 1e-14, maxit = 1000, workspace = ws)
            @test st.converged
            push!(iters, st.iterations)
            p = maximum(interior_view(φ))
            @test p < peak                  # peak amplitude decays monotonically
            @test all(isfinite, interior_view(φ))
            @test minimum(interior_view(φ)) >= 0      # positivity preserved
            peak = p
        end
        @test peak < peak0
        @test peak <= peak0                  # bounded — no growth at any step
        @test abs(sum(interior_view(φ)) - mass0) < 1e-10 * max(1.0, abs(mass0))
        println("  [implicit] 20 backward-Euler steps at dt = 100·dt_explicit " *
                "($(dt) vs $(dt_explicit)): $(minimum(iters))–$(maximum(iters)) CG " *
                "iterations/step, peak $(peak0) → $(peak), mass drift " *
                "$(abs(sum(interior_view(φ)) - mass0))")
    end

    # =====================================================================
    # Phase 25 — IMEX schemes
    # =====================================================================
    @testset "IMEX tableaux satisfy their order conditions" begin
        for scheme in (ARK2(), Rosenbrock2())
            @test scheme isa IMEXScheme
            @test Aeolus.TimeIntegration.order(scheme) == 2
            @test Aeolus.TimeIntegration.stages(scheme) == 2
        end
        a = ARK2()
        # implicit half 2nd-order + L-stable  ⇔  γ² - 2γ + 1/2 = 0
        @test abs(a.γ^2 - 2a.γ + 0.5) < 1e-15
        # explicit half 2nd-order            ⇔  γ(1 - δ) = 1/2
        @test abs(a.γ * (1 - a.δ) - 0.5) < 1e-15
        @test a.a_implicit == (a.γ, 1 - a.γ, a.γ)
        @test a.a_explicit == (a.γ, a.δ, 1 - a.δ)
        r = Rosenbrock2()
        @test abs(r.γ^2 - 2r.γ + 0.5) < 1e-15      # R(∞) = 0
        @test r.b == (1.5, 0.5) && r.a21 == 1.0 && r.c21 == -2.0
        @test ARK2(Float32) isa ARK2{Float32}
        @test Rosenbrock2(Float32) isa Rosenbrock2{Float32}
    end

    @testset "IMEX order of accuracy (exact single-mode diffusion)" begin
        # θ diffuses with u ≡ 0, so the semi-discrete problem is exactly
        #     dθ/dt = λ_h θ,   λ_h = -ν (2 - 2cos(k Δx)) / Δx²
        # for the k = 1 Fourier mode. Comparing against exp(λ_h t) isolates
        # the TEMPORAL error (using the continuum -νk² would add a constant
        # O(Δx²) spatial offset and flatten the measured order).
        g = RegularGrid(; size = (16, 4, 4), extent = (2π, 1.0, 1.0), halo = (2, 2, 2))
        ν = 0.05
        λ_h = -ν * (2 - 2cos(g.dx)) / g.dx^2
        T_end = 1.0
        xs = xnodes(g, CellCenter())
        exact = [exp(λ_h * T_end) * sin(x) for x in xs]

        for (name, scheme) in (("ARK2", ARK2()), ("Rosenbrock2", Rosenbrock2()))
            errs = Float64[]
            for nsteps in (8, 16, 32)
                sim = BoussinesqSimulation(g; ν = ν)
                set!(sim.θ, (x, y, z) -> sin(x))
                dt = T_end / nsteps
                for _ in 1:nsteps
                    step_imex!(sim, dt, scheme; tol = 1e-14, maxit = 500)
                end
                @test sim.t ≈ T_end
                push!(errs, maximum(abs, interior_view(sim.θ)[:, 1, 1] .- exact))
            end
            orders = [log2(errs[i] / errs[i+1]) for i in 1:length(errs)-1]
            @test all(o -> abs(o - 2) < 0.15, orders)
            @test errs[end] < errs[1]
            println("  [imex] $name errors = $(errs)")
            println("  [imex] $name measured orders = $(orders)")
        end
    end

    @testset "IMEX preserves the divergence-free constraint" begin
        g = RegularGrid(; size = (16, 16, 16), extent = (2π, 2π, 2π), halo = (2, 2, 2))
        for scheme in (ARK2(), Rosenbrock2())
            sim = BoussinesqSimulation(g; ν = 0.05, θ₀ = 300.0, f_coriolis = 1e-4)
            set!(sim.u, (x, y, z) -> sin(x) * cos(y) * cos(z))
            set!(sim.v, (x, y, z) -> -cos(x) * sin(y) * cos(z))
            set!(sim.θ, (x, y, z) -> 300.0 + 0.5sin(z))
            d = cfl_diagnostics(sim, CFLConstraints())
            dt = 0.5 * d.dt_limiting
            for _ in 1:3
                step_imex!(sim, dt, scheme; tol = 1e-12, maxit = 500)
                @test max_divergence(sim) < 1e-10
            end
            @test all(isfinite, interior_view(sim.u))
            println("  [imex] $(nameof(typeof(scheme))) max|∇·u| after 3 steps = " *
                    "$(max_divergence(sim))")
        end
    end

    @testset "IMEX with walls, SGS and a body force stays sane" begin
        g = RegularGrid(; size = (12, 12, 12), extent = (1.0, 1.0, 1.0), halo = (2, 2, 2))
        bcs = FieldBoundaryConditions(:periodic, :periodic, :wall)
        sim = BoussinesqSimulation(g; ν = 1e-3, C_s = 0.16, Pr_t = 1/3,
                                   θ₀ = 300.0, bcs = bcs)
        set!(sim.u, (x, y, z) -> sin(2π * x) * cos(2π * y) * sin(π * z))
        set!(sim.v, (x, y, z) -> -cos(2π * x) * sin(2π * y) * sin(π * z))
        set!(sim.θ, (x, y, z) -> 300.0 + 2z)

        nudge_calls = Ref(0)
        function nudge!(s, Δt)
            nudge_calls[] += 1
            ii, jj, kk = interior(s.mesh)
            @views s.tu.data[ii, jj, kk] .+= 0.01
            return nothing
        end
        post_calls = Ref(0)
        post!(s, Δt) = (post_calls[] += 1; nothing)

        ws = imex_workspace(sim)
        d = cfl_diagnostics(sim, CFLConstraints())
        dt = 0.5 * d.dt_limiting
        for _ in 1:2
            step_imex!(sim, dt, ARK2(); body_force! = nudge!, post_predictor! = post!,
                       workspace = ws, tol = 1e-12)
        end
        @test nudge_calls[] == 4          # two explicit evaluations per step
        @test post_calls[] == 4           # one per stage
        @test max_divergence(sim) < 1e-10
        @test all(isfinite, interior_view(sim.u))
        @test all(iszero, sim.w.data[:, :, g.Hz + 1])   # wall stays impermeable
        @test ws.total_iterations > 0
        @test ws.max_residual <= 1e-12
        println("  [imex] walls+SGS: $(ws.total_iterations) CG iterations in the " *
                "last step, max relative residual $(ws.max_residual)")
    end

    @testset "IMEX agrees with the explicit driver inside its stability region" begin
        g = RegularGrid(; size = (16, 16, 16), extent = (2π, 2π, 2π), halo = (2, 2, 2))
        ν = 0.05
        function fresh()
            s = BoussinesqSimulation(g; ν = ν)
            set!(s.u, (x, y, z) -> sin(x) * cos(y))
            set!(s.v, (x, y, z) -> -cos(x) * sin(y))
            set!(s.θ, (x, y, z) -> sin(z))
            return s
        end
        ref = fresh()
        d = cfl_diagnostics(ref, CFLConstraints())
        dt = 0.2 * min(d.dt_adv, d.dt_diff)      # comfortably explicit-stable

        for scheme in (ARK2(), Rosenbrock2())
            a = fresh(); b = fresh()
            step_imex!(a, dt, scheme; tol = 1e-12, maxit = 500)
            step!(b, dt)
            scale  = maximum(abs, interior_view(b.u))
            du = maximum(abs, interior_view(a.u) .- interior_view(b.u))
            dθ = maximum(abs, interior_view(a.θ) .- interior_view(b.θ))
            # step! is 1st order, the IMEX schemes 2nd: the difference is the
            # O(dt²) truncation term, i.e. ~dt²·|u_tt| ≪ dt·|u|.
            @test du < 1e-3 * scale
            @test dθ < 1e-3
            @test du > 0                         # they are genuinely different schemes
            @test du < 50 * dt^2 * scale
            println("  [imex] $(nameof(typeof(scheme))) vs step! at dt = $(dt): " *
                    "Δu = $(du), Δθ = $(dθ), dt² = $(dt^2), |u|max = $(scale)")
        end
    end

    @testset "imex_stability_limit quantifies the removed Δz² constraint" begin
        # A wall-resolving grid: Δz ≪ Δx and a large eddy viscosity make the
        # diffusive limit the binding one for a fully explicit step.
        g = RegularGrid(; size = (16, 16, 32), extent = (200.0, 200.0, 20.0),
                        halo = (2, 2, 2))
        sim = BoussinesqSimulation(g; ν = 2.0)
        set!(sim.u, (x, y, z) -> 5.0 + sin(2π * x / 200))
        set!(sim.v, 1.0)

        c = CFLConstraints()
        lim = imex_stability_limit(sim, ARK2(); constraints = c)
        d = cfl_diagnostics(sim, c)

        @test lim.dt_explicit ≈ min(d.dt_adv, d.dt_buoy, d.dt_cor)
        @test lim.dt_diffusive ≈ min(d.dt_diff, d.dt_sgs)
        @test lim.dt_fully_explicit ≈ min(lim.dt_explicit, lim.dt_diffusive)
        @test lim.dt_diffusive < lim.dt_explicit          # diffusion binds explicitly
        @test lim.explicit_limiter == :diffusive
        @test lim.limiter == :advective
        @test lim.speedup ≈ lim.dt_explicit / lim.dt_diffusive
        @test lim.speedup > 1

        # ... and the IMEX step really is stable at the larger dt.
        dt = lim.dt_explicit
        ws = imex_workspace(sim)
        for _ in 1:5
            step_imex!(sim, dt, ARK2(); workspace = ws, tol = 1e-10)
        end
        @test all(isfinite, interior_view(sim.u))
        @test max_divergence(sim) < 1e-10
        @test maximum(abs, interior_view(sim.u)) < 20     # no runaway
        println("  [imex] Δz = $(g.dz) m, ν = 2: dt_explicit = $(lim.dt_explicit) s, " *
                "dt_diffusive = $(lim.dt_diffusive) s, speed-up = $(lim.speedup)×")
    end

    @testset "adaptive_dt(...; scheme = ARK2()) drops the diffusive limit (#133)" begin
        # Same diffusion-bound grid: an explicit step is limited by Δz²/ν,
        # an IMEX step only by advection / buoyancy / Coriolis.
        g = RegularGrid(; size = (16, 16, 32), extent = (200.0, 200.0, 20.0),
                        halo = (2, 2, 2))
        sim = BoussinesqSimulation(g; ν = 2.0)
        set!(sim.u, (x, y, z) -> 5.0 + sin(2π * x / 200))
        set!(sim.v, 1.0)
        c = CFLConstraints()
        lim = imex_stability_limit(sim, ARK2(); constraints = c)
        dt_ex = adaptive_dt(sim, c)
        dt_im = adaptive_dt(sim, c; scheme = ARK2())
        @test dt_ex ≈ lim.dt_fully_explicit
        @test dt_im ≈ lim.dt_explicit
        @test dt_im / dt_ex ≈ lim.speedup
        @test dt_im > 5 * dt_ex                         # ν = 2 on Δz = 0.625 m
        # growth limiting and clamping still apply
        @test adaptive_dt(sim, c; scheme = ARK2(), dt_prev = dt_ex) ≈ 1.1 * dt_ex
        c2 = CFLConstraints(dt_max = dt_ex)
        @test adaptive_dt(sim, c2; scheme = ARK2()) == dt_ex
        # step_adaptive! with the scheme picks the IMEX limit and is stable there
        ke0 = kinetic_energy(sim)
        for _ in 1:3
            _, dt, nretry = step_adaptive!(sim, c; scheme = ARK2())
            @test nretry == 0
            @test dt > 5 * dt_ex
        end
        @test max_divergence(sim) < 1e-10
        @test kinetic_energy(sim) < ke0                 # viscous decay only
        println("  [adaptive_dt] ν = 2, Δz = $(g.dz) m: explicit dt = $(dt_ex) s, " *
                "IMEX dt = $(dt_im) s ($(round(dt_im / dt_ex; digits = 1))×)")
    end

    @testset "step_imex! takes step!'s operator keywords (#133)" begin
        # A driver passes the same keywords to step! and step_imex!: the
        # momentum-advection operator and the eddy-viscosity closure. The
        # defaults are the built-in ones, so passing them explicitly is
        # bit for bit the default step; a different operator / closure is
        # actually used.
        g = RegularGrid(; size = (12, 12, 12), extent = (120.0, 120.0, 60.0),
                        halo = (2, 2, 2))
        function mk()
            sim = BoussinesqSimulation(g; ν = 0.5, C_s = 0.18, θ₀ = 300.0)
            set!(sim.u, (x, y, z) -> 2.0 + sin(2π * x / 120) * cos(2π * z / 60))
            set!(sim.v, (x, y, z) -> cos(2π * y / 120))
            set!(sim.θ, (x, y, z) -> 300.0 + 0.01z + 0.1sin(2π * x / 120))
            return sim
        end
        state(s) = (copy(interior_view(s.u)), copy(interior_view(s.v)),
                    copy(interior_view(s.w)), copy(interior_view(s.θ)))
        C_s = 0.18
        smag!(νt, u, v, w) = Aeolus.Operators.compute_smagorinsky_νt!(νt, u, v, w, C_s)
        for scheme in (ARK2(), Rosenbrock2())
            ref = mk(); step_imex!(ref, 2.0, scheme)
            a = mk(); step_imex!(a, 2.0, scheme; momentum_advection = Aeolus.Operators.momentum_advect!,
                                 eddy_viscosity! = smag!)
            @test state(a) == state(ref)                     # bitwise
            # no momentum advection: u changes differently
            b = mk(); step_imex!(b, 2.0, scheme; momentum_advection = (tu, tv, tw, u, v, w) -> nothing)
            @test maximum(abs.(interior_view(b.u) .- interior_view(ref.u))) > 1e-3
            # a closure that doubles νt is the one frozen into ν_eff
            calls = Ref(0)
            dbl!(νt, u, v, w) = (calls[] += 1; smag!(νt, u, v, w); interior_view(νt) .*= 2; nothing)
            c = mk(); step_imex!(c, 2.0, scheme; eddy_viscosity! = dbl!)
            @test calls[] == 1                                # once per step (frozen)
            @test maximum(interior_view(c.νt)) ≈ 2 * maximum(interior_view(ref.νt))
            @test interior_view(c.θ) != interior_view(ref.θ)
            @test max_divergence(c) < 1e-10
        end
    end

    # =====================================================================
    # The two phases together
    # =====================================================================
    @testset "step_adaptive! drives step_imex! over many steps" begin
        g = RegularGrid(; size = (16, 16, 16), extent = (2π, 2π, 2π), halo = (2, 2, 2))
        sim = BoussinesqSimulation(g; ν = 0.02, θ₀ = 300.0)
        set!(sim.u, (x, y, z) -> sin(x) * cos(y))
        set!(sim.v, (x, y, z) -> -cos(x) * sin(y))
        set!(sim.θ, (x, y, z) -> 300.0 + 0.1sin(z))

        c = CFLConstraints(growth_max = 1.1, dt_min = 1e-6, dt_max = 0.5)
        ke0 = kinetic_energy(sim)
        dt = 1e-3
        dts = Float64[]
        for _ in 1:15
            _, dt, nretry = step_adaptive!(sim, c; dt_prev = dt, scheme = ARK2())
            @test nretry == 0
            push!(dts, dt)
        end
        @test all(i -> dts[i+1] <= 1.1 * dts[i] * (1 + 1e-12), 1:length(dts)-1)
        @test sim.t ≈ sum(dts)
        @test max_divergence(sim) < 1e-10
        @test kinetic_energy(sim) < ke0        # viscous decay, no energy injection
        @test all(isfinite, interior_view(sim.u))
        println("  [combined] 15 adaptive IMEX steps: dt $(dts[1]) → $(dts[end]), " *
                "t = $(sim.t), KE $(ke0) → $(kinetic_energy(sim))")
    end
    # KA-c (#295): step_rk3! keeps uⁿ in a cached RK3Workspace. The step is
    # bit-identical to the former copy-based one (reproduced here with the
    # same substep and combinations) and allocates only the cache lookup.
    @testset "KA-c: step_rk3! with a cached workspace" begin
        S = Aeolus.Simulations
        mk() = begin
            g = RegularGrid(; size = (16, 16, 16), extent = (1.0, 1.0, 1.0), halo = (2, 2, 2))
            s = BoussinesqSimulation(g; ν = 1e-3, θ₀ = 300.0, g_accel = 9.81, C_s = 0.17,
                                     moisture = true)
            set!(s.u, (x, y, z) -> sin(2π * x) * cos(2π * y))
            set!(s.v, (x, y, z) -> -cos(2π * x) * sin(2π * y))
            set!(s.θ, (x, y, z) -> 300.0 + 0.1 * sin(2π * z))
            set!(s.qv, (x, y, z) -> 0.008 + 1e-4 * cos(2π * x))
            s
        end
        # the pre-#295 step_rk3! for a closed box: copies of uⁿ, three substeps
        function rk3_reference!(sim, dt)
            closure = S._active_sgs(sim, nothing, nothing)
            t0 = sim.t
            d = (sim.u.data, sim.v.data, sim.w.data, sim.θ.data, sim.qv.data)
            d0 = map(copy, d)
            S._rk3_substep!(sim, dt, closure, nothing, nothing, :laplacian, t0 + dt)
            S._rk3_substep!(sim, dt, closure, nothing, nothing, :laplacian, t0 + 2dt)
            for (a, a0) in zip(d, d0); a .= 0.75 .* a0 .+ 0.25 .* a; end
            S._rk3_substep!(sim, dt, closure, nothing, nothing, :laplacian, t0 + 3dt / 2)
            inv3 = 1 / 3; two3 = 2 * inv3
            for (a, a0) in zip(d, d0); a .= inv3 .* a0 .+ two3 .* a; end
            sim.t -= 2dt
            closure === nothing || S.advance_sgs!(closure, sim, dt)
            return sim
        end
        a, b = mk(), mk()
        for _ in 1:3
            step_rk3!(a, 2e-3); rk3_reference!(b, 2e-3)
        end
        @test all(f -> getproperty(a, f).data == getproperty(b, f).data, (:u, :v, :w, :θ, :qv))
        @test a.t == b.t
        @test S.rk3_workspace(a) === S.rk3_workspace(a)          # cached, one per simulation
        nb = @allocated step_rk3!(a, 2e-3)
        println("  [KA-c] one step_rk3! (16³, SGS, moist) allocates $nb bytes")
        @test nb <= 64
    end
end

