# =========================================================================
# Era II — Phases 23 & 25: adaptive time stepping and semi-implicit IMEX.
# =========================================================================

using Test
using Aeolus
using Aeolus.TimeIntegration
using Aeolus.Fields: set!, interior_view
using Aeolus.Grids: interior, xnodes, znodes

@testset "Era II — Adaptive dt & IMEX" begin

    @testset "CFL diagnostics: advective, diffusive, buoyancy limits" begin
        g = RegularGrid(; size = (8, 8, 8), extent = (1.0, 2.0, 4.0), halo = (2, 2, 2))
        sim = BoussinesqSimulation(g)
        U, V, W = 1.0, -2.0, 0.5
        set!(sim.u, U); set!(sim.v, V); set!(sim.w, W)
        c = CFLConstraints()
        d = cfl_diagnostics(sim, c)
        # multidimensional CFL: the per-cell SUM of the directional rates
        @test d.dt_adv ≈ c.C_advective / (abs(U) / g.dx + abs(V) / g.dy + abs(W) / g.dz) rtol = 1e-12
        @test d.limiter == :advective && d.dt_diff == Inf

        simν = BoussinesqSimulation(g; ν = 10.0)
        set!(simν.u, 0.01)
        dν = cfl_diagnostics(simν, c)
        @test dν.dt_diff ≈ c.C_diffusive / (1 / g.dx^2 + 1 / g.dy^2 + 1 / g.dz^2) / 10.0 rtol = 1e-12
        @test dν.limiter == :diffusive

        gb = RegularGrid(; size = (4, 4, 16), extent = (100.0, 100.0, 400.0), halo = (2, 2, 2))
        simb = BoussinesqSimulation(gb; θ₀ = 300.0, f_coriolis = 1e-4)
        set!(simb.θ, (x, y, z) -> 300.0 + 0.01z)
        set!(simb.u, 0.001)
        db = cfl_diagnostics(simb, c)
        @test db.N ≈ sqrt(9.81 / 300.0 * 0.01) rtol = 1e-12
        @test db.limiter == :buoyancy
    end

    @testset "adaptive_dt growth limiting, clamping, NaN guard" begin
        g = RegularGrid(; size = (8, 8, 8), extent = (1.0, 1.0, 1.0), halo = (2, 2, 2))
        sim = BoussinesqSimulation(g)
        set!(sim.u, 0.1)
        c = CFLConstraints(growth_max = 1.1, dt_min = 1e-8, dt_max = 10.0)
        lim = cfl_diagnostics(sim, c).dt_limiting
        history = accumulate((dt, _) -> adaptive_dt(sim, c; dt_prev = dt), 1:120; init = 1e-4)
        hist = [1e-4; history]
        @test maximum(hist[i+1] / hist[i] for i in 1:length(hist)-1) <= 1.1 * (1 + 1e-12)
        @test hist[end] ≈ lim rtol = 1e-12          # saturates at the CFL limit, no overshoot
        @test adaptive_dt(sim, CFLConstraints(dt_max = 0.01)) == 0.01
        set!(sim.u, NaN)
        @test adaptive_dt(sim, c) == c.dt_min      # a broken state gives no NaN dt
    end

    @testset "step_adaptive!: substepping and retries" begin
        g = RegularGrid(; size = (8, 8, 8), extent = (1.0, 1.0, 1.0), halo = (2, 2, 2))
        sim = BoussinesqSimulation(g)
        set!(sim.u, 1.0)                      # uniform ⇒ advection and ∇·u vanish
        micro_dts = Float64[]
        procs = (microphysics = (s, Δt) -> (push!(micro_dts, Δt); nothing),)
        _, dt_used, _ = step_adaptive!(sim, CFLConstraints(dt_max = 0.05); fast_processes = procs,
                                       schedule = SubsteppingSchedule(microphysics = 4))
        @test length(micro_dts) == 4 && all(≈(dt_used / 4), micro_dts)

        # stiff explicit growth: one Euler step multiplies u by (1 + Δt·G)
        sim = BoussinesqSimulation(g)
        set!(sim.u, 1.0)
        G = 1.0e4
        function blowup!(s, Δt)
            ii, jj, kk = interior(s.mesh)
            @views s.tu.data[ii, jj, kk] .+= G .* s.u.data[ii, jj, kk]
            return nothing
        end
        c = CFLConstraints(dt_min = 1e-8, dt_max = 1.0)
        # dt = 1e-3 ⇒ |u| = 11 (reject), 5e-4 ⇒ 6 (reject), 2.5e-4 ⇒ 3.5 (accept)
        _, dt_used, nretry = step_adaptive!(sim, c; dt = 1e-3, body_force! = blowup!,
                                            max_velocity = 4.0, max_retries = 3)
        @test nretry == 2 && dt_used ≈ 1e-3 / 4
        @test maximum(abs, interior_view(sim.u)) ≈ 1 + dt_used * G
        @test sim.t ≈ dt_used          # the rejected attempts did not advance the clock
        # exhausting the retries restores the state and throws
        simf = BoussinesqSimulation(g)
        set!(simf.u, 1.0)
        u_before = copy(simf.u.data)
        @test_throws ErrorException step_adaptive!(simf, c; dt = 1e-3, max_retries = 2,
                                                   body_force! = (s, Δt) -> (fill!(s.tu.data, NaN); nothing))
        @test simf.u.data == u_before && simf.t == 0.0     # snapshot copied back
    end

    @testset "implicit diffusion: manufactured solution, unconditional stability" begin
        g = RegularGrid(; size = (8, 8, 8), extent = (2π, 2π, 2π), halo = (2, 2, 2))
        bcs = FieldBoundaryConditions(:periodic)
        ν, dt = 0.7, 0.3
        φ = Field(CellCenter(), g)
        set!(φ, (x, y, z) -> sin(x) * cos(2y) + 0.5sin(3x) * sin(z) - 0.25cos(y) * cos(2z))
        fill_halos!(φ, bcs)
        ii, jj, kk = interior(g)
        # variable ν: (1 - dt ∇·ν∇) φ = rhs solved back to φ
        νvar = Field(CellCenter(), g)
        set!(νvar, (x, y, z) -> 0.3 + 0.2sin(x) + 0.1cos(z))
        fill_halos!(νvar, bcs)
        lap = Field(CellCenter(), g)
        Aeolus.TimeIntegration._diffusion!(lap, φ, νvar.data, g)
        rhs = Field(CellCenter(), g)
        @views rhs.data[ii, jj, kk] .= φ.data[ii, jj, kk] .- dt .* lap.data[ii, jj, kk]
        sol = Field(CellCenter(), g)
        st = solve_implicit_diffusion!(sol, rhs, νvar.data, dt, g, bcs; tol = 1e-13, maxit = 500)
        @test st.converged
        @test maximum(abs, interior_view(sol) .- interior_view(φ)) < 1e-8

        # backward Euler at 100× the explicit limit: peak decays, positivity, mass
        ψ = Field(CellCenter(), g)
        set!(ψ, (x, y, z) -> exp(-((x - π)^2 + (y - π)^2 + (z - π)^2)))
        dt_big = 100 * 0.5 / (0.1 * 3 / g.dx^2)
        mass0, peak0 = sum(interior_view(ψ)), maximum(interior_view(ψ))
        ws = DiffusionWorkspace(ψ)
        for n in 1:3
            @views rhs.data[ii, jj, kk] .= ψ.data[ii, jj, kk]
            solve_implicit_diffusion!(ψ, rhs, 0.1, dt_big, g, bcs; tol = 1e-14, maxit = 1000,
                                      workspace = ws)
        end
        @test maximum(interior_view(ψ)) < peak0 && minimum(interior_view(ψ)) >= 0
        @test abs(sum(interior_view(ψ)) - mass0) < 1e-10 * mass0
    end

    @testset "IMEX: tableau, 2nd-order accuracy, divergence-free" begin
        a = ARK2()
        @test abs(a.γ^2 - 2a.γ + 0.5) < 1e-15      # implicit half 2nd order + L-stable
        @test abs(a.γ * (1 - a.δ) - 0.5) < 1e-15   # explicit half 2nd order
        # single-mode diffusion: dθ/dt = λ_h θ with the discrete eigenvalue,
        # so only the temporal error is measured
        g = RegularGrid(; size = (16, 4, 4), extent = (2π, 1.0, 1.0), halo = (2, 2, 2))
        ν = 0.05
        λ_h = -ν * (2 - 2cos(g.dx)) / g.dx^2
        exact = [exp(λ_h) * sin(x) for x in xnodes(g, CellCenter())]
        for scheme in (ARK2(), Rosenbrock2())
            errs = map((8, 16)) do n
                sim = BoussinesqSimulation(g; ν = ν)
                set!(sim.θ, (x, y, z) -> sin(x))
                for _ in 1:n
                    step_imex!(sim, 1.0 / n, scheme; tol = 1e-14, maxit = 500)
                end
                maximum(abs, interior_view(sim.θ)[:, 1, 1] .- exact)
            end
            @test abs(log2(errs[1] / errs[2]) - 2) < 0.15
        end

        gw = RegularGrid(; size = (8, 8, 8), extent = (1.0, 1.0, 1.0), halo = (2, 2, 2))
        sim = BoussinesqSimulation(gw; ν = 1e-3, C_s = 0.16, θ₀ = 300.0,
                                   bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
        set!(sim.u, (x, y, z) -> sin(2π * x) * cos(2π * y) * sin(π * z))
        set!(sim.v, (x, y, z) -> -cos(2π * x) * sin(2π * y) * sin(π * z))
        set!(sim.θ, (x, y, z) -> 300.0 + 2z)
        dt = 0.5 * cfl_diagnostics(sim, CFLConstraints()).dt_limiting
        for _ in 1:2
            step_imex!(sim, dt, ARK2(); tol = 1e-12)
        end
        @test max_divergence(sim) < 1e-10
        @test all(iszero, sim.w.data[:, :, gw.Hz + 1])   # wall stays impermeable
    end

    @testset "adaptive_dt(...; scheme = ARK2()) drops the diffusive limit (#133)" begin
        g = RegularGrid(; size = (8, 8, 16), extent = (100.0, 100.0, 10.0), halo = (2, 2, 2))
        sim = BoussinesqSimulation(g; ν = 2.0)
        set!(sim.u, (x, y, z) -> 5.0 + sin(2π * x / 100))
        c = CFLConstraints()
        lim = imex_stability_limit(sim, ARK2(); constraints = c)
        dt_ex = adaptive_dt(sim, c)
        dt_im = adaptive_dt(sim, c; scheme = ARK2())
        @test lim.explicit_limiter == :diffusive
        @test dt_ex ≈ lim.dt_fully_explicit && dt_im ≈ lim.dt_explicit
        @test dt_im > 5 * dt_ex
        for _ in 1:2
            step_imex!(sim, dt_im, ARK2(); tol = 1e-10)
        end
        @test max_divergence(sim) < 1e-10 && maximum(abs, interior_view(sim.u)) < 7
    end

    @testset "KA-b (#296): step guard maxima and NaN propagation" begin
        TI = Aeolus.TimeIntegration
        g = RegularGrid(; size = (6, 5, 4), extent = (1.0, 2.0, 0.5), halo = (2, 2, 2))
        sim = BoussinesqSimulation(g; θ₀ = 300.0)
        set!(sim.u, (x, y, z) -> sin(5x) + z)
        @test agree(TI.step_guard(sim).speed, maximum(abs, interior_view(sim.u)))
        @test TI.step_guard(sim).finite
        interior_view(sim.v)[2, 3, 2] = NaN
        @test isnan(TI.step_guard(sim).speed) && !TI.step_guard(sim).finite
        interior_view(sim.v)[2, 3, 2] = 0.0
        sim.p.data[1, 1, 1] = NaN                     # a halo NaN is not in the interior
        @test TI.step_guard(sim).finite
    end
end
