using Test
using Random
using Aeolus
using Aeolus.Grids: CellCenter
using Aeolus.Fields: set!, interior_view
using Aeolus.Multigrid

@testset "Era II — Multigrid pressure solver (Phase 29)" begin

    # ---------------------------------------------------------------------
    # Manufactured solution: pick φ, apply the discrete Laplacian to get
    # the rhs, solve, recover φ. Machine-precision recovery is expected
    # because the solver inverts exactly the operator that built the rhs.
    # ---------------------------------------------------------------------
    @testset "Manufactured solution (triply periodic)" begin
        L = 1.0
        g = RegularGrid(; size=(32,32,32), extent=(L,L,L), halo=(1,1,1))
        φ   = Field(CellCenter(), g)
        src = Field(CellCenter(), g)
        out = Field(CellCenter(), g)
        set!(φ, (x,y,z) -> sin(2π*x/L) * cos(2π*y/L) * sin(2π*z/L))
        fill_halos!(φ, FieldBoundaryConditions(:periodic))
        laplacian(src, φ)

        mg = MultigridPoissonSolver(g; bc_x=:periodic, bc_y=:periodic,
                                       bc_z=:periodic, tol=1e-12,
                                       maxcycles=60)
        solve!(out, src, mg)

        # φ has zero mean, so it is the unique zero-mean solution
        err = maximum(abs.(interior_view(out) .- interior_view(φ)))
        @test err < 1e-9
        @test mg_residual_norm(mg) < 1e-11
        @test mg_level_count(mg) >= 3
    end

    # ---------------------------------------------------------------------
    # The headline property: iteration count must NOT grow with grid size.
    # CG on the Poisson operator scales like O(N); multigrid is O(1).
    # ---------------------------------------------------------------------
    @testset "Grid-size independence of cycle count" begin
        cycles = Int[]
        sizes  = (16, 32, 64)
        for N in sizes
            g = RegularGrid(; size=(N,N,N), extent=(1.0,1.0,1.0), halo=(1,1,1))
            Random.seed!(1234)
            rhs = randn(N, N, N)
            rhs .-= sum(rhs) / length(rhs)          # compatibility condition
            out = zeros(N, N, N)
            mg = MultigridPoissonSolver(g; bc_x=:periodic, bc_y=:periodic,
                                           bc_z=:periodic, tol=1e-8,
                                           maxcycles=60)
            res = mg_solve!(mg, out, rhs)
            @test res.converged
            push!(cycles, res.cycles)
        end
        # 4x refinement in each direction (64x more cells) must not blow the
        # cycle count up; allow a small drift but nothing like O(N).
        @test maximum(cycles) <= minimum(cycles) + 3
        @info "Multigrid cycles vs grid size" sizes cycles
    end

    # ---------------------------------------------------------------------
    # Channel BCs: periodic in x, y; Neumann in z (the pressure BC against
    # impermeable walls). Compare against the exact cosine-FFT solver.
    # ---------------------------------------------------------------------
    @testset "Channel BCs agree with CosineFFTPoissonSolver" begin
        L = 1.0
        g = RegularGrid(; size=(32,32,32), extent=(L,L,L), halo=(1,1,1))
        φ = Field(CellCenter(), g)
        src = Field(CellCenter(), g)
        p_fft = Field(CellCenter(), g)
        p_mg  = Field(CellCenter(), g)
        n = 2
        set!(φ, (x,y,z) -> cos(2π*x/L) * cos(2π*y/L) * cos(π*n*z/L))
        fill_halos!(φ, FieldBoundaryConditions(:periodic, :periodic, :noflux))
        laplacian(src, φ)

        solve!(p_fft, src, CosineFFTPoissonSolver(g))
        mg = MultigridPoissonSolver(g; bc_x=:periodic, bc_y=:periodic,
                                       bc_z=:neumann, tol=1e-12, maxcycles=80)
        solve!(p_mg, src, mg)

        # both are zero-mean solutions of the same singular system
        a = interior_view(p_mg) .- sum(interior_view(p_mg))/length(interior_view(p_mg))
        b = interior_view(p_fft) .- sum(interior_view(p_fft))/length(interior_view(p_fft))
        err = maximum(abs.(a .- b))
        @test err < 1e-8
    end

    # ---------------------------------------------------------------------
    # Residual must fall geometrically — the defining behaviour of a
    # working V-cycle. A stalled cycle (bad transfer operators, missing
    # null-space projection) shows up immediately here.
    # ---------------------------------------------------------------------
    @testset "Geometric convergence rate" begin
        N = 32
        g = RegularGrid(; size=(N,N,N), extent=(1.0,1.0,1.0), halo=(1,1,1))
        Random.seed!(7)
        rhs = randn(N,N,N); rhs .-= sum(rhs)/length(rhs)
        out = zeros(N,N,N)
        mg = MultigridPoissonSolver(g; bc_x=:periodic, bc_y=:periodic,
                                       bc_z=:neumann, tol=1e-12, maxcycles=40)
        mg_solve!(mg, out, rhs)
        h = mg_iteration_history(mg)
        @test length(h) >= 3
        # per-cycle reduction factor, measured over the geometric phase
        ratios = h[2:min(6,end)] ./ h[1:min(5,end-1)]
        @test all(r -> r < 0.5, ratios)     # at least 2x per cycle
        @info "Multigrid convergence factors" ratios
    end

    # ---------------------------------------------------------------------
    # W-cycle and full-multigrid variants must also converge.
    # ---------------------------------------------------------------------
    @testset "W-cycle and FMG variants" begin
        N = 32
        g = RegularGrid(; size=(N,N,N), extent=(1.0,1.0,1.0), halo=(1,1,1))
        Random.seed!(99)
        rhs = randn(N,N,N); rhs .-= sum(rhs)/length(rhs)

        for (cyc, use_fmg) in ((:V, false), (:W, false), (:V, true))
            out = zeros(N,N,N)
            mg = MultigridPoissonSolver(g; bc_x=:periodic, bc_y=:periodic,
                                           bc_z=:neumann, tol=1e-9,
                                           maxcycles=40, cycle=cyc, fmg=use_fmg)
            res = mg_solve!(mg, out, rhs)
            @test res.converged
            @info "variant" cycle=cyc fmg=use_fmg cycles=res.cycles
        end
    end

    # ---------------------------------------------------------------------
    # Variable-coefficient operator ∇·(c∇φ). Build the rhs by applying the
    # same variable-coefficient stencil, then check recovery. This is the
    # path cut-cell and anelastic solves will use.
    # ---------------------------------------------------------------------
    @testset "Variable-coefficient operator" begin
        N = 16
        L = 1.0
        g = RegularGrid(; size=(N,N,N), extent=(L,L,L), halo=(1,1,1))
        # smoothly varying coefficient, factor-3 contrast across the domain
        cx = Array{Float64,3}(undef, N+1, N, N)
        cy = Array{Float64,3}(undef, N, N+1, N)
        cz = Array{Float64,3}(undef, N, N, N+1)
        cfun(z) = 1.0 + 2.0 * z / L
        for k in 1:N, j in 1:N, i in 1:N+1; cx[i,j,k] = cfun((k-0.5)*g.dz); end
        for k in 1:N, j in 1:N+1, i in 1:N; cy[i,j,k] = cfun((k-0.5)*g.dz); end
        for k in 1:N+1, j in 1:N, i in 1:N; cz[i,j,k] = cfun((k-1)*g.dz);   end

        mg = MultigridPoissonSolver(g; bc_x=:periodic, bc_y=:periodic,
                                       bc_z=:neumann, tol=1e-10, maxcycles=80,
                                       coeff_x=cx, coeff_y=cy, coeff_z=cz)

        # manufactured: pick φ, apply the solver's own operator to get rhs
        φ_ex = [sin(2π*(i-0.5)*g.dx/L) * cos(2π*(j-0.5)*g.dy/L) *
                cos(π*(k-0.5)*g.dz/L) for i in 1:N, j in 1:N, k in 1:N]
        φ_ex .-= sum(φ_ex)/length(φ_ex)
        rhs = similar(φ_ex)
        Aeolus.Multigrid._apply_fine!(rhs, φ_ex, mg)
        rhs .-= sum(rhs)/length(rhs)

        out = zeros(N,N,N)
        res = mg_solve!(mg, out, rhs)
        @test res.converged
        out .-= sum(out)/length(out)
        err = maximum(abs.(out .- φ_ex))
        @test err < 1e-7
        @info "variable-coefficient recovery" err cycles=res.cycles
    end

    # ---------------------------------------------------------------------
    # Multigrid-preconditioned CG: the robust configuration. Must converge
    # in far fewer iterations than unpreconditioned CG.
    # ---------------------------------------------------------------------
    @testset "MG-preconditioned CG beats plain CG" begin
        N = 32
        L = 1.0
        g = RegularGrid(; size=(N,N,N), extent=(L,L,L), halo=(1,1,1))
        φ = Field(CellCenter(), g); src = Field(CellCenter(), g)
        set!(φ, (x,y,z) -> sin(2π*x/L)*cos(2π*y/L)*sin(2π*z/L))
        fill_halos!(φ, FieldBoundaryConditions(:periodic))
        laplacian(src, φ)

        p_mgcg = Field(CellCenter(), g)
        mgcg = MGPreconditionedCG(g; bc_x=:periodic, bc_y=:periodic,
                                     bc_z=:periodic, tol=1e-10, maxit=200)
        solve!(p_mgcg, src, mgcg)
        its_mgcg = mg_iteration_history(mgcg)

        p_cg = Field(CellCenter(), g)
        cg = CGPoissonSolver(g; bc_top=:periodic, bc_bottom=:periodic,
                                tol=1e-10, maxit=2000)
        solve!(p_cg, src, cg)

        err_mgcg = maximum(abs.(interior_view(p_mgcg) .- interior_view(φ)))
        @test err_mgcg < 1e-7
        @test its_mgcg < 25          # MG-PCG should need only a handful
        @info "MG-PCG iterations" its_mgcg err_mgcg
    end

    # ---------------------------------------------------------------------
    # The solution of a linear problem scales with its right-hand side, so
    # the iteration must too. Found by the Berlin case (#132): the
    # breakdown test was an absolute `|p·Ap| < eps`, which stopped CG after
    # 6 iterations at a relative residual of 8e-6 for a 1e-6 RHS.
    # ---------------------------------------------------------------------
    @testset "MG-preconditioned CG: convergence independent of the RHS scale" begin
        g = RegularGrid(; size=(16,16,16), extent=(250.0,250.0,100.0), halo=(1,1,1))
        b0 = [sin(0.7i) * cos(1.3j) + 0.1 * sin(0.9k) * i / 16 for i in 1:16, j in 1:16, k in 1:16]
        sols = Dict{Float64,Array{Float64,3}}()
        for scale in (1.0, 1e-3, 1e-6)
            s = MGPreconditionedCG(g; bc_x=:periodic, bc_y=:periodic, bc_z=:neumann, tol=1e-10)
            src = Field(CellCenter(), g); p = Field(CellCenter(), g)
            interior_view(src) .= scale .* b0
            solve!(p, src, s)
            @test mg_residual_norm(s) < 1e-10
            sols[scale] = interior_view(p) ./ scale
        end
        @test sols[1e-6] ≈ sols[1.0] rtol = 1e-8
    end

    # ---------------------------------------------------------------------
    # Interface conformance: MultigridPoissonSolver must be usable as the
    # simulation's pressure solver and keep the projection exact.
    # ---------------------------------------------------------------------
    @testset "Drop-in use inside BoussinesqSimulation" begin
        N = 16
        L = 2π
        mesh = RegularGrid(; size=(N,N,N), extent=(L,L,L), halo=(2,2,2))
        mg = MultigridPoissonSolver(mesh; bc_x=:periodic, bc_y=:periodic,
                                          bc_z=:neumann, tol=1e-11,
                                          maxcycles=60)
        sim = BoussinesqSimulation(mesh; ν=1e-3,
                                   bcs=FieldBoundaryConditions(:periodic,
                                                               :periodic, :wall),
                                   solver=mg)
        Random.seed!(3)
        interior_view(sim.u) .= randn(N,N,N)
        interior_view(sim.v) .= randn(N,N,N)
        interior_view(sim.w) .= randn(N,N,N)
        @test max_divergence(sim) > 0.1          # started divergent
        for _ in 1:3
            step!(sim, 1e-3)
        end
        d = max_divergence(sim)
        @test d < 1e-7
        @test all(isfinite, interior_view(sim.u))
        @info "divergence after MG projection" d
    end

    # ---------------------------------------------------------------------
    # MG-preconditioned CG as the pressure solver of a 100-step run gives
    # the cosine-FFT run (the default for periodic x, y and walls) to 1e-8
    # and keeps the flow divergence-free. Moved from test/berlin_numerics.jl
    # (#320), where it ran on the 16³ Berlin smoke window (250 m × 250 m ×
    # 100 m); here the same box with a warm bubble instead of the city.
    # ---------------------------------------------------------------------
    @testset "MG-preconditioned CG in a 100-step run: the cosine-FFT result to 1e-8" begin
        mesh = RegularGrid(; size=(16,16,16), extent=(250.0,250.0,100.0), halo=(3,3,3))
        function run(solver)
            kw = solver === nothing ? (;) : (solver = solver,)
            sim = BoussinesqSimulation(mesh; ν = 2.0, θ₀ = 300.0,
                                       bcs = FieldBoundaryConditions(:periodic, :periodic, :wall),
                                       θ_bcs = FieldBoundaryConditions(:periodic, :periodic, :noflux),
                                       kw...)
            set!(sim.θ, (x, y, z) -> 300.0 + 0.5 * exp(-((x - 125)^2 + (y - 100)^2 + (z - 30)^2) / 40^2))
            set!(sim.u, (x, y, z) -> 1.0 + 0.2 * sin(2π * y / 250) * sin(π * z / 100))
            for _ in 1:100
                step!(sim, 1.0)
            end
            return sim
        end
        ref = run(nothing)
        @test ref.solver isa CosineFFTPoissonSolver
        mg = run(MGPreconditionedCG(mesh; bc_x = :periodic, bc_y = :periodic, bc_z = :neumann,
                                    tol = 1e-10, maxit = 200))
        d = Dict(f => maximum(abs, interior_view(getfield(mg, f)) .- interior_view(getfield(ref, f)))
                 for f in (:u, :v, :w, :θ))
        div = max_divergence(mg)
        @info "MG-CG vs cosine FFT after 100 steps" d div max_w = maximum(abs, interior_view(mg.w))
        @test maximum(abs, interior_view(mg.w)) > 1e-3     # the bubble moves the flow
        @test all(<(1e-8), values(d))
        @test div < 1e-8
    end

    # ---------------------------------------------------------------------
    # Odd dimensions must degrade gracefully (fewer levels), not error.
    # ---------------------------------------------------------------------
    @testset "Non-power-of-two grids degrade gracefully" begin
        g = RegularGrid(; size=(24,20,12), extent=(1.0,1.0,1.0), halo=(1,1,1))
        mg = MultigridPoissonSolver(g; bc_x=:periodic, bc_y=:periodic,
                                       bc_z=:neumann, tol=1e-8, maxcycles=80)
        @test mg_level_count(mg) >= 1
        Random.seed!(11)
        rhs = randn(24,20,12); rhs .-= sum(rhs)/length(rhs)
        out = zeros(24,20,12)
        res = mg_solve!(mg, out, rhs)
        @test res.converged
        @info "non-power-of-two" levels=mg_level_count(mg) cycles=res.cycles
    end

    # ---------------------------------------------------------------------
    # Zero rhs must give exactly zero, not a NaN from dividing by ‖b‖.
    # ---------------------------------------------------------------------
    @testset "Zero right-hand side" begin
        g = RegularGrid(; size=(16,16,16), extent=(1.0,1.0,1.0), halo=(1,1,1))
        mg = MultigridPoissonSolver(g)
        out = ones(16,16,16)
        res = mg_solve!(mg, out, zeros(16,16,16))
        @test res.converged
        @test all(iszero, out)
    end
    # ---------------------------------------------------------------------
    # KA-c (#295): a solve allocates nothing. Since KA-g (#300) the dot
    # products are the canonical plane sums of the one CG driver `pcg!`
    # instead of `sum(w .= a .* b)` (pairwise): a named one-time change of
    # the last bits, with the same iteration counts (test/ka_pcg.jl). The
    # assertion that the dot equals the pairwise sum is gone with it.
    # ---------------------------------------------------------------------
    @testset "KA-c: allocation-free solves (dot products: KA-g canonical sums)" begin
        g = RegularGrid(; size=(16,16,16), extent=(1.0,1.0,1.0), halo=(2,2,2))
        Random.seed!(295)
        src = Aeolus.Field(CellCenter(), g); interior_view(src) .= randn(16,16,16)
        for S in (Aeolus.PressureSolvers.CGPoissonSolver(g; tol = 1e-10, maxit = 500),
                  MultigridPoissonSolver(g), MGPreconditionedCG(g))
            p = Aeolus.Field(CellCenter(), g)
            Aeolus.PressureSolvers.solve!(p, src, S)
            # the solution: ∇²p = src − mean(src) to the solver tolerance
            b = interior_view(src) .- sum(interior_view(src)) / 16^3
            Lp = similar(b)
            Aeolus.PressureSolvers._apply_laplacian!(Lp, collect(interior_view(p)), g.dx, g.dy, g.dz,
                                                     :neumann, :neumann, :periodic, :periodic)
            r = Lp .- b
            @test sqrt(sum(abs2, r)) <= 1e-6 * sqrt(sum(abs2, b))
            a = @allocated Aeolus.PressureSolvers.solve!(p, src, S)
            @info "KA-c: $(nameof(typeof(S))) solve! allocates $a bytes"
            @test a == 0
        end
        x, y = randn(16,16,16), randn(16,16,16)
        I3 = (1:16, 1:16, 1:16)
        ws = Aeolus.Backends.PCGWorkspace(x)
        @test Aeolus.Backends._pcg_reduce!(ws.s1, Aeolus.Backends.backend(x), I3,
                                           Aeolus.Backends._pcg_dot_body, x, y) === interior_dot(x, y, I3)
    end
end

