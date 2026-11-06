using Test
using Random
using Aeolus
using Aeolus.Grids: CellCenter
using Aeolus.Fields: set!, interior_view
using Aeolus.Multigrid

@testset "Multigrid pressure solver (Phase 29)" begin

    @testset "Manufactured solution; channel BCs agree with the cosine FFT" begin
        g = RegularGrid(; size=(16,16,16), extent=(1.0,1.0,1.0), halo=(1,1,1))
        φ = Field(CellCenter(), g); src = Field(CellCenter(), g); out = Field(CellCenter(), g)
        set!(φ, (x,y,z) -> sin(2π*x) * cos(2π*y) * sin(2π*z))
        fill_halos!(φ, FieldBoundaryConditions(:periodic))
        laplacian(src, φ)
        mg = MultigridPoissonSolver(g; bc_x=:periodic, bc_y=:periodic, bc_z=:periodic,
                                       tol=1e-12, maxcycles=60)
        solve!(out, src, mg)
        # φ has zero mean: the unique zero-mean solution of the same operator
        @test maximum(abs.(interior_view(out) .- interior_view(φ))) < 1e-9
        @test mg_level_count(mg) >= 3

        set!(φ, (x,y,z) -> cos(2π*x) * cos(2π*y) * cos(2π*z))
        fill_halos!(φ, FieldBoundaryConditions(:periodic, :periodic, :noflux))
        laplacian(src, φ)
        p_fft = Field(CellCenter(), g); p_mg = Field(CellCenter(), g)
        solve!(p_fft, src, CosineFFTPoissonSolver(g))
        solve!(p_mg, src, MultigridPoissonSolver(g; bc_x=:periodic, bc_y=:periodic,
                                                     bc_z=:neumann, tol=1e-12, maxcycles=80))
        dm(p) = interior_view(p) .- sum(interior_view(p))/length(interior_view(p))
        @test maximum(abs.(dm(p_mg) .- dm(p_fft))) < 1e-8
    end

    @testset "Cycle count independent of grid size, geometric convergence" begin
        cycles = map((16, 32)) do N
            g = RegularGrid(; size=(N,N,N), extent=(1.0,1.0,1.0), halo=(1,1,1))
            Random.seed!(1234)
            rhs = randn(N, N, N); rhs .-= sum(rhs) / length(rhs)
            mg = MultigridPoissonSolver(g; bc_x=:periodic, bc_y=:periodic, bc_z=:neumann,
                                           tol=1e-8, maxcycles=60)
            res = mg_solve!(mg, zeros(N, N, N), rhs)
            h = mg_iteration_history(mg)
            (n = res.converged ? res.cycles : typemax(Int),
             rate = maximum(h[2:min(6,end)] ./ h[1:min(5,end-1)]))
        end
        @test cycles[2].n <= cycles[1].n + 3          # O(1) cycles, not O(N)
        @test cycles[2].rate < 0.5                    # at least 2x per V-cycle
    end

    @testset "Variable-coefficient operator ∇·(c∇φ)" begin
        N = 8
        g = RegularGrid(; size=(N,N,N), extent=(1.0,1.0,1.0), halo=(1,1,1))
        cfun(z) = 1.0 + 2.0 * z                       # factor-3 contrast
        cx = [cfun((k-0.5)*g.dz) for i in 1:N+1, j in 1:N, k in 1:N]
        cy = [cfun((k-0.5)*g.dz) for i in 1:N, j in 1:N+1, k in 1:N]
        cz = [cfun((k-1)*g.dz) for i in 1:N, j in 1:N, k in 1:N+1]
        mg = MultigridPoissonSolver(g; bc_x=:periodic, bc_y=:periodic, bc_z=:neumann,
                                       tol=1e-10, maxcycles=80, coeff_x=cx, coeff_y=cy, coeff_z=cz)
        φ_ex = [sin(2π*(i-0.5)*g.dx) * cos(2π*(j-0.5)*g.dy) * cos(π*(k-0.5)*g.dz)
                for i in 1:N, j in 1:N, k in 1:N]
        φ_ex .-= sum(φ_ex)/length(φ_ex)
        rhs = similar(φ_ex)
        Aeolus.Multigrid._apply_fine!(rhs, φ_ex, mg)
        rhs .-= sum(rhs)/length(rhs)
        out = zeros(N,N,N)
        @test mg_solve!(mg, out, rhs).converged
        @test maximum(abs.(out .- sum(out)/length(out) .- φ_ex)) < 1e-7
    end

    # #132 (found by the Berlin case): an absolute breakdown test `|p·Ap| < eps`
    # stopped CG early for a small RHS; the iteration must scale with the RHS
    @testset "MG-preconditioned CG: convergence independent of the RHS scale (#132)" begin
        g = RegularGrid(; size=(8,8,8), extent=(250.0,250.0,100.0), halo=(1,1,1))
        b0 = [sin(0.7i) * cos(1.3j) + 0.1 * sin(0.9k) * i / 8 for i in 1:8, j in 1:8, k in 1:8]
        sols = map((1.0, 1e-6)) do scale
            s = MGPreconditionedCG(g; bc_x=:periodic, bc_y=:periodic, bc_z=:neumann, tol=1e-10)
            src = Field(CellCenter(), g); p = Field(CellCenter(), g)
            interior_view(src) .= scale .* b0
            solve!(p, src, s)
            interior_view(p) ./ scale
        end
        @test sols[2] ≈ sols[1] rtol = 1e-8
    end

    @testset "Drop-in pressure solver of BoussinesqSimulation; zero RHS" begin
        mesh = RegularGrid(; size=(16,16,16), extent=(2π,2π,2π), halo=(2,2,2))
        mg = MultigridPoissonSolver(mesh; bc_x=:periodic, bc_y=:periodic, bc_z=:neumann,
                                          tol=1e-11, maxcycles=60)
        sim = BoussinesqSimulation(mesh; ν=1e-3, solver=mg,
                                   bcs=FieldBoundaryConditions(:periodic, :periodic, :wall))
        Random.seed!(3)
        for f in (sim.u, sim.v, sim.w)
            interior_view(f) .= randn(16,16,16)
        end
        step!(sim, 1e-3)
        @test max_divergence(sim) < 1e-7
        out = ones(16,16,16)
        @test mg_solve!(mg, out, zeros(16,16,16)).converged && all(iszero, out)   # no 0/0
    end
end
