# KA-g (#300): the one preconditioned-CG driver `Backends.pcg!` and the four
# solvers that run on it (CGPoissonSolver, MGPreconditionedCG,
# DensityWeightedPoissonSolver, solve_implicit_diffusion!).
using Test
using Random
using LinearAlgebra: norm
using Aeolus
using Aeolus.Grids: CellCenter, FaceX, FaceZ, interior
using Aeolus.Fields: interior_view
using Aeolus.BoundaryConditions: Periodic, NoSlipWall
const BK = Aeolus.Backends
const PSK = Aeolus.PressureSolvers
const MGK = Aeolus.Multigrid
const ANK = Aeolus.Anelastic
const TIK = Aeolus.TimeIntegration

# A small SPD system on a unit-halo array: (I·σ − ∇²) with a Neumann mirror,
# given point by point, and its dense matrix for a reference solve.
@inline _shifted_at(i, j, k, u, σ) =
    @inbounds σ * u[i, j, k] - (u[i+1, j, k] + u[i-1, j, k] + u[i, j+1, k] + u[i, j-1, k] +
                                u[i, j, k+1] + u[i, j, k-1] - 6u[i, j, k])

function dense_shifted(n, σ)
    N = n^3
    A = zeros(N, N)
    lin(i, j, k) = i + n * (j - 1) + n^2 * (k - 1)
    for k in 1:n, j in 1:n, i in 1:n
        r = lin(i, j, k)
        A[r, r] += σ
        for (di, dj, dk) in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
            ii, jj, kk = clamp(i + di, 1, n), clamp(j + dj, 1, n), clamp(k + dk, 1, n)
            A[r, r] += 1                      # −(u_nb − u): the mirror ghost
            A[r, lin(ii, jj, kk)] -= 1
        end
    end
    return A
end

# solve! cases: label => (solver, mesh, source)
function solver_cases(N)
    g = RegularGrid(; size = N, extent = (1000.0, 1000.0, 200.0), halo = (3, 3, 3))
    smesh = StretchedGrid(; size = N, extent = (1000.0, 1000.0, 2000.0), halo = (3, 3, 3),
                          stretching = (:geometric, 1.04))
    ref = ANK.isothermal_reference(g; T_surf = 300.0)
    gd = RegularGrid(; size = N, extent = (1000.0, 1000.0, 3000.0), halo = (3, 3, 3))
    refd = ANK.isothermal_reference(gd; T_surf = 300.0)
    C = Pair{String,Any}[]
    for (bz, bxy) in ((:neumann, :periodic), (:neumann, :neumann), (:periodic, :periodic))
        push!(C, "cg($bxy,$bz)" => (PSK.CGPoissonSolver(g; tol = 1e-8, maxit = 2000, bc_top = bz,
                                                         bc_bottom = bz, bc_xy = bxy), g))
    end
    for (bz, bxy) in ((:neumann, :periodic), (:periodic, :periodic))
        push!(C, "mgcg($bxy,$bz)" => (MGPreconditionedCG(g; bc_x = bxy, bc_y = bxy, bc_z = bz,
                                                         tol = 1e-10, maxit = 200), g))
    end
    push!(C, "mgcg(stretched)" => (MGPreconditionedCG(smesh; tol = 1e-10, maxit = 200), smesh))
    for (bz, bx) in ((:neumann, :periodic), (:periodic, :periodic), (:neumann, :neumann))
        push!(C, "dw($bx,$bz)" => (DensityWeightedPoissonSolver(g, ref; bc_z = bz, bc_x = bx, bc_y = bx,
                                                                tol = 1e-10, maxit = 2000), g))
    end
    push!(C, "dw(deep 3 km)" => (DensityWeightedPoissonSolver(gd, refd; tol = 1e-10, maxit = 2000), gd))
    return C
end
iterations(s::PSK.CGPoissonSolver) = s.last_iter[]
iterations(s::MGPreconditionedCG) = s.last_iters[]
iterations(s::DensityWeightedPoissonSolver) = solver_iterations(s)

function imex_cases(g)
    ν = 2.0
    νvar = zeros(g.Nx + 2g.Hx, g.Ny + 2g.Hy, g.Nz + 2g.Hz)
    for k in axes(νvar, 3), i in axes(νvar, 1)
        νvar[i, :, k] .= ν * (1 + 0.5sin(i / 7) + 0.3k / g.Nz)
    end
    return ["imex(θ, ν scalar, periodic)" => (CellCenter(), FieldBoundaryConditions(:periodic), ν),
            "imex(θ, ν(x,z), walls)" => (CellCenter(), FieldBoundaryConditions(:periodic, :periodic, :wall), νvar),
            "imex(w, walls, pinned)" => (FaceZ(), FieldBoundaryConditions(:periodic, :periodic, :wall), ν),
            "imex(u, moving no-slip)" => (FaceX(), FieldBoundaryConditions(Periodic(), Periodic(), Periodic(),
                                                                         Periodic(), NoSlipWall(),
                                                                         NoSlipWall(u_wall = 1.0)), ν)]
end

function imex_solve(g, (loc, bcs, ν); dt = 20.0)
    f = Field(loc, g)
    interior_view(f) .= randn(Xoshiro(302), size(interior_view(f)))
    st = solve_implicit_diffusion!(f, f, ν, dt, g, bcs; tol = 1e-10, maxit = 500)
    return f, st
end

# Iteration counts of the loops KA-g replaced (measured on the merge base,
# 32×32×16, same seeds): the expectation is ±1 (reduction order), measured
# identical.
const OLD_ITERS = Dict("cg(periodic,neumann)" => 147, "cg(neumann,neumann)" => 259,
                       "cg(periodic,periodic)" => 133, "mgcg(periodic,neumann)" => 12,
                       "mgcg(periodic,periodic)" => 12, "mgcg(stretched)" => 49,
                       "dw(periodic,neumann)" => 166, "dw(periodic,periodic)" => 150,
                       "dw(neumann,neumann)" => 286, "dw(deep 3 km)" => 268,
                       "imex(θ, ν scalar, periodic)" => 14, "imex(θ, ν(x,z), walls)" => 18,
                       "imex(w, walls, pinned)" => 14, "imex(u, moving no-slip)" => 15)

@testset "KA-g: one pcg! driver (#300)" begin
    @testset "pcg! solves an SPD system (dense reference)" begin
        n, σ = 6, 0.3
        A = dense_shifted(n, σ)
        rng = Xoshiro(1)
        bi = randn(rng, n, n, n)
        xref = reshape(A \ vec(bi), n, n, n)
        b = zeros(n + 2, n + 2, n + 2); b[2:n+1, 2:n+1, 2:n+1] .= bi
        x = zero(b)
        ws = PCGWorkspace(b; ranges = (2:n+1, 2:n+1, 2:n+1))
        op = PointOperator(_shifted_at, σ; prepare = a -> PSK.fill_unit_halo!(a, :neumann, :neumann, :neumann))
        st = pcg!(op, IdentityPreconditioner(), x, b, ws; ranges = (2:n+1, 2:n+1, 2:n+1),
                  tol = 1e-13, maxit = 500)
        @test st.status === :converged
        @test maximum(abs, x[2:n+1, 2:n+1, 2:n+1] .- xref) < 1e-11 * maximum(abs, xref)
        # Jacobi (diag = σ + 6 − #mirrored faces … any SPD diagonal works): fewer or equal iterations
        x2 = zero(b)
        d(i, j, k) = σ + count(((i == 2), (i == n + 1), (j == 2), (j == n + 1), (k == 2), (k == n + 1)) .== false)
        jac = PointPreconditioner((rv, i, j, k) -> rv / d(i, j, k))
        st2 = pcg!(op, jac, x2, b, ws; ranges = (2:n+1, 2:n+1, 2:n+1), tol = 1e-13, maxit = 500)
        @test st2.status === :converged
        @test maximum(abs, x2[2:n+1, 2:n+1, 2:n+1] .- xref) < 1e-11 * maximum(abs, xref)
        # a function preconditioner gives the same iterates as the point form
        x3 = zero(b)
        Mf!(z, r) = (for k in 2:n+1, j in 2:n+1, i in 2:n+1; z[i, j, k] = r[i, j, k] / d(i, j, k); end; z)
        st3 = pcg!(op, Mf!, x3, b, ws; ranges = (2:n+1, 2:n+1, 2:n+1), tol = 1e-13, maxit = 500)
        @test st3.iterations == st2.iterations
        @test x3 == x2
        # zero right-hand side: x = 0, no iteration
        fill!(x, 1.0)
        st0 = pcg!(op, IdentityPreconditioner(), x, zero(b), ws; ranges = (2:n+1, 2:n+1, 2:n+1),
                   tol = 1e-10, maxit = 10)
        @test st0.status === :zero_rhs && st0.iterations == 0
        @test all(==(0), x[2:n+1, 2:n+1, 2:n+1])
        # an indefinite operator is a breakdown for definite = :positive
        neg = PointOperator((i, j, k, u) -> @inbounds(-u[i, j, k]))
        stb = pcg!(neg, IdentityPreconditioner(), x, b, ws; ranges = (2:n+1, 2:n+1, 2:n+1),
                   tol = 1e-10, maxit = 10)
        @test stb.status === :breakdown && stb.iterations == 0
        stn = pcg!(neg, IdentityPreconditioner(), x, b, ws; ranges = (2:n+1, 2:n+1, 2:n+1),
                   tol = 1e-10, maxit = 10, definite = :negative)
        @test stn.status === :converged && stn.iterations == 1
        @test x[2:n+1, 2:n+1, 2:n+1] ≈ -bi
        @test_throws ArgumentError pcg!(op, IdentityPreconditioner(), x, b, ws; tol = 1e-8, maxit = 1,
                                        project_mean = true, constraint = (0, 0, 2))
    end

    @testset "pinned rows stay out of the system" begin
        n, σ = 6, 0.3
        b = zeros(n + 2, n + 2, n + 2); b[2:n+1, 2:n+1, 2:n+1] .= randn(Xoshiro(2), n, n, n)
        x = zero(b); x[:, :, 3] .= 7.0           # the pinned plane keeps its value
        ws = PCGWorkspace(b; ranges = (2:n+1, 2:n+1, 2:n+1))
        op = PointOperator(_shifted_at, σ; prepare = a -> PSK.fill_unit_halo!(a, :neumann, :neumann, :neumann))
        st = pcg!(op, IdentityPreconditioner(), x, b, ws; ranges = (2:n+1, 2:n+1, 2:n+1),
                  tol = 1e-12, maxit = 500, constraint = (0, 0, 3), zero_guess = false)
        @test st.status === :converged
        @test all(==(7.0), x[2:n+1, 2:n+1, 3])
        @test all(==(0), ws.r[2:n+1, 2:n+1, 3]) && all(==(0), ws.p[2:n+1, 2:n+1, 3])
        @test st.bnorm ≈ norm(b[2:n+1, 2:n+1, [2; 4:n+1]])
    end

    @testset "iteration counts, solutions and determinism of the four solvers" begin
        N = (32, 32, 16)
        for (lab, (s, m)) in solver_cases(N)
            src = Field(CellCenter(), m)
            interior_view(src) .= randn(Xoshiro(m isa StretchedGrid ? 301 : 300), N...)
            p = Field(CellCenter(), m)
            solve!(p, src, s)
            it = iterations(s)
            @info "KA-g $lab: $it iterations (before KA-g: $(OLD_ITERS[lab]))"
            @test abs(it - OLD_ITERS[lab]) <= 1
            x = copy(interior_view(p))
            @test abs(sum(x)) < 1e-10 * sum(abs, x)          # the zero-mean solution
            # the serial backend: the same bits and iteration count
            q = Field(CellCenter(), m)
            with_backend(CPUBackend(serial = true)) do
                solve!(q, src, s)
            end
            @test iterations(s) == it
            @test interior_view(q) == x
            # no allocation per solve (one thread; threaded launches allocate KA's tasks)
            if Threads.nthreads() == 1
                solve!(p, src, s)
                @test (@allocated solve!(p, src, s)) == 0
            end
        end
        # the residual of the solve, against the operator itself
        let g = RegularGrid(; size = N, extent = (1000.0, 1000.0, 200.0), halo = (3, 3, 3)),
            s = DensityWeightedPoissonSolver(g, ANK.isothermal_reference(g; T_surf = 300.0); tol = 1e-12, maxit = 2000)
            src = Field(CellCenter(), g); interior_view(src) .= randn(Xoshiro(300), N...)
            interior_view(src) .-= sum(interior_view(src)) / prod(N)
            p = Field(CellCenter(), g); solve!(p, src, s)
            Lp = Field(CellCenter(), g); apply_weighted_laplacian!(Lp, p, s)
            @test norm(interior_view(Lp) .- interior_view(src)) <= 2e-12 * norm(interior_view(src))
        end
        g = RegularGrid(; size = N, extent = (1000.0, 1000.0, 200.0), halo = (3, 3, 3))
        for (lab, c) in imex_cases(g)
            f, st = imex_solve(g, c)
            @info "KA-g $lab: $(st.iterations) iterations (before KA-g: $(OLD_ITERS[lab]))"
            @test st.converged
            @test abs(st.iterations - OLD_ITERS[lab]) <= 1
            f2, st2 = with_backend(() -> imex_solve(g, c), CPUBackend(serial = true))
            @test st2.iterations == st.iterations && f2.data == f.data
        end
    end

    @testset "_diffusion! is the operator the implicit solve inverts" begin
        g = RegularGrid(; size = (16, 16, 8), extent = (1.0, 1.0, 0.5), halo = (2, 2, 2))
        bcs = FieldBoundaryConditions(:periodic)
        φ = Field(CellCenter(), g); interior_view(φ) .= randn(Xoshiro(5), 16, 16, 8)
        dt, ν = 0.01, 0.3
        rhs = Field(CellCenter(), g); interior_view(rhs) .= interior_view(φ)
        sol = Field(CellCenter(), g); interior_view(sol) .= 0
        st = solve_implicit_diffusion!(sol, rhs, ν, dt, g, bcs; tol = 1e-13, maxit = 500)
        @test st.converged
        fill_halos!(sol, bcs)
        L = Field(CellCenter(), g)
        TIK._diffusion!(L, sol, ν, g)
        @test maximum(abs, interior_view(sol) .- dt .* interior_view(L) .- interior_view(rhs)) < 1e-11
    end
end
