# N-open-a (#251, epic #246): pressure solves for non-periodic lateral
# boundaries.
#
# A nested (or open-boundary) child has a prescribed normal velocity on
# every face, so the projection's φ obeys ∂φ/∂n = 0 everywhere. This file
# checks the per-axis Neumann option of CosineFFTPoissonSolver (DCT-II on
# a Neumann axis, Schumann & Sweet 1988, JCP 75, 123), the lateral Neumann
# closure of the two CG solvers — CGPoissonSolver used to store `bc_xy`
# and wrap x, y periodically anyway (a code bug: the "CG bug regression"
# set below fails on 3a23110) — and the BC-aware default solver.
#
#   1. exact discrete inverse (manufactured DCT eigenvectors, random data)
#   2. continuum convergence, order 2
#   3. cross-solver agreement: DCT, multigrid, CG, density-weighted CG
#   4. CG bug regression
#   5. variable coefficient (isothermal ρ̄) with Neumann sides
#   6. compatibility: an incompatible source leaves the uniform residual −s̄
#   7. default selection from the six faces; channel default bit for bit
#   8. cost per solve (periodic cosine, all-Neumann DCT, multigrid)

using Test
using Aeolus
using Aeolus.Grids: interior, xnodes, ynodes, znodes
using Aeolus.Fields: interior_view, set!
using Aeolus.Multigrid: MultigridPoissonSolver, MGPreconditionedCG
using Aeolus.Anelastic: DensityWeightedPoissonSolver, apply_weighted_laplacian!,
                        isothermal_reference, AnelasticSimulation, solver_iterations
using Aeolus.BoundaryConditions: ImpermeableWall, NoSlipWall, axis_poisson_bc
using Aeolus.Simulations: _default_solver
using FFTW
using LinearAlgebra: mul!
using Printf, Random
using Statistics: mean

# Independent reference 7-point Laplacian: pad with explicit ghosts (wrap
# for a periodic axis, mirror φ₀ = φ₁ for a Neumann axis), then difference.
function _op_ref_lap(φ::Array{Float64,3}, Δ, bcs)
    N = size(φ)
    P = zeros(N .+ 2)
    P[2:end-1, 2:end-1, 2:end-1] .= φ
    for (d, bc) in enumerate(bcs)
        lo = ntuple(a -> a == d ? (1:1)            : (2:N[a]+1), 3)
        hi = ntuple(a -> a == d ? (N[d]+2:N[d]+2)  : (2:N[a]+1), 3)
        slo = ntuple(a -> a == d ? (bc === :periodic ? (N[d]+1:N[d]+1) : (2:2)) : (2:N[a]+1), 3)
        shi = ntuple(a -> a == d ? (bc === :periodic ? (2:2) : (N[d]+1:N[d]+1)) : (2:N[a]+1), 3)
        P[lo...] .= P[slo...]
        P[hi...] .= P[shi...]
    end
    L = zeros(N)
    c = P[2:end-1, 2:end-1, 2:end-1]
    L .+= (P[3:end, 2:end-1, 2:end-1] .- 2c .+ P[1:end-2, 2:end-1, 2:end-1]) ./ Δ[1]^2
    L .+= (P[2:end-1, 3:end, 2:end-1] .- 2c .+ P[2:end-1, 1:end-2, 2:end-1]) ./ Δ[2]^2
    L .+= (P[2:end-1, 2:end-1, 3:end] .- 2c .+ P[2:end-1, 2:end-1, 1:end-2]) ./ Δ[3]^2
    return L
end

_op_demean(a) = a .- mean(a)
_op_rel(a, b) = maximum(abs, a .- b) / maximum(abs, b)

function _op_solve(solver, g, src::Array{Float64,3})
    s = Field(CellCenter(), g); p = Field(CellCenter(), g)
    interior_view(s) .= src
    solve!(p, s, solver)
    return collect(interior_view(p))
end

_op_cos(bc, L) = bc === :periodic ? (m, x) -> cos(2π * m * x / L) : (m, x) -> cos(π * m * x / L)

const _OP_MIXES = [(:neumann, :neumann, :neumann), (:neumann, :periodic, :neumann),
                   (:periodic, :neumann, :neumann), (:neumann, :neumann, :periodic),
                   (:periodic, :periodic, :neumann)]

@testset "N-open-a: pressure solves for non-periodic lateral BCs (#251)" begin

g  = RegularGrid(; size = (32, 24, 16), extent = (1.0, 1.5, 0.7), halo = (1, 1, 1))
Δ  = (g.dx, g.dy, g.dz)
xc, yc, zc = xnodes(g, CellCenter()), ynodes(g, CellCenter()), znodes(g, CellCenter())

@testset "1. exact discrete inverse, 32 × 24 × 16" begin
    for bcs in _OP_MIXES
        s  = CosineFFTPoissonSolver(g; bc_x = bcs[1], bc_y = bcs[2], bc_z = bcs[3])
        @test poisson_axis_bcs(s) == (bc_x = bcs[1], bc_y = bcs[2], bc_z = bcs[3])
        cx, cy, cz = _op_cos(bcs[1], 1.0), _op_cos(bcs[2], 1.5), _op_cos(bcs[3], 0.7)
        # On the cell centres cos(π m x/L) is an exact eigenvector of the
        # Neumann-closed second difference (a DCT-II basis vector), and
        # cos(2π m x/L) of the periodic one: L_h φ = λ_h φ analytically.
        φ = [cx(1, x) * cy(2, y) * cz(1, z) + 0.4 * cx(3, x) * cy(1, y) * cz(0, z) +
             0.2 * cx(0, x) * cy(0, y) * cz(3, z) for x in xc, y in yc, z in zc]
        f = _op_ref_lap(φ, Δ, bcs)
        e_mms = _op_rel(_op_demean(_op_solve(s, g, f)), _op_demean(φ))
        # Random source: L_h(solve(s)) = s − s̄ to round-off.
        rng = MersenneTwister(251)
        r = randn(rng, size(φ))
        e_rnd = _op_rel(_op_ref_lap(_op_solve(s, g, r), Δ, bcs), r .- mean(r))
        @printf("  [N-open-a] %-28s manufactured rel. err %.2e, random-source residual %.2e\n",
                string(bcs), e_mms, e_rnd)
        @test e_mms < 1e-12
        @test e_rnd < 1e-12
    end
    @test_throws ArgumentError CosineFFTPoissonSolver(g; bc_x = :dirichlet)
end

@testset "2. continuum convergence, order 2" begin
    # g(ξ) = 2ξ³ − 3ξ² has g'(0) = g'(1) = 0: a Neumann-compatible
    # non-eigenfunction, so the error is not a single-mode artefact.
    gξ(ξ) = 2ξ^3 - 3ξ^2
    g2(ξ) = 12ξ - 6
    for (label, bcs, φf, ff) in (
        ("all Neumann", (:neumann, :neumann, :neumann),
         (x, y, z) -> gξ(x) * cos(π * y) * cos(π * z),
         (x, y, z) -> (g2(x) - 2π^2 * gξ(x)) * cos(π * y) * cos(π * z)),
        ("Neumann x, periodic y", (:neumann, :periodic, :neumann),
         (x, y, z) -> gξ(x) * cos(2π * y) * cos(π * z),
         (x, y, z) -> (g2(x) - 5π^2 * gξ(x)) * cos(2π * y) * cos(π * z)))
        errs = Float64[]
        for N in (16, 32, 64)
            gN = RegularGrid(; size = (N, N, N), extent = (1.0, 1.0, 1.0), halo = (1, 1, 1))
            X, Y, Z = xnodes(gN, CellCenter()), ynodes(gN, CellCenter()), znodes(gN, CellCenter())
            s = CosineFFTPoissonSolver(gN; bc_x = bcs[1], bc_y = bcs[2], bc_z = bcs[3])
            f = [ff(x, y, z) for x in X, y in Y, z in Z]
            φ = [φf(x, y, z) for x in X, y in Y, z in Z]
            push!(errs, maximum(abs, _op_demean(_op_solve(s, gN, f)) .- _op_demean(φ)))
        end
        p1, p2 = log2(errs[1] / errs[2]), log2(errs[2] / errs[3])
        @printf("  [N-open-a] convergence %-22s max err %.3e %.3e %.3e   order %.3f %.3f\n",
                label, errs..., p1, p2)
        @test abs(p1 - 2) < 0.1
        @test abs(p2 - 2) < 0.1
    end
end

@testset "3. cross-solver agreement, all Neumann" begin
    rng = MersenneTwister(2510)
    r = randn(rng, g.Nx, g.Ny, g.Nz)
    nb = (bc_x = :neumann, bc_y = :neumann, bc_z = :neumann)
    p_dct = _op_solve(CosineFFTPoissonSolver(g; nb...), g, r)
    mg    = MultigridPoissonSolver(g; nb..., tol = 1e-12, maxcycles = 200)
    p_mg  = _op_solve(mg, g, r)
    mgcg  = MGPreconditionedCG(g; nb..., tol = 1e-12, maxit = 200)
    p_mgcg = _op_solve(mgcg, g, r)
    cg    = CGPoissonSolver(g; tol = 1e-13, maxit = 5000, bc_x = :neumann, bc_y = :neumann)
    p_cg  = _op_solve(cg, g, r)
    dw    = DensityWeightedPoissonSolver(g, 1.0; tol = 1e-13, maxit = 5000,
                                         bc_x = :neumann, bc_y = :neumann)
    p_dw  = _op_solve(dw, g, r)
    d = Dict("multigrid" => _op_rel(_op_demean(p_mg), p_dct),
             "MG-CG"     => _op_rel(_op_demean(p_mgcg), p_dct),
             "CG"        => _op_rel(p_cg, p_dct),
             "DW(ρ̄=1)"  => _op_rel(p_dw, p_dct))
    for (k, v) in sort(collect(d))
        @printf("  [N-open-a] all-Neumann: %-10s vs DCT, max rel. diff %.2e\n", k, v)
        @test v < 1e-9
    end
    @test abs(mean(p_dct)) < 1e-14 * maximum(abs, p_dct)    # the zero-mean solution
    @test poisson_axis_bcs(mg) == nb && poisson_axis_bcs(mgcg) == nb
    @test poisson_axis_bcs(cg) == nb && poisson_axis_bcs(dw) == nb
end

@testset "4. CG bug regression: bc_x/bc_y = :neumann is honoured" begin
    # On 3a23110 CGPoissonSolver stored `bc_xy` and wrapped x and y
    # periodically regardless, so bc_xy = :neumann returned the periodic
    # solution. A code bug, not a test expectation.
    rng = MersenneTwister(77)
    r = randn(rng, g.Nx, g.Ny, g.Nz)
    p_per = _op_solve(CGPoissonSolver(g; tol = 1e-13, maxit = 5000, bc_xy = :periodic), g, r)
    p_neu = _op_solve(CGPoissonSolver(g; tol = 1e-13, maxit = 5000, bc_xy = :neumann), g, r)
    p_dct = _op_solve(CosineFFTPoissonSolver(g; bc_x = :neumann, bc_y = :neumann), g, r)
    p_cos = _op_solve(CosineFFTPoissonSolver(g), g, r)
    @printf("  [N-open-a] CG bc_xy=:neumann vs periodic rel. diff %.3f; vs Neumann DCT %.2e; periodic CG vs channel DCT %.2e\n",
            _op_rel(p_neu, p_per), _op_rel(p_neu, p_dct), _op_rel(p_per, p_cos))
    @test _op_rel(p_neu, p_per) > 0.1
    @test _op_rel(p_neu, p_dct) < 1e-9
    @test _op_rel(p_per, p_cos) < 1e-9          # the periodic path is unchanged
    # Per-axis: Neumann x only.
    p_x  = _op_solve(CGPoissonSolver(g; tol = 1e-13, maxit = 5000, bc_x = :neumann), g, r)
    p_xd = _op_solve(CosineFFTPoissonSolver(g; bc_x = :neumann), g, r)
    @test _op_rel(p_x, p_xd) < 1e-9
    @test_throws ArgumentError CGPoissonSolver(g; bc_x = :open)
end

@testset "5. variable coefficient: isothermal ρ̄, Neumann sides" begin
    gv  = RegularGrid(; size = (16, 12, 24), extent = (800.0, 600.0, 6000.0), halo = (2, 2, 2))
    ref = isothermal_reference(gv)
    tol = 1e-12
    s   = DensityWeightedPoissonSolver(gv, ref; tol = tol, maxit = 5000,
                                       bc_x = :neumann, bc_y = :neumann)
    sp  = DensityWeightedPoissonSolver(gv, ref; tol = tol, maxit = 5000)
    ratio = maximum(ref.ρ̄) / minimum(ref.ρ̄)
    φ = Field(CellCenter(), gv); f = Field(CellCenter(), gv); out = Field(CellCenter(), gv)
    set!(φ, (x, y, z) -> cos(π * x / 800) * cos(2π * y / 600) * cos(π * z / 6000) +
                         0.5 * (x / 800)^2 * (1 - 2x / 2400) * cos(2π * z / 6000))
    apply_weighted_laplacian!(f, φ, s)
    # Discrete Gauss: with zero flux through every face the weighted
    # divergence sums to zero over the domain (to round-off).
    fsum = sum(interior_view(f)) / (length(interior_view(f)) * maximum(abs, interior_view(f)))
    solve!(out, f, s)
    Lout = Field(CellCenter(), gv); apply_weighted_laplacian!(Lout, out, s)
    fi = collect(interior_view(f)); fi .-= mean(fi)
    res = sqrt(sum(abs2, collect(interior_view(Lout)) .- fi)) / sqrt(sum(abs2, fi))
    e   = _op_rel(_op_demean(collect(interior_view(out))), _op_demean(collect(interior_view(φ))))
    # The same data through the periodic-side operator is a different problem.
    Lp = Field(CellCenter(), gv); apply_weighted_laplacian!(Lp, φ, sp)
    dop = _op_rel(collect(interior_view(Lp)), collect(interior_view(f)))
    @printf("  [N-open-a] DW isothermal, Neumann sides: ρ̄ ratio %.2f, ‖Lφ−f‖/‖f‖ %.2e (tol %.0e), φ rel. err %.2e, %d iterations, Σf/(N max|f|) %.1e, periodic-vs-Neumann operator %.2f\n",
            ratio, res, tol, e, solver_iterations(s), fsum, dop)
    @test ratio > 1.5
    @test res < 10tol
    @test e < 1e-8
    @test abs(fsum) < 1e-14
    @test dop > 0.1

    # The sound-proof core builds its solver from its velocity faces.
    a_def = AnelasticSimulation(gv)
    @test poisson_axis_bcs(a_def.solver) == (bc_x = :periodic, bc_y = :periodic, bc_z = :neumann)
    a_box = AnelasticSimulation(gv; bcs = FieldBoundaryConditions(:noflux, :noflux, :wall))
    @test poisson_axis_bcs(a_box.solver) == (bc_x = :neumann, bc_y = :neumann, bc_z = :neumann)
end

@testset "6. compatibility: residual −s̄" begin
    rng = MersenneTwister(5)
    r0 = randn(rng, g.Nx, g.Ny, g.Nz); r0 .-= mean(r0)
    s̄ = 0.37
    src = r0 .+ s̄                        # a net source Q = s̄ V
    for bcs in ((:neumann, :neumann, :neumann), (:periodic, :periodic, :neumann))
        s = CosineFFTPoissonSolver(g; bc_x = bcs[1], bc_y = bcs[2], bc_z = bcs[3])
        res = _op_ref_lap(_op_solve(s, g, src), Δ, bcs) .- src
        dev = maximum(abs, res .+ s̄)
        @printf("  [N-open-a] compatibility %-30s mean residual %.15f (−s̄ = %.2f), max|res + s̄| %.1e\n",
                string(bcs), mean(res), -s̄, dev)
        @test dev < 1e-12
    end
    cg  = CGPoissonSolver(g; tol = 1e-14, maxit = 5000, bc_xy = :neumann)
    res = _op_ref_lap(_op_solve(cg, g, src), Δ, (:neumann, :neumann, :neumann)) .- src
    @test maximum(abs, res .+ s̄) < 1e-10
end

@testset "7. default solver from the six faces" begin
    # Current defaults: unchanged solver types.
    @test _default_solver(g, FieldBoundaryConditions(:periodic)) isa FFTPoissonSolver
    ch = _default_solver(g, FieldBoundaryConditions(:periodic, :periodic, :wall))
    @test ch isa CosineFFTPoissonSolver
    @test poisson_axis_bcs(ch) == (bc_x = :periodic, bc_y = :periodic, bc_z = :neumann)
    @test poisson_axis_bcs(BoussinesqSimulation(g).solver) ==
          (bc_x = :periodic, bc_y = :periodic, bc_z = :periodic)
    # Every z wall pair (free slip, no slip, mixed) is Neumann.
    for (b, t) in ((:noslip, :wall), (:noslip, :noslip), (:wall, :wall))
        @test poisson_axis_bcs(FieldBoundaryConditions(:periodic, :periodic; bottom = b, top = t)) ==
              (bc_x = :periodic, bc_y = :periodic, bc_z = :neumann)
    end
    # The closed box of a child (prescribed normal velocity on every face;
    # NoFlux sides stand in for the OpenBoundary of N-open-b, which extends
    # axis_poisson_bc with :neumann) → all-Neumann DCT.
    box = _default_solver(g, FieldBoundaryConditions(:noflux, :noflux, :wall))
    @test box isa CosineFFTPoissonSolver
    @test poisson_axis_bcs(box) == (bc_x = :neumann, bc_y = :neumann, bc_z = :neumann)
    @test poisson_axis_bcs(BoussinesqSimulation(g; bcs = FieldBoundaryConditions(:noflux, :noflux, :wall)).solver) ==
          (bc_x = :neumann, bc_y = :neumann, bc_z = :neumann)
    # Unsupported: a periodic face opposite a Neumann one; a face with no
    # Poisson kind.
    @test_throws ArgumentError _default_solver(g,
        FieldBoundaryConditions(Periodic(), NoFlux(), Periodic(), Periodic(), ImpermeableWall(), ImpermeableWall()))
    @test_throws ArgumentError _default_solver(g,
        FieldBoundaryConditions(ValueBC(1.0), ValueBC(1.0), Periodic(), Periodic(), ImpermeableWall(), ImpermeableWall()))
    @test axis_poisson_bc(NoSlipWall(1.0, 0.0)) === :neumann

    # Bit for bit: the channel default is the construction of 3a23110 —
    # DCT-II along z, rfft along (x, y), the same eigenvalues, 1/(2Nz).
    T = Float64; Nx, Ny, Nz = g.Nx, g.Ny, g.Nz
    rb = zeros(T, Nx, Ny, Nz); rb2 = similar(rb); hb = zeros(Complex{T}, Nx ÷ 2 + 1, Ny, Nz)
    pdf = FFTW.plan_r2r(rb, FFTW.REDFT10, 3; flags = FFTW.MEASURE)
    pdi = FFTW.plan_r2r(rb, FFTW.REDFT01, 3; flags = FFTW.MEASURE)
    pff = FFTW.plan_rfft(rb, (1, 2); flags = FFTW.MEASURE)
    pfi = FFTW.plan_irfft(hb, Nx, (1, 2); flags = FFTW.MEASURE)
    kx² = [4 / g.dx^2 * sin(π * (m - 1) / Nx)^2 for m in 1:Nx ÷ 2 + 1]
    ky² = [4 / g.dy^2 * sin(π * (n - 1) / Ny)^2 for n in 1:Ny]
    kz² = [4 / g.dz^2 * sin(π * (l - 1) / (2Nz))^2 for l in 1:Nz]
    invλ = [(λ = -(kx²[m] + ky²[n] + kz²[l]); λ == 0 ? 0.0 : 1 / λ)
            for m in 1:Nx ÷ 2 + 1, n in 1:Ny, l in 1:Nz]
    rb .= randn(MersenneTwister(9), Nx, Ny, Nz)
    src = copy(rb)
    mul!(rb2, pdf, rb); mul!(hb, pff, rb2); hb .*= invλ; mul!(rb2, pfi, hb); mul!(rb, pdi, rb2)
    rb .*= 1 / T(2Nz)
    @test ch.inv_eigenvalues == invλ
    @test _op_solve(ch, g, src) == rb
end

@testset "8. cost per solve" begin
    function tsolve(s, gg; reps = 5)
        f = Field(CellCenter(), gg); p = Field(CellCenter(), gg)
        interior_view(f) .= randn(MersenneTwister(1), gg.Nx, gg.Ny, gg.Nz)
        interior_view(f) .-= mean(interior_view(f))
        solve!(p, f, s)                                   # compile / warm
        return minimum(@elapsed(solve!(p, f, s)) for _ in 1:reps)
    end
    for (label, sz, ext) in (("Berlin 64×64×32", (64, 64, 32), (1000.0, 1000.0, 200.0)),
                             ("Berlin deep 64×64×64", (64, 64, 64), (1000.0, 1000.0, 400.0)),
                             ("160×160×64", (160, 160, 64), (6000.0, 6000.0, 3000.0)))
        gg = RegularGrid(; size = sz, extent = ext, halo = (1, 1, 1))
        t_c = tsolve(CosineFFTPoissonSolver(gg), gg)
        t_n = tsolve(CosineFFTPoissonSolver(gg; bc_x = :neumann, bc_y = :neumann), gg)
        mg  = MultigridPoissonSolver(gg; bc_x = :neumann, bc_y = :neumann, tol = 1e-10)
        t_m = tsolve(mg, gg; reps = 2)
        @printf("  [N-open-a] cost %-22s periodic cosine %.2f ms, all-Neumann DCT %.2f ms (×%.2f), multigrid (1e-10, %d cycles) %.1f ms (×%.1f)\n",
                label, 1e3t_c, 1e3t_n, t_n / t_c, mg.last_cycles[], 1e3t_m, t_m / t_c)
        # Loose guard on a shared machine (issue expects ≤ 1.3×; quoted in
        # the PR): a DCT must not cost a different order than an FFT.
        @test t_n / t_c < 2.5
        @test t_n < t_m
    end
end

end
