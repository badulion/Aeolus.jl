# Multigrid solver cases of KA-i (#303), shared by test/ka_multigrid.jl and
# the thread-invariance worker (section `multigrid`): uniform with every
# periodic / Neumann closure, W-cycle, FMG, stretched z, and cut-cell face
# coefficients around a solid block, for MultigridPoissonSolver and
# MGPreconditionedCG.
using Aeolus
using Aeolus.Fields: Field, interior_view
using Aeolus.Multigrid: MultigridPoissonSolver, MGPreconditionedCG
using Random: Xoshiro, randn

# Cut-cell face coefficients: a solid block (every face of its cells
# closed, so its cells are skipped by the smoother) and partly open faces.
function cutcell_coeffs(Nx, Ny, Nz; block = (5:10, 9:14, 1:6))
    cx = ones(Nx + 1, Ny, Nz); cy = ones(Nx, Ny + 1, Nz); cz = ones(Nx, Ny, Nz + 1)
    bi, bj, bk = block
    for k in bk, j in bj
        cx[first(bi):last(bi)+1, j, k] .= 0
    end
    for k in bk, i in bi
        cy[i, first(bj):last(bj)+1, k] .= 0
    end
    for j in bj, i in bi
        cz[i, j, first(bk):last(bk)+1] .= 0
    end
    cx[4, :, 1:7] .*= 0.3; cy[:, 8, 1:7] .*= 0.6; cz[:, :, 8] .*= 0.8
    return cx, cy, cz
end

# A compatible source for the cut-cell operator: zero in the solid cells,
# zero mean over the fluid.
function cutcell_source!(b; block = (5:10, 9:14, 1:6))
    solid = falses(size(b)); solid[block...] .= true
    b[solid] .= 0
    b[.!solid] .-= sum(b[.!solid]) / count(.!solid)
    return b
end

function mg_solver_cases(N)
    g = RegularGrid(; size = N, extent = (1000.0, 1000.0, 200.0), halo = (3, 3, 3))
    sm = StretchedGrid(; size = N, extent = (1000.0, 1000.0, 2000.0), halo = (3, 3, 3),
                       stretching = (:geometric, 1.04))
    cx, cy, cz = cutcell_coeffs(N...)
    C = Pair{String,Any}[]
    for (bx, bz) in ((:periodic, :neumann), (:neumann, :neumann), (:periodic, :periodic))
        push!(C, "mg($bx,$bz)" => (g, MultigridPoissonSolver(g; bc_x = bx, bc_y = bx, bc_z = bz, tol = 1e-10)))
    end
    push!(C, "mg(W)" => (g, MultigridPoissonSolver(g; cycle = :W, tol = 1e-10)))
    push!(C, "mg(fmg)" => (g, MultigridPoissonSolver(g; fmg = true, tol = 1e-10)))
    push!(C, "mg(stretched)" => (sm, MultigridPoissonSolver(sm; tol = 1e-10, maxcycles = 200)))
    push!(C, "mg(cutcell)" => (g, MultigridPoissonSolver(g; tol = 1e-10, maxcycles = 200,
                                                         coeff_x = cx, coeff_y = cy, coeff_z = cz)))
    push!(C, "mgcg(periodic,neumann)" => (g, MGPreconditionedCG(g; tol = 1e-10)))
    push!(C, "mgcg(neumann,neumann)" => (g, MGPreconditionedCG(g; bc_x = :neumann, bc_y = :neumann, tol = 1e-10)))
    push!(C, "mgcg(stretched)" => (sm, MGPreconditionedCG(sm; tol = 1e-10)))
    push!(C, "mgcg(cutcell)" => (g, MGPreconditionedCG(g; tol = 1e-10, coeff_x = cx, coeff_y = cy, coeff_z = cz)))
    return C
end
mg_cycles(s::MultigridPoissonSolver) = s.last_cycles[]
mg_cycles(s::MGPreconditionedCG) = s.last_iters[]

function mg_solve_case(name, m, s)
    src = Field(m); src.data .= randn(Xoshiro(303), size(src.data))
    occursin("cutcell", name) && cutcell_source!(interior_view(src))
    p = Field(m)
    solve!(p, src, s)
    return p, src
end

