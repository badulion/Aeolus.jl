module PressureSolvers

using FFTW
using LinearAlgebra: mul!
using ..Grids
using ..Grids: RegularGrid, interior, CellCenter
using ..Fields
using ..Fields: Field
using ..Operators: fftw_planner_flags   # the one FFTW planning policy (#245)

export AbstractPressureSolver, FFTPoissonSolver, CosineFFTPoissonSolver,
       CGPoissonSolver, solve!

"""
    AbstractPressureSolver

Pluggable Poisson-solver interface. Every concrete solver implements a
single method

    solve!(p, source, solver)

where `p` and `source` are cell-centered `Field`s living on the mesh
the solver was constructed with. The solver populates `p`'s interior
from `source`'s interior; halos of `p` are left untouched.

Phase 1a ships [`FFTPoissonSolver`](@ref) for triply-periodic
`RegularGrid`. Future planned backends:

  * `CosineFFTPoissonSolver` — FFT in horizontal, DCT-II in vertical
    for channel-like domains (Phase 1b).
  * `CGPoissonSolver` — matrix-free preconditioned conjugate gradient
    for arbitrary BCs and immersed boundaries (Phase 3).
  * `SEMPoissonSolver` — element-local mass + matrix-free CG for the
    spectral-element backend (Phase 5).
"""
abstract type AbstractPressureSolver end

# -------------------------------------------------------------------------
# FFTPoissonSolver — triply-periodic direct solve
# -------------------------------------------------------------------------

"""
    FFTPoissonSolver(mesh::RegularGrid)

Direct Poisson solver for triply-periodic boundary conditions on a
`RegularGrid`, with source and solution both at cell centers.

The FFT is diagonalized using the *exact* eigenvalues of the discrete
2nd-order centered 7-point Laplacian:

    λ_h(m_x, m_y, m_z) = - 4/dx² sin²(π m_x/Nx)
                         - 4/dy² sin²(π m_y/Ny)
                         - 4/dz² sin²(π m_z/Nz)

so the returned `p` satisfies the *discrete* equation `L_h p = source` to
machine precision (up to projection of `source` onto its zero-mean part,
which is the periodic-Poisson compatibility condition).
"""
struct FFTPoissonSolver{T,
                        FP,
                        IP,
                        RA<:Array{T,3},
                        CA<:Array{Complex{T},3}} <: AbstractPressureSolver
    forward_plan::FP
    inverse_plan::IP
    src_buffer::RA
    hat_buffer::CA
    inv_eigenvalues::Array{T,3}   # 1 / λ_h,  zero at the k=0 mode
end

function FFTPoissonSolver(mesh::RegularGrid{T}) where {T}
    Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
    dx, dy, dz = mesh.dx, mesh.dy, mesh.dz

    src = zeros(T, Nx, Ny, Nz)
    hat = zeros(Complex{T}, div(Nx, 2) + 1, Ny, Nz)
    forward_plan = plan_rfft(src; flags = fftw_planner_flags())
    inverse_plan = plan_irfft(hat, Nx; flags = fftw_planner_flags())

    kx² = [ 4/dx^2 * sin(π * (m - 1) / Nx)^2 for m in 1:div(Nx, 2) + 1 ]
    ky² = [ 4/dy^2 * sin(π * (m - 1) / Ny)^2 for m in 1:Ny ]
    kz² = [ 4/dz^2 * sin(π * (m - 1) / Nz)^2 for m in 1:Nz ]

    invλ = zeros(T, div(Nx, 2) + 1, Ny, Nz)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:div(Nx, 2) + 1
        λ = -(kx²[i] + ky²[j] + kz²[k])
        invλ[i, j, k] = λ == 0 ? zero(T) : 1 / λ
    end

    return FFTPoissonSolver{T, typeof(forward_plan), typeof(inverse_plan),
                            typeof(src), typeof(hat)}(
        forward_plan, inverse_plan, src, hat, invλ
    )
end

"""
    solve!(p::Field{CellCenter}, source::Field{CellCenter}, solver::FFTPoissonSolver)

Solve `L_h p = source` in-place. Both fields must live on the mesh the
solver was constructed with. The mean of `source` is projected out; if
your source has a large mean you have a compatibility issue, not a solver
bug.
"""
function solve!(p::Field{CellCenter,T}, source::Field{CellCenter,T},
                solver::FFTPoissonSolver{T}) where {T}
    m  = source.mesh
    ii, jj, kk = interior(m)

    # copy interior of source into contiguous buffer
    @views solver.src_buffer .= source.data[ii, jj, kk]

    # forward transform
    mul!(solver.hat_buffer, solver.forward_plan, solver.src_buffer)

    # divide by eigenvalues (zero mode annihilated by inv_eigenvalues[1,1,1] = 0)
    @inbounds solver.hat_buffer .*= solver.inv_eigenvalues

    # inverse transform → real buffer
    mul!(solver.src_buffer, solver.inverse_plan, solver.hat_buffer)

    # write back
    @views p.data[ii, jj, kk] .= solver.src_buffer
    return p
end

# -------------------------------------------------------------------------
# CosineFFTPoissonSolver — periodic in x, y; Neumann in z (channel).
# -------------------------------------------------------------------------

"""
    CosineFFTPoissonSolver(mesh::RegularGrid)

Direct Poisson solver for a channel-like domain: periodic in `x, y` and
homogeneous-Neumann in `z` (both walls). Cell-centered source and
solution.

The transform in `z` is a DCT-II (type-II discrete cosine transform),
which is the correct spectral basis for cell-centered Neumann BCs:
values are even-extended about half-integer positions, so
`p_{i,j,0} = p_{i,j,1}` and `p_{i,j,Nz+1} = p_{i,j,Nz}` are enforced
implicitly. In `x` and `y` we use the same real FFT as
[`FFTPoissonSolver`](@ref).

Discrete Laplacian eigenvalues used here:

```math
\\lambda_{m,n,l} = -\\frac{4}{\\Delta x^2}\\sin^2\\!\\left(\\frac{\\pi m}{N_x}\\right)
                 -\\frac{4}{\\Delta y^2}\\sin^2\\!\\left(\\frac{\\pi n}{N_y}\\right)
                 -\\frac{4}{\\Delta z^2}\\sin^2\\!\\left(\\frac{\\pi l}{2 N_z}\\right)
```

Note the `2 N_z` in the `z` eigenvalue — this is the DCT-II analogue of
the periodic FFT formula and reflects the extended period of the even
reflection at the walls.

The `(0, 0, 0)` mode has zero eigenvalue and is projected out (pressure
is defined up to an additive constant with Neumann BCs). The zero-mean
compatibility condition on the source is enforced by that projection.
"""
struct CosineFFTPoissonSolver{T,
                              PDF, PDI, PFF, PFI,
                              RA<:Array{T,3},
                              CA<:Array{Complex{T},3}} <: AbstractPressureSolver
    plan_dct_fwd::PDF        # DCT-II  along z (REDFT10)
    plan_dct_inv::PDI        # DCT-III along z (REDFT01)
    plan_fft_fwd::PFF        # rfft    along (x, y)
    plan_fft_inv::PFI        # irfft   along (x, y)
    real_buf::RA
    real_buf2::RA            # scratch — plan_r2r is out-of-place
    hat_buf::CA
    inv_eigenvalues::Array{T,3}
    dct_scale::T             # 1 / (2 Nz), the DCT-II × DCT-III normalizer
end

function CosineFFTPoissonSolver(mesh::RegularGrid{T}) where {T}
    Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
    dx, dy, dz = mesh.dx, mesh.dy, mesh.dz

    real_buf  = zeros(T, Nx, Ny, Nz)
    real_buf2 = zeros(T, Nx, Ny, Nz)
    hat_buf   = zeros(Complex{T}, div(Nx, 2) + 1, Ny, Nz)

    plan_dct_fwd = FFTW.plan_r2r(real_buf, FFTW.REDFT10, 3; flags = fftw_planner_flags())
    plan_dct_inv = FFTW.plan_r2r(real_buf, FFTW.REDFT01, 3; flags = fftw_planner_flags())
    plan_fft_fwd = FFTW.plan_rfft(real_buf,   (1, 2);        flags = fftw_planner_flags())
    plan_fft_inv = FFTW.plan_irfft(hat_buf, Nx, (1, 2);      flags = fftw_planner_flags())

    kx² = [ 4/dx^2 * sin(π * (m - 1) / Nx)^2       for m in 1:div(Nx, 2) + 1 ]
    ky² = [ 4/dy^2 * sin(π * (n - 1) / Ny)^2       for n in 1:Ny ]
    kz² = [ 4/dz^2 * sin(π * (l - 1) / (2 * Nz))^2 for l in 1:Nz ]

    invλ = zeros(T, div(Nx, 2) + 1, Ny, Nz)
    @inbounds for l in 1:Nz, n in 1:Ny, m in 1:div(Nx, 2) + 1
        λ = -(kx²[m] + ky²[n] + kz²[l])
        invλ[m, n, l] = λ == 0 ? zero(T) : 1 / λ
    end

    dct_scale = 1 / T(2 * Nz)

    return CosineFFTPoissonSolver{T, typeof(plan_dct_fwd), typeof(plan_dct_inv),
                                  typeof(plan_fft_fwd), typeof(plan_fft_inv),
                                  typeof(real_buf), typeof(hat_buf)}(
        plan_dct_fwd, plan_dct_inv, plan_fft_fwd, plan_fft_inv,
        real_buf, real_buf2, hat_buf, invλ, T(dct_scale)
    )
end

function solve!(p::Field{CellCenter,T}, source::Field{CellCenter,T},
                solver::CosineFFTPoissonSolver{T}) where {T}
    m  = source.mesh
    ii, jj, kk = interior(m)

    @views solver.real_buf .= source.data[ii, jj, kk]

    mul!(solver.real_buf2, solver.plan_dct_fwd, solver.real_buf)
    mul!(solver.hat_buf,   solver.plan_fft_fwd, solver.real_buf2)

    @inbounds solver.hat_buf .*= solver.inv_eigenvalues

    mul!(solver.real_buf2, solver.plan_fft_inv, solver.hat_buf)
    mul!(solver.real_buf,  solver.plan_dct_inv, solver.real_buf2)

    solver.real_buf .*= solver.dct_scale

    @views p.data[ii, jj, kk] .= solver.real_buf
    return p
end

# -------------------------------------------------------------------------
# CGPoissonSolver — matrix-free preconditioned conjugate gradient.
# -------------------------------------------------------------------------

"""
    CGPoissonSolver(mesh::RegularGrid; tol=1e-8, maxit=200,
                    bc_top=:neumann, bc_bottom=:neumann,
                    bc_xy=:periodic)

Matrix-free conjugate-gradient Poisson solver on the 7-point discrete
Laplacian. Handles any BC combination that keeps the operator symmetric
positive semi-definite (periodic / Neumann; Dirichlet needs a Modified
kernel — not implemented). Slower than the FFT direct solvers but works
on immersed geometry, non-uniform stretching, etc.

Preconditioner: diagonal-Jacobi (constant `-2(1/dx² + 1/dy² + 1/dz²)`),
which for a uniform-grid Laplacian is just a scaling. Cheap and always
positive-definite.

Convergence: `‖r‖ < tol · ‖b‖`. Not guaranteed within `maxit`; a warning
is emitted if the iteration hits the cap.
"""
mutable struct CGPoissonSolver{T, RA<:Array{T,3}} <: AbstractPressureSolver
    mesh::RegularGrid{T}
    tol::T
    maxit::Int
    bc_top::Symbol
    bc_bottom::Symbol
    bc_xy::Symbol
    # scratch buffers on interior sizes
    r::RA
    p_dir::RA
    Ap::RA
    z::RA
    last_iter::Base.RefValue{Int}
    last_residual::Base.RefValue{T}
end

function CGPoissonSolver(mesh::RegularGrid{T};
                         tol::Real = 1e-8, maxit::Int = 200,
                         bc_top::Symbol = :neumann,
                         bc_bottom::Symbol = :neumann,
                         bc_xy::Symbol = :periodic) where {T}
    Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
    r    = zeros(T, Nx, Ny, Nz)
    p_dir= zeros(T, Nx, Ny, Nz)
    Ap   = zeros(T, Nx, Ny, Nz)
    z    = zeros(T, Nx, Ny, Nz)
    return CGPoissonSolver{T, typeof(r)}(mesh, T(tol), maxit,
                                          bc_top, bc_bottom, bc_xy,
                                          r, p_dir, Ap, z,
                                          Ref(0), Ref(zero(T)))
end

# 7-point Laplacian evaluated on an interior-sized array, applying BCs
# implicitly (periodic wrap or Neumann mirror).
function _apply_laplacian!(out::Array{T,3}, u::Array{T,3},
                           dx::T, dy::T, dz::T,
                           bc_top::Symbol, bc_bottom::Symbol, bc_xy::Symbol) where {T}
    Nx, Ny, Nz = size(u)
    idx2 = 1 / dx^2
    idy2 = 1 / dy^2
    idz2 = 1 / dz^2
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        # x neighbours (periodic wrap always)
        ip = i == Nx ? 1  : i + 1
        im = i == 1  ? Nx : i - 1
        # y neighbours
        jp = j == Ny ? 1  : j + 1
        jm = j == 1  ? Ny : j - 1
        # z neighbours
        kp = if k == Nz
                bc_top == :periodic ? 1 : k     # Neumann mirror ⇒ same index
             else
                k + 1
             end
        km = if k == 1
                bc_bottom == :periodic ? Nz : k
             else
                k - 1
             end
        c = u[i, j, k]
        out[i, j, k] = (u[ip, j, k] - 2c + u[im, j, k]) * idx2 +
                       (u[i, jp, k] - 2c + u[i, jm, k]) * idy2 +
                       (u[i, j,  kp] - 2c + u[i, j, km]) * idz2
    end
    return out
end

function solve!(p::Field{CellCenter,T}, source::Field{CellCenter,T},
                solver::CGPoissonSolver{T}) where {T}
    m  = source.mesh
    ii, jj, kk = interior(m)

    # b (source), zero-mean so the singular system is compatible
    b = Array(view(source.data, ii, jj, kk))
    b .-= sum(b) / length(b)
    b_norm = sqrt(sum(abs2, b))
    b_norm == 0 && (fill!(view(p.data, ii, jj, kk), 0); return p)

    # initial guess = 0
    x = fill!(view(p.data, ii, jj, kk) |> collect, 0)

    r    = solver.r    .= b
    pdir = solver.p_dir .= r
    z    = solver.z    .= r     # diagonal preconditioner is identity here
    rz_old = sum(r .* z)

    for k in 1:solver.maxit
        _apply_laplacian!(solver.Ap, pdir, m.dx, m.dy, m.dz,
                          solver.bc_top, solver.bc_bottom, solver.bc_xy)
        pap = sum(pdir .* solver.Ap)
        α   = rz_old / pap
        @. x = x + α * pdir
        @. r = r - α * solver.Ap
        res_norm = sqrt(sum(abs2, r))
        solver.last_iter[] = k
        solver.last_residual[] = res_norm
        if res_norm < solver.tol * b_norm
            break
        end
        z .= r     # identity preconditioner
        rz_new = sum(r .* z)
        β = rz_new / rz_old
        @. pdir = z + β * pdir
        rz_old = rz_new
        k == solver.maxit && @warn "CGPoissonSolver hit maxit=$k with ‖r‖=$res_norm"
    end
    # zero-mean projection on the solution
    x .-= sum(x) / length(x)
    view(p.data, ii, jj, kk) .= x
    return p
end

end # module
