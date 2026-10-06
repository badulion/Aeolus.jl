module PressureSolvers

using FFTW
using LinearAlgebra: mul!
using ..RunLog: current_domain   # structured events (LOG-a, #333)
using ..Grids
using ..Grids: RegularGrid, interior, CellCenter
using ..Fields
using ..Fields: Field
using ..BoundaryConditions: FieldBoundaryConditions, axis_poisson_bc
using ..Operators: fftw_planner_flags   # the one FFTW planning policy (#245)
using ..Backends: backend, launch!, apply_transform!

export AbstractPressureSolver, FFTPoissonSolver, CosineFFTPoissonSolver,
       CGPoissonSolver, solve!, poisson_axis_bcs

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
    b = backend(solver.src_buffer)
    buf, hat = solver.src_buffer, solver.hat_buffer
    copy_interior_in!(b, buf, source)                      # interior → contiguous buffer
    apply_transform!(b, hat, solver.forward_plan, buf)     # forward transform
    # divide by eigenvalues (zero mode annihilated by inv_eigenvalues[1,1,1] = 0)
    scale_by!(b, hat, solver.inv_eigenvalues)
    apply_transform!(b, buf, solver.inverse_plan, hat)     # inverse transform
    copy_interior_out!(b, p, buf)                          # write back
    return p
end

# ---- pointwise kernels of the spectral solvers (KA-h, #298) -----------------
#
# Bodies for `Backends.launch!`: each writes only its own point, so they give
# the same bits serially, threaded, and as the broadcasts they replace
# (`buf .= src[I...]`, `hat .*= invλ`, `buf .*= s`). The transforms between
# them run from the host (`apply_transform!`), never inside a kernel.

@inline function _copy_in_body!(i, j, k, buf, src, oi, oj, ok)
    @inbounds buf[i, j, k] = src[i + oi, j + oj, k + ok]
    return nothing
end
@inline function _copy_out_body!(i, j, k, dst, buf, oi, oj, ok)
    @inbounds dst[i + oi, j + oj, k + ok] = buf[i, j, k]
    return nothing
end
@inline function _copy_out_scaled_body!(i, j, k, dst, buf, oi, oj, ok, s)
    @inbounds dst[i + oi, j + oj, k + ok] = buf[i, j, k] * s
    return nothing
end
@inline function _scale_by_body!(i, j, k, a, w)
    @inbounds a[i, j, k] = a[i, j, k] * w[i, j, k]
    return nothing
end

_ranges(a::AbstractArray{<:Any,3}) = (1:size(a, 1), 1:size(a, 2), 1:size(a, 3))

function _check_interior(f::Field, buf::AbstractArray{<:Any,3})
    m = f.mesh
    (m.Nx, m.Ny, m.Nz) == size(buf) ||
        throw(DimensionMismatch("field interior $((m.Nx, m.Ny, m.Nz)) ≠ solver buffer $(size(buf))"))
    checkbounds(f.data, interior(m)...)
    return (m.Hx, m.Hy, m.Hz)
end

"""
    copy_interior_in!(backend, buf, f::Field) -> buf

`buf[i, j, k] = f`'s interior value at `(i, j, k)` (`buf` has the
interior's size). A pointwise kernel (KA-h, #298).
"""
function copy_interior_in!(b, buf::AbstractArray{T,3}, f::Field) where {T}
    oi, oj, ok = _check_interior(f, buf)
    launch!(b, _copy_in_body!, _ranges(buf), buf, f.data, oi, oj, ok)
    return buf
end

"""
    copy_interior_out!(backend, f::Field, buf[, s]) -> f

Write `buf` (interior-sized), times `s` if given, into `f`'s interior;
`f`'s halos are left untouched. A pointwise kernel (KA-h, #298).
"""
function copy_interior_out!(b, f::Field, buf::AbstractArray{T,3}) where {T}
    oi, oj, ok = _check_interior(f, buf)
    launch!(b, _copy_out_body!, _ranges(buf), f.data, buf, oi, oj, ok)
    return f
end
function copy_interior_out!(b, f::Field, buf::AbstractArray{T,3}, s) where {T}
    oi, oj, ok = _check_interior(f, buf)
    launch!(b, _copy_out_scaled_body!, _ranges(buf), f.data, buf, oi, oj, ok, s)
    return f
end

"""
    scale_by!(backend, a, w) -> a

`a .*= w` (same size; `a` may be complex, `w` real) as a pointwise kernel:
the division by the Laplacian eigenvalues in spectral space (KA-h, #298).
"""
function scale_by!(b, a::AbstractArray{<:Any,3}, w::AbstractArray{<:Any,3})
    size(a) == size(w) || throw(DimensionMismatch("scale_by!: $(size(a)) vs $(size(w))"))
    launch!(b, _scale_by_body!, _ranges(a), a, w)
    return a
end

# -------------------------------------------------------------------------
# CosineFFTPoissonSolver — per-axis periodic (FFT) or Neumann (DCT-II).
# -------------------------------------------------------------------------

const _POISSON_AXIS_KINDS = (:periodic, :neumann)

function _check_axis_bc(name, bc::Symbol)
    bc in _POISSON_AXIS_KINDS ||
        throw(ArgumentError("$name must be :periodic or :neumann, got :$bc"))
    return bc
end

"""
    CosineFFTPoissonSolver(mesh::RegularGrid;
                           bc_x = :periodic, bc_y = :periodic, bc_z = :neumann)

Direct Poisson solver on a `RegularGrid` whose three axes are each either
**periodic** or **homogeneous Neumann** (both faces of the axis), with
cell-centred source and solution. The default is the channel
(periodic `x, y`, walls in `z`); `bc_x = bc_y = bc_z = :neumann` is the
closed box of a nested or open-boundary child, whose normal velocity is
prescribed on every face so that `∂φ/∂n = 0` everywhere.

Each axis is diagonalised separately (Schumann & Sweet 1988):

* a **periodic** axis by the real FFT, eigenvalue
  ``4/Δ^2 \\sin^2(π m/N)``, ``m = 0 … N-1``;
* a **Neumann** axis by the DCT-II (FFTW `REDFT10`, inverse `REDFT01`
  = DCT-III), eigenvalue ``4/Δ^2 \\sin^2(π m/2N)``. The DCT-II even-extends
  the data about the half-integer positions ``i = 1/2`` and
  ``i = N + 1/2``, which are the faces of the cell-centred grid, so the
  mirror ghosts `φ₀ = φ₁`, `φ_{N+1} = φ_N` (zero face gradient) are
  enforced implicitly. The `2N` is the period of that even extension.

```math
λ_{m,n,l} = -\\frac{4}{Δx^2}\\sin^2\\frac{π m}{N_x^*}
            -\\frac{4}{Δy^2}\\sin^2\\frac{π n}{N_y^*}
            -\\frac{4}{Δz^2}\\sin^2\\frac{π l}{N_z^*},
\\qquad N^* = N \\text{ (periodic)},\\quad 2N \\text{ (Neumann)}.
```

The eigenvalues are those of the *discrete* 7-point Laplacian with that
closure, so `L_h p = source` is satisfied to round-off. The cost is one
`O(N log N)` transform pair; a DCT costs about the same as a real FFT of
the same length.

**Compatibility.** With no Dirichlet axis the `(0,0,0)` mode has zero
eigenvalue: `p` is fixed up to a constant (the solver returns the
zero-mean solution) and only the zero-mean part of `source` can be
matched. The discrete compatibility condition is ``\\sum_{ijk} s_{ijk} = 0``
(Gauss: the net boundary flux of `∇φ`, zero for homogeneous Neumann and
periodic faces). An incompatible source with mean ``\\bar s = Q/V`` — a
net source `Q` in volume `V`, e.g. a non-zero net inflow through prescribed
boundary velocities — is solved as `L_h p = s − \\bar s`: the residual
`L_h p − s` is exactly the uniform `−\\bar s`. That is why the open-boundary
mass-flux correction must make the net boundary volume flux vanish
*before* the projection.

The default `(:periodic, :periodic, :neumann)` builds exactly the plans,
eigenvalues and normalisation of the original channel solver, so results
are bit-for-bit unchanged.

Reference: Schumann, U. & Sweet, R. A. (1988). Fast Fourier transforms for
direct solution of Poisson's equation with staggered boundary conditions.
*J. Comput. Phys.*, 75, 123-137.
"""
struct CosineFFTPoissonSolver{T,
                              PDF, PDI, PFF, PFI,
                              RA<:Array{T,3},
                              CA<:Union{Nothing,Array{Complex{T},3}}} <: AbstractPressureSolver
    plan_dct_fwd::PDF        # DCT-II  along the Neumann axes (REDFT10), or nothing
    plan_dct_inv::PDI        # DCT-III along the Neumann axes (REDFT01), or nothing
    plan_fft_fwd::PFF        # rfft    along the periodic axes, or nothing
    plan_fft_inv::PFI        # irfft   along the periodic axes, or nothing
    real_buf::RA
    real_buf2::RA            # scratch — plan_r2r is out-of-place
    hat_buf::CA              # spectral buffer (nothing when no axis is periodic)
    inv_eigenvalues::Array{T,3}
    dct_scale::T             # 1 / ∏(2N) over the Neumann axes (DCT-II × DCT-III normaliser)
    bc_x::Symbol
    bc_y::Symbol
    bc_z::Symbol
end

# Squared modified wavenumbers 4/Δ² sin²(π m / N*) of one axis, for the
# `n` retained modes (n = N, or N÷2+1 on the halved rfft axis).
_k2(Δ, N, n, bc::Symbol) = bc === :periodic ?
    [ 4/Δ^2 * sin(π * (m - 1) / N)^2       for m in 1:n ] :
    [ 4/Δ^2 * sin(π * (m - 1) / (2 * N))^2 for m in 1:n ]

function CosineFFTPoissonSolver(mesh::RegularGrid{T};
                                bc_x::Symbol = :periodic,
                                bc_y::Symbol = :periodic,
                                bc_z::Symbol = :neumann) where {T}
    _check_axis_bc("bc_x", bc_x); _check_axis_bc("bc_y", bc_y); _check_axis_bc("bc_z", bc_z)
    Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
    dx, dy, dz = mesh.dx, mesh.dy, mesh.dz
    N   = (Nx, Ny, Nz)
    bcs = (bc_x, bc_y, bc_z)
    pdims = Tuple(d for d in 1:3 if bcs[d] === :periodic)
    ndims_ = Tuple(d for d in 1:3 if bcs[d] === :neumann)

    real_buf  = zeros(T, Nx, Ny, Nz)
    real_buf2 = zeros(T, Nx, Ny, Nz)

    # The rfft halves the first periodic axis.
    nhat = collect(N)
    isempty(pdims) || (nhat[pdims[1]] = div(N[pdims[1]], 2) + 1)
    hat_buf = isempty(pdims) ? nothing : zeros(Complex{T}, nhat...)

    plan_dct_fwd = isempty(ndims_) ? nothing :
        FFTW.plan_r2r(real_buf, FFTW.REDFT10, ndims_; flags = fftw_planner_flags())
    plan_dct_inv = isempty(ndims_) ? nothing :
        FFTW.plan_r2r(real_buf, FFTW.REDFT01, ndims_; flags = fftw_planner_flags())
    plan_fft_fwd = isempty(pdims) ? nothing :
        FFTW.plan_rfft(real_buf, pdims; flags = fftw_planner_flags())
    plan_fft_inv = isempty(pdims) ? nothing :
        FFTW.plan_irfft(hat_buf, N[pdims[1]], pdims; flags = fftw_planner_flags())

    kx² = _k2(dx, Nx, nhat[1], bc_x)
    ky² = _k2(dy, Ny, nhat[2], bc_y)
    kz² = _k2(dz, Nz, nhat[3], bc_z)

    invλ = zeros(T, nhat...)
    @inbounds for l in 1:nhat[3], n in 1:nhat[2], m in 1:nhat[1]
        λ = -(kx²[m] + ky²[n] + kz²[l])
        invλ[m, n, l] = λ == 0 ? zero(T) : 1 / λ
    end

    dct_scale = 1 / T(prod((2 * N[d] for d in ndims_); init = 1))

    return CosineFFTPoissonSolver{T, typeof(plan_dct_fwd), typeof(plan_dct_inv),
                                  typeof(plan_fft_fwd), typeof(plan_fft_inv),
                                  typeof(real_buf), typeof(hat_buf)}(
        plan_dct_fwd, plan_dct_inv, plan_fft_fwd, plan_fft_inv,
        real_buf, real_buf2, hat_buf, invλ, T(dct_scale), bc_x, bc_y, bc_z
    )
end

"""
    poisson_axis_bcs(solver) -> (bc_x, bc_y, bc_z)

The per-axis boundary kinds (`:periodic` / `:neumann`) a pressure solver
was built with.
"""
poisson_axis_bcs(s::CosineFFTPoissonSolver) = (bc_x = s.bc_x, bc_y = s.bc_y, bc_z = s.bc_z)
poisson_axis_bcs(::FFTPoissonSolver) = (bc_x = :periodic, bc_y = :periodic, bc_z = :periodic)

"""
    poisson_axis_bcs(bcs::FieldBoundaryConditions) -> (bc_x, bc_y, bc_z)

The per-axis pressure-Poisson kinds implied by the six *velocity* faces
of `bcs`, each read through
[`axis_poisson_bc`](@ref Aeolus.BoundaryConditions.axis_poisson_bc). The
two faces of an axis must give the same kind; a periodic face opposite a
Neumann one is not a well-defined Poisson problem for these solvers and
throws an `ArgumentError`. Splat the result into a solver constructor:
`CosineFFTPoissonSolver(mesh; poisson_axis_bcs(bcs)...)`.
"""
function poisson_axis_bcs(bcs::FieldBoundaryConditions)
    pair(name, lo, hi) = begin
        a, b = axis_poisson_bc(lo), axis_poisson_bc(hi)
        a === b || throw(ArgumentError(
            "unsupported $name boundary pair for the pressure solve: " *
            "$(nameof(typeof(lo))) (:$a) against $(nameof(typeof(hi))) (:$b); " *
            "both faces of an axis must be periodic or both Neumann"))
        a
    end
    return (bc_x = pair("x", bcs.west, bcs.east),
            bc_y = pair("y", bcs.south, bcs.north),
            bc_z = pair("z", bcs.bottom, bcs.top))
end

function solve!(p::Field{CellCenter,T}, source::Field{CellCenter,T},
                solver::CosineFFTPoissonSolver{T}) where {T}
    b = backend(solver.real_buf)
    cur, other = solver.real_buf, solver.real_buf2   # the data lives in `cur`
    copy_interior_in!(b, cur, source)

    if solver.plan_dct_fwd !== nothing                     # DCT-II along the Neumann axes
        apply_transform!(b, other, solver.plan_dct_fwd, cur)
        cur, other = other, cur
    end

    if solver.plan_fft_fwd === nothing                     # all Neumann: real spectrum
        scale_by!(b, cur, solver.inv_eigenvalues)
    else
        apply_transform!(b, solver.hat_buf, solver.plan_fft_fwd, cur)
        scale_by!(b, solver.hat_buf, solver.inv_eigenvalues)
        apply_transform!(b, cur, solver.plan_fft_inv, solver.hat_buf)
    end

    if solver.plan_dct_inv !== nothing                     # DCT-III back
        apply_transform!(b, other, solver.plan_dct_inv, cur)
        cur, other = other, cur
    end

    # the DCT normalisation fused into the write-back (same bits as the
    # former `real_buf .*= dct_scale` followed by the copy)
    copy_interior_out!(b, p, cur, solver.dct_scale)
    return p
end

# -------------------------------------------------------------------------
# CGPoissonSolver — matrix-free preconditioned conjugate gradient.
# -------------------------------------------------------------------------

"""
    CGPoissonSolver(mesh::RegularGrid; tol=1e-8, maxit=200,
                    bc_top=:neumann, bc_bottom=:neumann,
                    bc_xy=:periodic, bc_x=bc_xy, bc_y=bc_xy)

Matrix-free conjugate-gradient Poisson solver on the 7-point discrete
Laplacian. Every axis is periodic or homogeneous Neumann: `bc_x`, `bc_y`
(`bc_xy` sets both) and `bc_bottom`, `bc_top` in `z` (any value other than
`:periodic` there is Neumann; the two z faces must agree). A Neumann face
uses the mirror ghost `φ₀ = φ₁`, i.e. zero flux through the face — the
same discrete operator [`CosineFFTPoissonSolver`](@ref) diagonalises with
the matching `bc_x`, `bc_y`, `bc_z`. Slower than the direct solvers; its
use is as an independent check and for operators they cannot diagonalise.

Before #251 `bc_xy` was stored but ignored: x and y were wrapped
periodically whatever it said, so `bc_xy = :neumann` silently returned
the periodic solution.

Preconditioner: none (the diagonal of the uniform-grid Laplacian is
constant in the interior, so Jacobi would only rescale).

Convergence: `‖r‖ < tol · ‖b‖`. Not guaranteed within `maxit`; a warning
is emitted if the iteration hits the cap.
"""
mutable struct CGPoissonSolver{T, RA<:Array{T,3}} <: AbstractPressureSolver
    mesh::RegularGrid{T}
    tol::T
    maxit::Int
    bc_top::Symbol
    bc_bottom::Symbol
    bc_x::Symbol
    bc_y::Symbol
    # scratch buffers on interior sizes
    r::RA
    p_dir::RA
    Ap::RA
    z::RA
    b::RA                          # zero-mean source
    x::RA                          # iterate
    w::RA                          # scratch of the dot products (no allocation per solve, #295)
    last_iter::Base.RefValue{Int}
    last_residual::Base.RefValue{T}
end

function CGPoissonSolver(mesh::RegularGrid{T};
                         tol::Real = 1e-8, maxit::Int = 200,
                         bc_top::Symbol = :neumann,
                         bc_bottom::Symbol = :neumann,
                         bc_xy::Symbol = :periodic,
                         bc_x::Symbol = bc_xy,
                         bc_y::Symbol = bc_xy) where {T}
    _check_axis_bc("bc_x", bc_x); _check_axis_bc("bc_y", bc_y)
    (bc_top === :periodic) == (bc_bottom === :periodic) ||
        throw(ArgumentError("CGPoissonSolver: bc_top and bc_bottom must both be periodic or both not"))
    Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
    r    = zeros(T, Nx, Ny, Nz)
    p_dir= zeros(T, Nx, Ny, Nz)
    Ap   = zeros(T, Nx, Ny, Nz)
    z    = zeros(T, Nx, Ny, Nz)
    return CGPoissonSolver{T, typeof(r)}(mesh, T(tol), maxit,
                                          bc_top, bc_bottom, bc_x, bc_y,
                                          r, p_dir, Ap, z, zero(r), zero(r), zero(r),
                                          Ref(0), Ref(zero(T)))
end

poisson_axis_bcs(s::CGPoissonSolver) =
    (bc_x = s.bc_x, bc_y = s.bc_y, bc_z = s.bc_top === :periodic ? :periodic : :neumann)

# Neighbour index across a face of an axis of length N: periodic wrap, or
# the cell itself (Neumann mirror ghost ⇒ zero flux through the face).
@inline _nbr_hi(i, N, periodic::Bool) = i < N ? i + 1 : (periodic ? 1 : N)
@inline _nbr_lo(i, N, periodic::Bool) = i > 1 ? i - 1 : (periodic ? N : 1)

# 7-point Laplacian evaluated on an interior-sized array, applying BCs
# implicitly (periodic wrap or Neumann mirror) on every axis.
function _apply_laplacian!(out::Array{T,3}, u::Array{T,3},
                           dx::T, dy::T, dz::T,
                           bc_top::Symbol, bc_bottom::Symbol,
                           bc_x::Symbol, bc_y::Symbol) where {T}
    Nx, Ny, Nz = size(u)
    idx2 = 1 / dx^2
    idy2 = 1 / dy^2
    idz2 = 1 / dz^2
    px = bc_x === :periodic
    py = bc_y === :periodic
    pt = bc_top === :periodic
    pb = bc_bottom === :periodic
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        ip = _nbr_hi(i, Nx, px); im = _nbr_lo(i, Nx, px)
        jp = _nbr_hi(j, Ny, py); jm = _nbr_lo(j, Ny, py)
        kp = _nbr_hi(k, Nz, pt); km = _nbr_lo(k, Nz, pb)
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
    b = copyto!(solver.b, view(source.data, ii, jj, kk))
    b .-= sum(b) / length(b)
    b_norm = sqrt(sum(abs2, b))
    b_norm == 0 && (fill!(view(p.data, ii, jj, kk), 0); return p)

    # initial guess = 0
    x = fill!(solver.x, 0)

    r    = solver.r    .= b
    pdir = solver.p_dir .= r
    z    = solver.z    .= r     # diagonal preconditioner is identity here
    w = solver.w        # r·z as sum(r .* z), the product in scratch: same bits
    rz_old = sum(w .= r .* z)

    for k in 1:solver.maxit
        _apply_laplacian!(solver.Ap, pdir, m.dx, m.dy, m.dz,
                          solver.bc_top, solver.bc_bottom, solver.bc_x, solver.bc_y)
        pap = sum(w .= pdir .* solver.Ap)
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
        rz_new = sum(w .= r .* z)
        β = rz_new / rz_old
        @. pdir = z + β * pdir
        rz_old = rz_new
        k == solver.maxit &&
            @warn("CGPoissonSolver hit maxit=$k with ‖r‖=$res_norm",
                  subsystem = :pressure, domain = current_domain(), solver = :cg,
                  iterations = k, residual = res_norm,
                  relative_residual = res_norm / b_norm, tol = solver.tol)
    end
    # zero-mean projection on the solution
    x .-= sum(x) / length(x)
    view(p.data, ii, jj, kk) .= x
    return p
end

end # module
