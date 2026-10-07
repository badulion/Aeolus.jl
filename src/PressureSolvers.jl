module PressureSolvers

using FFTW
using ..RunLog: current_domain   # structured events (LOG-a, #333)
using ..Grids
using ..Grids: RegularGrid, interior, CellCenter
using ..Fields
using ..Fields: Field
using ..BoundaryConditions: FieldBoundaryConditions, axis_poisson_bc, _halo_periodic!, _halo_mirror!
using ..Operators: fftw_planner_flags   # the one FFTW planning policy (#245)
using ..Backends: backend, launch!, launch_weighted!, runs_serially, apply_transform!,
                  pcg!, PCGWorkspace, PointOperator, IdentityPreconditioner

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
# ColumnTridiagonalSolver — horizontal r2r transform + Thomas in z.
# -------------------------------------------------------------------------

"""
    ColumnTridiagonalSolver(T, (Nx, Ny, Nz), (dx, dy); lower, upper, hweight,
                            measure, inv_coupling, gauge,
                            bc_x = :periodic, bc_y = :periodic)

The direct FFT–tridiagonal solver (Schumann & Sweet 1988; Press et al.
2007, *Numerical Recipes* 3rd ed. §20.4) of every cell-centred Poisson
problem whose horizontal operator is the uniform 3-point stencil and whose
coefficients depend on height only:

```math
h_k\\,(δ_x^2 + δ_y^2)φ_k + a_k(φ_{k-1} - φ_k) + c_k(φ_{k+1} - φ_k) = b_k,
```

with `a_k = lower[k]`, `c_k = upper[k]` (`a_1 = c_{N_z} = 0`: zero flux
through the bottom and top faces), `h_k = hweight[k] > 0`. It is the one
implementation behind two operators:

* the conservative stretched-z Laplacian of
  [`StretchedPoissonSolver`](@ref Aeolus.StretchedGrids.StretchedPoissonSolver)
  (`h = 1`, `a_k = 1/(Δz_c[k] Δz_f[k])`);
* the density-weighted operator `∇·(w(z)∇φ)` of the sound-proof cores,
  [`DirectWeightedPoissonSolver`](@ref Aeolus.Anelastic.DirectWeightedPoissonSolver)
  (`h = w_c`, `a_k = w_f[k]/Δz²`).

`x` and `y` are each `:periodic` (half-complex DFT, FFTW `R2HC`/`HC2R`) or
`:neumann` (DCT-II/III, `REDFT10`/`REDFT01`), diagonalised with the exact
3-point eigenvalues `λ = 4/Δ² sin²(πm/N)` or `4/Δ² sin²(πm/2N)`; each
horizontal wavenumber leaves the tridiagonal system
`a_k φ̂_{k-1} − (a_k + c_k + λ h_k) φ̂_k + c_k φ̂_{k+1} = b̂_k`, which is
strictly diagonally dominant for `λ > 0` and is solved by the Thomas
algorithm with the factorisation computed once at construction.

The mean mode (`λ = 0`) is singular. It is the 1-D problem
`G_{k+1} − G_k = m_k b̂_k` for the face flux `G_k = (φ̂_k − φ̂_{k−1}) / ι_k`
(`m = measure`, `ι = inv_coupling`, so `a_k = 1/(m_k ι_k)` and
`c_k = 1/(m_k ι_{k+1})`) with `G_1 = 0`: the `m`-weighted mean of the source
is removed (the compatibility condition of the closed column), the flux is
integrated up, and the result is shifted to zero `gauge`-weighted mean,
`Σ_k gauge_k Σ_{ij} φ_{ijk} = 0`.

Cost: one 2-D transform pair and `O(N_x N_y N_z)` for the column solves;
exact to round-off. Periodic `z` is not supported.
"""
struct ColumnTridiagonalSolver{T, PF, PI} <: AbstractPressureSolver
    plan_fwd::PF
    plan_inv::PI
    buf::Array{T,3}          # transform buffer, interior size
    buf2::Array{T,3}
    cp::Array{T,3}           # Thomas: modified upper diagonal c'_k
    inv_den::Array{T,3}      # Thomas: 1 / (b_k - a_k c'_{k-1})
    lower::Vector{T}         # a_k: coefficient of φ_{k-1} (0 at k = 1)
    measure::Vector{T}       # m_k of the mean-mode column (length Nz)
    inv_coupling::Vector{T}  # ι_k (face k = bottom of cell k, length Nz+1)
    gauge::Vector{T}         # weights of the zero-mean gauge (length Nz)
    measure_sum::T           # Σ m_k
    gauge_sum::T             # Σ gauge_k
    col::Vector{T}           # the mean-mode column (scratch)
    norm::T                  # 1 / (forward × inverse transform scale)
    bc_x::Symbol
    bc_y::Symbol
end

_r2r_kinds(bc::Symbol) = bc === :periodic ? (FFTW.R2HC, FFTW.HC2R, 1) :
                         bc === :neumann  ? (FFTW.REDFT10, FFTW.REDFT01, 2) :
                         throw(ArgumentError("column tridiagonal Poisson solver: lateral BC " *
                                             "must be :periodic or :neumann, got :$bc"))

# Eigenvalue magnitudes 4/Δ² sin²(·) of the 3-point second difference in the
# r2r basis: half-complex index p ↔ frequency min(p, N-p) for the DFT, p for
# the DCT-II (period 2N).
function _r2r_eigen(::Type{T}, N::Int, Δ::T, bc::Symbol) where {T}
    if bc === :periodic
        return T[4 / Δ^2 * sin(π * min(p, N - p) / N)^2 for p in 0:N-1]
    else
        return T[4 / Δ^2 * sin(π * p / (2N))^2 for p in 0:N-1]
    end
end

function ColumnTridiagonalSolver(::Type{T}, (Nx, Ny, Nz)::NTuple{3,Int}, (dx, dy);
                                 lower::AbstractVector, upper::AbstractVector,
                                 hweight::AbstractVector, measure::AbstractVector,
                                 inv_coupling::AbstractVector, gauge::AbstractVector,
                                 bc_x::Symbol = :periodic, bc_y::Symbol = :periodic) where {T}
    length(lower) == length(upper) == length(hweight) == length(measure) ==
        length(gauge) == Nz && length(inv_coupling) == Nz + 1 ||
        throw(DimensionMismatch("column coefficients must have length Nz = $Nz " *
                                "(inv_coupling: Nz + 1)"))
    lower[1] == 0 && upper[Nz] == 0 ||
        throw(ArgumentError("the bottom and top faces carry no flux: lower[1] = upper[Nz] = 0"))
    kx_f, kx_i, sx = _r2r_kinds(bc_x)
    ky_f, ky_i, sy = _r2r_kinds(bc_y)
    buf  = zeros(T, Nx, Ny, Nz)
    buf2 = zeros(T, Nx, Ny, Nz)
    plan_fwd = FFTW.plan_r2r(buf, (kx_f, ky_f), (1, 2); flags = fftw_planner_flags())
    plan_inv = FFTW.plan_r2r(buf, (kx_i, ky_i), (1, 2); flags = fftw_planner_flags())
    norm = one(T) / T(sx * Nx * sy * Ny)

    lo = T.(collect(lower)); up = T.(collect(upper)); hw = T.(collect(hweight))
    λx = _r2r_eigen(T, Nx, T(dx), bc_x)
    λy = _r2r_eigen(T, Ny, T(dy), bc_y)
    cp = zeros(T, Nx, Ny, Nz); inv_den = zeros(T, Nx, Ny, Nz)
    @inbounds for q in 1:Ny, p in 1:Nx
        λ = λx[p] + λy[q]
        (p == 1 && q == 1) && continue          # the mean mode: integrated directly
        cprev = zero(T)
        for k in 1:Nz
            b   = -(lo[k] + up[k]) - λ * hw[k]
            den = b - lo[k] * cprev
            inv_den[p, q, k] = 1 / den
            cprev = up[k] / den
            cp[p, q, k] = cprev
        end
    end
    m = T.(collect(measure)); g = T.(collect(gauge))
    return ColumnTridiagonalSolver{T, typeof(plan_fwd), typeof(plan_inv)}(
        plan_fwd, plan_inv, buf, buf2, cp, inv_den, lo, m, T.(collect(inv_coupling)), g,
        sum(m), sum(g), zeros(T, Nz), norm, bc_x, bc_y)
end

poisson_axis_bcs(s::ColumnTridiagonalSolver) = (bc_x = s.bc_x, bc_y = s.bc_y, bc_z = :neumann)

function solve!(p::Field{CellCenter,T}, source::Field{CellCenter,T},
                s::ColumnTridiagonalSolver{T}) where {T}
    be = backend(s.buf)
    Nx, Ny, Nz = size(s.buf)
    copy_interior_in!(be, s.buf, source)
    apply_transform!(be, s.buf2, s.plan_fwd, s.buf)
    b = s.buf2
    @views s.col .= b[1, 1, :]          # the mean mode, solved separately
    # one tridiagonal solve per horizontal wavenumber, serial in z within
    # the work item; a work item is a slab of ky rows, see _thomas_slab!
    checkbounds(b, 1:Nx, 1:Ny, 1:Nz)
    size(s.cp) == size(s.inv_den) == size(b) && length(s.lower) == Nz ||
        throw(DimensionMismatch("ColumnTridiagonalSolver: inconsistent Thomas factors"))
    np = Nx * Ny * Nz
    nslab = runs_serially(be, np) ? 1 : min(Ny, Threads.nthreads())
    launch_weighted!(be, cld(np, nslab), _thomas_slab!, (1:nslab,),
                     b, s.cp, s.inv_den, s.lower, Nx, Ny, Nz, nslab)
    _mean_mode!(s)
    @views b[1, 1, :] .= s.col
    apply_transform!(be, s.buf, s.plan_inv, s.buf2)
    copy_interior_out!(be, p, s.buf, s.norm)
    return p
end

# The Thomas algorithm for the columns (pp, q) of the ky rows q of slab
# `sl` (of `nslab` contiguous slabs) of the transformed source `b`, in
# place, with the factorisation `cp` (c'_k) and `inv_den` (1/(b_k − a_k
# c'_{k−1})) computed at construction: forward elimination
# d'_k = (d_k − a_k d'_{k−1}) / den_k, then back substitution
# φ_k = d'_k − c'_k φ_{k+1}. Every column is serial in z and owned by one
# work item, and every operation is the one the former level-by-level loop
# did, so the bits are those of that loop whatever the slabs are: the slab
# count may follow the thread count (one slab, the old loop exactly, when
# the launch runs serially). The mean mode (1, 1) has cp = inv_den = 0 and
# is overwritten by `_mean_mode!`.
#
# Why slabs of rows and not one column per work item: with kx innermost
# the loop vectorises and streams through memory. One column per item
# walks z with a stride of Nx·Ny·8 bytes, a power of two on the usual
# grids, so every level of a column falls in the same L1 set: measured
# 4.6× slower than the old loop at 64×64×53 on one thread. One ky row per
# item still cost 1.2× (64×64×53) to 1.6× (192×192×53) on one thread (Nx
# contiguous values per level are too short a stream). A GPU backend will
# launch one column per item (Phase 81).
@inline function _thomas_slab!(sl, b, cp, inv_den, lower, Nx, Ny, Nz, nslab)
    q1 = (sl - 1) * Ny ÷ nslab + 1
    q2 = sl * Ny ÷ nslab
    @inbounds begin
        for q in q1:q2, pp in 1:Nx
            b[pp, q, 1] *= inv_den[pp, q, 1]
        end
        for k in 2:Nz
            a = lower[k]
            for q in q1:q2, pp in 1:Nx
                b[pp, q, k] = (b[pp, q, k] - a * b[pp, q, k-1]) * inv_den[pp, q, k]
            end
        end
        for k in Nz-1:-1:1, q in q1:q2, pp in 1:Nx
            b[pp, q, k] -= cp[pp, q, k] * b[pp, q, k+1]
        end
    end
    return nothing
end

# Column of the λ = 0 mode, in transform units (the operator is linear,
# so the horizontal-sum scaling of the transform carries through): remove
# the m-weighted mean of the source (the compatibility condition of the
# zero-flux ends), integrate the flux up from the zero bottom flux,
#   G_k = G_{k-1} + b_{k-1} m[k-1],   φ_k = φ_{k-1} + G_k ι[k],
# then remove the gauge-weighted mean of φ.
function _mean_mode!(s::ColumnTridiagonalSolver{T}) where {T}
    col, m, ι, g = s.col, s.measure, s.inv_coupling, s.gauge
    Nz = length(col)
    mb = zero(T)
    @inbounds for k in 1:Nz
        mb += col[k] * m[k]
    end
    mb /= s.measure_sum
    F = zero(T); φk = zero(T)
    @inbounds begin
        prev = col[1] - mb
        col[1] = zero(T)
        for k in 2:Nz
            F  += prev * m[k-1]
            φk += F * ι[k]
            prev = col[k] - mb
            col[k] = φk
        end
    end
    mφ = zero(T)
    @inbounds for k in 1:Nz
        mφ += col[k] * g[k]
    end
    col .-= mφ / s.gauge_sum
    return s
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

The iteration is [`Aeolus.Backends.pcg!`](@ref), the one CG driver
(KA-g, #300): the 7-point operator is a point kernel fused with `p·Ap`,
the reductions are canonical plane sums, so a solve (and its iteration
count) is identical at every thread count. The mean is removed from the
source, from every residual direction and from the solution.

Convergence: `‖r‖ ≤ tol · ‖b‖`. Not guaranteed within `maxit`; a warning
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
    # CG arrays with one halo layer, (Nx+2, Ny+2, Nz+2); interior 2:N+1
    b::RA                          # the source
    x::RA                          # iterate
    ws::PCGWorkspace{T,RA}         # r, z, p, Ap and the reduction partials (no allocation per solve)
    last_iter::Base.RefValue{Int}
    last_residual::Base.RefValue{T}
end

_unit_halo_ranges(Nx, Ny, Nz) = (2:Nx+1, 2:Ny+1, 2:Nz+1)

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
    b = zeros(T, mesh.Nx + 2, mesh.Ny + 2, mesh.Nz + 2)
    return CGPoissonSolver{T, typeof(b)}(mesh, T(tol), maxit,
                                          bc_top, bc_bottom, bc_x, bc_y,
                                          b, zero(b),
                                          PCGWorkspace(b; ranges = _unit_halo_ranges(mesh.Nx, mesh.Ny, mesh.Nz)),
                                          Ref(0), Ref(zero(T)))
end

poisson_axis_bcs(s::CGPoissonSolver) =
    (bc_x = s.bc_x, bc_y = s.bc_y, bc_z = s.bc_top === :periodic ? :periodic : :neumann)

"""
    fill_unit_halo!(a, bc_x, bc_y, bc_z) -> a

Fill the one-cell halo of `a` (size `(nx+2, ny+2, nz+2)`, interior
`2:n+1`): per axis `:periodic` wraps, anything else mirrors (the
homogeneous-Neumann ghost `φ₀ = φ₁`), x then y then z. The halo kernels of
`BoundaryConditions` (KA-f, #297): the halo of the CG iterates of
[`CGPoissonSolver`](@ref) and `DensityWeightedPoissonSolver`, and of the
multigrid levels.
"""
function fill_unit_halo!(a::AbstractArray{<:Any,3}, bcx::Symbol, bcy::Symbol, bcz::Symbol)
    for (ax, bc) in ((1, bcx), (2, bcy), (3, bcz))
        n = size(a, ax) - 2
        bc === :periodic ? _halo_periodic!(a, ax, 1, n) : _halo_mirror!(a, ax, 1, n)
    end
    return a
end

# the halo fill before each operator application of the CG driver
struct _UnitHalo
    bc_x::Symbol
    bc_y::Symbol
    bc_z::Symbol
end
(h::_UnitHalo)(a) = fill_unit_halo!(a, h.bc_x, h.bc_y, h.bc_z)

# The 7-point Laplacian at one point of a unit-halo array whose halo holds
# the periodic wrap or the Neumann mirror (the face difference is then
# zero: no flux through a Neumann face). A `PointOperator` body (KA-g,
# #300); the same values as the wrap-around indexing it replaces.
@inline function _laplacian_at(i, j, k, u, idx2, idy2, idz2)
    @inbounds begin
        c = u[i, j, k]
        return (u[i+1, j, k] - 2c + u[i-1, j, k]) * idx2 +
               (u[i, j+1, k] - 2c + u[i, j-1, k]) * idy2 +
               (u[i, j, k+1] - 2c + u[i, j, k-1]) * idz2
    end
end

@inline function _laplacian_body!(i, j, k, out, u, idx2, idy2, idz2)
    @inbounds out[i, j, k] = _laplacian_at(i, j, k, u, idx2, idy2, idz2)
    return nothing
end

"""
    _apply_laplacian!(out, u, dx, dy, dz, bc_top, bc_bottom, bc_x, bc_y) -> out

The 7-point Laplacian of [`CGPoissonSolver`](@ref) on interior-sized
arrays (a test helper: pads `u` with the BC halo, then launches the
operator its CG iterates).
"""
function _apply_laplacian!(out::Array{T,3}, u::Array{T,3},
                           dx::T, dy::T, dz::T,
                           bc_top::Symbol, bc_bottom::Symbol,
                           bc_x::Symbol, bc_y::Symbol) where {T}
    size(out) == size(u) || throw(DimensionMismatch("_apply_laplacian!: $(size(out)) vs $(size(u))"))
    Nx, Ny, Nz = size(u)
    up = zeros(T, Nx + 2, Ny + 2, Nz + 2); op = similar(up)
    up[2:Nx+1, 2:Ny+1, 2:Nz+1] .= u
    fill_unit_halo!(up, bc_x, bc_y, bc_top === :periodic ? :periodic : :neumann)
    launch!(backend(u), _laplacian_body!, _unit_halo_ranges(Nx, Ny, Nz), op, up,
            1 / dx^2, 1 / dy^2, 1 / dz^2)
    out .= view(op, 2:Nx+1, 2:Ny+1, 2:Nz+1)
    return out
end

"""
    copy_unit_halo_in!(backend, buf, f::Field) -> buf
    copy_unit_halo_out!(backend, f::Field, buf) -> f

Copy `f`'s interior into the interior `2:n+1` of the unit-halo array
`buf`, or back. Pointwise kernels (KA-g, #300).
"""
function copy_unit_halo_in!(b, buf::AbstractArray{T,3}, f::Field) where {T}
    m = f.mesh
    size(buf) == (m.Nx + 2, m.Ny + 2, m.Nz + 2) || throw(DimensionMismatch("unit-halo buffer $(size(buf))"))
    checkbounds(f.data, interior(m)...)
    launch!(b, _copy_in_body!, _unit_halo_ranges(m.Nx, m.Ny, m.Nz), buf, f.data, m.Hx - 1, m.Hy - 1, m.Hz - 1)
    return buf
end
function copy_unit_halo_out!(b, f::Field, buf::AbstractArray{T,3}) where {T}
    m = f.mesh
    size(buf) == (m.Nx + 2, m.Ny + 2, m.Nz + 2) || throw(DimensionMismatch("unit-halo buffer $(size(buf))"))
    checkbounds(f.data, interior(m)...)
    launch!(b, _copy_out_body!, _unit_halo_ranges(m.Nx, m.Ny, m.Nz), f.data, buf, m.Hx - 1, m.Hy - 1, m.Hz - 1)
    return f
end

function solve!(p::Field{CellCenter,T}, source::Field{CellCenter,T},
                solver::CGPoissonSolver{T}) where {T}
    m  = solver.mesh
    bk = backend(solver.b)
    copy_unit_halo_in!(bk, solver.b, source)
    A = PointOperator(_laplacian_at, 1 / m.dx^2, 1 / m.dy^2, 1 / m.dz^2;
                      prepare = _UnitHalo(solver.bc_x, solver.bc_y,
                                          solver.bc_top === :periodic ? :periodic : :neumann))
    st = pcg!(A, IdentityPreconditioner(), solver.x, solver.b, solver.ws;
              ranges = _unit_halo_ranges(m.Nx, m.Ny, m.Nz), tol = solver.tol,
              maxit = solver.maxit, project_mean = true, definite = :negative, backend = bk)
    solver.last_iter[] = st.iterations
    solver.last_residual[] = st.residual
    st.status === :maxit &&
        @warn("CGPoissonSolver hit maxit=$(solver.maxit) with ‖r‖=$(st.residual)",
              subsystem = :pressure, domain = current_domain(), solver = :cg,
              iterations = st.iterations, residual = st.residual,
              relative_residual = st.residual / st.bnorm, tol = solver.tol)
    st.status === :breakdown &&
        @warn("CGPoissonSolver: CG breakdown (vanishing curvature) after $(st.iterations) iterations",
              subsystem = :pressure, domain = current_domain(), solver = :cg,
              reason = :breakdown, iterations = st.iterations, residual = st.residual,
              curvature = st.curvature)
    copy_unit_halo_out!(bk, p, solver.x)
    return p
end

end # module
