module Grids

export AbstractMesh, RegularGrid,
       interior, xnodes, ynodes, znodes,
       CellCenter, FaceX, FaceY, FaceZ, Location

# -------------------------------------------------------------------------
# Abstract mesh interface
# -------------------------------------------------------------------------
# Aeolus is being built with two spatial-discretization backends in mind:
#
#   1. Finite volume on a staggered Cartesian grid (Phase 0–1, this file).
#   2. Spectral element on an unstructured mesh of high-order elements
#      (future — will introduce `SpectralElementMesh <: AbstractMesh`).
#
# All user-facing code (Fields, BCs, Operators, TimeSteppers) is written
# against `AbstractMesh` and dispatches on concrete subtypes via multiple
# dispatch. Adding SEM should not require touching call sites.
#
# The `Location` types below are FV-specific (Arakawa-C staggering).
# A future SEM backend will introduce its own node-location semantics.

"""
    AbstractMesh{T <: AbstractFloat}

Root of the mesh type hierarchy. `T` is the element type of the coordinate
data (typically `Float64`). Every concrete subtype must implement:

  * `totalsize(mesh)` — tuple of storage-array dimensions (halos included).
  * `interior(mesh)` — tuple of interior index ranges `(iᵢ, iⱼ, iₖ)`.
  * `xnodes(mesh, ::Location)`, `ynodes(mesh, ::Location)`, `znodes(mesh, ::Location)`
    — physical coordinate vectors at the given [`Location`](@ref).

Concrete subtypes: [`RegularGrid`](@ref).  Planned: `SpectralElementMesh`.
"""
abstract type AbstractMesh{T<:AbstractFloat} end

Base.eltype(::AbstractMesh{T}) where {T} = T

# Interface every concrete mesh must implement:
#   totalsize(mesh)          -> tuple of array dimensions (with halos)
#   interior(mesh)           -> tuple of interior index ranges
#   xnodes/ynodes/znodes(mesh, location)  -> physical coord vectors
#
# The mesh does NOT need to be Cartesian; SEM will return element-local
# GLL node coordinates from a different method.

# -------------------------------------------------------------------------
# Location tags for staggered variables (Arakawa C-grid)
# -------------------------------------------------------------------------

"""
    Location

Abstract tag type identifying where a scalar field's samples sit on a
staggered mesh. On the Arakawa-C grid used by [`RegularGrid`](@ref):

  * [`CellCenter`](@ref) — cell interior; used for pressure, θ, tracers.
  * [`FaceX`](@ref) — west face of each cell; used for `u`.
  * [`FaceY`](@ref) — south face of each cell; used for `v`.
  * [`FaceZ`](@ref) — bottom face of each cell; used for `w`.

Finite-volume operator kernels dispatch on `(::Location, ::Location, …)`
so that e.g. `∂x(FaceX → CellCenter)` (divergence) picks a different
stencil from `∂x(CellCenter → FaceX)` (gradient).
"""
abstract type Location end

"Cell-center location. See [`Location`](@ref)."
struct CellCenter <: Location end

"West-face location — the `u`-velocity live here. See [`Location`](@ref)."
struct FaceX      <: Location end

"South-face location — the `v`-velocity lives here. See [`Location`](@ref)."
struct FaceY      <: Location end

"Bottom-face location — the `w`-velocity lives here. See [`Location`](@ref)."
struct FaceZ      <: Location end

# -------------------------------------------------------------------------
# RegularGrid: uniform 3D Cartesian FV mesh
# -------------------------------------------------------------------------

"""
    RegularGrid{T}(; size, extent, halo=(2,2,2), origin=(0,0,0))

Uniform 3D Cartesian mesh (finite-volume backend) with a halo region on
each side.

`size = (Nx, Ny, Nz)`  number of interior cells per dimension.
`extent = (Lx, Ly, Lz)` physical domain size.
`halo = (Hx, Hy, Hz)`  halo width per dimension (>= 1).

Storage arrays have shape `(Nx+2Hx, Ny+2Hy, Nz+2Hz)`.
Interior indices are `Hx+1 : Hx+Nx`, etc.
"""
struct RegularGrid{T<:AbstractFloat} <: AbstractMesh{T}
    Nx::Int; Ny::Int; Nz::Int
    Hx::Int; Hy::Int; Hz::Int
    Lx::T;   Ly::T;   Lz::T
    dx::T;   dy::T;   dz::T
    x0::T;   y0::T;   z0::T
end

function RegularGrid(::Type{T}=Float64;
                     size::NTuple{3,Int},
                     extent::NTuple{3,Real},
                     halo::NTuple{3,Int}=(2,2,2),
                     origin::NTuple{3,Real}=(0,0,0)) where {T<:AbstractFloat}
    Nx, Ny, Nz = size
    Hx, Hy, Hz = halo
    Lx, Ly, Lz = T.(extent)
    x0, y0, z0 = T.(origin)
    dx = Lx / Nx
    dy = Ly / Ny
    dz = Lz / Nz
    return RegularGrid{T}(Nx, Ny, Nz, Hx, Hy, Hz, Lx, Ly, Lz, dx, dy, dz, x0, y0, z0)
end

"""
    totalsize(mesh)

Tuple of the storage-array dimensions for a scalar field on `mesh`,
including the halo cells on every side.
"""
totalsize(g::RegularGrid) = (g.Nx + 2g.Hx, g.Ny + 2g.Hy, g.Nz + 2g.Hz)

"""
    interior(mesh) -> (irange, jrange, krange)

Index ranges that walk the *interior* (halo-excluded) portion of a field's
backing array on `mesh`. Used everywhere kernels loop over data.
"""
function interior(g::RegularGrid)
    return (g.Hx+1 : g.Hx+g.Nx,
            g.Hy+1 : g.Hy+g.Ny,
            g.Hz+1 : g.Hz+g.Nz)
end

# --- physical coordinates -------------------------------------------------

"""
    xnodes(mesh, location)
    ynodes(mesh, location)
    znodes(mesh, location)

Physical coordinates of a field at the given `location`, one entry per
interior sample. Length matches the interior extent along that axis
(`Nx`, `Ny`, or `Nz` for a [`RegularGrid`](@ref)). Face locations return
the coordinate of the *west* / *south* / *bottom* face of each cell; the
opposite face wraps periodically and is not stored separately.
"""
xnodes(g::RegularGrid, ::CellCenter) = @. g.x0 + ((1:g.Nx) - 0.5) * g.dx
ynodes(g::RegularGrid, ::CellCenter) = @. g.y0 + ((1:g.Ny) - 0.5) * g.dy
znodes(g::RegularGrid, ::CellCenter) = @. g.z0 + ((1:g.Nz) - 0.5) * g.dz

# Face nodes are stored in arrays of the same shape as cell-centered fields.
# Convention: interior index i in (Hx+1 : Hx+Nx) for a FaceX field holds the
# value at the *west* face of cell i, at physical x = x0 + (i-1) * dx.
# The east face of cell Nx (= x0 + Nx*dx) is identified with west face of
# cell 1 by periodic wrap and is not stored separately.

xnodes(g::RegularGrid, ::FaceX) = @. g.x0 + (0:g.Nx-1) * g.dx
ynodes(g::RegularGrid, ::FaceY) = @. g.y0 + (0:g.Ny-1) * g.dy
znodes(g::RegularGrid, ::FaceZ) = @. g.z0 + (0:g.Nz-1) * g.dz

xnodes(g::RegularGrid, ::Union{FaceY,FaceZ}) = xnodes(g, CellCenter())
ynodes(g::RegularGrid, ::Union{FaceX,FaceZ}) = ynodes(g, CellCenter())
znodes(g::RegularGrid, ::Union{FaceX,FaceY}) = znodes(g, CellCenter())

function Base.show(io::IO, g::RegularGrid{T}) where {T}
    print(io, "RegularGrid{$T}(size=($(g.Nx),$(g.Ny),$(g.Nz)), ")
    print(io, "extent=($(g.Lx),$(g.Ly),$(g.Lz)), halo=($(g.Hx),$(g.Hy),$(g.Hz)))")
end

end # module
