module Fields

using ..Grids
using ..Grids: AbstractMesh, totalsize, CellCenter, FaceX, FaceY, FaceZ, Location

export Field, location, mesh, interior_view, set!

"""
    Field{L, T, A, M}

A scalar field on `mesh::M <: AbstractMesh` at `Location L`.

The backing array `A` covers the interior region plus halos for FV meshes;
future backends (SEM) may use a different array layout — user code should
go through `mesh(f)`, `location(f)`, `interior_view(f)`, and the operator
API rather than indexing `f.data` directly.
"""
mutable struct Field{L<:Location, T<:AbstractFloat, A<:AbstractArray{T}, M<:AbstractMesh{T}}
    data::A
    mesh::M
end

function Field(loc::Location, mesh::AbstractMesh{T}) where {T}
    data = zeros(T, totalsize(mesh))
    return Field{typeof(loc), T, typeof(data), typeof(mesh)}(data, mesh)
end

Field(mesh::AbstractMesh) = Field(CellCenter(), mesh)

"""
    location(field) -> Location

Return the [`Location`](@ref) instance where `field`'s samples sit
(`CellCenter()`, `FaceX()`, `FaceY()`, or `FaceZ()`). Used by operator
kernels to dispatch on the appropriate discrete stencil.
"""
location(::Field{L}) where {L} = L()

"""
    mesh(field) -> AbstractMesh

The mesh on which `field` is defined. All fields participating in an
operator call must share the same mesh.
"""
mesh(f::Field) = f.mesh

# `grid` retained as an alias for readability in FV-heavy call sites
grid(f::Field) = f.mesh

Base.eltype(::Field{L,T}) where {L,T} = T
Base.size(f::Field) = size(f.data)

Base.@propagate_inbounds Base.getindex(f::Field, i, j, k) = f.data[i, j, k]
Base.@propagate_inbounds Base.setindex!(f::Field, v, i, j, k) = (f.data[i, j, k] = v)

"""
    interior_view(field) -> SubArray

A view (no copy) of the interior (halo-excluded) region of `field`'s
storage. Prefer this over indexing raw `.data` — it works for any mesh
backend that keeps its halo layout out of user code.
"""
function interior_view(f::Field)
    ii, jj, kk = Grids.interior(f.mesh)
    return view(f.data, ii, jj, kk)
end

"""
    set!(f::Field, g::Function)

Populate `f`'s interior with `g(x, y, z)` at that location's physical coords.
Halos are left untouched — call `fill_halos!` afterwards.
"""
function set!(f::Field, g::Function)
    loc = location(f)
    m   = f.mesh
    xs = Grids.xnodes(m, loc); ys = Grids.ynodes(m, loc); zs = Grids.znodes(m, loc)
    ii, jj, kk = Grids.interior(m)
    @inbounds for (kk_, kv) in enumerate(kk), (jj_, jv) in enumerate(jj), (ii_, iv) in enumerate(ii)
        f.data[iv, jv, kv] = g(xs[ii_], ys[jj_], zs[kk_])
    end
    return f
end

function set!(f::Field, value::Real)
    v = eltype(f)(value)
    ii, jj, kk = Grids.interior(f.mesh)
    @inbounds for k in kk, j in jj, i in ii
        f.data[i, j, k] = v
    end
    return f
end

function Base.show(io::IO, f::Field{L,T}) where {L,T}
    print(io, "Field{$L,$T}(mesh=$(f.mesh))")
end

end # module
