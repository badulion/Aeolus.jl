module Fields

using ..Grids
using ..Grids: AbstractMesh, totalsize, CellCenter, FaceX, FaceY, FaceZ, Location
using ..Backends: Backends, AbstractBackend, allocate, default_backend, launch!, range_map!,
                  plane_partials!, interior_sum, interior_dot, plane_mean!, interior_max,
                  interior_all_finite

export Field, location, mesh, interior_view, set!, interior_map!

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

"""
    Field(loc, mesh; backend = default_backend())
    Field(mesh; backend = default_backend())        # CellCenter

A zero field at `loc` on `mesh`, its storage (interior plus halos)
allocated on `backend` ([`Backends.allocate`](@ref Aeolus.Backends.allocate)).
"""
function Field(loc::Location, mesh::AbstractMesh{T};
               backend::AbstractBackend = default_backend()) where {T}
    data = allocate(backend, T, totalsize(mesh))
    return Field{typeof(loc), T, typeof(data), typeof(mesh)}(data, mesh)
end

Field(mesh::AbstractMesh; kw...) = Field(CellCenter(), mesh; kw...)

"The backend that owns `f`'s storage."
Backends.backend(f::Field) = Backends.backend(f.data)

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

# ---- set!: the first KA kernels and the pattern example (KA-a, #294) ----------
#
# A kernel is a plain body `body!(i, j, k, args...)` that updates the one
# point (i, j, k) given in storage indices; `launch!` walks the ranges
# (serially in the old `for k, for j, for i` order, or threaded). The body
# gets everything it reads as arguments — no closures over globals — and
# carries its own @inbounds. Pointwise, so every path gives the same bits
# as the loops it replaced.

@inline function _set_function_body!(i, j, k, data, g::G, xs, ys, zs, oi, oj, ok) where {G}
    @inbounds data[i, j, k] = g(xs[i - oi], ys[j - oj], zs[k - ok])
    return nothing
end

@inline function _set_value_body!(i, j, k, data, v)
    @inbounds data[i, j, k] = v
    return nothing
end

"""
    set!(f::Field, g::Function)

Populate `f`'s interior with `g(x, y, z)` at that location's physical coords.
Halos are left untouched — call `fill_halos!` afterwards. A pointwise
kernel ([`launch!`](@ref Aeolus.Backends.launch!)): threaded when Julia
has threads, bit-identical to a serial loop.
"""
function set!(f::Field, g::Function)
    loc = location(f)
    m   = f.mesh
    xs = Grids.xnodes(m, loc); ys = Grids.ynodes(m, loc); zs = Grids.znodes(m, loc)
    ii, jj, kk = Grids.interior(m)
    checkbounds(f.data, ii, jj, kk)
    (length(xs), length(ys), length(zs)) == (length(ii), length(jj), length(kk)) ||
        throw(DimensionMismatch("set!: node vectors do not match the interior"))
    launch!(Backends.backend(f), _set_function_body!, (ii, jj, kk), f.data, g, xs, ys, zs,
            first(ii) - 1, first(jj) - 1, first(kk) - 1)
    return f
end

"""
    set!(f::Field, value::Real)

Set every interior value of `f` to `value` (halos untouched). A pointwise
kernel, bit-identical serially and threaded.
"""
function set!(f::Field, value::Real)
    v = eltype(f)(value)
    ii, jj, kk = Grids.interior(f.mesh)
    checkbounds(f.data, ii, jj, kk)
    launch!(Backends.backend(f), _set_value_body!, (ii, jj, kk), f.data, v)
    return f
end

# ---- field-level map and reductions over the interior ------------------------

_data(f::Field) = f.data
_data(x) = x

"""
    interior_map!(f, out::Field, args...) -> out

`out[i,j,k] = f(args[1][i,j,k], args[2][i,j,k], …)` over the interior of
`out`'s mesh; an argument may be a `Field` (same mesh) or a number. E.g.
an axpy, `interior_map!((y, x) -> y + a * x, y, y, x)`. Pointwise: the
same bits serially and threaded. Halos are untouched.
"""
function interior_map!(f::F, out::Field, args...) where {F}
    for a in args
        a isa Field && (size(a.data) == size(out.data) ||
                        throw(DimensionMismatch("interior_map!: fields of different sizes")))
    end
    range_map!(Backends.backend(out), f, out.data, Grids.interior(out.mesh), map(_data, args)...)
    return out
end

# The canonical, thread-invariant reductions (Backends) over a field's interior.
Backends.interior_sum(f::Field) = interior_sum(Backends.backend(f), f.data, Grids.interior(f.mesh))
Backends.interior_dot(a::Field, b::Field) =
    interior_dot(Backends.backend(a), a.data, b.data, Grids.interior(a.mesh))
Backends.interior_max(f::Field) = interior_max(Backends.backend(f), f.data, Grids.interior(f.mesh))
Backends.interior_all_finite(f::Field) =
    interior_all_finite(Backends.backend(f), f.data, Grids.interior(f.mesh))
Backends.plane_mean!(profile::AbstractVector, f::Field) =
    plane_mean!(profile, Backends.backend(f), f.data, Grids.interior(f.mesh))

function Base.show(io::IO, f::Field{L,T}) where {L,T}
    print(io, "Field{$L,$T}(mesh=$(f.mesh))")
end

end # module
