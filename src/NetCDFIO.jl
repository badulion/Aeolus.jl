"""
    NetCDFIO

A dependency-free reader/writer for **NetCDF-3 (CDF-2, 64-bit offset)** files.

# Why this exists

NetCDF is the lingua franca of atmospheric model output: `ncview`,
`xarray`, `Paraview`, `Panoply`, `CDO` and `NCO` all read it. But
`NCDatasets`, `NetCDF.jl` and `HDF5.jl` all fail to precompile on this
host — they need a C library stack that does not build here (the same
`libmpifort.so` failure that drove the rest of this package toward
pure-Julia I/O and plotting).

The choice was therefore between a Julia-only serialization format that
no external tool can read, and writing the container format directly.
NetCDF-3 classic is a small, completely specified, stable binary format
— roughly 250 lines to write — so writing it directly is the better
trade. The files this produces are ordinary `.nc` files; nothing
downstream needs to know they were not written by the C library.

HDF5 (and hence NetCDF-4) was not attempted: it is a far more complex
format with B-trees, heaps and chunking, and hand-writing it would be a
genuinely bad idea.

# Format notes

CDF-2 is used rather than CDF-1 purely for the 64-bit offsets, which
lifts the 2 GiB ceiling — a 24-hour LES run with full 3-D fields goes
past that easily.

All integers are big-endian. Names and attribute values are padded to
4-byte boundaries. The layout is: header, then non-record variables in
declaration order, then record variables interleaved one record at a
time. `numrecs` in the header is patched on close, which is what allows
records to be streamed rather than buffered in memory.

# Dimension order

NetCDF is row-major: the **last** declared dimension varies fastest in
the file. Julia is column-major: the **first** index varies fastest in
memory. So a Julia array `A[i,j,k]` written raw is read correctly by any
NetCDF tool if the variable is declared with dimensions `(k, j, i)` —
i.e. **reversed**. This writer declares them reversed for you; pass
dimension names in Julia index order and the file will be correct.

# References

  * Unidata, *NetCDF Classic Format Specification*.
  * CF Metadata Conventions v1.8 (the attribute conventions used here).
"""
module NetCDFIO

export NCWriter, nc_create, nc_defdim!, nc_defvar!, nc_endef!,
       nc_write!, nc_append_record!, nc_close!, nc_write_fields,
       nc_open, nc_varnames, nc_dimnames, nc_read, nc_read_record,
       nc_attributes, nc_nrecords, NCFile

# --- type codes ---------------------------------------------------------
const NC_BYTE   = Int32(1)
const NC_CHAR   = Int32(2)
const NC_SHORT  = Int32(3)
const NC_INT    = Int32(4)
const NC_FLOAT  = Int32(5)
const NC_DOUBLE = Int32(6)

const NC_DIMENSION = Int32(0x0A)
const NC_VARIABLE  = Int32(0x0B)
const NC_ATTRIBUTE = Int32(0x0C)

nctype(::Type{Float32}) = NC_FLOAT
nctype(::Type{Float64}) = NC_DOUBLE
nctype(::Type{Int32})   = NC_INT
nctype(::Type{Int16})   = NC_SHORT
nctype(::Type{Int8})    = NC_BYTE
ncsize(t::Int32) = t == NC_BYTE || t == NC_CHAR ? 1 :
                   t == NC_SHORT ? 2 : 4 + 4 * (t == NC_DOUBLE)

# --- primitive writers --------------------------------------------------
_w32(io, x::Integer) = write(io, hton(Int32(x)))
_w64(io, x::Integer) = write(io, hton(Int64(x)))

"Pad to the next multiple of 4 with zero bytes."
function _pad(io, n::Integer)
    r = mod(-n, 4)
    r > 0 && write(io, zeros(UInt8, r))
    return nothing
end

function _wname(io, s::AbstractString)
    b = Vector{UInt8}(s)
    _w32(io, length(b)); write(io, b); _pad(io, length(b))
end
_namelen(s::AbstractString) = (n = ncodeunits(s); 4 + n + mod(-n, 4))

# --- attribute values ---------------------------------------------------
struct NCAttr
    name::String
    typ::Int32
    data::Vector{UInt8}
    nelem::Int
end

function ncattr(name::AbstractString, v::AbstractString)
    b = Vector{UInt8}(v)
    NCAttr(String(name), NC_CHAR, b, length(b))
end
function ncattr(name::AbstractString, v::Real)
    io = IOBuffer(); write(io, hton(Float64(v)))
    NCAttr(String(name), NC_DOUBLE, take!(io), 1)
end
function ncattr(name::AbstractString, v::AbstractVector{<:Real})
    io = IOBuffer(); for x in v; write(io, hton(Float64(x))); end
    NCAttr(String(name), NC_DOUBLE, take!(io), length(v))
end

_attrlen(a::NCAttr) = _namelen(a.name) + 4 + 4 + length(a.data) + mod(-length(a.data), 4)

function _wattr(io, a::NCAttr)
    _wname(io, a.name); _w32(io, a.typ); _w32(io, a.nelem)
    write(io, a.data); _pad(io, length(a.data))
end

function _wattrlist(io, as::Vector{NCAttr})
    if isempty(as)
        _w32(io, 0); _w32(io, 0)                 # ABSENT
    else
        _w32(io, NC_ATTRIBUTE); _w32(io, length(as))
        for a in as; _wattr(io, a); end
    end
end
_attrlistlen(as::Vector{NCAttr}) = isempty(as) ? 8 : 8 + sum(_attrlen, as)

# --- dimensions and variables -------------------------------------------
mutable struct NCDim
    name::String
    len::Int          # 0 ⇒ the unlimited (record) dimension
end

mutable struct NCVar
    name::String
    typ::Int32
    dimids::Vector{Int}      # 0-based, in FILE order (already reversed)
    attrs::Vector{NCAttr}
    isrecord::Bool
    vsize::Int               # bytes per record slab (or whole var)
    offset::Int
end

"""
    NCWriter

An open NetCDF-3 file being written. Create with [`nc_create`](@ref),
declare structure with [`nc_defdim!`](@ref) / [`nc_defvar!`](@ref), close
the definition with [`nc_endef!`](@ref), then write.
"""
mutable struct NCWriter
    io::IOStream
    path::String
    dims::Vector{NCDim}
    vars::Vector{NCVar}
    gattrs::Vector{NCAttr}
    defmode::Bool
    numrecs::Int
    recstart::Int
    recsize::Int
end

"""
    nc_create(path; attributes = Dict()) -> NCWriter

Open `path` for writing and begin definition mode. Global attributes are
written as CF-style metadata.
"""
function nc_create(path::AbstractString; attributes::AbstractDict = Dict())
    io = open(path, "w")
    g = NCAttr[]
    push!(g, ncattr("Conventions", "CF-1.8"))
    for (k, v) in attributes
        push!(g, ncattr(String(k), v))
    end
    return NCWriter(io, String(path), NCDim[], NCVar[], g, true, 0, 0, 0)
end

"""
    nc_defdim!(nc, name, len) -> nothing

Declare a dimension. `len = 0` (or `nothing`) declares the *unlimited*
record dimension; at most one is allowed, as the format requires.
"""
function nc_defdim!(nc::NCWriter, name::AbstractString, len)
    nc.defmode || error("nc_defdim!: file is no longer in definition mode")
    n = len === nothing ? 0 : Int(len)
    n == 0 && any(d -> d.len == 0, nc.dims) &&
        error("nc_defdim!: NetCDF-3 permits only one unlimited dimension")
    push!(nc.dims, NCDim(String(name), n))
    return nothing
end

_dimid(nc::NCWriter, name::AbstractString) =
    (i = findfirst(d -> d.name == name, nc.dims);
     i === nothing ? error("nc_defvar!: unknown dimension $name") : i - 1)

"""
    nc_defvar!(nc, name, T, dimnames; attributes = Dict())

Declare a variable of element type `T` over `dimnames`, **given in Julia
index order** — they are reversed internally so the on-disk layout
matches Julia's column-major memory without a transpose.
"""
function nc_defvar!(nc::NCWriter, name::AbstractString, ::Type{T},
                    dimnames; attributes::AbstractDict = Dict()) where {T}
    nc.defmode || error("nc_defvar!: file is no longer in definition mode")
    # reverse: NetCDF's last dimension varies fastest, Julia's first does
    ids = [_dimid(nc, string(d)) for d in reverse(collect(dimnames))]
    isrec = !isempty(ids) && nc.dims[ids[1] + 1].len == 0
    n = 1
    for (k, id) in enumerate(ids)
        L = nc.dims[id + 1].len
        (k == 1 && isrec) && continue
        n *= L
    end
    esz = ncsize(nctype(T))
    vsize = n * esz
    vsize += mod(-vsize, 4)
    as = NCAttr[]
    for (k, v) in attributes
        push!(as, ncattr(String(k), v))
    end
    push!(nc.vars, NCVar(String(name), nctype(T), ids, as, isrec, vsize, 0))
    return nothing
end

_varlen(v::NCVar) = _namelen(v.name) + 4 + 4 * length(v.dimids) +
                    _attrlistlen(v.attrs) + 4 + 4 + 8   # type, vsize, begin(64-bit)

function _headerlen(nc::NCWriter)
    n = 4 + 4                                        # magic + numrecs
    n += isempty(nc.dims) ? 8 : 8 + sum(d -> _namelen(d.name) + 4, nc.dims)
    n += _attrlistlen(nc.gattrs)
    n += isempty(nc.vars) ? 8 : 8 + sum(_varlen, nc.vars)
    return n
end

"""
    nc_endef!(nc)

Finish definition mode: compute variable offsets, write the header.
After this, data may be written.
"""
function nc_endef!(nc::NCWriter)
    nc.defmode || return nc
    off = _headerlen(nc)
    for v in nc.vars                      # non-record variables first
        v.isrecord && continue
        v.offset = off; off += v.vsize
    end
    nc.recstart = off
    for v in nc.vars                      # then the record variables
        v.isrecord || continue
        v.offset = off; off += v.vsize
    end
    nc.recsize = off - nc.recstart
    # With exactly one record variable the format specifies NO padding
    # between successive records; with several, each slab is already
    # padded to 4 by vsize.
    nrecvars = count(v -> v.isrecord, nc.vars)
    if nrecvars == 1
        v = nc.vars[findfirst(v -> v.isrecord, nc.vars)]
        nc.recsize = v.vsize
    end

    io = nc.io
    seekstart(io)
    write(io, UInt8['C', 'D', 'F']); write(io, UInt8(2))   # CDF-2
    _w32(io, nc.numrecs)
    if isempty(nc.dims)
        _w32(io, 0); _w32(io, 0)
    else
        _w32(io, NC_DIMENSION); _w32(io, length(nc.dims))
        for d in nc.dims; _wname(io, d.name); _w32(io, d.len); end
    end
    _wattrlist(io, nc.gattrs)
    if isempty(nc.vars)
        _w32(io, 0); _w32(io, 0)
    else
        _w32(io, NC_VARIABLE); _w32(io, length(nc.vars))
        for v in nc.vars
            _wname(io, v.name)
            _w32(io, length(v.dimids))
            for id in v.dimids; _w32(io, id); end
            _wattrlist(io, v.attrs)
            _w32(io, v.typ); _w32(io, v.vsize); _w64(io, v.offset)
        end
    end
    @assert position(io) == _headerlen(nc) "header length mismatch: $(position(io)) vs $(_headerlen(nc))"
    nc.defmode = false
    return nc
end

_findvar(nc::NCWriter, name) =
    (i = findfirst(v -> v.name == name, nc.vars);
     i === nothing ? error("no variable $name") : nc.vars[i])

_elt(t::Int32) = t == NC_FLOAT ? Float32 : t == NC_DOUBLE ? Float64 :
                 t == NC_INT ? Int32 : t == NC_SHORT ? Int16 : Int8

function _wdata(io, A, T::Type)
    for x in A
        write(io, hton(T(x)))
    end
    n = length(A) * sizeof(T)
    _pad(io, n)
end

"""
    nc_write!(nc, name, A)

Write a **non-record** variable's data in full.
"""
function nc_write!(nc::NCWriter, name::AbstractString, A::AbstractArray)
    nc.defmode && nc_endef!(nc)
    v = _findvar(nc, name)
    v.isrecord && error("nc_write!: $name is a record variable; use nc_append_record!")
    seek(nc.io, v.offset)
    _wdata(nc.io, A, _elt(v.typ))
    return nothing
end

"""
    nc_append_record!(nc, data::AbstractDict)

Append one record (one time level). `data` maps variable name to array —
every record variable must be present. Records are streamed straight to
disk, so a long run never holds its output in memory.
"""
function nc_append_record!(nc::NCWriter, data::AbstractDict)
    nc.defmode && nc_endef!(nc)
    rec = nc.numrecs
    for v in nc.vars
        v.isrecord || continue
        haskey(data, v.name) || haskey(data, Symbol(v.name)) ||
            error("nc_append_record!: missing record variable $(v.name)")
        A = haskey(data, v.name) ? data[v.name] : data[Symbol(v.name)]
        seek(nc.io, v.offset + rec * nc.recsize)
        _wdata(nc.io, A, _elt(v.typ))
    end
    nc.numrecs += 1
    return nothing
end

"""
    nc_close!(nc)

Patch `numrecs` into the header and close the file.
"""
function nc_close!(nc::NCWriter)
    nc.defmode && nc_endef!(nc)
    seek(nc.io, 4)
    _w32(nc.io, nc.numrecs)
    close(nc.io)
    return nc.path
end

"""
    nc_write_fields(path, coords, fields; attributes = Dict())

Convenience: write a single-time-level file from a `Dict` of named
arrays. `coords` maps dimension name to its coordinate vector; each
field's dimensions are inferred from its size by matching against
`coords`. Written as `Float32`, which is the usual precision for model
output and halves the file size.
"""
function nc_write_fields(path::AbstractString, coords::AbstractDict,
                         fields::AbstractDict; attributes::AbstractDict = Dict())
    nc = nc_create(path; attributes = attributes)
    cnames = collect(keys(coords))
    for k in cnames
        nc_defdim!(nc, string(k), length(coords[k]))
    end
    for k in cnames
        nc_defvar!(nc, string(k), Float32, (string(k),);
                   attributes = Dict("units" => "m", "axis" => uppercase(string(k))))
    end
    for (name, A) in fields
        dn = String[]
        for s in size(A)
            i = findfirst(k -> length(coords[k]) == s, cnames)
            i === nothing && error("nc_write_fields: no coordinate of length $s for $name")
            push!(dn, string(cnames[i]))
        end
        nc_defvar!(nc, string(name), Float32, dn)
    end
    nc_endef!(nc)
    for k in cnames
        nc_write!(nc, string(k), collect(Float32, coords[k]))
    end
    for (name, A) in fields
        nc_write!(nc, string(name), A)
    end
    return nc_close!(nc)
end


# =========================================================================
# Reading
# =========================================================================
#
# A writer whose output cannot be read back is not a working format, so
# the reader lives here too. It parses the same specification
# independently of the writing path — it does not share the offset
# arithmetic — so a round-trip test genuinely exercises both.

"""
    NCFile

A parsed NetCDF-3 file: dimensions, global attributes and variable
metadata, with the raw bytes retained for lazy reads.
"""
struct NCFile
    bytes::Vector{UInt8}
    dims::Vector{Pair{String,Int}}
    gattrs::Dict{String,Any}
    vars::Dict{String,NamedTuple}
    numrecs::Int
    recsize::Int
end

struct _Rd
    b::Vector{UInt8}
    p::Base.RefValue{Int}
end
_r32(r::_Rd) = (v = ntoh(reinterpret(Int32, r.b[r.p[]:r.p[]+3])[1]); r.p[] += 4; Int(v))
_r64(r::_Rd) = (v = ntoh(reinterpret(Int64, r.b[r.p[]:r.p[]+7])[1]); r.p[] += 8; Int(v))
function _rname(r::_Rd)
    n = _r32(r)
    s = String(r.b[r.p[]:r.p[]+n-1]); r.p[] += n + mod(-n, 4); return s
end
function _rvals(r::_Rd, t::Int, n::Int)
    T = t == NC_FLOAT ? Float32 : t == NC_DOUBLE ? Float64 :
        t == NC_INT ? Int32 : t == NC_SHORT ? Int16 : UInt8
    if t == NC_CHAR
        s = String(r.b[r.p[]:r.p[]+n-1]); r.p[] += n + mod(-n, 4); return s
    end
    sz = sizeof(T) * n
    raw = reinterpret(T, r.b[r.p[]:r.p[]+sz-1])
    out = [ntoh(x) for x in raw]
    r.p[] += sz + mod(-sz, 4)
    return out
end
function _rattrs(r::_Rd)
    tag = _r32(r); n = _r32(r); out = Dict{String,Any}()
    tag == 0 && return out
    for _ in 1:n
        nm = _rname(r); t = _r32(r); ne = _r32(r)
        out[nm] = _rvals(r, t, ne)
    end
    return out
end

"""
    nc_open(path) -> NCFile

Parse a NetCDF-3 (CDF-1 or CDF-2) file.
"""
function nc_open(path::AbstractString)
    b = read(path)
    (b[1] == UInt8('C') && b[2] == UInt8('D') && b[3] == UInt8('F')) ||
        error("nc_open: $path is not a NetCDF classic file")
    ver = Int(b[4])
    ver in (1, 2) || error("nc_open: unsupported CDF version $ver")
    r = _Rd(b, Ref(5))
    numrecs = _r32(r)
    tag = _r32(r); nd = _r32(r)
    dims = Pair{String,Int}[]
    if tag != 0
        for _ in 1:nd; push!(dims, _rname(r) => _r32(r)); end
    end
    gattrs = _rattrs(r)
    tag = _r32(r); nv = _r32(r)
    vars = Dict{String,NamedTuple}()
    dd = Dict(dims)
    if tag != 0
        for _ in 1:nv
            nm = _rname(r); nds = _r32(r)
            ids = [_r32(r) for _ in 1:nds]
            va = _rattrs(r); t = _r32(r); vsize = _r32(r)
            off = ver == 2 ? _r64(r) : _r32(r)
            dnames = [dims[i+1].first for i in ids]
            isrec = !isempty(dnames) && dd[dnames[1]] == 0
            vars[nm] = (dims = dnames, typ = t, vsize = vsize,
                        offset = off, attrs = va, isrecord = isrec)
        end
    end
    nrec = count(v -> v.isrecord, values(vars))
    recsize = nrec == 1 ?
        first(v.vsize for v in values(vars) if v.isrecord) :
        sum(v.vsize for v in values(vars) if v.isrecord; init = 0)
    return NCFile(b, dims, gattrs, vars, numrecs, recsize)
end

nc_varnames(f::NCFile) = sort(collect(keys(f.vars)))
nc_dimnames(f::NCFile) = [d.first for d in f.dims]
nc_nrecords(f::NCFile) = f.numrecs
nc_attributes(f::NCFile) = f.gattrs
nc_attributes(f::NCFile, v::AbstractString) = f.vars[v].attrs

function _shape(f::NCFile, v)
    dd = Dict(f.dims)
    shp = [dd[d] for d in v.dims]
    v.isrecord && (shp[1] = f.numrecs)
    return shp
end

"""
    nc_read(f, name) -> Array

Read a whole variable. Dimensions are returned in **Julia index order**
(i.e. reversed from the file), so a field written as `("x","y","z","time")`
comes back with that same shape.
"""
function nc_read(f::NCFile, name::AbstractString)
    haskey(f.vars, name) || error("nc_read: no variable $name")
    v = f.vars[name]
    T = v.typ == NC_FLOAT ? Float32 : v.typ == NC_DOUBLE ? Float64 :
        v.typ == NC_INT ? Int32 : Int16
    shp = _shape(f, v)
    if !v.isrecord
        n = prod(shp)
        raw = reinterpret(T, f.bytes[v.offset+1 : v.offset+n*sizeof(T)])
        A = reshape([ntoh(x) for x in raw], reverse(shp)...)
        return A
    end
    nslab = prod(shp[2:end])
    out = Array{T}(undef, reverse(shp)...)
    flat = reshape(out, nslab, f.numrecs)
    for rec in 0:(f.numrecs - 1)
        o = v.offset + rec * f.recsize
        raw = reinterpret(T, f.bytes[o+1 : o+nslab*sizeof(T)])
        @inbounds for i in 1:nslab
            flat[i, rec + 1] = ntoh(raw[i])
        end
    end
    return out
end

"""
    nc_read_record(f, name, rec) -> Array

Read a single time level (1-based) of a record variable, without pulling
the whole series into memory.
"""
function nc_read_record(f::NCFile, name::AbstractString, rec::Integer)
    v = f.vars[name]
    v.isrecord || error("nc_read_record: $name is not a record variable")
    1 <= rec <= f.numrecs || error("nc_read_record: record $rec out of range 1:$(f.numrecs)")
    T = v.typ == NC_FLOAT ? Float32 : v.typ == NC_DOUBLE ? Float64 :
        v.typ == NC_INT ? Int32 : Int16
    shp = _shape(f, v)
    nslab = prod(shp[2:end])
    o = v.offset + (rec - 1) * f.recsize
    raw = reinterpret(T, f.bytes[o+1 : o+nslab*sizeof(T)])
    return reshape([ntoh(x) for x in raw], reverse(shp[2:end])...)
end

end # module
