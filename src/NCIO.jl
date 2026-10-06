"""
    NCIO

NetCDF output policy for Aeolus, on top of
[NCDatasets.jl](https://github.com/Alexander-Barth/NCDatasets.jl) (#310).

NCDatasets is the one reader and writer of NetCDF in the package: it reads
NetCDF-3 (CDF-1, CDF-2, CDF-5) and NetCDF-4/HDF5, and writes NetCDF-4.
This module adds only what is Aeolus policy, so that every writer in the
package and its examples produces the same kind of file:

* [`nc_create`](@ref): a **NetCDF-4** file with the CF `Conventions`
  global attribute;
* [`nc_defvar!`](@ref): lossless **deflate** (level
  [`NC_DEFLATE_LEVEL`](@ref)) with the byte **shuffle** filter, and
  chunks of one horizontal level and one record ([`nc_chunksizes`](@ref)),
  so reading one level or one time of a 3-D field decompresses only that;
* [`nc_append_record!`](@ref): stream one record along the unlimited
  dimension, checked to be complete, and flush it to disk;
* [`cf_time_units`](@ref): CF time units from a reference date.

Reading needs nothing from here: open a file with `NCDataset(path)`
(re-exported) and index it, `ds["theta"][:, :, :, r]`. Dimensions are in
Julia index order, the reverse of the order `ncdump` prints.

# Why NetCDF-4

NetCDF-4 is the format of ERA5, ICON, COSMO and PALM, and of the tools
that read them (xarray, CDO, NCO, ncview, Paraview). Over NetCDF-3 it adds
compression and chunking, and drops the 4 GiB limit per variable record.
The package wrote NetCDF-3 with an in-package writer (`NetCDFIO`, removed
in #310) because NCDatasets did not build on the development machine at
the time; it does now (NCDatasets 0.14.15, NixOS, Julia 1.13.1).

# Crash safety

An HDF5 file left open by a crashed writer can be unreadable.
[`nc_append_record!`](@ref) therefore calls `sync` after every record, so
the file on disk always holds every completed record.

# References

  * Rew, R., Hartnett, E., Caron, J. (2006): *NetCDF-4: software
    implementing an enhanced data model for the geosciences*, 22nd Int.
    Conf. on IIPS, AMS.
  * CF Metadata Conventions v1.8.
"""
module NCIO

using Dates: DateTime, Millisecond, format
using NCDatasets: NCDataset, defVar, unlimited, dimnames, sync

export NCDataset, nc_create, nc_defvar!, nc_append_record!, nc_chunksizes,
       cf_time_units, NC_DEFLATE_LEVEL

"""
    NC_DEFLATE_LEVEL = 4

Default zlib deflate level of [`nc_defvar!`](@ref). Level 4 is the usual
choice for model output (the CDO and xarray examples use 1 to 5): higher
levels cost write time for a percent or two of size.
"""
const NC_DEFLATE_LEVEL = 4

"Target number of values per chunk along the unlimited dimension of a small record variable."
const _MIN_CHUNK = 4096

"""
    nc_create(path; attributes = Dict()) -> NCDataset

Create (or overwrite) `path` as a NetCDF-4 file open for writing, with the
global attribute `Conventions = "CF-1.8"` followed by `attributes`. Define
dimensions with `ds.dim["x"] = Nx` (`Inf` for the unlimited record
dimension), variables with [`nc_defvar!`](@ref), and `close(ds)` at the
end.
"""
function nc_create(path::AbstractString; attributes::AbstractDict = Dict())
    ds = NCDataset(path, "c"; format = :netcdf4)
    ds.attrib["Conventions"] = "CF-1.8"
    for (k, v) in attributes
        ds.attrib[String(k)] = v
    end
    return ds
end

"""
    nc_chunksizes(ds, dims) -> Vector{Int}

The default chunk shape of a variable over `dims` (Julia index order):
the full extent of the first two (horizontal) dimensions, and 1 along
every further dimension (levels) and along the unlimited dimension, so a
chunk is one horizontal level of one record. A variable whose slab is
smaller than 4096 values (a profile, a time series) takes enough records
per chunk to reach 4096, since a chunk per value would cost more in index
than in data. Non-record variables are one chunk.
"""
function nc_chunksizes(ds::NCDataset, dims)
    dims = String.(collect(dims))
    unl = unlimited(ds.dim)
    chunks = Int[]
    for (k, d) in enumerate(dims)
        push!(chunks, d in unl ? 1 : k <= 2 ? Int(ds.dim[d]) : 1)
    end
    for (k, d) in enumerate(dims)
        d in unl || continue
        slab = prod(chunks; init = 1)
        chunks[k] = max(1, cld(_MIN_CHUNK, slab))
    end
    return chunks
end

"""
    nc_defvar!(ds, name, T, dims; attributes = Dict(),
               deflatelevel = NC_DEFLATE_LEVEL, shuffle = true,
               chunksizes = nc_chunksizes(ds, dims))

Define variable `name` of element type `T` over `dims` (in Julia index
order, the order of the array written), compressed with deflate and the
shuffle filter and chunked as [`nc_chunksizes`](@ref). `attributes` are
the CF attributes (`units`, `long_name`, `axis`, `coordinates`, …).
Returns the variable. `deflatelevel = 0` stores it uncompressed.
"""
function nc_defvar!(ds::NCDataset, name::AbstractString, ::Type{T}, dims;
                    attributes::AbstractDict = Dict(),
                    deflatelevel::Integer = NC_DEFLATE_LEVEL, shuffle::Bool = true,
                    chunksizes = nc_chunksizes(ds, dims)) where {T}
    dims = String.(collect(dims))
    attrib = [String(k) => v for (k, v) in attributes]
    if deflatelevel > 0
        return defVar(ds, String(name), T, dims; attrib = attrib,
                      deflatelevel = deflatelevel, shuffle = shuffle,
                      chunksizes = collect(Int, chunksizes))
    end
    return defVar(ds, String(name), T, dims; attrib = attrib)
end

"""
    nc_append_record!(ds, data::AbstractDict)

Append one record along the unlimited dimension. `data` maps variable name
(`String` or `Symbol`) to the array of one record: every variable over the
unlimited dimension must be present, with that dimension last; a variable
over the unlimited dimension alone takes a one-element vector. The whole
record is checked before any of it is written. Values are
written raw (no CF time or scale encoding: pass the numbers the units
attribute describes), then the file is flushed with `sync`.
"""
function nc_append_record!(ds::NCDataset, data::AbstractDict)
    unl = unlimited(ds.dim)
    length(unl) == 1 ||
        error("nc_append_record!: expected one unlimited dimension, found $(length(unl))")
    udim = only(unl)
    n = Int(ds.dim[udim]) + 1
    # check the whole record before writing any of it, so a bad call does
    # not leave a partial record behind
    todo = Tuple{Any,Any,Int}[]
    for name in keys(ds)
        dn = dimnames(ds[name])
        udim in dn || continue
        last(dn) == udim ||
            error("nc_append_record!: $name has the record dimension $udim not last")
        A = haskey(data, name) ? data[name] :
            haskey(data, Symbol(name)) ? data[Symbol(name)] :
            error("nc_append_record!: missing record variable $name")
        v = ds[name].var                       # raw: no CF decoding on write
        if length(dn) == 1
            length(A) == 1 ||
                error("nc_append_record!: $name takes one value per record, got $(length(A))")
        else
            size(A) == size(v)[1:end-1] ||
                error("nc_append_record!: $name has size $(size(A)), expected $(size(v)[1:end-1])")
        end
        push!(todo, (v, A, length(dn)))
    end
    for (v, A, nd) in todo
        if nd == 1
            v[n:n] = collect(A)
        else
            v[ntuple(_ -> Colon(), nd - 1)..., n] = A
        end
    end
    sync(ds)
    return nothing
end

"""
    cf_time_units(t0::DateTime) -> String

CF time units `"seconds since yyyy-mm-dd HH:MM:SS"` for a time axis that
starts at `t0` (UTC: CF reads a reference time without a zone as UTC).
`NCDataset` decodes a variable with these units to `DateTime`; its raw
seconds are `ds["time"].var[:]`.
"""
cf_time_units(t0::DateTime) = "seconds since " * format(t0, "yyyy-mm-dd HH:MM:SS")

end # module
