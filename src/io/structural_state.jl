# =========================================================================
# Structural checkpoint state: the whole mutable state of an object graph
# (N-berlin-restart, #269). Included into `module IO` after checkpoint.jl.
# =========================================================================
#
# `checkpoint_state` (checkpoint.jl) asks every stateful component for its
# state by name. A composite run such as a nested parent–child pair or the
# Berlin case holds state in dozens of objects (two simulations with their
# open-boundary buffers and SGS memory, the two parent time levels, flux
# registers, an inflow generator with its RNG, soil and building columns,
# ledgers, …), and a component left out silently cold-starts on restart.
# `structural_state` instead walks the object graph and records every value
# a later step could read: every array, every field of every struct, every
# dict entry, with the aliasing between them. `restore_structural_state!`
# walks a freshly built graph of the same construction alongside the record
# and puts the values back in place, so arrays keep their identity (fields,
# hooks and views that share them keep sharing them).
#
# What is not state is skipped (*opaque*): pointers and anything holding one
# (FFTW plans: the fresh build has made its own), tasks, locks, conditions,
# channels, `IO` handles, `WeakKeyDict`s, NetCDF handles, and whatever
# `checkpoint_opaque` declares. An `IdDict` is state (the open boundaries
# bind field arrays by identity, `OpenBoundaryState.bound`): its entries are
# recorded after the rest of the graph, so that its keys are the objects at
# their own places, and restored the same way. A restore leaves the fresh object
# at an opaque position untouched.

using Serialization: Serialization

"Format tag of a [`StructuralState`](@ref)."
const STRUCTURAL_FORMAT = "Aeolus.structural_state"

# --- the record ---------------------------------------------------------------

struct _SLeaf; v::Any; end                       # isbits, Symbol, String, Type, Module
struct _SRef; id::Int; end                       # an object recorded earlier (aliasing)
struct _SOpaque; T::String; end                  # not state (see checkpoint_opaque)
struct _SUndef; end                              # an undefined field / array slot
struct _SPlain; id::Int; a::Array; end           # an array of inline-stored elements
struct _SArray; id::Int; T::Type; dims::Tuple; elems::Vector{Any}; end
struct _SStruct; id::Int; T::Type; fields::Vector{Any}; extra::Any; end   # id 0: immutable
struct _SDict; id::Int; T::Type; keys::Vector{Any}; vals::Vector{Any}; end
struct _SIdDict; id::Int; T::Type; pairs::Vector{Any}; end   # (key record, value record), filled last

"""
    StructuralState

The record [`structural_state`](@ref) makes of an object graph: a tree of
the graph's values (arrays copied), self-contained and serialisable with
`Serialization` (`root` the record of the object, `type` its type name,
`objects` the number of mutable objects and arrays recorded, `bytes` the
array data held).
"""
struct StructuralState
    format::String
    type::String
    root::Any
    objects::Int
    bytes::Int
end

Base.show(io::Base.IO, s::StructuralState) =
    print(io, "StructuralState(", s.type, ": ", s.objects, " objects, ",
          round(s.bytes / 2^20; digits = 1), " MiB of arrays)")

# --- what is not state ------------------------------------------------------------

"""
    checkpoint_opaque(x) -> Bool

Whether [`structural_state`](@ref) skips `x` (records it as opaque, and a
restore leaves the freshly built object at that place as it is): `true`
for pointers and any struct with a pointer field (FFTW plans, compiled
regexes), tasks, locks, conditions, channels, `IO` handles,
`WeakKeyDict` and `WeakRef`, and NetCDF handles. Add a
method for a type of yours that the fresh build recreates and a restart
must not overwrite (an output writer, a plan cache).
"""
checkpoint_opaque(x) = _opaque_type(typeof(x))

const _OPAQUE_MEMO = IdDict{Any,Bool}()
const _OPAQUE_LOCK = ReentrantLock()

function _opaque_type(T::DataType)
    lock(_OPAQUE_LOCK) do
        get!(_OPAQUE_MEMO, T) do
            (T <: Ptr || T <: Core.LLVMPtr || T <: Task || T <: Base.AbstractLock ||
             T <: Base.GenericCondition || T <: Channel || T <: Base.IO || T <: WeakRef ||
             T <: WeakKeyDict || T <: Core.MethodInstance || T <: Core.CodeInstance ||
             T <: Method || parentmodule(T) === NCDatasets ||
             (isstructtype(T) && !(T <: Array) && any(i -> fieldtype(T, i) <: Ptr, 1:fieldcount(T))))
        end
    end
end
_opaque_type(T) = false

# values stored as they are (immutable, no mutable reachable through them)
_leaf(x) = isbits(x) || x isa Symbol || x isa String || x isa Type || x isa Module

# --- save ---------------------------------------------------------------------------

mutable struct _SaveCtx
    ids::IdDict{Any,Int}
    n::Int
    bytes::Int
    deferred::Vector{Any}          # (IdDict, its record), filled after the walk
end

_new_id!(c::_SaveCtx, x) = (c.n += 1; c.ids[x] = c.n; c.n)

"""
    structural_state(x) -> StructuralState
    restore_structural_state!(x, state) -> x

Record the whole mutable state of the object graph of `x`, and put it
back into an `x` **built the same way** (in this or another process):
every array (copied), every field of every struct, the entries of every
`Dict`, and which places share an object. The restore walks the fresh
graph alongside the record: arrays and mutable structs are refilled in
place (so everything that shares them still does), a place whose fresh
value differs in type or identity from the recorded one (a buffer
allocated lazily, a member swapped during the run) takes a rebuilt or the
mapped object, and positions that are [`checkpoint_opaque`](@ref) keep
the fresh object. A recorded value that cannot be put back (a `const`
field that differs, an opaque object that would have to be rebuilt)
throws an `ArgumentError` naming its path.

A mutable type can adapt the walk with
[`structural_save_begin`](@ref) / [`structural_save_end`](@ref) and
[`structural_restore_begin`](@ref) / [`structural_restore_end`](@ref)
(the nested run settles a parent step running ahead on another task this
way). The record is the [`checkpoint_state`](@ref) of components that opt
in (`checkpoint_state(x::MyRun) = structural_state(x)`): the nested run
([`nest_state`](@ref Aeolus.Nesting.nest_state)) and the Berlin case.

The record holds a copy of every array reachable from `x`, static ones
included (meshes, masks, metric terms): a checkpoint of the structure is
larger than one of the prognostic fields alone, by the measured factor
quoted in the IO API page.
"""
function structural_state(x)
    ismutable(x) || throw(ArgumentError("structural_state: the root must be a mutable object, got a $(typeof(x))"))
    c = _SaveCtx(IdDict{Any,Int}(), 0, 0, Any[])
    root = _save(x, c)
    i = 0
    while i < length(c.deferred)                  # entries may reach further IdDicts
        i += 1
        d, e = c.deferred[i]
        for (k, v) in d
            push!(e.pairs, (_save(k, c), _save(v, c)))
        end
    end
    return StructuralState(STRUCTURAL_FORMAT, string(typeof(x)), root, c.n, c.bytes)
end

function _save(x, c::_SaveCtx)
    x isa Ptr && return _SOpaque(string(typeof(x)))
    _leaf(x) && return _SLeaf(x)
    checkpoint_opaque(x) && return _SOpaque(string(typeof(x)))
    if ismutable(x) || x isa Array
        id = get(c.ids, x, 0)
        id > 0 && return _SRef(id)
    end
    x isa Array && return _save_array(x, c)
    x isa Dict && return _save_dict(x, c)
    if x isa IdDict
        e = _SIdDict(_new_id!(c, x), typeof(x), Any[])
        push!(c.deferred, (x, e))
        return e
    end
    return _save_struct(x, c)
end

function _save_array(a::Array, c::_SaveCtx)
    id = _new_id!(c, a)
    T = eltype(a)
    if isbitstype(T) || Base.isbitsunion(T)     # not allocatedinline: inline structs may hold references
        c.bytes += sizeof(a)
        return _SPlain(id, copy(a))
    end
    el = Vector{Any}(undef, length(a))
    for i in eachindex(a)
        el[i] = isassigned(a, i) ? _save(a[i], c) : _SUndef()
    end
    return _SArray(id, typeof(a), size(a), el)
end

# keys in a process-independent order (a Symbol's hash is its address)
_key_order(d::Dict) = sort!(collect(keys(d)); by = k -> string(typeof(k), ':', repr(k)))

function _save_dict(d::Dict, c::_SaveCtx)
    all(_leaf_key, keys(d)) || return _SOpaque("$(typeof(d)) with non-value keys")
    id = _new_id!(c, d)
    ks = _key_order(d)
    return _SDict(id, typeof(d), Any[ks...], Any[_save(d[k], c) for k in ks])
end
_leaf_key(k) = _leaf(k) || (k isa Tuple && all(_leaf_key, k))

function _save_struct(x, c::_SaveCtx)
    T = typeof(x)
    mut = ismutable(x)
    id = mut ? _new_id!(c, x) : 0
    extra = mut ? structural_save_begin(x) : nothing
    try
        n = fieldcount(T)
        fs = Vector{Any}(undef, n)
        for i in 1:n
            fs[i] = !isdefined(x, i) ? _SUndef() :
                    (mut && Base.isfieldatomic(T, i)) ? _SOpaque("atomic field") :
                    _save(getfield(x, i), c)
        end
        return _SStruct(id, T, fs, extra)
    finally
        mut && structural_save_end(x, extra)
    end
end

"""
    structural_save_begin(x) -> extra
    structural_save_end(x, extra)
    structural_restore_begin(x, extra)
    structural_restore_end(x, extra)

Hooks of [`structural_state`](@ref) for a mutable type whose live form is
not the form to record. `structural_save_begin` runs before the fields of
`x` are recorded and returns `extra` (any serialisable value, stored
with the record); it may change `x` into its recordable form, which
`structural_save_end` undoes after the fields are recorded. On restore,
`structural_restore_begin(x, extra)` runs on the fresh object before its
fields are refilled and `structural_restore_end(x, extra)` after. All four
are no-ops by default. Example: a nested run whose parent step is running
ahead on another task waits for it, records it as `extra` and detaches it
(`Aeolus.Nesting`, #269).
"""
structural_save_begin(x) = nothing
@doc (@doc structural_save_begin) structural_save_end(x, extra) = nothing
@doc (@doc structural_save_begin) structural_restore_begin(x, extra) = nothing
@doc (@doc structural_save_begin) structural_restore_end(x, extra) = nothing

# --- restore ------------------------------------------------------------------------

mutable struct _RestoreCtx
    objs::Dict{Int,Any}            # record id => the live object now at its place
    claimed::IdDict{Any,Int}       # live object => the record id it took
    path::Vector{Any}
    deferred::Vector{Any}          # (live IdDict, its record), refilled after the walk
end

_pathstr(c::_RestoreCtx) = isempty(c.path) ? "(root)" : join(string.(c.path), "")

_fail(c::_RestoreCtx, msg) =
    throw(ArgumentError("restore_structural_state!: at $(_pathstr(c)): $msg"))

function restore_structural_state!(x, s::StructuralState)
    s.format == STRUCTURAL_FORMAT ||
        throw(ArgumentError("restore_structural_state!: not a structural state (format $(repr(s.format)))"))
    string(typeof(x)) == s.type ||
        throw(ArgumentError("restore_structural_state!: the state is of a $(s.type), restoring into a $(typeof(x))"))
    c = _RestoreCtx(Dict{Int,Any}(), IdDict{Any,Int}(), Any[], Any[])
    y = _restore(x, s.root, c)
    y === x || _fail(c, "the root object would have to be replaced")
    i = 0
    while i < length(c.deferred)
        i += 1
        d, e = c.deferred[i]
        empty!(d)
        push!(c.path, "<IdDict $(e.id)>")
        for (ke, ve) in e.pairs
            k = _restore(nothing, ke, c)
            d[k] = _restore(nothing, ve, c)
        end
        pop!(c.path)
    end
    return x
end
restore_structural_state!(x, s) =
    throw(ArgumentError("restore_structural_state!: expected a StructuralState, got a $(typeof(s))"))

# Map record `id` to the live object `y`; false if `y` already holds
# another record's state (then that place needs an object of its own).
function _claim!(c::_RestoreCtx, id::Int, y)
    j = get(c.claimed, y, 0)
    j == 0 || j == id || return false
    c.claimed[y] = id
    c.objs[id] = y
    return true
end

# The value the place of `x` must hold after the restore (`x` itself,
# refilled in place, or another object the caller puts there).
_restore(x, e::_SLeaf, c::_RestoreCtx) = e.v
_restore(x, e::_SOpaque, c::_RestoreCtx) = x
_restore(x, e::_SUndef, c::_RestoreCtx) = x
function _restore(x, e::_SRef, c::_RestoreCtx)
    haskey(c.objs, e.id) || _fail(c, "reference to object $(e.id) before it was restored")
    return c.objs[e.id]
end

function _restore(x, e::_SPlain, c::_RestoreCtx)
    a = e.a
    if x isa Array && typeof(x) === typeof(a) && _claim!(c, e.id, x)
        if size(x) == size(a)
            copyto!(x, a)
            return x
        elseif x isa Vector
            resize!(x, length(a)); copyto!(x, a)
            return x
        end
        delete!(c.claimed, x)
    end
    y = copy(a)
    c.objs[e.id] = y; c.claimed[y] = e.id
    return y
end

function _restore(x, e::_SArray, c::_RestoreCtx)
    if !(typeof(x) === e.T && size(x) == e.dims && _claim!(c, e.id, x))
        return _build(e, c)
    end
    for i in eachindex(e.elems)
        push!(c.path, "[$i]")
        old = isassigned(x, i) ? x[i] : nothing
        v = _restore(old, e.elems[i], c)
        (e.elems[i] isa _SUndef || v === old) || (x[i] = v)
        pop!(c.path)
    end
    return x
end

function _restore(x, e::_SDict, c::_RestoreCtx)
    (typeof(x) === e.T && _claim!(c, e.id, x)) || return _build(e, c)
    for k in collect(keys(x))
        k in e.keys || delete!(x, k)
    end
    for (k, ve) in zip(e.keys, e.vals)
        push!(c.path, "[$(repr(k))]")
        if haskey(x, k)
            old = x[k]
            v = _restore(old, ve, c)
            v === old || (x[k] = v)
        else
            x[k] = _build(ve, c)
        end
        pop!(c.path)
    end
    return x
end

function _restore(x, e::_SIdDict, c::_RestoreCtx)
    (typeof(x) === e.T && _claim!(c, e.id, x)) || return _build(e, c)
    push!(c.deferred, (x, e))
    return x
end

function _restore(x, e::_SStruct, c::_RestoreCtx)
    T = e.T
    typeof(x) === T || return _build(e, c)
    if e.id == 0                                     # immutable: rebuild if a field changed
        vals = Vector{Any}(undef, length(e.fields))
        same = true
        for i in eachindex(e.fields)
            push!(c.path, ".", fieldname(T, i))
            old = isdefined(x, i) ? getfield(x, i) : nothing
            vals[i] = _restore(old, e.fields[i], c)
            same &= vals[i] === old || e.fields[i] isa _SUndef
            pop!(c.path); pop!(c.path)
        end
        same && return x
        any(f -> f isa _SUndef, e.fields) && _fail(c, "an immutable $(T) with undefined fields changed")
        return _new_immutable(T, vals)
    end
    _claim!(c, e.id, x) || return _build(e, c)
    structural_restore_begin(x, e.extra)
    for i in eachindex(e.fields)
        fe = e.fields[i]
        (fe isa _SOpaque || fe isa _SUndef) && continue
        push!(c.path, ".", fieldname(T, i))
        old = isdefined(x, i) ? getfield(x, i) : nothing
        v = _restore(old, fe, c)
        if v !== old
            isconst(T, i) && _fail(c, "a const field differs from the checkpoint " *
                                      "(fresh $(_short(old)), checkpoint $(_short(v)))")
            setfield!(x, i, v)
        end
        pop!(c.path); pop!(c.path)
    end
    structural_restore_end(x, e.extra)
    return x
end

_short(v) = (s = sprint(show, v; context = :limit => true); length(s) > 60 ? s[1:60] * "…" : s)

# --- rebuilding a recorded object from scratch ----------------------------------------

_build(e::_SLeaf, c::_RestoreCtx) = e.v
_build(e::_SRef, c::_RestoreCtx) = _restore(nothing, e, c)
_build(e::_SOpaque, c::_RestoreCtx) =
    _fail(c, "a $(e.T) is not checkpointed (opaque) and the fresh object at this place " *
             "is missing or of another type, so it cannot be rebuilt")
_build(e::_SUndef, c::_RestoreCtx) = _fail(c, "an undefined value cannot be rebuilt")

function _build(e::_SPlain, c::_RestoreCtx)
    y = copy(e.a)
    c.objs[e.id] = y; c.claimed[y] = e.id
    return y
end

function _build(e::_SArray, c::_RestoreCtx)
    y = e.T(undef, e.dims)
    c.objs[e.id] = y; c.claimed[y] = e.id
    for i in eachindex(e.elems)
        e.elems[i] isa _SUndef && continue
        push!(c.path, "[$i]")
        y[i] = _build(e.elems[i], c)
        pop!(c.path)
    end
    return y
end

function _build(e::_SDict, c::_RestoreCtx)
    y = e.T()
    c.objs[e.id] = y; c.claimed[y] = e.id
    for (k, ve) in zip(e.keys, e.vals)
        push!(c.path, "[$(repr(k))]")
        y[k] = _build(ve, c)
        pop!(c.path)
    end
    return y
end

function _build(e::_SIdDict, c::_RestoreCtx)
    y = e.T()
    c.objs[e.id] = y; c.claimed[y] = e.id
    push!(c.deferred, (y, e))      # its entries after the walk
    return y
end

function _build(e::_SStruct, c::_RestoreCtx)
    T = e.T
    if e.id == 0
        any(f -> f isa _SUndef, e.fields) && _fail(c, "an immutable $(T) with undefined fields cannot be rebuilt")
        vals = Any[]
        for i in eachindex(e.fields)
            push!(c.path, ".", fieldname(T, i))
            push!(vals, _build(e.fields[i], c))
            pop!(c.path); pop!(c.path)
        end
        return _new_immutable(T, vals)
    end
    y = ccall(:jl_new_struct_uninit, Any, (Any,), T)
    c.objs[e.id] = y; c.claimed[y] = e.id
    structural_restore_begin(y, e.extra)
    for i in eachindex(e.fields)
        e.fields[i] isa _SUndef && continue
        push!(c.path, ".", fieldname(T, i))
        v = _build(e.fields[i], c)
        # the constructor's way of setting a field, const fields included
        ccall(:jl_set_nth_field, Cvoid, (Any, Csize_t, Any), y, i - 1, v)
        pop!(c.path); pop!(c.path)
    end
    structural_restore_end(y, e.extra)
    return y
end

_new_immutable(T::DataType, vals::Vector{Any}) =
    ccall(:jl_new_structv, Any, (Any, Ptr{Any}, UInt32), T, vals, length(vals))

# --- as a checkpoint component ------------------------------------------------------

# a StructuralState is already a self-contained record
checkpoint_state(s::StructuralState) = s
