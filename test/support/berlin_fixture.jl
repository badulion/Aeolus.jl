# Normalisation and state digests shared by the CFG-c fixture capture
# (test/data/config_berlin/capture_fixture.jl) and test/config_berlin.jl.

using Aeolus.IO: toml_value, fnv1a64
using Aeolus.Fields: interior_view

const _PKG = normpath(joinpath(@__DIR__, "..", ".."))

"`toml_value(x)` with the package directory written `<pkg>` and functions as their values at 0, 3600, 43200 s."
function normalise(x)
    T = typeof(x)
    t = Dict{String,Any}()
    for f in fieldnames(T)
        v = getfield(x, f)
        t[string(f)] = v isa Function || (v !== nothing && applicable(v, 0.0) && !(v isa Number)) ?
                       Any[Float64(v(s)) for s in (0.0, 3600.0, 43200.0)] : _norm(toml_value(v))
    end
    return t
end
_norm(v::AbstractString) = replace(v, _PKG => "<pkg>/")
_norm(v::AbstractDict) = Dict{String,Any}(k => _norm(x) for (k, x) in v)
_norm(v::AbstractVector) = Any[_norm(x) for x in v]
_norm(v) = v

"fnv1a64 of the bytes of an array, as 16 hex digits."
digest(a) = string(fnv1a64(collect(reinterpret(UInt8, vec(collect(a))))); base = 16, pad = 16)

"""
    berlin_state(case) -> NamedTuple-like Vector of name => array

The prognostic state of a built Berlin case, interior cells: the child's
u, v, w, p, θ, every tracer, the surface temperature Ts, and, with the
Aeolus nest parent, the parent's u, v, w, p, θ (and qv).
"""
function berlin_state(case)
    BCm = parentmodule(typeof(case))
    s = case.sim
    out = Pair{Symbol,Array{Float64}}[]
    for f in (:u, :v, :w, :p, :θ)
        push!(out, f => Array{Float64}(interior_view(getproperty(s, f))))
    end
    for t in case.tracers.list
        push!(out, t.name => Array{Float64}(interior_view(t.field)))
    end
    push!(out, :Ts => Array{Float64}(case.bus.Ts))
    c = BCm.find_coupling(case, :nesting)
    if c !== nothing && c.parent !== nothing
        ps = c.parent.sim
        for f in (:u, :v, :w, :p, :θ)
            push!(out, Symbol("parent_", f) => Array{Float64}(interior_view(getproperty(ps, f))))
        end
        c.parent.pc.moisture && push!(out, :parent_qv => Array{Float64}(interior_view(ps.qv)))
    end
    return (; out...)
end
