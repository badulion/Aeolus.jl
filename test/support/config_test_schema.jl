# A test schema covering every option type of Aeolus.Config (CFG-a, #356),
# and a seeded generator of random raw configs drawn from a schema.
# Included by test/config_schema.jl and test/config_load.jl.

using Aeolus.Config
using Dates: Date, Time
using Random

if !isdefined(@__MODULE__, :CFGT_DIR)

const CFGT_DIR = mktempdir()        # the cfgtest driver's config directory

function cfgt_sections()
    grid = section("grid"; doc = "the mesh") do s
        option!(s, "size", NTuple{3,Int}; default = (16, 16, 16), range = (1, 4096),
                doc = "cells per direction")
        option!(s, "extent", NTuple{3,Float64}; default = (100.0, 100.0, 50.0),
                range = (0, Inf), open_lower = true, units = "m", doc = "domain size")
        option!(s, "stretch", Float64; default = 1.0, range = (1.0, 2.0), doc = "stretching factor")
        option!(s, "levels", Vector{Float64}; default = Float64[], doc = "explicit levels")
    end
    clock = section("clock"; doc = "time stepping") do s
        option!(s, "nsteps", Int; default = 10, range = (0, typemax(Int)))
        option!(s, "dt", Float64; default = 0.5, units = "s")          # unconstrained
        option!(s, "start_date", Date; default = Date(2024, 7, 1))
        option!(s, "start_time", Time; default = Time(6, 0, 0))
        option!(s, "scheme", Symbol; default = :rk3, allowed = [:rk3, :ab2, :imex])
        option!(s, "adaptive", Bool; default = true)
        option!(s, "dt_max", Optional{Float64}; range = (0, Inf), open_lower = true)
    end
    physics = section("physics"; doc = "physics switches") do s
        option!(s, "label", String; default = "test")
        option!(s, "terrain", Optional{Path})
        option!(s, "inputs", Vector{Path}; default = String[])
        option!(s, "tags", Vector{Symbol}; default = Symbol[], allowed = [:a, :b, :c])
        option!(s, "profile", Symbol; default = :linear, allowed = [:linear, :custom],
                allow_julia = true)
        option!(s, "couplings", OpenTable(t -> ["$k: expected a table" for (k, v) in t
                                                 if !(v isa AbstractDict)]))
    end
    return [grid, clock, physics]
end

"Cross-field check: the IMEX scheme runs at a fixed step."
cfgt_check(cfg) = cfg.clock.scheme === :imex && cfg.clock.adaptive ?
    [ConfigError("clock.adaptive", "scheme = \"imex\" needs adaptive = false")] : ConfigError[]

register_driver!("cfgtest"; sections = cfgt_sections(), checks = [cfgt_check], config_dir = CFGT_DIR)

# ---------------------------------------------------------------------------
# random raw configs
# ---------------------------------------------------------------------------

const CFGT_CHARS = vcat(collect('a':'z'), collect("AZ09 _-.\"\\\n\t\r\x01\x7fäöü€😀ß/"))

cfgt_rand_string(rng) = String(rand(rng, CFGT_CHARS, rand(rng, 0:12)))

function cfgt_rand_float(rng, o)
    if o.range === nothing
        r = rand(rng)
        r < 0.3 && return rand(rng, [0.1, 1 / 3, 1e-4, nextfloat(1.0), 5e-324, floatmax(),
                                     Inf, -Inf, NaN, -0.0, 0.0, -1e300])
        r < 0.6 && return reinterpret(Float64, rand(rng, UInt64))      # any bit pattern
        return randn(rng) * 10.0^rand(rng, -20:20)
    end
    lo, hi = Float64.(o.range)
    x = isinf(hi) ? lo + randexp(rng) * 10.0^rand(rng, -3:6) : lo + rand(rng) * (hi - lo)
    return (o.open_lower && x == lo) ? nextfloat(lo) : x
end

function cfgt_rand_scalar(rng, T, o)
    T === Float64 && return cfgt_rand_float(rng, o)
    if T === Int
        lo, hi = o.range === nothing ? (-10^9, 10^9) : (Int(o.range[1]), Int(min(o.range[2], 10^9)))
        return rand(rng, lo:hi)
    end
    T === Bool && return rand(rng, Bool)
    T === String && return cfgt_rand_string(rng)
    T === Symbol && return String(rand(rng, o.allowed))
    T === Path && return rand(rng, Bool) ? "data/f$(rand(rng, 1:999)).nc" : "/abs/in$(rand(rng, 1:9)).dat"
    T === Date && return Date(rand(rng, 1900:2100), rand(rng, 1:12), rand(rng, 1:28))
    T === Time && return Time(rand(rng, 0:23), rand(rng, 0:59), rand(rng, 0:59), rand(rng, 0:999))
    error("no generator for $T")
end

function cfgt_rand_table(rng, depth = 0)
    d = Dict{String,Any}()
    for i in 1:rand(rng, 0:3)
        k = rand(rng, ["x", "y", "wall_model", "a-b", "k$i", "with space"])
        d[k] = depth < 1 ? cfgt_rand_table(rng, depth + 1) :
               rand(rng, [1, 2.5, "s", true, [1, 2], Date(2020, 1, 1)])
    end
    return d
end

function cfgt_rand_value(rng, T, o)
    T isa OpenTable && return Dict{String,Any}(k => cfgt_rand_table(rng, 1) for k in ("c1", "c2")[1:rand(rng, 0:2)])
    if T <: Optional
        return rand(rng) < 0.3 ? "none" : cfgt_rand_value(rng, T.parameters[1], o)
    elseif T <: Tuple
        return Any[cfgt_rand_scalar(rng, fieldtype(T, 1), o) for _ in 1:fieldcount(T)]
    elseif T <: Vector
        return Any[cfgt_rand_value(rng, eltype(T), o) for _ in 1:rand(rng, 0:4)]
    end
    return cfgt_rand_scalar(rng, T, o)
end

"A random raw config of `schema` (some options left to their defaults)."
function cfgt_random_raw(rng, schema)
    raw = Dict{String,Any}()
    for s in schema.sections
        t = Dict{String,Any}()
        for o in s.options
            o.key == "driver" && (t[o.key] = schema.name; continue)
            rand(rng) < 0.25 && continue
            t[o.key] = cfgt_rand_value(rng, o.type, o)
        end
        if s.name == "clock" && get(t, "scheme", "") == "imex"
            t["adaptive"] = false
        end
        raw[s.name] = t
    end
    return raw
end

end # if !isdefined
