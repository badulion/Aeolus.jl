# The smoke run matrix of the urban run type and its moist seeding, shared
# by the urban run-type tests (test/urban_run.jl, EX-a #438; formerly the
# by-hand examples/urban/berlin_checks.jl) and the thread-invariance
# harness (test/thread_invariance.jl, KA-a #294), which hashes the same
# runs at 1, 4 and 24 threads. The includer defines `BC = Aeolus.UrbanCase`
# and loads Aeolus, `interior_view` and `Random.Xoshiro`.
#
# Every run is examples/urban/configs/berlin_smoke.toml with `--set`
# overrides (CFG-c, #358): `smoke_specs` derives what each run changes
# from the registry, `smoke_overrides` writes it as overrides, and
# `smoke_matrix` loads the configs.

using Aeolus: load_run_config
using TOML
include(joinpath(@__DIR__, "berlin_configs.jl"))

"The shipped smoke config every run starts from."
const SMOKE_TOML = joinpath(berlin_configs_dir(), "berlin_smoke.toml")

"""
    file_overrides(path) -> Vector{String}

The values a scenario file sets (its tables flattened to `--set`
overrides; `extends`, `schema_version`, `[run] name` and the file's
`[test]` table, which is not configuration, left out): the
differences of `berlin_winter.toml` applied to the smoke run.
"""
function file_overrides(path)
    d = TOML.parsefile(path)
    out = String[]
    function walk(t, prefix)
        for k in sort!(collect(keys(t)))
            key = isempty(prefix) ? k : prefix * "." * k
            key in ("extends", "schema_version", "run.name", "test") && continue
            v = t[k]
            v isa AbstractDict ? walk(v, key) :
                push!(out, key * "=" * Aeolus.Config._emit_value(Aeolus.Config._toml_out(v)))
        end
    end
    walk(d, "")
    return out
end

"""
    smoke_overrides(spec) -> Vector{String}

The `--set` overrides of `berlin_smoke.toml` for a run of
[`smoke_specs`](@ref): `physics.<slot or addon>`, every option as
`physics.options.<coupling>.<key>` (the nest parent's as
`physics.options.nesting.parent.<key>`), every `CaseConfig` field by its
TOML key (`BC.CASE_FIELDS`), then the spec's raw overrides.
"""
function smoke_overrides(spec)
    out = String[]
    for k in sort!(collect(keys(spec.physics)))
        push!(out, "physics.$k=" * BC.toml_literal(spec.physics[k]))
    end
    # berlin_summer.toml turns [checkpoint] on, and only the Boussinesq
    # core can be checkpointed (#269): a run on another core turns it off
    core = get(spec.physics, :core, :boussinesq)
    Symbol(core) === :boussinesq || push!(out, "checkpoint.every_seconds=" * BC.toml_literal(Inf))
    for c in sort!(collect(keys(spec.options))), (k, v) in pairs(spec.options[c])
        if c === :nesting && k === :parent
            for (pk, pv) in pairs(v)
                push!(out, "physics.options.nesting.parent.$(BC.PARENT_FIELDS[pk])=" * BC.toml_literal(pv))
            end
        else
            push!(out, "physics.options.$c.$k=" * BC.toml_literal(v))
        end
    end
    for (f, v) in pairs(spec.fields)
        keys_ = BC.CASE_FIELDS[f]
        length(keys_) == 1 || error("smoke_overrides: the field $f is made of several options ($(join(keys_, ", ")))")
        push!(out, keys_[1] * "=" * BC.toml_literal(v))
    end
    return vcat(out, spec.extra)
end

"The `CaseConfig` of a run of [`smoke_specs`](@ref): `berlin_smoke.toml` with its overrides."
smoke_config(spec) = BC.CaseConfig(load_run_config(SMOKE_TOML; overrides = smoke_overrides(spec),
                                                   schema = BC.urban_schema()))

"""
    smoke_matrix(registry = BC.REGISTRY) -> Vector{Pair{String,CaseConfig}}

The smoke-test configurations: [`smoke_config`](@ref) of every run of
[`smoke_specs`](@ref).
"""
smoke_matrix(registry = BC.REGISTRY) = [label => smoke_config(spec) for (label, spec) in smoke_specs(registry)]

"""
    smoke_specs(registry = BC.REGISTRY) -> Vector{Pair{String,NamedTuple}}

The smoke runs derived from `registry` (see the smoke section of
docs/src/design/berlin_ledgers.md), each `label => (physics, options, fields, extra)`:
the `[physics]` selection, the per-coupling options, `CaseConfig` field
values and raw overrides that run sets over `berlin_smoke.toml`. Every
run has every additive coupling on; a run that selects a coupling
declared incompatible with an additive one switches that additive
coupling off, and one incompatible with the default provider of another
slot switches that slot to its first compatible provider (registry
order).
"""
function smoke_specs(registry = BC.REGISTRY)
    name = BC.coupling_name
    addons = [C for C in registry if BC.slot(C) === :addon]
    defaults = Dict{Symbol,Symbol}()
    for C in registry
        BC.slot(C) !== :addon && BC.is_default(C) && (defaults[BC.slot(C)] = BC.provider(C))
    end
    base_options() = Dict{Symbol,Any}(
        # the flagship stacks are all outside the smoke window: put one in
        # it, 40 m up (above the tallest roof, 31 m), so the source is used
        :pollutant => (stacks = [(0.0, -312.5, 40.0)],))
    hits(C, D) = any(x -> x isa Pair ? (BC.slot(D) === x.first && BC.provider(D) === x.second) :
                                        name(D) === x, BC.incompatible(C))
    function mk(label; physics = Dict{Symbol,Any}(), options = Dict{Symbol,Any}(), extra = String[], kw...)
        phys = Dict{Symbol,Any}(name(C) => true for C in addons)
        merge!(phys, physics)
        # The selection this physics makes. A coupling it conflicts with
        # that the run did not ask for gives way: an addon is switched
        # off, a slot's default provider is replaced by the slot's first
        # provider (registry order) compatible with the rest (e.g. core =
        # anelastic brings pressure_solver = density_weighted_cg and
        # timestepper = fixed along).
        selection() = [C for C in registry if (BC.slot(C) === :addon ? phys[name(C)] :
                       BC.provider(C) === get(phys, BC.slot(C), defaults[BC.slot(C)]))]
        clash(C, D) = hits(C, D) || hits(D, C)
        for _ in 1:length(registry)
            sel = selection()
            changed = false
            for C in sel, D in sel
                clash(C, D) || continue
                for X in (C, D)
                    s = BC.slot(X)
                    if s === :addon
                        haskey(physics, name(X)) || (phys[name(X)] = false; changed = true)
                    elseif !haskey(physics, s)
                        others = [Y for Y in sel if BC.slot(Y) !== s]
                        fits(P, Ys) = BC.slot(P) === s && !any(Y -> clash(P, Y), Ys)
                        i = findfirst(P -> fits(P, others), registry)
                        # none fits everything: take one that fits what the run
                        # asked for, and let the next pass move the slots that
                        # now clash with it (pressure_solver = density_weighted_cg
                        # brings a sound-proof core, which brings the fixed stepper)
                        asked = [Y for Y in others if BC.slot(Y) === :addon ? haskey(physics, name(Y)) :
                                                                              haskey(physics, BC.slot(Y))]
                        i === nothing && (i = findfirst(P -> fits(P, asked), registry))
                        i === nothing && error("smoke run \"$label\": no provider of slot :$s is compatible with the selection")
                        phys[s] = BC.provider(registry[i]); changed = true
                    end
                end
                changed && break
            end
            changed || break
        end
        sel = selection()
        left = [(name(C), name(D)) for C in sel, D in sel if hits(C, D)]
        isempty(left) || error("smoke run \"$label\": the selection still clashes: $left")
        opts = base_options()
        haskey(opts, :pollutant) && !any(C -> name(C) === :pollutant, registry) && delete!(opts, :pollutant)
        merge!(opts, options)
        return label => (physics = phys, options = opts, fields = (; kw...), extra = String[extra...])
    end
    runs = [mk("summer, everything on"),
            # the winter scenario's differences (berlin_winter.toml) over the smoke run
            mk("winter, everything on"; extra = file_overrides(joinpath(berlin_configs_dir(), "berlin_winter.toml"))),
            # the observed initial state of the flagship (#247): the smoke
            # scenario's idealised profile replaced by the summer default
            # sounding (Lindenberg, 29 June 2019, interpolated to noon)
            mk("summer sounding, everything on"; sounding = BC.CaseConfig(berlin_toml("summer")).sounding)]
    for C in registry
        (BC.slot(C) === :addon || BC.is_default(C)) && continue
        push!(runs, mk("$(BC.slot(C)) = $(BC.provider(C))";
                       physics = Dict{Symbol,Any}(BC.slot(C) => BC.provider(C))))
    end
    for C in registry, v in BC.smoke_variants(C)
        label = string(name(C), ": ", get(v, :label, "variant"))
        kw = Base.structdiff(v, NamedTuple{(:label, :physics, :options)})
        push!(runs, mk(label; physics = Dict{Symbol,Any}(get(v, :physics, Dict{Symbol,Any}())),
                       options = Dict{Symbol,Any}(get(v, :options, Dict{Symbol,Any}())), kw...))
    end
    return runs
end

"""
    seed_fog!(case, rng)

Saturate the lowest three levels (``q_v = q_s(T, \bar p_k)`` with
``T = θ \bar Π_k`` on the case's hydrostatic reference state), put ``q_l = 0.2 \\,(1 ± 0.1)`` g/kg there, and a little
rain, ``q_r ∈ [0, 5\\times10^{-5})``, in levels 2–3.
"""
function seed_fog!(case, rng)
    qv = interior_view(BC.tracer(case, :qv))
    ql = interior_view(BC.tracer(case, :ql))
    qr = interior_view(BC.tracer(case, :qr))
    θ = interior_view(case.sim.θ)
    for k in 1:3
        p, Π = BC.ref_pressure(case, k), BC.ref_exner(case, k)
        for j in axes(qv, 2), i in axes(qv, 1)
            qv[i, j, k] = Aeolus.MoistThermodynamics.saturation_specific_humidity(θ[i, j, k] * Π, p)
            ql[i, j, k] = 2e-4 * (1 + 0.1 * (2 * rand(rng) - 1))
            k >= 2 && (qr[i, j, k] = 5e-5 * rand(rng))
        end
    end
    return case
end
