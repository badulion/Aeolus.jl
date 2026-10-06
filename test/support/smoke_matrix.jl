# The Berlin smoke-test run matrix and its moist seeding, shared by
# test/berlin_smoke.jl and the thread-invariance harness
# (test/thread_invariance.jl, KA-a #294), which hashes the same runs at
# 1, 4 and 24 threads. The includer defines `BC = Main.BerlinCase` and
# loads Aeolus, `interior_view` and `Random.Xoshiro`.

"""
    smoke_matrix(registry = BC.REGISTRY) -> Vector{Pair{String,CaseConfig}}

The smoke-test configurations derived from `registry` (see the file
header of test/berlin_smoke.jl). Every run starts from `CaseConfig(:smoke)` with every additive
coupling on; a run that selects a coupling declared incompatible with an
additive one switches that additive coupling off, and one incompatible
with the default provider of another slot switches that slot to its
first compatible provider (registry order).
"""
function smoke_matrix(registry = BC.REGISTRY)
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
    function mk(label; physics = Dict{Symbol,Any}(), options = Dict{Symbol,Any}(), kw...)
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
        return label => BC.CaseConfig(:smoke; verbose = false, output_path = nothing,
                                      physics = phys, options = opts, kw...)
    end
    runs = [mk("summer, everything on"),
            mk("winter, everything on"; pairs(BC.SCENARIOS[:winter])..., scenario = :winter),
            # the observed initial state of the flagship (#247): the smoke
            # scenario's idealised profile replaced by the summer default
            # sounding (Lindenberg, 29 June 2019, interpolated to noon)
            mk("summer sounding, everything on"; sounding = BC.CaseConfig(:summer).sounding)]
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
