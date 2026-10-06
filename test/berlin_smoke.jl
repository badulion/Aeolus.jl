# The Berlin smoke test (#127): every coupling of the Berlin case runs in
# a short window of the real OSM scene, in every configuration the
# registry knows about, and must leave a finite state, pass its own
# `smoke_check`s and close the surface-energy and water ledgers.
#
# The run matrix is derived from the registry, so a new coupling is
# exercised without editing this file:
#
#   * "summer, everything on" — every additive coupling on (default-off
#     ones included) and the default provider of every slot;
#   * "winter, everything on" — the same with the :winter initial state;
#   * one run per non-default provider of every exclusive slot;
#   * one run per entry of every coupling's `smoke_variants`.
#
# Every run must close the surface-energy, water and air-heat (#253)
# ledgers.
#
# Each run is CaseConfig(:smoke): a 250 m × 250 m × 100 m window of the
# real Alexanderplatz OSM file (16 × 16 × 16, dx = 15.6 m) at local noon,
# 12 steps of 2 s, radiation every 8 s, soil every 4 s. A fog layer
# (saturated air with ql ≈ 0.2 g/kg in the lowest three levels) and a
# little rain are seeded from a fixed RNG seed so the moist code paths do
# real work.
#
# Run it on its own with `julia --project=. test/berlin_smoke.jl`.

using Test
using Aeolus
using Aeolus.Fields: interior_view
using Printf
using Random: Xoshiro

isdefined(Main, :BerlinCase) ||
    include(joinpath(@__DIR__, "..", "examples", "urban", "BerlinCase.jl"))
const BC = Main.BerlinCase

"""
    smoke_matrix(registry = BC.REGISTRY) -> Vector{Pair{String,CaseConfig}}

The smoke-test configurations derived from `registry` (see the file
header). Every run starts from `CaseConfig(:smoke)` with every additive
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

allfinite(x) = all(isfinite, x)

@testset "Berlin smoke test (#127)" begin
    t_start = time()
    runs = smoke_matrix()
    @info "Berlin smoke matrix: $(length(runs)) runs — " * join(first.(runs), "; ")
    exercised = Set{Symbol}()
    nchecks = 0

    for (irun, (label, cfg)) in enumerate(runs)
        @testset "$label" begin
            t0 = time()
            case = BC.build_case(cfg; io = devnull)
            seed_fog!(case, Xoshiro(127))
            BC.run_case!(case)
            union!(exercised, BC.coupling_name.(case.couplings))
            sim, bus = case.sim, case.bus

            # nsteps·dt of simulated time (in nsteps steps, unless the
            # time stepper chooses its own step, #133)
            @test !case.clock.blowup && isapprox(sim.t, cfg.nsteps * cfg.dt; rtol = 1e-12)
            BC.variable_dt(typeof(BC.timestepper(case))) || @test case.clock.n == cfg.nsteps
            @test all(allfinite ∘ interior_view, (sim.u, sim.v, sim.w, sim.θ))
            @test all(t -> allfinite(interior_view(t.field)), case.tracers.list)
            @test all(allfinite, (bus.Ts, bus.sw, bus.lw, bus.dew))

            # every coupling's own physics checks
            failed = String[]
            for c in case.couplings, (what, ok) in BC.smoke_check(c, case)
                nchecks += 1
                ok || push!(failed, "$(BC.coupling_name(c)): $what")
            end
            @test failed == String[]

            # The surface energy ledger closes per class: every term that
            # changes a surface temperature is reported by some coupling.
            rows = BC.surface_budget(case.ledger)
            @test !isempty(rows)
            for r in rows
                @test abs(r.closure) < 1e-6          # W/m²; measured ≲ 3e-10
            end
            dom = rows[end]
            # The water ledger closes: the change of water in the air is
            # the sum of the booked sources and sinks.
            w = BC.water_budget(case.ledger)
            @test abs(w.residual) < 1e-12            # kg/m²; measured ≲ 3e-16
            # The air-heat ledger closes (#253): the measured change of the
            # heat of the air is the sum of what it attributed to the stage
            # entries and the dynamics, and nothing changed θ outside them.
            hb = BC.heat_budget(case.ledger)
            @test hb.elapsed > 0
            @test abs(hb.closure / hb.elapsed) <= 1e-10     # W/m²
            @test !haskey(hb.terms, (:core, :unattributed))
            @test !haskey(hb.terms, (:output, :between_steps))
            @info @sprintf("  %-26s %5.1f s  domain Rn %7.2f H %6.2f LE %5.2f G %7.2f dS %7.2f W/m², Rn−H−LE−G−dS %7.3f, closure %.1e; water residual %.1e kg/m², transport %.2e kg/m²; air heat closure %.1e W/m²",
                           label, time() - t0, dom.Rn, dom.H, dom.LE, dom.G, dom.dS, dom.residual,
                           dom.closure, w.residual, get(w.sources, :transport, 0.0),
                           hb.closure / hb.elapsed)
            if irun == 1
                print(BC.ledger_report(case))
                # Midday sun: the surfaces gain radiative energy and warm.
                @test dom.Rn > 0 && dom.dS > 0
                # The flux-form transport conserves water: the scalar-
                # advection coupling refreshes the velocity halos step!
                # leaves at their predictor values (#132; before, step 1
                # alone added 7.5e-3 kg/m²).
                @test abs(get(w.sources, :transport, 0.0)) < 1e-12
            end
        end
    end

    @testset "every registered coupling ran in some smoke configuration" begin
        missing_c = setdiff(Set(BC.coupling_name.(BC.REGISTRY)), exercised)
        @test isempty(missing_c)
    end

    elapsed = time() - t_start
    @info @sprintf("Berlin smoke test: %d runs, %d coupling checks, %.1f s (first-call compilation included)",
                   length(runs), nchecks, elapsed)
    @test elapsed < 300
end

struct _SmokeA <: BC.AbstractCoupling end
struct _SmokeB <: BC.AbstractCoupling end
struct _SmokeAddon <: BC.AbstractCoupling end
struct _SmokeC <: BC.AbstractCoupling end
struct _SmokeD <: BC.AbstractCoupling end

@testset "smoke matrix: derived from the registry" begin
    # a synthetic registry (not registered): a slot with a default and an
    # alternative provider, an addon that is off by default and conflicts
    # with the alternative, a second slot whose default provider the
    # alternative also conflicts with, and a coupling that asks for one
    # extra variant
    BC.coupling_name(::Type{_SmokeA}) = :smoke_a
    BC.coupling_name(::Type{_SmokeB}) = :smoke_b
    BC.coupling_name(::Type{_SmokeAddon}) = :smoke_addon
    BC.slot(::Type{_SmokeA}) = :smoke_slot
    BC.slot(::Type{_SmokeB}) = :smoke_slot
    BC.provider(::Type{_SmokeA}) = :a
    BC.provider(::Type{_SmokeB}) = :b
    BC.is_default(::Type{_SmokeB}) = false
    BC.is_default(::Type{_SmokeAddon}) = false
    BC.coupling_name(::Type{_SmokeC}) = :smoke_c
    BC.coupling_name(::Type{_SmokeD}) = :smoke_d
    BC.slot(::Type{_SmokeC}) = :smoke_slot2
    BC.slot(::Type{_SmokeD}) = :smoke_slot2
    BC.provider(::Type{_SmokeC}) = :c
    BC.provider(::Type{_SmokeD}) = :d
    BC.is_default(::Type{_SmokeD}) = false
    BC.incompatible(::Type{_SmokeB}) = (:smoke_addon, :smoke_slot2 => :c)
    BC.smoke_variants(::Type{_SmokeA}) = ((label = "coarse", options = Dict(:smoke_a => (n = 2,)), nsteps = 4),)
    runs = Dict(smoke_matrix(Type[_SmokeA, _SmokeB, _SmokeAddon, _SmokeC, _SmokeD]))
    @test Set(keys(runs)) == Set(["summer, everything on", "winter, everything on",
                                  "summer sounding, everything on", "smoke_slot = b", "smoke_slot2 = d", "smoke_a: coarse"])
    @test runs["summer, everything on"].physics[:smoke_addon] == true       # default-off addon on
    @test runs["winter, everything on"].scenario === :winter
    @test runs["winter, everything on"].θ_surface == BC.SCENARIOS[:winter].θ_surface
    @test runs["smoke_slot = b"].physics[:smoke_slot] === :b
    @test runs["smoke_slot = b"].physics[:smoke_addon] == false             # conflict resolved
    @test runs["smoke_slot = b"].physics[:smoke_slot2] === :d               # slot default gives way
    @test runs["smoke_slot2 = d"].physics[:smoke_slot2] === :d
    @test !haskey(runs["summer, everything on"].physics, :smoke_slot2)      # defaults untouched
    @test runs["smoke_a: coarse"].nsteps == 4
    @test runs["smoke_a: coarse"].options[:smoke_a] == (n = 2,)
    @test !haskey(runs["summer, everything on"].options, :pollutant)        # not in this registry
end

@testset "smoke matrix: the \"everything on\" runs enable every addon and every default provider" begin
    # The flagship rule (CLAUDE.md): the "everything on" runs must really run
    # every additive coupling (none switched off by a declared
    # incompatibility) and the default provider of every exclusive slot.
    runs = Dict(smoke_matrix())
    addons = Set(BC.coupling_name(C) for C in BC.REGISTRY if BC.slot(C) === :addon)
    defaults = Set(BC.coupling_name(C) for C in BC.REGISTRY if BC.slot(C) !== :addon && BC.is_default(C))
    for label in ("summer, everything on", "winter, everything on", "summer sounding, everything on")
        cfg = runs[label]
        @test all(a -> cfg.physics[a] === true, addons)
        got = Set(BC.coupling_name.(BC.selected_couplings(cfg)))
        miss = sort!(collect(setdiff(union(addons, defaults), got)))
        isempty(miss) || @error "smoke run \"$label\" does not run: $(join(miss, ", ")) " *
            "(switched off by an `incompatible` declaration?)"
        @test isempty(miss)
    end
end
