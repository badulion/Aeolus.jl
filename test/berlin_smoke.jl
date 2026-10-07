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
# Each run is examples/urban/configs/berlin_smoke.toml with the run's
# `--set` overrides (CFG-c, #358; test/support/smoke_matrix.jl): a 250 m × 250 m × 100 m window of the
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

# smoke_matrix(registry) and seed_fog!(case, rng): shared with the
# thread-invariance harness (KA-a, #294).
include(joinpath(@__DIR__, "support", "smoke_matrix.jl"))

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
            case = BC.build_case(cfg)
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
    # Wall-clock budget: 300 s on a development machine. Slower CI runners
    # (4-core GitHub runners take about 340 s) set AEOLUS_SMOKE_BUDGET_S.
    budget = parse(Float64, get(ENV, "AEOLUS_SMOKE_BUDGET_S", "300"))
    @test elapsed < budget
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
    runs = Dict(smoke_specs(Type[_SmokeA, _SmokeB, _SmokeAddon, _SmokeC, _SmokeD]))
    @test Set(keys(runs)) == Set(["summer, everything on", "winter, everything on",
                                  "summer sounding, everything on", "smoke_slot = b", "smoke_slot2 = d", "smoke_a: coarse"])
    @test runs["summer, everything on"].physics[:smoke_addon] == true       # default-off addon on
    # the winter run is the smoke file plus berlin_winter.toml's values (CFG-c, #358)
    @test "run.scenario=\"winter\"" in runs["winter, everything on"].extra
    @test "initial.theta_surface=271.0" in runs["winter, everything on"].extra
    @test !any(startswith("run.name="), runs["winter, everything on"].extra)
    @test runs["smoke_slot = b"].physics[:smoke_slot] === :b
    @test runs["smoke_slot = b"].physics[:smoke_addon] == false             # conflict resolved
    @test runs["smoke_slot = b"].physics[:smoke_slot2] === :d               # slot default gives way
    @test runs["smoke_slot2 = d"].physics[:smoke_slot2] === :d
    @test !haskey(runs["summer, everything on"].physics, :smoke_slot2)      # defaults untouched
    @test runs["smoke_a: coarse"].fields.nsteps == 4
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
        # a selection equal to the registry default is not stored (CFG-c, #358)
        @test all(a -> get(cfg.physics, a, BC.is_default(BC.coupling_type(a))) === true, addons)
        got = Set(BC.coupling_name.(BC.selected_couplings(cfg)))
        miss = sort!(collect(setdiff(union(addons, defaults), got)))
        isempty(miss) || @error "smoke run \"$label\" does not run: $(join(miss, ", ")) " *
            "(switched off by an `incompatible` declaration?)"
        @test isempty(miss)
    end
end
