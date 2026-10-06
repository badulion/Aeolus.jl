# The Berlin coverage test (#127): every submodule of Aeolus is either run
# by some coupling of the Berlin case (it appears in that coupling's
# `modules_used`), exempt for a stated reason, or on the TEMPORARY
# pending list with the issue that wires it in.
#
# The test ratchets:
#   * a new module in src/ fails it until it is wired in or exempted;
#   * a module on the pending list that a coupling now uses fails it
#     until the module is removed from the list — so the list only
#     shrinks, and the PR that wires a module in says so here.
#
# When your coupling runs a module: list it in `modules_used` and delete
# its line from PENDING (and BERLIN_PENDING_FROZEN) below. See
# docs/src/design/berlin_case.md.
#
# The flagship guardrails (the maintainer's rule: the Berlin case runs
# every component, on by default, and stays up to date with new physics):
#   * BERLIN_EXEMPT, BERLIN_PENDING, BERLIN_DEFAULT_OFF and
#     BERLIN_OPTION_ONLY are frozen: they may only shrink;
#   * every addon coupling is on by default, every exclusive slot has one
#     default provider, and the default CaseConfig (and every scenario)
#     runs them all;
#   * a module that only a non-default provider runs is a reviewed,
#     frozen exception.

using Test
using Aeolus

isdefined(Main, :BerlinCase) ||
    include(joinpath(@__DIR__, "..", "examples", "urban", "BerlinCase.jl"))
const BC = Main.BerlinCase

"""
Modules the Berlin case is not required to couple, with the reason.
Exemptions are for **non-physics** code only: the infrastructure the
case's core runs on, or a scaffold with no operator to call. Every
reason starts with `infrastructure:` or `scaffold:`. Physics is never
exempt; "not done yet" is what PENDING is for. The list is frozen
(`BERLIN_EXEMPT_FROZEN` below).
"""
const BERLIN_EXEMPT = Dict{Symbol,String}(
    # infrastructure the core itself runs on (core/case.jl, stages.jl, output.jl)
    :Grids              => "infrastructure: the RegularGrid of every case (core/case.jl)",
    :Fields             => "infrastructure: every field of the case (core)",
    :BoundaryConditions => "infrastructure: the periodic/wall boundary conditions of the case (core/case.jl)",
    :Operators          => "infrastructure: the finite-difference operators step! is built from",
    :Simulations        => "infrastructure: BoussinesqSimulation and step!, the dynamical core (core/case.jl)",
    :PressureSolvers    => "infrastructure: the cosine-FFT pressure solve inside step!",
    :TimeSteppers       => "infrastructure: the RK3 stepper Simulations is built on",
    :NetCDFIO           => "infrastructure: the NetCDF output writer (core/output.jl)",
    # scaffolds with no operators to couple
    :Parallel           => "scaffold: partition types with no distributed operators (no MPI); nothing to run",
    :Backends           => "scaffold: CPU/GPU backend abstraction with no GPU kernels; nothing to run",
    :SpectralElement    => "scaffold: a spectral-element mesh type with no solver; the case is finite-volume",
)

"""
TEMPORARY: modules not yet wired into the Berlin case, with the issue that
wires each one in. This list may only shrink. The PR that couples a
module removes it here, and from `BERLIN_PENDING_FROZEN`, in the same PR
(the ratchet test below fails until it does). It is frozen: a new src
module cannot be parked here, it must be coupled (or, if it is not
physics, exempted).
"""
const BERLIN_PENDING = Dict{Symbol,String}(
    :Particles             => "#134 Lagrangian particles",
    :DataAssimilation      => "#134 nudging assimilation",
    :IO                    => "#134 checkpoint / restart",
    :MonteCarloRT          => "#136 Monte-Carlo validation of the radiosity shortwave",
    :Visibility            => "#137 visibility diagnostic",
    :LandSurface           => "#140 park = VegetatedLandColumn",
    :Photosynthesis        => "#140 park = VegetatedLandColumn",
    :OceanMixedLayer       => "#141 water surfaces",
    :AirSeaFlux            => "#141 water surfaces (COARE option)",
    :AerosolOptics         => "#142 urban aerosol optics",
    :P3Microphysics        => "#143 microphysics slot, P3 provider",
    :Snowpack              => "#146 winter scenario",
    :IceThermodynamics     => "#146 winter scenario (river / lake ice)",
    :AerosolCloud          => "#147 activation",
    :TurbulentMicrophysics => "#147 SGS cloud",
    :IceNucleation         => "#148 INP tracers and P3 ice",
    :SurfaceTiles          => "#149 tiles / blended mosaic (coarse configuration)",
    :Superdroplet          => "#150 superdroplets in a fog sub-volume",
    :StretchedGrids        => "#151 stretched-grid option",
    :CutCell               => "#152 cut-cell buildings option",
    :AdaptiveMesh          => "#153 nesting with forward hooks (AMR blocks themselves cannot carry the IBM/surfaces: exempt that part with the reason then)",
    :Nesting               => "#153 multi-domain case (legacy one-way forcing, deferred with AdaptiveMesh)",
    :RunLog                => "awaiting #320 (#338: the Berlin console text through the logger)",
)

# --- frozen snapshots ---------------------------------------------------------
#
# BERLIN_EXEMPT, BERLIN_PENDING, BERLIN_DEFAULT_OFF and BERLIN_OPTION_ONLY
# are frozen at the sets below (taken on master at PR #160, 2026-10-01).
# The tests require each list to equal its snapshot, so:
#
#   * an entry may only be DELETED, from the list and its snapshot
#     together (a pending module that gets coupled; an exemption that
#     goes away);
#   * ADDING an entry means editing a snapshot. Do not, unless the
#     maintainer has agreed in the issue: new physics is wired into the
#     Berlin case, on by default, in the PR that adds it (CLAUDE.md, "New
#     physics lands in the Berlin example"). A reviewer who sees a line
#     added to a snapshot should ask why.

"Frozen: the modules of `BERLIN_EXEMPT`. Entries may only be deleted."
const BERLIN_EXEMPT_FROZEN = Set{Symbol}([
    :Grids, :Fields, :BoundaryConditions, :Operators, :Simulations, :PressureSolvers,
    :TimeSteppers, :NetCDFIO, :Parallel, :Backends, :SpectralElement,
])

"Frozen: the modules of `BERLIN_PENDING`. Entries may only be deleted."
const BERLIN_PENDING_FROZEN = Set{Symbol}([
    :Particles, :DataAssimilation, :IO, :MonteCarloRT,
    :Visibility, :LandSurface, :Photosynthesis, :OceanMixedLayer, :AirSeaFlux,
    :AerosolOptics, :P3Microphysics, :Snowpack, :IceThermodynamics, :AerosolCloud,
    :TurbulentMicrophysics, :IceNucleation, :SurfaceTiles, :Superdroplet,
    :StretchedGrids, :CutCell, :AdaptiveMesh, :Nesting,
    :RunLog,   # #333 logging core, added under the transition rule (CLAUDE.md) until #320
])

"""
Additive couplings (`slot = :addon`) that are OFF in the default run,
with the reason (e.g. the measured cost that makes it an option). Every
other addon must have `is_default = true`: the flagship runs every
component. Empty at #160: every addon is on. Adding an entry needs the
maintainer's agreement in the issue, and the reason must quote the
measured cost or the conflict.
"""
const BERLIN_DEFAULT_OFF = Dict{Symbol,String}()

"Frozen: the couplings of `BERLIN_DEFAULT_OFF`. Entries may only be deleted."
const BERLIN_DEFAULT_OFF_FROZEN = Set{Symbol}()

"""
Modules the Berlin case runs ONLY through a non-default provider of an
exclusive slot (`physics = Dict(slot => provider)`), with the reason the
provider is an alternative and not the default. They count as coupled
(the smoke test runs every provider), but they are not in the default
run, so the list is reviewed and frozen like the others. New physics
that is a genuine alternative (P3 vs Kessler) may land here, with the
reason and the maintainer's agreement; never by default.
"""
const BERLIN_OPTION_ONLY = Dict{Symbol,String}(
    :Multigrid => "pressure_solver = :multigrid / :mg_cg: the same 7-point Laplacian as the exact " *
                  "cosine-FFT default, which MG-CG matches to 6.1e-10 m/s, at 23-36x the cost per " *
                  "solve (45.8-72.6 vs 2.0 ms); kept for the stretched-grid / cut-cell solvers (#151, #152)",
    :ML        => "sgs = :mlsgs: wraps ML.MLSGS, and there is no trained model; with the default " *
                  "closure it is bit for bit the Smagorinsky default (berlin_numerics.md, numerics slots)",
    :Anelastic => "core = :anelastic / :pseudo_incompressible (with pressure_solver = " *
                  ":density_weighted_cg and timestepper = :fixed): in the flagship's 200 m deep " *
                  "domain the density changes by 2 % and the sound-proof cores give the Boussinesq " *
                  "mean profile to 0.002-0.004 K after 1 h at 2.2x the cost, without " *
                  "WENO-Z momentum or the adaptive step (#133, #166); the maintainer kept the " *
                  "Boussinesq core as the default",
)

"Frozen: the modules of `BERLIN_OPTION_ONLY`. Entries may only be deleted."
const BERLIN_OPTION_ONLY_FROZEN = Set{Symbol}([:Multigrid, :ML, :Anelastic])

"""
Reviewed indirect uses: `(coupling, module) => how`. A coupling may list
a module it does not call itself when a src function it does call runs
that module's physics every time. The honesty check below accepts these.
"""
const BERLIN_INDIRECT = Dict{Tuple{Symbol,Symbol},String}(
    (:radiosity_sun, :SurfaceOptics) =>
        "Radiosity.default_urban_materials band-averages the SurfaceOptics spectral " *
        "library (tiled_surface, broadband_albedo, FresnelWater) at every radiation update",
)

"Every submodule of Aeolus (direct children)."
aeolus_submodules() =
    Set(n for n in names(Aeolus; all = true)
        if n !== :Aeolus && isdefined(Aeolus, n) && (m = getfield(Aeolus, n)) isa Module &&
           parentmodule(m) === Aeolus)

"module => the couplings that list it in `modules_used`."
function modules_in_use()
    d = Dict{Symbol,Vector{Symbol}}()
    for C in BC.REGISTRY, m in BC.modules_used(C)
        push!(get!(d, m, Symbol[]), BC.coupling_name(C))
    end
    return d
end

"Julia source with docstrings, string literals and comments removed."
function code_only(src::AbstractString)
    src = replace(src, r"\"\"\"(?s:.*?)\"\"\"" => " ")
    src = replace(src, r"\"(?:\\\\.|[^\"\\\\])*\"" => "\"\"")
    src = replace(src, r"#=(?s:.*?)=#" => " ")
    return replace(src, r"#[^\n]*" => " ")
end

"The source file a coupling type is defined in (where its `coupling_name` method is)."
coupling_file(C) = string(first(methods(BC.coupling_name, (Type{C},))).file)

@testset "Berlin coverage test (#127)" begin
    mods = aeolus_submodules()
    used = modules_in_use()

    @testset "the lists name real modules and do not overlap" begin
        @test length(mods) >= 50
        @test isempty(setdiff(keys(BERLIN_EXEMPT), mods))       # no stale names
        @test isempty(setdiff(keys(BERLIN_PENDING), mods))
        @test isempty(intersect(keys(BERLIN_EXEMPT), keys(BERLIN_PENDING)))
        @test all(r -> length(r) >= 10, values(BERLIN_EXEMPT))  # a reason, not a label
        @test all(r -> occursin(r"#\d+", r), values(BERLIN_PENDING))  # an owning issue
        for (C, ms) in ((C, BC.modules_used(C)) for C in BC.REGISTRY)
            @test all(m -> m in mods, ms)
        end
    end

    @testset "every Aeolus module is coupled, exempt or pending" begin
        unaccounted = sort!(collect(setdiff(mods, keys(used), keys(BERLIN_EXEMPT), keys(BERLIN_PENDING))))
        isempty(unaccounted) || @error "Aeolus modules the Berlin case neither runs, exempts nor lists as pending: " *
            join(unaccounted, ", ") * ". Wire them into a coupling (docs/src/design/berlin_case.md), " *
            "or add them to BERLIN_EXEMPT in test/berlin_coverage.jl with the reason."
        @test unaccounted == Symbol[]
    end

    @testset "ratchet: a coupled module is no longer pending" begin
        stale = sort!([m for m in keys(BERLIN_PENDING) if haskey(used, m)])
        isempty(stale) || @error "Remove from BERLIN_PENDING (test/berlin_coverage.jl), now coupled: " *
            join(("$m (by $(join(used[m], ", ")))" for m in stale), "; ")
        @test stale == Symbol[]
    end

    @testset "modules_used is honest: the coupling's file calls into each module" begin
        # A cheap check that a claim in `modules_used` is not just a label:
        # the code of the coupling's file (comments and docstrings removed)
        # must use at least one name the module exports, or qualify a name
        # with the module (`MoistThermodynamics.latent_heat_vapor`), or be
        # a reviewed indirect use (BERLIN_INDIRECT).
        for C in BC.REGISTRY
            src = code_only(read(coupling_file(C), String))
            words = Set(m.match for m in eachmatch(r"[A-Za-z_][A-Za-z_0-9!]*", src))
            for m in BC.modules_used(C)
                M = getfield(Aeolus, m)
                ok = haskey(BERLIN_INDIRECT, (BC.coupling_name(C), m)) ||
                     occursin(Regex("\\b$(m)\\."), src) ||
                     any(n -> n !== m && string(n) in words, names(M))
                ok || @error "coupling :$(BC.coupling_name(C)) lists :$m in modules_used but $(coupling_file(C)) uses nothing from it"
                @test ok
            end
        end
    end

    @testset "BerlinCase shadows no Aeolus export" begin
        # BerlinCase does `using Aeolus`; a function it defines under an
        # exported Aeolus name silently hides the src one, and a coupling
        # that later calls the src function inside BerlinCase gets a
        # MethodError (the ledger's former `column_water` hid
        # RadiationFog.column_water this way).
        shadowed = sort!([n for n in names(Aeolus) if n !== :Aeolus && isdefined(BC, n) &&
                          Base.binding_module(BC, n) === BC])
        isempty(shadowed) || @error "BerlinCase defines names Aeolus exports: $(join(shadowed, ", ")). Rename them."
        @test shadowed == Symbol[]
    end

    @testset "frozen lists: EXEMPT / PENDING / DEFAULT_OFF / OPTION_ONLY may only shrink" begin
        for (lname, list, frozen) in (("BERLIN_EXEMPT", BERLIN_EXEMPT, BERLIN_EXEMPT_FROZEN),
                                      ("BERLIN_PENDING", BERLIN_PENDING, BERLIN_PENDING_FROZEN),
                                      ("BERLIN_DEFAULT_OFF", BERLIN_DEFAULT_OFF, BERLIN_DEFAULT_OFF_FROZEN),
                                      ("BERLIN_OPTION_ONLY", BERLIN_OPTION_ONLY, BERLIN_OPTION_ONLY_FROZEN))
            grown = sort!(collect(setdiff(keys(list), frozen)))
            isempty(grown) || @error "$lname has grown: $(join(grown, ", ")). It may only shrink: new " *
                "physics is wired into the Berlin case, on by default (CLAUDE.md). Only the maintainer " *
                "extends $(lname)_FROZEN in test/berlin_coverage.jl."
            @test grown == Symbol[]
            shrunk = sort!(collect(setdiff(frozen, keys(list))))
            isempty(shrunk) || @error "$lname no longer lists $(join(shrunk, ", ")): " *
                "delete them from $(lname)_FROZEN in test/berlin_coverage.jl too."
            @test shrunk == Symbol[]
        end
        # exemptions are for non-physics code only
        notinfra = sort!([m for (m, r) in BERLIN_EXEMPT
                          if !(startswith(r, "infrastructure:") || startswith(r, "scaffold:"))])
        isempty(notinfra) || @error "BERLIN_EXEMPT reasons must start with `infrastructure:` or " *
            "`scaffold:` (physics is never exempt): $(join(notinfra, ", "))"
        @test notinfra == Symbol[]
        @test all(r -> length(r) >= 20, values(BERLIN_DEFAULT_OFF))
        @test all(r -> length(r) >= 20, values(BERLIN_OPTION_ONLY))
    end

    @testset "flagship: every addon on by default, every slot's default provider in the default run" begin
        addons = [C for C in BC.REGISTRY if BC.slot(C) === :addon]
        @test isempty(setdiff(keys(BERLIN_DEFAULT_OFF), BC.coupling_name.(addons)))   # no stale names
        off = sort!([BC.coupling_name(C) for C in addons
                     if !BC.is_default(C) && !haskey(BERLIN_DEFAULT_OFF, BC.coupling_name(C))])
        isempty(off) || @error "Addon couplings off by default: $(join(off, ", ")). The Berlin case is the " *
            "flagship and runs every component: set is_default = true (or, with the maintainer's " *
            "agreement and the measured cost, list it in BERLIN_DEFAULT_OFF in test/berlin_coverage.jl)."
        @test off == Symbol[]
        # an allow-listed addon really is off by default (else the entry is stale)
        @test all(C -> !haskey(BERLIN_DEFAULT_OFF, BC.coupling_name(C)) || !BC.is_default(C), addons)

        # Every exclusive slot has exactly one default provider, and it is not
        # an "off" switch; so every non-default provider is an alternative in
        # a slot whose default runs.
        slots = BC.slot_providers()
        for (s, provs) in slots
            defs = [C for C in BC.REGISTRY if BC.slot(C) === s && BC.is_default(C)]
            length(defs) == 1 || @error "slot :$s has $(length(defs)) default providers; it needs exactly one"
            @test length(defs) == 1
            offs = [BC.coupling_name(C) for C in defs if BC.provider(C) in (:off, :none, :disabled)]
            isempty(offs) || @error "slot :$s defaults to an off switch ($(join(offs, ", "))); " *
                "the flagship runs the physics of every slot by default"
            @test isempty(offs)
        end

        # The default CaseConfig, and every scenario preset, select every addon
        # not allow-listed and the default provider of every slot.
        want = Set(BC.coupling_name(C) for C in BC.REGISTRY
                   if BC.slot(C) === :addon ? !haskey(BERLIN_DEFAULT_OFF, BC.coupling_name(C)) :
                                              BC.is_default(C))
        cfgs = Pair{String,Any}["CaseConfig()" => BC.CaseConfig()]
        for s in sort!(collect(keys(BC.SCENARIOS)))
            push!(cfgs, "CaseConfig(:$s)" => BC.CaseConfig(s))
        end
        for (label, cfg) in cfgs
            got = Set(BC.coupling_name.(BC.selected_couplings(cfg)))
            miss = sort!(collect(setdiff(want, got)))
            isempty(miss) || @error "$label does not run: $(join(miss, ", "))"
            @test isempty(miss)
        end
    end

    @testset "flagship: a module only a non-default provider runs is a reviewed exception" begin
        # A module in some coupling's `modules_used` is coupled (the smoke test
        # runs every provider). One the DEFAULT run does not use must be in
        # BERLIN_OPTION_ONLY with a reason: new physics is on by default.
        default_mods = Set{Symbol}()
        for C in BC.selected_couplings(BC.CaseConfig())
            union!(default_mods, BC.modules_used(C))
        end
        option_only = Set(m for m in keys(used) if !(m in default_mods))
        unreviewed = sort!(collect(setdiff(option_only, keys(BERLIN_OPTION_ONLY))))
        isempty(unreviewed) || @error "Modules the default Berlin run does not use (only a non-default " *
            "provider does): " * join(("$m (by $(join(used[m], ", ")))" for m in unreviewed), "; ") *
            ". Make that coupling the default, or (a genuine alternative, with the maintainer's " *
            "agreement) list the module in BERLIN_OPTION_ONLY in test/berlin_coverage.jl with the reason."
        @test unreviewed == Symbol[]
        stale = sort!(collect(setdiff(keys(BERLIN_OPTION_ONLY), option_only)))
        isempty(stale) || @error "BERLIN_OPTION_ONLY lists modules the default run now uses (or no " *
            "coupling uses): $(join(stale, ", ")). Delete them there and from BERLIN_OPTION_ONLY_FROZEN."
        @test stale == Symbol[]
    end

    ncoupled = length(intersect(mods, keys(used)))
    @info "Berlin coverage: $(length(mods)) Aeolus modules — $ncoupled coupled, " *
          "$(length(BERLIN_EXEMPT)) exempt, $(length(BERLIN_PENDING)) pending" *
          " (coupled: $(join(sort!(collect(intersect(mods, keys(used)))), ", ")))"
end
