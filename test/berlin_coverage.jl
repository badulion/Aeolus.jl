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
# its line from PENDING below. See docs/src/design/berlin_case.md.

using Test
using Aeolus

isdefined(Main, :BerlinCase) ||
    include(joinpath(@__DIR__, "..", "examples", "urban", "BerlinCase.jl"))
const BC = Main.BerlinCase

"""
Modules the Berlin case is not required to couple, with the reason. A
reviewer has to agree with the reason; "not done yet" is not one (that
is what PENDING is for).
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
module removes it here in the same PR (the ratchet test below fails
until it does).
"""
const BERLIN_PENDING = Dict{Symbol,String}(
    :SurfaceExchange       => "#129 MOST surface fluxes",
    :Anelastic             => "#133 anelastic / pseudo-incompressible core (#131 may use its ReferenceState first)",
    :Multigrid             => "#132 pressure-solver option",
    :ML                    => "#132 MLSGS option",
    :TimeIntegration       => "#133 adaptive dt / IMEX",
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
)

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

    ncoupled = length(intersect(mods, keys(used)))
    @info "Berlin coverage: $(length(mods)) Aeolus modules — $ncoupled coupled, " *
          "$(length(BERLIN_EXEMPT)) exempt, $(length(BERLIN_PENDING)) pending" *
          " (coupled: $(join(sort!(collect(intersect(mods, keys(used)))), ", ")))"
end
