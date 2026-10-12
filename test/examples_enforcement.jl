# Examples are configuration (single_model.md, "Tests" and "Repository
# layout"; ONE-i #458). Zero exemptions: there is no list of files that may
# break these rules.
#
#   * No `.jl` file under examples/ builds or steps a model. Each file is
#     parsed and every call in it is checked against the model's API, which
#     is read from the package (not written down here), so a new entry
#     point is caught without editing this file:
#       - a function of Aeolus whose name ends in `!` (it changes state: a
#         step, a halo fill, a field set, a microphysics update, ...);
#       - a constructor of a type of the modules that build a model (grids,
#         fields, simulations, domains, components, nests, pressure
#         solvers, time steppers, run types);
#       - the entry points that run a configuration (`run_config`,
#         `Aeolus.main`, `Config.build`, `run!`) or register a run type
#         (`register_driver!`);
#       - `include` of a file outside examples/, or of a path that is not a
#         literal (it could not be checked).
#     The analysis scripts may read files: run.toml, results.toml,
#     diagnostics.csv/.nc and the NetCDF output.
#   * The layout: a case is a directory examples/<case>/ with its TOML files
#     at its top, its NetCDF drivers in input/ and its reference data in
#     data/; there are no configs/ directories; a directory without a TOML
#     (examples/viz/, the plotting library) holds Julia files only.
#
# The checks are run on synthetic files too, so that a check that never
# fires cannot pass for a clean tree.

using Test
using Aeolus

const EF_ROOT = joinpath(pkgdir(Aeolus), "examples")

# --- the model's API, from the package ---------------------------------------

"Every module of Aeolus (the package and its submodules, recursively)."
function ef_modules(m::Module = Aeolus, out = Module[])
    m in out && return out
    push!(out, m)
    for s in names(m; all = true)
        isdefined(m, s) || continue
        x = getfield(m, s)
        x isa Module && x !== m && parentmodule(x) === m && ef_modules(x, out)
    end
    return out
end

const EF_MODULES = ef_modules()

"Whether `x` (a function or type) is defined in Aeolus. A `Union` alias (`const Wall = Union{A, B}`) is no type of its own and constructs nothing."
ef_owned(x) = Base.unwrap_unionall(x) isa Union ? false :
              (pm = x isa Type ? parentmodule(x) : parentmodule(typeof(x)); pm in EF_MODULES)

"The functions of Aeolus whose name ends in `!`: they change the state they are given."
const EF_MUTATING = Set(s for m in EF_MODULES for s in names(m; all = true)
                        if endswith(string(s), "!") && !startswith(string(s), "#") && isdefined(m, s) &&
                           getfield(m, s) isa Function && ef_owned(getfield(m, s)))

# the modules whose types are a model or a part of one; a module that a
# later refactor removes simply drops out (isdefined)
const EF_BUILD_MODULES = (:Grids, :StretchedGrids, :AdaptiveMesh, :Fields, :BoundaryConditions, :Simulations,
                          :Anelastic, :TimeSteppers, :TimeIntegration, :PressureSolvers, :Multigrid,
                          :Components, :Nesting, :IBM, :CutCell, :Particles, :Superdroplet, :SpectralElement,
                          :Parallel, :Backends, :Config, :RunDrivers, :SimulationDriver, :UrbanCase)

"The types of the modules that build a model: calling one constructs a part of a model."
const EF_CONSTRUCTORS = Set(s for n in EF_BUILD_MODULES if isdefined(Aeolus, n)
                            for m in EF_MODULES if m === getfield(Aeolus, n) || parentmodule(m) === getfield(Aeolus, n)
                            for s in names(m; all = true)
                            if isdefined(m, s) && getfield(m, s) isa Type && ef_owned(getfield(m, s)) &&
                               !startswith(string(s), "#"))

"The entry points that run a configuration or register a run type."
const EF_ENTRY = Set([:run_config, :main, :build, :run!, :register_driver!])

const EF_FORBIDDEN = union(EF_MUTATING, EF_CONSTRUCTORS, EF_ENTRY)

# --- the scan -----------------------------------------------------------------

"The called name of `f` (`f`, `M.f`, `M.N.f`), or nothing."
ef_name(f::Symbol) = f
ef_name(f::Expr) = f.head === :. && length(f.args) == 2 && f.args[2] isa QuoteNode ? f.args[2].value :
                   f.head === :curly ? ef_name(f.args[1]) : nothing
ef_name(_) = nothing

"A literal path expression `include(joinpath(@__DIR__, \"..\", \"viz\", \"x.jl\"))` resolved against `dir`, or nothing."
function ef_literal_path(ex, dir)
    ex isa String && return normpath(joinpath(dir, ex))
    ex isa Expr && ex.head === :call && ex.args[1] === :joinpath || return nothing
    parts = String[]
    for a in ex.args[2:end]
        if a isa Expr && a.head === :macrocall && a.args[1] === Symbol("@__DIR__")
            push!(parts, dir)
        elseif a isa String
            push!(parts, a)
        else
            return nothing
        end
    end
    return normpath(joinpath(parts...))
end

"""
    ef_offences(src, path) -> Vector{String}

What the Julia source `src` (the file `path` under examples/) does that an
example may not: forbidden calls and includes, one line each.
"""
function ef_offences(src::AbstractString, path::AbstractString; root = EF_ROOT)
    out = String[]
    ast = Meta.parseall(src; filename = path)
    line = Ref(0)
    function walk(ex)
        ex isa LineNumberNode && (line[] = ex.line; return)
        ex isa Expr || return
        f = ex.head === :call ? ex.args[1] :
            ex.head === :. && length(ex.args) == 2 && ex.args[2] isa Expr && ex.args[2].head === :tuple ? ex.args[1] :
            nothing                                            # f.(x): a broadcast call
        if f !== nothing
            n = ef_name(f)
            if n === :include || n === :evalfile
                p = length(ex.args) >= 2 ? ef_literal_path(ex.args[end], dirname(path)) : nothing
                if p === nothing
                    push!(out, "$path:$(line[]): $n of a path that is not a literal")
                elseif !startswith(p, normpath(root) * "/")
                    push!(out, "$path:$(line[]): $n of $p, outside examples/")
                end
            elseif n !== nothing && n in EF_FORBIDDEN
                push!(out, "$path:$(line[]): calls $n")
            end
        end
        # `using Aeolus: step!` names a forbidden function too
        if ex.head in (:using, :import)
            for a in ex.args, s in (a isa Expr && a.head === :(:) ? a.args[2:end] : ())
                n = s isa Expr && s.head === :. ? s.args[end] : nothing
                n isa Symbol && n in EF_FORBIDDEN && push!(out, "$path:$(line[]): imports $n")
            end
        end
        foreach(walk, ex.args)
    end
    walk(ast)
    return out
end

"""
    ef_files(root) -> Vector{String}

The files under `root`, relative to it: in a git checkout the tracked ones
and the untracked ones git does not ignore (a new script is seen before it
is committed; run output, which .gitignore names, is not); else every file.
"""
function ef_files(root::AbstractString = EF_ROOT)
    git = Sys.which("git")
    if git !== nothing && success(`$git -C $root rev-parse --is-inside-work-tree`)
        out = read(`$git -C $root ls-files --cached --others --exclude-standard -z .`, String)
        return sort!(filter(f -> isfile(joinpath(root, f)), unique(split(out, '\0'; keepempty = false))))
    end
    return sort!([relpath(joinpath(d, f), root) for (d, _, fs) in walkdir(root) for f in fs])
end

"Every `.jl` file under `root`, with its path."
ef_scripts(root = EF_ROOT) = [joinpath(root, f) for f in ef_files(root) if endswith(f, ".jl")]

# --- the layout -----------------------------------------------------------------

"""
    ef_layout_offences(root) -> Vector{String}

What under `root` breaks the layout of an example case (see the header).
"""
function ef_layout_offences(root::AbstractString = EF_ROOT; files = ef_files(root))
    out = String[]
    parts = [splitpath(f) for f in files]
    any(p -> "configs" in p[1:end-1], parts) &&
        push!(out, "configs/: a configs/ directory (the TOML files sit in the case directory)")
    for p in parts
        r = joinpath(p...); f = p[end]; depth = length(p) - 1
        if f == "Project.toml" && depth == 0
            continue                                   # the examples' environment
        elseif endswith(f, ".toml")
            depth == 1 || push!(out, "$r: a TOML file outside a case directory's top level")
        elseif endswith(f, ".nc")
            (depth == 2 && p[2] == "input") || push!(out, "$r: a NetCDF driver outside <case>/input/")
        elseif endswith(f, ".csv")
            (depth == 2 && p[2] == "data") || push!(out, "$r: data outside <case>/data/")
        elseif !endswith(f, ".jl")
            push!(out, "$r: neither a config, a driver, data nor a script")
        end
    end
    # a directory without a configuration (examples/viz/) holds scripts only
    for case in unique(p[1] for p in parts if length(p) > 1)
        mine = [p for p in parts if length(p) > 1 && p[1] == case]
        any(p -> length(p) == 2 && endswith(p[2], ".toml"), mine) && continue
        all(p -> length(p) == 2 && endswith(p[2], ".jl"), mine) ||
            push!(out, "$case: a directory without a configuration may hold scripts only")
    end
    return out
end

# --- the tests --------------------------------------------------------------------

@testset "examples: the model's API as read from the package" begin
    # the stepping and building entry points are in the forbidden set
    for n in (:step!, :step_rk3!, :step_adaptive!, :fill_halos!, :set!, :apply_kessler_step!, :RegularGrid,
              :Field, :BoussinesqSimulation, :run_config, :register_driver!)
        @test n in EF_FORBIDDEN
    end
    isdefined(Aeolus, :Components) && @test :Domain in EF_FORBIDDEN && :Model in EF_FORBIDDEN
    # what an analysis script needs is not: reading files and the config
    for n in (:NCDataset, :load_run_config, :read_run_toml, :read_dynamic_driver, :println, :joinpath)
        @test !(n in EF_FORBIDDEN)
    end
end

@testset "examples: the scan finds what builds or steps a model (synthetic files)" begin
    p = joinpath(EF_ROOT, "case", "x.jl")
    bad = """
        using Aeolus
        using Aeolus.Simulations: step!
        grid = RegularGrid(; size = (8, 8, 8), extent = (1.0, 1.0, 1.0))
        sim = Aeolus.BoussinesqSimulation(grid)
        for _ in 1:3
            Aeolus.Simulations.step_rk3!(sim, 0.1)
        end
        r = run_config(load_run_config("case.toml"))
        Aeolus.main(["run", "case.toml"])
        include(joinpath(@__DIR__, "..", "..", "test", "fixtures", "dns_decay.jl"))
        include(path_from_somewhere)
        set!.(fields, 0.0)
        """
    o = ef_offences(bad, p)
    for what in ("imports step!", "calls RegularGrid", "calls BoussinesqSimulation", "calls step_rk3!",
                 "calls run_config", "calls main", "outside examples/", "not a literal", "calls set!")
        @test count(x -> occursin(what, x), o) == 1
    end
    @test length(o) == 9
    @test occursin("x.jl:6:", only(filter(x -> occursin("step_rk3!", x), o)))     # with its line
    good = """
        using Aeolus
        using Aeolus.NCIO: NCDataset
        include(joinpath(@__DIR__, "..", "viz", "run_output.jl"))
        cfg = load_run_config(ARGS[1])
        ds = NCDataset(joinpath(ARGS[1], "output", "output.nc"))
        u = Array(ds["u"][:, :, :, end]); close(ds)
        println(maximum(u))
        """
    @test isempty(ef_offences(good, p))
end

@testset "examples: no .jl file builds or steps a model (zero exemptions)" begin
    files = ef_scripts()
    @test length(files) >= 30                          # the scan sees the tree
    offences = reduce(vcat, (ef_offences(read(f, String), f) for f in files); init = String[])
    @test isempty(offences)
    isempty(offences) || @info "examples that build or step a model" offences
end

@testset "examples: the layout of a case directory" begin
    # synthetic trees: every rule fires once
    root = mktempdir()                                 # not a git checkout: every file counts
    mk(p, s = "") = (mkpath(dirname(joinpath(root, p))); write(joinpath(root, p), s))
    mk("good/good.toml"); mk("good/good_variant.toml"); mk("good/input/good_static.nc"); mk("good/data/ref.csv")
    mk("good/plot_good.jl"); mk("viz/plot_utils.jl"); mk("Project.toml")
    @test isempty(ef_layout_offences(root))
    mk("old/configs/old.toml"); mk("loose.toml"); mk("good/static.nc"); mk("good/ref.csv"); mk("good/notes.txt")
    mk("scripts/x.jl"); mk("scripts/a.csv")
    o = ef_layout_offences(root)
    for what in ("configs/: a configs/", "old/configs/old.toml", "loose.toml", "good/static.nc",
                 "good/ref.csv", "good/notes.txt", "scripts/a.csv", "scripts: a directory without",
                 "old: a directory without")
        @test count(x -> startswith(x, what), o) == 1
    end
    @test length(o) == 9
    # the shipped tree
    o = ef_layout_offences()
    @test isempty(o)
    isempty(o) || @info "examples/ layout" o
    # every case directory holds at least one configuration
    cases = unique(first(splitpath(f)) for f in ef_files() if count('/', f) == 1 && endswith(f, ".toml"))
    @test length(cases) >= 20
    @test "taylor_green_vortex" in cases && "sbl" in cases && "urban" in cases
end
