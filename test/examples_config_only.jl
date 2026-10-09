# Examples are configuration (EX-e, #442; the maintainer's rule of
# 2026-10-09): every example is TOML configuration plus static and dynamic
# input drivers. A `.jl` under examples/ may only read a run's output files
# (and input or reference data) and analyse or plot them; no simulation
# code is called from a Julia file. Model code and run assembly live in
# src/ as run types, data fetching as `aeolus preprocess …`.
#
# Checked here, statically, on the parsed code of every examples/**/*.jl
# and of every file it includes (an allow-list, no per-file exemptions):
#   * imports: whole packages only from FREE_PACKAGES (standard library and
#     NetCDF reading); from Aeolus only the names of OUTPUT_API, imported by
#     name (`using Aeolus.IO: latest_run`). `using Aeolus` / `import Aeolus`
#     and any qualified `Aeolus.x` are refused, so nothing else of the
#     package is reachable;
#   * calls: no run, step or build function (FORBIDDEN_CALL), no eval, no
#     external command (a wrapper that shells out to `aeolus run`);
#   * include: only files under examples/ or the reference data in
#     test/data/, which are checked by the same rules.
# And per example directory: at least one configs/*.toml, and every one of
# them is accepted by `aeolus check`.

using Test
using Aeolus

const EXC_DIR = joinpath(pkgdir(Aeolus), "examples")
const EXC_REF_DATA = joinpath(pkgdir(Aeolus), "test", "data")

# Packages a script may load whole: the standard library it uses and the
# NetCDF reader of the output files.
const FREE_PACKAGES = Set([:Printf, :Statistics, :TOML, :Dates, :Test, :LinearAlgebra, :NCDatasets])

# The only names of Aeolus a script may use: readers of run output, configs
# and input/reference files, and pure functions of numbers (constants,
# thermodynamic relations, fits). None of them builds a grid, a model or a
# simulation, or steps one.
const OUTPUT_API = Dict(
    (:Aeolus,) => Set([:NCDataset,                      # the NetCDF reader (NCIO)
                       :load_run_config,                # a run's resolved config from its run.toml
                       :boundary_layer_height,          # z_i of a θv profile (a diagnostic)
                       :forcing_value]),                # a value of a forcing table (interpolation)
    (:Aeolus, :IO) => Set([:latest_run, :read_run_toml]),
    (:Aeolus, :Config) => Set([:sweep_members]),         # the members of a sweep file (no run)
    (:Aeolus, :InputDrivers) => Set([:read_dynamic_driver, :shift_table]),
    (:Aeolus, :StableBoundaryLayer) => Set([:sbl_read_csv, :sbl_csv_value,   # CSV readers
                                            :sbl_hockey_threshold,          # a fit to tower bins
                                            :relam_heat_flux]),             # Q of an h/L (a formula)
    (:Aeolus, :SurfaceExchange) => Set([:von_karman]),
    (:Aeolus, :MoistThermodynamics) => Set([:virtual_temperature, :PhasePartition]),
)

# Calls no example may make, whatever module the function comes from: the
# run entry points, stepping, building a case, eval and external commands.
const FORBIDDEN_CALL = r"^(run|run_config|run_configs|run_driver|eval|evalfile|include_string|time_step!|
                          step[A-Za-z0-9_]*!|run[A-Za-z0-9_]*!|[A-Za-z0-9_]*_run!|
                          build_(case|parent|model|simulation|grid|les)[A-Za-z0-9_]*)$"x

# Directories of examples/ that are not examples: shared plotting helpers
# and shared input data. They hold no configs; viz/'s scripts are checked
# like every other.
const SUPPORT_DIRS = Dict("viz" => "plotting and output-reading helpers the analysis scripts include",
                          "data" => "input data (ERA5 sample) that example configs read")

"The dotted path of `a.b.c` (an `Expr(:., …)` chain) as symbols; `nothing` if it is not a plain path."
function dotted_path(e)
    e isa Symbol && return [e]
    if e isa Expr && e.head === :. && length(e.args) == 2
        q = e.args[2]
        name = q isa QuoteNode ? q.value : q
        name isa Symbol || return nothing
        head = dotted_path(e.args[1])
        return head === nothing ? nothing : push!(head, name)
    end
    return nothing
end

"The statically known path of an `include(...)` argument (string, `joinpath(@__DIR__, \"…\")`), else `nothing`."
function include_target(arg, dir)
    arg isa String && return normpath(joinpath(dir, arg))
    if arg isa Expr && arg.head === :call && arg.args[1] === :joinpath
        parts = String[]
        for a in arg.args[2:end]
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
    return nothing
end

# the module path and imported names of one `using`/`import` argument
function import_items(a)
    a isa Expr && a.head === :as && (a = a.args[1])
    if a isa Expr && a.head === :(:)
        mod = a.args[1].args
        names = [x.args[end] for x in a.args[2:end]]
        return Tuple(mod), names
    end
    return Tuple(a.args), Symbol[]
end

"""
    check_example(path; roots = (EXC_DIR, EXC_REF_DATA)) -> (problems, includes, aeolus_names)

The rule violations of one file (`"file:line: what"`), the files it
includes (which must lie under one of `roots`) and the Aeolus names it
imports.
"""
function check_example(path; roots = (EXC_DIR, EXC_REF_DATA))
    problems = String[]; includes = String[]; used = Set{Tuple{Tuple,Symbol}}()
    line = Ref(0)
    bad(what) = push!(problems, "$(basename(dirname(path)))/$(basename(path)):$(line[]): $what")
    function walk(e)
        if e isa LineNumberNode
            line[] = e.line
            return
        end
        e isa Expr || return
        if e.head in (:using, :import)
            for a in e.args
                mod, names = import_items(a)
                if !isempty(mod) && mod[1] in FREE_PACKAGES
                    continue
                elseif !isempty(mod) && mod[1] === :Aeolus
                    allowed = get(OUTPUT_API, mod, nothing)
                    if isempty(names)
                        bad("$(e.head) $(join(mod, '.')) brings in the whole module: import the output-reading names by name")
                    elseif allowed === nothing
                        bad("$(join(mod, '.')) is not an output-reading module of OUTPUT_API")
                    else
                        for n in names
                            n in allowed ? push!(used, (mod, n)) :
                                bad("$(join(mod, '.')).$n is not in OUTPUT_API (output reading and analysis only)")
                        end
                    end
                else
                    bad("$(e.head) $(join(mod, '.')): not a package an analysis script may load")
                end
            end
            return
        end
        if e.head === :.
            p = dotted_path(e)
            if p !== nothing && p[1] === :Aeolus
                bad("qualified access $(join(p, '.')): import output-reading names by name instead")
                return
            end
        end
        if e.head === :macrocall
            m = e.args[1] isa GlobalRef ? e.args[1].name : e.args[1]     # a `command` literal is Core.@cmd
            m in (Symbol("@cmd"), Symbol("@eval"), Symbol("@everywhere")) && bad("$m: no external commands or eval")
        end
        if e.head === :call
            f = e.args[1]
            p = dotted_path(f)
            name = p === nothing ? nothing : String(p[end])
            if name == "include"
                t = length(e.args) == 2 ? include_target(e.args[2], dirname(path)) : nothing
                if t === nothing
                    bad("include with a path that is not static")
                elseif !any(r -> startswith(t, r * "/"), roots) || !isfile(t)
                    bad("include of $t: only files under examples/ or test/data/")
                else
                    push!(includes, t)
                end
            elseif name !== nothing && occursin(FORBIDDEN_CALL, name)
                bad("call of $name: examples run nothing (use `aeolus run <config>.toml`)")
            end
        end
        foreach(walk, e.args)
    end
    walk(Meta.parseall(read(path, String); filename = path))
    return problems, includes, used
end

"Every .jl under examples/ (sorted)."
function example_jl_files()
    out = String[]
    for (dir, _, fs) in walkdir(EXC_DIR), f in fs
        endswith(f, ".jl") && push!(out, joinpath(dir, f))
    end
    return sort!(out)
end

"Check every example and, transitively, what it includes: (problems, files checked, names used)."
function check_all_examples(files = example_jl_files(); roots = (EXC_DIR, EXC_REF_DATA))
    todo = copy(files); seen = Set{String}(); problems = String[]; used = Set{Tuple{Tuple,Symbol}}()
    while !isempty(todo)
        f = pop!(todo)
        f in seen && continue
        push!(seen, f)
        p, inc, u = check_example(f; roots = roots)
        append!(problems, p); union!(used, u); append!(todo, inc)
    end
    return problems, seen, used
end

@testset "examples are configuration: the rule catches what it must (synthetic scripts)" begin
    dir = mktempdir()
    roots = (dir,)
    write_case(name, text) = (p = joinpath(dir, name); write(p, text); p)
    cases = [
        "using Aeolus\nsim = 1\n" => "brings in the whole module",
        "import Aeolus\nAeolus.run_config(c)\n" => "brings in the whole module",
        "using Aeolus: NCDataset\nAeolus.Config.run_configs(\"a.toml\", ARGS)\n" => "qualified access Aeolus.Config.run_configs",
        "using Aeolus: RegularGrid\n" => "Aeolus.RegularGrid is not in OUTPUT_API",
        "using Aeolus.Simulations: BoussinesqSimulation\n" => "not an output-reading module",
        "using Aeolus.Config: run_config\n" => "Aeolus.Config.run_config is not in OUTPUT_API",
        "using FFTW\n" => "not a package an analysis script may load",
        "f(x) = x\nfor i in 1:3\n    step_rk3!(s, 0.1)\nend\n" => "call of step_rk3!",
        "case = build_case(cfg)\n" => "call of build_case",
        "run(`julia --project -m Aeolus run x.toml`)\n" => "@cmd",
        "@eval using Aeolus\n" => "@eval",
        "include(joinpath(@__DIR__, \"..\", \"..\", \"src\", \"Aeolus.jl\"))\n" => "only files under examples/ or test/data/",
        "include(p)\n" => "not static",
    ]
    for (i, (src, msg)) in enumerate(cases)
        p = write_case("case$i.jl", src)
        probs, _, _ = check_example(p; roots = roots)
        @test any(occursin(msg, x) for x in probs)
        any(occursin(msg, x) for x in probs) || @info "the rule missed" src probs
    end
    # an analysis script that keeps to the rule passes, and its include is followed
    lib = write_case("lib.jl", "using TOML\nread_res(f) = TOML.parsefile(joinpath(f, \"output\", \"results.toml\"))\n")
    ok = write_case("ok.jl", """
        using Printf
        using Statistics: mean
        using Aeolus: NCDataset
        using Aeolus.IO: latest_run, read_run_toml
        include(joinpath(@__DIR__, "lib.jl"))
        run_folder = ARGS[1]                        # a variable named like a run is fine
        ds = NCDataset(joinpath(run_folder, "output", "x.nc"))
        @printf("%.3f\\n", mean(ds["theta"][:]))
        """)
    probs, inc, used = check_example(ok; roots = roots)
    @test isempty(probs)
    @test inc == [lib]
    @test used == Set([((:Aeolus,), :NCDataset), ((:Aeolus, :IO), :latest_run), ((:Aeolus, :IO), :read_run_toml)])
    # an included file is checked by the same rules
    write(lib, "using Aeolus\n")
    probs, seen, _ = check_all_examples([ok]; roots = roots)
    @test lib in seen && any(occursin("lib.jl", x) && occursin("whole module", x) for x in probs)
end

@testset "examples are configuration: no example script builds or runs a model" begin
    files = example_jl_files()
    @test length(files) >= 40
    problems, seen, used = check_all_examples(files)
    @test isempty(problems)
    isempty(problems) || @info "examples that call simulation code (EX-e, #442)" problems
    # the reference data the validation scripts include is checked too
    @test any(startswith(f, EXC_REF_DATA) for f in seen)
    # the allow-list is live: every entry is a name of the package and some script uses it
    for (mod, names) in OUTPUT_API, n in names
        m = foldl((m, s) -> getfield(m, s), mod[2:end]; init = Aeolus)
        @test isdefined(m, n)
        @test (mod, n) in used
        @test !occursin(FORBIDDEN_CALL, String(n))
    end
    # and the imports of every script resolve (a stale name would fail when the script runs)
    for f in sort!(collect(seen))
        for e in Meta.parseall(read(f, String); filename = f).args
            e isa Expr && e.head in (:using, :import) || continue
            @test (Core.eval(Module(), e); true)
        end
    end
end

@testset "examples are configuration: every example directory has configs that aeolus check accepts" begin
    dirs = vcat(EXC_DIR, [joinpath(EXC_DIR, d) for d in readdir(EXC_DIR) if isdir(joinpath(EXC_DIR, d)) &&
                                                               !startswith(d, ".") && d != "configs"])
    @test all(isdir(joinpath(EXC_DIR, d)) for d in keys(SUPPORT_DIRS))
    nchecked = 0
    for d in dirs
        name = relpath(d, EXC_DIR)
        cdir = joinpath(d, "configs")
        if haskey(SUPPORT_DIRS, name)
            @test !isdir(cdir)                      # a support directory is not an example
            continue
        end
        tomls = isdir(cdir) ? sort!(filter(f -> endswith(f, ".toml"), readdir(cdir; join = true))) : String[]
        @test !isempty(tomls)
        isempty(tomls) && @info "an example directory without configs/*.toml" name
        for f in tomls
            out = IOBuffer()
            rc = Aeolus.main(["check", f]; io = out, err = out)
            @test rc == 0
            rc == 0 || @info "aeolus check refused a shipped config" f String(take!(out))
            nchecked += 1
        end
    end
    @test nchecked >= 50
    # no TOML lies outside a configs/ directory but the examples' environment
    stray = String[]
    for (dir, _, fs) in walkdir(EXC_DIR), f in fs
        endswith(f, ".toml") || continue
        p = joinpath(dir, f)
        (basename(dir) == "configs" || p == joinpath(EXC_DIR, "Project.toml")) || push!(stray, p)
    end
    @test isempty(stray)
end
