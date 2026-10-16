# The generated references (CFG-i, #364): the option reference of the model
# and of every legacy case, and the input-format specifications of the driver schemas, both
# generated from the schemas (docs/make.jl writes them to
# docs/src/reference/ before makedocs; nothing is committed).
#
# Acceptance tests of the issue covered here:
#   * the option reference has one table row per schema option, for the
#     model (the urban site's sections included) and every legacy case,
#     counted against the schema;
#   * every option has a doc, every float option has units ("1" for a
#     dimensionless one), every cross-field check is described in words;
#   * the input-format pages list every variable of the static and dynamic
#     DriverSchemas, counted the same way, and their global attributes;
#   * ncdump_header renders a file's header as `ncdump -h` does (CDL).

using Test
using Aeolus
using Aeolus.Config
using Aeolus.InputDrivers
import Aeolus.Config as C
import NCDatasets
import Markdown

const REF_DRIVERS = ["model"]
const REF_INPUT = normpath(joinpath(@__DIR__, "..", "examples", "urban", "input"))

"The keys of the table rows of a Markdown option reference, per `[table]` heading."
function md_rows(md::AbstractString)
    rows = Dict{String,Vector{String}}(); order = String[]; cur = nothing
    for l in split(md, '\n')
        m = match(r"^#+ `\[(.+)\]`$", l)
        # a provider block of a slot section (ONE-m, #476): ### `[sgs]` model = `"tke"` (kinds)
        p = match(r"^#+ `\[(.+?)\]` (\w+) = `\"(.+?)\"`(.*)$", l)
        if m !== nothing || p !== nothing
            cur = m !== nothing ? String(m[1]) : "$(p[1]).$(p[2])=$(p[3])$(p[4])"
            rows[cur] = String[]; push!(order, cur)
        elseif cur !== nothing && startswith(l, "| `")
            push!(rows[cur], String(match(r"^\| `([^`]+)` \|", l)[1]))
        end
    end
    return rows, order
end

"Whether an option type holds floats (scalar, tuple, vector, Optional, OneOf)."
isfloat(T) = T === Float64 || (T isa C.OneOf && any(isfloat, T.alts)) ||
             (T isa Type && T <: C.Optional && T !== C.Optional && isfloat(T.parameters[1])) ||
             (T isa Type && T <: Tuple && isconcretetype(T) && fieldcount(T) > 0 && fieldtype(T, 1) === Float64) ||
             (T isa Type && T <: Vector && isfloat(eltype(T)))

"The documentation problems of a schema: options without doc, float options without units, checks without words."
function doc_problems(sch::ConfigSchema)
    out = String[]
    for (name, s) in reference_sections(sch), o in s.options
        o.type isa AbstractString && continue        # a reference_option: the coupling documents it
        isempty(strip(o.doc)) && push!(out, "$(sch.name): $name.$(o.key) has no doc")
        isfloat(o.type) && isempty(strip(o.units)) && push!(out, "$(sch.name): $name.$(o.key) is a float without units")
    end
    for (doc, _) in sch.checks
        isempty(strip(doc)) && push!(out, "$(sch.name): a cross-field check without a description")
    end
    return out
end

@testset "option reference: one Markdown table row per option, the model and every legacy case" begin
    for d in REF_DRIVERS
        sch = driver_schema(d)
        md = option_reference(d; format = :markdown)
        rows, order = md_rows(md)
        refs = reference_sections(sch)
        @test order == first.(refs)                                   # every table, in order
        for (name, s) in refs
            @test rows[name] == [o.key for o in s.options]
        end
        # the count the issue asks for: rows of the schema's sections = its options
        # (a slot section: its selector and shared options, then one block per provider)
        @test sum(length(rows[s.name]) for s in sch.sections) ==
              sum(length(s.options) + (s.selector === nothing ? 0 : 1) for s in sch.sections)
        @test sum(length(rows[provider_block_name(s, p)]) for s in sch.sections if s.selector !== nothing
                  for p in s.selector.providers) ==
              sum(length(p.options) for s in sch.sections if s.selector !== nothing for p in s.selector.providers)
        # every row is a well-formed six-cell row (a `|` in a cell is escaped)
        for t in filter(x -> x isa Markdown.Table, Markdown.parse(md).content)
            @test all(r -> length(r) == 6, t.rows)
        end
        # the checks: one bullet per described line
        lines = C._check_lines(sch)
        isempty(lines) || @test count(l -> startswith(l, "- "), split(split(md, "## Cross-field checks")[end], '\n')) ==
                                length(lines)
    end
    # the urban site (the Berlin configs, ONE-e #454): one row per option of a
    # resolved RunConfig (its sections hold every option)
    sch = driver_schema("model")
    cfg = load_run_config(joinpath(@__DIR__, "..", "examples", "urban", "berlin_smoke_site.toml"))   # (one domain)
    rows, _ = md_rows(option_reference("model"; format = :markdown))
    # a slot section (ONE-m, #476): its selector, and the options of the provider it selects
    function documented(s)
        n = length(rows[s.name])
        s.selector === nothing && return n
        vals = getfield(getproperty(cfg, Symbol(s.name)), :values)
        haskey(vals, s.selector.key) || return 0
        p = only(p for p in s.selector.providers if p.name == String(vals[s.selector.key]) && :urban in p.kinds)
        return n + length(rows[provider_block_name(s, p)])
    end
    @test sum(documented(s) for s in sch.sections) ==
          sum(length(getfield(getproperty(cfg, Symbol(s.name)), :values)) for s in sch.sections)
    # an additive component's section, a provider's options (aq_chem's default_options
    # under [chemistry] scheme = "aq_chem"); the nest parent is a [[domain]] of its own
    # (NEST-p3, #483), no [nesting.parent] table
    @test "pollutant" in keys(rows) && "enabled" in rows["pollutant"] && "stacks" in rows["pollutant"]
    @test "chemistry.scheme=aq_chem" in keys(rows) && "nox_street" in rows["chemistry.scheme=aq_chem"] &&
          rows["chemistry"] == ["scheme"]
    @test occursin("### `[chemistry]` scheme = `\"aq_chem\"`", option_reference("model"; format = :markdown))
    @test !("nesting.parent" in keys(rows)) && !("mode" in rows["nesting"])
    # a section asked for alone leaves the checks out
    one = option_reference("model"; section = "nesting", format = :markdown)
    @test occursin("`[nesting]`", one) && !occursin("Cross-field checks", one)
end

"Every `Markdown.Italic` / `Bold` node under `x` (the option docs are plain prose)."
emphasis(x) = x isa Union{Markdown.Italic,Markdown.Bold} ? [x] :
              x isa AbstractVector ? reduce(vcat, map(emphasis, x); init = Any[]) :
              hasproperty(x, :content) ? emphasis(x.content) :
              hasproperty(x, :rows) ? emphasis(reduce(vcat, x.rows; init = Any[])) : Any[]

@testset "option reference: identifiers in prose are not read as emphasis" begin
    @test C._md_prose("see set_fftw_policy! and AEOLUS_LOG, `a_b*c`, 2*3*4") ==
          "see set\\_fftw\\_policy! and AEOLUS\\_LOG, `a_b*c`, 2\\*3\\*4"
    for d in REF_DRIVERS
        md = Markdown.parse(option_reference(d; format = :markdown))
        @test isempty(emphasis(md.content))
    end
    # not vacuous: the same prose unescaped is read as emphasis
    @test !isempty(emphasis(Markdown.parse("| doc |\n| :-- |\n| default: AEOLUS_ALLOW_UNCOMMITTED=1 |\n").content))
end

@testset "every option documented: a doc, units on every float option, checks in words" begin
    for d in REF_DRIVERS
        p = doc_problems(driver_schema(d))
        @test isempty(p)
        isempty(p) || @info "undocumented options" p
    end
    # the check is not vacuous: an option without units or doc, a bare check
    s = section("t364") do s
        option!(s, "speed", Float64; default = 1.0, doc = "a speed")
        option!(s, "n", Int; default = 1)
    end
    sch = ConfigSchema("t364doc"; sections = [s], checks = Any[cfg -> ConfigError[]])
    @test sort(doc_problems(sch)) == ["t364doc: a cross-field check without a description",
                                      "t364doc: t364.n has no doc", "t364doc: t364.speed is a float without units"]
end

@testset "undocumented_keys: the Berlin configs, parent and site domains, document every key" begin
    for f in readdir(joinpath(@__DIR__, "..", "examples", "urban"); join = true)
        endswith(f, ".toml") || continue
        cfg = load_run_config(f)
        @test isempty(undocumented_keys(cfg))
    end
end

@testset "input-format reference: one row per variable of each driver schema" begin
    for (schema, file) in ((STATIC_SCHEMA, "berlin_smoke_static.nc"),
                           (STATIC_TILES_SCHEMA, "berlin_smoke_parent_static.nc"),
                           (DYNAMIC_SCHEMA, "berlin_20190629_dynamic.nc"))
        path = joinpath(REF_INPUT, file)
        md = format_reference(schema; example = path)
        vars = split(split(md, "### Variables")[2], "### Global attributes")[1]
        names = [String(m[1]) for m in eachmatch(r"^\| `([^`]+)` \|"m, vars)]
        @test names == [v.name for v in schema.variables]          # the count, and the order
        attrs = split(split(md, "### Global attributes")[2], "### Example")[1]
        for a in vcat(collect(DRIVER_ATTRIBUTES), [strip(x) for (k, _) in schema.attributes for x in split(k, ',')])
            @test occursin("`$a`", attrs)
        end
        @test occursin("aeolus_schema_version = $(DRIVER_SCHEMA_VERSION)", md)
        @test occursin("aeolus_driver = \"$(schema.kind)\"", md) && occursin("aeolus_form = \"$(schema.form)\"", md)
        # the example is the file's own header: every dimension and variable of the file
        NCDatasets.NCDataset(path) do ds
            for d in keys(ds.dim)
                @test occursin("\n\t$d = $(ds.dim[d]) ;", md)
            end
            for v in keys(ds)
                @test occursin(Regex("\\n\\t\\w+ $(v)( ;|\\()"), md)
            end
        end
        @test isempty(check_driver(path, schema))                  # the example is a valid driver
    end
end

@testset "ncdump_header: CDL as ncdump -h prints it" begin
    p = joinpath(mktempdir(), "tiny.nc")
    NCDatasets.NCDataset(p, "c") do ds
        NCDatasets.defDim(ds, "x", 3)
        NCDatasets.defDim(ds, "time", Inf)
        v = NCDatasets.defVar(ds, "t", Float32, ("x", "time"); attrib = ["units" => "K", "valid_range" => Float32[0, 400]])
        v[:, 1:2] = zeros(Float32, 3, 2)
        NCDatasets.defVar(ds, "flag", UInt8, ("x",); attrib = ["flag_values" => UInt8[1, 2]])
        NCDatasets.defVar(ds, "n", Int32, ())
        ds.attrib["title"] = "say \"hi\""
        ds.attrib["dx"] = 2.5
    end
    @test ncdump_header(p) == """
        netcdf tiny {
        dimensions:
        \tx = 3 ;
        \ttime = UNLIMITED ; // (2 currently)
        variables:
        \tfloat t(time, x) ;
        \t\tt:units = "K" ;
        \t\tt:valid_range = 0.0f, 400.0f ;
        \tubyte flag(x) ;
        \t\tflag:flag_values = 1UB, 2UB ;
        \tint n ;

        // global attributes:
        \t\t:title = "say \\"hi\\"" ;
        \t\t:dx = 2.5 ;
        }
        """
    @test occursin(":title = \"say …", ncdump_header(p; max_attr = 4))
end
