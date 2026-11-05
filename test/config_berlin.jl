# The Berlin case configured from TOML (CFG-c, #358): the urban site of the
# model (src/components/urban/site/config_schema.jl, EX-a #438, ONE-e #454) and the
# configs examples/urban/berlin_*.toml, at the level of the configuration
# (nothing is built or run here; test/urban_run.jl runs a small synthetic
# city, validation/urban_checks.jl the Berlin configs):
#
#   * one source of truth: the site's options map one to one onto the
#     schema's options, and their defaults are the schema's;
#   * the analysis buffer of the observation diagnostics (#268);
#   * errors: an unknown provider, a parent grid of 2 integers, a parent
#     corner that does not centre the site, an incompatible pair, an unknown
#     per-coupling option.

using Test
using TOML
using Aeolus
using Aeolus.Config

const BCC = Aeolus.Urban
include(joinpath(@__DIR__, "support", "berlin_configs.jl"))

"The ConfigError `f()` throws (fails the test if it does not throw one)."
function berlin_error(f)
    try
        f()
    catch e
        e isa ConfigError && return e
        rethrow()
    end
    error("expected a ConfigError")
end
errtext(e) = sprint(showerror, e)

@testset "Config berlin: one source of truth for the defaults" begin
    sch = driver_schema("model")
    sec(n) = sch.sections[findfirst(x -> x.name == n, sch.sections)]
    # every field of the site's options is made of the model's options, each used once
    @test collect(keys(BCC.SITE_FIELDS)) == collect(keys(BCC.site_options(BCC.default_run_config())))
    used = String[]
    # the additive components' sections, and the urban slot sections with their
    # urban providers (ONE-m, #476: one section per slot, a selector key)
    names = [String(BCC.component_name(C)) for C in BCC.URBAN_COMPONENTS if BCC.slot(C) === nothing]
    slots = [BCC.URBAN_SLOT_SECTIONS[sl] for sl in BCC.URBAN_SLOTS]
    uprov(n) = [p for p in sec(n).selector.providers if :urban in p.kinds]
    provider_options = ["$n.$(p.name).$(x.key)" for (n, _) in slots for p in uprov(n) for x in p.options]
    unknown = String[]
    for keys_ in values(BCC.SITE_FIELDS), k in keys_
        s_, o = split(k, '.')
        if s_ == "<slot>"            # every urban slot section's selector (components)
            append!(used, ["$n.$key" for (n, key) in slots])
        elseif s_ == "*"             # every additive section's enabled; every component's options (components)
            for n in names, x in sec(n).options
                (o == "enabled") == (x.key == "enabled") && push!(used, "$n.$(x.key)")
            end
            o == "enabled" || append!(used, provider_options)
        else
            any(x -> x.key == o, sec(s_).options) || push!(unknown, k)
            push!(used, k)
        end
    end
    @test isempty(unknown)
    @test allunique(used)
    # every option of the urban site's sections makes a field ([urban] enabled: the site itself)
    site = vcat(["urban.$(o.key)" for o in sec("urban").options if o.key != "enabled"],
                ["$n.$(o.key)" for n in names for o in sec(n).options], ["$n.$key" for (n, key) in slots],
                provider_options)
    @test issubset(site, used)
    # every urban slot has one section, whose urban providers are the slot's components
    @test length(slots) == length(BCC.URBAN_SLOTS) && allunique(first.(slots))
    function slot_ok(sl)
        n, key = BCC.URBAN_SLOT_SECTIONS[sl]
        return sec(n).selector.key == key &&
               Set(p.name for p in uprov(n)) == Set(String(BCC.provider(C)) for C in BCC.URBAN_COMPONENTS if BCC.slot(C) === sl) &&
               sec(n).selector.defaults[:urban] == String(BCC.provider(only(C for C in BCC.URBAN_DEFAULTS if BCC.slot(C) === sl)))
    end
    @test all(slot_ok, BCC.URBAN_SLOTS)
    # berlin_summer.toml is the reference copy: every option of the urban site
    # and every component's option is written out (in the site's [[domain]] block)
    f = TOML.parsefile(berlin_toml("summer"))["domain"][site_domain(berlin_toml("summer"))[1]]
    missing_ = String[]
    for o in sec("urban").options
        haskey(f["urban"], o.key) || push!(missing_, "urban.$(o.key)")
    end
    for n in names, o in sec(n).options
        haskey(get(f, n, Dict()), o.key) || push!(missing_, "$n.$(o.key)")
    end
    for (n, key) in slots          # the selector and every option of the provider it selects
        t = get(f, n, Dict())
        haskey(t, key) || (push!(missing_, "$n.$key"); continue)
        for o in only(p for p in uprov(n) if p.name == t[key]).options
            haskey(t, o.key) || push!(missing_, "$n.$(o.key)")
        end
    end
    @test isempty(missing_)
    # the struct defaults are the schema's
    c = BCC.site_options(BCC.default_run_config())
    dflt = BCC.default_run_config()
    @test c.size == dflt.grid.size && c.ν === dflt.dynamics.nu && c.θ_top === :initial_gradient
    @test c.cadence == Dict(:radiation => 600.0, :soil => 30.0, :output => 1800.0)
    @test c.name === :urban && Set(keys(c.components)) == Set(BCC.component_name.(BCC.URBAN_DEFAULTS))
    @test all(C -> isequal(c.components[BCC.component_name(C)], BCC.default_options(C)), BCC.URBAN_DEFAULTS)   # isequal: NaN defaults
    # the generic site points at no input data of an example (EX-a, #438)
    @test c.sounding === nothing && c.static === nothing
    # a key that does not exist is an error, not silently ignored
    @test_throws ConfigError load_run_config(berlin_toml("smoke"); overrides = ["domain.site.urban.nstep=3"])
    # a component's options change with an override of its section in the
    # site's block; the components are not fields of the site's options (ONE-e, #454)
    sm = site_config(berlin_toml("smoke"); overrides = ["domain.site.pollutant.tau=60.0"])
    @test sm.components[:pollutant].tau == 60.0
    # every urban component is off until its section enables it: a model
    # config without [urban] is no urban site, and selects none of them; the
    # urban slot sections hold nothing on it (no provider on a model domain)
    bare = load_run_config(IOBuffer(""))
    @test !BCC.urban_enabled(bare) && !any(n -> bare[n].enabled, names)
    @test !any(C -> BCC.is_selected(bare, C), BCC.URBAN_COMPONENTS)
    @test all(n -> isempty(getfield(bare[n], :values)),
              [n for (n, _) in slots if !(n in ("core", "timestep", "sgs", "microphysics"))])
end

@testset "Config berlin: the analysis buffer of the observation diagnostics (#268)" begin
    @test BCC.site_options(BCC.default_run_config()).analysis_buffer == BCC.ANALYSIS_BUFFER_DEFAULT == 140.0
    sm = site_config(berlin_toml("smoke"))
    m = RegularGrid(; size = sm.size, extent = sm.extent, halo = sm.halo)
    A = BCC.analysis_columns(sm, m)            # 31.25 m = 2 cells of 15.6 m from each nest face
    @test sm.analysis_buffer == 31.25 && size(A) == (16, 16) && count(A) == 12 * 12
    @test all(A[3:14, 3:14]) && !any(A[1:2, :]) && !any(A[:, 15:16])
    # the periodic box has no inflow face: every column counts
    pb = site_config(berlin_toml("smoke_site"))
    @test all(BCC.analysis_columns(pb, m))
end

@testset "Config berlin: errors name the key and the allowed values" begin
    smoke = berlin_toml("smoke")
    # (the overrides address the site's [[domain]] block; a problem of it is domain[2]'s)
    err(ovr...) = berlin_error(() -> load_run_config(smoke; overrides = [site_key(smoke, o) for o in ovr]))
    key(e) = replace(only(e.problems).key, "domain[2]." => "")
    # a slot's selector names a provider of the urban site (ONE-m, #476); the
    # model's own providers of a shared section are not the site's
    e = err("microphysics.scheme=\"none\"")
    @test key(e) == "microphysics.scheme" && occursin("\"kessler\"", errtext(e)) &&
          occursin("urban site", errtext(e))
    e = err("sgs.model=\"dynamic_smagorinsky\"")
    @test key(e) == "sgs.model" && occursin("\"dynamic\"", errtext(e)) &&
          occursin("model's core only", errtext(e))
    # an option of another provider than the selected one names its owners;
    # the old per-provider sections are gone
    e = err("sgs.diapycnal=0.1")
    @test key(e) == "sgs.diapycnal" && occursin("not of model = \"tke\"", errtext(e)) &&
          occursin("\"smagorinsky\"", errtext(e)) && occursin("e_min", errtext(e))
    e = err("sgs_smagorinsky.enabled=true")
    @test occursin("sgs_smagorinsky", key(e)) && occursin("unknown table", errtext(e))
    # the parent [[domain]] (NEST-p3, #483): its own options, the nest's rules
    e = err("domain.parent.grid.size=[16, 16]")
    @test occursin("domain[1].grid.size", errtext(e)) && occursin("3", errtext(e))
    e = err("domain.parent.grid.origin=[0.0, 0.0]")
    @test any(p -> p.key == "domain[1].grid.origin" && occursin("its own corner", p.message), e.problems)
    e = err("nesting.enabled=false")
    @test any(p -> p.key == "domain[2].nesting.enabled" && occursin("parent_driver = \"aeolus\"", p.message), e.problems)
    e = err("domain.site.refinement=[4, 4, 4]")
    @test any(p -> occursin("grid.size", p.key) && occursin("refined by", p.message), e.problems)
    e = err("domain.parent.surface.q_surface={kind = \"series\", t = [1.0, 0.0], value = [0.0, 1.0]}",
            "domain.parent.surface.model=\"prescribed\"")
    @test occursin("strictly increasing", errtext(e))
    # a declared incompatibility, with its reason
    e = err("core.model=\"anelastic\"")
    reason = BCC.incompatible_reason(BCC.component_type(:core_anelastic))
    @test !isempty(reason) && any(p -> endswith(p.key, "pressure.solver") &&
                                       occursin("[core] model = \"anelastic\" is incompatible with [pressure] solver = \"cosine_fft\"", p.message) &&
                                       occursin(reason, p.message), e.problems)
    # an unknown option of a component lists its options
    e = err("geostrophic.ug=2.0")
    @test occursin("geostrophic.ug", errtext(e)) && occursin("did you mean \"u_g\"", errtext(e))
    e = err("geostrophic.u_g=\"fast\"")
    @test occursin("geostrophic.u_g", errtext(e))
    e = err("geostrofic.u_g=1.0")
    @test occursin("geostrofic", errtext(e)) && occursin("geostrophic", errtext(e))
    e = err("nesting.z_anterp=\"roof\"")
    @test occursin("z_anterp", errtext(e)) && occursin("canopy", errtext(e))
    # five independent problems are all reported
    e = err("microphysics.qv_max=\"x\"", "grid.size=[16, 16]", "time.dt=-1.0",
            "domain.parent.grid.size=[16, 16]", "urban.lateral=\"closed\"")
    @test length(e.problems) == 5
end
