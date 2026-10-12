# One section per exclusive slot (ONE-m, #476): a selector key names the
# provider, the section holds that provider's options and no other's, and the
# resolved config shows the chosen provider with its options. The model's
# slots ([core] model, [timestep] scheme, [sgs] model, [surface] model,
# [microphysics] scheme, [advection] scheme) and the urban site's ([pressure]
# solver, [chemistry] scheme, ...) share the mechanism; a slot section that
# both have offers each domain the providers of its kind.

using Test
using Aeolus
using Aeolus.Config
import Aeolus.Config as C
import Aeolus.Urban as SU
using Aeolus.SimulationDriver: SimulationDriverType, simulation
using Aeolus.Components: Components

const SL_EXAMPLES = normpath(joinpath(@__DIR__, "..", "examples"))
sl_error(f) = try
    f(); error("expected a ConfigError")
catch e
    e isa ConfigError ? e : rethrow()
end
sl_keys(e) = Dict(p.key => p.message for p in e.problems)
sl_section(cfg, n) = getfield(getproperty(cfg, Symbol(n)), :values)

@testset "slot sections: the selector names the provider, the section holds its options only" begin
    # the resolved config: the chosen provider and its options, defaults included, nothing else
    cfg = load_run_config(IOBuffer("[sgs]\nmodel = \"amd\"\n"))
    @test sl_section(cfg, "sgs") == Dict("model" => :amd, "c_s" => 0.3, "pr_t" => 1 / 3, "stress" => :laplacian)
    cfg = load_run_config(IOBuffer("[sgs]\nmodel = \"tke\"\ne_min = 1e-4\n"))
    @test Set(keys(sl_section(cfg, "sgs"))) == Set(["model", "e_min", "surface_factor", "wall_cap", "wall_cap_top",
                                                     "surface_value", "stress"])
    @test cfg.sgs.e_min == 1e-4
    @test occursin("[domain.sgs]\ne_min = 0.0001\nmodel = \"tke\"\nstress = \"laplacian\"\nsurface_factor = 3.9\n" *
                   "surface_value = false\nwall_cap = false\nwall_cap_top = true\n",
                   config_text(cfg))
    @test load_run_config(IOBuffer(config_text(cfg))) == cfg                  # the resolved text reloads to itself
    # every model slot has its default: the bare config
    bare = load_run_config(IOBuffer(""))
    @test (bare.core.model, bare.timestep.scheme, bare.sgs.model, bare.surface.model, bare.microphysics.scheme,
           bare.advection.scheme) == (:boussinesq, :rk3, :smagorinsky, :none, :none, :centred)
    @test sl_section(bare, "surface") == Dict("model" => :none)               # no provider options
    # an option of another provider names its owners and the chosen provider's options
    e = sl_error(() -> load_run_config(IOBuffer("[sgs]\nmodel = \"tke\"\nc_s = 0.2\n")))
    m = sl_keys(e)["sgs.c_s"]
    @test occursin("an option of model = \"smagorinsky\" or \"vreman\" or \"sigma\" or \"amd\"", m) &&
          occursin("not of model = \"tke\"", m) && occursin("e_min", m)
    e = sl_error(() -> load_run_config(IOBuffer("[surface]\ntemperature = 290.0\n")))      # the default is none
    m = sl_keys(e)["surface.temperature"]                  # the slab's (a number) and the surface layer's (a table)
    @test occursin("\"slab\"", m) && occursin("\"most\"", m) && occursin("not of model = \"none\" (it has no options)", m)
    e = sl_error(() -> load_run_config(IOBuffer("[sgs]\nmodel = \"tke\"\ne_mn = 1.0\n")))
    @test occursin("did you mean \"e_min\"", sl_keys(e)["sgs.e_mn"])
    e = sl_error(() -> load_run_config(IOBuffer("[sgs]\nmodel = \"smagorinksy\"\n")))
    @test occursin("did you mean \"smagorinsky\"", sl_keys(e)["sgs.model"])
    # one key per slot: the old forms are gone, with what replaced them
    @test haskey(sl_keys(sl_error(() -> load_run_config(IOBuffer("[sgs]\nclosure = \"amd\"\n")))), "sgs.closure")
    @test haskey(sl_keys(sl_error(() -> load_run_config(IOBuffer("[dynamics]\ncore = \"anelastic\"\n")))), "dynamics.core")
    @test haskey(sl_keys(sl_error(() -> load_run_config(IOBuffer("[time]\nscheme = \"euler\"\n")))), "time.scheme")
    @test haskey(sl_keys(sl_error(() -> load_run_config(IOBuffer("[surface_layer]\nenabled = true\n")))), "surface_layer")
end

@testset "slot sections: a selector set above the file replaces the provider" begin
    p = joinpath(mktempdir(), "c.toml")
    write(p, "[sgs]\nmodel = \"tke\"\ne_min = 1e-4\nstress = \"symmetric\"\n")
    # --set another provider: the file's options are dropped, the new provider starts from its defaults
    # (a stress = "symmetric" set for tke is not silently the vreman closure's)
    cfg = load_run_config(p; overrides = ["sgs.model=\"vreman\""])
    @test sl_section(cfg, "sgs") == Dict("model" => :vreman, "c_s" => 0.17, "pr_t" => 1 / 3, "stability" => :none,
                                         "stress" => :laplacian)
    @test config_origin(cfg, "sgs.stress") == "default" && config_origin(cfg, "sgs.model") == "--set"
    # with its own options set at the same level
    cfg = load_run_config(p; overrides = ["sgs.model=\"vreman\"", "sgs.stress=\"symmetric\"", "sgs.c_s=0.1"])
    @test (cfg.sgs.stress, cfg.sgs.c_s) == (:symmetric, 0.1)
    # --set the provider the file names: nothing is dropped
    cfg = load_run_config(p; overrides = ["sgs.model=\"tke\""])
    @test (cfg.sgs.e_min, cfg.sgs.stress) == (1e-4, :symmetric)
    # a file that leaves the selector at its default: an override naming the default keeps its options
    q = joinpath(mktempdir(), "d.toml"); write(q, "[sgs]\nc_s = 0.1\n")
    @test load_run_config(q; overrides = ["sgs.model=\"smagorinsky\""]).sgs.c_s == 0.1
    @test load_run_config(q; overrides = ["sgs.model=\"sigma\""]).sgs.c_s == 1.35
    # an override of an option of a provider the file does not select is still an error
    e = sl_error(() -> load_run_config(p; overrides = ["sgs.c_s=0.2"]))
    @test haskey(sl_keys(e), "sgs.c_s")
    # with(): the same rule, from Julia
    c2 = with(load_run_config(p), "sgs.model" => :sigma)
    @test sl_section(c2, "sgs") == Dict("model" => :sigma, "c_s" => 1.35, "pr_t" => 1 / 3, "stability" => :none,
                                        "stress" => :laplacian)
    @test with(load_run_config(p), "sgs.model" => :tke).sgs.e_min == 1e-4        # the same provider: unchanged
    @test with(c2, "sgs.c_s" => 1.5).sgs.c_s == 1.5
    @test_throws ConfigError with(c2, "sgs.e_min" => 1e-3)
    @test_throws ConfigError with(c2, "sgs.model" => :lasd)
    # each [[domain]] block selects its own provider, with its own options
    two = """
        [time]
        dt = 1.0
        [[domain]]
        id = "outer"
        [domain.sgs]
        model = "tke"
        e_min = 1e-4
        [domain.grid]
        size = [8, 8, 8]
        [[domain]]
        id = "inner"
        parent = "outer"
        origin = [1200.0, 1200.0]
        refinement = [8, 8, 8]
        [domain.grid]
        size = [8, 8, 8]
        extent = [400.0, 400.0, 200.0]
        [domain.sgs]
        model = "amd"
        """
    ds = config_domains(load_run_config(IOBuffer(two)))
    @test [d.sgs.model for d in ds] == [:tke, :amd]
    @test ds[1].sgs.e_min == 1e-4 && !haskey(getfield(ds[2].sgs, :values), "e_min")
    # nothing is inherited: an option of another domain's provider is not this one's
    e = sl_error(() -> load_run_config(IOBuffer(replace(two, "model = \"amd\"" => "c_s = 0.3\ne_min = 1e-3"))))
    @test haskey(sl_keys(e), "domain[2].sgs.e_min")
    # an override that changes one domain's provider drops that domain's options of the old one
    ds = config_domains(load_run_config(IOBuffer(two); overrides = ["domain.outer.sgs.model=\"vreman\""]))
    @test [d.sgs.model for d in ds] == [:vreman, :amd] && !haskey(getfield(ds[1].sgs, :values), "e_min")
end

@testset "slot sections: the providers are those of the domain's kind" begin
    smoke = joinpath(SL_EXAMPLES, "urban", "berlin_smoke.toml")
    # [sgs] on an urban site offers the site's closures, on a model domain the model's
    u = load_run_config(smoke)
    @test u.sgs.model === :tke && u.sgs.e_min == 1e-4 && u.sgs.wall_cap               # the site's tke and its options
    @test load_run_config(IOBuffer("[sgs]\nmodel = \"tke\"\n")).sgs.e_min == 1e-3     # the model's tke
    e = sl_error(() -> load_run_config(smoke; overrides = ["sgs.model=\"none\""]))
    @test occursin("(a provider on a domain of the model's core only)", sl_keys(e)["sgs.model"])
    # the urban vreman has its own options (its c_s defaults to the site's [urban] c_s)
    @test load_run_config(smoke; overrides = ["sgs.model=\"vreman\""]).sgs.c_s === nothing
    # an urban slot's section has no provider on a model domain
    e = sl_error(() -> load_run_config(IOBuffer("[pressure]\nsolver = \"multigrid\"\n")))
    @test occursin("no provider on a domain of the model's core", sl_keys(e)["pressure.solver"]) &&
          occursin("urban site", sl_keys(e)["pressure.solver"])
    @test isempty(sl_section(load_run_config(IOBuffer("")), "pressure"))
    # the urban site's selection is what its slot sections name
    for (ov, C) in (("pressure.solver=\"multigrid\"", SU.PressureMultigrid), ("chemistry.scheme=\"none\"", SU.NoChem),
                    ("momentum_advection.scheme=\"weno_z\"", SU.MomentumWENOZ), ("sgs.model=\"smagorinsky\"", SU.SGSSmagorinsky))
        cfg = load_run_config(smoke; overrides = [ov])
        so = SU.site_options(cfg)
        @test C in SU.selected_components(so)
        others = [D for D in SU.URBAN_COMPONENTS if SU.slot(D) === SU.slot(C) && D !== C]
        @test !any(D -> D in SU.selected_components(so), others)
        @test so.components[SU.component_name(C)] == SU.default_options(C)     # its options: the defaults
    end
    # one provider per urban slot, always: the default selection of a bare site
    cfg = load_run_config(IOBuffer("[urban]\nenabled = true\n[time]\ndt = 1.0\nduration = \"none\"\nnsteps = 1\n" *
                                   "[grid]\nhalo = [3, 3, 3]\n"))
    on = [C for C in SU.URBAN_COMPONENTS if SU.is_selected(cfg, C)]
    @test Set(on) == Set(C for C in SU.URBAN_DEFAULTS if SU.slot(C) !== nothing)
    # on a model domain [sgs] model = "none" builds no SGS component
    @test Aeolus.SimulationDriver.enabled_components(load_run_config(IOBuffer("[sgs]\nmodel = \"none\"\n"))) ==
          Any[Components.CentredAdvection]
end

@testset "slot sections: the option reference lists each provider's options" begin
    md = option_reference("model"; format = :markdown)
    @test occursin("## `[pressure]`", md) && occursin("### `[pressure]` solver = `\"multigrid\"`", md)
    @test occursin("### `[sgs]` model = `\"tke\"` (a domain of the model's core)", md) &&
          occursin("### `[sgs]` model = `\"tke\"` (an urban site ([urban] enabled = true))", md)
    txt = option_reference("model"; section = "surface")
    @test occursin("[surface] model = \"most\"", txt) && occursin("z0m", txt) && occursin("[surface] model = \"slab\"", txt)
    sch = driver_schema("model")
    s = only(x for x in sch.sections if x.name == "timestep")
    @test Set(p.name for p in C.providers(s, :model)) == Set(["rk3", "euler"]) &&
          Set(p.name for p in C.providers(s, :urban)) == Set(["fixed", "adaptive", "imex"])
    # a schema whose slot has no default on a kind it serves is refused
    bad = slot_section("t476"; key = "model") do s
        provider!(s, "a")
    end
    @test_throws ArgumentError ConfigSchema("t476"; sections = [bad])
end

@testset "slot sections: the converted configs mean what the old ones did" begin
    # bit for bit on the Boussinesq core (the simulation digests and the
    # Berlin fixture are the full check): a [sgs] c_s default per closure
    # replaces "none" → the closure's own default, so the closures are the same
    for (m, cs) in ((:smagorinsky, 0.17), (:vreman, 0.17), (:sigma, 1.35), (:amd, 0.3))
        cfg = load_run_config(IOBuffer("[sgs]\nmodel = \"$m\"\n"))
        @test cfg.sgs.c_s == cs
        c = Components.build_closure(cfg.sgs, Aeolus.RegularGrid(; size = (8, 8, 8), extent = (1.0, 1.0, 1.0)))
        @test getfield(c, 1) == cs                  # C_s, C_σ, C: the closure's coefficient
    end
    # the Monin–Obukhov surface layer is [surface] model = "most"
    g = load_run_config(joinpath(SL_EXAMPLES, "sbl", "gabls1.toml"))
    @test g.surface.model === :most && g.surface.z0m == 0.1 && Components.enabled(Components.SurfaceLayer, g.surface)
    @test Components.SurfaceLayer in Aeolus.SimulationDriver.enabled_components(g)
end
