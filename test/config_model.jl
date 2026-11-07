# The one config schema (ONE-b, #451): src/config/{schema,load,resolve}.jl,
# src/config/simulation_driver.jl, src/components/{interface,pipeline}.jl.
#
#   * a bare config is a minimal dry LES: the defaults resolve to it, the
#     advection and the Smagorinsky SGS are its only components, and it runs:
# a fluid at rest stays at rest at the adaptive limit dt_max;
#     perturbed, θ is conserved between its no-flux walls, the SGS acts and
#     ∇·u stays at round-off;
#   * the component sections are generated from the component schemas;
#   * [[domain]]: every domain carries its complete configuration (NEST-p3: no
#     run-level module sections), --set domain.<id>.… addresses one; the
#     resolved config lists every component's options per domain and reloads
#     to itself; a single-domain config may write its domain keys at the top
#     level, which resolves to the same config as one [[domain]] block;
#   * validation: per-domain problems and cross-field checks keyed by
#     domain[i], duplicate ids, unknown block keys, [checks], slots,
#     requirements and incompatibilities named by their sections;
#   * several domains are a nest (ONE-f, #455): the build couples them.

using Test
using Aeolus
using Aeolus.Config
using Aeolus.Components
const CMC = Aeolus.Components
using Aeolus.SimulationDriver: SimulationDriverType, COMPONENTS, DOMAIN_SECTIONS, enabled_components, simulation
using Aeolus.Fields: interior_view
using Aeolus.Simulations: max_divergence, kinetic_energy

cm_error(f) = try
    f(); error("expected a ConfigError")
catch e
    e isa ConfigError ? e : rethrow()
end
cm_file(text) = (p = joinpath(mktempdir(), "c.toml"); write(p, text); p)
cm_run(cfg) = run_config(cfg; root = mktempdir(), console = false, repo = mktempdir())

@testset "a bare config is a minimal dry LES" begin
    cfg = load_run_config(IOBuffer(""))
    @test (Config.config_case(cfg) == "model" && cfg.run.name == "aeolus") &&
          (cfg.grid.size == (32, 32, 32) && cfg.grid.extent == (3200.0, 3200.0, 1600.0)) &&
          (cfg.core.model === :boussinesq && cfg.dynamics.theta0 == 300.0 && !cfg.dynamics.moisture)
    @test ((cfg.boundaries.x, cfg.boundaries.y, cfg.boundaries.z) == (:periodic, :periodic, :free_slip)) &&
          (cfg.sgs.model === :smagorinsky)
    @test cfg.timestep.scheme === :rk3 && cfg.time.dt === :adaptive && cfg.time.cfl == 0.5 &&
          cfg.time.dt_max == 10.0 && cfg.time.duration == 3600.0 && cfg.time.nsteps === nothing
    # the components: the advection and the SGS, nothing else until a section enables it
    @test (enabled_components(cfg) == Any[CentredAdvection, SGS]) &&
          (only(config_domains(cfg)).domain.id == "main") &&
          (cfg.domain.static_driver === nothing && cfg.domain.dynamic_driver === nothing)
    # the default output set (ONE-c): u, v, w, theta and the time series of the domain means
    @test cfg.output.variables == ["u", "v", "w", "theta", "u_mean", "v_mean", "w_mean", "theta_mean"]
    # a fluid at rest stays at rest, at the adaptive limit dt_max
    rest = load_run_config(IOBuffer(""); overrides = ["grid.size=[16,16,16]", "time.duration=60.0",
                                                     "fftw.planner=\"estimate\""])
    r = cm_run(rest)
    sim = simulation(r.obj)
    @test sim.t == 60.0 && r.obj.stepper.step == 6                       # 6 steps of dt_max = 10 s
    @test all(iszero, interior_view(sim.u)) && all(iszero, interior_view(sim.v)) &&
          all(iszero, interior_view(sim.w))
    @test (all(==(300.0), interior_view(sim.θ))) &&
          (only(r.obj.model.domains).scheme === :rk3)
    # perturbed: θ is conserved between the no-flux walls, the SGS acts, ∇·u at round-off
    pert = load_run_config(IOBuffer("""
        [initial]
        perturbation_amplitude = 0.5
        """); overrides = ["grid.size=[16,16,16]", "time.duration=\"none\"", "time.nsteps=10",
                           "fftw.planner=\"estimate\""])
    b = Config.build(SimulationDriverType(), pert)
    θ0 = sum(interior_view(simulation(b).θ))
    r = cm_run(pert)
    sim = simulation(r.obj)
    # slow motion: dt_max binds
    @test (r.obj.stepper.step == 10 && sim.t == 100.0) &&
          (isapprox(sum(interior_view(sim.θ)), θ0; rtol = 1e-12)) &&
          (maximum(interior_view(sim.νt)) > 0)
    @test (max_divergence(sim) < 1e-10) &&
          (isfinite(kinetic_energy(sim)) && kinetic_energy(sim) > 0)
end

@testset "the component sections are generated from the component schemas" begin
    sch = driver_schema()
    opts(x) = [(o.key, o.default) for o in x.options]
    function generated(C)
        s = only(x for x in sch.sections if x.name == CMC.section(C))
        own = CMC.schema(C)
        same = own.selector === nothing ? isequal(opts(s), opts(own)) :
            # a slot section (ONE-m, #476): the component's providers are those of the merged section
            all(p -> isequal(opts(only(x for x in s.selector.providers if x.name == p.name && x.kinds == p.kinds)), opts(p)),
                own.selector.providers)
        return same && is_domain_section(sch, s.name)
    end
    @test all(generated, COMPONENTS)
    @test [CMC.section(C) for C in COMPONENTS] == ["advection", "sgs", "coriolis", "sponge", "surface_theta_flux",
                                                   "large_scale_forcing", "surface", "towers",   # ONE-g #456, ONE-m #476
                                                   "inflow_statistics", "kinematic_flow", "fringe",    # ONE-l #475
                                                   "surface", "surface", "radiation"]   # tiles, prescribed, mean_column (NEST-p3 #483)
    @test Set(sch.domain_sections) == Set(DOMAIN_SECTIONS)
    # run-level sections: the common ones and time, timestep (one scheme steps every domain, ONE-m #476),
    # output, checks, budgets (one switch for every domain, #539) (no preprocess: AeolusPrep.jl, ONE-h)
    @test Set(s.name for s in sch.sections if !is_domain_section(sch, s.name)) ==
          Set(["run", "fftw", "logging", "checkpoint", "output", "time", "spinup", "checks", "budgets"])
    # a component is enabled by its own section
    cfg = load_run_config(IOBuffer("[coriolis]\nf = 1e-4\n[sponge]\nalpha_max = 0.01\nz_start = 1200.0\n[sgs]\nmodel = \"none\"\n"))
    @test enabled_components(cfg) == Any[CentredAdvection, Coriolis, Sponge]
    d = only(Config.build(SimulationDriverType(), cfg).model.domains)
    @test map(CMC.component_name, d.components) == (:advection, :coriolis, :sponge)
end

@testset "every model's build puts FFTW on Julia's threads (#524)" begin
    # Before #524 only the urban site called init_threads!: a model domain kept
    # FFTW at its start-up count of 1. The policy is put on a count that is
    # not Julia's (3, chosen by no one) to tell the two apart in a 1-thread suite.
    st = Aeolus.Operators._FFTW_POLICY
    t0, e0 = st.threads, st.threads_explicit
    nfftw() = Aeolus.Operators.FFTW.get_num_threads()
    cfg = load_run_config(IOBuffer("[coriolis]\nf = 1e-4\n[sgs]\nmodel = \"none\"\n"))
    try
        st.threads, st.threads_explicit = 3, false
        Config.build(SimulationDriverType(), cfg)
        @test Aeolus.fftw_policy().threads == Threads.nthreads() == nfftw()
        # an explicit count is kept
        Aeolus.Operators.set_fftw_policy!(; threads = 2)
        Config.build(SimulationDriverType(), cfg)
        @test Aeolus.fftw_policy().threads == 2 == nfftw()
    finally
        Aeolus.Backends.init_threads!(; fftw_threads = t0)
        st.threads_explicit = e0
    end
end

const CM_TWO = """
    schema_version = 2
    [time]
    dt = 1.0
    [[domain]]
    id = "outer"
    [domain.sgs]
    model = "vreman"
    [domain.grid]
    size = [8, 8, 8]
    [[domain]]
    id = "inner"
    parent = "outer"
    origin = [1200.0, 1200.0]
    refinement = [8, 8, 8]
    [domain.sgs]
    model = "amd"
    [domain.grid]
    size = [8, 8, 8]
    extent = [400.0, 400.0, 200.0]
    """

@testset "[[domain]]: complete per-domain sections, per-domain overrides, the resolved config" begin
    p = cm_file(CM_TWO)
    cfg = load_run_config(p)
    ds = config_domains(cfg)
    # each block its own
    @test ([d.domain.id for d in ds] == ["outer", "inner"]) &&
          ([d.sgs.model for d in ds] == [:vreman, :amd]) &&
          ([d.grid.size for d in ds] == [(8, 8, 8), (8, 8, 8)])
    @test ([d.grid.extent for d in ds] == [(3200.0, 3200.0, 1600.0), (400.0, 400.0, 200.0)]) &&
          (config_origin(cfg, "domain[2].sgs.model") == p && config_origin(cfg, "domain[1].grid.extent") == "default")
    # per-domain sections are reached through the domains
    @test_throws ArgumentError cfg.grid
    @test domain_view(cfg, 2).sgs.model === :amd && domain_view(cfg, 2).time == cfg.time
    # --set addresses one domain, by its id or its number (NEST-p3)
    c2 = load_run_config(p; overrides = ["domain.inner.sgs.model=\"sigma\"", "domain.1.grid.halo=[3, 3, 3]"])
    @test ([d.sgs.model for d in config_domains(c2)] == [:vreman, :sigma]) &&
          ([d.grid.halo for d in config_domains(c2)] == [(3, 3, 3), (2, 2, 2)]) &&
          (config_origin(c2, "domain[2].sgs.model") == "--set")
    # a run-level override of a per-domain section is refused, as a run-level section is
    e = cm_error(() -> load_run_config(p; overrides = ["sgs.model=\"sigma\""]))
    @test only(e.problems).key == "sgs" && occursin("[domain.sgs]", only(e.problems).message)
    e = cm_error(() -> load_run_config(p; overrides = ["domain.middle.sgs.model=\"sigma\""]))
    @test (occursin("no [[domain]] block \"middle\"", only(e.problems).message)) &&
          ([d.sgs.model for d in config_domains(with(cfg, "sgs.model" => :none))] == [:none, :none])
    # the resolved config lists every component's options, per domain, and reloads to itself
    t = config_text(cfg)
    @test (count("\n[[domain]]\n", t) == 2) &&
          (all(s -> s == "domain" || count("\n[domain.$s]\n", t) == 2, DOMAIN_SECTIONS))
    @test occursin("\n[[domain]]\nfeedback = \"two_way\"\nid = \"inner\"\norigin = [1200.0, 1200.0]\nparent = \"outer\"\n" *
                   "refinement = [8, 8, 8]\nstatic_driver = \"none\"\n", replace(t, "dynamic_driver = \"none\"\n" => ""))
    @test !occursin("\n[sgs]\n", t) && !occursin("\n[grid]\n", t)            # only per domain
    c3 = load_run_config(IOBuffer(t))
    @test (c3 == cfg && config_text(c3) == t) &&
          ([d.sgs.model for d in config_domains(c3)] == [:vreman, :amd])
    # a run record keeps it: replayed from run.toml, the same config
    rf = start_run(cfg; root = mktempdir(), repo = mktempdir(), console = false); finish_run!(rf)
    @test load_run_config(rf.path) == cfg
    # several domains are a nest (ONE-f, #455; test/model_nesting.jl has the rest)
    m = Config.build(SimulationDriverType(), cfg).model
    # the self-nesting default
    @test ([d.id for d in m.domains] == ["outer", "inner"] && m.nest isa CMC.ModelNest) &&
          (only(m.nest.links).feedback === :two_way)
end

@testset "[[domain]]: one block is the top-level form" begin
    top = load_run_config(IOBuffer("dynamic_driver = \"d.nc\"\n[sgs]\nmodel = \"amd\"\n"))
    one = load_run_config(IOBuffer("[[domain]]\nid = \"main\"\ndynamic_driver = \"d.nc\"\n[domain.sgs]\nmodel = \"amd\"\n"))
    @test config_text(top) == config_text(one)
    @test top.domain.dynamic_driver == joinpath(pwd(), "d.nc")
    # the domain's input drivers are hashed under [inputs] per domain
    dir = mktempdir(); f = joinpath(dir, "s.nc"); write(f, "static")
    cfg = load_run_config(cm_file("static_driver = \"$f\"\n"))
    @test Config.config_inputs(cfg) == Dict("domain.main.static_driver" => Dict("path" => f, "fnv1a64" => content_hash(f)))
end

@testset "validation: per domain, with the sections named" begin
    # a block's own bad value, under domain[i]; a run-level module section is refused
    e = cm_error(() -> load_run_config(IOBuffer(CM_TWO * "[domain.time]\nscheme = \"rk3\"\n")))
    @test any(p -> p.key == "domain[2].time" && occursin("unknown table [domain.time]", p.message), e.problems)
    e = cm_error(() -> load_run_config(IOBuffer(replace(CM_TWO, "model = \"amd\"" => "model = \"amd2\""))))
    @test only(e.problems).key == "domain[2].sgs.model"
    e = cm_error(() -> load_run_config(IOBuffer(replace(CM_TWO, "model = \"vreman\"" => "model = \"vreman2\""))))
    @test only(e.problems).key == "domain[1].sgs.model"
    e = cm_error(() -> load_run_config(IOBuffer(replace(CM_TWO, "schema_version = 2\n" =>
                                                            "schema_version = 2\nstatic_driver = \"s.nc\"\n[coriolis]\nf = 1e-4\n"))))
    @test sort([p.key for p in e.problems]) == ["coriolis", "static_driver"]
    e = cm_error(() -> load_run_config(IOBuffer(replace(CM_TWO, "id = \"inner\"" => "id = \"outer\""))))
    @test only(e.problems).key == "domain" && occursin("\"outer\" is given 2 times", only(e.problems).message)
    e = cm_error(() -> load_run_config(IOBuffer("domain = 3\n")))
    @test occursin("[[domain]] blocks", only(e.problems).message)
    # a cross-field check sees one domain: the sponge starts below the top of *that* domain
    e = cm_error(() -> load_run_config(IOBuffer(CM_TWO * "[domain.sponge]\nalpha_max = 0.01\nz_start = 300.0\n")))
    @test only(e.problems).key == "domain[2].sponge.z_start" && occursin("below the top", only(e.problems).message)
    # [checks]: a check nobody publishes (ONE-d, #453; test/checks.jl has the rest)
    e = cm_error(() -> load_run_config(IOBuffer("[checks]\nenergy_closures = { required = true }\n")))
    @test only(e.problems).key == "checks.energy_closures" && occursin("did you mean", only(e.problems).message)
    # the obstacles come from the domain's static driver
    e = cm_error(() -> load_run_config(IOBuffer("[obstacles]\nenabled = true\n")))
    @test any(p -> p.key == "obstacles.enabled" && occursin("static_driver", p.message), e.problems)
end

# test components for the slot, requirement and incompatibility rules
struct CMHeavySGS <: AbstractComponent end
CMC.component_name(::Type{CMHeavySGS}) = :heavy_sgs
CMC.section(::Type{CMHeavySGS}) = "heavy_sgs"
CMC.slot(::Type{CMHeavySGS}) = :sgs
struct CMNeedsSponge <: AbstractComponent end
CMC.component_name(::Type{CMNeedsSponge}) = :needs_sponge
CMC.section(::Type{CMNeedsSponge}) = "needs_sponge"
CMC.requires(::Type{CMNeedsSponge}) = (Sponge,)
CMC.incompatible(::Type{CMNeedsSponge}) = (Coriolis,)

@testset "slots, requirements and incompatibilities, before anything is built" begin
    msg(ps) = Dict(p.key => p.message for p in ps)
    @test isempty(CMC.component_problems(Any[CentredAdvection, SGS, Coriolis, Sponge]))
    m = msg(CMC.component_problems(Any[CentredAdvection, SGS, CMHeavySGS]))
    @test occursin("the :sgs slot holds one component; [sgs] (sgs) and [heavy_sgs] (heavy_sgs) are both enabled", m["heavy_sgs"])
    ps = CMC.component_problems(Any[CentredAdvection, CMNeedsSponge, Coriolis])
    @test any(p -> occursin("requires Sponge ([sponge]), which is not enabled", p.message), ps)
    @test any(p -> occursin("incompatible with coriolis ([coriolis])", p.message), ps)
    @test any(p -> p.key == "advection", CMC.component_problems(Any[SGS]))       # no :advection provider
    # the same rules hold for built components (validate_components adds the stage rules)
    @test_throws ConfigError CMC.validate_components((CentredAdvection(), CMHeavySGS(), SGS(Aeolus.Operators.StaticSmagorinsky(0.17))))
end
