# Nesting by domains (ONE-f, #455): src/components/nesting.jl,
# src/config/simulation_nesting.jl.
#
#   * the self-nesting default is two-way (NestedRun, add_nest!, [[domain]]);
#   * a config with several [[domain]] blocks is one model: the graph is
#     validated (parents, sizes, placement, faces, a fixed dt, no budgets);
#   * each domain runs its own stage pipeline inside the nest: a child's
#     stages run once per substep, its in-core stages once per stage;
#   * one-way: the parent is bit for bit the parent run alone;
#   * two-way: the parent changes, and the composite θ of the nest is
#     conserved to round-off (Berger–Colella refluxing); with three levels
#     and a one-way sibling too;
#   * a nested run writes one output file per domain, evaluates its checks
#     on every domain, and resumes bit for bit from its checkpoint.

using Test
using TOML
using NCDatasets: NCDataset
using Aeolus
using Aeolus.Config
using Aeolus.Components
const MNC = Aeolus.Components
using Aeolus.SimulationDriver: SimulationDriverType, simulation
using Aeolus.Nesting: NestedRun, NestHierarchy, add_nest!, nested_grid, nest_bcs, composite_integral, nest_link,
                      step_nested!
using Aeolus.Grids: RegularGrid
using Aeolus.Fields: interior_view, set!
using Aeolus.BoundaryConditions: FieldBoundaryConditions
using Aeolus.Simulations: BoussinesqSimulation, max_divergence

mn_error(f) = try
    f(); error("expected a ConfigError")
catch e
    e isa ConfigError ? e : rethrow()
end
mn_run(cfg; kw...) = run_config(cfg; root = mktempdir(), console = false, repo = mktempdir(), kw...)

# a 16 × 16 × 8 parent (Δ = 100 m), its child of 800 × 800 × 400 m at
# (400, 400) refined by 2: a convective mixed layer in a 3 m/s wind
const MN_HEAD = """
    schema_version = 2
    [run]
    seed = 7
    [fftw]
    planner = "estimate"
    [timestep]
    scheme = "rk3"
    [time]
    dt = 4.0
    duration = "none"
    nsteps = 6
    [grid]
    size = [16, 16, 8]
    extent = [1600.0, 1600.0, 800.0]
    [initial]
    u = {kind = "constant", value = 3.0}
    theta = {kind = "mixed_layer_lapse", value = 300.0, z_i = 400.0, lapse = 0.005}
    perturbation_amplitude = 0.5
    perturbation_depth = 400.0
    [output]
    variables = ["u", "theta", "theta_mean"]
    """
mn_child(; id = "inner", parent = "outer", origin = "[400.0, 400.0]", feedback = "two_way",
         size = "[16, 16, 8]", extent = "[800.0, 800.0, 400.0]", refinement = "[2, 2, 2]") = """
    [[domain]]
    id = "$id"
    parent = "$parent"
    origin = $origin
    refinement = $refinement
    feedback = "$feedback"
    [domain.grid]
    size = $size
    extent = $extent
    """
mn_two(; kw...) = MN_HEAD * "[[domain]]\nid = \"outer\"\n" * mn_child(; kw...)
mn_cfg(text; overrides = String[]) = load_run_config(IOBuffer(text); overrides = overrides)

@testset "the self-nesting default is two-way" begin
    m = RegularGrid(; size = (8, 8, 8), extent = (800.0, 800.0, 800.0))
    p = BoussinesqSimulation(m; θ₀ = 300.0)
    nest = nested_grid(m, ((200.0, 600.0), (200.0, 600.0), (0.0, 400.0)), 2)
    c = BoussinesqSimulation(nest.child_mesh; θ₀ = 300.0, bcs = nest_bcs(nest.child_mesh; bottom = :wall))
    @test NestedRun(p, c, nest; cfl_parent = nothing, cfl_child = nothing).feedback === :two_way
    h = NestHierarchy(BoussinesqSimulation(m; θ₀ = 300.0); cfl = nothing)
    c2 = BoussinesqSimulation(nest.child_mesh; θ₀ = 300.0, bcs = nest_bcs(nest.child_mesh; bottom = :wall))
    add_nest!(h, :root => c2, nest; id = :child, cfl = nothing)
    @test nest_link(h, :child).feedback === :two_way
    @test [d.domain.feedback for d in config_domains(mn_cfg(MN_HEAD * "[[domain]]\nid = \"outer\"\n" *
                                                          replace(mn_child(), "feedback = \"two_way\"\n" => "")))] ==
          [:two_way, :two_way]
end

@testset "[[domain]] graph: validated before anything is built" begin
    cfg = mn_cfg(mn_two())
    ds = config_domains(cfg)
    @test [d.domain.parent for d in ds] == [nothing, "outer"]
    @test ds[2].domain.origin == (400.0, 400.0) && ds[2].domain.refinement == (2, 2, 2)
    # the resolved config keeps the graph and reloads to itself
    t = config_text(cfg)
    @test occursin("parent = \"outer\"", t) && occursin("feedback = \"two_way\"", t)
    @test load_run_config(IOBuffer(t)) == cfg
    keyed(e) = Dict(p.key => p.message for p in e.problems)
    # a parent must be an earlier domain; the first is the root
    e = mn_error(() -> mn_cfg(mn_two(; parent = "nowhere")))
    @test occursin("not an earlier domain", keyed(e)["domain[2].domain.parent"])
    e = mn_error(() -> mn_cfg(MN_HEAD * "[[domain]]\nid = \"outer\"\n[[domain]]\nid = \"inner\"\n"))
    @test occursin("every domain after the first is nested", keyed(e)["domain[2].domain.parent"])
    e = mn_error(() -> mn_cfg(MN_HEAD * "[[domain]]\nid = \"outer\"\norigin = [0.0, 0.0]\n" * mn_child()))
    @test occursin("root domain", keyed(e)["domain[1].domain.origin"])
    e = mn_error(() -> mn_cfg(MN_HEAD * "[[domain]]\nid = \"outer\"\n" * replace(mn_child(), "refinement = [2, 2, 2]\n" => "")))
    @test occursin("needs its refinement", keyed(e)["domain[2].domain.refinement"])
    # the child's cells follow from its extent, the parent spacing and the refinement
    e = mn_error(() -> mn_cfg(mn_two(; size = "[8, 8, 8]")))
    @test occursin("(16, 16, 8) cells", keyed(e)["domain[2].grid.size"])
    e = mn_error(() -> mn_cfg(mn_two(; extent = "[830.0, 800.0, 400.0]", size = "[17, 16, 8]")))
    @test occursin("not a whole number of child cells", keyed(e)["domain[2].grid.extent"])
    # inside the parent
    e = mn_error(() -> mn_cfg(mn_two(; origin = "[1000.0, 400.0]")))
    @test occursin("outside its parent", keyed(e)["domain[2].domain.origin"])
    # the faces are the nest's; a fixed dt; no budgets; no couplings in a child
    e = mn_error(() -> mn_cfg(mn_two() * "[domain.boundaries]\nx = \"free_slip\"\n"))
    @test occursin("keeps its parent's faces", keyed(e)["domain[2].boundaries.x"])
    e = mn_error(() -> mn_cfg(mn_two(); overrides = ["time.dt=\"adaptive\""]))
    @test occursin("fixed dt", keyed(e)["time.dt"])
    e = mn_error(() -> mn_cfg(mn_two(); overrides = ["output.variables=[\"budget_energy\"]"]))
    @test occursin("measures no budgets", keyed(e)["output.variables"])
    e = mn_error(() -> mn_cfg(mn_two() * "[domain.forcing]\nradiative_cooling = -1e-5\n"))
    @test occursin("not nested", keyed(e)["domain[2].domain.parent"])
    e = mn_error(() -> mn_cfg(mn_two() * "[domain.core]\nmodel = \"anelastic\"\n"))
    @test occursin("boussinesq", keyed(e)["domain[2].core.model"])
    # placement the stencils cannot take is refused at build, naming the domain
    cfg = mn_cfg(mn_two(; origin = "[0.0, 400.0]"); overrides = ["boundaries.x=\"free_slip\""])
    e = mn_error(() -> Config.build(SimulationDriverType(), cfg))
    @test occursin("inner", only(e.problems).message) && occursin("properly nested", only(e.problems).message)
end

@testset "the build: the child on its nest's mesh with nest faces" begin
    r = Config.build(SimulationDriverType(), mn_cfg(mn_two()))
    m = r.model
    outer, inner = m.domains
    @test m.nest isa MNC.ModelNest && length(m.nest.hierarchy.domains) == 2
    g = inner.state.mesh
    @test (g.x0, g.y0, g.z0) == (400.0, 400.0, 0.0) && (g.dx, g.dy, g.dz) == (50.0, 50.0, 50.0)
    @test (g.Nx, g.Ny, g.Nz) == (16, 16, 8)
    # the child keeps its own initial state from its [initial] (the run-level one)
    @test interior_view(inner.state.u)[3, 3, 3] == 3.0
    @test nest_link(m, "inner").feedback === :two_way
    @test map(MNC.component_name, inner.components) == map(MNC.component_name, outer.components)
end

# a counting component: how often each stage of a domain runs
mutable struct MNCount <: AbstractComponent
    n::Dict{Symbol,Int}
end
MNC.component_name(::Type{MNCount}) = :count
MNC.section(::Type{MNCount}) = "count"
MNC.stages(::MNCount) = (Stage(:clock), Stage(:tendencies), Stage(:surface_update))
MNC.run_stage!(c::MNCount, d, ::Val{S}, dt) where {S} = (c.n[S] = get(c.n, S, 0) + 1; nothing)

function mn_pair(; scheme = :euler, feedback = :one_way)
    pm = RegularGrid(; size = (16, 16, 8), extent = (1600.0, 1600.0, 800.0))
    nest = nested_grid(pm, ((400.0, 1200.0), (400.0, 1200.0), (0.0, 400.0)), 2)
    wall = FieldBoundaryConditions(:periodic, :periodic, :wall)
    p = BoussinesqSimulation(pm; θ₀ = 300.0, bcs = wall, θ_bcs = FieldBoundaryConditions(:periodic, :periodic, :noflux))
    c = BoussinesqSimulation(nest.child_mesh; θ₀ = 300.0, bcs = nest_bcs(nest.child_mesh; bottom = :wall))
    for s in (p, c)
        set!(s.θ, (x, y, z) -> 300.0 + 0.003z + 0.1sin(2π * x / 800))
        set!(s.u, (x, y, z) -> 2.0)
    end
    cp = MNCount(Dict{Symbol,Int}()); cc = MNCount(Dict{Symbol,Int}())
    dp = Domain(p; id = "p", scheme = scheme, components = (CentredAdvection(), cp))
    dc = Domain(c; id = "c", scheme = scheme, components = (CentredAdvection(), cc))
    return nested_model((dp, dc), [NestLink(2, 1, nest; feedback = feedback)]), cp, cc
end

@testset "each domain runs its own pipeline: a child's stages once per substep ($scheme)" for scheme in (:euler, :rk3)
    m, cp, cc = mn_pair(; scheme = scheme)
    for _ in 1:3
        step!(m, 5.0)
    end
    ns = scheme === :rk3 ? 3 : 1
    @test cp.n == Dict(:clock => 3, :surface_update => 3, :tendencies => 3ns)
    @test cc.n == Dict(:clock => 6, :surface_update => 6, :tendencies => 6ns)   # 2 substeps per parent step
    @test m.domains[1].state.t == m.domains[2].state.t == 15.0
    @test max_divergence(m.domains[2].state) < 1e-10
    # a nested model takes fixed steps; the domain pairs are checked
    @test_throws ArgumentError MNC.step_adaptive!(m)
    @test_throws ArgumentError NestLink(1, 2, nothing)
    @test_throws ArgumentError NestLink(2, 1, nothing; feedback = :both)
end

# the engine (NEST-p2, #482): with fixed steps every domain's prognostic
# state is that of Nesting.step_nested!(::NestHierarchy), bit for bit
mn_state_equal(a, b) = all(i -> all(f -> interior_view(getproperty(a.domains[i].state, f)) ==
                                         interior_view(getproperty(b.domains[i].state, f)), (:u, :v, :w, :θ, :p)),
                           eachindex(a.domains))

@testset "the nest engine with fixed steps is Nesting.step_nested!, bit for bit ($scheme, $fb)" for scheme in (:euler, :rk3), fb in (:one_way, :two_way)
    a, = mn_pair(; scheme = scheme, feedback = fb)
    b, = mn_pair(; scheme = scheme, feedback = fb)
    for _ in 1:3
        step!(a, 5.0)
        step_nested!(b.nest.hierarchy; dt = 5.0)
    end
    @test mn_state_equal(a, b)
    @test a.domains[2].state.t == b.domains[2].state.t == 15.0
    @test nest_link(a, "c").stats.child_substeps == 6 && nest_link(a, "c").stats.n_last == 2
end

@testset "the nest engine with three levels and a sibling is Nesting.step_nested!, bit for bit" begin
    text = MN_HEAD * "[[domain]]\nid = \"outer\"\n" *
           mn_child(; id = "a", origin = "[300.0, 300.0]", size = "[12, 12, 8]", extent = "[600.0, 600.0, 400.0]") *
           mn_child(; id = "g", parent = "a", origin = "[200.0, 200.0]", size = "[8, 8, 8]",
                    extent = "[200.0, 200.0, 200.0]") *
           mn_child(; id = "b", origin = "[1100.0, 1100.0]", feedback = "one_way", size = "[8, 8, 4]",
                    extent = "[400.0, 400.0, 200.0]")
    a = Config.build(SimulationDriverType(), mn_cfg(text)).model
    b = Config.build(SimulationDriverType(), mn_cfg(text)).model
    for _ in 1:3
        step!(a, 4.0)
        step_nested!(b.nest.hierarchy; dt = 4.0)
    end
    @test mn_state_equal(a, b)
end

@testset "one-way: the parent is the parent run alone, bit for bit" begin
    nested = mn_run(mn_cfg(mn_two(; feedback = "one_way")))
    alone = mn_run(mn_cfg(MN_HEAD))
    # the prognostic state: the interiors (the halos are filled from them at
    # every stage, and the nest's prolongation fills the parent's after a step)
    a = simulation(alone.obj); b = simulation(nested.obj)
    for f in (:u, :v, :w, :p, :θ, :νt, :κt)
        @test interior_view(getproperty(a, f)) == interior_view(getproperty(b, f))
    end
    @test haskey(state_fields(nested.obj), :inner_θ) && b.t == a.t == 24.0
    ci = nested.obj.model.domains[2].state
    @test ci.t == 24.0 && all(isfinite, interior_view(ci.θ))
    @test nest_link(nested.obj.model, "inner").stats.child_substeps == 12
end

@testset "two-way: the child feeds back, and the composite θ is conserved" begin
    alone = mn_run(mn_cfg(MN_HEAD))
    cfg = mn_cfg(mn_two())
    r = Config.build(SimulationDriverType(), cfg)
    h = r.model.nest.hierarchy
    I0 = composite_integral(h, :θ)
    for _ in 1:6
        step!(r.model, 4.0)
    end
    I1 = composite_integral(h, :θ)
    rel = abs(I1 - I0) / abs(I0)
    @info "two-way nest by domains: composite θ relative change over 6 steps $rel"
    @test rel < 1e-13
    # the parent sees the child: it differs from the parent alone where the child covers it
    pa = simulation(alone.obj); pn = r.model.domains[1].state
    @test pn.t == pa.t
    d = interior_view(pn.θ) .- interior_view(pa.θ)
    @test maximum(abs, d[6:11, 6:11, 1:3]) > 1e-6
    @test max_divergence(pn) < 1e-9
end

@testset "three levels and siblings, one-way and two-way per child" begin
    # outer (16², Δ 100 m) ⊃ a (two-way, Δ 50 m) ⊃ g (two-way, Δ 25 m); b a one-way sibling of a
    text = MN_HEAD * "[[domain]]\nid = \"outer\"\n" *
           mn_child(; id = "a", origin = "[300.0, 300.0]", size = "[12, 12, 8]", extent = "[600.0, 600.0, 400.0]") *
           mn_child(; id = "g", parent = "a", origin = "[200.0, 200.0]", size = "[8, 8, 8]",
                    extent = "[200.0, 200.0, 200.0]") *
           mn_child(; id = "b", origin = "[1100.0, 1100.0]", feedback = "one_way", size = "[8, 8, 4]",
                    extent = "[400.0, 400.0, 200.0]")
    cfg = mn_cfg(text)
    r = Config.build(SimulationDriverType(), cfg)
    m = r.model; h = m.nest.hierarchy
    @test [d.id for d in m.domains] == ["outer", "a", "g", "b"]
    @test [l.feedback for l in m.nest.links] == [:two_way, :two_way, :one_way]
    @test m.domains[3].state.mesh.x0 == 500.0 && m.domains[3].state.mesh.dx == 25.0
    for _ in 1:3
        step!(m, 4.0)
    end
    @test all(d -> d.state.t == 12.0, m.domains)
    @test all(d -> all(isfinite, interior_view(d.state.θ)), m.domains)
    @test nest_link(m, "g").stats.child_substeps == 3 * 2 * 2      # 2 substeps of a, each 2 of g
    # the two-way levels conserve their composite θ (the one-way sibling's parent cells stay the parent's)
    two = mn_cfg(MN_HEAD * "[[domain]]\nid = \"outer\"\n" *
                 mn_child(; id = "a", origin = "[300.0, 300.0]", size = "[12, 12, 8]", extent = "[600.0, 600.0, 400.0]") *
                 mn_child(; id = "g", parent = "a", origin = "[200.0, 200.0]", size = "[8, 8, 8]",
                          extent = "[200.0, 200.0, 200.0]"))
    r2 = Config.build(SimulationDriverType(), two)
    h2 = r2.model.nest.hierarchy
    I0 = composite_integral(h2, :θ)
    for _ in 1:3
        step!(r2.model, 4.0)
    end
    rel = abs(composite_integral(h2, :θ) - I0) / abs(I0)
    @info "three-level two-way nest: composite θ relative change over 3 steps $rel"
    @test rel < 1e-13
    # overlapping siblings are refused, naming them
    bad = mn_cfg(text * mn_child(; id = "c", origin = "[1200.0, 1200.0]", size = "[8, 8, 4]",
                                 extent = "[400.0, 400.0, 200.0]", feedback = "one_way"))
    e = mn_error(() -> Config.build(SimulationDriverType(), bad))
    @test occursin("overlaps its sibling", only(e.problems).message)
end

@testset "a nested run: one output file per domain, checks on every domain, a bit-identical resume" begin
    cfg = mn_cfg(mn_two() * "[checks]\nfinite = {required = true}\n";
                 overrides = ["checkpoint.every_steps=3"])
    full = mn_run(cfg)
    out = Aeolus.IO.output_dir(full.folder)
    @test isfile(joinpath(out, "output.nc")) && isfile(joinpath(out, "output_inner.nc"))
    NCDataset(joinpath(out, "output_inner.nc")) do ds
        @test ds["x"][1] == 425.0 && length(ds["x"]) == 16
        @test all(isfinite, ds["theta"][:, :, :, end])
    end
    NCDataset(joinpath(out, "output.nc")) do ds
        @test ds["x"][1] == 50.0
    end
    res = TOML.parsefile(joinpath(out, "results.toml"))
    f = res["checks"]["finite"]
    @test res["passed"] && f["passed"] && f["value"] == 0.0
    @test haskey(f, "outer") && haskey(f, "inner") && f["inner"]["value"] == 0.0
    # interrupted after step 4, resumed from the checkpoint of step 3: the same state, both domains
    root = mktempdir()
    stop = (r, n) -> n == 4 && throw(InterruptException())
    @test_throws InterruptException run_config(cfg; root = root, console = false, repo = mktempdir(), callback = stop)
    folder = only(readdir(root; join = true))
    resumed = run_config(load_run_config(folder); root = root, console = false, repo = mktempdir(), resume = folder)
    a = state_fields(full.obj); b = state_fields(resumed.obj)
    @test haskey(a, :inner_u) && keys(a) == keys(b)
    @test all(k -> a[k] == b[k], keys(a))
end

@testset "a one-way child that goes non-finite is a blow-up of the run" begin
    # A one-way child feeds nothing back, so a NaN in it never reaches the
    # root core the blow-up guard is given; the guard inspects every domain
    # (component_nonfinite_fields of the nest). Without [checks] finite, a
    # NaN put into the child after the last step used to end the run
    # `completed`.
    cfg = mn_cfg(mn_two(; feedback = "one_way"))
    root = mktempdir()
    poison = (r, n) -> n == 6 && (r.model.domains[2].state.θ.data[6, 6, 4] = NaN)
    err = try
        run_config(cfg; root = root, console = false, repo = mktempdir(), callback = poison)
        nothing
    catch e
        e
    end
    @test err isa Aeolus.IO.BlowupError
    # checks at step 1 (the first) and, forced, at the last step 6
    @test err.step == 6 && err.checkpoint_step == 1 && err.cause === nothing
    @test [f.name for f in err.fields] == [:var"inner.θ"] && err.fields[1].count == 1
    folder = only(readdir(root; join = true))
    s = Aeolus.IO.read_run_toml(folder)["session"][end]
    @test s["status"] == "failed" && occursin("BlowupError", s["error"])
    # a diagnostics row per model step (the monitor reports after the whole
    # nest step, not from the root's step core)
    rows = [split(l, ',') for l in readlines(joinpath(folder, "diagnostics.csv"))[2:end]]
    @test [parse(Int, r[1]) for r in rows] == collect(1:6) && all(r -> parse(Float64, r[4]) == 4.0, rows)
    # the crash checkpoint is the good state of step 1, the whole nest at t = 4 s
    # (taken after the children's substeps, not between the root's step and theirs)
    ck = Aeolus.IO.read_checkpoint(err.checkpoint)
    r = Config.build(SimulationDriverType(), cfg)
    MNC.restore_checkpoint_state!(r.model.nest, ck[:components][:nest].state)
    c = r.model.domains[2].state
    @test ck[:t] == 4.0 && c.t == 4.0 && all(isfinite, interior_view(c.θ))
    @test isempty(Aeolus.IO.component_nonfinite_fields(r.model.nest))
end
