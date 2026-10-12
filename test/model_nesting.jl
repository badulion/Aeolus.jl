# Nesting by domains (ONE-f, #455): src/components/nesting.jl,
# src/config/simulation_nesting.jl.
#
#   * the self-nesting default is two-way (NestedRun, add_nest!, [[domain]]);
#   * a config with several [[domain]] blocks is one model: the graph is
#     validated (parents, sizes, placement, faces, no budgets);
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
const SimulationDriver_warn = Aeolus.SimulationDriver.nest_handover_warnings
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
    # the faces are the nest's; no budgets; no couplings in a child
    e = mn_error(() -> mn_cfg(mn_two() * "[domain.boundaries]\nx = \"free_slip\"\n"))
    @test occursin("keeps its parent's faces", keyed(e)["domain[2].boundaries.x"])
    e = mn_error(() -> mn_cfg(mn_two(); overrides = ["output.variables=[\"budget_energy\"]",
                                                     "nesting.budget_exchange=\"off\""]))
    @test occursin("measures no budgets", keyed(e)["output.variables"])
    e = mn_error(() -> mn_cfg(mn_two() * "[domain.forcing]\nradiative_cooling = -1e-5\n"))
    @test occursin("not nested", keyed(e)["domain[2].domain.parent"])
    # (mixed cores are nested since NEST-p2, #482; an anelastic child needs the Euler step)
    e = mn_error(() -> mn_cfg(mn_two() * "[domain.core]\nmodel = \"anelastic\"\n"))
    @test occursin("SSP-RK3", keyed(e)["domain[2].core.model"])
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

# ---------------------------------------------------------------------------
# adaptive steps and window retries (NEST-p2, #482)
# ---------------------------------------------------------------------------

using Aeolus.TimeIntegration: CFLConstraints, adaptive_dt

# Arms a failure: the next stage-11 entry of the component's domain while
# armed puts a NaN into θ (the arm lives outside the model, so the window
# restore does not re-arm it).
const MN_ARM = Dict{String,Int}()
struct MNFragile <: AbstractComponent
    id::String
end
MNC.component_name(::Type{MNFragile}) = :fragile
MNC.section(::Type{MNFragile}) = "fragile"
MNC.stages(::MNFragile) = (Stage(:surface_update),)
function MNC.run_stage!(c::MNFragile, d, ::Val{:surface_update}, dt)
    if get(MN_ARM, c.id, 0) > 0
        MN_ARM[c.id] -= 1
        d.state.θ.data[d.state.mesh.Hx + 3, d.state.mesh.Hy + 3, d.state.mesh.Hz + 2] = NaN
    end
    return nothing
end

# an adaptive pair: a 2 m/s parent wind, and in the child a jet of up to
# 2 + jet m/s away from its faces (so the child needs more than r substeps)
function mn_adaptive(; feedback = :two_way, jet = 4.0, scheme = :rk3, C = 0.5, growth = 1.5, extra = ())
    pm = RegularGrid(; size = (16, 16, 8), extent = (1600.0, 1600.0, 800.0))
    nest = nested_grid(pm, ((400.0, 1200.0), (400.0, 1200.0), (0.0, 400.0)), 2)
    wall = FieldBoundaryConditions(:periodic, :periodic, :wall)
    p = BoussinesqSimulation(pm; θ₀ = 300.0, bcs = wall, θ_bcs = FieldBoundaryConditions(:periodic, :periodic, :noflux))
    c = BoussinesqSimulation(nest.child_mesh; θ₀ = 300.0, bcs = nest_bcs(nest.child_mesh; bottom = :wall))
    for s in (p, c)
        set!(s.θ, (x, y, z) -> 300.0 + 0.003z + 0.1sin(2π * x / 800))
        set!(s.u, (x, y, z) -> 2.0)
    end
    set!(c.u, (x, y, z) -> 2.0 + jet * exp(-((x - 800)^2 + (y - 800)^2) / (2 * 120.0^2)))
    cfl = CFLConstraints(; C_advective = C, dt_max = 30.0, growth_max = growth)
    dp = Domain(p; id = "p", scheme = scheme, cfl = cfl, components = (CentredAdvection(), MNFragile("p")))
    dc = Domain(c; id = "c", scheme = scheme, cfl = cfl, components = (CentredAdvection(), MNFragile("c"), extra...))
    return nested_model((dp, dc), [NestLink(2, 1, nest; feedback = feedback)])
end

@testset "adaptive nest: the root's own step, the child's smallest stable whole number of equal substeps" begin
    m = mn_adaptive()
    L = nest_link(m, "c")
    c = m.domains[2]
    @test_throws ArgumentError step!(m, 1.0)          # an adaptive nest steps adaptively
    pattern = Tuple{Float64,Int}[]
    for w in 1:8
        n0 = L.stats.child_substeps
        lim_p = Aeolus.Components.nest_domain_limit(m.nest, 1)
        _, Δt, retries = step_adaptive!(m)
        @test retries == 0
        @test Δt == lim_p                                # the root's own limit
        n = L.stats.child_substeps - n0
        push!(pattern, (Δt, n))
        @test n == L.stats.n_last
        # equal substeps: every substep of the window is Δt/n
        @test L.stats.dt_child_max ≈ L.stats.dt_child_min || w > 1
        # the smallest stable n: Δt/n within the child's limit, Δt/(n-1) not
        # (the child's limit as its first substep picked it, from the window start)
        @test n >= 2
    end
    @info "adaptive nest: (root step, child substeps) per window" pattern
    @test first(pattern)[2] > 2                           # the jet makes the child need more than r = 2 at first
    @test length(unique(last.(pattern))) >= 2             # and n follows the child's state, window by window
    @test m.domains[1].state.t ≈ sum(first.(pattern)) && m.domains[2].state.t == m.domains[1].state.t
    @test all(isfinite, interior_view(c.state.u))
end

# the child's own stability limit at the start of each of its steps,
# computed here from its state (no growth limit: growth_max is large)
const MN_LIMITS = Tuple{Float64,Float64}[]
struct MNLimit <: AbstractComponent end
MNC.component_name(::Type{MNLimit}) = :limit_log
MNC.section(::Type{MNLimit}) = "limit_log"
MNC.stages(::MNLimit) = (Stage(:clock),)
MNC.run_stage!(::MNLimit, d, ::Val{:clock}, dt) =
    (push!(MN_LIMITS, (Float64(d.state.t), Float64(adaptive_dt(d.state, d.cfl)))); nothing)

@testset "adaptive nest: n is the smallest whole number with Δt/n inside the child's limit" begin
    m = mn_adaptive(; growth = 1e6, extra = (MNLimit(),))
    L = nest_link(m, "c")
    ns = Int[]
    for _ in 1:8
        empty!(MN_LIMITS)
        t0 = m.domains[1].state.t
        _, Δt, _ = step_adaptive!(m)
        n = L.stats.n_last
        push!(ns, n)
        lim = last(only(filter(e -> first(e) == t0, MN_LIMITS)))
        @test Δt / n <= lim * (1 + 1e-9)                 # stable
        @test n == 1 || Δt / (n - 1) > lim               # and the fewest substeps that are
        @test length(MN_LIMITS) == n                      # n substeps, equal ones
        @test all(k -> MN_LIMITS[k+1][1] - MN_LIMITS[k][1] ≈ Δt / n, 1:n-1)
    end
    @info "adaptive nest, no growth limit: child substeps per window" ns
end

@testset "window retry: a failing root redoes the window with half the step, bit for bit" begin
    a = mn_adaptive(); b = mn_adaptive()
    MN_ARM["p"] = 1
    _, Δt, retries = step_adaptive!(a)
    @test retries == 1 && a.nest.retries == 1
    _, Δt1, _ = step_adaptive!(b)                        # the twin, unarmed
    @test Δt == Δt1 / 2
    # the twin at the halved step from the same start: same state, budgets, components
    c = mn_adaptive()
    _, Δtc, rc = step_adaptive!(c; t_end = Δt)
    @test Δtc == Δt && rc == 0
    @test mn_state_equal(a, c)
    @test nest_link(a, "c").stats.child_substeps == nest_link(c, "c").stats.child_substeps
    @test nest_link(a, "c").stats.parent_retries == 1
end

@testset "window retry: a failing child redoes its window with twice the substeps ($fb)" for fb in (:one_way, :two_way)
    a = mn_adaptive(; feedback = fb); b = mn_adaptive(; feedback = fb)
    _, Δtb, _ = step_adaptive!(b)
    nb = nest_link(b, "c").stats.n_last
    MN_ARM["c"] = 1
    _, Δt, retries = step_adaptive!(a)
    La = nest_link(a, "c")
    @test retries == 1 && La.stats.child_retries == 1
    @test Δt == Δtb                                       # the root step is kept
    @test La.stats.n_last == 2nb                          # the child doubled its substeps
    @test La.stats.child_substeps == 2nb                  # the failed attempt left no count behind
    @test all(isfinite, interior_view(a.domains[2].state.θ))
    # one-way: the parent never sees the child: bit for bit the unarmed twin
    fb === :one_way && @test all(f -> interior_view(getproperty(a.domains[1].state, f)) ==
                                      interior_view(getproperty(b.domains[1].state, f)), (:u, :v, :w, :θ, :p))
    # a child that keeps failing: an error after max_retries, the window start restored
    MN_ARM["c"] = 100
    t0 = a.domains[1].state.t
    @test_throws ErrorException step_adaptive!(a)
    @test a.domains[1].state.t == t0 && a.domains[2].state.t == t0
    @test all(isfinite, interior_view(a.domains[2].state.θ))
    MN_ARM["c"] = 0
end

@testset "adaptive two-way nest with retries conserves the composite θ" begin
    m = mn_adaptive()
    h = m.nest.hierarchy
    I0 = composite_integral(h, :θ)
    MN_ARM["c"] = 1
    for _ in 1:3
        step_adaptive!(m)
    end
    MN_ARM["p"] = 1
    for _ in 1:3
        step_adaptive!(m)
    end
    @test m.nest.retries == 2
    rel = abs(composite_integral(h, :θ) - I0) / abs(I0)
    @info "adaptive two-way nest with 2 retried windows: composite θ relative change over 6 windows $rel"
    @test rel < 1e-13
    @test max_divergence(m.domains[1].state) < 1e-9 && max_divergence(m.domains[2].state) < 1e-9
end

@testset "an adaptive nested config: runs, lands on its duration, resumes bit for bit" begin
    cfg = mn_cfg(mn_two(); overrides = ["time.dt=\"adaptive\"", "time.nsteps=6", "time.cfl=0.5",
                                        "checkpoint.every_steps=3"])
    full = mn_run(cfg)
    m = full.obj.model
    @test m.nest.adaptive && all(d -> d.state.t == m.domains[1].state.t, m.domains)
    L = nest_link(m, "inner")
    @test L.stats.parent_steps == 6 && L.stats.n_min >= 2
    root = mktempdir()
    stop = (r, n) -> n == 4 && throw(InterruptException())
    @test_throws InterruptException run_config(cfg; root = root, console = false, repo = mktempdir(), callback = stop)
    folder = only(readdir(root; join = true))
    resumed = run_config(load_run_config(folder); root = root, console = false, repo = mktempdir(), resume = folder)
    a = state_fields(full.obj); b = state_fields(resumed.obj)
    @test keys(a) == keys(b) && all(k -> a[k] == b[k], keys(a))
end

# ---------------------------------------------------------------------------
# budgets across a nest: each domain closes its own, with a nest exchange
# ---------------------------------------------------------------------------

using Aeolus.Components: with_budgets, nest_exchange, NEST_EXCHANGE, budgets
using Aeolus.BudgetSystem: stage_budget
using Aeolus.Nesting: nest_covered_range

function mn_budget_pair(; feedback = :two_way, scheme = :rk3, adaptive = false, qs = (:energy, :momentum_x))
    pm = RegularGrid(; size = (16, 16, 8), extent = (1600.0, 1600.0, 800.0))
    nest = nested_grid(pm, ((400.0, 1200.0), (400.0, 1200.0), (0.0, 400.0)), 2)
    wall = FieldBoundaryConditions(:periodic, :periodic, :wall)
    p = BoussinesqSimulation(pm; θ₀ = 300.0, bcs = wall, θ_bcs = FieldBoundaryConditions(:periodic, :periodic, :noflux))
    c = BoussinesqSimulation(nest.child_mesh; θ₀ = 300.0, bcs = nest_bcs(nest.child_mesh; bottom = :wall))
    for s in (p, c)
        set!(s.θ, (x, y, z) -> 300.0 + 0.003z + 0.5sin(2π * x / 1600) + 0.2cos(2π * y / 800))
        set!(s.u, (x, y, z) -> 2.0)
        set!(s.v, (x, y, z) -> 0.5)
    end
    cfl = adaptive ? CFLConstraints(; C_advective = 0.5, dt_max = 30.0) : nothing
    dp = with_budgets(Domain(p; id = "p", scheme = scheme, cfl = cfl, components = (CentredAdvection(),)), qs)
    dc = with_budgets(Domain(c; id = "c", scheme = scheme, cfl = cfl, components = (CentredAdvection(),)), qs)
    return nested_model((dp, dc), [NestLink(2, 1, nest; feedback = feedback)]), nest
end

# the heat of the parent's covered cells, J (ρ₀ c_p Σ θ ΔV)
function mn_covered_heat(p, nest, ρcp)
    cx, cy, cz = nest_covered_range(nest)
    m = p.mesh
    return ρcp * sum(p.θ.data[m.Hx .+ cx, m.Hy .+ cy, m.Hz .+ cz]) * m.dx * m.dy * m.dz
end

@testset "budgets across a nest: each domain closes, the nest exchange accounts for the composite ($fb, $scheme, adaptive $ad)" for fb in (:one_way, :two_way), scheme in (:euler, :rk3), ad in (false, true)
    m, nest = mn_budget_pair(; feedback = fb, scheme = scheme, adaptive = ad)
    p = m.domains[1]; c = m.domains[2]
    ρcp = budgets(p).ρ[1] * Aeolus.MoistThermodynamics.DEFAULT_CONSTANTS.cp_dry
    Hcov0 = mn_covered_heat(p.state, nest, ρcp)
    for _ in 1:5
        ad ? step_adaptive!(m) : step!(m, 5.0)
    end
    for d in (p, c), q in (:energy, :momentum_x)
        meas = budgets(d).measured[q]
        sb = stage_budget(meas)
        content = abs(Aeolus.BudgetSystem.measure(meas.meter, q === :energy ? d.state.θ : d.state.u))
        @test abs(sb.closure) <= 1e-12 * content                       # every domain closes its own
        @test !haskey(sb.terms, (:core, :unattributed))
        @test !haskey(sb.terms, (:output, :between_steps))              # the nest's work is named
    end
    Ep = nest_exchange(p, :energy); Ec = nest_exchange(c, :energy)
    @info "nest budgets ($fb, $scheme, adaptive $ad): energy exchange parent $Ep, child $Ec J/m²"
    @test abs(Ec) > 1e3                                                 # the child's inflow
    # the child's θ changes only through its faces (no sources): its heat
    # change is its exchange
    sbc = stage_budget(budgets(c).measured[:energy])
    @test abs(sbc.storage - Ec) <= 1e-9 * abs(Ec)
    if fb === :one_way
        @test Ep == 0.0                                                 # the parent never sees the child
    else
        # the composite is conserved: the parent's exchange (the feedback) and
        # the child's (its inflow) add up to what the covered cells gained
        Ap = 1600.0^2; Ac = 800.0^2
        ΔHcov = mn_covered_heat(p.state, nest, ρcp) - Hcov0
        @test abs(Ep) > 10                                             # the feedback changed the parent
        @test abs(Ep * Ap + Ec * Ac - ΔHcov) <= 1e-9 * (abs(Ep * Ap) + abs(Ec * Ac))
    end
end

@testset "[nesting] budget_exchange: a nested config measures budgets, every domain closes" begin
    cfg = mn_cfg(mn_two() * "[checks]\nenergy_closure = {required = true}\n";
                 overrides = ["output.variables=[\"u\", \"budget_energy\"]"])
    full = mn_run(cfg)
    m = full.obj.model
    @test all(d -> budgets(d) !== nothing, m.domains)
    res = TOML.parsefile(joinpath(Aeolus.IO.output_dir(full.folder), "results.toml"))
    @test res["passed"] && res["checks"]["energy_closure"]["passed"]
    @test haskey(res["checks"]["energy_closure"], "inner")
    @test nest_exchange(m.domains[2], :energy) != 0
    @test_throws ArgumentError nested_model((m.domains[1], m.domains[2]), m.nest.links; budget_exchange = :off)
end

# ---------------------------------------------------------------------------
# concurrency: siblings in parallel, a one-way parent ahead of its children
# ---------------------------------------------------------------------------

# outer ⊃ a (one-way) ⊃ {g, g2} (one-way siblings); b a two-way sibling of a
const MN_TREE = MN_HEAD * "[[domain]]\nid = \"outer\"\n" *
    mn_child(; id = "a", origin = "[300.0, 300.0]", size = "[12, 12, 8]", extent = "[600.0, 600.0, 400.0]",
             feedback = "one_way") *
    mn_child(; id = "g", parent = "a", origin = "[100.0, 100.0]", size = "[8, 8, 8]",
             extent = "[200.0, 200.0, 200.0]", feedback = "one_way") *
    mn_child(; id = "g2", parent = "a", origin = "[350.0, 350.0]", size = "[4, 4, 4]",
             extent = "[100.0, 100.0, 100.0]", feedback = "one_way") *
    mn_child(; id = "b", origin = "[1100.0, 1100.0]", size = "[8, 8, 4]", extent = "[400.0, 400.0, 200.0]")

@testset "concurrent nest (siblings in parallel, a one-way parent ahead of its children) is the sequential one, bit for bit (adaptive $ad)" for ad in (false, true)
    over = ad ? ["time.dt=\"adaptive\"", "time.cfl=0.5"] : String[]
    seq = Config.build(SimulationDriverType(), mn_cfg(MN_TREE; overrides = over)).model
    con = Config.build(SimulationDriverType(), mn_cfg(MN_TREE; overrides = over)).model
    seq.nest.concurrent = false
    con.nest.concurrent = true
    @test Aeolus.Components.nest_prefetches(con.nest, 2)            # a: all its children one-way
    @test !Aeolus.Components.nest_prefetches(con.nest, 1)           # the root never
    for _ in 1:3
        ad ? (step_adaptive!(seq); step_adaptive!(con)) : (step!(seq, 4.0); step!(con, 4.0))
    end
    @test mn_state_equal(seq, con)
    @test all(d -> d.state.t == seq.domains[1].state.t, con.domains)
    @test nest_link(con, "g").stats.child_substeps == nest_link(seq, "g").stats.child_substeps > 0
    @info "concurrent nest (adaptive $ad) on $(Threads.nthreads()) threads: identical to sequential"
end

# ---------------------------------------------------------------------------
# synthetic inflow turbulence on a nest child (#268), two-way (#418/#419)
# ---------------------------------------------------------------------------

using Aeolus.Transition: InflowTurbulence, added_mass_flux, inflow_perturbation, feedback_exclude_width
using Aeolus.BoundaryConditions: open_boundary_states

mn_inflow(; feedback = "two_way") = mn_cfg(MN_HEAD * "[[domain]]\nid = \"outer\"\n" * mn_child(; feedback = feedback) *
                                           "[domain.inflow_turbulence]\nenabled = true\nstatistics_interval = 4.0\n")

@testset "inflow turbulence on a nest child: the parent's statistics, no added mass, conservation ($fb)" for fb in ("one_way", "two_way")
    r = Config.build(SimulationDriverType(), mn_inflow(; feedback = fb))
    m = r.model
    c = m.domains[2].state
    it = only(filter(!isnothing, m.nest.inflow))
    @test it isa InflowTurbulence && all(st -> st.data === it, open_boundary_states(c))
    @test it.parent === m.domains[1].state                              # statistics measured on the parent
    h = m.nest.hierarchy
    I0 = composite_integral(h, :θ)
    n0 = first(it.faces).stats.nsamples
    for _ in 1:6
        step!(m, 4.0)
    end
    @test first(it.faces).stats.nsamples >= n0 + 5                      # folded at every window opening
    @test maximum(a.relative for a in added_mass_flux(it)) <= 1e-12     # no added mass flux
    @test any(f -> any(!iszero, inflow_perturbation(it, f.side)[1]), it.faces)
    @test max_divergence(c) < 1e-9
    L = nest_link(m, "inner")
    if fb == "two_way"
        need = ceil(Int, feedback_exclude_width(it) / 100.0 - 1e-9)
        @test L.twoway.exclude_width >= need                            # the feedback skips the inflow band
        @test L.nest.ledger[:mass] < 1e-12                              # the interface fluxes agree without the synthetic part (#418)
        rel = abs(composite_integral(h, :θ) - I0) / abs(I0)
        @info "two-way nest with inflow turbulence: composite θ relative change over 6 windows $rel"
        @test rel < 1e-13
    end
end

@testset "inflow turbulence: config rules, and a bit-identical resume with the generator" begin
    keyed(e) = Dict(p.key => p.message for p in e.problems)
    e = mn_error(() -> mn_cfg(MN_HEAD * "[inflow_turbulence]\nenabled = true\n[[domain]]\nid = \"outer\"\n" * mn_child() *
                                  "[domain.inflow_turbulence]\nenabled = true\n"))
    @test occursin("the root has none", keyed(e)["domain[1].inflow_turbulence.enabled"])
    e = mn_error(() -> mn_cfg(MN_HEAD * "[[domain]]\nid = \"outer\"\n" * mn_child() *
                                  "[domain.inflow_turbulence]\nenabled = true\nstatistics = \"parametrized\"\n"))
    @test occursin("parent_resolved", keyed(e)["domain[2].inflow_turbulence.statistics"])
    cfg = mn_cfg(MN_HEAD * "[[domain]]\nid = \"outer\"\n" * mn_child() *
                 "[domain.inflow_turbulence]\nenabled = true\nstatistics_interval = 4.0\n";
                 overrides = ["checkpoint.every_steps=3"])
    full = mn_run(cfg)
    root = mktempdir()
    stop = (r, n) -> n == 4 && throw(InterruptException())
    @test_throws InterruptException run_config(cfg; root = root, console = false, repo = mktempdir(), callback = stop)
    folder = only(readdir(root; join = true))
    resumed = run_config(load_run_config(folder); root = root, console = false, repo = mktempdir(), resume = folder)
    a = state_fields(full.obj); b = state_fields(resumed.obj)
    @test keys(a) == keys(b) && all(k -> a[k] == b[k], keys(a))
end

# ---------------------------------------------------------------------------
# exchanged fields (the PALM rule) and the subgrid handover
# ---------------------------------------------------------------------------

@testset "exchanged fields: the PALM rule, naming both domains" begin
    keyed(e) = Dict(p.key => p.message for p in e.problems)
    # (a --set override applies to every domain: the run-level sections are written out)
    runlevel(sec) = replace(mn_two(), "[[domain]]\nid = \"outer\"\n" => sec * "[[domain]]\nid = \"outer\"\n"; count = 1)
    moist = mn_cfg(runlevel("[dynamics]\nmoisture = true\n"))                # both moist: fine
    @test length(config_domains(moist)) == 2
    e = mn_error(() -> mn_cfg(runlevel("[dynamics]\nmoisture = true\n") * "[domain.dynamics]\nmoisture = false\n"))
    msg = keyed(e)["domain[2].dynamics.moisture"]
    @test occursin("\"inner\" carries no qv", msg) && occursin("\"outer\" does", msg) && occursin("PALM", msg)
    e = mn_error(() -> mn_cfg(mn_two() * "[domain.dynamics]\nmoisture = true\n"))
    msg = keyed(e)["domain[2].dynamics.moisture"]
    @test occursin("\"inner\" carries qv", msg) && occursin("dynamic driver", msg)
    # a tracer of the parent that the child does not carry
    e = mn_error(() -> mn_cfg(MN_HEAD * "[tracer]\nenabled = true\n[[domain]]\nid = \"outer\"\n" * mn_child() *
                                  "[domain.tracer]\nenabled = false\n"))
    @test occursin("carries no tracer", keyed(e)["domain[2].tracer.enabled"])
end

@testset "subgrid handover: the models may differ per domain; TKE on one side only; the smoothness warning" begin
    # a TKE child in a Smagorinsky parent (and the reverse): runs, its e is
    # zero-gradient at the nest faces, nothing of e is exchanged
    for (pm, cm) in (("smagorinsky", "tke"), ("tke", "smagorinsky"))
        text = replace(mn_two(), "[[domain]]\nid = \"outer\"\n" => "[sgs]\nmodel = \"$pm\"\n[[domain]]\nid = \"outer\"\n"; count = 1)
        cfg = mn_cfg(text * "[domain.sgs]\nmodel = \"$cm\"\n")
        @test MNC.sgs_closure(Config.build(SimulationDriverType(), cfg).model.domains[2]) isa
              (cm == "tke" ? Aeolus.PrognosticSGS.TKEClosure : Aeolus.Operators.StaticSmagorinsky)
        r = Config.build(SimulationDriverType(), cfg)
        m = r.model
        for _ in 1:4
            step!(m, 4.0)
        end
        tke = cm == "tke" ? m.domains[2] : m.domains[1]
        e = MNC.sgs_closure(tke).e
        @test all(isfinite, interior_view(e)) && minimum(interior_view(e)) >= 0
        if cm == "tke"
            g = m.domains[2].state.mesh
            Aeolus.BoundaryConditions.fill_halos!(e, Aeolus.BoundaryConditions.passive_scalar_bcs(m.domains[2].state.θ_bcs))
            @test e.data[g.Hx, g.Hy+3, g.Hz+2] == e.data[g.Hx+1, g.Hy+3, g.Hz+2]          # west nest face: zero gradient
            @test e.data[g.Hx+g.Nx+1, g.Hy+3, g.Hz+2] == e.data[g.Hx+g.Nx, g.Hy+3, g.Hz+2]
        end
        @test all(d -> all(isfinite, interior_view(d.state.u)), m.domains)
    end
    # the warning: a parent much smoother than the child, no inflow turbulence
    fine = mn_two(; size = "[32, 32, 8]", refinement = "[4, 4, 2]")
    @test isempty(SimulationDriver_warn(mn_cfg(mn_two())))
    w = SimulationDriver_warn(mn_cfg(fine))
    @test length(w) == 1 && occursin("\"inner\" refines its parent \"outer\" by 4", only(w))
    @test_logs (:warn, r"refines its parent \"outer\" by 4") match_mode = :any Config.build(SimulationDriverType(), mn_cfg(fine))
    @test isempty(SimulationDriver_warn(mn_cfg(fine * "[domain.inflow_turbulence]\nenabled = true\n")))
end

# ---------------------------------------------------------------------------
# mixed cores: a Boussinesq domain and an anelastic one, either way round
# ---------------------------------------------------------------------------

using Aeolus.Anelastic: AnelasticSimulation, isentropic_reference, constraint_weights

function mn_core(kind, mesh, bcs, θbcs)
    kind === :boussinesq ? BoussinesqSimulation(mesh; θ₀ = 300.0, bcs = bcs, θ_bcs = θbcs) :
        AnelasticSimulation(mesh; ref = isentropic_reference(mesh; θ_const = 300.0, g_accel = 9.81), bcs = bcs, θ_bcs = θbcs)
end

# a 3.2 km deep parent and a 1.6 km deep child (ρ̄ falls by 17 % over it)
function mn_mixed(pk, ck; feedback = :two_way, qs = (), adaptive = false)
    pm = RegularGrid(; size = (16, 16, 16), extent = (3200.0, 3200.0, 3200.0))
    nest = nested_grid(pm, ((800.0, 2400.0), (800.0, 2400.0), (0.0, 1600.0)), 2)
    p = mn_core(pk, pm, FieldBoundaryConditions(:periodic, :periodic, :wall), FieldBoundaryConditions(:periodic, :periodic, :noflux))
    cb = nest_bcs(nest.child_mesh; bottom = :wall)
    c = mn_core(ck, nest.child_mesh, cb, cb)
    for s in (p, c)
        set!(s.θ, (x, y, z) -> 300.0 + 0.003z + 0.3sin(2π * x / 1600))
        set!(s.u, (x, y, z) -> 2.0 + 0.5cos(2π * z / 3200))
        set!(s.v, (x, y, z) -> 0.7)
    end
    scheme = ck === :anelastic ? :euler : :rk3                   # the anelastic RK3 step takes no open faces
    cfl = adaptive ? CFLConstraints(; C_advective = 0.5, dt_max = 30.0) : nothing
    dp = with_budgets(Domain(p; id = "p", scheme = scheme, cfl = cfl, components = (CentredAdvection(),)), qs)
    dc = with_budgets(Domain(c; id = "c", scheme = scheme, cfl = cfl, components = (CentredAdvection(),)), qs)
    return nested_model((dp, dc), [NestLink(2, 1, nest; feedback = feedback)]), nest
end

# the cell density weights of a core (ρ₀ of the Boussinesq one)
mn_W(sim, ρ0) = sim isa BoussinesqSimulation ? fill(ρ0, size(sim.θ.data, 3)) : constraint_weights(sim)[1]

@testset "mixed cores ($pk parent, $ck child): the boundary data carry the parent's mass flux" for (pk, ck) in ((:anelastic, :boussinesq), (:boussinesq, :anelastic))
    m, nest = mn_mixed(pk, ck; feedback = :one_way)
    step!(m, 10.0)
    L = nest_link(m, "c")
    @test L.conv !== nothing
    p = m.domains[1].state; c = m.domains[2].state
    ρ0 = L.conv.ρ0
    Wp = mn_W(p, ρ0); Wc = mn_W(c, ρ0)
    pm = p.mesh; cm = c.mesh
    # the west interface plane: parent face (ix) against the child's boundary faces in B(t_end)
    ub = L.s_new.u.data
    worst = 0.0; Fp = 0.0; Fc = 0.0; naive = 0.0
    for K in 1:nest.nz, J in 1:nest.ny
        Kp = nest.iz + K - 1; Jp = nest.iy + J - 1
        fp = Wp[pm.Hz + Kp] * p.u.data[pm.Hx + nest.ix, pm.Hy + Jp, pm.Hz + Kp] * pm.dy * pm.dz
        fc = 0.0; fv = 0.0
        for k in (K - 1) * nest.rz + 1:K * nest.rz, j in (J - 1) * nest.ry + 1:J * nest.ry
            fc += Wc[cm.Hz + k] * ub[cm.Hx + 1, cm.Hy + j, cm.Hz + k] * cm.dy * cm.dz
            fv += Wp[pm.Hz + Kp] * ub[cm.Hx + 1, cm.Hy + j, cm.Hz + k] * cm.dy * cm.dz
        end
        worst = max(worst, abs(fc - fp) / abs(fp))
        Fp += fp; Fc += fc; naive += fv
    end
    ratio = maximum(abs(Wp[pm.Hz + nest.iz + K - 1] / Wc[cm.Hz + 2K] - 1) for K in 1:nest.nz)
    @info "mixed cores ($pk → $ck): west-plane mass flux, worst patch relative error $worst; the density ratios differ from 1 by up to $ratio"
    @test worst < 1e-12                       # every parent face patch: the same mass flux on both sides
    @test abs(Fc - Fp) <= 1e-12 * abs(Fp)
    @test ratio > 0.01                        # (and the conversion is not a no-op: ρ̄/ρ₀ ≠ 1 there)
end

@testset "mixed cores ($pk parent, $ck child), two-way: the composite heat and the interface mass flux are conserved" for (pk, ck) in ((:anelastic, :boussinesq), (:boussinesq, :anelastic))
    m, nest = mn_mixed(pk, ck)
    h = m.nest.hierarchy
    I0 = composite_integral(h, :θ)
    for _ in 1:6
        step!(m, 10.0)
    end
    rel = abs(composite_integral(h, :θ) - I0) / abs(I0)
    L = nest_link(m, "c")
    @info "mixed cores ($pk → $ck) two-way: composite ∫Wθ relative change over 6 windows $rel; interface mass-flux imbalance $(L.nest.ledger[:mass]) (before the correction $(L.nest.ledger[:mass_before_correction]))"
    @test rel < 1e-13
    @test L.nest.ledger[:mass] < 1e-12
    @test all(d -> all(isfinite, interior_view(d.state.θ)), m.domains)
    # the parent sees the child (the feedback converted the velocity back to its own density)
    @test max_divergence(m.domains[2].state) < (ck === :boussinesq ? 1e-9 : Inf)
end

@testset "mixed cores in a config: an anelastic root with a Boussinesq child, and the reverse" begin
    text = replace(mn_two(), "[[domain]]\nid = \"outer\"\n" => "[core]\nmodel = \"anelastic\"\n[[domain]]\nid = \"outer\"\n"; count = 1) *
           "[domain.core]\nmodel = \"boussinesq\"\n"
    r = Config.build(SimulationDriverType(), mn_cfg(text))
    m = r.model
    @test MNC.core(m.domains[1]) === :anelastic && MNC.core(m.domains[2]) === :boussinesq
    I0 = composite_integral(m.nest.hierarchy, :θ)
    for _ in 1:3
        step!(m, 4.0)
    end
    @test abs(composite_integral(m.nest.hierarchy, :θ) - I0) / abs(I0) < 1e-13
    rev = mn_cfg(mn_two() * "[domain.core]\nmodel = \"anelastic\"\n"; overrides = ["timestep.scheme=\"euler\""])
    m2 = Config.build(SimulationDriverType(), rev).model
    for _ in 1:3
        step!(m2, 4.0)
    end
    @test MNC.core(m2.domains[2]) === :anelastic && all(isfinite, interior_view(m2.domains[2].state.θ))
end

@testset "both cores: budgets close and adaptive windows run on a mixed nest ($pk parent, $ck child)" for (pk, ck) in ((:anelastic, :boussinesq), (:boussinesq, :anelastic), (:anelastic, :anelastic))
    m, nest = mn_mixed(pk, ck; qs = (:energy, :momentum_x), adaptive = true)
    ns = Int[]
    for _ in 1:4
        _, Δt, r = step_adaptive!(m)
        push!(ns, nest_link(m, "c").stats.n_last)
    end
    @test all(>=(2), ns)
    for d in m.domains, q in (:energy, :momentum_x)
        meas = budgets(d).measured[q]
        sb = stage_budget(meas)
        content = abs(Aeolus.BudgetSystem.measure(meas.meter, q === :energy ? d.state.θ : d.state.u))
        @test abs(sb.closure) <= 1e-12 * content
    end
    @test nest_exchange(m.domains[2], :energy) != 0 && nest_exchange(m.domains[1], :energy) != 0
end
