# The `berlin` run driver (CFG-c, #358): the Berlin case configured from
# TOML (examples/urban/core/config_schema.jl, examples/urban/configs/).
#
# Acceptance tests of the issue covered here:
#   * field for field: CaseConfig(load_run_config("berlin_<s>.toml")) equals
#     the fixture captured from the Julia-built scenarios before SCENARIOS
#     was deleted (test/data/config_berlin/fixture.toml), floats bitwise,
#     for the case and the nest parent it runs;
#   * one source of truth: CaseConfig and ParentConfig fields map one to
#     one onto the schema's options, and their defaults are the schema's;
#   * bit-identical to the Julia-built case: berlin_smoke.toml run 12 steps
#     through run_config gives the fixture's state digests (child, tracers,
#     surface, nest parent);
#   * round trip, bit-identical under ESTIMATE FFTW, nest on;
#   * errors: an unknown provider, a parent refinement of 2 integers, an
#     incompatible pair, an unknown per-coupling option;
#   * the CLI: `aeolus run berlin_smoke.toml` leaves a completed run folder,
#     and the run's ledgers close.

using Test
using TOML
using Printf
using Aeolus
using Aeolus.Config
using Aeolus.IO: read_run_toml, diagnostics_path

isdefined(Main, :BerlinCase) ||
    include(joinpath(@__DIR__, "..", "examples", "urban", "BerlinCase.jl"))
const BCC = Main.BerlinCase
include(joinpath(@__DIR__, "support", "berlin_fixture.jl"))
include(joinpath(@__DIR__, "config_roundtrip_harness.jl"))

const BERLIN_FIXTURE = TOML.parsefile(joinpath(@__DIR__, "data", "config_berlin", "fixture.toml"))
berlin_toml(s) = joinpath(BCC.CONFIG_DIR, "berlin_$s.toml")

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

"Paths where `a` and `b` (TOML tables) differ; floats compared by their bits."
function table_diff(a, b, path = "", out = String[])
    if a isa AbstractDict && b isa AbstractDict
        for k in sort!(collect(union(keys(a), keys(b))))
            (haskey(a, k) && haskey(b, k)) || (push!(out, "$path.$k (missing)"); continue)
            table_diff(a[k], b[k], "$path.$k", out)
        end
    elseif a isa AbstractVector && b isa AbstractVector
        length(a) == length(b) || (push!(out, "$path (length)"); return out)
        for i in eachindex(a)
            table_diff(a[i], b[i], "$path[$i]", out)
        end
    elseif a isa AbstractFloat && b isa AbstractFloat
        reinterpret(UInt64, Float64(a)) == reinterpret(UInt64, Float64(b)) || push!(out, "$path: $a ≠ $b")
    else
        isequal(a, b) || push!(out, "$path: $(repr(a)) ≠ $(repr(b))")
    end
    return out
end
# through TOML text, as the fixture was stored
as_toml(x) = TOML.parse(sprint(TOML.print, x))

@testset "Config berlin: the four scenarios equal the Julia-built ones, field for field" begin
    for s in ("summer", "winter", "smoke", "mitte_sample")
        cfg = load_run_config(berlin_toml(s))
        cc = BCC.CaseConfig(cfg)
        d_case = table_diff(BERLIN_FIXTURE[s]["case"], as_toml(normalise(cc)), "case")
        d_par = table_diff(BERLIN_FIXTURE[s]["parent"], as_toml(normalise(BCC.nest_parent_config(cc))), "parent")
        isempty(d_case) && isempty(d_par) || @info "berlin_$s.toml differs from the fixture" d_case d_par
        @test isempty(d_case)
        @test isempty(d_par)
        # the scenario wrapper loads the same file
        @test isempty(table_diff(as_toml(normalise(BCC.CaseConfig(Symbol(s)))), as_toml(normalise(cc))))
    end
    # the alias keeps its own name; everything else is the summer case
    a, su = BCC.CaseConfig(:alexanderplatz), BCC.CaseConfig(:summer)
    @test a.name === :alexanderplatz && su.name === :summer
    a.name = :summer
    @test isempty(table_diff(as_toml(normalise(a)), as_toml(normalise(su))))
    # winter is a minimal diff of the summer file (extends), resolved by bare name
    w = load_run_config(berlin_toml("winter"))
    @test config_sources(w)["files"] == [berlin_toml("winter"), berlin_toml("summer")]
    @test config_origin(w, "grid.size") == berlin_toml("summer")
    @test config_origin(w, "clock.date") == berlin_toml("winter")
    @test_throws ErrorException BCC.CaseConfig(:autumn)
end

@testset "Config berlin: one source of truth for the defaults" begin
    sch = BCC.berlin_schema()
    # every CaseConfig field is made of schema options, each used once
    @test collect(keys(BCC.CASE_FIELDS)) == collect(fieldnames(BCC.CaseConfig))
    used = String[]
    for keys_ in values(BCC.CASE_FIELDS), k in keys_
        s, o = split(k, '.')
        sec = sch.sections[findfirst(x -> x.name == s, sch.sections)]
        if o == "*"                   # the whole section ([physics]: but its options)
            append!(used, ["$s.$(x.key)" for x in sec.options if !(s == "physics" && x.key == "options")])
        else
            @test any(x -> x.key == o, sec.options)
            push!(used, k)
        end
    end
    @test allunique(used)
    common = Set(["run.driver", "run.seed", "run.root", "run.allow_uncommitted", "logging.console_level", "logging.file_level",
                  "logging.flush_interval", "logging.diagnostics_cadence"])
    case_opts = ["$(s.name).$(o.key)" for s in sch.sections for o in s.options
                 if !(s.name in ("fftw", "checkpoint")) && !("$(s.name).$(o.key)" in common)]
    @test sort(used) == sort(case_opts)
    # berlin_summer.toml is the reference copy: every Berlin option is
    # written out (the common [fftw], [logging], [checkpoint] and run.root /
    # allow_uncommitted are left to their defaults, which environment
    # variables fill), every coupling option and every parent option
    f = TOML.parsefile(berlin_toml("summer"))
    left = Set(["run.root", "run.allow_uncommitted"])
    missing_ = ["$(s.name).$(o.key)" for s in sch.sections for o in s.options
                if !(s.name in ("fftw", "logging", "checkpoint")) && !("$(s.name).$(o.key)" in left) &&
                   !haskey(get(f, s.name, Dict()), o.key)]
    for C in BCC.REGISTRY, k in keys(BCC.default_options(C))
        haskey(get(f["physics"]["options"], String(BCC.coupling_name(C)), Dict()), String(k)) ||
            push!(missing_, "physics.options.$(BCC.coupling_name(C)).$k")
    end
    for o in BCC.parent_schema_section().options
        haskey(f["physics"]["options"]["nesting"]["parent"], o.key) || push!(missing_, "parent.$(o.key)")
    end
    @test isempty(missing_)
    # ParentConfig: every field but the child is a key of the parent schema
    ps = BCC.parent_schema_section()
    @test collect(keys(BCC.PARENT_FIELDS)) == collect(fieldnames(BCC.ParentConfig))[2:end]
    @test sort(collect(values(BCC.PARENT_FIELDS))) == sort([o.key for o in ps.options])
    # the struct defaults are the schema's
    c = BCC.CaseConfig()
    dflt = BCC.default_run_config()
    @test c.size == dflt.grid.size && c.ν === dflt.dynamics.nu && c.θ_top === :initial_gradient
    @test c.cadence == Dict(:radiation => 600.0, :soil => 30.0, :output => 1800.0)   # the driver's output cadence
    @test c.name === :alexanderplatz && c.physics == Dict() && c.options == Dict()
    p = BCC.ParentConfig(c)
    @test p.extent == (6000.0, 6000.0, 3000.0) && p.refinement == (2, 2, 4) && isnan(p.z_g)
    @test p.Q_surface isa BCC.ConstantForcing && p.Q_surface(3600.0) == 0.0
    @test p.u_init === :geostrophic && p.θ_init === nothing
    # a field that does not exist is an error, not silently ignored
    @test_throws ArgumentError BCC.CaseConfig(; nstep = 3)
    @test_throws ArgumentError BCC.ParentConfig(c; refinment = (2, 2, 2))
    # Julia kwargs on a scenario merge options per coupling over the file's
    # (the smoke file's nest parent survives)
    sm = BCC.CaseConfig(:smoke; options = Dict(:pollutant => (tau = 60.0,)))
    @test sm.options[:pollutant] == (tau = 60.0,)
    @test sm.options[:nesting].parent.refinement == (4, 4, 8)
end

@testset "Config berlin: [physics] and per-coupling options" begin
    smoke = berlin_toml("smoke")
    ld(ovr...) = BCC.CaseConfig(load_run_config(smoke; overrides = String[ovr...]))
    # the type of an option is the type of its default
    c = ld("physics.options.geostrophic.mode=\"nudging\"", "physics.options.geostrophic.U_g=3",
           "physics.options.geostrophic.τ=900.0",            # a Julia name as a key
           "physics.options.surface_exchange.z0m={3 = 0.2}",   # merged over the class table
           "physics.options.nesting.z_anterp=12.5", "physics.options.sgs_smagorinsky.diapycnal=0.1",
           "physics.options.pollutant.stacks=[[1.0, 2.0, 3.0]]", "physics.options.ground_soil.classes=[1, 3, 5]")
    g = c.options[:geostrophic]
    @test g.mode === :nudging && g.U_g === 3.0 && g.τ === 900.0
    o = BCC.coupling_options(BCC.GeostrophicCoupling, c)
    @test o.V_g == 0.0 && o.sponge_to_geostrophic                     # the rest: the defaults
    z = c.options[:surface_exchange].z0m
    @test z[0x03] == 0.2 && z[0x01] == 0.05 && length(z) == 6
    @test c.options[:nesting].z_anterp === 12.5 && length(c.options[:nesting]) == 2
    @test c.options[:nesting].parent == (extent = (1000.0, 1000.0, 500.0), refinement = (4, 4, 8), vertical = :uniform)
    @test c.options[:sgs_smagorinsky].diapycnal === 0.1
    @test c.options[:pollutant].stacks == [(1.0, 2.0, 3.0)]
    @test c.options[:ground_soil].classes === (0x01, 0x03, 0x05)
    @test ld("physics.options.surface_exchange.z0m=\"dynamic\"").options[:surface_exchange].z0m === :dynamic
    # options equal to the defaults are not stored
    @test !haskey(ld("physics.options.geostrophic.U_g=1.0").options, :geostrophic)
    # [physics]: slot = provider, addon = true/false; the registry default is not stored
    c = ld("physics.sgs=\"smagorinsky\"", "physics.pollutant=false", "physics.microphysics=\"kessler\"")
    @test c.physics == Dict{Symbol,Any}(:sgs => :smagorinsky, :pollutant => false)
    @test BCC.SGSSmagorinsky in BCC.selected_couplings(c) && !(BCC.PollutantCoupling in BCC.selected_couplings(c))
    # the parent's function-valued fields from their TOML tables
    c = ld("physics.options.nesting.parent.surface=\"prescribed\"",
           "physics.options.nesting.parent.q_surface={kind = \"series\", t = [0.0, 3600.0], value = [0.0, 0.2]}",
           "physics.options.nesting.parent.theta_init={kind = \"table\", z = [0.0, 1000.0], values = [290.0, 293.0]}",
           "physics.options.nesting.parent.u_init={kind = \"constant\", value = 2.5}")
    p = BCC.nest_parent_config(c)
    @test p.surface === :prescribed && p.Q_surface(1800.0) ≈ 0.1 && p.Q_surface(7200.0) == 0.2
    @test p.E_surface(0.0) == 0.0
    @test p.θ_init(500.0) ≈ 291.5 && p.θ_init(-5.0) == 290.0 && p.u_init === 2.5
    @test BCC.ParentConfig(c; u_init = BCC.ProfileTable([0.0, 100.0], [1.0, 3.0])).u_init(50.0) == 2.0
    # an arbitrary Julia function only through the escape hatch: not replayable
    cfg = load_run_config(smoke)
    f = t -> 1e-3 * sin(t)
    cfg2 = with(cfg, "physics.options" => Dict("nesting" => Dict("parent" => Dict("q_surface" => f))))
    @test nonreplayable_keys(cfg2) == ["physics.options"]
    @test BCC.nest_parent_config(BCC.CaseConfig(cfg2)).Q_surface === f
    @test BCC.toml_literal(BCC.SeriesForcing([0.0, 1.0], [2.0, 3.0])) ==
          "{kind = \"series\", t = [0.0, 1.0], value = [2.0, 3.0]}"
    @test_throws ArgumentError BCC.toml_form(f)
    # [output]: netcdf in path; "none" → no file; the run folder's output/ under run_config
    c = BCC.CaseConfig(load_run_config(berlin_toml("summer")))
    @test c.output_path == normpath(joinpath(BCC.URBAN_DIR, "output", "alexanderplatz.nc"))
    c = BCC.CaseConfig(load_run_config(berlin_toml("summer"); overrides = ["output.path=\"none\""]); run_output = "/tmp/x")
    @test c.output_path == "/tmp/x/alexanderplatz.nc"
    @test ld().output_path === nothing
    # the cadences the couplings declare
    @test Set(BCC.declared_cadences(BCC.RadiositySun)) == Set([:radiation])
    @test :surface in BCC.declared_cadences(BCC.SurfaceExchangeMOST)
    @test ld("cadence.surface=2.0").cadence[:surface] == 2.0
    @test !haskey(ld().cadence, :surface)
end

@testset "Config berlin: errors name the key and the allowed values" begin
    smoke = berlin_toml("smoke")
    err(ovr...) = berlin_error(() -> load_run_config(smoke; overrides = String[ovr...]))
    # an unknown provider lists the registered ones
    e = err("physics.microphysics=\"p4\"")
    @test only(e.problems).key == "physics.microphysics" && occursin("\"kessler\"", only(e.problems).message)
    e = err("physics.sgs=\"tk\"")
    @test occursin("\"smagorinsky\"", errtext(e)) && occursin("\"tke\"", errtext(e))
    # the parent's own schema
    e = err("physics.options.nesting.parent.refinement=[2, 2]")
    @test only(e.problems).key == "physics.options"
    @test occursin("nesting.parent.refinement", errtext(e)) && occursin("array of 3 integers", errtext(e))
    e = err("physics.options.nesting.parent.vertikal=\"uniform\"")
    @test occursin("did you mean \"vertical\"", errtext(e))
    e = err("physics.options.nesting.parent.q_surface={kind = \"series\", t = [1.0, 0.0], value = [0.0, 1.0]}")
    @test occursin("strictly increasing", errtext(e))
    # a declared incompatibility, with its reason
    e = err("physics.core=\"anelastic\"")
    reason = BCC.incompatible_reason(BCC.coupling_type(:core_anelastic))
    @test !isempty(reason) && occursin(reason, errtext(e)) && only(e.problems).key == "physics"
    # an unknown per-coupling option lists that coupling's options
    e = err("physics.options.geostrophic.Ug=2.0")
    @test occursin("geostrophic.Ug: unknown option", errtext(e))
    @test occursin("U_g, V_g, mode, τ, sponge_to_geostrophic", errtext(e)) && occursin("did you mean \"U_g\"", errtext(e))
    e = err("physics.options.geostrophic.U_g=\"fast\"")
    @test occursin("geostrophic.U_g: expected a number", errtext(e))
    e = err("physics.options.geostrofic.U_g=1.0")
    @test occursin("no coupling named \"geostrofic\"", errtext(e)) && occursin("did you mean \"geostrophic\"", errtext(e))
    e = err("physics.options.nesting.z_anterp=\"roof\"")
    @test occursin("\"canopy\" or a height", errtext(e))
    # θ top: a name or a number
    @test BCC.CaseConfig(load_run_config(smoke; overrides = ["dynamics.theta_top=0.003"])).θ_top === 0.003
    @test occursin("theta_top", errtext(err("dynamics.theta_top=\"lid\"")))
    # no checkpoint/restart for the Berlin case yet
    @test occursin("no checkpoint/restart", errtext(err("checkpoint.every_steps=5")))
    # five independent problems are all reported
    e = err("physics.microphysics=\"p4\"", "grid.size=[16, 16]", "clock.dt=-1.0",
            "physics.options.nesting.parent.refinement=[2, 2]", "boundaries.lateral=\"closed\"")
    @test length(e.problems) == 5
end

@testset "Config berlin: berlin_smoke.toml runs bit-identical to the Julia-built case, round trip, ledgers" begin
    t0 = time()
    r = roundtrip_check(berlin_toml("smoke"); nsteps = 12, nsteps_overrides = n -> ["clock.nsteps=$n"])
    wall = time() - t0
    @test r.fields_equal
    isempty(r.fields_differing) || @info "fields differing in the round trip" r.fields_differing
    @test r.diagnostics_equal && r.config_equal
    case = r.runs[1].obj.case
    st = BCC.case_state(case)
    # the nest is on: the parent's fields are compared too
    @test :parent_θ in keys(st) && :parent_qv in keys(st) && :qv in keys(st)
    # bit-identical to build_case(CaseConfig(:smoke)) on master (the fixture's digests; ESTIMATE)
    want = BERLIN_FIXTURE["smoke_digests"]
    got = Dict(string(k) => digest(v) for (k, v) in pairs(st))
    @test sort(collect(keys(got))) == sort(collect(keys(want)))
    bad = sort!([k for k in keys(want) if get(got, k, "") != want[k]])
    isempty(bad) || @info "state digests differing from the Julia-built smoke case" bad
    @test isempty(bad)
    # the support copy of the state set (the fixture's capture) is the driver's
    @test collect(keys(berlin_state(case))) == collect(keys(st))
    # nsteps × dt = 24 s simulated (the default :adaptive stepper takes 15
    # steps to get there), completed, every step a diagnostics row
    rec = read_run_toml(r.runs[1].folder.path)
    @test case.sim.t == 24.0 && !case.clock.blowup
    @test rec["config"]["run"]["driver"] == "berlin" && rec["replayable"] == true
    @test length(readlines(diagnostics_path(r.runs[1].folder))) == case.clock.n + 1   # header + a row per step
    # the ledgers close to the smoke test's tolerances
    rows = BCC.surface_budget(case.ledger)
    w = BCC.water_budget(case.ledger)
    hb = BCC.heat_budget(case.ledger)
    @test maximum(abs(x.closure) for x in rows) < 1e-6        # W/m²
    @test abs(w.residual) < 1e-12                              # kg/m²
    @test abs(hb.closure / hb.elapsed) <= 1e-10                # W/m²
    @info @sprintf("berlin_smoke.toml via run_config: 2 runs in %.1f s; surface closure max %.1e W/m², water residual %.1e kg/m², air-heat closure %.1e W/m²; domain Rn %.2f H %.2f LE %.2f W/m²",
                   wall, maximum(abs(x.closure) for x in rows), w.residual, hb.closure / hb.elapsed,
                   rows[end].Rn, rows[end].H, rows[end].LE)
end

@testset "Config berlin: aeolus run berlin_smoke.toml" begin
    root = mktempdir()
    rc = redirect_stdout(devnull) do
        Aeolus.main(["run", berlin_toml("smoke"), "--root", root, "--allow-uncommitted",
                     "--set", "fftw.planner=\"estimate\"", "--set", "clock.nsteps=2"])
    end
    @test rc == 0
    runs = readdir(root; join = true)
    @test length(runs) == 1
    rec = read_run_toml(only(runs))
    @test rec["session"][end]["status"] == "completed"
    @test rec["config"]["clock"]["nsteps"] == 2 && rec["config"]["run"]["name"] == "smoke"
    # a config error exits 2 before anything is written
    root2 = mktempdir()
    rc = redirect_stderr(devnull) do
        Aeolus.main(["run", berlin_toml("smoke"), "--root", root2, "--set", "physics.microphysics=\"p4\""])
    end
    @test rc == 2 && isempty(readdir(root2))
end

@testset "Config berlin: a fresh process loads the berlin driver on first use" begin
    # BerlinCase.jl is included while the config is being loaded, so its
    # validators and checks are newer than the loader's world (the
    # invokelatest calls in Aeolus.Config); without them `aeolus check`
    # failed with a world-age MethodError in a fresh process
    root = pkgdir(Aeolus)
    aeolus(args...) = VERSION >= v"1.12" ?
        `$(Base.julia_cmd()) --startup-file=no --project=$root -m Aeolus $args` :
        `$(Base.julia_cmd()) --startup-file=no --project=$root $(joinpath(root, "bin", "aeolus")) $args`
    err = IOBuffer()
    p = run(pipeline(ignorestatus(aeolus("check", berlin_toml("smoke"), "--set", "physics.microphysics=\"p4\""));
                     stdout = devnull, stderr = err))
    @test p.exitcode == 2
    msg = String(take!(err))
    @test occursin("physics.microphysics", msg) && occursin("kessler", msg)
end
