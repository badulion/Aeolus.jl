# The Berlin case configured from TOML (CFG-c, #358): the urban site of the
# model (src/components/urban/site/config_schema.jl, EX-a #438, ONE-e #454) and the
# configs examples/urban/berlin_*.toml.
#
# Acceptance tests of the issue covered here:
#   * field for field: site_options(load_run_config("berlin_<s>.toml")) equals
#     the fixture captured from the Julia-built scenarios before SCENARIOS
#     was deleted (test/data/config_berlin/fixture.toml), floats bitwise,
#     for the case and the nest parent it runs;
#   * one source of truth: the site's options and ParentConfig fields map one to
#     one onto the schema's options, and their defaults are the schema's;
#   * bit-identical to the Julia-built case: berlin_smoke.toml run 12 steps
#     through run_config gives the fixture's state digests (child, tracers,
#     surface, nest parent);
#   * round trip, bit-identical under ESTIMATE FFTW, nest on;
#   * errors: an unknown provider, a parent refinement of 2 integers, an
#     incompatible pair, an unknown per-coupling option;
#   * the CLI: `aeolus run berlin_smoke.toml` leaves a completed run folder,
#     and the run's ledgers close;
#   * the case reports through the diagnostics registry (LOG-g, #338):
#     every diagnostic evaluated at every step leaves the state bit for bit
#     unchanged; the CSV's ledger, timing, time-stepper and dt columns equal
#     the case's own numbers; the log events are structured and carry the
#     same values.

using Test
using TOML
using Printf
using Aeolus
using Aeolus.Config
using Aeolus.IO: read_run_toml, diagnostics_path
using Aeolus.Fields: interior_view
using Logging

const BCC = Aeolus.Urban
include(joinpath(@__DIR__, "support", "berlin_configs.jl"))
include(joinpath(@__DIR__, "support", "berlin_fixture.jl"))
include(joinpath(@__DIR__, "config_roundtrip_harness.jl"))

const BERLIN_FIXTURE = TOML.parsefile(joinpath(@__DIR__, "data", "config_berlin", "fixture.toml"))

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
        cc = BCC.site_options(cfg)
        d_case = table_diff(BERLIN_FIXTURE[s]["case"], as_toml(normalise(cc)), "case")
        d_par = table_diff(BERLIN_FIXTURE[s]["parent"], as_toml(normalise(BCC.nest_parent_config(cc))), "parent")
        isempty(d_case) && isempty(d_par) || @info "berlin_$s.toml differs from the fixture" d_case d_par
        @test isempty(d_case)
        @test isempty(d_par)
        # site_config(path) (the tests' helper over --set overrides) loads the same file
        @test isempty(table_diff(as_toml(normalise(site_config(berlin_toml(s)))), as_toml(normalise(cc))))
    end
    # winter is a complete file of its own (no extends, ONE-b #451); its
    # difference from the summer file, stated explicitly (its ts_water_offset
    # = 5.0, set when it extended summer, is summer's value)
    w = load_run_config(berlin_toml("winter"))
    su = load_run_config(berlin_toml("summer"))
    @test config_sources(w)["files"] == [berlin_toml("winter")]
    @test config_origin(w, "grid.size") == berlin_toml("winter")
    tw = to_toml(w); ts = to_toml(su)
    @test sort!([lstrip(split(d, ':')[1], '.') for d in table_diff(ts, tw, "")]) ==
          sort!(["run.name", "domain[1].urban.scenario", "domain[1].urban.date", "domain[1].urban.utc_offset",
                 "domain[1].dynamics.theta0", "domain[1].urban.sounding", "domain[1].dynamic_driver",
                 "domain[1].urban.theta_surface", "domain[1].urban.qv0", "domain[1].urban.ts0",
                 "domain[1].urban.t_soil0"])
    @test_throws ConfigError load_run_config(berlin_toml("autumn"))
end

@testset "Config berlin: one source of truth for the defaults" begin
    sch = driver_schema("model")
    sec(n) = sch.sections[findfirst(x -> x.name == n, sch.sections)]
    # every field of the site's options is made of the model's options, each used once
    @test collect(keys(BCC.SITE_FIELDS)) == collect(keys(BCC.site_options(BCC.default_run_config())))
    used = String[]
    names = [String(BCC.component_name(C)) for C in BCC.URBAN_COMPONENTS]
    for keys_ in values(BCC.SITE_FIELDS), k in keys_
        s_, o = split(k, '.')
        if s_ == "*"                 # every urban component's section: enabled and its options (components)
            for n in names, x in sec(n).options
                (o == "enabled") == (x.key == "enabled") && push!(used, "$n.$(x.key)")
            end
        else
            @test any(x -> x.key == o, sec(s_).options)
            push!(used, k)
        end
    end
    @test allunique(used)
    # every option of the urban site's sections makes a field ([urban] enabled: the site itself)
    site = vcat(["urban.$(o.key)" for o in sec("urban").options if o.key != "enabled"],
                ["$n.$(o.key)" for n in names for o in sec(n).options])
    @test issubset(site, used)
    # berlin_summer.toml is the reference copy: every option of the urban site,
    # every component's option and every parent option is written out
    f = TOML.parsefile(berlin_toml("summer"))
    missing_ = String[]
    for o in sec("urban").options
        haskey(f["urban"], o.key) || push!(missing_, "urban.$(o.key)")
    end
    for n in names, o in sec(n).options
        haskey(get(f, n, Dict()), o.key) || push!(missing_, "$n.$(o.key)")
    end
    for o in BCC.parent_schema_section().options
        haskey(f["nesting"]["parent"], o.key) || push!(missing_, "nesting.parent.$(o.key)")
    end
    @test isempty(missing_)
    isempty(missing_) || @info "options berlin_summer.toml does not write out" missing_
    # ParentConfig: every field but the child is a key of the parent schema
    ps = BCC.parent_schema_section()
    @test collect(keys(BCC.PARENT_FIELDS)) == collect(fieldnames(BCC.ParentConfig))[2:end]
    @test sort(collect(values(BCC.PARENT_FIELDS))) == sort([o.key for o in ps.options])
    # the struct defaults are the schema's
    c = BCC.site_options(BCC.default_run_config())
    dflt = BCC.default_run_config()
    @test c.size == dflt.grid.size && c.ν === dflt.dynamics.nu && c.θ_top === :initial_gradient
    @test c.cadence == Dict(:radiation => 600.0, :soil => 30.0, :output => 1800.0)
    @test c.name === :urban && Set(keys(c.components)) == Set(BCC.component_name.(BCC.URBAN_DEFAULTS))
    @test all(C -> isequal(c.components[BCC.component_name(C)], BCC.default_options(C)), BCC.URBAN_DEFAULTS)   # isequal: NaN defaults
    # the generic site points at no input data of an example (EX-a, #438)
    @test c.landuse === nothing && c.sounding === nothing && c.static === nothing
    p = BCC.ParentConfig(c)
    @test p.extent == (6000.0, 6000.0, 3000.0) && p.refinement == (2, 2, 4) && isnan(p.z_g)
    @test p.Q_surface isa BCC.ConstantForcing && p.Q_surface(3600.0) == 0.0
    @test p.u_init === :geostrophic && p.θ_init === nothing
    # a key that does not exist is an error, not silently ignored
    @test_throws ConfigError load_run_config(berlin_toml("smoke"); overrides = ["urban.nstep=3"])
    @test_throws ArgumentError BCC.ParentConfig(c; refinment = (2, 2, 2))
    # a component's options change with an override of its section (the
    # smoke file's nest parent survives one of another component's); the
    # components are not fields of the site's options (ONE-e, #454)
    sm = site_config(berlin_toml("smoke"); overrides = ["pollutant.tau=60.0"])
    @test sm.components[:pollutant].tau == 60.0
    @test sm.components[:nesting].parent.refinement == (4, 4, 8)
    # every urban component is off until its section enables it: a model
    # config without [urban] is no urban site, and enables none of them
    bare = load_run_config(IOBuffer(""))
    @test !BCC.urban_enabled(bare) && !any(n -> bare[n].enabled, names)
end

@testset "Config berlin: the analysis buffer of the observation diagnostics (#268)" begin
    @test BCC.site_options(BCC.default_run_config()).analysis_buffer == BCC.ANALYSIS_BUFFER_DEFAULT == 140.0
    sm = site_config(berlin_toml("smoke"))
    m = RegularGrid(; size = sm.size, extent = sm.extent, halo = sm.halo)
    A = BCC.analysis_columns(sm, m)            # 31.25 m = 2 cells of 15.6 m from each nest face
    @test sm.analysis_buffer == 31.25 && size(A) == (16, 16) && count(A) == 12 * 12
    @test all(A[3:14, 3:14]) && !any(A[1:2, :]) && !any(A[:, 15:16])
    # the periodic box has no inflow face: every column counts
    pb = site_config(berlin_toml("smoke"); overrides = ["nesting.enabled=false"])
    @test all(BCC.analysis_columns(pb, m))
end

@testset "Config berlin: the components' sections and their options" begin
    smoke = berlin_toml("smoke")
    ld(ovr...) = BCC.site_options(load_run_config(smoke; overrides = String[ovr...]))
    # the form of an option is the form of its default
    c = ld("geostrophic.mode=\"nudging\"", "geostrophic.u_g=3",
           "geostrophic.tau=900.0",                    # τ: its ASCII key
           "surface_exchange.z0m={3 = 0.2}",           # merged over the class table
           "nesting.z_anterp=12.5", "sgs_smagorinsky.diapycnal=0.1",
           "pollutant.stacks=[[1.0, 2.0, 3.0]]", "ground_soil.classes=[1, 3, 5]")
    g = c.components[:geostrophic]
    @test g.mode === :nudging && g.U_g === 3.0 && g.τ === 900.0
    o = BCC.component_options(BCC.GeostrophicCoupling, c)
    @test o === g && o.V_g == 0.0 && o.sponge_to_geostrophic          # the rest: the defaults
    @test keys(o) == keys(BCC.default_options(BCC.GeostrophicCoupling))
    z = c.components[:surface_exchange].z0m
    @test z[0x03] == 0.2 && z[0x01] == 0.05 && length(z) == 6
    @test c.components[:nesting].z_anterp === 12.5
    @test c.components[:nesting].parent == (extent = (1000.0, 1000.0, 500.0), refinement = (4, 4, 8), vertical = :uniform)
    @test c.components[:pollutant].stacks == [(1.0, 2.0, 3.0)]
    @test c.components[:ground_soil].classes === (0x01, 0x03, 0x05)
    @test ld("surface_exchange.z0m=\"dynamic\"").components[:surface_exchange].z0m === :dynamic
    # a component the config does not enable has no entry; its options are its defaults
    @test !haskey(c.components, :sgs_smagorinsky)
    @test BCC.component_options(BCC.SGSSmagorinsky, c) == BCC.default_options(BCC.SGSSmagorinsky)
    # the selection: one component per urban slot, additive components on or
    # off, exactly what the sections enable
    c = ld("sgs_tke.enabled=false", "sgs_smagorinsky.enabled=true", "sgs_smagorinsky.diapycnal=0.1",
           "pollutant.enabled=false")
    @test c.components[:sgs_smagorinsky].diapycnal === 0.1
    @test BCC.SGSSmagorinsky in BCC.selected_components(c) && !(BCC.PollutantCoupling in BCC.selected_components(c))
    @test !(BCC.SGSTKE in BCC.selected_components(c))
    @test BCC.selected_components(c) == [C for C in BCC.URBAN_COMPONENTS if haskey(c.components, BCC.component_name(C))]
    # the parent's function-valued fields from their TOML tables
    c = ld("nesting.parent.surface=\"prescribed\"",
           "nesting.parent.q_surface={kind = \"series\", t = [0.0, 3600.0], value = [0.0, 0.2]}",
           "nesting.parent.theta_init={kind = \"table\", z = [0.0, 1000.0], values = [290.0, 293.0]}",
           "nesting.parent.u_init={kind = \"constant\", value = 2.5}")
    p = BCC.nest_parent_config(c)
    @test p.surface === :prescribed && p.Q_surface(1800.0) ≈ 0.1 && p.Q_surface(7200.0) == 0.2
    @test p.E_surface(0.0) == 0.0
    @test p.θ_init(500.0) ≈ 291.5 && p.θ_init(-5.0) == 290.0 && p.u_init === 2.5
    @test BCC.ParentConfig(c; u_init = BCC.ProfileTable([0.0, 100.0], [1.0, 3.0])).u_init(50.0) == 2.0
    # an arbitrary Julia function only through the escape hatch: not replayable
    cfg = load_run_config(smoke)
    f = t -> 1e-3 * sin(t)
    cfg2 = with(cfg, "nesting.parent" => Dict("q_surface" => f))
    @test !isempty(nonreplayable_keys(cfg2)) && all(k -> occursin("nesting", k), nonreplayable_keys(cfg2))
    @test BCC.nest_parent_config(BCC.site_options(cfg2)).Q_surface === f
    @test BCC.toml_literal(BCC.SeriesForcing([0.0, 1.0], [2.0, 3.0])) ==
          "{kind = \"series\", t = [0.0, 1.0], value = [2.0, 3.0]}"
    @test_throws ArgumentError BCC.toml_form(f)
    # the NetCDF file is the run's writer's ([output] netcdf in [output] path,
    # ONE-k #474): the flagship's for the renderers, none for the smoke run
    su = load_run_config(berlin_toml("summer"))
    @test su.output.netcdf == "alexanderplatz.nc" && "Ts" in su.output.variables && "H_sfc" in su.output.variables
    @test isempty(load_run_config(smoke).output.variables)
    # the periods
    @test ld("urban.surface_period=2.0").cadence[:surface] == 2.0
    @test !haskey(ld().cadence, :surface)
    @test ld().cadence == Dict(:radiation => 8.0, :soil => 4.0, :output => 12.0)
end

@testset "Config berlin: errors name the key and the allowed values" begin
    smoke = berlin_toml("smoke")
    err(ovr...) = berlin_error(() -> load_run_config(smoke; overrides = String[ovr...]))
    # a slot without a provider, or with two, names its providers
    e = err("micro_kessler.enabled=false")
    @test only(e.problems).key == "micro_kessler" && occursin(":microphysics", errtext(e)) && occursin("none is enabled", errtext(e))
    e = err("sgs_smagorinsky.enabled=true")
    @test occursin(":sgs", errtext(e)) && occursin("sgs_smagorinsky", errtext(e)) && occursin("sgs_tke", errtext(e))
    # the parent's own schema
    e = err("nesting.parent.refinement=[2, 2]")
    @test occursin("parent.refinement", errtext(e)) && occursin("array of 3 integers", errtext(e))
    e = err("nesting.parent.vertikal=\"uniform\"")
    @test occursin("did you mean \"vertical\"", errtext(e))
    e = err("nesting.parent.q_surface={kind = \"series\", t = [1.0, 0.0], value = [0.0, 1.0]}")
    @test occursin("strictly increasing", errtext(e))
    # a declared incompatibility, with its reason
    e = err("core_boussinesq.enabled=false", "core_anelastic.enabled=true")
    reason = BCC.incompatible_reason(BCC.component_type(:core_anelastic))
    @test !isempty(reason) && any(p -> p.key == "pressure_fft" && occursin("[core_anelastic] is incompatible with [pressure_fft]", p.message) &&
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
    # θ top: a name or a number
    @test BCC.site_options(load_run_config(smoke; overrides = ["urban.theta_top=0.003"])).θ_top === 0.003
    @test occursin("theta_top", errtext(err("urban.theta_top=\"lid\"")))
    # checkpoints (#269): on with the Boussinesq core; the sound-proof cores have none
    @test load_run_config(smoke; overrides = ["checkpoint.every_steps=5"]).checkpoint.every_steps == 5
    e = err("checkpoint.every_steps=5", "core_boussinesq.enabled=false", "core_anelastic.enabled=true",
            "pressure_fft.enabled=false", "pressure_density_direct.enabled=true",
            "timestep_adaptive.enabled=false", "timestep_fixed.enabled=true")
    @test any(p -> p.key == "checkpoint.every_steps" && occursin("Boussinesq core", p.message), e.problems)
    # the urban site's own rules: a fixed dt, nsteps, a uniform grid
    e = err("time.dt=\"adaptive\"")
    @test any(p -> p.key == "time.dt", e.problems)
    # five independent problems are all reported
    e = err("micro_kessler.qv_max=\"x\"", "grid.size=[16, 16]", "time.dt=-1.0",
            "nesting.parent.refinement=[2, 2]", "urban.lateral=\"closed\"")
    @test length(e.problems) == 5
end

@testset "Config berlin: berlin_smoke.toml runs bit-identical to the Julia-built case, round trip, ledgers" begin
    t0 = time()
    # every diagnostic evaluated at every step (#338): the state below must
    # still be the fixture's, bit for bit (the diagnostics only read it)
    r = roundtrip_check(berlin_toml("smoke"); nsteps = 12, nsteps_overrides = n -> ["time.nsteps=$n"],
                        overrides = ["logging.diagnostics_cadence=1e-3"])
    wall = time() - t0
    @test r.fields_equal
    isempty(r.fields_differing) || @info "fields differing in the round trip" r.fields_differing
    @test r.diagnostics_equal && r.config_equal
    case = urban_site(r.runs[1].obj)
    st = BCC.case_state(case)
    # the nest is on: the parent's fields are compared too
    @test :parent_θ in keys(st) && :parent_qv in keys(st) && :qv in keys(st)
    # bit-identical to the Julia-built smoke case on master (the fixture's digests; ESTIMATE),
    # apart from the named one-time change of KA-m (#301), the fused-exponential
    # saturation vapour pressure, for which the 13 child digests were re-recorded
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
    @test !haskey(rec["config"]["run"], "legacy") && rec["replayable"] == true
    @test length(readlines(diagnostics_path(r.runs[1].folder))) == case.clock.n + 1   # header + a row per step
    # the ledgers close to the smoke test's tolerances
    rows = BCC.surface_budget(case.budgets)
    w = BCC.water_budget(case.budgets)
    hb = BCC.heat_budget(case.budgets)
    @test maximum(abs(x.closure) for x in rows) < 1e-6        # W/m²
    @test abs(w.residual) < 1e-12                              # kg/m²
    @test abs(hb.closure / hb.elapsed) <= 1e-10                # W/m²
    @info @sprintf("berlin_smoke.toml via run_config: 2 runs in %.1f s; surface closure max %.1e W/m², water residual %.1e kg/m², air-heat closure %.1e W/m²; domain Rn %.2f H %.2f LE %.2f W/m²",
                   wall, maximum(abs(x.closure) for x in rows), w.residual, hb.closure / hb.elapsed,
                   rows[end].Rn, rows[end].H, rows[end].LE)

    # the diagnostics files hold the case's own numbers (#338): the last row
    # is after the last step, so it equals the final ledgers, timing and
    # time-stepper record exactly (shortest round-trip printing)
    lines = readlines(diagnostics_path(r.runs[1].folder))
    head = [first(split(h, " [")) for h in split(lines[1], ',')]
    col(name) = (i = findfirst(==(name), head); i === nothing ? error("no column $name") : i)
    cells = [split(l, ',') for l in lines[2:end]]
    num(row, name) = parse(Float64, row[col(name)])
    last_row = cells[end]
    @test length(head) == length(unique(head))
    dom = rows[end]
    @test dom.name == "domain"
    for t in (:Rn, :H, :LE, :G, :dS, :closure)
        @test num(last_row, "budgets.surface.domain.$t") == getfield(dom, t)
    end
    for x in rows[1:end-1]     # every class present has its columns
        @test num(last_row, "budgets.surface.$(x.name).closure") == x.closure
    end
    @test num(last_row, "budgets.water.residual") == w.residual
    @test num(last_row, "budgets.water.storage") == w.storage
    @test num(last_row, "budgets.heat.closure") == hb.closure / hb.elapsed
    @test num(last_row, "budgets.heat.storage") == hb.storage / hb.elapsed
    # the stage timing: one wall.* column per timed entry, the final value
    # the case's own (and dropped, with the wall column, by the round trip)
    @test num(last_row, "wall.dynamics.step") == case.timing[(:dynamics, :step!)]
    wallcols = filter(startswith("wall."), head)
    @test length(wallcols) >= length(case.timing)
    @test all(k -> "wall.$(BCC.diag_name(k[1])).$(BCC.diag_name(k[2]))" in wallcols, keys(case.timing))
    # the time stepper's record and the rows' dt / limiter / retries
    ts = BCC.find_coupling(case, :timestep_adaptive)
    @test num(last_row, "timestep_adaptive.steps") == ts.stats.n == case.clock.n
    @test num(last_row, "timestep_adaptive.dt_mean") == ts.stats.dt_sum / ts.stats.n
    dts = [num(c, "dt") for c in cells]
    @test sum(dts) ≈ case.sim.t rtol = 1e-12        # every step's dt, summing to the 24 s run
    @test minimum(dts) == ts.stats.dt_min && maximum(dts) == ts.stats.dt_max
    lims = [c[col("dt_limiter")] for c in cells]
    @test all(l -> l in ("advective", "diffusive", "buoyancy", "coriolis", "sgs", "dt_min", "dt_max",
                         "growth", "nesting"), lims)
    @test sum(num(c, "retries") for c in cells) == ts.stats.retries
    # a coupling's diagnostics: the pressure solve and the nest, every step
    @test num(last_row, "nesting.parent_windows") >= 1
    @test isfinite(num(last_row, "nesting.boundary_heat_top"))
    @test "wall.nesting.parent" in head        # a coupling's wall clock is filed under wall.
    @info @sprintf("berlin_smoke.toml diagnostics.csv: %d columns (%d wall.*), %d rows; dt %.3f–%.3f s, limiters %s",
                   length(head), length(wallcols), length(cells), minimum(dts), maximum(dts),
                   join(sort!(unique(lims)), ","))
end

@testset "Config berlin: the log events are structured and carry the diagnostics' values (#338)" begin
    cc = BCC.site_options(load_run_config(berlin_toml("smoke"); overrides = ["fftw.planner=\"estimate\""]))
    case = BCC.build_case(cc)
    tl = Test.TestLogger(min_level = Logging.Info)
    with_logger(tl) do
        run_site!(case)
    end
    recs = filter(r -> get(r.kwargs, :subsystem, nothing) === :urban, tl.logs)
    ev(e) = filter(r -> get(r.kwargs, :event, nothing) === e, recs)
    # no diagnostic source failed (a failing one logs a warning and gives NaN)
    @test isempty(filter(r -> r.level >= Logging.Warn, tl.logs))
    @test length(ev(:run_start)) == 1 && length(ev(:run_end)) == 1
    # one frame per output record; the last frame is the final state
    frames = ev(:frame)
    @test length(frames) == case.clock.nrecord == 2
    f = frames[end].kwargs
    @test f[:frame] == 2
    @test f[:Ts_min] == round(minimum(case.bus.Ts); sigdigits = 4)
    @test f[:Ts_max] == round(maximum(case.bus.Ts); sigdigits = 4)
    @test f[:theta1_max] == round(maximum(view(Aeolus.Fields.interior_view(case.sim.θ), :, :, 1)); sigdigits = 4)
    # one :coupling event per coupling with diagnostics per frame, its keys
    # the coupling's declared names
    with_diags = [c for c in case.couplings if !isempty(BCC.diagnostics(c, case))]
    @test length(ev(:coupling)) == 2 * length(with_diags)
    for c in with_diags
        e = last(filter(r -> r.kwargs[:coupling] === BCC.component_name(c), ev(:coupling)))
        @test Set(k for k in keys(e.kwargs) if !(k in (:subsystem, :domain, :event, :coupling))) ==
              Set(Symbol(d.name) for d in BCC.diagnostics(c, case))
    end
    tsev = last(filter(r -> r.kwargs[:coupling] === :timestep_adaptive, ev(:coupling))).kwargs
    @test tsev[:steps] == case.clock.n
    # the end of the run: steps, the timing of every entry, the ledgers
    @test ev(:run_end)[1].kwargs[:steps] == case.clock.n
    tim = ev(:timing)
    @test length(tim) == length(case.timing)
    @test issorted([r.kwargs[:seconds] for r in tim]; rev = true)
    rows = BCC.surface_budget(case.budgets)
    led = ev(:budget)
    sfc = filter(r -> r.kwargs[:budget] === :surface, led)
    @test [r.kwargs[:class] for r in sfc] == [x.name for x in rows]
    @test [r.kwargs[:closure] for r in sfc] == [x.closure for x in rows]
    wat = only(filter(r -> r.kwargs[:budget] === :water, led)).kwargs
    @test wat[:residual] == BCC.water_budget(case.budgets).residual
    hb = BCC.heat_budget(case.budgets)
    col = only(filter(r -> r.kwargs[:budget] === :heat && r.kwargs[:layer] === :column, led)).kwargs
    @test col[:closure] == hb.closure / hb.elapsed
    @test length(filter(r -> r.kwargs[:budget] === :heat_term, led)) == length(hb.terms)
    # the profiles of diagnostics.nc: the closure's and the turbulence diagnostics', one value per level
    ps = BCC.case_profiles(case)
    @test "turbulence_diag.Ri_g" in [p.name for p in ps]
    @test all(p -> length(Aeolus.RunLog.evaluate_profile(p)) == case.mesh.Nz, ps)
end

# #392: step!(model, cfg.dt) on the default case overrode the adaptive
# step (2 s against a stable step of 0.09-0.47 s); the dynamics guard does
# not test the CFL number, so the run amplified (|v| 7.8 -> 49 m/s) and
# failed at step 16. A dt above the stepper's stable_dt is now rejected
# before any stage runs. The 20-step default-case run is the "flagship"
# part of test/urban_checks.jl.
@testset "Config berlin: step!(model, dt) rejects a dt above the adaptive stable step (#392)" begin
    case = BCC.build_case(site_config(berlin_toml("smoke")))
    for _ in 1:2
        @test (step!(case.model); !case.clock.blowup)
    end
    ts = BCC.timestepper(case)
    @test BCC.variable_dt(typeof(ts))
    s = ts.stats
    lims = copy(s.limiters)
    lim = BCC.stable_dt(ts, case)
    # pure: asked twice, the same; nothing counted
    @test BCC.stable_dt(ts, case) == lim && s.limiters == lims
    @test ts.constraints.dt_min <= lim <= ts.constraints.dt_max
    # the CFL / diffusive limit of the current flow
    d = BCC.sgs_cfl_diagnostics(case, ts.constraints)
    @test lim == clamp(d.dt_limiting, ts.constraints.dt_min, ts.constraints.dt_max)
    # the stepper's own pick (growth limit, nest landing) never exceeds it
    dt_ok = BCC.step_dt(case)
    @test 0 < dt_ok <= lim
    # rejected before any stage runs: clock, time and state unchanged
    n0, t0, u0 = case.clock.n, case.sim.t, copy(interior_view(case.sim.u))
    err = try
        step!(case.model, 1.5lim); nothing
    catch e
        e
    end
    @test err isa ArgumentError && occursin("stable step", err.msg)
    @test case.clock.n == n0 && case.sim.t == t0 && interior_view(case.sim.u) == u0
    # an explicit dt within it is taken (the nest needs the stepper's own pick:
    # it lands on the parent window): one piece, no retry, CFL ≤ C_advective
    p0, r0 = s.pieces, s.retries
    @test (step!(case.model, dt_ok); !case.clock.blowup)
    @test case.sim.t ≈ t0 + dt_ok && s.pieces == p0 + 1 && s.retries == r0
    @test dt_ok * ts.rate <= ts.constraints.C_advective + 1e-9
end

@testset "Config berlin: aeolus run berlin_smoke.toml" begin
    root = mktempdir()
    rc = redirect_stdout(devnull) do
        Aeolus.main(["run", berlin_toml("smoke"), "--root", root, "--allow-uncommitted",
                     "--set", "fftw.planner=\"estimate\"", "--set", "time.nsteps=2"])
    end
    @test rc == 0
    runs = readdir(root; join = true)
    @test length(runs) == 1
    rec = read_run_toml(only(runs))
    @test rec["session"][end]["status"] == "completed"
    @test rec["config"]["time"]["nsteps"] == 2
    @test rec["config"]["run"]["name"] == "smoke" && !haskey(rec["config"]["run"], "legacy")
    # the run type's own checks (EX-a, #438): results.toml in the run folder,
    # the ledgers closed within BUDGET_TOLERANCES, the state finite
    res = TOML.parsefile(joinpath(only(runs), "output", "results.toml"))
    led = res["budgets"]
    @test res["finite"] === true && led["closed"] === true && led["unattributed"] === false
    @test led["surface_max_abs_closure"] < BCC.BUDGET_TOLERANCES.surface
    @test abs(led["water_residual"]) < BCC.BUDGET_TOLERANCES.water
    @test abs(led["air_heat_closure"]) <= BCC.BUDGET_TOLERANCES.air_heat
    @test haskey(led["surface"], "domain") && led["surface"]["domain"]["Rn"] > 0     # local noon: net gain
    @test res["timing"]["simulated_s"] ≈ 4.0          # nsteps × dt (the adaptive stepper picks the steps)
    # the idealised profile: θ 293 K at the ground, an inversion at 60 m (berlin_smoke.toml)
    @test 292.0 < res["theta_mean_below_100m_start"] < 294.0
    @test abs(res["theta_mean_below_100m_end"] - res["theta_mean_below_100m_start"]) < 0.5
    # the air-heat terms are W/m² of ground: the storage equals their sum to the closure
    air = res["air_heat"]
    @test abs(air["storage"] - sum(v for (k, v) in air if k != "storage") - led["air_heat_closure"]) < 1e-9
    # a config error exits 2 before anything is written
    root2 = mktempdir()
    rc = redirect_stderr(devnull) do
        Aeolus.main(["run", berlin_toml("smoke"), "--root", root2, "--set", "micro_kessler.enabled=false"])
    end
    @test rc == 2 && isempty(readdir(root2))
end

@testset "Config berlin: aeolus check in a fresh process" begin
    # The urban site's validators and checks run in a fresh process (when
    # the urban run type was an example file, included while the config was
    # being loaded, `aeolus check` failed there with a world-age MethodError)
    root = pkgdir(Aeolus)
    aeolus(args...) = VERSION >= v"1.12" ?
        `$(Base.julia_cmd()) --startup-file=no --project=$root -m Aeolus $args` :
        `$(Base.julia_cmd()) --startup-file=no --project=$root $(joinpath(root, "bin", "aeolus")) $args`
    err = IOBuffer()
    p = run(pipeline(ignorestatus(aeolus("check", berlin_toml("smoke"), "--set", "micro_kessler.enabled=false"));
                     stdout = devnull, stderr = err))
    @test p.exitcode == 2
    msg = String(take!(err))
    @test occursin("micro_kessler", msg) && occursin(":microphysics", msg)
end
