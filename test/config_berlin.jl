# The Berlin case configured from TOML (CFG-c, #358): the urban site of the
# model (src/components/urban/site/config_schema.jl, EX-a #438, ONE-e #454) and the
# configs examples/urban/berlin_*.toml.
#
# Acceptance tests of the issue covered here:
#   * field for field: the site's options of "berlin_<s>.toml" equal the
#     fixture captured from the Julia-built scenarios before SCENARIOS was
#     deleted (test/data/config_berlin/fixture.toml), floats bitwise, and its
#     parent [[domain]] is the nest parent (ParentConfig) the fixture
#     recorded, option for option (NEST-p3, #483);
#   * one source of truth: the site's options map one to one onto the
#     schema's options, and their defaults are the schema's;
#   * berlin_smoke.toml run 12 steps through run_config gives the fixture's
#     state digests (child, tracers, surface, nest parent; re-captured for
#     NEST-p3, the parent a domain of the model, see the fixture's note);
#   * round trip, bit-identical under ESTIMATE FFTW, nest on;
#   * errors: an unknown provider, a parent grid of 2 integers, a parent
#     corner that does not centre the site, an incompatible pair, an unknown
#     per-coupling option;
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

# The site's options as the fixture holds them: the fields NEST-p3 (#483)
# removed (the parent's static driver and land-use table are the parent
# domain's; the nest's parent options are the parent [[domain]] block)
function _site_fixture(t)
    t = deepcopy(t)
    delete!(t, "parent_static"); delete!(t, "landuse")
    n = get(t["options"], "nesting", nothing)
    if n !== nothing
        delete!(n, "parent"); delete!(n, "mode")
        isempty(n) && delete!(t["options"], "nesting")
    end
    return t
end

"""
    parent_as_fixture(cfg) -> Dict

The parent [[domain]] of the nested Berlin config `cfg` in the terms of the
`ParentConfig` the fixture recorded (#259): every field it has a
counterpart of. Not compared: `center` (the parent's corner centres the
site, asserted on its own), `hours` and `output_period` (the parent-alone
run type's length and cadence, gone with it), `Q_surface`, `E_surface`
(the prescribed fluxes: the parent's surface is its tiles), and, without
a sounding, the sounding forcing (`nudging_*`, `z_g`).
"""
function parent_as_fixture(cfg)
    ds = config_domains(cfg)
    p = domain_view(cfg, 1)
    sv = site_view(cfg)
    g = p.grid; lsf = p.large_scale_forcing
    sounding = p.domain.dynamic_driver !== nothing
    out = Dict{String,Any}(
        "extent" => collect(g.extent), "refinement" => collect(sv.domain.refinement), "halo" => collect(g.halo),
        "vertical" => g.stretching === :uniform ? "uniform" : "stretched",
        "surface" => string(p.surface.model), "surface_period" => p.surface.period, "U_min" => p.surface.u_min,
        "radiation" => p.radiation.scheme === :mean_column, "radiation_period" => p.radiation.period,
        "geostrophic" => lsf.u_g != 0 || lsf.v_g != 0, "divergence" => lsf.divergence,
        "damping_rate" => lsf.damping_rate, "damping_frac" => lsf.damping_frac,
        "θ_top" => string(p.boundaries.theta_top), "perturbation" => p.initial.perturbation_amplitude / 2,
        "perturbation_depth" => p.initial.perturbation_depth, "seed" => cfg.run.seed,
        "sgs" => string(p.sgs.model), "stepper" => string(p.timestep.scheme), "C_advective" => cfg.time.cfl,
        "moisture" => p.dynamics.moisture,
        # the initial profiles: the site's (its sounding driver, or its idealised profile), the geostrophic wind
        "θ_init" => "nothing", "qv_init" => "nothing", "u_init" => p.initial.u == Dict("kind" => "geostrophic") ? "geostrophic" : "?")
    if g.stretching !== :uniform
        out["z_uniform"] = g.z_uniform; out["ratio"] = g.stretching_parameter; out["dz_max"] = g.dz_max
    end
    if sounding
        out["nudging_τ"] = lsf.sounding_nudging_tau; out["nudging_factor"] = lsf.nudging_factor
        out["nudging_depth"] = lsf.nudging_depth; out["z_g"] = lsf.z_g
    end
    return out
end

@testset "Config berlin: the four scenarios equal the Julia-built ones, field for field" begin
    for s in ("summer", "winter", "smoke", "mitte_sample")
        cfg = load_run_config(berlin_toml(s))
        @test length(config_domains(cfg)) == 2
        cc = BCC.site_options(site_view(cfg))
        d_case = table_diff(_site_fixture(BERLIN_FIXTURE[s]["case"]), as_toml(normalise(cc)), "case")
        # the parent [[domain]]: the ParentConfig of the fixture, option for option
        want = BERLIN_FIXTURE[s]["parent"]
        have = as_toml(parent_as_fixture(cfg))
        d_par = table_diff(Dict(k => want[k] for k in keys(have)), have, "parent")
        isempty(d_case) && isempty(d_par) || @info "berlin_$s.toml differs from the fixture" d_case d_par
        @test isempty(d_case)
        @test isempty(d_par)
        # the parent is centred on the site (ParentConfig's center = nothing):
        # its corner at -origin, the site in its middle
        p = domain_view(cfg, 1).grid; sv = site_view(cfg)
        @test collect(p.origin) == -collect(sv.domain.origin)
        @test all(a -> sv.domain.origin[a] + sv.grid.extent[a] / 2 == p.extent[a] / 2, 1:2)
        # the parent's initial state: the site's sounding driver, or the site's idealised profile
        pv = domain_view(cfg, 1)
        if cc.sounding !== nothing
            @test pv.domain.dynamic_driver == cc.dynamic && isempty(pv.initial.theta)
            @test pv.large_scale_forcing.u_g === :sounding && pv.large_scale_forcing.damping_target === :sounding
        else
            @test pv.domain.dynamic_driver === nothing
            @test pv.initial.theta == Dict("kind" => "mixed_layer_lapse", "value" => cc.θ_surface,
                                          "z_i" => cc.θ_inversion_z, "lapse" => cc.θ_lapse)
            @test pv.initial.qv == Dict("kind" => "exponential", "value" => cc.qv0, "scale_height" => cc.qv_scale)
        end
        @test pv.dynamics.theta0 == cc.θ₀ && pv.coriolis.f == cc.f_coriolis
        @test pv.radiation.start_hour == cc.start_h && pv.radiation.date == Aeolus.Urban.Date(cc.date...)
        @test pv.radiation.humid === (cc.scenario === :winter ? :subarctic_winter : :midlatitude_summer)
        # site_config(path) (the tests' helper over --set overrides) loads the same file
        @test isempty(table_diff(as_toml(normalise(site_config(berlin_toml(s)))), as_toml(normalise(cc))))
    end
    # winter is a complete file of its own (no extends, ONE-b #451); its
    # difference from the summer file, stated explicitly (its ts_water_offset
    # = 5.0, set when it extended summer, is summer's value)
    w = load_run_config(berlin_toml("winter"))
    su = load_run_config(berlin_toml("summer"))
    @test config_sources(w)["files"] == [berlin_toml("winter")]
    @test config_origin(w, "domain[2].grid.size") == berlin_toml("winter")
    tw = to_toml(w); ts = to_toml(su)
    # (a parent's initial profile table counts once)
    @test sort!(unique([replace(lstrip(split(d, ':')[1], '.'), r" \(missing\)$" => "",
                                r"^(domain\[1\]\.initial\.\w+)\..*$" => s"\1") for d in table_diff(ts, tw, "")])) ==
          sort!(["run.name", "domain[2].urban.scenario", "domain[2].urban.date", "domain[2].urban.utc_offset",
                 "domain[2].dynamics.theta0", "domain[2].urban.sounding", "domain[2].dynamic_driver",
                 "domain[2].urban.theta_surface", "domain[2].urban.qv0", "domain[2].urban.ts0",
                 "domain[2].urban.t_soil0",
                 # the parent: no sounding (the site's idealised profile), the winter date and sun
                 "domain[1].dynamic_driver", "domain[1].dynamics.theta0", "domain[1].initial.theta",
                 "domain[1].initial.qv", "domain[1].large_scale_forcing.u_g", "domain[1].large_scale_forcing.v_g",
                 "domain[1].large_scale_forcing.sounding_nudging_tau",
                 "domain[1].large_scale_forcing.damping_target", "domain[1].radiation.date",
                 "domain[1].radiation.humid"])
    @test_throws ConfigError load_run_config(berlin_toml("autumn"))
end

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
            @test any(x -> x.key == o, sec(s_).options)
            push!(used, k)
        end
    end
    @test allunique(used)
    # every option of the urban site's sections makes a field ([urban] enabled: the site itself)
    site = vcat(["urban.$(o.key)" for o in sec("urban").options if o.key != "enabled"],
                ["$n.$(o.key)" for n in names for o in sec(n).options], ["$n.$key" for (n, key) in slots],
                provider_options)
    @test issubset(site, used)
    # every urban slot has one section, whose urban providers are the slot's components
    @test length(slots) == length(BCC.URBAN_SLOTS) && allunique(first.(slots))
    for sl in BCC.URBAN_SLOTS
        n, key = BCC.URBAN_SLOT_SECTIONS[sl]
        @test sec(n).selector.key == key
        @test Set(p.name for p in uprov(n)) == Set(String(BCC.provider(C)) for C in BCC.URBAN_COMPONENTS if BCC.slot(C) === sl)
        @test sec(n).selector.defaults[:urban] == String(BCC.provider(only(C for C in BCC.URBAN_DEFAULTS if BCC.slot(C) === sl)))
    end
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
    isempty(missing_) || @info "options berlin_summer.toml does not write out" missing_
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

@testset "Config berlin: the components' sections and their options" begin
    smoke = berlin_toml("smoke")
    # (the site's options: the overrides address its [[domain]] block)
    ld(ovr...) = BCC.site_options(site_view(load_run_config(smoke; overrides = [site_key(smoke, o) for o in ovr])))
    # the form of an option is the form of its default
    c = ld("geostrophic.mode=\"nudging\"", "geostrophic.u_g=3",
           "geostrophic.tau=900.0",                    # τ: its ASCII key
           "surface_exchange.z0m={3 = 0.2}",           # merged over the class table
           "nesting.z_anterp=12.5",
           "pollutant.stacks=[[1.0, 2.0, 3.0]]", "ground.classes=[1, 3, 5]")
    g = c.components[:geostrophic]
    @test g.mode === :nudging && g.U_g === 3.0 && g.τ === 900.0
    o = BCC.component_options(BCC.GeostrophicCoupling, c)
    @test o === g && o.V_g == 0.0 && o.sponge_to_geostrophic          # the rest: the defaults
    @test keys(o) == keys(BCC.default_options(BCC.GeostrophicCoupling))
    z = c.components[:surface_exchange].z0m
    @test z[0x03] == 0.2 && z[0x01] == 0.05 && length(z) == 6
    @test c.components[:nesting].z_anterp === 12.5
    @test c.components[:pollutant].stacks == [(1.0, 2.0, 3.0)]
    @test c.components[:ground_soil].classes === (0x01, 0x03, 0x05)
    @test ld("surface_exchange.z0m=\"dynamic\"").components[:surface_exchange].z0m === :dynamic
    # a component the config does not enable has no entry; its options are its defaults
    @test !haskey(c.components, :sgs_smagorinsky)
    @test BCC.component_options(BCC.SGSSmagorinsky, c) == BCC.default_options(BCC.SGSSmagorinsky)
    # the selection: one component per urban slot, the provider its section
    # names (ONE-m, #476), additive components on or off as their sections say
    c = ld("sgs.model=\"smagorinsky\"", "sgs.diapycnal=0.1", "pollutant.enabled=false")
    @test c.components[:sgs_smagorinsky].diapycnal === 0.1
    @test BCC.SGSSmagorinsky in BCC.selected_components(c) && !(BCC.PollutantCoupling in BCC.selected_components(c))
    @test !(BCC.SGSTKE in BCC.selected_components(c))
    @test BCC.selected_components(c) == [C for C in BCC.URBAN_COMPONENTS if haskey(c.components, BCC.component_name(C))]
    # a function of time in TOML (a prescribed surface flux, [surface] model = "prescribed")
    @test BCC.toml_literal(BCC.SeriesForcing([0.0, 1.0], [2.0, 3.0])) ==
          "{kind = \"series\", t = [0.0, 1.0], value = [2.0, 3.0]}"
    @test_throws ArgumentError BCC.toml_form(t -> 1e-3 * sin(t))
    # the NetCDF file is the run's writer's ([output] netcdf in [output] path,
    # ONE-k #474): the flagship's for the renderers, none for the smoke run
    su = load_run_config(berlin_toml("summer"))
    # (one file per domain of the nest; the parent's has the selected variables it has)
    @test su.output.netcdf == "alexanderplatz.nc" && "Ts" in su.output.variables && "H_sfc" in su.output.variables
    @test isempty(load_run_config(smoke).output.variables)
    # the periods
    @test ld("urban.surface_period=2.0").cadence[:surface] == 2.0
    @test !haskey(ld().cadence, :surface)
    @test ld().cadence == Dict(:radiation => 8.0, :soil => 4.0, :output => 12.0)
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
    # θ top: a name or a number
    @test BCC.site_options(site_view(load_run_config(smoke; overrides = ["domain.site.urban.theta_top=0.003"]))).θ_top === 0.003
    @test occursin("theta_top", errtext(err("urban.theta_top=\"lid\"")))
    # checkpoints (#269): on with the Boussinesq core; the sound-proof cores have none
    @test load_run_config(smoke; overrides = ["checkpoint.every_steps=5"]).checkpoint.every_steps == 5
    e = err("checkpoint.every_steps=5", "core.model=\"anelastic\"", "pressure.solver=\"density_weighted_direct\"",
            "timestep.scheme=\"fixed\"")
    @test any(p -> endswith(p.key, "checkpoint.every_steps") && occursin("Boussinesq core", p.message), e.problems)
    # the urban site's own rules: a fixed dt, nsteps, a uniform grid
    e = err("time.dt=\"adaptive\"")
    @test any(p -> endswith(p.key, "time.dt"), e.problems)
    # five independent problems are all reported
    e = err("microphysics.qv_max=\"x\"", "grid.size=[16, 16]", "time.dt=-1.0",
            "domain.parent.grid.size=[16, 16]", "urban.lateral=\"closed\"")
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
    m = r.runs[1].obj.model
    case = urban_site(m)
    st = berlin_state(case; parent = m.domains[1])
    # the nest is on: the parent domain's fields are compared too
    @test all(k -> k in first.(st), (:parent_θ, :parent_qv, :qv))
    # the fixture's digests (ESTIMATE): re-recorded for the named changes of
    # KA-m (#301, the fused-exponential saturation vapour pressure) and
    # NEST-p3 (#483, the parent a [[domain]] of the model: see the fixture's note)
    want = BERLIN_FIXTURE["smoke_digests"]
    got = Dict(string(k) => digest(v) for (k, v) in st)
    @test sort(collect(keys(got))) == sort(collect(keys(want)))
    bad = sort!([k for k in keys(want) if get(got, k, "") != want[k]])
    isempty(bad) || @info "state digests differing from the Julia-built smoke case" bad
    @test isempty(bad)
    # the support copy of the state set (the fixture's capture) is the driver's
    @test collect(keys(BCC.case_state(case))) == first.(st)[1:end-6]
    # nsteps × dt = 24 s simulated (the default :adaptive stepper takes 11
    # steps to get there in the parent's 8 windows), completed, every root
    # window a diagnostics row
    rec = read_run_toml(r.runs[1].folder.path)
    @test case.sim.t == 24.0 && !case.clock.blowup
    @test !haskey(rec["config"]["run"], "legacy") && rec["replayable"] == true
    @test length(readlines(diagnostics_path(r.runs[1].folder))) == m.nest.hierarchy.root_steps + 1   # header + a row per window
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
    # (a row per root window: its step, the parent's)
    L = Aeolus.Nesting.nest_link(m, "site")
    @test minimum(dts) == L.stats.dt_parent_min && maximum(dts) == L.stats.dt_parent_max
    lims = [c[col("dt_limiter")] for c in cells]
    @test all(l -> l in ("advective", "diffusive", "buoyancy", "coriolis", "sgs", "dt_min", "dt_max",
                         "growth", "nesting", "prescribed"), lims)    # (prescribed: the window that lands on the end)
    @test sum(num(c, "retries") for c in cells) == m.nest.retries
    # a coupling's diagnostics: the pressure solve and the nest, every step
    @test num(last_row, "nesting.parent_windows") >= 1
    @test isfinite(num(last_row, "nesting.boundary_heat_top"))
    @test any(startswith("wall.turbulence_diag."), head)   # a coupling's wall clock is filed under wall.
    @info @sprintf("berlin_smoke.toml diagnostics.csv: %d columns (%d wall.*), %d rows; dt %.3f–%.3f s, limiters %s",
                   length(head), length(wallcols), length(cells), minimum(dts), maximum(dts),
                   join(sort!(unique(lims)), ","))
end

@testset "Config berlin: the log events are structured and carry the diagnostics' values (#338)" begin
    m = build_config_model(load_run_config(berlin_toml("smoke"); overrides = ["fftw.planner=\"estimate\""]))
    case = urban_site(m)
    tl = Test.TestLogger(min_level = Logging.Info)
    with_logger(tl) do
        run_model!(m)
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
    # (the site alone, its nest off: the stepper's own limits)
    case = BCC.build_case(site_config(berlin_toml("smoke_site")))
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
    # an explicit dt within it is taken: one piece, no retry, CFL ≤ C_advective
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
        Aeolus.main(["run", berlin_toml("smoke"), "--root", root2, "--set", "domain.site.microphysics.scheme=\"none\""])
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
    p = run(pipeline(ignorestatus(aeolus("check", berlin_toml("smoke"), "--set", "domain.site.microphysics.scheme=\"none\""));
                     stdout = devnull, stderr = err))
    @test p.exitcode == 2
    msg = String(take!(err))
    @test occursin("microphysics.scheme", msg) && occursin("kessler", msg)
end
