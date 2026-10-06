# examples/validation/ configured from TOML (CFG-e, #360).
#
# Acceptance tests of the issue covered here:
#   * bit-identical to today: the case built from each shipped config, at
#     the test resolution, gives fields bitwise equal to the in-code build
#     of the old script (captured below), after 20 steps (1 column step
#     for the `column` driver);
#   * round trip, bit-identical under ESTIMATE FFTW: one config of the
#     directory (neutral_abl, one member) and dycoms_rf01.toml;
#   * sweeps: neutral_abl.toml expands to the 3 closures × 2 resolutions
#     in a fixed order, and each member validates.

using Test
using Aeolus
using Aeolus.Config

include(joinpath(@__DIR__, "config_roundtrip_harness.jl"))

const VAL_CFG = joinpath(pkgdir(Aeolus), "examples", "validation", "configs")
vcfg(name) = joinpath(VAL_CFG, name * ".toml")

"Run one member of a shipped config quietly into a fresh root: the RunResult."
function val_run(name, member, overrides = String[]; kw...)
    cfg = only(load_sweep(vcfg(name); member = member,
                          overrides = vcat("fftw.planner=\"estimate\"", overrides)))[2]
    return run_config(cfg; root = mktempdir(), console = false, repo = mktempdir(), kw...)
end

"Bitwise equality of every array of two NamedTuples of state fields."
fields_bitwise(a, b) = keys(a) == keys(b) && all(bitwise_equal(a[k], b[k]) for k in keys(a))

# ---------------------------------------------------------------------------
# the channel driver: neutral_abl.toml, log_layer_mismatch.toml
# ---------------------------------------------------------------------------

@testset "neutral_abl.toml: the sweep is 3 closures × 2 resolutions, in order, each valid" begin
    runs = load_sweep(vcfg("neutral_abl"))
    @test [m.label for (m, _) in runs] ==
          ["smagorinsky_32x32x32", "smagorinsky_48x48x48", "lasd_32x32x32", "lasd_48x48x48",
           "lasi_32x32x32", "lasi_48x48x48"]
    @test [(c.sgs.closure, c.channel.size[1]) for (_, c) in runs] ==
          [(:smagorinsky, 32), (:smagorinsky, 48), (:lasd, 32), (:lasd, 48), (:lasi, 32), (:lasi, 48)]
    @test [c.wall_model.filter for (_, c) in runs] == [:none, :none, :spatial, :spatial, :spatial, :spatial]
    # the old script's constants: T_END = 50, T_AVG = 0.6 T_END, C_S = 0.1, seed 1
    @test all(c.stepper.end_time == 50.0 && c.stepper.average_from_fraction * c.stepper.end_time == 30.0 &&
              c.sgs.c_s == 0.1 && c.run.seed == 1 && c.channel.kind === :half for (_, c) in runs)
    @test length(unique(config_hash(c) for (_, c) in runs)) == 6
end

@testset "channel driver: neutral_abl and log_layer_mismatch equal the old in-code cases, bit for bit (≈20 steps)" begin
    # examples/validation/neutral_abl.jl before CFG-e (make_case + run_case),
    # at the test resolution 24³ of test/era6_neutral_abl.jl
    t_end = 0.05                                       # ≈20 steps of the adaptive step at 24³ (counted below)
    for (label, legacy) in (("smagorinsky_32x32x32",
                             () -> half_channel_case(; N = 24, closure = StaticSmagorinsky(0.1))),
                            ("lasd_32x32x32",
                             () -> half_channel_case(; N = 24, filter = :spatial,
                                     closure = LagrangianDynamicSmagorinsky(
                                         RegularGrid(; size = (24, 24, 24), extent = (2π, 2π, 1.0));
                                         scale_dependent = true))))
        c = legacy()
        st = run_channel!(c, t_end; average_from = 0.6 * t_end)
        r = val_run("neutral_abl", label, ["channel.size=[24,24,24]", "stepper.end_time=$t_end"])
        @test r.obj.case.nsteps == c.nsteps
        @test 15 <= c.nsteps <= 30
        @test fields_bitwise(state_fields(r.obj), (u = c.sim.u.data, v = c.sim.v.data,
                                                   w = c.sim.w.data, p = c.sim.p.data))
        @test bitwise_equal(r.obj.stats.U, st.U) && r.obj.stats.τ_wall === st.τ_wall
        @test isfile(joinpath(r.folder.path, "output", "results.toml"))
    end
    # examples/validation/log_layer_mismatch.jl before CFG-e: run(Re; h_wm = 3DZ, filter = :temporal)
    DZ = 2 / 16
    c = channel_case(; Re_τ = 1000.512, N = (24, 12, 16), closure = StaticSmagorinsky(0.1),
                     wall_law = SpaldingLaw(), h_wm = 3DZ, filter = :temporal)
    st = run_channel!(c, 0.2; average_from = 0.2 - 0.2 / 2)
    r = val_run("log_layer_mismatch", "re1000_fix", ["stepper.end_time=0.2"])
    @test r.obj.case.nsteps == c.nsteps >= 15
    @test fields_bitwise(state_fields(r.obj), (u = c.sim.u.data, v = c.sim.v.data,
                                               w = c.sim.w.data, p = c.sim.p.data))
    @test bitwise_equal(r.obj.stats.U, st.U)
    # and the dynamic closure at Re_τ = 1000, first-cell matching
    c = channel_case(; Re_τ = 1000, N = (24, 12, 16), wall_law = SpaldingLaw(),
                     closure = DynamicSmagorinsky(RegularGrid(; size = (24, 12, 16), extent = (2π, π, 2.0))))
    run_channel!(c, 0.2; average_from = 0.1)
    r = val_run("log_layer_mismatch", "dynamic_first", ["stepper.end_time=0.2"])
    @test fields_bitwise(state_fields(r.obj), (u = c.sim.u.data, v = c.sim.v.data,
                                               w = c.sim.w.data, p = c.sim.p.data))
    # all fifteen members validate, with the labels the script reads
    labels = [m.label for (m, _) in load_sweep(vcfg("log_layer_mismatch"))]
    @test length(labels) == 15 && "tf2.0" in labels && "re5186_first" in labels
end

@testset "channel driver: config errors" begin
    e = try
        load_sweep(vcfg("neutral_abl"); overrides = ["channel.walls=\"bottom\"", "wall_model.t_f=0.5"])
    catch err
        err
    end
    @test e isa ConfigError
    s = sprint(showerror, e)
    @test occursin("channel.walls", s) && occursin("wall_model.t_f", s)
end

@testset "round trip from run.toml (validation: neutral_abl, one member), bit-identical" begin
    r = roundtrip_check(vcfg("neutral_abl"); member = "lasd_32x32x32",
                        overrides = ["channel.size=[16,16,16]", "stepper.end_time=0.1"])
    @test r.fields_equal && r.diagnostics_equal && r.config_equal && r.results_equal
    @test r.runs[1].obj.case.nsteps >= 10
end

# ---------------------------------------------------------------------------
# the column driver: dycoms_rf01.toml, parisfog_2011.toml, spectral_broadening.toml
# ---------------------------------------------------------------------------

"A digest of the bits of a Float64 array (stable across Julia versions)."
bits_digest(x) = Aeolus.IO.fnv1a64(reinterpret(UInt8, collect(Float64, vec(x))))

@testset "column driver: dycoms_rf01 equals the old script's column and solve, bit for bit" begin
    # digests of the arrays the old examples/validation/dycoms_rf01_radiation.jl
    # computed in main() (rf01_column, the gas and gas+cloud solves,
    # prescribed_flux, radiative_heating!), captured from origin/master
    # (b4f6f82) before the code moved to examples/validation/dycoms_rf01/rf01.jl
    old = Dict("F_up_all" => 0x453794face33fc6c, "F_up_clear" => 0x2bb06aee91cc5195,
               "F_spec" => 0x5bf8e22806a024e5, "K_day" => 0x8447ff8b88e2aebf, "ql" => 0xa3120862d449fcdb)
    r = val_run("dycoms_rf01", nothing)
    s = r.obj.state
    for (k, x) in (("F_up_all", s.Fup_all), ("F_up_clear", s.Fup_clear), ("F_spec", s.Fspec),
                   ("K_day", s.Kday), ("ql", s.col.ql))
        @test bits_digest(x) == old[k]
    end
    # the physics the script reports: a cloud capped by the inversion, the
    # prescribed flux jumping across it by (F0 − F1)(1 − e^{−κ LWP}) (outside
    # the cloud ql = 0, so Q is constant above and below it), and longwave
    # cooling of several K/day at its top
    d = r.obj.cfg.dycoms_rf01; col = s.col
    cloudy = findall(>(0), col.ql)
    @test !isempty(cloudy) && s.z_lay[last(cloudy)] < d.zi + 5
    kt = last(cloudy) + 1; kb = first(cloudy)
    Q = d.kappa * sum(col.ρ[k] * col.ql[k] * col.dz[k] for k in eachindex(col.ql))
    @test s.Fspec[kt] - s.Fspec[kb] ≈ (d.f0 - d.f1) * (1 - exp(-Q)) rtol = 1e-12
    @test 30 < s.Fspec[kt] - s.Fspec[kb] < d.f0 - d.f1
    @test minimum(s.Kday) < -5
    @test isfile(joinpath(r.folder.path, "output", "results.toml"))
end

@testset "column driver: parisfog_2011 and the parcels equal the old in-code runs, bit for bit" begin
    # parisfog_2011_radiation_fog.jl before CFG-e: parisfog_2011_column(; settings) + run!(col, t);
    # one column step (dt = 5 s) of the reference and of the no-feedback member
    for (label, set) in (("reference", FogColumnSettings()),
                         ("no_feedback", FogColumnSettings(droplet_feedback = false)))
        col = parisfog_2011_column(; settings = set)
        Aeolus.RadiationFog.run!(col, set.dt)
        r = val_run("parisfog_2011", label, ["fog_sounding.duration=$(set.dt)"])
        @test r.obj.state.set == set
        @test fields_bitwise(Base.invokelatest(state_fields, r.obj),
                             (T = col.T, qv = col.qv, ql = col.ql, u = col.u, v = col.v))
    end
    # spectral_broadening.jl before CFG-e: BroadeningCase(σ_S = σ, n_sd = 8000) and the
    # stationary case, superdroplet and bulk; shortened to 20 s / 400 superdroplets
    short = ["parcel.t_end=20.0", "parcel.n_sd=400"]
    for (label, case, model) in
        (("parcel_0.1_superdroplet", BroadeningCase(σ_S = 1e-3, n_sd = 400, t_end = 20.0), :superdroplet),
         ("parcel_0.2_bulk", BroadeningCase(σ_S = 2e-3, n_sd = 400, t_end = 20.0), :bulk),
         ("stationary_superdroplet", BroadeningCase(ε0 = 0.0, r0 = 10e-6, w = 0.0, σ_S = 1e-3, τ = 5.0,
                                                    t_end = 20.0, n_sd = 400), :superdroplet))
        h = run_broadening_case(case; model)
        r = val_run("spectral_broadening", label, short)
        @test fields_bitwise(Base.invokelatest(state_fields, r.obj),
                             (epsilon = h.ε, sigma_r2 = h.σ_r2, S = h.S, T = h.T))
    end
    # the shipped members are the old script's six runs
    runs = load_sweep(vcfg("spectral_broadening"))
    @test [m.label for (m, _) in runs] ==
          ["parcel_0.0_superdroplet", "parcel_0.0_bulk", "parcel_0.1_superdroplet", "parcel_0.1_bulk",
           "parcel_0.2_superdroplet", "parcel_0.2_bulk", "stationary_superdroplet", "stationary_bulk"]
    @test [c.parcel.sigma_s for (_, c) in runs] == [0.0, 0.0, 1e-3, 1e-3, 2e-3, 2e-3, 1e-3, 1e-3]
    @test all(c.parcel.n_sd == (i <= 6 ? 8000 : 4000) for (i, (_, c)) in enumerate(runs))
    # a key of a case that is not selected is refused
    e = try
        load_sweep(vcfg("dycoms_rf01"); overrides = ["fog.dt=10.0"])
    catch err
        err
    end
    @test e isa ConfigError && occursin("fog.dt", sprint(showerror, e))
end

@testset "round trip from run.toml (column driver: dycoms_rf01.toml), bit-identical" begin
    r = roundtrip_check(vcfg("dycoms_rf01"))
    @test r.fields_equal && r.config_equal && r.results_equal
end

# ---------------------------------------------------------------------------
# cbc_decay, patchy_surface, dns_decay
# ---------------------------------------------------------------------------

# (guarded: the serial runtests.jl includes these files from other tests too)
isdefined(@__MODULE__, :cbc_benchmark) || include(joinpath(@__DIR__, "cbc_benchmark_driver.jl"))
isdefined(@__MODULE__, :patchy_les) ||
    include(joinpath(@__DIR__, "..", "examples", "validation", "patchy_surface", "driver.jl"))
isdefined(@__MODULE__, :dns_decay) || include(joinpath(@__DIR__, "fixtures", "dns_decay.jl"))

@testset "cbc_decay, patchy_surface, dns_decay drivers equal the old in-code runs, bit for bit" begin
    # cbc_decay.jl before CFG-e: cbc_benchmark(closure; N) for closure in (dynamic, smagorinsky, none)
    runs = load_sweep(vcfg("cbc_decay"))
    @test [c.sgs.closure for (_, c) in runs] == [:dynamic, :smagorinsky, :none]
    @test all(c.grid.n == 64 && c.benchmark.seeds == [1, 2, 3] && c.benchmark.spinup == 2 for (_, c) in runs)
    # (at 16³ the decay from tU0/M = 42 to 171 is 16 steps of CFL 0.5, after the spin-up)
    old = cbc_benchmark(:smagorinsky; N = 16, seeds = [1], spinup = 1)
    r = val_run("cbc_decay", "smagorinsky", ["grid.n=16", "benchmark.seeds=[1]", "benchmark.spinup=1"])
    @test r.obj.result.nsteps == old.nsteps >= 15
    @test fields_bitwise(Base.invokelatest(state_fields, r.obj),
                         (E_42 = old.E_les[1], E_98 = old.E_les[2], E_171 = old.E_les[3]))
    @test r.obj.result.K_ratio == old.K_ratio

    # patchy_surface_les.jl before CFG-e: patchy_les(; A_grid..., case...) + patchy_run!(; A_run...),
    # at 16 × 8 × 8 for 20 steps of dt = 3 s; the tiled and the windy strips
    for (label, kw) in (("A_tiled", (λ = 800.0, T_hot = 310.0, T_cool = 300.0, mode = :tiled)),
                        ("A_windy_strips_3200", (λ = 3200.0, T_hot = 310.0, T_cool = 300.0,
                                                 Ug = 5.0, f_cor = 1e-4)))
        pl = patchy_les(; Nx = 16, Ny = 8, Nz = 8, Lx = 3200.0, Ly = 1600.0, Lz = 1600.0, kw...)
        res = patchy_run!(pl; t_end = 60.0, dt = 3.0, t_avg = 30.0, every = 5)
        r = val_run("patchy_surface", label, ["grid.size=[16,8,8]", "stepper.end_time=60.0",
                                              "stepper.average_from=30.0"])
        s = pl.sim
        @test fields_bitwise(Base.invokelatest(state_fields, r.obj),
                             (u = s.u.data, v = s.v.data, w = s.w.data, p = s.p.data, θ = s.θ.data))
        @test r.obj.result.les === res.les && r.obj.result.heat_error === res.heat_error
    end
    labels = [m.label for (m, _) in load_sweep(vcfg("patchy_surface"))]
    @test length(labels) == 13 && count(startswith("A_"), labels) == 7

    # apriori_sgs.jl before CFG-e: dns_decay(; N), here at 16³ to t = 0.3
    d = dns_decay(; N = 16, t_end = 0.3)
    r = val_run("apriori_sgs", nothing, ["grid.n=16", "dns.end_time=0.3"])
    @test r.obj.result.nsteps == d.nsteps >= 5
    s = d.sim
    @test fields_bitwise(Base.invokelatest(state_fields, r.obj),
                         (u = s.u.data, v = s.v.data, w = s.w.data, p = s.p.data))
    # the fixture's energy budget closes in the driver's run too
    @test maximum(abs, r.obj.result.budget_err) < 0.05
end
