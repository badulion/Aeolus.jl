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
import TOML

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
# cbc_decay, patchy_surface, dns (apriori_sgs)
# ---------------------------------------------------------------------------

using Aeolus.ValidationRuns     # cbc_benchmark, patchy_les / patchy_run!, dns_decay (EX-c #440)

@testset "cbc_decay, patchy_surface, dns (apriori_sgs) drivers equal the old in-code runs, bit for bit" begin
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

    # apriori_sgs.jl before CFG-e: dns_decay(; N), here at 16³ to t = 0.3; since EX-c (#440)
    # the `dns` run type (nu_margin = 1, past_peak = 0) runs it, with the a-priori record
    d = dns_decay(; N = 16, t_end = 0.3)
    r = val_run("apriori_sgs", nothing, ["grid.n=16", "stepper.end_time=0.3", "apriori.widths=[4,8]"])
    @test r.obj.cfg.run.driver == "dns"
    @test r.obj.summary.nsteps == d.nsteps >= 5
    @test r.obj.summary.ν === d.ν
    s = d.sim
    @test fields_bitwise(Base.invokelatest(state_fields, r.obj),
                         (u = s.u.data, v = s.v.data, w = s.w.data, p = s.p.data))
    @test bitwise_equal(r.obj.summary.S, d.S) && bitwise_equal(r.obj.summary.ε, d.ε)
    # the fixture's energy budget closes in the driver's run too
    @test maximum(abs, r.obj.summary.budget_err) < 0.05
    # the a-priori tables the example prints are the record of the run, per width
    rec = read_dns_apriori(joinpath(r.folder.path, "output", "dns_apriori.toml"))
    @test sort(collect(keys(rec["filter"]))) == ["4h", "8h"]
    @test all(haskey(rec["filter"]["4h"], m) for m in ("smagorinsky", "bardina", "gradient"))
    # the Bardina model correlates better with the exact stress than Smagorinsky (Liu, Meneveau & Katz 1994)
    @test rec["filter"]["4h"]["bardina"]["rho_Pi"] > rec["filter"]["4h"]["smagorinsky"]["rho_Pi"]
end

# ---------------------------------------------------------------------------
# EX-c (#440): the numbers the example scripts computed by running the model
# are diagnostics of the run types now; the scripts only read them
# ---------------------------------------------------------------------------

"The `results.toml` of a run."
val_results(r) = TOML.parsefile(joinpath(r.folder.path, "output", "results.toml"))

@testset "channel: [spectra] is the spectrum the neutral_abl script sampled, bit for bit" begin
    # neutral_abl.jl before EX-c: a run_configs callback summing streamwise_spectrum every 10th
    # step of the statistics window at the cells round(f N + 1/2), f = 0.06, 0.1, 0.25, 0.5
    ovr = ["channel.size=[16,16,16]", "stepper.end_time=0.6"]
    acc = Ref{Any}(nothing); n = Ref(0); var = zeros(4)
    legacy = function (r, step)
        st = r.cfg.stepper
        (r.case.sim.t >= st.average_from_fraction * st.end_time && step % 10 == 0) || return
        lv = unique(clamp.(round.(Int, (0.06, 0.1, 0.25, 0.5) .* 16 .+ 0.5), 1, 16))
        sp = streamwise_spectrum(r.case.sim.u, lv)
        acc[] === nothing ? (acc[] = copy(sp.E11)) : (acc[] .+= sp.E11); n[] += 1
        # the plane variance of u′ (line means removed), computed directly, for the Parseval check
        u = Aeolus.Fields.interior_view(r.case.sim.u)
        for (m, k) in enumerate(lv)
            pl = u[:, :, k]
            var[m] += sum(abs2, pl .- sum(pl; dims = 1) ./ size(pl, 1)) / length(pl)
        end
    end
    r0 = val_run("neutral_abl", "smagorinsky_32x32x32", vcat(ovr, "spectra.levels=[]"); callback = legacy)
    r = val_run("neutral_abl", "smagorinsky_32x32x32", ovr)
    @test r.obj.cfg.spectra.levels == [0.06, 0.1, 0.25, 0.5] && r.obj.cfg.spectra.every == 10
    @test n[] >= 2 && r.obj.spectra.nsamples[] == n[]
    @test r.obj.spectra.levels == [1, 2, 4, 8] == spectrum_levels([0.06, 0.1, 0.25, 0.5], 16)
    sp = val_results(r)["spectra"]
    @test all(bitwise_equal(Float64.(sp["E11"][m]), acc[][:, m] ./ n[]) for m in 1:4)
    # sampling does not touch the flow: the fields are those of the run without spectra
    @test fields_bitwise(state_fields(r.obj), state_fields(r0.obj))
    # Parseval: each sample's spectrum integrates to the variance of u′ on its plane, so the
    # average does too, to the variance averaged over the samples (one-sided, Δk1 = 2π/L_x)
    Δk = sp["k1"][2] - sp["k1"][1]
    @test all(isapprox(sum(sp["E11"][m]) * Δk, var[m] / n[]; rtol = 1e-10) for m in 1:4)
    @test all(var .> 0)
    @test sp["z_over_depth"] == [(k - 0.5) / 16 for k in (1, 2, 4, 8)]
end

@testset "cbc_decay: the run records CBC's table 3 and its interpolant" begin
    r = val_run("cbc_decay", "smagorinsky", ["grid.n=16", "benchmark.seeds=[1]", "benchmark.spinup=1"])
    ref = val_results(r)["reference"]
    k, E, _ = cbc_table()
    @test Float64.(ref["k_table"]) == k
    @test isequal(Float64.(ref["E_table_98"]), E[:, 2])
    kk = Float64.(ref["k_curve"])
    @test length(kk) == 200 && issorted(kk) && kk[1] ≈ r.obj.result.k[2] / 1.3 && kk[end] ≈ 2.2 * r.obj.result.k[9]
    # the interpolant passes through the table and decays: E(98) < E(42) at every k (decaying turbulence)
    @test all(Float64.(ref["E_curve_98"]) .< Float64.(ref["E_curve_42"]))
    @test val_results(r)["nseeds"] == 1
end

@testset "column: dycoms_rf01, parisfog_2011 and the parcels record what their scripts derive" begin
    r = val_run("dycoms_rf01", nothing)
    res = val_results(r); s = r.obj.state
    @test bitwise_equal(Float64.(res["rho"]), s.col.ρ) && bitwise_equal(Float64.(res["dz"]), s.col.dz)
    # the liquid water path the script reports, from the recorded ρ, ql, dz: RF01 observed 45-60 g/m²
    lwp = sum(Float64.(res["rho"]) .* Float64.(res["ql"]) .* Float64.(res["dz"])) * 1000
    @test 30 < lwp < 120
    r = val_run("parisfog_2011", "reference", ["fog_sounding.duration=1800.0"])
    res = val_results(r)
    H = r.obj.state.history
    @test length(res["C_LW"]) == length(H) && isequal(Float64.(res["lw_net_top"]), [h.lw_net_top for h in H])
    @test res["t_max_lwp_utc_hour"] == fog_life_cycle(H).t_max_lwp
    # the stationary parcel: the OU width is 0 at t = 0 and grows like t^{1/2} once t ≫ τ
    r = val_run("spectral_broadening", "stationary_bulk", ["parcel.t_end=200.0"])
    res = val_results(r); t = Float64.(res["t"]); ou = Float64.(res["sigma_r2_ou"])
    @test ou[1] == 0 && issorted(ou)
    w = t .>= 10 * r.obj.cfg.parcel.tau
    p = fit_power_law(t[w], ou[w])[1]
    @test 0.45 < p < 0.55
    @test bitwise_equal(Float64.(res["T"]), r.obj.state.T)
end

# canopy_profiles.jl before EX-c (the script's own wind / temperature, verbatim)
function legacy_canopy_wind(p, h; C_d, ztop, N = 400, β = 0.3)
    zf = collect(range(0, ztop; length = N + 1))
    lad = layer_leaf_area_density(p, zf)
    l_c = harman_finnigan_mixing_length(β, C_d, leaf_area_index(p) / h)
    sol = solve_canopy_wind_column(zf, lad; C_d = C_d, u_top = 5.0, bottom = :no_slip,
                                   l_mix = z -> canopy_mixing_length(z; h = h, l_c = l_c))
    return (; sol, lad, zf, l_c, d = canopy_displacement_height(sol, lad; C_d = C_d))
end

@testset "plant_canopy: canopy_profiles.toml is the old script's canopies, bit for bit" begin
    include(joinpath(@__DIR__, "data", "canopy_profiles.jl"))
    OBS = Base.invokelatest(canopy_profile_observations)
    runs = Dict(m.label => c for (m, c) in load_sweep(vcfg("canopy_profiles")))
    @test sort(collect(keys(runs))) == sort(["wind_tunnel", "wind_tunnel_cd_1.2", "chats_leaves", "chats_noleaves",
                                             "hyytiala", "hyytiala_day", "hyytiala_night"])
    # the config's canopies are the data file's (one source: the digitised profiles)
    for (lab, ds) in (("chats_leaves", OBS.chats_leaves), ("chats_noleaves", OBS.chats_noleaves))
        c = runs[lab].canopy
        @test c.h == ds.h && c.lad_z == ds.pad_z && c.lad_values == ds.pad_ah && c.lad_units === :ah
    end
    for lab in ("hyytiala", "hyytiala_day", "hyytiala_night")
        c = runs[lab].canopy
        @test c.h == OBS.hyytiala.h && c.c_d == OBS.hyytiala.C_d && c.lad_z == OBS.hyytiala.lad_z &&
              c.lad_values == OBS.hyytiala.lad
    end
    wt = OBS.wind_tunnel
    @test runs["wind_tunnel"].canopy.ah == wt.ah && runs["wind_tunnel"].canopy.c_d == wt.C_d &&
          runs["wind_tunnel_cd_1.2"].canopy.c_d == wt.C_d_alt
    # the runs equal the old in-script solves
    hy = OBS.hyytiala
    for (lab, legacy) in (("wind_tunnel", () -> legacy_canopy_wind(UniformLAD(wt.ah, wt.h), wt.h; C_d = wt.C_d, ztop = 2wt.h)),
                          ("chats_leaves", () -> (ds = OBS.chats_leaves;
                                                  legacy_canopy_wind(TabulatedLAD(ds.pad_z .* ds.h, ds.pad_ah ./ ds.h), ds.h;
                                                                     C_d = 0.2, ztop = 2.9ds.h))),
                          ("hyytiala", () -> legacy_canopy_wind(TabulatedLAD(hy.lad_z .* hy.h, hy.lad), hy.h;
                                                                C_d = hy.C_d, ztop = 1.47hy.h)))
        m = legacy()
        r = val_run("canopy_profiles", lab)
        @test bitwise_equal(r.obj.profile.sol.u, m.sol.u) && r.obj.profile.d === m.d
        res = val_results(r)
        # physics: the wind grows with height through the canopy, and d lies inside it
        @test issorted(Float64.(res["u"])) && 0 < res["d"] < r.obj.cfg.canopy.h
    end
    # day: leaves warmer than the air above (unstable, ΔT > 0 inside); night: colder (ΔT < 0)
    day = val_results(val_run("canopy_profiles", "hyytiala_day"))
    night = val_results(val_run("canopy_profiles", "hyytiala_night"))
    zd = Float64.(day["z"]); i = argmin(abs.(zd .- 8.4))
    @test day["dT"][i] > 0.1 && night["dT"][i] < -0.5
    # a key of the temperature mode in a wind run is refused
    e = try load_sweep(vcfg("canopy_profiles"); member = "hyytiala", overrides = ["canopy_energy.rh=0.9"]) catch err; err end
    @test e isa ConfigError && occursin("canopy_energy.rh", sprint(showerror, e))
end

@testset "dynamic_roughness: dynamic_roughness_cubes.toml is the old script's table, bit for bit" begin
    # dynamic_roughness_cubes.jl before EX-c, the row Δ = h/2, C_d = 1 (20 cubes per row, 16 points per h)
    hf = staggered_cube_heights(20, 16)
    @test sum(hf .> 0) / length(hf) == 0.25                              # λ_p of the staggered array
    st = subfilter_height(hf, 8; dx = 0.5, dy = 0.5)
    dr = DynamicRoughness(st; z0_floor = 1e-4, C_d = 1.0)
    uc = fill(5.0, 80, 80); vc = zeros(80, 80)
    update_dynamic_roughness!(dr, uc, vc, 2.0)
    D = roughness_surface_drag(dr, uc, vc, 2.0)
    r = val_run("dynamic_roughness_cubes", "d0.5_cd1")
    x = r.obj.result
    @test x.alpha === dr.α && x.ustar_over_u === sqrt(D.total) / 5.0
    mac = Aeolus.SurfaceTiles.macdonald_roughness(0.25, 0.25, 1.0)
    @test x.d_mac === mac.d && x.z0_mac === mac.z0
    # the numbers of docs/src/theory/wall_models.md (example 3): α = 2.26, ⟨z0⟩/h = 0.53,
    # z0_eff/h = 0.71 with the Macdonald displacement, 5.8× Macdonald's 0.123
    @test x.converged
    @test x.alpha ≈ 2.26 rtol = 0.01
    @test x.z0_mean ≈ 0.53 rtol = 0.02
    @test x.z0_eff_dmac ≈ 0.71 rtol = 0.02
    @test x.ratio_mac ≈ 5.8 rtol = 0.02
    @test x.z0_mac ≈ 0.123 rtol = 0.01
    @test 0 < x.form_fraction < 1                                         # resolved form drag is part of τ
    # at Δ = h no α balances the resolved drag of the cube walls (no root)
    @test !val_run("dynamic_roughness_cubes", "d1_cd1").obj.result.converged
    @test length(load_sweep(vcfg("dynamic_roughness_cubes"))) == 5
end
