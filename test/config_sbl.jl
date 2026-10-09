# The `sbl` run driver (CFG-d, #359): the stable-boundary-layer cases of
# examples/sbl/ configured from TOML (examples/sbl/config_schema.jl,
# examples/sbl/configs/*.toml).
#
# Acceptance tests of the issue covered here:
#   * bit-identical to the pre-#359 code: each shipped config at its test
#     resolution gives fields bitwise equal to the old builder call with the
#     same arguments (digests captured from the code before the switch;
#     sbl_case, stratified_channel_run, relam_run and the collapse script's
#     loop are also still called directly where they exist);
#   * round trip: roundtrip_check is bit-identical under ESTIMATE FFTW for
#     gabls1, gabls3, cases99 and stratified_channel_re180 (20 steps each);
#   * errors: sponge.target = "top" lists geostrophic, mean; a GABLS3
#     config whose case table is missing names the path;
# plus what each section wires in (perturbation kinds, profile shapes,
# surface temperature kinds, stability families, nudging, the SGS stress
# form, towers, series and statistics outputs).

using Test
using Aeolus
using Aeolus.Config
using Aeolus.Fields: interior_view
using Aeolus.IO: fnv1a64, read_run_toml
using Statistics: mean, std
import TOML

using Aeolus.StableBoundaryLayer
isdefined(Main, :gabls3_data) || include(joinpath(@__DIR__, "support", "sbl_data.jl"))
include(joinpath(@__DIR__, "config_roundtrip_harness.jl"))

const SBLCFG = normpath(joinpath(@__DIR__, "..", "examples", "sbl", "configs"))
sblcfg(name) = joinpath(SBLCFG, name * ".toml")
"Run quietly into a fresh root, outside any git work tree, with the ESTIMATE planner."
sbl_run(cfg; kw...) = run_config(cfg; root = mktempdir(), console = false, repo = mktempdir(), kw...)
load_est(name, ov = String[]) = load_run_config(sblcfg(name); overrides = vcat(ov, "fftw.planner=\"estimate\""))

"The canonical text of `cfg` in a fresh file."
cfg_text_file(cfg) = write_config(joinpath(mktempdir(), "canonical.toml"), cfg)

function sbl_cfg_error(f)
    try
        f()
    catch e
        e isa ConfigError && return e
        rethrow()
    end
    error("expected a ConfigError")
end

hexdigest(a) = string(fnv1a64(reinterpret(UInt8, vec(collect(a)))); base = 16, pad = 16)
interior_digests(sim) = Dict(k => hexdigest(interior_view(getproperty(sim, s)))
                             for (k, s) in (("u", :u), ("v", :v), ("w", :w), ("p", :p), ("θ", :θ)))

# Interior digests (fnv1a64 of the Float64 bytes) of u, v, w, p, θ after the
# runs below, captured with the code before the switch to TOML (merge base
# of #359: master 3a23110 + #356 + #357; ESTIMATE FFTW, one thread, this
# machine), from the old calls:
#   gabls1       sbl_case(; N = (32, 32, 32)), 50 × sbl_step!(c, sbl_dt(c; cfl = 1.2))
#   gabls2       gabls2_case(; N = (16, 16, 32)), 50 steps (cases99: the same case)
#   gabls3       gabls3_case(; N = (16, 16, 48), extent = (400, 400, 400), sponge_z = 300), 50 steps
#   gabls4_exp3  gabls4_case(; N = (16, 16, 16)), 50 steps
#   stratified   stratified_channel_run(18, :brown94; N = (24, 16, 32), T_end = 0.6, T_avg = 0.45,
#                spinup = stratified_channel_spinup(; N, T_spin = 0.3))     (80 steps)
#   relam        relam_run(180, 1.0, relam_spinup(180; N = (16, 16, 16), T_spin = 0.3);
#                T_end = 0.3, sample = 0.1)                                   (86 steps)
#   collapse     examples/sbl/collapse_channel.jl's loop, N = (16, 16, 32), H/H_max = 0.5,
#                spin-up 0.3, run 0.3 H/u*                                     (71 steps)
# The homogeneous digests include the one fix of sbl_surface! made with the
# switch (the periodic u, v halos are refilled before the first-level wind
# of column Nx, Ny is read; before, that column saw the stale halo of the
# last RK3 combination, so a diagnostic that filled halos between steps
# changed the run): captured with the old builders plus that one line. The
# digests without it (u: gabls1 9985fd304d18b0e8, gabls2 7d89df0730c261ca,
# gabls3 0618ee1575b41846, gabls4_exp3 89b6b892624e5faa) were reproduced
# by the old code on the merge with master before the fix. The channel
# kinds (WallBoundedCase) were unaffected.
# FFTW ESTIMATE plans are deterministic on one machine; on other hardware
# (a different SIMD path) the digests may need re-capturing with the old code.
const SBL_REF = Dict(
    "gabls1" => Dict("u" => "7e75e08e335d0af4", "v" => "a2c95dd35b5ab3ef", "w" => "678f7d9f87f760a8", "p" => "3ca7b20b49286fea", "θ" => "f87662ad5f701ac1"),
    "gabls2" => Dict("u" => "80b0e2577e146a23", "v" => "a2e009b45921a0cc", "w" => "063c175195a55e89", "p" => "a29d8a5f672b378b", "θ" => "28ab81badf7df097"),
    "gabls3" => Dict("u" => "65331d3fb5e04caa", "v" => "e95dee427de2378d", "w" => "5e1de8f10c1e8fab", "p" => "19c322cceb048273", "θ" => "59734570be6010d9"),
    "gabls4_exp3" => Dict("u" => "7800742c4d16f0a4", "v" => "6455eedd22d70ba8", "w" => "94b510f9b612fc6d", "p" => "d53b38f83427af27", "θ" => "cc58d6563420ce1c"),
    "stratified_channel_re180" => Dict("u" => "b001a4b5cde1e87f", "v" => "3c7beafc4a64b76e", "w" => "7e2cab7916a6c8c6", "p" => "e16a14b8d99c14bf", "θ" => "3d801afcd0255cab"),
    "relaminarisation" => Dict("u" => "780f3b9e3adfacd9", "v" => "14ac64ccc5333f71", "w" => "0710097bb4d95c1f", "p" => "8dae520b4277eac4", "θ" => "22250558ce481fab"),
    "collapse_channel" => Dict("u" => "cf49cd55a7643913", "v" => "1f18c9c700879b2f", "w" => "27a13b480ddb8176", "p" => "3ee8302b31208a2b", "θ" => "d993740eaba05328"))

# the test resolution of each config: what test/era6_gabls*.jl and the
# channel tests use, or a reduced grid for the cases without a test
const SBL_TEST = Dict(
    "gabls1" => ["grid.size=[32,32,32]"],
    "gabls2" => ["grid.size=[16,16,32]"],
    "cases99" => ["grid.size=[16,16,32]"],
    "gabls3" => ["grid.size=[16,16,48]", "grid.extent=[400.0,400.0,400.0]", "sponge.z=300.0"],
    "gabls4_exp3" => ["grid.size=[16,16,16]"],
    "stratified_channel_re180" => ["grid.size=[24,16,32]", "case.spinup_time=0.3", "stepper.end_time=0.6", "averaging.from=0.45"],
    "relaminarisation" => ["grid.size=[16,16,16]", "case.spinup_time=0.3", "stepper.end_time=0.3", "averaging.interval=0.1"],
    "collapse_channel" => ["grid.size=[16,16,32]", "case.spinup_time=0.3", "stepper.end_time=0.3"])
const SBL_HOMOGENEOUS = ("gabls1", "gabls2", "cases99", "gabls3", "gabls4_exp3")

@testset "sbl driver: every shipped config loads and is listed" begin
    names = sort!([splitext(f)[1] for f in readdir(SBLCFG) if endswith(f, ".toml")])
    @test names == sort!(collect(keys(SBL_TEST)))
    for n in names
        cfg = load_run_config(sblcfg(n))
        @test cfg.run.driver == "sbl"
        @test load_run_config(cfg_text_file(cfg)) == cfg              # the canonical form reloads
    end
    @test sblcfg("gabls1") in shipped_configs()
    # the case table is an input: hashed under [inputs] of run.toml
    inp = config_inputs(load_run_config(sblcfg("gabls3")))
    @test only(values(inp))["path"] == normpath(joinpath(SBLCFG, "..", "data", "gabls3_case.csv"))
    ref = option_reference("sbl")
    @test all(occursin("[$s]", ref) for s in ("case", "forcing", "surface", "initial", "sgs", "sponge",
                                                 "stepper", "averaging", "towers"))
end

@testset "sbl driver: bit-identical to the pre-#359 builders (50 steps / the channel times)" begin
    for n in keys(SBL_TEST)
        ov = copy(SBL_TEST[n])
        n in SBL_HOMOGENEOUS && append!(ov, ["stepper.end_time=\"none\"", "stepper.nsteps=50"])
        res = sbl_run(load_est(n, ov))
        d = interior_digests(res.obj.case.sim)
        ref = SBL_REF[n == "cases99" ? "gabls2" : n]
        @test d == ref
        d == ref || @info "sbl bit-identity: $n differs" d ref
        n in SBL_HOMOGENEOUS && @test res.obj.case.nsteps == 50
    end
    # the builders that stay are also called directly (one source of truth:
    # the driver passes them its arguments, nothing else)
    c = sbl_case(; N = (32, 32, 32))
    for _ in 1:50; sbl_step!(c, sbl_dt(c; cfl = 1.2)); end
    @test interior_digests(c.sim) == SBL_REF["gabls1"]
    cfg = load_est("relaminarisation", SBL_TEST["relaminarisation"])
    sp = relam_spinup(180; N = (16, 16, 16), T_spin = 0.3)
    r = relam_run(180, 1.0, sp; T_end = 0.3, sample = 0.1)
    @test interior_digests(r.case.sim) == SBL_REF["relaminarisation"]
    r2 = sbl_channel_run(cfg, sbl_channel_case(cfg), sbl_channel_spinup(cfg))
    @test r2.tke == r.tke && r2.γ == r.γ && r2.nsteps == r.nsteps == 86
end

@testset "sbl driver: round trip from run.toml is bit-identical (20 steps)" begin
    chan(n) = ["stepper.end_time=0.15", "averaging.from=0.1", "case.spinup_time=0.1"]
    for (n, nso) in (("gabls1", simulation_nsteps), ("gabls3", simulation_nsteps), ("cases99", simulation_nsteps),
                     ("stratified_channel_re180", chan))
        ov = n == "cases99" ? vcat(SBL_TEST[n], "towers.hours=[0.0,24.0]", "towers.window=30.0") : SBL_TEST[n]
        n == "stratified_channel_re180" && (ov = ["grid.size=[24,16,32]"])
        r = roundtrip_check(sblcfg(n); nsteps = 20, overrides = vcat(ov, "averaging.series_interval=0.0"),
                            nsteps_overrides = nso)
        @test r.fields_equal && isempty(r.fields_differing)
        @test r.diagnostics_equal
        @test r.config_equal
        r1, r2 = r.runs
        @test r1.folder.run_id != r2.folder.run_id
        n == "stratified_channel_re180" ? @test(r1.obj.case.nsteps > 10) : @test(r1.obj.case.nsteps == 20)
        # the rows of every step are in the diagnostics (header + one per step + the start)
        @test length(readlines(Aeolus.IO.diagnostics_path(r1.folder))) >= r1.obj.case.nsteps + 1
        n == "cases99" && @test isfile(joinpath(r1.folder.path, "output", "towers.csv")) &&
                                isfile(joinpath(r2.folder.path, "output", "towers.csv")) &&
                                read(joinpath(r1.folder.path, "output", "towers.csv")) ==
                                read(joinpath(r2.folder.path, "output", "towers.csv"))
    end
end

@testset "sbl driver: errors name the key and what is allowed" begin
    e = sbl_cfg_error(() -> load_run_config(sblcfg("gabls1"); overrides = ["sponge.target=\"top\""]))
    msg = sprint(showerror, e)
    @test occursin("sponge.target", msg) && occursin("\"geostrophic\"", msg) && occursin("\"mean\"", msg)
    # a GABLS3 config whose case table (the forcing's source) is missing names the path
    dir = mktempdir()
    p = joinpath(dir, "g3.toml")
    write(p, "extends = \"$(sblcfg("gabls3"))\"\n[case]\ndata = \"missing/gabls3_case.csv\"\n")
    msg = sprint(showerror, sbl_cfg_error(() -> load_run_config(p)))
    @test occursin("case.data", msg) && occursin(normpath(joinpath(dir, "missing", "gabls3_case.csv")), msg)
    # a table the file does not have, and a column it does not have
    msg = sprint(showerror, sbl_cfg_error(() -> load_run_config(sblcfg("gabls3");
        overrides = ["initial.theta={kind=\"table\", table=\"sounding\", column=3}"])))
    @test occursin("initial.theta", msg) && occursin("no table \"sounding\"", msg) && occursin("thermo", msg)
    msg = sprint(showerror, sbl_cfg_error(() -> load_run_config(sblcfg("gabls3");
        overrides = ["forcing.u_g={kind=\"table\", z=[0.0, 2000.0], time={table=\"geostrophic\", column=1}, levels=[{column=9}, {value=-2.0}]}"])))
    @test occursin("forcing.u_g.levels[1]", msg) && occursin("has 4 rows and 3 columns", msg)
    # an option of another kind is not silently ignored
    msg = sprint(showerror, sbl_cfg_error(() -> load_run_config(sblcfg("gabls1"); overrides = ["case.h_over_l=1.0"])))
    @test occursin("case.h_over_l", msg) && occursin("not used by case.kind = \"homogeneous\"", msg)
    msg = sprint(showerror, sbl_cfg_error(() -> load_run_config(sblcfg("relaminarisation"); overrides = ["sponge.z=100.0"])))
    @test occursin("sponge.z", msg) && occursin("relaminarisation_channel", msg)
    # cross-field: f or latitude, end_time or nsteps, the builder's closure
    msg = sprint(showerror, sbl_cfg_error(() -> load_run_config(sblcfg("gabls1"); overrides = ["forcing.latitude=73.0"])))
    @test occursin("exactly one of forcing.f and forcing.latitude", msg)
    msg = sprint(showerror, sbl_cfg_error(() -> load_run_config(sblcfg("gabls1"); overrides = ["stepper.nsteps=3"])))
    @test occursin("exactly one of stepper.end_time and stepper.nsteps", msg)
    msg = sprint(showerror, sbl_cfg_error(() -> load_run_config(sblcfg("stratified_channel_re180"); overrides = ["sgs.closure=\"sigma\""])))
    @test occursin("runs dynamic_smagorinsky", msg)
    # table validators: a perturbation kind that does not exist lists the kinds
    msg = sprint(showerror, sbl_cfg_error(() -> load_run_config(sblcfg("gabls1"); overrides = ["initial.perturbation={kind=\"white\"}"])))
    @test occursin("\"gaussian\"", msg) && occursin("\"theta_uniform\"", msg) && occursin("\"uniform\"", msg)
    msg = sprint(showerror, sbl_cfg_error(() -> load_run_config(sblcfg("gabls1"); overrides = ["surface.stability={family=\"louis\"}"])))
    @test occursin("businger_dyer_log_linear", msg)
end

@testset "sbl driver: what each section builds" begin
    cfg = load_est("gabls3", SBL_TEST["gabls3"])
    c = sbl_build(cfg).case
    # [forcing]: the case description's tables (Basu 2008): U_g (-6.5, 4.5) at
    # 23 UTC, (-5, 4.5) at 03 UTC, linear in t, linear in z to (-2, 2) at 2000 m;
    # advection constant above 200 m, linear to zero at the surface
    @test forcing_value(c.U_g, 0.0, -3600.0) ≈ -6.5 && forcing_value(c.U_g, 0.0, 10800.0) ≈ -5.0
    @test forcing_value(c.U_g, 1000.0, 10800.0) ≈ (-5.0 - 2.0) / 2 && forcing_value(c.V_g, 2000.0, 0.0) ≈ 2.0
    adv = c.forcing[2]
    @test adv isa AdvectiveTendencies && forcing_value(adv.u, 500.0, 0.0) ≈ 5e-4 && forcing_value(adv.u, 100.0, 0.0) ≈ 2.5e-4
    @test c.sim.f_coriolis == 2 * 7.2921e-5 * sind(51.9711) && c.sponge_target === :mean
    # [surface]: the 0.25 m θ series, ψ_M = ψ_H = -5 z/L stable
    @test surface_temperature(c.Ts, 1800.0) ≈ (291.28 + 290.34) / 2
    @test psi_m(c.sf, 0.3) ≈ -1.5 && psi_h(c.sf, 0.3) ≈ -1.5 && c.z0m == 0.15 && c.z0h == 0.25
    # [initial] profiles: below the 10 m sounding level θ is linear from the
    # 0.25 m value, the wind logarithmic from zero at z0m
    P(sec) = sbl_profile(cfg.initial[sec], sbl_case_tables(cfg.case.data))
    θ, u = P("theta"), P("u")
    d = gabls3_data()
    @test θ(0.25) == d.surface[1, 3] && θ(10.0) == d.thermo[1, 3]
    @test θ(5.125) ≈ (d.surface[1, 3] + d.thermo[1, 3]) / 2
    @test u(0.15) == 0.0 && u(10.0) == d.wind[1, 2] && u(1.0) ≈ d.wind[1, 2] * log(1 / 0.15) / log(10 / 0.15)
    @test θ(1e4) == d.thermo[end, 3]                       # held above the table
    # GABLS4: the case file's T converted to θ with p_s = 651 hPa
    c4 = sbl_build(load_est("gabls4_exp3", SBL_TEST["gabls4_exp3"])).case
    @test surface_temperature(c4.Ts, 36000.0) ≈ 243.31 * (1e5 / 65100)^0.286
    @test c4.sim.t == 36000.0 && c4.sim.f_coriolis < 0 && c4.sf === BusingerDyerLogLinear(β_m = 4.8, β_h = 7.8)
    # GABLS2: the case formula sampled every minute from 16 LT, the subsidence ramp
    c2 = sbl_build(load_est("gabls2", SBL_TEST["gabls2"])).case
    @test surface_temperature(c2.Ts, 57600.0) ≈ (gabls2_surface_temperature_C(16.0) + 273.15) * (1e5 / 97200)^0.286
    sub = c2.forcing[2]
    @test sub isa Subsidence && sub.fields == (:θ,) && forcing_value(sub.w_s, 500.0, 144001.0) ≈ -0.0025
    @test forcing_value(sub.w_s, 500.0, 143999.0) == 0.0
    # [surface] temperature: constant and a cooling rate (GABLS1 0.25 K/h)
    g(ov) = sbl_build(load_est("gabls1", vcat("grid.size=[8,8,8]", ov))).case
    @test surface_temperature(g(String[]).Ts, 3600.0) ≈ 265 - 0.25
    @test surface_temperature(g(["surface.temperature={kind=\"constant\", value=270.0}"]).Ts, 7200.0) == 270.0
    @test g(["surface.stability={family=\"businger_dyer_beljaars_holtslag\"}"]).sf === BusingerDyerBeljaarsHoltslag()
    # [initial] perturbation kinds: none leaves every level uniform; theta_uniform
    # is bounded by its amplitude below its depth and absent above
    cn = g(["initial.perturbation={kind=\"none\"}"])
    θn = interior_view(cn.sim.θ)
    @test all(std(θn[:, :, k]) == 0 for k in axes(θn, 3))
    ct = g(String[])
    θt = interior_view(ct.sim.θ); zc = [(k - 0.5) * 50.0 for k in 1:8]
    prof(z) = z <= 100 ? 265.0 : 265.0 + 0.01 * (z - 100)
    @test all(abs(θt[i, j, k] - prof(zc[k])) <= 0.1 for i in 1:8, j in 1:8, k in 1:8 if zc[k] < 50)
    @test all(θt[i, j, k] == prof(zc[k]) for i in 1:8, j in 1:8, k in 1:8 if zc[k] >= 50)
    @test std(θt[:, :, 1]) > 0.02
    # uniform u, v, θ noise: the velocity is projected (solenoidal to round-off)
    cu = g(["initial.perturbation={kind=\"uniform\", amplitude=[0.5, 0.5, 0.1], depth=200.0}"])
    @test max_divergence(cu.sim) < 1e-10 && std(interior_view(cu.sim.u)[:, :, 1]) > 0.1
    # [sgs] stress: the Laplacian and the stress form give different momentum
    cs = g(String[]); cl = g(["sgs.stress=\"laplacian\""])
    @test cs.sgs_stress === :symmetric && cl.sgs_stress === :laplacian
    for _ in 1:3; sbl_step!(cs, 2.0); sbl_step!(cl, 2.0); end
    @test interior_view(cs.sim.u) != interior_view(cl.sim.u)
    @test g(["sgs.closure=\"vreman\"", "sgs.stability=\"none\""]).closure isa VremanClosure
    # [forcing] nudging: ProfileNudging toward 280 K relaxes the column toward
    # it, booked in the large-scale heat ledger (the domain is colder)
    cn = g(["forcing.nudging={tau=600.0, theta=280.0}"])
    @test cn.forcing[2] isa ProfileNudging && cn.forcing[2].τ == 600.0
    H0 = sbl_column_heat(cn)
    for _ in 1:3; sbl_step!(cn, 2.0); end
    @test cn.ls_heat_applied > 0
    @test sbl_column_heat(cn) - H0 ≈ cn.heat_applied + cn.ls_heat_applied rtol = 1e-9
end

@testset "sbl driver: run outputs (statistics, series, towers)" begin
    cfg = load_est("gabls1", ["grid.size=[16,16,16]", "stepper.end_time=600.0", "averaging.from=300.0",
                              "averaging.sample_every=2", "averaging.series_interval=120.0"])
    res = sbl_run(cfg)
    out = joinpath(res.folder.path, "output")
    st = TOML.parsefile(joinpath(out, "statistics.toml"))
    @test st["nsamples"] == res.obj.stats.nsamples > 0
    @test st["u_star"] == res.obj.stats.u_star && 0.1 < st["u_star"] < 0.6
    @test st["H0"] < 0                                         # the cooled surface takes heat
    rows = readlines(joinpath(out, "series.csv"))
    @test startswith(rows[1], "t,u_star,wθ,H,h,S_max,z_jet,")
    ts = [parse(Float64, split(l, ",")[1]) for l in rows[2:end]]
    # 0-600 s every 120 s: each row is the first step at or past 120 k s (steps of a few s)
    @test length(ts) == 5 && all(120k - 1e-9 <= ts[k] < 120k + 10 for k in 1:5)
    @test res.obj.case.sim.t ≈ 600.0
    # towers record only in their hours: GABLS2 starts at 16 LT
    cfg99 = load_est("cases99", ["grid.size=[8,8,32]", "stepper.end_time=\"none\"", "stepper.nsteps=6",
                                 "towers.window=10.0"])
    r = sbl_run(cfg99)
    @test r.obj.towers.nseg == 0 && !isfile(joinpath(r.folder.path, "output", "towers.csv"))
    r = sbl_run(with(cfg99, "towers.hours" => (0.0, 24.0)))
    @test r.obj.towers.nseg >= 1
    tw = readlines(joinpath(r.folder.path, "output", "towers.csv"))
    @test tw[1] == "z,V,V_TKE,sigma,n,V_s" && length(tw) > 1
    # every tower sample is binned: the counts add up to towers × levels × segments
    n = sum(parse(Float64, split(l, ",")[5]) for l in tw[2:end])
    @test n == 8 * 8 * length(r.obj.towers.ks) * r.obj.towers.nseg       # every wind below the 16 m/s bins
end
