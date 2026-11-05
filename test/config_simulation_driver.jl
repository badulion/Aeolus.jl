# The model's run from a config and the the same round trip (CFG-b,
# #357; the schema-2 sections, ONE-b #451):
# src/config/simulation_driver.jl, src/config/drivers.jl,
# test/config_roundtrip_harness.jl.
#
# Acceptance tests of the issue covered here:
#   * round trip, the same under ESTIMATE FFTW: the dry CBL config,
#     6 steps at 16³, re-run from its run.toml;
#   * the same run as the hand-built simulation: the TGV of
# taylor_green_vortex.toml equals after 10 steps at 16³,
#     the simulation examples/taylor_green_vortex.jl built in code before
#     it became a config (its code is captured below), and its kinetic
#     energy decays at the analytic viscous rate 2νk²;
#   (resume from a checkpoint: test/checks.jl and test/config_cli.jl)
# Plus the physics each section wires in: the surface heat flux closes the
# θ budget, the profiles and the perturbation, the anelastic core keeps
# ∇·(ρ̄u) at round-off, the stretched grid, the closures, the output files.

using Test
using Aeolus
using Aeolus.Config
using Aeolus.SimulationDriver
using Aeolus.Fields: set!, interior_view
using Aeolus.IO: read_run_toml, diagnostics_path
using Aeolus.NCIO: NCDataset

include(joinpath(@__DIR__, "config_roundtrip_harness.jl"))

const CFGS_DIR = joinpath(pkgdir(Aeolus), "examples")
cfgs(name) = joinpath(CFGS_DIR, name, name * ".toml")
const SIMD = SimulationDriverType()

"Load a shipped config with the ESTIMATE planner and `n` steps."
sim_cfg(name, n, extra = String[]) =
    load_run_config(cfgs(name); overrides = vcat(simulation_nsteps(n), "fftw.planner=\"estimate\"", extra))
"Run quietly into `root`, outside any git work tree."
sim_run(cfg; root = mktempdir(), kw...) = run_config(cfg; root = root, console = false, repo = mktempdir(), kw...)

function sim_error(f)
    try
        f()
    catch e
        e isa ConfigError && return e
        rethrow()
    end
    error("expected a ConfigError")
end

@testset "the model: round trip from run.toml is the same (16³, 6 steps)" begin
    for name in ("dry_convective_bl",)
        r = roundtrip_check(cfgs(name); nsteps = 6, overrides = ["grid.size=[16,16,16]"])
        @test r.fields_agree
        @test isempty(r.fields_differing)
        @test r.diagnostics_agree
        @test r.config_equal
        r1, r2 = r.runs
        @test r1.folder.run_id != r2.folder.run_id                   # a new run, a new folder
        @test simulation(r1.obj).t > 0 && r1.obj.stepper.step == 6
        # the rows of 20 steps are there, with values (not only the header)
        lines = readlines(diagnostics_path(r1.folder))
        @test length(lines) == 7
        @test all(isfinite, state_fields(r1.obj).u)
        @test read_run_toml(r2.folder.path)["config_sources"]["files"] == [joinpath(r1.folder.path, "run.toml")]
    end
end

@testset "the model: the TGV config is the hand-built TGV (16³, 10 steps)" begin
    # The simulation examples/taylor_green_vortex.jl built in code before it
    # became a config, captured verbatim (its dt rule: min of the convective
    # and viscous limits, shortened to T_final / ceil(T_final / dt)).
    L  = 2π
    N  = 16                       # (the config runs at 32³; 16³ here and in its overrides)
    ν  = 1e-2
    mesh = RegularGrid(; size=(N, N, N), extent=(L, L, L), halo=(2, 2, 2))
    sim  = BoussinesqSimulation(mesh; ν = ν)
    set!(sim.u, (x, y, z) ->  sin(x) * cos(y) * cos(z))
    set!(sim.v, (x, y, z) -> -cos(x) * sin(y) * cos(z))
    set!(sim.w, (x, y, z) ->  0.0)
    dx = L / 32                   # the config's dt rule: that of the 32³ grid it was written for
    dt_conv = 0.4 * dx / 1.0
    dt_visc = 0.4 * dx^2 / (2ν)
    dt      = min(dt_conv, dt_visc)
    T_final = 2.0
    nsteps  = ceil(Int, T_final / dt)
    dt      = T_final / nsteps
    for n in 1:10
        step!(sim, dt)
    end
    # the config: its own dt rule gives the same steps ...
    cfg0 = load_run_config(cfgs("taylor_green_vortex"); overrides = ["grid.size=[16,16,16]"])
    @test SimulationDriver._fixed_steps(cfg0.time) == (dt, nsteps)
    # ... and run for 10 steps of that dt it is the same simulation
    cfg = load_run_config(cfgs("taylor_green_vortex");
                          overrides = ["grid.size=[16,16,16]", "time.duration=\"none\"", "time.nsteps=10",
                                       "time.dt=$(repr(dt))", "fftw.planner=\"estimate\""])
    res = sim_run(cfg)
    s = state_fields(res.obj)
    @test all(k -> agree(s[k], getproperty(sim, k).data), (:u, :v, :w, :p, :θ))
    @test simulation(res.obj).t === sim.t
end

@testset "the model: TGV kinetic energy decays at 2νk² (t ∈ [0, 0.1])" begin
    # For the single Taylor-Green mode, |k|² = 3 and the initial dissipation
    # is exactly 2ν|k|² KE; the nonlinear transfer is O(t²), so over
    # t = 0.1 KE(t) = KE₀ exp(−2ν|k|² t) up to the discretisation (the
    # discrete Laplacian's eigenvalue is 3 (1 − dx²/12 + …) at 32³).
    cfg = sim_cfg("taylor_green_vortex", 10, ["time.dt=0.01"])
    ke0 = kinetic_energy(simulation(Config.build(SIMD, cfg)))
    res = sim_run(cfg)
    sim = simulation(res.obj)
    @test sim.t ≈ 0.1
    rate = -log(kinetic_energy(sim) / ke0) / sim.t
    expected = 2 * 0.01 * 3
    @test isapprox(rate, expected; rtol = 0.03)
    @test maximum(abs, [max_divergence(sim)]) < 1e-10
end

@testset "the model: the surface heat flux closes the θ budget" begin
    # Free-slip walls, no flux through the top, the sponge damps velocity
    # only: the domain-mean θ rises by Q t / Lz exactly (flux form).
    cfg = sim_cfg("dry_convective_bl", 12, ["grid.size=[16,16,16]"])
    m0 = sum(interior_view(simulation(Config.build(SIMD, cfg)).θ)) / 16^3
    res = sim_run(cfg)
    sim = simulation(res.obj)
    m1 = sum(interior_view(sim.θ)) / 16^3
    expected = 0.24 * sim.t / 1600.0
    @test sim.t == 60.0
    @test isapprox(m1 - m0, expected; rtol = 1e-6)
    # convection develops: w grows from zero
    @test maximum(abs, interior_view(sim.w)) > 1e-3
end

@testset "the model: initial profiles and the perturbation" begin
    tbl = Dict{String,Any}("kind" => "table", "z" => [0.0, 100.0, 300.0], "values" => [1.0, 3.0, 2.0])
    @test profile_value(tbl, -5.0) == 1.0 && profile_value(tbl, 50.0) == 2.0
    @test profile_value(tbl, 200.0) == 2.5 && profile_value(tbl, 1e4) == 2.0
    ml = Dict{String,Any}("kind" => "mixed_layer_lapse", "value" => 300.0, "z_i" => 800.0, "lapse" => 0.003)
    @test profile_value(ml, 799.0) == 300.0 && profile_value(ml, 1000.0) ≈ 300.6
    # the DCBL θ: the mixed layer, plus noise of amplitude 0.1 K below 100 m only
    cfg = sim_cfg("dry_convective_bl", 1, ["grid.size=[16,16,16]"])
    sim = simulation(Config.build(SIMD, cfg))
    θ = interior_view(sim.θ); zc = [sim.mesh.z0 + (k - 0.5) * sim.mesh.dz for k in 1:16]
    below = findall(<(100.0), zc); above = findall(>=(100.0), zc)
    δ = θ[:, :, below] .- 300.0
    @test maximum(abs, δ) <= 0.05 && maximum(abs, δ) > 0.04         # uniform in [−a/2, a/2]
    @test abs(sum(δ) / length(δ)) < 0.005
    @test all(θ[:, :, k] == fill(profile_value(cfg.initial.theta, zc[k]), 16, 16) for k in above)
    # the noise comes from [run] seed: the same seed, the same field; another, another
    @test agree(interior_view(simulation(Config.build(SIMD, cfg)).θ), θ)
    cfg2 = load_run_config(cfgs("dry_convective_bl"); overrides = ["grid.size=[16,16,16]", "run.seed=2"])
    @test interior_view(simulation(Config.build(SIMD, cfg2)).θ) != θ
    # a table profile of u, and θ = theta0 when no θ profile is given
    cfg3 = load_run_config(cfgs("buoyant_convection");
                           overrides = ["initial.theta={}", "initial.perturbation_amplitude=0.0",
                                        "initial.u={kind=\"table\", z=[0.0, 1.0], values=[0.0, 2.0]}"])
    s3 = simulation(Config.build(SIMD, cfg3))
    @test all(==(300.0), interior_view(s3.θ))
    @test interior_view(s3.u)[3, 5, 8] ≈ 2.0 * (7.5 / 32)              # 32³ on the unit box
end

@testset "the model: cross-field checks and profile validation" begin
    # cfl and dt_max belong to adaptive steps: with a fixed dt they would be ignored
    e = sim_error(() -> load_run_config(cfgs("taylor_green_vortex"); overrides = ["time.cfl=0.3"]))
    @test any(p -> p.key == "time.cfl" && occursin("would be ignored", p.message), e.problems)
    e = sim_error(() -> load_run_config(cfgs("taylor_green_vortex"); overrides = ["time.dt_max=2.0"]))
    @test any(p -> p.key == "time.dt_max" && occursin("adaptive", p.message), e.problems)
    e = sim_error(() -> load_run_config(cfgs("taylor_green_vortex"); overrides = ["time.nsteps=5"]))
    @test any(p -> p.key == "time.duration" && occursin("exactly one", p.message), e.problems)
    e = sim_error(() -> load_run_config(cfgs("taylor_green_vortex"); overrides = ["time.dt=\"fixed\""]))
    @test any(p -> p.key == "time.dt", e.problems)
    # the sponge starts below the top
    e = sim_error(() -> load_run_config(cfgs("dry_convective_bl"); overrides = ["sponge.z_start=1600.0"]))
    @test any(p -> p.key == "sponge.z_start" && occursin("below the top", p.message), e.problems)
    e = sim_error(() -> load_run_config(cfgs("taylor_green_vortex");
                                        overrides = ["core.model=\"anelastic\"", "boundaries.x=\"noslip\""]))
    keys_ = [p.key for p in e.problems]
    @test "boundaries.z" in keys_ && "dynamics.theta0" in keys_ && "boundaries.x" in keys_ &&
          "initial.u" in keys_                  # Taylor-Green needs periodic x, y
    e = sim_error(() -> load_run_config(cfgs("dry_convective_bl");
                                        overrides = ["grid.stretching=\"tanh\"", "sgs.model=\"sigma\""]))
    @test any(p -> p.key == "sgs.model", e.problems)
    e = sim_error(() -> load_run_config(cfgs("dry_convective_bl");
                                        overrides = ["initial.theta={kind=\"taylor_green\", amplitude=1.0, k=1.0}"]))
    @test any(p -> p.key == "initial.theta" && occursin("kind", p.message), e.problems)
    # (cross-field checks run once every option is valid on its own)
    e = sim_error(() -> load_run_config(cfgs("dry_convective_bl");
                                        overrides = ["initial.qv={kind=\"constant\", value=0.01}"]))
    @test any(p -> p.key == "initial.qv" && occursin("moisture", p.message), e.problems)
    e = sim_error(() -> load_run_config(cfgs("dry_convective_bl");
                                        overrides = ["initial.theta={kind=\"table\", z=[0.0, 0.0], values=[1.0]}"]))
    @test any(p -> p.key == "initial.theta" && occursin("lengths", p.message), e.problems)
end

@testset "the model: anelastic core, stretched grid, closures, schemes" begin
    # anelastic: θ starts at the isentropic base state, ∇·(ρ̄u) stays at round-off
    cfg = sim_cfg("dry_convective_bl", 6, ["grid.size=[16,16,16]", "core.model=\"anelastic\"",
                                           "initial.theta={}"])
    res = sim_run(cfg)
    sim = simulation(res.obj)
    @test sim isa AnelasticSimulation
    @test Aeolus.Anelastic.max_anelastic_divergence(sim) < 1e-12
    @test maximum(interior_view(sim.θ)) > 300.0          # heated from below
    # the checkpoint fields, as on the Boussinesq core but without qv (ONE-j, #460: νt, κt added)
    @test keys(state_fields(res.obj)) == (:u, :v, :w, :p, :θ, :νt, :κt)
    # stretched grid: cells cluster at the bottom, the run stays divergence-free
    cfg = sim_cfg("dry_convective_bl", 4, ["grid.size=[16,16,16]", "grid.stretching=\"tanh\"",
                                           "grid.stretching_parameter=1.5"])
    res = sim_run(cfg)
    m = simulation(res.obj).mesh
    @test m isa StretchedGrid
    @test m.dzc[m.Hz+1] < m.dzc[m.Hz+8] < m.dzc[m.Hz+16]
    @test max_divergence(simulation(res.obj)) < 1e-10
    # the transition outputs on it (#505: they read mesh.dz and failed at the
    # first write): written, and the last record is the diagnostic of the
    # final state, through the domain's closure
    tv = ["shear_production_profile", "buoyancy_production_profile", "turbulent_fraction_profile", "vorticity_y"]
    res = sim_run(sim_cfg("dry_convective_bl", 4, ["grid.size=[16,16,16]", "grid.stretching=\"tanh\"",
                                                   "grid.stretching_parameter=1.5", "output.cadence=5.0",
                                                   "output.variables=[$(join(repr.(tv), ", "))]"]))
    d = only(res.obj.model.domains)
    fr = Aeolus.Components._flux_richardson(d)
    NCDataset(joinpath(res.folder.path, "output", "output.nc")) do nc
        P = Float64.(nc["shear_production_profile"].var[:, end])
        B = Float64.(nc["buoyancy_production_profile"].var[:, end])
        @test length(P) == length(B) == 16 && all(isfinite, P) && all(isfinite, B)
        @test agree(P, fr.P; rtol = 1e-6) && agree(B, fr.B; rtol = 1e-6)      # Float32 output
        @test size(nc["vorticity_y"].var[:, :, :, end]) == (16, 16, 16)
        @test all(x -> 0 <= x <= 1, nc["turbulent_fraction_profile"].var[:, end])
        # heated from below: the buoyancy flux of the lowest level is upward
        @test B[1] > 0
    end
    # closures: every one is the :sgs component's closure object (ONE-a: the
    # plain Smagorinsky too, no longer the core's built-in C_s); none, no component
    b(over) = Config.build(SIMD, load_run_config(cfgs("dry_convective_bl"); overrides = over))
    clo(r) = Aeolus.Components.sgs_closure(only(r.model.domains))
    r = b(String[]); @test clo(r) isa StaticSmagorinsky && clo(r).C_s == 0.17 && simulation(r).C_s == 0
    r = b(["sgs.model=\"none\""]); @test clo(r) === nothing
    r = b(["sgs.model=\"sigma\""]); @test clo(r) isa SigmaModel && clo(r).C_σ == 1.35   # its own default c_s (ONE-m)
    r = b(["sgs.model=\"vreman\"", "sgs.c_s=0.07"]); @test clo(r) isa VremanClosure && clo(r).C_s == 0.07
    r = b(["sgs.stability=\"brown_et_al_1994\""]); @test clo(r) isa StabilityCorrected
    # under strong stable stratification (Ri ≈ 13 from N² = g/θ₀ 0.01 K/m and a
    # 0.005 s⁻¹ shear) Brown et al. (1994) switch the vertical mixing off
    # beyond Ri_c = 1/4, while the horizontal coefficient stays on
    stable = ["grid.size=[16,16,16]", "initial.theta={kind=\"mixed_layer_lapse\", value=300.0, z_i=0.0, lapse=0.01}",
              "initial.u={kind=\"table\", z=[0.0, 1600.0], values=[0.0, 8.0]}", "surface_theta_flux.bottom=0.0",
              "sgs.stability=\"brown_et_al_1994\""]
    rs = sim_run(sim_cfg("dry_convective_bl", 3, stable)).obj
    Ri = interior_view(clo(rs).Ri); νv = interior_view(clo(rs).νv)
    @test count(>(0.25), Ri) > length(Ri) ÷ 2
    @test all(iszero, νv[Ri .> 0.25])
    @test max_νt(simulation(rs)) > 0
end

@testset "the model: one output file, records at the cadence (ONE-c)" begin
    cfg = sim_cfg("dry_convective_bl", 6, ["grid.size=[16,16,16]", "output.variables=[\"u\", \"theta\"]",
                                           "output.cadence=10.0"])
    res = sim_run(cfg)
    dir = joinpath(res.folder.path, "output")
    @test readdir(dir) == ["output.nc", "results.toml"]                    # results.toml: the file's [checks] (ONE-d)
    NCDataset(joinpath(dir, "output.nc")) do nc
        @test nc["time"].var[:] == [0.0, 10.0, 20.0, 30.0]                   # dt = 5 s
        @test isempty(nc["time_series"].var[:])                              # no scalar selected
        @test nc["theta"].var[:, :, :, 4] == Float32.(interior_view(simulation(res.obj).θ))
        @test nc["u"].var[:, :, :, 4] == Float32.(interior_view(simulation(res.obj).u))
        @test !haskey(nc, "w")
    end
end
