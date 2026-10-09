# Both dynamical cores for every component (ONE-j, #460): the core-side
# infrastructure on the anelastic core, and the ONE-a components on it.
#
#   * the sound-proof flux form of the SGS and surface fluxes: with W ≡ 1 the
#     weighted operators are the unweighted ones bit for bit; with W = ρ̄ they
#     conserve Σ W uᵢ ΔV and Σ W θ ΔV and dissipate the weighted kinetic energy;
#   * a dry CBL on both cores through the `simulation` config: the θ budget
#     closes to round-off (Σ θ on the Boussinesq core, Σ ρ̄ θ on the anelastic
#     one), and the anelastic mixed layer warms more (its heat goes into less
#     mass aloft);
#   * SSP-RK3 and the adaptive step on the anelastic core: the constraint at
#     round-off, the guard on the core's own constraint, a rejected step
#     restored bit for bit, the components' limits;
#   * checkpoints on both cores and every scheme: a restart (in-process,
#     through the components, and through the `simulation` run type's
#     resume) is bitwise the uninterrupted run; another base state is refused;
#   * the component contracts on both cores and both schemes.

using Test
using Printf
using Random: Xoshiro
using Aeolus
using Aeolus.Components
using Aeolus.Config
using Aeolus.Fields: set!, interior_view
using Aeolus.Anelastic: AnelasticSimulation, PseudoIncompressibleSimulation, isentropic_reference,
                        isothermal_reference, constraint_weights, max_anelastic_divergence
using Aeolus.IO: checkpoint!, restart!, checkpoint_state, restore_checkpoint_state!, can_checkpoint
using Aeolus.TimeIntegration: CFLConstraints
const BCC = Aeolus.Components
const BC_OPS = Aeolus.Operators

# a tall box: ρ̄ falls by ~35 % over 4 km (isentropic, θ = 300 K)
const BC_MESH = RegularGrid(; size = (8, 8, 16), extent = (800.0, 800.0, 4000.0), halo = (2, 2, 2))
bc_bcs() = (bcs = FieldBoundaryConditions(:periodic, :periodic, :wall),
            θ_bcs = FieldBoundaryConditions(:periodic, :periodic, :noflux))
function bc_core(core::Symbol; mesh = BC_MESH, kw...)
    core === :boussinesq && return BoussinesqSimulation(mesh; θ₀ = 300.0, bcs_kw(kw)...)
    return AnelasticSimulation(mesh; ref = isentropic_reference(mesh; θ_const = 300.0), bcs_kw(kw)...)
end
bcs_kw(kw) = merge(bc_bcs(), kw)
"A random state (θ ≈ 300 K plus noise), w = 0 on the walls."
function bc_state!(sim; seed = 7, a = 0.5)
    rng = Xoshiro(seed)
    for f in (sim.u, sim.v, sim.w)
        interior_view(f) .= a .* (rand(rng, size(interior_view(f))...) .- 0.5)
    end
    interior_view(sim.w)[:, :, 1] .= 0
    sim isa BoussinesqSimulation && (interior_view(sim.θ) .= 300.0)
    interior_view(sim.θ) .+= 0.3 .* rand(rng, size(interior_view(sim.θ))...)
    return sim
end
bc_bits(sim) = [copy(f.data) for f in (sim.u, sim.v, sim.w, sim.θ, sim.p, sim.νt, sim.κt)]
"The constraint weights of `sim` (ones on the Boussinesq core)."
bc_weights(sim) = sim isa BoussinesqSimulation ? (ones(size(sim.θ.data, 3)), ones(size(sim.θ.data, 3) + 1)) :
                                                 constraint_weights(sim)
"Σ_k W[k] Σ_ij f[i,j,k] over the interior levels of a centre-located field."
bc_wsum(f, W, H) = sum(W[H+k] * sum(interior_view(f)[:, :, k]) for k in 1:size(interior_view(f), 3))

const BC_SGS = (closure = :smagorinsky, c_s = 0.17, pr_t = 1 / 3, stability = :none)

@testset "the weighted SGS and surface fluxes: W ≡ 1 is the unweighted operator, bit for bit" begin
    sim = bc_state!(bc_core(:anelastic))
    Hz = BC_MESH.Hz; n = size(sim.θ.data, 3)
    one_w = (ones(n), ones(n + 1))
    fill_halos!((sim.u, sim.v, sim.w), sim.bcs); fill_halos!(sim.θ, sim.θ_bcs)
    c = StaticSmagorinsky(0.17); f = compute_sgs!(c, sim)
    for stress in (:laplacian, :symmetric)
        ta = map(x -> Aeolus.Fields.Field(Aeolus.Fields.location(x), BC_MESH), (sim.tu, sim.tv, sim.tw))
        tb = map(x -> Aeolus.Fields.Field(Aeolus.Fields.location(x), BC_MESH), (sim.tu, sim.tv, sim.tw))
        ops = (BC_OPS.add_eddy_viscosity_u!, BC_OPS.add_eddy_viscosity_v!, BC_OPS.add_eddy_viscosity_w!)
        if stress === :laplacian
            for (op, t, u) in zip(ops, ta, (sim.u, sim.v, sim.w)); op(t, u, f.νh); end
            for (op, t, u) in zip(ops, tb, (sim.u, sim.v, sim.w)); op(t, u, f.νh; weights = one_w); end
        else
            BC_OPS.add_sgs_stress_divergence!(ta..., sim.u, sim.v, sim.w, f.νh, f.νv)
            BC_OPS.add_sgs_stress_divergence!(tb..., sim.u, sim.v, sim.w, f.νh, f.νv; weights = one_w)
        end
        @test all(i -> ta[i].data == tb[i].data, 1:3)
        @test maximum(abs, ta[3].data) > 0
    end
    tθa = Aeolus.Fields.Field(CellCenter(), BC_MESH); tθb = Aeolus.Fields.Field(CellCenter(), BC_MESH)
    BC_OPS.add_eddy_diffusion_scalar!(tθa, sim.θ, f.κh); BC_OPS.add_eddy_diffusion_scalar!(tθb, sim.θ, f.κh; weights = one_w)
    @test tθa.data == tθb.data
    # the wall faces of the base state carry the adjacent cell's weight, so
    # the Boussinesq form F/Δz of a prescribed surface flux is the weighted one
    Wc, Wf = constraint_weights(sim)
    @test Wf[Hz+1] == Wc[Hz+1] && Wf[Hz+BC_MESH.Nz+1] == Wc[Hz+BC_MESH.Nz]
    @test BC_OPS.flux_weights(bc_core(:boussinesq)) === nothing
    @test BC_OPS.flux_weights(sim) === constraint_weights(sim)
end

@testset "the weighted SGS terms conserve Σ W uᵢ, Σ W θ and dissipate the weighted energy ($name)" for
        (name, mk) in (("anelastic", () -> bc_core(:anelastic)),
                       ("pseudo-incompressible", () -> PseudoIncompressibleSimulation(BC_MESH; bc_bcs()...)))
    sim = bc_state!(mk())
    Wc, Wf = constraint_weights(sim); H = BC_MESH.Hz
    @test Wc[H+1] / Wc[H+16] > 1.3                                # a strongly varying weight
    fill_halos!((sim.u, sim.v, sim.w), sim.bcs); fill_halos!(sim.θ, sim.θ_bcs)
    for c in (StaticSmagorinsky(0.17), VremanClosure(0.17), AMDClosure(),
              StabilityCorrected(StaticSmagorinsky(0.17), BC_MESH))
        for stress in (:laplacian, :symmetric)
            f = compute_sgs!(c, sim)
            @test BC_OPS.sgs_weighted_form(c, f, sim)
            foreach(t -> fill!(t.data, 0), (sim.tu, sim.tv, sim.tw, sim.tθ))
            add_sgs_momentum!(sim.tu, sim.tv, sim.tw, c, f, sim; stress = stress)
            add_sgs_scalar!(sim.tθ, sim.θ, c, f, sim)
            su = bc_wsum(sim.tu, Wc, H); sv = bc_wsum(sim.tv, Wc, H); sθ = bc_wsum(sim.tθ, Wc, H)
            scale = maximum(abs, interior_view(sim.tu)) * 8 * 8 * 16
            @test abs(su) < 1e-13 * scale && abs(sv) < 1e-13 * scale
            @test abs(sθ) < 1e-13 * maximum(abs, interior_view(sim.tθ)) * 8 * 8 * 16
            # the weighted kinetic energy Σ W (u tu + v tv) decays (u, v at the centre levels)
            ke = sum(Wc[H+k] * sum(interior_view(sim.u)[:, :, k] .* interior_view(sim.tu)[:, :, k] .+
                                   interior_view(sim.v)[:, :, k] .* interior_view(sim.tv)[:, :, k]) for k in 1:16)
            @test ke < 0
        end
    end
    # the unweighted operator (the anelastic core before ONE-j) does not conserve Σ W θ
    f = compute_sgs!(StaticSmagorinsky(0.17), sim)
    fill!(sim.tθ.data, 0)
    BC_OPS.add_eddy_diffusion_scalar!(sim.tθ, sim.θ, f.κh)
    old = abs(bc_wsum(sim.tθ, Wc, H)) / (maximum(abs, interior_view(sim.tθ)) * 8 * 8 * 16)
    @info @sprintf("%s: the unweighted SGS θ flux divergence changes Σ ρ̄θ by %.1e (relative to its scale); the weighted one by < 1e-13", name, old)
    @test old > 1e-6
end

"A closure whose scalar flux has no face form (a full tensor): no weighted form."
struct BCTensor <: Aeolus.Operators.AbstractSGSClosure end
Aeolus.Operators.sgs_coefficients!(f, ::BCTensor, sim) = (fill!(f.νh.data, 1.0); fill!(f.κh.data, 1.0); f)
Aeolus.Operators.sgs_scalar_face_diffusivity(::BCTensor, f::Aeolus.Operators.SGSFields, sim) = nothing

@testset "a sound-proof core refuses a closure without a weighted form" begin
    @test_throws ArgumentError step!(bc_state!(bc_core(:anelastic)), 1.0; sgs = BCTensor())
    step!(bc_state!(bc_core(:boussinesq)), 1.0; sgs = BCTensor())    # the Boussinesq core: unweighted, fine
    @test true
end

# ---------------------------------------------------------------------------
# a dry CBL on both cores
# ---------------------------------------------------------------------------

const BC_CFG = joinpath(pkgdir(Aeolus), "examples", "configs", "dry_convective_bl.toml")
bc_cbl_cfg(core, n; extra = String[]) =
    load_run_config(BC_CFG; overrides = vcat(["grid.size=[16,16,16]", "stepper.end_time=\"none\"",
                                              "stepper.nsteps=$n", "fftw.planner=\"estimate\""],
                                             core === :anelastic ? ["dynamics.core=\"anelastic\""] : String[], extra))
bc_run(cfg; kw...) = run_config(cfg; root = mktempdir(), console = false, repo = mktempdir(), kw...)

"The θ budget of a `simulation` run of the DCBL config: (Σ W δθ ΔV, W_bot F t A)."
function bc_budget(core, n; extra = String[])
    cfg = bc_cbl_cfg(core, n; extra = extra)
    s0 = Aeolus.SimulationDriver.simulation(Config.build(Aeolus.SimulationDriver.SimulationDriverType(), cfg))
    θ0 = copy(interior_view(s0.θ))
    r = bc_run(cfg)
    s = Aeolus.SimulationDriver.simulation(r.obj); m = s.mesh; H = m.Hz
    Wc, Wf = bc_weights(s)
    dθ = interior_view(s.θ) .- θ0
    got = sum(Wc[H+k] * sum(dθ[:, :, k]) for k in 1:m.Nz) * m.dx * m.dy * m.dz
    return (got = got, expected = Wf[H+1] * 0.24 * s.t * m.Lx * m.Ly, sim = s, θ0 = θ0)
end

@testset "dry CBL on both cores: the θ budget closes to round-off (euler, rk3, adaptive)" begin
    for core in (:boussinesq, :anelastic),
        (sch, extra) in (("euler", String[]), ("rk3", ["stepper.scheme=\"rk3\""]),
                         ("adaptive", ["stepper.dt=\"none\"", "stepper.cfl=0.3", "stepper.dt_max=5.0"]))
        b = bc_budget(core, 12; extra = extra)
        err = (b.got - b.expected) / b.expected
        @info @sprintf("DCBL 16³, %s %s, t = %.1f s: Σ W δθ ΔV = %.12e, W_bot F t A = %.12e (relative %.1e)",
                       core, sch, b.sim.t, b.got, b.expected, err)
        # round-off: δθ ≈ 0.01 K is the difference of two θ ≈ 300 K, so each
        # cell's δθ carries ~300 ε / 0.01 ≈ 7e-12 of relative error
        @test abs(err) < 1e-9
        core === :anelastic && @test max_anelastic_divergence(b.sim) < 1e-10
    end
end

@testset "dry CBL on both cores: the anelastic mixed layer warms more (30 min, 16³)" begin
    # The same kinematic flux F: the Boussinesq core heats a column of ρ₀
    # (Σ δθ Δz = F t), the anelastic one a column of ρ̄(z) that thins upward
    # (Σ ρ̄ δθ Δz = ρ̄_bot F t). Mixed over a layer of depth h, the anelastic
    # warming is larger by ρ̄_bot h / ∫₀ʰ ρ̄ dz ≈ 1 + h/(2 H_ρ) (H_ρ ≈ 12 km for
    # the 300 K isentropic base state): about 4 % at h ≈ 1 km.
    B = bc_budget(:boussinesq, 360); A = bc_budget(:anelastic, 360)
    @test abs(B.got / B.expected - 1) < 1e-11 && abs(A.got / A.expected - 1) < 1e-11   # δθ ≈ 0.5 K here
    m = A.sim.mesh; zc = [(k - 0.5) * m.dz for k in 1:m.Nz]
    ml = findall(z -> 200 <= z <= 800, zc)                         # the mixed layer, away from the surface and the top
    mean_dθ(b) = sum(sum(interior_view(b.sim.θ)[:, :, k] .- b.θ0[:, :, k]) for k in ml) / (length(ml) * m.Nx * m.Ny)
    dB = mean_dθ(B); dA = mean_dθ(A)
    ρ̄ = A.sim.ref.ρ̄; h = 1000.0
    pred = ρ̄[1] * h / (sum(ρ̄[k] for k in 1:m.Nz if zc[k] < h) * m.dz) - 1
    @info @sprintf("DCBL 30 min: mixed-layer warming Boussinesq %.4f K, anelastic %.4f K, ratio − 1 = %.4f (column-mass estimate %.4f at h = 1 km)",
                   dB, dA, dA / dB - 1, pred)
    @test 0.3 < dB < 0.7                                           # F t / h ≈ 0.43 K
    @test 0.5pred < dA / dB - 1 < 1.5pred
    # the profiles agree to the core difference: within 0.05 K below 1.6 km
    pA = vec(sum(interior_view(A.sim.θ); dims = (1, 2))) ./ 256; pB = vec(sum(interior_view(B.sim.θ); dims = (1, 2))) ./ 256
    @test maximum(abs, pA .- pB) < 0.05
end

# ---------------------------------------------------------------------------
# SSP-RK3 and the adaptive step on the anelastic core
# ---------------------------------------------------------------------------

@testset "anelastic SSP-RK3: the constraint at round-off, third order in time" begin
    sim = bc_state!(bc_core(:anelastic; C_s = 0.17))
    Aeolus.Anelastic._project_mass_weighted!(sim)                 # a solenoidal start
    for _ in 1:5
        step_rk3!(sim, 2.0)
    end
    @test max_anelastic_divergence(sim) < 1e-12
    @test sim.t == 10.0
    @test all(isfinite, interior_view(sim.θ))
    # temporal convergence: errors against a dt/8 RK3 reference over 40 s
    run(dt) = (s = bc_state!(bc_core(:anelastic; C_s = 0.17)); Aeolus.Anelastic._project_mass_weighted!(s);
               for _ in 1:round(Int, 40 / dt); step_rk3!(s, dt); end; s)
    ref = run(0.5)
    errs = [maximum(abs, interior_view(run(dt).u) .- interior_view(ref.u)) for dt in (8.0, 4.0)]
    order = log2(errs[1] / errs[2])
    @info @sprintf("anelastic SSP-RK3: max|δu| %.2e (dt = 8 s), %.2e (dt = 4 s): observed order %.2f", errs..., order)
    @test order > 2.5
    # Euler is first order on the same problem
    runE(dt) = (s = bc_state!(bc_core(:anelastic; C_s = 0.17)); Aeolus.Anelastic._project_mass_weighted!(s);
                for _ in 1:round(Int, 40 / dt); step!(s, dt); end; s)
    eE = [maximum(abs, interior_view(runE(dt).u) .- interior_view(ref.u)) for dt in (8.0, 4.0)]
    @test log2(eE[1] / eE[2]) < 1.5
    # open faces are refused (their buffers would need combining)
    @test_throws ArgumentError step_rk3!(bc_core(:anelastic; bcs = Aeolus.BoundaryConditions.open_boundary_conditions(
        BC_MESH, nothing; x = :open, y = :periodic, bottom = :wall, top = :wall, fields = (:u, :v, :w, :θ), halo = 1)), 1.0)
end

@testset "anelastic adaptive step: the guard on ∇·(ρ̄u), a rejected step restored bit for bit, the limits" begin
    sim = bc_state!(bc_core(:anelastic; C_s = 0.17)); Aeolus.Anelastic._project_mass_weighted!(sim)
    c = CFLConstraints(; C_advective = 0.5, dt_max = 50.0)
    _, dt, nr = step_adaptive!(sim, c)
    @test nr == 0 && 0 < dt <= 50.0
    g = Aeolus.TimeIntegration.step_guard(sim)
    @test g.finite && g.divergence < 1e-12
    @test max_divergence(sim) > 1e-6        # the plain ∇·u: what a Boussinesq guard would reject on
    # a step that always fails (a speed cap below the flow) is retried, then
    # the pre-step state is restored bit for bit and the driver throws (the
    # prognostic interiors: the dt pick itself fills halos and forms νt)
    iv(s) = [copy(interior_view(f)) for f in (s.u, s.v, s.w, s.θ, s.p)]
    before = iv(sim); t0 = sim.t
    @test_throws ErrorException step_adaptive!(sim, c; max_velocity = 1e-6, max_retries = 2)
    @test iv(sim) == before && sim.t == t0
    # the components' limits bind through the model's adaptive step
    d = Domain(bc_core(:anelastic); cfl = CFLConstraints(; C_advective = 0.5, dt_max = 1e3),
               components = (CentredAdvection(), Coriolis(2.0e-3)))
    _, dtm, _ = BCC.step_adaptive!(Model(d))
    @test dtm == 0.5 / 2.0e-3 && Aeolus.Simulations.step_record(d.state).limiter === :coriolis
    @test_throws ArgumentError step_adaptive!(bc_core(:anelastic), c; scheme = Aeolus.TimeIntegration.ARK2())
end

# ---------------------------------------------------------------------------
# checkpoints on both cores
# ---------------------------------------------------------------------------

"A domain on `core` with every ONE-a component (the Lagrangian closure carries memory)."
function bc_domain(core; scheme = :euler, cfl = nothing)
    sim = bc_state!(bc_core(core; ν = 0.2))
    d0 = Domain(sim; scheme = scheme, cfl = cfl, validate = false)
    comps = (CentredAdvection(), SGS(LagrangianDynamicSmagorinsky(BC_MESH)),
             BCC.build_components((Coriolis => (; f_coriolis = 1e-3),
                                   Sponge => (sponge_alpha_max = 0.01, sponge_z_start = 3000.0),
                                   SurfaceThetaFlux => (theta_flux_bottom = 0.1, theta_flux_top = 0.0)), d0)...)
    return Domain(d0; components = comps)
end
bc_div(s) = s isa BoussinesqSimulation ? max_divergence(s) : max_anelastic_divergence(s)
bc_advance!(d, n) = (for _ in 1:n; d.cfl === nothing ? step!(d, 2.0) : BCC.step_adaptive!(d); end; d)

@testset "checkpoint round trip on both cores: a restart is the uninterrupted run ($core, $sch)" for
        core in (:boussinesq, :anelastic), (sch, cfl) in ((:euler, nothing), (:rk3, nothing),
                                                          (:euler, CFLConstraints(; C_advective = 0.4, dt_max = 4.0)))
    @test can_checkpoint(bc_core(core))
    d = bc_advance!(bc_domain(core; scheme = sch, cfl = cfl), 3)
    path = joinpath(mktempdir(), "ck.jls")
    checkpoint!(d.state, path; components = BCC.checkpoint_components(d))
    bc_advance!(d, 3)
    d2 = bc_domain(core; scheme = sch, cfl = cfl)
    restart!(d2.state, path; components = BCC.checkpoint_components(d2))
    bc_advance!(d2, 3)
    @test bc_bits(d2.state) == bc_bits(d.state)
    @test d2.state.t === d.state.t
    if core === :anelastic
        # another base state is another experiment: refused, or warned with allow_mismatch
        other = Domain(Domain(AnelasticSimulation(BC_MESH; ref = isothermal_reference(BC_MESH; T_surf = 290.0), bc_bcs()...);
                              validate = false); components = d2.components)
        e = try restart!(other.state, path; components = BCC.checkpoint_components(other)); nothing catch x; x end
        @test e isa ArgumentError && occursin("reference", e.msg)
        @test_logs (:warn, r"reference") match_mode = :any restart!(other.state, path;
                                                                     components = BCC.checkpoint_components(other),
                                                                     allow_mismatch = true)
    end
end

@testset "the simulation run type: an anelastic run resumes bitwise from its checkpoint ($label)" for
        (label, extra) in (("euler", String[]),
                           ("rk3 adaptive", ["stepper.scheme=\"rk3\"", "stepper.dt=\"none\"", "stepper.cfl=0.3",
                                             "stepper.dt_max=5.0"]))
    # checkpoints of the anelastic core were refused by the config before ONE-j
    cfg = bc_cbl_cfg(:anelastic, 20; extra = vcat(extra, "checkpoint.every_steps=10"))
    full = bc_run(cfg)
    root = mktempdir()
    stop = (r, n) -> n == 15 && throw(InterruptException())
    @test_throws InterruptException run_config(cfg; root = root, console = false, repo = mktempdir(), callback = stop)
    folder = only(readdir(root; join = true))
    res = run_config(load_run_config(folder); root = root, console = false, repo = mktempdir(), resume = folder)
    a = state_fields(full.obj); b = state_fields(res.obj)
    @test keys(a) == (:u, :v, :w, :p, :θ, :νt, :κt)
    @test all(k -> a[k] == b[k], keys(a))
    @test Aeolus.SimulationDriver.simulation(res.obj).t === Aeolus.SimulationDriver.simulation(full.obj).t
end

# ---------------------------------------------------------------------------
# the component contracts on both cores and both schemes
# ---------------------------------------------------------------------------

@testset "component contracts on both cores ($core, $sch)" for core in (:boussinesq, :anelastic), sch in (:euler, :rk3)
    d = bc_domain(core; scheme = sch)
    for c in d.components
        T = typeof(c)
        @test BCC.schema(T).name == BCC.section(T)
        @test restore_checkpoint_state!(c, checkpoint_state(c)) === c
    end
    w0 = sum(abs, interior_view(d.state.w))
    bc_advance!(d, 2)
    s = d.state
    @test all(f -> all(isfinite, interior_view(f)), (s.u, s.v, s.w, s.θ, s.νt))
    @test maximum(interior_view(s.νt)) > 0                         # the :sgs component ran
    # the constraint after an Euler step; SSP-RK3 keeps ⅓² of the random start's divergence after two steps
    div0 = bc_div(bc_state!(bc_core(core; ν = 0.2)))
    @test sch === :euler ? bc_div(s) < 1e-10 : isapprox(bc_div(s), div0 / 9; rtol = 0.05)
    # each component alone, on this core and scheme, against its own physics
    # (Coriolis keeps the horizontal kinetic energy to the scheme's
    # amplification; the sponge only damps; the surface flux closes Σ W θ)
    for (comp, check) in (
            (Coriolis => (; f_coriolis = 1e-3), (a, b, s) -> abs(b.ke / a.ke - 1) < (sch === :rk3 ? 1e-9 : 2e-4)),
            (Sponge => (sponge_alpha_max = 0.01, sponge_z_start = 2000.0), (a, b, s) -> b.ke < a.ke),
            (SurfaceThetaFlux => (theta_flux_bottom = 0.1, theta_flux_top = -0.02),
             # Σ W δθ over the 64 columns: (W_bot F_bot − W_top F_top) t / Δz each
             (a, b, s) -> isapprox(sum(bc_weights(s)[1][BC_MESH.Hz+k] * sum(b.θ[:, :, k] .- a.θ[:, :, k]) for k in 1:16),
                                   (bc_weights(s)[2][BC_MESH.Hz+1] * 0.1 -
                                    bc_weights(s)[2][BC_MESH.Hz+17] * -0.02) * 4.0 * 64 / BC_MESH.dz;
                                   rtol = 1e-9)))
        sim = bc_core(core)
        core === :boussinesq && set!(sim.θ, (x, y, z) -> 300.0)
        set!(sim.u, (x, y, z) -> 2.0); set!(sim.v, (x, y, z) -> -1.0)
        dd = Domain(Domain(sim; scheme = sch, validate = false);
                    components = (CentredAdvection(), BCC.build_components((comp,), Domain(sim; validate = false))...))
        Wc, _ = bc_weights(sim); H = BC_MESH.Hz
        state(s) = (ke = sum(Wc[H+k] * sum(interior_view(s.u)[:, :, k] .^ 2 .+ interior_view(s.v)[:, :, k] .^ 2) for k in 1:16),
                    θ = copy(interior_view(s.θ)))
        a = state(sim)
        step!(dd, 2.0); step!(dd, 2.0)
        @test (core, sch, first(comp), check(a, state(sim), sim)) == (core, sch, first(comp), true)
    end
end
