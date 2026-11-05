# KA-j (#306), SC-d (#404), SC-e (#405): the step cores.
#
#   1. a blow-up collapses adaptive_dt to dt_min, the step guard fires, and
#      a rejected step restores the state with νt and κt (#346); a retry
#      that recovers equals a direct step at the short dt;
#   2. the per-simulation workspaces replace the global caches (two domains
#      stepped concurrently give the sequential results);
#   3. one step of the Boussinesq, RK3 and adaptive drivers allocates nothing;
#   4. physics of the fused / blocked cores: the projected field is
#      solenoidal at round-off, RK3 keeps its third order in time, and a
#      small-amplitude viscous Taylor–Green vortex decays at the rate of the
#      discrete Laplacian.
# (The ported kernels against the pre-port loops were checked when they were
# ported, test/support/legacy_*.jl; serial = threaded: the `steps` section of
# the thread-invariance harness.)

using Test
using Aeolus
using Aeolus.Fields: Field, set!, interior_view
using Aeolus.Grids: interior, CellCenter, FaceX, FaceY, FaceZ
using Aeolus.Backends: CPUBackend, with_backend
using Aeolus.Simulations: BoussinesqSimulation, step_rk3!, max_divergence, kinetic_energy
using Aeolus.TimeIntegration: step_adaptive!, CFLConstraints, adaptive_dt, state_snapshot, imex_workspace,
                              step_guard
using Aeolus.Anelastic: AnelasticSimulation
using Aeolus.BoundaryConditions: FieldBoundaryConditions
using Random: Xoshiro, randn

const TI = Aeolus.TimeIntegration
const SIM = Aeolus.Simulations

@testset "step cores (#306, #404, #405)" begin

@testset "blow-up: adaptive_dt collapses, the guard fires, the state is restored" begin
    g = RegularGrid(; size = (16, 16, 8), extent = (160.0, 160.0, 80.0), halo = (3, 3, 3))
    sim = BoussinesqSimulation(g; ν = 1e-2, θ₀ = 300.0, C_s = 0.17,
                               bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
    rng = Xoshiro(3063)
    for f in (sim.u, sim.v, sim.w); interior_view(f) .= 3 .* randn(rng, size(interior_view(f))); end
    set!(sim.θ, (x, y, z) -> 300 + 0.003z)
    c = CFLConstraints(dt_max = 100.0, dt_min = 1e-6)
    step!(sim, 0.01)                                 # a projected state, νt filled
    u0 = copy(sim.u.data); νt0 = copy(sim.νt.data); κt0 = copy(sim.κt.data); t0 = sim.t
    # dt = 1e4 runs away: every retry exceeds the speed cap (dt halves per
    # retry), and the state comes back
    err = try
        step_adaptive!(sim, c; dt = 1e4, max_retries = 2, max_velocity = 50.0); nothing
    catch e
        e
    end
    @test err isa ErrorException && occursin("after 2 retries", err.msg)
    # the state is copied back (not computed): exact
    @test sim.u.data == u0 && sim.t == t0
    @test sim.νt.data == νt0 && sim.κt.data == κt0   # νt, κt restored with the state (#346)
    # a NaN velocity: the dt pick collapses to dt_min
    interior_view(sim.u)[2, 2, 2] = NaN
    @test adaptive_dt(sim, c) == c.dt_min
    @test step_guard(sim).finite == false
    interior_view(sim.u)[2, 2, 2] = 0.0
    # a retry that recovers: the accepted state equals a direct step at the short dt
    a = deepcopy(sim); b = deepcopy(sim)
    _, dt, nr = step_adaptive!(a, c; dt = 40.0, max_retries = 12, max_velocity = 50.0)
    @test nr > 0 && dt == 40.0 / 2^nr
    step!(b, dt)
    # (the guard fills the velocity halos after the step; compare interiors)
    @test all(f -> agree(interior_view(getfield(a, f)), interior_view(getfield(b, f))), (:u, :v, :w))
    @test agree(a.θ.data, b.θ.data) && agree(a.p.data, b.p.data) && agree(a.νt.data, b.νt.data)
end

@testset "per-simulation workspaces: no global caches" begin
    @test !isdefined(TI, :_SNAPSHOTS) && !isdefined(TI, :_IMEX_WORKSPACES)
    @test !isdefined(SIM, :_RK3_WORKSPACES)
    g = RegularGrid(; size = (16, 16, 8), extent = (1.0, 1.0, 0.5), halo = (2, 2, 2))
    a = BoussinesqSimulation(g; ν = 1e-3); b = BoussinesqSimulation(g; ν = 1e-3)
    @test state_snapshot(a) === state_snapshot(a) === a.workspaces.snapshot
    @test state_snapshot(a) !== state_snapshot(b)
    @test imex_workspace(a) === a.workspaces.imex && SIM.rk3_workspace(a) === a.workspaces.rk3
    an = AnelasticSimulation(RegularGrid(; size = (8, 8, 8), extent = (1.0, 1.0, 1.0)))
    @test state_snapshot(an) === an.workspaces.snapshot && size(state_snapshot(an).qv) == (0, 0, 0)
    # two domains stepped concurrently (the race the global caches allowed)
    mk(seed) = (s = BoussinesqSimulation(g; ν = 1e-3, θ₀ = 300.0, C_s = 0.17);
                rng = Xoshiro(seed);
                for f in (s.u, s.v, s.w); interior_view(f) .= 0.1 .* randn(rng, size(interior_view(f))); end;
                set!(s.θ, 300.0); s)
    drive!(s) = (for _ in 1:3; step_rk3!(s, 1e-3); step_adaptive!(s, CFLConstraints(dt_max = 1e-3)); end; s)
    seq = [drive!(mk(k)) for k in 1:2]
    par = [mk(k) for k in 1:2]
    @sync for s in par
        Threads.@spawn drive!(s)
    end
    @test all(k -> agree(par[k].u.data, seq[k].u.data) && agree(par[k].θ.data, seq[k].θ.data), 1:2)
end

@testset "step cores: no allocation per step" begin
    g = RegularGrid(; size = (16, 16, 8), extent = (160.0, 160.0, 80.0), halo = (3, 3, 3))
    walls = FieldBoundaryConditions(:periodic, :periodic, :wall)
    mk() = (s = BoussinesqSimulation(g; ν = 1e-2, θ₀ = 300.0, C_s = 0.17, moisture = true, bcs = walls);
            rng = Xoshiro(3064);
            for f in (s.u, s.v, s.w); interior_view(f) .= 0.3 .* randn(rng, size(interior_view(f))); end;
            set!(s.θ, (x, y, z) -> 300 + 0.003z); set!(s.qv, 0.008); s)
    drivers = (("step!", s -> step!(s, 0.05)), ("rk3", s -> step_rk3!(s, 0.05)),
               ("adaptive", s -> step_adaptive!(s, CFLConstraints(dt_max = 0.05))))
    with_backend(CPUBackend(serial = true)) do
        s = mk()
        for (nm, drv!) in drivers
            drv!(s); drv!(s)
            nb = @allocated drv!(s)
            @test nb <= (nm == "adaptive" ? 2048 : 64)
        end
    end
end


@testset "physics: solenoidal, RK3 third order in time, Taylor–Green decay at the discrete rate" begin
    N = 16; ν = 0.05
    gt = RegularGrid(; size = (N, N, N), extent = (2π, 2π, 2π), halo = (3, 3, 3))
    # max |∇·u| at round-off after a step from a random, divergent start
    s = BoussinesqSimulation(gt; ν = 1e-2)
    for f in (s.u, s.v, s.w); interior_view(f) .= randn(Xoshiro(404), N, N, N); end
    step!(s, 0.05)
    @test max_divergence(s) < 1e-12
    # Taylor–Green vortex, ν = 0.05: the energy after t = 1 against a dt/4
    # reference; halving dt divides the error by ≈ 2³ = 8 (the spatial error is
    # common to every run). Measured: ratio 7.47 from dt = 0.1 to 0.05; a stage
    # combination in the wrong place drops to first order.
    function tg(dt; U = 1.0, T = 1.0)
        s = BoussinesqSimulation(gt; ν = ν)
        set!(s.u, (x, y, z) -> U * sin(x) * cos(y) * cos(z))
        set!(s.v, (x, y, z) -> -U * cos(x) * sin(y) * cos(z))
        for _ in 1:round(Int, T / dt)
            step_rk3!(s, dt)
        end
        return s
    end
    Kref = kinetic_energy(tg(1 / 80))
    e1 = abs(kinetic_energy(tg(0.1)) - Kref); e2 = abs(kinetic_energy(tg(0.05)) - Kref)
    @test 6.5 < e1 / e2 < 9.5
    # small amplitude (advection ~ 1e-6 of diffusion): K(t) = K₀ exp(−2ν k̃² t)
    # with k̃² the modified wavenumber of the 7-point Laplacian for k = (1, 1, 1)
    U = 1e-3
    K0 = kinetic_energy(tg(1.0; U = U, T = 0.0))
    s = tg(0.05; U = U, T = 1.0)
    Δ = 2π / N
    k2 = 3 * (2 / Δ * sin(Δ / 2))^2
    @test abs(kinetic_energy(s) / K0 / exp(-2ν * k2) - 1) < 1e-4
    @test max_divergence(s) < 1e-12
end

end
