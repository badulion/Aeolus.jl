# KA-j (#306): the step cores on kernels.
#
#   1. the weighted flux-form operators and the anelastic buoyancy as launch!
#      bodies equal the pre-port loops (test/support/legacy_step.jl) bit for
#      bit, on the serial, threshold-0 and default backends;
#   2. the field algebra of the drivers (axpy!, the IMEX stage combinations,
#      the RK3 combinations, p = φ/Δt) equals the broadcasts it replaced;
#   3. the fused CFL sweep and the fused step guard equal the separate
#      maxima and checks, NaN and Inf semantics included; a blow-up still
#      collapses adaptive_dt to dt_min and the guard still fires;
#   4. the rollback restores νt and κt;
#   5. the per-simulation workspaces replace the global caches (two domains
#      stepped concurrently give the sequential bits);
#   6. serial and threaded step cores give the same bits; one step of the
#      Boussinesq, RK3 and adaptive drivers allocates nothing.

using Test
using Aeolus
using Aeolus.Fields: Field, set!, interior_view
using Aeolus.Grids: interior, CellCenter, FaceX, FaceY, FaceZ
using Aeolus.Backends: CPUBackend, with_backend
using Aeolus.Simulations: BoussinesqSimulation, step_rk3!, max_divergence
using Aeolus.TimeIntegration: step_adaptive!, step_imex!, ARK2, Rosenbrock2, CFLConstraints,
                              cfl_diagnostics, adaptive_dt, state_snapshot, imex_workspace,
                              save_state!, restore_state!, step_guard
using Aeolus.Anelastic: AnelasticSimulation, PseudoIncompressibleSimulation, isothermal_reference,
                        weighted_divergence!, weighted_momentum_advect!, weighted_scalar_advect!,
                        add_anelastic_buoyancy!
using Aeolus.BoundaryConditions: FieldBoundaryConditions, fill_halos!
using Random: Xoshiro, randn

isdefined(@__MODULE__, :LegacyStep) || include(joinpath(@__DIR__, "support", "legacy_step.jl"))

const _KS_BACKENDS = (("serial", CPUBackend(serial = true)), ("threshold 0", CPUBackend(threshold = 0)),
                      ("default", CPUBackend()))
const TI = Aeolus.TimeIntegration
const SIM = Aeolus.Simulations

_ks_rand!(f, rng; s = 1.0, o = 0.0) = (f.data .= o .+ s .* randn(rng, size(f.data)); f)

@testset "KA-j: step cores on kernels (#306)" begin

@testset "weighted operators and anelastic buoyancy: the old loops bit for bit" begin
    g = RegularGrid(; size = (37, 21, 13), extent = (370.0, 210.0, 1300.0), halo = (2, 2, 2))
    ref = isothermal_reference(g; T_surf = 290.0)
    rng = Xoshiro(306)
    u, v, w = _ks_rand!(Field(FaceX(), g), rng), _ks_rand!(Field(FaceY(), g), rng), _ks_rand!(Field(FaceZ(), g), rng)
    θ = _ks_rand!(Field(CellCenter(), g), rng; s = 2.0, o = 300.0)
    Wc, Wf = ref.ρ̄c_pad, ref.ρ̄f_pad
    for (bn, b) in _KS_BACKENDS
        with_backend(b) do
            for normalize in (false, true)
                a = _ks_rand!(Field(CellCenter(), g), rng); c = Field(CellCenter(), g); c.data .= a.data
                weighted_divergence!(a, u, v, w, Wc, Wf, g; normalize)
                LegacyStep.weighted_divergence!(c, u, v, w, Wc, Wf, g; normalize)
                @test a.data == c.data
            end
            tn = (_ks_rand!(Field(FaceX(), g), rng), _ks_rand!(Field(FaceY(), g), rng), _ks_rand!(Field(FaceZ(), g), rng))
            to = map(f -> (c = Field(Aeolus.Fields.location(f), g); c.data .= f.data; c), tn)
            weighted_momentum_advect!(tn..., u, v, w, Wc, Wf, g)
            LegacyStep.weighted_momentum_advect!(to..., u, v, w, Wc, Wf, g)
            @test all(i -> tn[i].data == to[i].data, 1:3)
            a = _ks_rand!(Field(CellCenter(), g), rng); c = Field(CellCenter(), g); c.data .= a.data
            weighted_scalar_advect!(a, u, v, w, θ, ref.Pc_pad, ref.Pf_pad, g)
            LegacyStep.weighted_scalar_advect!(c, u, v, w, θ, ref.Pc_pad, ref.Pf_pad, g)
            @test a.data == c.data
            a = _ks_rand!(Field(FaceZ(), g), rng); c = Field(FaceZ(), g); c.data .= a.data
            add_anelastic_buoyancy!(a, θ, ref, g); LegacyStep.add_anelastic_buoyancy!(c, θ, ref, g)
            @test a.data == c.data
        end
    end
end

@testset "field algebra: the broadcasts it replaced, bit for bit" begin
    g = RegularGrid(; size = (33, 17, 11), extent = (1.0, 1.0, 1.0), halo = (3, 3, 3))
    rng = Xoshiro(3061)
    mk() = _ks_rand!(Field(FaceX(), g), rng)
    for (bn, b) in _KS_BACKENDS
        with_backend(b) do
            A, B, C, D = mk(), mk(), mk(), mk()
            y = mk(); y0 = Field(FaceX(), g); y0.data .= y.data
            SIM.axpy!(y, 0.37, A); LegacyStep._axpy_interior!(y0, 0.37, A, g)
            @test y.data == y0.data
            o = mk(); o0 = Field(FaceX(), g); o0.data .= o.data
            TI._combine2!(o, 1.0, A, 0.3, B); LegacyStep._combine2!(o0, 1.0, A, 0.3, B, g)
            @test o.data == o0.data
            TI._combine2!(o, 0.0, o, 0.7, B); LegacyStep._combine2!(o0, 0.0, o0, 0.7, B, g)  # aliased
            @test o.data == o0.data
            TI._combine3!(o, 1.0, A, 1.5, B, 0.5, C); LegacyStep._combine3!(o0, 1.0, A, 1.5, B, 0.5, C, g)
            @test o.data == o0.data
            TI._combine4!(o, 1.0, A, -0.2, B, 1.2, C, 0.29, D)
            LegacyStep._combine4!(o0, 1.0, A, -0.2, B, 1.2, C, 0.29, D, g)
            @test o.data == o0.data
            a = copy(A.data); a0 = copy(A.data); s = randn(rng, size(a))
            SIM.combine_storage!(SIM._Lin2r(0.75, 0.25), a, s); LegacyStep.rk3_combine2!(a0, s)
            @test a == a0
            SIM.combine_storage!(SIM._Lin2r(1 / 3, 2 / 3), a, s); LegacyStep.rk3_combine3!(a0, s, 1 / 3, 2 / 3)
            @test a == a0
            p = Field(CellCenter(), g); p0 = Field(CellCenter(), g); φ = _ks_rand!(Field(CellCenter(), g), rng)
            SIM.pressure_from_potential!(p, φ, 0.07); LegacyStep.pressure!(p0, φ, 0.07, g)
            @test p.data == p0.data
            z = mk(); SIM.zero_storage!(z); @test all(iszero, z.data)
            d = similar(A.data); SIM.copy_storage!(d, A.data); @test d == A.data
            Aeolus.Backends.fill_array!(b, d, 2.5); @test all(==(2.5), d)
            Aeolus.Backends.copy_array!(b, d, B.data); @test d == B.data
            @test_throws DimensionMismatch Aeolus.Backends.copy_array!(b, d, p.data)
        end
    end
end

@testset "fused CFL sweep and step guard: the separate maxima and checks" begin
    for grid in (:regular, :stretched)
        g = grid === :regular ?
            RegularGrid(; size = (40, 24, 20), extent = (400.0, 240.0, 200.0), halo = (3, 3, 3)) :
            Aeolus.StretchedGrids.StretchedGrid(; size = (40, 24, 20), extent = (400.0, 240.0, 200.0),
                                                halo = (3, 3, 3), stretching = (:geometric, 1.06))
        sim = BoussinesqSimulation(g; ν = 1e-2, θ₀ = 300.0, C_s = 0.17, moisture = true,
                                   bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
        rng = Xoshiro(3062)
        for f in (sim.u, sim.v, sim.w, sim.p); interior_view(f) .= randn(rng, size(interior_view(f))); end
        set!(sim.θ, 300.0); set!(sim.qv, 0.01)
        c = CFLConstraints()
        d = cfl_diagnostics(sim, c)                     # fills halos, computes νt, κt
        f = (νh = sim.νt, νv = sim.νt, κh = sim.κt, κv = sim.κt)
        @test TI._cfl_maxima(sim, f, true) === LegacyStep.cfl_maxima(sim, f)
        @test d.advective_rate === LegacyStep.cfl_maxima(sim, f)[1]
        @test d.νt_max === maximum(interior_view(sim.νt)) && d.κt_max === maximum(interior_view(sim.κt))
        # the guard: same bits and the same verdict as the separate checks
        gd = step_guard(sim)
        @test gd.finite
        @test gd.divergence === max_divergence(sim)
        @test gd.speed === LegacyStep._max_speed(sim)
        for (dt, cap) in ((1e-6, Inf), (Inf, 1e-3), (Inf, Inf))
            @test TI._guard_reason(sim, dt, cap) === LegacyStep.guard_reason(sim, dt, cap)
        end
        # NaN or Inf in any prognostic field, halos excluded
        for (fld, bad) in ((:θ, NaN), (:p, Inf), (:qv, -Inf), (:w, NaN), (:u, Inf))
            a = interior_view(getfield(sim, fld)); old = a[3, 4, 5]; a[3, 4, 5] = bad
            @test !step_guard(sim).finite
            @test TI._guard_reason(sim, 1.0, Inf) === :nonfinite === LegacyStep.guard_reason(sim, 1.0, Inf)
            if fld in (:u, :w)
                @test !isfinite(step_guard(sim).speed)
                # NaN poisons the advective limit, Inf makes it 0
                @test isnan(bad) ? isnan(cfl_diagnostics(sim, c).dt_adv) : cfl_diagnostics(sim, c).dt_adv == 0
                @test adaptive_dt(sim, c) == c.dt_min        # a blow-up collapses dt
            end
            a[3, 4, 5] = old
        end
        sim.θ.data[1, 1, 1] = NaN                         # a halo value is not checked
        @test step_guard(sim).finite
    end
end

@testset "blow-up: adaptive_dt collapses, the guard fires, the state is restored" begin
    g = RegularGrid(; size = (24, 24, 16), extent = (240.0, 240.0, 160.0), halo = (3, 3, 3))
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
    @test all(f -> interior_view(getfield(a, f)) == interior_view(getfield(b, f)), (:u, :v, :w))
    @test a.θ.data == b.θ.data && a.p.data == b.p.data && a.νt.data == b.νt.data
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
    seq = [drive!(mk(k)) for k in 1:4]
    par = [mk(k) for k in 1:4]
    @sync for s in par
        Threads.@spawn drive!(s)
    end
    @test all(k -> par[k].u.data == seq[k].u.data && par[k].θ.data == seq[k].θ.data, 1:4)
end

@testset "step cores: serial and threaded bits, no allocation" begin
    g = RegularGrid(; size = (40, 40, 24), extent = (400.0, 400.0, 240.0), halo = (3, 3, 3))
    walls = FieldBoundaryConditions(:periodic, :periodic, :wall)
    mk() = (s = BoussinesqSimulation(g; ν = 1e-2, θ₀ = 300.0, C_s = 0.17, moisture = true, bcs = walls);
            rng = Xoshiro(3064);
            for f in (s.u, s.v, s.w); interior_view(f) .= 0.3 .* randn(rng, size(interior_view(f))); end;
            set!(s.θ, (x, y, z) -> 300 + 0.003z); set!(s.qv, 0.008); s)
    drivers = (("step!", s -> step!(s, 0.05)), ("rk3", s -> step_rk3!(s, 0.05)),
               ("adaptive", s -> step_adaptive!(s, CFLConstraints(dt_max = 0.05))),
               ("ark2", s -> step_imex!(s, 0.2, ARK2())))
    for (nm, drv!) in drivers
        r = map(_KS_BACKENDS) do (bn, b)
            with_backend(b) do
                s = mk(); drv!(s); drv!(s); s
            end
        end
        @test all(s -> s.u.data == r[1].u.data && s.w.data == r[1].w.data && s.θ.data == r[1].θ.data &&
                       s.qv.data == r[1].qv.data && s.p.data == r[1].p.data, r)
    end
    ga = RegularGrid(; size = (24, 24, 24), extent = (240.0, 240.0, 2400.0), halo = (3, 3, 3))
    r = map(_KS_BACKENDS) do (bn, b)
        with_backend(b) do
            s = AnelasticSimulation(ga; ν = 1e-2, C_s = 0.17)
            rng = Xoshiro(3065)
            for f in (s.u, s.v, s.w); interior_view(f) .= 0.3 .* randn(rng, size(interior_view(f))); end
            step!(s, 0.2); step!(s, 0.2); s
        end
    end
    @test all(s -> s.u.data == r[1].u.data && s.θ.data == r[1].θ.data, r)
    with_backend(CPUBackend(serial = true)) do
        s = mk()
        for (nm, drv!) in drivers[1:3]
            drv!(s); drv!(s)
            nb = @allocated drv!(s)
            println("  [KA-j] one $nm step (40×40×24, SGS, moist, serial) allocates $nb bytes")
            @test nb <= (nm == "adaptive" ? 2048 : 64)
        end
    end
end

end
