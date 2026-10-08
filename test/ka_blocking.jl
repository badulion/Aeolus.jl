# SC-e (#405): temporal blocking of the hook-free step core.
#
#   1. `FiniteVolume.blocked_predictor!` (the tendencies and the predictor
#      in one sweep per k-slab) gives the bits of the fused tendency launch
#      followed by the predictor passes, for every number of slabs, on the
#      serial and threaded backends, regular and stretched grids, dry and
#      moist, with and without the RK3 stage combination; it neither reads
#      nor writes the tendency fields;
#   2. step! and step_rk3! on the hook-free core give the bits of the
#      pre-fusion cores (test/support/legacy_fusion.jl) for the state, and a
#      core with a tendency term (buoyancy) still takes the unblocked path;
#   3. physics: a small-amplitude viscous Taylor–Green vortex decays at the
#      rate of the discrete Laplacian, and the projected velocity is
#      solenoidal at round-off.

using Test
using Aeolus
using Aeolus.Fields: Field, set!, interior_view
using Aeolus.Grids: interior, CellCenter, FaceX, FaceY, FaceZ
using Aeolus.Backends: CPUBackend, with_backend
using Aeolus.Simulations: BoussinesqSimulation, step_rk3!, max_divergence, kinetic_energy
using Aeolus.BoundaryConditions: FieldBoundaryConditions, fill_halos!
using Random: Xoshiro, randn

const FVb = Aeolus.Operators.FiniteVolume
const SIMb = Aeolus.Simulations

isdefined(@__MODULE__, :LegacyFusion) || include(joinpath(@__DIR__, "support", "legacy_fusion.jl"))

const _KB_BACKENDS = (("serial", CPUBackend(serial = true)), ("threshold 0", CPUBackend(threshold = 0)),
                      ("default", CPUBackend()))

function _kb_seed!(s; U = 0.3, seed = 405)
    rng = Xoshiro(seed)
    for f in (s.u, s.v, s.w)
        interior_view(f) .= U .* randn(rng, size(interior_view(f)))
    end
    interior_view(s.θ) .= 300 .+ 0.1 .* randn(rng, size(interior_view(s.θ)))
    s.has_moisture && (interior_view(s.qv) .= 0.008 .+ 1e-4 .* randn(rng, size(interior_view(s.qv))))
    return s
end
_kb_state(s) = (s.u, s.v, s.w, s.p, s.θ, s.qv, s.φ)
_kb_same(a, b) = all(i -> _kb_state(a)[i].data == _kb_state(b)[i].data, eachindex(_kb_state(a))) &&
                 a.t == b.t

@testset "SC-e: temporal blocking of the hook-free core (#405)" begin

g = RegularGrid(; size = (20, 12, 10), extent = (200.0, 120.0, 100.0), halo = (3, 3, 3))
sg = StretchedGrid(; size = (12, 10, 16), extent = (240.0, 200.0, 600.0), stretching = (:tanh, 1.5))
walls = FieldBoundaryConditions(:periodic, :periodic, :wall)

@testset "blocked_predictor! = fused tendencies + predictor, for any slab count" begin
    for (gname, mesh, bcs) in (("regular", g, FieldBoundaryConditions(:periodic)),
                               ("stretched, walls", sg, FieldBoundaryConditions(:periodic, :periodic; bottom = :noslip, top = :wall)))
        nk = length(interior(mesh)[3])
        for moist in (false, true), rk3 in (false, true), (bn, b) in _KB_BACKENDS
            with_backend(b) do
                mk() = (s = _kb_seed!(BoussinesqSimulation(mesh; ν = 1e-2, moisture = moist, bcs = bcs));
                        SIMb.fill_state_halos!(s); s)
                Δt = 0.3
                c = SIMb._Lin2r(0.75, 0.25)
                # reference: the SC-d passes into the tendency fields, then the predictors
                r = mk()
                snap = (copy(r.θ.data), copy(r.qv.data))
                FVb._fused_tendencies!(r.tu, r.tv, r.tw, r.u, r.v, r.w, r.ν, (r.tθ, r.θ),
                                       moist ? (r.tqv, r.qv) : nothing, mesh)
                for (f, t) in ((r.u, r.tu), (r.v, r.tv), (r.w, r.tw))
                    SIMb.axpy!(f, Δt, t)
                end
                for (f, t, s0) in (moist ? ((r.θ, r.tθ, snap[1]), (r.qv, r.tqv, snap[2])) : ((r.θ, r.tθ, snap[1]),))
                    if rk3
                        SIMb._advance_scalar!(f, Δt, t, (c = c, ws = nothing), s0)   # (+ the halo shell; compared on the interior)
                    else
                        SIMb.axpy!(f, Δt, t)
                    end
                end
                for ns in 1:min(5, nk ÷ 2)
                    a = mk()
                    for t in (a.tu, a.tv, a.tw, a.tθ, a.tqv)
                        fill!(t.data, NaN)                 # neither read nor written
                    end
                    bufs = FVb.sweep_buffers(nothing, a.u.data, moist ? 5 : 4, ns, nk)
                    ps = rk3 ? SIMb._AxpyStage(SIMb._Axpy(Δt), c) : SIMb._Axpy(Δt)
                    s1 = (a.θ, ps, rk3 ? snap[1] : nothing)
                    s2 = moist ? (a.qv, ps, rk3 ? snap[2] : nothing) : nothing
                    FVb.blocked_predictor!(bufs, a.u, a.v, a.w, a.ν, SIMb._Axpy(Δt), s1, s2, mesh)
                    same = all(f -> interior_view(f[1]) == interior_view(f[2]),
                               ((a.u, r.u), (a.v, r.v), (a.w, r.w), (a.θ, r.θ), (a.qv, r.qv)))
                    same || println("  [SC-e] mismatch: $gname moist=$moist rk3=$rk3 $bn ns=$ns")
                    @test same
                    @test a.u.data == r.u.data && a.v.data == r.v.data && a.w.data == r.w.data
                    @test all(t -> all(isnan, t.data), (a.tu, a.tv, a.tw, a.tθ, a.tqv))
                end
            end
        end
    end
    # the buffers are reused when they fit, and reject a slab count they do not
    s = BoussinesqSimulation(g; ν = 1e-2)
    b2 = FVb.sweep_buffers(nothing, s.u.data, 4, 2, 10)
    @test FVb.sweep_buffers(b2, s.u.data, 4, 2, 10) === b2
    @test FVb.sweep_buffers(b2, s.u.data, 4, 3, 10) !== b2
    @test_throws DimensionMismatch FVb.blocked_predictor!(FVb.sweep_buffers(nothing, s.u.data, 4, 6, 10),
                                                          s.u, s.v, s.w, s.ν, SIMb._Axpy(0.1),
                                                          (s.θ, SIMb._Axpy(0.1), nothing), nothing, g)
    @test FVb.sweep_slabs(CPUBackend(serial = true), 10^9, 256) == 1
    @test FVb.sweep_slabs(CPUBackend(threshold = 0), 10^9, 256) == min(Threads.nthreads(), 128)
    @test FVb.sweep_slabs(CPUBackend(threshold = 0), 10^9, 3) == 1
end

cases = (
    ("periodic, viscous (DNS)",   () -> BoussinesqSimulation(g; ν = 1e-2)),
    ("periodic, inviscid",        () -> BoussinesqSimulation(g)),
    ("periodic, moist",           () -> BoussinesqSimulation(g; ν = 1e-2, moisture = true)),
    ("walls (cosine solver)",     () -> BoussinesqSimulation(g; ν = 1e-2, bcs = walls)),
    ("stretched, walls (column)", () -> BoussinesqSimulation(sg; ν = 1e-2,
                                            bcs = FieldBoundaryConditions(:periodic, :periodic; bottom = :noslip, top = :wall))),
)
@testset "$name: step! and step_rk3! give the pre-fusion bits" for (name, mk) in cases
    for (bn, b) in _KB_BACKENDS
        with_backend(b) do
            for (drv, ref) in ((step!, LegacyFusion.step!), (step_rk3!, LegacyFusion.step_rk3!))
                a = _kb_seed!(mk()); r = _kb_seed!(mk())
                @test SIMb._blocked_core(a, nothing, nothing, nothing, Aeolus.Operators.momentum_advect!)
                fill!(a.tu.data, NaN)
                for _ in 1:3
                    drv(a, 0.5); ref(r, 0.5)
                end
                @test _kb_same(a, r)
                @test all(isnan, a.tu.data)          # the blocked path: tendencies unused
            end
        end
    end
end

@testset "a tendency term or a hook keeps the unblocked path" begin
    for (kw, hook) in (((θ₀ = 300.0,), NamedTuple()), ((f_coriolis = 1e-4,), NamedTuple()),
                       ((C_s = 0.17,), NamedTuple()),
                       (NamedTuple(), (body_force! = (s, Δt) -> nothing,)),
                       (NamedTuple(), (post_predictor! = (s, Δt) -> nothing,)))
        a = _kb_seed!(BoussinesqSimulation(g; ν = 1e-2, kw...)); r = _kb_seed!(BoussinesqSimulation(g; ν = 1e-2, kw...))
        fill!(a.tu.data, NaN)
        step!(a, 0.5; hook...); LegacyFusion.step!(r, 0.5; hook...)
        @test !any(isnan, interior_view(a.tu))       # the tendencies were written
        @test _kb_same(a, r)
    end
end

@testset "allocation: the blocked step reuses its buffers" begin
    s = _kb_seed!(BoussinesqSimulation(g; ν = 1e-2))
    step_rk3!(s, 0.1); step_rk3!(s, 0.1)
    bufs = s.workspaces.sweep
    @test bufs isa FVb.SweepBuffers
    step_rk3!(s, 0.1)
    @test s.workspaces.sweep === bufs
    n = @allocated step_rk3!(s, 0.1)
    println("  [SC-e] step_rk3! on the blocked core allocates $n bytes")
    @test n <= 64          # (measured 16 bytes, as the SC-d core)
end

@testset "physics: the blocked core decays a Taylor–Green vortex at the discrete rate" begin
    # small amplitude (advection ~ 1e-6 of diffusion): K(t) = K₀ exp(−2ν k̃² t)
    # with k̃² the modified wavenumber of the 7-point Laplacian for k = (1, 1, 1)
    N = 16; ν = 0.05; U = 1e-3
    gt = RegularGrid(; size = (N, N, N), extent = (2π, 2π, 2π), halo = (3, 3, 3))
    s = BoussinesqSimulation(gt; ν = ν)
    set!(s.u, (x, y, z) -> U * sin(x) * cos(y) * cos(z))
    set!(s.v, (x, y, z) -> -U * cos(x) * sin(y) * cos(z))
    K0 = kinetic_energy(s)
    dt = 0.05; nsteps = 20
    for _ in 1:nsteps
        step_rk3!(s, dt)
    end
    Δ = 2π / N
    k2 = 3 * (2 / Δ * sin(Δ / 2))^2
    ratio = kinetic_energy(s) / K0 / exp(-2ν * k2 * dt * nsteps)
    println("  [SC-e] Taylor–Green K(1)/K₀ over the discrete exp(−2νk̃²t): $ratio")
    @test abs(ratio - 1) < 1e-4
    @test max_divergence(s) < 1e-12
end

end
