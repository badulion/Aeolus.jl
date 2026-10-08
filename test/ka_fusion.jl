# SC-d (#404): kernel fusion in the step cores.
#
#   1. step! and step_rk3! with the fused passes (write-only tendency
#      launches, the divergence into the solver's buffer, one correction pass
#      that also writes p and the RK3 combination, the halo shell) give the
#      bits of the pre-fusion cores (test/support/legacy_fusion.jl), for
#      every field of the simulation, halos included: triply periodic (FFT
#      solver) and walls (cosine solver), with and without viscosity,
#      moisture, buoyancy, Coriolis, surface flux, sponge, the SGS closure,
#      body-force and post-predictor hooks, a custom momentum advection (the
#      unfused path), a stretched grid (the column solver) and open
#      boundaries (the open-face check);
#   2. on the serial, threshold-0 and default backends;
#   3. the fused correction leaves a discretely solenoidal field
#      (max |∇·u| at round-off) and RK3 keeps its order of accuracy in time
#      on a viscous Taylor–Green vortex (the combination is in the right
#      place: physics, not just bits).

using Test
using Aeolus
using Aeolus.Fields: Field, set!, interior_view
using Aeolus.Grids: interior, CellCenter, FaceX, FaceY, FaceZ
using Aeolus.Backends: CPUBackend, with_backend
using Aeolus.Simulations: BoussinesqSimulation, step_rk3!, max_divergence, kinetic_energy
using Aeolus.BoundaryConditions: FieldBoundaryConditions, fill_halos!, ProfileBoundaryData,
                                 open_boundary_conditions
using Aeolus.Operators: momentum_advect!
using Random: Xoshiro, randn

isdefined(@__MODULE__, :LegacyFusion) || include(joinpath(@__DIR__, "support", "legacy_fusion.jl"))

const _KF_BACKENDS = (("serial", CPUBackend(serial = true)), ("threshold 0", CPUBackend(threshold = 0)),
                      ("default", CPUBackend()))

# every array of the state and the scratch the step writes. A hook-free
# core (no SGS, buoyancy, Coriolis, flux, sponge, hooks or open faces) takes
# the blocked sweep of SC-e (#405), which keeps the tendencies in its own
# buffers: there the tendency fields are not written and only the state is
# compared (test/ka_blocking.jl).
_kf_fields(s) = (s.u, s.v, s.w, s.p, s.θ, s.qv, s.tu, s.tv, s.tw, s.tθ, s.tqv, s.φ)
_kf_state(s) = (s.u, s.v, s.w, s.p, s.θ, s.qv, s.φ)
_kf_same(a, b; fields = _kf_fields) = all(i -> fields(a)[i].data == fields(b)[i].data, eachindex(fields(a))) &&
                                      a.t == b.t

function _kf_seed!(s; U = 0.3, seed = 404)
    rng = Xoshiro(seed)
    for f in (s.u, s.v, s.w)
        interior_view(f) .= U .* randn(rng, size(interior_view(f)))
    end
    interior_view(s.θ) .= 300 .+ 0.1 .* randn(rng, size(interior_view(s.θ)))
    s.has_moisture && (interior_view(s.qv) .= 0.008 .+ 1e-4 .* randn(rng, size(interior_view(s.qv))))
    return s
end

# a hook that adds a tendency and one that corrects the predicted velocity
_kf_force!(s, Δt) = (interior_view(s.tu) .+= 1e-3; nothing)
_kf_post!(s, Δt) = (interior_view(s.w)[:, :, 1] .*= 0.5; nothing)
# the built-in operator behind a different function: the unfused path
_kf_custom_adv!(tu, tv, tw, u, v, w) = momentum_advect!(tu, tv, tw, u, v, w)

@testset "SC-d: kernel fusion in the step cores (#404)" begin

g = RegularGrid(; size = (20, 12, 10), extent = (200.0, 120.0, 100.0), halo = (3, 3, 3))
walls = FieldBoundaryConditions(:periodic, :periodic, :wall)
cases = (
    ("periodic, viscous (DNS)",   () -> BoussinesqSimulation(g; ν = 1e-2), NamedTuple()),
    ("periodic, inviscid",        () -> BoussinesqSimulation(g), NamedTuple()),
    ("walls, all terms + SGS",    () -> BoussinesqSimulation(g; ν = 1e-2, θ₀ = 300.0, f_coriolis = 1e-4,
                                           C_s = 0.17, moisture = true, θ_flux_bottom = 0.1,
                                           sponge_α_max = 0.01, sponge_z_start = 70.0, bcs = walls),
                                   NamedTuple()),
    ("walls, hooks",              () -> BoussinesqSimulation(g; ν = 1e-2, θ₀ = 300.0, moisture = true, bcs = walls),
                                   (body_force! = _kf_force!, post_predictor! = _kf_post!)),
)
@testset "$name: step! and step_rk3! give the pre-fusion bits" for (name, mk, kw) in cases
    for (bn, b) in _KF_BACKENDS
        with_backend(b) do
            for (drv, ref) in ((step!, LegacyFusion.step!), (step_rk3!, LegacyFusion.step_rk3!))
                a = _kf_seed!(mk()); r = _kf_seed!(mk())
                for _ in 1:3
                    drv(a, 0.5; kw...); ref(r, 0.5; kw...)
                end
                @test _kf_same(a, r; fields = startswith(name, "periodic") ? _kf_state : _kf_fields)
            end
        end
    end
end

@testset "step! with a custom momentum advection: the unfused path, the same bits" begin
    mk() = _kf_seed!(BoussinesqSimulation(g; ν = 1e-2, θ₀ = 300.0, moisture = true, bcs = walls))
    a, r, f = mk(), mk(), mk()
    for _ in 1:3
        step!(a, 0.5; momentum_advection = _kf_custom_adv!)
        LegacyFusion.step!(r, 0.5; momentum_advection = _kf_custom_adv!)
        step!(f, 0.5)
    end
    @test _kf_same(a, r)
    @test _kf_same(a, f)        # the fused and the unfused path agree too
end

@testset "stretched grid (the column solver's source buffer): the same bits" begin
    sg = StretchedGrid(; size = (12, 12, 16), extent = (240.0, 240.0, 600.0), stretching = (:tanh, 1.5))
    bcs = FieldBoundaryConditions(:periodic, :periodic; bottom = :noslip, top = :wall)
    @test Aeolus.PressureSolvers.source_buffer(BoussinesqSimulation(sg; bcs = bcs).solver) isa Array
    # projected first (one step! of the pre-fusion core): an RK3 step combines
    # with the step-start state, solenoidal only if that state is
    mk() = (s = _kf_seed!(BoussinesqSimulation(sg; ν = 1e-2, θ₀ = 300.0, C_s = 0.17, bcs = bcs));
            LegacyFusion.step!(s, 0.1); s)
    for (drv, ref) in ((step!, LegacyFusion.step!), (step_rk3!, LegacyFusion.step_rk3!))
        a, r = mk(), mk()
        for _ in 1:3
            drv(a, 0.5); ref(r, 0.5)
        end
        @test _kf_same(a, r)
        @test max_divergence(a) < 1e-10
    end
end

@testset "open x faces: the open-face check and the stage combination, the same bits" begin
    mc = RegularGrid(size = (24, 1, 12), extent = (24.0, 1.0, 12.0), origin = (0.0, 0.0, 0.0),
                     halo = (3, 3, 3))
    data = ProfileBoundaryData(u = 1.0, v = 0.0, w = 0.0, θ = 300.0)
    mk() = (s = BoussinesqSimulation(mc; ν = 1e-3,
                                     bcs = open_boundary_conditions(mc, data; x = :open, y = :periodic,
                                                                    top = :wall));
            _kf_seed!(s; U = 0.1); interior_view(s.u) .+= 1.0; LegacyFusion.step!(s, 0.05); s)
    for (drv, ref) in ((step!, LegacyFusion.step!), (step_rk3!, LegacyFusion.step_rk3!))
        a, r = mk(), mk()
        for _ in 1:4
            drv(a, 0.2); ref(r, 0.2)
        end
        @test _kf_same(a, r)
        @test max_divergence(a) < 1e-10
    end
end

@testset "physics: solenoidal after the fused correction, RK3 third order in time" begin
    # max |∇·u| at round-off after each fused step (FFT and cosine solvers).
    # step! first: the random start is not solenoidal, and an RK3 step
    # combines the projected substages with its step-start state
    for bcs in (FieldBoundaryConditions(:periodic), walls)
        s = _kf_seed!(BoussinesqSimulation(g; ν = 1e-2, bcs = bcs); U = 1.0)
        step!(s, 0.5)
        @test max_divergence(s) < 1e-12
        step_rk3!(s, 0.5)
        @test max_divergence(s) < 1e-12
    end
    # Taylor–Green vortex, ν = 0.05 on 2π³: the energy after t = 1 against a
    # dt/4 reference; halving dt divides the error by ≈ 2³ = 8 (third order
    # in time; the spatial error is common to every run)
    gt = RegularGrid(; size = (16, 16, 16), extent = (2π, 2π, 2π), halo = (3, 3, 3))
    function tg(dt)
        s = BoussinesqSimulation(gt; ν = 0.05)
        set!(s.u, (x, y, z) -> sin(x) * cos(y) * cos(z))
        set!(s.v, (x, y, z) -> -cos(x) * sin(y) * cos(z))
        for _ in 1:round(Int, 1 / dt)
            step_rk3!(s, dt)
        end
        return kinetic_energy(s)
    end
    # (measured: ratio 6.57 from dt = 0.2 to 0.1, 7.47 from 0.1 to 0.05,
    # approaching 8; a combination in the wrong place drops to first order)
    Kref = tg(1 / 80)
    e1 = abs(tg(0.1) - Kref); e2 = abs(tg(0.05) - Kref)
    println("  [SC-d] Taylor–Green RK3: |ΔK| at dt = 0.1, 0.05: $e1, $e2 (ratio $(e1 / e2))")
    @test 6.5 < e1 / e2 < 9.5
end

end
