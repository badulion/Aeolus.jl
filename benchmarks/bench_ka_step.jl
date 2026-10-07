# KA-j (#306): the step-core kernels against the pre-port loops
# (test/support/legacy_step.jl), and whole steps of every core.
#
#   julia -t N --project=. benchmarks/bench_ka_step.jl kernels [N1,N2,N3]
#   julia -t N --project=. benchmarks/bench_ka_step.jl steps
#
# `kernels`: minimum of 200 runs (8 at 256³-class sizes) of every ported
# kernel and of the old loop it replaces, in µs; the old loops run in the
# same process. `steps`: ms per step (minimum of 5 after 2 warm-up steps)
# of every core; it uses only pre-KA-j APIs, so the same file times the
# old code from a checkout of master (the 1-thread gate).
ENV["AEOLUS_FFTW_PLANNER"] = "estimate"
using Aeolus
using Aeolus.Fields: Field, set!, interior_view
using Aeolus.Grids: interior, CellCenter, FaceX, FaceY, FaceZ
using Aeolus.Simulations: BoussinesqSimulation, step_rk3!, max_divergence
using Aeolus.TimeIntegration: step_adaptive!, step_imex!, ARK2, CFLConstraints, cfl_diagnostics,
                              adaptive_dt
using Aeolus.Anelastic: AnelasticSimulation
using Aeolus.BoundaryConditions: FieldBoundaryConditions, fill_halos!
using Printf, Random

init_threads!()

function tmin(f, n)
    f(); f()
    t = Inf
    for _ in 1:n
        t = min(t, @elapsed f())
    end
    return t
end

function seeded_les(g; moist = true, sgs = true)
    s = BoussinesqSimulation(g; ν = 1e-2, θ₀ = 300.0, C_s = sgs ? 0.17 : 0.0, moisture = moist,
                             bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
    rng = Xoshiro(306)
    for f in (s.u, s.v, s.w); interior_view(f) .= 0.3 .* randn(rng, size(interior_view(f))); end
    set!(s.θ, (x, y, z) -> 300 + 0.003z)
    moist && set!(s.qv, 0.008)
    return s
end

include(joinpath(@__DIR__, "..", "test", "support", "legacy_step.jl"))

function bench_kernels(N)
    L = LegacyStep
    TI, S = Aeolus.TimeIntegration, Aeolus.Simulations
    g = RegularGrid(; size = N, extent = (640.0, 640.0, 3000.0), halo = (3, 3, 3))
    A = Aeolus.Anelastic
    ref = A.isothermal_reference(g; T_surf = 290.0)
    rng = Xoshiro(1)
    rf(loc; s = 1.0, o = 0.0) = (f = Field(loc, g); f.data .= o .+ s .* randn(rng, size(f.data)); f)
    u, v, w = rf(FaceX()), rf(FaceY()), rf(FaceZ()); θ = rf(CellCenter(); o = 300.0)
    tu, tv, tw, tθ, c = rf(FaceX()), rf(FaceY()), rf(FaceZ()), rf(CellCenter()), rf(CellCenter())
    B, C, D = rf(FaceX()), rf(FaceX()), rf(FaceX())
    Wc, Wf = ref.ρ̄c_pad, ref.ρ̄f_pad
    nrep = prod(N) > 300_000 ? 30 : 200
    sim = seeded_les(g); fill_halos!(sim.u, sim.bcs); fill_halos!(sim.v, sim.bcs); fill_halos!(sim.w, sim.bcs)
    cfl_diagnostics(sim)
    f = (νh = sim.νt, νv = sim.νt, κh = sim.κt, κv = sim.κt)
    old_cfl() = (TI._max_advective_rate(sim.u.data, sim.v.data, sim.w.data, g),
                 Aeolus.Backends.interior_max(f.νh), Aeolus.Backends.interior_max(f.νv),
                 Aeolus.Backends.interior_max(f.κh), Aeolus.Backends.interior_max(f.κv))
    IAF = Aeolus.Backends.interior_all_finite
    old_guard() = (IAF(sim.u) && IAF(sim.v) && IAF(sim.w) && IAF(sim.θ) && IAF(sim.p) && IAF(sim.qv),
                   max_divergence(sim), max(TI._max_abs(sim.u), TI._max_abs(sim.v), TI._max_abs(sim.w)))
    a = copy(u.data); s0 = copy(u.data)
    rows = (
        ("weighted divergence",  () -> L.weighted_divergence!(c, u, v, w, Wc, Wf, g),
                                 () -> A.weighted_divergence!(c, u, v, w, Wc, Wf, g)),
        ("weighted momentum adv", () -> L.weighted_momentum_advect!(tu, tv, tw, u, v, w, Wc, Wf, g),
                                  () -> A.weighted_momentum_advect!(tu, tv, tw, u, v, w, Wc, Wf, g)),
        ("weighted scalar adv",  () -> L.weighted_scalar_advect!(tθ, u, v, w, θ, Wc, Wf, g),
                                 () -> A.weighted_scalar_advect!(tθ, u, v, w, θ, Wc, Wf, g)),
        ("anelastic buoyancy",   () -> L.add_anelastic_buoyancy!(tw, θ, ref, g),
                                 () -> A.add_anelastic_buoyancy!(tw, θ, ref, g)),
        ("axpy (interior)",      () -> L._axpy_interior!(tu, 1e-3, B, g), () -> S.axpy!(tu, 1e-3, B)),
        ("combine4 (interior)",  () -> L._combine4!(tu, 1.0, B, 0.2, C, 0.3, D, 0.1, u, g),
                                 () -> TI._combine4!(tu, 1.0, B, 0.2, C, 0.3, D, 0.1, u)),
        ("RK3 combine (storage)", () -> L.rk3_combine2!(a, s0), () -> S.combine_storage!(S._Lin2r(0.75, 0.25), a, s0)),
        ("CFL maxima",           old_cfl, () -> TI._cfl_maxima(sim, f, true)),
        ("step guard",           old_guard, () -> TI.step_guard(sim)),
    )
    @printf("kernels %s, -t %d (µs, minimum of %d)\n", string(N), Threads.nthreads(), nrep)
    for (name, old, new) in rows
        to = tmin(old, nrep) * 1e6; tn = tmin(new, nrep) * 1e6
        @printf("  %-24s old %9.1f  new %9.1f  new/old %5.2f\n", name, to, tn, tn / to)
    end
end

function bench_steps()
    g64 = RegularGrid(; size = (64, 64, 64), extent = (2π, 2π, 2π), halo = (3, 3, 3))
    dns = BoussinesqSimulation(g64; ν = 1e-3, θ₀ = 300.0)
    rng = Xoshiro(64)
    for f in (dns.u, dns.v, dns.w); interior_view(f) .= 0.1 .* randn(rng, size(interior_view(f))); end
    set!(dns.θ, 300.0)
    g = RegularGrid(; size = (64, 64, 32), extent = (640.0, 640.0, 320.0), halo = (3, 3, 3))
    ga = RegularGrid(; size = (64, 64, 32), extent = (640.0, 640.0, 3000.0), halo = (3, 3, 3))
    an = AnelasticSimulation(ga; ν = 1e-2, C_s = 0.17)
    for f in (an.u, an.v, an.w); interior_view(f) .= 0.3 .* randn(rng, size(interior_view(f))); end
    c = CFLConstraints(dt_max = 0.05)
    cases = (
        ("64³ DNS step!", dns, s -> step!(s, 1e-3), 5),
        ("LES step! (64×64×32, SGS, moist)", seeded_les(g), s -> step!(s, 0.05), 5),
        ("LES step_rk3!", seeded_les(g), s -> step_rk3!(s, 0.05), 5),
        ("LES step_adaptive!", seeded_les(g), s -> step_adaptive!(s, c), 5),
        ("LES adaptive_dt (step_dt)", seeded_les(g), s -> adaptive_dt(s, c), 20),
        ("LES step_imex! ARK2", seeded_les(g), s -> step_imex!(s, 0.2, ARK2()), 5),
        ("anelastic step! (64×64×32, SGS)", an, s -> step!(s, 0.2), 3),
    )
    @printf("steps, -t %d (ms, minimum)\n", Threads.nthreads())
    for (name, s, f!, n) in cases
        f!(s); f!(s)
        t = Inf
        for _ in 1:n
            t = min(t, @elapsed f!(s))
        end
        @printf("  %-36s %9.2f\n", name, t * 1e3)
    end
end

mode = get(ARGS, 1, "steps")
if mode == "kernels"
    sizes = length(ARGS) >= 2 ? [Tuple(parse.(Int, split(a, ','))) for a in ARGS[2:end]] : [(64, 64, 32)]
    for N in sizes
        bench_kernels(N)
    end
else
    bench_steps()
end
