# KA-o (#308): the scaling report of the KernelAbstractions port (epic #291).
# One process per thread count; run each mode at -t 1, 4, 12 and 24 on an
# idle machine and compare (docs/src/design/ka_port.md, "Scaling report").
#
#   julia -t N --project=. benchmarks/bench_ka_scaling.jl flagship [nwarm=2] [nsteps=9]
#   julia -t N --project=. benchmarks/bench_ka_scaling.jl child    [nwarm=2] [nsteps=9]
#   julia -t N --project=. benchmarks/bench_ka_scaling.jl dns      [N=256] [nwarm=1] [nsteps=3]
#   julia -t N --project=. benchmarks/bench_ka_scaling.jl kernels  [64,64,32] [256,256,256]
#   julia -t N --project=. benchmarks/bench_ka_scaling.jl launch
#   julia -t N --project=. benchmarks/bench_ka_scaling.jl launches [nwarm=2] [nsteps=9]
#
# The FFTW planner is the run's (MEASURE by default). Planning a threaded
# 256³ c2r transform under MEASURE segfaults intermittently at 10-14
# threads (#397), so run `kernels` and `dns` with AEOLUS_FFTW_PLANNER=estimate.
#
# flagship: the Berlin case `examples/urban/berlin_summer.toml`: 64 × 64 × 64 child, nested
#   in its Aeolus parent (the #265 default), no output file. `nwarm` steps
#   (compilation), then `nsteps` timed steps. Per step: the pick
#   (`Components.pick_dt(model)`) and the step (`step!(model, dt)`, the
#   urban site's step through the pipeline, ONE-k #474; `step_case!`
#   before) timed around the call (min and mean), and every stage entry's
#   wall time (`site.timing`) as ms per timed step. "driver" is the step
#   minus the entries it contains (the pipeline, the cadence tests, the
#   sanity check). Entries whose cadence did not fire in the
#   timed window (radiation, soil, ...) are then timed per call: two more
#   calls each, the minimum. The issue asked for 300 + 300 steps; the
#   default case fails near step 12 (#392) and long Berlin runs are paused,
#   so the default is 2 + 9.
# child: the same with the nest switched off (the child alone, periodic):
#   flagship − child is the cost of the parent and the nest exchange.
# dns: the 256³ decaying-turbulence DNS of the former a-priori record
#   (dns256.toml and test/fixtures/dns256_apriori.toml, removed by ONE-g
#   #456; examples/dns_vs_les/dns.toml is the DNS run today)
#   (`dns_step!`: step_rk3! at CFL 0.5 plus the per-step K / ε / skewness
#   history). Both are timed per step; that run was 1264 steps, so its DNS
#   time was 1264 × the `dns_step!` time. Needs about 8 GB at 256³.
# kernels: idealised kernels on seeded fields (min of repeats, µs), with an
#   effective bandwidth from the arrays each one streams (8 B per point per
#   array read or written): memory-bound kernels stop where the bandwidth
#   does, compute-bound ones keep scaling.
# launches: the number of launches per flagship step that run threaded and
#   serially (counted at `Backends.runs_serially`, which every launch and
#   reduction asks once; run at -t > 1). With the per-launch cost from
#   `launch`, it bounds the launch overhead of a step.
# launch: the KA launch overhead: a no-op body launched threaded
#   (threshold = 0) over 1 to 4096 rows, against the serial loop.
#
# flagship (NEST-p3 #483): the site nested in its 6 km parent domain, timed
#   per window of the model's nest engine (the parent's step, the site's
#   steps to its end); child: the site alone (its nest off).
#
# Each mode prints readable lines and machine-readable `ROW` lines
# (`ROW <mode> <threads> <name> <value> <unit>`).

using Aeolus
using Aeolus.Fields: Field, set!, interior_view
using Aeolus.Grids: CellCenter, FaceX, FaceY, FaceZ
using Aeolus.Backends: CPUBackend, launch!, interior_sum, init_threads!
using Aeolus.PressureSolvers: FFTPoissonSolver, solve!
using Printf, Random

const MODE = get(ARGS, 1, "launch")
# loaded at top level, before the functions that use them run (world age)
MODE == "dns" && include(joinpath(@__DIR__, "..", "test", "fixtures", "dns_decay.jl"))

const NT = Threads.nthreads()
row(mode, name, v, unit) = @printf("ROW %s %d %s %.6g %s\n", mode, NT, name, v, unit)

function tmin(f, n; warm = 2)
    for _ in 1:warm; f(); end
    t = Inf
    for _ in 1:n
        t = min(t, @elapsed f())
    end
    return t
end

# ---------------------------------------------------------------------------
# Berlin flagship
# ---------------------------------------------------------------------------

"The Berlin summer config's model: the site nested in its parent (`flagship`), or the site alone (`child`)."
function berlin_model(mode)
    BC = Aeolus.Urban
    cfg = load_run_config(joinpath(pkgdir(Aeolus), "examples", "urban", "berlin_summer.toml");
                          overrides = ["output.variables=[]"])
    if mode == "child"
        # the site's options without its nest component: the periodic box
        c = BC.site_options(Aeolus.Config.domain_view(cfg, 2))
        c = merge(c, (; components = delete!(copy(c.components), :nesting)))
        return BC.build_case(c).model
    end
    return Aeolus.Config.build(Aeolus.SimulationDriver.SimulationDriverType(), cfg).model
end

function bench_berlin(mode; nwarm = 2, nsteps = 9)
    BC = Aeolus.Urban
    CP = Aeolus.Components
    tb = @elapsed model = berlin_model(mode)
    nested = length(model.domains) > 1
    d = model.domains[end]
    case = BC.urban_case(d)
    m = case.mesh
    @printf("%s: %d × %d × %d site, -t %d, build %.1f s\n", mode, m.Nx, m.Ny, m.Nz, NT, tb)
    # one step: a site step alone, a window of the nest
    stepper() = nested ? CP.step_adaptive!(model) : step!(model, CP.pick_dt(model); checked = false)
    for _ in 1:nwarm
        stepper()
    end
    L = nested ? Aeolus.Nesting.nest_link(model, d.id) : nothing
    wc0 = nested ? L.stats.wall_child : 0.0
    n0 = case.clock.n
    empty!(case.timing)
    t_step = Float64[]
    for _ in 1:nsteps
        push!(t_step, @elapsed stepper())
    end
    n = case.clock.n - n0
    ms(v) = 1e3 * v
    @printf("  %d timed %s, %d site steps, simulated %.2f s\n", nsteps, nested ? "windows" : "steps", n, case.sim.t)
    @printf("  %-34s min %9.2f  mean %9.2f ms\n", nested ? "window" : "step!(model)", ms(minimum(t_step)),
            ms(sum(t_step) / nsteps))
    row(mode, nested ? "window" : "step_case", ms(sum(t_step) / nsteps), "ms")
    row(mode, "site_step", ms(sum(t_step) / n), "ms")
    for ((S, name), t) in sort!(collect(case.timing); by = last, rev = true)
        @printf("  %-16s %-17s %9.3f ms/site step\n", S, name, ms(t / n))
        row(mode, "$(S)/$(name)", ms(t / n), "ms")
    end
    if nested
        # the parent's share of a window: the window less the site's steps
        wc = L.stats.wall_child - wc0
        @printf("  %-34s %9.2f ms/window\n", "parent (window − site steps)", ms((sum(t_step) - wc) / nsteps))
        row(mode, "parent_per_window", ms((sum(t_step) - wc) / nsteps), "ms")
    end
    return nothing
end

const N_THREADED = Threads.Atomic{Int}(0)
const N_SERIAL = Threads.Atomic{Int}(0)

function count_launches(; nwarm = 2, nsteps = 9)
    BC = Aeolus.Urban
    case = BC.urban_case(only(berlin_model("child").domains))
    for _ in 1:nwarm
        step!(case.model)
    end
    # count every decision from here on (the method is replaced for this process)
    @eval Aeolus.Backends runs_serially(b::CPUBackend, npoints::Integer = typemax(Int)) =
        (s = b.serial || Threads.nthreads() == 1 || npoints < b.threshold;
         Threads.atomic_add!(s ? $N_SERIAL : $N_THREADED, 1); s)
    for _ in 1:nsteps
        Base.invokelatest(step!, case.model)
    end
    @printf("launches per flagship step (-t %d, %d steps): threaded %.0f, serial %.0f\n",
            NT, nsteps, N_THREADED[] / nsteps, N_SERIAL[] / nsteps)
    row("launches", "threaded_per_step", N_THREADED[] / nsteps, "count")
    row("launches", "serial_per_step", N_SERIAL[] / nsteps, "count")
    return nothing
end

# ---------------------------------------------------------------------------
# 256³ DNS
# ---------------------------------------------------------------------------

function bench_dns(N; nwarm = 1, nsteps = 3)
    ν = 1.035 * 0.018 * (48 / N)^(4 / 3)                 # the former dns256.toml: nu = "scaled", margin 1.035
    tb = @elapsed st = Main.dns_start(; N, seed = 1, ν, spectrum = q -> q^4 * exp(-2 * (q / 1.5)^2))
    @printf("dns %d³, -t %d, ν = %.4g, start %.1f s\n", N, NT, ν, tb)
    for _ in 1:nwarm
        Main.dns_step!(st; cfl = 0.5, t_end = 2.0, t_max = 6.0)
    end
    sim = st.sim
    t_full = Float64[]; t_rk3 = Float64[]
    for _ in 1:nsteps
        push!(t_full, @elapsed Main.dns_step!(st; cfl = 0.5, t_end = 2.0, t_max = 6.0))
    end
    dt = sim.t / (length(st.hist.t) - 1)
    for _ in 1:nsteps
        push!(t_rk3, @elapsed step_rk3!(sim, dt))
    end
    # the per-step history dns_step! adds to step_rk3! (test/fixtures/dns_decay.jl)
    t_ke = tmin(() -> kinetic_energy(sim), 2; warm = 1)
    t_eps = tmin(() -> Main._dns_epsilon(sim, st.lap), 2; warm = 1)
    t_sk = tmin(() -> Main.velocity_derivative_skewness(sim), 2; warm = 1)
    @printf("  history: kinetic_energy %.3f s, ε %.3f s, skewness %.3f s\n", t_ke, t_eps, t_sk)
    row("dns$N", "kinetic_energy", t_ke, "s")
    row("dns$N", "epsilon", t_eps, "s")
    row("dns$N", "skewness", t_sk, "s")
    @printf("  dns_step!  min %8.3f s  (1264 steps: %.2f h)\n", minimum(t_full), 1264 * minimum(t_full) / 3600)
    @printf("  step_rk3!  min %8.3f s\n", minimum(t_rk3))
    row("dns$N", "dns_step", minimum(t_full), "s")
    row("dns$N", "step_rk3", minimum(t_rk3), "s")
    row("dns$N", "run_1264_steps", 1264 * minimum(t_full) / 3600, "h")
    return nothing
end

# ---------------------------------------------------------------------------
# idealised kernels
# ---------------------------------------------------------------------------

struct _Ax{T}; a::T; end
(f::_Ax)(y, x) = y + f.a * x

function bench_kernels(N)
    g = RegularGrid(; size = N, extent = (2π, 2π, 2π), halo = (3, 3, 3))
    bcs = FieldBoundaryConditions(:periodic)
    rng = Xoshiro(308)
    rf(loc) = (f = Field(loc, g); f.data .= randn(rng, size(f.data)); f)
    u, v, w, θ = rf(FaceX()), rf(FaceY()), rf(FaceZ()), rf(CellCenter())
    tu, tv, tw, tθ, νt, o = rf(FaceX()), rf(FaceY()), rf(FaceZ()), rf(CellCenter()), rf(CellCenter()), rf(CellCenter())
    src = rf(CellCenter()); interior_view(src) .-= sum(interior_view(src)) / prod(N)
    p = rf(CellCenter())
    ps = FFTPoissonSolver(g)
    np = prod(N)
    nrep = np > 1_000_000 ? 5 : 100
    # (name, kernel, arrays streamed per point)
    rows = (
        ("axpy",                () -> Aeolus.Simulations.axpy!(tu, 1e-3, u), 3),
        ("set! value",          () -> set!(o, 1.5), 1),
        ("set! sin·exp",        () -> set!(o, (x, y, z) -> sin(3x) * exp(-y) + z^2 / 7), 1),
        ("interior_sum",        () -> interior_sum(θ), 1),
        ("laplacian",           () -> laplacian(o, θ), 2),
        ("advect_centered!",    () -> advect_centered!(tθ, u, v, w, θ), 6),
        ("advect_weno_z! (5)",  () -> advect_weno_z!(tθ, u, v, w, θ), 6),
        ("momentum_advect!",    () -> momentum_advect!(tu, tv, tw, u, v, w), 9),
        ("smagorinsky νt",      () -> compute_smagorinsky_νt!(νt, u, v, w, 0.17), 4),
        ("fill_halos! (×4)",    () -> (fill_halos!(u, bcs); fill_halos!(v, bcs); fill_halos!(w, bcs); fill_halos!(θ, bcs)), 0),
        ("FFT pressure solve!", () -> solve!(p, src, ps), 0),
    )
    tag = join(N, "x")
    @printf("kernels %s, -t %d (µs, minimum of %d; GB/s from the streamed arrays)\n", tag, NT, nrep)
    for (name, f, ns) in rows
        t = tmin(f, nrep)
        bw = ns == 0 ? NaN : ns * 8 * np / t / 1e9
        @printf("  %-22s %12.1f µs  %7.1f GB/s\n", name, 1e6 * t, bw)
        row("kernels$tag", replace(name, ' ' => '_'), 1e6 * t, "us")
    end
    return nothing
end

# ---------------------------------------------------------------------------
# launch overhead
# ---------------------------------------------------------------------------

@inline _noop!(i, j, k, a) = nothing
@inline function _touch!(i, j, k, a)
    @inbounds a[i, j, k] += 1.0
    return nothing
end

function bench_launch()
    bt = CPUBackend(threshold = 0)
    bs = CPUBackend(serial = true)
    a = zeros(64, 4096, 1)
    @printf("launch overhead, -t %d (µs, minimum of 2000)\n", NT)
    for nj in (1, 4, 16, 64, 256, 1024, 4096)
        r = (1:64, 1:nj, 1:1)
        t0 = tmin(() -> launch!(bt, _noop!, r, a), 2000)
        t1 = tmin(() -> launch!(bt, _touch!, r, a), 2000)
        ts = tmin(() -> launch!(bs, _touch!, r, a), 2000)
        @printf("  %5d rows of 64: no-op threaded %7.2f   touch threaded %7.2f   touch serial %7.2f\n",
                nj, 1e6 * t0, 1e6 * t1, 1e6 * ts)
        row("launch", "noop_rows$nj", 1e6 * t0, "us")
        row("launch", "touch_threaded_rows$nj", 1e6 * t1, "us")
        row("launch", "touch_serial_rows$nj", 1e6 * ts, "us")
    end
    return nothing
end

# ---------------------------------------------------------------------------

mode = MODE
argint(i, d) = length(ARGS) >= i ? parse(Int, ARGS[i]) : d
if mode in ("flagship", "child")
    bench_berlin(mode; nwarm = argint(2, 2), nsteps = argint(3, 9))
elseif mode == "launches"
    count_launches(; nwarm = argint(2, 2), nsteps = argint(3, 9))
elseif mode == "dns"
    init_threads!()
    bench_dns(argint(2, 256); nwarm = argint(3, 1), nsteps = argint(4, 3))
elseif mode == "kernels"
    init_threads!()
    sizes = length(ARGS) >= 2 ? [Tuple(parse.(Int, split(a, ','))) for a in ARGS[2:end]] :
            [(64, 64, 32), (256, 256, 256)]
    foreach(bench_kernels, sizes)
elseif mode == "launch"
    init_threads!()
    bench_launch()
else
    error("unknown mode $mode (flagship, child, dns, kernels, launch)")
end
