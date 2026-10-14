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
# AEOLUS_BENCH_PREFETCH=1 (flagship, SC-c #403): the nest's parent step of
#   the next window runs on its own task while the child steps (what a
#   run does). It is started after the warm-up — a run would
#   start it when the open window opened, during step 1's compilation —
#   and the run's horizon ends with that window, so the timed steps hold
#   the same parent work as the sequential run: one parent step, here
#   alongside the child's steps.
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
const PREFETCH = get(ENV, "AEOLUS_BENCH_PREFETCH", "0") == "1"
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

function bench_berlin(mode; nwarm = 2, nsteps = 9)
    BC = Aeolus.Urban
    CP = Aeolus.Components
    ovr = mode == "child" ? ["nesting.enabled=false"] : String[]
    cfg = BC.site_options(load_run_config(joinpath(pkgdir(Aeolus), "examples", "urban", "berlin_summer.toml");
                                          overrides = vcat(ovr, "output.variables=[]")))
    tb = @elapsed case = BC.build_case(cfg)
    model, d = case.model, case.domain
    m = case.mesh
    @printf("%s: %d × %d × %d child, -t %d, build %.1f s\n", mode, m.Nx, m.Ny, m.Nz, NT, tb)
    dtw = 0.0
    for _ in 1:nwarm
        dtw = CP.pick_dt(model)
        step!(model, dtw; checked = false)
    end
    nest = BC.find_coupling(case, :nesting)
    wp0 = nest === nothing ? 0.0 : nest.wall_parent
    nw0 = nest === nothing ? 0 : nest.windows
    pt0 = nest === nothing ? Dict{Symbol,Float64}() : copy(nest.parent.timing)
    if PREFETCH && nest !== nothing && nest.window !== nothing
        case.clock.t_stop = Inf; case.clock.stop_margin = 0.0
        BC._prefetch_parent!(nest, case, dtw)
        nest.run.ahead === nothing || (case.clock.t_stop = nest.window.t_child_end + nest.run.ahead.dt)
        @printf("  parent's next window prefetched: %s\n", nest.run.ahead !== nothing)
    end
    n0 = case.clock.n
    fired = Dict{Tuple{Symbol,Symbol},Int}()
    t_dt = Float64[]; t_step = Float64[]
    empty!(case.timing)
    for _ in 1:nsteps
        dt = 0.0
        push!(t_dt, @elapsed dt = CP.pick_dt(model))
        t_old = case.sim.t
        for c in d.components, st in CP.stages(c)
            CP._due(st, t_old, t_old + dt) || continue
            k = (st.name, CP.component_name(c))
            fired[k] = get(fired, k, 0) + 1
        end
        push!(t_step, @elapsed(step!(model, dt; checked = false)))
    end
    n = case.clock.n - n0
    @printf("  %d timed steps (steps %d-%d), simulated %.2f s\n", n, n0 + 1, case.clock.n, case.sim.t)
    ms(v) = 1e3 * v
    @printf("  %-34s min %9.2f  mean %9.2f ms\n", "pick_dt", ms(minimum(t_dt)), ms(sum(t_dt) / n))
    @printf("  %-34s min %9.2f  mean %9.2f ms\n", "step!(model)", ms(minimum(t_step)), ms(sum(t_step) / n))
    row(mode, "step_dt", ms(sum(t_dt) / n), "ms")
    row(mode, "step_dt_min", ms(minimum(t_dt)), "ms")
    row(mode, "step_case", ms(sum(t_step) / n), "ms")
    row(mode, "step_case_min", ms(minimum(t_step)), "ms")
    inside = 0.0
    for ((S, name), t) in sort!(collect(case.timing); by = last, rev = true)
        # :body_force / :post_predictor run inside (:dynamics, :step!)
        S in (:body_force, :post_predictor) || (inside += t)
        @printf("  %-16s %-17s %9.3f ms/step  (%d calls)\n", S, name, ms(t / n),
                S in (:dynamics, :ledger, :body_force, :post_predictor) ? n : get(fired, (S, name), 0))
        row(mode, "$(S)/$(name)", ms(t / n), "ms")
    end
    drv = sum(t_step) - inside
    # the air-heat ledger time of the :body_force / :post_predictor hooks is
    # inside (:dynamics, :step!) and in (:ledger, :air_heat): counted twice
    @printf("  %-34s %9.3f ms/step\n", "driver (step! − entries)", ms(drv / n))
    row(mode, "driver", ms(drv / n), "ms")
    nest === nothing || wait_parent_step!(nest.run)     # the parent may be read now
    if nest !== nothing
        st = nest.run.stats
        @printf("  parent steps prefetched %d, waited for %.3f s\n", st.prefetched, st.wall_parent_wait)
        row(mode, "parent_prefetched", st.prefetched, "count")
        nw = nest.windows - nw0
        @printf("  %-34s %9.2f ms/step, %d parent windows opened\n", "parent steps (nest.wall_parent)",
                ms((nest.wall_parent - wp0) / n), nw)
        row(mode, "parent_steps", ms((nest.wall_parent - wp0) / n), "ms")
        row(mode, "parent_windows", nw, "count")
        nw > 0 && row(mode, "parent_per_window", ms((nest.wall_parent - wp0) / nw), "ms")
        pm = nest.parent.mesh
        pw = [k => ms((get(nest.parent.timing, k, 0.0) - get(pt0, k, 0.0)) / max(nw, 1))
              for k in (:total, :dynamics, :surface, :radiation)]
        @printf("  parent %d × %d × %d, per window (ms):%s\n", pm.Nx, pm.Ny, pm.Nz,
                join((@sprintf(" %s %.1f", k, v) for (k, v) in pw)))
        for (k, v) in pw
            row(mode, "parent_$(k)_per_window", v, "ms")
        end
    end
    # the cadenced entries that did not fire in the window: per call, the
    # minimum of two after one untimed call (a first call may compile a path
    # the warm-up did not take; the radiosity then reuses its factorisations,
    # so its per-call cost is measured on its own: KA-n, ka_port.md)
    dt = BC.step_dt(case)
    for e in d.components, st in CP.stages(e)
        S = st.name
        e isa BC.UrbanEntry || continue
        k = (S, CP.component_name(e))
        (get(fired, k, 0) > 0 || S in (:tendencies, :body_force, :post_predictor)) && continue
        t = tmin(() -> CP.run_stage!(e, d, Val(S), dt), 2; warm = 1)
        @printf("  per call %-16s %-17s %9.2f ms\n", S, k[2], ms(t))
        row(mode, "call:$(S)/$(k[2])", ms(t), "ms")
    end
    return nothing
end

const N_THREADED = Threads.Atomic{Int}(0)
const N_SERIAL = Threads.Atomic{Int}(0)

function count_launches(; nwarm = 2, nsteps = 9)
    BC = Aeolus.Urban
    case = BC.build_case(BC.site_options(load_run_config(joinpath(pkgdir(Aeolus), "examples", "urban", "berlin_summer.toml"))))
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
