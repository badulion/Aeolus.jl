# One process of the thread-invariance harness (KA-a, #294; see
# test/thread_invariance.jl and docs/src/design/ka_port.md).
#
#   julia -t N --project=. test/support/thread_invariance_worker.jl OUT SECTIONS [MODE]
#
# Runs SECTIONS (comma list of primitives, dns, reductions, smoke, flagship,
# nested, fft, fv, pcg, multigrid, steps, radiation, hooks, columns,
# advection, sgs) and writes
# one line per checked quantity, `key<TAB>hash`, to OUT. The hashes are of
# the raw bits (Float64 → UInt64), so two processes agree iff the results
# are bit-identical. MODE is `default` (CPUBackend(): threaded when N > 1)
# or `serial` (CPUBackend(serial = true)).
#
# What is hashed:
#   primitives  set!, interior_map! and every reduction on seeded random
#               fields at 64×64×32 (NaN and Inf cases included);
#   dns         a 64³ Boussinesq DNS (ν > 0, buoyancy, no SGS) from a seeded
#               random state: every field of the simulation after 2 steps;
#               and the DNS fixture (test/fixtures/dns_decay.jl) at 32³,
#               3 steps: its fields and its history t, K, ε, S, budget
#               error (the reductions of #400);
#   reductions  the canonical reductions of KA-b (#296) on a seeded 64³ LES
#               state: the energy / variance budgets, kinetic energy, the
#               horizontal-mean profiles, the SGS dissipation, the CFL maxima
#               and finiteness checks, plane_reduce! (tuple, fused write),
#               and the chemistry P/L/H and wet-deposition totals with the
#               species fields they update;
#   smoke       every run of the Berlin smoke matrix (test/support/
#               smoke_matrix.jl), 12 steps (a variable-step run: to
#               12 dt, as the run loop runs it): every field of the simulation,
#               every tracer, every Float64 array of the surface bus, the
#               surface / water / air-heat ledgers, the dt sequence and the
#               pressure-solver iteration counts per step;
#   flagship    one dt (2 s: two adaptive steps, the second in the parent's
#               parent's window) of the Berlin summer config (the 64×64×64
#               Alexanderplatz run), hashed the same way;
#   nested      nested runs (KA-f, #297): one-way, two-way with a sponge,
#               two-way with Orlanski outflow faces (16³ parent, 16³
#               child at r = 2, 3 steps) and an anelastic two-way pair
#               (4 steps): every field of parent and child.
#   columns     (KA-m, #301) the column-physics kernels on seeded inputs, every
#               launch threaded (threshold 0): fog adjustment, Kessler (both
#               sedimentation schemes), dew exchange, dry deposition, the
#               Berlin aq_chem emissions, facet
#               scenes (implicit, direct, lagged), the dynamic roughness, and
#               the partials of a foreach_column! launch;
#   fft         (KA-h, #298) one solve of every spectral pressure solver —
#               FFT, CosineFFT for all 8 axis-BC combinations, Stretched for
#               the 4 lateral pairs — at 64×64×32 and 128×128×64, with FFTW
#               on the process's threads (`init_threads!`).
#   pcg         (KA-g, #300) every solver on the one CG driver: the
#               solution and the iteration count.
#   multigrid   (KA-i, #303) MultigridPoissonSolver and MGPreconditionedCG
#               at 64×64×32 and 128×128×64: periodic / Neumann / all-Neumann,
#               W-cycle, FMG, stretched z and cut-cell coefficients (a solid
#               block): the solution, the cycle count and the residual history.
#   fv          (KA-d, #299) every finite-volume kernel on seeded random
#               fields with filled halos, on a 64×64×32 RegularGrid, a
#               geometric StretchedGrid and a TerrainFollowingGrid over a
#               hill; the damping kernel with `levels` (one plane per work
#               item; its per-level sums, also at 128×128×32, where a 3-D
#               launch would race); and 3 steps of a 64×64×32
#               Smagorinsky LES on the stretched grid (every field).
#   sgs         (KA-l, #304) every SGS closure on a seeded 64×64×32 LES
#               state: compute_sgs! of the static, dynamic (plane, local,
#               masked), Lagrangian (scale-invariant, scale-dependent,
#               masked), AMD (+ its scalar diffusivity), Vreman, σ and
#               stability-corrected closures (every coefficient field);
#               3 commits of the Lagrangian memories and of the TKE and
#               Reynolds-stress closures (memories, coefficients, budget
#               statistics); the dual strain, the box filter (masked), the
#               TKE / RSM initialisations; and 3 steps of an LES with the
#               dynamic closure and with the TKE closure (every field).
#   steps       (KA-j, #306) every step core (test/support/step_cases.jl):
#               Boussinesq Euler, RK3 and adaptive (the dt sequence, the CFL
#               diagnostics, a forced retry), IMEX ARK2 / Rosenbrock2,
#               anelastic and pseudo-incompressible, a stretched grid and
#               every core with open boundaries: every field after 1-4 steps.
#   hooks       (KA-k, #307) the dynamics hooks with every launch on the KA
#               path: the IBM direct forcing, the immersed-wall operators
#               (per-face gather) on a 64×64×32 city with their
#               diagnostics, the canopy drag, the Berlin moist buoyancy, and
#               every large-scale forcing on a regular and a stretched grid.
#   advection   (KA-e, #302) the scalar-transport and momentum-advection
#               kernels of AdvectionSchemes on seeded random fields at
#               64×64×32: the flux-form sweep and face fluxes of every
#               reconstruction, advect_positive! with the limiter active
#               (periodic wrap and wall halos; its ratio field), the bare
#               limiter, the WENO-Z / skew momentum and variance-conserving
#               advection, advect_positive! on a stretched grid, and
#               boundary_scalar_flux on a 256×256×128 open box (face
#               launches on several tasks; totals, levels, correction).
#   radiation   (KA-n, #305) the radiosity view factors of a 16×14 city
#               (n ≈ 600 facets), its shortwave / longwave solves through
#               the LU cache (BLAS pinned to one thread here: LAPACK's bits
#               depend on the BLAS thread count, docs/src/design/ka_port.md)
#               and aggregate_to_grid; gas + cloud optics and the
#               multi-column two-stream solves of 6 columns; the cloud
#               optical-depth field and shadow rays; a Monte-Carlo solve
#               (12×10×10, 16 batches); 4 collision steps of 2000
#               superdroplets in 16 cells and the sub-grid fluctuations.

ENV["AEOLUS_FFTW_PLANNER"] = "estimate"    # deterministic plans (#245)
using Aeolus
using Aeolus.Fields: set!, interior_view, Field
using Aeolus.Grids: interior
using Random: Xoshiro, randn, rand
using Printf
import LinearAlgebra

const OUT = ARGS[1]
const SECTIONS = split(ARGS[2], ',')
const MODE = length(ARGS) >= 3 ? ARGS[3] : "default"
const BACKEND = MODE == "serial" ? CPUBackend(serial = true) : CPUBackend()

const RESULTS = Pair{String,String}[]

bitsof(x::AbstractArray{<:AbstractFloat}) = reinterpret(UInt64, vec(Float64.(collect(x))))
bitsof(x::AbstractArray) = vec(collect(x))
bitsof(x::AbstractFloat) = reinterpret(UInt64, Float64(x))
bitsof(x) = x
record!(key, x) = push!(RESULTS, key => string(hash(bitsof(x)); base = 16))
record_repr!(key, x) = push!(RESULTS, key => string(hash(repr(x)); base = 16))

"Every Field of `obj` (one level of its fields), by field name."
fields_of(obj) = [(n, getfield(obj, n)) for n in fieldnames(typeof(obj))
                  if getfield(obj, n) isa Field]

"Iteration counters a pressure solver exposes (multigrid / CG)."
function solver_iters(sim)
    its = Int[]
    for n in fieldnames(typeof(sim))
        x = getfield(sim, n)
        if hasproperty(x, :last_iters)
            v = getproperty(x, :last_iters)
            push!(its, v isa Base.RefValue ? v[] : Int(v))
        end
    end
    return its
end

function record_sim!(prefix, sim)
    for (n, f) in fields_of(sim)
        record!("$prefix/sim.$n", f.data)
    end
end

# --------------------------------------------------------------------------
function section_primitives()
    mesh = RegularGrid(; size = (64, 64, 32), extent = (1.0, 1.0, 0.5), halo = (3, 3, 3))
    I3 = interior(mesh)
    rng = Xoshiro(2940)
    a = Field(mesh); b = Field(mesh); c = Field(mesh)
    a.data .= randn(rng, size(a.data)) .* 10 .^ (3 .* rand(rng, size(a.data)))
    b.data .= randn(rng, size(b.data))
    set!(c, (x, y, z) -> sin(7x) * cos(5y) + z)
    record!("primitives/set!(f, g)", c.data)
    set!(c, 0.125); record!("primitives/set!(f, v)", c.data)
    y = Field(mesh); y.data .= b.data
    interior_map!((yv, xv) -> yv + 0.37 * xv, y, y, a)
    record!("primitives/interior_map! axpy", y.data)
    record!("primitives/interior_sum", interior_sum(a))
    record!("primitives/interior_dot", interior_dot(a, b))
    p = zeros(32); plane_mean!(p, a); record!("primitives/plane_mean!", p)
    pp = zeros(32); plane_partials!(pp, abs2, I3, a.data); record!("primitives/plane_partials!(abs2)", pp)
    C = column_partials(a.data, I3)
    record!("primitives/column_partials", C)
    record!("primitives/ordered_sum(columns)", ordered_sum(C))
    record!("primitives/interior_max", interior_max(a))
    for (lab, bad) in (("NaN", NaN), ("Inf", Inf), ("-Inf", -Inf))
        d = copy(a.data); d[20, 30, 10] = bad
        record!("primitives/interior_max[$lab]", interior_max(d, I3))
        record!("primitives/interior_all_finite[$lab]", interior_all_finite(d, I3))
    end
    # set fills (SC-b, #402): u, v, w, θ in two launches, closed and open faces
    G = Aeolus.Grids
    for (lab, bcs) in (("periodic", FieldBoundaryConditions(:periodic)),
                       ("walls", FieldBoundaryConditions(:periodic, :noflux, :wall)),
                       ("open", open_boundary_conditions(mesh, ProfileBoundaryData(u = 1.0, v = 0.5, w = 0.0, θ = 300.0);
                                                         normal_halo = :odd)))
        fs = (Field(G.FaceX(), mesh), Field(G.FaceY(), mesh), Field(G.FaceZ(), mesh), Field(mesh))
        foreach(f -> f.data .= randn(rng, size(f.data)), fs)
        if lab == "open"
            st = only(open_boundary_states(bcs)); st.bound[fs[4].data] = :θ
            for q in st.planes; q.data .= randn(rng, size(q.data)); end
        end
        with_backend(() -> fill_halos!(fs, bcs), BACKEND)
        for (n, f) in zip((:u, :v, :w, :θ), fs)
            record!("primitives/fill_halos!(set, $lab).$n", f.data)
        end
    end
end

# --------------------------------------------------------------------------
include(joinpath(@__DIR__, "..", "fixtures", "dns_decay.jl"))

function section_dns()
    mesh = RegularGrid(; size = (64, 64, 64), extent = (1.0, 1.0, 1.0), halo = (3, 3, 3))
    sim = BoussinesqSimulation(mesh; ν = 1e-3, θ₀ = 300.0)
    rng = Xoshiro(64)
    for f in (sim.u, sim.v, sim.w)
        interior_view(f) .= 0.1 .* randn(rng, size(interior_view(f)))
    end
    set!(sim.θ, (x, y, z) -> 300 + 2z)
    interior_view(sim.θ) .+= 0.01 .* randn(rng, size(interior_view(sim.θ)))
    dts = Float64[]
    for _ in 1:2
        t0 = sim.t
        step!(sim, 1e-3)
        push!(dts, sim.t - t0)
    end
    record_sim!("dns64", sim)
    record!("dns64/dt", dts)
    # the DNS fixture's per-step history (#400): S, ε and the CFL max on
    # the KA-b reductions, every launch threaded (threshold 0); 32³, 3 steps
    with_backend(MODE == "serial" ? BACKEND : CPUBackend(threshold = 0)) do
        st = dns_start(; N = 32)
        for _ in 1:3
            dns_step!(st)
        end
        record_sim!("dns_fixture32", st.sim)
        for k in (:t, :K, :ε, :S, :budget_err)
            record!("dns_fixture32/hist.$k", getfield(st.hist, k))
        end
    end
end

# --------------------------------------------------------------------------
function section_reductions()
    mesh = RegularGrid(; size = (64, 64, 64), extent = (1.0, 1.0, 1.0), halo = (3, 3, 3))
    sim = BoussinesqSimulation(mesh; ν = 1e-3, θ₀ = 300.0, C_s = 0.17)
    rng = Xoshiro(296)
    for f in (sim.u, sim.v, sim.w)
        interior_view(f) .= 0.3 .* randn(rng, size(interior_view(f)))
    end
    set!(sim.θ, (x, y, z) -> 300 + 2z)
    interior_view(sim.θ) .+= 0.05 .* randn(rng, size(interior_view(sim.θ)))
    step!(sim, 2e-3)
    S, TI, AS, O = Aeolus.Simulations, Aeolus.TimeIntegration, Aeolus.AdvectionSchemes, Aeolus.Operators
    for (k, v) in pairs(AS.kinetic_energy_budget(sim))
        record!("reductions/ke_budget.$k", v)
    end
    for (k, v) in pairs(AS.scalar_variance_budget(sim, sim.θ))
        record!("reductions/variance_budget.$k", v)
    end
    record!("reductions/kinetic_energy", kinetic_energy(sim))
    record!("reductions/max_divergence", max_divergence(sim))
    record!("reductions/horizontal_mean", S.horizontal_mean(sim.θ))
    record!("reductions/horizontal_variance", S.horizontal_variance(sim.θ))
    record!("reductions/resolved_flux_wθ", S.resolved_flux_wθ(sim))
    record!("reductions/resolved_tke_profile", O.resolved_tke_profile(sim.u, sim.v, sim.w))
    record!("reductions/sgs_stress_dissipation", O.sgs_stress_dissipation(sim.u, sim.v, sim.w, sim.νt))
    record!("reductions/max_advective_rate", TI._max_advective_rate(sim.u.data, sim.v.data, sim.w.data, mesh))
    record!("reductions/brunt_vaisala", TI._brunt_vaisala(sim))
    record!("reductions/max_speed", TI.step_guard(sim).speed)   # KA-j: the speed cap of the fused guard
    record!("reductions/all_finite", TI._all_finite(sim))
    # plane_reduce!: a tuple body and a fused write
    I3 = interior(mesh)
    s1, s2 = zeros(64), zeros(64)
    plane_reduce!((s1, s2), (i, j, k, a) -> (@inbounds (a[i, j, k], abs2(a[i, j, k]))), I3, sim.u.data)
    record!("reductions/plane_reduce!(tuple)", vcat(s1, s2))
    c = copy(sim.θ.data); d = zeros(64)
    plane_reduce!(d, (i, j, k, c) -> (@inbounds (x = 1e-3 * c[i, j, k]; c[i, j, k] -= x; x)), I3, c)
    record!("reductions/plane_reduce!(fused)", d); record!("reductions/plane_reduce!(fused).field", c)
    record!("reductions/enstrophy", AS.enstrophy(sim.u, sim.v, sim.w))
    c = copy(sim.θ.data)
    record!("reductions/plane_map!", plane_map!(zeros(64), (k, ir, jr, c) -> (s = 0.0;
        @inbounds(for j in jr, i in ir; x = 1e-3 * c[i, j, k]; c[i, j, k] -= x; s += x; end); s), I3, c))
    record!("reductions/plane_map!.field", c)
    record!("reductions/range_max", range_max((i, j, k, a) -> (@inbounds abs(a[i, j, k])), I3, sim.w.data))
    # chemistry: P/L/H and wet deposition, with the fields they update
    C = Aeolus.Chemistry
    n = (64, 64, 32)
    NO = 5 .+ rand(Xoshiro(1), n...); NO2 = 10 .+ rand(Xoshiro(2), n...); O3 = 40 .+ rand(Xoshiro(3), n...)
    VOC = 2 .+ rand(Xoshiro(4), n...); HNO3 = zeros(n...)
    r = C.apply_photochemistry!(NO, NO2, O3, VOC, HNO3, 30.0; μ₀ = 0.7, OH = 1e6)
    record!("reductions/photochemistry.P_L_H", [r.P_Ox, r.L_Ox, r.HNO3_formed])
    record!("reductions/photochemistry.fields", vcat(vec(NO), vec(NO2), vec(O3), vec(VOC), vec(HNO3)))
    qr = (rand(Xoshiro(5), n...) .< 0.1) .* 1e-4
    record!("reductions/wet_deposition", C.apply_wet_deposition!(HNO3, qr, 30.0; scav = 1.0))
    record!("reductions/wet_deposition.field", HNO3)
end

# --------------------------------------------------------------------------
const BC = Aeolus.Urban
include(joinpath(@__DIR__, "berlin_configs.jl"))
include(joinpath(@__DIR__, "smoke_matrix.jl"))

function run_hashed!(label, cfg; seed = true, nsteps = nothing)
    m = build_config_model(cfg)
    case = urban_site(m)
    seed && seed_fog!(case, Xoshiro(127))
    ts = BC.timestepper(case)
    variable = ts !== nothing && BC.variable_dt(typeof(ts))
    dts = Float64[]; iters = Int[]
    t_end = case.sim.t + something(nsteps, case.cfg.nsteps) * case.cfg.dt
    if length(m.domains) > 1
        # the site nested in its parent (NEST-p3 #483): the model's nest
        # engine, one window per root step, landing on t_end; the dt
        # sequence is the site's steps
        t_prev = Ref(case.sim.t)
        n_prev = Ref(case.clock.n)
        Aeolus.Components.run_to!(m, t_end; callback = (m, k) -> begin
            push!(dts, case.sim.t - t_prev[]); t_prev[] = case.sim.t
            append!(iters, solver_iters(case.sim))
        end)
        record_sim!("$label/nest_parent", m.domains[1].state)
    else
        # the run loop's horizon: a variable-step run goes to t_end, as the
        # run loop does, not nsteps steps
        clk = case.clock
        clk.t_stop = t_end
        clk.stop_margin = variable ? 1e-9 * case.cfg.dt : 0.5 * case.cfg.dt
        n = 0
        while variable ? BC.step_will_run(clk, case.sim.t) : n < something(nsteps, case.cfg.nsteps)
            n += 1
            t0 = case.sim.t
            variable ? step!(case.model, Aeolus.Components.last_step_dt(BC.step_dt(case), t_end - case.sim.t)) :
                       step!(case.model, case.cfg.dt)
            ok = !case.clock.blowup
            push!(dts, case.sim.t - t0)
            append!(iters, solver_iters(case.sim))
            ok || break
        end
        case.clock.t_stop = NaN
        foreach(c -> BC.end_run!(c, case), case.couplings)
    end
    record_sim!(label, case.sim)
    for t in case.tracers.list
        record!("$label/tracer.$(t.name)", t.field.data)
    end
    bus = case.bus
    for n in fieldnames(typeof(bus))
        x = getfield(bus, n)
        x isa AbstractArray{Float64} && record!("$label/bus.$n", x)
    end
    record_repr!("$label/ledger.surface", BC.surface_budget(case.budgets))
    record_repr!("$label/ledger.water", BC.water_budget(case.budgets))
    record_repr!("$label/ledger.air_heat", BC.heat_budget(case.budgets))
    record!("$label/dt", dts)
    record!("$label/solver_iterations", iters)
    return case
end

function section_smoke()
    for (label, cfg) in smoke_matrix(; serial = MODE == "serial")
        run_hashed!("smoke[$label]", cfg)
    end
end

function section_flagship()
    f = berlin_toml("summer")
    cfg = load_run_config(f; overrides = [site_key(f, "urban.serial") * "=" * string(MODE == "serial")])
    run_hashed!("flagship", cfg; seed = false, nsteps = 1)
end

# --------------------------------------------------------------------------
using Aeolus.Nesting

function nested_boussinesq(fb; sponge = nothing, outflow = :dirichlet)
    pm = RegularGrid(Float64; size = (16, 16, 16), extent = (1.0, 1.0, 1.0), halo = (3, 3, 3))
    nest = nested_grid(pm, ((0.25, 0.75), (0.25, 0.75), (0.25, 0.75)), 2; halo = (3, 3, 3))
    cm = nest_child_mesh(nest)
    rng = Xoshiro(329)
    p = BoussinesqSimulation(pm; ν = 1e-3)
    for f in (p.u, p.v, p.w)
        interior_view(f) .= 0.3 .* randn(rng, size(interior_view(f)))
    end
    interior_view(p.θ) .= 300 .+ randn(rng, size(interior_view(p.θ)))
    step!(p, 1e-10); p.t = 0.0
    c = BoussinesqSimulation(cm; ν = 1e-3, bcs = nest_bcs(cm; outflow))
    for (cf, pf) in ((c.u, p.u), (c.v, p.v), (c.w, p.w), (c.θ, p.θ))
        interpolate_parent_to_child!(cf, pf, nest)
    end
    kw = sponge === nothing ? (;) : (; sponge)
    return NestedRun(p, c, nest; cfl_parent = nothing, cfl_child = nothing, feedback = fb, kw...)
end

function nested_anelastic()
    A = Aeolus.Anelastic
    pm = RegularGrid(Float64; size = (12, 12, 12), extent = (1200.0, 1200.0, 1200.0), halo = (3, 3, 3))
    nest = nested_grid(pm, ((300.0, 900.0), (300.0, 900.0), (0.0, 600.0)), 2)
    cm = nest_child_mesh(nest)
    rp = A.isothermal_reference(pm; T_surf = 300.0); rc = A.isothermal_reference(cm; T_surf = 300.0)
    p = A.AnelasticSimulation(pm; ref = rp, bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
    c = A.AnelasticSimulation(cm; ref = rc, bcs = nest_bcs(cm, :nest, :nest; bottom = :wall, top = :nest))
    rng = Xoshiro(3)
    for k in axes(p.θ.data, 3), j in axes(p.θ.data, 2), i in axes(p.θ.data, 1)
        p.θ.data[i, j, k] = rp.θ̄c_pad[k] + 0.5 * exp(-((i - 7)^2 + (j - 7)^2 + (k - 5)^2) / 5)
    end
    fill!(p.u.data, 4.0)
    interior_view(p.v) .+= 0.3 .* randn(rng, size(interior_view(p.v)))
    A._project_mass_weighted!(p)
    meas = nest_measure(nest, p, c)
    for (cf, pf) in ((c.θ, p.θ), (c.u, p.u), (c.v, p.v), (c.w, p.w))
        interpolate_parent_to_child!(cf, pf, nest; measure = meas)
    end
    return NestedRun(p, c, nest; feedback = :two_way, cfl_parent = nothing, cfl_child = nothing)
end

function section_nested()
    for (label, mk, nsteps, dt) in (("one_way", () -> nested_boussinesq(:one_way), 3, 2e-3),
                                    ("two_way_sponge", () -> nested_boussinesq(:two_way; sponge = NestSponge()), 3, 2e-3),
                                    ("two_way_orlanski", () -> nested_boussinesq(:two_way; outflow = :orlanski), 3, 2e-3),
                                    ("anelastic_two_way", nested_anelastic, 4, 5.0))
        run = mk()
        for _ in 1:nsteps
            step_nested!(run; dt)
        end
        record_sim!("nested[$label]/parent", run.parent)
        record_sim!("nested[$label]/child", run.child)
    end
end

# --------------------------------------------------------------------------
function section_fft()
    PS = Aeolus.PressureSolvers
    @printf("thread-invariance worker: -t %d: FFTW plans on %d threads\n",
            Threads.nthreads(), Aeolus.fftw_policy().threads)
    for (Nx, Ny, Nz) in ((64, 64, 32), (128, 128, 64))
        tag = "$(Nx)x$(Ny)x$(Nz)"
        mesh = RegularGrid(; size = (Nx, Ny, Nz), extent = (1.0, 1.3, 0.7), halo = (3, 3, 3))
        smesh = StretchedGrid(; size = (Nx, Ny, Nz), extent = (800.0, 800.0, 3000.0),
                                halo = (3, 3, 3), stretching = (:geometric, 1.04))
        solvers = Pair{String,Any}["fft" => (mesh, PS.FFTPoissonSolver(mesh))]
        for bx in (:periodic, :neumann), by in (:periodic, :neumann), bz in (:periodic, :neumann)
            push!(solvers, "cosine($bx,$by,$bz)" =>
                  (mesh, PS.CosineFFTPoissonSolver(mesh; bc_x = bx, bc_y = by, bc_z = bz)))
        end
        for bx in (:periodic, :neumann), by in (:periodic, :neumann)
            push!(solvers, "stretched($bx,$by)" =>
                  (smesh, Aeolus.StretchedGrids.StretchedPoissonSolver(smesh; bc_x = bx, bc_y = by)))
        end
        # the direct density-weighted projection of the sound-proof cores (#309)
        dmesh = RegularGrid(; size = (Nx, Ny, Nz), extent = (1000.0, 1000.0, 3000.0), halo = (3, 3, 3))
        dref = Aeolus.Anelastic.isothermal_reference(dmesh; T_surf = 300.0)
        for bx in (:periodic, :neumann)
            push!(solvers, "direct_weighted($bx)" =>
                  (dmesh, Aeolus.Anelastic.DirectWeightedPoissonSolver(dmesh, dref; bc_x = bx, bc_y = bx)))
        end
        for (name, (m, solver)) in solvers
            src = Field(m); src.data .= randn(Xoshiro(298), size(src.data))
            p = Field(m)
            PS.solve!(p, src, solver)
            record!("fft/$tag/$name", p.data)
        end
    end
end

# --------------------------------------------------------------------------
# KA-g (#300): every solver on the one CG driver, singular (Neumann and
# periodic) and stretched-coefficient cases: the solution and the
# iteration count.
function section_pcg()
    PS, MGm, A, TI = Aeolus.PressureSolvers, Aeolus.Multigrid, Aeolus.Anelastic, Aeolus.TimeIntegration
    N = (64, 64, 32)
    g = RegularGrid(; size = N, extent = (1000.0, 1000.0, 200.0), halo = (3, 3, 3))
    smesh = StretchedGrid(; size = N, extent = (1000.0, 1000.0, 2000.0), halo = (3, 3, 3),
                          stretching = (:geometric, 1.04))
    gd = RegularGrid(; size = N, extent = (1000.0, 1000.0, 3000.0), halo = (3, 3, 3))
    ref = A.isothermal_reference(g; T_surf = 300.0); refd = A.isothermal_reference(gd; T_surf = 300.0)
    solvers = Pair{String,Any}[]
    for (bz, bxy) in ((:neumann, :periodic), (:neumann, :neumann), (:periodic, :periodic))
        push!(solvers, "cg($bxy,$bz)" => (g, PS.CGPoissonSolver(g; tol = 1e-8, maxit = 2000, bc_top = bz,
                                                                  bc_bottom = bz, bc_xy = bxy), s -> s.last_iter[]))
    end
    for (bz, bxy) in ((:neumann, :periodic), (:periodic, :periodic))
        push!(solvers, "mgcg($bxy,$bz)" => (g, MGm.MGPreconditionedCG(g; bc_x = bxy, bc_y = bxy, bc_z = bz,
                                                                      tol = 1e-10, maxit = 200), s -> s.last_iters[]))
    end
    push!(solvers, "mgcg(stretched)" => (smesh, MGm.MGPreconditionedCG(smesh; tol = 1e-10, maxit = 200),
                                         s -> s.last_iters[]))
    for (bz, bx) in ((:neumann, :periodic), (:periodic, :periodic), (:neumann, :neumann))
        push!(solvers, "dw($bx,$bz)" => (g, A.DensityWeightedPoissonSolver(g, ref; bc_z = bz, bc_x = bx, bc_y = bx,
                                                                           tol = 1e-10, maxit = 2000),
                                         A.solver_iterations))
    end
    push!(solvers, "dw(deep)" => (gd, A.DensityWeightedPoissonSolver(gd, refd; tol = 1e-10, maxit = 2000),
                                  A.solver_iterations))
    for (name, (m, solver, its)) in solvers
        src = Field(m); src.data .= randn(Xoshiro(300), size(src.data))
        p = Field(m)
        PS.solve!(p, src, solver)
        record!("pcg/$name", p.data)
        record!("pcg/$name/iterations", its(solver))
    end
    νvar = zeros(size(Field(g).data))
    for k in axes(νvar, 3), i in axes(νvar, 1)
        νvar[i, :, k] .= 2.0 * (1 + 0.5sin(i / 7) + 0.3k / N[3])
    end
    for (name, loc, bcs, ν) in (("imex(θ,periodic)", CellCenter(), FieldBoundaryConditions(:periodic), 2.0),
                                ("imex(θ,ν(x,z),walls)", CellCenter(), FieldBoundaryConditions(:periodic, :periodic, :wall), νvar),
                                ("imex(w,pinned)", FaceZ(), FieldBoundaryConditions(:periodic, :periodic, :wall), 2.0),
                                ("imex(u,moving wall)", FaceX(),
                                 FieldBoundaryConditions(Periodic(), Periodic(), Periodic(), Periodic(),
                                                         NoSlipWall(), NoSlipWall(u_wall = 1.0)), 2.0))
        f = Field(loc, g); f.data .= randn(Xoshiro(302), size(f.data))
        st = TI.solve_implicit_diffusion!(f, f, ν, 20.0, g, bcs; tol = 1e-10, maxit = 500)
        record!("pcg/$name", f.data)
        record!("pcg/$name/iterations", st.iterations)
    end
end

# --------------------------------------------------------------------------
# KA-m (#301): the column-physics kernels on seeded inputs, every launch on
# the threaded path in the default mode (threshold 0), serial in `serial`
function section_columns()
    b = MODE == "serial" ? BACKEND : CPUBackend(threshold = 0)
    MT = Aeolus.MoistThermodynamics
    # fog adjustment (64×64×32): dry, near-saturated, supersaturated and foggy cells
    n = (64, 64, 32); rng = Xoshiro(301)
    p = collect(range(1.0e5, 9.0e4, length = n[3]))
    Π = [MT.exner_function(pk, MT.PhasePartition(0.0)) for pk in p]
    θ = 285 .+ 6 .* rand(rng, n...)
    qs = [MT.saturation_specific_humidity(θ[i, j, k] * Π[k], p[k]) for i in 1:n[1], j in 1:n[2], k in 1:n[3]]
    qv = qs .* (0.85 .+ 0.2 .* rand(rng, n...))
    ql = ifelse.(rand(rng, n...) .< 0.2, 2e-4 .* rand(rng, n...), 0.0)
    tot = fog_adjust!(θ, qv, ql, p, 2.0; backend = b)
    record!("columns/fog.fields", vcat(vec(θ), vec(qv), vec(ql))); record!("columns/fog.total", tot)
    # Kessler: conversion per cell, both sedimentation schemes per column
    mesh = RegularGrid(; size = (48, 40, 32), extent = (2400.0, 2000.0, 1600.0), halo = (2, 2, 2))
    I3 = interior(mesh)
    for Vt in (:spectrum, 3.0)
        f = [Field(mesh) for _ in 1:3]
        f[1].data .= 0.01
        f[2].data[I3...] .= ifelse.(rand(Xoshiro(2), 48, 40, 32) .< 0.3, 2e-3 .* rand(Xoshiro(3), 48, 40, 32), 0.0)
        f[3].data[I3...] .= ifelse.(rand(Xoshiro(4), 48, 40, 32) .< 0.2, 1e-3 .* rand(Xoshiro(5), 48, 40, 32), 0.0)
        apply_kessler_step!(f..., 20.0; Vt = Vt, ρ_air = collect(range(1.2, 0.95, length = 32)), backend = b)
        record!("columns/kessler[$Vt]", vcat(vec(f[2].data), vec(f[3].data)))
    end
    # dew exchange and dry deposition (128×96 columns)
    Ts = 280 .+ 8 .* rand(Xoshiro(6), 128, 96)
    q = [MT.saturation_specific_humidity(T, 1.0e5) * (0.9 + 0.2r) for (T, r) in
         zip(282 .+ 4 .* rand(Xoshiro(7), 128, 96), rand(Xoshiro(8), 128, 96))]
    d = ifelse.(rand(Xoshiro(9), 128, 96) .< 0.5, 1e-3 .* rand(Xoshiro(10), 128, 96), 0.0)
    H = with_backend(() -> dew_exchange!(q, d, Ts, 1.0e5, 60.0; dz = 4.0, ρ = 1.2), b)
    record!("columns/dew", vcat(vec(q), vec(d), vec(H)))
    c = 10 .+ rand(Xoshiro(11), 128, 96, 8)
    r = Aeolus.Chemistry.apply_dry_deposition!(c, 0.01 .* rand(Xoshiro(12), 128, 96), 30.0; dz = 2.0,
                                               k = rand(Xoshiro(13), 1:8, 128, 96), backend = b)
    record!("columns/dry_deposition", vcat(vec(c), r))
    # aq_chem emissions: at one thread the site's own serial loop (only the
    # emitted terms added), threaded the column_sum launch (a 3-tuple per
    # column, zeros included); equal hashes across the counts mean the two
    # paths give the same fields and totals
    ac = BC.build_case(site_config(berlin_toml("smoke")))
    aq = BC.find_coupling(ac, :aq_chem)
    BC._aq_sun!(aq, ac)
    with_backend(() -> BC._aq_emit!(aq, ac, 2.0), b)
    record!("columns/aq_emit", vcat(vec(BC._aq_sp(aq, :NO)), vec(BC._aq_sp(aq, :NO2)), vec(BC._aq_sp(aq, :VOC)),
                                    [aq.emitted[s] for s in (:NO, :NO2, :VOC)]))
    # facet scenes: implicit (iterative, direct) and lagged. BLAS on one
    # thread in every process: the scene's longwave response is an LU solve
    # whose bits depend on the BLAS thread count, and init_threads! leaves
    # BLAS threaded at -t 1 (it pins BLAS only when Julia runs threaded), so
    # the -t 1 reference would differ from -t 4 for a reason that is not a
    # Julia-thread one (measured: response hash 3b3d… with 12 BLAS threads,
    # 517a… with 1, at -t 1 and -t 4 alike)
    nblas = LinearAlgebra.BLAS.get_num_threads()
    LinearAlgebra.BLAS.set_num_threads(1)
    hm = zeros(10, 8); hm[3:4, :] .= 12; hm[7, 2:6] .= 20
    sc = Aeolus.Radiosity.build_scene(hm, 4.0, 4.0; wall_subdivisions = 2)
    vf = Aeolus.Radiosity.view_factors(sc)
    air = SurfaceForcing(sw_down = 500.0, lw_down = 330.0, T_air = 295.0, U = 3.0, q_air = 0.008)
    for (lab, kw) in (("implicit", (;)), ("direct", (solver = :direct,)), ("lagged", (coupling = :lagged,)))
        u = UrbanFacetScene(sc, vf; T = 290.0, options = f -> (T_interior = 293.0,))
        fl = with_backend(b) do
            local fl
            for _ in 1:3; fl = step_urban_facets!(u, air, 600.0; kw...); end
            fl
        end
        record!("columns/facets[$lab]", vcat(facet_temperatures(u), [x.H for x in fl],
                                              reduce(vcat, [x.column.T for x in u.facets])))
    end
    LinearAlgebra.BLAS.set_num_threads(nblas)
    # dynamic roughness (masked)
    WM = Aeolus.WallModels
    Nx, Ny, rr = 64, 16, 4
    dx = 2π / Nx
    xf = [(i - 0.5) * dx / rr for i in 1:Nx*rr]
    hf = [0.02 * sin(3x) + 0.006 * sin(10x) + 0.004 * cos(2y) for x in xf, y in xf[1:Ny*rr]]
    st = WM.subfilter_height(hf, rr; dx = dx, dy = dx, filter = :box)
    uc = 3 .+ 0.3 .* randn(Xoshiro(14), Nx, Ny); vc = 0.3 .* randn(Xoshiro(15), Nx, Ny)
    dr = WM.DynamicRoughness(st; z0_floor = 1e-5, C_d = 0.3, mask = rand(Xoshiro(16), Nx, Ny) .< 0.85)
    with_backend(() -> WM.update_dynamic_roughness!(dr, uc, vc, 0.05), b)
    record!("columns/dynamic_roughness", vcat(dr.α, vec(dr.z0)))
    # the primitive: partials of a column launch and their fold
    A = randn(Xoshiro(17), 96, 80, 24)
    P = foreach_column!(b, (i, j, A) -> (s = 0.0; @inbounds(for k in axes(A, 3); s += A[i, j, k]; end);
                                         (s, abs(s))), (1:96, 1:80), A; partials = zeros(96 * 80, 2))
    record!("columns/foreach_column!", vcat(vec(P), column_sums(P)))
end

# --------------------------------------------------------------------------
# KA-i (#303): the multigrid V-cycle on launch! (colour-indexed RBGS, residual,
# restriction, prolongation, canonical means and norms).
include(joinpath(@__DIR__, "multigrid_cases.jl"))
function section_multigrid()
    MGm = Aeolus.Multigrid
    for N in ((64, 64, 32), (128, 128, 64))
        tag = join(N, 'x')
        for (name, (m, s)) in mg_solver_cases(N)
            p, _ = mg_solve_case(name, m, s)
            record!("multigrid/$tag/$name", p.data)
            record!("multigrid/$tag/$name/cycles", mg_cycles(s))
            s isa MGm.MultigridPoissonSolver && record!("multigrid/$tag/$name/history", mg_iteration_history(s))
        end
    end
end

# --------------------------------------------------------------------------
# KA-d (#299): the finite-volume kernels on launch!.
function section_fv()
    FVK = Aeolus.Operators.FiniteVolume
    SG = Aeolus.StretchedGrids
    CC, FX, FY, FZ = Aeolus.Grids.CellCenter(), Aeolus.Grids.FaceX(), Aeolus.Grids.FaceY(), Aeolus.Grids.FaceZ()
    rg = RegularGrid(; size = (64, 64, 32), extent = (640.0, 640.0, 320.0), halo = (3, 3, 3))
    sg = SG.StretchedGrid(; size = (64, 64, 32), extent = (640.0, 640.0, 320.0), halo = (3, 3, 3),
                          stretching = (:geometric, 1.05))
    rng = Xoshiro(299)
    for (gn, g) in (("regular", rg), ("stretched", sg))
        rf(loc; s = 1.0, o = 0.0) = (f = Field(loc, g); f.data .= o .+ s .* randn(rng, size(f.data)); f)
        u, v, w = rf(FX), rf(FY), rf(FZ)
        θ = rf(CC; s = 2.0, o = 300.0); qv = rf(CC; s = 1e-3, o = 8e-3); K = rf(CC; s = 0.1, o = 1.0)
        c = rf(CC)
        rec(name, f) = record!("fv/$gn/$name", f isa Field ? f.data : f)
        for (nm, loc, inp, k!) in (("∂x cc", CC, c, FVK._∂x!), ("∂z cc", CC, c, FVK._∂z!),
                                   ("∂x c→f", FX, c, FVK._∂x!), ("∂z c→f", FZ, c, FVK._∂z!),
                                   ("∂z f→c", CC, w, FVK._∂z!), ("interp c→fz", FZ, c, FVK._interp_to_face!),
                                   ("interp fz→c", CC, w, FVK._interp_to_center!))
            o = rf(loc); k!(o, inp, g); rec(nm, o)
        end
        for (ln, loc) in (("cc", CC), ("w", FZ))
            f = rf(loc); o = rf(loc); FVK._laplacian!(o, f, g); rec("laplacian $ln", o)
        end
        o = rf(CC); FVK._divergence!(o, u, v, w, g); rec("divergence", o)
        o = rf(CC); FVK._add_surface_flux!(o, 0.1, -0.05, g); rec("surface flux", o)
        a, b, d = rf(FX), rf(FY), rf(FZ); FVK._momentum_advect!(a, b, d, u, v, w, g)
        rec("momentum.u", a); rec("momentum.v", b); rec("momentum.w", d)
        o = rf(CC); FVK._advect_centered!(o, u, v, w, θ, g); rec("scalar advection", o)
        o = rf(FZ); FVK._add_buoyancy_moist!(o, θ, qv, 300.0, 9.81, g); rec("moist buoyancy", o)
        a, b = rf(FX), rf(FY); FVK._add_coriolis!(a, b, u, v, 1e-4, g); rec("coriolis.u", a); rec("coriolis.v", b)
        o = rf(CC); FVK._compute_smagorinsky_νt!(o, u, v, w, 0.17, g); rec("smagorinsky", o)
        o = rf(FX); FVK._add_eddy_viscosity_u!(o, u, K, g); rec("eddy viscosity u", o)
        o = rf(FZ); FVK._add_eddy_viscosity_w!(o, w, K, g); rec("eddy viscosity w", o)
        o = rf(CC); FVK._add_eddy_diffusion_scalar!(o, θ, K, K, 1.0, g); rec("eddy diffusion", o)
        # the damping kernel: pointwise, and with levels (one plane per work item)
        zs = FVK.damping_coordinates(g, CC); p = FVK.sponge_profile(0.01, 100.0, 320.0)
        o = rf(CC); FVK._add_damping!(o, θ, p, zs, 3, 300.0, g); rec("sponge", o)
        o = rf(CC); lv = zeros(g.Nz)
        FVK._add_damping!(o, θ, p, zs, 3, collect(range(299.0, 301.0; length = g.Nz)), g; levels = lv)
        rec("damping levels.tendency", o); rec("damping levels.sums", lv)
        o = rf(CC); SG._set_physical!(o, (x, y, z) -> sin(x / 50) * cos(y / 70) + z / 100, g)
        rec("set_physical!", o)
        if g === rg
            nh = abs.(randn(rng, size(θ.data))); nv = abs.(randn(rng, size(θ.data)))
            a, b, d = rf(FX), rf(FY), rf(FZ)
            FVK._add_sgs_stress_divergence!(a, b, d, u, v, w, nh, nv, 1, 1, g)
            rec("sgs stress div.u", a); rec("sgs stress div.v", b); rec("sgs stress div.w", d)
            τ = ntuple(_ -> zeros(size(θ.data)), 6); FVK._sgs_stress_tensor!(τ..., u, v, w, nh, nv, g)
            rec("sgs stress tensor", vcat(map(vec, τ)...))
            Ks = ntuple(_ -> 0.1 .* randn(rng, size(θ.data)), 6); F = ntuple(_ -> zeros(size(θ.data)), 3)
            o = rf(CC); FVK._add_tensor_diffusion_scalar!(o, θ, Ks..., F..., (false, false, true), true, true, g)
            rec("tensor diffusion", o)
            mask = rand(rng, size(θ.data)...) .< 0.1
            o = rf(CC); FVK._add_eddy_diffusion_scalar_masked!(o, θ, K, K, g, mask, nothing, nothing)
            rec("masked eddy diffusion", o)
            FVK._isentropic_tensor!(Ks..., θ, 2.0, 0.3, 0.3, 1e-3, g); rec("isentropic tensor", vcat(map(vec, Ks)...))
            o = rf(CC); FVK._advect_weno5!(o, u, v, w, θ, g); rec("weno5", o)
        end
    end
    # the damping kernel with levels at 128×128×32: large enough that a 3-D
    # launch would split a plane between tasks at -t 24 (rows in workgroups
    # of four), which races on levels[k]; the plane-per-work-item launch does not
    bg = RegularGrid(; size = (128, 128, 32), extent = (1280.0, 1280.0, 320.0), halo = (3, 3, 3))
    θb = Field(CC, bg); θb.data .= 300 .+ randn(rng, size(θb.data))
    tb = Field(CC, bg); tb.data .= randn(rng, size(tb.data)); lvb = zeros(bg.Nz)
    FVK._add_damping!(tb, θb, FVK.sponge_profile(0.01, 100.0, 320.0),
                      FVK.damping_coordinates(bg, CC), 3, 300.0, bg; levels = lvb)
    record!("fv/damping levels 128².tendency", tb.data); record!("fv/damping levels 128².sums", lvb)
    # a TerrainFollowingGrid over a hill: its own kernels on launch!
    tg = SG.terrain_following_grid(SG.StretchedGrid(; size = (64, 32, 32), extent = (2000.0, 400.0, 1500.0),
                                                    halo = (2, 2, 2), stretching = (:tanh, 1.4)),
                                   (x, y) -> 150.0 / (1 + ((x - 1000.0) / 300.0)^2))
    rt(loc) = (f = Field(loc, tg); f.data .= randn(rng, size(f.data)); f)
    u, v, w, c = rt(FX), rt(FY), rt(FZ), rt(CC)
    o = rt(CC); FVK._divergence!(o, u, v, w, tg); record!("fv/terrain/divergence", o.data)
    o = rt(CC); FVK._laplacian!(o, c, tg); record!("fv/terrain/laplacian", o.data)
    o = rt(FX); FVK._∂x!(o, c, tg); record!("fv/terrain/∂x c→f", o.data)
    # 3 steps of a Smagorinsky LES on the stretched grid
    sim = BoussinesqSimulation(sg; ν = 1e-2, θ₀ = 300.0, C_s = 0.17,
                               bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
    for f in (sim.u, sim.v, sim.w)
        interior_view(f) .= 0.3 .* randn(rng, size(interior_view(f)))
    end
    set!(sim.θ, (x, y, z) -> 300 + 0.01z)
    interior_view(sim.θ) .+= 0.05 .* randn(rng, size(interior_view(sim.θ)))
    for _ in 1:3
        step!(sim, 0.05)
    end
    record_sim!("fv/stretched LES", sim)
end

# KA-k (#307): the dynamics hooks and WallModels operators. Every launch on
# the KA path (`threshold = 0`; the sparse face lists are below the default
# threshold on these grids), or serially in `serial` mode.
function section_hooks()
    hb = MODE == "serial" ? BACKEND : CPUBackend(threshold = 0)
    LSF = Aeolus.LargeScaleForcing
    rng = Xoshiro(307)
    g = RegularGrid(; size = (64, 64, 32), extent = (640.0, 640.0, 320.0), halo = (3, 3, 3))
    sim = BoussinesqSimulation(g; ν = 0.05, C_s = 0.17, f_coriolis = 1e-4,
                               bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
    sm = Aeolus.IBM.SolidMask(g)
    # blocks of random height with one-cell streets between some of them
    hts = 30 .+ 150 .* rand(rng, 8, 8)
    Aeolus.IBM.mark_solid!(sm, (x, y, z) -> (bx = Int(fld(x, 80)) + 1; by = Int(fld(y, 80)) + 1;
                                             mod(x, 80) < 60 + 10 * (bx % 2) && mod(y, 80) < 50 + 20 * (by % 2) &&
                                             z < hts[bx, by]))
    for f in (sim.u, sim.v, sim.w, sim.tu, sim.tv, sim.tw)
        interior_view(f) .= randn(rng, size(interior_view(f)))
    end
    sim.νt.data .= 0.1 .* abs.(randn(rng, size(sim.νt.data)))
    with_backend(hb) do
        Aeolus.IBM.implicit_ibm_correction!(sim, sm; τ = 2.0, dt = 0.5)
        Aeolus.IBM.add_ibm_penalty!(sim, sm; τ = 0.7)
        Aeolus.IBM.implicit_ibm_correction!(sim, sm)
        record!("hooks/ibm.u", sim.u.data); record!("hooks/ibm.tu", sim.tu.data); record!("hooks/ibm.w", sim.w.data)
        law = (i, j, k) -> isodd(i ÷ 8) ? RoughLogLaw(0.05) : RoughLogLaw(0.5)
        iw = immersed_walls(sim, sm, law)
        record!("hooks/immersed.npairs", Float64(length(iw.faces)))
        remove_stencil_wall_flux!(sim, iw; ν = 0.05, sgs = true)
        apply_immersed_wall_stress!(sim, iw, 0.5)
        for (n, f) in (("u", sim.u), ("v", sim.v), ("w", sim.w), ("tu", sim.tu), ("tv", sim.tv), ("tw", sim.tw))
            record!("hooks/immersed.$n", f.data)
        end
        for k in (:F, :τA, :area, :τ_min, :F_stencil)
            record!("hooks/immersed.stats.$k", Float64[iw.stats[k]...])
        end
        prof = [(10 < i < 40 && 45 < j < 60) ? Aeolus.PlantCanopy.UniformLAD(3.0, 60.0) : nothing
                for i in 1:64, j in 1:64]
        drag = Aeolus.PlantCanopy.ResolvedCanopyDrag(g, Aeolus.PlantCanopy.canopy_lad_field(g, prof); C_d = 0.2)
        F = Aeolus.PlantCanopy.apply_canopy_drag!(sim, drag, 2.0)
        record!("hooks/canopy.u", sim.u.data); record!("hooks/canopy.w", sim.w.data)
        record!("hooks/canopy.F", collect(Float64, Tuple(F)))
        # the Berlin moist buoyancy
        θ = 300 .+ randn(rng, 64, 64, 32); qv = 8e-3 .+ 1e-3 .* randn(rng, 64, 64, 32)
        qc = [abs.(1e-4 .* randn(rng, 64, 64, 32))]; fluid = rand(rng, 64, 64, 32) .< 0.8
        Δ = zeros(64, 64, 32)
        Aeolus.Operators.moist_buoyancy_perturbation!(Δ, θ, qv, qc, fluid)
        Aeolus.Operators.add_moist_buoyancy!(sim.tw, Δ, 300.0, 9.81)
        record!("hooks/moist.Δ", Δ); record!("hooks/moist.tw", sim.tw.data)
    end
    # the large-scale forcings on a regular and a stretched grid
    for (gn, lg) in (("regular", RegularGrid(; size = (64, 64, 32), extent = (6400.0, 6400.0, 3200.0), halo = (3, 3, 3))),
                     ("stretched", Aeolus.StretchedGrids.StretchedGrid(; size = (64, 64, 32), extent = (6400.0, 6400.0, 3200.0),
                                                                       halo = (3, 3, 3), stretching = (:geometric, 1.05))))
        ls = BoussinesqSimulation(lg; ν = 0.0, f_coriolis = 1e-4,
                                  bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
        for f in (ls.u, ls.v, ls.θ, ls.tu, ls.tv, ls.tθ)
            f.data .= randn(rng, size(f.data))
        end
        ls.θ.data .+= 300
        tab = LSF.ForcingTable([0.0, 1000.0, 3200.0], [299.0, 301.0, 307.0])
        Fs = (LSF.GeostrophicPressureGradient((z, t) -> 5 + 0.002z, -1.0),
              LSF.ProfileNudging((θ = tab, u = 4.0), 900.0; weight = LSF.NudgingWeight(800.0; depth = 400.0)),
              LSF.Subsidence((z, t) -> -3e-6 * z; fields = (:θ, :u)),
              LSF.Subsidence((z, t) -> -3e-6 * z; fields = (:θ, :v), form = :local),
              LSF.AdvectiveTendencies(; θ = (z, t) -> 1e-5 * sin(z / 300), u = tab),
              LSF.ReferenceDamping(:θ, Aeolus.Operators.sponge_profile(0.01, 1600.0, 3200.0), :mean))
        with_backend(hb) do
            for (n, F) in enumerate(Fs)
                apply_forcing!(ls, F, 60.0, 2.0)
                record!("hooks/lsf.$gn.$n.integrals", Float64[values(LSF.tendency_integrals(F))...])
            end
        end
        record!("hooks/lsf.$gn.tu", ls.tu.data); record!("hooks/lsf.$gn.tv", ls.tv.data)
        record!("hooks/lsf.$gn.tθ", ls.tθ.data)
        record!("hooks/lsf.$gn.damping", Fs[end].levels)
    end
end

# --------------------------------------------------------------------------
function section_sgs()
    g = RegularGrid(; size = (64, 64, 32), extent = (640.0, 640.0, 320.0), halo = (3, 3, 3))
    bcs = FieldBoundaryConditions(:periodic, :periodic, :wall)
    rng = Xoshiro(304)
    function mksim()
        sim = BoussinesqSimulation(g; ν = 1e-3, θ₀ = 300.0, f_coriolis = 1e-4, bcs = bcs)
        r = Xoshiro(3040)
        for f in (sim.u, sim.v, sim.w)
            interior_view(f) .= 0.5 .* randn(r, size(interior_view(f)))
        end
        set!(sim.θ, (x, y, z) -> 300 + 0.003z)
        interior_view(sim.θ) .+= 0.05 .* randn(r, size(interior_view(sim.θ)))
        for f in (sim.u, sim.v, sim.w)
            fill_halos!(f, sim.bcs)
        end
        fill_halos!(sim.θ, sim.θ_bcs)
        return sim
    end
    mask = falses(size(Field(Aeolus.Grids.CellCenter(), g).data))
    mask[g.Hx+20:g.Hx+30, g.Hy+20:g.Hy+40, g.Hz+1:g.Hz+12] .= true
    mask[g.Hx+45:g.Hx+55, g.Hy+5:g.Hy+15, g.Hz+1:g.Hz+20] .= true
    recf(name, f) = record!("sgs/$name", f isa Field ? f.data : f)
    sim = mksim()
    for (n, c) in ("static" => StaticSmagorinsky(0.17), "dynamic plane" => DynamicSmagorinsky(g),
                   "dynamic local" => DynamicSmagorinsky(g; averaging = :local),
                   "dynamic masked" => DynamicSmagorinsky(g; mask = mask),
                   "AMD" => AMDClosure(), "Vreman" => VremanClosure(), "sigma" => SigmaModel(),
                   "stability smag" => StabilityCorrected(StaticSmagorinsky(0.17), g),
                   "stability dynamic" => StabilityCorrected(DynamicSmagorinsky(g), g))
        f = compute_sgs!(c, sim)
        recf("$n.νh", f.νh); recf("$n.νv", f.νv); recf("$n.κh", f.κh); recf("$n.κv", f.κv)
        c isa AMDClosure && recf("$n.κ(θ as tracer)", sgs_tracer_diffusivity!(Field(Aeolus.Grids.CellCenter(), g),
                                                                             c, f, sim, sim.θ))
    end
    for (n, c) in ("lagrangian" => LagrangianDynamicSmagorinsky(g),
                   "LASD" => LagrangianDynamicSmagorinsky(g; scale_dependent = true),
                   "lagrangian masked" => LagrangianDynamicSmagorinsky(g; mask = mask))
        for _ in 1:3
            advance_sgs!(c, sim, 0.05)
        end
        f = compute_sgs!(c, sim)
        recf("$n.νt", f.νh); recf("$n.I_LM", c.I_LM); recf("$n.I_MM", c.I_MM)
        c.scale_dependent && (recf("$n.I_QN", c.I_QN); recf("$n.I_NN", c.I_NN); recf("$n.beta", c.beta))
    end
    for (n, c) in ("TKE" => TKEClosure(g), "TKE masked exact" => TKEClosure(g; mask = mask, dissipation = :exact))
        initialize_tke!(c, sim; e_min = 1e-4); recf("$n.init", c.e)
        for _ in 1:3
            advance_sgs!(c, sim, 0.05)
        end
        f = compute_sgs!(c, sim)
        recf("$n.e", c.e); recf("$n.νt", f.νh); recf("$n.κt", f.κh); recf("$n.stats", c.stats)
    end
    rsm = ReynoldsStressClosure(g); initialize_reynolds_stress!(rsm, sim; e_min = 1e-4)
    recf("RSM.init", vcat(map(f -> vec(f.data), rsm.τ)...))
    for _ in 1:3
        advance_sgs!(rsm, sim, 0.05)
    end
    f = compute_sgs!(rsm, sim)
    recf("RSM.τ", vcat(map(f -> vec(f.data), rsm.τ)...)); recf("RSM.νt", f.νh); recf("RSM.stats", rsm.stats)
    D = Field(Aeolus.Grids.CellCenter(), g); dual_strain!(D, sim.u, sim.v, sim.w, sim.bcs); recf("dual strain", D)
    o = Field(Aeolus.Grids.CellCenter(), g); box_filter!(o, sim.θ, 3; mask = mask); recf("box filter masked", o)
    # 3 steps of an LES with the dynamic closure, and with the TKE closure
    for (n, mk) in ("dynamic" => () -> DynamicSmagorinsky(g), "TKE" => () -> TKEClosure(g; e0 = 1e-3))
        s2 = mksim(); c = mk()
        for _ in 1:3
            step!(s2, 0.05; sgs = c)
        end
        record_sim!("sgs/LES $n", s2)
    end
end

# --------------------------------------------------------------------------
function section_radiation()
    R = Aeolus.Radiosity
    hm = zeros(16, 14)
    for (i, j, h) in ((3, 3, 15.0), (4, 3, 15.0), (7, 5, 30.0), (8, 5, 24.0), (8, 6, 24.0),
                      (12, 9, 9.0), (12, 10, 9.0), (13, 10, 12.0), (5, 11, 40.0), (14, 3, 6.0))
        hm[i, j] = h
    end
    scene = build_scene(hm, 3.0, 3.0; wall_subdivisions = 2)
    for i in 1:nfacets(scene)
        set_temperature!(scene, i, 285.0 + 15 * sin(0.37 * i))
    end
    vf = view_factors(scene; backend = BACKEND)
    record!("radiation/radiosity/nfacets", nfacets(scene))
    record!("radiation/radiosity/F", vf.F)
    record!("radiation/radiosity/F_sky", vf.F_sky)
    nb = Aeolus.Backends.BLAS.get_num_threads()
    Aeolus.Backends.BLAS.set_num_threads(1)
    csw, clw = RadiosityFactorCache(), RadiosityFactorCache()
    sun = solar_unit_vector((0.3, -0.4, 0.8))
    sw = solve_shortwave(scene, vf, sun; dni = 800.0, dhi = 90.0, cache = csw)
    lw = solve_longwave(scene, vf; L_sky = 320.0, cache = clw)
    lw2 = solve_longwave(scene, vf; L_sky = 330.0, cache = clw)
    Aeolus.Backends.BLAS.set_num_threads(nb)
    record!("radiation/radiosity/sw.B", sw.radiosity)
    record!("radiation/radiosity/lw.B", lw.radiosity)
    record!("radiation/radiosity/lw2.B (cached LU)", lw2.radiosity)
    agg = aggregate_to_grid(scene, vf; sw = sw, lw = lw, backend = BACKEND)
    for f in fieldnames(typeof(agg))
        record!("radiation/aggregate/$f", getfield(agg, f))
    end
    # gas + cloud optics and the multi-column two-stream solves
    nz = 30
    cols = [us_standard_column(; nz = nz, T_sfc = 280.0 + 3c) for c in 1:6]
    cols[2].ql[3:6] .= 3e-4; cols[5].qi[12:14] .= 4e-5
    for (tag, spec) in (("sw", shortwave_bands()), ("lw", longwave_bands()))
        o = allocate_column_optics(Float64, nz, 6, spec)
        compute_optics!(o, correlated_k_optics(spec), cols; backend = BACKEND)
        oc = allocate_column_optics(Float64, nz, 6, spec)
        compute_optics!(oc, mie_cloud_optics(spec), cols; backend = BACKEND)
        combine_optics!(o, oc)
        record!("radiation/optics/$tag.τ", o.τ); record!("radiation/optics/$tag.ssa", o.ssa)
        f = allocate_fluxes(Float64, (6,), nz + 1)
        if tag == "sw"
            solve_sw!(f, two_stream_solver(spec), o; μ₀ = [0.2 + 0.1c for c in 1:6],
                      sfc_albedo_direct = 0.15, backend = BACKEND)
        else
            solve_lw!(f, two_stream_solver(spec), o, cols; sfc_emissivity = 0.96, backend = BACKEND)
        end
        record!("radiation/twostream/$tag.up", f.flux_up)
        record!("radiation/twostream/$tag.dn", f.flux_dn)
        record!("radiation/twostream/$tag.dir", f.flux_dn_dir)
    end
    # cloud shadow
    mesh = RegularGrid(; size = (32, 24, 16), extent = (640.0, 480.0, 320.0), halo = (2, 2, 2))
    rng = Xoshiro(305)
    ql = max.(1e-4 .* randn(rng, 32, 24, 16), 0.0); qi = zeros(32, 24, 16)
    τ = cloud_optical_depth_column(ql, qi, 1.1, 20.0; backend = BACKEND)
    tr = zeros(32, 24)
    cloud_shadow_transmittance!(tr, τ, mesh, (0.4, 0.2, 0.89); backend = BACKEND)
    record!("radiation/cloud_shadow/τ", τ)
    record!("radiation/cloud_shadow/trans", tr)
    # Monte Carlo
    dom = mc_domain(Float64; size = (12, 10, 10), spacing = (40.0, 40.0, 40.0))
    kext = zeros(12, 10, 10); ssa = zeros(12, 10, 10); gg = zeros(12, 10, 10)
    kext[3:8, 2:7, 3:6] .= 0.04; ssa[3:8, 2:7, 3:6] .= 0.99; gg[3:8, 2:7, 3:6] .= 0.85
    med = mc_medium(dom; kext = kext, ssa = ssa, g = gg, sfc_albedo = 0.15)
    r = mc_solve(med, monte_carlo_solver(; nphoton = 16_000, nbatch = 16); seed = 305,
                 sun_dir = mc_sun_direction(0.6; azimuth = 100.0), solar_flux = 900.0,
                 backend = BACKEND)
    record!("radiation/montecarlo/absorbed", r.absorbed)
    record!("radiation/montecarlo/absorbed_err", r.absorbed_err)
    record!("radiation/montecarlo/sfc", r.sfc_direct .+ r.sfc_diffuse)
    record!("radiation/montecarlo/fluxes", vcat(vec(r.flux_up), vec(r.flux_dn)))
    record!("radiation/montecarlo/scalars", [r.albedo, r.albedo_err, r.transmittance])
    # superdroplets
    sd = init_exponential_superdroplets(Xoshiro(1), 2000, 1e8, 15e-6, 1.6e-4; backend = BACKEND)
    for i in eachindex(sd.cell)
        sd.cell[i] = 1 + (i * 7) % 16
    end
    rsd = Xoshiro(2)
    counts = Int[]
    for _ in 1:4
        c, e = collision_coalescence!(rsd, sd, Aeolus.Superdroplet.GolovinKernel(), 1.0,
                                      fill(1e-5, 16); backend = BACKEND)
        push!(counts, c, e)
    end
    record!("radiation/superdroplet/ξ", sd.ξ)
    record!("radiation/superdroplet/radius", sd.radius)
    record!("radiation/superdroplet/counts", counts)
    fl = supersaturation_fluctuations(2000; σ_S = 1e-3, τ = 2.0, seed = 9, backend = BACKEND)
    advance_fluctuations!(fl, 0.5; backend = BACKEND)
    record!("radiation/superdroplet/fluctuations", fl.ξ)
end

# --------------------------------------------------------------------------
function section_advection()
    AS = Aeolus.AdvectionSchemes
    SG = Aeolus.StretchedGrids
    CC, FX, FY, FZ = Aeolus.Grids.CellCenter(), Aeolus.Grids.FaceX(), Aeolus.Grids.FaceY(), Aeolus.Grids.FaceZ()
    g = RegularGrid(; size = (64, 64, 32), extent = (640.0, 640.0, 320.0), halo = (5, 5, 5))
    rng = Xoshiro(302)
    rf(loc, m = g; s = 1.0, o = 0.0) = (f = Field(loc, m); f.data .= o .+ s .* randn(rng, size(f.data)); f)
    pos(m = g) = (f = Field(CC, m); f.data .= max.(randn(rng, size(f.data)), 0.0) .^ 3; f)
    u, v, w = rf(FX), rf(FY), rf(FZ; s = 0.5)
    θ = rf(CC; s = 2.0, o = 300.0); q = pos()
    walls = FieldBoundaryConditions(:periodic, :periodic, :wall)
    ws = AS.FluxLimiterWorkspace(g)
    for (rn, r) in (("weno_z5", AS.WENOZ(5, 2)), ("weno_z7", AS.WENOZ(7, 2)), ("teno5", AS.TENO(5)),
                    ("teno8", AS.TENO(8)), ("weno_js5", AS.WENOJS(5)), ("centered", AS.Centered2()),
                    ("upwind", AS._UpwindFirst()))
        o = rf(CC); AS._advect_reconstructed!(o, u, v, w, θ, r, g); record!("advection/$rn/sweep", o.data)
        AS._reconstructed_fluxes!(ws.Fx, ws.Fy, ws.Fz, u, v, w, θ, r, g)
        record!("advection/$rn/face fluxes", vcat(vec(ws.Fx.data), vec(ws.Fy.data), vec(ws.Fz.data)))
        for (bn, bcs) in (("wrap", nothing), ("walls", walls))
            o = rf(CC); advect_positive!(o, u, v, w, q, 400.0; scheme = r, workspace = ws, bcs = bcs)
            record!("advection/$rn/advect_positive! $bn", o.data)
            record!("advection/$rn/advect_positive! $bn.ratio", ws.ratio.data)
        end
    end
    # the bare limiter with absurd fluxes (no low-order fallback)
    Fx, Fy, Fz = rf(FX; s = 50.0), rf(FY; s = 50.0), rf(FZ; s = 50.0)
    apply_positivity_limiter!(Fx, Fy, Fz, q, 10.0, g; ratio = ws.ratio)
    record!("advection/bare limiter", vcat(vec(Fx.data), vec(Fy.data), vec(Fz.data), vec(ws.ratio.data)))
    for (rn, r) in (("weno_z5", AS.WENOZ(5, 2)), ("weno_z7", AS.WENOZ(7, 2)))
        a, b, c = rf(FX), rf(FY), rf(FZ); AS._momentum_advect_recon!(a, b, c, u, v, w, r, g)
        record!("advection/momentum $rn", vcat(vec(a.data), vec(b.data), vec(c.data)))
    end
    a, b, c = rf(FX), rf(FY), rf(FZ); AS._momentum_advect_skew!(a, b, c, u, v, w, g)
    record!("advection/momentum skew", vcat(vec(a.data), vec(b.data), vec(c.data)))
    o = rf(CC); AS._advect_variance_conserving!(o, u, v, w, θ, g); record!("advection/variance conserving", o.data)
    # the positive-definite path on a stretched grid (centred)
    sg = SG.StretchedGrid(; size = (64, 64, 32), extent = (640.0, 640.0, 320.0), halo = (3, 3, 3),
                          stretching = (:geometric, 1.05))
    sws = AS.FluxLimiterWorkspace(sg)
    o = rf(CC, sg)
    advect_positive!(o, rf(FX, sg), rf(FY, sg), rf(FZ, sg; s = 0.5), pos(sg), 400.0;
                     scheme = :centered, workspace = sws, bcs = walls)
    record!("advection/stretched advect_positive!", o.data)
    # boundary_scalar_flux on an open box large enough that its 2-D face
    # launches run on several tasks (a 256×128 side plane: 2 tasks; the
    # 256×256 bottom/top plane: 4), with a mass-flux correction δ ≠ 0,
    # diffusion, weights and levels
    bg = RegularGrid(; size = (256, 256, 128), extent = (2560.0, 2560.0, 1280.0), halo = (5, 5, 5))
    data = FunctionBoundaryData(u = 2.0, v = -1.0, w = 0.1, θ = 300.0)
    bcs = open_boundary_conditions(bg, data; halo = 5)
    bcs.west.state.δ = 0.003
    bu, bv, bw = rf(FX, bg; o = 2.0), rf(FY, bg; o = -1.0), rf(FZ, bg; s = 0.1)
    bθ = rf(CC, bg; s = 0.5, o = 300.0); bκ = rf(CC, bg; s = 0.1, o = 1.0)
    fake = (u = bu, v = bv, w = bw, bcs = bcs, θ_bcs = bcs)
    W = (1.0 .+ 0.1 .* rand(rng, size(bθ.data, 3)), 1.0 .+ 0.1 .* rand(rng, size(bθ.data, 3)))
    for sch in (:centered, :upwind, :weno_z5, :teno8)
        lv = zeros(bg.Nz); cl = zeros(bg.Nz)
        F = boundary_scalar_flux(fake, bθ; scheme = sch, fill = false, levels = lv, correction_levels = cl,
                                 weights = W, diffusivity = (ν = 0.1, κh = bκ, κv = bκ))
        record!("advection/boundary_scalar_flux $sch.totals", collect(values(F)))
        record!("advection/boundary_scalar_flux $sch.levels", vcat(lv, cl))
    end
end

# --------------------------------------------------------------------------
# KA-j (#306): every step core (test/support/step_cases.jl): every field of
# the simulation after one to four steps, the dt sequences, the CFL
# diagnostics and a forced retry.
include(joinpath(@__DIR__, "step_cases.jl"))
section_steps() = (run_cases(); nothing)

# --------------------------------------------------------------------------
# BLAS on one thread at every count. `init_threads!` pins it only when Julia
# runs threaded, so at -t 1 OpenBLAS keeps its own default; LAPACK's bits
# depend on the BLAS thread count, and the Berlin radiosity solves through
# it (bus sw/lw, the surface ledger of the smoke and flagship sections).
init_threads!(BACKEND; blas_threads = 1)
t0 = time()
for s in SECTIONS
    ts = time()
    s == "primitives" ? section_primitives() :
    s == "dns"        ? section_dns() :
    s == "reductions" ? section_reductions() :
    s == "smoke"      ? section_smoke() :
    s == "flagship"   ? section_flagship() :
    s == "nested"     ? section_nested() :
    s == "fft"        ? section_fft() :
    s == "fv"         ? section_fv() :
    s == "sgs"        ? section_sgs() :
    s == "hooks"      ? section_hooks() :
    s == "pcg"        ? section_pcg() :
    s == "multigrid"  ? section_multigrid() :
    s == "steps"      ? section_steps() :
    s == "radiation"  ? section_radiation() :
    s == "columns"    ? section_columns() :
    s == "advection"  ? section_advection() :
                        error("unknown section $s")
    @printf("thread-invariance worker: -t %d %s: %s in %.1f s\n", Threads.nthreads(), MODE, s, time() - ts)
end
open(OUT, "w") do io
    for (k, v) in RESULTS
        println(io, k, '\t', v)
    end
end
@printf("thread-invariance worker: -t %d %s: %d hashes in %.1f s\n",
        Threads.nthreads(), MODE, length(RESULTS), time() - t0)
