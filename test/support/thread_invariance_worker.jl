# One process of the thread-invariance harness (KA-a, #294; see
# test/thread_invariance.jl and docs/src/design/ka_port.md).
#
#   julia -t N --project=. test/support/thread_invariance_worker.jl OUT SECTIONS [MODE]
#
# Runs SECTIONS (comma list of primitives, dns, reductions, smoke, flagship, nested, fft,
# columns) and writes
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
#   reductions  the canonical reductions of KA-b (#296) on a seeded 64³ LES
#               state: the energy / variance budgets, kinetic energy, the
#               horizontal-mean profiles, the SGS dissipation, the CFL maxima
#               and finiteness checks, plane_reduce! (tuple, fused write),
#               and the chemistry P/L/H and wet-deposition totals with the
#               species fields they update;
#   smoke       every run of the Berlin smoke matrix (test/support/
#               smoke_matrix.jl), 12 steps: every field of the simulation,
#               every tracer, every Float64 array of the surface bus, the
#               surface / water / air-heat ledgers, the dt sequence and the
#               pressure-solver iteration counts per step;
#   flagship    one step of the default CaseConfig() (the 64×64×32
#               Alexanderplatz run), hashed the same way;
#   nested      nested runs (KA-f, #297): one-way, two-way with a sponge,
#               two-way with Orlanski outflow faces (16³ parent, 16³
#               child at r = 2, 3 steps) and an anelastic two-way pair
#               (4 steps): every field of parent and child.
#   columns     (KA-m, #301) the column-physics kernels on seeded inputs, every
#               launch threaded (threshold 0): fog adjustment, Kessler (both
#               sedimentation schemes), dew exchange, dry deposition, facet
#               scenes (implicit, direct, lagged), the dynamic roughness, and
#               the partials of a foreach_column! launch;
#   fft         (KA-h, #298) one solve of every spectral pressure solver —
#               FFT, CosineFFT for all 8 axis-BC combinations, Stretched for
#               the 4 lateral pairs — at 64×64×32 and 128×128×64, with FFTW
#               on the process's threads (`init_threads!`).

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
end

# --------------------------------------------------------------------------
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
    record!("reductions/max_speed", TI._max_speed(sim))
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
isdefined(Main, :BerlinCase) ||
    include(joinpath(@__DIR__, "..", "..", "examples", "urban", "BerlinCase.jl"))
const BC = Main.BerlinCase
include(joinpath(@__DIR__, "smoke_matrix.jl"))

function run_hashed!(label, cfg; seed = true, nsteps = cfg.nsteps)
    case = BC.build_case(cfg; io = devnull)
    seed && seed_fog!(case, Xoshiro(127))
    ts = BC.timestepper(case)
    variable = ts !== nothing && BC.variable_dt(typeof(ts))
    dts = Float64[]; iters = Int[]
    t_end = case.sim.t + nsteps * cfg.dt
    for n in 1:nsteps
        t0 = case.sim.t
        ok = variable ? BC.step_case!(case, BC.last_step_dt(BC.step_dt(case), t_end - case.sim.t)) :
                        BC.step_case!(case, cfg.dt)
        push!(dts, case.sim.t - t0)
        append!(iters, solver_iters(case.sim))
        ok || break
        variable && t_end - case.sim.t <= 1e-9 * cfg.dt && break
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
    record_repr!("$label/ledger.surface", BC.surface_budget(case.ledger))
    record_repr!("$label/ledger.water", BC.water_budget(case.ledger))
    record_repr!("$label/ledger.air_heat", BC.heat_budget(case.ledger))
    record!("$label/dt", dts)
    record!("$label/solver_iterations", iters)
    return case
end

function section_smoke()
    for (label, cfg) in smoke_matrix()
        cfg.backend = BACKEND
        run_hashed!("smoke[$label]", cfg)
    end
end

function section_flagship()
    cfg = BC.CaseConfig(; verbose = false, output_path = nothing, backend = BACKEND)
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
        for (name, (m, solver)) in solvers
            src = Field(m); src.data .= randn(Xoshiro(298), size(src.data))
            p = Field(m)
            PS.solve!(p, src, solver)
            record!("fft/$tag/$name", p.data)
        end
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
init_threads!(BACKEND)
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
    s == "columns"    ? section_columns() :
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
