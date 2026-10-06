# One process of the thread-invariance harness (KA-a, #294; see
# test/thread_invariance.jl and docs/src/design/ka_port.md).
#
#   julia -t N --project=. test/support/thread_invariance_worker.jl OUT SECTIONS [MODE]
#
# Runs SECTIONS (comma list of primitives, dns, reductions, smoke, flagship,
# nested, fft, radiation) and writes
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
#   fft         (KA-h, #298) one solve of every spectral pressure solver —
#               FFT, CosineFFT for all 8 axis-BC combinations, Stretched for
#               the 4 lateral pairs — at 64×64×32 and 128×128×64, with FFTW
#               on the process's threads (`init_threads!`).
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
    case = BC.build_case(cfg)
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
    cfg = BC.CaseConfig(; output_path = nothing, backend = BACKEND)
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
    s == "radiation"  ? section_radiation() :
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
