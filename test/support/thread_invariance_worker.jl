# One process of the thread-invariance harness (KA-a, #294; see
# test/thread_invariance.jl and docs/src/design/ka_port.md).
#
#   julia -t N --project=. test/support/thread_invariance_worker.jl OUT SECTIONS [MODE]
#
# Runs SECTIONS (comma list of primitives, dns, smoke, flagship, nested) and writes
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
init_threads!(BACKEND)
t0 = time()
for s in SECTIONS
    ts = time()
    s == "primitives" ? section_primitives() :
    s == "dns"        ? section_dns() :
    s == "smoke"      ? section_smoke() :
    s == "flagship"   ? section_flagship() :
    s == "nested"     ? section_nested() :
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
