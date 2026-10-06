# One process of the thread-invariance harness (KA-a, #294; see
# test/thread_invariance.jl and docs/src/design/ka_port.md).
#
#   julia -t N --project=. test/support/thread_invariance_worker.jl OUT SECTIONS [MODE]
#
# Runs SECTIONS (comma list of primitives, dns, reductions, smoke, flagship) and writes
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
#               Alexanderplatz run), hashed the same way.

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
init_threads!(BACKEND)
t0 = time()
for s in SECTIONS
    ts = time()
    s == "primitives" ? section_primitives() :
    s == "dns"        ? section_dns() :
    s == "reductions" ? section_reductions() :
    s == "smoke"      ? section_smoke() :
    s == "flagship"   ? section_flagship() :
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
