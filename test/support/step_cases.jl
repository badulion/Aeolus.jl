# KA-j (#306): one or a few steps of every dynamical core from seeded
# states — Boussinesq Euler / RK3 / adaptive (dt sequence, CFL diagnostics,
# a forced retry), IMEX ARK2 and Rosenbrock2, anelastic and
# pseudo-incompressible, a stretched grid, open boundaries. Section `steps`
# of the thread-invariance worker; uses only APIs that predate KA-j, so the
# same cases hash the pre-port code (the bit-for-bit check of the KA-j PR).
# `record!(key, x)` must be defined by the includer.
using Aeolus
using Aeolus.Fields: set!, interior_view, Field
using Aeolus.Grids: interior
using Aeolus.Simulations: BoussinesqSimulation, step_rk3!, max_divergence
using Aeolus.TimeIntegration: step_imex!, ARK2, Rosenbrock2, step_adaptive!, CFLConstraints,
                              cfl_diagnostics, adaptive_dt
using Aeolus.Anelastic: AnelasticSimulation, PseudoIncompressibleSimulation, isothermal_reference
using Aeolus.BoundaryConditions: FieldBoundaryConditions, open_boundary_conditions, FunctionBoundaryData
using Random: Xoshiro, randn

fieldsof(sim) = [(n, getfield(sim, n)) for n in fieldnames(typeof(sim)) if getfield(sim, n) isa Field]
function rec_sim!(pre, sim)
    for (n, f) in fieldsof(sim)
        record!("$pre/$n", f.data)
    end
    record!("$pre/t", sim.t)
end

function seed!(sim, rng; amp = 0.3, moist = false)
    for f in (sim.u, sim.v, sim.w)
        interior_view(f) .= amp .* randn(rng, size(interior_view(f)))
    end
    set!(sim.θ, (x, y, z) -> 300 + 0.003z)
    interior_view(sim.θ) .+= 0.1 .* randn(rng, size(interior_view(sim.θ)))
    moist && (set!(sim.qv, 8e-3); interior_view(sim.qv) .+= 1e-4 .* randn(rng, size(interior_view(sim.qv))))
    return sim
end

const OBDATA = FunctionBoundaryData(
    u = (x, y, z, t) -> 1.0 + 0.001x + 0.002y + 0.003z + 0.1t,
    v = (x, y, z, t) -> -0.5 + 0.002x - 0.001y + 0.001z,
    w = (x, y, z, t) -> 0.0,
    θ = (x, y, z, t) -> 300.0 + 0.003z + 0.0001x)

function run_cases(; N = (64, 64, 32))
    ext = (640.0, 640.0, 320.0)
    g = RegularGrid(; size = N, extent = ext, halo = (3, 3, 3))
    walls = FieldBoundaryConditions(:periodic, :periodic, :wall)
    mk(; kw...) = BoussinesqSimulation(g; ν = 1e-2, θ₀ = 300.0, C_s = 0.17, f_coriolis = 1e-4,
                                       θ_flux_bottom = 0.05, sponge_α_max = 0.01,
                                       sponge_z_start = 250.0, bcs = walls, kw...)
    # Boussinesq Euler, moist
    s = seed!(mk(; moisture = true), Xoshiro(1); moist = true)
    for _ in 1:2; step!(s, 0.05); end
    rec_sim!("euler", s)
    # RK3, moist
    s = seed!(mk(; moisture = true), Xoshiro(2); moist = true)
    for _ in 1:2; step_rk3!(s, 0.05); end
    rec_sim!("rk3", s)
    # adaptive: the dt sequence, CFL diagnostics, and a forced retry
    s = seed!(mk(), Xoshiro(3))
    dts = Float64[]; dt = nothing
    c = CFLConstraints(dt_max = 1.0)
    d = cfl_diagnostics(s, c)
    for k in keys(d)
        getfield(d, k) isa Symbol || record!("cfl/$k", Float64(getfield(d, k)))   # not the limiter labels
    end
    for _ in 1:4
        _, dt, nr = step_adaptive!(s, c; dt_prev = dt)
        push!(dts, dt, nr)
    end
    record!("adaptive/dts", dts)
    rec_sim!("adaptive", s)
    s = seed!(mk(), Xoshiro(4); amp = 3.0)
    _, dt, nr = step_adaptive!(s, CFLConstraints(dt_max = 50.0); dt = 40.0, max_retries = 8)
    record!("adaptive retry/dt,n", [dt, nr])
    rec_sim!("adaptive retry", s)
    # IMEX
    for (nm, sch) in (("ark2", ARK2()), ("ros2", Rosenbrock2()))
        s = seed!(mk(; moisture = true), Xoshiro(5); moist = true)
        for _ in 1:2; step_imex!(s, 0.2, sch); end
        rec_sim!("imex $nm", s)
    end
    # sound-proof cores
    gd = RegularGrid(; size = N, extent = (640.0, 640.0, 3000.0), halo = (3, 3, 3))
    for (nm, ctor) in (("anelastic", AnelasticSimulation), ("pseudo", PseudoIncompressibleSimulation))
        s = ctor(gd; ν = 1e-2, C_s = 0.17, f_coriolis = 1e-4)
        rng = Xoshiro(6)
        for f in (s.u, s.v, s.w)
            interior_view(f) .= 0.3 .* randn(rng, size(interior_view(f)))
        end
        interior_view(s.θ) .+= 0.1 .* randn(rng, size(interior_view(s.θ)))
        for _ in 1:2; step!(s, 0.2); end
        rec_sim!(nm, s)
        record!("$nm/adaptive_dt", adaptive_dt(s, CFLConstraints()))
    end
    # stretched grid
    sg = Aeolus.StretchedGrids.StretchedGrid(; size = N, extent = ext, halo = (3, 3, 3),
                                             stretching = (:geometric, 1.04))
    s = BoussinesqSimulation(sg; ν = 1e-2, θ₀ = 300.0, C_s = 0.17, bcs = walls)
    seed!(s, Xoshiro(7))
    for _ in 1:2; step!(s, 0.05); end
    rec_sim!("stretched euler", s)
    dt = nothing; dts = Float64[]
    for _ in 1:3
        _, dt, nr = step_adaptive!(s, CFLConstraints(dt_max = 1.0); dt_prev = dt)
        push!(dts, dt, nr)
    end
    record!("stretched adaptive/dts", dts)
    rec_sim!("stretched adaptive", s)
    s = BoussinesqSimulation(sg; ν = 1e-2, θ₀ = 300.0, C_s = 0.17, bcs = walls)
    seed!(s, Xoshiro(8))
    for _ in 1:2; step_rk3!(s, 0.05); end
    rec_sim!("stretched rk3", s)
    # open boundaries: every core
    go = RegularGrid(; size = (32, 24, 16), extent = (320.0, 240.0, 160.0), halo = (3, 3, 3))
    ob() = (b = open_boundary_conditions(go, OBDATA);
            s = BoussinesqSimulation(go; θ₀ = 300.0, C_s = 0.18, ν = 1e-2, bcs = b);
            set!(s.θ, 300.0); set!(s.u, 1.0); s)
    for (nm, adv!) in (("step!", s -> step!(s, 0.5)), ("rk3", s -> step_rk3!(s, 0.5)),
                       ("ark2", s -> step_imex!(s, 0.5, ARK2())),
                       ("ros2", s -> step_imex!(s, 0.5, Rosenbrock2())),
                       ("adaptive", s -> step_adaptive!(s, CFLConstraints(dt_max = 0.5))))
        s = ob()
        for _ in 1:2; adv!(s); end
        rec_sim!("open $nm", s)
    end
    s = AnelasticSimulation(go; bcs = open_boundary_conditions(go, OBDATA), ν = 1e-2)
    set!(s.u, 1.0)
    for _ in 1:2; step!(s, 0.5); end
    rec_sim!("open anelastic", s)
    return nothing
end
