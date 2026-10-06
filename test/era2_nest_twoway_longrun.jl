using Test
using Printf
using Random
using Aeolus
using Aeolus.Nesting
using Aeolus.Grids: interior
using Aeolus.Fields: interior_view

# =========================================================================
# N-nest-e (#263) — long-run stability of two-way nesting.
#
#   1. a convective boundary layer (a coarser version of Hellsten et al.
#      2021, GMD 14, 3185, §4.1: parent 3.2 × 3.2 × 1.6 km at 100 m, child
#      r = 3 on the ground, surface heat flux 0.1 K m/s, 2 h; the 4 h case
#      at 50 m is benchmarks/bench_twoway_cbl.jl): no NaN, bounded kinetic
#      energy, child max|∇·u| < 1e-10 every step, the composite θ budget
#      closes, the 2Δ interface noise does not grow; the child-mean
#      secondary-circulation updraft is measured for one-way and two-way;
#   2. a forced neutral flow (Kolmogorov forcing, Smagorinsky) for more
#      than 20 eddy turnovers: bounded kinetic energy;
#   3. N-open-c's outgoing Lamb–Oseen vortex harness (#260) in two-way
#      mode, 2000 steps: the child's ∫|u|² stays within 1 % of the energy
#      the parent supplies (a standalone parent run).
# =========================================================================

isdefined(@__MODULE__, :VortexHarness) || include("open_boundary_vortex_harness.jl")

# ---- 1. convective boundary layer: the case is shared with the 4 h
#         benchmark, benchmarks/bench_twoway_cbl.jl ---------------------------
include("twoway_cbl_case.jl")

# ---- 2. forced neutral flow ----------------------------------------------

function _twl_neutral(feedback; N = 16, r = 2, F0 = 0.15, T = 120.0, seed = 2)
    L = 2π
    pm = RegularGrid(Float64; size = (N, N, N), extent = (L, L, L), halo = (3, 3, 3))
    p = BoussinesqSimulation(pm; C_s = 0.17, bcs = FieldBoundaryConditions(:periodic))
    nest = nested_grid(pm, ((L / 4, 3L / 4), (L / 4, 3L / 4), (L / 4, 3L / 4)), r)
    cm = nest_child_mesh(nest)
    c = BoussinesqSimulation(cm; C_s = 0.17, bcs = nest_bcs(cm, :nest, :nest, :nest))
    zc(m, k) = m.z0 + (k - m.Hz - 0.5) * m.dz
    rng = Xoshiro(seed)
    for k in axes(p.u.data, 3)
        p.u.data[:, :, k] .= sin(zc(pm, k))           # the laminar Kolmogorov profile
    end
    for f in (p.u, p.v, p.w)
        interior_view(f) .+= 0.1 .* randn(rng, size(interior_view(f)))
    end
    Aeolus.Simulations._project_velocity!(p)
    for (cf, pf) in ((c.u, p.u), (c.v, p.v), (c.w, p.w))
        interpolate_parent_to_child!(cf, pf, nest)
    end
    # Kolmogorov forcing F₀ sin(z) x̂ in both domains (Kolmogorov flow; the
    # laminar profile is unstable and the flow becomes turbulent)
    force(m) = (sim, Δt) -> begin
        for k in axes(sim.tu.data, 3)
            view(sim.tu.data, :, :, k) .+= F0 * sin(zc(m, k))
        end
    end
    run = NestedRun(p, c, nest; feedback, cfl_parent = CFLConstraints(C_advective = 0.5),
                    advance_parent = domain_advance(step!; body_force! = force(pm)),
                    advance_child = domain_advance(step!; body_force! = force(cm)))
    ts = Float64[]; ke = Float64[]; divmax = 0.0
    while p.t < T - 1e-9
        step_nested!(run)
        divmax = max(divmax, max_divergence(c))
        push!(ts, p.t); push!(ke, kinetic_energy(p) / L^3)
    end
    return ts, ke, divmax, run, all(isfinite, c.u.data) && all(isfinite, p.u.data)
end

# ---- 3. the outgoing-vortex harness in two-way mode -----------------------

function _twl_vortex(h::VortexHarness; nsteps = 2000, record_every = 20)
    truth = _ovh_truth(h)                               # standalone parent (no nest)
    parent = _ovh_truth(h)                              # the two-way parent
    L = h.Nx * h.Δ
    nest = nested_grid(parent.mesh, ((0.0, L), (0.0, h.Δ), (0.0, h.Nz * h.Δ)), 1)
    cm = nest_child_mesh(nest)
    child = BoussinesqSimulation(cm; bcs = nest_bcs(cm, :nest, :periodic; bottom = :wall,
                                                    top = :wall))
    for (cf, pf) in ((child.u, parent.u), (child.v, parent.v), (child.w, parent.w))
        interpolate_parent_to_child!(cf, pf, nest)
    end
    run = NestedRun(parent, child, nest; feedback = :two_way, cfl_parent = nothing,
                    cfl_child = nothing, advance_parent = domain_advance(step_rk3!),
                    advance_child = domain_advance(step_rk3!))
    ii, jj, kk = interior(cm)
    kw = first(kk)+1:last(kk)
    ke_c() = sum(abs2, view(child.u.data, ii, jj, kk)) + sum(abs2, view(child.w.data, ii, jj, kw))
    ke_t() = sum(abs2, view(truth.u.data, ii .+ h.Nx, jj, kk)) +
             sum(abs2, view(truth.w.data, ii .+ h.Nx, jj, kw))
    dev = 0.0
    for n in 1:nsteps
        step_nested!(run; dt = h.dt)
        step_rk3!(truth, h.dt)
        n % record_every == 0 && (dev = max(dev, abs(ke_c() / ke_t() - 1)))
    end
    return dev, child, run
end

@testset "Era II — two-way nesting, long runs (N-nest-e, #263)" begin

@testset "two-way CBL (Hellsten et al. 2021 §4.1, scaled): 2 h at 100 m, r = 3" begin
    tw, run2, wall2 = _twl_cbl(:two_way)
    ow, run1, wall1 = _twl_cbl(:one_way)
    for (name, o, run, wall) in (("one-way", ow, run1, wall1), ("two-way", tw, run2, wall2))
        st = run.stats
        @printf("  CBL %s: %d parent steps, %d child substeps, %d/%d retries, wall %.1f s; max child |∇·u| %.1e; composite θ budget %.1e\n",
                name, st.parent_steps, st.child_substeps, st.parent_retries, st.child_retries,
                wall, o.divmax[], o.budget[])
    end
    # Kinetic energy. A CBL deepening under its inversion gains energy
    # (KE ∝ zᵢ w*² ∝ zᵢ^{5/3}; over the 2nd hour here by ~30–50 %), so the
    # issue's "last-hour mean within 20 % of the hour before" does not hold
    # for 2 h of a growing CBL (it is checked by the 4 h benchmark). The
    # one-way run, whose parent never sees the child, is the reference:
    # two-way must stay within 20 % of it in every quarter-hour of the
    # 2nd hour.
    t = tw.t
    q = [(3600 + 900(i - 1), 3600 + 900i) for i in 1:4]
    ratios = [_twl_window(t, tw.ke, a, b) / _twl_window(ow.t, ow.ke, a, b) for (a, b) in q]
    hr = t .>= 3600
    sk, sek = _twl_fit(t[hr] ./ 3600, log.(tw.ke[hr]))
    so, seo = _twl_fit(ow.t[ow.t .>= 3600] ./ 3600, log.(ow.ke[ow.t .>= 3600]))
    @printf("  KE two-way / one-way, quarter-hours of the 2nd hour: %s; d ln KE/dt: two-way %.3f, one-way %.3f h⁻¹\n",
            join([@sprintf("%.3f", x) for x in ratios], ", "), sk, so)
    # the secondary circulation: the child-mean w at 400 m, mean over the 2nd hour
    wo = _twl_window(ow.t, ow.wc, 3600, 7200); ww = _twl_window(t, tw.wc, 3600, 7200)
    @printf("  child-mean w at 400 m, 2nd hour: one-way %+.3f m/s, two-way %+.3f m/s (Hellsten et al. 2021: 0.4–0.9 m/s mean child updraft)\n",
            wo, ww)
    # the 2Δ mode of θ along the parent rows that cross the nest
    sn, sen = _twl_fit(t[hr] ./ 3600, log.(tw.noise[hr]))
    @printf("  2Δ θ noise on the rows crossing the nest: %.2e K at 1 h, %.2e K at 2 h; fitted growth %.3f ± %.3f h⁻¹\n",
            tw.noise[findfirst(hr)], tw.noise[end], sn, sen)
    @test tw.finite[] && ow.finite[]
    @test all(r -> abs(r - 1) <= 0.2, ratios)
    @test tw.divmax[] < 1e-10
    @test tw.budget[] < 1e-10
    @test sn <= sen                                   # the 2Δ noise does not grow
    @test run2.stats.parent_retries == 0 && run2.stats.child_retries == 0
end

@testset "forced neutral flow: 2 × 20 eddy turnovers, bounded kinetic energy" begin
    # Kolmogorov flow in a 16³ box: the box holds few eddies, and 10-turnover
    # means of its kinetic energy still differ by up to 25 % between windows
    # (one-way as well), so the 20 % criterion is applied to 20-turnover
    # windows (ratios 0.85–1.06 over three seeds, one- and two-way)
    t0 = time()
    T = 300.0
    ts, ke, divmax, run, finite = _twl_neutral(:two_way; T)
    urms = sqrt(2 * _twl_window(ts, ke, 60.0, T))
    Te = 2π / urms                                    # eddy turnover: domain size / U_rms
    k1 = _twl_window(ts, ke, T - 40Te, T - 20Te)
    k2 = _twl_window(ts, ke, T - 20Te, T)
    late = ts .>= T - 40Te
    s, se = _twl_fit(ts[late], log.(ke[late]))
    @printf("  neutral two-way: %d parent steps, %d child substeps, wall %.1f s; U_rms %.3f, turnover %.2f, %.1f turnovers; KE over turnovers −40..−20 / −20..0: %.4f / %.4f (ratio %.3f); trend %.4f ± %.4f per turnover; max child |∇·u| %.1e\n",
            run.stats.parent_steps, run.stats.child_substeps, time() - t0, urms, Te,
            T / Te, k1, k2, k2 / k1, s * Te, se * Te, divmax)
    @test finite
    @test T / Te >= 50
    @test abs(k2 / k1 - 1) <= 0.2
    @test s * Te <= 0.01                              # ≤ 1 % growth per turnover
    @test divmax < 1e-10
end

@testset "outgoing vortex harness (N-open-c, #260) in two-way mode: 2000 steps" begin
    t0 = time()
    dev, child, run = _twl_vortex(VortexHarness())
    @printf("  vortex harness, two-way, 2000 steps: max |∫|u|²_child / ∫|u|²_standalone parent − 1| = %.2e (wall %.1f s)\n",
            dev, time() - t0)
    @test dev < 0.01
    @test all(isfinite, child.u.data)
    @test max_divergence(child) < 1e-10
end

end # testset
