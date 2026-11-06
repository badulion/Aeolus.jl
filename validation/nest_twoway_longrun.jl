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
#      secondary circulation and the covered-column θ anomaly behind it are
#      measured for one-way and two-way, and the circulation is thermally
#      direct (#327 M5);
#   1b. the resolution dependence of a standalone CBL that drives it (#327);
#   2. a forced neutral flow (Kolmogorov forcing, Smagorinsky) for more
#      than 20 eddy turnovers: bounded kinetic energy;
#   3. N-open-c's outgoing Lamb–Oseen vortex harness (#260) in two-way
#      mode, 2000 steps: the child's ∫|u|² stays within 1 % of the energy
#      the parent supplies (a standalone parent run).
# =========================================================================

isdefined(@__MODULE__, :VortexHarness) || include(joinpath(@__DIR__, "support", "open_boundary_vortex_harness.jl"))

# ---- 1. convective boundary layer: the case is shared with the 4 h
#         benchmark, benchmarks/bench_twoway_cbl.jl ---------------------------
include(joinpath(@__DIR__, "support", "twoway_cbl_case.jl"))

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

@testset "two-way nesting, long runs (N-nest-e, #263)" begin

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
    # and the buoyancy behind it: the parent's θ over the covered columns
    # minus the rest, in the child's lower half (#327 M5)
    θo = _twl_window(ow.t, ow.dθ, 3600, 7200); θw = _twl_window(t, tw.dθ, 3600, 7200)
    @printf("  child-mean w at 400 m, 2nd hour: one-way %+.3f m/s, two-way %+.3f m/s; covered − outside parent θ (0–400 m): one-way %+.4f K, two-way %+.4f K\n",
            wo, ww, θo, θw)
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
    # #327 M5: the two-way circulation is thermally direct — the covered
    # column is heavier than its surroundings where the mean flow over the
    # child descends (in a 6-member ensemble of this case every member has
    # w < 0 and a cool covered column; see the theory page)
    @test ww * θw > 0
    @test ww < 0 && θw < 0
end

# #327 M5: why the covered column is cool. The circulation is driven by the
# resolution dependence of the CBL (Hellsten et al. 2021's hypothesis: a
# different divergence of the vertical heat flux at a different grid
# spacing). Standalone periodic CBLs of this case at the parent's and at the
# child's spacing hold the same heat (the surface flux is prescribed), but
# the finer one carries more of it into the entrainment zone: its layer
# below 800 m (the child's depth) is cooler, so the anterpolated parent
# column is cooler and sinks.
@testset "resolution dependence of the CBL behind the secondary circulation (#327 M5)" begin
    function sa_cbl(Δ; L = 1600.0, H = 1600.0, T = 7200.0, seed = 1)
        N = round(Int, L / Δ); Nz = round(Int, H / Δ)
        m = RegularGrid(Float64; size = (N, N, Nz), extent = (L, L, H), halo = (3, 3, 3))
        s = BoussinesqSimulation(m; θ₀ = _TWL_θS, C_s = 0.17, θ_flux_bottom = 0.1,
                                 bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
        rng = Xoshiro(seed)
        for k in axes(s.θ.data, 3)
            z = (k - m.Hz - 0.5) * m.dz
            s.θ.data[:, :, k] .= _TWL_θS + (z > 600 ? 0.005 * (z - 600) : 0.0)
        end
        θi = interior_view(s.θ); nl = max(1, round(Int, 150 / Δ))
        θi[:, :, 1:nl] .+= 0.2 .* (rand(rng, size(θi, 1), size(θi, 2), nl) .- 0.5)
        c = CFLConstraints(C_advective = 0.5); dtp = nothing
        k8 = round(Int, 800 / Δ); low = 0.0; col = 0.0; acc = 0.0
        while s.t < T - 1e-9
            tp = s.t
            _, dtp, _ = step_adaptive!(s, c; dt_prev = dtp, divergence_tol = Inf)
            s.t <= 3600 && continue
            dt = s.t - tp; acc += dt
            low += dt * (sum(view(θi, :, :, 1:k8)) / (N^2 * k8) - _TWL_θS)
            col += dt * sum(θi .- _TWL_θS) * Δ / N^2
        end
        return low / acc, col / acc
    end
    lc, cc = sa_cbl(100.0)                            # the parent's spacing
    lf, cf = sa_cbl(100.0 / 3)                        # the child's (r = 3)
    @printf("  standalone CBL, 2nd hour: mean θ − θ_s below 800 m %.4f K at 100 m, %.4f K at 33 m (fine − coarse %+.4f K); column heat %.2f / %.2f K m\n",
            lc, lf, lf - lc, cc, cf)
    @test lf - lc < -0.03                             # measured −0.087 (2 seeds: −0.087, −0.072)
    @test abs(cf - cc) / cc < 1e-3                    # the same heat, redistributed upward
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
