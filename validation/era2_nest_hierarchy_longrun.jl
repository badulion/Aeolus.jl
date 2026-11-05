using Test
using Printf
using Random
using Aeolus
using Aeolus.Nesting
using Aeolus.Fields: interior_view

# =========================================================================
# #328 — long-run stability of a two-way nest hierarchy: two nest levels
# (root → child → grandchild, both links two-way with refluxing) in a
# forced neutral flow (Kolmogorov forcing, Smagorinsky), adaptive steps on
# every level, for 50 eddy turnovers. The same case with the child only
# (two-level) is the reference for the kinetic energy and the cost.
# =========================================================================

function _nhl_neutral(levels::Int; N = 16, F0 = 0.15, T = 300.0, seed = 2)
    L = 2π
    pm = RegularGrid(Float64; size = (N, N, N), extent = (L, L, L), halo = (3, 3, 3))
    p = BoussinesqSimulation(pm; C_s = 0.17, bcs = FieldBoundaryConditions(:periodic))
    zc(m, k) = m.z0 + (k - m.Hz - 0.5) * m.dz
    rng = Xoshiro(seed)
    for k in axes(p.u.data, 3)
        p.u.data[:, :, k] .= sin(zc(pm, k))           # the laminar Kolmogorov profile
    end
    for f in (p.u, p.v, p.w)
        interior_view(f) .+= 0.1 .* randn(rng, size(interior_view(f)))
    end
    Aeolus.Simulations._project_velocity!(p)
    # a passive θ blob: the composite integral the registers must conserve
    blob!(s) = (m = s.mesh; for k in axes(s.θ.data, 3), j in axes(s.θ.data, 2), i in axes(s.θ.data, 1)
                    x = m.x0 + (i - m.Hx - 0.5) * m.dx; y = m.y0 + (j - m.Hy - 0.5) * m.dy
                    s.θ.data[i, j, k] = 1 + exp(-((x - L / 2)^2 + (y - L / 2)^2 + (zc(m, k) - L / 2)^2) / 0.5)
                end; s)
    blob!(p)
    force(m) = (sim, Δt) -> begin
        for k in axes(sim.tu.data, 3)
            view(sim.tu.data, :, :, k) .+= F0 * sin(zc(m, k))
        end
    end
    cfl = CFLConstraints(C_advective = 0.5)
    h = NestHierarchy(p; advance = domain_advance(step!; body_force! = force(pm)), cfl)
    sims = Any[p]
    parent = :root; pmesh = pm; ext = (L / 4, 3L / 4)
    for n in 1:levels
        nest = nested_grid(pmesh, (ext, ext, ext), 2)
        cm = nest_child_mesh(nest)
        c = BoussinesqSimulation(cm; C_s = 0.17, bcs = nest_bcs(cm, :nest))
        for (cf, pf) in ((c.u, sims[end].u), (c.v, sims[end].v), (c.w, sims[end].w))
            interpolate_parent_to_child!(cf, pf, nest)
        end
        blob!(c)
        id = Symbol(:L, n)
        add_nest!(h, parent => c, nest; id, advance = domain_advance(step!; body_force! = force(cm)),
                  feedback = :two_way)
        push!(sims, c); parent = id; pmesh = cm
        ext = (ext[1] + (ext[2] - ext[1]) / 4, ext[2] - (ext[2] - ext[1]) / 4)
    end
    ts = Float64[]; ke = Float64[]; divmax = zeros(length(sims)); worst = 0.0
    I0 = composite_integral(h, :θ); Ip = I0
    t0 = time()
    while p.t < T - 1e-9
        step_nested!(h)
        for (n, s) in enumerate(sims)
            divmax[n] = max(divmax[n], max_divergence(s))
        end
        I = composite_integral(h, :θ); worst = max(worst, abs(I - Ip) / I0); Ip = I
        push!(ts, p.t); push!(ke, kinetic_energy(p) / L^3)
    end
    wall = time() - t0
    finite = all(s -> all(isfinite, s.u.data) && all(isfinite, s.θ.data), sims)
    return (; ts, ke, divmax, worst, h, finite, wall, sims)
end

_nhl_window(t, x, a, b) = (m = (t .>= a) .& (t .<= b); sum(x[m]) / count(m))

function _nhl_fit(x, y)
    n = length(x); xm = sum(x) / n; ym = sum(y) / n
    sxx = sum((x .- xm) .^ 2); s = sum((x .- xm) .* (y .- ym)) / sxx
    r = y .- (ym .+ s .* (x .- xm))
    return s, sqrt(sum(r .^ 2) / (n - 2) / sxx)
end

@testset "Era II — nest hierarchy long run (#328)" begin

@testset "forced neutral flow, two two-way nest levels: 50 eddy turnovers" begin
    T = 300.0
    r3 = _nhl_neutral(2; T)
    r2 = _nhl_neutral(1; T)
    for (name, r) in (("three domains", r3), ("two domains", r2))
        urms = sqrt(2 * _nhl_window(r.ts, r.ke, 60.0, T))
        Te = 2π / urms
        k1 = _nhl_window(r.ts, r.ke, T - 40Te, T - 20Te); k2 = _nhl_window(r.ts, r.ke, T - 20Te, T)
        late = r.ts .>= T - 40Te
        s, se = _nhl_fit(r.ts[late], log.(r.ke[late]))
        st = [nest_link(r.h, d.id).stats for d in r.h.domains[2:end]]
        @printf("  %s: %d root steps, substeps per level %s, retries %s; wall %.1f s (%.2f s per unit time); %.1f turnovers; KE windows −40..−20 / −20..0 turnovers ratio %.3f, trend %.4f ± %.4f per turnover; max |∇·u| per level %s; composite ∫θ per step %.1e\n",
                name, r.h.root_steps, string([x.child_substeps for x in st]),
                string([x.child_retries + x.parent_retries for x in st]), r.wall, r.wall / T, T / Te,
                k2 / k1, s * Te, se * Te, string(round.(r.divmax; sigdigits = 2)), r.worst)
        @test r.finite
        @test T / Te >= 50
        @test abs(k2 / k1 - 1) <= 0.2
        @test s * Te <= 0.01                          # ≤ 1 % growth per turnover
        @test all(<(1e-10), r.divmax)
        @test r.worst <= 1e-11
        @test all(x -> x.child_retries == 0 && x.parent_retries == 0, st)
    end
    # the kinetic energy of the root does not depend on the extra level
    # beyond the scatter of 20-turnover means (≤ 20 %)
    urms = sqrt(2 * _nhl_window(r2.ts, r2.ke, 60.0, T)); Te = 2π / urms
    q = _nhl_window(r3.ts, r3.ke, T - 20Te, T) / _nhl_window(r2.ts, r2.ke, T - 20Te, T)
    @printf("  root KE, last 20 turnovers, three / two domains: %.3f; cost ratio %.2f\n", q, r3.wall / r2.wall)
    @test abs(q - 1) <= 0.2
end

end # testset
