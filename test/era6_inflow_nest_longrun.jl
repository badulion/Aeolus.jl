# N-inflow-e (#268, epic #246): the synthetic inflow on a nest child over
# two hours — what the Berlin nest runs, on an idealised nest (the Berlin
# case has no test suite of its own; maintainer decision 2026-10-06).
#
# The case: the two-way CBL of N-nest-e (test/twoway_cbl_case.jl, after
# Hellsten et al. 2021, GMD 14, 3185, §4.1: 3.2 × 3.2 × 1.6 km periodic
# parent at 100 m, surface heat flux 0.1 K m/s, 600 m mixed layer under
# 5 K/km) with a mean wind of (3, 1) m/s, so the child's west and south
# faces are inflow faces; the child covers 1.2 × 1.2 km × 800 m on the
# ground at r = 3 (33 m). Generator: DigitalFilterGenerator with the
# parent's resolved statistics (the Berlin default); two-way: the default
# TwoWayFeedback with the inflow band (exclude_inflow_feedback!).
#
#   1. physical effect: one-way, after 2 h the resolved TKE within 200 m of
#      the west (upwind) face, 50-150 m above the ground, is higher with the
#      generator than without (the interpolated parent field lacks the
#      eddies between the parent's and the child's resolution);
#   2. two-way stability with the generator (the maintainer's requirement):
#      no NaN; child and parent domain-mean KE within 1.5x of the one-way
#      run with the generator; parent overlap θ drift < 0.2 K against it, in
#      two realisations (seeds 1 and 2 of the initial θ noise). Two-way
#      *without* the generator meets all three. With the generator it failed
#      until #418 (child KE ×9.7-12.4, parent ×2.1-2.6, overlap θ
#      +0.23-0.26 K): the interface flux correction copied the generator's
#      normal perturbation into the parent's interface faces, the next
#      window's boundary data returned it to the child and the generator
#      added the next one on top; the parent-resolved statistics then
#      measured the grown eddies and raised the amplitude. The correction
#      now gives the parent the child's flux minus the synthetic part
#      (synthetic_normal_velocity); docs/src/theory/transition.md
#      ("Two-way with the generator").

using Test
using Printf
using Random
using Aeolus
using Aeolus.Nesting
using Aeolus.Grids: interior
using Aeolus.Fields: interior_view
using Aeolus.Simulations: max_divergence

const _INL_θS = 300.0

function _inl_run(feedback; generator::Bool, T = 7200.0, U = 3.0, V = 1.0, seed = 1)
    pm = RegularGrid(Float64; size = (32, 32, 16), extent = (3200.0, 3200.0, 1600.0), halo = (3, 3, 3))
    p = BoussinesqSimulation(pm; θ₀ = _INL_θS, C_s = 0.17, θ_flux_bottom = 0.1,
                             bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
    rng = Xoshiro(seed)
    for k in axes(p.θ.data, 3)
        z = (k - pm.Hz - 0.5) * pm.dz
        p.θ.data[:, :, k] .= _INL_θS + (z > 600 ? 0.005 * (z - 600) : 0.0)
    end
    θi = interior_view(p.θ)
    θi[:, :, 1:2] .+= 0.2 .* (rand(rng, size(θi, 1), size(θi, 2), 2) .- 0.5)
    p.u.data .= U; p.v.data .= V
    nest = nested_grid(pm, ((1000.0, 2200.0), (1000.0, 2200.0), (0.0, 800.0)), 3)
    cm = nest_child_mesh(nest)
    c = BoussinesqSimulation(cm; θ₀ = _INL_θS, C_s = 0.17, θ_flux_bottom = 0.1,
                             bcs = nest_bcs(cm, :nest, :nest; bottom = :wall, top = :nest))
    st = nest_interp_state(nest)
    interpolate_parent_state!(st, p, nest; order = 3, region = :all)
    for (cf, sf) in ((c.u, st.u), (c.v, st.v), (c.w, st.w), (c.θ, st.θ))
        cf.data .= sf.data
    end
    run = NestedRun(p, c, nest; feedback, cfl_parent = CFLConstraints(C_advective = 0.5))
    it = generator ? InflowTurbulence(run) : nothing
    excl = generator ? exclude_inflow_feedback!(run, it) : 0
    t0 = time(); divmax = 0.0; added = 0.0; ke_flux = 0.0; n = 0
    while p.t < T - 1e-9
        step_nested!(run)
        divmax = max(divmax, max_divergence(c))
        if it !== nothing
            added = max(added, maximum(a.relative for a in added_mass_flux(it)))
            ke_flux += injected_ke_flux(it).total; n += 1
        end
    end
    cx, cy, cz = nest_covered_range(nest)
    θov = sum(interior_view(p.θ)[cx, cy, cz]) / (length(cx) * length(cy) * length(cz))
    finite = all(isfinite, p.u.data) && all(isfinite, c.u.data) && all(isfinite, p.θ.data) &&
             all(isfinite, c.θ.data)
    return (p = p, c = c, run = run, wall = time() - t0, divmax = divmax, added = added, excl = excl,
            ke_p = kinetic_energy(p) / (3200.0^2 * 1600.0), ke_c = kinetic_energy(c) / (1200.0^2 * 800.0),
            θov = θov, finite = finite, tke_w = _inl_west_tke(c, 200.0, 50.0, 150.0),
            ke_flux = n == 0 ? 0.0 : ke_flux / n)
end

# resolved TKE of the columns within xmax of the west face, between z1 and z2:
# per column and level the variance across y (the plane mean of that column
# row removed), cell-centred, averaged
function _inl_west_tke(c, xmax, z1, z2)
    m = c.mesh
    u, v, w = c.u.data, c.v.data, c.w.data
    nx = round(Int, xmax / m.dx)
    ks = max(1, round(Int, z1 / m.dz)):round(Int, z2 / m.dz)
    mu(a) = sum(a) / length(a)
    var(a) = mu((a .- mu(a)) .^ 2)
    s = 0.0; n = 0
    for k in ks, i in 1:nx
        I, K = m.Hx + i, m.Hz + k
        uu = [(u[I, m.Hy+j, K] + u[I+1, m.Hy+j, K]) / 2 for j in 1:m.Ny]
        vv = [(v[I, m.Hy+j, K] + v[I, m.Hy+j+1, K]) / 2 for j in 1:m.Ny]
        ww = [(w[I, m.Hy+j, K] + w[I, m.Hy+j, K+1]) / 2 for j in 1:m.Ny]
        s += (var(uu) + var(vv) + var(ww)) / 2; n += 1
    end
    return s / n
end

@testset "N-inflow-e: synthetic inflow on a nest child, 2 h (#268)" begin
    off = _inl_run(:one_way; generator = false)
    one = _inl_run(:one_way; generator = true)
    tw0 = _inl_run(:two_way; generator = false)
    two = _inl_run(:two_way; generator = true)
    one2 = _inl_run(:one_way; generator = true, seed = 2)      # the second realisation
    two2 = _inl_run(:two_way; generator = true, seed = 2)
    for (name, r) in (("one-way, no generator", off), ("one-way, generator", one),
                      ("two-way, no generator", tw0), ("two-way, generator", two),
                      ("one-way, generator, s2", one2), ("two-way, generator, s2", two2))
        @printf("  %-22s wall %6.1f s, %4d parent steps: KE parent %.3f child %.3f m²/s², overlap θ %.4f K, west TKE %.3f m²/s², max|∇·u| %.1e, band %d\n",
                name, r.wall, r.run.stats.parent_steps, r.ke_p, r.ke_c, r.θov, r.tke_w, r.divmax, r.excl)
    end
    @printf("  generator: KE flux %.3f W/m² (run mean), worst added flux %.1e of the mean-flow flux\n",
            one.ke_flux, one.added)
    @testset "physical effect: more resolved TKE behind the upwind face" begin
        @printf("  west 200 m, 50-150 m: TKE %.3f (generator) vs %.3f (none): %+.0f %%\n",
                one.tke_w, off.tke_w, 100 * (one.tke_w / off.tke_w - 1))
        @test one.finite && off.finite
        @test one.tke_w > 1.2 * off.tke_w                # measured +39 %
        @test one.added < 1e-12
        @test one.divmax < 1e-10
        @test one.ke_flux > 0
    end
    @testset "two-way without the generator: within the criteria of one-way" begin
        @test tw0.finite
        @test 1 / 1.5 <= tw0.ke_c / off.ke_c <= 1.5 && 1 / 1.5 <= tw0.ke_p / off.ke_p <= 1.5
        @test abs(tw0.θov - off.θov) < 0.2
    end
    @testset "two-way with the generator: within the criteria of one-way (#268, #418)" begin
        for (o, t) in ((one, two), (one2, two2))
            @printf("  two-way/one-way with the generator: KE child ×%.3f, parent ×%.3f, overlap θ %+.4f K\n",
                    t.ke_c / o.ke_c, t.ke_p / o.ke_p, t.θov - o.θov)
            @test t.finite                                   # no NaN
            @test t.excl >= 1                                # the inflow band is set
            @test 1 / 1.5 <= t.ke_c / o.ke_c <= 1.5          # measured ×0.94, ×1.09 (×9.7-12.4 before #418)
            @test 1 / 1.5 <= t.ke_p / o.ke_p <= 1.5          # measured ×1.02, ×1.02 (×2.1-2.6)
            @test abs(t.θov - o.θov) < 0.2                   # measured −0.003, −0.009 K (+0.23-0.26 K)
        end
    end
end
