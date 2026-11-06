# Era II / Phase 27 — the warm-bubble benchmark in the three dynamical cores
# (Boussinesq, anelastic, pseudo-incompressible): a +2 K bubble in a 3 km
# neutral domain integrated 130 steps (260 s). Validates that the bubble
# rises in all three, that each core keeps its own constraint, and that the
# sound-proof cores carry more (and more elevated) vertical velocity than
# Boussinesq for the same forcing. The modules stay tested in
# test/anelastic.jl (reference states, solvers, constraints, the
# anelastic ≡ pseudo-incompressible identity on θ̄ = const, diagnostics).

using Test
using Printf
using Aeolus
using Aeolus.Anelastic
using Aeolus.Grids: CellCenter, znodes
using Aeolus.Fields: set!, interior_view

isdefined(@__MODULE__, :_reldiff) || include(joinpath(@__DIR__, "..", "test", "support", "anelastic_norms.jl"))

@testset "Era II — Anelastic & pseudo-incompressible" begin

# =====================================================================
# 8. Warm-bubble benchmark in all three systems.
# =====================================================================
@testset "Warm bubble in Boussinesq / anelastic / pseudo-incompressible" begin
    L, Lz = 3000.0, 3000.0
    g   = RegularGrid(; size=(16,16,32), extent=(L,L,Lz), halo=(2,2,2))
    bcs = FieldBoundaryConditions(:periodic, :periodic, :wall)
    ref = isentropic_reference(g; θ_const=300.0)      # neutral stratification
    # +2 K Gaussian bubble centred at z = 800 m, radius 400–500 m
    bubble(x,y,z) = 2.0 * exp(-(((x-L/2)/500)^2 + ((y-L/2)/500)^2 + ((z-800.0)/400)^2))
    initθ(s) = set!(s.θ, (x,y,z) -> 300.0 + bubble(x,y,z))

    bq = BoussinesqSimulation(g; θ₀=300.0, bcs=bcs);                           initθ(bq)
    an = AnelasticSimulation(g; ref=ref, bcs=bcs, tol=1e-13, maxit=2000);      initθ(an)
    pi_ = PseudoIncompressibleSimulation(g; ref=ref, bcs=bcs, tol=1e-13, maxit=2000); initθ(pi_)

    top0 = bubble_top_height(bq)
    @printf("  [warm bubble] deep domain Lz = %.0f m, H = %.0f m, Lz/H = %.3f, ρ̄ ∈ [%.4f, %.4f]; initial bubble top = %.1f m\n",
            Lz, reference_scale_height(ref), Lz/reference_scale_height(ref),
            minimum(ref.ρ̄), maximum(ref.ρ̄), top0)

    dt, nsteps = 2.0, 130
    for _ in 1:nsteps
        step!(bq, dt)
        step_anelastic!(an, dt)
        step_pseudo_incompressible!(pi_, dt)
    end

    d = system_comparison_diagnostics((boussinesq = bq, anelastic = an, pseudo = pi_))
    for name in keys(d)
        r = d[name]
        @printf("  [warm bubble] %-11s t = %5.1f s  bubble_top = %7.1f m  max_w = %7.4f m/s  θ'var = %.5f K²  KE(unw) = %.4e  constraint = %.2e  |∇·u| = %.3e\n",
                String(name), r.t, r.bubble_top, r.max_w, r.θ_variance,
                r.kinetic_energy_unweighted, r.constraint_residual, r.plain_divergence)
    end
    @printf("  [warm bubble] relative to Boussinesq:  anelastic  ΔKE = %+.4f  Δmax_w = %+.5f  Δθ'var = %+.4f\n",
            d.anelastic.rel_ke, d.anelastic.rel_max_w, d.anelastic.rel_θ_variance)
    @printf("  [warm bubble] relative to Boussinesq:  pseudo-inc ΔKE = %+.4f  Δmax_w = %+.5f  Δθ'var = %+.4f\n",
            d.pseudo.rel_ke, d.pseudo.rel_max_w, d.pseudo.rel_θ_variance)

    # --- (a) all three are physically sensible -------------------------
    for (name, sim) in (("boussinesq", bq), ("anelastic", an), ("pseudo", pi_))
        @test all(isfinite, interior_view(sim.w))
        @test all(isfinite, interior_view(sim.θ))
        @test bubble_top_height(sim) > top0            # the bubble rose
        @test max_w(sim) > 1.0                         # and is still rising
        @test maximum(interior_view(sim.θ)) > 300.0    # warm core survives
        @test θ_variance(sim) > 0
    end
    # w > 0 at the bubble top: the leading edge is ascending
    zc = collect(znodes(g, CellCenter()))
    for sim in (bq, an, pi_)
        ktop = argmin(abs.(zc .- bubble_top_height(sim)))
        @test maximum(@view interior_view(sim.w)[:, :, ktop]) > 0.5
    end

    # --- (b) each core satisfies its own constraint --------------------
    @test max_divergence(bq)                       < 1e-10
    @test max_anelastic_divergence(an)             < 1e-10
    @test max_pseudo_incompressible_divergence(pi_) < 1e-10
    # and the sound-proof cores are NOT divergence-free
    @test max_divergence(an)  > 1e-6
    @test max_divergence(pi_) > 1e-6

    # --- (c) the sound-proof cores differ from Boussinesq, in the
    #         expected direction: with ∇·(ρ̄u) = 0 the *mass* flux is what
    #         is conserved, so the same buoyancy forcing produces a larger
    #         velocity where ρ̄ is smaller — i.e. the anelastic run carries
    #         more vertical velocity overall, and proportionally more of it
    #         aloft.
    #
    #         Note the discriminator is a norm, not a pointwise extremum:
    #         on a 16×16×32 grid `max_w` is a single cell value that flips
    #         sign of difference from step to step, whereas ‖w‖₂ and the
    #         kinetic energy separate the two systems monotonically.
    rel_w = _reldiff(an.w, bq.w)
    wi_b = interior_view(bq.w); wi_a = interior_view(an.w)
    upper = (g.Nz ÷ 2 + 1):g.Nz
    ratio_all   = _l2(wi_a) / _l2(wi_b)
    ratio_upper = _l2(@view wi_a[:, :, upper]) / _l2(@view wi_b[:, :, upper])
    ke_ratio    = unweighted_kinetic_energy(an) / unweighted_kinetic_energy(bq)
    @printf("  [warm bubble] anelastic vs Boussinesq: rel‖Δw‖ = %.4f;  ‖w‖₂ ratio = %.5f;  ‖w‖₂ ratio in the upper half = %.5f;  KE ratio = %.5f\n",
            rel_w, ratio_all, ratio_upper, ke_ratio)
    @test 0.01 < rel_w < 0.5          # genuinely different, not blown up
    @test ratio_all   > 1.0           # stronger ascent for the same forcing
    @test ratio_upper > ratio_all     # and the excess is concentrated aloft
    @test ke_ratio    > 1.0
    @test d.anelastic.rel_ke > 0      # same statement via the comparison tool
    @test d.pseudo.rel_ke    > 0
end

end # @testset "Era II — Anelastic & pseudo-incompressible"
