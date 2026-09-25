# Phase 43, issue #5 — prognostic ice-nucleating-particle species.
#
# The tests are organised by the physics each operator must respect:
# analytic log-normal moments against an independent quadrature,
# conservation of number and mass through every transport-free source
# and sink, the available/consumed split that stops an INP nucleating
# twice, and the published Stokes settling speeds.

using Test, Printf, Random
using Aeolus
using Aeolus.Superdroplet: stokes_settling_velocity, MicrophysicalConstants,
                           DEFAULT_MICRO


# Independent reference: midpoint rule in ln r over a very wide range.
# Deliberately NOT the module's own quadrature (which works in shifted
# standard-normal variables), so agreement is a real check.
function _ref_moment_above(m::ARGMode, k::Integer, r_cut::Real; n = 200_000)
    lnσ = log(m.σ_g)
    lo = max(log(r_cut), log(m.r_g) - 14lnσ)
    hi = log(m.r_g) + 14lnσ
    lo >= hi && return 0.0
    h = (hi - lo) / n
    s = 0.0
    for j in 1:n
        x = lo + (j - 0.5) * h
        z = (x - log(m.r_g)) / lnσ
        s += exp(k * x) * exp(-0.5z^2)
    end
    return m.N / (sqrt(2π) * lnσ) * s * h
end


# u with erfc(u)/2 = f, by bisection (independent of the module's copy)
function _erfc_inv_half(f)
    lo, hi = -10.0, 10.0
    for _ in 1:200
        mid = (lo + hi) / 2
        0.5 * Aeolus.GasOptics.erfc_series(mid) > f ? (lo = mid) : (hi = mid)
    end
    return (lo + hi) / 2
end

@testset "Prognostic INP species (issue #5)" begin

    # ---------------------------------------------------------------------
    @testset "Shared ingredients live in their home modules" begin
        # Biological INP are a species of the Phase 37 library, not a
        # private record of this module.
        bio = aerosol_species(:biological)
        @test bio === aerosol_species(:bioaerosol)
        @test bio.ρ == 1.2
        @test bio.κ == 0.1
        @test aerosol_species(:dust) === aerosol_species(:mineral_dust)
        m = lognormal_mode(:biological, 0.01, 1.0, 1.6)
        @test m.species === bio

        # Stokes settling is regime 1 of the droplet terminal velocity:
        # one drag law, bit-identical for water.
        for r in (1e-7, 5e-7, 2e-6, 9e-6)
            @test stokes_settling_velocity(r, DEFAULT_MICRO.ρ_water) ==
                  terminal_velocity(r)
        end
        # v ∝ r² without slip, ∝ (ρ_p - ρ_a)
        ns = MicrophysicalConstants(λ_air = 0.0)
        @test stokes_settling_velocity(2e-6, 2600.0, ns) /
              stokes_settling_velocity(1e-6, 2600.0, ns) ≈ 4 rtol = 1e-14
        @test stokes_settling_velocity(1e-6, 2600.0, ns) /
              stokes_settling_velocity(1e-6, 1200.0, ns) ≈
              (2600 - 1.225) / (1200 - 1.225) rtol = 1e-14
        # Cunningham slip: 1 + 2.51 λ/d = 1.83 at d = 0.2 μm
        @test stokes_settling_velocity(1e-7, 2600.0) /
              stokes_settling_velocity(1e-7, 2600.0, ns) ≈
              1 + 2.51 * 6.62e-8 / 2e-7 rtol = 1e-14
        @test stokes_settling_velocity(0.0, 2600.0) == 0.0
    end

    # ---------------------------------------------------------------------
    @testset "Published benchmark: Stokes settling speeds" begin
        # Seinfeld & Pandis (2016), Table 9.4: unit-density spheres at
        # 298 K, 1 atm — 3.5e-3 cm s⁻¹ at D = 1 μm, 0.31 cm s⁻¹ at 10 μm.
        # The tabulation uses a slightly different slip correction and
        # viscosity (298 vs 293 K), hence the 3 % tolerance.
        v1 = stokes_settling_velocity(0.5e-6, 1000.0)
        v10 = stokes_settling_velocity(5e-6, 1000.0)
        @test v1 ≈ 3.5e-5 rtol = 0.03
        @test v10 ≈ 3.1e-3 rtol = 0.03
        @info "Stokes settling vs Seinfeld & Pandis Table 9.4" v_1μm_cm_s = 100v1 v_10μm_cm_s = 100v10
    end

    # ---------------------------------------------------------------------
    @testset "Log-normal surface area and N(>r): analytic vs quadrature" begin
        worst = 0.0
        for σ in (1.4, 1.8, 2.2, 2.8), r_g in (5e-8, 4e-7, 2e-6)
            m = arg_mode(1e6, r_g, σ, 0.03; ρ_p = 2600.0)
            A = mode_surface_area(m)
            Aref = 4π * _ref_moment_above(m, 2, 0.0)
            worst = max(worst, abs(A - Aref) / Aref)
            for rc in (0.3r_g, r_g, 3r_g, DEMOTT_CUTOFF_RADIUS)
                a = surface_area_above_radius(m, rc)
                aref = 4π * _ref_moment_above(m, 2, rc)
                n = number_above_radius(m, rc)
                nref = _ref_moment_above(m, 0, rc)
                aref > 1e-12A && (worst = max(worst, abs(a - aref) / aref))
                nref > 1e-12m.N && (worst = max(worst, abs(n - nref) / nref))
            end
        end
        @test worst < 1e-8
        @info "log-normal surface area / N(>r): analytic vs quadrature" worst_rel_error = worst

        # limits and the area median
        m = arg_mode(1e6, 3e-7, 2.0, 0.03; ρ_p = 2600.0)
        A = mode_surface_area(m)
        @test surface_area_above_radius(m, 0.0) == A
        @test surface_area_above_radius(m, 1.0) < 1e-30A
        r_area_median = m.r_g * exp(2log(m.σ_g)^2)
        @test surface_area_above_radius(m, r_area_median) ≈ A / 2 rtol = 1e-12
        # the r²-weighted cut sits above the number cut: at any radius a
        # larger fraction of area than of number lies above it
        for rc in (1e-7, 3e-7, 1e-6)
            @test surface_area_above_radius(m, rc) / A > number_above_radius(m, rc) / m.N
        end

        # LogNormalMode convenience = SI conversion (cm⁻³, μm → m⁻³, m)
        lm = lognormal_mode(:mineral_dust, 2.0, 0.4, 2.0)
        @test mode_surface_area(lm) ≈
              4π * 2e6 * (0.4e-6)^2 * exp(2log(2.0)^2) rtol = 1e-14
        @test surface_area_above_radius(lm, 0.25e-6) ≈
              surface_area_above_radius(arg_mode(lm), 0.25e-6) rtol = 1e-14
    end

    # ---------------------------------------------------------------------
    @testset "Population construction and diagnostics" begin
        dust = lognormal_mode(:mineral_dust, 1.0, 0.4, 2.0)   # 1 cm⁻³
        bio = lognormal_mode(:biological, 0.01, 1.0, 1.6)
        inp = inp_population([dust, bio])
        @test inp_nmodes(inp) == 2
        @test inp_species(inp, 1).name === :mineral_dust
        @test inp_species(inp, 2).name === :biological
        @test inp_number(inp) ≈ 1.01e6 rtol = 1e-14
        @test inp_number(inp; mode = 2) ≈ 1e4 rtol = 1e-14
        @test inp_number(inp; population = :consumed) == 0.0
        # dry mass agrees with the Phase 37 value (μg m⁻³ → kg m⁻³)
        @test inp_mass(inp; mode = 1) ≈ mode_mass(dust) * 1e-9 rtol = 1e-12
        # the modal closure recovers the median radius it was built from
        @test inp_median_radius(inp, 1) ≈ 0.4e-6 rtol = 1e-12
        @test inp_median_radius(inp, 2) ≈ 1.0e-6 rtol = 1e-12
        @test inp_modes(inp)[1].r_g ≈ 0.4e-6 rtol = 1e-12
        # DeMott N(D > 0.5 μm): a mode whose median sits exactly at the
        # cut-off has half its number above it
        half = inp_population([lognormal_mode(:mineral_dust, 1.0, 0.25, 2.0)])
        @test inp_number_above_radius(half, DEMOTT_CUTOFF_RADIUS) ≈ 0.5e6 rtol = 1e-12
        @test inp_surface_area(inp) ≈
              mode_surface_area(dust) + mode_surface_area(bio) rtol = 1e-12
        @test inp_surface_area_above_radius(inp, 0.0) ≈ inp_surface_area(inp) rtol = 1e-14

        # ARGMode constructor with a species symbol
        inp2 = inp_population([arg_mode(1e5, 5e-7, 1.8, 0.03; ρ_p = 2600.0)],
                              :mineral_dust)
        @test inp_species(inp2, 1) === aerosol_species(:mineral_dust)
        @test_throws ErrorException inp_population([arg_mode(1e5, 5e-7, 1.8, 0.03)],
                                                   [:dust, :biological])
        # density comes from the species: arg_mode's default ρ_p is
        # sulphate's 1770 kg m⁻³, which must not silently become dust
        @test_throws ErrorException inp_population([arg_mode(1e5, 5e-7, 1.8, 0.03)],
                                                   :mineral_dust)
        @test inp2.ρ_p[1] == 1e3 * aerosol_species(:mineral_dust).ρ
    end

    # ---------------------------------------------------------------------
    @testset "Transport-free conservation of number and mass" begin
        # A random sequence of every source and sink operator. After each
        # step: available + consumed = initial + sources - removals, and
        # consumed = nucleated - released - consumed-removed, for number
        # and mass, per mode and in total; nothing ever goes negative.
        rng = MersenneTwister(43)
        worst = 0.0
        nsteps = 0
        massless = false
        for trial in 1:5
            inp = inp_population([lognormal_mode(:mineral_dust, 0.5, 0.3, 2.0),
                                  lognormal_mode(:mineral_dust, 0.05, 1.5, 1.9),
                                  lognormal_mode(:biological, 0.02, 1.0, 1.6)])
            env_N = [4e5, 3e4, 1e4]
            env_M = [inp_mass(inp; mode = i) * 0.8 for i in 1:3]
            for step in 1:300
                i = rand(rng, 1:3)
                op = rand(rng, 1:9)
                if op == 1
                    inp_emit!(inp, i, rand(rng) * 1e4, rand(rng) * 1e-12)
                elseif op == 2
                    inp_emit!(inp, i, arg_mode(50.0, 4e-7, 2.0, 0.03; ρ_p = 2600.0),
                              10rand(rng))
                elseif op == 3
                    inp_entrain!(inp, env_N, env_M, 0.1rand(rng))
                elseif op == 4
                    inp_consume!(inp, i, rand(rng) * 0.2 * inp_number(inp; mode = i);
                                 selection = rand(rng, (:largest, :random)))
                elseif op == 5
                    c = 1e10 * rand(rng)
                    inp_consume!(inp, i, r -> 1 - exp(-c * 4π * r^2))
                elseif op == 6
                    ns = 10.0^(9 + 3rand(rng))
                    inp_nucleate_singular!(inp, i, r -> 1 - exp(-ns * 4π * r^2))
                elseif op == 7
                    inp_release!(inp, 0.3rand(rng); mode = rand(rng, 0:3))
                elseif op == 8
                    inp_scavenge!(inp, rand(rng) < 0.5 ? 1e-4 : (r -> 1e8 * r^2),
                                  100rand(rng);
                                  population = rand(rng, (:available, :consumed, :both)))
                else
                    inp_sediment!(inp, 50.0, 600rand(rng))
                end
                nsteps += 1
                for md in 0:3
                    res = inp_budget_residual(inp; mode = md)
                    worst = max(worst, maximum(abs, values(res)))
                end
                @assert all(>=(0), inp.N_avail) && all(>=(0), inp.M_avail) &&
                        all(>=(0), inp.N_cons) && all(>=(0), inp.M_cons)
                # every transfer moves particles together with their mass
                massless = massless ||
                    any((inp.N_cons .> 0) .!= (inp.M_cons .> 0)) ||
                    any((inp.N_avail .> 0) .!= (inp.M_avail .> 0))
            end
            b = inp_budget(inp)
            @test b.N_total ≈ b.N_avail + b.N_cons
            @test all(>=(0), inp.N_avail) && all(>=(0), inp.M_avail)
            @test all(>=(0), inp.N_cons) && all(>=(0), inp.M_cons)
            # number and mass jointly positive: no massless particles
            @test all((inp.N_avail .> 0) .== (inp.M_avail .> 0))
        end
        @test worst < 1e-13
        @test !massless
        @info "INP budget closure over random source/sink sequences" steps = nsteps worst_rel_residual = worst
    end

    # ---------------------------------------------------------------------
    @testset "An INP cannot nucleate twice" begin
        dust = lognormal_mode(:mineral_dust, 1.0, 0.4, 2.0)
        inp = inp_population([dust])
        N0 = inp_number(inp)
        # singular INAS-type indicator at two temperatures: n_s grows
        # roughly exponentially on cooling (Niemand et al. 2012 form)
        p(ns) = r -> 1 - exp(-ns * 4π * r^2)
        ns_warm, ns_cold = 1e10, 1e11

        Δ1 = inp_nucleate_singular!(inp, 1, p(ns_warm))
        target_warm = first(inp_frozen_fraction(inp, 1, p(ns_warm))) * N0
        @test Δ1 > 0
        @test inp_number(inp; population = :consumed) ≈ target_warm rtol = 1e-12
        # same temperature again: exactly nothing
        @test inp_nucleate_singular!(inp, 1, p(ns_warm)) == 0.0
        @test inp_nucleate_singular!(inp, 1, p(ns_warm)) == 0.0
        # colder: only the increment
        Δ2 = inp_nucleate_singular!(inp, 1, p(ns_cold))
        target_cold = first(inp_frozen_fraction(inp, 1, p(ns_cold))) * N0
        @test Δ2 ≈ target_cold - target_warm rtol = 1e-10
        @test inp_number(inp; population = :consumed) ≈ target_cold rtol = 1e-12
        # warming back does not un-freeze or freeze anything
        @test inp_nucleate_singular!(inp, 1, p(ns_warm)) == 0.0
        @info "singular nucleation increments" frac_warm = target_warm / N0 frac_cold = target_cold / N0

        # number form with a prescribed N_frozen
        inp = inp_population([dust])
        @test inp_nucleate_singular!(inp, 1, 2e5) ≈ 2e5
        @test inp_nucleate_singular!(inp, 1, 2e5) == 0.0
        @test inp_nucleate_singular!(inp, 1, 3e5) ≈ 1e5 rtol = 1e-12

        # ice recycling: released INP re-freeze at the same temperature,
        # and exactly the released number does
        rel = inp_release!(inp, 0.5)
        @test rel ≈ 1.5e5 rtol = 1e-12
        @test inp_nucleate_singular!(inp, 1, 3e5) ≈ rel rtol = 1e-12

        # exhaustion: asking for more than exists can never consume more
        # than was ever available
        inp = inp_population([dust])
        total = 0.0
        for _ in 1:5
            total += inp_consume!(inp, 1, 0.6N0)
        end
        @test total ≈ N0 rtol = 1e-14
        @test inp_number(inp) == 0.0
        @test inp_mass(inp) == 0.0
        @test inp_consume!(inp, 1, 1.0) == 0.0

        # Mixing selections must still move a realisable set of particles.
        # Take the largest 3 % first (the DeMott number form does this),
        # then apply an INAS indicator whose frozen number exceeds that but
        # whose p-weighted mass (38 %) is below what the largest 3 % already
        # carry (58 %). The naive mass increment fM·M_tot - M_cons is then
        # negative; the review of issue #5 found 1.6e4 m⁻³ particles moving
        # with zero mass here. The increment must carry between the mass of
        # the ΔN smallest and ΔN largest available particles.
        inp = inp_population([dust])
        inp_consume!(inp, 1, 0.03N0; selection = :largest)
        m_av = inp_modes(inp)[1]
        Mc0 = inp_mass(inp; population = :consumed)
        ΔN = inp_nucleate_singular!(inp, 1, p(ns_warm))
        ΔM = inp_mass(inp; population = :consumed) - Mc0
        @test ΔN > 0
        @test ΔM > 0
        # bounds: mass of the ΔN smallest / ΔN largest available particles
        r_hold(m, n) = m.r_g * exp(sqrt(2) * log(m.σ_g) *
                                   _erfc_inv_half(n / m.N))
        hi = mass_above_radius(m_av, r_hold(m_av, ΔN))
        lo = mode_dry_mass(m_av) - mass_above_radius(m_av, r_hold(m_av, m_av.N - ΔN))
        @test lo * (1 - 1e-9) <= ΔM <= hi * (1 + 1e-9)
        # and it is the mean mass of the available particles p selects
        # (the fallback), measured independently by the frozen fraction
        fNa, fMa = inp_frozen_fraction(inp_population([m_av], :mineral_dust), 1,
                                       p(ns_warm); population = :available)
        @test ΔM / ΔN ≈ fMa * mode_dry_mass(m_av) / (fNa * m_av.N) rtol = 1e-9
        # the per-particle mass moved is above the available mean: INAS
        # freezes the larger of the particles still available
        @test ΔM / ΔN > mode_dry_mass(m_av) / m_av.N
        @info "singular increment after largest-first consumption" ΔN ΔM_over_lo = ΔM / lo ΔM_over_hi = ΔM / hi
    end

    # ---------------------------------------------------------------------
    @testset "Which particles nucleate: selection physics" begin
        dust = lognormal_mode(:mineral_dust, 1.0, 0.4, 2.0)
        rg0 = 0.4e-6

        # size-independent removal leaves the median radius unchanged
        inp = inp_population([dust])
        inp_consume!(inp, 1, 3e5; selection = :random)
        @test inp_median_radius(inp, 1) ≈ rg0 rtol = 1e-12
        @test inp_median_radius(inp, 1; population = :consumed) ≈ rg0 rtol = 1e-12

        # largest-first: consumed particles are coarser than average, the
        # remainder finer
        inp = inp_population([dust])
        inp_consume!(inp, 1, 3e5; selection = :largest)
        @test inp_median_radius(inp, 1) < rg0
        @test inp_median_radius(inp, 1; population = :consumed) > rg0
        # and the mass moved is exactly the log-normal mass above the
        # radius that holds 3e5 particles
        m = arg_mode(dust)
        # independent inversion: N(>r_c) = 3e5 ⇔ erfc(u)/2 = 0.3
        r_c = m.r_g * exp(sqrt(2) * log(m.σ_g) * _erfc_inv_half(0.3))
        @test number_above_radius(m, r_c) ≈ 3e5 rtol = 1e-10
        @test inp_mass(inp; population = :consumed) ≈ mass_above_radius(m, r_c) rtol = 1e-10
        mfrac = inp_mass(inp; population = :consumed) / mode_dry_mass(m)
        # for σ_g = 2 the largest 30 % of particles carry 0.5 erfc(u - 3lnσ/√2)
        # = 94 % of the mass (u = erfc⁻¹(0.6) = 0.371, shift 1.470)
        @test mfrac ≈ 0.5 * Aeolus.GasOptics.erfc_series(_erfc_inv_half(0.3) -
                                                          3log(2.0) / sqrt(2)) rtol = 1e-10
        @test 0.93 < mfrac < 0.95
        @info "largest-first: mass fraction carried by 30 % of number" mfrac

        # probability form: p ≡ c moves exactly c of every moment
        inp = inp_population([dust])
        N0, M0 = inp_number(inp), inp_mass(inp)
        inp_consume!(inp, 1, r -> 0.37)
        @test inp_number(inp; population = :consumed) ≈ 0.37N0 rtol = 1e-12
        @test inp_mass(inp; population = :consumed) ≈ 0.37M0 rtol = 1e-12
        @test inp_median_radius(inp, 1) ≈ rg0 rtol = 1e-12

        # INAS small-n_s limit: frozen number → n_s × surface area
        # (Hoose & Möhler 2012; Niemand et al. 2012) — ties the
        # probability integral to the analytic second moment
        inp = inp_population([dust])
        A = inp_surface_area(inp)
        for ns in (1e2, 1e4, 1e6)
            fN, _ = inp_frozen_fraction(inp, 1, r -> 1 - exp(-ns * 4π * r^2))
            @test fN * inp_number(inp) ≈ ns * A rtol = 5 * ns * A / inp_number(inp) + 1e-10
        end

        # smooth steep freezing curve vs an independent quadrature
        ns = 3e10
        fN, fM = inp_frozen_fraction(inp, 1, r -> 1 - exp(-ns * 4π * r^2))
        lnσ = log(2.0); n = 200_000; lo = log(rg0) - 14lnσ; h = 28lnσ / n
        sN = 0.0; sM = 0.0; sN0 = 0.0; sM0 = 0.0
        for j in 1:n
            x = lo + (j - 0.5)h; r = exp(x); w = exp(-0.5((x - log(rg0)) / lnσ)^2)
            pr = 1 - exp(-ns * 4π * r^2)
            sN += pr * w; sM += pr * w * r^3; sN0 += w; sM0 += w * r^3
        end
        errN = abs(fN - sN / sN0); errM = abs(fM - sM / sM0)
        @test errN < 1e-10
        @test errM < 1e-10
        @info "INAS frozen fraction: module quadrature vs reference" fN fM errN errM

        # a hard step (DeMott-like "everything above r_c") through the
        # probability form reproduces the analytic cut-off to the node
        # spacing of the quadrature, O(h)
        inp = inp_population([dust])
        rc = DEMOTT_CUTOFF_RADIUS
        fN, _ = inp_frozen_fraction(inp, 1, r -> r > rc ? 1.0 : 0.0)
        exact = number_above_radius(m, rc) / m.N
        @test fN ≈ exact atol = 5e-3
        @info "step freezing curve vs analytic N(>r)" quadrature = fN analytic = exact
    end

    # ---------------------------------------------------------------------
    @testset "Sources: emission and entrainment" begin
        dust = lognormal_mode(:mineral_dust, 1.0, 0.4, 2.0)
        inp = inp_population([dust])
        N0, M0 = inp_number(inp), inp_mass(inp)
        # a source with the resident shape leaves r_g unchanged
        src = arg_mode(100.0, 0.4e-6, 2.0, 0.03; ρ_p = 2600.0)   # m⁻³ s⁻¹
        inp_emit!(inp, 1, src, 600.0)
        @test inp_number(inp) ≈ N0 + 6e4 rtol = 1e-14
        @test inp_median_radius(inp, 1) ≈ 0.4e-6 rtol = 1e-12
        # a coarser source coarsens the population
        inp_emit!(inp, 1, arg_mode(100.0, 1.5e-6, 2.0, 0.03; ρ_p = 2600.0), 600.0)
        @test inp_median_radius(inp, 1) > 0.4e-6
        @test_throws ErrorException inp_emit!(inp, 1, -1.0, 1e-12)
        @test_throws ErrorException inp_emit!(inp, 1, 1.0, 0.0)

        # entrainment: linear relaxation toward the environment
        inp = inp_population([dust])
        inp_consume!(inp, 1, 1e5)
        Nc = inp_number(inp; population = :consumed)
        Na = inp_number(inp)
        inp_entrain!(inp, [2e5], [1e-10], 0.25)
        @test inp_number(inp) ≈ 0.75Na + 0.25 * 2e5 rtol = 1e-14
        @test inp_number(inp; population = :consumed) == Nc   # ice owns these
        inp_entrain!(inp, [2e5], [1e-10], 1.0)
        @test inp_number(inp) ≈ 2e5 rtol = 1e-14
        @test inp_mass(inp) ≈ 1e-10 rtol = 1e-14
        @test inp_budget(inp).N_src < 0          # cleaner environment
        @test maximum(abs, values(inp_budget_residual(inp))) < 1e-14
        # an environment with particles but no mass (or mass but no
        # particles) is not a population; f = 1 would copy it in
        @test_throws ErrorException inp_entrain!(inp, [2e5], [0.0], 1.0)
        @test_throws ErrorException inp_entrain!(inp, [0.0], [1e-10], 1.0)
        @test inp_entrain!(inp, [0.0], [0.0], 1.0) ≈ -2e5 rtol = 1e-14
        @test inp_number(inp) == 0.0 && inp_mass(inp) == 0.0

        # a NaN probability (e.g. 0·Inf from an overflowing n_s(T)) is an
        # error, not a silent NaN in the tracer
        inp = inp_population([dust])
        @test_throws ErrorException inp_consume!(inp, 1, r -> NaN)
        @test inp_number(inp) == N0
        # but an overflowing rate that saturates is fine: everything freezes
        @test inp_consume!(inp, 1, r -> 1 - exp(-Inf * r^2)) ≈ N0 rtol = 1e-14
    end

    # ---------------------------------------------------------------------
    @testset "Scavenging" begin
        dust = lognormal_mode(:mineral_dust, 1.0, 0.4, 2.0)
        Λ = 2e-4
        # exact exponential: 1 step of 3600 s = 36 steps of 100 s
        a = inp_population([dust]); b = inp_population([dust])
        inp_scavenge!(a, Λ, 3600.0)
        for _ in 1:36
            inp_scavenge!(b, Λ, 100.0)
        end
        @test inp_number(a) ≈ 1e6 * exp(-Λ * 3600) rtol = 1e-12
        @test inp_number(b) ≈ inp_number(a) rtol = 1e-12
        @test inp_mass(b) ≈ inp_mass(a) rtol = 1e-12
        @test inp_median_radius(a, 1) ≈ 0.4e-6 rtol = 1e-12    # size-blind
        # size-dependent Λ ∝ r² strips the coarse tail
        c = inp_population([dust])
        inp_scavenge!(c, r -> 1e9 * r^2, 3600.0)
        @test inp_median_radius(c, 1) < 0.4e-6
        @test inp_mass(c) / inp_mass(inp_population([dust])) <
              inp_number(c) / 1e6
        # removal of consumed INP with precipitating ice
        d = inp_population([dust]); inp_consume!(d, 1, 5e5)
        inp_scavenge!(d, 1e-3, 1000.0; population = :consumed)
        @test inp_number(d; population = :consumed) ≈ 5e5 * exp(-1.0) rtol = 1e-12
        @test inp_budget(d).N_cons_lost ≈ 5e5 * (1 - exp(-1.0)) rtol = 1e-12
        @test inp_number(d) ≈ 5e5 rtol = 1e-12
    end

    # ---------------------------------------------------------------------
    @testset "Sedimentation" begin
        ns = MicrophysicalConstants(λ_air = 0.0)
        worst = 0.0
        for σ in (1.5, 2.0, 2.4), rg in (2e-7, 1e-6)
            inp = inp_population([arg_mode(1e6, rg, σ, 0.03; ρ_p = 2600.0)],
                                 :mineral_dust)
            k = 2 * (2600 - ns.ρ_air) * ns.grav / (9 * ns.μ_air)
            vN = k * rg^2 * exp(2log(σ)^2)
            vM = k * rg^2 * exp(8log(σ)^2)
            worst = max(worst,
                abs(inp_settling_velocity(inp, 1; micro = ns) - vN) / vN,
                abs(inp_settling_velocity(inp, 1; moment = 3, micro = ns) - vM) / vM)
        end
        @test worst < 1e-10
        @info "moment-weighted Stokes velocity vs closed form" worst_rel_error = worst

        # mass settles faster than number: e^{6 ln²2} = 17.8 for σ_g = 2
        inp = inp_population([lognormal_mode(:mineral_dust, 1.0, 0.4, 2.0)])
        ratio = inp_settling_velocity(inp, 1; moment = 3, micro = ns) /
                inp_settling_velocity(inp, 1; micro = ns)
        @test ratio ≈ exp(6log(2.0)^2) rtol = 1e-10
        # slip speeds up the fine mode
        @test inp_settling_velocity(inp, 1) > inp_settling_velocity(inp, 1; micro = ns)

        # short-step limit: ΔN/N = ⟨v⟩_N Δt/Δz, ΔM/M = ⟨v⟩_M Δt/Δz
        Δz, Δt = 100.0, 1.0
        vN = inp_settling_velocity(inp, 1); vM = inp_settling_velocity(inp, 1; moment = 3)
        N0, M0 = inp_number(inp), inp_mass(inp)
        inp_sediment!(inp, Δz, Δt)
        @test (N0 - inp_number(inp)) / N0 ≈ vN * Δt / Δz rtol = 1e-4
        @test (M0 - inp_mass(inp)) / M0 ≈ vM * Δt / Δz rtol = 1e-3
        @info "INP settling speeds (σ_g = 2, r_g = 0.4 μm dust)" v_N_mm_s = 1e3vN v_M_mm_s = 1e3vM

        # a long settle: coarse tail gone, remainder finer, all positive
        inp = inp_population([lognormal_mode(:mineral_dust, 1.0, 0.4, 2.0)])
        inp_sediment!(inp, 100.0, 86400.0)
        @test inp_median_radius(inp, 1) < 0.4e-6
        @test inp_mass(inp) / M0 < inp_number(inp) / N0
        @test inp_number(inp) > 0
        # an absurd step empties the layer without going negative
        inp_sediment!(inp, 1e-6, 1e12)
        @test inp_number(inp) >= 0 && inp_mass(inp) >= 0
        @test maximum(abs, values(inp_budget_residual(inp))) < 1e-14
        # biological INP are lighter: slower at the same size
        both = inp_population([arg_mode(1e6, 1e-6, 1.6, 0.1; ρ_p = 2600.0),
                               arg_mode(1e6, 1e-6, 1.6, 0.1; ρ_p = 1200.0)],
                              [:mineral_dust, :biological])
        @test inp_settling_velocity(both, 2) < inp_settling_velocity(both, 1)
    end
end
