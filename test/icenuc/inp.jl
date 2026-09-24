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
end
