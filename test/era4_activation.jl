# Era IV, Phase 42 — aerosol-cloud interaction.
#
# Köhler theory, the Abdul-Razzak & Ghan (ARG) parameterisation and the
# explicit adiabatic parcel it approximates. The headline validation, ARG
# against the parcel over a range of updraughts and aerosol loadings, is
# validation/era4_activation.jl.

using Test, Printf
using Aeolus

@testset "Era IV — Aerosol-cloud interaction (Phase 42)" begin

    @testset "Köhler theory agrees with the Phase 39 contract" begin
        # Köhler lives in MoistThermodynamics so there is one implementation
        # in the package. This module's closed forms must track it; the only
        # permitted difference is the dilute approximation itself.
        worst = 0.0
        for rd in (2e-8, 5e-8, 1e-7, 2e-7), κ in (0.1, 0.6, 1.1, 1.28)
            exact = last(kohler_critical_point(rd, κ, 298.15))
            closed = critical_supersaturation(rd, κ)
            numerical = kohler_critical(rd, κ).s_crit
            worst = max(worst, abs(closed - exact) / exact)
            @test numerical ≈ exact rtol = 5e-3
            @test closed > 0
        end
        @test worst < 3e-2
        @info "closed form vs exact Köhler maximum" worst_rel_diff = worst
    end

    @testset "Köhler curve structure" begin
        rd, κ, T = 1e-7, 0.6, 293.15
        rc, sc = kohler_critical_point(rd, κ, T)
        @test rc > rd
        @test sc > 0
        # it really is a maximum
        @test kohler_supersaturation(rc * 0.7, rd, κ, T) < sc
        @test kohler_supersaturation(rc * 1.5, rd, κ, T) < sc
        # s_crit falls as r_d^(-3/2) and κ^(-1/2)
        s1 = last(kohler_critical_point(5e-8, κ, T))
        s2 = last(kohler_critical_point(1e-7, κ, T))
        @test s1 / s2 ≈ 2^1.5 rtol = 0.05
        sa = last(kohler_critical_point(rd, 0.3, T))
        sb = last(kohler_critical_point(rd, 1.2, T))
        @test sa / sb ≈ 2.0 rtol = 0.05
        # insoluble particles: pure Kelvin, no activation barrier maximum
        @test last(kohler_critical_point(rd, 0.0, T)) > sc
        # stable haze equilibrium exists below s_crit, not above
        req = kohler_equilibrium_radius(0.5 * sc, rd, κ, T)
        @test rd < req < rc
        @test kohler_supersaturation(req, rd, κ, T) ≈ 0.5 * sc rtol = 1e-6
        @test isnan(kohler_equilibrium_radius(1.5 * sc, rd, κ, T))
        @info "Köhler critical point" r_dry_nm = rd*1e9 r_crit_um = rc*1e6 s_crit_pct = 100sc
    end

    @testset "Hygroscopicity ordering" begin
        rd, T = 8e-8, 293.15
        s_bc   = last(kohler_critical_point(rd, 1e-6, T))
        s_so4  = last(kohler_critical_point(rd, 0.6,  T))
        s_salt = last(kohler_critical_point(rd, 1.1,  T))
        # sea salt activates most readily, black carbon least
        @test s_salt < s_so4 < s_bc
        @info "critical supersaturation by species (%)" sea_salt=100s_salt sulphate=100s_so4 black_carbon=100s_bc
    end

    @testset "explicit parcel: one case, S_max rises with the updraught" begin
        # (ARG against the parcel over w = 0.1-5 m/s, N_a = 100-1000 cm⁻³ is
        # the validation run validation/era4_activation.jl.)
        T, p = 283.0, 9.0e4
        modes = [arg_mode(100e6, 0.05e-6, 2.0, 0.6)]
        lo = parcel_activation(modes, 0.5, T, p)
        hi = parcel_activation(modes, 2.0, T, p)
        for pr in (lo, hi)
            @test pr.S_max > 0
            @test 0 < pr.fraction[1] <= 1
        end
        @test hi.S_max > lo.S_max
        @test hi.fraction[1] >= lo.fraction[1]
    end

    @testset "Activation monotonicity" begin
        T, p = 283.0, 9.0e4
        co = activation_coefficients(T, p)
        # more updraught ⇒ higher S_max and more activation
        fr = [arg_activation([arg_mode(500e6, 0.05e-6, 2.0, 0.6)], w, co).fraction[1]
              for w in (0.05, 0.1, 0.3, 1.0, 3.0)]
        @test all(diff(fr) .> 0)
        Sm = [arg_activation([arg_mode(500e6, 0.05e-6, 2.0, 0.6)], w, co).S_max
              for w in (0.05, 0.1, 0.3, 1.0, 3.0)]
        @test all(diff(Sm) .> 0)

        # more aerosol ⇒ MORE activated droplets but a SMALLER fraction.
        # That split is the physically interesting part: the particles
        # compete for the same vapour, so each gets less.
        res = [arg_activation([arg_mode(N * 1e6, 0.05e-6, 2.0, 0.6)], 0.5, co)
               for N in (50.0, 200.0, 1000.0, 5000.0)]
        Nact = [r.N_act_total for r in res]
        frac = [r.fraction[1]  for r in res]
        @test all(diff(Nact) .> 0)
        @test all(diff(frac) .< 0)
        Smax = [r.S_max for r in res]
        @test all(diff(Smax) .< 0)     # competition suppresses the peak
        @info "aerosol competition" N_activated=Nact fraction=frac S_max_pct=100 .* Smax
    end

    @testset "Twomey first indirect effect" begin
        # At fixed liquid water content, doubling droplet number shrinks the
        # effective radius by 2^(1/3) because r_eff ~ (LWC/N)^(1/3).
        T, p, co = 283.0, 9.0e4, activation_coefficients(283.0, 9.0e4)
        N1 = arg_activation([arg_mode(200e6, 0.05e-6, 2.0, 0.6)], 0.5, co).N_act_total
        N2 = arg_activation([arg_mode(400e6, 0.05e-6, 2.0, 0.6)], 0.5, co).N_act_total
        @test N2 > N1
        lwc = 3e-4                     # kg m^-3
        reff(N) = cbrt(3 * lwc / (4π * 1000.0 * N))
        r1, r2 = reff(N1), reff(N2)
        @test r2 < r1
        @test r1 / r2 ≈ cbrt(N2 / N1) rtol = 1e-10
        @info "Twomey" N1=N1 N2=N2 r_eff1_um=r1*1e6 r_eff2_um=r2*1e6 ratio=r1/r2
    end

    @testset "Prognostic aerosol conserves number" begin
        res = aerosol_reservoir([arg_mode(500e6, 0.05e-6, 2.0, 0.6),
                                 arg_mode(100e6, 0.2e-6,  1.8, 1.1)])
        n0 = aerosol_total_number(res)
        @test n0 > 0
        activate!(res, 0.003; T_K = 283.0)
        @test aerosol_total_number(res) ≈ n0 rtol = 1e-10
        @test activated_number(res) > 0
        @test interstitial_number(res) < n0
        regenerate!(res, 1.0)
        @test aerosol_total_number(res) ≈ n0 rtol = 1e-10
        @info "aerosol reservoir" total=n0 conserved_after_cycle=aerosol_total_number(res)
    end

    @testset "Multiple modes" begin
        co = activation_coefficients(283.0, 9.0e4)
        modes = [arg_mode(800e6, 0.02e-6, 1.6, 0.6),    # Aitken
                 arg_mode(200e6, 0.08e-6, 2.0, 0.6),    # accumulation
                 arg_mode( 10e6, 0.5e-6,  2.2, 1.1)]    # coarse sea salt
        r = arg_activation(modes, 0.5, co)
        @test length(r.fraction) == 3
        @test all(0 .<= r.fraction .<= 1)
        # coarse particles activate essentially completely, Aitken barely
        @test r.fraction[3] > r.fraction[2] > r.fraction[1]
        @test r.N_act_total ≈ sum(r.N_act) rtol = 1e-12
        @info "multi-mode activation" fractions=r.fraction S_max_pct=100r.S_max
    end
end
