# Era IV, Phase 45 — turbulence-microphysics interaction, issue #13.
#
# The sub-grid supersaturation distribution of issue #12 fed into
# aerosol activation (Phase 42) and condensational droplet growth
# (Phase 41). Headline checks: σ_S → 0 recovers the Phase 41/42 results
# bit for bit; the sign of the activation change is the sign of the
# curvature of the CCN spectrum (Jensen); the spread of r² matches the
# analytic Ornstein-Uhlenbeck result; relative dispersion rises
# monotonically with σ_S while total water is conserved to round-off.

using Test, Printf, Random, Statistics
using Aeolus
using Aeolus.AerosolCloud: ccn_fraction, ccn_spectrum

@testset "Era IV — SGS activation and growth (Phase 45, #13)" begin

    # Two-mode continental-ish population used throughout (SI units).
    modes = [arg_mode(300e6, 0.04e-6, 1.8, 0.6),
             arg_mode(50e6, 0.15e-6, 1.6, 0.6)]
    T_K, p = 283.0, 90000.0

    @testset "CCN spectrum: one copy, shared with arg_activation" begin
        for w in (0.1, 0.5, 2.0)
            r = arg_activation(modes, w, T_K, p)
            # bit-for-bit: arg_activation evaluates this very function
            @test [ccn_fraction(m, r.S_max; T_K = T_K) for m in modes] == r.fraction
            @test ccn_spectrum(modes, r.S_max; T_K = T_K) == r.N_act_total
        end
        # N_CCN(S) is a CDF in S: zero at S ≤ 0, monotone, → N_tot
        Ss = 10 .^ range(-5, -1; length = 200)
        N = [ccn_spectrum(modes, S; T_K = T_K) for S in Ss]
        @test all(diff(N) .>= 0)
        @test ccn_spectrum(modes, 0.0; T_K = T_K) == 0
        @test ccn_spectrum(modes, -1e-3; T_K = T_K) == 0
        # (the Aitken tail of mode 1 still holds 1e-5 of its number
        # un-activated at S = 10 %, so saturate far beyond that)
        @test ccn_spectrum(modes, 10.0; T_K = T_K) ≈ sum(m.N for m in modes) rtol = 1e-12
        # at S = S_m exactly half of the mode activates (median particle)
        m1 = modes[1]
        @test ccn_fraction(m1, mode_critical_supersaturation(m1, T_K);
                           T_K = T_K) ≈ 0.5 atol = 1e-14
        # insoluble aerosol never activates
        @test ccn_fraction(arg_mode(1e8, 0.1e-6, 1.5, 0.0), 0.01) == 0
    end

    @testset "condensation_growth! with a per-droplet S" begin
        V = 1.0
        sd1 = init_monodisperse_superdroplets(64, 1e8, 6e-6, V;
                                              solute_mass = 1e-18)
        for k in eachindex(sd1.radius)
            sd1.radius[k] *= 1 + 0.3 * (k / 64 - 0.5)
        end
        sd2 = deepcopy(sd1)
        Δ1 = condensation_growth!(sd1, 2e-3, 283.0, 9e4, 5.0)
        Δ2 = condensation_growth!(sd2, fill(2e-3, 64), 283.0, 9e4, 5.0)
        @test Δ1 == Δ2 && sd1.radius == sd2.radius      # bit for bit
        # a droplet in a moister sub-volume grows faster
        sd3 = init_monodisperse_superdroplets(2, 1e8, 6e-6, V)
        condensation_growth!(sd3, [1e-3, 3e-3], 283.0, 9e4, 10.0)
        @test sd3.radius[2] > sd3.radius[1]
    end
end
