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

    # Composite Simpson on a uniform grid — the tests' own quadrature,
    # sharing no code with pdf_expectation / sgs_ccn_fraction.
    function simpson(f, a, b, n)
        h = (b - a) / n
        acc = f(a) + f(b)
        for i in 1:n-1
            acc += (isodd(i) ? 4 : 2) * f(a + i * h)
        end
        return acc * h / 3
    end
    Nccn(S) = ccn_spectrum(modes, S; T_K = T_K)

    @testset "SGS activation: σ_S → 0 is Phase 42 exactly" begin
        for w in (0.05, 0.5, 3.0)
            r42 = arg_activation(modes, w, T_K, p)
            r45 = sgs_activation(modes, w, T_K, p, 0.0)
            @test r45.fraction == r42.fraction
            @test r45.N_act == r42.N_act
            @test r45.N_act_total == r42.N_act_total
            @test r45.mean.S_max == r42.S_max
            # and continuously so: σ_S = 1e-9 (vs S_max ~ 1e-3)
            r_eps = sgs_activation(modes, w, T_K, p, 1e-9)
            @test r_eps.N_act_total ≈ r42.N_act_total rtol = 1e-9
        end
        # a clear-sky SubgridCloud with no variance carries σ_S = 0
        sc = subgrid_cloud(288.0, 9e4, 8e-3)
        @test sc.σ_S == 0
        @test sgs_activation(modes, 0.5, T_K, p, sc).N_act_total ==
              arg_activation(modes, 0.5, T_K, p).N_act_total
        # no updraught, no activation — with or without fluctuations
        @test sgs_activation(modes, 0.0, T_K, p, 1e-3).N_act_total == 0
        @test sgs_activation(modes, -1.0, T_K, p, 1e-3).N_act_total == 0
    end

    @testset "SGS activation: quadrature against an independent integral" begin
        worst_ind = 0.0; worst_meth = 0.0
        for S̄ in (2e-4, 1e-3, 4e-3), σ in (0.1, 0.5, 1.0, 3.0) .* S̄
            ref = simpson(S -> Nccn(S) * exp(-(S - S̄)^2 / (2σ^2)) / (sqrt(2π) * σ),
                          max(0.0, S̄ - 12σ), S̄ + 12σ, 20_000)
            got = sgs_ccn_spectrum(modes, S̄, σ; T_K = T_K)
            worst_ind = max(worst_ind, abs(got - ref) / ref)
            a = sgs_ccn_spectrum(modes, S̄, σ; T_K = T_K, method = :radius)
            b = sgs_ccn_spectrum(modes, S̄, σ; T_K = T_K, method = :supersaturation)
            worst_meth = max(worst_meth, abs(a - b) / b)
        end
        @test worst_ind < 1e-9
        # Forcing the S-ordering where σ ≫ S̄ (its integrand then has the
        # sharp ln S rise at S → 0) is the least accurate case, 8e-8;
        # :auto picks the radius ordering there, hence worst_ind ~ 1e-11.
        @test worst_meth < 1e-6
        @info "SGS activation quadrature" max_rel_err_vs_simpson = worst_ind max_rel_diff_radius_vs_S_order = worst_meth
        # a double Gaussian with zero skewness is the Gaussian
        @test sgs_ccn_spectrum(modes, 1e-3, 5e-4; T_K = T_K,
                               pdf = DoubleGaussianSaturationPDF(0.0)) ≈
              sgs_ccn_spectrum(modes, 1e-3, 5e-4; T_K = T_K) rtol = 1e-13
        # bounds: an activated fraction is a probability, N_act ≤ N
        for σ in (1e-4, 1e-3, 1e-2)
            r = sgs_activation(modes, 0.5, T_K, p, σ)
            @test all(0 .<= r.fraction .<= 1)
            @test r.N_act_total <= sum(m.N for m in modes)
        end
    end

    @testset "SGS activation: Jensen — the sign follows the CCN curvature" begin
        # ⟨N_CCN(S)⟩ − N_CCN(S̄) = ½ N''(S̄) σ² + O(σ⁴). The coarse mode
        # (S_m = 0.033 %) makes the spectrum convex at S̄ = 0.02 %; from
        # 0.05 % upward it is concave. Sub-grid fluctuations must add
        # droplets in the first regime and remove them in the second.
        signs = Int[]; ratios = Float64[]
        for S̄ in (1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 4e-3, 8e-3)
            h = 1e-3 * S̄
            d2 = (Nccn(S̄ + h) - 2Nccn(S̄) + Nccn(S̄ - h)) / h^2
            σ = 0.05 * S̄
            ΔN = sgs_ccn_spectrum(modes, S̄, σ; T_K = T_K) - Nccn(S̄)
            @test sign(ΔN) == sign(d2)
            push!(signs, Int(sign(d2)))
            push!(ratios, ΔN / (0.5 * d2 * σ^2))
        end
        @test 1 in signs && -1 in signs          # both regimes exercised
        @test all(r -> 0.95 < r < 1.05, ratios)   # second-order Jensen term
        @info "Jensen: ΔN / (½ N'' σ²) at σ = 0.05 S̄" S̄ = (1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 4e-3, 8e-3) curvature_sign = string(signs) ratio = string(round.(ratios; digits = 4))
        # With the ARG S_max as the mean: at w = 0.5 m/s (S_max ≈ 0.21 %)
        # the spectrum is concave, so fluctuations reduce N_d.
        r0 = arg_activation(modes, 0.5, T_K, p)
        N_sgs = [sgs_activation(modes, 0.5, T_K, p, σ).N_act_total
                 for σ in (0.25e-3, 0.5e-3, 1e-3)]
        @test all(N_sgs .< r0.N_act_total)
        @test issorted(N_sgs; rev = true)
        @info "N_d at w = 0.5 m/s vs σ_S (0, 0.025, 0.05, 0.1 %)" S_max = r0.S_max N_cm3 = round.([r0.N_act_total; N_sgs] .* 1e-6; digits = 2)
    end

    @testset "SGS activation driven by a SubgridCloud" begin
        # σ_S from the issue #12 closure: the clear-air relative
        # supersaturation spread implied by q_t and θ_li variances.
        sc = subgrid_cloud(288.0, 9e4, 9.5e-3; var_qt = (4e-4)^2)
        @test sc.σ_S > 0
        r1 = sgs_activation(modes, 0.5, T_K, p, sc)
        r2 = sgs_activation(modes, 0.5, T_K, p, sc.σ_S)
        @test r1.N_act_total == r2.N_act_total
        @test r1.σ_S == sc.σ_S
    end

    # ---------------------------------------------------------------------
    # Condensational growth in a fluctuating supersaturation
    # ---------------------------------------------------------------------

    # Polydisperse 5 μm population in a supersaturated cell, as in the
    # Phase 41 narrowing test, with 1000 superdroplets.
    function growth_case(σ_S; τ = 10.0, seed = 7, steps = 50, dt = 0.5)
        V = 1.0
        sd = init_monodisperse_superdroplets(1000, 1e8, 5e-6, V;
                                             solute_mass = 1e-17)
        for k in eachindex(sd.radius)
            sd.radius[k] *= 1 + 0.4 * (k / 1000 - 0.5)
        end
        cell = EulerianCell(temperature = 283.0, q_vap = 9.6e-3,
                            density = 1.1, volume = V)
        fl = supersaturation_fluctuations(sd; σ_S = σ_S, τ = τ,
                                          rng = MersenneTwister(seed))
        W0 = cell_total_water(sd, cell)
        ε0 = relative_dispersion(sd)[2]
        for _ in 1:steps
            sgs_condensation_step!(sd, cell, dt, fl)
        end
        return (; sd, cell, ε0, ε = relative_dispersion(sd)[2],
                  dW = (cell_total_water(sd, cell) - W0) / W0)
    end

    @testset "SGS growth: σ_S = 0 is Phase 41 condensation_step! bit for bit" begin
        a = growth_case(0.0)
        # the same population through the Phase 41 routine
        V = 1.0
        sd = init_monodisperse_superdroplets(1000, 1e8, 5e-6, V;
                                             solute_mass = 1e-17)
        for k in eachindex(sd.radius)
            sd.radius[k] *= 1 + 0.4 * (k / 1000 - 0.5)
        end
        cell = EulerianCell(temperature = 283.0, q_vap = 9.6e-3,
                            density = 1.1, volume = V)
        for _ in 1:50
            condensation_step!(sd, cell, 0.5)
        end
        @test a.sd.radius == sd.radius
        @test a.cell.temperature == cell.temperature
        @test a.cell.q_vap == cell.q_vap
        # and pure condensation narrows the spectrum (Phase 41 result)
        @test a.ε < a.ε0
    end

    @testset "Ornstein-Uhlenbeck supersaturation" begin
        # stationary unit variance, lag-one correlation e^{-dt/τ}
        fl = supersaturation_fluctuations(200_000; σ_S = 1e-3, τ = 4.0,
                                          rng = MersenneTwister(3))
        ξ0 = copy(fl.ξ)
        advance_fluctuations!(fl, 2.0)
        @test var(fl.ξ) ≈ 1 atol = 0.01
        @test cor(ξ0, fl.ξ) ≈ exp(-0.5) atol = 0.01
        @test mean(local_supersaturation(fl, 2e-3)) ≈ 2e-3 atol = 1e-5
        # frozen sub-volumes: τ = Inf never changes ξ
        fz = supersaturation_fluctuations(10; σ_S = 1e-3)
        ξz = copy(fz.ξ); advance_fluctuations!(fz, 100.0)
        @test fz.ξ == ξz
        # the analytic integral variance and its limits
        @test supersaturation_integral_variance(1e-3, Inf, 30.0) ≈ (1e-3 * 30)^2
        @test supersaturation_integral_variance(1e-3, 1e6, 30.0) ≈ (1e-3 * 30)^2 rtol = 1e-4
        @test supersaturation_integral_variance(1e-3, 0.01, 1000.0) ≈
              2 * (1e-3)^2 * 0.01 * 1000 rtol = 1e-4
    end

    @testset "SGS growth: Var(r²) against the analytic OU result" begin
        # Uncoupled growth at fixed mean S̄ from a monodisperse 10 μm
        # population without solute: r² gains 2G∫S dt, so
        # Var(r²) = (2G)² · 2σ²τ[t − τ(1−e^{−t/τ})] (Sardina et al. 2015).
        # 4000 droplets: sampling s.e. of a variance ≈ √(2/4000) = 2.2 %.
        T, pr = 283.0, 9e4
        G = 1 / (fk_thermal(T) + fd_vapor(T, pr))
        S̄, σ, t = 5e-3, 2e-3, 60.0
        ratios = Float64[]
        for τ in (5.0, Inf)
            sd = init_monodisperse_superdroplets(4000, 1e8, 10e-6, 1.0)
            fl = supersaturation_fluctuations(sd; σ_S = σ, τ = τ,
                                              rng = MersenneTwister(1))
            h = isfinite(τ) ? τ / 10 : 1.0
            for _ in 1:round(Int, t / h)
                sgs_condensation_growth!(sd, fl, S̄, T, pr, h)
            end
            pred = (2G)^2 * supersaturation_integral_variance(σ, τ, t)
            push!(ratios, var(sd.radius .^ 2) / pred)
        end
        @test all(r -> abs(r - 1) < 0.07, ratios)     # ≈ 3 s.e.
        # correlation time matters: frozen sub-volumes spread r² 6.5x more
        @test supersaturation_integral_variance(σ, Inf, t) /
              supersaturation_integral_variance(σ, 5.0, t) ≈ 6.545 rtol = 1e-3
        @info "Var(r²) measured / analytic OU (τ = 5 s, τ = ∞)" ratios
    end

    @testset "SGS growth: dispersion rises monotonically with σ_S, water conserved" begin
        σs = (0.0, 5e-4, 1e-3, 2e-3, 4e-3)
        runs = [growth_case(σ) for σ in σs]
        εs = [r.ε for r in runs]
        @test issorted(εs) && allunique(εs)            # strictly increasing
        @test all(abs(r.dW) < 1e-14 for r in runs)      # total water, round-off
        # common random numbers: every run starts from the same spectrum
        @test all(r.ε0 == runs[1].ε0 for r in runs)
        # the broadening is the fluctuations', not a change of the mean
        r̄ = [mean_radius(r.sd) for r in runs]
        @test maximum(r̄) / minimum(r̄) - 1 < 0.005
        @info "relative dispersion after 25 s vs σ_S" σ_S = σs ε0 = runs[1].ε0 ε = string(round.(εs; digits = 4)) max_dW_over_W = maximum(abs(r.dW) for r in runs)
        # Over a longer run (100 s ≫ τ = 10 s and ≫ τ_phase ≈ 6 s) the mean
        # supersaturation has been consumed (S̄ ≈ 1e-4), so pure
        # condensation has stopped narrowing — ε is frozen to within 1 %
        # (it drifts up by 0.6 % through the curvature/solute terms) — but
        # the fluctuations keep broadening: σ_{r²} ∝ √t for t ≫ τ.
        long0 = growth_case(0.0; steps = 200)
        long2 = growth_case(2e-3; steps = 200)
        @test abs(long0.ε / εs[1] - 1) < 0.01
        @test long2.ε > 1.5 * εs[4]
        @test abs(long2.dW) < 1e-14
        @info "100 s run" ε_σ0 = long0.ε ε_σ2e_3 = long2.ε
    end

    @testset "SGS growth: entrainment-driven variance broadens the spectrum" begin
        # Sub-grid (q_t, θ_li) variance at a cloud edge, mapped to σ_S by
        # the issue #12 closure. Entrained air above a cloud is warm AND
        # dry (q_t'θ_li' < 0), which adds to the saturation-deficit
        # variance; surface thermals are moist AND warm (> 0), which
        # partly cancels. Same variances, opposite correlation.
        σq, σθ = 3e-5, 0.03
        σ_S = Float64[]
        for ρ in (-0.8, 0.0, 0.8)
            sc = subgrid_cloud(288.0, 9e4, 9.8e-3; var_qt = σq^2, var_θ = σθ^2,
                               cov_qθ = ρ * σq * σθ)
            push!(σ_S, sc.σ_S)
        end
        @test issorted(σ_S; rev = true)
        εs = [growth_case(σ).ε for σ in σ_S]
        dWs = [growth_case(σ).dW for σ in σ_S]
        @test issorted(εs; rev = true)
        @test all(abs.(dWs) .< 1e-14)
        @info "entrainment (ρ_qθ = -0.8, 0, +0.8)" σ_S_percent = round.(100 .* σ_S; digits = 3) ε = round.(εs; digits = 4)
    end
end
