# P3 part-file tests: four-regime mass-dimension relation, predicted rime
# density, slope solve, fall speed (src/p3/massdim.jl, issue #2).
#
# The point of P3 is that ice properties vary *continuously* with rime
# fraction and rime density: no category jump from snow to graupel. The
# sweeps below measure the largest step change in mass, bulk density and
# fall speed between neighbouring (F_rim, ρ_rim) samples and check that it
# shrinks in proportion to the sample spacing (continuity), rather than
# staying finite (a jump).

@testset "P3 mass-dimension and rime density" begin
    relerr(x, ref) = abs(x - ref) / abs(ref)
    p = P3_DEFAULT_ICE

    @testset "Thresholds solved from F_rim, ρ_rim" begin
        # D_th: sphere of solid ice has the same mass as an aggregate.
        md0 = p3_mass_dimension(0.0, 400.0)
        @test π / 6 * p.ρ_ice * md0.D_th^3 ≈ p.α_va * md0.D_th^p.β_va rtol = 1e-13
        @printf("    D_th = %.3f μm\n", 1e6 * md0.D_th)
        @test 60e-6 < md0.D_th < 70e-6     # MM15: D_th ≈ 66 μm for ρ_ice = 917

        # Closed-form ρ_g against the MM15 / P3 reference fixed-point
        # iteration of the coupled (ρ_g, D_gr, D_cr, ρ_d) system, run in
        # 256-bit BigFloat: in Float64 the reference itself loses digits to
        # the D_cr^{β-2} - D_gr^{β-2} cancellation at small F_rim.
        maxfp = 0.0
        for F in (0.01, 0.1, 0.4, 0.8, 0.97), ρr in (50.0, 250.0, 600.0, 900.0)
            ρg = setprecision(256) do
                α, β, Fb, ρb = big(p.α_va), big(p.β_va), big(F), big(ρr)
                ρg = ρb
                # contraction factor ≈ (1-F)·g ≈ 1 - 3F/2: slow for small F
                for _ in 1:20_000
                    Dcr = (6α / (big(π) * ρg * (1 - Fb)))^(1 / (3 - β))
                    Dgr = (6α / (big(π) * ρg))^(1 / (3 - β))
                    ρd = 6α * (Dcr^(β - 2) - Dgr^(β - 2)) /
                         (big(π) * (β - 2) * (Dcr - Dgr))
                    ρnew = Fb * ρb + (1 - Fb) * ρd
                    done = abs(ρnew - ρg) < big(1e-40) * ρg
                    ρg = ρnew
                    done && break
                end
                Float64(ρg)
            end
            md = p3_mass_dimension(F, ρr)
            maxfp = max(maxfp, relerr(md.ρ_g, ρg))
            # the defining equations hold at the returned state
            @test md.ρ_g ≈ F * ρr + (1 - F) * md.ρ_d rtol = 1e-13
            @test π / 6 * md.ρ_g * md.D_gr^3 ≈ p.α_va * md.D_gr^p.β_va rtol = 1e-12
            @test π / 6 * md.ρ_g * md.D_cr^3 ≈
                  p.α_va / (1 - F) * md.D_cr^p.β_va rtol = 1e-12
            # ρ_d = mean aggregate density over [D_gr, D_cr], by quadrature
            n = 4000
            h = (md.D_cr - md.D_gr) / n
            ρmean = sum(6p.α_va * (md.D_gr + (i - 0.5) * h)^(p.β_va - 3) / π
                        for i in 1:n) / n
            @test md.ρ_d ≈ ρmean rtol = 1e-6
        end
        @printf("    ρ_g closed form vs fixed-point iteration: max rel diff %.2e\n", maxfp)
        @test maxfp < 1e-13
    end

    @testset "Threshold ordering and limits" begin
        for F in range(0.0, 1.0; length = 101), ρr in range(p.ρ_rim_min, p.ρ_rim_max; length = 18)
            md = p3_mass_dimension(F, ρr)
            @test md.D_th <= md.D_gr <= md.D_cr
            @test md.ρ_d <= md.ρ_g <= ρr + 1e-9 * ρr
            @test md.ρ_g <= p.ρ_ice
        end
        # F → 0⁺: ρ_g → 2ρ_rim/3 (analytic limit, any β), D_cr → D_gr
        for ρr in (50.0, 400.0, 900.0)
            md = p3_mass_dimension(1e-12, ρr)
            @test md.ρ_g ≈ 2ρr / 3 rtol = 1e-11
            @test md.D_cr ≈ md.D_gr * (1 + 1e-12 / (3 - p.β_va)) rtol = 1e-15
        end
        # The limit must hold all the way down to subnormal F_rim. Review
        # regression: forming L² = log(D_cr/D_gr)² underflowed, giving
        # ρ_g = -Inf for F_rim ≲ 1e-200 and a 1.1e-5 error at 1e-160.
        for F in (1e-160, 1e-200, 1e-300, nextfloat(0.0)), ρr in (50.0, 900.0)
            md = p3_mass_dimension(F, ρr)
            @test md.ρ_g ≈ 2ρr / 3 rtol = 1e-14
            @test md.ρ_d ≈ md.ρ_g rtol = 1e-14
            @test md.D_th <= md.D_gr <= md.D_cr
        end
        # F = 1: all graupel beyond D_gr, at the rime density
        md = p3_mass_dimension(1.0, 500.0)
        @test md.ρ_g == 500.0 && isinf(md.D_cr)
        @test p3_mass_dimension(1 - 1e-12, 500.0).ρ_g ≈ 500.0 rtol = 1e-9
        # rime density is bounded
        @test p3_mass_dimension(0.5, 5.0).ρ_rim == p.ρ_rim_min
        @test p3_mass_dimension(0.5, 2000.0).ρ_rim == p.ρ_rim_max
        @test_throws DomainError p3_mass_dimension(1.2, 400.0)
        @test_throws DomainError p3_mass_dimension(-0.1, 400.0)
    end

    @testset "F_rim = 0 reduces to unrimed aggregates" begin
        md = p3_mass_dimension(0.0, 400.0)
        @test isinf(md.D_gr) && isinf(md.D_cr)
        for D in 10.0 .^ range(-6, -1.5; length = 200)
            if D <= md.D_th
                @test p3_regime(D, md) == 1
                @test p3_particle_mass(D, md) ≈ π / 6 * p.ρ_ice * D^3 rtol = 1e-14
                @test p3_particle_area(D, md) ≈ π / 4 * D^2 rtol = 1e-14
            else
                @test p3_regime(D, md) == 2
                @test p3_particle_mass(D, md) ≈ p.α_va * D^p.β_va rtol = 1e-14
                @test p3_particle_area(D, md) ≈ p.γ_va * D^p.σ_va rtol = 1e-14
            end
        end
        # Independent of the (irrelevant) rime density when unrimed
        for D in (1e-5, 1e-4, 1e-3, 1e-2)
            @test p3_particle_mass(D, p3_mass_dimension(0.0, 50.0)) ==
                  p3_particle_mass(D, p3_mass_dimension(0.0, 900.0))
        end
        # Brown & Francis aggregate density falls off as D^{β-3}:
        # a 1 cm aggregate is ~ 4 kg m⁻³ effective density, far below ice
        ρeff = p3_particle_mass(1e-2, md) / (π / 6 * 1e-6)
        @printf("    effective density of a 1 cm unrimed aggregate: %.2f kg m⁻³\n", ρeff)
        @test 1 < ρeff < 20
    end

    @testset "Mass continuity across regime boundaries" begin
        maxjump = 0.0
        for F in (0.05, 0.3, 0.6, 0.9), ρr in (80.0, 400.0, 850.0)
            md = p3_mass_dimension(F, ρr)
            @test [p3_regime(D, md) for D in (0.5md.D_th, 0.5(md.D_th + md.D_gr),
                                              0.5(md.D_gr + md.D_cr), 2md.D_cr)] == [1, 2, 3, 4]
            for Db in (md.D_th, md.D_gr, md.D_cr)
                below = p3_particle_mass(Db * (1 - 1e-12), md)
                above = p3_particle_mass(Db * (1 + 1e-12), md)
                maxjump = max(maxjump, relerr(above, below))
            end
            # mass is monotone in D
            Ds = 10.0 .^ range(-6, -1; length = 2000)
            ms = [p3_particle_mass(D, md) for D in Ds]
            @test all(diff(ms) .> 0)
        end
        @printf("    max relative mass jump across D_th, D_gr, D_cr (±1e-12 D): %.2e\n", maxjump)
        @test maxjump < 1e-11
    end

    @testset "Prognostic rime mass and volume" begin
        # F and ρ from the prognostics
        F, ρ, B = p3_rime_state(1e-3, 4e-4, 4e-4 / 300)
        @test F ≈ 0.4 rtol = 1e-15
        @test ρ ≈ 300 rtol = 1e-14
        @test B == 4e-4 / 300
        # clamping writes back a consistent rime volume
        F, ρ, B = p3_rime_state(1e-3, 4e-4, 4e-4 / 2000)
        @test ρ == p.ρ_rim_max
        @test B ≈ 4e-4 / p.ρ_rim_max rtol = 1e-15
        F, ρ, B = p3_rime_state(1e-3, 4e-4, 4e-4 / 10)
        @test ρ == p.ρ_rim_min
        @test B ≈ 4e-4 / p.ρ_rim_min rtol = 1e-15
        # no rime / no ice
        @test p3_rime_state(1e-3, 0.0, 0.0)[1] == 0
        @test p3_rime_state(0.0, 0.0, 0.0)[1] == 0
        @test p3_rime_state(1e-3, 2e-3, 2e-3 / 400)[1] == 1    # F clamped to 1

        # Density emerges from carrying volume: accreting rime of density
        # 100 then 900 gives the mass-weighted harmonic mean, with no rule
        # for combining densities anywhere in the code.
        q_rim, B_rim = 0.0, 0.0
        for (dq, ρnew) in ((3e-4, 100.0), (1e-4, 900.0))
            q_rim += dq
            B_rim += dq / ρnew
        end
        _, ρ, _ = p3_rime_state(1e-3, q_rim, B_rim)
        @test ρ ≈ 4e-4 / (3e-4 / 100 + 1e-4 / 900) rtol = 1e-14
        @test ρ ≈ 400 * 900 / 2800 rtol = 1e-14         # = 128.57 kg m⁻³
        # and the mass-dimension state built from prognostics matches
        md1 = p3_mass_dimension(1e-3, q_rim, B_rim)
        md2 = p3_mass_dimension(0.4, ρ)
        @test md1.ρ_g ≈ md2.ρ_g rtol = 1e-14
        @test md1.D_cr ≈ md2.D_cr rtol = 1e-14
    end
end

@testset "P3 size distribution: slope solve and bulk density" begin
    relerr(x, ref) = abs(x - ref) / abs(ref)
    p = P3_DEFAULT_ICE
    # Largest relative step between neighbouring samples of a sweep
    maxstep(v) = maximum(abs.(diff(v)) ./ abs.(v[1:end-1]))

    @testset "Mean mass vs direct quadrature of m(D) N(D)" begin
        # composite Simpson on a log-spaced grid split at the thresholds
        function quad_mean_mass(μ, λ, md)
            Dmax = 80 / λ
            edges = [0.0; sort(filter(x -> 0 < x < Dmax,
                                       [md.D_th, md.D_gr, md.D_cr])); Dmax]
            num = 0.0; den = 0.0
            for k in 1:(length(edges) - 1)
                a, b = edges[k], edges[k + 1]
                n = 4000
                h = (b - a) / n
                for i in 0:n
                    D = a + i * h
                    w = (i == 0 || i == n) ? 1 : (isodd(i) ? 4 : 2)
                    nd = D^μ * exp(-λ * D)
                    num += w * h / 3 * p3_particle_mass(D, md) * nd
                    den += w * h / 3 * nd
                end
            end
            return num / den
        end
        maxe = 0.0
        for (F, ρr) in ((0.0, 400.0), (0.3, 150.0), (0.7, 900.0), (1.0, 500.0)),
            μ in (0.0, 2.0), λ in (1e3, 1e4, 1e5)
            md = p3_mass_dimension(F, ρr)
            maxe = max(maxe, relerr(p3_mean_mass(μ, λ, md), quad_mean_mass(μ, λ, md)))
        end
        @printf("    mean mass (truncated moments) vs Simpson quadrature: max rel err %.2e\n", maxe)
        @test maxe < 1e-6
        # q_i = N_i · mean mass
        md = p3_mass_dimension(0.4, 300.0)
        N0 = gamma_psd_intercept(1e5, 1.0, 3e3)
        @test p3_ice_mass(N0, 1.0, 3e3, md) ≈ 1e5 * p3_mean_mass(1.0, 3e3, md) rtol = 1e-13
    end

    @testset "Analytic limits of bulk density" begin
        # Tiny particles (λ D_th ≫ 1): all solid-ice spheres, ρ_bulk → ρ_ice
        md = p3_mass_dimension(0.5, 400.0)
        @test p3_bulk_density(0, 1e7, md) ≈ p.ρ_ice rtol = 1e-12
        # F_rim = 1 with λ small enough that nearly all volume is graupel:
        # ρ_bulk → ρ_g = ρ_rim
        md = p3_mass_dimension(1.0, 500.0)
        ρb = p3_bulk_density(2, 50.0, md)
        @printf("    ρ_bulk of pure graupel (ρ_rim = 500, λ = 50 m⁻¹): %.4f\n", ρb)
        @test ρb ≈ 500 rtol = 1e-4
        # unrimed, exponential PSD fully in the aggregate regime:
        # ρ_bulk = 6α Γ(β+1)λ^{3-β}/(π Γ(4)) as D_th λ → 0
        md = p3_mass_dimension(0.0, 400.0)
        λ = 20.0
        ref = 6p.α_va * gamma_lanczos(p.β_va + 1) * λ^(3 - p.β_va) / (π * 6)
        @test relerr(p3_bulk_density(0, λ, md), ref) < 1e-6
    end

    @testset "Slope solve round trip" begin
        maxe = 0.0
        n = 0
        for F in (0.0, 0.2, 0.6, 0.95, 1.0), ρr in (50.0, 400.0, 900.0),
            μ in (0.0, 1.5, 4.0), λ in (3e2, 2e3, 1e4, 5e4, 3e5)
            md = p3_mass_dimension(F, ρr)
            N0 = gamma_psd_intercept(1e5, μ, λ)
            q = p3_ice_mass(N0, μ, λ, md)
            N0b, μb, λb = p3_size_distribution(q, 1e5, md; μ = μ)
            maxe = max(maxe, relerr(λb, λ))
            @test relerr(N0b, N0) < 1e-11
            # and q/N is recovered from the solved PSD
            @test p3_ice_mass(N0b, μb, λb, md) ≈ q rtol = 1e-12
            n += 1
        end
        @printf("    λ round trip over %d (F_rim, ρ_rim, μ, λ) cases: max rel err %.2e\n", n, maxe)
        @test maxe < 1e-12
        # out-of-range mean mass clamps to the bracket
        md = p3_mass_dimension(0.0, 400.0)
        @test p3_slope(1.0, 1.0, md) == 1.0          # 1 kg particles
        @test p3_slope(1e-30, 1.0, md) ≈ 1e8          # far below a 1 nm sphere
        @test_throws DomainError p3_slope(0.0, 1e3, md)
    end

    @testset "Continuity of bulk properties across the F_rim, ρ_rim sweep" begin
        # (a) fixed PSD (λ): mean mass and bulk density vs F_rim
        # (b) fixed q_i/N_i: the solved λ and bulk density vs F_rim
        # Continuity ⇔ the largest step shrinks ∝ the sample spacing.
        steps = Dict{Int, Float64}()
        for nF in (1001, 10001)
            Fs = range(0.0, 1.0; length = nF)
            worst = 0.0
            for ρr in (50.0, 400.0, 900.0)
                for λ in (2e3, 1e4)
                    mm = [p3_mean_mass(0, λ, p3_mass_dimension(F, ρr)) for F in Fs]
                    rb = [p3_bulk_density(0, λ, p3_mass_dimension(F, ρr)) for F in Fs]
                    worst = max(worst, maxstep(mm), maxstep(rb))
                end
                ls = [p3_slope(1e-4, 1e4, p3_mass_dimension(F, ρr)) for F in Fs]
                rb = [p3_bulk_density(0, λ, p3_mass_dimension(F, ρr))
                      for (F, λ) in zip(Fs, ls)]
                worst = max(worst, maxstep(ls), maxstep(rb))
            end
            steps[nF] = worst
        end
        @printf("    F_rim sweep, max step: ΔF=1e-3 → %.3e, ΔF=1e-4 → %.3e (ratio %.2f)\n",
                steps[1001], steps[10001], steps[1001] / steps[10001])
        @test steps[10001] < 3e-3
        @test 8 < steps[1001] / steps[10001] < 12      # linear in the step: no jump

        # ρ_rim sweep at fixed F_rim
        rsteps = Float64[]
        for nρ in (851, 8501)
            ρs = range(p.ρ_rim_min, p.ρ_rim_max; length = nρ)
            worst = 0.0
            for F in (0.1, 0.5, 0.9, 1.0)
                ls = [p3_slope(1e-4, 1e4, p3_mass_dimension(F, ρ)) for ρ in ρs]
                rb = [p3_bulk_density(0, λ, p3_mass_dimension(F, ρ)) for (ρ, λ) in zip(ρs, ls)]
                worst = max(worst, maxstep(ls), maxstep(rb))
            end
            push!(rsteps, worst)
        end
        @printf("    ρ_rim sweep, max step: Δρ=1 → %.3e, Δρ=0.1 → %.3e\n", rsteps...)
        @test 8 < rsteps[1] / rsteps[2] < 12

        # One-sided limit at F_rim = 0⁺ equals the unrimed value
        for ρr in (50.0, 900.0), λ in (1e3, 1e4)
            m0 = p3_mean_mass(0, λ, p3_mass_dimension(0.0, ρr))
            m1 = p3_mean_mass(0, λ, p3_mass_dimension(1e-10, ρr))
            @test relerr(m1, m0) < 1e-9
        end

        # Riming at fixed q_i/N_i makes particles denser and smaller:
        # λ and ρ_bulk increase monotonically with F_rim and with ρ_rim.
        for ρr in (100.0, 400.0, 900.0)
            ls = [p3_slope(1e-4, 1e4, p3_mass_dimension(F, ρr)) for F in 0:0.05:1]
            @test all(diff(ls) .> 0)
        end
        ls = [p3_slope(1e-4, 1e4, p3_mass_dimension(0.5, ρ)) for ρ in 50:50:900]
        @test all(diff(ls) .> 0)
    end
end

@testset "P3 fall speed" begin
    relerr(x, ref) = abs(x - ref) / abs(ref)
    maxstep(v) = maximum(abs.(diff(v)) ./ abs.(v[1:end-1]))
    p = P3_DEFAULT_ICE
    P3 = Aeolus.P3Microphysics
    ρs, η = 1.2, p3_air_viscosity(283.15)       # near-surface air

    @testset "Air viscosity (Sutherland)" begin
        @test p3_air_viscosity(273.15) ≈ 1.716e-5 rtol = 1e-3
        @test p3_air_viscosity(300.0) ≈ 1.846e-5 rtol = 2e-3   # tabulated dry air
    end

    @testset "Single-particle limits" begin
        md0 = p3_mass_dimension(0.0, 400.0)
        # Small solid-ice spheres against Stokes' law. MH05's boundary-
        # layer form gives Re = X/(δ₀²C₀) ≈ X/20.4 rather than X/24 as X → 0,
        # partly offset by the a₀ correction; measured deviation quoted.
        maxdev = 0.0
        for D in (10e-6, 20e-6, 30e-6, 50e-6)
            vs = p.ρ_ice * 9.80665 * D^2 / (18η)
            maxdev = max(maxdev, relerr(p3_fall_speed(D, md0; ρ_air = ρs, η_air = η), vs))
        end
        @printf("    ice spheres 10-50 μm vs Stokes: max rel dev %.3f\n", maxdev)
        @test maxdev < 0.10
        # Unrimed aggregates against Locatelli & Hobbs (1974),
        # V = 0.8 D^0.16 (D in mm), aggregates of unrimed side planes etc.
        maxdev = 0.0
        for D in (0.5e-3, 1e-3, 2e-3, 5e-3, 1e-2)
            v = p3_fall_speed(D, md0; ρ_air = ρs, η_air = η)
            maxdev = max(maxdev, relerr(v, 0.8 * (1e3D)^0.16))
        end
        @printf("    unrimed aggregates 0.5-10 mm vs Locatelli & Hobbs (1974): max rel dev %.3f\n", maxdev)
        @test maxdev < 0.15
        # graupel falls faster than an aggregate of the same size, and
        # denser rime falls faster
        for D in (0.5e-3, 1e-3, 2e-3, 4e-3)
            va = p3_fall_speed(D, md0; ρ_air = ρs, η_air = η)
            vg = [p3_fall_speed(D, p3_mass_dimension(1.0, ρ); ρ_air = ρs, η_air = η)
                  for ρ in (100.0, 400.0, 900.0)]
            @test all(diff(vg) .> 0)
            @test vg[2] > 1.4va
        end
        # lower air density (aloft) → faster fall
        @test p3_fall_speed(2e-3, md0; ρ_air = 0.5, η_air = η) >
              p3_fall_speed(2e-3, md0; ρ_air = 1.2, η_air = η)
        # zero / tiny sizes
        @test p3_fall_speed(0.0, md0) == 0
        @test p3_fall_speed(5e-8, md0) >= 0
    end

    @testset "Bulk fall speeds" begin
        # Quadrature check: the same composite rule applied to m(D)
        # reproduces the analytic truncated-moment mean mass.
        maxe = 0.0
        for (F, ρr) in ((0.0, 400.0), (0.5, 400.0), (0.9, 900.0), (1.0, 100.0)),
            μ in (0.0, 2.5), λ in (1e3, 1e4, 1e5)
            md = p3_mass_dimension(F, ρr)
            mq, one_ = P3._p3_psd_expectation(D -> (p3_particle_mass(D, md), 1.0),
                                               μ, λ, md)
            maxe = max(maxe, relerr(mq, p3_mean_mass(μ, λ, md)), abs(one_ - 1))
        end
        @printf("    PSD quadrature vs analytic mean mass / normalisation: max rel err %.2e\n", maxe)
        @test maxe < 1e-12

        # V_M ≥ V_N (bigger particles carry the mass and fall faster)
        for F in (0.0, 0.5, 1.0), λ in (1e3, 1e4)
            VN, VM = p3_bulk_fall_speed(0, λ, p3_mass_dimension(F, 400.0))
            @test VM > VN > 0
        end

        # Continuity over the F_rim sweep at fixed q_i/N_i
        steps = Float64[]
        for nF in (401, 4001)
            Fs = range(0.0, 1.0; length = nF)
            VNs = Float64[]; VMs = Float64[]
            for F in Fs
                md = p3_mass_dimension(F, 400.0)
                VN, VM = p3_bulk_fall_speed(0, p3_slope(1e-4, 1e3, md), md)
                push!(VNs, VN); push!(VMs, VM)
            end
            push!(steps, max(maxstep(VNs), maxstep(VMs)))
            if nF == 401
                @printf("    q/N = 1e-7 kg: V_M(F=0) = %.3f, V_M(F=1) = %.3f m/s; V_N %.3f → %.3f\n",
                        VMs[1], VMs[end], VNs[1], VNs[end])
                # riming converts slow snow into fast graupel
                @test VMs[end] > 2VMs[1]
                # F_rim = 0⁺ joins the unrimed value
                md = p3_mass_dimension(1e-9, 400.0)
                @test relerr(p3_bulk_fall_speed(0, p3_slope(1e-4, 1e3, md), md)[2], VMs[1]) < 1e-7
            end
        end
        @printf("    V_N, V_M F_rim sweep max step: ΔF=2.5e-3 → %.3e, ΔF=2.5e-4 → %.3e\n", steps...)
        @test steps[2] < 5e-3
        @test 8 < steps[1] / steps[2] < 12
    end

    @testset "Generic float type" begin
        # Review regression: a Float64 D with a Float32 state threw a
        # MethodError in the fall-speed kernel. Float32 must agree with
        # Float64 to single-precision rounding and stay Float32.
        p32 = P3IceParameters{Float32}()
        m32 = p3_mass_dimension(0.5f0, 400f0, p32)
        m64 = p3_mass_dimension(0.5, 400.0)
        @test m32.ρ_g ≈ m64.ρ_g rtol = 1e-6
        for D in (30e-6, 1e-3), Dx in (D, Float32(D))
            v = p3_fall_speed(Dx, m32)
            @test v isa Float32
            @test v ≈ p3_fall_speed(D, m64) rtol = 1e-5
        end
        VN32, VM32 = p3_bulk_fall_speed(0f0, 1f3, m32)
        VN, VM = p3_bulk_fall_speed(0, 1e3, m64)
        @test VN32 isa Float32 && VM32 isa Float32
        @test VN32 ≈ VN rtol = 1e-5
        @test VM32 ≈ VM rtol = 1e-5
        @test p3_slope(1f-4, 1f3, m32) ≈ p3_slope(1e-4, 1e3, m64) rtol = 1e-5
    end
end
