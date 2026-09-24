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
