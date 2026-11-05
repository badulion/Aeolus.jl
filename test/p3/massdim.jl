# P3 part-file tests: four-regime mass-dimension relation, predicted rime
# density, slope solve, fall speed (src/p3/massdim.jl, issue #2).
#
# The point of P3 is that ice properties vary *continuously* with rime
# fraction and rime density: no category jump from snow to graupel.

@testset "P3 mass-dimension, size distribution and fall speed" begin
    relerr(x, ref) = abs(x - ref) / abs(ref)
    maxstep(v) = maximum(abs.(diff(v)) ./ abs.(v[1:end-1]))
    p = P3_DEFAULT_ICE

    @testset "Thresholds and the defining equations" begin
        md0 = p3_mass_dimension(0.0, 400.0)
        # D_th: solid-ice sphere as heavy as an aggregate; MM15: ≈ 66 μm
        @test π / 6 * p.ρ_ice * md0.D_th^3 ≈ p.α_va * md0.D_th^p.β_va rtol = 1e-13
        @test 60e-6 < md0.D_th < 70e-6
        md = p3_mass_dimension(0.4, 250.0)
        @test md.ρ_g ≈ 0.4 * 250 + 0.6 * md.ρ_d rtol = 1e-13
        @test π / 6 * md.ρ_g * md.D_gr^3 ≈ p.α_va * md.D_gr^p.β_va rtol = 1e-12
        @test π / 6 * md.ρ_g * md.D_cr^3 ≈ p.α_va / 0.6 * md.D_cr^p.β_va rtol = 1e-12
        @test all(let m = p3_mass_dimension(F, ρr)
                      m.D_th <= m.D_gr <= m.D_cr && m.ρ_d <= m.ρ_g <= ρr * (1 + 1e-9)
                  end for F in 0:0.1:1, ρr in (50.0, 400.0, 900.0))
        # F → 0⁺: ρ_g → 2ρ_rim/3, down to subnormal F_rim (review
        # regression: log(D_cr/D_gr)² underflowed, ρ_g = -Inf at F ≲ 1e-200)
        @test maximum(relerr(p3_mass_dimension(F, 400.0).ρ_g, 800 / 3)
                      for F in (1e-12, 1e-200, nextfloat(0.0))) < 1e-11
        @test_throws DomainError p3_mass_dimension(1.2, 400.0)
    end

    @testset "Mass continuity across regime boundaries" begin
        maxjump = 0.0
        for F in (0.05, 0.6), ρr in (80.0, 850.0)
            md = p3_mass_dimension(F, ρr)
            @test [p3_regime(D, md) for D in (0.5md.D_th, 0.5(md.D_th + md.D_gr),
                                              0.5(md.D_gr + md.D_cr), 2md.D_cr)] == [1, 2, 3, 4]
            for Db in (md.D_th, md.D_gr, md.D_cr)
                maxjump = max(maxjump, relerr(p3_particle_mass(Db * (1 + 1e-12), md),
                                              p3_particle_mass(Db * (1 - 1e-12), md)))
            end
        end
        @test maxjump < 1e-11
        # unrimed: spheres below D_th, Brown-Francis aggregates above
        md = p3_mass_dimension(0.0, 400.0)
        @test p3_particle_mass(0.5md.D_th, md) ≈ π / 6 * p.ρ_ice * (0.5md.D_th)^3 rtol = 1e-14
        @test p3_particle_mass(1e-3, md) ≈ p.α_va * 1e-3^p.β_va rtol = 1e-14
    end

    @testset "Prognostic rime volume: density emerges" begin
        # rime of density 100 then 900: the mass-weighted harmonic mean
        q_rim, B_rim = 4e-4, 3e-4 / 100 + 1e-4 / 900
        F, ρ, _ = p3_rime_state(1e-3, q_rim, B_rim)
        @test F ≈ 0.4 rtol = 1e-15
        @test ρ ≈ 400 * 900 / 2800 rtol = 1e-14
        # clamping writes back a consistent rime volume
        _, ρc, Bc = p3_rime_state(1e-3, 4e-4, 4e-4 / 2000)
        @test ρc == p.ρ_rim_max && relerr(Bc, 4e-4 / p.ρ_rim_max) < 1e-15
    end

    @testset "Mean mass, bulk density and slope solve" begin
        # bulk density limits: tiny particles → ρ_ice; unrimed exponential
        # PSD in the aggregate regime → 6αΓ(β+1)λ^{3-β}/(πΓ(4))
        @test p3_bulk_density(0, 1e7, p3_mass_dimension(0.5, 400.0)) ≈ p.ρ_ice rtol = 1e-12
        λ = 20.0
        @test relerr(p3_bulk_density(0, λ, p3_mass_dimension(0.0, 400.0)),
                     6p.α_va * gamma_lanczos(p.β_va + 1) * λ^(3 - p.β_va) / (π * 6)) < 1e-6
        # λ round trip from (q, N)
        maxe = 0.0
        for F in (0.0, 0.6, 1.0), μ in (0.0, 4.0), λ in (3e2, 1e4, 3e5)
            md = p3_mass_dimension(F, 400.0)
            q = p3_ice_mass(gamma_psd_intercept(1e5, μ, λ), μ, λ, md)
            maxe = max(maxe, relerr(p3_size_distribution(q, 1e5, md; μ = μ)[3], λ))
        end
        @test maxe < 1e-12
    end

    @testset "Continuity of bulk properties over F_rim" begin
        # continuity ⇔ the largest step between neighbours shrinks ∝ spacing
        step(nF) = (Fs = range(0.0, 1.0; length = nF);
                    ls = [p3_slope(1e-4, 1e4, p3_mass_dimension(F, 400.0)) for F in Fs];
                    rb = [p3_bulk_density(0, λ, p3_mass_dimension(F, 400.0)) for (F, λ) in zip(Fs, ls)];
                    max(maxstep(ls), maxstep(rb)))
        s1, s2 = step(51), step(501)
        @test 8 < s1 / s2 < 12
        # riming at fixed q/N makes particles denser and smaller
        @test all(diff([p3_slope(1e-4, 1e4, p3_mass_dimension(F, 400.0)) for F in 0:0.1:1]) .> 0)
    end

    @testset "Fall speed" begin
        ρs, η = 1.2, p3_air_viscosity(283.15)
        @test p3_air_viscosity(273.15) ≈ 1.716e-5 rtol = 1e-3       # Sutherland
        md0 = p3_mass_dimension(0.0, 400.0)
        # tiny ice spheres: MH05 boundary-layer form → V/V_Stokes = 24/(δ₀²C₀)
        vt = p3_fall_speed(1e-7, md0; ρ_air = ρs, η_air = η)
        @test vt / (p.ρ_ice * 9.80665 * 1e-14 / (18η)) ≈ 24 / (p.δ0^2 * p.C0) rtol = 1e-4
        # unrimed aggregates vs Locatelli & Hobbs (1974) V = 0.8 D_mm^0.16
        @test maximum(relerr(p3_fall_speed(D, md0; ρ_air = ρs, η_air = η), 0.8 * (1e3D)^0.16)
                      for D in (0.5e-3, 2e-3, 1e-2)) < 0.15
        # dense graupel keeps speeding up with size (review regression: it
        # peaked at 11 m/s near 2 cm and fell at 0 m/s from 5 cm up)
        mg = p3_mass_dimension(1.0, 900.0)
        @test all(diff([p3_fall_speed(D, mg) for D in range(1e-3, 0.1; length = 20)]) .> 0)
        VN, VM = p3_bulk_fall_speed(0, 1e3, md0)
        @test VM > VN > 0
    end
end
