# Phase 43, issue #7 — nucleation sources coupled into the P3 ice category.
#
# Asserted: every INP consumed becomes exactly one P3 ice particle; total
# water exact; the heating is the P3 enthalpy form
# c_pm dT = -L_v dq_v + L_f dq_i; frozen droplets enter the rime at
# ρ_rim_max and vapour-nucleated crystals as unrimed ice of mass mi0.

using Aeolus.MoistThermodynamics: PhasePartition, air_density, heat_capacity_moist,
                                  latent_heat_fusion, latent_heat_vapor,
                                  saturation_specific_humidity

@testset "Nucleation into the P3 ice category (issue #7)" begin
    pp = P3_DEFAULT_PROCESS
    c = pp.thermo
    newinp() = inp_population([lognormal_mode(:mineral_dust, 1.0, 0.4, 2.0),
                               lognormal_mode(:biological, 0.01, 1.0, 1.6)])
    ρof(s) = air_density(s.T, s.p, PhasePartition(p3_total_water(s), s.q_c + s.q_r, s.q_i), c)
    cpm(s) = heat_capacity_moist(PhasePartition(p3_total_water(s), s.q_c + s.q_r, s.q_i), c)

    @test NUCLEATED_ICE_MASS === P3_NUCLEATED_ICE_MASS             # one constant
    @test P3_NUCLEATED_ICE_MASS ≈ 4π / 3 * 900 * 1e-18 rtol = 1e-15

    @testset "Immersion freezing of cloud: number, water, energy, rime" begin
        T, p = 248.15, 7.0e4
        s = P3ParcelState(T = T, p = p, q_v = saturation_specific_humidity(T, p, c),
                          q_c = 2e-4, N_c = 1.2e8, q_i = 1e-6, N_i = 1e3,
                          q_rim = 2e-7, B_rim = 2e-7 / 400)
        inp = newinp()
        s1, t = p3_ice_nucleation(s, inp, 10.0)
        ρ = ρof(s)
        @test t.N_ice_imm > 0
        @test (s1.N_i - s.N_i) * ρ ≈ inp_number(inp; population = :consumed) rtol = 1e-12
        @test p3_total_water(s1) ≈ p3_total_water(s) rtol = 1e-15
        @test (s1.q_rim - s.q_rim) / (s1.B_rim - s.B_rim) ≈ pp.ice.ρ_rim_max rtol = 1e-10
        # ΔT ~ 1e-5 K on 248 K: s1.T - s.T carries ~1e-8 relative rounding
        @test cpm(s) * (s1.T - s.T) ≈ -latent_heat_vapor(T, c) * (s1.q_v - s.q_v) +
                                      latent_heat_fusion(T, c) * (s1.q_i - s.q_i) rtol = 1e-6
        # singular: a second call at the same state nucleates nothing
        @test ice_number_formed(p3_ice_nucleation(s1, inp, 10.0)[2]) == 0
    end

    @testset "Deposition nucleation: unrimed crystals of mass mi0" begin
        T, p = 225.0, 3.0e4
        s = P3ParcelState(T = T, p = p, q_v = 1.25 * saturation_specific_humidity(T, p, c; over = :ice))
        s1, t = p3_ice_nucleation(s, newinp(), 5.0)
        ρ = ρof(s)
        @test t.N_ice_dep > 0
        @test (s1.q_i - s.q_i) ≈ t.N_ice_dep * P3_NUCLEATED_ICE_MASS / ρ rtol = 1e-12
        @test s1.q_rim == 0 && s1.B_rim == 0
        @test p3_total_water(s1) ≈ p3_total_water(s) rtol = 1e-15
    end

    @testset "INP released by sublimating ice" begin
        T, p = 248.15, 7.0e4
        s = P3ParcelState(T = T, p = p, q_v = saturation_specific_humidity(T, p, c),
                          q_c = 2e-4, N_c = 1.2e8)
        inp = newinp()
        s1, _ = p3_ice_nucleation(s, inp, 10.0)
        dry = P3ParcelState(T = T, p = p, q_v = 0.3 * saturation_specific_humidity(T, p, c; over = :ice),
                            q_i = s1.q_i, N_i = s1.N_i, q_rim = s1.q_rim, B_rim = s1.B_rim)
        cons = inp_number(inp; population = :consumed)
        _, r = p3_step_with_rates(dry, 20.0)
        ΔN = p3_inp_release!(inp, dry, r, 20.0)
        @test ΔN ≈ min(1, 20.0 * r.n_sub / dry.N_i) * cons rtol = 1e-12
        @test maximum(abs, values(inp_budget_residual(inp))) < 1e-14
    end
end
