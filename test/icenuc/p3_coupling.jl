# Phase 43, issue #7 — nucleation sources coupled into the P3 ice category.
#
# Asserted: one nucleated-ice mass constant; every INP consumed becomes
# exactly one P3 ice particle (number conservation); total water exact;
# the heating is the P3 enthalpy form c_pm dT = -L_v dq_v + L_f dq_i;
# frozen droplets enter the rime at 900 kg m⁻³ and vapour-nucleated
# crystals as unrimed ice; INP of sublimated ice are released.

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

    @testset "One nucleated-ice mass" begin
        @test NUCLEATED_ICE_MASS === P3_NUCLEATED_ICE_MASS
        @test P3_NUCLEATED_ICE_MASS ≈ 4π / 3 * 900 * 1e-18 rtol = 1e-15
        @test P3_DEFAULT_HM.m_splinter === P3_NUCLEATED_ICE_MASS
    end

    @testset "Immersion freezing of cloud: number, water, energy, rime" begin
        T, p = 248.15, 7.0e4
        qs = saturation_specific_humidity(T, p, c)
        s = P3ParcelState(T = T, p = p, q_v = qs, q_c = 2e-4, N_c = 1.2e8,
                          q_r = 1e-5, N_r = 1e3,
                          q_i = 1e-6, N_i = 1e3, q_rim = 2e-7, B_rim = 2e-7 / 400)
        inp = newinp()
        cons0 = inp_number(inp; population = :consumed)
        s1, t = p3_ice_nucleation(s, inp, 10.0)
        ρ = ρof(s)
        @printf("    -25 °C cloud: %.3f L⁻¹ crystals, %.3e kg/kg of cloud frozen\n",
                1e-3 * ice_number_formed(t), s.q_c - s1.q_c)
        @test t.N_ice_imm > 0
        # number: each consumed INP is one P3 ice particle
        @test (s1.N_i - s.N_i) * ρ ≈ ice_number_formed(t) rtol = 1e-12
        @test (s1.N_i - s.N_i) * ρ ≈ inp_number(inp; population = :consumed) - cons0 rtol = 1e-12
        @test (s.N_c - s1.N_c) * ρ ≈ t.N_liq rtol = 1e-10
        # water: exact to rounding; rain untouched
        @test p3_total_water(s1) ≈ p3_total_water(s) rtol = 1e-15
        @test s1.q_r == s.q_r && s1.N_r == s.N_r
        @test (s1.q_i - s.q_i) ≈ (s.q_c - s1.q_c) + (s.q_v - s1.q_v) rtol = 1e-12
        # frozen drops are rime at ρ_rim_max; ice mass from liquid only here
        @test s1.q_rim - s.q_rim ≈ s.q_c - s1.q_c rtol = 1e-12
        @test (s1.q_rim - s.q_rim) / (s1.B_rim - s.B_rim) ≈ pp.ice.ρ_rim_max rtol = 1e-10
        # energy: the P3 enthalpy form, c_pm ΔT = -L_v Δq_v + L_f Δq_i.
        # ΔT here is ~10⁻⁵ K added to 248 K, so the difference s1.T - s.T
        # carries ~10⁻⁸ relative rounding: 1e-6 is a rounding tolerance.
        dqv, dqi = s1.q_v - s.q_v, s1.q_i - s.q_i
        @test cpm(s) * (s1.T - s.T) ≈
              -latent_heat_vapor(T, c) * dqv + latent_heat_fusion(T, c) * dqi rtol = 1e-6
        @test t.heat / ρ ≈ latent_heat_fusion(T, c) * (s.q_c - s1.q_c) rtol = 1e-10
        # a second call at the same state nucleates nothing (singular, consumed)
        s2, t2 = p3_ice_nucleation(s1, inp, 10.0)
        @test ice_number_formed(t2) == 0 && s2 == s1
    end

    @testset "Deposition nucleation: unrimed crystals of mass mi0" begin
        T, p = 225.0, 3.0e4
        qsi = saturation_specific_humidity(T, p, c; over = :ice)
        s = P3ParcelState(T = T, p = p, q_v = 1.25 * qsi)
        inp = newinp()
        s1, t = p3_ice_nucleation(s, inp, 5.0)
        ρ = ρof(s)
        @test t.N_ice_dep > 0 && t.M_liq == 0
        @test (s1.N_i - s.N_i) * ρ ≈ t.N_ice_dep rtol = 1e-12
        @test (s1.q_i - s.q_i) ≈ t.N_ice_dep * P3_NUCLEATED_ICE_MASS / ρ rtol = 1e-12
        @test s1.q_rim == 0 && s1.B_rim == 0
        @test p3_total_water(s1) ≈ p3_total_water(s) rtol = 1e-15
        @test cpm(s) * (s1.T - s.T) ≈
              -latent_heat_vapor(T, c) * (s1.q_v - s.q_v) +
              latent_heat_fusion(T, c) * (s1.q_i - s.q_i) rtol = 1e-6   # ΔT ~ 10⁻⁶ K: rounding
    end

    @testset "Homogeneous freezing: all cloud to rime" begin
        T, p = 232.0, 3.5e4
        s = P3ParcelState(T = T, p = p, q_v = saturation_specific_humidity(T, p, c),
                          q_c = 1e-4, N_c = 2e8)
        s1, t = p3_ice_nucleation(s, newinp(), 5.0)
        @test s1.q_c == 0 && s1.N_c == 0
        @test s1.q_rim ≈ s.q_c rtol = 1e-14
        @test (s1.N_i - s.N_i) ≥ s.N_c * (1 - 1e-12)
        @test p3_total_water(s1) ≈ p3_total_water(s) rtol = 1e-15
    end

    @testset "Warm and ice-free cases leave the state unchanged" begin
        s = P3ParcelState(T = 275.0, p = 9e4, q_v = 5e-3, q_c = 1e-4, N_c = 1e8)
        inp = newinp()
        s1, t = p3_ice_nucleation(s, inp, 10.0)
        @test s1 === s && ice_number_formed(t) == 0
        @test inp_number(inp; population = :consumed) == 0
    end

    @testset "INP released by sublimating ice" begin
        # nucleate, then put the ice in dry air so it sublimates
        T, p = 248.15, 7.0e4
        s = P3ParcelState(T = T, p = p, q_v = saturation_specific_humidity(T, p, c),
                          q_c = 2e-4, N_c = 1.2e8)
        inp = newinp()
        s1, _ = p3_ice_nucleation(s, inp, 10.0)
        dry = P3ParcelState(T = T, p = p, q_v = 0.3 * saturation_specific_humidity(T, p, c; over = :ice),
                            q_i = s1.q_i, N_i = s1.N_i, q_rim = s1.q_rim, B_rim = s1.B_rim)
        cons = inp_number(inp; population = :consumed)
        avail = inp_number(inp)
        s2, r = p3_step_with_rates(dry, 20.0)
        @test r.sub > 0 && r.n_sub > 0
        f = min(1, 20.0 * r.n_sub / dry.N_i)
        ΔN = p3_inp_release!(inp, dry, r, 20.0)
        @test ΔN ≈ f * cons rtol = 1e-12
        @test inp_number(inp) ≈ avail + ΔN rtol = 1e-12
        @test inp_number(inp) + inp_number(inp; population = :consumed) ≈ avail + cons rtol = 1e-12
        @test maximum(abs, values(inp_budget_residual(inp))) < 1e-14
        # growing ice (no sublimation) releases nothing
        wet = P3ParcelState(T = T, p = p, q_v = saturation_specific_humidity(T, p, c),
                            q_i = s1.q_i, N_i = s1.N_i, q_rim = s1.q_rim, B_rim = s1.B_rim)
        _, rw = p3_step_with_rates(wet, 20.0)
        @test p3_inp_release!(inp, wet, rw, 20.0) == 0.0
    end
end
