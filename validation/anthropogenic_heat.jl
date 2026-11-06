# Phase 49 — anthropogenic heat validation (#29): magnitude of the building
# part of Q_F. A 60-day winter run of the dense residential centre of
# Toulouse (BuildingEnergy zones in a facet scene, PVGIS TMY weather)
# against the CAPITOUL inventory of building electricity and gas use
# (Pigeon et al. 2007, as reported by Bueno et al. 2012, Table 2) and the
# range of published city Q_F values. The first law, the closed forms and
# step_buildings! are unit-tested in test/anthropogenic_heat.jl.

using Test, Printf
using Aeolus

include(joinpath(@__DIR__, "support", "toulouse_bem_driver.jl"))

@testset "anthropogenic heat from the building energy model (#29)" begin

    # =====================================================================
    # Magnitude: the dense centre of Toulouse in winter (CAPITOUL)
    # =====================================================================

    @testset "Toulouse centre, winter: building energy use vs the CAPITOUL inventory" begin
        # Observations: the CAPITOUL city-scale inventory of building
        # electricity and natural-gas consumption (Pigeon et al. 2007), as
        # compared with TEB-BEM for 19 Dec-17 Feb by Bueno et al. (2012),
        # GMD 5, 433-448, Table 2: model means 30.8 (electricity) and 19.0
        # (gas) W m-2 of urban area with mean bias errors -2.8 and -6.0 W m-2.
        # The table gives the model mean (REF) and MBE; the inventory means
        # below are REF - MBE, with MBE = model - observation (the text:
        # "BEM-TEB slightly underpredicts electricity and gas consumption").
        obs_elec = 30.8 + 2.8
        obs_gas = 19.0 + 6.0
        obs_total = obs_elec + obs_gas                              # 58.6 W m-2
        teb_bem_total = 30.8 + 19.0                                 # 49.8 W m-2
        sim = tls_scene()
        w = tls_weather()
        spin = 14
        hrs = tls_period(w, (12, 19 - spin), (2, 17))
        rec = [k > 24spin for k in eachindex(hrs)]
        out = tls_run(sim, w, hrs; record = rec)
        A = sim.tile
        mean_(v) = sum(v) / length(v)
        # Table 3: 2/3 of the heating electric, 1/3 gas (boiler η = 0.9,
        # assumed); 70 % of the internal gains electric ("electric fraction").
        # Table 3's wording, "fraction of electric heating systems over gas
        # heating systems 2/3", can also be read as electric : gas = 2 : 3;
        # the total moves by < 2 % between the readings (theory page).
        # The run itself heats electrically (η = 1), so the released heat
        # below omits the flue loss of the gas third, Q_h/3 (1/0.9 - 1).
        Q_h = mean_(out.Q_heating); Q_i = mean_(out.Q_internal)
        elec = (0.7 * Q_i + 2 / 3 * Q_h) / A
        gas = (0.3 * Q_i + Q_h / (3 * 0.9)) / A
        total = elec + gas
        rel = mean_(out.release) / A; direct = mean_(out.direct) / A
        facets = mean_(out.via_facets) / A; sun = mean_(out.Q_solar) / A
        @printf("  Toulouse 19 Dec-17 Feb (PVGIS TMY, T_out mean %.2f °C), per m² of urban area (central 60 × 60 m tile):\n",
                mean_(out.T_out) - 273.15)
        @printf("    heating delivered %.2f, internal gains %.2f W m-2; T_air %.2f-%.2f K\n",
                Q_h / A, Q_i / A, extrema(out.T_air)...)
        @printf("    energy use: electricity %.2f (inventory %.1f), gas %.2f (inventory %.1f), total %.2f (inventory %.1f, TEB-BEM %.1f): %+.1f %%\n",
                elec, obs_elec, gas, obs_gas, total, obs_total, teb_bem_total, 100 * (total / obs_total - 1))
        @printf("    heat released %.2f W m-2 = direct to air %.2f + through facets %.2f; transmitted sun %.2f\n",
                rel, direct, facets, sun)
        # Bueno et al.: "A relative error of 20 % in building energy
        # consumption is acceptable given the state-of-the-art of urban
        # canopy models."
        @test abs(total / obs_total - 1) < 0.20
        # Over the period, heat released = energy used + sun, up to the change
        # in storage (the run's heating is electric: energy used = Q_i + Q_h)
        @test rel ≈ (Q_i + Q_h) / A + sun rtol = 0.02
        # Magnitude: typical city Q_F values 5-100 W m-2, dense cores above
        # 100 W m-2 (Chrysoulakis & Grimmond 2016, citing Christen & Vogt
        # 2004, Allen et al. 2011, Iamarino et al. 2012; see also Sailor 2011)
        @test 5 < rel < 100
    end
end
