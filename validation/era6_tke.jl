# Era VI, Phase 56 (P56-b, #182): validation of the Deardorff (1980)
# 1.5-order prognostic SGS TKE closure with the Moeng (1984) coefficients
# in simulated flows:
#
#   5. positivity over 500 RK3 steps of decaying turbulence (neutral, and
#      stably stratified with WENO-Z5 TKE advection): e, ν_t ≥ 0,
#      κ_t ≥ ν_t, zero clips, conservative transport;
#   6. the reduced convective boundary layer (24³, Δ = 50 m, 1 h): the
#      entrainment flux ratio, the SGS share, the surface-layer e peak,
#      encroachment growth, z_i by two methods; the κ·d_wall cap.
#
# The closure stays tested in test/era6_tke.jl (wall distance, free decay,
# shear equilibrium, critical Ri, energy exchange, a 20-step positivity
# run, the contract, the saturated column).

using Test
using Aeolus
using Aeolus.Fields: interior_view, set!
using Random: Xoshiro
using Statistics: mean
using Printf

# the CBL harness tke_cbl (a test fixture; the model's configuration is examples/cbl_tke/cbl_tke.toml)
isdefined(@__MODULE__, :TKECBL) || include(joinpath(@__DIR__, "..", "test", "fixtures", "tke_cbl.jl"))
using .TKECBL: tke_cbl, tke_cbl_summary

@testset "Era VI — prognostic SGS TKE: decaying turbulence and the CBL (P56-b, #182)" begin


@testset "5. positivity over 500 steps, zero clips" begin
    Ek(k) = 0.05 * k^4 * exp(-2 * (k / 4)^2)
    for (H, strat) in ((2, false), (3, true))
        m = RegularGrid(; size = (16, 16, 16), extent = (2π, 2π, 2π), halo = (H, H, H))
        sim = strat ? BoussinesqSimulation(m; θ₀ = 300.0,
                                           bcs = FieldBoundaryConditions(:periodic, :periodic, :wall),
                                           θ_bcs = FieldBoundaryConditions(:periodic, :periodic, :noflux)) :
                      BoussinesqSimulation(m)
        isotropic_random_field!(sim, Ek; rng = Xoshiro(2))
        strat && (sim.w.data[:, :, m.Hz+1] .= 0; set!(sim.θ, (x, y, z) -> 300.0 + 3z))
        c = TKEClosure(m)
        @test c.advection === (H >= 3 ? :weno_z5 : :centered)
        initialize_tke!(c, sim)
        mine, minν, minΔκ = Inf, Inf, Inf
        for _ in 1:500
            step_rk3!(sim, 0.01; sgs = c)
            mine = min(mine, minimum(interior_view(c.e)))
            minν = min(minν, minimum(interior_view(sim.νt)))
            minΔκ = min(minΔκ, minimum(interior_view(sim.κt) .- interior_view(sim.νt)))
        end
        st = tke_statistics(c)
        @test mine >= 0 && minν >= 0 && minΔκ >= 0            # e ≥ 0, ν_t ≥ 0, κ_t ≥ ν_t
        @test st.clips == 0 && st.clipped_energy == 0
        @test st.steps == 500                                  # one commit per RK3 step
        @test abs(st.transport) < 1e-12 * st.dissipation       # flux form, conservative
        strat && @test st.buoyancy < 0 && st.buoyant_destruction > 0
    end
end

@testset "6. reduced CBL: 24³, Δ = 50 m, Q_s = 0.1 K m/s, 1 h" begin
    r = tke_cbl(24; Δ = 50.0, Q = 0.1, T_end = 3600.0, avg = 900.0)
    @info "  " * tke_cbl_summary(r)
    @test r.clips == 0 && r.min_e >= 0
    # heat-flux ratio minimum (at z_i) in the entrainment range
    @test -0.25 < r.min_flux_ratio < -0.10
    # SGS share of the TKE below 25 % in the mixed layer
    @test r.max_sgs_fraction < 0.25
    # SGS TKE peaks in the surface layer (z ≤ 0.1 z_i)
    @test r.z_epeak <= 0.1 * r.zi
    # the boundary layer grew by encroachment: z_i² − z_i0² ≈ 2(1 + 2A) Q t / Γ
    @test 750 < r.zi < 1000
    # N-parent-a (#247): the harness's z_i is the one diagnostic,
    # boundary_layer_height(:min_flux); it equals the former local code
    # (zf[argmin(flux)]), and the parcel method on the mean θ
    # profile agrees within 10 %
    @test values_agree(r.zi, r.zf[argmin(r.flux)])
    zi_parcel = boundary_layer_height(r.zc, r.θ; method = :parcel)
    @info @sprintf("  z_i: min flux %.0f m, parcel (mean θ) %.0f m (%.1f %%)", r.zi, zi_parcel,
                   100 * (zi_parcel / r.zi - 1))
    @test abs(zi_parcel - r.zi) < 0.1 * r.zi
    # With the mixing length also capped at κ·d_wall (on top of Moeng's 3.9)
    # the first levels are over-damped: measured min flux ratio −0.08, SGS
    # fraction 0.31, e peak at 125 m — outside the three criteria above,
    # which come from Moeng-type LES without a wall cap (see sgs_tke.md).
    rc = tke_cbl(24; Δ = 50.0, Q = 0.1, T_end = 3600.0, avg = 900.0, wall_cap = true)
    @info "  wall cap: " * tke_cbl_summary(rc)
    @test rc.clips == 0
    @test rc.e[1] < r.e[1]                                  # the cap lowers e at the first level
end

end
