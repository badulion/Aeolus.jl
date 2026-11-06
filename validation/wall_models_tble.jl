# Phase 57 (P57-c, #189): validation of the non-equilibrium (TBLE)
# wall model in a simulated flow — the wall-modelled channel at Re_τ = 1000
# (24×12×16, h_wm = 3Δz, time filter, the pressure-gradient TBLE): its bulk
# velocity against Lee & Moser (2015), ⟨τ_w⟩ = u_τ², and the dynamic κ_wm
# of Wang & Moin (2002). The model stays tested in
# test/wall_models_tble.jl (equilibrium limit, Stokes' first problem,
# the pressure-gradient balance, column conservation, the operator, the
# non-equilibrium response).

using Test
using Printf
using Random
using Aeolus
# the wall-bounded benchmark harness (half channel, channel) is a test fixture (ONE-l, #475)
isdefined(@__MODULE__, :ChannelCases) || include(joinpath(@__DIR__, "..", "test", "fixtures", "channel_cases.jl"))
using .ChannelCases
using Aeolus.WallModels

isdefined(@__MODULE__, :lm2015_ub_plus) || include(joinpath(@__DIR__, "..", "test", "data", "lee_moser_2015.jl"))

@testset "non-equilibrium (TBLE) wall model: channel at Re_τ = 1000 (#189)" begin

@testset "5. Channel Re_τ = 1000 (24×12×16, h_wm = 3Δz, time filter, :pressure_gradient)" begin
    Ub_LM = lm2015_ub_plus(1000)
    dz = 2 / 16
    c = channel_case(; Re_τ = 1000, N = (24, 12, 16), closure = StaticSmagorinsky(0.1),
                     wall_law = SpaldingLaw(), seed = 1, h_wm = 3dz, filter = :temporal,
                     wall_model = :nonequilibrium,
                     tble = (terms = :pressure_gradient, dynamic_kappa = :total, κ_feedback = false,
                             T_κ = 0.5))
    ks = Float64[]
    st = run_channel!(c, 8.0; average_from = 5.0,
                      callback = cc -> cc.sim.t >= 5 && push!(ks, (tble_kappa(cc.wall, :bottom; dynamic = true) +
                                                                  tble_kappa(cc.wall, :top; dynamic = true)) / 2))
    Ub = sum(st.U) / length(st.U)
    κd = sum(ks) / length(ks)
    @info @sprintf("  channel Re_τ=1000 TBLE-PG: U_b⁺ %.3f (%+.2f %% vs Lee & Moser %.2f), ⟨τ_w⟩ %.3f, Φ_m(Δz) %.3f, dynamic κ_wm %.3f (%.1f s)",
                   Ub, 100(Ub / Ub_LM - 1), Ub_LM, st.τ_wall, st.phi_m[1], κd, st.wall_time)
    @test abs(Ub / Ub_LM - 1) < 0.05
    @test abs(st.τ_wall - 1) < 0.07
    # the dynamic κ_wm (Wang & Moin 2002: ≈ 0.2): the effective LES eddy
    # viscosity at h = 0.375δ, close to 0.41(1 − h/δ) = 0.256, the
    # outer-layer stress decrease the mixing-length ν_t,wm lacks
    @test 0.15 < κd < 0.32
    @test tble_kappa(c.wall) == 0.41                  # not fed back
end

end
