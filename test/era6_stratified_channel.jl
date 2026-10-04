# P55-f (#191): the wall-resolved, stably stratified channel against the
# DNS of García-Villalba & del Álamo (2011, Phys. Fluids 23, 045104), on
# the P54-d `channel_case` builder with the P59-a `NoSlipWall`
# (`bottom_bc = :noslip`) and isothermal θ walls (`θ_walls`).

using Test
using Aeolus
using Statistics: mean

@testset "Stratified channel (P55-f)" begin

@testset "Harness: no-slip θ-walled channel, laminar exact solutions" begin
    # Ri_τ = g Δθ δ / (θ₀ u_τ²) round trip
    w = stratified_channel_walls(60.0; θ₀ = 300.0, g_accel = 9.81, δ = 0.5, u_τ = 0.2)
    c = channel_case(; Re_τ = 10, N = (4, 4, 16), closure = nothing, bottom_bc = :noslip,
                     noise = 0.0, θ_walls = w, θ₀ = 300.0, g_accel = 9.81, δ = 0.5, u_τ = 0.2)
    @test channel_richardson(c) ≈ 60.0 rtol = 1e-12
    @test c.sim.bcs.bottom isa NoSlipWall && c.sim.bcs.top isa NoSlipWall
    @test c.sim.has_buoyancy && c.bottom_bc === :noslip

    # Laminar Poiseuille flow with conduction between the walls (Re_τ = 10,
    # stable to perturbations): Δt ≪ dz²/ν. The linear conduction profile
    # is the exact discrete steady state of the half-cell wall conduction,
    # so the wall heat flux is the laminar one, Nu = 1, to round-off.
    Pr = 0.71
    c = channel_case(; Re_τ = 10, N = (4, 4, 16), closure = nothing, bottom_bc = :noslip,
                     noise = 0.0, θ_walls = stratified_channel_walls(5.0), Pr = Pr)
    ν = c.sim.ν; κ = ν / Pr; Δθ = c.θ_walls[2] - c.θ_walls[1]
    q_lam = -κ * Δθ / 2                                  # upward (down-gradient) flux, δ = 1
    θ0 = copy(c.sim.θ.data)
    for _ in 1:3000
        advance_channel!(c, 0.02)
    end
    @test c.q_wall_bottom ≈ q_lam rtol = 1e-12          # Nu = 1 at both walls
    @test c.q_wall_top ≈ q_lam rtol = 1e-12
    @test maximum(abs, c.sim.θ.data[3:6, 3:6, 3:18] .- θ0[3:6, 3:6, 3:18]) < 1e-12
    @test maximum(abs, Aeolus.Operators.interior_view(c.sim.w)) < 1e-12   # hydrostatic
    # steady Poiseuille: the wall stress balances the forcing (τ_w = u_τ² = 1)
    # and the total stress is u_τ²(1 - z/δ) (viscous at every face, incl. the walls)
    @test c.τ_wall_bottom ≈ 1.0 rtol = 1e-6
    @test c.τ_wall_top ≈ 1.0 rtol = 1e-6
    p = total_stress_profile(c)
    @test maximum(abs, p.total .- (1 .- p.z)) < 1e-6
    @test maximum(abs, p.wall) == 0                      # no modelled wall stress

    # Pr enters only the diffusivity: with Pr = 1 the same flux is κ = ν
    c1 = channel_case(; Re_τ = 10, N = (4, 4, 16), closure = nothing, bottom_bc = :noslip,
                      noise = 0.0, θ_walls = (0.0, 2.0), Pr = 1.0)
    advance_channel!(c1, 0.02)
    @test c1.q_wall_bottom ≈ -c1.sim.ν * 2.0 / 2 rtol = 1e-12

    # open channel: free-slip, insulated top
    co = channel_case(; Re_τ = 10, N = (4, 4, 16), closure = nothing, bottom_bc = :noslip,
                      walls = :bottom, noise = 0.0, θ_walls = (1.0, 0.0))
    @test co.sim.bcs.top isa Aeolus.BoundaryConditions.ImpermeableWall
    advance_channel!(co, 0.02)
    @test co.q_wall_top == 0 && co.τ_wall_top == 0

    # invalid combinations
    @test_throws ArgumentError channel_case(; bottom_bc = :wall_model, θ_walls = (0.0, 1.0))
    @test_throws ArgumentError channel_case(; bottom_bc = :noslip, θ_walls = (0.0, 1.0), Pr = 0.0)
    @test_throws ArgumentError channel_case(; sgs_stress = :bogus)
end

end # testset
