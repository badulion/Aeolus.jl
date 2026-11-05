# The dry convective boundary layer on both dynamical cores (ONE-j, #460),
# 30 min of the `simulation` config at 16³ (360 steps per core): the
# anelastic mixed layer warms more than the Boussinesq one by about the
# column-mass factor ρ̄_bot h / ∫₀ʰ ρ̄ dz − 1 (its heat goes into less mass
# aloft), the θ budget still closes, and the two cores' profiles agree to
# the core difference.
#
# The cores and their components stay tested in test/components_both_cores.jl
# (the θ budget to round-off over 12 steps on every scheme, the weighted
# operators, checkpoints and the component contracts on both cores).

using Test
using Printf
using Aeolus
using Aeolus.Fields: interior_view

# BC_CFG, bc_cbl_cfg, bc_run, bc_budget, bc_weights (shared with the test file)
isdefined(@__MODULE__, :bc_budget) || include(joinpath(@__DIR__, "..", "test", "support", "dcbl_both_cores.jl"))

@testset "dry CBL on both cores: the anelastic mixed layer warms more (30 min, 16³)" begin
    # The same kinematic flux F: the Boussinesq core heats a column of ρ₀
    # (Σ δθ Δz = F t), the anelastic one a column of ρ̄(z) that thins upward
    # (Σ ρ̄ δθ Δz = ρ̄_bot F t). Mixed over a layer of depth h, the anelastic
    # warming is larger by ρ̄_bot h / ∫₀ʰ ρ̄ dz ≈ 1 + h/(2 H_ρ) (H_ρ ≈ 12 km for
    # the 300 K isentropic base state): about 4 % at h ≈ 1 km.
    B = bc_budget(:boussinesq, 360); A = bc_budget(:anelastic, 360)
    @test abs(B.got / B.expected - 1) < 1e-11 && abs(A.got / A.expected - 1) < 1e-11   # δθ ≈ 0.5 K here
    m = A.sim.mesh; zc = [(k - 0.5) * m.dz for k in 1:m.Nz]
    ml = findall(z -> 200 <= z <= 800, zc)                         # the mixed layer, away from the surface and the top
    mean_dθ(b) = sum(sum(interior_view(b.sim.θ)[:, :, k] .- b.θ0[:, :, k]) for k in ml) / (length(ml) * m.Nx * m.Ny)
    dB = mean_dθ(B); dA = mean_dθ(A)
    ρ̄ = A.sim.ref.ρ̄; h = 1000.0
    pred = ρ̄[1] * h / (sum(ρ̄[k] for k in 1:m.Nz if zc[k] < h) * m.dz) - 1
    @info @sprintf("DCBL 30 min: mixed-layer warming Boussinesq %.4f K, anelastic %.4f K, ratio − 1 = %.4f (column-mass estimate %.4f at h = 1 km)",
                   dB, dA, dA / dB - 1, pred)
    @test 0.3 < dB < 0.7                                           # F t / h ≈ 0.43 K
    @test 0.5pred < dA / dB - 1 < 1.5pred
    # the profiles agree to the core difference: within 0.05 K below 1.6 km
    pA = vec(sum(interior_view(A.sim.θ); dims = (1, 2))) ./ 256; pB = vec(sum(interior_view(B.sim.θ); dims = (1, 2))) ./ 256
    @test maximum(abs, pA .- pB) < 0.05
end
