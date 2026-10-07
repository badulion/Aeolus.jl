# The decaying-turbulence initial field shared by the SGS tests (#317):
# test/era2_sgs_numerics.jl, test/era6_lagrangian_sgs.jl, test/era6_lasd.jl,
# test/era6_amd_vreman.jl and test/era6_sigma_sgs.jl each carried a copy of
# it (three named, two inline). The includer loads Aeolus.

using Random: AbstractRNG
using Statistics: mean

"""
    kolmogorov_field!(sim, rng; kc = sim.mesh.Nx / 4) -> sim

Overwrite the velocity of `sim` with a solenoidal random field of energy
spectrum ``E(k) ∝ k^{-5/3}`` up to ``k_c`` fundamental wavenumbers
``k_0 = 2π/L_x``, random phases from `rng` (the src initialiser
`isotropic_random_field!`, #168), scaled to ``u_\\mathrm{rms} = 1``
and with its halos filled. The same `rng` state gives the same field bit
for bit, so the copies it replaced give identical runs.
"""
function kolmogorov_field!(sim, rng::AbstractRNG; kc = sim.mesh.Nx / 4)
    k0 = 2π / sim.mesh.Lx
    isotropic_random_field!(sim, k -> (k / k0)^(-5/3); rng, kmax = kc * k0)
    s = 1 / sqrt(mean(abs2, Aeolus.Fields.interior_view(sim.u)))
    for f in (sim.u, sim.v, sim.w)
        Aeolus.Fields.interior_view(f) .*= s     # u_rms = 1
        fill_halos!(f, sim.bcs)
    end
    return sim
end
