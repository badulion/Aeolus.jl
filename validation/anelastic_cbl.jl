# Boussinesq against anelastic: a convective boundary layer growing to
# about 2 km on a uniform grid (check 5 of the Berlin parent validation,
# N-parent-e #259; moved from test/berlin_parent.jl, #320 — it never used
# the parent case, only the two cores and the TKE closure).
#
# The same kinematic surface heat flux Q into the same θ profile: the
# anelastic CBL is deeper and warmer, because the heat goes into air whose
# density falls with height (ρ̄ is ~20 % lower at 2 km), which the
# Boussinesq core (one ρ₀) does not see.
#
# A validation run (5 h of simulated CBL in each core). The two cores stay
# tested in test/anelastic.jl, the TKE closure in its own test files
# (e.g. test/tke_stretched.jl).

using Test
using Printf
using Random: Xoshiro
using Aeolus
using Aeolus.Fields: interior_view, set!

"""
The plane-mean total kinematic heat flux at the interior w faces of `sim`:
resolved ⟨w'θ'⟩ (θ averaged to the face) plus the SGS −⟨κ ∂θ/∂z⟩, the
profile whose minimum is the `:min_flux` boundary-layer height (Sullivan
et al. 1998).
"""
function _cbl_heat_flux(sim, κ)
    m = sim.mesh
    Hx, Hy, Hz, Nx, Ny, Nz = m.Hx, m.Hy, m.Hz, m.Nx, m.Ny, m.Nz
    θd, wd, κd = sim.θ.data, sim.w.data, κ.data
    zf = [(k - 1) * m.dz for k in 2:Nz]
    flux = zeros(Nz - 1)
    n = Nx * Ny
    for k in 2:Nz
        K = Hz + k
        sw = 0.0; sθ = 0.0; swθ = 0.0; sg = 0.0
        for j in Hy+1:Hy+Ny, i in Hx+1:Hx+Nx
            θf = 0.5 * (θd[i, j, K-1] + θd[i, j, K])
            w = wd[i, j, K]
            sw += w; sθ += θf; swθ += w * θf
            sg -= 0.5 * (κd[i, j, K-1] + κd[i, j, K]) * (θd[i, j, K] - θd[i, j, K-1]) / m.dz
        end
        flux[k-1] = swθ / n - (sw / n) * (sθ / n) + sg / n
    end
    return zf, flux
end

function _cbl_run(core; N = 24, Nz = 40, L = 3000.0, H = 3000.0, Q = 0.25, Γ = 3e-3, hours = 5.0, dt = 5.0)
    m = RegularGrid(; size = (N, N, Nz), extent = (L, L, H), halo = (3, 3, 3))
    θf(z) = 300 + Γ * z
    bcs = FieldBoundaryConditions(:periodic, :periodic, :wall)
    θb = FieldBoundaryConditions(:periodic, :periodic; bottom = :noflux, top = FluxBC(Γ))
    kw = (θ_flux_bottom = Q, sponge_α_max = 0.01, sponge_z_start = 0.75H, bcs = bcs, θ_bcs = θb)
    sim = core === :boussinesq ? BoussinesqSimulation(m; θ₀ = 300.0, kw...) :
          Aeolus.Anelastic.AnelasticSimulation(m; ref = Aeolus.Anelastic.reference_from_profile(m, θf, 101325.0), kw...)
    set!(sim.θ, (x, y, z) -> θf(z))
    rng = Xoshiro(5)
    interior_view(sim.θ)[:, :, 1:2] .+= 0.2 .* (rand(rng, N, N, 2) .- 0.5)
    c = TKEClosure(m; transport = true)
    initialize_tke!(c, sim; e_min = 1e-4)
    for _ in 1:round(Int, hours * 3600 / dt)
        step!(sim, dt; sgs = c)
    end
    # the κt of the step (sim.κt, as gabls1.jl); the closure's own κt is
    # scratch that the cache-blocked TKE update (#554) does not keep
    zf, fl = _cbl_heat_flux(sim, sim.κt)
    zi = boundary_layer_height(zf, fl; method = :min_flux)
    θm = vec(sum(interior_view(sim.θ); dims = (1, 2))) ./ N^2
    zc = [(k - 0.5) * m.dz for k in 1:Nz]
    ml = [θm[k] for k in 1:Nz if 0.2zi < zc[k] < 0.8zi]
    return (zi = zi, θml = sum(ml) / length(ml))
end

@testset "Boussinesq against anelastic: 5 h CBL to ~2 km (#259, #320)" begin
    b = _cbl_run(:boussinesq); a = _cbl_run(:anelastic)
    dzi = (a.zi - b.zi) / b.zi; dθ = a.θml - b.θml
    @info @sprintf("core sensitivity, 5 h CBL (Q = 0.25 K m/s, Γ = 3 K/km): z_i Boussinesq %.0f m, anelastic %.0f m (Δz_i/z_i = %+.3f); θ_ML %.2f vs %.2f K (Δθ_ML = %+.2f K)",
                   b.zi, a.zi, dzi, b.θml, a.θml, dθ)
    @test 1700 < b.zi < 2500        # "to about 2 km"
    # measured +0.034 (2175 vs 2250 m: z_i moves in whole 75 m levels) on
    # 2026-10-07; +0.107 (2100 vs 2325 m) when #259 was merged
    @test 0 <= dzi < 0.2
    @test 0 <= dθ < 0.5             # measured +0.22 K (+0.23 K at #259)
end
