# P59-e (#195): validation of relaminarisation in strongly stable
# stratification — decaying turbulence on a held laminar shear (32³) at
# Ri = N²/S² = 1 against the neutral control: the resolved buoyancy flux
# reverses, γ falls below 0.1 within about 10 N⁻¹, the σ-model + stability
# correction switches its vertical viscosity off, Smagorinsky does not.
# Relaminarisation stays tested in test/era6_relaminarisation.jl (the
# laminar stratified Poiseuille states, the criteria, laminar_mask!).

using Test
using Aeolus
using Aeolus.Fields: interior_view, Field
using Aeolus.Grids: CellCenter
using Aeolus.BoundaryConditions: fill_halos!
using Printf, Random
using Statistics: mean

@testset "P59-e: relaminarisation in decaying stratified turbulence" begin

# A laminar shear S held fixed (the plane mean of u is reset to S(z − H/2)
# after every step; a function of z only, so ∇·u is untouched) under a
# random-phase turbulent perturbation (E ∝ k⁴e^{−2(k/k_p)²}, k_p = 8π/H,
# u_rms = 0.05 SH), free-slip walls, 32³, ν = 3e-4 S H². Stratified at
# Ri = N²/S² = 1 (θ linear, g/θ₀ = 1), or neutral. The homogeneous
# stratified shear flow decays for Ri above Ri_s ≈ 0.1–0.2 and grows
# below it (Holt, Koseff & Ferziger 1992; Jacobitz, Sarkar & Van Atta
# 1997).
function sheared_decay(Ri; T = 12.0, closure = :sigma_corrected)
    N = 32; H = 1.0; S = 1.0; ν = 3e-4; u0 = 0.05
    mesh = RegularGrid(; size = (N, N, N), extent = (H, H, H))
    sim = BoussinesqSimulation(mesh; ν = ν, bcs = FieldBoundaryConditions(:periodic, :periodic, :wall),
                               θ₀ = 1.0, g_accel = Ri == 0 ? 0.0 : 1.0)
    cl = StabilityCorrected(SigmaModel(), mesh; length_scales = :isotropic)
    kp = 8π
    isotropic_random_field!(sim, k -> k^4 * exp(-2 * (k / kp)^2); rng = Xoshiro(1))
    ii, jj, kk = Aeolus.Grids.interior(mesh)
    r = sqrt(mean(abs2, view(sim.u.data, ii, jj, kk)))
    for f in (sim.u, sim.v, sim.w)
        f.data .*= u0 / r
    end
    sim.w.data[:, :, mesh.Hz+1] .= 0                     # the wall face
    zc = [(k - 0.5) * mesh.dz for k in 1:N]
    U = S .* (zc .- H / 2)
    hold!(sim) = for k in 1:N
        K = mesh.Hz + k
        sim.u.data[ii, jj, K] .+= U[k] - mean(view(sim.u.data, ii, jj, K))
    end
    hold!(sim)
    for k in 1:N
        sim.θ.data[:, :, mesh.Hz+k] .= 1.0 + Ri * S^2 * zc[k]
    end
    inner = 3:N-2                                         # away from the free-slip walls
    γ() = mean(turbulent_fraction(sim; normalization = :volume)[inner])
    function wθ()
        fill_halos!(sim.θ, sim.θ_bcs)
        mean(flux_richardson(sim.u, sim.v, sim.w, sim.θ; θ₀ = 1.0, g = 1.0).wθ[inner])
    end
    dt = 0.25 * mesh.dx / (S * H / 2 + 3u0)
    ts = Float64[]; γs = Float64[]; fs = Float64[]
    nsample = round(Int, 0.5 / dt)
    for n in 0:round(Int, T / dt)
        if n % nsample == 0
            push!(ts, sim.t); push!(γs, γ()); push!(fs, wθ())
        end
        step_rk3!(sim, dt; sgs = cl)
        hold!(sim)
    end
    # the collapsed state: the closure's coefficients and Smagorinsky's on it
    for f in (sim.u, sim.v, sim.w)
        fill_halos!(f, sim.bcs)
    end
    fc = compute_sgs!(cl, sim)
    νs = Field(CellCenter(), mesh); compute_smagorinsky_νt!(νs, sim.u, sim.v, sim.w, 0.17)
    return (t = ts, γ = γs, wθ = fs, ν = ν,
            νh = maximum(interior_view(fc.νh)), νv = maximum(interior_view(fc.νv)),
            νs = maximum(interior_view(νs)), tke = mean(resolved_tke_profile(sim.u, sim.v, sim.w)))
end

@testset "2. decaying turbulence in stable stratification vs neutral (Ri = 1)" begin
    st = sheared_decay(1.0)
    nt = sheared_decay(0.0)
    N = 1.0                                               # N = S = 1: t is in units of N⁻¹
    # the resolved buoyancy flux: down-gradient (w'θ' < 0) while the
    # turbulence mixes, then reversed (counter-gradient, w'θ' > 0) as the
    # displaced parcels spring back — the restratification of decaying
    # stratified turbulence
    i1 = argmin(st.wθ)
    @test st.wθ[i1] < 0
    i2 = findfirst(>(0), st.wθ[i1:end])
    @test i2 !== nothing
    rev = maximum(st.wθ[i1:end])
    @test rev > 0.05 * abs(st.wθ[i1])                     # measured 13 %
    # γ (volume normalisation: the turbulent cells against the held shear)
    # falls below 0.1 within about 10 N⁻¹ (measured 11; the issue's "about
    # 10" is held to ≤ 12)
    ic = findfirst(<(0.1), st.γ)
    @test ic !== nothing && st.t[ic] * N <= 12.0 + 1e-9
    # the neutral control stays turbulent throughout (shear production)
    @test minimum(nt.γ) > 0.9
    # the collapsed state: the σ-model + stability correction has switched
    # the vertical viscosity off (Ri_g > Ri_c, f_m = 0 for Brown et al.
    # 1994); Smagorinsky, on the same field, has not
    @test st.νv / st.ν < 1e-2
    @test st.νs / st.ν > 0.1
    @test st.νs > 100 * st.νv
    # the σ-model's horizontal viscosity (not stability-corrected) is not
    # zero: the residual internal waves are three-dimensional, so σ₃ ≠ 0;
    # it is below Smagorinsky's
    @test st.νh < st.νs
    @info @sprintf("  sheared decay, Ri = 1: w'θ' %.2e → reversed %.2e at t N = %.1f; γ < 0.1 at t N = %.1f (γ(12) = %.3f); neutral min γ %.3f; collapsed max ν_t/ν: σ+corr vertical %.2e, horizontal %.3f, Smagorinsky %.3f; TKE(12) %.2e (neutral %.2e)",
                   st.wθ[i1], rev, st.t[i1 + i2 - 1], st.t[ic], st.γ[end], minimum(nt.γ),
                   st.νv / st.ν, st.νh / st.ν, st.νs / st.ν, st.tke, nt.tke)
end

end
