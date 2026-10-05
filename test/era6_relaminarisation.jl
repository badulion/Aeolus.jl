# P59-e (#195): relaminarisation in strongly stable stratification.
#
#   1. laminar stratified Poiseuille flow (closed channel with θ walls, and
#      the flux-cooled open channel): the exact steady states, ν_t ≡ 0 for
#      the laminar-consistent closures, buoyancy does not touch u; the
#      relaminarisation criteria (h/L, L⁺, Ri_b, Re_τ) against their
#      closed forms;
#   2. decaying turbulence on a held laminar shear in stable stratification
#      (Ri = N²/S² = 1) against the neutral control: the resolved buoyancy
#      flux reverses, γ falls below 0.1, the σ-model + stability correction
#      switches its vertical viscosity off, Smagorinsky does not.
#
# Test 3 of the issue (the 32³ open channel at Re_τ = 180, collapse at
# h/L = 2 and survival at h/L = 0.3 over 20 h/u_τ) costs 4.5–9 min per run
# here (6200–9900 SSP-RK3 steps), over the 2-min CI budget the issue sets:
# it is examples/sbl/relaminarisation_channel.jl, which also sweeps h/L at
# Re_τ = 180 and 360 against Nieuwstadt (2005) and Flores & Riley (2011).
# Measured there: h/L = 2 collapses, but so does h/L = 0.3 (critical h/L
# 0.12 at Re_τ = 180 on 32³, 0.47 at Re_τ = 360 on 48³; Nieuwstadt's 1.23
# is for an L without κ, i.e. 1.23 κ ≈ 0.49 with the κ-including L of
# `relaminarisation_criteria`; h/L = 0.5 collapses on 96³ too) — see
# docs/src/theory/transition.md,
# "The cooled open channel".

using Test
using Aeolus
using Aeolus.Fields: interior_view, Field
using Aeolus.Grids: CellCenter
using Aeolus.BoundaryConditions: fill_halos!
using Aeolus.SurfaceExchange: von_karman
using Printf, Random
using Statistics: mean

@testset "P59-e: relaminarisation" begin

@testset "1. laminar stratified Poiseuille: exact, ν_t ≡ 0, u blind to buoyancy" begin
    N = (4, 4, 16)
    grid2 = RegularGrid(; size = N, extent = (2π, π, 2.0))
    lc(cl; g = 1.0, θw = stratified_channel_walls(50.0), stress = :symmetric) =
        channel_case(; Re_τ = 10, N = N, closure = cl, bottom_bc = :noslip, noise = 0.0,
                     θ_walls = θw, g_accel = g, sgs_stress = stress, Pr = 0.71)
    closures = (("none", nothing), ("σ", SigmaModel()),
                ("σ + correction", StabilityCorrected(SigmaModel(), grid2)),
                ("σ + correction, anisotropic", StabilityCorrected(SigmaModel(), grid2;
                                                                   length_scales = :anisotropic)))
    runs = Dict{String,Any}()
    for (name, cl) in closures
        c = lc(cl)
        for _ in 1:3000
            advance_channel!(c, 0.02)
        end
        runs[name] = c
    end
    ref = runs["none"]
    # the closed-channel Poiseuille flow with the stable conduction profile
    # (Ri_τ = 50): the wall stress balances the forcing, u_τ² = 1
    @test ref.τ_wall_bottom ≈ 1 rtol = 1e-6
    @test friction_reynolds(ref.sim; h = 1.0) ≈ 10 rtol = 1e-6       # δ = 1
    for (name, _) in closures[2:end]
        c = runs[name]
        f = compute_sgs!(c.closure, c.sim)
        # laminar-consistent: no eddy viscosity or diffusivity at all
        @test maximum(abs, interior_view(f.νh)) == 0
        @test maximum(abs, interior_view(f.νv)) == 0
        @test maximum(abs, interior_view(f.κv)) == 0
        @test Aeolus.Operators.laminar_consistent(c.closure)
        # and so the flow is the molecular one
        @test maximum(abs, interior_view(c.sim.u) .- interior_view(ref.sim.u)) < 1e-10
        @test maximum(abs, interior_view(c.sim.θ) .- interior_view(ref.sim.θ)) < 1e-10
    end
    # buoyancy does not alter u: θ(z) is balanced by the hydrostatic pressure
    c0 = lc(nothing; g = 0.0)
    for _ in 1:3000
        advance_channel!(c0, 0.02)
    end
    du = maximum(abs, interior_view(c0.sim.u) .- interior_view(ref.sim.u))
    @test du < 1e-10
    # Smagorinsky is not laminar-consistent: ν_t = (C_sΔ)²|S| > 0 in the
    # laminar shear, and its wall stress is the molecular one minus that
    smag = lc(StaticSmagorinsky(0.17); stress = :laplacian)
    for _ in 1:3000
        advance_channel!(smag, 0.02)
    end
    νs = maximum(interior_view(compute_sgs!(smag.closure, smag.sim).νh))
    @test !Aeolus.Operators.laminar_consistent(smag.closure)
    @test νs / smag.sim.ν > 0.1
    @test smag.τ_wall_bottom < 0.9
    @info @sprintf("  laminar Poiseuille (Re_τ = 10, Ri_τ = 50): τ_w %.9f; σ-family ν_t ≡ 0, max|Δu| vs no closure %.1e, vs g = 0 %.1e; Smagorinsky 0.17: max ν_t/ν %.3f, τ_w %.4f",
                   ref.τ_wall_bottom,
                   maximum(maximum(abs, interior_view(runs[n].sim.u) .- interior_view(ref.sim.u))
                           for (n, _) in closures[2:end]), du, νs / smag.sim.ν, smag.τ_wall_bottom)

    # The flux-cooled open channel (P59-e builder hook): Re_τ = 10, the
    # stable laminar flow with a prescribed bottom flux Q < 0. The long-time
    # conduction solution under a constant flux and an insulated top is
    # θ = θ̄(t) + Q (z − h)²/(2κh), θ̄ falling at Q/h; u is the open-channel
    # Poiseuille profile, whatever the buoyancy.
    Q = -0.05; θ₀ = 1000.0
    oc(cl; g = θ₀) = channel_case(; Re_τ = 10, N = N, closure = cl, bottom_bc = :noslip,
                                  walls = :bottom, extent = (2π, π, 1.0), noise = 0.0,
                                  surface_heat_flux = Q, θ₀ = θ₀, g_accel = g, Pr = 0.71,
                                  sgs_stress = :symmetric)
    grid1 = RegularGrid(; size = N, extent = (2π, π, 1.0))
    c1 = oc(StabilityCorrected(SigmaModel(), grid1))
    c0 = oc(nothing; g = 0.0)
    @test c1.sim.has_buoyancy && c1.sim.θ_flux_bottom == Q && c1.sim.bcs.bottom isa NoSlipWall
    @test c1.sim.bcs.top isa Aeolus.BoundaryConditions.ImpermeableWall
    m = c1.sim.mesh
    S0 = sum(interior_view(c1.sim.θ))
    nsteps = 3000; dt = 0.01                     # κ dt/dz² = 0.36 (RK3 limit 0.63)
    for _ in 1:nsteps
        advance_channel!(c1, dt); advance_channel!(c0, dt)
    end
    T = nsteps * dt
    @test c1.q_wall_bottom == Q && c1.q_wall_top == 0
    @test maximum(abs, interior_view(c1.sim.u) .- interior_view(c0.sim.u)) < 1e-10
    @test maximum(abs, interior_view(compute_sgs!(c1.closure, c1.sim).νv)) == 0
    # θ is conserved to the prescribed flux (to the round-off of 9000
    # substep updates of θ ≈ 10³: measured 1.5e-10 relative)
    dS = (sum(interior_view(c1.sim.θ)) - S0) * m.dx * m.dy * m.dz
    @test dS ≈ Q * m.Lx * m.Ly * T rtol = 1e-9
    # the conduction profile (transient e^{-κπ²t/h²} ≈ 1e-18 by t = 30)
    κ = c1.sim.ν / 0.71
    θc = [mean(view(interior_view(c1.sim.θ), :, :, k)) for k in 1:m.Nz]
    zc = [(k - 0.5) * m.dz for k in 1:m.Nz]
    shape = Q .* (zc .- 1).^2 ./ (2κ)
    err = maximum(abs, (θc .- mean(θc)) .- (shape .- mean(shape))) / (abs(Q) / κ)
    @test err < 1e-10
    @test mean(θc) ≈ θ₀ + Q * T rtol = 1e-12
    # u: the open-channel Poiseuille flow, τ_w = u_τ² = 1 in steady state
    @test c1.τ_wall_bottom ≈ 1 rtol = 1e-3
    @info @sprintf("  cooled open channel (Re_τ = 10, Q = %.2f): max|Δu| g on/off %.1e, θ budget rel. error %.1e, conduction-profile error %.1e, τ_w %.6f",
                   Q, maximum(abs, interior_view(c1.sim.u) .- interior_view(c0.sim.u)),
                   abs(dS / (Q * m.Lx * m.Ly * T) - 1), err, c1.τ_wall_bottom)

    # The relaminarisation criteria. h/L = κ g |Q| h/(θ₀ u_τ³) of the
    # nominal u_τ, L⁺ = Re_τ/(h/L), and Ri_b the SurfaceExchange bulk
    # Richardson number of the top/bottom θ and the bulk u.
    for hL in (0.3, 1.23, 2.0)
        Qh = -hL * θ₀ / (von_karman() * θ₀)            # g = θ₀, h = u_τ = 1
        cc = channel_case(; Re_τ = 180, N = (4, 4, 8), closure = nothing, bottom_bc = :noslip,
                          walls = :bottom, extent = (2π, π, 1.0), noise = 0.0,
                          surface_heat_flux = Qh, θ₀ = θ₀, g_accel = θ₀)
        cr = relaminarisation_criteria(cc.sim; u_τ = 1.0, h = 1.0)
        @test cr.h_over_L ≈ hL rtol = 1e-12
        @test cr.L_plus ≈ 180 / hL rtol = 1e-12
        @test cr.L ≈ obukhov_length(1.0, -Qh, θ₀) * 9.80665 / θ₀ rtol = 1e-12   # g = θ₀ instead of 9.80665
        @test cr.Re_τ ≈ 180 rtol = 1e-12
        # a linear θ and a uniform u: Ri_b = g h Δθ/(θ_top U²) of the cell layers
        s = cc.sim; g = s.mesh
        for k in 1:g.Nz
            s.θ.data[:, :, g.Hz+k] .= θ₀ + 2.0 * (k - 0.5) * g.dz
        end
        s.u.data .= 15.0
        cr = relaminarisation_criteria(s; u_τ = 1.0, h = 1.0)
        θb = θ₀ + 2.0 * 0.5 * g.dz; θt = θ₀ + 2.0 * (g.Nz - 0.5) * g.dz
        @test cr.Ri_b ≈ θ₀ * 1.0 * (θt - θb) / (θt * 15.0^2) rtol = 1e-12
    end
    # The κ convention (#236). The L here includes κ; Nieuwstadt (2005)'s
    # does not (h/L_N = g|Q|h/(θ₀ u_τ³)). The flux of his critical
    # h/L_N = 1.23 at Re_τ = 360 is h/L = 1.23 κ = 0.492 here, L⁺ = 732.
    QN = -1.23 * θ₀ / θ₀                                # g|Q|h/(θ₀u_τ³) = 1.23
    cN = channel_case(; Re_τ = 360, N = (4, 4, 8), closure = nothing, bottom_bc = :noslip,
                      walls = :bottom, extent = (2π, π, 1.0), noise = 0.0,
                      surface_heat_flux = QN, θ₀ = θ₀, g_accel = θ₀)
    crN = relaminarisation_criteria(cN.sim; u_τ = 1.0, h = 1.0)
    @test crN.h_over_L ≈ 1.23 * von_karman() rtol = 1e-12
    @test crN.h_over_L ≈ 0.492 rtol = 1e-12                # κ = 0.40
    @test crN.L_plus ≈ 360 / 0.492 rtol = 1e-12
    # neutral: L = ∞
    cn = channel_case(; Re_τ = 180, N = (4, 4, 8), closure = nothing, bottom_bc = :noslip,
                      walls = :bottom, extent = (2π, π, 1.0), noise = 0.0,
                      surface_heat_flux = 0.0, θ₀ = θ₀, g_accel = θ₀)
    @test relaminarisation_criteria(cn.sim; u_τ = 1.0).h_over_L == 0
    @test relaminarisation_criteria(cn.sim; u_τ = 1.0).L_plus == Inf
    # misuse is an error
    nob = channel_case(; Re_τ = 10, N = N, closure = nothing, bottom_bc = :noslip, noise = 0.0)
    @test !nob.sim.has_buoyancy
    @test_throws ArgumentError relaminarisation_criteria(nob.sim)       # no buoyancy (θ₀ = 0)
    @test_throws ArgumentError channel_case(; bottom_bc = :wall_model, surface_heat_flux = -1.0)
    @test_throws ArgumentError channel_case(; bottom_bc = :noslip, surface_heat_flux = -1.0,
                                            θ_walls = (0.0, 1.0))
end

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

end # testset
