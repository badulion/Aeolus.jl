# Era VI, Phase 54 (P54-d, #174): the wall-bounded benchmark harness —
# the pressure-driven half-channel ABL and the two-wall channel case
# builders (WallModels) and their diagnostics (Operators).
#
# Physics asserted: Φ_m of the analytic log law is 1 to O(Δz²) with the
# exact staggered coefficient; E11 satisfies Parseval; the stress profile
# is the flux the solver integrates (its divergence is the plane-mean
# tendency); the half channel's momentum budget (wall stress = u*², total
# stress = u*²(1 - z/H)); the two-wall channel's ⟨τ_w⟩ = u_τ²; and the
# Smagorinsky near-wall overshoot of Mason & Thomson (1992).

using Test
using Printf
using Random
using Aeolus
using Aeolus.SurfaceExchange: von_karman
using Aeolus.WallModels

@testset "Era VI — neutral ABL / channel harness (#174)" begin

@testset "Diagnostics: Φ_m of the analytic log law is 1 + (Δz/z)²/12 + O(Δz⁴)" begin
    κ = von_karman(); us = 0.45; z0 = 1e-3
    for Nz in (16, 32, 64)
        H = 1.0; dz = H / Nz
        zc = [(k - 0.5) * dz for k in 1:Nz]
        U = @. us / κ * log(zc / z0)
        prof = phi_m_profile(U, dz, us; κ = κ)
        @test prof.z ≈ [k * dz for k in 1:Nz-1]
        # exact staggered value (z/Δz) ln((z+Δz/2)/(z-Δz/2)), independent of z0
        exact = [z / dz * log((z + dz / 2) / (z - dz / 2)) for z in prof.z]
        @test maximum(abs.(prof.Φ .- exact)) < 1e-12
        @test abs(prof.Φ[1] - log(3)) < 1e-12
        # second order: Φ - 1 - (Δz/z)²/12 = (Δz/z)⁴/80 + …
        # (2/ε) artanh(ε/2) = 1 + ε²/12 + ε⁴/80 + ε⁶/448 + …, ε = Δz/z: every
        # term is positive, so the remainder after ε²/12 lies in
        # [ε⁴/80, ε⁴/80 + ε⁶/400] for ε ≤ 1/2 (plus 1e-13 of round-off in
        # forming Φ - 1 from a logarithm difference)
        for k in 2:Nz-1
            ε = dz / prof.z[k]
            rem = prof.Φ[k] - 1 - ε^2 / 12
            @test ε^4 / 80 - 1e-13 <= rem <= ε^4 / 80 + ε^6 / 400 + 1e-13
        end
    end
    # at a fixed height the error falls by 4× per halving of Δz
    zf = 0.25
    err = map((8, 16, 32, 64)) do Nz
        dz = 1.0 / Nz
        U = [log(((k - 0.5) * dz) / 1e-4) / κ for k in 1:Nz]
        p = phi_m_profile(U, dz, 1.0; κ = κ)
        abs(p.Φ[findfirst(≈(zf), p.z)] - 1)
    end
    orders = log2.(err[1:end-1] ./ err[2:end])
    @info @sprintf("  Φ_m - 1 at z = H/4: %s; observed orders %s",
                   join((@sprintf("%.2e", e) for e in err), ", "),
                   join((@sprintf("%.3f", o) for o in orders), ", "))
    @test all(o -> abs(o - 2) < 0.05, orders)
    # the Field method uses the plane-mean speed |(⟨u⟩, ⟨v⟩)|
    g = RegularGrid(; size = (8, 6, 16), extent = (1.0, 1.0, 1.0))
    u = Field(FaceX(), g); v = Field(FaceY(), g)
    Random.seed!(3)
    for k in 1:16
        z = (k - 0.5) / 16
        S = log(z / 1e-4) / κ
        noise = randn(8, 6); noise .-= sum(noise) / 48
        u.data[3:10, 3:8, 2+k] .= 0.6S .+ noise
        v.data[3:10, 3:8, 2+k] .= 0.8S
    end
    pf = phi_m_profile(u, v, 1.0; κ = κ)
    @test maximum(abs.(pf.Φ .- [z * 16 * log((z + 1/32) / (z - 1/32)) for z in pf.z])) < 1e-12
end

@testset "Diagnostics: streamwise_spectrum satisfies Parseval to 1e-12" begin
    Random.seed!(11)
    for (Nx, Lx) in ((24, 2π), (25, 3.0))
        g = RegularGrid(; size = (Nx, 7, 5), extent = (Lx, 1.0, 1.0))
        u = Field(FaceX(), g)
        u.data[3:2+Nx, 3:9, 3:7] .= 2 .+ randn(Nx, 7, 5)
        sp = streamwise_spectrum(u, 1:5)
        dk = 2π / Lx
        @test sp.k1 ≈ [n * dk for n in 0:Nx÷2]
        for k in 1:5
            a = u.data[3:2+Nx, 3:9, 2+k]
            var = sum((a .- sum(a; dims = 1) ./ Nx) .^ 2) / length(a)
            @test abs(sum(sp.E11[:, k]) * dk - var) / var < 1e-12
            @test sp.E11[1, k] < 1e-12 * var        # the mean is removed
        end
        sp2 = streamwise_spectrum(u, [2]; remove_mean = false)
        a = u.data[3:2+Nx, 3:9, 4]
        @test abs(sum(sp2.E11) * dk - sum(a .^ 2) / length(a)) / (sum(a .^ 2) / length(a)) < 1e-12
    end
    # a single mode lands in its own bin with energy A²/2
    g = RegularGrid(; size = (32, 4, 3), extent = (4.0, 1.0, 1.0))
    u = Field(FaceX(), g)
    x = [(i - 1) * 4.0 / 32 for i in 1:32]
    u.data[3:34, 3:6, 4] .= 0.7 .* cos.(2π * 5 .* x ./ 4.0)
    sp = streamwise_spectrum(u, [2])
    @test abs(sp.E11[6, 1] * (2π / 4.0) - 0.7^2 / 2) < 1e-14
    @test sum(sp.E11) - sp.E11[6, 1] < 1e-25
end

@testset "Diagnostics: stress profile divergence = plane-mean momentum tendency" begin
    # The profile is the flux the solver integrates: on any velocity and
    # eddy-viscosity field in a periodic-x/y, wall-z box, the plane mean of
    # the advective + eddy-viscous u tendency is ∂τ/∂z = Δτ/Δz exactly
    # (τ is the downward flux, ∂⟨u⟩/∂t = ∂τ/∂z).
    Random.seed!(5)
    g = RegularGrid(; size = (12, 10, 9), extent = (2.0, 1.5, 1.0))
    bcs = FieldBoundaryConditions(:periodic, :periodic, :wall)
    u = Field(FaceX(), g); v = Field(FaceY(), g); w = Field(FaceZ(), g)
    νt = Field(CellCenter(), g)
    ii, jj, kk = Aeolus.Grids.interior(g)
    u.data[ii, jj, kk] .= 3 .+ randn(12, 10, 9)
    v.data[ii, jj, kk] .= randn(12, 10, 9)
    w.data[ii, jj, kk] .= randn(12, 10, 9)
    w.data[ii, jj, first(kk)] .= 0                     # the bottom wall face
    νt.data[ii, jj, kk] .= 0.05 .+ 0.02 .* rand(12, 10, 9)
    for f in (u, v, w, νt); fill_halos!(f, bcs); end
    for comp in (:x, :y)
        tu = Field(FaceX(), g); tv = Field(FaceY(), g); tw = Field(FaceZ(), g)
        momentum_advect!(tu, tv, tw, u, v, w)
        add_eddy_viscosity_u!(tu, u, νt); add_eddy_viscosity_v!(tv, v, νt)
        t = comp === :x ? tu : tv
        prof = total_stress_profile(u, v, w, νt; component = comp)
        mt = [sum(t.data[ii, jj, 2+k]) / 120 for k in 1:9]
        @test maximum(abs.(mt .- diff(prof.total) ./ g.dz)) < 1e-12 * maximum(abs.(mt))
        # nothing crosses the free-slip walls
        @test prof.total[1] == 0 && prof.total[end] == 0
    end
    p = total_stress_profile(u, v, w, νt; ν = 1e-3, τ_wall_bottom = 0.2, τ_wall_top = 0.1)
    @test p.wall[1] == 0.2 && p.wall[end] == -0.1 && all(iszero, p.wall[2:end-1])
    @test p.total ≈ p.resolved .+ p.sgs .+ p.viscous .+ p.wall
    @test p.viscous[2] ≈ 1e-3 * (sum(u.data[ii, jj, 4]) - sum(u.data[ii, jj, 3])) / 120 / g.dz
end

@testset "step_rk3!: body_force! / post_predictor! hooks run in every substage" begin
    g = RegularGrid(; size = (8, 8, 8), extent = (1.0, 1.0, 1.0))
    sim = BoussinesqSimulation(g)
    Random.seed!(2)
    ii, jj, kk = Aeolus.Grids.interior(g)
    sim.u.data[ii, jj, kk] .= randn(8, 8, 8)
    G = 0.37; dt = 1e-3; calls = Ref(0)
    m0 = sum(sim.u.data[ii, jj, kk]) / 512
    step_rk3!(sim, dt; body_force! = (s, h) -> (s.tu.data[ii, jj, kk] .+= G),
              post_predictor! = (s, h) -> (calls[] += 1))
    # each substage adds G·dt; the SSP-RK3 weights 1/6 + 1/6 + 2/3 sum to 1
    @test abs(sum(sim.u.data[ii, jj, kk]) / 512 - m0 - G * dt) < 1e-14
    @test calls[] == 3
    @test sim.t ≈ dt
end

@testset "Builders: initial profile, reserved hooks, options" begin
    κ = von_karman()
    c = half_channel_case(; N = (12, 10, 16), noise = 0.0)
    g = c.sim.mesh
    @test (g.Lx, g.Ly, g.Lz) == (2π, 2π, 1.0)
    @test c.forcing == 1.0 && c.sim.ν == 0 && c.walls === :bottom && c.kind === :half_channel
    @test c.wall.law isa RoughLogLaw && c.wall.law.z0 == 1e-4
    # the initial profile is the law's: U = (u*/κ) ln(z/z0) at the centres,
    # so the first cell carries τ_w = u*² exactly
    U = [c.sim.u.data[3, 3, 2+k] for k in 1:16]
    @test maximum(abs.(U .- [log((k - 0.5) / 16 / 1e-4) / κ for k in 1:16])) < 1e-10
    @test abs(friction_velocity(c.wall.law, U[1], g.dz / 2) - 1) < 1e-12
    ch = channel_case(; Re_τ = 2000, N = 8, noise = 0.0, δ = 0.5, u_τ = 0.2)
    @test ch.sim.ν ≈ 0.2 * 0.5 / 2000 && ch.wall.ν == ch.sim.ν
    @test ch.forcing ≈ 0.2^2 / 0.5 && ch.walls === :both
    Uc = [ch.sim.u.data[3, 3, 2+k] for k in 1:8]
    @test Uc ≈ reverse(Uc)                         # symmetric about the centreline
    @test abs(friction_velocity(ch.wall.law, Uc[1], ch.sim.mesh.dz / 2; ν = ch.sim.ν) - 0.2) < 1e-12
    # reserved hooks of later issues are documented but refuse to run
    # (surface_heat_flux was reserved here; P58-e (#193) implemented it — see era6_collapse.jl)
    @test_throws ArgumentError half_channel_case(; surface_heat_flux = :cooling)
    @test_throws ArgumentError half_channel_case(; bottom_bc = :noslip)
    @test_throws ArgumentError half_channel_case(; θ_walls = (300.0, 290.0))
    @test_throws ArgumentError half_channel_case(; filter = :box)
    # channel_case's bottom_bc = :noslip and θ_walls are implemented by
    # P55-f (#191; exercised by test/era6_relaminarisation.jl since the former
    # test/era6_stratified_channel.jl was removed by ONE-g, #456); only invalid values throw
    @test_throws ArgumentError channel_case(; bottom_bc = :bogus)
    @test_throws ArgumentError channel_case(; θ_walls = (0.0, 1.0))   # θ walls need :noslip
    # a resolved (:noslip) wall has no wall model, so the wall-model keywords
    # of P57-b / P57-c (#183, #189) do not combine with it
    @test_throws ArgumentError channel_case(; bottom_bc = :noslip, h_wm_cells = 2)
    @test_throws ArgumentError channel_case(; bottom_bc = :noslip, filter = :temporal)
    @test_throws ArgumentError channel_case(; bottom_bc = :noslip, wall_model = :nonequilibrium)
    @test_throws ArgumentError channel_case(; walls = :top)
    @test_throws ArgumentError half_channel_case(; formulation = :bogus)
    # closure and wall law are arguments: MOST and a Moeng wall run as given
    m = half_channel_case(; N = 8, wall_law = MOSTWallLaw(1e-3, Inf), formulation = :moeng,
                          closure = nothing)
    @test m.closure === nothing && m.wall.law isa MOSTWallLaw && m.formulation === :moeng
    advance_channel!(m, 1e-3)
    @test m.τ_wall_bottom > 0 && m.nsteps == 1
end

@testset "Harness: the plane-mean momentum budget of every step is exact" begin
    # Lz Δ⟨u⟩ = (G Lz - τ_b - τ_t) Δt per step: the projection, advection
    # and SGS fluxes move no net x-momentum, and the recorded wall stress is
    # the momentum the wall operator removed (SSP-RK3-weighted substages).
    for (mk, stepper) in ((() -> half_channel_case(; N = 12), :rk3),
                          (() -> half_channel_case(; N = 12, stepper = :euler), :euler),
                          (() -> channel_case(; N = (12, 8, 12)), :rk3),
                          (() -> half_channel_case(; N = 12, formulation = :schumann), :rk3))
        c = mk()
        s = c.sim; g = s.mesh
        ii, jj, kk = Aeolus.Grids.interior(g)
        bulk() = sum(view(s.u.data, ii, jj, kk)) / (g.Nx * g.Ny * g.Nz)
        lhs = 0.0; rhs = 0.0; scale = 0.0
        for n in 1:20
            b0 = bulk(); dt = channel_dt(c; cfl = 0.4)
            advance_channel!(c, dt)
            lhs += g.Lz * (bulk() - b0)
            rhs += (c.forcing * g.Lz - c.τ_wall_bottom - c.τ_wall_top) * dt
            scale += c.forcing * g.Lz * dt
        end
        @test abs(lhs - rhs) < 1e-11 * scale
        @test c.stepper === stepper
    end
end

# ---------------------------------------------------------------------------
# The Phase 54 benchmarks (Porté-Agel et al. 2000; Moin & Kim 1982)
# ---------------------------------------------------------------------------

@testset "Half channel 24³, Smagorinsky 0.1: wall stress = u*², τ(z) = u*²(1 - z/H)" begin
    # The issue planned "about 10 H/u*, average over the last 5". That
    # WINDOW was too short, not the code: the 5-10 H/u* mean wall stress
    # over 10 seeded runs ranged 0.946-1.027 u*² (0.922-0.985 over 12 more
    # with noise = 5; the flow still accelerates toward its Smagorinsky
    # equilibrium, and 5 H/u* averages of a 24³ box scatter by ±3 %), so a
    # ±5 % test on it fails for some seeds (seed 4: 0.946). Averaged over 8-20 H/u* the same quantity was 0.961-0.991
    # (12 seeds) and max|τ_tot - u*²(1 - z/H)| ≤ 0.040 u*². The tolerances
    # (±5 %, ±10 %) are the issue's.
    t0 = time()
    hc = half_channel_case(; N = 24, closure = StaticSmagorinsky(0.1))
    run_channel!(hc, 8.0)
    st = run_channel!(hc, 20.0; average_from = 8.0)
    ref = 1 .- st.stress.z
    dev = maximum(abs.(st.stress.total .- ref))
    @info @sprintf("  half channel 24³, C_s = 0.1 (%d steps, %.1f s): ⟨τ_w⟩ = %.4f u*² over 8-20 H/u*; max|τ_tot - u*²(1 - z/H)| = %.3f u*²",
                   st.nsteps, st.wall_time, st.τ_wall, dev)
    @info "  τ_tot / resolved / sgs at z/H = 0, 0.25, 0.5, 0.75: " *
          join((@sprintf("%.3f/%.3f/%.3f", st.stress.total[k], st.stress.resolved[k], st.stress.sgs[k]) for k in (1, 7, 13, 19)), ", ")
    @test abs(st.τ_wall - 1) < 0.05
    # ±10 % of the wall value u*² at every level (a relative tolerance is
    # meaningless near the free-slip top, where τ → 0)
    @test dev < 0.10
    @test st.stress.total[end] == 0                  # nothing leaves through the free-slip top
    @test st.stress.resolved[13] > st.stress.sgs[13] # resolved LES: mid-depth flux is resolved

    # Known failure reproduced: StaticSmagorinsky(0.17), no damping function.
    # From the developed C_s = 0.1 state (at 24³ a C_s = 0.17 start from
    # noise stays laminar-like for > 40 H/u*, with τ_w = 0.68-0.93 u*²),
    # 10 H/u* more, averaged over the last 5. Over 12 seeds max Φ_m of the
    # first three faces was 1.45-1.65, and ⟨τ_w⟩ 0.91-1.06 u*² (the flow
    # re-adjusts to the more dissipative closure).
    hc.closure = StaticSmagorinsky(0.17)
    s2 = run_channel!(hc, 30.0; average_from = 25.0)
    Φ3 = maximum(s2.phi_m[1:3])
    @info @sprintf("  C_s = 0.17, no damping: Φ_m at z/H = 1/24, 2/24, 3/24 = %.3f, %.3f, %.3f (max %.3f); max below z = H/4: %.3f; ⟨τ_w⟩ = %.4f u*²",
                   s2.phi_m[1], s2.phi_m[2], s2.phi_m[3], Φ3, maximum(s2.phi_m[1:6]), s2.τ_wall)
    @test Φ3 > 1.3        # Mason & Thomson 1992; Porté-Agel et al. 2000 Fig. 7: ≈ 1.5-2
    @test abs(s2.τ_wall - 1) < 0.15
    @info @sprintf("  half-channel benchmark wall time: %.1f s", time() - t0)
end

@testset "Two-wall channel 16×8×16 at Re_τ = 1000: ⟨τ_w⟩ = u_τ²" begin
    ch = channel_case(; Re_τ = 1000, N = (16, 8, 16))
    run_channel!(ch, 2.0)                               # 2 δ/u_τ from the law profile + noise
    st = run_channel!(ch, 10.0; average_from = 2.0)
    @info @sprintf("  channel 16×8×16, Re_τ = 1000, C_s = 0.065: ⟨τ_w⟩ over both walls, 2-10 δ/u_τ = %.4f u_τ² (bottom %.4f, top %.4f); %d steps",
                   st.τ_wall, st.τ_wall_bottom, st.τ_wall_top, st.nsteps)
    @test abs(st.τ_wall - 1) < 0.02
    # the total stress is antisymmetric, u_τ²(1 - z/δ), within 0.1 u_τ²
    @test maximum(abs.(st.stress.total .- (1 .- st.stress.z))) < 0.10
end

end # top testset
