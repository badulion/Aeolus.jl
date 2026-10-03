# Era VI, Phase 57 (P57-c, #189): the non-equilibrium (TBLE) wall model.
# The thin-boundary-layer equations on an embedded wall-normal grid under
# every wall column (Balaras, Benocci & Piomelli 1996; Wang & Moin 2002;
# Kawai & Larsson 2012; Park & Moin 2014), plugged into the shared
# apply_wall_stress! operator.
#
# Physics asserted: the equilibrium limit against Spalding's law and
# against direct quadrature of the ODE; Stokes' first problem with its
# second-order convergence; the laminar pressure-gradient balance
# (τ = τ_w + y ∂p/∂x, Poiseuille) to round-off; exact column momentum
# conservation per step; the stress the operator removes; the channel at
# Re_τ = 1000 against Lee & Moser (2015), with the dynamic κ_wm of Wang &
# Moin (2002); the faster response of the TBLE to a sudden pressure-
# gradient increase (He & Seddighi 2013, qualitative).

using Test
using Printf
using Random
using Aeolus
using Aeolus.WallModels

include(joinpath(@__DIR__, "data", "lee_moser_2015.jl"))

# independent adaptive Simpson quadrature (for the equilibrium-ODE check)
function _simpson(f, a, b; tol = 1e-13, depth = 60)
    c = (a + b) / 2
    fa, fb, fc = f(a), f(b), f(c)
    S = (b - a) / 6 * (fa + 4fc + fb)
    return _simpson_rec(f, a, b, fa, fb, fc, S, tol, depth)
end
function _simpson_rec(f, a, b, fa, fb, fc, S, tol, depth)
    c = (a + b) / 2
    d, e = (a + c) / 2, (c + b) / 2
    fd, fe = f(d), f(e)
    Sl = (c - a) / 6 * (fa + 4fd + fc)
    Sr = (b - c) / 6 * (fc + 4fe + fb)
    (depth <= 0 || abs(Sl + Sr - S) <= 15tol) && return Sl + Sr + (Sl + Sr - S) / 15
    return _simpson_rec(f, a, c, fa, fc, fd, Sl, tol / 2, depth - 1) +
           _simpson_rec(f, c, b, fc, fb, fe, Sr, tol / 2, depth - 1)
end

# A test-only eddy viscosity (the generic at work): the damped mixing
# length in the distance to the nearer wall of a closed channel of height 2δ.
struct ChannelMixingLength <: AbstractTBLEEddyViscosity
    δ::Float64
end
Aeolus.WallModels.tble_eddy_viscosity(m::ChannelMixingLength, y, u_τ, ν, κ, z0) =
    tble_eddy_viscosity(DampedMixingLength(), min(y, 2m.δ - y), u_τ, ν, κ, z0)

@testset "Era VI — non-equilibrium (TBLE) wall model (#189)" begin

@testset "1. Equilibrium limit: Spalding's law, and the ODE against direct quadrature" begin
    ν = 1e-5
    sp = SpaldingLaw()                      # κ = 0.41, B = 5.2
    diffs = Dict{Int,Float64}()
    for hp in (30, 300, 3000)
        h = hp * ν
        U = spalding_uplus(sp, hp)          # the speed for u_τ = 1 at h⁺
        col = TBLEColumn(h)                 # 30 points, r = 1.2
        r = tble_solve!(col, U, 0.0, 0.0, 0.0, Inf; ν = ν, κ = sp.κ, terms = :equilibrium_ode)
        ut = friction_velocity(sp, U, h; ν = ν)
        diffs[hp] = r.u_τ / ut - 1
        # the solution is the equilibrium ODE: U(h) = u_τ² ∫₀ʰ dy/(ν + ν_t)
        eddy = DampedMixingLength()
        Uq = r.u_τ^2 * _simpson(y -> 1 / (ν + tble_eddy_viscosity(eddy, y, r.u_τ, ν, sp.κ, 0.0)), 0.0, h)
        @info @sprintf("  equilibrium h⁺ = %4d: u_τ(TBLE)/u_τ(Spalding) − 1 = %+.3f %%, |U_quad/U − 1| = %.1e (%d iterations)",
                       hp, 100diffs[hp], abs(Uq / U - 1), r.iterations)
        @test abs(Uq / U - 1) < 1e-6
        # τ is constant across the layer (the unforced ODE)
        m = tble_column_momentum(col)
        @test m.τ1_x ≈ r.τx rtol = 1e-12
        @test m.τtop_x ≈ r.τx rtol = 1e-9
        @test r.τy == 0
    end
    @test abs(diffs[300]) < 0.01
    @test abs(diffs[3000]) < 0.01
    # EXPECTATION THAT PROVED WRONG at h⁺ = 30 (kept as a broken test):
    # "within 1 % of Spalding for h⁺ ∈ {30, 300, 3000}". At h⁺ = 30 the
    # matching point is in the buffer layer, where the van Driest-damped
    # ν_t of the issue (A⁺ = 17) and Spalding's fit are two different
    # buffer-layer profiles: −3.0 % here (A⁺ = 19 would give −6.6 %, A⁺ = 25
    # −14 %; A⁺ = 17 is the closest of the usual values). In the log layer
    # (h⁺ = 300, 3000) they agree to 0.003 % and 0.2 %.
    @test abs(diffs[30]) < 0.035
    @test_broken abs(diffs[30]) < 0.01
    # a rough wall (z0 > 0): the equilibrium ODE is the rough log law
    # u_τ = κU/ln((h + z0)/z0)
    col = TBLEColumn(3.125; z0 = 0.05)
    r = tble_solve!(col, 3.0, 1.0, 0.0, 0.0, Inf; ν = 1.5e-5, κ = 0.4, terms = :equilibrium_ode)
    @test r.u_τ ≈ 0.4 * hypot(3.0, 1.0) / log((3.125 + 0.05) / 0.05) rtol = 2e-3
    @test r.τy / r.τx ≈ 1 / 3 rtol = 1e-10            # the stress along the input wind
end

@testset "2. Stokes' first problem (ν_t ≡ 0): τ_w = νU/√(πνt), second order" begin
    ν, U = 1.0, 1.0
    targets = (10, 30, 100, 300, 1000)               # t in units of Δy₀²/ν (Δy₀ = 1)
    function stokes(dy)
        N = round(Int, 250 / dy)
        col = TBLEColumn(N * dy; N = N, r = 1.0)
        col.u .= U; col.u[1] = 0                      # the wall starts at rest under a uniform stream
        dt = 0.1 * dy^2 / ν
        err = Dict{Int,Float64}(); t = 0.0
        for n in 1:round(Int, 1000 / dt)
            tble_solve!(col, U, 0.0, 0.0, 0.0, dt; ν = ν, eddy = NoEddyViscosity())
            t = n * dt
            for tt in targets
                abs(t - tt) < 1e-9 * tt && (err[tt] = col.τx / (ν * U / sqrt(π * ν * t)) - 1)
            end
        end
        return err
    end
    a = stokes(1.0); b = stokes(0.5)                  # Δy halved, Δt quartered
    for tt in targets
        @info @sprintf("  Stokes I t = %4d Δy²/ν: τ_w error %+.2e (Δy), %+.2e (Δy/2), ratio %.2f",
                       tt, a[tt], b[tt], a[tt] / b[tt])
        @test abs(a[tt]) < 0.01 && abs(b[tt]) < 0.01
        @test 3.5 < a[tt] / b[tt] < 4.5
    end
end

@testset "3. Pressure-gradient balance (ν_t ≡ 0): τ = τ_w + y ∂p/∂x, Poiseuille to 1e-10" begin
    ν, h, Ut, dpdx = 1.5e-5, 0.02, 0.7, -0.3
    col = TBLEColumn(h; N = 30, r = 1.2)               # stretched: exactness is not a uniform-grid accident
    r = tble_solve!(col, Ut, 0.0, dpdx, 0.0, Inf; ν = ν, eddy = NoEddyViscosity())
    y = col.y
    uex = @. Ut * y / h + dpdx / (2ν) * (y^2 - y * h)
    @test maximum(abs.(col.u .- uex)) < 1e-10 * maximum(abs.(uex))
    @test r.τx ≈ ν * Ut / h - dpdx * h / 2 rtol = 1e-10
    τmid = [col.G[j] * (col.u[j+1] - col.u[j]) for j in eachindex(col.G)]
    ymid = [(y[j] + y[j+1]) / 2 for j in eachindex(col.G)]
    @test maximum(abs.(τmid .- (r.τx .+ ymid .* dpdx))) < 1e-12 * abs(r.τx)
    # the balance holds whatever ν_t: with the damped mixing length too
    col2 = TBLEColumn(h)
    r2 = tble_solve!(col2, 12.0, 0.0, dpdx, 0.0, Inf; ν = ν, κ = 0.41)
    τ2 = [col2.G[j] * (col2.u[j+1] - col2.u[j]) for j in eachindex(col2.G)]
    ym2 = [(col2.y[j] + col2.y[j+1]) / 2 for j in eachindex(col2.G)]
    @test maximum(abs.(τ2 .- (r2.τx .+ ym2 .* dpdx))) < 1e-12 * abs(r2.τx)
    # a favourable gradient raises the wall stress of the same U(h)
    col3 = TBLEColumn(h)
    @test tble_solve!(col3, 12.0, 0.0, 0.0, 0.0, Inf; ν = ν, κ = 0.41).τx < r2.τx
    # time-marching from rest reaches the steady Poiseuille state
    col4 = TBLEColumn(h; N = 30, r = 1.2)
    for _ in 1:400
        tble_solve!(col4, Ut, 0.0, dpdx, 0.0, 2.0; ν = ν, eddy = NoEddyViscosity())
    end
    @test maximum(abs.(col4.u .- uex)) < 1e-8 * maximum(abs.(uex))
end

@testset "4. Column momentum conservation to round-off, every step" begin
    rng = Random.Xoshiro(7)
    ν, h = 1e-4, 0.1
    for terms in (:pressure_gradient, :full)
        col = TBLEColumn(h)
        tble_solve!(col, 1.0, 0.0, 0.0, 0.0, Inf; ν = ν, terms = :equilibrium_ode)
        worst = 0.0
        for n in 1:200
            Ut, Vt = 1 + 0.5 * randn(rng), 0.3 * randn(rng)
            dpx, dpy = 0.2 * randn(rng), 0.2 * randn(rng)
            dt = 10.0^(-3 + 3rand(rng))
            if terms === :full
                col.cx .= 0.1 .* randn(rng, length(col.y)); col.cy .= 0.1 .* randn(rng, length(col.y))
            end
            m0 = tble_column_momentum(col)
            tble_solve!(col, Ut, Vt, dpx, dpy, dt; ν = ν, κ = 0.4, terms = terms)
            m1 = tble_column_momentum(col)
            y = col.y; N = length(col.G)
            V = [(y[j+1] - y[j-1]) / 2 for j in 2:N]
            Fx = -dpx .* sum(V) - (terms === :full ? sum(V .* col.cx[2:N]) : 0.0)
            Fy = -dpy .* sum(V) - (terms === :full ? sum(V .* col.cy[2:N]) : 0.0)
            rx = (m1.Mx - m0.Mx) / dt - (m1.τtop_x - m1.τ1_x + Fx)
            ry = (m1.My - m0.My) / dt - (m1.τtop_y - m1.τ1_y + Fy)
            scale = abs(m1.τtop_x) + abs(m1.τ1_x) + abs(Fx) + abs(m1.Mx / dt) + abs(m0.Mx / dt)
            worst = max(worst, (abs(rx) + abs(ry)) / scale)
        end
        @info @sprintf("  column momentum (%s): max |ΔM/Δt − (τ_top − τ_1 + ΣV F)| / scale = %.1e over 200 random steps",
                       terms, worst)
        @test worst < 1e-12
    end
end

@testset "Operator: the TBLE stress is applied by apply_wall_stress!, both walls" begin
    g = RegularGrid(; size = (8, 6, 8), extent = (4.0, 3.0, 2.0), halo = (2, 2, 2))
    sim = BoussinesqSimulation(g; bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
    Aeolus.Fields.interior_view(sim.u) .= 2.0
    Aeolus.Fields.interior_view(sim.v) .= 1.0
    wm = TBLEWallModel(; ν = 1e-4, terms = :pressure_gradient, pressure = false)
    A = g.Lx * g.Ly
    F = apply_wall_stress!(sim, wm, 1e-5; side = :both)
    cols, act = tble_columns(wm, :bottom)
    c = cols[1, 1]
    @test all(act) && c.y[end] ≈ g.dz / 2
    # primed with the equilibrium solution of the first-cell wind: Spalding-like
    @test c.τy / c.τx ≈ 0.5 rtol = 1e-10
    # what the operator removed is the columns' stress (both walls; implicit:
    # 1 + Δt a/Δz = 1 + 3e-7)
    @test -F.Fx / A ≈ 2c.τx rtol = 1e-6
    @test -F.Fy / A ≈ 2c.τy rtol = 1e-6
    # the stress of an equilibrium column equals the law with the same κ, A⁺
    U = hypot(2.0, 1.0)
    col = TBLEColumn(g.dz / 2)
    @test tble_solve!(col, 2.0, 1.0, 0, 0, Inf; ν = 1e-4, κ = wm.κ, terms = :equilibrium_ode).u_τ ≈ c.u_τ rtol = 1e-12
    @test tble_kappa(wm) == wm.κ
    # a uniform imposed favourable gradient raises the stress over a step
    τ0 = c.τx
    update_wall_model!(wm, sim, 0.05; side = :both, dpdx = -1.0)
    @test cols[1, 1].τx > τ0
    @test tble_columns(wm, :top).cols[1, 1].τx ≈ cols[1, 1].τx rtol = 1e-12
    @test_throws ArgumentError apply_wall_stress!(sim, wm, 0.1; formulation = :schumann)
    @test_throws ArgumentError TBLEWallModel(; terms = :bogus)
end

end
