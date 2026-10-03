# Era VI, Phase 59 (P59-d, #185) — spatially developing flat-plate boundary
# layer: the shared damping profile (sponge + fringe), the Blasius solution,
# the fringe region, the free-stream disturbance generator, boundary-layer
# integral diagnostics and bypass transition.

using Aeolus
using Aeolus.Fields: set!, interior_view
using Aeolus.Operators
using Aeolus.Operators: add_sponge!, smooth_step
using Aeolus.Operators.FiniteVolume: in_damping
using Aeolus.StretchedGrids
using Random
using Printf

# ---------------------------------------------------------------------------
# 0. The damping profile shared by add_sponge! and the fringe
# ---------------------------------------------------------------------------

# The Phase-7 sponge kernel as it stood before the shared profile, kept here
# verbatim as the bit-for-bit oracle for the refactored add_sponge!.
function _phase7_sponge_reference!(tf, f, α_max, z_start, z_top, f_ref, zs)
    m = Aeolus.Fields.mesh(f)
    ii, jj, kk = Aeolus.Grids.interior(m)
    inv_width = 1 / (z_top - z_start)
    half_π = 0.5 * π
    for (kk_, kv) in enumerate(kk), j in jj, i in ii
        z = zs[kk_]
        if z >= z_start
            s = sin(half_π * (z - z_start) * inv_width)
            α = α_max * s * s
            tf.data[i, j, kv] -= α * (f.data[i, j, kv] - f_ref)
        end
    end
    return tf
end

@testset "P59-d damping profile (sponge + fringe share one implementation)" begin
    # Smooth step of Nordström et al. (1999), eq. (2.3).
    @test smooth_step(-0.3) == 0 && smooth_step(0.0) == 0
    @test smooth_step(1.0) == 1 && smooth_step(1.7) == 1
    @test smooth_step(0.5) ≈ 0.5 atol = 1e-15
    for x in (0.01, 0.2, 0.37, 0.8, 0.99)
        @test smooth_step(x) + smooth_step(1 - x) ≈ 1 atol = 1e-14   # antisymmetry
    end
    xs = range(0, 1; length = 1001)
    @test all(diff(smooth_step.(xs)) .>= 0)                          # monotone

    # Nordström fringe λ(x): zero outside, λ_max on the plateau.
    p = DampingProfile(NordstromRamp(), 2.0, 10.0, 3.0; stop = 30.0, fall = 2.0)
    @test damping_rate(p, 9.0) == 0 && damping_rate(p, 31.0) == 0
    @test damping_rate(p, 10.0) == 0 && damping_rate(p, 30.0) == 0
    @test damping_rate(p, 13.0) ≈ 2.0 atol = 1e-15
    @test damping_rate(p, 20.0) ≈ 2.0 atol = 1e-15
    @test damping_rate(p, 28.0) ≈ 2.0 atol = 1e-15
    @test damping_rate(p, 11.5) ≈ 1.0 atol = 1e-14    # mid-rise: S(1/2) = 1/2
    @test damping_rate(p, 29.0) ≈ 1.0 atol = 1e-14    # mid-fall

    # add_sponge! is bit for bit the Phase-7 kernel on every location and on
    # Float64 and Float32 grids (the rate now comes from sponge_profile).
    for T in (Float64, Float32)
        g = RegularGrid(T; size = (6, 5, 24), extent = (1, 1, 2))
        for L in (FaceX(), FaceY(), FaceZ(), CellCenter())
            rng = MersenneTwister(7)
            f = Field(L, g); tf = Field(L, g)
            f.data .= T.(randn(rng, size(f.data)))
            tf.data .= T.(randn(rng, size(tf.data)))
            tf_ref = deepcopy(tf)
            zs = collect(Aeolus.Grids.znodes(g, L))
            add_sponge!(tf, f, 0.37, T(1.3), T(2.0); f_ref = 0.2)
            _phase7_sponge_reference!(tf_ref, f, 0.37, T(1.3), T(2.0), 0.2, zs)
            @test tf.data == tf_ref.data
        end
    end
    # The sin² profile equals the Phase-7 formula pointwise.
    sp = sponge_profile(0.5, 1.0, 3.0)
    for z in (0.5, 1.0, 1.7, 2.9, 3.0)
        ref = z >= 1.0 ? 0.5 * sin(0.5π * (z - 1.0) * (1 / 2.0))^2 : 0.0
        @test damping_rate(sp, z) === ref
    end
    @test !in_damping(sp, 0.99) && in_damping(sp, 1.0)
end

# ---------------------------------------------------------------------------
# 1. Blasius (RK4 shooting + secant, in-package)
# ---------------------------------------------------------------------------

@testset "P59-d Blasius solution" begin
    s = blasius_solution()
    # Howarth (1938) / Schlichting & Gersten (2017, Table 6.1): f″(0) = 0.332057,
    # η₉₉ = 4.91, δ* constant 1.7208, θ constant 0.664, H = 2.59.
    @test s.fpp0 ≈ 0.332057 atol = 1e-6
    @test s.eta99 ≈ 4.91 atol = 1e-2
    @test s.delta_star ≈ 1.7208 atol = 1e-4
    @test abs(s.residual) < 1e-12                        # far-field f′ = 1 met
    # Momentum-integral identity for the zero-pressure-gradient plate:
    # dθ/dx = c_f/2 ⇒ θ constant = 2 f″(0) (an independent quadrature check).
    @test s.theta ≈ 2s.fpp0 rtol = 1e-10
    @test s.H ≈ 2.5911 atol = 1e-4
    # Converged in the step and in η_max.
    @test abs(blasius_solution(n = 6000).fpp0 - s.fpp0) < 1e-12
    @test abs(blasius_solution(η_max = 15, n = 15000).delta_star - s.delta_star) < 1e-10
    # The interpolant satisfies the ODE f‴ = −½ f f″ (finite difference of f″).
    for η in (0.4, 1.3, 2.75, 4.1, 6.0)
        h = 1e-4
        fppp = (blasius_profile(η + h)[3] - blasius_profile(η - h)[3]) / 2h
        f, _, fpp = blasius_profile(η)
        @test fppp ≈ -0.5f * fpp atol = 1e-8
    end
    # Tabulated values (Howarth 1938): f′(1) = 0.32979, f′(3) = 0.84605, f″(3) = 0.16136.
    @test blasius_profile(1.0)[2] ≈ 0.32979 atol = 1e-5
    @test blasius_profile(3.0)[2] ≈ 0.84605 atol = 1e-5
    @test blasius_profile(3.0)[3] ≈ 0.16136 atol = 1e-5
    @test blasius_profile(20.0) == (20.0 - s.delta_star, 1.0, 0.0)
    # Velocity: u = U f′, the far-field w → 0.8604 U/√Re_x (= ½ δ*_c U/√Re_x).
    ν = 1.5e-5; U = 2.0; x = 0.3
    u, w = blasius_velocity(x, 0.0; U∞ = U, ν = ν)
    @test u == 0 && w == 0
    u, w = blasius_velocity(x, 0.05; U∞ = U, ν = ν)
    @test u ≈ U atol = 1e-12
    @test w ≈ 0.5 * s.delta_star * U / sqrt(U * x / ν) rtol = 1e-10
    @test blasius_skin_friction(1e5) ≈ 0.664 / sqrt(1e5) rtol = 2e-4
    @printf("    Blasius: f″(0) = %.10f, δ*_c = %.10f, θ_c = %.10f, η99 = %.5f, H = %.5f (%d secant its)\n",
            s.fpp0, s.delta_star, s.theta, s.eta99, s.H, s.iterations)
end

# ---------------------------------------------------------------------------
# 2. The free-stream (transpiration) top boundary
# ---------------------------------------------------------------------------

@testset "P59-d FreeStreamBoundary" begin
    g = RegularGrid(size = (32, 4, 16), extent = (8.0, 1.0, 2.0))
    xs = [g.x0 + (i - g.Hx - 0.5) * g.dx for i in 1:(g.Nx + 2g.Hx)]
    wb = 0.01 .* sin.(2π .* xs ./ g.Lx)                     # zero net transpiration
    top = FreeStreamBoundary(wb)
    bcs = FieldBoundaryConditions(:periodic, :periodic; bottom = :noslip, top = top)
    @test bcs.top === top
    @test FieldBoundaryConditions(:periodic, :periodic; bottom = :wall, top = top).top === top
    @test_throws ArgumentError FieldBoundaryConditions(:periodic, :periodic; bottom = top, top = :noslip)
    @test_throws ArgumentError FieldBoundaryConditions(:periodic, :periodic; bottom = :noflux, top = top)
    @test_throws ArgumentError FieldBoundaryConditions(:periodic, :periodic; bottom = top, top = top)

    # Halo fill: the prescribed w on the top face, zero shear (mirror) for u.
    w = Field(FaceZ(), g); u = Field(FaceX(), g)
    rng = MersenneTwister(3)
    w.data .= randn(rng, size(w.data)); u.data .= randn(rng, size(u.data))
    fill_halos!(w, bcs); fill_halos!(u, bcs)
    kw = g.Hz + g.Nz + 1
    @test all(w.data[i, j, kw] == wb[i] for i in axes(w.data, 1), j in axes(w.data, 2))
    @test all(w.data[:, :, kw+1] .≈ 2 .* wb .- w.data[:, :, kw-1])
    @test u.data[:, :, g.Hz+g.Nz+1] == u.data[:, :, g.Hz+g.Nz]
    @test all(w.data[:, :, g.Hz+1] .== 0)                     # the no-slip bottom
    @test_throws DimensionMismatch fill_halos!(w, FieldBoundaryConditions(:periodic, :periodic;
                                       bottom = :noslip, top = FreeStreamBoundary(wb[1:end-1])))

    # The projection makes the field divergence-free *with* the transpiration
    # and leaves the prescribed face velocity untouched.
    sim = BoussinesqSimulation(g; ν = 1e-3, bcs = bcs)
    sim.u.data .= 1 .+ 0.1 .* randn(rng, size(sim.u.data))
    step!(sim, 1e-3)
    @test max_divergence(sim) < 1e-12
    fill_halos!(sim.w, sim.bcs)
    @test sim.w.data[:, 1, kw] == wb
end

# ---------------------------------------------------------------------------
# 3. Fringe region: momentum budget (one explicit step, exact)
# ---------------------------------------------------------------------------

# Interior sum of a field times the cell volume.
_vsum(f, g) = sum(interior_view(f)) * g.dx * g.dy * g.dz

@testset "P59-d fringe momentum budget" begin
    for top in (:wall, :freestream)
        c = flat_plate_setup(; Nx = 64, Nz = 16, Lx = 128, H = 8, fringe_length = 40,
                             rise = 10, fall = 8, Re_δ0 = 150, top = top)
        sim, g = c.sim, c.sim.mesh
        sim.u.data .= 1.0                                  # impulsively started plate
        rng = MersenneTwister(11)
        sim.w.data .+= 1e-3 .* randn(rng, size(sim.w.data))
        step_rk3!(sim, 0.5; body_force! = c.forcing!)      # a developed, non-trivial state
        # Terms of d/dt ∫u dV over the next explicit step, all on the pre-step state:
        fill_halos!(sim.u, sim.bcs); fill_halos!(sim.v, sim.bcs); fill_halos!(sim.w, sim.bcs)
        tu = Field(FaceX(), g); tv = Field(FaceY(), g); tw = Field(FaceZ(), g)
        add_fringe!(tu, tv, tw, sim.u, sim.v, sim.w, c.fringe, c.target)
        F_fringe = _vsum(tu, g)
        τw = wall_shear_stress(sim; side = :bottom)        # kinematic, +∂u/∂z at the wall
        A = g.Lx * g.Ly
        # momentum carried out through the top face: u_face w_top with the
        # mirror u halo (u_face = u_N), w averaged to the u point
        flux_top = 0.0
        if top === :freestream
            wb = sim.bcs.top.w_top; kN = g.Hz + g.Nz
            for j in (g.Hy+1):(g.Hy+g.Ny), i in (g.Hx+1):(g.Hx+g.Nx)
                flux_top += sim.u.data[i, j, kN] * 0.5 * (wb[i-1] + wb[i]) * g.dx * g.dy
            end
        end
        M0 = _vsum(sim.u, g)
        dt = 0.25
        step!(sim, dt; body_force! = c.forcing!)
        dMdt = (_vsum(sim.u, g) - M0) / dt
        resid = dMdt - (F_fringe - τw * A - flux_top)
        scale = abs(F_fringe) + abs(τw * A) + abs(flux_top)
        @printf("    budget (%s top): dM/dt = %.10f, fringe = %.10f, -τA = %.10f, -top flux = %.10f, residual = %.2e\n",
                top, dMdt, F_fringe, -τw * A, -flux_top, resid)
        @test abs(F_fringe) > 1e-2 && abs(τw * A) > 1e-2       # both terms active
        @test abs(resid) < 1e-10
        @test abs(resid) / scale < 1e-12
    end
end

# ---------------------------------------------------------------------------
# 4. Laminar plate (Tu = 0, 2-D, Re_x 1.35e4 → 1e5, NoSlipWall from P59-a)
# ---------------------------------------------------------------------------
#
# 256 × 1 × 64 on 512 × 16 δ*₀, dx = 2 δ*₀, dz = δ*₀/4, Re_δ*₀ = 200; RK3 at
# CFL 0.5, 1200 steps (2.3 flow-through times) from a uniform stream, so the
# boundary layer develops from the fringe inflow instead of being imposed.

@testset "P59-d laminar flat plate" begin
    c = flat_plate_setup()
    sim, g = c.sim, c.sim.mesh
    sim.u.data .= 1.0
    t0 = time()
    for n in 1:1200
        step_rk3!(sim, 1.0; body_force! = c.forcing!)
    end
    wall = time() - t0
    @test max_divergence(sim) < 1e-12
    x = collect(Aeolus.Grids.xnodes(g, FaceX()))
    Lp = g.Lx - (c.fringe.profile.stop - c.fringe.profile.start)     # plate length
    mid = findall(xi -> 0.2Lp <= xi <= 0.8Lp, x)
    Re_x = (x .- c.x_le) ./ c.ν
    cf = skin_friction(sim; U_e = 1.0)
    H = shape_factor(sim)
    ratio = cf[mid] ./ blasius_skin_friction.(Re_x[mid])
    @printf("    laminar plate (%.0f s): c_f/c_f,Blasius ∈ [%.4f, %.4f] over Re_x ∈ [%.0f, %.0f]; H ∈ [%.3f, %.3f]\n",
            wall, minimum(ratio), maximum(ratio), Re_x[mid[1]], Re_x[mid[end]],
            minimum(H[mid]), maximum(H[mid]))
    @test all(abs.(ratio .- 1) .< 0.05)
    @test all(abs.(H[mid] .- 2.59) .< 0.1)
    # The plane mean of τ_w(x) is the P59-a wall_shear_stress (one definition).
    τx = cf .* 0.5                                            # U_e = 1
    @test sum(τx) / length(τx) ≈ wall_shear_stress(sim) rtol = 1e-12
    # Fringe: the target is recovered at the fringe exit (x = 0) to 1 % of U.
    ub = streamwise_profile(sim)
    ut = interior_view(c.target.u)[:, 1, :]
    err_exit = maximum(abs, ub[1, :] .- ut[1, :])
    @printf("    fringe exit: max|u − u_target| = %.2e U; inflow δ* = %.4f δ*₀, θ = %.4f δ*₀\n",
            err_exit, displacement_thickness(ub, g.dz)[1], momentum_thickness(ub, g.dz)[1])
    @test err_exit < 0.01
    # The free stream enters the plate at U∞ (the transpiration top keeps
    # it from accelerating by Δδ*/H).
    Ue = edge_velocity(ub)
    @test maximum(abs.(Ue[mid] .- 1)) < 0.005
end

@testset "P59-d integral diagnostics on analytic profiles" begin
    # Sampled Blasius profiles at three stations: δ*, θ, H, c_f from the
    # midpoint-rule quadrature match the similarity constants.
    ν = 1 / 200; dz = 0.025; Nz = 800
    xs = [100.0, 300.0, 500.0]
    ū = [blasius_velocity(x, (k - 0.5) * dz; ν = ν)[1] for x in xs, k in 1:Nz]
    s = blasius_solution()
    δs = displacement_thickness(ū, dz; U_e = 1.0)
    θ = momentum_thickness(ū, dz; U_e = 1.0)
    for (n, x) in enumerate(xs)
        @test δs[n] ≈ s.delta_star * sqrt(ν * x) rtol = 1e-4
        @test θ[n] ≈ s.theta * sqrt(ν * x) rtol = 1e-4
    end
    @test all(abs.(shape_factor(ū, dz; U_e = 1.0) .- s.H) .< 1e-3)
    cf = skin_friction(ū, dz, ν; U_e = 1.0)
    @test all(abs.(cf ./ blasius_skin_friction.(xs ./ ν) .- 1) .< 1e-3)
    @test edge_velocity(ū) ≈ ones(3) atol = 1e-12
    # δ99 edge: the first level with ū ≥ 0.99 max ū sits at η99 = 4.91.
    ke = boundary_layer_edge(ū)
    for (n, x) in enumerate(xs)
        @test abs((ke[n] - 0.5) * dz - s.eta99 * sqrt(ν * x)) <= dz
    end
    # Integrating to the δ99.5 edge with U_e = U∞ costs < 0.5 % of θ (the
    # Blasius tail; measured 0.33 %).
    θe = momentum_thickness(ū, dz; U_e = 1.0, kmax = boundary_layer_edge(ū; fraction = 0.995))
    @test all(0 .< (θ .- θe) ./ θ .< 5e-3)

    # Onset finder on a synthetic laminar → turbulent c_f distribution: the
    # minimum of c_f upstream of the persistent departure from laminar.
    Re = range(2e4, 6e5; length = 400)
    γ = @. 0.5 * (1 + tanh((Re - 2.5e5) / 3e4))
    cfs = @. (1 - γ) * blasius_skin_friction(Re) + γ * turbulent_skin_friction(Re)
    xo, io = transition_onset(collect(Re), cfs, blasius_skin_friction.(Re))
    @test io == argmin(cfs)
    @test 1.8e5 < xo < 2.5e5
    @test isequal(transition_onset(collect(Re), blasius_skin_friction.(Re), blasius_skin_friction.(Re)), (NaN, 0))
    # The correlations at the T3A level Tu = 3.3 % (issue #185: ≈ 190 and ≈ 200).
    @test onset_reynolds_theta_mayle(3.3) ≈ 189.7 atol = 0.5
    @test onset_reynolds_theta_abu_ghannam_shaw(3.3) ≈ 200.0 atol = 0.5
    @test turbulent_skin_friction(1e6) ≈ 0.0576 / 10^1.2 rtol = 1e-12
end

# ---------------------------------------------------------------------------
# 5. Free-stream disturbances (random Fourier modes, von Kármán spectrum)
# ---------------------------------------------------------------------------

_VK_TAIL(κ) = 1.606 * κ^(-2 / 3)      # ∫_κ^∞ (2/3)a x^{-5/3} dx, the far tail of E

# Longitudinal correlation f(r) = ⟨u(x)u(x+r)⟩/⟨u²⟩ along x (array index 1),
# integrated (trapezoid) up to its first zero crossing.
function _longitudinal_integral_scale(ui, dx)
    Nx = size(ui, 1); f = zeros(Nx ÷ 2)
    for r in 0:(Nx÷2 - 1)
        s = 0.0; n = 0
        @inbounds for k in axes(ui, 3), j in axes(ui, 2), i in 1:(Nx-r)
            s += ui[i, j, k] * ui[i+r, j, k]; n += 1
        end
        f[r+1] = s / n
    end
    f ./= f[1]
    iz = findfirst(<(0), f)
    return dx * (sum(f[1:iz-1]) - 0.5)
end

@testset "P59-d FreeStreamDisturbance" begin
    # Spectrum normalisation (Brandt et al. 2004): ∫E dκ = q and
    # L11 = (π/2u'²)∫E/κ dκ = 0.643 L (trapezoid on a log grid).
    κ = exp.(range(log(1e-4), log(1e4); length = 200_001))
    E = von_karman_spectrum.(κ, 1.0, 1.0)
    trap(y) = sum(0.5 .* (y[1:end-1] .+ y[2:end]) .* diff(κ))
    # the κ^{-5/3} tail beyond κ = 1e4 carries a·(1e4)^{-2/3} = 3.5e-3 of q
    @test trap(E) + _VK_TAIL(1e4) ≈ 1.0 rtol = 1e-3
    @test (π / 2) * trap(E ./ κ) / (2 / 3) ≈ 0.643 rtol = 2e-3

    Tu, L = 0.05, 1.0
    fst = FreeStreamDisturbance(Tu, L, 400; seed = 1)
    @test length(fst.A) == 400
    @test all(abs(sum(fst.k[:, m] .* fst.σ[:, m])) < 1e-12 for m in 1:400)   # σ ⊥ k
    @test sum(fst.A .^ 2) / 4 ≈ 1.5 * Tu^2 rtol = 1e-12                    # energy = q

    g = RegularGrid(size = (512, 32, 32), extent = (64.0, 4.0, 4.0))
    u = Field(FaceX(), g); v = Field(FaceY(), g); w = Field(FaceZ(), g)
    disturbance_velocity!(u, v, w, fst)
    d = Field(CellCenter(), g)
    Aeolus.Operators.divergence(d, u, v, w)
    ui = interior_view(u); vi = interior_view(v); wi = interior_view(w)
    rms(a) = sqrt(sum(abs2, a) / length(a))
    Tu_m = sqrt((rms(ui)^2 + rms(vi)^2 + rms(wi)^2) / 3)
    divmax = maximum(abs, interior_view(d))
    L11 = _longitudinal_integral_scale(ui, g.dx)
    @printf("    FST: Tu = %.4f (requested %.4f), u/v/w rms = %.4f/%.4f/%.4f, L11 = %.3f (requested %.3f), max|∇·u′| = %.1e\n",
            Tu_m, Tu, rms(ui), rms(vi), rms(wi), L11, L, divmax)
    @test abs(Tu_m / Tu - 1) < 0.10
    @test abs(L11 / L - 1) < 0.15
    @test divmax < 1e-13                    # divergence-free to round-off (|∂u′/∂x| ~ 0.5)

    # Spanwise quantisation: periodic in y with the domain width.
    fy = FreeStreamDisturbance(Tu, L, 100; seed = 2, period_y = 4.0)
    @test all(abs(mod(fy.k[2, m] / (2π / 4.0) + 0.5, 1) - 0.5) < 1e-12 for m in 1:100)
    u2 = Field(FaceX(), g); v2 = Field(FaceY(), g); w2 = Field(FaceZ(), g)
    disturbance_velocity!(u2, v2, w2, fy)
    @test maximum(abs, u2.data[:, 1, :] .- u2.data[:, g.Ny+1, :]) < 1e-12
    # Frozen advection: the pattern at t is the t = 0 pattern shifted by U t.
    u3 = Field(FaceX(), g); v3 = Field(FaceY(), g); w3 = Field(FaceZ(), g)
    disturbance_velocity!(u3, v3, w3, fy, 8 * g.dx)               # U∞ = 1
    @test maximum(abs, u3.data[20:200, :, :] .- u2.data[12:192, :, :]) < 1e-12
end

# ---------------------------------------------------------------------------
# 6. Bypass sanity (coarse 3-D): Tu = 6 % transitions, Tu = 0 stays laminar
# ---------------------------------------------------------------------------
#
# Re_δ*₀ = 300, 400 × 36 × 24 δ*₀ on 96 × 24 × 48 (dx = 4.2, dy = 1.5,
# dz = 0.5 δ*₀), 80 δ*₀ fringe, plate Re_x = 3.0e4 → 1.26e5 at the fringe
# start; c_f is averaged over the last 200 of 700 steps (Δt = 1.5) in the
# window 0.75–0.9 of the plate, Re_x ≈ 1.03e5–1.17e5. No SGS model (a coarse DNS:
# the Smagorinsky ν_t of the laminar wall shear alone would be ≈ 4ν here).
# Free-stream turbulence (L11 = 10 δ*₀, 200 modes, κ_max = 2π/3Δ) enters
# through the fringe target, kept out of the inflow layer below 1.5 δ₉₉.
# The Tu = 0 run is the same plate with Ny = 1: without a seed the 3-D run
# stays exactly 2-D, so the 2-D run is the same calculation at 1/24 the cost.

function _bypass_run(Tu; Ny, nsteps = 700, navg = 200, dt = 1.5, Lx = 400.0)
    c = flat_plate_setup(; Re_δ0 = 300, Lx = Lx, Ly = 36, H = 24, Nx = round(Int, Lx / 4.1667),
                         Ny = Ny, Nz = 48, fringe_length = 80)
    sim, g = c.sim, c.sim.mesh
    force! = c.forcing!
    if Tu > 0
        fst = FreeStreamDisturbance(Tu, 10.0, 200; seed = 1, period_y = g.Ly,
                                    κ_max = 2π / (3 * max(g.dx, g.dy, g.dz)))
        δ99 = blasius_solution().eta99 * sqrt(c.ν * c.x_in)
        tgt = deepcopy(c.target)
        upd = disturbance_fringe_update(fst, c.fringe, c.target;
                                        damping = z -> smooth_step(z / (1.5δ99)))
        force! = fringe_forcing(c.fringe, tgt; update! = upd)
    end
    # both runs start from the laminar Blasius layer: the disturbances must trip it
    set_blasius!(sim; x_le = c.x_le, x_min = c.x_in)
    cf = zeros(g.Nx)
    for n in 1:nsteps
        step_rk3!(sim, dt; body_force! = force!)
        n > nsteps - navg && (cf .+= skin_friction(sim; U_e = 1.0))
    end
    cf ./= navg
    x = collect(Aeolus.Grids.xnodes(g, FaceX()))
    Lp = g.Lx - 80
    out = findall(xi -> 0.75Lp <= xi <= 0.9Lp, x)
    Re_x = (x[out] .- c.x_le) ./ c.ν
    return sum(cf[out]) / sum(blasius_skin_friction.(Re_x)), max_divergence(sim),
           (Re_x[1], Re_x[end])
end

@testset "P59-d bypass transition sanity (coarse 3-D)" begin
    t0 = time()
    r6, div6, rng = _bypass_run(0.06; Ny = 24)
    t6 = time() - t0
    t0 = time()
    r0, div0, _ = _bypass_run(0.0; Ny = 1)
    t00 = time() - t0
    @printf("    bypass: outlet c_f / c_f,Blasius (Re_x %.0f–%.0f) = %.3f at Tu = 6 %% (%.0f s), %.4f at Tu = 0 (%.0f s)\n",
            rng[1], rng[2], r6, t6, r0, t00)
    @test r6 > 1.5
    @test abs(r0 - 1) < 0.05
    @test div6 < 1e-12 && div0 < 1e-12
end
