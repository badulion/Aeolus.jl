# Era VI, Phase 59 (P59-d, #185) — spatially developing flat-plate boundary
# layer: the shared damping profile (sponge + fringe), the Blasius solution,
# the free-stream top boundary, the fringe momentum budget, boundary-layer
# integral diagnostics and the free-stream disturbance generator.
# The simulated plates (laminar plate against Blasius, bypass transition)
# are in validation/era6_flat_plate.jl.

using Test
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
# verbatim as the oracle for the refactored add_sponge!.
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

    # add_sponge! is the Phase-7 kernel on every location and on
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
    @test divmax < 1e-13                    # divergence-free to round-off (|∂u′/∂x| ~ 0.5)
    # L11 of one realisation scatters with the seed (16 seeds: 0.82–1.43 L,
    # standard deviation 0.14 L; seed 1 gives 1.04 L, seed 2 1.14 L), so the
    # calibration is asserted on the ensemble mean of seeds 1–16 (1.027 L,
    # standard error 0.036 L): the 15 % bound is then ~4 standard errors,
    # independent of which seed happens to be drawn.
    L11s = map(1:16) do s
        fs = FreeStreamDisturbance(Tu, L, 400; seed = s)
        us = Field(FaceX(), g); vs = Field(FaceY(), g); ws = Field(FaceZ(), g)
        disturbance_velocity!(us, vs, ws, fs)
        _longitudinal_integral_scale(interior_view(us), g.dx)
    end
    L11_mean = sum(L11s) / length(L11s)
    @printf("    FST: L11 over seeds 1–16 = %.3f–%.3f, mean %.3f (requested %.3f)\n",
            minimum(L11s), maximum(L11s), L11_mean, L)
    @test L11s[1] ≈ L11                     # seed 1 reproduces the measurement above
    @test abs(L11_mean / L - 1) < 0.15

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
