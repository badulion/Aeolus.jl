# Era VI, Phase 59 (P59-d, #185) — spatially developing flat-plate boundary
# layer: the shared damping profile (sponge + fringe), the Blasius solution,
# the fringe region, the free-stream disturbance generator, boundary-layer
# integral diagnostics and bypass transition.

using Aeolus
using Aeolus.Fields: set!, interior_view
using Aeolus.Operators
using Aeolus.Operators: add_sponge!
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
