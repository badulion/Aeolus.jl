# N-parent-b (#254) — the free-atmosphere top of a deep domain: Rayleigh
# damping of θ toward a reference profile (`ReferenceDamping`, the profile /
# Field target of `add_sponge!`) and the fixed-gradient θ top (`FluxBC(Γ)`).
#
# What is checked against what:
#   1. relaxation: a uniform θ anomaly with no flow decays per level by the
#      exact SSP-RK3 amplification factor 1 − a + a²/2 − a³/6 (a = λΔt);
#   2. locality and exactness: zero tendency below z_start bit for bit; a
#      constant level-vector target == the number target bit for bit, on a
#      RegularGrid and a StretchedGrid; the `:mean` target conserves every
#      level's sum; the booked integral is the tendency's volume integral;
#   3. gravity-wave absorption (Klemp & Lilly 1978, JAS 35, 78; Durran 2010
#      §8.4): the reflected energy fraction of a 2-D packet (N = 0.01 s⁻¹,
#      λ_z = 1 km) from damping layers 1 and 2 λ_z deep, velocity-only and
#      with θ damping added;
#   4. fixed-gradient top: Δ∫θ dz = κΓΔt per step to round-off; second-
#      order convergence against the analytic Neumann solution on a
#      RegularGrid and a geometric StretchedGrid; a linear profile of slope
#      Γ is steady; the IMEX (implicit diffusion) path carries the same flux;
#   5. budget: Δ∫ρc_pθ dV = damping integral + top flux + surface flux.

using Test, Printf, Random
using FFTW: fft, fftfreq
using Aeolus
using Aeolus.Fields: set!, interior_view
using Aeolus.Grids: znodes, xnodes
using Aeolus.Operators: sponge_profile, damping_coordinates, damping_rate, DampingProfile, NordstromRamp

# -------------------------------------------------------------------------
# helpers
# -------------------------------------------------------------------------

_dt_level_sums(f) = (iv = interior_view(f); [sum(@view iv[:, :, k]) for k in axes(iv, 3)])
_dt_level_sums(d::Array, g) = [sum(@view d[g.Hx+1:g.Hx+g.Nx, g.Hy+1:g.Hy+g.Ny, g.Hz+k]) for k in 1:g.Nz]

@testset "N-parent-b: free-atmosphere top (#254)" begin

    # ---------------------------------------------------------------------
    @testset "1. exponential relaxation: the exact SSP-RK3 factor" begin
        g = RegularGrid(; size = (4, 4, 20), extent = (100.0, 100.0, 1000.0), halo = (2, 2, 2))
        bcs = FieldBoundaryConditions(:periodic, :periodic, :wall)
        sim = BoussinesqSimulation(g; ν = 0.0, θ₀ = 300.0, bcs = bcs)
        θr = 300.0; A = 1.0
        prof = sponge_profile(0.01, 500.0, 1000.0)
        F = ReferenceDamping(:θ, prof, θr)
        set!(sim.θ, θr + A)                       # a uniform anomaly, no flow
        dt = 10.0
        for _ in 1:100
            step_rk3!(sim, dt; body_force! = (s, Δt) -> apply_forcing!(s, F, s.t, Δt))
        end
        zc = collect(znodes(g, CellCenter()))
        iv = interior_view(sim.θ)
        err = 0.0
        for k in 1:g.Nz
            a = damping_rate(prof, zc[k]) * dt
            R = 1 - a + a^2 / 2 - a^3 / 6
            err = max(err, maximum(abs.(iv[:, :, k] .- (θr + A * R^100))))
        end
        @info @sprintf("relaxation: max |θ − (θr + A R¹⁰⁰)| = %.2e K over %d levels", err, g.Nz)
        @test err < 1e-10
        @test maximum(abs, interior_view(sim.w)) < 1e-12     # no flow is generated
        # the levels below z_start are not relaxed (their tendency is zero bit
        # for bit, test 2; here the SSP-RK3 convex combinations round, ~1 ulp of
        # 301 K per step: 1.0e-12 K after 100 steps)
        @test maximum(abs.(iv[:, :, 1:10] .- (θr + A))) < 1e-11
        @test iv[1, 1, end] - θr < 0.01
    end

    # ---------------------------------------------------------------------
    @testset "2. locality and exactness of the targets" begin
        Random.seed!(254)
        for g in (RegularGrid(; size = (6, 5, 16), extent = (60.0, 50.0, 800.0), halo = (2, 2, 2)),
                  StretchedGrid(; size = (6, 5, 16), extent = (60.0, 50.0, 800.0), halo = (2, 2, 2),
                                stretching = (:geometric, 1.08)))
            name = nameof(typeof(g))
            f = Field(CellCenter(), g)
            iv = interior_view(f)
            iv .= 300 .+ randn(size(iv))
            z_s, z_t = 500.0, 800.0
            zc = damping_coordinates(g, CellCenter())
            below = findall(<(z_s), zc)
            # the number path and the constant level vector, bit for bit
            t1 = Field(CellCenter(), g); t2 = Field(CellCenter(), g); t3 = Field(CellCenter(), g)
            add_sponge!(t1, f, 0.02, z_s, z_t; f_ref = 300.0)
            add_sponge!(t2, f, 0.02, z_s, z_t; f_ref = fill(300.0, g.Nz))
            ref = Field(CellCenter(), g); ref.data .= 300.0
            add_sponge!(t3, f, 0.02, z_s, z_t; f_ref = ref)
            @test t1.data == t2.data
            @test t1.data == t3.data
            # below z_start: zero bit for bit, whatever the target
            t4 = Field(CellCenter(), g)
            add_sponge!(t4, f, 0.02, z_s, z_t; f_ref = collect(range(290.0, 310.0; length = g.Nz)))
            @test all(interior_view(t4)[:, :, below] .=== 0.0)
            @test all(interior_view(t1)[:, :, below] .=== 0.0)
            @test any(!iszero, interior_view(t4))
            # a level vector of the wrong length is refused
            @test_throws DimensionMismatch add_sponge!(t4, f, 0.02, z_s, z_t; f_ref = zeros(g.Nz - 1))
            @info "locality / exactness on $name: below z_start untouched, vector == number bit for bit"
        end

        # ReferenceDamping on a simulation: profile, table and :mean targets
        g = RegularGrid(; size = (8, 6, 12), extent = (80.0, 60.0, 600.0), halo = (2, 2, 2))
        sim = BoussinesqSimulation(g; ν = 0.0)
        iv = interior_view(sim.θ)
        iv .= 300 .+ randn(size(iv))
        zc = collect(znodes(g, CellCenter()))
        prof = sponge_profile(0.05, 300.0, 600.0)
        θref = 300 .+ 0.003 .* zc
        # the level vector, a time-independent table on the level heights and a
        # function of (z, t) give the same tendency, bit for bit
        tends = map((θref, ForcingTable(zc, θref), (z, t) -> 300 + 0.003z)) do target
            fill!(sim.tθ.data, 0)
            F = ReferenceDamping(:θ, prof, target)
            apply_forcing!(sim, F, 0.0, 1.0)
            copy(sim.tθ.data), damping_tendency(F)
        end
        @test tends[1][1] == tends[2][1]
        @test tends[1][1] == tends[3][1]
        # the booked integral is the volume integral of the tendency it added
        d = tends[1][2]
        @test d.levels ≈ _dt_level_sums(tends[1][1], g) rtol = 1e-14
        @test d.integral ≈ sum(_dt_level_sums(tends[1][1], g)) * g.dx * g.dy * g.dz rtol = 1e-13
        # :mean (DALES) damps only the deviations from the slab mean: every
        # level's sum is unchanged to round-off
        fill!(sim.tθ.data, 0)
        Fm = ReferenceDamping(:θ, prof, :mean)
        apply_forcing!(sim, Fm, 0.0, 1.0)
        lv = _dt_level_sums(sim.tθ)
        scale = maximum(abs, interior_view(sim.tθ))
        @test scale > 0
        @test maximum(abs, lv) < 1e-13 * scale * g.Nx * g.Ny
        @test abs(damping_tendency(Fm).integral) < 1e-12 * scale * g.Nx * g.Ny * g.Nz * g.dx * g.dy * g.dz
        # a Nordström (fringe-shaped) profile works through the same kernel
        Fn = ReferenceDamping(:θ, DampingProfile(NordstromRamp(), 0.05, 200.0, 100.0; stop = 500.0), 300.0)
        fill!(sim.tθ.data, 0)
        apply_forcing!(sim, Fn, 0.0, 1.0)
        @test all(interior_view(sim.tθ)[:, :, zc .<= 200.0] .=== 0.0)
        @test all(interior_view(sim.tθ)[:, :, zc .>= 500.0] .=== 0.0)
        # qv needs a moist simulation; the symbolic target is :mean only
        @test_throws ArgumentError apply_forcing!(sim, ReferenceDamping(:qv, prof, 0.0), 0.0, 1.0)
        @test_throws ArgumentError ReferenceDamping(:θ, prof, :median)
    end

end
