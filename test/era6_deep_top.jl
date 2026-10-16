# N-parent-b (#254) — the free-atmosphere top of a deep domain: Rayleigh
# damping of θ toward a reference profile (`ReferenceDamping`, the profile /
# Field target of `add_sponge!`) and the fixed-gradient θ top (`FluxBC(Γ)`).
#
# What is checked against what:
#   1. relaxation: a uniform θ anomaly with no flow decays per level by the
#      exact SSP-RK3 amplification factor 1 − a + a²/2 − a³/6 (a = λΔt);
#   2. locality and exactness: zero tendency below z_start; a
# constant level-vector target == the number target, on a
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

# The 2-D gravity-wave packet of test 3. Returns the reflected energy
# fraction: the time-integrated energy of the downward-propagating part
# (m > 0 for k > 0, ω > 0) over the time-integrated energy of the upward part
# (m < 0), both from the vertical wavenumber spectrum of the complex x-mode
# amplitude of w in a window below the damping layer. Both packets cross the
# window at the same |c_gz| (a reflection keeps k, ω and |m|), so the ratio
# of the time integrals is the energy ratio.
function _gw_reflection(; depth, θdamp, α, dz = 50.0, dt = 20.0, t_end = 22_000.0,
                        z_s = 6000.0, z_packet = 2000.0, σ = 600.0, window = (3000.0, 5500.0))
    N = 0.01; λ = 1000.0; Nx = 16
    k = 2π / λ; m = -2π / λ                          # m < 0: upward group velocity
    ω = N * k / sqrt(k^2 + m^2)
    H = z_s + depth
    Nz = round(Int, H / dz)
    g = RegularGrid(; size = (Nx, 2, Nz), extent = (λ, 2λ / Nx, H), halo = (2, 2, 2))
    θ0 = 300.0; ga = 9.81
    bcs = FieldBoundaryConditions(:periodic, :periodic, :wall)
    sim = BoussinesqSimulation(g; ν = 0.0, θ₀ = θ0, g_accel = ga, bcs = bcs, θ_bcs = bcs,
                               sponge_α_max = α, sponge_z_start = z_s)
    θbar(z) = θ0 * (1 + N^2 * z / ga)
    A = 1e-3
    env(z) = exp(-(z - z_packet)^2 / (2σ^2))
    # velocity from a stream function on the C-grid: discretely divergence-free
    ψ(x, z) = (A / k) * env(z) * sin(k * x + m * z)
    set!(sim.u, (x, y, z) -> -(ψ(x, z + dz / 2) - ψ(x, z - dz / 2)) / dz)
    set!(sim.w, (x, y, z) -> (ψ(x + g.dx / 2, z) - ψ(x - g.dx / 2, z)) / g.dx)
    # b = g θ'/θ0 = (N²/ω) A sin φ (linear polarization of the upward wave)
    set!(sim.θ, (x, y, z) -> θbar(z) + θ0 / ga * (N^2 / ω) * A * env(z) * sin(k * x + m * z))
    zc = collect(znodes(g, CellCenter()))
    F = ReferenceDamping(:θ, sponge_profile(α, z_s, H), θbar.(zc))
    bf! = θdamp ? ((s, Δt) -> apply_forcing!(s, F, s.t, Δt)) : nothing
    kr = findall(z -> window[1] <= z <= window[2], zc)
    nk = length(kr)
    hann = [0.5 - 0.5cos(2π * (n - 0.5) / nk) for n in 1:nk]
    ms = 2π .* fftfreq(nk, 1 / dz)
    xs = collect(xnodes(g, CellCenter()))
    Hx, Hy, Hz = g.Hx, g.Hy, g.Hz
    Eup = 0.0; Edn = 0.0
    W = zeros(ComplexF64, nk)
    nsteps = round(Int, t_end / dt)
    for n in 0:nsteps
        if n % 5 == 0
            for (q, kk) in enumerate(kr)
                s = 0.0im
                for i in 1:Nx
                    wc = 0.5 * (sim.w.data[Hx+i, Hy+1, Hz+kk] + sim.w.data[Hx+i, Hy+1, Hz+kk+1])
                    s += wc * cis(-k * xs[i])
                end
                W[q] = 2s / Nx * hann[q]
            end
            Wm = fft(W)
            for i in 1:nk
                ms[i] < 0 ? (Eup += abs2(Wm[i])) : ms[i] > 0 ? (Edn += abs2(Wm[i])) : nothing
            end
        end
        n == nsteps && break
        step_rk3!(sim, dt; body_force! = bf!)
    end
    return Edn / Eup
end

# Analytic solution of θ_t = κθ_zz on [0, L], θ_z(0) = 0, θ_z(L) = Γ,
# θ(z, 0) = Γz²/(2L) + a cos(πz/L).
_neumann_exact(z, t; κ, Γ, L, a) = Γ * (z^2 / (2L) + κ * t / L) + a * cos(π * z / L) * exp(-κ * π^2 * t / L^2)

# Forward-Euler diffusion θ ← θ + Δt κ ∇²θ on any mesh (fill_halos! + laplacian),
# from the analytic initial condition to t_end; returns the max error at the cell
# centres and the per-step check of the heat content Σθ Δz_k.
function _neumann_run(g, κ, Γ, a, t_end, dt)
    L = g.Lz
    bcs = FieldBoundaryConditions(:periodic, :periodic; bottom = :noflux, top = FluxBC(Γ))
    θ = Field(CellCenter(), g); lap = Field(CellCenter(), g)
    zc = collect(damping_coordinates(g, CellCenter()))
    Δz = hasproperty(g, :dzc) ? [g.dzc[g.Hz+k] for k in 1:g.Nz] : fill(g.dz, g.Nz)
    iv = interior_view(θ)
    for k in 1:g.Nz
        iv[:, :, k] .= _neumann_exact(zc[k], 0.0; κ, Γ, L, a)
    end
    nsteps = round(Int, t_end / dt)
    dt = t_end / nsteps
    worst = 0.0
    for n in 1:nsteps
        H0 = sum(iv[1, 1, k] * Δz[k] for k in 1:g.Nz)
        fill_halos!(θ, bcs)
        laplacian(lap, θ)
        iv .+= dt * κ .* interior_view(lap)
        H1 = sum(iv[1, 1, k] * Δz[k] for k in 1:g.Nz)
        worst = max(worst, abs((H1 - H0) - κ * Γ * dt) / (κ * Γ * dt))
    end
    err = maximum(abs(iv[1, 1, k] - _neumann_exact(zc[k], t_end; κ, Γ, L, a)) for k in 1:g.Nz)
    return err, worst
end

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
            # the number path and the constant level vector
            t1 = Field(CellCenter(), g); t2 = Field(CellCenter(), g); t3 = Field(CellCenter(), g)
            add_sponge!(t1, f, 0.02, z_s, z_t; f_ref = 300.0)
            add_sponge!(t2, f, 0.02, z_s, z_t; f_ref = fill(300.0, g.Nz))
            ref = Field(CellCenter(), g); ref.data .= 300.0
            add_sponge!(t3, f, 0.02, z_s, z_t; f_ref = ref)
            @test agree(t1.data, t2.data)
            @test agree(t1.data, t3.data)
            # below z_start: zero, whatever the target
            t4 = Field(CellCenter(), g)
            add_sponge!(t4, f, 0.02, z_s, z_t; f_ref = collect(range(290.0, 310.0; length = g.Nz)))
            @test all(interior_view(t4)[:, :, below] .=== 0.0)
            @test all(interior_view(t1)[:, :, below] .=== 0.0)
            @test any(!iszero, interior_view(t4))
            # a level vector of the wrong length is refused
            @test_throws DimensionMismatch add_sponge!(t4, f, 0.02, z_s, z_t; f_ref = zeros(g.Nz - 1))
            @info "locality / exactness on $name: below z_start untouched, vector == number"
        end

        # TerrainFollowingGrid: the one kernel sampled at the 3-D physical
        # heights equals the per-cell sin² formula (the pre-#254 loop)
        sg = StretchedGrid(; size = (8, 6, 12), extent = (800.0, 600.0, 1200.0), halo = (2, 2, 2))
        tg = terrain_following_grid(sg, (x, y) -> 150 * exp(-((x - 400)^2) / 200^2))
        f = Field(CellCenter(), tg)
        f.data .= 300 .+ randn(size(f.data))
        tt = Field(CellCenter(), tg)
        add_sponge!(tt, f, 0.03, 700.0, 1200.0; f_ref = 300.0)
        zs = physical_heights(tg, CellCenter())
        oracle = zeros(size(f.data))
        ii, jj, kk = Aeolus.Grids.interior(tg)
        for k in kk, j in jj, i in ii
            z = zs[i, j, k]
            z >= 700.0 || continue
            sn = sin(0.5π * (z - 700.0) * (1 / (1200.0 - 700.0)))
            oracle[i, j, k] -= 0.03 * sn * sn * (f.data[i, j, k] - 300.0)
        end
        @test agree(tt.data, oracle)
        @test count(!iszero, tt.data) > 0

        # ReferenceDamping on a simulation: profile, table and :mean targets
        g = RegularGrid(; size = (8, 6, 12), extent = (80.0, 60.0, 600.0), halo = (2, 2, 2))
        sim = BoussinesqSimulation(g; ν = 0.0)
        iv = interior_view(sim.θ)
        iv .= 300 .+ randn(size(iv))
        zc = collect(znodes(g, CellCenter()))
        prof = sponge_profile(0.05, 300.0, 600.0)
        θref = 300 .+ 0.003 .* zc
        # the level vector, a time-independent table on the level heights and a
        # function of (z, t) give the same tendency
        tends = map((θref, ForcingTable(zc, θref), (z, t) -> 300 + 0.003z)) do target
            fill!(sim.tθ.data, 0)
            F = ReferenceDamping(:θ, prof, target)
            apply_forcing!(sim, F, 0.0, 1.0)
            copy(sim.tθ.data), damping_tendency(F)
        end
        @test values_agree(tends[1][1], tends[2][1])
        @test values_agree(tends[1][1], tends[3][1])
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

    # ---------------------------------------------------------------------
    @testset "3. gravity-wave absorption (Klemp & Lilly 1978)" begin
        # rigid lid, no damping: the measurement sees the full reflection
        R_lid = _gw_reflection(; depth = 1000.0, θdamp = false, α = 0.0)
        α = 1e-3                                   # λ_max = 0.1 N ≈ ω/7
        R = Dict{Tuple{Float64,Bool},Float64}()
        for depth in (1000.0, 2000.0), θd in (false, true)
            R[(depth, θd)] = _gw_reflection(; depth = depth, θdamp = θd, α = α)
        end
        @info @sprintf("gravity-wave reflection, α_max = %.0e s⁻¹: lid %.3f; depth 1 λz: uvw %.4f, +θ %.4f; depth 2 λz: uvw %.4f, +θ %.4f",
                       α, R_lid, R[(1000.0, false)], R[(1000.0, true)], R[(2000.0, false)], R[(2000.0, true)])
        @test R_lid > 0.9
        for depth in (1000.0, 2000.0)
            @test R[(depth, true)] < 0.05                     # the issue's target
            @test R[(depth, true)] < R[(depth, false)] / 4    # θ damping absorbs what velocity damping lets through
        end
        @test R[(2000.0, true)] < R[(1000.0, true)]
    end

    # ---------------------------------------------------------------------
    @testset "4. fixed-gradient top" begin
        Γ = 0.003
        # the ghost continues a linear profile of slope Γ exactly into every halo layer
        for g in (RegularGrid(; size = (4, 4, 8), extent = (40.0, 40.0, 200.0), halo = (3, 3, 3)),
                  StretchedGrid(; size = (4, 4, 8), extent = (40.0, 40.0, 200.0), halo = (3, 3, 3),
                                stretching = (:geometric, 1.2)))
            f = Field(CellCenter(), g)
            zst = hasproperty(g, :zc) ? g.zc : [g.z0 + (k - g.Hz - 0.5) * g.dz for k in 1:g.Nz+2g.Hz]
            for k in 1:g.Nz
                f.data[:, :, g.Hz+k] .= 290 + Γ * zst[g.Hz+k]
            end
            fill_halos!(f, FieldBoundaryConditions(:periodic, :periodic; bottom = :noflux, top = FluxBC(Γ)))
            @test maximum(abs(f.data[3, 3, g.Hz+g.Nz+h] - (290 + Γ * zst[g.Hz+g.Nz+h])) for h in 1:g.Hz) < 1e-12
            @test f.data[3, 3, g.Hz] == f.data[3, 3, g.Hz+1]          # the no-flux bottom
        end
        # pairs: a FluxBC top above a wall or no-flux bottom only
        @test FieldBoundaryConditions(:periodic, :periodic; bottom = :wall, top = FluxBC(Γ)).top isa FluxBC
        @test FieldBoundaryConditions(:periodic, :periodic; bottom = :noslip, top = FluxBC(Γ)).top isa FluxBC
        @test_throws ArgumentError FieldBoundaryConditions(:periodic, :periodic; bottom = FluxBC(Γ), top = :noflux)
        @test_throws ArgumentError FieldBoundaryConditions(:periodic, :periodic; bottom = FluxBC(Γ), top = FluxBC(Γ))
        gr = RegularGrid(; size = (4, 4, 8), extent = (40.0, 40.0, 200.0), halo = (2, 2, 2))
        @test_throws ArgumentError fill_halos!(Field(FaceZ(), gr),
            FieldBoundaryConditions(:periodic, :periodic; bottom = :noflux, top = FluxBC(Γ)))

        # heat content grows by κΓΔt per step, to round-off (Boussinesq core, step!)
        g = RegularGrid(; size = (6, 6, 20), extent = (60.0, 60.0, 400.0), halo = (2, 2, 2))
        κ = 1.5
        θbcs = FieldBoundaryConditions(:periodic, :periodic; bottom = :noflux, top = FluxBC(Γ))
        sim = BoussinesqSimulation(g; ν = κ, θ_bcs = θbcs, bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
        set!(sim.θ, (x, y, z) -> 0.5 * sin(2π * x / 60) + 0.001 * z)
        dt = 2.0
        worst = 0.0
        for _ in 1:50
            S0 = sum(interior_view(sim.θ)) * g.dz / (g.Nx * g.Ny)
            step!(sim, dt)
            S1 = sum(interior_view(sim.θ)) * g.dz / (g.Nx * g.Ny)
            worst = max(worst, abs((S1 - S0) - κ * Γ * dt) / (κ * Γ * dt))
        end
        @info @sprintf("top flux, step!: max |Δ∫θdz − κΓΔt|/(κΓΔt) = %.2e over 50 steps", worst)
        @test worst < 1e-10
        # the same on the IMEX path (implicit diffusion: the FluxBC is the affine part)
        sim2 = BoussinesqSimulation(g; ν = κ, θ_bcs = θbcs, bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
        set!(sim2.θ, (x, y, z) -> 0.5 * sin(2π * x / 60) + 0.001 * z)
        worst2 = 0.0
        dt_imex = 100.0
        dt_expl = 1 / (2κ * (1 / g.dx^2 + 1 / g.dy^2 + 1 / g.dz^2))     # forward-Euler diffusion limit
        for _ in 1:10
            S0 = sum(interior_view(sim2.θ)) * g.dz / (g.Nx * g.Ny)
            step_imex!(sim2, dt_imex, ARK2(); tol = 1e-13)
            S1 = sum(interior_view(sim2.θ)) * g.dz / (g.Nx * g.Ny)
            worst2 = max(worst2, abs((S1 - S0) - κ * Γ * dt_imex) / (κ * Γ * dt_imex))
        end
        @info @sprintf("top flux, step_imex! (Δt = %.0f s, %.1f× the explicit diffusion limit): max rel. error %.2e",
                       dt_imex, dt_imex / dt_expl, worst2)
        @test worst2 < 1e-8

        # qv shares θ's walls but not θ's prescribed top gradient: a uniform qv
        # stays uniform under diffusion while θ takes up the top flux
        simq = BoussinesqSimulation(g; ν = κ, θ_bcs = θbcs, moisture = true,
                                    bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
        set!(simq.θ, 1.0); set!(simq.qv, 0.008)
        for _ in 1:10
            step!(simq, dt)
        end
        @test maximum(abs.(interior_view(simq.qv) .- 0.008)) < 1e-15
        @test sum(interior_view(simq.θ) .- 1.0) * g.dz / (g.Nx * g.Ny) ≈ 10 * κ * Γ * dt rtol = 1e-10

        # a linear profile of slope Γ is steady (the surface takes the κΓ the top gives)
        sim3 = BoussinesqSimulation(g; ν = κ, θ_bcs = θbcs, θ_flux_bottom = -κ * Γ,
                                    bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
        set!(sim3.θ, (x, y, z) -> 1.0 + Γ * z)
        θi = copy(interior_view(sim3.θ))
        for _ in 1:100
            step!(sim3, dt)
        end
        drift = maximum(abs.(interior_view(sim3.θ) .- θi))
        @info @sprintf("linear profile of slope Γ after 100 steps: max drift %.2e K", drift)
        @test drift < 1e-12

        # second-order convergence against the analytic Neumann solution
        κn, Γn, an, tn = 1.0, 1.0, 0.5, 0.05
        for (label, mk) in (("RegularGrid", Nz -> RegularGrid(; size = (2, 2, Nz), extent = (1.0, 1.0, 1.0), halo = (2, 2, 2))),
                            ("StretchedGrid (geometric)", Nz -> StretchedGrid(; size = (2, 2, Nz), extent = (1.0, 1.0, 1.0),
                                                                             halo = (2, 2, 2), stretching = (:geometric, 1.1^(16 / Nz)))))
            errs = Float64[]; flux = 0.0
            for Nz in (16, 32, 64)
                g = mk(Nz)
                dzmin = hasproperty(g, :dzc) ? minimum(g.dzc[g.Hz+1:g.Hz+Nz]) : g.dz
                e, w = _neumann_run(g, κn, Γn, an, tn, 0.2 * dzmin^2 / κn)
                push!(errs, e); flux = max(flux, w)
            end
            r1, r2 = errs[1] / errs[2], errs[2] / errs[3]
            @info @sprintf("Neumann convergence, %s: errors %.3e %.3e %.3e, ratios %.2f %.2f; flux error %.1e",
                           label, errs..., r1, r2, flux)
            @test 3.5 < r1 < 4.5
            @test 3.5 < r2 < 4.5
            @test flux < 1e-9
        end
    end

    # ---------------------------------------------------------------------
    @testset "5. budget: Δ∫ρc_pθ dV = damping + top flux + surface flux" begin
        Random.seed!(2541)
        g = RegularGrid(; size = (8, 8, 16), extent = (160.0, 160.0, 320.0), halo = (2, 2, 2))
        ρ, cp = 1.2, 1004.0
        κ, Γ, Fs = 0.5, 0.004, 0.1                 # m²/s, K/m, K m/s (≈ 120 W/m²)
        θbcs = FieldBoundaryConditions(:periodic, :periodic; bottom = :noflux, top = FluxBC(Γ))
        sim = BoussinesqSimulation(g; ν = κ, θ₀ = 300.0, θ_flux_bottom = Fs, θ_bcs = θbcs,
                                   bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
        set!(sim.θ, (x, y, z) -> 300 + 0.003z + 0.2 * randn())
        set!(sim.u, (x, y, z) -> 0.5 * randn()); set!(sim.v, (x, y, z) -> 0.5 * randn())
        zc = collect(znodes(g, CellCenter()))
        F = ReferenceDamping(:θ, sponge_profile(0.01, 200.0, 320.0), 300 .+ 0.003 .* zc)
        ΔV = g.dx * g.dy * g.dz
        A = g.Lx * g.Ly
        heat(s) = ρ * cp * sum(interior_view(s.θ) .- 300) * ΔV
        dt = 1.0
        worst = 0.0; Dsum = 0.0
        for n in 1:30
            H0 = heat(sim)
            step!(sim, dt; body_force! = (s, Δt) -> apply_forcing!(s, F, s.t, Δt))
            D = damping_tendency(F).integral
            Dsum += D * dt
            expected = ρ * cp * dt * (D + κ * Γ * A + Fs * A)
            worst = max(worst, abs((heat(sim) - H0) - expected) / abs(expected))
        end
        @info @sprintf("budget: max |ΔH − (D + top + surface)|/|ΔH| = %.2e over 30 steps (damping %.3e J, top %.3e J, surface %.3e J per step)",
                       worst, ρ * cp * Dsum / 30, ρ * cp * κ * Γ * A * dt, ρ * cp * Fs * A * dt)
        @test worst < 1e-10
        @test Dsum < 0                             # the anomaly above z_start is damped
    end
end
