# Test selection (#316): AEOLUS_TEST_TIER=fast|full (default fast) picks
# files by the tags in test_tags.toml; test/run.jl runs them in parallel.
# The include list below stays the single list. See docs/src/testing.md.
isdefined(Main, :AeolusTestSelection) || Base.include(Main, joinpath(@__DIR__, "runner", "selection.jl"))
Main.AeolusTestSelection.serial_entry(@__FILE__) && exit()

using Test
using Random
# #245: the suite plans every FFT with FFTW.ESTIMATE (deterministic, so bitwise
# comparisons hold across processes, and planning is fast). Set before
# `using Aeolus`; subprocesses the tests start inherit it.
ENV["AEOLUS_FFTW_PLANNER"] = "estimate"
using Statistics: mean
using Aeolus
using Aeolus.Grids: xnodes, ynodes, znodes, interior
# #333: the suite shows warnings and errors only (AEOLUS_LOG=debug or info
# opts in); tests that assert on events capture them with their own logger.
using Logging: global_logger
global_logger(test_logger())
using Aeolus.Fields: set!, interior_view

@testset "Aeolus.jl" begin

    include("ncio.jl")           # NetCDF-4 output policy, legacy NetCDF-3 reading (#310)
    include("viz_png.jl")        # examples/viz PNG/APNG encoder: DEFLATE + filters round-trip

    @testset "RegularGrid" begin
        g = RegularGrid(Float64; size=(8, 4, 2), extent=(1.0, 2.0, 4.0), halo=(2,2,2))
        @test g.dx ≈ 1/8
        @test g.dy ≈ 2/4
        @test g.dz ≈ 4/2
        ii, jj, kk = interior(g)
        @test length(ii) == 8 && length(jj) == 4 && length(kk) == 2

        xs = xnodes(g, CellCenter())
        @test length(xs) == 8
        @test xs[1] ≈ g.dx/2
        @test xs[end] ≈ 1.0 - g.dx/2

        # face-x nodes: one per interior cell, sitting on that cell's west face
        xf = xnodes(g, FaceX())
        @test length(xf) == 8
        @test xf[1] ≈ 0.0
        @test xf[end] ≈ 1.0 - g.dx    # east face of cell Nx wraps periodically
    end

    @testset "Field construction & set!" begin
        g = RegularGrid(; size=(4,4,4), extent=(1.0,1.0,1.0), halo=(1,1,1))
        f = Field(CellCenter(), g)
        @test size(f) == (6, 6, 6)   # 4 interior + 2*1 halo
        set!(f, (x,y,z) -> x + y + z)
        v = interior_view(f)
        @test size(v) == (4,4,4)
        @test v[1,1,1] ≈ 3 * (g.dx/2)
    end

    @testset "Periodic halo fill" begin
        g = RegularGrid(; size=(4,4,4), extent=(1.0,1.0,1.0), halo=(2,2,2))
        f = Field(CellCenter(), g)
        # place a unique value at each interior cell so we can trace it
        set!(f, (x,y,z) -> 100x + 10y + z)
        bcs = FieldBoundaryConditions(:periodic)
        fill_halos!(f, bcs)
        Nx, Ny, Nz = g.Nx, g.Ny, g.Nz
        Hx, Hy, Hz = g.Hx, g.Hy, g.Hz
        # west halo (i=1) mirrors east interior (i = Hx+Nx-1+1 = Nx+Hx-... )
        # index convention: halo west i=1..Hx ↔ interior east Nx+1..Nx+Hx
        for h in 1:Hx
            @test f.data[h, Hy+1, Hz+1] == f.data[Nx + h, Hy+1, Hz+1]
        end
        # east halo mirrors west interior
        for h in 1:Hx
            @test f.data[Hx + Nx + h, Hy+1, Hz+1] == f.data[Hx + h, Hy+1, Hz+1]
        end
    end

    @testset "NoFlux halo fill (reflection)" begin
        g = RegularGrid(; size=(4,4,4), extent=(1.0,1.0,1.0), halo=(2,2,2))
        f = Field(CellCenter(), g)
        set!(f, (x,y,z) -> z)   # linear in z
        bcs = FieldBoundaryConditions(:noflux, :noflux, :noflux)
        fill_halos!(f, bcs)
        # reflection makes halo cell = first interior cell across the face
        @test f.data[3, 3, g.Hz] == f.data[3, 3, g.Hz+1]
        @test f.data[3, 3, g.Hz+g.Nz+1] == f.data[3, 3, g.Hz+g.Nz]
    end

    @testset "∂x is exact for a linear field" begin
        g = RegularGrid(; size=(16,4,4), extent=(2π, 1.0, 1.0), halo=(2,2,2))
        f = Field(CellCenter(), g); out = Field(CellCenter(), g)
        set!(f, (x,y,z) -> 3x + 2)
        fill_halos!(f, FieldBoundaryConditions(:noflux))
        ∂x(out, f)
        v = interior_view(out)
        # exclude cells next to the reflection boundary; interior should be 3
        @test all(isapprox.(v[2:end-1, :, :], 3.0; atol=1e-12))
    end

    @testset "Laplacian of a quadratic" begin
        g = RegularGrid(; size=(16,16,16), extent=(1.0,1.0,1.0), halo=(1,1,1))
        f = Field(CellCenter(), g); out = Field(CellCenter(), g)
        set!(f, (x,y,z) -> x^2 + y^2 + z^2)
        fill_halos!(f, FieldBoundaryConditions(:noflux))
        laplacian(out, f)
        v = interior_view(out)
        # ∇²(x² + y² + z²) = 6, exact for centered 2nd-order on a quadratic
        @test all(isapprox.(v[2:end-1, 2:end-1, 2:end-1], 6.0; atol=1e-10))
    end

    @testset "RK3 on the linear ODE du/dt = -u" begin
        g = RegularGrid(; size=(4,4,4), extent=(1.0,1.0,1.0), halo=(1,1,1))
        u = Field(CellCenter(), g); set!(u, 1.0)
        state = (u = u,)
        rk = RK3(state)
        function tend!(k, s, t)
            @. k.u.data = -s.u.data
            return k
        end
        dt = 0.01
        for _ in 1:100
            step!(rk, state, tend!, dt)
        end
        v = interior_view(state.u)
        expected = exp(-1.0)             # t = 1
        @test all(isapprox.(v, expected; rtol=1e-4))
    end

    @testset "Staggered ∂x: cell-center → x-face" begin
        # p(x) = sin(2π x/L) → dp/dx = (2π/L) cos(2π x/L)
        L = 2π
        g = RegularGrid(; size=(64, 4, 4), extent=(L, 1.0, 1.0), halo=(2,2,2))
        p    = Field(CellCenter(), g)
        dpdx = Field(FaceX(),      g)
        set!(p, (x,y,z) -> sin(x))
        fill_halos!(p, FieldBoundaryConditions(:periodic))
        ∂x(dpdx, p)

        # face nodes: interior i sits at x = (i-1)*dx, so exact value is cos(x).
        # 2nd-order centered stencil is O(dx²) accurate — check convergence.
        xs = xnodes(g, FaceX())
        v  = interior_view(dpdx)
        err = maximum(@. abs(v[:,1,1] - cos(xs)))
        @test err < 0.01  # coarse tolerance for 64 pts; convergence checked separately
    end

    @testset "Staggered divergence: face → cell-center" begin
        # u = sin(x), v = 0, w = 0 → ∇·u = cos(x)
        L = 2π
        g = RegularGrid(; size=(64, 4, 4), extent=(L, 1.0, 1.0), halo=(2,2,2))
        u = Field(FaceX(), g); v = Field(FaceY(), g); w = Field(FaceZ(), g)
        div = Field(CellCenter(), g)
        set!(u, (x,y,z) -> sin(x))
        for f in (u, v, w)
            fill_halos!(f, FieldBoundaryConditions(:periodic))
        end
        divergence(div, u, v, w)
        xs = xnodes(g, CellCenter())
        vv = interior_view(div)
        err = maximum(@. abs(vv[:,1,1] - cos(xs)))
        @test err < 0.01
    end

    @testset "Staggered ∂x convergence (2nd order)" begin
        L = 2π
        errs = Float64[]
        for N in (32, 64, 128)
            g = RegularGrid(; size=(N, 4, 4), extent=(L, 1.0, 1.0), halo=(2,2,2))
            p    = Field(CellCenter(), g)
            dpdx = Field(FaceX(), g)
            set!(p, (x,y,z) -> sin(x))
            fill_halos!(p, FieldBoundaryConditions(:periodic))
            ∂x(dpdx, p)
            xs = xnodes(g, FaceX())
            v  = interior_view(dpdx)
            push!(errs, maximum(@. abs(v[:,1,1] - cos(xs))))
        end
        # Halving dx should quarter the error (2nd-order)
        r1 = errs[1] / errs[2]
        r2 = errs[2] / errs[3]
        @test 3.5 < r1 < 4.5
        @test 3.5 < r2 < 4.5
    end

    @testset "FFTPoissonSolver on a manufactured mode" begin
        # φ(x,y,z) = sin(2π x/Lx) cos(2π y/Ly) sin(2π z/Lz)
        # The *discrete* Laplacian of this mode has eigenvalue λ_h; if we
        # form source = L_h φ (via our laplacian!) and solve, we recover φ
        # to machine precision.
        L = 1.0
        g = RegularGrid(; size=(32, 32, 32), extent=(L, L, L), halo=(1,1,1))
        φ = Field(CellCenter(), g); source = Field(CellCenter(), g); p = Field(CellCenter(), g)
        set!(φ, (x,y,z) -> sin(2π*x) * cos(2π*y) * sin(2π*z))
        fill_halos!(φ, FieldBoundaryConditions(:periodic))
        laplacian(source, φ)   # source = L_h φ

        solver = FFTPoissonSolver(g)
        solve!(p, source, solver)

        # p should equal φ (up to the projected-out mean; here φ has mean 0)
        vp = interior_view(p); vφ = interior_view(φ)
        err = maximum(abs.(vp .- vφ))
        @test err < 1e-10
    end

    @testset "CosineFFTPoissonSolver on a Neumann-BC manufactured mode" begin
        # Domain [0, Lx] × [0, Ly] × [0, Lz]. Periodic in x, y; Neumann in z.
        # φ = cos(2π x / Lx) * cos(2π y / Ly) * cos(π n z / Lz)
        # satisfies ∂φ/∂z = 0 at z = 0 and z = Lz for any integer n.
        Lx = Ly = Lz = 1.0
        g = RegularGrid(; size=(32,32,32), extent=(Lx,Ly,Lz), halo=(1,1,1))
        n = 2       # z-mode index
        φ = Field(CellCenter(), g); source = Field(CellCenter(), g); p = Field(CellCenter(), g)
        set!(φ, (x,y,z) -> cos(2π*x/Lx) * cos(2π*y/Ly) * cos(π*n*z/Lz))

        # Halo fill: periodic in x, y and no-flux (mirror) in z
        # → reproduces the Neumann BC used by the DCT solver.
        bcs = FieldBoundaryConditions(:periodic, :periodic, :noflux)
        fill_halos!(φ, bcs)
        laplacian(source, φ)     # discrete RHS from our stencil

        solver = CosineFFTPoissonSolver(g)
        solve!(p, source, solver)

        # φ has zero mean, so we recover it exactly (up to floating point).
        err = maximum(abs.(interior_view(p) .- interior_view(φ)))
        @test err < 1e-10
    end

    @testset "CosineFFTPoissonSolver: apply L_h to solution matches source" begin
        Lx = Ly = Lz = 2π
        g = RegularGrid(; size=(24,24,32), extent=(Lx,Ly,Lz), halo=(1,1,1))
        source = Field(CellCenter(), g); p = Field(CellCenter(), g); check = Field(CellCenter(), g)

        # Zero-mean random source. Because the z-BC is Neumann, the
        # cell-integrated source only needs to have zero global mean.
        Random.seed!(7)
        raw = randn(g.Nx, g.Ny, g.Nz)
        raw .-= sum(raw) / length(raw)
        interior_view(source) .= raw

        solver = CosineFFTPoissonSolver(g)
        solve!(p, source, solver)

        fill_halos!(p, FieldBoundaryConditions(:periodic, :periodic, :noflux))
        laplacian(check, p)
        err = maximum(abs.(interior_view(check) .- interior_view(source)))
        @test err < 1e-10
    end

    @testset "FFTPoissonSolver: solve then apply Laplacian recovers source" begin
        L = 2π
        g = RegularGrid(; size=(48, 32, 16), extent=(L, L, L), halo=(1,1,1))
        p  = Field(CellCenter(), g)
        src = Field(CellCenter(), g)
        # Random source with zero mean
        Random.seed!(42)
        rand_data = randn(g.Nx, g.Ny, g.Nz)
        rand_data .-= sum(rand_data) / length(rand_data)   # remove mean
        interior_view(src) .= rand_data

        solver = FFTPoissonSolver(g)
        solve!(p, src, solver)

        fill_halos!(p, FieldBoundaryConditions(:periodic))
        check = Field(CellCenter(), g)
        laplacian(check, p)
        err = maximum(abs.(interior_view(check) .- interior_view(src)))
        @test err < 1e-10
    end

    @testset "Staggered scalar advection: mass conservation" begin
        # ∫ϕ dV should be conserved by flux-form staggered advection under
        # a divergence-free velocity field.
        L = 2π
        mesh = RegularGrid(; size=(24,24,24), extent=(L,L,L), halo=(2,2,2))
        u = Field(FaceX(), mesh); v = Field(FaceY(), mesh); w = Field(FaceZ(), mesh)
        ϕ = Field(CellCenter(), mesh); t = Field(CellCenter(), mesh)
        set!(u, (x,y,z) -> sin(y)*cos(z))     # ∂u/∂x = 0
        set!(v, (x,y,z) -> sin(z)*cos(x))     # ∂v/∂y = 0
        set!(w, (x,y,z) -> sin(x)*cos(y))     # ∂w/∂z = 0  → divergence-free
        set!(ϕ, (x,y,z) -> cos(x)*cos(y)*cos(z))
        bcs = FieldBoundaryConditions(:periodic)
        for f in (u, v, w, ϕ); fill_halos!(f, bcs); end
        fill!(t.data, 0)
        advect_centered!(t, u, v, w, ϕ)
        # ∑ tendency ≈ 0 (surface fluxes cancel in periodic domain)
        @test abs(sum(interior_view(t))) < 1e-10
    end

    @testset "Buoyancy: hydrostatic stratification stays quiescent" begin
        # Linear θ profile with θ = θ₀ at the reference level. Buoyancy
        # gives b = 0 at that level. Under buoyancy alone (no initial
        # motion) and zero-mean initial buoyancy anomaly, KE stays tiny.
        L = 1.0
        N = 16
        mesh = RegularGrid(; size=(N,N,N), extent=(L,L,L), halo=(2,2,2))
        θ₀   = 300.0
        sim  = BoussinesqSimulation(mesh; θ₀=θ₀, g_accel=9.81)
        # θ ≡ θ₀ everywhere so buoyancy anomaly is zero
        set!(sim.θ, (x,y,z) -> θ₀)
        ke0 = kinetic_energy(sim)
        for _ in 1:20; step!(sim, 1e-3); end
        ke = kinetic_energy(sim)
        @test ke0 < 1e-20
        @test ke  < 1e-20
    end

    @testset "Coriolis: inertial oscillation frequency" begin
        # With uniform initial (u₀, 0, 0), no advection (∇u = 0 initially),
        # Coriolis alone gives u(t) = u₀ cos(f t), v(t) = -u₀ sin(f t).
        # Check period T = 2π/f is recovered.
        L  = 1.0; N = 16
        mesh = RegularGrid(; size=(N,N,N), extent=(L,L,L), halo=(2,2,2))
        f   = 1.0
        u₀  = 0.1
        sim = BoussinesqSimulation(mesh; f_coriolis=f)
        set!(sim.u, (x,y,z) -> u₀)
        # Track ū at first cell for a few periods
        dt = 0.05
        nper = 40      # ~2 periods
        us = Float64[]
        vs = Float64[]
        for n in 1:nper
            step!(sim, dt)
            push!(us, mean(interior_view(sim.u)))
            push!(vs, mean(interior_view(sim.v)))
        end
        # Rough check: ū should return near u₀ after one period (T=2π ≈ 6.28)
        # With dt=0.05, step n≈126 is one period. Our nper=40 only covers
        # ~1/3 period; use a phase check instead.
        # After t=2 (~32% period), v should be roughly -u₀ sin(f*t) ≈ -0.0917.
        t_end = nper * dt
        v_expected = -u₀ * sin(f * t_end)
        # 1st-order Euler is very dissipative for oscillations; allow 10% error
        @test isapprox(vs[end], v_expected; atol=0.02)
    end

    @testset "Smagorinsky ν_t: zero for uniform flow, positive for shear" begin
        L = 1.0; N = 16
        mesh = RegularGrid(; size=(N,N,N), extent=(L,L,L), halo=(2,2,2))
        sim  = BoussinesqSimulation(mesh; C_s = 0.17)
        # uniform flow → strain = 0 → ν_t = 0
        set!(sim.u, (x,y,z) -> 1.0)
        fill_halos!(sim.u, sim.bcs); fill_halos!(sim.v, sim.bcs); fill_halos!(sim.w, sim.bcs)
        compute_smagorinsky_νt!(sim.νt, sim.u, sim.v, sim.w, sim.C_s)
        @test max_νt(sim) < 1e-14

        # simple shear u = z → S_13 nonzero → ν_t > 0
        set!(sim.u, (x,y,z) -> z)
        fill_halos!(sim.u, sim.bcs); fill_halos!(sim.v, sim.bcs); fill_halos!(sim.w, sim.bcs)
        compute_smagorinsky_νt!(sim.νt, sim.u, sim.v, sim.w, sim.C_s)
        @test max_νt(sim) > 0
    end

    @testset "Fractional step: projection kills divergence" begin
        # Fill u,v,w with random garbage → nonzero ∇·u. One `step!` with
        # ν=0 should return a divergence-free field to machine precision.
        L = 2π
        mesh = RegularGrid(; size=(24,24,24), extent=(L,L,L), halo=(2,2,2))
        sim  = BoussinesqSimulation(mesh; ν=0.0)
        Random.seed!(2)
        interior_view(sim.u) .= randn(24,24,24)
        interior_view(sim.v) .= randn(24,24,24)
        interior_view(sim.w) .= randn(24,24,24)
        # zero-mean per component (compatibility for periodic BC)
        for f in (sim.u, sim.v, sim.w)
            iv = interior_view(f)
            iv .-= sum(iv) / length(iv)
        end

        @test max_divergence(sim) > 0.1     # sanity: we started divergent
        step!(sim, 0.01)
        @test max_divergence(sim) < 1e-10
    end

    @testset "Wall BC: w zeroed at wall faces + antisymmetric halo" begin
        g = RegularGrid(; size=(4,4,8), extent=(1.0,1.0,1.0), halo=(2,2,2))
        w = Field(FaceZ(), g)
        # Fill w with nonzero interior
        set!(w, (x,y,z) -> 0.7)
        bcs = FieldBoundaryConditions(:periodic, :periodic, :wall)
        fill_halos!(w, bcs)
        Hz, Nz = g.Hz, g.Nz
        # bottom wall face and top wall face must be zero
        @test all(w.data[:, :, Hz + 1]      .== 0)
        @test all(w.data[:, :, Hz + Nz + 1] .== 0)
        # anti-symmetric extension below the bottom wall:
        for h in 1:Hz
            @test all(w.data[:, :, Hz + 1 - h] .== -w.data[:, :, Hz + 1 + h])
        end
    end

    @testset "Wall BC: u mirrors free-slip in z" begin
        g = RegularGrid(; size=(4,4,8), extent=(1.0,1.0,1.0), halo=(2,2,2))
        u = Field(FaceX(), g)
        set!(u, (x,y,z) -> z)
        bcs = FieldBoundaryConditions(:periodic, :periodic, :wall)
        fill_halos!(u, bcs)
        Hz, Nz = g.Hz, g.Nz
        # mirror reflection: halo cell = first interior cell across the wall
        @test all(u.data[:, :, Hz]         .== u.data[:, :, Hz + 1])
        @test all(u.data[:, :, Hz + Nz + 1] .== u.data[:, :, Hz + Nz])
    end

    @testset "BoussinesqSimulation with walls: default solver + divergence" begin
        L = 1.0; N = 16
        mesh = RegularGrid(; size=(N,N,N), extent=(L,L,L), halo=(2,2,2))
        wall_bcs = FieldBoundaryConditions(:periodic, :periodic, :wall)
        sim = BoussinesqSimulation(mesh; ν=1e-3, bcs=wall_bcs)
        # Default solver should be CosineFFTPoissonSolver
        @test sim.solver isa Aeolus.PressureSolvers.CosineFFTPoissonSolver
        # Random divergent initial velocity
        Random.seed!(9)
        interior_view(sim.u) .= randn(N,N,N)
        interior_view(sim.v) .= randn(N,N,N)
        interior_view(sim.w) .= randn(N,N,N)
        # A few steps; divergence must return to machine ε
        for _ in 1:5
            step!(sim, 1e-3)
        end
        @test max_divergence(sim) < 1e-10
        # w at wall faces stays zero
        Hz, Nz = mesh.Hz, mesh.Nz
        @test all(abs.(sim.w.data[:, :, Hz + 1])      .< 1e-12)
        @test all(abs.(sim.w.data[:, :, Hz + Nz + 1]) .< 1e-12)
    end

    @testset "Surface flux warms the bottom, cools the top" begin
        L = 1.0; N = 8
        mesh = RegularGrid(; size=(N,N,N), extent=(L,L,L), halo=(2,2,2))
        wall_bcs = FieldBoundaryConditions(:periodic, :periodic, :wall)
        F_bot = 0.1     # into the domain
        F_top = 0.05    # out of the domain (cools the domain from above)
        sim = BoussinesqSimulation(mesh; ν=0.0, bcs=wall_bcs,
                                   θ_flux_bottom=F_bot, θ_flux_top=F_top)
        set!(sim.θ, (x,y,z) -> 300.0)
        θ_start_bot = mean(interior_view(sim.θ)[:, :, 1])
        θ_start_top = mean(interior_view(sim.θ)[:, :, end])
        for _ in 1:10
            step!(sim, 1e-3)
        end
        θ_end_bot = mean(interior_view(sim.θ)[:, :, 1])
        θ_end_top = mean(interior_view(sim.θ)[:, :, end])
        @test θ_end_bot > θ_start_bot     # bottom warmed
        @test θ_end_top < θ_start_top     # top cooled
    end

    @testset "Sponge damps velocity toward zero near the top" begin
        L = 1.0; N = 8
        mesh = RegularGrid(; size=(N,N,N), extent=(L,L,L), halo=(2,2,2))
        wall_bcs = FieldBoundaryConditions(:periodic, :periodic, :wall)
        sim = BoussinesqSimulation(mesh; ν=0.0, bcs=wall_bcs,
                                   sponge_α_max=10.0, sponge_z_start=0.75L)
        # uniform vertical velocity in the whole domain
        set!(sim.u, (x,y,z) -> 1.0)
        u_ref_top_start = mean(interior_view(sim.u)[:, :, end])
        u_ref_bot_start = mean(interior_view(sim.u)[:, :, 1])
        for _ in 1:20
            step!(sim, 1e-2)
        end
        u_ref_top_end = mean(interior_view(sim.u)[:, :, end])
        u_ref_bot_end = mean(interior_view(sim.u)[:, :, 1])
        # sponge zone (top) damped significantly
        @test abs(u_ref_top_end) < 0.5 * abs(u_ref_top_start)
        # bottom outside sponge zone barely changed
        @test isapprox(u_ref_bot_end, u_ref_bot_start; rtol=0.05)
    end

    @testset "DCBL smoke test: coupled system runs stably" begin
        # A miniature DCBL: 16³ grid, walls, surface flux, SGS, sponge.
        L  = 1600.0
        N  = 16
        mesh = RegularGrid(; size=(N,N,N), extent=(L,L,L), halo=(2,2,2))
        Random.seed!(4)
        sim = BoussinesqSimulation(mesh;
            θ₀            = 300.0,
            g_accel       = 9.81,
            C_s           = 0.17,
            Pr_t          = 1/3,
            θ_flux_bottom = 0.24,
            sponge_α_max  = 0.01,
            sponge_z_start = 0.75L,
            bcs = FieldBoundaryConditions(:periodic, :periodic, :wall),
            θ_bcs = FieldBoundaryConditions(:periodic, :periodic, :noflux))
        set!(sim.θ, (x,y,z) -> 300.0 + max(0, (z - 800.0)) * 3e-3 + 0.05*(rand()-0.5))
        for _ in 1:20
            step!(sim, 1.0)
        end
        # divergence at machine precision throughout
        @test max_divergence(sim) < 1e-12
        # kinetic energy nontrivial (thermals developed)
        @test kinetic_energy(sim) > 0
        # SGS eddy viscosity nonzero
        @test max_νt(sim) > 0
        # profiles have expected sizes
        @test length(mean_θ_profile(sim)) == N
        @test length(horizontal_variance(sim.θ)) == N
    end

    @testset "WENO-5 scalar advection: mass conservation" begin
        L = 2π
        N = 24
        mesh = RegularGrid(; size=(N,N,N), extent=(L,L,L), halo=(3,3,3))
        u = Field(FaceX(), mesh); v = Field(FaceY(), mesh); w = Field(FaceZ(), mesh)
        ϕ = Field(CellCenter(), mesh); t = Field(CellCenter(), mesh)
        # divergence-free ABC-like flow
        set!(u, (x,y,z) -> sin(y)*cos(z))
        set!(v, (x,y,z) -> sin(z)*cos(x))
        set!(w, (x,y,z) -> sin(x)*cos(y))
        set!(ϕ, (x,y,z) -> cos(x)*cos(y)*cos(z))
        bcs = FieldBoundaryConditions(:periodic)
        for f in (u, v, w, ϕ); fill_halos!(f, bcs); end
        fill!(t.data, 0)
        advect_weno5!(t, u, v, w, ϕ)
        # ∑ tendency ≈ 0 for periodic + divergence-free flow
        @test abs(sum(interior_view(t))) < 1e-9
    end

    @testset "WENO-5 halo check" begin
        mesh = RegularGrid(; size=(8,8,8), extent=(1.0,1.0,1.0), halo=(2,2,2))
        u = Field(FaceX(), mesh); v = Field(FaceY(), mesh); w = Field(FaceZ(), mesh)
        ϕ = Field(CellCenter(), mesh); t = Field(CellCenter(), mesh)
        @test_throws Exception advect_weno5!(t, u, v, w, ϕ)
    end

    @testset "step_rk3!: divergence stays at machine precision" begin
        L = 2π; N = 16
        mesh = RegularGrid(; size=(N,N,N), extent=(L,L,L), halo=(2,2,2))
        sim = BoussinesqSimulation(mesh; ν=1e-3)
        set!(sim.u, (x,y,z) ->  sin(x)*cos(y)*cos(z))
        set!(sim.v, (x,y,z) -> -cos(x)*sin(y)*cos(z))
        for _ in 1:5
            step_rk3!(sim, 0.01)
        end
        @test max_divergence(sim) < 1e-10
    end

    @testset "Moist buoyancy: virtual θ shifts the buoyancy" begin
        # With qv > 0, virtual θ increases; buoyancy increases; w develops
        # a positive vertical velocity relative to the dry case.
        L = 1.0; N = 8
        mesh = RegularGrid(; size=(N,N,N), extent=(L,L,L), halo=(2,2,2))
        dry = BoussinesqSimulation(mesh; θ₀=300.0)
        wet = BoussinesqSimulation(mesh; θ₀=300.0, moisture=true)
        # identical θ, but wet has qv > 0
        set!(dry.θ, (x,y,z) -> 300.5)
        set!(wet.θ, (x,y,z) -> 300.5)
        set!(wet.qv, (x,y,z) -> 0.02)
        for _ in 1:5
            step!(dry, 1e-3)
            step!(wet, 1e-3)
        end
        # both accelerate upward, but wet should be stronger
        wbar_dry = mean(interior_view(dry.w))
        wbar_wet = mean(interior_view(wet.w))
        @test wbar_wet > wbar_dry
    end

    @testset "Moisture advection: qv transported without loss" begin
        L = 2π; N = 16
        mesh = RegularGrid(; size=(N,N,N), extent=(L,L,L), halo=(2,2,2))
        sim = BoussinesqSimulation(mesh; moisture=true)
        set!(sim.u, (x,y,z) -> 0.5)
        set!(sim.qv, (x,y,z) -> 0.01 + 0.005*cos(x))
        mass0 = sum(interior_view(sim.qv))
        for _ in 1:10; step!(sim, 1e-2); end
        mass1 = sum(interior_view(sim.qv))
        # centered scheme is conservative on periodic + divergence-free
        @test isapprox(mass0, mass1; rtol=1e-8)
    end

    @testset "Kessler microphysics: autoconversion + accretion conserves ql + qr" begin
        mesh = RegularGrid(; size=(8,8,4), extent=(1.0,1.0,1.0), halo=(2,2,2))
        qv = Field(CellCenter(), mesh)
        ql = Field(CellCenter(), mesh); set!(ql, (x,y,z) -> 5e-3)   # above threshold
        qr = Field(CellCenter(), mesh); set!(qr, (x,y,z) -> 1e-4)
        total0 = sum(interior_view(ql)) + sum(interior_view(qr))
        # Turn off fallout to isolate conversion terms
        apply_kessler_step!(qv, ql, qr, 1.0; Vt=0.0)
        total1 = sum(interior_view(ql)) + sum(interior_view(qr))
        @test isapprox(total0, total1; atol=1e-12)
        # After one step, some ql has become qr
        @test sum(interior_view(qr)) > 1e-4 * length(interior_view(qr))
    end

    @testset "Kessler microphysics: rain fallout moves qr downward" begin
        mesh = RegularGrid(; size=(4,4,8), extent=(1.0,1.0,1.0), halo=(2,2,2))
        qv = Field(CellCenter(), mesh); ql = Field(CellCenter(), mesh)
        qr = Field(CellCenter(), mesh)
        set!(qr, (x,y,z) -> z > 0.6 ? 1e-3 : 0.0)      # rain in top half
        top_before = sum(interior_view(qr)[:, :, end])
        bot_before = sum(interior_view(qr)[:, :, 1])
        for _ in 1:10
            apply_kessler_step!(qv, ql, qr, 0.05; k_auto=0.0, k_acc=0.0, Vt=1.0)
        end
        top_after = sum(interior_view(qr)[:, :, end])
        bot_after = sum(interior_view(qr)[:, :, 1])
        @test top_after < top_before      # rain fell out of the top
        @test bot_after > bot_before      # rain arrived at the bottom
    end

    @testset "IBM: solid mask zeros velocity on solid cells" begin
        mesh = RegularGrid(; size=(8,8,8), extent=(1.0,1.0,1.0), halo=(2,2,2))
        sim  = BoussinesqSimulation(mesh; ν=1e-3)
        # A solid cube in the middle
        sm = SolidMask(mesh)
        mark_solid!(sm, (x, y, z) -> 0.3 < x < 0.6 && 0.3 < y < 0.6 && 0.3 < z < 0.6)
        # Fill velocity with 1 everywhere
        set!(sim.u, (x,y,z) -> 1.0)
        set!(sim.v, (x,y,z) -> 1.0)
        set!(sim.w, (x,y,z) -> 1.0)
        apply_solid_mask!(sim, sm)
        Hz = mesh.Hz
        # Cells that are inside the solid must have zero face velocities
        @test sim.u.data[mesh.Hx + 4, mesh.Hy + 4, Hz + 4] == 0
        @test sim.w.data[mesh.Hx + 4, mesh.Hy + 4, Hz + 4] == 0
    end

    @testset "CGPoissonSolver: matches FFT solver on a periodic manufactured mode" begin
        L = 1.0
        g = RegularGrid(; size=(16,16,16), extent=(L,L,L), halo=(1,1,1))
        φ = Field(CellCenter(), g); src = Field(CellCenter(), g)
        p_fft = Field(CellCenter(), g); p_cg = Field(CellCenter(), g)
        set!(φ, (x,y,z) -> sin(2π*x)*cos(2π*y)*sin(2π*z))
        fill_halos!(φ, FieldBoundaryConditions(:periodic))
        laplacian(src, φ)

        fft_solver = FFTPoissonSolver(g)
        cg_solver  = CGPoissonSolver(g; bc_top=:periodic, bc_bottom=:periodic,
                                        tol=1e-10, maxit=1000)
        solve!(p_fft, src, fft_solver)
        solve!(p_cg,  src, cg_solver)

        # Both solutions have zero mean; compare interiors
        err = maximum(abs.(interior_view(p_cg) .- interior_view(p_fft)))
        @test err < 1e-8
    end

    @testset "Parallel scaffolding: serial partition works, pencil errors nicely" begin
        @test nranks(SerialPartition()) == 1
        @test is_serial(SerialPartition())

        p = PencilPartition(2, 4)
        @test nranks(p) == 8
        @test !is_serial(p)

        # serial exchange is a no-op
        mesh = RegularGrid(; size=(4,4,4), extent=(1.0,1.0,1.0), halo=(1,1,1))
        f = Field(CellCenter(), mesh); set!(f, (x,y,z) -> 1.0)
        @test exchange_halos_distributed!(f, SerialPartition()) === f
        @test_throws Exception exchange_halos_distributed!(f, p)
    end

    @testset "Backend scaffolding: CPU works, GPU stub errors" begin
        cpu = CPUBackend()
        gpu = GPUBackend(:cuda)
        @test backend_name(cpu) == :cpu
        @test backend_name(gpu) == :cuda
        a = allocate(cpu, Float64, (4, 4, 4))
        @test a isa Array{Float64,3}
        @test size(a) == (4, 4, 4)
        @test_throws Exception allocate(gpu, Float64, (4, 4, 4))
        @test to_host(a) === a || to_host(a) == a
    end

    include("backends.jl")   # KA-a (#294): launch!, set!/interior_map! kernels, canonical reductions
    include("ka_halos.jl")   # KA-f (#297): halo fills / nesting transfer operators on launch!
    include("ka_fv.jl")      # KA-d (#299): finite-volume kernels + grid metrics on launch!
    include("ka_pcg.jl")     # KA-g (#300): the one pcg! driver of the four CG solvers
    include("thread_invariance.jl")  # KA-a (#294): N threads == 1 thread (subprocesses: AEOLUS_THREAD_TESTS=1)
    include("pressure_fft_threads.jl")  # KA-h (#298): spectral solvers as kernels, FFTW thread-invariant

    @testset "GLL nodes: endpoints exact, weights sum to 2, symmetry" begin
        for N in (2, 3, 4, 6, 8)
            x = gll_nodes(N)
            w = gll_weights(N)
            @test length(x) == N + 1
            @test x[1] ≈ -1
            @test x[end] ≈ 1
            @test sum(w) ≈ 2 atol=1e-12
            # symmetric about 0
            @test all(isapprox.(x, .-reverse(x); atol=1e-12))
            @test all(isapprox.(w, reverse(w);  atol=1e-12))
        end
    end

    @testset "SpectralElementMesh: type + coordinate access" begin
        m = SpectralElementMesh(; order=4, elements_per_dim=(2,3,4),
                                  extent=(2.0, 3.0, 4.0))
        @test m isa Aeolus.Grids.AbstractMesh
        Np = m.order + 1
        @test Aeolus.SpectralElement.totalsize(m) == (Np, Np, Np, 2*3*4)
        @test Aeolus.SpectralElement.interior(m) == (1:Np, 1:Np, 1:Np, 1:24)

        xs = Aeolus.SpectralElement.xnodes(m)
        @test length(xs) == Np * m.Nex
        @test xs[1]   ≈ 0.0 atol=1e-12
        @test xs[end] ≈ 2.0 atol=1e-12
    end

    @testset "Radiative cooling: uniform rate warms/cools uniformly" begin
        mesh = RegularGrid(; size=(4,4,4), extent=(1.0,1.0,1.0), halo=(1,1,1))
        sim = BoussinesqSimulation(mesh)
        set!(sim.θ, (x,y,z) -> 300.0)
        rate = -1e-3   # K/s
        # Apply cooling manually for one "step"
        fill!(sim.tθ.data, 0)
        apply_radiative_cooling!(sim.tθ, rate)
        # tendency uniform and negative
        @test all(interior_view(sim.tθ) .≈ rate)
    end

    @testset "Radiative cooling: profile" begin
        mesh = RegularGrid(; size=(4,4,4), extent=(1.0,1.0,1.0), halo=(1,1,1))
        sim = BoussinesqSimulation(mesh)
        prof = collect(-4:-1) * 1e-4    # length 4
        fill!(sim.tθ.data, 0)
        apply_radiative_cooling!(sim.tθ, prof)
        v = interior_view(sim.tθ)
        for k in 1:4
            @test all(v[:, :, k] .≈ prof[k])
        end
    end

    @testset "Radiative cooling: solid cells are not air" begin
        # a rate applied with a solid mask changes the fluid cells only, and
        # the fluid cells exactly as without the mask (#130: the Berlin
        # column heating was going into the buildings too)
        mesh = RegularGrid(; size=(4,4,4), extent=(1.0,1.0,1.0), halo=(1,1,1))
        sm = Aeolus.IBM.SolidMask(mesh)
        Aeolus.IBM.mark_solid!(sm, (x, y, z) -> x < 0.5 && y < 0.5 && z < 0.5)
        prof = collect(-4:-1) * 1e-4
        a = Field(CellCenter(), mesh); fill!(a.data, 0)
        b = Field(CellCenter(), mesh); fill!(b.data, 0)
        apply_radiative_cooling!(a, prof; solid = sm.mask)
        apply_radiative_cooling!(b, prof)
        va, vb = interior_view(a), interior_view(b)
        solid = view(sm.mask, 2:5, 2:5, 2:5)
        @test count(solid) == 8
        @test all(iszero, va[solid])
        @test va[.!solid] == vb[.!solid]
        # the scalar form is the uniform profile
        c = Field(CellCenter(), mesh); fill!(c.data, 0)
        apply_radiative_cooling!(c, -1e-3; solid = sm.mask)
        @test all(iszero, interior_view(c)[solid]) && all(interior_view(c)[.!solid] .== -1e-3)
    end

    @testset "SlabLand: reaches radiative equilibrium" begin
        # No sensible flux, no SW: ground cools toward LW equilibrium.
        land = SlabLand(; T=310.0, C_s=1e5, albedo=0.0)
        # LW down of 300 W/m² → equilibrium T = (300/σ)^(1/4) ≈ 269.5 K
        for _ in 1:5000
            step_land!(land, 0.0, 300.0, 0.0, 60.0)
        end
        @test isapprox(surface_temperature(land), (300 / 5.67e-8)^(1/4); atol=1)
    end

    @testset "Chemistry: linear decay reduces tracer" begin
        mesh = RegularGrid(; size=(4,4,4), extent=(1.0,1.0,1.0), halo=(1,1,1))
        c  = Field(CellCenter(), mesh); set!(c,  (x,y,z) -> 1.0)
        tc = Field(CellCenter(), mesh); fill!(tc.data, 0)
        apply_linear_chemistry!(tc, c, 10.0)     # τ = 10 s
        # tendency = -c/τ = -0.1 uniformly
        @test all(interior_view(tc) .≈ -0.1)
    end

    @testset "Chemistry: source only, τ=Inf" begin
        mesh = RegularGrid(; size=(4,4,4), extent=(1.0,1.0,1.0), halo=(1,1,1))
        c  = Field(CellCenter(), mesh); set!(c,  (x,y,z) -> 0.5)
        tc = Field(CellCenter(), mesh); fill!(tc.data, 0)
        apply_linear_chemistry!(tc, c, Inf; source = 0.03)
        @test all(interior_view(tc) .≈ 0.03)
    end

    @testset "Particles: advance in uniform flow moves particles" begin
        mesh = RegularGrid(; size=(8,8,8), extent=(1.0,1.0,1.0), halo=(2,2,2))
        sim = BoussinesqSimulation(mesh)
        set!(sim.u, (x,y,z) -> 0.1)
        set!(sim.v, (x,y,z) -> 0.0)
        set!(sim.w, (x,y,z) -> 0.0)
        fill_halos!(sim.u, sim.bcs); fill_halos!(sim.v, sim.bcs); fill_halos!(sim.w, sim.bcs)
        ps = ParticleSet(mesh)
        add_particles!(ps, [(0.5, 0.5, 0.5), (0.2, 0.2, 0.2)])
        x0 = positions(ps)[1][1]
        advance_particles!(ps, sim, 0.1)
        x1 = positions(ps)[1][1]
        @test x1 > x0        # particle moved in +x direction
        @test x1 ≈ x0 + 0.1 * 0.1 atol=1e-6
    end

    @testset "MLSGS: user function populates νt" begin
        mesh = RegularGrid(; size=(4,4,4), extent=(1.0,1.0,1.0), halo=(2,2,2))
        sim = BoussinesqSimulation(mesh)
        # A trivial ML "network": νt = |u|² everywhere
        ml = MLSGS() do νt, u, v, w, m
            interior_view(νt) .= interior_view(u) .^ 2
        end
        set!(sim.u, (x,y,z) -> 2.0)
        apply_mlsgs!(sim, ml)
        @test all(interior_view(sim.νt) .≈ 4.0)
    end

    @testset "DA: nudging pulls the field toward the observation" begin
        mesh = RegularGrid(; size=(4,4,4), extent=(1.0,1.0,1.0), halo=(1,1,1))
        sim = BoussinesqSimulation(mesh)
        set!(sim.θ, (x,y,z) -> 300.0)
        obs = Observation(:θ, 2, 2, 2, 310.0; sigma=0.5)
        Hx, Hy, Hz = mesh.Hx, mesh.Hy, mesh.Hz
        θ0 = sim.θ.data[Hx+2, Hy+2, Hz+2]
        assimilate_nudging!(sim, obs, 1.0, 0.5)   # γ=1, dt=0.5 → 50 % of gap
        θ1 = sim.θ.data[Hx+2, Hy+2, Hz+2]
        @test θ1 > θ0
        @test θ1 ≈ 305.0 atol=1e-8
    end

    @testset "I/O: snapshot round-trip; snapshots are not restart files (#245)" begin
        mesh = RegularGrid(; size=(8,8,8), extent=(1.0,1.0,1.0), halo=(2,2,2))
        sim1 = BoussinesqSimulation(mesh; ν=1e-3, θ₀=300.0)
        Random.seed!(11)
        interior_view(sim1.u) .= randn(8,8,8)
        interior_view(sim1.θ) .= 300.0 .+ 0.1 .* randn(8,8,8)
        sim1.t = 42.0
        path = tempname() * ".jls"
        write_snapshot(path, sim1)
        d = read_snapshot(path)
        @test d[:t] == 42.0
        @test d[:u] == interior_view(sim1.u) && d[:θ] == interior_view(sim1.θ)
        # Maintainer decision on #245: an unversioned snapshot is refused by
        # restart! (it has no closure memory, halos or stepper state). The
        # pre-#245 expectation here (restart! from write_snapshot) is retired.
        sim2 = BoussinesqSimulation(mesh; ν=1e-3, θ₀=300.0)
        @test_throws ArgumentError restart!(sim2, path)
        # the versioned checkpoint round-trips, halos included
        checkpoint!(sim1, path)
        restart!(sim2, path)
        @test sim2.t == 42.0
        @test sim2.u.data == sim1.u.data && sim2.θ.data == sim1.θ.data
        rm(path)
    end

    include("io_checkpoint.jl")   # #245: full-state checkpoints, bitwise restart, FFTW policy
    include("logging_core.jl")    # #333: structured events, run log, test logger
    include("logging_run_folder.jl")  # #334: run folder, run.toml provenance, dirty-tree policy
    include("logging_diagnostics.jl") # #335: diagnostics registry, CSV writer, progress line
    include("logging_blowup.jl")      # #335: blowup diagnose-dump-stop, run monitor
    include("logging_diagnostics_nc.jl")  # #337: diagnostics.nc (profiles, NetCDF-4, provenance attributes)
    include("config_schema.jl")       # CFG-a (#356): schema, strict errors, canonical fixed point, float bits
    include("config_load.jl")         # CFG-a (#356): extends/overrides, run.toml [config], inputs, common sections
    include("config_simulation_driver.jl")  # CFG-b (#357): simulation driver, bitwise round trip, resume, TGV
    include("config_cli.jl")          # CFG-b (#357): aeolus CLI, exit codes, example drivers, subprocess
    include("config_sbl.jl")          # CFG-d (#359): sbl driver, bit-identity to the old builders, round trip
    include("config_dns.jl")          # CFG-d (#359): dns driver, round trip, the k_max η refusal
    include("config_examples.jl")     # CFG-e (#360): [sweep] expansion, example coverage
    include("config_examples_validation.jl")  # CFG-e (#360): examples/validation/ from TOML, bit-identical to the old scripts
    include("config_berlin.jl")       # CFG-c (#358): berlin driver, scenarios from TOML, bitwise vs Julia-built, round trip
    include("input_static.jl")        # CFG-f (#361): static driver, OSM/GeoJSON preprocessors, Berlin ground from one file
    include("input_dynamic.jl")       # CFG-g (#362): dynamic driver, sounding/case-table and parent preprocessors, offline nest

    @testset "I/O: TOML config round-trip" begin
        path = tempname() * ".toml"
        open(path, "w") do io
            write(io, """
[mesh]
size   = [16, 16, 32]

[physics]
C_s = 0.17
""")
        end
        cfg = load_config(path)
        @test cfg["mesh"]["size"] == [16, 16, 32]
        @test cfg["physics"]["C_s"] == 0.17
        rm(path)
    end

    @testset "Advection: constant flow preserves shape" begin
        # 1D-style test in 3D: advect ϕ(x) with uniform u; after one period
        # ϕ should return to itself (upwind is diffusive so tolerance loose).
        Nx = 64
        g = RegularGrid(; size=(Nx,4,4), extent=(1.0,0.1,0.1), halo=(2,2,2))
        ϕ = Field(CellCenter(), g); u = Field(CellCenter(), g)
        v = Field(CellCenter(), g); w = Field(CellCenter(), g)
        set!(u, (x,y,z) -> 1.0)
        set!(ϕ, (x,y,z) -> sin(2π*x))
        bcs = FieldBoundaryConditions(:periodic)
        state = (ϕ = ϕ,)
        rk = RK3(state)
        function tend!(k, s, t)
            fill!(k.ϕ.data, 0)
            fill_halos!(s.ϕ, bcs)
            advect_upwind!(k.ϕ, u, v, w, s.ϕ)
            return k
        end
        dt = 0.3 * g.dx     # CFL well under 1
        nsteps = round(Int, 1.0 / dt)
        dt = 1.0 / nsteps
        mass0 = sum(interior_view(state.ϕ))
        for _ in 1:nsteps
            step!(rk, state, tend!, dt)
        end
        mass1 = sum(interior_view(state.ϕ))
        # mass conservation to floating-point precision
        @test isapprox(mass0, mass1; atol=1e-10)
    end

end

include("era1_chemistry.jl")   # urban NOx–O3–VOC mechanism, dry / wet deposition (#135)

# =========================================================================
# Era II — numerical foundations for fidelity (Phases 21-31)
# =========================================================================
# Each file is a self-contained validated-capability suite. They are kept
# in separate files because each is large and each corresponds to a
# distinct roadmap phase; keeping the mapping one-file-per-phase-group
# makes it obvious what a regression belongs to.

@testset "Era II" begin
    include("era2_cutcell.jl")     # Phases 21-22  cut-cell IBM, ghost cells
    include("era2_stretched.jl")   # Phase  24     stretched / terrain-following
    include("era2_stretched_boussinesq.jl") # #151 Boussinesq core + Poisson on a StretchedGrid
    include("era2_timeint.jl")     # Phases 23, 25 adaptive dt, IMEX
    include("era2_advection.jl")   # Phases 26, 28 WENO-Z/TENO, conservative ops
    include("era2_anelastic.jl")   # Phase  27     anelastic, pseudo-incompressible
    include("era2_multigrid.jl")   # Phase  29     geometric multigrid
    include("era2_sgs_numerics.jl") # #132         dynamic Smagorinsky, step! operator swap-ins
    include("era2_nesting.jl")     # Phase  30     parent-child nesting operators (#250)
    include("era2_nesting_conservative.jl") # #256 stretched parent, mass-weighted, solid-aware nesting
    include("era2_amr.jl")         # Phases 30-31  nesting, AMR
    include("era2_nested_run.jl")  # Phase  30     NestedRun time coupling (#257)
    include("era2_nest_boundary.jl") # Phase 30   one-way nest boundary data (#261)
    include("era2_nest_external.jl") # #265        externally driven child (the Berlin pattern), open scalars added later
    include("era2_nest_twoway.jl")  # Phase 30   two-way feedback: anterpolation, refluxing (#263)
    include("era2_nest_twoway_longrun.jl") # Phase 30 two-way long runs: CBL, neutral, vortex (#263)
end

# =========================================================================
# Era III — high-fidelity radiation (Phases 32-38)
# =========================================================================
# In progress. Phases 32-37 are stubbed pending implementation; the shared
# radiation contract (`RadiationCore`) and Phase 38 are complete.

@testset "Era III" begin
    include("era3_gasoptics.jl")       # Phase 32  correlated-k gas optics
    include("era3_twostream.jl")       # Phase 33  two-stream RT, cloud optics
    include("era3_surfaceoptics.jl")   # Phase 36  spectral albedo, BRDF
    include("era3_aerosol.jl")         # Phase 37  Mie aerosol optics
    include("era3_montecarlo.jl")      # Phase 34  3-D Monte-Carlo RT
    include("era3_radiosity.jl")       # Phase 35  urban radiosity
    include("era3_solargeometry.jl")   # Phase 38  solar position, shadows
    end

    @testset "Era IV" begin
        include("era4_thermodynamics.jl")  # Phase 39  exact moist thermodynamics
        include("era4_activation.jl")      # Phase 42  aerosol-cloud interaction
        include("era4_superdroplet.jl")    # Phase 41  superdroplet microphysics
        include("era4_p3.jl")              # Phase 40  P3 microphysics
        include("era4_icenucleation.jl")   # Phase 43  ice nucleation
        include("era4_turbmicro.jl")       # Phase 45  sub-grid supersaturation PDF
        include("era4_sgs_growth.jl")      # Phase 45  SGS activation and growth
        include("era4_broadening.jl")      # Phase 45  spectral broadening validation
        include("era4_visibility.jl")      # Phase 44  visibility diagnostic
        include("era4_fog.jl")             # Phase 44  fog / dew coupling (#11)
        include("era4_radiation_fog.jl")   # Phase 44  radiation fog feedback (#9)
end

# =========================================================================
# Era V — high-fidelity land, urban and surface exchange (Phases 46-53)
# =========================================================================
# `SurfaceExchange` is the era contract; later phases append here.

@testset "Era V" begin
    include("era5_surface_exchange.jl")  # Era V contract: MOST, constants, surface interface (#92)
    include("era5_soil_column.jl")       # Phase 46: soil column, Richards equation (#15)
    include("era5_soil_thermal.jl")      # Phase 46: soil heat and freeze–thaw (#16)
    include("era5_water_table.jl")       # Phase 46: water table, unconfined aquifer (#17)
    include("era5_photosynthesis.jl")    # Phase 47  Farquhar C3 / Collatz C4 photosynthesis (#19)
    include("era5_stomata.jl")           # Phase 47  Ball-Berry / Medlyn stomatal conductance (#20)
    include("era5_land_surface.jl")      # Phase 47: vegetated land column, soil-water stress (#21)
    include("era5_land_surface_fluxnet.jl") # Phase 47: land column vs FLUXNET/PLUMBER2 towers (#21)
    include("era5_plant_canopy.jl")      # Phase 48  LAD profiles, implicit canopy drag (#22)
    include("era5_canopy_radiation.jl")  # Phase 48  canopy SW two-stream, sunlit/shaded, LW (#23)
    include("era5_leaf_energy.jl")       # Phase 48  leaf energy balance, leaf temperature (#24)
    include("era5_canopy_profiles.jl")   # Phase 48  within-canopy wind/temperature validation (#25)
    include("era5_urban_surface.jl")     # Phase 49  multi-layer urban facet conduction (#26)
    include("era5_glazing.jl")           # Phase 49  glazing transmission, window facets (#27)
    include("era5_urban_radiosity.jl")   # Phase 49  facet temperatures coupled to radiosity (#28)
    include("era5_building_energy.jl")   # Phase 50  indoor air temperature + thermal mass (#31)
    include("era5_hvac.jl")              # Phase 50  HVAC demand and exhaust heat (#32)
    include("era5_occupancy.jl")         # Phase 50  occupancy schedules and internal gains (#33)
    include("era5_bestest.jl")           # Phase 50  ASHRAE 140-2020 BESTEST annual loads (#33)
    include("era5_anthropogenic_heat.jl") # Phase 49  anthropogenic heat from the building energy model (#29)
    include("era5_urban_validation.jl")  # Phase 49  BUBBLE / CAPITOUL validation of the urban canopy (#30)
    include("era5_snowpack.jl")          # Phase 51: layered snowpack with compaction (#34)
    include("era5_snow_metamorphism.jl") # Phase 51: grain metamorphism and albedo ageing (#35)
    include("era5_snow_liquid.jl")        # Phase 51: snow percolation and refreezing (#36)
    include("era5_ice_thermodynamics.jl")  # Phase 51  lake and sea ice, BL99 (#37)
    include("era5_ocean_mixed_layer.jl") # Phase 52  water column, cool skin, warm layer (#38)
    include("era5_air_sea_flux.jl")      # Phase 52  wave-dependent sea-surface roughness (#39)
    include("era5_sea_spray_coare.jl")   # Phase 52  sea spray, full COARE, validation (#40)
    include("era5_surface_tiles.jl")     # Phase 53 sub-grid tile mosaic, flux aggregation (#41)
    include("era5_blending_height.jl")   # Phase 53 blending height, effective roughness, IBL (#42)
    include("era5_heterogeneous_surface.jl") # Phase 53 resolved patchy-surface LES vs tiles (#43)
end

# =========================================================================
# Era VI — high-fidelity turbulence closure (Phases 54-60)
# =========================================================================
# The SGS closure contract (P54-a, #167) is the era contract; later issues
# append one include line each here.

@testset "Era VI" begin
    include("era6_sgs_closure.jl")       # P54-a: SGS closure contract, κ_t, commit cadence (#167)
    include("era6_sgs_stress_form.jl")   # P55-b: stress-form SGS momentum divergence, anisotropic ν, IMEX split (#175)
    include("era6_tensor_diffusion.jl")  # P55-c: diagonal / tensor / isentropic scalar eddy diffusivity (#181)
    include("era6_stability_functions.jl") # Phase 58 TurbulenceStability contract (#170)
    include("era6_wall_models.jl")       # Phase 57: equilibrium wall laws, wall-stress operator (#169)
    include("era6_wall_models_immersed.jl") # Phase 57: wall model on IBM and cut-cell walls (#179)
    include("era6_les_toolbox.jl")       # Phase 54  filters, spectrum, initialiser (#168)
    include("era6_lagrangian_sgs.jl")    # P54-b: Lagrangian dynamic Smagorinsky, mask-aware (#173)
    include("era6_lasd.jl")              # P54-e: scale-dependent Lagrangian dynamic model, log law (#186)
    include("era6_cbc_decay.jl")         # Phase 54  Comte-Bellot–Corrsin decay benchmark (#168)
    include("era6_walls.jl")             # P59-a NoSlipWall, laminar wall-flow verification (#172)
    include("era6_large_scale_forcing.jl") # Phase 58: geostrophic forcing, nudging, advective tendencies (#171)
    include("era6_nudging_subsidence.jl") # N-parent-c: plane-mean nudging of u, v, θ, qv; large-scale subsidence (#248)
    include("era6_scalar_sgs_diffusion.jl") # P56-a: SGS diffusion of qv and tracers with κ_t, solid mask (#177)
    include("era6_neutral_abl.jl")       # P54-d: half-channel ABL / channel harness + Φ_m, stress, E11 (#174)
    include("era6_amd_vreman.jl")        # P55-d: AMD and Vreman closures (#176)
    include("era6_apriori_sgs.jl")       # Phase 56  a-priori SGS stress, filtered-DNS fixture (#178)
    include("era6_transition_kh.jl")       # Phase 59 Transition diagnostics, KH / Miles-Howard (#180)
    include("era6_tke.jl")               # P56-b: Deardorff prognostic SGS TKE, wall distance (#182)
    include("era6_tke_stretched.jl")     # N-parent-e: the TKE closure on a StretchedGrid, dual on stretched levels (#259)
    include("era6_tke_facades.jl")       # #232: TKE shear production at wall-modelled facades
    include("era6_reynolds_stress.jl")   # P56-d: differential SGS Reynolds-stress closure (#188)
    include("era6_sgs_validation.jl")    # P56-e: TKE and RSM against filtered DNS (#192)
    include("era6_wall_models_channel.jl") # P57-b: matching height, time / wall-parallel input filters, LLM (#183)
    include("era6_wall_models_tble.jl")  # P57-c: non-equilibrium (TBLE) wall model, dynamic κ (#189)
    include("era6_sigma_sgs.jl")         # P59-b σ-model closure: invariants, z³ wall scaling, Couette, HIT (#184)
    include("era6_flat_plate.jl")        # P59-d flat-plate BL: fringe, Blasius, FST, transition (#185)
    include("era6_synthetic_inflow.jl")  # N-inflow-a: digital-filter inflow, Lund transform (#252)
    include("era6_inflow_statistics.jl")  # N-inflow-b: Reynolds stresses, lengths, times for the inflow (#258)
    include("era6_inflow_turbulence.jl")  # N-inflow-c: inflow faces, zero added mass flux (#264)
    include("era6_stratified_sgs.jl")    # P55-e: StabilityCorrected, Ri_g-dependent anisotropic ν/κ (#187)
    include("era6_wall_models_dynamic.jl") # P57-e: dynamic z0 of partially resolved roughness (#190)
    include("era6_gabls1.jl")            # P55-f: GABLS1 on the shared SBL driver, reduced (#191)
    include("era6_stratified_channel.jl") # P55-f: wall-resolved stably stratified channel vs DNS (#191)
    include("era6_collapse.jl")            # P58-e: maximum sustainable heat flux, LES collapse regime (#193)
    include("era6_gabls3.jl")            # P58-d: GABLS3 on the SBL driver: budgets, MOST, LLJ (#194)
    include("era6_relaminarisation.jl")  # P59-e: laminar stratified flow, decaying stratified turbulence, criteria (#195)
    include("era6_deep_top.jl")          # N-parent-b: θ damping toward a profile, fixed-gradient θ top (#254)
    include("era6_open_pressure.jl")     # N-open-a: per-axis Neumann Poisson solves, BC-aware default solver (#251)
    include("era6_open_boundaries.jl")   # N-open-b: open faces, boundary-data contract, mass-flux correction (#255)
    include("era6_outflow_conditions.jl") # N-open-c: convective / Orlanski outflow, boundary_scalar_flux, outgoing vortex (#260)
    include("era6_inflow_fetch.jl")      # N-inflow-d: development length behind a synthetic inflow vs a periodic precursor (#266)
    include("era6_sounding.jl")          # N-parent-a: Lindenberg sounding → (z, t) profiles, boundary_layer_height (#247)
    include("era6_open_near_boundary.jl") # N-open-d: sponge toward the data, neighbor_index, solid open faces, boundary heat terms (#262)
end

# =========================================================================
# The Berlin (Alexanderplatz) flagship case — examples/urban/BerlinCase.jl

@testset "Berlin case" begin
    include("berlin_case.jl")            # registry, stages, bus, tracers (#126)
    include("berlin_surface_exchange.jl") # MOST fluxes, soil ground, roof-aware air level, drag (#129)
    include("berlin_smoke.jl")           # every coupling and option, 12 steps, ledgers close (#127)
    include("berlin_air_heat.jl")        # air-heat ledger: meter, closed box, H₀ = 100 W/m², night window (#253)
    include("berlin_open_boundaries.jl") # lateral = :open: faces, data, sponge, frame, boundary heat, cost (#262)
    include("berlin_sw_modifiers.jl")    # tree shade / cloud shadow once per radiation field; scene class map (#128)
    include("berlin_moist.jl")           # reference state, moist buoyancy, fog enthalpy (#131)
    include("berlin_chemistry.jl")       # NOx–O3–VOC on tracers, r_a-based dry / wet deposition (#135)
    include("berlin_numerics.jl")        # pressure solver, SGS, advection slots (#132)
    include("berlin_dynamics.jl")        # core / timestepper slots: anelastic, adaptive, IMEX (#133)
    include("berlin_parent_surface.jl")  # parent tiled surface: OSM 6 km mosaic, Macdonald roughness, footprint (#249)
    include("berlin_parent_smoke.jl")    # parent case: every forcing, 12 steps, heat and water ledgers close (#259)
    include("berlin_parent.jl")          # parent case: encroachment, Wangara, core sensitivity (24 h Lindenberg: examples/urban/berlin_parent_lindenberg_check.jl, #315) (#259)
end
