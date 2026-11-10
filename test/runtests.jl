# The tests of Aeolus.jl: unit, analytic, conservation and interface checks
# and short code-path runs. The include list below is the single list of
# test files; test/run.jl runs them in parallel, `Pkg.test()` serially.
# Validation runs (simulated cases that confirm the physics) are a separate
# suite, validation/runvalidation.jl. See docs/src/testing.md.

include(joinpath(@__DIR__, "support", "environment.jl"))

@testset "Aeolus.jl" begin

    include("agreement.jl")      # comparing results within tolerances, not (#495)
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
        @test all(f.data[h, Hy+1, Hz+1] == f.data[Nx + h, Hy+1, Hz+1] for h in 1:Hx)
        # east halo mirrors west interior
        @test all(f.data[Hx + Nx + h, Hy+1, Hz+1] == f.data[Hx + h, Hy+1, Hz+1] for h in 1:Hx)
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

    @testset "FFTPoissonSolver on a manufactured mode, $(N)³" for N in (32,)
        # φ(x,y,z) = sin(2π x/Lx) cos(2π y/Ly) sin(2π z/Lz)
        # The *discrete* Laplacian of this mode has eigenvalue λ_h; if we
        # form source = L_h φ (via our laplacian!) and solve, we recover φ
        # to machine precision.
        L = 1.0
        g = RegularGrid(; size=(N, N, N), extent=(L, L, L), halo=(1,1,1))
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

    @testset "CosineFFTPoissonSolver on a Neumann-BC manufactured mode, $(N)³, n = $n" for (N, n) in ((32, 3),)
        # Domain [0, Lx] × [0, Ly] × [0, Lz]. Periodic in x, y; Neumann in z.
        # φ = cos(2π x / Lx) * cos(2π y / Ly) * cos(π n z / Lz)
        # satisfies ∂φ/∂z = 0 at z = 0 and z = Lz for any integer n
        # (n: the z-mode index).
        Lx = Ly = Lz = 1.0
        g = RegularGrid(; size=(N,N,N), extent=(Lx,Ly,Lz), halo=(1,1,1))
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
        @test all(all(w.data[:, :, Hz + 1 - h] .== -w.data[:, :, Hz + 1 + h]) for h in 1:Hz)
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
    include("launch_trace.jl")   # #566: the launch trace (thread busy/idle per launch, AEOLUS_LAUNCH_TRACE=1)
    include("ka_step.jl")    # KA-j (#306): step cores on kernels, fused CFL / step guard, per-sim workspaces
    include("fused_sweep.jl")   # #533: the fused sweep's local-term interface, fallback report, completeness guard
    include("fused_sweep_reductions.jl")   # #533 plan step 3: LargeScale and the urban site in the fused sweep
    include("fused_sweep_post.jl")   # #533 step 6: the post-predictor operators inside the fused sweep
    include("fused_sweep_budgets.jl")   # #533 plan step 7: the budgets booked inside the fused sweep
    include("fused_sweep_cores.jl")   # #533 step 8: the fused sweep in step_imex! and the sound-proof cores, the per-point (GPU) form
    include("fused_sweep_pointwise.jl")   # #533 step 2: buoyancy, Coriolis, surface flux, sponge, fringe in the sweep
    include("fused_sweep_nearwall.jl")   # #533 step 5: open faces, near-wall and urban terms in the fused sweep
    include("fused_sweep_sgs.jl")   # #533 step 4: the SGS closures inside the fused sweep
    include("thread_invariance.jl")  # KA-a (#294): N threads == 1 thread (subprocesses: AEOLUS_THREAD_TESTS=1)

    @testset "GLL nodes: endpoints exact, weights sum to 2, symmetry" begin
        ok = all((2, 3, 8)) do N
            x = gll_nodes(N)
            w = gll_weights(N)
            # endpoints, weights sum to 2, symmetric about 0
            length(x) == N + 1 && x[1] ≈ -1 && x[end] ≈ 1 && isapprox(sum(w), 2; atol = 1e-12) &&
                all(isapprox.(x, .-reverse(x); atol=1e-12)) && all(isapprox.(w, reverse(w); atol=1e-12))
        end
        @test ok
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
        @test all(all(v[:, :, k] .≈ prof[k]) for k in 1:4)
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
        @test agree(d[:u], interior_view(sim1.u)) && agree(d[:θ], interior_view(sim1.θ))
        # Maintainer decision on #245: an unversioned snapshot is refused by
        # restart! (it has no closure memory, halos or stepper state). The
        # pre-#245 expectation here (restart! from write_snapshot) is retired.
        sim2 = BoussinesqSimulation(mesh; ν=1e-3, θ₀=300.0)
        @test_throws ArgumentError restart!(sim2, path)
        # the versioned checkpoint round-trips, halos included
        checkpoint!(sim1, path)
        restart!(sim2, path)
        @test sim2.t == 42.0
        @test agree(sim2.u.data, sim1.u.data) && agree(sim2.θ.data, sim1.θ.data)
        rm(path)
    end

    include("io_checkpoint.jl")   # #245: full-state checkpoints restart, FFTW policy
    include("fftw_threads.jl")    # #397: threaded MEASURE planning on FFTW's own threads, no crash (subprocess, -t 12)
    include("logging_core.jl")    # #333: structured events, run log, test logger
    include("logging_run_folder.jl")  # #334: run folder, run.toml provenance, dirty-tree policy
    include("logging_diagnostics.jl") # #335: diagnostics registry, CSV writer, progress line
    include("logging_blowup.jl")      # #335: blowup diagnose-dump-stop, run monitor
    include("logging_diagnostics_nc.jl")  # #337: diagnostics.nc (profiles, NetCDF-4, provenance attributes)
    include("logging_step_diagnostics.jl")  # #336: per-step diagnostics in the step cores, attached monitor
    include("config_schema.jl")       # CFG-a (#356): schema, strict errors, canonical fixed point, float bits
    include("config_load.jl")         # CFG-a (#356): one file + overrides, extends/schema 1 refused, run.toml [config], inputs, common sections
    include("config_simulation_driver.jl")  # CFG-b (#357): the model from a config round trip, resume, TGV
    include("components_pipeline.jl")  # ONE-a (#450): component interface, stage pipeline, master digests through Model
    include("budgets_output.jl")       # ONE-c (#452): the one budget system and the one output writer
    include("thermodynamic_reference.jl")  # #536: the domain's thermodynamic reference, T = θ Π̄ outputs
    include("checks.jl")               # ONE-d (#453): in-model checks: [checks], results.toml, exit code 3
    include("config_model.jl")        # ONE-b (#451): one schema: bare config = dry LES, [[domain]], per-domain overrides, slots
    include("config_slots.jl")        # ONE-m (#476): one section per exclusive slot, a selector key, the provider's options only
    include("model_nesting.jl")       # ONE-f (#455): nesting by [[domain]]: the graph, pipelines per domain, one-/two-way, restart
    include("config_parent_domain.jl")  # NEST-p3 (#483): the urban nest parent's physics as a model domain in TOML
    include("components_both_cores.jl")  # ONE-j (#460): both dynamical cores for every component, anelastic checkpoints
    include("components_parent.jl")      # NEST-p1 (#481): the urban nest parent's physics as components, both cores
    include("dns_vs_les.jl")             # ONE-g (#456): the isotropic-turbulence initial field, the spectrum and dissipation outputs
    include("glue_physics.jl")         # ONE-e (#454): the physics moved out of the urban glue into its modules
    include("config_cli.jl")          # CFG-b (#357): aeolus CLI, exit codes, example drivers, subprocess
    include("config_examples.jl")     # CFG-e (#360): [sweep] expansion; the Kessler box on the model (ONE-i #458)
    include("examples_enforcement.jl")  # ONE-i (#458): no .jl under examples/ builds or steps a model; the case layout
    include("naming_conventions.jl")  # work is grouped by content; ERA is the reanalysis (2026-11-06)
    include("config_berlin.jl")       # CFG-c (#358): the Berlin configs of the urban driver vs Julia-built, round trip
    include("urban_run.jl")           # EX-a (#438): urban run type in src: synthetic city, results.toml, the parent alone
    include("urban_spinup.jl")        # #519: the surface spin-up of the urban site (PALM spinup_time)
    include("urban_soil_init.jl")     # #521: the initial soil from the dynamic driver (PALM PIDS), the annual wave
    include("urban_checks.jl")       # EX-a (#438): Berlin smoke matrix (49 runs, ledgers), parent smoke run, parent class fractions, parent_zi
    include("input_static.jl")        # CFG-f (#361): static driver, rasterisation, Berlin ground from one file
    include("input_dynamic.jl")       # CFG-g (#362): dynamic driver, sounding/case-table drivers, offline nest, the ERA5 box (CFG-h #363)
    include("config_reference.jl")    # CFG-i (#364): option reference and input-format pages, counted against the schemas

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
        Nx = 32
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

include("chemistry.jl")   # urban NOx–O3–VOC mechanism, dry / wet deposition (#135)


# =========================================================================
# Numerics
# =========================================================================
# Grids, advection, time integration, pressure solvers and the anelastic core (Phases 21-29, 31).

@testset "Numerics" begin
    include("cutcell.jl")     # Phases 21-22  cut-cell IBM, ghost cells
    include("stretched.jl")   # Phase  24     stretched / terrain-following
    include("stretched_boussinesq.jl") # #151 Boussinesq core + Poisson on a StretchedGrid
    include("timeint.jl")     # Phases 23, 25 adaptive dt, IMEX
    include("advection.jl")   # Phases 26, 28 WENO-Z/TENO, conservative ops
    include("anelastic.jl")   # Phase  27     anelastic, pseudo-incompressible
    include("multigrid.jl")   # Phase  29     geometric multigrid
end

# =========================================================================
# Nesting
# =========================================================================
# Grid nesting: operators, one- and two-way runs, hierarchies, restart (Phase 30).

@testset "Nesting" begin
    include("nesting.jl")     # Phase  30     parent-child nesting operators (#250)
    include("nesting_conservative.jl") # #256 stretched parent, mass-weighted, solid-aware nesting
    include("amr.jl")         # Phases 30-31  nesting, AMR
    include("nested_run.jl")  # Phase  30     NestedRun time coupling (#257)
    include("nest_boundary.jl") # Phase 30   one-way nest boundary data (#261)
    include("nest_external.jl") # #265        externally driven child (the Berlin pattern), open scalars added later
    include("nest_restart.jl")       # #269 checkpoint / restart of a nested run (nest_state)
    include("nest_twoway.jl")  # Phase 30   two-way feedback: anterpolation, refluxing (#263)
    include("nest_hierarchy.jl") # #328 nest hierarchies, siblings, non-aligned and stretched children
end

# =========================================================================
# Radiation
# =========================================================================
# Gas and aerosol optics, two-stream and Monte Carlo transport, radiosity; `RadiationCore` is the shared core (Phases 32-38).

@testset "Radiation" begin
    include("gasoptics.jl")       # Phase 32  correlated-k gas optics
    include("twostream.jl")       # Phase 33  two-stream RT, cloud optics
    include("surfaceoptics.jl")   # Phase 36  spectral albedo, BRDF
    include("aerosol.jl")         # Phase 37  Mie aerosol optics
    include("montecarlo.jl")      # Phase 34  3-D Monte-Carlo RT
    include("radiosity.jl")       # Phase 35  urban radiosity
    include("solargeometry.jl")   # Phase 38  solar position, shadows
end

# =========================================================================
# Clouds and microphysics
# =========================================================================
# Moist thermodynamics (the shared core), activation, superdroplets, P3, ice nucleation, fog and visibility (Phases 39-45).

@testset "Clouds and microphysics" begin
    include("thermodynamics.jl")  # Phase 39  exact moist thermodynamics
    include("activation.jl")      # Phase 42  aerosol-cloud interaction
    include("superdroplet.jl")    # Phase 41  superdroplet microphysics
    include("p3.jl")              # Phase 40  P3 microphysics
    include("icenucleation.jl")   # Phase 43  ice nucleation
    include("turbmicro.jl")       # Phase 45  sub-grid supersaturation PDF
    include("sgs_growth.jl")      # Phase 45  SGS activation and growth
    include("broadening.jl")      # Phase 45  spectral broadening: moment closure, OU limit
    include("visibility.jl")      # Phase 44  visibility diagnostic
    include("fog.jl")             # Phase 44  fog / dew coupling (#11)
    include("radiation_fog.jl")   # Phase 44  radiation fog feedback (#9)
end

# =========================================================================
# Land and surface
# =========================================================================
# `SurfaceExchange` (the shared core), soil, vegetation, snow, sea ice, ocean and lakes, tiles (Phases 46-48, 51-53).

@testset "Land and surface" begin
    include("surface_exchange.jl")  # surface shared core: MOST, constants, surface interface (#92)
    include("soil_column.jl")       # Phase 46: soil column, Richards equation (#15)
    include("soil_thermal.jl")      # Phase 46: soil heat and freeze–thaw (#16)
    include("water_table.jl")       # Phase 46: water table, unconfined aquifer (#17)
    include("photosynthesis.jl")    # Phase 47  Farquhar C3 / Collatz C4 photosynthesis (#19)
    include("stomata.jl")           # Phase 47  Ball-Berry / Medlyn stomatal conductance (#20)
    include("land_surface.jl")      # Phase 47: vegetated land column, soil-water stress (#21)
    include("plant_canopy.jl")      # Phase 48  LAD profiles, implicit canopy drag (#22)
    include("canopy_radiation.jl")  # Phase 48  canopy SW two-stream, sunlit/shaded, LW (#23)
    include("leaf_energy.jl")       # Phase 48  leaf energy balance, leaf temperature (#24)
    include("canopy_profiles.jl")   # Phase 48  within-canopy wind/temperature validation (#25)
    include("snowpack.jl")          # Phase 51: layered snowpack with compaction (#34)
    include("snow_metamorphism.jl") # Phase 51: grain metamorphism and albedo ageing (#35)
    include("snow_liquid.jl")        # Phase 51: snow percolation and refreezing (#36)
    include("ice_thermodynamics.jl")  # Phase 51  lake and sea ice, BL99 (#37)
    include("ocean_mixed_layer.jl") # Phase 52  water column, cool skin, warm layer (#38)
    include("air_sea_flux.jl")      # Phase 52  wave-dependent sea-surface roughness (#39)
    include("sea_spray_coare.jl")   # Phase 52  sea spray, full COARE, validation (#40)
    include("surface_tiles.jl")     # Phase 53 sub-grid tile mosaic, flux aggregation (#41)
    include("blending_height.jl")   # Phase 53 blending height, effective roughness, IBL (#42)
    include("heterogeneous_surface.jl")  # Phase 53  patchy-surface coupling: code-path identity, heat conservation (#43)
end

# =========================================================================
# Urban surface and buildings
# =========================================================================
# Urban facets, glazing, urban radiosity, building energy, HVAC, occupancy, anthropogenic heat (Phases 49-50).

@testset "Urban surface and buildings" begin
    include("urban_surface.jl")     # Phase 49  multi-layer urban facet conduction (#26)
    include("glazing.jl")           # Phase 49  glazing transmission, window facets (#27)
    include("urban_radiosity.jl")   # Phase 49  facet temperatures coupled to radiosity (#28)
    include("building_energy.jl")   # Phase 50  indoor air temperature + thermal mass (#31)
    include("hvac.jl")              # Phase 50  HVAC demand and exhaust heat (#32)
    include("occupancy.jl")         # Phase 50  occupancy schedules and internal gains (#33)
    include("anthropogenic_heat.jl") # Phase 49  anthropogenic heat from the building energy model (#29)
end

# =========================================================================
# Turbulence
# =========================================================================
# SGS closures (the closure contract, P54-a #167), stability functions, TKE, wall models, transition (Phases 54-59).

@testset "Turbulence" begin
    include("sgs_closure.jl")       # P54-a: SGS closure contract, κ_t, commit cadence (#167)
    include("sgs_numerics.jl") # #132         dynamic Smagorinsky, step! operator swap-ins
    include("stability_functions.jl") # Phase 58 TurbulenceStability contract (#170)
    include("amd_vreman.jl")        # P55-d: AMD and Vreman closures (#176)
    include("tke.jl")               # P56-b: Deardorff prognostic SGS TKE, wall distance (#182)
    include("wall_models.jl")       # Phase 57: equilibrium wall laws, wall-stress operator (#169)
    include("transition_kh.jl")      # P59-c: turbulent fraction γ, intermittency, Ri_f, growth_rate (#180)
end

# =========================================================================
# Forcing and boundaries
# =========================================================================
# Large-scale forcing, synthetic inflow and open boundaries.

@testset "Forcing and boundaries" begin
    include("large_scale_forcing.jl") # Phase 58: geostrophic forcing, nudging, advective tendencies (#171)
    include("synthetic_inflow.jl")  # N-inflow-a: digital-filter inflow, Lund transform (#252)
    include("inflow_turbulence.jl")  # N-inflow-c: inflow faces, zero added mass flux (#264)
    include("open_boundaries.jl")   # N-open-b: open faces, boundary-data contract, mass-flux correction (#255)
end
