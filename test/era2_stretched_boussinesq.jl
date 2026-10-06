# =========================================================================
# #151 (N-grid) — BoussinesqSimulation on a vertically stretched grid
# =========================================================================
# What the Berlin nesting parent (#246) needs from the core: a 3 km deep
# domain with Δz = 25 m near the ground, stretched above. Validation:
#
#   1. Layout      — `surface_refined_faces` builds the parent grid exactly.
#   2. Reduction   — a `:uniform` StretchedGrid reproduces the RegularGrid
#                    run to 1e-12 over 20 steps (every coupling on).
#   3. Projection  — the direct stretched Poisson solve is exact: the
#                    projected velocity is solenoidal to < 1e-10 on the
#                    parent-like grid; the multigrid `coeff_z` route gives
#                    the same pressure; a manufactured φ converges at 2nd
#                    order.
#   4. Conservation— momentum advection conserves kinetic energy for a
#                    solenoidal field, scalar advection conserves mass and
#                    variance, eddy diffusion conserves mass; a stratified
#                    fluid at rest stays at rest.
#   5. Convergence — a manufactured Navier–Stokes solution on a geometric
#                    stretching (ratio 1.08 on the coarsest grid) converges
#                    at second order through the full projected RK3 step.
#   6. Closures    — Smagorinsky ν_t of a linear shear with the local
#                    filter width; the adaptive dt sees the thinnest cell.

using Test
using Random
using Aeolus
using Aeolus.StretchedGrids
using Aeolus.Grids: interior
using Aeolus.Fields: Field, interior_view
using Aeolus.BoundaryConditions: fill_halos!
using Aeolus.Operators: divergence, momentum_advect!, advect_centered!

const FV151  = Aeolus.Operators.FiniteVolume
const SIM151 = Aeolus.Simulations

# the parent grid of the nesting epic (maintainer decision 2026-10-05:
# uniform 25 m to 500 m, ×1.08 per level, capped at 100 m, top 3 km)
_parent_faces() = surface_refined_faces(dz_surface = 25, z_uniform = 500, z_top = 3000,
                                        ratio = 1.08, dz_max = 100)
_xface(g, i)   = g.x0 + (i - g.Hx - 1) * g.dx
_xcent(g, i)   = g.x0 + (i - g.Hx - 0.5) * g.dx

function _random_state!(sim; seed = 1, U = 0.0, θamp = 0.0)
    Random.seed!(seed)
    for f in (sim.u, sim.v, sim.w)
        interior_view(f) .= randn(size(interior_view(f)))
    end
    interior_view(sim.u) .+= U
    sim.θ₀ > 0 && (interior_view(sim.θ) .= sim.θ₀ .+ θamp .* randn(size(interior_view(sim.θ))))
    return sim
end

# volume-weighted inner products of the staggered grid: u, v, θ cells have
# height dzc[k], w cells dzf[k]
function _ke_rate(g, tu, tv, tw, u, v, w)
    ii, jj, kk = interior(g)
    s = 0.0
    for k in kk, j in jj, i in ii
        s += (u.data[i,j,k]*tu.data[i,j,k] + v.data[i,j,k]*tv.data[i,j,k]) * g.dzc[k] +
             w.data[i,j,k]*tw.data[i,j,k] * g.dzf[k]
    end
    return s * g.dx * g.dy
end
function _vsum(g, a)
    ii, jj, kk = interior(g)
    return sum(a.data[i,j,k] * g.dzc[k] for k in kk, j in jj, i in ii) * g.dx * g.dy
end

@testset "#151 Boussinesq core on a StretchedGrid" begin

    @testset "parent layout: surface_refined_faces" begin
        zf  = _parent_faces()
        dz  = diff(zf)
        @test length(dz) == 53                       # 20 uniform + 33 stretched
        @test all(dz[1:20] .== 25.0)
        @test zf[21] == 500.0 && zf[end] == 3000.0
        @test maximum(dz) ≤ 100.0 + 1e-12
        # the stretched cells keep the common ratio until the cap
        r = dz[22:end] ./ dz[21:end-1]
        @test all(x -> isapprox(x, 1.08; atol = 1e-12) || x < 1.08, r)
        @test maximum(r) ≤ 1.08 + 1e-12
        @test grid_stretching_ratio(StretchedGrid(; size = (4, 4, 53), extent = (100.0, 100.0, 3000.0),
                                                   stretching = (:custom, zf))) ≤ 1.08 + 1e-12
        @test_throws ArgumentError surface_refined_faces(dz_surface = 25, z_uniform = 510,
                                                          z_top = 3000)
    end

    @testset "uniform StretchedGrid reproduces RegularGrid (20 steps)" begin
        kw = (size = (12, 10, 16), extent = (600.0, 500.0, 400.0))
        rg = RegularGrid(; kw...)
        sg = StretchedGrid(; kw..., stretching = :uniform)
        bcs = FieldBoundaryConditions(:periodic, :periodic; bottom = :noslip, top = :wall)
        mk(m) = _random_state!(BoussinesqSimulation(m; ν = 1e-2, θ₀ = 300, C_s = 0.17,
                                   f_coriolis = 1e-4, moisture = true, θ_flux_bottom = 0.1,
                                   sponge_α_max = 0.01, sponge_z_start = 300.0, bcs = bcs);
                               U = 3.0, θamp = 0.2)
        a = mk(rg); b = mk(sg)
        interior_view(a.qv) .= 0.008; interior_view(b.qv) .= 0.008
        for n in 1:20
            step!(a, 0.5); step!(b, 0.5)
        end
        for (fa, fb) in ((a.u, b.u), (a.v, b.v), (a.w, b.w), (a.θ, b.θ), (a.qv, b.qv), (a.p, b.p))
            @test maximum(abs, interior_view(fa) .- interior_view(fb)) < 1e-12
        end
        @test kinetic_energy(a) ≈ kinetic_energy(b) rtol = 1e-13
    end

    @testset "projection: divergence-free on the parent-like grid" begin
        zf = _parent_faces()
        g  = StretchedGrid(; size = (32, 32, 53), extent = (800.0, 800.0, 3000.0),
                             stretching = (:custom, zf))
        # (non-periodic sides are not divergence-free on the RegularGrid core
        # either; open lateral boundaries are N-open-a, #251)
        for bcs in (FieldBoundaryConditions(:periodic, :periodic; bottom = :noslip, top = :wall),
                    FieldBoundaryConditions(:periodic, :periodic; bottom = :wall, top = :wall))
            sim = _random_state!(BoussinesqSimulation(g; ν = 1e-2, θ₀ = 300, C_s = 0.17, bcs = bcs);
                                 U = 5.0, θamp = 0.1)
            @test sim.solver isa StretchedPoissonSolver
            for n in 1:3
                step!(sim, 0.5)
            end
            @test max_divergence(sim) < 1e-10
        end
    end

    @testset "stretched Poisson: manufactured φ, 2nd order; multigrid coeff_z agrees" begin
        # φ = cos(2πx/L) cos(πz/H) (Neumann walls), ∇²φ sampled at the centres,
        # on the smooth geometric map with ratio 1.08 at N = 16 (r^N fixed)
        L = 1.0; H = 1.0
        errs = Float64[]
        for N in (16, 32, 64)
            r  = 1.08^(16 / N)
            g  = StretchedGrid(; size = (N, 2, N), extent = (L, L * 2 / N, H),
                                 stretching = (:geometric, r))
            src = Field(CellCenter(), g); φ = Field(CellCenter(), g)
            ii, jj, kk = interior(g)
            k2 = (2π / L)^2 + (π / H)^2
            for k in kk, j in jj, i in ii
                src.data[i,j,k] = -k2 * cos(2π * _xcent(g, i) / L) * cos(π * g.zc[k] / H)
            end
            solve!(φ, src, StretchedPoissonSolver(g))
            e = 0.0
            for k in kk, j in jj, i in ii
                e = max(e, abs(φ.data[i,j,k] - cos(2π * _xcent(g, i) / L) * cos(π * g.zc[k] / H)))
            end
            push!(errs, e)
        end
        orders = log2.(errs[1:end-1] ./ errs[2:end])
        @info "#151 stretched Poisson MMS" errs orders
        @test all(o -> abs(o - 2) < 0.15, orders)

        # Neumann sides (the DCT-II branch): φ = cos(πx/L) cos(πz/H)
        errs = Float64[]
        for N in (16, 32, 64)
            g  = StretchedGrid(; size = (N, 2, N), extent = (L, L * 2 / N, H),
                                 stretching = (:geometric, 1.08^(16 / N)))
            src = Field(CellCenter(), g); φ = Field(CellCenter(), g)
            ii, jj, kk = interior(g)
            ex(i, k) = cos(π * _xcent(g, i) / L) * cos(π * g.zc[k] / H)
            for k in kk, j in jj, i in ii
                src.data[i,j,k] = -((π / L)^2 + (π / H)^2) * ex(i, k)
            end
            solve!(φ, src, StretchedPoissonSolver(g; bc_x = :neumann, bc_y = :periodic))
            push!(errs, maximum(abs(φ.data[i,j,k] - ex(i, k)) for k in kk, j in jj, i in ii))
        end
        orders = log2.(errs[1:end-1] ./ errs[2:end])
        @info "#151 stretched Poisson MMS, Neumann sides" errs orders
        @test all(o -> abs(o - 2) < 0.15, orders)

        # the multigrid route (coeff_z / rhs_scale on the equivalent uniform
        # grid) solves the same discrete operator: same φ up to a constant
        zf = surface_refined_faces(dz_surface = 25, z_uniform = 200, z_top = 1000,
                                   ratio = 1.08, dz_max = 100)
        g  = StretchedGrid(; size = (16, 16, length(zf) - 1), extent = (400.0, 400.0, 1000.0),
                             stretching = (:custom, zf))
        Random.seed!(3)
        src = Field(CellCenter(), g)
        ii, jj, kk = interior(g)
        interior_view(src) .= randn(length(ii), length(jj), length(kk))
        # compatibility: zero volume-weighted mean
        m = sum(src.data[i,j,k] * g.dzc[k] for k in kk, j in jj, i in ii) / (length(ii)*length(jj)*g.Lz)
        interior_view(src) .-= m
        φd = Field(CellCenter(), g); φm = Field(CellCenter(), g)
        solve!(φd, src, StretchedPoissonSolver(g))
        solve!(φm, src, MGPreconditionedCG(g; bc_x = :periodic, bc_y = :periodic, bc_z = :neumann,
                                           tol = 1e-12))
        a = interior_view(φd) .- sum(interior_view(φd)) / length(interior_view(φd))
        b = interior_view(φm) .- sum(interior_view(φm)) / length(interior_view(φm))
        @test maximum(abs, a .- b) < 1e-8 * maximum(abs, a)
    end

    @testset "conservation: advection and diffusion on a stretched grid" begin
        zf = _parent_faces()
        g  = StretchedGrid(; size = (16, 16, 53), extent = (400.0, 400.0, 3000.0),
                             stretching = (:custom, zf))
        bcs = FieldBoundaryConditions(:periodic, :periodic; bottom = :wall, top = :wall)
        sim = _random_state!(BoussinesqSimulation(g; θ₀ = 300, bcs = bcs); seed = 7, θamp = 1.0)
        step!(sim, 1e-6)                      # project onto the solenoidal subspace
        @test max_divergence(sim) < 1e-12
        u, v, w = sim.u, sim.v, sim.w
        for f in (u, v, w); fill_halos!(f, bcs); end
        fill_halos!(sim.θ, bcs)
        tu = Field(FaceX(), g); tv = Field(FaceY(), g); tw = Field(FaceZ(), g)
        momentum_advect!(tu, tv, tw, u, v, w)
        E = kinetic_energy(sim)
        dE = _ke_rate(g, tu, tv, tw, u, v, w)
        # dE/dt of advection alone vanishes (skew-symmetric operator)
        @test abs(dE) < 1e-12 * E
        tθ = Field(CellCenter(), g)
        advect_centered!(tθ, u, v, w, sim.θ)
        θm = _vsum(g, sim.θ)
        @test abs(_vsum(g, tθ)) < 1e-12 * abs(θm)              # Σ ϕ ΔV
        ii, jj, kk = interior(g)
        var_rate = sum(sim.θ.data[i,j,k] * tθ.data[i,j,k] * g.dzc[k] for k in kk, j in jj, i in ii)
        @test abs(var_rate) < 1e-11 * sum(sim.θ.data[i,j,k]^2 * g.dzc[k] for k in kk, j in jj, i in ii)
        # eddy diffusion: Σ tϕ ΔV telescopes to the (zero) wall fluxes
        K = Field(CellCenter(), g); interior_view(K) .= 1 .+ rand(size(interior_view(K))...)
        fill_halos!(K, bcs)
        td = Field(CellCenter(), g)
        FV151._add_eddy_diffusion_scalar!(td, sim.θ, K, 1.0, g)
        @test abs(_vsum(g, td)) < 1e-11 * maximum(abs, interior_view(td)) * g.Lx * g.Ly * g.Lz
        # and it dissipates: Σ ϕ tϕ ΔV < 0
        @test sum(sim.θ.data[i,j,k] * td.data[i,j,k] * g.dzc[k] for k in kk, j in jj, i in ii) < 0
    end

    @testset "stratified rest state stays at rest" begin
        zf = _parent_faces()
        g  = StretchedGrid(; size = (8, 8, 53), extent = (200.0, 200.0, 3000.0),
                             stretching = (:custom, zf))
        bcs = FieldBoundaryConditions(:periodic, :periodic; bottom = :noslip, top = :wall)
        sim = BoussinesqSimulation(g; ν = 1e-2, θ₀ = 300, C_s = 0.17, bcs = bcs)
        for k in axes(sim.θ.data, 3)
            sim.θ.data[:, :, k] .= 300 + 0.004 * g.zc[k] + 2 * exp(-g.zc[k] / 200)
        end
        for n in 1:20
            step!(sim, 2.0)
        end
        @test maximum(abs, interior_view(sim.w)) < 1e-12
        @test maximum(abs, interior_view(sim.u)) < 1e-12
    end

    @testset "manufactured Navier–Stokes solution: 2nd order (ratio 1.08)" begin
        # 2-D cellular flow ψ = Σₙ aₙ sin(nkx) sin(nmz), m = π/H, between
        # free-slip walls; F = (u·∇)u − ν∇²u makes it a steady solution with
        # p = const. Run the full projected RK3 step to t = T with dt ∝ Δ.
        # (amplitudes 1.0/0.4 and T = 0.2 left N = 16 outside the asymptotic
        # range on the *uniform* grid too: observed order 2.63 then 2.01.)
        L = 1.0; H = 1.0; ν = 0.02; T = 0.1
        k = 2π / L; mz = π / H
        A = ((1, 0.5), (2, 0.2))
        uex(x, z) = sum(a * n * mz * sin(n*k*x) * cos(n*mz*z) for (n, a) in A)
        wex(x, z) = -sum(a * n * k * cos(n*k*x) * sin(n*mz*z) for (n, a) in A)
        ux(x, z)  = sum(a * n^2 * k * mz * cos(n*k*x) * cos(n*mz*z) for (n, a) in A)
        uz(x, z)  = -sum(a * n^2 * mz^2 * sin(n*k*x) * sin(n*mz*z) for (n, a) in A)
        wx(x, z)  = sum(a * n^2 * k^2 * sin(n*k*x) * sin(n*mz*z) for (n, a) in A)
        lapu(x, z) = -sum(a * n * mz * n^2 * (k^2 + mz^2) * sin(n*k*x) * cos(n*mz*z) for (n, a) in A)
        lapw(x, z) = sum(a * n * k * n^2 * (k^2 + mz^2) * cos(n*k*x) * sin(n*mz*z) for (n, a) in A)
        Fu(x, z) = uex(x, z) * ux(x, z) + wex(x, z) * uz(x, z) - ν * lapu(x, z)
        Fw(x, z) = uex(x, z) * wx(x, z) + wex(x, z) * (-ux(x, z)) - ν * lapw(x, z)

        errs = Float64[]
        for N in (16, 32, 64)
            r = 1.08^(16 / N)
            g = StretchedGrid(; size = (N, 2, N), extent = (L, 2L / N, H),
                                stretching = (:geometric, r))
            bcs = FieldBoundaryConditions(:periodic, :periodic; bottom = :wall, top = :wall)
            sim = BoussinesqSimulation(g; ν = ν, bcs = bcs)
            ii, jj, kk = interior(g)
            fu = zeros(size(sim.u.data)); fw = zeros(size(sim.w.data))
            for kq in kk, j in jj, i in ii
                sim.u.data[i,j,kq] = uex(_xface(g, i), g.zc[kq]); fu[i,j,kq] = Fu(_xface(g, i), g.zc[kq])
                sim.w.data[i,j,kq] = wex(_xcent(g, i), g.zf[kq]); fw[i,j,kq] = Fw(_xcent(g, i), g.zf[kq])
            end
            force!(s, Δt) = (s.tu.data .+= fu; s.tw.data .+= fw; nothing)
            dzmin = minimum(diff(g.zf[g.Hz+1:g.Hz+N+1]))
            nsteps = ceil(Int, T / min(0.1 * dzmin, 0.1 * dzmin^2 / ν))   # advective, diffusive
            dt = T / nsteps
            for n in 1:nsteps
                step_rk3!(sim, dt; body_force! = force!)
            end
            e = 0.0
            for kq in kk, j in jj, i in ii
                e = max(e, abs(sim.u.data[i,j,kq] - uex(_xface(g, i), g.zc[kq])),
                           abs(sim.w.data[i,j,kq] - wex(_xcent(g, i), g.zf[kq])))
            end
            push!(errs, e)
        end
        orders = log2.(errs[1:end-1] ./ errs[2:end])
        @info "#151 Navier–Stokes MMS on a stretched grid" errs orders
        @test all(o -> abs(o - 2) < 0.15, orders)
    end

    @testset "Smagorinsky with the local filter width; adaptive dt" begin
        zf = _parent_faces()
        g  = StretchedGrid(; size = (8, 8, 53), extent = (200.0, 200.0, 3000.0),
                             stretching = (:custom, zf))
        bcs = FieldBoundaryConditions(:periodic, :periodic; bottom = :wall, top = :wall)
        u = Field(FaceX(), g); v = Field(FaceY(), g); w = Field(FaceZ(), g)
        S = 0.01
        for k in axes(u.data, 3)
            u.data[:, :, k] .= S * g.zc[k]           # linear shear, |S| = S exactly
        end
        νt = Field(CellCenter(), g)
        Cs = 0.17
        FV151._compute_smagorinsky_νt!(νt, u, v, w, Cs, g)
        ii, jj, kk = interior(g)
        rel = maximum(abs(νt.data[ii[1], jj[1], k] / ((Cs * cbrt(g.dx*g.dy*g.dzc[k]))^2 * S) - 1)
                      for k in kk[2:end-1])
        @test rel < 1e-12
        # the local width: the 100 m cells get (100/25)^(2/3) = 2.52× the ν_t of the 25 m cells
        @test νt.data[ii[1], jj[1], kk[end-1]] / νt.data[ii[1], jj[1], kk[2]] ≈
              (g.dzc[kk[end-1]] / 25)^(2/3) rtol = 1e-12

        # adaptive dt: a uniform w through the column is limited by the
        # thinnest (25 m) cell, not by the mean spacing (3000/53 = 57 m)
        sim = BoussinesqSimulation(g; bcs = bcs)
        interior_view(sim.w) .= 0.0
        sim.w.data[:, :, kk[2]:kk[end]] .= 2.0
        c = CFLConstraints(C_advective = 0.5)
        d = cfl_diagnostics(sim, c)
        @test d.advective_rate ≈ 2.0 / 25 rtol = 1e-12
        @test adaptive_dt(sim, c) ≈ 0.5 * 25 / 2.0 rtol = 1e-12
        sim.w.data .= 0
        _random_state!(sim; seed = 11)
        dt0 = adaptive_dt(sim, c)
        step_adaptive!(sim, c)
        @test max_divergence(sim) < 1e-10
        @test sim.t ≈ dt0
    end

    @testset "out-of-scope paths fail by name, not by MethodError" begin
        g = StretchedGrid(; size = (4, 4, 8), extent = (1.0, 1.0, 1.0), stretching = (:geometric, 1.1))
        bcs = FieldBoundaryConditions(:periodic, :periodic; bottom = :wall, top = :wall)
        sim = BoussinesqSimulation(g; ν = 1e-3, C_s = 0.17, bcs = bcs)
        @test_throws ArgumentError step!(sim, 0.01; sgs_stress = :symmetric)
        @test_throws ArgumentError step_imex!(sim, 0.01, ARK2())
        @test_throws ArgumentError BoussinesqSimulation(
            StretchedGrid(; size = (4, 4, 8), extent = (1.0, 1.0, 1.0), stretching = (:geometric, 1.1));
            bcs = FieldBoundaryConditions(:periodic))
    end
end
