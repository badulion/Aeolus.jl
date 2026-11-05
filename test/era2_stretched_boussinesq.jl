# =========================================================================
# #151 (N-grid) — BoussinesqSimulation on a vertically stretched grid
# =========================================================================
# Parent-grid layout, the direct stretched Poisson solve (2nd order, agrees
# with the multigrid coeff_z route, solenoidal projection), conservation of
# advection/diffusion, the rest state, Smagorinsky with the local filter
# width and the adaptive dt, and the transition diagnostics (#505).

using Test
using Random
using Aeolus
using Aeolus.StretchedGrids
using Aeolus.Grids: interior
using Aeolus.Fields: Field, interior_view
using Aeolus.BoundaryConditions: fill_halos!
using Aeolus.Operators: momentum_advect!, advect_centered!
using Aeolus.Transition: flux_richardson, turbulent_fraction
using Aeolus.Grids: FaceX, FaceY, FaceZ, CellCenter

const FV151 = Aeolus.Operators.FiniteVolume

# the parent grid of the nesting epic: uniform 25 m to 500 m, ×1.08, ≤ 100 m, top 3 km
_parent_faces() = surface_refined_faces(dz_surface = 25, z_uniform = 500, z_top = 3000,
                                        ratio = 1.08, dz_max = 100)
_xface(g, i) = g.x0 + (i - g.Hx - 1) * g.dx
_xcent(g, i) = g.x0 + (i - g.Hx - 0.5) * g.dx

function _random_state!(sim; seed = 1, U = 0.0, θamp = 0.0)
    Random.seed!(seed)
    for f in (sim.u, sim.v, sim.w)
        interior_view(f) .= randn(size(interior_view(f)))
    end
    interior_view(sim.u) .+= U
    sim.θ₀ > 0 && (interior_view(sim.θ) .= sim.θ₀ .+ θamp .* randn(size(interior_view(sim.θ))))
    return sim
end
function _vsum(g, a)
    ii, jj, kk = interior(g)
    return sum(a.data[i,j,k] * g.dzc[k] for k in kk, j in jj, i in ii) * g.dx * g.dy
end

@testset "#151 Boussinesq core on a StretchedGrid" begin

    @testset "parent layout: surface_refined_faces" begin
        dz = diff(_parent_faces())
        @test length(dz) == 53 && all(dz[1:20] .== 25.0)   # 20 uniform + 33 stretched
        @test maximum(dz) ≤ 100.0 + 1e-12
        @test maximum(dz[22:end] ./ dz[21:end-1]) ≤ 1.08 + 1e-12
    end

    @testset "stretched Poisson: manufactured φ, 2nd order; multigrid agrees" begin
        # φ = cos(2πx/L) cos(πz/H) (Neumann walls) on a geometric map with
        # ratio 1.08 at N = 16 (r^N fixed under refinement)
        errs = map((16, 32)) do N
            g  = StretchedGrid(; size = (N, 2, N), extent = (1.0, 2 / N, 1.0),
                                 stretching = (:geometric, 1.08^(16 / N)))
            src = Field(CellCenter(), g); φ = Field(CellCenter(), g)
            ii, jj, kk = interior(g)
            ex(i, k) = cos(2π * _xcent(g, i)) * cos(π * g.zc[k])
            for k in kk, j in jj, i in ii
                src.data[i,j,k] = -((2π)^2 + π^2) * ex(i, k)
            end
            solve!(φ, src, StretchedPoissonSolver(g))
            maximum(abs(φ.data[i,j,k] - ex(i, k)) for k in kk, j in jj, i in ii)
        end
        @test abs(log2(errs[1] / errs[2]) - 2) < 0.15

        # multigrid route (coeff_z on the equivalent uniform grid): same φ up to a constant
        zf = surface_refined_faces(dz_surface = 25, z_uniform = 200, z_top = 1000,
                                   ratio = 1.08, dz_max = 100)
        g  = StretchedGrid(; size = (8, 8, length(zf) - 1), extent = (200.0, 200.0, 1000.0),
                             stretching = (:custom, zf))
        Random.seed!(3)
        src = Field(CellCenter(), g)
        ii, jj, kk = interior(g)
        interior_view(src) .= randn(length(ii), length(jj), length(kk))
        interior_view(src) .-= _vsum(g, src) / (g.Lx * g.Ly * g.Lz)   # compatibility
        φd = Field(CellCenter(), g); φm = Field(CellCenter(), g)
        solve!(φd, src, StretchedPoissonSolver(g))
        solve!(φm, src, MGPreconditionedCG(g; bc_x = :periodic, bc_y = :periodic, bc_z = :neumann,
                                           tol = 1e-12))
        a = interior_view(φd) .- sum(interior_view(φd)) / length(interior_view(φd))
        b = interior_view(φm) .- sum(interior_view(φm)) / length(interior_view(φm))
        @test maximum(abs, a .- b) < 1e-8 * maximum(abs, a)
    end

    @testset "projection, conservation, rest state on the parent-like grid" begin
        g  = StretchedGrid(; size = (8, 8, 53), extent = (200.0, 200.0, 3000.0),
                             stretching = (:custom, _parent_faces()))
        bcs = FieldBoundaryConditions(:periodic, :periodic; bottom = :wall, top = :wall)
        sim = _random_state!(BoussinesqSimulation(g; θ₀ = 300, bcs = bcs); seed = 7, θamp = 1.0)
        @test sim.solver isa StretchedPoissonSolver
        step!(sim, 1e-6)                      # project onto the solenoidal subspace
        @test max_divergence(sim) < 1e-12
        u, v, w = sim.u, sim.v, sim.w
        for f in (u, v, w, sim.θ); fill_halos!(f, bcs); end
        # skew-symmetric momentum advection: dE/dt = 0 (volume-weighted, w cells dzf)
        tu = Field(FaceX(), g); tv = Field(FaceY(), g); tw = Field(FaceZ(), g)
        momentum_advect!(tu, tv, tw, u, v, w)
        ii, jj, kk = interior(g)
        dE = sum((u.data[i,j,k]*tu.data[i,j,k] + v.data[i,j,k]*tv.data[i,j,k]) * g.dzc[k] +
                 w.data[i,j,k]*tw.data[i,j,k] * g.dzf[k] for k in kk, j in jj, i in ii) * g.dx * g.dy
        @test abs(dE) < 1e-12 * kinetic_energy(sim)
        # scalar advection conserves Σ ϕ ΔV
        tθ = Field(CellCenter(), g)
        advect_centered!(tθ, u, v, w, sim.θ)
        @test abs(_vsum(g, tθ)) < 1e-12 * abs(_vsum(g, sim.θ))

        # a stratified fluid at rest stays at rest
        rest = BoussinesqSimulation(g; ν = 1e-2, θ₀ = 300, C_s = 0.17,
                                    bcs = FieldBoundaryConditions(:periodic, :periodic;
                                                                  bottom = :noslip, top = :wall))
        for k in axes(rest.θ.data, 3)
            rest.θ.data[:, :, k] .= 300 + 0.004 * g.zc[k] + 2 * exp(-g.zc[k] / 200)
        end
        for n in 1:3
            step!(rest, 2.0)
        end
        @test maximum(abs, interior_view(rest.w)) < 1e-12
    end

    @testset "Smagorinsky with the local filter width; adaptive dt" begin
        g  = StretchedGrid(; size = (4, 4, 53), extent = (100.0, 100.0, 3000.0),
                             stretching = (:custom, _parent_faces()))
        bcs = FieldBoundaryConditions(:periodic, :periodic; bottom = :wall, top = :wall)
        u = Field(FaceX(), g); v = Field(FaceY(), g); w = Field(FaceZ(), g)
        S, Cs = 0.01, 0.17
        for k in axes(u.data, 3)
            u.data[:, :, k] .= S * g.zc[k]           # linear shear, |S| = S exactly
        end
        νt = Field(CellCenter(), g)
        FV151._compute_smagorinsky_νt!(νt, u, v, w, Cs, g)
        ii, jj, kk = interior(g)
        @test maximum(abs(νt.data[ii[1], jj[1], k] / ((Cs * cbrt(g.dx*g.dy*g.dzc[k]))^2 * S) - 1)
                      for k in kk[2:end-1]) < 1e-12
        # adaptive dt is limited by the thinnest (25 m) cell, not the mean (57 m)
        sim = BoussinesqSimulation(g; bcs = bcs)
        sim.w.data[:, :, kk[2]:kk[end]] .= 2.0
        @test adaptive_dt(sim, CFLConstraints(C_advective = 0.5)) ≈ 0.5 * 25 / 2.0 rtol = 1e-12
    end

    @testset "transition diagnostics: analytic on a surface-refined grid (#505)" begin
        # u = S z + a sin kx, w = b sin kx, θ = θ₀ + Γ z + c sin kx, constant ν_t:
        # ⟨u′w′⟩ = ab cos(kΔx/2)/2 − ν_t S, ⟨w′θ′⟩ = bc/2 − (ν_t/Pr_t) Γ, P = −⟨u′w′⟩ S
        zf = surface_refined_faces(dz_surface = 10, z_uniform = 50, z_top = 800, ratio = 1.15, dz_max = 60)
        Nx, Ny, Nz = 16, 2, length(zf) - 1
        g = StretchedGrid(; size = (Nx, Ny, Nz), extent = (1000.0, 125.0, 800.0), halo = (2, 2, 2),
                          stretching = (:custom, zf))
        S, a, b, c, Γ, k = 0.02, 0.3, 0.2, -0.5, 0.01, 2π / 1000.0
        νt0, Pr, θ₀, gr = 1.7, 0.7, 300.0, 9.81
        fill_analytic!(f, fn, xs, zs) =
            (for K in axes(f.data, 3), J in axes(f.data, 2), I in axes(f.data, 1)
                 f.data[I, J, K] = fn(xs(I), zs(K))
             end; f)
        xf(I) = _xface(g, I); xc(I) = _xcent(g, I); zc(K) = g.zc[K]; zw(K) = g.zf[K]
        u = fill_analytic!(Field(FaceX(), g), (x, z) -> S * z + a * sin(k * x), xf, zc)
        v = Field(FaceY(), g)
        w = fill_analytic!(Field(FaceZ(), g), (x, z) -> b * sin(k * x), xc, zw)
        θ = fill_analytic!(Field(CellCenter(), g), (x, z) -> θ₀ + Γ * z + c * sin(k * x), xc, zc)
        νt = Field(CellCenter(), g); fill!(νt.data, νt0)
        r = flux_richardson(u, v, w, θ; θ₀ = θ₀, g = gr, νt = νt, Pr_t = Pr)
        uw = a * b * cos(k * g.dx / 2) / 2 - νt0 * S
        @test agree(r.P, fill(-uw * S, Nz))
        @test agree(r.B, fill(gr / θ₀ * (b * c / 2 - νt0 / Pr * Γ), Nz))
        # a pure linear shear has no vorticity fluctuation: γ = 0 inside
        w.data .= 0
        @test all(iszero, turbulent_fraction(u, v, w)[2:Nz-1])
    end
end
