using Test
using Random
using Aeolus
using Aeolus.Grids: RegularGrid, interior, totalsize, CellCenter, FaceX, FaceY, FaceZ
using Aeolus.Fields: Field, set!, interior_view
using Aeolus.CutCell

# =========================================================================
# Era II, Phases 21-22 — cut-cell IBM with ghost-cell wall reconstruction
# =========================================================================

# cell-center z of a storage index (origin at 0)
_zc(g, k) = (k - g.Hz - 0.5) * g.dz

"""
Ghost-cell reconstruction on a flat wall at zw = (N/4 + 0.7)·dz (always
0.2·dz above the cut cell's center, self-similar under refinement) of the
field `wall + (z-zw)^power`. Returns the max error over the ghost cells.
"""
function ghost_case(N::Int, power::Int, wall::Float64)
    g  = RegularGrid(; size = (4, 4, N), extent = (1.0, 1.0, 1.0), halo = (2, 2, 2))
    zw = (N ÷ 4 + 0.7) * g.dz
    geom = cut_cell_geometry(g, (x, y, z) -> z - zw)
    st   = build_ghost_stencils(geom, g)
    f    = Field(CellCenter(), g)
    for k in axes(f.data, 3), j in axes(f.data, 2), i in axes(f.data, 1)
        f.data[i, j, k] = wall + (_zc(g, k) - zw)^power
    end
    reconstruct_wall_scalar!(f, st, geom; wall_value = wall)
    err = maximum(abs(f.data[s.i, s.j, s.k] - (wall + (_zc(g, s.k) - zw)^power))
                  for s in st if s.ghost)
    return (err = err, nghost = count(s -> s.ghost, st), dz = g.dz, zw = zw,
            geom = geom, stencils = st, grid = g)
end

@testset "Era II — Cut-cell IBM" begin

    g = RegularGrid(; size = (4, 4, 8), extent = (1.0, 1.0, 1.0), halo = (2, 2, 2))
    ii, jj, kk = interior(g)

    @testset "Geometry — cell cut exactly in half" begin
        zw = 0.5 + g.dz / 2                          # mid-plane of interior cell 5
        kc = kk[5]
        geom = cut_cell_geometry(g, (x, y, z) -> z - zw)
        @test all(isapprox.(geom.vol_frac[ii, jj, kc], 0.5; atol = 1e-12))
        @test all(isapprox.(geom.ax[ii, jj, kc], 0.5; atol = 1e-12))
        @test all(geom.az[ii, jj, kc] .== 0) && all(geom.az[ii, jj, kc+1] .== 1)
        # wetted area from the aperture balance is exact for a plane
        @test all(isapprox.(geom.area[ii, jj, kc], g.dx * g.dy; atol = 1e-14))
        @test cut_cell_count(geom) == (16, 64, 0)
        @test fluid_volume(geom) ≈ 1 - zw
        # tilted plane φ = z - 0.3 - 0.2x  =>  n ∝ (-0.2, 0, 1)
        gt = cut_cell_geometry(g, (x, y, z) -> z - 0.3 - 0.2x)
        cut = gt.is_cut[ii, jj, kk]
        @test any(cut)
        @test maximum(abs.(gt.nx[ii, jj, kk][cut] .+ 0.2 / sqrt(1.04))) < 1e-12
    end

    @testset "Divergence–gradient adjointness, L = D·G" begin
        g2 = RegularGrid(; size = (8, 8, 8), extent = (1.0, 1.0, 1.0), halo = (2, 2, 2))
        Random.seed!(20260921)
        # fluid box closed on all six sides: no boundary flux to zero
        lsB = (x, y, z) -> min(x - 0.13, 0.86 - x, y - 0.11, 0.88 - y,
                               z - 0.14, 0.85 - z)
        gB = cut_cell_geometry(g2, lsB)
        @test cut_cell_count(gB)[1] > 0
        pB = Field(CellCenter(), g2)
        uB = Field(FaceX(), g2); vB = Field(FaceY(), g2); wB = Field(FaceZ(), g2)
        for A in (pB, uB, vB, wB)
            A.data .= randn.()
        end
        div = Field(CellCenter(), g2)
        gx  = Field(FaceX(), g2); gy = Field(FaceY(), g2); gz = Field(FaceZ(), g2)
        cutcell_divergence!(div, uB, vB, wB, gB, g2)
        cutcell_gradient_x!(gx, pB, gB, g2)
        cutcell_gradient_y!(gy, pB, gB, g2)
        cutcell_gradient_z!(gz, pB, gB, g2)
        lhs = cutcell_cell_inner(pB, div, gB, g2)
        rhs = cutcell_face_inner(gx, gy, gz, uB, vB, wB, gB, g2)
        @test abs(lhs + rhs) / max(abs(lhs), abs(rhs)) < 1e-10

        lap = Field(CellCenter(), g2); dgp = Field(CellCenter(), g2)
        cutcell_laplacian!(lap, pB, gB, g2)
        cutcell_divergence!(dgp, gx, gy, gz, gB, g2)
        @test maximum(abs, lap.data .- dgp.data) < 1e-10 * maximum(abs, lap.data)
        @test cutcell_cell_inner(pB, lap, gB, g2) < 0          # negative semi-definite
    end

    @testset "Ghost-cell reconstruction — exact for linear, 2nd order" begin
        r = ghost_case(16, 1, 3.0)
        @test r.nghost == 16                      # one ghost layer, 4×4 columns
        @test r.err < 1e-12
        # Neumann: κ ∂f/∂n = α, n = (0,0,1)
        gg = r.grid
        f = Field(CellCenter(), gg)
        for k in axes(f.data, 3), j in axes(f.data, 2), i in axes(f.data, 1)
            f.data[i, j, k] = 2.0 + 3.0 * (_zc(gg, k) - r.zw)
        end
        reconstruct_wall_scalar!(f, r.stencils, r.geom; wall_flux = 1.5, κ = 0.5)
        @test maximum(abs(f.data[s.i, s.j, s.k] - (2.0 + 3.0 * (_zc(gg, s.k) - r.zw)))
                      for s in r.stencils) < 1e-12
        e16, e32 = ghost_case(16, 2, 0.0).err, ghost_case(32, 2, 0.0).err
        @test 3.2 < e16 / e32 < 4.8              # ~4 ⇒ second order
    end

    @testset "Wall shear stress on the true normal" begin
        r  = ghost_case(16, 1, 0.0)
        gg = r.grid
        ν, α = 1.5e-2, 2.5                        # u = α (z - zw)  ⇒  τ_w = ν α
        u = Field(FaceX(), gg); v = Field(FaceY(), gg); w = Field(FaceZ(), gg)
        for k in axes(u.data, 3), j in axes(u.data, 2), i in axes(u.data, 1)
            u.data[i, j, k] = α * (_zc(gg, k) - r.zw)
        end
        τ = wall_shear_stress(u, v, w, r.stencils, r.geom, ν)
        @test all(t -> isapprox(t, ν * α; rtol = 1e-10), τ)
    end

    @testset "No-penetration on blocked faces; SolidMask interop" begin
        g2 = RegularGrid(; size = (8, 8, 8), extent = (1.0, 1.0, 1.0), halo = (2, 2, 2))
        ls = (x, y, z) -> min(z - 0.18 - 0.12x,
                              sqrt((x - 0.5)^2 + (y - 0.45)^2 + (z - 0.6)^2) - 0.22)
        geom = cut_cell_geometry(g2, ls)
        Random.seed!(7)
        u = Field(FaceX(), g2); v = Field(FaceY(), g2); w = Field(FaceZ(), g2)
        u.data .= randn.(); v.data .= randn.(); w.data .= randn.()
        u0 = copy(u.data)
        apply_cutcell_velocity_bc!(u, v, w, geom, g2)
        @test all(u.data[geom.ax .== 0] .== 0.0) && all(w.data[geom.az .== 0] .== 0.0)
        open = geom.ax .> 0
        @test u.data[open] == u0[open]            # open faces untouched (copied data)
        sm = solid_mask(geom)
        @test sm isa SolidMask && sm.mask == geom.is_solid
    end

    @testset "Small-cell merging conserves the divergence" begin
        gs = RegularGrid(; size = (4, 4, 4), extent = (1.0, 1.0, 1.0), halo = (2, 2, 2))
        geom = cut_cell_geometry(gs, (x, y, z) -> z - 0.26; nsub = 4)
        i0, j0, k0 = interior(gs)
        i, j, k = i0[2], j0[2], k0[2]             # turn this cell into a sliver
        geom.vol_frac[i, j, k] = 1e-4
        geom.vol_eff[i, j, k]  = geom.vol_min
        geom.is_small[i, j, k] = true
        geom.is_cut[i, j, k]   = true
        geom.merge_target[i, j, k] = LinearIndices(size(geom.vol_frac))[i, j, k+1]
        u = Field(FaceX(), gs); v = Field(FaceY(), gs); w = Field(FaceZ(), gs)
        Random.seed!(3)
        u.data .= randn.(); v.data .= randn.(); w.data .= randn.()
        div = Field(CellCenter(), gs)
        msum() = sum(geom.vol_eff[i2, j2, k2] * div.data[i2, j2, k2]
                     for k2 in k0, j2 in j0, i2 in i0)
        cutcell_divergence!(div, u, v, w, geom, gs; redistribute = false)
        m_before = msum()
        cutcell_divergence!(div, u, v, w, geom, gs; redistribute = true)
        @test isapprox(msum(), m_before; rtol = 1e-12, atol = 1e-14)
        @test div.data[i, j, k] ≈ div.data[i, j, k+1]   # the cluster shares one value
    end
end
