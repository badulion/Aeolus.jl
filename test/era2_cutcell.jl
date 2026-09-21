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
Ghost-cell reconstruction on a flat wall at zw = (N/4 + 0.7)·dz, i.e. a
wall that always sits 0.2·dz above the cut cell's center, so the geometry
is self-similar under refinement. The field is `wall + (z-zw)^power`.
Returns the max reconstruction error over the ghost cells.
"""
function ghost_case(N::Int, power::Int, wall::Float64)
    g  = RegularGrid(; size = (4, 4, N), extent = (1.0, 1.0, 1.0), halo = (2, 2, 2))
    dz = g.dz
    zw = (N ÷ 4 + 0.7) * dz
    geom = cut_cell_geometry(g, (x, y, z) -> z - zw)
    st   = build_ghost_stencils(geom, g)
    f    = Field(CellCenter(), g)
    for k in axes(f.data, 3), j in axes(f.data, 2), i in axes(f.data, 1)
        f.data[i, j, k] = wall + (_zc(g, k) - zw)^power
    end
    reconstruct_wall_scalar!(f, st, geom; wall_value = wall)
    err = 0.0
    nghost = 0
    for s in st
        s.ghost || continue
        nghost += 1
        exact = wall + (_zc(g, s.k) - zw)^power
        err = max(err, abs(f.data[s.i, s.j, s.k] - exact))
    end
    return (err = err, nghost = nghost, dz = dz, zw = zw,
            geom = geom, stencils = st, grid = g, field = f)
end

@testset "Era II — Cut-cell IBM" begin

    g = RegularGrid(; size = (4, 4, 8), extent = (1.0, 1.0, 1.0), halo = (2, 2, 2))
    ii, jj, kk = interior(g)

    # ---------------------------------------------------------------------
    @testset "Geometry — all-fluid level set" begin
        geom = cut_cell_geometry(g, (x, y, z) -> 1.0)
        @test all(view(geom.vol_frac, ii, jj, kk) .== 1)
        @test all(view(geom.vol_eff,  ii, jj, kk) .== 1)
        @test all(view(geom.ax, ii, jj, kk) .== 1)
        @test all(view(geom.ay, ii, jj, kk) .== 1)
        @test all(view(geom.az, ii, jj, kk) .== 1)
        @test !any(view(geom.is_cut,   ii, jj, kk))
        @test !any(view(geom.is_solid, ii, jj, kk))
        @test !any(view(geom.is_small, ii, jj, kk))
        @test fluid_volume(geom) ≈ 1.0
        @test wetted_area(geom) == 0
        @test all(view(geom.nx, ii, jj, kk) .== 0)   # no normal where uncut
        @test all(view(geom.nz, ii, jj, kk) .== 0)
    end

    @testset "Geometry — wall on a cell boundary" begin
        # z = 0.5 is exactly the top face of cell 4 (dz = 0.125)
        geom = cut_cell_geometry(g, (x, y, z) -> z - 0.5)
        vf = view(geom.vol_frac, ii, jj, kk)
        @test all(v -> v == 0 || v == 1, vf)
        @test count(==(1.0), vf) == 4 * 4 * 4       # top half wetted
        @test count(==(0.0), vf) == 4 * 4 * 4       # bottom half dry
        @test !any(view(geom.is_cut, ii, jj, kk))   # nothing is cut
        @test all(geom.az[ii, jj, kk[5]] .== 0)     # the wall face carries no flux
        @test all(geom.az[ii, jj, kk[6]] .== 1)     # next face up is fully open
        @test all(geom.ax[ii, jj, kk[5]] .== 1)     # vertical faces above the wall
        @test all(geom.ax[ii, jj, kk[4]] .== 0)     # ... and below it (dry cells)
        @test fluid_volume(geom) ≈ 0.5
    end

    @testset "Geometry — cell cut exactly in half" begin
        zw = 0.5 + g.dz / 2                          # mid-plane of interior cell 5
        kc = kk[5]
        for nsub in (4, 8)
            geom = cut_cell_geometry(g, (x, y, z) -> z - zw; nsub = nsub)
            @test all(isapprox.(geom.vol_frac[ii, jj, kc], 0.5; atol = 1e-12))
            @test all(geom.is_cut[ii, jj, kc])
            @test all(isapprox.(geom.ax[ii, jj, kc], 0.5; atol = 1e-12))
            @test all(isapprox.(geom.ay[ii, jj, kc], 0.5; atol = 1e-12))
            @test all(geom.az[ii, jj, kc] .== 0)     # bottom face fully blocked
            @test all(geom.az[ii, jj, kc+1] .== 1)   # top face fully open
            # wetted area from the aperture balance is exact for a plane
            @test all(isapprox.(geom.area[ii, jj, kc], g.dx * g.dy; atol = 1e-14))
            @test !any(geom.is_small)
            cutc, solidc, smallc = cut_cell_count(geom)
            @test cutc == 4 * 4                      # exactly one cut layer
            @test solidc == 4 * 4 * 4
            @test smallc == 0
        end
    end

    @testset "Geometry — normals" begin
        zw = 0.5 + g.dz / 2
        kc = kk[5]
        geom = cut_cell_geometry(g, (x, y, z) -> z - zw)
        @test all(isapprox.(geom.nz[ii, jj, kc], 1.0; atol = 1e-12))
        @test all(abs.(geom.nx[ii, jj, kc]) .< 1e-12)
        @test all(abs.(geom.ny[ii, jj, kc]) .< 1e-12)

        # a non-normalized level set must give the same unit normal
        geom5 = cut_cell_geometry(g, (x, y, z) -> 5 * (z - zw))
        @test all(isapprox.(geom5.nz[ii, jj, kc], 1.0; atol = 1e-12))

        # tilted plane φ = z - 0.3 - 0.2x  =>  n ∝ (-0.2, 0, 1)
        geomt = cut_cell_geometry(g, (x, y, z) -> z - 0.3 - 0.2x)
        nrm = sqrt(1 + 0.2^2)
        ncut = 0
        for k in kk, j in jj, i in ii
            geomt.is_cut[i, j, k] || continue
            ncut += 1
            @test isapprox(geomt.nx[i, j, k], -0.2 / nrm; atol = 1e-12)
            @test isapprox(geomt.ny[i, j, k],  0.0;       atol = 1e-12)
            @test isapprox(geomt.nz[i, j, k],  1.0 / nrm; atol = 1e-12)
        end
        @test ncut > 0
    end

    @testset "Geometry — heightmap constructor" begin
        h = fill(0.3, 4, 4)
        gmap = cut_cell_geometry_from_heightmap(g, h)
        gref = cut_cell_geometry(g, (x, y, z) -> z - 0.3)
        @test gmap.vol_frac == gref.vol_frac
        @test gmap.az == gref.az
        # a flat-roofed building in one column
        h2 = fill(0.0, 4, 4); h2[2, 3] = 0.5
        gb = cut_cell_geometry_from_heightmap(g, h2)
        @test all(gb.is_solid[ii[2], jj[3], kk[1]:kk[4]])   # column blocked to z=0.5
        @test !any(gb.is_solid[ii[2], jj[3], kk[5]:kk[8]])
        @test !any(gb.is_solid[ii[1], jj[1], kk])           # neighbouring column clear
    end

    # ---------------------------------------------------------------------
    @testset "Divergence of a uniform flow past a flat wall" begin
        geom = cut_cell_geometry(g, (x, y, z) -> z - 0.3)
        @test any(view(geom.is_cut, ii, jj, kk))     # the wall really cuts cells
        u = Field(FaceX(), g); v = Field(FaceY(), g); w = Field(FaceZ(), g)
        u.data .= 1.3
        v.data .= -0.7
        w.data .= 0.0
        div = Field(CellCenter(), g)
        cutcell_divergence!(div, u, v, w, geom, g)
        @test maximum(abs, view(div.data, ii, jj, kk)) < 1e-12
        # solid cells carry no divergence at all
        for k in kk, j in jj, i in ii
            geom.is_solid[i, j, k] && @test div.data[i, j, k] == 0
        end
    end

    # ---------------------------------------------------------------------
    @testset "Divergence–gradient adjointness" begin
        g2 = RegularGrid(; size = (8, 8, 8), extent = (1.0, 1.0, 1.0), halo = (2, 2, 2))
        i2, j2, k2 = interior(g2)
        Random.seed!(20260921)

        # --- geometry A: sloping ground + an embedded sphere --------------
        lsA = (x, y, z) -> min(z - 0.18 - 0.12x,
                               sqrt((x - 0.5)^2 + (y - 0.45)^2 + (z - 0.6)^2) - 0.22)
        gA = cut_cell_geometry(g2, lsA)
        cutA, solidA, smallA = cut_cell_count(gA)
        @test cutA > 0 && solidA > 0
        @test smallA == 0        # sampled fractions are quantized in 1/nsub³

        p = Field(CellCenter(), g2)
        u = Field(FaceX(), g2); v = Field(FaceY(), g2); w = Field(FaceZ(), g2)
        for k in k2, j in j2, i in i2
            p.data[i, j, k] = randn()
        end
        # random velocity on the interior faces only: the domain-boundary
        # faces must carry no flux for the telescoping identity to close
        for k in k2, j in j2, i in (first(i2)+1):last(i2)
            u.data[i, j, k] = randn()
        end
        for k in k2, j in (first(j2)+1):last(j2), i in i2
            v.data[i, j, k] = randn()
        end
        for k in (first(k2)+1):last(k2), j in j2, i in i2
            w.data[i, j, k] = randn()
        end

        div = Field(CellCenter(), g2)
        gx  = Field(FaceX(), g2); gy = Field(FaceY(), g2); gz = Field(FaceZ(), g2)
        cutcell_divergence!(div, u, v, w, gA, g2)
        cutcell_gradient_x!(gx, p, gA, g2)
        cutcell_gradient_y!(gy, p, gA, g2)
        cutcell_gradient_z!(gz, p, gA, g2)

        lhs = cutcell_cell_inner(p, div, gA, g2)
        rhs = cutcell_face_inner(gx, gy, gz, u, v, w, gA, g2)
        @test abs(lhs) > 1e-8                                  # non-trivial
        relA = abs(lhs + rhs) / max(abs(lhs), abs(rhs))
        @info "cut-cell adjointness (sphere + slope)" lhs rhs residual=abs(lhs+rhs) relative=relA
        @test relA < 1e-10

        # --- geometry B: fluid box closed on all six sides ----------------
        lsB = (x, y, z) -> min(x - 0.13, 0.86 - x, y - 0.11, 0.88 - y,
                               z - 0.14, 0.85 - z)
        gB = cut_cell_geometry(g2, lsB)
        @test cut_cell_count(gB)[1] > 0
        # every domain-boundary face is blocked, so nothing needs zeroing
        pB = Field(CellCenter(), g2)
        uB = Field(FaceX(), g2); vB = Field(FaceY(), g2); wB = Field(FaceZ(), g2)
        for A in (pB, uB, vB, wB)
            A.data .= randn.()
        end
        cutcell_divergence!(div, uB, vB, wB, gB, g2)
        cutcell_gradient_x!(gx, pB, gB, g2)
        cutcell_gradient_y!(gy, pB, gB, g2)
        cutcell_gradient_z!(gz, pB, gB, g2)
        lhsB = cutcell_cell_inner(pB, div, gB, g2)
        rhsB = cutcell_face_inner(gx, gy, gz, uB, vB, wB, gB, g2)
        relB = abs(lhsB + rhsB) / max(abs(lhsB), abs(rhsB))
        @info "cut-cell adjointness (closed box)" lhsB rhsB residual=abs(lhsB+rhsB) relative=relB
        @test relB < 1e-10

        # --- L == D·G, and L self-adjoint in the wetted inner product ----
        lap  = Field(CellCenter(), g2)
        dgp  = Field(CellCenter(), g2)
        cutcell_laplacian!(lap, pB, gB, g2)          # default: exactly D·G
        cutcell_divergence!(dgp, gx, gy, gz, gB, g2)
        scale = maximum(abs, view(lap.data, i2, j2, k2))
        errDG = maximum(abs, view(lap.data, i2, j2, k2) .- view(dgp.data, i2, j2, k2))
        @info "cut-cell L == D·G" scale absolute=errDG relative=errDG/scale
        @test errDG < 1e-10 * scale

        qB = Field(CellCenter(), g2)
        qB.data .= randn.()
        lapq = Field(CellCenter(), g2)
        cutcell_laplacian!(lapq, qB, gB, g2)
        s1 = cutcell_cell_inner(qB, lap,  gB, g2)
        s2 = cutcell_cell_inner(pB, lapq, gB, g2)
        relS = abs(s1 - s2) / max(abs(s1), abs(s2))
        @info "cut-cell Laplacian symmetry" s1 s2 relative=relS
        @test relS < 1e-10
        # negative semi-definite
        @test cutcell_cell_inner(pB, lap, gB, g2) < 0

        # the flux-form (single-aperture) variant is symmetric too
        lap1  = Field(CellCenter(), g2)
        lapq1 = Field(CellCenter(), g2)
        cutcell_laplacian!(lap1,  pB, gB, g2; aperture_power = 1)
        cutcell_laplacian!(lapq1, qB, gB, g2; aperture_power = 1)
        t1 = cutcell_cell_inner(qB, lap1,  gB, g2)
        t2 = cutcell_cell_inner(pB, lapq1, gB, g2)
        @test abs(t1 - t2) / max(abs(t1), abs(t2)) < 1e-10
        @test cutcell_cell_inner(pB, lap1, gB, g2) < 0
        @test_throws ArgumentError cutcell_laplacian!(lap1, pB, gB, g2; aperture_power = 3)
    end

    # ---------------------------------------------------------------------
    @testset "Ghost-cell reconstruction — exactness for a linear profile" begin
        r = ghost_case(16, 1, 3.0)                # wall value 3, f = 3 + (z-zw)
        @test r.nghost == 16                      # one ghost layer, 4×4 columns
        @test length(r.stencils) == r.nghost
        @test r.err < 1e-12
        @info "ghost-cell linear reconstruction" max_error=r.err nghost=r.nghost

        # geometric bookkeeping of the stencils
        for s in r.stencils
            @test s.ghost
            @test s.valid
            @test s.exact_linear                      # all donors are fluid
            @test isapprox(s.dn, -0.2 * r.dz; atol = 1e-12)
            @test isapprox(s.probe, r.dz; atol = 1e-12)   # floored at one Δz
            @test isapprox(s.n[3], 1.0; atol = 1e-12)
            @test isapprox(s.xw[3], r.zw; atol = 1e-12)   # foot point on the true wall
            @test isapprox(s.ximg[3], r.zw + r.dz; atol = 1e-12)
            @test isapprox(sum(s.weights), 1.0; atol = 1e-12)
        end

        # zero wall value (no-slip) through the velocity entry point
        gg = r.grid
        f = Field(CellCenter(), gg)
        for k in axes(f.data, 3), j in axes(f.data, 2), i in axes(f.data, 1)
            f.data[i, j, k] = _zc(gg, k) - r.zw
        end
        reconstruct_wall_velocity!(f, r.stencils, r.geom)
        for s in r.stencils
            @test isapprox(f.data[s.i, s.j, s.k], _zc(gg, s.k) - r.zw; atol = 1e-12)
        end

        # position-dependent wall value
        fv = Field(CellCenter(), gg)
        for k in axes(fv.data, 3), j in axes(fv.data, 2), i in axes(fv.data, 1)
            fv.data[i, j, k] = 2 * _zc(gg, k)
        end
        reconstruct_wall_velocity!(fv, r.stencils, r.geom;
                                   wall_value = (x, y, z) -> 2 * z)
        for s in r.stencils
            @test isapprox(fv.data[s.i, s.j, s.k], 2 * _zc(gg, s.k); atol = 1e-12)
        end
    end

    @testset "Ghost-cell reconstruction — Neumann (flux) condition" begin
        r  = ghost_case(16, 1, 2.0)   # geometry only; rebuild the field below
        gg = r.grid
        α  = 3.0
        f  = Field(CellCenter(), gg)
        for k in axes(f.data, 3), j in axes(f.data, 2), i in axes(f.data, 1)
            f.data[i, j, k] = 2.0 + α * (_zc(gg, k) - r.zw)
        end
        # κ ∂f/∂n = α with κ = 1, n = (0,0,1)
        reconstruct_wall_scalar!(f, r.stencils, r.geom; wall_flux = α, κ = 1)
        for s in r.stencils
            @test isapprox(f.data[s.i, s.j, s.k], 2.0 + α * (_zc(gg, s.k) - r.zw);
                           atol = 1e-12)
        end
        # same condition written as a flux with a diffusivity
        f2 = Field(CellCenter(), gg)
        for k in axes(f2.data, 3), j in axes(f2.data, 2), i in axes(f2.data, 1)
            f2.data[i, j, k] = α * (_zc(gg, k) - r.zw)
        end
        reconstruct_wall_scalar!(f2, r.stencils, r.geom; wall_flux = 0.5 * α, κ = 0.5)
        for s in r.stencils
            @test isapprox(f2.data[s.i, s.j, s.k], α * (_zc(gg, s.k) - r.zw); atol = 1e-12)
        end
        @test_throws ArgumentError reconstruct_wall_scalar!(f2, r.stencils, r.geom)
        @test_throws ArgumentError reconstruct_wall_scalar!(f2, r.stencils, r.geom;
                                                           wall_value = 0.0, wall_flux = 1.0)
    end

    @testset "Ghost-cell reconstruction — second-order convergence" begin
        rs = [ghost_case(N, 2, 0.0) for N in (16, 32, 64)]
        errs = [r.err for r in rs]
        ratios = [errs[i] / errs[i+1] for i in 1:length(errs)-1]
        @info "ghost-cell quadratic-profile convergence" errors=errs ratios=ratios
        @test all(e -> e > 0, errs)
        for ρ in ratios
            @test 3.2 < ρ < 4.8          # ~4 ⇒ second order
        end
    end

    @testset "Wall shear stress on the true normal" begin
        r  = ghost_case(32, 1, 0.0)
        gg = r.grid
        ν  = 1.5e-2
        α  = 2.5                          # u = α (z - zw)  ⇒  τ_w = ν α
        u = Field(FaceX(), gg); v = Field(FaceY(), gg); w = Field(FaceZ(), gg)
        for k in axes(u.data, 3), j in axes(u.data, 2), i in axes(u.data, 1)
            u.data[i, j, k] = α * (_zc(gg, k) - r.zw)
        end
        τ = wall_shear_stress(u, v, w, r.stencils, r.geom, ν)
        @test length(τ) == length(r.stencils)
        @info "wall shear stress" analytic=ν*α measured=maximum(τ) spread=(maximum(τ)-minimum(τ))
        @test all(t -> isapprox(t, ν * α; rtol = 1e-10), τ)
    end

    # ---------------------------------------------------------------------
    @testset "No-penetration on blocked faces" begin
        g2 = RegularGrid(; size = (8, 8, 8), extent = (1.0, 1.0, 1.0), halo = (2, 2, 2))
        ls = (x, y, z) -> min(z - 0.18 - 0.12x,
                              sqrt((x - 0.5)^2 + (y - 0.45)^2 + (z - 0.6)^2) - 0.22)
        geom = cut_cell_geometry(g2, ls)
        Random.seed!(7)
        u = Field(FaceX(), g2); v = Field(FaceY(), g2); w = Field(FaceZ(), g2)
        u.data .= randn.(); v.data .= randn.(); w.data .= randn.()
        u0 = copy(u.data)
        @test any(geom.ax .== 0) && any(geom.az .== 0)

        apply_cutcell_velocity_bc!(u, v, w, geom, g2)

        @test all(u.data[geom.ax .== 0] .== 0.0)      # exactly zero
        @test all(v.data[geom.ay .== 0] .== 0.0)
        @test all(w.data[geom.az .== 0] .== 0.0)
        # open faces are untouched
        open = geom.ax .> 0
        @test u.data[open] == u0[open]
        # no flux may enter a dry cell
        for k in axes(geom.ax, 3), j in axes(geom.ax, 2), i in 2:size(geom.ax, 1)
            if geom.is_solid[i-1, j, k] || geom.is_solid[i, j, k]
                @test u.data[i, j, k] == 0.0
            end
        end

        # a uniform field made divergence-free by the geometry alone
        u.data .= 0.0; v.data .= 0.0; w.data .= 0.0
        div = Field(CellCenter(), g2)
        cutcell_divergence!(div, u, v, w, geom, g2)
        @test maximum(abs, div.data) == 0.0
    end

    @testset "Interop — SolidMask and BoussinesqSimulation" begin
        g2 = RegularGrid(; size = (8, 8, 8), extent = (1.0, 1.0, 1.0), halo = (2, 2, 2))
        geom = cut_cell_geometry(g2, (x, y, z) -> z - 0.26)
        sm = solid_mask(geom)
        @test sm isa SolidMask
        @test sm.mask == geom.is_solid

        # staircase geometry rebuilt from the mask
        gs = cut_cell_geometry(g2, sm)
        @test all(v -> v == 0 || v == 1, gs.vol_frac)
        @test gs.is_solid == geom.is_solid
        @test !any(gs.is_cut)

        sim = BoussinesqSimulation(g2)
        sim.u.data .= 1.0; sim.v.data .= 1.0; sim.w.data .= 1.0
        apply_cutcell_velocity_bc!(sim, geom)
        @test all(sim.u.data[geom.ax .== 0] .== 0.0)
        @test all(sim.w.data[geom.az .== 0] .== 0.0)
        @test all(sim.u.data[geom.ax .> 0] .== 1.0)

        # the same, used as a post-predictor hook of the real driver. The
        # legacy FFT projection still uses Cartesian gradients, so the
        # cut-cell BC has to be re-applied after the projection until the
        # cut-cell Poisson operator is wired into a solver.
        step!(sim, 1e-3; post_predictor! = (s, Δt) -> apply_cutcell_velocity_bc!(s, geom))
        @test isfinite(maximum(abs, sim.u.data))
        apply_cutcell_velocity_bc!(sim, geom)
        @test all(sim.u.data[geom.ax .== 0] .== 0.0)
        @test all(sim.w.data[geom.az .== 0] .== 0.0)

        # face-located ghost stencils exist for every velocity component
        for loc in (FaceX(), FaceY(), FaceZ())
            st = build_ghost_stencils(geom, g2, loc)
            @test !isempty(st)
            @test all(s -> s.valid, st)
        end
        su = build_ghost_stencils(geom, g2, FaceX())
        reconstruct_wall_velocity!(sim.u, su, geom)     # no-slip on the true wall
        @test isfinite(maximum(abs, sim.u.data))
    end

    # ---------------------------------------------------------------------
    @testset "Small-cell handling" begin
        # A hand-built sliver: one cell with a 1e-4 wetted fraction.
        gs = RegularGrid(; size = (4, 4, 4), extent = (1.0, 1.0, 1.0), halo = (2, 2, 2))
        geom = cut_cell_geometry(gs, (x, y, z) -> z - 0.26; nsub = 4)
        i0, j0, k0 = interior(gs)
        # turn interior cell (2,2,2) into a sliver
        i, j, k = i0[2], j0[2], k0[2]
        geom.vol_frac[i, j, k] = 1e-4
        geom.vol_eff[i, j, k]  = geom.vol_min
        geom.is_small[i, j, k] = true
        geom.is_cut[i, j, k]   = true
        geom.merge_target[i, j, k] = LinearIndices(size(geom.vol_frac))[i, j, k+1]
        @test geom.vol_eff[i, j, k] == geom.vol_min   # volume is floored

        u = Field(FaceX(), gs); v = Field(FaceY(), gs); w = Field(FaceZ(), gs)
        Random.seed!(3)
        u.data .= randn.(); v.data .= randn.(); w.data .= randn.()
        div = Field(CellCenter(), gs)
        V = gs.dx * gs.dy * gs.dz

        cutcell_divergence!(div, u, v, w, geom, gs; redistribute = false)
        m_before = sum(geom.vol_eff[i2, j2, k2] * div.data[i2, j2, k2]
                       for k2 in k0, j2 in j0, i2 in i0) * V
        d_small = div.data[i, j, k]

        cutcell_divergence!(div, u, v, w, geom, gs; redistribute = true)
        m_after = sum(geom.vol_eff[i2, j2, k2] * div.data[i2, j2, k2]
                      for k2 in k0, j2 in j0, i2 in i0) * V
        # merging conserves the volume-weighted divergence exactly ...
        @test isapprox(m_after, m_before; rtol = 1e-12, atol = 1e-14)
        # ... and the cluster now shares one value
        @test div.data[i, j, k] == div.data[i, j, k+1]
        @test abs(div.data[i, j, k]) <= abs(d_small) + 1e-12
    end

end
