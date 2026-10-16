# =========================================================================
# Era II / Phase 24 — stretched & terrain-following vertical coordinates
# =========================================================================
# Validation strategy, in order of increasing strength:
#
#   1. *Reduction*   — with `stretching=:uniform` every new kernel must
#      reproduce the `RegularGrid` kernel. Catches sign errors,
#      swapped metrics and off-by-one halo indexing in one shot.
#   2. *Dispatch*    — prove the public `Operators.∂z` façade actually
#      lands in the StretchedGrids methods (the whole retrofit rests on it).
#   3. *Exactness*   — the stencils annihilate the polynomial degrees they
#      claim to reproduce on a genuinely non-uniform grid, and the naive
#      alternatives measurably do not.
#   4. *Convergence* — 2nd order under refinement for a smooth profile.
#   5. *Physics*     — solenoidal fields have zero discrete divergence;
#      a hydrostatic atmosphere over a hill stays hydrostatic.

using Test
using Aeolus
using Aeolus.StretchedGrids
using Aeolus.Grids: interior, znodes, totalsize
using Aeolus.Fields: Field, set!, interior_view, mesh
using Aeolus.Operators: ∂x, ∂y, ∂z, laplacian, divergence,
                        interpolate_to_face!, interpolate_to_center!

const FV = Aeolus.Operators.FiniteVolume

# ---------------------------------------------------------------- helpers

"Max |a-b| over the interior of two fields."
function maxdiff(a::Field, b::Field)
    return maximum(abs.(interior_view(a) .- interior_view(b)))
end

"Max |field - fun(x,y,z)| over the interior, at the field's own location."
function maxerr(f::Field, fun)
    g   = mesh(f)
    ref = Field(Aeolus.Fields.location(f), g)
    set_physical!(ref, fun)
    return maxdiff(f, ref)
end

@testset "Era II — Stretched & terrain-following grids" begin

    SZ    = (6, 5, 24)
    EXT   = (600.0, 500.0, 1000.0)
    HALO  = (2, 2, 2)

    # =====================================================================
    @testset "grid construction & metrics" begin
        u = StretchedGrid(Float64; size = SZ, extent = EXT, halo = HALO,
                          stretching = :uniform)
        @test u isa StretchedGrid{Float64}
        @test u isa Aeolus.Grids.AbstractMesh{Float64}
        @test totalsize(u) == (SZ[1] + 4, SZ[2] + 4, SZ[3] + 4)
        @test interior(u) == (3:8, 3:7, 3:26)
        @test length(zfaces(u))         == SZ[3] + 2HALO[3] + 1
        @test length(zcenters(u))       == SZ[3] + 2HALO[3]
        @test length(cell_thickness(u)) == SZ[3] + 2HALO[3]
        @test length(center_spacing(u)) == SZ[3] + 2HALO[3] + 1

        # halo levels populated (no zeros / NaNs leaking in)
        @test all(isfinite, zfaces(u))
        @test all(>(0), cell_thickness(u))
        @test all(>(0), center_spacing(u))

        # interior faces span exactly [z0, z0+Lz]
        kf = HALO[3] + 1
        @test zfaces(u)[kf]           ≈ 0.0 atol = 0
        @test zfaces(u)[kf + SZ[3]]   ≈ EXT[3] atol = 1e-12

        # znodes match RegularGrid for the uniform case. Not the same:
        # `RegularGrid` builds its node vectors as `StepRangeLen`s, whose
        # TwicePrecision arithmetic is *more* accurate than the stored
        # `zc`/`zf` vectors' single multiply. The residual is one ulp of the
        # coordinate (~1e-16 relative), not a metric discrepancy — the
        # operator-level reduction below is the load-bearing check.
        rg = RegularGrid(Float64; size = SZ, extent = EXT, halo = HALO)
        @test maximum(abs.(znodes(u, CellCenter()) .- znodes(rg, CellCenter()))) < 1e-12
        @test maximum(abs.(znodes(u, FaceZ())      .- znodes(rg, FaceZ())))      < 1e-12
        @test grid_stretching_ratio(u) == 1
        # the *metrics*, which the stencils actually use, are exact
        @test all(==(rg.dz), cell_thickness(u))
        @test all(==(rg.dz), center_spacing(u))

        # all stretching kinds fill the domain exactly
        for st in (:uniform, (:geometric, 1.08), (:tanh, 1.5), (:tanh_both, 1.5))
            g = StretchedGrid(Float64; size = SZ, extent = EXT, halo = HALO,
                              stretching = st)
            kk = interior(g)[3]
            @test sum(cell_thickness(g)[kk]) ≈ EXT[3] rtol = 1e-13
            @test all(>(0), cell_thickness(g))
        end

        # custom faces
        zf_custom = [EXT[3] * (k / SZ[3])^1.5 for k in 0:SZ[3]]
        gc = StretchedGrid(Float64; size = SZ, extent = EXT, halo = HALO,
                           stretching = (:custom, zf_custom))
        kk = interior(gc)[3]
        @test zfaces(gc)[first(kk):last(kk)+1] ≈ zf_custom rtol = 1e-13

        # argument validation
        @test_throws ArgumentError StretchedGrid(Float64; size = SZ, extent = EXT,
                                                halo = HALO, stretching = (:custom, [0.0, 1.0]))
        @test_throws ArgumentError StretchedGrid(Float64; size = SZ, extent = EXT,
                                                halo = HALO, stretching = :nonsense)
        @test_throws ArgumentError StretchedGrid(Float64; size = SZ, extent = EXT,
                                                halo = (0, 2, 2))

        # RegularGrid lift
        lifted = StretchedGrid(rg)
        @test all(==(rg.dz), cell_thickness(lifted))
        @test StretchedGrid(lifted) === lifted
    end

    # =====================================================================
    @testset "dispatch reaches the StretchedGrids methods" begin
        sg = StretchedGrid(Float64; size = SZ, extent = EXT, halo = HALO,
                           stretching = (:tanh, 1.5))
        c  = Field(CellCenter(), sg)
        fz = Field(FaceZ(), sg)
        fx = Field(FaceX(), sg)
        fy = Field(FaceY(), sg)
        cc = Field(CellCenter(), sg)

        # Before KA-d (#299) the StretchedGrid had its own copies of these
        # kernels and this test asked `which` method ran. Since KA-d one
        # kernel serves both grids and reads the grid's metrics: what must hold
        # is that a StretchedGrid hands the kernels its per-level vertical
        # metrics, not the uniform ones of the RegularGrid.
        m = FV.grid_metrics(sg)
        @test m.z isa FV.LevelZ
        @test m.z === sg.zm
        k = sg.Hz + 3
        @test FV._idz(m, k) == 1 / sg.dzc[k] && FV._idzf(m, k) == 1 / sg.dzf[k]
        @test FV._idz(m, k) != FV._idz(m, k + 1)          # really per level
        for (out, inp) in ((fz, c), (cc, fz), (cc, c))
            @test hasmethod(FV._∂z!, Tuple{typeof(out), typeof(inp), typeof(sg)})
        end

        # ... and behaviourally: a uniform-dz stencil would give a different
        # answer on this stretched grid, so the metric really is being used.
        set_physical!(c, (x, y, z) -> z^2)
        ∂z(fz, c)
        kk   = interior(sg)[3]
        k    = first(kk) + 3
        zc   = zcenters(sg)
        got  = fz.data[4, 4, k]
        exact = (zc[k]^2 - zc[k-1]^2) / (zc[k] - zc[k-1])
        naive = (zc[k]^2 - zc[k-1]^2) / (EXT[3] / SZ[3])   # uniform-dz version
        @test got ≈ exact rtol = 1e-14
        @test abs(got - naive) > 1e-3 * abs(exact)

        # the public façade routes through the same method
        fz2 = Field(FaceZ(), sg)
        FV._∂z!(fz2, c, sg)
        @test maxdiff(fz, fz2) == 0
    end

    # =====================================================================
    @testset "uniform stretching reduces to RegularGrid" begin
        rg = RegularGrid(Float64; size = SZ, extent = EXT, halo = HALO)
        sg = StretchedGrid(Float64; size = SZ, extent = EXT, halo = HALO,
                           stretching = :uniform)

        prof = (x, y, z) -> sin(2π * z / EXT[3]) + 0.4cos(2π * x / EXT[1]) +
                            0.2sin(2π * y / EXT[2]) + 1e-3 * z
        worst = 0.0

        # --- ∂z, all three location pairs --------------------------------
        for (LO, LI) in ((FaceZ, CellCenter), (CellCenter, FaceZ), (CellCenter, CellCenter))
            ir = Field(LI(), rg); is = Field(LI(), sg)
            set_physical!(ir, prof); set_physical!(is, prof)
            @test maximum(abs.(ir.data .- is.data)) == 0    # identical input data
            or = Field(LO(), rg); os = Field(LO(), sg)
            ∂z(or, ir); ∂z(os, is)
            d = maxdiff(or, os); worst = max(worst, d)
            @test d < 1e-14
        end

        # --- laplacian at every location ---------------------------------
        for L in (CellCenter, FaceX, FaceY, FaceZ)
            ir = Field(L(), rg); is = Field(L(), sg)
            set_physical!(ir, prof); set_physical!(is, prof)
            or = Field(L(), rg); os = Field(L(), sg)
            laplacian(or, ir); laplacian(os, is)
            d = maxdiff(or, os); worst = max(worst, d)
            @test d < 1e-14
        end

        # --- staggered divergence ----------------------------------------
        ufun = (x, y, z) -> cos(2π * x / EXT[1]) * z / EXT[3]
        vfun = (x, y, z) -> sin(2π * y / EXT[2]) * (1 - z / EXT[3])
        wfun = (x, y, z) -> sin(π * z / EXT[3]) * cos(2π * x / EXT[1])
        ur = Field(FaceX(), rg); vr = Field(FaceY(), rg); wr = Field(FaceZ(), rg)
        us = Field(FaceX(), sg); vs = Field(FaceY(), sg); ws = Field(FaceZ(), sg)
        for (a, b, fn) in ((ur, us, ufun), (vr, vs, vfun), (wr, ws, wfun))
            set_physical!(a, fn); set_physical!(b, fn)
        end
        dr = Field(CellCenter(), rg); ds = Field(CellCenter(), sg)
        divergence(dr, ur, vr, wr); divergence(ds, us, vs, ws)
        d = maxdiff(dr, ds); worst = max(worst, d)
        @test d < 1e-14

        # collocated divergence too
        ur2 = Field(CellCenter(), rg); vr2 = Field(CellCenter(), rg); wr2 = Field(CellCenter(), rg)
        us2 = Field(CellCenter(), sg); vs2 = Field(CellCenter(), sg); ws2 = Field(CellCenter(), sg)
        for (a, b, fn) in ((ur2, us2, ufun), (vr2, vs2, vfun), (wr2, ws2, wfun))
            set_physical!(a, fn); set_physical!(b, fn)
        end
        divergence(dr, ur2, vr2, wr2); divergence(ds, us2, vs2, ws2)
        d = maxdiff(dr, ds); worst = max(worst, d)
        @test d < 1e-14

        # --- interpolation, both directions ------------------------------
        cr = Field(CellCenter(), rg); cs = Field(CellCenter(), sg)
        set_physical!(cr, prof); set_physical!(cs, prof)
        fr = Field(FaceZ(), rg); fs = Field(FaceZ(), sg)
        interpolate_to_face!(fr, cr); interpolate_to_face!(fs, cs)
        d = maxdiff(fr, fs); worst = max(worst, d)
        @test d < 1e-14
        interpolate_to_center!(cr, wr); interpolate_to_center!(cs, ws)
        d = maxdiff(cr, cs); worst = max(worst, d)
        @test d < 1e-14

        # --- horizontal derivatives (should be literally identical) ------
        set_physical!(cr, prof); set_physical!(cs, prof)
        for (op, LO) in ((∂x, FaceX), (∂y, FaceY))
            or = Field(LO(), rg); os = Field(LO(), sg)
            op(or, cr); op(os, cs)
            d = maxdiff(or, os); worst = max(worst, d)
            @test d == 0
        end

        @info "uniform → RegularGrid reduction: worst absolute difference = $worst"
        @test worst < 1e-14
    end

    # =====================================================================
    @testset "tanh stretching sanity" begin
        Nz = 32
        g  = StretchedGrid(Float64; size = (4, 4, Nz), extent = EXT, halo = HALO,
                           stretching = (:tanh, 1.5))
        kk  = interior(g)[3]
        dz  = cell_thickness(g)[kk]
        @test length(dz) == Nz
        @test all(diff(dz) .> 0)                    # thickens away from the wall
        @test sum(dz) ≈ EXT[3] rtol = 1e-13
        r = grid_stretching_ratio(g)
        @test r > 1
        @test r < 1.3
        @test dz[1] < 0.5 * EXT[3] / Nz             # genuinely refined at the wall
        @info "tanh(β=1.5, Nz=$Nz): dz₁ = $(round(dz[1]; digits=3)) m, " *
              "dz_top = $(round(dz[end]; digits=3)) m, r_max = $(round(r; digits=5))"

        # centres are the midpoints of their faces, faces are monotone
        zf, zc = zfaces(g), zcenters(g)
        @test all(k -> zc[k] ≈ 0.5 * (zf[k] + zf[k+1]), eachindex(zc))
        @test all(diff(zf) .> 0)

        # geometric stretching honours its ratio exactly
        gg = StretchedGrid(Float64; size = (4, 4, Nz), extent = EXT, halo = HALO,
                           stretching = (:geometric, 1.07))
        dzg = cell_thickness(gg)[interior(gg)[3]]
        @test all(k -> dzg[k+1] / dzg[k] ≈ 1.07, 1:Nz-1)
        @test grid_stretching_ratio(gg) ≈ 1.07 rtol = 1e-12

        # tanh_both is symmetric
        gb = StretchedGrid(Float64; size = (4, 4, Nz), extent = EXT, halo = HALO,
                           stretching = (:tanh_both, 1.5))
        dzb = cell_thickness(gb)[interior(gb)[3]]
        @test dzb ≈ reverse(dzb) rtol = 1e-12
        @test dzb[1] < dzb[Nz ÷ 2]
    end

    # =====================================================================
    @testset "derivative exactness on polynomials (stretched grid)" begin
        g = StretchedGrid(Float64; size = (4, 4, 32), extent = EXT, halo = HALO,
                          stretching = (:tanh, 1.8))
        a, b = 0.037, -1.25
        lin  = (x, y, z) -> a * z + b

        # --- centre → face gradient: exact for linear --------------------
        c  = Field(CellCenter(), g); set_physical!(c, lin)
        fz = Field(FaceZ(), g);      ∂z(fz, c)
        e_lin_face = maximum(abs.(interior_view(fz) .- a))
        @test e_lin_face < 1e-12
        @info "∂z (centre→face) on linear profile, stretched grid: max error = $e_lin_face"

        # --- face → centre: exact for linear -----------------------------
        wf = Field(FaceZ(), g); set_physical!(wf, lin)
        cc = Field(CellCenter(), g); ∂z(cc, wf)
        @test maximum(abs.(interior_view(cc) .- a)) < 1e-12

        # --- collocated non-uniform three-point stencil ------------------
        set_physical!(c, lin)
        d1 = Field(CellCenter(), g); ∂z(d1, c)
        e_lin_c = maximum(abs.(interior_view(d1) .- a))
        @test e_lin_c < 1e-12

        # The unequal-spacing Lagrange weights are exact through degree 2,
        # so a quadratic is also differentiated exactly...
        quad = (x, y, z) -> 3.0e-6 * z^2 + a * z + b
        set_physical!(c, quad)
        d2 = Field(CellCenter(), g); ∂z(d2, c)
        e_quad = maxerr(d2, (x, y, z) -> 6.0e-6 * z + a)
        @test e_quad < 1e-11

        # ... whereas the naive "(f[k+1]-f[k-1])/(h₋+h₊)" form — the stencil
        # you get by forgetting the f[k] term — is NOT: it carries a
        # first-order error proportional to the local stretching.
        dzf  = center_spacing(g)
        zc   = zcenters(g)
        kk   = interior(g)[3]
        e_naive = 0.0
        for k in kk
            naive = (c.data[4,4,k+1] - c.data[4,4,k-1]) / (dzf[k] + dzf[k+1])
            e_naive = max(e_naive, abs(naive - (6.0e-6 * zc[k] + a)))
        end
        @test e_naive > 1e4 * max(e_quad, eps())
        @info "collocated ∂z on quadratic: metric-correct = $e_quad, " *
              "naive (f₊-f₋)/(h₋+h₊) = $e_naive"

        # A cubic is not reproduced exactly; the error must be 2nd order.
        cub = (x, y, z) -> (z / EXT[3])^3
        errs = Float64[]
        for Nz in (32, 64)
            gg = StretchedGrid(Float64; size = (4, 4, Nz), extent = EXT, halo = HALO,
                               stretching = (:tanh, 1.8))
            cf = Field(CellCenter(), gg); set_physical!(cf, cub)
            df = Field(CellCenter(), gg); ∂z(df, cf)
            push!(errs, maxerr(df, (x, y, z) -> 3z^2 / EXT[3]^3))
        end
        @test errs[1] / errs[2] > 3.0
        @info "collocated ∂z on cubic: error ratio 32→64 levels = " *
              "$(round(errs[1] / errs[2]; digits=3)) (2nd order ⇒ ≈4)"
    end

    # =====================================================================
    @testset "Laplacian 2nd-order convergence on a stretched grid" begin
        fun  = (x, y, z) -> sin(π * z / EXT[3])
        d2   = (x, y, z) -> -(π / EXT[3])^2 * sin(π * z / EXT[3])
        errs = Float64[]
        for Nz in (24, 48, 96)
            g = StretchedGrid(Float64; size = (4, 4, Nz), extent = EXT, halo = HALO,
                              stretching = (:tanh, 1.5))
            f = Field(CellCenter(), g); set_physical!(f, fun)
            l = Field(CellCenter(), g); laplacian(l, f)
            push!(errs, maxerr(l, d2))
        end
        r1 = errs[1] / errs[2]
        r2 = errs[2] / errs[3]
        @info "stretched Laplacian convergence: errors = $errs, ratios = " *
              "$(round(r1; digits=3)), $(round(r2; digits=3)) (2nd order ⇒ ≈4)"
        @test r1 > 3.4
        @test r2 > 3.7
        @test r2 < 4.4

        # and it is exact (to round-off) on a linear profile, at every level
        g = StretchedGrid(Float64; size = (4, 4, 32), extent = EXT, halo = HALO,
                          stretching = (:tanh, 1.8))
        f = Field(CellCenter(), g); set_physical!(f, (x, y, z) -> 2.5z - 7)
        l = Field(CellCenter(), g); laplacian(l, f)
        @test maximum(abs.(interior_view(l))) < 1e-12
    end

    # =====================================================================
    @testset "volume-weighted z-interpolation is exact for linear profiles" begin
        g = StretchedGrid(Float64; size = (4, 4, 32), extent = EXT, halo = HALO,
                          stretching = (:tanh, 1.8))
        a, b = 0.42, 13.0
        lin  = (x, y, z) -> a * z + b
        c  = Field(CellCenter(), g); set_physical!(c, lin)
        fz = Field(FaceZ(), g)
        interpolate_to_face!(fz, c)

        zf, dzc = zfaces(g), cell_thickness(g)
        kk = interior(g)[3]
        e_weighted = 0.0
        e_naive    = 0.0
        for k in kk
            exact = a * zf[k] + b
            e_weighted = max(e_weighted, abs(fz.data[4, 4, k] - exact))
            naive = 0.5 * (c.data[4, 4, k-1] + c.data[4, 4, k])
            e_naive = max(e_naive, abs(naive - exact))
        end
        @test e_weighted < 1e-12
        @test e_naive > 1e-3              # naive averaging is visibly wrong
        @test e_naive > 1e6 * max(e_weighted, eps())
        @info "z-interpolation of a linear profile: volume-weighted = $e_weighted, " *
              "naive ½(f_{k-1}+f_k) = $e_naive"

        # face → centre is exact too (centres are face midpoints by definition)
        w = Field(FaceZ(), g); set_physical!(w, lin)
        cc = Field(CellCenter(), g); interpolate_to_center!(cc, w)
        @test maxerr(cc, lin) < 1e-12

        # sanity: on a uniform grid the two weightings coincide
        gu = StretchedGrid(Float64; size = (4, 4, 32), extent = EXT, halo = HALO)
        cu = Field(CellCenter(), gu); set_physical!(cu, lin)
        fu = Field(FaceZ(), gu); interpolate_to_face!(fu, cu)
        ku = interior(gu)[3]
        @test maximum(abs(fu.data[4,4,k] - 0.5 * (cu.data[4,4,k-1] + cu.data[4,4,k]))
                      for k in ku) < 1e-12
    end

    # =====================================================================
    @testset "divergence of a solenoidal field vanishes (stretched grid)" begin
        g = StretchedGrid(Float64; size = (8, 4, 32), extent = EXT, halo = HALO,
                          stretching = (:tanh, 1.8))
        Hx, Hy, Hz = g.Hx, g.Hy, g.Hz
        ii, jj, kk = interior(g)
        dzc = cell_thickness(g)
        zf  = zfaces(g)

        # Build the velocity from a discrete streamfunction Ψ(x_face, z_face)
        # so that the *discrete* fluxes telescope exactly:
        #     u dzc dy = ΔzΨ,   w dx dy = -ΔxΨ
        # ⇒ (u[i+1]-u[i])/dx + (w[k+1]-w[k])/dzc ≡ 0 to round-off, on any
        # face distribution. This tests the metric, not the smoothness.
        Ψ(x, z) = 1.0e4 * sin(2π * x / EXT[1]) * (1 - cos(π * z / EXT[3]))
        xf(i) = g.x0 + (i - Hx - 1) * g.dx

        u = Field(FaceX(), g); v = Field(FaceY(), g); w = Field(FaceZ(), g)
        for k in first(kk):last(kk), j in jj, i in first(ii):last(ii)+1
            u.data[i, j, k] = (Ψ(xf(i), zf[k+1]) - Ψ(xf(i), zf[k])) / dzc[k]
        end
        for k in first(kk):last(kk)+1, j in jj, i in ii
            w.data[i, j, k] = -(Ψ(xf(i+1), zf[k]) - Ψ(xf(i), zf[k])) / g.dx
        end
        d = Field(CellCenter(), g)
        divergence(d, u, v, w)
        maxdiv = maximum(abs.(interior_view(d)))
        scale  = maximum(abs.(interior_view(u))) / g.dx
        @test maxdiv < 1e-12
        @info "solenoidal field on stretched grid: max|∇·u| = $maxdiv " *
              "(relative to term scale $(round(scale; sigdigits=3)): " *
              "$(round(maxdiv / scale; sigdigits=3)))"
    end

    # =====================================================================
    @testset "terrain-following grid: flat terrain reduces exactly" begin
        for st in (:uniform, (:tanh, 1.6))
            sg = StretchedGrid(Float64; size = SZ, extent = EXT, halo = HALO,
                               stretching = st)
            tf = terrain_following_grid(sg, 0.0)

            @test tf isa TerrainFollowingGrid{Float64}
            @test totalsize(tf) == totalsize(sg)
            @test interior(tf)  == interior(sg)
            J = metric_jacobian(tf)
            @test all(==(1.0), J)                      # Jacobian identically 1
            @test all(==(0.0), terrain_height(tf))
            @test size(physical_heights(tf)) == totalsize(tf)
            # physical heights collapse onto the σ levels
            zc = zcenters(sg)
            @test maximum(abs(physical_heights(tf)[i, j, k] - zc[k])
                          for i in axes(J, 1), j in axes(J, 2), k in axes(J, 3)) == 0

            prof = (x, y, z) -> sin(2π * z / EXT[3]) * cos(2π * x / EXT[1]) +
                                0.3sin(2π * y / EXT[2]) + 4e-4 * z
            worst = 0.0

            for (LO, LI, op) in ((FaceZ, CellCenter, ∂z), (CellCenter, FaceZ, ∂z),
                                 (CellCenter, CellCenter, ∂z),
                                 (FaceX, CellCenter, ∂x), (CellCenter, FaceX, ∂x),
                                 (FaceY, CellCenter, ∂y), (CellCenter, FaceY, ∂y),
                                 (CellCenter, CellCenter, ∂x),
                                 (CellCenter, CellCenter, ∂y))
                is = Field(LI(), sg); it = Field(LI(), tf)
                set_physical!(is, prof); set_physical!(it, prof)
                @test maximum(abs.(is.data .- it.data)) == 0
                os = Field(LO(), sg); ot = Field(LO(), tf)
                op(os, is); op(ot, it)
                worst = max(worst, maxdiff(os, ot))
            end

            # laplacian at every location
            for L in (CellCenter, FaceX, FaceY, FaceZ)
                is = Field(L(), sg); it = Field(L(), tf)
                set_physical!(is, prof); set_physical!(it, prof)
                os = Field(L(), sg); ot = Field(L(), tf)
                laplacian(os, is); laplacian(ot, it)
                worst = max(worst, maxdiff(os, ot))
            end

            # staggered + collocated divergence
            ufun = (x, y, z) -> cos(2π * x / EXT[1]) * z / EXT[3]
            vfun = (x, y, z) -> sin(2π * y / EXT[2]) * (1 - z / EXT[3])
            wfun = (x, y, z) -> sin(π * z / EXT[3]) * cos(2π * x / EXT[1])
            for locs in ((FaceX, FaceY, FaceZ),
                         (CellCenter, CellCenter, CellCenter))
                us = Field(locs[1](), sg); vs = Field(locs[2](), sg); ws = Field(locs[3](), sg)
                ut = Field(locs[1](), tf); vt = Field(locs[2](), tf); wt = Field(locs[3](), tf)
                for (a, bb, fn) in ((us, ut, ufun), (vs, vt, vfun), (ws, wt, wfun))
                    set_physical!(a, fn); set_physical!(bb, fn)
                end
                ds = Field(CellCenter(), sg); dt = Field(CellCenter(), tf)
                divergence(ds, us, vs, ws); divergence(dt, ut, vt, wt)
                worst = max(worst, maxdiff(ds, dt))
            end

            # interpolation
            cs = Field(CellCenter(), sg); ct = Field(CellCenter(), tf)
            set_physical!(cs, prof); set_physical!(ct, prof)
            fs = Field(FaceZ(), sg); ft = Field(FaceZ(), tf)
            interpolate_to_face!(fs, cs); interpolate_to_face!(ft, ct)
            worst = max(worst, maxdiff(fs, ft))

            @info "flat-terrain reduction ($st): worst |TFG - StretchedGrid| = $worst"
            @test worst < 1e-13
        end
    end

    # =====================================================================
    @testset "terrain-following grid: metrics over a bell hill" begin
        Lx, Lz = 20_000.0, 12_000.0
        h₀, aa = 500.0, 2_000.0
        bell(x, y) = h₀ / (1 + ((x - Lx / 2) / aa)^2)

        sg = StretchedGrid(Float64; size = (64, 4, 48), extent = (Lx, 400.0, Lz),
                           halo = HALO, stretching = (:tanh, 1.4))
        tf = terrain_following_grid(sg, bell)

        H = Lz
        J = metric_jacobian(tf)
        h = terrain_height(tf)
        @test maximum(J) ≤ 1.0 + 1e-15
        @test minimum(J) ≈ (H - maximum(h)) / H rtol = 1e-14
        @test maximum(h) ≤ h₀ && maximum(h) > 0.99h₀   # grid resolves the summit
        # J is the analytic (H-h)/H, column by column
        @test maximum(abs(J[i, j, k] - (H - h[i, j]) / H)
                      for i in axes(J,1), j in axes(J,2), k in axes(J,3)) < 1e-15

        # coordinate surfaces: σ=0 follows the hill, σ=H is flat
        zfp = physical_heights(tf, FaceZ())
        kk  = interior(tf)[3]
        @test maximum(abs(zfp[i, j, first(kk)] - h[i, j]) for i in axes(J,1), j in axes(J,2)) < 1e-12
        ztop = [zfp[i, j, last(kk)] for i in axes(J,1), j in axes(J,2)]
        @test maximum(ztop) - minimum(ztop) < 0.02 * Lz   # nearly flat near the lid

        # the terrain correction must vanish at the lid and be O(h') at the ground
        @test maximum(abs.(tf.dσdx_zf[:, :, last(kk) + 1])) < 1e-12
        @test maximum(abs.(tf.dσdx[:, :, first(kk)])) > 1e-3

        # a function of physical height only has (almost) no horizontal
        # gradient at constant z — and the metric term is what delivers that
        f = Field(CellCenter(), tf); set_physical!(f, (x, y, z) -> z)
        gx = Field(CellCenter(), tf); ∂x(gx, f)
        with_metric = maximum(abs.(interior_view(gx)))
        # the same stencil without the cross-term ( = plain σ-surface difference )
        ii, jj, kkk = interior(tf)
        no_metric = 0.0
        for k in kkk, j in jj, i in ii
            no_metric = max(no_metric,
                            abs((f.data[i+1,j,k] - f.data[i-1,j,k]) / (2 * tf.dx)))
        end
        @info "∂x of z over a bell hill: with metric = $(round(with_metric; sigdigits=3)), " *
              "σ-surface only = $(round(no_metric; sigdigits=3))"
        @test with_metric < 0.02 * no_metric
        @test no_metric > 0.05

        # ∂z of z is 1 everywhere
        gz = Field(CellCenter(), tf); ∂z(gz, f)
        @test maximum(abs.(interior_view(gz) .- 1)) < 1e-11
    end

    # =====================================================================
    @testset "hydrostatic balance" begin
        g_accel, R_d, c_p, p_ref = 9.80665, 287.05, 1004.6, 100_000.0
        T₀, p_s = 280.0, 101_325.0
        H_s = R_d * T₀ / g_accel
        pfun(z) = p_s * exp(-z / H_s)
        θfun(z) = T₀ * (p_ref / pfun(z))^(R_d / c_p)

        # --- stretched grid, no terrain ---------------------------------
        sg = StretchedGrid(Float64; size = (8, 4, 48), extent = (2_000.0, 400.0, 12_000.0),
                           halo = HALO, stretching = (:tanh, 1.4))
        θ = Field(CellCenter(), sg); p = Field(CellCenter(), sg)
        set_physical!(θ, (x, y, z) -> θfun(z))
        set_physical!(p, (x, y, z) -> pfun(z))
        abs_s = hydrostatic_balance_error(θ, p, sg)
        rel_s = hydrostatic_balance_error(θ, p, sg; relative = true)
        @info "hydrostatic residual, stretched grid: $(round(abs_s; sigdigits=4)) Pa/m " *
              "(relative $(round(rel_s; sigdigits=4)))"
        @test rel_s < 1e-3
        @test abs_s < 1e-2

        # a deliberately wrong metric (dzc instead of dzf) must be far worse
        dzc, dzf = cell_thickness(sg), center_spacing(sg)
        ii, jj, kk = interior(sg)
        wrong = 0.0
        for k in first(kk)+1:last(kk)
            Π = (p.data[4,4,k] / p_ref)^(R_d / c_p)
            ρ = p.data[4,4,k] / (R_d * θ.data[4,4,k] * Π)
            wrong = max(wrong, abs((p.data[4,4,k] - p.data[4,4,k-1]) / dzc[k] + ρ * g_accel))
        end
        @info "same column with the wrong metric (dzc for dzf): " *
              "$(round(wrong; sigdigits=4)) Pa/m — $(round(wrong / abs_s; sigdigits=3))× worse"
        @test wrong > 20 * abs_s

        # --- isothermal atmosphere over a bell hill ---------------------
        Lx, Lz = 20_000.0, 12_000.0
        h₀, aa = 500.0, 2_000.0
        bell(x, y) = h₀ / (1 + ((x - Lx / 2) / aa)^2)
        tg = StretchedGrid(Float64; size = (64, 4, 48), extent = (Lx, 400.0, Lz),
                           halo = HALO, stretching = (:tanh, 1.4))
        tf = terrain_following_grid(tg, bell)
        θt = Field(CellCenter(), tf); pt = Field(CellCenter(), tf)
        set_physical!(θt, (x, y, z) -> θfun(z))
        set_physical!(pt, (x, y, z) -> pfun(z))
        abs_t = hydrostatic_balance_error(θt, pt, tf)
        rel_t = hydrostatic_balance_error(θt, pt, tf; relative = true)
        @info "hydrostatic residual over bell hill (h₀=$h₀ m, a=$aa m): " *
              "$(round(abs_t; sigdigits=4)) Pa/m (relative $(round(rel_t; sigdigits=4)))"
        @test rel_t < 1e-3
        @test abs_t < 1e-2

        # the terrain must not degrade the balance relative to flat ground
        @test abs_t < 5 * abs_s

        # it converges away at 2nd order under vertical refinement
        rels = Float64[]
        for Nz in (24, 48, 96)
            gg = StretchedGrid(Float64; size = (4, 4, Nz), extent = (Lx, 400.0, Lz),
                               halo = HALO, stretching = (:tanh, 1.4))
            tt = terrain_following_grid(gg, bell)
            θi = Field(CellCenter(), tt); pi_ = Field(CellCenter(), tt)
            set_physical!(θi, (x, y, z) -> θfun(z))
            set_physical!(pi_, (x, y, z) -> pfun(z))
            push!(rels, hydrostatic_balance_error(θi, pi_, tt; relative = true))
        end
        @info "hydrostatic residual convergence (relative): $rels, " *
              "ratios $(round(rels[1]/rels[2]; digits=2)), $(round(rels[2]/rels[3]; digits=2))"
        @test rels[1] / rels[2] > 3.0
        @test rels[2] / rels[3] > 3.5

        # also works on a plain RegularGrid (useful as a regression baseline)
        rg = RegularGrid(Float64; size = (4, 4, 48), extent = (2_000.0, 400.0, 12_000.0),
                         halo = HALO)
        θr = Field(CellCenter(), rg); pr = Field(CellCenter(), rg)
        set_physical!(θr, (x, y, z) -> θfun(z))
        set_physical!(pr, (x, y, z) -> pfun(z))
        @test hydrostatic_balance_error(θr, pr, rg; relative = true) < 1e-3
    end

    # =====================================================================
    @testset "integration: Fields, halos and physics kernels" begin
        g = StretchedGrid(Float64; size = (6, 5, 20), extent = EXT, halo = HALO,
                          stretching = (:tanh, 1.6))
        bcs = FieldBoundaryConditions(:periodic, :noflux)
        f = Field(CellCenter(), g)
        set!(f, (x, y, z) -> z)                 # interior only
        fill_halos!(f, bcs)                     # must work unchanged on the new mesh
        ii, jj, kk = interior(g)
        @test f.data[1, 3, first(kk)] == f.data[1 + g.Nx, 3, first(kk)]

        # metric-aware surface flux: thin bottom cell ⇒ bigger tendency
        t = Field(CellCenter(), g)
        Aeolus.Operators.add_surface_flux!(t, 1.0, 0.0)
        dzc = cell_thickness(g)
        @test t.data[4, 4, first(kk)] ≈ 1 / dzc[first(kk)] rtol = 1e-14
        @test t.data[4, 4, first(kk)] > 1 / (EXT[3] / 20)   # refined near the wall

        # metric-aware buoyancy uses volume weighting
        θ = Field(CellCenter(), g); set_physical!(θ, (x, y, z) -> 300.0 + 0.01z)
        tw = Field(FaceZ(), g)
        Aeolus.Operators.add_buoyancy!(tw, θ, 300.0, 9.81)
        zf = zfaces(g)
        k = first(kk) + 4
        @test tw.data[4, 4, k] ≈ 9.81 / 300.0 * (0.01 * zf[k]) rtol = 1e-10

        # sponge keyed on the true stretched heights
        s = Field(CellCenter(), g); set_physical!(s, (x, y, z) -> 1.0)
        ts = Field(CellCenter(), g)
        Aeolus.Operators.add_sponge!(ts, s, 0.1, 0.8EXT[3], EXT[3])
        zc = zcenters(g)
        for k in kk
            zc[k] < 0.8EXT[3] && @test ts.data[4, 4, k] == 0
        end
        @test minimum(interior_view(ts)) < 0
    end
end
