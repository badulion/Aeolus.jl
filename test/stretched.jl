# =========================================================================
# Phase 24 — stretched & terrain-following vertical coordinates
# =========================================================================
# Reduction to RegularGrid, exactness of the non-uniform stencils (and the
# naive ones measurably not), 2nd-order convergence, a solenoidal field,
# and hydrostatic balance over a hill.

using Test
using Aeolus
using Aeolus.StretchedGrids
using Aeolus.Grids: interior, znodes, totalsize
using Aeolus.Fields: Field, set!, interior_view, mesh
using Aeolus.Operators: ∂x, ∂y, ∂z, laplacian, divergence,
                        interpolate_to_face!, interpolate_to_center!

maxdiff(a::Field, b::Field) = maximum(abs.(interior_view(a) .- interior_view(b)))
function maxerr(f::Field, fun)
    ref = Field(Aeolus.Fields.location(f), mesh(f))
    set_physical!(ref, fun)
    return maxdiff(f, ref)
end

@testset "Stretched & terrain-following grids" begin

    SZ, EXT, HALO = (6, 5, 24), (600.0, 500.0, 1000.0), (2, 2, 2)

    @testset "grid construction" begin
        for st in ((:geometric, 1.08), (:tanh, 1.5))
            g = StretchedGrid(Float64; size = SZ, extent = EXT, halo = HALO, stretching = st)
            dz = cell_thickness(g)[interior(g)[3]]
            @test sum(dz) ≈ EXT[3] rtol = 1e-13
            @test all(>(0), cell_thickness(g))
        end
        gg = StretchedGrid(Float64; size = SZ, extent = EXT, halo = HALO,
                           stretching = (:geometric, 1.07))
        @test grid_stretching_ratio(gg) ≈ 1.07 rtol = 1e-12
        @test_throws ArgumentError StretchedGrid(Float64; size = SZ, extent = EXT,
                                                halo = HALO, stretching = :nonsense)
    end

    @testset "uniform stretching reduces to RegularGrid" begin
        rg = RegularGrid(Float64; size = SZ, extent = EXT, halo = HALO)
        sg = StretchedGrid(Float64; size = SZ, extent = EXT, halo = HALO, stretching = :uniform)
        prof = (x, y, z) -> sin(2π * z / EXT[3]) + 0.4cos(2π * x / EXT[1]) + 1e-3 * z
        worst = 0.0
        for (LO, LI, op) in ((FaceZ, CellCenter, ∂z), (CellCenter, CellCenter, laplacian))
            ir = Field(LI(), rg); is = Field(LI(), sg)
            set_physical!(ir, prof); set_physical!(is, prof)
            or = Field(LO(), rg); os = Field(LO(), sg)
            op(or, ir); op(os, is)
            worst = max(worst, maxdiff(or, os))
        end
        @test worst < 1e-14
    end

    @testset "stencil exactness on a stretched grid" begin
        g = StretchedGrid(Float64; size = (4, 4, 32), extent = EXT, halo = HALO,
                          stretching = (:tanh, 1.8))
        a, b = 0.037, -1.25
        # collocated non-uniform three-point stencil: exact through degree 2 ...
        c = Field(CellCenter(), g); set_physical!(c, (x, y, z) -> 3.0e-6 * z^2 + a * z + b)
        d2 = Field(CellCenter(), g); ∂z(d2, c)
        e_quad = maxerr(d2, (x, y, z) -> 6.0e-6 * z + a)
        @test e_quad < 1e-11
        # ... the naive (f₊-f₋)/(h₋+h₊) is not: first-order error from the stretching
        dzf, zc, kk = center_spacing(g), zcenters(g), interior(g)[3]
        e_naive = maximum(abs((c.data[4,4,k+1] - c.data[4,4,k-1]) / (dzf[k] + dzf[k+1]) -
                              (6.0e-6 * zc[k] + a)) for k in kk)
        @test e_naive > 1e4 * max(e_quad, eps())
        # volume-weighted centre → face interpolation exact for linear
        lin = (x, y, z) -> 0.42z + 13.0
        set_physical!(c, lin)
        fz = Field(FaceZ(), g); interpolate_to_face!(fz, c)
        @test maxerr(fz, lin) < 1e-12
        # Laplacian exact on a linear profile
        l = Field(CellCenter(), g); laplacian(l, c)
        @test maximum(abs.(interior_view(l))) < 1e-12
    end

    @testset "Laplacian 2nd-order convergence" begin
        fun = (x, y, z) -> sin(π * z / EXT[3])
        errs = map((24, 48)) do Nz
            g = StretchedGrid(Float64; size = (4, 4, Nz), extent = EXT, halo = HALO,
                              stretching = (:tanh, 1.5))
            f = Field(CellCenter(), g); set_physical!(f, fun)
            l = Field(CellCenter(), g); laplacian(l, f)
            maxerr(l, (x, y, z) -> -(π / EXT[3])^2 * sin(π * z / EXT[3]))
        end
        @test errs[1] / errs[2] > 3.4
    end

    @testset "divergence of a solenoidal field vanishes" begin
        g = StretchedGrid(Float64; size = (8, 4, 32), extent = EXT, halo = HALO,
                          stretching = (:tanh, 1.8))
        Hx = g.Hx
        ii, jj, kk = interior(g)
        dzc, zf = cell_thickness(g), zfaces(g)
        # velocity from a discrete streamfunction: fluxes telescope exactly
        Ψ(x, z) = 1.0e4 * sin(2π * x / EXT[1]) * (1 - cos(π * z / EXT[3]))
        xf(i) = g.x0 + (i - Hx - 1) * g.dx
        u = Field(FaceX(), g); v = Field(FaceY(), g); w = Field(FaceZ(), g)
        for k in kk, j in jj, i in first(ii):last(ii)+1
            u.data[i, j, k] = (Ψ(xf(i), zf[k+1]) - Ψ(xf(i), zf[k])) / dzc[k]
        end
        for k in first(kk):last(kk)+1, j in jj, i in ii
            w.data[i, j, k] = -(Ψ(xf(i+1), zf[k]) - Ψ(xf(i), zf[k])) / g.dx
        end
        d = Field(CellCenter(), g); divergence(d, u, v, w)
        @test maximum(abs.(interior_view(d))) < 1e-12
    end

    @testset "terrain-following grid" begin
        # flat terrain: Jacobian 1, operators identical to the StretchedGrid
        sg = StretchedGrid(Float64; size = SZ, extent = EXT, halo = HALO, stretching = (:tanh, 1.6))
        tf0 = terrain_following_grid(sg, 0.0)
        @test all(==(1.0), metric_jacobian(tf0))
        prof = (x, y, z) -> sin(2π * z / EXT[3]) * cos(2π * x / EXT[1]) + 4e-4 * z
        is = Field(CellCenter(), sg); it = Field(CellCenter(), tf0)
        set_physical!(is, prof); set_physical!(it, prof)
        os = Field(FaceX(), sg); ot = Field(FaceX(), tf0)
        ∂x(os, is); ∂x(ot, it)
        @test maxdiff(os, ot) < 1e-13

        # bell hill: J = (H-h)/H, ∂x z ≈ 0 thanks to the metric term, ∂z z = 1
        Lx, Lz, h₀, aa = 20_000.0, 12_000.0, 500.0, 2_000.0
        bell(x, y) = h₀ / (1 + ((x - Lx / 2) / aa)^2)
        tg = StretchedGrid(Float64; size = (32, 4, 24), extent = (Lx, 400.0, Lz),
                           halo = HALO, stretching = (:tanh, 1.4))
        tf = terrain_following_grid(tg, bell)
        J, h = metric_jacobian(tf), terrain_height(tf)
        @test maximum(abs(J[i, j, k] - (Lz - h[i, j]) / Lz)
                      for i in axes(J,1), j in axes(J,2), k in axes(J,3)) < 1e-15
        f = Field(CellCenter(), tf); set_physical!(f, (x, y, z) -> z)
        gx = Field(CellCenter(), tf); ∂x(gx, f)
        ii, jj, kk = interior(tf)
        no_metric = maximum(abs((f.data[i+1,j,k] - f.data[i-1,j,k]) / (2 * tf.dx))
                            for k in kk, j in jj, i in ii)
        @test maximum(abs.(interior_view(gx))) < 0.02 * no_metric
        gz = Field(CellCenter(), tf); ∂z(gz, f)
        @test maximum(abs.(interior_view(gz) .- 1)) < 1e-11
    end

    @testset "hydrostatic balance over a hill" begin
        g_accel, R_d, c_p, p_ref = 9.80665, 287.05, 1004.6, 100_000.0
        T₀, p_s = 280.0, 101_325.0
        pfun(z) = p_s * exp(-z * g_accel / (R_d * T₀))
        θfun(z) = T₀ * (p_ref / pfun(z))^(R_d / c_p)
        Lx, Lz, h₀, aa = 20_000.0, 12_000.0, 500.0, 2_000.0
        bell(x, y) = h₀ / (1 + ((x - Lx / 2) / aa)^2)
        rels = map((24, 48)) do Nz
            gg = StretchedGrid(Float64; size = (4, 4, Nz), extent = (Lx, 400.0, Lz),
                               halo = HALO, stretching = (:tanh, 1.4))
            tt = terrain_following_grid(gg, bell)
            θ = Field(CellCenter(), tt); p = Field(CellCenter(), tt)
            set_physical!(θ, (x, y, z) -> θfun(z)); set_physical!(p, (x, y, z) -> pfun(z))
            hydrostatic_balance_error(θ, p, tt; relative = true)
        end
        @test rels[2] < 1e-3
        @test rels[1] / rels[2] > 3.0               # 2nd order in the vertical
    end

    @testset "metric-aware physics kernels" begin
        g = StretchedGrid(Float64; size = (6, 5, 20), extent = EXT, halo = HALO,
                          stretching = (:tanh, 1.6))
        kk = interior(g)[3]
        t = Field(CellCenter(), g)
        Aeolus.Operators.add_surface_flux!(t, 1.0, 0.0)
        @test t.data[4, 4, first(kk)] ≈ 1 / cell_thickness(g)[first(kk)] rtol = 1e-14
        θ = Field(CellCenter(), g); set_physical!(θ, (x, y, z) -> 300.0 + 0.01z)
        tw = Field(FaceZ(), g)
        Aeolus.Operators.add_buoyancy!(tw, θ, 300.0, 9.81)
        k = first(kk) + 4
        @test tw.data[4, 4, k] ≈ 9.81 / 300.0 * (0.01 * zfaces(g)[k]) rtol = 1e-10
    end
end
