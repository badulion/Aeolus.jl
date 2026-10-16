# KA-i (#303): the multigrid kernels on launch!. The red-black smoother (one
# launch per colour over that colour's cells), the residual, the restriction
# and the prolongation must give the bits of the serial loops they replaced
# (copied below as the reference); the means and norms are canonical plane
# sums (a named change of the reduction order: the cycle counts of the old
# loops are kept); a solve is the same on the serial and the default backend
# and allocates nothing on one thread. The kernel checks also run on
# `CPUBackend(threshold = 1)`, which threads every launch when Julia has
# several threads (`julia -t 4`; the harness section `multigrid` of
# test/thread_invariance.jl covers whole solves at 1/4/24 threads).
using Test
using Random
using Aeolus
using Aeolus.Grids: CellCenter
using Aeolus.Fields: interior_view
using Aeolus.Multigrid
const MGI = Aeolus.Multigrid
const BKI = Aeolus.Backends

# ---- the loops of master before KA-i (src/Multigrid.jl), verbatim ----------

function ref_smooth!(φ, b, cx, cy, cz, nx, ny, nz, dx, dy, dz, bcs, nsweeps)
    ix2 = 1 / dx^2; iy2 = 1 / dy^2; iz2 = 1 / dz^2
    for _ in 1:nsweeps
        for colour in 0:1
            MGI._fill_halo!(φ, bcs...)
            if cx === nothing
                diag = -2 * (ix2 + iy2 + iz2)
                @inbounds for k in 2:nz+1, j in 2:ny+1, i in 2:nx+1
                    ((i + j + k) & 1) == colour || continue
                    off = (φ[i+1, j, k] + φ[i-1, j, k]) * ix2 +
                          (φ[i, j+1, k] + φ[i, j-1, k]) * iy2 +
                          (φ[i, j, k+1] + φ[i, j, k-1]) * iz2
                    φ[i, j, k] = (b[i, j, k] - off) / diag
                end
            else
                @inbounds for k in 2:nz+1, j in 2:ny+1, i in 2:nx+1
                    ((i + j + k) & 1) == colour || continue
                    ii = i - 1; jj = j - 1; kk = k - 1
                    fxp = cx[ii+1, jj, kk]; fxm = cx[ii, jj, kk]
                    fyp = cy[ii, jj+1, kk]; fym = cy[ii, jj, kk]
                    fzp = cz[ii, jj, kk+1]; fzm = cz[ii, jj, kk]
                    diag = -((fxp + fxm) * ix2 + (fyp + fym) * iy2 + (fzp + fzm) * iz2)
                    abs(diag) < eps(Float64) && continue
                    off = (fxp * φ[i+1, j, k] + fxm * φ[i-1, j, k]) * ix2 +
                          (fyp * φ[i, j+1, k] + fym * φ[i, j-1, k]) * iy2 +
                          (fzp * φ[i, j, k+1] + fzm * φ[i, j, k-1]) * iz2
                    φ[i, j, k] = (b[i, j, k] - off) / diag
                end
            end
        end
    end
    return φ
end

function ref_residual!(r, φ, b, cx, cy, cz, nx, ny, nz, dx, dy, dz, bcs)
    MGI._fill_halo!(φ, bcs...)
    ix2 = 1 / dx^2; iy2 = 1 / dy^2; iz2 = 1 / dz^2
    if cx === nothing
        @inbounds for k in 2:nz+1, j in 2:ny+1, i in 2:nx+1
            c = φ[i, j, k]
            r[i, j, k] = (φ[i+1, j, k] - 2c + φ[i-1, j, k]) * ix2 +
                         (φ[i, j+1, k] - 2c + φ[i, j-1, k]) * iy2 +
                         (φ[i, j, k+1] - 2c + φ[i, j, k-1]) * iz2
        end
    else
        @inbounds for k in 2:nz+1, j in 2:ny+1, i in 2:nx+1
            ii = i - 1; jj = j - 1; kk = k - 1
            fxp = cx[ii+1, jj, kk]; fxm = cx[ii, jj, kk]
            fyp = cy[ii, jj+1, kk]; fym = cy[ii, jj, kk]
            fzp = cz[ii, jj, kk+1]; fzm = cz[ii, jj, kk]
            r[i, j, k] =
                (fxp * (φ[i+1, j, k] - φ[i, j, k]) - fxm * (φ[i, j, k] - φ[i-1, j, k])) * ix2 +
                (fyp * (φ[i, j+1, k] - φ[i, j, k]) - fym * (φ[i, j, k] - φ[i, j-1, k])) * iy2 +
                (fzp * (φ[i, j, k+1] - φ[i, j, k]) - fzm * (φ[i, j, k] - φ[i, j, k-1])) * iz2
        end
    end
    @inbounds for k in 2:nz+1, j in 2:ny+1, i in 2:nx+1
        r[i, j, k] = b[i, j, k] - r[i, j, k]
    end
    return r
end

function ref_restrict!(b, r, cnx, cny, cnz)
    @inbounds for K in 1:cnz, J in 1:cny, I in 1:cnx
        i = 2I; j = 2J; k = 2K
        s = r[i,     j,     k    ] + r[i + 1, j,     k    ] +
            r[i,     j + 1, k    ] + r[i + 1, j + 1, k    ] +
            r[i,     j,     k + 1] + r[i + 1, j,     k + 1] +
            r[i,     j + 1, k + 1] + r[i + 1, j + 1, k + 1]
        b[I + 1, J + 1, K + 1] = s * (1 / 8)
    end
    return b
end

function ref_prolong!(φ, Φ, fnx, fny, fnz, bcs)
    MGI._fill_halo!(Φ, bcs...)
    w_near = 3 / 4; w_far = 1 / 4
    @inbounds for k in 1:fnz, j in 1:fny, i in 1:fnx
        I = cld(i, 2); sx = isodd(i) ? -1 : 1
        J = cld(j, 2); sy = isodd(j) ? -1 : 1
        K = cld(k, 2); sz = isodd(k) ? -1 : 1
        Ih = I + 1; Jh = J + 1; Kh = K + 1
        acc = 0.0
        for (dk, wk) in ((0, w_near), (sz, w_far))
            for (dj, wj) in ((0, w_near), (sy, w_far))
                for (di, wi) in ((0, w_near), (sx, w_far))
                    acc += wi * wj * wk * Φ[Ih + di, Jh + dj, Kh + dk]
                end
            end
        end
        φ[i + 1, j + 1, k + 1] += acc
    end
    return φ
end

# ---- cases --------------------------------------------------------------------

include(joinpath(@__DIR__, "support", "multigrid_cases.jl"))

# Cycle / iteration counts of the loops KA-i replaced (measured on the merge
# base, 32×32×16, seed 303). The means and norms changed their reduction
# order, which the issue allowed to move a count by ±1: measured identical.
const OLD_CYCLES = Dict("mg(periodic,neumann)" => 22, "mg(neumann,neumann)" => 25,
                        "mg(periodic,periodic)" => 21, "mg(W)" => 21, "mg(fmg)" => 21,
                        "mg(stretched)" => 117, "mg(cutcell)" => 43,
                        "mgcg(periodic,neumann)" => 12, "mgcg(neumann,neumann)" => 14,
                        "mgcg(stretched)" => 32, "mgcg(cutcell)" => 17)

@testset "KA-i: multigrid on launch! (#303)" begin
    @testset "kernels equal the serial loops ($(N), $(cut ? "cut-cell" : "constant"), $bcs)" for
            N in ((16, 12, 8), (15, 12, 9)), cut in (false, true),
            bcs in ((:periodic, :periodic, :neumann), (:neumann, :periodic, :periodic), (:neumann, :neumann, :neumann))
        nx, ny, nz = N
        g = RegularGrid(; size = N, extent = (300.0, 250.0, 80.0), halo = (1, 1, 1))
        cx, cy, cz = cut ? cutcell_coeffs(nx, ny, nz; block = (3:6, 4:7, 1:3)) : (nothing, nothing, nothing)
        s = MultigridPoissonSolver(g; bc_x = bcs[1], bc_y = bcs[2], bc_z = bcs[3], min_size = 2,
                                   coeff_x = cx, coeff_y = cy, coeff_z = cz)
        L = s.levels[1]
        rng = Xoshiro(3030)
        φ0 = randn(rng, size(L.φ)); b0 = randn(rng, size(L.b))
        args = (cx, cy, cz, nx, ny, nz, L.dx, L.dy, L.dz)
        for backend in (CPUBackend(), CPUBackend(serial = true), CPUBackend(threshold = 1))
            with_backend(backend) do
                # smoother: three sweeps
                L.φ .= φ0; L.b .= b0
                MGI._smooth_rbgs!(L, bcs..., 3)
                @test L.φ == ref_smooth!(copy(φ0), b0, args..., bcs, 3)
                # residual
                L.φ .= φ0; fill!(L.r, 0.0)
                MGI._compute_residual!(L, bcs...)
                @test L.r == ref_residual!(zeros(size(L.r)), copy(φ0), b0, args..., bcs)
                # the fine operator of MG-CG (interior-sized in and out)
                x = randn(rng, nx, ny, nz); out = similar(x)
                MGI._apply_fine!(out, x, s)
                xh = zeros(size(L.φ)); xh[2:nx+1, 2:ny+1, 2:nz+1] .= x
                @test out == .-ref_residual!(zeros(size(L.r)), xh, zeros(size(L.b)), args..., bcs)[2:nx+1, 2:ny+1, 2:nz+1]
            end
        end
        # grid transfer needs a coarse level (even sizes only)
        if length(s.levels) >= 2
            C = s.levels[2]
            for backend in (CPUBackend(), CPUBackend(serial = true), CPUBackend(threshold = 1))
                with_backend(backend) do
                    L.r .= randn(rng, size(L.r))
                    MGI._restrict!(C, L)
                    @test C.b[2:C.nx+1, 2:C.ny+1, 2:C.nz+1] ==
                          ref_restrict!(zeros(size(C.b)), L.r, C.nx, C.ny, C.nz)[2:C.nx+1, 2:C.ny+1, 2:C.nz+1]
                    # the residual's mean folded into the restriction = project, then restrict
                    m = 0.37
                    MGI._restrict!(C, L, m)
                    rp = copy(L.r); rp[2:nx+1, 2:ny+1, 2:nz+1] .-= m
                    @test C.b[2:C.nx+1, 2:C.ny+1, 2:C.nz+1] ==
                          ref_restrict!(zeros(size(C.b)), rp, C.nx, C.ny, C.nz)[2:C.nx+1, 2:C.ny+1, 2:C.nz+1]
                    C.φ .= randn(rng, size(C.φ)); L.φ .= φ0
                    Φ = copy(C.φ)
                    MGI._prolong_add!(L, C, bcs...)
                    @test L.φ == ref_prolong!(copy(φ0), Φ, nx, ny, nz, bcs)
                end
            end
        end
    end

    @testset "means and norms are the canonical plane sums" begin
        g = RegularGrid(; size = (24, 16, 12), extent = (1.0, 1.0, 1.0), halo = (1, 1, 1))
        s = MultigridPoissonSolver(g)
        L = s.levels[1]
        a = randn(Xoshiro(1), size(L.φ)) .+ 3.0
        I3 = (2:25, 2:17, 2:13)
        @test MGI._l2(L, a) == sqrt(BKI.ordered_sum(plane_partials!(zeros(12), abs2, I3, a)))
        @test MGI._l2(L, a) ≈ sqrt(sum(abs2, a[I3...])) rtol = 1e-14
        m = interior_sum(a, I3) / (24 * 16 * 12)
        b = copy(a); b[I3...] .-= m
        MGI._project_mean!(L, a)
        @test a == b
        @test abs(sum(a[I3...])) < 1e-12
    end

    @testset "cycle counts of the old loops; solutions to tolerance; serial = default" begin
        for (name, (m, s)) in mg_solver_cases((32, 32, 16))
            p, src = mg_solve_case(name, m, s)
            @test mg_cycles(s) == OLD_CYCLES[name]
            @test mg_residual_norm(s) < 1e-10
            @test abs(sum(interior_view(p))) < 1e-9 * sum(abs, interior_view(p))   # zero-mean solution
            p2, _ = with_backend(() -> mg_solve_case(name, m, s), CPUBackend(serial = true))
            @test agree(p2.data, p.data)
            @test mg_cycles(s) == OLD_CYCLES[name]
            a = @allocated solve!(p, src, s)
            @test a == 0
        end
    end
end
