# Era VI, P54-a (#167): the SGS closure contract.
#
#   * the velocity-gradient helper every Era VI closure reads, and the
#     strain stencil built from it;
#   * the built-in closures as wrappers: bit for bit the pre-contract runs;
#   * the commit cadence: advance_sgs! once per accepted step on every core,
#     rolled back with the state when step_adaptive! rejects a step;
#   * θ-aware closures see the current state;
#   * the κ_t field is the scalar diffusivity (discrete eigenvalue, CFL,
#     conservation);
#   * the anelastic core takes the same closures.

using Test
using Aeolus
using Aeolus.Fields: interior_view, Field
using Aeolus.Grids: interior
using Random: Xoshiro, rand, randn
using Printf

const OPS = Aeolus.Operators

# The pre-contract strain stencil (Operators._strain_at at cc4a150), kept
# here verbatim as the bit-for-bit reference.
function _strain_ref(u, v, w, i, j, k, idx, idy, idz)
    S11 = (u[i+1,j,k] - u[i,j,k]) * idx
    S22 = (v[i,j+1,k] - v[i,j,k]) * idy
    S33 = (w[i,j,k+1] - w[i,j,k]) * idz
    du_dy = ((u[i,j+1,k] + u[i+1,j+1,k]) - (u[i,j-1,k] + u[i+1,j-1,k])) * (0.25 * idy)
    dv_dx = ((v[i+1,j,k] + v[i+1,j+1,k]) - (v[i-1,j,k] + v[i-1,j+1,k])) * (0.25 * idx)
    du_dz = ((u[i,j,k+1] + u[i+1,j,k+1]) - (u[i,j,k-1] + u[i+1,j,k-1])) * (0.25 * idz)
    dw_dx = ((w[i+1,j,k] + w[i+1,j,k+1]) - (w[i-1,j,k] + w[i-1,j,k+1])) * (0.25 * idx)
    dv_dz = ((v[i,j,k+1] + v[i,j+1,k+1]) - (v[i,j,k-1] + v[i,j+1,k-1])) * (0.25 * idz)
    dw_dy = ((w[i,j+1,k] + w[i,j+1,k+1]) - (w[i,j-1,k] + w[i,j-1,k+1])) * (0.25 * idy)
    S12 = 0.5 * (du_dy + dv_dx)
    S13 = 0.5 * (du_dz + dw_dx)
    S23 = 0.5 * (dv_dz + dw_dy)
    S2 = 2 * (S11^2 + S22^2 + S33^2) + 4 * (S12^2 + S13^2 + S23^2)
    return S11, S22, S33, S12, S13, S23, sqrt(S2)
end

@testset "Era VI — SGS closure contract (P54-a, #167)" begin

    @testset "velocity-gradient helper: exact on linear fields; strain bit for bit" begin
        dx, dy, dz = 15.625, 15.625, 6.25
        g = RegularGrid(; size = (8, 8, 8), extent = (8dx, 8dy, 8dz), halo = (2, 2, 2))
        A = [ 0.013 -0.021  0.034;
              0.008  0.017 -0.052;
             -0.027  0.041 -0.030]             # ∂u_a/∂x_b, s⁻¹ (trace-free)
        u = Field(FaceX(), g); v = Field(FaceY(), g); w = Field(FaceZ(), g)
        Hx, Hy, Hz = g.Hx, g.Hy, g.Hz
        # face positions of every stored value (halos included)
        xf(i) = (i - Hx - 1) * dx; xc(i) = (i - Hx - 0.5) * dx
        yf(j) = (j - Hy - 1) * dy; yc(j) = (j - Hy - 0.5) * dy
        zf(k) = (k - Hz - 1) * dz; zc(k) = (k - Hz - 0.5) * dz
        lin(a, x, y, z) = 0.7 + A[a, 1] * x + A[a, 2] * y + A[a, 3] * z
        for I in CartesianIndices(u.data)
            i, j, k = Tuple(I)
            u.data[I] = lin(1, xf(i), yc(j), zc(k))
            v.data[I] = lin(2, xc(i), yf(j), zc(k))
            w.data[I] = lin(3, xc(i), yc(j), zf(k))
        end
        ii, jj, kk = interior(g)
        err = 0.0
        for k in kk, j in jj, i in ii
            G = OPS._velocity_gradient_at(u.data, v.data, w.data, i, j, k, 1 / dx, 1 / dy, 1 / dz)
            err = max(err, maximum(abs(G[a][b] - A[a, b]) for a in 1:3, b in 1:3))
        end
        @test err < 1e-12
        @printf("  [gradient] linear field on (15.625, 15.625, 6.25) m: max|g − A| = %.1e s⁻¹\n", err)

        # _strain_at, now built on the helper, is the pre-contract stencil bit for bit
        rng = Xoshiro(167)
        for f in (u, v, w)
            f.data .= randn(rng, size(f.data))
        end
        same = all(_strain_ref(u.data, v.data, w.data, i, j, k, 1 / dx, 1 / dy, 1 / dz) ===
                   OPS._strain_at(u.data, v.data, w.data, i, j, k, 1 / dx, 1 / dy, 1 / dz)
                   for k in kk, j in jj, i in ii)
        @test same
        # ... and the Smagorinsky kernel built on _strain_at is (C_s Δ)²|S|
        νt = Field(CellCenter(), g)
        compute_smagorinsky_νt!(νt, u, v, w, 0.17)
        Δ = (dx * dy * dz)^(1 / 3)
        @test all(νt.data[i, j, k] == (0.17 * Δ)^2 *
                  _strain_ref(u.data, v.data, w.data, i, j, k, 1 / dx, 1 / dy, 1 / dz)[7]
                  for k in kk, j in jj, i in ii)
    end
end
