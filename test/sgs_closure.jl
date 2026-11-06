# Era VI, P54-a (#167): the SGS closure contract.
#
#   * the velocity-gradient helper every Era VI closure reads, and the
#     strain stencil built from it;
#   * StaticSmagorinsky as a closure reproduces the built-in C_s run;
#   * the commit cadence: advance_sgs! once per accepted step;
#   * the κ_t field is the scalar diffusivity (discrete eigenvalue,
#     conservation).

using Test
using Aeolus
using Aeolus.Fields: interior_view, Field
using Aeolus.Grids: interior
using Random: Xoshiro, rand, randn

const OPS = Aeolus.Operators

# The pre-contract strain stencil (Operators._strain_at at cc4a150), kept
# here verbatim as the reference.
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


# ---- test closures -------------------------------------------------------

# Smagorinsky with memory: the commit count and the dt of every commit.
# `coef_calls` is instrumentation outside the memory (it is not rolled back).
mutable struct CountingSGS <: AbstractSGSClosure
    inner::StaticSmagorinsky{Float64,Nothing}
    n::Int
    dts::Vector{Float64}
    coef_calls::Int
end
CountingSGS(C_s) = CountingSGS(StaticSmagorinsky(C_s), 0, Float64[], 0)
Aeolus.Operators.sgs_coefficients!(f::SGSFields, c::CountingSGS, sim) =
    (c.coef_calls += 1; sgs_coefficients!(f, c.inner, sim))
Aeolus.Operators.advance_sgs!(c::CountingSGS, sim, dt) = (c.n += 1; push!(c.dts, dt); c)
Aeolus.Operators.sgs_state(c::CountingSGS) = (c.n, copy(c.dts))
Aeolus.Operators.restore_sgs_state!(c::CountingSGS, s) = (c.n = s[1]; c.dts = copy(s[2]); c)

# κ = κ₀ everywhere (a prescribed field when `κ` is an array), ν = 0.
struct PrescribedKappa{K} <: AbstractSGSClosure
    κ::K
end
function Aeolus.Operators.sgs_coefficients!(f::SGSFields, c::PrescribedKappa, sim)
    fill!(f.νh.data, 0)
    c.κ isa Real ? fill!(f.κh.data, c.κ) : (interior_view(f.κh) .= c.κ)
    return f
end

# A discretely divergence-free random field: the staggered curl of a
# smoothed random vector potential (periodic), scaled to max|u| = umax.
function solenoidal!(sim, rng; umax = 1.0, smooth = 2)
    m = sim.mesh; N = (m.Nx, m.Ny, m.Nz)
    A = [randn(rng, N...) for _ in 1:3]
    sm(a) = (b = copy(a); for _ in 1:smooth, d in 1:3
                 b = (circshift(b, ntuple(i -> i == d ? 1 : 0, 3)) .+ 2 .* b .+
                      circshift(b, ntuple(i -> i == d ? -1 : 0, 3))) ./ 4
             end; b)
    Ax, Ay, Az = sm.(A)
    D(a, d) = circshift(a, ntuple(i -> i == d ? -1 : 0, 3)) .- a      # forward difference
    u = D(Az, 2) ./ m.dy .- D(Ay, 3) ./ m.dz
    v = D(Ax, 3) ./ m.dz .- D(Az, 1) ./ m.dx
    w = D(Ay, 1) ./ m.dx .- D(Ax, 2) ./ m.dy
    s = umax / maximum(abs, vcat(vec(u), vec(v), vec(w)))
    interior_view(sim.u) .= s .* u
    interior_view(sim.v) .= s .* v
    interior_view(sim.w) .= s .* w
    for f in (sim.u, sim.v, sim.w)
        fill_halos!(f, sim.bcs)
    end
    return sim
end

# 16³ periodic decaying turbulence with a θ perturbation and buoyancy.
function decaying(; C_s = 0.0, seed = 54, kw...)
    g = RegularGrid(; size = (16, 16, 16), extent = (1.0, 1.0, 1.0), halo = (3, 3, 3))
    sim = BoussinesqSimulation(g; C_s = C_s, ν = 1e-5, θ₀ = 300.0, kw...)
    rng = Xoshiro(seed)
    solenoidal!(sim, rng)
    interior_view(sim.θ) .= 300 .+ 0.5 .* randn(rng, 16, 16, 16)
    return sim
end
state(s) = (copy(interior_view(s.u)), copy(interior_view(s.v)), copy(interior_view(s.w)),
            copy(interior_view(s.θ)))

@testset "Era VI — SGS closure contract (P54-a, #167)" begin

    @testset "velocity-gradient helper: exact on linear fields; strain" begin
        dx, dy, dz = 15.625, 15.625, 6.25
        g = RegularGrid(; size = (8, 8, 8), extent = (8dx, 8dy, 8dz), halo = (2, 2, 2))
        A = [ 0.013 -0.021  0.034;
              0.008  0.017 -0.052;
             -0.027  0.041 -0.030]             # ∂u_a/∂x_b, s⁻¹ (trace-free)
        u = Field(FaceX(), g); v = Field(FaceY(), g); w = Field(FaceZ(), g)
        Hx, Hy, Hz = g.Hx, g.Hy, g.Hz
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

        # _strain_at, built on the helper, is the pre-contract stencil, and the
        # Smagorinsky kernel built on it is (C_s Δ)²|S|
        rng = Xoshiro(167)
        for f in (u, v, w)
            f.data .= randn(rng, size(f.data))
        end
        same = all(_strain_ref(u.data, v.data, w.data, i, j, k, 1 / dx, 1 / dy, 1 / dz) ===
                   OPS._strain_at(u.data, v.data, w.data, i, j, k, 1 / dx, 1 / dy, 1 / dz)
                   for k in kk, j in jj, i in ii)
        @test same
        νt = Field(CellCenter(), g)
        compute_smagorinsky_νt!(νt, u, v, w, 0.17)
        Δ = (dx * dy * dz)^(1 / 3)
        @test all(νt.data[i, j, k] ≈ (0.17 * Δ)^2 *
                  _strain_ref(u.data, v.data, w.data, i, j, k, 1 / dx, 1 / dy, 1 / dz)[7]
                  for k in kk, j in jj, i in ii)
    end
    dt = 2e-3

    @testset "closure ≡ built-in C_s; commit cadence once per accepted step" begin
        n = 6
        ref = decaying(C_s = 0.17); st = decaying(); c = CountingSGS(0.17); cs = decaying()
        for _ in 1:n
            step!(ref, dt); step!(st, dt; sgs = StaticSmagorinsky(0.17)); step!(cs, dt; sgs = c)
        end
        @test values_agree(state(st), state(ref))
        @test values_agree(interior_view(st.κt), interior_view(st.νt) ./ st.Pr_t)
        @test c.n == n && c.dts == fill(dt, n) && c.coef_calls == n
    end

    @testset "κ_t is the scalar diffusivity: discrete eigenvalue" begin
        # θ = sin(2πz/Lz), u = 0, κ ≡ κ₀ (≠ νt/Prₜ = 0): the mode is an exact
        # eigenvector of the discrete diffusion, eigenvalue −κ₀(4/dz²)sin²(π dz/Lz);
        # forward Euler multiplies it by (1 + hλ) per step.
        g = RegularGrid(; size = (4, 4, 32), extent = (1.0, 1.0, 2.0), halo = (2, 2, 2))
        κ0 = 6e-2; h = 0.02; nst = 40
        λ = -κ0 * (4 / g.dz^2) * sin(π * g.dz / g.Lz)^2
        zc = [(k - 0.5) * g.dz for k in 1:g.Nz]
        prof = sin.(2π .* zc ./ g.Lz)
        amp(s) = sum(interior_view(s.θ)[1, 1, :] .* prof) / sum(abs2, prof)
        sim = BoussinesqSimulation(g)            # ν = 0, no buoyancy, C_s = 0
        for k in 1:g.Nz
            interior_view(sim.θ)[:, :, k] .= prof[k]
        end
        for _ in 1:nst
            step!(sim, h; sgs = PrescribedKappa(κ0))
        end
        @test abs(amp(sim) / (1 + h * λ)^nst - 1) <= 1e-10
        @test amp(sim) < 0.75                    # it decayed (κ₀ is what acts)
        @test maximum(abs, interior_view(sim.u)) == 0
    end

    @testset "conservation: Σθ dV with a random positive κt" begin
        g = RegularGrid(; size = (16, 16, 16), extent = (1.0, 1.0, 1.0), halo = (3, 3, 3))
        rng = Xoshiro(11)
        κ = 1e-3 .* (0.2 .+ rand(rng, 16, 16, 16))
        sim = BoussinesqSimulation(g)
        interior_view(sim.θ) .= 290 .+ randn(rng, 16, 16, 16)
        S0 = sum(interior_view(sim.θ)); V0 = sum(abs2, interior_view(sim.θ) .- S0 / length(κ))
        for _ in 1:20
            step!(sim, 0.5; sgs = PrescribedKappa(κ))
        end
        S = sum(interior_view(sim.θ))
        V = sum(abs2, interior_view(sim.θ) .- S / length(κ))
        @test abs(S - S0) / abs(S0) <= 1e-13
        @test V < 0.8 * V0                                  # and it did diffuse
    end
end
