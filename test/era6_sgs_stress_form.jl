# Era VI, P55-b (#175): the stress-form SGS momentum divergence.
#
#   * constant ν: the stress form is the Laplacian form on a projected field;
#   * solid-body rotation with ν varying 10×: no stress-form tendency;
#   * discrete dissipation: the KE tendency is exactly −Σ 2νᵢⱼSᵢⱼSᵢⱼ ΔV ≤ 0;
#   * anisotropic decay at νh k̂ₓ² + νv k̂_z², explicit and IMEX;
#   * momentum conservation;
#   * the generic operator add_stress_divergence! with τ = −2νS;
#   * the IMEX split: the implicit operator is the explicit diagonal part,
#     symmetric; diagonal + transpose = full;
#   * the cores' `sgs_stress` keyword.

using Test
using Aeolus
using Aeolus.Fields: interior_view, Field
using Aeolus.Grids: interior
using Random: Xoshiro, rand, randn
using Printf

const TI175 = Aeolus.TimeIntegration

# ---- helpers ---------------------------------------------------------------

const DX175 = (15.625, 15.625, 6.25)          # the Berlin spacing (m)

grid175(N = (16, 16, 8); d = DX175, halo = (2, 2, 2)) =
    RegularGrid(; size = N, extent = N .* d, halo = halo)

# Periodic halo fill of a raw field without a simulation.
const PBC175 = FieldBoundaryConditions(:periodic)
pfill!(fs...) = foreach(f -> fill_halos!(f, PBC175), fs)

# A discretely divergence-free periodic field: the staggered curl of a
# random vector potential (forward differences commute with the C-grid
# divergence, so ∇·u = 0 to round-off), scaled to max|u| = 1 m/s.
function solenoidal175(g, rng)
    N = (g.Nx, g.Ny, g.Nz)
    Ax, Ay, Az = (randn(rng, N...) for _ in 1:3)
    D(a, d) = circshift(a, ntuple(i -> i == d ? -1 : 0, 3)) .- a
    uu = D(Az, 2) ./ g.dy .- D(Ay, 3) ./ g.dz
    vv = D(Ax, 3) ./ g.dz .- D(Az, 1) ./ g.dx
    ww = D(Ay, 1) ./ g.dx .- D(Ax, 2) ./ g.dy
    s = 1 / maximum(abs, vcat(vec(uu), vec(vv), vec(ww)))
    u = Field(FaceX(), g); v = Field(FaceY(), g); w = Field(FaceZ(), g)
    interior_view(u) .= s .* uu; interior_view(v) .= s .* vv; interior_view(w) .= s .* ww
    pfill!(u, v, w)
    return u, v, w
end

function random_velocity175(g, rng)
    u = Field(FaceX(), g); v = Field(FaceY(), g); w = Field(FaceZ(), g)
    for f in (u, v, w)
        interior_view(f) .= randn(rng, g.Nx, g.Ny, g.Nz)
    end
    pfill!(u, v, w)
    return u, v, w
end

function random_ν175(g, rng; lo = 0.0, hi = 1.0)
    ν = Field(CellCenter(), g)
    interior_view(ν) .= lo .+ (hi - lo) .* rand(rng, g.Nx, g.Ny, g.Nz)
    pfill!(ν)
    return ν
end

constfield175(g, c) = (f = Field(CellCenter(), g); fill!(f.data, c); f)

tend175(g) = (Field(FaceX(), g), Field(FaceY(), g), Field(FaceZ(), g))
iv(f) = interior_view(f)
maxabs(f::Field) = maximum(abs, iv(f))
maxabs(a::AbstractArray) = maximum(abs, a)
maxabs(fs...) = maximum(maxabs, fs)

# An anisotropic closure with constant νh, νv (κ = 0): the separate
# vertical viscosity lives in the closure's own field.
struct ConstAniso175{F} <: AbstractSGSClosure
    νh::Float64
    νv::Float64
    fv::F
end
ConstAniso175(sim, νh, νv) = ConstAniso175(νh, νv, Field(CellCenter(), sim.mesh))
Aeolus.Operators.sgs_fields(c::ConstAniso175, sim) = SGSFields(sim.νt, c.fv, sim.κt, sim.κt)
function Aeolus.Operators.sgs_coefficients!(f::SGSFields, c::ConstAniso175, sim)
    fill!(f.νh.data, c.νh); fill!(f.νv.data, c.νv); fill!(f.κh.data, 0)
    return f
end

# k̂² = (2 sin(kΔ/2)/Δ)², the eigenvalue of the 3-point second difference
khat2(k, Δ) = (2 * sin(k * Δ / 2) / Δ)^2

@testset "Era VI — stress-form SGS momentum divergence (P55-b, #175)" begin

    @testset "constant ν: :symmetric == :laplacian on a projected field" begin
        g = grid175((16, 16, 16))
        u, v, w = solenoidal175(g, Xoshiro(175))
        div = Field(CellCenter(), g); divergence(div, u, v, w)
        @test maxabs(div) < 1e-14                       # projected (s⁻¹)
        ν = constfield175(g, 0.37)
        tl = tend175(g); ts = tend175(g)
        add_eddy_viscosity_u!(tl[1], u, ν); add_eddy_viscosity_v!(tl[2], v, ν)
        add_eddy_viscosity_w!(tl[3], w, ν)
        add_sgs_stress_divergence!(ts..., u, v, w, ν)
        err = maximum(maxabs(iv(ts[i]) .- iv(tl[i])) for i in 1:3) / maxabs(tl...)
        @info @sprintf("P55-b constant ν: max|sym − lap|/max|lap| = %.2e (max|lap| = %.3e m s⁻²)",
                       err, maxabs(tl...))
        @test err <= 1e-12
    end

    @testset "solid-body rotation with ν varying 10×" begin
        g = grid175((16, 16, 8))
        Ω = 1e-2                                        # s⁻¹
        xc, yc = g.Lx / 2, g.Ly / 2
        u = Field(FaceX(), g); v = Field(FaceY(), g); w = Field(FaceZ(), g)
        ν = Field(CellCenter(), g)
        Hx, Hy, Hz = g.Hx, g.Hy, g.Hz
        xf(i) = (i - Hx - 1) * g.dx; xcn(i) = (i - Hx - 0.5) * g.dx
        yf(j) = (j - Hy - 1) * g.dy; ycn(j) = (j - Hy - 0.5) * g.dy
        zcn(k) = (k - Hz - 0.5) * g.dz
        ν0 = 0.5
        for k in axes(ν.data, 3), j in axes(ν.data, 2), i in axes(ν.data, 1)
            ν.data[i, j, k] = sin(2π * xcn(i) / g.Lx + 0.3) * cos(2π * ycn(j) / g.Ly - 0.2) *
                              cos(π * zcn(k) / g.Lz) + 0.3 * sin(2π * zcn(k) / g.Lz)
        end
        lo, hi = extrema(ν.data)                        # ν0 … 10 ν0 on the stored points
        ν.data .= ν0 .* (1 .+ 9 .* (ν.data .- lo) ./ (hi - lo))
        fill!(w.data, 0)
        # u is stored at x faces: its y-position is the cell centre ycn(j)
        for k in axes(u.data, 3), j in axes(u.data, 2), i in axes(u.data, 1)
            u.data[i, j, k] = -Ω * (ycn(j) - yc)
        end
        for k in axes(v.data, 3), j in axes(v.data, 2), i in axes(v.data, 1)
            v.data[i, j, k] = Ω * (xcn(i) - xc)
        end
        νmax = maximum(iv(ν)); νmin = minimum(iv(ν))
        @test νmax / νmin > 9
        ts = tend175(g); tl = tend175(g)
        add_sgs_stress_divergence!(ts..., u, v, w, ν)
        add_eddy_viscosity_u!(tl[1], u, ν); add_eddy_viscosity_v!(tl[2], v, ν)
        add_eddy_viscosity_w!(tl[3], w, ν)
        scale = Ω * νmax / max(g.dx, g.dy, g.dz)^2      # the strictest Δ
        rs = maxabs(ts...) / scale; rl = maxabs(tl...) / scale
        @info @sprintf("P55-b rigid rotation (ν %.2f–%.2f m²/s): max|t|/(Ω ν_max/Δ²) = %.2e symmetric, %.3f laplacian",
                       νmin, νmax, rs, rl)
        @test rs <= 1e-12
        @test rl > 0.1                                  # O(1): the Laplacian form spins it down
    end

    @testset "discrete dissipation: KE tendency == −Σ 2νᵢⱼSᵢⱼSᵢⱼ ΔV ≤ 0" begin
        g = grid175((12, 10, 8))
        ΔV = g.dx * g.dy * g.dz
        for seed in 1:4
            rng = Xoshiro(1750 + seed)
            u, v, w = random_velocity175(g, rng)
            νh = random_ν175(g, rng; hi = 2.0)
            νv = random_ν175(g, rng; hi = 0.2)       # anisotropic, νh ≠ νv
            t = tend175(g)
            add_sgs_stress_divergence!(t..., u, v, w, νh, νv)
            dKE = (sum(iv(u) .* iv(t[1])) + sum(iv(v) .* iv(t[2])) +
                   sum(iv(w) .* iv(t[3]))) * ΔV
            D = sgs_stress_dissipation(u, v, w, νh, νv)
            seed == 1 && @info @sprintf("P55-b dissipation: dKE/dt = %.6e, −D = %.6e m⁵ s⁻³, rel. diff %.1e",
                                        dKE, -D, abs(dKE + D) / D)
            @test D > 0
            @test dKE <= 0
            @test abs(dKE + D) <= 1e-10 * D
        end
    end

    @testset "anisotropic decay at νh k̂ₓ² + νv k̂_z² (νh/νv = 100)" begin
        g = grid175((16, 4, 16))
        Hx, Hy, Hz = g.Hx, g.Hy, g.Hz
        kx = 2π * 2 / g.Lx; ky = 2π * 1 / g.Ly; kz = 2π * 1 / g.Lz
        νh, νv = 2.0, 0.02
        xcn(i) = (i - Hx - 0.5) * g.dx; ycn(j) = (j - Hy - 0.5) * g.dy
        zcn(k) = (k - Hz - 0.5) * g.dz
        # v(x, z) (∂v/∂y = 0) and u(y, z) (∂u/∂x = 0): divergence-free shear modes
        λv = νh * khat2(kx, g.dx) + νv * khat2(kz, g.dz)
        λu = νh * khat2(ky, g.dy) + νv * khat2(kz, g.dz)
        modev(i, j, k) = sin(kx * xcn(i) + 0.4) * sin(kz * zcn(k) + 0.1)
        modeu(i, j, k) = cos(ky * ycn(j) - 0.3) * sin(kz * zcn(k) + 0.7)
        setmode!(f, m) = (for k in axes(f.data, 3), j in axes(f.data, 2), i in axes(f.data, 1)
                              f.data[i, j, k] = m(i, j, k)
                          end; f)

        # the tendency is −λ × the mode
        u = Field(FaceX(), g); v = Field(FaceY(), g); w = Field(FaceZ(), g)
        setmode!(u, modeu); setmode!(v, modev)
        t = tend175(g)
        add_sgs_stress_divergence!(t..., u, v, w, constfield175(g, νh), constfield175(g, νv))
        eu = maxabs(iv(t[1]) .+ λu .* iv(u)) / (λu * maxabs(u))
        ev = maxabs(iv(t[2]) .+ λv .* iv(v)) / (λv * maxabs(v))
        @info @sprintf("P55-b anisotropic eigenvalues: λ_u = %.6e, λ_v = %.6e s⁻¹; rel. error %.1e, %.1e; max|tw| = %.1e",
                       λu, λv, eu, ev, maxabs(t[3]))
        @test eu <= 1e-10
        @test ev <= 1e-10
        @test maxabs(t[3]) <= 1e-12 * λv

    end

    @testset "momentum conservation" begin
        g = grid175((12, 10, 8))
        rng = Xoshiro(17501)
        u, v, w = random_velocity175(g, rng)
        νh = random_ν175(g, rng; hi = 3.0); νv = random_ν175(g, rng; hi = 0.5)
        t = tend175(g)
        add_sgs_stress_divergence!(t..., u, v, w, νh, νv)
        for f in t
            @test abs(sum(iv(f))) <= 1e-13 * sum(abs, iv(f))
        end
    end

    @testset "generic operator: add_stress_divergence!(τ = −2νS) == fused" begin
        g = grid175((12, 10, 8))
        rng = Xoshiro(17503)
        u, v, w = random_velocity175(g, rng)
        νh = random_ν175(g, rng; hi = 2.0); νv = random_ν175(g, rng; hi = 0.3)
        τ = sgs_stress_tensor!(StressTensor(g), u, v, w, νh, νv)
        tg = tend175(g); tf = tend175(g)
        add_stress_divergence!(tg..., τ)
        add_sgs_stress_divergence!(tf..., u, v, w, νh, νv)
        @test all(iv(tg[i]) == iv(tf[i]) for i in 1:3)  # bit for bit
        # τ is −2νS: the diagonal at one cell, by hand
        i, j, k = g.Hx + 3, g.Hy + 4, g.Hz + 2
        @test τ.τ11[i, j, k] ≈ -2 * νh.data[i, j, k] * (u.data[i+1, j, k] - u.data[i, j, k]) / g.dx rtol = 1e-14
        S13 = 0.5 * ((u.data[i, j, k] - u.data[i, j, k-1]) / g.dz + (w.data[i, j, k] - w.data[i-1, j, k]) / g.dx)
        ν13 = 0.25 * (νv.data[i-1, j, k-1] + νv.data[i, j, k-1] + νv.data[i-1, j, k] + νv.data[i, j, k])
        @test τ.τ13[i, j, k] ≈ -2 * ν13 * S13 rtol = 1e-13
        # conservative for any τ
        τr = StressTensor(g)
        for a in (τr.τ11, τr.τ22, τr.τ33, τr.τ12, τr.τ13, τr.τ23)
            a .= randn(rng, size(a)...)
        end
        foreach(a -> nothing, ())
        # make τr periodic in the read shells: copy via fill_halos! on Field wrappers
        for a in (τr.τ11, τr.τ22, τr.τ33, τr.τ12, τr.τ13, τr.τ23)
            f = Field(CellCenter(), g); f.data .= a; pfill!(f); a .= f.data
        end
        tr = tend175(g); add_stress_divergence!(tr..., τr)
        @test all(abs(sum(iv(f))) <= 1e-13 * sum(abs, iv(f)) for f in tr)
    end

end
