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
#   * the cores' `sgs_stress` keyword;
#   * the explicit CFL limit of the stress form: power-iterated spectral
#     radius vs `stress_form_cfl_viscosity` and the old Laplacian limit.

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

# A closure whose νh = νv is a prescribed field (κ = 0), so the CFL estimate
# sees exactly the ν the operator under test is built with.
struct GivenNu175{F} <: AbstractSGSClosure
    ν::F
end
Aeolus.Operators.sgs_fields(c::GivenNu175, sim) = SGSFields(sim.νt, sim.κt)
function Aeolus.Operators.sgs_coefficients!(f::SGSFields, c::GivenNu175, sim)
    copyto!(f.νh.data, c.ν.data); fill!(f.κh.data, 0)
    return f
end

# The largest eigenvalue magnitude of the explicit SGS momentum operator L
# (symmetric, negative semi-definite), by power iteration on −L; with
# `project = true`, of P L P, P the core's own pressure projection. Returns
# the Rayleigh quotient ⟨x, −Lx⟩/⟨x, x⟩ (a lower bound on ρ that converges to
# it) and the converged vector, left in (sim.u, sim.v, sim.w).
function spectral_radius175(sim, νh, νv, stress::Symbol; project = false, iters = 1500,
                            rng = Xoshiro(17507))
    g = sim.mesh
    u, v, w = sim.u, sim.v, sim.w
    for f in (u, v, w); iv(f) .= randn(rng, g.Nx, g.Ny, g.Nz); end
    t = tend175(g)
    λ = 0.0
    for _ in 1:iters
        project && TI175._project!(sim, 1.0)
        pfill!(u, v, w)
        foreach(x -> fill!(x.data, 0), t)
        if stress === :symmetric
            add_sgs_stress_divergence!(t..., u, v, w, νh, νv)
        else
            add_eddy_viscosity_u!(t[1], u, νh); add_eddy_viscosity_v!(t[2], v, νh)
            add_eddy_viscosity_w!(t[3], w, νh)
        end
        x2 = sum(abs2, iv(u)) + sum(abs2, iv(v)) + sum(abs2, iv(w))
        λ = -(sum(iv(u) .* iv(t[1])) + sum(iv(v) .* iv(t[2])) + sum(iv(w) .* iv(t[3]))) / x2
        n = sqrt(sum(abs2, iv(t[1])) + sum(abs2, iv(t[2])) + sum(abs2, iv(t[3])))
        iv(u) .= .-iv(t[1]) ./ n; iv(v) .= .-iv(t[2]) ./ n; iv(w) .= .-iv(t[3]) ./ n
    end
    project && TI175._project!(sim, 1.0)
    pfill!(u, v, w)
    return λ
end

# ‖u‖ after `n` forward-Euler steps uⁿ⁺¹ = (P)(uⁿ + dt L uⁿ) from the state in
# sim, relative to the start.
function euler_growth175(sim, ν, dt, n; project = false)
    g = sim.mesh
    nrm() = sqrt(sum(abs2, iv(sim.u)) + sum(abs2, iv(sim.v)) + sum(abs2, iv(sim.w)))
    n0 = nrm()
    t = tend175(g)
    for _ in 1:n
        pfill!(sim.u, sim.v, sim.w)
        foreach(x -> fill!(x.data, 0), t)
        add_sgs_stress_divergence!(t..., sim.u, sim.v, sim.w, ν)
        iv(sim.u) .+= dt .* iv(t[1]); iv(sim.v) .+= dt .* iv(t[2]); iv(sim.w) .+= dt .* iv(t[3])
        project && TI175._project!(sim, dt)
    end
    return nrm() / n0
end

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

        # time stepping: explicit Euler and IMEX ARK2 decay the v-mode at the
        # discrete amplification factor of λv
        function decay_run(stepper, h, n)
            sim = BoussinesqSimulation(g; ν = 0.0, θ₀ = 300.0)
            fill!(sim.θ.data, 300.0)                # no buoyancy
            setmode!(sim.v, modev)
            v0 = copy(iv(sim.v))
            c = ConstAniso175(sim, νh, νv)
            for _ in 1:n
                stepper(sim, h, c)
            end
            return v0, copy(iv(sim.v)), sim
        end
        # Euler: hλ_v = 0.05, so that h·λ_max ≈ 0.7 for the grid-scale
        # round-off modes (stability needs ≤ 2)
        hex = 0.05 / λv
        v0, v1, sim = decay_run((s, h, c) -> step!(s, h; sgs = c, sgs_stress = :symmetric), hex, 40)
        G = (1 - hex * λv)^40
        e_ex = maximum(abs, v1 .- G .* v0) / (G * maximum(abs, v0))
        @test e_ex <= 1e-10
        @test maxabs(sim.u) <= 1e-12 && maxabs(sim.w) <= 1e-12

        γ = 1 - 1 / sqrt(2)
        him = 5.0 / λv                      # hλ = 5: 25× the explicit limit
        v0, v1, sim = decay_run((s, h, c) -> step_imex!(s, h, ARK2(); sgs = c, sgs_stress = :symmetric,
                                                         tol = 1e-14, maxit = 500), him, 4)
        a = him * λv
        r1 = 1 / (1 + γ * a)
        Rimex = (1 - (1 - γ) * a * r1) / (1 + γ * a)
        Gim = Rimex^4       # 4 steps: |G| ≈ 1e-3 stays far above the round-off of the O(1) start
        e_im = maximum(abs, v1 .- Gim .* v0) / (abs(Gim) * maximum(abs, v0))
        @info @sprintf("P55-b anisotropic decay: Euler hλ=0.05 ×40 rel. err %.1e; ARK2 hλ=5 ×4 rel. err %.1e (G = %.4e)",
                       e_ex, e_im, Gim)
        @test e_im <= 1e-10
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
        # through the core: 20 explicit steps with the stress form
        sim = BoussinesqSimulation(grid175((16, 16, 16)); ν = 0.0, θ₀ = 300.0)
        fill!(sim.θ.data, 300.0)
        uu, vv, ww = solenoidal175(sim.mesh, Xoshiro(17502))
        iv(sim.u) .= iv(uu); iv(sim.v) .= iv(vv); iv(sim.w) .= iv(ww)
        iv(sim.u) .+= 0.3; iv(sim.v) .-= 0.2               # a mean wind
        Su0, Sv0 = sum(iv(sim.u)), sum(iv(sim.v))
        for _ in 1:20
            step!(sim, 0.5; sgs = StaticSmagorinsky(0.17), sgs_stress = :symmetric)
        end
        du = abs(sum(iv(sim.u)) - Su0) / abs(Su0); dv = abs(sum(iv(sim.v)) - Sv0) / abs(Sv0)
        @info @sprintf("P55-b momentum: |ΔΣu|/|Σu| = %.1e, |ΔΣv|/|Σv| = %.1e after 20 steps", du, dv)
        @test du <= 1e-13
        @test dv <= 1e-13
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

    @testset "IMEX split: implicit = explicit diagonal part, symmetric; diag + transpose = full" begin
        g = grid175((12, 10, 8))
        rng = Xoshiro(17504)
        u, v, w = random_velocity175(g, rng)
        νh = random_ν175(g, rng; lo = 0.1, hi = 2.0); νv = random_ν175(g, rng; lo = 0.01, hi = 0.2)
        # diagonal + transpose == full
        tf = tend175(g); td = tend175(g)
        add_sgs_stress_divergence!(tf..., u, v, w, νh, νv)
        add_sgs_stress_divergence!(td..., u, v, w, νh, νv; part = :diagonal)
        add_sgs_stress_divergence!(td..., u, v, w, νh, νv; part = :transpose)
        @test maximum(maxabs(iv(td[i]) .- iv(tf[i])) for i in 1:3) <= 1e-13 * maxabs(tf...)
        # with νh = νv the diagonal part is the Laplacian kernel
        tl = tend175(g); tdi = tend175(g)
        add_eddy_viscosity_u!(tl[1], u, νh); add_eddy_viscosity_v!(tl[2], v, νh)
        add_eddy_viscosity_w!(tl[3], w, νh)
        add_sgs_stress_divergence!(tdi..., u, v, w, νh; part = :diagonal)
        @test maximum(maxabs(iv(tdi[i]) .- iv(tl[i])) for i in 1:3) <= 1e-13 * maxabs(tl...)
        # the implicit operator of each component == the explicit diagonal part
        h = TI175._coef(νh.data, g); vv = TI175._coef(νv.data, g)
        td2 = tend175(g)
        add_sgs_stress_divergence!(td2..., u, v, w, νh, νv; part = :diagonal)
        for (f, L, t) in ((u, FaceX(), td2[1]), (v, FaceY(), td2[2]), (w, FaceZ(), td2[3]))
            out = Field(L, g)
            TI175._diffusion!(out, f, TI175._StressCoef(L, h, vv), g)
            @test maxabs(iv(out) .- iv(t)) <= 1e-12 * maxabs(t)
            # symmetric: ⟨x, L y⟩ == ⟨L x, y⟩ (periodic)
            x = Field(L, g); y = Field(L, g)
            iv(x) .= randn(rng, g.Nx, g.Ny, g.Nz); iv(y) .= randn(rng, g.Nx, g.Ny, g.Nz)
            pfill!(x, y)
            Lx = Field(L, g); Ly = Field(L, g)
            TI175._diffusion!(Lx, x, TI175._StressCoef(L, h, vv), g)
            TI175._diffusion!(Ly, y, TI175._StressCoef(L, h, vv), g)
            a1 = sum(iv(x) .* iv(Ly)); a2 = sum(iv(Lx) .* iv(y))
            @test abs(a1 - a2) <= 1e-12 * abs(a1)
            @test sum(iv(x) .* iv(Lx)) < 0                 # negative definite
        end
        # the per-direction scalar coefficient (νh, νv): x/y faces νh, z faces νv
        θ = Field(CellCenter(), g); kz = 2π / g.Lz
        for k in axes(θ.data, 3); θ.data[:, :, k] .= sin(kz * (k - g.Hz - 0.5) * g.dz); end
        out = Field(CellCenter(), g)
        TI175._diffusion!(out, θ, (3.0, 0.04), g)
        @test maxabs(iv(out) .+ 0.04 * khat2(kz, g.dz) .* iv(θ)) <= 1e-12 * 0.04 * khat2(kz, g.dz)
    end

    @testset "the cores' sgs_stress keyword" begin
        function mk()
            sim = BoussinesqSimulation(grid175((16, 16, 16)); ν = 1e-5, θ₀ = 300.0, C_s = 0.17)
            fill!(sim.θ.data, 300.0)
            uu, vv, ww = solenoidal175(sim.mesh, Xoshiro(17505))
            iv(sim.u) .= iv(uu); iv(sim.v) .= iv(vv); iv(sim.w) .= iv(ww)
            return sim
        end
        st(s) = (copy(iv(s.u)), copy(iv(s.v)), copy(iv(s.w)))
        a = mk(); b = mk(); c = mk()
        for _ in 1:5
            step!(a, 0.5); step!(b, 0.5; sgs_stress = :laplacian)
            step!(c, 0.5; sgs_stress = :symmetric)
        end
        @test st(a) == st(b)                          # the default is :laplacian, bit for bit
        @test st(a) != st(c)
        # with constant νt the forms agree; with Smagorinsky νt(x) they differ by
        # the transposed term only — small next to the Laplacian part
        d = maximum(maximum(abs, x .- y) for (x, y) in zip(st(a), st(c)))
        @test 0 < d < 1e-2
        @test_throws ArgumentError step!(mk(), 0.5; sgs_stress = :bogus)
        @test_throws ArgumentError step_rk3!(mk(), 0.5; sgs_stress = :bogus)
        @test_throws ArgumentError step_imex!(mk(), 0.5, ARK2(); sgs_stress = :bogus)
        # step_rk3!, step_imex!, step_adaptive! take it
        r = mk(); step_rk3!(r, 0.5; sgs_stress = :symmetric); @test all(isfinite, iv(r.u))
        i1 = mk(); step_imex!(i1, 0.5, ARK2(); sgs_stress = :symmetric)
        @test max_divergence(i1) < 1e-10
        i2 = mk(); step_imex!(i2, 0.5, Rosenbrock2(); sgs_stress = :symmetric)
        @test max_divergence(i2) < 1e-10
        # IMEX :symmetric vs explicit :symmetric at a small dt: same physics
        e = mk(); im = mk()
        for _ in 1:4
            step!(e, 0.05; sgs_stress = :symmetric)
            step_imex!(im, 0.05, ARK2(); sgs_stress = :symmetric, tol = 1e-12)
        end
        dd = maximum(maximum(abs, x .- y) for (x, y) in zip(st(e), st(im)))
        @info @sprintf("P55-b IMEX vs explicit (:symmetric, 4 × 0.05 s): max|Δu| = %.2e m/s", dd)
        @test dd < 1e-3
        ad = mk()
        _, h, nr = step_adaptive!(ad, CFLConstraints(); sgs_stress = :symmetric)
        @test nr == 0 && h > 0
        # the anelastic core (walls in z): default == :laplacian bit for bit,
        # :symmetric runs
        ga = RegularGrid(; size = (16, 16, 12), extent = (800.0, 800.0, 600.0), halo = (3, 3, 3))
        bcs = FieldBoundaryConditions(:periodic, :periodic, :wall)
        rng = Xoshiro(17506)
        U = (0.3 .* randn(rng, 16, 16, 12), 0.3 .* randn(rng, 16, 16, 12), 0.3 .* randn(rng, 16, 16, 12))
        function mka()
            s = AnelasticSimulation(ga; bcs = bcs, C_s = 0.17)
            iv(s.u) .= U[1]; iv(s.v) .= U[2]; iv(s.w) .= U[3]; iv(s.w)[:, :, 1] .= 0
            return s
        end
        a1 = mka(); a2 = mka(); a3 = mka()
        for _ in 1:3
            step!(a1, 0.5); step!(a2, 0.5; sgs_stress = :laplacian); step!(a3, 0.5; sgs_stress = :symmetric)
        end
        @test st(a1) == st(a2)
        @test st(a1) != st(a3) && all(isfinite, iv(a3.u))
        @test_throws ArgumentError step!(mka(), 0.5; sgs_stress = :bogus)
    end

    @testset "explicit CFL limit of the stress form (power iteration)" begin
        g = grid175((12, 12, 8))
        S = 1 / g.dx^2 + 1 / g.dy^2 + 1 / g.dz^2
        Δ²harm = 1 / S
        sim = BoussinesqSimulation(g)           # periodic; ν = 0, so dt_diff = dt_sgs
        # C_diffusive = 1/2 puts the limit *on* the forward-Euler boundary
        # dt ρ = 2, so a limit is right exactly when dt ρ ≤ 2 there
        k = CFLConstraints(; C_diffusive = 0.5)
        rng = Xoshiro(17508)
        νfields = (
            "constant 1"            => (_, _, _) -> 1.0,
            "log-uniform 0.01–1"    => (_, _, _) -> 10.0^(-2 * rand(rng)),
            "checkerboard 0.01 / 1" => (i, j, l) -> isodd(i + j + l) ? 1.0 : 0.01,
            "blob, 100×"            => (i, j, l) -> 0.01 + exp(-((i - 6.5)^2 + (j - 6.5)^2 + 4 * (l - 4.5)^2) / 4))
        for (name, fν) in νfields
            ν = Field(CellCenter(), g)
            iv(ν) .= [fν(i, j, l) for i in 1:g.Nx, j in 1:g.Ny, l in 1:g.Nz]
            pfill!(ν)
            νmax = maxabs(ν)
            c = GivenNu175(ν)
            d_old = cfl_diagnostics(sim, k; sgs = c)                       # :laplacian
            d_new = cfl_diagnostics(sim, k; sgs = c, sgs_stress = :symmetric)
            @test d_old.νt_max == d_new.νt_max == νmax
            @test d_old.νt_cfl == νmax                                       # unchanged
            @test d_new.νt_cfl ≈ 2νmax                                       # νh = νv
            @test d_new.dt_sgs ≈ d_old.dt_sgs / 2
            @test d_old.dt_sgs == 0.5 * Δ²harm / νmax                        # old formula, bit for bit
            ρ  = spectral_radius175(sim, ν, ν, :symmetric)
            ρL = spectral_radius175(sim, ν, ν, :laplacian)
            ρP = spectral_radius175(sim, ν, ν, :symmetric; project = true)
            @info @sprintf("P55-b CFL [%s]: ρ(L_sym) = %.5f, ρ(L_lap) = %.5f, ρ(P L_sym P) = %.5f s⁻¹; 4νmaxΣΔ⁻² = %.5f; dt_old·ρ = %.3f, dt_new·ρ = %.3f (Euler bound 2)",
                           name, ρ, ρL, ρP, 4νmax * S, d_old.dt_sgs * ρ, d_new.dt_sgs * ρ)
            # the derived bound holds and the new limit is stable for L_sym
            @test ρ <= 4 * d_new.νt_cfl * S * (1 + 1e-12)
            @test d_new.dt_sgs * ρ <= 2 * (1 + 1e-12)
            # the old (Laplacian) limit is beyond the Euler boundary for L_sym
            @test d_old.dt_sgs * ρ > 2
            # ρ(L_sym) is twice ρ(L_lap) here: the grid-scale gradient modes,
            # on which the stress form adds ν∇(∇·u) ≈ ν∇² again
            @test 1.9 < ρ / ρL < 2 + 1e-9
            # on the projected (solenoidal) subspace the old bound suffices:
            # Σ 2S:S = Σ|∇u|² there, so ρ(P L P) ≤ 4νmax ΣΔ⁻²
            @test ρP <= 4νmax * S * (1 + 1e-12)
            @test d_old.dt_sgs * ρP <= 2 * (1 + 1e-12)
        end
        # sharpness: with constant ν the bound is attained, ρ = 8νΣΔ⁻²
        # (the checkerboard gradient mode)
        ν1 = constfield175(g, 1.0)
        @test spectral_radius175(sim, ν1, ν1, :symmetric; iters = 3000) ≈ 8S rtol = 1e-6
        # time stepping, unprojected: from the leading mode, 20 Euler steps
        # at the new limit do not grow, at the old limit grow by |1 − 4|²⁰
        c1 = GivenNu175(ν1)
        dtn = cfl_diagnostics(sim, k; sgs = c1, sgs_stress = :symmetric).dt_sgs
        dto = cfl_diagnostics(sim, k; sgs = c1).dt_sgs
        lead = (copy(sim.u.data), copy(sim.v.data), copy(sim.w.data))
        setlead!() = (copyto!(sim.u.data, lead[1]); copyto!(sim.v.data, lead[2]);
                      copyto!(sim.w.data, lead[3]))
        gn = euler_growth175(sim, ν1, dtn, 20)
        setlead!(); go = euler_growth175(sim, ν1, dto, 20)
        # projected (the cores' predictor + projection), from the leading
        # mode of P L P: no growth at the old limit
        spectral_radius175(sim, ν1, ν1, :symmetric; project = true)
        gp = euler_growth175(sim, ν1, dto, 20; project = true)
        @info @sprintf("P55-b CFL, constant ν, 20 Euler steps from the leading mode: growth %.3g at the new limit, %.3g at the old limit; %.3g at the old limit with the projection (leading mode of P L P)",
                       gn, go, gp)
        @test gn <= 1 + 1e-6
        @test go > 1e6
        @test gp <= 1 + 1e-6
        # anisotropic: random νh ∈ [0, 1], νv ∈ [0, 0.05]
        νh = random_ν175(g, Xoshiro(17509); hi = 1.0)
        νv = random_ν175(g, Xoshiro(17510); hi = 0.05)
        νc = stress_form_cfl_viscosity(maxabs(νh), maxabs(νv), g)
        ρa = spectral_radius175(sim, νh, νv, :symmetric)
        @info @sprintf("P55-b CFL anisotropic: ρ = %.5f, 4 ν_cfl ΣΔ⁻² = %.5f (ν_cfl = %.4f, 2 max ν = %.4f)",
                       ρa, 4νc * S, νc, 2 * max(maxabs(νh), maxabs(νv)))
        @test ρa <= 4νc * S * (1 + 1e-12)
        @test νc < 2 * max(maxabs(νh), maxabs(νv))          # tighter than 2 max(νh, νv)
        # adaptive_dt and imex_stability_limit pass the form through
        sim2 = BoussinesqSimulation(g; ν = 1e-5)
        cν = GivenNu175(ν1)
        k2 = CFLConstraints(; C_advective = 1e9)              # diffusion binds
        @test adaptive_dt(sim2, k2; sgs = cν, sgs_stress = :symmetric) ≈
              adaptive_dt(sim2, k2; sgs = cν) * (1e-5 + 1) / (1e-5 + 2)
        @test imex_stability_limit(sim2, ARK2(); constraints = k2, sgs = cν,
                                   sgs_stress = :symmetric).diagnostics.νt_cfl ≈ 2
    end
end
