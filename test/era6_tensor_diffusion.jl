# Era VI, P55-c (#181): the tensorial scalar eddy diffusivity.
#
#   * diagonal (κh, κv): a Fourier mode decays at κh(k̂ₓ² + k̂_y²) + κv k̂_z²,
#     explicit and IMEX, with κh/κv = 10³;
#   * general symmetric K (triads, Griffies et al. 1998): the plane-wave
#     eigenvalue for constant K, conservation, the down-gradient property
#     for random positive semi-definite K (one step and 200 steps), symmetry,
#     reduction of diag(κh, κh, κv) to the diagonal operator, walls;

using Test
using Aeolus
using Aeolus.Fields: interior_view, Field
using Aeolus.Grids: interior
using Random: Xoshiro, rand, randn
using Printf

const TI181 = Aeolus.TimeIntegration

# ---- helpers ---------------------------------------------------------------

const DX181 = (15.625, 15.625, 6.25)          # the Berlin spacing (m)

grid181(N = (16, 16, 8); d = DX181, halo = (2, 2, 2)) =
    RegularGrid(; size = N, extent = N .* d, halo = halo)

const PBC181 = FieldBoundaryConditions(:periodic)
pfill181!(fs...) = foreach(f -> fill_halos!(f, PBC181), fs)

iv181(f) = interior_view(f)
maxabs181(f::Field) = maximum(abs, iv181(f))
maxabs181(a::AbstractArray) = maximum(abs, a)
constfield181(g, c) = (f = Field(CellCenter(), g); fill!(f.data, c); f)

# cell-centre coordinates of the storage index (halo included)
xc181(g, i) = (i - g.Hx - 0.5) * g.dx
yc181(g, j) = (j - g.Hy - 0.5) * g.dy
zc181(g, k) = (k - g.Hz - 0.5) * g.dz

function setfield181!(f, g, fn)
    for k in axes(f.data, 3), j in axes(f.data, 2), i in axes(f.data, 1)
        f.data[i, j, k] = fn(xc181(g, i), yc181(g, j), zc181(g, k))
    end
    return f
end

# k̂² = (2 sin(kΔ/2)/Δ)², the eigenvalue of the 3-point second difference
khat2_181(k, Δ) = (2 * sin(k * Δ / 2) / Δ)^2

# A diagonal closure with constant κh, κv (ν = 0): the vertical
# diffusivity lives in the closure's own field, so f.κv !== f.κh.
struct ConstDiag181{F} <: AbstractSGSClosure
    κh::Float64
    κv::Float64
    fv::F
end
ConstDiag181(sim, κh, κv) = ConstDiag181(κh, κv, Field(CellCenter(), sim.mesh))
Aeolus.Operators.sgs_fields(c::ConstDiag181, sim) = SGSFields(sim.νt, sim.νt, sim.κt, c.fv)
function Aeolus.Operators.sgs_coefficients!(f::SGSFields, c::ConstDiag181, sim)
    fill!(f.νh.data, 0); fill!(f.κh.data, c.κh); fill!(f.κv.data, c.κv)
    return f
end


# A random positive semi-definite tensor per cell, K = A Aᵀ (scale ~ s),
# with a vertical squeeze (K33 ~ s·v²) so that it is strongly anisotropic.
function random_psd181(g, rng; s = 1.0, v = 0.3, bcs = PBC181)
    K = DiffusivityTensor(g)
    for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        A = randn(rng, 3, 3) .* sqrt(s / 3)
        A[3, :] .*= v
        M = A * A'
        I, J, Kk = i + g.Hx, j + g.Hy, k + g.Hz
        K.K11.data[I, J, Kk] = M[1, 1]; K.K22.data[I, J, Kk] = M[2, 2]
        K.K33.data[I, J, Kk] = M[3, 3]; K.K12.data[I, J, Kk] = M[1, 2]
        K.K13.data[I, J, Kk] = M[1, 3]; K.K23.data[I, J, Kk] = M[2, 3]
    end
    fill_halos!(K, bcs)
    return K
end

function random_scalar181(g, rng; bcs = PBC181)
    ϕ = Field(CellCenter(), g)
    iv181(ϕ) .= randn(rng, g.Nx, g.Ny, g.Nz)
    fill_halos!(ϕ, bcs)
    return ϕ
end

tensor_tend181(ϕ, K; kw...) = (t = Field(CellCenter(), ϕ.mesh);
                                add_tensor_diffusion_scalar!(t, ϕ, K; kw...); t)

@testset "Era VI — tensorial scalar eddy diffusivity (P55-c, #181)" begin

    @testset "diagonal (κh, κv): decay at κh(k̂ₓ² + k̂_y²) + κv k̂_z², κh/κv = 10³" begin
        g = grid181((16, 8, 16))
        kx = 2π * 2 / g.Lx; ky = 2π * 1 / g.Ly; kz = 2π * 3 / g.Lz
        κh, κv = 2.0, 2.0e-3
        mode(x, y, z) = sin(kx * x + 0.4) * cos(ky * y - 0.2) * sin(kz * z + 0.1)
        λ = κh * (khat2_181(kx, g.dx) + khat2_181(ky, g.dy)) + κv * khat2_181(kz, g.dz)

        # the operator: tendency = −λ × the mode
        ϕ = setfield181!(Field(CellCenter(), g), g, mode)
        t = Field(CellCenter(), g)
        add_eddy_diffusion_scalar!(t, ϕ, constfield181(g, κh), constfield181(g, κv))
        e_op = maxabs181(iv181(t) .+ λ .* iv181(ϕ)) / (λ * maxabs181(ϕ))
        λz = κv * khat2_181(kz, g.dz)
        @test e_op <= 1e-10
        # κv's share of the mixed mode's λ is < 1 %, so check the z faces on
        # their own: a horizontally uniform mode decays at κv k̂_z² (an
        # operator with κh on the z faces would be 10³× too fast)
        ϕz = setfield181!(Field(CellCenter(), g), g, (x, y, z) -> cos(kz * z - 0.3))
        tz = Field(CellCenter(), g)
        add_eddy_diffusion_scalar!(tz, ϕz, constfield181(g, κh), constfield181(g, κv))
        e_z = maxabs181(iv181(tz) .+ λz .* iv181(ϕz)) / (λz * maxabs181(ϕz))
        @test e_z <= 1e-10
        # κh = κv: the isotropic three-argument form, bit for bit
        t3 = Field(CellCenter(), g); t4 = Field(CellCenter(), g)
        K = constfield181(g, 0.7)
        add_eddy_diffusion_scalar!(t3, ϕ, K)
        add_eddy_diffusion_scalar!(t4, ϕ, K, K)
        @test t3.data == t4.data

        # time stepping through the closure contract: θ decays at the discrete
        # amplification factor of λ
        function decay_run(stepper, h, n)
            sim = BoussinesqSimulation(g; ν = 0.0)          # θ₀ = 0: no buoyancy
            setfield181!(sim.θ, g, mode)
            θ0 = copy(iv181(sim.θ))
            c = ConstDiag181(sim, κh, κv)
            for _ in 1:n
                stepper(sim, h, c)
            end
            return θ0, copy(iv181(sim.θ)), sim
        end
        hex = 0.05 / λ                       # h·λ_max ≈ 0.7 for the grid-scale modes
        θ0, θ1, sim = decay_run((s, h, c) -> step!(s, h; sgs = c), hex, 40)
        G = (1 - hex * λ)^40
        e_ex = maximum(abs, θ1 .- G .* θ0) / (G * maximum(abs, θ0))
        @test e_ex <= 1e-10

        γ = 1 - 1 / sqrt(2)
        him = 5.0 / λ                        # hλ = 5; the grid-scale modes are at h·λ_max ≈ 70
        θ0, θ1, sim = decay_run((s, h, c) -> step_imex!(s, h, ARK2(); sgs = c,
                                                         tol = 1e-14, maxit = 500), him, 4)
        a = him * λ
        r1 = 1 / (1 + γ * a)
        Rimex = (1 - (1 - γ) * a * r1) / (1 + γ * a)
        Gim = Rimex^4
        e_im = maximum(abs, θ1 .- Gim .* θ0) / (abs(Gim) * maximum(abs, θ0))
        @info @sprintf("P55-c diagonal decay (κh/κv = 10³): λ = %.6e s⁻¹ (κv share %.1f %%); operator rel. err %.1e (vertical mode %.1e); Euler hλ=0.05 ×40 %.1e; ARK2 hλ=5 ×4 %.1e",
                       λ, 100λz / λ, e_op, e_z, e_ex, e_im)
        @test e_im <= 1e-10
    end
    @testset "general K: plane-wave eigenvalue for constant K" begin
        g = grid181((16, 12, 16))
        kv = (2π * 3 / g.Lx, 2π * 2 / g.Ly, 2π * 5 / g.Lz)
        K11, K22, K33 = 2.0, 1.5, 0.4
        K12, K13, K23 = 0.3, -0.6, 0.25                  # positive definite
        # Sylvester's criterion: positive definite
        @test K11 > 0 && K11 * K22 - K12^2 > 0 &&
              K11 * (K22 * K33 - K23^2) - K12 * (K12 * K33 - K23 * K13) + K13 * (K12 * K23 - K22 * K13) > 0
        K = DiffusivityTensor(g)
        for (c, x) in zip((K.K11, K.K22, K.K33, K.K12, K.K13, K.K23), (K11, K22, K33, K12, K13, K23))
            fill!(c.data, x)
        end
        ϕ = setfield181!(Field(CellCenter(), g), g,
                         (x, y, z) -> cos(kv[1] * x + kv[2] * y + kv[3] * z + 0.3))
        kh = khat2_181.(kv, (g.dx, g.dy, g.dz))
        sn = sin.(kv .* (g.dx, g.dy, g.dz)) ./ (g.dx, g.dy, g.dz)
        λ = K11 * kh[1] + K22 * kh[2] + K33 * kh[3] +
            2 * (K12 * sn[1] * sn[2] + K13 * sn[1] * sn[3] + K23 * sn[2] * sn[3])
        t = tensor_tend181(ϕ, K)
        e = maxabs181(iv181(t) .+ λ .* iv181(ϕ)) / (λ * maxabs181(ϕ))
        # the cross terms carry a sizeable share of λ
        λx = 2 * (K12 * sn[1] * sn[2] + K13 * sn[1] * sn[3] + K23 * sn[2] * sn[3])
        @info @sprintf("P55-c constant-K plane wave: λ = %.6e s⁻¹ (cross terms %.6e); rel. err %.1e", λ, λx, e)
        @test abs(λx) / λ > 0.05
        @test e <= 1e-10
    end

    @testset "general K: conservation, down-gradient, symmetry (random PSD K)" begin
        g = grid181((12, 10, 8))
        ΔV = g.dx * g.dy * g.dz
        wall = FieldBoundaryConditions(:periodic, :periodic, :noflux)
        for (bcs, label) in ((PBC181, "periodic"), (wall, "no-flux z walls"))
            bnd = bounded_directions(bcs)
            for seed in 1:3
                rng = Xoshiro(1810 + seed)
                K = random_psd181(g, rng; bcs = bcs)
                ϕ = random_scalar181(g, rng; bcs = bcs)
                ψ = random_scalar181(g, rng; bcs = bcs)
                tϕ = tensor_tend181(ϕ, K; bounded = bnd)
                tψ = tensor_tend181(ψ, K; bounded = bnd)
                scale = sum(abs, iv181(tϕ)) * ΔV
                cons = abs(sum(iv181(tϕ)) * ΔV) / scale
                dvar = sum(iv181(ϕ) .* iv181(tϕ)) * ΔV
                sym = abs(sum(iv181(ψ) .* iv181(tϕ)) - sum(iv181(ϕ) .* iv181(tψ))) /
                      (sum(abs, iv181(ψ) .* iv181(tϕ)))
                seed == 1 && @info @sprintf("P55-c random PSD K (%s): |Σt ΔV|/Σ|t|ΔV = %.1e; Σϕ t ΔV = %.4e; symmetry %.1e",
                                            label, cons, dvar, sym)
                @test cons <= 1e-13
                @test dvar < 0
                @test sym <= 1e-12
            end
        end
        # a non-PSD K (negative K33 in every cell) can make variance grow — the
        # property is not automatic, it is the PSD condition
        rng = Xoshiro(18100)
        K = random_psd181(g, rng); K.K33.data .-= 10
        ϕ = setfield181!(Field(CellCenter(), g), g, (x, y, z) -> cos(2π * z / g.Lz))
        @test sum(iv181(ϕ) .* iv181(tensor_tend181(ϕ, K))) > 0
    end

    @testset "general K: 200 explicit steps, Σϕ dV conserved and Σϕ² non-increasing" begin
        g = grid181((12, 10, 8))
        ΔV = g.dx * g.dy * g.dz
        wall = FieldBoundaryConditions(:periodic, :periodic, :noflux)
        rng = Xoshiro(18101)
        K = random_psd181(g, rng; bcs = wall)
        ϕ = random_scalar181(g, rng; bcs = wall)
        Kmax = maximum(maximum(abs, iv181(c)) for c in (K.K11, K.K22, K.K33))
        h = 0.1 * min(g.dx, g.dy, g.dz)^2 / Kmax      # well inside the explicit limit
        bnd = bounded_directions(wall)
        m0 = sum(iv181(ϕ)) * ΔV
        v = sum(abs2, iv181(ϕ)) * ΔV
        v0 = v
        nonincr = true
        t = Field(CellCenter(), g)
        for n in 1:200
            fill!(t.data, 0)
            add_tensor_diffusion_scalar!(t, ϕ, K; bounded = bnd)
            iv181(ϕ) .+= h .* iv181(t)
            fill_halos!(ϕ, wall)
            vn = sum(abs2, iv181(ϕ)) * ΔV
            nonincr &= vn <= v
            v = vn
        end
        dm = abs(sum(iv181(ϕ)) * ΔV - m0) / (sum(abs, iv181(ϕ)) * ΔV)
        @info @sprintf("P55-c 200 steps (random PSD K, z walls): Σϕ² %.4e → %.4e; |ΔΣϕ dV|/Σ|ϕ|dV = %.1e",
                       v0, v, dm)
        @test nonincr
        @test v < v0
        @test dm <= 1e-13
    end

    @testset "reduction: K = diag(κh, κh, κv) is the diagonal operator" begin
        g = grid181((12, 10, 8))
        wall = FieldBoundaryConditions(:periodic, :periodic, :noflux)
        for bcs in (PBC181, wall)
            rng = Xoshiro(18102)
            κh = Field(CellCenter(), g); κv = Field(CellCenter(), g)
            iv181(κh) .= 2 .* rand(rng, g.Nx, g.Ny, g.Nz)
            iv181(κv) .= 0.02 .* rand(rng, g.Nx, g.Ny, g.Nz)
            fill_halos!(κh, bcs); fill_halos!(κv, bcs)
            ϕ = random_scalar181(g, rng; bcs = bcs)
            K = diagonal_tensor!(DiffusivityTensor(g), κh, κv)
            td = Field(CellCenter(), g)
            add_eddy_diffusion_scalar!(td, ϕ, κh, κv)
            bnd = bounded_directions(bcs)
            tf = tensor_tend181(ϕ, K; bounded = bnd)
            tdi = tensor_tend181(ϕ, K; bounded = bnd, part = :diagonal)
            to = tensor_tend181(ϕ, K; bounded = bnd, part = :offdiagonal)
            e = maxabs181(iv181(tf) .- iv181(td)) / maxabs181(td)
            @test e <= 1e-12
            @test maxabs181(iv181(tdi) .- iv181(td)) <= 1e-12 * maxabs181(td)
            @test maxabs181(to) == 0
        end
        # :diagonal + :offdiagonal == :full for a general K
        rng = Xoshiro(18103)
        K = random_psd181(g, rng); ϕ = random_scalar181(g, rng)
        tf = tensor_tend181(ϕ, K)
        ts = tensor_tend181(ϕ, K; part = :diagonal)
        add_tensor_diffusion_scalar!(ts, ϕ, K; part = :offdiagonal)
        @test maxabs181(iv181(ts) .- iv181(tf)) <= 1e-13 * maxabs181(tf)
        @test_throws ArgumentError add_tensor_diffusion_scalar!(ts, ϕ, K; part = :bogus)
    end
end
