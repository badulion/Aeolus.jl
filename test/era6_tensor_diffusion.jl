# Era VI, P55-c (#181): the tensorial scalar eddy diffusivity.
#
#   * diagonal (κh, κv): a Fourier mode decays at κh(k̂ₓ² + k̂_y²) + κv k̂_z²,
#     explicit and IMEX, with κh/κv = 10³;

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
end
