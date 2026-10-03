# Era VI, Phase 57 (P57-e, #190): the dynamic wall model for partially
# resolved roughness — z0 = α σ_Δ with α from the invariance of the total
# surface drag (modelled stress + resolved form drag) between the grid and
# the test-filter scale (Anderson & Meneveau 2011, JFM 679).
#
# Physics asserted:
#   1. analytic invariance — a flat surface returns the prescribed
#      fallback z0 exactly; on a sinusoidal surface with a uniform log-law
#      wind every term of the drag balance has a closed form (block means
#      of a sinusoid, the box / spectral transfer functions, central
#      differences), and the solver reproduces the α of that closed-form
#      balance to 1e-8; the test-filter variance is the exact variance of
#      the fine surface over the composite kernel;
#   2. grid invariance (reduced) — on a neutral rough half channel over a
#      random k⁻² surface, the plane-mean u_* the model predicts at 32²
#      and 16² (a priori, on the 32² LES wind) agrees within 5 % with the
#      dynamic α and disagrees by more with α fixed at its 32² value; the
#      half channel's plane-mean momentum budget, with the form drag of
#      the resolved relief, closes to round-off.

using Test
using Printf
using Random
using Statistics
using FFTW: ifft
using Aeolus
using Aeolus.WallModels

const _WM = Aeolus.WallModels

# A periodic random surface on an Nf² map of side L whose radial spectrum
# is E(k) ∝ k^-2 for 1 ≤ k ≤ kmax (in units of 2π/L), rms σh, zero mean.
function _k2_surface(Nf, L, σh; seed = 1, kmax = Nf ÷ 2)
    rng = Xoshiro(seed)
    Hk = zeros(ComplexF64, Nf, Nf)
    for b in 1:Nf, a in 1:Nf
        kx = a - 1 <= Nf ÷ 2 ? a - 1 : a - 1 - Nf
        ky = b - 1 <= Nf ÷ 2 ? b - 1 : b - 1 - Nf
        k = hypot(kx, ky)
        (1 <= k <= kmax) || continue
        Hk[a, b] = k^-1.5 * cis(2π * rand(rng))      # |ĥ|² ∝ k⁻³ per mode → E(k) ∝ k⁻²
    end
    h = real(ifft(Hk))
    h .-= mean(h)
    return h .* (σh / std(h))
end

_blockmean(A, r) = [mean(@view A[(i-1)*r+1:i*r, (j-1)*r+1:j*r])
                    for i in 1:size(A, 1)÷r, j in 1:size(A, 2)÷r]

@testset "Era VI — dynamic roughness of partially resolved relief (#190)" begin

@testset "Surface-gradient drag: windward faces only, quadratic in the wind" begin
    @test surface_gradient_drag(3.0, 1.0, 0.0, 0.0) == (0.0, 0.0)          # flat
    @test surface_gradient_drag(3.0, 0.0, -0.2, 0.0) == (0.0, 0.0)         # leeward
    fx, fy = surface_gradient_drag(3.0, 0.0, 0.2, 0.1; C_d = 2.0)
    @test fx ≈ -2.0 * (0.6)^2 * 0.2 rtol = 1e-14
    @test fy ≈ -2.0 * 0.36 * 0.1 rtol = 1e-14
    @test surface_gradient_drag(6.0, 0.0, 0.2, 0.1; C_d = 2.0)[1] ≈ 4fx rtol = 1e-14
end

@testset "Sub-filter statistics: composite-kernel variance, flat surface" begin
    Nx, Ny, r = 8, 6, 4
    dx, dy = 1.0, 1.5
    rng = Xoshiro(3)
    hf = randn(rng, Nx * r, Ny * r)
    s = subfilter_height(hf, r; dx = dx, dy = dy, filter = :box)
    @test s.h ≈ _blockmean(hf, r) rtol = 1e-14
    # σ_Δ² is the variance of the fine height inside each cell
    @test s.σ2 ≈ [var(vec(hf[(i-1)*r+1:i*r, (j-1)*r+1:j*r]); corrected = false) for i in 1:Nx, j in 1:Ny] rtol = 1e-12
    # σ_Δ̂² is the variance of the fine height about h_Δ̂ over the 3 × 3 cells
    # around (periodic) — the composite kernel of block mean and box filter
    direct = similar(s.σ2)
    for j in 1:Ny, i in 1:Nx
        vals = [hf[(mod1(i + a, Nx) - 1) * r + p, (mod1(j + b, Ny) - 1) * r + q]
                for a in -1:1, b in -1:1, p in 1:r, q in 1:r]
        direct[i, j] = mean((vals .- s.ht[i, j]) .^ 2)
    end
    @test maximum(abs.(s.σ2t .- direct)) < 1e-12
    @test mean(s.ht) ≈ mean(s.h) rtol = 1e-12                          # the test filter keeps the mean
    # a coarser filter sees more sub-filter relief, on average
    @test mean(s.σ2t) > mean(s.σ2)
    flat = subfilter_height(fill(2.5, Nx * r, Ny * r), r; dx = dx, dy = dy)
    @test all(==(0), flat.σ2) && all(==(0), flat.σ2t) && all(==(0), flat.dhx)
end

@testset "Analytic invariance: flat surface returns the fallback z0 exactly" begin
    Nx, Ny, r = 16, 4, 4
    for filter in (:box, :spectral)
        st = subfilter_height(zeros(Nx * r, Ny * r), r; dx = 0.3, dy = 0.3, filter = filter)
        dr = DynamicRoughness(st; z0_floor = 0.013, α_fallback = 0.0)
        uc = fill(4.0, Nx, Ny); vc = fill(-1.0, Nx, Ny)
        update_dynamic_roughness!(dr, uc, vc, 0.5)
        @test dr.z0 == fill(0.013, Nx, Ny)                    # exactly, no NaN from σ = 0
        @test !dr.converged && dr.α == 0.0 && dr.nfail == 1
        @test all(isfinite, dr.z0)
        # with a nonzero fallback α the flat surface still has z0 = z0_floor
        dr2 = DynamicRoughness(st; z0_floor = 0.013, α_fallback = 0.7)
        update_dynamic_roughness!(dr2, uc, vc, 0.5)
        @test dr2.z0 == fill(0.013, Nx, Ny)
        laws = dynamic_wall_laws(dr, RoughLogLaw(0.1))
        @test all(l -> l.z0 == 0.013, laws)
    end
end

@testset "Analytic invariance: sinusoidal surface, log-law wind → α to 1e-8" begin
    # h = A₁ sin(k₁x) + A₂ sin(k₂x) sampled on an r-times finer map; the LES
    # grid sees the r-point block means. Closed forms, with s_m the offsets
    # of the fine points in a cell and D(q) = ⟨cos q s_m⟩:
    #   m₁ = Σ Aₙ D(kₙ) sin(kₙx),
    #   m₂ = Σ Aₙ²/2 (1 − D(2kₙ) cos 2kₙx)
    #        + A₁A₂ [D(k₁−k₂) cos (k₁−k₂)x − D(k₁+k₂) cos (k₁+k₂)x];
    # the test filter multiplies a mode of wavenumber q by T(q) =
    # (1 + 2 cos qΔ)/3 (box) or 1 for |q| ≤ π/(2Δ), else 0 (spectral);
    # central differences: ∂ₓ sin(kx) → cos(kx) sin(kΔ)/Δ. k₁ = 3 is kept
    # by both test filters, k₂ = 10 removed by the spectral one.
    Nx, Ny, r = 32, 4, 4
    Lx = 2π; dx = Lx / Nx; dy = dx
    A1, A2, k1, k2 = 0.02, 0.006, 3.0, 10.0
    xf = [(i - 0.5) * dx / r for i in 1:Nx*r]
    hf = [A1 * sin(k1 * x) + A2 * sin(k2 * x) for x in xf, _ in 1:Ny*r]
    xc = [(i - 0.5) * dx for i in 1:Nx]
    offs = [((m - 0.5) / r - 0.5) * dx for m in 1:r]
    D(q) = mean(cos.(q .* offs))
    U0, z, z0f, αs = 3.0, 0.05, 1e-5, 0.3
    κ = Aeolus.SurfaceExchange.von_karman()
    for filter in (:box, :spectral)
        kc = π / (2dx)
        T(q) = filter === :box ? (1 + 2cos(q * dx)) / 3 : (abs(q) <= kc ? 1.0 : 0.0)
        I(q) = 1.0                                  # the unfiltered transfer
        M1(F) = A1 * D(k1) * F(k1) .* sin.(k1 .* xc) .+ A2 * D(k2) * F(k2) .* sin.(k2 .* xc)
        M2(F) = A1^2 / 2 .* (1 .- D(2k1) * F(2k1) .* cos.(2k1 .* xc)) .+
                A2^2 / 2 .* (1 .- D(2k2) * F(2k2) .* cos.(2k2 .* xc)) .+
                A1 * A2 .* (D(k1 - k2) * F(k1 - k2) .* cos.((k1 - k2) .* xc) .-
                            D(k1 + k2) * F(k1 + k2) .* cos.((k1 + k2) .* xc))
        G(F) = A1 * D(k1) * F(k1) * sin(k1 * dx) / dx .* cos.(k1 .* xc) .+
               A2 * D(k2) * F(k2) * sin(k2 * dx) / dx .* cos.(k2 .* xc)
        m1 = M1(I); ht = M1(T)
        σ2 = M2(I) .- m1 .^ 2
        σ2t = max.(M2(T) .- ht .^ 2, 0)
        g1 = G(I); g2 = G(T)
        st = subfilter_height(hf, r; dx = dx, dy = dy, filter = filter)
        @test maximum(abs.(st.h[:, 2] .- m1)) < 1e-15
        @test maximum(abs.(st.σ2[:, 2] .- σ2)) < 1e-15
        @test maximum(abs.(st.σ2t[:, 2] .- σ2t)) < 1e-15
        @test maximum(abs.(st.dhx[:, 2] .- g1)) < 1e-14
        @test maximum(abs.(st.dhxt[:, 2] .- g2)) < 1e-14
        # the closed-form balance D_Δ̂(α*) = D_Δ(α*) fixes C_d for α* = 0.3
        W(σ) = mean(κ^2 * U0^2 ./ log.(z ./ sqrt.(z0f^2 .+ αs^2 .* σ)) .^ 2)
        Rp(g) = mean(U0^2 .* g .^ 3 .* (g .> 0))
        C_d = (W(σ2t) - W(σ2)) / (Rp(g1) - Rp(g2))
        @test C_d > 0
        dr = DynamicRoughness(st; z0_floor = z0f, C_d = C_d)
        uc = fill(U0, Nx, Ny); vc = zeros(Nx, Ny)
        α, conv = dynamic_alpha(dr, uc, vc, z)
        @info @sprintf("  sinusoid (%s test filter): C_d = %.4g, α = %.15f (target %.1f), |Δα| = %.1e",
                       filter, C_d, α, αs, abs(α - αs))
        @test conv
        @test abs(α - αs) < 1e-8
        # the residual is the drag mismatch between the two scales
        Fg = roughness_surface_drag(dr, uc, vc, z; α = α, scale = :grid)
        Ft = roughness_surface_drag(dr, uc, vc, z; α = α, scale = :test)
        @test roughness_drag_residual(dr, α, uc, vc, z) ≈ Ft.total - Fg.total atol = 1e-14
        @test abs(Ft.total - Fg.total) < 1e-12 * Fg.total
        @test Fg.form ≈ C_d * Rp(g1) rtol = 1e-12
        # α is the same for a wind of a different strength (every term ∝ U²)
        @test dynamic_alpha(dr, 2.5 .* uc, vc, z)[1] ≈ α rtol = 1e-9
        # z0 and the laws it provides
        update_dynamic_roughness!(dr, uc, vc, z)
        @test dr.z0[:, 1] ≈ sqrt.(z0f^2 .+ α^2 .* σ2) rtol = 1e-12
        L = dynamic_wall_laws(dr, MOSTWallLaw(0.1, 50.0; U_min = 0.2))
        @test all(L[I].z0m == dr.z0[I] && L[I].L == 50.0 && L[I].U_min == 0.2 for I in eachindex(L))
        laws = [RoughLogLaw(0.1) for _ in 1:Nx, _ in 1:Ny]
        dynamic_wall_laws!(laws, dr)
        @test all(laws[I].z0 == dr.z0[I] for I in eachindex(laws))
        # fixed α: the non-dynamic model
        @test dynamic_alpha(DynamicRoughness(st; z0_floor = z0f, α = 0.123), uc, vc, z) == (0.123, true)
    end
    # the residual has no root when the resolved relief makes no drag
    st = subfilter_height(hf, r; dx = dx, dy = dy)
    dr0 = DynamicRoughness(st; z0_floor = z0f, C_d = 0.0, α_fallback = 0.05)
    α0, c0 = dynamic_alpha(dr0, fill(U0, Nx, Ny), zeros(Nx, Ny), z)
    @test (α0, c0) == (0.05, false)
    @test_throws ArgumentError DynamicRoughness(st; z0_floor = 0.0)
    @test_throws ArgumentError dynamic_alpha(DynamicRoughness(st; z0_floor = 0.1),
                                             fill(U0, Nx, Ny), zeros(Nx, Ny), 0.05)
end

@testset "Grid invariance (reduced): rough half channel, 32² vs 16²" begin
    # A random k⁻² surface (256² map, 1 ≤ k ≤ 128, σ_h = 0.04 H), resolved
    # through the surface-gradient drag (C_d = 4), the rest sub-filter.
    # Half channel 2πH × 2πH × H, 32 × 32 × 8, StaticSmagorinsky(0.1), with
    # the dynamic roughness, run to 6 H/u*; at t = 4, 5, 6 the first-cell
    # wind is taken as is (32²) and block-averaged 2 × 2 (16²), and the
    # plane-mean u_* = √(wall + form drag) of the model is evaluated at both
    # resolutions with the dynamic α of each, and with α fixed at its 32²
    # value.
    Nf, L, H = 256, 2π, 1.0
    hf = _k2_surface(Nf, L, 0.04)
    C_d, z0f, Nz = 4.0, 1e-5, 8
    st32 = subfilter_height(hf, Nf ÷ 32; dx = L / 32, dy = L / 32)
    st16 = subfilter_height(hf, Nf ÷ 16; dx = L / 16, dy = L / 16)
    @test st32.filter === :spectral
    @test mean(st16.σ2) > mean(st32.σ2)            # more of the relief is sub-filter at 16²
    dr = DynamicRoughness(st32; z0_floor = z0f, C_d = C_d)
    case = half_channel_case(; N = (32, 32, Nz), roughness = dr, z0_over_H = 0.01)
    z = case.sim.mesh.dz / 2
    # the momentum budget with the form drag closes to round-off
    sim = case.sim; A = sim.mesh.Lx * sim.mesh.Ly
    um() = sum(Aeolus.Fields.interior_view(sim.u)) / (sim.mesh.Nx * sim.mesh.Ny * Nz)
    budget_err = 0.0
    t0 = time()
    run_channel!(case, 1.0)
    for _ in 1:5
        u0 = um(); dt = channel_dt(case)
        advance_channel!(case, dt)
        lhs = (um() - u0) * H / dt
        rhs = case.forcing * H - case.τ_wall_bottom - case.τ_form
        budget_err = max(budget_err, abs(lhs - rhs))
    end
    @info @sprintf("  half channel with form drag: plane-mean budget error %.1e (τ_w %.3f, τ_form %.3f u*²)",
                   budget_err, case.τ_wall_bottom, case.τ_form)
    @test budget_err < 1e-10
    @test case.τ_form > 0 && dr.converged
    run_channel!(case, 4.0)
    dr32 = DynamicRoughness(st32; z0_floor = z0f, C_d = C_d)
    dr16 = DynamicRoughness(st16; z0_floor = z0f, C_d = C_d)
    us = Dict(k => Float64[] for k in (:d32, :d16, :f16)); αs = Dict(k => Float64[] for k in (:a32, :a16))
    for t in (4.0, 5.0, 6.0)
        run_channel!(case, t)
        uc, vc = _WM._first_cell_velocity(case.sim)
        u16, v16 = _blockmean(uc, 2), _blockmean(vc, 2)
        a32, c32 = dynamic_alpha(dr32, uc, vc, z)
        a16, c16 = dynamic_alpha(dr16, u16, v16, z)
        @test c32 && c16
        push!(αs[:a32], a32); push!(αs[:a16], a16)
        push!(us[:d32], sqrt(roughness_surface_drag(dr32, uc, vc, z; α = a32).total))
        push!(us[:d16], sqrt(roughness_surface_drag(dr16, u16, v16, z; α = a16).total))
        push!(us[:f16], sqrt(roughness_surface_drag(dr16, u16, v16, z; α = a32).total))
    end
    u32, ud, uf = mean(us[:d32]), mean(us[:d16]), mean(us[:f16])
    ed, ef = ud / u32 - 1, uf / u32 - 1
    @info @sprintf("  grid invariance: α(32²) = %.3f, α(16²) = %.3f; u*(32²) = %.4f, u*(16²) dynamic = %.4f (%+.1f %%), fixed α = %.4f (%+.1f %%); %.1f s",
                   mean(αs[:a32]), mean(αs[:a16]), u32, ud, 100ed, uf, 100ef, time() - t0)
    @test abs(ed) < 0.05
    @test abs(ef) > abs(ed)
    @test mean(αs[:a16]) < mean(αs[:a32])          # α is scale dependent: the point of the dynamic model
end

end
