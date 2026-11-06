# Phase 48 — PlantCanopy (issue #23): radiation attenuation through
# the canopy. Beer–Lambert with the Ross–Goudriaan G-function, the Sellers
# (1985) two-stream solved with the TwoStream layer formulae and
# adding solver, sunlit/shaded partition, and the longwave exchange.

using Test
using Aeolus
using Aeolus.PlantCanopy
using Aeolus.RadiationCore: STEFAN_BOLTZMANN, _gauss_legendre

# ---- independent references used below ---------------------------------

# Exponential integral E₁ by its power series (x ≤ ~5 is all we need), and
# E₃ from the recurrence E₃ = ½[e^{-x}(1 - x) + x² E₁]. Independent of the
# module's quadrature of the same hemispheric integral.
function _E1(x)
    s = 0.0; term = 1.0
    for n in 1:200
        term *= -x / n
        s += term / n
        abs(term / n) < 1e-18 && break
    end
    return -Base.MathConstants.eulergamma - log(x) - s
end
_E3(x) = 0.5 * (exp(-x) * (1 - x) + x^2 * _E1(x))

# Sellers' (1985) canopy two-stream ODEs integrated directly in cumulative
# leaf area Λ (downward) with RK4, the boundary-value problem closed by
# superposition of one particular and one homogeneous solution. Shares
# nothing with the module except the optical parameters (ω, β, β₀, μ̄, K).
function _sellers_ode(; Λt, ω, β, β0, μ̄, K, S0, D0, αb, αd, n = 20_000)
    γ1 = (1 - (1 - β) * ω) / μ̄; γ2 = ω * β / μ̄
    f(L, y, src) = (γ1 * y[1] - γ2 * y[2] - src * ω * K * β0 * S0 * exp(-K * L),
                    -γ1 * y[2] + γ2 * y[1] + src * ω * K * (1 - β0) * S0 * exp(-K * L))
    function run(y0, src)
        h = Λt / n; y = y0; L = 0.0
        prof = Vector{NTuple{2,Float64}}(undef, n + 1); prof[1] = y
        for i in 1:n
            k1 = f(L, y, src)
            k2 = f(L + h/2, y .+ (h/2) .* k1, src)
            k3 = f(L + h/2, y .+ (h/2) .* k2, src)
            k4 = f(L + h, y .+ h .* k3, src)
            y = y .+ (h/6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4); L += h
            prof[i+1] = y
        end
        return prof
    end
    p = run((0.0, D0), 1.0)          # (I↑, I↓): I↓(0) = D0, I↑(0) = 0
    hm = run((1.0, 0.0), 0.0)        # homogeneous, I↑(0) = 1
    Sb = S0 * exp(-K * Λt)
    # I↑(Λt) = αd I↓(Λt) + αb S(Λt)
    c = (αd * p[end][2] + αb * Sb - p[end][1]) / (hm[end][1] - αd * hm[end][2])
    up = [p[i][1] + c * hm[i][1] for i in eachindex(p)]
    dn = [p[i][2] + c * hm[i][2] for i in eachindex(p)]
    return up, dn
end

@testset "PlantCanopy: radiation attenuation (#23)" begin

    par = LeafOptics(0.10, 0.05)     # typical broadleaf PAR
    nir = LeafOptics(0.45, 0.25)     # typical broadleaf NIR

    @testset "Ross–Goudriaan G-function, μ̄ and scattering parameters" begin
        sph = SphericalLeaves()
        @test beam_extinction_coefficient(sph, 0.3) ≈ 0.5 / 0.3
        @test average_inverse_optical_depth(sph) ≈ 1.0 rtol = 1e-14
        # μ̄ closed form (and its series branch) vs quadrature of ∫₀¹ μ/G dμ;
        # a_s closed form (inside β₀) vs quadrature of its integral
        x, w = _gauss_legendre(200)
        worst_μ̄ = 0.0; worst_as = 0.0
        for χ in (-0.4, -1e-5, 2e-4, 0.6)
            d = RossGoudriaanLeaves(χ)
            ref = 0.0; lo = 0.0
            for m in 20:-1:0
                hi = 2.0^(-m)
                ref += sum(w[q] * (hi - lo) / 2 *
                           (μ -> μ / leaf_projection_function(d, μ))((hi + lo) / 2 + (hi - lo) / 2 * x[q])
                           for q in eachindex(x))
                lo = hi
            end
            μ̄ = average_inverse_optical_depth(d)
            worst_μ̄ = max(worst_μ̄, abs(μ̄ / ref - 1))
            # χ_L = −0.4: μ* = φ₁/(2|φ₂|) is the 0/0 point of a_s (#23 bug: NaN)
            for μ in (0.1, 0.5, χ == -0.4 ? -d.φ1 / (2d.φ2) : 0.9)
                ω, β, β0 = canopy_scattering_parameters(d, par, μ)
                G = leaf_projection_function(d, μ); K = G / μ
                a_s = (ω / 2) * sum(w[q] / 2 * (m -> m * G / (μ * leaf_projection_function(d, m) + m * G))((x[q] + 1) / 2)
                                    for q in eachindex(x))
                worst_as = max(worst_as, abs(β0 / ((1 + μ̄ * K) / (μ̄ * K) * a_s / ω) - 1))
            end
        end
        @test worst_μ̄ < 1e-10
        @test worst_as < 1e-8
        # horizontal leaves scatter α up and τ down exactly
        ωh, βh, β0h = canopy_scattering_parameters(HorizontalLeaves(), par, 0.3)
        @test ωh ≈ 0.15 && βh ≈ 2 / 3 && β0h ≈ 2 / 3
    end

    @testset "Beer–Lambert: direct beam and black-leaf diffuse" begin
        forest = LalicMihailovicLAD(5.0, 20.0; z_m = 13.0)
        zf = collect(0.0:1.0:25.0)
        leaves = RossGoudriaanLeaves(-0.3)
        s = canopy_shortwave(forest, zf; sun = 0.6, sw_direct = 700.0, sw_diffuse = 0.0,
                             leaves = leaves, optics = nir, albedo_direct = 0.15)
        K = beam_extinction_coefficient(leaves, 0.6)
        Λ = [cumulative_leaf_area(forest, z) for z in zf]
        @test s.flux_dn_dir ≈ 700.0 .* exp.(-K .* Λ) rtol = 1e-9
        # exact hemispheric Beer–Lambert for spherical leaves is 2E₃(Λ/2)
        @test canopy_diffuse_transmittance(SphericalLeaves(), 3.0) ≈ 2 * _E3(1.5) rtol = 1e-10
    end

    @testset "energy conservation and sunlit/shaded partition" begin
        worst = 0.0; worst_split = 0.0; worst_lai = 0.0; worst_sun = 0.0
        for p in (LalicMihailovicLAD(6.0, 25.0; z_m = 17.0), BetaLAD(2.5, 1.0; p = 3, q = 2)),
            leaves in (SphericalLeaves(), RossGoudriaanLeaves(0.6)),
            μ in (0.05, 1.0), (ab, ad) in ((0.0, 0.0), (0.4, 0.3))
            h = canopy_height(p)
            zf = collect(range(0.0, 1.3h; length = 27))
            s = canopy_shortwave(p, zf; sun = μ, sw_direct = 650.0, sw_diffuse = 120.0,
                                 leaves = leaves, optics = nir, albedo_direct = ab, albedo_diffuse = ad)
            worst = max(worst, abs(s.incident - s.reflected - s.canopy_absorbed - s.ground_absorbed) / s.incident)
            worst_split = max(worst_split, maximum(abs.(s.absorbed_sunlit .+ s.absorbed_shaded .- s.absorbed)))
            dΛ = s.Λ_faces[1:end-1] .- s.Λ_faces[2:end]
            worst_lai = max(worst_lai, maximum(abs.(s.lai_sunlit .+ s.lai_shaded .- dΛ)))
            # total sunlit LAI = (1 − e^{−K·LAI})/K for any profile
            worst_sun = max(worst_sun, abs(sum(s.lai_sunlit) / ((1 - exp(-s.K * leaf_area_index(p))) / s.K) - 1))
        end
        @test worst < 1e-12
        @test worst_split < 1e-10
        @test worst_lai < 1e-12
        @test worst_sun < 1e-9
        # conservative leaves over a white ground: nothing absorbed
        s = canopy_shortwave(UniformLAD(3.0, 10.0), collect(0.0:1.0:10.0); sun = 0.6,
                             sw_direct = 500.0, sw_diffuse = 100.0,
                             optics = LeafOptics(0.6, 0.4), albedo_direct = 1.0)
        @test s.reflected ≈ s.incident rtol = 1e-9
        # one layer vs 40 layers of the same leaf area: same boundary fluxes
        kw = (sun = 0.55, sw_direct = 600.0, sw_diffuse = 150.0, optics = nir,
              leaves = RossGoudriaanLeaves(0.25), albedo_direct = 0.2, albedo_diffuse = 0.25)
        s1 = canopy_shortwave([0.0, 10.0], [0.4]; kw...)
        s40 = canopy_shortwave(UniformLAD(4.0, 10.0), collect(range(0.0, 10.0; length = 41)); kw...)
        @test s40.reflected ≈ s1.reflected rtol = 1e-10
        @test s40.canopy_absorbed ≈ s1.canopy_absorbed rtol = 1e-10
    end

    @testset "Sellers two-stream vs direct integration of its ODEs, incl. resonance K = k" begin
        # The beam source has the denominator 1 − k²/K², which vanishes at an
        # ordinary sun angle; TwoStream once left 3–14 % errors there.
        worst = 0.0
        LAI = 4.5; n = 20_000
        cases = Any[(RossGoudriaanLeaves(-0.3), 0.25, nir, 500.0, 120.0)]
        let leaves = SphericalLeaves(), o = par
            ω, β, _ = canopy_scattering_parameters(leaves, o, 0.5)
            μ̄ = average_inverse_optical_depth(leaves)
            γ1 = (1 - (1 - β) * ω) / μ̄; γ2 = ω * β / μ̄
            μr = leaves.φ1 / (sqrt(γ1^2 - γ2^2) - leaves.φ2)        # G(μ)/μ = k
            push!(cases, (leaves, μr, o, 500.0, 0.0))
        end
        for (leaves, μ, o, S0, D0) in cases
            s = canopy_shortwave(UniformLAD(LAI, 15.0), collect(0.0:0.75:15.0); sun = μ,
                                 sw_direct = S0, sw_diffuse = D0, leaves = leaves, optics = o,
                                 albedo_direct = 0.12, albedo_diffuse = 0.15)
            up, dn = _sellers_ode(; Λt = LAI, ω = s.ω, β = s.β, β0 = s.β0, μ̄ = s.μ̄, K = s.K,
                                  S0 = S0, D0 = D0, αb = 0.12, αd = 0.15, n = n)
            for k in eachindex(s.Λ_faces)
                i = round(Int, s.Λ_faces[k] / LAI * n) + 1
                worst = max(worst, abs(s.flux_up[k] - up[i]) / (S0 + D0),
                                   abs(s.flux_dn_dif[k] - dn[i]) / (S0 + D0))
            end
        end
        @test worst < 1e-8
    end

    @testset "Goudriaan (1977) reflection coefficient" begin
        # horizontal leaves, α = τ = σ/2, deep canopy: ρ_h = (1 − √(1−σ))/(1 + √(1−σ))
        worst = 0.0
        for σ in (0.15, 0.8)
            s = canopy_shortwave([0.0, 20.0], [10.0]; sun = 0.7, sw_direct = 0.0, sw_diffuse = 100.0,
                                 leaves = HorizontalLeaves(), optics = LeafOptics(σ / 2, σ / 2),
                                 albedo_direct = 0.0)
            worst = max(worst, abs(s.reflected / 100 - (1 - sqrt(1 - σ)) / (1 + sqrt(1 - σ))))
        end
        @test worst < 1e-9
    end

    @testset "longwave exchange" begin
        zf = collect(0.0:1.0:20.0)
        forest = LalicMihailovicLAD(4.0, 18.0; z_m = 12.0)
        σ = STEFAN_BOLTZMANN
        # isothermal, black sky at the same temperature: no net exchange
        s = canopy_longwave(forest, zf; lw_down = σ * 290.0^4, T_leaf = 290.0,
                            T_ground = 290.0, ε_leaf = 0.97, ε_ground = 0.95)
        @test maximum(abs.(s.net_absorbed)) < 1e-9 && abs(s.ground_net) < 1e-9
        # energy with a temperature profile
        Tl = [285.0 + 0.4k for k in 1:length(zf)-1]
        s = canopy_longwave(forest, zf; lw_down = 300.0, T_leaf = Tl, T_ground = 288.0,
                            ε_leaf = 0.95, ε_ground = 0.96, leaves = RossGoudriaanLeaves(0.4))
        @test abs((300.0 - s.outgoing) - sum(s.net_absorbed) - s.ground_net) < 1e-9
        # black leaves over a black ground: the analytic slab solution
        Tl, Tg, Ld = 295.0, 285.0, 320.0
        s = canopy_longwave(forest, zf; lw_down = Ld, T_leaf = Tl, T_ground = Tg,
                            ε_leaf = 1.0, ε_ground = 1.0)
        Tr = exp(-leaf_area_index(forest) / average_inverse_optical_depth(SphericalLeaves()))
        @test s.outgoing ≈ σ * Tg^4 * Tr + σ * Tl^4 * (1 - Tr) rtol = 1e-10
    end

    @testset "LES field: per-column absorption" begin
        mesh = RegularGrid(; size = (6, 4, 10), extent = (60.0, 40.0, 40.0))
        mask = [i <= 3 for i in 1:6, j in 1:4]
        lad = canopy_lad_field(mesh, LalicMihailovicLAD(4.0, 20.0; z_m = 13.0), mask)
        q = zeros(6, 4, 10)
        g = canopy_shortwave_field!(q, mesh, lad; sun = 0.6, sw_direct = 500.0, sw_diffuse = 100.0,
                                    optics = nir, albedo_direct = 0.15)
        @test all(iszero, q[4:6, :, :]) && g[5, 1] ≈ 0.85 * 600.0
        zf = [(k - 1) * mesh.dz for k in 1:11]
        s = canopy_shortwave(zf, lad[1, 1, :]; sun = 0.6, sw_direct = 500.0, sw_diffuse = 100.0,
                             optics = nir, albedo_direct = 0.15)
        @test q[1, 1, :] .* mesh.dz ≈ s.absorbed
    end
end
