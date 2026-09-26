# Era V, Phase 48 — PlantCanopy (issue #23): radiation attenuation through
# the canopy. Beer–Lambert with the Ross–Goudriaan G-function, the Sellers
# (1985) two-stream solved with the Era III TwoStream layer formulae and
# adding solver, sunlit/shaded partition, and the longwave exchange.

using Test, Printf
using Aeolus
using Aeolus.PlantCanopy
using Aeolus.RadiationCore: STEFAN_BOLTZMANN, _gauss_legendre
using Aeolus.TwoStream: adding_solver!, layer_reflectance_transmittance_sw, DeltaEddington

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

@testset "Era V — PlantCanopy: radiation attenuation (#23)" begin

    par = LeafOptics(0.10, 0.05)     # typical broadleaf PAR
    nir = LeafOptics(0.45, 0.25)     # typical broadleaf NIR

    @testset "Ross–Goudriaan G-function, μ̄ and scattering parameters" begin
        sph = SphericalLeaves()
        for μ in (0.05, 0.3, 0.7, 1.0)
            @test leaf_projection_function(sph, μ) == 0.5
            @test beam_extinction_coefficient(sph, μ) ≈ 0.5 / μ
        end
        @test average_inverse_optical_depth(sph) ≈ 1.0 rtol = 1e-14
        @test_throws ArgumentError RossGoudriaanLeaves(0.8)
        @test_throws ArgumentError LeafOptics(0.7, 0.5)

        # μ̄ closed form (and its series branch) vs brute-force quadrature
        # of ∫₀¹ μ/G(μ) dμ, and ∫₀¹ G dμ (= 1/2 for a real distribution;
        # the Goudriaan *fit* departs from it — measured, not asserted tight).
        x, w = _gauss_legendre(200)
        worst_μ̄ = 0.0; worst_norm = 0.0
        for χ in (-0.4, -0.2, -1e-5, 0.0, 2e-4, 0.1, 0.25, 0.5, 0.6)
            d = RossGoudriaanLeaves(χ)
            # graded quadrature on (0,1): G is small near μ=0 for χ → 0.6
            ref = 0.0; lo = 0.0
            for m in 20:-1:0
                hi = 2.0^(-m)
                ref += sum(w[q] * (hi - lo) / 2 *
                           (μ -> μ / leaf_projection_function(d, μ))((hi + lo) / 2 + (hi - lo) / 2 * x[q])
                           for q in eachindex(x))
                lo = hi
            end
            worst_μ̄ = max(worst_μ̄, abs(average_inverse_optical_depth(d) / ref - 1))
            norm = d.φ1 + d.φ2 / 2
            worst_norm = max(worst_norm, abs(norm - 0.5))
            @test all(leaf_projection_function(d, μ) > 0 for μ in 0:0.01:1)
        end
        @test worst_μ̄ < 1e-10
        @info @sprintf("μ̄ closed form vs quadrature: worst rel. error %.2e; Goudriaan fit ∫G dμ departs from 1/2 by up to %.3f",
                       worst_μ̄, worst_norm)

        # a_s closed form (inside β₀) vs direct quadrature of its integral
        worst_as = 0.0
        for χ in (-0.4, 0.0, 0.25, 0.6), μ in (0.1, 0.5, 0.9), o in (par, nir)
            d = RossGoudriaanLeaves(χ)
            ω, β, β0 = canopy_scattering_parameters(d, o, μ)
            @test ω ≈ o.reflectance + o.transmittance
            @test 0 <= β <= 1 && 0 <= β0 <= 1
            G = leaf_projection_function(d, μ); K = G / μ
            μ̄ = average_inverse_optical_depth(d)
            a_s = (ω / 2) * sum(w[q] / 2 * (m -> m * G / (μ * leaf_projection_function(d, m) + m * G))((x[q] + 1) / 2)
                                for q in eachindex(x))
            worst_as = max(worst_as, abs(β0 / ((1 + μ̄ * K) / (μ̄ * K) * a_s / ω) - 1))
        end
        @test worst_as < 1e-8
        # χ_L = -0.4 has μφ₂ + G = 0 at μ* = φ₁/(2|φ₂|) ≈ 0.9963, where the
        # closed form of a_s is 0/0 (it returned NaN there before the series
        # branch, and lost 5 digits within 1e-6 of μ*).
        d = RossGoudriaanLeaves(-0.4); μ̄ = average_inverse_optical_depth(d)
        μstar = -d.φ1 / (2d.φ2)
        worst_star = 0.0
        for δ in (0.0, 1e-12, 1e-8, -1e-8, 1e-6, -1e-6, 1e-3)
            μ = μstar + δ
            ω, β, β0 = canopy_scattering_parameters(d, par, μ)
            G = leaf_projection_function(d, μ); K = G / μ
            a_s = (ω / 2) * sum(w[q] / 2 * (m -> m * G / (μ * leaf_projection_function(d, m) + m * G))((x[q] + 1) / 2)
                                for q in eachindex(x))
            worst_star = max(worst_star, abs(β0 / ((1 + μ̄ * K) / (μ̄ * K) * a_s / ω) - 1))
        end
        @test worst_star < 1e-12
        # Spherical leaves, α = τ: ωβ = ω/2 (up and down scatter equally)
        ω, β, _ = canopy_scattering_parameters(sph, LeafOptics(0.2, 0.2), 0.5)
        @test β ≈ 0.5
        # Horizontal leaves scatter α up and τ down exactly
        ωh, βh, β0h = canopy_scattering_parameters(HorizontalLeaves(), par, 0.3)
        @test ωh ≈ 0.15 && βh ≈ 2 / 3 && β0h ≈ 2 / 3
    end

    @testset "Beer–Lambert: direct beam and black-leaf diffuse" begin
        forest = LalicMihailovicLAD(5.0, 20.0; z_m = 13.0)
        zf = collect(0.0:1.0:25.0)
        for leaves in (SphericalLeaves(), RossGoudriaanLeaves(-0.3), RossGoudriaanLeaves(0.5)),
            μ in (0.2, 0.6, 1.0), o in (par, nir)
            s = canopy_shortwave(forest, zf; sun = μ, sw_direct = 700.0, sw_diffuse = 0.0,
                                 leaves = leaves, optics = o, albedo_direct = 0.15)
            K = beam_extinction_coefficient(leaves, μ)
            # the unscattered beam follows e^{-KΛ(z)} whatever the scattering
            Λ = [cumulative_leaf_area(forest, z) for z in zf]
            @test s.flux_dn_dir ≈ 700.0 .* exp.(-K .* Λ) rtol = 1e-9
            @test s.Λ_faces ≈ Λ rtol = 1e-9 atol = 1e-12
        end
        # black leaves, black soil, diffuse only: two-stream gives e^{-Λ/μ̄}
        d = RossGoudriaanLeaves(0.25)
        s = canopy_shortwave(UniformLAD(3.0, 10.0), collect(0.0:0.5:12.0); sun = 0.5,
                             sw_direct = 0.0, sw_diffuse = 200.0, leaves = d,
                             optics = LeafOptics(0.0, 0.0), albedo_direct = 0.0)
        @test s.flux_dn_dif ≈ 200.0 .* exp.(-s.Λ_faces ./ average_inverse_optical_depth(d)) rtol = 1e-10
        @test maximum(s.flux_up) == 0

        # Exact hemispheric Beer–Lambert for spherical leaves is 2E₃(Λ/2)
        worst = 0.0
        for Λ in (0.1, 0.5, 1.0, 3.0, 6.0)
            Td = canopy_diffuse_transmittance(SphericalLeaves(), Λ)
            worst = max(worst, abs(Td / (2 * _E3(Λ / 2)) - 1))
        end
        @test worst < 1e-10
        # horizontal leaves: every direction sees K = 1, so T_d = e^{-Λ}
        @test canopy_diffuse_transmittance(HorizontalLeaves(), 2.0) ≈ exp(-2.0) rtol = 1e-10
        @info @sprintf("black spherical leaves, Λ = 3: exact diffuse T = %.4f, two-stream e^{-Λ/μ̄} = %.4f; Λ = 1: %.4f vs %.4f",
                       canopy_diffuse_transmittance(SphericalLeaves(), 3.0), exp(-3.0),
                       canopy_diffuse_transmittance(SphericalLeaves(), 1.0), exp(-1.0))
    end

    @testset "energy conservation and sunlit/shaded partition" begin
        worst = 0.0; worst_split = 0.0; worst_lai = 0.0
        profs = (UniformLAD(4.0, 20.0), LalicMihailovicLAD(6.0, 25.0; z_m = 17.0),
                 BetaLAD(2.5, 1.0; p = 3, q = 2), UniformLAD(1.0, 8.0; z_base = 3.0))
        for p in profs, leaves in (SphericalLeaves(), RossGoudriaanLeaves(-0.4),
                                   RossGoudriaanLeaves(0.6), HorizontalLeaves()),
            μ in (0.05, 0.4, 1.0), o in (par, nir), (ab, ad) in ((0.0, 0.0), (0.1, 0.12), (0.4, 0.3))
            h = canopy_height(p)
            zf = collect(range(0.0, 1.3h; length = 27))
            s = canopy_shortwave(p, zf; sun = μ, sw_direct = 650.0, sw_diffuse = 120.0,
                                 leaves = leaves, optics = o, albedo_direct = ab, albedo_diffuse = ad)
            worst = max(worst, abs(s.incident - s.reflected - s.canopy_absorbed - s.ground_absorbed) / s.incident)
            worst_split = max(worst_split, maximum(abs.(s.absorbed_sunlit .+ s.absorbed_shaded .- s.absorbed)))
            dΛ = s.Λ_faces[1:end-1] .- s.Λ_faces[2:end]
            worst_lai = max(worst_lai, maximum(abs.(s.lai_sunlit .+ s.lai_shaded .- dΛ)))
            @test all(>=(-1e-9), s.absorbed_sunlit) && all(>=(-1e-9), s.absorbed_shaded)
            @test all(>=(-1e-9), s.absorbed)
            # total sunlit LAI = (1 - e^{-K·LAI})/K for any profile
            K = s.K
            @test sum(s.lai_sunlit) ≈ (1 - exp(-K * leaf_area_index(p))) / K rtol = 1e-9
        end
        @test worst < 1e-12
        @test worst_split < 1e-10
        @test worst_lai < 1e-12
        # NB: the closure is an identity of the bookkeeping (layer absorption
        # is defined as the divergence of the face fluxes), so it checks the
        # assembly, not the physics; the physics is checked against RK4 below.
        @info @sprintf("SW energy closure (bookkeeping identity): worst |in - refl - abs_canopy - abs_ground|/in = %.1e over 576 cases", worst)

        # Conservative leaves over a white ground: nothing is absorbed.
        s = canopy_shortwave(UniformLAD(3.0, 10.0), collect(0.0:1.0:10.0); sun = 0.6,
                             sw_direct = 500.0, sw_diffuse = 100.0,
                             optics = LeafOptics(0.6, 0.4), albedo_direct = 1.0)
        @test s.reflected ≈ s.incident rtol = 1e-9
        @test abs(s.canopy_absorbed) < 1e-7 * s.incident

        # Sun below the horizon: no direct beam, no sunlit leaves.
        s = canopy_shortwave(UniformLAD(3.0, 10.0), collect(0.0:1.0:10.0); sun = -0.1,
                             sw_direct = 500.0, sw_diffuse = 50.0, optics = par, albedo_direct = 0.1)
        @test s.incident == 50.0 && all(iszero, s.flux_dn_dir) && all(iszero, s.lai_sunlit)
        @test s.incident ≈ s.reflected + s.canopy_absorbed + s.ground_absorbed rtol = 1e-12
    end

    @testset "layering independence and TwoStream coupling" begin
        # The equations depend on height only through Λ, so one layer and 40
        # layers of the same leaf area must give the same boundary fluxes.
        kw = (sun = 0.55, sw_direct = 600.0, sw_diffuse = 150.0, optics = nir,
              leaves = RossGoudriaanLeaves(0.25), albedo_direct = 0.2, albedo_diffuse = 0.25)
        s1 = canopy_shortwave([0.0, 10.0], [0.4]; kw...)
        s40 = canopy_shortwave(UniformLAD(4.0, 10.0), collect(range(0.0, 10.0; length = 41)); kw...)
        @test s40.reflected ≈ s1.reflected rtol = 1e-10
        @test s40.ground_absorbed ≈ s1.ground_absorbed rtol = 1e-10
        @test s40.canopy_absorbed ≈ s1.canopy_absorbed rtol = 1e-10

        # TwoStream.adding_solver! with the new diffuse top boundary: an
        # atmosphere lit only by diffuse light from above conserves energy,
        # and the default (F_dif_top = 0) is unchanged.
        nl = 5
        R = zeros(nl); T = zeros(nl); Rr = zeros(nl); Tr = zeros(nl); Tb = zeros(nl)
        for k in 1:nl
            R[k], T[k], Rr[k], Tr[k], Tb[k] =
                layer_reflectance_transmittance_sw(DeltaEddington(), 0.3k, 0.9, 0.7, 0.5)
        end
        fu = zeros(nl + 1); fd = zeros(nl + 1); fdir = zeros(nl + 1)
        adding_solver!(fu, fd, fdir, R, T, Rr, Tr, Tb, 0.2, 0.2, 0.0, 0.5; F_dif_top = 100.0)
        @test fd[end] == 100.0
        absorbed = (fd[end] - fu[end]) - (fd[1] - fu[1])
        ground = fd[1] - fu[1]
        @test fu[1] ≈ 0.2 * fd[1] rtol = 1e-12
        @test 100.0 ≈ fu[end] + absorbed + ground     # trivially, but also:
        @test ground > 0 && fu[end] > 0 && all(fd .>= 0)
        fu2 = zeros(nl + 1); fd2 = zeros(nl + 1)
        adding_solver!(fu2, fd2, fdir, R, T, Rr, Tr, Tb, 0.2, 0.2, 1000.0, 0.5)
        @test fd2[end] == 0.0
    end

    @testset "Sellers two-stream vs direct integration of its ODEs" begin
        worst = 0.0
        for leaves in (SphericalLeaves(), RossGoudriaanLeaves(-0.3), RossGoudriaanLeaves(0.5)),
            μ in (0.25, 0.8), o in (par, nir), (S0, D0) in ((700.0, 0.0), (0.0, 200.0), (500.0, 120.0))
            LAI = 4.5
            s = canopy_shortwave(UniformLAD(LAI, 15.0), collect(0.0:0.75:15.0); sun = μ,
                                 sw_direct = S0, sw_diffuse = D0, leaves = leaves, optics = o,
                                 albedo_direct = 0.12, albedo_diffuse = 0.15)
            n = 20_000
            up, dn = _sellers_ode(; Λt = LAI, ω = s.ω, β = s.β, β0 = s.β0, μ̄ = s.μ̄, K = s.K,
                                  S0 = S0, D0 = D0, αb = 0.12, αd = 0.15, n = n)
            # compare at every face: face k is at Λ = s.Λ_faces[k]
            for k in eachindex(s.Λ_faces)
                i = round(Int, s.Λ_faces[k] / LAI * n) + 1
                worst = max(worst, abs(s.flux_up[k] - up[i]) / (S0 + D0),
                                   abs(s.flux_dn_dif[k] - dn[i]) / (S0 + D0))
            end
        end
        @test worst < 1e-8
        @info @sprintf("two-stream (TwoStream adding) vs RK4 of the Sellers ODEs: worst flux error %.1e of incident", worst)
    end

    @testset "two-stream resonance K = k (TwoStream removable singularity)" begin
        # The beam source terms have the denominator 1 - k²/K², which
        # vanishes at a perfectly ordinary sun angle (μ ≈ 0.54 for spherical
        # PAR leaves). TwoStream used to shift μ₀ there but keep e^{-τ/μ₀},
        # which left errors of 3-14 % of the incident flux; compare with the
        # RK4 solution exactly at resonance.
        worst = 0.0
        for (leaves, o) in ((SphericalLeaves(), par), (SphericalLeaves(), nir),
                            (RossGoudriaanLeaves(0.5), par))
            ω, β, _ = canopy_scattering_parameters(leaves, o, 0.5)
            μ̄ = average_inverse_optical_depth(leaves)
            γ1 = (1 - (1 - β) * ω) / μ̄; γ2 = ω * β / μ̄
            k = sqrt(γ1^2 - γ2^2)
            μr = leaves.φ1 / (k - leaves.φ2)             # G(μ)/μ = k
            for δ in (0.0, 1e-7, 1e-5)
                LAI = 4.5; n = 20_000
                s = canopy_shortwave(UniformLAD(LAI, 15.0), collect(0.0:0.75:15.0); sun = μr + δ,
                                     sw_direct = 500.0, sw_diffuse = 0.0, leaves = leaves,
                                     optics = o, albedo_direct = 0.12, albedo_diffuse = 0.15)
                up, dn = _sellers_ode(; Λt = LAI, ω = s.ω, β = s.β, β0 = s.β0, μ̄ = s.μ̄, K = s.K,
                                      S0 = 500.0, D0 = 0.0, αb = 0.12, αd = 0.15, n = n)
                for f in eachindex(s.Λ_faces)
                    i = round(Int, s.Λ_faces[f] / LAI * n) + 1
                    worst = max(worst, abs(s.flux_up[f] - up[i]) / 500, abs(s.flux_dn_dif[f] - dn[i]) / 500)
                end
            end
        end
        @test worst < 1e-8
        @info @sprintf("two-stream at the resonance K = k vs RK4: worst flux error %.1e of incident", worst)
    end

    @testset "Goudriaan (1977) reflection coefficient" begin
        # Horizontal leaves with α = τ = σ/2 over a deep canopy: the diffuse
        # canopy reflectance is ρ_h = (1 - √(1-σ))/(1 + √(1-σ)) exactly.
        # "Deep" must be deep for σ → 1: the finite-depth correction decays
        # as e^{-2√(1-σ)Λ}, which at σ = 0.95 and Λ = 30 is still 6e-7.
        worst = 0.0
        for σ in (0.15, 0.3, 0.5, 0.8, 0.95)
            s = canopy_shortwave([0.0, 20.0], [10.0]; sun = 0.7, sw_direct = 0.0, sw_diffuse = 100.0,
                                 leaves = HorizontalLeaves(), optics = LeafOptics(σ / 2, σ / 2),
                                 albedo_direct = 0.0)
            ρh = (1 - sqrt(1 - σ)) / (1 + sqrt(1 - σ))
            worst = max(worst, abs(s.reflected / 100 - ρh))
        end
        @test worst < 1e-9
        @info @sprintf("horizontal leaves, deep canopy: |two-stream albedo - Goudriaan ρ_h| ≤ %.1e", worst)

        # Spherical leaves, PAR (σ = 0.15), deep canopy, direct beam: the
        # widely used approximation ρ_cb = 1 - exp(-2ρ_h K/(1+K))
        # (Goudriaan & van Laar 1994, as used by de Pury & Farquhar 1997).
        # It is an approximation to a different angular closure, so the
        # agreement is measured and only loosely bounded.
        σ = 0.15; ρh = (1 - sqrt(1 - σ)) / (1 + sqrt(1 - σ))
        for μ in (0.3, 0.6, 1.0)
            K = 0.5 / μ
            s = canopy_shortwave([0.0, 20.0], [1.0]; sun = μ, sw_direct = 100.0, sw_diffuse = 0.0,
                                 optics = LeafOptics(σ / 2, σ / 2), albedo_direct = 0.0)
            ρcb = 1 - exp(-2ρh * K / (1 + K))
            @info @sprintf("μ = %.1f: two-stream canopy albedo %.4f, Goudriaan ρ_cb %.4f", μ, s.reflected / 100, ρcb)
            @test abs(s.reflected / 100 - ρcb) < 0.3ρcb
        end
    end

    @testset "SolarGeometry coupling" begin
        st = solar_time(2024, 6, 21, 11, 0, 0; latitude = 52.52, longitude = 13.40)
        sp = solar_position(st)
        irr = clear_sky_irradiance(sp, BirdClearSky())
        zf = collect(0.0:2.0:30.0); forest = LalicMihailovicLAD(5.0, 25.0; z_m = 17.0)
        a = canopy_shortwave(forest, zf; sun = sp, sw_direct = irr.dni * sp.μ₀,
                             sw_diffuse = irr.dhi, optics = par, albedo_direct = 0.1)
        b = canopy_shortwave(forest, zf; sun = sp.μ₀, sw_direct = irr.dni * sp.μ₀,
                             sw_diffuse = irr.dhi, optics = par, albedo_direct = 0.1)
        @test a.flux_up == b.flux_up && a.K == b.K
        @test a.incident ≈ irr.ghi rtol = 1e-12
    end

    @testset "longwave exchange" begin
        zf = collect(0.0:1.0:20.0)
        forest = LalicMihailovicLAD(4.0, 18.0; z_m = 12.0)
        σ = STEFAN_BOLTZMANN
        # Isothermal, black sky at the same temperature: no net exchange.
        for εl in (1.0, 0.97, 0.9), εg in (1.0, 0.95)
            s = canopy_longwave(forest, zf; lw_down = σ * 290.0^4, T_leaf = 290.0,
                                T_ground = 290.0, ε_leaf = εl, ε_ground = εg)
            @test maximum(abs.(s.net_absorbed)) < 1e-9
            @test abs(s.ground_net) < 1e-9
            @test s.outgoing ≈ σ * 290.0^4 rtol = 1e-12
        end
        # Energy: sky in - out = leaves + ground, with a temperature profile.
        worst = 0.0
        for εl in (1.0, 0.95), leaves in (SphericalLeaves(), RossGoudriaanLeaves(0.4))
            Tl = [285.0 + 0.4k for k in 1:length(zf)-1]
            s = canopy_longwave(forest, zf; lw_down = 300.0, T_leaf = Tl, T_ground = 288.0,
                                ε_leaf = εl, ε_ground = 0.96, leaves = leaves)
            worst = max(worst, abs((300.0 - s.outgoing) - sum(s.net_absorbed) - s.ground_net))
        end
        @test worst < 1e-9
        # Black leaves over a black ground: the analytic slab solution.
        Tl, Tg, Ld = 295.0, 285.0, 320.0
        s = canopy_longwave(forest, zf; lw_down = Ld, T_leaf = Tl, T_ground = Tg,
                            ε_leaf = 1.0, ε_ground = 1.0)
        Tr = exp(-leaf_area_index(forest) / average_inverse_optical_depth(SphericalLeaves()))
        @test s.outgoing ≈ σ * Tg^4 * Tr + σ * Tl^4 * (1 - Tr) rtol = 1e-10
        @test s.ground_net ≈ Ld * Tr + σ * Tl^4 * (1 - Tr) - σ * Tg^4 rtol = 1e-10
        # Warm leaves under a cold sky lose energy, and the top layers most
        s = canopy_longwave(UniformLAD(4.0, 20.0), zf; lw_down = 280.0, T_leaf = 295.0, T_ground = 290.0)
        @test all(s.net_absorbed .<= 0) && s.net_absorbed[20] < s.net_absorbed[5]
    end

    @testset "LES field: per-column absorption" begin
        mesh = RegularGrid(; size = (6, 4, 10), extent = (60.0, 40.0, 40.0))
        mask = [i <= 3 for i in 1:6, j in 1:4]
        lad = canopy_lad_field(mesh, LalicMihailovicLAD(4.0, 20.0; z_m = 13.0), mask)
        q = zeros(6, 4, 10)
        Sdir = [i == 2 ? 0.0 : 500.0 for i in 1:6, j in 1:4]      # a shadowed column
        g = canopy_shortwave_field!(q, mesh, lad; sun = 0.6, sw_direct = Sdir, sw_diffuse = 100.0,
                                    optics = nir, albedo_direct = 0.15)
        # bare columns: all to the ground; vegetated: matches the 1-D solve
        @test all(iszero, q[4:6, :, :])
        @test g[5, 1] ≈ 0.85 * 600.0
        zf = [(k - 1) * mesh.dz for k in 1:11]
        s = canopy_shortwave(zf, lad[1, 1, :]; sun = 0.6, sw_direct = 500.0, sw_diffuse = 100.0,
                             optics = nir, albedo_direct = 0.15)
        @test q[1, 1, :] .* mesh.dz ≈ s.absorbed
        @test g[1, 1] ≈ s.ground_absorbed
        @test sum(q[2, 1, :]) * mesh.dz < sum(q[1, 1, :]) * mesh.dz
    end
end
