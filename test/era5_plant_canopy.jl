# Era V, Phase 48 — PlantCanopy (issue #22): leaf-area-density profiles
# and implicit canopy drag.

using Test, Printf
using Aeolus
using Aeolus.PlantCanopy

@testset "Era V — PlantCanopy: LAD and implicit drag (#22)" begin

    @testset "LAD profiles integrate to LAI and have the published shape" begin
        h = 20.0
        profs = (UniformLAD(4.0, h),
                 UniformLAD(3.0, h; z_base = 8.0),
                 LalicMihailovicLAD(5.0, h; z_m = 0.6h),
                 LalicMihailovicLAD(2.0, h; z_m = 0.8h),
                 BetaLAD(4.0, h; p = 3, q = 2),
                 BetaLAD(6.0, h; p = 1, q = 1),
                 BetaLAD(3.5, h; p = 6.5, q = 1.5))
        worst = 0.0
        for p in profs
            # ∫₀ʰ a dz = LAI — here with an independent brute-force midpoint
            # rule (1e5 cells) rather than the module's own quadrature.
            N = 100_000; dz = h / N
            s = sum(leaf_area_density(p, (k - 0.5) * dz) for k in 1:N) * dz
            worst = max(worst, abs(s / leaf_area_index(p) - 1))
            @test s ≈ leaf_area_index(p) rtol = 1e-6
            # Cumulative leaf area: 0 at top, LAI at ground, monotone.
            @test cumulative_leaf_area(p, h) == 0
            @test cumulative_leaf_area(p, 0.0) ≈ leaf_area_index(p) rtol = 1e-12
            Λ = [cumulative_leaf_area(p, z) for z in 0:0.5:h]
            @test all(diff(Λ) .<= 1e-14)
            # No leaves outside the canopy.
            @test leaf_area_density(p, h + 0.1) == 0
            @test leaf_area_density(p, -0.1) == 0
            @test canopy_height(p) == h
        end
        @info "LAD normalisation (midpoint, 1e5 cells)" worst_rel_err = worst

        # Uniform: a = LAI / crown depth.
        @test leaf_area_density(UniformLAD(4.0, h), 10.0) ≈ 0.2
        @test leaf_area_density(UniformLAD(3.0, h; z_base = 8.0), 5.0) == 0
        @test leaf_area_density(UniformLAD(3.0, h; z_base = 8.0), 10.0) ≈ 0.25

        # Lalic & Mihailovic (2004): maximum L_m at z_m, n = 6 below and
        # n = 1/2 above, vanishing at the top.
        lm = LalicMihailovicLAD(5.0, h; z_m = 12.0)
        zs = range(0, h; length = 4001)
        a = leaf_area_density.(Ref(lm), zs)
        @test zs[argmax(a)] ≈ 12.0 atol = h / 4000
        @test leaf_area_density(lm, 12.0) ≈ lm.L_m
        @test leaf_area_density(lm, h) == 0
        @test leaf_area_density(lm, h - 1e-6) < 1e-3 * lm.L_m
        # The published formula, evaluated by hand at two heights.
        r(z) = (h - 12.0) / (h - z)
        @test leaf_area_density(lm, 6.0)  ≈ lm.L_m * r(6.0)^6 * exp(6 * (1 - r(6.0)))
        @test leaf_area_density(lm, 16.0) ≈ lm.L_m * sqrt(r(16.0)) * exp(0.5 * (1 - r(16.0)))
        # n = 6 below makes the lower trunk space sparse and the crown dense:
        # the lowest z_m/2 (30 % of the depth) holds < 15 % of the leaf
        # area, the 40 % of the depth above z_m holds > 40 % of it.
        @test leaf_area_index(lm) - cumulative_leaf_area(lm, 6.0) < 0.15 * leaf_area_index(lm)
        @test cumulative_leaf_area(lm, 12.0) > 0.40 * leaf_area_index(lm)

        # Beta distribution: known moments of Beta(p, q).
        for (pp, qq) in ((3.0, 2.0), (2.0, 5.0), (6.5, 1.5))
            b = BetaLAD(4.0, h; p = pp, q = qq)
            N = 200_000; dz = h / N
            zc = [(k - 0.5) * dz for k in 1:N]
            w  = leaf_area_density.(Ref(b), zc)
            zmean = sum(w .* zc) / sum(w)
            @test zmean ≈ h * pp / (pp + qq) rtol = 1e-6
            @test zc[argmax(w)] ≈ h * (pp - 1) / (pp + qq - 2) atol = 2dz
        end
        # Beta(1,1) is the uniform canopy.
        @test leaf_area_density(BetaLAD(6.0, h; p = 1, q = 1), 7.3) ≈
              leaf_area_density(UniformLAD(6.0, h), 7.3)

        @test_throws ArgumentError UniformLAD(-1.0, h)
        @test_throws ArgumentError LalicMihailovicLAD(1.0, h; z_m = h)
        @test_throws ArgumentError BetaLAD(1.0, h; p = 0.5)
    end

    @testset "Layer LAD conserves LAI on any grid" begin
        h = 20.0
        for p in (UniformLAD(4.0, h), LalicMihailovicLAD(5.0, h; z_m = 12.0),
                  BetaLAD(4.0, h; p = 3, q = 2))
            # Coarse, canopy-top-straddling, and stretched grids.
            for zf in (collect(0:6.25:50.0),             # the urban example's dz
                       collect(range(0, 31; length = 4)),
                       cumsum([0.0; 0.5 .* 1.15 .^ (0:30)]))
                a = layer_leaf_area_density(p, zf)
                col = sum(a .* diff(zf))
                @test col ≈ leaf_area_index(p) rtol = 1e-12
            end
        end
        # A point sample at layer centres would not conserve LAI when a layer
        # straddles the top: with dz = 6.25 m and h = 20 m the top canopy
        # layer is [18.75, 25] and centre-sampling puts zero leaf there.
        u = UniformLAD(4.0, h)
        zf = collect(0:6.25:50.0)
        sampled = sum(leaf_area_density.(Ref(u), zf[1:end-1] .+ 3.125) .* 6.25)
        @test sampled ≈ 3.75            # 3 of 3.2 layers: 6 % of LAI lost
        @test sum(layer_leaf_area_density(u, zf) .* 6.25) ≈ 4.0
    end
end
