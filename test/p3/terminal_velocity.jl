# Phase 44 (issue #8): size-dependent terminal-velocity spectrum for
# single-moment Kessler rain, closed with the Marshall-Palmer spectrum and
# computed by the P3 rain functions (one source of truth).

@testset "Terminal-velocity spectrum (Phase 44)" begin
    ρw = Aeolus.Superdroplet.DEFAULT_MICRO.ρ_water

    @testset "Marshall-Palmer closure and Kessler fall speed" begin
        # ρ q = π ρ_w N0 / λ⁴ with λ = N0 / (ρ N_r)
        q, ρ = 1e-3, 0.8
        λ = KESSLER_N0 / (ρ * kessler_rain_number(q, ρ))
        @test π * ρw * KESSLER_N0 / λ^4 ≈ ρ * q rtol = 1e-12
        # Klemp & Wilhelmson (1978) fit of the MP mass-weighted speed
        kw(ρq) = 36.34 * (ρq * 1e-3)^0.1346
        vs = [kessler_fall_speed(ρq / 1.225, 1.225) for ρq in (1e-4, 1e-2)]
        @test maximum(abs(v / kw(ρq) - 1) for (v, ρq) in zip(vs, (1e-4, 1e-2))) < 0.10
        @test vs[2] / vs[1] > 1.9                    # not a single fall speed
        λ2 = KESSLER_N0 / (1.225 * kessler_rain_number(2e-3, 1.225))
        @test kessler_fall_speed(2e-3) ≈ p3_rain_bulk_fall_speed(0.0, λ2)[2] rtol = 1e-12
    end
    # Column sedimentation of Kessler rain goes through p3_sediment!, whose
    # conservation is asserted in p3/sedimentation.jl (rain shaft).
end
