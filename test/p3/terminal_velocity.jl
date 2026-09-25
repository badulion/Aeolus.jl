# Phase 44 (issue #8) tests: size-dependent terminal-velocity spectrum for
# the schemes that used a single fall speed.
#
# The physics asserted: single-moment Kessler rain, closed with the
# Marshall-Palmer spectrum, falls at the mass-weighted speed of that
# spectrum as computed by the P3 rain functions (one source of truth);
# the speed rises with rain content and matches the independent
# Klemp & Wilhelmson (1978) power-law fit of the same spectrum; the air
# density correction is exactly Foote-du Toit's; and the column update
# conserves rain mass exactly and moves the rain centroid at V_M.

@testset "Terminal-velocity spectrum (Phase 44)" begin
    ρw = Aeolus.Superdroplet.DEFAULT_MICRO.ρ_water
    interior_view = Aeolus.Fields.interior_view

    @testset "Marshall-Palmer closure of Kessler rain" begin
        # N_r from the closure reproduces q_r through the exponential PSD:
        # ρ q = π ρ_w N0 / λ⁴ with λ = N0 / (ρ N_r).
        for ρ in (1.225, 0.8), q in (1e-5, 1e-4, 1e-3, 5e-3)
            Nr = kessler_rain_number(q, ρ)
            λ = KESSLER_N0 / (ρ * Nr)
            @test π * ρw * KESSLER_N0 / λ^4 ≈ ρ * q rtol = 1e-12
        end
        @test kessler_rain_number(0.0, 1.2) == 0
    end

    @testset "Kessler fall speed: size-dependent, one source of truth" begin
        # Klemp & Wilhelmson (1978) fit of the Marshall-Palmer mass-weighted
        # speed, ρq in g cm⁻³: an independent published benchmark.
        kw(ρq) = 36.34 * (ρq * 1e-3)^0.1346
        vs = Float64[]
        for ρq in (1e-4, 1e-3, 3e-3, 1e-2)          # kg m⁻³ of rain
            v = kessler_fall_speed(ρq / 1.225, 1.225)
            push!(vs, v)
            @printf("    Kessler V_M(%.0e kg/m³) = %.3f m/s, Klemp-Wilhelmson %.3f (ratio %.4f)\n",
                    ρq, v, kw(ρq), v / kw(ρq))
            @test abs(v / kw(ρq) - 1) < 0.10
        end
        @test issorted(vs)                           # heavier rain falls faster
        @test vs[end] / vs[1] > 1.9                  # not a single fall speed
        # The speed IS the P3 mass-weighted speed of the MP spectrum.
        q = 2e-3
        λ = KESSLER_N0 / (1.225 * kessler_rain_number(q, 1.225))
        @test kessler_fall_speed(q) ≈ p3_rain_bulk_fall_speed(0.0, λ)[2] rtol = 1e-12
        # At fixed ρq (same spectrum) the density correction is Foote-du Toit.
        @test kessler_fall_speed(1e-3 / 0.7, 0.7) ≈
              kessler_fall_speed(1e-3 / 1.225, 1.225) * (1.225 / 0.7)^0.54 rtol = 1e-12
    end

    # One column of rain in a Kessler Field state.
    function rain_column(qtop; nz = 40, H = 4000.0, top = 5)
        mesh = RegularGrid(; size = (1, 1, nz), extent = (100.0, 100.0, H), halo = (2, 2, 2))
        qv = Field(CellCenter(), mesh); ql = Field(CellCenter(), mesh)
        qr = Field(CellCenter(), mesh)
        interior_view(qr)[1, 1, nz-top+1:nz] .= qtop
        return mesh, qv, ql, qr
    end
    colsum(qr) = sum(interior_view(qr))
    centroid(qr, dz) = (c = vec(interior_view(qr)[1, 1, :]);
                        sum(c .* ((1:length(c)) .- 0.5) .* dz) / sum(c))

    @testset "Kessler spectrum sedimentation: conservation and speed" begin
        for qtop in (1e-4, 5e-3)
            mesh, qv, ql, qr = rain_column(qtop)
            m0, z0 = colsum(qr), centroid(qr, mesh.dz)
            dt = 20.0                                # V Δt/Δz up to 1.5: sub-stepped
            apply_kessler_step!(qv, ql, qr, dt; k_auto = 0.0, k_acc = 0.0)
            # nothing reached the ground in 20 s from 3.5 km: exact conservation
            @test interior_view(qr)[1, 1, 1] == 0
            @test colsum(qr) ≈ m0 rtol = 1e-13
            @test minimum(interior_view(qr)) >= 0
            # centroid descends at (close to) the spectrum's V_M of the slab
            w = (z0 - centroid(qr, mesh.dz)) / dt
            V = kessler_fall_speed(qtop)
            @printf("    q_r=%.0e: centroid speed %.3f m/s vs V_M %.3f m/s\n", qtop, w, V)
            # the spreading slab thins, so its mean speed can only drop
            @test 0.8V < w <= V * (1 + 1e-12)
        end
        # heavy rain outruns light rain — impossible with one fall speed
        _, qv, ql, qa = rain_column(1e-4); _, _, _, qb = rain_column(5e-3)
        apply_kessler_step!(qv, ql, qa, 60.0; k_auto = 0.0, k_acc = 0.0)
        apply_kessler_step!(qv, ql, qb, 60.0; k_auto = 0.0, k_acc = 0.0)
        z0 = 4000.0 - 2.5 * 100.0
        @test (z0 - centroid(qb, 100.0)) > 1.5 * (z0 - centroid(qa, 100.0))
        # a legacy numeric Vt is still accepted, anything but :spectrum is not
        _, qv, ql, qr = rain_column(1e-3)
        @test_throws ArgumentError apply_kessler_step!(qv, ql, qr, 1.0; Vt = :constant)
    end
end
