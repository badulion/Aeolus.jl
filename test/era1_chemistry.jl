# Reduced urban NOx–O3–VOC mechanism, dry and wet deposition
# (src/Chemistry.jl, #135).

using Test
using Aeolus
using Aeolus.Chemistry
using Random: Xoshiro

# Classical RK4 reference for the triad R1–R2 (tiny steps).
function triad_rk4(NO, NO2, O3, t, J, k; n = 200_000)
    f(x) = (r = J * x[2] - k * x[1] * x[3]; (r, -r, r))
    x = (NO, NO2, O3); h = t / n
    for _ in 1:n
        a = f(x); b = f(x .+ h / 2 .* a); c = f(x .+ h / 2 .* b); d = f(x .+ h .* c)
        x = x .+ h / 6 .* (a .+ 2 .* b .+ 2 .* c .+ d)
    end
    return x
end

@testset "Chemistry: urban NOx–O3–VOC mechanism (#135)" begin

    @testset "units and rate constants" begin
        M = air_number_density(101325.0, 298.15)
        @test M ≈ 2.461e19 rtol = 1e-3
        @test k_NO_O3(298.0) ≈ 1.95e-14 rtol = 0.01          # JPL 19-5
        @test k_NO_O3(280.0) < k_NO_O3(300.0)                # activation energy > 0
        @test k_VOC_OH(298.0) ≈ 1.0e-10 rtol = 0.02          # isoprene + OH, IUPAC
        @test k_VOC_O3(298.0) ≈ 1.27e-17 rtol = 0.02         # isoprene + O3, IUPAC
        @test 1.0e-11 < k_NO2_OH(298.0, M) < 1.1e-11         # termolecular, near 1 atm
        # falloff: low-pressure limit is linear in M, high-pressure limit is k∞
        # (the broadening factor approaches 1 only logarithmically, so the
        # limits are checked far out: at M = 1e15 the ratio is still 0.502)
        @test k_NO2_OH(298.0, 1e8) / k_NO2_OH(298.0, 2e8) ≈ 0.5 rtol = 1e-3
        @test k_NO2_OH(298.0, 1e40) ≈ 2.8e-11 rtol = 0.005
        # 1 ppb⁻¹ s⁻¹ ⇔ k M 1e-9
        @test ppb_rate_constant(k_NO_O3(298.15), 101325.0, 298.15) ≈ 4.8e-4 rtol = 0.01
        # 1 µg m⁻² s⁻¹ of NO (30 g/mol) ≈ 0.80 ppb m s⁻¹ at 20 °C
        @test emission_ppb_flux(1e-9, 0.030; T = 293.15) ≈ 0.8016 rtol = 1e-3
    end

    @testset "NO2 photolysis frequency" begin
        @test j_NO2(1.0) ≈ 1.165e-2 * exp(-0.267) rtol = 1e-12   # 8.9e-3 s⁻¹ overhead
        @test 8.8e-3 < j_NO2(1.0) < 9.0e-3
        @test j_NO2(0.0) == 0 && j_NO2(-0.3) == 0
        μs = 0.05:0.05:1.0
        @test issorted(j_NO2.(μs))
        # at 60° zenith the MCM curve gives 65 % of the overhead value
        @test 0.6 < j_NO2(0.5) / j_NO2(1.0) < 0.7
    end

    @testset "triad: exact solution, conservation, stability" begin
        J, k = 8e-3, ppb_rate_constant(k_NO_O3(298.0), 101325.0, 298.0)
        for (NO, NO2, O3) in ((0.5, 5.0, 40.0), (30.0, 10.0, 25.0), (2.0, 0.0, 0.0), (0.0, 3.0, 0.0))
            for t in (1.0, 20.0, 300.0)
                ref = triad_rk4(NO, NO2, O3, t, J, k)
                got = nox_triad(NO, NO2, O3, t, J, k)
                @test all(isapprox.(got, ref; rtol = 1e-9, atol = 1e-12))
            end
            for t in (2.0, 1e3, 1e7)      # any dt: invariants to round-off, positive
                a, b, c = nox_triad(NO, NO2, O3, t, J, k)
                @test a + b ≈ NO + NO2 rtol = 1e-14 atol = 1e-14
                @test c + b ≈ O3 + NO2 rtol = 1e-14 atol = 1e-14
                @test min(a, b, c) >= 0
            end
        end
        # night (J = 0): titration, O3 in excess eats all the NO; Ox conserved
        a, b, c = nox_triad(10.0, 5.0, 40.0, 1e6, 0.0, k)
        @test a < 1e-9 && b ≈ 15.0 && c ≈ 30.0
        @test (b + c) == 45.0
        # degenerate root (J = 0, NOx = Ox): y - y1 = (y0-y1)/(1 - k (y0-y1) t)
        NO, O3 = 7.0, 7.0
        for t in (1.0, 50.0, 1e3)
            _, y, _ = nox_triad(NO, 0.0, O3, t, 0.0, k)
            @test y ≈ 7.0 - 7.0 / (1 + k * 7.0 * t) rtol = 1e-12
        end
        # the photostationary state: Leighton ratio 1
        N, X = 5.5, 45.0
        y = photostationary_NO2(N, X, J, k)
        @test leighton_ratio(N - y, y, X - y, J, k) ≈ 1 rtol = 1e-12
        a, b, c = nox_triad(3.0, 2.5, 42.5, 3600.0, J, k)       # relaxes to it
        @test leighton_ratio(a, b, c, J, k) ≈ 1 rtol = 1e-10
        @test b ≈ y rtol = 1e-10
        # relaxation time 1/(k r) = 1/D: after one e-folding the gap shrinks by e
        B = k * (N + X) + J; D = sqrt(B^2 - 4k^2 * N * X)
        a0 = 5.5 - y                                            # start with all NOx as NO2
        _, y1, _ = nox_triad(0.0, 5.5, 39.5, 1 / D, J, k)
        @test (y1 - y) / a0 ≈ exp(-1) rtol = 0.05               # quadratic term: a few %
    end

    @testset "full mechanism: budgets and ozone production" begin
        J, T, p = j_NO2(0.9), 298.0, 101325.0
        M = air_number_density(p, T)
        k1 = ppb_rate_constant(k_NO_O3(T), p, T)
        k3 = ppb_rate_constant(k_VOC_O3(T), p, T)
        OH = 5e6
        L4, L5 = k_VOC_OH(T) * OH, k_NO2_OH(T, M) * OH
        s0 = (4.0, 6.0, 40.0, 5.0, 0.0)
        s = s0
        Pc = 0.0; Lc = 0.0
        for _ in 1:1800                         # 1 h at dt = 2 s
            r = photochemistry_cell(s..., 2.0; J = J, k1 = k1, k3 = k3, L4 = L4, L5 = L5)
            # exact per-step budgets
            @test (r[3] + r[2]) - (s[3] + s[2]) ≈ r[6] - r[7] atol = 1e-12
            @test (r[1] + r[2]) - (s[1] + s[2]) ≈ -(r[5] - s[5]) atol = 1e-12
            Pc += r[6]; Lc += r[7]
            s = r[1:5]
        end
        @test all(>=(0), s)
        # VOC decays at k_OH[OH] + k_O3[O3] (O3 changes little over the hour)
        @test s[4] ≈ 5.0 * exp(-(L4 + k3 * 40.0) * 3600) rtol = 0.01
        # with NO available, each VOC oxidised by OH makes ν = 2 Ox: net production
        @test Pc ≈ 2 * (5.0 - s[4]) * L4 / (L4 + k3 * 40.0) rtol = 0.02
        @test s[3] + s[2] > s0[3] + s0[2]       # Ox rises: photochemical ozone production
        # RO2 conversion lifts the Leighton ratio above 1: φ = 1 + P_RO2 /(k1 NO O3)
        φ = leighton_ratio(s[1], s[2], s[3], J, k1)
        φ_ext = 1 + 2 * L4 * s[4] / (k1 * s[1] * s[3])
        @test φ > 1
        @test φ ≈ φ_ext rtol = 0.05

        # night: NO titrated; NOx then all NO2 and lost only to HNO3 at L5 (OH kept on)
        n0 = (0.0, 10.0, 40.0, 0.0, 0.0)
        s = n0
        for _ in 1:3600
            s = photochemistry_cell(s..., 10.0; J = 0.0, k1 = k1, L5 = L5)[1:5]
        end
        @test s[1] + s[2] ≈ 10.0 * exp(-L5 * 36000.0) rtol = 1e-9
        @test s[1] + s[2] + s[5] ≈ 10.0 rtol = 1e-13         # N conserved
        @test 1 / L5 / 3600 ≈ 5.3 rtol = 0.05                 # NOx lifetime 5 h at OH = 5e6

        # array driver: per-level T and p, and the returned budget sums
        A() = fill(0.0, 3, 2, 4)
        NO, NO2, O3, VOC, HNO3 = A() .+ 1, A() .+ 5, A() .+ 40, A() .+ 2, A()
        Ox0 = sum(O3) + sum(NO2); N0 = sum(NO) + sum(NO2)
        b = apply_photochemistry!(NO, NO2, O3, VOC, HNO3, 60.0; μ₀ = 0.9,
                                  T = [300.0, 299.0, 298.0, 297.0],
                                  p = [1.0e5, 0.99e5, 0.98e5, 0.97e5], OH = 5e6)
        @test sum(O3) + sum(NO2) - Ox0 ≈ b.P_Ox - b.L_Ox rtol = 1e-10
        @test N0 - sum(NO) - sum(NO2) ≈ b.HNO3_formed rtol = 1e-10
        @test sum(HNO3) ≈ b.HNO3_formed
        @test NO[1, 1, 1] != NO[1, 1, 4]                         # levels differ (T, p)
    end

    @testset "dry deposition: resistance model" begin
        # Sc of O3 at 20 °C ≈ 1, r_b(u* = 0.3) = 2/(κ u*) (Sc/Pr)^{2/3}
        @test schmidt_number(:O3) ≈ 1.0 rtol = 0.05
        @test schmidt_number(:NO) < schmidt_number(:O3) < schmidt_number(:HNO3)
        @test quasi_laminar_resistance(0.3; Sc = 1.0) ≈ 2 / (0.4 * 0.3) * (1 / 0.72)^(2 / 3)
        @test quasi_laminar_resistance(0.0; Sc = 1.0) == quasi_laminar_resistance(0.01; Sc = 1.0)
        # series resistances: v_d below each conductance, and v_d → 1/r_a when r_b, r_c → 0
        ra, rb = 50.0, quasi_laminar_resistance(0.2; Sc = schmidt_number(:O3))
        for sfc in (:vegetation, :water, :paved, :glass)
            vd = dry_deposition_velocity(ra, rb, surface_resistance(:O3, sfc))
            @test vd < 1 / ra && vd < 1 / surface_resistance(:O3, sfc)
        end
        @test dry_deposition_velocity(ra, 0.0, 0.0) == 1 / ra
        # O3 deposits fastest to vegetation (stomata), HNO3 is aerodynamically limited
        @test surface_resistance(:O3, :vegetation) < surface_resistance(:O3, :paved)
        # HNO3 (r_c = 10) is transport-limited: r_a + r_b dominate, and it
        # deposits faster than O3 on every surface
        rbH = quasi_laminar_resistance(0.2; Sc = schmidt_number(:HNO3))
        vH = dry_deposition_velocity(ra, rbH, surface_resistance(:HNO3, :paved))
        @test (ra + rbH) / (ra + rbH + 10) > 0.85
        for sfc in (:vegetation, :water, :paved, :glass)
            @test dry_deposition_velocity(ra, rbH, surface_resistance(:HNO3, sfc)) >
                  dry_deposition_velocity(ra, rb, surface_resistance(:O3, sfc))
        end
        @test surface_resistance(:VOC, :vegetation) == Inf
        @test dry_deposition_velocity(ra, rb, Inf) == 0
        @test_throws ArgumentError surface_resistance(:O3, :lava)

        # column sink: exact exponential, flux = mass removed, roof-aware level
        c = fill(40.0, 2, 2, 5)
        vd = [0.01 0.002; 0.0 0.005]
        kair = [1 3; 1 2]
        dz, dt = 10.0, 60.0
        removed = apply_dry_deposition!(c, vd, dt; dz = dz, k = kair)
        @test c[1, 1, 1] ≈ 40 * exp(-0.01 * 60 / 10)
        @test c[1, 2, 3] ≈ 40 * exp(-0.002 * 6) && c[1, 2, 1] == 40.0   # below the roof untouched
        @test c[2, 1, 1] == 40.0                                          # v_d = 0
        @test removed ≈ dz * (4 * 5 * 40.0 - sum(c)) / 1                  # column amount
        @test removed ≈ sum(40 * (1 - exp(-vd[I] * dt / dz)) * dz for I in eachindex(vd)) rtol = 1e-12
        # small-dt limit is the flux v_d c
        c1 = fill(40.0, 1, 1, 1)
        @test apply_dry_deposition!(c1, 0.005, 1e-3; dz = 10.0) ≈ 0.005 * 40 * 1e-3 rtol = 1e-6
    end

    @testset "wet deposition: rain scavenging" begin
        @test rain_scavenging_coefficient(:HNO3) * 1e-4 ≈ 1e-4       # Λ at ~2 mm/h
        @test rain_scavenging_coefficient(:O3) == 0
        c = fill(2.0, 3, 3, 3)
        qr = zeros(3, 3, 3); qr[2, 2, 1] = 1e-4; qr[1, 3, 2] = 5e-4
        s = rain_scavenging_coefficient(:HNO3)
        removed = apply_wet_deposition!(c, qr, 600.0; scav = s)
        @test removed > 0
        @test c[2, 2, 1] ≈ 2 * exp(-s * 1e-4 * 600)
        @test c[1, 3, 2] ≈ 2 * exp(-s * 5e-4 * 600)
        @test count(!=(2.0), c) == 2                                   # rain-free cells untouched
        @test removed ≈ 27 * 2.0 - sum(c) rtol = 1e-12                 # mass accounting
        @test apply_wet_deposition!(c, qr, 600.0; scav = 0.0) == 0     # insoluble: nothing
        @test_throws DimensionMismatch apply_wet_deposition!(c, zeros(2, 2, 2), 1.0; scav = 1.0)
        # KA-b (#296): the total is a running sum per plane in cell order (i
        # fastest), planes added in ascending k: thread-invariant, and the same
        # bits for plain arrays (linear indexing) and views (Cartesian)
        c0 = 1 .+ rand(Xoshiro(11), 13, 9, 7)
        qr = (rand(Xoshiro(12), size(c0)...) .< 0.3) .* 1e-4
        ref = 0.0
        cr = copy(c0)
        for k in axes(cr, 3)
            sk = 0.0
            for j in axes(cr, 2), i in axes(cr, 1)
                qr[i, j, k] > 0 || continue
                c1 = cr[i, j, k] * exp(-s * qr[i, j, k] * 600.0)
                sk += cr[i, j, k] - c1; cr[i, j, k] = c1
            end
            ref += sk
        end
        ca = copy(c0)
        ra = apply_wet_deposition!(ca, qr, 600.0; scav = s)
        @test ra === ref && ca == cr
        big = zeros(15, 11, 9); big[2:14, 2:10, 2:8] .= c0
        qbig = zeros(size(big)); qbig[2:14, 2:10, 2:8] .= qr
        rv = apply_wet_deposition!(view(big, 2:14, 2:10, 2:8), view(qbig, 2:14, 2:10, 2:8), 600.0; scav = s)
        @test rv === ref && big[2:14, 2:10, 2:8] == cr
    end
end
