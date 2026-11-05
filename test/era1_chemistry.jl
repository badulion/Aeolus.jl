# Reduced urban NOx–O3–VOC mechanism, dry and wet deposition
# (src/Chemistry.jl, #135).

using Test
using Aeolus
using Aeolus.Chemistry

# Classical RK4 reference for the triad R1–R2 (tiny steps).
function triad_rk4(NO, NO2, O3, t, J, k; n = 20_000)
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
        @test k_VOC_OH(298.0) ≈ 1.0e-10 rtol = 0.02          # isoprene + OH, IUPAC
        # falloff: low-pressure limit linear in M, high-pressure limit k∞
        @test k_NO2_OH(298.0, 1e8) / k_NO2_OH(298.0, 2e8) ≈ 0.5 rtol = 1e-3
        @test k_NO2_OH(298.0, 1e40) ≈ 2.8e-11 rtol = 0.005
        @test ppb_rate_constant(k_NO_O3(298.15), 101325.0, 298.15) ≈ 4.8e-4 rtol = 0.01
        @test 8.8e-3 < j_NO2(1.0) < 9.0e-3 && j_NO2(-0.3) == 0
    end

    @testset "triad: exact solution, conservation, photostationary state" begin
        J, k = 8e-3, ppb_rate_constant(k_NO_O3(298.0), 101325.0, 298.0)
        # against a fine RK4 integration (RK4 error ~ (λh)^4 with λh ~ 3e-4: negligible)
        ref = triad_rk4(30.0, 10.0, 25.0, 300.0, J, k)
        @test all(isapprox.(nox_triad(30.0, 10.0, 25.0, 300.0, J, k), ref; rtol = 1e-9))
        # any dt: NOx and Ox invariants to round-off, positive
        a, b, c = nox_triad(0.5, 5.0, 40.0, 1e7, J, k)
        @test a + b ≈ 5.5 rtol = 1e-14
        @test c + b ≈ 45.0 rtol = 1e-14
        @test min(a, b, c) >= 0
        # relaxes to the photostationary state (Leighton ratio 1)
        y = photostationary_NO2(5.5, 45.0, J, k)
        a, b, c = nox_triad(3.0, 2.5, 42.5, 3600.0, J, k)
        @test leighton_ratio(a, b, c, J, k) ≈ 1 rtol = 1e-10
        @test b ≈ y rtol = 1e-10
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
        Pc = 0.0; Lc = 0.0; eOx = 0.0; eN = 0.0
        for _ in 1:1800                         # 1 h at dt = 2 s
            r = photochemistry_cell(s..., 2.0; J = J, k1 = k1, k3 = k3, L4 = L4, L5 = L5)
            eOx = max(eOx, abs((r[3] + r[2]) - (s[3] + s[2]) - (r[6] - r[7])))
            eN = max(eN, abs((r[1] + r[2]) - (s[1] + s[2]) + (r[5] - s[5])))
            Pc += r[6]; Lc += r[7]
            s = r[1:5]
        end
        @test eOx < 1e-12 && eN < 1e-12          # exact per-step Ox and N budgets
        @test all(>=(0), s)
        @test s[4] ≈ 5.0 * exp(-(L4 + k3 * 40.0) * 3600) rtol = 0.01
        @test Pc ≈ 2 * (5.0 - s[4]) * L4 / (L4 + k3 * 40.0) rtol = 0.02
        φ = leighton_ratio(s[1], s[2], s[3], J, k1)
        @test φ ≈ 1 + 2 * L4 * s[4] / (k1 * s[1] * s[3]) rtol = 0.05

        # array driver: returned budget sums
        A() = fill(0.0, 3, 2, 4)
        NO, NO2, O3, VOC, HNO3 = A() .+ 1, A() .+ 5, A() .+ 40, A() .+ 2, A()
        Ox0 = sum(O3) + sum(NO2); N0 = sum(NO) + sum(NO2)
        b = apply_photochemistry!(NO, NO2, O3, VOC, HNO3, 60.0; μ₀ = 0.9,
                                  T = [300.0, 299.0, 298.0, 297.0],
                                  p = [1.0e5, 0.99e5, 0.98e5, 0.97e5], OH = 5e6)
        @test sum(O3) + sum(NO2) - Ox0 ≈ b.P_Ox - b.L_Ox rtol = 1e-10
        @test N0 - sum(NO) - sum(NO2) ≈ b.HNO3_formed rtol = 1e-10
    end

    @testset "dry deposition: resistance model" begin
        @test quasi_laminar_resistance(0.3; Sc = 1.0) ≈ 2 / (0.4 * 0.3) * (1 / 0.72)^(2 / 3)
        ra = 50.0
        @test dry_deposition_velocity(ra, 0.0, 0.0) == 1 / ra
        @test surface_resistance(:O3, :vegetation) < surface_resistance(:O3, :paved)
        @test_throws ArgumentError surface_resistance(:O3, :lava)
        # column sink: exact exponential, roof-aware level, flux = mass removed
        c = fill(40.0, 2, 2, 5)
        vd = [0.01 0.002; 0.0 0.005]
        kair = [1 3; 1 2]
        removed = apply_dry_deposition!(c, vd, 60.0; dz = 10.0, k = kair)
        @test c[1, 1, 1] ≈ 40 * exp(-0.01 * 60 / 10)
        @test c[1, 2, 1] == 40.0                                  # below the roof untouched
        @test removed ≈ 10.0 * (4 * 5 * 40.0 - sum(c)) rtol = 1e-12
    end

    @testset "wet deposition: rain scavenging" begin
        c = fill(2.0, 3, 3, 3)
        qr = zeros(3, 3, 3); qr[2, 2, 1] = 1e-4; qr[1, 3, 2] = 5e-4
        s = rain_scavenging_coefficient(:HNO3)
        removed = apply_wet_deposition!(c, qr, 600.0; scav = s)
        @test c[2, 2, 1] ≈ 2 * exp(-s * 1e-4 * 600)
        @test count(!=(2.0), c) == 2                                   # rain-free cells untouched
        @test removed ≈ 27 * 2.0 - sum(c) rtol = 1e-12                 # mass accounting
        @test rain_scavenging_coefficient(:O3) == 0
    end
end
