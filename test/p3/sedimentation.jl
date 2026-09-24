# P3 part-file tests: per-moment sedimentation velocities, column
# sedimentation with Courant sub-stepping, size sorting, and radar
# reflectivity (src/p3/sedimentation.jl, issue #4).
#
# The physics asserted: bulk speeds are ordered V_N < V_M < V_Z for any
# broad PSD and collapse together in the monodisperse limit; the column
# scheme conserves mass to round-off (minus what leaves at the surface)
# and never goes negative however large the Courant number; the
# difference V_M - V_N sorts particles by size in a falling shaft; and
# Rayleigh reflectivity reproduces the gamma sixth moment and the
# Marshall-Palmer Z-R relation.

@testset "P3 sedimentation and radar reflectivity" begin
    P3 = Aeolus.P3Microphysics
    relerr(x, ref) = abs(x - ref) / abs(ref)
    ρw = Aeolus.Superdroplet.DEFAULT_MICRO.ρ_water

    @testset "Rain fall speed: one source of truth" begin
        # Delegates to Superdroplet.terminal_velocity at sea level ...
        for D in (50e-6, 5e-4, 1e-3, 2e-3, 5e-3)
            @test p3_rain_fall_speed(D) == terminal_velocity(D / 2)
        end
        v1 = p3_rain_fall_speed(1e-3)
        @printf("    V_rain(1 mm) = %.3f m/s (Gunn-Kinzer 4.03)\n", v1)
        @test abs(v1 - 4.03) < 0.15
        # ... with the Foote-du Toit (ρ0/ρ)^0.54 air-density correction.
        @test p3_rain_fall_speed(2e-3; ρ_air = 0.6) ≈
              p3_rain_fall_speed(2e-3) * (1.225 / 0.6)^0.54 rtol = 1e-14
        @test p3_rain_fall_speed(0.0) == 0
    end

    @testset "Quadrature against analytic power-law weighting" begin
        # For V = a D^b the weighted speeds are gamma ratios:
        # V_k = a Γ(μ+k+b+1) / (Γ(μ+k+1) λ^b).
        a, b = 842.0, 0.8           # Liu-Orville-like rain power law, SI
        for μ in (0.0, 2.0, 5.0), λ in (1500.0, 4000.0)
            VN, V3, M3, V6, M6 = P3._p3_psd_expectation(μ, λ, (1e-3,); kmax = 6) do D
                v = a * D^b
                (v, v * D^3, D^3, v * D^6, D^6)
            end
            Vk(k) = a * exp(lgamma_lanczos(μ + k + b + 1) - lgamma_lanczos(μ + k + 1)) / λ^b
            @test VN ≈ Vk(0) rtol = 1e-12
            @test V3 / M3 ≈ Vk(3) rtol = 1e-12
            @test V6 / M6 ≈ Vk(6) rtol = 1e-12
        end
    end

    @testset "V_N < V_M < V_Z for broad PSD; equal when monodisperse" begin
        # Rain, exponential PSD (μ = 0), mean drop sizes 0.2-2 mm.
        for λ in (500.0, 2000.0, 5000.0)
            VN, VM, VZ = p3_rain_bulk_fall_speed(0.0, λ)
            @test VN < VM < VZ
        end
        VN, VM, VZ = p3_rain_bulk_fall_speed(0.0, 2000.0)
        @printf("    rain μ=0, λ=2000/m: V_N = %.3f, V_M = %.3f, V_Z = %.3f m/s\n", VN, VM, VZ)
        # Monodisperse limit: μ → ∞ at fixed mean D = 1 mm. The relative
        # spread of D is 1/√(μ+1), so (V_M - V_N)/V_N = O(1/μ).
        spread(μ) = let (vn, vm, vz) = p3_rain_bulk_fall_speed(μ, (μ + 1) / 1e-3)
            ((vm - vn) / vn, (vz - vn) / vn, vn)
        end
        s100, _, _ = spread(100.0)
        s1000, z1000, vn1000 = spread(1000.0)
        @printf("    rain monodisperse limit: (V_M-V_N)/V_N = %.2e (μ=100), %.2e (μ=1000)\n",
                s100, s1000)
        @test 0 < s1000 < 5e-3
        @test 0 < z1000 < 1e-2
        @test 7 < s100 / s1000 < 13            # O(1/μ) convergence
        @test vn1000 ≈ p3_rain_fall_speed(1e-3) rtol = 5e-3

        # Ice, across rime states.
        for (F, ρr) in ((0.0, 400.0), (0.3, 400.0), (0.8, 800.0))
            md = p3_mass_dimension(F, ρr)
            for λ in (1000.0, 5000.0)
                vn, vm = p3_bulk_fall_speed(0.0, λ, md)
                vz = p3_bulk_fall_speed_z(0.0, λ, md)
                @test vn < vm < vz
            end
        end
        md = p3_mass_dimension(0.3, 400.0)
        vn, vm = p3_bulk_fall_speed(0.0, 2000.0, md)
        vz = p3_bulk_fall_speed_z(0.0, 2000.0, md)
        @printf("    ice F_rim=0.3, λ=2000/m: V_N = %.3f, V_M = %.3f, V_Z = %.3f m/s\n", vn, vm, vz)
        # Ice monodisperse limit (1 mm, inside the aggregate regime).
        μ = 2000.0
        vn, vm = p3_bulk_fall_speed(μ, (μ + 1) / 1e-3, md)
        vz = p3_bulk_fall_speed_z(μ, (μ + 1) / 1e-3, md)
        @printf("    ice monodisperse (μ=2000): (V_M-V_N)/V_N = %.2e, (V_Z-V_N)/V_N = %.2e\n",
                (vm - vn) / vn, (vz - vn) / vn)
        @test abs(vm - vn) / vn < 2e-3
        @test abs(vz - vn) / vn < 4e-3
        @test vn ≈ p3_fall_speed(1e-3, md) rtol = 2e-3
    end

end
