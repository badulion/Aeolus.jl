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

    @testset "Rayleigh reflectivity = analytic gamma sixth moment" begin
        # Independent check: composite Simpson in D of ∫ D⁶ N(D) dD.
        function simpson(f, a, b, n)
            h = (b - a) / n
            s = f(a) + f(b)
            for i in 1:(n - 1)
                s += (isodd(i) ? 4 : 2) * f(a + i * h)
            end
            return s * h / 3
        end
        worst = 0.0
        for (N0, μ, λ) in ((8e6, 0.0, 2000.0), (1e9, 2.0, 5000.0), (3e11, 4.5, 1.2e4))
            Z = p3_rain_reflectivity(N0, μ, λ)
            Zq = 1e18 * simpson(D -> N0 * D^(μ + 6) * exp(-λ * D), 0.0, 60 / λ, 20_000)
            worst = max(worst, relerr(Z, Zq))
            @test Z ≈ 1e18 * N0 * gamma_lanczos(μ + 7) / λ^(μ + 7) rtol = 1e-13
        end
        @printf("    rain Z vs Simpson quadrature: max rel err %.1e\n", worst)
        @test worst < 1e-9

        # Ice: small solid spheres only (mean size 1 μm ≪ D_th) scatter
        # like water drops of the same size times |K_i|²/|K_w|².
        md = p3_mass_dimension(0.0, 400.0)
        N0, μ, λ = 1e14, 0.0, 1e6
        r = p3_ice_reflectivity(N0, μ, λ, md) / p3_rain_reflectivity(N0, μ, λ)
        @test r ≈ P3_K2_ICE / P3_K2_WATER rtol = 1e-12
        @test r ≈ 0.176 / 0.93 rtol = 1e-14

        # Ice across all regimes: analytic truncated-moment sum vs
        # quadrature of m(D)² — and the melting (bright band) ratio: the
        # same particles melted into drops are brighter by exactly
        # (ρ_ice/ρ_w)² |K_w|²/|K_i|², i.e. Z_ice/Z_melted = 0.2248 (-6.48 dB).
        worst = 0.0
        for (F, ρr) in ((0.0, 400.0), (0.4, 300.0), (0.9, 850.0)), λ in (800.0, 3000.0, 2e4)
            md = p3_mass_dimension(F, ρr)
            m2, = P3._p3_psd_expectation(0.0, λ, md; kmax = 6) do D
                m = p3_particle_mass(D, md)
                (m * m,)
            end
            N0 = 1e7
            M0 = gamma_psd_moment(N0, 0.0, λ, 0)
            Zi = p3_ice_reflectivity(N0, 0.0, λ, md)
            Zq = 1e18 * (P3_K2_ICE / P3_K2_WATER) * (6 / (π * md.params.ρ_ice))^2 * m2 * M0
            worst = max(worst, relerr(Zi, Zq))
            Zmelt = 1e18 * (6 / (π * ρw))^2 * m2 * M0
            @test Zi / Zmelt ≈ (P3_K2_ICE / P3_K2_WATER) * (ρw / md.params.ρ_ice)^2 rtol = 1e-9
        end
        @printf("    ice Z_e analytic vs quadrature: max rel err %.1e; Z_ice/Z_melted = %.2f dB\n",
                worst, 10log10((0.176 / 0.93) * (1000 / 917)^2))
        @test worst < 1e-10

        # Low-density snow is a weak target: unrimed ice vs graupel with
        # the same PSD.
        @test p3_ice_reflectivity(1e7, 0.0, 2000.0, p3_mass_dimension(0.0, 400.0)) <
              p3_ice_reflectivity(1e7, 0.0, 2000.0, p3_mass_dimension(1.0, 900.0))

        @test p3_dbz(1.0) == 0.0
        @test p3_dbz(1e4) ≈ 40.0
        @test p3_dbz(0.0) == -100.0
    end

    @testset "Rain rate and dBZ: Marshall-Palmer" begin
        # Marshall-Palmer DSD N0 = 8e6 m⁻⁴ (0.08 cm⁻⁴), exponential. Solve
        # for the slope whose rain rate — computed from our own fall
        # speeds, not MP's λ(R) fit — is 5 mm/h, and compare the dBZ with
        # the MP Z-R relation Z = 200 R^1.6.
        N0 = 8e6
        function λ_for(R)
            lo, hi = 500.0, 2e4
            for _ in 1:200
                mid = sqrt(lo * hi)
                p3_rain_rate(N0, 0.0, mid) > R ? (lo = mid) : (hi = mid)
            end
            return sqrt(lo * hi)
        end
        λ5 = λ_for(5.0)
        @test p3_rain_rate(N0, 0.0, λ5) ≈ 5.0 rtol = 1e-10
        dbz5 = p3_dbz(p3_rain_reflectivity(N0, 0.0, λ5))
        mp5 = p3_dbz(200 * 5.0^1.6)
        @printf("    5 mm/h: λ = %.0f /m, %.2f dBZ (Marshall-Palmer Z-R: %.2f dBZ)\n",
                λ5, dbz5, mp5)
        # Note: 200·5^1.6 = 2627 mm⁶/m³ = 34.2 dBZ, not ≈ 37 dBZ.
        @test abs(dbz5 - mp5) < 1.0
        @test 30 < dbz5 < 40
        for R in (1.0, 10.0, 25.0, 50.0)
            d = p3_dbz(p3_rain_reflectivity(N0, 0.0, λ_for(R))) - p3_dbz(200 * R^1.6)
            @printf("    %4.0f mm/h: dBZ - MP = %+.2f dB\n", R, d)
            @test abs(d) < 1.5
        end
        # Rain rate is V_M times the water volume flux.
        _, VM, _ = p3_rain_bulk_fall_speed(0.0, λ5)
        @test p3_rain_rate(N0, 0.0, λ5) ≈
              π / 6 * gamma_psd_moment(N0, 0.0, λ5, 3) * VM * 3.6e6 rtol = 1e-14
        # Air-density correction: at 600 hPa rain falls faster.
        @test p3_rain_rate(N0, 0.0, λ5; ρ_air = 0.7) > p3_rain_rate(N0, 0.0, λ5)

        # PSD from (q_r, N_r) reproduces the moments.
        q, N = 1e-3, 5e3
        N0r, μr, λr = p3_rain_size_distribution(q, N)
        @test gamma_psd_moment(N0r, μr, λr, 0) ≈ N rtol = 1e-12
        @test π / 6 * ρw * gamma_psd_moment(N0r, μr, λr, 3) ≈ q rtol = 1e-12
        @test p3_rain_size_distribution(0.0, 1.0) === nothing
        # Mean-size limiter: N_r → tiny caps the mean diameter at 0.8 mm,
        # and p3_rain_limit_number gives the number that makes the PSD
        # consistent with that limit, keeping the mass.
        _, μl, λl = p3_rain_size_distribution(1e-3, 1e-3)
        @test (μl + 1) / λl ≈ 8e-4 rtol = 1e-14
        for μr in (0.0, 3.0)
            Nl = p3_rain_limit_number(1e-3, 1e-3; μ_r = μr)
            N0l, μl, λl = p3_rain_size_distribution(1e-3, Nl; μ_r = μr)
            @test (μl + 1) / λl ≈ 8e-4 rtol = 1e-12
            @test π / 6 * ρw * gamma_psd_moment(N0l, μl, λl, 3) ≈ 1e-3 rtol = 1e-12
        end
        @test p3_rain_limit_number(1e-3, 5e3) == 5e3      # inside limits: unchanged
        @test p3_rain_limit_number(1e-3, 0.0) > 0         # mass without number
        @test p3_rain_limit_number(0.0, 7.0) == 7.0
    end

    @testset "Column scheme: exact shift, conservation, positivity" begin
        # Constant speed with C = 1 per sub-step: upwind is an exact shift.
        nz = 40
        ψ = zeros(nz); ψ[30:32] .= (1.0, 2.0, 3.0)
        ρ = ones(nz)
        surf, nsub = p3_sediment!((ψ,), f -> (fill(5.0, nz),), ρ, 10.0, 20.0)
        @test nsub == 10
        @test ψ[20:22] == [1.0, 2.0, 3.0]
        @test count(!iszero, ψ) == 3
        @test surf[1] == 0

        # Varying ρ and Δz, speeds up to 9 m/s, dt = 900 s: Courant ≫ 1.
        nz = 60
        dz = 50.0 .+ 30.0 .* (1 .+ sin.(1.7 .* (1:nz)))    # 50-110 m, irregular
        z = cumsum(dz) .- dz ./ 2
        ρ = 1.2 .* exp.(-z ./ 8000)
        q = [3000 < zk < 4500 ? 2e-3 : 0.0 for zk in z]
        m0 = sum(ρ .* q .* dz)
        vel = f -> (9.0 .* ones(nz),)
        Cmax = 9.0 * 900 / minimum(dz)
        surf, nsub = p3_sediment!((q,), vel, ρ, dz, 900.0)
        m1 = sum(ρ .* q .* dz)
        cons = abs(m1 + surf[1] - m0) / m0
        @printf("    constant V: C_max(no substep) = %.0f, %d sub-steps, surface %.3f kg/m², mass err %.1e\n",
                Cmax, nsub, surf[1], cons)
        @test nsub >= ceil(Cmax)
        @test cons < 1e-14
        @test minimum(q) >= 0
        @test surf[1] > 0

        @test_throws ArgumentError p3_sediment!((copy(q),), vel, ρ, dz, 1.0; courant_max = 1.5)
        @test_throws DomainError p3_sediment!((copy(q),), f -> (-ones(nz),), ρ, dz, 1.0)
    end

    @testset "Rain shaft: conservation, positivity, size sorting" begin
        nz = 100
        dz = 50.0
        z = ((1:nz) .- 0.5) .* dz
        ρ = 1.2 .* exp.(-z ./ 8500)
        # Rain layer 3.5-4.5 km: q = 1 g/kg, N = 1e4 /kg (mass-mean D ≈ 0.6 mm).
        inlayer = (z .> 3500) .& (z .< 4500)
        q = ifelse.(inlayer, 1e-3, 0.0)
        N = ifelse.(inlayer, 1e4, 0.0)
        mq0, mN0 = sum(ρ .* q .* dz), sum(ρ .* N .* dz)
        Dm(qk, Nk) = (6 * qk / (π * ρw * Nk))^(1 / 3)
        D0 = Dm(1e-3, 1e4)

        # One big step: ~8 m/s × 300 s / 50 m ≈ 50 without sub-stepping.
        # Without the number limiter both moments are conserved.
        qa, Na = copy(q), copy(N)
        Pq, PN, nsub = p3_sediment_rain!(qa, Na, ρ, dz, 300.0; limit_number = false)
        eq = abs(sum(ρ .* qa .* dz) + Pq - mq0) / mq0
        eN = abs(sum(ρ .* Na .* dz) + PN - mN0) / mN0
        @printf("    rain shaft, 300 s, no limiter: %d sub-steps, mass err %.1e, number err %.1e\n",
                nsub, eq, eN)
        @test nsub > 30
        @test eq < 1e-13 && eN < 1e-13
        @test minimum(qa) >= 0 && minimum(Na) >= 0
        # ... but the leading edge is left with mass and almost no number:
        # unbounded diagnosed drop size (excessive two-moment size sorting).
        Dmax_free = maximum(Dm(qa[k], Na[k]) for k in 1:nz if qa[k] > 1e-8)
        @printf("    unlimited: largest diagnosed D_m = %.2g m\n", Dmax_free)
        @test Dmax_free > 1e-2

        # Default (P3): number limited to mean size ≤ 0.8 mm; mass still exact.
        Pq, PN, nsub = p3_sediment_rain!(q, N, ρ, dz, 300.0)
        eq = abs(sum(ρ .* q .* dz) + Pq - mq0) / mq0
        @printf("    rain shaft, 300 s, limited: %d sub-steps, surface %.2e mm, mass err %.1e\n",
                nsub, Pq, eq)
        @test eq < 1e-13
        @test minimum(q) >= 0 && minimum(N) >= 0
        Dcap = cbrt(gamma_lanczos(4.0)) * 8e-4          # mass-mean D at the limit, μ = 0
        Dall = [Dm(q[k], N[k]) for k in 1:nz if q[k] > 1e-10]
        @test maximum(Dall) <= Dcap * (1 + 1e-12)

        # Size sorting: large drops lead, small ones are left behind. D_m
        # sampled every 400 m from the leading edge up to the top of the
        # shaft decreases monotonically.
        k_lo = findfirst(q .> 0.05 * maximum(q))
        k_hi = findlast(q .> 1e-3 * maximum(q))
        ks = k_lo:8:k_hi
        Dprof = [Dm(q[k], N[k]) for k in ks]
        @printf("    D_m from edge (z = %.0f m) upward, every 400 m: %s μm (initial %.0f)\n",
                z[k_lo], join(round.(Int, 1e6 .* Dprof), ", "), 1e6 * D0)
        @test length(Dprof) >= 5
        @test all(diff(Dprof) .< 0)
        @test Dprof[1] > 2 * D0                 # leading edge: large drops
        @test Dm(q[k_hi], N[k_hi]) < D0 / 2     # top: small drops left behind
        # Front position: ~2.4 km of fall in 300 s ⇒ V_M ≈ 8 m/s at the edge.
        @test 800 < z[k_lo] < 1600

        # Monodisperse rain (μ_r → ∞) does not sort: D_m stays uniform.
        q2 = ifelse.(inlayer, 1e-3, 0.0)
        N2 = ifelse.(inlayer, 1e4, 0.0)
        p3_sediment_rain!(q2, N2, ρ, dz, 120.0; μ_r = 2000.0)
        sel = q2 .> 1e-3 * maximum(q2)
        Ds = [Dm(q2[k], N2[k]) for k in findall(sel)]
        spread_mono = (maximum(Ds) - minimum(Ds)) / D0
        @printf("    monodisperse shaft: D_m spread %.1e (relative)\n", spread_mono)
        @test spread_mono < 0.02
    end

    @testset "Ice column: conservation, rime carried with mass, sorting" begin
        nz = 80
        dz = 100.0
        z = ((1:nz) .- 0.5) .* dz
        ρ = 1.2 .* exp.(-z ./ 8000)
        Tair = 288.0 .- 6.5e-3 .* z
        lay = (z .> 5000) .& (z .< 6000)
        qi = ifelse.(lay, 5e-4, 0.0)
        Frim = 0.4
        qrim = Frim .* qi
        Brim = qrim ./ 500.0
        Ni = ifelse.(lay, 5e4, 0.0)
        m0 = [sum(ρ .* f .* dz) for f in (qi, qrim, Brim, Ni)]
        out = p3_sediment_ice!(qi, qrim, Brim, Ni, ρ, dz, 1200.0; T_air = Tair)
        nsub = out[5]
        errs = [abs(sum(ρ .* f .* dz) + out[i] - m0[i]) / m0[i]
                for (i, f) in enumerate((qi, qrim, Brim, Ni))]
        @printf("    ice column, 1200 s: %d sub-steps, max conservation err %.1e\n",
                nsub, maximum(errs))
        @test maximum(errs) < 1e-13
        @test all(minimum.((qi, qrim, Brim, Ni)) .>= 0)
        # q_rim and B_rim fall with q_i, so F_rim and ρ_rim are carried
        # unchanged wherever there is ice.
        has = qi .> 1e-10
        @test count(has) > 10
        @test maximum(abs.(qrim[has] ./ qi[has] .- Frim)) < 1e-12
        @test maximum(abs.(qrim[has] ./ Brim[has] .- 500.0)) < 1e-9
        # Size sorting: mean mass q/N largest at the leading edge.
        k_lo = findfirst(qi .> 1e-3 * maximum(qi))
        mm = [qi[k] / Ni[k] for k in k_lo:(k_lo + 3)]
        @printf("    ice leading edge: q/N (bottom→up) = %s × initial\n",
                join(round.(mm ./ 1e-8, digits = 2), ", "))
        @test all(diff(mm) .< 0)
        @test mm[1] > 1e-8
    end
end
