# P3 part-file tests: per-moment sedimentation velocities, column
# sedimentation with Courant sub-stepping, size sorting, and radar
# reflectivity (src/p3/sedimentation.jl, issue #4).

@testset "P3 sedimentation and radar reflectivity" begin
    P3 = Aeolus.P3Microphysics
    relerr(x, ref) = abs(x - ref) / abs(ref)
    ρw = Aeolus.Superdroplet.DEFAULT_MICRO.ρ_water

    @testset "Fall speeds: one source of truth, power-law moments, ordering" begin
        @test p3_rain_fall_speed(1e-3) == terminal_velocity(0.5e-3)
        @test abs(p3_rain_fall_speed(1e-3) - 4.03) < 0.15           # Gunn-Kinzer
        @test p3_rain_fall_speed(2e-3; ρ_air = 0.6) ≈
              p3_rain_fall_speed(2e-3) * (1.225 / 0.6)^0.54 rtol = 1e-14   # Foote-du Toit
        # V = a D^b: V_k = a Γ(μ+k+b+1)/(Γ(μ+k+1) λ^b)
        a, b, μ, λ = 842.0, 0.8, 2.0, 4000.0
        VN, V6, M6 = P3._p3_psd_expectation(μ, λ, (1e-3,); kmax = 6) do D
            (a * D^b, a * D^(b + 6), D^6)
        end
        Vk(k) = a * exp(lgamma_lanczos(μ + k + b + 1) - lgamma_lanczos(μ + k + 1)) / λ^b
        @test VN ≈ Vk(0) rtol = 1e-12
        @test V6 / M6 ≈ Vk(6) rtol = 1e-12
        VN, VM, VZ = p3_rain_bulk_fall_speed(0.0, 2000.0)
        @test VN < VM < VZ
        # monodisperse limit (μ → ∞, mean D 1 mm): spread O(1/μ)
        vn, vm, _ = p3_rain_bulk_fall_speed(1000.0, 1001 / 1e-3)
        @test 0 < (vm - vn) / vn < 5e-3
    end

    @testset "Rayleigh reflectivity and Marshall-Palmer Z-R" begin
        N0, μ, λ = 1e9, 2.0, 5000.0
        @test p3_rain_reflectivity(N0, μ, λ) ≈ 1e18 * N0 * gamma_lanczos(μ + 7) / λ^(μ + 7) rtol = 1e-13
        # small solid ice spheres scatter as drops × |K_i|²/|K_w|²
        @test p3_ice_reflectivity(1e14, 0.0, 1e6, p3_mass_dimension(0.0, 400.0)) /
              p3_rain_reflectivity(1e14, 0.0, 1e6) ≈ 0.176 / 0.93 rtol = 1e-12
        # MP rain, slope solved from our own fall speeds for R = 5 mm/h.
        # (Issue #4 asked for 35-45 dBZ: wrong; MP gives 200·5^1.6 = 34.2 dBZ.)
        N0 = 8e6
        lo, hi = 500.0, 2e4
        for _ in 1:80
            mid = sqrt(lo * hi)
            p3_rain_rate(N0, 0.0, mid) > 5.0 ? (lo = mid) : (hi = mid)
        end
        λ5 = sqrt(lo * hi)
        @test abs(p3_dbz(p3_rain_reflectivity(N0, 0.0, λ5)) - p3_dbz(200 * 5.0^1.6)) < 1.0
        # PSD from (q_r, N_r) reproduces the moments
        N0r, μr, λr = p3_rain_size_distribution(1e-3, 5e3)
        @test gamma_psd_moment(N0r, μr, λr, 0) ≈ 5e3 rtol = 1e-12
        @test π / 6 * ρw * gamma_psd_moment(N0r, μr, λr, 3) ≈ 1e-3 rtol = 1e-12
    end

    @testset "Rain shaft: conservation, mean-size limit, size sorting" begin
        nz, dz = 100, 50.0
        z = ((1:nz) .- 0.5) .* dz
        ρ = 1.2 .* exp.(-z ./ 8500)
        inlayer = (z .> 3500) .& (z .< 4500)
        q = ifelse.(inlayer, 1e-3, 0.0); N = ifelse.(inlayer, 1e4, 0.0)
        mq0 = sum(ρ .* q .* dz)
        Dm(qk, Nk) = (6 * qk / (π * ρw * Nk))^(1 / 3)
        D0 = Dm(1e-3, 1e4)
        Pq, _, nsub = p3_sediment_rain!(q, N, ρ, dz, 300.0)
        @test nsub > 30
        @test abs(sum(ρ .* q .* dz) + Pq - mq0) / mq0 < 1e-13
        @test minimum(q) >= 0 && minimum(N) >= 0
        # mean size ≤ 2 mm (mass-mean D at the limit, μ = 0)
        Dcap = cbrt(gamma_lanczos(4.0)) * P3_RAIN_D_MEAN_MAX
        @test maximum(Dm(q[k], N[k]) for k in 1:nz if q[k] > 1e-10) <= Dcap * (1 + 1e-12)
        # large drops lead, small ones are left behind
        k_lo = findfirst(q .> 0.05 * maximum(q)); k_hi = findlast(q .> 1e-3 * maximum(q))
        Dprof = [Dm(q[k], N[k]) for k in k_lo:8:k_hi]
        @test length(Dprof) >= 5 && all(diff(Dprof) .< 0)
        @test Dprof[1] > 2D0
    end

    @testset "Ice column: conservation, rime carried with mass" begin
        nz, dz = 80, 100.0
        z = ((1:nz) .- 0.5) .* dz
        ρ = 1.2 .* exp.(-z ./ 8000)
        lay = (z .> 5000) .& (z .< 6000)
        qi = ifelse.(lay, 5e-4, 0.0); qrim = 0.4 .* qi; Brim = qrim ./ 500.0
        Ni = ifelse.(lay, 5e4, 0.0)
        m0 = [sum(ρ .* f .* dz) for f in (qi, qrim, Brim, Ni)]
        out = p3_sediment_ice!(qi, qrim, Brim, Ni, ρ, dz, 1200.0; T_air = 288.0 .- 6.5e-3 .* z)
        @test maximum(abs(sum(ρ .* f .* dz) + out[i] - m0[i]) / m0[i]
                      for (i, f) in enumerate((qi, qrim, Brim, Ni))) < 1e-13
        has = qi .> 1e-10
        @test count(has) > 10
        @test maximum(abs.(qrim[has] ./ qi[has] .- 0.4)) < 1e-12      # F_rim carried
        @test maximum(abs.(qrim[has] ./ Brim[has] .- 500.0)) < 1e-9   # ρ_rim carried
    end
end
