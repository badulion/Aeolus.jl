using Test
using Aeolus
using Aeolus.Grids: CellCenter
using Aeolus.RadiationCore
using Aeolus.SolarGeometry

@testset "Solar geometry & cloud shadows (Phase 38)" begin

    # NREL SPA reference case (Reda & Andreas 2004, Table A5.1):
    #   2003-10-17 19:30:30 UTC, lon -105.1786°, lat 39.742476°, 1830.14 m,
    #   820 mbar, 11 °C ⇒ topocentric zenith 50.11162°, azimuth 194.34024°.
    # The Meeus-grade implementation targets ~0.01°.
    @testset "NREL SPA benchmark case" begin
        p = solar_position(solar_time(2003, 10, 17, 19, 30, 30.0;
                latitude = 39.742476, longitude = -105.1786, elevation = 1830.14,
                pressure = 820.0, temperature = 11.0, delta_t = 67.0))
        @test abs(p.zenith - 50.11162) < 0.02
        @test abs(p.azimuth - 194.34024) < 0.02
        @test 0.99 < p.distance_au < 1.00
    end

    @testset "Declination, equation of time, distance" begin
        δ(y, mo, d, h, mi) = solar_declination_accurate(solar_time(y, mo, d, h, mi, 0.0; latitude = 0.0, longitude = 0.0))
        @test abs(δ(2020, 6, 20, 21, 43) - 23.436) < 0.02      # June solstice
        @test abs(δ(2020, 12, 21, 10, 2) + 23.436) < 0.02      # December solstice
        @test abs(δ(2020, 3, 20, 3, 50)) < 0.02                # March equinox
        eot(m, d) = equation_of_time(solar_time(2020, m, d, 12, 0, 0.0; latitude = 0.0, longitude = 0.0))
        @test -15 < eot(2, 11) < -13 && 15 < eot(11, 3) < 17.5   # ≈ −14.2 and +16.4 min
        f(m, d) = earth_sun_distance_factor(solar_position(solar_time(2020, m, d, 12, 0, 0.0; latitude = 0.0, longitude = 0.0)))
        @test 0.06 < f(1, 4) - f(7, 5) < 0.08                   # ±3.4 % in irradiance
        @test 1305 < extraterrestrial_irradiance(solar_position(solar_time(2020, 7, 5, 12, 0, 0.0; latitude = 0.0, longitude = 0.0))) < 1325
        # polar night: μ₀ clamped to 0, infinite air mass
        pn = solar_position(solar_time(2020, 12, 21, 12, 0, 0.0; latitude = 89.9, longitude = 0.0))
        @test pn.elevation < 0 && pn.μ₀ == 0 && pn.air_mass == Inf
    end

    @testset "Refraction and air mass" begin
        @test 0.45 < refraction_bennett(0.0) < 0.65      # ≈ 34 arcmin at the horizon
        @test refraction_bennett(90.0) < 0.002
        @test isapprox(air_mass(90.0), 1.0; atol = 1e-3)
        @test air_mass(30.0) ≈ 2.0 atol = 0.02           # AM2
        @test 35 < air_mass(0.0) < 42                    # Kasten-Young, finite at the horizon
    end

    @testset "Sun direction and diurnal cycle in Berlin" begin
        p = solar_position(solar_time(2020, 6, 21, 12, 0, 0.0; latitude = 52.0, longitude = 0.0))
        e, n, u = sun_direction_enu(p)
        @test isapprox(e^2 + n^2 + u^2, 1.0; atol = 1e-12) && u > 0 && n < 0
        @test sun_direction_enu(solar_position(solar_time(2020, 6, 21, 6, 0, 0.0; latitude = 52.0, longitude = 0.0)))[1] > 0
        elevs = [solar_position(solar_time(2020, 6, 21, h, 0, 0.0; latitude = 52.522, longitude = 13.41)).elevation for h in 0:23]
        @test abs(maximum(elevs) - (90 - 52.522 + 23.44)) < 0.5
        @test argmax(elevs) - 1 in (10, 11) && count(>(0), elevs) >= 16
    end

    # Bird clear-sky: plausible summer-noon magnitudes, GHI = DNI·μ₀ + DHI, and
    # Erbs/Reindl partitioning recovers the GHI it was given.
    @testset "Clear sky and partitioning" begin
        p = solar_position(solar_time(2020, 6, 21, 12, 0, 0.0; latitude = 40.0, longitude = 0.0))
        r = clear_sky_irradiance(p, BirdClearSky(; aod380 = 0.10, aod500 = 0.07,
                                 precipitable_water = 1.4, ozone = 0.3, albedo = 0.2))
        @test 800 < r.dni < 1050 && 850 < r.ghi < 1150 && 40 < r.dhi < 200
        @test isapprox(r.ghi, r.dni * p.μ₀ + r.dhi; rtol = 1e-10)
        @test issorted([clear_sky_irradiance(p, BirdClearSky(; aod500 = a)).dni for a in (0.05, 0.40)]; rev = true)
        @test diffuse_fraction_erbs(0.9) == 0.165 && all(diff(diffuse_fraction_erbs.(0.25:0.1:0.75)) .< 0)
        @test all((:erbs, :reindl)) do meth
            q = partition_global_irradiance(r.ghi, p; method = meth)
            isapprox(q.dni * p.μ₀ + q.dhi, r.ghi; rtol = 1e-10) && q.dni >= 0 && q.dhi >= 0
        end
    end

    # 3-D cloud shadow: displaced away from the sun (what no independent-column
    # scheme can do), bounded, and transparent without cloud.
    @testset "3-D cloud shadow transmittance" begin
        N = 16; mesh = RegularGrid(; size = (N, N, 8), extent = (500.0, 500.0, 250.0), halo = (1, 1, 1))
        τ = zeros(Float64, N, N, 8)
        τ[7:10, 7:10, 3:4] .= 2.0                       # low thick cloud, z ≈ 62-125 m
        @test all(cloud_shadow_map(τ, mesh, (0.0, 0.0, 1.0))[7:10, 7:10] .< 0.5)
        t = cloud_shadow_map(τ, mesh, (cosd(45.0), 0.0, sind(45.0)))   # sun due east
        idx = findall(t .< 0.5)
        @test !isempty(idx) && sum(first(Tuple(I)) for I in idx) / length(idx) < 8.5 - 1.0   # shadow west
        @test all(0 .<= t .<= 1) && all(iszero, cloud_shadow_map(τ, mesh, (0.7, 0.0, -0.7)))
    end

    # ---------------------------------------------------------------------
    # Slant path is exact (#130). A horizontally uniform layer of vertical
    # optical depth τ must give exp(-τ/μ₀) at every sun elevation, and the
    # lowest cell must count in full. The first implementation marched in
    # half-cell steps from the first cell CENTRE (so cell 1 counted half)
    # and multiplied the already slant path by 1/μ₀ again, giving
    # exp(-τ/μ₀²): at 30° elevation 0.20 instead of 0.45 for τ = 0.4.
    # ---------------------------------------------------------------------
    @testset "Cloud shadow slant path is exact" begin
        mesh = RegularGrid(; size=(16,16,16), extent=(160.0,160.0,160.0), halo=(1,1,1))
        τ = zeros(Float64, 16, 16, 16)
        τ[:, :, 5:8] .= 0.1                      # column τ = 0.4
        for (el, az) in ((90.0, 0.0), (60.0, 30.0), (30.0, 200.0), (12.0, 95.0))
            sd = (cosd(el) * sind(az), cosd(el) * cosd(az), sind(el))
            t = cloud_shadow_map(τ, mesh, sd)
            @test maximum(abs.(t .- exp(-0.4 / sind(el)))) < 1e-12
        end
        τ .= 0; τ[:, :, 1] .= 0.4                # the lowest cell alone
        @test all(isapprox.(cloud_shadow_map(τ, mesh, (0.0, 0.0, 1.0)), exp(-0.4); rtol = 1e-12))
        # a single cell: a vertical ray through its centre sees exactly its τ
        τ .= 0; τ[3, 4, 2] = 0.7
        t = cloud_shadow_map(τ, mesh, (0.0, 0.0, 1.0))
        @test t[3, 4] ≈ exp(-0.7) rtol = 1e-12
        @test count(<(1), t) == 1
        # sun due east at 45°: the rays from the centres of (1,4) and (2,4)
        # each cross cell (3,4,2) (x ∈ [20,30] m, z ∈ [10,20] m) along
        # ℓ = 5√2 m, so both see exp(-(0.7/10)·5√2); nothing else is shaded
        el = 45.0
        t45 = cloud_shadow_map(τ, mesh, (cosd(el), 0.0, sind(el)))
        @test t45[1, 4] ≈ exp(-0.07 * 5sqrt(2)) rtol = 1e-12
        @test t45[2, 4] ≈ exp(-0.07 * 5sqrt(2)) rtol = 1e-12
        @test count(<(1), t45) == 2
        # building mask: a roof starts the ray on top of its own stack
        sm = SolidMask(mesh)
        mark_solid!(sm, (x, y, z) -> 60 <= x <= 100 && 60 <= y <= 100 && z <= 40)
        τ .= 0; τ[:, :, 3:4] .= 0.2              # cloud inside the building levels only
        tr = Matrix{Float64}(undef, 16, 16)
        combined_shadow_transmittance!(tr, τ, sm.mask, mesh, (0.0, 0.0, 1.0))
        roof = [sm.mask[1 + i, 1 + j, 1 + 1] for i in 1:16, j in 1:16]
        @test all(tr[roof] .== 1.0)              # roof above the cloud: unattenuated
        @test all(isapprox.(tr[.!roof], exp(-0.4); rtol = 1e-12))
        # k_start does the same without the building blocking the beam: at
        # a slant sun a roof cell next to the edge is lit through the cloud
        # above the roof only, and nothing is fully shadowed
        kst = [count(k -> sm.mask[1 + i, 1 + j, 1 + k], 1:16) for i in 1:16, j in 1:16]
        τ[:, :, 6] .= 0.3                        # a cloud above the roofs too
        sd = (cosd(40.0), 0.0, sind(40.0))
        t_ks = cloud_shadow_map(τ, mesh, sd)     # allocating form: from the ground
        cloud_shadow_transmittance!(t_ks, τ, mesh, sd; k_start = kst)
        @test all(isapprox.(t_ks[roof], exp(-0.3 / sind(40.0)); rtol = 1e-12))
        @test all(isapprox.(t_ks[.!roof], exp(-0.7 / sind(40.0)); rtol = 1e-12))
    end

    # One source of truth: the same cloud has the same visible τ in the 3-D
    # shadow and in the TwoStream cloud optics (k = 1.5/r_eff = 0.15 m²/g, #130).
    @testset "Cloud optical depth from condensate" begin
        ql = zeros(Float64, 4, 4, 8); ql[:, :, 4:6] .= 5e-4
        τ = cloud_optical_depth_column(ql, zeros(size(ql)), 1.0, 50.0)
        @test isapprox(τ[1, 1, 5], 0.15 * 0.5 * 50; rtol = 1e-12) && all(iszero, τ[:, :, 1:3])
        col = AtmosphericColumn(Float64, 8; ql = ql[1, 1, :], ρ = fill(1.0, 8), dz = fill(50.0, 8))
        @test isapprox(sum(τ[1, 1, :]), cloud_optical_depth(col, mie_cloud_optics(shortwave_bands())); rtol = 1e-12)
        @test cloud_optical_depth_column(zeros(size(ql)), ql, 1.0, 50.0)[1, 1, 5] < τ[1, 1, 5]   # ice
    end
end
