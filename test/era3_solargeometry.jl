using Test
using Aeolus
using Aeolus.Grids: CellCenter
using Aeolus.RadiationCore
using Aeolus.SolarGeometry

@testset "Era III — Solar geometry & cloud shadows (Phase 38)" begin

    # ---------------------------------------------------------------------
    # NREL SPA reference case (Reda & Andreas 2004, Table A5.1):
    #   2003-10-17 12:30:30 local (UT-7)  →  19:30:30 UTC
    #   lon -105.1786°, lat 39.742476°, elev 1830.14 m, P 820 mbar, T 11 °C
    #   Published: topocentric zenith 50.11162°, azimuth 194.34024°
    # Our Meeus-grade implementation targets ~0.01°, so we assert that band
    # and report the actual deviation.
    # ---------------------------------------------------------------------
    @testset "NREL SPA benchmark case" begin
        st = solar_time(2003, 10, 17, 19, 30, 30.0;
                        latitude = 39.742476, longitude = -105.1786,
                        elevation = 1830.14, pressure = 820.0,
                        temperature = 11.0, delta_t = 67.0)
        p = solar_position(st)

        zen_ref = 50.11162
        az_ref  = 194.34024
        dzen = abs(p.zenith - zen_ref)
        daz  = abs(p.azimuth - az_ref)
        @info "NREL SPA benchmark" zenith=p.zenith zenith_ref=zen_ref Δzenith=dzen azimuth=p.azimuth azimuth_ref=az_ref Δazimuth=daz

        # Meeus-grade accuracy claim: ~0.01°. Allow a little headroom.
        @test dzen < 0.02
        @test daz  < 0.02
        # Earth-Sun distance in mid-October
        @test 0.99 < p.distance_au < 1.00
    end

    # ---------------------------------------------------------------------
    # Equation of time: the term the Era-I sinusoid omitted entirely.
    # Known structure: ≈ −14 min in mid-February, ≈ +16 min in early
    # November, two zeros in between. Range must be within ±17 min.
    # ---------------------------------------------------------------------
    @testset "Equation of time" begin
        eots = Float64[]
        for doy in 1:5:365
            # convert day-of-year to a date in 2020 (leap year)
            d = doy
            mdays = (31,29,31,30,31,30,31,31,30,31,30,31)
            m = 1
            while d > mdays[m]; d -= mdays[m]; m += 1; end
            st = solar_time(2020, m, d, 12, 0, 0.0;
                            latitude = 0.0, longitude = 0.0)
            push!(eots, equation_of_time(st))
        end
        @info "equation of time" min=minimum(eots) max=maximum(eots)
        @test -17 < minimum(eots) < -12      # February minimum
        @test  12 < maximum(eots) <  18      # November maximum
        # must cross zero at least twice over the year
        signs = sign.(eots)
        crossings = count(i -> signs[i] != signs[i+1], 1:length(signs)-1)
        @test crossings >= 2
    end

    # ---------------------------------------------------------------------
    # Declination: solstices and equinoxes are hard constraints.
    # ---------------------------------------------------------------------
    @testset "Declination at solstices and equinoxes" begin
        # June solstice 2020: 2020-06-20 21:43 UTC
        st_jun = solar_time(2020, 6, 20, 21, 43, 0.0;
                            latitude = 0.0, longitude = 0.0)
        δ_jun = solar_declination_accurate(st_jun)
        # December solstice 2020: 2020-12-21 10:02 UTC
        st_dec = solar_time(2020, 12, 21, 10, 2, 0.0;
                            latitude = 0.0, longitude = 0.0)
        δ_dec = solar_declination_accurate(st_dec)
        # March equinox 2020: 2020-03-20 03:50 UTC
        st_mar = solar_time(2020, 3, 20, 3, 50, 0.0;
                            latitude = 0.0, longitude = 0.0)
        δ_mar = solar_declination_accurate(st_mar)
        @info "declination" june=δ_jun december=δ_dec march_equinox=δ_mar

        @test abs(δ_jun - 23.436) < 0.02      # obliquity, positive
        @test abs(δ_dec + 23.436) < 0.02      # obliquity, negative
        @test abs(δ_mar) < 0.02               # equinox: δ ≈ 0

        # The Era-I approximation, for comparison. Note it must be compared
        # across the whole year, not at a solstice: at an extremum the
        # sinusoid's derivative vanishes and it happens to be accurate
        # there (~0.004°). Its real error shows up near the equinoxes,
        # where the true declination is changing fastest.
        approx(doy) = 23.44 * sin(2π * (284 + doy) / 365)
        mdays = (31,29,31,30,31,30,31,31,30,31,30,31)
        worst = 0.0; worst_doy = 0
        for doy in 1:5:365
            d = doy; m = 1
            while d > mdays[m]; d -= mdays[m]; m += 1; end
            st = solar_time(2020, m, d, 12, 0, 0.0;
                            latitude = 0.0, longitude = 0.0)
            e = abs(approx(doy) - solar_declination_accurate(st))
            if e > worst; worst = e; worst_doy = doy; end
        end
        @info "Era-I sinusoid vs accurate declination" worst_error_deg=worst at_doy=worst_doy at_june_solstice=abs(approx(172) - δ_jun)
        @test worst > 0.5         # the approximation really is materially worse
    end

    # ---------------------------------------------------------------------
    # Refraction: standard values are well known.
    # ---------------------------------------------------------------------
    @testset "Atmospheric refraction" begin
        r_horizon = refraction_bennett(0.0)
        r_45      = refraction_bennett(45.0)
        r_zenith  = refraction_bennett(90.0)
        @info "refraction (degrees)" horizon=r_horizon at_45=r_45 zenith=r_zenith horizon_arcmin=r_horizon*60

        # at the horizon refraction is ~34 arcmin ≈ 0.57°
        @test 0.45 < r_horizon < 0.65
        # at 45° it is ~1 arcmin
        @test 0.01 < r_45 < 0.03
        # at the zenith it vanishes
        @test r_zenith < 0.002
        # monotonically decreasing with elevation
        elevs = 0.0:5.0:90.0
        rs = refraction_bennett.(elevs)
        @test all(diff(rs) .<= 1e-12)
        # pressure scaling: thinner air refracts less
        @test refraction_bennett(5.0; pressure = 500.0) <
              refraction_bennett(5.0; pressure = 1013.25)
        # apparent elevation exceeds true elevation
        @test apply_refraction(0.0) > 0.0
    end

    # ---------------------------------------------------------------------
    # Air mass: Kasten-Young. Must be 1 at zenith and finite at horizon,
    # unlike the plane-parallel 1/μ₀ which diverges.
    # ---------------------------------------------------------------------
    @testset "Air mass" begin
        @test isapprox(air_mass(90.0), 1.0; atol = 1e-3)
        am_h = air_mass(0.0)
        @info "air mass" zenith=air_mass(90.0) at_30=air_mass(30.0) horizon=am_h plane_parallel_at_1deg=1/sind(1.0)
        @test 35 < am_h < 42                       # ~38 at the horizon
        @test isfinite(am_h)
        @test air_mass(30.0) ≈ 2.0 atol=0.02       # classic AM2
        # below the horizon
        @test air_mass(-1.0) == Inf
        # pressure correction
        @test air_mass(90.0; pressure = 500.0) < air_mass(90.0; pressure = 1013.25)
    end

    # ---------------------------------------------------------------------
    # Sun direction vector: consistency with elevation/azimuth, and correct
    # geometry at known configurations.
    # ---------------------------------------------------------------------
    @testset "Sun direction vector (ENU)" begin
        # local noon in summer at mid-latitude: sun to the south, up
        st = solar_time(2020, 6, 21, 12, 0, 0.0;
                        latitude = 52.0, longitude = 0.0)
        p = solar_position(st)
        e, n, u = sun_direction_enu(p)
        @test isapprox(sqrt(e^2 + n^2 + u^2), 1.0; atol = 1e-12)   # unit vector
        @test u > 0                     # above the horizon
        @test n < 0                     # northern hemisphere → sun is south
        @info "summer noon at 52N" elevation=p.elevation azimuth=p.azimuth enu=(e,n,u)
        @test 170 < p.azimuth < 190     # roughly due south

        # morning: sun in the east
        st_am = solar_time(2020, 6, 21, 6, 0, 0.0;
                           latitude = 52.0, longitude = 0.0)
        p_am = solar_position(st_am)
        e_am, _, u_am = sun_direction_enu(p_am)
        @test u_am > 0
        @test e_am > 0                  # eastward component in the morning

        # evening: sun in the west
        st_pm = solar_time(2020, 6, 21, 18, 0, 0.0;
                           latitude = 52.0, longitude = 0.0)
        p_pm = solar_position(st_pm)
        e_pm, _, _ = sun_direction_enu(p_pm)
        @test e_pm < 0                  # westward component in the evening
    end

    # ---------------------------------------------------------------------
    # Polar day/night — a strong geometric constraint.
    # ---------------------------------------------------------------------
    @testset "Polar day and night" begin
        # midnight sun at the North Pole in June
        st = solar_time(2020, 6, 21, 0, 0, 0.0;
                        latitude = 89.9, longitude = 0.0)
        @test solar_position(st).elevation > 0
        # polar night at the North Pole in December
        st2 = solar_time(2020, 12, 21, 12, 0, 0.0;
                         latitude = 89.9, longitude = 0.0)
        p2 = solar_position(st2)
        @test p2.elevation < 0
        @test p2.μ₀ == 0                       # clamped, no negative cosine
        @test p2.air_mass == Inf
    end

    # ---------------------------------------------------------------------
    # Earth-Sun distance: perihelion early January, aphelion early July,
    # total variation ~3.4% in irradiance.
    # ---------------------------------------------------------------------
    @testset "Earth-Sun distance factor" begin
        st_jan = solar_time(2020, 1, 4, 12, 0, 0.0; latitude=0.0, longitude=0.0)
        st_jul = solar_time(2020, 7, 5, 12, 0, 0.0; latitude=0.0, longitude=0.0)
        f_jan = earth_sun_distance_factor(solar_position(st_jan))
        f_jul = earth_sun_distance_factor(solar_position(st_jul))
        @info "distance factor" perihelion=f_jan aphelion=f_jul variation_pct=100*(f_jan-f_jul)
        @test f_jan > f_jul                       # closer in January
        @test 0.06 < (f_jan - f_jul) < 0.08       # ~6.9% in R⁻², i.e. ±3.4%
        # At aphelion R ≈ 1.0167 AU so E = 1361/1.0167² ≈ 1317 W/m²;
        # at perihelion R ≈ 0.9833 AU so E ≈ 1408 W/m². These are the
        # textbook extremes of the "solar constant".
        e_jul = extraterrestrial_irradiance(solar_position(st_jul))
        e_jan = extraterrestrial_irradiance(solar_position(st_jan))
        @info "TOA irradiance extremes" aphelion=e_jul perihelion=e_jan
        @test 1305 < e_jul < 1325                 # aphelion
        @test 1400 < e_jan < 1420                 # perihelion
    end

    # ---------------------------------------------------------------------
    # Bird clear-sky model: published behaviour.
    # ---------------------------------------------------------------------
    @testset "Bird clear-sky model" begin
        st = solar_time(2020, 6, 21, 12, 0, 0.0;
                        latitude = 40.0, longitude = 0.0)
        p = solar_position(st)
        m = BirdClearSky(; aod380 = 0.10, aod500 = 0.07,
                           precipitable_water = 1.4, ozone = 0.3, albedo = 0.2)
        r = clear_sky_irradiance(p, m)
        @info "Bird clear-sky (summer noon, 40N)" elevation=p.elevation dni=r.dni dhi=r.dhi ghi=r.ghi

        # near-noon summer clear sky: DNI ~850-1000, GHI ~900-1100 W/m²
        @test 800 < r.dni < 1050
        @test 850 < r.ghi < 1150
        @test 40  < r.dhi < 200
        # GHI must exceed the beam-horizontal component
        @test r.ghi > r.dni * p.μ₀
        # energy bookkeeping: GHI = DNI·μ₀ + DHI
        @test isapprox(r.ghi, r.dni * p.μ₀ + r.dhi; rtol = 1e-10)

        # night gives zeros
        st_n = solar_time(2020, 6, 21, 0, 0, 0.0;
                          latitude = 40.0, longitude = 0.0)
        rn = clear_sky_irradiance(solar_position(st_n), m)
        @test rn.dni == 0 && rn.dhi == 0 && rn.ghi == 0

        # more aerosol → less direct beam, and the reduction is monotone
        dnis = [clear_sky_irradiance(p, BirdClearSky(; aod500 = a)).dni
                for a in (0.05, 0.15, 0.40)]
        @info "DNI vs AOD500" dnis
        @test issorted(dnis; rev = true)
    end

    # ---------------------------------------------------------------------
    # Direct/diffuse partitioning.
    # ---------------------------------------------------------------------
    @testset "Clearness index and diffuse fraction" begin
        st = solar_time(2020, 6, 21, 12, 0, 0.0;
                        latitude = 40.0, longitude = 0.0)
        p = solar_position(st)

        # Erbs endpoints and monotonic decline through the mid-range
        @test diffuse_fraction_erbs(0.05) > 0.95        # overcast ⇒ all diffuse
        @test diffuse_fraction_erbs(0.9) == 0.165       # clear plateau
        fs = diffuse_fraction_erbs.(0.25:0.05:0.75)
        @test all(diff(fs) .< 0)                        # strictly decreasing
        @info "Erbs diffuse fraction" kt_0_05=diffuse_fraction_erbs(0.05) kt_0_5=diffuse_fraction_erbs(0.5) kt_0_9=diffuse_fraction_erbs(0.9)

        # Reindl is bounded and elevation-dependent
        @test 0 <= diffuse_fraction_reindl(0.5, 0.9) <= 1
        @test diffuse_fraction_reindl(0.5, 0.2) != diffuse_fraction_reindl(0.5, 0.9)

        # round-trip: partition a clear-sky GHI and recover the components
        m = BirdClearSky()
        cs = clear_sky_irradiance(p, m)
        for meth in (:erbs, :reindl)
            q = partition_global_irradiance(cs.ghi, p; method = meth)
            @test isapprox(q.dni * p.μ₀ + q.dhi, cs.ghi; rtol = 1e-10)
            @test q.dni >= 0 && q.dhi >= 0
            @test 0 <= q.diffuse_fraction <= 1
            @info "partition" method=meth kt=q.kt fd=q.diffuse_fraction dni=q.dni dhi=q.dhi
        end

        # overcast: kt small ⇒ nearly all diffuse ⇒ almost no beam
        q_oc = partition_global_irradiance(120.0, p; method = :erbs)
        @test q_oc.diffuse_fraction > 0.85
        @test q_oc.dni < 0.2 * 120.0 / p.μ₀

        # clearness index is 0 at night
        st_n = solar_time(2020, 6, 21, 0, 0, 0.0; latitude=40.0, longitude=0.0)
        @test clearness_index(0.0, solar_position(st_n)) == 0
    end

    # ---------------------------------------------------------------------
    # Cloud shadows — the 3-D part. A cloud displaced horizontally from a
    # surface cell must shadow it when the sun is low, which no
    # independent-column calculation can reproduce.
    # ---------------------------------------------------------------------
    @testset "3-D cloud shadow transmittance" begin
        Nx = Ny = 32; Nz = 16
        L = 1000.0; H = 500.0
        mesh = RegularGrid(; size=(Nx,Ny,Nz), extent=(L,L,H), halo=(1,1,1))

        # A single optically thick cloud block, centred. It is kept LOW
        # (k = 4:6, i.e. z ≈ 110-190 m) so that at the test sun elevation
        # the horizontal shadow throw stays well inside the 1000 m periodic
        # domain. With a high cloud and a low sun the throw exceeds the
        # domain width and the shadow wraps around, which makes a centroid
        # displacement test meaningless rather than wrong.
        τ = zeros(Float64, Nx, Ny, Nz)
        i0, i1 = 13, 20
        j0, j1 = 13, 20
        k0, k1 = 4, 6
        τ[i0:i1, j0:j1, k0:k1] .= 2.0          # τ = 2 per cell ⇒ very thick

        # --- overhead sun: shadow directly beneath the cloud
        trans_zenith = cloud_shadow_map(τ, mesh, (0.0, 0.0, 1.0))
        shadow_zenith = trans_zenith .< 0.5
        @test all(shadow_zenith[i0:i1, j0:j1])            # under the cloud: dark
        @test !any(shadow_zenith[1:5, 1:5])                # far corner: lit
        @info "zenith sun" n_shadowed=count(shadow_zenith) min_trans=minimum(trans_zenith) max_trans=maximum(trans_zenith)

        # --- sun from the east at 45°: shadow displaced westward by
        # (cloud height)/tan(45°) ≈ 150 m ≈ 5 cells — comfortably inside
        # the domain, so no periodic wrap-around.
        elev = 45.0
        sd = (cosd(elev), 0.0, sind(elev))                # due east
        trans_low = cloud_shadow_map(τ, mesh, sd)
        shadow_low = trans_low .< 0.5
        # centroid of the shadow must move in −x relative to the cloud centre
        idx = findall(shadow_low)
        @test !isempty(idx)
        cx_shadow = sum(first(Tuple(I)) for I in idx) / length(idx)
        cx_cloud  = (i0 + i1) / 2
        @info "low sun from east" elevation=elev shadow_centroid_i=cx_shadow cloud_centroid_i=cx_cloud displacement=cx_shadow-cx_cloud n_shadowed=length(idx)
        @test cx_shadow < cx_cloud - 1.0    # shadow is displaced westward

        # --- this is the 3-D effect: the shadowed set differs from the
        # column-below-cloud set, which is all a 1-D scheme can produce
        col_below = falses(Nx, Ny); col_below[i0:i1, j0:j1] .= true
        disagreement = count(shadow_low .!= col_below) / (Nx*Ny)
        @info "3-D vs independent-column disagreement" fraction=disagreement
        @test disagreement > 0.02           # genuinely different

        # --- transmittance bounds and clear-sky limit
        @test all(0 .<= trans_low .<= 1)
        trans_clear = cloud_shadow_map(zeros(Float64,Nx,Ny,Nz), mesh, (0.0,0.0,1.0))
        @test all(isapprox.(trans_clear, 1.0; atol = 1e-12))

        # --- sun below horizon ⇒ no direct beam anywhere
        trans_night = cloud_shadow_map(τ, mesh, (0.7, 0.0, -0.7))
        @test all(iszero, trans_night)

        # --- slant path attenuates more than the vertical path
        sd60 = (cosd(60.0), 0.0, sind(60.0))
        t60 = cloud_shadow_map(τ, mesh, sd60)
        @test minimum(t60) <= minimum(trans_zenith) + 1e-12
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

    # ---------------------------------------------------------------------
    # Optical depth from the condensate field.
    # ---------------------------------------------------------------------
    @testset "Cloud optical depth from condensate" begin
        Nx = Ny = 4; Nz = 8
        ql = zeros(Float64, Nx, Ny, Nz)
        qi = zeros(Float64, Nx, Ny, Nz)
        ql[:, :, 4:6] .= 5e-4               # 0.5 g/kg liquid
        ρ = 1.0; dz = 50.0
        τ = cloud_optical_depth_column(ql, qi, ρ, dz)
        @test all(τ .>= 0)
        @test all(iszero, τ[:, :, 1:3])
        # τ = k·LWC·dz with k = 1.5/r_eff = 0.15 m²/g at 10 μm:
        # 0.15 * (5e-4*1.0*1000) * 50 = 3.75. (The original expectation,
        # 0.06 m²/g, was the old default, which is the extinction of 25 μm
        # droplets, not the ~10 μm the docstring claimed; #130.)
        @test isapprox(τ[1,1,5], 0.15 * 0.5 * 50; rtol = 1e-12)
        # one source of truth: the same cloud has the same visible optical
        # depth in the 3-D shadow and in the TwoStream cloud optics
        col = AtmosphericColumn(Float64, Nz; ql = ql[1, 1, :], ρ = fill(1.0, Nz),
                                dz = fill(50.0, Nz))
        τ_ts = cloud_optical_depth(col, mie_cloud_optics(shortwave_bands()))
        @test isapprox(sum(τ[1, 1, :]), τ_ts; rtol = 1e-12)
        @info "cloud τ per layer" value=τ[1,1,5] column_total=sum(τ[1,1,:])
        # more condensate ⇒ more optical depth, linearly
        τ2 = cloud_optical_depth_column(2 .* ql, qi, ρ, dz)
        @test isapprox(τ2[1,1,5], 2 * τ[1,1,5]; rtol = 1e-12)
        # ice contributes with its own (smaller) coefficient
        τi = cloud_optical_depth_column(zeros(size(ql)), ql, ρ, dz)
        @test τi[1,1,5] < τ[1,1,5]
    end

    # ---------------------------------------------------------------------
    # Combined cloud + building shadows in one march.
    # ---------------------------------------------------------------------
    @testset "Combined cloud and building shadows" begin
        Nx = Ny = 24; Nz = 12
        mesh = RegularGrid(; size=(Nx,Ny,Nz), extent=(240.0,240.0,120.0),
                              halo=(2,2,2))
        sm = SolidMask(mesh)
        # a tall block in the middle
        mark_solid!(sm, (x,y,z) -> 100 <= x <= 140 && 100 <= y <= 140 && z <= 60)

        τ = zeros(Float64, Nx, Ny, Nz)
        trans = Matrix{Float64}(undef, Nx, Ny)

        # low sun from the east: the building must cast an opaque shadow west
        elev = 25.0
        sd = (cosd(elev), 0.0, sind(elev))
        combined_shadow_transmittance!(trans, τ, sm.mask, mesh, sd)
        n_opaque = count(iszero, trans)
        @info "building shadow only" n_fully_shadowed=n_opaque
        @test n_opaque > 0
        @test all(0 .<= trans .<= 1)

        # add cloud: some previously lit cells become partially shaded
        τ[:, :, 9:11] .= 0.5
        trans_both = Matrix{Float64}(undef, Nx, Ny)
        combined_shadow_transmittance!(trans_both, τ, sm.mask, mesh, sd)
        @test all(trans_both .<= trans .+ 1e-12)       # cloud can only reduce
        @test count(t -> 0 < t < 1, trans_both) > 0    # partial shading exists
        @info "building + cloud" mean_trans_building_only=sum(trans)/length(trans) mean_trans_both=sum(trans_both)/length(trans_both)

        # no buildings ⇒ identical to the cloud-only routine
        t_cloud_only = Matrix{Float64}(undef, Nx, Ny)
        combined_shadow_transmittance!(t_cloud_only, τ, nothing, mesh, sd)
        t_ref = cloud_shadow_map(τ, mesh, sd)
        @test maximum(abs.(t_cloud_only .- t_ref)) < 1e-12
    end

    # ---------------------------------------------------------------------
    # End-to-end: drive the geometry from a real date and place, and check
    # the diurnal cycle is physically sane.
    # ---------------------------------------------------------------------
    @testset "Diurnal cycle at Alexanderplatz" begin
        lat, lon = 52.5220, 13.4100
        m = BirdClearSky()
        elevs = Float64[]; ghis = Float64[]
        for hour in 0:23
            st = solar_time(2020, 6, 21, hour, 0, 0.0;
                            latitude = lat, longitude = lon)
            p = solar_position(st)
            push!(elevs, p.elevation)
            push!(ghis, clear_sky_irradiance(p, m).ghi)
        end
        imax = argmax(elevs)
        @info "Berlin summer solstice" max_elevation=maximum(elevs) hour_of_max_UTC=imax-1 max_ghi=maximum(ghis) n_daylight_hours=count(>(0), elevs)

        # max solar elevation at summer solstice ≈ 90 − lat + 23.44 = 60.9°
        @test abs(maximum(elevs) - (90 - lat + 23.44)) < 0.5
        # solar noon in Berlin (13.41°E) is ≈ 10:56 UTC, so hour 11 UTC
        @test imax - 1 in (10, 11)
        # long summer day at 52.5°N
        @test count(>(0), elevs) >= 16
        # GHI peaks with elevation and is zero at night
        @test argmax(ghis) == imax
        @test ghis[1] == 0
    end
end
