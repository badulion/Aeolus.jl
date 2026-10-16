# ONE-e (#454): the physics that lived in the urban case's glue files, now
# in its physics modules. Each test asserts the physics of the moved
# function at module level, independent of any case:
#
#   * Operators.moist_buoyancy_perturbation! / add_moist_buoyancy!
#     (from components/urban/moist_buoyancy.jl): θ_ρ − θ = θ (0.608 q_v − q_c),
#     zero level mean over the fluid, the w-face average;
#   * BuildingEnergy.traffic_heat_flux / add_traffic_heat! (from
#     components/urban/traffic_heat.jl): the heat released equals Q_V × area × dt,
#     on both dynamical cores;
#   * PlantCanopy.vegetation_canopy_drag (from components/urban/veg_drag.jl): each
#     crown integrates to its LAI over k_height layers, the implicit factor;
#   * Chemistry.traffic_profile (from components/urban/aq_chem.jl): floor, peaks;
#   * SurfaceExchange.aerodynamic_resistance (from components/urban/aq_chem.jl):
#     r_a = 1/(C_H U), the neutral log-law value;
#   * PlantCanopy.apply_tree_shade! (from components/urban/sw_modifiers.jl): e^{-kΛ};
#   * UrbanSurface.glazing_areal_heat_capacity / glazing_solar_transmittance
#     (from components/urban/glass.jl): Σρcd, R + T + ΣA = 1;
#   * BuildingEnergy.city_buildings / step_city_buildings! /
#     apply_building_heat! (from components/urban/buildings_citymean.jl): the grid
#     maps carry the zone's release exactly, the air gets it exactly;
#   * RadiationFog.dew_exchange_columns! (from components/urban/fog_dew.jl): water
#     and latent heat conserved per column, at the column's own air cell;
#   * Radiosity.direct_absorbed_share! and SolarGeometry.shade_direct_beam!
#     (from components/urban/cloud_shadow.jl): flat ground, sun only, share 1;
#   * Microphysics.clamp_booked! (from components/urban/micro_kessler.jl): the
#     clamp's removal, ρ-weighted;
#   * Anelastic.hydrostatic_pressure (from site/reference_state.jl): the
#     exact isentropic atmosphere, RK4 convergence;
#   * ColumnRadiation (from components/urban/column_radiation.jl): the stitched
#     column, energy closure of the heating, the TOA insolation, fog, night;
#   * SurfaceExchange.deliver_surface_fluxes! and MoistThermodynamics.θ_per_heat
#     (from components/urban/surface_exchange.jl, site/reference_state.jl): the air
#     receives exactly H τ and E τ;
#   * Radiosity.surface_irradiance! (from components/urban/radiosity_sun.jl): the
#     column's shortwave energy and net longwave spread over its surface.

using Test
using Aeolus
using Aeolus.Fields: Field, interior_view
using Aeolus.Grids: FaceZ
using Aeolus.Anelastic: AnelasticSimulation, isentropic_reference
using Random: Xoshiro
using Aeolus.UrbanSurface: double_glazing, glazing_optics_hemispherical, glazing_optics
using Aeolus.MoistThermodynamics: latent_heat_vapor, saturation_specific_humidity

const GP_MT = Aeolus.MoistThermodynamics

@testset "ONE-e: physics moved out of the urban glue (#454)" begin

    @testset "Operators: moist buoyancy perturbation (virtual + condensate loading)" begin
        rng = Xoshiro(454)
        Nx, Ny, Nz = 12, 10, 6
        g = RegularGrid(; size = (Nx, Ny, Nz), extent = (12.0, 10.0, 6.0), halo = (2, 2, 2))
        θ = 300 .+ randn(rng, Nx, Ny, Nz)
        qv = 8e-3 .+ 1e-3 .* randn(rng, Nx, Ny, Nz)
        qc = [abs.(1e-4 .* randn(rng, Nx, Ny, Nz)), abs.(1e-5 .* randn(rng, Nx, Ny, Nz))]
        fluid = rand(rng, Nx, Ny, Nz) .< 0.8
        fluid[:, :, Nz] .= false                         # a level with no fluid
        Δ = zeros(Nx, Ny, Nz)
        Aeolus.Operators.moist_buoyancy_perturbation!(Δ, θ, qv, qc, fluid)
        c = GP_MT.DEFAULT_CONSTANTS
        a = c.molmass_dry / c.molmass_water - 1          # 1/ε − 1
        @test abs(a - 0.608) < 1e-3
        maxdev = 0.0
        for k in 1:Nz
            f = view(fluid, :, :, k)
            n = count(f)
            if n == 0
                @test all(iszero, view(Δ, :, :, k))
                continue
            end
            # the closed form of θ_ρ − θ: θ ((1/ε − 1) q_v − q_c), minus its fluid mean
            ref = [θ[i, j, k] * (a * qv[i, j, k] - qc[1][i, j, k] - qc[2][i, j, k]) for i in 1:Nx, j in 1:Ny]
            m = sum(ref[f]) / n
            @test abs(sum(Δ[:, :, k][f])) <= 1e-12 * n          # no net force on the level
            maxdev = max(maxdev, maximum(abs.(Δ[:, :, k][f] .- (ref[f] .- m))))
        end
        @test maxdev < 1e-10                                     # K; Δ is O(0.3 K)
        @test all(Δ[.!fluid] .== 0)                              # no buoyancy in solids
        # a horizontally uniform moisture field is a pure pressure gradient: no perturbation
        Δu = zeros(Nx, Ny, Nz)
        Aeolus.Operators.moist_buoyancy_perturbation!(Δu, fill(300.0, Nx, Ny, Nz), fill(8e-3, Nx, Ny, Nz),
                                                      Array{Float64,3}[], trues(Nx, Ny, Nz))
        @test maximum(abs, Δu) < 1e-12
        # moister air is lighter, cloudy air heavier
        Δs = zeros(2, 1, 1)
        Aeolus.Operators.moist_buoyancy_perturbation!(Δs, fill(300.0, 2, 1, 1), reshape([9e-3, 8e-3], 2, 1, 1),
                                                      [zeros(2, 1, 1)], trues(2, 1, 1))
        @test Δs[1] > 0 > Δs[2]
        @test Δs[1] - Δs[2] ≈ 300 * a * 1e-3 rtol = 1e-9
        Aeolus.Operators.moist_buoyancy_perturbation!(Δs, fill(300.0, 2, 1, 1), fill(8e-3, 2, 1, 1),
                                                      [reshape([1e-3, 0.0], 2, 1, 1)], trues(2, 1, 1))
        @test Δs[1] < 0 < Δs[2]
        # the w tendency: g Δ/θ₀ averaged to the interior faces, the walls untouched
        tw = Field(FaceZ(), g)
        tw.data .= randn(rng, size(tw.data))
        tw0 = copy(tw.data)
        Aeolus.Operators.add_moist_buoyancy!(tw, Δ, 300.0, 9.81)
        H = 2
        dw = (tw.data .- tw0)[H+1:H+Nx, H+1:H+Ny, H+1:H+Nz+1]      # the w faces k = 1 … Nz + 1
        @test all(iszero, dw[:, :, 1]) && all(iszero, dw[:, :, Nz+1])
        @test maximum(abs.(dw[:, :, 2:Nz] .- 9.81 / 300 .* 0.5 .* (Δ[:, :, 1:Nz-1] .+ Δ[:, :, 2:Nz]))) < 1e-15
    end

    @testset "BuildingEnergy: traffic heat = Q_V × area × dt, on both cores" begin
        @test traffic_heat_flux(8.0, 100, 25) == 32.0            # concentrated on a quarter of the area
        Nx, Ny, Nz = 8, 6, 4
        mesh = RegularGrid(; size = (Nx, Ny, Nz), extent = (80.0, 60.0, 40.0), halo = (2, 2, 2))
        bcs = (bcs = FieldBoundaryConditions(:periodic, :periodic, :wall),
               θ_bcs = FieldBoundaryConditions(:periodic, :periodic, :noflux))
        road = [CartesianIndex(i, 2) for i in 1:Nx]              # one street
        for sim in (BoussinesqSimulation(mesh; θ₀ = 300.0, bcs...),
                    AnelasticSimulation(mesh; ref = isentropic_reference(mesh; θ_const = 300.0), bcs...))
            interior_view(sim.θ) .= 300.0
            θ0 = copy(interior_view(sim.θ))
            ρcp, dt, Q = 1180.0, 2.0, 8.0
            add_traffic_heat!(sim, road, Nx * Ny, dt; Q_urban = Q, ρcp = ρcp)
            dθ = interior_view(sim.θ) .- θ0
            # only the first layer of the road columns warms
            @test all(iszero, dθ[:, :, 2:end]) && all(iszero, dθ[:, [1; 3:Ny], 1])
            # the heat added, Σ ρc_p Δθ Δz ΔxΔy, is Q_V over the whole area for dt
            A = mesh.dx * mesh.dy
            @test sum(dθ) * ρcp * mesh.dz * A ≈ Q * Nx * Ny * A * dt rtol = 1e-12
            # no roads: no heat
            θ1 = copy(interior_view(sim.θ))
            add_traffic_heat!(sim, CartesianIndex{2}[], Nx * Ny, dt; Q_urban = Q, ρcp = ρcp)
            @test agree(interior_view(sim.θ), θ1)
        end
    end

    @testset "PlantCanopy: vegetation canopy drag of a LAI map" begin
        Nx, Ny, Nz = 6, 5, 8
        mesh = RegularGrid(; size = (Nx, Ny, Nz), extent = (60.0, 50.0, 40.0), halo = (2, 2, 2))
        veg = falses(Nx, Ny); veg[2:4, 2:3] .= true
        LAI = zeros(Nx, Ny); LAI[2:4, 2:3] .= 3.0; LAI[4, 3] = 0.0   # a vegetated column without leaves
        LAI[6, 5] = 5.0                                                 # leaves off the vegetation mask
        d = vegetation_canopy_drag(mesh, veg, LAI; cd = 0.25, k_height = 3)
        @test d.C_d == 0.25
        for i in 1:Nx, j in 1:Ny
            col = d.lad[i, j, :]
            if veg[i, j] && LAI[i, j] > 0
                @test sum(col) * mesh.dz ≈ LAI[i, j] rtol = 1e-12      # the crown holds its LAI
                @test all(>(0), col[1:3]) && all(iszero, col[4:end])   # over k_height layers
            else
                @test all(iszero, col)
            end
        end
        # the default crown: Nz ÷ 4 layers
        d2 = vegetation_canopy_drag(mesh, veg, LAI)
        @test all(>(0), d2.lad[2, 2, 1:2]) && all(iszero, d2.lad[2, 2, 3:end]) && d2.C_d == 0.2
        # the implicit factor 1/(1 + Δt C_d a |U|) on a uniform u in the crown
        sim = BoussinesqSimulation(mesh; θ₀ = 300.0,
                                   bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
        interior_view(sim.u) .= 2.0
        apply_canopy_drag!(sim, d, 10.0)
        a = d.lad[3, 2, 1]                               # an inner crown cell: both u faces in the crown
        @test interior_view(sim.u)[3, 2, 1] ≈ 2.0 / (1 + 10.0 * 0.25 * a * 2.0) rtol = 1e-12
        @test interior_view(sim.u)[1, 1, 5] == 2.0       # above and outside the crowns
    end

    @testset "Chemistry: the diurnal traffic profile" begin
        h = 0:0.01:24
        f = traffic_profile.(h)
        @test all(x -> 0.15 <= x <= 1.05, f)
        @test traffic_profile(3.0) ≈ 0.15 atol = 1e-3                # night floor
        am = findall(x -> 6 <= x <= 12, h); pm = findall(x -> 14 <= x <= 20, h)
        @test 7.5 <= h[am[argmax(f[am])]] <= 8.5 && 0.95 <= maximum(f[am]) <= 1.05   # the morning rush hour
        @test 17.0 <= h[pm[argmax(f[pm])]] <= 18.0 && 0.95 <= maximum(f[pm]) <= 1.05 # the evening rush hour
        @test 0.65 <= traffic_profile(12.5) <= 0.75                     # midday plateau
        @test traffic_profile(17.5) > traffic_profile(12.5)             # the evening rush hour
    end

    @testset "SurfaceExchange: aerodynamic resistance r_a = 1/(C_H U)" begin
        sol = solve_surface_layer(; U = 4.0, z_m = 10.0, θ_a = 300.0, θ_s = 300.0, q_a = 0.0, q_s = 0.0,
                                  z0m = 0.1, z0h = 0.01)
        ra = aerodynamic_resistance(sol)
        @test ra == 1 / (sol.C_H * sol.U)
        # neutral: the log law, r_a = ln(z/z0m) ln(z/z0h) / (κ² U)
        κ = DEFAULT_SURFACE_CONSTANTS.κ
        @test ra ≈ log(10.0 / 0.1) * log(10.0 / 0.01) / (κ^2 * 4.0) rtol = 1e-6
        # unstable air (warm skin) mixes better: a lower resistance
        su = solve_surface_layer(; U = 4.0, z_m = 10.0, θ_a = 300.0, θ_s = 305.0, q_a = 0.0, q_s = 0.0,
                                 z0m = 0.1, z0h = 0.01)
        @test aerodynamic_resistance(su) < ra
        # in series with r_b and r_c it bounds the deposition velocity
        @test dry_deposition_velocity(ra, 10.0, 100.0) < 1 / ra
    end
    @testset "PlantCanopy: big-leaf tree shade e^{-k LAI}" begin
        sw = fill(500.0, 3, 2)
        veg = falses(3, 2); veg[2, 1] = true; veg[3, 2] = true
        LAI = fill(4.0, 3, 2)
        apply_tree_shade!(sw, veg, LAI; k_leaf = 0.5)
        @test sw[2, 1] ≈ 500 * exp(-2.0) rtol = 1e-14
        @test sw[3, 2] ≈ 500 * exp(-2.0) rtol = 1e-14
        @test count(==(500.0), sw) == 4                            # bare cells untouched
        apply_tree_shade!(sw, veg, zeros(3, 2))                    # no leaves: no shade
        @test sw[2, 1] ≈ 500 * exp(-2.0) rtol = 1e-14
    end

    @testset "UrbanSurface: the glazing of a one-temperature glass surface" begin
        g = double_glazing()
        C = glazing_areal_heat_capacity(g)
        @test C ≈ sum(p -> p.material.density * p.material.specific_heat * p.thickness, g.panes) rtol = 1e-15
        @test 1.0e4 < C < 2.0e4                                    # 2 × 4 mm float: ≈ 1.5e4 J m⁻² K⁻¹
        h = glazing_optics_hemispherical(g)
        @test glazing_solar_transmittance(g) == h.T
        @test glazing_solar_transmittance(g; cos_incidence = -0.2) == h.T   # sun below the pane
        @test glazing_solar_transmittance(g; cos_incidence = cosd(60)) == glazing_optics(g, acos(cosd(60))).T
        @test glazing_solar_transmittance(g; cos_incidence = 1.0) > glazing_solar_transmittance(g; cos_incidence = cosd(75))
        @test h.R + h.T + sum(h.A) ≈ 1 atol = 1e-9                 # a glass cell absorbs (1 − R − T) SW
    end

    @testset "BuildingEnergy: the city-mean building's release reaches the grid and the air exactly" begin
        Nx, Ny, Nz = 8, 6, 10
        dx = dy = 10.0; dz = 5.0
        hm = zeros(Nx, Ny); hm[3:4, 2:4] .= 20.0; hm[6, 3:5] .= 12.0
        walls = [CartesianIndex(i, j) for i in 1:Nx, j in 1:Ny if hm[i, j] == 0 && (i in (2, 5, 7))]
        cb = city_buildings(hm, walls, dx, dy, dz, 1500.0; start_weekday = 2)
        @test cb.zone.floor_area ≈ sum(hm) * dx * dy / 3.96 rtol = 1e-12
        @test cb.kair[3, 2] == 5 && cb.kair[6, 3] == round(Int, 12 / dz) + 1 && cb.kair[1, 1] == 1
        # 10:00 on a weekday, warm and sunny: office hours, cooling
        step_city_buildings!(cb, 10 * 3600.0, 600.0, 303.0, 0.010; sw_wall = 300.0)
        r = cb.release
        z = cb.zone
        Q_wall = -z.envelope[1].area * z.envelope[1].F_outer
        Q_roof = -z.envelope[2].area * z.envelope[2].F_outer
        direct = r.Q_exhaust_sensible + r.Q_exhaust_latent + r.Q_ventilation_sensible + r.Q_ventilation_latent +
                 r.Q_windows + r.Q_lost + r.Q_latent_vented + Q_wall + Q_roof
        @test sum(cb.H .+ cb.LE) * dx * dy ≈ direct rtol = 1e-12
        @test sum(cb.S) * dx * dy ≈ -r.Q_solar rtol = 1e-12
        @test r.Q_solar > 0 && all(<=(0), cb.S)
        @test all(I -> cb.S[I] == 0 || I in walls, CartesianIndices(cb.S))   # window sun only off the wall cells
        # into the air: θ by H/(ρ c_p Π) and qv by LE/(ρ L_v), per Δz, at each column's first air layer
        mesh = RegularGrid(; size = (Nx, Ny, Nz), extent = (Nx * dx, Ny * dy, Nz * dz), halo = (2, 2, 2))
        sim = BoussinesqSimulation(mesh; θ₀ = 300.0)
        interior_view(sim.θ) .= 0.0                                # count the increments only
        qv = Field(Aeolus.Grids.CellCenter(), mesh)
        sens = zeros(Nx, Ny)
        ρ = collect(range(1.2, 1.1; length = Nz)); Π = collect(range(1.0, 0.99; length = Nz))
        apply_building_heat!(sim, qv, sens, cb, 2.0; ρ = ρ, cp = 1005.0, π = Π)
        θ = interior_view(sim.θ); q = interior_view(qv)
        heat = sum(θ[i, j, k] * ρ[k] * 1005.0 * Π[k] * dz for i in 1:Nx, j in 1:Ny, k in 1:Nz)
        water = sum(q[i, j, k] * ρ[k] * dz for i in 1:Nx, j in 1:Ny, k in 1:Nz)
        @test heat ≈ sum(cb.H) * 2.0 rtol = 1e-12
        @test water ≈ sum(cb.LE) * 2.0 / CITY_BUILDING_L_V rtol = 1e-12
        @test θ[3, 2, 5] != 0 && all(iszero, θ[3, 2, 1:4])        # above the roof, not inside
        @test sens == -cb.S                                        # the window sun leaves the wall surfaces
    end

    @testset "RadiationFog: dew on each column's own air cell conserves water and heat" begin
        Nx, Ny, Nz = 3, 2, 4
        p = [1.0e5, 0.995e5, 0.99e5, 0.985e5]; ρ = [1.2, 1.19, 1.18, 1.17]
        Ts = fill(283.0, Nx, Ny); Ts[3, 2] = 300.0                # one warm surface: no dew
        qv = fill(saturation_specific_humidity(285.0, p[1]), Nx, Ny, Nz)  # moist air over a cold surface
        q0 = copy(qv)
        k_air = [1 2 1; 3 1 2]' |> collect                         # (Nx, Ny)
        mask = trues(Nx, Ny); mask[1, 2] = false                   # a column with its own vapour exchange
        dew = zeros(Nx, Ny)
        H = fill(NaN, Nx, Ny)
        dt, dz = 60.0, 5.0
        dew_exchange_columns!(H, qv, dew, Ts, mask, k_air, p, ρ, dz, dt)
        for i in 1:Nx, j in 1:Ny
            k = k_air[i, j]
            if !mask[i, j] || Ts[i, j] == 300.0
                @test H[i, j] == 0 && dew[i, j] == 0 && qv[i, j, :] == q0[i, j, :]
                continue
            end
            @test dew[i, j] > 0 && H[i, j] > 0                     # dew forms, its latent heat warms the surface
            @test (q0[i, j, k] - qv[i, j, k]) * ρ[k] * dz ≈ dew[i, j] rtol = 1e-9   # water conserved
            @test H[i, j] ≈ latent_heat_vapor(Ts[i, j]) * dew[i, j] / dt rtol = 1e-12
            @test all(kk -> kk == k || qv[i, j, kk] == q0[i, j, kk], 1:Nz)          # only the air cell
        end
        # the surface warms past the dew point: the dew re-evaporates, the reservoir stays ≥ 0
        Ts .= 295.0
        dew_exchange_columns!(H, qv, dew, Ts, mask, k_air, p, ρ, dz, 3600.0)
        @test all(>=(0), dew) && all(<=(0), H)
    end

    @testset "Radiosity + SolarGeometry: the cloud shadow on the direct beam" begin
        # flat ground: no interreflection, so the absorbed shortwave is (1 − α) (E_dir + E_dif)
        hm = zeros(4, 3)
        sc = build_scene(hm, 5.0, 5.0)
        vf = view_factors(sc)
        sun = (0.3, -0.2, 0.93)
        cols = [f.col for f in sc.facets]
        f1 = zeros(4, 3)
        direct_absorbed_share!(f1, sc, solve_shortwave(sc, vf, sun; dni = 800.0, dhi = 0.0), cols)
        @test all(x -> isapprox(x, 1; atol = 1e-12), f1)            # sun only: all of it is beam
        sol = solve_shortwave(sc, vf, sun; dni = 800.0, dhi = 120.0)
        direct_absorbed_share!(f1, sc, sol, cols)
        @test all(x -> isapprox(x, sol.direct[1] / (sol.direct[1] + sol.diffuse_sky[1]); rtol = 1e-12), f1)
        direct_absorbed_share!(f1, sc, solve_shortwave(sc, vf, sun; dni = 0.0, dhi = 120.0), cols)
        @test all(iszero, f1)                                       # overcast: no beam
        # the shadow takes f_dir (1 − T) of the shortwave and nothing else
        sw = fill(400.0, 2, 2); removed = fill(NaN, 2, 2)
        fdir = [0.6 0.0; 1.0 0.5]; T = [0.5 0.2; 1.0 0.0]
        shade_direct_beam!(sw, removed, fdir, T)
        @test removed ≈ 400 .* fdir .* (1 .- T) rtol = 1e-15
        @test sw .+ removed ≈ fill(400.0, 2, 2) rtol = 1e-15
        @test sw[1, 2] == 400 && sw[2, 1] == 400                   # no beam, or a clear path
    end

    @testset "Microphysics: the booked vapour clamp" begin
        rng = Xoshiro(7)
        q = 0.02 .+ 0.02 .* randn(rng, 5, 4, 3)                    # some below 0, some above 0.03
        q0 = copy(q)
        ρ = [1.2, 1.1, 1.0]
        removed = Aeolus.Microphysics.clamp_booked!(q, 0.0, 0.03, ρ)
        @test q == clamp.(q0, 0.0, 0.03)
        @test removed ≈ sum(ρ[k] * sum(q0[:, :, k] .- q[:, :, k]) for k in 1:3) rtol = 1e-12
        @test Aeolus.Microphysics.clamp_booked!(q, 0.0, 0.03, ρ) == 0   # nothing left to remove
    end

    @testset "Anelastic: the continuous hydrostatic pressure" begin
        c = GP_MT.DEFAULT_CONSTANTS
        R_d = c.gas_constant / c.molmass_dry
        κ = R_d / c.cp_dry
        # constant θ: Π(z) = Π_s − g z/(c_pd θ) exactly (an isentropic atmosphere)
        z = collect(5.0:10.0:995.0)
        p = hydrostatic_pressure(_ -> 300.0, z, 101325.0; g = 9.81, nsub = 50)
        Πs = (101325.0 / c.p_ref)^κ
        p_exact = c.p_ref .* (Πs .- 9.81 .* z ./ (c.cp_dry * 300.0)) .^ (1 / κ)
        @test maximum(abs.(p .- p_exact) ./ p_exact) < 1e-12
        # a stable profile (θ warmer aloft, the air lighter) falls slower than the
        # isentropic one of its surface θ; RK4 is 4th order: halving the
        # sub-step cuts the error about 16-fold (coarse levels, so the error is
        # far above round-off)
        θs(zz) = 300.0 + 0.01 * zz
        @test all(hydrostatic_pressure(θs, z, 101325.0) .> p)
        zc = [500.0, 2000.0, 5000.0]
        pr = hydrostatic_pressure(θs, zc, 1.0e5; nsub = 4000)
        e(n) = maximum(abs.(hydrostatic_pressure(θs, zc, 1.0e5; nsub = n) .- pr))
        @test 12 < e(1) / e(2) < 20
    end

    @testset "ColumnRadiation: a model column under the standard atmosphere" begin
        Nz = 12
        dz = fill(25.0, Nz)
        θ = 300.0 .+ 0.003 .* (cumsum(dz) .- dz ./ 2)
        qv = fill(8e-3, Nz)
        c = column_radiation_solver(Nz, 101325.0; nz_upper = 30)
        @test c.p_surf == 101325.0 && length(c.heat) == Nz && c.ncalls == 0
        col = stitched_column(c, θ, qv, zeros(Nz), dz; T_rad = 301.0)
        # the model layers under 30 standard-atmosphere layers to 30 km
        @test length(col.dz) == Nz + 30
        @test col.dz[1:Nz] == dz
        @test sum(col.dz) ≈ 30_000 rtol = 1e-9
        noon = solar_position(solar_time(2019, 6, 29, 11, 0, 0.0; latitude = 52.52, longitude = 13.41))
        column_radiation!(c, col, noon, 0.95, 0.15)
        cp = Aeolus.RadiationCore.CP_DRY_AIR
        # energy: ∫ ρ c_p (dT/dt) dz over the model layers is the net flux into them
        ΔF = (c.flw.flux_net[Nz+1] - c.flw.flux_net[1]) + (c.fsw.flux_net[Nz+1] - c.fsw.flux_net[1])
        Q = sum(col.ρ[k] * cp * c.heat[k] * col.dz[k] for k in 1:Nz)
        @test Q ≈ ΔF rtol = 1e-10
        @test c.heat ≈ c.heat_lw .+ c.heat_sw rtol = 1e-14
        # the top of the column receives S₀ (distance-corrected) μ₀
        @test c.fsw.flux_dn[end] ≈ Aeolus.RadiationCore.SOLAR_CONSTANT * earth_sun_distance_factor(noon) * c.μ₀ rtol = 1e-10
        @test 0 <= c.fsw.flux_dn_dir[1] <= c.fsw.flux_dn[1] <= c.fsw.flux_dn[end]
        # the sky longwave is between nothing and the black body of the warmest level
        Tmax = max(maximum(col.T_lev), maximum(col.T_lay))
        @test 0 < c.flw.flux_dn[1] < Aeolus.RadiationCore.STEFAN_BOLTZMANN * Tmax^4
        @test c.fsw_clear === c.fsw                     # no condensate: one shortwave solve
        k, kl, ks = column_heating_kday(c)
        w = col.ρ[1:Nz] .* col.dz[1:Nz]
        @test k ≈ 86400 * sum(w .* c.heat) / sum(w) rtol = 1e-14
        @test k ≈ kl + ks rtol = 1e-12
        # a fog layer: less direct beam at the ground than in the clear column
        ql = [k <= 3 ? 3e-4 : 0.0 for k in 1:Nz]
        fog = stitched_column(c, θ, qv, ql, dz; T_rad = 301.0)
        column_radiation!(c, fog, noon, 0.95, 0.15; clear = () -> col)
        @test c.fsw.flux_dn_dir[1] < 0.5 * c.fsw_clear.flux_dn_dir[1]
        @test c.ncalls == 2
        # night: no shortwave, longwave cooling only
        night = solar_position(solar_time(2019, 6, 29, 23, 0, 0.0; latitude = 52.52, longitude = 13.41))
        column_radiation!(c, col, night, 0.95, 0.15)
        @test c.fsw === nothing && c.fsw_clear === nothing && all(iszero, c.heat_sw) && c.μ₀ == 0
        @test c.heat == c.heat_lw
    end

    @testset "SurfaceExchange: the surface fluxes into the first air cell" begin
        rng = Xoshiro(11)
        Nx, Ny, Nz, H = 6, 5, 4, 2
        θ = fill(300.0, Nx, Ny, Nz); qv = fill(8e-3, Nx, Ny, Nz)
        Hs = 200 .* rand(rng, Nx, Ny) .- 50; Es = 1e-4 .* rand(rng, Nx, Ny)
        k_air = rand(rng, 1:2, Nx, Ny)
        surfaces = Matrix{Any}(fill(:ground, Nx, Ny)); surfaces[2, 3] = nothing   # one column without a surface
        cp = 1004 .+ 10 .* rand(rng, Nx, Ny)
        ρ̄ = [1.20, 1.19, 1.18, 1.17]; Π̄ = [1.003, 1.002, 1.001, 1.0]
        mask = falses(Nx + 2H, Ny + 2H, Nz + 2H); mask[H+4, H+4, H+k_air[4, 4]] = true   # one roof cell
        dz, τ = 5.0, 2.0
        θ0, q0 = copy(θ), copy(qv)
        evap, nsolid = deliver_surface_fluxes!(θ, qv, Hs, Es, k_air, surfaces, cp, ρ̄, Π̄, dz, τ;
                                               solid = mask, offsets = (H, H, H))
        @test θ[2, 3, :] == θ0[2, 3, :] && qv[2, 3, :] == q0[2, 3, :]     # no surface: nothing
        @test nsolid == 1
        heat = water = 0.0; Hτ = Eτ = 0.0
        for j in 1:Ny, i in 1:Nx
            surfaces[i, j] === nothing && continue
            k = k_air[i, j]
            # only the first air cell changes
            @test all(θ[i, j, kk] == θ0[i, j, kk] for kk in 1:Nz if kk != k)
            heat += ρ̄[k] * cp[i, j] * Π̄[k] * (θ[i, j, k] - θ0[i, j, k]) * dz
            water += ρ̄[k] * (qv[i, j, k] - q0[i, j, k]) * dz
            Hτ += Hs[i, j] * τ; Eτ += Es[i, j] * τ
        end
        # the air gets exactly H τ and E τ per unit area (the θ_per_heat conversion)
        @test heat ≈ Hτ rtol = 1e-10
        @test water ≈ Eτ rtol = 1e-10
        @test evap ≈ Eτ rtol = 1e-14
        @test θ_per_heat(1.2, 1004.0, 1.01) ≈ 1 / (1.2 * 1004.0 * 1.01) rtol = 1e-15
    end

    @testset "Radiosity: the irradiance per unit of surface of a column" begin
        σ = Aeolus.RadiationCore.STEFAN_BOLTZMANN
        Ts = [300.0 310.0; 290.0 305.0]
        f = (area_density = [0.4 1.0; 4.0 13.0], sw_effective = [600.0 500.0; 2400.0 3900.0],
             lw_down_effective = [380.0 360.0; 900.0 4000.0])
        sw = zeros(2, 2); lw = zeros(2, 2)
        surface_irradiance!(sw, lw, f, Ts)
        a = max.(f.area_density, 1.0)
        # flat columns (area ≤ plan area) keep the plan fluxes
        @test sw[1, 1] == 600 && sw[1, 2] == 500
        @test lw[1, 1] ≈ 380 rtol = 1e-14
        @test lw[1, 2] ≈ 360 rtol = 1e-14
        # the column's shortwave energy and its net longwave are kept, spread over its surface
        @test sw .* a ≈ f.sw_effective rtol = 1e-14
        @test (lw .- σ .* Ts .^ 4) .* a ≈ f.lw_down_effective .- σ .* Ts .^ 4 rtol = 1e-12
    end
end
