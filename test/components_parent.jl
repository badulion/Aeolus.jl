# The urban nest parent's physics as components of a domain (NEST-p1, #481),
# on both dynamical cores:
#
#   * TiledUrbanSurface: the heat and vapour the air receives over a step are the
#     tiles' area-mean fluxes × the step, exactly (the canopy levels' depth
#     weights sum to one); u* = √(Σ fᵢ τᵢ / ρ̄₁); the drag is the implicit
# deceleration u/(1 + Δt a/U); a checkpoint restores the tiles;
#   * PrescribedSurfaceFlux: Q Δt/Δz₁ into the first level of every column;
#   * MeanColumnRadiation: the sun of the site's clock (sw ≈ 0 at night, a
#     clear-sky noon sun by day), the heating applied as Ṫ Δt / Π̄, the
#     downwelling flux handed to the tiles, one call per period;
#   * LargeScale: the boundary-layer tracker (the parcel z_i of a mixed layer
#     every `every` steps, the floor), the nudging weight zero below
#     factor × z_i, the two clocks of an SSP-RK3 step (the step mean versus
#     the core's substage clock, which leads by 3Δt/2);
#   * [sgs] model = "tke": the exact wall distance of a stretched grid
#     (bottom only, or both walls), and the surface value u*²/√(c_k c_ε) of
#     the :surface component's u*;
#   * (the parent as a [[domain]] of the model's config, NEST-p3 #483:
#     test/config_parent_domain.jl; the tiled Berlin parent is in
#     urban_checks.jl, full tier).

using Test
using Aeolus
using Aeolus.Components
using Aeolus.Fields: interior_view, set!
using Aeolus.Anelastic: AnelasticSimulation, isentropic_reference
using Aeolus.TimeIntegration: CFLConstraints
using Aeolus.Radiation: SlabLand
using Aeolus.SurfaceTiles: SurfaceMosaic, last_tile_fluxes, mosaic_fractions
using Aeolus.LargeScaleForcing: AdvectiveTendencies, ProfileNudging, NudgingWeight
using Aeolus.IO: checkpoint_state, restore_checkpoint_state!
const PU = Aeolus.Urban
const PC = Aeolus.Components

include(joinpath(@__DIR__, "support", "berlin_configs.jl"))

const PM = RegularGrid(; size = (6, 4, 8), extent = (600.0, 400.0, 800.0), halo = (2, 2, 2))
pbcs() = (bcs = FieldBoundaryConditions(:periodic, :periodic, :wall),
          θ_bcs = FieldBoundaryConditions(:periodic, :periodic, :noflux))
function p_core(core::Symbol; mesh = PM, moist = true)
    core === :boussinesq && return BoussinesqSimulation(mesh; θ₀ = 300.0, moisture = moist, pbcs()...)
    return AnelasticSimulation(mesh; ref = isentropic_reference(mesh; θ_const = 300.0), pbcs()...)
end
"The reference the components convert with: the isentropic base state (the anelastic core's own)."
p_ref(sim) = sim isa BoussinesqSimulation ? isentropic_reference(sim.mesh; θ_const = 300.0) : nothing
"θ = 300 K, a uniform wind (5, 1) m/s, qv = 8 g/kg on a moist core."
function p_state!(sim)
    interior_view(sim.θ) .= 300.0
    interior_view(sim.u) .= 5.0; interior_view(sim.v) .= 1.0
    PU._moist(sim) && (interior_view(sim.qv) .= 0.008)
    return sim
end
zcentres(m) = [m.z0 + (k - 0.5) * m.dz for k in 1:m.Nz]

"""
A tiled surface on `m`: every column a street tile (hot, dry, rough) and a
park tile (cooler, moist), 70/30; the street of the odd columns forced at
level 3 (its canopy the two levels below), of the even ones at level 1.
"""
function p_tiles(m)
    nx, ny = m.Nx, m.Ny
    street() = SlabLand(; T = 308.0, C_s = 1e6, albedo = 0.12, emissivity = 0.95, z0m = 0.5, β = 0.0)
    park() = SlabLand(; T = 299.0, C_s = 2e5, albedo = 0.2, emissivity = 0.97, z0m = 0.1, β = 0.6)
    cols = Matrix{Union{Nothing,PU.ParentColumn}}(nothing, nx, ny)
    for I in CartesianIndices((nx, ny))
        kt = isodd(I[1]) ? 3 : 1
        cols[I] = PU.ParentColumn(SurfaceMosaic([street(), park()], [0.7, 0.3]), UInt8[1, 3], [0.0, 0.0], [kt, 1], kt)
    end
    z = zeros(nx, ny)
    pl = PU.ParentLanduse(nx, ny, m.dx, 0.0, 0.0, 1, zeros(Int, nx, ny, 5), z, z, z, String[])
    return PU.ParentSurface(pl, zeros(nx, ny, 5), z, z, z, z, z, zcentres(m), nothing, cols)
end
p_tiles_opts(sim) = (; surface = p_tiles(sim.mesh), period = 60.0, u_min = 0.5, reference = p_ref(sim))

"Σ_k w_k Σ_ij (a − b) over the interior levels (J m⁻² with the heat weights)."
wsum(w, a, b) = sum(w[k] * sum(view(a, :, :, k) .- view(b, :, :, k)) for k in eachindex(w))

const NOON = (start_h = 13.0, utc_offset = 2.0, date = (2019, 6, 29), lat0 = 52.522, lon0 = 13.41, elevation = 34.0)
const NIGHT = merge(NOON, (start_h = 1.0,))

@testset "TiledUrbanSurface: the air receives the tiles' mean fluxes × Δt, u*, the implicit drag ($core)" for
        core in (:boussinesq, :anelastic)
    sim = p_state!(p_core(core))
    d0 = Domain(sim; validate = false)
    ts = PC.build(PU.TiledUrbanSurface, p_tiles_opts(sim), d0)
    d = Domain(sim; components = (CentredAdvection(), ts))
    @test Components.slot(ts) === :surface && Components.slot_component(d, :surface) === ts
    @test size(ts.θsrc, 3) == 2                          # the canopy of a tile forced at level 3
    θ0 = copy(interior_view(sim.θ))
    dt = 10.0
    Components.run_stage!(ts, d, Val(:surface_flux), dt)
    H = ts.sfc_mean[:H] / ts.period                      # the tiles' area-mean H over the surface step (W m⁻²)
    @test H > 50                                         # hot street, cool air
    got = wsum(ts.wθ, interior_view(sim.θ), θ0)          # what the air received, J m⁻²
    @test got ≈ H * dt rtol = 1e-9                      # the round-off of θ ≈ 300 K
    @test ts.sfc_mean[:delivered_H] ≈ H * dt rtol = 1e-12
    # only the canopy levels are heated: the street's two below level 3, the park's first
    @test all(interior_view(sim.θ)[:, :, 3:end] .== θ0[:, :, 3:end])
    @test all(interior_view(sim.θ)[2:2:end, :, 2] .== θ0[2:2:end, :, 2])
    if PU._moist(sim)
        E = ts.sfc_mean[:E] / ts.period
        @test E > 0
        @test wsum(ts.wq, interior_view(sim.qv), fill(0.008, size(θ0))) ≈ E * dt rtol = 1e-9
    end
    # u* of every column from its tiles' stress
    ref = ts.ref
    for I in CartesianIndices(ts.ustar)
        col = ts.surface.columns[I]
        tf = last_tile_fluxes(col.surface); fr = mosaic_fractions(col.surface)
        @test ts.ustar[I] == sqrt(max(sum(fr[n] * tf[n].τ for n in eachindex(tf)), 0.0) / ref.ρ̄[1])
    end
    @test Components.surface_friction_velocity(ts, d) === ts.ustar
    # the drag: u ← u / (1 + Δt a / U) at the west face of column (1, 1) (a
    # the mean of the two columns it separates, U the speed there, the
    # stress along the wind); the first face the sweep updates, so its v
    # neighbours are still the predictor's
    m = sim.mesh
    I, J, K = m.Hx + 1, m.Hy + 1, m.Hz + 1
    fill_halos!(sim.u, sim.bcs); fill_halos!(sim.v, sim.bcs)
    u0 = sim.u.data[I, J, K]
    vf = 0.25 * (sim.v.data[I-1, J, K] + sim.v.data[I, J, K] + sim.v.data[I-1, J+1, K] + sim.v.data[I, J+1, K])
    a = 0.5 * (ts.drag[m.Nx, 1, 1] + ts.drag[1, 1, 1])
    Components.run_stage!(ts, d, Val(:post_predictor), dt)
    @test sim.u.data[I, J, K] ≈ u0 / (1 + dt * a / max(hypot(u0, vf), 0.5)) rtol = 1e-14
    @test sim.u.data[I, J, K] < u0
    @test sim.u.data[I, J, m.Hz+3] == 5.0              # above every canopy: no drag
end

@testset "TiledUrbanSurface: a step through the pipeline on both cores, and its checkpoint ($core, $sch)" for
        core in (:boussinesq, :anelastic), sch in (:euler, :rk3)
    mk() = (s = p_state!(p_core(core)); d0 = Domain(s; scheme = sch, validate = false);
            Domain(s; scheme = sch, components = (CentredAdvection(), PC.build(PU.TiledUrbanSurface, p_tiles_opts(s), d0))))
    d = mk()
    step!(d, 5.0)
    @test all(isfinite, interior_view(d.state.θ)) && all(isfinite, interior_view(d.state.u))
    ts = Components.slot_component(d, :surface)
    st = checkpoint_state(ts)
    d2 = mk()
    step!(d2, 5.0)                                      # the same state, the tiles then overwritten
    restore_checkpoint_state!(Components.slot_component(d2, :surface), st)
    for x in (d, d2)
        step!(x, 5.0); step!(x, 70.0)                   # into the second surface interval
    end
    @test d.state.θ.data == d2.state.θ.data && d.state.u.data == d2.state.u.data
    @test Components.slot_component(d, :surface).sfc_mean == Components.slot_component(d2, :surface).sfc_mean
end

@testset "PrescribedSurfaceFlux: Q Δt/Δz₁ into every column of the first level ($core)" for core in (:boussinesq, :anelastic)
    sim = p_state!(p_core(core))
    d0 = Domain(sim; validate = false)
    pf = PC.build(PU.PrescribedSurfaceFlux, (; q_surface = PU.ConstantForcing(0.12),
                                             e_surface = PU.SeriesForcing([0.0, 100.0], [0.0, 2e-5]),
                                             reference = p_ref(sim)), d0)
    d = Domain(sim; components = (CentredAdvection(), pf))
    θ0 = copy(interior_view(sim.θ))
    Components.run_stage!(pf, d, Val(:surface_flux), 10.0)
    @test interior_view(sim.θ)[:, :, 1] ≈ θ0[:, :, 1] .+ 0.12 * 10.0 / PM.dz rtol = 1e-15
    @test interior_view(sim.θ)[:, :, 2:end] == θ0[:, :, 2:end]
    ref = pf.ref
    @test pf.sfc_mean[:H] ≈ 0.12 * 10.0 * ref.ρ̄[1] * ref.cp * ref.π̄[1] rtol = 1e-14
    # E over [0, 10] s of the series: its exact mean, 1e-6 kg/kg m/s
    PU._moist(sim) && @test interior_view(sim.qv)[:, :, 1] ≈ fill(0.008 + 1e-6 * 10.0 / PM.dz, PM.Nx, PM.Ny) rtol = 1e-14
    @test Components.wall_theta_flux(pf, d) == (0.0, 0.0)       # none to the TKE (as the parent since #259)
end

@testset "MeanColumnRadiation: the site's sun, Ṫ Δt / Π̄, the tiles' downwelling, one call per period ($core)" for
        core in (:boussinesq, :anelastic)
    for (cfg, day) in ((NOON, true), (NIGHT, false))
        sim = p_state!(p_core(core))
        d0 = Domain(sim; validate = false)
        ts = PC.build(PU.TiledUrbanSurface, p_tiles_opts(sim), d0)
        rad = PC.build(PU.MeanColumnRadiation, (; period = 600.0, sun = PU.SiteSun(cfg), reference = p_ref(sim)), d0)
        d = Domain(sim; components = (CentredAdvection(), ts, rad))
        # the surface first, then the radiation over it (stage 3, order 1)
        @test findfirst(c -> c === ts, d.components) < findfirst(c -> c === rad, d.components)
        θ0 = copy(interior_view(sim.θ))
        Components.run_stage!(rad, d, Val(:surface_flux), 10.0)
        @test rad.model.ncalls == 1 && rad.t_rad[] == 600.0
        @test (ts.sw_down[], ts.lw_down[]) == (rad.sw_down[], rad.lw_down[])
        @test 250 < rad.lw_down[] < 450
        if day
            @test rad.sw_down[] > 600                    # clear sky, 13:00 CEST, 29 June, Berlin
            @test maximum(rad.model.heat_sw) > 0         # the vapour absorbs sunlight
        else
            @test rad.sw_down[] == 0
            @test all(==(0), rad.model.heat_sw) && rad.model.heat == rad.model.heat_lw
        end
        π̄ = rad.π̄
        for k in 1:PM.Nz
            @test interior_view(sim.θ)[:, :, k] ≈ θ0[:, :, k] .+ rad.model.heat[k] * 10.0 / π̄[k] rtol = 1e-15
        end
        # the next step, within the period: the heating again, no call
        sim.t = 10.0
        Components.run_stage!(rad, d, Val(:surface_flux), 10.0)
        @test rad.model.ncalls == 1
        # its checkpoint: the last call's fluxes and heating, the next call's time
        st = checkpoint_state(rad)
        h = copy(rad.model.heat); sw = rad.sw_down[]
        rad.model.heat .= 0; rad.t_rad[] = 0.0; rad.sw_down[] = -1.0
        restore_checkpoint_state!(rad, st)
        @test rad.model.heat == h && rad.t_rad[] == 600.0 && rad.sw_down[] == sw
    end
end

@testset "LargeScale: the boundary-layer tracker, the nudging weight above it, a checkpoint ($core)" for
        core in (:boussinesq, :anelastic)
    sim = p_core(core; moist = false)
    zi0 = 400.0
    θm(z) = z < zi0 ? 300.0 : 300.0 + 0.01 * (z - zi0)
    set!(sim.θ, (x, y, z) -> θm(z))
    tr = BoundaryLayerTracker(; floor = 150.0, every = 2)
    nudge = ProfileNudging((θ = (z, t) -> θm(z) + 1.0,), 3600.0;
                           weight = NudgingWeight(_ -> nudging_height(tr); factor = 1.2, depth = 0.0))
    ls = LargeScale(nothing, (nudge,); zi = tr)
    d = Domain(sim; components = (CentredAdvection(), ls))
    @test any(s -> s.name === :column_tendency, Components.stages(ls))
    @test nudging_height(tr) == 150.0                    # the floor before the first estimate
    Components.run_stage!(ls, d, Val(:column_tendency), 1.0)
    @test tr.zi[] == 0.0                                  # every 2 steps
    Components.run_stage!(ls, d, Val(:column_tendency), 1.0)
    @test abs(tr.zi[] - zi0) <= PM.dz                     # the parcel height of the mixed layer, to a level
    @test tr.zi[] == parcel_zi(sim)
    # the nudging acts above 1.2 z_i only
    fill!(sim.tθ.data, 0.0)
    Components.run_stage!(ls, d, Val(:body_force), 1.0)
    zc = zcentres(PM)
    tθ = interior_view(sim.tθ)
    zn = 1.2 * nudging_height(tr)
    @test all(tθ[:, :, k] == zeros(PM.Nx, PM.Ny) for k in 1:PM.Nz if zc[k] <= zn)
    @test all(all(tθ[:, :, k] .≈ 1 / 3600) for k in 1:PM.Nz if zc[k] > zn)
    # its state is the tracker's
    s = checkpoint_state(ls)
    zi = tr.zi[]; tr.zi[] = 0.0; tr.n[] = 7
    restore_checkpoint_state!(ls, s)
    @test tr.zi[] == zi && tr.n[] == 2
end

@testset "LargeScale clocks in an SSP-RK3 step: the step mean, or the core's substage clock ($core)" for
        core in (:boussinesq, :anelastic)
    c = 1e-4                                              # dθ/dt = c t (K s⁻¹, t in s)
    Δθ = Dict{Symbol,Float64}()
    for clk in (:step, :core)
        sim = p_core(core; moist = false)
        interior_view(sim.θ) .= 300.0
        sim.t = 100.0
        ls = LargeScale(nothing, (AdvectiveTendencies(; θ = (z, t) -> c * t),); clock = clk)
        d = Domain(sim; scheme = :rk3, components = (CentredAdvection(), ls))
        θ0 = interior_view(sim.θ)[3, 2, 4]
        step!(d, 10.0)
        Δθ[clk] = interior_view(sim.θ)[3, 2, 4] - θ0
    end
    t0, Δt = 100.0, 10.0
    # :step — the exact step mean c (t0 + Δt/2) Δt;
    # :core — windows [t0 + (s-1)Δt, t0 + sΔt] weighted 1/6, 1/6, 2/3: c (t0 + 2Δt) Δt
    @test Δθ[:step] ≈ c * (t0 + Δt / 2) * Δt rtol = 1e-9
    @test Δθ[:core] ≈ c * (t0 + 2Δt) * Δt rtol = 1e-9
    @test_throws ArgumentError LargeScale(nothing, (); clock = :substage)
end

@testset "[sgs] model = \"tke\" on a stretched grid: the exact wall distance" begin
    faces = [0.0, 10.0, 25.0, 45.0, 75.0, 120.0, 180.0, 260.0, 360.0]
    m = StretchedGrid(; size = (4, 4, 8), extent = (100.0, 100.0, 360.0), halo = (2, 2, 2), stretching = (:custom, faces))
    zc = [(faces[k] + faces[k+1]) / 2 for k in 1:8]
    for (top, dist) in ((false, zc), (true, min.(zc, 360.0 .- zc)))
        sim = BoussinesqSimulation(m; θ₀ = 300.0, pbcs()...)
        interior_view(sim.θ) .= 300.0
        d0 = Domain(sim; validate = false)
        s = PC.build(SGS, (; model = :tke, e_min = 1e-4, surface_factor = 3.9, wall_cap = true, wall_cap_top = top,
                            surface_value = false, stress = :laplacian), d0)
        c = s.closure
        @test all(c.l_wall[:, :, k] == fill(0.4 * dist[k], 4, 4) for k in 1:8)
        @test all(c.first_level[:, :, 1]) && !any(c.first_level[:, :, 2:end])
    end
end

@testset "[sgs] surface_value: the first-level e is u*²/√(c_k c_ε) of the surface's u* ($core)" for
        core in (:boussinesq, :anelastic)
    sim = p_state!(p_core(core; moist = false))
    d0 = Domain(sim; validate = false)
    sl = PC.build(SurfaceLayer, (; model = :most, z0m = 0.1, z0h = nothing,
                                 temperature = Dict{String,Any}("kind" => "constant", "value" => 302.0),
                                 stability = Dict{String,Any}("family" => "businger_dyer_beljaars_holtslag"),
                                 rho_cp = 1206.0), d0)
    sgs = PC.build(SGS, (; model = :tke, e_min = 1e-4, surface_factor = 3.9, wall_cap = false, wall_cap_top = true,
                          surface_value = true, stress = :laplacian), d0)
    d = Domain(sim; components = (CentredAdvection(), sl, sgs))
    step!(d, 2.0)
    c = sgs.closure
    cε = Aeolus.PrognosticSGS.tke_dissipation_coefficient(c.Δk[1], c.Δk[1]; factor = 3.9)
    e1 = interior_view(c.e)[:, :, 1]
    @test all(sl.ustar .> 0.1)
    @test e1 ≈ Aeolus.PrognosticSGS.tke_surface_value.(sl.ustar, cε) rtol = 1e-14
    # without a surface component that computes u*, the closure says so
    sim2 = p_state!(p_core(core; moist = false))
    s2 = PC.build(SGS, (; model = :tke, e_min = 1e-4, surface_factor = 3.9, wall_cap = false, wall_cap_top = true,
                         surface_value = true, stress = :laplacian), Domain(sim2; validate = false))
    @test_throws ArgumentError step!(Domain(sim2; components = (CentredAdvection(), s2)), 2.0)
end
