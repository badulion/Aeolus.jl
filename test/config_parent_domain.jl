# The urban nest parent's physics as a model domain configured in TOML
# (NEST-p3, #483): the deep boundary-layer domain of #259 is a `[[domain]]`
# (here a single-domain config) of the model's components, every one of
# them a section of its block:
#
#   * `[grid] stretching = "surface_refined"`: Δz uniform up to z_uniform,
#     then growing by the ratio per level up to dz_max (#151), `origin`;
#   * `[surface] model = "tiles"` on the tile form of the domain's static
#     driver (`Urban.landuse_static_tiles`: the land-use table cropped and
#     coarsened to the domain), the tiles' classes those of the urban site
#     (`Urban.TileClasses`, `tile_class_defaults`);
#   * `[radiation] scheme = "mean_column"` at the sun of its date and place;
#   * `[large_scale_forcing]` from a sounding driver: the geostrophic wind
#     of the sounding above z_g, the nudging above the tracked boundary
#     layer, the θ damping toward the sounding over the top;
#   * `[initial] u, v = {kind = "geostrophic"}`, θ and qv of the sounding
#     driver at the start, `[boundaries] theta_top = "initial_gradient"`;
#   * the hydrostatic reference of a Boussinesq domain whose components
#     convert fluxes with one (`Urban.domain_reference`).
# The physics are asserted: the grid's spacing, the profiles, the fixed
# top gradient, the hydrostatic balance, and the domain's energy and water
# budgets closed over 12 adaptive steps (every component's change of θ and
# qv attributed).

using Test
using TOML
using Aeolus
using Aeolus.Config
using Aeolus.Fields: interior_view
using Aeolus.SimulationDriver: SimulationDriverType
using Aeolus.MoistThermodynamics: DEFAULT_CONSTANTS, R_d
using Statistics: mean

const PD = Aeolus.Components
const PU = Aeolus.Urban
const PD_EX = joinpath(pkgdir(Aeolus), "examples", "urban")

# The smoke parent of berlin_smoke.toml (1 km × 1 km × 500 m on 16 × 16 × 10
# cells at local noon, the idealised profile of the smoke site) as a model domain.
pd_smoke(; extra = "") = """
    schema_version = 2
    static_driver = "$(joinpath(PD_EX, "input", "berlin_smoke_parent_static.nc"))"
    [run]
    name = "parent_smoke"
    seed = 259
    [time]
    dt = "adaptive"
    cfl = 0.9
    dt_max = 10.0
    duration = 120.0
    [grid]
    size = [16, 16, 10]
    extent = [1000.0, 1000.0, 500.0]
    halo = [2, 2, 2]
    origin = [-375.0, -375.0]
    [dynamics]
    nu = 0.0
    theta0 = 293.0
    moisture = true
    [boundaries]
    z = "free_slip"
    theta_top = "initial_gradient"
    [initial]
    theta = {kind = "mixed_layer_lapse", value = 293.0, z_i = 60.0, lapse = 0.003}
    qv = {kind = "exponential", value = 0.01, scale_height = 800.0}
    u = {kind = "geostrophic"}
    v = {kind = "geostrophic"}
    perturbation_amplitude = 0.4
    perturbation_depth = 100.0
    [coriolis]
    f = 1e-4
    [sponge]
    alpha_max = 0.01
    z_start = 375.0
    [sgs]
    model = "tke"
    e_min = 1e-4
    wall_cap = true
    wall_cap_top = false
    surface_value = true
    [surface]
    model = "tiles"
    [radiation]
    scheme = "mean_column"
    start_hour = 12.0
    [large_scale_forcing]
    u_g = 1.0
    v_g = 0.0
    damping_rate = 0.01
    [output]
    variables = []
    """ * extra

pd_build(text; overrides = String[]) =
    Aeolus.Config.build(SimulationDriverType(), load_run_config(IOBuffer(text); overrides = vcat(["fftw.planner=\"estimate\""], overrides)))

@testset "[grid] stretching = \"surface_refined\": uniform levels, then a geometric growth up to dz_max" begin
    t = pd_smoke() * ""
    # the 6 km parent of the flagship: 25 m to 500 m, × 1.08 per level, at most 100 m: 53 levels
    big = replace(t, "size = [16, 16, 10]\nextent = [1000.0, 1000.0, 500.0]" =>
                     "size = [8, 8, 53]\nextent = [1000.0, 1000.0, 3000.0]\nstretching = \"surface_refined\"\n" *
                     "dz_surface = 25.0\nz_uniform = 500.0\nstretching_parameter = 1.08\ndz_max = 100.0")
    cfg = load_run_config(IOBuffer(big))
    m = Aeolus.SimulationDriver._mesh(cfg.grid)
    dz = [m.zf[m.Hz+k+1] - m.zf[m.Hz+k] for k in 1:m.Nz]
    @test m.Nz == 53 && m.zf[m.Hz+m.Nz+1] ≈ 3000.0
    @test all(≈(25.0), dz[1:20])                                   # 20 × 25 m up to 500 m
    @test maximum(dz) <= 100.0 && dz[end] > 95.0
    @test all(k -> dz[k+1] <= 1.08 * dz[k] * (1 + 1e-12), 20:52)  # the growth never exceeds the ratio
    @test m.x0 == -375.0 && m.y0 == -375.0                         # [grid] origin
end

@testset "the smoke parent as a model domain: components, initial state, top gradient, reference" begin
    r = pd_build(pd_smoke())
    d = only(r.model.domains); s = d.state; m = s.mesh
    @test map(PD.component_name, d.components) ==
          (:advection, :tiled_surface, :sgs, :mean_column_radiation, :coriolis, :large_scale_forcing, :sponge)
    # u, v: the geostrophic wind of [large_scale_forcing] everywhere
    @test all(==(1.0), interior_view(s.u)) && all(==(0.0), interior_view(s.v))
    # θ: the mixed-layer-and-lapse profile plus the noise in the lowest 100 m (|noise| ≤ 0.2 K)
    zc = [m.z0 + (k - 0.5) * m.dz for k in 1:m.Nz]
    θp(z) = z < 60 ? 293.0 : 293.0 + 0.003 * (z - 60)
    θi = interior_view(s.θ)
    @test all(k -> zc[k] >= 100 ? all(==(θp(zc[k])), θi[:, :, k]) :
                                  maximum(abs, θi[:, :, k] .- θp(zc[k])) <= 0.2 + 1e-12, 1:m.Nz)
    @test any(k -> zc[k] < 100 && maximum(abs, θi[:, :, k] .- θp(zc[k])) > 0.05, 1:m.Nz)
    # the fixed-gradient θ top (#254): the profile's 3 K/km
    @test s.θ_bcs.top.f ≈ 0.003
    # the hydrostatic reference the tiles and the radiation convert with
    ref = d.drivers.reference
    c = DEFAULT_CONSTANTS
    @test ref.p_surface == 101325.0
    @test all(diff(ref.p̄) .< 0) && ref.p̄ ≈ ref.ρ̄ .* R_d(c) .* ref.T̄
    # dp/dz = -ρ g between the first two levels (to the trapezoid's order)
    @test (ref.p̄[2] - ref.p̄[1]) / m.dz ≈ -9.81 * (ref.ρ̄[1] + ref.ρ̄[2]) / 2 rtol = 1e-4
    @test PD.slot_component(d, :surface).ref === ref
end

@testset "the smoke parent: 12 adaptive steps, the energy and water budgets closed" begin
    r = pd_build(pd_smoke(); overrides = ["checks.energy_closure={required=true}", "checks.water_closure={required=true}"])
    d = only(r.model.domains); s = d.state
    @test Set(d.budgets.quantities) == Set([:energy, :water])
    dt = nothing
    for _ in 1:12
        _, dt, _ = PD.step_adaptive!(r.model; dt_prev = something(dt, 2.0))
    end
    @test s.t > 12 * 2.0 && all(isfinite, interior_view(s.θ)) && all(isfinite, interior_view(s.qv))
    cs = Dict(c.name => c for c in PD.domain_checks(d))
    for n in ("energy_closure", "water_closure")
        v, info = cs[n].evaluate(cs[n].acc)
        @info "smoke parent domain: $n $(v) over $(info["elapsed"]) s"
        @test info["unattributed"] == 0.0
        @test v <= (n == "energy_closure" ? 1e-6 : 1e-12)
    end
    # the cold tiles (285 K, water 290 K) under 293 K air at noon cool the first level
    @test mean(interior_view(s.θ)[:, :, 1]) < 293.0
    # the surface delivered heat: what it booked as delivered is what the air gained in its entries
    sfc = PD.slot_component(d, :surface)
    @test sfc.sfc_mean[:delivered_H] < 0 && isfinite(sfc.sfc_mean[:Rn])
end

@testset "the tiled surface's driver: the land-use table cropped and coarsened to the domain" begin
    cfg = load_run_config(IOBuffer(pd_smoke()))
    m = Aeolus.SimulationDriver._mesh(cfg.grid)
    file = Aeolus.SimulationDriver._read_tiles(cfg.domain.static_driver, m)
    a = PU.landuse_static_tiles(PU.read_parent_landuse(joinpath(PD_EX, "data", "berlin_landuse_6km.csv")), m;
                                origin_lat = file.origin_lat, origin_lon = file.origin_lon,
                                origin_x = file.origin_x, origin_y = file.origin_y)
    for f in (:x, :y, :dx, :x0, :y0, :refinement, :class_count, :lambda_f, :h_mean, :h_max)
        @test isequal(getfield(a, f), getfield(file, f))
    end
    # every column's class counts are its refinement² child cells
    @test all(==(a.refinement^2), sum(a.class_count; dims = 3))
    # a box off the table's cell faces is refused
    m2 = Aeolus.Grids.RegularGrid(; size = (16, 16, 10), extent = (1000.0, 1000.0, 500.0), origin = (-370.0, -375.0, 0.0))
    @test_throws ErrorException PU.landuse_static_tiles(PU.read_parent_landuse(joinpath(PD_EX, "data", "berlin_landuse_6km.csv")), m2;
                                                        origin_lat = file.origin_lat, origin_lon = file.origin_lon,
                                                        origin_x = file.origin_x, origin_y = file.origin_y)
end

@testset "the tiles' classes are the urban site's" begin
    d = PU.tile_class_defaults()
    tc = PU.TileClasses((;))
    @test tc.z0m == d.z0m && tc.Ts0 == 285.0 && tc.ΔT_water == 5.0 && tc.T_soil0 == 288.0
    @test tc.soil !== nothing && Set(tc.soil.classes) == Set(d.soil_classes)
    # a soil class is a soil column with the class's albedo; a built class a slab
    park = PU.class_surface(tc, PU.SurfaceSite(PU.SURFACE_PARK; T_skin = 285.0))
    roof = PU.class_surface(tc, PU.SurfaceSite(PU.SURFACE_ROOF; T_skin = 285.0))
    @test park isa Aeolus.SoilColumn.SoilThermalColumn && park.albedo == tc.params.albedo[PU.SURFACE_PARK]
    @test roof isa Aeolus.Radiation.SlabLand && roof.albedo == tc.params.albedo[PU.SURFACE_ROOF]
    @test PU.tile_skin(tc, PU.SURFACE_WATER) == 290.0 && PU.tile_skin(tc, PU.SURFACE_PARK) == 285.0
    # the options of [surface] model = "tiles" set them
    tc2 = PU.TileClasses((; ts0 = 280.0, z0m = Dict("3" => 0.5)))
    @test tc2.Ts0 == 280.0 && tc2.z0m[PU.SURFACE_PARK] == 0.5
end

@testset "a deep domain from a sounding driver: θ, qv at the start, the geostrophic wind above z_g, the damping" begin
    dyn = joinpath(PD_EX, "input", "berlin_20190629_dynamic.nc")
    text = replace(pd_smoke(), "[initial]\ntheta = {kind = \"mixed_layer_lapse\", value = 293.0, z_i = 60.0, lapse = 0.003}\n" *
                               "qv = {kind = \"exponential\", value = 0.01, scale_height = 800.0}\n" => "[initial]\n")
    text = replace(text, "schema_version = 2\n" => "schema_version = 2\ndynamic_driver = \"$dyn\"\n")
    text = replace(text, "u_g = 1.0\nv_g = 0.0\ndamping_rate = 0.01\n" =>
                         "u_g = \"sounding\"\nv_g = \"sounding\"\nsounding_nudging_tau = 21600.0\n" *
                         "damping_rate = 0.01\ndamping_target = \"sounding\"\n")
    text = replace(text, "start_hour = 12.0" => "start_hour = 6.0")
    r = pd_build(text)
    d = only(r.model.domains); s = d.state; m = s.mesh
    dd = d.drivers.dynamic
    zc = [m.z0 + (k - 0.5) * m.dz for k in 1:m.Nz]
    Tθ, Tq = dd.tables[:profile_theta], dd.tables[:profile_qv]
    fv = Aeolus.LargeScaleForcing.forcing_value
    # θ above the noise and qv everywhere: the sounding driver's at the start
    @test all(k -> all(==(fv(Tθ, zc[k], 0.0)), interior_view(s.θ)[:, :, k]), findall(>=(100), zc))
    @test all(k -> all(==(fv(Tq, zc[k], 0.0)), interior_view(s.qv)[:, :, k]), 1:m.Nz)
    # the geostrophic wind is the sounding's at max(z, z_g): constant below z_g (1.2 × the largest parcel
    # z_i of the six launches, 1356 m; the urban site's own sounding gives the same, Urban.sounding_zi_max)
    zi = PD.sounding_zi_max(Tθ, Tq)
    @test 1300 < zi < 1400
    Ug, Vg = PD.sounding_geostrophic(dd, NaN)
    @test Ug(0.0, 0.0) == Ug(1.2zi, 0.0) == fv(dd.tables[:profile_u], 1.2zi, 0.0)
    @test Ug(2000.0, 0.0) == fv(dd.tables[:profile_u], 2000.0, 0.0)
    @test all(k -> all(==(Ug(zc[k], 0.0)), interior_view(s.u)[:, :, k]), 1:m.Nz)
    # the large-scale forcing: the nudging above the tracked boundary layer (its floor the sounding's z_i), the damping
    ls = only(c for c in d.components if c isa PD.LargeScale)
    @test ls.zi !== nothing && PD.nudging_height(ls.zi) == zi
    damp = only(f for f in ls.extra if f isa Aeolus.LargeScaleForcing.ReferenceDamping)
    nudge = only(f for f in ls.extra if f isa Aeolus.LargeScaleForcing.ProfileNudging)
    @test nudge.τ == 21600.0
    @test damp !== nothing
    for _ in 1:3
        PD.step_adaptive!(r.model; dt_prev = 2.0)
    end
    @test all(isfinite, interior_view(s.θ))
end
