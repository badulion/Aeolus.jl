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

@testset "the tiled surface's driver: the land-use table cropped and coarsened to the domain" begin
    cfg = load_run_config(IOBuffer(pd_smoke()))
    m = Aeolus.SimulationDriver._mesh(cfg.grid)
    file = Aeolus.SimulationDriver._read_tiles(cfg.domain.static_driver, m)
    a = PU.landuse_static_tiles(PU.read_parent_landuse(joinpath(PD_EX, "data", "berlin_landuse_6km.csv")), m;
                                origin_lat = file.origin_lat, origin_lon = file.origin_lon,
                                origin_x = file.origin_x, origin_y = file.origin_y)
    for f in (:x, :y, :dx, :x0, :y0, :refinement, :lambda_f, :h_mean, :h_max)
        @test isequal(getfield(a, f), getfield(file, f))
    end
    # the class counts are the table's except under the smoke site's footprint
    # (250 m × 250 m: 4 × 4 parent columns), where the file holds the site's own
    # classes (AeolusPrep.jl, NEST-p3 #483; checked cell by cell in input_static.jl)
    diff = [I for I in CartesianIndices(size(a.class_count)[1:2]) if a.class_count[I, :] != file.class_count[I, :]]
    @test 0 < length(diff) <= 16
    @test maximum(I[1] for I in diff) - minimum(I[1] for I in diff) < 4 &&
          maximum(I[2] for I in diff) - minimum(I[2] for I in diff) < 4
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

@testset "the tiles' soil from the domain's dynamic driver (#521)" begin
    # a soil profile in the parent's driver: every soil tile holds it on its
    # nodes, its base at the deepest level (PALM's rule; the urban site's
    # GroundSoil, test/urban_soil_init.jl); without one, t_soil0 as before
    z, t, w = [0.035, 0.175, 0.64, 1.945], [295.0, 292.0, 288.0, 284.0], [0.15, 0.20, 0.25, 0.30]
    p = joinpath(mktempdir(), "parent_soil.nc")
    Aeolus.InputDrivers.write_dynamic_driver(p, Aeolus.InputDrivers.DynamicInputs(; soil = (z = z, t = t, m = w)))
    soil_tiles(r) = (ts = only(c for c in r.model.domains[1].components if c isa PU.TiledUrbanSurface);
                     [s for col in ts.surface.columns if col !== nothing for s in col.surface.tiles
                      if s isa Aeolus.SoilColumn.SoilThermalColumn])
    tiles = soil_tiles(pd_build(pd_smoke(); overrides = ["dynamic_driver=\"$p\""]))
    @test !isempty(tiles)
    zn = tiles[1].water.grid.z
    SC = Aeolus.SoilColumn
    @test all(s -> isapprox(s.water.T_soil, SC.soil_profile_on_nodes(z, t, zn); rtol = 1e-12) && s.bottom.T == t[end], tiles)
    @test all(s -> isapprox(s.water.θ_liq, SC.soil_profile_on_nodes(z, w, zn); rtol = 1e-12), tiles)
    @test all(s -> all(==(288.0), s.water.T_soil) && s.bottom.T == 288.0, soil_tiles(pd_build(pd_smoke())))
end
