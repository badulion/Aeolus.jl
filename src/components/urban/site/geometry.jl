# Geometry of the Berlin case: `build_geometry`, which turns the site's options
# into the solid mask, the surface-class map and the radiosity scene with
# its view factors.
#
# The ground comes from the static driver (CFG-f, #361: `[geometry]
# static`, written by AeolusPrep.jl, `aeolusprep static`; ONE-h #457). The
# reader, the surface classes and the rasterisation are `Aeolus.InputDrivers`,
# the one copy.

# ---------------------------------------------------------------------------
# The Berlin-case geometry
# ---------------------------------------------------------------------------

"""
    Geometry

Static geometry of a case, built once by [`build_geometry`](@ref):

* `city` — the `UrbanCity` (its footprint polygons);
* `sm` — the IBM `SolidMask` of the buildings;
* `sfc` — the `(Nx, Ny)` surface-class map (`SURFACE_*`; couplings may
  add classes during `init!`, e.g. glass);
* `lai` — the leaf-area index per column (of the static driver);
* `heightmap` — building top height (m) per column;
* `scene`, `vfac` — the radiosity scene and its view factors;
* `facet_col` — the `(i, j)` column of every facet;
* `svf` — the area-weighted sky-view factor per column;
* `wall_ids` — indices of the wall facets;
* `open_cols` — the columns with no building.
"""
struct Geometry{SM, HM, SC, VF}
    city::UrbanCity
    sm::SM
    sfc::Matrix{UInt8}
    lai::Matrix{Float64}
    heightmap::HM
    scene::SC
    vfac::VF
    facet_col::Vector{Tuple{Int,Int}}
    svf::Matrix{Float64}
    wall_ids::Vector{Int}
    open_cols::Vector{CartesianIndex{2}}
end

"The mesh coordinates of the OSM anchor `cfg.lat0/lon0`: the point `cfg.offset` m from it is the domain centre."
anchor_xy(cfg, mesh) = (mesh.Lx / 2 - cfg.offset[1], mesh.Ly / 2 - cfg.offset[2])

"""
    case_static_inputs(cfg, mesh) -> StaticInputs

The case's ground, read from the static driver `cfg.static` (checked to
be on `mesh` with the anchor where `cfg` puts it, see
`InputDrivers.read_static_driver`). A case without one is an error: the
model never reads OpenStreetMap or GeoJSON itself (ONE-h, #457).
"""
function case_static_inputs(cfg, mesh)
    cfg.static === nothing &&
        error("[geometry] static is \"none\": the case has no ground. Make its static driver with AeolusPrep.jl:\n" *
              "  julia --project -m AeolusPrep static <the config.toml> --source <file.osm | file.geojson> --out <file.nc>")
    isfile(cfg.static) ||
        error("Missing the static driver $(cfg.static). Make it with AeolusPrep.jl:\n" *
              "  julia --project -m AeolusPrep static <the config.toml naming it> --source <file.osm | file.geojson>")
    ox, oy = anchor_xy(cfg, mesh)
    return read_static_driver(cfg.static; mesh = mesh, origin_lat = cfg.lat0, origin_lon = cfg.lon0,
                              origin_x = ox, origin_y = oy)
end

"""
    build_geometry(cfg, mesh) -> Geometry

The ground of the case ([`case_static_inputs`](@ref): the static driver)
as the solid mask and the
surface-class map on `mesh`, building-adjacent cells tagged
`SURFACE_WALL` (`InputDrivers.solid_and_classes`); then the radiosity
scene and its view factors (the geometry is static, so this happens
once). For an open case (`lateral_open`) the columns within
`cfg.boundary_frame` cells of the open faces are kept free of buildings
(#262). The urban case has no terrain: a driver with `zt ≠ 0` is an
error.
"""
function build_geometry(cfg, mesh)
    st = case_static_inputs(cfg, mesh)
    all(iszero, st.zt) ||
        error("the static driver $(cfg.static) has terrain (zt from $(minimum(st.zt)) to $(maximum(st.zt)) m); " *
              "the urban case has none")
    w_f = lateral_open(cfg) ? cfg.boundary_frame : 0
    sm, sfc = solid_and_classes(st, mesh; tag_walls = true, frame = (w_f, w_f), bcs = case_scalar_bcs(cfg))

    # Era III: a radiosity scene replaces the ray-cast sky-view factor.
    # Geometry is static, so facets and view factors are built once. The
    # ground facet of every open column takes the material of its surface
    # class (park, water, street), see `scene_class_map`.
    t_svf = time()
    heightmap = heightmap_from_solid_mask(sm)
    scene = build_scene(mesh, heightmap; wall_subdivisions = cfg.wall_subdivisions,
                        class_map = scene_class_map(sfc))
    vfac  = view_factors(scene)
    facet_col = Tuple{Int,Int}[f.col for f in scene.facets]
    svf = aggregate_to_grid(scene, vfac).svf
    @debug("radiosity scene", subsystem = :urban, domain = current_domain(),
           event = :radiosity_scene, facets = nfacets(scene), wall = round(time() - t_svf; digits = 2),
           reciprocity = reciprocity_error(vfac), closure = closure_error(vfac),
           svf_min = minimum(svf), svf_mean = mean(svf), svf_max = maximum(svf))
    wall_ids = [i for (i, f) in enumerate(scene.facets) if f.kind === :wall]
    open_cols = findall(==(0), heightmap)
    return Geometry(st.city, sm, sfc, st.lai, heightmap, scene, vfac, facet_col,
                    Matrix{Float64}(svf), wall_ids, open_cols)
end

"""
    scene_class_map(sfc) -> Matrix{Int}

The radiosity ground-material class (`Radiosity.URBAN_*`) of every
column, from the surface-class map `sfc`. The `SURFACE_*` codes 1–4
(street, roof, park, water) are the `URBAN_*` material indices, so they
pass through. `SURFACE_WALL` marks a street cell next to a building so
the 2-D slab can carry the wall heat load; the ground facet in that
column is still street, so it maps to `URBAN_STREET`. `build_facets`
reads the map only for open (unbuilt) columns.
"""
function scene_class_map(sfc::AbstractMatrix{UInt8})
    return [c == SURFACE_PARK  ? URBAN_PARK  :
            c == SURFACE_WATER ? URBAN_WATER :
            c == SURFACE_ROOF  ? URBAN_ROOF  : URBAN_STREET
            for c in sfc]
end

"Building height (m) per column: top of the highest solid cell."
building_height(g::Geometry, mesh) =
    [maximum(k for k in 1:mesh.Nz if g.sm.mask[mesh.Hx+i, mesh.Hy+j, mesh.Hz+k]; init = 0) * mesh.dz
     for i in 1:mesh.Nx, j in 1:mesh.Ny]

"""
    park_lai_map(case) -> (veg_mask, LAI_map)

The park columns (`SURFACE_PARK`) and their leaf-area index (the static
driver's `lai` on park cells, 0 elsewhere).
"""
function park_lai_map(case)
    sfc = case.geom.sfc
    veg_mask = (sfc .== SURFACE_PARK)
    LAI_map = case.geom.lai .* veg_mask
    return veg_mask, LAI_map
end
