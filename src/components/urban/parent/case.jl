# The Berlin parent case (N-parent-e, #259): build, step, run.
#
# The parent assembles its physics from src and from the child's case code;
# it carries no copy of either:
#
#   core        BoussinesqSimulation on the StretchedGrid (#151), moist, f
#               of the child; the core's Rayleigh sponge damps u, v, w over
#               the top `damping_frac`;
#   SGS         the child's default `:sgs` provider, PrognosticSGS's TKE
#               closure (on the stretched grid since this issue), l capped
#               at κz, Moeng's c_ε factor in the first level, the surface
#               value from the tiles' u*;
#   time        TimeIntegration.step_adaptive! with the child's AdaptiveStep
#               constraints (`default_options(AdaptiveStep)`, the advective
#               Courant number 0.9), each step SSP-RK3 (`step_rk3!`);
#   initial     θ, qv from the child's `initial_θ` / `initial_qv` (the
#               Lindenberg sounding of #247 interpolated to the case start),
#               u, v = the geostrophic wind;
#   forcing     LargeScaleForcing: `GeostrophicPressureGradient` of the
#               sounding wind above z_g; `ProfileNudging` of θ, qv toward the
#               time-interpolated sounding (τ = 6 h) above 1.2 × z_i (#248);
#               `Subsidence` w_s = −D z (#248); `ReferenceDamping` of θ
#               toward the sounding over the top 25 % (#254);
#   top         the fixed-gradient θ top (#254, `θ_top_gradient` of the
#               child's initial profile at the parent's top face);
#   surface     the tiled urban mosaic (#249, `build_parent_surface`, the
#               child's class-to-surface factory) under the parent's air,
#               or a prescribed kinematic flux (the idealised tests);
#   radiation   the child's column radiation (`column_radiation!` on the
#               parent's mean column, `stitched_column`), the sun of the
#               child's clock (`solar_position_at`);
#   ledgers     two `StageBudget`s (#253): the heat of the air
#               ``Σ ρ̄ c_p Π̄ θ ΔV / A`` and the water of the air
#               ``Σ ρ̄ q_v ΔV / A`` (the same meter with Π̄ = c_p = 1). Every
#               change is attributed: the surface and radiation stages by
#               their contexts, the dynamics split into transport and
#               redistribution, with the damping, nudging, subsidence and
#               top-flux terms booked from inside it.

"""
    ParentCase

A built Berlin parent run: `pc` ([`ParentConfig`](@ref)), `mesh`, `sim`
(the Boussinesq core), `ref` (the hydrostatic reference state, a
[`CaseReference`](@ref)), `domain` (the parent's
`Components.Domain`: its physics as components, [`parent_domain`](@ref)),
and, the same objects as its components': `closure` (the SGS closure or
`nothing`), `forcing` (named tuple of the large-scale forcings),
`surface` (the [`ParentSurface`](@ref) or `nothing`), `rad` (the
`Aeolus.ColumnRadiation.ColumnRadiationModel` or `nothing`), `sfc_mean`
(the surface's domain-mean J m⁻²); the ledgers `heat` and `water` ([`StageBudget`](@ref)),
`top` (the [`TopHeatFlux`](@ref) or `nothing`), the time-stepping state,
the sounding time `t0` of the start and `diag`, the diagnostics recorded
every `output_period`.

`ParentCase` is the shell the nest and the `urban_parent` run type step
([`step_parent!`](@ref): the components' stage entries in the pipeline's
order inside the parent's ledgers); the domain alone runs through the
model's pipeline (`Components.step_adaptive!`) bit for bit the same
(NEST-p1, #481).
"""
mutable struct ParentCase{M, S, C, D}
    pc::ParentConfig
    mesh::M
    sim::S
    ref::CaseReference
    closure::C
    domain::D
    forcing::NamedTuple
    surface::Any
    child::Any
    rad::Any
    sfc_mean::Dict{Symbol,Float64}
    heat::StageBudget
    water::StageBudget
    top::Union{Nothing,TopHeatFlux}
    constraints::CFLConstraints{Float64}
    dt_prev::Float64
    t0::Float64                 # sounding time (s) of the start; NaN without a sounding
    stage::Base.RefValue{Int}   # stage of the current dynamics step (1 for Euler, 1-3 for RK3)
    top_acc::Base.RefValue{Float64}   # stage-weighted top heat flux of the step (W m⁻²)
    # per-step scratch of the dynamics hooks
    lev::Dict{Symbol,Vector{Float64}}
    wθ::Vector{Float64}         # meter weights (J m⁻² per K of level sum)
    wq::Vector{Float64}         # water weights (kg m⁻² per kg/kg of level sum)
    # bookkeeping
    nsteps::Int
    retries::Int
    dt_hist::Vector{Float64}
    timing::Dict{Symbol,Float64}
    diag::Vector{Any}
end

# ---------------------------------------------------------------------------
# Reference state, initial state, sounding-derived forcing
# ---------------------------------------------------------------------------

"The initial θ(z) of the parent: `pc.θ_init`, else the child's `initial_θ`."
parent_θ_init(pc::ParentConfig) = pc.θ_init === nothing ?
    (θi = initial_θ(pc.child); z -> θi(0.0, 0.0, z)) : pc.θ_init
"The initial qv(z) of the parent: `pc.qv_init`, else the child's `initial_qv`."
parent_qv_init(pc::ParentConfig) = pc.qv_init === nothing ?
    (qi = initial_qv(pc.child); z -> qi(0.0, 0.0, z)) : pc.qv_init

"""
    parent_reference(pc, mesh) -> CaseReference

The hydrostatic reference state of the parent on its (stretched) levels:
θ̄ the initial profile ([`parent_θ_init`](@ref)), p̄ from
[`hydrostatic_pressure`](@ref) up from the child's ground pressure
(`case_surface_pressure`), ``Π̄ = (p̄/p_{ref})^{R_d/c_{pd}}``,
``T̄ = θ̄Π̄``, ``ρ̄ = p̄/(R_d T̄)``, with the dry-air constants of
`MoistThermodynamics` — the constants of the child's reference state.
"""
function parent_reference(pc::ParentConfig, mesh)
    c = Aeolus.MoistThermodynamics.DEFAULT_CONSTANTS
    R_d = c.gas_constant / c.molmass_dry
    κ = R_d / c.cp_dry
    θf = parent_θ_init(pc)
    zc = level_centres(mesh)
    ps = case_surface_pressure(pc.child)
    p̄ = hydrostatic_pressure(θf, zc, ps; g = pc.child.g)
    θ̄ = θf.(zc)
    π̄ = (p̄ ./ c.p_ref) .^ κ
    T̄ = θ̄ .* π̄
    ρ̄ = p̄ ./ (R_d .* T̄)
    return CaseReference(p̄, ρ̄, θ̄, T̄, π̄, nothing, ps, Float64(c.cp_dry))
end

"""
    sounding_zi_max(cfg) -> m

The largest parcel boundary-layer height (`boundary_layer_height`,
`:parcel`, on θ_v) of the launches of the child's sounding: 1171 m for the
10:45 UTC release of 29 June 2019 (#247). 0 without a sounding.
"""
function sounding_zi_max(cfg::NamedTuple)
    S = case_sounding(cfg)
    S === nothing && return 0.0
    MT = Aeolus.MoistThermodynamics
    P = S.profiles
    zi = 0.0
    for t in S.times                                     # the launches
        θv = [MT.virtual_temperature(forcing_value(P.θ, z, t), MT.PhasePartition(forcing_value(P.qv, z, t)))
              for z in P.z]
        zi = max(zi, boundary_layer_height(P.z, θv; method = :parcel))
    end
    return zi
end

"""
    parent_geostrophic_wind(pc, t0, z_g) -> (U_g, V_g)

The geostrophic wind of the parent as functions `(z, t)` (t the
simulation time): the sounding wind (`sounding_profiles` u, v,
linear in time between the launches) at ``\\max(z, z_g)`` — the wind
above the boundary layer is taken as geostrophic, and held constant with
height below `z_g` (no thermal wind inside the boundary layer). `z_g` is
`pc.z_g`, or 1.2 × [`sounding_zi_max`](@ref). Without a sounding
(an idealised child profile), the constant `(U_g, V_g)` of the child's
`:geostrophic` coupling options. The nested child's `:geostrophic`
coupling takes the same functions ([`nest_geostrophic_wind`](@ref)).
"""
function parent_geostrophic_wind(pc::ParentConfig, t0::Real, z_g::Real)
    S = case_sounding(pc.child)
    if S === nothing
        # no sounding: the child's own geostrophic wind (its :geostrophic
        # coupling options), so that a nest is driven by one wind
        g = component_options(GeostrophicCoupling, pc.child)
        U, V = Float64(g.U_g), Float64(g.V_g)
        return ((z, t) -> U, (z, t) -> V)
    end
    P = S.profiles
    return ((z, t) -> forcing_value(P.u, max(z, z_g), t0 + t),
            (z, t) -> forcing_value(P.v, max(z, z_g), t0 + t))
end

"The height (m) above which the sounding wind is geostrophic: `pc.z_g`, or 1.2 × [`sounding_zi_max`](@ref)."
parent_z_g(pc::ParentConfig) = isnan(pc.z_g) ? 1.2 * sounding_zi_max(pc.child) : pc.z_g

"""
    parent_geostrophic(pc) -> (U_g, V_g)

The geostrophic wind of the parent of `pc`, functions `(z, t)` of the
simulation time: [`parent_geostrophic_wind`](@ref) at the sounding time
of the case start and [`parent_z_g`](@ref). The one definition the
parent's `GeostrophicPressureGradient` and the nested child's
`:geostrophic` coupling ([`nest_geostrophic_wind`](@ref)) are built from.
"""
function parent_geostrophic(pc::ParentConfig)
    S = case_sounding(pc.child)
    return parent_geostrophic_wind(pc, S === nothing ? 0.0 : case_start_time(pc.child), parent_z_g(pc))
end

"Sounding table `name` as a function `(z, t)` of the simulation time."
function sounding_target(pc::ParentConfig, t0::Real, name::Symbol)
    S = case_sounding(pc.child)
    tab = getproperty(S.profiles, name)
    return (z, t) -> forcing_value(tab, z, t0 + t)
end

# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------

"""
    crop_landuse(pl, pc::ParentConfig) -> ParentLanduse
    crop_landuse(pl, Is, Js) -> ParentLanduse

The land-use table under the parent box of `pc` (its centre
[`parent_center`](@ref) and extent): the cells `Is × Js` of `pl` with
`x0`, `y0` moved to the box's corner. The full 6 km table when the box is
the table's own; an error when the box is not on the table's cell faces
or leaves it. The small parent of the `:smoke` nest (1 km, #265) is a
crop of the 6 km table, coarsened by `build_parent`.
"""
function crop_landuse(pl::ParentLanduse, pc::ParentConfig)
    c = parent_center(pc)
    fi = (c[1] - pc.extent[1] / 2 - pl.x0) / pl.dx
    fj = (c[2] - pc.extent[2] / 2 - pl.y0) / pl.dx
    ni, nj = pc.extent[1] / pl.dx, pc.extent[2] / pl.dx
    all(x -> abs(x - round(x)) < 1e-9, (fi, fj, ni, nj)) ||
        error("crop_landuse: the parent box is not on the land-use cell faces ($fi, $fj, $ni, $nj cells)")
    I0, J0 = round(Int, fi) + 1, round(Int, fj) + 1
    Is, Js = I0:I0+round(Int, ni)-1, J0:J0+round(Int, nj)-1
    (first(Is) >= 1 && first(Js) >= 1 && last(Is) <= pl.nx && last(Js) <= pl.ny) ||
        error("crop_landuse: the parent box ($Is, $Js) leaves the land-use table (1:$(pl.nx), 1:$(pl.ny))")
    return crop_landuse(pl, Is, Js)
end

"""
    parent_landuse_source(pc, mesh) -> ParentLanduse

The land use of the parent `pc` on its `mesh` from the table
`pc.child.landuse` ([`read_parent_landuse`](@ref)): cropped to the box
([`crop_landuse`](@ref)) and coarsened to the mesh
([`coarsen_landuse`](@ref)). What AeolusPrep.jl's `aeolusprep static`
writes to the parent's static driver.
"""
function parent_landuse_source(pc::ParentConfig, mesh)
    pl = crop_landuse(read_parent_landuse(pc.child.landuse), pc)
    Nx = mesh.Nx
    b = pl.nx ÷ Nx
    (b * Nx == pl.nx && pl.dx * b ≈ mesh.dx) ||
        error("the land use ($(pl.nx) cells of $(pl.dx) m) does not coarsen to $Nx cells of $(mesh.dx) m")
    return coarsen_landuse(pl, b)
end

"The mesh coordinates (the child's) of the OSM anchor; the parent mesh shares them."
parent_anchor_xy(pc::ParentConfig) = (pc.child.extent[1] / 2 - pc.child.offset[1],
                                      pc.child.extent[2] / 2 - pc.child.offset[2])

"""
    parent_landuse(pc, mesh) -> ParentLanduse

The parent's land use: read from its static driver `pc.child.parent_static`
(the tile form, `InputDrivers.read_static_tiles`, checked to be on the
parent `mesh` with the anchor in place), or, when that is `nothing`,
[`parent_landuse_source`](@ref) in memory. Bit for bit the same.
"""
function parent_landuse(pc::ParentConfig, mesh)
    cfg = pc.child
    cfg.parent_static === nothing && return parent_landuse_source(pc, mesh)
    isfile(cfg.parent_static) ||
        error("Missing the parent's static driver $(cfg.parent_static). Make it with AeolusPrep.jl:\n" *
              "  julia --project -m AeolusPrep static <the config.toml naming it> --source <file.osm>\n" *
              "or set [geometry] parent_static = \"none\" to read $(cfg.landuse) in memory.")
    ox, oy = parent_anchor_xy(pc)
    t = read_static_tiles(cfg.parent_static; mesh = mesh, origin_lat = cfg.lat0, origin_lon = cfg.lon0,
                          origin_x = ox, origin_y = oy)
    size(t.class_count, 3) == 5 ||
        error("the parent's static driver $(cfg.parent_static) has $(size(t.class_count, 3)) classes, expected 5")
    return ParentLanduse(length(t.x), length(t.y), t.dx, t.x0, t.y0, t.refinement, Int.(t.class_count),
                         t.lambda_f, t.h_mean, t.h_max,
                         vcat(t.notes, "# read from the static driver $(cfg.parent_static)"))
end

"""
    parent_static_tiles(pl, pc, mesh) -> StaticTiles

The land use `pl` of the parent `pc` (on its `mesh`) as the tile form of
the static driver.
"""
function parent_static_tiles(pl::ParentLanduse, pc::ParentConfig, mesh)
    cfg = pc.child
    ox, oy = parent_anchor_xy(pc)
    return StaticTiles(collect(Float64, Aeolus.Grids.xnodes(mesh, CellCenter())), collect(Float64, Aeolus.Grids.ynodes(mesh, CellCenter())),
                       pl.dx, pl.dx, cfg.lat0, cfg.lon0, ox, oy, pl.x0, pl.y0, pl.r,
                       Int32.(pl.counts), pl.λf, pl.hmean, pl.hmax, pl.header, Dict{String,Any}())
end

"""
    parent_static_driver(pc, mesh) -> StaticTiles

The tile form of the parent's static driver (`STATIC_TILES_SCHEMA`) on
its `mesh`: the land use of [`parent_landuse`](@ref) (read from
`pc.child.parent_static`, or made in memory from the land-use table) as
`InputDrivers.StaticTiles`, what the parent domain's [`TiledUrbanSurface`](@ref)
reads (`drivers.static_tiles`).
"""
parent_static_driver(pc::ParentConfig, mesh) = parent_static_tiles(parent_landuse(pc, mesh), pc, mesh)

"""
    parent_domain(pc::ParentConfig; child = nothing) -> (domain, info)

The parent of `pc` as a `Components.Domain` (NEST-p1, #481): the mesh
([`parent_mesh`](@ref)), the reference state, the Boussinesq core with
the Coriolis parameter, the velocity sponge and the θ top of
[`ParentConfig`](@ref), the initial state, and the parent's physics as
components:

  * the `:surface` slot — [`TiledUrbanSurface`](@ref) on the tile form of the
    parent's static driver ([`parent_static_driver`](@ref),
    `drivers.static_tiles`; the class-to-surface factory and the class
    map under its footprint are those of the urban site `child`, built
    from `pc.child` if not given), or [`PrescribedSurfaceFlux`](@ref);
  * the `:radiation` slot — [`MeanColumnRadiation`](@ref) (`pc.radiation`)
    at the sun of the child's clock ([`SiteSun`](@ref));
  * `Components.LargeScale` — the geostrophic pressure gradient of the
    sounding wind ([`parent_geostrophic`](@ref)), the `ProfileNudging` of
    θ, qv toward the sounding with the `NudgingWeight` above
    `nudging_factor` × the tracked z_i (`Components.BoundaryLayerTracker`
    with the sounding's z_i as its floor), the `Subsidence` ``w_s = -D z``
    and the `ReferenceDamping` of θ, on the core's substage clock
    (`clock = :core`);
  * the `:sgs` slot — `[sgs] model = "tke"`: `Components.SGS` with the
    wall cap at the exact level heights of the stretched grid (bottom
    only) and the surface value of the tiles' u*, ``e_{min} = 10^{-4}``;
    with `sgs = :smagorinsky` the core's own Smagorinsky closure;
  * the `:advection` slot — `Components.CentredAdvection`.

The domain steps adaptively (`cfl`: the child's `AdaptiveStep`
constraints with `C_advective`, `dt_max` capped at the shortest period)
with `pc.stepper`. `info` holds `ref`, `θtop` (the core's θ top
condition), `t0` (the sounding time of the start, NaN without a sounding),
`zi_obs` and `child`.
"""
function parent_domain(pc::ParentConfig; child = nothing)
    cfg = pc.child
    mesh = parent_mesh(pc)
    Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
    H = pc.extent[3]
    ref = parent_reference(pc, mesh)
    zc = level_centres(mesh)
    dzc = level_thickness(mesh)

    # the θ top (#254) and the core
    θtop = pc.θ_top === :initial_gradient ?
        FluxBC((parent_θ_init(pc)(H + dzc[end] / 2) - parent_θ_init(pc)(H - dzc[end] / 2)) / dzc[end]) :
        pc.θ_top === :noflux ? NoFlux() :
        error("ParentConfig.θ_top must be :initial_gradient or :noflux, got $(repr(pc.θ_top))")
    pc.surface in (:tiles, :prescribed) ||
        error("ParentConfig.surface must be :tiles or :prescribed, got $(repr(pc.surface))")
    pc.sgs in (:tke, :smagorinsky) || error("ParentConfig.sgs must be :tke or :smagorinsky, got $(repr(pc.sgs))")
    z_damp = (1 - pc.damping_frac) * H
    sim = BoussinesqSimulation(mesh; ν = 0.0, θ₀ = cfg.θ₀, g_accel = cfg.g, f_coriolis = cfg.f_coriolis,
                               C_s = pc.sgs === :smagorinsky ? cfg.C_s : 0.0, Pr_t = cfg.Pr_t,
                               sponge_α_max = pc.damping_rate, sponge_z_start = z_damp,
                               moisture = pc.moisture,
                               bcs = FieldBoundaryConditions(:periodic, :periodic, :wall),
                               θ_bcs = FieldBoundaryConditions(:periodic, :periodic; bottom = :noflux,
                                                               top = θtop))

    # sounding time axis, observed z_i, geostrophic wind
    S = case_sounding(cfg)
    t0 = S === nothing ? NaN : case_start_time(cfg)
    zi_obs = sounding_zi_max(cfg)
    Ug, Vg = parent_geostrophic(pc)

    # initial state
    θf, qf = parent_θ_init(pc), parent_qv_init(pc)
    set!(sim.θ, (x, y, z) -> θf(z))
    pc.moisture && set!(sim.qv, (x, y, z) -> qf(z))
    if pc.u_init === :geostrophic
        set!(sim.u, (x, y, z) -> Ug(z, 0.0)); set!(sim.v, (x, y, z) -> Vg(z, 0.0))
    elseif pc.u_init isa Number
        set!(sim.u, (x, y, z) -> Float64(pc.u_init))
    else                                     # a profile u(z) (ProfileTable or a function)
        set!(sim.u, (x, y, z) -> Float64(pc.u_init(z)))
    end
    rng = Random.Xoshiro(pc.seed)
    θi = interior_view(sim.θ)
    for k in 1:Nz
        zc[k] < pc.perturbation_depth || break
        for j in 1:Ny, i in 1:Nx
            θi[i, j, k] += pc.perturbation * (2rand(rng) - 1)
        end
    end

    # the adaptive step: the child's AdaptiveStep constraints, dt_max at most the shortest period
    o = default_options(AdaptiveStep)
    periods = filter(isfinite, [pc.surface === :tiles ? pc.surface_period : Inf,
                                pc.radiation ? pc.radiation_period : Inf, pc.output_period])
    k = CFLConstraints(; C_advective = pc.C_advective, C_diffusive = o.C_diffusive,
                       C_buoyancy = o.C_buoyancy, C_coriolis = o.C_coriolis, dt_min = o.dt_min,
                       dt_max = min(o.dt_max, minimum(periods; init = Inf)), growth_max = o.growth_max)

    drivers = (;)
    if pc.surface === :tiles
        child === nothing && (child = build_case(cfg))
        drivers = (; static_tiles = parent_static_driver(pc, mesh))
    end
    d0 = Domain(sim; id = "parent", scheme = pc.stepper, cfl = k, drivers = drivers, validate = false)

    # the components
    comps = Any[Components.CentredAdvection()]
    push!(comps, pc.surface === :tiles ?
        Components.build(TiledUrbanSurface, (; classes = child, period = pc.surface_period, u_min = pc.U_min,
                                         reference = ref), d0) :
        Components.build(PrescribedSurfaceFlux, (; q_surface = pc.Q_surface, e_surface = pc.E_surface,
                                                  reference = ref), d0))
    pc.radiation &&
        push!(comps, Components.build(MeanColumnRadiation, (; period = pc.radiation_period, humid = season_humidity(cfg),
                                                             sun = SiteSun(cfg), reference = ref), d0))
    ls = parent_large_scale(pc, (; Ug, Vg, t0, zi_obs, z_damp, θf))
    ls === nothing || push!(comps, ls)
    pc.sgs === :tke &&
        push!(comps, Components.build(Components.SGS, (; model = :tke, e_min = 1e-4, surface_factor = 3.9,
                                                        wall_cap = true, wall_cap_top = false,
                                                        surface_value = pc.surface === :tiles,
                                                        stress = :laplacian), d0))
    d = Domain(sim; id = "parent", scheme = pc.stepper, cfl = k, drivers = drivers, components = Tuple(comps))
    return d, (; ref, θtop, t0, zi_obs, child)
end

"""
    parent_large_scale(pc, s) -> Union{Nothing,Components.LargeScale}

The large-scale forcing of the parent `pc` (see [`parent_domain`](@ref);
`s` holds the geostrophic wind `Ug`, `Vg`, the sounding time `t0`,
`zi_obs`, the damping start `z_damp` and the initial θ profile `θf`), or
`nothing` when every forcing is off.
"""
function parent_large_scale(pc::ParentConfig, s)
    sounding = case_sounding(pc.child) !== nothing
    t0 = isnan(s.t0) ? 0.0 : s.t0
    geo = pc.geostrophic ? GeostrophicPressureGradient(s.Ug, s.Vg) : nothing
    zi = nothing
    nudge = nothing
    if isfinite(pc.nudging_τ) && sounding
        zi = Components.BoundaryLayerTracker(; floor = s.zi_obs, every = 10)
        targets = pc.moisture ? (θ = sounding_target(pc, t0, :θ), qv = sounding_target(pc, t0, :qv)) :
                                (θ = sounding_target(pc, t0, :θ),)
        nudge = ProfileNudging(targets, pc.nudging_τ;
                               weight = NudgingWeight(_ -> Components.nudging_height(zi); factor = pc.nudging_factor,
                                                      depth = pc.nudging_depth))
    end
    D = pc.divergence
    subs = D != 0 ? Subsidence((z, t) -> -D * z; form = :local) : nothing
    damp = pc.damping_rate > 0 ?
        ReferenceDamping(:θ, Aeolus.Operators.sponge_profile(pc.damping_rate, s.z_damp, pc.extent[3]),
                         sounding && pc.θ_init === nothing ? sounding_target(pc, t0, :θ) :
                         (z, t) -> s.θf(z)) : nothing
    extra = Tuple(f for f in (nudge, subs, damp) if f !== nothing)
    geo === nothing && isempty(extra) && return nothing
    return Components.LargeScale(geo, extra; U_g = s.Ug, V_g = s.Vg, clock = :core, zi = zi)
end

_component(d, ::Type{T}) where {T} = (i = findfirst(c -> c isa T, d.components); i === nothing ? nothing : d.components[i])

"""
    build_parent(pc::ParentConfig; child = nothing) -> ParentCase

Build the parent: its domain of components ([`parent_domain`](@ref)) and
the shell that steps it inside the air-heat and water ledgers.
"""
function build_parent(pc::ParentConfig; child = nothing)
    d, info = parent_domain(pc; child = child)
    sim = d.state
    mesh = sim.mesh
    Nz = mesh.Nz
    ref = info.ref
    sfc = Components.slot_component(d, :surface)
    rad = Components.slot_component(d, :radiation)
    ls = _component(d, Components.LargeScale)
    extra(T) = ls === nothing ? nothing : (i = findfirst(f -> f isa T, ls.extra); i === nothing ? nothing : ls.extra[i])
    forcing = (geostrophic = ls === nothing ? nothing : ls.geostrophic, nudging = extra(ProfileNudging),
               subsidence = extra(Subsidence), damping = extra(ReferenceDamping))
    closure = Components.sgs_closure(d)

    # ledgers: heat of the air, water of the air
    hm = LevelMeter(mesh, ref.ρ̄, ref.π̄, ref.cp)
    wm = LevelMeter(mesh, ref.ρ̄, ones(Nz), 1.0)
    heat = budget_start!(StageBudget(; order = ALL_STAGES), hm; canopy_top = 100.0)
    water = budget_start!(StageBudget(; order = ALL_STAGES), wm; canopy_top = 100.0)
    θtop = info.θtop
    top = θtop isa FluxBC ? TopHeatFlux(Float64(θtop.f), ref.ρ̄[Nz] * ref.cp * ref.π̄[Nz], 0.0, 0.0) : nothing

    lev = Dict{Symbol,Vector{Float64}}(s => zeros(Nz) for s in
              (:a, :b, :θnudging, :θsubsidence, :θdamping, :qnudging, :qsubsidence,
               :acc_θnudging, :acc_θsubsidence, :acc_θdamping, :acc_qnudging, :acc_qsubsidence))
    pcase = ParentCase(pc, mesh, sim, ref, closure, d, forcing, sfc isa TiledUrbanSurface ? sfc.surface : nothing,
                       info.child, rad === nothing ? nothing : rad.model, sfc.sfc_mean,
                       heat, water, top, d.cfl, pc.child.dt, isnan(info.t0) ? 0.0 : info.t0,
                       Ref(0), Ref(0.0), lev, copy(hm.w), copy(wm.w), 0, 0, Float64[], Dict{Symbol,Float64}(), Any[])
    @debug("Berlin parent built", subsystem = :urban, domain = current_domain(), event = :parent_built,
           grid = (mesh.Nx, mesh.Ny, Nz), dx = mesh.dx, dz_min = minimum(level_thickness(mesh)),
           dz_max = maximum(level_thickness(mesh)), surface = pc.surface, sgs = pc.sgs,
           components = map(Components.component_name, d.components), z_g = parent_z_g(pc), info.zi_obs)
    return pcase
end

"""
    parent_downwelling(p) -> (sw, lw)

The downwelling short- and longwave (W m⁻²) at the parent's ground of the
last radiation call (0 and 350 W m⁻² before it, and without radiation the
surface's own).
"""
function parent_downwelling(p::ParentCase)
    r = Components.slot_component(p.domain, :radiation)
    r !== nothing && return (sw = r.sw_down[], lw = r.lw_down[])
    s = Components.slot_component(p.domain, :surface)
    return s isa TiledUrbanSurface ? (sw = s.sw_down[], lw = s.lw_down[]) : (sw = 0.0, lw = 350.0)
end

# ---------------------------------------------------------------------------
# Horizontal means, z_i
# ---------------------------------------------------------------------------

# the plane means per level: Components.plane_mean (one definition)
const plane_mean = Components.plane_mean

"""
    mean_heat_flux(sim, κ) -> (zf, flux)

The plane-mean total kinematic heat flux (K m s⁻¹) of any core `sim` at
its interior w-faces (heights `zf` above the bottom): resolved
``⟨w'θ'⟩`` (θ averaged to the face) plus the SGS ``−⟨κ_v ∂θ/∂z⟩`` with the
cell-centred diffusivity `κ` (the centre spacing of a stretched grid in
the gradient) — the profile whose minimum is the `:min_flux` z_i
(Sullivan et al. 1998). Works on a `RegularGrid` and a `StretchedGrid`.
"""
function mean_heat_flux(sim, κ)
    m = sim.mesh
    Hx, Hy, Hz, Nx, Ny, Nz = m.Hx, m.Hy, m.Hz, m.Nx, m.Ny, m.Nz
    θd, wd, κd = sim.θ.data, sim.w.data, κ.data
    zfk(K) = hasproperty(m, :zf) ? m.zf[K] - m.z0 : (K - Hz - 1) * m.dz
    dzfk(K) = hasproperty(m, :dzf) ? m.dzf[K] : m.dz
    zf = Float64[zfk(Hz + k) for k in 2:Nz]
    flux = zeros(Nz - 1)
    n = Nx * Ny
    @inbounds for k in 2:Nz
        K = Hz + k
        sw = 0.0; sθ = 0.0; swθ = 0.0; sg = 0.0
        idz = 1 / dzfk(K)
        for j in Hy+1:Hy+Ny, i in Hx+1:Hx+Nx
            θf = 0.5 * (θd[i, j, K-1] + θd[i, j, K])
            w = wd[i, j, K]
            sw += w; sθ += θf; swθ += w * θf
            sg -= 0.5 * (κd[i, j, K-1] + κd[i, j, K]) * (θd[i, j, K] - θd[i, j, K-1]) * idz
        end
        flux[k-1] = swθ / n - (sw / n) * (sθ / n) + sg / n
    end
    return zf, flux
end

"The plane-mean total heat flux of the parent ([`mean_heat_flux`](@ref) with the closure's vertical κ)."
parent_heat_flux(p::ParentCase) =
    mean_heat_flux(p.sim, p.closure === nothing ? p.sim.κt : sgs_fields(p.closure, p.sim).κv)

"""
    parent_zi(p; method = :min_flux) -> m

The boundary-layer height of the parent now: `:min_flux` (the height of
the minimum of [`parent_heat_flux`](@ref), the CBL definition) or
`:parcel` (on the plane-mean θ_v), both `boundary_layer_height`.
"""
function parent_zi(p::ParentCase; method::Symbol = :min_flux)
    if method === :min_flux
        zf, fl = parent_heat_flux(p)
        return boundary_layer_height(zf, fl; method = :min_flux)
    end
    method === :parcel || throw(ArgumentError("parent_zi: method is :min_flux or :parcel, got :$method"))
    return Components.parcel_zi(p.sim)
end

# ---------------------------------------------------------------------------
# The dynamics: hooks, step, booking
# ---------------------------------------------------------------------------

# the per-level sums in the canonical, thread-invariant order (KA-b, #296)
_level_sums!(out, f) = plane_partials!(out, identity, Aeolus.Grids.interior(f.mesh), f.data)

# The stage weights of the parent's time step: the net effect of a
# tendency evaluated at each stage is Σ w_s Δt T_s — one stage of weight 1
# for the forward-Euler projection, (1/6, 1/6, 2/3) for SSP-RK3
# (u³ = u⁰ + Δt(L⁰/6 + L¹/6 + 2L²/3) for the Shu–Osher form of step_rk3!).
const _RK3_WEIGHTS = (1 / 6, 1 / 6, 2 / 3)
_stage_weight(p::ParentCase, s::Int) = p.pc.stepper === :rk3 ? _RK3_WEIGHTS[s] : 1.0

const _BOOKED = (:θnudging, :θsubsidence, :θdamping, :qnudging, :qsubsidence)

# Stage 7 of the parent's LargeScale component (its forcings, `p.forcing`,
# applied in its order on its `clock = :core`: what `run_stage!` of the
# component does) with each θ (and qv) forcing's per-level tendency sum
# recorded between them and accumulated with the stage weight, for the ledgers
function _parent_body_force!(p::ParentCase, sim, Δt)
    F, L = p.forcing, p.lev
    stage = (p.stage[] += 1)
    w = _stage_weight(p, stage)
    if stage == 1
        for n in _BOOKED
            fill!(L[Symbol(:acc_, n)], 0.0)
        end
        p.top_acc[] = 0.0
    end
    t = sim.t
    F.geostrophic === nothing || apply_forcing!(sim, F.geostrophic, t, Δt)
    moist = p.pc.moisture
    for (name, f) in ((:nudging, F.nudging), (:subsidence, F.subsidence), (:damping, F.damping))
        f === nothing && continue
        _level_sums!(L[:a], sim.tθ)
        moist && name !== :damping && _level_sums!(L[:b], sim.tqv)
        apply_forcing!(sim, f, t, Δt)
        Lθ = L[Symbol(:θ, name)]
        _level_sums!(Lθ, sim.tθ); Lθ .-= L[:a]
        L[Symbol(:acc_θ, name)] .+= w .* Lθ
        if moist && name !== :damping
            Lq = L[Symbol(:q, name)]
            _level_sums!(Lq, sim.tqv); Lq .-= L[:b]
            L[Symbol(:acc_q, name)] .+= w .* Lq
        end
    end
    if p.top !== nothing
        capture_top_flux!(p.top, sim, p.closure)
        p.top_acc[] += w * p.top.rate
    end
    return sim
end

# book what the accepted (sub)step of length h added through the forcings
function _book_forcings!(p::ParentCase, h::Real)
    F, L = p.forcing, p.lev
    for name in (:nudging, :subsidence, :damping)
        getproperty(F, name) === nothing && continue
        book!(p.heat, name, p.wθ .* L[Symbol(:acc_θ, name)] .* h)
        if p.pc.moisture && name !== :damping
            book!(p.water, name, p.wq .* L[Symbol(:acc_q, name)] .* h)
        end
    end
    if p.top !== nothing
        J = p.top_acc[] * h
        p.top.total += J
        book!(p.heat, :top_flux, J; k = p.mesh.Nz)
    end
    return p
end

"""
    parent_dt(p) -> dt

The length of the parent's next step: `adaptive_dt` with the child's
constraints and the SGS closure's diffusive limit, shortened to land on
the next output time.
"""
function parent_dt(p::ParentCase)
    sim, pc = p.sim, p.pc
    t = sim.t
    d = cfl_diagnostics(sim, p.constraints; sgs = p.closure)
    dt = adaptive_dt(sim, p.constraints; dt_prev = p.dt_prev, diagnostics = d)
    tn = (floor(t / pc.output_period + 1e-9) + 1) * pc.output_period
    return min(dt, tn - t)
end

"""
    step_parent!(p) -> dt
    step_parent!(p, dt) -> dt

One step of the parent: the step length from [`parent_dt`](@ref) (or the
given `dt`: the nest's window, #265), the surface fluxes over it, the
radiation (every `radiation_period`), then the dynamics with
`step_adaptive!` (in stable pieces if `dt` is longer than the parent's
own limit, retried if it must), with every change of the heat and the
water of the air attributed in the two ledgers.

`body_force!(sim, Δt, w)` (optional) runs at every dynamics stage after
the parent's own hooks, with `w` the stage's weight in the step (1 for
the forward-Euler projection, 1/6, 1/6, 2/3 for SSP-RK3): the two-way
Berlin nest books the parent's interface fluxes with it (#267).
"""
step_parent!(p::ParentCase; kw...) = step_parent!(p, parent_dt(p); kw...)

function step_parent!(p::ParentCase, dt::Real; body_force! = nothing)
    sim, pc = p.sim, p.pc
    t = sim.t
    hour = hour_of_day(pc.child, t)
    tw = time_ns()
    dt = Float64(dt)
    θ, q = sim.θ, (pc.moisture ? sim.qv : sim.θ)
    budget_begin_step!(p.heat, θ, hour)
    pc.moisture && budget_begin_step!(p.water, q, hour)

    # stage 3 in the pipeline's order: the surface, the TKE closure's u*
    # (the SGS component), then the radiation (which sees the surface it
    # has just stepped); each entry that changes the air in its ledger context
    d = p.domain
    t1 = time_ns()
    budget_open!(p.heat, :surface_flux, pc.surface, θ)
    pc.moisture && budget_open!(p.water, :surface_flux, pc.surface, q)
    Components.run_stage!(Components.slot_component(d, :surface), d, Val(:surface_flux), dt)
    budget_close!(p.heat, θ)
    pc.moisture && budget_close!(p.water, q)
    sgs = Components.slot_component(d, :sgs)
    sgs === nothing || Components.run_stage!(sgs, d, Val(:surface_flux), dt)
    p.timing[:surface] = get(p.timing, :surface, 0.0) + (time_ns() - t1) * 1e-9
    rad = Components.slot_component(d, :radiation)
    if rad !== nothing
        t1 = time_ns()
        budget_open!(p.heat, :column_tendency, :column_radiation, θ)
        Components.run_stage!(rad, d, Val(:surface_flux), dt)
        budget_close!(p.heat, θ)
        p.timing[:radiation] = get(p.timing, :radiation, 0.0) + (time_ns() - t1) * 1e-9
    end

    # dynamics
    t1 = time_ns()
    budget_open!(p.heat, :dynamics, :core, θ; split = true)
    pc.moisture && budget_open!(p.water, :dynamics, :core, q; split = true)
    bf = body_force! === nothing ? (s, Δt) -> _parent_body_force!(p, s, Δt) :
         (s, Δt) -> (_parent_body_force!(p, s, Δt);
                     body_force!(s, Δt, _stage_weight(p, p.stage[])); s)
    pp = Components.core_hooks(d).post_predictor!       # the tiles' drag (stage 8)
    # every attempt (a retry too) starts its stage count afresh
    stepper = pc.stepper === :rk3 ?
        ((s, h; kw...) -> (p.stage[] = 0; step_rk3!(s, h; kw..., sgs = p.closure))) :
        ((s, h; kw...) -> (p.stage[] = 0; step!(s, h; kw..., sgs = p.closure)))
    # pieces of at most the parent's own limit: a step of parent_dt is one
    # piece; a longer one (a nest window over fixed child steps, #265) is
    # taken in stable pieces
    remaining, nret, piece_min, npieces, prev = Float64(dt), 0, Inf, 0, p.dt_prev
    while remaining > 1e-12 * dt
        lim = adaptive_dt(sim, p.constraints; dt_prev = prev, sgs = p.closure)
        h = remaining <= lim * (1 + 1e-12) ? remaining : lim
        _, used, r = step_adaptive!(sim, p.constraints; dt = h, body_force! = bf,
                                    post_predictor! = pp, sgs = p.closure, stepper = stepper)
        _book_forcings!(p, used)
        remaining -= used; nret += r; npieces += 1
        piece_min = min(piece_min, used); prev = used
    end
    budget_close!(p.heat, θ)
    pc.moisture && budget_close!(p.water, q)
    p.timing[:dynamics] = get(p.timing, :dynamics, 0.0) + (time_ns() - t1) * 1e-9

    budget_end_step!(p.heat, θ, dt)
    pc.moisture && budget_end_step!(p.water, q, dt)
    p.dt_prev = nret == 0 && npieces == 1 ? dt : piece_min
    p.retries += nret
    p.nsteps += 1
    push!(p.dt_hist, dt)
    p.timing[:total] = get(p.timing, :total, 0.0) + (time_ns() - tw) * 1e-9
    # stage 11: the nudging height follows the boundary layer (every 10
    # steps; cheap: one profile)
    ls = _component(d, Components.LargeScale)
    (ls === nothing || ls.zi === nothing) || Components.run_stage!(ls, d, Val(:column_tendency), dt)
    return dt
end

"""
    parent_record!(p) -> NamedTuple

Record the diagnostics of the current state in `p.diag`: time (s, local
hour), z_i (`:min_flux` and `:parcel`), the plane-mean θ, qv, u, v
profiles, the mean surface fluxes since the last record, the ledgers'
cumulative terms (J m⁻², kg m⁻²), the air heat storage and its closure,
dt statistics and the wall time.
"""
function parent_record!(p::ParentCase)
    sim = p.sim
    hb = stage_budget(p.heat)
    wb = p.pc.moisture ? stage_budget(p.water) : nothing
    r = (t = sim.t, hour = hour_of_day(p.pc.child, sim.t),
         zi = parent_zi(p), zi_parcel = parent_zi(p; method = :parcel),
         θ = plane_mean(sim.θ), qv = p.pc.moisture ? plane_mean(sim.qv) : Float64[],
         u = plane_mean(sim.u), v = plane_mean(sim.v),
         sfc = copy(p.sfc_mean), heat = hb, water = wb,
         nsteps = p.nsteps, retries = p.retries, wall = get(p.timing, :total, 0.0),
         sw_down = parent_downwelling(p).sw, lw_down = parent_downwelling(p).lw)
    push!(p.diag, r)
    return r
end

"""
    run_parent!(p; hours = p.pc.hours, record = true) -> p

Run the parent for `hours` of simulated time, recording
[`parent_record!`](@ref) at the start and every `output_period`.
"""
function run_parent!(p::ParentCase; hours::Real = p.pc.hours, record::Bool = true)
    pc = p.pc
    t_end = p.sim.t + 3600hours
    record && isempty(p.diag) && parent_record!(p)
    while p.sim.t < t_end - 1e-6
        step_parent!(p)
        all(isfinite, interior_view(p.sim.θ)) || error("run_parent!: non-finite θ at t = $(p.sim.t) s")
        if record && abs(p.sim.t / pc.output_period - round(p.sim.t / pc.output_period)) < 1e-6
            r = parent_record!(p)
            @info("parent at $(round(r.t / 3600; digits = 2)) h", subsystem = :urban,
                  domain = current_domain(), event = :parent_record, t = r.t,
                  hour = round(r.hour; digits = 2), zi = round(r.zi; digits = 1),
                  zi_parcel = round(r.zi_parcel; digits = 1), theta1 = round(r.θ[1]; digits = 3),
                  H = length(p.diag) > 1 ? round((r.sfc[:H] - p.diag[end-1].sfc[:H]) / pc.output_period;
                                                 sigdigits = 4) : 0.0,
                  steps = r.nsteps, wall = round(r.wall; digits = 1))
        end
    end
    return p
end
