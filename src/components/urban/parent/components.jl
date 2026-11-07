# The urban nest parent's physics as components of a domain (NEST-p1, #481;
# NEST-p3, #483: the parent is a `[[domain]]` of the model's config).
#
# The deep boundary-layer domain of #259 is an ordinary `Components.Domain`
# whose components its `[[domain]]` block enables, each built from its
# section and the domain (`Components.build`) and run by the stage pipeline:
#
#   TiledUrbanSurface           the `:surface` slot: the tiled urban mosaic of the
#                          coarse columns (surface.jl, #249), whose land use
#                          is the tile form of the parent's static driver
#                          (`STATIC_TILES_SCHEMA`); stage 3 delivers the
#                          tiles' heat and vapour into the canopy levels,
#                          stage 8 their momentum flux;
#   PrescribedSurfaceFlux  the `:surface` slot: prescribed kinematic heat and
#                          vapour fluxes Q(t), E(t) into the first level (the
#                          idealised parents);
#   MeanColumnRadiation    the `:radiation` slot: `ColumnRadiation` on the
#                          plane-mean column every `period`, its heating
#                          every step, its downwelling flux to the surface;
#   Components.LargeScale  the geostrophic pressure gradient, the nudging with
#                          its boundary-layer-dependent weight
#                          (`BoundaryLayerTracker`), the subsidence and the
#                          θ reference damping;
#   Components.SGS         `[sgs] model = "tke"` with the exact wall distance
#                          of the stretched grid and the surface value of the
#                          tiles' u*.
#
# Each works on both dynamical cores; the domain's budgets attribute what
# each changes (an urban site nested in it books the nest exchange).
# The physics is in the modules (SurfaceTiles, ColumnRadiation,
# LargeScaleForcing, PrognosticSGS); these are the adapters.

# Urban has a `build`, `stages` and `run_stage!` of its own (the urban
# components' interface): the component interface's are qualified here.

# ---------------------------------------------------------------------------
# the reference profiles of the air: the parent's hydrostatic state on the
# Boussinesq core, the base state on the anelastic one
# ---------------------------------------------------------------------------

"""
    air_reference(opts, domain) -> reference

The reference profiles (`p̄`, `ρ̄`, `π̄` per level and `cp`) a surface or
radiation component converts its fluxes with: `opts.reference` (a
[`CaseReference`](@ref)), else the domain's
(`Components.thermodynamic_reference`: `domain.drivers.reference`, the
hydrostatic state of a Boussinesq domain's initial θ profile,
[`domain_reference`](@ref); or on the anelastic core its base state
`sim.ref`). The absolute-temperature outputs (#536) read the same one.
"""
function air_reference(opts, domain)
    r = Components.option(opts, :reference, nothing)
    r !== nothing && return r
    r = Components.thermodynamic_reference(domain)
    r === nothing &&
        throw(ArgumentError("the Boussinesq core has no thermodynamic reference: give `reference` (a CaseReference)"))
    return r
end

_moist(sim) = hasproperty(sim, :has_moisture) && sim.has_moisture

# interior face heights (m, length Nz + 1)
_face_heights(m) = hasproperty(m, :zf) ? Float64[m.zf[m.Hz+k] for k in 1:m.Nz+1] :
                                         Float64[m.z0 + (k - 1) * m.dz for k in 1:m.Nz+1]

# ---------------------------------------------------------------------------
# TiledUrbanSurface
# ---------------------------------------------------------------------------

"""
    TiledUrbanSurface

The `:surface` slot's tiled urban mosaic (NEST-p1, #481; the parent
surface of #249): every column of the domain is a [`ParentColumn`](@ref)
of the [`ParentSurface`](@ref) `surface`, one tile per land-use class with
its area fraction, the unresolved buildings' Macdonald roughness and
displacement. Its land use is the tile form of the domain's static driver
(`InputDrivers.read_static_tiles`, `STATIC_TILES_SCHEMA`); the tiles are
those of a class-to-surface factory ([`class_surface`](@ref)).

Stage 3 (`:surface_flux`) delivers over the step `[t, t + Δt]` the sources
held from the last surface step, and every `period` seconds steps every
tile under the air of its column ([`step_parent_column!`](@ref): ``T = θ
Π̄_k``, ``q_v``, the cell-centre wind, ``p̄_k`` at each tile's forcing
level; the downwelling radiation of the domain's `:radiation` component)
and turns the tiles' fluxes into the sources of the canopy levels below
each tile's forcing level ``k_i``, uniformly per volume over their depth
``Z_i``: ``\\dot θ_k = f_i H_i/(Z_i ρ̄_k c_p Π̄_k)``, ``\\dot q_k = f_i
E_i/(Z_i ρ̄_k)`` and a deceleration ``f_i τ_i/(Z_i ρ̄_k)``. Stage 8
(`:post_predictor`) applies that deceleration implicitly along the
instantaneous wind, ``u ← u/(1 + Δt a/U)`` with ``U ≥ U_{min}``, at the
u and v faces of the canopy levels (``a`` the mean of the two columns a
face separates; every factor from the pre-drag velocities; no drag on an
open boundary face, one-sided at a no-flux one; #538). The
friction velocity ``u_* = \\sqrt{Σ f_i τ_i/ρ̄_1}`` is the surface value of
a TKE closure ([`surface_friction_velocity`](@ref)).

Fields: `surface`, `ref` ([`air_reference`](@ref)), `period` (s), `U_min`
(m s⁻¹), the held sources `θsrc`, `qsrc`, `drag` (per column and canopy
level), `ustar`, `t_surf` (the end of the current surface interval), the
downwelling `sw_down`, `lw_down` (W m⁻²; set by the radiation, the
options' values until then), `sfc_mean` (domain-mean J m⁻² of `Rn`, `H`,
`LE`, `G`, `E` and what the air received, `delivered_H`, `delivered_E`),
and the level weights `wθ`, `wq` of the delivered amounts, `temps` the
cache of [`tile_temperatures`](@ref), `drag_den` the drag's scratch (the
u and v face denominators of one call).

Its outputs (#536, [`tile_temperatures`](@ref)): `surface_temperature`
(the area-weighted skin temperature of each column, K),
`surface_temperature_<class>` (street, roof, park, water, wall; NaN where
the column has no such tile), `surface_theta` (the potential skin
temperature, ``T_s/Π̄``), `theta_2m` and `temperature_2m` (the screen
level, area-weighted over the tiles).
"""
mutable struct TiledUrbanSurface{R} <: AbstractComponent
    surface::ParentSurface
    ref::R
    period::Float64
    U_min::Float64
    θsrc::Array{Float64,3}
    qsrc::Array{Float64,3}
    drag::Array{Float64,3}
    ustar::Matrix{Float64}
    t_surf::Base.RefValue{Float64}
    sw_down::Base.RefValue{Float64}
    lw_down::Base.RefValue{Float64}
    sfc_mean::Dict{Symbol,Float64}
    wθ::Vector{Float64}
    wq::Vector{Float64}
    temps::Base.RefValue{Any}        # (time, TileTemperatures) of the outputs of one record (#536)
    drag_den::Array{Float64,4}       # the drag's u, v face denominators of one call (scratch, #538)
end
# the scratch of the drag (#538) sized from `drag`
TiledUrbanSurface(ps, ref::R, period, U_min, θsrc, qsrc, drag, ustar, t_surf, sw_down, lw_down, sfc_mean, wθ, wq,
                  temps) where {R} =
    TiledUrbanSurface{R}(ps, ref, period, U_min, θsrc, qsrc, drag, ustar, t_surf, sw_down, lw_down, sfc_mean, wθ, wq,
                         temps, ones(size(drag)..., 2))

Components.component_name(::Type{<:TiledUrbanSurface}) = :tiled_surface
Components.section(::Type{<:TiledUrbanSurface}) = "surface"
Components.slot(::Type{<:TiledUrbanSurface}) = :surface
Components.schema(::Type{<:TiledUrbanSurface}) =
    Config.slot_section("surface"; key = "model") do sec
        Config.provider!(sec, "tiles"; doc = "The tiled urban mosaic of the domain's static driver (its tile form, " *
                                             "STATIC_TILES_SCHEMA), one tile per land-use class of each column.") do s
            Config.option!(s, "period", Float64; default = 60.0, range = (0, Inf), open_lower = true, units = "s",
                           doc = "step of the surface tiles")
            Config.option!(s, "u_min", Float64; default = 0.5, range = (0, Inf), units = "m s^-1",
                           doc = "wind floor of the surface layer")
            Config.option!(s, "sw_down", Float64; default = 0.0, units = "W m^-2",
                           doc = "downwelling shortwave without a :radiation component (and before its first call)")
            Config.option!(s, "lw_down", Float64; default = 350.0, units = "W m^-2",
                           doc = "downwelling longwave without a :radiation component (and before its first call)")
            for (k, (key, doc, units)) in TILE_CLASS_OPTIONS
                d = tile_class_defaults()[k]
                Config.option!(s, key, _option_spec(TiledUrbanSurface, k, d); default = _option_default(TiledUrbanSurface, k, d),
                               units = units, doc = doc)
            end
        end
    end

# the tiles' class options: TOML key, doc, units (their defaults are the urban
# site's: tile_class_defaults)
const TILE_CLASS_OPTIONS = (
    :ts0 => ("ts0", "initial skin temperature of the tiles", "K"),
    :ts_water_offset => ("ts_water_offset", "water tiles start at ts0 + ts_water_offset", "K"),
    :t_soil0 => ("t_soil0", "initial soil temperature (and the soil base's)", "K"),
    :z0m => ("z0m", "momentum roughness length per surface class (the built tiles take at least the " *
                    "Macdonald roughness of their column's buildings)", "m"),
    :z0h_ratio => ("z0h_ratio", "the scalar roughness z0h = z0h_ratio × the class's z0m", "1"),
    :soil_classes => ("soil_classes", "the classes whose tiles are soil columns (the others slabs)", ""),
    :soil_theta0 => ("soil_theta0", "initial volumetric soil water per soil class", "m3 m-3"),
    :texture => ("texture", "the soil texture (Carsel & Parrish 1988 hydraulics)", ""),
    :sand => ("sand", "sand percentage of the soil (CLM thermal constants)", "%"),
    :clay => ("clay", "clay percentage of the soil (CLM thermal constants)", "%"),
    :nlayers => ("nlayers", "soil layers (the CLM exponential grid)", ""))

"""
    tile_class_defaults() -> NamedTuple

The defaults of the tiles' class options: those of the urban site (its
`[urban] ts0`, `ts_water_offset`, `t_soil0`, the `:surface_exchange`
provider's `z0m` table and `z0h_ratio`, the `:ground` slot's soil), so a
tiled surface and an urban site with default options build the same
surface for a class.
"""
function tile_class_defaults()
    g = default_options(GroundSoil); se = default_options(SurfaceExchangeMOST)
    return (ts0 = 285.0, ts_water_offset = 5.0, t_soil0 = 288.0, z0m = se.z0m, z0h_ratio = se.z0h_ratio,
            soil_classes = g.classes, soil_theta0 = g.θ0, texture = g.texture, sand = g.sand, clay = g.clay,
            nlayers = g.nlayers)
end

"""
    TileClasses

The class-to-surface factory of a tiled surface ([`TiledUrbanSurface`](@ref),
NEST-p3 #483): the per-class optical and thermal parameters (the urban
site's, [`default_surface_params`](@ref)), the momentum roughness table
`z0m`, the ratio `z0h_ratio`, the soil (`soil`, a [`GroundSoil`](@ref)
of the soil classes, or `nothing`), the initial soil temperature `T_soil0`
and the initial skin temperatures `Ts0` (water `+ ΔT_water`). A soil
class's tile is a soil column ([`soil_surface`](@ref)), any other a slab
([`fallback_slab`](@ref)): the constructions of the urban site's surfaces.
"""
struct TileClasses
    params::SurfaceParams
    z0m::Dict{UInt8,Float64}
    z0h_ratio::Float64
    soil::Union{Nothing,GroundSoil}
    T_soil0::Float64
    Ts0::Float64
    ΔT_water::Float64
end

"The [`TileClasses`](@ref) of the resolved `[surface] model = \"tiles\"` options `opts` (TOML forms or Julia values)."
function TileClasses(opts)
    d = tile_class_defaults()
    val(k) = (v = Components.option(opts, k, getproperty(d, k));
              v isa AbstractDict && !(v isa Dict{UInt8}) ? _typed_option(getproperty(d, k), v, String(k)) : v)
    soil = ground_soil((classes = val(:soil_classes), θ0 = val(:soil_theta0), texture = val(:texture),
                        sand = val(:sand), clay = val(:clay), nlayers = val(:nlayers)))
    return TileClasses(default_surface_params(), Dict{UInt8,Float64}(val(:z0m)), Float64(val(:z0h_ratio)),
                       isempty(soil.classes) ? nothing : soil, Float64(val(:t_soil0)), Float64(val(:ts0)),
                       Float64(val(:ts_water_offset)))
end

function _typed_option(d, v, key)
    ok, x, why = option_value(d, v, key)
    ok || throw(ArgumentError("[surface] $why"))
    return x
end

function class_roughness(tc::TileClasses, site::SurfaceSite)
    z0m = site.z0m !== nothing ? site.z0m : get(tc.z0m, site.class, 0.05)
    return z0m, site.z0h !== nothing ? site.z0h : tc.z0h_ratio * z0m
end
function class_surface(tc::TileClasses, site::SurfaceSite)
    z0m, z0h = class_roughness(tc, site)
    tc.soil !== nothing && site.class in tc.soil.classes &&
        return soil_surface(tc.soil, tc.params, tc.T_soil0, z0m, z0h, site)
    return fallback_slab(tc.params, z0m, z0h, site)
end
tile_skin(tc::TileClasses, c::Integer) = tc.Ts0 + (c == SURFACE_WATER ? tc.ΔT_water : 0.0)

# The tiled surface's maps, written once into its domain's output file (a
# file with variables; the parent's surface map of plot_parent_surface.jl):
# the area fraction of each surface class, the plan area density of the
# morphometry window and the Macdonald roughness and displacement height of
# the unresolved buildings (#249).
function Components.static_outputs(c::TiledUrbanSurface)
    ps = c.surface
    names = ("street", "roof", "park", "water", "wall")
    out = OutputVariable[OutputVariable("tile_fraction_$(names[k])", "1", "area fraction of the $(names[k]) tiles", :xy,
                                        d -> ps.fractions[:, :, k]) for k in 1:min(5, size(ps.fractions, 3))]
    append!(out, OutputVariable[
        OutputVariable("tile_lambda_p", "1", "plan area density of the buildings (the morphometry window)", :xy, d -> ps.λp),
        OutputVariable("tile_z0", "m", "roughness length of the unresolved buildings (Macdonald et al. 1998)", :xy, d -> ps.z0),
        OutputVariable("tile_d", "m", "displacement height of the unresolved buildings (Macdonald et al. 1998)", :xy, d -> ps.d)])
    return out
end

const _SFC_KEYS = (:Rn, :H, :LE, :G, :E, :delivered_H, :delivered_E)

# ---------------------------------------------------------------------------
# the surface and screen-level temperatures of the tiles (#536)
# ---------------------------------------------------------------------------

"The surface classes with a temperature map of their own (`surface_temperature_<name>`)."
const TILE_TEMPERATURE_CLASSES = SURFACE_CLASSES

"The screen level (m above a tile's surface) of `theta_2m`, `temperature_2m`."
const SCREEN_HEIGHT = 2.0

"""
    TileTemperatures

The temperature maps of a [`TiledUrbanSurface`](@ref) at one time
([`tile_temperatures`](@ref)), each `(Nx, Ny)` (K; NaN in a column
without tiles): `Ts` the area-weighted skin temperature, `θs` the
area-weighted potential skin temperature, `θ2`, `T2` the area-weighted
2 m potential and absolute air temperature, and `Ts_class[:, :, c]` the
skin temperature of class `c`'s tile (NaN where the column has none).
"""
struct TileTemperatures
    Ts::Matrix{Float64}
    θs::Matrix{Float64}
    θ2::Matrix{Float64}
    T2::Matrix{Float64}
    Ts_class::Array{Float64,3}
end

"""
    screen_theta(sf, t, f, θ_column, z_column, z_s, Π_s; z = 2.0, U_min = 0.5) -> θ(z)

The potential temperature (referred to `p_ref`, the model's θ) at height
`z` above the surface of tile `t` under the forcing `f` of its level,
following PALM's `calc_2m_temperature`:

  * when the forcing level is above `z` (`f.z_ref > z`), Monin–Obukhov
    similarity from the tile's current surface layer
    (`SurfaceExchange.surface_layer_solution`, the solve of its fluxes,
    with the stability functions `sf`):
    ``θ(z) = θ_s + θ_*/κ [\\ln(z/z_{0h}) - ψ_h(z/L) + ψ_h(z_{0h}/L)]``
    (`similarity_theta`). The solve's θ scale is the tile's
    surface-referenced one (``θ_s = T_s``), divided here by the Exner
    value `Π_s` at the tile's surface;
  * otherwise linear interpolation of the column's θ (`θ_column` at the
    level centres `z_column`, m above the ground) at ``z_s + z``, with the
    tile's potential skin temperature ``T_s/Π_s`` at its surface `z_s`
    (held above the top level).

`z_s` is the tile's surface height: its displacement height (0 for park
and water, the column's ``d`` for the built tiles, whose surface layer is
measured from it).
"""
function screen_theta(sf, t, f::SurfaceForcing, θcol, zcol, zs::Real, Πs::Real; z::Real = SCREEN_HEIGHT,
                      U_min::Real = 0.5)
    Ts = surface_temperature(t)
    if f.z_ref > z
        sol = surface_layer_solution(t, f; sf = sf, U_min = U_min)
        return similarity_theta(sf, z; θ_s = Ts, θ_star = sol.θ_star, L = sol.L, z0h = roughness_lengths(t).z0h) / Πs
    end
    zq = zs + z
    zl, θl = zs, Ts / Πs                       # the last point at or below zq
    for k in eachindex(zcol)
        zcol[k] <= zs && continue
        zcol[k] >= zq && return θl + (θcol[k] - θl) * (zq - zl) / (zcol[k] - zl)
        zl, θl = zcol[k], θcol[k]
    end
    return θl
end

"""
    tile_temperatures(c::TiledUrbanSurface, d) -> TileTemperatures

The temperature maps of the tiled surface `c` of domain `d` in the
current state ([`TileTemperatures`](@ref)), computed once per model time
(the outputs of one record share them). Every tile ``i`` of a column,
with area fraction ``f_i``, skin temperature ``T_{s,i}`` and surface
height ``z_{s,i}`` (its displacement height), gives

  * ``θ_{s,i} = T_{s,i}/Π̄(z_{s,i})``,
  * ``θ_{2m,i}`` ([`screen_theta`](@ref): Monin–Obukhov from the tile's
    surface layer under the air of its forcing level when that level is
    above 2 m, else interpolation) and ``T_{2m,i} = θ_{2m,i} Π̄(z_{s,i} + 2)``,

with ``Π̄`` the Exner function of the reference the surface converts its
fluxes with (`Components.reference_exner`; PALM's `exner` of the
barometric `hyp`); each map is the area-weighted ``Σ_i f_i x_i``. The
tiles of a blended mosaic are evaluated at their own levels too.
"""
function tile_temperatures(c::TiledUrbanSurface, d)
    t = Components.clock(d)
    x = c.temps[]
    (x !== nothing && x[1] == t) && return x[2]
    sim = d.state; m = sim.mesh; ps = c.surface; ref = c.ref
    zc = last(Output.cell_centres(m))
    g = hasproperty(ref, :g) ? ref.g : sim.g_accel
    Π(z) = Components.reference_exner(ref, zc, z; g = g)
    θd = sim.θ.data
    qd = _moist(sim) ? sim.qv.data : nothing
    fill_halos!(sim.u, sim.bcs); fill_halos!(sim.v, sim.bcs)
    sf = Aeolus.SurfaceExchange.default_stability_functions()   # the tiles' own (`turbulent_fluxes` without `sf`)
    nc = length(TILE_TEMPERATURE_CLASSES)
    nan() = fill(NaN, m.Nx, m.Ny)
    out = TileTemperatures(nan(), nan(), nan(), nan(), fill(NaN, m.Nx, m.Ny, nc))
    Threads.@threads for J in 1:m.Ny
        θcol = Vector{Float64}(undef, m.Nz)
        for I in 1:m.Nx
            col = ps.columns[I, J]
            col === nothing && continue
            for k in 1:m.Nz
                θcol[k] = θd[m.Hx+I, m.Hy+J, m.Hz+k]
            end
            air = _column_air(c, sim, θd, qd, I, J, maximum(col.ktile))
            fs = parent_tile_forcings(ps, col, air)
            tiles = _mosaic(col).tiles
            fr = mosaic_fractions(col.surface)
            Ts = θs = θ2 = T2 = 0.0
            for n in eachindex(tiles)
                tl = tiles[n]; zs = col.d[n]
                Πs = Π(zs)
                Tsn = surface_temperature(tl)
                θ2n = screen_theta(sf, tl, fs[n], θcol, zc, zs, Πs; U_min = c.U_min)
                Ts += fr[n] * Tsn; θs += fr[n] * Tsn / Πs
                θ2 += fr[n] * θ2n; T2 += fr[n] * θ2n * Π(zs + SCREEN_HEIGHT)
                cl = col.classes[n]
                1 <= cl <= nc && fr[n] > 0 && (out.Ts_class[I, J, cl] = Tsn)
            end
            out.Ts[I, J] = Ts; out.θs[I, J] = θs; out.θ2[I, J] = θ2; out.T2[I, J] = T2
        end
    end
    c.temps[] = (t, out)
    return out
end

"The names of a tiled surface's outputs ([`tile_temperatures`](@ref))."
const _TILE_OUTPUTS = ("surface_temperature",
                       ("surface_temperature_" * last(p) for p in TILE_TEMPERATURE_CLASSES)...,
                       "surface_theta", "theta_2m", "temperature_2m")

Components.output_names(::Type{<:TiledUrbanSurface}, opts) = _TILE_OUTPUTS

function Components.outputs(c::TiledUrbanSurface)
    tt(d) = tile_temperatures(c, d)
    vs = OutputVariable[
        OutputVariable("surface_temperature", "K", "surface (skin) temperature, area-weighted over the tiles", :xy,
                       d -> tt(d).Ts; standard_name = "surface_temperature")]
    for (cl, name) in TILE_TEMPERATURE_CLASSES
        k = Int(cl)
        push!(vs, OutputVariable("surface_temperature_$name", "K",
                                 "skin temperature of the $name tile (NaN where the column has none)", :xy,
                                 d -> tt(d).Ts_class[:, :, k]))
    end
    append!(vs, OutputVariable[
        OutputVariable("surface_theta", "K", "potential skin temperature T_s / Exner, area-weighted over the tiles", :xy,
                       d -> tt(d).θs),
        OutputVariable("theta_2m", "K", "2 m potential temperature, area-weighted over the tiles", :xy,
                       d -> tt(d).θ2),
        OutputVariable("temperature_2m", "K", "2 m air temperature, area-weighted over the tiles", :xy,
                       d -> tt(d).T2; standard_name = "air_temperature")])
    return vs
end

"""
    build(TiledUrbanSurface, opts, domain)

The tiled surface of `domain`: its land use is the tile form of the
domain's static driver (`domain.drivers.static_tiles`, a `StaticTiles`)
and its tiles those of the [`TileClasses`](@ref) of `opts`; or the
[`ParentSurface`](@ref) `opts.surface` as built. `period`, `u_min`,
`sw_down`, `lw_down`; the reference profiles of [`air_reference`](@ref).
"""
function Components.build(::Type{<:TiledUrbanSurface}, opts, domain)
    m = Components.grid(domain)
    ps = Components.option(opts, :surface, nothing)
    if ps === nothing
        t = get(domain.drivers, :static_tiles, nothing)
        t === nothing && throw(ArgumentError("TiledUrbanSurface: the domain $(domain.id) has no static driver with tiles " *
                                             "(its static_driver in the tile form, AeolusPrep.jl: aeolusprep static)"))
        tc = TileClasses(opts)
        sd = get(domain.drivers, :soil, nothing)
        # the tiles' soil from the dynamic driver (#521): its profile, or a
        # volume's mean profile (a tile has no column of its own)
        sd === nothing || tc.soil === nothing || (tc.soil.init[] = _driver_soil_init(tc.soil, sd.data, sd.path, m))
        ps = build_parent_surface(tc, landuse_of_tiles(t); z_centres = level_centres(m))
    end
    (size(ps.columns) == (m.Nx, m.Ny)) ||
        throw(DimensionMismatch("TiledUrbanSurface: $(size(ps.columns)) columns on a $(m.Nx) × $(m.Ny) grid"))
    ref = air_reference(opts, domain)
    kmax = maximum(max(1, maximum(col.ktile) - 1) for col in ps.columns if col !== nothing)
    z = zeros(m.Nx, m.Ny, kmax)
    hm = LevelMeter(m, ref.ρ̄, ref.π̄, ref.cp)
    wm = LevelMeter(m, ref.ρ̄, ones(m.Nz), 1.0)
    return TiledUrbanSurface(ps, ref, Float64(Components.option(opts, :period, 60.0)),
                        Float64(Components.option(opts, :u_min, 0.5)), copy(z), copy(z), copy(z),
                        fill(NaN, m.Nx, m.Ny), Ref(-Inf), Ref(Float64(Components.option(opts, :sw_down, 0.0))),
                        Ref(Float64(Components.option(opts, :lw_down, 350.0))),
                        Dict{Symbol,Float64}(k => 0.0 for k in _SFC_KEYS), copy(hm.w), copy(wm.w), Ref{Any}(nothing))
end

"The land use of the tile form of a static driver (`StaticTiles`) as a [`ParentLanduse`](@ref)."
landuse_of_tiles(t::StaticTiles) =
    ParentLanduse(length(t.x), length(t.y), t.dx, t.x0, t.y0, t.refinement, Int.(t.class_count),
                  t.lambda_f, t.h_mean, t.h_max, copy(t.notes))

Components.stages(::TiledUrbanSurface) = (Stage(:surface_flux), Stage(:post_predictor))
Components.surface_friction_velocity(c::TiledUrbanSurface, d) = c.ustar

"""
    set_downwelling!(surface, sw, lw)

Give a surface component the downwelling short- and longwave (W m⁻²) of
the radiation's last call; a no-op for a surface that takes none.
"""
set_downwelling!(::Any, sw, lw) = nothing
set_downwelling!(c::TiledUrbanSurface, sw, lw) = (c.sw_down[] = sw; c.lw_down[] = lw; nothing)

"""
    surface_optics(surface, θ1, π̄1) -> (ε, α, T_rad)

The emissivity, albedo and radiative temperature (K) of a domain's surface
for the column radiation: the tile means of a [`TiledUrbanSurface`](@ref) (the
emissivity-weighted radiative temperature of all tiles), else 0.95, 0.15
and the first level's air temperature ``θ_1 Π̄_1``.
"""
surface_optics(::Any, θ1, π̄1) = (0.95, 0.15, θ1 * π̄1)
function surface_optics(c::TiledUrbanSurface, θ1, π̄1)
    cols = [col for col in c.surface.columns if col !== nothing]
    εs = [surface_emissivity(col.surface) for col in cols]
    ε = sum(εs) / length(cols)
    α = sum(surface_albedo(col.surface) for col in cols) / length(cols)
    Trad = (sum(e * surface_temperature(col.surface)^4 for (e, col) in zip(εs, cols)) / sum(εs))^(1 / 4)
    return ε, α, Trad
end

function _cell_speed(sim, i, j, k)
    m = sim.mesh
    I, J, K = m.Hx + i, m.Hy + j, m.Hz + k
    return hypot(0.5 * (sim.u.data[I, J, K] + sim.u.data[I+1, J, K]),
                 0.5 * (sim.v.data[I, J, K] + sim.v.data[I, J+1, K]))
end

# the work of one column (up to six tiles, each a surface step) in
# grid-point equivalents, for the launch's serial threshold
const TILE_COLUMN_WEIGHT = 4096

"""
    step_tiles!(c::TiledUrbanSurface, sim, period) -> c

Advance every tile of every column by `period` seconds under the air of
its column and turn the tiles' fluxes into the sources the air receives
until the next surface step (see [`TiledUrbanSurface`](@ref)).
"""
function step_tiles!(c::TiledUrbanSurface, sim, period::Real)
    m = sim.mesh
    zf = _face_heights(m)
    fill!(c.θsrc, 0.0); fill!(c.qsrc, 0.0); fill!(c.drag, 0.0)
    kmax = size(c.θsrc, 3)
    θd = sim.θ.data
    qd = _moist(sim) ? sim.qv.data : nothing
    fill_halos!(sim.u, sim.bcs); fill_halos!(sim.v, sim.bcs)
    # one work item per column (KA-m, #301): a column steps its own tiles
    # and writes only its own sources and u*; the area sums of its fluxes
    # are its partials, folded in column-major order (the same bits at
    # every thread count)
    sRn, sH, sLE, sG, sE = column_sum(Aeolus.Backends.backend(c.θsrc), _tile_column!,
                                      (1:m.Nx, 1:m.Ny), c, sim, θd, qd, zf, kmax, period;
                                      weight = TILE_COLUMN_WEIGHT)
    n = m.Nx * m.Ny
    for (key, v) in ((:Rn, sRn), (:H, sH), (:LE, sLE), (:G, sG), (:E, sE))
        c.sfc_mean[key] += v / n * period
    end
    return c
end

# the air of column (I, J) at its levels 1:kf, as the tiles see it: T = θ Π̄_k,
# q_v, the cell-centre wind (at least U_min), p̄_k
function _column_air(c::TiledUrbanSurface, sim, θd, qd, I, J, kf)
    m, ref = sim.mesh, c.ref
    Hx, Hy, Hz = m.Hx, m.Hy, m.Hz
    return [SurfaceForcing(; sw_down = c.sw_down[], lw_down = c.lw_down[],
                           T_air = θd[Hx+I, Hy+J, Hz+k] * ref.π̄[k],
                           q_air = qd === nothing ? 0.0 : max(qd[Hx+I, Hy+J, Hz+k], 0.0),
                           U = max(_cell_speed(sim, I, J, k), c.U_min), p = ref.p̄[k], z_ref = 10.0)
            for k in 1:kf]
end

function _tile_column!(I, J, c, sim, θd, qd, zf, kmax, period)
    ref, ps = c.ref, c.surface
    col = ps.columns[I, J]
    kf = maximum(col.ktile)
    U_min = c.U_min
    air = _column_air(c, sim, θd, qd, I, J, kf)
    agg = step_parent_column!(ps, col, air, period; U_min = U_min)
    tf = last_tile_fluxes(_mosaic(col))
    fr = mosaic_fractions(col.surface)
    τsum = 0.0
    for (n, f) in enumerate(tf)
        nl = max(1, col.ktile[n] - 1)
        Z = zf[nl+1]
        for k in 1:min(nl, kmax)
            c.θsrc[I, J, k] += fr[n] * f.H / (Z * ref.ρ̄[k] * ref.cp * ref.π̄[k])
            c.qsrc[I, J, k] += fr[n] * f.E / (Z * ref.ρ̄[k])
            c.drag[I, J, k] += fr[n] * f.τ / (Z * ref.ρ̄[k])
        end
        τsum += fr[n] * f.τ
    end
    c.ustar[I, J] = sqrt(max(τsum, 0.0) / ref.ρ̄[1])
    return (agg.Rn, agg.H, agg.LE, agg.G, agg.E)
end

# add θ, qv sources (per s) over τ seconds into the canopy levels
function _apply_sources!(c::TiledUrbanSurface, sim, τ::Real)
    m = sim.mesh
    Hx, Hy, Hz = m.Hx, m.Hy, m.Hz
    θd = sim.θ.data
    qd = _moist(sim) ? sim.qv.data : nothing
    @inbounds for k in axes(c.θsrc, 3)
        sθ = 0.0; sq = 0.0
        for j in 1:m.Ny, i in 1:m.Nx
            θd[Hx+i, Hy+j, Hz+k] += c.θsrc[i, j, k] * τ
            qd === nothing || (qd[Hx+i, Hy+j, Hz+k] += c.qsrc[i, j, k] * τ)
            sθ += c.θsrc[i, j, k]; sq += c.qsrc[i, j, k]
        end
        # what the sources delivered, by the level weights (J m⁻², kg m⁻²)
        c.sfc_mean[:delivered_H] += c.wθ[k] * sθ * τ
        c.sfc_mean[:delivered_E] += c.wq[k] * sq * τ
    end
    return c
end

# stage 3: the held sources up to the end of the current surface interval,
# then a surface step and its sources for the rest of the step (the air
# receives exactly what the tiles gave away over the time it covers)
function Components.run_stage!(c::TiledUrbanSurface, d, ::Val{:surface_flux}, dt)
    sim = d.state
    P = c.period
    s = Float64(sim.t)
    te = s + dt
    while s < te - 1e-9 * dt
        if s >= c.t_surf[] - 1e-9 * P
            step_tiles!(c, sim, P)
            c.t_surf[] = s + P
        end
        τ = min(te, c.t_surf[]) - s
        _apply_sources!(c, sim, τ)
        s += τ
    end
    return nothing
end

# stage 8: the tiles' momentum flux as an implicit deceleration of the
# canopy levels, along the instantaneous wind: the stress magnitude held
# from the surface step, stable for any Δt (#538). Two launches over the
# canopy points (Jacobi): the denominators of the u and v faces from the
# pre-drag velocities, then u, v divided by them, each point writing only
# its own faces, so the result does not depend on the order the points run
# in or the thread count. A face's drag is the mean of the two columns it
# separates; at a non-periodic west / south face there is no column
# beyond: a face whose normal velocity the boundary fixes (an open face,
# `pinned_face_indices`) gets no drag, any other (no-flux) the drag of the
# column it bounds (`neighbor_index`). The PALM canopy model discretises
# its drag the same way (`plant_canopy_model_mod.f90`, `pcm_tendency`:
# the speed at a u face with v averaged from its four neighbours; the u
# loop from `nxlu`, which skips the boundary face of a non-cyclic inflow or
# radiation boundary, `init_pegrid.f90`).
function Components.run_stage!(c::TiledUrbanSurface, d, ::Val{:post_predictor}, Δt)
    sim = d.state
    m, bcs = sim.mesh, sim.bcs
    Nx, Ny = m.Nx, m.Ny
    a, den = c.drag, c.drag_den
    kc = size(a, 3)
    fill_halos!((sim.u, sim.v), bcs)
    # the column standing for i - 1 at i = 1 (j - 1 at j = 1): the opposite
    # edge on a periodic axis, 0 (no drag) at a pinned boundary face, the
    # boundary column itself otherwise
    pu = pinned_face_indices(Aeolus.Fields.location(sim.u), m, bcs)[1] != 0
    pv = pinned_face_indices(Aeolus.Fields.location(sim.v), m, bcs)[2] != 0
    iw = neighbor_index(m, bcs, 0, 1; outside = pu ? :none : :mirror)
    js = neighbor_index(m, bcs, 0, 2; outside = pv ? :none : :mirror)
    b = Aeolus.Backends.backend(a)
    r = (1:Nx, 1:Ny, 1:kc)
    Aeolus.Backends.launch!(b, _drag_denominators!, r, den, a, sim.u.data, sim.v.data, Float64(Δt), c.U_min,
                            m.Hx, m.Hy, m.Hz, iw, js)
    Aeolus.Backends.launch!(b, _drag_apply!, r, den, sim.u.data, sim.v.data, m.Hx, m.Hy, m.Hz)
    return nothing
end

# the implicit denominators 1 + Δt a/U of the u (west) and v (south) face of
# canopy point (i, j, k), from the pre-drag velocities; 1 where the face has
# no drag (iw / js = 0: a boundary face the boundary fixes)
@inline function _drag_denominators!(i, j, k, den, a, ud, vd, Δt, U_min, Hx, Hy, Hz, iw, js)
    @inbounds begin
        I, J, K = Hx + i, Hy + j, Hz + k
        im = i == 1 ? iw : i - 1
        jm = j == 1 ? js : j - 1
        du = 1.0
        if im != 0
            au = 0.5 * (a[im, j, k] + a[i, j, k])
            vf = 0.25 * (vd[I-1, J, K] + vd[I, J, K] + vd[I-1, J+1, K] + vd[I, J+1, K])
            du = 1 + Δt * au / max(hypot(ud[I, J, K], vf), U_min)
        end
        dv = 1.0
        if jm != 0
            av = 0.5 * (a[i, jm, k] + a[i, j, k])
            uf = 0.25 * (ud[I, J-1, K] + ud[I+1, J-1, K] + ud[I, J, K] + ud[I+1, J, K])
            dv = 1 + Δt * av / max(hypot(vd[I, J, K], uf), U_min)
        end
        den[i, j, k, 1] = du
        den[i, j, k, 2] = dv
    end
    return nothing
end

@inline function _drag_apply!(i, j, k, den, ud, vd, Hx, Hy, Hz)
    @inbounds begin
        I, J, K = Hx + i, Hy + j, Hz + k
        ud[I, J, K] /= den[i, j, k, 1]
        vd[I, J, K] /= den[i, j, k, 2]
    end
    return nothing
end

"""
    TiledSurfaceState

The packed checkpoint state of a [`TiledUrbanSurface`](@ref) (#526): every
built column's surface state in one vector (`SurfaceExchange.pack_state!`,
column by column in the order of `columns`, `offsets[n]` the first entry
of the `n`-th built column), plus copies of the component's own arrays.
What is not here is set by the configuration (land use, fractions of the
column, tile constants) or is scratch (the soil solvers' workspaces), so it
restores into a surface built from the same configuration. It replaced the
generic `structural_state` of the whole component, whose walk of every
object took 7.2 s per snapshot on a 200 × 200 grid (86,229 soil tiles)
and made the blow-up guard 80 % of a run's wall time.
"""
struct TiledSurfaceState
    ncolumns::Int
    offsets::Vector{Int}
    tiles::Vector{Float64}
    θsrc::Array{Float64,3}
    qsrc::Array{Float64,3}
    drag::Array{Float64,3}
    ustar::Matrix{Float64}
    t_surf::Float64
    sw_down::Float64
    lw_down::Float64
    sfc_mean::Dict{Symbol,Float64}
    wθ::Vector{Float64}
    wq::Vector{Float64}
end

_built_columns(c::TiledUrbanSurface) = [col for col in c.surface.columns if col !== nothing]

function _column_offsets(cols)
    offsets = Vector{Int}(undef, length(cols) + 1)
    offsets[1] = 1
    for (n, col) in enumerate(cols)
        offsets[n+1] = offsets[n] + packed_state_length(col.surface)
    end
    return offsets
end

function Aeolus.IO.checkpoint_state(c::TiledUrbanSurface)
    cols = _built_columns(c)
    # a surface type without a packed form: the generic walk
    all(col -> has_packed_state(col.surface), cols) || return Aeolus.IO.structural_state(c)
    offsets = _column_offsets(cols)
    buf = Vector{Float64}(undef, offsets[end] - 1)
    Threads.@threads for n in eachindex(cols)
        pack_state!(buf, offsets[n], cols[n].surface)
    end
    return TiledSurfaceState(length(cols), offsets, buf, copy(c.θsrc), copy(c.qsrc), copy(c.drag),
                             copy(c.ustar), c.t_surf[], c.sw_down[], c.lw_down[], copy(c.sfc_mean),
                             copy(c.wθ), copy(c.wq))
end

function Aeolus.IO.restore_checkpoint_state!(c::TiledUrbanSurface, s::TiledSurfaceState)
    cols = _built_columns(c)
    length(cols) == s.ncolumns && _column_offsets(cols) == s.offsets ||
        throw(ArgumentError("TiledUrbanSurface: the checkpoint holds $(s.ncolumns) columns of " *
                            "$(length(s.tiles)) values, this surface is built differently"))
    Threads.@threads for n in eachindex(cols)
        unpack_state!(cols[n].surface, s.tiles, s.offsets[n])
    end
    copyto!(c.θsrc, s.θsrc); copyto!(c.qsrc, s.qsrc); copyto!(c.drag, s.drag); copyto!(c.ustar, s.ustar)
    c.t_surf[] = s.t_surf; c.sw_down[] = s.sw_down; c.lw_down[] = s.lw_down
    empty!(c.sfc_mean); merge!(c.sfc_mean, s.sfc_mean)
    copyto!(c.wθ, s.wθ); copyto!(c.wq, s.wq)
    return c
end

# checkpoints written before #526 (the generic walk), and surfaces without a packed form
Aeolus.IO.restore_checkpoint_state!(c::TiledUrbanSurface, s::Aeolus.IO.StructuralState) = Aeolus.IO.restore_structural_state!(c, s)

# ---------------------------------------------------------------------------
# PrescribedSurfaceFlux
# ---------------------------------------------------------------------------

"""
    PrescribedSurfaceFlux(Q, E, ref)

The `:surface` slot's prescribed kinematic fluxes (NEST-p1, #481; the
idealised parents of #259): the heat flux `Q(t)` (K m s⁻¹) and, on a
moist core, the vapour flux `E(t)` (kg kg⁻¹ m s⁻¹), functions of the
model time — a [`ConstantForcing`](@ref), a [`SeriesForcing`](@ref) or
any Julia function. In stage 3 (`:surface_flux`) their exact means over
the step (`LargeScaleForcing.forcing_mean`) enter the first level, ``Δθ_1
= \\bar Q Δt/Δz_1``; no momentum flux. `sfc_mean[:H]` counts the heat in
J m⁻² (``ρ̄_1 c_p Π̄_1`` of the reference, [`air_reference`](@ref)). It
reports no wall flux to a TKE closure (the parents' closure has taken
none from it since #259).
"""
struct PrescribedSurfaceFlux{Q,E,R} <: AbstractComponent
    Q::Q
    E::E
    ref::R
    sfc_mean::Dict{Symbol,Float64}
end

Components.component_name(::Type{<:PrescribedSurfaceFlux}) = :prescribed_surface_flux
Components.section(::Type{<:PrescribedSurfaceFlux}) = "surface"
Components.slot(::Type{<:PrescribedSurfaceFlux}) = :surface
Components.schema(::Type{<:PrescribedSurfaceFlux}) =
    Config.slot_section("surface"; key = "model") do sec
        Config.provider!(sec, "prescribed"; doc = "Prescribed kinematic heat and vapour fluxes into the first level.") do s
            Config.option!(s, "q_surface", Config.OpenTable(t -> _form_problems(_forcing_value, t, "q_surface"));
                           default = Dict{String,Any}("kind" => "constant", "value" => 0.0), units = "K m s^-1",
                           doc = "kinematic heat flux, a function of the model time t (s): {kind = \"constant\", value} or " *
                                 "{kind = \"series\", t = […], value = […]} (linear in t, constant beyond the ends)")
            Config.option!(s, "e_surface", Config.OpenTable(t -> _form_problems(_forcing_value, t, "e_surface"));
                           default = Dict{String,Any}("kind" => "constant", "value" => 0.0), units = "kg kg^-1 m s^-1",
                           doc = "kinematic vapour flux (a moist core), a function of the model time as q_surface")
        end
    end

# a function of time from its TOML table ({kind = "constant" | "series"}), or as given
_forcing_value(x, path) = surface_forcing_value(x, path)
function _surface_function(x, key)
    x isa AbstractDict || return x
    ok, f, why = _forcing_value(x, key)
    ok || throw(ArgumentError("[surface] $why"))
    return f
end

Components.build(::Type{<:PrescribedSurfaceFlux}, opts, domain) =
    PrescribedSurfaceFlux(_surface_function(opts.q_surface, "q_surface"), _surface_function(opts.e_surface, "e_surface"),
                          air_reference(opts, domain), Dict{Symbol,Float64}(k => 0.0 for k in _SFC_KEYS))
Components.enabled(::Type{<:PrescribedSurfaceFlux}, opts) = opts.model === :prescribed
Components.enabled(::Type{<:TiledUrbanSurface}, opts) = opts.model === :tiles

Components.stages(::PrescribedSurfaceFlux) = (Stage(:surface_flux),)

function Components.run_stage!(c::PrescribedSurfaceFlux, d, ::Val{:surface_flux}, dt)
    sim = d.state
    t = sim.t
    dz1 = level_thickness(sim.mesh)[1]
    moist = _moist(sim)
    Q = forcing_mean((z, s) -> c.Q(s), 0.0, t, t + dt)
    E = moist ? forcing_mean((z, s) -> c.E(s), 0.0, t, t + dt) : 0.0
    a = view(interior_view(sim.θ), :, :, 1)
    a .+= Q * dt / dz1
    moist && (view(interior_view(sim.qv), :, :, 1) .+= E * dt / dz1)
    c.sfc_mean[:H] += Q * dt * c.ref.ρ̄[1] * c.ref.cp * c.ref.π̄[1]
    return nothing
end

Aeolus.IO.checkpoint_state(c::PrescribedSurfaceFlux) = (sfc_mean = copy(c.sfc_mean),)
Aeolus.IO.restore_checkpoint_state!(c::PrescribedSurfaceFlux, s::NamedTuple) = (merge!(c.sfc_mean, s.sfc_mean); c)

# ---------------------------------------------------------------------------
# MeanColumnRadiation
# ---------------------------------------------------------------------------

"""
    SiteSun(cfg)

The sun of an urban site's clock as a function of the model time `t`:
`solar_position_at(cfg, hour_of_day(cfg, t))` (the site's date, location
and local clock).
"""
struct SiteSun
    cfg::NamedTuple
end
(s::SiteSun)(t) = solar_position_at(s.cfg, hour_of_day(s.cfg, t))

"""
    MeanColumnRadiation

The `:radiation` slot's column radiation of the plane-mean column
(NEST-p1, #481; the parent's radiation since #259): every `period`
seconds `ColumnRadiation.column_radiation!` on the domain's plane-mean
θ and q_v (`stitched_column` on the domain's levels, under a standard
atmosphere of humidity `humid`), at the sun `sun(t)` ([`SiteSun`](@ref)),
over the surface of the domain's `:surface` component
([`surface_optics`](@ref)). Its downwelling short- and longwave at the
ground force that surface until the next call ([`set_downwelling!`](@ref)),
and its heating ``\\dot T`` warms θ every step by ``\\dot T Δt / Π̄``
(`Radiation.apply_radiative_cooling!`).

It runs in stage 3 (`:surface_flux`) after the surface (stage order 1):
the surface steps under the downwelling flux of the last call, then the
radiation sees the surface it has just stepped, as the parent has done
since #259; its heating is applied there too, before the core step. The
shortwave at the ground is that of the column without the model's
condensate (`fsw_clear`).
"""
mutable struct MeanColumnRadiation{M,S} <: AbstractComponent
    model::M
    sun::S
    humid::Symbol
    period::Float64
    π̄::Vector{Float64}
    t_rad::Base.RefValue{Float64}
    sw_down::Base.RefValue{Float64}
    lw_down::Base.RefValue{Float64}
end

Components.component_name(::Type{<:MeanColumnRadiation}) = :mean_column_radiation
Components.section(::Type{<:MeanColumnRadiation}) = "radiation"
Components.slot(::Type{<:MeanColumnRadiation}) = :radiation
Components.schema(::Type{<:MeanColumnRadiation}) =
    Config.slot_section("radiation"; key = "scheme", doc = "The radiation of the domain's air and ground.") do sec
        Config.provider!(sec, "none"; default = [:model], doc = "No radiation.")
        Config.provider!(sec, "mean_column"; doc = "Two-stream radiation of the plane-mean column (ColumnRadiation).") do s
            Config.option!(s, "period", Float64; default = 600.0, range = (0, Inf), open_lower = true, units = "s",
                           doc = "period of the column radiation")
            Config.option!(s, "humid", Symbol; default = :midlatitude_summer,
                           allowed = [:midlatitude_summer, :subarctic_winter],
                           doc = "the humidity of the standard atmosphere above the domain")
            # the sun: the place and the local clock of the domain (the urban site's [urban] defaults)
            Config.option!(s, "latitude", Float64; default = 52.5220, range = (-90, 90), units = "degree",
                           doc = "latitude of the solar position")
            Config.option!(s, "longitude", Float64; default = 13.4100, range = (-180, 180), units = "degree",
                           doc = "longitude of the solar position")
            Config.option!(s, "elevation", Float64; default = 34.0, units = "m", doc = "height above sea level")
            Config.option!(s, "date", Date; default = Date(2019, 6, 29), doc = "the local date at t = 0")
            Config.option!(s, "utc_offset", Float64; default = 2.0, range = (-14, 14), units = "h",
                           doc = "local clock = UTC + utc_offset")
            Config.option!(s, "start_hour", Float64; default = 6.0, range = (0, 24), units = "h",
                           doc = "local hour at the model time 0")
        end
    end

"""
    radiation_sun(opts) -> SiteSun

The sun of the `[radiation] scheme = \"mean_column\"` options: the solar
position ([`solar_position_at`](@ref)) of their date, place and local
clock as a function of the model time.
"""
radiation_sun(opts) = SiteSun((date = (Dates.year(opts.date), Dates.month(opts.date), Dates.day(opts.date)),
                               lat0 = Float64(opts.latitude), lon0 = Float64(opts.longitude),
                               elevation = Float64(opts.elevation), utc_offset = Float64(opts.utc_offset),
                               start_h = Float64(opts.start_hour)))

Components.enabled(::Type{<:MeanColumnRadiation}, opts) = opts.scheme === :mean_column

"""
    build(MeanColumnRadiation, opts, domain)

The column radiation of `domain`: `period`, `humid`, `sun` (a function of
the model time, [`SiteSun`](@ref)), `p_surface` (Pa; default the
reference's, [`air_reference`](@ref)) and `reference`.
"""
function Components.build(::Type{<:MeanColumnRadiation}, opts, domain)
    m = Components.grid(domain)
    ref = air_reference(opts, domain)
    ps = Components.option(opts, :p_surface, nothing)
    ps === nothing && (ps = hasproperty(ref, :p_surface) ? ref.p_surface : ref.p̄[1])
    sun = Components.option(opts, :sun, nothing)
    sun === nothing && (sun = radiation_sun(opts))
    return MeanColumnRadiation(column_radiation_solver(m.Nz, ps), sun,
                               Components.option(opts, :humid, :midlatitude_summer),
                               Float64(Components.option(opts, :period, 600.0)), collect(Float64, ref.π̄),
                               Ref(0.0), Ref(0.0), Ref(350.0))
end

Components.stages(::MeanColumnRadiation) = (Stage(:surface_flux; order = 1),)

"""
    radiate!(c::MeanColumnRadiation, d) -> c

One call of the column radiation of `c` on the plane-mean column of the
domain `d`, and its downwelling flux to the domain's surface.
"""
function radiate!(c::MeanColumnRadiation, d)
    sim = d.state
    m = sim.mesh
    sp = c.sun(sim.t)
    θ = Components.plane_mean(sim.θ)
    qv = _moist(sim) ? Components.plane_mean(sim.qv) : zeros(m.Nz)
    sfc = Components.slot_component(d, :surface)
    ε, α, Trad = surface_optics(sfc, θ[1], c.π̄[1])
    dz = level_thickness(m)
    col = stitched_column(c.model, θ, qv, zeros(m.Nz), dz; T_rad = Trad, humid = c.humid)
    column_radiation!(c.model, col, sp, ε, α)
    c.lw_down[] = c.model.flw.flux_dn[1]
    c.sw_down[] = c.model.fsw_clear === nothing ? 0.0 : c.model.fsw_clear.flux_dn[1]
    set_downwelling!(sfc, c.sw_down[], c.lw_down[])
    return c
end

function Components.run_stage!(c::MeanColumnRadiation, d, ::Val{:surface_flux}, dt)
    sim = d.state
    t = sim.t
    if t >= c.t_rad[] - 1e-6
        radiate!(c, d)
        c.t_rad[] = (floor(t / c.period + 1e-9) + 1) * c.period
    end
    apply_radiative_cooling!(sim.θ, c.model.heat .* dt ./ c.π̄)
    c.model.dt_applied = dt
    return nothing
end

Aeolus.IO.checkpoint_state(c::MeanColumnRadiation) = Aeolus.IO.structural_state(c)
Aeolus.IO.restore_checkpoint_state!(c::MeanColumnRadiation, s::Aeolus.IO.StructuralState) = Aeolus.IO.restore_structural_state!(c, s)

"""
    ConstantForcing(value)

A prescribed surface forcing (`[surface] model = "prescribed"`) that does not change in time:
`f(t) = value`. The TOML form `{kind = "constant", value = …}` of
`q_surface` / `e_surface` (CFG-c, #358; NEST-p3 #483).
"""
struct ConstantForcing
    value::Float64
end
(f::ConstantForcing)(t) = f.value

"""
    SeriesForcing(t, value)

A prescribed surface forcing given at the times `t` (s, strictly
increasing): linear in time between them, constant beyond the ends. The
TOML form `{kind = "series", t = […], value = […]}` (CFG-c, #358).
"""
struct SeriesForcing
    t::Vector{Float64}
    value::Vector{Float64}
end
(f::SeriesForcing)(t) = _interp1(f.t, f.value, t)
Base.:(==)(a::SeriesForcing, b::SeriesForcing) = a.t == b.t && a.value == b.value
Base.isequal(a::SeriesForcing, b::SeriesForcing) = isequal(a.t, b.t) && isequal(a.value, b.value)
Base.hash(a::SeriesForcing, h::UInt) = hash(a.value, hash(a.t, hash(:SeriesForcing, h)))

"Linear interpolation of `(xs, ys)` at `x`, constant beyond the ends."
function _interp1(xs, ys, x)
    x <= xs[1] && return ys[1]
    x >= xs[end] && return ys[end]
    i = searchsortedlast(xs, x)
    w = (x - xs[i]) / (xs[i+1] - xs[i])
    return (1 - w) * ys[i] + w * ys[i+1]
end

"Level-centre heights (m) of the interior levels of a mesh."
level_centres(m) = hasproperty(m, :zc) ? Float64[m.zc[m.Hz+k] for k in 1:m.Nz] :
                   Float64[m.z0 + (k - 0.5) * m.dz for k in 1:m.Nz]
