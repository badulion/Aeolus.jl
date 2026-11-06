# The ground: one production Aeolus.SoilColumn `SoilThermalColumn` per
# street, building-adjacent and park column, used directly as that
# column's surface (an `AbstractSurface`: skin energy balance, heat
# conduction and freeze–thaw, Richards soil water, and a moisture
# availability β from the water of the top layer).
#
# It replaces the #126 pair "slab temperature + UrbanSoil below it through
# a contact conductance" and the example's per-class β table: the skin is
# now the soil's own massless skin, solved implicitly with the soil heat
# equation, and β follows the soil water (Lee & Pielke 1992, in
# SoilColumn). All soil physics lives in src/SoilColumn.jl; this file only
# registers the factory the surface exchange builds the columns with.
#
# KNOWN LIMITATIONS (stated, not hidden):
#   * the streets are bare loam: SoilColumn has no sealed (impervious)
#     surface, so a street evaporates from its top soil layer until that
#     layer dries (a sealed-surface option belongs in src);
#   * the park is bare soil too until #140 puts LandSurface's vegetated
#     column there;
#   * no rain reaches the soil: the microphysics books surface
#     precipitation as a domain total, not per column.

"""
    GroundSoil

Slot `:ground`, provider `:soil_column` (default). Registers a surface
factory for the classes in `classes` (default street, park and the
building-adjacent `SURFACE_WALL` cells): a `SoilThermalColumn` over a
`SoilWaterColumn` with

* the CLM 10-layer exponential grid (top node 7 mm, base 3.43 m),
  Carsel & Parrish (1988) hydraulics of `texture` and the CLM thermal
  constants for `sand`/`clay` percent;
* the initial state of `soil_init` (#521; see [`soil_initial_state`](@ref)):
  * `"uniform"`: volumetric water `θ0[class]` (default: 0.10 under streets
    and the building-adjacent cells, a dry bedding with β ≈ 0.44 at the
    start; 0.25 in the park, above field capacity, β = 1), and temperature
    `cfg.T_soil0` with the base held there (`BottomTemperature`);
  * `"driver"`: the dynamic driver's `init_soil_t` / `init_soil_m` (PALM's
    PIDS variables, AeolusPrep.jl's ERA5 soil) on the column's nodes, the
    base at the deepest level's temperature (PALM's rule); a level or a
    column without data (NaN) falls back to the profile's other levels or
    to the mean profile of the columns with data; water clipped into
    ``(θ_r, θ_s]`` of the texture;
  * `"annual_wave"`: the analytic annual wave ([`annual_soil_temperature`](@ref
    Aeolus.SoilColumn.annual_soil_temperature)) of `wave_t_mean` (default
    `T_soil0`), `wave_amplitude`, `wave_day_max` (the day of the year of
    the surface maximum) and `wave_damping_depth` (default: the soil's,
    [`soil_damping_depth`](@ref Aeolus.SoilColumn.soil_damping_depth) at
    the class's water) on the run's date, the base at the wave's value
    there; water `θ0`;
  * `"auto"` (default): `"driver"` when the domain's dynamic driver holds
    the soil, else `"uniform"` (with a warning when there is a driver
    without it);
* the skin starting at the case's surface temperature `bus.Ts`;
* albedo and emissivity of the class from `bus.params` (the radiosity
  materials), roughness `z0m` from the surface-exchange options;
* `hydrology = true`: evaporation is drawn from, and dew added to, the
  soil water, so β changes as the top layer dries or wets.

Outputs `T_soil1` (top layer, 7 mm), `G_soil` (ground heat flux of the
last surface step) and `theta_soil1` (top-layer liquid water); NaN where
the column is not soil.
"""
struct GroundSoil{H,P} <: UrbanComponent
    classes::Vector{UInt8}
    θ0::Dict{UInt8,Float64}
    hydraulics::H
    thermal::P
    nlayers::Int
    cols::Vector{CartesianIndex{2}}
    init_opts::NamedTuple
    init::Base.RefValue{Any}
    T_env::Vector{Float64}       # [min, max] of the initial soil and base temperatures (the smoke check's bounds)
end

component_name(::Type{<:GroundSoil}) = :ground_soil
slot(::Type{<:GroundSoil}) = :ground
provider(::Type{<:GroundSoil}) = :soil_column
modules_used(::Type{<:GroundSoil}) = (:SoilColumn,)
default_options(::Type{<:GroundSoil}) =
    (classes = (SURFACE_STREET, SURFACE_WALL, SURFACE_PARK),
     θ0 = Dict(SURFACE_STREET => 0.10, SURFACE_WALL => 0.10, SURFACE_PARK => 0.25),
     texture = :loam, sand = 40, clay = 20, nlayers = 10,
     soil_init = :auto, wave_t_mean = nothing, wave_amplitude = 8.0, wave_day_max = 200.0,
     wave_damping_depth = nothing)

const SOIL_INIT_KEYS = (:soil_init, :wave_t_mean, :wave_amplitude, :wave_day_max, :wave_damping_depth)

"""
The soil of the options `o` of the `:ground` slot's `soil_column` provider
(no columns yet; its initial state uniform until [`resolve_soil_init!`](@ref)).
"""
function ground_soil(o)
    d = default_options(GroundSoil)
    io = NamedTuple{SOIL_INIT_KEYS}(Tuple(get(o, k, getproperty(d, k)) for k in SOIL_INIT_KEYS))
    io.soil_init in (:auto, :uniform, :driver, :annual_wave) ||
        throw(ArgumentError("[ground] soil_init: expected \"auto\", \"uniform\", \"driver\" or \"annual_wave\", got $(repr(String(io.soil_init)))"))
    return GroundSoil(UInt8[o.classes...], Dict{UInt8,Float64}(o.θ0),
                      carsel_parrish_texture(o.texture), clm_soil_thermal_properties(o.sand, o.clay),
                      o.nlayers, CartesianIndex{2}[], io, Ref{Any}((kind = :uniform,)), [Inf, -Inf])
end

function build(::Type{<:GroundSoil}, case, o)
    c = ground_soil(o)
    resolve_soil_init!(c, case)
    for cls in c.classes
        register_surface_factory!(case, cls, :ground_soil, (cs, site) -> soil_surface(c, cs, site))
    end
    return c
end

"""
    soil_surface(c::GroundSoil, case, site::SurfaceSite) -> SoilThermalColumn
    soil_surface(c::GroundSoil, case, I::CartesianIndex{2}) -> SoilThermalColumn

The soil column of a [`SurfaceSite`](@ref) (see [`GroundSoil`](@ref));
the second form builds the site of child column `I`. A child column is
recorded in `c.cols` (its outputs and checks); a parent tile
(`site.column === nothing`) is not.
"""
function soil_surface(c::GroundSoil, case, site::SurfaceSite)
    col = soil_surface(c, case.bus.params, case.cfg.T_soil0, class_roughness(case, site)..., site)
    site.column === nothing || push!(c.cols, site.column)
    return col
end

"""
    soil_surface(c::GroundSoil, params::SurfaceParams, T0, z0m, z0h, site) -> SoilThermalColumn

The soil column of class `site.class` (the one construction; the urban
site's columns and the tiles of a tiled surface, [`TileClasses`](@ref)):
`c`'s soil, the class's albedo and emissivity of `params`, soil and base
temperature `T0` (K), the roughness lengths `z0m`, `z0h` (m).
"""
function soil_surface(c::GroundSoil, p::SurfaceParams, T0::Real, z0m::Real, z0h::Real, site::SurfaceSite)
    cls = site.class
    grid = clm_exponential_soil_grid(c.nlayers)
    ini = soil_initial_state(c, grid, cls, site.column, T0)
    c.T_env[1] = min(c.T_env[1], minimum(ini.T), ini.T_base)
    c.T_env[2] = max(c.T_env[2], maximum(ini.T), ini.T_base)
    water = SoilWaterColumn(grid, c.hydraulics; θ = ini.θ)
    return SoilThermalColumn(water, c.thermal; T_soil = ini.T, T_skin = site.T_skin,
                             bottom = Aeolus.SoilColumn.BottomTemperature(ini.T_base),
                             albedo = p.albedo[cls], emissivity = p.emissivity[cls],
                             z0m = z0m, z0h = z0h, hydrology = true)
end
# ---------------------------------------------------------------------------
# the initial state (#521)
# ---------------------------------------------------------------------------

"""
    resolve_soil_init!(c::GroundSoil, case) -> c

Resolve `c`'s `soil_init` against the urban site `case` (see
[`GroundSoil`](@ref)): read the soil of the domain's dynamic driver, or
fix the annual wave on the run's date. What [`soil_initial_state`](@ref)
then gives every column.
"""
function resolve_soil_init!(c::GroundSoil, case)
    o, cfg = c.init_opts, case.cfg
    mode = o.soil_init
    soil = nothing
    if mode in (:auto, :driver) && cfg.dynamic !== nothing
        soil = read_dynamic_driver(cfg.dynamic; mesh = case.mesh, planes = false).soil
    end
    if mode === :driver && soil === nothing
        throw(ArgumentError("[ground] soil_init = \"driver\": the domain's dynamic driver " *
                            (cfg.dynamic === nothing ? "is not set" : "$(cfg.dynamic) holds no init_soil_t / init_soil_m") *
                            " (AeolusPrep.jl writes them from ERA5)"))
    end
    if soil !== nothing
        c.init[] = _driver_soil_init(c, soil, cfg.dynamic, case.mesh)
    elseif mode === :annual_wave
        doy = Dates.dayofyear(Date(cfg.date...))
        c.init[] = (kind = :annual_wave, t = (doy - 1) * 86400.0 + cfg.start_h * 3600.0,
                    T_mean = something(o.wave_t_mean, cfg.T_soil0), amplitude = Float64(o.wave_amplitude),
                    t_max = (o.wave_day_max - 1) * 86400.0, d = o.wave_damping_depth)
        @info(@sprintf("the soil starts on the annual wave: mean %.2f K, amplitude %.2f K, surface maximum on day %.0f, day %d of the run's year",
                       c.init[].T_mean, c.init[].amplitude, o.wave_day_max, doy), subsystem = :urban, event = :soil_init)
    else
        c.init[] = (kind = :uniform,)
        msg = @sprintf("the soil starts uniform at t_soil0 = %.2f K (its slow modes relax over weeks, τ ≈ 30 d): set [ground] soil_init = \"annual_wave\", or give the dynamic driver init_soil_t / init_soil_m (#521)",
                       cfg.T_soil0)
        cfg.dynamic === nothing || mode !== :auto ? @info(msg, subsystem = :urban, event = :soil_init) :
                                                    @warn(msg, subsystem = :urban, event = :soil_init)
    end
    return c
end

# The driver's soil, checked and prepared: the water clipped into the
# texture's (θ_r, θ_s] (logged), and for a volume the mean profile of the
# columns with data, for the columns without (and the tiles).
function _driver_soil_init(c::GroundSoil, soil, path, mesh)
    h = c.hydraulics
    lo = Aeolus.SoilColumn.residual_water_content(h) + 1e-3 * (soil_porosity(h) - Aeolus.SoilColumn.residual_water_content(h))
    hi = soil_porosity(h)
    m = soil.m === nothing ? nothing : copy(soil.m)
    if m !== nothing
        nclip = count(x -> !isnan(x) && !(lo <= x <= hi), m)
        nclip > 0 && @warn(@sprintf("the driver's init_soil_m: %d of %d values outside the texture's (θ_r, θ_s] = (%.3f, %.3f], clipped",
                                    nclip, count(!isnan, m), Aeolus.SoilColumn.residual_water_content(h), hi),
                           subsystem = :urban, event = :soil_init, driver = path)
        map!(x -> isnan(x) ? x : clamp(x, lo, hi), m, m)
    end
    colmean(A) = A === nothing || ndims(A) == 1 ? A :
        [(v = filter(!isnan, vec(A[:, :, k])); isempty(v) ? NaN : mean(v)) for k in axes(A, 3)]
    for (n, A) in (("init_soil_t", soil.t), ("init_soil_m", m))
        A === nothing || ndims(A) == 1 || size(A)[1:2] == (mesh.Nx, mesh.Ny) ||
            throw(ArgumentError("$path: the soil volume $n is $(size(A)), the run's grid $((mesh.Nx, mesh.Ny))"))
    end
    tm, mm = colmean(soil.t), colmean(m)
    (tm !== nothing && all(isnan, tm)) && throw(ArgumentError("$path: init_soil_t has no data (all fill values)"))
    @info(@sprintf("the soil starts from the dynamic driver: %d levels to %.2f m%s%s", length(soil.z), soil.z[end],
                   soil.t === nothing ? "" : ndims(soil.t) == 3 ? ", temperature per column" : ", temperature profile",
                   m === nothing ? "" : ndims(m) == 3 ? ", water per column" : ", water profile"),
          subsystem = :urban, event = :soil_init, driver = path)
    return (kind = :driver, z = soil.z, t = soil.t, m = m, t_mean = tm, m_mean = mm)
end

# a column's data: the volume's column (the mean profile where it has none)
# or the profile
function _soil_column_data(A, Amean, I)
    A === nothing && return nothing
    ndims(A) == 1 && return A
    I === nothing && return Amean
    v = A[I[1], I[2], :]
    return all(isnan, v) ? Amean : v
end

"""
    soil_initial_state(c::GroundSoil, grid, class, I, T0) -> (T, θ, T_base)

The initial layer temperatures `T` (K), liquid water `θ` (m³ m⁻³, a
vector or the class's `θ0`) and base temperature `T_base` of a soil column
of `class` on `grid` at column `I` (`nothing`: a tile, the profile or the
mean profile), from `c`'s resolved `soil_init` ([`resolve_soil_init!`](@ref));
`T0` the uniform temperature (`t_soil0`).
"""
function soil_initial_state(c::GroundSoil, grid, cls, I, T0::Real)
    ini = c.init[]
    θ0 = get(c.θ0, cls, 0.2)
    if ini.kind === :driver
        SC = Aeolus.SoilColumn
        t = _soil_column_data(ini.t, ini.t_mean, I)
        m = _soil_column_data(ini.m, ini.m_mean, I)
        T = t === nothing ? nothing : SC.soil_profile_on_nodes(ini.z, t, grid.z)
        θ = m === nothing ? nothing : SC.soil_profile_on_nodes(ini.z, m, grid.z)
        Tb = T === nothing ? Float64(T0) : t[findlast(!isnan, t)]
        return (T = something(T, Float64(T0)), θ = something(θ, θ0), T_base = Tb)
    elseif ini.kind === :annual_wave
        SC = Aeolus.SoilColumn
        h = c.hydraulics
        d = ini.d !== nothing ? Float64(ini.d) :
            SC.soil_damping_depth(SC.soil_thermal_conductivity(c.thermal, soil_porosity(h), θ0, 0.0),
                                  SC.soil_heat_capacity(c.thermal, soil_porosity(h), θ0, 0.0))
        wave(z) = SC.annual_soil_temperature(z, ini.t; T_mean = ini.T_mean, amplitude = ini.amplitude,
                                             damping_depth = d, t_max = ini.t_max)
        return (T = wave.(grid.z), θ = θ0, T_base = wave(grid.zh[end]))
    end
    return (T = Float64(T0), θ = θ0, T_base = Float64(T0))
end

soil_surface(c::GroundSoil, case, I::CartesianIndex{2}) =
    soil_surface(c, case, SurfaceSite(case.geom.sfc[I]; T_skin = case.bus.Ts[I], column = I))

# The surface exchange's per-column closure check: the soil's enthalpy
# less what left through the base and what the moved water carried, so
# its change over a step is the ground heat flux G the step returned.
surface_store_energy(s::SoilThermalColumn) =
    soil_enthalpy(s) + s.budget.bottom - s.budget.water

"(Nx, Ny) map of `f(col, I)` over the soil columns, NaN elsewhere."
function soil_map(f, c::GroundSoil, case)
    out = fill(NaN, case.mesh.Nx, case.mesh.Ny)
    for I in c.cols
        s = case.bus.surface[I]
        s isa SoilThermalColumn && (out[I] = f(s, I))
    end
    return out
end

function output_fields(c::GroundSoil, case)
    return OutputVariable[
        OutputVariable("T_soil1", "K", "soil temperature, top layer (node 7 mm; NaN where not soil)",
                    :xy, cs -> soil_map((s, I) -> s.water.T_soil[1], c, cs)),
        OutputVariable("G_soil", "W m-2", "ground heat flux into the soil, last surface step (NaN where not soil)",
                    :xy, cs -> soil_map((s, I) -> cs.bus.G[I], c, cs)),
        OutputVariable("theta_soil1", "m3 m-3", "liquid soil water, top layer (NaN where not soil)",
                    :xy, cs -> soil_map((s, I) -> s.water.θ_liq[1], c, cs))]
end

"""
    diagnostics(c::GroundSoil, case)

Over the soil columns: the moisture availability `beta_min`,
`beta_max`, `beta_mean` (1) and the liquid water of the top layer
`theta_top_min`, `theta_top_max` (m³ m⁻³). None without soil columns.
"""
function diagnostics(c::GroundSoil, case)
    isempty(c.cols) && return Diagnostic[]
    β() = [moisture_availability(case.bus.surface[I]) for I in c.cols]
    θ1() = [case.bus.surface[I].water.θ_liq[1] for I in c.cols]
    return Diagnostic[
        Diagnostic("beta_min", "1", β; reduction = :min, description = "moisture availability"),
        Diagnostic("beta_max", "1", β; reduction = :max, description = "moisture availability"),
        Diagnostic("beta_mean", "1", β; reduction = :mean, description = "moisture availability"),
        Diagnostic("theta_top_min", "m3 m-3", θ1; reduction = :min, description = "top-layer liquid water"),
        Diagnostic("theta_top_max", "m3 m-3", θ1; reduction = :max, description = "top-layer liquid water")]
end

function smoke_check(c::GroundSoil, case)
    cols = [case.bus.surface[I] for I in c.cols]
    isempty(cols) && return ["soil columns in the window" => false]
    Ts = case.bus.Ts
    # heat diffusion with a Dirichlet base and the skin on top obeys a
    # maximum principle: within the initial soil and base temperatures and
    # the skin's (1 K margin for the skin's move in a step)
    sk = [s.T_skin for s in cols]
    lo, hi = min(c.T_env[1], minimum(sk)) - 1, max(c.T_env[2], maximum(sk)) + 1
    Tall = reduce(vcat, (s.water.T_soil for s in cols))
    h = c.hydraulics
    θall = reduce(vcat, (s.water.θ_liq for s in cols))
    β = moisture_availability.(cols)
    return ["soil columns in the window, one per street/wall/park column with air" =>
                length(c.cols) == count(I -> case.geom.sfc[I] in c.classes &&
                                             case.bus.surface[I] !== nothing,
                                        CartesianIndices(case.geom.sfc)),
            "soil temperatures within [min(T_initial, T_skin) − 1, max(T_initial, T_skin) + 1] K" =>
                all(T -> lo <= T <= hi, Tall),
            "soil liquid water within [θ_r, θ_s]" =>
                all(θ -> h.θ_r <= θ <= h.θ_s, θall),
            "β ∈ [0, 1], and < 1 where the top layer is below field capacity" =>
                all(i -> 0 <= β[i] <= 1 &&
                         (cols[i].water.θ_liq[1] >= soil_field_capacity(h) || β[i] < 1),
                    eachindex(cols)),
            "the skin is the bus surface temperature" =>
                all(I -> case.bus.surface[I].T_skin == Ts[I], c.cols)]
end


# The surface spin-up (#519): the soil water is frozen (PALM's default,
# calc_soil_moisture_during_spinup = .F.) unless [spinup] soil_moisture;
# the columns' heat runs. The columns are the surface exchange's surfaces.
function spinup_begin!(c::GroundSoil, case, sp, t0, dt)
    sp.soil_moisture && return nothing
    frozen = SoilThermalColumn[]
    for s in case.bus.surface
        (s isa SoilThermalColumn && s.hydrology) || continue
        s.hydrology = false
        push!(frozen, s)
    end
    case.spin.saved[:ground_soil_frozen] = frozen
    return nothing
end
function spinup_end!(c::GroundSoil, case, t_end)
    for s in pop!(case.spin.saved, :ground_soil_frozen, SoilThermalColumn[])
        s.hydrology = true
    end
    return nothing
end
