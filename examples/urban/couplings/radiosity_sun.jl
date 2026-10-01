# Surface radiation: the sun (SolarGeometry + Bird clear sky) and the
# radiosity solve over the OSM facet scene (Radiosity), every radiation
# interval.
#
# Radiosity subsumes what used to be four separate approximations: the
# binary shadow mask, the SVF-weighted diffuse, the wall directional
# factor, and the canyon-longwave proxy. Shadowing, wall orientation,
# inter-reflection and longwave trapping are all consequences of the
# view-factor matrix.
#
# The ground facet of each open column carries the material of its
# surface class (`scene_class_map` in core/geometry.jl, #128), so park
# and water get their own albedo and emissivity here — including the
# water's Fresnel zenith dependence, which the μ₀-dependent materials
# below update at every radiation call.
#
# KNOWN (fixed by later issues, kept for bit-for-bit):
#   * sky longwave is taken from the surface, L_sky = 0.95 σ T̄s⁴
#     (column-radiation issue);
#   * one temperature per column is pushed onto every facet of that
#     column and the column's facet fluxes are collapsed back by the area
#     density (UrbanFacetScene issue).

"""
    lw_from_walls(Ts_mean, ε = 0.95) -> W/m²

Longwave `ε σ Ts_mean⁴` of surfaces at `Ts_mean`. Used (as a known
defect) as the sky longwave of the radiosity solve.
"""
lw_from_walls(Ts_mean::Real, ε::Real = 0.95) = ε * STEFAN * Ts_mean^4

"""
    RadiositySun

Slot `:surface_radiation`, provider `:radiosity`. At the `:radiation`
cadence (and on step 1): the solar position (`SolarGeometry`), Bird
clear-sky DNI/DHI with the `sky` model, the μ₀-dependent materials
(`default_urban_materials`), the shortwave and longwave radiosity solves
over the static scene, and the per-column aggregation into `bus.sw`,
`bus.lw` (per unit surface). Leaves the solutions in `bus.swsol`,
`bus.lwsol` for the buildings.
"""
struct RadiositySun{SK} <: AbstractCoupling
    sky::SK
end

coupling_name(::Type{<:RadiositySun}) = :radiosity_sun
slot(::Type{<:RadiositySun}) = :surface_radiation
provider(::Type{<:RadiositySun}) = :radiosity
modules_used(::Type{<:RadiositySun}) = (:Radiosity, :SolarGeometry, :SurfaceOptics)
default_options(::Type{<:RadiositySun}) =
    (aod380 = 0.24, aod500 = 0.16, precipitable_water = 2.0, ozone = 0.33, albedo = 0.15)
stages(::Type{<:RadiositySun}) = ((:radiation, 10, Cadence(:radiation; first = true)),)

build(::Type{<:RadiositySun}, case, o) =
    RadiositySun(BirdClearSky(; aod380 = o.aod380, aod500 = o.aod500,
                              precipitable_water = o.precipitable_water,
                              ozone = o.ozone, albedo = o.albedo))

function run_stage!(c::RadiositySun, case, ::Val{:radiation}, dt)
    scene, vfac, bus = case.geom.scene, case.geom.vfac, case.bus
    Ts = bus.Ts
    case.clock.sun = solar_position_at(case.cfg, case.clock.hour)
    sp  = case.clock.sun
    irr = clear_sky_irradiance(sp, c.sky)
    μ   = max(sp.μ₀, 0.0)

    mats = default_urban_materials(; μ₀ = max(μ, 0.05))
    scene.materials.albedo     .= mats.albedo
    scene.materials.emissivity .= mats.emissivity
    for (fi, (ci, cj)) in enumerate(case.geom.facet_col)
        set_temperature!(scene, fi, Ts[ci, cj])
    end
    sunv  = solar_unit_vector(sp)
    swsol = μ > 0 ? solve_shortwave(scene, vfac, sunv;
                                    dni = irr.dni, dhi = irr.dhi) : nothing
    lwsol = solve_longwave(scene, vfac; L_sky = lw_from_walls(mean(Ts)))
    bus.swsol = swsol
    bus.lwsol = lwsol
    forcing = aggregate_to_grid(scene, vfac; sw = swsol, lw = lwsol, T_surface = Ts)
    # `aggregate_to_grid` returns fluxes per unit PLAN area, so a column
    # holding a 200 m tower reports ~13x the incident flux — its wall
    # area is 13x its footprint. That is the correct column ENERGY, but
    # the slab carries one temperature per column and wants an irradiance
    # per unit of actual SURFACE: divide by the area density to get the
    # area-weighted mean over the column's facets. `UrbanFacetScene`
    # removes this approximation (one skin temperature per facet).
    ad = max.(forcing.area_density, 1.0)
    σSB = 5.670374419e-8
    bus.sw .= forcing.sw_effective ./ ad
    bus.lw .= σSB .* Ts.^4 .+
              (forcing.lw_down_effective .- σSB .* Ts.^4) ./ ad
    bus.sw_beam = irr.dni * μ
    return nothing
end

function smoke_check(c::RadiositySun, case)
    bus, sw = case.bus, case.bus.swsol
    checks = ["surface shortwave ≥ 0" => all(>=(0), bus.sw),
              "surface longwave within 150–600 W/m²" => all(x -> 150 <= x <= 600, bus.lw)]
    # every open column's ground facet has the material of its class
    g = case.geom
    cm = scene_class_map(g.sfc)
    push!(checks, "ground facets carry their column's class material (park, water, street)" =>
          all(f -> f.kind !== :ground || f.material == cm[f.col...], g.scene.facets))
    lw = bus.lwsol
    # the facet longwave solve closes: absorbed = emitted + sky in − escaping
    push!(checks, "longwave scene budget closes (rel. 1e-9)" =>
          isapprox(lw.total_absorbed, lw.total_emitted + lw.total_from_sky - lw.total_escaping;
                   rtol = 1e-9))
    if case.clock.sun.μ₀ > 0
        push!(checks, "sun up: shortwave solved and reaching the surface" =>
              sw !== nothing && maximum(bus.sw) > 0)
        # every watt that enters the scene is absorbed or escapes to the sky
        push!(checks, "shortwave scene budget closes: absorbed + escaping = incident (rel. 1e-9)" =>
              isapprox(sw.total_absorbed + sw.total_escaping, sw.total_incident; rtol = 1e-9))
        push!(checks, "effective albedo within (0, max facet albedo]" =>
              0 < sw.effective_albedo <= maximum(case.geom.scene.materials.albedo) + 1e-12)
    end
    return checks
end

register!(RadiositySun)
