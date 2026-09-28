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
# The sky longwave and (unless the column radiation is told to leave it
# to Bird) the direct/diffuse split come from the :column_radiation slot
# through the bus (`bus.L_sky`, `bus.dni`, `bus.dhi`, #130).
#
# KNOWN (fixed by later issues, kept for bit-for-bit):
#   * no class_map in the scene (shortwave-modifier issue);
#   * one temperature per column is pushed onto every facet of that
#     column and the column's facet fluxes are collapsed back by the area
#     density (UrbanFacetScene issue).

"""
    RadiositySun

Slot `:surface_radiation`, provider `:radiosity`. At the `:radiation`
cadence (and on step 1): the solar position (`SolarGeometry`), Bird
DNI/DHI from the bus (the column radiation) or, when those are `NaN`,
the Bird clear sky `sky`, the sky longwave `bus.L_sky`, the μ₀-dependent materials
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
    isnan(bus.L_sky) && error("radiosity_sun: no sky longwave on the bus (bus.L_sky); the :column_radiation slot sets it before this stage")
    irr = isnan(bus.dni) ? clear_sky_irradiance(sp, c.sky) : (dni = bus.dni, dhi = bus.dhi)
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
    lwsol = solve_longwave(scene, vfac; L_sky = bus.L_sky)
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
