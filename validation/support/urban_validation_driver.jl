# Offline street-canyon cases for the BUBBLE (Basel) and CAPITOUL (Toulouse)
# validation of the urban canopy (#30) — test infrastructure, not model code.
#
# Every piece of physics is the package's: the canyon is a Radiosity scene
# (ray-cast shading, exact view factors), every facet an UrbanSurface conduction
# column closed by its surface energy balance, the facets coupled implicitly
# through the longwave radiosity system, and the buildings BuildingEnergy zones
# solved together with their facets (`step_buildings!`). The TEB canyon closures
# (`canyon_wind_speed`, `rowley_convection_coefficient`) set the air-side
# exchange of walls and street. This file only builds the geometry and reads the
# forcing.
#
# Forcing: NASA POWER hourly point data for the campaign dates
# (test/data/nasa_power_*_hourly.csv; CERES SYN1deg radiation, MERRA-2
# meteorology). This is NOT the tower forcing the reference model (TEB) was run
# with — see the data-file headers and docs/src/theory/urban_validation.md.
#
# What is taken from where (and what is assumed) is labelled at each line.

using Aeolus
using Aeolus.UrbanSurface
using Aeolus.BuildingEnergy
using Aeolus.SurfaceExchange
using Aeolus.Radiosity
using Aeolus.SolarGeometry
using Aeolus.MoistThermodynamics
using Statistics: mean, std

# ---------------------------------------------------------------------------
# Forcing
# ---------------------------------------------------------------------------

"Hourly columns of a NASA POWER extract in test/data (see its header for units)."
function power_forcing(file)
    path = joinpath(@__DIR__, "..", "data", file)
    rows = [split(l, ',') for l in eachline(path) if !startswith(l, '#') && !startswith(l, "year")]
    col(i, T = Float64) = [parse(T, r[i]) for r in rows]
    return (year = col(1, Int), month = col(2, Int), day = col(3, Int), hour = col(4, Int),
            LW = col(5), SW = col(6), DNI = col(7), DIF = col(8), T2M = col(9), RH = col(10),
            WS10 = col(11), WS50 = col(12), PS = 1000 .* col(13))
end

"Index of the record (year, month, day, hour 0)."
power_index(w, y, m, d) =
    findfirst(k -> (w.year[k], w.month[k], w.day[k], w.hour[k]) == (y, m, d, 0), eachindex(w.year))

# ---------------------------------------------------------------------------
# Sites and canyons
# ---------------------------------------------------------------------------

"""
A street canyon: `H`, `W` (m), the azimuth of the street axis (deg from north)
and the grid it is built on (cell `d`, `ny` cells along the street, building
rows `nb` cells deep, so the roof width of one repeat unit is `nb d`).
"""
struct ValidationCanyon
    name::String
    H::Float64
    W::Float64
    axis::Float64
    d::Float64
    ny::Int
    nb::Int
end

# CAPITOUL canyons. H and W: Moscicki & Voogt (2007), Table 2.1 (Alsace 20.5/14.0,
# Pomme 20.5/10.0, Remusat 16.8/10.0 m). Street azimuths: Pigeon et al. (2009,
# AMS 8th Symposium on the Urban Environment, paper 3.6 — the extended abstract of
# Pigeon et al. 2008, Meteorol. Atmos. Phys. 102, 159), Table 3 (0, 120, 60 deg;
# walls facing 90/270, 210/30, 150/330). Grid: 2 cells across the street, canyon
# length >= 100 m ≈ 5 H so that the central road's sky-view factor is within
# 2 % of the infinite canyon (measured in the test).
# The building depth is not needed for facet temperatures; 2 cells.
const CAPITOUL_CANYONS = (ValidationCanyon("Alsace", 20.5, 14.0, 0.0, 7.0, 15, 2),
                          ValidationCanyon("Pomme", 20.5, 10.0, 120.0, 5.0, 20, 2),
                          ValidationCanyon("Remusat", 16.8, 10.0, 60.0, 5.0, 20, 2))

const CAPITOUL_SITE = (latitude = 43.6045, longitude = 1.4440, elevation = 146.0)  # Qu (2011) §6.2; elevation: assumption

# Construction and optics of the TEB run of Pigeon et al. (2009, paper 3.6),
# Table 2 (the same values in Henon et al. 2009, AMS 8th Urban Env. Symp. JP3.2).
# Only C = ρc is published; ρ = 1000 kg m⁻³ is a nominal split that does not
# enter the conduction (only k and ρc do).
const _PIGEON = "Pigeon et al. (2009), AMS 8th Symp. Urban Environment, paper 3.6, Table 2 (TEB input, CAPITOUL)"
_capitoul_material(name, C, k) = ThermalMaterial(name, k, 1000.0, C / 1000.0; source = _PIGEON)
const CAPITOUL_BRICK     = _capitoul_material(:capitoul_red_brick, 1.58e6, 1.15)
const CAPITOUL_TILE      = _capitoul_material(:capitoul_red_tile, 1.58e6, 1.15)
const CAPITOUL_WOOD      = _capitoul_material(:capitoul_wood, 2.20e6, 0.20)
const CAPITOUL_ASPHALT   = _capitoul_material(:capitoul_asphalt, 1.74e6, 0.82)
const CAPITOUL_AGGREGATE = _capitoul_material(:capitoul_aggregate, 2.00e6, 2.1)
const CAPITOUL_GRAVEL    = _capitoul_material(:capitoul_gravel_soil, 1.40e6, 0.4)

# walls 1 + 5 + 18 + 5 + 1 cm of brick; roofs 6 cm tile + 3 cm wood; road 5 cm
# asphalt + 20 cm aggregate + 1 m gravel and soil (Pigeon et al. Table 2)
capitoul_wall_layers() = [WallLayer(CAPITOUL_BRICK, 0.30; max_cell = 0.02)]
capitoul_roof_layers() = [WallLayer(CAPITOUL_TILE, 0.06; ncells = 4),
                          WallLayer(CAPITOUL_WOOD, 0.03; ncells = 2)]
capitoul_road_layers() = [WallLayer(CAPITOUL_ASPHALT, 0.05; ncells = 4),
                          WallLayer(CAPITOUL_AGGREGATE, 0.20),
                          WallLayer(CAPITOUL_GRAVEL, 1.00; max_cell = 0.1)]
# (albedo, emissivity) per Radiosity class: street, roof, park (unused), water (unused), wall.
capitoul_materials() = UrbanMaterials([0.08, 0.15, 0.2, 0.05, 0.25], [0.95, 0.90, 0.95, 0.96, 0.92];
                                      name = ["street", "roof", "park", "water", "wall"])

# BUBBLE, Basel-Sperrstrasse (U1). Hamdi & Schayes (2005, ACPD 5, 4257):
# Table 1 — 47°33' N 7°35' E, 255 m, building height 14.6 m, streets at 67° and
# 157°; Table 3 — facet sizes wall 14.6, roof 15, road 11.23 m (H/W = 1.30,
# plan fraction of roofs 15/26.23 = 0.572), albedo wall/roof/road 0.14/0.14/0.08,
# emissivity 0.90/0.90/0.94, thermal diffusivity (their "KS", m² s⁻¹) and
# volumetric heat capacity 6.7e-7 / 1.4e6 (wall, roof) and 2.8e-7 / 1.8e6 (road).
# Grid: d = W/3 = 3.743 m, roofs 4 cells (14.97 m), canyon 21 cells (79 m ≈ 5.4 H).
const BUBBLE_SITE = (latitude = 47.55, longitude = 7.5833, elevation = 255.0)
const BUBBLE_CANYONS = (ValidationCanyon("Sperrstrasse 67", 14.6, 11.23, 67.0, 11.23 / 3, 21, 4),
                        ValidationCanyon("Sperrstrasse 157", 14.6, 11.23, 157.0, 11.23 / 3, 21, 4))
const _HS05 = "Hamdi & Schayes (2005), Atmos. Chem. Phys. Discuss. 5, 4257, Table 3 (Basel U1)"
_bubble_material(name, κ, C) = ThermalMaterial(name, κ * C, 1000.0, C / 1000.0; source = _HS05)
const BUBBLE_FABRIC = _bubble_material(:bubble_wall_roof, 6.7e-7, 1.4e6)
const BUBBLE_ROAD   = _bubble_material(:bubble_road, 2.8e-7, 1.8e6)
# Layer thicknesses are not published; 0.30 m masonry walls and roofs and a 0.50 m
# road over a fixed deep temperature are assumptions (≥ 2.2 diurnal damping
# depths √(2κ/ω) = 0.136 m and 0.088 m, so the diurnal surface response does not
# depend on them).
bubble_wall_layers() = [WallLayer(BUBBLE_FABRIC, 0.30)]
bubble_roof_layers() = [WallLayer(BUBBLE_FABRIC, 0.30)]
bubble_road_layers() = [WallLayer(BUBBLE_ROAD, 0.50)]
bubble_materials() = UrbanMaterials([0.08, 0.14, 0.2, 0.05, 0.14], [0.94, 0.90, 0.95, 0.96, 0.90];
                                    name = ["street", "roof", "park", "water", "wall"])

# ---------------------------------------------------------------------------
# Published observations (test/data; provenance in the file headers)
# ---------------------------------------------------------------------------

"CAPITOUL statistics of Moscicki & Voogt (2007), Table 4.1: `stat[:O].walls` etc."
function capitoul_statistics()
    path = joinpath(@__DIR__, "..", "data", "capitoul_surface_temperature_statistics.csv")
    rows = [split(l, ',') for l in eachline(path) if !startswith(l, '#') && !startswith(l, "statistic")]
    return Dict(Symbol(r[1]) => (roads = parse(Float64, r[2]), walls = parse(Float64, r[3]),
                                 roof = parse(Float64, r[4])) for r in rows)
end

"BUBBLE table: `bubble[(source, period, quantity)] = (obs, SBL, REF, rmseSBL, rmseREF)` (NaN if absent)."
function bubble_observations()
    path = joinpath(@__DIR__, "..", "data", "bubble_sperrstrasse_energy_balance.csv")
    rows = [split(l, ',') for l in eachline(path) if !startswith(l, '#') && !startswith(l, "source")]
    num(x) = isempty(x) ? NaN : parse(Float64, x)
    return Dict((r[1], r[2], r[3]) => (obs = num(r[4]), SBL = num(r[5]), REF = num(r[6]),
                                       rmseSBL = num(r[7]), rmseREF = num(r[8])) for r in rows)
end

# ---------------------------------------------------------------------------
# Building the case
# ---------------------------------------------------------------------------

"""
    canyon_scene(c::ValidationCanyon; materials, nsub = 3) -> (scene, view_factors, ns)

The radiosity scene of the canyon `c`: two building rows `c.nb` cells deep
either side of a street `ns = W/d` cells wide, `c.ny` cells long, walls split
into `nsub` strips.
"""
function canyon_scene(c::ValidationCanyon; materials, nsub = 3)
    ns = round(Int, c.W / c.d)
    ns * c.d ≈ c.W || throw(ArgumentError("canyon_scene: W must be a multiple of d"))
    nb = c.nb
    hm = zeros(2nb + ns, c.ny); hm[1:nb, :] .= c.H; hm[nb+ns+1:end, :] .= c.H
    sc = build_scene(hm, c.d, c.d; wall_subdivisions = nsub, materials = materials)
    return sc, view_factors(sc), ns
end

"The central `rows` of the canyon (the two middle cells along the street)."
central_rows(c::ValidationCanyon) = (c.ny ÷ 2, c.ny ÷ 2 + 1)

"""
    canyon_case(c::ValidationCanyon; layers, materials, T0, T_deep, T_heat, nsub = 3)

A canyon of `c` between two building rows (`c.nb` cells deep), its
`UrbanFacetScene`, one `BuildingZone` per row (heated to `T_heat`, no cooling —
the TEB assumption of Pigeon et al. 2009, T_i,min = 292 K) and the facet sets of
the central two rows: `road`, `wall_pos` (normal +x of the scene), `wall_neg`
and `roof`. The street axis is the scene's +y; the real facing azimuths of the
two walls are `c.axis + 90` and `c.axis + 270`.
"""
function canyon_case(c::ValidationCanyon; layers, materials, T0, T_deep, T_heat = 292.15,
                     z0 = (road = (0.05, 2.5e-4), roof = (0.15, 7.5e-4)), nsub = 3)
    nb = c.nb
    sc, vf, ns = canyon_scene(c; materials, nsub)
    nx = 2nb + ns
    # Roughness lengths: TEB values quoted by Masson et al. (2002) and Lemonsu
    # et al. (2004) — road z0 = 0.05 m, z0h 2.5e-4 m; roof 0.15 / 7.5e-4 m.
    opts(f) = f.kind === :wall ? (h_convective = 10.0, T_interior = T_heat) :
              f.kind === :roof ? (z0m = z0.roof[1], z0h = z0.roof[2], T_interior = T_heat) :
                                 (z0m = z0.road[1], z0h = z0.road[2], T_interior = T_deep)
    u = UrbanFacetScene(sc, vf; T = T0, layers = f -> layers[f.kind], options = opts)
    fps = [falses(nx, c.ny), falses(nx, c.ny)]
    fps[1][1:nb, :] .= true; fps[2][nb+ns+1:end, :] .= true
    zones = [scene_building_zone(u, fp; floor_height = 3.0, T_air = T_heat, T_ground = T_deep,
                                 infiltration_ach = 0.5) for fp in fps]           # 3 m floors, 0.5 ACH: assumptions
    rows = central_rows(c)
    street = nb+1:nb+ns
    sel(pred) = [i for (i, f) in enumerate(sc.facets) if f.col[2] in rows && pred(f)]
    road = sel(f -> f.kind === :ground)
    wall_pos = sel(f -> f.kind === :wall && f.normal[1] > 0.5 && f.col[1] in street)
    wall_neg = sel(f -> f.kind === :wall && f.normal[1] < -0.5 && f.col[1] in street)
    roof = sel(f -> f.kind === :roof)
    # One repeat unit of an infinite array of such canyons, over the central
    # rows: the street, both canyon walls and one building width of roof
    # (half of each row's roof). Plan area (nb d + W) × 2d.
    unit = vcat(road, wall_pos, wall_neg, roof)
    unit_w = vcat(ones(length(road) + length(wall_pos) + length(wall_neg)), fill(0.5, length(roof)))
    A_plan = (nb * c.d + c.W) * 2c.d
    return (c = c, u = u, zones = zones, road = road, wall_pos = wall_pos, wall_neg = wall_neg,
            roof = roof, unit = unit, unit_w = unit_w, A_plan = A_plan,
            λp = nb * c.d / (nb * c.d + c.W),
            hvac = [IdealLoadsHVAC(T_heat = T_heat, T_cool = 400.0, cooling_capacity = 0.0,
                                   heating = :electric) for _ in 1:2])
end

"Area-weighted mean skin temperature of the facets `ids`."
area_mean_T(u, ids) = sum(u.vf.area[i] * u.facets[i].T_skin for i in ids) / sum(u.vf.area[ids])

"""
    run_canyon!(case, w, k0, nhours; site, z_forcing = 50.0, dt = 1800.0, T_offset = 0.0)

Drive `case` through `nhours` hourly POWER records from index `k0`, with `dt`
sub-steps (the hour's mean forcing held over the hour; the sun at mid-step).

Air-side forcing (all labelled assumptions of this offline case):
  * the above-roof air temperature, humidity and pressure are POWER T2M, RH2M,
    PS (+ `T_offset`, K); the canyon air is taken equal to it (no canyon-air
    model — see the theory page);
  * the wind is POWER WS50M, taken at `z_forcing` m above the street, so
    `z_forcing - H` above the roofs; roofs see it through MOST, the street
    through MOST with the TEB canyon wind at H/2, the walls through
    `rowley_convection_coefficient(canyon_wind_speed(...))`.

`T_offset` is a number or a vector indexed like the records.

Returns per-sample vectors (one sample per sub-step, at its end): the skin
temperatures of the central road, the two canyon walls and the roofs, the heat
the roofs conduct into the rooms below (`roof_interior`, W m⁻²), and the
fluxes of one repeat unit of the canyon array per unit plan area (W m⁻²):
`Qstar`, `QH`, `dQS` (heat into the fabric, including what the facets conduct
into the rooms), `Sup` and `Lup` (the shortwave and longwave leaving the unit,
`S↓ − Σ A·absorbed / A_plan` and `L↓ − Σ A·lw_net / A_plan`).
"""
function run_canyon!(case, w, k0, nhours; site, z_forcing = 50.0, dt = 1800.0, T_offset = 0.0)
    u, c = case.u, case.c
    nsub = round(Int, 3600 / dt)
    n = nfacets(u)
    out = (road = Float64[], wall_pos = Float64[], wall_neg = Float64[], roof = Float64[],
           roof_interior = Float64[],
           T_air = Float64[], hour = Int[], month = Int[], day = Int[], SWdn = Float64[],
           Qstar = Float64[], QH = Float64[], dQS = Float64[], Sup = Float64[], Lup = Float64[])
    A = u.vf.area
    wA = [case.unit_w[j] * A[i] for (j, i) in enumerate(case.unit)]
    unit_sum(x) = sum(wA[j] * x[i] for (j, i) in enumerate(case.unit)) / case.A_plan
    for k in k0:k0+nhours-1
        T_air = w.T2M[k] + 273.15 + (T_offset isa Real ? T_offset : T_offset[k])
        p = w.PS[k]
        q = specific_humidity_from_relative_humidity(T_air, p, clamp(w.RH[k] / 100, 0.0, 1.0))
        U = max(w.WS50[k], 0.5)
        z_top = z_forcing - c.H
        U_can = canyon_wind_speed(U, z_top, c.H, c.W)
        h_wall = rowley_convection_coefficient(U_can)
        roof_air = SurfaceForcing(lw_down = w.LW[k], T_air = T_air, q_air = q, U = U, p = p,
                                  z_ref = z_top)
        can_air = SurfaceForcing(lw_down = w.LW[k], T_air = T_air, q_air = q,
                                 U = max(U_can, 0.1), p = p, z_ref = c.H / 2)
        air = [u.scene.facets[i].kind === :roof ? roof_air : can_air for i in 1:n]
        for f in u.facets
            f.kind === :wall && (f.h_convective = h_wall)
        end
        zf = [ZoneForcing(T_out = T_air, q_out = q) for _ in case.zones]
        for s in 1:nsub
            t_mid = (s - 0.5) * dt
            hh = w.hour[k] + floor(Int, t_mid / 3600)
            mm = floor(Int, (t_mid % 3600) / 60)
            ss = t_mid % 60
            st = solar_time(w.year[k], w.month[k], w.day[k], hh, mm, ss;
                            latitude = site.latitude, longitude = site.longitude,
                            elevation = site.elevation)
            sp = solar_position(st)
            # the scene's +y is the street axis: rotate the sun into the scene frame
            sun = (elevation = sp.elevation, azimuth = sp.azimuth - c.axis)
            # Beam and diffuse such that the horizontal irradiance the scene
            # receives is POWER's hourly mean global irradiance exactly: the
            # beam part SW - DIF is applied at the mid-step sun as DNI =
            # (SW - DIF)/sin(elevation); below 3° elevation it is added to the
            # diffuse part instead (no grazing beam from an hourly mean).
            s_el = sind(sp.elevation)
            beam = max(w.SW[k] - w.DIF[k], 0.0)
            dni, dhi = s_el > sind(3.0) ? (beam / s_el, w.DIF[k]) : (0.0, w.DIF[k] + beam)
            sw = if dni > 0
                solve_shortwave(u.scene, u.vf, sun; dni = dni, dhi = dhi)
            elseif dhi > 0
                solve_shortwave(u.scene, u.vf, (0.0, 0.0, 1.0); dni = 0.0, dhi = dhi)
            else
                nothing
            end
            fl, _, _ = step_buildings!(u, case.zones, zf, air, dt; hvac = case.hvac, sw = sw,
                                       L_sky = w.LW[k])
            sw_dn = dni * s_el + dhi          # = POWER SW (horizontal) whenever sw ≠ nothing
            push!(out.Qstar, unit_sum([f.Rn for f in fl]))
            push!(out.QH, unit_sum([f.H for f in fl]))
            push!(out.dQS, unit_sum([f.G for f in fl]))
            push!(out.Sup, sw === nothing ? 0.0 : sw_dn - unit_sum(u.sw_absorbed))
            push!(out.Lup, w.LW[k] - unit_sum(u.lw_net))
            push!(out.SWdn, sw === nothing ? 0.0 : sw_dn)
            push!(out.road, area_mean_T(u, case.road))
            push!(out.wall_pos, area_mean_T(u, case.wall_pos))
            push!(out.wall_neg, area_mean_T(u, case.wall_neg))
            push!(out.roof, area_mean_T(u, case.roof))
            push!(out.roof_interior, mean(u.facets[i].Q_interior for i in case.roof))
            push!(out.T_air, T_air)
            push!(out.hour, w.hour[k]); push!(out.month, w.month[k]); push!(out.day, w.day[k])
        end
    end
    return out
end
