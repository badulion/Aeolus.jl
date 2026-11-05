# Dense residential centre of Toulouse (CAPITOUL) as a coupled urban scene
# with building zones — test infrastructure for #29, not model code.
#
# Configuration: Bueno et al. (2012), Geosci. Model Dev. 5, 433-448, Table 3
# ("an urban area composed of residential buildings in the dense urban centre
# of Toulouse"): average building height 20 m, building density 0.68,
# vertical-to-horizontal area ratio 1.05, glazing ratio 0.3 (double clear
# glass 6 mm / 6 mm gap), walls and roofs brick 30 cm + insulation board 3 cm,
# infiltration 0.5 ACH, internal heat gains 5.8 W m-2 of floor (radiant 0.2,
# latent 0.2), set points 19-24 degC, 2/3 electric : gas heating.
#
# Geometry: a 3 x 3 array of 50 m square blocks, 20 m tall, separated by 10 m
# streets (period 60 m): building density 0.694 and wall-to-plan ratio 1.11
# against the 0.68 and 1.05 of Table 3. The scene's lateral boundary is open,
# so only the central block, surrounded by the others, is used for per-area
# numbers; every block is a zone, so its neighbours are heated too.
#
# Weather: PVGIS TMY for the Toulouse centre (validation/data/pvgis_tmy_toulouse_hourly.csv;
# ERA5 + SARAH2), a typical year, not the 2004-05 campaign.
#
# Assumptions made here that Bueno et al. do not state (each is labelled in
# the code): floor height 3 m, window U = 3.1 W m-2 K-1 and SHGC 0.7 (6 mm
# air-filled double clear glazing), expanded-polystyrene insulation board,
# 10 cm concrete intermediate floors (Bueno et al. Table 1's internal mass),
# wall convective coefficient 10 W m-2 K-1, boiler efficiency 0.9.

using Aeolus
using Aeolus.BuildingEnergy
using Aeolus.UrbanSurface
using Aeolus.SurfaceExchange
using Aeolus.Radiosity
using Aeolus.SolarGeometry
using Aeolus.MoistThermodynamics

const TLS_SITE = (latitude = 43.604, longitude = 1.444, elevation = 155.0, irr_offset_h = 0.1746)

"Hourly columns of the PVGIS Toulouse TMY file (°C, %, W m⁻², m s⁻¹, Pa)."
function tls_weather(path = joinpath(@__DIR__, "..", "data", "pvgis_tmy_toulouse_hourly.csv"))
    rows = [split(l, ',') for l in eachline(path) if !startswith(l, '#') && !startswith(l, "month")]
    col(i, T = Float64) = [parse(T, r[i]) for r in rows]
    return (month = col(1, Int), day = col(2, Int), hour = col(3, Int), T2m = col(4), RH = col(5),
            Gh = col(6), Gbn = col(7), Gdh = col(8), IRh = col(9), WS = col(10), SP = col(11))
end

# Solar position at the PVGIS irradiance time of the record (a non-leap year).
function tls_sun(month, day, hour_utc)
    t = hour_utc + TLS_SITE.irr_offset_h
    h = floor(Int, t)
    st = solar_time(2001, month, day, h, 0, (t - h) * 3600;
                    latitude = TLS_SITE.latitude, longitude = TLS_SITE.longitude,
                    elevation = TLS_SITE.elevation)
    return solar_position(st)
end

# Bueno et al. Table 3 envelope: brick 30 cm + insulation board 3 cm (outside
# first). "Insulation board" is taken as expanded polystyrene (assumption).
tls_envelope() = [WallLayer(:brick, 0.30), WallLayer(:eps_insulation, 0.03)]

"""
    tls_scene(; nblocks = 3, block = 5, street = 1, cell = 10.0, H = 20.0, subdiv = 2)

The block array, its `UrbanFacetScene` and one `BuildingZone` per block.
Returns `(u, zones, footprints, central)`; `central` is the index of the
central block and `tile` its 60 × 60 m plan area.
"""
function tls_scene(; nblocks = 3, block = 5, street = 1, cell = 10.0, H = 20.0, subdiv = 2,
                   T0 = 285.0, T_in = 292.15)
    n = nblocks * (block + street) + street
    hm = zeros(n, n)
    fps = BitMatrix[]
    for bj in 1:nblocks, bi in 1:nblocks
        i0 = street + (bi - 1) * (block + street)
        j0 = street + (bj - 1) * (block + street)
        fp = falses(n, n); fp[i0+1:i0+block, j0+1:j0+block] .= true
        hm[fp] .= H
        push!(fps, fp)
    end
    sc = build_scene(hm, cell, cell; wall_subdivisions = subdiv)
    vf = view_factors(sc)
    layers(f) = f.kind === :ground ? example_ground_layers() : tls_envelope()
    opts(f) = f.kind === :wall ? (h_convective = 10.0, T_interior = T_in) :     # assumption
              f.kind === :roof ? (T_interior = T_in,) : (T_interior = 285.15,)
    u = UrbanFacetScene(sc, vf; T = T0, layers, options = opts)
    zones = map(fps) do fp
        wall_area = sum(u.vf.area[i] for i in building_facets(u, fp) if sc.facets[i].kind === :wall)
        scene_building_zone(u, fp; floor_height = 3.0,                          # assumption
                            floor_layers = [WallLayer(:concrete, 0.10)],         # Bueno Table 1
                            window_area = 0.3 * wall_area, window_U = 3.1,       # U: assumption
                            infiltration_ach = 0.5, radiative_fraction = 0.2, T_air = T_in)
    end
    central = (nblocks * nblocks + 1) ÷ 2
    tile = ((block + street) * cell)^2
    return (u = u, zones = zones, footprints = fps, central = central, tile = tile)
end

"""
    tls_run(sim, w, hours; ihg = 5.8, T_heat = 292.15, T_cool = 297.15, shgc = 0.7)

Step the scene and its zones through the TMY records `hours` (indices into
`w`). Returns per-hour vectors for the central zone and the domain.
"""
function tls_run(sim, w, hours; ihg = 5.8, T_heat = 292.15, T_cool = 297.15, shgc = 0.7,
                 record = trues(length(hours)))
    u, zones = sim.u, sim.zones
    m = length(zones)
    hv = [IdealLoadsHVAC(T_heat = T_heat, T_cool = T_cool, heating = :electric) for _ in 1:m]
    wall_ids = [[i for i in building_facets(u, fp) if u.scene.facets[i].kind === :wall]
                for fp in sim.footprints]
    c = sim.central
    out = (Q_heating = Float64[], Q_cooling = Float64[], Q_internal = Float64[],
           release = Float64[], direct = Float64[], via_facets = Float64[], T_air = Float64[],
           T_out = Float64[], Q_solar = Float64[], H_scene = Float64[])
    dt = 3600.0
    for (n, k) in enumerate(hours)
        T_out = w.T2m[k] + 273.15
        p = w.SP[k]
        q_out = specific_humidity_from_relative_humidity(T_out, p, clamp(w.RH[k] / 100, 0.0, 1.0))
        air = SurfaceForcing(lw_down = w.IRh[k], T_air = T_out, q_air = q_out,
                             U = max(w.WS[k], 0.5), p = p, z_ref = 10.0)
        sp = tls_sun(w.month[k], w.day[k], w.hour[k])
        up = sp.elevation > 0 && (w.Gbn[k] > 0 || w.Gdh[k] > 0)
        sw = up ? solve_shortwave(u.scene, u.vf, sp; dni = w.Gbn[k], dhi = w.Gdh[k]) :
                  (w.Gdh[k] > 0 ? solve_shortwave(u.scene, u.vf, (0.0, 0.0, 1.0); dni = 0.0,
                                                  dhi = w.Gdh[k]) : nothing)
        # IHG per floor area, radiant 0.2 and latent 0.2 of the total (Table 3)
        fs = map(1:m) do j
            z = zones[j]
            Qs = sw === nothing ? 0.0 :
                 shgc * 0.3 * sum(u.vf.area[i] * sw.incident[i] for i in wall_ids[j])
            ZoneForcing(T_out = T_out, q_out = q_out, Q_internal = 0.8 * ihg * z.floor_area,
                        Q_latent = 0.2 * ihg * z.floor_area, Q_solar = Qs)
        end
        fl, bs, hbs = step_buildings!(u, zones, fs, air, dt; hvac = hv, sw = sw,
                                      L_sky = w.IRh[k])
        record[n] || continue
        rel = building_heat_release(zones[c], fs[c], bs[c]; dt = dt, hvac = hv[c],
                                    hvac_budget = hbs[c])
        push!(out.Q_heating, hbs[c].Q_heating); push!(out.Q_cooling, hbs[c].Q_cooling)
        push!(out.Q_internal, ihg * zones[c].floor_area)
        push!(out.release, total_heat_release(rel))
        push!(out.direct, anthropogenic_sensible_heat(rel) + anthropogenic_latent_heat(rel))
        push!(out.via_facets, rel.Q_envelope_facets)
        push!(out.T_air, bs[c].T_air); push!(out.T_out, T_out); push!(out.Q_solar, fs[c].Q_solar)
        agg = aggregate_facet_fluxes(u, fl)
        push!(out.H_scene, sum(agg.H) * u.scene.dx * u.scene.dy)
    end
    return out
end

# Record indices for a period of the TMY: (month, day) inclusive, wrapping the
# year end (e.g. 19 Dec → 17 Feb).
function tls_period(w, from, to)
    key(k) = (w.month[k], w.day[k])
    idx = Int[]
    k0 = findfirst(k -> key(k) == from && w.hour[k] == 0, eachindex(w.month))
    k = k0
    while true
        push!(idx, k)
        (key(k) == to && w.hour[k] == 23) && break
        k = k == length(w.month) ? 1 : k + 1
    end
    return idx
end
