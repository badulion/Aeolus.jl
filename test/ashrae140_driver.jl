# ASHRAE Standard 140-2020 §5.2 (BESTEST) driver for BuildingEnergy.
#
# This file is test infrastructure, not model code: it reads the Standard 140
# Denver TMY3 weather (test/data/ashrae140_725650TYCST_hourly.csv), puts the
# sun on the six faces of the 8 × 6 × 2.7 m test cell, builds the in-depth
# test cases of §5.2.3 that have no transparent windows (220-250, 395-430,
# 800) from the normative material tables, and runs them for a year with an
# ideal-loads thermostat. The physics under test (conduction, air node,
# infiltration, internal gains, ideal loads) is all BuildingEnergy /
# UrbanSurface; the only modelling done here is the plane-of-array
# irradiance, which the Standard leaves to the program.
#
# Surface heat transfer follows the Standard's option for programs without
# time-step-varying surface coefficients or interior long-wave exchange
# (§5.2.1.9.3 b, §5.2.1.10.3 b, §5.2.3.1.4.3-4): constant *combined*
# coefficients, Tables 5-7, 5-9 (140-2020 Addendum a values), 5-38 and 5-39.
# BuildingEnergy's `EnvelopeElement` is exactly such a surface: combined
# outside coefficient with a sol-air source, combined inside coefficient to
# the air node.
#
# Sources
#   ANSI/ASHRAE Addendum a to Standard 140-2017 (2020): §5.2.1 (Case 600),
#     §5.2.2.2.1 (Case 900 materials), §5.2.3 (in-depth cases), Table A1-1.
#   ASHRAE 140-2020 Informative Annex B8, example results (RESULTS5-2A.xlsx,
#     as distributed with NREL BESTEST-GSR).

using Aeolus
using Aeolus.BuildingEnergy
using Aeolus.UrbanSurface
using Aeolus.SolarGeometry

const A140_SITE = (latitude = 39.833, longitude = -104.6509, utc_offset = -7, ground_albedo = 0.2)

"Hourly weather columns of the Standard 140 TMY3 file (°C, Pa, W m⁻², Wh m⁻²)."
function a140_weather(path = joinpath(@__DIR__, "data", "ashrae140_725650TYCST_hourly.csv"))
    rows = [split(l, ',') for l in eachline(path) if !startswith(l, '#') && !startswith(l, "month")]
    col(i, T = Float64) = [parse(T, r[i]) for r in rows]
    return (month = col(1, Int), day = col(2, Int), hour = col(3, Int), T_db = col(4),
            p = col(5), IR_h = col(6), GHI = col(7), DNI = col(8), DHI = col(9))
end

# Solar position at `hour_lst` local standard time of the given month/day.
# The TMY year is a composite; a non-leap year is used.
function a140_sun(month, day, hour_lst)
    utc = hour_lst - A140_SITE.utc_offset
    h = floor(Int, utc)
    st = solar_time(2001, month, day, h, 0, (utc - h) * 3600;
                    latitude = A140_SITE.latitude, longitude = A140_SITE.longitude,
                    elevation = 1650.0, pressure = 835.0, temperature = 10.0)
    return solar_position(st)
end

# Outward unit normal of a plane with tilt β (0 = facing up) and azimuth γ
# (clockwise from north: S = 180°), in east-north-up.
a140_normal(β, γ) = (sind(β) * sind(γ), sind(β) * cosd(γ), cosd(β))

# Perez et al. (1990), Solar Energy 44, 271-289, Table 6 ("all sites
# composite"): rows are the 8 sky-clearness bins, columns
# F11 F12 F13 F21 F22 F23 (as tabulated in pvlib `irradiance.py` and used by
# EnergyPlus).
const PEREZ_1990 = [-0.008  0.588 -0.062 -0.060  0.072 -0.022
                     0.130  0.683 -0.151 -0.019  0.066 -0.029
                     0.330  0.487 -0.221  0.055 -0.064 -0.026
                     0.568  0.187 -0.295  0.109 -0.152 -0.014
                     0.873 -0.392 -0.362  0.226 -0.462  0.001
                     1.132 -1.237 -0.412  0.288 -0.823  0.056
                     1.060 -1.600 -0.359  0.264 -1.127  0.131
                     0.678 -0.327 -0.250  0.156 -1.377  0.251]
const PEREZ_EPS_BINS = (1.065, 1.23, 1.5, 1.95, 2.8, 4.5, 6.2)

# Relative optical air mass, Kasten & Young (1989), zenith in degrees.
a140_airmass(z) = 1 / (cosd(z) + 0.50572 * (96.07995 - z)^-1.6364)

# Perez (1990) circumsolar and horizon-brightening coefficients (F1, F2)
# for sun zenith `z` (deg), DNI, DHI (W m⁻²) and extraterrestrial E0.
function a140_perez_F(z, dni, dhi, E0)
    zr = deg2rad(z); κ = 1.041
    ε = ((dhi + dni) / dhi + κ * zr^3) / (1 + κ * zr^3)
    Δ = dhi * a140_airmass(z) / E0
    b = something(findfirst(>(ε), PEREZ_EPS_BINS), 8)
    c = @view PEREZ_1990[b, :]
    F1 = max(0.0, c[1] + c[2] * Δ + c[3] * zr)
    F2 = c[4] + c[5] * Δ + c[6] * zr
    return F1, F2
end

"""
Incident irradiance (W m⁻², hour means) on each plane from DNI, DHI and
GHI: beam from the sun at `nsub` instants in the hour; sky diffuse by the
Perez (1990) anisotropic model (`sky = :perez`) or isotropically
(`sky = :isotropic`); ground reflection isotropic with ρ_g = 0.2.
"""
function a140_incident(w, planes; nsub = 6, sky = :perez)
    N = length(w.T_db)
    out = Dict(k => zeros(N) for k in keys(planes))
    E0 = Aeolus.RadiationCore.SOLAR_CONSTANT
    for i in 1:N
        sps = [a140_sun(w.month[i], w.day[i], w.hour[i] - 1 + (j - 0.5) / nsub) for j in 1:nsub]
        for (k, (β, γ)) in planes
            n = a140_normal(β, γ)
            beam = 0.0; dif = 0.0
            for sp in sps
                s = sun_direction_enu(sp)
                cosi = max(0.0, s[1] * n[1] + s[2] * n[2] + s[3] * n[3])
                iso = (1 + cosd(β)) / 2
                if s[3] > 0
                    beam += w.DNI[i] * cosi
                    if sky === :perez && w.DHI[i] > 0
                        F1, F2 = a140_perez_F(sp.zenith, w.DNI[i], w.DHI[i], E0 / sp.distance_au^2)
                        dif += w.DHI[i] * max(0.0, (1 - F1) * iso + F1 * cosi / max(cosd(sp.zenith), cosd(85)) +
                                                   F2 * sind(β))
                    else
                        dif += w.DHI[i] * iso
                    end
                else
                    dif += w.DHI[i] * iso
                end
            end
            out[k][i] = (beam + dif) / nsub +
                        A140_SITE.ground_albedo * w.GHI[i] * (1 - cosd(β)) / 2
        end
    end
    return out
end

const A140_PLANES = Dict(:horizontal => (0.0, 0.0), :north => (90.0, 0.0), :east => (90.0, 90.0),
                         :south => (90.0, 180.0), :west => (90.0, 270.0))

# ---- constructions (Addendum a, Tables 5-2, 5-27, 5-36) — outside first -------

a140_mat(name, k, ρ, c) = ThermalMaterial(name, k, ρ, c; source = "ANSI/ASHRAE 140-2020 §5.2")
# "minimum density and specific heat the program allows" for the underfloor
# insulation: 1e-3 of the fibreglass values, i.e. negligible storage.
const A140_M = (
    plasterboard = a140_mat(:plasterboard, 0.16, 950.0, 840.0),
    fiberglass   = a140_mat(:fiberglass_quilt, 0.04, 12.0, 840.0),
    wood_siding  = a140_mat(:wood_siding, 0.14, 530.0, 900.0),
    roofdeck     = a140_mat(:roofdeck, 0.14, 530.0, 900.0),
    timber_floor = a140_mat(:timber_flooring, 0.14, 650.0, 1200.0),
    floor_insul  = a140_mat(:floor_insulation, 0.04, 0.012, 0.84),
    block        = a140_mat(:concrete_block, 0.51, 1400.0, 1000.0),
    foam         = a140_mat(:foam_insulation, 0.04, 10.0, 1400.0),
    slab         = a140_mat(:concrete_slab, 1.13, 1400.0, 1000.0),
    pane         = a140_mat(:opaque_pane, 1.06, 2500.0, 750.0),
    # 13 mm gap, R = 0.1588 m² K W⁻¹ (Table 5-36, 140-2020 values)
    gap          = a140_mat(:hc_air_gap, 0.013 / 0.1588, 1.292498, 1006.103271),
)
const A140_LAYERS = (
    lt_wall  = () -> [WallLayer(A140_M.wood_siding, 0.009), WallLayer(A140_M.fiberglass, 0.066),
                      WallLayer(A140_M.plasterboard, 0.012)],
    lt_floor = () -> [WallLayer(A140_M.floor_insul, 1.003; ncells = 4),
                      WallLayer(A140_M.timber_floor, 0.025)],
    roof     = () -> [WallLayer(A140_M.roofdeck, 0.019), WallLayer(A140_M.fiberglass, 0.1118),
                      WallLayer(A140_M.plasterboard, 0.010)],
    hw_wall  = () -> [WallLayer(A140_M.wood_siding, 0.009), WallLayer(A140_M.foam, 0.0615),
                      WallLayer(A140_M.block, 0.100; max_cell = 0.01)],
    hw_floor = () -> [WallLayer(A140_M.floor_insul, 1.007; ncells = 4),
                      WallLayer(A140_M.slab, 0.080; max_cell = 0.01)],
    hc_wall  = () -> [WallLayer(A140_M.pane, 0.003175), WallLayer(A140_M.gap, 0.013),
                      WallLayer(A140_M.pane, 0.003175)],
)

# Constant surface coefficients, W m⁻² K⁻¹ (Tables 5-7, 5-9, 5-38, 5-39):
# interior combined `in`; exterior combined `out`, and exterior convective
# `conv` for programs that compute the exterior long-wave exchange.
const A140_H = (wall = (in = 1.8, out = 21.6, conv = 11.9), roof = (in = 1.7, out = 21.8, conv = 14.4),
                floor = (in = 3.7, out = 5.2, conv = 0.8), hc = (in = 4.5, out = 17.8, conv = 8.0))
const A140_EMISSIVITY = 0.9

# Infiltration heat-capacity flow per ach: ρ c_p V / 3600 with the Standard's
# moist-air values at 1609 m (Table 5-3 note f: 1.0156 kg m⁻³, 1015 J kg⁻¹ K⁻¹).
const A140_RHOCP = 1.0156 * 1015.0

"""
    a140_case(case) -> NamedTuple

Specification of an in-depth case without transparent windows (§5.2.3):
mass (:low/:high), south high-conductance element (true/false), infiltration
(ach), internal gain (W), exterior solar absorptance, setpoints (K).
"""
function a140_case(case::Integer)
    base = (mass = :low, hc = true, ach = 0.0, gain = 0.0, α = 0.1,
            T_heat = 293.15, T_cool = 293.15)          # Case 220: "20,20" thermostat
    c = case == 220 ? base :
        case == 230 ? merge(base, (ach = 1.0,)) :
        case == 240 ? merge(base, (gain = 200.0,)) :
        case == 250 ? merge(base, (α = 0.9,)) :
        case == 400 ? merge(base, (T_cool = 300.15,)) :
        case == 395 ? merge(base, (T_cool = 300.15, hc = false)) :
        case == 410 ? merge(base, (T_cool = 300.15, ach = 0.5)) :
        case == 420 ? merge(base, (T_cool = 300.15, ach = 0.5, gain = 200.0)) :
        case == 430 ? merge(base, (T_cool = 300.15, ach = 0.5, gain = 200.0, α = 0.6)) :
        case == 800 ? merge(base, (T_cool = 300.15, ach = 0.5, gain = 200.0, α = 0.6, mass = :high)) :
        throw(ArgumentError("a140_case: case $case is not a no-window in-depth case"))
    return merge(c, (case = case,))
end

"""
    a140_zone(spec; exterior = :combined, T0 = 293.15) -> (zone, irradiance keys, tilts)

The 8 × 6 × 2.7 m cell: four walls, roof and raised floor; with `spec.hc`
the south wall carries the 12 m² high-conductance element. The raised
floor sees the outdoor air and no sun (§5.2.1.5). Interior: the combined
coefficients of `A140_H`. Exterior:

  * `exterior = :combined` — `EnvelopeElement`s with the constant combined
    coefficients (sol-air form; long-wave exchange lumped with the air);
  * `exterior = :longwave` — `FacetElement`s: an `UrbanFacet` whose outer
    face has the convective coefficient of `A140_H`, emissivity 0.9 and
    the full long-wave balance against sky and ground (see `a140_run`).

Returns the zone and, for each envelope element, the key of its plane in
`A140_PLANES` (`nothing`: no sun) and its tilt (deg).
"""
function a140_zone(spec; exterior::Symbol = :combined, T0 = 293.15)
    wall = spec.mass === :low ? A140_LAYERS.lt_wall : A140_LAYERS.hw_wall
    floorL = spec.mass === :low ? A140_LAYERS.lt_floor : A140_LAYERS.hw_floor
    H = A140_H
    function ext(layers, A, h, α; name)
        exterior === :combined &&
            return EnvelopeElement(layers, A; exterior = :air, h_inside = h.in, h_outside = h.out,
                                   absorptance = α, T = T0, name = name)
        # (the floor gets no sun; its albedo is irrelevant but must be < 1)
        f = UrbanFacet(:wall, layers; T = T0, h_interior = h.in, albedo = α > 0 ? 1 - α : 0.5,
                       emissivity = A140_EMISSIVITY, h_convective = h.conv)
        return FacetElement(f, A; name = name)
    end
    els = AbstractZoneElement[]; keys = Union{Symbol, Nothing}[]; tilts = Float64[]
    A_south = spec.hc ? 21.6 - 12.0 : 21.6
    for (A, key) in ((A_south, :south), (21.6, :north), (16.2, :east), (16.2, :west))
        push!(els, ext(wall(), A, H.wall, spec.α; name = key)); push!(keys, key); push!(tilts, 90.0)
    end
    if spec.hc
        push!(els, ext(A140_LAYERS.hc_wall(), 12.0, H.hc, spec.α; name = :hc_element))
        push!(keys, :south); push!(tilts, 90.0)
    end
    push!(els, ext(A140_LAYERS.roof(), 48.0, H.roof, spec.α; name = :roof))
    push!(keys, :horizontal); push!(tilts, 0.0)
    push!(els, ext(floorL(), 48.0, H.floor, 0.0; name = :floor))
    push!(keys, nothing); push!(tilts, 180.0)
    zone = BuildingZone(; volume = 129.6, floor_area = 48.0, envelope = els,
                        infiltration_ach = 0.0, radiative_fraction = 0.6, T_air = T0,
                        p_ref = 83_500.0)
    # set the infiltration to the Standard's mass flow (Table 5-3 note f)
    zone.infiltration_ach = spec.ach * A140_RHOCP / zone.ρc_air
    return zone, keys, tilts
end

"""
    a140_run(spec, w, I; exterior = :combined, substeps = 6, warmup_days = 31) -> NamedTuple

One year of the case under the Standard 140 weather `w` with plane
irradiances `I` (from `a140_incident`), `substeps` zone steps per hour
(outdoor temperature interpolated linearly between the hourly values,
irradiance held over the hour), after a warm-up on the last
`warmup_days` of the year. With `exterior = :longwave` each outer face
receives the long-wave irradiance ``F_{sky} L_{sky} + (1 - F_{sky}) σ
T_{out}^4`` with ``F_{sky} = (1 + \\cos β)/2``, ``L_{sky}`` the weather
file's horizontal infrared radiation and the ground at the outdoor air
temperature (§5.2.1.8.2.2.1); the raised floor sees only ground.

Returns annual heating and sensible cooling (MWh), annual hourly-integrated
peaks (kW), the hourly loads (Wh) and the zone's energy-balance residual.
"""
function a140_run(spec, w, I; exterior::Symbol = :combined, substeps::Integer = 6,
                  warmup_days::Integer = 31)
    zone, keys, tilts = a140_zone(spec; exterior)
    hvac = IdealLoadsHVAC(; T_heat = spec.T_heat, T_cool = spec.T_cool, heating = :electric,
                          curves = nothing)
    N = length(w.T_db); dt = 3600.0 / substeps
    σ = 5.670374419e-8
    irr = zeros(length(keys))
    Fsky = [(1 + cosd(β)) / 2 for β in tilts]
    resid = Ref(0.0)
    function hour!(i)
        Tprev = 273.15 + w.T_db[i == 1 ? N : i - 1]; Tnow = 273.15 + w.T_db[i]
        for (j, k) in enumerate(keys)
            irr[j] = k === nothing ? 0.0 : I[k][i]
        end
        qh = 0.0; qc = 0.0
        for s in 1:substeps
            T_out = Tprev + (Tnow - Tprev) * (s - 0.5) / substeps
            sf = exterior === :combined ? nothing :
                 [SurfaceForcing(; sw_down = irr[j], T_air = T_out, p = w.p[i], U = 0.0,
                                 lw_down = Fsky[j] * w.IR_h[i] + (1 - Fsky[j]) * σ * T_out^4)
                  for j in eachindex(keys)]
            E0 = zone_heat_content(zone)
            b, h = step_zone!(zone, ZoneForcing(; T_out, Q_internal = spec.gain, irradiance = irr,
                                                surface = sf), dt, hvac)
            resid[] = max(resid[], abs(zone_heat_content(zone) - E0 -
                                       dt * (b.Q_exterior + b.Q_windows + b.Q_ventilation +
                                             b.Q_internal + b.Q_solar + b.Q_hvac)))
            qh += h.Q_heating * dt; qc += h.Q_cooling_sensible * dt
        end
        return qh, qc
    end
    for i in (N - 24 * warmup_days + 1):N
        hour!(i)
    end
    heat = zeros(N); cool = zeros(N)
    for i in 1:N
        heat[i], cool[i] = hour!(i)
    end
    return (case = spec.case, exterior = exterior, heating = sum(heat) / 3.6e9,
            cooling = sum(cool) / 3.6e9, peak_heating = maximum(heat) / 3.6e6,
            peak_cooling = maximum(cool) / 3.6e6, heat_hourly = heat ./ 3.6e3,
            cool_hourly = cool ./ 3.6e3, max_step_residual = resid[])
end
