# Shared by the BuildingEnergy tests and validation runs
# (test/building_energy.jl, test/hvac.jl and their namesakes in
# validation/): the single-zone test boxes and an independent steady network
# solution for them (the reference both suites compare with).

using Aeolus.BuildingEnergy
using Aeolus.UrbanSurface

# Independent steady network solution for zones of EnvelopeElements (:air,
# :ground, :adiabatic) and InternalMass, with no irradiance. A radiative
# flux q on an inner face with outward resistance R_o (to T_e) splits:
# h R_o/(1 + h R_o) of it reaches the air; on an adiabatic face or internal
# mass, all of it.
function steady_air_temperature(z, T_out, Φ_int, Φ_sol, Φ_hvac)
    A_int = sum(el -> Aeolus.BuildingEnergy._exposed_area(el), Iterators.flatten((z.envelope, z.internal_mass)))
    q = (z.radiative_fraction * Φ_int + Φ_sol) / A_int
    H_w = z.window_U * z.window_area
    num = (H_w + ventilation_conductance(z)) * T_out + (1 - z.radiative_fraction) * Φ_int + Φ_hvac
    den = H_w + ventilation_conductance(z)
    for el in z.envelope
        if el.exterior === :adiabatic
            num += el.area * q
            continue
        end
        R_o = Aeolus.UrbanSurface.thermal_resistance(el.column.layers) + (el.exterior === :air ? 1 / el.h_outside : 0.0)
        T_e = el.exterior === :air ? T_out : el.T_ground
        U = 1 / (1 / el.h_inside + R_o)
        num += el.area * (U * T_e + q * el.h_inside * R_o / (1 + el.h_inside * R_o))
        den += el.area * U
    end
    num += sum((Aeolus.BuildingEnergy._exposed_area(el) * q for el in z.internal_mass); init = 0.0)
    return num / den
end

# An 8 m × 6 m × 2.7 m single-zone box (the plan dimensions of the ASHRAE 140
# case-600 shoebox, used only as a geometry; no BESTEST result is claimed).
function test_box(; heavy = true, θ = 0.5, T = 293.15, facets = false)
    wl = heavy ? [WallLayer(:brick, 0.24), WallLayer(:gypsum_plaster, 0.015)] :
                 [WallLayer(:plywood, 0.012), WallLayer(:glass_fiber_board, 0.10),
                  WallLayer(:gypsum_plaster, 0.012)]
    A_wall = 2 * (8 + 6) * 2.7 - 12
    walls = facets ?
        FacetElement(UrbanFacet(:wall, wl; T = T, T_interior = T, h_convective = 15.0, θ = θ), A_wall) :
        EnvelopeElement(wl, A_wall; absorptance = 0.6, T = T, θ = θ, name = :walls)
    roof = facets ?
        FacetElement(UrbanFacet(:roof, example_roof_layers(); T = T, T_interior = T, θ = θ), 48.0) :
        EnvelopeElement(example_roof_layers(), 48.0; h_inside = 10.0, T = T, θ = θ, name = :roof)
    floor = EnvelopeElement([WallLayer(:xps_insulation, 0.1), WallLayer(:concrete, 0.15)], 48.0;
                            exterior = :adiabatic, h_inside = 1 / 0.17, T = T, θ = θ, name = :floor)
    im = heavy ? [InternalMass([WallLayer(:brick, 0.12)], 20.0; sides = 2, T = T, θ = θ)] :
                 InternalMass{Float64}[]
    return BuildingZone(; volume = 8 * 6 * 2.7, floor_area = 48.0, envelope = [walls, roof, floor],
                        internal_mass = im, window_area = 12.0, window_U = 2.8,
                        infiltration_ach = 0.5, T_air = T)
end

# A massive 8 m × 6 m × 2.7 m box (brick walls, roof, insulated slab, a brick
# partition), as in the #31 tests.
function hvac_mass_box(; T = 297.15)
    wl = [WallLayer(:brick, 0.24), WallLayer(:gypsum_plaster, 0.015)]
    walls = EnvelopeElement(wl, 2 * (8 + 6) * 2.7 - 12; absorptance = 0.6, T = T, name = :walls)
    roof = EnvelopeElement(example_roof_layers(), 48.0; h_inside = 10.0, absorptance = 0.7, T = T,
                           name = :roof)
    floor = EnvelopeElement([WallLayer(:xps_insulation, 0.1), WallLayer(:concrete, 0.15)], 48.0;
                            exterior = :adiabatic, h_inside = 1 / 0.17, T = T, name = :floor)
    im = [InternalMass([WallLayer(:brick, 0.12)], 20.0; sides = 2, T = T)]
    return BuildingZone(; volume = 8 * 6 * 2.7, floor_area = 48.0, envelope = [walls, roof, floor],
                        internal_mass = im, window_area = 12.0, window_U = 2.8,
                        infiltration_ach = 0.5, T_air = T)
end
