# A saturated moist-adiabatic column for the cloudy-air N² tests (#239),
# shared by test/stability_functions.jl, test/stratified_sgs.jl
# and test/tke.jl. The includer loads Aeolus.

"""
    moist_adiabat_column!(θ, qv, qc; q_w = 0.012, T0 = 285.0, p0 = 1.0e5) -> p

Fill every storage level (halos included) of the `CellCenter` fields θ,
`qv`, `qc` with a saturated column whose temperature follows the moist
adiabat, ``dT/dz = −Γ_m(T, p)`` (`MoistThermodynamics.moist_adiabatic_lapse_rate`),
in hydrostatic balance with dry air, ``dp/dz = −g p/(R_d T)``, integrated
by RK4 in steps of at most 5 cm from `(T0, p0)` at the lowest storage
level. ``q_v = q_s(T, p)``, ``q_c = q_w − q_s`` (constant total water) and
``θ = T/Π(p)``. Returns the storage-indexed pressure vector.
"""
function moist_adiabat_column!(θ, qv, qc; q_w = 0.012, T0 = 285.0, p0 = 1.0e5)
    MT = Aeolus.MoistThermodynamics; c = MT.DEFAULT_CONSTANTS
    m = θ.mesh
    f(T, p) = (-MT.moist_adiabatic_lapse_rate(T, p, c), -c.grav * p / (MT.R_d(c) * T))
    nk = size(θ.data, 3)
    pv = zeros(nk)
    T, p, z = T0, p0, 0.0
    for k in 1:nk
        zt = (k - 1) * m.dz
        while z < zt - 1e-12
            h = min(0.05, zt - z)
            k1 = f(T, p); k2 = f(T + h/2 * k1[1], p + h/2 * k1[2])
            k3 = f(T + h/2 * k2[1], p + h/2 * k2[2]); k4 = f(T + h * k3[1], p + h * k3[2])
            T += h / 6 * (k1[1] + 2k2[1] + 2k3[1] + k4[1])
            p += h / 6 * (k1[2] + 2k2[2] + 2k3[2] + k4[2]); z += h
        end
        qs = MT.saturation_specific_humidity(T, p, c)
        q_w > qs || throw(ArgumentError("q_w = $q_w is below saturation ($qs) at level $k"))
        θ.data[:, :, k] .= T / MT.exner_function(p)
        qv.data[:, :, k] .= qs
        qc.data[:, :, k] .= q_w - qs
        pv[k] = p
    end
    return pv
end
