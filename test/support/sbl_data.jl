# The case and reference tables of the SBL tests (examples/sbl/data/),
# read with the run type's readers (`sbl_read_csv`, `sbl_case_tables` of
# Aeolus.StableBoundaryLayer). Moved from examples/sbl/cases.jl by EX-b
# (#439): the run type in src reads only the table a config names; these
# named readers are for the tests.

using Aeolus.StableBoundaryLayer: sbl_read_csv, sbl_case_tables

const SBL_DATA = joinpath(pkgdir(Aeolus), "examples", "sbl", "data")

"""
    gabls3_data() -> NamedTuple

The GABLS3 LES case tables (`examples/sbl/data/gabls3_case.csv`): initial
`wind` (z, u, v), `thermo` (z, p, θ, q), the 0.25 m `surface` series
(t, p, θ, q), the surface `geostrophic` wind (t, U_g, V_g) and the
`advection` tendencies (t, u, v, θ, q above 200 m), times in s from
00 UTC 2 July 2006.
"""
function gabls3_data()
    T = sbl_case_tables(joinpath(SBL_DATA, "gabls3_case.csv"))
    return (wind = T["wind"], thermo = T["thermo"], surface = T["surface"],
            geostrophic = T["geostrophic"], advection = T["advection"])
end

"The digitised GABLS3 LES-ensemble envelope at 03-04 UTC (`gabls3_envelope.csv`)."
gabls3_envelope() = sbl_read_csv(joinpath(SBL_DATA, "gabls3_envelope.csv"))

"The GABLS4 Table 4 values with each run's Table 2 grid and the envelopes (`gabls4_reference.csv`)."
gabls4_reference() = sbl_read_csv(joinpath(SBL_DATA, "gabls4_reference.csv"))

"The GABLS2 reference values (`gabls2_reference.csv`)."
gabls2_reference() = sbl_read_csv(joinpath(SBL_DATA, "gabls2_reference.csv"))
