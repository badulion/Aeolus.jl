# The case and reference tables of the SBL tests (examples/sbl/data/) and
# the committed dynamic drivers of the SBL configs (examples/sbl/
# input/). The model reads only the drivers and the reference files its
# configs name ([checks.reference]); these readers are the tests' own, for
# the provenance of the data and for checking the drivers against the case
# tables they were made from (ONE-g, #456: the sbl run type, whose readers
# these were, is gone).

const SBL_DATA = joinpath(pkgdir(Aeolus), "examples", "sbl", "data")
const SBL_CONFIGS = joinpath(pkgdir(Aeolus), "examples", "sbl")

"""
    sbl_read_csv(path) -> (; header, columns, rows)

A case or reference-data CSV: the leading `#` lines are its provenance
header (verbatim), the first other line the column names, the rest the
rows (numbers parsed where possible).
"""
function sbl_read_csv(path)
    header = String[]; columns = String[]; rows = Vector{Any}[]
    for line in eachline(path)
        isempty(strip(line)) && continue
        if startswith(line, "#")
            push!(header, line)
        elseif isempty(columns)
            columns = strip.(split(line, ","))
        else
            push!(rows, [something(tryparse(Float64, strip(x)), strip(x)) for x in split(line, ",")])
        end
    end
    return (header = header, columns = columns, rows = rows)
end

"The value of column `col` in the row whose first column is `key`."
function sbl_csv_value(tab, key, col)
    j = findfirst(==(col), tab.columns)
    for r in tab.rows
        r[1] == key && return r[j]
    end
    throw(KeyError(key))
end

"The tables of a case CSV whose first column names each row's table: numeric matrices by name."
function sbl_case_tables(path)
    out = Dict{String,Vector{Vector{Float64}}}()
    for r in sbl_read_csv(path).rows
        push!(get!(out, string(r[1]), Vector{Float64}[]), Float64[x for x in r[2:end] if x isa Real])
    end
    return Dict(k => permutedims(reduce(hcat, v)) for (k, v) in out)
end

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

"""
    gabls2_surface_temperature_C(h)

The GABLS2 prescribed surface temperature (°C) at local time `h` (hours
from 00 LT 22 October 1999): the piecewise formula of the case (CLUBB
`gabls2.F90`), which the committed driver `gabls2_dynamic.nc` samples.
"""
function gabls2_surface_temperature_C(h)
    h <= 17.4 && return -10 - 25 * cos(0.22h + 0.2)
    h <= 30.0 && return -0.54h + 15.2
    h <= 41.9 && return -7 - 25 * cos(0.21h + 1.8)
    h <= 53.3 && return -0.37h + 18.0
    h <= 65.6 && return -4 - 25 * cos(0.22h + 2.5)
    return 4.4
end
