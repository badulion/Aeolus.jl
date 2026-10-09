# =========================================================================
# The case tables of the observed stable-boundary-layer cases — P58-d
# (#194), CFG-d (#359), EX-b (#439)
#
#   GABLS2 / CASES-99  (Kansas, 22-24 Oct 1999)   examples/sbl/configs/gabls2.toml, cases99.toml
#   GABLS3             (Cabauw, 00-09 UTC 2 Jul 2006)   configs/gabls3.toml
#   GABLS4 stage 3     (Dome C, 11 Dec 2009, experiment 3 from 10 UTC)   configs/gabls4_exp3.toml
#
# The cases are configs of the `sbl` run type: every case constant lives
# in its TOML file, which points to the case table under
# examples/sbl/data/ (provenance in every CSV header). This file only
# reads those tables (`[case] data`), and holds the one case formula a
# table cannot carry (the GABLS2 surface temperature, the `gabls2`
# surface-temperature kind) and the residual-layer inversion height of
# the series output; it adds no physics. The reference data (envelopes,
# observations) are read by the analysis scripts of examples/sbl/.
#
# All the runs are dry: the Boussinesq core's SSP-RK3 (`step_rk3!`) does
# not advance `qv` and the SGS closures do not diffuse it, so the GABLS3
# humidity (q = 8.7-10 g/kg, virtual-temperature correction ~0.2 K over the
# column) is not carried. Stated as an exception on the theory page.
# =========================================================================

const SBL_OMEGA = 7.2921e-5                    # Earth's rotation rate (s⁻¹)

"`2Ω sin(latitude)` (s⁻¹)."
sbl_coriolis(lat_deg) = 2 * SBL_OMEGA * sind(lat_deg)

"""
    sbl_case_tables(path) -> Dict{String, Matrix{Float64}}

Read a case CSV (`examples/sbl/data/`) whose first column names the
table of each row (`wind`, `thermo`, `surface`, …): returns the numeric
columns of each table as a matrix (empty fields dropped from the right).
"""
function sbl_case_tables(path)
    tab = sbl_read_csv(path)
    out = Dict{String, Vector{Vector{Float64}}}()
    for r in tab.rows
        vals = Float64[x for x in r[2:end] if x isa Real]
        push!(get!(out, string(r[1]), Vector{Float64}[]), vals)
    end
    return Dict(k => permutedims(reduce(hcat, v)) for (k, v) in out)
end

# piecewise-linear interpolation, held at the ends
function _sbl_interp(x, y, xq)
    xq <= x[1] && return y[1]
    xq >= x[end] && return y[end]
    i = searchsortedlast(x, xq)
    return y[i] + (y[i+1] - y[i]) * (xq - x[i]) / (x[i+1] - x[i])
end

# =========================================================================
# GABLS2 / CASES-99 (Svensson et al. 2011; see the CSV header on provenance)
# =========================================================================

"""
    gabls2_surface_temperature_C(h) -> °C

The GABLS2 prescribed surface temperature at local time `h` (hours from
00 LT 22 October 1999), the piecewise formula of the case (CLUBB
`gabls2.F90`, identical in van Hooft's Basilisk set-up). The `sbl`
driver samples it for `[surface] temperature = {kind = "gabls2", …}`.
"""
function gabls2_surface_temperature_C(h)
    h <= 17.4 && return -10 - 25 * cos(0.22h + 0.2)
    h <= 30.0 && return -0.54h + 15.2
    h <= 41.9 && return -7 - 25 * cos(0.21h + 1.8)
    h <= 53.3 && return -0.37h + 18.0
    h <= 65.6 && return -4 - 25 * cos(0.22h + 2.5)
    return 4.4
end

"""
    sbl_inversion_height(z, θ; z_min = 300) -> m

The height of the capping inversion of a plane-mean profile `θ` at cell
centres `z` (uniform spacing): the face of the largest ``∂⟨θ⟩/∂z`` above
`z_min`, refined by the parabola through that gradient and its two
neighbours. With `z_min` above the nocturnal surface inversion it is the
top of the residual layer left by the previous day's mixed layer, which
large-scale subsidence lowers.
"""
function sbl_inversion_height(z, θ; z_min = 300.0)
    dz = z[2] - z[1]
    zf = [(z[k] + z[k+1]) / 2 for k in 1:length(z)-1]
    G = diff(θ) ./ dz
    ks = [k for k in eachindex(zf) if zf[k] >= z_min]
    k = ks[argmax(G[ks])]
    (k == 1 || k == length(G)) && return zf[k]
    a, b, c = G[k-1], G[k], G[k+1]
    den = a - 2b + c
    return den == 0 ? zf[k] : zf[k] + 0.5 * (a - c) / den * dz
end
