# =========================================================================
# Observed stable-boundary-layer cases on the shared SBL driver — P58-d (#194)
#
#   GABLS2 / CASES-99  (Kansas, 22-24 Oct 1999)   gabls2_case
#   GABLS3             (Cabauw, 00-09 UTC 2 Jul 2006)   gabls3_case
#   GABLS4 stage 3     (Dome C, 11 Dec 2009, experiment 3 from 10 UTC)   gabls4_case
#
# Each builder reads its CSV under examples/sbl/data/ (provenance in every
# header) and returns an `SBLCase` of examples/sbl/driver.jl: the forcing is
# LargeScaleForcing's (GeostrophicPressureGradient with U_g(z, t) tables,
# AdvectiveTendencies, SurfaceTemperatureSeries), the surface fluxes are
# SurfaceExchange.solve_surface_layer, the stress is WallModels.MOSTWallLaw
# and the closure is Operators.StabilityCorrected (P55-e) or, with
# `sgs_stability = :none`, the uncorrected base closure. This file defines
# case constants and data readers only; it adds no physics.
#
# All three runs are dry: the Boussinesq core's SSP-RK3 (`step_rk3!`) does
# not advance `qv` and the SGS closures do not diffuse it, so the GABLS3
# humidity (q = 8.7-10 g/kg, virtual-temperature correction ~0.2 K over the
# column) is not carried. Stated as an exception on the theory page.
# =========================================================================

include(joinpath(@__DIR__, "driver.jl"))

const SBL_DATA = joinpath(@__DIR__, "data")
const SBL_OMEGA = 7.2921e-5                    # Earth's rotation rate (s⁻¹)

"`2Ω sin(latitude)` (s⁻¹)."
sbl_coriolis(lat_deg) = 2 * SBL_OMEGA * sind(lat_deg)

"""
    sbl_case_tables(path) -> Dict{String, Matrix{Float64}}

Read a case CSV of `examples/sbl/data/` whose first column names the
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
# GABLS3 (Basu 2008 case description; Bosveld et al. 2014a,b)
# =========================================================================

const GABLS3_LATITUDE = 51.9711
const GABLS3_Z0M = 0.15
const GABLS3_Z_T = 0.25             # height of the prescribed θ (and q): the scalar reference height
const GABLS3_STABILITY = BusingerDyerLogLinear(β = 5.0)   # stable ψ_M = ψ_H = -5 z/L (case description)

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

"""
    gabls3_forcing(d = gabls3_data()) -> (U_g, V_g, adv, T_s)

The GABLS3 forcing as LargeScaleForcing objects: `U_g`, `V_g` are
height-time `ForcingTable`s, linear in time between the surface values of
the table and linear in height to (−2, 2) m s⁻¹ at 2000 m (exact as a
bilinear table on z = 0, 2000 m); `adv` the `AdvectiveTendencies` of u, v
and θ, constant above 200 m and linear to zero at the surface (z = 0,
200 m); `T_s` the 0.25 m potential temperature as a
`SurfaceTemperatureSeries`.
"""
function gabls3_forcing(d = gabls3_data())
    g = d.geostrophic
    tg = g[:, 1]
    Ug = ForcingTable([0.0, 2000.0], tg, permutedims(hcat(g[:, 2], fill(-2.0, length(tg)))))
    Vg = ForcingTable([0.0, 2000.0], tg, permutedims(hcat(g[:, 3], fill(2.0, length(tg)))))
    a = d.advection; ta = a[:, 1]
    tab(col) = ForcingTable([0.0, 200.0], ta, permutedims(hcat(zeros(length(ta)), a[:, col])))
    adv = AdvectiveTendencies(; u = tab(2), v = tab(3), θ = tab(4))
    Ts = SurfaceTemperatureSeries(d.surface[:, 1], d.surface[:, 3])
    return (U_g = Ug, V_g = Vg, adv = adv, T_s = Ts)
end

"""
    gabls3_initial_profiles(d = gabls3_data()) -> (θ, u, v)

Functions of z for the 00 UTC state. Above 10 m: the tabulated sounding,
linearly interpolated. Below 10 m (the table starts at 10 m): θ linear
between the prescribed 0.25 m value and the 10 m value; the wind
logarithmic from zero at z0 = 0.15 m to the 10 m vector.
"""
function gabls3_initial_profiles(d = gabls3_data())
    w = d.wind; th = d.thermo
    θs0 = d.surface[1, 3]
    θ(z) = z >= th[1, 1] ? _sbl_interp(th[:, 1], th[:, 3], z) :
           θs0 + (th[1, 3] - θs0) * (max(z, GABLS3_Z_T) - GABLS3_Z_T) / (th[1, 1] - GABLS3_Z_T)
    logfac(z) = z >= w[1, 1] ? 1.0 : log(max(z, GABLS3_Z0M) / GABLS3_Z0M) / log(w[1, 1] / GABLS3_Z0M)
    u(z) = z >= w[1, 1] ? _sbl_interp(w[:, 1], w[:, 2], z) : w[1, 2] * logfac(z)
    v(z) = z >= w[1, 1] ? _sbl_interp(w[:, 1], w[:, 3], z) : w[1, 3] * logfac(z)
    return (θ = θ, u = u, v = v)
end

"""
    gabls3_case(; N = (128, 128, 128), extent = (800, 800, 800),
                sgs_stability = :brown94, base = StaticSmagorinsky(0.1),
                sponge_z = 600, sponge_α = 0.01, seed = 1) -> SBLCase

The GABLS3 LES case (Basu 2008 case description; Bosveld et al. 2014a,b)
from 00 UTC: Cabauw, f = 2Ω sin 51.97°, z0 = 0.15 m, the surface heat flux
from MOST between the prescribed 0.25 m θ and the first level
(`z0h = 0.25` in `solve_surface_layer` gives exactly the case formula
``w'θ' = u_* κ (θ_{0.25} - θ_{z_1}) / [\\ln(z_1/0.25) - Ψ_H(z_1/L) +
Ψ_H(0.25/L)]``), stable ψ_M = ψ_H = −5z/L (`BusingerDyerLogLinear(β = 5)`;
the morning's unstable branch is that family's Businger–Dyer, γ = 16),
time-height geostrophic wind and advective tendencies of u, v, θ, the
perturbations u, v ~ N(0, 0.2(1 − z/200)²) and θ ~ N(0, 0.1) below 200 m,
and a sponge from `sponge_z` damping only the departures from the plane
mean (the mean wind there evolves under the prescribed advection). The
reference resolution is 6.25 m (128³); the reduced CI test uses
16 × 16 × 48 on 400 × 400 × 400 m with the sponge from 300 m.
"""
function gabls3_case(; N = (128, 128, 128), extent = (800.0, 800.0, 800.0),
                     sgs_stability::Symbol = :brown94, base = StaticSmagorinsky(0.1),
                     sponge_z = 600.0, sponge_α = 0.01, seed = 1)
    d = gabls3_data()
    F = gabls3_forcing(d)
    P = gabls3_initial_profiles(d)
    pert(rng, x, y, z) = z <= 200 ?
        (sqrt(0.2) * (1 - z / 200) * randn(rng), sqrt(0.2) * (1 - z / 200) * randn(rng),
         sqrt(0.1) * randn(rng)) : (0.0, 0.0, 0.0)
    ρ = d.surface[1, 2] / (287.05 * d.surface[1, 3])         # ≈ 1.23 kg m⁻³ at the surface
    return sbl_case(; N = N, extent = extent, U_g = F.U_g, V_g = F.V_g,
                    f = sbl_coriolis(GABLS3_LATITUDE), z0m = GABLS3_Z0M, z0h = GABLS3_Z_T,
                    θ₀ = 290.0, T_s = F.T_s, base = base, sgs_stability = sgs_stability,
                    stability = GABLS3_STABILITY, sponge_z = sponge_z, sponge_α = sponge_α,
                    seed = seed, extra_forcing = (F.adv,), θ_init = P.θ, u_init = P.u,
                    v_init = P.v, perturb = pert, t_start = 0.0, sponge_target = :mean,
                    ρcp = ρ * 1005)
end

"""
    gabls3_envelope() -> table

The digitised LES-ensemble envelope at 03-04 UTC
(`examples/sbl/data/gabls3_envelope.csv`; read values with
`sbl_csv_value(tab, "jet_height", "low")`).
"""
gabls3_envelope() = sbl_read_csv(joinpath(SBL_DATA, "gabls3_envelope.csv"))

# =========================================================================
# GABLS4 stage 3, experiment 3 (Couvreux et al. 2020)
# =========================================================================

const GABLS4_LATITUDE = -75.10
const GABLS4_P_SFC = 65100.0
const GABLS4_EXNER_EXP = 0.286          # (1000/p)^0.286, as the jax-alfa GABLS4 set-up

"""
    gabls4_case(; N = (64, 64, 64), extent = (128, 128, 128), experiment = 3,
                sgs_stability = :brown94, base = StaticSmagorinsky(0.1),
                sponge_frac = 0.75, sponge_α = 0.01, seed = 1) -> SBLCase

GABLS4 (Couvreux et al. 2020, Dome C): dry, f = 2Ω sin(−75.1°) (southern
hemisphere), U_g = (1.25, 4.5) m s⁻¹ constant, the surface temperature of
the official case file converted to θ with p_s = 651 hPa. `experiment = 3`
(default) starts at 10 UTC from the experiment-2 ensemble mean (Table 5)
with z0m = 1e-3 m, z0h = 1e-4 m and the log-linear surface functions
β_m = 4.8, β_h = 7.8 that experiment 3 prescribed for its surface-flux
tests. Six of the seven experiment-3 LES ran at 1 m isotropic on
256 × 256 × 80 m to 512 × 512 × 150 m; CSIRO-LES ran at Δx = 3.9 m,
Δz = 1.17 m (Couvreux et al. 2020, Table 2).
`experiment = 1` starts at 00 UTC with z0m = 1e-2 m, z0h = 1e-3 m.
Perturbations: u, v, θ uniform ±0.1 (m s⁻¹, K) below 50 m.
"""
function gabls4_case(; N = (64, 64, 64), extent = (128.0, 128.0, 128.0), experiment::Int = 3,
                     sgs_stability::Symbol = :brown94, base = StaticSmagorinsky(0.1),
                     sponge_frac = 0.75, sponge_α = 0.01, seed = 1)
    experiment in (1, 3) || throw(ArgumentError("gabls4_case: experiment must be 1 or 3"))
    T = sbl_case_tables(joinpath(SBL_DATA, "gabls4_case.csv"))
    p = experiment == 3 ? T["exp3"] : T["exp1"]
    s = T["surface"]
    θs = s[:, 2] .* (1e5 / GABLS4_P_SFC)^GABLS4_EXNER_EXP
    z = p[:, 1]
    θ(zq) = _sbl_interp(z, p[:, 2], zq)
    u(zq) = _sbl_interp(z, p[:, 3], zq)
    v(zq) = _sbl_interp(z, p[:, 4], zq)
    pert(rng, x, y, zq) = zq < 50 ? (0.1 * (2rand(rng) - 1), 0.1 * (2rand(rng) - 1),
                                     0.1 * (2rand(rng) - 1)) : (0.0, 0.0, 0.0)
    z0m, z0h = experiment == 3 ? (1e-3, 1e-4) : (1e-2, 1e-3)
    ρ = GABLS4_P_SFC / (287.05 * 238.0)
    return sbl_case(; N = N, extent = extent, U_g = 1.25, V_g = 4.5,
                    f = sbl_coriolis(GABLS4_LATITUDE), z0m = z0m, z0h = z0h, θ₀ = 275.0,
                    T_s = SurfaceTemperatureSeries(s[:, 1], θs), base = base,
                    sgs_stability = sgs_stability, stability = SBL_GABLS1_STABILITY,
                    sponge_z = sponge_frac * extent[3], sponge_α = sponge_α, seed = seed,
                    θ_init = θ, u_init = u, v_init = v, perturb = pert,
                    t_start = experiment == 3 ? 36000.0 : 0.0, ρcp = ρ * 1005)
end

"The GABLS4 Table 4 values with each run's Table 2 grid and the envelopes (`examples/sbl/data/gabls4_reference.csv`)."
gabls4_reference() = sbl_read_csv(joinpath(SBL_DATA, "gabls4_reference.csv"))

# =========================================================================
# GABLS2 / CASES-99 (Svensson et al. 2011; see the CSV header on provenance)
# =========================================================================

"""
    gabls2_surface_temperature_C(h) -> °C

The GABLS2 prescribed surface temperature at local time `h` (hours from
00 LT 22 October 1999), the piecewise formula of the case (CLUBB
`gabls2.F90`, identical in van Hooft's Basilisk set-up).
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
    gabls2_case(; N = (48, 48, 80), extent = (2400, 2400, 1600),
                sgs_stability = :brown94, base = StaticSmagorinsky(0.1),
                sponge_frac = 0.75, sponge_α = 0.01, seed = 1) -> SBLCase

GABLS2 / CASES-99 from 16 LT 22 October 1999: constant U_g = (3, −9) m s⁻¹,
f = 8.898e-5 s⁻¹, z0m = 0.03 m, z0h = 0.003 m, the CLUBB θ sounding, the
prescribed surface temperature (sampled every minute into a
`SurfaceTemperatureSeries`; converted to θ with p_s = 972 hPa), the
log-linear surface functions with β = 5. Perturbations ±0.1 K below 200 m
and ±0.5 m s⁻¹ in u, v below 200 m (the case was spun up from rest in the
afternoon). No subsidence (see `examples/sbl/data/gabls2_case.csv`).
"""
function gabls2_case(; N = (48, 48, 80), extent = (2400.0, 2400.0, 1600.0),
                     sgs_stability::Symbol = :brown94, base = StaticSmagorinsky(0.1),
                     sponge_frac = 0.75, sponge_α = 0.01, seed = 1, t_end = 64 * 3600.0)
    T = sbl_case_tables(joinpath(SBL_DATA, "gabls2_case.csv"))
    th = T["theta"]
    tab = sbl_read_csv(joinpath(SBL_DATA, "gabls2_case.csv"))
    cst(k) = only(r[3] for r in tab.rows if r[1] == "constant" && r[2] == k)
    Ug, Vg, f = cst("U_g"), cst("V_g"), cst("f")
    z0m, z0h, ps, t0 = cst("z0m"), cst("z0h"), cst("p_sfc"), cst("t_start")
    ts = collect(t0:60.0:t_end)
    θs = [(gabls2_surface_temperature_C(t / 3600) + 273.15) * (1e5 / ps)^0.286 for t in ts]
    pert(rng, x, y, z) = z < 200 ? (0.5 * (2rand(rng) - 1), 0.5 * (2rand(rng) - 1),
                                    0.1 * (2rand(rng) - 1)) : (0.0, 0.0, 0.0)
    ρ = ps / (287.05 * 285.0)
    return sbl_case(; N = N, extent = extent, U_g = Ug, V_g = Vg, f = f, z0m = z0m, z0h = z0h,
                    θ₀ = 288.0, T_s = SurfaceTemperatureSeries(ts, θs), base = base,
                    sgs_stability = sgs_stability, stability = BusingerDyerLogLinear(β = 5.0),
                    sponge_z = sponge_frac * extent[3], sponge_α = sponge_α, seed = seed,
                    θ_init = z -> _sbl_interp(th[:, 1], th[:, 2], z), perturb = pert,
                    t_start = t0, ρcp = ρ * 1005)
end

"The GABLS2 reference values (`examples/sbl/data/gabls2_reference.csv`)."
gabls2_reference() = sbl_read_csv(joinpath(SBL_DATA, "gabls2_reference.csv"))

"The Sun et al. (2012) threshold wind speeds (`examples/sbl/data/cases99_sun2012_threshold.csv`)."
function sun2012_threshold()
    t = sbl_read_csv(joinpath(SBL_DATA, "cases99_sun2012_threshold.csv"))
    return (z = [r[1] for r in t.rows], V_s = [r[2] for r in t.rows], header = t.header)
end
