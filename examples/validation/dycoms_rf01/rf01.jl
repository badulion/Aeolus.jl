# DYCOMS-II RF01 single column (Stevens et al. 2005): the sounding, the
# saturation adjustment, the column for the longwave solve and the
# intercomparison's prescribed flux. Shared by the `column` driver
# (examples/validation/column_driver.jl, case = "dycoms_rf01") and, through
# it, examples/validation/dycoms_rf01_radiation.jl. Moved here from that
# script (CFG-e, #360); the constants are the fields of `p`
# (RF01_DEFAULTS, or the [dycoms_rf01] section of a config).

using Aeolus

const CP    = 1004.6      # J/kg/K
const RD    = 287.05      # J/kg/K
const RV    = 461.5       # J/kg/K
const LV    = 2.5e6       # J/kg
const GRAV  = 9.80665     # m/s2
const P00   = 1.0e5       # Pa

"""
The RF01 case constants: the initial sounding (Stevens et al. 2005,
Table 1), the prescribed-radiation constants of the intercomparison
specification, and the vertical grid of the column.
"""
const RF01_DEFAULTS = (zi = 840.0,        # m, inversion height
                       psfc = 101780.0,   # Pa
                       sst = 292.5,       # K
                       qt_bl = 9.0e-3,    # kg/kg, total water below inversion
                       qt_ft = 1.5e-3,    # kg/kg, free troposphere
                       thl_bl = 289.0,    # K, liquid-water potential temperature below
                       reff = 10.0,       # um, droplet effective radius (RF01 uses ~10)
                       f0 = 70.0,         # W/m2
                       f1 = 22.0,         # W/m2
                       kappa = 85.0,      # m2/kg
                       dz_fine = 5.0, z_les = 1500.0, n_coarse = 40, z_top = 40_000.0)

"Saturation vapour pressure over liquid water, Bolton (1980), Pa."
esat(T) = 611.2 * exp(17.67 * (T - 273.15) / (T - 29.65))

"Saturation specific humidity at temperature T and pressure p."
qsat(T, p) = (eps = RD / RV; e = esat(T); eps * e / max(p - (1 - eps) * e, 1.0))

"""
Saturation adjustment: given liquid-water potential temperature `thl`,
total water `qt` and pressure `p`, return `(T, ql)`.

theta_l = theta - (Lv/cp) (P00/p)^(Rd/cp) q_l, so T satisfies

    f(T) = T - thl*exner - (Lv/cp) * max(0, qt - qsat(T,p)) = 0.

NOTE: the obvious fixed-point iteration T <- thl*exner + (Lv/cp)*ql(T)
does NOT converge here — it oscillates, because condensing enough water
to raise T also raises qsat past qt, which un-condenses all of it, which
drops T back. (Measured on the RF01 sounding at 840 m: 281.9 -> 285.2 ->
281.9 -> ... forever, so the routine silently returns ql = 0 and the
cloud never forms.) Bisection on f is used instead: f is continuous and
monotonically increasing in T, and the bracket below is guaranteed,
since

    f(T_lo) = -(Lv/cp) * ql(T_lo)  <= 0     with T_lo = thl*exner
    f(T_hi) =  (Lv/cp) * (qt - ql(T_hi)) >= 0   with T_hi = T_lo + Lv*qt/cp

because 0 <= ql <= qt everywhere. Note that `Tdry` below is a *parameter*
of f and must not be aliased to the moving bracket endpoint — doing so
makes the function change underneath the solver, which converges to a
plausible but wrong answer (0.34 g/kg instead of 0.47 at cloud top).
"""
function saturation_adjust(thl, qt, p)
    exner = (p / P00)^(RD / CP)
    Tdry = thl * exner                    # the unsaturated temperature
    # `Tdry` must stay fixed: it is a parameter of f, not a bracket end.
    f(T) = T - Tdry - (LV / CP) * max(0.0, qt - qsat(T, p))
    f(Tdry) >= 0 && return Tdry, 0.0      # subsaturated: nothing to condense
    a, b = Tdry, Tdry + (LV / CP) * qt    # bracket: f(a) < 0 <= f(b)
    for _ in 1:80
        m = 0.5 * (a + b)
        f(m) < 0 ? (a = m) : (b = m)
    end
    T = 0.5 * (a + b)
    return T, max(0.0, qt - qsat(T, p))
end

"""
Build the RF01 column: a finely resolved boundary layer from the surface
to `z_les`, then a coarser standard-atmosphere extension to `z_top` so
the longwave solve sees a real top-of-atmosphere rather than a lid at
1.5 km. Returns the column plus the height arrays for diagnostics.
"""
function rf01_column(p = RF01_DEFAULTS; dz_fine = p.dz_fine, z_les = p.z_les,
                       n_coarse = p.n_coarse, z_top = p.z_top)
    # --- fine LES layer --------------------------------------------------
    nf = round(Int, z_les / dz_fine)
    z_lev_f = collect(0.0:dz_fine:z_les)

    # --- coarse extension, geometrically stretched ------------------------
    r = (z_top / z_les)^(1 / n_coarse)
    z_lev_c = [z_les * r^k for k in 1:n_coarse]
    z_lev = vcat(z_lev_f, z_lev_c)
    nz = length(z_lev) - 1
    z_lay = 0.5 .* (z_lev[1:end-1] .+ z_lev[2:end])
    dz = diff(z_lev)

    # --- hydrostatic pressure, integrated upward through the moist BL ----
    p_lev = zeros(nz + 1); p_lev[1] = p.psfc
    p_lay = zeros(nz); T_lay = zeros(nz); ql = zeros(nz)
    q_lay = zeros(nz); ρ = zeros(nz)

    for k in 1:nz
        z = z_lay[k]
        # RF01 sounding
        if z < p.zi
            thl = p.thl_bl
            qt  = p.qt_bl
        else
            thl = 297.5 + (z - p.zi)^(1/3)
            qt  = p.qt_ft
        end
        # Above the LES domain, relax to a standard mid-latitude profile:
        # keeping the RF01 free-tropospheric formula to 40 km would give a
        # physically absurd stratosphere and corrupt the OLR.
        if z > z_les
            # US-standard-like: 6.5 K/km to 11 km, isothermal to 20 km,
            # then warming; humidity decaying exponentially.
            T_std = z < 11_000 ? 288.15 - 6.5e-3 * z :
                    z < 20_000 ? 216.65 :
                    z < 32_000 ? 216.65 + 1.0e-3 * (z - 20_000) :
                                 228.65 + 2.8e-3 * (z - 32_000)
            p_guess = p_lev[k]
            qt = max(3e-6, p.qt_ft * exp(-(z - z_les) / 2500.0))
            T_lay[k] = T_std
            ql[k] = 0.0
            q_lay[k] = qt
        else
            # provisional pressure from the level below
            p_guess = p_lev[k] * exp(-GRAV * (z - z_lev[k]) / (RD * 290.0))
            T, qlk = saturation_adjust(thl, qt, p_guess)
            T_lay[k] = T
            ql[k] = qlk
            q_lay[k] = qt - qlk        # vapour
        end
        p_lay[k] = p_lev[k] * exp(-GRAV * (z_lay[k] - z_lev[k]) /
                                  (RD * T_lay[k] * (1 + 0.608 * q_lay[k])))
        p_lev[k+1] = p_lev[k] * exp(-GRAV * dz[k] /
                                    (RD * T_lay[k] * (1 + 0.608 * q_lay[k])))
        ρ[k] = p_lay[k] / (RD * T_lay[k] * (1 + 0.608 * q_lay[k]))
    end

    # level temperatures by interpolation (needed for the linear-in-tau source)
    T_lev = zeros(nz + 1)
    T_lev[1] = p.sst
    for k in 2:nz
        T_lev[k] = 0.5 * (T_lay[k-1] + T_lay[k])
    end
    T_lev[nz+1] = T_lay[nz]

    # --- gas volume mixing ratios ----------------------------------------
    vmr = Dict{Symbol, Vector{Float64}}(
        :h2o => q_lay .* (RV / RD),              # mass -> volume mixing ratio
        :co2 => fill(420e-6, nz),
        :ch4 => fill(1.9e-6, nz),
        :n2o => fill(0.33e-6, nz),
        :o2  => fill(0.209, nz),
        :o3  => [z_lay[k] < 10_000 ? 4e-8 :
                 z_lay[k] < 35_000 ? 4e-8 + 7e-6 * (z_lay[k] - 10_000) / 25_000 :
                 7e-6 * exp(-(z_lay[k] - 35_000) / 8000) for k in 1:nz],
    )

    col = AtmosphericColumn(Float64, nz;
        p_lay = p_lay, T_lay = T_lay, p_lev = p_lev, T_lev = T_lev,
        dz = dz, ρ = ρ, vmr = vmr, ql = ql, qi = zeros(nz),
        r_eff_liq = fill(p.reff, nz), r_eff_ice = fill(30.0, nz),
        T_sfc = p.sst)

    return col, z_lay, z_lev, nf
end

"""
The DYCOMS-II prescribed net longwave flux, radiation terms only
(the subsidence-warming term is excluded — it is not radiation).
"""
function prescribed_flux(col, z_lev, p = RF01_DEFAULTS)
    nz = length(col.ql)
    # Q(0,z) and Q(z,top): cumulative kappa * rho * q_l
    cum = zeros(nz + 1)
    for k in 1:nz
        cum[k+1] = cum[k] + p.kappa * col.ρ[k] * col.ql[k] * col.dz[k]
    end
    total = cum[end]
    return [p.f0 * exp(-(total - cum[k])) + p.f1 * exp(-cum[k])
            for k in 1:nz+1]
end


"""
    rf01_radiation(p = RF01_DEFAULTS; ngpt_per_band = 4, sfc_emissivity = 0.98) -> NamedTuple

Build the RF01 column and solve the longwave (correlated-k gas optics,
Mie cloud optics, two-stream with the linear-in-τ source), clear sky and
all sky: `col`, `z_lay`, `z_lev`, `nf`, the net upward fluxes `Fup_all`,
`Fup_clear`, the prescribed `Fspec`, and the heating rate `Kday` (K/day).
"""
function rf01_radiation(p = RF01_DEFAULTS; ngpt_per_band = 4, sfc_emissivity = 0.98)
    col, z_lay, z_lev, nf = rf01_column(p)
    nz = length(col.ql)
    lw     = longwave_bands(Float64; ngpt_per_band = ngpt_per_band)
    gas    = correlated_k_optics(lw)
    cloud  = mie_cloud_optics(lw)
    solver = two_stream_solver(lw; lw_variant = Quadrature(),
                                   linear_lw_source = true)

    # clear-sky reference (gas only)
    o_clear = allocate_optics(Float64, (nz,), lw)
    compute_optics!(o_clear, gas, col)
    f_clear = allocate_fluxes(Float64, (), nz + 1)
    solve_lw!(f_clear, solver, o_clear, col; sfc_emissivity = sfc_emissivity)

    # all-sky (gas + cloud)
    o_all = allocate_optics(Float64, (nz,), lw)
    compute_optics!(o_all, gas, col)
    o_cld = allocate_optics(Float64, (nz,), lw)
    compute_optics!(o_cld, cloud, col)
    combine_optics!(o_all, o_cld)
    f_all = allocate_fluxes(Float64, (), nz + 1)
    solve_lw!(f_all, solver, o_all, col; sfc_emissivity = sfc_emissivity)

    # net UPWARD flux (the DYCOMS F_rad convention is positive upward)
    Fup_all   = [f_all.flux_up[k] - f_all.flux_dn[k] for k in 1:nz+1]
    Fup_clear = [f_clear.flux_up[k] - f_clear.flux_dn[k] for k in 1:nz+1]
    Fspec     = prescribed_flux(col, z_lev, p)

    dθdt = zeros(nz)
    radiative_heating!(dθdt, f_all, col)
    return (; col, z_lay, z_lev, nf, Fup_all, Fup_clear, Fspec, Kday = dθdt .* 86400)
end
