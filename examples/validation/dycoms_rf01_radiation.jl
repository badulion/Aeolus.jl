# =========================================================================
# DYCOMS-II RF01 — cloud-top radiative cooling
#
# Validation of the Era III radiation stack (Phase 32 correlated-k gas
# optics + Phase 33 two-stream with cloud optics) against the radiation
# specification of the DYCOMS-II RF01 LES intercomparison.
#
#   Stevens, B. et al. (2005). Evaluation of large-eddy simulations via
#   observations of nocturnal marine stratocumulus. Mon. Wea. Rev., 133,
#   1443-1462.
#
# WHY THIS CASE
#
# RF01 is nocturnal marine stratocumulus off the California coast. There
# is no sun, no surface heating worth speaking of, and almost no wind
# shear. The entire turbulent circulation of the boundary layer is driven
# by ONE term: longwave radiative cooling in the top few tens of metres
# of the cloud. If a radiation scheme gets that term wrong, the LES it
# feeds produces the wrong boundary layer no matter how good its
# dynamics are. That makes RF01 the sharpest available single-column
# test of a longwave scheme intended for cloudy boundary layers.
#
# The intercomparison does not ask participants to run full radiative
# transfer. It PRESCRIBES the net longwave flux analytically:
#
#   F_rad(z) = F0 exp(-Q(z, ztop)) + F1 exp(-Q(0, z))
#              + rho_i cp D alpha_z [ (z-zi)^(4/3)/4 + zi (z-zi)^(1/3) ]
#
# with F0 = 70 W/m2, F1 = 22 W/m2, kappa = 85 m2/kg, and
# Q(z1,z2) = kappa * integral of rho*q_l between the two heights. The
# third term applies only above the inversion and represents subsidence
# warming, not radiation, so it is excluded from the comparison here.
#
# That prescription encodes what the LES community agreed the answer is:
# the cloud radiates to space essentially as a black body, so the net
# upward flux jumps by ~F0 = 70 W/m2 across cloud top, concentrated in
# the topmost optical depth. THAT is the number this script checks our
# physically-derived solution against. We compute it from first
# principles - real gas optics, real cloud optics, real two-stream -
# and see whether we land on the prescription the intercomparison
# arrived at empirically.
#
# Run:  julia --project=. examples/validation/dycoms_rf01_radiation.jl
# =========================================================================

using Printf
using Aeolus

const CP    = 1004.6      # J/kg/K
const RD    = 287.05      # J/kg/K
const RV    = 461.5       # J/kg/K
const LV    = 2.5e6       # J/kg
const GRAV  = 9.80665     # m/s2
const P00   = 1.0e5       # Pa

# --- DYCOMS-II RF01 initial sounding (Stevens et al. 2005, Table 1) -----
const ZI      = 840.0     # m, inversion height
const PSFC    = 101780.0  # Pa
const SST     = 292.5     # K
const QT_BL   = 9.0e-3    # kg/kg, total water below inversion
const QT_FT   = 1.5e-3    # kg/kg, free troposphere
const THL_BL  = 289.0     # K, liquid-water potential temperature below
const REFF    = 10.0      # um, droplet effective radius (RF01 uses ~10)

# Prescribed-radiation constants from the intercomparison specification
const F0_SPEC = 70.0      # W/m2
const F1_SPEC = 22.0      # W/m2
const KAPPA   = 85.0      # m2/kg

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
function rf01_column(; dz_fine = 5.0, z_les = 1500.0,
                       n_coarse = 40, z_top = 40_000.0)
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
    p_lev = zeros(nz + 1); p_lev[1] = PSFC
    p_lay = zeros(nz); T_lay = zeros(nz); ql = zeros(nz)
    q_lay = zeros(nz); ρ = zeros(nz)

    for k in 1:nz
        z = z_lay[k]
        # RF01 sounding
        if z < ZI
            thl = THL_BL
            qt  = QT_BL
        else
            thl = 297.5 + (z - ZI)^(1/3)
            qt  = QT_FT
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
            qt = max(3e-6, QT_FT * exp(-(z - z_les) / 2500.0))
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
    T_lev[1] = SST
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
        r_eff_liq = fill(REFF, nz), r_eff_ice = fill(30.0, nz),
        T_sfc = SST)

    return col, z_lay, z_lev, nf
end

"""
The DYCOMS-II prescribed net longwave flux, radiation terms only
(the subsidence-warming term is excluded — it is not radiation).
"""
function prescribed_flux(col, z_lev)
    nz = length(col.ql)
    # Q(0,z) and Q(z,top): cumulative kappa * rho * q_l
    cum = zeros(nz + 1)
    for k in 1:nz
        cum[k+1] = cum[k] + KAPPA * col.ρ[k] * col.ql[k] * col.dz[k]
    end
    total = cum[end]
    return [F0_SPEC * exp(-(total - cum[k])) + F1_SPEC * exp(-cum[k])
            for k in 1:nz+1]
end

# =========================================================================
function main()
    col, z_lay, z_lev, nf = rf01_column()
    nz = length(col.ql)

    # --- cloud diagnostics -----------------------------------------------
    lwp = sum(col.ρ[k] * col.ql[k] * col.dz[k] for k in 1:nz) * 1000  # g/m2
    cloudy = findall(>(1e-6), col.ql)
    isempty(cloudy) && error("no cloud formed — the saturation adjustment is broken")
    zb = z_lev[first(cloudy)]
    zt = z_lev[last(cloudy) + 1]
    qlmax = maximum(col.ql) * 1000

    println("="^70)
    println("DYCOMS-II RF01 — cloud-top radiative cooling")
    println("="^70)
    @printf("  layers                 %d (%d in the LES layer)\n", nz, nf)
    @printf("  cloud base             %7.1f m\n", zb)
    @printf("  cloud top              %7.1f m   (spec inversion z_i = %.0f m)\n", zt, ZI)
    @printf("  max liquid water       %7.3f g/kg\n", qlmax)
    @printf("  liquid water path      %7.1f g/m2  (RF01 observed 45-60)\n", lwp)

    # --- radiation solve -------------------------------------------------
    lw     = longwave_bands(Float64; ngpt_per_band = 4)
    gas    = correlated_k_optics(lw)
    cloud  = mie_cloud_optics(lw)
    solver = two_stream_solver(lw; lw_variant = Quadrature(),
                                   linear_lw_source = true)

    # clear-sky reference (gas only)
    o_clear = allocate_optics(Float64, (nz,), lw)
    compute_optics!(o_clear, gas, col)
    f_clear = allocate_fluxes(Float64, (), nz + 1)
    solve_lw!(f_clear, solver, o_clear, col; sfc_emissivity = 0.98)

    # all-sky (gas + cloud)
    o_all = allocate_optics(Float64, (nz,), lw)
    compute_optics!(o_all, gas, col)
    o_cld = allocate_optics(Float64, (nz,), lw)
    compute_optics!(o_cld, cloud, col)
    combine_optics!(o_all, o_cld)
    f_all = allocate_fluxes(Float64, (), nz + 1)
    solve_lw!(f_all, solver, o_all, col; sfc_emissivity = 0.98)

    # net UPWARD flux (the DYCOMS F_rad convention is positive upward)
    Fup_all   = [f_all.flux_up[k] - f_all.flux_dn[k] for k in 1:nz+1]
    Fup_clear = [f_clear.flux_up[k] - f_clear.flux_dn[k] for k in 1:nz+1]
    Fspec     = prescribed_flux(col, z_lev)

    # --- the headline number: the jump across cloud top ------------------
    kb = first(cloudy); kt = last(cloudy) + 1
    ΔF_all  = Fup_all[kt] - Fup_all[kb]
    ΔF_spec = Fspec[kt] - Fspec[kb]

    println()
    println("  Net upward longwave flux (W/m2, positive up)")
    println("  " * "-"^58)
    @printf("  %-34s %9s %9s\n", "", "Aeolus", "DYCOMS spec")
    @printf("  %-34s %9.2f %9.2f\n", "at cloud base",  Fup_all[kb], Fspec[kb])
    @printf("  %-34s %9.2f %9.2f\n", "at cloud top",   Fup_all[kt], Fspec[kt])
    @printf("  %-34s %9.2f %9.2f\n", "jump across cloud", ΔF_all, ΔF_spec)
    @printf("  %-34s %9.2f %9s\n",   "clear-sky jump (no cloud)",
            Fup_clear[kt] - Fup_clear[kb], "--")

    # --- cooling rate ----------------------------------------------------
    dθdt = zeros(nz)
    radiative_heating!(dθdt, f_all, col)
    Kday = dθdt .* 86400
    kpk = argmin(Kday)

    println()
    println("  Radiative heating rate")
    println("  " * "-"^58)
    @printf("  peak cooling            %8.1f K/day  at z = %.0f m\n",
            Kday[kpk], z_lay[kpk])
    @printf("  depth below cloud top   %8.1f m\n", zt - z_lay[kpk])
    @printf("  mean over top 50 m      %8.1f K/day\n",
            begin
                sel = findall(k -> z_lay[k] > zt - 50 && z_lay[k] <= zt, 1:nz)
                isempty(sel) ? NaN : sum(Kday[sel]) / length(sel)
            end)
    @printf("  sub-cloud mean          %8.2f K/day\n",
            begin
                sel = findall(k -> z_lay[k] < zb, 1:nz)
                isempty(sel) ? NaN : sum(Kday[sel]) / length(sel)
            end)

    # --- profile through cloud top ---------------------------------------
    println()
    println("  Profile through cloud top")
    println("  " * "-"^58)
    @printf("  %8s %8s %10s %10s %10s\n", "z (m)", "ql g/kg", "F_up", "F_spec", "K/day")
    for k in max(1, kt - 14):min(nz, kt + 3)
        @printf("  %8.0f %8.3f %10.2f %10.2f %10.1f\n",
                z_lay[k], col.ql[k]*1000, Fup_all[k], Fspec[k], Kday[k])
    end

    # --- verdict ----------------------------------------------------------
    println()
    println("  Assessment")
    println("  " * "-"^58)
    rel = 100 * (ΔF_all - ΔF_spec) / ΔF_spec
    @printf("  cloud-top flux jump differs from the prescription by %+.1f%%\n", rel)
    if abs(rel) < 25
        println("  -> within the spread of the intercomparison ensemble.")
    else
        println("  -> OUTSIDE the ensemble spread; see notes in the roadmap.")
    end
    println("="^70)

    return (; lwp, zb, zt, ΔF_all, ΔF_spec, rel,
              peak_cooling = Kday[kpk], z_peak = z_lay[kpk])
end

isinteractive() || main()
