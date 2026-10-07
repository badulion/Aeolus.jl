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
# The column and the solve are examples/validation/configs/dycoms_rf01.toml
# (driver "column", case "dycoms_rf01"; the code is in
# examples/validation/dycoms_rf01/rf01.jl). This script runs it and prints
# the report.
#
# Run:  julia --project=. examples/validation/dycoms_rf01_radiation.jl [--set key=value]
# =========================================================================

using Printf
using Aeolus

# =========================================================================
function main(args = ARGS)
    run = only(run_configs(joinpath(@__DIR__, "configs", "dycoms_rf01.toml"), args)).obj
    (; col, z_lay, z_lev, nf, Fup_all, Fup_clear, Fspec, Kday) = run.state
    ZI = run.cfg.dycoms_rf01.zi
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
