# =========================================================================
# Turbulence collapse in the flux-cooled half channel (P58-e, #193)
#
# The neutral half channel of P54-d (`half_channel_case`, 2πH × 2πH × H,
# -∂p/∂x = u*²/H, z0/H = 1e-4) with a prescribed surface cooling
# `surface_heat_flux = -Q`, the flux-prescribed MOST wall law
# (`MOSTFluxWallLaw`, P57-a/P58-e) and the stability-corrected Smagorinsky
# closure (`StabilityCorrected`, P55-e). A neutral run is spun up for
# 10 H/u*; from its state the cooling is switched on and each run is
# continued for 20 H/u*. The runs are the sweep
# examples/sbl/configs/collapse_channel_sweep.toml (the `sbl` run type),
# one run folder per heat_flux_fraction.
#
# The sweep is in units of the maximum sustainable heat flux (Van de
# Wiel et al. 2012, `max_sustainable_heat_flux`) of the neutral state at
# two heights:
#   Q_wall — at the wall model's height Δz/2 = H/64 (beyond it the wall
#            layer cannot carry the flux and the wall stress vanishes);
#   Q_bulk — at the channel height H (the channel-scale criterion of
#            Donda et al. 2015).
# For each run this script prints, from its output/statistics.toml, the
# resolved TKE over 15-20 H/u* relative to the neutral run (whole domain
# and lowest third), the number of lowest-third levels with turbulent
# fraction γ < 0.2 (Transition, P59-c) at 20 H/u*, the first-level wind
# relative to the neutral U₀ and the wall stress.
#
#   julia --project -m Aeolus run examples/sbl/configs/collapse_channel_sweep.toml     # ~4 min per member
#   julia --project examples/sbl/collapse_channel.jl runs/collapse_channel_*
#
# The theory page (docs/src/theory/turbulence_stability.md, "Maximum
# sustainable heat flux") quotes the table this prints.
# =========================================================================

include(joinpath(@__DIR__, "runs.jl"))
using Printf

const RUNS = sbl_runs(ARGS, "julia --project examples/sbl/collapse_channel.jl <run folder>…")

let
    rows = sort!([merge(r.stats, (; f = cfgval(r, "case.heat_flux_fraction"))) for r in RUNS]; by = r -> r.f)
    r1 = rows[1]
    @printf("MSHF of the neutral state: Q_wall = %.1f K m/s (U₀ = %.2f u* at H/64), Q_bulk = %.2f K m/s\n",
            r1.Q_wall, r1.U0, r1.Q_bulk)
    i0 = findfirst(r -> r.f == 0, rows)
    i0 === nothing && error("the table is relative to the neutral run (heat_flux_fraction = 0): pass its run folder too")
    neu = rows[i0]
    println("| H/H_max(wall) | H/H_max(bulk) | TKE/TKE₀ | TKE/TKE₀ lowest third | levels γ<0.2 (of $(Int(r1.nlowest))) | U₁/U₀ | τ_w/u*² | wall time |")
    println("| --: | --: | --: | --: | --: | --: | --: | --: |")
    for r in rows
        @printf("| %.4g | %.3g | %.3f | %.3f | %d | %.3f | %.2f | %.0f s |\n", r.f, r.f * r.Q_wall / r.Q_bulk,
                r.e / neu.e, r.el / neu.el, r.nlam, r.u1, r.τ, r.wall)
    end
    @printf("\n(2 H_max(wall) predicts U₁/U₀ = 2^(1/3) = %.4f)\n", 2^(1 / 3))
end
