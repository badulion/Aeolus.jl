# =========================================================================
# Turbulence collapse in the flux-cooled half channel (P58-e, #193)
#
# The neutral half channel of P54-d (`half_channel_case`, 2πH × 2πH × H,
# -∂p/∂x = u*²/H, z0/H = 1e-4) with a prescribed surface cooling
# `surface_heat_flux = -Q`, the flux-prescribed MOST wall law
# (`MOSTFluxWallLaw`, P57-a/P58-e) and the stability-corrected Smagorinsky
# closure (`StabilityCorrected`, P55-e). A neutral run is spun up for
# 10 H/u*; from its state the cooling is switched on and each run is
# continued for 20 H/u*. The cases are examples/sbl/configs/collapse_channel.toml
# (the `sbl` run driver, CFG-d #359) with each heat_flux_fraction set on
# it; the runs share one spin-up (`julia -m Aeolus run
# examples/sbl/configs/collapse_channel.toml` is one run with its own).
#
# The sweep is in units of the maximum sustainable heat flux (Van de
# Wiel et al. 2012, `max_sustainable_heat_flux`) of the neutral state at
# two heights:
#   Q_wall — at the wall model's height Δz/2 = H/64 (beyond it the wall
#            layer cannot carry the flux and the wall stress vanishes);
#   Q_bulk — at the channel height H (the channel-scale criterion of
#            Donda et al. 2015).
# For each run it prints the resolved TKE over 15-20 H/u* relative to the
# neutral run (whole domain and lowest third), the number of lowest-third
# levels with turbulent fraction γ < 0.2 (Transition, P59-c) at 20 H/u*,
# the first-level wind relative to the neutral U₀ and the wall stress.
#
#   julia --project=. examples/sbl/collapse_channel.jl        # ~4 min
#   FRACS=0,0.5,2 julia --project=. examples/sbl/collapse_channel.jl
#
# The theory page (docs/src/theory/turbulence_stability.md, "Maximum
# sustainable heat flux") quotes the table this prints.
# =========================================================================

using Aeolus
using Printf
include(joinpath(@__DIR__, "config_schema.jl"))   # the `sbl` driver: collapse_case, collapse_run

# H/Q_wall of each run (0 = neutral)
const FRACS = parse.(Float64, split(get(ENV, "FRACS", "0,0.0159,0.0636,0.25,0.5,1,2"), ","))
const CFG = sbl_config("collapse_channel")          # 16 × 16 × 32, z0/H = 1e-4, θ_ref = 300 K

function main()
    N = CFG.grid.size; Nz = N[3]
    sc = collapse_heat_flux_scales(N; z0 = CFG.case.z0, θ_ref = CFG.case.theta_ref)
    Qwall, Qbulk, U0 = sc.Q_wall, sc.Q_bulk, sc.U0
    @printf("MSHF of the neutral state: Q_wall = %.1f K m/s (U₀ = %.2f u* at H/64), Q_bulk = %.2f K m/s\n",
            Qwall, U0, Qbulk)
    t0 = time()
    spin = sbl_channel_spinup(CFG)
    @printf("neutral spin-up to %.0f H/u*: %.0f s\n\n", CFG.case.spinup_time, time() - t0)
    lowest = 1:cld(Nz, 3)
    rows = []
    for f in FRACS
        cfg = with(CFG, "case.heat_flux_fraction" => f)
        r = sbl_channel_run(cfg, sbl_channel_case(cfg), spin)
        push!(rows, (f = f, e = r.e, el = r.el, nlam = r.nlam, u1 = r.u1, τ = r.τ, wall = r.wall))
    end
    neu = rows[findfirst(r -> r.f == 0, rows)]
    println("| H/H_max(wall) | H/H_max(bulk) | TKE/TKE₀ | TKE/TKE₀ lowest third | levels γ<0.2 (of $(length(lowest))) | U₁/U₀ | τ_w/u*² | wall time |")
    println("| --: | --: | --: | --: | --: | --: | --: | --: |")
    for r in rows
        @printf("| %.4g | %.3g | %.3f | %.3f | %d | %.3f | %.2f | %.0f s |\n", r.f, r.f * Qwall / Qbulk,
                r.e / neu.e, r.el / neu.el, r.nlam, r.u1, r.τ, r.wall)
    end
    @printf("\n(2 H_max(wall) predicts U₁/U₀ = 2^(1/3) = %.4f; total %.0f s)\n", 2^(1 / 3), time() - t0)
end

main()
