# =========================================================================
# Offline 256³ filtered-DNS a-priori record — Phase 56 (P56-c, #178)
#
# The hybrid-DNS decision on #208: the CI suite runs the 96³ fixture
# (test/fixtures/dns_decay.jl, Re_λ ≈ 64); this script runs the same
# fixture at 256³ once, offline, and writes its filtered a-priori
# statistics (ρ per component, ρ_Π, matched and dynamic C_s, backscatter
# at Δ = 4h, 8h, 16h, 32h; Re_λ, k_max η, skewness, energy budget, the
# energy spectrum) to test/fixtures/dns256_apriori.toml, which
# test/era6_apriori_sgs.jl then asserts against.
#
#   julia --project=. examples/validation/dns256_generate.jl [--set key=value …] [--resume <run folder>]
#
# The run is examples/validation/configs/dns256.toml, through the `dns`
# run driver (examples/validation/dns_apriori_driver.jl, CFG-d #359): this script
# is `aeolus run examples/validation/configs/dns256.toml` that, at the
# shipped grid.n = 256 and with no apriori.out of its own, writes the
# record to test/fixtures/dns256_apriori.toml, then prints its table.
# Other resolutions go to the run folder's output/ (`--set grid.n=32`
# proves the pipeline without touching the fixture). The old positional
# arguments map to `--set grid.n=N --set apriori.out="out" --set fluid.nu=ν`.
#
# Viscosity: ν = 1.035 · 0.018 (48/N)^{4/3} = 2.00e-3 at 256³: the
# fixture's scaling, which holds k_max η roughly fixed (measured minima
# 1.494, 1.544, 1.626, 1.598 at 32³, 48³, 64³, 96³), with a 3.5 % margin
# (k_max η × 1.026). At 256³ that gives k_max η ≥ 1.53 even if the peak
# ε reached 0.39, the largest measured on any grid (32³; 0.30 at 96³).
# The 2nd-order FV solver needs k_max η ≥ 1.5, and the driver refuses to
# write the record otherwise (it goes to <out>.failed; the run fails).
#
# Run length: the dissipation peak moves later as Re_λ grows (t ≈ 0.99,
# 1.45, 1.85, 2.41 at 32³, 48³, 64³, 96³, about +1.29 per e-fold in N;
# extrapolated ≈ 3.7 at 256³), so the run goes to t = 2 and then on to
# 0.25 past the peak (capped at t = 6), where k_max η has had its minimum.
#
# Pipeline validated end to end at 32³, 48³, 64³ and 96³ (this script,
# 3 threads, shared machine with load ≈ 12):
#
#   N    ν          steps  t_end  ε peak  Re_λ  k_max η  ρ_Π 4h (Δ/η)  DNS    total  max RSS
#   32   3.199e-2     70   2.00   0.96    21.4  1.53     0.819 (7.6)    2 s    31 s   0.61 GB
#   48   1.863e-2     99   2.00   1.43    31.1  1.58     0.869 (7.8)    4 s    29 s   0.73 GB
#   64   1.269e-2    168   2.06   1.81    45.3  1.67     0.811 (7.5)   15 s    46 s   0.84 GB
#   96   7.393e-3    290   2.59   2.33    52.1  1.64     0.822 (7.7)  115 s   146 s   1.64 GB
#
# (64³ misses the advisory skewness-settling heuristic by 0.001: 0.051
# against 0.05; recorded, not required.) A failed required check, e.g.
# k_max η = 1.49 at 32³ without the ν margin, writes <out>.failed only.
#
# 256³ estimate, extrapolated from these runs and from 96³–192³ step
# timings (0.37, 0.78, 1.76, 3.13 s/step at 96, 128, 160, 192³, i.e.
# 374–442 ns per cell-step, rising slowly): ≈ 7.5 s/step; ≈ 340 steps
# per unit time (115 at 96³, ∝ N with a slowly growing u_max) to
# t ≈ 3.95 (peak ≈ 3.7 + 0.25), ≈ 1350 steps, so ≈ 2.8 h of DNS (2.5–4 h
# with load) plus ≈ 5 min of statistics (58 s at 160³). Peak memory
# ≈ 8–10 GB: max RSS 1.00, 1.68, 2.60 GB at 96, 128, 160³ with the
# solver's work arrays dropped before the statistics (≈ 500 B per cell).
# Measured (test/fixtures/dns256_apriori.toml, regenerated after #205 on
# 2026-10-05, #235): 1264 steps to t = 3.85 (ε peak at t = 3.60),
# wall_s = 5671 s = 1.58 h of DNS on one core (the solver was then
# single-threaded) plus 195 s of statistics; max RSS 9.81 GB as printed
# by this script, not stored in the TOML. The pre-#205 record took
# 7397 s (2.05 h) and 9.9 GB for the same 1264 steps on a busier machine.
# Re-captured after #400 (2026-10-08; the history reductions threaded,
# docs/src/theory/les_filtering.md): `julia -t 12`, 1975 s of DNS
# (1.56 s/step) plus 174 s of statistics, max RSS 8.53 GB.
#
# Checkpoints: every 30 min of wall clock ([checkpoint] of dns256.toml);
# an interrupted run (killed job, reboot) resumes bit for bit with
# `--resume <its run folder>`.
#
# The equivalent through the suite: AEOLUS_DNS256=1 julia --project=.
# test/runtests.jl (the regeneration testset of era6_apriori_sgs.jl).
# =========================================================================

using Aeolus
using Aeolus.Config: run_config, with
using Printf
include(joinpath(@__DIR__, "dns_apriori_driver.jl"))

const FIXTURE = normpath(joinpath(@__DIR__, "..", "..", "test", "fixtures", "dns256_apriori.toml"))

function dns256_args(args)
    sets = String[]; resume = nothing; i = 1
    while i <= length(args)
        a = args[i]
        a in ("--set", "--resume") && i < length(args) ||
            error("dns256_generate.jl takes --set key=value and --resume <run folder>, got $(repr(a))")
        a == "--set" ? push!(sets, args[i+1]) : (resume = args[i+1])
        i += 2
    end
    return sets, resume
end

sets, resume = dns256_args(ARGS)
cfg = dns_config(joinpath(@__DIR__, "configs", "dns256.toml"); overrides = sets)
cfg.apriori.out === nothing && cfg.grid.n == 256 && (cfg = with(cfg, "apriori.out" => FIXTURE))
total = @elapsed res = run_config(cfg; resume = resume)
rec, d = res.obj.record, res.obj.summary
@printf("total %.0f s (DNS %.0f s, statistics %.0f s); max RSS %.2f GB\n",
        total, d.wall, total - d.wall, Sys.maxrss() / 2^30)
rec === nothing && exit(0)

println("\n| Δ | Δ/η | Smag. ρ12 ρ13 ρ23 | ρ_Π | C_s matched | dynamic (DNS grid) | dynamic (LES grid) | backscatter |")
println("| --: | --: | :-- | --: | --: | --: | --: | --: |")
for key in sort(collect(keys(rec["filter"])); by = k -> parse(Int, chop(k)))
    f = rec["filter"][key]; s = f["smagorinsky"]
    @printf("| %s | %.1f | %.3f %.3f %.3f | %.3f | %.3f | %.3f | %.3f | %.1f%% |\n", key, f["delta_over_eta"],
            s["rho"]["xy"], s["rho"]["xz"], s["rho"]["yz"], s["rho_Pi"], f["cs_matched"],
            f["cs_dynamic_dns_grid"], f["cs_dynamic_les_grid"], 100f["backscatter"])
end
println("checks: ", join(("$k=$v" for (k, v) in sort(collect(rec["checks"]))), ", "))
