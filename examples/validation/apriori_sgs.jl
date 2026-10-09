# =========================================================================
# A-priori SGS-stress analysis on filtered DNS — Phase 56 (P56-c, #178)
#
# The tables of the a-priori record a `dns` run writes
# (output/dns_apriori.toml): decaying isotropic turbulence computed by
# Aeolus, filtered with Gaussians at Δ = 4h, 8h, 16h (32h for dns256); per
# width the exact subfilter stress against the Smagorinsky model (at the
# dissipation-matched C_s), the Bardina scale-similarity model and the
# Clark gradient model, and the dynamic C_s on the DNS grid (second-moment
# test-to-grid ratio) and on the LES grid of spacing Δ.
#
# This script only reads the record. Run the DNS first:
#
#   julia --project -m Aeolus run examples/validation/configs/apriori_sgs.toml   # 96³
#   julia --project examples/validation/apriori_sgs.jl [runs-root] [run name]
#
# (run name: default "apriori_sgs"; "dns256" prints the offline 256³ record)
# writes examples/validation/output/<run name>_<N>.md and prints it.
# =========================================================================

using Printf
include(joinpath(@__DIR__, "read_runs.jl"))

root = runs_root()
name = length(ARGS) >= 2 ? ARGS[2] : "apriori_sgs"
rec = load_run(root, name; file = "dns_apriori.toml").res
d = rec["dns"]; N = d["N"]

order(keys_) = sort(collect(keys_); by = k -> parse(Int, chop(k)))
out = joinpath(@__DIR__, "output", "$(name)_$(N).md")
mkpath(dirname(out))
open(out, "w") do io
    println(io, "# A-priori SGS stress, filtered DNS $(N)³ (Gaussian filter)\n")
    @printf(io, "DNS: ν = %.4g, t = %.2f, %d RK3 steps, %.0f s; Re_λ = %.1f; k_max η ≥ %.2f; derivative skewness %.3f; max |dK/dt/(−ε) − 1| = %.1e.\n\n",
            d["nu"], d["t_end"], d["nsteps"], d["wall_s"], d["Re_lambda"], d["kmax_eta_min"], d["skewness_end"],
            d["budget_err_max"])
    println(io, "DNS checks: ", join(("$k = $v" for (k, v) in sort(collect(rec["checks"]))), ", "), ".\n")
    println(io, "## Correlations with the exact stress (deviatoric)\n")
    println(io, "| Δ | model | ρ11 | ρ22 | ρ33 | ρ12 | ρ13 | ρ23 | ρ_Π | ⟨Π⟩ ratio | backscatter | II | III |")
    println(io, "| --: | :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |")
    for k in order(keys(rec["filter"]))
        f = rec["filter"][k]
        for (name_, key) in (("Smagorinsky", "smagorinsky"), ("Bardina", "bardina"), ("gradient", "gradient"))
            m = f[key]; ρ = m["rho"]
            @printf(io, "| %s | %-11s | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.1f%% | %.3f | %.4f |\n",
                    k, name_, ρ["xx"], ρ["yy"], ρ["zz"], ρ["xy"], ρ["xz"], ρ["yz"], m["rho_Pi"], m["Pi_ratio"],
                    100f["backscatter"], f["II"], f["III"])
        end
    end
    println(io, "\nII, III: mean Lumley invariants of the exact stress; backscatter: fraction of points with Π_exact < 0.\n")
    println(io, "## Smagorinsky coefficient\n")
    println(io, "| Δ | Δ/η | C_s (Π-matched) | dynamic, DNS grid | dynamic, LES grid | min λ₁/tr τ |")
    println(io, "| --: | --: | --: | --: | --: | --: |")
    for k in order(keys(rec["filter"]))
        f = rec["filter"][k]; cs = f["cs_matched"]; cf = f["cs_dynamic_dns_grid"]; cc = f["cs_dynamic_les_grid"]
        coarse = isnan(cc) ? "— (grid < 8³)" : @sprintf("%.3f (%+.1f%%)", cc, 100(cc / cs - 1))
        @printf(io, "| %s | %.1f | %.3f | %.3f (%+.1f%%) | %s | %.1e |\n", k, f["delta_over_eta"], cs,
                cf, 100(cf / cs - 1), coarse, f["min_lambda1_over_trace"])
    end
end
print(read(out, String))
println("wrote ", out)
