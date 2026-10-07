# =========================================================================
# Log-layer mismatch in wall-modelled channel LES (P57-b, #183)
#
# The smooth-wall plane channel of `channel_case` (24 × 12 × 16 on
# 2πδ × πδ × 2δ, SpaldingLaw at both walls) at Re_τ = 182, 544, 1000 and
# 5186, each run with
#
#   * first-cell matching (the law fed with the instantaneous velocity of
#     the wall-adjacent cell), and
#   * the fix: matching height h_wm = 3Δz (Kawai & Larsson 2012, Phys.
#     Fluids 24, 015105) and the exponential time filter of Yang, Park &
#     Moin (2017, Phys. Rev. Fluids 2, 104601), T_f = 0.5 δ/u_τ,
#
# and compared with the bulk velocity U_b⁺ of the DNS of Lee & Moser
# (2015, JFM 774, 395) and Dean's (1978) correlation
# (test/data/lee_moser_2015.jl). It then loops over the SGS closures on
# this branch (StaticSmagorinsky, DynamicSmagorinsky) at Re_τ = 1000, and
# over T_f.
#
# The fifteen runs are the sweep examples/validation/configs/
# log_layer_mismatch.toml (driver "channel"); this script runs them and
# prints the tables.
#
#   julia --project=. examples/validation/log_layer_mismatch.jl     # ~3 min
#   julia --project=. examples/validation/log_layer_mismatch.jl --set stepper.end_time=8.0
#
# Output: tables on stdout (and one run folder per member).
# =========================================================================

using Aeolus
using Printf

include(joinpath(@__DIR__, "..", "..", "test", "data", "lee_moser_2015.jl"))

runs = Dict(r.obj.cfg.run.name => r.obj
            for r in run_configs(joinpath(@__DIR__, "configs", "log_layer_mismatch.toml"), ARGS))

function result(label)
    r = runs["log_layer_mismatch_" * label]; st = r.stats
    sel = findall(z -> 0.1 < z < 0.3, st.zf)
    return (Ub = sum(st.U) / length(st.U), Φ1 = st.phi_m[1],
            Φmax = maximum(abs.(st.phi_m[sel] .- 1)), τ = st.τ_wall, t = st.wall_time,
            cfg = r.cfg)
end

pct(x, r) = 100 * (x / r - 1)

a0 = result("re182_first"); N = a0.cfg.channel.size; T_END = a0.cfg.stepper.end_time
T_AVG = T_END - a0.cfg.stepper.average_from_fraction * T_END
println("Bulk velocity, $(N[1])×$(N[2])×$(N[3]), StaticSmagorinsky($(a0.cfg.sgs.c_s)), averaged over ",
        "$(T_END - T_AVG)-$(T_END) δ/u_τ")
@printf("%8s %8s %8s | %-26s | %-26s | %s\n", "Re_τ", "LM U_b⁺", "Dean", "first cell: U_b⁺ (err LM)",
        "h_wm=3Δz + T_f: U_b⁺ (err)", "Φ_m(Δz) first / fix; max|Φ_m−1| 0.1-0.3δ first / fix")
for (c, name) in zip(LM2015_CHANNEL, ("182", "544", "1000", "5186"))
    a = result("re$(name)_first")
    b = result("re$(name)_fix")
    @printf("%8.1f %8.2f %8.2f | %7.3f (%+5.2f %%, τ_w %.3f) | %7.3f (%+5.2f %%, τ_w %.3f) | %.2f / %.2f; %.3f / %.3f\n",
            c.Re_τ, c.Ub_plus, dean_ub_plus(c.Re_τ), a.Ub, pct(a.Ub, c.Ub_plus), a.τ,
            b.Ub, pct(b.Ub, c.Ub_plus), b.τ, a.Φ1, b.Φ1, a.Φmax, b.Φmax)
end

println("\nClosures at Re_τ = 1000")
ref = lm2015_ub_plus(1000)
for (name, label) in (("StaticSmagorinsky(0.1)", "static"), ("DynamicSmagorinsky (plane)", "dynamic"))
    a = result(label * "_first")
    b = result(label * "_fix")
    @printf("  %-28s first cell %+6.2f %% (Φ_m(Δz) %.2f) | fix %+6.2f %% (Φ_m(Δz) %.2f, max|Φ_m−1| %.3f)\n",
            name, pct(a.Ub, ref), a.Φ1, pct(b.Ub, ref), b.Φ1, b.Φmax)
end

println("\nTime-filter scale at Re_τ = 1000, h_wm = 3Δz")
for Tf in ("0.1", "0.5", "2.0")
    b = result("tf" * Tf)
    @printf("  T_f = %s δ/u_τ: U_b⁺ %+6.2f %%, Φ_m(Δz) %.2f, max|Φ_m−1| %.3f\n", Tf, pct(b.Ub, ref), b.Φ1, b.Φmax)
end
