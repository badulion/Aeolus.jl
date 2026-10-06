# =========================================================================
# Inflow statistics of a convective Aeolus parent — N-inflow-b (#258)
#
# The convective boundary layer of examples/sgs/cbl_tke.jl (free convection,
# Q = 0.1 K m/s, Deardorff–Moeng TKE closure; harness test/tke_cbl_driver.jl)
# is the parent. Over the last `avg` seconds every step is folded into a
# `ParentResolvedStatistics` (τ_avg = 600 s, whole plane, west face) in
# scale-gap mode for a child refined r = 2 horizontally (the Berlin
# (2, 2, 4) nest), and the boundary statistics are printed split into
#
#   resolved   the parent's resolved stress (what the interpolated
#              boundary values carry, k ≤ k_c, plus the band below)
#   missing    what the generator must add: SGS + the band k_c < k ≤ 2k_c
#              (k_c = 2π/(4Δx), wavelengths 100–200 m at Δx = 50 m)
#
#   julia --project=. examples/nesting/cbl_inflow_statistics.jl [N] [hours]
#
# Default 64³ on 3200 × 3200 × 1600 m (Δx = 50 m, Δz = 25 m), z_i0 = 800 m,
# 2 h, statistics over the last 30 min.
# =========================================================================

using Aeolus
using Printf
include(joinpath(@__DIR__, "..", "..", "test", "tke_cbl_driver.jl"))

N = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 64
hours = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 2.0
T_end, avg = 3600hours, 1800.0
Δz = 1600.0 / N
nlev = min(N, round(Int, 500 / Δz))           # the child needs ≤ 400 m; keep 500 m

src = Ref{Any}(nothing)
cost = Ref(0.0); nupd = Ref(0)
function fold!(sim, c, h)
    sim.t > T_end - avg || return
    if src[] === nothing
        src[] = ParentResolvedStatistics(sim; closure = c, τ_avg = 600.0, refine = (2, 2, 4),
                                         levels = 1:nlev)
    end
    cost[] += @elapsed update_inflow_statistics!(src[], h)
    nupd[] += 1
end
r = tke_cbl(N; Δ = 3200.0 / N, Δz = Δz, Q = 0.1, z_i0 = 800.0, Γ = 0.003,
            T_end = T_end, avg = avg, progress = true, on_step = fold!)
println(tke_cbl_summary(r))
s = src[]
p = inflow_stress_parts(s)
@printf("ParentResolvedStatistics: %d updates, %.1f ms each (%d × %d plane, %d levels, lag_z = %d)\n",
        nupd[], 1e3 * cost[] / nupd[], N, N, nlev, s.lag_z)
@printf("w* = %.2f m/s, z_i = %.0f m; band: wavelengths %.0f–%.0f m\n", r.w_star, r.zi,
        s.c_gap * s.Δh / s.refine, s.c_gap * s.Δh)
println("   z    σ_u res  σ_u miss (band+SGS)   σ_w res  σ_w miss   u'w' res   u'w' miss   e_SGS   L_z res  L_z band  L_s band")
for z in (50.0, 100.0, 200.0, 300.0, 400.0)
    k = findmin(abs.(p.z .- z))[2]
    miss = p.generated[:, k]
    @printf("%5.0f   %6.3f    %6.3f (%5.3f+%5.3f)    %6.3f    %6.3f   %8.4f   %8.4f   %6.4f   %6.1f   %6.1f   %6.1f\n",
            p.z[k], sqrt(p.resolved[1, k]), sqrt(miss[1]), p.band[1, k], p.sgs[1, k],
            sqrt(p.resolved[3, k]), sqrt(miss[3]), p.resolved[5, k], miss[5], p.e_sgs[k],
            p.L_z_full[1, k], p.L_z[1, k], p.L_s[1, k])
end
println("(σ in m/s; the band and SGS columns of σ_u are variances, m²/s²; lengths of u in m)")
