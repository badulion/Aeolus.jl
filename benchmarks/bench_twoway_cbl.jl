# Long-run stability of two-way nesting in a convective boundary layer
# (N-nest-e, #263): the scaled Hellsten et al. (2021, GMD 14, 3185, §4.1)
# case at the issue's resolution, 4 hours.
#
# Parent 3.2 × 3.2 × 1.6 km at Δ = 50 m (64 × 64 × 32, periodic sides),
# child 1.2 × 1.2 km × 800 m on the ground at r = 3 (Δ = 16.7 m,
# 72 × 72 × 48), surface heat flux 0.1 K m/s, no mean wind, Smagorinsky
# SGS, adaptive steps (C = 0.5). The case is test/twoway_cbl_case.jl, the
# one the suite runs at Δ = 100 m for 2 h.
#
# Reports, for one-way and two-way feedback:
#   * the mean kinetic energy over hours 3 and 4 and whether the last hour
#     grows monotonically (12 five-minute means);
#   * the largest child |∇·u| of any step and the composite θ budget error;
#   * the child-mean secondary-circulation w at 400 m over the last hour
#     (Hellsten et al. 2021 report a mean child updraft of 0.4–0.9 m/s in
#     their two-way CBL nests) — a measurement, not a target;
#   * the growth rate of the 2Δ θ mode along the parent rows crossing the
#     nest over the last two hours;
#   * wall time.
#
# Run: julia --project=. benchmarks/bench_twoway_cbl.jl [hours=4] [Δ=50]

include(joinpath(@__DIR__, "..", "test", "twoway_cbl_case.jl"))

const HOURS = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 4.0
const DELTA = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 50.0

function report(name, o, run, wall)
    T = HOURS * 3600
    st = run.stats
    @printf("\n%s (Δ = %.0f m, %.1f h)\n", name, DELTA, HOURS)
    @printf("  %d parent steps (Δt %.2f–%.2f s), %d child substeps (n %d..%d), retries %d/%d, wall %.0f s\n",
            st.parent_steps, st.dt_parent_min, st.dt_parent_max, st.child_substeps,
            st.n_min, st.n_max, st.parent_retries, st.child_retries, wall)
    @printf("  finite: %s; max child |∇·u| %.1e s⁻¹; composite θ budget error %.1e\n",
            o.finite[], o.divmax[], o.budget[])
    kl = _twl_window(o.t, o.ke, T - 3600, T); kp = _twl_window(o.t, o.ke, T - 7200, T - 3600)
    means = [_twl_window(o.t, o.ke, T - 3600 + 300(i - 1), T - 3600 + 300i) for i in 1:12]
    mono = all(diff(means) .> 0)
    @printf("  KE: mean over the last hour %.4e, over the hour before %.4e (ratio %.3f); last hour monotone growth: %s\n",
            kl, kp, kl / kp, mono)
    @printf("  child-mean w at 400 m over the last hour: %+.3f m/s\n",
            _twl_window(o.t, o.wc, T - 3600, T))
    late = o.t .>= T - 7200
    s, se = _twl_fit(o.t[late] ./ 3600, log.(o.noise[late]))
    @printf("  2Δ θ mode on the rows crossing the nest: %.2e K at the end; growth over the last 2 h %.3f ± %.3f h⁻¹\n",
            o.noise[end], s, se)
end

for fb in (:one_way, :two_way)
    o, run, wall = _twl_cbl(fb; Δ = DELTA, T = HOURS * 3600)
    report(string(fb), o, run, wall)
end
