# =========================================================================
# Neutral ABL half channel — the Phase 54 benchmark (P54-d, #174)
#
# The pressure-driven half channel of Porté-Agel, Meneveau & Parlange
# (2000, JFM 415, 261) and Bou-Zeid, Meneveau & Parlange (2005, Phys.
# Fluids 17, 025105): 2πH × 2πH × H, -∂p/∂x = u*²/H, free-slip top, rough
# log-law wall stress at the bottom (z0/H = 1e-4), inviscid, no damping
# function. Built with `half_channel_case` (WallModels); Φ_m and E11 are
# Operators' `phi_m_profile` and `streamwise_spectrum`.
#
# Runs each closure of CLOSURES at each resolution of NS for T_END H/u*,
# averages over the last 40 %, and writes a two-panel figure:
#   (a) Φ_m(z/H) — the Smagorinsky overshoot near the wall (Mason & Thomson
#       1992, JFM 242, 51; Porté-Agel et al. 2000 Fig. 7) and the
#       scale-dependent Lagrangian model (LASD, Bou-Zeid et al. 2005;
#       P54-e, #186), which removes it without a damping function;
#   (b) E11(k1 H) of the last run at z/H ≈ 0.06, 0.1, 0.25, 0.5 with k1^(-1)
#       and k1^(-5/3) references.
# Closures: `smagorinsky` (StaticSmagorinsky(0.1)), `lasd`
# (LagrangianDynamicSmagorinsky(scale_dependent = true), with the P57-b 2Δ
# input filter of the wall model), `lasi` (scale invariant, same filter).
# The fitted spectral slopes (least squares in log-log) are printed: near
# the wall over k1 H ∈ [1, 6] (the k1^(-1) production range), at z/H ≈ 0.25
# over k1 H ∈ [4, N/3] (the k1^(-5/3) inertial range).
#
# Measured (T_END = 30, averages over 18-30 H/u*, seed 1; P54-e, #186):
#
#   run          Φ_m over 0.05-0.3 H   E11 slope near wall      E11 slope z/H ≈ 0.24
#                                      (k1 H ∈ [1, 6])          (k1 H ∈ [4, N/3])
#   Smag 0.1 32³  [0.965, 1.589]        -2.01 (0.047), -1.28 (0.109)   -5.14
#   LASD 32³      [0.935, 1.133]        -0.66 (0.047), -0.79 (0.109)   -4.27
#   LASD 48³      [0.961, 1.094]        -0.43 (0.052), -0.61 (0.094)   -3.48
#
# Near the wall LASD gives the shallow production-range spectrum (k1^(-1)
# expected; the Smagorinsky overshoot steepens it). At z/H ≈ 0.25 the
# k1^(-5/3) range (k1 z > 1, so k1 H > 4) lies at wavelengths below
# 8-12 Δx, where the second-order scheme and the closure damp: the fitted
# slope steepens with coarser grids (-3.5 at 48³, -4.3 at 32³, -5.1 for
# Smagorinsky at 32³) instead of showing -5/3. Cost: LASD 544 s at 32³,
# 3338 s at 48³ (loaded machine), Smagorinsky 199 s at 32³.
#
#   julia --project=. examples/validation/neutral_abl.jl         # ~15 min
#   T_END=20 NS=32 CLOSURES=lasd julia --project=. examples/validation/neutral_abl.jl
#
# writes examples/validation/output/neutral_abl.svg
# =========================================================================

using Aeolus
using Printf

const T_END = parse(Float64, get(ENV, "T_END", "50"))
const T_AVG = 0.6 * T_END
const C_S = 0.1
const CLOSURES = Symbol.(split(get(ENV, "CLOSURES", "smagorinsky,lasd"), ","))
const NS = parse.(Int, split(get(ENV, "NS", "32,48"), ","))

function make_case(N, name)
    name === :smagorinsky && return half_channel_case(; N = N, closure = StaticSmagorinsky(C_S))
    g = RegularGrid(; size = (N, N, N), extent = (2π, 2π, 1.0))
    cl = LagrangianDynamicSmagorinsky(g; scale_dependent = name === :lasd)
    name in (:lasd, :lasi) || error("unknown closure $name")
    return half_channel_case(; N = N, closure = cl, filter = :spatial)
end

"Least-squares slope of log E against log k over k ∈ [klo, khi]."
function fit_slope(k, E, klo, khi)
    m = (k .>= klo) .& (k .<= khi) .& (E .> 0)
    x = log.(k[m]); y = log.(E[m])
    xm = sum(x) / length(x); ym = sum(y) / length(y)
    return sum((x .- xm) .* (y .- ym)) / sum((x .- xm) .^ 2), count(m)
end

function run_case(N, name)
    c = make_case(N, name)
    levels = unique(clamp.(round.(Int, (0.06, 0.1, 0.25, 0.5) .* N .+ 0.5), 1, N))
    Esum = nothing; ns = Ref(0); n = Ref(0)
    cb = function (case)
        n[] += 1
        (case.sim.t >= T_AVG && n[] % 10 == 0) || return
        sp = streamwise_spectrum(case.sim.u, levels)
        Esum === nothing ? (Esum = copy(sp.E11)) : (Esum .+= sp.E11)
        ns[] += 1
    end
    t0 = time()
    st = run_channel!(c, T_END; average_from = T_AVG, callback = cb)
    k1 = streamwise_spectrum(c.sim.u, levels).k1
    @printf("%s, N = %d: %d steps, %.0f s; ⟨τ_w⟩ = %.3f u*² over %.0f-%.0f H/u*\n",
            name, N, c.nsteps, time() - t0, st.τ_wall, T_AVG, T_END)
    @printf("   Φ_m at z/H = %s\n", join((@sprintf("%.3f:%.2f", z, p) for (z, p) in
                                          zip(st.zf[1:min(8, end)], st.phi_m[1:min(8, end)])), "  "))
    @printf("   max Φ_m below z = 0.3H: %.3f; Φ_m over 0.05 ≤ z/H ≤ 0.3 in [%.3f, %.3f]\n",
            maximum(st.phi_m[st.zf .<= 0.3]), extrema(st.phi_m[0.05 .<= st.zf .<= 0.3])...)
    E = Esum ./ ns[]
    zl = [(k - 0.5) / N for k in levels]
    for (m, z) in enumerate(zl)
        lo, hi = z < 0.2 ? (1.0, 6.0) : (4.0, N / 3)
        sl, npts = fit_slope(k1, E[:, m], lo, hi)
        @printf("   E11 slope at z/H = %.3f over k1 H ∈ [%.0f, %.1f] (%d points): %.2f\n", z, lo, hi, npts, sl)
    end
    return (; N, name, st, k1, E, zl)
end

results = [run_case(N, name) for name in CLOSURES for N in NS]

# ---- the figure ----------------------------------------------------------
function write_figure(path, results)
    mkpath(dirname(path))
    W, H = 1000, 460
    pw, ph = 380, 340
    panels = ((80, 60), (580, 60))
    cols = ("#2563eb", "#dc2626", "#059669", "#d97706")
    io = IOBuffer()
    println(io, """<?xml version="1.0" encoding="UTF-8"?>""")
    println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$W" height="$H" font-family="DejaVu Sans, sans-serif">""")
    println(io, """<rect width="100%" height="100%" fill="white"/>""")
    println(io, """<text x="$(W/2)" y="28" font-size="17" text-anchor="middle" font-weight="bold">Neutral half channel, no wall damping ($(round(Int, T_AVG))-$(round(Int, T_END)) H/u*)</text>""")
    # (a) Φ_m: x ∈ [0, 3], y = z/H ∈ [0, 1]
    x0, y0 = panels[1]
    X(x) = x0 + x / 3 * pw; Y(z) = y0 + ph - z * ph
    println(io, """<rect x="$x0" y="$y0" width="$pw" height="$ph" fill="none" stroke="#333"/>""")
    for xv in 0:0.5:3
        println(io, """<line x1="$(X(xv))" y1="$(y0+ph)" x2="$(X(xv))" y2="$(y0+ph+5)" stroke="#333"/><text x="$(X(xv))" y="$(y0+ph+20)" font-size="12" text-anchor="middle">$(xv)</text>""")
    end
    for zv in 0:0.2:1
        println(io, """<line x1="$(x0-5)" y1="$(Y(zv))" x2="$x0" y2="$(Y(zv))" stroke="#333"/><text x="$(x0-8)" y="$(Y(zv)+4)" font-size="12" text-anchor="end">$(round(zv; digits=1))</text>""")
    end
    println(io, """<line x1="$(X(1))" y1="$y0" x2="$(X(1))" y2="$(y0+ph)" stroke="#888" stroke-dasharray="5,4"/>""")
    println(io, """<text x="$(X(1)+4)" y="$(y0+14)" font-size="12" fill="#555">log law, Φ_m = 1</text>""")
    for (i, r) in enumerate(results)
        pts = join(("$(X(clamp(p, 0, 3))),$(Y(z))" for (z, p) in zip(r.st.zf, r.st.phi_m)), " ")
        println(io, """<polyline fill="none" stroke="$(cols[i])" stroke-width="2" points="$pts"/>""")
        for (z, p) in zip(r.st.zf, r.st.phi_m)
            println(io, """<circle cx="$(X(clamp(p, 0, 3)))" cy="$(Y(z))" r="2.5" fill="$(cols[i])"/>""")
        end
        println(io, """<line x1="$(x0+pw-120)" y1="$(y0+20i)" x2="$(x0+pw-95)" y2="$(y0+20i)" stroke="$(cols[i])" stroke-width="2"/><text x="$(x0+pw-88)" y="$(y0+20i+4)" font-size="12">$(r.name == :smagorinsky ? "Smag $(C_S)" : uppercase(String(r.name))) $(r.N)³</text>""")
    end
    println(io, """<text x="$(x0+pw/2)" y="$(y0+ph+42)" font-size="14" text-anchor="middle">Φ_m = (κz/u*) ∂&lt;U&gt;/∂z</text>""")
    println(io, """<text x="$(x0-48)" y="$(y0+ph/2)" font-size="14" text-anchor="middle" transform="rotate(-90 $(x0-48),$(y0+ph/2))">z / H</text>""")
    println(io, """<text x="$(x0+6)" y="$(y0+18)" font-size="14" font-weight="bold">(a)</text>""")
    # (b) E11, finest run: log-log
    r = results[end]
    x0, y0 = panels[2]
    k = r.k1[2:end]; E = r.E[2:end, :]
    lx = log10.(k); lxmin, lxmax = floor(minimum(lx)), ceil(maximum(lx))
    ly = log10.(max.(E, 1e-30)); lymin = floor(minimum(ly)); lymax = ceil(maximum(ly))
    XL(v) = x0 + (v - lxmin) / (lxmax - lxmin) * pw
    YL(v) = y0 + ph - (v - lymin) / (lymax - lymin) * ph
    println(io, """<rect x="$x0" y="$y0" width="$pw" height="$ph" fill="none" stroke="#333"/>""")
    for v in lxmin:lxmax
        println(io, """<line x1="$(XL(v))" y1="$(y0+ph)" x2="$(XL(v))" y2="$(y0+ph+5)" stroke="#333"/><text x="$(XL(v))" y="$(y0+ph+20)" font-size="12" text-anchor="middle">1e$(Int(v))</text>""")
    end
    for v in lymin:lymax
        println(io, """<line x1="$(x0-5)" y1="$(YL(v))" x2="$x0" y2="$(YL(v))" stroke="#333"/><text x="$(x0-8)" y="$(YL(v)+4)" font-size="12" text-anchor="end">1e$(Int(v))</text>""")
    end
    for m in axes(E, 2)
        pts = join(("$(XL(a)),$(YL(b))" for (a, b) in zip(lx, ly[:, m])), " ")
        println(io, """<polyline fill="none" stroke="$(cols[m])" stroke-width="2" points="$pts"/>""")
        println(io, """<line x1="$(x0+pw-130)" y1="$(y0+20m)" x2="$(x0+pw-105)" y2="$(y0+20m)" stroke="$(cols[m])" stroke-width="2"/><text x="$(x0+pw-98)" y="$(y0+20m+4)" font-size="12">z/H = $(round(r.zl[m]; digits=3))</text>""")
    end
    # k^(-5/3) through the middle curve at the 4th wavenumber
    m2 = min(2, size(E, 2)); i0 = min(4, length(k))
    a, b = lx[i0], ly[i0, m2] + 0.3
    a2 = lx[end]; b2 = b - 5 / 3 * (a2 - a)
    println(io, """<line x1="$(XL(a))" y1="$(YL(b))" x2="$(XL(a2))" y2="$(YL(b2))" stroke="#555" stroke-dasharray="6,4"/>""")
    println(io, """<text x="$(XL((a + a2) / 2) + 6)" y="$(YL((b + b2) / 2) - 6)" font-size="13" fill="#555">k₁^(-5/3)</text>""")
    # k1^(-1) through the lowest curve at k1 H = 1
    b1 = ly[1, 1] + 0.3; a1e = log10(6.0); b1e = b1 - (a1e - lx[1])
    println(io, """<line x1="$(XL(lx[1]))" y1="$(YL(b1))" x2="$(XL(a1e))" y2="$(YL(b1e))" stroke="#999" stroke-dasharray="2,3"/>""")
    println(io, """<text x="$(XL(lx[1]) + 6)" y="$(YL(b1) - 8)" font-size="13" fill="#777">k₁^(-1)</text>""")
    println(io, """<text x="$(x0+pw/2)" y="$(y0+ph+42)" font-size="14" text-anchor="middle">k₁ H  ($(r.name == :smagorinsky ? "Smagorinsky" : uppercase(String(r.name))), $(r.N)³)</text>""")
    println(io, """<text x="$(x0-55)" y="$(y0+ph/2)" font-size="14" text-anchor="middle" transform="rotate(-90 $(x0-55),$(y0+ph/2))">E₁₁(k₁) / (u*² H)</text>""")
    println(io, """<text x="$(x0+6)" y="$(y0+18)" font-size="14" font-weight="bold">(b)</text>""")
    println(io, "</svg>")
    write(path, String(take!(io)))
    return path
end

out = write_figure(get(ENV, "OUT", joinpath(@__DIR__, "output", "neutral_abl.svg")), results)
println("wrote ", out)
