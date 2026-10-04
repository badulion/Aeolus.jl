# =========================================================================
# GABLS1 — the first GEWEX stable-boundary-layer LES intercomparison
# (Beare et al. 2006, Boundary-Layer Meteorol. 118, 247), P55-f (#191)
#
# 400 m cube, U_g = 8 m/s, f = 1.39e-4 s⁻¹, z0 = 0.1 m, θ = 265 K to 100 m
# then +0.01 K/m, surface cooled at 0.25 K/h, 9 h; statistics of hours 8-9.
# Built on the shared SBL driver (examples/sbl/driver.jl): geostrophic
# forcing from LargeScaleForcing, surface heat flux from SurfaceExchange,
# surface stress from WallModels.MOSTWallLaw, SGS closure stability-
# corrected by Operators.StabilityCorrected.
#
# Runs, at N³ (default 64³, Δ = 6.25 m), one simulation per closure in
# CLOSURES (default "brown94,none,lilly": Smagorinsky C_s = 0.1 with the
# Brown et al. 1994 correction — the Berlin default —, uncorrected, and with
# the Lilly cutoff). AMD (P55-d, #176), LASD (P54-e, #186) and the TKE
# closure (P56-b, #182) are not merged into this branch's base and are not
# run. Compares h, u*, the surface heat flux and the low-level jet with the
# intercomparison ranges of examples/sbl/data/gabls1_reference.csv.
#
#   julia --project=. examples/sbl/gabls1.jl                    # 64³: hours
#   N=32 julia --project=. examples/sbl/gabls1.jl               # 32³: ~10 min per closure
#   N=32 CLOSURES=brown94 T_END=7200 julia --project=. examples/sbl/gabls1.jl
#   CLOSURES=none julia … & CLOSURES=lilly julia … ; REUSE=1 julia …   # in parallel, then combine
#
# writes examples/sbl/output/gabls1_N<N>.svg and gabls1_N<N>_profiles.csv
# =========================================================================

include(joinpath(@__DIR__, "driver.jl"))
using Printf

const N = parse(Int, get(ENV, "N", "64"))
const T_END = parse(Float64, get(ENV, "T_END", string(9 * 3600)))
const T_AVG = T_END - 3600
const CLOSURES = Symbol.(split(get(ENV, "CLOSURES", "brown94,none,lilly"), ","))
const REF = sbl_read_csv(joinpath(@__DIR__, "data", "gabls1_reference.csv"))

function run_closure(stab)
    c = sbl_case(; N = (N, N, N), sgs_stability = stab)
    series = NamedTuple[]
    cb = function (c)
        c.nsteps % 50 == 0 || return
        push!(series, (t = c.sim.t, u_star = sqrt(hypot(c.τx, c.τy)), wθ = mean(c.wθ)))
    end
    st = sbl_run!(c, T_END; average_from = T_AVG, callback = cb)
    @printf("%-8s N=%d: h = %5.1f m  u* = %.3f m/s  w'θ'0 = %.5f K m/s (H = %5.1f W/m²)  LLJ %.2f m/s at %3.0f m  L = %4.0f m  (%d steps, %.0f s)\n",
            stab, N, st.h, st.u_star, st.wθ0, st.H0, st.S_max, st.z_jet, st.L, st.nsteps, st.wall_time)
    return (stab = stab, st = st, series = series)
end

# One result file per closure (examples/sbl/output/gabls1_N<N>_<closure>.jls),
# so the closures can be run as separate processes and the table and figure
# made from all of them: REUSE=1 loads a closure's result instead of running it.
using Serialization
const OUTDIR = joinpath(@__DIR__, "output"); mkpath(OUTDIR)
function cached(stab)
    path = joinpath(OUTDIR, "gabls1_N$(N)_$(stab).jls")
    if get(ENV, "REUSE", "0") == "1" && isfile(path)
        r = deserialize(path)
        @printf("%-8s N=%d (cached): h = %5.1f m  u* = %.3f m/s  w'θ'0 = %.5f K m/s (H = %5.1f W/m²)  LLJ %.2f m/s at %3.0f m\n",
                stab, N, r.st.h, r.st.u_star, r.st.wθ0, r.st.H0, r.st.S_max, r.st.z_jet)
        return r
    end
    r = run_closure(stab)
    serialize(path, r)
    return r
end
results = [cached(s) for s in CLOSURES]

# ---- the comparison table -------------------------------------------------
lo(k) = sbl_csv_value(REF, k, "low"); hi(k) = sbl_csv_value(REF, k, "high")
println("\nHours 8-9 against the intercomparison (Beare et al. 2006, via Basu & Porté-Agel 2006):")
@printf("  ensemble: u* ∈ [%.3f, %.3f] m/s, w'θ'0 ∈ [%.4f, %.4f] K m/s (H ∈ [%.1f, %.1f] W/m²), h ≈ 200 m; LASD 64³: h = %.0f m, u* = %.3f\n",
        lo("u_star"), hi("u_star"), lo("surface_wtheta"), hi("surface_wtheta"),
        lo("surface_heat_flux"), hi("surface_heat_flux"), lo("h_LASD_64"), lo("u_star_LASD_64"))
for r in results
    inu = lo("u_star") <= r.st.u_star <= hi("u_star")
    inq = lo("surface_wtheta") <= r.st.wθ0 <= hi("surface_wtheta")
    @printf("  %-8s u* %s, w'θ'0 %s\n", r.stab, inu ? "inside" : "OUTSIDE", inq ? "inside" : "OUTSIDE")
end

# ---- profiles CSV ------------------------------------------------------------
outdir = OUTDIR
open(joinpath(outdir, "gabls1_N$(N)_profiles.csv"), "w") do io
    println(io, "# GABLS1 hours $(T_AVG/3600)-$(T_END/3600), Aeolus.jl SBL driver, $(N)^3 (P55-f, #191). Model output, not reference data.")
    println(io, "closure,z_m,U,V,speed,theta,tau_total")
    for r in results, k in eachindex(r.st.z)
        @printf(io, "%s,%.3f,%.4f,%.4f,%.4f,%.4f,%.6f\n", r.stab, r.st.z[k], r.st.U[k], r.st.V[k],
                r.st.S[k], r.st.θ[k], 0.5 * (r.st.τ[k] + r.st.τ[k+1]))
    end
end

# ---- the figure: (a) wind speed, (b) θ, (c) total stress, (d) u*(t) ----------
function write_figure(path, results)
    W, H = 1200, 470
    pw, ph = 230, 330
    cols = ("#2563eb", "#dc2626", "#059669", "#d97706")
    io = IOBuffer()
    println(io, """<?xml version="1.0" encoding="UTF-8"?>""")
    println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$W" height="$H" font-family="DejaVu Sans, sans-serif">""")
    println(io, """<rect width="100%" height="100%" fill="white"/>""")
    println(io, """<text x="$(W/2)" y="26" font-size="17" text-anchor="middle" font-weight="bold">GABLS1, $(N)³ (Δ = $(400/N) m), hours $(round(Int, T_AVG/3600))-$(round(Int, T_END/3600)): Smagorinsky C_s = 0.1 ± stability correction</text>""")
    function panel(ix, xlo, xhi, xticks, xlabel, curves; ylo = 0.0, yhi = 400.0, ylabel = "z (m)", band = nothing, hband = nothing, tag = "")
        x0 = 70 + (ix - 1) * (pw + 70); y0 = 55
        X(x) = x0 + (x - xlo) / (xhi - xlo) * pw; Y(z) = y0 + ph - (z - ylo) / (yhi - ylo) * ph
        if band !== nothing
            println(io, """<rect x="$(X(band[1]))" y="$y0" width="$(X(band[2]) - X(band[1]))" height="$ph" fill="#e5e7eb"/>""")
        end
        if hband !== nothing
            println(io, """<rect x="$x0" y="$(Y(hband[2]))" width="$pw" height="$(Y(hband[1]) - Y(hband[2]))" fill="#e5e7eb"/>""")
        end
        println(io, """<rect x="$x0" y="$y0" width="$pw" height="$ph" fill="none" stroke="#333"/>""")
        for xv in xticks
            println(io, """<line x1="$(X(xv))" y1="$(y0+ph)" x2="$(X(xv))" y2="$(y0+ph+5)" stroke="#333"/><text x="$(X(xv))" y="$(y0+ph+19)" font-size="12" text-anchor="middle">$(xv)</text>""")
        end
        for zv in range(ylo, yhi; length = 5)
            lab = yhi > 10 ? string(round(Int, zv)) : string(round(zv; digits = 2))
            println(io, """<line x1="$(x0-5)" y1="$(Y(zv))" x2="$x0" y2="$(Y(zv))" stroke="#333"/><text x="$(x0-8)" y="$(Y(zv)+4)" font-size="12" text-anchor="end">$lab</text>""")
        end
        for (i, (xs, zs)) in enumerate(curves)
            pts = join(("$(X(clamp(x, xlo, xhi))),$(Y(clamp(z, ylo, yhi)))" for (x, z) in zip(xs, zs)), " ")
            println(io, """<polyline fill="none" stroke="$(cols[i])" stroke-width="2" points="$pts"/>""")
        end
        println(io, """<text x="$(x0+pw/2)" y="$(y0+ph+40)" font-size="14" text-anchor="middle">$xlabel</text>""")
        println(io, """<text x="$(x0-46)" y="$(y0+ph/2)" font-size="14" text-anchor="middle" transform="rotate(-90 $(x0-46),$(y0+ph/2))">$ylabel</text>""")
        println(io, """<text x="$(x0+6)" y="$(y0+18)" font-size="14" font-weight="bold">$tag</text>""")
        return x0, y0
    end
    panel(1, 0, 10, 0:2:10, "wind speed (m/s)", [(r.st.S, r.st.z) for r in results]; tag = "(a)")
    panel(2, 262, 268, 262:2:268, "θ (K)", [(r.st.θ, r.st.z) for r in results]; tag = "(b)")
    panel(3, 0, 0.1, 0:0.025:0.1, "total stress (m²/s²)", [(r.st.τ, r.st.zf) for r in results];
          band = (lo("surface_momentum_flux"), hi("surface_momentum_flux")), tag = "(c)")
    x0, y0 = panel(4, 0, T_END / 3600, 0:3:9, "time (h)",
                   [([p.t / 3600 for p in r.series], [p.u_star for p in r.series]) for r in results];
                   ylo = 0.0, yhi = 0.5, ylabel = "u* (m/s)", hband = (lo("u_star"), hi("u_star")), tag = "(d)")
    for (i, r) in enumerate(results)
        println(io, """<line x1="$(70 + 20)" y1="$(y0+ph+58)" x2="$(70 + 45)" y2="$(y0+ph+58)" stroke="$(cols[i])" stroke-width="2" transform="translate($(180*(i-1)),0)"/><text x="$(70 + 52 + 180*(i-1))" y="$(y0+ph+62)" font-size="13">$(r.stab)</text>""")
    end
    println(io, """<text x="$(W-20)" y="$(y0+ph+62)" font-size="12" text-anchor="end" fill="#555">grey: intercomparison range (Beare et al. 2006, via Basu &amp; Porté-Agel 2006)</text>""")
    println(io, "</svg>")
    write(path, String(take!(io)))
    return path
end

println("wrote ", write_figure(joinpath(outdir, "gabls1_N$(N).svg"), results))
