# =========================================================================
# Relaminarisation of the flux-cooled open channel — P59-e (#195)
#
# The stable open channel of Nieuwstadt (2005, Boundary-Layer Meteorol.
# 116, 277; DNS at Re_τ = 360) and Flores & Riley (2011, BLM 139, 241):
# a pressure-driven channel of depth h with a no-slip bottom and a
# free-slip, insulated top, cooled at the bottom by a constant kinematic
# heat flux Q < 0. The stability parameter is h/L with the Obukhov length
# of the surface flux and the nominal u_τ (`relaminarisation_criteria`;
# L from `SurfaceExchange.obukhov_length`), L⁺ = L u_τ/ν = Re_τ/(h/L).
#
# Each Re_τ is spun up neutral for T_SPIN h/u_τ (the driver's
# `relam_spinup`); from that state the cooling is switched on and each h/L
# run for T_END h/u_τ (`relam_run`). A run has *collapsed* when the
# volume-mean resolved TKE at T_END is below 10 % of its switch-on value,
# and *survived* when it is above 50 %. The critical h/L is the geometric
# mean of the largest surviving and the smallest collapsed h/L; its L⁺ is
# Re_τ divided by it.
#
# References: Nieuwstadt (2005), h/L ≈ 1.23 at Re_τ = 360 with an L that
# excludes κ, i.e. h/L ≈ 1.23 κ ≈ 0.49 with Aeolus's L, which includes κ
# (target ±30 %); Flores & Riley (2011), collapse for L⁺ ≲ 100 (target
# within a factor 1.5; at Re_τ = 360 not consistent with Nieuwstadt's
# threshold, L⁺ ≈ 730 there); Donda et al. (2015, QJRMS 141, 2137) for
# the transient collapse.
#
#   julia --project=. examples/sbl/relaminarisation_channel.jl
#   RE=180 HL=0.3,2 julia --project=. examples/sbl/relaminarisation_channel.jl
#   RE=360 N=48,48,48 HL=1 julia … &  (one process per run; then REUSE=1)
#
# writes examples/sbl/output/relaminarisation_<Re>_<N>.svg and caches every
# run (.jls); prints the table quoted on the theory page
# (docs/src/theory/transition.md, "Relaminarisation").
# =========================================================================

using Aeolus
using Printf, Serialization
using Statistics: mean
include(joinpath(@__DIR__, "driver.jl"))

const RES = parse.(Int, split(get(ENV, "RE", "180,360"), ","))
const HLS = parse.(Float64, split(get(ENV, "HL", "0.3,0.6,1,1.5,2"), ","))
const T_END = parse(Float64, get(ENV, "T_END", "20"))
const T_SPIN = parse(Float64, get(ENV, "T_SPIN", "10"))
grid(Re) = haskey(ENV, "N") ? Tuple(parse.(Int, split(ENV["N"], ","))) :
           (Re <= 180 ? (32, 32, 32) : (48, 48, 48))
const OUTDIR = joinpath(@__DIR__, "output"); mkpath(OUTDIR)
# Nieuwstadt (2005)'s critical h/L at Re_τ = 360 is 1.23 with an Obukhov
# length that excludes κ; Aeolus's L (`SurfaceExchange.obukhov_length`)
# includes it, so the same threshold is 1.23 κ here.
const H_NIEUWSTADT = 1.23 * Aeolus.SurfaceExchange.von_karman()
tag(Re) = "Re$(Re)_" * join(grid(Re), "x")

function spinup(Re)
    path = joinpath(OUTDIR, "relam_$(tag(Re))_spinup_T$(round(Int, T_SPIN)).jls")
    isfile(path) && return deserialize(path)
    sp = relam_spinup(Re; N = grid(Re), T_spin = T_SPIN)
    serialize(path, sp)
    return sp
end

function cached(Re, hL)
    path = joinpath(OUTDIR, "relam_$(tag(Re))_hL$(hL)_T$(round(Int, T_END)).jls")
    get(ENV, "REUSE", "0") == "1" && isfile(path) && return deserialize(path)
    r = relam_run(Re, hL, spinup(Re); T_end = T_END)
    serialize(path, r)
    return r
end

# the critical h/L from the collapse classification of a sweep
function critical(rs)
    surv = [r.h_over_L for r in rs if r.tke_end > 0.5]
    coll = [r.h_over_L for r in rs if r.tke_end < 0.1]
    (isempty(surv) || isempty(coll)) && return NaN
    a = maximum(surv); bs = filter(>(a), coll)
    return isempty(bs) ? NaN : sqrt(a * minimum(bs))
end

function figure(Re, rs)
    W, H = 1080, 430; pw, ph = 360, 300; y0 = 60
    pal = ["#1d4ed8", "#059669", "#d97706", "#dc2626", "#7c3aed", "#0891b2", "#be185d",
           "#65a30d", "#6b7280", "#92400e"]
    io = IOBuffer()
    println(io, """<?xml version="1.0" encoding="UTF-8"?>""")
    println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$W" height="$H" font-family="DejaVu Sans, sans-serif">""")
    println(io, """<rect width="100%" height="100%" fill="white"/>""")
    println(io, """<text x="$(W/2)" y="28" font-size="17" text-anchor="middle" font-weight="bold">Cooled open channel, Re_τ = $Re, $(join(grid(Re), "×")), σ-model + stability correction</text>""")
    # (a) TKE ratio (log) against time; legend to its right
    x0 = 80
    X(t) = x0 + t / T_END * pw; Y(e) = y0 + ph - (log10(clamp(e, 1e-4, 10)) + 4) / 5 * ph
    println(io, """<rect x="$x0" y="$y0" width="$pw" height="$ph" fill="none" stroke="#333"/>""")
    for t in 0:5:round(Int, T_END)
        println(io, """<text x="$(X(t))" y="$(y0+ph+18)" font-size="12" text-anchor="middle">$t</text>""")
    end
    for (e, lab) in ((1e-4, "1e-4"), (1e-3, "0.001"), (1e-2, "0.01"), (1e-1, "0.1"), (1.0, "1"), (10.0, "10"))
        println(io, """<line x1="$(x0-5)" y1="$(Y(e))" x2="$x0" y2="$(Y(e))" stroke="#333"/><text x="$(x0-8)" y="$(Y(e)+4)" font-size="12" text-anchor="end">$lab</text>""")
    end
    for e in (0.1, 0.5)
        println(io, """<line x1="$x0" y1="$(Y(e))" x2="$(x0+pw)" y2="$(Y(e))" stroke="#999" stroke-dasharray="4,4"/>""")
    end
    xl = x0 + pw + 14
    println(io, """<text x="$xl" y="$(y0+10)" font-size="12" font-weight="bold">h/L</text>""")
    for (i, r) in enumerate(rs)
        col = pal[mod1(i, length(pal))]
        pts = join(("$(X(t)),$(Y(e))" for (t, e) in zip(r.t, r.tke)), " ")
        println(io, """<polyline fill="none" stroke="$col" stroke-width="2" points="$pts"/>""")
        println(io, """<line x1="$xl" y1="$(y0+10+18i)" x2="$(xl+22)" y2="$(y0+10+18i)" stroke="$col" stroke-width="2.5"/><text x="$(xl+28)" y="$(y0+14+18i)" font-size="12">$(r.h_over_L)</text>""")
    end
    println(io, """<text x="$(x0+pw/2)" y="$(y0+ph+38)" font-size="14" text-anchor="middle">t u_τ/h after the switch-on</text>""")
    println(io, """<text x="$(x0-52)" y="$(y0+ph/2)" font-size="14" text-anchor="middle" transform="rotate(-90 $(x0-52),$(y0+ph/2))">resolved TKE / TKE(0)</text>""")
    println(io, """<text x="$(x0+8)" y="$(y0+20)" font-size="14" font-weight="bold">(a)</text>""")
    # (b) final TKE ratio and γ against h/L, with the reference thresholds
    x0 = 640
    hmax = max(1.1 * maximum(r.h_over_L for r in rs), 1.4)
    ymax = max(1.2, 0.2 * ceil(1.05 * maximum(r.tke_end for r in rs) / 0.2))
    XH(h) = x0 + h / hmax * pw; YE(e) = y0 + ph - e / ymax * ph
    println(io, """<rect x="$x0" y="$y0" width="$pw" height="$ph" fill="none" stroke="#333"/>""")
    for e in 0:0.2:ymax
        println(io, """<text x="$(x0-8)" y="$(YE(e)+4)" font-size="12" text-anchor="end">$(round(e, digits = 1))</text>""")
    end
    for h in 0:0.5:hmax
        println(io, """<text x="$(XH(h))" y="$(y0+ph+18)" font-size="12" text-anchor="middle">$h</text>""")
    end
    for e in (0.1, 0.5)
        println(io, """<line x1="$x0" y1="$(YE(e))" x2="$(x0+pw)" y2="$(YE(e))" stroke="#999" stroke-dasharray="4,4"/>""")
    end
    hN = H_NIEUWSTADT                       # Nieuwstadt (2005) at Re_τ = 360, κ convention
    println(io, """<line x1="$(XH(hN))" y1="$y0" x2="$(XH(hN))" y2="$(y0+ph)" stroke="#dc2626" stroke-dasharray="5,4"/><text x="$(XH(hN)+6)" y="$(YE(0.72))" font-size="11" fill="#dc2626">Nieuwstadt 2005</text><text x="$(XH(hN)+6)" y="$(YE(0.72)+14)" font-size="11" fill="#dc2626">(Re_τ = 360: 1.23 κ)</text>""")
    hF = Re / 100                           # Flores & Riley (2011): L⁺ = 100
    if hF < hmax
        println(io, """<line x1="$(XH(hF))" y1="$y0" x2="$(XH(hF))" y2="$(y0+ph)" stroke="#7c3aed" stroke-dasharray="2,3"/><text x="$(XH(hF)-4)" y="$(y0+ph/2+40)" font-size="11" fill="#7c3aed" text-anchor="end">L⁺ = 100</text><text x="$(XH(hF)-4)" y="$(y0+ph/2+54)" font-size="11" fill="#7c3aed" text-anchor="end">(Flores &amp; Riley)</text>""")
    else
        println(io, """<text x="$(XH(hN)+6)" y="$(YE(0.36))" font-size="11" fill="#7c3aed">L⁺ = 100 (Flores &amp; Riley)</text><text x="$(XH(hN)+6)" y="$(YE(0.36)+14)" font-size="11" fill="#7c3aed">at h/L = $(round(hF, digits = 2)), off scale</text>""")
    end
    for r in rs
        println(io, """<circle cx="$(XH(r.h_over_L))" cy="$(YE(min(r.tke_end, ymax)))" r="5" fill="#1d4ed8"/><rect x="$(XH(r.h_over_L)-4)" y="$(YE(r.γ_end)-4)" width="8" height="8" fill="none" stroke="#059669" stroke-width="1.5"/>""")
    end
    xg = XH(hN) + 6; yg = y0 + 30
    println(io, """<text x="$xg" y="$yg" font-size="12" fill="#1d4ed8">● TKE(T)/TKE(0)</text><text x="$xg" y="$(yg+16)" font-size="12" fill="#059669">□ γ of the lower half at T</text>""")
    println(io, """<text x="$(x0+pw/2)" y="$(y0+ph+38)" font-size="14" text-anchor="middle">h/L</text>""")
    println(io, """<text x="$(x0+8)" y="$(y0+20)" font-size="14" font-weight="bold">(b)</text>""")
    println(io, "</svg>")
    path = joinpath(OUTDIR, "relaminarisation_$(tag(Re)).svg")
    write(path, String(take!(io)))
    println("wrote ", path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    for Re in RES
        sp = spinup(Re)
        @printf("\nRe_τ = %d on %s: neutral spin-up to %.0f h/u_τ: TKE %.3f u_τ², ⟨τ_w⟩ %.3f u_τ², U_b %.2f u_τ\n",
                Re, join(grid(Re), "×"), T_SPIN, sp.tke, sp.τ_wall, sp.Ub)
        rs = [cached(Re, hL) for hL in HLS]
        @printf("%6s %8s %8s %10s %8s %8s %8s %8s\n", "h/L", "L⁺", "Q", "TKE(T)/0", "γ(T)", "τ_w(T)", "steps", "wall s")
        for r in rs
            @printf("%6.2f %8.1f %8.3f %10.3f %8.3f %8.3f %8d %8.0f   %s\n", r.h_over_L, r.crit.L_plus,
                    relam_heat_flux(r.h_over_L), r.tke_end, r.γ_end, r.τ_wall[end],
                    r.nsteps, r.wall_time,
                    r.tke_end < 0.1 ? "collapsed" : r.tke_end > 0.5 ? "survived" : "intermediate")
        end
        hc = critical(rs)
        @printf("critical h/L = %.3f (Nieuwstadt 2005: 1.23 κ = %.3f at Re_τ = 360, ratio %.2f); L⁺_c = %.1f (Flores & Riley 2011: ≈ 100, ratio %.2f)\n",
                hc, H_NIEUWSTADT, hc / H_NIEUWSTADT, Re / hc, Re / hc / 100)
        figure(Re, rs)
    end
end
