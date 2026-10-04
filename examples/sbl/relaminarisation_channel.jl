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
# References: Nieuwstadt (2005), h/L ≈ 1.23 at Re_τ = 360 (target ±30 %);
# Flores & Riley (2011), collapse for L⁺ ≲ 100 (target within a factor
# 1.5); Donda et al. (2015, QJRMS 141, 2137) for the transient collapse.
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
    a = maximum(surv); b = minimum(filter(>(a), coll); init = NaN)
    return isnan(b) ? NaN : sqrt(a * b)
end

function figure(Re, rs)
    W, H = 960, 420; pw, ph = 360, 300; y0 = 60
    pal = ["#2563eb", "#059669", "#d97706", "#dc2626", "#7c3aed", "#0891b2", "#be185d"]
    io = IOBuffer()
    println(io, """<?xml version="1.0" encoding="UTF-8"?>""")
    println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$W" height="$H" font-family="DejaVu Sans, sans-serif">""")
    println(io, """<rect width="100%" height="100%" fill="white"/>""")
    println(io, """<text x="$(W/2)" y="28" font-size="17" text-anchor="middle" font-weight="bold">Cooled open channel, Re_τ = $Re, $(join(grid(Re), "×")), σ-model + stability correction</text>""")
    # (a) TKE ratio (log) against time
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
    for (i, r) in enumerate(rs)
        pts = join(("$(X(t)),$(Y(e))" for (t, e) in zip(r.t, r.tke)), " ")
        println(io, """<polyline fill="none" stroke="$(pal[mod1(i, 7)])" stroke-width="2" points="$pts"/>""")
        println(io, """<line x1="$(x0+pw-120)" y1="$(y0+14+16i)" x2="$(x0+pw-98)" y2="$(y0+14+16i)" stroke="$(pal[mod1(i, 7)])" stroke-width="2"/><text x="$(x0+pw-92)" y="$(y0+18+16i)" font-size="12">h/L = $(r.h_over_L)</text>""")
    end
    println(io, """<text x="$(x0+pw/2)" y="$(y0+ph+38)" font-size="14" text-anchor="middle">t u_τ/h after the switch-on</text>""")
    println(io, """<text x="$(x0-52)" y="$(y0+ph/2)" font-size="14" text-anchor="middle" transform="rotate(-90 $(x0-52),$(y0+ph/2))">resolved TKE / TKE(0)</text>""")
    println(io, """<text x="$(x0+6)" y="$(y0+18)" font-size="14" font-weight="bold">(a)</text>""")
    # (b) final TKE ratio and γ against h/L
    x0 = 560
    hmax = 1.1 * maximum(r.h_over_L for r in rs)
    XH(h) = x0 + h / hmax * pw; YE(e) = y0 + ph - e / 1.2 * ph
    println(io, """<rect x="$x0" y="$y0" width="$pw" height="$ph" fill="none" stroke="#333"/>""")
    for e in 0:0.2:1.2
        println(io, """<text x="$(x0-8)" y="$(YE(e)+4)" font-size="12" text-anchor="end">$(round(e, digits = 1))</text>""")
    end
    for h in 0:0.5:hmax
        println(io, """<text x="$(XH(h))" y="$(y0+ph+18)" font-size="12" text-anchor="middle">$h</text>""")
    end
    hN = 1.23                               # Nieuwstadt (2005) at Re_τ = 360
    println(io, """<line x1="$(XH(hN))" y1="$y0" x2="$(XH(hN))" y2="$(y0+ph)" stroke="#dc2626" stroke-dasharray="5,4"/><text x="$(XH(hN)+4)" y="$(y0+ph-8)" font-size="11" fill="#dc2626">Nieuwstadt 2005 (Re_τ = 360)</text>""")
    hF = Re / 100                           # Flores & Riley (2011): L⁺ = 100
    hF < hmax && println(io, """<line x1="$(XH(hF))" y1="$y0" x2="$(XH(hF))" y2="$(y0+ph)" stroke="#7c3aed" stroke-dasharray="2,3"/><text x="$(XH(hF)+4)" y="$(y0+ph-24)" font-size="11" fill="#7c3aed">L⁺ = 100 (Flores &amp; Riley 2011)</text>""")
    for r in rs
        println(io, """<circle cx="$(XH(r.h_over_L))" cy="$(YE(min(r.tke_end, 1.2)))" r="5" fill="#2563eb"/><rect x="$(XH(r.h_over_L)-4)" y="$(YE(r.γ_end)-4)" width="8" height="8" fill="none" stroke="#059669" stroke-width="1.5"/>""")
    end
    println(io, """<text x="$(x0+30)" y="$(y0+18)" font-size="12" fill="#2563eb">● TKE(T)/TKE(0)</text><text x="$(x0+30)" y="$(y0+34)" font-size="12" fill="#059669">□ γ, lower half, at T</text>""")
    println(io, """<text x="$(x0+pw/2)" y="$(y0+ph+38)" font-size="14" text-anchor="middle">h/L</text>""")
    println(io, """<text x="$(x0+6)" y="$(y0+18)" font-size="14" font-weight="bold">(b)</text>""")
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
        @printf("%6s %8s %8s %8s %10s %8s %8s %8s %8s\n", "h/L", "L⁺", "Ri_b", "Q", "TKE(T)/0", "γ(T)", "τ_w(T)", "steps", "wall s")
        for r in rs
            @printf("%6.2f %8.1f %8.3f %8.3f %10.3f %8.3f %8.3f %8d %8.0f   %s\n", r.h_over_L, r.crit.L_plus,
                    r.crit.Ri_b, relam_heat_flux(r.h_over_L), r.tke_end, r.γ_end, r.τ_wall[end],
                    r.nsteps, r.wall_time,
                    r.tke_end < 0.1 ? "collapsed" : r.tke_end > 0.5 ? "survived" : "intermediate")
        end
        hc = critical(rs)
        @printf("critical h/L = %.3f (Nieuwstadt 2005: 1.23 at Re_τ = 360, ratio %.2f); L⁺_c = %.1f (Flores & Riley 2011: ≈ 100, ratio %.2f)\n",
                hc, hc / 1.23, Re / hc, Re / hc / 100)
        figure(Re, rs)
    end
end
