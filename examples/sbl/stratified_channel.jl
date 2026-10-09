# =========================================================================
# Stably stratified plane channel against DNS — P55-f (#191)
#
# The wall-resolved channel of García-Villalba & del Álamo (2011, Phys.
# Fluids 23, 045104): isothermal walls, the top hotter than the bottom,
# friction Richardson number Ri_τ = gΔθδ/(θ₀u_τ²), Pr = 0.71, driven by a
# constant pressure gradient (u_τ = δ = 1). Built with
# `channel_case(bottom_bc = :noslip, θ_walls = …)` (P54-d builder, P59-a
# NoSlipWall), the SGS closure the plane-averaged dynamic Smagorinsky model
# (Germano et al. 1991; no wall damping needed) with and without
# `StabilityCorrected` (P55-e, Brown et al. 1994, anisotropic length scales:
# the Berlin default), stress-form momentum divergence.
#
# Re_τ = 180, the lowest Re_τ of the DNS: the DNS Re_τ of 550 is not
# affordable wall-resolved here (≈ 3× the points per direction), and at
# Re_τ = 180 the DNS bulk Reynolds and Nusselt numbers are tabulated
# (examples/sbl/data/stratified_channel_gv11_re180.csv) rather than
# digitised. Ri_τ ∈ {0 (passive θ), 18, 60}. As in the DNS, the
# stratification is switched on in a developed neutral channel (a neutral
# spin-up, then the run, statistics from averaging.from). The runs are the
# sweep examples/sbl/configs/stratified_channel_sweep.toml (the `sbl` run
# type), one run folder per Ri_τ and closure.
#
# This script only reads the run folders (statistics.toml): it prints U_b⁺,
# Nu and the momentum-budget residual against the DNS and makes the figure.
#
#   julia --project -m Aeolus run examples/sbl/configs/stratified_channel_sweep.toml      # 32×32×48, T = 40
#   julia --project examples/sbl/stratified_channel.jl runs/stratified_channel_re180_*
#
# writes examples/sbl/output/stratified_channel_<N>.svg
# =========================================================================

include(joinpath(@__DIR__, "runs.jl"))
using Printf

const RUNS = sbl_runs(ARGS, "julia --project examples/sbl/stratified_channel.jl <run folder>…")
const NN = Tuple(cfgval(RUNS[1], "grid.size"))
const T_END = cfgval(RUNS[1], "stepper.end_time")
const T_AVG = cfgval(RUNS[1], "averaging.from")
const T_SPIN = cfgval(RUNS[1], "case.spinup_time")
const DNS = sbl_read_csv(joinpath(SBL_DATA, "stratified_channel_gv11_re180.csv"))
const OUTDIR = mkpath(SBL_OUT)
const TAG = join(NN, "x")

let
    results = [r.stats for r in RUNS]
    dns(Ri, col) = sbl_csv_value(DNS, Ri, col)
    @printf("Stratified channel, Re_τ = 180, %s, neutral spin-up to %.0f, averaged over %.0f-%.0f δ/u_τ (DNS: García-Villalba & del Álamo 2011)\n",
            TAG, T_SPIN, T_AVG, T_END)
    @printf("%6s %-8s %8s %8s %7s %7s %8s %8s %7s\n", "Ri_τ", "closure", "U_b⁺", "DNS", "Nu", "DNS", "⟨τ_w⟩", "budget", "ΔU_b")
    for r in results
        @printf("%6.0f %-8s %8.3f %8.3f %7.3f %7.2f %8.3f %8.4f %7.3f\n", r.Ri, r.stab, r.Ub,
                dns(r.Ri, "U_b_plus"), r.Nu, dns(r.Ri, "Nu"), r.τ_wall, r.budget, r.ΔUb)
    end
    # the figure: (a) mean profiles U⁺(y⁺) of the lower half, (b) U_b⁺ and Nu against Ri_τ
    W, H = 1060, 460; pw, ph = 380, 330
    cols = Dict(0.0 => "#2563eb", 18.0 => "#059669", 60.0 => "#dc2626")
    io = IOBuffer()
    println(io, """<?xml version="1.0" encoding="UTF-8"?>""")
    println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$W" height="$H" font-family="DejaVu Sans, sans-serif">""")
    println(io, """<rect width="100%" height="100%" fill="white"/>""")
    println(io, """<text x="$(W/2)" y="26" font-size="17" text-anchor="middle" font-weight="bold">Stratified channel, Re_τ = 180, $TAG, dynamic Smagorinsky: solid ± correction (dashed: none)</text>""")
    x0, y0 = 80, 55
    X(yp) = x0 + (log10(yp) - log10(1)) / (log10(180) - log10(1)) * pw; Y(u) = y0 + ph - u / 30 * ph
    println(io, """<defs><clipPath id="pa"><rect x="$x0" y="$y0" width="$pw" height="$ph"/></clipPath></defs>""")
    println(io, """<rect x="$x0" y="$y0" width="$pw" height="$ph" fill="none" stroke="#333"/>""")
    for yp in (1, 10, 100)
        println(io, """<line x1="$(X(yp))" y1="$(y0+ph)" x2="$(X(yp))" y2="$(y0+ph+5)" stroke="#333"/><text x="$(X(yp))" y="$(y0+ph+19)" font-size="12" text-anchor="middle">$yp</text>""")
    end
    for u in 0:5:30
        println(io, """<line x1="$(x0-5)" y1="$(Y(u))" x2="$x0" y2="$(Y(u))" stroke="#333"/><text x="$(x0-8)" y="$(Y(u)+4)" font-size="12" text-anchor="end">$u</text>""")
    end
    for r in results
        n2 = length(r.zc) ÷ 2
        Us = 0.5 .* (r.U[1:n2] .+ reverse(r.U)[1:n2])            # fold the two halves
        pts = join(("$(X(180z)),$(Y(u))" for (z, u) in zip(r.zc[1:n2], Us)), " ")
        dash = r.stab == "none" ? """ stroke-dasharray="6,4\"""" : ""
        println(io, """<polyline clip-path="url(#pa)" fill="none" stroke="$(cols[r.Ri])" stroke-width="2"$dash points="$pts"/>""")
    end
    for (i, Ri) in enumerate((0.0, 18.0, 60.0))
        println(io, """<line x1="$(x0+15)" y1="$(y0+20i)" x2="$(x0+40)" y2="$(y0+20i)" stroke="$(cols[Ri])" stroke-width="2"/><text x="$(x0+47)" y="$(y0+20i+4)" font-size="12">Ri_τ = $(round(Int, Ri)) (DNS U_b⁺ = $(dns(Ri, "U_b_plus")))</text>""")
    end
    println(io, """<text x="$(x0+pw/2)" y="$(y0+ph+40)" font-size="14" text-anchor="middle">y⁺ (lower and upper halves folded)</text>""")
    println(io, """<text x="$(x0-46)" y="$(y0+ph/2)" font-size="14" text-anchor="middle" transform="rotate(-90 $(x0-46),$(y0+ph/2))">⟨U⟩/u_τ</text>""")
    println(io, """<text x="$(x0+6)" y="$(y0+ph-8)" font-size="14" font-weight="bold">(a)</text>""")
    # (b) Nu and U_b against Ri
    x0 = 580
    XR(Ri) = x0 + Ri / 70 * pw; YN(nu) = y0 + ph - nu / 8 * ph; YU(u) = y0 + ph - (u - 12) / 16 * ph
    println(io, """<rect x="$x0" y="$y0" width="$pw" height="$ph" fill="none" stroke="#333"/>""")
    for Ri in 0:10:70
        println(io, """<line x1="$(XR(Ri))" y1="$(y0+ph)" x2="$(XR(Ri))" y2="$(y0+ph+5)" stroke="#333"/><text x="$(XR(Ri))" y="$(y0+ph+19)" font-size="12" text-anchor="middle">$Ri</text>""")
    end
    for nu in 0:2:8
        println(io, """<text x="$(x0-8)" y="$(YN(nu)+4)" font-size="12" text-anchor="end" fill="#7c3aed">$nu</text>""")
    end
    for u in 12:4:28
        println(io, """<text x="$(x0+pw+8)" y="$(YU(u)+4)" font-size="12" fill="#b45309">$u</text>""")
    end
    dRi = [r[1] for r in DNS.rows]
    println(io, """<polyline fill="none" stroke="#7c3aed" stroke-width="1" stroke-dasharray="2,3" points="$(join(("$(XR(Ri)),$(YN(dns(Ri, "Nu")))" for Ri in dRi), " "))"/>""")
    println(io, """<polyline fill="none" stroke="#b45309" stroke-width="1" stroke-dasharray="2,3" points="$(join(("$(XR(Ri)),$(YU(dns(Ri, "U_b_plus")))" for Ri in dRi), " "))"/>""")
    for Ri in dRi
        println(io, """<rect x="$(XR(Ri)-4)" y="$(YN(dns(Ri, "Nu"))-4)" width="8" height="8" fill="none" stroke="#7c3aed"/><rect x="$(XR(Ri)-4)" y="$(YU(dns(Ri, "U_b_plus"))-4)" width="8" height="8" fill="none" stroke="#b45309"/>""")
    end
    for r in results
        f = r.stab == "none" ? "white" : nothing
        fillN = r.stab == "none" ? "white" : "#7c3aed"; fillU = r.stab == "none" ? "white" : "#b45309"
        println(io, """<circle cx="$(XR(r.Ri))" cy="$(YN(r.Nu))" r="4.5" fill="$fillN" stroke="#7c3aed" stroke-width="1.5"/><circle cx="$(XR(r.Ri))" cy="$(YU(r.Ub))" r="4.5" fill="$fillU" stroke="#b45309" stroke-width="1.5"/>""")
    end
    println(io, """<text x="$(x0+pw/2)" y="$(y0+ph+40)" font-size="14" text-anchor="middle">Ri_τ</text>""")
    println(io, """<text x="$(x0-40)" y="$(y0+ph/2)" font-size="14" text-anchor="middle" fill="#7c3aed" transform="rotate(-90 $(x0-40),$(y0+ph/2))">Nu</text>""")
    println(io, """<text x="$(x0+pw+42)" y="$(y0+ph/2)" font-size="14" text-anchor="middle" fill="#b45309" transform="rotate(90 $(x0+pw+42),$(y0+ph/2))">U_b⁺</text>""")
    println(io, """<text x="$(x0+6)" y="$(y0+18)" font-size="14" font-weight="bold">(b)</text>""")
    println(io, """<text x="$(x0+pw-8)" y="$(y0+18)" font-size="11" text-anchor="end" fill="#555">squares: DNS (GV11, Table I of Ahmed et al. 2021)</text>""")
    println(io, """<text x="$(x0+pw-8)" y="$(y0+33)" font-size="11" text-anchor="end" fill="#555">filled: corrected; open: none</text>""")
    println(io, "</svg>")
    path = joinpath(OUTDIR, "stratified_channel_$(TAG).svg")
    write(path, String(take!(io)))
    println("wrote ", path)
end
