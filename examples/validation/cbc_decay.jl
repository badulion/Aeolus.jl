# =========================================================================
# Decaying grid turbulence of Comte-Bellot & Corrsin (1971) — Phase 54 (#168)
#
# LES of the CBC wind-tunnel experiment (M = 5.08 cm, U0 = 10 m/s) in a
# periodic box of side 11 M at 64³, started from the measured spectrum at
# tU0/M = 42 and compared with the measured spectra at 98 and 171, for the
# dynamic Smagorinsky model, static Smagorinsky (C_s = 0.17) and no model.
# The harness and the data are shared with the CI test (32³):
# test/cbc_benchmark_driver.jl, test/data/cbc1971_table3.txt,
# test/era6_cbc_decay.jl.
#
#   julia --project=. examples/validation/cbc_decay.jl [N]
#
# writes examples/validation/output/cbc_decay_<N>.svg
# =========================================================================

using Aeolus
using Printf
include(joinpath(@__DIR__, "..", "..", "test", "cbc_benchmark_driver.jl"))

N = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 64

closures = (:dynamic, :smagorinsky, :none)
res = Dict{Symbol,Any}()
for c in closures
    res[c] = cbc_benchmark(c; N)
    println(cbc_summary(String(c), res[c]))
end

# --- figure: E(k) at the three stations, log-log --------------------------
function write_figure(path, res, N)
    W, H = 760, 560
    l, r, t, b = 80, 200, 30, 60
    r0 = res[:dynamic]
    kc = r0.k[N ÷ 2 + 1]
    kmin, kmax = r0.k[2] / 1.3, 2.2 * kc
    k_tab, E_tab, _ = cbc_table()
    Emin, Emax = 1e-7, 1e-2
    X(k) = l + (W - l - r) * (log10(k) - log10(kmin)) / (log10(kmax) - log10(kmin))
    Y(E) = t + (H - t - b) * (1 - (log10(E) - log10(Emin)) / (log10(Emax) - log10(Emin)))
    shade = ("#1b6ca8", "#c8553d", "#4a7c3a")           # stations 42, 98, 171
    io = IOBuffer()
    println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$W" height="$H" font-family="DejaVu Sans, sans-serif" font-size="13">""")
    println(io, """<rect width="$W" height="$H" fill="white"/>""")
    # axes and decade grid
    for e in -7:-2
        y = Y(10.0^e)
        println(io, """<line x1="$l" y1="$y" x2="$(W - r)" y2="$y" stroke="#e4e4e4"/>""")
        println(io, """<text x="$(l - 8)" y="$(y + 4)" text-anchor="end">1e$e</text>""")
    end
    for kt in (10, 20, 50, 100, 200, 500)
        (kmin <= kt <= kmax) || continue
        x = X(kt)
        println(io, """<line x1="$x" y1="$t" x2="$x" y2="$(H - b)" stroke="#e4e4e4"/>""")
        println(io, """<text x="$x" y="$(H - b + 18)" text-anchor="middle">$kt</text>""")
    end
    println(io, """<rect x="$l" y="$t" width="$(W - l - r)" height="$(H - t - b)" fill="none" stroke="black"/>""")
    println(io, """<text x="$((W - r + l) / 2)" y="$(H - 15)" text-anchor="middle">k  [rad/m]</text>""")
    println(io, """<text transform="translate(20,$((H - b + t) / 2)) rotate(-90)" text-anchor="middle">E(k)  [m³/s²]</text>""")
    xc = X(kc)
    println(io, """<line x1="$xc" y1="$t" x2="$xc" y2="$(H - b)" stroke="#888" stroke-dasharray="4,4"/>""")
    println(io, """<text x="$(xc + 4)" y="$(t + 14)" fill="#555">k_c</text>""")
    # CBC table 3 (points) and its interpolant (thin line)
    for s in 1:3
        kk = exp.(range(log(kmin), log(kmax), length = 200))
        pts = join(("$(round(X(k), digits=1)),$(round(Y(cbc_spectrum(s)(k)), digits=1))"
                    for k in kk if Emin < cbc_spectrum(s)(k) < Emax), " ")
        println(io, """<polyline points="$pts" fill="none" stroke="$(shade[s])" stroke-width="1" stroke-opacity="0.45"/>""")
        for (k, E) in zip(k_tab, E_tab[:, s])
            (isnan(E) || !(kmin <= k <= kmax)) && continue
            println(io, """<circle cx="$(X(k))" cy="$(Y(E))" r="4" fill="white" stroke="$(shade[s])" stroke-width="1.5"/>""")
        end
    end
    # LES spectra: dynamic (solid), Smagorinsky (dashed), no model (dotted)
    style = Dict(:dynamic => "", :smagorinsky => "stroke-dasharray=\"7,4\"", :none => "stroke-dasharray=\"1.5,3\"")
    for c in (:dynamic, :smagorinsky, :none), s in 1:3
        rr = res[c]
        (c != :dynamic && s == 1) && continue           # identical start
        pts = join(("$(round(X(rr.k[n+1]), digits=1)),$(round(Y(rr.E_les[s][n+1]), digits=1))"
                    for n in 1:N÷2 if Emin < rr.E_les[s][n+1] < Emax), " ")
        println(io, """<polyline points="$pts" fill="none" stroke="$(shade[s])" stroke-width="2" $(style[c])/>""")
    end
    # legend
    lx = W - r + 15; ly = t + 10
    for (i, s) in enumerate(("tU₀/M = 42", "tU₀/M = 98", "tU₀/M = 171"))
        println(io, """<rect x="$lx" y="$(ly + 20(i-1))" width="14" height="10" fill="$(shade[i])"/>""")
        println(io, """<text x="$(lx + 20)" y="$(ly + 9 + 20(i-1))">$s</text>""")
    end
    ly += 75
    for (i, (lab, st)) in enumerate((("LES, dynamic", ""), ("LES, C_s = 0.17", "stroke-dasharray=\"7,4\""),
                                     ("LES, no model", "stroke-dasharray=\"1.5,3\"")))
        y = ly + 20(i - 1)
        println(io, """<line x1="$lx" y1="$y" x2="$(lx + 30)" y2="$y" stroke="black" stroke-width="2" $st/>""")
        println(io, """<text x="$(lx + 36)" y="$(y + 4)">$lab</text>""")
    end
    y = ly + 62
    println(io, """<circle cx="$(lx + 15)" cy="$y" r="4" fill="white" stroke="black" stroke-width="1.5"/>""")
    println(io, """<line x1="$lx" y1="$y" x2="$(lx + 30)" y2="$y" stroke="black" stroke-width="1" stroke-opacity="0.45"/>""")
    println(io, """<text x="$(lx + 36)" y="$(y + 4)">CBC 1971, table 3</text>""")
    println(io, """<text x="$lx" y="$(y + 30)" font-size="11">$(N)³, box 11M, $(r0.nseeds) seeds</text>""")
    println(io, "</svg>")
    mkpath(dirname(path))
    write(path, String(take!(io)))
    return path
end

out = write_figure(joinpath(@__DIR__, "output", "cbc_decay_$(N).svg"), res, N)
println("wrote ", out)
