# =========================================================================
# Surface-mounted cube in a channel: reattachment length behind the cube
# for the Smagorinsky, TKE and Reynolds-stress SGS closures — Era VI,
# P56-e (#192)
#
# Martinuzzi & Tropea (1993), cube of height H in a channel of height 2H
# at Re_H = U_b H/ν = 40 000: the mean flow separates at the cube's
# leading edges and reattaches on the floor behind it at x_R/H = 1.612
# (measured from the rear face, on the centre line); ahead of the cube the
# horseshoe vortex separates the floor flow at x_F/H = 1.04 upstream of
# the front face.
#
# Set-up (an LES at 16 cells per H, as the issue asks):
#
#   * box 10H × 6H × 2H, periodic in x and y (Shah & Ferziger 1997 use a
#     periodic box of the same length), the cube at x = 3H–4H, centred in
#     y; the floor is a one-cell solid layer below z = 0 and the top a
#     free-slip lid at z = 2H (the channel's upper wall as a symmetry
#     plane; stated simplification);
#   * the cube and the floor are the immersed boundary of `IBM`
#     (direct forcing, `implicit_ibm_correction!` with τ = 0, at every RK3
#     substage through `step_rk3!`'s `post_predictor!`): a no-slip wall
#     with the SGS stress as the only wall model. The P57-d facade wall
#     model (#179) the issue names is not in this branch's stack (stated
#     exception);
#   * a uniform streamwise body force with a PI controller holds the bulk
#     velocity at U_b = 1; ν = U_b H/40 000;
#   * SSP-RK3, centred advection, CFL 0.4; the closures as in Berlin: the
#     TKE and Reynolds-stress closures with the wall cap κ·IBM.wall_distance
#     and Moeng's 3.9 at the first level above a surface, zero inside the
#     solid; Smagorinsky with C_s = 0.1 (the wall-bounded value of the
#     Berlin case).
#
# Statistics: after a spin-up of `T_SPIN` H/U the velocity is averaged
# over `T_AVG` H/U; x_R is where the mean u at the first fluid level
# (z = Δz/2) on the centre line turns from negative to positive behind
# the cube, x_F where it turns negative ahead of it.
#
#   julia --project=. examples/sgs/surface_mounted_cube.jl run <model>     # smagorinsky | tke | rsm
#   julia --project=. examples/sgs/surface_mounted_cube.jl plot
#
# Environment: CUBE_N (cells per H, 16), CUBE_T_SPIN (30), CUBE_T_AVG (30).
# `run` writes examples/sgs/output/cube_<model>_<N>.csv (x/H and the mean
# centre-line floor u) and prints x_R, x_F and the cost; `plot` writes
# examples/sgs/output/surface_mounted_cube_<N>.svg from the CSV files.
# =========================================================================

using Aeolus
using Aeolus.Fields: interior_view
using Aeolus.IBM: SolidMask, mark_solid!, implicit_ibm_correction!, wall_distance, first_level_cells
using Printf
import Random

const NH = parse(Int, get(ENV, "CUBE_N", "16"))
const T_SPIN = parse(Float64, get(ENV, "CUBE_T_SPIN", "30"))
const T_AVG = parse(Float64, get(ENV, "CUBE_T_AVG", "30"))
const LX, LY, LZ = 10.0, 6.0, 2.0
const XC0, XC1 = 3.0, 4.0                     # cube faces (front, rear)
const RE_H = 40_000.0
const XR_MT93 = 1.612
const XF_MT93 = 1.04
const OUT = joinpath(@__DIR__, "output")
const MODELS = (:smagorinsky, :tke, :rsm)

function build(model::Symbol)
    h = 1 / NH
    mesh = RegularGrid(; size = (round(Int, LX * NH), round(Int, LY * NH), round(Int, LZ * NH) + 1),
                       extent = (LX, LY, LZ + h), origin = (0.0, 0.0, -h), halo = (3, 3, 3))
    bcs = FieldBoundaryConditions(:periodic, :periodic, :wall)
    sim = BoussinesqSimulation(mesh; ν = 1 / RE_H, bcs = bcs)
    sm = SolidMask(mesh)
    yc = LY / 2
    mark_solid!(sm, (x, y, z) -> z < 0 || (XC0 < x < XC1 && abs(y - yc) < 0.5 && z < 1))
    mf = Field(CellCenter(), mesh)
    mf.data .= sm.mask
    fill_halos!(mf, FieldBoundaryConditions(:periodic, :periodic, :noflux))
    mask = mf.data .> 0.5
    d = wall_distance(sm; bottom = false)
    fl = first_level_cells(sm; bottom = false)
    closure = model === :smagorinsky ? StaticSmagorinsky(0.1) :
              model === :tke ? TKEClosure(mesh; mask = mask, wall_distance = d, first_level = fl) :
              model === :rsm ? ReynoldsStressClosure(mesh; mask = mask, wall_distance = d, first_level = fl) :
              throw(ArgumentError("model must be one of $MODELS"))
    # initial state: U_b plus 10 % noise in the fluid
    rng = Random.Xoshiro(1993)
    interior_view(sim.u) .= 1 .+ 0.1 .* randn(rng, size(interior_view(sim.u)))
    interior_view(sim.v) .= 0.1 .* randn(rng, size(interior_view(sim.v)))
    interior_view(sim.w) .= 0.1 .* randn(rng, size(interior_view(sim.w)))
    implicit_ibm_correction!(sim, sm)
    return sim, sm, closure
end

# bulk velocity over the fluid
function bulk_u(sim, sm)
    m = sim.mesh
    s = 0.0; n = 0
    u = sim.u.data; mk = sm.mask
    @inbounds for k in m.Hz+1:m.Hz+m.Nz, j in m.Hy+1:m.Hy+m.Ny, i in m.Hx+1:m.Hx+m.Nx
        mk[i, j, k] && continue
        s += u[i, j, k]; n += 1
    end
    return s / n
end

function run(model::Symbol)
    sim, sm, closure = build(model)
    m = sim.mesh
    model === :tke && initialize_tke!(closure, sim; e_min = 1e-4)
    model === :rsm && initialize_reynolds_stress!(closure, sim; e_min = 1e-4)
    Fi = Ref(0.0); Fp = Ref(0.0)
    force = (s, Δt) -> (interior_view(s.tu) .+= Fi[] + Fp[]; nothing)
    ibm = (s, Δt) -> (implicit_ibm_correction!(s, sm); nothing)
    kf = 2                                              # first fluid level (z = Δz/2)
    jc = (m.Ny ÷ 2, m.Ny ÷ 2 + 1)                      # the two cells about the centre line
    ufl = zeros(m.Nx); tavg = 0.0
    U = Float64[]; nsteps = 0
    wall = @elapsed while sim.t < T_SPIN + T_AVG - 1e-9
        ub = bulk_u(sim, sm)
        umax = maximum(f -> maximum(abs, interior_view(f)), (sim.u, sim.v, sim.w))
        dt = min(0.4 * m.dx / umax, T_SPIN + T_AVG - sim.t)
        # PI controller on U_b = 1: F = K_p e + K_i ∫e dt (K_p = 2, K_i = 0.5 per H/U)
        Fp[] = 2.0 * (1 - ub)
        Fi[] += 0.5 * dt * (1 - ub)
        ts = @elapsed step_rk3!(sim, dt; sgs = closure, body_force! = force, post_predictor! = ibm)
        nsteps += 1
        push!(U, ub)
        if !all(isfinite, interior_view(sim.u))
            error("$model: non-finite velocity at t = $(sim.t)")
        end
        if sim.t > T_SPIN
            u = interior_view(sim.u)
            # u at the cell centre in x: the average of the two faces
            for i in 1:m.Nx
                ip = i == m.Nx ? 1 : i + 1
                ufl[i] += dt * sum((u[i, j, kf] + u[ip, j, kf]) / 2 for j in jc) / 2
            end
            tavg += dt
        end
        nsteps % 200 == 0 && (@printf("  %s t = %.2f  dt = %.4f  U_b = %.4f  F = %.4f  %.2f s/step\n",
                                      model, sim.t, dt, ub, Fi[] + Fp[], ts); flush(stdout))
    end
    ufl ./= tavg
    x = ((1:m.Nx) .- 0.5) ./ NH
    xR, xF = reattachment(x, ufl)
    @printf("RESULT %s N = %d: x_R/H = %.3f (MT93 %.3f), x_F/H = %.3f (MT93 %.2f); %d steps, %.0f s (%.2f s/step); mean U_b %.4f; force %.4f\n",
            model, NH, xR, XR_MT93, xF, XF_MT93, nsteps, wall, wall / nsteps,
            sum(U[end ÷ 2:end]) / length(U[end ÷ 2:end]), Fi[] + Fp[])
    mkpath(OUT)
    open(joinpath(OUT, "cube_$(model)_$(NH).csv"), "w") do io
        println(io, "# $(model) N=$(NH) x_R/H=$(xR) x_F/H=$(xF) steps=$(nsteps) wall_s=$(wall) T_spin=$(T_SPIN) T_avg=$(T_AVG)")
        println(io, "x_over_H,u_mean_floor")
        for i in eachindex(x)
            println(io, x[i], ",", ufl[i])
        end
    end
    return xR, xF
end

"""
    reattachment(x, u) -> (x_R/H, x_F/H)

Behind the cube: the first sign change of the mean floor u from negative
to positive downstream of the rear face (linear interpolation), measured
from the rear face. Ahead of it: the first point upstream of the front
face where u turns negative, measured from the front face.
"""
function reattachment(x, u)
    xR = NaN
    for i in findfirst(>(XC1), x):length(x)-1
        if u[i] < 0 && u[i+1] >= 0
            xR = x[i] + (x[i+1] - x[i]) * (-u[i]) / (u[i+1] - u[i]) - XC1
            break
        end
    end
    xF = NaN
    i0 = findlast(<(XC0), x)
    if u[i0] < 0
        for i in i0:-1:2
            if u[i-1] >= 0 && u[i] < 0
                xF = XC0 - (x[i-1] + (x[i] - x[i-1]) * u[i-1] / (u[i-1] - u[i]))
                break
            end
        end
    end
    return xR, xF
end

function plot()
    W, Hh = 760, 420
    l, r, t, b = 70, 190, 30, 55
    X(x) = l + (W - l - r) * x / LX
    Y(u) = t + (Hh - t - b) * (1 - (u + 0.4) / 1.0)
    col = Dict(:smagorinsky => "#8c8c8c", :tke => "#1b6ca8", :rsm => "#c8553d")
    lab = Dict(:smagorinsky => "Smagorinsky", :tke => "TKE", :rsm => "Reynolds stress")
    io = IOBuffer()
    println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$W" height="$Hh" font-family="DejaVu Sans, sans-serif" font-size="12">""")
    println(io, """<rect width="$W" height="$Hh" fill="white"/>""")
    println(io, """<rect x="$(X(XC0))" y="$t" width="$(X(XC1) - X(XC0))" height="$(Hh - t - b)" fill="#dddddd"/>""")
    for u in -0.4:0.2:0.6
        println(io, """<line x1="$l" y1="$(Y(u))" x2="$(W - r)" y2="$(Y(u))" stroke="$(u == 0 ? "#888888" : "#ececec")"/>""")
        println(io, """<text x="$(l - 6)" y="$(Y(u) + 4)" text-anchor="end">$(round(u; digits = 1))</text>""")
    end
    for xv in 0:1:10
        println(io, """<text x="$(X(xv))" y="$(Hh - b + 16)" text-anchor="middle">$xv</text>""")
    end
    println(io, """<rect x="$l" y="$t" width="$(W - l - r)" height="$(Hh - t - b)" fill="none" stroke="black"/>""")
    println(io, """<text x="$((W - r + l) / 2)" y="$(Hh - 15)" text-anchor="middle">x / H  (cube: 3–4)</text>""")
    println(io, """<text transform="translate(22,$((Hh - b + t) / 2)) rotate(-90)" text-anchor="middle">mean u / U_b at z = Δz/2, centre line</text>""")
    # MT93 reattachment and front separation
    for (xv, s) in ((XC1 + XR_MT93, "x_R MT93"), (XC0 - XF_MT93, "x_F MT93"))
        println(io, """<line x1="$(X(xv))" y1="$t" x2="$(X(xv))" y2="$(Hh - b)" stroke="black" stroke-dasharray="5,4"/>""")
        println(io, """<text x="$(X(xv) + 3)" y="$(t + 14)" font-size="11">$s</text>""")
    end
    ly = t + 10
    for mdl in MODELS
        f = joinpath(OUT, "cube_$(mdl)_$(NH).csv")
        isfile(f) || continue
        lines = readlines(f)
        hdr = lines[1]
        rows = [parse.(Float64, split(s, ",")) for s in lines[3:end]]
        pts = join([@sprintf("%.1f,%.1f", X(rw[1]), Y(clamp(rw[2], -0.4, 0.6))) for rw in rows], " ")
        println(io, """<polyline fill="none" stroke="$(col[mdl])" stroke-width="1.8" points="$pts"/>""")
        xR = parse(Float64, match(r"x_R/H=([-0-9.eNa]+)", hdr)[1])
        println(io, """<line x1="$(W - r + 12)" y1="$ly" x2="$(W - r + 36)" y2="$ly" stroke="$(col[mdl])" stroke-width="1.8"/>""")
        println(io, """<text x="$(W - r + 42)" y="$(ly + 4)">$(lab[mdl])</text>""")
        println(io, """<text x="$(W - r + 42)" y="$(ly + 19)" font-size="11">x_R/H = $(round(xR; digits = 2))</text>""")
        ly += 44
    end
    println(io, """<text x="$(W - r + 12)" y="$(ly + 4)" font-size="11">MT93: x_R/H = $(XR_MT93)</text>""")
    println(io, "</svg>")
    path = joinpath(OUT, "surface_mounted_cube_$(NH).svg")
    write(path, String(take!(io)))
    println("wrote ", path)
end

if length(ARGS) >= 1 && ARGS[1] == "run"
    run(Symbol(ARGS[2]))
elseif length(ARGS) >= 1 && ARGS[1] == "plot"
    plot()
else
    println("usage: surface_mounted_cube.jl run <smagorinsky|tke|rsm> | plot")
end
