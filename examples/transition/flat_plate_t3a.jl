# ERCOFTAC T3A-like bypass transition — Era VI, Phase 59 (P59-d, #185).
#
# A spatially developing zero-pressure-gradient flat-plate boundary layer
# under free-stream turbulence of Tu = 3.3 % (the T3A level of Roach &
# Brierley 1992), in units of the inflow displacement thickness δ*₀ and
# U∞ = 1, Re_δ*₀ = 300 (Re_x = 3.0e4 at the inflow):
#
#   * x-periodic domain 600 × 36 × 24 δ*₀, 288 × 32 × 48 cells
#     (dx = 2.08, dy = 1.13, dz = 0.5 δ*₀); the plate spans Re_x = 3.0e4 →
#     1.9e5, the last 80 δ*₀ are the fringe (Nordström et al. 1999) that
#     relaxes the outflow to the Blasius inflow and carries the free-stream
#     turbulence in (Brandt, Schlatter & Henningson 2004);
#   * NoSlipWall plate (P59-a), FreeStreamBoundary top with the Blasius
#     displacement velocity, RK3 at dt = 1 δ*₀/U, no SGS model (a coarse
#     DNS; the Smagorinsky ν_t of the laminar wall shear alone would be
#     several times ν and keep the layer laminar);
#   * free-stream turbulence: 200 random Fourier modes, von Kármán
#     spectrum, L11 = 10 δ*₀, kept out of the inflow layer below 2 δ₉₉ in
#     the fringe;
#   * 1500 steps (2.5 flow-through times); statistics over the last 600.
#
# Diagnostics (Aeolus.Transition): c_f(x), the onset (minimum of c_f
# upstream of the persistent rise, `transition_onset`), Re_θ at onset with
# θ integrated to the δ₉₉.₅ edge, compared with Mayle (1991) and
# Abu-Ghannam & Shaw (1980); downstream c_f against 0.0576 Re_x^{-1/5};
# the shape factor H(x) and the free-stream Tu(x).
#
# Figures (docs/src/assets/transition/): t3a_cf.svg (c_f vs Re_x with the
# laminar and turbulent correlations), t3a_shape_factor.svg, t3a_tu.svg.
#
#   JULIA_NUM_THREADS=4 julia --project=. examples/transition/flat_plate_t3a.jl

using Aeolus
using Aeolus.Fields: interior_view
using Aeolus.Operators: smooth_step
using Printf
using Serialization

include(joinpath(@__DIR__, "..", "viz", "plot_utils.jl"))
const OUT = joinpath(@__DIR__, "..", "..", "docs", "src", "assets", "transition")
mkpath(OUT)

const Tu = 0.033
nsteps = parse(Int, get(ENV, "T3A_STEPS", "1500")); navg = 600; dt = 1.0

c = flat_plate_setup(; Re_δ0 = 300, Lx = 600, Ly = 36, H = 24, Nx = 288, Ny = 32, Nz = 48,
                     fringe_length = 80)
sim, g = c.sim, c.sim.mesh
fst = FreeStreamDisturbance(Tu, 10.0, 200; seed = 1, period_y = g.Ly,
                            κ_max = 2π / (3 * max(g.dx, g.dy, g.dz)))
δ99 = blasius_solution().eta99 * sqrt(c.ν * c.x_in)
target = deepcopy(c.target)
update! = disturbance_fringe_update(fst, c.fringe, c.target;
                                    damping = z -> smooth_step(z / (2δ99)))
force! = fringe_forcing(c.fringe, target; update! = update!)
set_blasius!(sim; x_le = c.x_le, x_min = c.x_in)

x = collect(Aeolus.Grids.xnodes(g, FaceX()))
Re_x = (x .- c.x_le) ./ c.ν
Lp = g.Lx - 80                                  # plate length (fringe excluded)
plate = findall(<(Lp - 20), x)                  # stay clear of the fringe rise

# T3A_SAVE=path stores the statistics, T3A_LOAD=path re-plots them without
# re-running the simulation.
if haskey(ENV, "T3A_LOAD")
    (cf, ū, urms, wall) = deserialize(ENV["T3A_LOAD"])
else
    cf = zeros(g.Nx); ū = zeros(g.Nx, g.Nz); u2 = zeros(g.Nx, g.Nz); nacc = 0
    t0 = time()
    for n in 1:nsteps
        step_rk3!(sim, dt; body_force! = force!)
        if n > nsteps - navg
            global nacc += 1
            cf .+= skin_friction(sim; U_e = 1.0)
            ui = interior_view(sim.u)
            ū .+= dropdims(sum(ui; dims = 2); dims = 2) ./ g.Ny
            u2 .+= dropdims(sum(ui .^ 2; dims = 2); dims = 2) ./ g.Ny
        end
        n % 250 == 0 && @printf("step %d  t = %.0f  max|u| = %.3f  max div = %.1e  (%.0f s)\n",
                                n, sim.t, maximum(abs, interior_view(sim.u)), max_divergence(sim),
                                time() - t0)
    end
    wall = time() - t0
    cf ./= nacc; ū ./= nacc; u2 ./= nacc
    urms = sqrt.(max.(u2 .- ū .^ 2, 0.0))
    haskey(ENV, "T3A_SAVE") && serialize(ENV["T3A_SAVE"], (cf, ū, urms, wall))
end

# boundary-layer integrals to the δ99.5 edge (the free stream carries a
# 0.5 % mean deficit from the disturbance forcing)
ke = boundary_layer_edge(ū; fraction = 0.995)
Ue = vec(maximum(ū; dims = 2))                  # free-stream velocity per x
θ = momentum_thickness(ū, g.dz; U_e = Ue, kmax = ke)
H = shape_factor(ū, g.dz; U_e = Ue, kmax = ke)
# free-stream Tu(x): streamwise rms over the levels from the δ99.5 edge to
# 2 δ*₀ below the top (the six levels below that if the layer reaches it)
function fs_levels(i)
    r = (ke[i] + 1):(g.Nz - 4)
    return isempty(r) ? ((g.Nz - 9):(g.Nz - 4)) : r
end
Tu_x = [sum(urms[i, k] for k in fs_levels(i)) / length(fs_levels(i)) for i in 1:g.Nx]

xo, io = transition_onset(x[plate], cf[plate], blasius_skin_friction.(Re_x[plate]))
Reθ_on = θ[io] / c.ν
Reθ_blasius = 2 * blasius_solution().fpp0 * sqrt(Re_x[io])
Mayle = onset_reynolds_theta_mayle(100Tu)
AGS = onset_reynolds_theta_abu_ghannam_shaw(100Tu)
down = findall(i -> Re_x[i] >= 1.2e5 && i in plate, eachindex(x))
ratio_t = cf[down] ./ turbulent_skin_friction.(Re_x[down])

@printf("\ngrid %d × %d × %d, %d steps, wall %.0f s (%.2f s/step)\n", g.Nx, g.Ny, g.Nz,
        nsteps, wall, wall / nsteps)
@printf("free-stream Tu: %.2f %% at the inflow, %.2f %% at Re_x = 1e5, %.2f %% at the plate end\n",
        100Tu_x[1], 100Tu_x[argmin(abs.(Re_x .- 1e5))], 100Tu_x[plate[end]])
@printf("onset: x = %.1f δ*₀, Re_x = %.0f, Re_θ = %.0f (θ measured; Blasius at that Re_x: %.0f)\n",
        xo, Re_x[io], Reθ_on, Reθ_blasius)
@printf("       Mayle (1991) %.0f  → %+.0f %%;  Abu-Ghannam & Shaw (1980) %.0f → %+.0f %%\n",
        Mayle, 100(Reθ_on / Mayle - 1), AGS, 100(Reθ_on / AGS - 1))
@printf("downstream (Re_x %.0f–%.0f): c_f / 0.0576 Re_x^-1/5 ∈ [%.3f, %.3f], mean %.3f; H ∈ [%.2f, %.2f]\n",
        Re_x[down[1]], Re_x[down[end]], minimum(ratio_t), maximum(ratio_t),
        sum(ratio_t) / length(ratio_t), minimum(H[down]), maximum(H[down]))

# ---- figures ----------------------------------------------------------------
R = Re_x[plate]
save_multiline(joinpath(OUT, "t3a_cf.svg"), R ./ 1e5,
               [1e3 .* cf[plate], 1e3 .* blasius_skin_friction.(R),
                1e3 .* turbulent_skin_friction.(R)],
               ["Tu = 3.3 %", "laminar", "turbulent"];
               xlabel = "Re_x / 1e5 (laminar 0.664 Re_x^-1/2, turbulent 0.0576 Re_x^-1/5)",
               ylabel = "1000 c_f",
               title = @sprintf("T3A-like plate: onset Re_θ = %.0f (Mayle %.0f, AGS %.0f)",
                                Reθ_on, Mayle, AGS))
save_multiline(joinpath(OUT, "t3a_shape_factor.svg"), R ./ 1e5, [H[plate]], ["H = δ*/θ"];
               xlabel = "Re_x / 1e5", ylabel = "shape factor H",
               title = "Shape factor H = δ*/θ (integrated to the δ99.5 edge)")
save_multiline(joinpath(OUT, "t3a_tu.svg"), R ./ 1e5, [100 .* Tu_x[plate]], ["u_rms / U∞"];
               xlabel = "Re_x / 1e5", ylabel = "Tu (%)",
               title = "Free-stream Tu (u_rms above the δ99.5 edge)")
println("figures in ", normpath(OUT))
