# The Berlin nest, measured (N-berlin-nest, #265): cost and the evening
# boundary-heat diagnostics of the nested flagship against the periodic box.
#
#   julia --project=. examples/urban/berlin_nest_check.jl cost-nested   [minutes]
#   julia --project=. examples/urban/berlin_nest_check.jl cost-periodic [minutes]
#   julia --project=. examples/urban/berlin_nest_check.jl cost-200m     [minutes]
#   julia --project=. examples/urban/berlin_nest_check.jl cost-periodic-wind [minutes]
#   julia --project=. examples/urban/berlin_nest_check.jl evening-nested   [hours, default 1]
#   julia --project=. examples/urban/berlin_nest_check.jl evening-periodic [hours, default 1]
#
# cost-*: wall seconds per simulated hour from 12:00 LT, 29 June 2019, the
#   first step (compilation) excluded: the nested default (6 km parent at
#   refinement (2, 2, 4), 400 m child), the 400 m child in the periodic box
#   (nest off), the same driven by the nest's geostrophic wind instead of the
#   default 1 m/s (periodic-wind), and the 200 m child of before the nest
#   (64 × 64 × 32, nest off).
# evening-*: 18:00 LT onwards (default 1 h; the 18–21 h window of the issue
#   needs about an hour of wall time per simulated hour nested, the full
#   evening is the 24 h validation of #270): the heat the child's lid and
#   sides carry (the air-heat ledger's boundary terms, W/m² of child ground
#   area) and the change of the mean θ of the air in 0–100 m, nested (the
#   uniform 62.5 m × 50 m fallback parent of #259, refinement (4, 4, 8), to
#   keep the run short) and periodic. Both start from the Lindenberg
#   sounding at 18:00 without spin-up.
#
# One run per process (they share nothing); each prints one summary line.

using Aeolus
using Aeolus.Fields: interior_view
using Printf

isdefined(Main, :BerlinCase) || include(joinpath(@__DIR__, "BerlinCase.jl"))
const BC = Main.BerlinCase

"Mean θ of the fluid cells below `zmax` (m)."
function mean_θ_below(case, zmax)
    m, θ = case.mesh, interior_view(case.sim.θ)
    s = 0.0; n = 0
    for k in 1:m.Nz
        (k - 0.5) * m.dz < zmax || break
        for j in 1:m.Ny, i in 1:m.Nx
            case.geom.sm.mask[m.Hx+i, m.Hy+j, m.Hz+k] && continue
            s += θ[i, j, k]; n += 1
        end
    end
    return s / n
end

function config(kind::AbstractString; start_h)
    base = (verbose = false, output_path = nothing, start_h = start_h)
    kind == "nested"   && return BC.CaseConfig(:summer; base...)
    kind == "fallback" && return BC.CaseConfig(:summer; base...,
        options = Dict{Symbol,Any}(:nesting => (parent = (refinement = (4, 4, 8), vertical = :uniform),)))
    kind == "periodic" && return BC.CaseConfig(:summer; base..., physics = Dict{Symbol,Any}(:nesting => false))
    kind == "200m"     && return BC.CaseConfig(:summer; base..., size = (64, 64, 32), extent = (1000.0, 1000.0, 200.0),
                                               physics = Dict{Symbol,Any}(:nesting => false))
    if kind == "periodic-wind"
        # the periodic box driven by the nest's geostrophic wind (the
        # sounding wind above z_g at the start), not the 1 m/s default: the
        # nest's cost without the wind's
        cfg = BC.CaseConfig(:summer; base...)
        Ug, Vg = BC.nest_geostrophic_wind(cfg)
        U, V = Ug(3000.0, 0.0), Vg(3000.0, 0.0)
        return BC.CaseConfig(:summer; base..., u0 = U, physics = Dict{Symbol,Any}(:nesting => false),
                             options = Dict{Symbol,Any}(:geostrophic => (U_g = U, V_g = V)))
    end
    error("unknown configuration $kind")
end

function run_cost(kind; minutes = 10.0)
    case = BC.build_case(config(kind; start_h = 12.0); io = devnull)
    BC.step_case!(case, BC.step_dt(case))                 # compilation
    t0, w0 = case.sim.t, time()
    BC.run_case!(case; nsteps = round(Int, minutes * 60 / case.cfg.dt))
    sim_s, wall = case.sim.t - t0, time() - w0
    c = BC.find_coupling(case, :nesting)
    @printf("cost %-9s %d × %d × %d child: %.1f simulated s in %.1f s wall → %.0f s per simulated hour; %d child steps (mean dt %.2f s)%s\n",
            kind, case.mesh.Nx, case.mesh.Ny, case.mesh.Nz, sim_s, wall, 3600 * wall / sim_s,
            case.clock.n - 1, sim_s / (case.clock.n - 1),
            c === nothing ? "" : @sprintf("; parent %d × %d × %d, %d windows, parent steps %.1f s wall (%.0f %%)",
                                          c.parent.mesh.Nx, c.parent.mesh.Ny, c.parent.mesh.Nz, c.windows,
                                          c.wall_parent, 100 * c.wall_parent / wall))
end

function run_evening(kind; hours = 1.0)
    case = BC.build_case(config(kind == "nested" ? "fallback" : kind; start_h = 18.0); io = devnull)
    θ0 = mean_θ_below(case, 100.0)
    w0 = time()
    rows = String[]
    for h in 1:round(Int, hours)
        hb0 = BC.heat_budget(case.ledger)
        ob0 = case.open === nothing ? (0.0, 0.0) : (case.open.lateral, case.open.top)
        BC.run_case!(case; nsteps = round(Int, 3600 / case.cfg.dt))
        hb = BC.heat_budget(case.ledger)
        el = hb.elapsed - hb0.elapsed
        d(k) = (get(hb.terms, k, 0.0) - get(hb0.terms, k, 0.0)) / el
        lat, top = case.open === nothing ? (0.0, 0.0) :
                   ((case.open.lateral - ob0[1]) / el, (case.open.top - ob0[2]) / el)
        push!(rows, @sprintf("  %02d–%02d LT: boundary heat %.2f W/m² (sides %.2f, lid %.2f; booked boundary_heat + mass_correction %.2f), relaxation %.2f, surface %.2f W/m², <θ>(0–100 m) %.3f K",
                             17 + h, 18 + h, lat + top, lat, top,
                             d((:dynamics, :boundary_heat)) + d((:dynamics, :mass_correction)),
                             d((:dynamics, :relaxation)), d((:surface_flux, :surface_exchange)),
                             mean_θ_below(case, 100.0)))
    end
    θ1 = mean_θ_below(case, 100.0)
    @printf("evening %-8s 18:00–%02d:00 LT: <θ>(0–100 m) %.3f → %.3f K (Δ %+.3f K); %.0f s wall\n",
            kind, 18 + round(Int, hours), θ0, θ1, θ1 - θ0, time() - w0)
    foreach(println, rows)
end

what = isempty(ARGS) ? "cost-nested" : ARGS[1]
arg = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : nothing
if startswith(what, "cost-")
    run_cost(what[6:end]; minutes = something(arg, 10.0))
elseif startswith(what, "evening-")
    run_evening(what[9:end]; hours = something(arg, 1.0))
else
    error("usage: berlin_nest_check.jl cost-nested|cost-periodic|cost-200m|evening-nested|evening-periodic [length]")
end
