# The dry convective boundary layer (examples/dry_convective_bl) on both
# dynamical cores through the `simulation` config, and the constraint
# weights of a core: shared by test/components_both_cores.jl (the θ budget
# to round-off over a few steps, checkpoints, the component contracts) and
# validation/components_both_cores.jl (the 30 min mixed layer on both
# cores). Functions and constants only; include with
#   isdefined(@__MODULE__, :bc_budget) || include(".../dcbl_both_cores.jl")

using Aeolus
using Aeolus.Config
using Aeolus.Fields: interior_view
using Aeolus.Anelastic: constraint_weights

"The constraint weights of `sim` (ones on the Boussinesq core)."
bc_weights(sim) = sim isa BoussinesqSimulation ? (ones(size(sim.θ.data, 3)), ones(size(sim.θ.data, 3) + 1)) :
                                                 constraint_weights(sim)

const BC_CFG = joinpath(pkgdir(Aeolus), "examples", "dry_convective_bl", "dry_convective_bl.toml")
bc_cbl_cfg(core, n; extra = String[]) =
    load_run_config(BC_CFG; overrides = vcat(["grid.size=[16,16,16]", "time.duration=\"none\"",
                                              "time.nsteps=$n", "fftw.planner=\"estimate\""],
                                             core === :anelastic ? ["core.model=\"anelastic\""] : String[], extra))
bc_run(cfg; kw...) = run_config(cfg; root = mktempdir(), console = false, repo = mktempdir(), kw...)

"The θ budget of a `simulation` run of the DCBL config: (Σ W δθ ΔV, W_bot F t A)."
function bc_budget(core, n; extra = String[])
    cfg = bc_cbl_cfg(core, n; extra = extra)
    s0 = Aeolus.SimulationDriver.simulation(Config.build(Aeolus.SimulationDriver.SimulationDriverType(), cfg))
    θ0 = copy(interior_view(s0.θ))
    r = bc_run(cfg)
    s = Aeolus.SimulationDriver.simulation(r.obj); m = s.mesh; H = m.Hz
    Wc, Wf = bc_weights(s)
    dθ = interior_view(s.θ) .- θ0
    got = sum(Wc[H+k] * sum(dθ[:, :, k]) for k in 1:m.Nz) * m.dx * m.dy * m.dz
    return (got = got, expected = Wf[H+1] * 0.24 * s.t * m.Lx * m.Ly, sim = s, θ0 = θ0)
end
