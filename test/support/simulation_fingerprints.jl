# The simulation-config fingerprints of the component pipeline (ONE-a, #450;
# fingerprints in place of digests since #495). The variants: the four shipped
# configs at 16³ with every SGS closure, Coriolis, the sponge, the surface θ
# flux, both cores and every stepper (SD_VARIANTS, SD_ANELASTIC_VARIANTS),
# and the EX-d configs with their couplings at their own smoke size
# (SD_EXD_VARIANTS). `variant_fingerprints` runs one and returns the
# fingerprint (test/support/agreement.jl) of every array of `state_fields`,
# the clock, and the fingerprint of every numeric column of its
# diagnostics.csv. test/data/components/simulation_fingerprints.toml holds
# them (capture_simulation_fingerprints.jl); test/components_pipeline.jl
# asserts every run reproduces them within ROUNDOFF_RTOL. The digests they
# replace (, captured on master ead7b986 before the pipeline, on the
# ONE-j branch for the anelastic weighting and on master a22b57ba for EX-d)
# are in git history: the fingerprints were captured from code that
# reproduced every one of those digests.

using Aeolus
using Aeolus.Config
include(joinpath(@__DIR__, "agreement.jl"))

const SD_EXAMPLES = normpath(joinpath(@__DIR__, "..", "..", "examples"))
"The shipped config of the example case `name`, `examples/<name>/<name>.toml`."
sd_config(name) = joinpath(SD_EXAMPLES, name, name * ".toml")
const SD_SMALL = ["grid.size=[16,16,16]", "fftw.planner=\"estimate\"", "time.duration=\"none\""]
sd_steps(n) = "time.nsteps=$n"

const SD_DCBL_STABLE = ["initial.theta={kind=\"mixed_layer_lapse\", value=300.0, z_i=0.0, lapse=0.01}",
                        "initial.u={kind=\"table\", z=[0.0, 1600.0], values=[0.0, 8.0]}"]
const SD_WIND = "initial.u={kind=\"table\", z=[0.0, 1600.0], values=[2.0, 8.0]}"
const SD_ADAPTIVE = ["time.dt=\"adaptive\"", "time.cfl=0.3", "time.dt_max=5.0"]

"`label => (config, overrides)` of every digest variant."
const SD_VARIANTS = [
    "tgv_euler"            => ("taylor_green_vortex", [sd_steps(8)]),
    "tgv_rk3"              => ("taylor_green_vortex", [sd_steps(6), "timestep.scheme=\"rk3\""]),
    "tgv_adaptive"         => ("taylor_green_vortex", [sd_steps(8), "time.dt=\"adaptive\"", "time.cfl=0.3",
                                                       "time.dt_max=1.0"]),
    "bc_euler"             => ("buoyant_convection", [sd_steps(8)]),
    "dcbl_euler"           => ("dry_convective_bl", [sd_steps(8)]),
    "dcbl_rk3"             => ("dry_convective_bl", [sd_steps(6), "timestep.scheme=\"rk3\""]),
    "dcbl_adaptive"        => ("dry_convective_bl", vcat([sd_steps(8)], SD_ADAPTIVE)),
    "dcbl_rk3_adaptive"    => ("dry_convective_bl", vcat([sd_steps(6), "timestep.scheme=\"rk3\""], SD_ADAPTIVE)),
    "dcbl_coriolis"        => ("dry_convective_bl", [sd_steps(8), "coriolis.f=1.0e-4", SD_WIND]),
    "dcbl_coriolis_rk3"    => ("dry_convective_bl", [sd_steps(6), "coriolis.f=1.0e-4", SD_WIND,
                                                     "timestep.scheme=\"rk3\""]),
    # |f| = 0.2 s⁻¹: the Coriolis limit 0.5/|f| = 2.5 s binds the adaptive step
    "dcbl_coriolis_limit"  => ("dry_convective_bl", vcat([sd_steps(6), "coriolis.f=0.2", SD_WIND], SD_ADAPTIVE)),
    "dcbl_no_sponge"       => ("dry_convective_bl", [sd_steps(8), "sponge.alpha_max=0.0"]),
    "dcbl_top_flux"        => ("dry_convective_bl", [sd_steps(8), "surface_theta_flux.top=-0.05"]),
    "dcbl_sgs_none"        => ("dry_convective_bl", [sd_steps(8), "sgs.model=\"none\""]),
    "dcbl_dynamic"         => ("dry_convective_bl", [sd_steps(8), "sgs.model=\"dynamic_smagorinsky\""]),
    "dcbl_vreman"          => ("dry_convective_bl", [sd_steps(8), "sgs.model=\"vreman\""]),
    # the file's c_s = 0.17 (smagorinsky's), which an override of the closure
    # kept before ONE-m (#476) and now resets: stated, so the runs are the captured ones
    "dcbl_sigma"           => ("dry_convective_bl", [sd_steps(8), "sgs.model=\"sigma\"", "sgs.c_s=0.17"]),
    "dcbl_amd"             => ("dry_convective_bl", [sd_steps(8), "sgs.model=\"amd\"", "sgs.c_s=0.17"]),
    "dcbl_brown"           => ("dry_convective_bl", vcat([sd_steps(6), "sgs.stability=\"brown_et_al_1994\""], SD_DCBL_STABLE)),
    "dcbl_vreman_lilly"    => ("dry_convective_bl", vcat([sd_steps(6), "sgs.model=\"vreman\"",
                                                     "sgs.stability=\"lilly_cutoff\""], SD_DCBL_STABLE)),
    "dcbl_moist"           => ("dry_convective_bl", [sd_steps(8), "dynamics.moisture=true",
                                                     "initial.qv={kind=\"constant\", value=0.008}"]),
    "dcbl_moist_rk3_amd"   => ("dry_convective_bl", [sd_steps(6), "dynamics.moisture=true", "sgs.model=\"amd\"", "sgs.c_s=0.17",
                                                     "initial.qv={kind=\"table\", z=[0.0, 1600.0], values=[0.01, 0.002]}",
                                                     "timestep.scheme=\"rk3\""]),
    "dcbl_stretched"       => ("dry_convective_bl", [sd_steps(6), "grid.stretching=\"tanh\"",
                                                     "grid.stretching_parameter=1.5"]),
    "dcbl_anelastic"       => ("dry_convective_bl", [sd_steps(6), "core.model=\"anelastic\"", "initial.theta={}"]),
    "dcbl_anelastic_cor"   => ("dry_convective_bl", [sd_steps(6), "core.model=\"anelastic\"", "initial.theta={}",
                                                     "coriolis.f=1.0e-4", SD_WIND,
                                                     "surface_theta_flux.top=-0.05"]),
    "dcbl_anelastic_vreman" => ("dry_convective_bl", [sd_steps(6), "core.model=\"anelastic\"", "initial.theta={}",
                                                      "sgs.model=\"vreman\""]),
    "dcbl_anelastic_none"  => ("dry_convective_bl", [sd_steps(6), "core.model=\"anelastic\"", "initial.theta={}",
                                                     "sgs.model=\"none\"", "dynamics.nu=1.0"]),
    "lindenberg_open_rk3"  => ("lindenberg_era5", [sd_steps(6)]),
]

# The anelastic variants: new physics in ONE-j (#460) — the SGS and surface
# fluxes in the ρ̄-weighted flux form, SSP-RK3 and the adaptive step on the
# anelastic core (the four of SD_VARIANTS on that core, and these).
const SD_ANELASTIC_VARIANTS = [
    "dcbl_anelastic_rk3"      => ("dry_convective_bl", [sd_steps(6), "core.model=\"anelastic\"",
                                                        "timestep.scheme=\"rk3\""]),
    "dcbl_anelastic_adaptive" => ("dry_convective_bl", vcat([sd_steps(8), "core.model=\"anelastic\""], SD_ADAPTIVE)),
    "dcbl_anelastic_rk3_adaptive_cor" => ("dry_convective_bl", vcat([sd_steps(6), "core.model=\"anelastic\"",
                                                        "timestep.scheme=\"rk3\"", "coriolis.f=1.0e-4", SD_WIND,
                                                        "sgs.model=\"dynamic_smagorinsky\""], SD_ADAPTIVE)),
]

# The simulation configs EX-d (#441) added with their couplings around the
# step ([forcing], [surface], [microphysics], [tracer], [particles],
# [obstacles]): `label => (config path under examples/, overrides)`, at
# their own [test] smoke size, plus RK3 runs of the couplings that act
# inside the step (the IBM obstacles, the slab's floor flux). Captured on
# master a22b57ba.
const SD_EXD_TEST = ["time.duration=\"none\"", "time.nsteps=10"]
const SD_EXD_VARIANTS = [
    "dcbl_radiation_lsm"     => ("dcbl_radiation_lsm/dcbl_radiation_lsm.toml", ["grid.size=[16,16,16]"]),
    "dcbl_radiation_lsm_rk3" => ("dcbl_radiation_lsm/dcbl_radiation_lsm.toml", ["grid.size=[16,16,16]", "timestep.scheme=\"rk3\""]),
    "heated_plume"           => ("heated_plume/heated_plume.toml", ["grid.size=[16,16,16]"]),
    "flow_past_cube"         => ("flow_past_cube/flow_past_cube.toml", String[]),
    "flow_past_cube_rk3"     => ("flow_past_cube/flow_past_cube.toml", ["timestep.scheme=\"rk3\""]),
    "moist_convection"       => ("moist_convection/moist_convection.toml", ["grid.size=[12,12,16]"]),
    "channel_flow"           => ("channel_flow/channel_flow.toml", String[]),
    "decaying_hit"           => ("decaying_hit/decaying_hit.toml", ["grid.size=[16,16,16]"]),
    "pollution_plume"        => ("pollution_plume/pollution_plume.toml", String[]),
]

"The rows of a run's diagnostics.csv without the machine columns (wall clock, process)."
function sd_diagnostics(folder)
    path = Aeolus.IO.diagnostics_path(folder)
    isfile(path) || return String[]
    lines = readlines(path)
    isempty(lines) && return lines
    machine(h) = startswith(h, "wall") || occursin(".wall", h) || startswith(h, "proc.")
    keep = findall(!machine, split(lines[1], ','))
    return [join(split(l, ',')[keep], ',') for l in lines]
end

# the clock of a run: `sim.t` of a `SimulationRun` before ONE-a, its `Model`'s after
sd_clock(obj) = hasproperty(obj, :sim) ? obj.sim.t : Aeolus.Components.clock(obj.model)

"""
    sd_made_inputs() -> Dict

Extra overrides of made input drivers. Since ONE-h (#457) `lindenberg_era5`
reads its committed dynamic driver (`input/lindenberg_era5_dynamic.nc`,
written by AeolusPrep), so there are none.
"""
sd_made_inputs() = Dict{String,Vector{String}}()

"""
    SD_ALL -> Vector

Every variant as `label => (config path, overrides, run directory)`: the
16³ variants (`SD_SMALL` plus their own overrides, run from the working
directory) and the EX-d configs (`SD_EXD_TEST`, run from their config's
directory, whose relative input paths they use).
"""
const SD_ALL = vcat(
    [l => (sd_config(n), vcat(SD_SMALL, o), nothing) for (l, (n, o)) in vcat(SD_VARIANTS, SD_ANELASTIC_VARIANTS)],
    [l => (joinpath(SD_EXAMPLES, rel), vcat(SD_EXD_TEST, "fftw.planner=\"estimate\"", o), dirname(joinpath(SD_EXAMPLES, rel)))
     for (l, (rel, o)) in SD_EXD_VARIANTS])

"""
    variant_fingerprints(spec) -> Dict{String,Any}

Run one variant of `SD_ALL` (`label => (path, overrides, dir)`) and return
`"fields"` (name => fingerprint of every array of `state_fields`), `"t"`
(the clock) and `"diagnostics"` (column => fingerprint of every numeric
column of its diagnostics.csv, the machine columns left out).
"""
function variant_fingerprints(spec)
    _, (path, ovr, dir) = spec
    go() = run_config(load_run_config(path; overrides = vcat(ovr, get(sd_made_inputs(), basename(dirname(path)), String[])));
                      root = mktempdir(), console = false, repo = mktempdir())
    r = dir === nothing ? go() : cd(go, dir)
    s = Base.invokelatest(state_fields, r.obj)
    rows = sd_diagnostics(r.folder)
    return Dict{String,Any}("fields" => Dict{String,Any}(string(k) => fingerprint(v) for (k, v) in pairs(s)),
                            "t" => Float64(Base.invokelatest(sd_clock, r.obj)),
                            "diagnostics" => isempty(rows) ? Dict{String,Any}() :
                                             Dict{String,Any}(k => fingerprint(v) for (k, v) in diagnostics_columns(rows)))
end
