# The `simulation`-config digests of the component pipeline (ONE-a, #450).
#
# Each variant is a shipped `examples/<case>/<case>.toml` with overrides that
# switch on one or more of the core's pieces (the SGS closures, Coriolis, the
# sponge, the surface θ flux, the advection), on both cores and with every
# stepper (euler, rk3, adaptive). `simulation_digests()` runs every variant
# at 16³ and returns the fnv1a64 digest of every array of `state_fields`
# and of the clock. test/data/components/simulation_digests.toml holds the
# digests captured on master `ead7b986`, before the core's step was run
# through `Aeolus.Components` (capture_simulation_digests.jl); the test
# test/components_pipeline.jl asserts the pipeline reproduces them bit for bit.

using Aeolus
using Aeolus.Config
using Aeolus.IO: fnv1a64

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
    "dcbl_sigma"           => ("dry_convective_bl", [sd_steps(8), "sgs.model=\"sigma\""]),
    "dcbl_amd"             => ("dry_convective_bl", [sd_steps(8), "sgs.model=\"amd\""]),
    "dcbl_brown"           => ("dry_convective_bl", vcat([sd_steps(6), "sgs.stability=\"brown_et_al_1994\""], SD_DCBL_STABLE)),
    "dcbl_vreman_lilly"    => ("dry_convective_bl", vcat([sd_steps(6), "sgs.model=\"vreman\"",
                                                     "sgs.stability=\"lilly_cutoff\""], SD_DCBL_STABLE)),
    "dcbl_moist"           => ("dry_convective_bl", [sd_steps(8), "dynamics.moisture=true",
                                                     "initial.qv={kind=\"constant\", value=0.008}"]),
    "dcbl_moist_rk3_amd"   => ("dry_convective_bl", [sd_steps(6), "dynamics.moisture=true", "sgs.model=\"amd\"",
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
# anelastic core. Their digests are captured on the ONE-j branch
# (simulation_digests_anelastic.toml, capture_anelastic_digests.jl): the
# four of SD_VARIANTS, whose master digests the weighting changed, and these.
const SD_ANELASTIC_VARIANTS = [
    "dcbl_anelastic_rk3"      => ("dry_convective_bl", [sd_steps(6), "core.model=\"anelastic\"",
                                                        "timestep.scheme=\"rk3\""]),
    "dcbl_anelastic_adaptive" => ("dry_convective_bl", vcat([sd_steps(8), "core.model=\"anelastic\""], SD_ADAPTIVE)),
    "dcbl_anelastic_rk3_adaptive_cor" => ("dry_convective_bl", vcat([sd_steps(6), "core.model=\"anelastic\"",
                                                        "timestep.scheme=\"rk3\"", "coriolis.f=1.0e-4", SD_WIND,
                                                        "sgs.model=\"dynamic_smagorinsky\""], SD_ADAPTIVE)),
]
"Whether a variant runs the anelastic core (its digests are ONE-j's, not master's)."
sd_anelastic(label) = occursin("anelastic", label)

# The simulation configs EX-d (#441) added with their couplings around the
# step ([forcing], [surface], [microphysics], [tracer], [particles],
# [obstacles]): `label => (config path under examples/, overrides)`, at
# their own [test] smoke size, plus RK3 runs of the couplings that act
# inside the step (the IBM obstacles, the slab's floor flux). Captured on
# master a22b57ba (simulation_digests_exd.toml).
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

"""
    exd_digests(; examples = <this checkout's examples/>, only = nothing) -> Dict

The digests of `SD_EXD_VARIANTS` (as [`simulation_digests`](@ref)), each
run from its config's directory (relative input paths).
"""
function exd_digests(; examples = normpath(joinpath(@__DIR__, "..", "..", "examples")), only = nothing)
    out = Dict{String,Dict{String,String}}()
    for (label, (rel, ovr)) in SD_EXD_VARIANTS
        only === nothing || label in only || continue
        path = joinpath(examples, rel)
        r = cd(dirname(path)) do
            cfg = load_run_config(path; overrides = vcat(SD_EXD_TEST, "fftw.planner=\"estimate\"", ovr))
            run_config(cfg; root = mktempdir(), console = false, repo = mktempdir())
        end
        s = Base.invokelatest(state_fields, r.obj)
        d = Dict{String,String}(string(k) => sd_digest(v) for (k, v) in pairs(s))
        d["t"] = sd_digest([Base.invokelatest(sd_clock, r.obj)])
        rows = sd_diagnostics(r.folder)
        d["diagnostics_csv"] = sd_digest(codeunits(join(rows, "\n")))
        # the rows without the `sgs.*` columns, and whether the closure is the
        # plain Smagorinsky (the core's built-in C_s before ONE-a, #450)
        d["diagnostics_csv_without_sgs"] = sd_digest(codeunits(join(sd_drop_sgs(rows), "\n")))
        d["plain_smagorinsky"] = string(r.obj.cfg.sgs.model === :smagorinsky && get(getfield(r.obj.cfg.sgs, :values), "stability", :none) === :none)
        out[label] = d
    end
    return out
end

"`rows` of a diagnostics.csv without its `sgs.*` columns."
function sd_drop_sgs(rows)
    isempty(rows) && return rows
    keep = findall(h -> !startswith(h, "sgs."), split(rows[1], ','))
    return [join(split(l, ',')[keep], ',') for l in rows]
end

"fnv1a64 of the bytes of an array, as 16 hex digits."
sd_digest(a) = string(fnv1a64(collect(reinterpret(UInt8, vec(collect(a))))); base = 16, pad = 16)

"""
    simulation_digests(; inputs = Dict(), only = nothing) -> Dict{String,Dict{String,String}}

Run every variant of `variants` (default `SD_VARIANTS`; or those in `only`) and return
`label => (field => digest)`, the fields of `state_fields` plus `t`.
`inputs` maps a config name to extra overrides (the made dynamic driver of
`lindenberg_era5`).
"""
function simulation_digests(; inputs = Dict{String,Vector{String}}(), only = nothing,
                            variants = SD_VARIANTS)
    out = Dict{String,Dict{String,String}}()
    for (label, (name, ovr)) in variants
        only === nothing || label in only || continue
        cfg = load_run_config(sd_config(name);
                              overrides = vcat(SD_SMALL, ovr, get(inputs, name, String[])))
        r = run_config(cfg; root = mktempdir(), console = false, repo = mktempdir())
        s = Base.invokelatest(state_fields, r.obj)
        d = Dict{String,String}(string(k) => sd_digest(v) for (k, v) in pairs(s))
        d["t"] = sd_digest([Base.invokelatest(sd_clock, r.obj)])
        d["diagnostics_csv"] = sd_digest(codeunits(join(sd_diagnostics(r.folder), "\n")))
        out[label] = d
    end
    return out
end

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
written by AeolusPrep, data variables bitwise those `aeolus preprocess`
wrote when the digests were captured), so there are none.
"""
sd_made_inputs() = Dict{String,Vector{String}}()
