# The `simulation`-config digests of the component pipeline (ONE-a, #450).
#
# Each variant is a shipped `examples/configs/*.toml` with overrides that
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

const SD_CONFIGS = normpath(joinpath(@__DIR__, "..", "..", "examples", "configs"))
const SD_SMALL = ["grid.size=[16,16,16]", "fftw.planner=\"estimate\"", "stepper.end_time=\"none\""]
sd_steps(n) = "stepper.nsteps=$n"

const SD_DCBL_STABLE = ["initial.theta={kind=\"mixed_layer_lapse\", value=300.0, z_i=0.0, lapse=0.01}",
                        "initial.u={kind=\"table\", z=[0.0, 1600.0], values=[0.0, 8.0]}"]
const SD_WIND = "initial.u={kind=\"table\", z=[0.0, 1600.0], values=[2.0, 8.0]}"
const SD_ADAPTIVE = ["stepper.dt=\"none\"", "stepper.cfl=0.3", "stepper.dt_max=5.0"]

"`label => (config, overrides)` of every digest variant."
const SD_VARIANTS = [
    "tgv_euler"            => ("taylor_green_vortex", [sd_steps(8)]),
    "tgv_rk3"              => ("taylor_green_vortex", [sd_steps(6), "stepper.scheme=\"rk3\""]),
    "tgv_adaptive"         => ("taylor_green_vortex", [sd_steps(8), "stepper.dt=\"none\"", "stepper.cfl=0.3",
                                                       "stepper.dt_max=1.0"]),
    "bc_euler"             => ("buoyant_convection", [sd_steps(8)]),
    "dcbl_euler"           => ("dry_convective_bl", [sd_steps(8)]),
    "dcbl_rk3"             => ("dry_convective_bl", [sd_steps(6), "stepper.scheme=\"rk3\""]),
    "dcbl_adaptive"        => ("dry_convective_bl", vcat([sd_steps(8)], SD_ADAPTIVE)),
    "dcbl_rk3_adaptive"    => ("dry_convective_bl", vcat([sd_steps(6), "stepper.scheme=\"rk3\""], SD_ADAPTIVE)),
    "dcbl_coriolis"        => ("dry_convective_bl", [sd_steps(8), "dynamics.f_coriolis=1.0e-4", SD_WIND]),
    "dcbl_coriolis_rk3"    => ("dry_convective_bl", [sd_steps(6), "dynamics.f_coriolis=1.0e-4", SD_WIND,
                                                     "stepper.scheme=\"rk3\""]),
    # |f| = 0.2 s⁻¹: the Coriolis limit 0.5/|f| = 2.5 s binds the adaptive step
    "dcbl_coriolis_limit"  => ("dry_convective_bl", vcat([sd_steps(6), "dynamics.f_coriolis=0.2", SD_WIND], SD_ADAPTIVE)),
    "dcbl_no_sponge"       => ("dry_convective_bl", [sd_steps(8), "dynamics.sponge_alpha_max=0.0"]),
    "dcbl_top_flux"        => ("dry_convective_bl", [sd_steps(8), "boundaries.theta_flux_top=-0.05"]),
    "dcbl_sgs_none"        => ("dry_convective_bl", [sd_steps(8), "sgs.closure=\"none\""]),
    "dcbl_dynamic"         => ("dry_convective_bl", [sd_steps(8), "sgs.closure=\"dynamic_smagorinsky\""]),
    "dcbl_vreman"          => ("dry_convective_bl", [sd_steps(8), "sgs.closure=\"vreman\""]),
    "dcbl_sigma"           => ("dry_convective_bl", [sd_steps(8), "sgs.closure=\"sigma\""]),
    "dcbl_amd"             => ("dry_convective_bl", [sd_steps(8), "sgs.closure=\"amd\""]),
    "dcbl_brown"           => ("dry_convective_bl", vcat([sd_steps(6), "sgs.stability=\"brown_et_al_1994\""], SD_DCBL_STABLE)),
    "dcbl_vreman_lilly"    => ("dry_convective_bl", vcat([sd_steps(6), "sgs.closure=\"vreman\"",
                                                     "sgs.stability=\"lilly_cutoff\""], SD_DCBL_STABLE)),
    "dcbl_moist"           => ("dry_convective_bl", [sd_steps(8), "dynamics.moisture=true",
                                                     "initial.qv={kind=\"constant\", value=0.008}"]),
    "dcbl_moist_rk3_amd"   => ("dry_convective_bl", [sd_steps(6), "dynamics.moisture=true", "sgs.closure=\"amd\"",
                                                     "initial.qv={kind=\"table\", z=[0.0, 1600.0], values=[0.01, 0.002]}",
                                                     "stepper.scheme=\"rk3\""]),
    "dcbl_stretched"       => ("dry_convective_bl", [sd_steps(6), "grid.stretching=\"tanh\"",
                                                     "grid.stretching_parameter=1.5"]),
    "dcbl_anelastic"       => ("dry_convective_bl", [sd_steps(6), "dynamics.core=\"anelastic\"", "initial.theta={}"]),
    "dcbl_anelastic_cor"   => ("dry_convective_bl", [sd_steps(6), "dynamics.core=\"anelastic\"", "initial.theta={}",
                                                     "dynamics.f_coriolis=1.0e-4", SD_WIND,
                                                     "boundaries.theta_flux_top=-0.05"]),
    "dcbl_anelastic_vreman" => ("dry_convective_bl", [sd_steps(6), "dynamics.core=\"anelastic\"", "initial.theta={}",
                                                      "sgs.closure=\"vreman\""]),
    "dcbl_anelastic_none"  => ("dry_convective_bl", [sd_steps(6), "dynamics.core=\"anelastic\"", "initial.theta={}",
                                                     "sgs.closure=\"none\"", "dynamics.nu=1.0"]),
    "lindenberg_open_rk3"  => ("lindenberg_era5", [sd_steps(6)]),
]

"fnv1a64 of the bytes of an array, as 16 hex digits."
sd_digest(a) = string(fnv1a64(collect(reinterpret(UInt8, vec(collect(a))))); base = 16, pad = 16)

"""
    simulation_digests(; inputs = Dict(), only = nothing) -> Dict{String,Dict{String,String}}

Run every variant of `SD_VARIANTS` (or those in `only`) and return
`label => (field => digest)`, the fields of `state_fields` plus `t`.
`inputs` maps a config name to extra overrides (the made dynamic driver of
`lindenberg_era5`).
"""
function simulation_digests(; inputs = Dict{String,Vector{String}}(), only = nothing)
    out = Dict{String,Dict{String,String}}()
    for (label, (name, ovr)) in SD_VARIANTS
        only === nothing || label in only || continue
        cfg = load_run_config(joinpath(SD_CONFIGS, name * ".toml");
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

# the clock of a run object: a `SimulationRun` (`sim.t`) before ONE-a, a `Model` after
sd_clock(obj) = hasproperty(obj, :sim) ? obj.sim.t : Aeolus.Components.clock(obj)

"The dynamic driver of `lindenberg_era5`, made from the committed ERA5 sample (as test/config_roundtrip.jl does)."
function sd_made_inputs()
    dir = mktempdir(); out = joinpath(dir, "lindenberg_era5_dynamic.nc")
    cfg = joinpath(SD_CONFIGS, "lindenberg_era5.toml")
    era5 = [normpath(joinpath(SD_CONFIGS, "..", "data", "era5_lindenberg_20190629_$x.nc")) for x in ("pl", "sfc")]
    rc = redirect_stdout(devnull) do
        Aeolus.main(["preprocess", "dynamic", "--from-era5", era5..., "--config", cfg, "--out", out])
    end
    rc == 0 || error("could not preprocess the ERA5 dynamic driver of $cfg")
    return Dict("lindenberg_era5" => ["initial.dynamic=\"$out\""])
end
