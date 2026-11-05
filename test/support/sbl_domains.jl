# The GABLS1 and GABLS3 domains as their configurations build them
# (examples/sbl/gabls1.toml, gabls3.toml), reduced for the suites. Shared
# by test/era6_gabls1.jl, test/era6_gabls3.jl (the protocol, forcing and
# diagnostics on a built or briefly stepped domain) and
# validation/era6_gabls1.jl, validation/era6_gabls3.jl (the reduced runs).

using Aeolus.Config: load_run_config, build
using Aeolus.SimulationDriver: SimulationDriverType
isdefined(@__MODULE__, :SBL_CONFIGS) || include(joinpath(@__DIR__, "sbl_data.jl"))

const GABLS1_TOML = joinpath(pkgdir(Aeolus), "examples", "sbl", "gabls1.toml")

function gabls1_domain(N, stab = "brown_et_al_1994"; extra = String[])
    cfg = load_run_config(GABLS1_TOML; overrides = vcat(["grid.size=[$N,$N,$N]", "sgs.stability=\"$stab\"",
                                                           "checkpoint.every_seconds=inf"], extra))
    return only(build(SimulationDriverType(), cfg).model.domains)
end

_comp(d, T) = only(c for c in d.components if c isa T)

const G3_TOML = joinpath(SBL_CONFIGS, "gabls3.toml")

function gabls3_domain(stab = "brown_et_al_1994"; N = (16, 16, 48), extra = String[])
    cfg = load_run_config(G3_TOML; overrides = vcat(["grid.size=[$(join(N, ","))]", "grid.extent=[400.0,400.0,400.0]",
                                                     "sponge.z_start=300.0", "sgs.stability=\"$stab\"",
                                                     "checkpoint.every_seconds=inf"], extra))
    return only(build(SimulationDriverType(), cfg).model.domains)
end

_g3comp(d, T) = only(c for c in d.components if c isa T)
