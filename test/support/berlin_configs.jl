# The Berlin case's run configurations (examples/urban/berlin_*.toml) and
# input data (examples/urban/data/): the shipped configurations of the
# urban site (Aeolus.Urban, EX-a #438, ONE-e #454) that the urban tests
# load. Functions, so that every test file can include this.

using Aeolus

"The directory of the Berlin configs, examples/urban."
berlin_configs_dir() = joinpath(pkgdir(Aeolus), "examples", "urban")
"The directory of the Berlin input data, examples/urban/data."
berlin_data_dir() = joinpath(pkgdir(Aeolus), "examples", "urban", "data")
"The shipped Berlin config `berlin_<s>.toml` (`s` = summer, winter, smoke, mitte_sample)."
berlin_toml(s) = joinpath(berlin_configs_dir(), "berlin_$s.toml")

"""
    site_domain(path) -> (index, id) or nothing

The `[[domain]]` block of the urban site in the config file `path` (the
Berlin configs nest it in a parent domain, NEST-p3 #483), or `nothing`
for a one-domain config.
"""
function site_domain(path)
    t = Aeolus.Config.TOML.parsefile(path)
    blocks = get(t, "domain", nothing)
    blocks === nothing && return nothing
    i = findfirst(b -> get(get(b, "urban", Dict()), "enabled", false) === true, blocks)
    return (i, blocks[i]["id"])
end

"""
    site_key(path, key) -> String

The override key of the site's option `key` (a dotted key of a one-domain
config: `urban.offset`, `static_driver`, `time.dt`) in the config `path`:
`domain.<site id>.<key>` for a per-domain key of a nested config.
"""
function site_key(path, key::AbstractString)
    sd = site_domain(path)
    sec = first(split(key, '.'))
    (sd === nothing || sec in ("run", "time", "output", "checkpoint", "checks", "logging", "fftw")) && return key
    return "domain.$(sd[2]).$key"
end

"The urban site's view of a loaded config: the config itself, or its site domain's (`Config.domain_view`)."
function site_view(cfg)
    ds = Aeolus.Config.config_domains(cfg)
    length(ds) <= 1 && return cfg
    i = findfirst(k -> Aeolus.Urban.urban_enabled(Aeolus.Config.domain_view(cfg, k)), eachindex(ds))
    return Aeolus.Config.domain_view(cfg, i)
end

"The model of a loaded config, as `aeolus run` builds it (a nested config: the parent and the site)."
build_config_model(cfg) = Aeolus.Config.build(Aeolus.SimulationDriver.SimulationDriverType(), cfg).model

"The urban site of a run's model (`run_config(…).obj`, or a `Model`): the first component of the domain that is one (ONE-k, #474)."
function urban_site(m::Aeolus.Components.Model)
    i = findfirst(d -> any(c -> c isa Aeolus.Urban.UrbanSite, d.components), collect(m.domains))
    return Aeolus.Urban.urban_case(m.domains[i])
end
urban_site(r) = urban_site(r.model)

"""
    run_site!(case) -> case
    run_model!(m) -> m

Run a built site `case` (a site alone) or a model `m` (a site nested in its
parent too) for `nsteps · dt` of model time, as the run loop does
(`Components.run_to!`).
"""
run_site!(case) = (Aeolus.Components.run_to!(case.model, case.sim.t + case.cfg.nsteps * case.cfg.dt); case)
function run_model!(m)
    case = urban_site(m)
    Aeolus.Components.run_to!(m, case.sim.t + case.cfg.nsteps * case.cfg.dt)
    return m
end

"""
    site_config(path; overrides = String[], kw...) -> NamedTuple

The urban site's options (`Urban.site_options`, ONE-k #474) of the
config `path` with `overrides` and the fields `kw` set through their keys
of the model (`Urban.SITE_FIELDS`; the driver paths as the domain's keys,
`backend` as `urban.serial`; in a nested config the site's
`domain.<id>.…`): every value goes through the config's validation, as
`--set` does.
"""
function site_config(path; overrides = String[], kw...)
    U = Aeolus.Urban
    return U.site_options(site_view(load_run_config(path; overrides = site_overrides(path; overrides, kw...))))
end

"The overrides of [`site_config`](@ref): `overrides`, then the fields `kw` by their keys."
function site_overrides(path; overrides = String[], kw...)
    U = Aeolus.Urban
    ovr = String[overrides...]
    for (f, v) in kw
        if f === :backend
            push!(ovr, site_key(path, "urban.serial") * "=" * U.toml_literal(v.serial))
        else
            key = only(U.SITE_FIELDS[f])
            push!(ovr, site_key(path, replace(key, r"^domain\." => "")) * "=" * U.toml_literal(v))
        end
    end
    return ovr
end
