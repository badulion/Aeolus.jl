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

"The urban site of a run's model (`run_config(…).obj`, or a `Model`): the first component of its one domain (ONE-k, #474)."
urban_site(m::Aeolus.Components.Model) = Aeolus.Urban.urban_case(only(m.domains))
urban_site(r) = urban_site(r.model)

"Run a built site `case` for `case.cfg.nsteps · case.cfg.dt` of model time, as the run loop does (`Components.run_to!`)."
run_site!(case) = (Aeolus.Components.run_to!(case.model, case.sim.t + case.cfg.nsteps * case.cfg.dt); case)

"""
    site_config(path; overrides = String[], kw...) -> NamedTuple

The urban site's options (`Urban.site_options`, ONE-k #474) of the
config `path` with `overrides` and the fields `kw` set through their keys
of the model (`Urban.SITE_FIELDS`; the driver paths as the top-level
keys of a one-domain config, `backend` as `urban.serial`): every value
goes through the config's validation, as `--set` does.
"""
function site_config(path; overrides = String[], kw...)
    U = Aeolus.Urban
    ovr = String[overrides...]
    for (f, v) in kw
        if f === :backend
            push!(ovr, "urban.serial=" * U.toml_literal(v.serial))
        else
            key = only(U.SITE_FIELDS[f])
            push!(ovr, replace(key, r"^domain\." => "") * "=" * U.toml_literal(v))
        end
    end
    return U.site_options(load_run_config(path; overrides = ovr))
end
