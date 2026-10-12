# The Berlin case's run configurations (examples/urban/berlin_*.toml) and
# input data (examples/urban/data/): the shipped configurations of the
# urban site (Aeolus.UrbanCase, EX-a #438, ONE-e #454) that the urban tests
# load. Functions, so that every test file can include this.

using Aeolus

"The directory of the Berlin configs, examples/urban."
berlin_configs_dir() = joinpath(pkgdir(Aeolus), "examples", "urban")
"The directory of the Berlin input data, examples/urban/data."
berlin_data_dir() = joinpath(pkgdir(Aeolus), "examples", "urban", "data")
"The shipped Berlin config `berlin_<s>.toml` (`s` = summer, winter, smoke, mitte_sample)."
berlin_toml(s) = joinpath(berlin_configs_dir(), "berlin_$s.toml")

"The urban site of a run's model (`run_config(…).obj`, or a `Model`): the first component of its one domain (ONE-k, #474)."
urban_site(m::Aeolus.Components.Model) = Aeolus.UrbanCase.urban_case(only(m.domains))
urban_site(r) = urban_site(r.model)

"Run a built site `case` for `case.cfg.nsteps · case.cfg.dt` of model time, as the run loop does (`Components.run_to!`)."
run_site!(case) = (Aeolus.Components.run_to!(case.model, case.sim.t + case.cfg.nsteps * case.cfg.dt); case)
