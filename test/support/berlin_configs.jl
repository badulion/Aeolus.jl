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
