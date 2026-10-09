# The Berlin case's run configurations (examples/urban/configs/) and input
# data (examples/urban/data/): the shipped configurations of the `urban`
# run driver (Aeolus.UrbanCase, EX-a #438) that the urban run-type tests
# load. Functions, so that every test file can include this.

using Aeolus

"The directory of the Berlin configs, examples/urban/configs."
berlin_configs_dir() = joinpath(pkgdir(Aeolus), "examples", "urban", "configs")
"The directory of the Berlin input data, examples/urban/data."
berlin_data_dir() = joinpath(pkgdir(Aeolus), "examples", "urban", "data")
"The shipped Berlin config `berlin_<s>.toml` (`s` = summer, winter, smoke, mitte_sample)."
berlin_toml(s) = joinpath(berlin_configs_dir(), "berlin_$s.toml")
