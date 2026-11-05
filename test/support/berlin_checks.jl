# The Berlin site and parent helpers shared by the urban run-type checks:
# test/urban_checks.jl (the smoke matrix, the parent smoke run, the parent
# class fractions) and validation/urban_checks.jl (the flagship run, the
# idealised parent validations, the 24 h parent surface against the child).
# Functions and constants only; the includer loads Aeolus. Include with
#   isdefined(@__MODULE__, :berlin_default) || include(".../berlin_checks.jl")

using Aeolus
include(joinpath(@__DIR__, "berlin_configs.jl"))

"""
    berlin_default(; kw...) -> NamedTuple

The Berlin summer site's options (berlin_summer.toml's site block, the
fields `kw` set through their keys) without its nest (the `:nesting`
component dropped from the selection): the site alone, for the offline
surface checks, which need its geometry and its class factories only.
"""
function berlin_default(; kw...)
    c = site_config(berlin_toml("summer"); kw...)
    delete!(c.components, :nesting)
    return c
end

"The Berlin land-use table of the nest parent's tiles (examples/urban/data)."
const BERLIN_LANDUSE = joinpath(berlin_data_dir(), "berlin_landuse_6km.csv")

"The energy (or water) budget of a domain: its stage budget (optionally of a layer)."
domain_budget(d, q; kw...) = Aeolus.BudgetSystem.stage_budget(d.budgets.budgets.measured[q]; kw...)
