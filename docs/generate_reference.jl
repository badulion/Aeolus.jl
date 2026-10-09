# The generated reference pages (CFG-i, #364). `docs/make.jl` includes this
# file and calls `generate_reference()` before `makedocs`; the round-trip
# coverage suite (test/config_roundtrip.jl) calls it too and checks every
# shipped config against the pages it writes. The pages are generated from
# the schemas, the one source of truth, and are not committed
# (docs/src/reference/ is gitignored).
#
#   docs/src/reference/options/<driver>.md   Aeolus.Config.option_reference, one per driver
#   docs/src/reference/input_static.md       InputDrivers.format_reference, static forms
#   docs/src/reference/input_dynamic.md      InputDrivers.format_reference, dynamic form

using Aeolus
import Aeolus.Config as AeolusConfig
import Aeolus.InputDrivers as AeolusInput

"Every driver the option reference documents: the built-in ones and every example driver of `DRIVER_FILES`."
reference_drivers() = sort!(unique!(vcat("common", "simulation", "urban", "urban_parent",
                                         collect(keys(AeolusConfig.DRIVER_FILES)))))

const _REF_INPUT = normpath(joinpath(@__DIR__, "..", "examples", "urban", "configs", "input"))

"""
    generate_reference(dir = joinpath(@__DIR__, "src", "reference")) -> Vector{String}

Write the option reference of every driver to `dir/options/<driver>.md` and
the input-format specifications to `dir/input_static.md` and
`dir/input_dynamic.md`; return the written paths. Each example driver
schema shows the header of a shipped driver file (`ncdump -h`).
"""
function generate_reference(dir::AbstractString = joinpath(@__DIR__, "src", "reference"))
    out = String[]
    odir = mkpath(joinpath(dir, "options"))
    for d in reference_drivers()
        p = joinpath(odir, d * ".md")
        open(p, "w") do io
            println(io, "# Options: driver `", d, "`\n")
            println(io, "Every key a config file of this driver may hold is a row here. The guide is ",
                    "[Run configuration](../../design/run_configuration.md).\n")
            print(io, AeolusConfig.option_reference(d; format = :markdown))
        end
        push!(out, p)
    end
    pages = ("input_static.md" => ("Static input driver",
                 "The static driver holds what does not change during a run: terrain, land use, buildings, " *
                 "trees, surfaces (CFG-f, #361). `aeolus preprocess static` writes it.",
                 [(AeolusInput.STATIC_SCHEMA, "berlin_smoke_static.nc"),
                  (AeolusInput.STATIC_TILES_SCHEMA, "berlin_smoke_parent_static.nc")]),
             "input_dynamic.md" => ("Dynamic input driver",
                 "The dynamic driver holds what changes in time: initial profiles, large-scale forcing, " *
                 "surface series and boundary data (CFG-g, #362). `aeolus preprocess dynamic` writes it.",
                 [(AeolusInput.DYNAMIC_SCHEMA, "berlin_20190629_dynamic.nc")]))
    for (name, (title, intro, schemas)) in pages
        p = joinpath(dir, name)
        open(p, "w") do io
            println(io, "# ", title, "\n")
            println(io, intro, " Generated from its `DriverSchema` by `InputDrivers.format_reference` ",
                    "when the docs are built.\n")
            for (s, ex) in schemas
                ep = joinpath(_REF_INPUT, ex)
                print(io, AeolusInput.format_reference(s; example = isfile(ep) ? ep : nothing), "\n")
            end
        end
        push!(out, p)
    end
    return out
end
