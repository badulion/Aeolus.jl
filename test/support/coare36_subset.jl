# Shared by test/era5_ocean_mixed_layer.jl, test/era5_sea_spray_coare.jl
# and their namesakes in validation/: the reader of the whitespace-separated
# COARE data tables (test/data/coare36_*.txt, validation/data/
# fairall1996a_table2.txt; see test/data/README_coare.md), the COARE 3.6
# cool-skin / warm-layer subset, and COARE's latitude-dependent gravity.

function read_coare_subset(path)
    lines = filter(l -> !startswith(l, "#") && !isempty(strip(l)), readlines(path))
    hdr = Symbol.(split(lines[1]))
    rows = [parse.(Float64, split(l)) for l in lines[2:end]]
    return Dict(h => [r[i] for r in rows] for (i, h) in enumerate(hdr))
end

"The COARE 3.6 cool-skin / warm-layer ship test subset as a Dict of columns."
coare36_warm_cool_subset() =
    read_coare_subset(joinpath(@__DIR__, "..", "data", "coare36_warm_cool_subset.txt"))

# COARE's latitude-dependent gravity (coare36vn_zrf_et.grv).
coare_grav(lat) = (x = sind(lat);
                   9.7803267715 * (1 + 0.0052790414x^2 + 2.32718e-5x^4 + 1.262e-7x^6 + 7e-10x^8))
