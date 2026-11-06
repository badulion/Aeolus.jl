# Naming conventions of the repository (maintainer decision, 2026-11-06).
#
# Work is grouped by content (milestones, test groups, API pages), never
# into numbered "eras": the grouping was arbitrary, and ERA is the name of
# the ECMWF reanalysis (ERA5), which Aeolus reads. This test fails on the
# word "era"/"eras" (any capitalisation except ERA), on "era-iv"-style
# labels and on eraN_-prefixed file names. ERA5 names (era5_lindenberg_*,
# data/era5/, examples/lindenberg_era5) are allowed.

@testset "Naming: no eras" begin
    root = normpath(joinpath(@__DIR__, ".."))
    dirs = ("src", "test", "validation", "docs", "examples", ".github", "benchmarks")
    tops = ("README.md", "CLAUDE.md", "Project.toml")
    exts = (".jl", ".md", ".toml", ".py", ".yml", ".yaml", ".txt")
    word  = r"\b[Ee]ras?\b"                         # ERA (upper case) is the reanalysis
    label = r"\bera-(i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii)\b"
    pref  = r"\bera(?!5_)[0-9]+_"                   # era5_ is ERA5 data
    allowed(line) = occursin("reserved for the ERA5", line)   # the rule in CLAUDE.md
    files = String[joinpath(root, t) for t in tops if isfile(joinpath(root, t))]
    for d in dirs, (dir, _, fs) in walkdir(joinpath(root, d))
        occursin(joinpath(root, "docs", "build"), dir) && continue
        for f in fs
            any(endswith(f, e) for e in exts) && push!(files, joinpath(dir, f))
        end
    end
    self = normpath(@__FILE__)
    hits = String[]
    for f in files
        normpath(f) == self && continue
        for (n, line) in enumerate(eachline(f))
            allowed(line) && continue
            if occursin(word, line) || occursin(label, line) || occursin(pref, line)
                push!(hits, "$(relpath(f, root)):$n: $(strip(line))")
            end
        end
    end
    badnames = [relpath(joinpath(dir, f), root) for d in ("test", "validation", "docs")
                for (dir, _, fs) in walkdir(joinpath(root, d)) for f in fs
                if occursin(r"^era[0-9]+(_|\.md$)", f) && !startswith(f, "era5_")]
    isempty(hits) || @info "\"era\" found (name the content instead)" hits = first(hits, 20) count = length(hits)
    @test isempty(hits)
    @test isempty(badnames)
end
