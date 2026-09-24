#!/usr/bin/env julia
"""
Create the roadmap milestones, labels and issues on GitHub from
`.github/roadmap-issues.toml`.

    gh auth login                                   # once, interactively
    julia .github/scripts/seed_github.jl --dry-run        # show what would happen
    julia .github/scripts/seed_github.jl                  # create what is missing
    julia .github/scripts/seed_github.jl --update-bodies  # also refresh existing bodies

Idempotent: milestones, labels and issues that already exist (matched by
title) are skipped, so a partial run can simply be re-run. Nothing is
ever deleted or closed by this script.
"""

using TOML

const ROOT    = abspath(joinpath(@__DIR__, "..", ".."))
const SPEC    = joinpath(ROOT, ".github", "roadmap-issues.toml")
const DRY     = "--dry-run" in ARGS
# Re-push issue bodies for issues that already exist, so a change to the
# acceptance criteria does not leave dozens of stale copies behind.
const UPDATE  = "--update-bodies" in ARGS

# Label colours by kind, so the board is readable at a glance.
const LABEL_COLOURS = Dict(
    "epic"           => "5319e7",
    "tech-debt"      => "d93f0b",
    "validation"     => "0e8a16",
    "infrastructure" => "1d76db",
    "microphysics"   => "c5def5",
    "radiation"      => "fef2c0",
    "land"           => "bfd4f2",
    "urban"          => "d4c5f9",
    "turbulence"     => "f9d0c4",
)
label_colour(l) = get(LABEL_COLOURS, l,
                      startswith(l, "era-")   ? "0052cc" :
                      startswith(l, "phase-") ? "ededed" : "cccccc")

function sh(args::Vector{String})
    if DRY
        println("  [dry-run] gh ", join(args, " "))
        return ""
    end
    out = IOBuffer(); err = IOBuffer()
    try
        run(pipeline(`gh $args`; stdout = out, stderr = err))
        return String(take!(out))
    catch
        msg = String(take!(err))
        return "ERROR: " * msg
    end
end

function preflight()
    success(`which gh`) || error("gh CLI not found on PATH")
    if !success(pipeline(`gh auth status`; stdout = devnull, stderr = devnull))
        error("""
              gh is not authenticated. Run this yourself first — it is
              interactive and needs your credentials:

                  gh auth login
              """)
    end
end

function existing(kind::String)
    kind == "milestone" && return Set(strip.(split(
        sh(String["api", "repos/{owner}/{repo}/milestones?state=all&per_page=100",
                  "--jq", ".[].title"]), "\n"; keepempty = false)))
    kind == "label" && return Set(strip.(split(
        sh(String["label", "list", "--limit", "200", "--json", "name",
                  "--jq", ".[].name"]), "\n"; keepempty = false)))
    return Set(strip.(split(
        sh(String["issue", "list", "--state", "all", "--limit", "500",
                  "--json", "title", "--jq", ".[].title"]), "\n"; keepempty = false)))
end

function main()
    DRY || preflight()
    spec = TOML.parsefile(SPEC)
    milestones = spec["milestone"]
    issues     = spec["issue"]

    have_ms    = DRY ? Set{String}() : existing("milestone")
    have_label = DRY ? Set{String}() : existing("label")
    have_issue = DRY ? Set{String}() : existing("issue")

    println("Milestones ($(length(milestones)))")
    for m in milestones
        t = m["title"]
        if t in have_ms
            println("  = $t (exists)"); continue
        end
        println("  + $t")
        sh(String["api", "repos/{owner}/{repo}/milestones", "-X", "POST",
                  "-f", "title=$t",
                  "-f", "description=$(m["description"])",
                  "-f", "state=$(m["state"])"])
    end

    labels = sort(unique(vcat([i["labels"] for i in issues]...)))
    println("\nLabels ($(length(labels)))")
    for l in labels
        if l in have_label
            println("  = $l (exists)"); continue
        end
        println("  + $l")
        sh(String["label", "create", l, "--color", label_colour(l), "--force"])
    end

    println("\nIssues ($(length(issues)))")
    created = skipped = updated = 0
    numbers = Dict{String,Int}()
    if UPDATE && !DRY
        raw = sh(String["issue", "list", "--state", "all", "--limit", "500",
                        "--json", "number,title",
                        "--jq", ".[] | \"\\(.number)\\t\\(.title)\""])
        for line in split(raw, "\n"; keepempty = false)
            parts = split(line, "\t"; limit = 2)
            length(parts) == 2 && (numbers[strip(parts[2])] = parse(Int, parts[1]))
        end
    end
    for i in issues
        t = i["title"]
        if t in have_issue
            if UPDATE && haskey(numbers, t)
                updated += 1
                println("  ~ $t")
                sh(String["issue", "edit", string(numbers[t]), "--body", i["body"]])
            elseif UPDATE
                println("  ? $t (exists but number not found)")
            else
                skipped += 1
            end
            continue
        end
        created += 1
        println("  + $t")
        args = String["issue", "create", "--title", t, "--body", i["body"],
                      "--milestone", i["milestone"]]
        for l in i["labels"]
            append!(args, String["--label", l])
        end
        r = sh(args)
        startswith(r, "ERROR") && println("      ", first(r, 200))
    end
    println("\n$(created) created, $(updated) updated, $(skipped) already present.")
    DRY && println("\nDry run — nothing was sent to GitHub.")
end

main()
