# Run folder and provenance (LOG-b, #334; design in epic #293):
# src/io/run_folder.jl.
#
# The acceptance tests of the issue:
#   * the provenance fields are correct: each equals the value read
#     independently (git rev-parse, VERSION, Threads.nthreads(), the
#     Manifest bytes, fftw_policy(), PROGRAM_FILE/ARGS);
#   * a dirty tree is blocked (nothing is written), or recorded with the
#     override (the diff and the untracked files are saved);
#   * a restart appends with the same run ID: the log keeps its old
#     content, run.toml gets a second session.
# Every test uses its own temporary git repository, so the state of the
# checkout running the suite does not matter.

using Test
using Logging
using Aeolus
using Aeolus.IO: fnv1a64, toml_value, aeolus_source_dir
using Aeolus.Operators: fftw_policy

using Aeolus.Fields: interior_view

struct RFCfg; n::Int; name::Symbol; inner::NamedTuple; end

const RF_GIT = `git -c user.name=aeolus-test -c user.email=test@example.org -c commit.gpgsign=false`

"A fresh git repository with one committed file; returns its path."
function rf_repo()
    d = mktempdir()
    run(`git -C $d init -q`)
    write(joinpath(d, "model.jl"), "x = 1\n")
    run(`$RF_GIT -C $d add model.jl`)
    run(`$RF_GIT -C $d commit -q -m initial`)
    return d
end
rf_head(repo) = strip(read(`git -C $repo rev-parse HEAD`, String))
rf_commit!(repo, msg) = (run(`$RF_GIT -C $repo commit -q -a -m $msg`); rf_head(repo))

@testset "run folder and provenance (#334)" begin
if Sys.which("git") === nothing
    @test_skip "git is not installed: the run-folder tests need temporary git repositories"
else

@testset "FNV-1a 64 against the published test vectors" begin
    @test fnv1a64("") == 0xcbf29ce484222325
    @test fnv1a64("a") == 0xaf63dc4c8601ec8c
    @test fnv1a64("foobar") == 0x85944171f73967e8
    @test fnv1a64(Vector{UInt8}("foobar")) == fnv1a64("foobar")
end

@testset "toml_value records any config object" begin
    t = toml_value(RFCfg(3, :abc, (; a = 1.5, b = nothing)))
    @test t["_type"] == "RFCfg" && t["n"] == 3 && t["name"] == "abc"
    @test t["inner"]["a"] == 1.5 && t["inner"]["b"] == "nothing"
    @test toml_value(typemax(UInt64)) == string(typemax(UInt64))
    @test toml_value(UInt32(7)) === Int64(7)
    @test toml_value((1, :x, "s")) == Any[1, "x", "s"]
    @test toml_value(zeros(100)) isa String          # long arrays: a summary
    @test toml_value(Dict(:k => [1, 2])) == Dict("k" => Any[1, 2])
    # TOML round trip
    io = IOBuffer(); Aeolus.IO.TOML.print(io, Dict("config" => t))
    @test Aeolus.IO.TOML.parse(String(take!(io)))["config"]["inner"]["a"] == 1.5
end

@testset "a new run: folder, provenance fields, run log" begin
    repo = rf_repo(); root = mktempdir()
    console = IOBuffer()
    cfg = Dict("N" => 64, "dt" => 0.5, "scheme" => :weno5)
    rf = start_run("case 1"; root = root, repo = repo, console = console,
                   config = cfg, seeds = (; noise = 7, particles = 11))
    @test isdir(rf.path) && dirname(rf.path) == abspath(root)
    @test occursin(r"^\d{4}-\d{2}-\d{2}_case-1_[0-9a-f]{8}$", basename(rf.path))
    @test endswith(basename(rf.path), rf.run_id) && length(rf.run_id) == 8
    @test rf.session == 1 && !rf.finished
    @test !isfile(joinpath(rf.path, "uncommitted-1.diff"))

    r = read_run_toml(rf.path)
    @test r["run_id"] == rf.run_id && r["case"] == "case 1"
    @test r["config"] == Dict("N" => 64, "dt" => 0.5, "scheme" => "weno5")
    @test r["seeds"] == Dict("noise" => 7, "particles" => 11)
    s = only(r["session"])
    @test s["number"] == 1 && s["kind"] == "start" && s["status"] == "running"
    # each field against an independent reading
    @test s["git"]["available"] && s["git"]["commit"] == rf_head(repo)
    @test s["git"]["dirty"] == false && isempty(s["git"]["changes"])
    @test s["git"]["branch"] == strip(read(`git -C $repo rev-parse --abbrev-ref HEAD`, String))
    @test s["julia"]["version"] == string(VERSION)
    @test s["aeolus"]["version"] == string(pkgversion(Aeolus))
    @test s["aeolus"]["source"] == aeolus_source_dir()
    @test s["host"]["julia_threads"] == Threads.nthreads()
    @test s["host"]["cpu_threads"] == Sys.CPU_THREADS
    @test s["host"]["hostname"] == gethostname()
    manifest = Base.project_file_manifest_path(Base.active_project())
    @test s["manifest"]["path"] == manifest
    @test s["manifest"]["fnv1a64"] == string(fnv1a64(read(manifest)); base = 16, pad = 16)
    @test s["packages"]["FFTW"] == string(pkgversion(Aeolus.Operators.FFTW))
    @test s["command"]["program"] == PROGRAM_FILE && s["command"]["args"] == ARGS
    @test s["command"]["cwd"] == pwd()
    p = fftw_policy()
    @test s["fftw"]["planner"] == string(p.planner) && s["fftw"]["threads"] == p.threads
    @test s["fftw"]["wisdom_loaded"] == p.wisdom_loaded
    @test s["fftw"]["bitwise_reproducible"] == p.bitwise_reproducible
    @test s["env"] == Dict(k => v for (k, v) in ENV if startswith(k, "AEOLUS_"))

    # the header: Info to the console and the file, the full record at Debug in the file only
    con = String(take!(console))
    @test occursin("INFO  run[main] run started", con) && occursin("run_id=$(rf.run_id)", con)
    @test occursin("commit=$(first(rf_head(repo), 12)) ", con)
    @test !occursin("provenance", con)
    Logging.with_logger(rf.logger) do
        @debug "a debug event of the run" subsystem = :test domain = :main
    end
    finish_run!(rf)
    @test rf.finished
    log = read(run_log_path(rf), String)
    @test occursin("run started", log) && occursin("DEBUG run[main] provenance", log)
    @test occursin("[config]", log) && occursin("scheme = \"weno5\"", log)
    @test occursin("a debug event of the run", log)
    @test occursin("run finished: completed", log)
    s = only(read_run_toml(rf.path)["session"])
    @test s["status"] == "completed" && haskey(s, "ended") && s["wall_seconds"] >= 0
    @test finish_run!(rf) === rf                      # a second call does nothing
    @test count("run finished", read(run_log_path(rf), String)) == 1
    # the accessors
    @test run_toml_path(rf) == joinpath(rf.path, "run.toml")
    @test diagnostics_path(rf) == joinpath(rf.path, "diagnostics.csv")
    @test isdir(output_dir(rf)) && isdir(checkpoint_dir(rf))
    # an explicit run ID that exists is refused
    @test_throws ArgumentError start_run("case 1"; root = root, repo = repo, console = false,
                                         run_id = rf.run_id)
end

@testset "uncommitted code: blocked, or recorded with the override" begin
    repo = rf_repo(); root = joinpath(mktempdir(), "runs")
    write(joinpath(repo, "model.jl"), "x = 2   # an uncommitted change\n")
    write(joinpath(repo, "new_scheme.jl"), "y = 3\n")
    err = try start_run("dirty"; root = root, repo = repo, console = false); nothing
          catch e; e; end
    @test err isa UncommittedCodeError
    @test any(occursin("model.jl", l) for l in err.changes)
    @test any(startswith(l, "??") && occursin("new_scheme.jl", l) for l in err.changes)
    msg = sprint(showerror, err)
    @test occursin("allow_uncommitted = true", msg) && occursin("new_scheme.jl", msg)
    @test !isdir(root)                                # blocked before anything is written
    # a staged change alone also blocks
    repo2 = rf_repo()
    write(joinpath(repo2, "model.jl"), "x = 5\n"); run(`git -C $repo2 add model.jl`)
    @test_throws UncommittedCodeError start_run("dirty"; root = root, repo = repo2, console = false)

    rf = start_run("dirty"; root = root, repo = repo, console = false, allow_uncommitted = true)
    finish_run!(rf)
    s = only(read_run_toml(rf.path)["session"])
    @test s["git"]["dirty"] && s["allow_uncommitted"]
    @test s["git"]["commit"] == rf_head(repo)
    @test s["git"]["diff_file"] == "uncommitted-1.diff"
    diff = read(joinpath(rf.path, "uncommitted-1.diff"), String)
    @test occursin("-x = 1", diff) && occursin("+x = 2   # an uncommitted change", diff)
    @test readlines(joinpath(rf.path, "untracked-1.txt")) == ["new_scheme.jl"]
    log = read(run_log_path(rf), String)
    @test occursin("WARN  run[main] running with uncommitted code", log)
    @test occursin("commit=$(first(rf_head(repo), 12))-dirty", log)
    # the environment variable is the same override
    rf2 = withenv("AEOLUS_ALLOW_UNCOMMITTED" => "1") do
        start_run("dirty"; root = root, repo = repo, console = false)
    end
    @test only(read_run_toml(rf2.path)["session"])["allow_uncommitted"]
    finish_run!(rf2)
end

@testset "run folders inside the work tree do not make it dirty" begin
    repo = rf_repo(); root = joinpath(repo, "runs")      # not gitignored on purpose
    rf = start_run("inside"; root = root, repo = repo, console = false)
    finish_run!(rf)
    @test !isempty(read(`git -C $repo status --porcelain --untracked-files=all`, String))
    rf2 = start_run("inside"; root = root, repo = repo, console = false, resume = rf.path)
    @test only(unique(s["git"]["dirty"] for s in rf2.record["session"])) == false
    finish_run!(rf2)
    write(joinpath(repo, "other.jl"), "z = 1\n")          # anything else still counts
    @test_throws UncommittedCodeError start_run("inside"; root = root, repo = repo,
                                                console = false, resume = rf.path)
    @test git_state(repo; exclude = [root]).changes == ["?? other.jl"]
end

@testset "a restart appends with the same run ID" begin
    repo = rf_repo(); root = mktempdir()
    rf = start_run("gabls"; root = root, repo = repo, console = false,
                   config = (; N = 64, dt = 1.0), seeds = 42)
    finish_run!(rf)
    log1 = read(run_log_path(rf), String)
    toml1 = read_run_toml(rf.path)
    @test latest_run(root, "gabls") == rf.path
    @test latest_run(root, "other") === nothing

    rf2 = start_run("gabls"; root = root, repo = repo, console = false, resume = :latest,
                    config = (; N = 64, dt = 1.0), seeds = 42)
    @test rf2.path == rf.path && rf2.run_id == rf.run_id && rf2.session == 2
    finish_run!(rf2)
    log2 = read(run_log_path(rf2), String)
    @test startswith(log2, log1)                       # appended, nothing rewritten
    @test occursin("run restarted (session 2)", log2[length(log1)+1:end])
    @test occursin("run_id=$(rf.run_id)", log2[length(log1)+1:end])
    @test !occursin("restart differs", log2)
    r = read_run_toml(rf.path)
    @test r["run_id"] == rf.run_id && r["created"] == toml1["created"]
    @test [s["kind"] for s in r["session"]] == ["start", "restart"]
    @test [s["number"] for s in r["session"]] == [1, 2]
    @test all(s["status"] == "completed" for s in r["session"])
    @test r["session"][1] == toml1["session"][1]       # the first session is kept as it was
    @test only(readdir(root)) == basename(rf.path)     # no second folder

    # a restart on another commit, with another config: recorded and warned about
    write(joinpath(repo, "model.jl"), "x = 10\n"); c2 = rf_commit!(repo, "second")
    rf3 = start_run("gabls"; root = root, repo = repo, console = false, resume = rf.path,
                    config = (; N = 128, dt = 1.0), seeds = 42)
    finish_run!(rf3)
    r = read_run_toml(rf.path)
    s3 = r["session"][3]
    @test s3["git"]["commit"] == c2 != r["session"][1]["git"]["commit"]
    @test s3["config"] == Dict("N" => 128, "dt" => 1.0) && !haskey(s3, "seeds")
    @test r["config"]["N"] == 64                        # the run's config stays the first one
    @test occursin("this restart differs from the first session in: config, git commit",
                   read(run_log_path(rf3), String))
    # resume of something that is not a run folder, and run_id mismatch
    @test_throws ArgumentError start_run("gabls"; root = root, repo = repo, console = false,
                                         resume = mktempdir())
    @test_throws ArgumentError start_run("gabls"; root = root, repo = repo, console = false,
                                         resume = rf.path, run_id = "deadbeef")
    @test_throws ArgumentError start_run("nope"; root = root, repo = repo, console = false,
                                         resume = :latest)
end

@testset "latest_run compares created_unix as a number, not as a string" begin
    # The decimal forms of these Float64s differ in length ("1.759748412123456e9"
    # vs "1.7597490001e9"); a zero-padded string comparison picked the older one.
    repo = rf_repo(); root = mktempdir()
    old = start_run("x"; root = root, repo = repo, console = false, run_id = "aaaaaaaa")
    finish_run!(old)
    new = start_run("x"; root = root, repo = repo, console = false, run_id = "bbbbbbbb")
    finish_run!(new)
    function set_created!(rf, t)
        r = read_run_toml(rf.path); r["created_unix"] = t
        open(io -> Aeolus.IO.TOML.print(io, r; sorted = true), joinpath(rf.path, "run.toml"), "w")
    end
    set_created!(new, 1.7597490001e9)          # 10 minutes later, shorter decimal form
    set_created!(old, 1.759748412123456e9)     # written last: the newer mtime, so no tie-break help
    @test length(string(1.7597490001e9)) < length(string(1.759748412123456e9))
    @test latest_run(root, "x") == new.path
    rf = start_run("x"; root = root, repo = repo, console = false, resume = :latest)
    @test rf.path == new.path && rf.run_id == "bbbbbbbb" && rf.session == 2
    finish_run!(rf)
    @test length(read_run_toml(old.path)["session"]) == 1    # the older run is untouched
    # and the other way round: the longer decimal form is the newer one
    set_created!(old, 1.7597500001234567e9)
    @test latest_run(root, "x") == old.path
end

@testset "a session that never finished is marked aborted on restart" begin
    repo = rf_repo(); root = mktempdir()
    rf = start_run("crash"; root = root, repo = repo, console = false)
    close(rf.logger)                                    # the process "dies": no finish_run!
    @test only(read_run_toml(rf.path)["session"])["status"] == "running"
    rf2 = start_run("crash"; root = root, repo = repo, console = false, resume = rf.path)
    finish_run!(rf2)
    r = read_run_toml(rf.path)
    @test [s["status"] for s in r["session"]] == ["aborted", "completed"]
    @test occursin("previous session(s) of this run did not finish", read(run_log_path(rf2), String))
end

@testset "with_run: logger installed, success and failure recorded" begin
    repo = rf_repo(); root = mktempdir()
    path = Ref("")
    v = with_run("ok"; root = root, repo = repo, console = false) do rf
        path[] = rf.path
        @info "inside the run" subsystem = :test domain = current_domain() value = 3
        41 + 1
    end
    @test v == 42
    @test occursin("inside the run  value=3", read(joinpath(path[], "run.log"), String))
    @test only(read_run_toml(path[])["session"])["status"] == "completed"

    @test_throws ErrorException with_run("bad"; root = root, repo = repo, console = false) do rf
        path[] = rf.path
        error("the solver blew up")
    end
    s = only(read_run_toml(path[])["session"])
    @test s["status"] == "failed" && s["error"] == "the solver blew up"
    log = read(joinpath(path[], "run.log"), String)
    @test occursin("ERROR run[main] run stopped by an error", log)
    @test occursin("the solver blew up", log) && occursin("run finished: failed", log)
end

@testset "code outside git: the run starts, provenance says so" begin
    repo = mktempdir(); root = mktempdir()
    @test git_state(repo).available == false
    rf = start_run("nogit"; root = root, repo = repo, console = false)
    finish_run!(rf)
    s = only(read_run_toml(rf.path)["session"])
    @test s["git"]["available"] == false && s["git"]["commit"] == "" && !s["git"]["dirty"]
    @test occursin("not in a git work tree", read(run_log_path(rf), String))
end

@testset "run_checkpoints: the schedule lives in the folder and resumes there" begin
    repo = rf_repo(); root = mktempdir()
    m = RegularGrid(; size = (8, 8, 8), extent = (800.0, 800.0, 400.0), halo = (3, 3, 3))
    mk() = BoussinesqSimulation(m; θ₀ = 300.0,
                                bcs = Aeolus.BoundaryConditions.FieldBoundaryConditions(:periodic, :periodic, :wall))
    rf = start_run("ck"; root = root, repo = repo, console = false)
    ck = run_checkpoints(rf; every_steps = 2, wisdom = false)
    @test ck.dir == joinpath(rf.path, "checkpoints")
    s = rf.record["session"][end]
    @test s["fftw"]["planner"] == string(fftw_policy().planner)
    sim = mk()
    @test resume!(ck, sim) === nothing
    for n in 1:4
        step!(sim, 0.5)
        maybe_checkpoint!(ck, sim; step = n)
    end
    finish_run!(rf)
    rf2 = start_run("ck"; root = root, repo = repo, console = false, resume = rf.path)
    ck2 = run_checkpoints(rf2; every_steps = 2, wisdom = false)
    sim2 = mk()
    d = Logging.with_logger(rf2.logger) do      # the run's events go to its run.log
        resume!(ck2, sim2)
    end
    @test d !== nothing && ck2.last_step == 4 && sim2.t == sim.t
    @test agree(interior_view(sim2.θ), interior_view(sim.θ))
    finish_run!(rf2)
    @test occursin("checkpoints: resumed", read(run_log_path(rf2), String))
end

end # git available
end # run folder testset
