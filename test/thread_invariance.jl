# The thread-invariance harness (KA-a, #294; docs/src/design/ka_port.md).
#
# The code gives the same results at N threads and at 1 thread, up to
# round-off: within ROUNDOFF_RTOL of each field's size
# (test/support/agreement.jl). Bit-identity is not required (maintainer
# decision 2026-10-16, epic #493, which replaced the rule of
# 2026-10-06); what this file catches is a race between threads, which
# moves a result by far more than round-off. It checks at two levels.
#
#  1. In process (always runs, fast): the primitives and a short Boussinesq
#     run with `CPUBackend(serial = true)` against the threaded default.
#     Under `julia -t 1` both run serially, so this level is meaningful
#     when the suite itself runs threaded (`test/run.jl --threads 4`).
#
#  2. Subprocesses (opt-in, AEOLUS_THREAD_TESTS=1; maintainer decision):
#     test/support/thread_invariance_worker.jl runs at `julia -t 1`, `-t 4`
#     and `-t 24` (AEOLUS_THREAD_COUNTS="1,4,24") and
#     records every prognostic field, the ledgers, the solver iteration counts and the dt
#     sequence of: the primitives, a 64³ DNS step, the Berlin smoke matrix
#     (12 steps per run), one flagship step and short nested runs (KA-f).
#     Every result must agree with the 1-thread one (fields within
#     ROUNDOFF_RTOL, integer data exactly); the differing keys are listed
#     otherwise.
#     AEOLUS_THREAD_SECTIONS selects sections (default
#     "primitives,dns,reductions,smoke,flagship,nested,fft,fv,pcg,multigrid,steps,radiation,hooks,columns,canopy,advection,sgs";
#     reductions: the KA-b #296 reduction call sites; nested: short nested
#     runs, KA-f #297; fft: every spectral pressure solver with FFTW on the
#     process's threads, KA-h #298; fv: the finite-volume kernels and a
#     stretched LES, KA-d #299; pcg: the four CG solvers, KA-g #300;
#     multigrid: the MG solvers incl. stretched and cut-cell, KA-i #303;
#     steps: every step core, KA-j #306; hooks: the dynamics hooks and
#     immersed-wall operators, KA-k #307; radiation: radiosity, optics,
#     two-stream, cloud shadow, Monte Carlo and superdroplets, KA-n #305;
#     columns: the column-physics kernels, KA-m #301; canopy: the tiled
#     surface's canopy drag, #538; advection: scalar
#     transport, limiter, momentum advection, KA-e #302; sgs: every SGS
#     closure's coefficients, commits and short LES runs, KA-l #304).
#     Every port PR runs and quotes it:
#
#       AEOLUS_THREAD_TESTS=1 julia --project=. test/run.jl --only thread_invariance.jl
#
# Later port PRs extend the worker (more sections), not this driver.

using Test
using Random: Xoshiro, randn
using Aeolus
using Aeolus.Fields: set!, interior_view
using Aeolus.Grids: interior
using Serialization
include(joinpath(@__DIR__, "support", "agreement.jl"))

const TI_WORKER = joinpath(@__DIR__, "support", "thread_invariance_worker.jl")

function ti_short_run(b)
    with_backend(b) do
        mesh = RegularGrid(; size = (16, 16, 16), extent = (1.0, 1.0, 1.0), halo = (3, 3, 3))
        sim = BoussinesqSimulation(mesh; ν = 1e-3, θ₀ = 300.0)
        rng = Xoshiro(7)
        for f in (sim.u, sim.v, sim.w)
            interior_view(f) .= 0.1 .* randn(rng, size(interior_view(f)))
        end
        set!(sim.θ, (x, y, z) -> 300 + 2z)
        for _ in 1:3
            step!(sim, 1e-3)
        end
        return [copy(f.data) for f in (sim.u, sim.v, sim.w, sim.θ, sim.p)]
    end
end

"`key => summary` from a worker output file."
ti_read(path) = Dict{String,Any}(deserialize(path))

@testset "Thread invariance (KA-a, #294)" begin
    @testset "in process: serial = true vs threaded ($(Threads.nthreads()) threads)" begin
        mesh = RegularGrid(; size = (24, 16, 8), extent = (1.0, 1.0, 0.5), halo = (3, 3, 3))
        I3 = interior(mesh)
        rng = Xoshiro(3)
        A = randn(rng, Aeolus.Grids.totalsize(mesh))
        A[10, 12, 7] = NaN
        B = randn(rng, size(A))
        ser, thr = CPUBackend(serial = true), CPUBackend(threshold = 0)
        bad = String[]
        for (what, f) in (("interior_sum", b -> interior_sum(b, B, I3)),
                          ("column_partials", b -> column_partials(b, B, I3)),
                          ("interior_max (NaN)", b -> interior_max(b, A, I3)),
                          ("interior_all_finite (NaN)", b -> interior_all_finite(b, A, I3)),
                          ("plane_reduce! (tuple)", b -> (p = (zeros(8), zeros(8));
                              plane_reduce!(p, b, (i, j, k, B) -> (@inbounds (B[i, j, k], abs(B[i, j, k]))), I3, B);
                              vcat(p...))),
                          ("plane_map! (fused)", b -> (C = copy(B); p = plane_map!(zeros(8), b,
                              (k, ir, jr, C) -> (s = 0.0; @inbounds(for j in jr, i in ir
                                  x = 0.5 * C[i, j, k]; C[i, j, k] -= x; s += x end); s), I3, C);
                              vcat(p, vec(C)))),
                          ("range_max (NaN)", b -> range_max(b, (i, j, k, A) -> (@inbounds abs(A[i, j, k])), I3, A)))
            r1, r2 = f(ser), f(thr)
            (r1 isa Bool ? r1 == r2 : values_agree(r1, r2)) || push!(bad, what)
        end
        @test isempty(bad)
        @test all(agree(a, b) for (a, b) in zip(ti_short_run(ser), ti_short_run(CPUBackend())))
    end

    if get(ENV, "AEOLUS_THREAD_TESTS", "0") == "1"
        counts = parse.(Int, split(get(ENV, "AEOLUS_THREAD_COUNTS", "1,4,24"), ','))
        sections = get(ENV, "AEOLUS_THREAD_SECTIONS", "primitives,dns,reductions,smoke,flagship,nested,fft,fv,pcg,multigrid,steps,radiation,hooks,columns,canopy,advection,sgs")
        project = dirname(Base.active_project())
        dir = mktempdir()
        results = Dict{Int,Dict{String,Any}}()
        @testset "subprocesses at -t $(join(counts, '/')): $sections" begin
            for n in counts
                out = joinpath(dir, "t$n.jls")
                cmd = addenv(`$(Base.julia_cmd()) -t $n --project=$project $TI_WORKER $out $sections default`,
                             "AEOLUS_FFTW_PLANNER" => "estimate")
                t0 = time()
                ok = success(pipeline(cmd; stdout = stdout, stderr = stderr))
                @test ok
                ok && (results[n] = ti_read(out))
                @info "thread-invariance worker -t $n: $(ok ? length(results[n]) : 0) results, $(round(time() - t0; digits = 1)) s"
            end
            ref = get(results, first(counts), nothing)
            @test ref !== nothing && !isempty(ref)
            if ref !== nothing
                for n in counts[2:end]
                    haskey(results, n) || continue
                    h = results[n]
                    @test Set(keys(h)) == Set(keys(ref))
                    diff = sort!([k for k in keys(ref) if !(haskey(h, k) && summaries_agree(h[k], ref[k]))])
                    isempty(diff) || @error "-t $n differs from -t $(first(counts)) in $(length(diff)) of " *
                        "$(length(ref)) results:\n  " * join(first(diff, 40), "\n  ")
                    @test isempty(diff)
                    isempty(diff) && @info "-t $n agrees with -t $(first(counts)): all $(length(ref)) results"
                end
            end
        end
    else
        @info "thread_invariance.jl: subprocess runs skipped (set AEOLUS_THREAD_TESTS=1 to run them)"
    end
end
