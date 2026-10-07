# The thread-invariance harness (KA-a, #294; docs/src/design/ka_port.md).
#
# "Bit-for-bit" in the KernelAbstractions port means: the code gives
# identical bits at N threads and at 1 thread (maintainer decision,
# 2026-10-06). This file checks it at two levels.
#
#  1. In process (always runs, fast): the primitives and a short Boussinesq
#     run with `CPUBackend(serial = true)` against the threaded default.
#     Under `julia -t 1` both run serially, so this level is meaningful
#     when the suite itself runs threaded (`test/run.jl --threads 4`).
#
#  2. Subprocesses (opt-in, AEOLUS_THREAD_TESTS=1; maintainer decision):
#     test/support/thread_invariance_worker.jl runs at `julia -t 1`, `-t 4`
#     and `-t 24` (AEOLUS_THREAD_COUNTS="1,4,24") and hashes every
#     prognostic field, the ledgers, the solver iteration counts and the dt
#     sequence of: the primitives, a 64³ DNS step, the Berlin smoke matrix
#     (12 steps per run), one flagship step and short nested runs (KA-f).
#     Every hash must agree with
#     the 1-thread one; the differing keys are listed otherwise.
#     AEOLUS_THREAD_SECTIONS selects sections (default
#     "primitives,dns,reductions,smoke,flagship,nested,fft,fv,hooks"; reductions: KA-b #296;
#     fv: the finite-volume kernels and a stretched LES, KA-d #299;
#     hooks: the dynamics hooks and immersed-wall operators, KA-k #307;
#     nested: short nested runs,
#     KA-f #297; fft: every spectral pressure solver with FFTW on the
#     process's threads, KA-h #298). Every port PR runs and quotes it:
#
#       AEOLUS_THREAD_TESTS=1 julia --project=. test/run.jl --only thread_invariance.jl
#
# Later port PRs extend the worker (more sections), not this driver.

using Test
using Random: Xoshiro, randn
using Aeolus
using Aeolus.Fields: set!, interior_view
using Aeolus.Grids: interior

const TI_WORKER = joinpath(@__DIR__, "support", "thread_invariance_worker.jl")

ti_bits(A) = reinterpret(UInt64, vec(collect(Float64, A)))

function ti_short_run(b)
    with_backend(b) do
        mesh = RegularGrid(; size = (32, 32, 32), extent = (1.0, 1.0, 1.0), halo = (3, 3, 3))
        sim = BoussinesqSimulation(mesh; ν = 1e-3, θ₀ = 300.0)
        rng = Xoshiro(7)
        for f in (sim.u, sim.v, sim.w)
            interior_view(f) .= 0.1 .* randn(rng, size(interior_view(f)))
        end
        set!(sim.θ, (x, y, z) -> 300 + 2z)
        for _ in 1:3
            step!(sim, 1e-3)
        end
        return [ti_bits(f.data) for f in (sim.u, sim.v, sim.w, sim.θ, sim.p)]
    end
end

"`key => hash` from a worker output file."
ti_read(path) = Dict(Pair(split(l, '\t')...) for l in eachline(path) if !isempty(l))

@testset "Thread invariance (KA-a, #294)" begin
    @testset "in process: serial = true vs threaded ($(Threads.nthreads()) threads)" begin
        mesh = RegularGrid(; size = (64, 64, 32), extent = (1.0, 1.0, 0.5), halo = (3, 3, 3))
        I3 = interior(mesh)
        rng = Xoshiro(3)
        A = randn(rng, Aeolus.Grids.totalsize(mesh))
        A[10, 20, 9] = NaN
        B = randn(rng, size(A))
        ser, thr = CPUBackend(serial = true), CPUBackend(threshold = 0)
        for (what, f) in (("interior_sum", b -> interior_sum(b, B, I3)),
                          ("interior_dot", b -> interior_dot(b, B, B, I3)),
                          ("plane_mean!", b -> plane_mean!(zeros(32), b, B, I3)),
                          ("column_partials", b -> column_partials(b, B, I3)),
                          ("interior_max (NaN)", b -> interior_max(b, A, I3)),
                          ("interior_all_finite (NaN)", b -> interior_all_finite(b, A, I3)),
                          ("plane_reduce! (tuple)", b -> (p = (zeros(32), zeros(32));
                              plane_reduce!(p, b, (i, j, k, B) -> (@inbounds (B[i, j, k], abs(B[i, j, k]))), I3, B);
                              vcat(p...))),
                          ("plane_map! (fused)", b -> (C = copy(B); p = plane_map!(zeros(32), b,
                              (k, ir, jr, C) -> (s = 0.0; @inbounds(for j in jr, i in ir
                                  x = 0.5 * C[i, j, k]; C[i, j, k] -= x; s += x end); s), I3, C);
                              vcat(p, vec(C)))),
                          ("range_max (NaN)", b -> range_max(b, (i, j, k, A) -> (@inbounds abs(A[i, j, k])), I3, A)))
            r1, r2 = f(ser), f(thr)
            @test isequal(r1, r2) && (r1 isa Bool || ti_bits(r1) == ti_bits(r2))
        end
        @test ti_short_run(ser) == ti_short_run(CPUBackend())
    end

    if get(ENV, "AEOLUS_THREAD_TESTS", "0") == "1"
        counts = parse.(Int, split(get(ENV, "AEOLUS_THREAD_COUNTS", "1,4,24"), ','))
        sections = get(ENV, "AEOLUS_THREAD_SECTIONS", "primitives,dns,reductions,smoke,flagship,nested,fft,fv,hooks")
        project = dirname(Base.active_project())
        dir = mktempdir()
        hashes = Dict{Int,Dict{String,String}}()
        @testset "subprocesses at -t $(join(counts, '/')): $sections" begin
            for n in counts
                out = joinpath(dir, "t$n.tsv")
                cmd = addenv(`$(Base.julia_cmd()) -t $n --project=$project $TI_WORKER $out $sections default`,
                             "AEOLUS_FFTW_PLANNER" => "estimate")
                t0 = time()
                ok = success(pipeline(cmd; stdout = stdout, stderr = stderr))
                @test ok
                ok && (hashes[n] = ti_read(out))
                @info "thread-invariance worker -t $n: $(ok ? length(hashes[n]) : 0) hashes, $(round(time() - t0; digits = 1)) s"
            end
            ref = get(hashes, first(counts), nothing)
            @test ref !== nothing && !isempty(ref)
            if ref !== nothing
                for n in counts[2:end]
                    haskey(hashes, n) || continue
                    h = hashes[n]
                    @test Set(keys(h)) == Set(keys(ref))
                    diff = sort!([k for k in keys(ref) if get(h, k, "") != ref[k]])
                    isempty(diff) || @error "-t $n differs from -t $(first(counts)) in $(length(diff)) of " *
                        "$(length(ref)) hashes:\n  " * join(first(diff, 40), "\n  ")
                    @test isempty(diff)
                    isempty(diff) && @info "-t $n == -t $(first(counts)): all $(length(ref)) hashes identical"
                end
            end
        end
    else
        @info "thread_invariance.jl: subprocess runs skipped (set AEOLUS_THREAD_TESTS=1 to run them)"
    end
end
