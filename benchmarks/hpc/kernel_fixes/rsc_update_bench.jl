# E6 (ea8c3fcb, #560): the Reynolds-stress source update without a heap
# allocation per cell. No old copy: one process per build, alternating
# bench/all and bench/all-revert-E6 (node17.sh). Berlin 100 m runs the TKE
# closure, not this one, so the kernel runs on the synthetic case of the
# build's test/support/rsc_blocked_cases.jl at Berlin's size, 200 x 200 x 60
# (WENO-Z5, a solid block; the laptop's case, #560): the source update
# alone (_rsm_update!, what E6 changed) and the whole update, separate loops
# (never) and blocked pass (if_possible); τ reset before every call (copy_τ
# times the reset alone). `alloc_update_bytes`: @allocated of one update.
#
# Without `ref` the process writes τ after one update (`<out>.E6.jls`); with
# `ref=<a bench/all TOML>` it compares with that (seeded input).
#
#   julia -t N --project=<build> rsc_update_bench.jl <out.toml> [ref=<toml>] [key=value ...]
@isdefined(FIXES) || include(joinpath(@__DIR__, "common.jl"))
include(joinpath(SUPPORT, "rsc_blocked_cases.jl"))

FIXES["E6"] = function ()
    g = RegularGrid(; size = (200, 200, 60), extent = (2000.0, 2000.0, 600.0), halo = (3, 3, 3))
    sim, c = rsc_case(g; mask = true, advection = :weno_z5)
    Δt = 0.5
    τ0 = map(f -> copy(f.data), c.τ); st0 = copy(c.stats)
    reset() = foreach((f, a) -> copyto!(f.data, a), c.τ, τ0)
    run(set) = (SIM.set_loop_fusion!(sim, set); reset(); advance_sgs!(c, sim, Δt))
    upd() = (reset(); PS._rsm_update!(c, sim, Δt))
    run(:never)                                # the scratch (N², tend) of one separate update
    upd(); τu = map(f -> copy(f.data), c.τ)
    upd(); al = @allocated PS._rsm_update!(c, sim, Δt)
    out = Dict{String,Any}("case" => "rsc_case 200x200x60 weno_z5, mask", "alloc_update_bytes" => al)
    ref = read_ref("E6")
    if ref === nothing
        write_dump("E6", (τ = τu,))
    else
        # agreement: the same arithmetic, only inlining and bounds checks differ (0 expected)
        out["rel_dtau"] = rsc_diff(τu, ref.τ); out["tol"] = 1e-12; out["agree"] = out["rel_dtau"] <= 1e-12
        println("  rel max|Δτ| against ref ", out["rel_dtau"])
    end
    println("  @allocated _rsm_update! ", al, " B")
    out["timings"] = measure(("update" => upd, "separate" => () -> run(:never),
                              "pass" => () -> run(:if_possible), "copy_tau" => reset))
    copyto!(c.stats, st0)
    return out
end

standalone("E6", @__FILE__)
