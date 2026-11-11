# E4 (0ff95557, #557): the moist N² (θ_v) with @propagate_inbounds. No old
# copy: one process per build, alternating bench/all and bench/all-revert-E4
# (node17.sh). On the Berlin 100 m state after 3 steps: the N² loop alone
# (_stratification!) and the whole TKE update, blocked pass and separate
# loops (e reset before every call; copy_e times the reset alone).
#
# Without `ref` the process writes θ, qv and its N² (`<out>.E4.jls`); with
# `ref=<a bench/all TOML>` it takes θ and qv from that dump (the very same
# input across processes) and compares its N² with it.
#
#   julia -t N --project=<build> moist_n2_bench.jl <out.toml> [ref=<toml>] [key=value ...]
@isdefined(FIXES) || include(joinpath(@__DIR__, "common.jl"))

FIXES["E4"] = function ()
    s = berlin(); c, sim = s.c, s.sim
    Δt = 2.0
    q = PS._qv(c, sim)                         # the closure's humidity (Berlin: sim.qv, θ_v)
    q === nothing && error("E4: the closure has no humidity, the moist N² does not run")
    out = Dict{String,Any}("cloud" => c.cloud !== nothing)
    ref = read_ref("E4")
    if ref !== nothing                         # the reference's input
        copyto!(sim.θ.data, ref.θ); copyto!(q.data, ref.qv)
    end
    PS._stratification!(c, sim); n2 = copy(c.N2.data)
    if ref === nothing
        write_dump("E4", (θ = sim.θ.data, qv = q.data, N2 = n2))
    else
        # agreement: the same expressions, only the bounds checks differ (0 expected)
        out["rel_dN2"] = relmax(n2, ref.N2); out["tol"] = 1e-12; out["agree"] = out["rel_dN2"] <= 1e-12
        println("  rel max|ΔN²| against ref ", out["rel_dN2"])
    end
    e0 = copy(c.e.data); st0 = copy(c.stats)
    run(set) = (SIM.set_loop_fusion!(sim, set); copyto!(c.e.data, e0); advance_sgs!(c, sim, Δt))
    out["timings"] = measure(("n2_loop" => () -> PS._stratification!(c, sim),
                              "pass" => () -> run(:if_possible), "separate" => () -> run(:never),
                              "copy_e" => () -> copyto!(c.e.data, e0)))
    SIM.set_loop_fusion!(sim, :if_possible); copyto!(c.e.data, e0); copyto!(c.stats, st0)
    return out
end

standalone("E4", @__FILE__)
