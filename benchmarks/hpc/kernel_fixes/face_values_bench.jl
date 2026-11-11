# E3 (7bb2cd6a, #557): branch-free face values (new) against the branching
# ones (old, `Branching` of the build's test/support/branching_faces.jl),
# interleaved in one process on the Berlin 100 m TKE closure after 3 steps:
# the flux-limited advection of e (advect_positive!) and the whole TKE
# update, blocked pass (loop_fusion if_possible) and separate loops (never).
# e is reset before every update (copy_e times the reset alone), the
# tendency zeroed before every advection.
#
#   julia -t N --project=<build> face_values_bench.jl <out.toml> [key=value ...]
@isdefined(FIXES) || include(joinpath(@__DIR__, "common.jl"))
include(joinpath(SUPPORT, "branching_faces.jl"))

# the closure `c` with another reconstruction, sharing its fields (a scratch
# of its own for the blocked pass)
function with_recon(c, r)
    vals = [f === :recon ? r : (f === :blocked ? Ref{Any}(nothing) : getfield(c, f)) for f in fieldnames(typeof(c))]
    return PS.TKEClosure(vals...)
end

FIXES["E3"] = function ()
    s = berlin(); c, sim = s.c, s.sim
    co = with_recon(c, Branching(c.recon))
    Δt = 2.0
    e0 = copy(c.e.data); st0 = copy(c.stats)
    ebcs = Aeolus.passive_scalar_bcs(sim.θ_bcs)
    adv(cc) = (fill!(cc.tend.data, 0);         # advect_positive! adds into tend
               Aeolus.AdvectionSchemes.advect_positive!(cc.tend, sim.u, sim.v, sim.w, cc.e, Δt;
                                                        scheme = cc.recon, workspace = cc.ws, bcs = ebcs))
    run(cc, set) = (SIM.set_loop_fusion!(sim, set); copyto!(cc.e.data, e0); advance_sgs!(cc, sim, Δt))
    # agreement: the same face expressions selected instead of branched (0 expected)
    adv(c); tn = copy(c.tend.data); adv(co); to = copy(co.tend.data)
    run(c, :if_possible); en = copy(c.e.data); run(co, :if_possible); eo = copy(co.e.data)
    run(c, :never); sn = copy(c.e.data); run(co, :never); so = copy(co.e.data)
    dt_, dp, ds = relmax(tn, to), relmax(en, eo), relmax(sn, so)
    println("  recon ", c.recon, "  rel max|Δtend| ", dt_, "  rel max|Δe| pass ", dp, " separate ", ds)
    t = measure(("advect_old" => () -> adv(co), "advect_new" => () -> adv(c),
                 "pass_old" => () -> run(co, :if_possible), "pass_new" => () -> run(c, :if_possible),
                 "separate_old" => () -> run(co, :never), "separate_new" => () -> run(c, :never),
                 "copy_e" => () -> copyto!(c.e.data, e0)))
    SIM.set_loop_fusion!(sim, :if_possible); copyto!(c.e.data, e0); copyto!(c.stats, st0)
    tol = 1e-12
    return Dict{String,Any}("timings" => t, "recon" => string(c.recon), "grid" => collect(size(c.e.data)),
        "rel_dtend" => dt_, "rel_de_pass" => dp, "rel_de_separate" => ds, "tol" => tol,
        "agree" => max(dt_, dp, ds) <= tol)
end

standalone("E3", @__FILE__)
